"""Baseline multi-seeds TF-IDF + régression logistique pour DEREEV.

Cette baseline utilise exactement le même manifeste de folds fixe que les modèles
Transformers. Les indices train/validation proviennent de
``data/processed/cv_splits`` et restent identiques pour toutes les seeds.

Commande recommandée depuis la racine :
    python3 -m src.tfidf_logistic

Pour relancer en écrasant uniquement les résultats TF-IDF :
    python3 -m src.tfidf_logistic --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from src.config import (
    BENCHMARK_MODELS_DIR,
    CV_SPLITS_DIR,
    CV_SPLIT_SEED,
    TRAINING_SEEDS,
    K_FOLDS,
    LABEL_COL,
    TEST_CSV,
    TEXT_COL,
    TFIDF_DISPLAY_NAME,
    TFIDF_OUTPUT_NAME,
    TRAIN_CV_CSV,
)
from src.cv_protocol import (
    SharedSplit,
    dataset_fingerprint,
    ensure_shared_split,
    parse_seed_list,
)
from src.threshold_metrics import (
    add_pr_auc_to_metrics,
    compute_threshold_metrics,
    evaluate_thresholds,
    plot_precision_recall_curve,
    plot_threshold_curve,
)


DEFAULT_DECISION_THRESHOLD = 0.5
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
OUTPUT_ROOT = os.path.join(BENCHMARK_MODELS_DIR, TFIDF_OUTPUT_NAME)


# ---------------------------------------------------------------------
# AFFICHAGE TERMINAL
# ---------------------------------------------------------------------
def print_banner(title: str, width: int = 84, character: str = "=") -> None:
    print("\n" + character * width)
    print(f" {title}")
    print(character * width)


def print_key_values(rows: Sequence[Tuple[str, object]]) -> None:
    width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        print(f"{str(key):<{width}} : {value}")


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        print("(aucune ligne)")
        return

    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print(
        "| "
        + " | ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
        + " |"
    )
    print(separator)
    for row in rendered:
        print(
            "| "
            + " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )
    print(separator)


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {seconds:02d} s"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"


def sample_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return 0.0
    return float(np.std(array, ddof=1))


def print_metrics_block(title: str, metrics: Dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print_key_values(
        [
            ("Seuil", f"{metrics['threshold']:.3f}"),
            ("F1 positif", f"{metrics['f1']:.4f}"),
            ("PR-AUC", f"{metrics['pr_auc']:.4f}"),
            ("Recall", f"{metrics['recall']:.4f}"),
            ("Précision", f"{metrics['precision']:.4f}"),
            ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ]
    )
    print("\nMatrice de confusion")
    print_table(
        ["Réel / Prédit", "Classe 0", "Classe 1"],
        [
            ["Classe 0", metrics["true_negative"], metrics["false_positive"]],
            ["Classe 1", metrics["false_negative"], metrics["true_positive"]],
        ],
    )


# ---------------------------------------------------------------------
# DONNÉES ET MODÈLE
# ---------------------------------------------------------------------
def read_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [
        column
        for column in (TEXT_COL, LABEL_COL)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {path} : "
            + ", ".join(missing_columns)
        )

    dataframe = dataframe[[TEXT_COL, LABEL_COL]].copy()
    dataframe[TEXT_COL] = dataframe[TEXT_COL].fillna("").astype(str)
    dataframe[LABEL_COL] = dataframe[LABEL_COL].astype(int)

    if not dataframe[LABEL_COL].isin([0, 1]).all():
        raise ValueError(
            f"La colonne {LABEL_COL} doit uniquement contenir 0 et 1."
        )

    empty_texts = dataframe[TEXT_COL].str.strip().eq("")
    if empty_texts.any():
        raise ValueError(
            f"{int(empty_texts.sum())} note(s) vide(s) trouvée(s) dans {path}."
        )

    return dataframe.reset_index(drop=True)


def build_pipeline(
    seed: int,
    max_features: int,
    min_df: int,
    ngram_max: int,
    c_value: float,
    max_iter: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                    dtype=np.float64,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def metrics_with_pr_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict:
    return add_pr_auc_to_metrics(
        metrics=compute_threshold_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=threshold,
        ),
        labels=labels,
        probabilities=probabilities,
    )


def prepare_output_root(overwrite: bool) -> None:
    if os.path.exists(OUTPUT_ROOT):
        if not overwrite:
            raise FileExistsError(
                f"Le dossier existe déjà : {OUTPUT_ROOT}\n"
                "Utilisez --overwrite pour relancer la baseline. "
                "Les manifests partagés ne seront pas supprimés."
            )
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------------------------------------------------------------------
# UNE SEED D'ENTRAÎNEMENT
# ---------------------------------------------------------------------
def run_one_seed(
    shared_split: SharedSplit,
    training_seed: int,
    train_texts: np.ndarray,
    train_labels: np.ndarray,
    test_texts: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
    completed_before: int,
    total_trainings: int,
) -> Dict:
    seed_start = time.perf_counter()
    seed_dir = os.path.join(OUTPUT_ROOT, f"seed_{training_seed}")
    os.makedirs(seed_dir, exist_ok=True)

    print_banner(
        f"SEED D'ENTRAÎNEMENT {training_seed} — {TFIDF_DISPLAY_NAME}",
        character="#",
    )
    print_key_values(
        [
            ("Modèle", TFIDF_DISPLAY_NAME),
            ("Seed d'entraînement", training_seed),
            ("Seed du split fixe", shared_split.split_seed),
            ("Seed du classifieur", training_seed),
            ("Manifeste partagé", shared_split.csv_path),
            ("Dossier de sortie", seed_dir),
        ]
    )

    oof_probabilities = np.full(len(train_labels), np.nan, dtype=float)
    oof_folds = np.full(len(train_labels), -1, dtype=int)
    test_fold_probabilities: List[np.ndarray] = []
    fold_f1_scores: List[float] = []
    fold_pr_auc_scores: List[float] = []
    fold_summaries: List[Dict] = []

    for fold_index, (train_idx, val_idx) in enumerate(
        shared_split.indices(),
        start=1,
    ):
        fold_start = time.perf_counter()
        global_index = completed_before + fold_index
        train_counts = np.bincount(train_labels[train_idx], minlength=2)
        validation_counts = np.bincount(train_labels[val_idx], minlength=2)

        print_banner(
            (
                f"ENTRAÎNEMENT {global_index}/{total_trainings} — SEED "
                f"{training_seed} — FOLD {fold_index}/{shared_split.fold_count}"
            ),
            character="-",
        )
        print_key_values(
            [
                ("Modèle", TFIDF_DISPLAY_NAME),
                ("Seed", training_seed),
                ("Fold", f"{fold_index}/{shared_split.fold_count}"),
                ("Train", f"{len(train_idx)} notes"),
                (
                    "Répartition train",
                    f"classe 0 = {train_counts[0]} | classe 1 = {train_counts[1]}",
                ),
                ("Validation", f"{len(val_idx)} notes"),
                (
                    "Répartition validation",
                    (
                        f"classe 0 = {validation_counts[0]} | "
                        f"classe 1 = {validation_counts[1]}"
                    ),
                ),
                ("N-grams", f"1 à {args.ngram_max}"),
                ("Max features", args.max_features),
                ("min_df", args.min_df),
                ("C", args.c),
            ]
        )

        pipeline = build_pipeline(
            seed=training_seed,
            max_features=args.max_features,
            min_df=args.min_df,
            ngram_max=args.ngram_max,
            c_value=args.c,
            max_iter=args.max_iter,
        )

        pipeline.fit(train_texts[train_idx], train_labels[train_idx])

        validation_probabilities = pipeline.predict_proba(
            train_texts[val_idx]
        )[:, 1]
        test_probabilities = pipeline.predict_proba(test_texts)[:, 1]

        oof_probabilities[val_idx] = validation_probabilities
        oof_folds[val_idx] = fold_index
        test_fold_probabilities.append(test_probabilities)

        fold_metrics = metrics_with_pr_auc(
            labels=train_labels[val_idx],
            probabilities=validation_probabilities,
            threshold=DEFAULT_DECISION_THRESHOLD,
        )
        fold_duration = time.perf_counter() - fold_start

        print_metrics_block(
            f"RÉSULTATS VALIDATION — SEED {training_seed} — FOLD {fold_index}",
            fold_metrics,
        )
        print(f"Durée du fold : {format_duration(fold_duration)}")

        fold_f1_scores.append(fold_metrics["f1"])
        fold_pr_auc_scores.append(fold_metrics["pr_auc"])

        fold_model_path = os.path.join(seed_dir, f"fold_{fold_index}.joblib")
        joblib.dump(pipeline, fold_model_path)

        vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)
        fold_summaries.append(
            {
                "fold": fold_index,
                "training_seed": training_seed,
                "train_size": int(len(train_idx)),
                "validation_size": int(len(val_idx)),
                "vocabulary_size": int(vocabulary_size),
                "duration_seconds": float(fold_duration),
                "validation_metrics": fold_metrics,
                "model_path": fold_model_path,
            }
        )

    if np.isnan(oof_probabilities).any() or np.any(oof_folds == -1):
        raise RuntimeError(
            "Certaines observations n'ont pas reçu de prédiction out-of-fold."
        )

    threshold_results, best_threshold_result = evaluate_thresholds(
        labels=train_labels,
        probabilities=oof_probabilities,
        minimum=THRESHOLD_MIN,
        maximum=THRESHOLD_MAX,
        step=THRESHOLD_STEP,
    )
    optimized_threshold = float(best_threshold_result["threshold"])

    oof_default_metrics = metrics_with_pr_auc(
        labels=train_labels,
        probabilities=oof_probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )
    oof_optimized_metrics = metrics_with_pr_auc(
        labels=train_labels,
        probabilities=oof_probabilities,
        threshold=optimized_threshold,
    )

    threshold_csv_path = os.path.join(seed_dir, "threshold_results.csv")
    threshold_plot_path = os.path.join(seed_dir, "threshold_results.png")
    oof_pr_curve_path = os.path.join(seed_dir, "precision_recall_curve.png")
    oof_predictions_path = os.path.join(
        seed_dir,
        "validation_predictions.csv",
    )
    threshold_json_path = os.path.join(seed_dir, "decision_threshold.json")

    pd.DataFrame(threshold_results).to_csv(threshold_csv_path, index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(train_labels), dtype=int),
            "training_seed": training_seed,
            "fold": oof_folds,
            "label": train_labels,
            "probability_class_1": oof_probabilities,
            "prediction_threshold_0_5": (
                oof_probabilities >= DEFAULT_DECISION_THRESHOLD
            ).astype(int),
            "prediction_optimized_threshold": (
                oof_probabilities >= optimized_threshold
            ).astype(int),
        }
    ).to_csv(oof_predictions_path, index=False)

    plot_threshold_curve(
        results=threshold_results,
        output_path=threshold_plot_path,
    )
    plot_precision_recall_curve(
        labels=train_labels,
        probabilities=oof_probabilities,
        output_path=oof_pr_curve_path,
    )

    with open(threshold_json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_display_name": TFIDF_DISPLAY_NAME,
                "training_seed": training_seed,
                "split_seed": shared_split.split_seed,
                "selection_data": "out_of_fold_validation",
                "objective": "maximum_f1_positive_class",
                "best_threshold": optimized_threshold,
                "default_metrics": oof_default_metrics,
                "best_metrics": oof_optimized_metrics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    test_probability_matrix = np.stack(test_fold_probabilities, axis=0)
    final_test_probabilities = np.mean(test_probability_matrix, axis=0)

    test_default_metrics = metrics_with_pr_auc(
        labels=test_labels,
        probabilities=final_test_probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )
    test_optimized_metrics = metrics_with_pr_auc(
        labels=test_labels,
        probabilities=final_test_probabilities,
        threshold=optimized_threshold,
    )
    test_log_loss = float(
        log_loss(
            test_labels,
            np.column_stack(
                [1.0 - final_test_probabilities, final_test_probabilities]
            ),
            labels=[0, 1],
        )
    )
    fold_disagreement = float(
        np.mean(np.std(test_probability_matrix, axis=0, ddof=0))
    )

    test_predictions_path = os.path.join(seed_dir, "test_predictions.csv")
    test_curve_path = os.path.join(
        seed_dir,
        "test_precision_recall_curve.png",
    )

    pd.DataFrame(
        {
            "label": test_labels,
            "training_seed": training_seed,
            "probability_class_1": final_test_probabilities,
            "prediction_threshold_0_5": (
                final_test_probabilities >= DEFAULT_DECISION_THRESHOLD
            ).astype(int),
            "prediction_optimized_threshold": (
                final_test_probabilities >= optimized_threshold
            ).astype(int),
        }
    ).to_csv(test_predictions_path, index=False)

    plot_precision_recall_curve(
        labels=test_labels,
        probabilities=final_test_probabilities,
        output_path=test_curve_path,
    )

    seed_duration = time.perf_counter() - seed_start

    print_banner(
        f"BILAN COMPLET — SEED {training_seed} — {TFIDF_DISPLAY_NAME}"
    )
    print_table(
        ["Fold", "F1 positif", "PR-AUC"],
        [
            [
                fold_index,
                f"{fold_f1_scores[fold_index - 1]:.4f}",
                f"{fold_pr_auc_scores[fold_index - 1]:.4f}",
            ]
            for fold_index in range(1, shared_split.fold_count + 1)
        ],
    )
    print_metrics_block("OOF — SEUIL STANDARD 0.5", oof_default_metrics)
    print_metrics_block("OOF — SEUIL OPTIMISÉ", oof_optimized_metrics)
    print_metrics_block("TEST — SEUIL STANDARD 0.5", test_default_metrics)
    print_metrics_block(
        "TEST — SEUIL OPTIMISÉ SUR LES OOF",
        test_optimized_metrics,
    )
    print_key_values(
        [
            ("Log loss test", f"{test_log_loss:.4f}"),
            ("Désaccord moyen entre folds", f"{fold_disagreement:.4f}"),
            ("Durée totale seed", format_duration(seed_duration)),
        ]
    )

    summary = {
        "protocol": "fixed_shared_stratified_kfold_multi_training_seeds",
        "model_display_name": TFIDF_DISPLAY_NAME,
        "training_seed": training_seed,
        "split_seed": shared_split.split_seed,
        "fold_count": shared_split.fold_count,
        "split_manifest_csv": shared_split.csv_path,
        "split_manifest_json": shared_split.metadata_path,
        "dataset_fingerprint": shared_split.dataset_fingerprint,
        "parameters": {
            "ngram_range": [1, args.ngram_max],
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": 0.98,
            "sublinear_tf": True,
            "class_weight": "balanced",
            "solver": "liblinear",
            "C": args.c,
            "max_iter": args.max_iter,
        },
        "fold_f1_scores": fold_f1_scores,
        "mean_fold_f1": float(np.mean(fold_f1_scores)),
        "std_fold_f1": sample_std(fold_f1_scores),
        "fold_pr_auc_scores": fold_pr_auc_scores,
        "mean_fold_pr_auc": float(np.mean(fold_pr_auc_scores)),
        "std_fold_pr_auc": sample_std(fold_pr_auc_scores),
        "folds": fold_summaries,
        "oof_default_metrics": oof_default_metrics,
        "oof_optimized_metrics": oof_optimized_metrics,
        "test_default_metrics": test_default_metrics,
        "test_optimized_metrics": test_optimized_metrics,
        "test_log_loss": test_log_loss,
        "mean_fold_probability_disagreement": fold_disagreement,
        "duration_seconds": float(seed_duration),
    }

    summary_path = os.path.join(seed_dir, "experiment_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------
# AGRÉGATION MULTI-SEEDS
# ---------------------------------------------------------------------
def aggregate_summaries(
    summaries: Sequence[Dict],
    total_duration: float,
) -> Dict:
    rows = []

    for summary in summaries:
        oof_default = summary["oof_default_metrics"]
        oof_optimized = summary["oof_optimized_metrics"]
        test_default = summary["test_default_metrics"]
        test_optimized = summary["test_optimized_metrics"]

        rows.append(
            {
                "training_seed": summary["training_seed"],
                "oof_f1_threshold_0_5": oof_default["f1"],
                "oof_pr_auc": oof_default["pr_auc"],
                "optimized_threshold": oof_optimized["threshold"],
                "oof_f1_optimized": oof_optimized["f1"],
                "test_f1_threshold_0_5": test_default["f1"],
                "test_precision_threshold_0_5": test_default["precision"],
                "test_recall_threshold_0_5": test_default["recall"],
                "test_accuracy_threshold_0_5": test_default["accuracy"],
                "test_pr_auc": test_default["pr_auc"],
                "test_f1_optimized": test_optimized["f1"],
                "test_precision_optimized": test_optimized["precision"],
                "test_recall_optimized": test_optimized["recall"],
                "test_accuracy_optimized": test_optimized["accuracy"],
                "test_log_loss": summary["test_log_loss"],
                "duration_seconds": summary["duration_seconds"],
            }
        )

    dataframe = pd.DataFrame(rows)
    results_path = os.path.join(OUTPUT_ROOT, "all_seeds_results.csv")
    dataframe.to_csv(results_path, index=False)

    metric_columns = [
        column
        for column in dataframe.columns
        if column not in {"training_seed", "duration_seconds"}
    ]
    aggregates = {}
    for column in metric_columns:
        values = dataframe[column].to_numpy(dtype=float)
        aggregates[column] = {
            "mean": float(np.mean(values)),
            "std": sample_std(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    benchmark_summary = {
        "protocol": "fixed_shared_stratified_kfold_multi_training_seeds",
        "model_display_name": TFIDF_DISPLAY_NAME,
        "training_seeds": [summary["training_seed"] for summary in summaries],
        "split_seed": summaries[0]["split_seed"],
        "split_manifest_csv": summaries[0]["split_manifest_csv"],
        "fold_count": summaries[0]["fold_count"],
        "total_trainings": int(len(summaries) * summaries[0]["fold_count"]),
        "primary_reporting": "mean_and_sample_std_across_training_seeds",
        "aggregate_metrics": aggregates,
        "seed_summaries": summaries,
        "all_seeds_results_csv": results_path,
        "total_duration_seconds": float(total_duration),
    }

    summary_path = os.path.join(OUTPUT_ROOT, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(benchmark_summary, file, indent=2, ensure_ascii=False)

    print_banner(f"RÉSULTAT FINAL MULTI-SEEDS — {TFIDF_DISPLAY_NAME}")
    print_table(
        [
            "Seed",
            "F1 OOF @0.5",
            "PR-AUC OOF",
            "Seuil OOF",
            "F1 test @0.5",
            "PR-AUC test",
            "F1 test optimisé",
        ],
        [
            [
                int(row["training_seed"]),
                f"{row['oof_f1_threshold_0_5']:.4f}",
                f"{row['oof_pr_auc']:.4f}",
                f"{row['optimized_threshold']:.3f}",
                f"{row['test_f1_threshold_0_5']:.4f}",
                f"{row['test_pr_auc']:.4f}",
                f"{row['test_f1_optimized']:.4f}",
            ]
            for row in rows
        ],
    )

    print("\nMoyenne ± écart-type entre les seeds")
    print_table(
        ["Métrique", "Moyenne", "Écart-type", "Min", "Max"],
        [
            [
                "F1 OOF @0.5",
                f"{aggregates['oof_f1_threshold_0_5']['mean']:.4f}",
                f"{aggregates['oof_f1_threshold_0_5']['std']:.4f}",
                f"{aggregates['oof_f1_threshold_0_5']['min']:.4f}",
                f"{aggregates['oof_f1_threshold_0_5']['max']:.4f}",
            ],
            [
                "PR-AUC OOF",
                f"{aggregates['oof_pr_auc']['mean']:.4f}",
                f"{aggregates['oof_pr_auc']['std']:.4f}",
                f"{aggregates['oof_pr_auc']['min']:.4f}",
                f"{aggregates['oof_pr_auc']['max']:.4f}",
            ],
            [
                "F1 test @0.5",
                f"{aggregates['test_f1_threshold_0_5']['mean']:.4f}",
                f"{aggregates['test_f1_threshold_0_5']['std']:.4f}",
                f"{aggregates['test_f1_threshold_0_5']['min']:.4f}",
                f"{aggregates['test_f1_threshold_0_5']['max']:.4f}",
            ],
            [
                "PR-AUC test",
                f"{aggregates['test_pr_auc']['mean']:.4f}",
                f"{aggregates['test_pr_auc']['std']:.4f}",
                f"{aggregates['test_pr_auc']['min']:.4f}",
                f"{aggregates['test_pr_auc']['max']:.4f}",
            ],
            [
                "F1 test optimisé",
                f"{aggregates['test_f1_optimized']['mean']:.4f}",
                f"{aggregates['test_f1_optimized']['std']:.4f}",
                f"{aggregates['test_f1_optimized']['min']:.4f}",
                f"{aggregates['test_f1_optimized']['max']:.4f}",
            ],
        ],
    )

    print("\nRésultat principal baseline à reporter :")
    print(
        "  F1 positif test @0.5 = "
        f"{aggregates['test_f1_threshold_0_5']['mean']:.4f} "
        f"± {aggregates['test_f1_threshold_0_5']['std']:.4f}"
    )
    print(
        "  PR-AUC test          = "
        f"{aggregates['test_pr_auc']['mean']:.4f} "
        f"± {aggregates['test_pr_auc']['std']:.4f}"
    )
    print(
        "  F1 test optimisé     = "
        f"{aggregates['test_f1_optimized']['mean']:.4f} "
        f"± {aggregates['test_f1_optimized']['std']:.4f}"
    )
    print(f"\nDurée totale : {format_duration(total_duration)}")
    print(f"CSV multi-seeds : {results_path}")
    print(f"Résumé JSON     : {summary_path}")

    return benchmark_summary


# ---------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne la baseline TF-IDF + LR avec les folds partagés "
            "des cinq seeds."
        )
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=K_FOLDS,
        help=f"Nombre de folds. Défaut : {K_FOLDS}.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in TRAINING_SEEDS),
        help=(
            "Seeds d'entraînement séparées par des virgules. Défaut : "
            + ",".join(str(seed) for seed in TRAINING_SEEDS)
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Nombre maximal de termes TF-IDF. Défaut : 50000.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Un terme doit apparaître dans au moins N notes. Défaut : 2.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="1 = unigrammes, 2 ajoute les bigrammes, 3 les trigrammes.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Force inverse de régularisation de la régression logistique.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Nombre maximal d'itérations du classifieur.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Supprime les anciens résultats TF-IDF. Les manifests partagés "
            "de folds sont conservés."
        ),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    total_start = time.perf_counter()
    seeds = parse_seed_list(args.seeds)

    if args.folds < 2:
        raise ValueError("Le nombre de folds doit être supérieur ou égal à 2.")
    if args.max_features <= 0:
        raise ValueError("--max-features doit être strictement positif.")
    if args.min_df <= 0:
        raise ValueError("--min-df doit être strictement positif.")
    if args.c <= 0:
        raise ValueError("--c doit être strictement positif.")
    if args.max_iter <= 0:
        raise ValueError("--max-iter doit être strictement positif.")

    train_dataframe = read_dataset(TRAIN_CV_CSV)
    test_dataframe = read_dataset(TEST_CSV)

    train_texts = train_dataframe[TEXT_COL].to_numpy(dtype=str)
    train_labels = train_dataframe[LABEL_COL].to_numpy(dtype=int)
    test_texts = test_dataframe[TEXT_COL].to_numpy(dtype=str)
    test_labels = test_dataframe[LABEL_COL].to_numpy(dtype=int)

    fingerprint = dataset_fingerprint(
        dataframe=train_dataframe,
        text_column=TEXT_COL,
        label_column=LABEL_COL,
    )

    prepare_output_root(overwrite=args.overwrite)

    print_banner("BASELINE DEREEV — TF-IDF + LR — PROTOCOLE A STRICT")
    print_key_values(
        [
            ("Modèle", TFIDF_DISPLAY_NAME),
            ("Train_CV", f"{len(train_dataframe)} notes"),
            ("Test fixe", f"{len(test_dataframe)} notes"),
            ("Seeds d'entraînement", ", ".join(map(str, seeds))),
            ("Seed du split fixe", CV_SPLIT_SEED),
            ("Folds", args.folds),
            ("Entraînements prévus", len(seeds) * args.folds),
            ("N-grams", f"1 à {args.ngram_max}"),
            ("Max features", args.max_features),
            ("min_df", args.min_df),
            ("C", args.c),
            ("Dossier de sortie", OUTPUT_ROOT),
            ("Fingerprint dataset", fingerprint),
        ]
    )
    print(
        "\nGARANTIE PROTOCOLE A : cette baseline recharge l'unique manifeste "
        "de folds fixe utilisé par CamemBERT-bio et les futurs modèles."
    )

    shared_split = ensure_shared_split(
        labels=train_labels,
        split_seed=CV_SPLIT_SEED,
        fold_count=args.folds,
        split_directory=CV_SPLITS_DIR,
        fingerprint=fingerprint,
    )

    print("\nManifeste de folds fixe et officiel")
    print_table(
        ["Seed du split", "Statut", "Folds", "Fichier partagé"],
        [[
            shared_split.split_seed,
            "créé" if shared_split.was_created else "rechargé",
            shared_split.fold_count,
            shared_split.csv_path,
        ]],
    )
    print(
        "CONFIRMATION : TF-IDF utilise exactement les mêmes indices "
        "train/validation que tous les entraînements Transformers."
    )

    summaries: List[Dict] = []
    total_trainings = len(seeds) * args.folds

    for seed_position, training_seed in enumerate(seeds):
        summaries.append(
            run_one_seed(
                shared_split=shared_split,
                training_seed=training_seed,
                train_texts=train_texts,
                train_labels=train_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                args=args,
                completed_before=seed_position * args.folds,
                total_trainings=total_trainings,
            )
        )

    aggregate_summaries(
        summaries=summaries,
        total_duration=time.perf_counter() - total_start,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())