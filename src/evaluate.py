"""Évaluation finale multi-seeds du modèle Transformer DEREEV.

Ce script charge, pour chaque seed d'entraînement, les K modèles des folds fixes,
moyenne leurs probabilités sur le jeu de test fixe, applique :

- le seuil standard 0.5 ;
- le seuil choisi uniquement sur les prédictions OOF de cette seed.

Le résultat principal est ensuite présenté sous la forme moyenne ± écart-type
entre les seeds d'entraînement. Le test n'est jamais utilisé pour choisir le seuil.

Commande recommandée :
    python3 -m evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

_FIXED_HASH_SEED = "0"

if os.environ.get("PYTHONHASHSEED") != _FIXED_HASH_SEED:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = _FIXED_HASH_SEED
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "src.evaluate", *sys.argv[1:]],
        environment,
    )

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from transformers import AutoModelForSequenceClassification, Trainer, set_seed

from src.config import (
    BENCHMARK_MODELS_DIR,
    CV_SPLIT_SEED,
    TRAINING_SEEDS,
    K_FOLDS,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    MODEL_OUTPUT_NAME,
    TEST_CSV,
    TRAIN_CV_CSV,
)
from src.cv_protocol import parse_seed_list
from src.data_loader import get_tokenized_datasets
from src.threshold_metrics import (
    add_pr_auc_to_metrics,
    compute_threshold_metrics,
    logits_to_positive_probabilities,
    plot_precision_recall_curve,
)


DEFAULT_DECISION_THRESHOLD = 0.5
MODEL_ROOT = os.path.join(BENCHMARK_MODELS_DIR, MODEL_OUTPUT_NAME)


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
# OUTILS
# ---------------------------------------------------------------------
def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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


def predict_with_model(model_path: str, test_ds) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model)
    prediction_output = trainer.predict(test_ds)

    probabilities = logits_to_positive_probabilities(
        prediction_output.predictions
    )
    labels = np.asarray(prediction_output.label_ids, dtype=int)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probabilities, labels


# ---------------------------------------------------------------------
# ÉVALUATION D'UNE SEED
# ---------------------------------------------------------------------
def evaluate_one_seed(
    training_seed: int,
    fold_count: int,
    test_ds,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    seed_start = time.perf_counter()
    configure_determinism(training_seed)

    seed_dir = os.path.join(MODEL_ROOT, f"seed_{training_seed}")
    summary_path = os.path.join(seed_dir, "experiment_summary.json")
    threshold_path = os.path.join(seed_dir, "decision_threshold.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Résumé d'entraînement introuvable pour la seed "
            f"{training_seed} : {summary_path}"
        )
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(
            f"Seuil OOF introuvable pour la seed {training_seed} : "
            f"{threshold_path}"
        )

    training_summary = read_json(summary_path)
    threshold_summary = read_json(threshold_path)

    trained_fold_count = int(training_summary.get("fold_count", fold_count))
    if trained_fold_count != fold_count:
        raise RuntimeError(
            f"La seed {training_seed} a été entraînée avec "
            f"{trained_fold_count} folds, mais l'évaluation en demande "
            f"{fold_count}."
        )

    trained_split_seed = int(
        training_summary.get("split_seed", CV_SPLIT_SEED)
    )
    if trained_split_seed != CV_SPLIT_SEED:
        raise RuntimeError(
            f"La seed d'entraînement {training_seed} utilise le split "
            f"{trained_split_seed}, alors que config.py impose le split "
            f"fixe {CV_SPLIT_SEED}."
        )

    expected_protocol = "fixed_shared_stratified_kfold_multi_training_seeds"
    trained_protocol = training_summary.get("protocol")
    if trained_protocol != expected_protocol:
        raise RuntimeError(
            f"Protocole incompatible pour la seed {training_seed} : "
            f"{trained_protocol!r}. Relancez l'entraînement avec la version "
            "corrigée du protocole A strict."
        )

    optimized_threshold = float(
        threshold_summary.get("best_threshold", DEFAULT_DECISION_THRESHOLD)
    )

    print_banner(
        f"TEST — SEED {training_seed} — {MODEL_DISPLAY_NAME}",
        character="#",
    )
    print_key_values(
        [
            ("Modèle", f"{MODEL_DISPLAY_NAME} ({MODEL_NAME})"),
            ("Seed d'entraînement", training_seed),
            ("Seed du split fixe", trained_split_seed),
            ("Folds à agréger", fold_count),
            ("Seuil OOF chargé", f"{optimized_threshold:.3f}"),
            ("Dossier", seed_dir),
        ]
    )

    fold_probabilities: List[np.ndarray] = []
    test_labels = None

    for fold_index in range(1, fold_count + 1):
        model_path = os.path.join(seed_dir, f"fold_{fold_index}")
        print(
            f"\n[{fold_index}/{fold_count}] Prédiction test avec "
            f"seed_{training_seed}/fold_{fold_index}"
        )
        probabilities, labels = predict_with_model(model_path, test_ds)

        if test_labels is None:
            test_labels = labels
        elif not np.array_equal(test_labels, labels):
            raise RuntimeError(
                "Les labels du test diffèrent entre les modèles des folds."
            )

        fold_probabilities.append(probabilities)

    probability_matrix = np.stack(fold_probabilities, axis=0)
    ensemble_probabilities = np.mean(probability_matrix, axis=0)

    default_metrics = metrics_with_pr_auc(
        labels=test_labels,
        probabilities=ensemble_probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )
    optimized_metrics = metrics_with_pr_auc(
        labels=test_labels,
        probabilities=ensemble_probabilities,
        threshold=optimized_threshold,
    )

    probability_pairs = np.column_stack(
        [1.0 - ensemble_probabilities, ensemble_probabilities]
    )
    final_log_loss = float(
        log_loss(test_labels, probability_pairs, labels=[0, 1])
    )
    fold_disagreement = float(
        np.mean(np.std(probability_matrix, axis=0, ddof=0))
    )

    duration = time.perf_counter() - seed_start

    print_metrics_block("TEST — SEUIL STANDARD 0.5", default_metrics)
    print_metrics_block(
        "TEST — SEUIL OPTIMISÉ SUR LES OOF DE CETTE SEED",
        optimized_metrics,
    )
    print_key_values(
        [
            ("Log loss", f"{final_log_loss:.4f}"),
            ("Désaccord moyen entre folds", f"{fold_disagreement:.4f}"),
            ("Durée de l'évaluation", format_duration(duration)),
        ]
    )

    predictions_path = os.path.join(seed_dir, "test_predictions.csv")
    pr_curve_path = os.path.join(seed_dir, "test_precision_recall_curve.png")
    results_path = os.path.join(seed_dir, "test_results.json")

    pd.DataFrame(
        {
            "label": test_labels,
            "training_seed": training_seed,
            "probability_class_1": ensemble_probabilities,
            "prediction_threshold_0_5": (
                ensemble_probabilities >= DEFAULT_DECISION_THRESHOLD
            ).astype(int),
            "prediction_optimized_threshold": (
                ensemble_probabilities >= optimized_threshold
            ).astype(int),
        }
    ).to_csv(predictions_path, index=False)

    plot_precision_recall_curve(
        labels=test_labels,
        probabilities=ensemble_probabilities,
        output_path=pr_curve_path,
    )

    result = {
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "training_seed": training_seed,
        "split_seed": trained_split_seed,
        "fold_count": fold_count,
        "optimized_threshold_selected_on_oof": optimized_threshold,
        "default_threshold_metrics": default_metrics,
        "optimized_threshold_metrics": optimized_metrics,
        "pr_auc": default_metrics["pr_auc"],
        "log_loss": final_log_loss,
        "mean_fold_probability_disagreement": fold_disagreement,
        "duration_seconds": float(duration),
        "predictions_path": predictions_path,
        "pr_curve_path": pr_curve_path,
    }

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Prédictions sauvegardées : {predictions_path}")
    print(f"Résumé test sauvegardé   : {results_path}")

    return result, ensemble_probabilities, test_labels


# ---------------------------------------------------------------------
# AGRÉGATION MULTI-SEEDS
# ---------------------------------------------------------------------
def aggregate_results(
    seed_results: Sequence[Dict],
    seed_probabilities: Sequence[np.ndarray],
    test_labels: np.ndarray,
    total_duration: float,
) -> Dict:
    rows = []

    for result in seed_results:
        default = result["default_threshold_metrics"]
        optimized = result["optimized_threshold_metrics"]
        rows.append(
            {
                "training_seed": result["training_seed"],
                "test_f1_threshold_0_5": default["f1"],
                "test_precision_threshold_0_5": default["precision"],
                "test_recall_threshold_0_5": default["recall"],
                "test_accuracy_threshold_0_5": default["accuracy"],
                "test_pr_auc": default["pr_auc"],
                "test_log_loss": result["log_loss"],
                "optimized_threshold": result[
                    "optimized_threshold_selected_on_oof"
                ],
                "test_f1_optimized": optimized["f1"],
                "test_precision_optimized": optimized["precision"],
                "test_recall_optimized": optimized["recall"],
                "test_accuracy_optimized": optimized["accuracy"],
                "fold_probability_disagreement": result[
                    "mean_fold_probability_disagreement"
                ],
                "duration_seconds": result["duration_seconds"],
            }
        )

    dataframe = pd.DataFrame(rows)
    results_csv = os.path.join(MODEL_ROOT, "all_seeds_test_results.csv")
    dataframe.to_csv(results_csv, index=False)

    metric_columns = [
        "test_f1_threshold_0_5",
        "test_precision_threshold_0_5",
        "test_recall_threshold_0_5",
        "test_accuracy_threshold_0_5",
        "test_pr_auc",
        "test_log_loss",
        "optimized_threshold",
        "test_f1_optimized",
        "test_precision_optimized",
        "test_recall_optimized",
        "test_accuracy_optimized",
        "fold_probability_disagreement",
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

    # Résultat secondaire : moyenne des ensembles de chaque seed.
    # Avec le même nombre de folds par seed, cela équivaut à la moyenne des
    # probabilités produites par les 25 modèles. Il ne remplace pas la moyenne
    # ± écart-type entre seeds, qui reste le résultat principal du protocole.
    seed_probability_matrix = np.stack(seed_probabilities, axis=0)
    all_models_probabilities = np.mean(seed_probability_matrix, axis=0)
    all_models_metrics = metrics_with_pr_auc(
        labels=test_labels,
        probabilities=all_models_probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )
    seed_disagreement = float(
        np.mean(np.std(seed_probability_matrix, axis=0, ddof=0))
    )

    grand_predictions_path = os.path.join(
        MODEL_ROOT,
        "all_seeds_ensemble_test_predictions.csv",
    )
    grand_curve_path = os.path.join(
        MODEL_ROOT,
        "all_seeds_ensemble_test_precision_recall_curve.png",
    )

    pd.DataFrame(
        {
            "label": test_labels,
            "probability_class_1": all_models_probabilities,
            "prediction_threshold_0_5": (
                all_models_probabilities >= DEFAULT_DECISION_THRESHOLD
            ).astype(int),
        }
    ).to_csv(grand_predictions_path, index=False)

    plot_precision_recall_curve(
        labels=test_labels,
        probabilities=all_models_probabilities,
        output_path=grand_curve_path,
    )

    summary = {
        "protocol": "fixed_shared_stratified_kfold_multi_training_seeds",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "training_seeds": [result["training_seed"] for result in seed_results],
        "split_seed": seed_results[0]["split_seed"],
        "fold_count": seed_results[0]["fold_count"],
        "primary_reporting": "mean_and_sample_std_across_training_seeds",
        "aggregate_test_metrics": aggregates,
        "seed_results": seed_results,
        "secondary_all_seed_fold_ensemble": {
            "description": (
                "Moyenne des probabilités de tous les modèles seed/fold ; "
                "résultat secondaire, sans estimation de variabilité."
            ),
            "threshold": DEFAULT_DECISION_THRESHOLD,
            "metrics": all_models_metrics,
            "mean_probability_disagreement_across_seeds": seed_disagreement,
            "predictions_path": grand_predictions_path,
            "pr_curve_path": grand_curve_path,
        },
        "all_seeds_test_csv": results_csv,
        "total_duration_seconds": float(total_duration),
    }

    summary_path = os.path.join(MODEL_ROOT, "test_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print_banner(
        f"RÉSULTAT TEST FINAL MULTI-SEEDS — {MODEL_DISPLAY_NAME}",
        character="=",
    )
    print_table(
        [
            "Seed",
            "F1 test @0.5",
            "PR-AUC test",
            "Seuil OOF",
            "F1 test optimisé",
            "Log loss",
        ],
        [
            [
                int(row["training_seed"]),
                f"{row['test_f1_threshold_0_5']:.4f}",
                f"{row['test_pr_auc']:.4f}",
                f"{row['optimized_threshold']:.3f}",
                f"{row['test_f1_optimized']:.4f}",
                f"{row['test_log_loss']:.4f}",
            ]
            for row in rows
        ],
    )

    print("\nMoyenne ± écart-type entre les seeds")
    print_table(
        ["Métrique", "Moyenne", "Écart-type", "Min", "Max"],
        [
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
            [
                "Log loss",
                f"{aggregates['test_log_loss']['mean']:.4f}",
                f"{aggregates['test_log_loss']['std']:.4f}",
                f"{aggregates['test_log_loss']['min']:.4f}",
                f"{aggregates['test_log_loss']['max']:.4f}",
            ],
        ],
    )

    print("\nRésultat principal à reporter sur le test :")
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

    print_metrics_block(
        "RÉSULTAT SECONDAIRE — ENSEMBLE DE TOUS LES MODÈLES @0.5",
        all_models_metrics,
    )
    print(
        "Désaccord moyen entre ensembles de seeds : "
        f"{seed_disagreement:.4f}"
    )
    print(
        "Attention : cet ensemble global est secondaire. Pour le benchmark, "
        "reportez la moyenne ± écart-type entre les seeds d'entraînement."
    )

    print(f"\nDurée totale de l'évaluation : {format_duration(total_duration)}")
    print(f"CSV multi-seeds : {results_csv}")
    print(f"Résumé JSON     : {summary_path}")

    return summary


# ---------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Évalue sur test les ensembles de folds des cinq seeds."
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
        "--folds",
        type=int,
        default=K_FOLDS,
        help=f"Nombre de folds entraînés par seed. Défaut : {K_FOLDS}.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    seeds = parse_seed_list(args.seeds)

    for path in (TRAIN_CV_CSV, TEST_CSV, MODEL_ROOT):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier ou dossier introuvable : {path}")

    _, test_ds = get_tokenized_datasets(TRAIN_CV_CSV, TEST_CSV)

    print_banner("ÉVALUATION TEST — PROTOCOLE A STRICT")
    print_key_values(
        [
            ("Modèle", MODEL_DISPLAY_NAME),
            ("Identifiant Hugging Face", MODEL_NAME),
            ("Dossier", MODEL_ROOT),
            ("Seeds d'entraînement", ", ".join(map(str, seeds))),
            ("Seed du split fixe", CV_SPLIT_SEED),
            ("Folds par seed", args.folds),
            ("Modèles chargés", len(seeds) * args.folds),
            ("Test fixe", TEST_CSV),
            ("Sélection des seuils", "uniquement sur les OOF de chaque seed"),
        ]
    )

    seed_results: List[Dict] = []
    seed_probabilities: List[np.ndarray] = []
    reference_labels = None

    for training_seed in seeds:
        result, probabilities, labels = evaluate_one_seed(
            training_seed=training_seed,
            fold_count=args.folds,
            test_ds=test_ds,
        )

        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError("Les labels du test diffèrent entre les seeds d'entraînement.")

        seed_results.append(result)
        seed_probabilities.append(probabilities)

    total_duration = time.perf_counter() - start
    aggregate_results(
        seed_results=seed_results,
        seed_probabilities=seed_probabilities,
        test_labels=reference_labels,
        total_duration=total_duration,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())