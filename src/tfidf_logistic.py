"""Baseline TF-IDF + régression logistique, group-aware et multi-seeds.

La baseline réutilise exactement les mêmes folds group-aware que les modèles
Transformers. Elle rapporte séparément les cinq modèles de fold individuels et
leur ensemble, puis calcule des IC bootstrap à 95 % sur le test.
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

from src.benchmark_statistics import (
    aggregate_metric_dicts,
    aggregate_scalar_values,
    bootstrap_mean_metrics_across_systems,
    metric_bundle,
)
from src.config import (
    BENCHMARK_MODELS_DIR,
    CV_SPLITS_DIR,
    CV_SPLIT_SEED,
    GROUP_COL,
    K_FOLDS,
    LABEL_COL,
    TEST_CSV,
    TEXT_COL,
    TFIDF_DISPLAY_NAME,
    TFIDF_OUTPUT_NAME,
    TRAINING_SEEDS,
    TRAIN_CV_CSV,
)
from src.cv_protocol import (
    SharedSplit,
    dataset_fingerprint,
    ensure_shared_split,
    parse_seed_list,
)
from src.threshold_metrics import (
    evaluate_thresholds,
    plot_precision_recall_curve,
    plot_threshold_curve,
)

DEFAULT_THRESHOLD = 0.5
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
OUTPUT_ROOT = os.path.join(BENCHMARK_MODELS_DIR, TFIDF_OUTPUT_NAME)
PROTOCOL = "fixed_shared_stratified_group_kfold_multi_training_seeds"


def print_banner(title: str, character: str = "=", width: int = 92) -> None:
    print("\n" + character * width)
    print(f" {title}")
    print(character * width)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print("| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |")
    print(separator)
    for row in rendered:
        print("| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |")
    print(separator)


def read_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dataframe = pd.read_csv(path)
    required = [TEXT_COL, LABEL_COL, GROUP_COL]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")
    dataframe = dataframe[required].copy().reset_index(drop=True)
    dataframe[TEXT_COL] = dataframe[TEXT_COL].fillna("").astype(str)
    dataframe[LABEL_COL] = dataframe[LABEL_COL].astype(int)
    dataframe[GROUP_COL] = dataframe[GROUP_COL].fillna("").astype(str)
    if dataframe[TEXT_COL].str.strip().eq("").any():
        raise ValueError(f"Note vide dans {path}.")
    if dataframe[GROUP_COL].str.strip().eq("").any():
        raise ValueError(f"Groupe vide dans {path}.")
    if not dataframe[LABEL_COL].isin([0, 1]).all():
        raise ValueError(f"Labels invalides dans {path}.")
    return dataframe


def build_pipeline(args: argparse.Namespace, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            sublinear_tf=True,
            dtype=np.float64,
        )),
        ("classifier", LogisticRegression(
            C=args.c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=args.max_iter,
            random_state=seed,
        )),
    ])


def prepare_output(overwrite: bool) -> None:
    if os.path.exists(OUTPUT_ROOT):
        if not overwrite:
            raise FileExistsError(
                f"{OUTPUT_ROOT} existe déjà. Utilisez --overwrite."
            )
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)


def run_one_seed(
    shared_split: SharedSplit,
    training_seed: int,
    train_texts: np.ndarray,
    train_labels: np.ndarray,
    test_texts: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict, np.ndarray]:
    seed_start = time.perf_counter()
    seed_dir = os.path.join(OUTPUT_ROOT, f"seed_{training_seed}")
    os.makedirs(seed_dir, exist_ok=True)

    oof_probabilities = np.full(train_labels.size, np.nan, dtype=float)
    oof_folds = np.full(train_labels.size, -1, dtype=int)
    fold_test_probabilities: List[np.ndarray] = []
    validation_rows: List[Dict] = []

    for fold_index, (train_indices, validation_indices) in enumerate(
        shared_split.indices(), start=1
    ):
        print_banner(
            f"TF-IDF — SEED {training_seed} — FOLD {fold_index}/{shared_split.fold_count}",
            character="-",
        )
        pipeline = build_pipeline(args, training_seed)
        fold_start = time.perf_counter()
        pipeline.fit(train_texts[train_indices], train_labels[train_indices])
        validation_probabilities = pipeline.predict_proba(
            train_texts[validation_indices]
        )[:, 1]
        test_probabilities = pipeline.predict_proba(test_texts)[:, 1]
        oof_probabilities[validation_indices] = validation_probabilities
        oof_folds[validation_indices] = fold_index
        fold_test_probabilities.append(test_probabilities)

        validation_metrics = metric_bundle(
            train_labels[validation_indices], validation_probabilities, 0.5
        )
        validation_rows.append({
            "training_seed": training_seed,
            "fold": fold_index,
            "train_size": int(train_indices.size),
            "validation_size": int(validation_indices.size),
            "validation_f1": validation_metrics["f1"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_pr_auc": validation_metrics["pr_auc"],
            "duration_seconds": float(time.perf_counter() - fold_start),
        })
        joblib.dump(pipeline, os.path.join(seed_dir, f"fold_{fold_index}.joblib"))

    if np.isnan(oof_probabilities).any() or np.any(oof_folds < 1):
        raise RuntimeError("Prédictions OOF incomplètes.")

    threshold_results, best = evaluate_thresholds(
        train_labels, oof_probabilities,
        minimum=THRESHOLD_MIN,
        maximum=THRESHOLD_MAX,
        step=THRESHOLD_STEP,
    )
    optimized_threshold = float(best["threshold"])
    oof_default = metric_bundle(train_labels, oof_probabilities, 0.5)
    oof_optimized = metric_bundle(
        train_labels, oof_probabilities, optimized_threshold
    )

    pd.DataFrame(threshold_results).to_csv(
        os.path.join(seed_dir, "threshold_results.csv"), index=False
    )
    plot_threshold_curve(
        threshold_results, os.path.join(seed_dir, "threshold_results.png")
    )
    plot_precision_recall_curve(
        train_labels, oof_probabilities,
        os.path.join(seed_dir, "precision_recall_curve.png"),
    )
    pd.DataFrame({
        "row_index": np.arange(train_labels.size),
        "training_seed": training_seed,
        "fold": oof_folds,
        "label": train_labels,
        "probability_class_1": oof_probabilities,
        "prediction_threshold_0_5": (oof_probabilities >= 0.5).astype(int),
        "prediction_optimized_threshold": (
            oof_probabilities >= optimized_threshold
        ).astype(int),
    }).to_csv(os.path.join(seed_dir, "validation_predictions.csv"), index=False)
    with open(os.path.join(seed_dir, "decision_threshold.json"), "w", encoding="utf-8") as file:
        json.dump({
            "training_seed": training_seed,
            "selection_data": "out_of_fold_validation",
            "best_threshold": optimized_threshold,
            "default_metrics": oof_default,
            "best_metrics": oof_optimized,
        }, file, indent=2, ensure_ascii=False)

    individual_rows: List[Dict] = []
    individual_default_metrics: List[Dict] = []
    individual_optimized_metrics: List[Dict] = []
    for fold_index, probabilities in enumerate(fold_test_probabilities, start=1):
        default_metrics = metric_bundle(test_labels, probabilities, 0.5)
        optimized_metrics = metric_bundle(
            test_labels, probabilities, optimized_threshold
        )
        individual_default_metrics.append(default_metrics)
        individual_optimized_metrics.append(optimized_metrics)
        individual_rows.append({
            "model_display_name": TFIDF_DISPLAY_NAME,
            "model_output_name": TFIDF_OUTPUT_NAME,
            "training_seed": training_seed,
            "fold": fold_index,
            "optimized_threshold_selected_on_seed_oof": optimized_threshold,
            "individual_default_f1": default_metrics["f1"],
            "individual_default_precision": default_metrics["precision"],
            "individual_default_recall": default_metrics["recall"],
            "individual_default_accuracy": default_metrics["accuracy"],
            "individual_default_pr_auc": default_metrics["pr_auc"],
            "individual_optimized_f1": optimized_metrics["f1"],
            "individual_optimized_precision": optimized_metrics["precision"],
            "individual_optimized_recall": optimized_metrics["recall"],
            "individual_optimized_accuracy": optimized_metrics["accuracy"],
            "individual_optimized_pr_auc": optimized_metrics["pr_auc"],
            "model_path": os.path.join(seed_dir, f"fold_{fold_index}.joblib"),
        })
    fold_results_path = os.path.join(seed_dir, "individual_fold_test_results.csv")
    pd.DataFrame(individual_rows).to_csv(fold_results_path, index=False)

    test_probability_matrix = np.stack(fold_test_probabilities, axis=0)
    ensemble_probabilities = np.mean(test_probability_matrix, axis=0)
    ensemble_default = metric_bundle(test_labels, ensemble_probabilities, 0.5)
    ensemble_optimized = metric_bundle(
        test_labels, ensemble_probabilities, optimized_threshold
    )
    individual_default_aggregate = aggregate_metric_dicts(individual_default_metrics)
    individual_optimized_aggregate = aggregate_metric_dicts(individual_optimized_metrics)

    prediction_columns = {
        "row_index": np.arange(test_labels.size),
        "label": test_labels,
        "training_seed": np.repeat(training_seed, test_labels.size),
    }
    for fold_index, probabilities in enumerate(fold_test_probabilities, start=1):
        prediction_columns[f"fold_{fold_index}_probability_class_1"] = probabilities
    prediction_columns.update({
        "probability_class_1": ensemble_probabilities,
        "optimized_threshold_from_oof": np.repeat(
            optimized_threshold, test_labels.size
        ),
        "prediction_threshold_0_5": (ensemble_probabilities >= 0.5).astype(int),
        "prediction_optimized_threshold": (
            ensemble_probabilities >= optimized_threshold
        ).astype(int),
    })
    predictions_path = os.path.join(seed_dir, "test_predictions.csv")
    pd.DataFrame(prediction_columns).to_csv(predictions_path, index=False)
    plot_precision_recall_curve(
        test_labels, ensemble_probabilities,
        os.path.join(seed_dir, "test_precision_recall_curve.png"),
    )

    logloss = float(log_loss(
        test_labels,
        np.column_stack([1.0 - ensemble_probabilities, ensemble_probabilities]),
        labels=[0, 1],
    ))
    seed_summary = {
        "protocol": PROTOCOL,
        "model_display_name": TFIDF_DISPLAY_NAME,
        "model_output_name": TFIDF_OUTPUT_NAME,
        "training_seed": training_seed,
        "split_seed": shared_split.split_seed,
        "fold_count": shared_split.fold_count,
        "optimized_threshold_selected_on_oof": optimized_threshold,
        "oof_default_metrics": oof_default,
        "oof_optimized_metrics": oof_optimized,
        "folds": validation_rows,
        "individual_fold_models": {
            "default_threshold_aggregate": individual_default_aggregate,
            "optimized_threshold_aggregate": individual_optimized_aggregate,
            "results_csv": fold_results_path,
        },
        "fold_ensemble": {
            "default_threshold_metrics": ensemble_default,
            "optimized_threshold_metrics": ensemble_optimized,
            "log_loss": logloss,
            "mean_fold_probability_disagreement": float(
                np.mean(np.std(test_probability_matrix, axis=0, ddof=0))
            ),
        },
        "training_configuration": {
            "vectorizer": "word_tfidf",
            "ngram_range": [1, args.ngram_max],
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "classifier": "logistic_regression_liblinear",
            "c": args.c,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
        },
        "duration_seconds": float(time.perf_counter() - seed_start),
        "predictions_path": predictions_path,
    }
    with open(os.path.join(seed_dir, "experiment_summary.json"), "w", encoding="utf-8") as file:
        json.dump(seed_summary, file, indent=2, ensure_ascii=False)
    with open(os.path.join(seed_dir, "test_results.json"), "w", encoding="utf-8") as file:
        json.dump(seed_summary, file, indent=2, ensure_ascii=False)
    return seed_summary, ensemble_probabilities


def aggregate_all(
    seed_summaries: Sequence[Dict],
    seed_probabilities: Sequence[np.ndarray],
    test_labels: np.ndarray,
    args: argparse.Namespace,
    total_duration: float,
) -> None:
    rows = []
    fold_frames = []
    thresholds = []
    for summary in seed_summaries:
        threshold = summary["optimized_threshold_selected_on_oof"]
        thresholds.append(threshold)
        individual_default = summary["individual_fold_models"]["default_threshold_aggregate"]
        individual_optimized = summary["individual_fold_models"]["optimized_threshold_aggregate"]
        ensemble_default = summary["fold_ensemble"]["default_threshold_metrics"]
        ensemble_optimized = summary["fold_ensemble"]["optimized_threshold_metrics"]
        rows.append({
            "training_seed": summary["training_seed"],
            "optimized_threshold": threshold,
            "individual_mean_f1_threshold_0_5": individual_default["f1"]["mean"],
            "individual_mean_precision_threshold_0_5": individual_default["precision"]["mean"],
            "individual_mean_recall_threshold_0_5": individual_default["recall"]["mean"],
            "individual_mean_pr_auc": individual_default["pr_auc"]["mean"],
            "individual_mean_f1_optimized": individual_optimized["f1"]["mean"],
            "ensemble_f1_threshold_0_5": ensemble_default["f1"],
            "ensemble_precision_threshold_0_5": ensemble_default["precision"],
            "ensemble_recall_threshold_0_5": ensemble_default["recall"],
            "ensemble_accuracy_threshold_0_5": ensemble_default["accuracy"],
            "ensemble_pr_auc": ensemble_default["pr_auc"],
            "ensemble_f1_optimized": ensemble_optimized["f1"],
            "ensemble_precision_optimized": ensemble_optimized["precision"],
            "ensemble_recall_optimized": ensemble_optimized["recall"],
            "ensemble_accuracy_optimized": ensemble_optimized["accuracy"],
            "ensemble_log_loss": summary["fold_ensemble"]["log_loss"],
            "fold_probability_disagreement": summary["fold_ensemble"][
                "mean_fold_probability_disagreement"
            ],
            "duration_seconds": summary["duration_seconds"],
        })
        fold_frames.append(pd.read_csv(
            summary["individual_fold_models"]["results_csv"]
        ))

    dataframe = pd.DataFrame(rows)
    seed_csv = os.path.join(OUTPUT_ROOT, "all_seeds_test_results.csv")
    dataframe.to_csv(seed_csv, index=False)
    fold_csv = os.path.join(OUTPUT_ROOT, "all_fold_models_test_results.csv")
    pd.concat(fold_frames, ignore_index=True).to_csv(fold_csv, index=False)

    aggregate_columns = [
        column for column in dataframe.columns
        if column not in {"training_seed", "duration_seconds"}
    ]
    aggregates = {
        column: aggregate_scalar_values(dataframe[column].to_numpy(float))
        for column in aggregate_columns
    }
    probability_matrix = np.stack(seed_probabilities, axis=0)
    default_bootstrap = bootstrap_mean_metrics_across_systems(
        test_labels, probability_matrix, np.repeat(0.5, len(seed_summaries)),
        n_resamples=args.bootstrap_resamples,
        random_seed=args.bootstrap_seed,
    )
    optimized_bootstrap = bootstrap_mean_metrics_across_systems(
        test_labels, probability_matrix, thresholds,
        n_resamples=args.bootstrap_resamples,
        random_seed=args.bootstrap_seed,
    )
    bootstrap = {
        "scope": "sampling_uncertainty_on_fixed_test_observations",
        "ensemble_threshold_0_5": default_bootstrap,
        "ensemble_threshold_optimized_on_oof": optimized_bootstrap,
    }
    with open(os.path.join(OUTPUT_ROOT, "bootstrap_test_confidence_intervals.json"), "w", encoding="utf-8") as file:
        json.dump(bootstrap, file, indent=2, ensure_ascii=False)

    benchmark_summary = {
        "protocol": PROTOCOL,
        "model_display_name": TFIDF_DISPLAY_NAME,
        "model_output_name": TFIDF_OUTPUT_NAME,
        "training_seeds": [summary["training_seed"] for summary in seed_summaries],
        "split_seed": seed_summaries[0]["split_seed"],
        "fold_count": seed_summaries[0]["fold_count"],
        "error_bar_scope": "inter_seed_variability_at_fixed_split_only",
        "training_configuration": seed_summaries[0]["training_configuration"],
        "aggregate_validation_metrics": {
            "oof_f1_threshold_0_5": aggregate_scalar_values([
                summary["oof_default_metrics"]["f1"] for summary in seed_summaries
            ]),
            "oof_pr_auc": aggregate_scalar_values([
                summary["oof_default_metrics"]["pr_auc"] for summary in seed_summaries
            ]),
            "optimized_threshold": aggregate_scalar_values(thresholds),
            "oof_f1_optimized": aggregate_scalar_values([
                summary["oof_optimized_metrics"]["f1"] for summary in seed_summaries
            ]),
        },
        "seed_summaries": list(seed_summaries),
        "total_duration_seconds": total_duration,
    }
    with open(os.path.join(OUTPUT_ROOT, "benchmark_summary.json"), "w", encoding="utf-8") as file:
        json.dump(benchmark_summary, file, indent=2, ensure_ascii=False)

    test_summary = {
        "protocol": PROTOCOL,
        "model_display_name": TFIDF_DISPLAY_NAME,
        "model_output_name": TFIDF_OUTPUT_NAME,
        "training_seeds": benchmark_summary["training_seeds"],
        "split_seed": benchmark_summary["split_seed"],
        "fold_count": benchmark_summary["fold_count"],
        "error_bar_scope": "inter_seed_variability_at_fixed_split_only",
        "declared_deployment_strategy": args.deployment_strategy,
        "aggregate_test_metrics": aggregates,
        "bootstrap_confidence_intervals": bootstrap,
        "seed_results": list(seed_summaries),
        "all_seeds_test_csv": seed_csv,
        "all_fold_models_test_csv": fold_csv,
        "total_duration_seconds": total_duration,
    }
    with open(os.path.join(OUTPUT_ROOT, "test_benchmark_summary.json"), "w", encoding="utf-8") as file:
        json.dump(test_summary, file, indent=2, ensure_ascii=False)

    print_banner("RÉSULTATS TF-IDF")
    print_table(
        ["Seed", "F1 individuel", "F1 ensemble", "Précision", "Rappel", "PR-AUC", "Seuil", "F1 opt."],
        [[
            int(row["training_seed"]),
            f"{row['individual_mean_f1_threshold_0_5']:.4f}",
            f"{row['ensemble_f1_threshold_0_5']:.4f}",
            f"{row['ensemble_precision_threshold_0_5']:.4f}",
            f"{row['ensemble_recall_threshold_0_5']:.4f}",
            f"{row['ensemble_pr_auc']:.4f}",
            f"{row['optimized_threshold']:.3f}",
            f"{row['ensemble_f1_optimized']:.4f}",
        ] for row in rows],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=K_FOLDS)
    parser.add_argument("--seeds", default=",".join(map(str, TRAINING_SEEDS)))
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.98)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--deployment-strategy",
        choices=["fold_ensemble", "single_model"],
        default="fold_ensemble",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    seeds = parse_seed_list(args.seeds)
    train = read_dataset(TRAIN_CV_CSV)
    test = read_dataset(TEST_CSV)
    overlap = set(train[GROUP_COL]) & set(test[GROUP_COL])
    if overlap:
        raise RuntimeError(f"{len(overlap)} groupes communs entre train et test.")
    fingerprint = dataset_fingerprint(train, TEXT_COL, LABEL_COL, GROUP_COL)
    shared_split = ensure_shared_split(
        labels=train[LABEL_COL].to_numpy(int),
        groups=train[GROUP_COL].to_numpy(str),
        split_seed=CV_SPLIT_SEED,
        fold_count=args.folds,
        split_directory=CV_SPLITS_DIR,
        fingerprint=fingerprint,
    )
    prepare_output(args.overwrite)

    run_configuration = {
        "protocol": PROTOCOL,
        "model_display_name": TFIDF_DISPLAY_NAME,
        "training_seeds": seeds,
        "split_seed": CV_SPLIT_SEED,
        "fold_count": args.folds,
        "ngram_range": [1, args.ngram_max],
        "max_features": args.max_features,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "c": args.c,
        "max_iter": args.max_iter,
        "error_bar_scope": "inter_seed_variability_at_fixed_split_only",
    }
    with open(os.path.join(OUTPUT_ROOT, "run_configuration.json"), "w", encoding="utf-8") as file:
        json.dump(run_configuration, file, indent=2, ensure_ascii=False)

    train_texts = train[TEXT_COL].to_numpy(str)
    train_labels = train[LABEL_COL].to_numpy(int)
    test_texts = test[TEXT_COL].to_numpy(str)
    test_labels = test[LABEL_COL].to_numpy(int)
    summaries = []
    probabilities = []
    for seed in seeds:
        summary, seed_probabilities = run_one_seed(
            shared_split, seed, train_texts, train_labels,
            test_texts, test_labels, args,
        )
        summaries.append(summary)
        probabilities.append(seed_probabilities)
    aggregate_all(
        summaries,
        probabilities,
        test_labels,
        args,
        time.perf_counter() - start,
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
