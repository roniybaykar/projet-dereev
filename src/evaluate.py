"""Évaluation finale multi-seeds du benchmark Transformer DEREEV.

Cette version rapporte séparément, pour chaque seed :

1. la moyenne des performances des cinq modèles de fold pris individuellement ;
2. la performance de l'ensemble obtenu en moyennant leurs probabilités.

Les seuils optimisés sont sélectionnés exclusivement sur les prédictions OOF.
Les ``moyenne ± écart-type`` mesurent uniquement la variabilité inter-seed à
partition fixe. Des IC bootstrap à 95 % sur les observations du test sont
également calculés pour l'incertitude d'échantillonnage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

from src.benchmark_statistics import (
    aggregate_metric_dicts,
    aggregate_scalar_values,
    bootstrap_mean_metrics_across_systems,
    metric_bundle,
)
from src.config import (
    BENCHMARK_MODELS_DIR,
    CV_SPLIT_SEED,
    K_FOLDS,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    MODEL_OUTPUT_NAME,
    TEST_CSV,
    TRAINING_SEEDS,
    TRAIN_CV_CSV,
)
from src.cv_protocol import parse_seed_list
from src.data_loader import get_tokenized_datasets
from src.threshold_metrics import (
    logits_to_positive_probabilities,
    plot_precision_recall_curve,
)

DEFAULT_DECISION_THRESHOLD = 0.5
MODEL_ROOT = os.path.join(BENCHMARK_MODELS_DIR, MODEL_OUTPUT_NAME)
GROUP_AWARE_PROTOCOL = "fixed_shared_stratified_group_kfold_multi_training_seeds"

# Audit des contradictions fortes entre les cinq modèles de fold.
# Exemple détecté : certains folds proches de 1 et d'autres proches de 0.
STRONG_DISAGREEMENT_HIGH = 0.80
STRONG_DISAGREEMENT_LOW = 0.20

# Caractères de contrôle interdits dans les fichiers Excel .xlsx.
EXCEL_ILLEGAL_CHARACTERS_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F]"
)


def print_banner(title: str, width: int = 92, character: str = "=") -> None:
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
    print("| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |")
    print(separator)
    for row in rendered:
        print("| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |")
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


def save_json(path: str, content: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=2, ensure_ascii=False)


def predict_with_model(model_path: str, test_dataset) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model)
    output = trainer.predict(test_dataset)
    probabilities = logits_to_positive_probabilities(output.predictions)
    labels = np.asarray(output.label_ids, dtype=int)
    del trainer, model, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probabilities, labels


def _metric_row(prefix: str, metrics: Dict) -> Dict[str, float]:
    return {
        f"{prefix}_f1": metrics["f1"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_pr_auc": metrics["pr_auc"],
    }



def sanitize_dataframe_for_excel(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Retire uniquement les caractères de contrôle interdits par Excel."""
    sanitized = dataframe.copy()

    for column in sanitized.columns:
        if (
            pd.api.types.is_object_dtype(sanitized[column])
            or pd.api.types.is_string_dtype(sanitized[column])
        ):
            sanitized[column] = sanitized[column].map(
                lambda value: (
                    EXCEL_ILLEGAL_CHARACTERS_RE.sub("", value)
                    if isinstance(value, str)
                    else value
                )
            )

    return sanitized


def export_disagreement_workbook(
    all_predictions: pd.DataFrame,
    strong_disagreements: pd.DataFrame,
    output_path: str,
) -> None:
    """Crée un seul classeur avec toutes les prédictions et les désaccords."""
    all_predictions_excel = sanitize_dataframe_for_excel(all_predictions)
    strong_disagreements_excel = sanitize_dataframe_for_excel(
        strong_disagreements
    )

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            all_predictions_excel.to_excel(
                writer,
                sheet_name="toutes_predictions",
                index=False,
            )
            strong_disagreements_excel.to_excel(
                writer,
                sheet_name="forts_desaccords",
                index=False,
            )
    except ImportError as error:
        raise RuntimeError(
            "openpyxl est requis pour l'export Excel. "
            "Installez-le avec : pip install openpyxl"
        ) from error


def evaluate_one_seed(
    training_seed: int,
    fold_count: int,
    test_dataset,
    test_metadata: pd.DataFrame,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    seed_start = time.perf_counter()
    configure_determinism(training_seed)

    seed_directory = os.path.join(MODEL_ROOT, f"seed_{training_seed}")
    summary_path = os.path.join(seed_directory, "experiment_summary.json")
    threshold_path = os.path.join(seed_directory, "decision_threshold.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Résumé d'entraînement introuvable : {summary_path}")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Seuil OOF introuvable : {threshold_path}")

    training_summary = read_json(summary_path)
    threshold_summary = read_json(threshold_path)
    trained_fold_count = int(training_summary.get("fold_count", fold_count))
    if trained_fold_count != fold_count:
        raise RuntimeError(
            f"Seed {training_seed} entraînée avec {trained_fold_count} folds, "
            f"mais évaluation demandée avec {fold_count}."
        )
    trained_protocol = training_summary.get("protocol")
    if trained_protocol != GROUP_AWARE_PROTOCOL:
        raise RuntimeError(
            f"Protocole incompatible : {trained_protocol!r}. "
            "Réentraînez avec StratifiedGroupKFold."
        )
    trained_split_seed = int(training_summary.get("split_seed", CV_SPLIT_SEED))
    if trained_split_seed != CV_SPLIT_SEED:
        raise RuntimeError("La seed du split ne correspond pas à config.py.")

    optimized_threshold = float(
        threshold_summary.get("best_threshold", DEFAULT_DECISION_THRESHOLD)
    )

    print_banner(f"TEST — SEED {training_seed} — {MODEL_DISPLAY_NAME}", character="#")
    print_key_values([
        ("Seed d'entraînement", training_seed),
        ("Folds", fold_count),
        ("Seuil OOF", f"{optimized_threshold:.3f}"),
        ("Comparaison", "modèles individuels de fold vs ensemble des folds"),
    ])

    fold_probabilities: List[np.ndarray] = []
    fold_rows: List[Dict] = []
    reference_labels: np.ndarray | None = None

    for fold_index in range(1, fold_count + 1):
        model_path = os.path.join(seed_directory, f"fold_{fold_index}")
        print(f"[{fold_index}/{fold_count}] Prédiction du test avec {model_path}")
        probabilities, labels = predict_with_model(model_path, test_dataset)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError("Les labels test diffèrent entre les folds.")

        default_metrics = metric_bundle(labels, probabilities, 0.5)
        optimized_metrics = metric_bundle(labels, probabilities, optimized_threshold)
        fold_rows.append({
            "model_display_name": MODEL_DISPLAY_NAME,
            "model_output_name": MODEL_OUTPUT_NAME,
            "training_seed": int(training_seed),
            "fold": int(fold_index),
            "optimized_threshold_selected_on_seed_oof": optimized_threshold,
            **_metric_row("individual_default", default_metrics),
            **_metric_row("individual_optimized", optimized_metrics),
            "model_path": model_path,
        })
        fold_probabilities.append(probabilities)

    assert reference_labels is not None
    probability_matrix = np.stack(fold_probabilities, axis=0)
    ensemble_probabilities = np.mean(probability_matrix, axis=0)
    ensemble_default = metric_bundle(reference_labels, ensemble_probabilities, 0.5)
    ensemble_optimized = metric_bundle(
        reference_labels, ensemble_probabilities, optimized_threshold
    )
    individual_default_metrics = [
        metric_bundle(reference_labels, probabilities, 0.5)
        for probabilities in fold_probabilities
    ]
    individual_optimized_metrics = [
        metric_bundle(reference_labels, probabilities, optimized_threshold)
        for probabilities in fold_probabilities
    ]
    individual_default_aggregate = aggregate_metric_dicts(individual_default_metrics)
    individual_optimized_aggregate = aggregate_metric_dicts(individual_optimized_metrics)

    probability_pairs = np.column_stack([
        1.0 - ensemble_probabilities,
        ensemble_probabilities,
    ])
    final_log_loss = float(log_loss(reference_labels, probability_pairs, labels=[0, 1]))
    fold_disagreement = float(np.mean(np.std(probability_matrix, axis=0, ddof=0)))

    fold_results_path = os.path.join(seed_directory, "individual_fold_test_results.csv")
    pd.DataFrame(fold_rows).to_csv(fold_results_path, index=False)

    if len(test_metadata) != reference_labels.size:
        raise RuntimeError(
            "Le nombre de lignes du fichier test brut ne correspond pas "
            "au nombre de prédictions du dataset tokenisé."
        )

    # On conserve les colonnes originales du test, notamment cn_notes et pat_ipp
    # lorsqu'elles existent, sans modifier le dataset ni l'entraînement.
    predictions_dataframe = test_metadata.reset_index(drop=True).copy()
    predictions_dataframe.insert(
        0,
        "row_index",
        np.arange(reference_labels.size, dtype=int),
    )
    predictions_dataframe["label_evaluation"] = reference_labels
    predictions_dataframe["training_seed"] = int(training_seed)

    for fold_index, probabilities in enumerate(fold_probabilities, start=1):
        predictions_dataframe[
            f"fold_{fold_index}_probability_class_1"
        ] = probabilities

    probability_min = np.min(probability_matrix, axis=0)
    probability_max = np.max(probability_matrix, axis=0)
    probability_range = probability_max - probability_min
    probability_std = np.std(probability_matrix, axis=0, ddof=0)
    positive_votes = np.sum(probability_matrix >= 0.5, axis=0)

    strong_disagreement = (
        (probability_max >= STRONG_DISAGREEMENT_HIGH)
        & (probability_min <= STRONG_DISAGREEMENT_LOW)
    )

    predictions_dataframe["probability_class_1"] = ensemble_probabilities
    predictions_dataframe["prediction_threshold_0_5"] = (
        ensemble_probabilities >= 0.5
    ).astype(int)
    predictions_dataframe["optimized_threshold_from_oof"] = (
        optimized_threshold
    )
    predictions_dataframe["prediction_optimized_threshold"] = (
        ensemble_probabilities >= optimized_threshold
    ).astype(int)

    predictions_dataframe["positive_votes_threshold_0_5"] = positive_votes
    predictions_dataframe["probability_min_across_folds"] = probability_min
    predictions_dataframe["probability_max_across_folds"] = probability_max
    predictions_dataframe["probability_range_across_folds"] = probability_range
    predictions_dataframe["probability_std_across_folds"] = probability_std
    predictions_dataframe["strong_disagreement"] = strong_disagreement
    predictions_dataframe["disagreement_rule"] = np.where(
        strong_disagreement,
        (
            f"max >= {STRONG_DISAGREEMENT_HIGH:.2f} "
            f"et min <= {STRONG_DISAGREEMENT_LOW:.2f}"
        ),
        "",
    )

    predictions_path = os.path.join(seed_directory, "test_predictions.csv")
    predictions_dataframe.to_csv(predictions_path, index=False)

    pr_curve_path = os.path.join(seed_directory, "test_precision_recall_curve.png")
    plot_precision_recall_curve(
        labels=reference_labels,
        probabilities=ensemble_probabilities,
        output_path=pr_curve_path,
    )

    duration = time.perf_counter() - seed_start
    result = {
        "protocol": GROUP_AWARE_PROTOCOL,
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_output_name": MODEL_OUTPUT_NAME,
        "training_seed": int(training_seed),
        "split_seed": int(trained_split_seed),
        "fold_count": int(fold_count),
        "optimized_threshold_selected_on_oof": optimized_threshold,
        "individual_fold_models": {
            "interpretation": (
                "Moyenne des métriques des cinq modèles de fold évalués "
                "séparément sur le même test."
            ),
            "default_threshold_aggregate": individual_default_aggregate,
            "optimized_threshold_aggregate": individual_optimized_aggregate,
            "results_csv": fold_results_path,
        },
        "fold_ensemble": {
            "interpretation": (
                "Probabilités moyennées entre les cinq modèles de fold ; "
                "c'est un système d'ensemble distinct des modèles individuels."
            ),
            "default_threshold_metrics": ensemble_default,
            "optimized_threshold_metrics": ensemble_optimized,
            "log_loss": final_log_loss,
            "mean_fold_probability_disagreement": fold_disagreement,
        },
        # Clés conservées pour compatibilité avec les anciens scripts.
        "default_threshold_metrics": ensemble_default,
        "optimized_threshold_metrics": ensemble_optimized,
        "pr_auc": ensemble_default["pr_auc"],
        "log_loss": final_log_loss,
        "mean_fold_probability_disagreement": fold_disagreement,
        "strong_disagreement_rule": {
            "high_probability_at_least": STRONG_DISAGREEMENT_HIGH,
            "low_probability_at_most": STRONG_DISAGREEMENT_LOW,
        },
        "strong_disagreement_count": int(np.sum(strong_disagreement)),
        "strong_disagreement_rate": float(np.mean(strong_disagreement)),
        "duration_seconds": float(duration),
        "predictions_path": predictions_path,
        "pr_curve_path": pr_curve_path,
    }
    results_path = os.path.join(seed_directory, "test_results.json")
    save_json(results_path, result)

    print_table(
        ["Objet évalué", "F1 @0.5", "Précision", "Rappel", "PR-AUC", "F1 seuil OOF"],
        [
            [
                "Moyenne des 5 modèles individuels",
                f"{individual_default_aggregate['f1']['mean']:.4f}",
                f"{individual_default_aggregate['precision']['mean']:.4f}",
                f"{individual_default_aggregate['recall']['mean']:.4f}",
                f"{individual_default_aggregate['pr_auc']['mean']:.4f}",
                f"{individual_optimized_aggregate['f1']['mean']:.4f}",
            ],
            [
                "Ensemble des 5 folds",
                f"{ensemble_default['f1']:.4f}",
                f"{ensemble_default['precision']:.4f}",
                f"{ensemble_default['recall']:.4f}",
                f"{ensemble_default['pr_auc']:.4f}",
                f"{ensemble_optimized['f1']:.4f}",
            ],
        ],
    )
    print(f"Durée : {format_duration(duration)}")
    return result, ensemble_probabilities, reference_labels


def _aggregate_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> Dict:
    return {
        column: aggregate_scalar_values(dataframe[column].to_numpy(dtype=float))
        for column in columns
    }


def aggregate_results(
    seed_results: Sequence[Dict],
    seed_probabilities: Sequence[np.ndarray],
    test_labels: np.ndarray,
    total_duration: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    deployment_strategy: str,
) -> Dict:
    rows: List[Dict] = []
    all_fold_rows: List[pd.DataFrame] = []
    all_prediction_rows: List[pd.DataFrame] = []
    optimized_thresholds: List[float] = []

    for result in seed_results:
        seed = int(result["training_seed"])
        threshold = float(result["optimized_threshold_selected_on_oof"])
        optimized_thresholds.append(threshold)
        individual_default = result["individual_fold_models"][
            "default_threshold_aggregate"
        ]
        individual_optimized = result["individual_fold_models"][
            "optimized_threshold_aggregate"
        ]
        ensemble_default = result["fold_ensemble"]["default_threshold_metrics"]
        ensemble_optimized = result["fold_ensemble"]["optimized_threshold_metrics"]
        rows.append({
            "training_seed": seed,
            "optimized_threshold": threshold,
            "individual_mean_f1_threshold_0_5": individual_default["f1"]["mean"],
            "individual_mean_precision_threshold_0_5": individual_default["precision"]["mean"],
            "individual_mean_recall_threshold_0_5": individual_default["recall"]["mean"],
            "individual_mean_pr_auc": individual_default["pr_auc"]["mean"],
            "individual_mean_f1_optimized": individual_optimized["f1"]["mean"],
            "individual_mean_precision_optimized": individual_optimized["precision"]["mean"],
            "individual_mean_recall_optimized": individual_optimized["recall"]["mean"],
            "ensemble_f1_threshold_0_5": ensemble_default["f1"],
            "ensemble_precision_threshold_0_5": ensemble_default["precision"],
            "ensemble_recall_threshold_0_5": ensemble_default["recall"],
            "ensemble_accuracy_threshold_0_5": ensemble_default["accuracy"],
            "ensemble_pr_auc": ensemble_default["pr_auc"],
            "ensemble_f1_optimized": ensemble_optimized["f1"],
            "ensemble_precision_optimized": ensemble_optimized["precision"],
            "ensemble_recall_optimized": ensemble_optimized["recall"],
            "ensemble_accuracy_optimized": ensemble_optimized["accuracy"],
            "ensemble_log_loss": result["fold_ensemble"]["log_loss"],
            "fold_probability_disagreement": result["fold_ensemble"][
                "mean_fold_probability_disagreement"
            ],
            "duration_seconds": result["duration_seconds"],
        })
        all_fold_rows.append(pd.read_csv(
            result["individual_fold_models"]["results_csv"]
        ))
        all_prediction_rows.append(
            pd.read_csv(result["predictions_path"], low_memory=False)
        )

    seed_dataframe = pd.DataFrame(rows)
    seed_csv = os.path.join(MODEL_ROOT, "all_seeds_test_results.csv")
    seed_dataframe.to_csv(seed_csv, index=False)
    fold_dataframe = pd.concat(all_fold_rows, ignore_index=True)
    fold_csv = os.path.join(MODEL_ROOT, "all_fold_models_test_results.csv")
    fold_dataframe.to_csv(fold_csv, index=False)

    all_predictions_dataframe = pd.concat(
        all_prediction_rows,
        ignore_index=True,
    )
    all_predictions_csv = os.path.join(
        MODEL_ROOT,
        "all_seeds_detailed_test_predictions.csv",
    )
    all_predictions_dataframe.to_csv(all_predictions_csv, index=False)

    strong_disagreements_dataframe = all_predictions_dataframe.loc[
        all_predictions_dataframe["strong_disagreement"].astype(bool)
    ].copy()

    disagreement_csv = os.path.join(
        MODEL_ROOT,
        "strong_fold_disagreements.csv",
    )
    strong_disagreements_dataframe.to_csv(disagreement_csv, index=False)

    disagreement_excel = os.path.join(
        MODEL_ROOT,
        "fold_disagreement_audit.xlsx",
    )
    export_disagreement_workbook(
        all_predictions=all_predictions_dataframe,
        strong_disagreements=strong_disagreements_dataframe,
        output_path=disagreement_excel,
    )

    aggregate_columns = [
        column for column in seed_dataframe.columns
        if column not in {"training_seed", "duration_seconds"}
    ]
    aggregates = _aggregate_columns(seed_dataframe, aggregate_columns)

    probability_matrix = np.stack(seed_probabilities, axis=0)
    default_bootstrap = bootstrap_mean_metrics_across_systems(
        labels=test_labels,
        probability_matrix=probability_matrix,
        thresholds=np.repeat(0.5, len(seed_results)),
        metric_names=("f1", "precision", "recall", "accuracy", "pr_auc"),
        n_resamples=bootstrap_resamples,
        random_seed=bootstrap_seed,
    )
    optimized_bootstrap = bootstrap_mean_metrics_across_systems(
        labels=test_labels,
        probability_matrix=probability_matrix,
        thresholds=optimized_thresholds,
        metric_names=("f1", "precision", "recall", "accuracy", "pr_auc"),
        n_resamples=bootstrap_resamples,
        random_seed=bootstrap_seed,
    )
    bootstrap_summary = {
        "scope": (
            "Incertitude d'échantillonnage sur les observations du test. "
            "Les mêmes indices bootstrap sont utilisés pour toutes les seeds."
        ),
        "ensemble_threshold_0_5": default_bootstrap,
        "ensemble_threshold_optimized_on_oof": optimized_bootstrap,
    }
    bootstrap_json = os.path.join(MODEL_ROOT, "bootstrap_test_confidence_intervals.json")
    save_json(bootstrap_json, bootstrap_summary)
    bootstrap_rows = []
    for threshold_name, metric_results in bootstrap_summary.items():
        if not isinstance(metric_results, dict):
            continue
        for metric_name, values in metric_results.items():
            bootstrap_rows.append({
                "evaluation": threshold_name,
                "metric": metric_name,
                **values,
            })
    bootstrap_csv = os.path.join(MODEL_ROOT, "bootstrap_test_confidence_intervals.csv")
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_csv, index=False)

    global_probabilities = np.mean(probability_matrix, axis=0)
    global_metrics = metric_bundle(test_labels, global_probabilities, 0.5)
    global_predictions_path = os.path.join(
        MODEL_ROOT, "all_seeds_ensemble_test_predictions.csv"
    )
    pd.DataFrame({
        "row_index": np.arange(test_labels.size, dtype=int),
        "label": test_labels,
        "probability_class_1": global_probabilities,
        "prediction_threshold_0_5": (global_probabilities >= 0.5).astype(int),
    }).to_csv(global_predictions_path, index=False)
    global_curve_path = os.path.join(
        MODEL_ROOT, "all_seeds_ensemble_test_precision_recall_curve.png"
    )
    plot_precision_recall_curve(test_labels, global_probabilities, global_curve_path)

    if deployment_strategy == "fold_ensemble":
        deployment_description = (
            "Déploiement déclaré : ensemble des cinq modèles de fold, avec "
            "moyenne des probabilités. Les métriques ensemble sont donc les "
            "estimations directement pertinentes."
        )
    else:
        deployment_description = (
            "Déploiement déclaré : modèle unique. La moyenne des modèles de "
            "fold est un proxy expérimental ; un modèle final réentraîné sur "
            "tout train_cv devra être évalué séparément avant déploiement."
        )

    summary = {
        "protocol": GROUP_AWARE_PROTOCOL,
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_output_name": MODEL_OUTPUT_NAME,
        "training_seeds": [int(result["training_seed"]) for result in seed_results],
        "split_seed": int(seed_results[0]["split_seed"]),
        "fold_count": int(seed_results[0]["fold_count"]),
        "error_bar_scope": (
            "Les écarts-types rapportés sont uniquement inter-seed à split fixe ; "
            "ils ne capturent pas la variance de partition."
        ),
        "bootstrap_scope": bootstrap_summary["scope"],
        "declared_deployment_strategy": deployment_strategy,
        "deployment_interpretation": deployment_description,
        "primary_reporting": {
            "individual_fold_models": (
                "Moyenne, pour chaque seed, des métriques des cinq modèles de "
                "fold pris individuellement."
            ),
            "fold_ensemble": (
                "Métriques après moyenne des probabilités des cinq folds."
            ),
        },
        "aggregate_test_metrics": aggregates,
        "bootstrap_confidence_intervals": bootstrap_summary,
        "seed_results": list(seed_results),
        "secondary_all_seed_fold_ensemble": {
            "description": (
                "Moyenne des probabilités des ensembles de toutes les seeds ; "
                "résultat descriptif secondaire."
            ),
            "metrics_threshold_0_5": global_metrics,
            "predictions_path": global_predictions_path,
            "pr_curve_path": global_curve_path,
        },
        "all_seeds_test_csv": seed_csv,
        "all_fold_models_test_csv": fold_csv,
        "all_seeds_detailed_predictions_csv": all_predictions_csv,
        "strong_fold_disagreements_csv": disagreement_csv,
        "fold_disagreement_audit_excel": disagreement_excel,
        "strong_fold_disagreement_count": int(
            len(strong_disagreements_dataframe)
        ),
        "bootstrap_json": bootstrap_json,
        "bootstrap_csv": bootstrap_csv,
        "total_duration_seconds": float(total_duration),
    }
    summary_path = os.path.join(MODEL_ROOT, "test_benchmark_summary.json")
    save_json(summary_path, summary)

    print_banner(f"RÉSULTATS TEST — {MODEL_DISPLAY_NAME}")
    print_table(
        [
            "Seed", "F1 individuel moyen", "F1 ensemble", "Précision ens.",
            "Rappel ens.", "PR-AUC ens.", "Seuil OOF", "F1 ens. optimisé"
        ],
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

    print("\nPortée des incertitudes")
    print("- ± : variabilité inter-seed, avec split fixe.")
    print(
        f"- IC 95 % : bootstrap stratifié de {bootstrap_resamples} réplications "
        "sur les observations du test."
    )
    for metric_name, label in (("f1", "F1 optimisé"), ("pr_auc", "PR-AUC")):
        ci = optimized_bootstrap[metric_name]
        print(
            f"  {label}: {ci['point_estimate']:.4f} "
            f"[IC95 {ci['ci_lower']:.4f} ; {ci['ci_upper']:.4f}]"
        )
    print(f"\n{deployment_description}")
    print(f"Résumé JSON : {summary_path}")
    print(f"Table seeds : {seed_csv}")
    print(f"Annexe folds: {fold_csv}")
    print(f"Audit Excel : {disagreement_excel}")
    print(
        "Désaccords forts : "
        f"{len(strong_disagreements_dataframe)} lignes seed-note"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Évalue les modèles individuels de fold et leur ensemble, puis "
            "calcule des IC bootstrap sur le test."
        )
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, TRAINING_SEEDS)),
        help="Seeds d'entraînement séparées par des virgules.",
    )
    parser.add_argument("--folds", type=int, default=K_FOLDS)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--deployment-strategy",
        choices=["fold_ensemble", "single_model"],
        default="fold_ensemble",
        help=(
            "Brique envisagée pour le déploiement. 'single_model' est signalé "
            "comme un proxy tant qu'un modèle final n'est pas réentraîné."
        ),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    seeds = parse_seed_list(args.seeds)
    for path in (TRAIN_CV_CSV, TEST_CSV, MODEL_ROOT):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier ou dossier introuvable : {path}")

    test_metadata = pd.read_csv(TEST_CSV)
    _, test_dataset = get_tokenized_datasets(TRAIN_CV_CSV, TEST_CSV)

    if len(test_metadata) != len(test_dataset):
        raise RuntimeError(
            "Le nombre de lignes de test.csv ne correspond pas au dataset "
            "tokenisé. Vérifiez qu'aucune ligne n'est filtrée par data_loader."
        )

    print_banner("ÉVALUATION TEST — PROTOCOLE GROUP-AWARE")
    print_key_values([
        ("Modèle", MODEL_DISPLAY_NAME),
        ("Seeds", ", ".join(map(str, seeds))),
        ("Folds par seed", args.folds),
        ("Modèles individuels évalués", len(seeds) * args.folds),
        ("Bootstrap", args.bootstrap_resamples),
        ("Déploiement déclaré", args.deployment_strategy),
    ])

    seed_results: List[Dict] = []
    seed_probabilities: List[np.ndarray] = []
    reference_labels: np.ndarray | None = None
    for seed in seeds:
        result, probabilities, labels = evaluate_one_seed(
            training_seed=seed,
            fold_count=args.folds,
            test_dataset=test_dataset,
            test_metadata=test_metadata,
        )
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError("Les labels du test diffèrent entre les seeds.")
        seed_results.append(result)
        seed_probabilities.append(probabilities)

    assert reference_labels is not None
    aggregate_results(
        seed_results=seed_results,
        seed_probabilities=seed_probabilities,
        test_labels=reference_labels,
        total_duration=time.perf_counter() - start,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        deployment_strategy=args.deployment_strategy,
    )


if __name__ == "__main__":
    main(build_arg_parser().parse_args())