"""Construit le benchmark final DEREEV demandé par la data scientist.

À lancer depuis la racine du projet :

    python3 build_benchmark.py

Le script rassemble automatiquement les résultats présents dans :

    models/benchmark/<modele>/

Il produit notamment un tableau principal contenant :

- moyenne des cinq modèles de fold pris individuellement ;
- performance de l'ensemble des cinq folds ;
- moyenne ± écart-type entre seeds, à split fixe ;
- précision, rappel et PR-AUC ;
- seuil sélectionné exclusivement sur les OOF ;
- F1 test obtenu avec ce seuil OOF ;
- intervalles de confiance bootstrap à 95 % ;
- statut « benchmark complet » ou « pilote » ;
- annexes par seed et par fold ;
- hyperparamètres disponibles ;
- huit graphes comparatifs prêts pour le rapport et la présentation.

Le script lit les fichiers bootstrap existants. S'ils sont absents, il tente
de recalculer les IC à partir des fichiers seed_*/test_predictions.csv.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path("models/benchmark")

OUTPUT_COMPARISON_CSV = BENCHMARK_DIR / "benchmark_comparison.csv"
OUTPUT_DETAILS_CSV = BENCHMARK_DIR / "benchmark_comparison_detailed.csv"
OUTPUT_BY_SEED_CSV = BENCHMARK_DIR / "benchmark_results_by_seed.csv"
OUTPUT_BY_FOLD_CSV = BENCHMARK_DIR / "benchmark_results_by_fold.csv"
OUTPUT_BOOTSTRAP_CSV = BENCHMARK_DIR / "benchmark_bootstrap_ci.csv"
OUTPUT_HYPERPARAMETERS_CSV = BENCHMARK_DIR / "benchmark_hyperparameters.csv"
OUTPUT_WARNINGS_CSV = BENCHMARK_DIR / "benchmark_warnings.csv"
OUTPUT_GRAPHS_MANIFEST_CSV = BENCHMARK_DIR / "benchmark_graphs.csv"
OUTPUT_XLSX = BENCHMARK_DIR / "benchmark_comparison.xlsx"
GRAPHS_DIR = BENCHMARK_DIR / "graphs"

EXPECTED_COMPLETE_SEEDS = 5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95

DEFAULT_THRESHOLD = 0.5

METRICS = ("f1", "precision", "recall", "accuracy", "pr_auc")

CANONICAL_COLUMNS: dict[str, list[str]] = {
    # Modèles de fold pris individuellement, puis moyennés pour chaque seed.
    "individual_mean_f1_threshold_0_5": [
        "individual_mean_f1_threshold_0_5",
        "individual_f1_threshold_0_5",
        "mean_individual_f1_threshold_0_5",
    ],
    "individual_mean_precision_threshold_0_5": [
        "individual_mean_precision_threshold_0_5",
        "individual_precision_threshold_0_5",
        "mean_individual_precision_threshold_0_5",
    ],
    "individual_mean_recall_threshold_0_5": [
        "individual_mean_recall_threshold_0_5",
        "individual_recall_threshold_0_5",
        "mean_individual_recall_threshold_0_5",
    ],
    "individual_mean_pr_auc": [
        "individual_mean_pr_auc",
        "individual_pr_auc",
        "mean_individual_pr_auc",
    ],
    "individual_mean_f1_optimized": [
        "individual_mean_f1_optimized",
        "individual_f1_optimized",
        "mean_individual_f1_optimized",
    ],

    # Ensemble des cinq folds pour chaque seed.
    "ensemble_f1_threshold_0_5": [
        "ensemble_f1_threshold_0_5",
        "f1_test_threshold_0_5",
        "test_f1_threshold_0_5",
        "f1_threshold_0_5",
        "f1",
    ],
    "ensemble_precision_threshold_0_5": [
        "ensemble_precision_threshold_0_5",
        "precision_test_threshold_0_5",
        "test_precision_threshold_0_5",
        "precision_threshold_0_5",
        "precision",
    ],
    "ensemble_recall_threshold_0_5": [
        "ensemble_recall_threshold_0_5",
        "recall_test_threshold_0_5",
        "test_recall_threshold_0_5",
        "recall_threshold_0_5",
        "recall",
    ],
    "ensemble_accuracy_threshold_0_5": [
        "ensemble_accuracy_threshold_0_5",
        "accuracy_test_threshold_0_5",
        "test_accuracy_threshold_0_5",
        "accuracy_threshold_0_5",
        "accuracy",
    ],
    "ensemble_pr_auc": [
        "ensemble_pr_auc",
        "pr_auc_test",
        "test_pr_auc",
        "pr_auc",
    ],
    "optimized_threshold": [
        "optimized_threshold",
        "optimized_threshold_selected_on_oof",
        "threshold_oof",
        "best_threshold",
    ],
    "ensemble_f1_optimized": [
        "ensemble_f1_optimized",
        "f1_test_optimized",
        "test_f1_optimized",
        "f1_optimized",
    ],
    "ensemble_precision_optimized": [
        "ensemble_precision_optimized",
        "precision_test_optimized",
        "test_precision_optimized",
        "precision_optimized",
    ],
    "ensemble_recall_optimized": [
        "ensemble_recall_optimized",
        "recall_test_optimized",
        "test_recall_optimized",
        "recall_optimized",
    ],
    "ensemble_accuracy_optimized": [
        "ensemble_accuracy_optimized",
        "accuracy_test_optimized",
        "test_accuracy_optimized",
        "accuracy_optimized",
    ],
    "ensemble_log_loss": [
        "ensemble_log_loss",
        "log_loss",
        "test_log_loss",
    ],
}

FOLD_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "individual_default_f1": [
        "individual_default_f1",
        "default_f1",
        "f1_threshold_0_5",
        "f1",
    ],
    "individual_default_precision": [
        "individual_default_precision",
        "default_precision",
        "precision_threshold_0_5",
        "precision",
    ],
    "individual_default_recall": [
        "individual_default_recall",
        "default_recall",
        "recall_threshold_0_5",
        "recall",
    ],
    "individual_default_pr_auc": [
        "individual_default_pr_auc",
        "default_pr_auc",
        "pr_auc",
    ],
    "individual_optimized_f1": [
        "individual_optimized_f1",
        "optimized_f1",
        "f1_optimized",
    ],
}


# ---------------------------------------------------------------------------
# OUTILS GÉNÉRAUX
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        return content if isinstance(content, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def numeric_series(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> pd.Series:
    column = first_existing_column(dataframe, candidates)
    if column is None:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce")


def safe_mean(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def sample_std(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Écart-type inter-seed.

    Avec une seule seed, la variabilité inter-seed est non estimable :
    on renvoie donc NaN, jamais 0.
    """
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if array.size < 2:
        return float("nan")
    return float(np.std(array, ddof=1))


def safe_min(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return float(np.min(array)) if array.size else float("nan")


def safe_max(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return float(np.max(array)) if array.size else float("nan")


def is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def format_mean_std(mean: float, std: float, seed_count: int) -> str:
    if not is_finite(mean):
        return "Non disponible"
    if seed_count < 2 or not is_finite(std):
        return f"{mean:.4f} (1 seed)"
    return f"{mean:.4f} ± {std:.4f}"


def format_threshold(mean: float, std: float, seed_count: int) -> str:
    if not is_finite(mean):
        return "Non disponible"
    if seed_count < 2 or not is_finite(std):
        return f"{mean:.3f} (1 seed)"
    return f"{mean:.3f} ± {std:.3f}"


def format_ci(lower: float, upper: float) -> str:
    if not (is_finite(lower) and is_finite(upper)):
        return "Non disponible"
    return f"[{lower:.4f} ; {upper:.4f}]"


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed[_-](\d+)", str(path))
    return int(match.group(1)) if match else None


def natural_seed_sort(path: Path) -> tuple[int, str]:
    seed = seed_from_path(path)
    return (seed if seed is not None else 10**12, str(path))


def stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def flatten_mapping(
    mapping: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


# ---------------------------------------------------------------------------
# MÉTRIQUES ET BOOTSTRAP
# ---------------------------------------------------------------------------

def metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= float(threshold)).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy_score(labels, predictions)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def stratified_bootstrap_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    class_zero = np.flatnonzero(labels == 0)
    class_one = np.flatnonzero(labels == 1)

    if class_zero.size == 0 or class_one.size == 0:
        raise ValueError(
            "Le bootstrap stratifié nécessite les classes 0 et 1 dans le test."
        )

    sampled_zero = rng.choice(class_zero, size=class_zero.size, replace=True)
    sampled_one = rng.choice(class_one, size=class_one.size, replace=True)
    indices = np.concatenate([sampled_zero, sampled_one])
    rng.shuffle(indices)
    return indices


def bootstrap_mean_metrics_across_seeds(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    thresholds: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels, dtype=int)
    probability_matrix = np.asarray(probability_matrix, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if probability_matrix.ndim != 2:
        raise ValueError("La matrice de probabilités doit être de dimension 2.")
    if probability_matrix.shape[0] != thresholds.size:
        raise ValueError("Une valeur de seuil est requise pour chaque seed.")
    if probability_matrix.shape[1] != labels.size:
        raise ValueError("Le nombre de probabilités ne correspond pas au test.")

    rng = np.random.default_rng(random_seed)
    replicates = {
        metric: np.empty(n_resamples, dtype=float)
        for metric in METRICS
    }

    for iteration in range(n_resamples):
        indices = stratified_bootstrap_indices(labels, rng)
        sampled_labels = labels[indices]

        per_seed_metrics = [
            metric_bundle(
                sampled_labels,
                probability_matrix[seed_index, indices],
                thresholds[seed_index],
            )
            for seed_index in range(probability_matrix.shape[0])
        ]

        for metric in METRICS:
            replicates[metric][iteration] = float(np.mean([
                values[metric] for values in per_seed_metrics
            ]))

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    point_metrics = [
        metric_bundle(labels, probabilities, threshold)
        for probabilities, threshold in zip(probability_matrix, thresholds)
    ]

    results: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = replicates[metric]
        results[metric] = {
            "point_estimate": float(np.mean([
                bundle[metric] for bundle in point_metrics
            ])),
            "ci_lower": float(np.percentile(values, lower_percentile)),
            "ci_upper": float(np.percentile(values, upper_percentile)),
            "confidence_level": float(confidence_level),
            "n_resamples": int(n_resamples),
            "bootstrap_seed": int(random_seed),
            "method": "stratified_percentile_bootstrap_paired_across_seeds",
        }

    return results


# ---------------------------------------------------------------------------
# CHARGEMENT DES PRÉDICTIONS ET RÉSULTATS PAR FOLD
# ---------------------------------------------------------------------------

def prediction_files(model_directory: Path) -> list[Path]:
    files = sorted(
        model_directory.glob("seed_*/test_predictions.csv"),
        key=natural_seed_sort,
    )
    return files


def load_prediction_systems(
    model_directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]] | None:
    files = prediction_files(model_directory)
    if not files:
        return None

    reference_labels: np.ndarray | None = None
    probabilities: list[np.ndarray] = []
    thresholds: list[float] = []
    seeds: list[int] = []

    for path in files:
        dataframe = pd.read_csv(path)
        if "label" not in dataframe.columns:
            continue

        probability_column = first_existing_column(
            dataframe,
            ["probability_class_1", "ensemble_probability_class_1"],
        )
        if probability_column is None:
            continue

        labels = pd.to_numeric(
            dataframe["label"], errors="raise"
        ).to_numpy(dtype=int)
        seed_probabilities = pd.to_numeric(
            dataframe[probability_column], errors="raise"
        ).to_numpy(dtype=float)

        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError(
                f"Les labels test ne sont pas identiques entre les seeds de "
                f"{model_directory.name}."
            )

        threshold = DEFAULT_THRESHOLD
        if "optimized_threshold_from_oof" in dataframe.columns:
            unique_thresholds = pd.to_numeric(
                dataframe["optimized_threshold_from_oof"],
                errors="coerce",
            ).dropna().unique()
            if unique_thresholds.size:
                threshold = float(unique_thresholds[0])
        else:
            decision_path = path.parent / "decision_threshold.json"
            decision = read_json(decision_path)
            for key in (
                "best_threshold",
                "optimized_threshold",
                "optimized_threshold_selected_on_oof",
            ):
                if key in decision:
                    threshold = float(decision[key])
                    break

        seed = seed_from_path(path)
        if seed is None and "training_seed" in dataframe.columns:
            values = pd.to_numeric(
                dataframe["training_seed"], errors="coerce"
            ).dropna().unique()
            seed = int(values[0]) if values.size else len(seeds)

        probabilities.append(seed_probabilities)
        thresholds.append(threshold)
        seeds.append(int(seed if seed is not None else len(seeds)))

    if reference_labels is None or not probabilities:
        return None

    return (
        reference_labels,
        np.stack(probabilities, axis=0),
        np.asarray(thresholds, dtype=float),
        seeds,
    )


def calculate_seed_rows_from_predictions(
    model_directory: Path,
) -> pd.DataFrame:
    files = prediction_files(model_directory)
    rows: list[dict[str, Any]] = []

    for path in files:
        dataframe = pd.read_csv(path)
        if "label" not in dataframe.columns:
            continue

        probability_column = first_existing_column(
            dataframe,
            ["probability_class_1", "ensemble_probability_class_1"],
        )
        if probability_column is None:
            continue

        labels = pd.to_numeric(dataframe["label"], errors="raise").to_numpy(int)
        probabilities = pd.to_numeric(
            dataframe[probability_column], errors="raise"
        ).to_numpy(float)

        threshold = DEFAULT_THRESHOLD
        if "optimized_threshold_from_oof" in dataframe.columns:
            values = pd.to_numeric(
                dataframe["optimized_threshold_from_oof"],
                errors="coerce",
            ).dropna().unique()
            if values.size:
                threshold = float(values[0])

        default_metrics = metric_bundle(labels, probabilities, DEFAULT_THRESHOLD)
        optimized_metrics = metric_bundle(labels, probabilities, threshold)

        seed = seed_from_path(path)
        rows.append({
            "training_seed": int(seed if seed is not None else len(rows)),
            "ensemble_f1_threshold_0_5": default_metrics["f1"],
            "ensemble_precision_threshold_0_5": default_metrics["precision"],
            "ensemble_recall_threshold_0_5": default_metrics["recall"],
            "ensemble_accuracy_threshold_0_5": default_metrics["accuracy"],
            "ensemble_pr_auc": default_metrics["pr_auc"],
            "optimized_threshold": threshold,
            "ensemble_f1_optimized": optimized_metrics["f1"],
            "ensemble_precision_optimized": optimized_metrics["precision"],
            "ensemble_recall_optimized": optimized_metrics["recall"],
            "ensemble_accuracy_optimized": optimized_metrics["accuracy"],
        })

    return pd.DataFrame(rows)


def load_fold_results(model_directory: Path) -> pd.DataFrame:
    global_file = model_directory / "all_fold_models_test_results.csv"
    if global_file.exists():
        return pd.read_csv(global_file)

    frames: list[pd.DataFrame] = []
    for path in sorted(
        model_directory.glob("seed_*/individual_fold_test_results.csv"),
        key=natural_seed_sort,
    ):
        dataframe = pd.read_csv(path)
        if "training_seed" not in dataframe.columns:
            seed = seed_from_path(path)
            dataframe.insert(
                0,
                "training_seed",
                int(seed if seed is not None else len(frames)),
            )
        frames.append(dataframe)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def individual_metrics_by_seed(
    fold_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if fold_dataframe.empty or "training_seed" not in fold_dataframe.columns:
        return pd.DataFrame()

    result = pd.DataFrame({
        "training_seed": pd.to_numeric(
            fold_dataframe["training_seed"], errors="coerce"
        )
    })

    mapping = {
        "individual_mean_f1_threshold_0_5": "individual_default_f1",
        "individual_mean_precision_threshold_0_5": "individual_default_precision",
        "individual_mean_recall_threshold_0_5": "individual_default_recall",
        "individual_mean_pr_auc": "individual_default_pr_auc",
        "individual_mean_f1_optimized": "individual_optimized_f1",
    }

    normalized = pd.DataFrame({
        "training_seed": result["training_seed"],
    })

    for target_column, fold_key in mapping.items():
        normalized[target_column] = numeric_series(
            fold_dataframe,
            FOLD_COLUMN_CANDIDATES[fold_key],
        )

    normalized = normalized.dropna(subset=["training_seed"])
    if normalized.empty:
        return pd.DataFrame()

    normalized["training_seed"] = normalized["training_seed"].astype(int)
    metric_columns = [
        column for column in normalized.columns
        if column != "training_seed"
    ]
    return (
        normalized
        .groupby("training_seed", as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )


# ---------------------------------------------------------------------------
# NORMALISATION DES SORTIES DE CHAQUE MODÈLE
# ---------------------------------------------------------------------------

def normalize_seed_results(
    model_directory: Path,
    result_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(result_file)

    if "training_seed" in raw.columns:
        seeds = pd.to_numeric(raw["training_seed"], errors="coerce")
    elif "seed" in raw.columns:
        seeds = pd.to_numeric(raw["seed"], errors="coerce")
    else:
        seeds = pd.Series(np.arange(len(raw)), index=raw.index, dtype=int)

    normalized = pd.DataFrame({"training_seed": seeds})

    for canonical_name, candidates in CANONICAL_COLUMNS.items():
        normalized[canonical_name] = numeric_series(raw, candidates)

    # Complément à partir des prédictions si certaines métriques ensemble sont absentes.
    prediction_rows = calculate_seed_rows_from_predictions(model_directory)
    if not prediction_rows.empty:
        normalized = normalized.merge(
            prediction_rows,
            on="training_seed",
            how="outer",
            suffixes=("", "_from_predictions"),
        )
        for column in prediction_rows.columns:
            if column == "training_seed":
                continue
            fallback = f"{column}_from_predictions"
            if fallback in normalized.columns:
                normalized[column] = normalized[column].fillna(
                    normalized[fallback]
                )
                normalized = normalized.drop(columns=[fallback])

    # Complément des métriques individuelles depuis les fichiers par fold.
    folds = load_fold_results(model_directory)
    per_seed_individual = individual_metrics_by_seed(folds)
    if not per_seed_individual.empty:
        normalized = normalized.merge(
            per_seed_individual,
            on="training_seed",
            how="outer",
            suffixes=("", "_from_folds"),
        )
        for column in per_seed_individual.columns:
            if column == "training_seed":
                continue
            fallback = f"{column}_from_folds"
            if fallback in normalized.columns:
                normalized[column] = normalized[column].fillna(
                    normalized[fallback]
                )
                normalized = normalized.drop(columns=[fallback])

    normalized = normalized.dropna(
        subset=["training_seed"],
    ).copy()
    normalized["training_seed"] = normalized["training_seed"].astype(int)
    normalized = normalized.sort_values("training_seed").reset_index(drop=True)

    return normalized, folds


def load_bootstrap(
    model_directory: Path,
    test_summary: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    bootstrap_path = (
        model_directory / "bootstrap_test_confidence_intervals.json"
    )
    bootstrap = read_json(bootstrap_path)

    if bootstrap:
        return bootstrap, str(bootstrap_path)

    embedded = test_summary.get("bootstrap_confidence_intervals")
    if isinstance(embedded, dict) and embedded:
        return embedded, "test_benchmark_summary.json"

    systems = load_prediction_systems(model_directory)
    if systems is None:
        return {}, ""

    labels, probability_matrix, optimized_thresholds, _ = systems

    default_bootstrap = bootstrap_mean_metrics_across_seeds(
        labels,
        probability_matrix,
        np.repeat(DEFAULT_THRESHOLD, probability_matrix.shape[0]),
    )
    optimized_bootstrap = bootstrap_mean_metrics_across_seeds(
        labels,
        probability_matrix,
        optimized_thresholds,
    )

    bootstrap = {
        "scope": (
            "Incertitude d'échantillonnage sur les observations du test, "
            "recalculée par build_benchmark.py."
        ),
        "ensemble_threshold_0_5": default_bootstrap,
        "ensemble_threshold_optimized_on_oof": optimized_bootstrap,
    }

    generated_path = (
        model_directory
        / "bootstrap_test_confidence_intervals_from_build.json"
    )
    generated_path.write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bootstrap, str(generated_path)


def bootstrap_value(
    bootstrap: Mapping[str, Any],
    evaluation: str,
    metric: str,
    key: str,
) -> float:
    evaluation_values = bootstrap.get(evaluation, {})
    if not isinstance(evaluation_values, Mapping):
        return float("nan")
    metric_values = evaluation_values.get(metric, {})
    if not isinstance(metric_values, Mapping):
        return float("nan")
    value = metric_values.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def bootstrap_rows_for_model(
    model_name: str,
    model_directory_name: str,
    bootstrap: Mapping[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for evaluation in (
        "ensemble_threshold_0_5",
        "ensemble_threshold_optimized_on_oof",
    ):
        metric_results = bootstrap.get(evaluation, {})
        if not isinstance(metric_results, Mapping):
            continue

        for metric, values in metric_results.items():
            if not isinstance(values, Mapping):
                continue
            rows.append({
                "model": model_name,
                "model_directory": model_directory_name,
                "evaluation": evaluation,
                "metric": metric,
                "point_estimate": values.get("point_estimate"),
                "ci_lower": values.get("ci_lower"),
                "ci_upper": values.get("ci_upper"),
                "confidence_level": values.get("confidence_level"),
                "n_resamples": values.get("n_resamples"),
                "bootstrap_seed": values.get("bootstrap_seed"),
                "method": values.get("method"),
                "source": source,
            })

    return rows


# ---------------------------------------------------------------------------
# HYPERPARAMÈTRES
# ---------------------------------------------------------------------------

def collect_hyperparameters(
    model_name: str,
    model_directory: Path,
    test_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
) -> list[dict[str, str]]:
    configurations: list[tuple[str, Mapping[str, Any]]] = []

    run_configuration = read_json(model_directory / "run_configuration.json")
    if run_configuration:
        configurations.append(("run_configuration.json", run_configuration))

    for source_name, summary in (
        ("test_benchmark_summary.json", test_summary),
        ("benchmark_summary.json", benchmark_summary),
    ):
        training_configuration = summary.get("training_configuration")
        if isinstance(training_configuration, Mapping):
            configurations.append((source_name, training_configuration))

    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for source_name, configuration in configurations:
        for parameter, value in flatten_mapping(configuration).items():
            if parameter in seen:
                continue
            seen.add(parameter)
            rows.append({
                "model": model_name,
                "parameter": parameter,
                "value": stringify(value),
                "source": source_name,
            })

    return rows



# ---------------------------------------------------------------------------
# GRAPHIQUES DU BENCHMARK
# ---------------------------------------------------------------------------

def save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path


def finite_rows(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    missing = [
        column for column in required_columns
        if column not in dataframe.columns
    ]
    if missing:
        return pd.DataFrame()

    mask = np.ones(len(dataframe), dtype=bool)
    for column in required_columns:
        values = pd.to_numeric(dataframe[column], errors="coerce")
        mask &= np.isfinite(values.to_numpy(float))

    return dataframe.loc[mask].copy()


def plot_metric_bar(
    details: pd.DataFrame,
    *,
    mean_column: str,
    std_column: str,
    title: str,
    xlabel: str,
    filename: str,
) -> Path | None:
    data = finite_rows(details, [mean_column])
    if data.empty:
        return None

    data = data.sort_values(mean_column, ascending=True)
    means = pd.to_numeric(data[mean_column], errors="coerce").to_numpy(float)
    if std_column in data.columns:
        errors = pd.to_numeric(
            data[std_column], errors="coerce"
        ).fillna(0.0).to_numpy(float)
    else:
        errors = np.zeros(len(data), dtype=float)

    labels = data["model"].astype(str).tolist()
    y_positions = np.arange(len(data))

    figure_height = max(4.8, 0.65 * len(data) + 2.0)
    plt.figure(figsize=(11, figure_height))
    bars = plt.barh(
        y_positions,
        means,
        xerr=errors,
        capsize=4,
    )
    plt.yticks(y_positions, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xlim(0.0, 1.02)
    plt.grid(axis="x", alpha=0.25)

    seed_counts = data["seed_count"].to_numpy(int)
    for bar, mean, std, seed_count in zip(
        bars,
        means,
        errors,
        seed_counts,
    ):
        if seed_count < 2:
            annotation = f"{mean:.4f} — pilote"
        else:
            annotation = f"{mean:.4f} ± {std:.4f}"
        plt.text(
            min(mean + max(std, 0.006) + 0.008, 0.985),
            bar.get_y() + bar.get_height() / 2,
            annotation,
            va="center",
            fontsize=9,
        )

    plt.figtext(
        0.5,
        0.01,
        "Barres d’erreur : variabilité inter-seed avec split fixe.",
        ha="center",
        fontsize=9,
    )
    return save_figure(GRAPHS_DIR / filename)


def plot_individual_vs_ensemble_f1(
    details: pd.DataFrame,
) -> Path | None:
    data = finite_rows(
        details,
        ["individual_f1_mean", "ensemble_f1_mean"],
    )
    if data.empty:
        return None

    data = data.sort_values(
        "ensemble_f1_mean",
        ascending=False,
    )
    labels = data["model"].astype(str).tolist()
    positions = np.arange(len(data))
    width = 0.36

    individual = data["individual_f1_mean"].to_numpy(float)
    ensemble = data["ensemble_f1_mean"].to_numpy(float)
    individual_std = pd.to_numeric(
        data["individual_f1_std"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    ensemble_std = pd.to_numeric(
        data["ensemble_f1_std"], errors="coerce"
    ).fillna(0.0).to_numpy(float)

    plt.figure(figsize=(max(10, len(data) * 1.45), 6.5))
    plt.bar(
        positions - width / 2,
        individual,
        width,
        yerr=individual_std,
        capsize=3,
        label="Moyenne des modèles de fold individuels",
    )
    plt.bar(
        positions + width / 2,
        ensemble,
        width,
        yerr=ensemble_std,
        capsize=3,
        label="Ensemble des cinq folds",
    )
    plt.xticks(positions, labels, rotation=25, ha="right")
    plt.ylabel("F1 positif sur le test")
    plt.ylim(0.0, 1.02)
    plt.title("Apport du fold ensembling")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    return save_figure(
        GRAPHS_DIR / "02_f1_individuel_vs_ensemble.png"
    )


def plot_precision_recall_tradeoff(
    details: pd.DataFrame,
) -> Path | None:
    data = finite_rows(
        details,
        ["ensemble_recall_mean", "ensemble_precision_mean"],
    )
    if data.empty:
        return None

    plt.figure(figsize=(9.5, 7.2))
    plt.scatter(
        data["ensemble_recall_mean"],
        data["ensemble_precision_mean"],
        s=90,
    )

    for _, row in data.iterrows():
        suffix = " (pilote)" if int(row["seed_count"]) < 2 else ""
        plt.annotate(
            f"{row['model']}{suffix}",
            (
                float(row["ensemble_recall_mean"]),
                float(row["ensemble_precision_mean"]),
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xlabel("Rappel de la classe positive")
    plt.ylabel("Précision de la classe positive")
    plt.title("Compromis précision–rappel à seuil 0,5")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)

    return save_figure(
        GRAPHS_DIR / "04_compromis_precision_rappel.png"
    )


def plot_threshold_effect(
    details: pd.DataFrame,
) -> Path | None:
    data = finite_rows(
        details,
        ["ensemble_f1_mean", "ensemble_f1_optimized_mean"],
    )
    if data.empty:
        return None

    data = data.sort_values(
        "ensemble_f1_mean",
        ascending=False,
    )
    labels = data["model"].astype(str).tolist()
    positions = np.arange(len(data))
    width = 0.36

    default_f1 = data["ensemble_f1_mean"].to_numpy(float)
    optimized_f1 = data["ensemble_f1_optimized_mean"].to_numpy(float)
    default_std = pd.to_numeric(
        data["ensemble_f1_std"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    optimized_std = pd.to_numeric(
        data["ensemble_f1_optimized_std"], errors="coerce"
    ).fillna(0.0).to_numpy(float)

    plt.figure(figsize=(max(10, len(data) * 1.45), 6.5))
    plt.bar(
        positions - width / 2,
        default_f1,
        width,
        yerr=default_std,
        capsize=3,
        label="Seuil standard 0,5",
    )
    plt.bar(
        positions + width / 2,
        optimized_f1,
        width,
        yerr=optimized_std,
        capsize=3,
        label="Seuil sélectionné sur les OOF",
    )
    plt.xticks(positions, labels, rotation=25, ha="right")
    plt.ylabel("F1 positif sur le test")
    plt.ylim(0.0, 1.02)
    plt.title("Effet de l’optimisation du seuil OOF")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    return save_figure(
        GRAPHS_DIR / "05_effet_seuil_oof.png"
    )


def plot_bootstrap_f1_forest(
    details: pd.DataFrame,
) -> Path | None:
    data = finite_rows(
        details,
        [
            "bootstrap_f1_default_point",
            "bootstrap_f1_default_ci_lower",
            "bootstrap_f1_default_ci_upper",
        ],
    )
    if data.empty:
        return None

    data = data.sort_values(
        "bootstrap_f1_default_point",
        ascending=True,
    )
    points = data["bootstrap_f1_default_point"].to_numpy(float)
    lower = data["bootstrap_f1_default_ci_lower"].to_numpy(float)
    upper = data["bootstrap_f1_default_ci_upper"].to_numpy(float)
    lower_errors = np.maximum(points - lower, 0.0)
    upper_errors = np.maximum(upper - points, 0.0)
    labels = data["model"].astype(str).tolist()
    positions = np.arange(len(data))

    figure_height = max(5.0, 0.68 * len(data) + 2.0)
    plt.figure(figsize=(11, figure_height))
    plt.errorbar(
        points,
        positions,
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        capsize=5,
    )
    plt.yticks(positions, labels)
    plt.xlabel("F1 positif sur le test")
    plt.xlim(0.0, 1.02)
    plt.title("F1 et intervalles de confiance bootstrap à 95 %")
    plt.grid(axis="x", alpha=0.25)

    for point, position, ci_lower, ci_upper in zip(
        points,
        positions,
        lower,
        upper,
    ):
        plt.text(
            min(ci_upper + 0.012, 0.985),
            position,
            f"{point:.4f} [{ci_lower:.4f} ; {ci_upper:.4f}]",
            va="center",
            fontsize=8.5,
        )

    return save_figure(
        GRAPHS_DIR / "06_f1_intervalles_bootstrap_95.png"
    )


def plot_f1_seed_variability(
    all_seeds: pd.DataFrame,
) -> Path | None:
    required = {
        "model",
        "ensemble_f1_threshold_0_5",
    }
    if all_seeds.empty or not required.issubset(all_seeds.columns):
        return None

    groups: list[np.ndarray] = []
    labels: list[str] = []

    for model_name, frame in all_seeds.groupby("model", sort=False):
        values = pd.to_numeric(
            frame["ensemble_f1_threshold_0_5"],
            errors="coerce",
        ).dropna().to_numpy(float)
        if values.size == 0:
            continue
        groups.append(values)
        labels.append(str(model_name))

    if not groups:
        return None

    plt.figure(figsize=(max(10, len(groups) * 1.45), 6.5))
    plt.boxplot(
        groups,
        labels=labels,
        showmeans=True,
    )

    rng = np.random.default_rng(2026)
    for position, values in enumerate(groups, start=1):
        jitter = rng.normal(0.0, 0.035, size=len(values))
        plt.scatter(
            np.repeat(position, len(values)) + jitter,
            values,
            s=36,
            zorder=3,
        )

    plt.xticks(rotation=25, ha="right")
    plt.ylabel("F1 ensemble sur le test")
    plt.ylim(0.0, 1.02)
    plt.title("Variabilité des résultats entre seeds")
    plt.grid(axis="y", alpha=0.25)
    plt.figtext(
        0.5,
        0.01,
        "Chaque point représente une seed ; la partition des données reste fixe.",
        ha="center",
        fontsize=9,
    )

    return save_figure(
        GRAPHS_DIR / "07_variabilite_f1_par_seed.png"
    )


def mean_pr_curve_for_model(
    model_directory: Path,
    recall_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    systems = load_prediction_systems(model_directory)
    if systems is None:
        return None

    labels, probability_matrix, _, _ = systems
    interpolated_precisions: list[np.ndarray] = []

    for probabilities in probability_matrix:
        precision, recall, _ = precision_recall_curve(
            labels,
            probabilities,
        )

        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]

        # Une même valeur de rappel peut apparaître plusieurs fois.
        curve_frame = pd.DataFrame({
            "recall": recall_sorted,
            "precision": precision_sorted,
        })
        curve_frame = (
            curve_frame
            .groupby("recall", as_index=False)["precision"]
            .max()
            .sort_values("recall")
        )

        interpolated = np.interp(
            recall_grid,
            curve_frame["recall"].to_numpy(float),
            curve_frame["precision"].to_numpy(float),
        )
        interpolated_precisions.append(interpolated)

    matrix = np.stack(interpolated_precisions, axis=0)
    mean_precision = np.mean(matrix, axis=0)
    std_precision = (
        np.std(matrix, axis=0, ddof=1)
        if matrix.shape[0] > 1
        else np.zeros(matrix.shape[1], dtype=float)
    )
    prevalence = float(np.mean(labels))
    return (
        mean_precision,
        std_precision,
        prevalence,
        int(matrix.shape[0]),
    )


def plot_top_two_mean_pr_curves(
    details: pd.DataFrame,
) -> Path | None:
    candidates = details.loc[
        details["status"].eq("Benchmark complet")
        & pd.to_numeric(
            details["ensemble_pr_auc_mean"],
            errors="coerce",
        ).notna()
    ].copy()

    if len(candidates) < 2:
        candidates = details.loc[
            pd.to_numeric(
                details["ensemble_pr_auc_mean"],
                errors="coerce",
            ).notna()
        ].copy()

    if len(candidates) < 2:
        return None

    candidates = candidates.sort_values(
        ["ensemble_pr_auc_mean", "ensemble_f1_mean"],
        ascending=[False, False],
    ).head(2)

    recall_grid = np.linspace(0.0, 1.0, 501)
    plt.figure(figsize=(9.5, 7.2))
    prevalence_values: list[float] = []
    plotted = 0

    for _, row in candidates.iterrows():
        model_directory = BENCHMARK_DIR / str(row["model_directory"])
        curve = mean_pr_curve_for_model(
            model_directory,
            recall_grid,
        )
        if curve is None:
            continue

        mean_precision, std_precision, prevalence, seed_count = curve
        prevalence_values.append(prevalence)

        label = (
            f"{row['model']} — PR-AUC "
            f"{float(row['ensemble_pr_auc_mean']):.4f}"
        )
        if seed_count < 2:
            label += " (1 seed)"

        line, = plt.plot(
            recall_grid,
            mean_precision,
            linewidth=2,
            label=label,
        )
        if seed_count > 1:
            lower = np.clip(
                mean_precision - std_precision,
                0.0,
                1.0,
            )
            upper = np.clip(
                mean_precision + std_precision,
                0.0,
                1.0,
            )
            plt.fill_between(
                recall_grid,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.14,
            )
        plotted += 1

    if plotted < 2:
        plt.close()
        return None

    if prevalence_values:
        baseline = float(np.mean(prevalence_values))
        plt.axhline(
            baseline,
            linestyle="--",
            label=f"Prévalence positive : {baseline:.3f}",
        )

    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.02)
    plt.title("Courbe précision–rappel moyenne des deux meilleurs modèles")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    plt.figtext(
        0.5,
        0.01,
        "La ligne est la moyenne des courbes par seed ; la zone représente ± 1 écart-type.",
        ha="center",
        fontsize=9,
    )

    return save_figure(
        GRAPHS_DIR / "08_courbes_pr_deux_meilleurs_modeles.png"
    )


def generate_benchmark_graphs(
    details: pd.DataFrame,
    all_seeds: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Évite de conserver des graphes obsolètes issus d'un ancien benchmark.
    for old_graph in GRAPHS_DIR.glob("*.png"):
        old_graph.unlink()

    manifest_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    graph_specs = [
        (
            plot_metric_bar(
                details,
                mean_column="ensemble_f1_mean",
                std_column="ensemble_f1_std",
                title="Comparaison du F1 des ensembles de folds",
                xlabel="F1 positif sur le jeu de test",
                filename="01_classement_f1_ensemble.png",
            ),
            "Classement F1",
            (
                "Compare le F1 de l'ensemble des cinq folds. Les barres "
                "d'erreur représentent la variabilité inter-seed à split fixe."
            ),
        ),
        (
            plot_individual_vs_ensemble_f1(details),
            "Modèles individuels contre ensemble",
            (
                "Montre le gain ou la perte obtenue en moyennant les "
                "probabilités des cinq modèles de fold."
            ),
        ),
        (
            plot_metric_bar(
                details,
                mean_column="ensemble_pr_auc_mean",
                std_column="ensemble_pr_auc_std",
                title="Comparaison de la PR-AUC",
                xlabel="PR-AUC sur le jeu de test",
                filename="03_classement_pr_auc.png",
            ),
            "Classement PR-AUC",
            (
                "Compare la qualité de classement indépendamment d'un seuil "
                "de décision particulier."
            ),
        ),
        (
            plot_precision_recall_tradeoff(details),
            "Compromis précision–rappel",
            (
                "Visualise le compromis décisionnel entre faux positifs et "
                "faux négatifs au seuil 0,5."
            ),
        ),
        (
            plot_threshold_effect(details),
            "Effet du seuil OOF",
            (
                "Compare le F1 au seuil standard 0,5 avec le seuil sélectionné "
                "uniquement sur les prédictions OOF."
            ),
        ),
        (
            plot_bootstrap_f1_forest(details),
            "IC bootstrap du F1",
            (
                "Présente l'incertitude d'échantillonnage sur le jeu de test "
                "au moyen des intervalles bootstrap à 95 %."
            ),
        ),
        (
            plot_f1_seed_variability(all_seeds),
            "Variabilité inter-seed",
            (
                "Affiche chaque run et permet de distinguer stabilité réelle, "
                "variabilité inter-seed et résultats pilotes."
            ),
        ),
        (
            plot_top_two_mean_pr_curves(details),
            "Courbes PR des deux meilleurs modèles",
            (
                "Répond à la demande de comparaison des deux meilleurs modèles "
                "avec une courbe précision–rappel moyenne par seed."
            ),
        ),
    ]

    for graph_path, title, interpretation in graph_specs:
        if graph_path is None:
            warnings.append(
                f"Graphe non créé faute de données suffisantes : {title}."
            )
            continue

        manifest_rows.append({
            "graphe": title,
            "fichier": str(graph_path),
            "interprétation": interpretation,
        })

    return pd.DataFrame(manifest_rows), warnings


# ---------------------------------------------------------------------------
# CONSTRUCTION DU BENCHMARK
# ---------------------------------------------------------------------------

def summarize_column(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, float]:
    values = (
        dataframe[column]
        if column in dataframe.columns
        else pd.Series(dtype=float)
    )
    return {
        "mean": safe_mean(values),
        "std": sample_std(values),
        "min": safe_min(values),
        "max": safe_max(values),
    }


def model_metadata(
    model_directory: Path,
    test_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
) -> dict[str, Any]:
    model_name = (
        test_summary.get("model_display_name")
        or benchmark_summary.get("model_display_name")
        or model_directory.name
    )

    fold_count = (
        test_summary.get("fold_count")
        or benchmark_summary.get("fold_count")
    )
    split_seed = (
        test_summary.get("split_seed")
        or benchmark_summary.get("split_seed")
    )
    protocol = (
        test_summary.get("protocol")
        or benchmark_summary.get("protocol")
        or ""
    )
    deployment = test_summary.get(
        "declared_deployment_strategy",
        "non renseignée",
    )

    return {
        "model_name": str(model_name),
        "fold_count": fold_count,
        "split_seed": split_seed,
        "protocol": protocol,
        "deployment": deployment,
    }


def build_status(
    seed_count: int,
    has_individual: bool,
    has_bootstrap_f1: bool,
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if seed_count < EXPECTED_COMPLETE_SEEDS:
        status = f"Pilote / incomplet ({seed_count} seed"
        status += "s)" if seed_count != 1 else ")"
        warnings.append(
            f"Seulement {seed_count} seed(s) : variabilité inter-seed non "
            "estimable ou protocole incomplet."
        )
    else:
        status = "Benchmark complet"

    if not has_individual:
        warnings.append(
            "Résultat moyen des modèles individuels absent."
        )
    if not has_bootstrap_f1:
        warnings.append(
            "IC bootstrap 95 % du F1 ensemble absent."
        )

    if warnings and seed_count >= EXPECTED_COMPLETE_SEEDS:
        status = "Incomplet : sortie(s) manquante(s)"

    return status, warnings


def style_excel_workbook(path: Path) -> None:
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.formatting.rule import ColorScaleRule
    except ImportError:
        return

    workbook = load_workbook(path)

    header_fill = PatternFill("solid", fgColor="17365D")
    secondary_fill = PatternFill("solid", fgColor="D9EAF7")
    pilot_fill = PatternFill("solid", fgColor="FFF2CC")
    incomplete_fill = PatternFill("solid", fgColor="FCE4D6")
    complete_fill = PatternFill("solid", fgColor="E2F0D9")
    white_font = Font(color="FFFFFF", bold=True)

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = white_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        for column_cells in worksheet.columns:
            values = [
                "" if cell.value is None else str(cell.value)
                for cell in column_cells[:200]
            ]
            max_length = max([len(value) for value in values] + [10])
            width = min(max(max_length + 2, 11), 42)
            worksheet.column_dimensions[
                column_cells[0].column_letter
            ].width = width

        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )

        if worksheet.title == "Comparaison":
            worksheet.row_dimensions[1].height = 42

            headers = {
                cell.value: cell.column
                for cell in worksheet[1]
            }

            status_column = headers.get("Statut")
            if status_column:
                for row_index in range(2, worksheet.max_row + 1):
                    cell = worksheet.cell(row_index, status_column)
                    text = str(cell.value or "")
                    if text.startswith("Benchmark complet"):
                        cell.fill = complete_fill
                    elif text.startswith("Pilote"):
                        cell.fill = pilot_fill
                    else:
                        cell.fill = incomplete_fill

            f1_column = headers.get("F1 ensemble test @0,5")
            if f1_column and worksheet.max_row >= 3:
                letter = worksheet.cell(1, f1_column).column_letter
                worksheet.conditional_formatting.add(
                    f"{letter}2:{letter}{worksheet.max_row}",
                    ColorScaleRule(
                        start_type="min",
                        start_color="F8696B",
                        mid_type="percentile",
                        mid_value=50,
                        mid_color="FFEB84",
                        end_type="max",
                        end_color="63BE7B",
                    ),
                )

        if worksheet.title == "Notes méthodologiques":
            for cell in worksheet["A"]:
                cell.fill = secondary_fill
                cell.font = Font(bold=True)
            worksheet.column_dimensions["A"].width = 34
            worksheet.column_dimensions["B"].width = 105

    workbook.save(path)


def main() -> None:
    if not BENCHMARK_DIR.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {BENCHMARK_DIR.resolve()}"
        )

    result_files = sorted(
        BENCHMARK_DIR.glob("*/all_seeds_test_results.csv")
    )

    if not result_files:
        raise FileNotFoundError(
            "Aucun fichier all_seeds_test_results.csv trouvé dans "
            f"{BENCHMARK_DIR.resolve()}"
        )

    detail_rows: list[dict[str, Any]] = []
    seed_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    all_bootstrap_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, str]] = []
    warning_rows: list[dict[str, str]] = []

    for result_file in result_files:
        model_directory = result_file.parent

        test_summary = read_json(
            model_directory / "test_benchmark_summary.json"
        )
        benchmark_summary = read_json(
            model_directory / "benchmark_summary.json"
        )
        metadata = model_metadata(
            model_directory,
            test_summary,
            benchmark_summary,
        )
        model_name = metadata["model_name"]

        seed_results, fold_results = normalize_seed_results(
            model_directory,
            result_file,
        )

        if seed_results.empty:
            warning_rows.append({
                "model": model_name,
                "warning": "Fichier de résultats vide ou inexploitable.",
            })
            continue

        seed_count = int(seed_results["training_seed"].nunique())

        summaries = {
            column: summarize_column(seed_results, column)
            for column in CANONICAL_COLUMNS
        }

        bootstrap, bootstrap_source = load_bootstrap(
            model_directory,
            test_summary,
        )
        all_bootstrap_rows.extend(
            bootstrap_rows_for_model(
                model_name,
                model_directory.name,
                bootstrap,
                bootstrap_source,
            )
        )

        bootstrap_f1_default = {
            key: bootstrap_value(
                bootstrap,
                "ensemble_threshold_0_5",
                "f1",
                key,
            )
            for key in ("point_estimate", "ci_lower", "ci_upper")
        }
        bootstrap_precision_default = {
            key: bootstrap_value(
                bootstrap,
                "ensemble_threshold_0_5",
                "precision",
                key,
            )
            for key in ("point_estimate", "ci_lower", "ci_upper")
        }
        bootstrap_recall_default = {
            key: bootstrap_value(
                bootstrap,
                "ensemble_threshold_0_5",
                "recall",
                key,
            )
            for key in ("point_estimate", "ci_lower", "ci_upper")
        }
        bootstrap_pr_auc_default = {
            key: bootstrap_value(
                bootstrap,
                "ensemble_threshold_0_5",
                "pr_auc",
                key,
            )
            for key in ("point_estimate", "ci_lower", "ci_upper")
        }
        bootstrap_f1_optimized = {
            key: bootstrap_value(
                bootstrap,
                "ensemble_threshold_optimized_on_oof",
                "f1",
                key,
            )
            for key in ("point_estimate", "ci_lower", "ci_upper")
        }

        has_individual = is_finite(
            summaries["individual_mean_f1_threshold_0_5"]["mean"]
        )
        has_bootstrap_f1 = (
            is_finite(bootstrap_f1_default["ci_lower"])
            and is_finite(bootstrap_f1_default["ci_upper"])
        )
        status, warnings = build_status(
            seed_count,
            has_individual,
            has_bootstrap_f1,
        )

        for warning in warnings:
            warning_rows.append({
                "model": model_name,
                "warning": warning,
            })

        detail_rows.append({
            "model": model_name,
            "model_directory": model_directory.name,
            "status": status,
            "seed_count": seed_count,
            "fold_count": metadata["fold_count"],
            "split_seed": metadata["split_seed"],
            "protocol": metadata["protocol"],
            "declared_deployment_strategy": metadata["deployment"],

            "individual_f1_mean": summaries[
                "individual_mean_f1_threshold_0_5"
            ]["mean"],
            "individual_f1_std": summaries[
                "individual_mean_f1_threshold_0_5"
            ]["std"],
            "individual_f1_min": summaries[
                "individual_mean_f1_threshold_0_5"
            ]["min"],
            "individual_f1_max": summaries[
                "individual_mean_f1_threshold_0_5"
            ]["max"],

            "ensemble_f1_mean": summaries[
                "ensemble_f1_threshold_0_5"
            ]["mean"],
            "ensemble_f1_std": summaries[
                "ensemble_f1_threshold_0_5"
            ]["std"],
            "ensemble_f1_min": summaries[
                "ensemble_f1_threshold_0_5"
            ]["min"],
            "ensemble_f1_max": summaries[
                "ensemble_f1_threshold_0_5"
            ]["max"],

            "ensemble_precision_mean": summaries[
                "ensemble_precision_threshold_0_5"
            ]["mean"],
            "ensemble_precision_std": summaries[
                "ensemble_precision_threshold_0_5"
            ]["std"],

            "ensemble_recall_mean": summaries[
                "ensemble_recall_threshold_0_5"
            ]["mean"],
            "ensemble_recall_std": summaries[
                "ensemble_recall_threshold_0_5"
            ]["std"],

            "ensemble_accuracy_mean": summaries[
                "ensemble_accuracy_threshold_0_5"
            ]["mean"],
            "ensemble_accuracy_std": summaries[
                "ensemble_accuracy_threshold_0_5"
            ]["std"],

            "ensemble_pr_auc_mean": summaries[
                "ensemble_pr_auc"
            ]["mean"],
            "ensemble_pr_auc_std": summaries[
                "ensemble_pr_auc"
            ]["std"],

            "optimized_threshold_mean": summaries[
                "optimized_threshold"
            ]["mean"],
            "optimized_threshold_std": summaries[
                "optimized_threshold"
            ]["std"],

            "ensemble_f1_optimized_mean": summaries[
                "ensemble_f1_optimized"
            ]["mean"],
            "ensemble_f1_optimized_std": summaries[
                "ensemble_f1_optimized"
            ]["std"],

            "bootstrap_f1_default_point": bootstrap_f1_default[
                "point_estimate"
            ],
            "bootstrap_f1_default_ci_lower": bootstrap_f1_default[
                "ci_lower"
            ],
            "bootstrap_f1_default_ci_upper": bootstrap_f1_default[
                "ci_upper"
            ],

            "bootstrap_precision_default_point": (
                bootstrap_precision_default["point_estimate"]
            ),
            "bootstrap_precision_default_ci_lower": (
                bootstrap_precision_default["ci_lower"]
            ),
            "bootstrap_precision_default_ci_upper": (
                bootstrap_precision_default["ci_upper"]
            ),

            "bootstrap_recall_default_point": (
                bootstrap_recall_default["point_estimate"]
            ),
            "bootstrap_recall_default_ci_lower": (
                bootstrap_recall_default["ci_lower"]
            ),
            "bootstrap_recall_default_ci_upper": (
                bootstrap_recall_default["ci_upper"]
            ),

            "bootstrap_pr_auc_default_point": (
                bootstrap_pr_auc_default["point_estimate"]
            ),
            "bootstrap_pr_auc_default_ci_lower": (
                bootstrap_pr_auc_default["ci_lower"]
            ),
            "bootstrap_pr_auc_default_ci_upper": (
                bootstrap_pr_auc_default["ci_upper"]
            ),

            "bootstrap_f1_optimized_point": bootstrap_f1_optimized[
                "point_estimate"
            ],
            "bootstrap_f1_optimized_ci_lower": bootstrap_f1_optimized[
                "ci_lower"
            ],
            "bootstrap_f1_optimized_ci_upper": bootstrap_f1_optimized[
                "ci_upper"
            ],

            "bootstrap_source": bootstrap_source,
            "results_file": str(result_file),
        })

        seed_export = seed_results.copy()
        seed_export.insert(0, "model", model_name)
        seed_export.insert(1, "model_directory", model_directory.name)
        seed_export.insert(2, "status", status)
        seed_frames.append(seed_export)

        if not fold_results.empty:
            fold_export = fold_results.copy()
            fold_export.insert(0, "model", model_name)
            fold_export.insert(1, "model_directory", model_directory.name)
            fold_frames.append(fold_export)

        hyperparameter_rows.extend(
            collect_hyperparameters(
                model_name,
                model_directory,
                test_summary,
                benchmark_summary,
            )
        )

    details = pd.DataFrame(detail_rows)
    if details.empty:
        raise RuntimeError("Aucun résultat exploitable trouvé.")

    # Classement réservé aux benchmarks complets.
    details["rank_f1"] = pd.Series(pd.NA, index=details.index, dtype="Int64")
    complete_mask = details["status"].eq("Benchmark complet")
    complete_order = (
        details.loc[complete_mask]
        .sort_values(
            ["ensemble_f1_mean", "ensemble_pr_auc_mean"],
            ascending=[False, False],
            na_position="last",
        )
        .index
    )
    for rank, index in enumerate(complete_order, start=1):
        details.loc[index, "rank_f1"] = rank

    details["_sort_complete"] = (~complete_mask).astype(int)
    details["_sort_rank"] = details["rank_f1"].fillna(10**6)
    details = (
        details.sort_values(
            ["_sort_complete", "_sort_rank", "ensemble_f1_mean"],
            ascending=[True, True, False],
            na_position="last",
        )
        .drop(columns=["_sort_complete", "_sort_rank"])
        .reset_index(drop=True)
    )

    presentation = pd.DataFrame({
        "Rang": details["rank_f1"],
        "Modèle": details["model"],
        "Statut": details["status"],
        "Seeds": details["seed_count"],
        "Folds": details["fold_count"],
        "F1 individuel test @0,5": [
            format_mean_std(mean, std, seeds)
            for mean, std, seeds in zip(
                details["individual_f1_mean"],
                details["individual_f1_std"],
                details["seed_count"],
            )
        ],
        "F1 ensemble test @0,5": [
            format_mean_std(mean, std, seeds)
            for mean, std, seeds in zip(
                details["ensemble_f1_mean"],
                details["ensemble_f1_std"],
                details["seed_count"],
            )
        ],
        "Précision ensemble @0,5": [
            format_mean_std(mean, std, seeds)
            for mean, std, seeds in zip(
                details["ensemble_precision_mean"],
                details["ensemble_precision_std"],
                details["seed_count"],
            )
        ],
        "Rappel ensemble @0,5": [
            format_mean_std(mean, std, seeds)
            for mean, std, seeds in zip(
                details["ensemble_recall_mean"],
                details["ensemble_recall_std"],
                details["seed_count"],
            )
        ],
        "PR-AUC ensemble": [
            format_mean_std(mean, std, seeds)
            for mean, std, seeds in zip(
                details["ensemble_pr_auc_mean"],
                details["ensemble_pr_auc_std"],
                details["seed_count"],
            )
        ],
        "Seuil OOF retenu": [
            format_threshold(mean, std, seeds)
            for mean, std, seeds in zip(
                details["optimized_threshold_mean"],
                details["optimized_threshold_std"],
                details["seed_count"],
            )
        ],
        "F1 test au seuil OOF": [
            format_mean_std(mean, std, seeds)
            for mean, std, seeds in zip(
                details["ensemble_f1_optimized_mean"],
                details["ensemble_f1_optimized_std"],
                details["seed_count"],
            )
        ],
        "IC 95 % F1 ensemble @0,5": [
            format_ci(lower, upper)
            for lower, upper in zip(
                details["bootstrap_f1_default_ci_lower"],
                details["bootstrap_f1_default_ci_upper"],
            )
        ],
        "IC 95 % PR-AUC": [
            format_ci(lower, upper)
            for lower, upper in zip(
                details["bootstrap_pr_auc_default_ci_lower"],
                details["bootstrap_pr_auc_default_ci_upper"],
            )
        ],
        "IC 95 % F1 au seuil OOF": [
            format_ci(lower, upper)
            for lower, upper in zip(
                details["bootstrap_f1_optimized_ci_lower"],
                details["bootstrap_f1_optimized_ci_upper"],
            )
        ],
        "Stratégie de déploiement": details[
            "declared_deployment_strategy"
        ],
    })

    all_seeds = (
        pd.concat(seed_frames, ignore_index=True)
        if seed_frames else pd.DataFrame()
    )
    all_folds = (
        pd.concat(fold_frames, ignore_index=True)
        if fold_frames else pd.DataFrame()
    )
    bootstrap_dataframe = pd.DataFrame(all_bootstrap_rows)
    hyperparameters = pd.DataFrame(hyperparameter_rows)

    graph_manifest, graph_warnings = generate_benchmark_graphs(
        details,
        all_seeds,
    )
    for warning in graph_warnings:
        warning_rows.append({
            "model": "Graphiques",
            "warning": warning,
        })

    warnings_dataframe = pd.DataFrame(warning_rows)

    # Sauvegardes CSV.
    presentation.to_csv(OUTPUT_COMPARISON_CSV, index=False)
    details.to_csv(OUTPUT_DETAILS_CSV, index=False)
    all_seeds.to_csv(OUTPUT_BY_SEED_CSV, index=False)
    all_folds.to_csv(OUTPUT_BY_FOLD_CSV, index=False)
    bootstrap_dataframe.to_csv(OUTPUT_BOOTSTRAP_CSV, index=False)
    hyperparameters.to_csv(OUTPUT_HYPERPARAMETERS_CSV, index=False)
    warnings_dataframe.to_csv(OUTPUT_WARNINGS_CSV, index=False)
    graph_manifest.to_csv(
        OUTPUT_GRAPHS_MANIFEST_CSV,
        index=False,
    )

    notes = pd.DataFrame([
        {
            "Élément": "Résultat principal",
            "Interprétation": (
                "Moyenne ± écart-type entre les seeds d'entraînement des "
                "performances de l'ensemble des cinq folds."
            ),
        },
        {
            "Élément": "F1 individuel",
            "Interprétation": (
                "Pour chaque seed, moyenne des performances des cinq modèles "
                "de fold évalués séparément sur le même jeu de test."
            ),
        },
        {
            "Élément": "F1 ensemble",
            "Interprétation": (
                "Pour chaque seed, les probabilités des cinq modèles de fold "
                "sont moyennées avant le calcul des métriques."
            ),
        },
        {
            "Élément": "Barres d'erreur ±",
            "Interprétation": (
                "Variabilité inter-seed uniquement, avec une partition fixe. "
                "Elle ne capture pas la variance liée à un autre split."
            ),
        },
        {
            "Élément": "IC bootstrap 95 %",
            "Interprétation": (
                "Incertitude d'échantillonnage sur les observations du jeu de "
                "test, obtenue par bootstrap stratifié."
            ),
        },
        {
            "Élément": "Seuil OOF",
            "Interprétation": (
                "Seuil sélectionné exclusivement sur les prédictions OOF du "
                "train, puis appliqué sans modification au test."
            ),
        },
        {
            "Élément": "Pilote BioMistral",
            "Interprétation": (
                "Avec une seule seed, l'écart-type inter-seed est non "
                "estimable. Le résultat ne doit pas être présenté comme "
                "moyenne ± 0."
            ),
        },
        {
            "Élément": "Déploiement",
            "Interprétation": (
                "Si la stratégie retenue est fold_ensemble, le système déployé "
                "utilise les cinq modèles de fold et moyenne leurs probabilités."
            ),
        },
        {
            "Élément": "Éléments encore séparés du tableau",
            "Interprétation": (
                "La comparaison appariée des deux meilleurs modèles et leur "
                "courbe précision-rappel moyenne restent des analyses "
                "complémentaires, non remplacées par ce tableau."
            ),
        },
    ])

    # Classeur Excel.
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            presentation.to_excel(
                writer,
                sheet_name="Comparaison",
                index=False,
            )
            details.to_excel(
                writer,
                sheet_name="Détails numériques",
                index=False,
            )
            all_seeds.to_excel(
                writer,
                sheet_name="Résultats par seed",
                index=False,
            )
            all_folds.to_excel(
                writer,
                sheet_name="Résultats par fold",
                index=False,
            )
            bootstrap_dataframe.to_excel(
                writer,
                sheet_name="Bootstrap 95",
                index=False,
            )
            hyperparameters.to_excel(
                writer,
                sheet_name="Hyperparamètres",
                index=False,
            )
            warnings_dataframe.to_excel(
                writer,
                sheet_name="Contrôles",
                index=False,
            )
            graph_manifest.to_excel(
                writer,
                sheet_name="Graphiques",
                index=False,
            )
            notes.to_excel(
                writer,
                sheet_name="Notes méthodologiques",
                index=False,
            )

        style_excel_workbook(OUTPUT_XLSX)
    except ImportError:
        print(
            "\nATTENTION : openpyxl n'est pas installé. Les CSV ont été "
            "créés, mais pas le fichier Excel."
        )
        print("Installation : python3 -m pip install openpyxl")

    # Affichage terminal.
    print("\n" + "=" * 150)
    print(" BENCHMARK DEREEV — TABLEAU FINAL")
    print("=" * 150)
    print(presentation.to_string(index=False))

    print("\nFichiers créés :")
    for path in (
        OUTPUT_COMPARISON_CSV,
        OUTPUT_DETAILS_CSV,
        OUTPUT_BY_SEED_CSV,
        OUTPUT_BY_FOLD_CSV,
        OUTPUT_BOOTSTRAP_CSV,
        OUTPUT_HYPERPARAMETERS_CSV,
        OUTPUT_WARNINGS_CSV,
        OUTPUT_GRAPHS_MANIFEST_CSV,
    ):
        print(f"- {path}")

    if OUTPUT_XLSX.exists():
        print(f"- {OUTPUT_XLSX}")

    if not graph_manifest.empty:
        print(f"- Dossier des graphes : {GRAPHS_DIR}")
        for graph_path in graph_manifest["fichier"]:
            print(f"  - {graph_path}")

    if not warnings_dataframe.empty:
        print("\nPOINTS À VÉRIFIER :")
        print(warnings_dataframe.to_string(index=False))

    print(
        "\nImportant : le classement est attribué uniquement aux modèles "
        "marqués « Benchmark complet ». Les pilotes restent affichés sans "
        "rang comparatif."
    )


if __name__ == "__main__":
    main()