"""Outils statistiques partagés par le benchmark DEREEV.

Ce module distingue explicitement :

- la variabilité inter-seed à partition fixe, résumée par moyenne ± écart-type ;
- l'incertitude d'échantillonnage sur le test, estimée par bootstrap stratifié ;
- les comparaisons appariées entre modèles sur les mêmes observations du test.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from src.threshold_metrics import (
    add_pr_auc_to_metrics,
    compute_threshold_metrics,
)

SUPPORTED_METRICS = ("f1", "precision", "recall", "accuracy", "pr_auc")


def sample_std(values: Sequence[float]) -> float:
    """Écart-type échantillonnal, avec 0 pour une seule valeur."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    return float(np.std(array, ddof=1))


def metric_bundle(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> Dict[str, float]:
    """Calcule précision, rappel, F1, accuracy et PR-AUC."""
    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    return add_pr_auc_to_metrics(
        metrics=compute_threshold_metrics(
            labels=labels_array,
            probabilities=probabilities_array,
            threshold=float(threshold),
        ),
        labels=labels_array,
        probabilities=probabilities_array,
    )


def aggregate_scalar_values(values: Sequence[float]) -> Dict[str, float]:
    """Résumé classique d'une série de valeurs."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Impossible d'agréger une série vide.")
    return {
        "mean": float(np.mean(array)),
        "std": sample_std(array),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_metric_dicts(
    metrics: Sequence[Mapping[str, float]],
    metric_names: Sequence[str] = SUPPORTED_METRICS,
) -> Dict[str, Dict[str, float]]:
    """Agrège plusieurs dictionnaires de métriques."""
    if not metrics:
        raise ValueError("Aucun dictionnaire de métriques à agréger.")
    return {
        name: aggregate_scalar_values([float(item[name]) for item in metrics])
        for name in metric_names
    }


def _validate_probability_matrix(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probability_matrix, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if labels.ndim != 1:
        raise ValueError("labels doit être un vecteur 1D.")
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]
    if probabilities.ndim != 2:
        raise ValueError("probability_matrix doit avoir la forme systèmes × observations.")
    if probabilities.shape[1] != labels.size:
        raise ValueError("Les probabilités et les labels n'ont pas la même longueur.")
    if thresholds.ndim == 0:
        thresholds = np.repeat(float(thresholds), probabilities.shape[0])
    if thresholds.shape != (probabilities.shape[0],):
        raise ValueError("Il faut un seuil par système.")
    if np.unique(labels).size != 2:
        raise ValueError("Le bootstrap stratifié nécessite les deux classes.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Les probabilités contiennent une valeur non finie.")
    return labels, probabilities, thresholds


def stratified_bootstrap_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rééchantillonne séparément les classes 0 et 1.

    La taille et la prévalence du jeu de test sont ainsi conservées dans chaque
    réplication, ce qui évite les réplications sans classe positive.
    """
    class_zero = np.flatnonzero(labels == 0)
    class_one = np.flatnonzero(labels == 1)
    sampled_zero = rng.choice(class_zero, size=class_zero.size, replace=True)
    sampled_one = rng.choice(class_one, size=class_one.size, replace=True)
    indices = np.concatenate([sampled_zero, sampled_one])
    rng.shuffle(indices)
    return indices


def _mean_metric_bundle_across_systems(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, float]:
    """Calcule une seule fois toutes les métriques de chaque système."""
    bundles = [
        metric_bundle(labels, probabilities, threshold)
        for probabilities, threshold in zip(probability_matrix, thresholds)
    ]
    return {
        metric_name: float(np.mean([bundle[metric_name] for bundle in bundles]))
        for metric_name in SUPPORTED_METRICS
    }


def _mean_metric_across_systems(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    thresholds: np.ndarray,
    metric_name: str,
) -> float:
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(f"Métrique non prise en charge : {metric_name}")
    return _mean_metric_bundle_across_systems(
        labels, probability_matrix, thresholds
    )[metric_name]


def bootstrap_mean_metrics_across_systems(
    labels: Iterable[int],
    probability_matrix: np.ndarray,
    thresholds: Sequence[float] | float,
    metric_names: Sequence[str] = SUPPORTED_METRICS,
    n_resamples: int = 2000,
    random_seed: int = 2026,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """IC bootstrap de la moyenne d'une métrique entre systèmes/seeds.

    Les mêmes indices bootstrap sont appliqués à tous les systèmes. Cela
    préserve l'appariement entre seeds et isole l'incertitude liée aux
    observations du test.
    """
    if n_resamples < 100:
        raise ValueError("Utilisez au moins 100 réplications bootstrap.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level doit être compris entre 0 et 1.")

    labels_array, probabilities, threshold_array = _validate_probability_matrix(
        np.asarray(labels, dtype=int),
        np.asarray(probability_matrix, dtype=float),
        np.asarray(thresholds, dtype=float),
    )
    rng = np.random.default_rng(random_seed)
    metric_names = tuple(metric_names)
    replicates = {
        metric: np.empty(n_resamples, dtype=float)
        for metric in metric_names
    }

    for iteration in range(n_resamples):
        indices = stratified_bootstrap_indices(labels_array, rng)
        sampled_labels = labels_array[indices]
        sampled_probabilities = probabilities[:, indices]
        sampled_bundle = _mean_metric_bundle_across_systems(
            sampled_labels, sampled_probabilities, threshold_array
        )
        for metric in metric_names:
            replicates[metric][iteration] = sampled_bundle[metric]

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    result: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        point = _mean_metric_across_systems(
            labels_array,
            probabilities,
            threshold_array,
            metric,
        )
        values = replicates[metric]
        result[metric] = {
            "point_estimate": float(point),
            "ci_lower": float(np.percentile(values, lower_percentile)),
            "ci_upper": float(np.percentile(values, upper_percentile)),
            "confidence_level": float(confidence_level),
            "n_resamples": int(n_resamples),
            "bootstrap_seed": int(random_seed),
            "method": "stratified_percentile_bootstrap_paired_across_systems",
        }
    return result


def paired_bootstrap_metric_differences(
    labels: Iterable[int],
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    thresholds_a: Sequence[float] | float,
    thresholds_b: Sequence[float] | float,
    metric_names: Sequence[str] = ("f1", "pr_auc", "precision", "recall"),
    n_resamples: int = 5000,
    random_seed: int = 2026,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap apparié de la différence B - A sur les mêmes observations."""
    labels_array, matrix_a, threshold_array_a = _validate_probability_matrix(
        np.asarray(labels, dtype=int),
        np.asarray(probabilities_a, dtype=float),
        np.asarray(thresholds_a, dtype=float),
    )
    labels_b, matrix_b, threshold_array_b = _validate_probability_matrix(
        np.asarray(labels, dtype=int),
        np.asarray(probabilities_b, dtype=float),
        np.asarray(thresholds_b, dtype=float),
    )
    if not np.array_equal(labels_array, labels_b):
        raise ValueError("Les labels des deux modèles diffèrent.")
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            "Les matrices doivent avoir la même forme et les mêmes seeds appariées."
        )
    if n_resamples < 100:
        raise ValueError("Utilisez au moins 100 réplications bootstrap.")

    rng = np.random.default_rng(random_seed)
    metric_names = tuple(metric_names)
    replicates = {
        metric: np.empty(n_resamples, dtype=float)
        for metric in metric_names
    }

    for iteration in range(n_resamples):
        indices = stratified_bootstrap_indices(labels_array, rng)
        sampled_labels = labels_array[indices]
        sampled_a = matrix_a[:, indices]
        sampled_b = matrix_b[:, indices]
        bundle_a = _mean_metric_bundle_across_systems(
            sampled_labels, sampled_a, threshold_array_a
        )
        bundle_b = _mean_metric_bundle_across_systems(
            sampled_labels, sampled_b, threshold_array_b
        )
        for metric in metric_names:
            replicates[metric][iteration] = bundle_b[metric] - bundle_a[metric]

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    result: Dict[str, Dict[str, float]] = {}

    for metric in metric_names:
        point_a = _mean_metric_across_systems(
            labels_array, matrix_a, threshold_array_a, metric
        )
        point_b = _mean_metric_across_systems(
            labels_array, matrix_b, threshold_array_b, metric
        )
        deltas = replicates[metric]
        # Correction +1 pour éviter un p exactement nul.
        probability_non_positive = (
            np.count_nonzero(deltas <= 0.0) + 1
        ) / (n_resamples + 1)
        probability_non_negative = (
            np.count_nonzero(deltas >= 0.0) + 1
        ) / (n_resamples + 1)
        p_value = min(1.0, 2.0 * min(
            probability_non_positive,
            probability_non_negative,
        ))
        ci_lower = float(np.percentile(deltas, lower_percentile))
        ci_upper = float(np.percentile(deltas, upper_percentile))
        result[metric] = {
            "model_a": float(point_a),
            "model_b": float(point_b),
            "difference_b_minus_a": float(point_b - point_a),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value_two_sided": float(p_value),
            "confidence_level": float(confidence_level),
            "n_resamples": int(n_resamples),
            "bootstrap_seed": int(random_seed),
            "significant_at_0_05": bool(ci_lower > 0.0 or ci_upper < 0.0),
            "method": "paired_stratified_percentile_bootstrap",
        }
    return result
