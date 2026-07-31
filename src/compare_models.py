"""Comparaison appariée de deux modèles DEREEV.

Le script réalise deux analyses complémentaires :

- Wilcoxon sur les métriques des mêmes seeds d'entraînement ;
- bootstrap apparié sur les mêmes indices du test.

Le bootstrap est l'analyse principale recommandée lorsque le test est de taille
modérée. La différence est toujours définie comme ``modèle B - modèle A``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.benchmark_statistics import paired_bootstrap_metric_differences
from src.config import BENCHMARK_MODELS_DIR

try:
    from scipy.stats import wilcoxon
except ImportError as error:  # pragma: no cover - dépend de l'environnement
    raise ImportError(
        "scipy est nécessaire pour le test de Wilcoxon : "
        "python3 -m pip install scipy"
    ) from error


def load_seed_table(model_root: Path) -> pd.DataFrame:
    path = model_root / "all_seeds_test_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Résultats multi-seeds introuvables : {path}")
    dataframe = pd.read_csv(path)
    if "training_seed" not in dataframe.columns:
        raise ValueError(f"training_seed manque dans {path}")
    return dataframe.sort_values("training_seed").reset_index(drop=True)


def load_prediction_matrix(
    model_root: Path,
    seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability_rows = []
    thresholds = []
    reference_labels = None
    reference_indices = None

    seed_table = load_seed_table(model_root).set_index("training_seed")
    for seed in seeds:
        path = model_root / f"seed_{seed}" / "test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Prédictions introuvables : {path}")
        dataframe = pd.read_csv(path)
        required = {"row_index", "label", "probability_class_1"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {path}: {sorted(missing)}. "
                "Relancez l'évaluation avec la nouvelle version."
            )
        dataframe = dataframe.sort_values("row_index").reset_index(drop=True)
        indices = dataframe["row_index"].to_numpy(int)
        labels = dataframe["label"].to_numpy(int)
        if reference_labels is None:
            reference_labels = labels
            reference_indices = indices
        elif not np.array_equal(reference_labels, labels) or not np.array_equal(
            reference_indices, indices
        ):
            raise RuntimeError(
                f"Ordre ou labels incompatibles pour la seed {seed} dans {model_root}."
            )
        probability_rows.append(dataframe["probability_class_1"].to_numpy(float))
        if seed not in seed_table.index:
            raise ValueError(f"Seed {seed} absente de {model_root}.")
        thresholds.append(float(seed_table.loc[seed, "optimized_threshold"]))

    assert reference_labels is not None
    return (
        np.stack(probability_rows, axis=0),
        np.asarray(thresholds, dtype=float),
        reference_labels,
    )


def wilcoxon_result(values_a: np.ndarray, values_b: np.ndarray) -> Dict:
    differences = values_b - values_a
    if np.allclose(differences, 0.0):
        return {
            "statistic": 0.0,
            "p_value_two_sided": 1.0,
            "difference_b_minus_a_mean": 0.0,
            "note": "Toutes les différences appariées sont nulles.",
        }
    result = wilcoxon(
        values_b,
        values_a,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )
    return {
        "statistic": float(result.statistic),
        "p_value_two_sided": float(result.pvalue),
        "difference_b_minus_a_mean": float(np.mean(differences)),
        "paired_values": int(values_a.size),
        "warning": (
            "Avec seulement cinq seeds, Wilcoxon a une puissance limitée ; "
            "interprétez-le avec le bootstrap apparié sur le test."
        ),
    }


def choose_wilcoxon_columns(dataframe: pd.DataFrame) -> Dict[str, str]:
    candidates = {
        "f1_optimized": ["ensemble_f1_optimized", "test_f1_optimized"],
        "f1_threshold_0_5": [
            "ensemble_f1_threshold_0_5", "test_f1_threshold_0_5"
        ],
        "pr_auc": ["ensemble_pr_auc", "test_pr_auc"],
        "precision_optimized": [
            "ensemble_precision_optimized", "test_precision_optimized"
        ],
        "recall_optimized": [
            "ensemble_recall_optimized", "test_recall_optimized"
        ],
    }
    selected = {}
    for metric, options in candidates.items():
        for column in options:
            if column in dataframe.columns:
                selected[metric] = column
                break
    return selected


def main(args: argparse.Namespace) -> None:
    root_a = Path(BENCHMARK_MODELS_DIR) / args.model_a
    root_b = Path(BENCHMARK_MODELS_DIR) / args.model_b
    table_a = load_seed_table(root_a)
    table_b = load_seed_table(root_b)

    common_seeds = sorted(
        set(table_a["training_seed"].astype(int))
        & set(table_b["training_seed"].astype(int))
    )
    if len(common_seeds) < 2:
        raise ValueError("Au moins deux seeds communes sont nécessaires.")

    aligned_a = table_a.set_index("training_seed").loc[common_seeds]
    aligned_b = table_b.set_index("training_seed").loc[common_seeds]
    columns_a = choose_wilcoxon_columns(aligned_a)
    columns_b = choose_wilcoxon_columns(aligned_b)
    common_metrics = sorted(set(columns_a) & set(columns_b))

    wilcoxon_results = {}
    for metric in common_metrics:
        wilcoxon_results[metric] = wilcoxon_result(
            aligned_a[columns_a[metric]].to_numpy(float),
            aligned_b[columns_b[metric]].to_numpy(float),
        )

    probabilities_a, thresholds_a, labels_a = load_prediction_matrix(
        root_a, common_seeds
    )
    probabilities_b, thresholds_b, labels_b = load_prediction_matrix(
        root_b, common_seeds
    )
    if not np.array_equal(labels_a, labels_b):
        raise RuntimeError("Les deux modèles n'utilisent pas le même test.")

    bootstrap_optimized = paired_bootstrap_metric_differences(
        labels=labels_a,
        probabilities_a=probabilities_a,
        probabilities_b=probabilities_b,
        thresholds_a=thresholds_a,
        thresholds_b=thresholds_b,
        metric_names=("f1", "precision", "recall", "pr_auc"),
        n_resamples=args.bootstrap_resamples,
        random_seed=args.bootstrap_seed,
    )
    bootstrap_default = paired_bootstrap_metric_differences(
        labels=labels_a,
        probabilities_a=probabilities_a,
        probabilities_b=probabilities_b,
        thresholds_a=np.repeat(0.5, len(common_seeds)),
        thresholds_b=np.repeat(0.5, len(common_seeds)),
        metric_names=("f1", "precision", "recall", "pr_auc"),
        n_resamples=args.bootstrap_resamples,
        random_seed=args.bootstrap_seed,
    )

    output_directory = (
        Path(BENCHMARK_MODELS_DIR)
        / "comparisons"
        / f"{args.model_a}_vs_{args.model_b}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "difference_definition": "model_b_minus_model_a",
        "paired_training_seeds": common_seeds,
        "wilcoxon_across_seed_metrics": wilcoxon_results,
        "paired_bootstrap_test_observations": {
            "threshold_0_5": bootstrap_default,
            "thresholds_optimized_on_each_model_oof": bootstrap_optimized,
        },
        "decision_rule": (
            "Ne pas annoncer un vainqueur si l'IC 95 % de la différence contient "
            "0. La PR-AUC est prioritaire pour comparer la qualité de classement ; "
            "le F1 optimisé décrit le fonctionnement au seuil retenu."
        ),
    }
    json_path = output_directory / "paired_comparison.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    rows = []
    for evaluation_name, results in (
        ("threshold_0_5", bootstrap_default),
        ("optimized_oof_threshold", bootstrap_optimized),
    ):
        for metric, values in results.items():
            rows.append({
                "evaluation": evaluation_name,
                "metric": metric,
                **values,
            })
    csv_path = output_directory / "paired_bootstrap_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Comparaison appariée : {args.model_b} - {args.model_a}")
    print(f"Seeds appariées      : {common_seeds}")
    for metric in ("pr_auc", "f1"):
        values = bootstrap_optimized[metric]
        print(
            f"{metric}: Δ={values['difference_b_minus_a']:+.4f}, "
            f"IC95 [{values['ci_lower']:+.4f}; {values['ci_upper']:+.4f}], "
            f"p={values['p_value_two_sided']:.4f}"
        )
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, help="Nom du dossier du modèle A.")
    parser.add_argument("--model-b", required=True, help="Nom du dossier du modèle B.")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
