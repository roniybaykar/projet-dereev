"""Trace la courbe précision-rappel moyenne de deux modèles DEREEV.

Pour chaque modèle, la courbe est calculée séparément pour chaque seed à partir
de l'ensemble des cinq folds. Les précisions sont interpolées sur une grille de
rappel commune, puis moyennées. La bande représente ± un écart-type inter-seed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import BENCHMARK_MODELS_DIR


def available_seeds(model_root: Path) -> List[int]:
    table_path = model_root / "all_seeds_test_results.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    dataframe = pd.read_csv(table_path)
    return sorted(dataframe["training_seed"].astype(int).tolist())


def load_model_curves(
    model_root: Path,
    recall_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    interpolated_curves = []
    average_precisions = []
    reference_labels = None
    seeds = available_seeds(model_root)

    for seed in seeds:
        path = model_root / f"seed_{seed}" / "test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        dataframe = pd.read_csv(path).sort_values("row_index")
        labels = dataframe["label"].to_numpy(int)
        probabilities = dataframe["probability_class_1"].to_numpy(float)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"Labels incompatibles dans {model_root}.")

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        # sklearn renvoie le rappel dans l'ordre décroissant.
        recall = recall[::-1]
        precision = precision[::-1]
        unique_recall, unique_indices = np.unique(recall, return_index=True)
        unique_precision = precision[unique_indices]
        interpolated = np.interp(
            recall_grid,
            unique_recall,
            unique_precision,
            left=unique_precision[0],
            right=unique_precision[-1],
        )
        interpolated_curves.append(interpolated)
        average_precisions.append(average_precision_score(labels, probabilities))

    return (
        np.stack(interpolated_curves, axis=0),
        np.asarray(average_precisions, dtype=float),
        seeds,
    )


def main(args: argparse.Namespace) -> None:
    model_a_root = Path(BENCHMARK_MODELS_DIR) / args.model_a
    model_b_root = Path(BENCHMARK_MODELS_DIR) / args.model_b
    recall_grid = np.linspace(0.0, 1.0, args.grid_points)

    curves_a, ap_a, seeds_a = load_model_curves(model_a_root, recall_grid)
    curves_b, ap_b, seeds_b = load_model_curves(model_b_root, recall_grid)

    output_directory = Path(args.output_dir) if args.output_dir else (
        Path(BENCHMARK_MODELS_DIR) / "comparisons" / f"{args.model_a}_vs_{args.model_b}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, curves, aps, seeds in (
        (args.model_a, curves_a, ap_a, seeds_a),
        (args.model_b, curves_b, ap_b, seeds_b),
    ):
        mean_precision = np.mean(curves, axis=0)
        std_precision = np.std(curves, axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean_precision)
        label = (
            f"{model_name} — PR-AUC {np.mean(aps):.4f} "
            f"± {np.std(aps, ddof=1) if aps.size > 1 else 0.0:.4f}"
        )
        line = ax.plot(recall_grid, mean_precision, label=label)[0]
        ax.fill_between(
            recall_grid,
            np.clip(mean_precision - std_precision, 0.0, 1.0),
            np.clip(mean_precision + std_precision, 0.0, 1.0),
            alpha=0.15,
            color=line.get_color(),
        )
        for index, recall_value in enumerate(recall_grid):
            rows.append({
                "model": model_name,
                "recall": recall_value,
                "mean_precision": mean_precision[index],
                "std_precision_across_seeds": std_precision[index],
                "seed_count": len(seeds),
                "mean_pr_auc": float(np.mean(aps)),
                "std_pr_auc_across_seeds": float(
                    np.std(aps, ddof=1) if aps.size > 1 else 0.0
                ),
            })

    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Courbes précision-rappel moyennes sur les seeds")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()

    png_path = output_directory / "mean_precision_recall_curves.png"
    csv_path = output_directory / "mean_precision_recall_curves.csv"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Courbe : {png_path}")
    print(f"Données: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--grid-points", type=int, default=501)
    parser.add_argument("--output-dir")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
