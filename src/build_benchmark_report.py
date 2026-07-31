"""Construit les tableaux principaux et les annexes du benchmark DEREEV.

Sorties :
- benchmark_main_table.csv : précision, rappel, F1, PR-AUC, seuil et IC 95 % ;
- benchmark_seed_appendix.csv : une ligne par modèle et seed ;
- benchmark_fold_appendix.csv : une ligne par modèle, seed et fold ;
- benchmark_hyperparameters.csv : audit des réglages par architecture ;
- benchmark_report.xlsx : les quatre tableaux dans un même classeur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.config import BENCHMARK_MODELS_DIR


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_models(requested: str | None) -> List[Path]:
    root = Path(BENCHMARK_MODELS_DIR)
    if requested:
        names = [item.strip() for item in requested.split(",") if item.strip()]
        paths = [root / name for name in names]
    else:
        paths = [
            path for path in root.iterdir()
            if path.is_dir()
            and (path / "test_benchmark_summary.json").exists()
        ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Dossiers introuvables : " + ", ".join(missing))
    return sorted(paths, key=lambda path: path.name)


def aggregate_value(summary: Dict, key: str, field: str = "mean"):
    value = summary.get("aggregate_test_metrics", {}).get(key, {})
    return value.get(field) if isinstance(value, dict) else None


def bootstrap_value(summary: Dict, evaluation: str, metric: str, field: str):
    return (
        summary.get("bootstrap_confidence_intervals", {})
        .get(evaluation, {})
        .get(metric, {})
        .get(field)
    )


def flatten_dict(value: Dict, prefix: str = "") -> Dict:
    result = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten_dict(item, full_key))
        elif isinstance(item, (list, tuple)):
            result[full_key] = ",".join(map(str, item))
        else:
            result[full_key] = item
    return result


def build_main_row(model_root: Path, train_summary: Dict, test_summary: Dict) -> Dict:
    training_configuration = train_summary.get("training_configuration", {})
    return {
        "model_output_name": model_root.name,
        "model_display_name": test_summary.get(
            "model_display_name", train_summary.get("model_display_name", model_root.name)
        ),
        "deployment_strategy": test_summary.get("declared_deployment_strategy"),
        "split_seed": test_summary.get("split_seed"),
        "fold_count": test_summary.get("fold_count"),
        "seed_count": len(test_summary.get("training_seeds", [])),
        "error_bar_scope": test_summary.get("error_bar_scope"),
        "max_length": training_configuration.get("max_length"),
        "learning_rate": training_configuration.get("learning_rate"),
        "batch_size": training_configuration.get("batch_size"),
        "epochs_max": training_configuration.get("epochs_max"),
        "individual_f1_0_5_mean": aggregate_value(
            test_summary, "individual_mean_f1_threshold_0_5"
        ),
        "individual_f1_0_5_inter_seed_std": aggregate_value(
            test_summary, "individual_mean_f1_threshold_0_5", "std"
        ),
        "ensemble_precision_0_5_mean": aggregate_value(
            test_summary, "ensemble_precision_threshold_0_5"
        ),
        "ensemble_precision_0_5_inter_seed_std": aggregate_value(
            test_summary, "ensemble_precision_threshold_0_5", "std"
        ),
        "ensemble_recall_0_5_mean": aggregate_value(
            test_summary, "ensemble_recall_threshold_0_5"
        ),
        "ensemble_recall_0_5_inter_seed_std": aggregate_value(
            test_summary, "ensemble_recall_threshold_0_5", "std"
        ),
        "ensemble_f1_0_5_mean": aggregate_value(
            test_summary, "ensemble_f1_threshold_0_5"
        ),
        "ensemble_f1_0_5_inter_seed_std": aggregate_value(
            test_summary, "ensemble_f1_threshold_0_5", "std"
        ),
        "ensemble_pr_auc_mean": aggregate_value(test_summary, "ensemble_pr_auc"),
        "ensemble_pr_auc_inter_seed_std": aggregate_value(
            test_summary, "ensemble_pr_auc", "std"
        ),
        "optimized_threshold_mean": aggregate_value(test_summary, "optimized_threshold"),
        "optimized_threshold_inter_seed_std": aggregate_value(
            test_summary, "optimized_threshold", "std"
        ),
        "ensemble_precision_optimized_mean": aggregate_value(
            test_summary, "ensemble_precision_optimized"
        ),
        "ensemble_recall_optimized_mean": aggregate_value(
            test_summary, "ensemble_recall_optimized"
        ),
        "ensemble_f1_optimized_mean": aggregate_value(
            test_summary, "ensemble_f1_optimized"
        ),
        "ensemble_f1_optimized_inter_seed_std": aggregate_value(
            test_summary, "ensemble_f1_optimized", "std"
        ),
        "bootstrap_f1_optimized_ci95_lower": bootstrap_value(
            test_summary, "ensemble_threshold_optimized_on_oof", "f1", "ci_lower"
        ),
        "bootstrap_f1_optimized_ci95_upper": bootstrap_value(
            test_summary, "ensemble_threshold_optimized_on_oof", "f1", "ci_upper"
        ),
        "bootstrap_pr_auc_ci95_lower": bootstrap_value(
            test_summary, "ensemble_threshold_optimized_on_oof", "pr_auc", "ci_lower"
        ),
        "bootstrap_pr_auc_ci95_upper": bootstrap_value(
            test_summary, "ensemble_threshold_optimized_on_oof", "pr_auc", "ci_upper"
        ),
    }


def validation_fold_rows(model_root: Path, train_summary: Dict) -> List[Dict]:
    rows = []
    seed_summaries = train_summary.get("seed_summaries", [])
    for seed_summary in seed_summaries:
        seed = seed_summary.get("training_seed")
        for fold in seed_summary.get("folds", []):
            metrics = fold.get("validation_metrics", {})
            row = {
                "model_output_name": model_root.name,
                "training_seed": seed,
                "fold": fold.get("fold"),
                "train_size": fold.get("train_size"),
                "validation_size": fold.get("validation_size"),
                "validation_f1": metrics.get("f1", fold.get("validation_f1")),
                "validation_precision": metrics.get(
                    "precision", fold.get("validation_precision")
                ),
                "validation_recall": metrics.get("recall", fold.get("validation_recall")),
                "validation_pr_auc": metrics.get("pr_auc", fold.get("validation_pr_auc")),
                "duration_seconds": fold.get("duration_seconds"),
            }
            rows.append(row)
    return rows


def main(args: argparse.Namespace) -> None:
    model_roots = discover_models(args.models)
    output_directory = Path(args.output_dir) if args.output_dir else (
        Path(BENCHMARK_MODELS_DIR) / "reports"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    main_rows = []
    seed_frames = []
    fold_frames = []
    validation_rows = []
    hyperparameter_rows = []

    for model_root in model_roots:
        test_path = model_root / "test_benchmark_summary.json"
        train_path = model_root / "benchmark_summary.json"
        test_summary = read_json(test_path)
        train_summary = read_json(train_path) if train_path.exists() else {}
        main_rows.append(build_main_row(model_root, train_summary, test_summary))

        seed_path = model_root / "all_seeds_test_results.csv"
        if seed_path.exists():
            seed_frame = pd.read_csv(seed_path)
            seed_frame.insert(0, "model_output_name", model_root.name)
            seed_frames.append(seed_frame)

        fold_path = model_root / "all_fold_models_test_results.csv"
        if fold_path.exists():
            fold_frame = pd.read_csv(fold_path)
            if "model_output_name" not in fold_frame.columns:
                fold_frame.insert(0, "model_output_name", model_root.name)
            fold_frames.append(fold_frame)

        validation_rows.extend(validation_fold_rows(model_root, train_summary))
        hyperparameters = train_summary.get("training_configuration", {})
        if not hyperparameters:
            run_configuration_path = model_root / "run_configuration.json"
            if run_configuration_path.exists():
                hyperparameters = read_json(run_configuration_path)
        hyperparameter_rows.append({
            "model_output_name": model_root.name,
            **flatten_dict(hyperparameters),
        })

    main_dataframe = pd.DataFrame(main_rows)
    seed_dataframe = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    test_fold_dataframe = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    validation_fold_dataframe = pd.DataFrame(validation_rows)
    if not test_fold_dataframe.empty and not validation_fold_dataframe.empty:
        fold_dataframe = validation_fold_dataframe.merge(
            test_fold_dataframe,
            on=["model_output_name", "training_seed", "fold"],
            how="outer",
            suffixes=("_validation", "_test"),
        )
    elif not test_fold_dataframe.empty:
        fold_dataframe = test_fold_dataframe
    else:
        fold_dataframe = validation_fold_dataframe
    hyperparameter_dataframe = pd.DataFrame(hyperparameter_rows)

    outputs = {
        "benchmark_main_table.csv": main_dataframe,
        "benchmark_seed_appendix.csv": seed_dataframe,
        "benchmark_fold_appendix.csv": fold_dataframe,
        "benchmark_hyperparameters.csv": hyperparameter_dataframe,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(output_directory / filename, index=False)

    excel_path = output_directory / "benchmark_report.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            main_dataframe.to_excel(writer, sheet_name="main", index=False)
            seed_dataframe.to_excel(writer, sheet_name="seeds", index=False)
            fold_dataframe.to_excel(writer, sheet_name="folds", index=False)
            hyperparameter_dataframe.to_excel(
                writer, sheet_name="hyperparameters", index=False
            )
    except ImportError:
        print("openpyxl absent : le classeur Excel n'a pas été créé.")
        excel_path = None

    print(f"Rapports créés dans : {output_directory}")
    for filename in outputs:
        print(f"- {filename}")
    if excel_path is not None:
        print(f"- {excel_path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        help="Noms de dossiers séparés par des virgules. Défaut : tous les modèles évalués.",
    )
    parser.add_argument("--output-dir")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
