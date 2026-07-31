"""Audit des hyperparamètres utilisés par les modèles du benchmark DEREEV.

Le script ne décide pas quels réglages sont optimaux. Il vérifie que les
réglages réellement utilisés sont enregistrés et rend explicite si plusieurs
architectures ont reçu exactement les mêmes valeurs de learning rate,
max_length, batch size ou nombre maximal d'époques.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import BENCHMARK_MODELS_DIR

FIELDS = [
    "max_length",
    "learning_rate",
    "batch_size",
    "epochs_max",
    "early_stopping_patience",
    "early_stopping_threshold",
]


def load_configuration(model_root: Path) -> Dict:
    benchmark_path = model_root / "benchmark_summary.json"
    run_path = model_root / "run_configuration.json"
    if benchmark_path.exists():
        with benchmark_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        configuration = summary.get("training_configuration", {})
        if configuration:
            return configuration
    if run_path.exists():
        with run_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def main(args: argparse.Namespace) -> None:
    root = Path(BENCHMARK_MODELS_DIR)
    if args.models:
        names = [name.strip() for name in args.models.split(",") if name.strip()]
        model_roots = [root / name for name in names]
    else:
        model_roots = [
            path for path in root.iterdir()
            if path.is_dir() and (path / "benchmark_summary.json").exists()
        ]

    rows: List[Dict] = []
    for model_root in sorted(model_roots):
        configuration = load_configuration(model_root)
        rows.append({
            "model_output_name": model_root.name,
            **{field: configuration.get(field) for field in FIELDS},
        })
    dataframe = pd.DataFrame(rows)
    output_path = Path(args.output) if args.output else root / "reports" / "hyperparameter_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    print(dataframe.to_string(index=False))
    print("\nAudit :")
    for field in FIELDS:
        non_missing = dataframe[field].dropna() if field in dataframe else pd.Series(dtype=object)
        if non_missing.empty:
            print(f"- {field}: non enregistré — à corriger avant interprétation.")
        elif non_missing.nunique(dropna=True) == 1 and len(non_missing) > 1:
            print(
                f"- {field}: valeur identique ({non_missing.iloc[0]}) pour tous "
                "les modèles renseignés. Ce n'est pas nécessairement incorrect, "
                "mais il faut le justifier ou effectuer une sélection OOF par architecture."
            )
        else:
            print(f"- {field}: réglages différenciés et enregistrés.")
    print(f"\nCSV : {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models")
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
