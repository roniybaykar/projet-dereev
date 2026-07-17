"""Supprime les doublons de notes avant l'equilibrage et le split.

Regles appliquees :
- une note repetee avec un seul label est conservee une seule fois ;
- une note repetee avec plusieurs labels est entierement exclue ;
- les lignes invalides sont exclues et placees dans un rapport d'audit.

Le texte original est conserve dans le fichier de sortie. Une version normalisee
du texte sert uniquement a comparer les notes (casse et espaces ignores).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "raw" / "dereev_data_3.xlsx"
DEFAULT_OUTPUT = BASE_DIR / "data" / "raw" / "dereev_data_3_deduplicated.xlsx"
DEFAULT_AUDIT_DIR = BASE_DIR / "data" / "processed" / "deduplication_audit"

TEXT_COL = "cn_notes"
LABEL_COL = "reev_classe"


def normalize_note(text: object) -> str:
    """Cree une cle de comparaison sans modifier le texte sauvegarde."""
    if pd.isna(text):
        return ""

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def deduplicate_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retourne les donnees gardees et les trois rapports d'audit."""
    missing_columns = [
        column for column in (TEXT_COL, LABEL_COL) if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing_columns)
            + f". Colonnes trouvees : {list(dataframe.columns)}"
        )

    df = dataframe.copy()
    df["_source_excel_row"] = df.index + 2
    df["_note_key"] = df[TEXT_COL].apply(normalize_note)
    numeric_labels = pd.to_numeric(df[LABEL_COL], errors="coerce")

    valid_mask = df["_note_key"].ne("") & numeric_labels.isin([0, 1])
    invalid_rows = df.loc[~valid_mask].copy()

    df = df.loc[valid_mask].copy()
    df[LABEL_COL] = numeric_labels.loc[valid_mask].astype(int)

    label_count_by_note = df.groupby("_note_key")[LABEL_COL].nunique()
    conflicting_keys = set(label_count_by_note[label_count_by_note > 1].index)

    conflicting_rows = df[df["_note_key"].isin(conflicting_keys)].copy()
    consistent_rows = df[~df["_note_key"].isin(conflicting_keys)].copy()

    first_source_row = (
        consistent_rows.drop_duplicates("_note_key", keep="first")
        .set_index("_note_key")["_source_excel_row"]
    )
    duplicate_mask = consistent_rows.duplicated("_note_key", keep="first")
    removed_duplicates = consistent_rows.loc[duplicate_mask].copy()
    removed_duplicates["_kept_excel_row"] = removed_duplicates["_note_key"].map(
        first_source_row
    )

    deduplicated = consistent_rows.loc[~duplicate_mask].copy()
    if deduplicated["_note_key"].duplicated().any():
        raise RuntimeError("La deduplication a echoue : des notes sont encore repetees.")

    helper_columns = ["_source_excel_row", "_note_key"]
    deduplicated = deduplicated.drop(columns=helper_columns).reset_index(drop=True)

    return deduplicated, conflicting_rows, removed_duplicates, invalid_rows


def save_audit_report(dataframe: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un rapport lisible en retirant la cle normalisee interne."""
    report = dataframe.drop(columns=["_note_key"], errors="ignore")
    report.to_excel(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supprime les doublons et exclut les notes aux labels contradictoires."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    audit_dir = args.audit_dir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier d'entree introuvable : {input_path}")
    if input_path == output_path:
        raise ValueError("Le fichier de sortie doit etre different du fichier d'entree.")

    source = pd.read_excel(input_path)
    deduplicated, conflicts, removed_duplicates, invalid_rows = (
        deduplicate_dataframe(source)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    deduplicated.to_excel(output_path, index=False)
    save_audit_report(conflicts, audit_dir / "conflicting_labels.xlsx")
    save_audit_report(
        removed_duplicates,
        audit_dir / "removed_same_label_duplicates.xlsx",
    )
    save_audit_report(invalid_rows, audit_dir / "invalid_rows.xlsx")

    conflicting_note_count = conflicts["_note_key"].nunique()

    print(f"Fichier d'entree : {input_path}")
    print(f"Lignes initiales : {len(source)}")
    print(f"Lignes invalides exclues : {len(invalid_rows)}")
    print(
        "Notes aux labels contradictoires exclues : "
        f"{conflicting_note_count} notes, {len(conflicts)} lignes"
    )
    print(f"Doublons avec le meme label supprimes : {len(removed_duplicates)}")
    print(f"Lignes finales : {len(deduplicated)}")
    print(f"Fichier sans doublons : {output_path}")
    print(f"Rapports d'audit : {audit_dir}")


if __name__ == "__main__":
    main()
