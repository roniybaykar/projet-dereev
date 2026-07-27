"""Protocole A strict : un seul découpage CV fixe, partagé partout.

La seed ``CV_SPLIT_SEED`` crée une unique affectation des observations aux
folds. Le manifeste est ensuite rechargé à l'identique pour :

- toutes les seeds d'entraînement ;
- tous les modèles du benchmark ;
- la baseline TF-IDF + régression logistique.

Un fingerprint du contenu et de l'ordre de ``train_cv.csv`` empêche de
réutiliser silencieusement le manifeste après une modification du dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class SharedSplit:
    """Description de l'unique affectation de folds officielle."""

    split_seed: int
    fold_count: int
    assignments: np.ndarray
    csv_path: str
    metadata_path: str
    dataset_fingerprint: str
    was_created: bool

    def indices(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Retourne les couples (indices train, indices validation)."""
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []

        for fold_index in range(1, self.fold_count + 1):
            validation_indices = np.flatnonzero(
                self.assignments == fold_index
            )
            training_indices = np.flatnonzero(
                self.assignments != fold_index
            )
            pairs.append((training_indices, validation_indices))

        return pairs


def parse_seed_list(value: str | Sequence[int]) -> List[int]:
    """Convertit ``42,123,...`` ou une séquence en liste de seeds uniques."""
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
        seeds = [int(part) for part in raw_values if part]
    else:
        seeds = [int(seed) for seed in value]

    if not seeds:
        raise ValueError("La liste des seeds ne peut pas être vide.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("La liste des seeds contient des doublons.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Les seeds doivent être positives ou nulles.")

    return seeds


def dataset_fingerprint(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> str:
    """Calcule une empreinte stable du contenu et de l'ordre du dataset."""
    missing = [
        column
        for column in (text_column, label_column)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            "Colonnes manquantes pour calculer le fingerprint : "
            + ", ".join(missing)
        )

    normalized = dataframe[[text_column, label_column]].copy()
    normalized[text_column] = normalized[text_column].fillna("").astype(str)
    normalized[label_column] = normalized[label_column].astype(int)
    normalized = normalized.reset_index(drop=True)

    row_hashes = pd.util.hash_pandas_object(
        normalized,
        index=True,
        categorize=False,
    ).to_numpy(dtype=np.uint64)

    digest = hashlib.sha256()
    digest.update(row_hashes.tobytes())
    digest.update(str(len(normalized)).encode("utf-8"))
    digest.update(text_column.encode("utf-8"))
    digest.update(label_column.encode("utf-8"))
    return digest.hexdigest()


def _split_paths(
    split_directory: str,
    split_seed: int,
    fold_count: int,
) -> Tuple[str, str]:
    base_name = f"fixed_split_seed_{split_seed}_folds_{fold_count}"
    return (
        os.path.join(split_directory, f"{base_name}.csv"),
        os.path.join(split_directory, f"{base_name}.json"),
    )


def _validate_labels(labels: np.ndarray, fold_count: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)

    if labels.ndim != 1:
        raise ValueError("Les labels doivent être un vecteur 1D.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Les labels doivent uniquement contenir 0 et 1.")

    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts < fold_count):
        raise ValueError(
            "Chaque classe doit contenir au moins autant d'observations "
            f"que de folds. Comptages : {class_counts.tolist()}, "
            f"folds : {fold_count}."
        )

    return labels


def _validate_assignments(
    assignments: np.ndarray,
    labels: np.ndarray,
    fold_count: int,
) -> None:
    assignments = np.asarray(assignments, dtype=int)

    if assignments.shape != labels.shape:
        raise ValueError(
            "Le manifeste de folds n'a pas la même taille que le dataset."
        )

    expected_folds = set(range(1, fold_count + 1))
    found_folds = set(np.unique(assignments).tolist())
    if found_folds != expected_folds:
        raise ValueError(
            "Folds invalides dans le manifeste. "
            f"Attendus : {sorted(expected_folds)}, "
            f"trouvés : {sorted(found_folds)}."
        )

    for fold_index in range(1, fold_count + 1):
        fold_labels = labels[assignments == fold_index]
        if np.unique(fold_labels).size != 2:
            raise ValueError(
                f"Le fold {fold_index} ne contient pas les deux classes."
            )


def _create_assignments(
    labels: np.ndarray,
    split_seed: int,
    fold_count: int,
) -> np.ndarray:
    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=split_seed,
    )

    assignments = np.full(len(labels), -1, dtype=int)

    for fold_index, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        assignments[validation_indices] = fold_index

    if np.any(assignments == -1):
        raise RuntimeError(
            "Certaines observations n'ont reçu aucune affectation de fold."
        )

    _validate_assignments(assignments, labels, fold_count)
    return assignments


def ensure_shared_split(
    labels: Iterable[int],
    split_seed: int,
    fold_count: int,
    split_directory: str,
    fingerprint: str,
) -> SharedSplit:
    """Crée ou recharge l'unique manifeste officiel du benchmark."""
    if fold_count < 2:
        raise ValueError("Le nombre de folds doit être supérieur ou égal à 2.")

    labels_array = _validate_labels(
        np.asarray(list(labels), dtype=int),
        fold_count,
    )

    os.makedirs(split_directory, exist_ok=True)
    csv_path, metadata_path = _split_paths(
        split_directory=split_directory,
        split_seed=split_seed,
        fold_count=fold_count,
    )

    csv_exists = os.path.exists(csv_path)
    metadata_exists = os.path.exists(metadata_path)

    if csv_exists != metadata_exists:
        raise RuntimeError(
            "Manifeste de folds incomplet : le CSV et son JSON doivent "
            f"exister ensemble. CSV={csv_path}, JSON={metadata_path}"
        )

    was_created = not csv_exists

    if csv_exists:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        expected_metadata = {
            "split_seed": int(split_seed),
            "fold_count": int(fold_count),
            "row_count": int(len(labels_array)),
            "dataset_fingerprint": fingerprint,
        }

        for key, expected_value in expected_metadata.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise RuntimeError(
                    "Le manifeste fixe ne correspond pas au dataset actuel "
                    f"({key}: attendu={expected_value!r}, "
                    f"trouvé={actual_value!r}). Supprimez ce manifeste "
                    "uniquement si train_cv.csv a volontairement changé."
                )

        manifest = pd.read_csv(csv_path)
        expected_columns = ["row_index", "label", "fold"]
        if list(manifest.columns) != expected_columns:
            raise RuntimeError(
                "Colonnes invalides dans le manifeste de folds. "
                f"Attendues : {expected_columns}."
            )

        expected_indices = np.arange(len(labels_array), dtype=int)
        row_indices = manifest["row_index"].to_numpy(dtype=int)
        stored_labels = manifest["label"].to_numpy(dtype=int)
        assignments = manifest["fold"].to_numpy(dtype=int)

        if not np.array_equal(row_indices, expected_indices):
            raise RuntimeError(
                "L'ordre des lignes du manifeste ne correspond pas à "
                "train_cv.csv."
            )
        if not np.array_equal(stored_labels, labels_array):
            raise RuntimeError(
                "Les labels du manifeste diffèrent de ceux de train_cv.csv."
            )

        _validate_assignments(assignments, labels_array, fold_count)
    else:
        assignments = _create_assignments(
            labels=labels_array,
            split_seed=split_seed,
            fold_count=fold_count,
        )

        pd.DataFrame(
            {
                "row_index": np.arange(len(labels_array), dtype=int),
                "label": labels_array,
                "fold": assignments,
            }
        ).to_csv(csv_path, index=False)

        fold_summaries = []
        for fold_index in range(1, fold_count + 1):
            mask = assignments == fold_index
            fold_labels = labels_array[mask]
            counts = np.bincount(fold_labels, minlength=2)
            fold_summaries.append(
                {
                    "fold": fold_index,
                    "validation_size": int(mask.sum()),
                    "class_0": int(counts[0]),
                    "class_1": int(counts[1]),
                }
            )

        metadata = {
            "protocol": "fixed_shared_stratified_kfold_multi_training_seeds",
            "split_seed": int(split_seed),
            "fold_count": int(fold_count),
            "row_count": int(len(labels_array)),
            "dataset_fingerprint": fingerprint,
            "class_counts": {
                "0": int(np.sum(labels_array == 0)),
                "1": int(np.sum(labels_array == 1)),
            },
            "folds": fold_summaries,
        }

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    return SharedSplit(
        split_seed=int(split_seed),
        fold_count=int(fold_count),
        assignments=np.asarray(assignments, dtype=int),
        csv_path=csv_path,
        metadata_path=metadata_path,
        dataset_fingerprint=fingerprint,
        was_created=was_created,
    )