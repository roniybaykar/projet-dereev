"""Préparation anti-fuite du dataset DEREEV sans identifiant patient.

Le script détecte de manière conservatrice les doublons et quasi-doublons,
forme des groupes documentaires par graphe de similarité, puis crée uniquement
le split externe ``train_cv.csv`` / ``test.csv`` au niveau de ces groupes.

Les cinq folds ne sont pas créés ici. Ils sont construits ensuite par
``src.cv_protocol`` avec ``StratifiedGroupKFold`` à partir de la colonne
``leakage_group_id`` sauvegardée dans ``train_cv.csv``.

Étapes recommandées :

    python3 -m src.prepare_data --stage review

Compléter ensuite la colonne ``decision`` de :

    data/processed/leakage_audit/pairs_to_review.csv

Puis finaliser :

    python3 -m src.prepare_data \
        --stage finalize \
        --review-file data/processed/leakage_audit/pairs_to_review.csv \
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from src.config import (
    DATA_PROCESSED_DIR,
    GROUP_COL,
    LABEL_COL,
    RANDOM_SEED,
    RAW_XLSX,
    TEST_SIZE,
    TEXT_COL,
)
from src.preprocessing import clean_text, load_abreviations


LOGGER = logging.getLogger("dereev_group_safe")


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityConfig:
    """Paramètres du détecteur de quasi-doublons.

    Les seuils sont volontairement conservateurs. Le label n'est jamais
    utilisé pour créer les groupes.
    """

    top_k_neighbors: int = 30
    candidate_char_similarity: float = 0.60

    # Les documents très courts sont facilement génériques (« RAS », etc.).
    short_document_chars: int = 120
    minimum_auto_document_chars: int = 40

    # Garde-fou contre l'effet de chaîne des composantes connexes.
    maximum_group_size: int = 30

    # Similarité utilisée pour l'audit final train/test et folds.
    residual_audit_similarity: float = 0.72

    # Un token présent dans au plus 1 % des documents est considéré rare,
    # avec un plancher/plafond raisonnable pour les petits corpus.
    rare_token_max_fraction: float = 0.01
    rare_token_min_df: int = 2
    rare_token_max_df_cap: int = 30


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = TEST_SIZE
    holdout_seed: int = RANDOM_SEED
    holdout_candidates: int = 2500


@dataclass
class PairFeatures:
    left_index: int
    right_index: int
    char_similarity: float
    word_similarity: float
    shingle_containment: float
    shingle_jaccard: float
    shared_rare_tokens: int
    length_ratio: float
    same_source: bool
    same_header_author: bool
    header_days_apart: Optional[int]
    minimum_chars: int
    left_generic_short: bool
    right_generic_short: bool
    classification: str = "ignore"
    reason: str = ""


# ---------------------------------------------------------------------------
# NORMALISATION DU TEXTE
# ---------------------------------------------------------------------------


HEADER_RE = re.compile(
    r"^\s*(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})[_\s-]+"
    r"(?P<author>[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
    r"\s+[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+)\s*",
    flags=re.IGNORECASE,
)

DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?"
    r"|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)
TIME_RE = re.compile(r"\b\d{1,2}\s*[h:]\s*\d{0,2}\b", re.IGNORECASE)
DAY_COUNTER_RE = re.compile(r"\bj\s*\d+\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]{2,}\b")

GENERIC_CLINICAL_TOKENS = {
    "patient",
    "patiente",
    "stable",
    "ras",
    "jour",
    "aujourd",
    "hui",
    "hier",
    "demain",
    "bien",
    "clinique",
    "cliniquement",
    "afebrile",
    "apyretique",
    "surveillance",
    "sortie",
    "transfert",
    "traitement",
    "poursuite",
    "examen",
    "sans",
    "signe",
    "plan",
    "general",
    "generale",
    "bon",
    "bonne",
    "etat",
    "prise",
    "charge",
    "hospitalisation",
    "hospitalise",
    "hospitalisee",
    "ce",
    "de",
    "des",
    "du",
    "la",
    "le",
    "les",
    "un",
    "une",
    "et",
    "en",
    "pour",
    "avec",
    "sur",
    "dans",
}


@dataclass(frozen=True)
class HeaderMetadata:
    note_date: Optional[date]
    author: Optional[str]


def fold_accents(value: str) -> str:
    """Retire uniquement les diacritiques pour la comparaison."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )


def extract_header_metadata(raw_text: str) -> HeaderMetadata:
    """Extrait prudemment la date et le rédacteur du préfixe éventuel.

    Ces informations sont des signaux faibles et ne sont jamais utilisées
    seules pour regrouper deux notes.
    """
    normalized = unicodedata.normalize("NFKC", raw_text or "")
    normalized = normalized.replace("_x000D_", " ").replace("_x000d_", " ")
    match = HEADER_RE.match(normalized)
    if match is None:
        return HeaderMetadata(note_date=None, author=None)

    parsed_date: Optional[date] = None
    for date_format in ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d-%m-%Y"):
        try:
            parsed_date = datetime.strptime(
                match.group("date"), date_format
            ).date()
            break
        except ValueError:
            continue

    author = fold_accents(match.group("author").lower()).strip()
    return HeaderMetadata(note_date=parsed_date, author=author or None)


def strip_header_metadata(raw_text: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw_text or "")
    normalized = normalized.replace("_x000D_", " ").replace("_x000d_", " ")
    return HEADER_RE.sub("", normalized, count=1)


def normalize_surface(raw_text: str) -> str:
    """Vue textuelle pour comparer le contenu en retirant le préfixe auteur."""
    value = strip_header_metadata(raw_text)
    value = value.replace("\r", " ").replace("\n", " ")
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    value = fold_accents(value.lower())
    value = re.sub(r"[^a-z0-9%/+.,'\- ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_masked(raw_text: str) -> str:
    """Vue où les valeurs variables sont masquées pour détecter le copy-forward."""
    value = normalize_surface(raw_text)
    value = DATE_RE.sub(" <date> ", value)
    value = TIME_RE.sub(" <heure> ", value)
    value = DAY_COUNTER_RE.sub(" <jour> ", value)
    value = NUMBER_RE.sub(" <num> ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_for_identity(raw_text: str) -> str:
    """Normalisation légère et stable pour l'identifiant documentaire."""
    value = unicodedata.normalize("NFKC", raw_text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def stable_document_id(raw_text: str) -> str:
    digest = hashlib.sha256(
        normalize_for_identity(raw_text).encode("utf-8")
    ).hexdigest()
    return f"doc_{digest[:16]}"


def tokens(value: str) -> Set[str]:
    return set(TOKEN_RE.findall(value))


def word_shingles(value: str, size: int = 3) -> Set[Tuple[str, ...]]:
    sequence = TOKEN_RE.findall(value)
    if len(sequence) < size:
        return {(token,) for token in sequence}
    return {
        tuple(sequence[index : index + size])
        for index in range(len(sequence) - size + 1)
    }


# ---------------------------------------------------------------------------
# CHARGEMENT ET VALIDATION
# ---------------------------------------------------------------------------


def build_model_text(raw_text: str, abbreviation_dictionary: Dict) -> str:
    """Applique exactement le nettoyage historique utilisé par le pipeline."""
    cleaned = clean_text(raw_text, abbreviation_dictionary)
    if cleaned is None:
        return ""
    return str(cleaned).strip()


def load_dataset(
    input_path: Path,
    text_column: str,
    label_column: str,
    source_column: str,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    dataframe = pd.read_excel(input_path)
    required = [text_column, label_column]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(missing)
            + f". Colonnes trouvées : {list(dataframe.columns)}"
        )

    selected_columns = [text_column, label_column]
    if source_column in dataframe.columns:
        selected_columns.append(source_column)

    dataframe = dataframe[selected_columns].copy()
    dataframe["original_row"] = np.arange(len(dataframe), dtype=int)

    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)
    dataframe[label_column] = pd.to_numeric(
        dataframe[label_column], errors="raise"
    ).astype(int)

    invalid_labels = ~dataframe[label_column].isin([0, 1])
    if invalid_labels.any():
        raise ValueError(
            "La colonne de labels doit uniquement contenir 0 et 1. "
            f"Exemples invalides :\n{dataframe.loc[invalid_labels].head(10)}"
        )

    invalid_text_values = {"", "na", "n/a", "nan", "none", "null"}
    empty_mask = (
        dataframe[text_column]
        .str.strip()
        .str.lower()
        .isin(invalid_text_values)
    )
    if empty_mask.any():
        LOGGER.warning(
            "Suppression de %d note(s) vide(s) ou NA.", int(empty_mask.sum())
        )
        dataframe = dataframe.loc[~empty_mask].copy()

    if source_column not in dataframe.columns:
        dataframe[source_column] = "unknown"
    else:
        dataframe[source_column] = (
            dataframe[source_column].fillna("unknown").astype(str)
        )

    dataframe["identity_text"] = dataframe[text_column].map(
        normalize_for_identity
    )
    duplicate_mask = dataframe.duplicated("identity_text", keep="first")
    if duplicate_mask.any():
        LOGGER.warning(
            "Le fichier annoncé comme dédupliqué contient encore %d "
            "doublon(s) exact(s) après normalisation ; ils sont retirés.",
            int(duplicate_mask.sum()),
        )
        dataframe = dataframe.loc[~duplicate_mask].copy()

    dataframe = dataframe.reset_index(drop=True)
    dataframe["document_id"] = dataframe[text_column].map(stable_document_id)

    if dataframe["document_id"].duplicated().any():
        raise RuntimeError("Collision inattendue dans les document_id.")

    dataframe["surface_text"] = dataframe[text_column].map(normalize_surface)
    dataframe["masked_text"] = dataframe[text_column].map(normalize_masked)

    empty_after_normalization = dataframe["surface_text"].str.strip().eq("")
    if empty_after_normalization.any():
        LOGGER.warning(
            "Suppression de %d ligne(s) ne contenant qu'un préfixe date/auteur.",
            int(empty_after_normalization.sum()),
        )
        dataframe = dataframe.loc[~empty_after_normalization].copy().reset_index(drop=True)

    metadata = dataframe[text_column].map(extract_header_metadata)
    dataframe["header_date"] = [
        item.note_date.isoformat() if item.note_date else ""
        for item in metadata
    ]
    dataframe["header_author"] = [item.author or "" for item in metadata]

    return dataframe


# ---------------------------------------------------------------------------
# SIMILARITÉ ET GRAPHE
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.component_size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.component_size[left_root] < self.component_size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size[right_root]


def sparse_row_cosine(matrix, left: int, right: int) -> float:
    return float(matrix[left].multiply(matrix[right]).sum())


def generic_short_flags(
    surface_texts: Sequence[str],
    token_sets: Sequence[Set[str]],
    short_chars: int,
) -> List[bool]:
    result: List[bool] = []
    for text, token_set in zip(surface_texts, token_sets):
        distinctive = {
            token
            for token in token_set
            if token not in GENERIC_CLINICAL_TOKENS
        }
        result.append(len(text) < short_chars and len(distinctive) < 6)
    return result


def build_similarity_matrices(dataframe: pd.DataFrame):
    LOGGER.info("Vectorisation TF-IDF caractères...")
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 6),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    char_matrix = char_vectorizer.fit_transform(dataframe["masked_text"])

    LOGGER.info("Vectorisation TF-IDF mots/bigrammes...")
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=150_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    word_matrix = word_vectorizer.fit_transform(dataframe["surface_text"])

    LOGGER.info(
        "Matrices créées : caractères=%s, mots=%s",
        char_matrix.shape,
        word_matrix.shape,
    )
    return char_matrix, word_matrix


def prepare_lexical_features(
    dataframe: pd.DataFrame,
    config: SimilarityConfig,
):
    token_sets = [tokens(value) for value in dataframe["surface_text"]]
    document_frequencies = Counter(
        token for token_set in token_sets for token in token_set
    )
    rare_max_df = min(
        config.rare_token_max_df_cap,
        max(
            config.rare_token_min_df,
            int(math.ceil(len(dataframe) * config.rare_token_max_fraction)),
        ),
    )
    rare_sets = [
        {
            token
            for token in token_set
            if config.rare_token_min_df
            <= document_frequencies[token]
            <= rare_max_df
        }
        for token_set in token_sets
    ]
    shingle_sets = [
        word_shingles(value, size=3) for value in dataframe["surface_text"]
    ]
    generic_flags = generic_short_flags(
        surface_texts=dataframe["surface_text"].tolist(),
        token_sets=token_sets,
        short_chars=config.short_document_chars,
    )
    return token_sets, rare_sets, shingle_sets, generic_flags, rare_max_df


def header_distance_days(
    left_date: str,
    right_date: str,
) -> Optional[int]:
    if not left_date or not right_date:
        return None
    try:
        return abs(
            (
                datetime.strptime(left_date, "%Y-%m-%d").date()
                - datetime.strptime(right_date, "%Y-%m-%d").date()
            ).days
        )
    except ValueError:
        return None


def classify_pair(
    feature: PairFeatures,
    left_surface: str,
    right_surface: str,
    config: SimilarityConfig,
) -> Tuple[str, str]:
    """Classe une paire sans jamais consulter son label."""
    char_similarity = feature.char_similarity
    word_similarity = feature.word_similarity
    containment = feature.shingle_containment
    rare = feature.shared_rare_tokens
    minimum_chars = feature.minimum_chars

    if (
        left_surface == right_surface
        and minimum_chars >= config.minimum_auto_document_chars
        and not feature.left_generic_short
        and not feature.right_generic_short
    ):
        return "auto", "contenu normalisé identique"

    if minimum_chars < config.short_document_chars:
        if (
            char_similarity >= 0.95
            and word_similarity >= 0.90
            and rare >= 2
            and not feature.left_generic_short
            and not feature.right_generic_short
        ):
            return "auto", "courtes notes très fortement similaires"
        if (
            char_similarity >= 0.75
            and not feature.left_generic_short
            and not feature.right_generic_short
        ):
            return "review", "courtes notes : validation manuelle requise"
        return "ignore", "notes courtes ou génériques"

    # Les rapprochements entre sources différentes demandent un niveau de
    # preuve supérieur afin d'éviter de regrouper de simples gabarits.
    if not feature.same_source:
        if (
            char_similarity >= 0.92
            and word_similarity >= 0.82
            and containment >= 0.70
            and rare >= 6
        ):
            return "auto", "quasi-doublon très fort entre sources"
        if (
            char_similarity >= 0.82
            and word_similarity >= 0.72
            and containment >= 0.60
            and rare >= 8
        ):
            return "review", "similarité inter-source à confirmer"
        return "ignore", "preuve inter-source insuffisante"

    # Règles haute confiance pour les notes de même source.
    if (
        char_similarity >= 0.90
        and containment >= 0.65
        and word_similarity >= 0.65
        and rare >= 3
    ):
        return "auto", "très forte similarité de contenu"

    if (
        char_similarity >= 0.86
        and word_similarity >= 0.70
        and rare >= 5
    ):
        return "auto", "forte similarité caractères + mots"

    if (
        char_similarity >= 0.78
        and (word_similarity >= 0.70 or containment >= 0.65)
        and rare >= 8
    ):
        return "auto", "copy-forward probable"

    if (
        minimum_chars >= 250
        and char_similarity >= 0.70
        and word_similarity >= 0.65
        and (containment >= 0.70 or rare >= 20)
    ):
        return "auto", "longues notes avec contenu distinctif partagé"

    if (
        minimum_chars >= 250
        and feature.same_header_author
        and feature.header_days_apart is not None
        and feature.header_days_apart <= 14
        and char_similarity >= 0.68
        and word_similarity >= 0.60
        and rare >= 12
    ):
        return "auto", "signal faible auteur/date + forte proximité textuelle"

    if (
        minimum_chars >= 250
        and char_similarity >= 0.70
        and word_similarity >= 0.60
        and containment >= 0.68
        and rare >= 5
    ):
        return "auto", "longue note avec forte inclusion de passages"

    if (
        char_similarity >= 0.68
        and word_similarity >= 0.60
        and (containment >= 0.50 or rare >= 12)
    ):
        return "review", "paire ambiguë mais potentiellement apparentée"

    return "ignore", "similarité insuffisante"


def compute_candidate_pairs(
    dataframe: pd.DataFrame,
    char_matrix,
    word_matrix,
    config: SimilarityConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    (
        _token_sets,
        rare_sets,
        shingle_sets,
        generic_flags,
        rare_max_df,
    ) = prepare_lexical_features(dataframe, config)

    LOGGER.info("Seuil de rareté des tokens : df <= %d", rare_max_df)

    neighbor_count = min(config.top_k_neighbors, len(dataframe))
    nearest_neighbors = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nearest_neighbors.fit(char_matrix)
    distances, indices = nearest_neighbors.kneighbors(char_matrix)

    seen_pairs: Set[Tuple[int, int]] = set()
    auto_rows: List[Dict] = []
    review_rows: List[Dict] = []

    for left_index in range(len(dataframe)):
        for distance, raw_right_index in zip(
            distances[left_index, 1:], indices[left_index, 1:]
        ):
            right_index = int(raw_right_index)
            pair = tuple(sorted((left_index, right_index)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            char_similarity = 1.0 - float(distance)
            if char_similarity < config.candidate_char_similarity:
                continue

            left_index, right_index = pair
            left_shingles = shingle_sets[left_index]
            right_shingles = shingle_sets[right_index]
            intersection_size = len(left_shingles & right_shingles)
            minimum_shingles = max(
                1, min(len(left_shingles), len(right_shingles))
            )
            union_shingles = max(
                1, len(left_shingles | right_shingles)
            )

            left_date = str(dataframe.at[left_index, "header_date"])
            right_date = str(dataframe.at[right_index, "header_date"])
            left_author = str(dataframe.at[left_index, "header_author"])
            right_author = str(dataframe.at[right_index, "header_author"])

            left_surface = str(dataframe.at[left_index, "surface_text"])
            right_surface = str(dataframe.at[right_index, "surface_text"])
            minimum_chars = min(len(left_surface), len(right_surface))
            maximum_chars = max(len(left_surface), len(right_surface), 1)

            feature = PairFeatures(
                left_index=left_index,
                right_index=right_index,
                char_similarity=char_similarity,
                word_similarity=sparse_row_cosine(
                    word_matrix, left_index, right_index
                ),
                shingle_containment=intersection_size / minimum_shingles,
                shingle_jaccard=intersection_size / union_shingles,
                shared_rare_tokens=len(
                    rare_sets[left_index] & rare_sets[right_index]
                ),
                length_ratio=minimum_chars / maximum_chars,
                same_source=(
                    str(dataframe.at[left_index, "src"])
                    == str(dataframe.at[right_index, "src"])
                ),
                same_header_author=(
                    bool(left_author)
                    and left_author == right_author
                ),
                header_days_apart=header_distance_days(
                    left_date, right_date
                ),
                minimum_chars=minimum_chars,
                left_generic_short=generic_flags[left_index],
                right_generic_short=generic_flags[right_index],
            )

            classification, reason = classify_pair(
                feature=feature,
                left_surface=left_surface,
                right_surface=right_surface,
                config=config,
            )
            feature.classification = classification
            feature.reason = reason

            if classification == "ignore":
                continue

            row = pair_to_output_row(dataframe, feature)
            if classification == "auto":
                auto_rows.append(row)
            else:
                row["decision"] = ""
                row["suggested_decision"] = suggest_review_decision(feature)
                review_rows.append(row)

    auto_dataframe = pd.DataFrame(auto_rows)
    review_dataframe = pd.DataFrame(review_rows)

    sort_columns = [
        "char_similarity",
        "word_similarity",
        "shingle_containment",
    ]
    if not auto_dataframe.empty:
        auto_dataframe = auto_dataframe.sort_values(
            sort_columns, ascending=False
        ).reset_index(drop=True)
    if not review_dataframe.empty:
        review_dataframe = review_dataframe.sort_values(
            sort_columns, ascending=False
        ).reset_index(drop=True)

    LOGGER.info(
        "Paires automatiques : %d | paires à revoir : %d",
        len(auto_dataframe),
        len(review_dataframe),
    )
    return auto_dataframe, review_dataframe


def suggest_review_decision(feature: PairFeatures) -> str:
    """Suggestion uniquement informative ; elle n'est jamais appliquée seule."""
    if (
        feature.same_header_author
        and feature.header_days_apart is not None
        and feature.header_days_apart <= 14
        and feature.char_similarity >= 0.68
        and feature.word_similarity >= 0.60
        and feature.shingle_containment >= 0.55
    ):
        return "merge"
    if (
        feature.char_similarity >= 0.70
        and feature.word_similarity >= 0.62
        and (
            feature.shingle_containment >= 0.62
            or feature.shared_rare_tokens >= 15
        )
    ):
        return "merge"
    return "separate"


def pair_to_output_row(
    dataframe: pd.DataFrame,
    feature: PairFeatures,
) -> Dict:
    left = dataframe.iloc[feature.left_index]
    right = dataframe.iloc[feature.right_index]
    return {
        "classification": feature.classification,
        "reason": feature.reason,
        "decision": "auto_merge" if feature.classification == "auto" else "",
        "left_index": feature.left_index,
        "right_index": feature.right_index,
        "left_document_id": left["document_id"],
        "right_document_id": right["document_id"],
        "left_original_row": int(left["original_row"]),
        "right_original_row": int(right["original_row"]),
        "left_label": int(left["reev_classe"]),
        "right_label": int(right["reev_classe"]),
        "left_source": left["src"],
        "right_source": right["src"],
        "char_similarity": round(feature.char_similarity, 6),
        "word_similarity": round(feature.word_similarity, 6),
        "shingle_containment": round(feature.shingle_containment, 6),
        "shingle_jaccard": round(feature.shingle_jaccard, 6),
        "shared_rare_tokens": feature.shared_rare_tokens,
        "length_ratio": round(feature.length_ratio, 6),
        "same_source": feature.same_source,
        "same_header_author": feature.same_header_author,
        "header_days_apart": feature.header_days_apart,
        "left_text": left["cn_notes"],
        "right_text": right["cn_notes"],
    }


def read_manual_merge_edges(
    review_path: Optional[Path],
    allow_unreviewed: bool,
) -> List[Tuple[str, str]]:
    if review_path is None:
        if allow_unreviewed:
            return []
        raise ValueError(
            "L'étape finalize exige --review-file, sauf si vous utilisez "
            "explicitement --allow-unreviewed."
        )
    if not review_path.exists():
        raise FileNotFoundError(f"Fichier de revue introuvable : {review_path}")

    review = pd.read_csv(review_path)
    required = ["left_document_id", "right_document_id", "decision"]
    missing = [column for column in required if column not in review.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le fichier de revue : "
            + ", ".join(missing)
        )

    decisions = review["decision"].fillna("").astype(str).str.strip().str.lower()
    allowed = {"", "merge", "separate", "auto_merge"}
    invalid = ~decisions.isin(allowed)
    if invalid.any():
        raise ValueError(
            "Décisions invalides. Valeurs autorisées : merge, separate. "
            f"Exemples :\n{review.loc[invalid].head(10)}"
        )

    unreviewed = decisions.eq("")
    if unreviewed.any() and not allow_unreviewed:
        raise RuntimeError(
            f"{int(unreviewed.sum())} paire(s) n'ont pas encore été revues. "
            "Renseignez 'merge' ou 'separate' dans la colonne decision, "
            "ou ajoutez --allow-unreviewed."
        )

    merge_rows = review.loc[decisions.eq("merge")]
    return list(
        zip(
            merge_rows["left_document_id"].astype(str),
            merge_rows["right_document_id"].astype(str),
        )
    )


def create_groups(
    dataframe: pd.DataFrame,
    auto_pairs: pd.DataFrame,
    manual_merge_edges: Sequence[Tuple[str, str]],
    config: SimilarityConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    document_to_index = {
        document_id: index
        for index, document_id in enumerate(dataframe["document_id"])
    }
    union_find = UnionFind(len(dataframe))

    if not auto_pairs.empty:
        for row in auto_pairs.itertuples(index=False):
            union_find.union(int(row.left_index), int(row.right_index))

    for left_document_id, right_document_id in manual_merge_edges:
        if left_document_id not in document_to_index:
            raise ValueError(f"document_id inconnu : {left_document_id}")
        if right_document_id not in document_to_index:
            raise ValueError(f"document_id inconnu : {right_document_id}")
        union_find.union(
            document_to_index[left_document_id],
            document_to_index[right_document_id],
        )

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(dataframe)):
        components[union_find.find(index)].append(index)

    largest_group = max(len(indices) for indices in components.values())
    if largest_group > config.maximum_group_size:
        raise RuntimeError(
            "Un groupe automatique contient "
            f"{largest_group} notes, au-delà du garde-fou "
            f"({config.maximum_group_size}). Vérifiez les seuils et les "
            "paires de revue avant de poursuivre."
        )

    group_ids: Dict[int, str] = {}
    group_rows: List[Dict] = []

    for indices in components.values():
        sorted_document_ids = sorted(
            dataframe.iloc[indices]["document_id"].astype(str).tolist()
        )
        digest = hashlib.sha256(
            "|".join(sorted_document_ids).encode("utf-8")
        ).hexdigest()[:14]
        group_id = f"grp_{digest}"

        labels = dataframe.iloc[indices]["reev_classe"].astype(int)
        sources = sorted(
            dataframe.iloc[indices]["src"].astype(str).unique().tolist()
        )
        for index in indices:
            group_ids[index] = group_id

        group_rows.append(
            {
                "leakage_group_id": group_id,
                "group_size": len(indices),
                "class_0": int((labels == 0).sum()),
                "class_1": int((labels == 1).sum()),
                "mixed_labels": bool(labels.nunique() > 1),
                "sources": "|".join(sources),
                "document_ids": "|".join(sorted_document_ids),
                "original_rows": "|".join(
                    map(
                        str,
                        sorted(
                            dataframe.iloc[indices]["original_row"]
                            .astype(int)
                            .tolist()
                        ),
                    )
                ),
            }
        )

    grouped = dataframe.copy()
    grouped["leakage_group_id"] = [group_ids[index] for index in range(len(grouped))]
    group_summary = pd.DataFrame(group_rows).sort_values(
        ["group_size", "leakage_group_id"], ascending=[False, True]
    )
    return grouped, group_summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SPLITS GROUP-AWARE
# ---------------------------------------------------------------------------


def source_balance_penalty(
    dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    source_column: str,
) -> float:
    overall = dataframe[source_column].value_counts(normalize=True)
    train = dataframe.iloc[train_indices][source_column].value_counts(normalize=True)
    test = dataframe.iloc[test_indices][source_column].value_counts(normalize=True)
    categories = overall.index.union(train.index).union(test.index)
    return float(
        sum(
            abs(float(train.get(category, 0.0)) - float(overall.get(category, 0.0)))
            + abs(float(test.get(category, 0.0)) - float(overall.get(category, 0.0)))
            for category in categories
        )
        / max(1, len(categories))
    )


def select_group_holdout(
    dataframe: pd.DataFrame,
    label_column: str,
    group_column: str,
    source_column: str,
    config: SplitConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    labels = dataframe[label_column].to_numpy(dtype=int)
    groups = dataframe[group_column].astype(str).to_numpy()
    overall_positive_rate = float(labels.mean())

    splitter = GroupShuffleSplit(
        n_splits=config.holdout_candidates,
        test_size=config.test_size,
        random_state=config.holdout_seed,
    )

    best: Optional[Tuple[float, np.ndarray, np.ndarray, Dict]] = None
    dummy_features = np.zeros((len(dataframe), 1), dtype=np.float32)

    for train_indices, test_indices in splitter.split(
        dummy_features, labels, groups
    ):
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        if np.unique(train_labels).size != 2 or np.unique(test_labels).size != 2:
            continue

        test_fraction = len(test_indices) / len(dataframe)
        train_positive_rate = float(train_labels.mean())
        test_positive_rate = float(test_labels.mean())

        size_penalty = abs(test_fraction - config.test_size) / config.test_size
        label_penalty = (
            abs(test_positive_rate - overall_positive_rate)
            + 0.5 * abs(train_positive_rate - overall_positive_rate)
        )
        source_penalty = source_balance_penalty(
            dataframe,
            train_indices,
            test_indices,
            source_column,
        )

        objective = size_penalty + 3.0 * label_penalty + 0.35 * source_penalty
        details = {
            "objective": float(objective),
            "test_fraction": float(test_fraction),
            "overall_positive_rate": overall_positive_rate,
            "train_positive_rate": train_positive_rate,
            "test_positive_rate": test_positive_rate,
            "source_penalty": source_penalty,
        }

        if best is None or objective < best[0]:
            best = (objective, train_indices, test_indices, details)

    if best is None:
        raise RuntimeError(
            "Impossible de construire un holdout group-aware contenant les deux classes."
        )

    _, train_indices, test_indices, details = best
    return train_indices, test_indices, details


def validate_no_group_overlap(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    group_column: str,
) -> None:
    overlap = set(train_dataframe[group_column]) & set(test_dataframe[group_column])
    if overlap:
        raise RuntimeError(
            f"Fuite de groupes entre train et test : {len(overlap)} groupe(s)."
        )


def audit_residual_cross_split_pairs(
    grouped_dataframe: pd.DataFrame,
    char_matrix,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    split_name: str,
    threshold: float,
) -> pd.DataFrame:
    """Signale les plus fortes similarités qui traversent une frontière.

    Ce contrôle ne fusionne pas automatiquement des textes génériques. Il crée
    un rapport permettant de vérifier les éventuels faux négatifs du graphe.
    """
    if len(left_indices) == 0 or len(right_indices) == 0:
        return pd.DataFrame()

    similarity_matrix = char_matrix[left_indices] @ char_matrix[right_indices].T
    rows: List[Dict] = []

    for local_left, global_left in enumerate(left_indices):
        row = similarity_matrix.getrow(local_left)
        if row.nnz == 0:
            continue
        best_position = int(np.argmax(row.data))
        similarity = float(row.data[best_position])
        if similarity < threshold:
            continue
        local_right = int(row.indices[best_position])
        global_right = int(right_indices[local_right])

        if (
            grouped_dataframe.at[global_left, "leakage_group_id"]
            == grouped_dataframe.at[global_right, "leakage_group_id"]
        ):
            continue

        rows.append(
            {
                "boundary": split_name,
                "similarity": round(similarity, 6),
                "left_document_id": grouped_dataframe.at[
                    global_left, "document_id"
                ],
                "right_document_id": grouped_dataframe.at[
                    global_right, "document_id"
                ],
                "left_group": grouped_dataframe.at[
                    global_left, "leakage_group_id"
                ],
                "right_group": grouped_dataframe.at[
                    global_right, "leakage_group_id"
                ],
                "left_label": int(
                    grouped_dataframe.at[global_left, "reev_classe"]
                ),
                "right_label": int(
                    grouped_dataframe.at[global_right, "reev_classe"]
                ),
                "left_text": grouped_dataframe.at[global_left, "cn_notes"],
                "right_text": grouped_dataframe.at[global_right, "cn_notes"],
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("similarity", ascending=False)


# ---------------------------------------------------------------------------
# SAUVEGARDE
# ---------------------------------------------------------------------------


def make_backup(paths: Sequence[Path], backup_root: Path) -> Optional[Path]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = backup_root / f"before_group_safe_split_{timestamp}"
    destination.mkdir(parents=True, exist_ok=False)

    for path in existing:
        if path.is_dir():
            shutil.copytree(path, destination / path.name)
        else:
            shutil.copy2(path, destination / path.name)
    return destination


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_review_outputs(
    audit_dir: Path,
    dataframe: pd.DataFrame,
    auto_pairs: pd.DataFrame,
    review_pairs: pd.DataFrame,
    group_summary: pd.DataFrame,
    similarity_config: SimilarityConfig,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    auto_pairs.to_csv(audit_dir / "auto_merge_pairs.csv", index=False)
    review_pairs.to_csv(audit_dir / "pairs_to_review.csv", index=False)
    group_summary.to_csv(audit_dir / "preliminary_group_summary.csv", index=False)

    audit_columns = [
        "document_id",
        "original_row",
        "cn_notes",
        "model_text",
        "reev_classe",
        "src",
        "leakage_group_id",
        "header_date",
        "header_author",
    ]
    dataframe[audit_columns].to_csv(
        audit_dir / "preliminary_grouped_dataset.csv", index=False
    )

    write_json(
        audit_dir / "grouping_configuration.json",
        {
            "similarity_config": asdict(similarity_config),
            "documents": int(len(dataframe)),
            "auto_merge_pairs": int(len(auto_pairs)),
            "pairs_to_review": int(len(review_pairs)),
            "groups": int(dataframe["leakage_group_id"].nunique()),
            "non_singleton_groups": int(
                (group_summary["group_size"] > 1).sum()
            ),
            "largest_group": int(group_summary["group_size"].max()),
            "important_limitation": (
                "Les groupes représentent des documents fortement apparentés, "
                "pas des identifiants patients certains."
            ),
        },
    )


def save_final_outputs(
    grouped_dataframe: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    holdout_details: Dict,
    output_dir: Path,
    audit_dir: Path,
    text_column: str,
    label_column: str,
    group_column: str,
    split_config: SplitConfig,
    residual_pairs: pd.DataFrame,
) -> Dict[str, Path]:
    train_full = grouped_dataframe.iloc[train_indices].copy()
    test_full = grouped_dataframe.iloc[test_indices].copy()

    # Préserver l'ordre d'origine garantit des sorties reproductibles.
    train_full = train_full.sort_values("original_row").reset_index(drop=True)
    test_full = test_full.sort_values("original_row").reset_index(drop=True)

    validate_no_group_overlap(train_full, test_full, group_column)

    if train_full["model_text"].str.strip().eq("").any():
        raise RuntimeError("Une note train est vide après clean_text.")
    if test_full["model_text"].str.strip().eq("").any():
        raise RuntimeError("Une note test est vide après clean_text.")

    train_csv = output_dir / "train_cv.csv"
    test_csv = output_dir / "test.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Les groupes sont conservés dans les CSV pour sécuriser le split externe
    # et permettre à cv_protocol.py de créer les folds group-aware. Le modèle
    # n'utilise que cn_notes et reev_classe (voir data_loader.py).
    train_output = pd.DataFrame(
        {
            text_column: train_full["model_text"].astype(str),
            label_column: train_full[label_column].astype(int),
            group_column: train_full[group_column].astype(str),
        }
    )
    test_output = pd.DataFrame(
        {
            text_column: test_full["model_text"].astype(str),
            label_column: test_full[label_column].astype(int),
            group_column: test_full[group_column].astype(str),
        }
    )

    if train_output[group_column].str.strip().eq("").any():
        raise RuntimeError("Un identifiant de groupe train est vide.")
    if test_output[group_column].str.strip().eq("").any():
        raise RuntimeError("Un identifiant de groupe test est vide.")

    train_output.to_csv(train_csv, index=False)
    test_output.to_csv(test_csv, index=False)

    audit_columns = [
        "document_id",
        "original_row",
        text_column,
        "model_text",
        label_column,
        "src",
        group_column,
        "header_date",
        "header_author",
    ]
    train_full[audit_columns].to_csv(
        audit_dir / "train_cv_with_groups.csv", index=False
    )
    test_full[audit_columns].to_csv(
        audit_dir / "test_with_groups.csv", index=False
    )
    grouped_dataframe[audit_columns].to_csv(
        audit_dir / "final_grouped_dataset.csv", index=False
    )
    residual_pairs.to_csv(
        audit_dir / "residual_train_test_pairs.csv", index=False
    )

    overlap = set(train_output[group_column]) & set(test_output[group_column])
    if overlap:
        raise RuntimeError(
            f"Fuite train/test après sauvegarde : {len(overlap)} groupe(s)."
        )

    split_summary = {
        "protocol": "group_aware_external_holdout_inferred_document_groups",
        "input_documents": int(len(grouped_dataframe)),
        "groups_total": int(grouped_dataframe[group_column].nunique()),
        "train_documents": int(len(train_full)),
        "test_documents": int(len(test_full)),
        "train_groups": int(train_full[group_column].nunique()),
        "test_groups": int(test_full[group_column].nunique()),
        "group_overlap_train_test": 0,
        "overall_positive_rate": float(grouped_dataframe[label_column].mean()),
        "train_positive_rate": float(train_full[label_column].mean()),
        "test_positive_rate": float(test_full[label_column].mean()),
        "test_size_requested": float(split_config.test_size),
        "holdout_seed": int(split_config.holdout_seed),
        "holdout_optimization": holdout_details,
        "residual_train_test_pairs_to_review": int(len(residual_pairs)),
        "outputs": {
            "train_cv": str(train_csv),
            "test": str(test_csv),
        },
        "next_step": (
            "Les folds seront créés par src.cv_protocol avec "
            "StratifiedGroupKFold lors du premier entraînement."
        ),
        "limitation": (
            "Les groupes sont inférés par similarité documentaire et ne "
            "constituent pas des identifiants patients certains."
        ),
    }
    write_json(audit_dir / "split_summary.json", split_summary)

    return {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "split_summary": audit_dir / "split_summary.json",
    }


# ---------------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regroupement documentaire conservateur et split train/test "
            "group-aware pour DEREEV."
        )
    )
    parser.add_argument("--input", type=Path, default=Path(RAW_XLSX))
    parser.add_argument("--output-dir", type=Path, default=Path(DATA_PROCESSED_DIR))
    parser.add_argument(
        "--stage",
        choices=["review", "finalize"],
        default="review",
        help="review crée les rapports ; finalize crée train_cv.csv et test.csv.",
    )
    parser.add_argument("--review-file", type=Path, default=None)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Finalise avec les groupes automatiques malgré les paires non revues.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Sauvegarde puis remplace train_cv.csv et test.csv. Les anciens "
            "manifests CV sont archivés puis supprimés car ils deviennent obsolètes."
        ),
    )
    parser.add_argument("--text-column", default=TEXT_COL)
    parser.add_argument("--label-column", default=LABEL_COL)
    parser.add_argument("--source-column", default="src")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--holdout-seed", type=int, default=RANDOM_SEED)
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(args: argparse.Namespace) -> None:
    configure_logging()

    if not 0.05 <= args.test_size <= 0.40:
        raise ValueError("--test-size doit être compris entre 0.05 et 0.40.")

    similarity_config = SimilarityConfig()
    split_config = SplitConfig(
        test_size=float(args.test_size),
        holdout_seed=int(args.holdout_seed),
    )

    audit_dir = args.output_dir / "leakage_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Chargement : %s", args.input)
    dataframe = load_dataset(
        input_path=args.input,
        text_column=args.text_column,
        label_column=args.label_column,
        source_column=args.source_column,
    )
    dataframe = dataframe.rename(
        columns={
            args.text_column: "cn_notes",
            args.label_column: "reev_classe",
            args.source_column: "src",
        }
    )

    LOGGER.info("Chargement du dictionnaire d'abréviations...")
    abbreviation_dictionary = load_abreviations()
    dataframe["model_text"] = dataframe["cn_notes"].map(
        lambda value: build_model_text(value, abbreviation_dictionary)
    )
    empty_after_cleaning = dataframe["model_text"].str.strip().eq("")
    if empty_after_cleaning.any():
        LOGGER.warning(
            "Suppression de %d note(s) vide(s) après clean_text.",
            int(empty_after_cleaning.sum()),
        )
        dataframe = dataframe.loc[~empty_after_cleaning].copy().reset_index(drop=True)

    LOGGER.info(
        "Dataset validé : %d notes | classe 0=%d | classe 1=%d",
        len(dataframe),
        int((dataframe["reev_classe"] == 0).sum()),
        int((dataframe["reev_classe"] == 1).sum()),
    )

    char_matrix, word_matrix = build_similarity_matrices(dataframe)
    auto_pairs, review_pairs = compute_candidate_pairs(
        dataframe=dataframe,
        char_matrix=char_matrix,
        word_matrix=word_matrix,
        config=similarity_config,
    )

    manual_edges: List[Tuple[str, str]] = []
    if args.stage == "finalize":
        manual_edges = read_manual_merge_edges(
            review_path=args.review_file,
            allow_unreviewed=args.allow_unreviewed,
        )

    grouped_dataframe, group_summary = create_groups(
        dataframe=dataframe,
        auto_pairs=auto_pairs,
        manual_merge_edges=manual_edges,
        config=similarity_config,
    )

    save_review_outputs(
        audit_dir=audit_dir,
        dataframe=grouped_dataframe,
        auto_pairs=auto_pairs,
        review_pairs=review_pairs,
        group_summary=group_summary,
        similarity_config=similarity_config,
    )

    print("\n" + "=" * 78)
    print(" REGROUPEMENT DOCUMENTAIRE DEREEV")
    print("=" * 78)
    print(f"Notes analysées             : {len(grouped_dataframe)}")
    print(f"Paires fusionnées auto      : {len(auto_pairs)}")
    print(f"Paires à examiner           : {len(review_pairs)}")
    print(f"Groupes totaux              : {grouped_dataframe[GROUP_COL].nunique()}")
    print(f"Groupes de plusieurs notes  : {(group_summary['group_size'] > 1).sum()}")
    print(f"Plus grand groupe           : {group_summary['group_size'].max()}")
    print(f"Rapport de revue            : {audit_dir / 'pairs_to_review.csv'}")

    if args.stage == "review":
        print("\nÉtape suivante :")
        print("1. Ouvrir pairs_to_review.csv.")
        print("2. Renseigner decision avec 'merge' ou 'separate'.")
        print("3. Relancer avec --stage finalize --review-file ... --overwrite.")
        return

    train_path = args.output_dir / "train_cv.csv"
    test_path = args.output_dir / "test.csv"
    cv_split_dir = args.output_dir / "cv_splits"
    target_paths = [train_path, test_path, cv_split_dir]

    if any(path.exists() for path in target_paths) and not args.overwrite:
        raise FileExistsError(
            "Des sorties officielles existent déjà. Utilisez --overwrite "
            "pour les sauvegarder puis les remplacer."
        )

    backup = make_backup(
        paths=target_paths,
        backup_root=args.output_dir / "backups",
    )
    if backup is not None:
        LOGGER.info("Anciennes sorties sauvegardées dans : %s", backup)

    # Les anciens manifests ne sont plus valides après un nouveau train/test.
    # Ils seront recréés par cv_protocol.py au premier entraînement.
    if cv_split_dir.exists():
        shutil.rmtree(cv_split_dir)

    train_indices, test_indices, holdout_details = select_group_holdout(
        dataframe=grouped_dataframe,
        label_column="reev_classe",
        group_column=GROUP_COL,
        source_column="src",
        config=split_config,
    )

    raw_train = grouped_dataframe.iloc[train_indices]
    raw_test = grouped_dataframe.iloc[test_indices]
    validate_no_group_overlap(raw_train, raw_test, GROUP_COL)

    residual_pairs = audit_residual_cross_split_pairs(
        grouped_dataframe=grouped_dataframe,
        char_matrix=char_matrix,
        left_indices=np.asarray(test_indices, dtype=int),
        right_indices=np.asarray(train_indices, dtype=int),
        split_name="test_vs_train_cv",
        threshold=similarity_config.residual_audit_similarity,
    )
    if residual_pairs.empty:
        residual_pairs = pd.DataFrame(
            columns=[
                "boundary",
                "similarity",
                "left_document_id",
                "right_document_id",
                "left_group",
                "right_group",
                "left_label",
                "right_label",
                "left_text",
                "right_text",
            ]
        )

    outputs = save_final_outputs(
        grouped_dataframe=grouped_dataframe,
        train_indices=np.asarray(train_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        holdout_details=holdout_details,
        output_dir=args.output_dir,
        audit_dir=audit_dir,
        text_column="cn_notes",
        label_column="reev_classe",
        group_column=GROUP_COL,
        split_config=split_config,
        residual_pairs=residual_pairs,
    )

    print("\n" + "=" * 78)
    print(" SPLIT TRAIN/TEST GROUP-AWARE CRÉÉ")
    print("=" * 78)
    print(f"Train_CV                   : {outputs['train_csv']}")
    print(f"Test                       : {outputs['test_csv']}")
    print(f"Résumé complet             : {outputs['split_summary']}")
    print(f"Paires résiduelles à revoir: {len(residual_pairs)}")
    print("\nLes folds ne sont pas créés par prepare_data.py.")
    print("Ils seront créés par cv_protocol.py au premier entraînement.")
    print("Tous les modèles doivent être réentraînés sur les nouveaux splits.")


if __name__ == "__main__":
    main(build_parser().parse_args())
