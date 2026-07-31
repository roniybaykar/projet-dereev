"""Chargement et tokenisation group-aware pour BioMistral-LoRA."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from biomistral.src.config import (
    GROUP_COL,
    LABEL_COL,
    MAX_LENGTH,
    MODEL_NAME,
    PADDING_SIDE,
    TEXT_COL,
    TRUNCATION_SIDE,
)


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("BioMistral ne fournit ni pad_token ni eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = PADDING_SIDE
    tokenizer.truncation_side = TRUNCATION_SIDE
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_limit, int)
        and tokenizer_limit < 10**9
        and MAX_LENGTH > tokenizer_limit
    ):
        raise ValueError(
            f"MAX_LENGTH={MAX_LENGTH} dépasse la limite tokenizer={tokenizer_limit}."
        )
    return tokenizer


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    dataframe = pd.read_csv(path)
    required = [TEXT_COL, LABEL_COL, GROUP_COL]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")
    # GROUP_COL reste dans le CSV pour les splits mais n'est jamais fourni au modèle.
    model_dataframe = dataframe[[TEXT_COL, LABEL_COL]].copy()
    model_dataframe[TEXT_COL] = model_dataframe[TEXT_COL].fillna("").astype(str)
    model_dataframe[LABEL_COL] = model_dataframe[LABEL_COL].astype(int)
    if model_dataframe[TEXT_COL].str.strip().eq("").any():
        raise ValueError(f"Note vide dans {path}.")
    if not model_dataframe[LABEL_COL].isin([0, 1]).all():
        raise ValueError(f"Labels invalides dans {path}.")
    return model_dataframe.reset_index(drop=True)


def _tokenize_dataframe(dataframe: pd.DataFrame) -> Dataset:
    tokenizer = get_tokenizer()
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)
    dataset = dataset.rename_column(LABEL_COL, "labels")

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, desc="Tokenisation BioMistral")
    tokenized = tokenized.remove_columns([TEXT_COL])
    columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized.column_names:
        columns.append("token_type_ids")
    tokenized.set_format(type="torch", columns=columns)
    return tokenized


def get_tokenized_datasets(train_cv_path: str, test_path: str) -> Tuple[Dataset, Dataset]:
    train = _read_csv(train_cv_path)
    test = _read_csv(test_path)
    tokenizer = get_tokenizer()
    print(f"Train_CV shape (colonnes modèle): {train.shape}")
    print(f"Test shape (colonnes modèle)    : {test.shape}")
    print(
        f"Tokenizer: padding_side={tokenizer.padding_side}, "
        f"truncation_side={tokenizer.truncation_side}, max_length={MAX_LENGTH}"
    )
    return _tokenize_dataframe(train), _tokenize_dataframe(test)
