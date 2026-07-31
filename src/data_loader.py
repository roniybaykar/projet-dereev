import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.config import LABEL_COL, MAX_LENGTH, MODEL_NAME, TEXT_COL


def _read_model_columns(path: str) -> pd.DataFrame:
    """Charge uniquement les colonnes réellement consommées par le modèle."""
    dataframe = pd.read_csv(path)
    missing = [
        column
        for column in (TEXT_COL, LABEL_COL)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path} : " + ", ".join(missing)
        )

    # leakage_group_id reste dans le CSV pour les splits, mais n'entre jamais
    # dans le Dataset Hugging Face ni dans le modèle.
    dataframe = dataframe[[TEXT_COL, LABEL_COL]].copy()
    dataframe[TEXT_COL] = dataframe[TEXT_COL].fillna("").astype(str)
    dataframe[LABEL_COL] = dataframe[LABEL_COL].astype(int)

    if dataframe[TEXT_COL].str.strip().eq("").any():
        raise ValueError(f"Au moins une note vide est présente dans {path}.")
    if not dataframe[LABEL_COL].isin([0, 1]).all():
        raise ValueError(f"La colonne {LABEL_COL} doit contenir uniquement 0/1.")
    return dataframe.reset_index(drop=True)


def get_tokenized_datasets(train_cv_path, test_path):
    train_cv_df = _read_model_columns(train_cv_path)
    test_df = _read_model_columns(test_path)

    print("Train_CV shape (colonnes modèle):", train_cv_df.shape)
    print("Test shape (colonnes modèle)    :", test_df.shape)

    train_cv_ds = Dataset.from_pandas(train_cv_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_limit, int)
        and tokenizer_limit < 10**9
        and MAX_LENGTH > tokenizer_limit
    ):
        raise ValueError(
            f"MAX_LENGTH={MAX_LENGTH} dépasse la limite du tokenizer "
            f"({tokenizer_limit}) pour {MODEL_NAME}."
        )
    print(
        "Configuration tokenizer : "
        f"max_length={MAX_LENGTH}, limite déclarée={tokenizer_limit}"
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_cv_tok = train_cv_ds.map(tokenize_batch, batched=True)
    test_tok = test_ds.map(tokenize_batch, batched=True)
    train_cv_tok = train_cv_tok.rename_column(LABEL_COL, "labels")
    test_tok = test_tok.rename_column(LABEL_COL, "labels")

    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_cv_tok = train_cv_tok.remove_columns(
        [c for c in train_cv_tok.column_names if c not in cols_to_keep]
    )
    test_tok = test_tok.remove_columns(
        [c for c in test_tok.column_names if c not in cols_to_keep]
    )

    train_cv_tok.set_format("torch")
    test_tok.set_format("torch")

    sample = train_cv_tok[0]
    print("\nSample tokenized")
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("label:", sample["labels"].item())
    print("\nTokenization OK. leakage_group_id n'est pas transmis au modèle.")

    return train_cv_tok, test_tok
