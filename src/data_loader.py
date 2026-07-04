import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH, TEXT_COL, LABEL_COL

def get_tokenized_datasets(train_cv_path, test_path):
    # 1) Lire les CSV
    train_cv_df = pd.read_csv(train_cv_path)
    test_df = pd.read_csv(test_path)

    print("Train_CV shape:", train_cv_df.shape)
    print("Test shape :", test_df.shape)

    # 2) Convertir en Dataset
    train_cv_ds = Dataset.from_pandas(train_cv_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    # 3) Charger tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4) Fonction de tokenization
    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    # 5) Appliquer tokenization
    train_cv_tok = train_cv_ds.map(tokenize_batch, batched=True)
    test_tok = test_ds.map(tokenize_batch, batched=True)

    # 6) Mettre le label au bon format pour Transformers ("labels")
    train_cv_tok = train_cv_tok.rename_column(LABEL_COL, "labels")
    test_tok = test_tok.rename_column(LABEL_COL, "labels")

    # 7) Garder uniquement ce dont torch a besoin
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_cv_tok = train_cv_tok.remove_columns([c for c in train_cv_tok.column_names if c not in cols_to_keep])
    test_tok = test_tok.remove_columns([c for c in test_tok.column_names if c not in cols_to_keep])

    train_cv_tok.set_format("torch")
    test_tok.set_format("torch")

    # 8) Vérifs rapides
    sample = train_cv_tok[0]
    print("\nSample tokenized")
    print("input_ids shape:", sample["input_ids"].shape)
    print("attention_mask shape:", sample["attention_mask"].shape)
    print("label:", sample["labels"].item())

    print("\n Tokenization OK. Prêt pour l'entraînement.")

    return train_cv_tok, test_tok