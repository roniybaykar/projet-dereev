import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_XLSX,TRAIN_CV_CSV, TEST_CSV, TEXT_COL, LABEL_COL, TEST_SIZE, RANDOM_SEED
from src.preprocessing import clean_text, load_abreviations

def main():
    # 1) Charger
    if not os.path.exists(RAW_XLSX):
        raise FileNotFoundError(
            f"Fichier introuvable: {RAW_XLSX}\n"
            )

    df = pd.read_excel(RAW_XLSX)
    abrev_dict = load_abreviations()

    # 2) Vérifier colonnes
    missing = [c for c in [TEXT_COL, LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}\nColonnes trouvées: {list(df.columns)}")

    # 3) Garder uniquement utile
    df = df[[TEXT_COL, LABEL_COL]].copy()

    # 4) Retirer lignes vides
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

    # 5) Nettoyage texte
    df[TEXT_COL] = df[TEXT_COL].apply(lambda x: clean_text(x, abrev_dict))

    # 6) Label en int + vérif 0/1
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    if not df[LABEL_COL].isin([0, 1]).all():
        bad = df[~df[LABEL_COL].isin([0, 1])].head(10)
        raise ValueError(f"Labels invalides (doivent être 0/1). Exemples:\n{bad}")
    
    # 7) Split stratifié unique (Train_CV vs Test)
    train_cv_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[LABEL_COL],
    )

    # 8) Sauvegarde
    train_cv_df.to_csv(TRAIN_CV_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("\nSauvegarde !")
    print(f"Train_CV: {TRAIN_CV_CSV} ({len(train_cv_df)} lignes)")
    print(f"Test : {TEST_CSV} ({len(test_df)} lignes)")
    print("\n Prétraitement terminé.")

if __name__ == "__main__":
    main()