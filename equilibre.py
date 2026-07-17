from pathlib import Path
from src.config import RAW_XLSX
import pandas as pd


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Dossier racine du projet : dossier contenant equilibre.py
DOSSIER_PROJET = Path(__file__).resolve().parent

# Fichier Excel d'entrée
FICHIER_ENTREE = (
    DOSSIER_PROJET
    / "data"
    / "raw"
    / RAW_XLSX
)

# Fichier Excel de sortie, créé dans data/raw
FICHIER_SORTIE = (
    DOSSIER_PROJET
    / "data"
    / "raw"
    / "dereev_data_3_deduplicated_45.xlsx"
)

COLONNE_NOTE = "cn_notes"
COLONNE_LABEL = "reev_classe"

# Répartition souhaitée :
# 25 % de labels 1
# 75 % de labels 0
PROPORTION_LABEL_1 = 0.45
PROPORTION_LABEL_0 = 0.55

# Permet d'obtenir le même tirage aléatoire à chaque exécution
RANDOM_STATE = 42


# ------------------------------------------------------------------
# Vérification du fichier d'entrée
# ------------------------------------------------------------------

if not FICHIER_ENTREE.exists():
    raise FileNotFoundError(
        f"Le fichier d'entrée est introuvable :\n{FICHIER_ENTREE}"
    )


# ------------------------------------------------------------------
# Lecture du fichier Excel
# ------------------------------------------------------------------

# keep_default_na=False permet de conserver le texte "NA"
# comme une chaîne de caractères afin de le supprimer explicitement.
df = pd.read_excel(
    FICHIER_ENTREE,
    keep_default_na=False
)

print(f"Fichier lu : {FICHIER_ENTREE}")
print(f"Nombre initial de lignes : {len(df)}")


# ------------------------------------------------------------------
# Vérification des colonnes
# ------------------------------------------------------------------

colonnes_attendues = [
    COLONNE_NOTE,
    COLONNE_LABEL
]

colonnes_manquantes = [
    colonne
    for colonne in colonnes_attendues
    if colonne not in df.columns
]

if colonnes_manquantes:
    raise ValueError(
        "Colonnes manquantes dans le fichier Excel : "
        + ", ".join(colonnes_manquantes)
    )


# ------------------------------------------------------------------
# Nettoyage des notes
# ------------------------------------------------------------------

# Supprime les espaces au début et à la fin des notes
notes_nettoyees = (
    df[COLONNE_NOTE]
    .astype(str)
    .str.strip()
)

# Conservation uniquement des notes :
# - non vides
# - différentes de "NA", sans tenir compte des majuscules/minuscules
masque_notes_valides = (
    notes_nettoyees.ne("")
    & notes_nettoyees.str.upper().ne("NA")
)

nombre_notes_supprimees = int(
    (~masque_notes_valides).sum()
)

df = df[masque_notes_valides].copy()

# Enregistre la version nettoyée de la note
df[COLONNE_NOTE] = (
    df[COLONNE_NOTE]
    .astype(str)
    .str.strip()
)

print(
    f"Notes vides ou égales à NA supprimées : "
    f"{nombre_notes_supprimees}"
)
print(
    f"Nombre de lignes après nettoyage des notes : "
    f"{len(df)}"
)


# ------------------------------------------------------------------
# Nettoyage des labels
# ------------------------------------------------------------------

# Conversion des labels en nombres
# Les valeurs invalides deviennent NaN
df[COLONNE_LABEL] = pd.to_numeric(
    df[COLONNE_LABEL],
    errors="coerce"
)

nombre_avant_nettoyage_labels = len(df)

# Suppression des lignes dont le label est vide ou invalide
df = df.dropna(
    subset=[COLONNE_LABEL]
).copy()

# Conservation uniquement des labels 0 et 1
df = df[
    df[COLONNE_LABEL].isin([0, 1])
].copy()

# Conversion en nombres entiers
df[COLONNE_LABEL] = (
    df[COLONNE_LABEL]
    .astype(int)
)

nombre_labels_invalides = (
    nombre_avant_nettoyage_labels - len(df)
)

print(
    f"Lignes avec un label invalide supprimées : "
    f"{nombre_labels_invalides}"
)


# ------------------------------------------------------------------
# Séparation des classes
# ------------------------------------------------------------------

df_0 = df[
    df[COLONNE_LABEL] == 0
].copy()

df_1 = df[
    df[COLONNE_LABEL] == 1
].copy()

nombre_0_initial = len(df_0)
nombre_1_initial = len(df_1)

print("\nRépartition avant équilibrage :")
print(f"Labels 0 : {nombre_0_initial}")
print(f"Labels 1 : {nombre_1_initial}")

if nombre_1_initial == 0:
    raise ValueError(
        "Le fichier ne contient aucune note labellisée 1."
    )

if nombre_0_initial == 0:
    raise ValueError(
        "Le fichier ne contient aucune note labellisée 0."
    )


# ------------------------------------------------------------------
# Calcul du nombre de labels 0 à conserver
# ------------------------------------------------------------------

# On conserve toutes les notes labellisées 1.
#
# On souhaite :
#
# nombre_1 / (nombre_0 + nombre_1) = 0.25
#
# Donc :
#
# nombre_0 = nombre_1 × 0.75 / 0.25

nombre_0_a_conserver = round(
    nombre_1_initial
    * PROPORTION_LABEL_0
    / PROPORTION_LABEL_1
)

if nombre_0_a_conserver > nombre_0_initial:
    raise ValueError(
        "\nImpossible d'obtenir 25 % de labels 1 et "
        "75 % de labels 0 uniquement en supprimant "
        "des labels 0.\n"
        f"Il faudrait conserver {nombre_0_a_conserver} "
        f"labels 0, mais le fichier n'en contient que "
        f"{nombre_0_initial}."
    )

nombre_0_supprimes = (
    nombre_0_initial - nombre_0_a_conserver
)

print(
    f"\nNombre de labels 0 à conserver : "
    f"{nombre_0_a_conserver}"
)
print(
    f"Nombre de labels 0 supprimés : "
    f"{nombre_0_supprimes}"
)


# ------------------------------------------------------------------
# Sélection aléatoire des labels 0
# ------------------------------------------------------------------

df_0_reduit = df_0.sample(
    n=nombre_0_a_conserver,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------------
# Création du jeu de données final
# ------------------------------------------------------------------

# On conserve :
# - tous les labels 1
# - seulement une partie aléatoire des labels 0
df_final = pd.concat(
    [
        df_0_reduit,
        df_1
    ],
    ignore_index=True
)

# Mélange aléatoire des lignes
# Chaque note reste associée à son label.
df_final = df_final.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ------------------------------------------------------------------
# Création du fichier Excel
# ------------------------------------------------------------------

df_final.to_excel(
    FICHIER_SORTIE,
    index=False
)


# ------------------------------------------------------------------
# Vérification de la répartition finale
# ------------------------------------------------------------------

repartition = (
    df_final[COLONNE_LABEL]
    .value_counts()
)

pourcentages = (
    df_final[COLONNE_LABEL]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nRépartition finale :")
print(
    f"Labels 0 : {repartition.get(0, 0)} "
    f"({pourcentages.get(0, 0):.2f} %)"
)
print(
    f"Labels 1 : {repartition.get(1, 0)} "
    f"({pourcentages.get(1, 0):.2f} %)"
)

print(
    f"\nFichier créé avec succès :\n"
    f"{FICHIER_SORTIE}"
)