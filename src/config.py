import os

# ---------------------------------------------------------------------
# DOSSIERS
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESSOURCES_DIR = os.path.join(BASE_DIR, "ressources")

# Un seul manifeste de folds est partagé par toutes les seeds d'entraînement
# et par tous les modèles du benchmark.
CV_SPLITS_DIR = os.path.join(DATA_PROCESSED_DIR, "cv_splits")
BENCHMARK_MODELS_DIR = os.path.join(MODELS_DIR, "benchmark")

# ---------------------------------------------------------------------
# FICHIERS
# ---------------------------------------------------------------------
RAW_XLSX = os.path.join(DATA_RAW_DIR, "dereev_data_4_deduplicated.xlsx")
TRAIN_CV_CSV = os.path.join(DATA_PROCESSED_DIR, "train_cv.csv")
TEST_CSV = os.path.join(DATA_PROCESSED_DIR, "test.csv")
DICT_ABREV = os.path.join(RESSOURCES_DIR, "abreviations", "dico_global.json")

# ---------------------------------------------------------------------
# PARAMÈTRES DATA
# ---------------------------------------------------------------------
TEXT_COL = "cn_notes"
LABEL_COL = "reev_classe"
GROUP_COL = "leakage_group_id"
TEST_SIZE = 0.15

# Seed utilisée uniquement pour créer le split externe train_cv / test.
RANDOM_SEED = 42

K_FOLDS = 5

# PROTOCOLE A STRICT
# ------------------
# Cette seed sert UNE SEULE FOIS à construire la répartition fixe des K folds.
# Elle ne change jamais entre les répétitions ni entre les modèles.
CV_SPLIT_SEED = 42

# Ces cinq seeds ne contrôlent que l'aléatoire de l'entraînement :
# initialisation de la tête, ordre des batchs, dropout, PyTorch/Transformers, etc.
TRAINING_SEEDS = [42, 123, 456, 789, 2026]

# Alias temporaire pour d'éventuels anciens scripts non encore migrés.
EXPERIMENT_SEEDS = TRAINING_SEEDS

# ---------------------------------------------------------------------
# PARAMÈTRES DU MODÈLE TRANSFORMER
# ---------------------------------------------------------------------
MODEL_DISPLAY_NAME = "DoctoBERT-fr-base"
MODEL_NAME = "doctolib-lab/doctobert-fr-base"
MODEL_OUTPUT_NAME = "doctobert_fr_base"

MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 2e-5

EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001

# ---------------------------------------------------------------------
# BASELINE TF-IDF + RÉGRESSION LOGISTIQUE
# ---------------------------------------------------------------------
TFIDF_DISPLAY_NAME = "TF-IDF + régression logistique"
TFIDF_OUTPUT_NAME = "tfidf_lr"
# ---------------------------------------------------------------------
# INCERTITUDE ET REPORTING
# ---------------------------------------------------------------------
# Les écarts-types du benchmark restent des écarts-types inter-seed à split fixe.
# Le bootstrap ci-dessous estime séparément l'incertitude d'échantillonnage du test.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
DEFAULT_DEPLOYMENT_STRATEGY = "fold_ensemble"
