import os

### DOSSIERS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESSOURCES_DIR = os.path.join(BASE_DIR, "ressources")

### FICHIERS
RAW_XLSX = os.path.join(DATA_RAW_DIR, "dereev_data_4_deduplicated.xlsx")
TRAIN_CV_CSV = os.path.join(DATA_PROCESSED_DIR, "train_cv.csv")
TEST_CSV = os.path.join(DATA_PROCESSED_DIR, "test.csv")
DICT_ABREV = os.path.join(RESSOURCES_DIR, "abreviations", "dico_global.json")

### PARAMÈTRES DATA
TEXT_COL = "cn_notes"
LABEL_COL = "reev_classe"
TEST_SIZE = 0.15
RANDOM_SEED = 42
K_FOLDS = 5

### PARAMÈTRES DU MODÈLE
MODEL_NAME = "almanach/camembert-bio-base"
# On garde MAX_LENGTH à 512 pour ne pas perdre d'information textuelle.
MAX_LENGTH = 512
# Taille de lot plus faible pour limiter la RAM pendant l'entraînement.
BATCH_SIZE = 16
# Limite maximale d'époques ; l'arrêt automatique pourra se faire plus tôt.
EPOCHS = 12
LEARNING_RATE = 2e-5
# Arrêt automatique si la métrique de validation ne s'améliore plus.
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001
