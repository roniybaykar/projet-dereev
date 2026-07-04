import os

### DOSSIERS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESSOURCES_DIR = os.path.join(BASE_DIR, "ressources")

### FICHIERS
RAW_XLSX = os.path.join(DATA_RAW_DIR, "dereev_synthetic_data_augmented.xlsx")
TRAIN_CV_CSV = os.path.join(DATA_PROCESSED_DIR, "train_cv.csv")
TEST_CSV = os.path.join(DATA_PROCESSED_DIR, "test.csv")
DICT_ABREV = os.path.join(RESSOURCES_DIR, "abreviations", "dico_global.json")

### PARAMÈTRES DATA
TEXT_COL = "cn_notes"
LABEL_COL = "reev_expert"
TEST_SIZE = 0.15
RANDOM_SEED = 42
K_FOLDS = 5

### PARAMÈTRES DU MODÈLE
MODEL_NAME = "almanach/camembert-bio-base"
MAX_LENGTH = 256 # estimation
BATCH_SIZE = 16 # standard pour camembert
EPOCHS = 3
LEARNING_RATE = 2e-5