import os

# Dossier contenant le code spécialisé BioMistral : projet-dereev/biomistral
BIOMISTRAL_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Racine générale du projet : projet-dereev
BASE_DIR = os.path.dirname(BIOMISTRAL_DIR)
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESSOURCES_DIR = os.path.join(BASE_DIR, "ressources")
CV_SPLITS_DIR = os.path.join(DATA_PROCESSED_DIR, "cv_splits")
BENCHMARK_MODELS_DIR = os.path.join(MODELS_DIR, "benchmark")

RAW_XLSX = os.path.join(DATA_RAW_DIR, "dereev_data_4_deduplicated.xlsx")
TRAIN_CV_CSV = os.path.join(DATA_PROCESSED_DIR, "train_cv.csv")
TEST_CSV = os.path.join(DATA_PROCESSED_DIR, "test.csv")
DICT_ABREV = os.path.join(RESSOURCES_DIR, "abreviations", "dico_global.json")

TEXT_COL = "cn_notes"
LABEL_COL = "reev_classe"
GROUP_COL = "leakage_group_id"
TEST_SIZE = 0.15
RANDOM_SEED = 42
K_FOLDS = 5
CV_SPLIT_SEED = 42
TRAINING_SEEDS = [42, 123, 456, 789, 2026]
# Par sécurité, le pilote BioMistral ne lance qu'une seed par défaut.
BIOMISTRAL_PILOT_SEEDS = [42]
EXPERIMENT_SEEDS = TRAINING_SEEDS

MODEL_DISPLAY_NAME = "BioMistral-7B-LoRA"
MODEL_NAME = "BioMistral/BioMistral-7B"
MODEL_OUTPUT_NAME = "biomistral_7b_lora"
MAX_LENGTH = 512

BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 12
# LoRA : valeur volontairement plus élevée qu'un fine-tuning complet.
LEARNING_RATE = 2e-4
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001
USE_BF16 = True
GRADIENT_CHECKPOINTING = False
ATTN_IMPLEMENTATION = "sdpa"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_MODULES_TO_SAVE = ["score"]

# AutoModelForSequenceClassification de Mistral utilise le dernier token non
# paddé. Le tokenizer est donc explicitement configuré avec padding à droite.
POOLING_STRATEGY = "last_non_padding_token"
PADDING_SIDE = "right"
TRUNCATION_SIDE = "right"

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
DEFAULT_DEPLOYMENT_STRATEGY = "fold_ensemble"