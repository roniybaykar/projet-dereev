from transformers import AutoModelForSequenceClassification
from src.config import MODEL_NAME
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def get_model():
    
    # avec la couche de classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )
    
    return model