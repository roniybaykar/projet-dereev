from transformers import AutoModelForSequenceClassification
from src.config import MODEL_NAME

def get_model():
    
    # avec la couche de classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )
    
    return model