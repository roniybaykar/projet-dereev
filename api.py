from fastapi import FastAPI
from pydantic import BaseModel, Field
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MODELS_DIR, MAX_LENGTH, MODEL_NAME
from src.preprocessing import clean_text, load_abreviations

app = FastAPI(
    title="API DEREEV - Hôpital Foch")

class NoteMedicale(BaseModel):
    texte: str = Field(..., description="note")
    seuil: float = Field(default=0.5, ge=0.0, le=1.0, description="seuil")

abrev_dict = load_abreviations()

try:
    # on charge le modèle entraîné
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR)
    print("Modèle chargé")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    tokenizer = None
    model = None

@app.post("/predict")
def predire_reevaluation(note: NoteMedicale):
    
    if model is None or tokenizer is None:
        return {"erreur": "Modèle introuvable dans le dossier models/"}
        
    # nettoyage de la note
    texte_propre = clean_text(note.texte, abrev_dict)
    
    # tokenization
    inputs = tokenizer(texte_propre, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    
    # travail du modèle
    with torch.no_grad():
        outputs = model(**inputs)
        probabilites = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    proba_reval = probabilites[0][1].item()
    #proba_reval = random.uniform(0.0, 1.0)

    # seuil
    prediction_finale = 1 if proba_reval >= note.seuil else 0
    
    # réponse
    return {
        "note" : texte_propre,
        "prediction": prediction_finale,
        "probabilite_calculee": round(proba_reval, 4),
        "seuil_utilise": note.seuil,
        "alerte": "Réévaluation détectée" if prediction_finale == 1 else "Pas de réévaluation"
    }