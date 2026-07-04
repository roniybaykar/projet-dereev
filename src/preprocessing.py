import re 
import json
import os 
import pandas as pd
from src.config import DICT_ABREV

def load_abreviations():
    if os.path.exists(DICT_ABREV):
        with open(DICT_ABREV, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def replace_abreviations(text, abrev_dict):
    if not abrev_dict:
        return text
    for abbrev, full_form in sorted(abrev_dict.items(), key=lambda x: len(x[0]), reverse=True):

        pattern = r'\b' + re.escape(abbrev) + r'\b' # on remplace pas l'intérieur d'un autre mot
        
        if len(abbrev) <= 2:
            text = re.sub(pattern, full_form, text)
        else:
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)

    return text

def clean_text(text: str, abbrev_dict=None) -> str:
    if pd.isna(text):
        return ""
    
    # nettoyage
    text = str(text)
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)      # espaces multiples
    text = re.sub(r"\n{3,}", "\n\n", text)   # trop de sauts de ligne
    
    # remplace abréviations
    if abbrev_dict:
        text = replace_abreviations(text, abbrev_dict)
    
    return text.strip()