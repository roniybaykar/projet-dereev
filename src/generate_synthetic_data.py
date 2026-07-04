import os
import pandas as pd
import random
import time
import google.generativeai as genai
from dotenv import load_dotenv
import config 

# Charge la clé cachée dans le fichier .env
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

TARGET_TOTAL_DATA = 750
BATCH_SIZE = 10 # Nombre de notes générées par requête (pour éviter les hallucinations)

# Nom du NOUVEAU fichier pour ne pas écraser l'ancien
NEW_XLSX_PATH = os.path.join(config.DATA_RAW_DIR, "dereev_synthetic_data_augmented.xlsx")

def generate_notes_batch(examples, label, num_to_generate):
    """
    Génère un lot de nouvelles notes médicales en se basant sur des exemples.
    """
    # Préparation des exemples pour le prompt
    examples_text = "\n\n".join([f"Exemple {i+1} :\n{text}" for i, text in enumerate(examples)])
    
    # Explication du label pour guider le modèle
    if label == 1:
        contexte = "Ces notes impliquent une réévaluation, un changement, ou un arrêt d'antibiotiques."
    else:
        contexte = "Ces notes décrivent un état clinique standard ou une simple surveillance sans changement majeur d'antibiothérapie."

    prompt = f"""Tu es un médecin francophone expert en réanimation et maladies infectieuses.
Ta tâche est de générer {num_to_generate} NOUVELLES notes médicales synthétiques (en français).
{contexte}

Voici quelques exemples de notes réelles de notre hôpital avec ce label :
{examples_text}

Règles strictes :
1. Génère {num_to_generate} nouvelles notes qui respectent le même style médical (abréviations, jargon, constantes vitales).
2. NE COPIE PAS les exemples. Modifie les antibiotiques, les valeurs des constantes, les âges, les pathologies sous-jacentes.
3. Alterne entre des styles narratifs et des styles sous forme de tirets.
4. Sépare chaque note générée par le délimiteur "###". Ne mets aucun autre texte avant ou après.
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.8) # 0.8 pour la créativité et éviter l'overfitting
    )
    
    generated_text = response.text
    # Découpage des notes via le délimiteur
    new_notes = [note.strip() for note in generated_text.split("###") if len(note.strip()) > 10]
    return new_notes[:num_to_generate]

def main():
    print(f"Chargement des données depuis {config.RAW_XLSX}...")
    
    # Adaptation selon que votre fichier brut soit un .csv ou un .xlsx
    if config.RAW_XLSX.endswith('.csv'):
        df_original = pd.read_csv(config.RAW_XLSX)
    else:
        df_original = pd.read_excel(config.RAW_XLSX)
    
    current_count = len(df_original)
    print(f"Nombre de données actuelles : {current_count}")
    
    if current_count >= TARGET_TOTAL_DATA:
        print("Vous avez déjà atteint ou dépassé les 750 données !")
        return

    to_generate = TARGET_TOTAL_DATA - current_count
    
    # Calcul des proportions pour garder la même distribution
    label_counts = df_original[config.LABEL_COL].value_counts(normalize=True)
    
    new_data = []
    
    for label, proportion in label_counts.items():
        # Combien de nouvelles notes pour ce label spécifique ?
        count_for_label = int(round(to_generate * proportion))
        print(f"\nGénération de {count_for_label} notes pour le label {label}...")
        
        # Isoler les notes existantes pour ce label
        existing_notes_for_label = df_original[df_original[config.LABEL_COL] == label][config.TEXT_COL].tolist()
        
        generated_count = 0
        while generated_count < count_for_label:
            batch_size = min(BATCH_SIZE, count_for_label - generated_count)
            
            # Tirer 5 exemples au hasard de notre base pour guider le modèle
            few_shot_examples = random.sample(existing_notes_for_label, min(5, len(existing_notes_for_label)))
            
            try:
                new_notes = generate_notes_batch(few_shot_examples, label, batch_size)
                
                for note in new_notes:
                    new_data.append({
                        config.TEXT_COL: note,
                        config.LABEL_COL: label
                    })
                
                generated_count += len(new_notes)
                print(f"Progression : {generated_count}/{count_for_label} pour le label {label}")
                time.sleep(5) #Car sinon quota/minutes atteint 
                
            except Exception as e:
                print(f"Erreur lors de l'appel API : {e}")
                break

    # Création du DataFrame final
    df_new = pd.DataFrame(new_data)
    df_final = pd.concat([df_original, df_new], ignore_index=True)
    
    # Sauvegarde dans le NOUVEAU fichier
    print(f"\nSauvegarde du nouveau dataset ({len(df_final)} lignes) dans {NEW_XLSX_PATH}...")
    df_final.to_excel(NEW_XLSX_PATH, index=False)
    print("Terminé avec succès ")

if __name__ == "__main__":
    main()