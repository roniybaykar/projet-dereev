import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.config import TEST_CSV, MODELS_DIR, TRAIN_CV_CSV
from src.data_loader import get_tokenized_datasets

### Calculer les scores

def compute_metrics(eval_pred):
    proba, labels = eval_pred
    predictions = np.argmax(proba, axis = 1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        'accuracy' : acc,
        'f1' : f1,
        'precision' : precision,
        'recall' : recall,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'true_positive_count': int(np.sum(labels == 1)),
        'predicted_positive_count': int(np.sum(predictions == 1)),
    }

def main():
    BEST_FOLD = 5 
    best_model_path = os.path.join(MODELS_DIR, f"fold_{BEST_FOLD}")

    if not os.path.exists(best_model_path):
        print(f"Le modèle {best_model_path} est introuvable. Lance main.py d'abord.")
        return
    
    print(f"Évaluation finale avec le modèle du Fold {BEST_FOLD}...")

    # 1. On va chercher le dataset test
    _, test_ds = get_tokenized_datasets(TRAIN_CV_CSV, TEST_CSV)

    # 2. On charge notre modèle 
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

    # 3. Trainer uniquement pour l'évaluation
    trainer = Trainer(model=model, compute_metrics=compute_metrics)

    # 4. Prédictions sur le jeu de test
    results = trainer.evaluate(eval_dataset=test_ds)

    print("\n" + "="*40)
    print(" RÉSULTATS SUR TEST.CSV")
    print("="*40)
    print(f"Accuracy  : {results['eval_accuracy']:.4f}") 
    print(f"F1 Score  : {results['eval_f1']:.4f}")
    print(f"Recall    : {results['eval_recall']:.4f}")
    print(f"Precision : {results['eval_precision']:.4f}")
    print("\nMatrice de confusion:")
    print("                 Predit 0    Predit 1")
    print(f"Vrai 0           {int(results['eval_true_negative']):>8}    {int(results['eval_false_positive']):>8}")
    print(f"Vrai 1           {int(results['eval_false_negative']):>8}    {int(results['eval_true_positive']):>8}")
    print(
        f"\nVrais positifs test: {int(results['eval_true_positive_count'])} | "
        f"Predictions positives: {int(results['eval_predicted_positive_count'])}"
    )
    print(f"Final Loss: {results['eval_loss']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
