import os
import numpy as np
import torch
from torch import nn
from transformers import TrainingArguments, Trainer
from src.config import TRAIN_CV_CSV, TEST_CSV, MODELS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, K_FOLDS, RANDOM_SEED
from src.data_loader import get_tokenized_datasets
from src.model import get_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

def compute_metrics(eval_pred):
    proba, labels = eval_pred
    predictions = np.argmax(proba, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'true_positive_count': int(np.sum(labels == 1)),
        'predicted_positive_count': int(np.sum(predictions == 1)),
    }


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    
    # 1. vérification des données
    if not os.path.exists(TRAIN_CV_CSV) or not os.path.exists(TEST_CSV):
        print("fichiers introuvables dans data/processed, il faut d'abord preprocess le dataset")
        return

    # 2. tokenization
    train_cv_ds, test_ds = get_tokenized_datasets(TRAIN_CV_CSV, TEST_CSV)

    # 3. Préparation du K-Fold
    labels = np.array(train_cv_ds["labels"]) 
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []

    # 4. Boucle d'entraînement croisé
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*40}")
        print(f" DÉMARRAGE DU FOLD {fold + 1}/{K_FOLDS}")
        print(f"{'='*40}")

        train_fold_ds = train_cv_ds.select(train_idx)
        val_fold_ds = train_cv_ds.select(val_idx)

        train_labels = np.array(train_fold_ds["labels"])
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        print(f"Repartition train fold: classe 0 = {class_counts[0]}, classe 1 = {class_counts[1]}")
        print(f"Poids de loss: classe 0 = {class_weights[0]:.4f}, classe 1 = {class_weights[1]:.4f}")

        # Modèle vierge pour chaque itération
        model = get_model()
        fold_model_dir = os.path.join(MODELS_DIR, f"fold_{fold + 1}")

        training_args = TrainingArguments(
            output_dir=fold_model_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            eval_strategy="epoch",  
            save_strategy="epoch",  
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f'./logs/fold_{fold + 1}',
            logging_steps=10,
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_fold_ds,
            eval_dataset=val_fold_ds,
            compute_metrics=compute_metrics,
        )

        # Entraînement
        trainer.train()

        # Évaluation sur le set de validation du fold
        eval_metrics = trainer.evaluate()
        print(f"Résultats du Fold {fold + 1}: F1 = {eval_metrics['eval_f1']:.4f} | Recall = {eval_metrics['eval_recall']:.4f}")
        print(
            "Matrice de confusion:\n"
            f"                 Predit 0    Predit 1\n"
            f"Vrai 0           {int(eval_metrics['eval_true_negative']):>8}    {int(eval_metrics['eval_false_positive']):>8}\n"
            f"Vrai 1           {int(eval_metrics['eval_false_negative']):>8}    {int(eval_metrics['eval_true_positive']):>8}"
        )
        print(
            f"Vrais positifs validation: {int(eval_metrics['eval_true_positive_count'])} | "
            f"Predictions positives: {int(eval_metrics['eval_predicted_positive_count'])}"
        )
        fold_results.append(eval_metrics['eval_f1'])

        # Sauvegarde
        trainer.save_model(fold_model_dir)

    print("\n" + "="*40)
    print(" BILAN DE LA CROSS-VALIDATION")
    print("="*40)
    print(f"Scores F1 par fold : {[round(score, 4) for score in fold_results]}")
    print(f"F1 Moyen : {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"Entraînement terminé, les modèles sont sauvegardés dans : {MODELS_DIR}")

if __name__ == "__main__":
    main()
