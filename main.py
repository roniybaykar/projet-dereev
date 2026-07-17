import os

from webencodings import labels
os.environ["HF_HUB_DISABLE_XET"] = "1"
import json
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from src.config import TRAIN_CV_CSV, TEST_CSV, MODELS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, K_FOLDS, RANDOM_SEED, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD
from src.data_loader import get_tokenized_datasets
from src.model import get_model
from sklearn.model_selection import StratifiedKFold

from src.threshold_metrics import (
    compute_threshold_metrics,
    evaluate_thresholds,
    logits_to_positive_probabilities,
)

# Seuil standard utilisé pendant l'entraînement.
DEFAULT_DECISION_THRESHOLD = 0.5

# Seuils qui seront testés après les cinq folds.
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

# Fichiers produits après la recherche du meilleur seuil.
DECISION_THRESHOLD_PATH = os.path.join(
    MODELS_DIR,
    "decision_threshold.json",
)

THRESHOLD_RESULTS_PATH = os.path.join(
    MODELS_DIR,
    "threshold_results.csv",
)

OOF_PREDICTIONS_PATH = os.path.join(
    MODELS_DIR,
    "oof_predictions.csv",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = (
        logits_to_positive_probabilities(
            logits
        )
    )

    return compute_threshold_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )

def print_metrics_block(
    title,
    metrics,
):
    print(f"\n{title}")
    print("-" * len(title))

    print(
        f"Seuil     : "
        f"{metrics['threshold']:.3f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Précision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print("\nMatrice de confusion :")
    print(
        "                 "
        "Prédit 0    Prédit 1"
    )

    print(
        f"Vrai 0           "
        f"{metrics['true_negative']:>8}    "
        f"{metrics['false_positive']:>8}"
    )

    print(
        f"Vrai 1           "
        f"{metrics['false_negative']:>8}    "
        f"{metrics['true_positive']:>8}"
    )

    print(
        f"\nVrais positifs : "
        f"{metrics['true_positive_count']} | "
        f"Prédictions positives : "
        f"{metrics['predicted_positive_count']}"
    )

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
    
    # Une probabilité sera stockée pour chaque observation.
    oof_probabilities = np.full(
    len(labels),
    np.nan,
    dtype=float,
    )

    # Fold ayant produit la probabilité de chaque observation.
    oof_folds = np.full(
    len(labels),
    -1,
    dtype=int,
    )

    

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
            greater_is_better=True,
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
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
                )
            ],
        )

        # Entraînement
        trainer.train()

        # Le meilleur modèle du fold prédit sa validation.
        prediction_output = trainer.predict(
            val_fold_ds
        )

        # Trainer renvoie des logits et non des probabilités.
        validation_probabilities = (
            logits_to_positive_probabilities(
                prediction_output.predictions
            )
        )

        validation_labels = np.asarray(
            prediction_output.label_ids,
            dtype=int,
        )

        # Stockage à la position originale des observations.
        #
        # Chaque probabilité est produite par un modèle
        # qui n'a jamais entraîné l'observation correspondante.
        oof_probabilities[val_idx] = (
            validation_probabilities
        )

        oof_folds[val_idx] = fold + 1

        # Affichage du résultat du fold au seuil standard.
        fold_metrics = compute_threshold_metrics(
            labels=validation_labels,
            probabilities=validation_probabilities,
            threshold=DEFAULT_DECISION_THRESHOLD,
        )

        print_metrics_block(
            title=(
                f"RÉSULTATS DU FOLD "
                f"{fold + 1} AU SEUIL 0.5"
            ),
            metrics=fold_metrics,
        )

        fold_results.append(
            fold_metrics["f1"]
        )

        # Sauvegarde
        trainer.save_model(fold_model_dir)

    print("\n" + "="*40)
    print(" BILAN DE LA CROSS-VALIDATION")
    print("="*40)
    print(f"Scores F1 par fold : {[round(score, 4) for score in fold_results]}")
    print(f"F1 Moyen : {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
        # Vérification : chaque observation doit avoir été
    # placée une fois en validation.
    if np.isnan(
        oof_probabilities
    ).any():
        raise RuntimeError(
            "Certaines observations n'ont pas reçu "
            "de probabilité out-of-fold."
        )

    if np.any(
        oof_folds == -1
    ):
        raise RuntimeError(
            "Certaines observations ne sont associées "
            "à aucun fold."
        )

    # Recherche du seuil maximisant le F1.
    threshold_results, best_result = (
        evaluate_thresholds(
            labels=labels,
            probabilities=oof_probabilities,
            minimum=THRESHOLD_MIN,
            maximum=THRESHOLD_MAX,
            step=THRESHOLD_STEP,
        )
    )

    # Résultat de référence avec le seuil 0.5.
    default_result = (
        compute_threshold_metrics(
            labels=labels,
            probabilities=oof_probabilities,
            threshold=(
                DEFAULT_DECISION_THRESHOLD
            ),
        )
    )

    os.makedirs(
        MODELS_DIR,
        exist_ok=True,
    )

    # Sauvegarde des résultats de tous les seuils.
    pd.DataFrame(
        threshold_results
    ).to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    # Sauvegarde des probabilités out-of-fold.
    pd.DataFrame({
        "row_index": np.arange(
            len(labels)
        ),
        "fold": oof_folds,
        "label": labels,
        "probability_class_1": (
            oof_probabilities
        ),
    }).to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    # Résumé du seuil retenu.
    threshold_summary = {
        "selection_data": (
            "out_of_fold_validation"
        ),
        "objective": "maximum_f1",
        "best_threshold": (
            best_result["threshold"]
        ),
        "best_metrics": best_result,
        "default_threshold": (
            DEFAULT_DECISION_THRESHOLD
        ),
        "default_metrics": default_result,
    }

    with open(
        DECISION_THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print(
        " OPTIMISATION DU SEUIL SUR "
        "LES VALIDATIONS OUT-OF-FOLD"
    )
    print("=" * 60)

    print(
        "Le test final n'est pas utilisé ici. "
        "Chaque probabilité provient d'un modèle "
        "qui n'a pas entraîné l'observation."
    )

    print_metrics_block(
        title="SEUIL STANDARD",
        metrics=default_result,
    )

    print_metrics_block(
        title=(
            "MEILLEUR SEUIL POUR LE F1"
        ),
        metrics=best_result,
    )

    f1_gain = (
        best_result["f1"]
        - default_result["f1"]
    )

    print(
        f"\nGain de F1 par rapport "
        f"au seuil 0.5 : {f1_gain:+.4f}"
    )

    # Afficher les dix meilleurs seuils.
    top_results = sorted(
        threshold_results,
        key=lambda result: result["f1"],
        reverse=True,
    )[:10]

    print("\nTop 10 des seuils testés :")

    print(
        " Seuil |    F1 | Précision | "
        "Recall |  FP |  FN"
    )

    print("-" * 55)

    for result in top_results:
        print(
            f" {result['threshold']:>5.2f} | "
            f"{result['f1']:>5.3f} | "
            f"{result['precision']:>9.3f} | "
            f"{result['recall']:>6.3f} | "
            f"{result['false_positive']:>3} | "
            f"{result['false_negative']:>3}"
        )

    print(
        f"\nSeuil sauvegardé dans : "
        f"{DECISION_THRESHOLD_PATH}"
    )

    print(
        f"Résultats détaillés : "
        f"{THRESHOLD_RESULTS_PATH}"
    )

    print(
        f"Prédictions out-of-fold : "
        f"{OOF_PREDICTIONS_PATH}"
    )
    print(f"Entraînement terminé, les modèles sont sauvegardés dans : {MODELS_DIR}")

if __name__ == "__main__":
    main()
