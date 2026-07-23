import json
import os
import random
import sys

# Permet de retrouver le package "src" même si le fichier est lancé
# depuis un environnement où la racine du projet n'est pas déjà dans sys.path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Variables définies avant l'import de PyTorch.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.metrics import log_loss
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    set_seed,
)

from src.config import (
    K_FOLDS,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_CSV,
    TRAIN_CV_CSV,
)
from src.data_loader import get_tokenized_datasets
from src.threshold_metrics import (
    compute_threshold_metrics,
    logits_to_positive_probabilities,
)


DEFAULT_DECISION_THRESHOLD = 0.5

DECISION_THRESHOLD_PATH = os.path.join(
    MODELS_DIR,
    "decision_threshold.json",
)


def configure_determinism(seed: int) -> None:
    """Configure autant que possible une prédiction reproductible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except (AttributeError, RuntimeError):
        pass


def print_metrics_block(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))

    print(f"Seuil     : {metrics['threshold']:.3f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Précision : {metrics['precision']:.4f}")

    print("\nMatrice de confusion :")
    print("                 Prédit 0    Prédit 1")

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


def load_optimized_threshold():
    """
    Charge le seuil calculé sur les prédictions out-of-fold.

    Si le fichier n'existe pas ou est invalide, le seuil 0.5 est utilisé.
    """
    if not os.path.exists(DECISION_THRESHOLD_PATH):
        print(
            "Aucun seuil optimisé trouvé. "
            "Le seuil 0.5 sera utilisé."
        )

        return DEFAULT_DECISION_THRESHOLD

    try:
        with open(
            DECISION_THRESHOLD_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            threshold_summary = json.load(file)

        return float(
            threshold_summary["best_threshold"]
        )

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            "Seuil optimisé inutilisable "
            f"({error}). Le seuil 0.5 sera utilisé."
        )

        return DEFAULT_DECISION_THRESHOLD


def main():
    configure_determinism(RANDOM_SEED)

    if not os.path.exists(TRAIN_CV_CSV):
        raise FileNotFoundError(
            f"Fichier introuvable : {TRAIN_CV_CSV}"
        )

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"Fichier introuvable : {TEST_CSV}"
        )

    # On réutilise exactement la même tokenisation que pendant l'entraînement.
    _, test_ds = get_tokenized_datasets(
        TRAIN_CV_CSV,
        TEST_CSV,
    )

    optimized_threshold = load_optimized_threshold()

    print("\n" + "=" * 70)
    print(
        " ÉVALUATION FINALE PAR ENSEMBLE "
        f"DES {K_FOLDS} FOLDS"
    )
    print("=" * 70)

    print(
        "Chaque modèle prédit le même jeu de test. "
        "Les probabilités de classe 1 sont ensuite moyennées."
    )

    print(
        f"Seuil optimisé chargé : "
        f"{optimized_threshold:.3f}"
    )

    all_fold_probabilities = []
    test_labels = None

    for fold_index in range(1, K_FOLDS + 1):
        fold_model_path = os.path.join(
            MODELS_DIR,
            f"fold_{fold_index}",
        )

        if not os.path.isdir(fold_model_path):
            raise FileNotFoundError(
                "Modèle de fold introuvable : "
                f"{fold_model_path}"
            )

        print(
            f"\nPrédiction avec fold_{fold_index}..."
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                fold_model_path
            )
        )

        # Aucun TrainingArguments n'est nécessaire pour une simple prédiction.
        # Cela évite aussi les incompatibilités entre versions de transformers.
        trainer = Trainer(model=model)

        prediction_output = trainer.predict(
            test_ds
        )

        fold_probabilities = (
            logits_to_positive_probabilities(
                prediction_output.predictions
            )
        )

        current_labels = np.asarray(
            prediction_output.label_ids,
            dtype=int,
        )

        if test_labels is None:
            test_labels = current_labels

        elif not np.array_equal(
            test_labels,
            current_labels,
        ):
            raise RuntimeError(
                "Les labels du test diffèrent "
                "entre les prédictions des folds."
            )

        all_fold_probabilities.append(
            fold_probabilities
        )

        # Libération de la mémoire avant de charger le fold suivant.
        del trainer
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probability_matrix = np.stack(
        all_fold_probabilities,
        axis=0,
    )

    ensemble_probabilities = np.mean(
        probability_matrix,
        axis=0,
    )

    default_metrics = compute_threshold_metrics(
        labels=test_labels,
        probabilities=ensemble_probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )

    optimized_metrics = compute_threshold_metrics(
        labels=test_labels,
        probabilities=ensemble_probabilities,
        threshold=optimized_threshold,
    )

    print("\n" + "=" * 70)
    print(
        f" RÉSULTATS DE L'ENSEMBLE DES {K_FOLDS} FOLDS"
    )
    print("=" * 70)

    print_metrics_block(
        title="ENSEMBLE AVEC LE SEUIL STANDARD",
        metrics=default_metrics,
    )

    print_metrics_block(
        title="ENSEMBLE AVEC LE SEUIL OPTIMISÉ OOF",
        metrics=optimized_metrics,
    )

    probabilities_for_log_loss = np.column_stack(
        [
            1.0 - ensemble_probabilities,
            ensemble_probabilities,
        ]
    )

    ensemble_log_loss = log_loss(
        test_labels,
        probabilities_for_log_loss,
        labels=[0, 1],
    )

    print("\nCOMPARAISON")
    print("-" * 40)

    print(
        f"Variation F1        : "
        f"{optimized_metrics['f1'] - default_metrics['f1']:+.4f}"
    )

    print(
        f"Variation précision : "
        f"{optimized_metrics['precision'] - default_metrics['precision']:+.4f}"
    )

    print(
        f"Variation recall    : "
        f"{optimized_metrics['recall'] - default_metrics['recall']:+.4f}"
    )

    print(
        f"Log loss ensemble   : "
        f"{ensemble_log_loss:.4f}"
    )

    mean_fold_disagreement = float(
        np.mean(
            np.std(
                probability_matrix,
                axis=0,
            )
        )
    )

    print(
        "Écart moyen entre les folds "
        "sur la probabilité de classe 1 : "
        f"{mean_fold_disagreement:.4f}"
    )


if __name__ == "__main__":
    main()
