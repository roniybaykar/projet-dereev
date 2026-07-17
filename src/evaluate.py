import json
import os

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
)

from src.config import (
    MODELS_DIR,
    TEST_CSV,
    TRAIN_CV_CSV,
)
from src.data_loader import (
    get_tokenized_datasets,
)
from src.threshold_metrics import (
    compute_threshold_metrics,
    logits_to_positive_probabilities,
)


DEFAULT_DECISION_THRESHOLD = 0.5

DECISION_THRESHOLD_PATH = os.path.join(
    MODELS_DIR,
    "decision_threshold.json",
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
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Précision : "
        f"{metrics['precision']:.4f}"
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


def load_optimized_threshold():
    if not os.path.exists(
        DECISION_THRESHOLD_PATH
    ):
        raise FileNotFoundError(
            "Seuil optimisé introuvable. "
            "Lancez main.py avant evaluate.py : "
            f"{DECISION_THRESHOLD_PATH}"
        )

    with open(
        DECISION_THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_summary = json.load(
            file
        )

    threshold = float(
        threshold_summary[
            "best_threshold"
        ]
    )

    return (
        threshold,
        threshold_summary,
    )


def main():
    best_fold = 4

    best_model_path = os.path.join(
        MODELS_DIR,
        f"fold_{best_fold}",
    )

    if not os.path.exists(
        best_model_path
    ):
        print(
            f"Modèle introuvable : "
            f"{best_model_path}"
        )
        print(
            "Lancez main.py avant "
            "evaluate.py."
        )
        return

    try:
        (
            optimized_threshold,
            threshold_summary,
        ) = load_optimized_threshold()

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            "Impossible de charger "
            f"le seuil optimisé : {error}"
        )
        return

    print("\n" + "=" * 60)
    print(
        " ÉVALUATION FINALE SUR TEST.CSV"
    )
    print("=" * 60)

    print(
        f"Modèle utilisé : "
        f"fold_{best_fold}"
    )

    print(
        f"Seuil optimisé chargé : "
        f"{optimized_threshold:.3f}"
    )

    print(
        "Le seuil a été choisi uniquement "
        "sur les validations out-of-fold."
    )

    print(
        "Le test final n'a pas participé "
        "au choix du seuil."
    )

    _, test_ds = (
        get_tokenized_datasets(
            TRAIN_CV_CSV,
            TEST_CSV,
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            best_model_path
        )
    )

    trainer = Trainer(
        model=model
    )

    prediction_output = trainer.predict(
        test_ds
    )

    test_probabilities = (
        logits_to_positive_probabilities(
            prediction_output.predictions
        )
    )

    test_labels = (
        prediction_output.label_ids
    )

    default_metrics = (
        compute_threshold_metrics(
            labels=test_labels,
            probabilities=(
                test_probabilities
            ),
            threshold=(
                DEFAULT_DECISION_THRESHOLD
            ),
        )
    )

    optimized_metrics = (
        compute_threshold_metrics(
            labels=test_labels,
            probabilities=(
                test_probabilities
            ),
            threshold=(
                optimized_threshold
            ),
        )
    )

    print_metrics_block(
        title=(
            "RÉSULTATS AVEC "
            "LE SEUIL STANDARD"
        ),
        metrics=default_metrics,
    )

    print_metrics_block(
        title=(
            "RÉSULTATS AVEC "
            "LE SEUIL OPTIMISÉ"
        ),
        metrics=optimized_metrics,
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

    test_loss = (
        prediction_output.metrics.get(
            "test_loss"
        )
    )

    if test_loss is not None:
        print(
            f"Loss finale         : "
            f"{test_loss:.4f}"
        )


if __name__ == "__main__":
    main()
