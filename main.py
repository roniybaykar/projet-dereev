import inspect
import json
import os
import random
import shutil
import sys

# Re-lance automatiquement Python une seule fois avec PYTHONHASHSEED fixé.
# L'utilisateur peut donc simplement exécuter : python3 -m main
_DEFAULT_HASH_SEED = "42"
if os.environ.get("PYTHONHASHSEED") != _DEFAULT_HASH_SEED:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = _DEFAULT_HASH_SEED
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        environment,
    )

# Doit être défini avant l'import de torch et avant l'initialisation de CUDA.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    EPOCHS,
    K_FOLDS,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_CSV,
    TRAIN_CV_CSV,
)
from src.data_loader import get_tokenized_datasets
from src.model import get_model
from src.threshold_metrics import (
    compute_threshold_metrics,
    evaluate_thresholds,
    logits_to_positive_probabilities,
    plot_threshold_curve,
)


DEFAULT_DECISION_THRESHOLD = 0.5

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

DECISION_THRESHOLD_PATH = os.path.join(
    MODELS_DIR,
    "decision_threshold.json",
)

THRESHOLD_RESULTS_PATH = os.path.join(
    MODELS_DIR,
    "threshold_results.csv",
)

THRESHOLD_PLOT_PATH = os.path.join(
    MODELS_DIR,
    "threshold_results.png",
)

OOF_PREDICTIONS_PATH = os.path.join(
    MODELS_DIR,
    "oof_predictions.csv",
)

EXPERIMENT_SUMMARY_PATH = os.path.join(
    MODELS_DIR,
    "experiment_summary.json",
)


def configure_determinism(seed: int) -> None:
    """
    Fixe toutes les graines accessibles afin que deux exécutions
    sur la même machine et avec les mêmes versions soient reproductibles.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

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

    # Force autant que possible des algorithmes déterministes.
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )


def build_training_arguments(
    fold_model_dir: str,
    fold_index: int,
    fold_seed: int,
):
    """Construit des TrainingArguments compatibles avec la version installée."""
    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    arguments = {
        "output_dir": fold_model_dir,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "seed": fold_seed,
        "data_seed": fold_seed,
        "full_determinism": True,
        "dataloader_num_workers": 0,
        "logging_dir": os.path.join(
            "logs",
            f"fold_{fold_index}",
        ),
        "logging_steps": 10,
        "report_to": "none",
    }

    if "eval_strategy" in parameters:
        arguments["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        arguments["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "Votre version de transformers ne propose ni "
            "eval_strategy ni evaluation_strategy."
        )

    unsupported_arguments = [
        name for name in arguments if name not in parameters
    ]

    for name in unsupported_arguments:
        arguments.pop(name)

    if unsupported_arguments:
        print(
            "Arguments ignorés car non pris en charge par "
            "votre version de transformers : "
            + ", ".join(unsupported_arguments)
        )

    return TrainingArguments(**arguments)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = logits_to_positive_probabilities(
        logits
    )

    return compute_threshold_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )


def print_metrics_block(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))

    print(f"Seuil     : {metrics['threshold']:.3f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Précision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")

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


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_function(
            logits,
            labels,
        )

        if return_outputs:
            return loss, outputs

        return loss


def remove_previous_outputs() -> None:
    """
    Supprime les résultats globaux de l'expérience précédente.
    Les dossiers fold_X sont supprimés séparément avant chaque fold.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    output_files = [
        DECISION_THRESHOLD_PATH,
        THRESHOLD_RESULTS_PATH,
        OOF_PREDICTIONS_PATH,
        EXPERIMENT_SUMMARY_PATH,
    ]

    for path in output_files:
        if os.path.exists(path):
            os.remove(path)


def main():
    configure_determinism(RANDOM_SEED)
    remove_previous_outputs()

    if not os.path.exists(TRAIN_CV_CSV):
        print(f"Fichier introuvable : {TRAIN_CV_CSV}")
        print("Lancez d'abord le preprocessing.")
        return

    if not os.path.exists(TEST_CSV):
        print(f"Fichier introuvable : {TEST_CSV}")
        print("Lancez d'abord le preprocessing.")
        return

    train_cv_ds, _ = get_tokenized_datasets(
        TRAIN_CV_CSV,
        TEST_CSV,
    )

    labels = np.asarray(
        train_cv_ds["labels"],
        dtype=int,
    )

    skf = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_results = []
    fold_summaries = []

    oof_probabilities = np.full(
        len(labels),
        np.nan,
        dtype=float,
    )

    oof_folds = np.full(
        len(labels),
        -1,
        dtype=int,
    )

    split_iterator = skf.split(
        np.zeros(len(labels)),
        labels,
    )

    for fold_index, (train_idx, val_idx) in enumerate(
        split_iterator,
        start=1,
    ):
        # Une graine déterministe différente par fold.
        # Elle reste identique entre deux exécutions.
        fold_seed = RANDOM_SEED + fold_index
        configure_determinism(fold_seed)

        print("\n" + "=" * 50)
        print(
            f" DÉMARRAGE DU FOLD "
            f"{fold_index}/{K_FOLDS}"
        )
        print(f" Graine du fold : {fold_seed}")
        print("=" * 50)

        train_fold_ds = train_cv_ds.select(
            train_idx
        )

        val_fold_ds = train_cv_ds.select(
            val_idx
        )

        train_labels = np.asarray(
            train_fold_ds["labels"],
            dtype=int,
        )

        class_counts = np.bincount(
            train_labels,
            minlength=2,
        )

        if np.any(class_counts == 0):
            raise RuntimeError(
                "Une classe est absente du fold "
                f"{fold_index}."
            )

        class_weights = (
            class_counts.sum()
            / (
                len(class_counts)
                * class_counts
            )
        )

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
        )

        print(
            "Répartition train fold : "
            f"classe 0 = {class_counts[0]}, "
            f"classe 1 = {class_counts[1]}"
        )

        print(
            "Poids de loss : "
            f"classe 0 = {class_weights[0]:.4f}, "
            f"classe 1 = {class_weights[1]:.4f}"
        )

        fold_model_dir = os.path.join(
            MODELS_DIR,
            f"fold_{fold_index}",
        )

        # Évite de mélanger de vieux checkpoints
        # avec le nouvel entraînement.
        if os.path.exists(fold_model_dir):
            shutil.rmtree(fold_model_dir)

        training_args = build_training_arguments(
            fold_model_dir=fold_model_dir,
            fold_index=fold_index,
            fold_seed=fold_seed,
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,

            # Important :
            # le Trainer crée le modèle après avoir fixé sa graine.
            model_init=get_model,

            args=training_args,
            train_dataset=train_fold_ds,
            eval_dataset=val_fold_ds,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=(
                        EARLY_STOPPING_PATIENCE
                    ),
                    early_stopping_threshold=(
                        EARLY_STOPPING_THRESHOLD
                    ),
                )
            ],
        )

        trainer.train()

        print(
            "\nMeilleur checkpoint : "
            f"{trainer.state.best_model_checkpoint}"
        )

        print(
            "Meilleur F1 de validation : "
            f"{trainer.state.best_metric}"
        )

        print(
            "Epoch finale : "
            f"{trainer.state.epoch}"
        )

        prediction_output = trainer.predict(
            val_fold_ds
        )

        validation_probabilities = (
            logits_to_positive_probabilities(
                prediction_output.predictions
            )
        )

        validation_labels = np.asarray(
            prediction_output.label_ids,
            dtype=int,
        )

        oof_probabilities[val_idx] = (
            validation_probabilities
        )

        oof_folds[val_idx] = fold_index

        fold_metrics = compute_threshold_metrics(
            labels=validation_labels,
            probabilities=validation_probabilities,
            threshold=DEFAULT_DECISION_THRESHOLD,
        )

        print_metrics_block(
            title=(
                f"RÉSULTATS DU FOLD "
                f"{fold_index} AU SEUIL 0.5"
            ),
            metrics=fold_metrics,
        )

        fold_results.append(
            fold_metrics["f1"]
        )

        fold_summaries.append({
            "fold": fold_index,
            "seed": fold_seed,
            "best_checkpoint": (
                trainer.state.best_model_checkpoint
            ),
            "best_validation_metric": (
                trainer.state.best_metric
            ),
            "final_epoch": trainer.state.epoch,
            "validation_metrics": fold_metrics,
        })

        # Le meilleur checkpoint a déjà été rechargé
        # grâce à load_best_model_at_end=True.
        trainer.save_model(
            fold_model_dir
        )

    print("\n" + "=" * 50)
    print(" BILAN DE LA CROSS-VALIDATION")
    print("=" * 50)

    rounded_scores = [
        round(score, 4)
        for score in fold_results
    ]

    mean_f1 = float(
        np.mean(fold_results)
    )

    std_f1 = float(
        np.std(fold_results)
    )

    print(
        f"Scores F1 par fold : "
        f"{rounded_scores}"
    )

    print(
        f"F1 moyen : "
        f"{mean_f1:.4f} "
        f"(+/- {std_f1:.4f})"
    )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(
            "Certaines observations n'ont pas reçu "
            "de probabilité out-of-fold."
        )

    if np.any(oof_folds == -1):
        raise RuntimeError(
            "Certaines observations ne sont associées "
            "à aucun fold."
        )

    threshold_results, best_result = (
        evaluate_thresholds(
            labels=labels,
            probabilities=oof_probabilities,
            minimum=THRESHOLD_MIN,
            maximum=THRESHOLD_MAX,
            step=THRESHOLD_STEP,
        )
    )

    default_result = compute_threshold_metrics(
        labels=labels,
        probabilities=oof_probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )

    pd.DataFrame(
        threshold_results
    ).to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

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

    experiment_summary = {
        "random_seed": RANDOM_SEED,
        "k_folds": K_FOLDS,
        "fold_f1_scores": fold_results,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "folds": fold_summaries,
    }

    with open(
        EXPERIMENT_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_summary,
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
        "Chaque observation est prédite par le fold "
        "qui ne l'a pas vue pendant l'entraînement."
    )

    print_metrics_block(
        title="SEUIL STANDARD",
        metrics=default_result,
    )

    print_metrics_block(
        title="MEILLEUR SEUIL POUR LE F1",
        metrics=best_result,
    )

    f1_gain = (
        best_result["f1"]
        - default_result["f1"]
    )

    print(
        "\nGain de F1 par rapport "
        f"au seuil 0.5 : {f1_gain:+.4f}"
    )

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

    plot_threshold_curve(
        results=threshold_results,
        output_path=THRESHOLD_PLOT_PATH,
    )

    print(
        "\nGraphique F1 vs seuil sauvegardé dans : "
        f"{THRESHOLD_PLOT_PATH}"
    )

    print(
        "\nSeuil sauvegardé dans : "
        f"{DECISION_THRESHOLD_PATH}"
    )

    print(
        "Résultats détaillés : "
        f"{THRESHOLD_RESULTS_PATH}"
    )

    print(
        "Prédictions out-of-fold : "
        f"{OOF_PREDICTIONS_PATH}"
    )

    print(
        "Résumé de l'expérience : "
        f"{EXPERIMENT_SUMMARY_PATH}"
    )

    print(
        "Entraînement terminé. "
        f"Les {K_FOLDS} modèles sont dans : "
        f"{MODELS_DIR}"
    )


if __name__ == "__main__":
    main()
    