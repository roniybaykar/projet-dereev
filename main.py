"""Fine-tuning multi-seeds de CamemBERT-bio pour le benchmark DEREEV.

Protocole A strict
------------------
1. une unique répartition stratifiée en K folds est créée avec CV_SPLIT_SEED ;
2. ce manifeste fixe est réutilisé pour toutes les seeds d'entraînement ;
3. tous les modèles du benchmark rechargent exactement le même manifeste ;
4. chaque seed d'entraînement lance K modèles indépendants sur les mêmes folds ;
5. les prédictions out-of-fold sont assemblées pour chaque seed ;
6. le seuil maximisant le F1 positif est choisi uniquement sur les OOF.

Commande recommandée depuis la racine du projet :
    python3 -m main --mode cv

Pour relancer en écrasant uniquement les anciens résultats de ce modèle :
    python3 -m main --mode cv --overwrite
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Sequence, Tuple

# PYTHONHASHSEED ne peut pas changer au milieu d'un même processus.
# On le fixe donc à une valeur constante pour neutraliser cette source de bruit,
# tandis que les seeds expérimentales pilotent NumPy, PyTorch et Transformers.
_FIXED_HASH_SEED = "0"

if os.environ.get("PYTHONHASHSEED") != _FIXED_HASH_SEED:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = _FIXED_HASH_SEED
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "main", *sys.argv[1:]],
        environment,
    )

# Doit être défini avant l'import de torch / CUDA.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import (
    BATCH_SIZE,
    BENCHMARK_MODELS_DIR,
    CV_SPLITS_DIR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    EPOCHS,
    CV_SPLIT_SEED,
    TRAINING_SEEDS,
    K_FOLDS,
    GROUP_COL,
    LABEL_COL,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    MODEL_OUTPUT_NAME,
    MODELS_DIR,
    TEST_CSV,
    TEXT_COL,
    TRAIN_CV_CSV,
)
from src.cv_protocol import (
    SharedSplit,
    dataset_fingerprint,
    ensure_shared_split,
    parse_seed_list,
)
from src.data_loader import get_tokenized_datasets
from src.model import get_model
from src.threshold_metrics import (
    add_pr_auc_to_metrics,
    compute_pr_auc,
    compute_threshold_metrics,
    evaluate_thresholds,
    logits_to_positive_probabilities,
    plot_precision_recall_curve,
    plot_threshold_curve,
)


DEFAULT_DECISION_THRESHOLD = 0.5
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

LATEST_EXPERIMENT_PATH = os.path.join(MODELS_DIR, "latest_experiment.json")
MODEL_ROOT = os.path.join(BENCHMARK_MODELS_DIR, MODEL_OUTPUT_NAME)


# ---------------------------------------------------------------------
# AFFICHAGE TERMINAL
# ---------------------------------------------------------------------
def print_banner(title: str, width: int = 84, character: str = "=") -> None:
    print("\n" + character * width)
    print(f" {title}")
    print(character * width)


def print_key_values(rows: Sequence[Tuple[str, object]]) -> None:
    key_width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        print(f"{str(key):<{key_width}} : {value}")


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        print("(aucune ligne)")
        return

    rendered_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in rendered_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header_line = "| " + " | ".join(
        header.ljust(widths[index])
        for index, header in enumerate(headers)
    ) + " |"

    print(separator)
    print(header_line)
    print(separator)

    for row in rendered_rows:
        print(
            "| "
            + " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )

    print(separator)


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes:02d} min {seconds:02d} s"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"


def sample_std(values: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    if len(values_array) < 2:
        return 0.0
    return float(np.std(values_array, ddof=1))


def print_metrics_block(title: str, metrics: Dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print_key_values(
        [
            ("Seuil", f"{metrics['threshold']:.3f}"),
            ("F1 positif", f"{metrics['f1']:.4f}"),
            ("PR-AUC", f"{metrics.get('pr_auc', float('nan')):.4f}"),
            ("Recall", f"{metrics['recall']:.4f}"),
            ("Précision", f"{metrics['precision']:.4f}"),
            ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ]
    )

    print("\nMatrice de confusion")
    print_table(
        ["Réel / Prédit", "Classe 0", "Classe 1"],
        [
            ["Classe 0", metrics["true_negative"], metrics["false_positive"]],
            ["Classe 1", metrics["false_negative"], metrics["true_positive"]],
        ],
    )


# ---------------------------------------------------------------------
# REPRODUCTIBILITÉ ET CONFIGURATION
# ---------------------------------------------------------------------
def configure_determinism(seed: int) -> None:
    """Fixe les principales sources d'aléatoire pour un entraînement."""
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

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def device_description() -> str:
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(0)}"
    return "CPU"


def build_training_arguments(
    output_dir: str,
    run_name: str,
    seed: int,
) -> TrainingArguments:
    """Construit des TrainingArguments compatibles avec plusieurs versions."""
    parameters = inspect.signature(TrainingArguments.__init__).parameters

    arguments = {
        "output_dir": output_dir,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "seed": seed,
        "data_seed": seed,
        "full_determinism": True,
        "dataloader_num_workers": 0,
        "logging_dir": os.path.join("logs", run_name),
        "logging_steps": 10,
        "report_to": "none",
    }

    if "eval_strategy" in parameters:
        arguments["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        arguments["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "Votre version de transformers ne propose ni 'eval_strategy' "
            "ni 'evaluation_strategy'."
        )

    unsupported = [name for name in arguments if name not in parameters]
    for name in unsupported:
        arguments.pop(name)

    if unsupported:
        print(
            "Arguments ignorés par votre version de transformers : "
            + ", ".join(unsupported)
        )

    return TrainingArguments(**arguments)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = logits_to_positive_probabilities(logits)

    metrics = compute_threshold_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )

    return add_pr_auc_to_metrics(
        metrics=metrics,
        labels=labels,
        probabilities=probabilities,
    )


# ---------------------------------------------------------------------
# TRAINER PONDÉRÉ
# ---------------------------------------------------------------------
def compute_class_weights(dataset) -> Tuple[torch.Tensor, np.ndarray]:
    labels = np.asarray(dataset["labels"], dtype=int)
    class_counts = np.bincount(labels, minlength=2)

    if np.any(class_counts == 0):
        raise RuntimeError(
            "Une des deux classes est absente des données d'entraînement."
        )

    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    return (
        torch.tensor(class_weights, dtype=torch.float32),
        class_counts,
    )


class WeightedLossTrainer(Trainer):
    """Trainer utilisant une CrossEntropyLoss pondérée."""

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
        loss = loss_function(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss


# ---------------------------------------------------------------------
# CHEMINS ET SAUVEGARDES
# ---------------------------------------------------------------------
def prepare_model_root(overwrite: bool) -> None:
    if os.path.exists(MODEL_ROOT):
        if not overwrite:
            raise FileExistsError(
                f"Le dossier existe déjà : {MODEL_ROOT}\n"
                "Utilisez --overwrite pour relancer ce modèle. "
                "Les manifests partagés de folds ne seront pas supprimés."
            )
        shutil.rmtree(MODEL_ROOT)

    os.makedirs(MODEL_ROOT, exist_ok=True)


def seed_directory(training_seed: int) -> str:
    return os.path.join(MODEL_ROOT, f"seed_{training_seed}")


def seed_paths(seed_dir: str) -> Dict[str, str]:
    return {
        "threshold_json": os.path.join(seed_dir, "decision_threshold.json"),
        "threshold_csv": os.path.join(seed_dir, "threshold_results.csv"),
        "threshold_plot": os.path.join(seed_dir, "threshold_results.png"),
        "pr_curve_plot": os.path.join(seed_dir, "precision_recall_curve.png"),
        "predictions_csv": os.path.join(
            seed_dir,
            "validation_predictions.csv",
        ),
        "summary_json": os.path.join(seed_dir, "experiment_summary.json"),
    }


def save_threshold_outputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
    row_indices: np.ndarray,
    fold_ids: np.ndarray,
    training_seed: int,
    paths: Dict[str, str],
) -> Tuple[Dict, Dict]:
    threshold_results, best_result = evaluate_thresholds(
        labels=labels,
        probabilities=probabilities,
        minimum=THRESHOLD_MIN,
        maximum=THRESHOLD_MAX,
        step=THRESHOLD_STEP,
    )

    pr_auc = compute_pr_auc(labels=labels, probabilities=probabilities)

    default_result = add_pr_auc_to_metrics(
        metrics=compute_threshold_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=DEFAULT_DECISION_THRESHOLD,
        ),
        labels=labels,
        probabilities=probabilities,
    )

    best_result = add_pr_auc_to_metrics(
        metrics=best_result,
        labels=labels,
        probabilities=probabilities,
    )

    pd.DataFrame(threshold_results).to_csv(paths["threshold_csv"], index=False)

    pd.DataFrame(
        {
            "row_index": row_indices,
            "training_seed": training_seed,
            "fold": fold_ids,
            "label": labels,
            "probability_class_1": probabilities,
            "prediction_threshold_0_5": (
                probabilities >= DEFAULT_DECISION_THRESHOLD
            ).astype(int),
            "prediction_optimized_threshold": (
                probabilities >= best_result["threshold"]
            ).astype(int),
        }
    ).to_csv(paths["predictions_csv"], index=False)

    threshold_summary = {
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "training_seed": int(training_seed),
        "split_seed": int(CV_SPLIT_SEED),
        "selection_data": "out_of_fold_validation",
        "objective": "maximum_f1_positive_class",
        "pr_auc": pr_auc,
        "best_threshold": best_result["threshold"],
        "best_metrics": best_result,
        "default_threshold": DEFAULT_DECISION_THRESHOLD,
        "default_metrics": default_result,
    }

    with open(paths["threshold_json"], "w", encoding="utf-8") as file:
        json.dump(threshold_summary, file, indent=2, ensure_ascii=False)

    plot_threshold_curve(
        results=threshold_results,
        output_path=paths["threshold_plot"],
    )
    plot_precision_recall_curve(
        labels=labels,
        probabilities=probabilities,
        output_path=paths["pr_curve_plot"],
    )

    return default_result, best_result


# ---------------------------------------------------------------------
# UNE SEED D'ENTRAÎNEMENT
# ---------------------------------------------------------------------
def run_one_training_seed(
    train_cv_ds,
    labels: np.ndarray,
    shared_split: SharedSplit,
    training_seed: int,
    completed_before: int,
    total_trainings: int,
) -> Dict:
    fold_pairs = shared_split.indices()
    seed_dir = seed_directory(training_seed)
    os.makedirs(seed_dir, exist_ok=True)
    paths = seed_paths(seed_dir)

    seed_start = time.perf_counter()

    print_banner(
        f"SEED D'ENTRAÎNEMENT {training_seed} — {MODEL_DISPLAY_NAME}",
        character="#",
    )
    print_key_values(
        [
            ("Modèle", MODEL_DISPLAY_NAME),
            ("Identifiant Hugging Face", MODEL_NAME),
            ("Seed d'entraînement", training_seed),
            ("Seed du split fixe", shared_split.split_seed),
            ("Nombre de folds", shared_split.fold_count),
            ("Manifeste partagé", shared_split.csv_path),
            ("Dossier de sortie", seed_dir),
        ]
    )

    oof_probabilities = np.full(len(labels), np.nan, dtype=float)
    oof_folds = np.full(len(labels), -1, dtype=int)

    fold_f1_scores: List[float] = []
    fold_pr_auc_scores: List[float] = []
    fold_summaries: List[Dict] = []

    for fold_index, (train_idx, val_idx) in enumerate(fold_pairs, start=1):
        global_training_index = completed_before + fold_index
        configure_determinism(training_seed)
        fold_start = time.perf_counter()

        train_fold_ds = train_cv_ds.select(train_idx)
        val_fold_ds = train_cv_ds.select(val_idx)
        class_weights, class_counts = compute_class_weights(train_fold_ds)

        validation_counts = np.bincount(labels[val_idx], minlength=2)
        fold_dir = os.path.join(seed_dir, f"fold_{fold_index}")

        print_banner(
            (
                f"ENTRAÎNEMENT {global_training_index}/{total_trainings} "
                f"— SEED {training_seed} — FOLD {fold_index}/"
                f"{shared_split.fold_count}"
            ),
            width=84,
            character="-",
        )
        print_key_values(
            [
                ("Modèle", f"{MODEL_DISPLAY_NAME} ({MODEL_NAME})"),
                ("Seed", training_seed),
                ("Fold", f"{fold_index}/{shared_split.fold_count}"),
                ("Train", f"{len(train_idx)} notes"),
                (
                    "Répartition train",
                    f"classe 0 = {class_counts[0]} | classe 1 = {class_counts[1]}",
                ),
                ("Validation", f"{len(val_idx)} notes"),
                (
                    "Répartition validation",
                    (
                        f"classe 0 = {validation_counts[0]} | "
                        f"classe 1 = {validation_counts[1]}"
                    ),
                ),
                (
                    "Poids de loss",
                    (
                        f"classe 0 = {class_weights[0].item():.4f} | "
                        f"classe 1 = {class_weights[1].item():.4f}"
                    ),
                ),
                ("Sortie", fold_dir),
            ]
        )

        training_args = build_training_arguments(
            output_dir=fold_dir,
            run_name=(
                f"{MODEL_OUTPUT_NAME}_seed_{training_seed}_fold_{fold_index}"
            ),
            seed=training_seed,
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model_init=get_model,
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

        trainer.train()
        prediction_output = trainer.predict(val_fold_ds)

        probabilities = logits_to_positive_probabilities(
            prediction_output.predictions
        )
        validation_labels = np.asarray(
            prediction_output.label_ids,
            dtype=int,
        )

        oof_probabilities[val_idx] = probabilities
        oof_folds[val_idx] = fold_index

        metrics = add_pr_auc_to_metrics(
            metrics=compute_threshold_metrics(
                labels=validation_labels,
                probabilities=probabilities,
                threshold=DEFAULT_DECISION_THRESHOLD,
            ),
            labels=validation_labels,
            probabilities=probabilities,
        )

        fold_duration = time.perf_counter() - fold_start
        print_metrics_block(
            f"RÉSULTATS VALIDATION — SEED {training_seed} — FOLD {fold_index}",
            metrics,
        )
        print(f"Durée du fold : {format_duration(fold_duration)}")

        fold_f1_scores.append(metrics["f1"])
        fold_pr_auc_scores.append(metrics["pr_auc"])

        fold_summaries.append(
            {
                "fold": fold_index,
                "training_seed": training_seed,
                "train_size": int(len(train_idx)),
                "validation_size": int(len(val_idx)),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_validation_metric": trainer.state.best_metric,
                "final_epoch": trainer.state.epoch,
                "duration_seconds": float(fold_duration),
                "validation_metrics": metrics,
                "model_path": fold_dir,
            }
        )

        # load_best_model_at_end=True a déjà rechargé le meilleur checkpoint.
        trainer.save_model(fold_dir)

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_probabilities).any() or np.any(oof_folds == -1):
        raise RuntimeError(
            "Certaines observations n'ont pas reçu de prédiction out-of-fold."
        )

    default_result, optimized_result = save_threshold_outputs(
        labels=labels,
        probabilities=oof_probabilities,
        row_indices=np.arange(len(labels), dtype=int),
        fold_ids=oof_folds,
        training_seed=training_seed,
        paths=paths,
    )

    seed_duration = time.perf_counter() - seed_start

    print_banner(
        f"BILAN OOF — SEED {training_seed} — {MODEL_DISPLAY_NAME}",
        character="=",
    )
    print_table(
        ["Fold", "F1 positif", "PR-AUC"],
        [
            [
                fold_index,
                f"{fold_f1_scores[fold_index - 1]:.4f}",
                f"{fold_pr_auc_scores[fold_index - 1]:.4f}",
            ]
            for fold_index in range(1, shared_split.fold_count + 1)
        ],
    )
    print(
        f"Moyenne F1 des folds : {np.mean(fold_f1_scores):.4f} "
        f"± {sample_std(fold_f1_scores):.4f}"
    )
    print(
        f"Moyenne PR-AUC folds : {np.mean(fold_pr_auc_scores):.4f} "
        f"± {sample_std(fold_pr_auc_scores):.4f}"
    )

    print_metrics_block("OOF — SEUIL STANDARD 0.5", default_result)
    print_metrics_block(
        "OOF — SEUIL OPTIMISÉ SUR CETTE SEED",
        optimized_result,
    )
    print(
        "Gain de F1 grâce au seuil optimisé : "
        f"{optimized_result['f1'] - default_result['f1']:+.4f}"
    )
    print(f"Durée totale de la seed : {format_duration(seed_duration)}")

    summary = {
        "protocol": "fixed_shared_stratified_group_kfold_multi_training_seeds",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_output_name": MODEL_OUTPUT_NAME,
        "training_seed": training_seed,
        "split_seed": shared_split.split_seed,
        "fold_count": shared_split.fold_count,
        "split_manifest_csv": shared_split.csv_path,
        "split_manifest_json": shared_split.metadata_path,
        "dataset_fingerprint": shared_split.dataset_fingerprint,
        "fold_f1_scores": fold_f1_scores,
        "mean_fold_f1": float(np.mean(fold_f1_scores)),
        "std_fold_f1": sample_std(fold_f1_scores),
        "fold_pr_auc_scores": fold_pr_auc_scores,
        "mean_fold_pr_auc": float(np.mean(fold_pr_auc_scores)),
        "std_fold_pr_auc": sample_std(fold_pr_auc_scores),
        "folds": fold_summaries,
        "oof_default_metrics": default_result,
        "oof_optimized_metrics": optimized_result,
        "duration_seconds": float(seed_duration),
        "model_paths": [
            os.path.join(seed_dir, f"fold_{fold_index}")
            for fold_index in range(1, shared_split.fold_count + 1)
        ],
        "training_configuration": {
            "max_length": int(MAX_LENGTH),
            "batch_size": int(BATCH_SIZE),
            "epochs_max": int(EPOCHS),
            "learning_rate": float(LEARNING_RATE),
            "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
            "early_stopping_threshold": float(EARLY_STOPPING_THRESHOLD),
            "loss": "weighted_cross_entropy",
            "threshold_selection": "maximum_positive_f1_on_seed_oof_only",
        },
        "uncertainty_scope": (
            "Les écarts-types entre seeds sont calculés à split fixe et ne "
            "capturent pas la variance de partition des données."
        ),
    }

    with open(paths["summary_json"], "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------
# AGRÉGATION DES CINQ SEEDS
# ---------------------------------------------------------------------
def aggregate_seed_summaries(
    summaries: Sequence[Dict],
    total_duration: float,
) -> Dict:
    rows = []

    for summary in summaries:
        default_metrics = summary["oof_default_metrics"]
        optimized_metrics = summary["oof_optimized_metrics"]
        rows.append(
            {
                "training_seed": summary["training_seed"],
                "mean_fold_f1": summary["mean_fold_f1"],
                "std_fold_f1": summary["std_fold_f1"],
                "oof_f1_threshold_0_5": default_metrics["f1"],
                "oof_precision_threshold_0_5": default_metrics["precision"],
                "oof_recall_threshold_0_5": default_metrics["recall"],
                "oof_accuracy_threshold_0_5": default_metrics["accuracy"],
                "oof_pr_auc": default_metrics["pr_auc"],
                "optimized_threshold": optimized_metrics["threshold"],
                "oof_f1_optimized": optimized_metrics["f1"],
                "oof_precision_optimized": optimized_metrics["precision"],
                "oof_recall_optimized": optimized_metrics["recall"],
                "oof_accuracy_optimized": optimized_metrics["accuracy"],
                "duration_seconds": summary["duration_seconds"],
            }
        )

    results_dataframe = pd.DataFrame(rows)
    results_csv = os.path.join(MODEL_ROOT, "all_seeds_validation_results.csv")
    results_dataframe.to_csv(results_csv, index=False)

    metric_columns = [
        "oof_f1_threshold_0_5",
        "oof_precision_threshold_0_5",
        "oof_recall_threshold_0_5",
        "oof_accuracy_threshold_0_5",
        "oof_pr_auc",
        "optimized_threshold",
        "oof_f1_optimized",
        "oof_precision_optimized",
        "oof_recall_optimized",
        "oof_accuracy_optimized",
    ]

    aggregate_metrics = {}
    for column in metric_columns:
        values = results_dataframe[column].to_numpy(dtype=float)
        aggregate_metrics[column] = {
            "mean": float(np.mean(values)),
            "std": sample_std(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    benchmark_summary = {
        "protocol": "fixed_shared_stratified_group_kfold_multi_training_seeds",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_output_name": MODEL_OUTPUT_NAME,
        "training_seeds": [summary["training_seed"] for summary in summaries],
        "split_seed": summaries[0]["split_seed"],
        "split_manifest_csv": summaries[0]["split_manifest_csv"],
        "fold_count": summaries[0]["fold_count"],
        "total_fine_tunings": int(len(summaries) * summaries[0]["fold_count"]),
        "standard_deviation_definition": "sample_std_ddof_1_across_seeds",
        "error_bar_scope": (
            "Variabilité inter-seed uniquement, avec partition train/test et "
            "folds figés. La variance de split n'est pas incluse."
        ),
        "training_configuration": {
            "max_length": int(MAX_LENGTH),
            "batch_size": int(BATCH_SIZE),
            "epochs_max": int(EPOCHS),
            "learning_rate": float(LEARNING_RATE),
            "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
            "early_stopping_threshold": float(EARLY_STOPPING_THRESHOLD),
            "loss": "weighted_cross_entropy",
        },
        "aggregate_validation_metrics": aggregate_metrics,
        "seed_summaries": summaries,
        "all_seeds_validation_csv": results_csv,
        "total_duration_seconds": float(total_duration),
    }

    summary_json = os.path.join(MODEL_ROOT, "benchmark_summary.json")
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(benchmark_summary, file, indent=2, ensure_ascii=False)

    print_banner(
        f"RÉSULTAT FINAL MULTI-SEEDS — {MODEL_DISPLAY_NAME}",
        character="=",
    )
    print_table(
        [
            "Seed",
            "F1 OOF @0.5",
            "PR-AUC OOF",
            "Seuil optimal",
            "F1 OOF optimisé",
            "Durée",
        ],
        [
            [
                int(row["training_seed"]),
                f"{row['oof_f1_threshold_0_5']:.4f}",
                f"{row['oof_pr_auc']:.4f}",
                f"{row['optimized_threshold']:.3f}",
                f"{row['oof_f1_optimized']:.4f}",
                format_duration(row["duration_seconds"]),
            ]
            for row in rows
        ],
    )

    print("\nMoyenne ± écart-type entre les seeds")
    print_table(
        ["Métrique", "Moyenne", "Écart-type", "Min", "Max"],
        [
            [
                "F1 OOF @0.5",
                f"{aggregate_metrics['oof_f1_threshold_0_5']['mean']:.4f}",
                f"{aggregate_metrics['oof_f1_threshold_0_5']['std']:.4f}",
                f"{aggregate_metrics['oof_f1_threshold_0_5']['min']:.4f}",
                f"{aggregate_metrics['oof_f1_threshold_0_5']['max']:.4f}",
            ],
            [
                "PR-AUC OOF",
                f"{aggregate_metrics['oof_pr_auc']['mean']:.4f}",
                f"{aggregate_metrics['oof_pr_auc']['std']:.4f}",
                f"{aggregate_metrics['oof_pr_auc']['min']:.4f}",
                f"{aggregate_metrics['oof_pr_auc']['max']:.4f}",
            ],
            [
                "Seuil optimal",
                f"{aggregate_metrics['optimized_threshold']['mean']:.3f}",
                f"{aggregate_metrics['optimized_threshold']['std']:.3f}",
                f"{aggregate_metrics['optimized_threshold']['min']:.3f}",
                f"{aggregate_metrics['optimized_threshold']['max']:.3f}",
            ],
            [
                "F1 OOF optimisé",
                f"{aggregate_metrics['oof_f1_optimized']['mean']:.4f}",
                f"{aggregate_metrics['oof_f1_optimized']['std']:.4f}",
                f"{aggregate_metrics['oof_f1_optimized']['min']:.4f}",
                f"{aggregate_metrics['oof_f1_optimized']['max']:.4f}",
            ],
        ],
    )

    print("\nRésultat principal à reporter pour la validation :")
    print(
        "  F1 positif OOF @0.5 = "
        f"{aggregate_metrics['oof_f1_threshold_0_5']['mean']:.4f} "
        f"± {aggregate_metrics['oof_f1_threshold_0_5']['std']:.4f}"
    )
    print(
        "  PR-AUC OOF          = "
        f"{aggregate_metrics['oof_pr_auc']['mean']:.4f} "
        f"± {aggregate_metrics['oof_pr_auc']['std']:.4f}"
    )
    print(
        "  F1 OOF optimisé     = "
        f"{aggregate_metrics['oof_f1_optimized']['mean']:.4f} "
        f"± {aggregate_metrics['oof_f1_optimized']['std']:.4f}"
    )
    print(f"\nDurée totale : {format_duration(total_duration)}")
    print(f"CSV multi-seeds : {results_csv}")
    print(f"Résumé JSON     : {summary_json}")

    return benchmark_summary


# ---------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne le modèle Transformer avec une CV partagée à cinq seeds."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["cv"],
        default="cv",
        help="Conservé pour compatibilité. Le protocole benchmark utilise la CV.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=K_FOLDS,
        help=f"Nombre de folds. Valeur par défaut : {K_FOLDS}.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in TRAINING_SEEDS),
        help=(
            "Seeds d'entraînement séparées par des virgules. Défaut : "
            + ",".join(str(seed) for seed in TRAINING_SEEDS)
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Supprime les anciens résultats de ce modèle avant l'entraînement. "
            "Les manifests partagés de folds sont conservés."
        ),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    total_start = time.perf_counter()
    seeds = parse_seed_list(args.seeds)

    if args.folds < 2:
        raise ValueError("Utilisez au moins 2 folds.")

    for path in (TRAIN_CV_CSV, TEST_CSV):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    train_dataframe = pd.read_csv(TRAIN_CV_CSV)
    missing_columns = [
        column
        for column in (TEXT_COL, LABEL_COL, GROUP_COL)
        if column not in train_dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans train_cv.csv : "
            + ", ".join(missing_columns)
        )

    labels = train_dataframe[LABEL_COL].to_numpy(dtype=int)
    groups = train_dataframe[GROUP_COL].fillna("").astype(str).to_numpy()
    fingerprint = dataset_fingerprint(
        dataframe=train_dataframe,
        text_column=TEXT_COL,
        label_column=LABEL_COL,
        group_column=GROUP_COL,
    )

    prepare_model_root(overwrite=args.overwrite)

    run_configuration = {
        "protocol": "fixed_shared_stratified_group_kfold_multi_training_seeds",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_output_name": MODEL_OUTPUT_NAME,
        "max_length": int(MAX_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "epochs_max": int(EPOCHS),
        "learning_rate": float(LEARNING_RATE),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "early_stopping_threshold": float(EARLY_STOPPING_THRESHOLD),
        "training_seeds": [int(seed) for seed in seeds],
        "split_seed": int(CV_SPLIT_SEED),
        "fold_count": int(args.folds),
        "error_bar_scope": "inter_seed_variability_at_fixed_split_only",
    }
    with open(
        os.path.join(MODEL_ROOT, "run_configuration.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_configuration, file, indent=2, ensure_ascii=False)

    print_banner("BENCHMARK DEREEV — PROTOCOLE A STRICT")
    print_key_values(
        [
            ("Modèle affiché", MODEL_DISPLAY_NAME),
            ("Modèle Hugging Face", MODEL_NAME),
            ("Dossier modèle", MODEL_ROOT),
            ("Jeu train_cv", f"{TRAIN_CV_CSV} ({len(train_dataframe)} notes)"),
            ("Jeu test fixe", TEST_CSV),
            ("Seeds d'entraînement", ", ".join(map(str, seeds))),
            ("Seed du split fixe", CV_SPLIT_SEED),
            ("Nombre de groupes train", len(set(groups))),
            ("Nombre de folds", args.folds),
            ("Fine-tunings prévus", len(seeds) * args.folds),
            ("Métriques principales", "F1 positif et PR-AUC"),
            ("Matériel", device_description()),
            ("Fingerprint dataset", fingerprint),
        ]
    )
    print(
        "\nGARANTIE PROTOCOLE A : une seule affectation des groupes documentaires aux folds "
        "est créée avec CV_SPLIT_SEED. Elle est réutilisée sans aucune "
        "modification pour toutes les seeds d'entraînement et tous les modèles."
    )

    shared_split = ensure_shared_split(
        labels=labels,
        groups=groups,
        split_seed=CV_SPLIT_SEED,
        fold_count=args.folds,
        split_directory=CV_SPLITS_DIR,
        fingerprint=fingerprint,
    )

    print("\nManifeste de folds fixe et officiel")
    print_table(
        ["Seed du split", "Statut", "Folds", "Fichier partagé"],
        [[
            shared_split.split_seed,
            "créé" if shared_split.was_created else "rechargé",
            shared_split.fold_count,
            shared_split.csv_path,
        ]],
    )
    print(
        "CONFIRMATION : les groupes et indices train/validation ci-dessus seront "
        f"strictement identiques pour les {len(seeds)} seeds d'entraînement."
    )

    train_cv_ds, _ = get_tokenized_datasets(TRAIN_CV_CSV, TEST_CSV)
    tokenized_labels = np.asarray(train_cv_ds["labels"], dtype=int)

    if len(train_cv_ds) != len(train_dataframe):
        raise RuntimeError(
            "Le dataset tokenisé n'a pas la même taille que train_cv.csv."
        )
    if not np.array_equal(tokenized_labels, labels):
        raise RuntimeError(
            "L'ordre des labels du dataset tokenisé diffère de train_cv.csv."
        )

    summaries: List[Dict] = []
    total_trainings = len(seeds) * args.folds

    for seed_position, training_seed in enumerate(seeds):
        summaries.append(
            run_one_training_seed(
                train_cv_ds=train_cv_ds,
                labels=labels,
                shared_split=shared_split,
                training_seed=training_seed,
                completed_before=seed_position * args.folds,
                total_trainings=total_trainings,
            )
        )

    total_duration = time.perf_counter() - total_start
    benchmark_summary = aggregate_seed_summaries(
        summaries=summaries,
        total_duration=total_duration,
    )

    latest_summary = {
        "task": "multi_seed_cross_validation",
        "protocol": "fixed_shared_stratified_group_kfold_multi_training_seeds",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_output_name": MODEL_OUTPUT_NAME,
        "model_directory": MODEL_ROOT,
        "training_seeds": seeds,
        "split_seed": CV_SPLIT_SEED,
        "split_manifest_csv": shared_split.csv_path,
        "fold_count": args.folds,
        "benchmark_summary": os.path.join(MODEL_ROOT, "benchmark_summary.json"),
        "validation_results": os.path.join(
            MODEL_ROOT,
            "all_seeds_validation_results.csv",
        ),
        "aggregate_validation_metrics": benchmark_summary[
            "aggregate_validation_metrics"
        ],
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LATEST_EXPERIMENT_PATH, "w", encoding="utf-8") as file:
        json.dump(latest_summary, file, indent=2, ensure_ascii=False)

    print(f"Dernière expérience enregistrée : {LATEST_EXPERIMENT_PATH}")
    print("Étape suivante : python3 -m src.evaluate")


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())