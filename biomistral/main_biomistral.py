"""Benchmark DEREEV : BioMistral-7B adapté avec LoRA, 5 seeds x 5 folds."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Sequence, Tuple

_FIXED_HASH_SEED = "0"
if os.environ.get("PYTHONHASHSEED") != _FIXED_HASH_SEED:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = _FIXED_HASH_SEED
    command = [
        sys.executable,
        "-m",
        "biomistral.main_biomistral",
        *sys.argv[1:],
    ]
    if os.name == "nt":
        raise SystemExit(subprocess.run(command, env=environment).returncode)
    os.execvpe(sys.executable, command, environment)

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from biomistral.src.config import (
    BATCH_SIZE,
    BENCHMARK_MODELS_DIR,
    CV_SPLITS_DIR,
    CV_SPLIT_SEED,
    BIOMISTRAL_PILOT_SEEDS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    EVAL_BATCH_SIZE,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    K_FOLDS,
    GROUP_COL,
    LABEL_COL,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_LENGTH,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    MODEL_OUTPUT_NAME,
    MODELS_DIR,
    TEST_CSV,
    TEXT_COL,
    TRAINING_SEEDS,
    TRAIN_CV_CSV,
    POOLING_STRATEGY,
    PADDING_SIDE,
    USE_BF16,
)
from src.cv_protocol import SharedSplit, dataset_fingerprint, ensure_shared_split, parse_seed_list
from biomistral.src.data_loader import get_tokenized_datasets, get_tokenizer
from biomistral.src.model import get_model
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


def print_banner(title: str, width: int = 84, character: str = "=") -> None:
    print("\n" + character * width)
    print(f" {title}")
    print(character * width)


def print_key_values(rows: Sequence[Tuple[str, object]]) -> None:
    width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        print(f"{str(key):<{width}} : {value}")


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not rows:
        print("(aucune ligne)")
        return
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print("| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |")
    print(separator)
    for row in rendered:
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
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
    array = np.asarray(values, dtype=float)
    return 0.0 if len(array) < 2 else float(np.std(array, ddof=1))


def print_metrics_block(title: str, metrics: Dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print_key_values([
        ("Seuil", f"{metrics['threshold']:.3f}"),
        ("F1 positif", f"{metrics['f1']:.4f}"),
        ("PR-AUC", f"{metrics.get('pr_auc', float('nan')):.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("Précision", f"{metrics['precision']:.4f}"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
    ])
    print("\nMatrice de confusion")
    print_table(
        ["Réel / Prédit", "Classe 0", "Classe 1"],
        [
            ["Classe 0", metrics["true_negative"], metrics["false_positive"]],
            ["Classe 1", metrics["false_negative"], metrics["true_positive"]],
        ],
    )


def configure_determinism(seed: int) -> None:
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
    return f"CUDA — {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"


def build_training_arguments(output_dir: str, run_name: str, seed: int) -> TrainingArguments:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    arguments = {
        "output_dir": output_dir,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "save_safetensors": True,
        "seed": seed,
        "data_seed": seed,
        "full_determinism": True,
        "dataloader_num_workers": 0,
        "logging_dir": os.path.join("logs", run_name),
        "logging_steps": 10,
        "report_to": "none",
        "bf16": bool(USE_BF16 and torch.cuda.is_available()),
        "fp16": False,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "remove_unused_columns": True,
    }
    if "eval_strategy" in parameters:
        arguments["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        arguments["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError("Transformers ne propose ni eval_strategy ni evaluation_strategy.")
    unsupported = [name for name in arguments if name not in parameters]
    for name in unsupported:
        arguments.pop(name)
    if unsupported:
        print("Arguments ignorés : " + ", ".join(unsupported))
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


def compute_class_weights(dataset) -> Tuple[torch.Tensor, np.ndarray]:
    labels = np.asarray(dataset["labels"], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise RuntimeError("Une classe est absente du fold d'entraînement.")
    weights = counts.sum() / (2 * counts)
    return torch.tensor(weights, dtype=torch.float32), counts


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(device=logits.device, dtype=torch.float32)
        )(logits.float(), labels)
        return (loss, outputs) if return_outputs else loss


def prepare_model_root(overwrite: bool) -> None:
    if os.path.exists(MODEL_ROOT):
        if not overwrite:
            raise FileExistsError(
                f"Le dossier existe déjà : {MODEL_ROOT}\nUtilisez --overwrite."
            )
        shutil.rmtree(MODEL_ROOT)
    os.makedirs(MODEL_ROOT, exist_ok=True)


def remove_checkpoint_subdirectories(fold_dir: str) -> None:
    for name in os.listdir(fold_dir):
        path = os.path.join(fold_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(path):
            shutil.rmtree(path)


def seed_paths(seed_dir: str) -> Dict[str, str]:
    return {
        "threshold_json": os.path.join(seed_dir, "decision_threshold.json"),
        "threshold_csv": os.path.join(seed_dir, "threshold_results.csv"),
        "threshold_plot": os.path.join(seed_dir, "threshold_results.png"),
        "pr_curve_plot": os.path.join(seed_dir, "precision_recall_curve.png"),
        "predictions_csv": os.path.join(seed_dir, "validation_predictions.csv"),
        "summary_json": os.path.join(seed_dir, "experiment_summary.json"),
    }


def save_threshold_outputs(labels, probabilities, fold_ids, training_seed, paths):
    threshold_results, best_result = evaluate_thresholds(
        labels=labels,
        probabilities=probabilities,
        minimum=THRESHOLD_MIN,
        maximum=THRESHOLD_MAX,
        step=THRESHOLD_STEP,
    )
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
    pd.DataFrame({
        "row_index": np.arange(len(labels), dtype=int),
        "training_seed": training_seed,
        "fold": fold_ids,
        "label": labels,
        "probability_class_1": probabilities,
        "prediction_threshold_0_5": (probabilities >= 0.5).astype(int),
        "prediction_optimized_threshold": (probabilities >= best_result["threshold"]).astype(int),
    }).to_csv(paths["predictions_csv"], index=False)
    threshold_summary = {
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "training_seed": int(training_seed),
        "split_seed": int(CV_SPLIT_SEED),
        "selection_data": "out_of_fold_validation",
        "objective": "maximum_f1_positive_class",
        "pr_auc": compute_pr_auc(
            labels=labels,
            probabilities=probabilities,
        ),
        "best_threshold": best_result["threshold"],
        "best_metrics": best_result,
        "default_threshold": 0.5,
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


def run_one_training_seed(
    train_cv_ds,
    labels: np.ndarray,
    shared_split: SharedSplit,
    training_seed: int,
    completed_before: int,
    total_trainings: int,
) -> Dict:
    seed_dir = os.path.join(MODEL_ROOT, f"seed_{training_seed}")
    os.makedirs(seed_dir, exist_ok=True)
    paths = seed_paths(seed_dir)
    seed_start = time.perf_counter()

    print_banner(f"SEED {training_seed} — {MODEL_DISPLAY_NAME}", character="#")
    oof_probabilities = np.full(len(labels), np.nan, dtype=float)
    oof_folds = np.full(len(labels), -1, dtype=int)
    fold_f1_scores: List[float] = []
    fold_pr_auc_scores: List[float] = []
    fold_summaries: List[Dict] = []

    for fold_index, (train_idx, val_idx) in enumerate(shared_split.indices(), start=1):
        configure_determinism(training_seed)
        fold_start = time.perf_counter()
        train_ds = train_cv_ds.select(train_idx)
        val_ds = train_cv_ds.select(val_idx)
        class_weights, class_counts = compute_class_weights(train_ds)
        val_counts = np.bincount(labels[val_idx], minlength=2)
        fold_dir = os.path.join(seed_dir, f"fold_{fold_index}")
        os.makedirs(fold_dir, exist_ok=True)
        global_index = completed_before + fold_index

        print_banner(
            f"ENTRAÎNEMENT {global_index}/{total_trainings} — SEED {training_seed} — FOLD {fold_index}/{shared_split.fold_count}",
            character="-",
        )
        print_key_values([
            ("Modèle", f"{MODEL_DISPLAY_NAME} ({MODEL_NAME})"),
            ("Train", f"{len(train_idx)} notes — 0:{class_counts[0]} | 1:{class_counts[1]}"),
            ("Validation", f"{len(val_idx)} notes — 0:{val_counts[0]} | 1:{val_counts[1]}"),
            ("Poids loss", f"0:{class_weights[0].item():.4f} | 1:{class_weights[1].item():.4f}"),
            ("Batch physique", BATCH_SIZE),
            ("Accumulation", GRADIENT_ACCUMULATION_STEPS),
            ("Batch effectif", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
            ("Sortie", fold_dir),
        ])

        training_args = build_training_arguments(
            fold_dir,
            f"{MODEL_OUTPUT_NAME}_seed_{training_seed}_fold_{fold_index}",
            training_seed,
        )

        # La seed est fixée avant cette création : pas de double chargement 7B.
        model, model_audit = get_model()
        with open(
            os.path.join(fold_dir, "model_audit.json"),
            "w",
            encoding="utf-8",
        ) as audit_file:
            json.dump(model_audit, audit_file, indent=2, ensure_ascii=False)
        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )],
        )

        trainer.train()
        output = trainer.predict(val_ds)
        probabilities = logits_to_positive_probabilities(output.predictions)
        validation_labels = np.asarray(output.label_ids, dtype=int)
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
        duration = time.perf_counter() - fold_start
        print_metrics_block(f"VALIDATION — SEED {training_seed} — FOLD {fold_index}", metrics)
        print(f"Durée : {format_duration(duration)}")

        best_checkpoint = trainer.state.best_model_checkpoint
        trainer.save_model(fold_dir)  # PEFT sauvegarde l'adaptateur + la tête score.
        remove_checkpoint_subdirectories(fold_dir)

        fold_f1_scores.append(metrics["f1"])
        fold_pr_auc_scores.append(metrics["pr_auc"])
        fold_summaries.append({
            "fold": fold_index,
            "training_seed": training_seed,
            "train_size": int(len(train_idx)),
            "validation_size": int(len(val_idx)),
            "best_checkpoint_during_training": best_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
            "final_epoch": trainer.state.epoch,
            "duration_seconds": float(duration),
            "validation_metrics": metrics,
            "adapter_path": fold_dir,
            "model_audit": model_audit,
        })

        del trainer, model, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_probabilities).any() or np.any(oof_folds == -1):
        raise RuntimeError("Certaines lignes n'ont pas reçu de prédiction OOF.")

    default_result, optimized_result = save_threshold_outputs(
        labels, oof_probabilities, oof_folds, training_seed, paths
    )
    seed_duration = time.perf_counter() - seed_start

    print_banner(f"BILAN OOF — SEED {training_seed}")
    print_metrics_block("OOF — SEUIL 0.5", default_result)
    print_metrics_block("OOF — SEUIL OPTIMISÉ", optimized_result)

    summary = {
        "protocol": "fixed_shared_stratified_group_kfold_multi_training_seeds",
        "adaptation": "BioMistral-7B base gelé + LoRA + tête score",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_output_name": MODEL_OUTPUT_NAME,
        "training_seed": training_seed,
        "split_seed": shared_split.split_seed,
        "fold_count": shared_split.fold_count,
        "split_manifest_csv": shared_split.csv_path,
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
        "adapter_paths": [os.path.join(seed_dir, f"fold_{i}") for i in range(1, shared_split.fold_count + 1)],
        "training_configuration": {
            "max_length": int(MAX_LENGTH),
            "batch_size": int(BATCH_SIZE),
            "gradient_accumulation_steps": int(GRADIENT_ACCUMULATION_STEPS),
            "effective_batch_size": int(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
            "epochs_max": int(EPOCHS),
            "learning_rate": float(LEARNING_RATE),
            "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
            "early_stopping_threshold": float(EARLY_STOPPING_THRESHOLD),
            "pooling_strategy": POOLING_STRATEGY,
            "padding_side": PADDING_SIDE,
            "lora_r": int(LORA_R),
            "lora_alpha": int(LORA_ALPHA),
            "lora_dropout": float(LORA_DROPOUT),
            "lora_target_modules": list(LORA_TARGET_MODULES),
        },
        "uncertainty_scope": "inter_seed_variability_at_fixed_split_only",
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


def aggregate_seed_summaries(summaries: Sequence[Dict], total_duration: float) -> Dict:
    rows = []
    for summary in summaries:
        default = summary["oof_default_metrics"]
        optimized = summary["oof_optimized_metrics"]
        rows.append({
            "training_seed": summary["training_seed"],
            "oof_f1_threshold_0_5": default["f1"],
            "oof_pr_auc": default["pr_auc"],
            "oof_precision_threshold_0_5": default["precision"],
            "oof_recall_threshold_0_5": default["recall"],
            "optimized_threshold": optimized["threshold"],
            "oof_f1_optimized": optimized["f1"],
            "duration_seconds": summary["duration_seconds"],
        })
    dataframe = pd.DataFrame(rows)
    csv_path = os.path.join(MODEL_ROOT, "all_seeds_validation_results.csv")
    dataframe.to_csv(csv_path, index=False)

    columns = ["oof_f1_threshold_0_5", "oof_pr_auc", "optimized_threshold", "oof_f1_optimized"]
    aggregates = {}
    for column in columns:
        values = dataframe[column].to_numpy(float)
        aggregates[column] = {
            "mean": float(np.mean(values)),
            "std": sample_std(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    result = {
        "protocol": "fixed_shared_stratified_group_kfold_multi_training_seeds",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "training_seeds": [s["training_seed"] for s in summaries],
        "fold_count": summaries[0]["fold_count"],
        "error_bar_scope": "inter_seed_variability_at_fixed_split_only",
        "training_configuration": summaries[0].get("training_configuration", {}),
        "aggregate_validation_metrics": aggregates,
        "seed_summaries": summaries,
        "total_duration_seconds": float(total_duration),
    }
    json_path = os.path.join(MODEL_ROOT, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print_banner(f"RÉSULTAT VALIDATION MULTI-SEEDS — {MODEL_DISPLAY_NAME}")
    print_table(
        ["Métrique", "Moyenne", "Écart-type", "Min", "Max"],
        [[
            name,
            f"{aggregates[key]['mean']:.4f}",
            f"{aggregates[key]['std']:.4f}",
            f"{aggregates[key]['min']:.4f}",
            f"{aggregates[key]['max']:.4f}",
        ] for name, key in [
            ("F1 OOF @0.5", "oof_f1_threshold_0_5"),
            ("PR-AUC OOF", "oof_pr_auc"),
            ("Seuil optimal", "optimized_threshold"),
            ("F1 OOF optimisé", "oof_f1_optimized"),
        ]],
    )
    print(f"Durée totale : {format_duration(total_duration)}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BioMistral-7B-LoRA — protocole A strict.")
    parser.add_argument("--mode", choices=["cv"], default="cv")
    parser.add_argument("--folds", type=int, default=K_FOLDS)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, BIOMISTRAL_PILOT_SEEDS)),
        help=(
            "Pilote par défaut : une seule seed sur les cinq folds. "
            "Pour plusieurs seeds, ajouter --confirm-multi-seed."
        ),
    )
    parser.add_argument(
        "--confirm-multi-seed",
        action="store_true",
        help="Confirmation explicite avant de lancer plus d'une seed BioMistral.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    total_start = time.perf_counter()
    seeds = parse_seed_list(args.seeds)
    if len(seeds) > 1 and not args.confirm_multi_seed:
        raise ValueError(
            "BioMistral est en mode pilote : lancez d'abord une seule seed. "
            "Pour confirmer plusieurs seeds, ajoutez --confirm-multi-seed."
        )
    if args.folds < 2:
        raise ValueError("Utilisez au moins 2 folds.")
    for path in (TRAIN_CV_CSV, TEST_CSV):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    train_df = pd.read_csv(TRAIN_CV_CSV)
    missing = [c for c in (TEXT_COL, LABEL_COL, GROUP_COL) if c not in train_df.columns]
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(missing))
    labels = train_df[LABEL_COL].to_numpy(dtype=int)
    groups = train_df[GROUP_COL].fillna("").astype(str).to_numpy()
    fingerprint = dataset_fingerprint(
        train_df, TEXT_COL, LABEL_COL, GROUP_COL
    )

    prepare_model_root(args.overwrite)
    run_configuration = {
        "protocol": "fixed_shared_stratified_group_kfold_multi_training_seeds",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "training_seeds": seeds,
        "split_seed": CV_SPLIT_SEED,
        "fold_count": args.folds,
        "pilot_mode": len(seeds) == 1,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "pooling_strategy": POOLING_STRATEGY,
        "padding_side": PADDING_SIDE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
    }
    with open(os.path.join(MODEL_ROOT, "run_configuration.json"), "w", encoding="utf-8") as file:
        json.dump(run_configuration, file, indent=2, ensure_ascii=False)
    tokenizer_dir = os.path.join(MODEL_ROOT, "tokenizer")
    get_tokenizer().save_pretrained(tokenizer_dir)

    print_banner("BENCHMARK DEREEV — BIOMISTRAL-7B-LORA — PROTOCOLE A STRICT")
    print_key_values([
        ("Modèle", MODEL_DISPLAY_NAME),
        ("Checkpoint", MODEL_NAME),
        ("Dossier", MODEL_ROOT),
        ("Seeds", ", ".join(map(str, seeds))),
        ("Seed split fixe", CV_SPLIT_SEED),
        ("Folds", args.folds),
        ("Adaptateurs prévus", len(seeds) * args.folds),
        ("Batch effectif", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
        ("Learning rate LoRA", LEARNING_RATE),
        ("Pooling", POOLING_STRATEGY),
        ("Padding", PADDING_SIDE),
        ("LoRA", f"r={LORA_R}, alpha={LORA_ALPHA}, cibles={LORA_TARGET_MODULES}"),
        ("Mode", "pilote une seed" if len(seeds) == 1 else "multi-seeds confirmé"),
        ("Matériel", device_description()),
        ("Fingerprint", fingerprint),
    ])

    shared_split = ensure_shared_split(
        labels=labels,
        groups=groups,
        split_seed=CV_SPLIT_SEED,
        fold_count=args.folds,
        split_directory=CV_SPLITS_DIR,
        fingerprint=fingerprint,
    )
    print(f"Manifeste fixe : {shared_split.csv_path}")

    train_ds, _ = get_tokenized_datasets(TRAIN_CV_CSV, TEST_CSV)
    if not np.array_equal(np.asarray(train_ds["labels"], dtype=int), labels):
        raise RuntimeError("L'ordre des labels tokenisés diffère de train_cv.csv.")

    summaries = []
    total_trainings = len(seeds) * args.folds
    for position, seed in enumerate(seeds):
        summaries.append(run_one_training_seed(
            train_ds,
            labels,
            shared_split,
            seed,
            position * args.folds,
            total_trainings,
        ))

    total_duration = time.perf_counter() - total_start
    benchmark_summary = aggregate_seed_summaries(summaries, total_duration)

    latest = {
        "task": "multi_seed_cross_validation_biomistral_lora",
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_name": MODEL_NAME,
        "model_directory": MODEL_ROOT,
        "training_seeds": seeds,
        "split_seed": CV_SPLIT_SEED,
        "fold_count": args.folds,
        "benchmark_summary": os.path.join(MODEL_ROOT, "benchmark_summary.json"),
        "aggregate_validation_metrics": benchmark_summary["aggregate_validation_metrics"],
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LATEST_EXPERIMENT_PATH, "w", encoding="utf-8") as file:
        json.dump(latest, file, indent=2, ensure_ascii=False)
    print("Étape suivante : python -m biomistral.src.evaluate")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
