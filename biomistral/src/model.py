"""Chargement audité de BioMistral-7B pour classification binaire avec LoRA."""

from __future__ import annotations

import inspect
import os
from typing import Dict, List

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from biomistral.src.config import (
    ATTN_IMPLEMENTATION,
    GRADIENT_CHECKPOINTING,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_MODULES_TO_SAVE,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    POOLING_STRATEGY,
    USE_BF16,
)

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def validate_cuda_bf16() -> None:
    """Arrête tôt avec un message clair si le matériel est incompatible."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "BioMistral nécessite un GPU NVIDIA compatible CUDA."
        )
    if USE_BF16:
        bf16_check = getattr(torch.cuda, "is_bf16_supported", None)
        if bf16_check is None or not bf16_check():
            raise RuntimeError(
                "USE_BF16=True, mais le GPU ne prend pas en charge BF16."
            )


def _load_kwargs() -> Dict:
    kwargs: Dict = {
        "num_labels": 2,
        "low_cpu_mem_usage": True,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    if torch.cuda.is_available() and USE_BF16:
        kwargs["dtype"] = torch.bfloat16
    return kwargs


def _load_base_classifier():
    validate_cuda_bf16()
    kwargs = _load_kwargs()
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **kwargs)

    if model.config.eos_token_id is None:
        raise RuntimeError("BioMistral ne fournit pas de eos_token_id.")
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.config.problem_type = "single_label_classification"
    # Attribut documentaire, sauvegardé dans la config du modèle.
    model.config.dereev_pooling_strategy = POOLING_STRATEGY

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def _matching_module_names(model, target: str) -> List[str]:
    return [
        name for name, _ in model.named_modules()
        if name == target or name.endswith(f".{target}")
    ]


def audit_base_model(model) -> Dict:
    """Vérifie pooling, padding et présence des modules LoRA cibles."""
    target_counts = {
        target: len(_matching_module_names(model, target))
        for target in LORA_TARGET_MODULES
    }
    missing_targets = [target for target, count in target_counts.items() if count == 0]
    if missing_targets:
        raise RuntimeError(
            "Modules LoRA cibles introuvables dans BioMistral : "
            + ", ".join(missing_targets)
        )

    pooling_detection = "unknown"
    source_excerpt_available = False
    try:
        source = inspect.getsource(model.__class__.forward).lower()
        source_excerpt_available = True
        if any(token in source for token in (
            "last_non_pad_token",
            "sequence_lengths",
            "last non-pad",
            "last_non_padding",
        )):
            pooling_detection = "last_non_padding_token"
        elif "mean" in source and "attention_mask" in source:
            pooling_detection = "possible_mean_pooling"
    except (OSError, TypeError):
        pass

    if pooling_detection not in {"unknown", POOLING_STRATEGY}:
        raise RuntimeError(
            f"Pooling détecté={pooling_detection}, attendu={POOLING_STRATEGY}."
        )

    return {
        "base_model_class": model.__class__.__name__,
        "configured_pooling_strategy": POOLING_STRATEGY,
        "detected_pooling_strategy": pooling_detection,
        "forward_source_inspectable": source_excerpt_available,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
        "lora_target_module_match_counts": target_counts,
        "lora_configuration": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "modules_to_save": LORA_MODULES_TO_SAVE,
        },
        "note": (
            "Le classifieur Mistral est configuré pour utiliser le dernier token "
            "non paddé. Si l'introspection de la version Transformers ne permet "
            "pas de le confirmer, l'audit le signale explicitement comme unknown."
        ),
    }


def get_model():
    """Crée un BioMistral-LoRA indépendant et retourne aussi son audit."""
    base_model = _load_base_classifier()
    audit = audit_base_model(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        modules_to_save=LORA_MODULES_TO_SAVE,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    audit["trainable_parameters"] = int(trainable)
    audit["total_parameters"] = int(total)
    audit["trainable_fraction"] = float(trainable / total)
    return model, audit


def load_model_for_inference(adapter_path: str):
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(adapter_path)
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(
            f"adapter_config.json absent dans {adapter_path}."
        )
    base_model = _load_base_classifier()
    audit_base_model(base_model)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.eval()
    return model
