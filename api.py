"""API DEREEV — DoctoBERT seed 42 issu de la validation croisée.

Charge exactement :
    models/production/doctobert_fr_seed42/
        tokenizer/
        fold_1/
        fold_2/
        fold_3/
        fold_4/
        fold_5/

La probabilité finale est la moyenne des probabilités de classe 1
des cinq folds. Le seuil est fixé à 0,5.
"""

from __future__ import annotations

import inspect
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing import clean_text, load_abreviations


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (
    BASE_DIR
    / "models"
    / "production"
    / "doctobert_fr_seed42"
)

FOLD_NAMES = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
TRAINING_SEED = 42
FIXED_THRESHOLD = 0.5
DEFAULT_MAX_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE_LOCK = threading.Lock()


class NoteMedicale(BaseModel):
    texte: str = Field(..., min_length=1)
    afficher_details: bool = False


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return value if isinstance(value, dict) else {}


def resolve_max_length() -> int:
    for filename in ("deployment_manifest.json", "experiment_summary.json"):
        data = load_json(MODEL_DIR / filename)

        for value in (
            data.get("max_length"),
            data.get("hyperparameters", {}).get("max_length")
            if isinstance(data.get("hyperparameters"), dict)
            else None,
        ):
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

    return DEFAULT_MAX_LENGTH


def has_weights(folder: Path) -> bool:
    candidates = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any((folder / filename).exists() for filename in candidates)


def validate_bundle() -> list[Path]:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Bundle introuvable : {MODEL_DIR.resolve()}"
        )

    tokenizer_dir = MODEL_DIR / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {tokenizer_dir.resolve()}"
        )

    fold_paths: list[Path] = []

    for fold_name in FOLD_NAMES:
        fold_path = MODEL_DIR / fold_name

        if not fold_path.is_dir():
            raise FileNotFoundError(
                f"Dossier absent : {fold_path.resolve()}"
            )

        if not (fold_path / "config.json").exists():
            raise FileNotFoundError(
                f"config.json absent : {fold_path.resolve()}"
            )

        if not has_weights(fold_path):
            raise FileNotFoundError(
                f"Poids absents : {fold_path.resolve()}"
            )

        fold_paths.append(fold_path)

    return fold_paths


def apply_project_cleaning(
    texte: str,
    abbreviations: dict[str, Any],
) -> str:
    parameter_count = len(inspect.signature(clean_text).parameters)

    if parameter_count >= 2:
        return str(clean_text(texte, abbreviations))

    return str(clean_text(texte))


def load_bundle():
    fold_paths = validate_bundle()
    max_length = resolve_max_length()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR / "tokenizer",
        local_files_only=True,
    )

    models = []

    for index, fold_path in enumerate(fold_paths, start=1):
        print(
            f"[CHARGEMENT] DoctoBERT seed 42 "
            f"— fold {index}/5 : {fold_path}"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            fold_path,
            local_files_only=True,
        )
        model.to(DEVICE)
        model.eval()
        models.append(model)

    print("\nEnsemble DoctoBERT CV chargé")
    print(f"- bundle      : {MODEL_DIR.resolve()}")
    print(f"- seed        : {TRAINING_SEED}")
    print(f"- folds       : {len(models)}")
    print(f"- seuil       : {FIXED_THRESHOLD}")
    print(f"- max_length  : {max_length}")
    print(f"- appareil    : {DEVICE}")

    return tokenizer, models, max_length


@asynccontextmanager
async def lifespan(app: FastAPI):
    (
        app.state.tokenizer,
        app.state.models,
        app.state.max_length,
    ) = load_bundle()

    try:
        app.state.abbreviations = load_abreviations()
    except Exception as error:
        app.state.abbreviations = {}
        print(
            "[AVERTISSEMENT] Dictionnaire d'abréviations "
            f"non chargé : {error}"
        )

    app.state.ready = True

    yield

    app.state.models.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="API DEREEV — DoctoBERT CV seed 42",
    version="1.0.0",
    description=(
        "Moyenne des probabilités des cinq folds issus de la CV "
        "DoctoBERT avec la seed 42."
    ),
    lifespan=lifespan,
)


def run_prediction(
    texte: str,
    afficher_details: bool,
) -> dict[str, Any]:
    if not getattr(app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="Les modèles ne sont pas chargés.",
        )

    texte_nettoye = apply_project_cleaning(
        texte,
        app.state.abbreviations,
    )

    if not texte_nettoye.strip():
        raise HTTPException(
            status_code=422,
            detail="La note est vide après nettoyage.",
        )

    inputs = app.state.tokenizer(
        texte_nettoye,
        return_tensors="pt",
        truncation=True,
        max_length=app.state.max_length,
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    probabilities: list[float] = []

    with INFERENCE_LOCK:
        with torch.inference_mode():
            for model in app.state.models:
                outputs = model(**inputs)
                probability = torch.softmax(
                    outputs.logits.float(),
                    dim=-1,
                )[0, 1].item()
                probabilities.append(float(probability))

    mean_probability = sum(probabilities) / len(probabilities)
    prediction = int(mean_probability >= FIXED_THRESHOLD)

    positive_votes = sum(
        probability >= FIXED_THRESHOLD
        for probability in probabilities
    )

    response: dict[str, Any] = {
        "prediction": prediction,
        "probabilite_classe_1": round(mean_probability, 6),
        "seuil_utilise": FIXED_THRESHOLD,
        "alerte": (
            "Réévaluation détectée"
            if prediction == 1
            else "Pas de réévaluation détectée"
        ),
        "modele": "DoctoBERT-fr — ensemble CV seed 42, folds 1 à 5",
        "seed_deploiement": TRAINING_SEED,
        "nombre_modeles": len(probabilities),
        "votes_classe_1": f"{positive_votes}/{len(probabilities)}",
    }

    if afficher_details:
        response["texte_nettoye"] = texte_nettoye
        response["probabilites_par_fold"] = [
            round(value, 6)
            for value in probabilities
        ]

    return response


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "API DEREEV opérationnelle",
        "interface": "http://127.0.0.1:8000/form",
        "documentation": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
        "bundle": str(MODEL_DIR),
        "seed": TRAINING_SEED,
        "folds": FOLD_NAMES,
        "threshold": FIXED_THRESHOLD,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "bundle": str(MODEL_DIR.resolve()),
        "model": "DoctoBERT-fr",
        "strategy": "mean_probability_across_5_cv_folds",
        "seed": TRAINING_SEED,
        "fold_count": len(app.state.models),
        "threshold": FIXED_THRESHOLD,
        "max_length": app.state.max_length,
        "device": str(DEVICE),
    }


@app.post("/predict")
def predict_json(note: NoteMedicale) -> dict[str, Any]:
    return run_prediction(
        note.texte,
        note.afficher_details,
    )


@app.post("/predict-text")
def predict_text(
    texte: str = Body(..., media_type="text/plain"),
    afficher_details: bool = Query(default=False),
) -> dict[str, Any]:
    return run_prediction(
        texte,
        afficher_details,
    )


@app.get("/form", response_class=HTMLResponse)
def form() -> HTMLResponse:
    return HTMLResponse(
        content=r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>API DEREEV — DoctoBERT seed 42</title>
<style>
body {
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 30px auto;
    padding: 0 20px;
    background: #f4f6f8;
}
h1 {
    color: #17365d;
}
.notice {
    background: #e8f4fd;
    border: 1px solid #b6d7f0;
    padding: 12px;
    margin-bottom: 15px;
}
textarea {
    width: 100%;
    min-height: 390px;
    padding: 12px;
    box-sizing: border-box;
    font-family: Consolas, monospace;
    font-size: 14px;
}
button {
    margin-top: 14px;
    padding: 12px 22px;
    font-size: 16px;
    cursor: pointer;
}
pre {
    margin-top: 20px;
    background: #1e1e1e;
    color: white;
    padding: 18px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    border-radius: 5px;
}
</style>
</head>
<body>

<h1>API DEREEV — DoctoBERT CV seed 42</h1>

<div class="notice">
Modèles chargés : <code>fold_1</code> à <code>fold_5</code>
du dossier <code>doctobert_fr_seed42</code>.
La probabilité finale est la moyenne des cinq folds.
</div>

<textarea
    id="note"
    placeholder="Collez directement une note médicale..."
></textarea>

<br>

<label>
<input type="checkbox" id="details" checked>
Afficher le texte nettoyé et les probabilités par fold
</label>

<br>

<button onclick="predict()">
Lancer la prédiction
</button>

<pre id="result">Le résultat apparaîtra ici.</pre>

<script>
async function predict() {
    const note =
        document.getElementById("note").value;

    const details =
        document.getElementById("details").checked;

    const result =
        document.getElementById("result");

    if (!note.trim()) {
        result.textContent = "La note est vide.";
        return;
    }

    result.textContent = "Prédiction en cours...";

    try {
        const response = await fetch(
            "/predict-text?afficher_details=" + details,
            {
                method: "POST",
                headers: {
                    "Content-Type":
                    "text/plain; charset=utf-8"
                },
                body: note
            }
        );

        const data = await response.json();

        result.textContent = JSON.stringify(
            data,
            null,
            2
        );
    } catch (error) {
        result.textContent =
            "Erreur : " + error;
    }
}
</script>

</body>
</html>
"""
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
