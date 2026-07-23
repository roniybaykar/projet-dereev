"""Fonctions permettant de convertir les logits et de tester des seuils."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def logits_to_positive_probabilities(logits):
    """
    Convertit les logits du modèle en probabilités de la classe 1.

    Forme attendue :
        (nombre_exemples, 2)
    """
    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "Forme de logits attendue : (nombre_exemples, 2). "
            f"Forme reçue : {logits.shape}"
        )

    # Softmax stable numériquement.
    shifted_logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(
        shifted_logits
    )

    probabilities = (
        exponentials
        / exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )

    return probabilities[:, 1]


def compute_threshold_metrics(
    labels,
    probabilities,
    threshold,
):
    """
    Calcule les métriques lorsque la classe 1 est prédite
    si sa probabilité est supérieure ou égale au seuil.
    """
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if labels.shape != probabilities.shape:
        raise ValueError(
            "Les labels et les probabilités doivent avoir "
            "la même forme."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Le seuil doit être compris entre 0 et 1."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "true_positive_count": int(
            labels.sum()
        ),
        "predicted_positive_count": int(
            predictions.sum()
        ),
    }


def evaluate_thresholds(
    labels,
    probabilities,
    minimum,
    maximum,
    step,
):
    """
    Teste plusieurs seuils et retourne :
    - les résultats pour tous les seuils ;
    - le seuil qui maximise le F1.
    """
    if step <= 0:
        raise ValueError(
            "Le pas doit être strictement positif."
        )

    if minimum > maximum:
        raise ValueError(
            "Le seuil minimum doit être inférieur "
            "au seuil maximum."
        )

    thresholds = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    thresholds = np.round(
        thresholds,
        decimals=10,
    )

    results = [
        compute_threshold_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]

    # En cas d'égalité de F1, le seuil le plus proche
    # de 0.5 est privilégié.
    best_result = max(
        results,
        key=lambda result: (
            result["f1"],
            -abs(
                result["threshold"] - 0.5
            ),
        ),
    )

    return results, best_result


def plot_threshold_curve(results, output_path):
    """Trace le F1 en fonction des seuils testés et sauvegarde le graphique."""
    thresholds = [result["threshold"] for result in results]
    f1_scores = [result["f1"] for result in results]

    if len(thresholds) == 0:
        raise ValueError(
            "Aucun résultat de seuil à tracer."
        )

    best_result = max(
        results,
        key=lambda result: (
            result["f1"],
            -abs(result["threshold"] - 0.5),
        ),
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )
    ax.plot(
        thresholds,
        f1_scores,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="F1",
    )
    ax.axvline(
        best_result["threshold"],
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label="Meilleur seuil",
    )
    ax.scatter(
        [best_result["threshold"]],
        [best_result["f1"]],
        color="tab:red",
        zorder=5,
    )

    ax.set_xlabel("Seuil")
    ax.set_ylabel("F1")
    ax.set_title("Évolution du F1 en fonction du seuil")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    min_f1 = min(f1_scores)
    max_f1 = max(f1_scores)
    padding = max(0.02, 0.05 * (max_f1 - min_f1))
    ax.set_ylim(min_f1 - padding, max_f1 + padding)
    ax.set_xlim(min(thresholds), max(thresholds))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path