"""Evaluation metrics for the 3-class chest X-ray task.

All functions accept numpy arrays and return a plain dict so results can be
serialised directly to MLflow / JSON without extra conversion.

Metric choices (for the methods section):
    - Macro-AUROC: primary metric for imbalanced multi-class medical imaging;
      class-size-independent, directly comparable to CheXNet / FedAvg baselines.
    - Macro-F1: balances precision and recall across all classes.
    - Per-class sensitivity (recall): critical for the COVID-19 minority class;
      high global accuracy can mask a collapsed COVID detector.
    - Global accuracy: reported for completeness but not the primary signal.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Compute the full evaluation metric set used in every FL round.

    Args:
        y_true: Integer ground-truth labels, shape (N,).
        y_prob: Softmax probability scores, shape (N, 3).
        prefix: Optional string prepended to every key (e.g. ``"round_42/``).

    Returns:
        Flat dict of scalar metrics, ready for ``mlflow.log_metrics``.

    Raises:
        ValueError: If ``y_prob`` does not have 3 columns.
    """
    if y_prob.ndim != 2 or y_prob.shape[1] != 3:
        raise ValueError(f"y_prob must be (N, 3), got {y_prob.shape}")

    y_pred = y_prob.argmax(axis=1)
    n_classes = 3

    # ── Global ───────────────────────────────────────────────────────────────
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUROC: handle the edge case where a class is absent from the eval set
    # (can happen in small val sets of highly-skewed clients).
    try:
        macro_auroc = float(
            roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError:
        # Fall back to NaN rather than crashing; logged so the reader notices.
        macro_auroc = float("nan")

    # ── Per-class ────────────────────────────────────────────────────────────
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.where(
            cm.sum(axis=1) > 0, cm.diagonal() / cm.sum(axis=1), 0.0
        )

    # ── Build flat dict ───────────────────────────────────────────────────────
    p = prefix
    metrics: dict[str, float] = {
        f"{p}accuracy": accuracy,
        f"{p}macro_f1": macro_f1,
        f"{p}macro_auroc": macro_auroc,
    }
    for i, name in enumerate(CLASS_NAMES):
        tag = name.lower().replace("-", "_")
        metrics[f"{p}{tag}_f1"] = float(per_class_f1[i])
        metrics[f"{p}{tag}_sensitivity"] = float(sensitivity[i])

    return metrics


def format_report(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """Return a human-readable classification report string (for logging)."""
    y_pred = y_prob.argmax(axis=1)
    return classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
