from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.metrics import average_precision_score
except Exception:  # pragma: no cover
    average_precision_score = None


def compute_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")

    if average_precision_score is not None:
        return float(average_precision_score(labels, scores))

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    positives = (sorted_labels == 1).astype(np.float64)
    positive_count = positives.sum()
    if positive_count == 0:
        return float("nan")

    cumulative_true_positives = np.cumsum(positives)
    precision = cumulative_true_positives / (np.arange(len(sorted_labels)) + 1)
    return float(np.sum(precision * positives) / positive_count)


def compute_binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    if labels.size == 0:
        return {
            "acc": float("nan"),
            "ap": float("nan"),
            "correct": 0,
            "total": 0,
        }

    correct = int((predictions == labels).sum())
    total = int(labels.size)
    return {
        "acc": 100.0 * correct / total,
        "ap": 100.0 * compute_ap(labels, scores),
        "correct": correct,
        "total": total,
    }
