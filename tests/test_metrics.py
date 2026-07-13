import numpy as np

from pire_dirt.utils.metrics import compute_binary_metrics


def test_binary_metrics_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(labels, predictions, scores)

    assert metrics["acc"] == 100.0
    assert metrics["ap"] == 100.0
    assert metrics["correct"] == 4
    assert metrics["total"] == 4
