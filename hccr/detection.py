"""Point-adjusted decisions and threshold-sweep metrics for the test sequence."""
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from .tail_calibration import stream_cutoff


def adjusted_metrics(truth, alarms):
    """Credit every point of a labeled event if at least one point was detected."""
    transitions = np.diff(np.r_[0, truth.astype(np.int8), 0])
    adjusted = alarms.copy()
    for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
        if alarms[start:stop].any():
            adjusted[start:stop] = True

    hits = np.count_nonzero(adjusted & truth)
    predicted = np.count_nonzero(adjusted)
    positive = np.count_nonzero(truth)
    precision = hits / predicted if predicted else 0.0
    recall = hits / positive if positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def ranking_metrics(truth, scores):
    """Unadjusted PR and ROC areas, plus the best pointwise F1 on the PR curve."""
    values = (scores - scores.min()) / (np.ptp(scores) + 1e-12)
    precision, recall, _ = precision_recall_curve(truth, values)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(f1.argmax())
    return {
        "auc_roc": float(roc_auc_score(truth, values)) if np.unique(truth).size == 2 else None,
        "auc_pr": float(auc(recall, precision)),
        "f_max": float(f1[best]),
        "p_max": float(precision[best]),
        "r_max": float(recall[best]),
    }


def report_scores(init_scores, test_scores, labels, risk=0.01):
    """Summarize the model's endpoint scores using the original scalar report keys.

    precision, recall and f1_score use point adjustment at the mean SPOT
    threshold. p_max, r_max and f_max use the unadjusted label-based PR sweep.
    """
    scores = np.asarray(test_scores).reshape(-1)
    truth = np.asarray(labels).reshape(-1) > 0
    if scores.size != truth.size:
        raise ValueError("each test score needs exactly one label")
    threshold = stream_cutoff(init_scores, scores, risk)
    return {
        "threshold": threshold,
        **adjusted_metrics(truth, scores > threshold),
        **ranking_metrics(truth, scores),
    }
