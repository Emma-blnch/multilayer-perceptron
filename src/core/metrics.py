"""Evaluation metrics, computed from the softmax output and the one-hot labels.

M (malignant) is the positive class: a true positive is a malignant tumor
predicted malignant, a false negative is a malignant tumor predicted benign.
"""
import numpy as np

from core.network import binary_cross_entropy

METRICS = ["loss", "accuracy", "precision", "recall", "f1"]


def confusion_matrix(probs, y):
    """Return (tp, fp, fn, tn) counts, M = positive class (column 1)."""
    pred = probs.argmax(axis=1) == 1  # prédit M ?
    true = y.argmax(axis=1) == 1  # vraiment M ?
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))  # fausse alerte : B prédit M
    fn = int(np.sum(~pred & true))  # cancer raté : M prédit B
    tn = int(np.sum(~pred & ~true))
    return tp, fp, fn, tn


def accuracy(probs, y):
    """Fraction of samples whose predicted class matches the label."""
    return float(np.mean(probs.argmax(axis=1) == y.argmax(axis=1)))


def precision(tp, fp):
    """Among the tumors predicted M, fraction really M."""
    return tp / (tp + fp) if tp + fp else 0.0


def recall(tp, fn):
    """Among the real M tumors, fraction the network found."""
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(p, r):
    """Harmonic mean of precision and recall: high only if both are high."""
    return 2 * p * r / (p + r) if p + r else 0.0


def compute_metrics(probs, y):
    """All the METRICS at once, as a dict."""
    tp, fp, fn, _ = confusion_matrix(probs, y)
    p, r = precision(tp, fp), recall(tp, fn)
    return {
        "loss": float(binary_cross_entropy(probs, y)),
        "accuracy": accuracy(probs, y),
        "precision": p,
        "recall": r,
        "f1": f1_score(p, r),
    }
