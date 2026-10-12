"""Evaluation metrics, computed from the softmax output and the one-hot labels.

M (malignant) is the positive class: a true positive is a malignant tumor
predicted malignant, a false negative is a malignant tumor predicted benign.
"""
import numpy as np

from core.network import binary_cross_entropy

METRICS = ["loss", "accuracy", "precision", "recall", "f1"]


def confusion_matrix(probs, y):
    """Return (tp, fp, fn, tn) counts, M = positive class (column 1)."""
    pred = probs.argmax(axis=1) == 1  # si 1 = prédit M
    true = y.argmax(axis=1) == 1  # verifie sur le one-hot que vraiment M
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))  # fausse alerte : B prédit M (~ = non comme !true)
    fn = int(np.sum(~pred & true))  # cancer raté : vrai M mais prédit B
    tn = int(np.sum(~pred & ~true)) # vrai négatif
    return tp, fp, fn, tn


def accuracy(tp, fp, fn, tn):
    """Fraction of samples correctly predicted (M or B)."""
    total = tp + fp + fn + tn
    return (tp + tn) / total if total else 0.0  # proportion de bonnes reponses


def precision(tp, fp):
    """Among the tumors predicted M, fraction really M."""
    return tp / (tp + fp) if tp + fp else 0.0  # si precision basse = bcp de fausses alertes


def recall(tp, fn):
    """Among the real M tumors, fraction the network found."""
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(p, r):
    """Harmonic mean of precision and recall: high only if both are high."""
    return 2 * p * r / (p + r) if p + r else 0.0


def print_confusion_matrix(tp, fp, fn, tn):
    """Print the counts as a 2x2 table, rows = true class, columns = predicted."""
    print(f"  {'':<9} {'pred M':>7} {'pred B':>7}")
    print(f"  {'true M':<9} {tp:>7} {fn:>7}")
    print(f"  {'true B':<9} {fp:>7} {tn:>7}")


def compute_metrics(probs, y):
    """All the METRICS at once, as a dict."""
    tp, fp, fn, tn = confusion_matrix(probs, y)
    p, r = precision(tp, fp), recall(tp, fn)
    return {
        "loss": float(binary_cross_entropy(probs, y)),
        "accuracy": accuracy(tp, fp, fn, tn),
        "precision": p,
        "recall": r,
        "f1": f1_score(p, r),
    }
