"""Data loading and preprocessing shared by
the training and prediction programs."""
import json

import numpy as np
import pandas as pd

CLASSES = ["B", "M"]  # fixed order: B -> [1, 0], M -> [0, 1]


def load_split(path):
    """Load a cleaned/split csv, return (ids, X, labels)."""
    df = pd.read_csv(path)
    ids = df["id"].to_numpy()
    labels = df["diagnosis"].to_numpy()  # M/B labels
    X = df.drop(columns=["id", "diagnosis"]).to_numpy(dtype=np.float64)  # matrice des features
    return ids, X, labels


def one_hot(labels):
    """Encode 'M'/'B' labels as one-hot rows, in CLASSES order."""
    y = np.zeros((len(labels), len(CLASSES)), dtype=np.float64)
    for col, cls in enumerate(CLASSES):
        y[labels == cls, col] = 1.0
    return y


def fit_normalization(X):
    """Compute per-column mean/std from X (call on the training set only)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero on a constant column
    return mean, std


def apply_normalization(X, mean, std):
    return (X - mean) / std


def save_normalization(path, mean, std):
    with open(path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)


def load_normalization(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["mean"]), np.array(data["std"])
