"""Load a trained model, predict a dataset and evaluate the predictions with
the binary cross-entropy (plus the other metrics and the confusion matrix)."""
import argparse
import csv
import sys
from pathlib import Path

from core.metrics import METRICS, compute_metrics, confusion_matrix, print_confusion_matrix
from core.network import feedforward, load_model
from core.preprocessing import CLASSES, apply_normalization, load_split, one_hot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="data/data_valid.csv",
                        help="labeled csv to predict, same format as the split files (default: data/data_valid.csv)")
    parser.add_argument("--model", default="saved_model.npz", help="model saved by train.py")
    parser.add_argument("--out", default="outputs/predictions.csv",
                        help="where to save the prediction of each patient (default: outputs/predictions.csv)")
    args = parser.parse_args()

    model_path, data_path = Path(args.model), Path(args.dataset)
    if not model_path.exists():
        sys.exit(f"error: {model_path} not found (run src/train.py first)")
    if not data_path.exists():
        sys.exit(f"error: {data_path} not found")

    network, mean, std = load_model(model_path)
    ids, X, labels = load_split(data_path)

    expected = network[0].W.shape[0]  # nbr d'entrées de la 1ère couche = nbr de features vues à l'entraînement
    if X.shape[1] != expected:
        sys.exit(f"error: {data_path} has {X.shape[1]} features, the model expects {expected}")

    # mean/std de l'entraînement, surtout pas recalculés sur ce dataset
    X = apply_normalization(X, mean, std)
    y = one_hot(labels)

    print(f"> loaded model '{model_path}' ({len(network)} layers)")
    print(f"x shape : {X.shape}")

    probs = feedforward(network, X)
    save_predictions(args.out, ids, probs, labels)
    print(f"> saving predictions '{args.out}' to disk...")

    metrics = compute_metrics(probs, y)
    print(f"\nbinary cross-entropy loss: {metrics['loss']:.4f}")
    print("\nMetrics:")
    for name in METRICS[1:]:  # la loss est déjà affichée au-dessus
        print(f"  {name:<10} {metrics[name]:>8.4f}")

    print("\nConfusion matrix:")
    print_confusion_matrix(*confusion_matrix(probs, y))


def save_predictions(path, ids, probs, labels):
    """One row per patient: id, predicted class, probability of M, true class."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prediction", "prob_M", "diagnosis"])
        for i, p, label in zip(ids, probs, labels):
            writer.writerow([i, CLASSES[p.argmax()], f"{p[1]:.4f}", label])


if __name__ == "__main__":
    main()
