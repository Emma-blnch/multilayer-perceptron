"""Train the multilayer perceptron on the WDBC dataset.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from core.history import new_history, run_name, save_history
from core.metrics import METRICS, compute_metrics, confusion_matrix
from core.network import backpropagation, build_network, feedforward, gradient_descent, save_model
from core.plots import plot_learning_curves
from core.preprocessing import apply_normalization, fit_normalization, load_split, one_hot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/data_train.csv", help="training set (from split_data.py)")
    parser.add_argument("--valid", default="data/data_valid.csv", help="validation set (from split_data.py)")
    parser.add_argument("--model-out", default="saved_model.npz",
                        help="where to save the model: topology, weights and normalization stats")
    parser.add_argument("--plot-out", default="outputs/learning_curves.png", help="where to save the learning curves")
    parser.add_argument("--history-dir", default="history", help="where to save the metrics history (JSON)")
    parser.add_argument("--name", default=None,
                        help="name of this training run, used for the history file (default: built from the "
                             "hyperparameters, e.g. sigmoid_24-24_lr0.05_bs8_e70)")
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[24, 24],
                        help="size of each hidden layer, at least 2 layers (default: 24 24)")
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"],
                        help="activation function for the hidden layers (default: sigmoid)")
    parser.add_argument("--epochs", type=int, default=70, help="number of passes over the training set (default: 70)")
    parser.add_argument("--batch-size", type=int, default=8, help="examples per gradient step (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="gradient descent step size (default: 0.05)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for weight initialization and shuffling")
    args = parser.parse_args()

    if len(args.hidden_layers) < 2:
        sys.exit("error: --hidden-layers must have at least 2 layers")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        sys.exit("error: --epochs, --batch-size and --learning-rate must be positive")

    train_path, valid_path = Path(args.train), Path(args.valid)
    if not train_path.exists() or not valid_path.exists():
        sys.exit(f"error: {train_path} / {valid_path} not found (run src/split_data.py first)")

    _, X_train, labels_train = load_split(train_path)
    _, X_valid, labels_valid = load_split(valid_path)

    y_train = one_hot(labels_train)  # convertit M/B en vecteurs one-hot
    y_valid = one_hot(labels_valid)

    mean, std = fit_normalization(X_train)
    X_train = apply_normalization(X_train, mean, std)
    X_valid = apply_normalization(X_valid, mean, std)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    network = build_network(
        input_size=X_train.shape[1],
        hidden_layers=args.hidden_layers,
        hidden_activation=args.activation,
        seed=args.seed,
    )

    print("\nNetwork architecture:")
    for i, layer in enumerate(network):
        print(f"  layer {i + 1}: W{layer.W.shape} b{layer.b.shape} activation={layer.activation}")
    print()

    history = fit(network, X_train, y_train, X_valid, y_valid,
                  args.epochs, args.batch_size, args.learning_rate, args.seed)

    # mean/std sauvegardés avec le modèle : predict.py doit appliquer
    # exactement la même normalisation que celle vue à l'entraînement
    print(f"> saving model '{args.model_out}' to disk...")
    save_model(args.model_out, network, mean, std)

    # hyperparamètres sauvegardés avec l'historique, pour savoir à quel
    # entraînement correspond chaque fichier au moment de comparer
    params = {
        "hidden_layers": args.hidden_layers,
        "activation": args.activation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    history_path = Path(args.history_dir) / f"{args.name or run_name(params)}.json"
    print(f"> saving metrics history '{history_path}' to disk...")
    save_history(history_path, history, params)

    print_final_report(history, feedforward(network, X_valid), y_valid)

    Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
    print(f"> saving learning curves '{args.plot_out}' to disk...")
    plot_learning_curves(history, args.plot_out)


def fit(network, X_train, y_train, X_valid, y_valid, epochs, batch_size, learning_rate, seed=None):
    """Mini-batch gradient descent. Returns the history of the metrics, one
    entry per epoch in each list."""
    rng = np.random.default_rng(seed)
    # une liste par métrique (train + val_), une entrée par epoch
    history = new_history()
    width = max(2, len(str(epochs)))  # "epoch 01/70" -> padding des numéros d'epoch

    for epoch in range(1, epochs + 1):
        # mélange à chaque epoch pour que les batchs ne soient jamais les mêmes
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            probs = feedforward(network, X_train[batch])
            backpropagation(network, probs, y_train[batch])
            gradient_descent(network, learning_rate)

        record_epoch(history, network, X_train, y_train, X_valid, y_valid)
        print(f"epoch {epoch:0{width}d}/{epochs} - loss: {history['loss'][-1]:.4f}"
              f" - val_loss: {history['val_loss'][-1]:.4f}"
              f" - acc: {history['accuracy'][-1]:.4f} - val_acc: {history['val_accuracy'][-1]:.4f}")

    return history


def record_epoch(history, network, X_train, y_train, X_valid, y_valid):
    """Evaluate the network on train and valid (no weight update) and append
    the metrics to history."""
    for prefix, X, y in (("", X_train, y_train), ("val_", X_valid, y_valid)):
        for name, value in compute_metrics(feedforward(network, X), y).items():
            history[prefix + name].append(value)


def print_final_report(history, probs_valid, y_valid):
    """All the metrics of the last epoch (train vs valid) + the confusion
    matrix on the validation set."""
    print("\nFinal metrics:")
    print(f"  {'':<10} {'train':>8} {'valid':>8}")
    for name in METRICS:
        print(f"  {name:<10} {history[name][-1]:>8.4f} {history['val_' + name][-1]:>8.4f}")

    tp, fp, fn, tn = confusion_matrix(probs_valid, y_valid)
    print("\nConfusion matrix (valid):")
    print(f"  {'':<9} {'pred M':>7} {'pred B':>7}")
    print(f"  {'true M':<9} {tp:>7} {fn:>7}")
    print(f"  {'true B':<9} {fp:>7} {tn:>7}\n")


if __name__ == "__main__":
    main()
