"""Train the multilayer perceptron on the WDBC dataset.

Step 1 (this file, so far): load the train/valid splits, one-hot encode the
diagnosis, and standardize the features using stats fitted on the training
set only. The network/backprop/training loop are added in later steps.
"""
import argparse
import sys
from pathlib import Path

from core.network import build_network
from core.preprocessing import apply_normalization, fit_normalization, load_split, one_hot, save_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/data_train.csv", help="training set (from split_data.py)")
    parser.add_argument("--valid", default="data/data_valid.csv", help="validation set (from split_data.py)")
    parser.add_argument("--norm-out", default="data/normalization.json", help="where to save the normalization stats")
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[24, 24],
                         help="size of each hidden layer, at least 2 layers (default: 24 24)")
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"],
                         help="activation function for the hidden layers (default: sigmoid)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for weight initialization")
    args = parser.parse_args()

    if len(args.hidden_layers) < 2:
        sys.exit("error: --hidden-layers must have at least 2 layers")

    train_path, valid_path = Path(args.train), Path(args.valid)
    if not train_path.exists() or not valid_path.exists():
        sys.exit(f"error: {train_path} / {valid_path} not found (run src/split_data.py first)")

    _, X_train, labels_train = load_split(train_path)
    _, X_valid, labels_valid = load_split(valid_path)

    y_train = one_hot(labels_train)
    y_valid = one_hot(labels_valid)

    mean, std = fit_normalization(X_train)
    X_train = apply_normalization(X_train, mean, std)
    X_valid = apply_normalization(X_valid, mean, std)
    save_normalization(args.norm_out, mean, std)

    print(f"Train: {X_train.shape[0]} rows, {X_train.shape[1]} features -> y {y_train.shape}")
    print(f"Valid: {X_valid.shape[0]} rows, {X_valid.shape[1]} features -> y {y_valid.shape}")
    print(f"Normalization stats saved -> {args.norm_out}")

    network = build_network(
        input_size=X_train.shape[1],
        hidden_layers=args.hidden_layers,
        hidden_activation=args.activation,
        seed=args.seed,
    )

    print("\nNetwork architecture:")
    for i, layer in enumerate(network):
        print(f"  layer {i + 1}: W{layer.W.shape} b{layer.b.shape} activation={layer.activation}")


if __name__ == "__main__":
    main()

