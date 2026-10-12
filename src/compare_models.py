"""Compare several trainings: draw the curves of their saved histories
(history/*.json, written by train.py) on the same graphs."""
import argparse
import sys
from pathlib import Path

from core.history import load_history, new_history
from core.plots import plot_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("histories", nargs="+", help="history JSON files to compare (e.g. history/*.json)")
    parser.add_argument("--metrics", nargs="+", default=["val_loss", "val_accuracy"],
                        choices=list(new_history()),  # toutes les clés de l'historique (loss, val_loss, f1...)
                        help="metrics to plot, one graph each (default: val_loss val_accuracy)")
    parser.add_argument("--out", default="outputs/comparison.png", help="where to save the graph")
    args = parser.parse_args()

    runs = []
    for p in map(Path, args.histories):
        if not p.exists():
            sys.exit(f"error: {p} not found")
        history, params = load_history(p)
        runs.append((p.stem, history))  # nom du fichier = nom du run (--name ou hyperparamètres)
        print(f"{p.stem}: final val_loss {history['val_loss'][-1]:.4f}"
              f" - val_accuracy {history['val_accuracy'][-1]:.4f} - {params}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"> saving comparison graph '{args.out}' to disk...")
    plot_comparison(runs, args.metrics, args.out)


if __name__ == "__main__":
    main()
