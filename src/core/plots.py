"""Learning curves drawn from a training history."""
import matplotlib.pyplot as plt


def plot_learning_curves(history, path=None):
    """Loss and accuracy (train vs valid) per epoch, side by side.
    Saved to `path` if given, then displayed."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["loss"], label="training loss")
    ax_loss.plot(epochs, history["val_loss"], "--", label="validation loss")
    ax_loss.set_title("Loss")
    ax_loss.set_ylabel("binary cross-entropy")

    ax_acc.plot(epochs, history["accuracy"], label="training acc")
    ax_acc.plot(epochs, history["val_accuracy"], "--", label="validation acc")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylabel("accuracy")

    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("epochs")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    fig.suptitle("Learning Curves")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    plt.show()


def plot_comparison(runs, metrics, path=None):
    """One subplot per metric, one curve per run, to compare several trainings.
    runs: list of (name, history). Saved to `path` if given, then displayed."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        for name, history in runs:
            epochs = range(1, len(history[metric]) + 1)  # les runs peuvent avoir un nbr d'epochs différent
            ax.plot(epochs, history[metric], label=name)
        ax.set_title(metric)
        ax.set_xlabel("epochs")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("Model comparison")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    plt.show()
