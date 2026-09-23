"""Training history: metrics per epoch + the hyperparameters that produced them,
saved as JSON so several trainings can be compared afterwards."""
import json
from pathlib import Path

from core.metrics import METRICS


def new_history():
    """One empty list per metric, for train (e.g. "f1") and valid ("val_f1")."""
    return {prefix + name: [] for name in METRICS for prefix in ("", "val_")}


def run_name(params):
    """Default name built from the hyperparameters, e.g. sigmoid_24-24_lr0.05_bs8_e70."""
    layers = "-".join(str(n) for n in params["hidden_layers"])
    return (f"{params['activation']}_{layers}_lr{params['learning_rate']}"
            f"_bs{params['batch_size']}_e{params['epochs']}")


def save_history(path, history, params):
    """Write {"params": ..., "history": ...} to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"params": params, "history": history}, f, indent=2)


def load_history(path):
    """Inverse of save_history: return (history, params)."""
    with open(path) as f:
        data = json.load(f)
    return data["history"], data["params"]
