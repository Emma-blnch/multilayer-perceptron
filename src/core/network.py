"""Network architecture: layer definitions and weight initialization."""
import numpy as np


class DenseLayer:
    """A fully connected layer: input_size -> n_neurons,
    plus its activation name."""

    def __init__(self, input_size, n_neurons, activation, rng):
        # small random weights, scaled down by input_size so the weighted sums
        # don't start too large (that would saturate sigmoid/softmax early on).
        # Biases start at zero, only the weights need to break symmetry.
        self.W = rng.normal(0, 1, size=(input_size, n_neurons)) * np.sqrt(1 / input_size)
        self.b = np.zeros(n_neurons)
        self.activation = activation


def build_network(input_size, hidden_layers, output_size=2, hidden_activation="sigmoid", seed=None):
    """Build the list of DenseLayer from an input size and a list of hidden layer sizes.

    hidden_layers: e.g. [24, 24] for two hidden layers of 24 neurons each (the
    subject requires at least 2 hidden layers by default). The output layer is
    fixed to `output_size` neurons with softmax, as required by the subject.
    """
    rng = np.random.default_rng(seed)
    sizes = [input_size] + list(hidden_layers) + [output_size]
    activations = [hidden_activation] * len(hidden_layers) + ["softmax"]

    return [
        DenseLayer(sizes[i], sizes[i + 1], activation, rng)
        for i, activation in enumerate(activations)
    ]
