"""Network architecture: layer definitions, weight initialization, activations,
feedforward, backpropagation, gradient descent, loss and model save/load."""
import numpy as np

EPSILON = 1e-15  # keeps log() away from 0 in the loss


# --- activations -------------------------------------------------------------
# Each activation comes with its derivative w.r.t. Z, needed by the backprop.
# Softmax has no entry in DERIVATIVES: combined with the cross-entropy loss, the
# output error simplifies to (A - Y), so its derivative is never used alone.

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)  # évite l'overflow de exp() sur des valeurs extrêmes
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(np.float64)


def softmax(Z):
    # soustraire le max de chaque ligne ne change pas le résultat mais évite
    # l'overflow de exp() (exp(1000) = inf)
    exps = np.exp(Z - Z.max(axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


ACTIVATIONS = {"sigmoid": sigmoid, "relu": relu, "softmax": softmax}
DERIVATIVES = {"sigmoid": sigmoid_derivative, "relu": relu_derivative}


# --- layers ------------------------------------------------------------------

class DenseLayer:
    """A fully connected layer: W (input_size, n_neurons), b (n_neurons,),
    plus its activation name."""

    def __init__(self, W, b, activation):
        self.W = W
        self.b = b
        self.activation = activation
        # cache rempli par forward(), réutilisé par la backprop
        self.A_prev = None
        self.Z = None
        # gradients remplis par backward(), utilisés par la descente de gradient
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        """A_prev: (n_samples, input_size) -> A: (n_samples, n_neurons)."""
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b  # somme pondérée de chaque neurone, pour chaque exemple
        return ACTIVATIONS[self.activation](self.Z)

    def backward(self, dZ):
        """From dZ = dLoss/dZ of this layer (n_samples, n_neurons), store dW/db
        and return dLoss/dA_prev, the gradient w.r.t. the previous layer output."""
        self.dW = self.A_prev.T @ dZ  # (input_size, n_neurons), même forme que W
        self.db = dZ.sum(axis=0)  # chaque exemple contribue au biais
        return dZ @ self.W.T  # renvoie l'erreur vers la couche précédente


def init_layer(input_size, n_neurons, activation, rng):
    """New DenseLayer with random weights."""
    # small random weights, scaled down by input_size so the weighted sums
    # don't start too large (that would saturate sigmoid/softmax early on).
    # Biases start at zero, only the weights need to break symmetry.
    W = rng.normal(0, 1, size=(input_size, n_neurons)) * np.sqrt(1 / input_size)
    return DenseLayer(W, np.zeros(n_neurons), activation)


def build_network(input_size, hidden_layers, output_size=2, hidden_activation="sigmoid", seed=None):
    """Build the list of DenseLayer from an input size and a list of hidden layer sizes.

    hidden_layers: e.g. [24, 24] for two hidden layers of 24 neurons each. The output layer is
    fixed to `output_size` neurons with softmax, as required by the subject.
    """
    rng = np.random.default_rng(seed)
    sizes = [input_size] + list(hidden_layers) + [output_size]
    activations = [hidden_activation] * len(hidden_layers) + ["softmax"]

    return [
        init_layer(sizes[i], sizes[i + 1], activation, rng)
        for i, activation in enumerate(activations)
    ]


# --- save / load -------------------------------------------------------------

def save_model(path, network, mean, std):
    """Save the topology (activations; layer sizes come from the W shapes),
    the weights and the normalization stats in a single .npz file, so the
    prediction program only needs this file."""
    arrays = {"activations": np.array([layer.activation for layer in network]), "mean": mean, "std": std}
    for i, layer in enumerate(network):
        arrays[f"W{i}"] = layer.W
        arrays[f"b{i}"] = layer.b
    np.savez(path, **arrays)


def load_model(path):
    """Inverse of save_model: return (network, mean, std)."""
    with np.load(path) as data:
        network = [
            DenseLayer(data[f"W{i}"], data[f"b{i}"], str(activation))
            for i, activation in enumerate(data["activations"])
        ]
        return network, data["mean"], data["std"]


# --- feedforward / backpropagation / loss ------------------------------------

def feedforward(network, X):
    """Propagate X through every layer, return the output probabilities
    (n_samples, output_size). Each layer caches its A_prev/Z along the way."""
    A = X
    for layer in network:
        A = layer.forward(A)  # la sortie d'une couche est l'entrée de la suivante
    return A


def backpropagation(network, probs, y):
    """Compute dW/db of every layer, from the output back to the input.
    Must be called right after feedforward(network, X), whose caches it reuses.

    probs: softmax output (n_samples, 2), y: one-hot labels (n_samples, 2).
    """
    # softmax + cross-entropy: la dérivée de la loss par rapport au Z de la
    # couche de sortie se simplifie en (probs - y). Divisé par n car la loss
    # est une moyenne sur les exemples.
    dZ = (probs - y) / len(y)
    for i in reversed(range(len(network))):
        dA_prev = network[i].backward(dZ)
        if i > 0:
            # règle de la chaîne : dZ = dA * f'(Z) pour la couche précédente
            prev = network[i - 1]
            dZ = dA_prev * DERIVATIVES[prev.activation](prev.Z)


def gradient_descent(network, learning_rate):
    """Move every weight/bias a small step against its gradient."""
    for layer in network:
        layer.W -= learning_rate * layer.dW
        layer.b -= learning_rate * layer.db


def binary_cross_entropy(probs, y):
    """Binary cross-entropy from the subject:
    E = -1/N * sum(y_n * log(p_n) + (1 - y_n) * log(1 - p_n))

    probs: softmax output (n_samples, 2), y: one-hot labels (n_samples, 2).
    p_n / y_n are taken on the M column (index 1): with a 2-neuron softmax the
    B column is just 1 - p_n, so it carries no extra information.
    """
    p = np.clip(probs[:, 1], EPSILON, 1 - EPSILON)
    t = y[:, 1]
    return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
