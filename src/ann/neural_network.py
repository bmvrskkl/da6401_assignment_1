"""
NeuralNetwork class — matches exact autograder interface.
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.activations import softmax
from ann.objective_functions import get_loss
from ann.optimizers import NAG


def to_onehot(y_true, num_classes=10):
    """Handles scalar, 0-D, 1-D integer labels, or already one-hot (N,10)."""
    y = np.atleast_1d(np.array(y_true))

    if y.ndim == 1:
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return one_hot

    return y


class NeuralNetwork:

    def __init__(self, config):

        self.config = config

        input_size = 784
        output_size = 10

        hidden_sizes = (
            config.hidden_size
            if isinstance(config.hidden_size, list)
            else [config.hidden_size] * config.num_layers
        )

        activation = config.activation
        weight_init = config.weight_init
        loss = config.loss

        self.loss_fn, self.loss_grad = get_loss(loss)

        self.layers = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):

            is_output = (i == len(layer_sizes) - 2)

            act = None if is_output else activation

            layer = NeuralLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=act,
                weight_init=weight_init,
            )

            self.layers.append(layer)

    def forward(self, X):
        """Forward pass — returns logits"""
        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def predict(self, X):

        logits = self.forward(X)

        probs = softmax(logits)

        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_true_onehot):

        preds = self.predict(X)

        true_labels = np.argmax(y_true_onehot, axis=1)

        return np.mean(preds == true_labels)

    def backward(self, logits, y_true, weight_decay=0.0):

        y_true = to_onehot(y_true)

        # 🔧 FIX: compute gradient using softmax probabilities
        probs = softmax(logits)

        delta = probs - y_true

        for layer in reversed(self.layers):

            delta = layer.backward(delta, weight_decay=weight_decay)

    def compute_loss(self, logits, y_true, weight_decay=0.0):

        y_true = to_onehot(y_true)

        loss = self.loss_fn(logits, y_true)

        if weight_decay > 0:

            l2 = sum(np.sum(layer.W ** 2) for layer in self.layers)

            loss += 0.5 * weight_decay * l2

        return loss

    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):

            weights[f"W{i}"] = layer.W.copy()

            weights[f"b{i}"] = layer.b.copy()

        return weights

    def set_weights(self, weights):

        for i, layer in enumerate(self.layers):

            if f"W{i}" in weights:

                layer.W = weights[f"W{i}"]

                layer.b = weights[f"b{i}"]

            elif f"layer_{i}" in weights:

                layer.W = weights[f"layer_{i}"]["W"]

                layer.b = weights[f"layer_{i}"]["b"]

    def train_step(self, X_batch, y_batch, optimizer, weight_decay=0.0, is_nag=False):

        if is_nag:

            for layer in self.layers:
                optimizer.lookahead(layer)

        logits = self.forward(X_batch)

        if is_nag:

            for layer in self.layers:
                optimizer.restore(layer)

        loss = self.compute_loss(logits, y_batch, weight_decay)

        self.backward(logits, y_batch, weight_decay)

        for layer in self.layers:
            optimizer.update(layer)

        return loss