"""
NeuralNetwork class — matches exact autograder interface.
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.activations import softmax
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer, NAG


class NeuralNetwork:
    def __init__(self, config):
        self.config = config

        input_size   = 784
        output_size  = 10
        hidden_sizes = config.hidden_size if isinstance(config.hidden_size, list) \
                       else [config.hidden_size] * config.num_layers
        activation   = config.activation
        weight_init  = config.weight_init
        loss         = config.loss

        self.loss_fn, self.loss_grad = get_loss(loss)

        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            is_output = (i == len(layer_sizes) - 2)
            act = None if is_output else activation
            self.layers.append(
                NeuralLayer(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    activation=act,
                    weight_init=weight_init,
                )
            )

    def forward(self, X):
        """Returns (probs, logits) — autograder expects tuple."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        logits = out
        probs  = softmax(logits)
        return probs, logits

    def backward(self, logits, y_true, weight_decay=0.0):
        delta = self.loss_grad(logits, y_true)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=weight_decay)

    def compute_loss(self, logits, y_true, weight_decay=0.0):
        loss = self.loss_fn(logits, y_true)
        if weight_decay > 0:
            l2 = sum(np.sum(l.W ** 2) for l in self.layers)
            loss += 0.5 * weight_decay * l2
        return loss

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_true_onehot):
        preds = self.predict(X)
        true_labels = np.argmax(y_true_onehot, axis=1)
        return np.mean(preds == true_labels)

    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}"] = {"W": layer.W.copy(), "b": layer.b.copy()}
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.layers):
            key = f"layer_{i}"
            layer.W = weights[key]["W"]
            layer.b = weights[key]["b"]

    def train_step(self, X_batch, y_batch, optimizer, weight_decay=0.0, is_nag=False):
        if is_nag:
            for layer in self.layers:
                optimizer.lookahead(layer)
        probs, logits = self.forward(X_batch)
        if is_nag:
            for layer in self.layers:
                optimizer.restore(layer)
        loss = self.compute_loss(logits, y_batch, weight_decay)
        self.backward(logits, y_batch, weight_decay)
        for layer in self.layers:
            optimizer.update(layer)
        return loss