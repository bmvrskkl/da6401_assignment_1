import numpy as np
from ann.neural_layer import NeuralLayer
from ann.activations import softmax
from ann.objective_functions import get_loss
from ann.optimizers import NAG

def to_onehot(y_true, num_classes=10):
    y = np.atleast_1d(np.array(y_true))
    if y.ndim == 1:
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return one_hot
    return y

class NeuralNetwork:
    def __init__(self, config):
        import argparse
        if not isinstance(config, argparse.Namespace):
            config = argparse.Namespace(**config)

        hidden_sizes = config.hidden_size if isinstance(config.hidden_size, list) \
                       else [config.hidden_size] * config.num_layers

        self.loss_fn, self.loss_grad = get_loss(config.loss)
        self.layers = []
        layer_sizes = [784] + hidden_sizes + [10]

        for i in range(len(layer_sizes) - 1):
            is_output = (i == len(layer_sizes) - 2)
            act = None if is_output else config.activation
            self.layers.append(NeuralLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i+1],
                activation=act,
                weight_init=config.weight_init,
            ))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        return np.argmax(softmax(self.forward(X)), axis=1)

    def accuracy(self, X, y_true_onehot):
        return np.mean(self.predict(X) == np.argmax(y_true_onehot, axis=1))

    def backward(self, logits, y_true, weight_decay=0.0):
        y_true = to_onehot(y_true)
        delta  = self.loss_grad(logits, y_true)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=weight_decay)

    def compute_loss(self, logits, y_true, weight_decay=0.0):
        y_true = to_onehot(y_true)
        loss   = self.loss_fn(logits, y_true)
        if weight_decay > 0:
            loss += 0.5 * weight_decay * sum(np.sum(l.W**2) for l in self.layers)
        return loss

    def get_weights(self):
        return {f"W{i}": l.W.copy() for i, l in enumerate(self.layers)} | \
               {f"b{i}": l.b.copy() for i, l in enumerate(self.layers)}

    def set_weights(self, weights):
        if isinstance(weights, np.ndarray) and weights.ndim == 0:
            weights = weights.item()
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weights:
                layer.W = weights[f"W{i}"]
                layer.b = weights[f"b{i}"]

    def train_step(self, X_batch, y_batch, optimizer, weight_decay=0.0, is_nag=False):
        logits = self.forward(X_batch)
        loss   = self.compute_loss(logits, y_batch, weight_decay)
        self.backward(logits, y_batch, weight_decay)
        for layer in self.layers:
            optimizer.update(layer)
        return loss