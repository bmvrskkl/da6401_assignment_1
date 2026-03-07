import numpy as np
from ann.neural_layer import DenseLayer
from ann.objective_functions import get_loss
from ann.activations import softmax

class NeuralNetwork:
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10

    def __init__(self, args):
        self.args = args
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.loss_fn = get_loss(args.loss)

        # Resolve hidden sizes
        hs = args.hidden_size
        nl = args.num_layers

        if isinstance(hs, (list, tuple)):
            hidden_sizes = [int(h) for h in hs]
            if len(hidden_sizes) == 1:
                hidden_sizes = hidden_sizes * nl
        else:
            hidden_sizes = [int(hs)] * nl

        self.layers = []
        in_size = self.INPUT_SIZE

        for h in hidden_sizes:
            self.layers.append(
                DenseLayer(in_size, h, activation=args.activation, weight_init=args.weight_init)
            )
            in_size = h

        self.layers.append(
            DenseLayer(in_size, self.OUTPUT_SIZE, activation="linear", weight_init=args.weight_init)
        )

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_loss(self, logits, y_true):
        loss, _ = self.loss_fn(logits, y_true)
        return loss

    def backward(self, logits, y_true):
        loss, dA = self.loss_fn(logits, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, weight_decay=self.weight_decay)
        gradients = [{"grad_W": l.grad_W, "grad_b": l.grad_b} for l in self.layers]
        return loss, gradients

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def predict_proba(self, X):
        return softmax(self.forward(X))

    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'W{i}'] = layer.W.copy()
            weights[f'b{i}'] = layer.b.copy()
        return weights

    def set_weights(self, weights):
        if isinstance(weights, np.ndarray) and weights.ndim == 0:
            weights = weights.item()
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f'W{i}' in weights:
                    layer.W = np.array(weights[f'W{i}']).copy()
                if f'b{i}' in weights:
                    layer.b = np.array(weights[f'b{i}']).copy()
            return
