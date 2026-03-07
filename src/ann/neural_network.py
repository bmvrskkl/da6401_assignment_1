import numpy as np
from ann.neural_layer import DenseLayer
from ann.objective_functions import get_loss
from ann.activations import softmax

class NeuralNetwork:
    def __init__(self,
                 input_size=784,
                 hidden_sizes=None,
                 output_size=10,
                 activation='relu',
                 weight_init='xavier',
                 loss='cross_entropy',
                 num_layers=None,
                 hidden_size=None):

        import argparse
        if isinstance(input_size, argparse.Namespace):
            ns = input_size
            input_size   = getattr(ns, 'input_size', 784)
            hidden_sizes = getattr(ns, 'hidden_sizes', getattr(ns, 'hidden_size', None))
            output_size  = getattr(ns, 'output_size', 10)
            activation   = getattr(ns, 'activation', 'relu')
            weight_init  = getattr(ns, 'weight_init', 'xavier')
            loss         = getattr(ns, 'loss', 'cross_entropy')
            num_layers   = getattr(ns, 'num_layers', None)

        if hidden_sizes is None:
            hs = hidden_size
            nl = num_layers
            if hs is not None and nl is not None:
                hidden_sizes = [int(hs)] * int(nl) if isinstance(hs, (int, np.integer)) \
                               else [int(h) for h in hs]
            elif hs is not None:
                hidden_sizes = [int(h) for h in hs] if hasattr(hs, '__iter__') \
                               else [int(hs)]
            elif nl is not None:
                hidden_sizes = [128] * int(nl)
            else:
                hidden_sizes = [128]

        if isinstance(hidden_sizes, (int, np.integer)):
            hidden_sizes = [int(hidden_sizes)]
        hidden_sizes = [int(h) for h in hidden_sizes]

        self.input_size   = int(input_size)
        self.output_size  = int(output_size)
        self.hidden_sizes = hidden_sizes
        self.activation   = str(activation)
        self.weight_init  = str(weight_init)
        self.loss_name    = str(loss)
        self.loss_fn      = get_loss(str(loss))
        self.weight_decay = 0.0

        sizes = [self.input_size] + hidden_sizes + [self.output_size]
        self.layers = []
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = "linear" if is_output else activation
            self.layers.append(
                DenseLayer(sizes[i], sizes[i+1],
                           activation=act, weight_init=weight_init)
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
        return gradients

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

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