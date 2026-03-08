"""
NeuralNetwork: a configurable, fully-connected MLP built from NeuralLayer objects.

Key design points:
  - forward()  returns raw logits (no softmax) as required by the spec.
  - backward() returns gradients from last layer to first (list of grad_W, grad_b).
  - get_weights() / set_weights() follow the format required by the autograder.
"""
import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, Identity
from .objective_functions import get_loss, softmax


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
        """
        Accepts either keyword arguments directly or an argparse.Namespace
        as the first positional argument (for CLI convenience).
        """
        import argparse
        # Support passing an argparse.Namespace as first argument
        if isinstance(input_size, argparse.Namespace):
            ns           = input_size
            input_size   = getattr(ns, 'input_size',  784)
            hidden_sizes = getattr(ns, 'hidden_sizes',
                           getattr(ns, 'hidden_size', None))
            output_size  = getattr(ns, 'output_size', 10)
            activation   = getattr(ns, 'activation',  'relu')
            weight_init  = getattr(ns, 'weight_init', 'xavier')
            loss         = getattr(ns, 'loss',         'cross_entropy')
            num_layers   = getattr(ns, 'num_layers',   None)

        # ---- Resolve hidden layer sizes ----
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

        # ---- Store config ----
        self.input_size   = int(input_size)
        self.output_size  = int(output_size)
        self.hidden_sizes = hidden_sizes
        self.activation   = str(activation)
        self.weight_init  = str(weight_init)
        self.loss_name    = str(loss)
        self.loss_fn      = get_loss(str(loss))

        # ---- Build layers ----
        # Hidden layers use the chosen activation; output layer is linear (logits).
        sizes = [self.input_size] + hidden_sizes + [self.output_size]
        self.layers = []
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = Identity() if is_output else get_activation(activation)
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i + 1],
                            activation=act, weight_init=weight_init)
            )

    # ------------------------------------------------------------------
    # Forward pass — returns LOGITS (no softmax)
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward through all layers; returns raw logits."""
        self._last_input = x
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out  # logits, shape (batch, output_size)

    # ------------------------------------------------------------------
    # Convenience prediction helpers
    # ------------------------------------------------------------------
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def compute_loss(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_fn.forward(logits, y_true)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, y_true=None, y_pred=None,
                 weight_decay: float = 0.0, **kwargs):
        """
        Compute gradients from the last layer back to the first.

        If y_pred (logits) and y_true are supplied the method re-derives the
        loss gradient; otherwise it uses whatever the loss_fn cached in its
        most recent forward() call.

        Returns:
            grad_W_list  – list of dL/dW for each layer (last → first)
            grad_b_list  – list of dL/db for each layer (last → first)
        """
        # Determine starting gradient (dL/d_logits)
        if y_pred is not None and y_true is not None:
            probs      = softmax(y_pred)
            batch_size = probs.shape[0]
            y_true     = np.array(y_true, dtype=int)
            grad       = probs.copy()
            grad[np.arange(batch_size), y_true] -= 1.0
            grad      /= batch_size
            # Manually apply gradient for output layer
            out_layer         = self.layers[-1]
            out_layer.grad_W  = out_layer.x.T @ grad
            out_layer.grad_b  = grad.sum(axis=0, keepdims=True)
            grad              = grad @ out_layer.W.T
            # Propagate through remaining layers (reversed, excluding output)
            for layer in reversed(self.layers[:-1]):
                grad = layer.backward(grad)
        else:
            # Use loss_fn's cached gradient
            grad = self.loss_fn.backward()
            if grad is None:
                grad = np.zeros((1, self.output_size))
            # Propagate through ALL layers in reverse
            for layer in reversed(self.layers):
                grad = layer.backward(grad)

        # Collect and return gradients last→first
        grad_W_list = [layer.grad_W for layer in reversed(self.layers)]
        grad_b_list = [layer.grad_b for layer in reversed(self.layers)]
        return grad_W_list, grad_b_list

    # ------------------------------------------------------------------
    # Weight serialisation (autograder-compatible)
    # ------------------------------------------------------------------
    def get_weights(self) -> dict:
        """Return a flat dict {W0, b0, W1, b1, ...} of numpy arrays."""
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'W{i}'] = layer.W.copy()
            weights[f'b{i}'] = layer.b.copy()
        return weights

    def set_weights(self, weights) -> None:
        """
        Accept weights as:
          - dict with keys W0/b0/W1/b1/...
          - numpy 0-d object array (as saved by np.save)
          - list of (W, b) pairs
        """
        if isinstance(weights, np.ndarray) and weights.ndim == 0:
            weights = weights.item()

        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f'W{i}' in weights:
                    layer.W = np.array(weights[f'W{i}']).copy()
                if f'b{i}' in weights:
                    layer.b = np.array(weights[f'b{i}']).copy()
            return

        # Fallback: list of arrays [W0, b0, W1, b1, ...]
        weights = list(weights)
        if len(weights) == 2 * len(self.layers):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[2 * i]).copy()
                layer.b = np.array(weights[2 * i + 1]).copy()
        else:
            for layer, pair in zip(self.layers, weights):
                pair = list(pair)
                layer.W = np.array(pair[0]).copy()
                layer.b = np.array(pair[1]).copy()

    def save(self, path: str) -> None:
        import os, json
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.get_weights())
        # Also persist a JSON copy (more portable)
        json_path = path.replace('.npy', '.json')
        with open(json_path, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.get_weights().items()}, f)
        print(f'Model saved → {path}')

    def load(self, path: str) -> None:
        import json
        json_path = path.replace('.npy', '.json')
        import os
        if os.path.exists(json_path):
            with open(json_path) as f:
                d = json.load(f)
            self.set_weights({k: np.array(v) for k, v in d.items()})
        else:
            data = np.load(path, allow_pickle=True)
            self.set_weights(data)
        print(f'Model loaded ← {path}')