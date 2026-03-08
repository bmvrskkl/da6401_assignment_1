"""
Activation functions for the MLP.
Each class implements forward() and backward() for use in backpropagation.
"""
import numpy as np


class ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)"""
    def forward(self, x):
        self.mask = x > 0          # cache for backward
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask    # gradient is 0 where x <= 0


class Sigmoid:
    """Sigmoid: f(x) = 1 / (1 + exp(-x))"""
    def forward(self, x):
        # clip to avoid overflow in exp
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Tanh:
    """Hyperbolic tangent: f(x) = tanh(x)"""
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)


class Identity:
    """No-op activation used for the output layer (returns logits)."""
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad


def get_activation(name: str):
    """Factory: return an activation instance by name."""
    mapping = {
        'relu':    ReLU,
        'sigmoid': Sigmoid,
        'tanh':    Tanh,
        'none':    Identity,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(mapping)}")
    return mapping[key]()