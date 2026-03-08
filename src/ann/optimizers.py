"""
Gradient-based optimizers.
Each optimizer exposes a single update(layer) method.
All support L2 weight decay via the weight_decay parameter.

Supported: sgd, momentum, nag (Nesterov), rmsprop
"""
import numpy as np


class SGD:
    """
    Vanilla mini-batch SGD (no momentum).
    W  ← W - lr * (grad_W + wd * W)
    """
    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr           = lr
        self.weight_decay = weight_decay

    def update(self, layer) -> None:
        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b
        layer.W -= self.lr * gW
        layer.b -= self.lr * gb


class MomentumSGD:
    """
    SGD with classical momentum.
    v  ← γ*v - lr * grad
    W  ← W + v
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.v: dict      = {}   # velocity buffer keyed by layer id

    def update(self, layer) -> None:
        lid = id(layer)
        if lid not in self.v:
            self.v[lid] = {
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b),
            }
        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        self.v[lid]['W'] = self.momentum * self.v[lid]['W'] - self.lr * gW
        self.v[lid]['b'] = self.momentum * self.v[lid]['b'] - self.lr * gb

        layer.W += self.v[lid]['W']
        layer.b += self.v[lid]['b']


class NAG:
    """
    Nesterov Accelerated Gradient (NAG).
    The gradient is evaluated at the 'lookahead' position:
        W_look = W + γ*v
    then:
        v  ← γ*v - lr * grad(W_look)
        W  ← W + v
    Implementation note: we apply the standard 'trick' for mini-batch NAG —
    at each update we undo the previous velocity, apply the gradient, then
    reapply the (new) velocity.
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.v: dict      = {}

    def update(self, layer) -> None:
        lid = id(layer)
        if lid not in self.v:
            self.v[lid] = {
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b),
            }
        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        v_prev_W = self.v[lid]['W'].copy()
        v_prev_b = self.v[lid]['b'].copy()

        self.v[lid]['W'] = self.momentum * v_prev_W - self.lr * gW
        self.v[lid]['b'] = self.momentum * v_prev_b - self.lr * gb

        # Nesterov update: -(γ*v_prev) + (1+γ)*v_new
        layer.W += -self.momentum * v_prev_W + (1 + self.momentum) * self.v[lid]['W']
        layer.b += -self.momentum * v_prev_b + (1 + self.momentum) * self.v[lid]['b']


class RMSProp:
    """
    RMSProp: adapts per-parameter learning rates using a running average of
    squared gradients.
    v  ← β*v + (1-β)*g²
    W  ← W - lr * g / sqrt(v + eps)
    """
    def __init__(self, lr: float = 0.001, beta: float = 0.9,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr           = lr
        self.beta         = beta
        self.eps          = eps
        self.weight_decay = weight_decay
        self.v: dict      = {}

    def update(self, layer) -> None:
        lid = id(layer)
        if lid not in self.v:
            self.v[lid] = {
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b),
            }
        gW = layer.grad_W + self.weight_decay * layer.W
        gb = layer.grad_b

        self.v[lid]['W'] = self.beta * self.v[lid]['W'] + (1 - self.beta) * gW ** 2
        self.v[lid]['b'] = self.beta * self.v[lid]['b'] + (1 - self.beta) * gb ** 2

        layer.W -= self.lr * gW / (np.sqrt(self.v[lid]['W']) + self.eps)
        layer.b -= self.lr * gb / (np.sqrt(self.v[lid]['b']) + self.eps)


def get_optimizer(name: str, lr: float = 0.001,
                  weight_decay: float = 0.0, **kwargs):
    """
    Factory function.  Maps CLI name to optimizer instance.
    Supported names: sgd, momentum, nag, rmsprop
    """
    key = name.lower()
    if key == 'sgd':
        return SGD(lr=lr, weight_decay=weight_decay)
    elif key == 'momentum':
        return MomentumSGD(lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif key == 'nag':
        return NAG(lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif key == 'rmsprop':
        return RMSProp(lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: sgd, momentum, nag, rmsprop"
        )