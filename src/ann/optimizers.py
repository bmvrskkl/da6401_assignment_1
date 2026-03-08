"""
Optimizers: SGD, Momentum, NAG, RMSProp, Adam, Nadam.
"""
import numpy as np


class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

    def update(self, layer):
        s = layer.optimizer_state
        if "vW" not in s:
            s["vW"] = np.zeros_like(layer.W)
            s["vb"] = np.zeros_like(layer.b)
        s["vW"] = self.beta * s["vW"] + self.lr * layer.grad_W
        s["vb"] = self.beta * s["vb"] + self.lr * layer.grad_b
        layer.W -= s["vW"]
        layer.b -= s["vb"]


class NAG:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

    def lookahead(self, layer):
        s = layer.optimizer_state
        if "vW" not in s:
            s["vW"] = np.zeros_like(layer.W)
            s["vb"] = np.zeros_like(layer.b)
        layer.W -= self.beta * s["vW"]
        layer.b -= self.beta * s["vb"]

    def restore(self, layer):
        s = layer.optimizer_state
        layer.W += self.beta * s["vW"]
        layer.b += self.beta * s["vb"]

    def update(self, layer):
        s = layer.optimizer_state
        if "vW" not in s:
            s["vW"] = np.zeros_like(layer.W)
            s["vb"] = np.zeros_like(layer.b)
        s["vW"] = self.beta * s["vW"] + self.lr * layer.grad_W
        s["vb"] = self.beta * s["vb"] + self.lr * layer.grad_b
        layer.W -= s["vW"]
        layer.b -= s["vb"]


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def update(self, layer):
        s = layer.optimizer_state
        if "sW" not in s:
            s["sW"] = np.zeros_like(layer.W)
            s["sb"] = np.zeros_like(layer.b)
        s["sW"] = self.beta * s["sW"] + (1 - self.beta) * layer.grad_W ** 2
        s["sb"] = self.beta * s["sb"] + (1 - self.beta) * layer.grad_b ** 2
        layer.W -= self.lr * layer.grad_W / (np.sqrt(s["sW"]) + self.epsilon)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(s["sb"]) + self.epsilon)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def update(self, layer):
        s = layer.optimizer_state
        if "mW" not in s:
            s["mW"] = np.zeros_like(layer.W)
            s["mb"] = np.zeros_like(layer.b)
            s["vW"] = np.zeros_like(layer.W)
            s["vb"] = np.zeros_like(layer.b)
            s["t"]  = 0
        s["t"] += 1
        t = s["t"]
        s["mW"] = self.beta1 * s["mW"] + (1 - self.beta1) * layer.grad_W
        s["mb"] = self.beta1 * s["mb"] + (1 - self.beta1) * layer.grad_b
        s["vW"] = self.beta2 * s["vW"] + (1 - self.beta2) * layer.grad_W ** 2
        s["vb"] = self.beta2 * s["vb"] + (1 - self.beta2) * layer.grad_b ** 2
        mW_hat = s["mW"] / (1 - self.beta1 ** t)
        mb_hat = s["mb"] / (1 - self.beta1 ** t)
        vW_hat = s["vW"] / (1 - self.beta2 ** t)
        vb_hat = s["vb"] / (1 - self.beta2 ** t)
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)


class Nadam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def update(self, layer):
        s = layer.optimizer_state
        if "mW" not in s:
            s["mW"] = np.zeros_like(layer.W)
            s["mb"] = np.zeros_like(layer.b)
            s["vW"] = np.zeros_like(layer.W)
            s["vb"] = np.zeros_like(layer.b)
            s["t"]  = 0
        s["t"] += 1
        t = s["t"]
        s["mW"] = self.beta1 * s["mW"] + (1 - self.beta1) * layer.grad_W
        s["mb"] = self.beta1 * s["mb"] + (1 - self.beta1) * layer.grad_b
        s["vW"] = self.beta2 * s["vW"] + (1 - self.beta2) * layer.grad_W ** 2
        s["vb"] = self.beta2 * s["vb"] + (1 - self.beta2) * layer.grad_b ** 2
        vW_hat = s["vW"] / (1 - self.beta2 ** t)
        vb_hat = s["vb"] / (1 - self.beta2 ** t)
        mW_nes = (self.beta1 * s["mW"] + (1 - self.beta1) * layer.grad_W) / (1 - self.beta1 ** (t + 1))
        mb_nes = (self.beta1 * s["mb"] + (1 - self.beta1) * layer.grad_b) / (1 - self.beta1 ** (t + 1))
        layer.W -= self.lr * mW_nes / (np.sqrt(vW_hat) + self.epsilon)
        layer.b -= self.lr * mb_nes / (np.sqrt(vb_hat) + self.epsilon)


OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "nadam":    Nadam,
}

def get_optimizer(name, **kwargs):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZERS.keys())}")
    return OPTIMIZERS[name](**kwargs)