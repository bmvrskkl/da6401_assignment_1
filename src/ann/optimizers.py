import numpy as np

class SGD:
    def __init__(self, lr=0.01, **kwargs):
        self.lr = lr

    def update(self, layers, weight_decay=0.0):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, lr=0.01, beta=0.9, **kwargs):
        self.lr = lr
        self.beta = beta

    def _init_cache(self, layer):
        if "vW" not in layer.cache:
            layer.cache["vW"] = np.zeros_like(layer.W)
            layer.cache["vb"] = np.zeros_like(layer.b)

    def update(self, layers, weight_decay=0.0):
        for layer in layers:
            self._init_cache(layer)
            layer.cache["vW"] = self.beta * layer.cache["vW"] + self.lr * layer.grad_W
            layer.cache["vb"] = self.beta * layer.cache["vb"] + self.lr * layer.grad_b
            layer.W -= layer.cache["vW"]
            layer.b -= layer.cache["vb"]

class NAG:
    def __init__(self, lr=0.01, beta=0.9, **kwargs):
        self.lr = lr
        self.beta = beta

    def _init_cache(self, layer):
        if "vW" not in layer.cache:
            layer.cache["vW"] = np.zeros_like(layer.W)
            layer.cache["vb"] = np.zeros_like(layer.b)

    def update(self, layers, weight_decay=0.0):
        for layer in layers:
            self._init_cache(layer)
            vW_prev = layer.cache["vW"].copy()
            vb_prev = layer.cache["vb"].copy()
            layer.cache["vW"] = self.beta * vW_prev + self.lr * layer.grad_W
            layer.cache["vb"] = self.beta * vb_prev + self.lr * layer.grad_b
            layer.W -= (1 + self.beta) * layer.cache["vW"] - self.beta * vW_prev
            layer.b -= (1 + self.beta) * layer.cache["vb"] - self.beta * vb_prev

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8, **kwargs):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

    def _init_cache(self, layer):
        if "sW" not in layer.cache:
            layer.cache["sW"] = np.zeros_like(layer.W)
            layer.cache["sb"] = np.zeros_like(layer.b)

    def update(self, layers, weight_decay=0.0):
        for layer in layers:
            self._init_cache(layer)
            layer.cache["sW"] = self.beta * layer.cache["sW"] + (1 - self.beta) * layer.grad_W ** 2
            layer.cache["sb"] = self.beta * layer.cache["sb"] + (1 - self.beta) * layer.grad_b ** 2
            layer.W -= self.lr * layer.grad_W / (np.sqrt(layer.cache["sW"]) + self.epsilon)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(layer.cache["sb"]) + self.epsilon)

OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nag": NAG,
    "rmsprop": RMSProp,
}

def get_optimizer(name, **kwargs):
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'.")
    return OPTIMIZERS[name](**kwargs)