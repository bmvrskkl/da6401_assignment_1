import numpy as np

class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
        layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self.v:
            self.v[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
        self.v[lid]['W'] = self.momentum * self.v[lid]['W'] - self.lr * (layer.grad_W + self.weight_decay * layer.W)
        self.v[lid]['b'] = self.momentum * self.v[lid]['b'] - self.lr * layer.grad_b
        layer.W += self.v[lid]['W']
        layer.b += self.v[lid]['b']

class NAG:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self.v:
            self.v[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
        v_prev_W = self.v[lid]['W'].copy()
        v_prev_b = self.v[lid]['b'].copy()
        self.v[lid]['W'] = self.momentum * v_prev_W - self.lr * (layer.grad_W + self.weight_decay * layer.W)
        self.v[lid]['b'] = self.momentum * v_prev_b - self.lr * layer.grad_b
        layer.W += -self.momentum * v_prev_W + (1 + self.momentum) * self.v[lid]['W']
        layer.b += -self.momentum * v_prev_b + (1 + self.momentum) * self.v[lid]['b']

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self.v:
            self.v[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
        gW = layer.grad_W + self.weight_decay * layer.W
        self.v[lid]['W'] = self.beta * self.v[lid]['W'] + (1 - self.beta) * gW ** 2
        self.v[lid]['b'] = self.beta * self.v[lid]['b'] + (1 - self.beta) * layer.grad_b ** 2
        layer.W -= self.lr * gW / (np.sqrt(self.v[lid]['W']) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.v[lid]['b']) + self.eps)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        self.t += 1
        lid = id(layer)
        if lid not in self.m:
            self.m[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            self.v[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
        gW = layer.grad_W + self.weight_decay * layer.W
        self.m[lid]['W'] = self.beta1 * self.m[lid]['W'] + (1 - self.beta1) * gW
        self.m[lid]['b'] = self.beta1 * self.m[lid]['b'] + (1 - self.beta1) * layer.grad_b
        self.v[lid]['W'] = self.beta2 * self.v[lid]['W'] + (1 - self.beta2) * gW ** 2
        self.v[lid]['b'] = self.beta2 * self.v[lid]['b'] + (1 - self.beta2) * layer.grad_b ** 2
        mW = self.m[lid]['W'] / (1 - self.beta1 ** self.t)
        mb = self.m[lid]['b'] / (1 - self.beta1 ** self.t)
        vW = self.v[lid]['W'] / (1 - self.beta2 ** self.t)
        vb = self.v[lid]['b'] / (1 - self.beta2 ** self.t)
        layer.W -= self.lr * mW / (np.sqrt(vW) + self.eps)
        layer.b -= self.lr * mb / (np.sqrt(vb) + self.eps)

class NAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        self.t += 1
        lid = id(layer)
        if lid not in self.m:
            self.m[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            self.v[lid] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
        gW = layer.grad_W + self.weight_decay * layer.W
        self.m[lid]['W'] = self.beta1 * self.m[lid]['W'] + (1 - self.beta1) * gW
        self.m[lid]['b'] = self.beta1 * self.m[lid]['b'] + (1 - self.beta1) * layer.grad_b
        self.v[lid]['W'] = self.beta2 * self.v[lid]['W'] + (1 - self.beta2) * gW ** 2
        self.v[lid]['b'] = self.beta2 * self.v[lid]['b'] + (1 - self.beta2) * layer.grad_b ** 2
        mW = self.m[lid]['W'] / (1 - self.beta1 ** self.t)
        mb = self.m[lid]['b'] / (1 - self.beta1 ** self.t)
        vW = self.v[lid]['W'] / (1 - self.beta2 ** self.t)
        vb = self.v[lid]['b'] / (1 - self.beta2 ** self.t)
        layer.W -= self.lr / (np.sqrt(vW) + self.eps) * (self.beta1 * mW + (1 - self.beta1) * gW / (1 - self.beta1 ** self.t))
        layer.b -= self.lr / (np.sqrt(vb) + self.eps) * (self.beta1 * mb + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.t))

def get_optimizer(name, lr=0.001, weight_decay=0.0, **kwargs):
    mapping = {
        'sgd':      SGD,
        'momentum': Momentum,
        'nag':      NAG,
        'rmsprop':  RMSProp,
        'adam':     Adam,
        'nadam':    NAdam,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown optimizer '{name}'")
    return mapping[key](lr=lr, weight_decay=weight_decay, **kwargs)