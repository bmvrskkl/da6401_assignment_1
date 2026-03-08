import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z = np.atleast_2d(z)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh":    (tanh,    tanh_derivative),
    "relu":    (relu,    relu_derivative),
}

def get_activation(name):
    if name is None:
        return None, None
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]