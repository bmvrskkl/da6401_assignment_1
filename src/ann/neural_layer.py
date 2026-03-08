import numpy as np
from ann.activations import get_activation

class NeuralLayer:
    def __init__(self, input_size, output_size, activation=None, weight_init="xavier"):
        self.input_size      = input_size
        self.output_size     = output_size
        self.activation_name = activation

        if weight_init == "xavier":
            scale = np.sqrt(1.0 / input_size)
            self.W = np.random.randn(input_size, output_size) * scale
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        else:
            self.W = np.zeros((input_size, output_size))

        self.b = np.zeros((1, output_size))
        self.act_fn, self.act_deriv = get_activation(activation)
        self.input  = None
        self.z      = None
        self.output = None
        self.grad_W = None
        self.grad_b = None
        self.optimizer_state = {}

    def forward(self, X):
        self.input = X
        self.z = X @ self.W + self.b
        self.output = self.act_fn(self.z) if self.act_fn else self.z
        return self.output

    def backward(self, delta, weight_decay=0.0):
        if self.act_deriv is not None:
            delta = delta * self.act_deriv(self.z)
        self.grad_W = (self.input.T @ delta) / self.input.shape[0]
        self.grad_b = np.mean(delta, axis=0, keepdims=True)
        if weight_decay > 0:
            self.grad_W += weight_decay * self.W
        return delta @ self.W.T