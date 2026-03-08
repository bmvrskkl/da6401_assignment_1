import numpy as np
from ann.activations import get_activation


class NeuralLayer:

    def __init__(self, input_size, output_size, activation=None, weight_init="xavier"):

        self.input_size = input_size
        self.output_size = output_size

        self.activation_name = activation
        self.activation, self.activation_grad = get_activation(activation)

        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))

        self.b = np.zeros((1, output_size))

    def forward(self, X):

        self.input = X

        self.Z = X @ self.W + self.b

        if self.activation is None:
            return self.Z

        self.A = self.activation(self.Z)

        return self.A

    def backward(self, delta, weight_decay=0.0):

        if self.activation is not None:
            delta = delta * self.activation_grad(self.Z)

        m = self.input.shape[0]

        self.dW = (self.input.T @ delta) / m + weight_decay * self.W

        self.db = np.sum(delta, axis=0, keepdims=True) / m

        delta_prev = delta @ self.W.T

        return delta_prev