import numpy as np
from ann.activations import get_activation

class DenseLayer:
    def __init__(self, input_size, output_size, activation="relu", weight_init="xavier"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        # Weight initialisation
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Optimizer cache
        self.cache = {}

        # Forward pass cache
        self._input = None
        self._Z = None
        self._A = None

        # Activation function
        if activation == "linear":
            self._act = lambda z: z
            self._act_d = lambda z: np.ones_like(z)
        else:
            self._act, self._act_d = get_activation(activation)

    def forward(self, A_prev):
        self._input = A_prev
        self._Z = A_prev @ self.W + self.b
        self._A = self._act(self._Z)
        return self._A

    def backward(self, dA, weight_decay=0.0):
        batch_size = self._input.shape[0]
        dZ = dA * self._act_d(self._Z)
        self.grad_W = (self._input.T @ dZ) / batch_size + weight_decay * self.W
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dA_prev = dZ @ self.W.T
        return dA_prev