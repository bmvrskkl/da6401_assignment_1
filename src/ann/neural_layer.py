"""
A single fully-connected layer with configurable activation and weight initialisation.
After each forward/backward call the gradients are exposed as self.grad_W and self.grad_b
for the autograder and optimizer to consume.
"""
import numpy as np
from .activations import get_activation, Identity


class NeuralLayer:
    def __init__(self, input_size: int, output_size: int,
                 activation, weight_init: str = 'xavier'):
        """
        Args:
            input_size:   Number of input features.
            output_size:  Number of neurons (output features).
            activation:   An instantiated activation object (ReLU, Sigmoid, etc.).
            weight_init:  'xavier' | 'random' | 'zeros'
        """
        self.activation = activation
        self.input_size  = input_size
        self.output_size = output_size

        # ---- Weight initialisation ----
        if weight_init == 'xavier':
            # Glorot uniform: keeps variance stable across layers
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'random':
            self.W = np.random.randn(input_size, output_size) * 0.01
        else:  # zeros (for symmetry-breaking experiment)
            self.W = np.zeros((input_size, output_size))

        self.b = np.zeros((1, output_size))

        # Gradients – initialised to zero; populated by backward()
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Slot for optimizer-specific state (velocities, moments, etc.)
        self.optimizer_state: dict = {}

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute  z = x @ W + b  then apply activation.
        Caches x and z for use in backward().
        """
        self.x = x                           # (batch, input_size)
        self.z = x @ self.W + self.b         # (batch, output_size)
        return self.activation.forward(self.z)

    # ------------------------------------------------------------------
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Given upstream gradient dL/d(activation_output), compute:
          dL/dz   via activation.backward
          dL/dW   = x^T  @ dL/dz
          dL/db   = sum over batch
          dL/dx   = dL/dz @ W^T   (passed to previous layer)

        Stores self.grad_W and self.grad_b for the optimizer.
        Returns dL/dx to propagate further back.
        """
        grad_z         = self.activation.backward(grad_output)  # (batch, output_size)
        self.grad_W    = self.x.T @ grad_z                      # (input_size, output_size)
        self.grad_b    = grad_z.sum(axis=0, keepdims=True)      # (1, output_size)
        return grad_z @ self.W.T                                 # (batch, input_size)