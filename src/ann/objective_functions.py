"""
Objective (loss) functions.
Both losses work on raw logits; softmax is applied internally.
The backward() method returns dL/d(logits) ready to be passed into the last layer.
"""
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class CrossEntropyLoss:
    """
    Cross-entropy loss combined with softmax.
    L = -mean( log( softmax(logits)[true_class] ) )
    """
    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        self.probs      = softmax(logits)
        self.y_true     = np.array(y_true, dtype=int)
        self.batch_size = logits.shape[0]
        eps = 1e-12
        chosen = self.probs[np.arange(self.batch_size), self.y_true]
        return float(-np.log(chosen + eps).mean())

    def backward(self) -> np.ndarray:
        """dL/d(logits) = (softmax - one_hot) / batch_size"""
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), self.y_true] -= 1.0
        return grad / self.batch_size


class MSELoss:
    """
    Mean-squared error between softmax probabilities and one-hot targets.
    L = mean( (softmax(logits) - one_hot)^2 )
    Gradient flows through the softmax Jacobian.
    """
    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        self.probs      = softmax(logits)
        self.y_true     = np.array(y_true, dtype=int)
        self.batch_size = logits.shape[0]
        one_hot         = np.eye(logits.shape[1])[self.y_true]   # (batch, C)
        self.diff       = self.probs - one_hot                    # (batch, C)
        return float((self.diff ** 2).mean())

    def backward(self) -> np.ndarray:
        """
        d(MSE)/d(logits) via chain rule through softmax.
        dL/d(logits_i) = sum_j [ dL/d(prob_j) * d(prob_j)/d(logit_i) ]
        where d(prob_j)/d(logit_i) = prob_i*(delta_ij - prob_j).
        """
        # dL/d(prob) = 2 * diff / (batch * C)
        dA = 2.0 * self.diff / (self.batch_size * self.diff.shape[1])
        # multiply by softmax Jacobian: p*(dA - sum(dA*p))
        return self.probs * (dA - (dA * self.probs).sum(axis=1, keepdims=True))


def get_loss(name: str):
    """Factory: return a loss instance by name."""
    mapping = {
        'cross_entropy':      CrossEntropyLoss,
        'mean_squared_error': MSELoss,
        'mse':                MSELoss,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(mapping)}")
    return mapping[key]()