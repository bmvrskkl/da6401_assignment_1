"""
Loss functions and their gradients w.r.t. output logits.
"""
import numpy as np
from ann.activations import softmax


def to_onehot(y, num_classes=10):
    """Handles scalar, 0-D, 1-D integer labels, or already one-hot (N,10)."""
    y = np.atleast_2d(np.atleast_1d(np.array(y, dtype=float)))
    # If shape is (N,1) or (1,N) and not (N,10), treat as integer labels
    if y.shape[1] != num_classes:
        labels = y.flatten().astype(int)
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1.0
        return one_hot
    return y


def cross_entropy_loss(y_pred_logits, y_true):
    y_true = to_onehot(y_true)
    probs  = softmax(y_pred_logits)
    probs  = np.clip(probs, 1e-12, 1.0)
    N      = y_true.shape[0]
    return -np.sum(y_true * np.log(probs)) / N


def cross_entropy_grad(y_pred_logits, y_true):
    y_true = to_onehot(y_true)
    probs  = softmax(y_pred_logits)
    N      = y_true.shape[0]
    return (probs - y_true) / N


def mse_loss(y_pred_logits, y_true):
    y_true = to_onehot(y_true)
    probs  = softmax(y_pred_logits)
    N      = y_true.shape[0]
    return np.sum((probs - y_true) ** 2) / (2 * N)


def mse_grad(y_pred_logits, y_true):
    y_true = to_onehot(y_true)
    probs  = softmax(y_pred_logits)
    N      = y_true.shape[0]
    dl_dp  = (probs - y_true) / N
    dot    = np.sum(dl_dp * probs, axis=1, keepdims=True)
    return probs * (dl_dp - dot)


LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "mse":           (mse_loss,           mse_grad),
}


def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSSES.keys())}")
    return LOSSES[name]