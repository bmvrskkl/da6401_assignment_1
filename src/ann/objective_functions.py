"""
Loss functions and their gradients w.r.t. output logits.
"""
import numpy as np
from ann.activations import softmax


def cross_entropy_loss(y_pred_logits, y_true_onehot):
    probs = softmax(y_pred_logits)
    probs = np.clip(probs, 1e-12, 1.0)
    N = y_true_onehot.shape[0]
    return -np.sum(y_true_onehot * np.log(probs)) / N

def cross_entropy_grad(y_pred_logits, y_true_onehot):
    probs = softmax(y_pred_logits)
    N = y_true_onehot.shape[0]
    return (probs - y_true_onehot) / N

def mse_loss(y_pred_logits, y_true_onehot):
    probs = softmax(y_pred_logits)
    N = y_true_onehot.shape[0]
    return np.sum((probs - y_true_onehot) ** 2) / (2 * N)

def mse_grad(y_pred_logits, y_true_onehot):
    probs = softmax(y_pred_logits)
    N = y_true_onehot.shape[0]
    dl_dp = (probs - y_true_onehot) / N
    dot = np.sum(dl_dp * probs, axis=1, keepdims=True)
    return probs * (dl_dp - dot)

LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "mse":           (mse_loss,           mse_grad),
}

def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSSES.keys())}")
    return LOSSES[name]