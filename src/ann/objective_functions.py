import numpy as np
from ann.activations import softmax

def to_onehot(y, num_classes=10):
    y = np.atleast_2d(np.atleast_1d(np.array(y, dtype=float)))
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
    return -np.sum(y_true * np.log(probs)) / y_true.shape[0]

def cross_entropy_grad(y_pred_logits, y_true):
    y_true = to_onehot(y_true)
    probs  = softmax(y_pred_logits)
    return (probs - y_true) / y_true.shape[0]

def mse_loss(y_pred_logits, y_true):
    y_true = to_onehot(y_true)
    probs  = softmax(y_pred_logits)
    return np.sum((probs - y_true) ** 2) / (2 * y_true.shape[0])

def mse_grad(y_pred_logits, y_true):
    y_true = to_onehot(y_true)
    probs  = softmax(y_pred_logits)
    dl_dp  = (probs - y_true) / y_true.shape[0]
    dot    = np.sum(dl_dp * probs, axis=1, keepdims=True)
    return probs * (dl_dp - dot)

LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "mse":           (mse_loss,           mse_grad),
}

def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'")
    return LOSSES[name]