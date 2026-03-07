import numpy as np
from ann.activations import softmax

def cross_entropy_loss(logits, y_true):
    batch_size = logits.shape[0]
    probs = softmax(logits)
    probs_clipped = np.clip(probs, 1e-12, 1.0)
    loss = -np.mean(np.log(probs_clipped[np.arange(batch_size), y_true]))
    dlogits = probs.copy()
    dlogits[np.arange(batch_size), y_true] -= 1
    return loss, dlogits

def mse_loss(logits, y_true):
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    y_onehot = np.zeros((batch_size, num_classes))
    y_onehot[np.arange(batch_size), y_true] = 1.0
    diff = logits - y_onehot
    loss = np.mean(np.sum(diff ** 2, axis=1)) / 2.0
    dlogits = diff
    return loss, dlogits

LOSSES = {
    "cross_entropy": cross_entropy_loss,
    "mse": mse_loss,
}

def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'.")
    return LOSSES[name]