"""
Data loading, normalization, splitting utilities.
"""
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(name="mnist"):
    from keras.datasets import mnist, fashion_mnist

    if name == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    # Flatten + normalize
    X_train_full = X_train_full.reshape(-1, 784).astype(np.float64) / 255.0
    X_test       = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    # 90/10 train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)
    y_test_oh  = one_hot(y_test)

    print(f"Dataset '{name}' | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, y_test


def one_hot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes), dtype=np.float64)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def get_batches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start: start + batch_size]
        yield X[idx], y[idx]