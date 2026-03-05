import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name):
    dataset_name = dataset_name.lower().replace("-", "_")
    if dataset_name == "mnist":
        from keras.datasets import mnist as ds
    elif dataset_name in ("fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist as ds
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    (X_train_full, y_train_full), (X_test, y_test) = ds.load_data()
    X_train_full = X_train_full.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_batches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        batch_idx = indices[start: start + batch_size]
        yield X[batch_idx], y[batch_idx]