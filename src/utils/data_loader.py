import numpy as np

def load_dataset(dataset="mnist", val_split=0.1):
    if dataset == "mnist":
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test  = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    val_size = int(len(X_train) * val_split)
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]

    def to_onehot(y, n=10):
        oh = np.zeros((len(y), n))
        oh[np.arange(len(y)), y] = 1.0
        return oh

    return (X_train, X_val, X_test,
            to_onehot(y_train), to_onehot(y_val), to_onehot(y_test),
            y_test)

def get_batches(X, y, batch_size):
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# alias
def load_data(dataset="mnist", val_split=0.1):
    X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, y_test_raw = load_dataset(dataset, val_split)
    return X_train, X_val, X_test, np.argmax(y_train_oh, axis=1), np.argmax(y_val_oh, axis=1), y_test_raw