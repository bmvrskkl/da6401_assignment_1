import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss

def numerical_gradient(model, X, y, epsilon=1e-5):
    loss_fn = get_loss(model.args.loss)
    num_grads = []
    for layer in model.layers:
        dW = np.zeros_like(layer.W)
        db = np.zeros_like(layer.b)
        it = np.nditer(layer.W, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = layer.W[idx]
            layer.W[idx] = orig + epsilon
            loss_plus, _ = loss_fn(model.forward(X), y)
            layer.W[idx] = orig - epsilon
            loss_minus, _ = loss_fn(model.forward(X), y)
            dW[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            layer.W[idx] = orig
            it.iternext()
        it = np.nditer(layer.b, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = layer.b[idx]
            layer.b[idx] = orig + epsilon
            loss_plus, _ = loss_fn(model.forward(X), y)
            layer.b[idx] = orig - epsilon
            loss_minus, _ = loss_fn(model.forward(X), y)
            db[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            layer.b[idx] = orig
            it.iternext()
        num_grads.append({"grad_W": dW, "grad_b": db})
    return num_grads

def relative_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-10))

def run_gradient_check():
    import types
    args = types.SimpleNamespace(
        dataset="mnist", epochs=1, batch_size=8, loss="cross_entropy",
        optimizer="sgd", learning_rate=0.01, weight_decay=0.0,
        num_layers=2, hidden_size=16, activation="tanh",
        weight_init="xavier", wandb_project="test",
        no_wandb=True, seed=0, model_path="best_model.npy",
    )
    np.random.seed(0)
    model = NeuralNetwork(args)
    X = np.random.randn(8, 784).astype(np.float32) * 0.1
    y = np.random.randint(0, 10, size=8)
    logits = model.forward(X)
    _, anal_grads = model.backward(logits, y)
    num_grads = numerical_gradient(model, X, y)
    print("Gradient Check Results")
    print("=" * 50)
    all_pass = True
    for i, (ag, ng) in enumerate(zip(anal_grads, num_grads)):
        err_W = relative_error(ag["grad_W"], ng["grad_W"])
        err_b = relative_error(ag["grad_b"], ng["grad_b"])
        status_W = "PASS" if err_W < 1e-5 else "FAIL"
        status_b = "PASS" if err_b < 1e-5 else "FAIL"
        print(f"Layer {i}: grad_W={err_W:.2e} {status_W} | grad_b={err_b:.2e} {status_b}")
        if err_W >= 1e-5 or err_b >= 1e-5:
            all_pass = False
    print("=" * 50)
    print("Overall:", "ALL PASSED" if all_pass else "SOME FAILED")

if __name__ == "__main__":
    run_gradient_check()
