import argparse
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, get_batches

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",         type=int,   default=20)
    p.add_argument("-b",   "--batch_size",     type=int,   default=64)
    p.add_argument("-l",   "--loss",           type=str,   default="cross_entropy", choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",   type=float, default=0.0005)
    p.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+", default=[128])
    p.add_argument("-a",   "--activation",     type=str,   default="relu", choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",    type=str,   default="xavier", choices=["random", "xavier","zeros"])
    p.add_argument("-w_p", "--wandb_project",  type=str,   default="da6401_assignment1")
    p.add_argument("--model_path",             type=str,   default="best_model.npy")
    return p.parse_args()

def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data

def main():
    args = parse_args()

    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    print(f"Loading dataset: {args.dataset}")
    _, _, X_test, _, _, y_test = load_dataset(args.dataset)

    model = NeuralNetwork(args)
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)

    print(f"Loading weights from: {model_path}")
    weights = load_model(model_path)
    model.set_weights(weights)

    preds = []
    for Xb, _ in get_batches(X_test, y_test, batch_size=512, shuffle=False):
        preds.append(model.predict(Xb))
    preds = np.concatenate(preds)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro", zero_division=0)
    f1   = f1_score(y_test, preds, average="macro", zero_division=0)
    cm   = confusion_matrix(y_test, preds)

    print("\n" + "=" * 50)
    print("           EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 50)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()