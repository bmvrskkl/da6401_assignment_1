"""
Inference Script — Load saved model and evaluate on test set.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ann.neural_network import NeuralNetwork
from ann.activations import softmax
from utils.data_loader import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="src/best_model.npy")
    parser.add_argument("--config",  type=str, default="src/best_config.json")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    return parser.parse_args()


def load_model(model_path, config):
    model   = NeuralNetwork(config)
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)
    return model


def evaluate_model(model, X_test, y_test_raw):
    logits = model.forward(X_test)
    probs  = softmax(logits)
    y_pred = np.argmax(probs, axis=1)

    acc       = accuracy_score(y_test_raw, y_pred)
    precision = precision_score(y_test_raw, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test_raw, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test_raw, y_pred, average="weighted", zero_division=0)

    return {
        "logits":    logits,
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def main():
    args = parse_arguments()
    with open(args.config, "r") as f:
        cfg_dict = json.load(f)
    config = argparse.Namespace(**cfg_dict)

    _, _, X_test, _, _, _, y_test_raw = load_dataset(args.dataset)
    model   = load_model(args.model, config)
    results = evaluate_model(model, X_test, y_test_raw)

    print("\n" + "=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print("=" * 50)
    return results


if __name__ == "__main__":
    main()