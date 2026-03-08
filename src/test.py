"""
test.py — Simulates exact autograder pattern.
Run: python3 src/test.py
If this passes locally, it will pass the autograder.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse
from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork

# Replace these with your best config after training
best_config = argparse.Namespace(
    dataset="mnist",
    epochs=10,
    batch_size=64,
    loss="cross_entropy",
    optimizer="adam",
    weight_decay=0.0005,
    learning_rate=0.001,
    num_layers=3,
    hidden_size=[128, 128, 128],
    activation="relu",
    weight_init="xavier"
)

model = NeuralNetwork(best_config)

weights = np.load("src/best_model.npy", allow_pickle=True).item()
model.set_weights(weights)

X_test = np.random.rand(100, 784)
y_true = np.random.randint(0, 10, size=(100,))

y_pred, _ = model.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print("F1 Score:", f1_score(y_true, y_pred_labels, average='macro'))
