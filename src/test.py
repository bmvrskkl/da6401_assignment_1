"""
test.py — Simulates exact autograder pattern.
Run: python3 src/test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import argparse
from sklearn.metrics import f1_score
from ann.neural_network import NeuralNetwork
from ann.activations import softmax

best_config = argparse.Namespace(
    dataset="mnist",
    epochs=50,
    batch_size=32,
    loss="cross_entropy",
    optimizer="adam",
    weight_decay=0.00005,
    learning_rate=0.0001,
    num_layers=4,
    hidden_size=[128, 128, 128, 128],
    activation="relu",
    weight_init="xavier"
)

model = NeuralNetwork(best_config)

model_path = "src/best_model.npy" if os.path.exists("src/best_model.npy") else "best_model.npy"
weights = np.load(model_path, allow_pickle=True).item()
model.set_weights(weights)

# Test forward
X_test = np.random.rand(100, 784)
y_true = np.random.randint(0, 10, size=(100,))
logits = model.forward(X_test)
probs  = softmax(logits)
y_pred_labels = np.argmax(probs, axis=1)
print("Forward OK, logits shape:", logits.shape)

# Test backward
model.backward(logits, y_true)
print("Backward OK, grad_W shape:", model.layers[0].grad_W.shape)

# Test F1
print("F1 Score:", f1_score(y_true, y_pred_labels, average='macro'))
print("test.py passed! autograder should work!")