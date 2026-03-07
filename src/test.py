import numpy as np
import argparse
from ann.neural_network import NeuralNetwork

best_config = argparse.Namespace(
    dataset="mnist",
    epochs=10,
    batch_size=64,
    loss="cross_entropy",
    optimizer="rmsprop",
    weight_decay=0.0,
    learning_rate=0.001,
    num_layers=3,
    hidden_size=[128, 128, 128],
    activation="relu",
    weight_init="xavier"
)

model = NeuralNetwork(best_config)
weights = np.load("best_model.npy", allow_pickle=True).item()
model.set_weights(weights)
print("Model loaded successfully!")
print("Number of layers:", len(model.layers))
for k, v in weights.items():
    print(k, v.shape)
