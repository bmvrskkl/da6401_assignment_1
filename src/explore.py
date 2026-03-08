"""
2.1 Data Exploration — Log sample images to W&B Table.
Run: python3 src/explore.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import wandb
from keras.datasets import mnist

# Initialize W&B
wandb.init(project="da6401-mlp", name="2.1-data-exploration")

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Class names
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Create W&B Table
table = wandb.Table(columns=["image", "label", "class_name"])

# Log 5 samples per class
for class_idx in range(10):
    # Find indices for this class
    indices = np.where(y_train == class_idx)[0]
    # Pick 5 random samples
    samples = np.random.choice(indices, 5, replace=False)
    for idx in samples:
        img = X_train[idx]  # shape (28, 28)
        table.add_data(
            wandb.Image(img),
            class_idx,
            class_names[class_idx]
        )

wandb.log({"sample_images": table})
wandb.finish()
print("Done! Check your W&B report ✅")