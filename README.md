# DA6401 — Assignment 1: Multi-Layer Perceptron for Image Classification

> **W&B Report:** https://wandb.ai/bskkl04-indian-institute-of-technology-madras/da6401_assignment1/reports/DA6401-Assignment-1-MLP-for-Image-Classification--VmlldzoxNjExNjIxNw?accessToken=4eiuyx1wbdd8dx01t7cwk9zlc5yg315gwy4fpm5d95e6u85spreftc9ouj7wkza7
> **GitHub Repository:** https://github.com/bmvrskkl/da6401_assignment_1

# Overview
A fully configurable MLP implemented from scratch using only NumPy.
Trained and evaluated on MNIST and Fashion-MNIST.

# Installation
pip install numpy scikit-learn matplotlib wandb keras tensorflow

# Training
py -3.11 train.py -d fashion_mnist -e 20 -b 64 -o rmsprop -lr 0.001 -nhl 3 -sz 128 -a relu -w_i xavier -w_p da6401_assignment1

# Inference
py -3.11 inference.py --model_path best_model.npy -d fashion_mnist