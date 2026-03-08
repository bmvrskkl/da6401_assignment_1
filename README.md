# DA6401 Assignment 1 — Multi-Layer Perceptron for Image Classification

Implementation of a Multi-Layer Perceptron (MLP) from scratch using NumPy only, trained on MNIST and Fashion-MNIST datasets.

## Links
- **GitHub**: https://github.com/bmvrskkl/da6401_assignment_1
- **W&B Report**: https://wandb.ai/bskkl04-indian-institute-of-technology-madras/da6401_assignment1/reports/DA6401-Assignment-1-MLP-for-Image-Classification--VmlldzoxNjEzMTI5OA?accessToken=klfy3eglgex5fgvmm8ar3ojbua3xwrymvgjsgx9n06dotwnpohjswzhg8ko5rwge
## Project Structure
```
da6401_assignment_1/
├── README.md
├── requirements.txt
├── models/
└── src/
    ├── train.py
    ├── inference.py
    ├── test.py
    ├── explore.py
    ├── best_model.npy
    ├── best_config.json
    ├── ann/
    │   ├── __init__.py
    │   ├── activations.py
    │   ├── neural_layer.py
    │   ├── neural_network.py
    │   ├── objective_functions.py
    │   └── optimizers.py
    └── utils/
        ├── __init__.py
        └── data_loader.py
```

## Setup
```bash
git clone https://github.com/bmvrskkl/da6401_assignment_1
cd da6401_assignment_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python3 src/train.py -d mnist -e 50 -b 32 -o adam -lr 0.0001 -nhl 4 -sz 128 128 128 128 -a relu -w_i xavier -wd 0.00005
```

## Train with W&B logging
```bash
python3 src/train.py -d mnist -e 50 -b 32 -o adam -lr 0.0001 -nhl 4 -sz 128 128 128 128 -a relu -w_i xavier -wd 0.00005 --wandb --wandb_project da6401-mlp
```

## Inference
```bash
python3 src/inference.py --model src/best_model.npy --config src/best_config.json --dataset mnist
```

## Best Results
| Dataset | Optimizer | Layers | Hidden Size | Activation | Test Accuracy |
|---------|-----------|--------|-------------|------------|---------------|
| MNIST | Adam | 4 | 128×4 | ReLU | 97.94% |
| Fashion-MNIST | Adam | 4 | 128×4 | ReLU | 88.74% |

## Supported Options
- **Optimizers**: sgd, momentum, nag, rmsprop, adam, nadam
- **Activations**: sigmoid, tanh, relu
- **Loss functions**: cross_entropy, mse
- **Weight init**: random, xavier, zeros