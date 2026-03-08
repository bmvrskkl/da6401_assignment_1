# DA6401 Assignment 1 — Multi layer Perceptron for Imgae Classification

- **GitHub**: https://github.com/bmvrskkl/da6401_assignment_1
- **W&B Report**: https://wandb.ai/bskkl04-indian-institute-of-technology-madras/da6401-mlp/reports/Untitled-Report--VmlldzoxNjE0MTM2OQ?accessToken=6sdjod5mdo0eaywar1wigzy28eavsji6ff3n9h149ccuv0swutbx8vqjhxkycbnh

## Results
- MNIST Test Accuracy: 97.94%
- Fashion-MNIST Test Accuracy: 88.74%

## Project Structure
```
├── README.md
├── requirements.txt
├── models/
└── src/
    ├── train.py
    ├── inference.py
    ├── test.py
    ├── best_model.npy
    ├── best_config.json
    ├── ann/
    │   ├── activations.py
    │   ├── neural_layer.py
    │   ├── neural_network.py
    │   ├── objective_functions.py
    │   └── optimizers.py
    └── utils/
        └── data_loader.py
```

## Installation
```bash
pip install -r requirements.txt
```

## How to Train
```bash
python3 src/train.py -d mnist -e 50 -b 32 -o adam -lr 0.0001 -nhl 4 -sz 128 128 128 128 -a relu -w_i xavier -wd 0.00005
```

## How to Run Inference
```bash
python3 src/inference.py --model src/best_model.npy --config src/best_config.json --dataset mnist
```

## How to Test (Autograder Pattern)
```bash
python3 src/test.py
```