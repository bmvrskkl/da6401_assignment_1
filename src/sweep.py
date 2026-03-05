import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))

import wandb
import train as train_module

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "parameters": {
        "epochs":        {"value": 10},
        "batch_size":    {"values": [32, 64, 128]},
        "loss":          {"values": ["cross_entropy", "mse"]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "learning_rate": {"values": [0.1, 0.01, 0.001, 0.0001]},
        "weight_decay":  {"values": [0.0, 0.0005, 0.001]},
        "num_layers":    {"values": [2, 3, 4, 5]},
        "hidden_size":   {"values": [32, 64, 128]},
        "activation":    {"values": ["sigmoid", "tanh", "relu"]},
        "weight_init":   {"values": ["random", "xavier"]},
    },
}

def sweep_run():
    run = wandb.init()
    cfg = wandb.config

    args = types.SimpleNamespace(
        dataset="fashion_mnist",
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        loss=cfg.loss,
        optimizer=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        activation=cfg.activation,
        weight_init=cfg.weight_init,
        wandb_project="da6401_assignment1",
        wandb_entity=None,
        no_wandb=False,
        seed=42,
        model_path="best_model.npy",
    )

    train_module.train(args)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_project", type=str, default="da6401_assignment1")
    p.add_argument("--count", type=int, default=100)
    cli = p.parse_args()

    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        project=cli.wandb_project
    )
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_run, count=cli.count)

if __name__ == "__main__":
    main()