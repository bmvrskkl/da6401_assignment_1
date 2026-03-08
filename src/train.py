import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import numpy as np

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer, NAG
from utils.data_loader import load_dataset, get_batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist")
    parser.add_argument("-e",   "--epochs",        type=int,   default=50)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=32)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy")
    parser.add_argument("-o",   "--optimizer",     type=str,   default="adam")
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.00005)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=4)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128,128,128,128])
    parser.add_argument("-a",   "--activation",    type=str,   default="relu")
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier")
    parser.add_argument("--wandb",                 action="store_true")
    parser.add_argument("--wandb_project",         type=str,   default="da6401-mlp")
    return parser.parse_args()

# also expose as parse_arguments for autograder
parse_arguments = parse_args


def train(config):
    use_wandb = getattr(config, 'wandb', False)
    if use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, config=vars(config))

    X_train, X_val, X_test, y_train, y_val, y_test_oh, y_test_raw = load_dataset(config.dataset)

    if isinstance(config.hidden_size, int):
        config.hidden_size = [config.hidden_size] * config.num_layers
    elif len(config.hidden_size) == 1:
        config.hidden_size = config.hidden_size * config.num_layers

    model     = NeuralNetwork(config)
    optimizer = get_optimizer(config.optimizer, lr=config.learning_rate, weight_decay=config.weight_decay)
    is_nag    = isinstance(optimizer, NAG)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, "best_model.npy")
    cfg_path   = os.path.join(script_dir, "best_config.json")

    best_val_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        total_loss, n = 0.0, 0
        for X_batch, y_batch in get_batches(X_train, y_train, config.batch_size):
            loss = model.train_step(X_batch, y_batch, optimizer,
                                    weight_decay=config.weight_decay, is_nag=is_nag)
            total_loss += loss
            n += 1

        avg_loss  = total_loss / n
        train_acc = model.accuracy(X_train, y_train)
        val_acc   = model.accuracy(X_val, y_val)
        val_logits = model.forward(X_val)
        val_loss   = model.compute_loss(val_logits, y_val)

        print(f"Epoch {epoch:3d}/{config.epochs} | loss={avg_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss,
                       "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.save(save_path, model.get_weights())
            with open(cfg_path, "w") as f:
                json.dump(vars(config), f, indent=2)

    print(f"\nBest val acc: {best_val_acc:.4f}")

    weights  = np.load(save_path, allow_pickle=True).item()
    model.set_weights(weights)
    test_acc = model.accuracy(X_test, y_test_oh)
    print(f"Test Accuracy: {test_acc:.4f}")

    if use_wandb:
        import wandb
        wandb.log({"test_acc": test_acc})
        wandb.finish()

    return model


if __name__ == "__main__":
    train(parse_args())