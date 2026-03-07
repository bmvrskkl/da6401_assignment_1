import argparse
import json
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, get_batches

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",         type=int,   default=20)
    p.add_argument("-b",   "--batch_size",     type=int,   default=64)
    p.add_argument("-l",   "--loss",           type=str,   default="cross_entropy", choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",   type=float, default=0.0005)
    p.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+", default=[128])
    p.add_argument("-a",   "--activation",     type=str,   default="relu", choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-w_i", "--weight_init",    type=str,   default="xavier", choices=["random", "xavier","zeros"])
    p.add_argument("-w_p", "--wandb_project",  type=str,   default="da6401_assignment1")
    p.add_argument("--wandb_entity",           type=str,   default=None)
    p.add_argument("--no_wandb",               action="store_true")
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--model_path",             type=str,   default="best_model.npy")
    return p.parse_args()

def evaluate(model, X, y, batch_size=512):
    preds = []
    for Xb, _ in get_batches(X, y, batch_size, shuffle=False):
        preds.append(model.predict(Xb))
    preds = np.concatenate(preds)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return acc, f1

def compute_loss(model, X, y, batch_size=512):
    losses = []
    for Xb, yb in get_batches(X, y, batch_size, shuffle=False):
        logits = model.forward(Xb)
        loss, _ = model.loss_fn(logits, yb)
        losses.append(loss)
    return float(np.mean(losses))

def train(args):
    np.random.seed(args.seed)

    print(f"Loading dataset: {args.dataset}")
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(args.dataset)

    if isinstance(args.hidden_size, (list, tuple)):
      if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    model = NeuralNetwork(args)
    optimizer = get_optimizer(args.optimizer, lr=args.learning_rate, beta=0.9, epsilon=1e-8)

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    best_f1 = -1.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            logits = model.forward(X_batch)
            loss, _ = model.backward(logits, y_batch)
            optimizer.update(model.layers, weight_decay=args.weight_decay)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_acc, train_f1 = evaluate(model, X_train, y_train)
        val_acc, val_f1 = evaluate(model, X_val, y_val)
        val_loss = compute_loss(model, X_val, y_val)
        test_acc, test_f1 = evaluate(model, X_test, y_test)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if use_wandb:
            grad_norm_l0 = float(np.linalg.norm(model.layers[0].grad_W))
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "val/loss": val_loss,
                "train/accuracy": train_acc,
                "val/accuracy": val_acc,
                "train/f1": train_f1,
                "val/f1": val_f1,
                "grad_norm/layer_0": grad_norm_l0,
            })

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_weights = model.get_weights()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(save_dir, "best_model.npy")
    config_path = os.path.join(save_dir, "best_config.json")

    np.save(model_path, best_weights)
    print(f"\nBest model saved → {model_path}  (test F1={best_f1:.4f})")

    config = vars(args).copy()
    config["best_test_f1"] = best_f1
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if use_wandb:
        wandb.finish()

    return model, best_weights

if __name__ == "__main__":
    args = parse_args()
    train(args)