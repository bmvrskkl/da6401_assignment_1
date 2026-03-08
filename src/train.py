import sys, os, argparse, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_data


# CLI
def parse_arguments():
    p = argparse.ArgumentParser(description='Train MLP on MNIST / Fashion-MNIST')

    # ----- data & training -----
    p.add_argument('-d',    '--dataset',       type=str,   default='mnist',
                   help='mnist | fashion_mnist')
    p.add_argument('-e',    '--epochs',        type=int,   default=10,
                   help='Number of training epochs')
    p.add_argument('-b',    '--batch_size',    type=int,   default=64,
                   help='Mini-batch size')
    p.add_argument('-l',    '--loss',          type=str,   default='cross_entropy',
                   help='cross_entropy | mean_squared_error')
    p.add_argument('-o',    '--optimizer',     type=str,   default='rmsprop',
                   help='sgd | momentum | nag | rmsprop')
    p.add_argument('-lr',   '--learning_rate', type=float, default=0.001,
                   help='Initial learning rate')
    p.add_argument('-wd',   '--weight_decay',  type=float, default=0.0,
                   help='L2 regularisation coefficient')

    # ----- architecture -----
    p.add_argument('-nhl',  '--num_layers',    type=int,   default=3,
                   help='Number of hidden layers')
    p.add_argument('-sz',   '--hidden_size',   type=int,   nargs='+', default=[128,128,128],
                   help='Neurons per hidden layer (one value or a list)')
    p.add_argument('-a',    '--activation',    type=str,   default='relu',
                   help='sigmoid | tanh | relu')
    p.add_argument('-w_i',  '--weight_init',   type=str,   default='xavier',
                   help='random | xavier')

    # ----- logging -----
    p.add_argument('-w_p',  '--wandb_project', type=str,   default=None,
                   help='W&B project name (optional)')
    p.add_argument('--wandb_entity',           type=str,   default=None)
    p.add_argument('--run_name',               type=str,   default=None)

    # ----- paths -----
    p.add_argument('--model_path', type=str, default=None,
                   help='Where to save best_model.npy (default: src/best_model.npy)')
    p.add_argument('--config_path', type=str, default=None,
                   help='Where to save best_config.json')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args,local_save=True):
    # ---- Load data ----
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(
        args.dataset, val_split=0.1)

    # ---- Resolve hidden sizes ----
    hs = args.hidden_size
    if isinstance(hs, int):
        hs = [hs]
    hs = [int(h) for h in hs]
    nl = args.num_layers
    if nl and nl != len(hs):
        hs = hs * nl if len(hs) == 1 else hs[:nl]

    # ---- Build model ----
    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=hs,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        loss=args.loss,
    )
    opt = get_optimizer(args.optimizer,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay)

    # ---- Optional W&B initialisation ----
    use_wandb = False
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=getattr(args, 'wandb_entity', None),
                name=getattr(args, 'run_name', None),
                config=vars(args),
            )
            use_wandb = True
        except Exception as e:
            print(f'[W&B] Could not initialise: {e}')

    # ---- Training ----
    from sklearn.metrics import f1_score as skf1

    best_val_f1      = -1.0
    best_weights     = None

    for epoch in range(1, args.epochs + 1):
        # Shuffle training data each epoch
        idx = np.random.permutation(len(x_train))
        xtr, ytr = x_train[idx], y_train[idx]

        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(xtr), args.batch_size):
            xb = xtr[i: i + args.batch_size]
            yb = ytr[i: i + args.batch_size]

            logits      = model.forward(xb)
            batch_loss  = model.compute_loss(logits, yb)
            total_loss += batch_loss
            n_batches  += 1

            model.backward()                          # compute gradients
            for layer in model.layers:
                opt.update(layer)                     # apply parameter update

        avg_loss = total_loss / max(n_batches, 1)

        # ---- Validation metrics ----
        val_preds  = model.predict(x_val)
        val_acc    = float(np.mean(val_preds == y_val))
        val_f1     = float(skf1(y_val, val_preds, average='weighted', zero_division=0))

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}')

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'train_loss': avg_loss,
                       'val_acc': val_acc, 'val_f1': val_f1})

        # Save best model by validation F1
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = model.get_weights()

    # ---- Restore best weights ----
    model.set_weights(best_weights)

    # ---- Test evaluation ----
    test_preds = model.predict(x_test)
    test_acc   = float(np.mean(test_preds == y_test))
    test_f1    = float(skf1(y_test, test_preds, average='weighted', zero_division=0))
    print(f'\nTest acc={test_acc:.4f} | Test F1={test_f1:.4f}')

    if use_wandb:
        import wandb
        wandb.log({'test_acc': test_acc, 'test_f1': test_f1})
        wandb.finish()

    # ---- Save model & config ----
    src_dir = _THIS_DIR   # same directory as train.py (project root)

    model_path  = args.model_path  or os.path.join(src_dir, 'best_model.npy')
    config_path = args.config_path or os.path.join(src_dir, 'best_config.json')

    os.makedirs(os.path.dirname(os.path.abspath(model_path)),  exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)

    if local_save:
        np.save(model_path, best_weights)

    cfg = {
        'dataset':       args.dataset,
        'hidden_sizes':  hs,
        'hidden_size':   hs,
        'num_layers':    len(hs),
        'activation':    args.activation,
        'weight_init':   args.weight_init,
        'loss':          args.loss,
        'optimizer':     args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay':  args.weight_decay,
        'batch_size':    args.batch_size,
        'epochs':        args.epochs,
        'best_val_f1':   best_val_f1,
        'test_f1':       test_f1,
    }
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f'Saved model  → {model_path}')
    print(f'Saved config → {config_path}')
    return model

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    train(parse_arguments(),False)