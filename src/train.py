import sys, os, argparse, json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_THIS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, get_batches

def parse_arguments():
    p = argparse.ArgumentParser(description='Train MLP on MNIST / Fashion-MNIST')
    p.add_argument('-d',   '--dataset',       type=str,   default='mnist')
    p.add_argument('-e',   '--epochs',        type=int,   default=10)
    p.add_argument('-b',   '--batch_size',    type=int,   default=64)
    p.add_argument('-l',   '--loss',          type=str,   default='cross_entropy')
    p.add_argument('-o',   '--optimizer',     type=str,   default='rmsprop')
    p.add_argument('-lr',  '--learning_rate', type=float, default=0.001)
    p.add_argument('-wd',  '--weight_decay',  type=float, default=0.0)
    p.add_argument('-nhl', '--num_layers',    type=int,   default=3)
    p.add_argument('-sz',  '--hidden_size',   type=int,   nargs='+', default=[128])
    p.add_argument('-a',   '--activation',    type=str,   default='relu')
    p.add_argument('-w_i', '--weight_init',   type=str,   default='xavier')
    p.add_argument('-w_p', '--wandb_project', type=str,   default=None)
    p.add_argument('--wandb_entity',          type=str,   default=None)
    p.add_argument('--no_wandb',              action='store_true')
    p.add_argument('--seed',                  type=int,   default=42)
    p.add_argument('--model_path',            type=str,   default=None)
    return p.parse_args()

def train(args):
    np.random.seed(args.seed)

    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(args.dataset)

    # Resolve hidden sizes
    hs = args.hidden_size
    if isinstance(hs, int):
        hs = [hs]
    hs = [int(h) for h in hs]
    nl = args.num_layers
    if nl and nl != len(hs):
        hs = hs * nl if len(hs) == 1 else hs[:nl]
    args.hidden_size = hs[0] if len(set(hs)) == 1 else hs
    args.num_layers = len(hs)

    model = NeuralNetwork(args)
    opt = get_optimizer(args.optimizer,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay)

    use_wandb = False
    if args.wandb_project and not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
            )
            use_wandb = True
        except Exception as e:
            print(f'[W&B] Could not initialise: {e}')

    from sklearn.metrics import f1_score as skf1

    best_val_f1  = -1.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(len(x_train))
        xtr, ytr = x_train[idx], y_train[idx]

        total_loss = 0.0
        n_batches  = 0

        for xb, yb in get_batches(xtr, ytr, batch_size=args.batch_size, shuffle=False):
            logits     = model.forward(xb)
            loss, _    = model.loss_fn(logits, yb)
            total_loss += loss
            n_batches  += 1

            model.backward(logits, yb)
            opt.update(model.layers)

        avg_loss  = total_loss / max(n_batches, 1)
        val_preds = model.predict(x_val)
        val_acc   = float(np.mean(val_preds == y_val))
        val_f1    = float(skf1(y_val, val_preds, average='weighted', zero_division=0))

        print(f'Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}')

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'train/loss': avg_loss,
                       'val/accuracy': val_acc, 'val/f1': val_f1})

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = model.get_weights()

    model.set_weights(best_weights)

    test_preds = model.predict(x_test)
    test_acc   = float(np.mean(test_preds == y_test))
    test_f1    = float(skf1(y_test, test_preds, average='weighted', zero_division=0))
    print(f'\nTest acc={test_acc:.4f} | Test F1={test_f1:.4f}')

    if use_wandb:
        import wandb
        wandb.log({'test/accuracy': test_acc, 'test/f1': test_f1})
        wandb.finish()

    model_path = args.model_path or os.path.join(_THIS_DIR, 'best_model.npy')
    np.save(model_path, best_weights)

    cfg = {
        'dataset':       args.dataset,
        'hidden_size':   args.hidden_size,
        'num_layers':    args.num_layers,
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
    config_path = os.path.join(_THIS_DIR, 'best_config.json')
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f'Saved model  → {model_path}')
    print(f'Saved config → {config_path}')
    return model

if __name__ == '__main__':
    train(parse_arguments())