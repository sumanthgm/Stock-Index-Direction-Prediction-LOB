"""
Optuna hyperparameter search for one model — a resumable, distributable worker.

Resumability + dual-GPU come for free from Optuna's RDB storage: several workers
(one per GPU, each launched with CUDA_VISIBLE_DEVICES set) call optimize() on the
SAME study_name + SQLite storage and coordinate through it, so trials are shared
and a killed session resumes from the DB. The notebook syncs the .db file to S3.

    CUDA_VISIBLE_DEVICES=0 python run_hpo.py --model convmambalob --symbol NIFTY \
        --horizon 100 --feature-set all --storage sqlite:///hpo/convmambalob.db \
        --study-name convmambalob_NIFTY_h100 --n-trials 10 --epochs 12 --data-dir nse_data/dhan

The objective maximises validation weighted-F1 (early-stopped). The search space
branches by model family (Mamba: d_model/d_state/n_layers; TLOB: hidden_dim/
num_layers/num_heads); lr, batch_size and weight_decay are common.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Load the data once per worker (not per trial) and cache it.
_DATA = {}


def _datasets(symbol, horizon, feature_set, data_dir):
    key = (symbol, horizon, feature_set)
    if key not in _DATA:
        from nse_dataset import load_nse

        tr, vl, te = load_nse(
            symbol=symbol,
            horizon=horizon,
            seq_len=100,
            feature_set=feature_set,
            label_scheme="A",
            data_dir=data_dir,
        )
        _DATA[key] = (tr, vl, te)
    return _DATA[key]


def make_objective(args):
    from models import build_model
    from train import train

    tr, vl, te = _datasets(args.symbol, args.horizon, args.feature_set, args.data_dir)
    n_features = tr[0][0].shape[-1]
    is_tlob = args.model.lower() == "tlob"

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        if is_tlob:
            kwargs = {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                "num_layers": trial.suggest_categorical("num_layers", [2, 4]),
                "num_heads": trial.suggest_categorical("num_heads", [1, 2]),
            }
        else:
            kwargs = {
                "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
                "d_state": trial.suggest_categorical("d_state", [8, 16, 32]),
                "n_layers": trial.suggest_categorical("n_layers", [1, 2, 3]),
            }
        net = build_model(args.model, seq_len=100, n_features=n_features, **kwargs)
        hist = train(
            net,
            tr,
            vl,
            epochs=args.epochs,
            patience=3,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            verbose=False,
        )
        return max(hist["val_f1"])

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--feature-set", default="all")
    ap.add_argument("--storage", required=True)  # e.g. sqlite:///hpo/convmambalob.db
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--n-trials", type=int, default=10)  # PER worker
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import optuna
    from stats import set_seed
    from train import DEVICE

    set_seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(
        f"[hpo {args.study_name}] device={DEVICE} CUDA_VISIBLE_DEVICES={cvd} "
        f"existing trials={len(study.trials)} | running {args.n_trials} more"
    )
    objective = make_objective(args)
    study.optimize(objective, n_trials=args.n_trials)
    print(f"[hpo {args.study_name}] best val_f1={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    main()
