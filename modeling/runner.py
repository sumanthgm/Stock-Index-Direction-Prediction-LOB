"""
Unified NSE experiment runner (importable, so subprocesses / the dual-GPU launcher can use it).

`run_one(spec, ...)` loads the requested NSE data (feature_set + label_scheme), builds the model with
`n_features` taken from the data (so any feature set works), trains, evaluates, saves a checkpoint, and
appends a result row to a CSV. It uses `train.DEVICE`, which resolves to whatever GPU is visible — so a
subprocess launched with `CUDA_VISIBLE_DEVICES=k` trains on physical GPU k with no code change.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from fi2010_dataset import make_loader
from models import build_model
from nse_dataset import load_nse
from stats import set_seed
from train import DEVICE, save_checkpoint, train

MODEL_LR = {
    "deeplob": 1e-3,
    "mlplob": 1e-3,
    "tlob": 1e-4,
    "mambalob": 3e-4,
    "bimambalob": 3e-4,
    "convmambalob": 3e-4,
    "biconvmambalob": 3e-4,
}
CLASS_NAMES = ["Down", "Stat", "Up"]


def collect_preds(model, ds, batch_size=256):
    model = model.to(DEVICE).eval()
    loader = make_loader(ds, batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(DEVICE)).argmax(1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def full_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def naive_baselines(y_train, y_test):
    maj = int(np.bincount(y_train, minlength=3).argmax())
    wf1 = lambda yp: f1_score(y_test, yp, average="weighted", zero_division=0)  # noqa: E731
    rng = np.random.default_rng(0)
    probs = np.bincount(y_train, minlength=3) / len(y_train)
    return {
        "baseline_majority_wf1": round(wf1(np.full_like(y_test, maj)), 4),
        "baseline_stat_wf1": round(wf1(np.full_like(y_test, 1)), 4),
        "baseline_random_wf1": round(wf1(rng.choice(3, size=len(y_test), p=probs)), 4),
    }


def spec_tag(spec: dict) -> str:
    sch = (
        ""
        if spec.get("label_scheme", "A").upper() == "A"
        else f"_{spec['label_scheme'].upper()}"
    )
    # strategy suffix only for the class-imbalance study (keeps default tags unchanged)
    strat = spec.get("strategy")
    strat_sfx = f"_{strat}" if strat and strat != "weighted_ce" else ""
    return (
        f"{spec['model']}_{spec['symbol']}_{spec.get('feature_set', 'base40')}"
        f"_h{spec['horizon']}{sch}_s{spec.get('seed', 0)}{strat_sfx}"
    )


def _strategy_train_kwargs(strategy: str | None, train_ds) -> dict:
    """Map a class-imbalance strategy name to train() kwargs. None/weighted_ce = default."""
    if not strategy or strategy == "weighted_ce":
        return {"weight_loss": True}
    if strategy == "plain_ce":
        return {"weight_loss": False}
    if strategy == "label_smoothing":
        import torch.nn as nn

        from fi2010_dataset import class_weights

        return {
            "criterion": nn.CrossEntropyLoss(
                weight=class_weights(train_ds), label_smoothing=0.1
            )
        }
    if strategy == "focal":
        from fi2010_dataset import class_weights

        from losses import FocalLoss

        return {"criterion": FocalLoss(gamma=2.0, alpha=class_weights(train_ds))}
    if strategy == "balanced_sampling":
        from torch.utils.data import WeightedRandomSampler

        from fi2010_dataset import class_weights

        per_sample = class_weights(train_ds)[train_ds.y].double()
        sampler = WeightedRandomSampler(
            per_sample, num_samples=len(train_ds.y), replacement=True
        )
        return {"sampler": sampler, "weight_loss": False}
    raise ValueError(f"unknown strategy '{strategy}'")


def run_one(
    spec: dict,
    data_dir: str,
    out_csv: str,
    ckpt_dir: str = "checkpoints",
    area: str | None = None,
    s3=None,
    epochs: int = 20,
    patience: int = 3,
    batch_size: int = 128,
    seq_len: int = 100,
) -> dict:
    """Train+evaluate one run spec; append row to out_csv; checkpoint -> ckpt_dir (+S3 if given)."""
    model = spec["model"]
    symbol = spec["symbol"]
    horizon = int(spec["horizon"])
    feature_set = spec.get("feature_set", "base40")
    label_scheme = spec.get("label_scheme", "A")
    seed = int(spec.get("seed", 0))
    set_seed(seed)
    tag = spec_tag(spec)
    print("=" * 76, f"\n  {tag}  | device={DEVICE}\n", "=" * 76)

    tr, vl, te = load_nse(
        symbol=symbol,
        horizon=horizon,
        seq_len=seq_len,
        feature_set=feature_set,
        label_scheme=label_scheme,
        data_dir=data_dir,
    )
    n_features = tr[0][0].shape[-1]
    net = build_model(model, seq_len=seq_len, n_features=n_features)
    lr = MODEL_LR.get(model, 1e-3)
    strategy = spec.get("strategy")
    strat_kwargs = _strategy_train_kwargs(strategy, tr)
    t0 = time.time()
    hist = train(
        net,
        tr,
        vl,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
        **strat_kwargs,
    )
    elapsed = time.time() - t0
    y_pred, y_true = collect_preds(net, te)
    mt = full_metrics(y_true, y_pred)
    bl = naive_baselines(tr.y.numpy(), y_true)

    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt = pathlib.Path(ckpt_dir) / f"{tag}.pt"
    save_checkpoint(net, str(ckpt))

    conf = mt["confusion"]
    recalls = [
        round(conf[c][c] / max(sum(conf[c]), 1), 4) for c in range(3)
    ]  # per-class recall: Down, Stat, Up
    row = {
        "model": model,
        "symbol": symbol,
        "feature_set": feature_set,
        "label_scheme": label_scheme.upper(),
        "horizon": horizon,
        "seed": seed,
        "strategy": strategy or "weighted_ce",
        "n_features": n_features,
        "n_params": sum(p.numel() for p in net.parameters()),
        "best_epoch": hist["best_epoch"],
        "epochs_run": len(hist["val_f1"]),
        "best_val_f1": round(max(hist["val_f1"]), 4),
        "test_accuracy": round(mt["accuracy"], 4),
        "test_macro_f1": round(mt["macro_f1"], 4),
        "test_weighted_f1": round(mt["weighted_f1"], 4),
        "test_mcc": round(mt["mcc"], 4),
        "recall_down": recalls[0],
        "recall_stat": recalls[1],
        "recall_up": recalls[2],
        **bl,
        "train_time_s": round(elapsed, 1),
    }
    out_csv = pathlib.Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(out_csv, index=False)
    metrics_json = out_csv.parent / f"{tag}_metrics.json"
    metrics_json.write_text(json.dumps(mt, indent=2))

    if s3 is not None and area is not None:
        from nbenv import s3_put_area

        s3_put_area(s3, ckpt, area)
        s3_put_area(s3, metrics_json, area)
        # Incremental results sync: push this shard's CSV after EVERY run so a
        # mid-session kill (Kaggle timeout) still leaves finished runs on S3 to
        # skip on resume — not just after the final merge cell.
        s3_put_area(s3, out_csv, area)
    print(
        f"  -> wF1={mt['weighted_f1']:.4f} mF1={mt['macro_f1']:.4f} (F={n_features}, {elapsed:.0f}s)"
    )
    return row
