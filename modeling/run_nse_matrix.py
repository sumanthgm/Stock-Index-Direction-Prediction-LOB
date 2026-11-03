"""
NSE futures full experiment matrix.

Runs every combination of:
    models    = [deeplob, mlplob, mambalob]
    symbols   = [NIFTY-MAY-FUT, BANKNIFTY-MAY-FUT]
    horizons  = [10, 20]

Total: 12 runs. Each (model, symbol, horizon) tuple gets:
    - load_nse with alpha=1e-5, seq_len=100
    - train with patience=3 (DeepLOB overfits within ~1 epoch on NSE)
    - evaluate on 2 OOS days (May 25, 26)
    - log weighted_f1, macro_f1, accuracy, baselines, best epoch, time
    - save checkpoint
    - append row to results CSV after each run (resilient to interruption)

Run:
    cd Modeling
    nohup python run_nse_matrix.py > nse_matrix.log 2>&1 &
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from models import build_model
from nse_dataset import load_nse
from sklearn.metrics import f1_score
from train import DEVICE, make_loader, save_checkpoint, train

# Root symbols → stitched front-month series (May-FUT + Jun-FUT). The Colab
# notebook notebooks/nse_matrix.ipynb is the primary runner; this script mirrors it.
MODELS = ["deeplob", "mlplob", "tlob", "mambalob"]
SYMBOLS = ["NIFTY", "BANKNIFTY"]
HORIZONS = [10, 20, 50, 100]

EPOCHS = 20
PATIENCE = 3
BATCH_SIZE = 64

RESULTS_CSV = Path("results/nse_results.csv")
CHECKPOINT_DIR = Path("checkpoints/nse")


def baselines(y: np.ndarray) -> dict:
    """Constant-class and random baselines for context."""
    classes, counts = np.unique(y, return_counts=True)
    maj = int(classes[counts.argmax()])
    pred_maj = np.full_like(y, maj)
    pred_stat = np.ones_like(y)
    rng = np.random.RandomState(0)
    pred_rand = rng.randint(0, 3, len(y))
    return {
        "baseline_majority_wf1": round(f1_score(y, pred_maj, average="weighted", zero_division=0), 4),
        "baseline_stat_wf1": round(f1_score(y, pred_stat, average="weighted", zero_division=0), 4),
        "baseline_random_wf1": round(f1_score(y, pred_rand, average="weighted", zero_division=0), 4),
    }


def eval_full(model, test_ds) -> dict:
    """Weighted + macro F1 + accuracy from a fresh forward pass."""
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            preds.append(model(X).argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "test_wf1": round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
        "test_mf1": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "test_acc": round(float((preds == labels).mean()), 4),
    }


def run_one(model_name: str, symbol: str, horizon: int) -> dict:
    tag = f"{model_name}_{symbol.replace('-', '_')}_h{horizon}"
    print(f"\n{'=' * 70}\n  {tag}\n{'=' * 70}")

    tr, vl, te = load_nse(symbol=symbol, horizon=horizon)

    model = build_model(model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    t0 = time.time()
    hist = train(
        model,
        tr,
        vl,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        verbose=True,
    )
    elapsed = time.time() - t0

    bl = baselines(te.y.numpy())
    res = eval_full(model, te)
    save_checkpoint(model, str(CHECKPOINT_DIR / f"{tag}.pt"))

    row = {
        "model": model_name,
        "symbol": symbol,
        "horizon": horizon,
        "n_params": n_params,
        "best_epoch": hist["best_epoch"],
        "epochs_run": len(hist["val_f1"]),
        "best_val_f1": round(max(hist["val_f1"]), 4),
        **res,
        **bl,
        "train_time_s": round(elapsed, 1),
        "train_size": len(tr),
        "val_size": len(vl),
        "test_size": len(te),
    }
    print(
        f"  → test_wf1={res['test_wf1']}  mf1={res['test_mf1']}  acc={res['test_acc']}  "
        f"random_wf1={bl['baseline_random_wf1']}  best_ep={hist['best_epoch']}  time={elapsed:.0f}s"
    )
    return row


def main():
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support: skip combinations already in the CSV
    done = set()
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        done = {(r.model, r.symbol, r.horizon) for r in old.itertuples()}
        print(f"Resuming: {len(done)} runs already in {RESULTS_CSV}")

    rows = []
    for m in MODELS:
        for sym in SYMBOLS:
            for h in HORIZONS:
                if (m, sym, h) in done:
                    print(f"  skip {m} {sym} h={h} (already done)")
                    continue
                try:
                    row = run_one(m, sym, h)
                except Exception as e:
                    row = {
                        "model": m,
                        "symbol": sym,
                        "horizon": h,
                        "error": repr(e),
                    }
                    print(f"  FAILED: {e!r}")
                rows.append(row)
                # Incremental write: combine prior + new
                df_new = pd.DataFrame(rows)
                if RESULTS_CSV.exists():
                    df_prev = pd.read_csv(RESULTS_CSV)
                    df_out = pd.concat([df_prev, df_new], ignore_index=True)
                else:
                    df_out = df_new
                df_out.to_csv(RESULTS_CSV, index=False)

    print(f"\nAll done. Results: {RESULTS_CSV}")
    final = pd.read_csv(RESULTS_CSV)
    pivot = final.pivot_table(
        index=["model"],
        columns=["symbol", "horizon"],
        values="test_wf1",
        aggfunc="mean",
    )
    print("\nWeighted F1 summary:")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
