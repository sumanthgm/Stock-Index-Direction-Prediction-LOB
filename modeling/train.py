"""
Training loop, evaluation, and experiment runner.

Quick start in Colab:
    from fi2010_dataset import load_fi2010, make_loader, class_weights
    from models import build_model
    from train import train, evaluate, run_all_experiments

    # Single experiment
    train_ds, val_ds, test_ds = load_fi2010("fi2010/.../NoAuction_Zscore", stock_idx=1, horizon=10)
    model = build_model("deeplob")
    history = train(model, train_ds, val_ds)
    results = evaluate(model, test_ds)
    print(results)   # {"f1": 0.72, "acc": 0.74, "loss": 0.81}

    # Full experiment matrix: all models × all stocks × all horizons
    df = run_all_experiments(data_dir="fi2010/.../NoAuction_Zscore")
    df.to_csv("results.csv", index=False)
"""

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from fi2010_dataset import LOBDataset, class_weights, load_fi2010, make_loader
from models import build_model

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
HORIZONS = [10, 20, 50]  # FI-2010 label columns available: 10, 20, 30, 50, 100


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    model: nn.Module,
    train_ds: LOBDataset,
    val_ds: LOBDataset,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    weight_loss: bool = True,
    criterion: nn.Module | None = None,
    sampler=None,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Train a model with early stopping on validation weighted-F1.

    Optional overrides (used by the HPO and class-imbalance studies; defaults
    reproduce the original behaviour exactly):
        criterion     custom loss module (e.g. focal / label-smoothing); when None,
                      a (weighted) CrossEntropyLoss is built from ``weight_loss``.
        sampler       a torch Sampler for the train loader (e.g. class-balanced
                      WeightedRandomSampler); when given, shuffle is disabled.
        weight_decay  AdamW weight decay (tunable in HPO).

    Returns:
        history dict with keys: train_loss, val_loss, val_f1, best_epoch
    """
    model = model.to(DEVICE)
    if sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )
    else:
        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False)

    if criterion is None:
        weights = class_weights(train_ds).to(DEVICE) if weight_loss else None
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = criterion.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "best_epoch": 0}
    best_val_f1 = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        total_loss, n_batches = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = total_loss / n_batches

        # --- validate ---
        val_loss, val_f1 = _eval_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  epoch {epoch:3d} | train_loss {train_loss:.4f} "
                f"| val_loss {val_loss:.4f} | val_f1 {val_f1:.4f}"
                + (" ← best" if no_improve == 0 else "")
            )

        if no_improve >= patience:
            if verbose:
                print(
                    f"  Early stop at epoch {epoch} (best epoch {history['best_epoch']})."
                )
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, test_ds: LOBDataset, batch_size: int = 256) -> dict:
    """
    Evaluate on a dataset. Returns weighted F1, accuracy, and loss.
    """
    model = model.to(DEVICE)
    loader = make_loader(test_ds, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    loss, f1 = _eval_epoch(model, loader, criterion)
    all_preds, all_labels = _collect_preds(model, loader)
    acc = (all_preds == all_labels).mean()
    return {"f1": round(f1, 4), "acc": round(float(acc), 4), "loss": round(loss, 4)}


def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            n_batches += 1
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / n_batches, float(f1)


def _collect_preds(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            all_preds.append(model(X).argmax(dim=1).cpu())
            all_labels.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------
def run_all_experiments(
    data_dir: str | Path,
    models: list[str] = ("deeplob", "mlplob", "mambalob"),
    folds: list[int] = (7,),
    horizons: list[int] = None,
    epochs: int = 50,
    batch_size: int = 64,
    checkpoint_dir: str = "checkpoints",
    results_path: str = "results/results.csv",
) -> pd.DataFrame:
    """
    Run all combinations of model × fold × horizon on FI-2010.
    Saves a checkpoint for each run and writes results incrementally to CSV
    so partial results survive interruptions.

    Args:
        data_dir:    Path to the NoAuction_Zscore folder.
        models:      Model names to run: "deeplob", "mlplob", "mambalob".
        folds:       Cross-fold numbers. Default [7] (standard 7-train/3-test split).
                     Use multiple folds (e.g. [5,6,7,8]) for cross-validated results.
        horizons:    Prediction horizons in LOB events. Default [10, 20, 50].
        epochs:      Max training epochs (early stopping usually kicks in earlier).
        batch_size:  Training batch size.
        checkpoint_dir: Directory for .pt checkpoint files.
        results_path:   CSV file for results (appended incrementally).

    Returns:
        DataFrame with one row per (model, fold, horizon) experiment.
    """
    if horizons is None:
        horizons = HORIZONS

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name in models:
        for fold in folds:
            for horizon in horizons:
                tag = f"{model_name}_cf{fold}_h{horizon}"
                print(f"\n{'=' * 60}")
                print(f"  {tag}")
                print(f"{'=' * 60}")

                train_ds, val_ds, test_ds = load_fi2010(
                    data_dir, fold=fold, horizon=horizon
                )

                model = build_model(model_name)
                t0 = time.time()
                history = train(
                    model,
                    train_ds,
                    val_ds,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=True,
                )
                elapsed = time.time() - t0

                test_results = evaluate(model, test_ds)
                save_checkpoint(model, f"{checkpoint_dir}/{tag}.pt")

                row = {
                    "model": model_name,
                    "fold": fold,
                    "horizon": horizon,
                    "test_f1": test_results["f1"],
                    "test_acc": test_results["acc"],
                    "test_loss": test_results["loss"],
                    "best_epoch": history["best_epoch"],
                    "train_time_s": round(elapsed, 1),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(results_path, index=False)
                print(
                    f"  → test_f1={test_results['f1']:.4f}  "
                    f"acc={test_results['acc']:.4f}  "
                    f"time={elapsed:.0f}s"
                )

    df = pd.DataFrame(rows)
    print(f"\nAll done. Results saved to {results_path}")
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    print("\n--- Results Summary (weighted F1) ---")
    pivot = df.pivot_table(
        index="model", columns="horizon", values="test_f1", aggfunc="mean"
    )
    print(pivot.round(4).to_string())
