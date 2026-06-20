"""
FI-2010 Dataset — loading, windowing, and PyTorch Dataset classes.

FI-2010 contains 10 days of LOB data for 5 Finnish stocks on NASDAQ Nordic Helsinki:
    KESKO, OUT1, SAMPO, RAUTARUUKKI, WRT1V

All 5 stocks are concatenated into one file per fold.

File layout (after unzip):
    BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/
        NoAuction_Zscore_Training/
            Train_Dst_NoAuction_ZScore_CF_1.txt   ← 1 training day
            Train_Dst_NoAuction_ZScore_CF_2.txt   ← 2 training days
            ...
            Train_Dst_NoAuction_ZScore_CF_7.txt   ← 7 training days (standard)
            Train_Dst_NoAuction_ZScore_CF_9.txt   ← 9 training days
        NoAuction_Zscore_Testing/
            Test_Dst_NoAuction_ZScore_CF_7.txt    ← remaining 3 days (standard)
            ...

File format (verified):
    - Shape on disk: (149, T) — rows=features, columns=timesteps → MUST TRANSPOSE
    - After transpose: (T, 149)
      - Cols 0–39:   LOB features (40 cols = 10 levels × [ask_price, ask_vol, bid_price, bid_vol])
      - Cols 40–143: Additional derived features (not used)
      - Cols 144–148: Labels for k=10, 20, 30, 50, 100
    - Label encoding: 1=Down, 2=Stationary, 3=Up → remapped to 0, 1, 2
    - Zscore variant is pre-normalised (mean=0, std=1) — do NOT normalise again

Class balance (CF_7, k=10):
    Down: 19.7%  |  Stationary: 60.5%  |  Up: 19.7%
    → Use weighted CrossEntropyLoss to handle imbalance.

Standard split used by Zhang 2019 (DeepLOB) and most papers:
    fold=7 → 254,750 train timesteps / 55,478 test timesteps (7 + 3 days)

Usage:
    from fi2010_dataset import load_fi2010, LOBDataset, make_loader, class_weights

    train_ds, val_ds, test_ds = load_fi2010(
        data_dir="BenchmarkDatasets/NoAuction/1.NoAuction_Zscore",
        fold=7,        # CF_7: 7 train days + 3 test days (standard)
        horizon=10,    # prediction horizon: 10, 20, 30, 50, or 100 LOB events
        seq_len=100,   # sliding window length
    )
    loader = make_loader(train_ds, batch_size=64, shuffle=True)
    X, y = next(iter(loader))   # X: (64, 100, 40)  y: (64,) in {0,1,2}
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Label column offset from the right end (after transpose)
# Last 5 columns = labels for k=10, 20, 30, 50, 100
_HORIZON_TO_COL = {10: -5, 20: -4, 30: -3, 50: -2, 100: -1}

N_FEATURES = 40  # first 40 columns after transpose


def _load_fold(data_dir: Path, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load train and test files for a given cross-fold.
    Returns (train_data, test_data), each shape (T, 149) — already transposed.
    """
    # Detect normalisation variant from directory name
    dir_name = data_dir.name  # e.g. "1.NoAuction_Zscore"

    # Build file paths — handle both capitalisation variants seen in the wild
    train_dir = data_dir / "NoAuction_Zscore_Training"
    test_dir = data_dir / "NoAuction_Zscore_Testing"

    if not train_dir.exists():
        # Try case-insensitive search
        for child in data_dir.iterdir():
            if "training" in child.name.lower():
                train_dir = child
            if "testing" in child.name.lower():
                test_dir = child

    # File names vary slightly by source — find CF_{fold} file
    def _find(directory: Path, fold: int) -> Path:
        candidates = sorted(directory.glob(f"*CF_{fold}.txt"))
        if not candidates:
            raise FileNotFoundError(
                f"No file matching '*CF_{fold}.txt' in {directory}.\n"
                f"Available: {sorted(f.name for f in directory.glob('*.txt'))}"
            )
        return candidates[0]

    train_path = _find(train_dir, fold)
    test_path = _find(test_dir, fold)

    # NPZ cache: first load parses text (~60s), subsequent loads are ~1s
    # Fall back to cwd if data_dir is read-only (e.g. /kaggle/input)
    cache_path = data_dir / f".cache_cf{fold}.npz"
    if not cache_path.exists():
        alt = Path(f".cache_cf{fold}.npz")
        if alt.exists():
            cache_path = alt
    if cache_path.exists():
        print(f"  Loading from cache: {cache_path}")
        d = np.load(cache_path)
        return d["train"], d["test"]

    print(f"  Train: {train_path.name}  ({_file_shape(train_path)})")
    print(f"  Test:  {test_path.name}  ({_file_shape(test_path)})")
    print("  Parsing text files (one-time, ~60s) ...")

    # Load and TRANSPOSE: (149, T) → (T, 149)
    train_data = pd.read_csv(train_path, sep=r"\s+", header=None).values.T.astype(
        np.float32
    )
    test_data = pd.read_csv(test_path, sep=r"\s+", header=None).values.T.astype(
        np.float32
    )

    try:
        np.savez_compressed(cache_path, train=train_data, test=test_data)
        print(f"  Cache saved → {cache_path} (future loads will be ~1s)")
    except OSError:
        # Read-only filesystem (Kaggle input dir) — save next to cwd instead
        cache_path = Path(f".cache_cf{fold}.npz")
        np.savez_compressed(cache_path, train=train_data, test=test_data)
        print(f"  Cache saved → {cache_path} (future loads will be ~1s)")
    return train_data, test_data


def _file_shape(path: Path) -> str:
    """Quick shape read without full load."""
    with open(path) as f:
        first_line = f.readline().split()
        ncols = len(first_line)
    with open(path) as f:
        nrows = sum(1 for _ in f)
    return f"{nrows}×{ncols} on disk → {ncols}×{nrows} after transpose"


def _make_windows(
    features: np.ndarray, labels: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window segmentation.
    features: (T, 40)  labels: (T,)
    Returns X: (N, seq_len, 40)  y: (N,) — label at the LAST timestep of each window.
    """
    T = len(features)
    n_windows = T - seq_len + 1
    if n_windows <= 0:
        raise ValueError(
            f"Only {T} timesteps but seq_len={seq_len}. Use a smaller seq_len."
        )
    idx = np.arange(seq_len)[None, :] + np.arange(n_windows)[:, None]  # (N, seq_len)
    X = features[idx]  # (N, seq_len, 40)
    y = labels[idx[:, -1]]  # (N,)  label at last step
    return X, y


class LOBDataset(Dataset):
    """Holds pre-windowed (N, seq_len, F) tensors. Fine for small splits, but
    materializing all windows OOMs on FI-2010 — prefer WindowedLOBDataset there."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class WindowedLOBDataset(Dataset):
    """Memory-efficient sliding windows: stores the (T, F) matrix once and slices
    each window on the fly in __getitem__. Avoids the dense (N, seq_len, F)
    expansion that OOMs on FI-2010 (~230k windows × 100 × 40, doubled by the
    numpy->torch copy → several GB per split).

    Keeps only window end-positions with a valid label (>= 0), matching the eager
    _make_windows behaviour. If `seg` is given (a per-row segment id, e.g. trading
    day / contract), windows that straddle a segment boundary are dropped — same
    contract as the NSE front-month stitching.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
        seg: np.ndarray | None = None,
    ):
        self.features = torch.as_tensor(
            np.ascontiguousarray(features), dtype=torch.float32
        )  # (T, F)
        self.seq_len = seq_len
        labels = np.asarray(labels)
        T = len(self.features)
        if T < seq_len:
            raise ValueError(f"Only {T} rows but seq_len={seq_len}.")
        ends = np.arange(seq_len - 1, T)
        valid = labels[ends] >= 0  # last-step label must be valid
        if seg is not None:
            seg = np.asarray(seg)
            valid = valid & (seg[ends - seq_len + 1] == seg[ends])  # intra-segment only
        ends = ends[valid]
        self.ends = ends
        self.y = torch.as_tensor(labels[ends], dtype=torch.long)  # per-window labels

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i):
        e = int(self.ends[i])
        return self.features[e - self.seq_len + 1 : e + 1], self.y[i]


def load_fi2010(
    data_dir: str | Path,
    fold: int = 7,
    horizon: int = 10,
    seq_len: int = 100,
    val_fraction: float = 0.1,
) -> tuple[LOBDataset, LOBDataset, LOBDataset]:
    """
    Load FI-2010 for one cross-fold and one prediction horizon.

    Args:
        data_dir:     Path to the NoAuction_Zscore folder.
        fold:         Cross-fold number 1–9. fold=7 is the standard (7 train + 3 test days).
        horizon:      Prediction horizon in LOB events: 10, 20, 30, 50, or 100.
        seq_len:      Sliding window length. Default 100 (as in DeepLOB).
        val_fraction: Fraction of training timesteps held out for validation.

    Returns:
        (train_dataset, val_dataset, test_dataset) as LOBDataset instances.
    """
    if horizon not in _HORIZON_TO_COL:
        raise ValueError(
            f"horizon must be one of {list(_HORIZON_TO_COL)}; got {horizon}."
        )

    data_dir = Path(data_dir)
    print(
        f"\nLoading FI-2010 | fold=CF_{fold} | horizon=k{horizon} | seq_len={seq_len}"
    )

    train_data, test_data = _load_fold(data_dir, fold)

    # Extract features (cols 0–39) and labels for chosen horizon
    horizon_col = _HORIZON_TO_COL[horizon]
    train_feat = train_data[:, :N_FEATURES]  # (T_train, 40)
    test_feat = test_data[:, :N_FEATURES]  # (T_test,  40)

    # Labels: {1,2,3} → {0,1,2}
    train_labels = train_data[:, horizon_col].astype(np.int64) - 1
    test_labels = test_data[:, horizon_col].astype(np.int64) - 1

    # Validation split: hold out last val_fraction of training timesteps
    val_start = int(len(train_feat) * (1 - val_fraction))
    val_feat, val_labels = train_feat[val_start:], train_labels[val_start:]
    train_feat, train_labels = train_feat[:val_start], train_labels[:val_start]

    # Lazy sliding windows (dense materialization OOMs on Colab for FI-2010)
    train_ds = WindowedLOBDataset(train_feat, train_labels, seq_len)
    val_ds = WindowedLOBDataset(val_feat, val_labels, seq_len)
    test_ds = WindowedLOBDataset(test_feat, test_labels, seq_len)

    print(
        f"  Splits → train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}"
    )
    _print_dist("  train labels", train_ds.y.numpy())
    _print_dist("  test  labels", test_ds.y.numpy())

    return train_ds, val_ds, test_ds


def make_loader(
    ds: LOBDataset, batch_size: int = 64, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False
    )


def class_weights(ds: LOBDataset) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    counts = torch.bincount(ds.y, minlength=3).float()
    w = 1.0 / (counts + 1e-8)
    return w / w.sum() * 3  # scale so weights sum to n_classes


def _print_dist(label: str, y: np.ndarray):
    names = ["Down", "Stat", "Up"]
    counts = np.bincount(y, minlength=3)
    pcts = counts / counts.sum() * 100
    dist = "  ".join(f"{n}:{p:.1f}%" for n, p in zip(names, pcts))
    print(f"  {label}: {dist}")
