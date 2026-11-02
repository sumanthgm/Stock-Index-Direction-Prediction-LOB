"""
Load collected NSE LOB data from S3 into a pandas DataFrame.
Reads Parquet files directly from S3 — no download step needed.

Setup:
    pip install pandas pyarrow s3fs

AWS credentials (needed in Colab — not on the EC2 VM which uses an IAM role):
    Option A — Colab Secrets (recommended):
        1. Left panel → key icon → add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        2. In your notebook before importing this module:
               from google.colab import userdata
               import os
               os.environ["AWS_ACCESS_KEY_ID"]     = userdata.get("AWS_ACCESS_KEY_ID")
               os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")

    Option B — env vars directly (less safe, don't commit):
               import os
               os.environ["AWS_ACCESS_KEY_ID"]     = "AKIA..."
               os.environ["AWS_SECRET_ACCESS_KEY"] = "..."

    The IAM user only needs s3:GetObject and s3:ListBucket on the bucket.

Usage in Colab:
    from load_nse_data import load_lob, to_lob_tensor

    df = load_lob("dhan")                                    # all available dates
    df = load_lob("dhan", start="20260507", end="20260510")  # date range
    df = load_lob("dhan", symbols=["HDFCBANK", "INFY"])      # filter symbols

    X = to_lob_tensor(df, levels=10, seq_len=100)            # (N, 100, 40) numpy array
"""

from datetime import timedelta, timezone

import numpy as np
import pandas as pd

BUCKET_NAME = "YOUR_BUCKET_NAME"  # same as in sync_to_s3.sh
IST = timezone(timedelta(hours=5, minutes=30))


def _s3_parquet_paths(source: str, start: str | None, end: str | None) -> list[str]:
    import s3fs

    fs = s3fs.S3FileSystem()  # picks up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from env
    prefix = f"{BUCKET_NAME}/lob-data/{source}/"
    try:
        all_files = fs.ls(prefix)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"No files found at s3://{prefix}\nCheck BUCKET_NAME and that sync_to_s3.sh has run at least once."
        ) from err

    paths = []
    for f in sorted(all_files):
        fname = f.split("/")[-1]
        if not fname.endswith(".parquet"):
            continue
        # filename: lob_dhan_YYYYMMDD.parquet or lob_kite_YYYYMMDD.parquet
        date_str = fname.rsplit("_", 1)[-1].replace(".parquet", "")
        if start and date_str < start:
            continue
        if end and date_str > end:
            continue
        paths.append(f"s3://{f}")

    if not paths:
        raise FileNotFoundError(f"No Parquet files for source='{source}', range {start}–{end}.")
    return paths


def load_lob(
    source: str = "dhan",
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load LOB snapshots from S3 into a DataFrame.

    Args:
        source:  "dhan" (20-level) or "kite" (5-level)
        start:   date string YYYYMMDD inclusive, e.g. "20260507"
        end:     date string YYYYMMDD inclusive
        symbols: filter to these symbols, e.g. ["HDFCBANK", "INFY"]
        columns: load only these columns (faster when you need a subset)

    Returns:
        DataFrame sorted by timestamp (IST-aware).
    """
    paths = _s3_parquet_paths(source, start, end)
    df = pd.read_parquet(paths, columns=columns)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if symbols:
        df = df[df["symbol"].isin(symbols)]

    return df


def to_lob_tensor(
    df: pd.DataFrame,
    levels: int = 10,
    seq_len: int = 100,
) -> np.ndarray:
    """
    Convert a single-symbol DataFrame to a (N, seq_len, 4*levels) float32 array
    in DeepLOB input order: [ask_price_i, ask_qty_i, bid_price_i, bid_qty_i] × levels.

    Args:
        df:      Single-symbol DataFrame from load_lob().
        levels:  Number of LOB levels to use (max 5 for Kite, 20 for Dhan).
        seq_len: Sliding window length (100 is standard for DeepLOB / Mamba-LOB).

    Returns:
        numpy array of shape (n_windows, seq_len, 4 * levels).

    Raises:
        ValueError if required columns are missing or df has fewer than seq_len rows.
    """
    feature_cols = []
    for i in range(1, levels + 1):
        feature_cols += [
            f"ask_price_{i}",
            f"ask_qty_{i}",
            f"bid_price_{i}",
            f"bid_qty_{i}",
        ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Reduce `levels` or check source.")

    data = df[feature_cols].to_numpy(dtype="float32")
    n_windows = len(data) - seq_len + 1
    if n_windows <= 0:
        raise ValueError(f"DataFrame has {len(data)} rows, need at least {seq_len} for seq_len={seq_len}.")

    idx = np.arange(seq_len)[None, :] + np.arange(n_windows)[:, None]
    return data[idx]  # (n_windows, seq_len, 4*levels)
