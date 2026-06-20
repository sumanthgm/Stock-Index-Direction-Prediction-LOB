"""
NSE Index Futures Dataset — adapter to the FI-2010 modeling pipeline.

Loads Dhan 20-level LOB Parquet files (one per trading day) for NIFTY or
BANKNIFTY index futures, applies outlier filtering, reorders features to
FI-2010 layout, computes alpha=1e-5 labels at a chosen horizon, and returns
PyTorch Dataset objects compatible with `models.py` and `train.py`.

Continuous front-month series (May + June 2026, 21 trading days):
    The May-FUT contract expired 2026-05-26 and collection rolled to Jun-FUT
    on 2026-06-01. Each daily Parquet holds exactly one contract for a given
    root (NIFTY / BANKNIFTY), so we match days by *root* and stitch the May and
    June contracts into one front-month series. Labels and sliding windows are
    computed PER TRADING DAY, so the roll gap (and every overnight gap) is just
    a segment boundary — no spurious cross-contract returns, no window spans two
    days. This is the §9.1 "continuous front-month with roll markers" design.

Train/Val/Test split (default):
    Train: 2026-05-12 .. 2026-06-04   (15 trading days, May-FUT + early Jun-FUT)
    Val:   last 10% of train windows  (temporal, consistent with FI-2010 loader)
    Test:  2026-06-05, 08, 09, 10, 11, 12  (6 days, most recent — true OOS)
    (Jun-FUT expiry is ~2026-06-26, so all June days are still front-month — no second roll.)

May 11 dropped (partial session; collector started 10:02 IST). May 27-29
dropped (broken collection / 0-byte files during the May-FUT expiry roll —
see futures_expiry_roll memory).

Feature reorder (Dhan -> FI-2010):
    Dhan columns are emitted as separate ranges:
        [bid_price_1..20, bid_qty_1..20, ask_price_1..20, ask_qty_1..20]
    FI-2010 expects interleaved levels:
        [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, ..., ask_v10, bid_v10]
    We take the first 10 levels only (matches FI-2010 dimensionality of 40).

Outlier filter:
    Drops rows where:
      - bid_price_1 >= ask_price_1     (crossed quote)
      - mid_price not in [median - 6*MAD, median + 6*MAD]  per-day, per-symbol

Normalisation:
    Per-feature z-score using statistics computed on the training set only,
    then applied identically to val and test (no information leak).

Labels (FI-2010 convention; alpha is the threshold on smoothed mid-price
return, k is the forward window in events):
    m_behind(t) = mean(mid[t-k+1 .. t])
    m_ahead(t)  = mean(mid[t+1 .. t+k])
    r(t)        = (m_ahead - m_behind) / m_behind
    label(t)    = 0 (Down) if r < -alpha
                  1 (Stat) if -alpha <= r <= alpha
                  2 (Up)   if r > alpha
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fi2010_dataset import LOBDataset, WindowedLOBDataset

# Portable: <repo>/data_acquisition/data/dhan  (this file lives in <repo>/modeling/).
# Local convenience only — on Colab, data is pulled from S3 instead.
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data_acquisition" / "data" / "dhan"
)

# Continuous front-month series: May-FUT (to expiry) + Jun-FUT.
# 0-byte files 20260527/28/29 are omitted (broken expiry-roll collection).
TRAIN_DATES = [
    # May-FUT
    "20260512",
    "20260513",
    "20260514",
    "20260515",
    "20260518",
    "20260519",
    "20260520",
    "20260521",
    "20260522",
    "20260525",
    "20260526",
    # Jun-FUT (early)
    "20260601",
    "20260602",
    "20260603",
    "20260604",
]
TEST_DATES = [
    "20260605",
    "20260608",
    "20260609",
    "20260610",
    "20260611",
    "20260612",
]

N_LEVELS = 10
N_FEATURES = 4 * N_LEVELS  # 40 (matches FI-2010)
SEQ_LEN_DEFAULT = 100
ALPHA_DEFAULT = 1e-5
# Tick-return spike threshold for glitch detection. Real ticks are well under 0.05%
# even on volatile days; observed Dhan glitches jump from ~₹23K to ~₹39K (~65%).
TICK_SPIKE_THRESHOLD = 0.005  # 0.5% per-tick return


def _dhan_to_fi2010_order() -> list[str]:
    """Column names in FI-2010 interleaved order: [ask_p1, ask_v1, bid_p1, bid_v1, ...] x 10."""
    cols = []
    for i in range(1, N_LEVELS + 1):
        cols += [f"ask_price_{i}", f"ask_qty_{i}", f"bid_price_{i}", f"bid_qty_{i}"]
    return cols


def _load_day(
    path: Path, root: str, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """Read one day's Parquet, filter to the front-month contract for `root`.

    `root` is "NIFTY" or "BANKNIFTY". Each daily file holds exactly one futures
    contract per root (e.g. NIFTY-MAY-FUT before the roll, NIFTY-JUN-FUT after),
    so we match by prefix and pick up whichever contract is present that day.
    `feature_cols` are the extra raw columns the chosen feature set needs (order
    counts, deep levels). Rows are dropped only when an L1-10 price/qty is missing
    (those are 100% populated); deep levels / order counts are NaN-filled later.
    """
    base40 = _dhan_to_fi2010_order()
    feature_cols = feature_cols or base40
    needed = (
        ["timestamp", "symbol", "bid_price_1", "ask_price_1"] + base40 + feature_cols
    )
    df = pd.read_parquet(path, columns=list(dict.fromkeys(needed)))
    is_front = df["symbol"].str.startswith(f"{root}-") & df["symbol"].str.endswith(
        "-FUT"
    )
    df = df[is_front].copy()
    if df.empty:
        return df
    df["_contract"] = df["symbol"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop rows missing any L1-10 price/qty (these are fully populated); deep
    # levels (11-20) and order counts are NaN-filled in features.build_features.
    df = df.dropna(subset=base40)
    # Drop crossed quotes
    df = df[df["bid_price_1"] < df["ask_price_1"]]
    return df.reset_index(drop=True)


def _outlier_mask(
    mid: np.ndarray, threshold: float = TICK_SPIKE_THRESHOLD
) -> np.ndarray:
    """Tick-return spike mask. True = keep.

    Drops only ticks that jump >`threshold` (default 0.5%) relative to the
    previous valid mid-price. Tolerant of trending days (which MAD-on-mid was
    not) and catches the observed glitches where mid jumps ~65%.
    """
    n = len(mid)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    prev = mid[0]
    for i in range(1, n):
        if abs(mid[i] / prev - 1.0) > threshold:
            keep[i] = False
        else:
            prev = mid[i]
    return keep


def _load_and_clean(
    dates: list[str], root: str, data_dir: Path, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """Load + clean a list of date strings for one root. Returns a concatenated frame.

    Each row carries `_date` (the trading day) used downstream as the per-day
    segment id, so labels and windows never cross a day or contract boundary.
    """
    pieces = []
    for date in dates:
        path = data_dir / f"lob_dhan_{date}.parquet"
        if not path.exists():
            print(f"  [warn] missing: {path.name}")
            continue
        df = _load_day(path, root, feature_cols)
        if df.empty:
            print(f"  [warn] empty after filtering: {path.name} {root}")
            continue
        # Per-day outlier filter on mid-price
        mid = (df["bid_price_1"].values + df["ask_price_1"].values) / 2
        keep = _outlier_mask(mid)
        n_dropped = (~keep).sum()
        if n_dropped:
            print(f"    {date} {root}: dropped {n_dropped} outlier rows")
        df = df.iloc[keep].reset_index(drop=True)
        df["_date"] = date
        pieces.append(df)
    if not pieces:
        raise RuntimeError(f"No usable data for {symbol} in dates {dates}")
    return pd.concat(pieces, ignore_index=True)


def _make_labels(
    mid: np.ndarray, k: int, alpha: float, seg: np.ndarray | None = None
) -> np.ndarray:
    """FI-2010 style smoothed-mid labels. Returns int8 array of {0,1,2} with -1 sentinel for edges.

    If `seg` is given, labels are computed independently within each contiguous
    segment (one trading day / contract), so the smoothed mid never averages
    across a day or roll boundary. The last k rows of every segment become -1.
    """
    if seg is not None:
        labels = np.full(len(mid), -1, dtype=np.int8)
        for s in np.unique(seg):
            m = seg == s
            labels[m] = _make_labels(mid[m], k=k, alpha=alpha)
        return labels
    n = len(mid)
    # Cumulative sum trick for O(n) rolling means
    cs = np.concatenate([[0.0], np.cumsum(mid)])
    # m_behind(t) = mean(mid[t-k+1 .. t]) for t >= k-1
    # m_ahead(t)  = mean(mid[t+1 .. t+k])   for t <= n-k-1
    idx = np.arange(n)
    m_behind = np.full(n, np.nan)
    valid_b = idx >= k - 1
    m_behind[valid_b] = (cs[idx[valid_b] + 1] - cs[idx[valid_b] - k + 1]) / k
    m_ahead = np.full(n, np.nan)
    valid_a = idx + k < n
    m_ahead[valid_a] = (cs[idx[valid_a] + k + 1] - cs[idx[valid_a] + 1]) / k
    valid = valid_a & valid_b
    r = np.full(n, np.nan)
    r[valid] = (m_ahead[valid] - m_behind[valid]) / m_behind[valid]

    labels = np.full(n, -1, dtype=np.int8)
    labels[valid & (r > alpha)] = 2  # Up
    labels[valid & (r < -alpha)] = 0  # Down
    labels[valid & (r >= -alpha) & (r <= alpha)] = 1  # Stat
    return labels


def _make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    seg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows; keeps windows whose last-step label is valid (>=0).

    If `seg` is given, windows that span more than one segment (i.e. straddle a
    day / contract boundary) are dropped. Segments are contiguous blocks, so a
    window is intra-segment iff its first and last row share a segment id.
    """
    T = len(features)
    n_windows = T - seq_len + 1
    if n_windows <= 0:
        raise ValueError(f"Only {T} rows but seq_len={seq_len}.")
    idx = np.arange(seq_len)[None, :] + np.arange(n_windows)[:, None]
    last_labels = labels[idx[:, -1]]
    keep = last_labels >= 0
    if seg is not None:
        keep &= seg[idx[:, 0]] == seg[idx[:, -1]]
    return features[idx[keep]], last_labels[keep].astype(np.int64)


def load_nse(
    symbol: str = "NIFTY-JUN-FUT",
    horizon: int = 10,
    seq_len: int = SEQ_LEN_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    label_scheme: str = "A",
    feature_set: str = "base40",
    val_fraction: float = 0.1,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_dates: list[str] | None = None,
    test_dates: list[str] | None = None,
) -> tuple[LOBDataset, LOBDataset, LOBDataset]:
    """
    Load NSE futures LOB data for one symbol and one horizon.

    Args:
        symbol:        contract root — "NIFTY" or "BANKNIFTY" (any *-FUT suffix is
                       accepted and ignored; the May + June contracts are stitched
                       into one continuous front-month series)
        horizon:       k = number of LOB events forward used for label computation
        seq_len:       sliding window length (100 = DeepLOB convention)
        alpha:         Scheme A threshold on smoothed-mid return (1e-5 calibrated for index futures)
        label_scheme:  "A" = fixed `alpha` (FI-2010-style, comparability);
                       "B" = spread-relative threshold θ = mean(spread/mid) over the train set
                       (TLOB-style, ties the threshold to transaction cost; expect a dominant
                       Stationary class at short horizons)
        val_fraction:  last fraction of training timesteps held out for validation
        data_dir:      directory containing lob_dhan_YYYYMMDD.parquet files
        train_dates:   override default training dates (list of YYYYMMDD strings)
        test_dates:    override default test dates

    Returns:
        (train_ds, val_ds, test_ds) — LOBDataset instances, X shape (N, 100, 40), y in {0,1,2}.
    """
    # Resolve to a contract root. "NIFTY-MAY-FUT", "NIFTY-JUN-FUT", "NIFTY-FUT"
    # and "NIFTY" all map to root "NIFTY"; days are matched by root and the May/
    # June contracts are stitched into one continuous front-month series.
    root = symbol.split("-")[0]
    if root not in {"NIFTY", "BANKNIFTY"}:
        raise ValueError(f"Unknown symbol/root: {symbol} (root={root})")

    data_dir = Path(data_dir)
    train_dates = train_dates or TRAIN_DATES
    test_dates = test_dates or TEST_DATES
    fi_cols = _dhan_to_fi2010_order()

    print(
        f"\nLoading NSE | root={root} (front-month) | horizon=k{horizon} | seq_len={seq_len} | alpha={alpha}"
    )
    print(f"  Train dates ({len(train_dates)}): {train_dates[0]}..{train_dates[-1]}")
    print(f"  Test  dates ({len(test_dates)}): {test_dates}")

    from features import required_columns

    feat_cols = required_columns(feature_set)
    train_df = _load_and_clean(train_dates, root, data_dir, feat_cols)
    test_df = _load_and_clean(test_dates, root, data_dir, feat_cols)

    # Per-day segment ids — labels and windows never cross a day or roll boundary.
    train_seg = pd.factorize(train_df["_date"])[0]
    test_seg = pd.factorize(test_df["_date"])[0]
    contracts = sorted(set(train_df["_contract"]) | set(test_df["_contract"]))
    print(f"  Contracts stitched: {contracts}")
    print(f"  Train rows: {len(train_df):,}  Test rows: {len(test_df):,}")

    # ----- Features (raw) — base40 or an engineered set (Phase 2, Tier A) -----
    from features import build_features

    train_feat_raw, feat_names = build_features(train_df, train_seg, feature_set)
    test_feat_raw, _ = build_features(test_df, test_seg, feature_set)
    print(f"  Feature set: {feature_set}  (F={train_feat_raw.shape[1]})")

    # ----- Labels on raw mid-price (alpha is in return space, scale-invariant) -----
    train_mid = ((train_df["bid_price_1"] + train_df["ask_price_1"]) / 2).to_numpy(
        dtype=np.float64
    )
    test_mid = ((test_df["bid_price_1"] + test_df["ask_price_1"]) / 2).to_numpy(
        dtype=np.float64
    )
    # ----- Threshold: Scheme A (fixed alpha) or Scheme B (spread-relative) -----
    if label_scheme.upper() == "B":
        # θ = mean relative spread over the train set (ties threshold to transaction cost)
        tr_spread = (train_df["ask_price_1"] - train_df["bid_price_1"]).to_numpy(
            dtype=np.float64
        )
        thr = float(np.mean(tr_spread / train_mid))
        print(
            f"  Scheme B (spread-relative): theta={thr:.2e}  (mean rel. spread, train)"
        )
    else:
        thr = alpha
        print(f"  Scheme A (fixed alpha): threshold={thr:.2e}")
    train_labels = _make_labels(train_mid, k=horizon, alpha=thr, seg=train_seg)
    test_labels = _make_labels(test_mid, k=horizon, alpha=thr, seg=test_seg)

    # ----- Z-score: stats from train ONLY -----
    mu = train_feat_raw.mean(axis=0)
    sigma = train_feat_raw.std(axis=0)
    sigma[sigma < 1e-9] = 1.0  # avoid div-by-zero on constant features
    train_feat = ((train_feat_raw - mu) / sigma).astype(np.float32)
    test_feat = ((test_feat_raw - mu) / sigma).astype(np.float32)

    # ----- Val split: last val_fraction of training (temporal) -----
    val_start = int(len(train_feat) * (1 - val_fraction))
    val_feat, val_labels, val_seg = (
        train_feat[val_start:],
        train_labels[val_start:],
        train_seg[val_start:],
    )
    train_feat, train_labels, train_seg = (
        train_feat[:val_start],
        train_labels[:val_start],
        train_seg[:val_start],
    )

    # ----- Lazy sliding windows: drop invalid-label windows and any spanning a
    # day / contract boundary (seg change). Lazy windowing keeps memory O(T*F) —
    # the eager (N, seq_len, F) expansion OOMs on Colab for the stitched series. -----
    train_ds = WindowedLOBDataset(train_feat, train_labels, seq_len, seg=train_seg)
    val_ds = WindowedLOBDataset(val_feat, val_labels, seq_len, seg=val_seg)
    test_ds = WindowedLOBDataset(test_feat, test_labels, seq_len, seg=test_seg)

    # Attach raw mid-price + segment ids to the test set for the cost-aware
    # backtest (entry/exit prices and segment-aware holding). Harmless extras.
    test_ds.mid = test_mid
    test_ds.seg = test_seg

    print(
        f"  Splits   train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}"
    )
    _print_dist("  train labels", train_ds.y.numpy())
    _print_dist("  val   labels", val_ds.y.numpy())
    _print_dist("  test  labels", test_ds.y.numpy())

    return train_ds, val_ds, test_ds


def _print_dist(label: str, y: np.ndarray):
    counts = np.bincount(y, minlength=3)
    pcts = counts / max(counts.sum(), 1) * 100
    names = ["Down", "Stat", "Up"]
    dist = "  ".join(f"{n}:{p:.1f}%" for n, p in zip(names, pcts))
    print(f"  {label}: {dist}  (n={counts.sum():,})")


if __name__ == "__main__":
    # Quick smoke test — stitched front-month series.
    tr, vl, te = load_nse(symbol="NIFTY", horizon=10)
    X, y = tr[0]
    print(f"\nSample window shape: {tuple(X.shape)}  label: {y.item()}")
    print(f"Feature stats: mean={X.mean():.3f}  std={X.std():.3f}")
