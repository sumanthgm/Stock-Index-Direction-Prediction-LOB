"""
Microstructure feature engineering for NSE LOB data (Phase 2, Tier A).

Builds engineered channels on top of the raw 40-feature LOB window and exposes named
feature sets for an ablation study ("how much / which microstructure information does the
NSE index-futures signal need?"). All features are computed on the cleaned, per-day-sorted
DataFrame; flow features (OFI) are computed PER SEGMENT (trading day / contract) so they
never cross an overnight or expiry-roll boundary.

Feature sets (use via nse_dataset.load_nse(feature_set=...)):
  - "base40"   : 10 levels x [ask_p, ask_v, bid_p, bid_v]                     -> 40  (FI-2010 layout)
  - "micro"    : base40 + 6 microstructure scalars (below)                    -> 46
  - "orders60" : base40 + per-level order counts (10 x [bid_orders, ask_orders]) -> 60  (Dhan-unique)
  - "depth80"  : 20 levels x [ask_p, ask_v, bid_p, bid_v]                     -> 80
  - "all"      : base40 + micro(6) + order counts(20)                          -> 66

Microstructure scalars (the "micro" block):
  micro_price, relative_spread, depth_imbalance_L1, depth_imbalance_agg(10),
  OFI_L1 (Cont et al.), OFI_sum5 (multi-level order-flow imbalance over levels 1-5).
"""

from __future__ import annotations

import numpy as np

N_LEVELS = 10


def _col(df, name):
    return df[name].to_numpy(dtype=np.float64)


def _base_order(levels: int) -> list[str]:
    """FI-2010 interleaved column order for `levels` levels."""
    cols = []
    for i in range(1, levels + 1):
        cols += [f"ask_price_{i}", f"ask_qty_{i}", f"bid_price_{i}", f"bid_qty_{i}"]
    return cols


def _safe_div(num, den):
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den != 0)
    return out


def _ofi_level(pb, vb, pa, va, seg) -> np.ndarray:
    """Single-level Order-Flow Imbalance (Cont et al.), computed per segment.

    e_n = dW_bid - dW_ask, where (with previous event n-1):
      dW_bid = vb_n            if pb_n > pb_{n-1}
               vb_n - vb_{n-1} if pb_n == pb_{n-1}
               -vb_{n-1}       if pb_n < pb_{n-1}
      dW_ask = -va_{n-1}       if pa_n > pa_{n-1}
               va_n - va_{n-1} if pa_n == pa_{n-1}
               va_n            if pa_n < pa_{n-1}
    First event of each segment is 0 (no predecessor).
    """
    n = len(pb)
    ofi = np.zeros(n, dtype=np.float64)
    # previous-event values
    pbp, vbp = np.roll(pb, 1), np.roll(vb, 1)
    pap, vap = np.roll(pa, 1), np.roll(va, 1)
    dW_b = np.where(pb > pbp, vb, np.where(pb == pbp, vb - vbp, -vbp))
    dW_a = np.where(pa > pap, -vap, np.where(pa == pap, va - vap, va))
    ofi = dW_b - dW_a
    # zero out the first row of every segment (its "previous" wrapped around)
    seg = np.asarray(seg)
    first = np.ones(n, dtype=bool)
    first[1:] = seg[1:] != seg[:-1]
    ofi[first] = 0.0
    return ofi


def micro_block(df, seg) -> tuple[np.ndarray, list[str]]:
    """The 6 microstructure scalar channels."""
    pb1, vb1 = _col(df, "bid_price_1"), _col(df, "bid_qty_1")
    pa1, va1 = _col(df, "ask_price_1"), _col(df, "ask_qty_1")
    mid = (pa1 + pb1) / 2.0

    micro_price = _safe_div(pb1 * va1 + pa1 * vb1, va1 + vb1)  # Stoikov micro-price
    rel_spread = _safe_div(pa1 - pb1, mid)
    depth_imb_l1 = _safe_div(vb1 - va1, vb1 + va1)

    vb_agg = np.sum([_col(df, f"bid_qty_{i}") for i in range(1, N_LEVELS + 1)], axis=0)
    va_agg = np.sum([_col(df, f"ask_qty_{i}") for i in range(1, N_LEVELS + 1)], axis=0)
    depth_imb_agg = _safe_div(vb_agg - va_agg, vb_agg + va_agg)

    ofi_l1 = _ofi_level(pb1, vb1, pa1, va1, seg)
    ofi_sum5 = np.sum(
        [
            _ofi_level(
                _col(df, f"bid_price_{i}"),
                _col(df, f"bid_qty_{i}"),
                _col(df, f"ask_price_{i}"),
                _col(df, f"ask_qty_{i}"),
                seg,
            )
            for i in range(1, 6)
        ],
        axis=0,
    )
    block = np.column_stack(
        [micro_price, rel_spread, depth_imb_l1, depth_imb_agg, ofi_l1, ofi_sum5]
    )
    names = [
        "micro_price",
        "rel_spread",
        "depth_imb_l1",
        "depth_imb_agg",
        "ofi_l1",
        "ofi_sum5",
    ]
    return block, names


def order_count_block(df, levels: int = N_LEVELS) -> tuple[np.ndarray, list[str]]:
    """Per-level order counts (Dhan-unique), interleaved [bid_orders_i, ask_orders_i]."""
    cols, names = [], []
    for i in range(1, levels + 1):
        cols += [_col(df, f"bid_orders_{i}"), _col(df, f"ask_orders_{i}")]
        names += [f"bid_orders_{i}", f"ask_orders_{i}"]
    return np.column_stack(cols), names


_VALID = ("base40", "micro", "orders60", "depth80", "all")


def required_columns(feature_set: str) -> list[str]:
    """Raw parquet columns a feature set needs (so the loader reads only what's used)."""
    fs = feature_set.lower()
    if fs not in _VALID:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. Choose one of {_VALID}."
        )
    levels = 20 if fs == "depth80" else N_LEVELS
    cols = set(_base_order(levels))
    cols |= set(_base_order(N_LEVELS))  # micro/labels always need L1..10 price+qty
    if fs in ("orders60", "all"):
        for i in range(1, N_LEVELS + 1):
            cols |= {f"bid_orders_{i}", f"ask_orders_{i}"}
    return sorted(cols)


def _depth80_matrix(df) -> tuple[np.ndarray, list[str]]:
    """20-level matrix with sparse deep levels filled: NaN qty -> 0, NaN price ->
    same-side best price (levels 11-20 are only ~19% populated in the Dhan feed)."""
    pb1, pa1 = _col(df, "bid_price_1"), _col(df, "ask_price_1")
    cols, names = [], []
    for i in range(1, 21):
        ap = np.nan_to_num(_col(df, f"ask_price_{i}"), nan=np.nan)
        ap = np.where(np.isnan(ap), pa1, ap)
        bp = _col(df, f"bid_price_{i}")
        bp = np.where(np.isnan(bp), pb1, bp)
        aq = np.nan_to_num(_col(df, f"ask_qty_{i}"), nan=0.0)
        bq = np.nan_to_num(_col(df, f"bid_qty_{i}"), nan=0.0)
        cols += [ap, aq, bp, bq]
        names += [f"ask_price_{i}", f"ask_qty_{i}", f"bid_price_{i}", f"bid_qty_{i}"]
    return np.column_stack(cols), names


def build_features(
    df, seg, feature_set: str = "base40"
) -> tuple[np.ndarray, list[str]]:
    """Assemble the raw (pre-z-score) feature matrix for a feature set.

    Returns (X: (T, F) float64, feature_names). `seg` is the per-row segment id (used by OFI).
    L1-10 are 100% populated (the loader drops incomplete rows there); order counts and deep
    levels are NaN-filled here.
    """
    fs = feature_set.lower()
    if fs not in _VALID:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. Choose one of {_VALID}."
        )
    if fs == "depth80":
        X, names = _depth80_matrix(df)
        return np.nan_to_num(X), names

    base_cols = _base_order(N_LEVELS)
    X = df[base_cols].to_numpy(dtype=np.float64)
    names = list(base_cols)
    if fs in ("micro", "all"):
        mb, mn = micro_block(df, seg)
        X = np.column_stack([X, mb])
        names += mn
    if fs in ("orders60", "all"):
        ob, on = order_count_block(df, N_LEVELS)
        X = np.column_stack([X, np.nan_to_num(ob)])
        names += on
    return np.nan_to_num(X), names
