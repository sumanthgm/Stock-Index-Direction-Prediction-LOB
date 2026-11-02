"""
Cost-aware backtest — a SIGNAL-QUALITY assessment, not a tradeable strategy.

Translates a model's class probabilities into long/short/flat signals at a confidence
threshold tau, holds for the label horizon, and reports net PnL / hit-rate / a per-trade
Sharpe-like ratio after transaction costs. Positions overlap (windows are dense), so this
measures whether the directional signal survives costs — it is not a realizable P&L.

Costs are charged round-trip as `cost_bps` of notional (≈ one spread crossing + STT/exchange/
SEBI/GST for NSE index futures; default ~5 bps). Frame results accordingly in the report.

Requires a test dataset produced by nse_dataset.load_nse (it attaches `.mid` raw mid-price and
`.seg` per-day segment ids, aligned to the windows' end indices in `.ends`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from fi2010_dataset import make_loader
from train import DEVICE


def predict_probs(model, ds, batch_size: int = 512) -> np.ndarray:
    """Softmax class probabilities, columns = [Down, Stat, Up]. Shape (N, 3)."""
    model = model.to(DEVICE).eval()
    loader = make_loader(ds, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for X, _ in loader:
            out.append(torch.softmax(model(X.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(out)


def run_backtest(
    model,
    test_ds,
    horizon: int,
    taus=(0.40, 0.50, 0.60, 0.70, 0.80),
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Sweep confidence thresholds; return per-tau signal-quality + cost-aware PnL stats.

    For each window: long if P(up) > tau, short if P(down) > tau, else flat. Enter at the
    window-end mid, exit `horizon` events later (only if still in the same trading day), and
    net out `cost_bps` round-trip. Reports trades, hit-rate (gross-positive trades),
    gross/net points, mean net return (bps), and a per-trade information ratio (mean/std).
    """
    probs = predict_probs(model, test_ds)
    mid = np.asarray(test_ds.mid, dtype=np.float64)
    ends = np.asarray(test_ds.ends)
    seg = np.asarray(test_ds.seg)
    p_down, p_up = probs[:, 0], probs[:, 2]

    exit_idx = ends + horizon
    in_range = exit_idx < len(mid)
    safe_exit = np.where(in_range, exit_idx, len(mid) - 1)
    same_seg = seg[safe_exit] == seg[ends]  # don't hold across a day/contract boundary

    rows = []
    for tau in taus:
        sig = np.zeros(len(ends), dtype=int)
        sig[p_up > tau] = 1
        sig[p_down > tau] = -1
        take = (sig != 0) & in_range & same_seg
        n = int(take.sum())
        if n == 0:
            rows.append(
                {
                    "tau": tau,
                    "trades": 0,
                    "hit_rate": None,
                    "gross_pts": 0.0,
                    "net_pts": 0.0,
                    "net_bps_mean": None,
                    "per_trade_ir": None,
                }
            )
            continue
        entry = mid[ends[take]]
        ex = mid[exit_idx[take]]
        direction = sig[take]
        gross = direction * (ex - entry)  # points
        cost = (cost_bps / 1e4) * entry  # round-trip cost in points
        net = gross - cost
        ret_bps = net / entry * 1e4
        # Per-trade information ratio (mean/std). NOT annualized / not sqrt(n)-scaled:
        # positions overlap heavily, so sqrt(n) scaling would be meaningless.
        ir = float(ret_bps.mean() / (ret_bps.std() + 1e-9))
        rows.append(
            {
                "tau": tau,
                "trades": n,
                "hit_rate": round(float((gross > 0).mean()), 4),
                "gross_pts": round(float(gross.sum()), 2),
                "net_pts": round(float(net.sum()), 2),
                "net_bps_mean": round(float(ret_bps.mean()), 3),
                "per_trade_ir": round(ir, 4),
            }
        )
    return pd.DataFrame(rows)
