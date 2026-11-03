import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys

CSV = "data/dhan/test_20260508_142259.csv"
df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
duration_s = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()

print("=== TICK RATES & GAP DISTRIBUTION ===")
for sym, g in df.groupby("symbol"):
    g = g.sort_values("timestamp")
    rate = len(g) / duration_s * 60
    gaps = g["timestamp"].diff().dt.total_seconds().dropna()
    cv = gaps.std() / gaps.mean()
    print(
        f"  {sym:22s}  {len(g):6,} ticks  {rate:5.1f}/min  gap_med={gaps.median():.3f}s  p95={gaps.quantile(0.95):.3f}s  max={gaps.max():.3f}s  CV={cv:.2f}"
    )

print("\n=== DEPTH COMPLETENESS ===")
for n in [5, 10, 20]:
    bp = [f"bid_price_{i}" for i in range(1, n + 1)]
    ap = [f"ask_price_{i}" for i in range(1, n + 1)]
    pct = df[bp + ap].notna().all(axis=1).mean() * 100
    print(f"  L1-L{n:2d}: {pct:.1f}%")

print("\n=== BID-ASK SPREAD ===")
df["spr_bps"] = (df["ask_price_1"] - df["bid_price_1"]) / df["bid_price_1"] * 10000
for sym, g in df.groupby("symbol"):
    s = g["spr_bps"].dropna()
    print(f"  {sym:22s}  median={s.median():.2f} bps  mean={s.mean():.2f}  p95={s.quantile(0.95):.2f}")

print("\n=== DEPTH PROFILE — avg qty per level ===")
for sym, g in df.groupby("symbol"):
    print(f"  {sym}")
    for prefix, side in [("bid", "BID"), ("ask", "ASK")]:
        vals = [g[f"{prefix}_qty_{i}"].mean() for i in [1, 2, 3, 5, 10, 20]]
        row = "  ".join(f"L{i}={v:5.0f}" for i, v in zip([1, 2, 3, 5, 10, 20], vals))
        print(f"    {side}: {row}")

print("\n=== ORDERS PER LEVEL (queue fragmentation) ===")
for sym, g in df.groupby("symbol"):
    vals = []
    for i in [1, 3, 5, 10]:
        b = g[f"bid_orders_{i}"].mean()
        a = g[f"ask_orders_{i}"].mean()
        vals.append(f"L{i}=bid{b:.1f}/ask{a:.1f}")
    print(f"  {sym:22s}  " + "  ".join(vals))

print("\n=== PRICE RANGE & VOLATILITY ===")
for sym, g in df.groupby("symbol"):
    mid = (g["bid_price_1"] + g["ask_price_1"]) / 2
    prange = (mid.max() - mid.min()) / mid.mean() * 100
    tick_vol = mid.pct_change().std() * 100
    print(f"  {sym:22s}  range={prange:.3f}%  per-tick std={tick_vol:.5f}%")

print("\n=== MID-PRICE LABELS (FI-2010 style: smooth k=5) ===")
for ALPHA, label in [
    (0.0002, "alpha=2e-4 (tight)"),
    (0.002, "alpha=2e-3 (FI-2010 H=10)"),
]:
    print(f"  {label}:")
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        mid = (g["bid_price_1"] + g["ask_price_1"]) / 2
        K = 5
        m_ahead = pd.Series([mid[i : i + K].mean() if i + K <= len(mid) else np.nan for i in range(len(mid))])
        m_behind = pd.Series([mid[max(0, i - K + 1) : i + 1].mean() for i in range(len(mid))])
        ret = (m_ahead - m_behind) / m_behind
        lbl = ret.apply(lambda r: "Up" if r > ALPHA else ("Down" if r < -ALPHA else "Stat") if pd.notna(r) else None)
        vc = lbl.value_counts(normalize=True) * 100
        n = lbl.notna().sum()
        print(
            f"    {sym:22s}  Up={vc.get('Up', 0):.1f}%  Stat={vc.get('Stat', 0):.1f}%  Down={vc.get('Down', 0):.1f}%  (n={n:,})"
        )

print("\n=== WINDOW COUNT & PROJECTION ===")
print(
    f"  {'Symbol':22s}  {'Ticks':>8}  {'seq100':>8}  {'1-day est':>10}  {'5-day':>10}  {'10-day':>10}  {'vs FI-2010':>12}"
)
for sym, g in df.groupby("symbol"):
    n = len(g)
    rate = n / duration_s * 60
    d1, d5, d10 = int(rate * 375), int(rate * 375 * 5), int(rate * 375 * 10)
    ratio = d10 / 400000
    print(f"  {sym:22s}  {n:8,}  {max(0, n - 99):8,}  {d1:10,}  {d5:10,}  {d10:10,}  {ratio:10.2f}x")
print("\n  FI-2010 (5 stocks, 10 days): ~400,000 events/stock")
sys.stdout.flush()
