"""LOB Data Sanity Check — memory-efficient, processes one symbol at a time."""

import warnings

warnings.filterwarnings("ignore")
import os
import gc
import pandas as pd
import numpy as np

SOURCES = {
    "dhan": {"dir": "data/dhan", "levels": 20},
    "kite": {"dir": "data/kite", "levels": 5},
}

for source, cfg in SOURCES.items():
    data_dir, max_levels = cfg["dir"], cfg["levels"]
    files = (
        sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".parquet")
            ]
        )
        if os.path.isdir(data_dir)
        else []
    )
    if not files:
        print(f"\n[{source.upper()}] No files in {data_dir}/")
        continue

    print(
        f"\n{'=' * 60}\n  SOURCE: {source.upper()}  ({len(files)} file(s))\n{'=' * 60}"
    )
    for f in files:
        print(f"  {os.path.basename(f)}  →  {os.path.getsize(f) // 1024} KB")

    # Read only essential columns first for overview
    key_cols = [
        "timestamp",
        "symbol",
        "bid_price_1",
        "ask_price_1",
        "bid_qty_1",
        "ask_qty_1",
    ]
    if max_levels == 20:
        key_cols += ["bid_orders_1", "ask_orders_1"]

    df = pd.concat(
        [pd.read_parquet(f, columns=key_cols) for f in files], ignore_index=True
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
        "Asia/Kolkata"
    )
    duration_s = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()

    print("\n--- OVERVIEW ---")
    print(f"  Total rows : {len(df):,}")
    print(f"  Symbols    : {sorted(df['symbol'].unique())}")
    print(
        f"  Time range : {df['timestamp'].min().strftime('%H:%M:%S')} → {df['timestamp'].max().strftime('%H:%M:%S')} IST"
    )
    print(f"  Duration   : {duration_s / 60:.1f} min")

    print("\n--- TICK RATES ---")
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        rate = len(g) / duration_s * 60
        gaps = g["timestamp"].diff().dt.total_seconds().dropna()
        cv = gaps.std() / gaps.mean()
        print(
            f"  {sym:22s}  {len(g):6,} ticks  {rate:5.1f}/min  gap_med={gaps.median():.3f}s  CV={cv:.2f}"
        )

    print("\n--- PRICE SANITY ---")
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    print(f"  Negative spreads : {(df['spread'] < 0).sum()}")
    print(f"  Zero spreads     : {(df['spread'] == 0).sum()}")
    for sym, g in df.groupby("symbol"):
        mid = (g["bid_price_1"] + g["ask_price_1"]) / 2
        spr = (g["ask_price_1"] - g["bid_price_1"]) / g["bid_price_1"] * 10000
        print(
            f"  {sym:22s}  mid={mid.mean():.1f}  spread_med={spr.median():.2f}bps  range=[{mid.min():.1f},{mid.max():.1f}]"
        )

    print("\n--- QTY SANITY ---")
    for col in ["bid_qty_1", "ask_qty_1"]:
        print(
            f"  {col}: min={df[col].min()}  max={df[col].max():,}  mean={df[col].mean():.1f}  nulls={df[col].isna().sum()}"
        )

    del df
    gc.collect()

    # Missing data — load full file but check level completeness
    print("\n--- MISSING DATA (by depth level) ---")
    df_full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for n in [5, 10, 20] if max_levels == 20 else [5]:
        bp = [f"bid_price_{i}" for i in range(1, n + 1)]
        ap = [f"ask_price_{i}" for i in range(1, n + 1)]
        pct = df_full[bp + ap].notna().all(axis=1).mean() * 100
        print(f"  L1-L{n:2d} complete : {pct:.1f}%")
    print("  Missing by level:")
    for i in [1, 5, 10, 15, 20][:max_levels]:
        b, a = f"bid_price_{i}", f"ask_price_{i}"
        if b in df_full.columns:
            print(
                f"    L{i:2d}: bid={df_full[b].isna().mean() * 100:.1f}%  ask={df_full[a].isna().mean() * 100:.1f}%"
            )
    del df_full
    gc.collect()

    # Labels — sample 5000 rows per symbol to save memory
    print("\n--- LABEL DISTRIBUTION (alpha=1e-5, k=5, sample=5000/sym) ---")
    df_lbl = pd.concat(
        [
            pd.read_parquet(
                f, columns=["timestamp", "symbol", "bid_price_1", "ask_price_1"]
            )
            for f in files
        ],
        ignore_index=True,
    )
    df_lbl["timestamp"] = pd.to_datetime(df_lbl["timestamp"], utc=True)
    ALPHA, K = 1e-5, 5
    for sym, g in df_lbl.groupby("symbol"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) > 5000:
            g = g.sample(5000, random_state=42).sort_index().reset_index(drop=True)
        mid = (g["bid_price_1"] + g["ask_price_1"]) / 2
        m_ahead = pd.Series(
            [
                mid[i : i + K].mean() if i + K <= len(mid) else np.nan
                for i in range(len(mid))
            ]
        )
        m_behind = pd.Series(
            [mid[max(0, i - K + 1) : i + 1].mean() for i in range(len(mid))]
        )
        ret = (m_ahead - m_behind) / m_behind
        lbl = ret.apply(
            lambda r: (
                "Up"
                if r > ALPHA
                else ("Down" if r < -ALPHA else "Stat")
                if pd.notna(r)
                else None
            )
        )
        vc = lbl.value_counts(normalize=True) * 100
        print(
            f"  {sym:22s}  Up={vc.get('Up', 0):.1f}%  Stat={vc.get('Stat', 0):.1f}%  Down={vc.get('Down', 0):.1f}%"
        )
    del df_lbl
    gc.collect()

    print("\n--- DAILY PROJECTION ---")
    df_proj = pd.concat(
        [pd.read_parquet(f, columns=["symbol", "timestamp"]) for f in files],
        ignore_index=True,
    )
    df_proj["timestamp"] = pd.to_datetime(df_proj["timestamp"], utc=True)
    dur = (df_proj["timestamp"].max() - df_proj["timestamp"].min()).total_seconds()
    for sym, g in df_proj.groupby("symbol"):
        day = int(len(g) / dur * 60 * 375)
        print(f"  {sym:22s}  ~{day:,} events/day  ~{max(0, day - 99):,} windows/day")
    del df_proj
    gc.collect()
