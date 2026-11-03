"""
Local Dhan Twenty-Depth LOB Test Script
========================================
Collects 20-level market depth via Dhan's WebSocket, validates the binary
packet parser, saves to CSV, and prints a usability report.

Key difference from collect_dhan.py (EC2 version):
  - Fixed duration (COLLECT_MINUTES), then stop + analyse
  - Prints raw hex of first N packets so you can validate the parser
  - Saves CSV instead of Parquet (easy to open and inspect)
  - Flags if parsed prices look implausible (out of range for NSE stocks)

Setup:
    Add to .env:
        DHAN_CLIENT_ID=your_client_id
        DHAN_ACCESS_TOKEN=your_access_token

    Token is valid 30 days — no daily login flow needed.

Instrument IDs:
    Download master: https://images.dhan.co/api-data/api-scrip-master.csv
    Column SEM_SMST_SECURITY_ID — use those values in INSTRUMENTS below.
    Supported segments: NSE_EQ (equity), NSE_FNO (futures & options).
    Use "NSE_FNO" for futures/options — NOT "NSE_FO" (that code does not exist in Dhan's API).

Usage:
    cd data_acquisition
    conda run -n pes_env python3 test_dhan_local.py
"""

import asyncio
import csv
import datetime
import os
import struct
import sys

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

if not CLIENT_ID or not ACCESS_TOKEN:
    sys.exit("ERROR: Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env")

# sid → display name
# Supported segments: NSE_EQ (equity) and NSE_FNO (futures & options).
# Note: the correct segment code for futures is "NSE_FNO", NOT "NSE_FO".
# Security IDs sourced from: https://images.dhan.co/api-data/api-scrip-master.csv
# Futures expiry: 2026-05-26. Roll to Jun (sid in comment) before that date.
INSTRUMENTS = {
    # --- NSE equities: top 10 by market cap ---
    "2885": "RELIANCE",
    "11536": "TCS",
    "1333": "HDFCBANK",
    "1594": "INFY",
    "4963": "ICICIBANK",
    "1394": "HINDUNILVR",
    "3045": "SBIN",
    "10604": "BHARTIARTL",
    "1660": "ITC",
    "1922": "KOTAKBANK",
    # --- NSE_FNO: index futures (exp 2026-05-26) ---
    "66071": "NIFTY-MAY-FUT",  # Jun: 62329
    "66068": "BANKNIFTY-MAY-FUT",  # Jun: 62326
    "66069": "FINNIFTY-MAY-FUT",  # Jun: 62327
    "66070": "MIDCPNIFTY-MAY-FUT",  # Jun: 62328
    # --- NSE_FNO: single-stock futures (exp 2026-05-26) ---
    "66355": "RELIANCE-MAY-FUT",  # Jun: 62802
    "66389": "TCS-MAY-FUT",  # Jun: 62851
    "66180": "HDFCBANK-MAY-FUT",  # Jun: 62593
    "66217": "INFY-MAY-FUT",  # Jun: 62620
    "66191": "ICICIBANK-MAY-FUT",  # Jun: 62604
    "66363": "SBIN-MAY-FUT",  # Jun: 62812
    "66102": "BHARTIARTL-MAY-FUT",  # Jun: 62384
    "66229": "ITC-MAY-FUT",  # Jun: 62625
    "66255": "KOTAKBANK-MAY-FUT",  # Jun: 62659
}

INSTRUMENT_SEGMENTS = {
    "2885": "NSE_EQ",
    "11536": "NSE_EQ",
    "1333": "NSE_EQ",
    "1594": "NSE_EQ",
    "4963": "NSE_EQ",
    "1394": "NSE_EQ",
    "3045": "NSE_EQ",
    "10604": "NSE_EQ",
    "1660": "NSE_EQ",
    "1922": "NSE_EQ",
    "66071": "NSE_FNO",
    "66068": "NSE_FNO",
    "66069": "NSE_FNO",
    "66070": "NSE_FNO",
    "66355": "NSE_FNO",
    "66389": "NSE_FNO",
    "66180": "NSE_FNO",
    "66217": "NSE_FNO",
    "66191": "NSE_FNO",
    "66363": "NSE_FNO",
    "66102": "NSE_FNO",
    "66229": "NSE_FNO",
    "66255": "NSE_FNO",
}

COLLECT_MINUTES = 15
LEVELS = 20
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "dhan")

WS_URL = f"wss://depth-api-feed.dhan.co/twentydepth?token={ACCESS_TOKEN}&clientId={CLIENT_ID}&authType=2"

SUBSCRIBE_MSG = {
    "RequestCode": 23,
    "InstrumentCount": len(INSTRUMENTS),
    "InstrumentList": [
        {"ExchangeSegment": INSTRUMENT_SEGMENTS.get(sid, "NSE_EQ"), "SecurityId": sid} for sid in INSTRUMENTS
    ],
}

# ---------------------------------------------------------------------------
# Binary parser — confirmed from packet analysis (debug_dhan_packet*.py)
#
# Two packet formats observed:
#
# FULL (1992B) — HDFCBANK-class stocks:
#   bytes  0-39  : header (security_id at bytes 4-7 as uint32 LE)
#   bytes 40-359 : BID  20 × 16B  [orders(4B)][price(8B f64)][qty(4B)]
#   bytes 360-679: ASK  20 × 16B  [price(8B f64)][qty(4B)][orders(4B)]
#   bytes 680+   : extra fields (timestamps, session data, etc.)
#
# COMPACT (664B) — INFY/RELIANCE-class stocks; two 332B sub-packets:
#   BID sub (bytes   0-331): header(12B) + 20 × 16B  [price(8B f64)][qty(4B)][orders(4B)]
#   ASK sub (bytes 332-663): header(12B) + 20 × 16B  [price(8B f64)][qty(4B)][orders(4B)]
#   (security_id at offset +4 within each sub-header)
#
# WebSocket messages pack multiple records: greedy parse FULL first, then COMPACT.
# ---------------------------------------------------------------------------
PACKET_SIZE = 1992  # full format
PACKET_SIZE_COMPACT = 664  # compact format

SECURITY_ID_OFFSET = 4

BID_START = 40
ASK_START = 360
LEVEL_SIZE = 16

BID_PRICE_OFFSET = 4  # float64 within full-format bid level
BID_QTY_OFFSET = 12  # uint32
BID_ORDERS_OFFSET = 0  # uint32

ASK_PRICE_OFFSET = 0  # float64 within full-format ask level
ASK_QTY_OFFSET = 8  # uint32
ASK_ORDERS_OFFSET = 12  # uint32

# Compact format offsets
COMPACT_BID_START = 12  # after 12B sub-header
COMPACT_ASK_START = 344  # 12B BID-sub-header + 20×16B BID + 12B ASK-sub-header

# Plausibility check — NSE large-cap prices (₹)
PRICE_MIN = 10.0
PRICE_MAX = 100_000.0


def _parse_single(data: bytes) -> tuple | None:
    """Parse one 1992B full-format record. Returns (security_id_str, bids, asks)."""
    if len(data) < PACKET_SIZE:
        return None
    security_id = str(struct.unpack_from("<I", data, SECURITY_ID_OFFSET)[0])

    bids = []
    for i in range(LEVELS):
        off = BID_START + i * LEVEL_SIZE
        price = struct.unpack_from("<d", data, off + BID_PRICE_OFFSET)[0]
        qty = struct.unpack_from("<I", data, off + BID_QTY_OFFSET)[0]
        orders = struct.unpack_from("<I", data, off + BID_ORDERS_OFFSET)[0]
        bids.append((price, qty, orders))

    asks = []
    for i in range(LEVELS):
        off = ASK_START + i * LEVEL_SIZE
        price = struct.unpack_from("<d", data, off + ASK_PRICE_OFFSET)[0]
        qty = struct.unpack_from("<I", data, off + ASK_QTY_OFFSET)[0]
        orders = struct.unpack_from("<I", data, off + ASK_ORDERS_OFFSET)[0]
        asks.append((price, qty, orders))

    return security_id, bids, asks


def _parse_compact(data: bytes) -> tuple | None:
    """Parse one 664B compact-format record. Returns (security_id_str, bids, asks)."""
    if len(data) < PACKET_SIZE_COMPACT:
        return None
    security_id = str(struct.unpack_from("<I", data, SECURITY_ID_OFFSET)[0])

    bids = []
    for i in range(LEVELS):
        off = COMPACT_BID_START + i * LEVEL_SIZE
        price = struct.unpack_from("<d", data, off)[0]
        qty = struct.unpack_from("<I", data, off + 8)[0]
        orders = struct.unpack_from("<I", data, off + 12)[0]
        bids.append((price, qty, orders))

    asks = []
    for i in range(LEVELS):
        off = COMPACT_ASK_START + i * LEVEL_SIZE
        price = struct.unpack_from("<d", data, off)[0]
        qty = struct.unpack_from("<I", data, off + 8)[0]
        orders = struct.unpack_from("<I", data, off + 12)[0]
        asks.append((price, qty, orders))

    return security_id, bids, asks


def _parse_packet(data: bytes) -> list[tuple]:
    """Parse a WebSocket message: greedy full-format first, then compact."""
    results = []
    offset = 0
    # Consume all full 1992B records
    while offset + PACKET_SIZE <= len(data):
        result = _parse_single(data[offset : offset + PACKET_SIZE])
        if result:
            results.append(result)
        offset += PACKET_SIZE
    # Consume remaining bytes as 664B compact records
    while offset + PACKET_SIZE_COMPACT <= len(data):
        result = _parse_compact(data[offset : offset + PACKET_SIZE_COMPACT])
        if result:
            results.append(result)
        offset += PACKET_SIZE_COMPACT
    return results


def _prices_plausible(bids: list, asks: list) -> bool:
    """Return True if L1 bid and L1 ask are both valid (bid < ask)."""
    b1 = bids[0][0] if bids else 0.0
    a1 = asks[0][0] if asks else 0.0
    return PRICE_MIN <= b1 <= PRICE_MAX and PRICE_MIN <= a1 <= PRICE_MAX and b1 < a1


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
rows: list[dict] = []
raw_packet_log: list[tuple] = []  # (size, hex_prefix) for first 5 packets
_odd_packet_saved = False  # save first non-multiple-1992 packet for offline analysis
parse_stats = {
    "total": 0,
    "parsed": 0,
    "wrong_size": 0,
    "bad_prices": 0,
    "unknown_id": 0,
}

COLUMNS = ["timestamp", "symbol"]
for _side in ("bid", "ask"):
    for _i in range(1, LEVELS + 1):
        COLUMNS += [f"{_side}_price_{_i}", f"{_side}_qty_{_i}", f"{_side}_orders_{_i}"]


def _build_row(security_id: str, bids: list, asks: list) -> dict:
    row: dict = {
        "timestamp": datetime.datetime.now(IST).isoformat(),
        "symbol": INSTRUMENTS[security_id],
    }
    for prefix, levels in (("bid", bids), ("ask", asks)):
        for i, (price, qty, orders) in enumerate(levels, start=1):
            row[f"{prefix}_price_{i}"] = price if PRICE_MIN <= price <= PRICE_MAX else None
            row[f"{prefix}_qty_{i}"] = qty if qty > 0 else None
            row[f"{prefix}_orders_{i}"] = orders if orders > 0 else None
    return row


# ---------------------------------------------------------------------------
# WebSocket collector
# ---------------------------------------------------------------------------
async def collect(deadline: float):
    import websockets

    print(f"[dhan] Connecting to: {WS_URL[:60]}...")

    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        open_timeout=15,
    ) as ws:
        import json

        await ws.send(json.dumps(SUBSCRIBE_MSG))
        print(f"[dhan] Subscribed to {len(INSTRUMENTS)} instruments.")
        print(f"[dhan] Collecting for {COLLECT_MINUTES} min. Ctrl+C to stop early.\n")

        async for message in ws:
            if asyncio.get_event_loop().time() > deadline:
                break

            if not isinstance(message, bytes):
                # JSON control message — print for visibility
                print(f"[dhan] JSON msg: {message[:120]}")
                continue

            parse_stats["total"] += 1

            # Log first 5 raw packets for parser validation
            if len(raw_packet_log) < 5:
                raw_packet_log.append((len(message), message[:32].hex()))
                print(f"  [raw packet #{len(raw_packet_log)}]  size={len(message)}B    first32={message[:32].hex()}")

            remainder = len(message) % PACKET_SIZE
            if remainder != 0:
                parse_stats["wrong_size"] += 1
                # still parse the complete 1992B records that fit — the tail is a
                # different message type (e.g. 1328B trade/session update) that we ignore
                global _odd_packet_saved
                if not _odd_packet_saved:
                    odd_path = os.path.join(OUTPUT_DIR, "debug_packet_odd.bin")
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    with open(odd_path, "wb") as _f:
                        _f.write(message)
                    print(f"  [debug] Saved first odd-size packet ({len(message)}B) → {odd_path}")
                    _odd_packet_saved = True

            for security_id, bids, asks in _parse_packet(message):
                if security_id not in INSTRUMENTS:
                    parse_stats["unknown_id"] += 1
                    continue

                if not _prices_plausible(bids, asks):
                    parse_stats["bad_prices"] += 1
                    if parse_stats["bad_prices"] <= 3:
                        prices = [p for p, _, _ in bids[:3]]
                        print(
                            f"  [WARN] implausible prices for {security_id}: "
                            f"bid L1-L3 = {prices} — parser may need adjustment"
                        )
                    continue

                parse_stats["parsed"] += 1
                rows.append(_build_row(security_id, bids, asks))

            if parse_stats["total"] % 200 == 0:
                elapsed = int(asyncio.get_event_loop().time() - (deadline - COLLECT_MINUTES * 60))
                print(
                    f"\r  [{elapsed:>4}s | "
                    f"{parse_stats['parsed']:>5} parsed | "
                    f"{parse_stats['wrong_size']} wrong_size | "
                    f"{parse_stats['bad_prices']} bad_prices]",
                    end="",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
def save_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save] {len(rows):,} rows → {path}")


# ---------------------------------------------------------------------------
# Parser diagnostics (printed before report if something looks off)
# ---------------------------------------------------------------------------
def print_parser_diagnostics():
    total = parse_stats["total"]
    if total == 0:
        print("\n[parser] No binary packets received at all.")
        print("         Check CLIENT_ID / ACCESS_TOKEN and Dhan API subscription.")
        return

    print("\n=== PARSER DIAGNOSTICS ===")
    print(f"  Total WS messages    : {total}")
    records = parse_stats["parsed"]
    print(f"  Records parsed       : {records}  (~{records / total:.1f} records/msg)")
    print(
        f"  With compact tail    : {parse_stats['wrong_size']}  "
        f"(msgs containing a 664B compact packet after full-format records)"
    )
    print(f"  Implausible prices   : {parse_stats['bad_prices']}")
    print(f"  Unknown instrument   : {parse_stats['unknown_id']}")

    if raw_packet_log:
        sizes = [s for s, _ in raw_packet_log]
        unique_sizes = set(sizes)
        bad_sizes = {s for s in unique_sizes if s % PACKET_SIZE != 0 and s % PACKET_SIZE_COMPACT != 0}
        if bad_sizes:
            print(f"\n  *** Truly unrecognised sizes: {bad_sizes}")
        else:
            print(f"\n  Packet sizes seen: {unique_sizes} — all parseable (full + compact)")

    if parse_stats["bad_prices"] > parse_stats["parsed"] * 0.5 and parse_stats["parsed"] < 10:
        print(
            "\n  *** Most prices are implausible. Parser offset mismatch — "
            "re-run debug_dhan_packet.py to confirm structure."
        )


# ---------------------------------------------------------------------------
# Usability report
# ---------------------------------------------------------------------------
def usability_report(path: str):
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("[report] pandas/numpy not available.")
        return

    print("\n" + "=" * 70)
    print("  USABILITY REPORT — DHAN 20-LEVEL vs FI-2010 vs KITE 5-LEVEL")
    print("=" * 70)

    if not rows:
        print("  No rows collected — see parser diagnostics above.")
        return

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print(f"\n  Instruments captured : {sorted(df['symbol'].unique())}")
    print(f"  Total ticks          : {len(df):,}")
    duration_s = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() if len(df) > 1 else 0
    print(f"  Collection window    : {duration_s:.0f}s ({duration_s / 60:.1f} min)")

    per_sym = df.groupby("symbol")
    print("\n  Ticks per instrument:")
    for sym, g in per_sym:
        rate = len(g) / duration_s * 60 if duration_s > 0 else 0
        n_windows = max(0, len(g) - 100 + 1)
        print(f"    {sym:15s}  {len(g):6,} ticks  (~{rate:.1f}/min)  windows={n_windows:,}")

    # Completeness
    price_cols_10 = [f"{s}_price_{i}" for s in ("bid", "ask") for i in range(1, 11)]
    price_cols_20 = [f"{s}_price_{i}" for s in ("bid", "ask") for i in range(1, 21)]
    complete_10 = df[price_cols_10].notna().all(axis=1).mean() * 100
    complete_20 = df[price_cols_20].notna().all(axis=1).mean() * 100
    print("\n  Depth completeness:")
    print(f"    10-level (FI-2010 equivalent) : {complete_10:.1f}%")
    print(f"    20-level (full Dhan depth)    : {complete_20:.1f}%")

    # Inter-tick gap
    gaps = []
    for _, g in per_sym:
        g = g.sort_values("timestamp")
        gaps.extend(g["timestamp"].diff().dt.total_seconds().dropna().tolist())
    if gaps:
        gaps = np.array(gaps)
        print("\n  Inter-tick gap:")
        print(f"    median {np.median(gaps):.2f}s  |  p95 {np.percentile(gaps, 95):.2f}s  |  max {gaps.max():.2f}s")

    # Spread
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        spread_bps = (df["ask_price_1"] - df["bid_price_1"]) / df["bid_price_1"] * 10000
        print("\n  Best bid-ask spread (bps):")
        print(f"    median {spread_bps.median():.1f}  |  p95 {spread_bps.quantile(0.95):.1f}")

    # Depth profile
    print("\n  Depth volume profile (avg qty, bid side L1-L5-L10-L20):")
    for sym, g in per_sym:
        sample = [int(g[f"bid_qty_{i}"].mean()) for i in [1, 5, 10, 20] if f"bid_qty_{i}" in g.columns]
        print(f"    {sym:15s}: L1={sample[0]:,}  L5={sample[1]:,}  L10={sample[2]:,}  L20={sample[3]:,}")

    # Label simulation (FI-2010 smoothed formula)
    # alpha=0.00002 (0.002%) instead of FI-2010's 0.002 (0.2%) — short test windows
    # don't produce enough price movement to cross a 0.2% threshold.
    print("\n  Simulated labels (FI-2010 smoothed mid-price, k=10, alpha=0.00002):")
    for sym, g in per_sym:
        g = g.sort_values("timestamp").reset_index(drop=True)
        mid = (g["bid_price_1"] + g["ask_price_1"]) / 2
        k, alpha = 10, 0.00002
        smooth_past = mid.rolling(k).mean()
        smooth_future = mid.shift(-k).rolling(k).mean()
        ret = (smooth_future - smooth_past) / smooth_past
        direction = ret.map(lambda r: 2 if r > alpha else (0 if r < -alpha else 1) if r == r else float("nan")).dropna()
        counts = direction.value_counts().sort_index()
        total = counts.sum()
        print(
            f"    {sym:15s}: Down={counts.get(0, 0) / total * 100:.1f}%"
            f"  Stat={counts.get(1, 0) / total * 100:.1f}%"
            f"  Up={counts.get(2, 0) / total * 100:.1f}%"
        )

    print("    FI-2010 CF7 k=10  : Down=19.9%  Stat=60.0%  Up=20.0%")

    print("\n" + "-" * 70)
    print("  COMPARISON TABLE")
    print("-" * 70)
    print(f"  {'':20s}  {'Kite':>10}  {'Dhan':>10}  {'FI-2010':>10}")
    print(f"  {'LOB levels':20s}  {'5':>10}  {'20':>10}  {'10':>10}")
    print(f"  {'Features (n)':20s}  {'20':>10}  {'80':>10}  {'40':>10}")
    dhan_rate = len(df) / duration_s * 60 if duration_s > 0 else 0
    print(f"  {'Ticks/min (combined)':20s}  {'~210':>10}  {int(dhan_rate):>10}  {'~653':>10}")
    print(f"  {'Depth complete':20s}  {'100%':>10}  {int(complete_10):>9}%  {'100%':>10}")
    print(f"  {'Historical data':20s}  {'No':>10}  {'No':>10}  {'Yes':>10}")
    print(f"  {'Event-driven':20s}  {'~500ms':>10}  {'?':>10}  {'Yes':>10}")

    print("\n  KEY QUESTION: Is Dhan event-driven or periodic (like Kite)?")
    if gaps.size > 0:
        gap_cv = np.std(gaps) / np.mean(gaps)  # coefficient of variation
        if gap_cv < 0.5:
            verdict = f"PERIODIC (low CV={gap_cv:.2f}) — similar to Kite ~500ms batching"
        else:
            verdict = f"EVENT-DRIVEN (high CV={gap_cv:.2f}) — gap variance suggests true LOB events"
        print(f"    Gap CV = {gap_cv:.2f} → {verdict}")

    print("\n  VERDICT FOR PROJECT:")
    if complete_10 > 95 and dhan_rate > 50:
        print("  ✓ Dhan 20-level data is HIGH quality and suitable for NSE pilot.")
        print("  ✓ Extract first 10 levels → 40-feature input, IDENTICAL to FI-2010 format.")
        print("  ✓ This is significantly better than Kite for this project.")
        print("  → Collect 5-10 days on EC2, use as NSE test/validation set.")
    else:
        print("  ✗ Data quality or rate below threshold — review parser diagnostics.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    ts = datetime.datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"test_{ts}.csv")

    now = datetime.datetime.now(IST)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    secs_to_close = max(0, (market_close - now).total_seconds())
    collect_secs = min(COLLECT_MINUTES * 60, secs_to_close)
    if collect_secs < COLLECT_MINUTES * 60:
        print(f"[dhan] Market closes in {int(secs_to_close / 60)}m — collecting until 15:30 IST.")

    loop = asyncio.get_event_loop()
    deadline = loop.time() + collect_secs

    try:
        await collect(deadline)
    except KeyboardInterrupt:
        print("\n[main] Interrupted early.")
    except Exception as e:
        import websockets.exceptions as _wse

        if isinstance(e, _wse.ConnectionClosedError):
            print(f"\n[main] Connection closed by server: {e}")
        else:
            print(f"\n[main] Error: {e}")
            raise

    print()
    print_parser_diagnostics()

    if rows:
        save_csv(out_path)
        usability_report(out_path)
    else:
        print("\n[main] No rows collected.")
        now_ist = datetime.datetime.now(IST)
        print(f"       Current IST: {now_ist.strftime('%H:%M:%S')}")
        print("       Market hours: 09:15–15:30 IST")


if __name__ == "__main__":
    asyncio.run(main())
