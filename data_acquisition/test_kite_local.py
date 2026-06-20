"""
Local Kite Connect LOB Test Script
===================================
Collects 5-level market depth for a fixed duration, saves to CSV,
then prints a usability report comparing against FI-2010 expectations.

Usage:
    cd data_acquisition
    python3 test_kite_local.py

Step 1: Set credentials in .env (create it if it doesn't exist):
    KITE_API_KEY=xxxx
    KITE_API_SECRET=xxxx

Step 2: Script prints a login URL. Open it in browser, complete login.
        Copy the `request_token` from the redirect URL and paste it here.

Step 3: Data collection runs for COLLECT_MINUTES (default 15).
        Output: data/kite/test_YYYYMMDD_HHMMSS.csv

Step 4: Usability report printed at the end — compare with FI-2010 format.
"""

import csv
import datetime
import os
import sys
import threading
import time

from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.environ.get("KITE_API_KEY", "")
API_SECRET = os.environ.get("KITE_API_SECRET", "")

if not API_KEY:
    sys.exit("ERROR: Set KITE_API_KEY in .env")

INSTRUMENTS = {
    # --- NSE equities: top 10 by market cap ---
    # Note: HDFCBANK token changed after Aug-2025 1:1 bonus (old 738561 is now RELIANCE)
    738561: "RELIANCE",
    2953217: "TCS",
    341249: "HDFCBANK",
    408065: "INFY",
    1270529: "ICICIBANK",
    356865: "HINDUNILVR",
    779521: "SBIN",
    2714625: "BHARTIARTL",
    424961: "ITC",
    492033: "KOTAKBANK",
    # --- NFO: index futures (exp 2026-05-26) ---
    16914178: "NIFTY-MAY-FUT",
    16913410: "BANKNIFTY-MAY-FUT",
    16913666: "FINNIFTY-MAY-FUT",
    16913922: "MIDCPNIFTY-MAY-FUT",
    # --- NFO: single-stock futures (exp 2026-05-26) ---
    16986882: "RELIANCE-MAY-FUT",
    16995586: "TCS-MAY-FUT",
    16942082: "HDFCBANK-MAY-FUT",
    16951554: "INFY-MAY-FUT",
    16944898: "ICICIBANK-MAY-FUT",
    16988930: "SBIN-MAY-FUT",
    16922114: "BHARTIARTL-MAY-FUT",
    16954626: "ITC-MAY-FUT",
    16961282: "KOTAKBANK-MAY-FUT",
}

COLLECT_MINUTES = 15
LEVELS = 5
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "kite")

# ---------------------------------------------------------------------------
# Step 1 — Daily auth (request_token → access_token)
# ---------------------------------------------------------------------------


def get_access_token() -> str:
    """
    Check .env for a cached token from today, otherwise run the auth flow.
    Kite access tokens expire at midnight IST.
    """
    today = datetime.datetime.now(IST).strftime("%Y%m%d")
    token_file = os.path.join(OUTPUT_DIR, f".token_{today}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(token_file):
        token = open(token_file).read().strip()
        print("[auth] Reusing today's access token (cached).")
        return token

    if not API_SECRET:
        sys.exit(
            "ERROR: Set KITE_API_SECRET in .env to generate a new token.\n"
            "Alternatively, set KITE_ACCESS_TOKEN=<token> and add it to .env."
        )

    kite = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()
    print("\n" + "=" * 70)
    print("STEP: Open this URL in your browser and complete login:")
    print(f"\n  {login_url}\n")
    print("After login, you'll be redirected to a URL like:")
    print("  https://127.0.0.1/?request_token=XXXX&action=login&status=success")
    print("Copy the request_token value and paste it below.")
    print("=" * 70)

    request_token = input("\nPaste request_token: ").strip()
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    open(token_file, "w").write(access_token)
    print(f"[auth] Access token saved for today ({today}).")
    return access_token


# ---------------------------------------------------------------------------
# Step 2 — Collect ticks into memory, flush to CSV on exit
# ---------------------------------------------------------------------------

COLUMNS = ["timestamp", "symbol", "last_price"]
for side in ("bid", "ask"):
    for i in range(1, LEVELS + 1):
        COLUMNS += [f"{side}_price_{i}", f"{side}_qty_{i}", f"{side}_orders_{i}"]


def _tick_to_row(tick: dict) -> dict:
    ts = datetime.datetime.now(IST).isoformat()
    symbol = INSTRUMENTS.get(tick["instrument_token"], str(tick["instrument_token"]))
    depth = tick.get("depth", {})
    buys = depth.get("buy", [])
    sells = depth.get("sell", [])

    row: dict = {
        "timestamp": ts,
        "symbol": symbol,
        "last_price": tick.get("last_price"),
    }
    for prefix, levels in (("bid", buys), ("ask", sells)):
        for i in range(1, LEVELS + 1):
            lvl = levels[i - 1] if i <= len(levels) else {}
            row[f"{prefix}_price_{i}"] = lvl.get("price", None)
            row[f"{prefix}_qty_{i}"] = lvl.get("quantity", None)
            row[f"{prefix}_orders_{i}"] = lvl.get("orders", None)
    return row


rows: list[dict] = []
_lock = threading.Lock()
_stop = threading.Event()


def on_ticks(ws, ticks):
    with _lock:
        for tick in ticks:
            if tick.get("instrument_token") in INSTRUMENTS:
                if tick.get("depth"):
                    rows.append(_tick_to_row(tick))


def on_connect(ws, response):
    tokens = list(INSTRUMENTS.keys())
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)
    print(f"[ws] Connected. Subscribed {len(tokens)} instruments in FULL mode.")


def on_close(ws, code, reason):
    print(f"[ws] Closed: {code} {reason}")


def on_error(ws, code, reason):
    print(f"[ws] Error: {code} {reason}")


# ---------------------------------------------------------------------------
# Step 3 — Save CSV and print usability report
# ---------------------------------------------------------------------------


def save_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save] {len(rows):,} rows → {path}")


def usability_report(path: str):
    """
    Compare collected data against FI-2010 format requirements.
    Answers: can we train DeepLOB / MambaLOB on Kite data?
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("[report] pandas/numpy not available, skipping analysis.")
        return

    print("\n" + "=" * 70)
    print("  USABILITY REPORT vs FI-2010")
    print("=" * 70)

    if not rows:
        print("  No data collected — did the WebSocket connect?")
        return

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print(f"\n  Instruments captured : {sorted(df['symbol'].unique())}")
    print(f"  Total ticks          : {len(df):,}")
    duration_s = (
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
        if len(df) > 1
        else 0
    )
    print(f"  Collection window    : {duration_s:.0f}s ({duration_s / 60:.1f} min)")

    per_sym = df.groupby("symbol")
    print("\n  Ticks per instrument:")
    for sym, g in per_sym:
        rate = len(g) / duration_s * 60 if duration_s > 0 else 0
        print(f"    {sym:15s}  {len(g):6,} ticks  (~{rate:.1f}/min)")

    # Completeness: how many rows have all 5 bid+ask levels filled?
    price_cols = [
        f"{s}_price_{i}" for s in ("bid", "ask") for i in range(1, LEVELS + 1)
    ]
    complete = df[price_cols].notna().all(axis=1).mean() * 100
    print(
        f"\n  Depth completeness   : {complete:.1f}% rows have all {LEVELS} levels filled"
    )

    # Gap distribution (time between ticks per instrument)
    gaps = []
    for _, g in per_sym:
        g = g.sort_values("timestamp")
        gaps.extend(g["timestamp"].diff().dt.total_seconds().dropna().tolist())
    if gaps:
        gaps = np.array(gaps)
        print("\n  Inter-tick gap (all instruments):")
        print(
            f"    median {np.median(gaps):.2f}s  |  p95 {np.percentile(gaps, 95):.2f}s  |  max {gaps.max():.2f}s"
        )

    # Bid-ask spread
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        spread = (df["ask_price_1"] - df["bid_price_1"]) / df["bid_price_1"] * 100
        print("\n  Best bid-ask spread (bps, ÷100 = %):")
        print(
            f"    median {spread.median() * 100:.1f}  |  p95 {spread.quantile(0.95) * 100:.1f}"
        )

    print("\n" + "-" * 70)
    print("  VERDICT vs FI-2010 requirements")
    print("-" * 70)

    n_features_kite = LEVELS * 4  # price + qty each side
    n_features_fi2010 = 10 * 4  # 10 levels
    print(f"\n  FI-2010 input shape  : (B, 100, {n_features_fi2010})  [10-level LOB]")
    print(f"  Kite input shape     : (B, 100, {n_features_kite})   [5-level LOB]")
    print("\n  ✗ Direct swap        : NOT compatible — half the feature depth")
    print(
        "  ✓ Adapt models       : Change n_features=40 → 20 in all model constructors"
    )
    print("  ✓ Train on FI-2010   : Use full 10-level FI-2010 for main results")
    print("  ✓ NSE pilot          : Train separate 5-level model on Kite data (n=20)")
    print("\n  Tick rate assessment:")

    if duration_s > 0:
        total_rate = len(df) / duration_s * 60
        if total_rate > 100:
            verdict = "HIGH — sufficient event density for seq_len=100 windows"
        elif total_rate > 20:
            verdict = "MODERATE — usable; longer collection windows recommended"
        else:
            verdict = "LOW — thin book; may need to aggregate or widen symbols"
        print(f"    {total_rate:.0f} ticks/min combined → {verdict}")

    print("\n  Recommendation: Use FI-2010 as primary training data.")
    print("  Kite data is suitable for a held-out NSE validation set")
    print("  using a 5-level (20-feature) model variant.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    access_token = get_access_token()

    ts = datetime.datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"test_{ts}.csv")

    kws = KiteTicker(API_KEY, access_token)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error

    # Cap collection at market close (15:30 IST) so the script exits cleanly
    now = datetime.datetime.now(IST)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    secs_to_close = max(0, (market_close - now).total_seconds())
    collect_secs = min(COLLECT_MINUTES * 60, secs_to_close)
    if collect_secs < COLLECT_MINUTES * 60:
        print(
            f"\n[main] Market closes in {int(secs_to_close / 60)}m — collecting until 15:30 IST."
        )
    else:
        print(
            f"\n[main] Collecting for {COLLECT_MINUTES} minutes. Ctrl+C to stop early."
        )
    kws.connect(threaded=True)

    deadline = time.time() + collect_secs
    try:
        while time.time() < deadline:
            elapsed = int(time.time() - (deadline - COLLECT_MINUTES * 60))
            remaining = int(deadline - time.time())
            with _lock:
                n = len(rows)
            print(
                f"\r[{elapsed:>4}s elapsed | {remaining:>4}s remaining | {n:>6,} ticks captured]",
                end="",
                flush=True,
            )
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[main] Interrupted early.")

    kws.close()
    print()

    if rows:
        save_csv(out_path)
        usability_report(out_path)
    else:
        print("[main] No ticks received. Check credentials and market hours.")
        print("       NSE market hours: 09:15 – 15:30 IST")
        now_ist = datetime.datetime.now(IST)
        print(f"       Current time IST: {now_ist.strftime('%H:%M:%S')}")
