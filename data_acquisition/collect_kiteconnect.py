"""
Kite Connect LOB Collector — Production EC2 Version
====================================================
Streams 5-level bid/ask depth for NIFTY and BANKNIFTY index futures via
Kite Connect WebSocket during NSE market hours (09:15–15:30 IST).
Writes one Parquet file per trading day (zstd compressed).

IMPORTANT — Daily Token Refresh:
    Kite Connect access tokens expire at midnight IST every day.
    Before market open each morning, refresh the token and update .env on EC2:

    Local machine (run once per trading day before 09:15 IST):
        python3 kite_refresh_token.py       ← run this locally, follow the prompts
        ssh ec2-user@YOUR_EC2_IP 'echo "KITE_ACCESS_TOKEN=<new_token>" >> .env'

    See kite_refresh_token.py for the interactive auth flow.

Credentials (.env on EC2):
    KITE_API_KEY=your_api_key
    KITE_API_SECRET=your_api_secret
    KITE_ACCESS_TOKEN=today_access_token   ← update this DAILY before 09:15 IST

Setup on EC2:
    pip install kiteconnect pyarrow python-dotenv

Crontab (crontab -e):
    # Start collector at 09:10 IST (03:40 UTC) every weekday
    40 3 * * 1-5 cd /home/ubuntu/capstone/Data_Acquisition_and_preprocessing && python3 collect_kiteconnect.py >> logs/kite.log 2>&1
    # Sync to S3 at 15:40 IST (10:10 UTC) every weekday
    10 10 * * 1-5 bash /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/sync_to_s3.sh >> logs/sync.log 2>&1

Note on depth levels:
    Kite provides only 5 bid/ask levels (20 features) vs FI-2010's 10 levels (40 features).
    Dhan provides 20 levels and is event-driven — preferred for the capstone.
    Use this Kite collector for supplementary/comparison data only.

Instrument tokens (NFO segment):
    Tokens below are for May 2026 expiry (2026-05-29).
    Get next-series tokens: kite.instruments("NFO") → filter SYM + expiry
    Or visit: https://kite.zerodha.com/api/instruments and filter by name
"""

import datetime
import logging
import os
import signal
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from kiteconnect import KiteTicker

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("KITE_API_KEY", "")
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")

if not API_KEY:
    sys.exit("ERROR: Set KITE_API_KEY in .env")
if not ACCESS_TOKEN:
    sys.exit(
        "ERROR: Set KITE_ACCESS_TOKEN in .env\n"
        "Tokens expire daily — run kite_refresh_token.py locally before 09:15 IST."
    )

# NIFTY and BANKNIFTY index futures — NFO segment
# Tokens for June 2026 expiry (2026-06-30). Roll to Jul before expiry.
INSTRUMENTS = {
    15956226: "NIFTY-JUN-FUT",  # NFO:NIFTY26JUNFUT
    15955458: "BANKNIFTY-JUN-FUT",  # NFO:BANKNIFTY26JUNFUT
}

LEVELS = 5
FLUSH_EVERY = 200
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
MARKET_OPEN = datetime.time(9, 15)
MARKET_CLOSE = datetime.time(15, 30)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "kite")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet schema + buffer
# ---------------------------------------------------------------------------
def _make_schema(levels: int) -> pa.Schema:
    fields = [
        pa.field("timestamp", pa.timestamp("us", tz="Asia/Kolkata")),
        pa.field("symbol", pa.string()),
        pa.field("last_price", pa.float64()),
    ]
    for side in ("bid", "ask"):
        for i in range(1, levels + 1):
            fields += [
                pa.field(f"{side}_price_{i}", pa.float64()),
                pa.field(f"{side}_qty_{i}", pa.int32()),
                pa.field(f"{side}_orders_{i}", pa.int32()),
            ]
    return pa.schema(fields)


SCHEMA = _make_schema(LEVELS)


class ParquetBuffer:
    def __init__(self, path: str, schema: pa.Schema, flush_every: int = FLUSH_EVERY):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._writer = pq.ParquetWriter(path, schema, compression="zstd")
        self._schema = schema
        self._flush_every = flush_every
        self._buffer: list[dict] = []
        self._total = 0
        log.info("Parquet writer opened: %s", path)

    def add(self, row: dict):
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._total += len(self._buffer)
        log.info("Flushed %d rows (total: %d)", len(self._buffer), self._total)
        self._buffer.clear()

    def close(self):
        self.flush()
        self._writer.close()
        log.info("Parquet writer closed. Total rows: %d", self._total)


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------
def _tick_to_row(tick: dict) -> dict:
    ts = datetime.datetime.now(IST)
    symbol = INSTRUMENTS.get(tick["instrument_token"], str(tick["instrument_token"]))
    depth = tick.get("depth", {})
    buys = depth.get("buy", [{}] * LEVELS)
    sells = depth.get("sell", [{}] * LEVELS)

    row: dict = {
        "timestamp": ts,
        "symbol": symbol,
        "last_price": tick.get("last_price"),
    }
    for prefix, levels in (("bid", buys), ("ask", sells)):
        for i, lvl in enumerate(levels[:LEVELS], start=1):
            row[f"{prefix}_price_{i}"] = lvl.get("price")
            row[f"{prefix}_qty_{i}"] = lvl.get("quantity")
            row[f"{prefix}_orders_{i}"] = lvl.get("orders")
    return row


# ---------------------------------------------------------------------------
# Kite WebSocket callbacks
# ---------------------------------------------------------------------------
_buffer: ParquetBuffer | None = None
tokens = list(INSTRUMENTS.keys())


def on_ticks(ws, ticks):
    now = datetime.datetime.now(IST).time()
    if now < MARKET_OPEN or now > MARKET_CLOSE:
        return
    for tick in ticks:
        if tick.get("instrument_token") in INSTRUMENTS and tick.get("depth"):
            _buffer.add(_tick_to_row(tick))


def on_connect(ws, response):
    log.info(
        "Connected. Subscribing %d instruments in FULL mode: %s",
        len(tokens),
        list(INSTRUMENTS.values()),
    )
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)


def on_close(ws, code, reason):
    log.warning("WebSocket closed: %s %s. Kite will auto-reconnect.", code, reason)


def on_error(ws, code, reason):
    log.error("WebSocket error: %s %s", code, reason)


def on_reconnect(ws, attempt):
    log.info("Reconnecting... attempt %d", attempt)


def on_noreconnect(ws):
    log.error("Reconnection exhausted. Exiting.")
    if _buffer:
        _buffer.close()
    sys.exit(1)


def shutdown(sig, frame):
    log.info("Signal %s received. Flushing and exiting.", sig)
    if _buffer:
        _buffer.close()
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    now = datetime.datetime.now(IST)
    if now.weekday() >= 5:
        log.info("Weekend — NSE is closed. Exiting.")
        sys.exit(0)

    if now.time() > MARKET_CLOSE:
        log.info("Market already closed for today. Exiting.")
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    date_str = now.strftime("%Y%m%d")
    path = os.path.join(OUTPUT_DIR, f"lob_kite_{date_str}.parquet")
    _buffer = ParquetBuffer(path, SCHEMA)

    log.info("Starting Kite collector. Output: %s", path)

    kws = KiteTicker(API_KEY, ACCESS_TOKEN)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.on_reconnect = on_reconnect
    kws.on_noreconnect = on_noreconnect

    kws.connect(threaded=True)

    while True:
        if datetime.datetime.now(IST).time() > MARKET_CLOSE:
            log.info("Market closed. Finalising.")
            _buffer.close()
            sys.exit(0)
        time.sleep(10)
