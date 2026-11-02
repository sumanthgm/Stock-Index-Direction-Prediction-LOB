"""
Dhan 20-Level LOB Collector — Production EC2 Version
=====================================================
Streams 20-level bid/ask depth for NIFTY and BANKNIFTY index futures via
Dhan's twenty-depth WebSocket during NSE market hours (09:15–15:30 IST).
Writes one Parquet file per trading day (zstd compressed).

Credentials (.env):
    DHAN_CLIENT_ID=your_client_id
    DHAN_ACCESS_TOKEN=your_access_token   ← valid 30 days, no daily refresh needed

Setup on EC2:
    pip install websockets pyarrow python-dotenv boto3

Crontab (crontab -e):
    # Start collector at 09:10 IST (03:40 UTC) every weekday
    40 3 * * 1-5 cd /home/ubuntu/capstone/Data_Acquisition_and_preprocessing && python3 collect_dhan.py >> logs/dhan.log 2>&1
    # Sync to S3 at 15:40 IST (10:10 UTC) every weekday
    10 10 * * 1-5 bash /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/sync_to_s3.sh >> logs/sync.log 2>&1

Binary packet format (verified from packet analysis, May 2026):
    Full packet (1992B):
        bytes   0-39 : header (security_id at bytes 4-7, uint32 LE)
        bytes  40-359: BID  20×16B  [orders(4B)][price(8B f64)][qty(4B)]
        bytes 360-679: ASK  20×16B  [price(8B f64)][qty(4B)][orders(4B)]
        bytes 680+   : trailing fields (timestamps, session data)
    Compact packet (664B) = two 332B sub-packets:
        BID sub (bytes   0-331): header(12B) + 20×16B [price(8B f64)][qty(4B)][orders(4B)]
        ASK sub (bytes 332-663): header(12B) + 20×16B [price(8B f64)][qty(4B)][orders(4B)]
        (security_id at offset +4 within each sub-header)
    WebSocket messages pack multiple records — greedy parser tries full format first,
    then compact, consuming bytes left-to-right until fewer than 664 bytes remain.
"""

import asyncio
import datetime
import json
import logging
import os
import signal
import struct
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

if not CLIENT_ID or not ACCESS_TOKEN:
    sys.exit("ERROR: Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env")

# NIFTY and BANKNIFTY index futures — primary instruments for the capstone
# Expiry 2026-06-30 (Jun series). Roll to Jul before expiry.
# Jul series IDs: NIFTY=61093, BANKNIFTY=61088
INSTRUMENTS = {
    "62329": "NIFTY-JUN-FUT",  # Jul: 61093
    "62326": "BANKNIFTY-JUN-FUT",  # Jul: 61088
}

INSTRUMENT_SEGMENTS = {
    "62329": "NSE_FNO",
    "62326": "NSE_FNO",
}

LEVELS = 20
FLUSH_EVERY = 200
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
MARKET_OPEN = datetime.time(9, 15)
MARKET_CLOSE = datetime.time(15, 30)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "dhan")

WS_URL = f"wss://depth-api-feed.dhan.co/twentydepth?token={ACCESS_TOKEN}&clientId={CLIENT_ID}&authType=2"

SUBSCRIBE_MSG = {
    "RequestCode": 23,
    "InstrumentCount": len(INSTRUMENTS),
    "InstrumentList": [{"ExchangeSegment": INSTRUMENT_SEGMENTS[sid], "SecurityId": sid} for sid in INSTRUMENTS],
}

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
    ]
    for side in ("bid", "ask"):
        for i in range(1, levels + 1):
            fields += [
                pa.field(f"{side}_price_{i}", pa.float64()),
                pa.field(f"{side}_qty_{i}", pa.int64()),
                pa.field(f"{side}_orders_{i}", pa.int64()),
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
# Binary parser — verified format (see module docstring)
# ---------------------------------------------------------------------------
PACKET_FULL = 1992
PACKET_COMPACT = 664

# Full-format offsets
_FULL_SECURITY_ID_OFF = 4
_FULL_BID_START = 40
_FULL_ASK_START = 360
_FULL_LEVEL_SIZE = 16
_FULL_BID_ORDERS_OFF = 0  # uint32 within bid level
_FULL_BID_PRICE_OFF = 4  # float64 within bid level
_FULL_BID_QTY_OFF = 12  # uint32 within bid level
_FULL_ASK_PRICE_OFF = 0  # float64 within ask level
_FULL_ASK_QTY_OFF = 8  # uint32 within ask level
_FULL_ASK_ORDERS_OFF = 12  # uint32 within ask level

# Compact-format offsets
_COMPACT_SUBHDR = 12  # 12B sub-header per sub-packet
_COMPACT_LEVEL_SIZE = 16
_COMPACT_PRICE_OFF = 0  # float64 within level
_COMPACT_QTY_OFF = 8  # uint32 within level
_COMPACT_ORDERS_OFF = 12  # uint32 within level

PRICE_MIN, PRICE_MAX = 10.0, 200_000.0  # plausibility range for NSE futures


def _parse_full(data: bytes, offset: int) -> tuple | None:
    """Parse one 1992B full-format record at `offset`. Returns (sid, bids, asks)."""
    if offset + PACKET_FULL > len(data):
        return None
    sid = str(struct.unpack_from("<I", data, offset + _FULL_SECURITY_ID_OFF)[0])
    bids, asks = [], []
    for i in range(LEVELS):
        off = offset + _FULL_BID_START + i * _FULL_LEVEL_SIZE
        price = struct.unpack_from("<d", data, off + _FULL_BID_PRICE_OFF)[0]
        qty = struct.unpack_from("<I", data, off + _FULL_BID_QTY_OFF)[0]
        orders = struct.unpack_from("<I", data, off + _FULL_BID_ORDERS_OFF)[0]
        bids.append((price if PRICE_MIN <= price <= PRICE_MAX else None, qty, orders))
    for i in range(LEVELS):
        off = offset + _FULL_ASK_START + i * _FULL_LEVEL_SIZE
        price = struct.unpack_from("<d", data, off + _FULL_ASK_PRICE_OFF)[0]
        qty = struct.unpack_from("<I", data, off + _FULL_ASK_QTY_OFF)[0]
        orders = struct.unpack_from("<I", data, off + _FULL_ASK_ORDERS_OFF)[0]
        asks.append((price if PRICE_MIN <= price <= PRICE_MAX else None, qty, orders))
    return sid, bids, asks


def _parse_compact(data: bytes, offset: int) -> tuple | None:
    """Parse one 664B compact-format record at `offset`. Returns (sid, bids, asks)."""
    if offset + PACKET_COMPACT > len(data):
        return None
    bid_sub_start = offset
    ask_sub_start = offset + 332
    sid = str(struct.unpack_from("<I", data, bid_sub_start + 4)[0])
    bids, asks = [], []
    for sub_start, levels_list in [(bid_sub_start, bids), (ask_sub_start, asks)]:
        lvl_start = sub_start + _COMPACT_SUBHDR
        for i in range(LEVELS):
            off = lvl_start + i * _COMPACT_LEVEL_SIZE
            price = struct.unpack_from("<d", data, off + _COMPACT_PRICE_OFF)[0]
            qty = struct.unpack_from("<I", data, off + _COMPACT_QTY_OFF)[0]
            orders = struct.unpack_from("<I", data, off + _COMPACT_ORDERS_OFF)[0]
            levels_list.append((price if PRICE_MIN <= price <= PRICE_MAX else None, qty, orders))
    return sid, bids, asks


def _l1_plausible(bids: list, asks: list) -> bool:
    """Check only L1 bid < L1 ask to avoid false negatives on deep levels."""
    b1 = bids[0][0]
    a1 = asks[0][0]
    if b1 is None or a1 is None:
        return False
    return b1 < a1


def parse_message(data: bytes) -> list[tuple]:
    """
    Greedy parse of a WebSocket message containing 1+ records.
    Returns list of (security_id_str, bids, asks).
    """
    records = []
    offset = 0
    while offset + PACKET_COMPACT <= len(data):
        # Try full format first (it's larger and takes priority)
        if offset + PACKET_FULL <= len(data):
            result = _parse_full(data, offset)
            if result and result[0] in INSTRUMENTS and _l1_plausible(result[1], result[2]):
                records.append(result)
                offset += PACKET_FULL
                continue

        # Try compact format
        result = _parse_compact(data, offset)
        if result and result[0] in INSTRUMENTS and _l1_plausible(result[1], result[2]):
            records.append(result)
            offset += PACKET_COMPACT
            continue

        # Neither matched — skip one byte and try again (avoids infinite loop)
        offset += 1

    return records


def _build_row(sid: str, bids: list, asks: list) -> dict:
    row: dict = {
        "timestamp": datetime.datetime.now(IST),
        "symbol": INSTRUMENTS[sid],
    }
    for prefix, levels in (("bid", bids), ("ask", asks)):
        for i, (price, qty, orders) in enumerate(levels, start=1):
            row[f"{prefix}_price_{i}"] = price
            row[f"{prefix}_qty_{i}"] = qty
            row[f"{prefix}_orders_{i}"] = orders
    return row


# ---------------------------------------------------------------------------
# WebSocket collector with reconnection
# ---------------------------------------------------------------------------
async def collect(buf: ParquetBuffer):
    import websockets
    import websockets.exceptions

    reconnect_delay = 5
    total_records = 0

    while True:
        now_ist = datetime.datetime.now(IST)
        if now_ist.time() > MARKET_CLOSE:
            log.info("Market closed. Stopping collector.")
            return

        try:
            log.info("Connecting to Dhan twenty-depth feed...")
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=30, open_timeout=15) as ws:
                await ws.send(json.dumps(SUBSCRIBE_MSG))
                log.info(
                    "Subscribed to %d instruments: %s",
                    len(INSTRUMENTS),
                    list(INSTRUMENTS.values()),
                )
                reconnect_delay = 5

                async for message in ws:
                    if datetime.datetime.now(IST).time() > MARKET_CLOSE:
                        log.info("Market closed. Stopping collector.")
                        return
                    if not isinstance(message, bytes):
                        continue
                    records = parse_message(message)
                    for rec in records:
                        buf.add(_build_row(*rec))
                        total_records += 1
                    if total_records % 1000 == 0 and total_records > 0:
                        log.info("Records received: %d", total_records)

        except websockets.exceptions.ConnectionClosedError as e:
            # Dhan server sends 1011 keepalive ping timeout at market close — normal
            if "keepalive" in str(e).lower() or "1011" in str(e):
                log.info("Server closed connection (%s). Likely market close.", e)
                if datetime.datetime.now(IST).time() >= MARKET_CLOSE:
                    return
            else:
                log.warning("Connection closed: %s. Reconnecting in %ds.", e, reconnect_delay)
        except Exception as e:
            log.warning("Connection error: %s. Reconnecting in %ds.", e, reconnect_delay)

        buf.flush()
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_buf: ParquetBuffer | None = None


def shutdown(sig, frame):
    log.info("Signal %s received. Flushing and exiting.", sig)
    if _buf:
        _buf.close()
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

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
    path = os.path.join(OUTPUT_DIR, f"lob_dhan_{date_str}.parquet")
    _buf = ParquetBuffer(path, SCHEMA)

    log.info("Starting Dhan collector. Output: %s", path)
    log.info("Instruments: %s", INSTRUMENTS)

    try:
        asyncio.run(collect(_buf))
    finally:
        _buf.close()
        log.info("Collector finished.")
