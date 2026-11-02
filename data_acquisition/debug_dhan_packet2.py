"""
Targeted Dhan Packet Parser — Stage 2
=======================================
We know ask prices start at offset 360 (16B per level, price-first float64).
This script probes where the BID prices are and validates the full structure.

Usage:
    conda run -n pes_env python3 debug_dhan_packet2.py
"""

import asyncio
import json
import os
import struct

from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

INSTRUMENTS = {
    "1333": "HDFCBANK",
    "10604": "INFY",
    "2885": "RELIANCE",
}

WS_URL = f"wss://depth-api-feed.dhan.co/twentydepth?token={ACCESS_TOKEN}&clientId={CLIENT_ID}&authType=2"

SUBSCRIBE_MSG = {
    "RequestCode": 23,
    "InstrumentCount": len(INSTRUMENTS),
    "InstrumentList": [{"ExchangeSegment": "NSE_EQ", "SecurityId": sid} for sid in INSTRUMENTS],
}

PRICE_MIN, PRICE_MAX = 50.0, 50_000.0

# Known: ask starts at offset 360, 16B per level, price = float64 at level+0
ASK_START = 360
LEVEL_SIZE = 16

# Candidate BID start offsets to probe (ask ends at 360 + 20×16 = 680)
# Try: 680 (right after ask), and also 40 with price at offset +8
BID_CANDIDATES = [40, 680, 700, 720, 740, 760]


async def capture_single_record() -> bytes:
    """Get one 1992-byte packet (skip 3984-byte doubles)."""
    import websockets

    async with websockets.connect(WS_URL, ping_interval=20, open_timeout=15) as ws:
        await ws.send(json.dumps(SUBSCRIBE_MSG))
        print("Connected. Waiting for a single-instrument packet (1992B)...")
        async for msg in ws:
            if not isinstance(msg, bytes):
                continue
            if len(msg) == 1992:
                return msg
            if len(msg) == 3984:
                # Two records — return first one
                return msg[:1992]


def read_levels(pkt: bytes, start: int, n: int, level_size: int, price_offset: int = 0) -> list:
    """Read n levels starting at `start`, price at level+price_offset (float64)."""
    prices = []
    for i in range(n):
        off = start + i * level_size + price_offset
        if off + 8 > len(pkt):
            break
        p = struct.unpack_from("<d", pkt, off)[0]
        prices.append(p)
    return prices


def is_sequential(prices: list, tolerance: float = 1.0) -> bool:
    """Check if prices are monotonically increasing or decreasing with small steps."""
    plausible = [p for p in prices if PRICE_MIN <= p <= PRICE_MAX]
    if len(plausible) < 5:
        return False
    diffs = [plausible[i + 1] - plausible[i] for i in range(len(plausible) - 1)]
    # All diffs same sign and small (< ₹5 per level)
    return all(abs(d) < tolerance and d == diffs[0] for d in diffs)


def analyse(pkt: bytes):
    print(f"\nPacket size: {len(pkt)}B")

    # Security ID
    sec_id = struct.unpack_from("<I", pkt, 4)[0]
    sym = INSTRUMENTS.get(str(sec_id), f"unknown({sec_id})")
    print(f"Security ID at bytes 4-7: {sec_id} = {sym}")

    # Confirm ask at 360
    ask_prices = read_levels(pkt, ASK_START, 20, LEVEL_SIZE, price_offset=0)
    plausible_ask = [p for p in ask_prices if PRICE_MIN <= p <= PRICE_MAX]
    print(f"\n=== CONFIRMED: Ask at offset {ASK_START}, level_size={LEVEL_SIZE}B, price at +0 ===")
    print(f"  Ask L1-L10: {[f'{p:.2f}' for p in ask_prices[:10]]}")
    print(f"  Plausible: {len(plausible_ask)}/20")

    # Probe bid candidates with price at +0
    print("\n=== Probing BID locations (price at +0 within 16B level) ===")
    for bid_start in BID_CANDIDATES:
        prices = read_levels(pkt, bid_start, 20, LEVEL_SIZE, price_offset=0)
        plausible = [p for p in prices if PRICE_MIN <= p <= PRICE_MAX]
        flag = " ← SEQUENTIAL!" if is_sequential(prices) else ""
        if len(plausible) >= 5 or flag:
            print(
                f"  bid_start={bid_start:4d}: {[f'{p:.2f}' for p in prices[:10]]}  ({len(plausible)}/20 plausible){flag}"
            )

    # Probe bid with price at +8 (format: qty(8B) + price(8B))
    print("\n=== Probing BID locations (price at +8 within 16B level) ===")
    for bid_start in [40, 100, 360, 680, 1000, 1040]:
        prices = read_levels(pkt, bid_start, 20, LEVEL_SIZE, price_offset=8)
        plausible = [p for p in prices if PRICE_MIN <= p <= PRICE_MAX]
        flag = " ← SEQUENTIAL!" if is_sequential(prices) else ""
        if len(plausible) >= 5 or flag:
            print(
                f"  bid_start={bid_start:4d} (+8): {[f'{p:.2f}' for p in prices[:10]]}  ({len(plausible)}/20 plausible){flag}"
            )

    # Probe bid with 24B levels
    print("\n=== Probing BID (24B per level, price at +0) ===")
    for bid_start in [40, 360, 680, 740, 800]:
        prices = read_levels(pkt, bid_start, 20, 24, price_offset=0)
        plausible = [p for p in prices if PRICE_MIN <= p <= PRICE_MAX]
        flag = " ← SEQUENTIAL!" if is_sequential(prices, tolerance=5.0) else ""
        if len(plausible) >= 5 or flag:
            print(
                f"  bid_start={bid_start:4d} (24B): {[f'{p:.2f}' for p in prices[:10]]}  ({len(plausible)}/20 plausible){flag}"
            )

    # Wide sequential search: scan entire packet for monotone sequences of 10+
    print("\n=== Wide sequential search (any offset, 8B steps) ===")
    window = 10
    for start in range(0, len(pkt) - window * 8, 4):
        candidates = []
        for i in range(window):
            off = start + i * 8
            if off + 8 > len(pkt):
                break
            p = struct.unpack_from("<d", pkt, off)[0]
            if PRICE_MIN <= p <= PRICE_MAX:
                candidates.append((off, p))
        if len(candidates) >= window:
            # Check if monotone
            prices_only = [p for _, p in candidates]
            diffs = [prices_only[i + 1] - prices_only[i] for i in range(len(prices_only) - 1)]
            if diffs and all(abs(d - diffs[0]) < 0.001 and abs(diffs[0]) < 2.0 for d in diffs):
                direction = "ASC" if diffs[0] > 0 else "DESC"
                print(f"  Monotone sequence at offset {start}: {prices_only} ({direction}, step={diffs[0]:.3f})")

    # Print hex of potential bid region (680-800)
    print("\n=== Hex dump: offset 680-800 (expected bid region) ===")
    for i in range(680, min(800, len(pkt)), 16):
        chunk = pkt[i : i + 16]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        # Try to decode as two float64s
        if len(chunk) >= 16:
            p1 = struct.unpack_from("<d", chunk, 0)[0]
            p2 = struct.unpack_from("<d", chunk, 8)[0]
            note = ""
            if PRICE_MIN <= p1 <= PRICE_MAX:
                note += f" f64[0]={p1:.4f}"
            if PRICE_MIN <= p2 <= PRICE_MAX:
                note += f" f64[8]={p2:.4f}"
            print(f"  {i:4d}: {hex_str:<47s}{note}")

    # Print hex of region 40-160 (possible abbreviated header depth)
    print("\n=== Hex dump: offset 40-160 (header / abbreviated depth) ===")
    for i in range(40, min(160, len(pkt)), 16):
        chunk = pkt[i : i + 16]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        if len(chunk) >= 16:
            p1 = struct.unpack_from("<d", chunk, 0)[0]
            p2 = struct.unpack_from("<d", chunk, 8)[0]
            note = ""
            if PRICE_MIN <= p1 <= PRICE_MAX:
                note += f" f64[0]={p1:.4f}"
            if PRICE_MIN <= p2 <= PRICE_MAX:
                note += f" f64[8]={p2:.4f}"
            print(f"  {i:4d}: {hex_str:<47s}{note}")


async def main():
    pkt = await capture_single_record()
    analyse(pkt)

    # Save for offline analysis
    with open("data/dhan/debug_packet2.bin", "wb") as f:
        f.write(pkt)
    print("\nPacket saved to data/dhan/debug_packet2.bin")


if __name__ == "__main__":
    asyncio.run(main())
