"""
Dhan Packet Format Diagnostics
================================
Captures 3 raw packets from the Dhan twenty-depth WebSocket and systematically
tries every reasonable binary interpretation to find where the prices live.

Run this to figure out the correct HEADER_SIZE and level format for test_dhan_local.py.

Usage:
    cd Data_Acquisition_and_preprocessing
    conda run -n pes_env python3 debug_dhan_packet.py
"""

import asyncio
import json
import os
import struct
import sys

from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

if not CLIENT_ID or not ACCESS_TOKEN:
    sys.exit("ERROR: Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env")

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

# NSE large-cap price range (plausibility check)
PRICE_MIN, PRICE_MAX = 50.0, 50_000.0


# ---------------------------------------------------------------------------
# Packet capture
# ---------------------------------------------------------------------------
async def capture_packets(n: int = 3) -> list[bytes]:
    import websockets

    packets = []
    print(f"Connecting to grab {n} binary packets...")

    async with websockets.connect(WS_URL, ping_interval=20, open_timeout=15) as ws:
        await ws.send(json.dumps(SUBSCRIBE_MSG))
        async for msg in ws:
            if not isinstance(msg, bytes):
                print(f"  [text msg] {msg[:120]}")
                continue
            packets.append(msg)
            print(f"  Captured packet {len(packets)}: {len(msg)} bytes")
            if len(packets) >= n:
                break

    return packets


# ---------------------------------------------------------------------------
# Hex dump
# ---------------------------------------------------------------------------
def hex_dump(data: bytes, max_bytes: int = 256):
    for i in range(0, min(max_bytes, len(data)), 16):
        chunk = data[i : i + 16]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        asc_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:4d}:  {hex_str:<47s}  {asc_str}")


# ---------------------------------------------------------------------------
# Scan for plausible prices in every interpretation
# ---------------------------------------------------------------------------
def scan_prices(data: bytes, label: str):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    found = []

    # float64 every 4 bytes
    for off in range(0, len(data) - 7, 4):
        v = struct.unpack_from("<d", data, off)[0]
        if PRICE_MIN <= v <= PRICE_MAX and v == v:  # not NaN
            found.append(("f64", off, v))

    # float32 every 4 bytes
    for off in range(0, len(data) - 3, 4):
        v = struct.unpack_from("<f", data, off)[0]
        if PRICE_MIN <= v <= PRICE_MAX and v == v:
            found.append(("f32", off, v))

    # int32 / 100 (paise)
    for off in range(0, len(data) - 3, 4):
        v = struct.unpack_from("<i", data, off)[0]
        p = v / 100.0
        if PRICE_MIN <= p <= PRICE_MAX:
            found.append(("i32/100", off, p))

    # int32 / 10 (0.1 rupee ticks)
    for off in range(0, len(data) - 3, 4):
        v = struct.unpack_from("<i", data, off)[0]
        p = v / 10.0
        if PRICE_MIN <= p <= PRICE_MAX:
            if not any(x[1] == off and x[0] == "i32/100" for x in found):
                found.append(("i32/10", off, p))

    if found:
        print(f"  *** {len(found)} plausible price(s) found:")
        for fmt, off, val in sorted(set(found)):
            print(f"      offset={off:4d}  fmt={fmt:8s}  value={val:.4f}")
    else:
        print("  No plausible prices found in this range.")

    return found


# ---------------------------------------------------------------------------
# Try repeating structure — look for a grid of prices
# ---------------------------------------------------------------------------
def try_structure(data: bytes, header: int, level_size: int, n_levels: int, fmt: str):
    """
    Attempt to parse data as: HEADER + n_levels bid levels + n_levels ask levels.
    Each level starts with a price field (fmt = '<d' or '<f' or '<i').
    Prints the first 5 bid prices and first 5 ask prices.
    """
    price_size = struct.calcsize(fmt)
    total_data = n_levels * 2 * level_size
    if header + total_data > len(data):
        return

    prices = []
    for side in range(2):
        side_prices = []
        for lvl in range(n_levels):
            off = header + (side * n_levels + lvl) * level_size
            if off + price_size > len(data):
                break
            raw = struct.unpack_from(fmt, data, off)[0]
            if fmt == "<i":
                raw /= 100.0
            side_prices.append(raw)
        prices.append(side_prices)

    bid_plausible = sum(1 for p in prices[0] if PRICE_MIN <= p <= PRICE_MAX)
    ask_plausible = sum(1 for p in prices[1] if PRICE_MIN <= p <= PRICE_MAX)
    total_plausible = bid_plausible + ask_plausible

    if total_plausible > 3:
        label = f"*** MATCH: header={header}B  level_size={level_size}B  n_levels={n_levels}  price_fmt={fmt}"
        print(f"\n  {label}")
        print(f"    bid L1-L5: {[f'{p:.2f}' for p in prices[0][:5]]}")
        print(f"    ask L1-L5: {[f'{p:.2f}' for p in prices[1][:5]]}")
        print(f"    plausible bid={bid_plausible}/{n_levels}  ask={ask_plausible}/{n_levels}")
        return True
    return False


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyse(pkt: bytes, idx: int):
    size = len(pkt)
    print(f"\n{'=' * 70}")
    print(f"  PACKET {idx}  ({size} bytes)")
    print(f"{'=' * 70}")

    # Hex dump
    print("\nHex dump (first 128 bytes):")
    hex_dump(pkt, max_bytes=128)

    # Find security ID
    print("\nSecurity ID search (uint32 LE, every 4 bytes, first 64 bytes):")
    for off in range(0, min(64, size - 3), 1):
        v = struct.unpack_from("<I", pkt, off)[0]
        if v in (1333, 10604, 2885):
            sym = {"1333": "HDFCBANK", "10604": "INFY", "2885": "RELIANCE"}.get(str(v), "?")
            print(f"  Offset {off}: security_id={v} ({sym})")

    # Scan entire packet for plausible prices
    print("\nPrice scan (entire packet):")
    all_found = scan_prices(pkt, f"Full {size}-byte packet")

    # Factor analysis
    print(f"\nPacket size factoring: {size}")
    for h in [4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 72, 80, 100, 120]:
        rem = size - h
        if rem <= 0:
            continue
        for n in [10, 20, 30, 40]:
            if rem % (n * 2) == 0:
                lvl_size = rem // (n * 2)
                print(
                    f"  header={h:3d}B  levels/side={n:2d}  level_size={lvl_size:3d}B  "
                    f"(total_depth={n * 2 * lvl_size}B)"
                )

    # Brute-force structure search
    print("\nStructure search (looking for repeating price sequence):")
    found_any = False
    for header in [4, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 72, 80, 100]:
        for n_levels in [10, 20, 30]:
            for level_size in [8, 12, 16, 20, 24, 32, 48, 49, 50]:
                for price_fmt in ["<d", "<f", "<i"]:
                    if try_structure(pkt, header, level_size, n_levels, price_fmt):
                        found_any = True

    if not found_any:
        print("  No matching structure found. Try the dhanhq official library.")
        print("  Or print the full packet to check manually with:")
        print("  python3 -c \"import sys; data=open(sys.argv[1],'rb').read(); print(data.hex())\" packet.bin")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    packets = await capture_packets(n=3)
    if not packets:
        print("\nNo packets received. Check credentials and market hours.")
        return

    # Save first packet to file for manual inspection
    os.makedirs("data/dhan", exist_ok=True)
    with open("data/dhan/debug_packet.bin", "wb") as f:
        f.write(packets[0])
    print("\nFirst packet saved to data/dhan/debug_packet.bin")

    # Analyse first packet only (all packets should have same structure)
    analyse(packets[0], 1)

    # Check if multiple packets have same structure
    sizes = [len(p) for p in packets]
    print(f"\n\nPacket sizes seen: {sizes}")
    if len(set(sizes)) > 1:
        print("  Multiple sizes → check if large = 2× small (two instruments per message)")
        for s in set(sizes):
            if s * 2 in sizes:
                print(f"  {s}B × 2 = {s * 2}B ✓ — large packets are two records back-to-back")


if __name__ == "__main__":
    asyncio.run(main())
