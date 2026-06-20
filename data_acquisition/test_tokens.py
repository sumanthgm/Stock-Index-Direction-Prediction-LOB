"""
Token Connectivity Test — Verify Dhan and Kite access tokens work.
==================================================================
Hits each broker's lightweight auth-protected endpoint and reports PASS/FAIL.

Usage:
    python3 test_tokens.py

Reads from local .env: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, KITE_API_KEY, KITE_ACCESS_TOKEN.
"""

import os
import sys

import requests
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")


def test_dhan() -> bool:
    print("\n[dhan]  Testing token...")
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        print("[dhan]  FAIL — DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN missing from .env")
        return False
    try:
        r = requests.get(
            "https://api.dhan.co/v2/fundlimit",
            headers={
                "access-token": DHAN_ACCESS_TOKEN,
                "client-id": DHAN_CLIENT_ID,
                "Accept": "application/json",
            },
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[dhan]  FAIL — network error: {e}")
        return False
    if r.status_code == 200:
        print(f"[dhan]  PASS — client {DHAN_CLIENT_ID} authenticated (HTTP 200)")
        return True
    print(f"[dhan]  FAIL — HTTP {r.status_code}: {r.text[:200]}")
    return False


def test_kite() -> bool:
    print("\n[kite]  Testing token...")
    if not KITE_API_KEY or not KITE_ACCESS_TOKEN:
        print("[kite]  FAIL — KITE_API_KEY or KITE_ACCESS_TOKEN missing from .env")
        return False
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(KITE_ACCESS_TOKEN)
        profile = kite.profile()
    except Exception as e:
        print(f"[kite]  FAIL — {type(e).__name__}: {e}")
        return False
    user_id = profile.get("user_id", "?")
    name = profile.get("user_name", "?")
    print(f"[kite]  PASS — authenticated as {user_id} ({name})")
    return True


def main():
    print("Token Connectivity Test — Dhan + Kite")
    dhan_ok = test_dhan()
    kite_ok = test_kite()
    print("\n" + "=" * 50)
    print(f"  Dhan: {'PASS' if dhan_ok else 'FAIL'}")
    print(f"  Kite: {'PASS' if kite_ok else 'FAIL'}")
    print("=" * 50)
    sys.exit(0 if (dhan_ok and kite_ok) else 1)


if __name__ == "__main__":
    main()
