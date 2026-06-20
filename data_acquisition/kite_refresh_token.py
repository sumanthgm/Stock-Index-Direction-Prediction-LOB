"""
Daily Token Refresh — Run Locally Before 09:15 IST Each Trading Day
====================================================================
Refreshes both Kite and Dhan access tokens, then pushes the local .env to EC2 via scp.

Usage:
    python3 kite_refresh_token.py

Prerequisites:
    pip install kiteconnect python-dotenv
    Set in local .env: KITE_API_KEY, KITE_API_SECRET, EC2_HOST, EC2_KEY_PATH

Dhan token: get from dhan.co → My Profile → API Access → Generate Token.
Kite token: obtained via browser OAuth flow (script prints URL).
"""

import getpass
import os
import re
import subprocess

from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
EC2_HOST = os.environ.get("EC2_HOST", "")
EC2_KEY_PATH = os.environ.get("EC2_KEY_PATH", "")
EC2_ENV_PATH = os.environ.get(
    "EC2_ENV_PATH",
    "/home/ubuntu/capstone/Data_Acquisition_and_preprocessing/.env",
)

LOCAL_ENV = os.path.join(os.path.dirname(__file__), ".env")

# Colon-separated extra .env paths to keep in sync (e.g. another project that
# uses the same Dhan/Kite tokens). Set in your local .env, e.g.:
#   EXTRA_ENV_PATHS=/Users/joseph.raj/Documents/personal/tradetron/.env
EXTRA_ENV_PATHS = [
    os.path.expanduser(p.strip())
    for p in os.environ.get("EXTRA_ENV_PATHS", "").split(":")
    if p.strip()
]

_any_token_updated = False


def _update_token_everywhere(key: str, value: str):
    """Write KEY=value to LOCAL_ENV and every path in EXTRA_ENV_PATHS."""
    _update_env_file(LOCAL_ENV, key, value)
    print(f"[local] Updated {key}")
    for extra in EXTRA_ENV_PATHS:
        if not os.path.exists(extra):
            print(f"[sync]  Skipped {extra} — file not found")
            continue
        _update_env_file(extra, key, value)
        print(f"[sync]  Updated {key} in {extra}")


def _update_env_file(path: str, key: str, value: str):
    """Replace or append KEY=value in an .env file."""
    try:
        content = open(path).read()
    except FileNotFoundError:
        content = ""
    pattern = rf"^{re.escape(key)}=.*$"
    new_line = f"{key}={value}"
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, new_line, content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\n{new_line}\n"
    with open(path, "w") as f:
        f.write(content)


def _push_env_to_ec2():
    """Copy the local .env to EC2 via scp — avoids shell-quoting issues with JWT tokens."""
    if not EC2_HOST or not EC2_KEY_PATH:
        print("[ec2]   EC2_HOST/EC2_KEY_PATH not set — skipping remote sync.")
        print(f"        Manually copy {LOCAL_ENV} to {EC2_HOST}:{EC2_ENV_PATH}")
        return
    ssh_key = os.path.expanduser(EC2_KEY_PATH)
    cmd = [
        "scp",
        "-i",
        ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
        LOCAL_ENV,
        f"{EC2_HOST}:{EC2_ENV_PATH}",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
        print(f"[ec2]   Synced .env to {EC2_HOST}:{EC2_ENV_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"[ec2]   scp failed: {e}")
    except subprocess.TimeoutExpired:
        print("[ec2]   scp timed out — check your SSH connection.")


def refresh_dhan():
    global _any_token_updated
    print("\n" + "=" * 70)
    print("DHAN TOKEN REFRESH")
    print("  1. Go to: https://web.dhan.co → My Profile → API Access")
    print("  2. Click 'Generate Token'")
    print("  3. Paste it below.")
    print("=" * 70)
    token = getpass.getpass(
        "\nPaste Dhan access token (hidden, blank to skip): "
    ).strip()
    if not token:
        print("[dhan]  Skipped.")
        return
    _update_token_everywhere("DHAN_ACCESS_TOKEN", token)
    _any_token_updated = True


def refresh_kite():
    global _any_token_updated
    if not KITE_API_KEY or not KITE_API_SECRET:
        print(
            "[kite]  KITE_API_KEY or KITE_API_SECRET not set — skipping Kite refresh."
        )
        return
    kite = KiteConnect(api_key=KITE_API_KEY)
    login_url = kite.login_url()

    print("\n" + "=" * 70)
    print("KITE TOKEN REFRESH")
    print("  1. Open this URL in your browser and log in:")
    print(f"\n     {login_url}\n")
    print("  2. After login you'll be redirected to a URL like:")
    print("       https://127.0.0.1/?request_token=XXXX&action=login&status=success")
    print("  3. Copy the request_token value.")
    print("=" * 70)

    request_token = getpass.getpass(
        "\nPaste request_token (hidden, blank to skip): "
    ).strip()
    if not request_token:
        print("[kite]  Skipped.")
        return
    data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    access_token = data["access_token"]
    print(f"[kite]  Access token obtained: {access_token[:8]}...")
    _update_token_everywhere("KITE_ACCESS_TOKEN", access_token)
    _any_token_updated = True


def main():
    print("\nDaily Token Refresh — Dhan + Kite")
    refresh_dhan()
    refresh_kite()
    if _any_token_updated:
        _push_env_to_ec2()
        print("\nDone. Collectors on EC2 will use new tokens on next connection.\n")
    else:
        print("\nNo tokens updated.\n")


if __name__ == "__main__":
    main()
