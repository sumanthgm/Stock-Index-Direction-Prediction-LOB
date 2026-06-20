"""
Notebook environment helpers — make the notebooks run on **Kaggle** as well as Colab.

After the repo is cloned and `modeling/` is on sys.path, notebooks do:
    from nbenv import get_secret, find_fi2010, detect_env

Differences handled:
  - Secrets: Kaggle `UserSecretsClient` vs Colab `userdata` vs plain env vars.
  - FI-2010 location: Kaggle mounts an attached dataset read-only under /kaggle/input
    (no download); Colab downloads via the Kaggle CLI into ./fi2010_data.
"""

from __future__ import annotations

import os
import pathlib
import zipfile


def detect_env() -> str:
    """Return 'kaggle', 'colab', or 'local'."""
    if os.path.isdir("/kaggle"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        return "local"


def get_secret(name: str) -> str:
    """Read a secret: Kaggle UserSecrets -> Colab userdata -> env var. '' if absent."""
    try:
        from kaggle_secrets import UserSecretsClient

        v = UserSecretsClient().get_secret(name)
        if v:
            return v
    except Exception:
        pass
    try:
        from google.colab import userdata

        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name, "")


def _zscore_under(root) -> pathlib.Path | None:
    root = pathlib.Path(root)
    if not root.exists():
        return None
    if (root / "NoAuction_Zscore_Training").exists():
        return root
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "noauction_zscore_training":
            return p.parent
    for p in root.rglob("*"):
        if p.is_dir() and "zscore" in p.name.lower() and "noauction" in p.name.lower():
            return p
    return None


def find_fi2010(roots=None) -> str:
    """Locate the NoAuction_Zscore folder across Kaggle/Colab locations.

    On **Kaggle**: attach the FI-2010 dataset (Add Data) — it mounts under /kaggle/input.
    On **Colab**: the Kaggle CLI downloads into ./fi2010_data (we also extract any nested zip).
    Returns the folder path as a string, or '' if not found.
    """
    roots = roots or ["/kaggle/input", "fi2010_data", "."]
    dd = pathlib.Path("fi2010_data")
    if dd.exists():
        for z in dd.rglob("*.zip"):  # extract a nested zip the CLI left behind
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.parent)
            except Exception:
                pass
    for r in roots:
        found = _zscore_under(r)
        if found:
            return str(found)
    return ""


# --- S3 sync helpers (cross-timeout resumability for the experiment notebooks) -------------
def s3_client(region: str = "ap-south-2"):
    """boto3 S3 client using AWS_* from secrets/env, or None if creds/boto3 absent
    (notebooks then keep results locally only)."""
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        os.environ.setdefault(k, get_secret(k))
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print(
            "No AWS creds -> S3 sync OFF (results kept locally only). Add AWS_* secrets."
        )
        return None
    try:
        import boto3

        return boto3.client("s3", region_name=region)
    except Exception as e:
        print("boto3 unavailable -> S3 sync OFF:", repr(e))
        return None


def s3_put(
    client, localpath, bucket: str = "lob-capstone-data", prefix: str = ""
) -> None:
    """Upload one file; key = '<prefix>/<localpath>'. No-op if client is None."""
    if client is None:
        return
    try:
        client.upload_file(str(localpath), bucket, f"{prefix}/{localpath}")
        print(f"   ^ s3://{bucket}/{prefix}/{localpath}")
    except Exception as e:
        print("   (s3 put skipped)", repr(e))


def s3_pull(
    client, localpath, bucket: str = "lob-capstone-data", prefix: str = ""
) -> bool:
    """Download '<prefix>/<localpath>' to localpath (to resume). False if absent/None."""
    if client is None:
        return False
    try:
        pathlib.Path(localpath).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, f"{prefix}/{localpath}", str(localpath))
        print(f"   pulled s3://{bucket}/{prefix}/{localpath} (resuming)")
        return True
    except Exception:
        return False


# --- Clean area-based layout: experiments/<area>/{results,figures,checkpoints}/<file> ---
def _exp_key(area: str, localpath) -> str:
    name = pathlib.Path(str(localpath)).name
    if name.endswith(".pt"):
        kind = "checkpoints"
    elif name.endswith(".png"):
        kind = "figures"
    else:
        kind = "results"
    return f"experiments/{area}/{kind}/{name}"


def s3_put_area(
    client, localpath, area: str, bucket: str = "lob-capstone-data"
) -> None:
    """Upload one file into the clean layout experiments/<area>/<kind>/<basename>,
    routing by extension (.pt->checkpoints, .png->figures, else results)."""
    if client is None:
        return
    key = _exp_key(area, localpath)
    try:
        client.upload_file(str(localpath), bucket, key)
        print(f"   ^ s3://{bucket}/{key}")
    except Exception as e:
        print("   (s3 put skipped)", repr(e))


def s3_pull_area(
    client, localpath, area: str, bucket: str = "lob-capstone-data"
) -> bool:
    """Download experiments/<area>/<kind>/<basename> to localpath (to resume)."""
    if client is None:
        return False
    key = _exp_key(area, localpath)
    try:
        pathlib.Path(localpath).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(localpath))
        print(f"   pulled s3://{bucket}/{key} (resuming)")
        return True
    except Exception:
        return False
