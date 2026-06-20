"""
Colab Quickstart — Mamba-LOB on FI-2010
========================================
Copy each numbered section into a separate Colab cell and run in order.
Runtime: Runtime → Change runtime type → T4 GPU (or A100 if available).

File layout expected in Colab working directory:
    /content/
        fi2010_dataset.py   ← upload from repo
        models.py           ← upload from repo
        train.py            ← upload from repo
        fi2010/             ← downloaded below
"""

# =============================================================================
# CELL 1 — Install dependencies
# =============================================================================
"""
!pip install -q torch torchvision scikit-learn pandas pyarrow
!pip install -q mamba-ssm causal-conv1d   # CUDA Mamba kernel

# Verify GPU + Mamba
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT FOUND")

try:
    from mamba_ssm import Mamba
    print("mamba-ssm: OK")
except ImportError:
    print("mamba-ssm: not available — MambaLOB will use pure-PyTorch fallback (slower)")
"""

# =============================================================================
# CELL 2 — Upload repo files to Colab
# =============================================================================
"""
# Option A: clone from GitHub (if repo is pushed)
#   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO /content/capstone
#   %cd /content/capstone/Modeling

# Option B: upload files manually
from google.colab import files
print("Upload fi2010_dataset.py, models.py, train.py from the modeling/ folder")
files.upload()   # select all three files
"""

# =============================================================================
# CELL 3 — Download FI-2010 dataset
# =============================================================================
"""
# FI-2010 is on Kaggle. You need a Kaggle API token (kaggle.json).
# Get it: kaggle.com → Account → Create API Token → download kaggle.json

from google.colab import files
print("Upload your kaggle.json")
files.upload()

import os, shutil
os.makedirs("/root/.kaggle", exist_ok=True)
shutil.copy("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)

# Download the dataset (search for "FI-2010" on Kaggle to confirm the exact slug)
!kaggle datasets download -d pavelfatin/stockmarket -p /content/fi2010 --unzip

# If the above slug is wrong, try:
#   !kaggle datasets list --search "FI-2010 limit order book"
# and use the correct dataset identifier.

import os
for root, dirs, files_list in os.walk("/content/fi2010"):
    for f in files_list:
        if f.endswith(".txt"):
            print(os.path.join(root, f))
"""

# =============================================================================
# CELL 4 — Locate the NoAuction_Zscore folder
# =============================================================================
"""
import glob

# Find all NoAuction_Zscore directories
candidates = glob.glob("/content/fi2010/**/NoAuction*Zscore*", recursive=True)
candidates += glob.glob("/content/fi2010/**/1.NoAuction*", recursive=True)

# Print unique directories containing .txt files
data_dirs = set()
for path in candidates:
    if os.path.isdir(path):
        if any(f.endswith(".txt") for f in os.listdir(path)):
            data_dirs.add(path)

print("Candidate data directories:")
for d in sorted(data_dirs):
    files_in = [f for f in os.listdir(d) if f.endswith(".txt")]
    print(f"  {d}  ({len(files_in)} txt files)")

# Set DATA_DIR to the correct one
DATA_DIR = sorted(data_dirs)[0]   # adjust if needed
print(f"\\nUsing: {DATA_DIR}")

# Verify label encoding: print unique label values in one file
import numpy as np
sample = sorted(glob.glob(f"{DATA_DIR}/*.txt"))[0]
data = np.loadtxt(sample)
print(f"File: {os.path.basename(sample)}, shape: {data.shape}")
print(f"Unique labels (last col): {np.unique(data[:, -1])}")
# Expected: [1. 2. 3.]  → Down=1, Stat=2, Up=3  (code remaps to 0,1,2)
# If you see [-1. 0. 1.] instead, set label_offset=1 in load_fi2010 calls below
"""

# =============================================================================
# CELL 5 — Smoke test: one stock, one model, one horizon
# =============================================================================
"""
from fi2010_dataset import load_fi2010, make_loader, class_weights
from models import build_model
from train import train, evaluate, DEVICE

print(f"Device: {DEVICE}")

# Load CF_7 (standard 7-train + 3-test days split), horizon=10
train_ds, val_ds, test_ds = load_fi2010(DATA_DIR, fold=7, horizon=10)

# Quick sanity check on shapes
X, y = train_ds[0]
print(f"Input shape: {X.shape}  Label: {y.item()}")  # expect (100, 40), 0/1/2

# Train DeepLOB (fastest baseline, ~2-3 min on T4)
model = build_model("deeplob")
print(f"\\nTraining DeepLOB on KESKO, H=10 ...")
history = train(model, train_ds, val_ds, epochs=50, verbose=True)

# Evaluate on test set
results = evaluate(model, test_ds)
print(f"\\nTest results: {results}")
# Target: F1 ≈ 0.65–0.75 for KESKO H=10 (matches Zhang 2019 Table II)
"""

# =============================================================================
# CELL 6 — Train MambaLOB on the same setup
# =============================================================================
"""
from models import MambaLOB
from train import train, evaluate

# Default: no spatial MHA (clean Mamba-only ablation)
mamba = MambaLOB(d_model=64, d_state=16, n_layers=2, spatial_heads=0)
print(f"MambaLOB params: {sum(p.numel() for p in mamba.parameters()):,}")

print("\\nTraining MambaLOB on KESKO, H=10 ...")
history_mamba = train(mamba, train_ds, val_ds, epochs=50, verbose=True)
results_mamba = evaluate(mamba, test_ds)
print(f"\\nMambaLOB test results: {results_mamba}")

# Compare
print(f"\\nDeepLOB  F1: {results['f1']}")
print(f"MambaLOB F1: {results_mamba['f1']}")
"""

# =============================================================================
# CELL 7 — Full experiment matrix (all models × all stocks × H=10,20,50)
# Estimated time: ~3-5 hours on T4 for all 3 models × 5 stocks × 3 horizons
# Run this as a background job or overnight.
# =============================================================================
"""
from train import run_all_experiments

df = run_all_experiments(
    data_dir=DATA_DIR,
    models=["deeplob", "mlplob", "mambalob"],
    folds=[7],
    horizons=[10, 20, 50],
    epochs=50,
    results_path="results/results.csv",
)

print(df)
"""

# =============================================================================
# CELL 8 — Ablation: context window (tests O(L) advantage of Mamba)
# =============================================================================
"""
from fi2010_dataset import load_fi2010
from models import MambaLOB
from train import train, evaluate

ablation_rows = []
for seq_len in [100, 200]:
    train_ds, val_ds, test_ds = load_fi2010(
        DATA_DIR, stock_idx=1, horizon=10, seq_len=seq_len
    )
    model = MambaLOB(seq_len=seq_len, d_model=64, n_layers=2)
    train(model, train_ds, val_ds, epochs=50, verbose=False)
    r = evaluate(model, test_ds)
    ablation_rows.append({"seq_len": seq_len, **r})
    print(f"seq_len={seq_len}: {r}")

import pandas as pd
print(pd.DataFrame(ablation_rows))
"""

# =============================================================================
# CELL 9 — Ablation: spatial MHA branch (on/off)
# =============================================================================
"""
from models import MambaLOB
from train import train, evaluate

for heads in [0, 4]:
    model = MambaLOB(d_model=64, n_layers=2, spatial_heads=heads)
    train(model, train_ds, val_ds, epochs=50, verbose=False)
    r = evaluate(model, test_ds)
    label = "MambaLOB (no spatial)" if heads == 0 else f"MambaLOB + spatial (heads={heads})"
    print(f"{label}: {r}")
"""

# =============================================================================
# CELL 10 — Save results to Google Drive (persist across Colab sessions)
# =============================================================================
"""
from google.colab import drive
drive.mount("/content/drive")

import shutil, os
dest = "/content/drive/MyDrive/capstone/results"
os.makedirs(dest, exist_ok=True)
shutil.copy("results/results.csv", f"{dest}/results.csv")

# Also save checkpoints
shutil.copytree("checkpoints", f"{dest}/checkpoints", dirs_exist_ok=True)
print("Saved to Google Drive.")
"""
