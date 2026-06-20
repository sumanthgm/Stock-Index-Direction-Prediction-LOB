# Implementation Plan: Mamba-LOB

**Status:** Implementation phase — 6-day sprint to May 12 deadline
**Compute:** Google Colab (T4 or A100)
**Solo implementation** — team handles documentation

---

## Data Decision

**Kite Connect cannot provide historical LOB data.** It offers only:
- 1-minute OHLCV candles (historical, up to 60 days back)
- 5-level bid/ask depth via WebSocket — **real-time streaming only, no historical snapshots**

**Primary dataset: FI-2010**
- Free, standard academic benchmark — every paper in the literature survey uses it
- 5 Finnish stocks × 10 days × 10-level LOB ≈ 4M events per stock
- Pre-labeled at H ∈ {10, 50, 100, 200, 500} horizons (Ntakaris 2018 format)
- Results directly comparable with DeepLOB, TLOB, BINCTABL published numbers

**NSE pilot (optional, stretch):** Use **Dhan API** (20-level depth, ₹499/month) via `collect_dhan.py` running on an **AWS EC2 t2.micro** (free tier). Collects live 20-level order book snapshots during market hours, syncs to **S3** via `sync_to_s3.sh`, read back in Colab via `load_nse_data.py`. Kite Connect (5-level only) is insufficient for 10-level LOB research.

**Framing:** The contribution remains valid — novel architecture (Mamba) applied to LOB prediction, with FI-2010 for reproducibility and NSE pilot for Indian market relevance.

---

## Implementation Files (Modeling/)

| File | Status | Purpose |
|---|---|---|
| `fi2010_dataset.py` | ✅ Done | FI-2010 loading, z-score norm, sliding windows, PyTorch Dataset |
| `models.py` | ✅ Done | DeepLOB, MLPLOB, MambaLOB (pure-PyTorch + CUDA Mamba fallback) |
| `train.py` | ✅ Done | Training loop, early stopping, evaluation, full experiment runner |
| `colab_quickstart.py` | ✅ Done | Cell-by-cell Colab script: download → smoke test → full run → ablations |

---

## 6-Day Sprint

### Day 1 — May 6 | Environment + Data
- [ ] Open Colab, set runtime to T4 GPU
- [ ] Upload `fi2010_dataset.py`, `models.py`, `train.py` from `Modeling/`
- [ ] Run Cell 1–4 from `colab_quickstart.py`: install deps, download FI-2010, locate data dir
- [ ] **Critical check (Cell 4):** verify label encoding — expect `[1. 2. 3.]` as unique label values
- [ ] Run Cell 5 smoke test: DeepLOB on KESKO H=10, confirm F1 ≈ 0.65–0.75
- [ ] (Optional) Launch AWS EC2 t2.micro, deploy `collect_dhan.py`, set cron — starts collecting NSE 20-level depth in background

### Day 2 — May 7 | DeepLOB Baseline
- [ ] Run `run_all_experiments` with `models=["deeplob"]`: all 5 stocks × H=10,20,50
- [ ] Training setup: CrossEntropy loss, Adam lr=0.01, batch=64, epochs=50, early stop on val F1
- [ ] Train on all 5 FI-2010 stocks × 3 horizons (H=10, 50, 100) = 15 experiments
- [ ] Record: weighted F1 per stock per horizon
- [ ] Sanity check: compare with Zhang 2019 Table 2 (target ~70-80% F1 on KESKO at H=10)

### Day 3 — May 8 | Second Baseline
- [ ] Implement MLPLOB (Berti 2025) — simpler than TLOB, good ablation baseline:
  - Flatten(100×40) → MLP layers → FC(3)
  - Implements BiN (Bilinear Normalization) — reuse this component for Mamba-LOB
- [ ] Train MLPLOB on same 15 experiments, record results
- [ ] If time allows, implement TLOB (Berti 2025 arXiv 2502.15757):
  - Temporal SA + Spatial SA + BiN + FC(3)
  - This is the SOTA to beat — important for the comparison table
- [ ] End of day: results table should have DeepLOB + MLPLOB columns filled

### Day 4 — May 9 | Mamba-LOB Core
- [ ] Implement `MambaLOB` class:
  ```
  Input: (B, L=100, 40)
    → Linear: (B, L, d_model=64)
    → Mamba Block × 2  [d_state=16, d_conv=4, expand=2]
    → [Optional] Spatial MHA over feature axis (toggle for ablation)
    → BiN (reuse from MLPLOB)
    → Flatten + FC(3) → Softmax
  ```
- [ ] Training loop: CrossEntropy (weighted for class imbalance), AdamW, cosine LR decay
- [ ] First run: 1 stock (KESKO) × H=10 — confirm loss decreases, F1 > baseline ~0.33
- [ ] Debug until model learns, then scale to all stocks

### Day 5 — May 10 | Full Training + Ablations
- [ ] Train Mamba-LOB: all 5 stocks × H=10, 50, 100 = 15 experiments
- [ ] Ablation A: with spatial MHA head vs without (toggle flag)
- [ ] Ablation B: context window L=100 vs L=200 (test O(L) advantage)
  - For L=200: re-run Day 1 preprocessing with longer sequences
- [ ] Ablation C: z-score normalization vs BiN only
- [ ] Record all results; identify which ablation variant performs best

### Day 6 — May 11 | Results + Report Handoff
- [ ] Compile final results table: DeepLOB | MLPLOB | (TLOB) | Mamba-LOB, all stocks × horizons
- [ ] Write 1-paragraph finding per model: what works, what doesn't, where Mamba wins/loses
- [ ] Generate charts: F1 by horizon per model (line chart), F1 by stock (grouped bar)
- [ ] Export: results CSV + figures for team to include in report
- [ ] If NSE data was collected: add data pipeline summary (schema, collection stats)
- [ ] Hand off clean notebook + results to team

---

## MVP vs Stretch

| Item | Priority | Day |
|---|---|---|
| FI-2010 preprocessing pipeline | **MVP** | 1 |
| DeepLOB baseline | **MVP** | 2 |
| MLPLOB baseline | **MVP** | 3 |
| Mamba-LOB core model | **MVP** | 4 |
| Mamba-LOB full training (15 experiments) | **MVP** | 5 |
| Results table (F1) | **MVP** | 6 |
| TLOB baseline | Stretch | 3 |
| Spatial MHA ablation | Stretch | 5 |
| Context window ablation (L=200) | Stretch | 5 |
| p_T metric computation | Stretch | 6 |
| MPRF metric | Stretch | 6 |
| NSE live data collection (Kite WebSocket) | Stretch | 1 onward |

---

## Environment Setup (Colab)

```python
# Cell 1: GPU check
import torch
print(torch.cuda.is_available())   # must be True
print(torch.version.cuda)          # need 11.6+

# Cell 2: Install
!pip install mamba-ssm causal-conv1d   # try this first
!pip install pandas numpy scikit-learn matplotlib seaborn tqdm

# Cell 3: Verify mamba
from mamba_ssm import Mamba
m = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
print("Mamba OK")
```

**If mamba-ssm fails to install on Colab:** use the pure-PyTorch reference implementation from `alxndrTL/mamba.py` on GitHub — slower (~3× training time) but no CUDA compilation required. For the number of experiments here, it's acceptable on Colab's A100.

---

## Key Repositories

| Repo | Purpose |
|---|---|
| `zcakhaa/DeepLOB-Deep-Learning-for-Limit-Order-Books` | DeepLOB reference implementation |
| `FinancialComputingUCL/LOBFrame` | Data pipeline, evaluation harness |
| `matteoprata/LOBCAST` | Baseline zoo, FI-2010 support, evaluation |
| `state-spaces/mamba` | Official Mamba + mamba-ssm package |
| `alxndrTL/mamba.py` | Pure-PyTorch Mamba fallback (no CUDA build) |

---

## FI-2010 Format Reference

- **Files:** 10 .txt files — `Train_Dst_NoAuction_DecPre_CF_7.txt` etc.
- **Columns:** 144 total — 40 LOB features + 5 label sets (for different k) + other market data
- **Label column indices:** H=1→col 40, H=2→41, H=3→42, H=5→43, H=10→44
  - Note: FI-2010 "H=10" means 10 LOB events ahead — maps to the capstone's shortest horizon
  - For H=50 and H=100: not in FI-2010 directly; use LOBSTER or recompute on raw data
  - **Practical approach:** use FI-2010's H=1, 2, 10 as proxies for "short, medium, long" horizons
- **Standard split:** 7 days train + 1 day val + 2 days test (used by Zhang 2019, Briola 2024, etc.)
- **Normalization:** Apply 5-day rolling z-score **after** the split (compute stats on training days only)

---

## Mamba-LOB Hyperparameter Starting Point

| Parameter | Value | Notes |
|---|---|---|
| d_model | 64 | Feature projection dimension |
| d_state | 16 | SSM state size |
| d_conv | 4 | Local convolution width |
| expand | 2 | Inner dim = d_model × expand |
| n_layers | 2 | Number of Mamba blocks |
| L | 100 | Input sequence length |
| batch_size | 64 | Reduce to 32 if OOM |
| lr | 1e-3 | AdamW |
| epochs | 50 | With early stopping (patience=10) |
| loss | CrossEntropy | Weighted by inverse class frequency |

---

## Open Questions (resolve Day 1)

1. Does `mamba-ssm` install on Colab's current CUDA version? → test immediately in cell 3
2. FI-2010 label column mapping: verify H=10 column index before building dataset class
3. Does FI-2010 from Kaggle include all 10 stocks or just 5? (Ntakaris 2018 uses 5 stocks)
4. Class imbalance: if Stationary class > 80%, use weighted loss from the start

---

*Last updated: 2026-05-06*
