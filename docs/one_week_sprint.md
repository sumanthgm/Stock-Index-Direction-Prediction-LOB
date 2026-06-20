# One-Week Sprint Plan — Reconciling the Execution Plan with Reality

*Created 2026-06-10. Hard deadline: ~2026-06-17 (1 week). Solo implementation; team handles docs.*

This is the **executable** version of `NSE_LOB_execution_plan.md`, compressed from 24 weeks to 7 days
and reconciled with work already completed. The full plan remains the methodology reference for the thesis;
this file is what actually gets done.

---

## Decisions locked (2026-06-10)
- **Timeline:** 1 week. Ruthless prioritization.
- **Refactor:** Incremental on existing `modeling/`. **No** Lightning/Hydra/W&B migration this week.
- **TLOB:** Thin wrapper around official repo (`LeonardoBerti00/TLOB`), not a reimplementation.
- **NSE data:** Dhan 20-level on S3 (`lob-capstone-data`, **ap-south-2**). Below the 5M-state target → NSE results
  are caveated as *architecture comparison*, not vs published numbers (already the plan's §1.3 framing).
- **Compute:** All training on **Colab GPU**. Mac (`pes_env` conda, py3.13/torch2.11, MPS-only) = preprocessing/ETL + smoke tests only.

### NSE data inventory (confirmed 2026-06-10 via `pes_env` + boto3)
20 valid Dhan trading days, ~660 MB, across two contracts (clean May→June expiry roll):
- **May-FUT** (`NIFTY-MAY-FUT`, `BANKNIFTY-MAY-FUT`): 12 days, May 11–26, ~684k events/instrument.
- **Jun-FUT** (`NIFTY-JUN-FUT`, `BANKNIFTY-JUN-FUT`): 8 days, Jun 1–10, ~456k events/instrument.
- ~57k events/instrument/day. **Combined ~1.1M events/instrument** — exceeds FI-2010 scale (~400k).
- Skip 3 empty 0-byte files: `20260527/28/29` (collection broke across the roll).
- Kite mirror (5-level, ~3.5 MB/day) exists but has gaps — secondary only.

## The three claims and the minimum evidence each needs
1. **Reproduction** — FI-2010 table: DeepLOB, MLPLOB, TLOB, MambaLOB macro-F1 across horizons vs published, with acceptance tests passed.
2. **Transfer** — NSE Dhan table: same 4 models, event-time Scheme A (+ Scheme B if time). Expected: near-baseline F1 = the finding.
3. **Novelty** — Mamba competitive at equal/lower compute: params / FLOPs / inference latency / memory-vs-sequence-length curve.

---

## What's already done (reuse, don't rebuild)
- Lit survey (19 papers); Dhan EC2→S3 collector live; data quality report.
- `modeling/`: `fi2010_dataset.py`, `nse_dataset.py`, `train.py`, `run_nse_matrix.py` — all working.
- `models.py`: DeepLOB, MLPLOB, MambaLOB (with pure-PyTorch fallback).
- Partial NSE matrix (10/12 runs) in `results/nse_results.csv` — **but run on Mac CPU; Mamba took 21h. Discard and rerun on GPU.**

## What's missing (the week's actual work)
- **FI-2010 reproduction results: do not exist anywhere.** No saved results/checkpoints. THE top priority.
- **TLOB: not implemented.**
- Statistical rigor (seeds, CIs); Scheme B labels; efficiency profiling; nothing committed to git.

---

## Day-by-day

### Day 1 (Tue Jun 10) — Foundation + safety net
- [ ] `git add` + first commit (repo currently has **zero commits**). Add `.gitignore` for data/checkpoints/__pycache__.
- [x] Confirm NSE data volume on S3 (boto3 in `pes_env`) — 20 days, two contracts (see inventory above).
- [x] **Front-month stitching done in `nse_dataset.py`**: match days by root (NIFTY/BANKNIFTY), stitch May+Jun
      contracts, segment labels/windows PER TRADING DAY (roll = segment boundary). New default split:
      train May 12–Jun 4 (15d) / val last 10% / test Jun 5,8,9,10. Validated on a roll-spanning subset:
      both contracts stitched, balanced labels (Down 38% / Stat 24% / Up 38%), `(100,40)` windows. ✅
- [ ] Colab: build `mamba-ssm` + `causal-conv1d` (verify import). Pull `fi2010_dataset.py`/`models.py`/`train.py`.
- [x] Colab notebook built: **`notebooks/fi2010_reproduction.ipynb`** — smoke test → DeepLOB + MLPLOB ×
      horizons {10,20,50,100} on CF_7 → full metrics (macro/weighted F1, per-class, MCC, confusion) →
      acceptance test → figures → **persist results + checkpoints to S3/Drive** (the bit lost last time).
- [ ] **Run it on a Colab GPU** (set `REPO_URL` or use the upload fallback; provide FI-2010 via Kaggle/Drive).
- **Acceptance gate:** DeepLOB FI-2010 H=10 macro-F1 in the literature band (~78–86%, paper 83.4%). If miss → debug labels/normalization before proceeding.

### Day 2 (Wed Jun 11) — TLOB
- [x] Vendored official `LeonardoBerti00/TLOB` (tlob.py + bin + mlp) into `modeling/tlob.py`, stripped of
      `constants`/`einops`/LOBSTER coupling, device-agnostic. Faithful architecture: BiN → embed+pos-enc →
      4 layers of alternating temporal/feature self-attention → progressive MLP head → 3 logits.
- [x] `build_model("tlob")` wrapper conforms to the contract `(B,100,40)` → `[B,3]`; 1.8M params (hidden=128,
      heads=1, sin-emb). Forward/backward + MPS verified locally.
- [x] Wired into the notebook full run (MODELS += "tlob") with the paper's lr=1e-4; PUBLISHED anchor 92.8.
- [ ] **Run on Colab GPU** and validate TLOB acceptance (note deviation: our seq_len=100 vs paper's 128). Complete the E1 4-model table.
- **Fallback if TLOB reproduction is far off:** keep MLPLOB as the transformer-adjacent baseline, cite TLOB published numbers, mark TLOB-reproduction as caveated. Don't let it sink the week.

### Day 3 (Thu Jun 12) — Mamba on FI-2010 + efficiency
- [x] Built `notebooks/mamba_and_efficiency.ipynb`: mamba-ssm CUDA build + backend check → E2 variants
      **M1** (pure Mamba), **M2** (Mamba temporal + feature attention), **M3** (long-context T=400) on FI-2010
      → vs-baselines table → efficiency profiling → scaling-curve figure → persist. Cells validated; profiling
      helper sanity-checked locally for all 4 models.
- [ ] **Run it on Colab GPU** (mamba-ssm builds on Colab Linux/CUDA).
- [~] Parity test: deferred — our Mac fallback (`_MambaBlock`) is an approximation, not `mambapy`; since all
      training is Colab-only (mamba-ssm kernel), strict parity is moot. Notebook instead asserts the CUDA
      kernel is active and reports backend.
- [x] Efficiency profiling for all 4 models: params, FLOPs (fvcore, best-effort — Mamba scan uncounted),
      inference latency (batch=1, HFT framing), peak GPU memory, + memory/latency-vs-seq-length curve (the
      O(L) vs O(L²) headline: TLOB vs MambaLOB).

### Day 4 (Fri Jun 13) — NSE matrix on GPU
- [x] Made the NSE loader Colab-safe: extended `WindowedLOBDataset` with per-day segment support and
      switched `nse_dataset.load_nse` to lazy windowing (eager would OOM like FI-2010 did). Validated:
      identical window counts (203,729/22,575/116,345) at ~1.2 GB vs ~6.5 GB.
- [x] Built `notebooks/nse_matrix.ipynb`: clone → mamba-ssm → **pull Dhan parquet from S3** (AWS secrets) →
      runner with naive baselines (majority/stationary/random) → 4 models × {NIFTY, BANKNIFTY} ×
      {10,20,50,100}, Scheme A (alpha=1e-5), stitched front-month, OOS June test → lift-vs-baseline table →
      persist. `run_nse_matrix.py` synced (adds tlob, root symbols). Cells validated.
- [ ] **Run it on Colab GPU** (needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY Colab secrets). 32 runs, resumable.

### Day 5 (Sat Jun 14) — Scheme B + seeds/CIs
- [x] Scheme B (spread-relative θ = mean(spread/mid) over train) added to `nse_dataset.load_nse(label_scheme=)`.
      Validated: θ≈2.47e-4 (matches measured NIFTY spread). **Finding:** at k=10 Scheme B is ~99% Stationary
      (spread ≫ short-horizon return) — degenerate by design; NSE notebook runs Scheme B at longer horizons
      {50,100,200} (cell 7b) and flags >95% Stationary.
- [x] `modeling/stats.py`: `set_seed`, `bootstrap_ci` (percentile bootstrap on test F1), `seed_summary` (mean±std).
- [x] Seeds + CIs wired in: FI-2010 reproduction notebook cell 9b (4 models × k=10 × 3 seeds) and NSE notebook
      cell 7c (4 models × NIFTY × k=10 × 3 seeds, Scheme A) → macro-F1 mean±std + bootstrap 95% CI; separate CSVs.
- [ ] **Run on Colab** (these retrain ×3 seeds — the slow rigor step; trim SEEDS if time-tight).

### Day 6 (Sun Jun 15) — Headline novelty + (stretch) backtest
- [x] **E6 long-context: covered** by `mamba_and_efficiency.ipynb` — the memory/latency-vs-seq-length curve
      (TLOB O(L²) vs MambaLOB O(L), cell 11) + M3 (MambaLOB accuracy at T=400, cell 8). Optional future
      add: a TLOB-at-T=400 *accuracy* point for a direct head-to-head (the curve already shows its cost).
- [x] **Cost-aware backtest (E8):** `modeling/backtest.py` + NSE notebook cell 8b. Signal-quality framing:
      P(up)/P(down) > τ → long/short, hold for the horizon, ~5 bps round-trip cost; reports hit-rate,
      net bps/trade, per-trade IR (mean/std, not √n) vs τ, with figures. Validated locally; honest finding
      that at short event-horizons net ≈ −cost (F1 ≠ profit). Picks the highest-lift config + reloads its checkpoint.

### Day 7 (Mon Jun 16) — Tables, figures, handoff
- [ ] Reproduction comparison table (ours vs published, deltas, seed variance).
- [ ] NSE transfer table; efficiency curves; E6 scaling figure.
- [ ] One-paragraph finding per model. Export CSVs + figures for the docs team. Commit + push everything.

---

## Explicitly deferred (out of scope this week — state as future work in thesis)
- Lightning/Hydra/W&B migration (§4.1) · Walk-forward E7 · Wall-clock 1s–60s grid E5 · OFI feature channel ·
  Mamba M3 at T=500 (covered partially by E6) · book-reconstruction (N/A — snapshot data, §9.1).

## Top risks this week
1. **TLOB wrapper eats >1 day** → fallback above.
2. **mamba-ssm won't build on Colab** → `mambapy` on GPU (still far faster than Mac); parity test covers correctness.
3. **NSE data thinner than hoped** → results stay valid as architecture comparison; lean on FI-2010 for the headline.
4. **GPU/session limits** → checkpoint every epoch to Drive/S3; results CSV appended per-run (already the pattern in `run_nse_matrix.py`).
