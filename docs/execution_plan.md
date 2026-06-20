# Execution Plan: DeepLOB, Transformer (TLOB) and Mamba for Short-Term Index Direction Prediction on NSE Data

*MTech Project — Execution & Coding Methodology Document (v1, June 2026)*

---

## 0. Objective Statement

Reproduce **DeepLOB** (CNN-LSTM) and **TLOB** (dual-attention transformer) as baselines, validate them against original-paper numbers on FI-2010, then port all models to **NSE index futures LOB data (NIFTY and/or BANKNIFTY)**, and introduce a **Mamba (selective state-space) model** as the novel architecture. Compare all three on identical data, labels, and evaluation protocol.

Three deliverable claims:
1. **Reproduction claim** — our DeepLOB/TLOB implementations match published FI-2010 results within tolerance.
2. **Transfer claim** — quantified performance of these architectures on NSE index futures (first such study).
3. **Novelty claim** — Mamba is competitive/superior at equal or lower compute, with linear scaling in sequence length.

---

## 1. Feasibility Analysis

### 1.1 Mamba for LOB — state of the literature
- Mamba (Gu & Dao, 2023) and its predecessors (S4, S5) offer **linear-time sequence modeling** vs. the transformer's quadratic attention cost.
- SSMs have already entered LOB research, but mostly on the **generative** side: S5 layers are used in generative LOB/message models precisely because LOB sequences are enormously long with long-memory effects, where quadratic attention is prohibitive.
- Commentary literature argues Mamba's **selective mechanism** suits non-stationary microstructure (regime switches: open, halts, expiry) because the model can learn to gate irrelevant inputs.
- There is **no widely established Mamba result on the FI-2010/SPTP classification task** — this is your novelty window. Risk: a paper may appear mid-project; mitigation: your NSE-data angle keeps the contribution distinct regardless.

### 1.2 Technical feasibility
| Item | Assessment |
|---|---|
| DeepLOB reproduction | LOW RISK. Architecture fully specified in paper; multiple open PyTorch implementations; FI-2010 public. |
| TLOB reproduction | LOW–MEDIUM RISK. Official repo exists (LeonardoBerti00/TLOB) with code, configs, and FI-2010 pipeline. |
| Mamba implementation | MEDIUM RISK. `mamba-ssm` package requires CUDA + Linux; CPU fallback is slow. Pure-PyTorch reimplementations (e.g., `mambapy`) exist as backup. Drop-in replacement of TLOB's attention blocks with Mamba blocks is a clean ablation design. |
| NSE data acquisition | **HIGHEST RISK — resolve first.** Options in §2.1. |
| Compute | A single mid-range GPU (RTX 3060/4070, 8–12 GB, or Colab Pro/Kaggle) suffices. DeepLOB ≈ 140K params; TLOB ≈ 1M; Mamba similar. FI-2010 epochs run in minutes; NSE-scale data in hours. |
| Timeline | Feasible in 4–6 months part-time (see §8). |

### 1.3 Scientific feasibility caveats (state these in the thesis)
- LOBCAST showed all deep LOB models **degrade on new data**; expect NSE numbers well below FI-2010 numbers. That is a finding, not a failure.
- TLOB documented **declining predictability over time**; choose a fixed, recent NSE sample and report per-month decay.
- F1 ≠ profit. Include a cost-aware backtest (§6.4) to keep claims honest.
- Comparison "with the outputs from the original papers" is only strictly valid on FI-2010 (same data). On NSE data, the comparison is **between architectures**, not against published numbers — document this distinction explicitly.

---

## 2. Data Plan

### 2.1 NSE data acquisition (decide in week 1)
Ranked options:
1. **NSE Historical Tick-by-Tick (TBT) order + trade data** for F&O via NSE Data & Analytics (DotEx). Full message stream (new/modify/cancel/trade) → exact LOB reconstruction at any depth. Academic pricing exists; your institute may already subscribe — **check with your guide/library first**.
2. **Institute/NSE–NYU initiative datasets** — several Indian academic groups hold NSE message data under the NSE–NYU Stern initiative; a supervisor request may unlock this.
3. **Broker streaming APIs (Zerodha Kite, Upstox, etc.)** — live 5-level (Kite full mode: 20-level for some segments) depth snapshots for NIFTY/BANKNIFTY futures. You must **record your own dataset** (run a capture daemon during market hours for 2–3 months). Cheapest; gives snapshot (not message) data; depth limited.
4. **Commercial vendors** (TrueData, GlobalDataFeeds, AlgoTest historical depth) — historical 5-level depth at ~1-sec or tick frequency; moderate cost.

**Minimum viable dataset**: 60+ trading days of near-month NIFTY futures, 10 levels (or 5 if snapshot-based), front-month rolled 2 days before expiry. Target ≥5M usable LOB states.

### 2.2 Benchmark data
- **FI-2010** (public) — for reproduction. Use the standard "DecPre" normalized version + raw version; anchored day-based splits as in DeepLOB (train days 1–7, test 8–10).
- Optional: LOBSTER sample files (TLOB used TSLA/INTC) if budget allows — strengthens the reproduction chapter but not mandatory.

### 2.3 LOB reconstruction (if message data)
- Parse message files → event stream → maintain a price-level book (two sorted maps / arrays of (price, qty)).
- Snapshot the top-10 levels **on every book-changing event** (event time), plus a resampled 1-second grid (wall-clock time) for the slower-horizon experiments.
- Validation: cross-check reconstructed best bid/ask against the trade file; assert no crossed books; assert monotone level prices.

### 2.4 Data quality filters (India-specific)
- Drop pre-open (09:00–09:15) and the first/last 5 minutes of continuous session (09:15–15:30).
- Handle expiry-day contracts (exclude expiry day or treat as separate regime).
- Filter quote-flicker: optionally apply event-debouncing (ignore states alive < x ms) — cite the 2025 NSE "order book filtration" paper as motivation.
- Tick size ₹0.05 (index futures): expect heavy price discreteness; mid-price changes are quantized — informs label threshold choice.

---

## 3. Labeling and Task Definition

Use **two labeling schemes** and report both:

**Scheme A — DeepLOB/FI-2010-compatible (for reproduction comparability):**
- m+(t) = mean mid-price over next k events; l(t) = (m+(t) − p(t)) / p(t)
- Label: up if l > θ, down if l < −θ, else stationary. Horizons k ∈ {10, 20, 50, 100} events. θ tuned per-horizon for rough class balance (report class frequencies).

**Scheme B — TLOB spread-relative (for economic realism):**
- θ = average spread as a fraction of mid-price (per instrument, per period). Removes horizon bias and ties the threshold to transaction cost. For NIFTY futures the spread is usually 1 tick — expect a dominant "stationary" class at short horizons; report class weights used.

For the **wall-clock variant** (closer to "index direction" framing): horizons of 1s, 5s, 30s, 60s on the 1-second grid.

**Splits**: strictly temporal. Train / val / test = 70 / 10 / 20 by contiguous days. Never shuffle across the split boundary. Additionally run a **walk-forward** evaluation (rolling monthly retrain) as a robustness section.

---

## 4. Repository and Coding Methodology

### 4.1 Stack
- Python 3.11, PyTorch ≥ 2.2, PyTorch Lightning (training loop discipline), Hydra or YAML configs, Weights & Biases or TensorBoard (experiment tracking), NumPy/Polars (fast tick processing), pytest (unit tests).
- Mamba: `mamba-ssm` + `causal-conv1d` (CUDA); fallback `mambapy` (pure PyTorch) for CPU/debug.
- Reuse with attribution: TLOB official repo; a vetted DeepLOB PyTorch port; LOBCAST utilities for FI-2010 handling.

### 4.2 Repo layout
```
lob-nse/
├── configs/                # hydra/yaml: data, model, train, eval configs
├── data/
│   ├── raw/                # immutable downloads/captures
│   ├── interim/            # reconstructed books (parquet, per day)
│   └── processed/          # windowed tensors + labels (npz/memmap)
├── src/
│   ├── ingest/
│   │   ├── nse_tbt_parser.py      # message file → event stream
│   │   ├── book_builder.py        # event stream → L10 snapshots
│   │   └── kite_recorder.py       # (option 3) live depth capture daemon
│   ├── datasets/
│   │   ├── fi2010.py
│   │   └── nse_lob.py             # windowing, labeling A/B, normalization
│   ├── models/
│   │   ├── deeplob.py
│   │   ├── tlob.py                # or thin wrapper around official impl
│   │   ├── mamba_lob.py
│   │   └── blocks.py              # BiN/DAIN norm, heads, shared MLP
│   ├── training/
│   │   ├── lit_module.py          # LightningModule: loss, metrics, optim
│   │   └── train.py
│   ├── eval/
│   │   ├── metrics.py             # F1 macro, per-class P/R, MCC, confusion
│   │   ├── backtest.py            # cost-aware simulation
│   │   └── significance.py        # bootstrap CIs, Diebold-Mariano
│   └── utils/seed.py, io.py
├── tests/                  # book reconstruction invariants, label leakage tests
├── notebooks/              # EDA only — no logic that experiments depend on
└── README.md
```

### 4.3 Coding rules (write these in the thesis methodology chapter)
1. **Determinism**: fixed seeds (torch, numpy, python), `torch.use_deterministic_algorithms(True)` where feasible; log seed per run; 3–5 seeds per result, report mean ± std.
2. **No leakage**: normalization statistics (z-score) computed from **previous 5 days only** (DeepLOB convention) or train split only; labels use future-only information; assert via unit test that feature window and label window do not overlap.
3. **Single data contract**: every model consumes the same tensor `X ∈ R[B, T, F]` (T = 100 timesteps, F = 40 features = 10 levels × {ask_p, ask_v, bid_p, bid_v}) and emits logits `[B, 3]`. DeepLOB internally reshapes to `[B, 1, T, F]`.
4. **Config-driven experiments**: a result is reproducible from `configs/<exp>.yaml` + commit hash; never edit code between runs of one comparison.
5. **Versioned data artifacts**: processed tensors are hashed; experiment logs record the data hash.

### 4.4 Model specifications

**DeepLOB (reproduction-faithful):**
- Conv block 1: 1×2 conv stride (1,2) → fuses price/volume pairs; Conv block 2: 1×2 stride (1,2) → fuses across sides; Conv block 3: 1×10 → fuses across 10 levels. Each block: 2–3 conv layers, LeakyReLU(0.01).
- Inception module (3 parallel branches), concat → 192 channels.
- LSTM(hidden=64) over time → last hidden state → FC(3) softmax.
- Adam, lr 1e-3 (paper used 0.01 with decay; tune mildly), batch 64–256, early stopping on val F1 (patience 15).
- **Acceptance test**: FI-2010, horizon k=10, macro-F1 in the ~80–84% band (paper: 83.4%). Within ±2 points → reproduction accepted.

**TLOB:**
- Use official architecture: stacked blocks of {Bilinear Normalization → Temporal self-attention → Feature self-attention → MLPLOB block}. Sequence length 100–128, 10 levels.
- **Acceptance test**: FI-2010 F1 within ±2 points of reported (≈92.8% at their settings); also verify their MLPLOB baseline to sanity-check the pipeline.

**MambaLOB (your contribution) — design as a controlled swap:**
- Keep TLOB's outer scaffold (BiN normalization, MLP head, depth, hidden dim) but replace each attention pair with a **Mamba block** (optionally bidirectional: run forward and time-reversed scans, concat — justified because classification, unlike generation, permits non-causal context within the window).
- Variants to ablate:
  - M1: pure Mamba stack (drop feature attention entirely; let channel mixing happen in MLP).
  - M2: Mamba (temporal) + feature self-attention (hybrid) — isolates which axis Mamba replaces well.
  - M3: M1 with 4× longer input window (T=400–500) at similar FLOPs — exploits linear scaling; the headline experiment showing *capability transformers can't afford*.
- Hyperparameters: d_model 128–256, d_state 16, d_conv 4, expand 2, depth 4–6, AdamW lr 3e-4 cosine decay, dropout 0.1, label smoothing 0.05, class-weighted cross-entropy.

### 4.5 Training protocol (identical across models)
- Same windows, labels, splits, batch size budget, early stopping criterion (val macro-F1), max 100 epochs, gradient clipping 1.0, mixed precision (bf16/fp16).
- Hyperparameter search: small grid per model (lr × depth × hidden), ≤ 20 trials, **on validation split only**, identical budget per model (fairness requirement).
- Log: params count, FLOPs (fvcore/torchinfo), wall-clock epoch time, peak GPU memory — needed for the efficiency comparison central to the Mamba claim.

---

## 5. Experiment Matrix

| Exp | Data | Models | Purpose |
|---|---|---|---|
| E1 | FI-2010 (k=10,20,50,100) | DeepLOB, MLPLOB, TLOB | Reproduction vs. original papers |
| E2 | FI-2010 | MambaLOB M1–M3 | Mamba vs. SOTA on common benchmark |
| E3 | NSE NIFTY futures, event-time, Scheme A | all | Architecture comparison on Indian data |
| E4 | NSE, Scheme B (spread-relative) | all | Economically realistic labels |
| E5 | NSE, wall-clock 1s–60s | all | "Index direction" framing for the thesis title |
| E6 | NSE, long-context (T=500) | TLOB vs MambaLOB | Linear-scaling advantage demo |
| E7 | Walk-forward monthly | best 2 models | Robustness/regime drift |
| E8 | Cost-aware backtest | best per family | F1 → economics translation |

---

## 6. Evaluation

### 6.1 Statistical metrics
Macro-F1 (primary, matches literature), per-class precision/recall, accuracy, MCC, confusion matrices. 3–5 seeds → mean ± std; bootstrap 95% CIs on the test set; pairwise model comparison via Diebold–Mariano or a permutation test on per-window losses.

### 6.2 Efficiency metrics
Params, training FLOPs/epoch, inference latency per window (batch=1, relevant to HFT framing), memory vs. sequence length curve (the Mamba selling point: linear vs quadratic).

### 6.3 Reproduction comparison table
Side-by-side: original-paper FI-2010 numbers vs. yours, with deltas and seed variance. Explicitly note any unavoidable protocol differences.

### 6.4 Cost-aware backtest (simple, defensible)
- Signal: enter long/short when P(up)/P(down) exceeds confidence τ; exit after horizon or signal flip.
- Costs: 1 tick (₹0.05 × lot size) slippage + NSE/SEBI charges + STT; report net Sharpe, hit rate, turnover, and PnL vs τ.
- Frame as *signal quality assessment*, not a trading-strategy claim.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| NSE TBT data unobtainable/expensive | Medium | Fall back to broker-API 5-level capture (start recording **immediately**, in week 1, in parallel with negotiations — data accrues while you build on FI-2010) |
| mamba-ssm CUDA build issues | Medium | mambapy fallback; Colab/Kaggle Linux GPUs build cleanly |
| TLOB FI-2010 numbers don't reproduce | Low–Med | Use official repo verbatim first, then port; report both |
| A Mamba-LOB paper publishes first | Medium | NSE-data novelty + long-context experiment (E6) keep contribution distinct |
| Class imbalance on tick-constrained NIFTY spreads | High | Class weighting, focal loss as ablation, report balanced metrics |
| Predictability too weak on NSE (near-random F1) | Possible | Still publishable as a market-efficiency finding; add volume/OFI feature channels (Lee 2024 shows volume imbalance rescues directional signal) |

---

## 8. Timeline (indicative, 24 weeks)

| Weeks | Milestone |
|---|---|
| 1–2 | Data decision + start broker-API recording daemon; repo scaffold; FI-2010 pipeline + unit tests |
| 3–5 | DeepLOB reproduction (E1 part); acceptance test passed |
| 6–8 | TLOB reproduction (E1 complete); MLPLOB sanity baseline |
| 9–11 | MambaLOB M1–M3 on FI-2010 (E2); efficiency profiling |
| 12–14 | NSE book reconstruction + validation tests; labeling A/B; EDA chapter material |
| 15–18 | E3–E5 full runs (3–5 seeds each) |
| 19–20 | E6 long-context + E7 walk-forward |
| 21 | E8 backtest |
| 22–24 | Statistical tests, writing, ablation cleanup, buffer |

---

## 9. Environment-Specific Plan (v2 — Dhan API data on S3, M4 MacBook + Colab)

### 9.1 Data (resolved)
- Source: **Dhan API depth feed for NIFTY and BANKNIFTY**, stored as **parquet on S3**.
- Implication: this is **snapshot data** (websocket depth packets), not order-message data. Consequences:
  - No book reconstruction step — §2.3 is replaced by a *snapshot validation* step (no crossed books, monotone levels, timestamp monotonicity, gap detection between packets).
  - Depth is 5 levels (standard feed) or 20 levels (Dhan's F&O Level-3 feed). Feature dim F = 4 × levels.
  - **Primary task = wall-clock horizons (E5: 1s, 5s, 30s, 60s)**; "event-time" experiments redefine an event as *any change in the top-of-book snapshot*. Frame the thesis accordingly — this is the honest framing for snapshot data and matches the "index direction" title.
  - OFI can still be approximated from consecutive snapshots (Cont et al. snapshot formulation) — keep it as an auxiliary feature channel.
- **Week-1 data audit checklist** (run on Mac with polars):
  1. Effective snapshot frequency (packets/sec) per instrument, intraday profile.
  2. Coverage: trading days present, missing intervals, session boundaries.
  3. Depth levels actually populated; spread distribution in ticks.
  4. Storage size and row counts → window count estimate per horizon.
  5. Contract roll handling: which expiry is in the data; build a continuous front-month series with roll markers.

### 9.2 Compute strategy (resolved)
| Environment | Role | Notes |
|---|---|---|
| M4 MacBook (MPS) | Development, unit tests, preprocessing, DeepLOB/TLOB smoke runs | PyTorch MPS backend; polars + s3fs/boto3 for parquet ETL |
| Colab (T4/L4/A100) | All real training runs | `mamba-ssm` + `causal-conv1d` build on Colab Linux/CUDA |
| Mamba on Mac | `mambapy` (pure PyTorch) only | **Parity test required**: fixed input → assert mambapy vs mamba-ssm outputs match within tolerance before mixing environments |

Colab session discipline: pull preprocessed tensors from S3 at session start (boto3); checkpoint model + optimizer + epoch to Drive or S3 every epoch; resumable Lightning training; log to W&B (survives disconnects).

### 9.3 Pipeline split (Mac vs Colab)
1. **Mac**: S3 parquet → clean/validate → resample to 1s grid + change-event stream → windowed tensors `X[B,T,F]` + labels (Schemes A & B) → compressed `.npz`/memmap shards → push to S3. One-time per dataset version; hash recorded.
2. **Colab**: download shards → train → push checkpoints + metrics. No raw-data processing in Colab.

### 9.4 Model input adaptations for Dhan depth
- FI-2010 experiments (E1, E2): unchanged, 10 levels, faithful reproduction.
- NSE experiments: F = 20 (5-level) or F = 80 (20-level). DeepLOB conv block 3 kernel 1×10 → 1×5 or 1×20 (document as deviation). TLOB/Mamba: only the input projection width changes.
- If both 5- and 20-level data exist, add ablation E9: depth-5 vs depth-20 inputs — "how much depth does the Indian index book signal need?" is itself a contribution.

### 9.5 Remaining open questions
1. Which Dhan feed was captured — 5-level or 20-level? And at what effective frequency?
2. Futures depth, options, or spot index quotes alongside? (Futures depth is the right primary instrument.)
3. How many days/months of history are on S3?
4. Prior work: any existing preprocessing or model code to fold into weeks 1–8?
