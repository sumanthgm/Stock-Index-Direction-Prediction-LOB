# Work Log — Mamba-LOB Capstone Project

**Deadline:** May 12, 2026 (6-day sprint from May 6)
**Goal:** LOB-based index direction prediction for Indian markets — novel Mamba architecture vs DeepLOB/MLPLOB baselines on FI-2010, NSE pilot via Dhan API.

---

## Phase 1 — Literature Survey

**Output:** `Literature_Survey/`

- Read and wrote learning documents for 19 papers (16 unique) covering LOB microstructure, deep learning for mid-price prediction, and state-space models.
- Key papers surveyed: DeepLOB (Zhang 2019), TLOB/MLPLOB (Berti 2025), Deep LOB Forecasting (Briola 2024), Mamba (Gu 2023), FI-2010 benchmark (Ntakaris 2018), and supporting microstructure theory papers.
- Produced: `literature_survey.md`, `project_recommendation.md`, `capstone_project_proposal.md`, `learning_order_guide.md`.
- Decision: Use FI-2010 as primary benchmark (reproducible, all SOTA papers use it). Dhan 20-level depth as NSE stretch pilot.

---

## Phase 2 — Modeling (Pre-Sprint)

**Output:** `Modeling/`

| File | Status | Notes |
|---|---|---|
| `fi2010_dataset.py` | ✅ Done | FI-2010 loading, sliding windows, LOBDataset, class_weights. Uses `NoAuction_Zscore` variant (pre-normalized). |
| `models.py` | ✅ Done | DeepLOB (Zhang 2019), MLPLOB (Berti 2025), MambaLOB with pure-PyTorch fallback if mamba-ssm unavailable. |
| `train.py` | ✅ Done | Training loop, early stopping on val F1, weighted CrossEntropy, full experiment runner. |
| `colab_quickstart.py` | ✅ Done | Cell-by-cell Colab script: install → download FI-2010 → smoke test → full run → ablations. |
| `requirements.txt` | ✅ Done | All Python dependencies. |

**Key design decisions:**
- FI-2010 `Zscore` variant is pre-normalized (mean=0, std=1 per feature) by the dataset authors — no further normalization applied.
- FI-2010 feature ordering: `[ask_price_i, ask_vol_i, bid_price_i, bid_vol_i]` interleaved for i=1..10. Different from Dhan/Kite CSV layout — needs reordering for NSE pilot.
- MambaLOB input: `(B, L=100, 40)` → Linear → 2× Mamba blocks → BiN → FC(3).
- Dhan twenty-depth feed supports both `NSE_EQ` and `NSE_FNO` (futures & options). NIFTY futures were previously receiving zero packets because the segment code was wrong (`"NSE_FO"` → correct value is `"NSE_FNO"`).

---

## Phase 3 — Data Acquisition (May 6–8, 2026)

**Output:** `Data_Acquisition_and_preprocessing/`

### Scripts Created

| File | Purpose |
|---|---|
| `test_dhan_local.py` | Local 15-min test collection via Dhan WebSocket. Validates parser, saves CSV, prints usability report. |
| `test_kite_local.py` | Local 15-min test collection via Kite Connect WebSocket. Saves CSV, prints FI-2010 compatibility report. |
| `collect_dhan.py` | Production EC2 collector — streams 20-level depth, writes daily Parquet (zstd), for continuous collection. |
| `collect_kiteconnect.py` | Production EC2 collector — streams 5-level depth, writes daily Parquet. |
| `load_nse_data.py` | Reads collected Parquet files from S3 into pandas for Colab analysis. |
| `sync_to_s3.sh` | Syncs `data/dhan/` to S3 bucket from EC2. |
| `debug_dhan_packet.py` | Full binary packet format diagnostics — brute-force structure search, hex dump, price scanning. |
| `debug_dhan_packet2.py` | Targeted bid-price probe — confirmed ASK at offset 360, probed BID candidates. |

---

### May 6 — Parser Development

**Dhan binary protocol decoded offline from `data/dhan/debug_packet.bin` (3984B = 2× HDFCBANK records).**

Confirmed packet structure:

**Full format (1992B):**
```
bytes   0–39 : header (security_id at bytes 4–7, uint32 LE)
bytes  40–359: BID  20×16B  [orders(4B)][price(8B f64)][qty(4B)]
bytes 360–679: ASK  20×16B  [price(8B f64)][qty(4B)][orders(4B)]
bytes 680+   : trailing fields (timestamps, session data)
```

**Compact format (664B) — two 332B sub-packets:**
```
BID sub (bytes   0–331): header(12B) + 20×16B  [price(8B f64)][qty(4B)][orders(4B)]
ASK sub (bytes 332–663): header(12B) + 20×16B  [price(8B f64)][qty(4B)][orders(4B)]
```

---

### May 7 — Parser Bugs Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| All bid prices implausible | BID price assumed at offset +0; actually at +4 (first 4B = orders count) | `BID_PRICE_OFFSET = 4` |
| `_prices_plausible` rejecting all rows | Denormalized near-zero floats (~2.83e-311) from empty LOB levels passed `p > 0` but failed `PRICE_MIN` check | Check only L1 bid < L1 ask instead of all 20 levels |
| Garbage values in CSV | Same denormalized floats written as prices | `PRICE_MIN <= price <= PRICE_MAX` guard in `_build_row` |
| INFY/RELIANCE records silently dropped | `len(message) % 1992 != 0` → `continue` skipped entire message, losing the valid 1992B HDFCBANK record embedded in a 3320B message | Removed `continue`; increment stat but still parse |
| Script crash at market close | Dhan server closes WebSocket without proper close frame, raising `ConnectionClosedError` | Caught `websockets.exceptions.ConnectionClosedError` as normal exit |
| Script runs past 15:30 IST | No market-close cap; loop kept waiting for COLLECT_MINUTES regardless of time | `collect_secs = min(COLLECT_MINUTES × 60, secs_to_close)` in both scripts |
| NIFTY futures missing from Dhan | Subscribed with `NSE_FO` segment — Dhan's twenty-depth feed only supports `NSE_EQ` | Removed NSE_FO instruments from Dhan config |

---

### May 7 — Kite Token Mapping Issue Discovered

**Root cause:** HDFC Bank issued a **1:1 bonus share on August 26, 2025** (first-ever in company history). Post-bonus, the share price halved (~₹1592 → ~₹796). Zerodha retired the old instrument token and reassigned it.

**Impact:** All "HDFCBANK" Kite data collected on May 7 using token `738561` was actually **RELIANCE Industries** data (current RELIANCE token = 738561 post-reassignment). The Dhan data was correct throughout.

**Fix applied:** `test_kite_local.py` — HDFCBANK token updated `738561` → `341249`.

**Confirmed correct tokens (as of May 2026):**

| Instrument | Kite Token | Dhan Security ID | Price (May 7) |
|---|---|---|---|
| HDFCBANK | 341249 | 1333 | ₹795–796 |
| RELIANCE | 738561 | 2885 | ₹1,442 |
| INFY | 408065 | 10604 | ₹1,165 |
| NIFTY-MAY-FUT (exp. May 26) | 16914178 | — (NSE_FO, not on Dhan feed) | ₹24,387 |

---

### May 7 — Data Collection Results (Dhan, corrected)

From `data/dhan/test_20260507_145026.csv` (14:50–15:05 IST, 15 min):

| Symbol | Ticks | Rate | Gap CV | Spread (bps) |
|---|---|---|---|---|
| HDFCBANK | 3,601 | 241/min | 0.90 | 3.1 |
| INFY | 141 | 9/min | 1.00 | 1.1 |
| RELIANCE | 161 | 11/min | 1.12 | 0.7 |

- HDFCBANK at 241/min is high event density — sufficient for seq_len=100 windows.
- Gap CV > 0.9 confirms Dhan feed is **event-driven** (not periodic polling like Kite).
- INFY/RELIANCE low rate because they use the compact format which has server-side throttling.

---

### May 8 — FI-2010 Format Clarification

- Confirmed: FI-2010 `Zscore` variant files are pre-normalized by the dataset authors (z-score, mean=0, std=1 per feature). Values in files are not raw prices — this is expected.
- `fi2010_dataset.py` loads these directly without re-normalizing.
- For NSE pilot: Dhan columns must be reordered from `[bid_p1..20, ask_p1..20]` to FI-2010's interleaved `[ask_p1, ask_v1, bid_p1, bid_v1, ...]` layout, then z-scored using stats from the Dhan training window.

---

## Remaining Sprint (May 8–11)

| Day | Task | Status |
|---|---|---|
| May 8 (today) | MLPLOB baseline — train on FI-2010, 5 stocks × H=10,20,30,50,100 | 🔲 Not started |
| May 8 (today) | DeepLOB baseline — same experiment set | 🔲 Not started |
| May 9 | MambaLOB core implementation + first run (KESKO, H=10) | 🔲 Not started |
| May 10 | Full MambaLOB training (15 experiments) + ablations | 🔲 Not started |
| May 11 | Results table, charts, report handoff | 🔲 Not started |
| Ongoing | NSE data collection via Dhan (EC2 stretch goal) | 🔲 Not started |

---

*Last updated: 2026-05-08*
