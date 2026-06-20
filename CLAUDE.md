# Capstone Project — Claude Context

## Project
MTech (DSAI) capstone at PES University, Sem 3.
**Topic:** Short-term index direction prediction using Limit Order Book (LOB) data on NSE India.
**Deadline:** revamp sprint ~June 17, 2026 (1-week push; original May 12 sprint produced the MVP).
**Solo implementation** — team handles documentation.
**Active plan:** `docs/one_week_sprint.md`. Methodology reference: `docs/execution_plan.md`.

## Directory Structure

Repo root is `nse-lob-capstone/` (the shareable repo). Local ETL env: conda `pes_env`.

```
nse-lob-capstone/
├── README.md
├── requirements.txt
├── CLAUDE.md                   # this file
├── docs/                       # execution_plan, one_week_sprint, implementation_plan, work_log, data_quality_report
├── literature/                 # 19 paper notes + survey docs (papers/, paper_notes/)
├── data_acquisition/           # Data collection scripts
│   ├── collect_dhan.py         # Production Dhan 20-level collector (EC2)
│   ├── collect_kiteconnect.py  # Production Kite 5-level collector (EC2)
│   ├── test_dhan_local.py      # Local test script (15-min, 23 instruments)
│   ├── test_kite_local.py      # Local test script (15-min, 23 instruments)
│   ├── sync_to_s3.sh           # Sync Parquet files to S3 (Parquet only)
│   ├── load_nse_data.py        # Read S3 Parquet into pandas for Colab
│   ├── kite_refresh_token.py   # Daily Dhan+Kite token refresh (run locally)
│   ├── ec2_commands.md         # All EC2 commands reference
│   ├── .env.example            # token template (real .env gitignored)
│   └── data/                   # Local test CSVs (gitignored, not synced to S3)
└── modeling/                   # Model code: fi2010_dataset.py, nse_dataset.py, models.py, train.py,
                                #   run_nse_matrix.py, colab_quickstart.py; results/, checkpoints/
```

Note: the live EC2 instance still deploys the old `Data_Acquisition_and_preprocessing/` path;
cron/SSH comments in the collectors describe that deployment until it is redeployed from this repo.

## Primary Datasets

### FI-2010 (main benchmark)
- 5 Finnish stocks × 10 days × 10-level LOB
- Pre-normalized (Zscore variant): values are already z-scored, do NOT re-normalize
- Feature order: `[ask_p1, ask_v1, bid_p1, bid_v1, ...]` interleaved × 10 levels = 40 features
- Labels: H=10 at col index 44 (0-indexed 40); H=50 at 45; H=100 at 46
- Standard split: 7 train + 1 val + 2 test days
- Download: Kaggle `lobster/fi-2010`; load via `modeling/fi2010_dataset.py`

### NSE Index Futures (pilot, via Dhan API)
- Instruments: NIFTY-MAY-FUT (SID 66071), BANKNIFTY-MAY-FUT (SID 66068)
- Collected live on EC2 during market hours (09:15–15:30 IST)
- Stored as Parquet in S3: `s3://lob-capstone-data/lob-data/dhan/lob_dhan_YYYYMMDD.parquet`
- Schema: timestamp, symbol, bid_price_1..20, bid_qty_1..20, bid_orders_1..20, ask_price_1..20...
- **Label alpha calibration:** FI-2010's alpha=0.002 is too large for index futures (prices ~₹24K-₹55K). Use alpha=1e-5 for balanced labels.
- Feature reorder needed: Dhan is `[bid_p1..20, ask_p1..20]` → FI-2010 expects interleaved

## Models

All in `modeling/models.py`:
- **DeepLOB**: CNN + Inception + LSTM, input (B, 100, 40)
- **MLPLOB**: Flatten → MLP + BiN, input (B, 100, 40)
- **MambaLOB**: Linear → 2× Mamba blocks → BiN → FC(3), input (B, 100, 40)
- **TLOB**: dual-attention transformer — WIP (thin wrapper around official repo, per one_week_sprint.md)

Training: `modeling/train.py` — weighted CrossEntropy, AdamW, early stopping on val F1.
Colab: `modeling/colab_quickstart.py` — full cell-by-cell script.

## EC2 Infrastructure

- **Instance:** t3.micro, 18.61.35.17, Ubuntu 24.04, ap-south-2 (Hyderabad)
- **SSH:** `ssh -i ~/.ssh/capstone.pem ubuntu@18.61.35.17`
- **Key:** chmod 400 required
- **S3:** lob-capstone-data (**ap-south-2** — use `region_name="ap-south-2"` for boto3/s3fs)
- **Cron:** collectors start 03:40 UTC (09:10 IST); S3 sync at 10:10 UTC (15:40 IST)
- **Dhan token:** expires daily at midnight IST — run `kite_refresh_token.py` (handles both Dhan + Kite)
- **Kite token:** expires daily at midnight IST — same script, same morning routine before 09:15 IST
- **Reference:** `data_acquisition/ec2_commands.md`

## Futures Expiry Roll (before 2026-05-29)

Dhan: update `collect_dhan.py` INSTRUMENTS to `{"62329": "NIFTY-JUN-FUT", "62326": "BANKNIFTY-JUN-FUT"}`
Kite: run `kite.instruments("NFO")` to find June expiry tokens for NIFTY/BANKNIFTY FUT.

## Key Technical Notes

- **Dhan binary parser:** two formats — full (1992B) and compact (664B). Greedy parser in `collect_dhan.py`. Schema uses int64 for qty/orders (uint32 overflows int32 max).
- **Pandas timestamp parsing:** use `format='ISO8601'` not `parse_dates=` for timezone-aware strings in pandas 2.x.
- **NSE_FNO not NSE_FO:** correct Dhan segment code for futures/options is `NSE_FNO`.
- **HDFCBANK token (Kite):** use 341249 (post Aug-2025 1:1 bonus). Old token 738561 = RELIANCE now.
- **awscli:** install via `pip install awscli` on Ubuntu (apt package unavailable on some versions).
