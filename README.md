# NSE LOB — Short-Term Index Direction Prediction

MTech (DSAI) capstone, PES University. Short-term index-direction prediction from
**Limit Order Book (LOB)** data, comparing **DeepLOB**, **MLPLOB**, **TLOB** and a
novel **Mamba** (selective state-space) architecture on the **FI-2010** benchmark and
on **NSE index futures** (NIFTY / BANKNIFTY) collected live via the Dhan API.

Three claims: (1) **reproduce** DeepLOB/TLOB on FI-2010, (2) **transfer** them to NSE
index futures (first such study), (3) show **Mamba** is competitive at lower compute
with linear scaling in sequence length.

## Repository layout

```
nse-lob-capstone/
├── README.md
├── requirements.txt
├── CLAUDE.md                 # context for Claude Code
├── docs/                     # plans, logs, reports
│   ├── execution_plan.md        # full methodology (24-week reference)
│   ├── one_week_sprint.md       # active executable plan
│   ├── implementation_plan.md   # original 6-day sprint (history)
│   ├── work_log.md              # session-by-session decisions / bug fixes
│   └── data_quality_report.md   # Dhan vs Kite vs FI-2010
├── literature/               # 19-paper survey: papers/, paper_notes/, surveys
├── data_acquisition/         # live LOB collection (EC2) + S3 sync
│   ├── collect_dhan.py          # Dhan 20-level collector (production)
│   ├── collect_kiteconnect.py   # Kite 5-level collector
│   ├── kite_refresh_token.py    # daily Dhan+Kite token refresh
│   ├── sync_to_s3.sh            # parquet → S3
│   ├── load_nse_data.py         # read S3 parquet for Colab
│   ├── ec2_commands.md          # EC2 ops reference
│   ├── .env.example             # token template (real .env is gitignored)
│   └── data/                    # local samples (gitignored)
└── modeling/                 # models, datasets, training (flat package)
    ├── models.py                # DeepLOB, MLPLOB, MambaLOB (TLOB: WIP)
    ├── fi2010_dataset.py        # FI-2010 loader
    ├── nse_dataset.py           # NSE front-month loader (May+Jun stitched)
    ├── train.py                 # training loop + evaluation
    ├── run_nse_matrix.py        # full NSE experiment matrix
    ├── colab_quickstart.py      # cell-by-cell Colab script
    ├── results/                 # metrics CSVs, figures
    └── checkpoints/             # model checkpoints (gitignored)
```

## Setup

```bash
# Local (ETL, smoke tests) — conda env `pes_env` (Python 3.13, torch 2.11, MPS).
pip install -r requirements.txt
cp data_acquisition/.env.example data_acquisition/.env   # then fill in tokens
```

- **Local (Mac, MPS)** = preprocessing, data checks, smoke tests only.
- **All real training runs on a cloud GPU** (`mamba-ssm` is CUDA-only; the Mac falls back
  to a pure-PyTorch Mamba that is ~100× slower).

### Notebooks run on Colab **or** Kaggle
The notebooks in `notebooks/` auto-detect the host via `modeling/nbenv.py`:
- **Secrets** — `nbenv.get_secret()` reads from Kaggle `UserSecretsClient`, Colab `userdata`, or env vars.
  Add `GH_PAT` (repo clone), and `KAGGLE_USERNAME`/`KAGGLE_KEY` (Colab FI-2010 download) or
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (NSE S3) as secrets in either host.
- **FI-2010 data** — `nbenv.find_fi2010()` finds it under `/kaggle/input` (Kaggle: *Add Data → FI-2010*)
  or downloads via the Kaggle CLI (Colab). On Kaggle, no download needed.
- **GPU** — Kaggle's free GPU quota is separate from Colab's; switch hosts when one is throttled.

## Data

- **FI-2010** (public benchmark) — Kaggle `ulfricirons/fi-2010`, loaded by `modeling/fi2010_dataset.py`.
- **NSE Dhan 20-level** on S3 `s3://lob-capstone-data/lob-data/dhan/` (region **ap-south-2**).
  20 valid trading days across the May→June expiry roll; `nse_dataset.py` stitches a
  continuous front-month series. See `docs/data_quality_report.md`.

## Status & plan

Active plan: **`docs/one_week_sprint.md`**. Methodology reference: `docs/execution_plan.md`.
