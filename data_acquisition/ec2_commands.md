# EC2 Commands Reference — LOB Data Collection

**EC2 IP:** 18.61.35.17  
**S3 Bucket:** lob-capstone-data  
**Region:** ap-south-2

> **Note on folder naming:** in this repo the collection code lives in `data_acquisition/`.
> The **live EC2 instance** still deploys it under `~/capstone/Data_Acquisition_and_preprocessing/`
> (the old name) and the running cron jobs `cd` into that path — the commands below use that path
> on purpose. Do **not** rename the EC2 directory without also updating crontab, or collection breaks.

---

## SSH Access

```bash
# Fix key permissions (one-time)
chmod 400 ~/.ssh/capstone.pem

# Connect
ssh -i ~/.ssh/capstone.pem ubuntu@18.61.35.17
```

---

## Initial Setup (one-time)

```bash
# On EC2 — create directory and install dependencies
mkdir -p ~/capstone
cd ~/capstone/Data_Acquisition_and_preprocessing
python3 -m venv venv
source venv/bin/activate
pip install websockets pyarrow python-dotenv kiteconnect awscli
mkdir -p logs data/dhan data/kite
```

```bash
# From local machine — copy scripts to EC2
ssh -i ~/.ssh/capstone.pem ubuntu@18.61.35.17 "mkdir -p ~/capstone"
scp -i ~/.ssh/capstone.pem -r Data_Acquisition_and_preprocessing ubuntu@18.61.35.17:~/capstone/
```

---

## Daily Token Refresh — Dhan + Kite (run locally before 09:15 IST)

Both tokens now expire daily at midnight IST.

```bash
cd Data_Acquisition_and_preprocessing
conda run -n pes_env python3 kite_refresh_token.py
```

The script will:
1. Prompt for Dhan token → get from https://web.dhan.co → My Profile → API Access → Generate Token
2. Run Kite OAuth flow → open the printed URL, log in, paste back the `request_token`
3. Auto-push both tokens to EC2 via SSH

---

## Manual Collection (if cron missed or for testing)

```bash
# SSH into EC2 first, then:
cd ~/capstone/Data_Acquisition_and_preprocessing
source venv/bin/activate

# Start both collectors in background
nohup python3 collect_dhan.py >> logs/dhan.log 2>&1 &
nohup python3 collect_kiteconnect.py >> logs/kite.log 2>&1 &
```

---

## Monitoring

```bash
# Check if collectors are running
ps aux | grep collect

# Watch Dhan logs live
tail -f logs/dhan.log

# Watch Kite logs live
tail -f logs/kite.log

# Check current time in IST
TZ='Asia/Kolkata' date
```

---

## Stop Collectors

```bash
# Kill both collectors
pkill -f collect_dhan.py
pkill -f collect_kiteconnect.py

# Or by PID
kill <PID>
```

---

## S3 Sync

```bash
# Manual sync (run after market close)
bash sync_to_s3.sh

# Verify files in S3
aws s3 ls s3://lob-capstone-data/lob-data/dhan/
aws s3 ls s3://lob-capstone-data/lob-data/kite/
```

---

## Verify Parquet Data

```bash
# Check Dhan data
python3 -c "
import pyarrow.parquet as pq
df = pq.read_table('data/dhan/lob_dhan_$(date +%Y%m%d).parquet').to_pandas()
print(df[['symbol','bid_price_1','ask_price_1','bid_qty_1','ask_qty_1']].head(10))
print('Symbols:', df['symbol'].unique())
print('Rows:', len(df))
"

# Check Kite data
python3 -c "
import pyarrow.parquet as pq
df = pq.read_table('data/kite/lob_kite_$(date +%Y%m%d).parquet').to_pandas()
print(df[['symbol','bid_price_1','ask_price_1','bid_qty_1','ask_qty_1']].head(10))
print('Symbols:', df['symbol'].unique())
print('Rows:', len(df))
"

# Check schema
python3 -c "import pyarrow.parquet as pq; print(pq.read_table('data/dhan/lob_dhan_$(date +%Y%m%d).parquet').schema)"
```

---

## Crontab

```bash
# View current crontab
crontab -l

# Edit crontab
crontab -e
```

Current entries:
```
# Dhan collector — 09:10 IST (03:40 UTC)
40 3 * * 1-5 cd /home/ubuntu/capstone/Data_Acquisition_and_preprocessing && /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/venv/bin/python3 collect_dhan.py >> /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/logs/dhan.log 2>&1

# Kite collector — 09:10 IST (03:40 UTC)
40 3 * * 1-5 cd /home/ubuntu/capstone/Data_Acquisition_and_preprocessing && /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/venv/bin/python3 collect_kiteconnect.py >> /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/logs/kite.log 2>&1

# S3 sync — 15:40 IST (10:10 UTC)
10 10 * * 1-5 bash /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/sync_to_s3.sh >> /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/logs/sync.log 2>&1
```

---

## Copy Updated Scripts to EC2

```bash
# Single file
scp -i ~/.ssh/capstone.pem Data_Acquisition_and_preprocessing/collect_dhan.py ubuntu@18.61.35.17:~/capstone/Data_Acquisition_and_preprocessing/

# All scripts at once
scp -i ~/.ssh/capstone.pem Data_Acquisition_and_preprocessing/collect_dhan.py \
    Data_Acquisition_and_preprocessing/collect_kiteconnect.py \
    Data_Acquisition_and_preprocessing/sync_to_s3.sh \
    ubuntu@18.61.35.17:~/capstone/Data_Acquisition_and_preprocessing/
```

---

## Futures Expiry Roll (before 2026-05-29)

Update `collect_dhan.py` INSTRUMENTS:
```python
INSTRUMENTS = {
    "62329": "NIFTY-JUN-FUT",
    "62326": "BANKNIFTY-JUN-FUT",
}
```

Update `collect_kiteconnect.py` — get new NFO tokens:
```bash
# Run locally to find June expiry tokens
conda run -n pes_env python3 -c "
from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv
load_dotenv()
kite = KiteConnect(api_key=os.environ['KITE_API_KEY'])
kite.set_access_token(os.environ['KITE_ACCESS_TOKEN'])
insts = kite.instruments('NFO')
nifty = [i for i in insts if 'NIFTY' in i['tradingsymbol'] and 'JUN' in i['tradingsymbol'] and i['instrument_type']=='FUT']
for i in nifty: print(i['instrument_token'], i['tradingsymbol'])
"
```
