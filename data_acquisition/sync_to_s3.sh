#!/bin/bash
# Sync collected LOB Parquet files from EC2 to S3.
# Run automatically after market close via cron, or manually at any time.
#
# One-time setup:
#   1. Create S3 bucket (ap-south-1 = Mumbai, lowest latency from NSE servers):
#        aws s3 mb s3://YOUR_BUCKET_NAME --region ap-south-1
#
#   2. Attach IAM role to EC2 instance (no hardcoded credentials needed):
#        AWS Console → EC2 → Your instance → Actions → Security →
#        Modify IAM role → attach role with AmazonS3FullAccess
#        (or a custom policy scoped to just this bucket — see AWS setup guide)
#
#   3. Set BUCKET_NAME below.
#
# Crontab entries (run: crontab -e):
#   Collect Dhan data — 09:10 IST = 03:40 UTC:
#     40 3 * * 1-5 cd /home/ubuntu/capstone/Data_Acquisition_and_preprocessing && python3 collect_dhan.py >> logs/dhan.log 2>&1
#
#   Collect Kite data — 09:10 IST = 03:40 UTC:
#     40 3 * * 1-5 cd /home/ubuntu/capstone/Data_Acquisition_and_preprocessing && python3 collect_kiteconnect.py >> logs/kite.log 2>&1
#
#   Sync to S3 — 15:40 IST = 10:10 UTC:
#     10 10 * * 1-5 bash /home/ubuntu/capstone/Data_Acquisition_and_preprocessing/sync_to_s3.sh >> logs/sync.log 2>&1

set -euo pipefail

BUCKET_NAME="lob-capstone-data"    # <-- set this once
REGION="ap-south-2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "[$TIMESTAMP] Starting sync: ${DATA_DIR} → s3://${BUCKET_NAME}/lob-data/"

# Sync Dhan Parquet files only
if [ -d "${DATA_DIR}/dhan" ]; then
    aws s3 sync "${DATA_DIR}/dhan/" "s3://${BUCKET_NAME}/lob-data/dhan/" \
        --region "${REGION}" \
        --storage-class STANDARD_IA \
        --exclude "*" \
        --include "*.parquet"
    echo "[$TIMESTAMP] Dhan sync complete."
fi

# Sync Kite Parquet files only
if [ -d "${DATA_DIR}/kite" ]; then
    aws s3 sync "${DATA_DIR}/kite/" "s3://${BUCKET_NAME}/lob-data/kite/" \
        --region "${REGION}" \
        --storage-class STANDARD_IA \
        --exclude "*" \
        --include "*.parquet"
    echo "[$TIMESTAMP] Kite sync complete."
fi

echo "[$TIMESTAMP] All done. Files at s3://${BUCKET_NAME}/lob-data/"
