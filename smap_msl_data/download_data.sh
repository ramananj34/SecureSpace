#!/bin/bash
set -euo pipefail

DATA_DIR="${1:-smap_msl_data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[1/3] Downloading data.zip from telemanom S3 bucket (1.5 GB)"
if [ ! -f data.zip ]; then
    wget -c https://s3-us-west-2.amazonaws.com/telemanom/data.zip
fi

echo "[2/3] Unzipping"
unzip -o -q data.zip

echo "[3/3] Downloading labeled_anomalies.csv - Hundman et al."
wget -c https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv -O labeled_anomalies.csv

echo "Done. Contents:"
ls -la
echo "train/ contains $(ls data/train 2>/dev/null | wc -l) channel files"
echo "test/  contains $(ls data/test  2>/dev/null | wc -l) channel files"