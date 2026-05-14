#!/bin/bash
set -euo pipefail

#Source secrets from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -f "$PROJECT_ROOT/secrets.sh" ]; then
    echo "ERROR: $PROJECT_ROOT/secrets.sh not found"
    exit 1
fi
source "$PROJECT_ROOT/secrets.sh"

export KAGGLE_USERNAME
export KAGGLE_KEY

DATA_DIR="${1:-$PROJECT_ROOT/smap_msl_data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[1/4] Downloading from Kaggle"
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl

echo "[2/4] Unzipping"
unzip -o -q nasa-anomaly-detection-dataset-smap-msl.zip
rm nasa-anomaly-detection-dataset-smap-msl.zip

echo "[3/4] Flattening directory structure"
if [ -d "data/data" ]; then
    mv data/data/* data/
    rmdir data/data
fi
if [ -d "data" ]; then
    mv data/* .
    rmdir data
fi

#Get the latest labeled_anomalies.csv from telemanom repo (Hundman et al.)
if [ ! -f labeled_anomalies.csv ]; then
    echo "Fetching labeled_anomalies.csv..."
    wget -q https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
fi

echo "[4/4] Verifying contents"
ls -la
echo "train/ contains $(ls train 2>/dev/null | wc -l) channel files"
echo "test/  contains $(ls test 2>/dev/null | wc -l) channel files"
echo "Done"