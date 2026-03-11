#!/usr/bin/env bash
set -euo pipefail

DEST_DIR=${1:-data/raw/eegmmidb}
mkdir -p "$DEST_DIR"

# Official PhysioNet mirror command for EEG Motor Movement/Imagery dataset.
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/ -P "$DEST_DIR"

echo "Download complete: $DEST_DIR"
