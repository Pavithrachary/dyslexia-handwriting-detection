#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# download_dataset.sh
# Downloads the Dyslexia Handwriting dataset from Kaggle into the dataset/ folder.
#
# Requirements:
#   - kaggle CLI installed: pip install kaggle
#   - Kaggle API key at ~/.kaggle/kaggle.json
#     Get yours at: https://www.kaggle.com/settings/account (API section)
# ─────────────────────────────────────────────────────────────────────────────

set -e

DATASET="sumitaich/dyslexia-datasets"
OUTPUT_DIR="dataset"

echo "📦 Downloading Dyslexia Handwriting Dataset from Kaggle..."
kaggle datasets download -d "$DATASET" -p "$OUTPUT_DIR" --unzip

echo "✅ Dataset downloaded and extracted to ./$OUTPUT_DIR"
echo ""
echo "Expected structure:"
echo "  dataset/train/  dataset/valid/  dataset/test/"
