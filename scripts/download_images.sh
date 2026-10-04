#!/usr/bin/env bash
# Resilient download of images.zip. The GCS host has had reliability issues, so
# this uses resume (-C -), retries, and a post-download integrity check.
set -euo pipefail

URL="${IMAGES_URL:-https://storage.googleapis.com/road-handwriting/images.zip}"
DEST="${DEST:-data}"
ZIP="${DEST}/images.zip"

mkdir -p "$DEST"

echo "Downloading $URL -> $ZIP (resumable, 10 retries)"
curl -L --fail --retry 10 --retry-delay 15 --retry-all-errors \
     --continue-at - --speed-time 60 --speed-limit 1024 \
     -o "$ZIP" "$URL"

echo "Verifying archive integrity..."
if ! unzip -t "$ZIP" > /dev/null 2>&1; then
  echo "ERROR: $ZIP is corrupt or truncated."
  echo "Re-run this script - curl -C - will resume from where it stopped."
  exit 1
fi

echo "Extracting..."
unzip -q -o "$ZIP" -d "$DEST"

# Normalize: we expect data/images/{ID}.jpg
if [ -d "${DEST}/images" ]; then
  N=$(find "${DEST}/images" -name '*.jpg' | wc -l)
  echo "OK: ${N} jpg files under ${DEST}/images"
  echo "Expected roughly 6,000 (4,098 train + 1,373 test = 5,471 referenced)."
else
  echo "WARNING: ${DEST}/images not found after extraction. Contents:"
  ls -la "$DEST"
fi

echo "Cross-checking against the manifests..."
python - <<'PY'
import os, pandas as pd
missing_total = 0
for name in ("Train.csv", "Test.csv"):
    p = os.path.join("data", name)
    if not os.path.exists(p):
        print(f"  {name}: not present, skipping")
        continue
    df = pd.read_csv(p, dtype=str)
    ids = df.iloc[:, 0].astype(str)
    missing = [i for i in ids if not os.path.exists(f"data/images/{i}.jpg")]
    missing_total += len(missing)
    print(f"  {name}: {len(ids)} IDs, {len(missing)} missing images")
    if missing[:5]:
        print(f"    e.g. {missing[:5]}")
print("ALL IMAGES PRESENT" if missing_total == 0 else
      f"{missing_total} MISSING - re-run this script to resume the download")
PY

echo "Note: the dataset is competition-use only. Never upload it anywhere, and"
echo "never include it in a shared archive."
