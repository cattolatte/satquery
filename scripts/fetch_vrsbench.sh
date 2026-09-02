#!/bin/bash
# Fetch the VRSBench validation images.
#
# HuggingFace throttles large unauthenticated transfers and drops the
# connection partway. The HF python client hangs on that without raising, so
# its own retry never fires; curl's range resume does, and each pass continues
# from where the last stopped. Set HF_TOKEN to avoid the throttle entirely.
set -u
cd "$(dirname "$0")/.."

URL=https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip
DEST=data/bench/vrsbench
PART="$DEST/Images_val.zip.part"
FULL="$DEST/Images_val.zip"
SIZE=4169000000

# Abort only on a genuine stall, never on a wall clock. --max-time kills a
# healthy transfer mid-file, and the restart does not always resume cleanly
# through the CDN redirect, so progress goes backwards: 1.4 GB, then 1.8 GB,
# then 0.5 GB. Under 10 KB/s for 120 s is a stall; anything faster is progress.
STALL="--speed-limit 10000 --speed-time 120"

mkdir -p "$DEST"
# Built as a string, not an array: macOS ships bash 3.2, where an empty array
# expands as unbound under `set -u`.
AUTH=""
[ -n "${HF_TOKEN:-}" ] && AUTH="Authorization: Bearer $HF_TOKEN"

for attempt in $(seq 1 100); do
  [ -f "$FULL" ] && break
  have=$(stat -f%z "$PART" 2>/dev/null || stat -c%s "$PART" 2>/dev/null || echo 0)
  [ "$have" -ge "$SIZE" ] && { mv "$PART" "$FULL"; break; }
  echo "attempt $attempt: resuming from $((have / 1000000)) MB of $((SIZE / 1000000))"
  if [ -n "$AUTH" ]; then
    curl -sSL -C - --retry 3 $STALL -H "$AUTH" -o "$PART" "$URL"
  else
    curl -sSL -C - --retry 3 $STALL -o "$PART" "$URL"
  fi
done

if [ -f "$FULL" ]; then
  ( cd "$DEST" && unzip -q -o Images_val.zip )
  echo "extracted $(find "$DEST" -name '*.png' | wc -l) images"
else
  echo "incomplete; rerun this script to continue from where it stopped"
fi
