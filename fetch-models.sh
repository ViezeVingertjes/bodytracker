#!/usr/bin/env bash
# Download the MediaPipe pose models. They are not in git -- 44 MB of binaries
# that are byte-identical for everyone and trivially re-fetched.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
BASE=https://storage.googleapis.com/mediapipe-models/pose_landmarker
for variant in lite full heavy; do
    out="models/pose_landmarker_${variant}.task"
    if [ -f "$out" ]; then
        echo "have  $out"
        continue
    fi
    echo "fetch $out"
    curl -sSLf -o "$out" \
        "${BASE}/pose_landmarker_${variant}/float16/latest/pose_landmarker_${variant}.task"
done
echo "done -- $(ls -1 models/*.task | wc -l) models present"
