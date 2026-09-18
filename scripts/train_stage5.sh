#!/usr/bin/env bash
# Stage 5: train the projector only. ~15-25 min on CPU with 32 threads.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export OMP_NUM_THREADS=32
$PY -m src.train \
    --root "$LIBRISPEECH/dev-clean" \
    --minutes "${MINUTES:-30}" --epochs "${EPOCHS:-6}" \
    --batch-size 8 --grad-accum 2 --lr 1e-3 \
    --num-workers 6 --log-every 10 --eval-every 40 \
    --out "runs/stage5_${MINUTES:-30}min"
