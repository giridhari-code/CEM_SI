#!/usr/bin/env bash
# AERIS-style staged CEM training (transformer-free model)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

STEPS="${MAX_STEPS:-}"
extra=()
[[ -n "$STEPS" ]] && extra+=(--max-steps "$STEPS")

echo "== [1] pretrain =="
python scripts/train.py --config configs/pretrain.yaml "${extra[@]}"

echo "== [2] midtrain =="
python scripts/train.py --config configs/midtrain.yaml \
  --load-from checkpoints/pretrain/checkpoint_latest.pt "${extra[@]}" || \
python scripts/train.py --config configs/midtrain.yaml \
  --load-from checkpoints/pretrain/checkpoint_final.pt "${extra[@]}"

echo "== [3] sft =="
python scripts/train.py --config configs/sft.yaml \
  --load-from checkpoints/midtrain/checkpoint_latest.pt "${extra[@]}" || true

echo "Done. Latest: checkpoints/*/checkpoint_latest.pt"

echo "== [optional] sft_grok =="
python scripts/train.py --config configs/sft_grok.yaml \
  --load-from checkpoints/sft/checkpoint_latest.pt "${extra[@]}" 2>/dev/null || true
