# CEM training (AERIS-64 style)

Same **ops shape** as AERIS/SENTINEL train CLI — **different model** (CEM is transformer-free).

## CLI (AERIS-like)

```bash
export PYTHONPATH=$PWD

python scripts/train.py --config configs/tiny.yaml --max-steps 50 --cpu

python scripts/train.py --config configs/pretrain.yaml \
  --max-steps 2000 --batch-size 4 --seq-len 128 --lr 1e-3 --grad-accum 2

python scripts/train.py --config configs/sft.yaml \
  --load-from checkpoints/pretrain/checkpoint_latest.pt --resume
```

Flags mirror AERIS spirit: `--config`, `--train-data`, `--output-dir`, `--max-steps`,
`--batch-size`, `--seq-len`, `--lr`, `--grad-accum`, `--precision`, `--resume`, `--load-from`.

## Checkpoints

```text
checkpoints/<run>/checkpoint_latest.pt
checkpoints/<run>/checkpoint_step_N.pt
checkpoints/<run>/checkpoint_final.pt
```

## Stages

```text
pretrain → midtrain → sft → (optional dpo / constitutional)
```

```bash
MAX_STEPS=30 bash scripts/run_pipeline.sh
```

## Not Transformer

Model = `cem.core.model.CEM` (BiLSTM/GRU/CNN/MLP + loops). No GQA/RoPE.
