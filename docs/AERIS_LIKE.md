# CEM made AERIS-like (stack parity)

```bash
# Train
python -m cem.cli train --config configs/default.yaml --max-steps 100 --cpu

# Token bin path (optional)
python scripts/data/build_token_bin.py --input data/raw/pretrain.txt --output data/tokens/train.bin
python -m cem.cli train --config configs/default.yaml --train-data data/tokens/train.bin --max-steps 100

# Chat CLI
python -m cem.cli chat --checkpoint runs/cem_default/checkpoint_latest.pt --prompt "Hello" --cpu

# HTTP (AERIS routes)
python -m cem.cli serve --checkpoint runs/cem_default/checkpoint_latest.pt --port 8000
curl -s localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

Model weights are **CEM**, not SENTINEL. Ops are **AERIS-shaped**.
