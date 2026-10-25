# CEM — Cognitive Emergence Machine

**Transformer-free**, **looped (adaptive compute)**, multimodal cognitive architecture.

Not LLaMA / not GQA / not RoPE attention. Core is **BiLSTM · GRU · CNN · MLP** with recurrent depth (L3–L6) gated by meta-cognition (L7).

## Features

- Multimodal I/O (text; audio/image when deps present)
- Looped core: think more only when uncertainty is high
- L7 meta-cognition · L8 workspace / self-model
- Training curriculum: pretrain → midtrain → SFT → DPO → constitutional SFT → preference

## Quick train

```bash
export PYTHONPATH=$PWD
pip install -r requirements.txt
python scripts/train.py --config configs/tiny.yaml --max_steps=5 --cpu
bash scripts/run_pipeline.sh
```

Docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/TRAINING.md](docs/TRAINING.md) · [docs/ALIGNMENT.md](docs/ALIGNMENT.md)

## Layout

| Path | Role |
|------|------|
| `cem/` | **Model** L1–L8 (source of truth) |
| `architecture/` | RNN helpers only (no attention) |
| `scripts/train.py` | Unified phase trainer |
| `post_training/` | Constitutional critique/revision, RM, RLOO sketches |
| `data_engineering/` | BPE, dedup, PII, quality filters |
| `infra/` | Trainer, checkpoint, optional FSDP |
| `eval/` | Harness + model card gating |

## Grok-style behavior (training + prompt)

Truth-seeking, low sycophancy, short answers — **methods only**, not xAI weights.

```bash
python scripts/train.py --config configs/sft_grok.yaml --load-from checkpoints/pretrain/checkpoint_latest.pt
```

See [docs/GROK_STYLE.md](docs/GROK_STYLE.md).
