#!/usr/bin/env python3
"""
CEM train entry — AERIS/SENTINEL-style CLI.

  python scripts/train.py --config configs/pretrain.yaml
  python scripts/train.py --config configs/tiny.yaml --max-steps 50 --cpu
  python scripts/train.py --config configs/sft.yaml --resume --load-from checkpoints/midtrain/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cem.core.model import CEM
from cem.data.dataloader import get_dataloader
from cem.data.dataset import DPODataset, PretrainDataset, SFTDataset
from cem.data.token_dataset import TokenBinDataset
from cem.training.trainer import CEMTrainer, TrainConfig
from cem.utils.config_loader import load_config
from cem.utils.tokenizer import CharTokenizer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_device(s: str) -> torch.device:
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loader(phase: str, data_path: str, tokenizer, seq_len: int, batch_size: int):
    from pathlib import Path as _P
    pp = _P(data_path)
    use_bin = phase == "pretrain" and (
        pp.suffix in (".bin", ".npy")
        or (pp.is_dir() and (any(pp.glob("*.bin")) or any(pp.glob("*.npy"))))
    )
    if use_bin:
        ds = TokenBinDataset(data_path, seq_len)
        phase_key = "pretrain"
    elif phase == "pretrain":
        ds = PretrainDataset(data_path, tokenizer, seq_len)
        phase_key = "pretrain"
    elif phase in ("midtrain", "sft", "constitutional_sft", "sft_grok"):
        ds = SFTDataset(data_path, tokenizer, seq_len)
        phase_key = "sft"
    elif phase in ("dpo", "rlaif_dpo", "dpo_grok"):
        ds = DPODataset(data_path, tokenizer, seq_len)
        phase_key = "dpo"
    else:
        raise ValueError(f"unknown phase: {phase}")
    return get_dataloader(phase_key, ds, batch_size, shuffle=True)



def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train CEM (transformer-free)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--train-data", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--precision", type=str, choices=["fp32", "bf16"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--load-from", type=str, default=None, help="warm-start weights .pt")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args(argv)

    setup_logging(args.log_level)
    log = logging.getLogger("cem.train")

    cfg = load_config(args.config)
    phase = str(cfg.get("phase", "pretrain")).lower()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    path_cfg = cfg.get("paths", {})

    data_path = args.train_data or path_cfg.get("data_path")
    if not data_path:
        raise SystemExit("need paths.data_path or --train-data")

    out_dir = args.output_dir or path_cfg.get("output_dir") or f"checkpoints/{phase}"
    device = resolve_device("cpu" if args.cpu else (args.device or train_cfg.get("device", "auto")))

    tcfg = TrainConfig(
        output_dir=out_dir,
        max_steps=int(args.max_steps or train_cfg.get("max_steps") or (int(train_cfg.get("epochs", 1)) * 100)),
        batch_size=int(args.batch_size or train_cfg.get("batch_size", 4)),
        seq_len=int(args.seq_len or train_cfg.get("max_seq_len", 128)),
        lr=float(args.lr or train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        grad_accum=int(args.grad_accum or train_cfg.get("grad_accum", 1)),
        clip_grad_norm=float(train_cfg.get("clip_grad_norm", 1.0)),
        precision=args.precision or train_cfg.get("precision", "fp32"),
        seed=int(args.seed or train_cfg.get("seed", 42)),
        log_every=int(train_cfg.get("log_every", 10)),
        save_every=int(train_cfg.get("save_every", 200)),
        device=str(device),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
    )

    torch.manual_seed(tcfg.seed)
    log.info(
        "CEM train | phase=%s device=%s steps=%d batch=%d seq=%d lr=%s | transformer-free",
        phase,
        device,
        tcfg.max_steps,
        tcfg.batch_size,
        tcfg.seq_len,
        tcfg.lr,
    )

    tokenizer = CharTokenizer()
    loader = build_loader(phase, data_path, tokenizer, tcfg.seq_len, tcfg.batch_size)
    log.info("dataset batches/epoch≈%d", len(loader))

    model = CEM(
        vocab_size=int(model_cfg.get("vocab_size", 256)),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        action_dim=int(model_cfg.get("action_dim", 64)),
        max_loops=int(model_cfg.get("max_loops", 2)),
    )

    if args.load_from:
        ckpt = torch.load(args.load_from, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("warm-start %s missing=%d unexpected=%d", args.load_from, len(missing), len(unexpected))

    trainer = CEMTrainer(model, tcfg, device)
    if args.resume:
        trainer.resume()

    # DPO uses simpler path in legacy train — for AERIS-parity LM path is primary
    if phase in ("dpo", "rlaif_dpo"):
        log.warning("DPO phase: use scripts/train_dpo.py or prefer SFT path first; running LM-style skip")
        # fall through only if loader is LM — else user should run dedicated script
    trainer.train(loader)


if __name__ == "__main__":
    main()
