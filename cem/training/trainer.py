"""
CEM Trainer — AERIS/SENTINEL-style step loop (not epoch-only demo).

- max_steps, grad accumulation, grad clip
- checkpoint_latest + periodic saves
- resume from checkpoint
- finite-loss guard (skip NaN steps)
- works with transformer-free cem.core.model.CEM
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints/run"
    max_steps: int = 1000
    batch_size: int = 4
    seq_len: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_accum: int = 1
    clip_grad_norm: float = 1.0
    precision: str = "fp32"  # fp32 | bf16
    seed: int = 42
    log_every: int = 10
    save_every: int = 200
    max_nan_skips: int = 50
    device: str = "auto"
    warmup_steps: int = 0


class CEMTrainer:
    def __init__(self, model: torch.nn.Module, cfg: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        warmup = getattr(cfg, "warmup_steps", 0) or 0
        max_steps = max(cfg.max_steps, 1)

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            # cosine to 10% lr
            progress = (step - warmup) / max(1, max_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.sched = LambdaLR(self.opt, lr_lambda)
        self.step = 0
        self.nan_skips = 0
        self.use_bf16 = cfg.precision == "bf16" and device.type == "cuda"
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _save(self, name: str, extra: Optional[Dict[str, Any]] = None) -> Path:
        out = Path(self.cfg.output_dir)
        path = out / name
        payload = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": asdict(self.cfg),
            **(extra or {}),
        }
        torch.save(payload, path)
        # also write a tiny meta json for humans
        meta = {"step": self.step, "path": str(path), "ts": time.time()}
        (out / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("saved %s (step=%d)", path, self.step)
        return path

    def resume(self, path: Optional[str] = None) -> None:
        p = Path(path) if path else Path(self.cfg.output_dir) / "checkpoint_latest.pt"
        if not p.is_file():
            logger.warning("no checkpoint to resume: %s", p)
            return
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                logger.warning("optimizer state not restored: %s", e)
        self.step = int(ckpt.get("step", 0))
        logger.info("resumed from %s at step=%d", p, self.step)

    def train(self, loader: DataLoader) -> None:
        self.model.train()
        cfg = self.cfg
        it = iter(loader)
        t0 = time.time()
        accum = 0
        self.opt.zero_grad(set_to_none=True)

        while self.step < cfg.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            if "input_ids" in batch:
                x = batch["input_ids"].to(self.device)
                y = batch["labels"].to(self.device)
            else:
                x = batch["char_ids"].to(self.device)
                y = batch["target_ids"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                out = self.model(char_ids=x, target_tokens=y)
                loss = out.loss

            if loss is None or not torch.isfinite(loss):
                self.nan_skips += 1
                logger.warning("non-finite loss at step %d (skip %d)", self.step, self.nan_skips)
                if self.nan_skips > cfg.max_nan_skips:
                    raise RuntimeError("too many NaN losses — abort")
                self.opt.zero_grad(set_to_none=True)
                accum = 0
                continue

            self.nan_skips = 0
            (loss / cfg.grad_accum).backward()
            accum += 1

            if accum >= cfg.grad_accum:
                if cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
                self.opt.step()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
                accum = 0
                self.step += 1

                if self.step % cfg.log_every == 0 or self.step == 1:
                    dt = time.time() - t0
                    logger.info(
                        "step %d/%d loss=%.4f (%.2fs)",
                        self.step,
                        cfg.max_steps,
                        float(loss.item()),
                        dt,
                    )
                if self.step % cfg.save_every == 0:
                    self._save("checkpoint_latest.pt")
                    self._save(f"checkpoint_step_{self.step}.pt")

        if accum > 0:
            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
            self.opt.step()
            self.step += 1

        self._save("checkpoint_latest.pt")
        self._save("checkpoint_final.pt")
        logger.info("training done steps=%d", self.step)
