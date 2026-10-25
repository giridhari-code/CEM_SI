"""AERIS-style nested config for CEM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 256
    latent_dim: int = 128
    action_dim: int = 64
    max_loops: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 4
    grad_accum: int = 1
    max_seq_len: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    clip_grad_norm: float = 1.0
    device: str = "auto"
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 500
    precision: str = "fp32"
    output_dir: str = "runs/cem"


@dataclass
class DataConfig:
    train_path: Optional[str] = "data/raw/pretrain.txt"
    val_path: Optional[str] = None
    phase: str = "pretrain"


@dataclass
class CEMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    phase: str = "pretrain"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None) -> CEMConfig:
    cfg = CEMConfig()
    if not path:
        return cfg
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def merge_dc(dc, block: dict):
        if not block:
            return dc
        for k, v in block.items():
            if hasattr(dc, k) and v is not None:
                setattr(dc, k, v)
        return dc

    if "model" in raw:
        merge_dc(cfg.model, raw["model"])
    if "training" in raw:
        # alias paths.output_dir
        merge_dc(cfg.training, raw["training"])
    if "data" in raw:
        merge_dc(cfg.data, raw["data"])
    if "paths" in raw:
        paths = raw["paths"]
        if paths.get("data_path"):
            cfg.data.train_path = paths["data_path"]
        if paths.get("output_dir"):
            cfg.training.output_dir = paths["output_dir"]
    if "phase" in raw:
        cfg.phase = str(raw["phase"])
        cfg.data.phase = cfg.phase
    # flat training keys sometimes under root
    return cfg
