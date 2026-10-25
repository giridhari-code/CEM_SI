"""AERIS-style flat token files (.bin uint16 / .npy) for CEM pretrain."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBinDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 128, stride: int | None = None):
        p = Path(path)
        if p.is_dir():
            files = sorted(list(p.glob("*.bin")) + list(p.glob("*.npy")))
            if not files:
                raise FileNotFoundError(f"no .bin/.npy under {p}")
            arrays = [self._load(f) for f in files]
            self.ids = np.concatenate(arrays)
        else:
            self.ids = self._load(p)
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)
        self.n = max(0, (len(self.ids) - seq_len) // self.stride)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        if path.suffix == ".npy":
            return np.load(path).astype(np.int64).reshape(-1)
        # uint16 bin
        return np.fromfile(path, dtype=np.uint16).astype(np.int64)

    def __len__(self) -> int:
        return max(self.n, 1)

    def __getitem__(self, idx: int):
        i = (idx * self.stride) % max(1, len(self.ids) - self.seq_len)
        chunk = self.ids[i : i + self.seq_len]
        if len(chunk) < self.seq_len:
            pad = np.zeros(self.seq_len - len(chunk), dtype=np.int64)
            chunk = np.concatenate([chunk, pad])
        t = torch.tensor(chunk, dtype=torch.long)
        return {"input_ids": t[:-1], "labels": t[1:]}
