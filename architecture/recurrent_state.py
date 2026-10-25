"""Recurrent hidden-state cache for CEM generation (not KV attention cache)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RecurrentState:
    h: Optional[torch.Tensor] = None
    c: Optional[torch.Tensor] = None  # LSTM only

    def clear(self) -> None:
        self.h = None
        self.c = None


# Alias used by older snippets
StateCache = RecurrentState
