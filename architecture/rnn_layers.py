"""CEM pure recurrent blocks — no attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from architecture.recurrent_state import RecurrentState


class GRUBlock(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, state: Optional[RecurrentState] = None
    ) -> Tuple[torch.Tensor, RecurrentState]:
        h0 = None
        if state is not None and state.h is not None:
            h0 = state.h
        out, h_n = self.gru(x, h0)
        out = self.norm(out + x)
        new_state = RecurrentState(h=h_n)
        return out, new_state


class LSTMBlock(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, state: Optional[RecurrentState] = None
    ) -> Tuple[torch.Tensor, RecurrentState]:
        hc = None
        if state is not None and state.h is not None and state.c is not None:
            hc = (state.h, state.c)
        out, (h_n, c_n) = self.lstm(x, hc)
        out = self.norm(out + x)
        return out, RecurrentState(h=h_n, c=c_n)


# Back-compat alias
PureCEMBlock = GRUBlock
