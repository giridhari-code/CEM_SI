"""
CEM architecture helpers — **transformer-free only**.

Allowed: RNN / GRU / LSTM, recurrent state, loop depth.
Forbidden here: multi-head attention, GQA, RoPE, KV-cache attention stacks.
Core model lives in `cem.core` + `cem.layers` (L1–L8).
"""

from architecture.rnn_layers import GRUBlock, LSTMBlock
from architecture.recurrent_state import RecurrentState

__all__ = ["GRUBlock", "LSTMBlock", "RecurrentState"]
