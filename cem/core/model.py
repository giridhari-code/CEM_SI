"""
CEM Core: Main Model.
Orchestrates all layers (L1 to L8) and connects them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any

from cem.layers.l1_io import L1MultimodalIO
from cem.layers.l1_5_fusion import L15_Fusion_and_L2
from cem.layers.l8_consciousness import L8Consciousness
from cem.core.looped_core import LoopedCore

@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None
    state: Optional[Dict[str, Any]] = None

class CEM(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 10000,
        latent_dim: int = 512,
        action_dim: int = 64,
        max_loops: int = 4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Layer 1 (I/O) and Layer 1.5/2 (Fusion)
        self.l1_io = L1MultimodalIO(vocab_size=vocab_size, latent_dim=latent_dim)
        self.l1_5_l2 = L15_Fusion_and_L2(latent_dim=latent_dim)
        
        # Looped Core (L3 - L7)
        self.looped_core = LoopedCore(latent_dim=latent_dim, action_dim=action_dim, max_loops=max_loops)
        
        # Layer 8 (Consciousness)
        self.l8_conscious = L8Consciousness(latent_dim=latent_dim, action_dim=action_dim)

    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """Initializes the hidden state for a fresh session."""
        return {
            "last_action": torch.zeros(batch_size, self.action_dim, device=device),
            "m_prev": torch.zeros(batch_size, self.latent_dim, device=device)
        }

    def forward(
        self, 
        char_ids: Optional[torch.Tensor] = None,
        waveform: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        
        B = self._get_batch_size(char_ids, waveform, image)
        device = self._get_device(char_ids, waveform, image)
        
        if state is None:
            state = self.init_state(B, device)

        # 1. L1: Multimodal Encoding
        z_text, z_audio, z_image = self.l1_io.encode(char_ids, waveform, image)
        
        if z_text is None:
            z_text = torch.zeros(B, self.latent_dim, device=device)
            
        # 2. L1.5 + L2: Fusion & Unified Representation
        z_t = self.l1_5_l2(z_text, z_audio, z_image)

        # 3. L3-L7: Looped Core (The "Thinking" Process)
        z_final, action, core_metrics, state = self.looped_core(z_t, state)

        # 4. L8: Consciousness (Global Workspace & Self-Model)
        # We stack z_t and z_final to simulate multiple modules feeding into workspace
        z_all_modules = torch.stack([z_t, z_final], dim=1) 
        
        final_context, self_surprise = self.l8_conscious(
            z_all=z_all_modules,
            m_prev=state["m_prev"],
            a_actual=action,
            z_sequence=z_t.unsqueeze(1)
        )
        
        # Update Self-Model state
        state["m_prev"] = final_context

        # 5. Decoding (L1h) — always produce logits for chat/generate
        logits = None
        loss = None
        # teacher tokens if training; else use input chars as decode anchors
        decode_tokens = target_tokens if target_tokens is not None else char_ids
        if decode_tokens is not None:
            logits = self.l1_io.decode_text(final_context, decode_tokens)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            logits = logits.clamp(-20.0, 20.0)
            if target_tokens is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    target_tokens.reshape(-1).clamp(min=0),
                    ignore_index=-100,
                )
                if not torch.isfinite(loss):
                    loss = logits.sum() * 0.0
        
        # 6. Metrics
        metrics = {
            **core_metrics,
            "self_surprise": float(self_surprise.mean().item())
        }

        return ModelOutput(logits=logits, loss=loss, metrics=metrics, state=state)

    def _get_batch_size(self, *args) -> int:
        for arg in args:
            if arg is not None:
                return arg.size(0)
        return 1

    def _get_device(self, *args) -> torch.device:
        for arg in args:
            if arg is not None:
                return arg.device
        return torch.device("cpu")

# Back-compat alias (tests/docs may say CEMModel)
CEMModel = CEM
