"""
CEM Layer 1.5 & Layer 2: Fusion and Representation.
Combines Text, Audio, and Image embeddings into a single Unified Latent (z).
Fully Neural (Transformer-Free) using Single-Head Attention and MLPs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ==========================================
# L1.5: CROSS-MODAL FUSION (L1g, L1g2)
# ==========================================

class CrossModalFusion(nn.Module):
    """
    L1g: Single-Head Attention. 
    Uses Text as the Query (Q) and Audio/Image as Key (K) and Value (V).
    
    Mathematical Formula:
    ---------------------
    Q = W_q * z_text
    K = [W_k * z_audio, W_k * z_image]
    V = [W_v * z_audio, W_v * z_image]
    A = Softmax(Q * K^T / sqrt(d))
    z_attn = A * V
    z_fused = LayerNorm(z_text + z_attn)
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.W_v = nn.Linear(latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(
        self, 
        z_text: torch.Tensor, 
        z_audio: Optional[torch.Tensor] = None, 
        z_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 1. Generate Query from Text
        Q = self.W_q(z_text).unsqueeze(1) # (B, 1, latent_dim)
        
        # 2. Generate Keys and Values from available modalities
        keys = []
        values = []
        if z_audio is not None:
            keys.append(self.W_k(z_audio).unsqueeze(1))
            values.append(self.W_v(z_audio).unsqueeze(1))
        if z_image is not None:
            keys.append(self.W_k(z_image).unsqueeze(1))
            values.append(self.W_v(z_image).unsqueeze(1))
            
        # If only text is present, skip attention
        if not keys:
            return z_text
            
        K = torch.cat(keys, dim=1)   # (B, num_modalities-1, latent_dim)
        V = torch.cat(values, dim=1) # (B, num_modalities-1, latent_dim)
        
        # 3. Scaled Dot-Product Attention
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5) # (B, 1, num_modalities)
        attn_weights = F.softmax(scores, dim=-1)                 # A = Softmax(...)
        
        z_attn = torch.bmm(attn_weights, V).squeeze(1)           # (B, latent_dim)
        
        # 4. Residual + LayerNorm
        z_fused = self.layer_norm(z_text + z_attn)
        return z_fused


class GatedModalityNetwork(nn.Module):
    """
    L1g2: Handles missing modalities by learning a gate.
    
    Mathematical Formula:
    ---------------------
    Concat = [z_text, z_audio, z_image]
    Gate = sigmoid(W_g * Concat + b_g)
    z_out = Gate * z_fused
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        # Input is 3 * latent_dim (text + audio + image), even if some are zero
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )

    def forward(
        self, 
        z_text: torch.Tensor, 
        z_fused: torch.Tensor,
        z_audio: Optional[torch.Tensor] = None, 
        z_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        B = z_text.size(0)
        # Replace missing modalities with zero vectors
        z_a = z_audio if z_audio is not None else torch.zeros_like(z_text)
        z_i = z_image if z_image is not None else torch.zeros_like(z_text)
        
        # Concat all 3
        concat_z = torch.cat([z_text, z_a, z_i], dim=-1) # (B, 3*latent_dim)
        
        # Compute gate
        gate = self.gate_net(concat_z)                   # Gate = sigmoid(...)
        
        # Apply gate
        z_out = gate * z_fused
        return z_out


# ==========================================
# L2: UNIFIED SEMANTIC REPRESENTATION (L2)
# ==========================================

class UnifiedSemanticEncoder(nn.Module):
    """
    L2: Projects the fused multimodal vector into final CEM latent space.
    
    Mathematical Formula:
    ---------------------
    z = LayerNorm(W_2 * ReLU(W_1 * z_out + b_1) + b_2)
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_out: torch.Tensor) -> torch.Tensor:
        # z_out: (B, latent_dim) -> Output: (B, latent_dim)
        z = self.mlp(z_out)
        z = self.layer_norm(z)
        return z


# ==========================================
# L1.5 + L2 HARNESS (Integration)
# ==========================================

class L15_Fusion_and_L2(nn.Module):
    """
    Main Harness for Layer 1.5 and Layer 2.
    Takes Text, Audio, Image -> Outputs Unified Latent (z).
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.fusion = CrossModalFusion(latent_dim=latent_dim)
        self.gating = GatedModalityNetwork(latent_dim=latent_dim)
        self.unified = UnifiedSemanticEncoder(latent_dim=latent_dim)

    def forward(
        self, 
        z_text: torch.Tensor, 
        z_audio: Optional[torch.Tensor] = None, 
        z_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 1. L1g: Cross-Modal Attention
        z_fused = self.fusion(z_text, z_audio, z_image)
        
        # 2. L1g2: Gating (handle missing)
        z_gated = self.gating(z_text, z_fused, z_audio, z_image)
        
        # 3. L2: Unified Representation
        z = self.unified(z_gated)
        
        return z