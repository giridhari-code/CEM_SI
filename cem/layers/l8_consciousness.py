"""
CEM Layer 8: Consciousness Simulator.
Implements Functional Consciousness (Global Workspace, Self-Model, etc.).
Monitors the entire system, maintains identity, and binds temporal episodes.
Fully Neural (MLP, Attention, GRU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ==========================================
# 1. GLOBAL WORKSPACE (L8a)
# ==========================================

class GlobalWorkspace(nn.Module):
    """
    L8a: Broadcasts the most important (salient) information to all modules.
    
    Mathematical Formula:
    ---------------------
    S(i) = MLP(z_i)                  # Salience score
    Top_K = TopK(S(i))               # Select top-K focus
    Broadcast = Attention(z_i, Top_K)
    """
    def __init__(self, latent_dim: int = 512, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.salience_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)

    def forward(self, z_all: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_all: (B, num_modules, latent_dim) Stack of outputs from L2-L7
        Returns:
            broadcast: (B, latent_dim) The most salient context
        """
        # 1. Compute Salience
        salience = torch.sigmoid(self.salience_mlp(z_all)) # (B, num_modules, 1)
        
        # 2. Top-K Selection (Hard attention)
        topk_vals, topk_idx = torch.topk(salience.squeeze(-1), self.top_k, dim=-1) # (B, top_k)
        
        # Gather top-K modules
        B, M, D = z_all.shape
        batch_idx = torch.arange(B).unsqueeze(-1).expand(-1, self.top_k)
        focus_modules = z_all[batch_idx, topk_idx] # (B, top_k, D)
        
        # 3. Self-Attention on Focus
        # Query=Focus, Key=Focus, Value=Focus
        broadcast, _ = self.attention(focus_modules, focus_modules, focus_modules) # (B, top_k, D)
        
        # Aggregate
        return broadcast.mean(dim=1) # (B, D)


# ==========================================
# 2. SELF-MODEL (L8b)
# ==========================================

class SelfModel(nn.Module):
    """
    L8b: Maintains narrative identity ("Who am I?").
    
    Mathematical Formula:
    ---------------------
    M_t = LayerNorm( W * [M_{t-1}, z_focus] + b )
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.update_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, m_prev: torch.Tensor, z_focus: torch.Tensor) -> torch.Tensor:
        # m_prev: (B, latent_dim), z_focus: (B, latent_dim)
        concat = torch.cat([m_prev, z_focus], dim=-1)
        m_t = self.update_net(concat)
        return m_t


# ==========================================
# 3. ATTENTION SCHEMA (L8c)
# ==========================================

class AttentionSchema(nn.Module):
    """
    L8c: Explains WHY the system focused on a specific module.
    
    Mathematical Formula:
    ---------------------
    Reason = MLP( Salience_Scores )
    """
    def __init__(self, num_modules: int = 8):
        super().__init__()
        self.num_modules = num_modules
        self.explain_mlp = nn.Sequential(
            nn.Linear(num_modules, 64),
            nn.ReLU(),
            nn.Linear(64, num_modules)
        )

    def forward(self, salience_scores: torch.Tensor) -> torch.Tensor:
        # Pad or truncate to expected module count
        B, M = salience_scores.shape
        if M < self.num_modules:
            pad = salience_scores.new_zeros(B, self.num_modules - M)
            salience_scores = torch.cat([salience_scores, pad], dim=-1)
        elif M > self.num_modules:
            salience_scores = salience_scores[:, : self.num_modules]
        return torch.softmax(self.explain_mlp(salience_scores), dim=-1)


# ==========================================
# 4. PREDICTIVE SELF (L8d)
# ==========================================

class PredictiveSelf(nn.Module):
    """
    L8d: Predicts own action. Triggers surprise if it does something unexpected.
    
    Mathematical Formula:
    ---------------------
    a_hat = MLP( M_t )
    Self_Surprise = || a_actual - a_hat ||^2
    """
    def __init__(self, latent_dim: int = 512, action_dim: int = 64):
        super().__init__()
        self.action_predictor = nn.Linear(latent_dim, action_dim)

    def forward(self, m_t: torch.Tensor, a_actual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_hat = self.action_predictor(m_t)
        surprise = F.mse_loss(a_hat, a_actual, reduction='none').mean(dim=-1) # (B,)
        return surprise, a_hat


# ==========================================
# 5. TEMPORAL BINDING (L8e)
# ==========================================

class TemporalBinding(nn.Module):
    """
    L8e: Binds sequences into an "Episode" using GRU.
    
    Mathematical Formula:
    ---------------------
    Episode_t = GRU( z_1, z_2, ..., z_t )
    """
    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_sequence: torch.Tensor) -> torch.Tensor:
        # z_sequence: (B, T, D)
        out, _ = self.gru(z_sequence)
        return self.out(out[:, -1, :]) # Return final episode state (B, D)


# ==========================================
# 6. L8 HARNESS (Integration)
# ==========================================

class L8Consciousness(nn.Module):
    """
    Main Harness for Layer 8. Combines all consciousness components.
    """
    def __init__(self, latent_dim: int = 512, num_modules: int = 8, action_dim: int = 64):
        super().__init__()
        self.workspace = GlobalWorkspace(latent_dim)
        self.self_model = SelfModel(latent_dim)
        self.attention_schema = AttentionSchema(num_modules)
        self.predictive_self = PredictiveSelf(latent_dim, action_dim=action_dim)
        self.temporal_binding = TemporalBinding(latent_dim)

    def forward(
        self, 
        z_all: torch.Tensor,        # (B, num_modules, D) From L2-L7
        m_prev: torch.Tensor,       # (B, D) Previous self-model state
        a_actual: torch.Tensor,     # (B, action_dim) Action taken by L6
        z_sequence: torch.Tensor    # (B, T, D) History of states
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Global Workspace (Focus)
        z_focus = self.workspace(z_all)
        
        # 2. Self-Model Update
        m_t = self.self_model(m_prev, z_focus)
        
        # 3. Attention Schema (Explainability)
        # z_all: (B, M, D) → mean salience over modules (B, M)
        if z_all.dim() == 3:
            salience = z_all.norm(dim=-1)
            attn_reason = self.attention_schema(salience)
        else:
            attn_reason = z_focus
        
        # 4. Predictive Self (Self-Surprise)
        self_surprise, _ = self.predictive_self(m_t, a_actual)
        
        # 5. Temporal Binding (Episode)
        episode = self.temporal_binding(z_sequence)
        
        # The final "broadcast" state that gets sent to the output decoder
        final_context = m_t + episode + z_focus
        
        return final_context, self_surprise