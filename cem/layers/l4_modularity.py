"""
CEM Layer 4: Emergent Modularity.
Contains the Expert Pool (learnable routing) and Meta-Gating Network.
Dynamically selects and combines expert outputs based on context.
Fully Neural (MLPs, Attention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ==========================================
# 1. EXPERT POOL (L4a)
# ==========================================

class Expert(nn.Module):
    """A single Expert network (MLP)."""
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection for stable training
        return x + self.net(x)


class ExpertPool(nn.Module):
    """
    L4a: A pool of N experts. 
    Uses a learnable router to select top-k experts for each input.
    
    Mathematical Formula:
    ---------------------
    logits = W_r * z_t
    top_k_idx = TopK(logits)
    z_hat = sum(softmax(logits[top_k_idx]) * Expert_i(z_t))
    """
    def __init__(self, latent_dim: int = 512, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create Experts
        self.experts = nn.ModuleList([Expert(latent_dim) for _ in range(num_experts)])
        
        # Router (Gating Network)
        self.router = nn.Linear(latent_dim, num_experts)

    def forward(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: (B, latent_dim) Input state
        Returns:
            z_hat: (B, latent_dim) Combined expert output
            routing_weights: (B, num_experts) Softmax weights
        """
        B, D = z_t.shape
        
        # 1. Compute router logits
        logits = self.router(z_t) # (B, num_experts)
        
        # 2. Get top-k routing
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        
        # 3. Create sparse weight matrix (Straight-Through Estimator concept)
        # We want gradients to flow through the dense logits
        routing_weights = torch.zeros_like(logits)
        routing_weights.scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1))
        
        # 4. Compute expert outputs
        expert_outs = torch.stack([expert(z_t) for expert in self.experts], dim=1) # (B, num_experts, D)
        
        # 5. Weighted sum of experts
        # z_hat = sum(routing_weights * expert_outs)
        z_hat = torch.einsum('bn,bnd->bd', routing_weights, expert_outs)
        
        return z_hat, routing_weights


# ==========================================
# 2. META-GATING NETWORK (L4b)
# ==========================================

class MetaGatingNetwork(nn.Module):
    """
    L4b: Combines the original state, expert output, and prediction error.
    Learns how much to trust the experts vs the original state.
    
    Mathematical Formula:
    ---------------------
    Gate = sigmoid(W_g * [z_t, z_hat, error] + b_g)
    z_final = Gate * z_hat + (1 - Gate) * z_t
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        # Input: z_t (512) + z_hat (512) + error (1) = 1025
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )

    def forward(
        self, 
        z_t: torch.Tensor, 
        z_hat: torch.Tensor, 
        error: torch.Tensor
    ) -> torch.Tensor:
        # error is (B,), unsqueeze to (B, 1)
        err = error.unsqueeze(-1)
        
        # Concatenate inputs
        concat_in = torch.cat([z_t, z_hat, err], dim=-1) # (B, 2*D + 1)
        
        # Compute gate
        gate = self.gate_net(concat_in)                  # (B, latent_dim)
        
        # Interpolate between original and expert output
        z_final = gate * z_hat + (1.0 - gate) * z_t
        return z_final


# ==========================================
# 3. L4 HARNESS (Integration)
# ==========================================

class L4EmergentModularity(nn.Module):
    """
    Main Harness for Layer 4.
    Connects Expert Pool (L4a) and Meta-Gating (L4b).
    """
    def __init__(self, latent_dim: int = 512, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.expert_pool = ExpertPool(latent_dim, num_experts, top_k)
        self.meta_gating = MetaGatingNetwork(latent_dim)

    def forward(
        self, 
        z_t: torch.Tensor, 
        error: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: (B, latent_dim) Input from L3
            error: (B,) Prediction error from L3
        Returns:
            z_final: (B, latent_dim)
            routing_weights: (B, num_experts)
        """
        # 1. Expert Pool (L4a)
        z_hat, routing_weights = self.expert_pool(z_t)
        
        # 2. Meta-Gating (L4b)
        z_final = self.meta_gating(z_t, z_hat, error)
        
        return z_final, routing_weights