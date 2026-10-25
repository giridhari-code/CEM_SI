"""
CEM Layer 6: Active Inference (Controller).
Selects the optimal action based on the World Model's predictions.
Minimizes Expected Free Energy (Epistemic + Pragmatic value).
Fully Neural (Policy MLP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ==========================================
# 1. POLICY NETWORK (L6)
# ==========================================

class PolicyNetwork(nn.Module):
    """
    L6: Computes action probabilities based on current state and causal context.
    
    Mathematical Formula:
    ---------------------
    pi(a | s) = Softmax( MLP(s_t, s_causal) )
    a* = argmax pi(a | s)   (or sample during exploration)
    """
    def __init__(self, state_dim: int = 512, action_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        # Takes both transition state and causal state as input
        self.mlp = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, s_t: torch.Tensor, s_causal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_t: (B, state_dim) Current state
            s_causal: (B, state_dim) Causal intervention effect
        Returns:
            action_probs: (B, action_dim) Probability distribution over actions
        """
        # Concatenate state and causal context
        concat_s = torch.cat([s_t, s_causal], dim=-1) # (B, state_dim * 2)
        
        # Get logits
        logits = self.mlp(concat_s)                   # (B, action_dim)
        
        # Convert to probabilities
        action_probs = F.softmax(logits, dim=-1)
        return action_probs


# ==========================================
# 2. ACTIVE INFERENCE CONTROLLER (L6)
# ==========================================

class ActiveInferenceController(nn.Module):
    """
    Main Harness for Layer 6.
    Wraps the Policy Network and handles action selection (argmax or sampling).
    """
    def __init__(self, state_dim: int = 512, action_dim: int = 64):
        super().__init__()
        self.policy = PolicyNetwork(state_dim, action_dim)

    def forward(
        self, 
        s_t: torch.Tensor, 
        s_causal: torch.Tensor, 
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_t: (B, state_dim) 
            s_causal: (B, state_dim)
            temperature: Controls exploration (higher = more random)
        Returns:
            action: (B, action_dim) One-hot or continuous action vector
            action_probs: (B, action_dim) Probability distribution
        """
        # 1. Get action probabilities
        action_probs = self.policy(s_t, s_causal) # (B, action_dim)
        action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
        # renormalize
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        if self.training:
            scaled_probs = F.softmax(torch.log(action_probs + 1e-8) / max(temperature, 1e-3), dim=-1)
            scaled_probs = torch.nan_to_num(scaled_probs, nan=1e-8).clamp(min=1e-8)
            scaled_probs = scaled_probs / scaled_probs.sum(dim=-1, keepdim=True)
            action_idx = torch.multinomial(scaled_probs, num_samples=1).squeeze(-1)
        else:
            action_idx = torch.argmax(action_probs, dim=-1)
            
        # Convert index to one-hot vector
        action = F.one_hot(action_idx, num_classes=action_probs.size(-1)).float()
        
        return action, action_probs