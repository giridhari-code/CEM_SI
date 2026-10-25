"""
CEM Layer 5: Causal World Model.
Simulates the next state given an action (Transition) and estimates 
the effect of interventions (Causal Graph).
Fully Neural (MLP, GNN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ==========================================
# 1. TRANSITION MODEL (L5a)
# ==========================================

class TransitionModel(nn.Module):
    """
    L5a: Predicts the next state (s_{t+1}) given current state (s_t) and action (a_t).
    
    Mathematical Formula:
    ---------------------
    s_hat_{t+1} = W_2 * ReLU( W_1 * [s_t, a_t] + b_1 ) + b_2
    """
    def __init__(self, state_dim: int = 512, action_dim: int = 64):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, state_dim)
        
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim)
        )

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_t: (B, state_dim) Current state
            a_t: (B, action_dim) Action taken
        Returns:
            s_hat_next: (B, state_dim) Predicted next state
        """
        # Embed action to same dimension as state
        a_emb = self.action_embed(a_t) # (B, state_dim)
        
        # Concatenate state and action
        concat = torch.cat([s_t, a_emb], dim=-1) # (B, state_dim * 2)
        
        # Predict next state
        s_hat_next = self.net(concat)
        return s_hat_next


# ==========================================
# 2. NEURAL CAUSAL GRAPH (L5b)
# ==========================================

class NeuralCausalGraph(nn.Module):
    """
    L5b: Graph Neural Network (GNN) to estimate causal interventions.
    Learns the causal relationship between different features/states.
    
    Mathematical Formula (GCN Layer):
    ---------------------------------
    H^{(l+1)} = ReLU( D^{-1/2} * A_hat * D^{-1/2} * H^{(l)} * W^{(l)} )
    Where A_hat is the learned adjacency matrix (causal graph).
    """
    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        # Learnable Adjacency Matrix (Causal Graph)
        # Represents causal links between features
        self.adjacency = nn.Parameter(torch.eye(state_dim) * 0.5)
        
        self.W1 = nn.Linear(state_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_t: (B, state_dim) 
        Returns:
            s_causal: (B, state_dim) Causal intervention effect
        """
        # Normalize adjacency matrix (D^-1/2 * A * D^-1/2)
        # Apply softmax to ensure it's a valid probability distribution
        A = F.softmax(self.adjacency, dim=-1)
        # Graph mix on feature dim: (B, D) @ (D, D)
        H = torch.matmul(s_t, A)
        H = F.relu(self.W1(H))
        H = self.W2(H)
        return s_t + H


# ==========================================
# 3. L5 HARNESS (Integration)
# ==========================================

class L5WorldModel(nn.Module):
    """
    Main Harness for Layer 5.
    Combines Transition Model and Causal Graph.
    """
    def __init__(self, state_dim: int = 512, action_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.transition = TransitionModel(state_dim, action_dim)
        self.causal_graph = NeuralCausalGraph(state_dim)

    def forward(
        self, 
        s_t: torch.Tensor, 
        a_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_t: (B, state_dim) Current state
            a_t: (B, action_dim) Optional action. If None, assume no action.
        Returns:
            s_next: (B, state_dim) Predicted next state
            s_causal: (B, state_dim) Causal intervention effect
        """
        # If no action provided, use a zero action vector
        if a_t is None:
            a_t = torch.zeros(s_t.size(0), self.action_dim, device=s_t.device)
            
        # 1. Transition Prediction (L5a)
        s_next = self.transition(s_t, a_t)
        
        # 2. Causal Intervention (L5b)
        s_causal = self.causal_graph(s_t)
        
        return s_next, s_causal