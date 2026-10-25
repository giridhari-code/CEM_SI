"""
CEM Layer 7: Meta-Cognition.
Calculates confidence based on prediction error and state.
Decides whether to 'Respond', 'Think More', or 'Ask User'.
Fully Neural (Learned Confidence Head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ==========================================
# 1. LEARNED CONFIDENCE HEAD (L7)
# ==========================================

class LearnedConfidenceHead(nn.Module):
    """
    L7: Learns to predict confidence (0 to 1) based on state and error.
    
    Mathematical Formula:
    ---------------------
    c = sigmoid( MLP( [z_t, error] ) )
    
    Decision Logic:
    ---------------
    if c >= 0.8: action = "Respond"
    elif 0.5 <= c < 0.8: action = "Think More" (Loop again)
    else: action = "Ask User" (Clarification)
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        # Input: z_t (512) + error (1)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z_t: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, latent_dim) Current latent state
            error: (B,) Prediction error from L3
        Returns:
            confidence: (B,) Value between 0 and 1
        """
        # Ensure error is (B, 1)
        if error.dim() == 1:
            error = error.unsqueeze(-1)
            
        # Concatenate state and error
        concat_in = torch.cat([z_t, error], dim=-1) # (B, latent_dim + 1)
        
        # Compute confidence
        confidence = torch.sigmoid(self.mlp(concat_in)).squeeze(-1) # (B,)
        return confidence


# ==========================================
# 2. META-COGNITION CONTROLLER (L7)
# ==========================================

class L7MetaCognition(nn.Module):
    """
    Main Harness for Layer 7.
    Wraps the Confidence Head and outputs the decision action.
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.confidence_head = LearnedConfidenceHead(latent_dim)

    def forward(
        self, 
        z_t: torch.Tensor, 
        error: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: (B, latent_dim)
            error: (B,)
        Returns:
            confidence: (B,)
            decision_mask: (B,) 0 = Respond, 1 = Think More, 2 = Ask User
        """
        # 1. Predict Confidence
        confidence = self.confidence_head(z_t, error) # (B,)
        
        # 2. Decision Logic (Vectorized for Batch Processing)
        # Initialize all as "Respond" (0)
        decision = torch.zeros_like(confidence, dtype=torch.long)
        
        # Where confidence is between 0.5 and 0.8 -> "Think More" (1)
        think_more_mask = (confidence >= 0.5) & (confidence < 0.8)
        decision[think_more_mask] = 1
        
        # Where confidence is < 0.5 -> "Ask User" (2)
        ask_user_mask = (confidence < 0.5)
        decision[ask_user_mask] = 2
        
        return confidence, decision