"""
CEM Post-Training: Reward Model.
A small model that scores responses based on helpfulness and safety.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, base_model_dim: int = 512):
        super().__init__()
        # In a real scenario, this takes hidden states from the base CEM model
        # Here, we simulate a simple scoring head
        self.scoring_head = nn.Sequential(
            nn.Linear(base_model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Outputs a single scalar score
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, seq_len, dim) - final layer hidden states
        Output: (B,) - scalar reward score
        """
        # Mean pooling over sequence
        pooled = hidden_states.mean(dim=1) 
        score = self.scoring_head(pooled)
        return score.squeeze(-1)

    def compute_loss(self, chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
        """
        Bradley-Terry preference loss.
        Model should score 'chosen' higher than 'rejected'.
        """
        # Loss = -log(sigmoid(chosen - rejected))
        loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
        return loss