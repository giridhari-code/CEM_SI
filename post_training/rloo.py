"""
CEM Post-Training: RLOO (REINFORCE Leave-One-Out).
A memory-efficient RL algorithm for improving reasoning.
"""

import torch
import torch.nn.functional as F

class RLOOTrainer:
    def __init__(self, reward_model: torch.nn.Module, kl_penalty: float = 0.1):
        self.reward_model = reward_model
        self.kl_penalty = kl_penalty

    def compute_rloo_loss(
        self,
        logprobs_policy: torch.Tensor, # (k, seq_len) - current model probs
        logprobs_ref: torch.Tensor,    # (k, seq_len) - reference (SFT) model probs
        rewards: torch.Tensor          # (k,) - reward scores from RewardModel
    ) -> torch.Tensor:
        """
        Calculates RLOO policy gradient loss.
        k = number of samples generated per prompt.
        """
        k = rewards.size(0)
        
        # 1. Leave-One-Out Baseline
        # For each sample, baseline = mean of all other samples
        baseline = (rewards.sum() - rewards) / (k - 1)
        
        # 2. Advantage = Reward - Baseline
        advantages = rewards - baseline.detach()
        
        # 3. KL Divergence Penalty (Prevent drifting too far from SFT model)
        kl_div = (logprobs_policy - logprobs_ref).sum(dim=-1) # (k,)
        
        # 4. Policy Loss
        # We want to maximize advantage * log_prob
        policy_loss = -(advantages * logprobs_policy.sum(dim=-1)).mean()
        kl_loss = self.kl_penalty * kl_div.mean()
        
        total_loss = policy_loss + kl_loss
        return total_loss