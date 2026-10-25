"""
CEM Utils: Metrics.
Tracks Accuracy, Perplexity, and Loss during training.
"""

import torch
import math
from typing import Dict

class MetricsTracker:
    """
    Accumulates metrics over an epoch and computes averages.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the counters for a new epoch."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_tokens = 0
        self.steps = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor):
        """
        Updates metrics with a new batch.
        
        Args:
            logits: (B, T, vocab_size)
            targets: (B, T)
            loss: Scalar loss tensor
        """
        self.steps += 1
        self.total_loss += loss.item()
        
        # Flatten for accuracy calculation
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Ignore index -100 (padding)
        mask = targets_flat != -100
        valid_targets = targets_flat[mask]
        
        if len(valid_targets) > 0:
            preds = logits_flat[mask].argmax(dim=-1)
            correct = (preds == valid_targets).sum().item()
            
            self.total_correct += correct
            self.total_tokens += len(valid_targets)

    def get_metrics(self) -> Dict[str, float]:
        """
        Computes final metrics for the epoch.
        """
        if self.steps == 0:
            return {"loss": 0.0, "accuracy": 0.0, "perplexity": 0.0}
            
        avg_loss = self.total_loss / self.steps
        accuracy = self.total_correct / self.total_tokens if self.total_tokens > 0 else 0.0
        
        # Perplexity = exp(Loss)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf') 
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity
        }