"""
CEM Memory: State Cache.
Stores the intermediate states and errors from each loop iteration (k-depth).
Useful for gradient tracking, debugging, and meta-learning.
"""

import torch
from typing import List, Dict, Any

class StateCache:
    """
    A simple container to cache loop states.
    Not an nn.Module, just a data structure.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Clears the cache for a new forward pass."""
        self.z_states: List[torch.Tensor] = []
        self.errors: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.confidences: List[torch.Tensor] = []

    def update(self, z: torch.Tensor, error: torch.Tensor, action: torch.Tensor, confidence: torch.Tensor):
        """Saves the current loop's state."""
        self.z_states.append(z)
        self.errors.append(error)
        self.actions.append(action)
        self.confidences.append(confidence)

    def get_final_state(self) -> Dict[str, torch.Tensor]:
        """Returns the last state of the loop."""
        if not self.z_states:
            return {}
        return {
            "z_final": self.z_states[-1],
            "error_final": self.errors[-1],
            "action_final": self.actions[-1],
            "confidence_final": self.confidences[-1]
        }