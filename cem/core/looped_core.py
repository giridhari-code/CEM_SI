"""
CEM Core: Looped Core (L3 - L7).
Handles the Recurrent Depth Loop. Runs L3-L6 multiple times until 
Meta-Cognition (L7) decides to stop.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from cem.layers.l3_predictive import L3PredictiveCoding
from cem.layers.l4_modularity import L4EmergentModularity
from cem.layers.l5_world_model import L5WorldModel
from cem.layers.l6_active_inference import ActiveInferenceController
from cem.layers.l7_metacognition import L7MetaCognition

class LoopedCore(nn.Module):
    def __init__(self, latent_dim: int = 512, action_dim: int = 64, max_loops: int = 4):
        super().__init__()
        self.max_loops = max_loops
        self.action_dim = action_dim
        
        # Initialize L3 to L7
        self.l3_pred = L3PredictiveCoding(latent_dim=latent_dim)
        self.l4_experts = L4EmergentModularity(latent_dim=latent_dim)
        self.l5_world = L5WorldModel(state_dim=latent_dim, action_dim=action_dim)
        self.l6_controller = ActiveInferenceController(state_dim=latent_dim, action_dim=action_dim)
        self.l7_meta = L7MetaCognition(latent_dim=latent_dim)

    def forward(
        self, 
        z_t: torch.Tensor, 
        state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            z_t: (B, latent_dim) Unified representation from L1.5+L2
            state: dict containing 'last_action' and other loop states
        Returns:
            z_final: (B, latent_dim) Final state after loops
            metrics: dict with loop stats
            state: updated state
        """
        B = z_t.size(0)
        device = z_t.device
        
        z_final = z_t
        total_error = torch.zeros(B, device=device)
        action = torch.zeros(B, self.action_dim, device=device)
        confidence = torch.zeros(B, device=device)
        loops_run = 0

        for loop_idx in range(self.max_loops):
            loops_run += 1
            
            # --- L3: Predictive Coding ---
            # We pass z_t as sequence of length 1 for simplicity in loop
            z_seq_input = z_t.unsqueeze(1) 
            error, _, _ = self.l3_pred(z_seq_input)
            total_error += error
            
            # --- L4: Emergent Modularity (Experts) ---
            z_expert, _ = self.l4_experts(z_final, error)
            
            # --- L5: World Model ---
            s_next, s_causal = self.l5_world(z_expert, state["last_action"])
            
            # --- L6: Active Inference (Controller) ---
            action, _ = self.l6_controller(s_next, s_causal)
            state["last_action"] = action
            
            # Sanitize — one bad op must not freeze the whole train run
            s_next = torch.nan_to_num(s_next, nan=0.0, posinf=0.0, neginf=0.0)
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            error = torch.nan_to_num(error, nan=0.0, posinf=1.0, neginf=0.0)
            z_final = s_next  # Update state for next loop

            # --- L7: Meta-Cognition (Should we stop?) ---
            confidence, decision = self.l7_meta(z_final, error)
            
            # Stop early if all samples in batch are confident
            if (decision == 0).all():
                break
                
        metrics = {
            "loops_run": float(loops_run),
            "mean_error": float(total_error.mean().item()),
            "mean_confidence": float(confidence.mean().item())
        }
        
        return z_final, action, metrics, state