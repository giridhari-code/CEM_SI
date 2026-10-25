"""
CEM Memory: Fast Memory (Episodic Store).
Stores recent episodes/context so the model can retrieve them in future turns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FastMemory(nn.Module):
    """
    A simple associative memory using a ring buffer.
    Stores key-value pairs (keys = latents, values = context).
    """
    def __init__(self, latent_dim: int = 512, capacity: int = 16):
        super().__init__()
        self.capacity = capacity
        self.latent_dim = latent_dim
        
        # Learnable projection to create keys for retrieval
        self.W_key = nn.Linear(latent_dim, latent_dim)
        
        # Buffers (not trained, just saved in state)
        self.register_buffer("keys", torch.zeros(capacity, latent_dim))
        self.register_buffer("values", torch.zeros(capacity, latent_dim))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_full", torch.tensor(False, dtype=torch.bool))

    def write(self, z_episode: torch.Tensor):
        """
        Writes a new episode to the buffer.
        Args:
            z_episode: (B, latent_dim) The episode vector from L8e.
        """
        # For simplicity, we only store the first item in batch (B=1 logic)
        # In production, you'd loop over batch or use a larger buffer.
        with torch.no_grad():
            idx = self.ptr.item()
            self.keys[idx] = z_episode[0].detach()
            self.values[idx] = z_episode[0].detach()
            
            self.ptr[0] = (idx + 1) % self.capacity
            if idx + 1 >= self.capacity:
                self.is_full[0] = True

    def read(self, query_z: torch.Tensor) -> torch.Tensor:
        """
        Retrieves relevant context based on the current query (z_t).
        Args:
            query_z: (B, latent_dim) Current state.
        Returns:
            context: (B, latent_dim) Weighted sum of stored episodes.
        """
        B = query_z.size(0)
        
        # If memory is empty, return zeros
        if not self.is_full.item() and self.ptr.item() == 0:
            return torch.zeros_like(query_z)
            
        # 1. Create Query
        q = F.normalize(self.W_key(query_z), p=2, dim=-1) # (B, D)
        
        # 2. Create Keys (Normalize for cosine similarity)
        k = F.normalize(self.keys, p=2, dim=-1) # (Capacity, D)
        
        # 3. Compute Attention Scores
        scores = torch.matmul(q, k.T) # (B, Capacity)
        
        # Mask out empty slots
        if not self.is_full.item():
            mask = torch.arange(self.capacity, device=q.device) >= self.ptr.item()
            scores = scores.masked_fill(mask, -1e9)
            
        # 4. Get Weights and Context
        weights = F.softmax(scores, dim=-1) # (B, Capacity)
        context = torch.matmul(weights, self.values) # (B, D)
        
        return context