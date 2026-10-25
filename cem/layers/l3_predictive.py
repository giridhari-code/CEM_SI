"""
CEM Layer 3: Predictive Coding.
Predicts the next latent state at sentence, paragraph, and cross-modal levels.
Calculates prediction error (surprise) which drives learning and metacognition.
Fully Neural (MLP, GRU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# ==========================================
# 1. SENTENCE PREDICTOR (L3a)
# ==========================================

class SentencePredictor(nn.Module):
    """
    L3a: Predicts the next sentence latent from the current one.
    
    Mathematical Formula:
    ---------------------
    z_hat_{t+1} = W_2 * ReLU(W_1 * z_t + b_1) + b_2
    Error (ε_s) = ||z_{t+1} - z_hat_{t+1}||^2
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        # z_t: (B, latent_dim) -> z_hat: (B, latent_dim)
        return self.mlp(z_t)

    @staticmethod
    def compute_error(z_next: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        """Returns MSE error for each batch item. Shape: (B,)"""
        return F.mse_loss(z_hat, z_next, reduction='none').mean(dim=-1)


# ==========================================
# 2. PARAGRAPH PREDICTOR (L3b)
# ==========================================

class ParagraphPredictor(nn.Module):
    """
    L3b: Predicts the next paragraph state using a GRU over sentence latents.
    
    Mathematical Formula:
    ---------------------
    p_t = (1/t) * sum(z_1 ... z_t)    # Running mean
    p_hat_{t+1} = W_out * GRU(p_t) + b_out
    Error (ε_p) = ||p_{t+1} - p_hat_{t+1}||^2
    """
    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_sequence: torch.Tensor) -> torch.Tensor:
        # z_sequence: (B, T, latent_dim)
        # Using GRU to process the sequence
        out, _ = self.gru(z_sequence)            # (B, T, hidden_dim)
        p_hat = self.out(out)                    # (B, T, latent_dim)
        return p_hat

    @staticmethod
    def compute_error(p_next: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:
        """Returns MSE error. Shape: (B,)"""
        return F.mse_loss(p_hat, p_next, reduction='none').mean(dim=-1)


# ==========================================
# 3. CROSS-MODAL PREDICTOR (L3c)
# ==========================================

class CrossModalPredictor(nn.Module):
    """
    L3c: Predicts Audio/Image latents from Text latent.
    Used to compute cross-modal error.
    
    Mathematical Formula:
    ---------------------
    z_hat_audio = MLP_a(z_text)
    z_hat_image = MLP_i(z_text)
    Error (ε_cross) = ||z_audio - z_hat_audio||^2 + ||z_image - z_hat_image||^2
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.audio_pred = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.image_pred = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, z_text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_hat_audio = self.audio_pred(z_text)
        z_hat_image = self.image_pred(z_text)
        return z_hat_audio, z_hat_image


# ==========================================
# 4. L3 HARNESS (Integration)
# ==========================================

class L3PredictiveCoding(nn.Module):
    """
    Main Harness for Layer 3.
    Combines Sentence, Paragraph, and Cross-Modal predictors.
    Returns total prediction error (Surprise).
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.sentence_pred = SentencePredictor(latent_dim)
        self.paragraph_pred = ParagraphPredictor(latent_dim)
        self.cross_modal_pred = CrossModalPredictor(latent_dim)

    def forward(
        self,
        z_sequence: torch.Tensor,          # (B, T, latent_dim)
        z_audio_seq: Optional[torch.Tensor] = None, # (B, T, latent_dim)
        z_image_seq: Optional[torch.Tensor] = None  # (B, T, latent_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_error: (B,) Combined prediction error
            sentence_errors: (B, T-1)
            cross_errors: (B, T)
        """
        B, T, D = z_sequence.shape

        # T==1 is common in the looped core (single latent frame). Empty
        # slices make mse_loss → NaN and freeze training at log(vocab).
        if T < 2:
            z_hat = self.sentence_pred(z_sequence)  # (B, 1, D) via Linear on last dim... 
            # SentencePredictor expects (B, D) or (B, T, D) — Linear applies on last dim
            if z_sequence.dim() == 3:
                z_hat = self.sentence_pred(z_sequence.reshape(B * T, D)).reshape(B, T, D)
            sentence_errors = F.mse_loss(z_hat, z_sequence, reduction="none").mean(dim=-1)
            p_hat_seq = self.paragraph_pred(z_sequence)
            para_errors = F.mse_loss(p_hat_seq, z_sequence, reduction="none").mean(dim=-1)
        else:
            # 1. Sentence Prediction (Predict next step)
            z_in = z_sequence[:, :-1, :].reshape(B * (T - 1), D)
            z_hat_seq = self.sentence_pred(z_in).reshape(B, T - 1, D)
            actual_next = z_sequence[:, 1:, :]
            sentence_errors = F.mse_loss(z_hat_seq, actual_next, reduction="none").mean(dim=-1)

            # 2. Paragraph Prediction (Sequence-level)
            p_hat_seq = self.paragraph_pred(z_sequence)
            para_errors = F.mse_loss(
                p_hat_seq[:, :-1], z_sequence[:, 1:], reduction="none"
            ).mean(dim=-1)
        
        # 3. Cross-Modal Prediction
        cross_errors = torch.zeros(B, T, device=z_sequence.device)
        
        if z_audio_seq is not None or z_image_seq is not None:
            z_hat_audio, z_hat_image = self.cross_modal_pred(z_sequence)
            
            if z_audio_seq is not None:
                err_a = F.mse_loss(z_hat_audio, z_audio_seq, reduction='none').mean(dim=-1)
                cross_errors += err_a
            if z_image_seq is not None:
                err_i = F.mse_loss(z_hat_image, z_image_seq, reduction='none').mean(dim=-1)
                cross_errors += err_i
        
        # Combine all errors (weighted as per formula: 0.5*S + 0.3*P + 0.2*C)
        # Taking mean over time steps for a single scalar per batch
        total_sentence_error = sentence_errors.mean(dim=-1)   # (B,)
        total_para_error = para_errors.mean(dim=-1)           # (B,)
        total_cross_error = cross_errors.mean(dim=-1)         # (B,)
        
        total_error = (0.5 * total_sentence_error) + (0.3 * total_para_error) + (0.2 * total_cross_error)
        
        return total_error, sentence_errors, cross_errors