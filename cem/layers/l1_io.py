"""
CEM Layer 1: Multimodal I/O.
Contains all encoders (Text, Voice, Image) and decoders (Text, TTS).
Fully Neural (Transformer-Free) using BiLSTM, CNN, and GRU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchaudio.transforms as T
except ImportError:  # text-only train/smoke without torchaudio
    T = None
from typing import Optional, Tuple

# ==========================================
# 1. TEXT PATH (L1a, L1b)
# ==========================================

class NeuralBoundaryDetector(nn.Module):
    """
    L1a: Predicts sentence boundaries (period, question mark, etc.) 
    character-by-character using BiLSTM.

    Mathematical Formula:
    ---------------------
    e_t = Embed(x_t)
    h_t = BiLSTM(e_1, ..., e_T)
    P(boundary_t) = sigmoid(W_cls * h_t + b_cls)
    """
    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # Input: (B, T) -> Output: (B, T, 1)
        x = self.embedding(char_ids)             # e_t = Embed(x_t)
        out, _ = self.bilstm(x)                  # h_t = BiLSTM(e)
        logits = self.classifier(out)            # W_cls * h_t + b_cls
        return torch.sigmoid(logits)             # P(boundary_t)


class TextEncoder(nn.Module):
    """
    L1b: Encodes a sequence of characters into a semantic latent vector.

    Mathematical Formula:
    ---------------------
    e_t = Embed(x_t)
    h_t = BiLSTM(e_1, ..., e_T)
    z_raw = (1/T) * sum(h_t)          # Mean Pooling
    z_text = LayerNorm(W_proj * z_raw + b_proj)
    z_text = z_text / ||z_text||_2    # L2 Normalize
    """
    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, hidden_dim: int = 128, latent_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # Input: (B, T) -> Output: (B, latent_dim)
        x = self.embedding(char_ids)             # e_t
        out, _ = self.bilstm(x)                  # h_t
        z_raw = out.mean(dim=1)                  # z_raw = (1/T) * sum(h_t)
        z_text = self.projection(z_raw)          # W_proj * z_raw + b_proj
        return F.normalize(z_text, p=2, dim=-1)  # z_text / ||z_text||_2


# ==========================================
# 2. VOICE PATH (L1c, L1d)
# ==========================================

class NeuralVAD(nn.Module):
    """
    L1c: Voice Activity Detection. Detects if the audio contains speech.

    Mathematical Formula:
    ---------------------
    C = GlobalPool(Conv1D(Waveform))
    P(speech) = sigmoid(W_fc * C + b_fc)
    """
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, T) -> Output: (B, 1)
        x = self.conv_layers(waveform)           # Conv1D + Pooling
        x = x.mean(dim=-1)                       # Global Average Pooling
        return torch.sigmoid(self.fc(x))         # P(speech)


class VoiceEncoder(nn.Module):
    """
    L1d: Encodes raw audio into a semantic latent vector using CNN+GRU.

    Mathematical Formula:
    ---------------------
    M = MelSpectrogram(Waveform)
    C = CNN(M)
    H = GRU(C)
    z_audio = Norm(W_proj * Mean(H) + b_proj)
    """
    def __init__(self, latent_dim: int = 512, n_mels: int = 80, sample_rate: int = 16000):
        super().__init__()
        self.n_mels = n_mels
        if T is not None:
            self.mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        else:
            self.mel_spec = None  # text-only installs without torchaudio
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((n_mels, 1)) 
        )
        self.gru = nn.GRU(32 * n_mels, 128, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(128 * 2, latent_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, T) -> Output: (B, latent_dim)
        if self.mel_spec is None:
            raise ImportError("torchaudio required for VoiceEncoder")
        mel = self.mel_spec(waveform)            # M = MelSpectrogram(W)
        mel = torch.log(mel + 1e-6).unsqueeze(1) # Log-Mel
        
        c = self.cnn(mel)                        # C = CNN(M)
        c = c.squeeze(-1).permute(0, 2, 1) 
        
        out, _ = self.gru(c)                     # H = GRU(C)
        z_raw = out.mean(dim=1)                  # Mean(H)
        
        z_audio = self.projection(z_raw)         # W_proj * Mean(H) + b
        return F.normalize(z_audio, p=2, dim=-1) # Normalize


# ==========================================
# 3. IMAGE PATH (L1e)
# ==========================================

class ImageEncoder(nn.Module):
    """
    L1e: Encodes an image into a semantic latent vector using CNN.

    Mathematical Formula:
    ---------------------
    F = CNN(Image)
    z_image = Norm(W_proj * Flatten(GlobalPool(F)) + b_proj)
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projection = nn.Linear(256, latent_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Input: (B, 3, H, W) -> Output: (B, latent_dim)
        x = self.cnn(image)                      # F = CNN(Image)
        x = x.flatten(1)                         # Flatten(GlobalPool(F))
        z_image = self.projection(x)             # W_proj * x + b
        return F.normalize(z_image, p=2, dim=-1) # Normalize


# ==========================================
# 4. OUTPUT GENERATION (L1h, L1i)
# ==========================================

class TextDecoder(nn.Module):
    """
    L1h: Decodes latent vector into text using LSTM.

    Mathematical Formula:
    ---------------------
    h_0 = W_z * z_fused + b_z
    h_t = LSTM([z_fused, Embed(word_{t-1})], h_{t-1})
    P(word_t) = Softmax(W_out * h_t + b_out)
    """
    def __init__(self, latent_dim: int = 512, vocab_size: int = 10000, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(latent_dim + hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z_fused: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        B, T = target_tokens.shape
        
        h0 = self.latent_proj(z_fused).unsqueeze(0).repeat(2, 1, 1) # h_0
        c0 = torch.zeros_like(h0)
        
        inputs = self.embedding(target_tokens)                      # Embed(word)
        z_expanded = z_fused.unsqueeze(1).repeat(1, T, 1)           # z_fused
        
        lstm_in = torch.cat([inputs, z_expanded], dim=-1)
        out, _ = self.lstm(lstm_in, (h0, c0))                       # h_t
        
        return self.out(out)                                        # Softmax(W_out * h_t)


class NeuralTTS(nn.Module):
    """
    L1i: Generates Mel-Spectrogram from latent text representation.

    Mathematical Formula:
    ---------------------
    M = RNN(TextEmbeddings)
    Wave = Vocoder(M)
    """
    def __init__(self, latent_dim: int = 512, n_mels: int = 80):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, 256, num_layers=2, batch_first=True)
        self.mel_out = nn.Linear(256, n_mels)

    def forward(self, text_latents: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(text_latents)         # M = RNN(Text)
        return self.mel_out(out)                # Mel-Spectrogram


# ==========================================
# 5. L1 HARNESS (Integration)
# ==========================================

class L1MultimodalIO(nn.Module):
    """
    Main Harness for Layer 1. 
    Integrates Encoders (L1a-e) and Decoders (L1h-i).
    """
    def __init__(self, vocab_size: int = 10000, char_vocab_size: int = 256, latent_dim: int = 512):
        super().__init__()
        
        self.boundary_detector = NeuralBoundaryDetector(vocab_size=char_vocab_size)
        self.text_encoder = TextEncoder(vocab_size=char_vocab_size, latent_dim=latent_dim)
        self.vad = NeuralVAD()
        self.voice_encoder = VoiceEncoder(latent_dim=latent_dim)
        self.image_encoder = ImageEncoder(latent_dim=latent_dim)
        
        self.text_decoder = TextDecoder(latent_dim=latent_dim, vocab_size=vocab_size)
        self.tts = NeuralTTS(latent_dim=latent_dim)

    def encode(self, char_ids=None, waveform=None, image=None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        z_text = self.text_encoder(char_ids) if char_ids is not None else None
        z_audio = self.voice_encoder(waveform) if waveform is not None else None
        z_image = self.image_encoder(image) if image is not None else None
        return z_text, z_audio, z_image

    def decode_text(self, z_fused: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.text_decoder(z_fused, target_tokens)

    def decode_audio(self, text_latents: torch.Tensor) -> torch.Tensor:
        return self.tts(text_latents)