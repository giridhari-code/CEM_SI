"""
CEM Data: Preprocessing.
Converts raw inputs (Text, Audio, Image) into tensor formats.
"""

import torch
import torchaudio.transforms as T
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Optional

# ==========================================
# TEXT PREPROCESSING (L1a)
# ==========================================
def chunk_text(text: str) -> List[str]:
    """
    Splits raw text into sentences (Neural Boundary Detector ka fallback).
    """
    import re
    sentences = re.split(r'[.!?।]\s*', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

# ==========================================
# AUDIO PREPROCESSING (L1d)
# ==========================================
class AudioPreprocessor:
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80):
        self.mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Converts raw waveform to Log-Mel Spectrogram.
        Input: (1, T) -> Output: (1, n_mels, T')
        """
        # Ensure waveform is (Channels, Time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel = self.mel_spec(waveform)
        log_mel = torch.log(mel + 1e-6)
        return log_mel

# ==========================================
# IMAGE PREPROCESSING (L1e)
# ==========================================
class ImagePreprocessor:
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # Converts to [0, 1] and (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])
        
    def __call__(self, image_path: str) -> torch.Tensor:
        """
        Loads image from path and applies transformations.
        Input: str -> Output: (3, H, W)
        """
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)