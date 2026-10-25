"""
Unit Tests for CEM Layer 1 (Multimodal I/O).
"""

import unittest
import torch
from cem.layers.l1_io import (
    NeuralBoundaryDetector, TextEncoder, NeuralVAD, 
    VoiceEncoder, ImageEncoder, TextDecoder, NeuralTTS, L1MultimodalIO
)

class TestL1IO(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 50
        self.latent_dim = 512
        self.char_vocab = 256
        self.vocab_size = 10000
        
        self.char_ids = torch.randint(0, 256, (self.B, self.T))
        self.waveform = torch.randn(self.B, 1, 16000) # 1 sec audio
        self.image = torch.randn(self.B, 3, 224, 224)
        self.target_tokens = torch.randint(0, 10000, (self.B, 20))

    def test_boundary_detector(self):
        model = NeuralBoundaryDetector(vocab_size=self.char_vocab)
        out = model(self.char_ids)
        self.assertEqual(out.shape, (self.B, self.T, 1))
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1)) # Sigmoid output

    def test_text_encoder(self):
        model = TextEncoder(vocab_size=self.char_vocab, latent_dim=self.latent_dim)
        out = model(self.char_ids)
        self.assertEqual(out.shape, (self.B, self.latent_dim))
        # Check L2 Normalization
        norms = out.norm(p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_voice_encoder(self):
        model = VoiceEncoder(latent_dim=self.latent_dim)
        out = model(self.waveform)
        self.assertEqual(out.shape, (self.B, self.latent_dim))

    def test_image_encoder(self):
        model = ImageEncoder(latent_dim=self.latent_dim)
        out = model(self.image)
        self.assertEqual(out.shape, (self.B, self.latent_dim))

    def test_text_decoder(self):
        model = TextDecoder(latent_dim=self.latent_dim, vocab_size=self.vocab_size)
        z_fused = torch.randn(self.B, self.latent_dim)
        out = model(z_fused, self.target_tokens)
        self.assertEqual(out.shape, (self.B, 20, self.vocab_size))

    def test_l1_harness_missing_modality(self):
        """Test that L1 works even if image/audio is missing."""
        model = L1MultimodalIO(latent_dim=self.latent_dim)
        z_text, z_audio, z_image = model.encode(char_ids=self.char_ids)
        
        self.assertIsNotNone(z_text)
        self.assertIsNone(z_audio)
        self.assertIsNone(z_image)
        self.assertEqual(z_text.shape, (self.B, self.latent_dim))

if __name__ == '__main__':
    unittest.main()