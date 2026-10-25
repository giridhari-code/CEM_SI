"""
Unit Tests for CEM Layer 1.5 and Layer 2 (Fusion & Representation).
"""

import unittest
import torch
from cem.layers.l1_5_fusion import CrossModalFusion, GatedModalityNetwork, UnifiedSemanticEncoder, L15_Fusion_and_L2

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.latent_dim = 512
        
        self.z_text = torch.randn(self.B, self.latent_dim)
        self.z_audio = torch.randn(self.B, self.latent_dim)
        self.z_image = torch.randn(self.B, self.latent_dim)

    def test_cross_modal_fusion(self):
        model = CrossModalFusion(latent_dim=self.latent_dim)
        z_fused = model(self.z_text, self.z_audio, self.z_image)
        self.assertEqual(z_fused.shape, (self.B, self.latent_dim))

    def test_gated_network(self):
        model = GatedModalityNetwork(latent_dim=self.latent_dim)
        z_fused = torch.randn(self.B, self.latent_dim)
        # Test with all modalities
        z_out = model(self.z_text, z_fused, self.z_audio, self.z_image)
        self.assertEqual(z_out.shape, (self.B, self.latent_dim))

    def test_l15_harness_missing_modality(self):
        """Test that Fusion works if only text is present."""
        model = L15_Fusion_and_L2(latent_dim=self.latent_dim)
        z = model(z_text=self.z_text, z_audio=None, z_image=None)
        self.assertEqual(z.shape, (self.B, self.latent_dim))

if __name__ == '__main__':
    unittest.main()