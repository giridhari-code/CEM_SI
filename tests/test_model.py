"""
Unit Tests for the Full CEM Core Model.
"""

import unittest
import torch
from cem.core.model import CEM, ModelOutput

class TestCEMModel(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 20
        self.vocab_size = 1000
        self.latent_dim = 128 # Smaller dim for faster test
        
        self.model = CEM(
            vocab_size=self.vocab_size, 
            latent_dim=self.latent_dim,
            max_loops=3
        )
        
        self.char_ids = torch.randint(0, 256, (self.B, self.T))
        self.target_tokens = torch.randint(0, self.vocab_size, (self.B, self.T))

    def test_forward_pass_training(self):
        """Test forward pass with targets (Training mode)."""
        self.model.train()
        output = self.model(
            char_ids=self.char_ids, 
            target_tokens=self.target_tokens
        )
        
        self.assertIsInstance(output, ModelOutput)
        self.assertIsNotNone(output.logits)
        self.assertIsNotNone(output.loss)
        
        # Causal teacher forcing → logits length T-1 when targets provided
        self.assertEqual(output.logits.shape[0], self.B)
        self.assertEqual(output.logits.shape[2], self.vocab_size)
        self.assertGreaterEqual(output.logits.shape[1], self.T - 1)

        # Check if loss is a valid scalar
        self.assertTrue(output.loss.item() > 0)
        self.assertTrue(torch.isfinite(output.loss))

    def test_forward_pass_inference(self):
        """Inference: no targets → still emit logits (chat/generate path)."""
        self.model.eval()
        output = self.model(char_ids=self.char_ids)
        
        self.assertIsInstance(output, ModelOutput)
        # Chat/serve path always decodes when char_ids present
        self.assertIsNotNone(output.logits)
        self.assertEqual(output.logits.shape, (self.B, self.T, self.vocab_size))
        self.assertIsNone(output.loss)

    def test_early_stopping_loop(self):
        """Test if L7 (Meta) stops the loop early based on confidence."""
        self.model.eval()
        output = self.model(char_ids=self.char_ids)
        
        # Check metrics
        loops_run = output.metrics["loops_run"]
        self.assertTrue(1 <= loops_run <= 3) # Should be between 1 and max_loops

    def test_state_carry_over(self):
        """State is returned and can be fed into the next step."""
        self.model.eval()
        out1 = self.model(char_ids=self.char_ids)
        self.assertIsNotNone(out1.state)
        self.assertIn("m_prev", out1.state)
        out2 = self.model(char_ids=self.char_ids, state=out1.state)
        self.assertIsNotNone(out2.state)
        self.assertTrue(torch.isfinite(out2.state["m_prev"]).all())

if __name__ == '__main__':
    unittest.main()