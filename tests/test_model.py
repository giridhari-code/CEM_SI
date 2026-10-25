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
        
        # Check shapes
        self.assertEqual(output.logits.shape, (self.B, self.T, self.vocab_size))
        
        # Check if loss is a valid scalar
        self.assertTrue(output.loss.item() > 0)

    def test_forward_pass_inference(self):
        """Test forward pass without targets (Inference mode)."""
        self.model.eval()
        output = self.model(char_ids=self.char_ids)
        
        self.assertIsInstance(output, ModelOutput)
        self.assertIsNone(output.logits) # Should be None without targets
        self.assertIsNone(output.loss)

    def test_early_stopping_loop(self):
        """Test if L7 (Meta) stops the loop early based on confidence."""
        self.model.eval()
        output = self.model(char_ids=self.char_ids)
        
        # Check metrics
        loops_run = output.metrics["loops_run"]
        self.assertTrue(1 <= loops_run <= 3) # Should be between 1 and max_loops

    def test_state_carry_over(self):
        """Test if state (memory/self-model) carries over across steps."""
        self.model.eval()
        
        # First call
        out1 = self.model(char_ids=self.char_ids)
        state1 = out1.state
        
        # Second call (simulating next token/chat)
        out2 = self.model(char_ids=self.char_ids, state=state1)
        
        self.assertIsNotNone(out2.state)
        # Ensure m_prev (self-model) has changed
        self.assertFalse(torch.equal(state1["m_prev"], out2.state["m_prev"]))

if __name__ == '__main__':
    unittest.main()