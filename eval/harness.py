"""
CEM Eval: Harness.
Evaluates model on multiple-choice (MCQ) tasks using log-probabilities.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

class EvalHarness:
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _get_choice_logprob(self, prompt: str, choice: str) -> float:
        """
        Calculates the log probability of a specific choice given a prompt.
        Used for MMLU / ARC style evaluations.
        """
        # 1. Encode prompt and choice separately
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
        
        # 2. Concatenate them (Input = Prompt + Choice)
        input_ids = torch.tensor([prompt_ids + choice_ids], device=self.device)
        target_ids = input_ids.clone()
        
        # Mask out prompt tokens (we only care about probability of choice)
        target_ids[:, :len(prompt_ids)] = -100 
        
        # 3. Forward pass
        with torch.no_grad():
            out = self.model(char_ids=input_ids, target_tokens=target_ids)
            
        # 4. Convert CrossEntropy loss back to LogProb
        # CE = -LogProb / num_tokens
        num_choice_tokens = len(choice_ids)
        if num_choice_tokens == 0:
            return -float('inf')
            
        # out.loss is average CE loss. Total CE = loss * num_tokens
        # Total LogProb = -Total CE
        total_logprob = -(out.loss.item() * num_choice_tokens)
        return total_logprob

    def evaluate_batch(self, batch: List[Dict]) -> Dict[str, float]:
        """
        Evaluates a batch of MCQ questions.
        Format: {"question": "...", "options": ["A", "B", "C", "D"], "answer_idx": 2}
        """
        correct = 0
        total = len(batch)
        
        for item in batch:
            prompt = item["question"]
            options = item["options"]
            answer_idx = item["answer_idx"]
            
            # Get logprob for each option
            logprobs = [self._get_choice_logprob(prompt, opt) for opt in options]
            
            # Select the option with the highest logprob (most likely)
            predicted_idx = int(torch.tensor(logprobs).argmax().item())
            
            if predicted_idx == answer_idx:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}