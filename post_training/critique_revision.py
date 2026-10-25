"""
CEM Post-Training: Critique and Revision.
Implements Constitutional AI (Stage 1).
"""

from typing import List, Dict

class ConstitutionalAI:
    def __init__(self, constitution: List[str]):
        """
        constitution: List of rules (e.g., "Do not provide medical advice", "Be helpful")
        """
        self.constitution = constitution

    def build_critique_prompt(self, prompt: str, response: str) -> str:
        """Creates a prompt asking the model to critique its own response."""
        rules_text = "\n".join([f"- {rule}" for rule in self.constitution])
        
        critique_prompt = f"""
        You are a helpful AI. Critique the following response based on the rules.
        
        Rules:
        {rules_text}
        
        User Prompt: {prompt}
        AI Response: {response}
        
        Critique (point out any violations):
        """
        return critique_prompt

    def build_revision_prompt(self, prompt: str, response: str, critique: str) -> str:
        """Creates a prompt asking the model to revise its response."""
        revision_prompt = f"""
        User Prompt: {prompt}
        Original Response: {response}
        Critique: {critique}
        
        Please revise the response to fix the issues mentioned in the critique.
        Revised Response:
        """
        return revision_prompt

    def process_batch(self, model, tokenizer, batch: List[Dict]) -> List[Dict]:
        """
        Simulates the Critique -> Revision loop for a batch.
        (In a real setup, this calls the model's generate function)
        """
        processed_data = []
        
        for item in batch:
            prompt = item["prompt"]
            original_response = item["original_response"]
            
            # 1. Generate Critique (using model)
            # critique = model.generate(build_critique_prompt(...))
            critique = "[Simulated Critique: Response lacks safety warning.]"
            
            # 2. Generate Revision
            # revised = model.generate(build_revision_prompt(...))
            revised = "[Simulated Revised Response: Includes safety warning.]"
            
            processed_data.append({
                "prompt": prompt,
                "original_response": original_response,
                "critique": critique,
                "revised_response": revised
            })
            
        return processed_data