"""
CEM Script: Inference.
Generates text autoregressively using a trained Safetensors checkpoint.
"""

import os
import argparse
import torch
from safetensors.torch import load_file

from cem.core.model import CEM
from cem.utils.tokenizer import CharTokenizer
from cem.utils.config_loader import load_config
from cem.utils.logger import setup_logger

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, device="cpu"):
    """
    Generates text autoregressively.
    """
    model.eval()
    
    # 1. Encode Prompt
    char_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
    
    # 2. Initial State (For L7/L8 consciousness and memory)
    state = model.init_state(batch_size=1, device=device)
    
    generated_ids = []
    
    print(f"\nUser: {prompt}")
    print("CEM: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward Pass (Passing full sequence so far)
            output = model(char_ids=input_tensor, state=state)
            
            # Get logits for the last token
            next_token_logits = output.logits[:, -1, :]
            
            # Greedy Sampling (argmax) - Can be replaced with top-k or temperature sampling
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Stop if End of Sequence (EOS) token
            if next_token_id == tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token_id)
            
            # Print character immediately
            char = tokenizer.decode([next_token_id])
            print(char, end="", flush=True)
            
            # Update input for next iteration (append generated token)
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
            
            # Update state (Carry over memory and self-model)
            state = output.state

    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEM Inference")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .safetensors file")
    parser.add_argument("--prompt", type=str, default="Ravi ko diabetes hai.", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    
    # 2. Setup Logger & Device
    logger = setup_logger("CEM-Inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. Initialize Tokenizer
    tokenizer = CharTokenizer()

    # 4. Initialize Model
    model = CEM(
        vocab_size=model_cfg["vocab_size"],
        latent_dim=model_cfg["latent_dim"],
        action_dim=model_cfg["action_dim"],
        max_loops=model_cfg["max_loops"]
    ).to(device)

    # 5. Load Safetensors
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found at {args.checkpoint}")
        exit(1)
        
    logger.info(f"Loading weights from {args.checkpoint}...")
    state_dict = load_file(args.checkpoint, device=device)
    model.load_state_dict(state_dict)
    logger.info("Weights loaded successfully.")

    # 6. Generate Text
    generate_text(
        model=model, 
        tokenizer=tokenizer, 
        prompt=args.prompt, 
        max_new_tokens=args.max_tokens, 
        device=device
    )