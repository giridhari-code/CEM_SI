"""
CEM Script: Evaluate.
Evaluates the model on a test set.
"""

import argparse
import torch
from tqdm import tqdm

from cem.core.model import CEM
from cem.utils.tokenizer import CharTokenizer
from cem.utils.logger import setup_logger
from cem.utils.metrics import MetricsTracker

def evaluate(args):
    logger = setup_logger("CEM-Eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on {device}")

    # Load Model
    model = CEM(vocab_size=args.vocab_size, latent_dim=args.latent_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Dummy Eval Data
    tokenizer = CharTokenizer()
    test_data = [{"text": "Ravi ko diabetes hai.", "target": "Metformin lo."}] * 50
    
    # (Assuming DataLoader is already set up similarly to train.py)
    # dataloader = create_dataloader(...)

    logger.info("Starting Evaluation...")
    metrics = MetricsTracker()
    
    with torch.no_grad():
        # for batch in tqdm(dataloader):
            # (Simulated batch)
            char_ids = torch.randint(0, 256, (4, 20)).to(device)
            target_ids = torch.randint(0, 256, (4, 20)).to(device)
            
            output = model(char_ids=char_ids, target_tokens=target_ids)
            metrics.update(output.logits, target_ids, output.loss)
            
    logger.info(f"Evaluation Results: {metrics.get_metrics()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()
    evaluate(args)