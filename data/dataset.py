"""
CEM Data: Dataset Loader (All Phases).
Supports Pre-training (TXT), Mid-training (JSONL), SFT (JSONL), and DPO (JSONL).
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from cem.utils.tokenizer import CharTokenizer

class PretrainDataset(Dataset):
    """Phase 1: Raw Text for Next-Character Prediction."""
    def __init__(self, txt_path: str, tokenizer: CharTokenizer, max_seq_len: int = 128):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        all_ids = tokenizer.encode(text, add_special_tokens=False)
        self.chunks = []
        for i in range(0, len(all_ids) - max_seq_len, max_seq_len // 2):
            self.chunks.append(torch.tensor(all_ids[i : i + max_seq_len], dtype=torch.long))

    def __len__(self): return len(self.chunks)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {"char_ids": chunk[:-1], "target_ids": chunk[1:]}

class SFTDataset(Dataset):
    """Phase 2 & 3: Instruction Tuning (Mid-training & SFT)."""
    def __init__(self, jsonl_path: str, tokenizer: CharTokenizer, max_seq_len: int = 256):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item.get("prompt", "") + item.get("response", "")
        char_ids = self.tokenizer.encode(text, add_special_tokens=True)[:self.max_seq_len]
        
        # Mask prompt tokens (so model only learns to generate response)
        prompt_len = len(self.tokenizer.encode(item.get("prompt", ""), add_special_tokens=False))
        labels = char_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len # Ignore prompt in loss
        
        return {
            "char_ids": torch.tensor(char_ids[:-1], dtype=torch.long),
            "target_ids": torch.tensor(labels[1:], dtype=torch.long)
        }

class DPODataset(Dataset):
    """Phase 4: Direct Preference Optimization."""
    def __init__(self, jsonl_path: str, tokenizer: CharTokenizer, max_seq_len: int = 256):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        def encode_pair(resp):
            text = prompt + resp
            ids = self.tokenizer.encode(text, add_special_tokens=True)[:self.max_seq_len]
            return torch.tensor(ids, dtype=torch.long)
            
        return {
            "chosen_ids": encode_pair(chosen),
            "rejected_ids": encode_pair(rejected)
        }