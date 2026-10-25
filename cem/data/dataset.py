"""CEM datasets — company-standard phase formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from cem.utils.tokenizer import CharTokenizer


class PretrainDataset(Dataset):
    """Phase 1: next-token LM on raw text."""

    def __init__(self, txt_path: str, tokenizer: CharTokenizer, max_seq_len: int = 128):
        path = Path(txt_path)
        if not path.is_file():
            raise FileNotFoundError(f"pretrain text not found: {path}")
        text = path.read_text(encoding="utf-8")
        if len(text) < max_seq_len + 1:
            text = (text + "\n") * ((max_seq_len // max(len(text), 1)) + 2)
        ids = tokenizer.encode(text, add_special_tokens=False)
        stride = max(1, max_seq_len // 2)
        self.chunks: List[torch.Tensor] = []
        for i in range(0, max(1, len(ids) - max_seq_len), stride):
            self.chunks.append(torch.tensor(ids[i : i + max_seq_len], dtype=torch.long))
        if not self.chunks:
            raise ValueError(f"no chunks from {path} (text too short?)")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class SFTDataset(Dataset):
    """Phase 2–3: midtrain / SFT JSONL {prompt, response}."""

    def __init__(self, jsonl_path: str, tokenizer: CharTokenizer, max_seq_len: int = 256):
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"sft/mid jsonl not found: {path}")
        self.rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"empty jsonl: {path}")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.rows[idx]
        prompt = str(item.get("prompt", item.get("instruction", "")))
        response = str(item.get("response", item.get("output", "")))
        full = prompt + response
        ids = self.tokenizer.encode(full, add_special_tokens=True)[: self.max_seq_len]
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = min(len(prompt_ids), len(ids))
        labels = list(ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        if len(ids) < 2:
            ids = ids + [0]
            labels = labels + [-100]
        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:], dtype=torch.long),
        }


class DPODataset(Dataset):
    """Phase 4: preference pairs {prompt, chosen, rejected}."""

    def __init__(self, jsonl_path: str, tokenizer: CharTokenizer, max_seq_len: int = 256):
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"dpo jsonl not found: {path}")
        self.rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"empty dpo jsonl: {path}")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_pair(self, prompt: str, completion: str) -> torch.Tensor:
        text = prompt + completion
        ids = self.tokenizer.encode(text, add_special_tokens=True)[: self.max_seq_len]
        if len(ids) < 2:
            ids = ids + [0]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.rows[idx]
        prompt = str(item.get("prompt", ""))
        chosen = str(item.get("chosen", item.get("response_chosen", "")))
        rejected = str(item.get("rejected", item.get("response_rejected", "")))
        return {
            "chosen_ids": self._encode_pair(prompt, chosen),
            "rejected_ids": self._encode_pair(prompt, rejected),
        }
