"""Collate + DataLoader factory."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


def collate_lm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    bsz = len(batch)
    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, row in enumerate(batch):
        n = row["input_ids"].size(0)
        input_ids[i, :n] = row["input_ids"]
        labels[i, :n] = row["labels"]
    return {"input_ids": input_ids, "labels": labels}


def collate_dpo(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    def pad(key: str) -> torch.Tensor:
        max_len = max(x[key].size(0) for x in batch)
        out = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, row in enumerate(batch):
            n = row[key].size(0)
            out[i, :n] = row[key]
        return out

    return {"chosen_ids": pad("chosen_ids"), "rejected_ids": pad("rejected_ids")}


def get_dataloader(
    phase: str,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    collate = collate_dpo if phase == "dpo" else collate_lm
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )
