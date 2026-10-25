"""
CEM Data: DataLoaders for all phases.
"""

from torch.utils.data import DataLoader
from .dataset import PretrainDataset, SFTDataset, DPODataset

def pretrain_collate(batch):
    return {
        "char_ids": torch.stack([b["char_ids"] for b in batch]),
        "target_ids": torch.stack([b["target_ids"] for b in batch])
    }

def sft_collate(batch):
    # Pad sequences for instruction tuning
    max_len = max(b["char_ids"].size(0) for b in batch)
    char_ids, target_ids = [], []
    for b in batch:
        pad_len = max_len - b["char_ids"].size(0)
        char_ids.append(torch.nn.functional.pad(b["char_ids"], (0, pad_len), value=0))
        target_ids.append(torch.nn.functional.pad(b["target_ids"], (0, pad_len), value=-100))
    return {"char_ids": torch.stack(char_ids), "target_ids": torch.stack(target_ids)}

def dpo_collate(batch):
    # Pad chosen and rejected sequences
    def pad_list(key):
        max_len = max(b[key].size(0) for b in batch)
        return torch.stack([torch.nn.functional.pad(b[key], (0, max_len - b[key].size(0)), value=0) for b in batch])
    return {"chosen_ids": pad_list("chosen_ids"), "rejected_ids": pad_list("rejected_ids")}

def get_dataloader(phase: str, dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    collate_fn = {
        "pretrain": pretrain_collate,
        "midtrain": sft_collate,
        "sft": sft_collate,
        "dpo": dpo_collate
    }[phase]
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)