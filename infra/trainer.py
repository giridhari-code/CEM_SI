"""CEM infra trainer — works with transformer-free `cem.core.model.CEM` only."""

from __future__ import annotations

import torch
from tqdm import tqdm


class CEMTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        grad_accum_steps: int = 1,
        max_skip_steps: int = 50,
        clip_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_skip_steps = max_skip_steps
        self.clip_grad_norm = clip_grad_norm
        self.skipped_steps = 0

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        n_ok = 0
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(dataloader, desc="Training")

        for step, batch in enumerate(progress):
            # Support both collate key styles
            if "input_ids" in batch:
                char_ids = batch["input_ids"].to(self.device)
                target_ids = batch["labels"].to(self.device)
            else:
                char_ids = batch["char_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

            out = self.model(char_ids=char_ids, target_tokens=target_ids)
            loss = out.loss

            if loss is None or not torch.isfinite(loss):
                self.skipped_steps += 1
                progress.set_postfix(skip=self.skipped_steps, loss="nan")
                if self.skipped_steps > self.max_skip_steps:
                    raise RuntimeError("Too many NaN steps — abort")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.skipped_steps = 0
            loss = loss / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * self.grad_accum_steps
            n_ok += 1
            progress.set_postfix(loss=f"{loss.item() * self.grad_accum_steps:.4f}")

        return total_loss / max(n_ok, 1)
