"""
CEM Infra: Checkpoint Manager.
Handles saving/loading models safely with Safetensors.
"""

import os
import torch
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class CheckpointManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, is_main_process: bool = True):
        """Saves model weights (safetensors) and optimizer state (.pt)."""
        if not is_main_process:
            return
            
        # 1. Unwrap FSDP to get clean state_dict
        state_dict = model.state_dict()
        if isinstance(model, FSDP):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = model.state_dict()

        # 2. Save Weights (Safetensors)
        weights_path = os.path.join(self.output_dir, f"model_step_{step}.safetensors")
        clean_state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(clean_state_dict, weights_path)

        # 3. Save Optimizer State (.pt)
        opt_path = os.path.join(self.output_dir, f"optimizer_step_{step}.pt")
        torch.save(optimizer.state_dict(), opt_path)

        print(f"💾 Saved checkpoint at step {step} -> {weights_path}")

    def load(self, model: torch.nn.Module, checkpoint_path: str, device: str = "cpu"):
        """Loads Safetensors weights into model."""
        state_dict = load_file(checkpoint_path, device=device)
        model.load_state_dict(state_dict)
        print(f"📂 Loaded weights from {checkpoint_path}")