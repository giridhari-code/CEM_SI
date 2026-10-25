"""
CEM Infra: FSDP Wrapper.
Wraps model in PyTorch Fully Sharded Data Parallel for multi-GPU.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

class FSDPWrapper:
    @staticmethod
    def wrap(model: nn.Module, is_distributed: bool = True) -> nn.Module:
        if not is_distributed:
            return model
            
        # 1. Mixed Precision (bf16 for faster training)
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        
        # 2. Auto-wrap policy (breaks model into chunks of ~1M params)
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000_000
        )
        
        # 3. Wrap Model
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD, # ZeRO-3 equivalent
            device_id=torch.cuda.current_device(),
        )
        
        print("✅ Model wrapped with FSDP (FULL_SHARD)")
        return model