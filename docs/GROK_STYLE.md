# CEM × Grok-style (method mirror, not xAI weights)

CEM is **not** Grok and has **no Grok weights**. This doc mirrors **public xAI product posture** into CEM training + prompts.

## What we mirror

| Grok / xAI idea | CEM implementation |
|-----------------|-------------------|
| Truth-seeking | SFT/DPO prefer correct + “I don’t know” over fluent nonsense |
| Low sycophancy | Reject pairs that agree with false claims to please the user |
| Low corporate filler | System prompt + SFT demos: short, blunt, no PR speak |
| Humor optional | Only in meta/chat demos — never on safety or facts |
| Maximum helpfulness | Answer the ask; give next step when blocked |
| Not a scold | Refusals: one clear line, not a sermon |

## What we do **not** copy

- Grok model architecture / MoE / data  
- Live X retrieval stack  
- Brand voice as a trademark imitation in shipping UI copy  

## Training map

```text
pretrain     → language structure
midtrain     → domains (ops, science, India chat)
sft_grok     → truth-seeking + anti-sycophancy demos
dpo_grok     → prefer honest over agreeable-wrong
```

Configs: `configs/sft_grok.yaml`, `configs/dpo_grok.yaml`  
Data: `data/raw/sft_grok.jsonl`, `data/raw/dpo_grok.jsonl`  
Prompt: `cem/prompts/system_default.txt`
