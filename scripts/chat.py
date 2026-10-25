#!/usr/bin/env python3
"""
CEM chat — simple input / output loop.

  python scripts/chat.py --checkpoint checkpoints/tiny/checkpoint_latest.pt
  python scripts/chat.py --checkpoint ... --prompt "Hello"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cem.core.model import CEM
from cem.utils.tokenizer import CharTokenizer


def load_model(ckpt_path: str, device: torch.device) -> CEM:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config") or {}
    # defaults if old ckpt
    vocab = 256
    latent = 64
    action = 32
    loops = 1
    if isinstance(cfg, dict):
        # TrainConfig doesn't store model dims — use filename heuristics / defaults
        pass
    model = CEM(vocab_size=vocab, latent_dim=latent, action_dim=action, max_loops=loops)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def generate(
    model: CEM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new: int = 80,
    device: torch.device = torch.device("cpu"),
    temperature: float = 0.8,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    state = model.init_state(1, device)
    out_ids = []

    for _ in range(max_new):
        out = model(char_ids=x, state=state)
        if out.logits is None:
            break
        logits = out.logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        if next_id == tokenizer.eos_token_id:
            break
        out_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if out.state is not None:
            state = out.state

    return tokenizer.decode(out_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default=None, help="Single-shot; else interactive")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok = CharTokenizer()
    model = load_model(args.checkpoint, device)

    system = ""
    sp = ROOT / "cem" / "prompts" / "system_default.txt"
    if sp.is_file():
        system = sp.read_text(encoding="utf-8").strip() + "\n"

    def once(user: str) -> str:
        prompt = f"{system}User: {user}\nAssistant: " if system else user
        return generate(model, tok, prompt, args.max_tokens, device)

    if args.prompt is not None:
        print("Input:", args.prompt)
        print("Output:", once(args.prompt))
        return

    print("CEM chat (empty line or /quit to exit)")
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user in ("/quit", "/exit", "quit"):
            break
        print("CEM>", once(user))


if __name__ == "__main__":
    main()
