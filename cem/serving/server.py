"""
CEM HTTP server — AERIS-shaped routes so 0zcode / curl can talk the same way.

  POST /v1/chat
  POST /v1/chat/stream   (SSE-ish plain chunks)
  GET  /health
  GET  /readyz
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cem.core.model import CEM
from cem.utils.tokenizer import CharTokenizer

logger = logging.getLogger("cem.serving")


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 80
    temperature: float = 0.8
    model: Optional[str] = None


class ChatResponse(BaseModel):
    id: str = "cem-chat"
    model: str = "CEM"
    output: str
    choices: List[Dict[str, Any]] = Field(default_factory=list)


def _device() -> torch.device:
    if os.environ.get("CEM_DEVICE") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> tuple[CEM, CharTokenizer, torch.device]:
    device = _device()
    tok = CharTokenizer()
    vocab, latent, action, loops = 256, 64, 32, 1
    ckpt_path = os.environ.get("CEM_CHECKPOINT", "checkpoints/tiny/checkpoint_latest.pt")
    model = CEM(vocab_size=vocab, latent_dim=latent, action_dim=action, max_loops=loops)
    p = Path(ckpt_path)
    if p.is_file():
        ckpt = torch.load(str(p), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        logger.info("loaded checkpoint %s", p)
    else:
        logger.warning("no checkpoint at %s — random weights (demo only)", p)
    model.to(device).eval()
    return model, tok, device


@torch.no_grad()
def _generate(
    model: CEM,
    tok: CharTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    ids = tok.encode(prompt, add_special_tokens=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    state = model.init_state(1, device)
    out_ids: List[int] = []
    for _ in range(max_tokens):
        out = model(char_ids=x, state=state)
        if out.logits is None:
            break
        logits = out.logits[:, -1, :] / max(temperature, 1e-5)
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        if next_id == tok.eos_token_id:
            break
        out_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if out.state is not None:
            state = out.state
    return tok.decode(out_ids)


def create_app() -> FastAPI:
    app = FastAPI(title="CEM API", version="0.2.0")
    model, tok, device = _load_model()
    system = ""
    sp = Path(__file__).resolve().parents[1] / "prompts" / "system_default.txt"
    if sp.is_file():
        system = sp.read_text(encoding="utf-8").strip() + "\n\n"

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "CEM", "transformer": False}

    @app.get("/readyz")
    def readyz():
        return {"ready": True, "device": str(device)}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        user_bits = [m.content for m in req.messages if m.role == "user"]
        user = user_bits[-1] if user_bits else ""
        prompt = f"{system}User: {user}\nAssistant: "
        text = _generate(model, tok, prompt, req.max_tokens, req.temperature, device)
        return ChatResponse(
            model=req.model or "CEM",
            output=text,
            choices=[{"message": {"role": "assistant", "content": text}}],
        )

    @app.post("/v1/chat/stream")
    def chat_stream(req: ChatRequest):
        user_bits = [m.content for m in req.messages if m.role == "user"]
        user = user_bits[-1] if user_bits else ""
        prompt = f"{system}User: {user}\nAssistant: "

        def gen():
            # simple non-token stream: one shot then chunk by chars (AERIS-compatible client can read)
            text = _generate(model, tok, prompt, req.max_tokens, req.temperature, device)
            for i in range(0, len(text), 8):
                chunk = text[i : i + 8]
                yield f"data: {json.dumps({'choices':[{'delta':{'content': chunk}}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
