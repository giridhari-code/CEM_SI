"""
CEM CLI — AERIS/sentinel.cli shaped.

  python -m cem.cli train --config configs/default.yaml --max-steps 100
  python -m cem.cli chat --checkpoint checkpoints/tiny/checkpoint_latest.pt --prompt Hi
  python -m cem.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(argv: list[str]) -> None:
    import runpy
    # execute scripts/train.py as __main__ with argv
    sys.argv = ["train"] + argv
    runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")


def cmd_chat(argv: list[str]) -> None:
    sys.argv = ["chat"] + argv
    from scripts.chat import main as chat_main

    chat_main()


def cmd_serve(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args(argv)
    if args.checkpoint:
        os.environ["CEM_CHECKPOINT"] = args.checkpoint
    import uvicorn

    uvicorn.run("cem.api.app:create_app", factory=True, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    setup_logging()
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("commands: train | chat | serve")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        cmd_train(rest)
    elif cmd == "chat":
        cmd_chat(rest)
    elif cmd == "serve":
        cmd_serve(rest)
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
