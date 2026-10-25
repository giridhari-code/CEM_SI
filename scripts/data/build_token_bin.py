#!/usr/bin/env python3
"""Text → uint16 token bin (char ids) for AERIS-style --data-dir training."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cem.utils.tokenizer import CharTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="text file")
    ap.add_argument("--output", required=True, help="out .bin path")
    args = ap.parse_args()
    tok = CharTokenizer()
    text = Path(args.input).read_text(encoding="utf-8")
    ids = np.array(tok.encode(text, add_special_tokens=False), dtype=np.uint16)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(out)
    print(f"wrote {out} tokens={len(ids)}")


if __name__ == "__main__":
    main()
