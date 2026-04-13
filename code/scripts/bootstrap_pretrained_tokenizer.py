"""
Materialize ~/.cache/nanochat/tokenizer/ from a tiktoken pretrained encoding (no BPE training).

Usage (from repo root / code):
  python -m scripts.bootstrap_pretrained_tokenizer
  python -m scripts.bootstrap_pretrained_tokenizer --encoding gpt2

This writes tokenizer.pkl + token_bytes.pt so get_tokenizer() / get_token_bytes() work like after tok_train.
Vocab size follows the encoding (e.g. gpt2 -> 50257), which is fine for smoke runs; full nanochat
runs typically use a RustBPE-trained 32k vocab from scripts.tok_train on ClimbMix.
"""
from __future__ import annotations

import argparse
import os

import torch

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoding",
        type=str,
        default="gpt2",
        help="tiktoken encoding name, e.g. gpt2, cl100k_base (see tiktoken.get_encoding)",
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    tok = RustBPETokenizer.from_pretrained(args.encoding)
    tok.save(tokenizer_dir)

    vocab_size = tok.get_vocab_size()
    special_set = set(tok.get_special_tokens())
    token_strings = [tok.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes: list[int] = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id]
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    tb = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(path, "wb") as f:
        torch.save(tb, f)
    print(f"Saved token_bytes to {path}")
    print(f"Done. Vocab size: {vocab_size:,} (encoding={args.encoding})")


if __name__ == "__main__":
    main()
