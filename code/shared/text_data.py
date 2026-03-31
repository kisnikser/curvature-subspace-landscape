"""Character-level text chunks for causal LM (Tiny Shakespeare by default)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def ensure_tiny_shakespeare(dest: Path) -> Path:
    """Download Karpathy Tiny Shakespeare if missing; return path to UTF-8 text."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dest)
    return dest


def build_char_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi, chars


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


class TextChunkDataset(Dataset):
    """
    Non-overlapping (or strided) chunks of length ``block_size``;
    targets are next-token prediction (shift by one within chunk).
    """

    def __init__(self, data: torch.Tensor, block_size: int, stride: int | None = None):
        self.data = data
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        max_start = len(data) - block_size - 1
        if max_start < 0:
            raise ValueError("Corpus too short for block_size")
        self.starts = list(range(0, max_start + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]
        return x, y


def load_text_corpus(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")
