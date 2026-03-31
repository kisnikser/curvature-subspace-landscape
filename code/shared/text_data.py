"""Character-level text chunks for causal LM corpora."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset


CORPUS_URLS = {
    "tiny_shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    ),
    "wikitext2": (
        "https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2/train.txt"
    ),
}


def ensure_text_corpus(
    dest: Path,
    corpus_name: str = "tiny_shakespeare",
    url_override: str | None = None,
) -> Path:
    """Download a supported UTF-8 corpus if missing; return the local path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url_override or CORPUS_URLS.get(corpus_name)
    if url is None:
        raise ValueError(
            f"Unknown corpus_name={corpus_name!r}. "
            f"Known corpora: {sorted(CORPUS_URLS)}"
        )
    if not dest.is_file():
        urllib.request.urlretrieve(url, dest)
    return dest


def ensure_tiny_shakespeare(dest: Path) -> Path:
    """Backward-compatible wrapper for the default reference corpus."""
    return ensure_text_corpus(dest, corpus_name="tiny_shakespeare")


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
