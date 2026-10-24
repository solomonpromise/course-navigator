"""Data contracts shared by training, evaluation and the teaching notebooks.

Vocabulary fitting is deliberately separate from encoding: validation and test
text must never change which words the scratch models are allowed to learn.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Foundations & PyTorch", "Training & evaluation", "Representations & sequences",
          "Transformers & attention", "Pretrained applications", "Model adaptation"]
MAX_LENGTH = 96


def words(text: str) -> list[str]:
    """Lowercase text and retain words/numbers; this is our scratch tokenizer.

    This simple rule is intentionally different from BERT's WordPiece tokenizer.
    Never substitute one model's tokenizer for another model's tokenizer.
    """
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def load_records(split: str | None = None) -> list[dict]:
    """Read the frozen, attributed dataset, optionally selecting one split."""
    rows = [json.loads(line) for line in (ROOT / "data/passages.jsonl").read_text().splitlines()]
    return rows if split is None else [row for row in rows if row["split"] == split]


def fit_vocabulary(rows: list[dict], limit: int = 6000) -> dict[str, int]:
    """Fit word IDs on training rows only, reserving 0 for PAD and 1 for UNK.

    PAD means 'no real token here'; UNK means 'a real word absent from training'.
    A deterministic frequency/name ordering makes the artifact reproducible.
    """
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Vocabulary fitting accepts training rows only")
    counts = Counter(word for row in rows for word in words(row["text"]))
    ordered = sorted(counts, key=lambda word: (-counts[word], word))[:limit - 2]
    return {"[PAD]": 0, "[UNK]": 1, **{word: i + 2 for i, word in enumerate(ordered)}}


def encode(text: str, vocabulary: dict, max_length: int = MAX_LENGTH) -> tuple[list[int], int]:
    """Return padded IDs and the real length, truncating overly long passages.

    Empty input becomes one UNK token so packed recurrence and masked means
    remain mathematically defined. The public app rejects blank input earlier.
    """
    ids = [vocabulary.get(word, 1) for word in words(text)][:max_length] or [1]
    length = len(ids)
    return ids + [0] * (max_length - length), length


class TextDataset(Dataset):
    """Expose fixed-length IDs, true lengths and integer labels to DataLoader."""
    def __init__(self, rows: list[dict], vocabulary: dict):
        encoded = [encode(row["text"], vocabulary) for row in rows]
        self.ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        self.labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ids[index], self.lengths[index], self.labels[index]
