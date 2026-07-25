"""Character-level encoding.

A tiny, dependency-light vocabulary over printable ASCII plus PAD/UNK. HTTP
requests are mostly ASCII; anything else maps to UNK. This is intentionally
simple and fully reproducible — the same text always yields the same ids — and
it serializes to a small JSON file that ships with the trained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAD_ID = 0
UNK_ID = 1


class CharVocab:
    def __init__(self, max_len: int = 1024) -> None:
        # printable ASCII 0x20–0x7E
        self._chars = [chr(c) for c in range(0x20, 0x7F)]
        self.max_len = max_len
        self.stoi = {ch: i + 2 for i, ch in enumerate(self._chars)}

    @property
    def size(self) -> int:
        return len(self._chars) + 2  # + PAD + UNK

    def encode(self, text: str) -> np.ndarray:
        ids = [self.stoi.get(ch, UNK_ID) for ch in text[: self.max_len]]
        if len(ids) < self.max_len:
            ids.extend([PAD_ID] * (self.max_len - len(ids)))
        return np.asarray(ids, dtype=np.int32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.max_len), dtype=np.int32)
        return np.stack([self.encode(t) for t in texts])

    # -- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"max_len": self.max_len, "size": self.size}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(max_len=data["max_len"])
