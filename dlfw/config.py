"""Central configuration — hyperparameters and paths in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # sequence
    max_len: int = 1024               # chars per request fed to the model

    # model
    embed_dim: int = 64
    kernel_sizes: tuple[int, ...] = (3, 5, 7)
    filters: int = 128
    dropout: float = 0.4

    # training
    batch_size: int = 128
    epochs: int = 15
    learning_rate: float = 1e-3
    val_split: float = 0.1
    patience: int = 3                 # early-stopping patience
    decision_threshold: float = 0.5

    # paths
    artifacts_dir: Path = field(default=ROOT / "artifacts")

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "firewall.keras"

    @property
    def vocab_path(self) -> Path:
        return self.artifacts_dir / "vocab.json"


DEFAULT = Config()
