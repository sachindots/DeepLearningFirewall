"""Inference wrapper — load once, score many.

    det = FirewallDetector.load()
    det.is_malicious(request_to_text("GET", "/p?id=1' OR '1'='1"))   # -> True
"""

from __future__ import annotations

from pathlib import Path

from .config import DEFAULT, Config
from .encoding import CharVocab


class FirewallDetector:
    def __init__(self, model, vocab: CharVocab, threshold: float = 0.5) -> None:
        self._model = model
        self.vocab = vocab
        self.threshold = threshold

    @classmethod
    def load(cls, cfg: Config = DEFAULT, threshold: float | None = None) -> "FirewallDetector":
        import tensorflow as tf

        if not Path(cfg.model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {cfg.model_path}. Run `python -m dlfw.train` first."
            )
        model = tf.keras.models.load_model(cfg.model_path)
        vocab = CharVocab.load(cfg.vocab_path)
        return cls(model, vocab, threshold if threshold is not None else cfg.decision_threshold)

    def score(self, text: str) -> float:
        """Return P(malicious) in [0, 1] for one canonical request string."""
        x = self.vocab.encode(text)[None, :]
        return float(self._model.predict(x, verbose=0)[0, 0])

    def score_batch(self, texts: list[str]) -> list[float]:
        x = self.vocab.encode_batch(texts)
        return [float(p) for p in self._model.predict(x, verbose=0)[:, 0]]

    def is_malicious(self, text: str) -> bool:
        return self.score(text) >= self.threshold
