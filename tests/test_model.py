"""Model tests — skipped automatically when TensorFlow isn't installed
(e.g. on Python versions without a TF wheel)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from dlfw.encoding import CharVocab  # noqa: E402
from dlfw.model import build_model    # noqa: E402


def test_model_builds_and_output_shape():
    v = CharVocab(max_len=64)
    m = build_model(v.size, 64, embed_dim=16, filters=16)
    x = np.zeros((4, 64), dtype="int32")
    y = m.predict(x, verbose=0)
    assert y.shape == (4, 1)
    assert ((0.0 <= y) & (y <= 1.0)).all()   # sigmoid output


def test_model_overfits_tiny_batch():
    # a couple of gradient steps should drive loss down on a trivial set
    v = CharVocab(max_len=64)
    m = build_model(v.size, 64, embed_dim=16, filters=16)
    X = v.encode_batch(["get /", "get /p?id=1' or '1'='1"])
    y = np.array([0.0, 1.0])
    h = m.fit(X, y, epochs=8, verbose=0)
    assert h.history["loss"][-1] < h.history["loss"][0]
