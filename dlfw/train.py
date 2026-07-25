"""Training entrypoint.

    python -m dlfw.train --data data/csic2010     # real CSIC 2010 dataset
    python -m dlfw.train --synthetic              # quick run on generated data

Saves the trained model and the vocabulary side by side so inference can load
both with one call.
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import DEFAULT, Config
from .data import load_csic, synthetic_dataset
from .encoding import CharVocab


def train(cfg: Config, data_dir: str | None, synthetic: bool) -> None:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    from .model import build_model

    texts, labels = (synthetic_dataset(1200) if synthetic or not data_dir
                     else load_csic(data_dir))
    print(f"loaded {len(texts)} requests "
          f"({sum(labels)} anomalous, {len(labels) - sum(labels)} normal)")

    vocab = CharVocab(cfg.max_len)
    X = vocab.encode_batch(texts)
    y = np.asarray(labels, dtype=np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # class weights guard against imbalance
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    class_weight = {0: len(y_tr) / (2 * neg), 1: len(y_tr) / (2 * pos)}

    model = build_model(vocab.size, cfg.max_len, cfg.embed_dim,
                        cfg.kernel_sizes, cfg.filters, cfg.dropout, cfg.learning_rate)
    model.summary()

    model.fit(
        X_tr, y_tr,
        validation_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                             patience=cfg.patience, restore_best_weights=True),
        ],
        verbose=2,
    )

    print("\nheld-out test:")
    metrics = model.evaluate(X_te, y_te, verbose=0, return_dict=True)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(cfg.model_path)
    vocab.save(cfg.vocab_path)
    print(f"\nsaved model -> {cfg.model_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Deep Learning Firewall.")
    ap.add_argument("--data", help="path to the CSIC 2010 dataset directory")
    ap.add_argument("--synthetic", action="store_true", help="use generated data")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()

    cfg = DEFAULT
    if args.epochs:
        cfg = Config(epochs=args.epochs)
    train(cfg, args.data, args.synthetic)


if __name__ == "__main__":
    main()
