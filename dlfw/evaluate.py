"""Evaluate a trained model — precision / recall / F1 + confusion matrix.

    python -m dlfw.evaluate --data data/csic2010
    python -m dlfw.evaluate --synthetic
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import DEFAULT
from .data import load_csic, synthetic_dataset
from .detector import FirewallDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix

    det = FirewallDetector.load(DEFAULT)
    texts, labels = (synthetic_dataset(600) if args.synthetic or not args.data
                     else load_csic(args.data))
    y = np.asarray(labels)
    scores = np.asarray(det.score_batch(texts))
    pred = (scores >= det.threshold).astype(int)

    print(classification_report(y, pred, target_names=["normal", "anomalous"], digits=4))
    print("confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y, pred))


if __name__ == "__main__":
    main()
