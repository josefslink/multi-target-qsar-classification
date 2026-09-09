#!/usr/bin/env python3
"""CLI entry point: write test-set predictions from already-trained models.

Added at publication time to give a standalone "predict" step matching this
repo's ``src/qsar/predict.py`` module. The original submission trained and
predicted in a single run (see ``scripts/train.py``'s docstring); this
script loads the pickles ``scripts/train.py`` writes to ``--model-dir`` and
reruns only ``qsar.predict.predict_multioutput`` / ``predict_per_target`` —
no retraining, and no change to the prediction logic itself (still the mean
of hard ``model.predict`` outputs — see README Limitations).
"""
import argparse
import glob
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qsar.data import load_datasets
from qsar.predict import predict_multioutput, predict_per_target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="submissions")
    args = parser.parse_args()

    _, _, _, _, X_test = load_datasets(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(args.model_dir, "single_*.pkl"))):
        model_name = re.match(r"single_(.+)\.pkl$", os.path.basename(path)).group(1)
        with open(path, "rb") as f:
            models = pickle.load(f)
        out = os.path.join(args.output_dir, f"predictions_single_{model_name}_test.csv")
        predict_multioutput(models, X_test, out)
        print(f"Wrote {out}")

    for path in sorted(glob.glob(os.path.join(args.model_dir, "per_target_*.pkl"))):
        model_name = re.match(r"per_target_(.+)\.pkl$", os.path.basename(path)).group(1)
        with open(path, "rb") as f:
            models = pickle.load(f)
        out = os.path.join(args.output_dir, f"predictions_per_target_{model_name}_test.csv")
        predict_per_target(models, X_test, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
