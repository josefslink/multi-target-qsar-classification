#!/usr/bin/env python3
"""CLI entry point: featurise + split training data, then featurise test data.

argparse wrapper around ``qsar.data.preprocess_and_split_data`` and
``qsar.data.preprocess_submission_test_data``. Both calls run in one
process, exactly as the original submission's ``data_prep.py`` ``__main__``
block did: the ``StandardScaler`` fit on the training split is passed to
the test-featurisation step in memory and is never serialized to disk,
matching the original.

Replaces the original's hardcoded Windows paths
(``r"QSAR Challenge\\Data\\data_train.csv"`` etc.) with arguments defaulting
to ``data/``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from qsar.data import preprocess_and_split_data, preprocess_submission_test_data
from qsar.seeding import SEED, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", default="data/data_train.csv")
    parser.add_argument("--test-file", default="data/smiles_test.csv")
    parser.add_argument("--out-dir", default="data", help="where the three *_preprocessed.csv files are written")
    parser.add_argument("--test-size", type=float, default=0.2, help="unchanged default from the original submission")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    _, _, scaler = preprocess_and_split_data(args.train_file, args.out_dir, test_size=args.test_size)
    preprocess_submission_test_data(args.test_file, args.out_dir, scaler)

    print(f"Wrote train_data_preprocessed.csv, valid_data_preprocessed.csv, test_data_preprocessed.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
