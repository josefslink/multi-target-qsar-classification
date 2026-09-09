#!/usr/bin/env python3
"""CLI entry point: train both modelling strategies for one or both base estimators.

Defaults (``--models both --strategy both``) reproduce the original
submission's ``train.py`` ``__main__`` block exactly: RandomForest then
LightGBM, single multi-output model then per-target models, in that order,
with the same hyperparameter grids and random states. The ``--models`` /
``--strategy`` flags (added at publication time, work item 5 of the
remediation plan) let a reviewer run one path without editing source — they
select a subset of the same loop; they do not change what happens inside it.

Also added at publication time: each set of trained models is pickled to
``--model-dir`` so ``scripts/predict.py`` can be run as a separate step. The
original submission trained and wrote predictions in a single run with
nothing persisted to disk. This is a structural addition for legibility,
not a change to the modelling approach.

``--log-file`` reproduces the original's ``training_auc_log.txt``
(previously created as an import-time side effect of
``logging.basicConfig`` at the top of ``train.py``; that call now happens
here, explicitly, inside ``main()``).
"""
import argparse
import logging
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from qsar.data import load_datasets
from qsar.models import cross_val_single_model, train_per_target
from qsar.predict import predict_multioutput, predict_per_target
from qsar.seeding import SEED, set_seed

# Order matches the original submission's `model_classes = [RandomForestClassifier, LGBMClassifier]`.
MODEL_REGISTRY = {
    "rf": (RandomForestClassifier, "RandomForest"),
    "lgbm": (LGBMClassifier, "LGBM"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="directory with the three *_preprocessed.csv files")
    parser.add_argument("--output-dir", default="submissions", help="where prediction CSVs are written")
    parser.add_argument("--model-dir", default="models", help="where trained models are pickled")
    parser.add_argument("--log-file", default="training_auc_log.txt")
    parser.add_argument("--models", choices=["rf", "lgbm", "both"], default="both")
    parser.add_argument("--strategy", choices=["per-target", "multi-output", "both"], default="both")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(message)s")

    X_train, y_train, X_valid, y_valid, X_test = load_datasets(args.data_dir)

    if args.models == "both":
        selected = [MODEL_REGISTRY["rf"], MODEL_REGISTRY["lgbm"]]
    else:
        selected = [MODEL_REGISTRY[args.models]]

    strategies = ["multi-output", "per-target"] if args.strategy == "both" else [args.strategy]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    for model_class, model_name in selected:
        if "multi-output" in strategies:
            print(f"\n--- Training Single Ensemble Model ({model_name}) ---")
            single_models = cross_val_single_model(X_train, y_train, X_valid, y_valid, model_class, model_name)
            with open(os.path.join(args.model_dir, f"single_{model_name}.pkl"), "wb") as f:
                pickle.dump(single_models, f)
            predict_multioutput(
                single_models, X_test, os.path.join(args.output_dir, f"predictions_single_{model_name}_test.csv")
            )

        if "per-target" in strategies:
            print(f"\n--- Training Per-Target Models ({model_name}) ---")
            per_target_models, per_target_scores = train_per_target(
                X_train, y_train, X_valid, y_valid, model_class, model_name
            )
            with open(os.path.join(args.model_dir, f"per_target_{model_name}.pkl"), "wb") as f:
                pickle.dump(per_target_models, f)
            predict_per_target(
                per_target_models, X_test, os.path.join(args.output_dir, f"predictions_per_target_{model_name}_test.csv")
            )


if __name__ == "__main__":
    main()
