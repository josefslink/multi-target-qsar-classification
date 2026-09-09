"""Multi-target QSAR / bioactivity classification pipeline.

Two strategies for predicting eleven anonymised bioactivity targets from a
molecule's 2D structure: independent per-target LightGBM/RandomForest
classifiers vs. one multi-output classifier. See the top-level README for
the problem statement, results, and limitations.

Importing this package has no side effects (in particular, it does not read
any CSV files) — the original submission's `train.py` did read three CSVs
at module import time; that has been moved into `qsar.data.load_datasets`,
called explicitly by `scripts/train.py`.
"""
