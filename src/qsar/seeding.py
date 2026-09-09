"""Seed helper for the numpy/sklearn/LightGBM stack.

Added at publication time (Stage 3 remediation). The original submission
set `random_state=42` on every individual estimator, `KFold` split, and
`RandomizedSearchCV` call (unchanged — see `qsar.models`), but never called
`random.seed` or `numpy.random.seed` directly. `set_seed` collects that one
constant in a single place for anything (e.g. this package's own scripts)
that draws from those global generators. The value, 42, is unchanged from
the original.
"""
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed the `random` and `numpy.random` global generators.

    Does not touch the per-estimator ``random_state=42`` arguments already
    hardcoded throughout ``qsar.data`` and ``qsar.models`` — those are
    unchanged from the original submission and are not routed through this
    function.
    """
    random.seed(seed)
    np.random.seed(seed)
