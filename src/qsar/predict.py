"""Writers for the two prediction strategies' test-set outputs.

Ported from the original submission's ``train.py``. ``predict_ensemble_rf``
is renamed ``predict_multioutput`` (work item 6 of the remediation plan) —
the original name was left over from before LightGBM was added and was
misleading, since this function is used for both RandomForest and LightGBM.
``os.makedirs`` calls are added before writing (the original relied on
``QSAR Challenge/submissions/`` already existing on disk).

Both functions average ``model.predict(X)`` — hard class labels drawn from
``{-1, 0, 1}`` — over an ensemble of models, rather than averaging class
*probabilities*. This is unchanged from the original submission, and is the
single largest issue flagged in the README Limitations: the challenge's own
``sample_submission.csv`` expects a continuous ranking score per task, not
an average of discrete labels.
"""
import os

import numpy as np
import pandas as pd


def predict_multioutput(models, X, output_path):
    """Average ``model.predict(X)`` over ``models`` and write one CSV.

    ``models`` is the list returned by ``qsar.models.cross_val_single_model``
    (each a fitted ``MultiOutputClassifier``). Writes one row per row of
    ``X`` with columns ``task1`` .. ``taskN``. Unchanged from the original
    submission's ``predict_ensemble_rf`` other than the name and the added
    ``os.makedirs``.
    """
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_preds = [model.predict(X) for model in models]
    y_pred = np.mean(all_preds, axis=0)
    df = pd.DataFrame(y_pred, index=X.index, columns=[f"task{i + 1}" for i in range(y_pred.shape[1])])
    df.to_csv(output_path)


def predict_per_target(models, X, output_path):
    """Average ``model.predict(X)`` over each target's fold models and write one CSV.

    ``models`` is the dict returned by ``qsar.models.train_per_target``
    (target name -> list of fitted per-target models). Column order follows
    dict insertion order, i.e. the order ``y.columns`` was iterated in
    during training. Unchanged from the original submission's
    ``predict_per_target`` other than the added ``os.makedirs``.
    """
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    preds = {}
    for i, (_, model_list) in enumerate(models.items()):
        fold_preds = [model.predict(X) for model in model_list]
        y_pred = np.mean(fold_preds, axis=0)
        preds[f"task{i + 1}"] = y_pred
    df = pd.DataFrame(preds, index=X.index)
    df.to_csv(output_path)
