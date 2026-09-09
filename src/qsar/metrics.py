"""AUC scorers used during hyperparameter search and cross-validation.

Ported unchanged from the original submission's ``train.py``. Both scorers
compute one-vs-rest macro ROC AUC over the three label values
``{-1, 0, 1}`` (active / inactive / not measured — see the README for why
unmeasured entries are not masked out).
"""
import logging

import numpy as np
from sklearn.metrics import roc_auc_score


def single_target_auc_scorer(estimator, X, y_true):
    """Scorer for a single-target classifier (used by ``train_per_target``).

    Returns ``np.nan`` (with a logged warning) if the validation fold has
    only one class present, or if ``predict_proba``/``roc_auc_score`` raise.
    """
    try:
        y_proba = estimator.predict_proba(X)

        if len(np.unique(y_true)) > 1:
            return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=[-1, 0, 1])
        else:
            logging.warning("Skipping auc calc due to constant labels in validation fold")
        return np.nan
    except Exception as e:
        logging.warning(f"Skipping auc calc due to error {e}")
        return np.nan


def multioutput_auc_scorer(estimator, X, y_true):
    """Scorer for a ``MultiOutputClassifier`` (used by ``cross_val_single_model``).

    Computes ``single_target_auc_scorer``-equivalent AUC per target and
    returns the mean over targets that had more than one class present in
    the fold. Returns ``np.nan`` if no target could be scored.
    """
    try:
        probas_list = estimator.predict_proba(X)
        aucs = []

        for i in range(len(probas_list)):
            try:
                if len(np.unique(y_true[:, i])) < 2:
                    logging.warning(f"auc skip Target {i}: only one class in y_true")
                    continue

                prob = probas_list[i]
                if isinstance(prob, list):
                    prob = np.asarray(prob)

                if prob.ndim == 2 and prob.shape[1] >= 2:
                    auc = roc_auc_score(y_true[:, i], prob, multi_class="ovr", average="macro", labels=[-1, 0, 1])
                    aucs.append(auc)
                else:
                    logging.warning(f"auc Target {i}: unexpected prob shape {prob.shape}")
            except Exception as e:
                logging.warning(f"auc error Target {i}: {e}")
                continue
        return np.mean(aucs) if aucs else np.nan
    except Exception as e:
        logging.warning(f"Skipping auc calc due to error {e}")
        return np.nan
