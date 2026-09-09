"""Hyperparameter grids and the two training strategies.

Ported unchanged from the original submission's ``train.py``:
``get_hyperparameter_grid``, ``cross_val_single_model`` (the "single
multi-output model" strategy), ``train_per_target`` (the "one model per
target" strategy). Docstrings added; no modelling logic, hyperparameter
grid, or default changed. See the README Limitations section for known
issues in this code (search fit on the full training set before CV,
``class_weight='balanced'`` not applied inside the search, etc.) — none of
them are fixed here, by design.

Logging: calls below use the root logger (``logging.info`` /
``logging.warning``) exactly as the original did. The original configured
this at import time via ``logging.basicConfig(filename="training_auc_log.txt", ...)``
at the top of ``train.py``; that side effect now lives in
``scripts/train.py``'s ``main()`` instead, so importing this module has no
effect on logging configuration.
"""
import logging

import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier

from .metrics import multioutput_auc_scorer, single_target_auc_scorer


def get_hyperparameter_grid(model_name: str, is_multi_output: bool) -> dict:
    """Return the ``RandomizedSearchCV`` parameter distribution for ``model_name``.

    ``model_name`` starting with ``"lgbm"`` (case-insensitive) selects the
    LightGBM grid; anything else selects the RandomForest grid. When
    ``is_multi_output`` is true, keys are prefixed with ``estimator__`` for
    ``MultiOutputClassifier``. Grid values are unchanged from the original
    submission, including the LightGBM ``bagging_fraction`` entries, which
    have no effect without ``bagging_freq > 0`` (see README Limitations).
    """
    if model_name.lower().startswith("lgbm"):
        return {
            ("estimator__" if is_multi_output else "") + k: v
            for k, v in {
                "n_estimators": [100, 200, 300, 500, 750, 1000],
                "max_depth": [5, 10, 15, 20, 25, -1],
                "learning_rate": [0.01, 0.05, 0.075, 0.1],
                "num_leaves": [30, 50, 75, 100, 125],
                "feature_fraction": [0.4, 0.6, 0.8, 1.0],
                "bagging_fraction": [0.6, 0.8, 1.0],
                "min_child_samples": [10, 20, 30],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0],
            }.items()
        }
    else:
        return {
            ("estimator__" if is_multi_output else "") + k: v
            for k, v in {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
            }.items()
        }


def cross_val_single_model(X, y, X_valid, y_valid, model_class, model_name="Model", n_splits=5):
    """Train one ``MultiOutputClassifier`` wrapping ``model_class`` for all 11 targets.

    Hyperparameters are searched once via ``RandomizedSearchCV`` (12 draws,
    3-fold) on the full training set, then re-fit across a 5-fold
    ``KFold`` plus one final full-data fit. Returns the list of 6 fitted
    models (5 fold models + 1 final model); predictions are the mean of
    ``model.predict`` over this list (see ``qsar.predict``).

    Unchanged from the original submission's ``cross_val_single_model``,
    including the biased-CV issue noted in the README (the search sees the
    same data the CV folds are drawn from) and ``class_weight='balanced'``
    not being applied inside the search itself.
    """
    base_model = MultiOutputClassifier(model_class(random_state=42))
    hyper_params = get_hyperparameter_grid(model_name, is_multi_output=True)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=hyper_params,
        n_iter=12,
        cv=3,
        scoring=multioutput_auc_scorer,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)
    best_params = search.best_params_

    models = []
    aucs = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = MultiOutputClassifier(
            model_class(
                **{k.replace("estimator__", ""): v for k, v in best_params.items()},
                class_weight="balanced",
                random_state=42,
            )
        )

        model.fit(X_tr, y_tr)
        auc = multioutput_auc_scorer(model, X_val.values, y_val.values)
        aucs.append(auc)
        models.append(model)

        logging.info(f"single model: {model_name} | Fold AUC: {auc:.4f}")

    final_model = MultiOutputClassifier(
        model_class(
            **{k.replace("estimator__", ""): v for k, v in best_params.items()},
            class_weight="balanced",
            random_state=42,
        )
    )
    final_model.fit(X, y)
    final_valid_auc = multioutput_auc_scorer(final_model, X_valid.values, y_valid.values)
    models.append(final_model)
    aucs.append(final_valid_auc)

    logging.info(f"single model: {model_name} | Final AUC: {final_valid_auc:.4f}")

    avg_auc = np.mean(aucs)
    logging.info(f"single model: {model_name} | Avg AUC: {avg_auc:.4f}")

    return models


def train_per_target(X, y, X_valid, y_valid, model_class, model_name="Model", n_splits=5):
    """Train one independent ``model_class`` classifier per target column of ``y``.

    Same search/CV/final-fit structure as ``cross_val_single_model``, but
    repeated independently for each of the 11 columns in ``y`` (each gets
    its own ``RandomizedSearchCV`` and its own 5 fold models + 1 final
    model). Returns ``(models, scores)``: ``models`` maps target name to its
    list of 6 fitted models; ``scores`` maps target name to its mean AUC
    across those 6.

    Unchanged from the original submission's ``train_per_target``.
    """
    models = {}
    scores = {}
    hyper_params = get_hyperparameter_grid(model_name, is_multi_output=False)

    for target in y.columns:
        base_model = model_class(random_state=42)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=hyper_params,
            n_iter=12,
            cv=3,
            scoring=single_target_auc_scorer,
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )
        search.fit(X, y[target])
        best_params = search.best_params_

        fold_models = []
        aucs = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]

            model = model_class(**best_params, class_weight="balanced", random_state=42)
            model.fit(X_tr, y_tr)
            auc = single_target_auc_scorer(model, X_val, y_val.values)
            aucs.append(auc)
            fold_models.append(model)
            logging.info(f"Per Target ({target}): {model_name} | Fold AUC: {auc:.4f}")

        final_model = model_class(**best_params, class_weight="balanced", random_state=42)
        final_model.fit(X, y[target])
        final_valid_auc = single_target_auc_scorer(final_model, X_valid, y_valid[target].values)
        fold_models.append(final_model)
        aucs.append(final_valid_auc)

        avg_auc = np.mean(aucs)
        scores[target] = avg_auc
        models[target] = fold_models
        logging.info(f"Per Target ({target}): {model_name} | Avg AUC: {avg_auc:.4f}")

    overall_avg_auc = np.mean(list(scores.values()))
    logging.info(f"Per Target: {model_name} | Overall Avg AUC across all targets: {overall_avg_auc:.4f}")
    return models, scores
