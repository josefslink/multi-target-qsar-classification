"""Loading, train/valid splitting, and descriptor scaling.

The first two functions are ported from the original submission's
``data_prep.py`` (its non-featurisation half). ``load_datasets`` is ported
from the module-level loads at the top of the original ``train.py``
(``train.py:10-20``), which read three CSVs as a side effect of importing
the module — that crashed ``import`` with no data present. Moving it into a
function is the only structural change; the read/slice logic is unchanged.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .features import get_descriptors


def preprocess_and_split_data(source_file: str, destination_path: str, test_size: float = 0.2):
    """Featurise ``source_file``, split into train/valid, scale, and write both to disk.

    Reads a CSV with an index column, a ``smiles`` column, and 11 label
    columns (see ``data/README.md`` for the schema). Descriptor columns
    (everything not starting with ``FP_``) are standardised with a
    ``StandardScaler`` fit on the training split only; fingerprint columns
    are left unstandardised. ``test_size`` and the split's
    ``random_state=42`` are unchanged from the original submission.

    Returns ``(train_df, valid_df, scaler)`` — the fitted scaler is reused
    by ``preprocess_submission_test_data`` for the held-out test set.
    """
    qsar_df = pd.read_csv(source_file, header=[0], index_col=[0])

    descriptors_df = qsar_df["smiles"].apply(get_descriptors)
    qsar_df = qsar_df.join(descriptors_df)

    qsar_df.drop(columns=["smiles"], inplace=True)

    qsar_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    qsar_df.fillna(0, inplace=True)

    y = qsar_df.iloc[:, :11]
    X = qsar_df.iloc[:, 11:]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42)

    desc_columns = [col for col in X.columns if not col.startswith("FP_")]
    scaler = StandardScaler()
    X_train.loc[:, desc_columns] = scaler.fit_transform(X_train[desc_columns])
    X_valid.loc[:, desc_columns] = scaler.transform(X_valid[desc_columns])

    train_df = pd.concat([y_train, X_train], axis=1)
    valid_df = pd.concat([y_valid, X_valid], axis=1)

    os.makedirs(destination_path, exist_ok=True)
    train_df.to_csv(os.path.join(destination_path, "train_data_preprocessed.csv"))
    valid_df.to_csv(os.path.join(destination_path, "valid_data_preprocessed.csv"))

    return train_df, valid_df, scaler


def preprocess_submission_test_data(test_file: str, destination_path: str, scaler):
    """Featurise the test SMILES and scale with an already-fitted ``scaler``.

    ``scaler`` must be the one returned by ``preprocess_and_split_data`` for
    the corresponding training data — it is applied with ``.transform``
    only, never refit, unchanged from the original submission.
    """
    qsar_df = pd.read_csv(test_file, header=[0], index_col=[0])

    descriptors_df = qsar_df["smiles"].apply(get_descriptors)
    qsar_df = qsar_df.join(descriptors_df)
    qsar_df = qsar_df.drop(columns=["smiles"])

    qsar_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    qsar_df.fillna(0, inplace=True)

    desc_columns = [col for col in qsar_df.columns if not col.startswith("FP_")]
    qsar_df[desc_columns] = scaler.transform(qsar_df[desc_columns])

    qsar_df.to_csv(os.path.join(destination_path, "test_data_preprocessed.csv"))
    return qsar_df


def load_datasets(data_dir: str):
    """Load the three preprocessed CSVs and split each into features/labels.

    Moved out of module scope (originally ``train.py:10-20``) into this
    function so that ``import qsar`` succeeds with no data present. The
    read/slice logic is unchanged: train/valid frames are split into the
    first 11 columns (``y``) and the remaining columns (``X``); the test
    frame has no label columns and is used whole as ``X_test``.

    Returns ``(X_train, y_train, X_valid, y_valid, X_test)``.
    """
    train_df = pd.read_csv(os.path.join(data_dir, "train_data_preprocessed.csv"), index_col=0)
    valid_df = pd.read_csv(os.path.join(data_dir, "valid_data_preprocessed.csv"), index_col=0)
    test_df = pd.read_csv(os.path.join(data_dir, "test_data_preprocessed.csv"), index_col=0)

    X_train = train_df.iloc[:, 11:]
    y_train = train_df.iloc[:, :11]
    X_valid = valid_df.iloc[:, 11:]
    y_valid = valid_df.iloc[:, :11]
    X_test = test_df

    return X_train, y_train, X_valid, y_valid, X_test
