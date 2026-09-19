"""The three datasets the agent is tested on.

Synthetic comes first because it is the only one where the correct answer is
known. If the agent cannot recover planted features there, nothing it does on
real data can be trusted.
"""

import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CACHE = "data"


def make_synthetic(n_features=40, n_pairs=2, n_samples=6000, noise=0.05, seed=0):
    """Builds a dataset where the signal only exists in pairs of columns.

    The label is the sign of x_a1*x_b1 + x_a2*x_b2 for two hidden pairs. Neither
    half of a pair tells you anything on its own: both columns are symmetric
    around zero whatever the label is, so the correlation between any single
    column and y is zero by construction.

    That is the whole point. Ranking columns one at a time cannot work here, and
    a model handed all 40 columns does little better than a coin flip because
    the 36 noise columns drown the interaction. Returns the four indices that
    matter so recovery can be scored directly.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    order = rng.permutation(n_features)
    truth = order[:2 * n_pairs]

    signal = np.zeros(n_samples)
    for i in range(n_pairs):
        signal += X[:, truth[2 * i]] * X[:, truth[2 * i + 1]]

    y = (signal > 0).astype(int)
    flipped = rng.random(n_samples) < noise
    y[flipped] = 1 - y[flipped]

    return X, y, np.sort(truth)


def load_madelon():
    """NIPS 2003 feature selection benchmark. 500 columns, 20 of them real.

    The relationship is XOR shaped, so anything that ranks features one at a
    time by correlation does badly here. That is the whole point of the dataset.
    """
    raw = fetch_openml(data_id=1485, as_frame=True, data_home=CACHE)
    X = raw.data.to_numpy(dtype=float)
    y = (raw.target.to_numpy().astype(int) == 2).astype(int)
    return X, y, None


def load_credit():
    """Same UCI credit default data used in the credit-risk project."""
    raw = fetch_openml(data_id=42477, as_frame=True, data_home=CACHE)
    X = raw.data.to_numpy(dtype=float)
    y = raw.target.to_numpy().astype(int)
    return X, y, None


def get(name, seed=0):
    if name == "synthetic":
        return make_synthetic(seed=seed)
    if name == "madelon":
        return load_madelon()
    if name == "credit":
        return load_credit()
    raise ValueError("unknown dataset: " + name)


def split(X, y, seed=0, search_rows=1200, val_rows=800):
    """Cuts the data three ways.

    The agent only ever sees search_train and search_val. The test split is kept
    aside so the final subset can be scored on rows that took no part in
    choosing it. Without this the reported gain is just overfitting to the
    reward signal.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest, y_rest, test_size=0.3, stratify=y_rest, random_state=seed
    )

    # Big datasets make every reward call slow, so the search runs on a sample.
    rng = np.random.default_rng(seed)
    if len(y_tr) > search_rows:
        keep = rng.choice(len(y_tr), search_rows, replace=False)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    if len(y_val) > val_rows:
        keep = rng.choice(len(y_val), val_rows, replace=False)
        X_val, y_val = X_val[keep], y_val[keep]

    scaler = StandardScaler().fit(X_tr)
    return {
        "train": (scaler.transform(X_tr), y_tr),
        "val": (scaler.transform(X_val), y_val),
        "test": (scaler.transform(X_test), y_test),
    }
