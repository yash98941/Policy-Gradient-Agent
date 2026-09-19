"""The methods the agent is measured against.

Every one of these picks exactly k features so the comparison is fair. Most of
them score features one at a time and take the top k, which is fast but blind to
the case where two useless columns become useful together.
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression


def all_features(X, y, k, seed=0):
    return np.arange(X.shape[1])


def random_subset(X, y, k, seed=0):
    return np.sort(np.random.default_rng(seed).choice(X.shape[1], k, replace=False))


def mutual_info(X, y, k, seed=0):
    scores = mutual_info_classif(X, y, random_state=seed)
    return np.sort(np.argsort(scores)[-k:])


def anova(X, y, k, seed=0):
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores)
    return np.sort(np.argsort(scores)[-k:])


def lasso(X, y, k, seed=0):
    """L1 shrinks useless coefficients to zero, so the survivors are the picks."""
    model = LogisticRegression(
        solver="saga", l1_ratio=1.0, C=0.05, max_iter=3000, random_state=seed
    )
    model.fit(X, y)
    return np.sort(np.argsort(np.abs(model.coef_[0]))[-k:])


def tree_importance(X, y, k, seed=0):
    model = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    model.fit(X, y)
    return np.sort(np.argsort(model.feature_importances_)[-k:])


def rfe(X, y, k, seed=0):
    """Drops the weakest feature and refits, over and over.

    Slower than the ranking methods but it does at least see features in
    combination, so it is the strongest baseline here.
    """
    selector = RFE(
        LogisticRegression(max_iter=1000, random_state=seed),
        n_features_to_select=k,
        step=0.1,
    )
    selector.fit(X, y)
    return np.sort(np.where(selector.support_)[0])


METHODS = {
    "all features": all_features,
    "random": random_subset,
    "mutual info": mutual_info,
    "anova f-test": anova,
    "l1 logistic": lasso,
    "tree importance": tree_importance,
    "rfe": rfe,
}
