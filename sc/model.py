"""Train the Tc surrogate and measure it honestly.

Two cross-validations are reported:
  * random 5-fold: how well the model interpolates among compounds like ones it has seen;
  * grouped 5-fold by chemical system (all compounds made of the same set of elements are
    held out together): closer to the real task of predicting a composition that is new.
The five grouped-fold models are kept as an ensemble; their spread is the uncertainty shown
for each candidate.
"""
import numpy as np
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

from .features import featurize

PARAMS = dict(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.6, min_child_weight=2, n_jobs=-1, random_state=0)


def chemsys(formula):
    return "-".join(sorted(str(e) for e in Composition(formula).elements))


def build_matrix(df, elements):
    X = np.vstack([featurize(f, elements) for f in df["reduced"]])
    return X, df["Tc"].to_numpy(dtype=float)


def _cv(X, y, splitter, groups=None):
    pred = np.zeros_like(y)
    models = []
    for tr, te in splitter.split(X, y, groups):
        m = XGBRegressor(**PARAMS)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        models.append(m)
    return pred, models


def train(df, elements):
    X, y = build_matrix(df, elements)
    groups = df["reduced"].map(chemsys).to_numpy()
    p_rand, _ = _cv(X, y, KFold(5, shuffle=True, random_state=0))
    p_grp, models = _cv(X, y, GroupKFold(5), groups)
    hi = y >= 30
    metrics = {
        "n_compounds": int(len(y)),
        "random_cv": {"r2": round(float(r2_score(y, p_rand)), 3),
                      "mae_K": round(float(mean_absolute_error(y, p_rand)), 2)},
        "grouped_cv": {"r2": round(float(r2_score(y, p_grp)), 3),
                       "mae_K": round(float(mean_absolute_error(y, p_grp)), 2),
                       "mae_K_on_Tc_30K_plus": round(float(mean_absolute_error(y[hi], p_grp[hi])), 2)},
    }
    return models, metrics


def predict(models, X):
    P = np.vstack([m.predict(X) for m in models])
    return P.mean(axis=0), P.std(axis=0)
