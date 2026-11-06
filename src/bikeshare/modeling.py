"""Model helpers: negative-binomial GLM, LightGBM wrappers and TreeSHAP contributions."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor

NB_FORMULA = ("cnt ~ C(hr):C(workingday) + C(weathersit) + temp_c + I(temp_c**2) + hum_pct + wind_kmh + wind_calm"
              " + holiday + C(season) + np.log1p(roll_28d_mean) + np.log1p(lag_168)")


def estimate_nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Cameron-Trivedi moment estimate of the NB2 dispersion from a Poisson fit."""
    aux = ((y - mu) ** 2 - y) / mu
    return float(max((mu * aux).sum() / (mu * mu).sum(), 1e-3))


def fit_nb_glm(X: pd.DataFrame, y: pd.Series, formula: str = NB_FORMULA):
    """Fit a negative-binomial GLM (log link). Returns (fitted result, alpha)."""
    data = X.assign(cnt=y).dropna()
    pois = smf.glm(formula, data=data, family=sm.families.Poisson()).fit()
    alpha = estimate_nb_alpha(data["cnt"].to_numpy(), pois.fittedvalues.to_numpy())
    res = smf.glm(formula, data=data, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    return res, alpha


def predict_glm(res, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(res.predict(X.assign(cnt=0.0)))


def fit_lgbm(X: pd.DataFrame, y: pd.Series, objective: str = "poisson", alpha: float | None = None, seed: int = 42, **params) -> LGBMRegressor:
    base = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=1)
    base.update(params)
    if objective == "quantile":
        base.update(objective="quantile", alpha=alpha)
    else:
        base.update(objective=objective)
    return LGBMRegressor(**base).fit(X, y)


def tree_shap(model: LGBMRegressor, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP contributions from LightGBM (log scale for a Poisson objective) and the base value."""
    contrib = model.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns), float(contrib[0, -1])
