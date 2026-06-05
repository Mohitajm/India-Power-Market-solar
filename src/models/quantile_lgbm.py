"""
src/models/quantile_lgbm.py — restored to original (your working version)
==========================================================================
This is the original file that produced:
  DAM  WMAPE: 14.40%
  RTM  WMAPE: 13.04%
  D+1  WMAPE: 14.27%

No monotone constraints (incompatible with LightGBM quantile objective).
No log-space clipping (handled by safe feature engineering instead).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path


class QuantileLGBM:
    """
    LightGBM quantile regression model.

    Parameters
    ----------
    alpha : float
        Quantile level (e.g. 0.10 for q10, 0.50 for median).
    params : dict
        LightGBM hyperparameters (from model_config.yaml lgbm_defaults).
    """

    def __init__(self, alpha: float, params: dict):
        self.alpha  = alpha
        self.params = params
        self.model  = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: pd.DataFrame,   y_val: np.ndarray) -> None:
        """
        Train the quantile model with early stopping on the validation set.
        """
        p = dict(self.params)  # copy to avoid mutating shared dict

        p["objective"]      = "quantile"
        p["alpha"]          = self.alpha
        p["metric"]         = "quantile"
        p["verbose"]        = -1

        early_stop = int(p.pop("early_stopping_rounds", 50))
        n_est      = int(p.pop("n_estimators",          500))

        self.model = lgb.LGBMRegressor(n_estimators=n_est, **p)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        raw = self.model.predict(X)
        return np.maximum(0.0, raw)   # prices cannot be negative

    def save(self, path: str) -> None:
        """Save booster to a LightGBM .txt file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(path)

    @classmethod
    def load(cls, path: str, alpha: float) -> "QuantileLGBM":
        """Load a saved booster from a .txt file."""
        instance        = cls(alpha=alpha, params={})
        booster         = lgb.Booster(model_file=path)
        wrapper         = lgb.LGBMRegressor()
        wrapper._Booster = booster
        wrapper._n_features = booster.num_feature()

        class _BoosterShim:
            def __init__(self, b):
                self._b              = b
                self.best_iteration  = b.best_iteration
            def num_trees(self):
                return self._b.num_trees()
            def predict(self, data):
                return self._b.predict(data)

        instance.model        = _BoosterShim(booster)
        instance._raw_booster = booster
        instance._wrapper     = wrapper
        return instance

    def predict(self, X) -> np.ndarray:          # noqa: F811
        if isinstance(self.model, lgb.LGBMRegressor):
            raw = self.model.predict(X)
        else:
            if hasattr(X, "values"):
                raw = self.model.predict(X.values)
            else:
                raw = self.model.predict(X)
        return np.maximum(0.0, raw)
