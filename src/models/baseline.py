"""
src/models/baseline.py
NaiveBaseline model used as a benchmark in train_models.py.

Predicts the next value as equal to a lagged feature column
(e.g. mcp_same_hour_yesterday for DAM, mcp_lag_1h for RTM).
"""

import numpy as np
import pandas as pd


class NaiveBaseline:
    """
    Naive baseline: predict target = value of a chosen lag feature.

    Parameters
    ----------
    feature_col : str
        Column name in the feature DataFrame to use as the prediction.
        e.g. 'mcp_same_hour_yesterday'  for DAM
             'mcp_lag_1h'               for RTM
    """

    def __init__(self, feature_col: str):
        self.feature_col = feature_col

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_col not in X.columns:
            # Fall back to zeros if the column is missing
            return np.zeros(len(X))
        return X[self.feature_col].fillna(0).values

    def evaluate(self, df: pd.DataFrame, y_true: np.ndarray,
                 quantiles=None) -> dict:
        """
        Compute metrics using the baseline prediction.

        Returns a dict with wmape, mae, rmse and optional pinball losses.
        """
        y_pred = self.predict(df)

        ae    = np.abs(y_true - y_pred)
        mae   = float(np.mean(ae))
        rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        denom = np.sum(y_true)
        wmape = float(np.sum(ae) / denom * 100) if denom != 0 else float("nan")

        metrics = {"wmape": wmape, "mae": mae, "rmse": rmse}

        # Pinball loss for each quantile (treating the point pred as every quantile)
        if quantiles:
            prob = {}
            for alpha in quantiles:
                diff    = y_true - y_pred
                pinball = float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))
                prob[f"pinball_q{int(alpha * 100)}"] = pinball
            metrics["probabilistic"] = prob

        return metrics
