"""
src/models/tuner.py
Optuna-based hyperparameter tuner for LightGBM q50 model.
Called by train_models.py when tuning.enabled = true in model_config.yaml.

For POC (tuning.enabled = false in model_config.yaml) this file is
imported but tune_q50 is never called, so no Optuna installation needed.
"""

import numpy as np
import pandas as pd


def tune_q50(X_train: pd.DataFrame, y_train: np.ndarray,
             X_val:   pd.DataFrame, y_val:   np.ndarray,
             n_trials: int = 20,
             config:   dict = None) -> dict:
    """
    Run Optuna hyperparameter search for the q50 (median) quantile model.
    Returns the best params dict ready to pass into QuantileLGBM.

    If Optuna is not installed, falls back to the lgbm_defaults from config.
    """
    # Fallback defaults (used when tuning is disabled or Optuna absent)
    defaults = {}
    if config and "lgbm_defaults" in config:
        defaults = dict(config["lgbm_defaults"])

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [INFO] Optuna not installed - using lgbm_defaults (set tuning.enabled=false to suppress)")
        return defaults

    import lightgbm as lgb

    search = config.get("tuning", {}).get("search_space", {}) if config else {}

    def objective(trial):
        params = {
            "objective":          "quantile",
            "alpha":              0.50,
            "metric":             "quantile",
            "verbose":            -1,
            "num_leaves":         trial.suggest_int(
                                    "num_leaves",
                                    search.get("num_leaves", [31, 127])[0],
                                    search.get("num_leaves", [31, 127])[1]),
            "learning_rate":      trial.suggest_float(
                                    "learning_rate",
                                    search.get("learning_rate", [0.01, 0.1])[0],
                                    search.get("learning_rate", [0.01, 0.1])[1],
                                    log=True),
            "min_child_samples":  trial.suggest_int(
                                    "min_child_samples",
                                    search.get("min_child_samples", [10, 50])[0],
                                    search.get("min_child_samples", [10, 50])[1]),
            "feature_fraction":   trial.suggest_float(
                                    "feature_fraction",
                                    search.get("feature_fraction", [0.6, 0.9])[0],
                                    search.get("feature_fraction", [0.6, 0.9])[1]),
            "bagging_fraction":   trial.suggest_float(
                                    "bagging_fraction",
                                    search.get("bagging_fraction", [0.6, 0.9])[0],
                                    search.get("bagging_fraction", [0.6, 0.9])[1]),
            "bagging_freq":       5,
            "reg_alpha":          trial.suggest_float(
                                    "reg_alpha",
                                    search.get("reg_alpha", [0.01, 1.0])[0],
                                    search.get("reg_alpha", [0.01, 1.0])[1],
                                    log=True),
            "reg_lambda":         trial.suggest_float(
                                    "reg_lambda",
                                    search.get("reg_lambda", [0.01, 1.0])[0],
                                    search.get("reg_lambda", [0.01, 1.0])[1],
                                    log=True),
        }
        n_est      = int(search.get("n_estimators", 500))
        early_stop = int(search.get("early_stopping_rounds", 30))

        model = lgb.LGBMRegressor(n_estimators=n_est, **params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds  = model.predict(X_val)
        alpha  = 0.50
        diff   = y_val - preds
        pinball = float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))
        return pinball

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # Add fixed params back
    best["bagging_freq"]         = 5
    best["early_stopping_rounds"] = int(search.get("early_stopping_rounds", 30))
    best["n_estimators"]          = int(search.get("n_estimators", 500))
    best["verbose"]               = -1

    print(f"  Best trial: pinball={study.best_value:.4f}  params={best}")
    return best
