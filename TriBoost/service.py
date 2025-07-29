#!/usr/bin/env python3
"""
Start with:
    uvicorn service:app --host 0.0.0.0 --port 4562
"""

# ─────────────── imports (unchanged) ────────────────
from pathlib import Path
import numpy as np
import pandas as pd
import pickle, warnings, gc

from lightgbm import LGBMRegressor, log_evaluation
from catboost  import CatBoostRegressor
from xgboost   import XGBRegressor

from fastapi import FastAPI, HTTPException
from utils    import *      # your add_features(), etc.

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────── config (unchanged) ────────────────
CSV_PATH     = Path("data/solusd_history.csv")
PARAM_PKL    = Path("models/best_params.pkl")
HORIZON      = 480                  # 8 h
SEED         = 42
BASE_MODELS  = ("lgb", "cb", "xgb")

# ───────────────── helpers (unchanged) ─────────────
def lgb_dir_obj(y_true, y_pred):
    """Directional-aware objective (same as tuning stage)."""
    err  = y_pred - y_true
    grad = err + 1.5 * np.sign(err) * (np.sign(y_pred) != np.sign(y_true))
    hess = np.ones_like(grad)
    return grad, hess

def price_from_logret(close, lr):
    """
    Convert the custom log-return target back to a price.
    `lr` is the model prediction; `close` is the current close price.
    """
    return close * np.exp(np.sign(lr) * (np.expm1(np.abs(lr)) / 100))

# ═════════════════ API ══════════════════════════════
app = FastAPI(
    title   ="SOL/USD 8-hour Forecast API",
    version ="0.1.0",
    summary ="Returns an 8-hour ahead price forecast built from LightGBM, CatBoost and XGBoost ensemble."
)

def compute_forecast():
    "Runs the whole pipeline once and returns a dict with results."

    # ─────────────── data prep (unchanged) ──────────
    raw = (
        pd.read_csv(CSV_PATH, parse_dates=["date"])
          .sort_values("date")
          .drop_duplicates("date")
    )

    # target engineering (same formula as in script #1)
    raw["future_close"] = raw["close"].shift(-HORIZON)
    log_ret             = 100 * np.log(raw["future_close"] / raw["close"])
    raw["target_ret"]   = np.sign(log_ret) * np.log1p(np.abs(log_ret))
    raw["w"]            = (1 + np.abs(raw["target_ret"])).clip(
                              upper=raw["target_ret"].abs().quantile(0.95)
                          )
    labeled      = add_features(raw)

    train_df   = labeled.dropna().copy()
    predict_df = labeled.tail(1)            # single most-recent bar

    EXCLUDE = {"ticker", "date", "future_close", "target_ret", "w"}
    FEATURE_COLS = [c for c in train_df.columns if c not in EXCLUDE]

    X, y, wts = train_df[FEATURE_COLS], train_df["target_ret"], train_df["w"]
    X_last    = predict_df[FEATURE_COLS]

    # ───────────── load tuned h-params (unchanged) ──
    with open(PARAM_PKL, "rb") as f:
        param_bank = pickle.load(f)

    # ───────────── fit models & predict (unchanged) ─
    base_preds = []
    for algo in BASE_MODELS:
        hp    = param_bank[algo]["best_params"].copy()
        n_est = param_bank[algo]["best_iter"]
        n_est = n_est + max(1, int(n_est * 0.10))   # +10 % cushion

        if algo == "lgb":
            params = {
                "learning_rate":      hp["lr"],
                "num_leaves":         hp["leaves"],
                "subsample":          hp["subsample"],
                "colsample_bytree":   hp["colsample"],
                "reg_alpha":          hp["reg_alpha"],
                "reg_lambda":         hp["reg_lambda"],
                "min_child_samples":  hp["min_child"],
                "objective":          lgb_dir_obj,
                "n_estimators":       n_est,
                "random_state":       SEED,
                "n_jobs":             -1,
                "verbose":            -1,
            }
            mdl = LGBMRegressor(**params)
            mdl.fit(
                X, y, sample_weight=wts,
                eval_set=[(X, y)],
                eval_sample_weight=[wts],
                callbacks=[log_evaluation(250)]
            )

        elif algo == "cb":
            params = {
                "learning_rate":       hp["lr"],
                "depth":               hp["depth"],
                "l2_leaf_reg":         hp["l2"],
                "bagging_temperature": hp["temp"],
                "loss_function":       "MAE",
                "iterations":          n_est,
                "random_seed":         SEED,
                "verbose":             False,
            }
            mdl = CatBoostRegressor(**params)
            mdl.fit(X, y, sample_weight=wts, verbose=False)

        else:  # xgb
            params = {
                "learning_rate":    hp["lr"],
                "max_depth":        hp["depth"],
                "subsample":        hp["subsample"],
                "colsample_bytree": hp["colsample"],
                "reg_lambda":       hp["reg_lambda"],
                "reg_alpha":        hp["reg_alpha"],
                "n_estimators":     n_est,
                "random_state":     SEED,
                "objective":        "reg:squarederror",
                "eval_metric":      "rmse",
            }
            mdl = XGBRegressor(**params, n_jobs=-1)
            mdl.fit(X, y, sample_weight=wts)

        pred_lr = mdl.predict(X_last)[0]
        base_preds.append(pred_lr)
        gc.collect()

    ensemble_lr   = float(np.mean(base_preds))
    latest_close  = predict_df["close"].values[0]
    predicted_px  = price_from_logret(latest_close, ensemble_lr)

    return {
        "timestamp":        predict_df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").values[0],
        "current_close":    float(latest_close),
        "pred_log_return":  ensemble_lr,
        "predicted_price":  float(predicted_px),
        "horizon_minutes":  HORIZON,
    }

# ─────────────── endpoints ─────────────────────────
@app.get("/forecast", summary="8-hour ahead SOL/USD price forecast")
def forecast():
    """
    Trains the three base learners with the most-recent data,
    ensembles them, and returns the prediction.
    """
    try:
        return compute_forecast()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ─────────────── CLI runner ────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=4562, reload=False)
