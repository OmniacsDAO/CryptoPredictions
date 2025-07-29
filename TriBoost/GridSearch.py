# ───────────────────── imports ──────────────────────
from pathlib import Path
import numpy as np
import pandas as pd
import optuna, warnings, pickle, gc
import time

from lightgbm import LGBMRegressor, early_stopping
from catboost  import CatBoostRegressor
from xgboost   import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from utils import *
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────── config  ──────────────────────
CSV_PATH   = Path("data/solusd_history.csv")
OUT_PKL    = Path("models/best_params.pkl")
HORIZON    = 480
SEED       = 42
FOLDS      = 5
TRIALS     = 5
BASE_MODELS = ("lgb", "cb", "xgb")

OUT_PKL.parent.mkdir(exist_ok=True, parents=True)

# ──────────────────── data prep ─────────────────────
print("Loading & engineering …")
raw = (
    pd.read_csv(CSV_PATH, parse_dates=["date"])
      .sort_values("date")
      .drop_duplicates("date")
)
raw["future_close"] = raw["close"].shift(-HORIZON)
log_ret             = 100 * np.log(raw["future_close"] / raw["close"])
raw["target_ret"]   = np.sign(log_ret) * np.log1p(np.abs(log_ret))
raw["w"]            = (1 + np.abs(raw["target_ret"])).clip(
                        upper=raw["target_ret"].abs().quantile(0.95))

labeled      = add_features(raw).dropna()
FEATURE_COLS = [
    c for c in labeled.columns
    if c not in ("ticker", "date", "future_close", "target_ret", "w")
]
X_full, y_full, w_full = (
    labeled[FEATURE_COLS],
    labeled["target_ret"],
    labeled["w"]
)

# ───────────────────— helpers ───────────────────────
def lgb_dir_obj(y_true, y_pred):
    err  = y_pred - y_true
    grad = err + 1.5 * np.sign(err) * (np.sign(y_pred) != np.sign(y_true))
    hess = np.ones_like(grad)
    return grad, hess

def time_series_splits(n_obs, n_folds=FOLDS):
    split = TimeSeriesSplit(n_splits=n_folds)
    idx   = np.arange(n_obs)
    for tr, va in split.split(idx):
        yield tr, va

# ─────────────── tuning routine (per model) ─────────
def tune_once(model_name: str):
    """Return (best_params:dict, best_iter:int) for one algo."""
    t0 = time.perf_counter()
    def objective(trial):
        # ← define search space ------------------------------------
        if model_name == "lgb":
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                "max_depth":     trial.suggest_int("depth", 4, 12),
                # "learning_rate":   trial.suggest_float("lr", 0.005, 0.05, log=True),
                "num_leaves":      trial.suggest_int("leaves", 63, 511, step=32),
                "subsample":       trial.suggest_float("subsample", .5, 1),
                "colsample_bytree":trial.suggest_float("colsample", .5, 1),
                "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
                "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
                "min_child_samples":trial.suggest_int("min_child", 10, 200),
                "objective":       lgb_dir_obj,
                "n_estimators":    4000,
                "random_state":    SEED,
                "n_jobs":          -1,
                "verbose":         -1,
            }
            model = LGBMRegressor(**params)

        elif model_name == "cb":
            params = {
                "learning_rate": trial.suggest_float("lr", 5e-4, 1e-2, log=True),
                # "learning_rate":   trial.suggest_float("lr", 0.01, 0.1, log=True),
                "depth":           trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg":     trial.suggest_float("l2", 1, 10, log=True),
                "bagging_temperature": trial.suggest_float("temp", 0, 1),
                "loss_function":   "MAE",
                "iterations":      4000,
                "random_seed":     SEED,
                "verbose":         False,
            }
            model = CatBoostRegressor(**params)

        else:  # xgb
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                # "learning_rate":   trial.suggest_float("lr", 0.01, 0.1, log=True),
                "max_depth":       trial.suggest_int("depth", 3, 10),
                "subsample":       trial.suggest_float("subsample", .6, 1),
                "colsample_bytree":trial.suggest_float("colsample", .6, 1),
                "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
                "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
                "n_estimators":    4000,
                "random_state":    SEED,
                "objective":       "reg:squarederror",
                "eval_metric":     "rmse",
                "early_stopping_rounds": 150,
            }
            model = XGBRegressor(**params, n_jobs=-1)

        # ← evaluate hyper-params ---------------------------------
        scores, best_iters = [], []

        for tr, va in time_series_splits(len(X_full)):
            X_tr, X_va = X_full.iloc[tr], X_full.iloc[va]
            y_tr, y_va = y_full.iloc[tr], y_full.iloc[va]
            w_tr, w_va = w_full.iloc[tr], w_full.iloc[va]

            if model_name == "cb":
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va)], early_stopping_rounds=150,
                          verbose=False)
                best_iters.append(model.get_best_iteration())

            elif model_name == "lgb":
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va)],
                          eval_sample_weight=[w_va],
                          callbacks=[early_stopping(150, verbose=False)])
                best_iters.append(model.best_iteration_)

            else:  # xgb
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va)],
                          sample_weight_eval_set=[w_va],
                          verbose=False)
                best_iters.append(getattr(model,
                                          "best_iteration_",
                                          getattr(model, "best_iteration", model.n_estimators)))

            scores.append(mean_squared_error(y_va, model.predict(X_va)))

        trial.set_user_attr("best_iter", int(np.mean(best_iters)))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

    elapsed = time.perf_counter() - t0
    print(f"[{model_name}] best CV-RMSE={study.best_value:8.5f}, "
          f"best_iter={study.best_trial.user_attrs['best_iter']}, "f"time={elapsed:7.1f}s")
    return study.best_params, study.best_trial.user_attrs["best_iter"]

# ────────────────── run tuning & store ─────────────────
t_global = time.perf_counter()
param_bank = {}
for algo in BASE_MODELS:
    param_bank[algo] = {}
    best_hp, best_iter = tune_once(algo)
    param_bank[algo]["best_params"] = best_hp
    param_bank[algo]["best_iter"]   = best_iter
    gc.collect()

with open(OUT_PKL, "wb") as f:
    pickle.dump(param_bank, f)

print(f"\n✅ Hyper-parameters serialized to {OUT_PKL.resolve()}")
print(f"🏁 Total script time: {time.perf_counter() - t_global:,.1f}s")