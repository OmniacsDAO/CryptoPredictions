# ───────────────────────────── imports ─────────────────────────────
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle, joblib, optuna, gc, warnings

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from utils   import *          # add_features, etc.
from BTbench import *          # evaluation + plotting helpers
# -------------------------------------------------------------------

# ───────────────────────────── config ──────────────────────────────
CSV_PATH       = Path("solusd_history.csv")
HORIZON        = 480               # 8-hour horizon  (1-min bars)
RESULTS_PKL    = Path("temp/ResBTbenchBT6.pkl")
PLOT_PATH      = Path("temp/PlotBTbenchBT6.png")

SEED          = 42
FOLDS         = 5                  # inner CV folds (Optuna)
TUNING_TRIALS = 5                 # trials / model / month
BASE_MODELS   = ("lgb", "cb", "xgb")

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── data ────────────────────────────────
print("Loading CSV …")
raw = (
    pd.read_csv(CSV_PATH, parse_dates=["date"])
      .sort_values("date")
      .drop_duplicates("date")
)
print(f"Loaded {len(raw):,} minute candles")

raw["future_close"] = raw["close"].shift(-HORIZON)
log_ret             = 100 * np.log(raw["future_close"] / raw["close"])
raw["target_ret"]   = np.sign(log_ret) * np.log1p(np.abs(log_ret))
raw["w"]            = (1 + np.abs(raw["target_ret"])).clip(
                        upper=raw["target_ret"].abs().quantile(0.95))

labeled      = add_features(raw).dropna()
monthly_dfs  = [
    g.reset_index(drop=True)
    for _, g in labeled.groupby(labeled["date"].dt.to_period("M"), sort=False)
]
FEATURE_COLS = [
    c for c in labeled.columns
    if c not in ("ticker", "date", "future_close", "target_ret", "w")
]

# ──────────────────────────── helpers ─────────────────────────────
def price_from_logret(close, lr):
    return close * np.exp(np.sign(lr) * (np.expm1(np.abs(lr)) / 100))

def lgb_dir_obj(y_true, y_pred):
    """LightGBM objective emphasising correct direction."""
    err  = y_pred - y_true
    grad = err + 1.5 * np.sign(err) * (np.sign(y_pred) != np.sign(y_true))
    hess = np.ones_like(grad)
    return grad, hess

def time_series_splits(X, n_folds=FOLDS):
    splitter = TimeSeriesSplit(n_splits=n_folds)
    idx = np.arange(len(X))
    for tr_idx, va_idx in splitter.split(idx):
        yield tr_idx, va_idx

# ──────────────────────── tuning + fit ────────────────────────────
def tune_model(name, X, y, w):
    """Bayesian tuning; returns a refitted model using the CV-optimal #iterations."""
    # ―――――――― inner objective ――――――――
    def objective(trial):
        # --- hyper-parameter space ---
        if name == "lgb":
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                "max_depth":     trial.suggest_int("depth", 4, 12),
                # "learning_rate": trial.suggest_float("lr", 0.005, 0.05, log=True),
                "num_leaves":    trial.suggest_int("leaves", 63, 511, step=32),
                "subsample":     trial.suggest_float("subsample", .5, 1),
                "colsample_bytree": trial.suggest_float("colsample", .5, 1),
                "reg_alpha":     trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
                "reg_lambda":    trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
                "min_child_samples": trial.suggest_int("min_child", 10, 200),
                "objective":     lgb_dir_obj,
                "n_estimators":  4000,
                "random_state":  SEED,
                "n_jobs":        -1,
                "verbose":       -1,
            }
            model = LGBMRegressor(**params)

        elif name == "cb":
            params = {
                "learning_rate": trial.suggest_float("lr", 5e-4, 1e-2, log=True),
                # "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                "depth":         trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg":   trial.suggest_float("l2", 1, 10, log=True),
                "bagging_temperature": trial.suggest_float("temp", 0, 1),
                "loss_function": "MAE",
                "iterations":    4000,
                "random_seed":   SEED,
                "verbose": False,
            }
            model = CatBoostRegressor(**params)

        else:  # xgb
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                # "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
                "max_depth":     trial.suggest_int("depth", 3, 10),
                "subsample":     trial.suggest_float("subsample", .6, 1),
                "colsample_bytree": trial.suggest_float("colsample", .6, 1),
                "reg_lambda":    trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
                "reg_alpha":     trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
                "n_estimators":  4000,
                "random_state":  SEED,
                "objective":    "reg:squarederror",
                "eval_metric":  "rmse",
                "early_stopping_rounds": 150,
            }
            model = XGBRegressor(**params, n_jobs=-1)

        # --- CV evaluation ---
        scores, best_iters = [], []

        for tr_idx, va_idx in time_series_splits(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            w_tr, w_va = w.iloc[tr_idx], w.iloc[va_idx]

            if name == "cb":
                model.fit(X_tr, y_tr,
                          sample_weight=w_tr,
                          eval_set=[(X_va, y_va)],
                          early_stopping_rounds=150,
                          verbose=False)
                best_iters.append(model.get_best_iteration())

            elif name == "lgb":
                model.fit(X_tr, y_tr,
                          sample_weight=w_tr,
                          eval_set=[(X_va, y_va)],
                          eval_sample_weight=[w_va],
                          callbacks=[early_stopping(150, verbose=False)])
                best_iters.append(model.best_iteration_)

            else:  # xgb
                model.fit(X_tr, y_tr,
                          sample_weight=w_tr,
                          eval_set=[(X_va, y_va)],
                          sample_weight_eval_set=[w_va],
                          verbose=False)
                # XGB may store either best_iteration_ or best_iteration
                best_iters.append(
                    getattr(model, "best_iteration_", getattr(model, "best_iteration", model.n_estimators))
                )

            scores.append(mean_squared_error(y_va, model.predict(X_va)))

        # store avg best-iter so we can pull it later
        trial.set_user_attr("best_iter", int(np.round(np.mean(best_iters))))
        return np.mean(scores)

    # ―――――――― Optuna search ――――――――
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=TUNING_TRIALS, show_progress_bar=False)

    best_hp   = study.best_params
    best_iter = study.best_trial.user_attrs["best_iter"]

    # ―――――――― final refit on full slice ――――――――
    if name == "lgb":
        params = {
            "learning_rate": best_hp["lr"],
            "num_leaves":    best_hp["leaves"],
            "subsample":     best_hp["subsample"],
            "colsample_bytree": best_hp["colsample"],
            "reg_alpha":     best_hp["reg_alpha"],
            "reg_lambda":    best_hp["reg_lambda"],
            "min_child_samples": best_hp["min_child"],
            "objective":     lgb_dir_obj,
            "n_estimators":  best_iter+max(1, int(best_iter * 0.10)),      # ← use optimal rounds
            "random_state":  SEED,
            "n_jobs":        -1,
            "verbose":       -1,
        }
        model = LGBMRegressor(**params)
        model.fit(X, y,
                  sample_weight=w,
                  eval_set=[(X, y)],
                  eval_sample_weight=[w],
                  callbacks=[log_evaluation(250)])

    elif name == "cb":
        params = {
            "learning_rate":     best_hp["lr"],
            "depth":             best_hp["depth"],
            "l2_leaf_reg":       best_hp["l2"],
            "bagging_temperature": best_hp["temp"],
            "loss_function":     "MAE",
            "iterations":        best_iter+max(1, int(best_iter * 0.10)),   # ← use optimal rounds
            "random_seed":       SEED,
            "verbose":           False,
        }
        model = CatBoostRegressor(**params)
        model.fit(X, y, sample_weight=w, verbose=False)

    else:  # xgb
        params = {
            "learning_rate": best_hp["lr"],
            "max_depth":     best_hp["depth"],
            "subsample":     best_hp["subsample"],
            "colsample_bytree": best_hp["colsample"],
            "reg_lambda":    best_hp["reg_lambda"],
            "reg_alpha":     best_hp["reg_alpha"],
            "n_estimators":  best_iter+max(1, int(best_iter * 0.10)),       # ← use optimal rounds
            "random_state":  SEED,
            "objective":    "reg:squarederror",
            "eval_metric":  "rmse",
        }
        model = XGBRegressor(**params, n_jobs=-1)
        model.fit(X, y, sample_weight=w)

    return model

# ───────────────────────── back-testing ────────────────────────────
results = {}

for test_idx in tqdm(range(len(monthly_dfs)), desc="Processing months"):
    month_lbl = monthly_dfs[test_idx]["date"].iloc[0].strftime("%Y-%m")
    print(f"\n🔍  Back-testing on {month_lbl}")

    # ── split current test month vs. “all other” rows ─────────────
    test_df   = monthly_dfs[test_idx]
    train_dfs = monthly_dfs[:test_idx] + monthly_dfs[test_idx + 1:]
    train_df  = pd.concat(train_dfs, ignore_index=True)

    X_tr, y_tr, w_tr = train_df[FEATURE_COLS], train_df["target_ret"], train_df["w"]
    X_te, close_te   = test_df[FEATURE_COLS], test_df["close"].values

    # ── base-model ensemble ───────────────────────────────────────
    base_preds = np.zeros((len(BASE_MODELS), len(test_df)))

    for m, name in enumerate(BASE_MODELS):
        print(f"   ↳ tuning {name} …")
        mdl = tune_model(name, X_tr, y_tr, w_tr)
        base_preds[m] = mdl.predict(X_te)
        gc.collect()

    final_pred = base_preds.mean(axis=0)       # simple mean ensemble
    pred_close = price_from_logret(close_te, final_pred)

    out = test_df.copy()
    out["pred_close"] = pred_close
    results[month_lbl] = out

    with open(RESULTS_PKL, "wb") as f:
        pickle.dump(results, f)

    metrics_df = evaluate_monthly_metrics(results)
    plot_metric_timeseries(metrics_df, filename=PLOT_PATH)

# ───────────────────────────── final ───────────────────────────────
with open(RESULTS_PKL, "wb") as f:
    pickle.dump(results, f)

metrics_df = evaluate_monthly_metrics(results)
plot_metric_timeseries(metrics_df, filename=PLOT_PATH)
print("\n✅ Done. Metrics saved and plot updated.")
