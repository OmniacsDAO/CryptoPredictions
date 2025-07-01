# TriBoost — triple-booster ensemble for 8-hour SOL/USDC price forecasts

TriBoost stacks **LightGBM + CatBoost + XGBoost** on top of a rich, minute-level feature set to predict the SOL/USDC closing price 8 hours (480 minutes) ahead.
Everything lives in this folder—no deep nets, no secret endpoints—just three fast tree learners, careful feature engineering, and exhaustive time-series cross-validation.

---

## Folder contents

| Step | Script / Dir               | Purpose                                                                                                                                                                            |
| ---- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `utils.py`                 | Generate >40 technical-analysis and price-action features (returns, EMAs, RSI, MACD, Bollinger-band width, VWAP delta, intraday time encodings, etc.).                             |
| 2    | `GridSearch.py`            | Use Optuna to tune each booster with rolling 5-fold time-series CV on the **log-return target**; best parameters are saved to `models/best_params.pkl`.                            |
| 3    | `FitModelPredictManual.py` | Load tuned parameters, refit the three models on all data, average their predictions, and print the 8-hour-ahead forecast.                                                         |
| 4    | `Backtesting/`             | `Backtesting.py` performs walk-forward evaluation; `BTbench.py` computes direction accuracy, Pearson *r*, and weighted error metrics; `BTbenchPlot.py` produces diagnostics plots. |

---

## Target definition

```text
target_ret  = sign(log_ret) * log1p(|log_ret|)
log_ret     = ln(future_close / close) * 100
horizon     = 480 minutes  # 8 hours
```

A custom **direction-aware objective** adjusts LightGBM gradients so wrong-way bets receive a steeper penalty than size errors.

---

## Quick start

```bash
# 0) create environment & install dependencies
pip install lightgbm catboost xgboost optuna ta numpy pandas scikit-learn tqdm matplotlib seaborn requests


# 1) hyper-parameter search  (~30–40 min on a modern laptop)
python GridSearch.py

# 2) one-off forecast
python FitModelPredictManual.py

# 3) full walk-forward back-test (hours; each slice re-tunes with Optuna)
python Backtesting/Backtesting.py
```

---

## Why “Tri” Boost?

1. **Diversified learners** – gradient-based (LightGBM, XGBoost) plus symmetry-aware oblivious trees (CatBoost).
2. **Robust ensemble** – the simple mean of three models consistently beats the single best model on direction accuracy, RMSE, and tail-error metrics.
3. **Transparent** – every line is here; fork it, tweak it, rerun it.

---

## Caveats & TODOs

* Currently hard-coded to SOL/USDC; update `TICKER` and feature windows for other assets.
* Back-tests evaluate forecast quality only—no slippage or trading-fee modelling.
* Optuna trial counts are modest for speed; expand them for production use.


---

⚡ Fueling public goods with [$IACS](http://dexscreener.com/base/0xd4d742cc8f54083f914a37e6b0c7b68c6005a024) — Get involved: 0x46e69Fa9059C3D5F8933CA5E993158568DC80EBf (Base)