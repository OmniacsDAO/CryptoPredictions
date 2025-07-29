# TriBoost — triple-booster ensemble for 8-hour SOL/USD forecasts

TriBoost stacks **LightGBM, CatBoost and XGBoost** on top of a rich, minute-level feature set to predict the SOL/USDC closing price 8 hours (480 minutes) ahead.

---

## Folder overview

| Step | File / Dir | Purpose |
|------|-------------|---------|
| 1    | `utils.py` | Creates > 40 technical-analysis & price-action features (returns, EMAs, RSI, MACD, Bollinger width, VWAP delta, intraday time encodings, etc.). |
| 2    | `GridSearch.py` | Tunes each booster with rolling 5-fold time-series CV; best params saved to `models/best_params.pkl`. |
| 3    | `FitModelPredictManual.py` | Loads tuned params, refits the three models on all data and prints a one-off 8-hour forecast. |
| 4    | `service.py` | **New.** FastAPI micro-service exposing `/forecast` for on-demand predictions . |
| 5    | `Dockerfile` | Slim Python 3.11 image with all requirements pre-installed (see `requirements.txt`). |
| 6    | `docker-compose.yml` | One-liner dev/prod deployment; maps port `4562` and mounts `data/` + `models/` volumes . |
| 7    | `requirements.txt` | Pinned runtime deps (FastAPI + Uvicorn, tree boosters, NumPy/Pandas) . |
| 8    | `data/` | OHLCV‐history CSV(s); `.gitignore`d by default. |
| 9    | `models/` | Stores `best_params.pkl` and any pickled model artefacts. |
| 10   | `Backtesting/` | `Backtesting.py` performs walk-forward evaluation; `BTbench.py` computes direction accuracy, Pearson *r*, and weighted error metrics; `BTbenchPlot.py` produces diagnostics|

---

## Target definition

```text
target_ret  = sign(log_ret) * log1p(|log_ret|)
log_ret     = ln(future_close / close) * 100
horizon     = 480 minutes  # 8 h
````

A custom **direction-aware objective** steepens LightGBM gradients whenever the sign is wrong.

---

## Quick-start options

<details>
<summary>🔧 Local Python workflow</summary>

```bash
# 1) Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # FastAPI, boosters, pandas, numpy …

# 2) Hyper-param search (~30 min laptop)
python GridSearch.py

# 3) One-off forecast
python FitModelPredictManual.py
```

</details>

<details>
<summary>🌐 Run the FastAPI service (no Docker)</summary>

```bash
uvicorn service:app --host 0.0.0.0 --port 4562
#               ↑ edit the port if you like
```

**Endpoint**

* `GET /forecast` – trains/ensembles the three boosters on the latest data and returns a JSON payload .

Example response (keys will match your runtime values):

```json
{
  "timestamp":        "2025-07-29T07:45:00+0000",
  "current_close":    173.28,
  "pred_log_return":  0.0182,
  "predicted_price":  176.44,
  "horizon_minutes":  480
}
```

The schema is generated automatically by FastAPI; open `http://localhost:4562/docs` for Swagger UI.

</details>

<details>
<summary>🐳 Docker (recommended for prod)</summary>

```bash
# Build & launch
docker compose up --build      # foreground
# or
docker compose up -d           # detached
```

The compose file exposes port `4562`, binds `./data` and `./models` for live retraining, and names the container `sol-forecast-api` .

</details>

---

## API anatomy

| Field             | Meaning                                                     |
| ----------------- | ----------------------------------------------------------- |
| `timestamp`       | UTC time of the last bar used for features                  |
| `current_close`   | Latest close price in USD                                   |
| `pred_log_return` | Ensembled log-return prediction (direction-aware transform) |
| `predicted_price` | Back-transformed SOL/USD price expected in 8 h              |
| `horizon_minutes` | Forecast horizon (480 = 8 hours)                            |

All values are returned by `compute_forecast()` in `service.py` .

---

## Why “Tri” Boost?

1. **Diversified learners** – gradient-based LightGBM & XGBoost plus symmetry-aware CatBoost.
2. **Robust ensemble** – the simple mean of three models beats the best single model on direction accuracy & RMSE.
3. **Transparent** – every line is in this repo; fork it, tweak it, rerun it.

---

## Caveats & TODOs

* Currently hard-coded to SOL/USDC; change `CSV_PATH`, `HORIZON`, and feature windows for other assets.
* Back-tests evaluate forecast quality only—no slippage or trading-fee modelling.
* Optuna trial counts are modest for speed; bump them for production use.

---

⚡ Fueling public goods with [\$IACS](https://dexscreener.com/base/0xd4d742cc8f54083f914a37e6b0c7b68c6005a024) — Get involved: 0x46e69Fa9059C3D5F8933CA5E993158568DC80EBf (Base)
