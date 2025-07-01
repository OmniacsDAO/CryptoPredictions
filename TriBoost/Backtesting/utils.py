import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta, timezone

# ---------------------- Helper functions -----------------------------------

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta, timezone

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # --- Return and sign features ---
    for lag in (1, 5, 15, 60, 120):
        out[f"ret_{lag}"] = 100 * out["close"].pct_change(lag)
        out[f"ret_sign_{lag}"] = np.sign(out["close"].pct_change(lag))
    
    out["avg_ret_sign_60"] = out[[f"ret_sign_{l}" for l in [1, 5, 15, 30, 60] if f"ret_sign_{l}" in out.columns]].mean(axis=1)

    # --- Momentum indicators ---
    out["roc_30"] = ta.momentum.roc(out["close"], window=30, fillna=True)
    out["roc_120"] = ta.momentum.roc(out["close"], window=120, fillna=True)

    # --- EMAs and ratios ---
    out["ema_20"] = ta.trend.ema_indicator(out["close"], window=20, fillna=True)
    out["ema_50"] = ta.trend.ema_indicator(out["close"], window=50, fillna=True)
    out["ema_100"] = ta.trend.ema_indicator(out["close"], window=100, fillna=True)
    out["ema20_close_ratio"] = out["close"] / (out["ema_20"] + 1e-8)
    out["ema50_close_ratio"] = out["close"] / (out["ema_50"] + 1e-8)
    out["ema_slope_20"] = out["ema_20"].diff()
    out["ema_slope_50"] = out["ema_50"].diff()
    out["ema_reversal"] = ((out["ema_20"] < out["ema_50"]) & (out["ema_20"].shift() > out["ema_50"].shift())).astype(int)

    # --- RSI ---
    out["rsi_14"] = ta.momentum.rsi(out["close"], window=14, fillna=True)
    out["rsi_overbought"] = (out["rsi_14"] > 70).astype(int)
    out["rsi_oversold"] = (out["rsi_14"] < 30).astype(int)
    out["rsi_cross_up"] = ((out["rsi_14"] > 30) & (out["rsi_14"].shift() <= 30)).astype(int)
    out["rsi_cross_down"] = ((out["rsi_14"] < 70) & (out["rsi_14"].shift() >= 70)).astype(int)

    # --- MACD ---
    out["macd"] = ta.trend.macd_diff(out["close"], fillna=True)
    out["macd_slope"] = out["macd"].diff()

    # --- Bollinger Bands ---
    bb_mid = ta.volatility.bollinger_mavg(out["close"], window=20, fillna=True)
    bb_high = ta.volatility.bollinger_hband(out["close"], window=20, fillna=True)
    bb_low = ta.volatility.bollinger_lband(out["close"], window=20, fillna=True)
    out["boll_width"] = (bb_high - bb_low) / (bb_mid + 1e-8)
    out["boll_z"] = (out["close"] - bb_mid) / ((bb_high - bb_low) + 1e-8)

    # --- Volatility ---
    out["rolling_std_30"] = out["close"].rolling(30).std()
    out["rolling_std_120"] = out["close"].rolling(120).std()
    out["volatility_z_120"] = (out["rolling_std_120"] - out["rolling_std_120"].mean()) / (out["rolling_std_120"].std() + 1e-8)

    # --- Candle body and volume ---
    out["body_size"] = np.abs(out["close"] - out["open"])
    out["full_range"] = out["high"] - out["low"]
    out["body_to_range"] = out["body_size"] / (out["full_range"] + 1e-8)

    out["vol_norm"] = out["volume"] / (out["volume"].rolling(60).mean() + 1e-8)
    out["cvd"] = (np.sign(out["close"].diff()) * out["volume"]).fillna(0).cumsum()

    # --- VWAP intraday delta ---
    grp = out["date"].dt.date
    volume_cumsum = out["volume"].groupby(grp).transform("cumsum")
    vwap_numerator = (out["volume"] * out["close"]).groupby(grp).transform("cumsum")
    vwap = vwap_numerator / (volume_cumsum + 1e-8)
    out["vwap_delta"] = out["close"] - vwap

    # --- Price action pattern ---
    out["direction_streak"] = (np.sign(out["close"].diff()) == np.sign(out["close"].diff().shift())).rolling(10).sum()
    out["is_local_high"] = (out["close"] == out["close"].rolling(60).max()).astype(int)
    out["is_local_low"] = (out["close"] == out["close"].rolling(60).min()).astype(int)

    # --- Time features ---
    out["minute_of_day"] = out["date"].dt.hour * 60 + out["date"].dt.minute
    out["sin_time"] = np.sin(2 * np.pi * out["minute_of_day"] / 1440)
    out["cos_time"] = np.cos(2 * np.pi * out["minute_of_day"] / 1440)

    out["dow"] = out["date"].dt.dayofweek
    out = pd.get_dummies(out, columns=["dow"], drop_first=False)

    return out



def align(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Match training feature layout (missing → 0, extra → dropped)."""
    return df.reindex(columns=feature_cols, fill_value=0)