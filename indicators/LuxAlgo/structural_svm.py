"""
Structural SVM Ranker [LuxAlgo]

Detects market structure breaks (BOS/CHoCH) using pivot highs/lows,
then scores each break with an SVM-like weighted sum of:
  - Relative volume (vol / SMA(vol,20))
  - RSI momentum (deviation from 50)
  - Break distance (ATR-normalized)

The score is passed through a sigmoid to produce a 0-100 quality ranking.

Pine source: indicators/LuxAlgo/Structural_SVM_Ranker__LuxAlgo_.pine

NOTE: Uses volume. For BTC Renko where volume=0, rel_vol will be 0,
so only RSI and distance features will contribute. Consider setting
vol_weight=0 for zero-volume data.

Outputs:
    trend       — market structure trend: +1 (bullish) / -1 (bearish) / 0 (uninitialized)
    break_type  — 0 (none), 1 (BOS), 2 (CHoCH) on break bars
    break_bull  — True if the break is bullish
    svm_score   — 0-100 SVM quality score on break bars, NaN otherwise
    pivot_hi    — detected pivot high values (NaN on non-pivot bars)
    pivot_lo    — detected pivot low values (NaN on non-pivot bars)

Usage:
    from indicators.luxalgo.structural_svm import calc_structural_svm
    result = calc_structural_svm(df, pivot_len=5)
"""

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    return 100.0 / (1.0 + np.exp(-x))


def calc_structural_svm(
    df: pd.DataFrame,
    pivot_len: int = 5,
    vol_weight: float = 0.4,
    rsi_weight: float = 0.3,
    dist_weight: float = 0.3,
    atr_len: int = 14,
) -> dict:
    """
    Parameters
    ----------
    df          : DataFrame with 'High', 'Low', 'Close', 'Volume'
    pivot_len   : Lookback/forward for pivot detection
    vol_weight  : Weight for relative volume feature
    rsi_weight  : Weight for RSI momentum feature
    dist_weight : Weight for break distance feature
    atr_len     : ATR period for distance normalization
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(len(close))
    n = len(close)

    # Pre-compute RSI(14)
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # EMA-style RSI (Wilder smoothing)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi_period = 14
    # Seed with SMA
    if n > rsi_period:
        avg_gain[rsi_period] = np.mean(gains[1:rsi_period + 1])
        avg_loss[rsi_period] = np.mean(losses[1:rsi_period + 1])
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gains[i]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + losses[i]) / rsi_period

    rsi_vals = np.full(n, 50.0)
    for i in range(rsi_period, n):
        if avg_loss[i] == 0:
            rsi_vals[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_vals[i] = 100.0 - 100.0 / (1.0 + rs)

    rsi_feature = np.abs(rsi_vals - 50.0) / 50.0

    # Pre-compute relative volume (vol / SMA(vol, 20))
    vol_sma = pd.Series(volume).rolling(20, min_periods=1).mean().values
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_vol = np.where(vol_sma > 0, volume / vol_sma, 0.0)

    # Pre-compute ATR
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).rolling(atr_len, min_periods=1).mean().values

    # Pivot detection (ta.pivothigh / ta.pivotlow with lookback=lookforward=pivot_len)
    # A pivot high at bar i means high[i] is the max in [i-pivot_len, i+pivot_len]
    # It's confirmed pivot_len bars later
    pivot_hi_vals = np.full(n, np.nan)
    pivot_lo_vals = np.full(n, np.nan)
    for i in range(pivot_len, n - pivot_len):
        is_pivot_hi = True
        is_pivot_lo = True
        for j in range(1, pivot_len + 1):
            if high[i] < high[i - j] or high[i] < high[i + j]:
                is_pivot_hi = False
            if low[i] > low[i - j] or low[i] > low[i + j]:
                is_pivot_lo = False
            if not is_pivot_hi and not is_pivot_lo:
                break
        if is_pivot_hi:
            pivot_hi_vals[i] = high[i]
        if is_pivot_lo:
            pivot_lo_vals[i] = low[i]

    # Market structure detection
    trend_out = np.zeros(n, dtype=int)
    break_type_out = np.zeros(n, dtype=int)  # 0=none, 1=BOS, 2=CHoCH
    break_bull_out = np.zeros(n, dtype=bool)
    svm_score_out = np.full(n, np.nan)

    last_pivot_hi = np.nan
    last_pivot_hi_idx = -1
    last_pivot_lo = np.nan
    last_pivot_lo_idx = -1
    ms_trend = 0

    for i in range(n):
        # Check for confirmed pivots (detected pivot_len bars ago)
        check_idx = i - pivot_len
        if check_idx >= 0:
            if not np.isnan(pivot_hi_vals[check_idx]):
                last_pivot_hi = pivot_hi_vals[check_idx]
                last_pivot_hi_idx = check_idx
            if not np.isnan(pivot_lo_vals[check_idx]):
                last_pivot_lo = pivot_lo_vals[check_idx]
                last_pivot_lo_idx = check_idx

        # Bullish break: close breaks above last pivot high
        if (not np.isnan(last_pivot_hi) and close[i] > last_pivot_hi
                and (i == 0 or high[i - 1] <= last_pivot_hi)):
            dist_feature = (close[i] - last_pivot_hi) / max(atr[i], 1e-10)
            raw_score = rel_vol[i] * vol_weight + rsi_feature[i] * rsi_weight + dist_feature * dist_weight
            score = _sigmoid(raw_score * 2 - 1)

            if ms_trend == -1:
                break_type_out[i] = 2  # CHoCH
                ms_trend = 1
            else:
                break_type_out[i] = 1  # BOS
                ms_trend = 1

            break_bull_out[i] = True
            svm_score_out[i] = score
            last_pivot_hi = np.nan  # Reset

        # Bearish break: close breaks below last pivot low
        elif (not np.isnan(last_pivot_lo) and close[i] < last_pivot_lo
                and (i == 0 or low[i - 1] >= last_pivot_lo)):
            dist_feature = (last_pivot_lo - close[i]) / max(atr[i], 1e-10)
            raw_score = rel_vol[i] * vol_weight + rsi_feature[i] * rsi_weight + dist_feature * dist_weight
            score = _sigmoid(raw_score * 2 - 1)

            if ms_trend == 1:
                break_type_out[i] = 2  # CHoCH
                ms_trend = -1
            else:
                break_type_out[i] = 1  # BOS
                ms_trend = -1

            break_bull_out[i] = False
            svm_score_out[i] = score
            last_pivot_lo = np.nan  # Reset

        trend_out[i] = ms_trend

    return {
        "trend": trend_out,
        "break_type": break_type_out,
        "break_bull": break_bull_out,
        "svm_score": svm_score_out,
        "pivot_hi": pivot_hi_vals,
        "pivot_lo": pivot_lo_vals,
    }
