"""
KNN Supertrend Horizon [LuxAlgo]

ML-enhanced Supertrend using K-Nearest Neighbors classification.
Uses RSI and ATR-normalized volatility as features, classifies
supertrend direction over a sliding window, then smooths with EMA
and applies a confidence buffer for hysteresis.

Pine source: indicators/LuxAlgo/KNN_Supertrend_Horizon__LuxAlgo_.pine

NOTE: Core ML engine does NOT use volume. Volume is only used in
Pine for rejection bubble sizing (not ported).

Outputs:
    ml_bullish    — bool, ML-smoothed trend direction
    ml_prob       — raw KNN probability (0-100, bull votes / total)
    smoothed_prob — EMA-smoothed probability (0-100)
    st_val        — underlying supertrend value (price level)
    st_dir        — supertrend raw direction: +1 bull / -1 bear

Usage:
    from indicators.luxalgo.knn_supertrend import calc_knn_supertrend
    result = calc_knn_supertrend(df)
"""

import numpy as np
import pandas as pd


def calc_knn_supertrend(
    df: pd.DataFrame,
    neighbors_k: int = 10,
    window_size: int = 500,
    atr_len: int = 10,
    factor: float = 3.0,
    smooth_source: bool = True,
    smooth_len_val: int = 10,
    ml_buffer: float = 5.0,
    prob_smooth_len: int = 20,
) -> dict:
    """
    Parameters
    ----------
    df              : DataFrame with 'High', 'Low', 'Close'
    neighbors_k     : Number of nearest neighbors for KNN
    window_size     : Historical search window for KNN
    atr_len         : ATR period for supertrend
    factor          : Supertrend multiplier
    smooth_source   : Whether to smooth price input with HMA
    smooth_len_val  : HMA smoothing length (if smooth_source=True)
    ml_buffer       : Confidence buffer % around 50 for hysteresis
    prob_smooth_len : EMA period for smoothing ML probability
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(close)

    # ── HMA (Hull Moving Average) for smoothed source ──
    if smooth_source and smooth_len_val > 1:
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = max(1, smooth_len_val // 2)
        sqrt_len = max(1, int(np.sqrt(smooth_len_val)))

        def _wma(arr, period):
            """Weighted Moving Average matching Pine's ta.wma."""
            out = np.full(len(arr), np.nan)
            weights = np.arange(1, period + 1, dtype=float)
            w_sum = weights.sum()
            for i in range(period - 1, len(arr)):
                out[i] = np.dot(arr[i - period + 1:i + 1], weights) / w_sum
            return out

        wma_half = _wma(close, half_len)
        wma_full = _wma(close, smooth_len_val)
        diff = 2.0 * wma_half - wma_full
        # Replace NaN in diff with close for warmup
        for i in range(len(diff)):
            if np.isnan(diff[i]):
                diff[i] = close[i]
        src = _wma(diff, sqrt_len)
        for i in range(len(src)):
            if np.isnan(src[i]):
                src[i] = close[i]
    else:
        src = close.copy()

    # ── Feature 1: RSI(14) of src ──
    rsi_period = 14
    delta_src = np.diff(src, prepend=src[0])
    gains = np.where(delta_src > 0, delta_src, 0.0)
    losses = np.where(delta_src < 0, -delta_src, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    if n > rsi_period:
        avg_gain[rsi_period] = np.mean(gains[1:rsi_period + 1])
        avg_loss[rsi_period] = np.mean(losses[1:rsi_period + 1])
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gains[i]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + losses[i]) / rsi_period

    f1 = np.full(n, 50.0)
    for i in range(rsi_period, n):
        if avg_loss[i] == 0:
            f1[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            f1[i] = 100.0 - 100.0 / (1.0 + rs)

    # ── Feature 2: ATR(14) / src * 100 (normalized volatility) ──
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    f2 = np.where(src > 0, (atr14 / src) * 100.0, 0.0)

    # ── Supertrend ──
    # ATR for supertrend (separate period)
    atr_st = pd.Series(tr).rolling(atr_len, min_periods=1).mean().values
    hl2 = (high + low) / 2.0

    st_val = np.zeros(n)
    st_dir = np.ones(n, dtype=int)  # +1 = bull (Pine st_dir < 0 means bull)

    upper_band = np.zeros(n)
    lower_band = np.zeros(n)

    for i in range(n):
        upper_band[i] = hl2[i] + factor * atr_st[i]
        lower_band[i] = hl2[i] - factor * atr_st[i]

        if i == 0:
            st_val[i] = upper_band[i]
            st_dir[i] = -1  # start bearish
            continue

        # Clamp bands (Pine supertrend logic)
        if lower_band[i] < lower_band[i - 1] and close[i - 1] > lower_band[i - 1]:
            lower_band[i] = lower_band[i - 1]
        if upper_band[i] > upper_band[i - 1] and close[i - 1] < upper_band[i - 1]:
            upper_band[i] = upper_band[i - 1]

        prev_dir = st_dir[i - 1]
        if prev_dir == -1:  # was bearish
            if close[i] > upper_band[i - 1]:
                st_dir[i] = 1  # flip to bull
                st_val[i] = lower_band[i]
            else:
                st_dir[i] = -1
                st_val[i] = upper_band[i]
        else:  # was bullish
            if close[i] < lower_band[i - 1]:
                st_dir[i] = -1  # flip to bear
                st_val[i] = upper_band[i]
            else:
                st_dir[i] = 1
                st_val[i] = lower_band[i]

    # Pine: targetTrend = st_dir < 0 ? 1 : -1
    # st_dir in Pine: -1 means uptrend. Our st_dir: +1 = bull
    # target_trend used for KNN labels
    target_trend = np.where(st_dir > 0, 1, -1)

    # ── KNN Classification ──
    ml_prob = np.full(n, 50.0)

    for i in range(window_size + 1, n):
        # Compute distances to all bars in [i-window_size, i-1]
        dists = np.sqrt(
            (f1[i] - f1[i - window_size:i]) ** 2 +
            (f2[i] - f2[i - window_size:i]) ** 2
        )

        # Find K-th smallest distance threshold
        k_actual = min(neighbors_k, len(dists))
        threshold = np.partition(dists, k_actual - 1)[k_actual - 1]

        # Vote among neighbors within threshold
        bull_votes = 0.0
        bear_votes = 0.0
        for j in range(len(dists)):
            if dists[j] <= threshold:
                # target_trend at bar (i - window_size + j), shifted by 1
                label_idx = i - window_size + j + 1
                if label_idx < n:
                    if target_trend[label_idx] > 0:
                        bull_votes += 1.0
                    else:
                        bear_votes += 1.0

        total = bull_votes + bear_votes
        if total > 0:
            ml_prob[i] = (bull_votes / total) * 100.0

    # ── EMA smoothing of probability ──
    smoothed_prob = np.full(n, 50.0)
    alpha = 2.0 / (prob_smooth_len + 1)
    smoothed_prob[0] = ml_prob[0]
    for i in range(1, n):
        smoothed_prob[i] = alpha * ml_prob[i] + (1 - alpha) * smoothed_prob[i - 1]

    # ── Hysteresis (confidence buffer) ──
    ml_bullish = np.zeros(n, dtype=bool)
    is_bull = False
    for i in range(n):
        if smoothed_prob[i] > 50 + ml_buffer:
            is_bull = True
        elif smoothed_prob[i] < 50 - ml_buffer:
            is_bull = False
        ml_bullish[i] = is_bull

    return {
        "ml_bullish": ml_bullish,
        "ml_prob": ml_prob,
        "smoothed_prob": smoothed_prob,
        "st_val": st_val,
        "st_dir": st_dir,
    }
