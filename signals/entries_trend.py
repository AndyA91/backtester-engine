"""
Trend-Following Entry Signals

All functions return dict with "long_entry" and "short_entry" boolean arrays.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import calc_ema, calc_sma, calc_hma, calc_wma, detect_crossover, detect_crossunder


# ── 1. EMA Crossover ────────────────────────────────────────────────────────

def sig_ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> dict:
    """Classic EMA crossover. Fast EMA crosses slow → entry."""
    fast_ema = calc_ema(df["Close"], length=fast)
    slow_ema = calc_ema(df["Close"], length=slow)
    return {
        "long_entry": detect_crossover(fast_ema, slow_ema),
        "short_entry": detect_crossunder(fast_ema, slow_ema),
        "name": f"EMA Cross {fast}/{slow}",
    }


# ── 2. Triple EMA (3-line) ──────────────────────────────────────────────────

def sig_triple_ema(df: pd.DataFrame, fast: int = 5, mid: int = 13, slow: int = 34) -> dict:
    """Three-EMA stack: enter when fast > mid > slow (or reverse)."""
    f = calc_ema(df["Close"], length=fast).values
    m = calc_ema(df["Close"], length=mid).values
    s = calc_ema(df["Close"], length=slow).values
    n = len(f)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        bull_now = f[i] > m[i] > s[i]
        bull_prev = f[i-1] > m[i-1] > s[i-1]
        bear_now = f[i] < m[i] < s[i]
        bear_prev = f[i-1] < m[i-1] < s[i-1]
        long_entry[i] = bull_now and not bull_prev
        short_entry[i] = bear_now and not bear_prev

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Triple EMA {fast}/{mid}/{slow}"}


# ── 3. HMA Direction Change ─────────────────────────────────────────────────

def sig_hma_turn(df: pd.DataFrame, length: int = 55) -> dict:
    """Hull MA direction flip. Low lag → catches trend changes early."""
    hma = calc_hma(df["Close"], length=length).values
    n = len(hma)
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(2, n):
        if np.isnan(hma[i]) or np.isnan(hma[i-1]) or np.isnan(hma[i-2]):
            continue
        # HMA turns up: was falling, now rising
        long_entry[i] = hma[i] > hma[i-1] and hma[i-1] <= hma[i-2]
        # HMA turns down: was rising, now falling
        short_entry[i] = hma[i] < hma[i-1] and hma[i-1] >= hma[i-2]

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"HMA Turn {length}"}


# ── 4. Supertrend Flip ──────────────────────────────────────────────────────

def sig_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> dict:
    """Supertrend direction flip — one of the cleanest trend signals."""
    from indicators.supertrend import calc_supertrend
    st = calc_supertrend(df, period=period, multiplier=mult)
    direction = st["direction"]
    n = len(direction)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        long_entry[i] = direction[i] == 1 and direction[i-1] == -1
        short_entry[i] = direction[i] == -1 and direction[i-1] == 1

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Supertrend {period}/{mult}"}


# ── 5. Donchian Breakout ────────────────────────────────────────────────────

def sig_donchian_breakout(df: pd.DataFrame, length: int = 20) -> dict:
    """Price breaks above/below Donchian channel — classic trend entry."""
    from engine import calc_donchian
    upper, lower, mid = calc_donchian(pd.Series(df["High"].values), pd.Series(df["Low"].values), length)
    close = df["Close"].values
    n = len(close)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(upper[i-1]) or np.isnan(lower[i-1]):
            continue
        # Close breaks above previous bar's upper channel
        long_entry[i] = close[i] > upper[i-1] and close[i-1] <= upper[i-1]
        # Close breaks below previous bar's lower channel
        short_entry[i] = close[i] < lower[i-1] and close[i-1] >= lower[i-1]

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Donchian Break {length}"}


# ── 6. Ichimoku Cloud Break ─────────────────────────────────────────────────

def sig_ichimoku_cloud(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict:
    """Price crosses above/below the Ichimoku cloud (Senkou Span A/B)."""
    from engine import calc_ichimoku
    ichi = calc_ichimoku(pd.Series(df["High"].values), pd.Series(df["Low"].values),
                         conversion_periods=tenkan, base_periods=kijun,
                         lagging_span2_periods=senkou_b)
    close = df["Close"].values
    span_a = ichi["lead_a"]
    span_b = ichi["lead_b"]
    n = len(close)

    cloud_top = np.maximum(span_a, span_b)
    cloud_bot = np.minimum(span_a, span_b)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(cloud_top[i]) or np.isnan(cloud_bot[i]):
            continue
        # Price breaks above cloud
        long_entry[i] = close[i] > cloud_top[i] and close[i-1] <= cloud_top[i]
        # Price breaks below cloud
        short_entry[i] = close[i] < cloud_bot[i] and close[i-1] >= cloud_bot[i]

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "Ichimoku Cloud Break"}


# ── 7. KAMA Slope ───────────────────────────────────────────────────────────

def sig_kama_slope(df: pd.DataFrame, period: int = 21, slope_len: int = 5, threshold: float = 0.0) -> dict:
    """KAMA slope turns positive/negative — adaptive trend detection."""
    from indicators.kama import calc_kama
    kama_series = calc_kama(df["Close"], length=period)
    kama = kama_series.values
    n = len(kama)

    slope = np.full(n, np.nan)
    for i in range(slope_len, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i - slope_len]):
            slope[i] = (kama[i] - kama[i - slope_len]) / kama[i - slope_len] * 100

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(slope[i]) or np.isnan(slope[i-1]):
            continue
        long_entry[i] = slope[i] > threshold and slope[i-1] <= threshold
        short_entry[i] = slope[i] < -threshold and slope[i-1] >= -threshold

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"KAMA Slope {period}"}


# ── 8. Price vs SMA (trend filter entry) ────────────────────────────────────

def sig_price_vs_sma(df: pd.DataFrame, length: int = 200) -> dict:
    """Price crosses above/below long-term SMA — simplest trend signal."""
    sma = calc_sma(df["Close"], length=length)
    return {
        "long_entry": detect_crossover(df["Close"], sma),
        "short_entry": detect_crossunder(df["Close"], sma),
        "name": f"Price vs SMA {length}",
    }
