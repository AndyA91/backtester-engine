"""
Volume-Based Entry Signals

All functions return dict with "long_entry" and "short_entry" boolean arrays.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import detect_crossover, detect_crossunder, calc_sma


# ── 1. OBV EMA Cross ────────────────────────────────────────────────────────

def sig_obv_ema(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> dict:
    """OBV with EMA cross — volume flow confirms momentum."""
    from engine import calc_obv, calc_ema
    obv = calc_obv(df["Close"], df["Volume"])
    obv_fast = calc_ema(obv, length=fast)
    obv_slow = calc_ema(obv, length=slow)

    return {
        "long_entry": detect_crossover(obv_fast, obv_slow),
        "short_entry": detect_crossunder(obv_fast, obv_slow),
        "name": f"OBV EMA {fast}/{slow}",
    }


# ── 2. A/D Line Cross ───────────────────────────────────────────────────────

def sig_ad_cross(df: pd.DataFrame, ema_period: int = 21) -> dict:
    """A/D Line crosses its EMA — accumulation/distribution shift."""
    from indicators.ad_line import calc_ad_line
    ad = calc_ad_line(df, ema_period=ema_period)
    ad_line = pd.Series(ad["ad"])
    ad_ema = pd.Series(ad["ad_ema"])

    return {
        "long_entry": detect_crossover(ad_line, ad_ema),
        "short_entry": detect_crossunder(ad_line, ad_ema),
        "name": f"A/D Cross {ema_period}",
    }


# ── 3. CMF Sign Change ──────────────────────────────────────────────────────

def sig_cmf_flip(df: pd.DataFrame, period: int = 20) -> dict:
    """Chaikin Money Flow crosses zero — money flow direction change."""
    from indicators.cmf import calc_cmf
    cmf = calc_cmf(df, period=period)["cmf"]
    n = len(cmf)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(cmf[i]) or np.isnan(cmf[i-1]):
            continue
        long_entry[i] = cmf[i] > 0 and cmf[i-1] <= 0
        short_entry[i] = cmf[i] < 0 and cmf[i-1] >= 0

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"CMF Flip {period}"}


# ── 4. Volume Spike + Direction ──────────────────────────────────────────────

def sig_volume_spike(df: pd.DataFrame, vol_mult: float = 2.0, vol_period: int = 20) -> dict:
    """Volume spike (> N× average) with directional close as entry."""
    close = df["Close"].values
    volume = df["Volume"].values
    n = len(close)

    vol_sma = calc_sma(pd.Series(volume), length=vol_period).values

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        if volume[i] > vol_mult * vol_sma[i]:
            if close[i] > close[i-1]:
                long_entry[i] = True
            elif close[i] < close[i-1]:
                short_entry[i] = True

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Vol Spike {vol_mult}x"}
