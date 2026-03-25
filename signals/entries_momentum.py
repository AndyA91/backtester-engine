"""
Momentum Entry Signals

All functions return dict with "long_entry" and "short_entry" boolean arrays.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import detect_crossover, detect_crossunder


# ── 1. MACD Cross ───────────────────────────────────────────────────────────

def sig_macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line crosses signal line — classic momentum entry."""
    from indicators.macd import calc_macd
    m = calc_macd(df, fast=fast, slow=slow, signal=signal)
    macd = pd.Series(m["macd"])
    sig = pd.Series(m["signal"])

    return {
        "long_entry": detect_crossover(macd, sig),
        "short_entry": detect_crossunder(macd, sig),
        "name": f"MACD Cross {fast}/{slow}/{signal}",
    }


# ── 2. MACD Histogram Flip ──────────────────────────────────────────────────

def sig_macd_hist_flip(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD histogram changes sign — earlier than crossover signal."""
    from indicators.macd import calc_macd
    m = calc_macd(df, fast=fast, slow=slow, signal=signal)
    hist = m["histogram"]
    n = len(hist)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        long_entry[i] = hist[i] > 0 and hist[i-1] <= 0
        short_entry[i] = hist[i] < 0 and hist[i-1] >= 0

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "MACD Hist Flip"}


# ── 3. VWMACD Cross ─────────────────────────────────────────────────────────

def sig_vwmacd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Volume-Weighted MACD cross — filters low-volume false signals."""
    from indicators.vwmacd import calc_vwmacd
    m = calc_vwmacd(df, fast=fast, slow=slow, signal=signal)
    vwmacd = pd.Series(m["vwmacd"])
    sig = pd.Series(m["signal"])

    return {
        "long_entry": detect_crossover(vwmacd, sig),
        "short_entry": detect_crossunder(vwmacd, sig),
        "name": "VWMACD Cross",
    }


# ── 4. Awesome Oscillator Zero Cross ────────────────────────────────────────

def sig_ao_zero(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> dict:
    """AO crosses zero line — momentum shift."""
    from indicators.awesome_oscillator import calc_ao
    ao = calc_ao(df, fast=fast, slow=slow)["ao"]
    n = len(ao)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(ao[i]) or np.isnan(ao[i-1]):
            continue
        long_entry[i] = ao[i] > 0 and ao[i-1] <= 0
        short_entry[i] = ao[i] < 0 and ao[i-1] >= 0

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "AO Zero Cross"}


# ── 5. Awesome Oscillator Saucer ────────────────────────────────────────────

def sig_ao_saucer(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> dict:
    """AO saucer pattern — 2 bars same color after opposite = continuation."""
    from indicators.awesome_oscillator import calc_ao
    result = calc_ao(df, fast=fast, slow=slow)
    ao = result["ao"]
    n = len(ao)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(3, n):
        if any(np.isnan(ao[i-j]) for j in range(4)):
            continue
        # Bullish saucer: AO > 0, bar[i-2] red (falling), bar[i-1] & bar[i] green (rising)
        if ao[i] > 0:
            if ao[i-2] > ao[i-3] and ao[i-1] < ao[i-2] and ao[i] > ao[i-1]:
                long_entry[i] = True
        # Bearish saucer: AO < 0, bar[i-2] green, bar[i-1] & bar[i] red
        if ao[i] < 0:
            if ao[i-2] < ao[i-3] and ao[i-1] > ao[i-2] and ao[i] < ao[i-1]:
                short_entry[i] = True

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "AO Saucer"}


# ── 6. RVI Cross ────────────────────────────────────────────────────────────

def sig_rvi_cross(df: pd.DataFrame, period: int = 10) -> dict:
    """Relative Vigor Index crosses signal — conviction-based momentum."""
    from indicators.rvi import calc_rvi
    r = calc_rvi(df, period=period)
    rvi = r["rvi"]
    signal = r["signal"]
    n = len(rvi)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(rvi[i]) or np.isnan(signal[i]) or np.isnan(rvi[i-1]) or np.isnan(signal[i-1]):
            continue
        long_entry[i] = rvi[i] > signal[i] and rvi[i-1] <= signal[i-1]
        short_entry[i] = rvi[i] < signal[i] and rvi[i-1] >= signal[i-1]

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"RVI Cross {period}"}


# ── 7. ADX Breakout ─────────────────────────────────────────────────────────

def sig_adx_breakout(df: pd.DataFrame, period: int = 14, threshold: float = 25) -> dict:
    """ADX rises above threshold with +DI/-DI giving direction."""
    from indicators.adx import calc_adx
    adx_result = calc_adx(df, di_period=period, adx_period=period)
    adx = adx_result["adx"]
    plus_di = adx_result["plus_di"]
    minus_di = adx_result["minus_di"]
    n = len(adx)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(adx[i]) or np.isnan(adx[i-1]):
            continue
        # ADX crosses above threshold
        adx_breakout = adx[i] > threshold and adx[i-1] <= threshold
        if adx_breakout:
            if plus_di[i] > minus_di[i]:
                long_entry[i] = True
            else:
                short_entry[i] = True

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"ADX Breakout {period}/{threshold}"}


# ── 8. Squeeze Momentum Fire ────────────────────────────────────────────────

def sig_squeeze_fire(df: pd.DataFrame) -> dict:
    """Squeeze releases → trade in direction of momentum."""
    from indicators.squeeze import calc_squeeze
    sq = calc_squeeze(df)
    squeeze_off = sq["squeeze_off"]
    momentum = sq["momentum"]
    n = len(momentum)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(n):
        if squeeze_off[i]:
            if not np.isnan(momentum[i]) and momentum[i] > 0:
                long_entry[i] = True
            elif not np.isnan(momentum[i]) and momentum[i] < 0:
                short_entry[i] = True

    return {"long_entry": long_entry, "short_entry": short_entry, "name": "Squeeze Fire"}
