"""
Mean-Reversion Entry Signals

All functions return dict with "long_entry" and "short_entry" boolean arrays.
Designed for fading extremes — best paired with an ADX < 25 regime filter.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── 1. RSI Extreme ──────────────────────────────────────────────────────────

def sig_rsi_extreme(df: pd.DataFrame, period: int = 14, ob: float = 70, os_: float = 30) -> dict:
    """RSI crosses back from overbought/oversold — classic mean-rev signal."""
    from indicators.rsi import calc_rsi
    rsi = calc_rsi(df, period=period)["rsi"]
    n = len(rsi)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # Was oversold, now crossing back above → buy
        long_entry[i] = rsi[i] > os_ and rsi[i-1] <= os_
        # Was overbought, now crossing back below → sell
        short_entry[i] = rsi[i] < ob and rsi[i-1] >= ob

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"RSI Extreme {period} ({os_}/{ob})"}


# ── 2. Connors RSI Extreme ──────────────────────────────────────────────────

def sig_crsi_extreme(df: pd.DataFrame, rsi_period: int = 3, streak: int = 2,
                     rank: int = 100, ob: float = 90, os_: float = 10) -> dict:
    """Connors RSI at extremes — high win-rate mean-reversion."""
    from indicators.connors_rsi import calc_connors_rsi
    crsi = calc_connors_rsi(df, rsi_period=rsi_period, streak_period=streak,
                            rank_period=rank)["crsi"]
    n = len(crsi)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(crsi[i]) or np.isnan(crsi[i-1]):
            continue
        long_entry[i] = crsi[i] > os_ and crsi[i-1] <= os_
        short_entry[i] = crsi[i] < ob and crsi[i-1] >= ob

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"CRSI Extreme {os_}/{ob}"}


# ── 3. Bollinger Band Bounce ────────────────────────────────────────────────

def sig_bb_bounce(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> dict:
    """Price touches lower BB then bounces back inside → long (and reverse)."""
    from indicators.bbands import calc_bbands
    bb = calc_bbands(df, period=period, mult=mult)
    close = df["Close"].values
    upper = bb["upper"]
    lower = bb["lower"]
    n = len(close)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        # Was below lower band, now back inside → long
        long_entry[i] = close[i] > lower[i] and close[i-1] <= lower[i-1]
        # Was above upper band, now back inside → short
        short_entry[i] = close[i] < upper[i] and close[i-1] >= upper[i-1]

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"BB Bounce {period}/{mult}"}


# ── 4. Keltner Touch ────────────────────────────────────────────────────────

def sig_keltner_touch(df: pd.DataFrame, period: int = 20, mult: float = 1.5) -> dict:
    """Price reverses from Keltner channel boundary."""
    from indicators.keltner import calc_keltner
    kc = calc_keltner(df, period=period, mult=mult)
    close = df["Close"].values
    n = len(close)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(kc["upper"][i]) or np.isnan(kc["lower"][i]):
            continue
        long_entry[i] = close[i] > kc["lower"][i] and close[i-1] <= kc["lower"][i-1]
        short_entry[i] = close[i] < kc["upper"][i] and close[i-1] >= kc["upper"][i-1]

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Keltner Touch {period}/{mult}"}


# ── 5. Stochastic Oversold/Overbought ───────────────────────────────────────

def sig_stoch_extreme(df: pd.DataFrame, k: int = 14, smooth_k: int = 3,
                      smooth_d: int = 3, ob: float = 80, os_: float = 20) -> dict:
    """Stochastic %K crosses %D in oversold/overbought zone."""
    from indicators.stochastic import calc_stochastic
    stoch = calc_stochastic(df, k_period=k, smooth_k=smooth_k, smooth_d=smooth_d)
    sk = stoch["slow_k"]
    sd = stoch["pct_d"]
    n = len(sk)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(sk[i]) or np.isnan(sd[i]) or np.isnan(sk[i-1]) or np.isnan(sd[i-1]):
            continue
        # %K crosses above %D while in oversold zone
        long_entry[i] = sk[i] > sd[i] and sk[i-1] <= sd[i-1] and sk[i] < os_ + 10
        # %K crosses below %D while in overbought zone
        short_entry[i] = sk[i] < sd[i] and sk[i-1] >= sd[i-1] and sk[i] > ob - 10

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Stoch Extreme {k} ({os_}/{ob})"}


# ── 6. Williams %R Extreme ──────────────────────────────────────────────────

def sig_williams_r(df: pd.DataFrame, period: int = 14, ob: float = -20, os_: float = -80) -> dict:
    """Williams %R crosses back from extreme."""
    from indicators.williams_r import calc_williams_r
    wpr = calc_williams_r(df, period=period)["wpr"]
    n = len(wpr)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(wpr[i]) or np.isnan(wpr[i-1]):
            continue
        long_entry[i] = wpr[i] > os_ and wpr[i-1] <= os_
        short_entry[i] = wpr[i] < ob and wpr[i-1] >= ob

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Williams %R {period}"}


# ── 7. CCI Extreme ──────────────────────────────────────────────────────────

def sig_cci_extreme(df: pd.DataFrame, period: int = 20, ob: float = 100, os_: float = -100) -> dict:
    """CCI crosses back from extreme — measures deviation from statistical mean."""
    from indicators.cci import calc_cci
    cci = calc_cci(df, period=period)["cci"]
    n = len(cci)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(cci[i]) or np.isnan(cci[i-1]):
            continue
        long_entry[i] = cci[i] > os_ and cci[i-1] <= os_
        short_entry[i] = cci[i] < ob and cci[i-1] >= ob

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"CCI Extreme {period}"}


# ── 8. Fisher Transform Cross ───────────────────────────────────────────────

def sig_fisher_cross(df: pd.DataFrame, period: int = 10) -> dict:
    """Fisher Transform crosses signal line — sharp reversal detection."""
    from indicators.fisher_transform import calc_fisher_transform
    ft = calc_fisher_transform(df, period=period)
    fisher = ft["fisher"]
    signal = ft["signal"]
    n = len(fisher)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(fisher[i]) or np.isnan(signal[i]):
            continue
        long_entry[i] = fisher[i] > signal[i] and fisher[i-1] <= signal[i-1]
        short_entry[i] = fisher[i] < signal[i] and fisher[i-1] >= signal[i-1]

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"Fisher Cross {period}"}


# ── 9. MFI Extreme ──────────────────────────────────────────────────────────

def sig_mfi_extreme(df: pd.DataFrame, period: int = 14, ob: float = 80, os_: float = 20) -> dict:
    """Money Flow Index — volume-weighted RSI variant at extremes."""
    from indicators.mfi import calc_mfi
    mfi = calc_mfi(df, period=period)["mfi"]
    n = len(mfi)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(mfi[i]) or np.isnan(mfi[i-1]):
            continue
        long_entry[i] = mfi[i] > os_ and mfi[i-1] <= os_
        short_entry[i] = mfi[i] < ob and mfi[i-1] >= ob

    return {"long_entry": long_entry, "short_entry": short_entry, "name": f"MFI Extreme {period}"}
