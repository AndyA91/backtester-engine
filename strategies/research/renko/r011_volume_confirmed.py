"""
R011-Renko: Volume-Confirmed Donchian Breakout

R001 Donchian breakout (the sweep winner) with volume confirmation.
Tests whether CMF, MFI, or OBV slope improves signal quality.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.cmf import calc_cmf
from indicators.mfi import calc_mfi
from indicators.obv import calc_obv

DESCRIPTION = "Donchian breakout + volume confirmation (CMF/MFI/OBV)"

HYPOTHESIS = (
    "R001 Donchian was the clear winner at PF 3.8-8.1. Volume confirmation "
    "should filter false breakouts — real breakouts have volume behind them. "
    "Testing three volume indicators: CMF (flow direction), MFI (money flow "
    "oscillator), and OBV slope (cumulative volume trend)."
)

PARAM_GRID = {
    "n_entry":       [20, 40],
    "n_exit":        [5, 10],
    "vol_indicator": ["cmf", "mfi", "obv"],
    "vol_period":    [14, 20],
    "cooldown":      [6, 12],
}


def generate_signals(df, n_entry=40, n_exit=5, vol_indicator="cmf",
                     vol_period=14, cooldown=6):
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(close)

    # Check if Volume column exists and has data
    has_volume = "Volume" in df.columns and df["Volume"].sum() > 0

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    don_high_entry = high_s.shift(1).rolling(n_entry).max().values
    don_low_entry = low_s.shift(1).rolling(n_entry).min().values
    don_high_exit = high_s.shift(1).rolling(n_exit).max().values
    don_low_exit = low_s.shift(1).rolling(n_exit).min().values

    # Volume indicator
    vol_bull = np.ones(n, dtype=bool)  # default: no filter
    vol_bear = np.ones(n, dtype=bool)

    if has_volume:
        if vol_indicator == "cmf":
            cmf = calc_cmf(df, period=vol_period)["cmf"]
            cmf_s = np.empty(n)
            cmf_s[0] = np.nan
            cmf_s[1:] = cmf[:-1]
            for i in range(n):
                if not np.isnan(cmf_s[i]):
                    vol_bull[i] = cmf_s[i] > 0
                    vol_bear[i] = cmf_s[i] < 0

        elif vol_indicator == "mfi":
            mfi = calc_mfi(df, period=vol_period)["mfi"]
            mfi_s = np.empty(n)
            mfi_s[0] = np.nan
            mfi_s[1:] = mfi[:-1]
            for i in range(n):
                if not np.isnan(mfi_s[i]):
                    vol_bull[i] = mfi_s[i] > 50
                    vol_bear[i] = mfi_s[i] < 50

        elif vol_indicator == "obv":
            obv_result = calc_obv(df, ema_period=vol_period)
            obv_ema = obv_result["obv_ema"]
            obv_raw = obv_result["obv"]
            for i in range(1, n):
                if not np.isnan(obv_raw[i - 1]) and not np.isnan(obv_ema[i - 1]):
                    vol_bull[i] = obv_raw[i - 1] > obv_ema[i - 1]
                    vol_bear[i] = obv_raw[i - 1] < obv_ema[i - 1]

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999

    for i in range(max(n_entry, n_exit, vol_period) + 2, n):
        if np.isnan(close[i]) or np.isnan(don_high_entry[i]):
            continue

        brk_up = close[i] > don_high_entry[i]
        brk_dn = close[i] < don_low_entry[i]

        lx[i] = close[i] < don_low_exit[i]
        sx[i] = close[i] > don_high_exit[i]

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if brk_up and vol_bull[i]:
                le[i] = True
                last_trade_bar = i
            elif brk_dn and vol_bear[i]:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
