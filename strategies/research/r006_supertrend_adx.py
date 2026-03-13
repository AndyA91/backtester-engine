"""
R006: Supertrend + ADX Regime Filter

Best signal from R003 (Supertrend flip, htf_agree=True) combined with ADX(14)
quality gate. adx_threshold=0 reproduces the R003 baseline for direct comparison.

Rationale: R003 produced 600+ trades at PF 0.92 — plenty of signal, but no quality
filter. From v4 analysis on this same dataset, ADX(14) was the #1 per-trade PnL
predictor (Pearson r = +0.521). Every strategy that lacked ADX (R001, R003, R004,
R005) had PF < 1. R002, the only strategy with ADX, was the only one above PF 1.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from indicators.supertrend import calc_supertrend
from indicators.adx import calc_adx

DESCRIPTION = "Supertrend flip + 1H agreement, gated by ADX(14) regime filter (adx=0 → R003 baseline)"

HYPOTHESIS = (
    "R003 showed Supertrend flips produce 600+ trades but PF < 1 with no quality "
    "filter. ADX(14) was the #1 per-trade PnL predictor in v4 analysis (r=+0.521). "
    "Requiring ADX > threshold should strip out low-quality flips in choppy/ranging "
    "conditions while keeping trend-validated entries. adx_threshold=0 is the unfiltered "
    "baseline; comparing it to adx=[20,25,30] isolates ADX's exact contribution."
)

PARAM_GRID = {
    "atr_period":    [7, 10, 14],
    "multiplier":    [3.0, 4.0],
    "adx_threshold": [0, 20, 25, 30],   # 0 = disabled → direct R003 baseline
    "cooldown":      [12, 24],
}


def _align_1h_direction(ltf_index: pd.DatetimeIndex, direction_1h: np.ndarray, idx_1h: pd.DatetimeIndex) -> np.ndarray:
    """Align 1H Supertrend direction to 5m bars via merge_asof."""
    htf = pd.DataFrame({
        "Date":      idx_1h,
        "direction": pd.Series(direction_1h, index=idx_1h).shift(1).values,
    })
    ltf = pd.DataFrame({"Date": ltf_index})
    merged = pd.merge_asof(ltf.sort_values("Date"), htf.sort_values("Date"),
                           on="Date", direction="backward")
    return merged["direction"].values


def generate_signals(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
    atr_period: int = 7,
    multiplier: float = 4.0,
    adx_threshold: int = 25,
    cooldown: int = 12,
) -> pd.DataFrame:
    close = df_5m["Close"].values
    n = len(close)

    # 5m Supertrend — compute on raw OHLC, shift direction output (Pitfall #7:
    # shifting full DataFrame collapses ATR via RMA NaN propagation → zero trades)
    st_result = calc_supertrend(df_5m, period=atr_period, multiplier=multiplier)
    direction_5m = pd.Series(st_result["direction"]).shift(1).values

    # 1H Supertrend direction aligned to 5m (htf_agree always True — best in R003)
    st_1h = calc_supertrend(df_1h, period=atr_period, multiplier=multiplier)
    direction_1h_aligned = _align_1h_direction(df_5m.index, st_1h["direction"], df_1h.index)

    # ADX(14) — compute on raw OHLC, shift output by 1 bar (Pitfall #7)
    adx_result = calc_adx(df_5m, di_period=14, adx_period=14)
    adx = pd.Series(adx_result["adx"], index=df_5m.index).shift(1).values

    hours = df_5m.index.hour.values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(atr_period, 14) + 2

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(direction_5m[i]):
            continue

        in_session = 7 <= hours[i] < 22

        dir_now  = direction_5m[i]
        dir_prev = direction_5m[i - 1]

        flip_bull = dir_now == 1  and dir_prev == -1
        flip_bear = dir_now == -1 and dir_prev == 1

        # Exit on Supertrend flip — no ADX or session gate on exits
        long_exit[i]  = flip_bear
        short_exit[i] = flip_bull

        htf_bull = direction_1h_aligned[i] == 1
        htf_bear = direction_1h_aligned[i] == -1

        # ADX regime gate: 0 = disabled (reproduces R003 unfiltered baseline)
        strong_trend = (adx_threshold == 0) or (not np.isnan(adx[i]) and adx[i] > adx_threshold)

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade and in_session and strong_trend:
            if flip_bull and htf_bull:
                long_entry[i]  = True
                last_trade_bar = i
            elif flip_bear and htf_bear:
                short_entry[i] = True
                last_trade_bar = i

    df_5m["long_entry"]  = long_entry
    df_5m["long_exit"]   = long_exit
    df_5m["short_entry"] = short_entry
    df_5m["short_exit"]  = short_exit
    return df_5m
