import sys
from pathlib import Path

# Project root must be in path for dgtrd imports (runner.py adds ROOT later).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.dgtrd.distance_oscillator import distance_oscillator_sr
from indicators.dgtrd.oscillators import oscillators_overlay
from indicators.dgtrd.raff_regression import raff_regression_channel
from indicators.dgtrd.volume_profile import volume_profile_pivot_anchored
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD R007 + dgtrd gate sweep (raff/vp/div/do)"

HYPOTHESIS = (
    "Raff slope (trend), Volume Profile POC side (structure), MACD divergence (reversal confirm), "
    "and Distance Oscillator extremes (mean reversion) may filter low-quality R007 entries on EURAUD"
)

RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT = 0.009
INITIAL_CAPITAL = 1000.0

PARAM_GRID = {
    "n_bricks": [2, 3, 4, 5],
    "cooldown": [10, 20, 30],
    "session_start": [0, 7, 13],
    "raff_gate": [False, True],
    "vp_gate": [False, True],
    "div_gate": [False, True],
    "do_gate": [False, True],
}


def _build_indicator_cache() -> pd.DataFrame:
    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    df = raff_regression_channel(df, source_col="Close", length=50, raff_length=100)
    df["rrc_slope"] = df["rrc_slope"].shift(1)

    df = volume_profile_pivot_anchored(df, pvt_length=20, num_bins=25, va_pct=0.68)
    df["vp_above_poc"] = df["vp_above_poc"].shift(1)

    df = oscillators_overlay(df, osc_type="MACD", macd_fast=12, macd_slow=26, macd_signal=9, prefix="osc_")
    df["osc_bull_div"] = df["osc_bull_div"].shift(1)
    df["osc_bear_div"] = df["osc_bear_div"].shift(1)

    df = distance_oscillator_sr(df, ma_length=21, bb_length=233, bb_mult=2.5)
    df["do_overbought"] = df["do_overbought"].shift(1)
    df["do_oversold"] = df["do_oversold"].shift(1)

    return df


_CACHE = _build_indicator_cache()


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
    session_start: int = 0,
    raff_gate: bool = False,
    vp_gate: bool = False,
    div_gate: bool = False,
    do_gate: bool = False,
) -> pd.DataFrame:
    rrc_slope = _CACHE["rrc_slope"].reindex(df.index).values
    vp_above = _CACHE["vp_above_poc"].reindex(df.index).values
    bull_div = _CACHE["osc_bull_div"].reindex(df.index).values
    bear_div = _CACHE["osc_bear_div"].reindex(df.index).values
    do_oversold = _CACHE["do_oversold"].reindex(df.index).values
    do_overbought = _CACHE["do_overbought"].reindex(df.index).values

    brick_up = df["brick_up"].values
    entry_hours = df.index.hour

    n = len(df)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_r001_bar = -999_999
    if raff_gate or vp_gate or div_gate or do_gate:
        warmup = 160
    else:
        warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i] = True
                in_position = False
                trade_dir = 0
            elif trade_dir == -1 and up:
                short_exit[i] = True
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        if session_start > 0 and entry_hours[i] < session_start:
            continue

        def gates_ok(direction: int) -> bool:
            if raff_gate:
                s = rrc_slope[i]
                if not np.isnan(s):
                    if direction == 1 and s <= 0:
                        return False
                    if direction == -1 and s >= 0:
                        return False

            if vp_gate:
                v = vp_above[i]
                if not np.isnan(v):
                    if direction == 1 and not v:
                        return False
                    if direction == -1 and v:
                        return False

            if do_gate:
                if direction == 1 and not do_oversold[i]:
                    return False
                if direction == -1 and not do_overbought[i]:
                    return False

            return True

        window_r2 = brick_up[i - n_bricks : i]
        prev_all_up = len(window_r2) == n_bricks and bool(np.all(window_r2))
        prev_all_down = len(window_r2) == n_bricks and bool(not np.any(window_r2))

        if prev_all_down and up:
            if gates_ok(1):
                if div_gate and not bull_div[i]:
                    pass
                else:
                    long_entry[i] = True
                    in_position = True
                    trade_dir = 1

        elif prev_all_up and not up:
            if gates_ok(-1):
                if div_gate and not bear_div[i]:
                    pass
                else:
                    short_entry[i] = True
                    in_position = True
                    trade_dir = -1

        elif (i - last_r001_bar) >= cooldown:
            window_r1 = brick_up[i - n_bricks + 1 : i + 1]
            if len(window_r1) == n_bricks:
                if bool(np.all(window_r1)):
                    if gates_ok(1):
                        long_entry[i] = True
                        in_position = True
                        trade_dir = 1
                        last_r001_bar = i
                elif bool(not np.any(window_r1)):
                    if gates_ok(-1):
                        short_entry[i] = True
                        in_position = True
                        trade_dir = -1
                        last_r001_bar = i

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
