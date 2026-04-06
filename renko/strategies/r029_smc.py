"""
R029: Smart Money Concepts (SMC) Structure Breaks

Entry on internal market structure breaks (BOS/CHoCH) with optional swing trend
alignment and PSAR gate.  Ported from LuxAlgo Smart Money Concepts Pine Script.

BOS  (Break of Structure)    = trend continuation -- price breaks previous swing
     high/low in the same trend direction.
CHoCH (Change of Character)  = trend reversal -- price breaks previous swing
     high/low against the current trend, signaling a structural shift.

Pivot detection mirrors Pine's leg() function: a bar is confirmed as a pivot
high when its high exceeds the highest high of the next `size` bars (and
symmetrically for pivot lows).  This introduces a natural `size`-bar lag --
no look-ahead.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "SMC internal structure BOS/CHoCH entries with swing trend + PSAR gates"

HYPOTHESIS = (
    "Market structure breaks identify trend continuation (BOS) and reversal (CHoCH) "
    "points. On Renko, structure is cleaner because noise is filtered by brick "
    "construction. CHoCH after trend exhaustion signals high-probability reversals; "
    "BOS confirms momentum. Swing trend alignment and PSAR filter low-quality entries."
)

PARAM_GRID = {
    "internal_size": [3, 5, 7],
    "swing_size": [25, 50],
    "entry_mode": ["choch", "bos", "both"],
    "swing_align": [True, False],
    "psar_gate": [True, False],
    "cooldown": [5, 10, 20, 30],
}


def generate_signals(
    df: pd.DataFrame,
    internal_size: int = 5,
    swing_size: int = 50,
    entry_mode: str = "choch",
    swing_align: bool = False,
    psar_gate: bool = False,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    Generate entry/exit signals based on Smart Money Concepts structure breaks.

    Internal structure (small pivot lookback) generates entry signals.
    Swing structure (large pivot lookback) provides optional trend alignment gate.

    Args:
        df: Renko DataFrame with OHLCV, brick_up, and pre-shifted indicator columns.
        internal_size: Lookback bars to confirm internal pivot highs/lows.
        swing_size: Lookback bars to confirm swing pivot highs/lows.
        entry_mode: "choch" (reversals only), "bos" (continuations), or "both".
        swing_align: If True, only enter when swing trend agrees with entry direction.
        psar_gate: If True, require PSAR direction to agree with entry direction.
        cooldown: Minimum bricks between entries.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit columns.
    """
    n = len(df)
    high_arr = df["High"].values
    low_arr = df["Low"].values
    close_arr = df["Close"].values
    brick_up = df["brick_up"].values
    psar_dir = df["psar_dir"].values if psar_gate else None

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    # Internal structure state  (mirrors Pine's internalHigh/Low + internalTrend)
    int_high_level = np.nan
    int_high_crossed = True
    int_low_level = np.nan
    int_low_crossed = True
    int_trend = 0   # +1 bullish, -1 bearish
    int_leg = 0     # 0=bearish leg, 1=bullish leg  (Pine: var leg = 0)

    # Swing structure state
    sw_high_level = np.nan
    sw_high_crossed = True
    sw_low_level = np.nan
    sw_low_crossed = True
    sw_trend = 0
    sw_leg = 0

    last_trade_bar = -999_999
    warmup = swing_size + 2

    for i in range(warmup, n):
        # ── Detect internal pivots ────────────────────────────────
        # Pine: high[size] > ta.highest(size) confirms pivot high at bar [i-size]
        bar_h = high_arr[i - internal_size]
        bar_l = low_arr[i - internal_size]
        win_h = np.max(high_arr[i - internal_size + 1 : i + 1])
        win_l = np.min(low_arr[i - internal_size + 1 : i + 1])

        prev_int_leg = int_leg
        if bar_h > win_h:
            int_leg = 0   # bearish leg (found pivot high)
        elif bar_l < win_l:
            int_leg = 1   # bullish leg (found pivot low)

        if int_leg != prev_int_leg:
            if int_leg == 1:   # bullish leg start -> new low pivot found
                int_low_level = bar_l
                int_low_crossed = False
            else:              # bearish leg start -> new high pivot found
                int_high_level = bar_h
                int_high_crossed = False

        # ── Detect swing pivots ───────────────────────────────────
        sw_bar_h = high_arr[i - swing_size]
        sw_bar_l = low_arr[i - swing_size]
        sw_win_h = np.max(high_arr[i - swing_size + 1 : i + 1])
        sw_win_l = np.min(low_arr[i - swing_size + 1 : i + 1])

        prev_sw_leg = sw_leg
        if sw_bar_h > sw_win_h:
            sw_leg = 0
        elif sw_bar_l < sw_win_l:
            sw_leg = 1

        if sw_leg != prev_sw_leg:
            if sw_leg == 1:
                sw_low_level = sw_bar_l
                sw_low_crossed = False
            else:
                sw_high_level = sw_bar_h
                sw_high_crossed = False

        # ── Internal structure breaks ─────────────────────────────
        int_bull_choch = False
        int_bull_bos = False
        int_bear_choch = False
        int_bear_bos = False

        # Bullish break: close crosses above internal high
        if (not np.isnan(int_high_level) and not int_high_crossed
                and close_arr[i] > int_high_level
                and close_arr[i - 1] <= int_high_level):
            int_high_crossed = True
            if int_trend <= 0:
                int_bull_choch = True
            else:
                int_bull_bos = True
            int_trend = 1

        # Bearish break: close crosses below internal low
        if (not np.isnan(int_low_level) and not int_low_crossed
                and close_arr[i] < int_low_level
                and close_arr[i - 1] >= int_low_level):
            int_low_crossed = True
            if int_trend >= 0:
                int_bear_choch = True
            else:
                int_bear_bos = True
            int_trend = -1

        # ── Swing structure breaks (for trend gate only) ──────────
        if (not np.isnan(sw_high_level) and not sw_high_crossed
                and close_arr[i] > sw_high_level
                and close_arr[i - 1] <= sw_high_level):
            sw_high_crossed = True
            sw_trend = 1

        if (not np.isnan(sw_low_level) and not sw_low_crossed
                and close_arr[i] < sw_low_level
                and close_arr[i - 1] >= sw_low_level):
            sw_low_crossed = True
            sw_trend = -1

        # ── Exit (unconditional, no cooldown) ─────────────────────
        long_exit[i] = not brick_up[i]
        short_exit[i] = brick_up[i]

        # ── Entry ─────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        bull_signal = False
        bear_signal = False

        if entry_mode == "choch":
            bull_signal = int_bull_choch
            bear_signal = int_bear_choch
        elif entry_mode == "bos":
            bull_signal = int_bull_bos
            bear_signal = int_bear_bos
        else:  # "both"
            bull_signal = int_bull_choch or int_bull_bos
            bear_signal = int_bear_choch or int_bear_bos

        # Swing alignment gate
        if swing_align:
            if bull_signal and sw_trend != 1:
                bull_signal = False
            if bear_signal and sw_trend != -1:
                bear_signal = False

        # PSAR gate
        if psar_gate and psar_dir is not None:
            if bull_signal and (np.isnan(psar_dir[i]) or psar_dir[i] != 1):
                bull_signal = False
            if bear_signal and (np.isnan(psar_dir[i]) or psar_dir[i] != -1):
                bear_signal = False

        if bull_signal:
            long_entry[i] = True
            last_trade_bar = i
        elif bear_signal:
            short_entry[i] = True
            last_trade_bar = i

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
