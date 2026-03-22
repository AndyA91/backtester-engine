"""EA006: Distance Divergence — DO Extreme + Oscillator Divergence Reversal

A reversal strategy that requires three concurrent conditions at entry:
  1. R002-style trigger: N consecutive bricks in one direction → opposing brick fires
  2. Distance Oscillator extreme: price is extended beyond its BB threshold (overbought/oversold)
  3. Oscillator divergence confirmation: MACD divergence AND/OR RSI divergence
     in the direction of the reversal

Signal logic:
  LONG:  N consecutive DOWN bricks → first UP brick
         + DO oversold (do_oversold=True)
         + MACD bull div OR RSI bull div (or BOTH if require_both_divs=True)
  SHORT: N consecutive UP bricks → first DOWN brick
         + DO overbought (do_overbought=True)
         + MACD bear div OR RSI bear div (or BOTH if require_both_divs=True)

Exit: first opposing Renko brick (standard).

Hypothesis: the R002 trigger detects momentum exhaustion at the bar-structure
level. Adding DO extreme confirms that price is genuinely overextended vs its
mean. Divergence (osc making higher lows while price makes lower lows, etc.)
confirms that selling/buying pressure is fading — the three-factor requirement
should yield high-conviction reversals with low false positive rate.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.dgtrd.distance_oscillator import distance_oscillator_sr
from indicators.dgtrd.oscillators import oscillators_overlay
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD DO Extreme + MACD/RSI Divergence reversal (R002 trigger)"

HYPOTHESIS = (
    "DO extreme (price outside BB envelope on the % deviation oscillator) identifies "
    "overextended runs. MACD and RSI divergences confirm fading momentum. Combined "
    "with the R002 trigger (first opposing brick after N same-direction bricks), "
    "the three-factor gate targets only the highest-conviction reversals."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# n_bricks:         consecutive same-direction bricks required before reversal trigger
# require_both_divs: True = need BOTH MACD AND RSI div; False = either one suffices
# cooldown:         minimum bricks between entries (0 = position flag only)
# session_start:    UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "n_bricks":          [2, 3, 4, 5],
    "require_both_divs": [False, True],
    "cooldown":          [0, 10, 20],
    "session_start":     [0, 13],
}


# ---------------------------------------------------------------------------
# Indicator cache  (built once at import time)
# ---------------------------------------------------------------------------

def _build_indicator_cache() -> pd.DataFrame:
    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    # ── Distance Oscillator S&R ─────────────────────────────────────────────
    # bb_length=233 is the longest warmup component (~300 bricks to stabilise)
    df = distance_oscillator_sr(df, ma_length=21, bb_length=233, bb_mult=2.5)
    df["do_overbought"] = df["do_overbought"].shift(1)
    df["do_oversold"]   = df["do_oversold"].shift(1)

    # ── MACD Divergence ────────────────────────────────────────────────────
    df = oscillators_overlay(
        df, osc_type="MACD",
        macd_fast=12, macd_slow=26, macd_signal=9,
        prefix="macd_",
    )
    df["macd_bull_div"] = df["macd_bull_div"].shift(1)
    df["macd_bear_div"] = df["macd_bear_div"].shift(1)

    # ── RSI Divergence ─────────────────────────────────────────────────────
    df = oscillators_overlay(
        df, osc_type="RSI",
        rsi_length=14,
        prefix="rsi_",
    )
    df["rsi_bull_div"] = df["rsi_bull_div"].shift(1)
    df["rsi_bear_div"] = df["rsi_bear_div"].shift(1)

    return df


_CACHE = _build_indicator_cache()

# bb_length=233 dominates warmup; 300 for safety
_WARMUP = 300


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    n_bricks:          int  = 3,
    require_both_divs: bool = False,
    cooldown:          int  = 0,
    session_start:     int  = 0,
) -> pd.DataFrame:
    """
    DO Extreme + divergence-confirmed R002 reversal.

    Three conditions must all be true at entry:
      1. R002 trigger: N consecutive bricks in one direction → first opposing brick
      2. Distance Oscillator extreme in the direction of the reversal
      3. MACD div OR RSI div (or BOTH if require_both_divs=True)

    Args:
        df:                Renko DataFrame with brick_up bool + OHLCV.
        n_bricks:          Consecutive bricks required before reversal trigger.
        require_both_divs: If True, need MACD div AND RSI div simultaneously.
        cooldown:          Minimum bars between entries (0 = position-flag gate only).
        session_start:     UTC hour gate (0 = disabled).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    c = _CACHE.reindex(df.index)

    do_oversold   = c["do_oversold"].fillna(False).values.astype(bool)
    do_overbought = c["do_overbought"].fillna(False).values.astype(bool)
    macd_bull_div = c["macd_bull_div"].fillna(False).values.astype(bool)
    macd_bear_div = c["macd_bear_div"].fillna(False).values.astype(bool)
    rsi_bull_div  = c["rsi_bull_div"].fillna(False).values.astype(bool)
    rsi_bear_div  = c["rsi_bear_div"].fillna(False).values.astype(bool)

    brick_up = df["brick_up"].values
    hours    = df.index.hour
    n        = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    in_position    = False
    trade_dir      = 0
    last_trade_bar = -999_999

    warmup = max(_WARMUP, n_bricks + 1)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────
        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i]  = True
                in_position   = False
                trade_dir     = 0
            elif trade_dir == -1 and up:
                short_exit[i] = True
                in_position   = False
                trade_dir     = 0

        if in_position:
            continue

        # ── Session gate ───────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        # ── Cooldown gate ──────────────────────────────────────────────────
        if cooldown > 0 and (i - last_trade_bar) < cooldown:
            continue

        # ── R002 trigger: N consecutive bricks → opposing brick ────────────
        window = brick_up[i - n_bricks : i]
        if len(window) < n_bricks:
            continue

        prev_all_down = bool(not np.any(window))
        prev_all_up   = bool(np.all(window))

        # ── LONG entry ─────────────────────────────────────────────────────
        if prev_all_down and up:
            # DO oversold check
            if not do_oversold[i]:
                continue

            # Divergence check
            if require_both_divs:
                div_ok = bool(macd_bull_div[i]) and bool(rsi_bull_div[i])
            else:
                div_ok = bool(macd_bull_div[i]) or bool(rsi_bull_div[i])

            if div_ok:
                long_entry[i]  = True
                in_position    = True
                trade_dir      = 1
                last_trade_bar = i

        # ── SHORT entry ────────────────────────────────────────────────────
        elif prev_all_up and not up:
            # DO overbought check
            if not do_overbought[i]:
                continue

            # Divergence check
            if require_both_divs:
                div_ok = bool(macd_bear_div[i]) and bool(rsi_bear_div[i])
            else:
                div_ok = bool(macd_bear_div[i]) or bool(rsi_bear_div[i])

            if div_ok:
                short_entry[i] = True
                in_position    = True
                trade_dir      = -1
                last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
