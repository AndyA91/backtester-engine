"""
Exit Signal Generators

All functions return dict with "long_exit" and "short_exit" boolean arrays.
Can be combined with any entry signal.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import calc_ema, calc_sma, calc_atr, detect_crossover, detect_crossunder


# ── 1. Opposite Signal Exit ─────────────────────────────────────────────────

def exit_opposite_signal(long_entry: np.ndarray, short_entry: np.ndarray) -> dict:
    """Use the opposite entry signal as exit — simplest reversal system."""
    return {
        "long_exit": short_entry.copy(),
        "short_exit": long_entry.copy(),
        "name": "Opposite Signal",
    }


# ── 2. ATR Trailing Stop ────────────────────────────────────────────────────

def exit_atr_trail(df: pd.DataFrame, atr_period: int = 14, atr_mult: float = 2.0) -> dict:
    """
    Chandelier-style ATR trailing stop. Tracks the highest high (for longs)
    or lowest low (for shorts) and exits when price retraces by N × ATR.

    Returns exit signals — must be applied AFTER entry tracking (the showdown
    runner handles position state).
    """
    atr = calc_atr(df, period=atr_period)["atr"]
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(close)

    # For long positions: trail from highest high
    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)
    trailing_high = np.full(n, np.nan)
    trailing_low = np.full(n, np.nan)

    # Compute chandelier levels for every bar
    for i in range(1, n):
        trailing_high[i] = max(high[i], trailing_high[i-1] if not np.isnan(trailing_high[i-1]) else high[i])
        trailing_low[i] = min(low[i], trailing_low[i-1] if not np.isnan(trailing_low[i-1]) else low[i])

        # Long exit: close drops below trailing high - ATR*mult
        if not np.isnan(atr[i]) and not np.isnan(trailing_high[i]):
            stop = trailing_high[i] - atr_mult * atr[i]
            if close[i] < stop:
                long_exit[i] = True
                trailing_high[i] = np.nan  # reset after exit

        # Short exit: close rises above trailing low + ATR*mult
        if not np.isnan(atr[i]) and not np.isnan(trailing_low[i]):
            stop = trailing_low[i] + atr_mult * atr[i]
            if close[i] > stop:
                short_exit[i] = True
                trailing_low[i] = np.nan

    return {"long_exit": long_exit, "short_exit": short_exit, "name": f"ATR Trail {atr_mult}x"}


# ── 3. N-Bar Time Exit ──────────────────────────────────────────────────────

def exit_n_bars(long_entry: np.ndarray, short_entry: np.ndarray, n_bars: int = 5) -> dict:
    """Exit N bars after entry — simple time-based exit."""
    n = len(long_entry)
    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    last_long_entry = -999
    last_short_entry = -999

    for i in range(n):
        if long_entry[i]:
            last_long_entry = i
        if short_entry[i]:
            last_short_entry = i

        if i - last_long_entry == n_bars:
            long_exit[i] = True
        if i - last_short_entry == n_bars:
            short_exit[i] = True

    return {"long_exit": long_exit, "short_exit": short_exit, "name": f"N-Bar Exit ({n_bars})"}


# ── 4. EMA Cross Exit ───────────────────────────────────────────────────────

def exit_ema_cross(df: pd.DataFrame, fast: int = 5, slow: int = 13) -> dict:
    """Fast EMA crosses slow EMA — independent exit timing from entry."""
    fast_ema = calc_ema(df["Close"], length=fast)
    slow_ema = calc_ema(df["Close"], length=slow)

    return {
        "long_exit": detect_crossunder(fast_ema, slow_ema),
        "short_exit": detect_crossover(fast_ema, slow_ema),
        "name": f"EMA Exit {fast}/{slow}",
    }


# ── 5. RSI Target Exit ──────────────────────────────────────────────────────

def exit_rsi_target(df: pd.DataFrame, period: int = 14, long_target: float = 60,
                    short_target: float = 40) -> dict:
    """Exit when RSI reaches a target level (not necessarily extreme)."""
    from indicators.rsi import calc_rsi
    rsi = calc_rsi(df, period=period)["rsi"]
    n = len(rsi)

    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # Long exit: RSI reaches upper target
        long_exit[i] = rsi[i] >= long_target and rsi[i-1] < long_target
        # Short exit: RSI drops to lower target
        short_exit[i] = rsi[i] <= short_target and rsi[i-1] > short_target

    return {"long_exit": long_exit, "short_exit": short_exit, "name": f"RSI Target {long_target}/{short_target}"}


# ── 6. Parabolic SAR Exit ───────────────────────────────────────────────────

def exit_psar(df: pd.DataFrame, start: float = 0.02, increment: float = 0.02,
              maximum: float = 0.2) -> dict:
    """Parabolic SAR flip as exit — accelerating trailing stop."""
    from indicators.parabolic_sar import calc_psar
    sar = calc_psar(df, start=start, increment=increment, maximum=maximum)
    direction = sar["direction"]
    n = len(direction)

    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    for i in range(1, n):
        long_exit[i] = direction[i] == -1 and direction[i-1] == 1
        short_exit[i] = direction[i] == 1 and direction[i-1] == -1

    return {"long_exit": long_exit, "short_exit": short_exit, "name": "PSAR Exit"}


# ── 7. BB Mid Reversion Exit ────────────────────────────────────────────────

def exit_bb_mid(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> dict:
    """Exit when price returns to Bollinger Band midline (SMA)."""
    from indicators.bbands import calc_bbands
    bb = calc_bbands(df, period=period, mult=mult)
    close = df["Close"].values
    mid = bb["mid"]
    n = len(close)

    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(mid[i]):
            continue
        # Long exit: price crosses above mid from below
        long_exit[i] = close[i] >= mid[i] and close[i-1] < mid[i]
        # Short exit: price crosses below mid from above
        short_exit[i] = close[i] <= mid[i] and close[i-1] > mid[i]

    return {"long_exit": long_exit, "short_exit": short_exit, "name": "BB Mid Exit"}


# ── 8. Supertrend Exit ──────────────────────────────────────────────────────

def exit_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> dict:
    """Supertrend flips direction — strong trend-following exit."""
    from indicators.supertrend import calc_supertrend
    st = calc_supertrend(df, period=period, multiplier=mult)
    direction = st["direction"]
    n = len(direction)

    long_exit = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    for i in range(1, n):
        long_exit[i] = direction[i] == -1 and direction[i-1] == 1
        short_exit[i] = direction[i] == 1 and direction[i-1] == -1

    return {"long_exit": long_exit, "short_exit": short_exit, "name": f"Supertrend Exit {period}/{mult}"}
