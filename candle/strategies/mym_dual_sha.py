"""
MYM Dual Smoothed Heikin Ashi — 1min Candle Strategy

Two smoothed HA layers: slow SHA sets trend direction, fast SHA times entries/exits.
Entry: fast SHA flips bullish while slow is bullish (long), or bearish while slow is bearish (short).
Exit: fast SHA flips against the position.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Dual Smoothed Heikin Ashi (fast/slow) on 1min candles"

HYPOTHESIS = (
    "Smoothed HA removes 1-min noise. Slow SHA filters trend direction, "
    "fast SHA times entries. Cooldown prevents whipsaw in consolidation zones."
)

PARAM_GRID = {
    "fast_len": [3, 4, 5, 6, 8, 10, 12],
    "slow_len": [14, 18, 22, 25, 30, 40, 50],
    "cooldown": [0, 5, 10, 15, 20, 30],
}


def _ema(src: np.ndarray, length: int) -> np.ndarray:
    """Vectorized EMA matching Pine's ta.ema()."""
    out = np.empty_like(src, dtype=float)
    out[0] = src[0]
    k = 2.0 / (length + 1)
    for i in range(1, len(src)):
        out[i] = src[i] * k + out[i - 1] * (1 - k)
    return out


def _smoothed_ha(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                 length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smoothed Heikin Ashi.

    Returns:
        (ha_open, ha_close, is_bull) arrays
    """
    s_o = _ema(o, length)
    s_h = _ema(h, length)
    s_l = _ema(l, length)
    s_c = _ema(c, length)

    n = len(o)
    ha_open = np.empty(n, dtype=float)
    ha_close = (s_o + s_h + s_l + s_c) / 4.0

    # First bar
    ha_open[0] = (s_o[0] + s_c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    is_bull = ha_close >= ha_open
    return ha_open, ha_close, is_bull


def generate_signals(
    df: pd.DataFrame,
    fast_len: int = 6,
    slow_len: int = 14,
    cooldown: int = 10,
) -> pd.DataFrame:
    """
    Generate dual SHA entry/exit signals.

    Args:
        df: DataFrame with Open, High, Low, Close columns.
        fast_len: EMA smoothing length for fast SHA.
        slow_len: EMA smoothing length for slow SHA.
        cooldown: Minimum bars between exit and next entry.

    Returns:
        df with long_entry, long_exit, short_entry, short_exit columns.
    """
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)

    _, _, f_bull = _smoothed_ha(o, h, l, c, fast_len)
    _, _, s_bull = _smoothed_ha(o, h, l, c, slow_len)

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    warmup = max(fast_len, slow_len) + 1
    last_exit_bar = -999_999
    pos = 0  # +1 = long, -1 = short, 0 = flat

    for i in range(warmup, n):
        fast_bull_flip = f_bull[i] and not f_bull[i - 1]
        fast_bear_flip = not f_bull[i] and f_bull[i - 1]

        # ── Exits: fast SHA flips against position ──
        if pos == 1 and fast_bear_flip:
            long_exit[i] = True
            pos = 0
            last_exit_bar = i

        elif pos == -1 and fast_bull_flip:
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        # ── Entries: fast flip + slow agreement + cooldown ──
        if pos == 0 and (i - last_exit_bar) >= cooldown:
            if fast_bull_flip and s_bull[i]:
                long_entry[i] = True
                pos = 1
            elif fast_bear_flip and not s_bull[i]:
                short_entry[i] = True
                pos = -1

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
