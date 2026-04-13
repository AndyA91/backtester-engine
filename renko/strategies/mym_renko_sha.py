"""
MYM Renko Dual SHA — Smoothed Heikin Ashi on Renko bricks

Same dual SHA concept but on Renko data. Key difference from candle SHA:
- Renko already removes noise, so SHA smoothing may compound well
- brick_up available as exit signal (first opposing brick = standard exit)
- Exit modes: ATR SL/TP, brick flip, or SHA flip
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Dual Smoothed HA on MYM Renko bricks"

HYPOTHESIS = (
    "SHA on Renko double-smooths price action. Renko removes time-noise, "
    "SHA removes brick-to-brick chop. Combined should give very clean signals. "
    "Brick flip exit is the proven Renko exit — may outperform ATR exits."
)


def _ema(src: np.ndarray, length: int) -> np.ndarray:
    out = np.empty_like(src, dtype=float)
    out[0] = src[0]
    k = 2.0 / (length + 1)
    for i in range(1, len(src)):
        out[i] = src[i] * k + out[i - 1] * (1 - k)
    return out


def _smoothed_ha(o, h, l, c, length):
    s_o = _ema(o, length)
    s_h = _ema(h, length)
    s_l = _ema(l, length)
    s_c = _ema(c, length)

    n = len(o)
    ha_open = np.empty(n, dtype=float)
    ha_close = (s_o + s_h + s_l + s_c) / 4.0

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
    min_slow_streak: int = 0,
    exit_mode: str = "brick_flip",
) -> pd.DataFrame:
    """
    Generate dual SHA signals on Renko data.

    Args:
        exit_mode: "brick_flip" = first opposing brick (standard Renko exit)
                   "sha_flip" = fast SHA flips against position
                   "both" = whichever fires first
    """
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    brick_up = df["brick_up"].values
    n = len(df)

    _, _, f_bull = _smoothed_ha(o, h, l, c, fast_len)
    _, _, s_bull = _smoothed_ha(o, h, l, c, slow_len)

    # Slow streak
    slow_streaks = np.zeros(n, dtype=int)
    count = 1
    for i in range(1, n):
        if s_bull[i] == s_bull[i - 1]:
            count += 1
        else:
            count = 1
        slow_streaks[i] = count

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    warmup = max(fast_len, slow_len) + 1
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        fast_bull_flip = f_bull[i] and not f_bull[i - 1]
        fast_bear_flip = not f_bull[i] and f_bull[i - 1]

        # ── Exits ──
        if pos == 1:
            do_exit = False
            if exit_mode == "brick_flip":
                do_exit = not brick_up[i]
            elif exit_mode == "sha_flip":
                do_exit = fast_bear_flip
            elif exit_mode == "both":
                do_exit = (not brick_up[i]) or fast_bear_flip

            if do_exit:
                long_exit[i] = True
                pos = 0
                last_exit_bar = i

        elif pos == -1:
            do_exit = False
            if exit_mode == "brick_flip":
                do_exit = brick_up[i]
            elif exit_mode == "sha_flip":
                do_exit = fast_bull_flip
            elif exit_mode == "both":
                do_exit = brick_up[i] or fast_bull_flip

            if do_exit:
                short_exit[i] = True
                pos = 0
                last_exit_bar = i

        # ── Entries ──
        if pos == 0 and (i - last_exit_bar) >= cooldown:
            streak_ok = (min_slow_streak == 0) or (slow_streaks[i] >= min_slow_streak)

            if streak_ok and fast_bull_flip and s_bull[i]:
                long_entry[i] = True
                pos = 1
            elif streak_ok and fast_bear_flip and not s_bull[i]:
                short_entry[i] = True
                pos = -1

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
