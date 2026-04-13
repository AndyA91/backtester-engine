"""
GJ001: Regime Flip — ER Regime + Brick Direction + Trailing Stop

Entry: Efficiency Ratio regime flips from RANGE → TREND.
       Up-brick on flip bar = long, down-brick = short.
Exit:  Trailing stop in bricks — tracks best price since entry,
       exits when price retraces trail_bricks * BRICK_SIZE from peak.
       trail_bricks=1 is equivalent to first-opposing-brick (R1).
       Optional TP bracket in bricks.

This is the first strategy to use a regime-flip as the PRIMARY entry trigger.
ER = |close - close[N]| / sum(|close[i] - close[i-1]|, i=0..N-1)
On Renko: ER ≈ |net bricks| / total bricks — pure direction consistency.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "ER regime flip + brick direction entry, trailing stop exit"

HYPOTHESIS = (
    "Efficiency Ratio detects when a Renko chart transitions from choppy "
    "(alternating bricks) to directional (runs of same-direction bricks). "
    "Entering on the regime flip catches the start of a trend move. A trailing "
    "stop lets winners run further than R1's first-opposing-brick exit while "
    "still capping losses. Edge comes from timing + ride length."
)

BRICK_SIZE = 0.09  # GBPJPY 0.09

PARAM_GRID = {
    "er_lookback":   [6, 8, 10, 14, 20],
    "er_smooth":     [1, 3, 5],
    "trend_thresh":  [0.35, 0.45, 0.55],
    "range_thresh":  [0.15, 0.25],
    "persistence":   [0, 2],
    "tp_bricks":     [0, 3, 5, 8],
    "trail_bricks":  [1, 2, 3],
}

# ── Caches ───────────────────────────────────────────────────────────────────
_ER_CACHE = {}


def _calc_er(close: np.ndarray, lookback: int, smooth: int) -> np.ndarray:
    """Compute Efficiency Ratio, optionally EMA-smoothed."""
    key = (lookback, smooth)
    if key in _ER_CACHE:
        return _ER_CACHE[key]

    n = len(close)
    raw_er = np.full(n, np.nan)

    for i in range(lookback, n):
        direction = abs(close[i] - close[i - lookback])
        volatility = 0.0
        for j in range(i - lookback + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        raw_er[i] = direction / volatility if volatility != 0 else 0.0

    # EMA smoothing
    if smooth <= 1:
        er = raw_er
    else:
        er = np.full(n, np.nan)
        k = 2.0 / (smooth + 1)
        first_valid = lookback
        er[first_valid] = raw_er[first_valid]
        for i in range(first_valid + 1, n):
            if np.isnan(raw_er[i]):
                continue
            prev = er[i - 1]
            if np.isnan(prev):
                er[i] = raw_er[i]
            else:
                er[i] = raw_er[i] * k + prev * (1 - k)

    # Pre-shift by 1 — ER at [i] is computed through bar i-1
    result = np.roll(er, 1)
    result[0] = np.nan

    _ER_CACHE[key] = result
    return result


def generate_signals(
    df: pd.DataFrame,
    er_lookback: int = 10,
    er_smooth: int = 5,
    trend_thresh: float = 0.45,
    range_thresh: float = 0.25,
    persistence: int = 0,
    tp_bricks: int = 0,
    trail_bricks: int = 1,
) -> pd.DataFrame:
    """
    GJ001: ER regime flip + brick direction + trailing stop.

    Position tracking is needed for the trailing stop (track peak price).
    On Renko this is reliable — each brick moves exactly BRICK_SIZE.
    trail_bricks=1 is identical to first-opposing-brick exit (R1).
    """
    n = len(df)
    close = df["Close"].values.astype(float)
    brick_up = df["brick_up"].values

    er = _calc_er(close, er_lookback, er_smooth)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # Regime state
    regime = 0  # 0=neutral, 1=trend, -1=range
    trend_count = 0
    range_count = 0
    prev_trending = False

    # Position + trailing stop state
    pos = 0          # +1=long, -1=short, 0=flat
    peak_price = 0.0  # best price since entry (highest for long, lowest for short)
    trail_dist = trail_bricks * BRICK_SIZE
    tp_dist = tp_bricks * BRICK_SIZE if tp_bricks > 0 else 0.0
    entry_price = 0.0

    warmup = er_lookback + er_smooth + 5

    for i in range(warmup, n):
        if np.isnan(er[i]):
            continue

        # ── Regime classification (hysteresis + persistence) ─────────
        raw_trending = er[i] >= trend_thresh
        raw_ranging  = er[i] <= range_thresh

        if raw_trending:
            trend_count += 1
            range_count = 0
        elif raw_ranging:
            range_count += 1
            trend_count = 0
        else:
            trend_count = 0
            range_count = 0

        if trend_count > persistence:
            regime = 1
        elif range_count > persistence:
            regime = -1

        is_trending = regime == 1

        # ── Exit: trailing stop + optional TP ────────────────────────
        if pos == 1:
            # Update peak (highest close since entry)
            if close[i] > peak_price:
                peak_price = close[i]
            # Trailing stop: price dropped trail_bricks from peak
            if peak_price - close[i] >= trail_dist:
                long_exit[i] = True
                pos = 0
            # TP: price reached tp_bricks above entry
            elif tp_dist > 0 and close[i] - entry_price >= tp_dist:
                long_exit[i] = True
                pos = 0

        elif pos == -1:
            # Update peak (lowest close since entry)
            if close[i] < peak_price:
                peak_price = close[i]
            # Trailing stop: price rose trail_bricks from trough
            if close[i] - peak_price >= trail_dist:
                short_exit[i] = True
                pos = 0
            # TP: price reached tp_bricks below entry
            elif tp_dist > 0 and entry_price - close[i] >= tp_dist:
                short_exit[i] = True
                pos = 0

        # ── Entry: regime flips to TREND (only when flat) ────────────
        regime_flip_to_trend = is_trending and not prev_trending

        if pos == 0 and regime_flip_to_trend:
            if brick_up[i]:
                long_entry[i] = True
                pos = 1
                entry_price = close[i]
                peak_price = close[i]
            else:
                short_entry[i] = True
                pos = -1
                entry_price = close[i]
                peak_price = close[i]

        prev_trending = is_trending

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    df["tp_offset"]   = 0.0
    df["sl_offset"]   = 0.0

    return df
