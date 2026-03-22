"""EA005: Value Area Breakout — Auction Theory Momentum

Auction-theory breakout strategy. When price escapes the Volume Profile Value
Area (above VAH or below VAL) after being contained within it, the structural
imbalance signals high-probability continuation.

Signal logic:
  LONG:  Last `n_inside` bricks were inside the Value Area (vp_in_va=True)
         AND current brick close > vp_vah (broke above VA High)
         AND current brick is UP → buy breakout continuation
  SHORT: Last `n_inside` bricks were inside the VA
         AND current brick close < vp_val (broke below VA Low)
         AND current brick is DOWN → sell breakout continuation

Exit: first opposing Renko brick (standard).

Auction theory rationale: the Value Area (68% of volume) represents price
acceptance / fair value. A sustained breakout above VAH signals buyer dominance
and unfilled demand above accepted value — price should run until a new
accepted value region is found. The converse holds for breaks below VAL.

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.dgtrd.volume_profile import volume_profile_pivot_anchored
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD Value Area Breakout — auction-theory momentum from VA escape"

HYPOTHESIS = (
    "When Renko price escapes the VP Value Area after N bricks of acceptance "
    "inside it, structural imbalance drives continuation. pvt_length controls "
    "profile responsiveness; va_pct controls acceptance band width; "
    "n_inside confirms acceptance; opposing-brick exit respects Renko edge."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# pvt_length:    pivot lookback (smaller = more responsive profiles)
# va_pct:        Value Area percentage (0.60–0.80 = 60%–80%)
# n_inside:      consecutive bricks that must be inside the VA before breakout
# cooldown:      minimum bricks between entries
# session_start: UTC hour gate (0 = no gate, 13 = London+NY only)
PARAM_GRID = {
    "pvt_length":    [10, 15, 20],
    "va_pct":        [0.60, 0.70, 0.80],
    "n_inside":      [1, 2, 3],
    "cooldown":      [5, 10, 20, 30],
    "session_start": [0, 13],
}


# ---------------------------------------------------------------------------
# Indicator cache — keyed by (pvt_length, va_pct) to avoid redundant VP calc
# VP computation is expensive; only 9 unique combinations need building for
# the full 216-combo sweep.
# ---------------------------------------------------------------------------

_CACHE_DICT: dict = {}


def _get_or_build_cache(pvt_length: int, va_pct: float) -> pd.DataFrame:
    key = (pvt_length, va_pct)
    if key not in _CACHE_DICT:
        df = load_renko_export(RENKO_FILE)
        add_renko_indicators(df)

        df = volume_profile_pivot_anchored(df, pvt_length=pvt_length, num_bins=25, va_pct=va_pct)

        # Shift levels so they can be read safely at [i] in the signal loop
        df["vp_vah"]   = df["vp_vah"].shift(1)
        df["vp_val"]   = df["vp_val"].shift(1)
        # vp_in_va: pre-shift (previous bar's VA membership, used for n_inside lookback)
        df["vp_in_va"] = df["vp_in_va"].shift(1)

        _CACHE_DICT[key] = df
    return _CACHE_DICT[key]


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    pvt_length:    int   = 20,
    va_pct:        float = 0.68,
    n_inside:      int   = 1,
    cooldown:      int   = 10,
    session_start: int   = 0,
) -> pd.DataFrame:
    """
    Value Area Breakout: enter on confirmed escape after n_inside bricks of VA acceptance.

    Args:
        df:            Renko DataFrame with brick_up bool + OHLCV.
        pvt_length:    Pivot lookback for VP computation (10/15/20).
        va_pct:        Value Area fraction (0.60/0.70/0.80).
        n_inside:      Consecutive bricks inside the VA required before breakout.
        cooldown:      Minimum bars between entries.
        session_start: UTC hour gate (0 = disabled).

    Returns:
        df with long_entry, long_exit, short_entry, short_exit bool columns.
    """
    # pvt_length * 2 to confirm first pivot, then another * 2 for profile window, + buffer
    warmup = pvt_length * 5

    c = _get_or_build_cache(pvt_length, va_pct).reindex(df.index)

    vp_vah   = c["vp_vah"].values
    vp_val   = c["vp_val"].values
    vp_in_va = c["vp_in_va"].fillna(False).values.astype(bool)

    close    = df["Close"].values
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

        # ── Cooldown check ─────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown:
            continue

        # ── Value Area level check ─────────────────────────────────────────
        vah_i = vp_vah[i]
        val_i = vp_val[i]

        # Skip if no profile established yet
        if np.isnan(vah_i) or np.isnan(val_i):
            continue

        # ── n_inside check: last n_inside bricks were inside the VA ────────
        if i < n_inside:
            continue

        window_in_va = vp_in_va[i - n_inside : i]
        if len(window_in_va) < n_inside or not bool(np.all(window_in_va)):
            continue

        # ── Entry: breakout above VAH (long) or below VAL (short) ──────────
        px = close[i]

        if up and px > vah_i:
            long_entry[i]  = True
            in_position    = True
            trade_dir      = 1
            last_trade_bar = i

        elif not up and px < val_i:
            short_entry[i] = True
            in_position    = True
            trade_dir      = -1
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
