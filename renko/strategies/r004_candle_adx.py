"""
R004: R001 + ADX Gate (5m Candles)

N consecutive same-direction Renko bricks -> momentum entry, but ONLY when
ADX(14) computed on the SOURCE 5m candle chart exceeds adx_threshold.

Rationale:
  R005 tested ADX computed on Renko bricks — marginally negative (ADX on
  directional-by-definition bricks adds noise). R004 tests ADX from the
  original OANDA 5m candle chart: a fundamentally different signal source
  that measures true price trend strength including wicks and bar dynamics.
  ADX was r=+0.521 with PnL in candle strategies (v3/v4 trade analysis).

Alignment: 5m candle ADX is .shift(1) before merge_asof — each brick uses
  the ADX from the last COMPLETED 5m candle before the brick formed (Pitfalls
  #5 and #7: compute on raw OHLC, shift output; backward merge = last closed bar).

NOTE: 5m data covers 2025-11-23 -> 2026-03-03 only.
  Run with: python runner.py r004_candle_adx --start 2025-11-24 --end 2026-03-03
  The standard IS dates (2023-01-23 -> 2025-09-30) predate the 5m file entirely.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from engine import load_tv_export         # noqa: E402
from indicators.adx import calc_adx       # noqa: E402

DESCRIPTION = "R001 momentum bricks + ADX(14) gate from 5m candle chart"

HYPOTHESIS = (
    "R005 showed ADX computed on Renko bricks is marginally negative — bricks are "
    "directional by nature so brick-space ADX adds noise. Candle ADX measures true "
    "trend strength (wicks, bar range, close-to-close dynamics). ADX was the #1 "
    "per-trade PnL predictor (r=+0.521) in KAMA candle strategies. This tests "
    "whether that predictive power transfers to Renko entry filtering when the "
    "signal source is the original 5m chart rather than brick space."
)

PARAM_GRID = {
    "n_bricks":      [2, 3, 4],
    "cooldown":      [10, 20, 30],
    "adx_threshold": [20, 25, 30],
}
# 3 x 3 x 3 = 27 combinations

# ── Module-level lazy cache for 5m candle ADX ─────────────────────────────
_CANDLE_ADX = None   # pd.Series, populated on first generate_signals call


def _load_candle_adx():
    """Load OANDA 5m candle ADX once, cache as module-level Series."""
    global _CANDLE_ADX
    if _CANDLE_ADX is not None:
        return _CANDLE_ADX

    df_5m = load_tv_export("OANDA_EURUSD, 5.csv")

    # Pitfall #7: compute ADX on raw OHLC, then .shift(1) the output.
    # Shifting the input would inject NaN into High/Low/Close simultaneously.
    adx_raw = calc_adx(df_5m, di_period=14, adx_period=14)
    _CANDLE_ADX = pd.Series(
        adx_raw["adx"], index=df_5m.index, name="candle_adx"
    ).shift(1)

    return _CANDLE_ADX


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 3,
    cooldown: int = 10,
    adx_threshold: float = 25,
) -> pd.DataFrame:
    """
    Generate R001-style momentum entries gated by 5m candle ADX.

    Entry:  N consecutive same-direction bricks AND candle ADX >= adx_threshold.
    Exit:   First opposing brick (unconditional — same as R001).
    Cooldown: minimum bricks between entries.

    Args:
        df:             Renko DataFrame with brick_up bool column.
        n_bricks:       Consecutive bricks required before entry signal.
        cooldown:       Minimum bricks between entries.
        adx_threshold:  Minimum 5m candle ADX(14) to allow entry. 0 = disabled.

    Returns:
        df with columns long_entry, long_exit, short_entry, short_exit (bool).
    """
    n        = len(df)
    brick_up = df["brick_up"].values

    # ── Align 5m candle ADX to brick timestamps ───────────────────────────
    # merge_asof direction='backward': each brick gets the ADX value whose
    # timestamp <= brick timestamp (last completed 5m bar before this brick).
    # Combined with the .shift(1) in _load_candle_adx(), this means each
    # brick sees ADX computed at the close of the bar BEFORE the current bar.
    candle_adx = _load_candle_adx()

    # Pandas merge_asof requires identical datetime dtypes on both keys.
    # Renko index can be ns/us while candle index can be s; normalize both to ns.
    left = pd.DataFrame(index=pd.DatetimeIndex(df.index).astype("datetime64[ns]"))
    right = candle_adx.to_frame()
    right.index = pd.DatetimeIndex(right.index).astype("datetime64[ns]")

    adx_aligned = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction="backward",
    )["candle_adx"].values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(n_bricks, 30)   # 30 bars covers ADX(14) Wilder's warmup

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick (unconditional, no cooldown) ────────
        long_exit[i]  = not up
        short_exit[i] = up

        # ── Entry ──────────────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown:
            continue

        # N consecutive same-direction bricks ending at bar i
        window   = brick_up[i - n_bricks + 1 : i + 1]
        all_up   = bool(np.all(window))
        all_down = bool(not np.any(window))

        if not (all_up or all_down):
            continue

        # ADX gate — candle ADX from last completed 5m bar
        adx_val = adx_aligned[i]
        if adx_threshold > 0 and (np.isnan(adx_val) or adx_val < adx_threshold):
            continue

        if all_up:
            long_entry[i] = True
        else:
            short_entry[i] = True
        last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
