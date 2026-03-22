"""EA003R: DGT Confluence Master — R007 Combined Core

Redesign of EA003 using the full R007 architecture as the trigger layer:
  R002 (reversal priority, no cooldown) + R001 (continuation fallback, cooldown)

Both trigger types are gated by the same 4-dimensional confluence score:
  1. Trend   — Raff Regression Channel slope agrees with entry direction
  2. Value   — Close within 5 bricks of a Fib retracement (38.2/50/61.8%)
               OR at the VP Value Area edge (VAL for longs, VAH for shorts)
  3. Timing  — Any major Fib time hit in the last FIB_TIME_WINDOW bricks
  4. Confirm — MACD divergence OR Distance Oscillator extreme

Entry fires when confluence score >= min_confluence.
Exit: first opposing Renko brick (standard).

Diff vs EA003:
  - Adds R001 continuation signal (same direction as run, with cooldown)
    as a fallback when no R002 reversal is available
  - strong_confirm removed (fixed at OR logic) to reduce combo count
  - Adds cooldown parameter for R001 gate

EURAUD IS:  2023-07-20 → 2025-09-30
EURAUD OOS: 2025-10-01 → 2026-03-17
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from indicators.dgtrd.distance_oscillator import distance_oscillator_sr
from indicators.dgtrd.fib_levels import fib_levels
from indicators.dgtrd.fib_time import fib_time_zones
from indicators.dgtrd.oscillators import oscillators_overlay
from indicators.dgtrd.raff_regression import raff_regression_channel
from indicators.dgtrd.volume_profile import volume_profile_pivot_anchored
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

DESCRIPTION = "EURAUD R007-core (R002 priority + R001 fallback) gated by 4-dim confluence score"

HYPOTHESIS = (
    "Adding R001 continuation trades (gated by the same confluence score) to the "
    "R002 reversal triggers should capture trend-continuation edge missed by EA003, "
    "while the confluence gate prevents low-quality continuation entries."
)

RENKO_FILE      = "OANDA_EURAUD, 1S renko 0.0006.csv"
COMMISSION_PCT  = 0.009
INITIAL_CAPITAL = 1000.0

# Number of recent bricks to scan for a Fib time hit (dim 3 window)
_FIB_TIME_WINDOW = 8

# ---------------------------------------------------------------------------
# PARAM_GRID
# ---------------------------------------------------------------------------
# cooldown applies ONLY to R001 entries (R002 is always exempt)
PARAM_GRID = {
    "n_bricks":       [2, 3, 4, 5],
    "cooldown":       [10, 20, 30],
    "session_start":  [0, 13],
    "min_confluence": [1, 2, 3],
}


# ---------------------------------------------------------------------------
# Indicator cache  (built once at import time)
# ---------------------------------------------------------------------------

def _build_indicator_cache() -> tuple[pd.DataFrame, float]:
    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)

    brick_size = float(np.median(df["High"].values - df["Low"].values))
    fib_tol    = 5.0 * brick_size

    # ── 1. Raff Regression Channel ─────────────────────────────────────────
    df = raff_regression_channel(df, source_col="Close", length=50, raff_length=100)
    df["rrc_slope"] = df["rrc_slope"].shift(1)

    # ── 2. Volume Profile (pivot-anchored) ─────────────────────────────────
    df = volume_profile_pivot_anchored(df, pvt_length=20, num_bins=25, va_pct=0.68)
    df["vp_val"] = df["vp_val"].shift(1)
    df["vp_vah"] = df["vp_vah"].shift(1)

    # ── 3. Fibonacci Price Levels ───────────────────────────────────────────
    df = fib_levels(df, deviation_mult=3.0, depth=11, htf="D")
    for col in ("fl_ret_0382", "fl_ret_05", "fl_ret_0618"):
        df[col] = df[col].shift(1)

    # ── 4. Fibonacci Time Zones ─────────────────────────────────────────────
    df = fib_time_zones(df, deviation_mult=3.0, depth=11)
    _FT_COLS = [
        "ft_trend_0618", "ft_trend_10", "ft_trend_1618",
        "ft_tz_5", "ft_tz_8", "ft_tz_13", "ft_tz_21",
    ]
    fib_hit = np.zeros(len(df), dtype=bool)
    for col in _FT_COLS:
        if col in df.columns:
            fib_hit |= df[col].values.astype(bool)
    df["ft_any_hit"] = pd.Series(fib_hit, index=df.index).shift(1).fillna(False)

    # ── 5. MACD Oscillator Divergences ─────────────────────────────────────
    df = oscillators_overlay(
        df, osc_type="MACD",
        macd_fast=12, macd_slow=26, macd_signal=9,
        prefix="osc_",
    )
    df["osc_bull_div"] = df["osc_bull_div"].shift(1)
    df["osc_bear_div"] = df["osc_bear_div"].shift(1)

    # ── 6. Distance Oscillator ─────────────────────────────────────────────
    df = distance_oscillator_sr(df, ma_length=21, bb_length=233, bb_mult=2.5)
    df["do_overbought"] = df["do_overbought"].shift(1)
    df["do_oversold"]   = df["do_oversold"].shift(1)

    return df, fib_tol


_CACHE, _FIB_TOL = _build_indicator_cache()

# Warmup: bb_length=233 (DO) is the longest warmup; use 300 for safety
_WARMUP = 300


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    n_bricks:       int = 3,
    cooldown:       int = 10,
    session_start:  int = 0,
    min_confluence: int = 2,
) -> pd.DataFrame:
    """
    R007-core (R002 + R001) gated by 4-dimensional confluence score.

    Confluence score per entry candidate (0–4):
      +1  Trend:   Raff slope direction matches trade direction
      +1  Value:   Close within fib_tol of a key Fib retracement,
                   OR Close at/beyond the VP Value Area edge
      +1  Timing:  Any major Fib time hit in last _FIB_TIME_WINDOW bricks
      +1  Confirm: MACD div OR Distance Oscillator extreme

    Enter when score >= min_confluence.
    R002 (reversal) takes priority; R001 (continuation) fires as fallback with cooldown.
    """
    c = _CACHE.reindex(df.index)

    rrc_slope     = c["rrc_slope"].values
    vp_val        = c["vp_val"].values
    vp_vah        = c["vp_vah"].values
    fl_ret_0382   = c["fl_ret_0382"].values
    fl_ret_05     = c["fl_ret_05"].values
    fl_ret_0618   = c["fl_ret_0618"].values
    ft_any_hit    = c["ft_any_hit"].fillna(False).values.astype(bool)
    bull_div      = c["osc_bull_div"].fillna(False).values.astype(bool)
    bear_div      = c["osc_bear_div"].fillna(False).values.astype(bool)
    do_oversold   = c["do_oversold"].fillna(False).values.astype(bool)
    do_overbought = c["do_overbought"].fillna(False).values.astype(bool)

    close    = df["Close"].values
    brick_up = df["brick_up"].values
    hours    = df.index.hour
    n        = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # Pre-build rolling OR of ft_any_hit over last _FIB_TIME_WINDOW bricks
    ft_recent = np.zeros(n, dtype=bool)
    for i in range(_FIB_TIME_WINDOW, n):
        ft_recent[i] = bool(np.any(ft_any_hit[i - _FIB_TIME_WINDOW : i + 1]))

    in_position    = False
    trade_dir      = 0
    last_r001_bar  = -999_999

    def _confluence_score(direction: int, px: float, i: int) -> int:
        score = 0

        # Dim 1 — Trend: Raff slope aligned with direction
        slope = rrc_slope[i]
        if not np.isnan(slope):
            if (direction == 1 and slope > 0) or (direction == -1 and slope < 0):
                score += 1

        # Dim 2 — Value: near Fib retracement OR at VP Value Area edge
        val_ok = False
        for fib_lvl in (fl_ret_0382[i], fl_ret_05[i], fl_ret_0618[i]):
            if not np.isnan(fib_lvl) and abs(px - fib_lvl) <= _FIB_TOL:
                val_ok = True
                break
        if not val_ok:
            if direction == 1 and not np.isnan(vp_val[i]):
                val_ok = px <= vp_val[i] + _FIB_TOL
            elif direction == -1 and not np.isnan(vp_vah[i]):
                val_ok = px >= vp_vah[i] - _FIB_TOL
        if val_ok:
            score += 1

        # Dim 3 — Timing: any major Fib time hit in last _FIB_TIME_WINDOW bricks
        if ft_recent[i]:
            score += 1

        # Dim 4 — Confirmation: MACD divergence OR DO extreme
        if direction == 1:
            confirm_ok = bool(bull_div[i]) or bool(do_oversold[i])
        else:
            confirm_ok = bool(bear_div[i]) or bool(do_overbought[i])
        if confirm_ok:
            score += 1

        return score

    for i in range(_WARMUP, n):
        up = bool(brick_up[i])

        # ── Exit: first opposing brick ─────────────────────────────────────
        if in_position:
            if trade_dir == 1 and not up:
                long_exit[i]   = True
                in_position    = False
                trade_dir      = 0
            elif trade_dir == -1 and up:
                short_exit[i]  = True
                in_position    = False
                trade_dir      = 0

        if in_position:
            continue

        # ── Session filter ─────────────────────────────────────────────────
        if session_start > 0 and hours[i] < session_start:
            continue

        px = close[i]

        # ── R002 reversal trigger (priority, no cooldown) ──────────────────
        window_r2 = brick_up[i - n_bricks : i]
        if len(window_r2) == n_bricks:
            prev_all_down_r2 = bool(not np.any(window_r2))
            prev_all_up_r2   = bool(np.all(window_r2))

            if prev_all_down_r2 and up:
                if _confluence_score(1, px, i) >= min_confluence:
                    long_entry[i] = True
                    in_position   = True
                    trade_dir     = 1
                    continue

            elif prev_all_up_r2 and not up:
                if _confluence_score(-1, px, i) >= min_confluence:
                    short_entry[i] = True
                    in_position    = True
                    trade_dir      = -1
                    continue

        # ── R001 continuation trigger (fallback, with cooldown) ────────────
        if (i - last_r001_bar) >= cooldown:
            window_r1 = brick_up[i - n_bricks + 1 : i + 1]
            if len(window_r1) == n_bricks:
                if bool(np.all(window_r1)):
                    if _confluence_score(1, px, i) >= min_confluence:
                        long_entry[i]  = True
                        in_position    = True
                        trade_dir      = 1
                        last_r001_bar  = i
                elif bool(not np.any(window_r1)):
                    if _confluence_score(-1, px, i) >= min_confluence:
                        short_entry[i] = True
                        in_position    = True
                        trade_dir      = -1
                        last_r001_bar  = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
