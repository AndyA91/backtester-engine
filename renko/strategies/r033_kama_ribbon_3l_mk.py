"""
R033: KAMA Ribbon 3L + MK v4 Filter + Distance Filter + TP/SL Bracket

Python port of the new visual Pine strategy (kama_ribbon.pine 3L variant).
Mechanism is fundamentally different from R030/R031/R032:

  - 3 KAMAs: K1=5 (fast ribbon), K2=13 (mid ribbon), K3=60 (trend confirm)
  - Entry on alignment FLIP (k1>k2>k3 for long, inverse for short)
  - Filters (composable, R6-friendly, all from different signal families):
      • Distance from K3 (block entries that fire after a stretched move)
      • MK v4 momentum regime (block flat / wrong-direction)
  - Exits driven by absolute TP/SL bracket via engine `tp_offset`/`sl_offset`
  - Optional gray exit (alignment break) — toggleable

This is NOT a re-test of the R030 dead-end family. The distinguishing mechanism
is the TP/SL bracket: TV results showed that with bracket-driven exits, the
gray exit becomes dead code and the strategy is fundamentally R/R-driven, not
trend-ride driven.

Tested first on EURAUD 0.0006. KAMA periods are LOCKED at 5/13/60 (the user's
visual choice) so this sweep isolates bracket geometry + filter contribution.
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama
from indicators.momentum_king_v4 import calc_momentum_king_v4

DESCRIPTION = "3-KAMA ribbon + MK v4 + distance filter + absolute TP/SL bracket"

HYPOTHESIS = (
    "Locked KAMA ribbon (5/13/60) provides directional triggers; the strategy's "
    "edge comes from the TP/SL bracket geometry and from filtering chop with MK "
    "v4 + distance-from-K3. R030/R031/R032 failed because they used gray-exit; "
    "this version uses bracket exits which TV showed survive better. Sweep should "
    "find a (TP, SL, dist, MK) combo that beats EA022 baseline AND survives FRESH."
)

# ── Locked params (visual choice from interactive Pine session) ───────────────
K1_LEN = 5
K2_LEN = 13
K3_LEN = 60
KAMA_FAST_SC = 2
KAMA_SLOW_SC = 30

# MK v4 inputs locked at indicator defaults
MK_EMA_LEN          = 14
MK_SMOOTH_BASE      = 3
MK_MAX_SMOOTH       = 10
MK_ATR_LEN          = 14
MK_VOL_FACTOR       = 1.5
MK_NORM_LOOKBACK    = 200
MK_STRENGTH_THRESH  = 0.6
MK_NEUTRAL_ATR_PCT  = 0.3
MK_SIGNAL_LEN       = 9

# ── Sweep grid ────────────────────────────────────────────────────────────────
PARAM_GRID = {
    # TP distance in price units (e.g. 0.0024 = 4 bricks)
    "tp_dist":     [0.0018, 0.0024, 0.0030, 0.0036, 0.0048],
    # SL distance in BRICKS (converted to price using BRICK_SIZE constant)
    "sl_bricks":   [1, 2, 3, 4],
    # Distance filter: max bricks above/below K3 (1000 = effectively off)
    "max_dist_bricks": [4, 8, 12, 1000],
    # MK v4 mode
    #   "off"        — no MK filter
    #   "direction"  — only direction match (smoothed_momentum sign)
    #   "block_flat" — direction + block flat
    #   "strong"     — direction + block flat + require strong (no weak)
    "mk_mode":     ["off", "direction", "block_flat", "strong"],
}

BRICK_SIZE = 0.0006  # EURAUD 0.0006 chart

# ── Worker-level caches (shared across combos in the same process) ────────────
_KAMA_CACHE = {}
_MK_CACHE   = {}


def _get_kama(close: pd.Series, length: int) -> np.ndarray:
    key = (length, KAMA_FAST_SC, KAMA_SLOW_SC)
    if key not in _KAMA_CACHE:
        # Shift by 1 — value at [i] is computed through bar i-1 (no lookahead)
        _KAMA_CACHE[key] = calc_kama(
            close, length=length, fast=KAMA_FAST_SC, slow=KAMA_SLOW_SC
        ).shift(1).values
    return _KAMA_CACHE[key]


def _get_mk(df: pd.DataFrame) -> dict:
    key = "mk_v4_default"
    if key not in _MK_CACHE:
        result = calc_momentum_king_v4(
            df,
            ema_length=MK_EMA_LEN,
            smoothing_base=MK_SMOOTH_BASE,
            max_smooth=MK_MAX_SMOOTH,
            atr_length=MK_ATR_LEN,
            volatility_factor=MK_VOL_FACTOR,
            norm_lookback=MK_NORM_LOOKBACK,
            strength_threshold=MK_STRENGTH_THRESH,
            neutral_atr_pct=MK_NEUTRAL_ATR_PCT,
            signal_length=MK_SIGNAL_LEN,
        )
        # Shift the regime + smoothed by 1 so value at [i] is causal (through i-1)
        smoothed = pd.Series(result["smoothed_momentum"]).shift(1).values
        regime   = pd.Series(result["regime"]).shift(1).values
        neutral  = pd.Series(result["neutral_zone_width"]).shift(1).values
        _MK_CACHE[key] = {
            "smoothed": smoothed,
            "regime":   regime,
            "neutral":  neutral,
        }
    return _MK_CACHE[key]


def generate_signals(
    df: pd.DataFrame,
    tp_dist: float = 0.0036,
    sl_bricks: int = 1,
    max_dist_bricks: int = 8,
    mk_mode: str = "block_flat",
    use_gray_exit: bool = False,
) -> pd.DataFrame:
    """
    R033: 3-KAMA ribbon + MK v4 + distance + bracket exits.

    Stateless — engine manages position. Returns df with long_entry, long_exit,
    short_entry, short_exit, tp_offset, sl_offset columns.
    """
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # Validate mk_mode
    if mk_mode not in ("off", "direction", "block_flat", "strong"):
        df["long_entry"]  = long_entry
        df["long_exit"]   = long_exit
        df["short_entry"] = short_entry
        df["short_exit"]  = short_exit
        df["tp_offset"]   = 0.0
        df["sl_offset"]   = 0.0
        return df

    close = df["Close"]
    k1 = _get_kama(close, K1_LEN)
    k2 = _get_kama(close, K2_LEN)
    k3 = _get_kama(close, K3_LEN)

    any_nan    = np.isnan(k1) | np.isnan(k2) | np.isnan(k3)
    bull_align = (k1 > k2) & (k2 > k3) & ~any_nan
    bear_align = (k1 < k2) & (k2 < k3) & ~any_nan

    bull_prev = np.roll(bull_align, 1); bull_prev[0] = False
    bear_prev = np.roll(bear_align, 1); bear_prev[0] = False

    raw_long_entry  = bull_align & ~bull_prev
    raw_short_entry = bear_align & ~bear_prev

    # ── Distance filter (price not too far from K3 at entry) ────────────────
    close_arr   = close.values
    max_dist_pr = max_dist_bricks * BRICK_SIZE
    dist_above_k3 = close_arr - k3
    dist_below_k3 = k3 - close_arr
    long_dist_ok  = dist_above_k3 <= max_dist_pr
    short_dist_ok = dist_below_k3 <= max_dist_pr

    # ── MK v4 filter ────────────────────────────────────────────────────────
    if mk_mode == "off":
        mk_long_ok  = np.ones(n, dtype=bool)
        mk_short_ok = np.ones(n, dtype=bool)
    else:
        mk = _get_mk(df)
        smoothed = mk["smoothed"]
        regime   = mk["regime"]
        # Direction match
        mk_long_dir  = ~np.isnan(smoothed) & (smoothed > 0)
        mk_short_dir = ~np.isnan(smoothed) & (smoothed < 0)
        if mk_mode == "direction":
            mk_long_ok  = mk_long_dir
            mk_short_ok = mk_short_dir
        elif mk_mode == "block_flat":
            not_flat = regime != "FLAT"
            mk_long_ok  = mk_long_dir  & not_flat
            mk_short_ok = mk_short_dir & not_flat
        else:  # strong
            mk_long_ok  = regime == "STRONG_UP"
            mk_short_ok = regime == "STRONG_DOWN"

    warmup = max(K3_LEN, MK_NORM_LOOKBACK) + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry  = raw_long_entry  & long_dist_ok  & mk_long_ok  & mask
    short_entry = raw_short_entry & short_dist_ok & mk_short_ok & mask

    if use_gray_exit:
        long_exit  = (~bull_align) & mask
        short_exit = (~bear_align) & mask
    # else: leave as all-False, bracket handles all exits

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    df["tp_offset"]   = tp_dist
    df["sl_offset"]   = sl_bricks * BRICK_SIZE
    return df
