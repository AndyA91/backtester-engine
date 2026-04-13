"""
KAMA R3 + MK v4 — Candle Adaptation

Candle port of R033 (renko/strategies/r033_kama_ribbon_3l_mk.py).
Same core mechanism:
  - 3 KAMAs: K1 (fast), K2 (mid), K3 (trend) — alignment flip triggers entry
  - MK v4 momentum regime filter
  - Distance filter: max distance from K3 (ATR-based, not brick-based)
  - TP/SL bracket exits (ATR-based, not brick-based)

Key differences from Renko R033:
  1. ATR replaces BRICK_SIZE for all distance calculations (TP, SL, distance filter)
  2. ATR period is a sweep param (default 14)
  3. TP/SL expressed as ATR multiples, not brick counts or fixed price
  4. All indicators pre-shifted by 1 bar (same causality convention)
  5. No stateful signal generator — fully vectorized, engine manages position
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama
from indicators.momentum_king_v4 import calc_momentum_king_v4
from engine import calc_atr

DESCRIPTION = "3-KAMA ribbon + MK v4 + ATR-based distance filter + ATR-based TP/SL bracket (candle)"

HYPOTHESIS = (
    "R033's edge comes from bracket geometry + chop filtering, not from Renko's "
    "denoising. By replacing brick-based distances with ATR multiples, the same "
    "mechanism should adapt to candle volatility. ATR normalizes TP/SL across "
    "instruments and timeframes automatically."
)

# ── Default KAMA periods (same as R033) ──────────────────────────────────────
K1_LEN = 5
K2_LEN = 13
K3_LEN = 60
KAMA_FAST_SC = 2
KAMA_SLOW_SC = 30

# ── MK v4 defaults (same as R033) ───────────────────────────────────────────
MK_EMA_LEN          = 14
MK_SMOOTH_BASE      = 3
MK_MAX_SMOOTH       = 10
MK_ATR_LEN          = 14
MK_VOL_FACTOR       = 1.5
MK_NORM_LOOKBACK    = 200
MK_STRENGTH_THRESH  = 0.6
MK_NEUTRAL_ATR_PCT  = 0.3
MK_SIGNAL_LEN       = 9

# ── Sweep grid ──────────────────────────────────────────────────────────────
PARAM_GRID = {
    # TP as ATR multiples
    "tp_atr_mult":     [1.5, 2.0, 2.5, 3.0, 4.0],
    # SL as ATR multiples
    "sl_atr_mult":     [0.5, 0.75, 1.0, 1.5, 2.0],
    # Distance filter: max ATR multiples from K3 (100 = effectively off)
    "max_dist_atr":    [2.0, 4.0, 6.0, 100.0],
    # MK v4 mode (same as R033)
    "mk_mode":         ["off", "direction", "block_flat", "strong"],
    # ATR period for all distance calculations
    "atr_period":      [14],
}

# ── Worker-level caches ─────────────────────────────────────────────────────
_KAMA_CACHE = {}
_MK_CACHE   = {}
_ATR_CACHE  = {}


def _get_kama(close: pd.Series, length: int) -> np.ndarray:
    key = (length, KAMA_FAST_SC, KAMA_SLOW_SC)
    if key not in _KAMA_CACHE:
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
        smoothed = pd.Series(result["smoothed_momentum"]).shift(1).values
        regime   = pd.Series(result["regime"]).shift(1).values
        neutral  = pd.Series(result["neutral_zone_width"]).shift(1).values
        _MK_CACHE[key] = {
            "smoothed": smoothed,
            "regime":   regime,
            "neutral":  neutral,
        }
    return _MK_CACHE[key]


def _get_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    if period not in _ATR_CACHE:
        atr = calc_atr(df, period=period)
        # Shift by 1 — ATR at [i] is computed through bar i-1
        _ATR_CACHE[period] = pd.Series(atr).shift(1).values
    return _ATR_CACHE[period]


def generate_signals(
    df: pd.DataFrame,
    tp_atr_mult: float = 2.5,
    sl_atr_mult: float = 1.0,
    max_dist_atr: float = 4.0,
    mk_mode: str = "block_flat",
    atr_period: int = 14,
    use_gray_exit: bool = False,
) -> pd.DataFrame:
    """
    KAMA R3 + MK v4 for candles — ATR-based bracket exits.

    Stateless — engine manages position. Returns df with long_entry, long_exit,
    short_entry, short_exit, tp_offset, sl_offset columns.

    TP/SL offsets are per-bar (ATR * multiplier), so the bracket adapts to
    current volatility at entry time.
    """
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    if mk_mode not in ("off", "direction", "block_flat", "strong"):
        df = df.copy()
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
    atr = _get_atr(df, atr_period)

    any_nan    = np.isnan(k1) | np.isnan(k2) | np.isnan(k3) | np.isnan(atr)
    bull_align = (k1 > k2) & (k2 > k3) & ~any_nan
    bear_align = (k1 < k2) & (k2 < k3) & ~any_nan

    bull_prev = np.roll(bull_align, 1); bull_prev[0] = False
    bear_prev = np.roll(bear_align, 1); bear_prev[0] = False

    raw_long_entry  = bull_align & ~bull_prev
    raw_short_entry = bear_align & ~bear_prev

    # ── Distance filter (price not too far from K3, ATR-scaled) ─────────────
    close_arr    = close.values
    max_dist_pr  = max_dist_atr * atr  # per-bar distance threshold
    dist_above   = close_arr - k3
    dist_below   = k3 - close_arr
    long_dist_ok  = dist_above <= max_dist_pr
    short_dist_ok = dist_below <= max_dist_pr

    # ── MK v4 filter (identical to R033) ────────────────────────────────────
    if mk_mode == "off":
        mk_long_ok  = np.ones(n, dtype=bool)
        mk_short_ok = np.ones(n, dtype=bool)
    else:
        mk = _get_mk(df)
        smoothed = mk["smoothed"]
        regime   = mk["regime"]
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

    warmup = max(K3_LEN, MK_NORM_LOOKBACK, atr_period) + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry  = raw_long_entry  & long_dist_ok  & mk_long_ok  & mask
    short_entry = raw_short_entry & short_dist_ok & mk_short_ok & mask

    if use_gray_exit:
        long_exit  = (~bull_align) & mask
        short_exit = (~bear_align) & mask

    # ── ATR-based TP/SL (per-bar, adapts to current volatility) ─────────────
    tp_offset = tp_atr_mult * atr
    sl_offset = sl_atr_mult * atr
    # Replace NaN with 0 so engine doesn't choke on early bars
    tp_offset = np.where(np.isnan(tp_offset), 0.0, tp_offset)
    sl_offset = np.where(np.isnan(sl_offset), 0.0, sl_offset)

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    df["tp_offset"]   = tp_offset
    df["sl_offset"]   = sl_offset
    return df
