"""
R034: KAMA Ribbon 3L + MK v4 Tunable (Config #2 locked, MK params exposed)

Follow-up to R033. Config #2 from R033 was TV-validated at:
  TRAIN    PF=2.30  WR=65.6%  T=482
  VAL      PF=2.09  WR=63.6%  T= 55
  HOLDOUT  PF=2.29  WR=65.4%  T= 52
  FRESH    PF=2.53  WR=68.0%  T= 25  (!!)

Config #2 used default MK v4 parameters. This sweep asks: can we tune the
MK v4 internal params to improve stability (min PF across TRAIN+VAL) while
keeping HOLDOUT/FRESH SEALED and post-hoc informational only?

LOCKED params (Config #2):
  - KAMA 5/13/60
  - TP 0.0018 (3 bricks)
  - SL 2 bricks (0.0012)
  - No distance filter (max_dist_bricks=1000)
  - MK v4 ENABLED
  - No gray exit

TUNABLE (sweep dimensions):
  - mk_mode ∈ {block_flat, strong}
  - ema_length ∈ [10, 14, 20]
  - max_smooth ∈ [8, 10, 14]
  - neutral_atr_pct ∈ [0.2, 0.3, 0.5]
  - strength_threshold ∈ [0.5, 0.6, 0.7]

= 2 × 3 × 3 × 3 × 3 = 162 combos

NOTE: FRESH and HOLDOUT are NOT used for finalist selection. Ranking is
by min(TRAIN PF, VAL PF) — the R26 candidate rule (stability-based ranking).
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama
from indicators.momentum_king_v4 import calc_momentum_king_v4

DESCRIPTION = "R033 Config #2 locked, MK v4 params swept (R26 stability ranking)"

HYPOTHESIS = (
    "Config #2 works with MK v4 at defaults. Tuning MK's neutral zone, "
    "smoothing cap, and strength threshold may improve stability (min PF "
    "across TRAIN+VAL) without contaminating FRESH. If no tuned config "
    "beats defaults on TRAIN+VAL-only selection, the defaults are near-"
    "optimal for this setup."
)

# ── Locked params (Config #2) ─────────────────────────────────────────────────
K1_LEN = 5
K2_LEN = 13
K3_LEN = 60
KAMA_FAST_SC = 2
KAMA_SLOW_SC = 30

TP_DIST         = 0.0018
SL_BRICKS       = 2
MAX_DIST_BRICKS = 1000   # distance filter effectively off
BRICK_SIZE      = 0.0006

# MK v4 params that are NOT swept (locked at indicator defaults)
MK_SMOOTH_BASE   = 3
MK_ATR_LEN       = 14
MK_VOL_FACTOR    = 1.5
MK_NORM_LOOKBACK = 200
MK_SIGNAL_LEN    = 9

# ── Sweep grid ────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "mk_mode":            ["block_flat", "strong"],
    "ema_length":         [10, 14, 20],
    "max_smooth":         [8, 10, 14],
    "neutral_atr_pct":    [0.2, 0.3, 0.5],
    "strength_threshold": [0.5, 0.6, 0.7],
}

# ── Worker-level caches ───────────────────────────────────────────────────────
_KAMA_CACHE = {}
_MK_CACHE   = {}


def _get_kama(close: pd.Series, length: int) -> np.ndarray:
    key = (length, KAMA_FAST_SC, KAMA_SLOW_SC)
    if key not in _KAMA_CACHE:
        _KAMA_CACHE[key] = calc_kama(
            close, length=length, fast=KAMA_FAST_SC, slow=KAMA_SLOW_SC
        ).shift(1).values
    return _KAMA_CACHE[key]


def _get_mk(df: pd.DataFrame, ema_length: int, max_smooth: int,
            neutral_atr_pct: float, strength_threshold: float) -> dict:
    key = (ema_length, max_smooth, neutral_atr_pct, strength_threshold)
    if key not in _MK_CACHE:
        result = calc_momentum_king_v4(
            df,
            ema_length=ema_length,
            smoothing_base=MK_SMOOTH_BASE,
            max_smooth=max_smooth,
            atr_length=MK_ATR_LEN,
            volatility_factor=MK_VOL_FACTOR,
            norm_lookback=MK_NORM_LOOKBACK,
            strength_threshold=strength_threshold,
            neutral_atr_pct=neutral_atr_pct,
            signal_length=MK_SIGNAL_LEN,
        )
        smoothed = pd.Series(result["smoothed_momentum"]).shift(1).values
        regime   = pd.Series(result["regime"]).shift(1).values
        _MK_CACHE[key] = {"smoothed": smoothed, "regime": regime}
    return _MK_CACHE[key]


def generate_signals(
    df: pd.DataFrame,
    mk_mode: str = "block_flat",
    ema_length: int = 14,
    max_smooth: int = 10,
    neutral_atr_pct: float = 0.3,
    strength_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    R034: Config #2 locked, MK v4 parameters per-call.
    """
    n = len(df)
    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    if mk_mode not in ("block_flat", "strong"):
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

    any_nan    = np.isnan(k1) | np.isnan(k2) | np.isnan(k3)
    bull_align = (k1 > k2) & (k2 > k3) & ~any_nan
    bear_align = (k1 < k2) & (k2 < k3) & ~any_nan

    bull_prev = np.roll(bull_align, 1); bull_prev[0] = False
    bear_prev = np.roll(bear_align, 1); bear_prev[0] = False

    raw_long_entry  = bull_align & ~bull_prev
    raw_short_entry = bear_align & ~bear_prev

    # MK filter
    mk = _get_mk(df, ema_length, max_smooth, neutral_atr_pct, strength_threshold)
    smoothed = mk["smoothed"]
    regime   = mk["regime"]

    mk_long_dir  = ~np.isnan(smoothed) & (smoothed > 0)
    mk_short_dir = ~np.isnan(smoothed) & (smoothed < 0)

    if mk_mode == "block_flat":
        not_flat = regime != "FLAT"
        mk_long_ok  = mk_long_dir  & not_flat
        mk_short_ok = mk_short_dir & not_flat
    else:  # strong
        mk_long_ok  = regime == "STRONG_UP"
        mk_short_ok = regime == "STRONG_DOWN"

    warmup = max(K3_LEN, MK_NORM_LOOKBACK) + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry  = raw_long_entry  & mk_long_ok  & mask
    short_entry = raw_short_entry & mk_short_ok & mask

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    df["tp_offset"]   = TP_DIST
    df["sl_offset"]   = SL_BRICKS * BRICK_SIZE
    return df
