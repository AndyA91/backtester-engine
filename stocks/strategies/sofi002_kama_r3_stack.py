"""
SOFI002: KAMA Ribbon 3L — Long-Only Channel-Breakout Stacked Variant (SOFI 0.15)

Direct port of uso002_kama_r3_stack.py to SOFI 0.15-brick Renko data.
Same logic, only data file + brick size differ. See uso002 for full mechanic.

  - Entry: bull alignment (k1>k2>k3) AND ta.crossover(close, k1)
  - Exit:  Close < K3 → close_all (entire stack unwinds)
  - Stack: PYRAMIDING=5 sub-positions max
  - KAMA periods locked at 5/13/60

Sweep isolates the optional R033 filters (distance from K3, MK v4 momentum)
on a NEW instrument (SOFI is much higher-volatility, much smaller cap than
USO — different drift dynamics, useful as a cross-instrument R6 test).
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama
from indicators.momentum_king_v4 import calc_momentum_king_v4

DESCRIPTION = "SOFI 0.15 — KAMA 3L channel-breakout stack, exit-all on slow-KAMA cross"

HYPOTHESIS = (
    "USO smoke test showed the channel-breakout stack mechanic is structurally "
    "high-PF (~4.3) on commodity ETF data. SOFI is a different beast — small-cap "
    "fintech, persistent volatility, multi-year drawdowns and explosive rallies. "
    "If the same mechanic survives on SOFI without re-tuning KAMA periods, that's "
    "a strong cross-instrument generalization signal (meta_rules R6). If it dies, "
    "we learn the mechanic is regime-dependent and need to investigate which "
    "feature of USO it relied on."
)

# ── Stacking / pyramiding cap (read by stocks/runner.py) ─────────────────────
PYRAMIDING = 5

# ── Locked KAMA params (same as uso002 / R033) ───────────────────────────────
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

# ── Stocks runner config ─────────────────────────────────────────────────────
COMMISSION_PCT  = 0.0
INITIAL_CAPITAL = 10000.0
RENKO_FILE      = "BATS_SOFI, 1S renko 0.15 ETH.csv"
BRICK_SIZE      = 0.15  # SOFI 0.15 chart

# ── Sweep grid ────────────────────────────────────────────────────────────────
PARAM_GRID = {
    # Distance filter: max bricks above K3 at entry time (1000 = off).
    "max_dist_bricks": [4, 8, 16, 1000],
    # MK v4 mode: "off" / "direction" / "block_flat" / "strong"
    "mk_mode":         ["off", "direction", "block_flat"],
}

# ── Worker-level caches (shared across combos in the same process) ────────────
_KAMA_CACHE = {}
_MK_CACHE   = {}


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
        _MK_CACHE[key] = {"smoothed": smoothed, "regime": regime}
    return _MK_CACHE[key]


def generate_signals(
    df: pd.DataFrame,
    max_dist_bricks: int = 1000,
    mk_mode: str = "off",
) -> pd.DataFrame:
    """
    Long-only channel-breakout stacked signals — SOFI 0.15.

    Entry: bull alignment (k1>k2>k3) AND Close crosses above k1.
    Exit:  Close < K3 → close_all.
    """
    n = len(df)
    long_entry = np.zeros(n, dtype=bool)
    long_exit  = np.zeros(n, dtype=bool)

    if mk_mode not in ("off", "direction", "block_flat", "strong"):
        df["long_entry"] = long_entry
        df["long_exit"]  = long_exit
        return df

    close   = df["Close"]
    close_a = close.values

    k1 = _get_kama(close, K1_LEN)
    k2 = _get_kama(close, K2_LEN)
    k3 = _get_kama(close, K3_LEN)

    any_nan    = np.isnan(k1) | np.isnan(k2) | np.isnan(k3)
    bull_align = (k1 > k2) & (k2 > k3) & ~any_nan

    # Channel-breakout: close crosses above k1 (top of channel in bull alignment)
    close_prev = np.roll(close_a, 1); close_prev[0] = close_a[0]
    k1_prev    = np.roll(k1, 1);      k1_prev[0]    = k1[0]
    crossover_k1 = (close_a > k1) & (close_prev <= k1_prev) & ~np.isnan(k1) & ~np.isnan(k1_prev)

    # Distance filter
    max_dist_pr = max_dist_bricks * BRICK_SIZE
    long_dist_ok = (close_a - k3) <= max_dist_pr

    # MK v4 filter
    if mk_mode == "off":
        mk_long_ok = np.ones(n, dtype=bool)
    else:
        mk = _get_mk(df)
        smoothed = mk["smoothed"]
        regime   = mk["regime"]
        mk_long_dir = ~np.isnan(smoothed) & (smoothed > 0)
        if mk_mode == "direction":
            mk_long_ok = mk_long_dir
        elif mk_mode == "block_flat":
            mk_long_ok = mk_long_dir & (regime != "FLAT")
        else:  # strong
            mk_long_ok = regime == "STRONG_UP"

    # Slow-KAMA exit: close below K3 ⇒ exit-all
    long_exit = (~np.isnan(k3)) & (close_a < k3)

    warmup = max(K3_LEN, MK_NORM_LOOKBACK) + 5
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True

    long_entry = bull_align & crossover_k1 & long_dist_ok & mk_long_ok & mask

    df = df.copy()
    df["long_entry"] = long_entry
    df["long_exit"]  = long_exit
    return df
