"""
USO002: KAMA Ribbon 3L — Long-Only Channel-Breakout Stacked Variant

Stock variant of R033 with channel-breakout entry semantics:

  1. LONG ONLY (no shorts — required by stocks/runner.py).
  2. CHANNEL-BREAKOUT ENTRIES — entries fire when bull alignment (k1>k2>k3)
     holds AND Close crosses ABOVE the top of the KAMA channel (= k1, the
     fast KAMA, which is the highest line in a bull-aligned ribbon).

       • First entry  : alignment is true and price punches above k1.
       • Re-entries   : after a fill, price must dip back INTO the channel
                        (close <= k1) and then break OUT again. Each
                        crossover counts as one stacked add.

     Crossover semantics enforce the "back-inside-then-out" rule
     automatically — no cooldown variable needed. The engine caps total
     stacked adds at PYRAMIDING.
  3. SLOW-KAMA EXIT — long_exit fires when Close drops below the slow KAMA
     (K3=60). engine.run_backtest treats long_exit as strategy.close_all,
     so all stacked sub-positions exit on the next bar's Open in one shot.

Optional R033 filters (distance from K3, MK v4 momentum) are kept so the
sweep can decide whether they help on stocks.

Tested first on USO 0.25 brick. KAMA periods are LOCKED at 5/13/60.
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama
from indicators.momentum_king_v4 import calc_momentum_king_v4

DESCRIPTION = "Long-only KAMA 3L channel-breakout stack, exit-all on slow-KAMA cross"

HYPOTHESIS = (
    "Within a confirmed bull regime (k1>k2>k3), each fresh breakout above the "
    "top of the KAMA channel (k1) signals a renewed thrust after a pullback "
    "into the ribbon. Stacking a position on every such breakout lets us scale "
    "in only on demonstrated continuation, not on stale alignment. Exiting the "
    "whole stack on Close < K3 keeps the asymmetry: many small adds at thrust "
    "moments, one decisive exit on regime break."
)

# ── Stacking / pyramiding cap (read by stocks/runner.py) ─────────────────────
PYRAMIDING = 5

# ── Locked KAMA params (visual choice from R033 Pine) ─────────────────────────
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
RENKO_FILE      = "BATS_USO, 1S renko 0.25.csv"
BRICK_SIZE      = 0.25  # USO 0.25 chart

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
    Long-only channel-breakout stacked signals for the stocks runner.

    Entry: bull alignment (k1>k2>k3) AND Close crosses ABOVE k1 (top of
           channel). Crossover semantics enforce "must dip back inside k1
           before next add". Engine caps stack at PYRAMIDING.
    Exit:  Close < slow KAMA (K3) — closes ALL stacked sub-positions.
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

    # Channel-breakout: close crosses above k1 (channel top in bull alignment)
    close_prev = np.roll(close_a, 1); close_prev[0] = close_a[0]
    k1_prev    = np.roll(k1, 1);      k1_prev[0]    = k1[0]
    crossover_k1 = (close_a > k1) & (close_prev <= k1_prev) & ~np.isnan(k1) & ~np.isnan(k1_prev)

    # Distance filter (price not too far above K3)
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
