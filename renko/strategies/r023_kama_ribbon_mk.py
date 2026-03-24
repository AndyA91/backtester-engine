"""
R023: KAMA Ribbon + Momentum King — Trend alignment with adaptive momentum confirmation

Combines the best from V2 (KAMA ribbon alignment + ADX≥25) with the Momentum King
oscillator's regime classification as an additional entry/exit filter.

Entry: KAMA ribbon aligns + brick direction matches + ADX≥25 + MK regime confirms
Exit:  Ribbon alignment breaks OR MK regime turns against position

MK regime mapping: STRONG_UP/WEAK_UP → +1, FLAT → 0, STRONG_DOWN/WEAK_DOWN → -1
"""

import numpy as np
import pandas as pd

from indicators.kama import calc_kama
from indicators.momentum_king import calc_momentum_king

DESCRIPTION = "KAMA ribbon + Momentum King regime — dual-system trend confirmation"

HYPOTHESIS = (
    "KAMA ribbon alignment catches trend direction. Momentum King independently "
    "measures momentum strength via adaptive smoothing. Requiring both to agree "
    "should eliminate false entries where alignment forms but momentum is flat/opposing. "
    "MK regime as exit trigger may also catch momentum exhaustion before alignment breaks."
)

RIBBONS = {
    "3L_8_21_55":   (8, 21, 55),
    "3L_5_13_34":   (5, 13, 34),
}

PARAM_GRID = {
    "ribbon":           list(RIBBONS.keys()),
    "cooldown":         [5],
    "adx_gate":         [25],
    "mk_entry_mode":    ["strong_only", "any_up", "off"],
    "mk_exit_mode":     ["regime_flip", "regime_flat", "off"],
    "mk_ema_length":    [14, 21],
}

_KAMA_CACHE = {}
_MK_CACHE = {}


def _get_kama(close_series: pd.Series, length: int) -> np.ndarray:
    if length not in _KAMA_CACHE:
        _KAMA_CACHE[length] = calc_kama(close_series, length=length).shift(1).values
    return _KAMA_CACHE[length]


def _get_mk_regime(df: pd.DataFrame, ema_length: int) -> np.ndarray:
    """Get pre-shifted MK regime array (+1, 0, -1)."""
    if ema_length not in _MK_CACHE:
        mk = calc_momentum_king(df, ema_length=ema_length)
        regime_map = {
            "STRONG_UP": 2, "WEAK_UP": 1,
            "FLAT": 0,
            "WEAK_DOWN": -1, "STRONG_DOWN": -2,
        }
        regime_arr = np.array([regime_map.get(str(r), 0) for r in mk["regime"]])
        # Pre-shift by 1 bar (no lookahead)
        shifted = np.empty_like(regime_arr, dtype=np.float64)
        shifted[0] = np.nan
        shifted[1:] = regime_arr[:-1]
        _MK_CACHE[ema_length] = shifted
    return _MK_CACHE[ema_length]


def generate_signals(
    df: pd.DataFrame,
    ribbon: str = "3L_8_21_55",
    cooldown: int = 5,
    adx_gate: int = 25,
    mk_entry_mode: str = "strong_only",
    mk_exit_mode: str = "regime_flip",
    mk_ema_length: int = 14,
) -> pd.DataFrame:
    n = len(df)
    lengths = RIBBONS[ribbon]

    brick_up = df["brick_up"].values
    adx = df["adx"].values

    # KAMA arrays (pre-shifted)
    kama_fast = _get_kama(df["Close"], lengths[0])
    kama_mid  = _get_kama(df["Close"], lengths[1])
    kama_slow = _get_kama(df["Close"], lengths[2])

    # MK regime (pre-shifted): +2=STRONG_UP, +1=WEAK_UP, 0=FLAT, -1=WEAK_DOWN, -2=STRONG_DOWN
    mk_regime = _get_mk_regime(df, mk_ema_length)

    # Precompute alignment
    any_nan = np.isnan(kama_fast) | np.isnan(kama_mid) | np.isnan(kama_slow)
    bull_align = (kama_fast > kama_mid) & (kama_mid > kama_slow) & ~any_nan
    bear_align = (kama_fast < kama_mid) & (kama_mid < kama_slow) & ~any_nan

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(lengths) + 5

    for i in range(warmup, n):
        b_up = bool(brick_up[i])
        bull = bool(bull_align[i])
        bear = bool(bear_align[i])
        mk = mk_regime[i] if not np.isnan(mk_regime[i]) else 0

        # ── Exits ───────────────────────────────────────────────────────
        # Base exit: alignment break + opposing brick (from V1/V2)
        base_long_exit  = not bull or not b_up
        base_short_exit = not bear or b_up

        # MK exit modes
        if mk_exit_mode == "regime_flip":
            # Exit when MK regime flips to opposing direction
            mk_long_exit  = mk < 0   # MK turns bearish
            mk_short_exit = mk > 0   # MK turns bullish
            long_exit[i]  = base_long_exit or mk_long_exit
            short_exit[i] = base_short_exit or mk_short_exit
        elif mk_exit_mode == "regime_flat":
            # Exit when MK goes flat OR opposing
            mk_long_exit  = mk <= 0
            mk_short_exit = mk >= 0
            long_exit[i]  = base_long_exit or mk_long_exit
            short_exit[i] = base_short_exit or mk_short_exit
        else:
            # off — use base exits only
            long_exit[i]  = base_long_exit
            short_exit[i] = base_short_exit

        # ── Entries ─────────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= cooldown
        if not can_trade:
            continue

        bull_prev = bool(bull_align[i - 1])
        bear_prev = bool(bear_align[i - 1])

        long_trigger  = bull and not bull_prev and b_up
        short_trigger = bear and not bear_prev and not b_up

        if not long_trigger and not short_trigger:
            continue

        # ADX gate
        if not np.isnan(adx[i]) and adx[i] < adx_gate:
            continue

        # MK entry modes
        if mk_entry_mode == "strong_only":
            # Require STRONG regime match
            if long_trigger and mk != 2:
                continue
            if short_trigger and mk != -2:
                continue
        elif mk_entry_mode == "any_up":
            # Require any positive/negative regime (not flat)
            if long_trigger and mk <= 0:
                continue
            if short_trigger and mk >= 0:
                continue
        # else: "off" — no MK filter on entry

        if long_trigger:
            long_entry[i] = True
            last_trade_bar = i
        elif short_trigger:
            short_entry[i] = True
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
