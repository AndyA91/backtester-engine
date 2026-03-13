"""
RXXX: Strategy Name

One-line description of the signal logic.

Compare/contrast with related strategies if applicable.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "short description for sweep output"

HYPOTHESIS = (
    "Why should this edge exist? What market behavior does it exploit?"
)

PARAM_GRID = {
    "param_a": [1, 2, 3],
    "param_b": [10, 20, 30],
}

# ── Indicator column reference (all pre-shifted, use at [i] directly) ─────────
#
#  TREND DIRECTION
#    st_dir       Supertrend(10,3)   +1=bullish, -1=bearish
#    psar_dir     Parabolic SAR      +1=bullish, -1=bearish
#    kama         KAMA(10)           adaptive MA level
#    kama_slope   KAMA slope         positive=rising, negative=falling
#    ema9/21/50/200  EMA levels      fast → long-term trend
#
#  TREND STRENGTH / REGIME
#    adx          ADX(14)            0-100, >25=trending
#    plus_di      +DI(14)            bullish directional pressure
#    minus_di     -DI(14)            bearish directional pressure
#    chop         CHOP(14)           <38.2=trending, >61.8=choppy
#
#  MOMENTUM / OSCILLATORS
#    rsi          RSI(14)            0-100
#    macd         MACD line          EMA12-EMA26
#    macd_sig     MACD signal        EMA9 of MACD
#    macd_hist    MACD histogram     macd - macd_sig
#    stoch_k      Stochastic %K(14,3) 0-100
#    stoch_d      Stochastic %D(3)   0-100 signal line
#    sq_momentum  TTM Squeeze        positive=bullish, negative=bearish
#    sq_on        Squeeze active     True=coiling (BB inside Keltner)
#
#  VOLATILITY / BANDS
#    atr          ATR(14)            brick volatility
#    bb_upper     BB(20,2) upper     resistance band
#    bb_lower     BB(20,2) lower     support band
#    bb_mid       BB(20,2) mid       SMA(20)
#    bb_bw        BB bandwidth       (upper-lower)/mid
#    bb_pct_b     BB %B              0=at lower, 1=at upper
#
#  VOLUME / MONEY FLOW
#    vol_ema      EMA(20) volume     smoothed baseline
#    vol_ratio    vol/vol_ema        >1.5=elevated, >2.0=spike
#    cmf          CMF(20)            -1 to +1, positive=buying pressure
#    mfi          MFI(14)            0-100, vol-weighted RSI
#    obv          OBV                cumulative volume trend
#    obv_ema      OBV EMA(21)        smoothed OBV signal
#
# ─────────────────────────────────────────────────────────────────────────────


def generate_signals(
    df: pd.DataFrame,
    param_a: int = 2,
    param_b: int = 10,
) -> pd.DataFrame:
    """
    Generate entry/exit signals.

    Args:
        df: Renko DataFrame with brick_up bool column + pre-shifted indicator columns.
        param_a: Description.
        param_b: Description.

    Returns:
        df with columns long_entry, long_exit, short_entry, short_exit (bool).
    """
    n = len(df)
    brick_up = df["brick_up"].values

    # Indicator arrays — pre-shifted, safe to use at [i] directly
    adx      = df["adx"].values
    st_dir   = df["st_dir"].values
    rsi      = df["rsi"].values
    # add more as needed...

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = 30  # adjust to max indicator warmup needed

    for i in range(warmup, n):
        # NaN guard for any indicator used as gate
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            continue

        # ── Exit (unconditional, no cooldown) ─────────────────────────────────
        long_exit[i]  = not brick_up[i]
        short_exit[i] = brick_up[i]

        # ── Entry ──────────────────────────────────────────────────────────────
        can_trade = (i - last_trade_bar) >= param_b
        if not can_trade:
            continue

        # TODO: replace with actual signal logic
        if brick_up[i] and adx[i] > 25:
            long_entry[i] = True
            last_trade_bar = i
        elif not brick_up[i] and adx[i] > 25:
            short_entry[i] = True
            last_trade_bar = i

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
