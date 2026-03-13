"""
Renko indicator enrichment.

Computes a standard set of indicators on the Renko brick DataFrame and
adds them as pre-shifted columns. "Pre-shifted" means each column at row i
contains the indicator value computed through bar i-1 — safe to use in a
signal loop without additional shifting (matches Pitfall #7 convention).

All indicators use brick OHLCV data (not source-chart candles), so they
operate in "brick space" — e.g. ADX reflects trend strength in terms of
brick direction consistency, RSI reflects brick close momentum.

Columns added to df:
    adx         ADX(14)            — trend strength 0-100
    plus_di     +DI(14)            — bullish directional pressure
    minus_di    -DI(14)            — bearish directional pressure
    rsi         RSI(14)            — brick close momentum 0-100
    macd        MACD line          — EMA(12) - EMA(26) of brick close
    macd_sig    Signal line        — EMA(9) of MACD line
    macd_hist   Histogram          — macd - macd_sig
    ema9        EMA(9) close       — fast trend direction
    ema21       EMA(21) close      — slow trend direction
    ema50       EMA(50) close      — medium trend direction
    ema200      EMA(200) close     — long-term trend direction
    atr         ATR(14)            — brick volatility (useful for wicked bricks)
    vol_ema     EMA(20) vol        — smoothed volume baseline
    vol_ratio   vol / vol_ema      — volume spike ratio (>1.5 = elevated)
    chop        CHOP(14)           — choppiness index (<38.2 trend, >61.8 chop)
    st_dir      Supertrend(10,3)   — direction +1 (up) / -1 (down)
    bb_upper    BB(20,2) upper     — upper band
    bb_lower    BB(20,2) lower     — lower band
    bb_mid      BB(20,2) mid       — middle band (SMA20)
    bb_bw       BB bandwidth       — (upper-lower)/mid, volatility proxy
    bb_pct_b    BB %B              — position within bands (0=lower, 1=upper)
    kama        KAMA(10) close     — adaptive moving average
    kama_slope  KAMA slope         — kama[i] - kama[i-1], momentum direction
    cmf         CMF(20)            — Chaikin Money Flow (-1 to +1)
    mfi         MFI(14)            — Money Flow Index 0-100 (vol-weighted RSI)
    obv         OBV                — On-Balance Volume (cumulative)
    obv_ema     OBV EMA(21)        — smoothed OBV signal
    psar_dir    PSAR direction     — +1 (bullish) / -1 (bearish)
    stoch_k     Stochastic %K(14,3)— slow %K oscillator 0-100
    stoch_d     Stochastic %D(3)   — signal line of %K 0-100
    sq_momentum Squeeze momentum   — LazyBear TTM momentum value
    sq_on       Squeeze active     — True when BB inside Keltner (coiling)

Usage in a strategy:
    # All columns are pre-shifted — use directly at index i in signal loop
    adx_val  = df["adx"].values      # ADX at bar i = computed through bar i-1
    rsi_val  = df["rsi"].values
    macd_val = df["macd"].values
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.adx         import calc_adx
from indicators.rsi         import calc_rsi
from indicators.macd        import calc_macd
from indicators.atr         import calc_atr
from indicators.ema         import calc_ema
from indicators.chop        import calc_chop
from indicators.supertrend  import calc_supertrend
from indicators.bbands      import calc_bbands
from indicators.kama        import calc_kama
from indicators.cmf         import calc_cmf
from indicators.mfi         import calc_mfi
from indicators.obv         import calc_obv
from indicators.parabolic_sar import calc_psar
from indicators.stochastic  import calc_stochastic
from indicators.squeeze     import calc_squeeze


def add_renko_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach standard indicators to a Renko brick DataFrame.

    All output columns are shifted by 1 bar so they can be used directly
    in a bar-loop at index i without additional shifting (Pitfall #7).

    Args:
        df: Renko DataFrame from load_renko_export() with OHLCV + brick_up.

    Returns:
        df with indicator columns added in-place (also returned for chaining).
    """
    # ── ADX / DI (multi-input: shift output, not input — Pitfall #7) ─────────
    adx_result = calc_adx(df, di_period=14, adx_period=14)
    df["adx"]      = pd.Series(adx_result["adx"],      index=df.index).shift(1).values
    df["plus_di"]  = pd.Series(adx_result["plus_di"],  index=df.index).shift(1).values
    df["minus_di"] = pd.Series(adx_result["minus_di"], index=df.index).shift(1).values

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    rsi_result = calc_rsi(df, period=14, source="close")
    df["rsi"] = pd.Series(rsi_result["rsi"], index=df.index).shift(1).values

    # ── MACD(12,26,9) ─────────────────────────────────────────────────────────
    macd_result = calc_macd(df)
    df["macd"]      = pd.Series(macd_result["macd"],      index=df.index).shift(1).values
    df["macd_sig"]  = pd.Series(macd_result["signal"],    index=df.index).shift(1).values
    df["macd_hist"] = pd.Series(macd_result["histogram"], index=df.index).shift(1).values

    # ── EMA(9), EMA(21), EMA(50), EMA(200) on brick close ────────────────────
    df["ema9"]   = calc_ema(df["Close"], length=9).shift(1).values
    df["ema21"]  = calc_ema(df["Close"], length=21).shift(1).values
    df["ema50"]  = calc_ema(df["Close"], length=50).shift(1).values
    df["ema200"] = calc_ema(df["Close"], length=200).shift(1).values

    # ── ATR(14) (multi-input: shift output) ──────────────────────────────────
    atr_result = calc_atr(df, period=14)
    df["atr"] = pd.Series(atr_result["atr"], index=df.index).shift(1).values

    # ── Volume ratio (vol / EMA20 of vol) ────────────────────────────────────
    # Volume on Renko = number of source bars per brick (meaningful signal)
    vol = df["Volume"].astype(float)
    vol_ema = calc_ema(vol, length=20)
    df["vol_ema"]   = vol_ema.shift(1).values
    df["vol_ratio"] = (vol.shift(1) / vol_ema.shift(1)).values

    # ── CHOP(14) — regime filter ──────────────────────────────────────────────
    chop_result = calc_chop(df, period=14)
    df["chop"] = pd.Series(chop_result["chop"], index=df.index).shift(1).values

    # ── Supertrend(10, 3.0) — trend direction ────────────────────────────────
    st_result = calc_supertrend(df, period=10, multiplier=3.0)
    df["st_dir"] = pd.Series(st_result["direction"], index=df.index).shift(1).values

    # ── Bollinger Bands(20, 2.0) ──────────────────────────────────────────────
    bb_result = calc_bbands(df, period=20, mult=2.0)
    df["bb_upper"]  = pd.Series(bb_result["upper"], index=df.index).shift(1).values
    df["bb_lower"]  = pd.Series(bb_result["lower"], index=df.index).shift(1).values
    df["bb_mid"]    = pd.Series(bb_result["mid"],   index=df.index).shift(1).values
    df["bb_bw"]     = pd.Series(bb_result["bw"],    index=df.index).shift(1).values
    df["bb_pct_b"]  = pd.Series(bb_result["pct_b"], index=df.index).shift(1).values

    # ── KAMA(10) + slope ──────────────────────────────────────────────────────
    kama_series = calc_kama(df["Close"], length=10)
    df["kama"]       = kama_series.shift(1).values
    df["kama_slope"] = kama_series.diff().shift(1).values

    # ── CMF(20) — Chaikin Money Flow ──────────────────────────────────────────
    cmf_result = calc_cmf(df, period=20)
    df["cmf"] = pd.Series(cmf_result["cmf"], index=df.index).shift(1).values

    # ── MFI(14) — Money Flow Index ────────────────────────────────────────────
    mfi_result = calc_mfi(df, period=14)
    df["mfi"] = pd.Series(mfi_result["mfi"], index=df.index).shift(1).values

    # ── OBV + OBV EMA(21) ────────────────────────────────────────────────────
    obv_result = calc_obv(df, ema_period=21)
    df["obv"]     = pd.Series(obv_result["obv"],     index=df.index).shift(1).values
    df["obv_ema"] = pd.Series(obv_result["obv_ema"], index=df.index).shift(1).values

    # ── Parabolic SAR direction (multi-input: shift output) ───────────────────
    psar_result = calc_psar(df)
    df["psar_dir"] = pd.Series(psar_result["direction"], index=df.index).shift(1).values

    # ── Stochastic(14, 3, 3) (multi-input: shift output) ─────────────────────
    stoch_result = calc_stochastic(df, k_period=14, smooth_k=3, smooth_d=3)
    df["stoch_k"] = pd.Series(stoch_result["slow_k"], index=df.index).shift(1).values
    df["stoch_d"] = pd.Series(stoch_result["pct_d"],  index=df.index).shift(1).values

    # ── Squeeze Momentum (LazyBear TTM) ───────────────────────────────────────
    sq_result = calc_squeeze(df)
    df["sq_momentum"] = pd.Series(sq_result["momentum"],   index=df.index).shift(1).values
    df["sq_on"]       = pd.Series(sq_result["squeeze_on"], index=df.index).shift(1).values

    return df
