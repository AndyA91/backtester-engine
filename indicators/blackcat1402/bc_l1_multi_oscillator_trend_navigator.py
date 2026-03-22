"""
Python translation of:
  [blackcat] L1 Multi-Oscillator Trend Navigator by blackcat1402
  https://www.tradingview.com/script/nHtDINtI-blackcat-L1-Multi-Oscillator-Trend-Navigator/

Combines four oscillators (price position, stochastic, MA-squared, normalised
price) with CCI-family components to produce two composite lines (DX, ZX) and
market-strength-gated buy/sell signals.

Pine equivalence notes
----------------------
avgDev (Mean Absolute Deviation):
  Pine loops over `length` bars: sum(abs(x[i] - mean)) / length.
  Python: rolling(length).apply(lambda x: mean(abs(x - mean(x))), raw=True)

DMA (Dynamic Moving Average):
  dmaValue = alpha * source + (1-alpha) * dmaValue[1]  where alpha = clamp(weight, 0,1)
  Pine's weight = volume / max(volume, 1):
    - volume >= 1 → weight = 1.0 → DMA degenerates to source (identity)
    - volume == 0 → weight = 0.0 → DMA holds previous value
  Python: computed bar-by-bar via a simple loop (unavoidable for state-dependent alpha).

stochSmoothed:
  Pine: ta.ema(0.667 * stochEmaDouble[1] + 0.333 * stochEmaDouble, 2)
  The source is a blend of the *previous* stochEmaDouble and current — NOT recursive.
  Python: blend the shifted series, then apply ewm(span=2).

shiftedOscillator:
  Pine: ta.ema(normalizedEma1, 13) + 200 - 100  = normalizedEma2 + 100

Signals:
  weakMarketCount = rolling sum of (marketStrength < threshold) over 3 bars
  buySignal  = crossover(dxLine, zxLine) AND weakMarketCount >= 1
             OR crossover(dxLine, 0)    AND weakMarketCount >= 1
  sellSignal = crossover(zxLine, dxLine) AND zxLine > 60

Output columns added to df
--------------------------
  bc_motn_dx          — DX composite line
  bc_motn_zx          — ZX composite line
  bc_motn_market_str  — Market strength ratio
  bc_motn_buy_level   — Dynamic buy level (-5 weak / -25 strong market)
  bc_motn_buy_signal  — bool: buy crossover with weak market filter
  bc_motn_sell_signal — bool: ZX crosses above DX and ZX > 60

Usage
-----
  from indicators.blackcat1402.bc_l1_multi_oscillator_trend_navigator import (
      calc_bc_multi_oscillator_trend_navigator
  )
  df = calc_bc_multi_oscillator_trend_navigator(df)
"""

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _mad(series: pd.Series, length: int) -> pd.Series:
    """Rolling Mean Absolute Deviation — matches Pine's avgDev function."""
    return series.rolling(length).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )


def _dma(source: pd.Series, weight: pd.Series) -> pd.Series:
    """
    Dynamic Moving Average with per-bar alpha = clamp(weight, 0, 1).
    Stateful — requires a loop (alpha changes each bar).
    """
    src  = source.values
    w    = np.clip(weight.values, 0.0, 1.0)
    out  = np.empty(len(src))
    out[0] = src[0]
    for i in range(1, len(src)):
        out[i] = w[i] * src[i] + (1.0 - w[i]) * out[i - 1]
    return pd.Series(out, index=source.index)


def _crossover(a: pd.Series, b) -> pd.Series:
    """a crosses above b (scalar or series)."""
    if isinstance(b, (int, float)):
        b_series = pd.Series(b, index=a.index)
    else:
        b_series = b
    return (a.shift(1) <= b_series.shift(1)) & (a > b_series)


# ── Public function ───────────────────────────────────────────────────────────

def calc_bc_multi_oscillator_trend_navigator(
    df: pd.DataFrame,
    weak_market_threshold: float = 1.04,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L1 Multi-Oscillator Trend Navigator.

    Parameters
    ----------
    df                     : DataFrame with columns High, Low, Close, Volume
    weak_market_threshold  : market strength ratio below which market is 'weak'
                             (default 1.04, matches Pine default)

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high   = df["High"]
    low    = df["Low"]
    close  = df["Close"]
    volume = df["Volume"]

    # ── Oscillator 1: Price position (0-200 scale) ───────────────────────────
    h100 = high.rolling(100).max()
    l100 = low.rolling(100).min()
    r100 = h100 - l100
    price_pos200 = np.where(r100 > 0, (close - l100) / r100 * 200.0, 100.0)
    price_pos200 = pd.Series(price_pos200, index=df.index)

    sp1 = price_pos200.rolling(3).mean()   # smoothedPosition1
    sp2 = sp1.rolling(3).mean()            # smoothedPosition2
    ms1 = 3.0 * sp1 - 2.0 * sp2           # momentumSignal1

    # ── Oscillator 2: Stochastic EMA double ──────────────────────────────────
    h5 = high.rolling(5).max()
    l5 = low.rolling(5).min()
    r5 = h5 - l5
    stoch5 = np.where(r5 > 0, (close - l5) / r5 * 100.0, 50.0)
    stoch5 = pd.Series(stoch5, index=df.index)

    stoch_ema_double = _ema(stoch5, 4) * 2.0

    # stochSmoothed source: 0.667 * prev_stochEmaDouble + 0.333 * stochEmaDouble
    stoch_ema_prev   = stoch_ema_double.shift(1).fillna(0.0)
    stoch_sm_src     = 0.667 * stoch_ema_prev + 0.333 * stoch_ema_double
    stoch_smoothed   = _ema(stoch_sm_src, 2)
    ms2              = 3.0 * stoch_ema_double - 2.0 * stoch_smoothed

    # ── Oscillator 3: MA-squared normalized ──────────────────────────────────
    sma5_c = close.rolling(5).mean()
    sma5_l = low.rolling(5).mean()
    sma5_h = high.rolling(5).mean()

    ma_sq_plus  = sma5_c ** 2 + sma5_c
    low_ma_sq   = sma5_l ** 2 + sma5_l
    high_ma_sq  = sma5_h ** 2 + sma5_h

    hh_sq = high_ma_sq.rolling(64).max()
    ll_sq = low_ma_sq.rolling(64).min()
    r_sq  = hh_sq - ll_sq

    norm_range = np.where(r_sq > 0, (ma_sq_plus - ll_sq) / r_sq * 150.0 + 55.0, 55.0)
    norm_range = pd.Series(norm_range, index=df.index)

    scaled_osc      = norm_range.rolling(3).mean() * 1.5 - 100.0
    scaled_smoothed = scaled_osc.rolling(3).mean()
    ms3             = 3.0 * scaled_osc - 2.0 * scaled_smoothed

    # ── Oscillator 4: Normalised price ───────────────────────────────────────
    wp    = (high + low + close * 2.0) / 4.0
    wp_e  = _ema(wp, 13)
    wp_sd = wp.rolling(13).std(ddof=0)

    norm_p   = np.where(wp_sd > 0, (wp - wp_e) * 100.0 / wp_sd, 0.0)
    norm_p   = pd.Series(norm_p, index=df.index)
    norm_e1  = _ema(norm_p, 5)
    norm_e2  = _ema(norm_e1, 13)                  # normalizedEma2

    shifted_osc = norm_e2 + 100.0                 # shiftedOscillator = ema(normE1,13) + 100
    shifted_ema = _ema(shifted_osc, 10)
    ms4         = 3.0 * shifted_osc - 2.0 * shifted_ema

    # ── Composite signals ─────────────────────────────────────────────────────
    comp1 = sp1 + stoch_ema_double + scaled_osc + shifted_osc
    comp2 = sp2 + stoch_smoothed   + scaled_smoothed + shifted_ema
    comp3 = ms1 + ms2 + ms3 + ms4

    # ── CCI family ────────────────────────────────────────────────────────────
    typ = (high + low + close) / 3.0
    # avgPrice ≈ close (Pine: close*volume/volume — identity when volume>0)
    avg_price = close.copy()

    vwap5  = (close * volume).rolling(5).sum() / volume.rolling(5).sum()

    cci34  = (typ - typ.rolling(34).mean()) / (0.015 * _mad(typ, 34))
    cci24  = (typ - typ.rolling(24).mean()) / (0.015 * _mad(typ, 24))
    cci21  = (avg_price - avg_price.rolling(21).mean()) / (0.015 * _mad(avg_price, 21))
    cci90  = (vwap5 - vwap5.rolling(90).mean()) / (0.015 * _mad(vwap5, 90))

    norm_typ_e  = _ema(norm_p, 5)               # normalizedEma5 (from oscillator 4 norm_p)
    comp_cci    = (cci90 + cci21 + norm_typ_e + cci24 + cci34) / 5.0

    # Correct normalizedEma5: use normalizedTypical, not norm_p reused
    norm_typ_24  = (typ - _ema(typ, 24)) / typ.rolling(24).std(ddof=0) * 100.0
    norm_ema5_v2 = _ema(norm_typ_24, 5)
    comp_cci     = (cci90 + cci21 + norm_ema5_v2 + cci24 + cci34) / 5.0
    comp_cci_scaled = (comp_cci + 165.0) / 4.0

    # ── DMA (dynamic moving average on close) ────────────────────────────────
    # weight = volume / max(volume, 1): effectively 1.0 for bars with volume
    dyn_weight = volume / volume.clip(lower=1.0)
    dma_close  = _dma(close, dyn_weight)

    triple_ema_cci = _ema(_ema(_ema(comp_cci_scaled, 15), 10), 5)
    dyn_ma_signal  = (triple_ema_cci - 25.0) * 2.0 + dma_close

    # ── Weighted composite ────────────────────────────────────────────────────
    weighted_comp = (comp1 + comp2 + comp3 * 30.0) / 32.0 / 5.0 - 15.0
    weighted_e13  = _ema(weighted_comp, 13)

    dx_signal = (comp_cci_scaled + weighted_comp) / 2.0 / 1.100
    combined  = (dyn_ma_signal + weighted_e13) / 2.0
    # zxRaw = ema(combined, 1) = identity
    zx_line   = _ema(combined, 2)
    dx_line   = dx_signal

    # ── Market strength ───────────────────────────────────────────────────────
    price_ratio50  = close * 0.5
    ma_ratio30     = price_ratio50.rolling(30).mean() * 0.830
    market_strength = (price_ratio50 / ma_ratio30).rolling(3).mean()

    buy_level = np.where(market_strength < weak_market_threshold, -5.0, -25.0)
    buy_level = pd.Series(buy_level, index=df.index)

    # ── Signal generation ─────────────────────────────────────────────────────
    weak_market_count = (market_strength < weak_market_threshold).astype(int).rolling(3).sum()

    buy_cond1  = _crossover(dx_line, zx_line) & (weak_market_count >= 1)
    buy_cond2  = _crossover(dx_line, pd.Series(0.0, index=df.index)) & (weak_market_count >= 1)
    buy_signal = buy_cond1 | buy_cond2

    sell_signal = (
        _crossover(zx_line, dx_line) & (zx_line > 60)
    )

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_motn_dx"]          = dx_line
    df["bc_motn_zx"]          = zx_line
    df["bc_motn_market_str"]  = market_strength
    df["bc_motn_buy_level"]   = buy_level
    df["bc_motn_buy_signal"]  = buy_signal
    df["bc_motn_sell_signal"] = sell_signal
    return df
