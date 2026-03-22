"""
Python translation of:
  [blackcat] L1 Momentum Crossover Pro by blackcat1402
  https://www.tradingview.com/script/vJ6OqsGP-blackcat-L1-Momentum-Crossover-Pro/

Advanced oscillator that identifies momentum crossover buy/sell signals using
a DIF oscillator, White Out line, and Pink In signal line — all normalized
to a 0–100 scale.

Pine equivalence notes
----------------------
Zero-Lag EMA (zeroLagEma):
  ema1 = ta.ema(src, length)
  ema2 = ta.ema(ema1, length)   ← second EMA of the first EMA
  zlema = ema1 + (ema1 - ema2)
  Implemented as two sequential ewm(span=length, adjust=False) calls.

Custom smoothing (smoothValue):
  "EMA"  → ewm(span=period, adjust=False)
  "WMA"  → linearly-weighted rolling average (weights = 1..n)
  "ALMA" → Arnaud Legoux Moving Average (offset=0.85, sigma=6)

DIF line:
  normalizedPrice = (close - lowest(low,27)) / range * 100  (50 if range==0)
  emaNorm = zlema(normalizedPrice, 5) if use_zero_lag else ema(normalizedPrice, 5)
  emaOfEma = zlema(emaNorm, 3) if use_zero_lag else ema(emaNorm, 3)
  difRaw = 3 * emaNorm - 2 * emaOfEma
  difValue = smooth(difRaw, 4, smoothType)

White Out line:
  normalizedCloseHL = (close - lowest(low,33)) / range * 100  (50 if range==0)
  whiteOutRaw = zlema(normalizedCloseHL, 8) if use_zero_lag else ema(normalizedCloseHL, 8)
  whiteOutLine = smooth(whiteOutRaw, 4, smoothType)

Pink In line:
  highestWhiteOut = rolling max of whiteOutLine over 3 bars
  pinkInRaw = ema(highestWhiteOut, 1)  ← identity (EMA-1 = source)
  pinkInLine = smooth(pinkInRaw, 4, smoothType)

Signals:
  buySignal  = crossover(difValue, whiteOutLine) AND pinkInLine < 80 AND difValue rising
  sellSignal = crossunder(difValue, whiteOutLine) AND pinkInLine > 20 AND difValue falling

Output columns added to df
--------------------------
  bc_mcp_dif          — DIF oscillator (main line, 0–100 scale)
  bc_mcp_white_out    — White Out line (0–100 scale)
  bc_mcp_pink_in      — Pink In signal line (0–100 scale)
  bc_mcp_buy_signal   — bool: buy crossover signal
  bc_mcp_sell_signal  — bool: sell crossover signal
  bc_mcp_buy_prep     — bool: DIF <= oversold level (buy preparation zone)

Usage
-----
  from indicators.blackcat1402.bc_l1_momentum_crossover_pro import calc_bc_momentum_crossover_pro
  df = calc_bc_momentum_crossover_pro(df)
"""

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _zlema(series: pd.Series, length: int) -> pd.Series:
    """Zero-Lag EMA: ema1 + (ema1 - ema2)."""
    ema1 = _ema(series, length)
    ema2 = _ema(ema1, length)
    return ema1 + (ema1 - ema2)


def _wma(series: pd.Series, length: int) -> pd.Series:
    """Linearly-weighted moving average — matches Pine's ta.wma."""
    weights = np.arange(1, length + 1, dtype=float)
    w_sum   = weights.sum()
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / w_sum, raw=True
    )


def _alma(series: pd.Series, length: int, offset: float = 0.85, sigma: int = 6) -> pd.Series:
    """Arnaud Legoux Moving Average — matches Pine's ta.alma."""
    m = offset * (length - 1)
    s = length / sigma
    w = np.exp(-((np.arange(length) - m) ** 2) / (2.0 * s * s))
    w_sum = w.sum()
    return series.rolling(length).apply(
        lambda x: np.dot(x, w) / w_sum, raw=True
    )


def _smooth(series: pd.Series, period: int, smooth_type: str) -> pd.Series:
    if smooth_type == "WMA":
        return _wma(series, period)
    if smooth_type == "ALMA":
        return _alma(series, period)
    return _ema(series, period)  # default EMA


# ── Public function ───────────────────────────────────────────────────────────

def calc_bc_momentum_crossover_pro(
    df: pd.DataFrame,
    period_volatility: int   = 27,
    period_ema_fast:   int   = 5,
    period_ema_slow:   int   = 3,
    period_high_low:   int   = 33,
    period_white_out:  int   = 8,
    period_pink_high:  int   = 3,
    smooth_period:     int   = 4,
    smooth_type:       str   = "EMA",
    use_zero_lag:      bool  = True,
    oversold_level:    int   = 10,
) -> pd.DataFrame:
    """
    Python translation of [blackcat] L1 Momentum Crossover Pro.

    Parameters
    ----------
    df                : DataFrame with columns High, Low, Close
    period_volatility : lookback for DIF normalization range (default 27)
    period_ema_fast   : fast EMA period inside DIF (default 5)
    period_ema_slow   : slow EMA period inside DIF (default 3)
    period_high_low   : lookback for White Out normalization range (default 33)
    period_white_out  : EMA period for White Out (default 8)
    period_pink_high  : rolling max lookback for Pink In (default 3)
    smooth_period     : final smoothing period (default 4)
    smooth_type       : "EMA", "WMA", or "ALMA" (default "EMA")
    use_zero_lag      : use Zero-Lag EMA instead of plain EMA (default True)
    oversold_level    : DIF level below which buy preparation activates (default 10)

    Returns
    -------
    df with new columns appended (see module docstring).
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # ── DIF oscillator ────────────────────────────────────────────────────────
    h_vol = high.rolling(period_volatility).max()
    l_vol = low.rolling(period_volatility).min()
    r_vol = h_vol - l_vol
    norm_price = np.where(r_vol > 0, (close - l_vol) / r_vol * 100.0, 50.0)
    norm_price = pd.Series(norm_price, index=df.index)

    fn = _zlema if use_zero_lag else _ema
    ema_norm   = fn(norm_price, period_ema_fast)
    ema_of_ema = fn(ema_norm,   period_ema_slow)
    dif_raw    = 3.0 * ema_norm - 2.0 * ema_of_ema
    dif_value  = _smooth(dif_raw, smooth_period, smooth_type)

    # ── White Out line ────────────────────────────────────────────────────────
    h_hl = high.rolling(period_high_low).max()
    l_hl = low.rolling(period_high_low).min()
    r_hl = h_hl - l_hl
    norm_close_hl = np.where(r_hl > 0, (close - l_hl) / r_hl * 100.0, 50.0)
    norm_close_hl = pd.Series(norm_close_hl, index=df.index)

    white_out_raw  = fn(norm_close_hl, period_white_out)
    white_out_line = _smooth(white_out_raw, smooth_period, smooth_type)

    # ── Pink In signal line ───────────────────────────────────────────────────
    # ta.ema(x, 1) = x (identity), then smoothed
    highest_white_out = white_out_line.rolling(period_pink_high).max()
    pink_in_line      = _smooth(highest_white_out, smooth_period, smooth_type)

    # ── Signal detection ──────────────────────────────────────────────────────
    dif_prev        = dif_value.shift(1)
    white_out_prev  = white_out_line.shift(1)

    cross_up   = (dif_prev <= white_out_prev) & (dif_value > white_out_line)
    cross_down = (dif_prev >= white_out_prev) & (dif_value < white_out_line)

    dif_rising  = dif_value > dif_value.shift(1)
    dif_falling = dif_value < dif_value.shift(1)

    in_overbought = pink_in_line > 80
    in_oversold   = pink_in_line < 20

    buy_signal  = cross_up   & ~in_overbought & dif_rising
    sell_signal = cross_down & ~in_oversold   & dif_falling
    buy_prep    = dif_value <= oversold_level

    # ── Attach to DataFrame ───────────────────────────────────────────────────
    df = df.copy()
    df["bc_mcp_dif"]         = dif_value
    df["bc_mcp_white_out"]   = white_out_line
    df["bc_mcp_pink_in"]     = pink_in_line
    df["bc_mcp_buy_signal"]  = buy_signal
    df["bc_mcp_sell_signal"] = sell_signal
    df["bc_mcp_buy_prep"]    = buy_prep
    return df
