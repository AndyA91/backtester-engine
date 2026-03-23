"""
Keltner Channels — ATR-Based Volatility Envelope

ATR-based bands around an EMA midline. Unlike Bollinger Bands (std-dev based),
Keltner Channels use Average True Range, making them more responsive to
volatility shifts and less susceptible to outlier bars.

Matches TradingView's built-in Keltner Channel indicator:
  Mid   = EMA(close, period)
  Upper = Mid + mult * ATR(period)
  Lower = Mid - mult * ATR(period)

Usage:
    from indicators.keltner import calc_keltner

    result = calc_keltner(df, period=20, mult=1.5, atr_period=10)
    # result["mid"]   — EMA midline
    # result["upper"] — upper channel
    # result["lower"] — lower channel
    # result["bw"]    — bandwidth: (upper - lower) / mid

Interpretation:
    Price above upper → strong uptrend or overbought
    Price below lower → strong downtrend or oversold
    Price crosses back inside from outside → mean-reversion signal
    Bandwidth contracting → volatility squeeze (breakout incoming)
    Best paired with ADX < 25 for mean-reversion fading of channel touches
"""

import numpy as np
import pandas as pd
from indicators.ema import calc_ema
from indicators.atr import calc_atr


def calc_keltner(
    df: pd.DataFrame,
    period: int = 20,
    mult: float = 1.5,
    atr_period: int = 10,
    use_true_range: bool = True,
) -> dict:
    """
    Parameters
    ----------
    df             : DataFrame with 'High', 'Low', 'Close'
    period         : EMA period for midline (default 20)
    mult           : ATR multiplier for channel width (default 1.5)
    atr_period     : ATR lookback period (default 10)
    use_true_range : True = ATR (includes gaps), False = simple HL range

    Returns
    -------
    dict with keys: mid, upper, lower, bw (all numpy arrays)
    """
    mid = calc_ema(df["Close"], length=period).values

    if use_true_range:
        atr_vals = calc_atr(df, period=atr_period, method="rma")["atr"]
    else:
        atr_vals = (df["High"].values - df["Low"].values)
        atr_s = pd.Series(atr_vals)
        atr_vals = atr_s.rolling(atr_period).mean().values

    upper = mid + mult * atr_vals
    lower = mid - mult * atr_vals

    with np.errstate(invalid="ignore", divide="ignore"):
        bw = np.where(mid > 0, (upper - lower) / mid, np.nan)

    return {
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "bw": bw,
    }
