"""
Ichimoku Cloud (Ichimoku Kinko Hyo)

Multi-component trend/support/resistance system widely used on TradingView.
Matches Pine's built-in Ichimoku Cloud indicator exactly.

Components:
    Tenkan-sen  (Conversion): midpoint of (highest high + lowest low) over tenkan_period [9]
    Kijun-sen   (Base):       midpoint of (highest high + lowest low) over kijun_period  [26]
    Senkou A    (Cloud front): (Tenkan + Kijun) / 2, plotted kijun_period bars ahead
    Senkou B    (Cloud back):  midpoint over senkou_b_period [52], plotted kijun_period bars ahead
    Chikou Span (Lagging):    today's close plotted kijun_period bars back

Usage:
    from indicators.ichimoku import calc_ichimoku

    result = calc_ichimoku(df)
    # result["tenkan"]     — Tenkan-sen
    # result["kijun"]      — Kijun-sen
    # result["senkou_a"]   — Senkou Span A (aligned to current bar, 26 bars future)
    # result["senkou_b"]   — Senkou Span B (aligned to current bar, 26 bars future)
    # result["chikou"]     — Chikou Span  (close shifted 26 bars back in time)
    # result["cloud_bull"] — bool array: senkou_a > senkou_b (bullish cloud)

Note on future displacement:
    Pine plots Senkou A/B 26 bars into the future. In backtesting these are
    returned aligned to their COMPUTATION bar (not the future plot bar).
    To check whether current price is above/below the cloud, compare close[i]
    to senkou_a[i - kijun_period] and senkou_b[i - kijun_period].

Interpretation:
    Price above cloud      → bullish bias
    Price below cloud      → bearish bias
    Price inside cloud     → consolidation / no clear bias
    Tenkan crosses Kijun   → trend signal (tk cross)
    Chikou above price     → long-term bullish confirmation
    Cloud twist (A crosses B) → potential trend change ahead
"""

import numpy as np
import pandas as pd


def calc_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> dict:
    """
    Parameters
    ----------
    df              : DataFrame with 'High', 'Low', 'Close'
    tenkan_period   : Conversion line period (default 9)
    kijun_period    : Base line period (default 26)
    senkou_b_period : Slow cloud line period (default 52)

    Returns
    -------
    dict with keys: tenkan, kijun, senkou_a, senkou_b, chikou, cloud_bull
    All arrays have the same length as df.
    senkou_a / senkou_b are at current index (i.e. what will be PLOTTED 26 bars ahead).
    """
    high  = pd.Series(df["High"].values)
    low   = pd.Series(df["Low"].values)
    close = df["Close"].values
    n     = len(close)

    def _midpoint(period: int) -> np.ndarray:
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return ((hh + ll) / 2.0).values

    tenkan  = _midpoint(tenkan_period)
    kijun   = _midpoint(kijun_period)
    senkou_b = _midpoint(senkou_b_period)

    senkou_a = (tenkan + kijun) / 2.0

    # Chikou: close shifted kijun_period bars into the past (look-back display)
    chikou = np.full(n, np.nan)
    chikou[: n - kijun_period] = close[kijun_period:]

    cloud_bull = senkou_a > senkou_b

    return {
        "tenkan":     tenkan,
        "kijun":      kijun,
        "senkou_a":   senkou_a,
        "senkou_b":   senkou_b,
        "chikou":     chikou,
        "cloud_bull": cloud_bull,
    }


def price_vs_cloud(df: pd.DataFrame, result: dict, kijun_period: int = 26) -> np.ndarray:
    """
    Convenience: returns +1 (above cloud), -1 (below cloud), 0 (inside cloud).

    Accounts for the kijun_period forward displacement of the cloud lines.
    Price at bar i is compared to senkou_a/b computed at bar i - kijun_period.

    Parameters
    ----------
    df           : original DataFrame (for length)
    result       : output of calc_ichimoku()
    kijun_period : must match the period used in calc_ichimoku()
    """
    close    = df["Close"].values
    sa       = result["senkou_a"]
    sb       = result["senkou_b"]
    n        = len(close)
    position = np.zeros(n, dtype=int)

    for i in range(kijun_period, n):
        cloud_top    = max(sa[i - kijun_period], sb[i - kijun_period])
        cloud_bottom = min(sa[i - kijun_period], sb[i - kijun_period])
        if close[i] > cloud_top:
            position[i] = 1
        elif close[i] < cloud_bottom:
            position[i] = -1
        # else 0 = inside

    return position
