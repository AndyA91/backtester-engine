"""
R008-Renko: Stochastic + CCI Double Mean-Reversion

Entry when both Stochastic %K and CCI agree on oversold/overbought reversal.
Dual confirmation should filter false reversals.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.stochastic import calc_stochastic
from indicators.cci import calc_cci

DESCRIPTION = "Stochastic %K + CCI dual oversold/overbought reversal"

HYPOTHESIS = (
    "Single oscillator reversals are noisy. Requiring both Stochastic and CCI "
    "to confirm exhaustion simultaneously should filter false signals. CCI is "
    "unbounded (captures momentum strength) while Stochastic is bounded (captures "
    "position within range). The combination captures different exhaustion dimensions."
)

PARAM_GRID = {
    "stoch_k":      [9, 14],
    "cci_period":   [14, 20],
    "stoch_os":     [20, 25],
    "cci_threshold": [100, 150],
    "cooldown":     [3, 6],
}


def generate_signals(df, stoch_k=14, cci_period=20, stoch_os=20, cci_threshold=100, cooldown=6):
    close = df["Close"].values
    n = len(close)

    stoch = calc_stochastic(df, k_period=stoch_k, smooth_k=3, smooth_d=3)
    sk = pd.Series(stoch["slow_k"]).shift(1).values

    cci = calc_cci(df, period=cci_period)["cci"]
    cci_s = np.empty(n)
    cci_s[0] = np.nan
    cci_s[1:] = cci[:-1]

    stoch_ob = 100 - stoch_os  # mirror: if os=20, ob=80

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(stoch_k, cci_period) + 5

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(sk[i]) or np.isnan(cci_s[i]):
            continue
        if i < 2 or np.isnan(sk[i - 1]) or np.isnan(cci_s[i - 1]):
            continue

        # Stochastic crosses
        sk_cross_up = sk[i - 1] <= stoch_os and sk[i] > stoch_os
        sk_cross_dn = sk[i - 1] >= stoch_ob and sk[i] < stoch_ob

        # CCI zones
        cci_oversold = cci_s[i] < -cci_threshold
        cci_overbought = cci_s[i] > cci_threshold

        # Long: stoch crosses up from oversold AND CCI is oversold
        long_cond = sk_cross_up and cci_oversold
        # Short: stoch crosses down from overbought AND CCI is overbought
        short_cond = sk_cross_dn and cci_overbought

        # Exit: stochastic returns to neutral
        lx[i] = sk[i] >= 50
        sx[i] = sk[i] <= 50

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if long_cond:
                le[i] = True
                last_trade_bar = i
            elif short_cond:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
