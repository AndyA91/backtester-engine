"""
R009-Renko: Fisher Transform + Awesome Oscillator Momentum

Fisher Transform cross as entry signal, confirmed by AO color (momentum direction).
Fisher is excellent at identifying turning points; AO confirms momentum.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from indicators.fisher_transform import calc_fisher_transform
from indicators.awesome_oscillator import calc_ao

DESCRIPTION = "Fisher Transform cross confirmed by Awesome Oscillator momentum direction"

HYPOTHESIS = (
    "Fisher Transform normalizes price into a Gaussian distribution, making "
    "turning points statistically identifiable. AO measures acceleration of "
    "the midpoint SMA. Together: Fisher identifies the turn, AO confirms momentum "
    "is building in that direction. Already showed promise on BTC Renko."
)

PARAM_GRID = {
    "fisher_period": [9, 10, 14],
    "ao_fast":       [5, 8],
    "ao_slow":       [21, 34],
    "require_ao":    [True, False],  # False = Fisher-only baseline
    "cooldown":      [3, 6],
}


def generate_signals(df, fisher_period=10, ao_fast=5, ao_slow=34, require_ao=True, cooldown=6):
    close = df["Close"].values
    n = len(close)

    ft = calc_fisher_transform(df, period=fisher_period)
    fisher = pd.Series(ft["fisher"]).shift(1).values
    fisher_sig = pd.Series(ft["signal"]).shift(1).values

    ao_result = calc_ao(df, fast=ao_fast, slow=ao_slow)
    ao_color = pd.Series(ao_result["color"]).shift(1).values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)

    last_trade_bar = -999_999
    warmup = max(fisher_period, ao_slow) + 3

    for i in range(warmup, n):
        if np.isnan(close[i]) or np.isnan(fisher[i]) or np.isnan(fisher_sig[i]):
            continue
        if i < 2 or np.isnan(fisher[i - 1]) or np.isnan(fisher_sig[i - 1]):
            continue

        # Fisher cross detection
        f_cross_up = fisher[i - 1] <= fisher_sig[i - 1] and fisher[i] > fisher_sig[i]
        f_cross_dn = fisher[i - 1] >= fisher_sig[i - 1] and fisher[i] < fisher_sig[i]

        # AO momentum confirmation
        ao_bull = (not require_ao) or (not np.isnan(ao_color[i]) and ao_color[i] > 0)
        ao_bear = (not require_ao) or (not np.isnan(ao_color[i]) and ao_color[i] < 0)

        # Exit: Fisher crosses back
        lx[i] = f_cross_dn
        sx[i] = f_cross_up

        can_trade = (i - last_trade_bar) >= cooldown
        if can_trade:
            if f_cross_up and ao_bull:
                le[i] = True
                last_trade_bar = i
            elif f_cross_dn and ao_bear:
                se[i] = True
                last_trade_bar = i

    df2 = df.copy()
    df2["long_entry"] = le
    df2["long_exit"] = lx | se
    df2["short_entry"] = se
    df2["short_exit"] = sx | le
    return df2
