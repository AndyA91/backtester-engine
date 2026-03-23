"""
Walk-Forward Optimization: KAMA Trend on EURUSD Daily.

Tests whether KAMA crossover trend-following generalises OOS.
Uses dual KAMA (fast + slow) with slope confirmation.
Optimizes: kama_fast, kama_slow, cooldown.

Chart data: OANDA:EURUSD 1D (6K+ bars, ~24 years)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine import (
    load_tv_export, BacktestConfig,
    run_walk_forward, print_wfo_results,
)
from indicators.kama import calc_kama


def kama_trend_signals(df, kama_fast=10, kama_slow=50, cd=5):
    """Dual KAMA trend-following: cross above slow KAMA = long, vice versa."""
    df = df.copy()
    close = df["Close"]
    n = len(close)

    kf = calc_kama(close, kama_fast)
    ks = calc_kama(close, kama_slow)
    kf_vals = kf.values
    ks_vals = ks.values

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    pos = 0
    bst = cd

    for i in range(2, n):
        bst += 1
        if np.isnan(kf_vals[i]) or np.isnan(ks_vals[i]):
            continue
        if np.isnan(kf_vals[i - 1]) or np.isnan(ks_vals[i - 1]):
            continue

        # KAMA fast crosses above/below KAMA slow
        cross_up = kf_vals[i - 1] <= ks_vals[i - 1] and kf_vals[i] > ks_vals[i]
        cross_dn = kf_vals[i - 1] >= ks_vals[i - 1] and kf_vals[i] < ks_vals[i]

        # Slope confirmation: slow KAMA rising/falling
        slope_up = ks_vals[i] > ks_vals[i - 1]
        slope_dn = ks_vals[i] < ks_vals[i - 1]

        # Exits: cross or slope reversal
        if pos == 1 and (cross_dn or not slope_up):
            lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cross_up or not slope_dn):
            sx[i] = True; pos = 0; bst = 0

        # Entries with cooldown
        if pos == 0 and bst >= cd:
            if cross_up and slope_up:
                le[i] = True; pos = 1; bst = 0
            elif cross_dn and slope_dn:
                se[i] = True; pos = -1; bst = 0

    df["long_entry"] = le
    df["long_exit"] = lx | se
    df["short_entry"] = se
    df["short_exit"] = sx | le
    return df


def main():
    df = load_tv_export("OANDA_EURUSD, 1D.csv")

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.1,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        start_date="2005-01-01",
        end_date="2069-12-31",
    )

    param_grid = {
        "kama_fast": [5, 8, 10, 14, 21],
        "kama_slow": [30, 40, 50, 80],
        "cd": [3, 5, 10, 15],
    }

    results = run_walk_forward(
        df,
        signal_fn=kama_trend_signals,
        param_grid=param_grid,
        config=config,
        train_months=24,
        test_months=6,
        objective="net_profit_pct",
        min_trades=3,
        long_short=True,
        warmup_bars=100,
    )

    print_wfo_results(results)


if __name__ == "__main__":
    main()
