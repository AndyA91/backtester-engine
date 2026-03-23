"""
Walk-Forward Optimization: ADX Crossover on EURUSD Renko.

Tests whether +DI/-DI crossover with ADX strength gate generalises OOS.
Optimizes: di_period, adx_thresh, cooldown.

Chart data: OANDA:EURUSD Renko 0.0005 (20K bars, ~4 years)
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
from indicators.adx import calc_adx


def adx_cross_signals(df, di_period=14, adx_thresh=25, cd=20):
    """+DI/-DI crossover with ADX strength gate."""
    df = df.copy()
    c = df["Close"].values
    n = len(c)
    res = calc_adx(df, di_period=di_period, adx_period=14)
    pdi = res["plus_di"]
    mdi = res["minus_di"]
    adx = res["adx"]

    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    pos = 0
    bst = cd

    for i in range(2, n):
        bst += 1
        if np.isnan(adx[i]) or np.isnan(pdi[i]) or np.isnan(pdi[i - 1]):
            continue

        cross_up = pdi[i - 1] <= mdi[i - 1] and pdi[i] > mdi[i]
        cross_down = pdi[i - 1] >= mdi[i - 1] and pdi[i] < mdi[i]
        trending = adx[i] >= adx_thresh

        if pos == 1 and cross_down:
            lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and cross_up:
            sx[i] = True; pos = 0; bst = 0

        if pos == 0 and bst >= cd and trending:
            if cross_up:
                le[i] = True; pos = 1; bst = 0
            elif cross_down:
                se[i] = True; pos = -1; bst = 0

    df["long_entry"] = le
    df["long_exit"] = lx | se
    df["short_entry"] = se
    df["short_exit"] = sx | le
    return df


def main():
    df = load_tv_export("OANDA_EURUSD, 1S renko 0.0005.csv")

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0043,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        start_date="2022-06-01",
        end_date="2069-12-31",
    )

    param_grid = {
        "di_period": [7, 10, 14, 21],
        "adx_thresh": [15, 20, 25, 30],
        "cd": [10, 20, 40, 60],
    }

    results = run_walk_forward(
        df,
        signal_fn=adx_cross_signals,
        param_grid=param_grid,
        config=config,
        train_months=6,
        test_months=2,
        objective="net_profit_pct",
        min_trades=5,
        long_short=True,
        warmup_bars=200,
    )

    print_wfo_results(results)


if __name__ == "__main__":
    main()
