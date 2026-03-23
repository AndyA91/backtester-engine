"""
Walk-Forward Optimization: Donchian Mean Reversion on EURUSD Renko.

Tests whether DC band-fade mean-reversion generalises OOS.
Optimizes: dc_period, cooldown.

Chart data: OANDA:EURUSD Renko 0.0005 (20K bars, ~4 years)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine import (
    load_tv_export, BacktestConfig, calc_donchian,
    run_walk_forward, print_wfo_results,
)


def dc_reversion_signals(df, dc_period=30, cd=30):
    """Donchian mean-reversion: fade band touch, exit at midline."""
    df = df.copy()
    upper_s, lower_s, mid_s = calc_donchian(df["High"], df["Low"], dc_period)
    upper = upper_s.values
    lower = lower_s.values
    mid = mid_s.values

    c = df["Close"].values
    n = len(c)
    le = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    pos = 0
    bst = cd

    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(upper[i - 1]):
            continue

        ls = c[i - 1] <= lower[i - 1] and c[i] > lower[i]
        ss = c[i - 1] >= upper[i - 1] and c[i] < upper[i]
        cuv = c[i - 1] <= mid[i - 1] and c[i] > mid[i]
        cdv = c[i - 1] >= mid[i - 1] and c[i] < mid[i]

        if pos == 1 and (cuv or ss):
            lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cdv or ls):
            sx[i] = True; pos = 0; bst = 0

        if pos == 0 and bst >= cd:
            if ls:
                le[i] = True; pos = 1; bst = 0
            elif ss:
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
        "dc_period": [14, 20, 30, 50, 80],
        "cd": [10, 20, 30, 60, 100],
    }

    results = run_walk_forward(
        df,
        signal_fn=dc_reversion_signals,
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
