"""
Walk-Forward Optimization: Gaussian Channel Reversion on EURUSD Daily.

Tests whether the Gaussian Channel mean-reversion strategy generalises OOS.
Optimizes: period, multiplier, cooldown_bars.

Chart data: OANDA:EURUSD 1D (6K+ bars, ~24 years)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from math import comb, cos, pi

from engine import (
    load_tv_export, BacktestConfig,
    run_walk_forward, print_wfo_results,
)


# --- Gaussian IIR Filter (from gaussian_channel_eurusd_1.py) ---

def gaussian_iir_alpha(period, poles):
    beta = (1 - cos(2 * pi / period)) / (1.414 ** (2.0 / poles) - 1)
    return -beta + (beta ** 2 + 2 * beta) ** 0.5


def gaussian_npole_iir(alpha, src, n_poles):
    x = 1.0 - alpha
    n = len(src)
    f = np.zeros(n)
    for i in range(n):
        s = src[i] if not np.isnan(src[i]) else 0.0
        val = alpha ** n_poles * s
        for k in range(1, n_poles + 1):
            prev = f[i - k] if i >= k else 0.0
            val += (-1) ** (k + 1) * comb(n_poles, k) * x ** k * prev
        f[i] = val
    return f


def gc_reversion_signals(df, period=200, mult=3.0, cooldown_bars=20):
    """Gaussian Channel mean-reversion signals for WFO."""
    df = df.copy()
    close = df["Close"].values
    n = len(close)

    alpha = gaussian_iir_alpha(period, 4)  # fixed 4-pole
    gc_mid = gaussian_npole_iir(alpha, close, 4)

    highs = df["High"].values
    lows = df["Low"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    true_range = np.maximum(highs - lows,
                 np.maximum(np.abs(highs - prev_close),
                            np.abs(lows - prev_close)))
    filtered_tr = gaussian_npole_iir(alpha, true_range, 4)
    gc_upper = gc_mid + filtered_tr * mult
    gc_lower = gc_mid - filtered_tr * mult

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    position = 0
    bars_since_trade = cooldown_bars

    for i in range(1, n):
        bars_since_trade += 1
        prev_c = close[i - 1]
        curr_c = close[i]

        cross_back_above_lower = prev_c <= gc_lower[i - 1] and curr_c > gc_lower[i]
        cross_back_below_upper = prev_c >= gc_upper[i - 1] and curr_c < gc_upper[i]
        cross_above_mid = prev_c <= gc_mid[i - 1] and curr_c > gc_mid[i]
        cross_below_mid = prev_c >= gc_mid[i - 1] and curr_c < gc_mid[i]

        if position == 1 and (cross_below_mid or cross_back_below_upper):
            long_exit[i] = True
            position = 0
            bars_since_trade = 0
        elif position == -1 and (cross_above_mid or cross_back_above_lower):
            short_exit[i] = True
            position = 0
            bars_since_trade = 0

        if position == 0 and bars_since_trade >= cooldown_bars:
            if cross_back_above_lower:
                long_entry[i] = True
                position = 1
                bars_since_trade = 0
            elif cross_back_below_upper:
                short_entry[i] = True
                position = -1
                bars_since_trade = 0

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit | short_entry
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit | long_entry
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
        "period": [100, 200, 300, 400, 500],
        "mult": [2.0, 3.0, 5.0, 7.0],
        "cooldown_bars": [5, 10, 20, 30],
    }

    results = run_walk_forward(
        df,
        signal_fn=gc_reversion_signals,
        param_grid=param_grid,
        config=config,
        train_months=24,
        test_months=6,
        objective="net_profit_pct",
        min_trades=3,
        long_short=True,
        warmup_bars=600,
    )

    print_wfo_results(results)


if __name__ == "__main__":
    main()
