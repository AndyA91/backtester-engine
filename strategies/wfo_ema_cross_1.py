"""
Walk-Forward Optimization: EMA Crossover on EURUSD Daily.

Tests whether the classic EMA crossover generalises out-of-sample by
optimizing fast/slow EMA lengths on rolling 12-month train windows
and evaluating on 6-month test windows.

Chart data: OANDA:EURUSD 1D
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    load_tv_export,
    BacktestConfig, calc_ema, detect_crossover, detect_crossunder,
    run_walk_forward, print_wfo_results,
)


def ema_cross_signals(df, fast=9, slow=21):
    """EMA crossover signal generator with configurable lengths."""
    df = df.copy()
    df["fast_ema"] = calc_ema(df["Close"], fast)
    df["slow_ema"] = calc_ema(df["Close"], slow)
    df["long_entry"] = detect_crossover(df["fast_ema"], df["slow_ema"])
    df["long_exit"] = detect_crossunder(df["fast_ema"], df["slow_ema"])
    return df


def main():
    df = load_tv_export("OANDA_EURUSD, 1D.csv")

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.1,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        start_date="2010-01-01",
        end_date="2069-12-31",
    )

    param_grid = {
        "fast": [5, 8, 10, 13, 15],
        "slow": [20, 25, 30, 40, 50],
    }

    results = run_walk_forward(
        df,
        signal_fn=ema_cross_signals,
        param_grid=param_grid,
        config=config,
        train_months=12,
        test_months=6,
        objective="net_profit_pct",
        min_trades=3,
        warmup_bars=100,
    )

    print_wfo_results(results)


if __name__ == "__main__":
    main()
