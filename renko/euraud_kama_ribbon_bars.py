"""
Quick diagnostic: compute avg / median bars held per trade for the best
R030 finalist on EURAUD 0.0006, split by window and by winner / loser.

Purpose: answer "what was avg/median profit bar count" after the postmortem
showed 0/5 HOLDOUT pass. Helps diagnose WHY the strategy fails — if winners
are held ~1-2 bricks and losers hang around, gray exit is firing too early.
"""

import contextlib
import io
import sys
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "renko" / "strategies"))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators

import r030_kama_ribbon_3line as strat

RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"

# Best HOLDOUT finalist (config #3): 5/13/30
PARAMS = {"fast_len": 5, "mid_len": 13, "slow_len": 30, "fast_sc": 2, "slow_sc": 30}

WINDOWS = [
    ("TRAIN",    "2023-07-20", "2025-09-30"),
    ("VALIDATE", "2025-10-01", "2025-12-31"),
    ("HOLDOUT",  "2026-01-01", "2026-03-14"),
]


def make_cfg(start, end):
    return BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0046,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )


def bar_index_of(df, ts):
    """Binary search df (sorted by time) for the bar at/after ts."""
    # df index is default 0..N; time column is df["time"] or datetime index
    # We'll use df["datetime"] set by the Renko loader
    dt = df["datetime"].values
    return int(np.searchsorted(dt, np.datetime64(ts)))


def main():
    print("Loading data...")
    df = load_renko_export(RENKO_FILE)
    add_renko_indicators(df)
    df_sig = strat.generate_signals(df.copy(), **PARAMS)

    # Ensure the loader produced a datetime column
    if "datetime" not in df_sig.columns:
        df_sig["datetime"] = df_sig.index

    print(f"Config: {PARAMS}")
    print("=" * 70)

    for name, start, end in WINDOWS:
        cfg = make_cfg(start, end)
        with contextlib.redirect_stdout(io.StringIO()):
            kpis = run_backtest_long_short(df_sig, cfg)

        trades = kpis.get("trades", [])
        if not trades:
            print(f"\n{name}: no trades")
            continue

        winners_bars, losers_bars = [], []
        for t in trades:
            if t.exit_date is None:
                continue
            i_in  = bar_index_of(df_sig, t.entry_date)
            i_out = bar_index_of(df_sig, t.exit_date)
            held = max(i_out - i_in, 0)
            if t.pnl is not None and t.pnl > 0:
                winners_bars.append(held)
            else:
                losers_bars.append(held)

        def fmt(lst):
            if not lst:
                return "n=0"
            return (f"n={len(lst):>3}  avg={statistics.mean(lst):5.2f}  "
                    f"median={statistics.median(lst):5.1f}  "
                    f"min={min(lst)}  max={max(lst)}")

        print(f"\n{name}")
        print(f"  Total trades : {len(winners_bars) + len(losers_bars)}")
        print(f"  Winners held : {fmt(winners_bars)}")
        print(f"  Losers  held : {fmt(losers_bars)}")
        all_bars = winners_bars + losers_bars
        print(f"  All     held : {fmt(all_bars)}")


if __name__ == "__main__":
    main()
