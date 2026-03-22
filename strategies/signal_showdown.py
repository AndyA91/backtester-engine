"""
Signal Showdown — Backtest every entry/exit signal combination.

Runs each entry signal with "opposite signal" as exit (the default reversal
system), then prints a ranked table of results.

Usage:
    python strategies/signal_showdown.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import traceback
import numpy as np
import pandas as pd

from engine import (
    load_tv_export,
    BacktestConfig, run_backtest, run_backtest_long_short,
)

# ── Entry signal imports ─────────────────────────────────────────────────────
from signals.entries_trend import (
    sig_ema_cross, sig_triple_ema, sig_hma_turn, sig_supertrend,
    sig_donchian_breakout, sig_ichimoku_cloud, sig_kama_slope,
    sig_price_vs_sma,
)
from signals.entries_meanrev import (
    sig_rsi_extreme, sig_crsi_extreme, sig_bb_bounce, sig_keltner_touch,
    sig_stoch_extreme, sig_williams_r, sig_cci_extreme, sig_fisher_cross,
    sig_mfi_extreme,
)
from signals.entries_momentum import (
    sig_macd_cross, sig_macd_hist_flip, sig_vwmacd_cross,
    sig_ao_zero, sig_ao_saucer, sig_rvi_cross,
    sig_adx_breakout, sig_squeeze_fire,
)
from signals.entries_volume import (
    sig_obv_ema, sig_ad_cross, sig_cmf_flip, sig_volume_spike,
)
from signals.entries_pattern import (
    sig_rsi_divergence, sig_macd_divergence, sig_pivot_breakout,
    sig_inside_bar, sig_engulfing, sig_psar_flip,
)

# ── Exit signal imports ──────────────────────────────────────────────────────
from signals.exits import (
    exit_opposite_signal, exit_atr_trail, exit_n_bars, exit_ema_cross,
    exit_rsi_target, exit_psar, exit_bb_mid, exit_supertrend,
)


# ── All entry signals to test ────────────────────────────────────────────────
ENTRY_SIGNALS = [
    # Trend
    lambda df: sig_ema_cross(df, 9, 21),
    lambda df: sig_ema_cross(df, 20, 50),
    lambda df: sig_triple_ema(df),
    lambda df: sig_hma_turn(df, 55),
    lambda df: sig_supertrend(df),
    lambda df: sig_donchian_breakout(df, 20),
    lambda df: sig_ichimoku_cloud(df),
    lambda df: sig_kama_slope(df),
    lambda df: sig_price_vs_sma(df, 200),
    # Mean reversion
    lambda df: sig_rsi_extreme(df),
    lambda df: sig_crsi_extreme(df),
    lambda df: sig_bb_bounce(df),
    lambda df: sig_keltner_touch(df),
    lambda df: sig_stoch_extreme(df),
    lambda df: sig_williams_r(df),
    lambda df: sig_cci_extreme(df),
    lambda df: sig_fisher_cross(df),
    lambda df: sig_mfi_extreme(df),
    # Momentum
    lambda df: sig_macd_cross(df),
    lambda df: sig_macd_hist_flip(df),
    lambda df: sig_vwmacd_cross(df),
    lambda df: sig_ao_zero(df),
    lambda df: sig_ao_saucer(df),
    lambda df: sig_rvi_cross(df),
    lambda df: sig_adx_breakout(df),
    lambda df: sig_squeeze_fire(df),
    # Volume
    lambda df: sig_obv_ema(df),
    lambda df: sig_ad_cross(df),
    lambda df: sig_cmf_flip(df),
    lambda df: sig_volume_spike(df),
    # Pattern
    lambda df: sig_rsi_divergence(df),
    lambda df: sig_macd_divergence(df),
    lambda df: sig_pivot_breakout(df),
    lambda df: sig_inside_bar(df),
    lambda df: sig_engulfing(df),
    lambda df: sig_psar_flip(df),
]


def run_signal_backtest(df_raw: pd.DataFrame, entry_fn, config: BacktestConfig,
                       long_only: bool = False) -> dict | None:
    """Run a single entry signal through the engine."""
    try:
        df = df_raw.copy()
        signals = entry_fn(df)
        name = signals["name"]
        long_entry = signals["long_entry"]
        short_entry = signals["short_entry"]

        if long_only:
            # Long only: use short signal as exit instead
            df["long_entry"] = long_entry
            df["long_exit"] = short_entry
            n_signals = np.sum(long_entry) + np.sum(short_entry)
        else:
            # Long+short reversal system
            df["long_entry"] = long_entry
            df["short_entry"] = short_entry
            df["long_exit"] = short_entry
            df["short_exit"] = long_entry
            n_signals = np.sum(long_entry) + np.sum(short_entry)

        if n_signals < 2:
            return {"name": name, "error": "< 2 signals", "trades": 0}

        if long_only:
            kpis = run_backtest(df, config)
        else:
            kpis = run_backtest_long_short(df, config)

        if kpis.get("total_trades", 0) == 0:
            return {"name": name, "error": "no trades", "trades": 0}

        return {
            "name": name,
            "trades": kpis.get("total_trades", 0),
            "net_profit_pct": kpis.get("net_profit_pct", 0),
            "win_rate": kpis.get("win_rate", 0),
            "profit_factor": kpis.get("profit_factor", 0),
            "max_dd_pct": kpis.get("max_drawdown_pct", 0),
            "avg_trade_pct": kpis.get("avg_trade_pct", 0),
        }
    except Exception as e:
        name = "?"
        try:
            name = entry_fn(df_raw.head(100))["name"]
        except Exception:
            pass
        return {"name": name, "error": str(e)[:60], "trades": 0}


def main():
    # Load data
    # Parse args: signal_showdown.py [data_file] [--long-only]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    data_file = args[0] if args else "SYNTH_EURUSD, 1D.csv"
    long_only = "--long-only" in flags

    mode_str = "LONG ONLY" if long_only else "LONG + SHORT"
    print(f"=" * 70)
    print(f"SIGNAL SHOWDOWN — Testing {len(ENTRY_SIGNALS)} entry signals ({mode_str})")
    print(f"Data: {data_file}")
    print(f"=" * 70)

    df = load_tv_export(data_file)

    config = BacktestConfig(
        initial_capital=10000.0,
        commission_pct=0.1,        # 0.1% commission (MEMORY.md standard)
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        start_date="2018-01-01",
        end_date="2069-12-31",
    )

    results = []
    for i, entry_fn in enumerate(ENTRY_SIGNALS):
        r = run_signal_backtest(df, entry_fn, config, long_only=long_only)
        if r:
            status = "OK" if "error" not in r else f"SKIP ({r['error']})"
            print(f"  [{i+1:2d}/{len(ENTRY_SIGNALS)}] {r['name']:<30s} {status}")
            results.append(r)

    # ── Print results table ──────────────────────────────────────────────
    valid = [r for r in results if "error" not in r and r["trades"] > 0]
    errors = [r for r in results if "error" in r]

    if valid:
        # Sort by net profit %
        valid.sort(key=lambda x: x["net_profit_pct"], reverse=True)

        print(f"\n{'=' * 90}")
        print(f"{'RESULTS — Ranked by Net Profit %':^90}")
        print(f"{'=' * 90}")
        print(f"{'#':<4} {'Signal':<30} {'Trades':>7} {'Net P/L%':>10} {'Win%':>7} "
              f"{'PF':>7} {'MaxDD%':>8} {'Avg%':>8}")
        print(f"{'-' * 90}")

        for i, r in enumerate(valid):
            pf = r['profit_factor']
            pf_str = f"{pf:.2f}" if pf < 999 else "∞"
            print(f"{i+1:<4} {r['name']:<30} {r['trades']:>7} "
                  f"{r['net_profit_pct']:>9.1f}% {r['win_rate']:>6.1f}% "
                  f"{pf_str:>7} {r['max_dd_pct']:>7.1f}% {r.get('avg_trade_pct', 0):>7.2f}%")

        print(f"\n{len(valid)} signals tested successfully, {len(errors)} skipped")
    else:
        print("\nNo valid results!")

    if errors:
        print(f"\nSkipped signals:")
        for r in errors:
            print(f"  {r['name']}: {r.get('error', '?')}")


if __name__ == "__main__":
    main()
