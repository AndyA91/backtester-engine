"""
Gaussian Channel REVERSAL strategy on OANDA:EURUSD 1-minute.

Uses the N-pole IIR Gaussian filter (matching TradingView's Gaussian Channel
indicator by DonovanWall).

Strategy logic (MEAN REVERSION — not breakout):
  - Compute Gaussian IIR filter on Close → midline
  - Upper band = midline + filtered True Range × multiplier
  - Lower band = midline - filtered True Range × multiplier
  - Long entry:  Close crosses back ABOVE the lower band (was overextended, reverting)
  - Long exit:   Close crosses below midline
  - Short entry: Close crosses back BELOW the upper band (was overextended, reverting)
  - Short exit:  Close crosses above midline

Optimized parameters (from grid search of 58 combinations):
  - Period: 500 (~8.3 hours of smoothing)
  - Poles: 4
  - Multiplier: 5.0
  - Cooldown: 90 bars (1.5 hours between trades)

Best results on 21 days of 1-min EURUSD data:
  - Profit Factor: 1.738
  - Net Profit: +$12.65 (+1.27%)
  - Max Drawdown: -$8.40 (-0.84%)
  - Win Rate: 56.9%
  - Trades: 72
  - Avg Trade: +$0.18

Chart data: OANDA:EURUSD 1-min (TradingView export)
Slippage: NOT simulated (set to 0) — requires tick-level data.

Settings (match in TradingView for comparison):
- Initial capital: $1,000
- 1000 units per trade (micro lot)
- 0.0085% commission (~$0.10/side on micro lot)
- 0 slippage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from math import comb, cos, pi

from engine import (
    load_tv_export,
    BacktestConfig, run_backtest_long_short,
    print_kpis, print_trades,
)


# ---------------------------------------------------------------------------
# Gaussian IIR Filter (matches Pine's f_filt9x / DonovanWall channel)
# ---------------------------------------------------------------------------

def gaussian_iir_alpha(period: int, poles: int) -> float:
    """Compute alpha for the Gaussian IIR filter (uses truncated sqrt(2) = 1.414)."""
    beta = (1 - cos(2 * pi / period)) / (1.414 ** (2.0 / poles) - 1)
    return -beta + (beta ** 2 + 2 * beta) ** 0.5


def gaussian_npole_iir(alpha: float, src: np.ndarray, n_poles: int) -> np.ndarray:
    """N-pole Gaussian IIR filter matching Pine's f_filt9x."""
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


# ---------------------------------------------------------------------------
# Signal generator — REVERSAL (mean reversion)
# ---------------------------------------------------------------------------

def gaussian_channel_signals(
    df: pd.DataFrame,
    period: int = 500,
    poles: int = 4,
    mult: float = 5.0,
    cooldown_bars: int = 90,
    start_date: str = "2000-01-01",
    end_date: str = "2069-12-31",
) -> pd.DataFrame:
    """
    Add Gaussian Channel REVERSAL signals to *df*.

    Mean-reversion logic:
      - Long when price crosses back ABOVE lower band (oversold → reverting up)
      - Short when price crosses back BELOW upper band (overbought → reverting down)
      - Exit at midline cross

    Columns added:
        gc_mid, gc_upper, gc_lower,
        long_entry, long_exit, short_entry, short_exit
    """
    df = df.copy()
    close = df["Close"].values
    n = len(close)

    # Compute IIR filter
    alpha = gaussian_iir_alpha(period, poles)
    gc_mid = gaussian_npole_iir(alpha, close, poles)

    # True Range → filtered TR → channel bands
    highs = df["High"].values
    lows = df["Low"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    true_range = np.maximum(highs - lows,
                 np.maximum(np.abs(highs - prev_close),
                            np.abs(lows - prev_close)))
    filtered_tr = gaussian_npole_iir(alpha, true_range, poles)

    gc_upper = gc_mid + filtered_tr * mult
    gc_lower = gc_mid - filtered_tr * mult

    df["gc_mid"] = gc_mid
    df["gc_upper"] = gc_upper
    df["gc_lower"] = gc_lower

    # --- Stateful signal generation with cooldown ---
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)
    dates = df.index

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    position = 0       # +1 = long, -1 = short, 0 = flat
    bars_since_trade = cooldown_bars  # start ready
    bars_in_trade = 0
    max_hold_bars = 0  # 0 = disabled (relying on SL/TP for exits)

    for i in range(1, n):
        bars_since_trade += 1
        if position != 0:
            bars_in_trade += 1
        prev_c = close[i - 1]
        curr_c = close[i]

        if not (ts_start <= dates[i] <= ts_end):
            continue

        # Reversal crossovers
        cross_back_above_lower = prev_c <= gc_lower[i - 1] and curr_c > gc_lower[i]
        cross_back_below_upper = prev_c >= gc_upper[i - 1] and curr_c < gc_upper[i]
        cross_above_mid = prev_c <= gc_mid[i - 1] and curr_c > gc_mid[i]
        cross_below_mid = prev_c >= gc_mid[i - 1] and curr_c < gc_mid[i]

        # --- Exit logic (no cooldown) ---
        # 1) Max hold time exceeded — force close
        should_exit = False
        if max_hold_bars > 0 and position != 0 and bars_in_trade >= max_hold_bars:
            should_exit = True

        # 2) Normal exit at midline
        if position == 1 and (cross_below_mid or should_exit):
            long_exit[i] = True
            position = 0
            bars_since_trade = 0
            bars_in_trade = 0

        elif position == -1 and (cross_above_mid or should_exit):
            short_exit[i] = True
            position = 0
            bars_since_trade = 0
            bars_in_trade = 0

        # --- Entry logic (requires cooldown) ---
        if position == 0 and bars_since_trade >= cooldown_bars:
            if cross_back_above_lower:
                long_entry[i] = True
                position = 1
                bars_since_trade = 0
                bars_in_trade = 0
            elif cross_back_below_upper:
                short_entry[i] = True
                position = -1
                bars_since_trade = 0
                bars_in_trade = 0

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Optimized parameters (from grid search) ---
    GC_PERIOD = 500
    GC_POLES = 4
    GC_MULT = 5.0
    COOLDOWN = 90

    # Load data
    df = load_tv_export("OANDA_EURUSD, 1.csv")
    print(f"\nData range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")

    start_date = str(df.index[0].date())
    end_date = "2069-12-31"

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0085,  # ~$0.10/side on 1000-unit micro lot
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start_date,
        end_date=end_date,
        take_profit_pct=0.20,  # ~24 pips TP
        stop_loss_pct=0.25,    # ~30 pips SL
    )

    # Generate signals
    df_sig = gaussian_channel_signals(
        df, period=GC_PERIOD, poles=GC_POLES, mult=GC_MULT,
        cooldown_bars=COOLDOWN, start_date=start_date, end_date=end_date,
    )

    # Reversals: short entry also closes long, long entry also closes short
    df_sig["long_exit"] = df_sig["long_exit"] | df_sig["short_entry"]
    df_sig["short_exit"] = df_sig["short_exit"] | df_sig["long_entry"]

    # Run
    kpis = run_backtest_long_short(df_sig, config)

    print("\n" + "=" * 60)
    print("  BACKTEST CONFIGURATION")
    print("=" * 60)
    print(f"  Chart Data:       OANDA:EURUSD 1-min (TradingView export)")
    print(f"  Date Range:       {kpis['actual_start_date']} to {kpis['actual_end_date']}")
    print(f"  Initial Capital:  ${config.initial_capital:,.0f}")
    print(f"  Order Size:       {config.qty_value:.0f} units (micro lot)")
    print(f"  Commission:       {config.commission_pct}% (~$0.10/side)")
    print(f"  Slippage:         {config.slippage_ticks}")
    print(f"  Strategy:         Gaussian Channel REVERSAL")
    print(f"  Parameters:       Period={GC_PERIOD}, Poles={GC_POLES}, Mult={GC_MULT}, Cooldown={COOLDOWN}")
    print("=" * 60)

    print_kpis(kpis)
    print_trades(kpis["trades"], max_trades=20)


if __name__ == "__main__":
    main()
