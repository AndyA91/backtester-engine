"""
Gaussian Channel Strategy v2 -- OANDA:EURUSD 2-min (UTC)
MAE/MFE-Driven Exit Optimisation + 100-Run Parameter Grid

Three phases:
  1. Baseline run (no TP/SL) -> collect per-trade MAE and MFE
  2. Analyse MAE/MFE distributions -> derive statistically-optimal TP% and SL%
  3. 100-run grid (5 periods × 4 multipliers × 5 cooldowns) using those exits,
     results printed live, final top-10 summary at end.

Chart data : OANDA:EURUSD 2-min UTC (TradingView export)
Commission : 0.0085% per order (~$0.10/side on 1000-unit micro lot)
Slippage   : NOT simulated (set to 0) -- requires tick-level data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import itertools
import numpy as np
import pandas as pd
from math import comb, cos, pi

from engine import (
    load_tv_export,
    BacktestConfig, run_backtest_long_short,
    print_kpis,
)


# ---------------------------------------------------------------------------
# Gaussian IIR Filter (matches Pine's f_filt9x / DonovanWall channel)
# Copied verbatim from gaussian_channel_eurusd_1.py -- do NOT change the 1.414 constant.
# ---------------------------------------------------------------------------

def gaussian_iir_alpha(period: int, poles: int) -> float:
    """Alpha for the Gaussian IIR filter (uses truncated sqrt(2) = 1.414)."""
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
# Signal generator -- REVERSAL (mean reversion), v2
# ---------------------------------------------------------------------------

def gc_signals_v2(
    df: pd.DataFrame,
    period: int = 150,
    poles: int = 4,
    mult: float = 8.0,
    cooldown_bars: int = 120,
    start_date: str = "2000-01-01",
    end_date: str = "2069-12-31",
) -> pd.DataFrame:
    """
    Gaussian Channel REVERSAL signals.

    Long  entry : Close crosses back ABOVE lower band (oversold -> reverting up)
    Short entry : Close crosses back BELOW upper band (overbought -> reverting down)
    Exit        : Close crosses midline (mean reversion complete)

    Stateful; gates all trading actions with bar_in_range to match Pine's timeCondition.
    """
    df = df.copy()
    close = df["Close"].values
    highs = df["High"].values
    lows  = df["Low"].values
    n = len(close)

    # Gaussian IIR on Close -> midline
    alpha = gaussian_iir_alpha(period, poles)
    gc_mid = gaussian_npole_iir(alpha, close, poles)

    # True Range -> filtered TR -> channel bands
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    true_range = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close))
    )
    filtered_tr = gaussian_npole_iir(alpha, true_range, poles)
    gc_upper = gc_mid + filtered_tr * mult
    gc_lower = gc_mid - filtered_tr * mult

    df["gc_mid"]   = gc_mid
    df["gc_upper"] = gc_upper
    df["gc_lower"] = gc_lower

    # Stateful signal generation
    ts_start = pd.Timestamp(start_date)
    ts_end   = pd.Timestamp(end_date)
    dates    = df.index

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    position         = 0
    bars_since_trade = cooldown_bars  # start ready to trade
    bars_in_trade    = 0

    for i in range(1, n):
        bars_since_trade += 1
        if position != 0:
            bars_in_trade += 1

        prev_c = close[i - 1]
        curr_c = close[i]

        if not (ts_start <= dates[i] <= ts_end):
            continue

        # Crossover conditions
        cross_back_above_lower = prev_c <= gc_lower[i - 1] and curr_c > gc_lower[i]
        cross_back_below_upper = prev_c >= gc_upper[i - 1] and curr_c < gc_upper[i]
        cross_above_mid        = prev_c <= gc_mid[i - 1]   and curr_c > gc_mid[i]
        cross_below_mid        = prev_c >= gc_mid[i - 1]   and curr_c < gc_mid[i]

        # Exit logic (no cooldown on exits)
        if position == 1 and cross_below_mid:
            long_exit[i] = True
            position = 0
            bars_since_trade = 0
            bars_in_trade    = 0

        elif position == -1 and cross_above_mid:
            short_exit[i] = True
            position = 0
            bars_since_trade = 0
            bars_in_trade    = 0

        # Entry logic (requires cooldown)
        if position == 0 and bars_since_trade >= cooldown_bars:
            if cross_back_above_lower:
                long_entry[i] = True
                position         = 1
                bars_since_trade = 0
                bars_in_trade    = 0
            elif cross_back_below_upper:
                short_entry[i] = True
                position         = -1
                bars_since_trade = 0
                bars_in_trade    = 0

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit

    return df


# ---------------------------------------------------------------------------
# MAE / MFE analysis
# ---------------------------------------------------------------------------

def compute_mae_mfe(trades, df: pd.DataFrame) -> list:
    """
    For each closed trade compute Maximum Adverse Excursion (MAE) and
    Maximum Favorable Excursion (MFE) as a percentage of entry price.

    Scans all bars between entry_date and exit_date inclusive.
    Returns a list of dicts: {direction, winner, mae_pct, mfe_pct, pnl}
    """
    results = []
    for t in trades:
        if t.exit_date is None:   # skip open trades
            continue
        mask = (df.index >= t.entry_date) & (df.index <= t.exit_date)
        bars = df.loc[mask]
        if bars.empty:
            continue

        ep = t.entry_price
        if t.direction == "long":
            mfe_pct = (bars["High"].max() - ep) / ep * 100
            mae_pct = (ep - bars["Low"].min()) / ep * 100
        else:  # short
            mfe_pct = (ep - bars["Low"].min()) / ep * 100
            mae_pct = (bars["High"].max() - ep) / ep * 100

        results.append({
            "direction": t.direction,
            "winner":    t.pnl > 0,
            "mae_pct":   mae_pct,
            "mfe_pct":   mfe_pct,
            "pnl":       t.pnl,
        })
    return results


def analyze_mae_mfe(data: list) -> tuple:
    """
    Print MAE/MFE distribution stats and return (tp_pct, sl_pct) derived from them.

    TP = 65th pct of winner MFE  (captures most upside without over-reaching)
    SL = 80th pct of loser MAE   (cuts clearly-failed trades, below winner MAE p75)
    Both rounded to nearest 0.05%.
    """
    if not data:
        print("  [MAE/MFE] No closed trades -- using defaults TP=0.20% SL=0.25%")
        return 0.20, 0.25

    winners = [d for d in data if d["winner"]]
    losers  = [d for d in data if not d["winner"]]

    def pcts(vals, label):
        if not vals:
            return [0.0] * 4
        arr = np.array(vals)
        p = [np.percentile(arr, q) for q in [25, 50, 75, 90]]
        print(f"  {label:50s}  p25={p[0]:.4f}%  p50={p[1]:.4f}%  p75={p[2]:.4f}%  p90={p[3]:.4f}%")
        return p

    print(f"\n  {'MAE / MFE DISTRIBUTION':50s}  p25         p50         p75         p90")
    print("  " + "-" * 90)
    pcts([d["mae_pct"] for d in winners], f"Winners MAE -- % adverse before recovering ({len(winners)} trades)")
    pcts([d["mae_pct"] for d in losers],  f"Losers  MAE -- % adverse before full loss  ({len(losers)} trades)")
    pcts([d["mfe_pct"] for d in winners], f"Winners MFE -- % favorable available       ({len(winners)} trades)")
    pcts([d["mfe_pct"] for d in losers],  f"Losers  MFE -- % favorable before turning  ({len(losers)} trades)")

    # Derive TP/SL
    raw_tp = np.percentile([d["mfe_pct"] for d in winners], 65) if winners else 0.20
    raw_sl = np.percentile([d["mae_pct"] for d in losers],  80) if losers  else 0.25

    # Round to nearest 0.05%
    tp_pct = round(round(raw_tp / 0.05) * 0.05, 4)
    sl_pct = round(round(raw_sl / 0.05) * 0.05, 4)

    # Sanity: TP must be at least 0.05%, SL must be at least 0.05%
    tp_pct = max(tp_pct, 0.05)
    sl_pct = max(sl_pct, 0.05)

    print(f"\n  Derived exits  ->  TP = {tp_pct:.2f}%  |  SL = {sl_pct:.2f}%")
    print(f"  (TP = winner MFE p65 = {raw_tp:.4f}%  ->  rounded to {tp_pct:.2f}%)")
    print(f"  (SL = loser  MAE p80 = {raw_sl:.4f}%  ->  rounded to {sl_pct:.2f}%)")

    return tp_pct, sl_pct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_FILE = "OANDA_EURUSD, 2 (3).csv"
    POLES     = 4

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df = load_tv_export(DATA_FILE)
    print(f"\nData  : {DATA_FILE}")
    print(f"Range : {df.index[0]} to {df.index[-1]}")
    print(f"Bars  : {len(df):,}  (2-min UTC)")

    start_date = str(df.index[0].date())
    end_date   = "2069-12-31"

    # -----------------------------------------------------------------------
    # Phase 1 -- Baseline run (no TP/SL)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 1 -- BASELINE RUN  (P=150  M=8.0  CD=120  no TP/SL)")
    print("=" * 70)

    df_base = gc_signals_v2(df, period=150, poles=POLES, mult=8.0,
                            cooldown_bars=120, start_date=start_date, end_date=end_date)
    df_base["long_exit"]  = df_base["long_exit"]  | df_base["short_entry"]
    df_base["short_exit"] = df_base["short_exit"] | df_base["long_entry"]

    base_cfg = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0085,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start_date,
        end_date=end_date,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )

    base_kpis = run_backtest_long_short(df_base, base_cfg)
    print_kpis(base_kpis)

    # -----------------------------------------------------------------------
    # Phase 2 -- MAE/MFE analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2 -- MAE / MFE ANALYSIS")
    print("=" * 70)

    mae_mfe_data = compute_mae_mfe(base_kpis["trades"], df)
    tp_pct, sl_pct = analyze_mae_mfe(mae_mfe_data)

    # -----------------------------------------------------------------------
    # Phase 3 -- 100-run parameter grid
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  PHASE 3 -- 100-RUN OPTIMIZER  (TP={tp_pct:.2f}%  SL={sl_pct:.2f}%)")
    print("=" * 70)

    periods     = [75, 100, 125, 150, 175]
    multipliers = [5.0, 7.0, 9.0, 11.0]
    cooldowns   = [60, 90, 120, 150, 180]

    grid = list(itertools.product(periods, multipliers, cooldowns))  # 100 combos

    print(f"\n  Grid : {len(periods)} periods × {len(multipliers)} multipliers × {len(cooldowns)} cooldowns = {len(grid)} runs")
    print(f"  {'Run':>5}  {'P':>4}  {'M':>5}  {'CD':>4}  {'PF':>6}  {'WR%':>6}  {'Tr':>4}  {'Net$':>8}  {'DD%':>7}")
    print("  " + "-" * 65)

    results = []

    for run_idx, (period, mult, cooldown) in enumerate(grid, start=1):
        df_sig = gc_signals_v2(df, period=period, poles=POLES, mult=mult,
                               cooldown_bars=cooldown, start_date=start_date, end_date=end_date)
        df_sig["long_exit"]  = df_sig["long_exit"]  | df_sig["short_entry"]
        df_sig["short_exit"] = df_sig["short_exit"] | df_sig["long_entry"]

        cfg = BacktestConfig(
            initial_capital=1000.0,
            commission_pct=0.0085,
            slippage_ticks=0,
            qty_type="fixed",
            qty_value=1000.0,
            pyramiding=1,
            start_date=start_date,
            end_date=end_date,
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct,
        )

        kpis = run_backtest_long_short(df_sig, cfg)

        pf      = kpis.get("profit_factor", 0.0) or 0.0
        wr      = kpis.get("win_rate", 0.0) or 0.0
        trades  = kpis.get("total_trades", 0)
        net     = kpis.get("net_profit", 0.0) or 0.0
        dd_pct  = kpis.get("max_drawdown_pct", 0.0) or 0.0

        print(f"  {run_idx:>5}  {period:>4}  {mult:>5.1f}  {cooldown:>4}  "
              f"{pf:>6.3f}  {wr:>5.1f}%  {trades:>4}  {net:>+8.2f}  {dd_pct:>6.2f}%")

        results.append({
            "run":      run_idx,
            "period":   period,
            "mult":     mult,
            "cooldown": cooldown,
            "pf":       pf,
            "wr":       wr,
            "trades":   trades,
            "net":      net,
            "dd_pct":   dd_pct,
        })

    # -----------------------------------------------------------------------
    # Summary -- top 10 by Profit Factor
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  TOP 10 BY PROFIT FACTOR  (TP={tp_pct:.2f}%  SL={sl_pct:.2f}%  Poles={POLES})")
    print("=" * 70)
    print(f"\n  {'Rank':>4}  {'P':>4}  {'M':>5}  {'CD':>4}  {'PF':>6}  {'WR%':>6}  {'Tr':>4}  {'Net$':>8}  {'DD%':>7}")
    print("  " + "-" * 65)

    top10 = sorted(results, key=lambda r: r["pf"], reverse=True)[:10]
    for rank, r in enumerate(top10, start=1):
        print(f"  {rank:>4}  {r['period']:>4}  {r['mult']:>5.1f}  {r['cooldown']:>4}  "
              f"{r['pf']:>6.3f}  {r['wr']:>5.1f}%  {r['trades']:>4}  {r['net']:>+8.2f}  {r['dd_pct']:>6.2f}%")

    best = top10[0]
    print(f"\n  Best  ->  P={best['period']}  M={best['mult']}  CD={best['cooldown']}  "
          f"PF={best['pf']:.3f}  DD={best['dd_pct']:.2f}%")
    print(f"\n  Chart data : OANDA:EURUSD 2-min UTC  ({DATA_FILE})")
    print(f"  Commission : 0.0085% per order (slippage NOT simulated, set to 0)")


if __name__ == "__main__":
    main()
