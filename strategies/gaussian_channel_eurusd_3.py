"""
Gaussian Channel Strategy v3 -- OANDA:EURUSD 2-min (UTC)
Finer Grid (Option 1) + Directional MAE/MFE Exits (Option 5)

Improvements over v2:
  1. Finer 100-run grid focused on the sweet spot found in v2:
       P=100-130  x  M=8.0-10.0  x  CD=120-210
  5. Separate MAE/MFE analysis for LONG vs SHORT trades;
     asymmetric TP/SL per direction via tp_offset/sl_offset columns.

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
# Signal generator -- REVERSAL (mean reversion), v3
# ---------------------------------------------------------------------------

def gc_signals_v3(
    df: pd.DataFrame,
    period: int = 125,
    poles: int = 4,
    mult: float = 9.0,
    cooldown_bars: int = 180,
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

    alpha = gaussian_iir_alpha(period, poles)
    gc_mid = gaussian_npole_iir(alpha, close, poles)

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

    ts_start = pd.Timestamp(start_date)
    ts_end   = pd.Timestamp(end_date)
    dates    = df.index

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    position         = 0
    bars_since_trade = cooldown_bars
    bars_in_trade    = 0

    for i in range(1, n):
        bars_since_trade += 1
        if position != 0:
            bars_in_trade += 1

        prev_c = close[i - 1]
        curr_c = close[i]

        if not (ts_start <= dates[i] <= ts_end):
            continue

        cross_back_above_lower = prev_c <= gc_lower[i - 1] and curr_c > gc_lower[i]
        cross_back_below_upper = prev_c >= gc_upper[i - 1] and curr_c < gc_upper[i]
        cross_above_mid        = prev_c <= gc_mid[i - 1]   and curr_c > gc_mid[i]
        cross_below_mid        = prev_c >= gc_mid[i - 1]   and curr_c < gc_mid[i]

        if position == 1 and cross_below_mid:
            long_exit[i] = True
            position         = 0
            bars_since_trade = 0
            bars_in_trade    = 0
        elif position == -1 and cross_above_mid:
            short_exit[i] = True
            position         = 0
            bars_since_trade = 0
            bars_in_trade    = 0

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
# MAE / MFE analysis -- directional
# ---------------------------------------------------------------------------

def compute_mae_mfe(trades, df: pd.DataFrame) -> list:
    """
    For each closed trade compute MAE and MFE as % of entry price.
    Returns list of {direction, winner, mae_pct, mfe_pct, pnl}.
    """
    results = []
    for t in trades:
        if t.exit_date is None:
            continue
        mask = (df.index >= t.entry_date) & (df.index <= t.exit_date)
        bars = df.loc[mask]
        if bars.empty:
            continue

        ep = t.entry_price
        if t.direction == "long":
            mfe_pct = (bars["High"].max() - ep) / ep * 100
            mae_pct = (ep - bars["Low"].min()) / ep * 100
        else:
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


def _derive_exits(winners, losers, direction_label: str) -> tuple:
    """
    Print MAE/MFE distribution for one direction and return (tp_pct, sl_pct).
    TP = winner MFE p65, SL = loser MAE p80, rounded to 0.05%.
    """
    def pcts(vals, label):
        if not vals:
            print(f"  {label:52s}  (no data)")
            return [0.0] * 4
        arr = np.array(vals)
        p = [np.percentile(arr, q) for q in [25, 50, 75, 90]]
        print(f"  {label:52s}  p25={p[0]:.4f}%  p50={p[1]:.4f}%  p75={p[2]:.4f}%  p90={p[3]:.4f}%")
        return p

    nw, nl = len(winners), len(losers)
    print(f"\n  {direction_label.upper()} TRADES  ({nw} winners / {nl} losers)")
    print("  " + "-" * 92)
    print(f"  {'':52s}  p25          p50          p75          p90")
    pcts([d["mae_pct"] for d in winners], f"Winners MAE -- % adverse before recovering  ({nw})")
    pcts([d["mae_pct"] for d in losers],  f"Losers  MAE -- % adverse before full loss   ({nl})")
    pcts([d["mfe_pct"] for d in winners], f"Winners MFE -- % favorable available        ({nw})")
    pcts([d["mfe_pct"] for d in losers],  f"Losers  MFE -- % favorable before turning   ({nl})")

    raw_tp = np.percentile([d["mfe_pct"] for d in winners], 65) if winners else 0.20
    raw_sl = np.percentile([d["mae_pct"] for d in losers],  80) if losers  else 0.25
    tp_pct = max(round(round(raw_tp / 0.05) * 0.05, 4), 0.05)
    sl_pct = max(round(round(raw_sl / 0.05) * 0.05, 4), 0.05)

    print(f"\n  Derived exits ({direction_label}) ->  TP = {tp_pct:.2f}%  |  SL = {sl_pct:.2f}%")
    print(f"  (TP = winner MFE p65 = {raw_tp:.4f}%  ->  {tp_pct:.2f}%)")
    print(f"  (SL = loser  MAE p80 = {raw_sl:.4f}%  ->  {sl_pct:.2f}%)")
    return tp_pct, sl_pct


def analyze_mae_mfe_directional(data: list) -> tuple:
    """
    Separate MAE/MFE analysis for long and short trades.
    Returns (tp_long, sl_long, tp_short, sl_short).
    """
    if not data:
        print("  [MAE/MFE] No closed trades -- using defaults")
        return 0.20, 0.25, 0.20, 0.25

    longs  = [d for d in data if d["direction"] == "long"]
    shorts = [d for d in data if d["direction"] == "short"]

    long_winners  = [d for d in longs  if d["winner"]]
    long_losers   = [d for d in longs  if not d["winner"]]
    short_winners = [d for d in shorts if d["winner"]]
    short_losers  = [d for d in shorts if not d["winner"]]

    tp_long,  sl_long  = _derive_exits(long_winners,  long_losers,  "long")
    tp_short, sl_short = _derive_exits(short_winners, short_losers, "short")

    return tp_long, sl_long, tp_short, sl_short


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
    print("  PHASE 1 -- BASELINE RUN  (P=125  M=9.0  CD=180  no TP/SL)")
    print("=" * 70)

    df_base = gc_signals_v3(df, period=125, poles=POLES, mult=9.0,
                            cooldown_bars=180, start_date=start_date, end_date=end_date)
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
    # Phase 2 -- Directional MAE/MFE analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2 -- DIRECTIONAL MAE / MFE ANALYSIS")
    print("=" * 70)

    mae_mfe_data = compute_mae_mfe(base_kpis["trades"], df)
    tp_long, sl_long, tp_short, sl_short = analyze_mae_mfe_directional(mae_mfe_data)

    print(f"\n  Summary ->  Long  TP={tp_long:.2f}%  SL={sl_long:.2f}%")
    print(f"              Short TP={tp_short:.2f}%  SL={sl_short:.2f}%")

    # -----------------------------------------------------------------------
    # Phase 3 -- 100-run finer grid
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  PHASE 3 -- 100-RUN OPTIMIZER")
    print(f"  TP_L={tp_long:.2f}%  SL_L={sl_long:.2f}%  TP_S={tp_short:.2f}%  SL_S={sl_short:.2f}%")
    print("=" * 70)

    periods     = [100, 110, 120, 125, 130]
    multipliers = [8.0, 8.5, 9.0, 9.5, 10.0]
    cooldowns   = [120, 150, 180, 210]

    grid = list(itertools.product(periods, multipliers, cooldowns))  # 100 runs

    print(f"\n  Grid : {len(periods)} periods x {len(multipliers)} multipliers x {len(cooldowns)} cooldowns = {len(grid)} runs")
    print(f"  {'Run':>5}  {'P':>4}  {'M':>5}  {'CD':>4}  {'PF':>6}  {'WR%':>6}  {'Tr':>4}  {'Net$':>8}  {'DD%':>7}")
    print("  " + "-" * 65)

    results = []

    for run_idx, (period, mult, cooldown) in enumerate(grid, start=1):
        df_sig = gc_signals_v3(df, period=period, poles=POLES, mult=mult,
                               cooldown_bars=cooldown, start_date=start_date, end_date=end_date)
        df_sig["long_exit"]  = df_sig["long_exit"]  | df_sig["short_entry"]
        df_sig["short_exit"] = df_sig["short_exit"] | df_sig["long_entry"]

        # Per-direction TP/SL via offset columns
        tp_off = pd.Series(np.nan, index=df_sig.index)
        sl_off = pd.Series(np.nan, index=df_sig.index)

        long_mask  = df_sig["long_entry"]
        short_mask = df_sig["short_entry"]

        tp_off[long_mask]  = df_sig.loc[long_mask,  "Close"] * tp_long  / 100
        sl_off[long_mask]  = df_sig.loc[long_mask,  "Close"] * sl_long  / 100
        tp_off[short_mask] = df_sig.loc[short_mask, "Close"] * tp_short / 100
        sl_off[short_mask] = df_sig.loc[short_mask, "Close"] * sl_short / 100

        # Shift 1 to fill bar, ffill so offset persists across all post-entry bars
        df_sig["tp_offset"] = tp_off.shift(1).ffill()
        df_sig["sl_offset"] = sl_off.shift(1).ffill()

        cfg = BacktestConfig(
            initial_capital=1000.0,
            commission_pct=0.0085,
            slippage_ticks=0,
            qty_type="fixed",
            qty_value=1000.0,
            pyramiding=1,
            start_date=start_date,
            end_date=end_date,
            take_profit_pct=0.0,   # disabled — using tp_offset column
            stop_loss_pct=0.0,     # disabled — using sl_offset column
        )

        kpis = run_backtest_long_short(df_sig, cfg)

        pf     = kpis.get("profit_factor", 0.0) or 0.0
        wr     = kpis.get("win_rate", 0.0) or 0.0
        trades = kpis.get("total_trades", 0)
        net    = kpis.get("net_profit", 0.0) or 0.0
        dd_pct = kpis.get("max_drawdown_pct", 0.0) or 0.0

        print(f"  {run_idx:>5}  {period:>4}  {mult:>5.1f}  {cooldown:>4}  "
              f"{pf:>6.3f}  {wr:>5.1f}%  {trades:>4}  {net:>+8.2f}  {dd_pct:>6.2f}%")

        results.append({
            "run": run_idx, "period": period, "mult": mult, "cooldown": cooldown,
            "pf": pf, "wr": wr, "trades": trades, "net": net, "dd_pct": dd_pct,
        })

    # -----------------------------------------------------------------------
    # Summary -- top 10 by Profit Factor
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  TOP 10 BY PROFIT FACTOR")
    print(f"  TP_L={tp_long:.2f}%  SL_L={sl_long:.2f}%  TP_S={tp_short:.2f}%  SL_S={sl_short:.2f}%  Poles={POLES}")
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
