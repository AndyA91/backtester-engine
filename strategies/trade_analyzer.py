"""
Trade-level analysis for MTF KAMA Dual v3 — Candidate A.

Replays the exact strategy with best params, captures per-entry conditions
(ADX, DI spread, slope magnitudes, ATR), and prints a ranked comparison
table to surface what makes the Jan 23-28 trades special vs. losers.

Run:
    python strategies/trade_analyzer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import BacktestConfig, load_tv_export, run_backtest_long_short
from indicators.adx import calc_adx
from indicators.atr import calc_atr
from indicators.kama import calc_kama

# ── Candidate A params ────────────────────────────────────────────────────────
PARAMS = dict(
    kama_len=30, kama_fast=2, kama_slow=60,
    tf1=60, tf2=1440,
    cooldown=90,
    use_session_filter=True,
    adx_threshold=25,
    use_kama_slope_filter=False,
)
START_DATE = "2025-11-24"
END_DATE   = "2069-12-31"


# ── Helpers (copied from v3 — do not import from there to keep standalone) ────

def _align_htf_to_ltf(ltf_index, htf_kama):
    htf_frame = pd.DataFrame({
        "Date":  htf_kama.index,
        "kama":  htf_kama.shift(1).values,
        "slope": htf_kama.diff().shift(1).values,
    })
    ltf_frame = pd.DataFrame({"Date": ltf_index})
    merged = pd.merge_asof(
        ltf_frame.sort_values("Date"),
        htf_frame.sort_values("Date"),
        on="Date", direction="backward",
    )
    return merged["kama"].values, merged["slope"].values


def _build_signals(close, kama_chart, slope_tf1, slope_tf2, adx_vals, hours,
                   cooldown, dates, start_date, end_date,
                   use_session_filter, adx_threshold, use_kama_slope_filter):
    ts_start = np.datetime64(start_date, "ns")
    ts_end   = np.datetime64(end_date,   "ns")
    n = len(close)
    long_entry  = np.zeros(n, bool)
    long_exit   = np.zeros(n, bool)
    short_entry = np.zeros(n, bool)
    short_exit  = np.zeros(n, bool)
    last_trade_bar = -999_999

    for i in range(1, n):
        if any(np.isnan(v) for v in [close[i], close[i-1], kama_chart[i],
                                      kama_chart[i-1], slope_tf1[i],
                                      slope_tf2[i], adx_vals[i]]):
            continue
        bar_in_range = ts_start <= dates[i] <= ts_end
        tf1_bull = slope_tf1[i] > 0;  tf1_bear = slope_tf1[i] < 0
        tf2_bull = slope_tf2[i] > 0;  tf2_bear = slope_tf2[i] < 0
        cross_up = close[i] > kama_chart[i] and close[i-1] <= kama_chart[i-1]
        cross_dn = close[i] < kama_chart[i] and close[i-1] >= kama_chart[i-1]
        long_cond  = tf1_bull and tf2_bull and cross_up
        short_cond = tf1_bear and tf2_bear and cross_dn
        long_exit[i]  = (not tf1_bull) or (not tf2_bull) or (close[i] < kama_chart[i])
        short_exit[i] = (not tf1_bear) or (not tf2_bear) or (close[i] > kama_chart[i])
        if use_kama_slope_filter:
            long_cond  = long_cond  and (kama_chart[i] > kama_chart[i-1])
            short_cond = short_cond and (kama_chart[i] < kama_chart[i-1])
        if adx_threshold > 0:
            long_cond  = long_cond  and (adx_vals[i] > adx_threshold)
            short_cond = short_cond and (adx_vals[i] > adx_threshold)
        if use_session_filter:
            in_sess = 7 <= hours[i] < 22
            long_cond  = long_cond  and in_sess
            short_cond = short_cond and in_sess
        can_trade = (i - last_trade_bar) >= cooldown
        if bar_in_range and can_trade and long_cond:
            long_entry[i] = True;  last_trade_bar = i
        if bar_in_range and can_trade and short_cond:
            short_entry[i] = True; last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).resolve().parent.parent

    print("Loading data...")
    df_5m  = load_tv_export("OANDA_EURUSD, 5.csv")
    df_60m = load_tv_export("OANDA_EURUSD, 60.csv")
    df_1d  = load_tv_export("OANDA_EURUSD, 1D.csv")

    print("Computing indicators...")
    kama_chart = calc_kama(df_5m["Close"],  length=30, fast=2, slow=60).values
    kama_1h    = calc_kama(df_60m["Close"], length=30, fast=2, slow=60)
    kama_1d    = calc_kama(df_1d["Close"],  length=30, fast=2, slow=60)
    _, slope_tf1 = _align_htf_to_ltf(df_5m.index, kama_1h)
    _, slope_tf2 = _align_htf_to_ltf(df_5m.index, kama_1d)

    adx_result = calc_adx(df_5m, di_period=14, adx_period=14)
    adx_vals  = adx_result["adx"]
    plus_di   = adx_result["plus_di"]
    minus_di  = adx_result["minus_di"]

    atr_vals  = calc_atr(df_5m, period=14)["atr"]

    print("Building signals...")
    le, lx, se, sx = _build_signals(
        close=df_5m["Close"].values,
        kama_chart=kama_chart,
        slope_tf1=slope_tf1,
        slope_tf2=slope_tf2,
        adx_vals=adx_vals,
        hours=df_5m.index.hour.values,
        cooldown=PARAMS["cooldown"],
        dates=df_5m.index.to_numpy(dtype="datetime64[ns]"),
        start_date=START_DATE,
        end_date=END_DATE,
        use_session_filter=PARAMS["use_session_filter"],
        adx_threshold=PARAMS["adx_threshold"],
        use_kama_slope_filter=PARAMS["use_kama_slope_filter"],
    )

    df_sig = df_5m[["Open", "High", "Low", "Close", "Volume"]].copy()
    df_sig["long_entry"]  = le
    df_sig["long_exit"]   = lx
    df_sig["short_entry"] = se
    df_sig["short_exit"]  = sx

    cfg = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.0043,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=START_DATE,
        end_date=END_DATE,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    kpis = run_backtest_long_short(df_sig, cfg)
    trades = kpis["trades"]
    print(f"\nTotal trades: {len(trades)}  |  Net profit: ${kpis['net_profit']:.2f}  |  PF: {kpis['profit_factor']:.4f}\n")

    # ── Enrich each trade with entry-bar conditions ───────────────────────────
    rows = []
    for t in trades:
        if t.exit_price is None:
            continue  # skip open trade if any

        # Find bar index of entry (fill on next-bar Open, signal on previous close)
        # Entry date = open of the fill bar; we want the signal bar = fill bar - 1
        # The engine fills at Open of bar after signal, so entry_date is the fill bar's open.
        # We find the fill bar index and look at the signal bar (i-1) for conditions.
        fill_mask = df_5m.index == t.entry_date
        if not fill_mask.any():
            continue
        fill_idx = int(np.where(fill_mask)[0][0])
        sig_idx  = fill_idx - 1  # signal was generated at close of this bar

        if sig_idx < 0:
            continue

        adx_at_entry   = adx_vals[sig_idx]
        plus_di_entry  = plus_di[sig_idx]
        minus_di_entry = minus_di[sig_idx]
        di_spread      = abs(plus_di_entry - minus_di_entry)
        slope1_mag     = abs(slope_tf1[sig_idx])
        slope2_mag     = abs(slope_tf2[sig_idx])
        atr_entry      = atr_vals[sig_idx]
        price_entry    = df_5m["Close"].iloc[sig_idx]
        atr_pct        = 100.0 * atr_entry / price_entry  # ATR as % of price

        # Kama slope on chart TF (signal bar vs previous)
        kama_slope_chart = kama_chart[sig_idx] - kama_chart[sig_idx - 1] if sig_idx > 0 else 0.0

        duration_bars = fill_idx - (int(np.where(df_5m.index == t.exit_date)[0][0]) if (df_5m.index == t.exit_date).any() else fill_idx)
        exit_mask = df_5m.index == t.exit_date
        exit_idx  = int(np.where(exit_mask)[0][0]) if exit_mask.any() else fill_idx
        duration_bars = exit_idx - fill_idx

        rows.append({
            "entry_date":     t.entry_date,
            "exit_date":      t.exit_date,
            "dir":            t.direction,
            "entry_px":       t.entry_price,
            "exit_px":        t.exit_price,
            "pnl":            t.pnl,
            "pnl_pct":        t.pnl_pct,
            "adx":            adx_at_entry,
            "+di":            plus_di_entry,
            "-di":            minus_di_entry,
            "di_spread":      di_spread,
            "slope1_mag":     slope1_mag,
            "slope2_mag":     slope2_mag,
            "atr":            atr_entry,
            "atr_pct":        atr_pct,
            "kama_slope_chart": kama_slope_chart,
            "dur_bars":       duration_bars,
            "result":         "WIN" if (t.pnl or 0) > 0 else "LOSS",
        })

    df_trades = pd.DataFrame(rows)

    # ── Print full trade table ────────────────────────────────────────────────
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")

    print("=" * 120)
    print("FULL TRADE TABLE (chronological, TV-reversed numbering in brackets)")
    print("=" * 120)
    tv_n = len(df_trades)
    for i, row in df_trades.iterrows():
        tv_num = tv_n - i  # TV shows most-recent first
        marker = " << FOCUS" if row["entry_date"].strftime("%Y-%m-%d") in ("2026-01-23","2026-01-26","2026-01-27") else ""
        print(
            f"TV#{tv_num:2d} [{row['result']:4s}]  {row['entry_date'].strftime('%Y-%m-%d %H:%M')} -> {row['exit_date'].strftime('%Y-%m-%d %H:%M')}"
            f"  {row['dir']:5s}  PnL={row['pnl']:+.4f}  "
            f"ADX={row['adx']:.1f}  DI_spread={row['di_spread']:.1f}  "
            f"slope1={row['slope1_mag']:.6f}  slope2={row['slope2_mag']:.6f}  "
            f"ATR%={row['atr_pct']:.4f}  dur={row['dur_bars']}bars"
            f"{marker}"
        )

    # ── Win vs Loss averages ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("WIN vs LOSS — average conditions at entry")
    print("=" * 80)
    for label, grp in df_trades.groupby("result"):
        print(f"\n  {label}  (n={len(grp)}, avg PnL={grp['pnl'].mean():+.4f})")
        for col in ["adx", "di_spread", "slope1_mag", "slope2_mag", "atr_pct", "dur_bars"]:
            print(f"    {col:20s}: {grp[col].mean():.5f}  (min={grp[col].min():.5f}, max={grp[col].max():.5f})")

    # ── Correlation between conditions and PnL ────────────────────────────────
    print("\n" + "=" * 80)
    print("CORRELATION with PnL (positive = higher value -> better trade)")
    print("=" * 80)
    numeric_cols = ["adx", "di_spread", "slope1_mag", "slope2_mag", "atr_pct", "kama_slope_chart"]
    for col in numeric_cols:
        corr = df_trades["pnl"].corr(df_trades[col])
        print(f"  {col:25s}: r={corr:+.3f}")

    # ── Threshold sweep: which ADX cutoff maximises PF? ──────────────────────
    print("\n" + "=" * 80)
    print("ADX THRESHOLD SWEEP on existing 30 trades")
    print("(shows how many trades survive and their sub-PF if we raise ADX cutoff)")
    print("=" * 80)
    for thresh in [25, 27, 30, 32, 35, 38, 40]:
        sub = df_trades[df_trades["adx"] >= thresh]
        wins  = sub[sub["pnl"] > 0]["pnl"].sum()
        losses = abs(sub[sub["pnl"] <= 0]["pnl"].sum())
        pf = wins / losses if losses > 0 else float("inf")
        print(f"  ADX >= {thresh:2d}: {len(sub):2d} trades, gross_win={wins:.4f}, gross_loss={losses:.4f}, PF={pf:.3f}")

    # ── DI spread sweep ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DI SPREAD THRESHOLD SWEEP (+DI - -DI margin)")
    print("=" * 80)
    for thresh in [5, 8, 10, 12, 15, 18, 20]:
        sub = df_trades[df_trades["di_spread"] >= thresh]
        wins   = sub[sub["pnl"] > 0]["pnl"].sum()
        losses = abs(sub[sub["pnl"] <= 0]["pnl"].sum())
        pf = wins / losses if losses > 0 else float("inf")
        print(f"  DI_spread >= {thresh:2d}: {len(sub):2d} trades, PF={pf:.3f}")

    # ── slope2 magnitude sweep ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DAILY SLOPE MAGNITUDE SWEEP (slope2_mag threshold × 1e-4)")
    print("= 'how strongly is the daily KAMA already trending?'")
    print("=" * 80)
    for mult in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]:
        thresh = mult * 1e-4
        sub = df_trades[df_trades["slope2_mag"] >= thresh]
        wins   = sub[sub["pnl"] > 0]["pnl"].sum()
        losses = abs(sub[sub["pnl"] <= 0]["pnl"].sum())
        pf = wins / losses if losses > 0 else float("inf")
        print(f"  slope2 >= {thresh:.5f}: {len(sub):2d} trades, PF={pf:.3f}")


if __name__ == "__main__":
    main()
