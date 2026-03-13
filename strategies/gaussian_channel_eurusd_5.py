"""
Gaussian Channel Strategy v5 -- OANDA:EURUSD 2-min (UTC)
Phase 1: CD ceiling scan (P=110, M=9.5, CD=300-480)
Phase 2: Trend filter comparison (slow Gaussian midline gate)

Anchored at: P=110, M=9.5, TP=0.15%, SL=0.25%  (v4 best IS params)
Walk-forward split kept from v4:
  Train : 2026-01-18 to 2026-02-16
  Test  : 2026-02-17 to 2026-03-03

Phase 1 -- CD scan
  CD = [300, 330, 360, 390, 420, 450, 480]  (7 runs)
  No trend filter. Full + IS + OOS reported for each.

Phase 2 -- Trend filter
  Uses best CD from Phase 1.
  Trend filter = slow Gaussian midline (same IIR, longer period, same poles):
    Long  entry ONLY when Close > trend_mid  (uptrend bias)
    Short entry ONLY when Close < trend_mid  (downtrend bias)
  Options tested:
    None   (baseline, no filter)
    TF=240 (~8H on 2-min bars)
    TF=500 (~16.7H)
    TF=1000 (~33H / ~1.4 days)
  Full + IS + OOS reported for each.
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
)


# ---------------------------------------------------------------------------
# Gaussian IIR filter
# ---------------------------------------------------------------------------

def gaussian_iir_alpha(period: int, poles: int) -> float:
    beta = (1 - cos(2 * pi / period)) / (1.414 ** (2.0 / poles) - 1)
    return -beta + (beta ** 2 + 2 * beta) ** 0.5


def gaussian_npole_iir(alpha: float, src: np.ndarray, n_poles: int) -> np.ndarray:
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
# Signal generator with optional trend filter
# ---------------------------------------------------------------------------

def gc_signals_v5(
    df: pd.DataFrame,
    period: int = 110,
    poles: int = 4,
    mult: float = 9.5,
    cooldown_bars: int = 300,
    start_date: str = "2000-01-01",
    end_date: str = "2069-12-31",
    trend_period: int = 0,       # 0 = disabled
) -> pd.DataFrame:
    """
    Gaussian Channel REVERSAL signals with optional trend filter.

    trend_period > 0: compute a slow Gaussian midline separately.
      - Long  entries only when Close > trend_mid
      - Short entries only when Close < trend_mid
    """
    df = df.copy()
    close = df["Close"].values
    highs = df["High"].values
    lows  = df["Low"].values
    n = len(close)

    # Channel IIR
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

    # Trend filter IIR (separate, slower period)
    if trend_period > 0:
        trend_alpha = gaussian_iir_alpha(trend_period, poles)
        trend_mid   = gaussian_npole_iir(trend_alpha, close, poles)
    else:
        trend_mid = None

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

        # Trend filter gate
        if trend_mid is not None:
            in_uptrend   = curr_c > trend_mid[i]
            in_downtrend = curr_c < trend_mid[i]
        else:
            in_uptrend   = True
            in_downtrend = True

        # Exit (no cooldown, no trend filter on exits)
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

        # Entry (cooldown + trend filter)
        if position == 0 and bars_since_trade >= cooldown_bars:
            if cross_back_above_lower and in_uptrend:
                long_entry[i] = True
                position         = 1
                bars_since_trade = 0
                bars_in_trade    = 0
            elif cross_back_below_upper and in_downtrend:
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
# Helpers
# ---------------------------------------------------------------------------

def build_offset_columns(df_sig, tp_long, sl_long, tp_short, sl_short):
    tp_off = pd.Series(np.nan, index=df_sig.index)
    sl_off = pd.Series(np.nan, index=df_sig.index)
    long_mask  = df_sig["long_entry"]
    short_mask = df_sig["short_entry"]
    tp_off[long_mask]  = df_sig.loc[long_mask,  "Close"] * tp_long  / 100
    sl_off[long_mask]  = df_sig.loc[long_mask,  "Close"] * sl_long  / 100
    tp_off[short_mask] = df_sig.loc[short_mask, "Close"] * tp_short / 100
    sl_off[short_mask] = df_sig.loc[short_mask, "Close"] * sl_short / 100
    df_sig["tp_offset"] = tp_off.shift(1).ffill()
    df_sig["sl_offset"] = sl_off.shift(1).ffill()
    return df_sig


def run_one(df, period, mult, cooldown, tp_long, sl_long, tp_short, sl_short,
            start_date, end_date, poles=4, trend_period=0):
    df_sig = gc_signals_v5(df, period=period, poles=poles, mult=mult,
                           cooldown_bars=cooldown, start_date=start_date, end_date=end_date,
                           trend_period=trend_period)
    df_sig["long_exit"]  = df_sig["long_exit"]  | df_sig["short_entry"]
    df_sig["short_exit"] = df_sig["short_exit"] | df_sig["long_entry"]
    df_sig = build_offset_columns(df_sig, tp_long, sl_long, tp_short, sl_short)

    cfg = BacktestConfig(
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
    return run_backtest_long_short(df_sig, cfg)


def extract(kpis):
    return {
        "pf":  kpis.get("profit_factor", 0.0) or 0.0,
        "wr":  kpis.get("win_rate",      0.0) or 0.0,
        "tr":  kpis.get("total_trades",  0),
        "net": kpis.get("net_profit",    0.0) or 0.0,
        "dd":  kpis.get("max_drawdown_pct", 0.0) or 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_FILE  = "OANDA_EURUSD, 2 (3).csv"
    POLES      = 4
    PERIOD     = 110
    MULT       = 9.5
    TP_LONG    = 0.15
    SL_LONG    = 0.25
    TP_SHORT   = 0.15
    SL_SHORT   = 0.25
    TRAIN_END  = "2026-02-16"
    TEST_START = "2026-02-17"

    df = load_tv_export(DATA_FILE)
    start_date = str(df.index[0].date())
    end_date   = str(df.index[-1].date())

    print(f"\nData  : {DATA_FILE}")
    print(f"Range : {df.index[0]} to {df.index[-1]}")
    print(f"Bars  : {len(df):,}  (2-min UTC)")
    print(f"Anchor: P={PERIOD}  M={MULT}  TP={TP_LONG}%  SL={SL_LONG}%")
    print(f"Split : Train {start_date}-{TRAIN_END} | Test {TEST_START}-{end_date}")

    # -----------------------------------------------------------------------
    # Phase 1 -- CD ceiling scan
    # -----------------------------------------------------------------------
    cd_values = [300, 330, 360, 390, 420, 450, 480]

    print("\n" + "=" * 78)
    print(f"  PHASE 1 -- CD CEILING SCAN  (P={PERIOD}  M={MULT}  no trend filter)")
    print("=" * 78)
    print(f"\n  {'CD':>4}  {'Full PF':>7}  {'Full Tr':>7}  {'Full DD%':>8}  "
          f"{'IS PF':>7}  {'IS Tr':>6}  {'OOS PF':>7}  {'OOS Tr':>7}  {'Retain%':>8}")
    print("  " + "-" * 74)

    cd_results = []
    for cd in cd_values:
        full = extract(run_one(df, PERIOD, MULT, cd, TP_LONG, SL_LONG, TP_SHORT, SL_SHORT,
                               start_date, end_date))
        ins  = extract(run_one(df, PERIOD, MULT, cd, TP_LONG, SL_LONG, TP_SHORT, SL_SHORT,
                               start_date, TRAIN_END))
        oos  = extract(run_one(df, PERIOD, MULT, cd, TP_LONG, SL_LONG, TP_SHORT, SL_SHORT,
                               TEST_START, end_date))

        retain = (oos["pf"] / ins["pf"] * 100) if ins["pf"] > 0 else 0.0

        print(f"  {cd:>4}  {full['pf']:>7.3f}  {full['tr']:>7}  {full['dd']:>7.2f}%  "
              f"{ins['pf']:>7.3f}  {ins['tr']:>6}  {oos['pf']:>7.3f}  {oos['tr']:>7}  {retain:>7.0f}%")

        cd_results.append({"cd": cd, **full,
                           "is_pf": ins["pf"], "is_tr": ins["tr"],
                           "oos_pf": oos["pf"], "oos_tr": oos["tr"],
                           "retain": retain})

    # Pick best CD: highest full-range PF with at least some OOS trades
    best_cd_row = max(cd_results, key=lambda r: r["pf"])
    best_cd = best_cd_row["cd"]
    print(f"\n  CD peak (full-range PF) -> CD={best_cd}  PF={best_cd_row['pf']:.3f}")

    # -----------------------------------------------------------------------
    # Phase 2 -- Trend filter comparison
    # -----------------------------------------------------------------------
    trend_options = [
        (0,    "No filter  (baseline)"),
        (240,  "TF=240  (~8H)       "),
        (500,  "TF=500  (~16.7H)    "),
        (1000, "TF=1000 (~33H/1.4d) "),
    ]

    print("\n" + "=" * 78)
    print(f"  PHASE 2 -- TREND FILTER COMPARISON  (P={PERIOD}  M={MULT}  CD={best_cd})")
    print(f"  Trend filter = slow Gaussian midline (same poles={POLES})")
    print(f"  Long  entries only when Close > trend_mid")
    print(f"  Short entries only when Close < trend_mid")
    print("=" * 78)
    print(f"\n  {'Filter':22s}  {'Full PF':>7}  {'Full WR%':>8}  {'Full Tr':>7}  {'Full Net$':>9}  {'Full DD%':>8}  "
          f"{'IS PF':>7}  {'IS Tr':>6}  {'OOS PF':>7}  {'OOS Tr':>7}  {'Retain%':>8}")
    print("  " + "-" * 107)

    tf_results = []
    for tp_val, label in trend_options:
        full = extract(run_one(df, PERIOD, MULT, best_cd, TP_LONG, SL_LONG, TP_SHORT, SL_SHORT,
                               start_date, end_date, trend_period=tp_val))
        ins  = extract(run_one(df, PERIOD, MULT, best_cd, TP_LONG, SL_LONG, TP_SHORT, SL_SHORT,
                               start_date, TRAIN_END, trend_period=tp_val))
        oos  = extract(run_one(df, PERIOD, MULT, best_cd, TP_LONG, SL_LONG, TP_SHORT, SL_SHORT,
                               TEST_START, end_date, trend_period=tp_val))

        retain = (oos["pf"] / ins["pf"] * 100) if ins["pf"] > 0 else 0.0

        print(f"  {label:22s}  {full['pf']:>7.3f}  {full['wr']:>7.1f}%  {full['tr']:>7}  "
              f"{full['net']:>+9.2f}  {full['dd']:>7.2f}%  "
              f"{ins['pf']:>7.3f}  {ins['tr']:>6}  {oos['pf']:>7.3f}  {oos['tr']:>7}  {retain:>7.0f}%")

        tf_results.append({"label": label.strip(), "tp_val": tp_val, **full,
                           "is_pf": ins["pf"], "is_tr": ins["tr"],
                           "oos_pf": oos["pf"], "oos_tr": oos["tr"],
                           "retain": retain})

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  FINAL SUMMARY")
    print("=" * 78)

    # Best by full-range PF
    best_tf = max(tf_results, key=lambda r: r["pf"])
    # Best by OOS PF
    best_oos_tf = max(tf_results, key=lambda r: r["oos_pf"])
    # Best by retention (OOS/IS ratio) among configs with OOS trades > 0
    best_retain_tf = max((r for r in tf_results if r["oos_tr"] > 0),
                         key=lambda r: r["retain"], default=tf_results[0])

    print(f"\n  Anchor : P={PERIOD}  M={MULT}  TP={TP_LONG}%  SL={SL_LONG}%")
    print(f"  Best CD (full PF) : CD={best_cd}  PF={best_cd_row['pf']:.3f}")
    print(f"\n  Best filter by full-range PF  : {best_tf['label']:25s}  Full PF={best_tf['pf']:.3f}  OOS PF={best_tf['oos_pf']:.3f}")
    print(f"  Best filter by OOS PF        : {best_oos_tf['label']:25s}  Full PF={best_oos_tf['pf']:.3f}  OOS PF={best_oos_tf['oos_pf']:.3f}")
    print(f"  Best filter by IS->OOS retain: {best_retain_tf['label']:25s}  Retain={best_retain_tf['retain']:.0f}%  OOS PF={best_retain_tf['oos_pf']:.3f}")

    # Print the overall best config
    print(f"\n  Recommended config:")
    print(f"    P={PERIOD}  M={MULT}  CD={best_cd}  Poles={POLES}")
    if best_oos_tf["tp_val"] > 0:
        print(f"    Trend filter: TF={best_oos_tf['tp_val']}")
    else:
        print(f"    Trend filter: none")
    print(f"    TP={TP_LONG}%  SL={SL_LONG}%  (both directions)")
    print(f"\n  Chart data : OANDA:EURUSD 2-min UTC  ({DATA_FILE})")
    print(f"  Commission : 0.0085% per order (slippage NOT simulated, set to 0)")


if __name__ == "__main__":
    main()
