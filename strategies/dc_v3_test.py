"""
Donchian Mean Reversion v3 -- Feature testing:
  Option 2: Fixed pip stop loss  (more room to mean-revert than ATR SL)
  Option 3: Higher-TF trend filter  (1h / 4h TV export data, no lookahead)

NOTE: SL exits fire at next-bar open (not exact intrabar price) -- approximation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from engine import (
    load_tv_export, BacktestConfig, run_backtest_long_short, print_kpis,
    calc_donchian, calc_ema,
)


def run_bt(df, sd, ed, qty=1000.0):
    cfg = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
        qty_type="fixed", qty_value=qty, pyramiding=1,
        start_date=sd, end_date=ed,
    )
    return run_backtest_long_short(df, cfg)


def make_htf_trend(df_5min, htf_csv, ema_period):
    """
    Load a higher-TF TV export, compute EMA, align to 5-min bars with no lookahead.
    Shift by 1 HTF bar so each 5-min bar only sees the last CLOSED HTF bar.
    """
    df_htf = load_tv_export(htf_csv)
    htf_ema = calc_ema(df_htf["Close"], ema_period)
    # Shift 1 bar: current HTF bar is still open, can only see prev closed bar
    htf_ema_shifted = htf_ema.shift(1)
    # Forward-fill onto 5-min index
    return htf_ema_shifted.reindex(df_5min.index, method="ffill")


def dc_signals_v3(df, period=50, cd=60, max_hold=48,
                  pip_sl=0,        # fixed pip SL (0 = off); 1 pip = 0.0001
                  htf_trend=None,  # precomputed HTF EMA Series (aligned to df)
                  directions="both"):
    """v2 core + optional pip SL and/or HTF trend filter."""
    upper, lower, mid = calc_donchian(df["High"], df["Low"], period)
    upper = upper.values; lower = lower.values; mid = mid.values
    c   = df["Close"].values
    hi  = df["High"].values
    lo  = df["Low"].values
    n   = len(c)
    hrs = df.index.hour
    dow = df.index.dayofweek

    htf_v = htf_trend.values if htf_trend is not None else None

    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = 99999; hold = 0
    sl_val = 0.0; entry_i = -1

    for i in range(2, n):
        bst += 1
        if pos != 0: hold += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue

        ls  = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss  = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1]   and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1]   and c[i] < mid[i]

        fri_close = dow[i] == 4 and hrs[i] >= 15
        timed_out = hold >= max_hold

        # -- Stop loss (intrabar sim; fires from bar after fill bar)
        if sl_val > 0 and i > entry_i + 1:
            if pos == 1 and lo[i] <= sl_val:
                lx[i] = True; pos = 0; hold = 0; bst = 0; sl_val = 0.0; entry_i = -1
            elif pos == -1 and hi[i] >= sl_val:
                sx[i] = True; pos = 0; hold = 0; bst = 0; sl_val = 0.0; entry_i = -1

        # -- Normal signal exits
        if pos == 1 and not lx[i] and (cuv or ss or timed_out or fri_close):
            lx[i] = True; pos = 0; bst = 0; hold = 0; sl_val = 0.0; entry_i = -1
        elif pos == -1 and not sx[i] and (cdv or ls or timed_out or fri_close):
            sx[i] = True; pos = 0; bst = 0; hold = 0; sl_val = 0.0; entry_i = -1

        # -- HTF trend filter (both directions must align; NaN = allow all)
        if htf_v is not None and not np.isnan(htf_v[i]):
            long_ok  = c[i] > htf_v[i]
            short_ok = c[i] < htf_v[i]
        else:
            long_ok = short_ok = True

        # -- Entry
        if pos == 0 and bst >= cd and not fri_close:
            if ls and long_ok and directions in ("both", "long"):
                le[i] = True; pos = 1; bst = 0; hold = 0; entry_i = i
                if pip_sl > 0:
                    sl_val = c[i] - pip_sl * 0.0001
            elif ss and short_ok and directions in ("both", "short"):
                se[i] = True; pos = -1; bst = 0; hold = 0; entry_i = i
                if pip_sl > 0:
                    sl_val = c[i] + pip_sl * 0.0001

    df2 = df.copy()
    df2["long_entry"]  = le;  df2["long_exit"]  = lx | se
    df2["short_entry"] = se;  df2["short_exit"] = sx | le
    return df2


def _row(label, k, ko):
    if "error" in k:
        return f"  {label:<42}  IS ERROR: {k['error']}"
    tr  = len([t for t in k["trades"]  if t.exit_date is not None])
    otr = 0 if "error" in ko else len([t for t in ko["trades"] if t.exit_date is not None])
    pf  = k.get("profit_factor",  0) or 0
    opf = 0 if "error" in ko else (ko.get("profit_factor", 0) or 0)
    fpf  = f"{pf:.2f}"  if pf  < 100 else "INF"
    fopf = "---" if "error" in ko else (f"{opf:.2f}" if opf < 100 else "INF")
    onet = "---" if "error" in ko else f"{ko.get('net_profit',0):>+7.2f}"
    odd  = "---" if "error" in ko else f"{ko.get('max_drawdown_pct',0):>6.2f}%"
    return (f"  {label:<42} {tr:>4} {k.get('win_rate',0):>5.1f}% {fpf:>6} "
            f"{k.get('net_profit',0):>+7.2f} {k.get('max_drawdown_pct',0):>6.2f}%"
            f" | {otr:>4} {fopf:>6} {onet} {odd}")


def main():
    df  = load_tv_export("OANDA_EURUSD, 5.csv")
    sd  = str(df.index[0].date())
    ed  = "2069-12-31"
    oos = "2026-02-17"

    print("Loading HTF data...")
    # 1h and 4h TV exports -- shift by 1 HTF bar to avoid lookahead
    h1_3  = make_htf_trend(df, "OANDA_EURUSD, 60.csv",  3)
    h1_5  = make_htf_trend(df, "OANDA_EURUSD, 60.csv",  5)
    h1_10 = make_htf_trend(df, "OANDA_EURUSD, 60.csv", 10)
    h4_3  = make_htf_trend(df, "OANDA_EURUSD, 240.csv",  3)
    h4_5  = make_htf_trend(df, "OANDA_EURUSD, 240.csv",  5)
    h4_10 = make_htf_trend(df, "OANDA_EURUSD, 240.csv", 10)

    W   = 109
    hdr = (f"  {'Config':<42} {'Tr':>4} {'WR%':>6} {'PF':>6} "
           f"{'Net$':>7} {'DD%':>7} | {'OTr':>4} {'OPF':>6} {'ONet$':>7} {'ODD%':>7}")

    TESTS = [
        # -- Baselines --------------------------------------------------
        ("v2 baseline",                      dict()),
        ("longs only",                       dict(directions="long")),

        # -- Option 2: Fixed pip SL ------------------------------------
        ("pip SL 15",                        dict(pip_sl=15)),
        ("pip SL 20",                        dict(pip_sl=20)),
        ("pip SL 25",                        dict(pip_sl=25)),
        ("pip SL 30",                        dict(pip_sl=30)),
        ("pip SL 40",                        dict(pip_sl=40)),
        ("pip SL 50",                        dict(pip_sl=50)),
        ("longs + pip SL 25",               dict(directions="long", pip_sl=25)),
        ("longs + pip SL 30",               dict(directions="long", pip_sl=30)),
        ("longs + pip SL 40",               dict(directions="long", pip_sl=40)),

        # -- Option 3: 1h HTF trend filter ----------------------------
        ("1h EMA-3 trend",                   dict(htf_trend=h1_3)),
        ("1h EMA-5 trend",                   dict(htf_trend=h1_5)),
        ("1h EMA-10 trend",                  dict(htf_trend=h1_10)),

        # -- Option 3: 4h HTF trend filter ----------------------------
        ("4h EMA-3 trend",                   dict(htf_trend=h4_3)),
        ("4h EMA-5 trend",                   dict(htf_trend=h4_5)),
        ("4h EMA-10 trend",                  dict(htf_trend=h4_10)),

        # -- Combinations: HTF trend + pip SL -------------------------
        ("1h EMA-5 + pip SL 25",             dict(htf_trend=h1_5, pip_sl=25)),
        ("1h EMA-5 + pip SL 30",             dict(htf_trend=h1_5, pip_sl=30)),
        ("4h EMA-5 + pip SL 25",             dict(htf_trend=h4_5, pip_sl=25)),
        ("4h EMA-5 + pip SL 30",             dict(htf_trend=h4_5, pip_sl=30)),
        ("longs + 1h EMA-5",                 dict(directions="long", htf_trend=h1_5)),
        ("longs + 4h EMA-5",                 dict(directions="long", htf_trend=h4_5)),
        ("longs + 1h EMA-5 + pip SL 25",    dict(directions="long", htf_trend=h1_5, pip_sl=25)),
        ("longs + 4h EMA-5 + pip SL 25",    dict(directions="long", htf_trend=h4_5, pip_sl=25)),
    ]

    print("=" * W)
    print("  DONCHIAN MEAN REVERSION v3 -- OPTIONS 2 & 3")
    print(f"  IS: {sd} to 2026-02-16    OOS: 2026-02-17 to end")
    print(f"  SL exits fill at next open (approximation)")
    print("=" * W)
    print(hdr)
    print("  " + "-" * (W - 2))

    best_label  = None
    best_params = None
    best_oos    = -9999

    prev_section = ""
    sections = {
        "v2 baseline":          "-- Baselines",
        "pip SL 15":            "-- Option 2: Fixed pip stop loss",
        "1h EMA-3 trend":       "-- Option 3: 1h HTF trend filter",
        "4h EMA-3 trend":       "-- Option 3: 4h HTF trend filter",
        "1h EMA-5 + pip SL 25": "-- Combinations: HTF trend + pip SL",
    }

    for label, params in TESTS:
        sec = sections.get(label)
        if sec and sec != prev_section:
            print(f"\n  {sec}")
            prev_section = sec

        df2 = dc_signals_v3(df, **params)
        k   = run_bt(df2, sd, ed)
        ko  = run_bt(df2, oos, ed)
        print(_row(label, k, ko))

        if "error" not in k and "error" not in ko:
            oos_net = ko.get("net_profit", 0)
            if oos_net > best_oos:
                best_oos    = oos_net
                best_label  = label
                best_params = params

    # -- Detailed view of OOS winner ------------------------------------
    print(f"\n{'=' * W}")
    print(f"  BEST OOS CONFIG: {best_label}")
    print(f"{'=' * W}")
    df_best = dc_signals_v3(df, **best_params)
    print("\n  -- In-Sample --")
    print_kpis(run_bt(df_best, sd, ed))
    print(f"\n  -- Out-of-Sample (from {oos}) --")
    print_kpis(run_bt(df_best, oos, ed))


if __name__ == "__main__":
    main()
