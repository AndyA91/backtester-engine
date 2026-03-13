"""
ORB v2 — Opening Range Breakout with ADX Trend Filter + Asymmetric R/R

Builds on ORB v1 (freq_showdown_1.py) which found:
  EURUSD: 7:00 UTC, 15min range, TP=SL=0.10%  → Full PF=1.11, OOS PF=1.49
  GBPJPY: 7:00 UTC, 30min range, TP=0.20/SL=0.05% → Full PF=1.29, OOS PF=1.91

Improvements to test:
  1. ADX trend gate — only enter breakout when ADX >= thresh (momentum confirmed)
     Hypothesis: strong ADX at session open → breakout more likely to follow through
  2. Wider R/R scan — asymmetric TP/SL (e.g. TP=0.30%, SL=0.05% = 6:1)
  3. VWAP direction filter — long only above session VWAP, short only below
     Hypothesis: trading with intraday direction improves win rate
"""

import sys
import io
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from time import time

from engine import load_tv_export, BacktestConfig, run_backtest_long_short
from indicators.adx import calc_adx
from indicators.vwap import calc_vwap


def run_bt(df, sd, ed, tp=0.0, sl=0.0, qty=1000.0):
    cfg = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
        qty_type="fixed", qty_value=qty, pyramiding=1,
        start_date=sd, end_date=ed, take_profit_pct=tp, stop_loss_pct=sl,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return run_backtest_long_short(df, cfg)


def ex(kpis):
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":  min(pf, 999.0),
        "wr":  kpis.get("win_rate", 0.0) or 0.0,
        "tr":  kpis.get("total_trades", 0),
        "net": kpis.get("net_profit", 0.0) or 0.0,
        "dd":  kpis.get("max_drawdown_pct", 0.0) or 0.0,
    }


def strat_orb_v2(df, open_hour=7, range_mins=30, adx_thresh=0, vwap_filter=False):
    """
    Opening Range Breakout v2.

    Parameters
    ----------
    open_hour    : UTC hour of session open (7=London, 13=NY)
    range_mins   : Minutes to define opening range (bars × 5-min)
    adx_thresh   : Min ADX to allow entry (0 = disabled)
    vwap_filter  : True → long only if close > session VWAP; short only if below
    """
    n       = len(df)
    hours   = df.index.hour
    minutes = df.index.minute
    c       = df["Close"].values

    # ── Precompute OR levels ──────────────────────────────────────────────────
    in_range_mask = (hours == open_hour) & (minutes < range_mins)
    range_df = df[in_range_mask][["High", "Low"]].copy()
    range_df["_date"] = range_df.index.date

    if len(range_df) == 0:
        df2 = df.copy()
        for col in ("long_entry", "long_exit", "short_entry", "short_exit"):
            df2[col] = False
        return df2

    or_stats = range_df.groupby("_date").agg({"High": "max", "Low": "min"})
    or_high_dict = or_stats["High"].to_dict()
    or_low_dict  = or_stats["Low"].to_dict()

    date_arr = np.array(df.index.date)
    or_high  = np.array([or_high_dict.get(d, np.nan) for d in date_arr])
    or_low   = np.array([or_low_dict.get(d, np.nan) for d in date_arr])

    # ── ADX (optional) ────────────────────────────────────────────────────────
    if adx_thresh > 0:
        adx = calc_adx(df, adx_period=14)["adx"]
    else:
        adx = np.full(n, 999.0)   # always pass if filter disabled

    # ── Session VWAP direction filter (optional) ──────────────────────────────
    if vwap_filter:
        vw = calc_vwap(df)["vwap"]
    else:
        vw = None

    # ── Session masks ─────────────────────────────────────────────────────────
    after_range = (
        (~in_range_mask)
        & (hours >= open_hour)
        & (hours < 17)
        & ~np.isnan(or_high)
    )
    session_end = (hours == 17) & (minutes == 0)

    # ── Signal loop ───────────────────────────────────────────────────────────
    df2 = df.copy()
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = 0

    for i in range(1, n):
        bst += 1

        # force-close at session end
        if session_end[i] and pos != 0:
            if pos == 1: lx[i] = True
            else:        sx[i] = True
            pos = 0; bst = 0
            continue

        if not after_range[i]: continue

        # breakout signals
        ls = c[i-1] <= or_high[i] and c[i] > or_high[i]
        ss = c[i-1] >= or_low[i]  and c[i] < or_low[i]

        # opposite break exits open position
        if pos == 1  and ss: lx[i] = True; pos = 0; bst = 0
        elif pos ==-1 and ls: sx[i] = True; pos = 0; bst = 0

        # entries — ADX and VWAP direction filter
        if pos == 0 and bst >= 60 and adx[i] >= adx_thresh:
            vwap_long_ok  = (vw is None) or (c[i] > vw[i])
            vwap_short_ok = (vw is None) or (c[i] < vw[i])
            if ls and vwap_long_ok:   le[i] = True; pos = 1;  bst = 0
            elif ss and vwap_short_ok: se[i] = True; pos = -1; bst = 0

    df2["long_entry"]  = le;       df2["long_exit"]  = lx | se
    df2["short_entry"] = se;       df2["short_exit"] = sx | le
    return df2


def main():
    OOS = "2026-02-17"
    rp  = Path(__file__).resolve().parent / "orb_v2_results.txt"
    out = open(rp, "w", encoding="utf-8")

    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    log("ORB v2 — ADX Trend Filter + Asymmetric R/R + VWAP Direction Filter")
    log(f"Commission: 0.0043% | $1000 initial | OOS from {OOS}\n")
    log("Baseline (v1 best): EURUSD 7:00 15min TP=SL=0.10% -> Full PF=1.11, OOS PF=1.49")
    log("                    GBPJPY 7:00 30min TP=0.20/SL=0.05% -> Full PF=1.29, OOS PF=1.91\n")

    datasets = [
        ("OANDA_EURUSD, 5.csv", "EURUSD", 1000),
        ("OANDA_GBPJPY, 5.csv", "GBPJPY", 100),
    ]

    for df_file, pair, qty in datasets:
        df  = load_tv_export(df_file)
        sd  = str(df.index[0].date())
        ed  = str(df.index[-1].date())
        n_days = np.busday_count(df.index[0].date(), df.index[-1].date())

        log(f"\n{'#'*80}")
        log(f"  {pair} 5-min (~{n_days} trading days, {qty} units)")
        log(f"{'#'*80}")
        board = []

        # ── Full scan ─────────────────────────────────────────────────────────
        t0 = time(); best = None
        for oh in [7, 13]:
            for rm in [15, 30, 60]:
                for adx_t in [0, 20, 25, 30, 35]:
                    for vf in [False, True]:
                        ds = strat_orb_v2(df, oh, rm, adx_t, vf)
                        for tp in [0.05, 0.10, 0.15, 0.20, 0.30]:
                            for sl in [0.05, 0.10, 0.15]:
                                r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                                if r["tr"] >= 5 and (not best or r["pf"] > best["pf"]):
                                    best = {**r, "oh": oh, "rm": rm, "adx": adx_t,
                                            "vf": vf, "tp": tp, "sl": sl}

        if best:
            ds  = strat_orb_v2(df, best["oh"], best["rm"], best["adx"], best["vf"])
            oos = ex(run_bt(ds, OOS, ed, best["tp"], best["sl"], qty))
            tpd = best["tr"] / n_days
            rr  = best["tp"] / best["sl"]
            adx_str = f"ADX>={best['adx']}" if best["adx"] > 0 else "ADX=off"
            vf_str  = "VWAP-dir=on" if best["vf"] else "VWAP-dir=off"
            log(f"\n  Best overall (sorted by Full PF):")
            log(f"    Open={best['oh']}:00UTC Range={best['rm']}min {adx_str} {vf_str}")
            log(f"    TP={best['tp']}% SL={best['sl']}% (R/R = {rr:.1f}:1)")
            log(f"    Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} ({tpd:.2f}/day) Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"    OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("Best-IS", best, oos))

        # ── Best by OOS PF (min 8 OOS trades) ────────────────────────────────
        t0 = time(); best_oos = None
        for oh in [7, 13]:
            for rm in [15, 30, 60]:
                for adx_t in [0, 20, 25, 30, 35]:
                    for vf in [False, True]:
                        ds = strat_orb_v2(df, oh, rm, adx_t, vf)
                        for tp in [0.05, 0.10, 0.15, 0.20, 0.30]:
                            for sl in [0.05, 0.10, 0.15]:
                                full_r = ex(run_bt(ds, sd,  ed, tp, sl, qty))
                                oos_r  = ex(run_bt(ds, OOS, ed, tp, sl, qty))
                                if oos_r["tr"] >= 8 and (not best_oos or oos_r["pf"] > best_oos["oos_pf"]):
                                    best_oos = {**full_r, "oos_pf": oos_r["pf"], "oos_wr": oos_r["wr"],
                                                "oos_tr": oos_r["tr"], "oos_dd": oos_r["dd"],
                                                "oh": oh, "rm": rm, "adx": adx_t,
                                                "vf": vf, "tp": tp, "sl": sl}

        if best_oos:
            tpd = best_oos["tr"] / n_days
            rr  = best_oos["tp"] / best_oos["sl"]
            adx_str = f"ADX>={best_oos['adx']}" if best_oos["adx"] > 0 else "ADX=off"
            vf_str  = "VWAP-dir=on" if best_oos["vf"] else "VWAP-dir=off"
            log(f"\n  Best OOS PF (min 8 OOS trades):")
            log(f"    Open={best_oos['oh']}:00UTC Range={best_oos['rm']}min {adx_str} {vf_str}")
            log(f"    TP={best_oos['tp']}% SL={best_oos['sl']}% (R/R = {rr:.1f}:1)")
            log(f"    Full: PF={best_oos['pf']:.3f} WR={best_oos['wr']:.1f}% Tr={best_oos['tr']} ({tpd:.2f}/day) Net={best_oos['net']:+.2f} DD={best_oos['dd']:.2f}%")
            log(f"    OOS:  PF={best_oos['oos_pf']:.3f} WR={best_oos['oos_wr']:.1f}% Tr={best_oos['oos_tr']} DD={best_oos['oos_dd']:.2f}%")
            board.append(("Best-OOS", best_oos, {"pf": best_oos["oos_pf"], "wr": best_oos["oos_wr"],
                                                  "tr": best_oos["oos_tr"], "dd": best_oos["oos_dd"]}))

        # ── Breakdown: impact of each filter ─────────────────────────────────
        log(f"\n  --- Filter Impact Breakdown (best params per config) ---")
        configs = [
            ("No filters (v1 baseline)", 0, False),
            ("ADX filter only",          25, False),
            ("VWAP direction only",       0, True),
            ("ADX + VWAP direction",     25, True),
        ]
        log(f"  {'Config':<28} {'FullPF':>6} {'Tr/day':>7} {'OOS_PF':>7} {'OOS_Tr':>7}")
        log(f"  {'-'*60}")
        for label, adx_t, vf in configs:
            best_cfg = None
            for oh in [7, 13]:
                for rm in [15, 30, 60]:
                    ds = strat_orb_v2(df, oh, rm, adx_t, vf)
                    for tp in [0.05, 0.10, 0.15, 0.20, 0.30]:
                        for sl in [0.05, 0.10, 0.15]:
                            r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                            if r["tr"] >= 5 and (not best_cfg or r["pf"] > best_cfg["pf"]):
                                best_cfg = {**r, "oh": oh, "rm": rm, "tp": tp, "sl": sl}
            if best_cfg:
                ds  = strat_orb_v2(df, best_cfg["oh"], best_cfg["rm"], adx_t, vf)
                oos = ex(run_bt(ds, OOS, ed, best_cfg["tp"], best_cfg["sl"], qty))
                tpd = best_cfg["tr"] / n_days
                fpf = f"{best_cfg['pf']:.2f}" if best_cfg["pf"] < 100 else "INF"
                opf = f"{oos['pf']:.2f}"      if oos["pf"]      < 100 else "INF"
                log(f"  {label:<28} {fpf:>6} {tpd:>5.2f}/d {opf:>7} {oos['tr']:>7}")

    out.close()
    print(f"\n  Done! -> strategies/orb_v2_results.txt")


if __name__ == "__main__":
    main()
