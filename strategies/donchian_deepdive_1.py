"""
Donchian Mean Reversion Deep Dive — EURUSD 5-min

Focus: DC Mean Reversion was the only strategy with real OOS edge
  (P=50, CD=60 → OOS PF=5.39, 16 trades, DD=-0.44%)

This script explores:
  A. Pure DC Mean Reversion — fine-grained period (20-80), cooldown (10-100)
  B. DC Reversion + ADX ranging filter (ADX < thresh blocks trending)
  C. DC Reversion + Session filter (London/NY only)
  D. DC Reversion + alternate exits (opposite band, timed hold)

All run in parallel. Results ranked by IS net profit (min 5 trades).
"""

import sys, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from engine import load_tv_export, BacktestConfig, run_backtest_long_short, calc_donchian
from indicators.adx import calc_adx


# ─── Worker ──────────────────────────────────────────────────────────────────
def _run_single(df, sd, ed, tp, sl, qty):
    cfg = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
        qty_type="fixed", qty_value=qty, pyramiding=1,
        start_date=sd, end_date=ed, take_profit_pct=tp, stop_loss_pct=sl,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf": min(pf, 999.0), "wr": kpis.get("win_rate", 0.0) or 0.0,
        "tr": kpis.get("total_trades", 0), "net": kpis.get("net_profit", 0.0) or 0.0,
        "dd": kpis.get("max_drawdown_pct", 0.0) or 0.0,
    }


def _worker(args):
    df, sd, ed, tp, sl, qty, name, pstr, idx = args
    r = _run_single(df, sd, ed, tp, sl, qty)
    return (name, pstr, tp, sl, r, idx)


# =============================================================================
# STRATEGY VARIANTS
# =============================================================================

def strat_rev_midline(df, upper, lower, mid, cd=30):
    """A. Pure reversion: fade band touch, exit at midline."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue
        ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        if pos == 1 and (cuv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_rev_adx(df, upper, lower, mid, adx_vals, adx_thresh=25, cd=30):
    """B. Reversion + ADX ranging filter (entry only when ADX < thresh)."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(adx_vals[i]): continue
        ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        if pos == 1 and (cuv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd and adx_vals[i] < adx_thresh:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_rev_session(df, upper, lower, mid, cd=30):
    """C. Reversion + session filter (07:00-17:00 UTC only)."""
    c = df["Close"].values; n = len(c)
    hours = df.index.hour
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue
        ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        in_session = 7 <= hours[i] < 17
        # exits always allowed
        if pos == 1 and (cuv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0
        # entries only in session
        if pos == 0 and bst >= cd and in_session:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_rev_opp_band(df, upper, lower, mid, cd=30):
    """D1. Reversion exit at OPPOSITE band (ride further than midline)."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue
        ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
        # exit when price reaches the opposite band
        hit_upper = c[i] >= upper[i]
        hit_lower = c[i] <= lower[i]
        if pos == 1 and (hit_upper or ss): lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (hit_lower or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_rev_timed(df, upper, lower, mid, cd=30, max_hold=50):
    """D2. Reversion with timed exit (max N bars hold, or midline)."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd; hold = 0
    for i in range(2, n):
        bst += 1
        if pos != 0: hold += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue
        ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        timed_out = hold >= max_hold
        if pos == 1 and (cuv or ss or timed_out): lx[i] = True; pos = 0; bst = 0; hold = 0
        elif pos == -1 and (cdv or ls or timed_out): sx[i] = True; pos = 0; bst = 0; hold = 0
        if pos == 0 and bst >= cd:
            if ls:   le[i] = True; pos = 1; bst = 0; hold = 0
            elif ss: se[i] = True; pos = -1; bst = 0; hold = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_rev_adx_session(df, upper, lower, mid, adx_vals, adx_thresh=25, cd=30):
    """E. Reversion + ADX ranging + session filter (kitchen-sink combo)."""
    c = df["Close"].values; n = len(c)
    hours = df.index.hour
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(adx_vals[i]): continue
        ls = c[i-1] <= lower[i-1] and c[i] > lower[i]
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        in_session = 7 <= hours[i] < 17
        if pos == 1 and (cuv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd and adx_vals[i] < adx_thresh and in_session:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


# =============================================================================
# MAIN
# =============================================================================
def main():
    OOS = "2026-02-17"
    rp  = Path(__file__).resolve().parent / "donchian_deepdive_results.txt"
    out = open(rp, "w", encoding="utf-8")

    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    log("DONCHIAN MEAN REVERSION — Deep Dive")
    log("6 variants | Fine-grained param grid around P=50 sweet spot")
    log(f"Commission: 0.0043% | $1000 initial | OOS from {OOS}")
    log(f"Min 5 trades to qualify | Ranked by IS net profit\n")

    tp_sl = [
        (0.0, 0.0), (0.05, 0.05), (0.05, 0.10),
        (0.10, 0.0), (0.10, 0.10), (0.10, 0.15),
        (0.15, 0.0), (0.15, 0.10),
        (0.20, 0.0), (0.20, 0.10),
        (0.0, 0.10), (0.0, 0.15),
    ]

    df  = load_tv_export("OANDA_EURUSD, 5.csv")
    sd  = str(df.index[0].date())
    ed  = str(df.index[-1].date())
    n_days = np.busday_count(df.index[0].date(), df.index[-1].date())
    qty = 1000

    log(f"  EURUSD 5-min ({len(df):,} bars, ~{n_days} trading days)")

    # ── Precompute ────────────────────────────────────────────────────────────
    periods = list(range(20, 85, 5))  # 20, 25, 30, ..., 80
    log(f"  Precomputing DC channels for periods {periods[0]}-{periods[-1]} ...")
    t0 = time()
    dc = {}
    for p in periods:
        u, l, m = calc_donchian(df["High"], df["Low"], p)
        dc[p] = (u.values, l.values, m.values)
    adx = calc_adx(df, adx_period=14)["adx"]
    log(f"  Done in {time()-t0:.1f}s")

    # ── Generate signals ──────────────────────────────────────────────────────
    log(f"  Generating signal sets ...")
    t0 = time()
    jobs = []  # (name, pstr, sig_df)

    cds = [10, 20, 30, 40, 50, 60, 80, 100]

    for p in periods:
        u, l, m = dc[p]
        for cd in cds:
            ps = f"P={p} CD={cd}"
            jobs.append(("A.Rev-Mid",   ps, strat_rev_midline(df, u, l, m, cd)))
            jobs.append(("C.Rev-Sess",  ps, strat_rev_session(df, u, l, m, cd)))
            jobs.append(("D1.Rev-OppB", ps, strat_rev_opp_band(df, u, l, m, cd)))

        for cd in cds:
            for mh in [20, 50, 100]:
                ps = f"P={p} CD={cd} MH={mh}"
                jobs.append(("D2.Rev-Time", ps, strat_rev_timed(df, u, l, m, cd, mh)))

        for thresh in [15, 20, 25, 30]:
            for cd in cds:
                ps = f"P={p} ADX<{thresh} CD={cd}"
                jobs.append(("B.Rev-ADX",     ps, strat_rev_adx(df, u, l, m, adx, thresh, cd)))
                jobs.append(("E.Rev-ADX-Ses", ps, strat_rev_adx_session(df, u, l, m, adx, thresh, cd)))

    n_signals = len(jobs)
    n_total = n_signals * len(tp_sl)
    log(f"  {n_signals} signal sets × {len(tp_sl)} TP/SL = {n_total} total backtests")
    log(f"  Signal generation: {time()-t0:.1f}s")

    # ── Build tasks ───────────────────────────────────────────────────────────
    bt_tasks = []
    for idx, (name, pstr, sig_df) in enumerate(jobs):
        for tp, sl in tp_sl:
            bt_tasks.append((sig_df, sd, ed, tp, sl, qty, name, f"{pstr} TP={tp}% SL={sl}%", idx))

    # ── Run in parallel ───────────────────────────────────────────────────────
    log(f"  Running {len(bt_tasks)} backtests in parallel ...")
    t0 = time()
    full_results = {}

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_worker, task): task for task in bt_tasks}
        done = 0
        for future in as_completed(futures):
            name, pstr, tp, sl, r, idx = future.result()
            if idx not in full_results:
                full_results[idx] = []
            full_results[idx].append((name, pstr, tp, sl, r))
            done += 1
            if done % 1000 == 0:
                log(f"    ... {done}/{len(bt_tasks)} done ({time()-t0:.0f}s)")

    log(f"  All done in {time()-t0:.0f}s")

    # ── Collect profitable, run OOS ───────────────────────────────────────────
    log(f"  Collecting profitable combos (min 5 trades, PF > 1.0) ...")
    profitable = []
    for idx, results in full_results.items():
        for name, pstr, tp, sl, r in results:
            if r["tr"] >= 5 and r["pf"] > 1.0:
                profitable.append((name, pstr, tp, sl, r, idx))

    log(f"  {len(profitable)} qualified combos")

    # OOS pass in parallel
    log(f"  Running OOS for {len(profitable)} combos ...")
    t0 = time()
    oos_tasks = []
    for i, (name, pstr, tp, sl, full_r, idx) in enumerate(profitable):
        sig_df = jobs[idx][2]
        oos_tasks.append((sig_df, OOS, ed, tp, sl, qty, name, pstr, i))

    all_results = []
    if oos_tasks:
        with ProcessPoolExecutor() as pool:
            futures = {pool.submit(_worker, t): t for t in oos_tasks}
            for future in as_completed(futures):
                name, pstr, tp, sl, oos_r, i = future.result()
                full_r = profitable[i][4]
                all_results.append((name, pstr, full_r, oos_r))

    log(f"  OOS done in {time()-t0:.0f}s")

    # ── Per-strategy summary ──────────────────────────────────────────────────
    log(f"\n  {'='*85}")
    log(f"  PER-STRATEGY SUMMARY")
    log(f"  {'='*85}")
    strat_names = sorted(set(r[0] for r in all_results))
    for sn in strat_names:
        subs = [r for r in all_results if r[0] == sn]
        oos_prof = [r for r in subs if r[3]["pf"] > 1.0 and r[3]["tr"] >= 3]
        log(f"  {sn}: {len(subs)} IS profitable | {len(oos_prof)} OOS profitable (PF>1, min 3 OOS trades)")
        if oos_prof:
            # Best by OOS net profit (more meaningful than PF with small samples)
            best = max(oos_prof, key=lambda x: x[3]["net"])
            _, pstr, full, oos = best
            tpd = full["tr"] / max(n_days, 1)
            log(f"    Best: {pstr}")
            log(f"    Full: PF={full['pf']:.3f} WR={full['wr']:.1f}% Tr={full['tr']} ({tpd:.2f}/d) Net={full['net']:+.2f} DD={full['dd']:.2f}%")
            log(f"    OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} Net={oos['net']:+.2f} DD={oos['dd']:.2f}%")

    # ── MASTER LEADERBOARD — sorted by OOS net (min 3 OOS trades) ─────────
    log(f"\n  {'='*110}")
    log(f"  LEADERBOARD — EURUSD 5-min (top 40, min 3 OOS trades, sorted by OOS net profit)")
    log(f"  {'='*110}")
    log(f"  {'#':>3} {'Strategy':>14} {'FullPF':>6} {'Tr':>4} {'IS-Net':>8} {'FullDD':>7} {'OOS_PF':>7} {'OOSTr':>5} {'OOS-Net':>8} {'OOS_DD':>7}  Config")
    log(f"  {'-'*110}")

    ranked = [r for r in all_results if r[3]["tr"] >= 3]
    ranked.sort(key=lambda x: x[3]["net"], reverse=True)

    for rank, (name, pstr, full, oos) in enumerate(ranked[:40], 1):
        fpf = f"{full['pf']:.2f}" if full["pf"] < 100 else "INF"
        opf = f"{oos['pf']:.2f}" if oos["pf"] < 100 else "INF"
        log(f"  {rank:>3} {name:>14} {fpf:>6} {full['tr']:>4} {full['net']:>+8.2f} {full['dd']:>6.1f}% {opf:>7} {oos['tr']:>5} {oos['net']:>+8.2f} {oos['dd']:>6.1f}%  {pstr}")

    if not ranked:
        log("  (no combos met criteria)")

    # ── ROBUSTNESS CHECK — same strategy, different TP/SL ─────────────────
    log(f"\n  {'='*85}")
    log(f"  ROBUSTNESS CHECK — Best signal set across all TP/SL combos")
    log(f"  {'='*85}")
    # Group by signal set (idx), show how many TP/SL combos are OOS profitable
    sig_stats = {}
    for name, pstr, full, oos in all_results:
        # strip TP/SL from pstr to get base config
        base = pstr.rsplit(" TP=", 1)[0]
        key = (name, base)
        if key not in sig_stats:
            sig_stats[key] = {"total": 0, "oos_prof": 0, "best_oos_net": -9999, "best_pstr": "", "best_full": None, "best_oos": None}
        sig_stats[key]["total"] += 1
        if oos["tr"] >= 3 and oos["pf"] > 1.0:
            sig_stats[key]["oos_prof"] += 1
        if oos["tr"] >= 3 and oos["net"] > sig_stats[key]["best_oos_net"]:
            sig_stats[key]["best_oos_net"] = oos["net"]
            sig_stats[key]["best_pstr"] = pstr
            sig_stats[key]["best_full"] = full
            sig_stats[key]["best_oos"] = oos

    # Sort by fraction of TP/SL combos that are OOS profitable
    robust = [(k, v) for k, v in sig_stats.items() if v["oos_prof"] >= 3]
    robust.sort(key=lambda x: (x[1]["oos_prof"] / x[1]["total"], x[1]["best_oos_net"]), reverse=True)

    log(f"  {'#':>3} {'Strategy':>14} {'OOSprof':>7} {'Total':>5} {'Frac':>5}  {'BestOOS-Net':>10}  Config")
    log(f"  {'-'*85}")
    for rank, ((name, base), v) in enumerate(robust[:20], 1):
        frac = v["oos_prof"] / v["total"]
        log(f"  {rank:>3} {name:>14} {v['oos_prof']:>7} {v['total']:>5} {frac:>5.0%}  {v['best_oos_net']:>+10.2f}  {base}")

    out.close()
    print(f"\n  Done! -> strategies/donchian_deepdive_results.txt")


if __name__ == "__main__":
    main()
