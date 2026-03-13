"""
Donchian Channel Showdown v2 — Wide-open parallel exploration

8 strategies, no min-trade restrictions. Only goal: find profitable setups.
Uses ProcessPoolExecutor to run backtests across all CPU cores.

1. DC Breakout         — Classic turtle: breakout of upper/lower, exit at mid
2. DC Mean Reversion   — Fade band touches, exit at midline
3. DC + ADX Trend BRK  — Breakout gated by ADX > thresh (trending)
4. DC + ADX Range REV  — Reversion gated by ADX < thresh (ranging)
5. DC Squeeze          — Breakout only when channel width contracted
6. DC Midline Trend    — Price above mid = long, short below, exit at band
7. DC Multi-Period     — Long-period direction, short-period entries
8. DC + RSI Confirm    — Reversion with RSI oversold/overbought confirmation

Scan: EURUSD + GBPJPY 5-min, OOS from 2026-02-17.
"""

import sys
import io
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from engine import load_tv_export, BacktestConfig, run_backtest_long_short, calc_donchian
from indicators.adx import calc_adx
from indicators.rsi import calc_rsi


# ─── Backtest worker (picklable for multiprocessing) ─────────────────────────
def _run_single(df, sd, ed, tp, sl, qty):
    """Run one backtest and return extracted KPIs."""
    cfg = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
        qty_type="fixed", qty_value=qty, pyramiding=1,
        start_date=sd, end_date=ed, take_profit_pct=tp, stop_loss_pct=sl,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":  min(pf, 999.0),
        "wr":  kpis.get("win_rate", 0.0) or 0.0,
        "tr":  kpis.get("total_trades", 0),
        "net": kpis.get("net_profit", 0.0) or 0.0,
        "dd":  kpis.get("max_drawdown_pct", 0.0) or 0.0,
    }


def _worker(args):
    """Worker function for process pool: (df, sd, ed, tp, sl, qty, name, pstr, is_oos)."""
    df, sd, ed, tp, sl, qty, name, pstr, job_id = args
    r = _run_single(df, sd, ed, tp, sl, qty)
    return (name, pstr, tp, sl, r, job_id)


# =============================================================================
# STRATEGIES
# =============================================================================

def strat_dc_breakout(df, upper, lower, mid, cd=30):
    """1. Classic breakout: enter on new high/low, exit at midline."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(upper[i-1]): continue
        ls = c[i-1] <= upper[i-1] and c[i] > upper[i]
        ss = c[i-1] >= lower[i-1] and c[i] < lower[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        if pos == 1 and (cdv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cuv or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_dc_reversion(df, upper, lower, mid, cd=30):
    """2. Mean reversion: fade band touches, exit at midline."""
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


def strat_dc_adx_trend(df, upper, lower, mid, adx_vals, adx_thresh=25, cd=30):
    """3. Breakout gated by ADX > thresh (trending regime)."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(adx_vals[i]): continue
        ls = c[i-1] <= upper[i-1] and c[i] > upper[i]
        ss = c[i-1] >= lower[i-1] and c[i] < lower[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        if pos == 1 and (cdv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cuv or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd and adx_vals[i] > adx_thresh:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_dc_adx_range(df, upper, lower, mid, adx_vals, adx_thresh=25, cd=30):
    """4. Mean reversion gated by ADX < thresh (ranging regime)."""
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


def strat_dc_squeeze(df, upper, lower, mid, squeeze_lookback=50, cd=30):
    """5. Breakout only when channel width contracted (squeeze)."""
    c = df["Close"].values; n = len(c)
    width = (upper - lower) / np.where(mid > 0, mid, np.nan)
    avg_w = pd.Series(width).rolling(squeeze_lookback, min_periods=squeeze_lookback).mean().values
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(avg_w[i]): continue
        ls = c[i-1] <= upper[i-1] and c[i] > upper[i]
        ss = c[i-1] >= lower[i-1] and c[i] < lower[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        if pos == 1 and (cdv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cuv or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd and width[i] < avg_w[i]:
            if ls:   le[i] = True; pos = 1; bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_dc_midline(df, upper, lower, mid, cd=30):
    """6. Midline trend: long when close crosses above mid, short below."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(mid[i]) or np.isnan(mid[i-1]): continue
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]
        at_upper = c[i] >= upper[i]
        at_lower = c[i] <= lower[i]
        if pos == 1 and (at_upper or cdv):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (at_lower or cuv): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd:
            if cuv:  le[i] = True; pos = 1; bst = 0
            elif cdv: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_dc_multi(df, upper_long, lower_long, mid_long,
                   upper_short, lower_short, mid_short, cd=30):
    """7. Multi-period: long-period for direction, short-period for entry."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(mid_long[i]) or np.isnan(upper_short[i]): continue
        bull_bias = c[i] > mid_long[i]
        bear_bias = c[i] < mid_long[i]
        ls = c[i-1] <= upper_short[i-1] and c[i] > upper_short[i]
        ss = c[i-1] >= lower_short[i-1] and c[i] < lower_short[i]
        cuv = c[i-1] <= mid_short[i-1] and c[i] > mid_short[i]
        cdv = c[i-1] >= mid_short[i-1] and c[i] < mid_short[i]
        if pos == 1 and (cdv or ss):   lx[i] = True; pos = 0; bst = 0
        elif pos == -1 and (cuv or ls): sx[i] = True; pos = 0; bst = 0
        if pos == 0 and bst >= cd:
            if ls and bull_bias:   le[i] = True; pos = 1; bst = 0
            elif ss and bear_bias: se[i] = True; pos = -1; bst = 0
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx | se
    df2["short_entry"] = se; df2["short_exit"] = sx | le
    return df2


def strat_dc_rsi(df, upper, lower, mid, rsi_vals, rsi_os=30, rsi_ob=70, cd=30):
    """8. Reversion confirmed by RSI oversold/overbought."""
    c = df["Close"].values; n = len(c)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if np.isnan(upper[i]) or np.isnan(rsi_vals[i]): continue
        ls = c[i-1] <= lower[i-1] and c[i] > lower[i] and rsi_vals[i] < rsi_os
        ss = c[i-1] >= upper[i-1] and c[i] < upper[i] and rsi_vals[i] > rsi_ob
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


# =============================================================================
# MAIN — parallel scan
# =============================================================================
def main():
    OOS = "2026-02-17"
    rp  = Path(__file__).resolve().parent / "donchian_showdown_results.txt"
    out = open(rp, "w", encoding="utf-8")

    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    log("DONCHIAN CHANNEL SHOWDOWN v2 — Parallel exploration")
    log("8 strategies | Only requirement: profitable (PF > 1.0)")
    log(f"Commission: 0.0043% | $1000 initial | OOS from {OOS}\n")

    tp_sl_grid = [
        (0.0, 0.0), (0.05, 0.05), (0.05, 0.10),
        (0.10, 0.0), (0.10, 0.10), (0.10, 0.15),
        (0.15, 0.0), (0.15, 0.10), (0.15, 0.15),
        (0.20, 0.0), (0.20, 0.10), (0.20, 0.15),
        (0.0, 0.10), (0.0, 0.15),
    ]

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
        log(f"  {pair} 5-min ({len(df):,} bars, ~{n_days} trading days, {qty} units)")
        log(f"{'#'*80}")

        # ── Precompute indicators ─────────────────────────────────────────────
        periods = [5, 10, 14, 20, 30, 50]
        log(f"  Precomputing DC channels {periods}, ADX, RSI ...")
        t0 = time()
        dc = {}
        for p in periods:
            u, l, m = calc_donchian(df["High"], df["Low"], p)
            dc[p] = (u.values, l.values, m.values)
        adx = calc_adx(df, adx_period=14)["adx"]
        rsi = calc_rsi(df, 14)["rsi"]
        log(f"  Done in {time()-t0:.1f}s")

        # ── Generate ALL signal DataFrames (serial — fast) ────────────────────
        log(f"  Generating signal sets ...")
        t0 = time()
        jobs = []  # list of (name, pstr, signaled_df)

        for p in periods:
            u, l, m = dc[p]
            for cd_val in [5, 10, 20, 30, 60]:
                ps = f"P={p} CD={cd_val}"
                jobs.append(("1.DC-BRK",     ps, strat_dc_breakout(df, u, l, m, cd_val)))
                jobs.append(("2.DC-REV",     ps, strat_dc_reversion(df, u, l, m, cd_val)))
                jobs.append(("6.DC-MID",     ps, strat_dc_midline(df, u, l, m, cd_val)))

            for thresh in [15, 20, 25, 30, 35]:
                for cd_val in [5, 10, 20, 30, 60]:
                    ps = f"P={p} ADX>{thresh} CD={cd_val}"
                    jobs.append(("3.DC+ADX-BRK", ps, strat_dc_adx_trend(df, u, l, m, adx, thresh, cd_val)))
                    ps = f"P={p} ADX<{thresh} CD={cd_val}"
                    jobs.append(("4.DC+ADX-REV", ps, strat_dc_adx_range(df, u, l, m, adx, thresh, cd_val)))

            for sq in [30, 50, 100, 200]:
                for cd_val in [5, 10, 20, 30, 60]:
                    ps = f"P={p} SqLB={sq} CD={cd_val}"
                    jobs.append(("5.DC-SQZ", ps, strat_dc_squeeze(df, u, l, m, sq, cd_val)))

            for os_lvl, ob_lvl in [(20, 80), (25, 75), (30, 70), (35, 65), (40, 60)]:
                for cd_val in [5, 10, 20, 30, 60]:
                    ps = f"P={p} RSI={os_lvl}/{ob_lvl} CD={cd_val}"
                    jobs.append(("8.DC+RSI", ps, strat_dc_rsi(df, u, l, m, rsi, os_lvl, ob_lvl, cd_val)))

        # Multi-period
        for pl in [20, 30, 50]:
            for ps_val in [5, 10, 14]:
                if ps_val >= pl: continue
                ul, ll, ml = dc[pl]
                us, ls_, ms = dc[ps_val]
                for cd_val in [5, 10, 20, 30]:
                    ps = f"Plong={pl} Pshort={ps_val} CD={cd_val}"
                    jobs.append(("7.DC-MULTI", ps, strat_dc_multi(df, ul, ll, ml, us, ls_, ms, cd_val)))

        n_signals = len(jobs)
        n_total = n_signals * len(tp_sl_grid)
        log(f"  {n_signals} signal sets × {len(tp_sl_grid)} TP/SL = {n_total} total backtests")
        log(f"  Signal generation: {time()-t0:.1f}s")

        # ── Build backtest tasks ──────────────────────────────────────────────
        bt_tasks = []
        for job_id, (name, pstr, sig_df) in enumerate(jobs):
            for tp, sl in tp_sl_grid:
                bt_tasks.append((sig_df, sd, ed, tp, sl, qty, name, f"{pstr} TP={tp}% SL={sl}%", job_id))

        # ── Run ALL backtests in parallel ─────────────────────────────────────
        log(f"  Running {len(bt_tasks)} backtests in parallel ...")
        t0 = time()
        full_results = {}  # job_id -> list of (name, pstr, tp, sl, kpis)

        with ProcessPoolExecutor() as pool:
            futures = {pool.submit(_worker, task): task for task in bt_tasks}
            done = 0
            for future in as_completed(futures):
                name, pstr, tp, sl, r, job_id = future.result()
                if job_id not in full_results:
                    full_results[job_id] = []
                full_results[job_id].append((name, pstr, tp, sl, r))
                done += 1
                if done % 500 == 0:
                    log(f"    ... {done}/{len(bt_tasks)} done ({time()-t0:.0f}s)")

        elapsed = time() - t0
        log(f"  All backtests done in {elapsed:.0f}s")

        # ── Collect profitable full-period results, then run OOS ──────────────
        log(f"  Collecting profitable combos and running OOS ...")
        t0 = time()
        profitable = []
        for job_id, results in full_results.items():
            for name, pstr, tp, sl, r in results:
                if r["tr"] >= 1 and r["pf"] > 1.0:
                    profitable.append((name, pstr, tp, sl, r, job_id))

        log(f"  {len(profitable)} profitable combos found")

        # Run OOS for profitable combos in parallel
        oos_tasks = []
        for idx, (name, pstr, tp, sl, full_r, job_id) in enumerate(profitable):
            sig_df = jobs[job_id][2]
            oos_tasks.append((sig_df, OOS, ed, tp, sl, qty, name, pstr, idx))

        all_results = []
        if oos_tasks:
            with ProcessPoolExecutor() as pool:
                futures = {pool.submit(_worker, task): idx for idx, task in enumerate(oos_tasks)}
                for future in as_completed(futures):
                    name, pstr, tp, sl, oos_r, idx = future.result()
                    orig = profitable[idx]
                    full_r = orig[4]
                    all_results.append((name, pstr, full_r, oos_r))

        log(f"  OOS done in {time()-t0:.0f}s")

        # ── Print per-strategy summary ────────────────────────────────────────
        log(f"\n  --- Per-strategy summary ---")
        strat_names = sorted(set(r[0] for r in all_results))
        for sn in strat_names:
            subs = [r for r in all_results if r[0] == sn]
            oos_prof = [r for r in subs if r[3]["pf"] > 1.0 and r[3]["tr"] >= 1]
            best_oos = max(subs, key=lambda x: x[3]["pf"]) if subs else None
            log(f"  {sn}: {len(subs)} profitable IS | {len(oos_prof)} profitable OOS")
            if best_oos:
                _, pstr, full, oos = best_oos
                tpd = full["tr"] / max(n_days, 1)
                log(f"    Best OOS: {pstr}")
                log(f"    Full: PF={full['pf']:.3f} WR={full['wr']:.1f}% Tr={full['tr']} ({tpd:.2f}/d) Net={full['net']:+.2f} DD={full['dd']:.2f}%")
                log(f"    OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")

        # ── MASTER LEADERBOARD ────────────────────────────────────────────────
        log(f"\n  {'='*100}")
        log(f"  MASTER LEADERBOARD — {pair} 5-min (top 30 by OOS PF, min 1 OOS trade)")
        log(f"  {'='*100}")
        log(f"  {'#':>3} {'Strategy':>14} {'FullPF':>6} {'Tr':>4} {'Net':>8} {'FullDD':>7} {'OOS_PF':>7} {'OOSTr':>5} {'OOS_DD':>7}  Config")
        log(f"  {'-'*100}")

        ranked = [r for r in all_results if r[3]["tr"] >= 1]
        ranked.sort(key=lambda x: x[3]["pf"], reverse=True)

        for rank, (name, pstr, full, oos) in enumerate(ranked[:30], 1):
            fpf = f"{full['pf']:.2f}" if full["pf"] < 100 else "INF"
            opf = f"{oos['pf']:.2f}" if oos["pf"] < 100 else "INF"
            log(f"  {rank:>3} {name:>14} {fpf:>6} {full['tr']:>4} {full['net']:>+8.2f} {full['dd']:>6.1f}% {opf:>7} {oos['tr']:>5} {oos['dd']:>6.1f}%  {pstr}")

        if not ranked:
            log("  (no profitable combos found)")

    out.close()
    print(f"\n  Done! -> strategies/donchian_showdown_results.txt")


if __name__ == "__main__":
    main()
