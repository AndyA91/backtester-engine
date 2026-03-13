"""
MTF KAMA v1 — Multi-Timeframe KAMA strategy (Multiprocessing version)

Execute on 5-min chart using 1H and 4H KAMA for directional bias.
Higher TF KAMA tells us WHICH direction to trade; 5-min handles entries/exits.

Three variants:
  1. HTF Trend-Follow  — Enter when HTF KAMA rising + price crosses 5m KAMA
  2. Dual-TF Alignment — Both 1H & 4H KAMA slopes agree + 5m KAMA cross
  3. HTF KAMA + RSI     — HTF bias + RSI pullback entry on 5-min

Scan: EURUSD only (5-min execution, 1H + 4H for bias).
"""

import sys
import io
import contextlib
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from time import time

from engine import load_tv_export, BacktestConfig, run_backtest_long_short
from indicators.kama import calc_kama
from indicators.rsi import calc_rsi


# ─── Helpers ──────────────────────────────────────────────────────────────────
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


# ─── Multi-Timeframe Alignment ───────────────────────────────────────────────
def align_htf_to_ltf(ltf_df, htf_df, kama_length, label):
    """
    Compute KAMA on a higher-timeframe DataFrame and forward-fill onto
    the lower-timeframe index using merge_asof.

    Returns numpy arrays (kama_values, kama_slope) aligned to ltf_df's index.
    """
    htf_kama = calc_kama(htf_df["Close"], length=kama_length).rename(f"kama_{label}")
    htf_slope = htf_kama.diff().rename(f"slope_{label}")

    htf_merge = pd.DataFrame({
        f"kama_{label}": htf_kama,
        f"slope_{label}": htf_slope,
    })
    htf_merge.index.name = "Date"
    htf_merge = htf_merge.reset_index()

    ltf_reset = ltf_df.reset_index()
    merged = pd.merge_asof(
        ltf_reset[["Date"]],
        htf_merge,
        on="Date",
        direction="backward",
    )

    kama_vals = merged[f"kama_{label}"].values
    slope_vals = merged[f"slope_{label}"].values
    return kama_vals, slope_vals


# ─── Strategy 1: HTF Trend-Follow ────────────────────────────────────────────
def strat_htf_trend(df, htf_slope, kama_5m, cd_bars=30):
    n = len(df)
    c = df["Close"].values
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; cd = 99999

    for i in range(1, n):
        cd += 1
        if np.isnan(htf_slope[i]) or np.isnan(kama_5m[i]):
            continue

        htf_bull = htf_slope[i] > 0
        htf_bear = htf_slope[i] < 0
        above_k = c[i] > kama_5m[i]
        below_k = c[i] < kama_5m[i]
        was_below = c[i-1] <= kama_5m[i-1] if not np.isnan(kama_5m[i-1]) else False
        was_above = c[i-1] >= kama_5m[i-1] if not np.isnan(kama_5m[i-1]) else False

        if pos == 1 and (below_k or htf_bear):
            lx[i] = True; pos = 0; cd = 0; continue
        if pos == -1 and (above_k or htf_bull):
            sx[i] = True; pos = 0; cd = 0; continue

        if pos == 0 and cd >= cd_bars:
            cross_up = above_k and was_below
            cross_dn = below_k and was_above
            if cross_up and htf_bull:
                le[i] = True; pos = 1; cd = 0
            elif cross_dn and htf_bear:
                se[i] = True; pos = -1; cd = 0

    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    return df2


# ─── Strategy 2: Dual-TF KAMA Alignment ──────────────────────────────────────
def strat_dual_tf(df, slope_1h, slope_4h, kama_5m, cd_bars=30):
    n = len(df)
    c = df["Close"].values
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; cd = 99999

    for i in range(1, n):
        cd += 1
        if (np.isnan(slope_1h[i]) or np.isnan(slope_4h[i])
                or np.isnan(kama_5m[i]) or np.isnan(kama_5m[i-1])):
            continue

        both_bull = slope_1h[i] > 0 and slope_4h[i] > 0
        both_bear = slope_1h[i] < 0 and slope_4h[i] < 0
        above_k = c[i] > kama_5m[i]
        below_k = c[i] < kama_5m[i]
        was_below = c[i-1] <= kama_5m[i-1]
        was_above = c[i-1] >= kama_5m[i-1]

        if pos == 1 and (below_k or not both_bull):
            lx[i] = True; pos = 0; cd = 0; continue
        if pos == -1 and (above_k or not both_bear):
            sx[i] = True; pos = 0; cd = 0; continue

        if pos == 0 and cd >= cd_bars:
            cross_up = above_k and was_below
            cross_dn = below_k and was_above
            if cross_up and both_bull:
                le[i] = True; pos = 1; cd = 0
            elif cross_dn and both_bear:
                se[i] = True; pos = -1; cd = 0

    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    return df2


# ─── Strategy 3: HTF KAMA + RSI Pullback ─────────────────────────────────────
def strat_htf_rsi(df, htf_slope, rsi_vals, rsi_os=35, rsi_ob=65, cd_bars=30):
    n = len(df)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; cd = 99999

    for i in range(1, n):
        cd += 1
        if np.isnan(htf_slope[i]) or np.isnan(rsi_vals[i]):
            continue

        htf_bull = htf_slope[i] > 0
        htf_bear = htf_slope[i] < 0
        rsi_cross_up = rsi_vals[i-1] <= rsi_os and rsi_vals[i] > rsi_os
        rsi_cross_dn = rsi_vals[i-1] >= rsi_ob and rsi_vals[i] < rsi_ob
        rsi_mid_up = rsi_vals[i-1] < 50 and rsi_vals[i] >= 50
        rsi_mid_dn = rsi_vals[i-1] > 50 and rsi_vals[i] <= 50

        if pos == 1 and (rsi_mid_up or htf_bear):
            lx[i] = True; pos = 0; cd = 0; continue
        if pos == -1 and (rsi_mid_dn or htf_bull):
            sx[i] = True; pos = 0; cd = 0; continue

        if pos == 0 and cd >= cd_bars:
            if rsi_cross_up and htf_bull:
                le[i] = True; pos = 1; cd = 0
            elif rsi_cross_dn and htf_bear:
                se[i] = True; pos = -1; cd = 0

    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    return df2


# ─── Workers for Multiprocessing ──────────────────────────────────────────────
def _worker_1(args):
    df, htf_slope, kama5, cd, tp, sl, sd, ed, oos_date, qty, params = args
    ds = strat_htf_trend(df, htf_slope, kama5, cd)
    r = ex(run_bt(ds, sd, ed, tp, sl, qty))
    
    if r["tr"] >= 10:
        oos = ex(run_bt(ds, oos_date, ed, tp, sl, qty))
        return (r["pf"], params, r, oos)
    return None

def _worker_2(args):
    df, slope_1h, slope_4h, kama5, cd, tp, sl, sd, ed, oos_date, qty, params = args
    ds = strat_dual_tf(df, slope_1h, slope_4h, kama5, cd)
    r = ex(run_bt(ds, sd, ed, tp, sl, qty))
    
    if r["tr"] >= 5:
        oos = ex(run_bt(ds, oos_date, ed, tp, sl, qty))
        return (r["pf"], params, r, oos)
    return None

def _worker_3(args):
    df, htf_slope, rsi_arr, os_v, ob_v, cd, tp, sl, sd, ed, oos_date, qty, params = args
    ds = strat_htf_rsi(df, htf_slope, rsi_arr, os_v, ob_v, cd)
    r = ex(run_bt(ds, sd, ed, tp, sl, qty))
    
    if r["tr"] >= 5:
        oos = ex(run_bt(ds, oos_date, ed, tp, sl, qty))
        return (r["pf"], params, r, oos)
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    rp  = Path(__file__).resolve().parent / "mtf_kama_results.txt"
    out = open(rp, "w", encoding="utf-8")

    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    num_cores = max(1, multiprocessing.cpu_count() - 1)

    log("MTF KAMA v1 — Multi-Timeframe KAMA (5m exec + 1H/4H bias)")
    log("Commission: 0.0043% | $1000 initial | EURUSD")
    log(f"Using {num_cores} logical cores for multiprocessing.\n")

    # ── Load all three timeframes ─────────────────────────────────────────
    t0 = time()
    df_5m = load_tv_export("OANDA_EURUSD, 5.csv")
    df_1h = load_tv_export("OANDA_EURUSD, 60.csv")
    df_4h = load_tv_export("OANDA_EURUSD, 240.csv")
    log(f"  Loaded 5m={len(df_5m):,}  1H={len(df_1h):,}  4H={len(df_4h):,} bars ({time()-t0:.1f}s)")

    sd = str(df_5m.index[0].date())
    ed = str(df_5m.index[-1].date())
    n_days = max(1, np.busday_count(df_5m.index[0].date(), df_5m.index[-1].date()))

    # OOS split — last ~25% of 5-min data
    oos_idx = int(len(df_5m) * 0.75)
    oos_date = str(df_5m.index[oos_idx].date())
    log(f"  Period: {sd} to {ed} ({n_days} trading days)")
    log(f"  OOS split from: {oos_date}\n")

    qty = 1000.0

    # ── Parameter grid ────────────────────────────────────────────────────
    htf_lengths = [10, 21, 30]
    k5m_lengths = [5, 10, 14]
    cd_vals     = [12, 30, 60]
    tp_vals     = [0.0, 0.10, 0.15]
    sl_vals     = [0.0, 0.05, 0.10]
    rsi_periods = [7, 14]

    # ── Precompute KAMA caches ────────────────────────────────────────────
    log("  Precomputing KAMA and RSI caches...")
    t0 = time()

    kama_5m_cache = {}
    for kl in k5m_lengths:
        kama_5m_cache[kl] = calc_kama(df_5m["Close"], length=kl).values

    htf_1h_cache = {}
    htf_4h_cache = {}
    for kl in htf_lengths:
        htf_1h_cache[kl] = align_htf_to_ltf(df_5m, df_1h, kl, f"1h_{kl}")
        htf_4h_cache[kl] = align_htf_to_ltf(df_5m, df_4h, kl, f"4h_{kl}")

    rsi_cache = {}
    for rp_ in rsi_periods:
        rsi_cache[rp_] = calc_rsi(df_5m, rp_)["rsi"]

    log(f"  Done in {time()-t0:.1f}s\n")

    all_boards = {}

    def _run_parallel(task_name, worker_fn, tasks_list):
        log(f"  {'='*70}")
        log(f"  {task_name}")
        log(f"  {'='*70}")
        t0 = time()
        board = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(worker_fn, tasks_list), total=len(tasks_list), desc=task_name))
            for result in results:
                if result:
                    board.append(result)
        
        board.sort(key=lambda x: -x[0])
        log(f"  Scanned {len(tasks_list)} combos in {time()-t0:.0f}s  |  {len(board)} met threshold\n")

        if board:
            keys = list(board[0][1].keys())
            headers = " ".join([f"{k:>4}" for k in keys])
            log(f"  {headers}  {'PF':>6} {'WR%':>5} {'Tr':>4} {'Net':>8} {'DD%':>6}  {'osPF':>6} {'osTr':>4} {'osDD':>6}")
            log(f"  {'-'*100}")
            for row in board[:10]:
                pf, p, r, o = row
                pf_s = f"{pf:.2f}" if pf < 100 else "INF"
                opf  = f"{o['pf']:.2f}" if o["pf"] < 100 else "INF"
                p_str = " ".join([f"{v:>4}" for v in p.values()])
                log(f"  {p_str}  {pf_s:>6} {r['wr']:>5.1f} {r['tr']:>4} {r['net']:>+8.2f} {r['dd']:>6.2f}%  "
                    f"{opf:>6} {o['tr']:>4} {o['dd']:>6.2f}%")
        all_boards[task_name] = board


    # ── Strategy 1: HTF Trend-Follow (4H) ───────────────────────────────
    tasks = []
    for htf_len in htf_lengths:
        _, htf_slope = htf_4h_cache[htf_len]
        for k5 in k5m_lengths:
            kama5 = kama_5m_cache[k5]
            for cd in cd_vals:
                for tp in tp_vals:
                    for sl in sl_vals:
                        params = {"htf": htf_len, "k5": k5, "cd": cd, "tp": tp, "sl": sl}
                        tasks.append((df_5m, htf_slope, kama5, cd, tp, sl, sd, ed, oos_date, qty, params))
    _run_parallel("Strategy 1: HTF Trend-Follow (4H KAMA bias + 5m KAMA cross)", _worker_1, tasks)

    # ── Strategy 1b: HTF Trend-Follow (1H) ──────────────────────────────
    tasks = []
    for htf_len in htf_lengths:
        _, htf_slope = htf_1h_cache[htf_len]
        for k5 in k5m_lengths:
            kama5 = kama_5m_cache[k5]
            for cd in cd_vals:
                for tp in tp_vals:
                    for sl in sl_vals:
                        params = {"htf": htf_len, "k5": k5, "cd": cd, "tp": tp, "sl": sl}
                        tasks.append((df_5m, htf_slope, kama5, cd, tp, sl, sd, ed, oos_date, qty, params))
    _run_parallel("Strategy 1b: HTF Trend-Follow (1H KAMA bias + 5m KAMA cross)", _worker_1, tasks)

    # ── Strategy 2: Dual-TF KAMA Alignment ───────────────────────────────
    tasks = []
    for h1_len in htf_lengths:
        _, slope_1h = htf_1h_cache[h1_len]
        for h4_len in htf_lengths:
            _, slope_4h = htf_4h_cache[h4_len]
            for k5 in k5m_lengths:
                kama5 = kama_5m_cache[k5]
                for cd in cd_vals:
                    for tp in tp_vals:
                        for sl in sl_vals:
                            params = {"h1": h1_len, "h4": h4_len, "k5": k5, "cd": cd, "tp": tp, "sl": sl}
                            tasks.append((df_5m, slope_1h, slope_4h, kama5, cd, tp, sl, sd, ed, oos_date, qty, params))
    _run_parallel("Strategy 2: Dual-TF Alignment (1H + 4H KAMA agree + 5m cross)", _worker_2, tasks)

    # ── Strategy 3: HTF KAMA + RSI Pullback (4H) ─────────────────────────
    tasks = []
    for htf_len in htf_lengths:
        _, htf_slope = htf_4h_cache[htf_len]
        for rp_ in rsi_periods:
            rsi = rsi_cache[rp_]
            for os_v in [30, 35]:
                ob_v = 100 - os_v
                for cd in cd_vals:
                    for tp in tp_vals:
                        for sl in sl_vals:
                            params = {"htf": htf_len, "rsi": rp_, "os": os_v, "cd": cd, "tp": tp, "sl": sl}
                            tasks.append((df_5m, htf_slope, rsi, os_v, ob_v, cd, tp, sl, sd, ed, oos_date, qty, params))
    _run_parallel("Strategy 3: HTF KAMA + RSI Pullback (4H bias + 5m RSI entries)", _worker_3, tasks)

    # ── Strategy 3b: HTF KAMA + RSI Pullback (1H) ────────────────────────
    tasks = []
    for htf_len in htf_lengths:
        _, htf_slope = htf_1h_cache[htf_len]
        for rp_ in rsi_periods:
            rsi = rsi_cache[rp_]
            for os_v in [30, 35]:
                ob_v = 100 - os_v
                for cd in cd_vals:
                    for tp in tp_vals:
                        for sl in sl_vals:
                            params = {"htf": htf_len, "rsi": rp_, "os": os_v, "cd": cd, "tp": tp, "sl": sl}
                            tasks.append((df_5m, htf_slope, rsi, os_v, ob_v, cd, tp, sl, sd, ed, oos_date, qty, params))
    _run_parallel("Strategy 3b: HTF KAMA + RSI Pullback (1H bias + 5m RSI entries)", _worker_3, tasks)

    # ── Grand Leaderboard ──────────────────────────────────────────────────
    log(f"\n{'#'*80}")
    log(f"  GRAND LEADERBOARD — Best from each variant (by Full PF)")
    log(f"{'#'*80}")
    log(f"  {'#':>3} {'Strategy':<56} {'PF':>6} {'WR%':>5} {'Tr':>4} {'Net':>9} {'DD%':>6}  "
        f"{'osPF':>6} {'osTr':>4} {'osDD':>6}")
    log(f"  {'-'*110}")

    rank = 0
    for name, board in sorted(all_boards.items(),
                               key=lambda kv: kv[1][0][0] if kv[1] else 0,
                               reverse=True):
        if not board:
            continue
        rank += 1
        pf, p, r, o = board[0]
        pf_s = f"{pf:.2f}" if pf < 100 else "INF"
        opf  = f"{o['pf']:.2f}" if o["pf"] < 100 else "INF"
        log(f"  {rank:>3} {name:<56} {pf_s:>6} {r['wr']:>5.1f} {r['tr']:>4} {r['net']:>+9.2f} {r['dd']:>6.2f}%  "
            f"{opf:>6} {o['tr']:>4} {o['dd']:>6.2f}%")
        log(f"      params: {p}")

    out.close()
    print(f"\n  Done! -> strategies/mtf_kama_results.txt")


if __name__ == "__main__":
    # Ensure multiprocessing executes properly on Windows
    multiprocessing.freeze_support()
    main()
