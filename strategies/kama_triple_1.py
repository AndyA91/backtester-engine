"""
Triple KAMA MA v1 — Trend-following via three adaptive moving average alignment

Three KAMA lines at different lengths (fast / mid / slow).
Entry: first bar where all three align (fast > mid > slow = long, reverse = short).
Exit:  alignment breaks in any direction.

Scan: EURUSD + GBPJPY 5-min, full period only (single pass).
"""

import sys
import io
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from time import time

from engine import load_tv_export, BacktestConfig, run_backtest_long_short
from indicators.kama import calc_kama


# ─── Runner ──────────────────────────────────────────────────────────────────
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


# ─── Strategy ─────────────────────────────────────────────────────────────────
def strat_triple_kama(df, kf, km, ks, cd_bars=10):
    """
    Triple KAMA alignment strategy.

    Parameters
    ----------
    df      : DataFrame (OHLCV)
    kf      : numpy array — fast KAMA values
    km      : numpy array — mid KAMA values
    ks      : numpy array — slow KAMA values
    cd_bars : cooldown bars after any exit before new entry allowed
    """
    n = len(df)
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)

    # Precompute alignment booleans (vectorised)
    bull = (kf > km) & (km > ks)
    bear = (kf < km) & (km < ks)

    pos = 0
    cd  = 99999  # start ready to trade

    for i in range(1, n):
        cd += 1
        b = bool(bull[i]); p = bool(bear[i])

        # ── Exits (alignment breaks) ──────────────────────────────────────────
        if pos == 1 and not b:
            lx[i] = True; pos = 0; cd = 0; continue
        if pos == -1 and not p:
            sx[i] = True; pos = 0; cd = 0; continue

        # ── Entries (first bar of new alignment) ──────────────────────────────
        if pos == 0 and cd >= cd_bars:
            b_prev = bool(bull[i - 1])
            p_prev = bool(bear[i - 1])
            if b and not b_prev:
                le[i] = True; pos = 1; cd = 0
            elif p and not p_prev:
                se[i] = True; pos = -1; cd = 0

    df2 = df.copy()
    df2["long_entry"]  = le; df2["long_exit"]  = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    return df2


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    rp  = Path(__file__).resolve().parent / "kama_triple_results.txt"
    out = open(rp, "w", encoding="utf-8")

    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    log("Triple KAMA MA v1 — Three-line alignment (fast/mid/slow)")
    log("Commission: 0.0043% | $1000 initial | Full period scan\n")

    datasets = [
        ("OANDA_EURUSD, 5.csv", "EURUSD", 1000),
        ("OANDA_GBPJPY, 5.csv", "GBPJPY", 100),
    ]

    # Parameter grid
    fast_ps = [3, 5, 8]
    mid_ps  = [14, 21]
    slow_ps = [40, 60, 80]
    cd_vals = [10, 30]
    tp_vals = [0.0, 0.05, 0.10, 0.15]
    sl_vals = [0.0, 0.05, 0.10]

    for df_file, pair, qty in datasets:
        df  = load_tv_export(df_file)
        sd  = str(df.index[0].date())
        ed  = str(df.index[-1].date())
        n_days = np.busday_count(df.index[0].date(), df.index[-1].date())

        log(f"\n{'#'*72}")
        log(f"  {pair} 5-min  ({n_days} trading days)  qty={qty}")
        log(f"{'#'*72}")

        # ── Precompute all unique KAMA series ─────────────────────────────────
        all_lengths = sorted(set(fast_ps + mid_ps + slow_ps))
        log(f"  Precomputing KAMA for lengths: {all_lengths} ...")
        t0 = time()
        kama_cache = {l: calc_kama(df["Close"], length=l).values for l in all_lengths}
        log(f"  Done in {time()-t0:.1f}s\n")

        # ── Single-pass scan — collect all results ────────────────────────────
        board = []; count = 0
        for fp in fast_ps:
            for mp in mid_ps:
                if mp <= fp:
                    continue
                for sp in slow_ps:
                    if sp <= mp:
                        continue
                    for cd in cd_vals:
                        ds = strat_triple_kama(df,
                                               kama_cache[fp],
                                               kama_cache[mp],
                                               kama_cache[sp],
                                               cd)
                        for tp in tp_vals:
                            for sl in sl_vals:
                                r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                                count += 1
                                if r["tr"] >= 5:
                                    board.append((r["pf"], fp, mp, sp, cd, tp, sl, r))

        board.sort(key=lambda x: -x[0])
        log(f"  Scanned {count} combos  |  {len(board)} met min-trades threshold.")

        if board:
            pf, fp, mp, sp, cd, tp, sl, best = board[0]
            tpd = best["tr"] / n_days
            log(f"\n  Best (sorted by Full PF, min 5 trades):")
            log(f"    KAMA lengths: fast={fp}  mid={mp}  slow={sp}")
            log(f"    CD={cd} bars  TP={tp}%  SL={sl}%")
            log(f"    PF={pf:.3f}  WR={best['wr']:.1f}%  "
                f"Tr={best['tr']} ({tpd:.2f}/day)  Net={best['net']:+.2f}  DD={best['dd']:.2f}%")
        else:
            log("\n  No combo met min-trades threshold.")

        # ── Top-5 leaderboard from cached results ────────────────────────────
        log(f"\n  {'—'*60}")
        log(f"  Top 5 by Full PF (min 5 trades):")
        log(f"  {'fast':>4} {'mid':>4} {'slow':>4} {'cd':>4} {'tp':>5} {'sl':>5} "
            f"{'PF':>6} {'WR%':>5} {'Tr':>4} {'Tr/d':>6} {'DD%':>6}")
        log(f"  {'—'*60}")

        for row in board[:5]:
            pf, fp, mp, sp, cd, tp, sl, r = row
            tpd = r["tr"] / n_days
            pf_s = f"{pf:.2f}" if pf < 100 else "INF"
            log(f"  {fp:>4} {mp:>4} {sp:>4} {cd:>4} {tp:>5} {sl:>5} "
                f"{pf_s:>6} {r['wr']:>5.1f} {r['tr']:>4} {tpd:>6.2f} {r['dd']:>6.2f}%")

    out.close()
    print(f"\n  Done -> strategies/kama_triple_results.txt")


if __name__ == "__main__":
    main()
