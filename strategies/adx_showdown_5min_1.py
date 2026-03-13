"""
ADX Showdown v1 — 5-min edition
Same 4 ADX architectures as adx_showdown_1.py but on 5-minute data.
Robustness check: does the ADX regime filter hold up at 5-min?
"""

import sys
import io
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time import time

from engine import load_tv_export, BacktestConfig, run_backtest_long_short
from strategies.adx_showdown_1 import (
    strat_adx_cross,
    strat_vwap_adx,
    strat_rsi_adx,
    strat_st_adx,
)


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


def main():
    TS = "2026-02-17"
    rp = Path(__file__).resolve().parent / "adx_showdown_5min_results.txt"
    out = open(rp, "w", encoding="utf-8")
    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    log("ADX SHOWDOWN v1 — 5-min edition — 4 architectures x 2 pairs")
    log(f"Commission: 0.0043% | $1000 initial | OOS from {TS}\n")

    for df_file, pair, qty in [
        ("OANDA_EURUSD, 5.csv", "EURUSD", 1000),
        ("OANDA_GBPJPY, 5.csv", "GBPJPY", 100),
    ]:
        df  = load_tv_export(df_file)
        sd  = str(df.index[0].date())
        ed  = str(df.index[-1].date())
        log(f"\n{'#'*80}")
        log(f"  {pair} 5-min ({len(df):,} bars, {qty} units)")
        log(f"{'#'*80}")
        board = []

        # ── 1. ADX Crossover ─────────────────────────────────────────────────
        t0 = time(); best = None
        for dip in [7, 14, 21]:
            for thresh in [20, 25, 30]:
                for cd in [20, 60]:
                    ds = strat_adx_cross(df, dip, thresh, cd)
                    for sl in [0.0, 0.10]:
                        r = ex(run_bt(ds, sd, ed, 0, sl, qty))
                        if r["tr"] >= 10 and (not best or r["pf"] > best["pf"]):
                            best = {**r, "dip": dip, "thresh": thresh, "cd": cd, "sl": sl}
        if best:
            ds  = strat_adx_cross(df, best["dip"], best["thresh"], best["cd"])
            oos = ex(run_bt(ds, TS, ed, 0, best["sl"], qty))
            log(f"\n  1. ADX Crossover  ({time()-t0:.0f}s)")
            log(f"     DI={best['dip']} ADX>{best['thresh']} CD={best['cd']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("ADX Cross", best, oos))

        # ── 2. VWAP + ADX Regime ─────────────────────────────────────────────
        t0 = time(); best = None
        for bm in [1.5, 2.0, 2.5, 3.0]:
            for cd in [30, 60, 100]:
                for thresh in [20, 25, 30, 35]:
                    ds = strat_vwap_adx(df, bm, cd, thresh)
                    for tp in [0.0, 0.15]:
                        for sl in [0.0, 0.10]:
                            r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                            if r["tr"] >= 10 and (not best or r["pf"] > best["pf"]):
                                best = {**r, "bm": bm, "cd": cd, "thresh": thresh, "tp": tp, "sl": sl}
        if best:
            ds  = strat_vwap_adx(df, best["bm"], best["cd"], best["thresh"])
            oos = ex(run_bt(ds, TS, ed, best["tp"], best["sl"], qty))
            log(f"\n  2. VWAP+ADX Regime  ({time()-t0:.0f}s)")
            log(f"     Band={best['bm']}sig CD={best['cd']} ADX<{best['thresh']} TP={best['tp']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("VWAP+ADX", best, oos))

        # ── 3. RSI + ADX Regime ──────────────────────────────────────────────
        t0 = time(); best = None
        for per in [7, 14, 21]:
            for ob in [70, 75]:
                for cd in [20, 60]:
                    for thresh in [20, 25, 30, 35]:
                        ds = strat_rsi_adx(df, per, 100-ob, ob, cd, thresh)
                        for sl in [0.0, 0.10]:
                            r = ex(run_bt(ds, sd, ed, 0, sl, qty))
                            if r["tr"] >= 5 and (not best or r["pf"] > best["pf"]):
                                best = {**r, "per": per, "ob": ob, "cd": cd, "thresh": thresh, "sl": sl}
        if best:
            ds  = strat_rsi_adx(df, best["per"], 100-best["ob"], best["ob"], best["cd"], best["thresh"])
            oos = ex(run_bt(ds, TS, ed, 0, best["sl"], qty))
            log(f"\n  3. RSI+ADX Regime  ({time()-t0:.0f}s)")
            log(f"     P={best['per']} OB/OS={best['ob']}/{100-best['ob']} CD={best['cd']} ADX<{best['thresh']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("RSI+ADX", best, oos))

        # ── 4. Supertrend + ADX ──────────────────────────────────────────────
        t0 = time(); best = None
        for per in [7, 10, 14]:
            for mult in [2.0, 3.0, 4.0]:
                for thresh in [20, 25, 30]:
                    ds = strat_st_adx(df, per, mult, thresh)
                    for sl in [0.0, 0.10, 0.15]:
                        r = ex(run_bt(ds, sd, ed, 0, sl, qty))
                        if r["tr"] >= 10 and (not best or r["pf"] > best["pf"]):
                            best = {**r, "per": per, "mult": mult, "thresh": thresh, "sl": sl}
        if best:
            ds  = strat_st_adx(df, best["per"], best["mult"], best["thresh"])
            oos = ex(run_bt(ds, TS, ed, 0, best["sl"], qty))
            log(f"\n  4. Supertrend+ADX  ({time()-t0:.0f}s)")
            log(f"     P={best['per']} M={best['mult']} ADX>{best['thresh']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("ST+ADX", best, oos))

        # ── Leaderboard ──────────────────────────────────────────────────────
        log(f"\n  {'='*75}")
        log(f"  LEADERBOARD — {pair} 5-min (sorted by OOS PF)")
        log(f"  {'='*75}")
        log(f"  {'#':>3} {'Strategy':>14}  {'FullPF':>6} {'FullTr':>6} {'FullDD':>7}  {'OOS_PF':>6} {'OOSTr':>5} {'OOS_DD':>7}")
        log(f"  {'-'*75}")
        for rank, (name, full, oos) in enumerate(
            sorted(board, key=lambda x: x[2]["pf"], reverse=True), 1
        ):
            fpf = f"{full['pf']:.2f}" if full["pf"] < 100 else "INF"
            opf = f"{oos['pf']:.2f}"  if oos["pf"]  < 100 else "INF"
            log(f"  {rank:>3} {name:>14}  {fpf:>6} {full['tr']:>6} {full['dd']:>6.1f}%  {opf:>6} {oos['tr']:>5} {oos['dd']:>6.1f}%")

    out.close()
    print(f"\n  Done! -> strategies/adx_showdown_5min_results.txt")


if __name__ == "__main__":
    main()
