"""
ADX Showdown v1 — Testing ADX in 3 roles across EURUSD and GBPJPY 3-min

4 architectures:
  1. ADX Crossover        — standalone +DI/-DI cross with ADX strength gate
  2. VWAP + ADX Regime    — VWAP band reversion only when ranging (ADX < thresh)
  3. RSI  + ADX Regime    — RSI reversion only when ranging (ADX < thresh)
  4. Supertrend + ADX     — Supertrend flips only when trending (ADX > thresh)
"""

import sys
import io
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from time import time

from engine import load_tv_export, BacktestConfig, run_backtest_long_short
from indicators.vwap import calc_vwap
from indicators.rsi import calc_rsi
from indicators.supertrend import calc_supertrend
from indicators.adx import calc_adx


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


# =============================================================================
# 1. ADX CROSSOVER — +DI crosses -DI with ADX strength gate
# =============================================================================
def strat_adx_cross(df, di_period=14, adx_thresh=25, cd=20):
    c = df["Close"].values; n = len(c)
    res = calc_adx(df, di_period=di_period, adx_period=14)
    pdi = res["plus_di"]; mdi = res["minus_di"]; adx = res["adx"]
    df2 = df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool)
    se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0; bst=cd
    for i in range(2, n):
        bst += 1
        cross_up   = pdi[i-1] <= mdi[i-1] and pdi[i] > mdi[i]
        cross_down = pdi[i-1] >= mdi[i-1] and pdi[i] < mdi[i]
        trending   = adx[i] >= adx_thresh
        # exits — no ADX gate
        if pos == 1  and cross_down: lx[i]=True; pos=0; bst=0
        elif pos ==-1 and cross_up:  sx[i]=True; pos=0; bst=0
        # entries — require trending
        if pos == 0 and bst >= cd and trending:
            if cross_up:   le[i]=True; pos=1;  bst=0
            elif cross_down: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se
    df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


# =============================================================================
# 2. VWAP + ADX REGIME — reversion only when ranging (ADX < thresh)
#    Keeps best-known RSI<35 / RSI>65 filter
# =============================================================================
def strat_vwap_adx(df, bm=1.5, cd=30, adx_thresh=25):
    c = df["Close"].values; n = len(c)
    vr  = calc_vwap(df, band_mult_2=bm)
    vw=vr["vwap"]; u2=vr["upper2"]; l2=vr["lower2"]
    rsi = calc_rsi(df, 14)["rsi"]
    adx = calc_adx(df, adx_period=14)["adx"]
    df2 = df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool)
    se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0; bst=cd
    for i in range(2, n):
        bst += 1
        ls = c[i-1] <= l2[i-1] and c[i] > l2[i] and rsi[i] < 35
        ss = c[i-1] >= u2[i-1] and c[i] < u2[i] and rsi[i] > 65
        cuv = c[i-1] <= vw[i-1] and c[i] > vw[i]
        cdv = c[i-1] >= vw[i-1] and c[i] < vw[i]
        # exits — always allowed
        if pos == 1  and (cuv or ss): lx[i]=True; pos=0; bst=0
        elif pos ==-1 and (cdv or ls): sx[i]=True; pos=0; bst=0
        # entries — only when ranging
        if pos == 0 and bst >= cd and adx[i] < adx_thresh:
            if ls: le[i]=True; pos=1;  bst=0
            elif ss: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se
    df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


# =============================================================================
# 3. RSI + ADX REGIME — reversion only when ranging (ADX < thresh)
#    Keeps VWAP direction filter
# =============================================================================
def strat_rsi_adx(df, per=14, os_=25, ob=75, cd=20, adx_thresh=25):
    c = df["Close"].values; n = len(c)
    r   = calc_rsi(df, per)["rsi"]
    vw  = calc_vwap(df)["vwap"]
    adx = calc_adx(df, adx_period=14)["adx"]
    df2 = df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool)
    se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0; bst=cd
    for i in range(2, n):
        bst += 1
        ls  = r[i-1] <= os_ and r[i] > os_
        ss  = r[i-1] >= ob  and r[i] < ob
        vl  = c[i] > vw[i]
        vs  = c[i] < vw[i]
        xl  = r[i-1] < 50 and r[i] >= 50
        xs  = r[i-1] > 50 and r[i] <= 50
        # exits — always allowed
        if pos == 1  and (xl or ss): lx[i]=True; pos=0; bst=0
        elif pos ==-1 and (xs or ls): sx[i]=True; pos=0; bst=0
        # entries — only when ranging
        if pos == 0 and bst >= cd and adx[i] < adx_thresh:
            if ls and vl: le[i]=True; pos=1;  bst=0
            elif ss and vs: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se
    df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


# =============================================================================
# 4. SUPERTREND + ADX — enter flips only when trending (ADX > thresh)
#    Exits always allowed (don't hold a losing trend)
# =============================================================================
def strat_st_adx(df, per=10, mult=3.0, adx_thresh=25):
    c = df["Close"].values; n = len(c)
    d   = calc_supertrend(df, per, mult)["direction"]
    adx = calc_adx(df, adx_period=14)["adx"]
    df2 = df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool)
    se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0
    for i in range(2, n):
        fl = d[i-1] == -1 and d[i] == 1   # bearish→bullish flip
        fs = d[i-1] ==  1 and d[i] == -1  # bullish→bearish flip
        # exits — always
        if pos == 1  and fs: lx[i]=True; pos=0
        elif pos ==-1 and fl: sx[i]=True; pos=0
        # entries — only when trending
        if pos == 0 and adx[i] >= adx_thresh:
            if fl: le[i]=True; pos=1
            elif fs: se[i]=True; pos=-1
    df2["long_entry"]=le; df2["long_exit"]=lx|se
    df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


def main():
    TS = "2026-02-17"
    rp = Path(__file__).resolve().parent / "adx_showdown_results.txt"
    out = open(rp, "w", encoding="utf-8")
    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    log("ADX SHOWDOWN v1 — 4 architectures × 2 pairs")
    log(f"Commission: 0.0043% | $1000 initial | OOS from {TS}\n")

    for df_file, pair, qty in [
        ("OANDA_EURUSD, 3.csv", "EURUSD", 1000),
        ("OANDA_GBPJPY, 3.csv", "GBPJPY", 100),
    ]:
        df  = load_tv_export(df_file)
        sd  = str(df.index[0].date())
        ed  = str(df.index[-1].date())
        log(f"\n{'#'*80}")
        log(f"  {pair} ({len(df):,} bars, {qty} units)")
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
        for bm in [1.5, 2.0, 2.5]:
            for cd in [30, 60]:
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
        log(f"  LEADERBOARD — {pair} (sorted by OOS PF)")
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
    print(f"\n  Done! -> strategies/adx_showdown_results.txt")


if __name__ == "__main__":
    main()
