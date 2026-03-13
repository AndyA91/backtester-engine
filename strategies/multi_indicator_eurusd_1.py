"""
STRATEGY SHOWDOWN v2 — LEAN (reduced param space for speed)
5 architectures × 2 pairs, ~500 total backtests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from time import time

from engine import load_tv_export, BacktestConfig, run_backtest_long_short
from strategies.gaussian_channel_eurusd_5 import gaussian_iir_alpha, gaussian_npole_iir
from indicators.vwap import calc_vwap
from indicators.kama import calc_kama
from indicators.atr import calc_atr
from indicators.rsi import calc_rsi
from indicators.supertrend import calc_supertrend


def run_bt(df, sd, ed, tp=0.0, sl=0.0, qty=1000.0):
    cfg = BacktestConfig(
        initial_capital=1000.0, commission_pct=0.0043, slippage_ticks=0,
        qty_type="fixed", qty_value=qty, pyramiding=1,
        start_date=sd, end_date=ed, take_profit_pct=tp, stop_loss_pct=sl,
    )
    return run_backtest_long_short(df, cfg)


def ex(kpis):
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf": min(pf, 999.0),  # cap for sorting
        "wr": kpis.get("win_rate", 0.0) or 0.0,
        "tr": kpis.get("total_trades", 0),
        "net": kpis.get("net_profit", 0.0) or 0.0,
        "dd": kpis.get("max_drawdown_pct", 0.0) or 0.0,
        "avg": kpis.get("avg_trade", 0.0) or 0.0,
    }


# ===========================================================================
# STRATEGY BUILDERS
# ===========================================================================
def strat_gc(df, per=110, mult=4.0, cd=120, use_vwap=False, use_kama=False):
    c = df["Close"].values; n = len(c)
    pc = np.roll(c, 1); pc[0] = c[0]
    a = gaussian_iir_alpha(per, 4)
    mid = gaussian_npole_iir(a, c, 4)
    tr = np.maximum(df["High"].values - df["Low"].values,
                    np.maximum(np.abs(df["High"].values - pc), np.abs(df["Low"].values - pc)))
    ftr = gaussian_npole_iir(a, tr, 4)
    up = mid + ftr * mult; lo = mid - ftr * mult
    vw = calc_vwap(df)["vwap"] if use_vwap else None
    ka = calc_kama(df["Close"]).values if use_kama else None
    df2 = df.copy()
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    for i in range(2, n):
        bst += 1
        if pos == 1 and c[i-1] >= mid[i-1] and c[i] < mid[i]: lx[i]=True; pos=0; bst=0
        elif pos == -1 and c[i-1] <= mid[i-1] and c[i] > mid[i]: sx[i]=True; pos=0; bst=0
        if pos == 0 and bst >= cd:
            rl = c[i-1] <= lo[i-1] and c[i] > lo[i]
            rs = c[i-1] >= up[i-1] and c[i] < up[i]
            vl = (not use_vwap) or c[i] > vw[i]
            vs = (not use_vwap) or c[i] < vw[i]
            kl = (not use_kama) or ka[i] > ka[i-1]
            ks = (not use_kama) or ka[i] < ka[i-1]
            if rl and vl and kl: le[i]=True; pos=1; bst=0
            elif rs and vs and ks: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se; df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


def strat_vwap(df, bm=2.0, cd=60, rsi_f=False, rsi_per=14):
    c = df["Close"].values; n = len(c)
    vr = calc_vwap(df, band_mult_2=bm)
    vw=vr["vwap"]; u2=vr["upper2"]; l2=vr["lower2"]
    rsi = calc_rsi(df, rsi_per)["rsi"] if rsi_f else None
    df2 = df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool); se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0; bst=cd
    for i in range(2, n):
        bst += 1
        ls = c[i-1] <= l2[i-1] and c[i] > l2[i]
        ss = c[i-1] >= u2[i-1] and c[i] < u2[i]
        if rsi_f:
            ls = ls and rsi[i] < 35
            ss = ss and rsi[i] > 65
        cuv = c[i-1] <= vw[i-1] and c[i] > vw[i]
        cdv = c[i-1] >= vw[i-1] and c[i] < vw[i]
        if pos==1 and (cuv or ss): lx[i]=True; pos=0; bst=0
        elif pos==-1 and (cdv or ls): sx[i]=True; pos=0; bst=0
        if pos==0 and bst>=cd:
            if ls: le[i]=True; pos=1; bst=0
            elif ss: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se; df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


def strat_st(df, per=10, mult=3.0, cd=10, use_vwap=False):
    c = df["Close"].values; n = len(c)
    d = calc_supertrend(df, per, mult)["direction"]
    vw = calc_vwap(df)["vwap"] if use_vwap else None
    df2 = df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool); se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0; bst=cd
    for i in range(2, n):
        bst += 1
        fl = d[i-1]==-1 and d[i]==1
        fs = d[i-1]==1 and d[i]==-1
        vl = (not use_vwap) or c[i]>vw[i]
        vs = (not use_vwap) or c[i]<vw[i]
        if pos==1 and fs: lx[i]=True; pos=0; bst=0
        elif pos==-1 and fl: sx[i]=True; pos=0; bst=0
        if pos==0 and bst>=cd:
            if fl and vl: le[i]=True; pos=1; bst=0
            elif fs and vs: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se; df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


def strat_rsi(df, per=14, os=25, ob=75, cd=60, use_vwap=False):
    c = df["Close"].values; n = len(c)
    r = calc_rsi(df, per)["rsi"]
    vw = calc_vwap(df)["vwap"] if use_vwap else None
    df2 = df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool); se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0; bst=cd
    for i in range(2, n):
        bst += 1
        ls = r[i-1]<=os and r[i]>os
        ss = r[i-1]>=ob and r[i]<ob
        vl = (not use_vwap) or c[i]>vw[i]
        vs = (not use_vwap) or c[i]<vw[i]
        xl = r[i-1]<50 and r[i]>=50
        xs = r[i-1]>50 and r[i]<=50
        if pos==1 and (xl or ss): lx[i]=True; pos=0; bst=0
        elif pos==-1 and (xs or ls): sx[i]=True; pos=0; bst=0
        if pos==0 and bst>=cd:
            if ls and vl: le[i]=True; pos=1; bst=0
            elif ss and vs: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se; df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


def strat_sess(df, cd=100, use_vwap=False):
    c=df["Close"].values; h=df["High"].values; lo_=df["Low"].values; n=len(c)
    hrs=df.index.hour; vw=calc_vwap(df)["vwap"] if use_vwap else None
    ah=np.full(n,np.nan); al=np.full(n,np.nan)
    shi=-np.inf; slo=np.inf; pd_=None
    for i in range(n):
        cd_=df.index[i].date()
        if cd_!=pd_: shi=-np.inf; slo=np.inf; pd_=cd_
        if hrs[i]>=22 or hrs[i]<7: shi=max(shi,h[i]); slo=min(slo,lo_[i])
        if shi>-np.inf: ah[i]=shi; al[i]=slo
    df2=df.copy()
    le=np.zeros(n,bool); lx=np.zeros(n,bool); se=np.zeros(n,bool); sx=np.zeros(n,bool)
    pos=0; bst=cd
    for i in range(2, n):
        bst += 1
        if np.isnan(ah[i]): continue
        hr=hrs[i]
        if not (7<=hr<16): continue
        mid=(ah[i]+al[i])/2
        bu=c[i-1]<=ah[i] and c[i]>ah[i]
        bd=c[i-1]>=al[i] and c[i]<al[i]
        vl=(not use_vwap) or c[i]>vw[i]
        vs=(not use_vwap) or c[i]<vw[i]
        if pos==1 and c[i]<mid: lx[i]=True; pos=0; bst=0
        elif pos==-1 and c[i]>mid: sx[i]=True; pos=0; bst=0
        if pos==0 and bst>=cd:
            if bu and vl: le[i]=True; pos=1; bst=0
            elif bd and vs: se[i]=True; pos=-1; bst=0
    df2["long_entry"]=le; df2["long_exit"]=lx|se; df2["short_entry"]=se; df2["short_exit"]=sx|le
    return df2


def main():
    TE="2026-02-16"; TS="2026-02-17"
    rp = Path(__file__).resolve().parent / "multi_indicator_results.txt"
    out = open(rp, "w", encoding="utf-8")
    def log(m=""):
        print(m); out.write(m+"\n"); out.flush()

    log("STRATEGY SHOWDOWN v2 — 5 architectures × 2 pairs")
    log(f"Commission: 0.0043% | $1000 initial\n")

    for df_file, pair, qty in [("OANDA_EURUSD, 3.csv","EURUSD",1000), ("OANDA_GBPJPY, 3.csv","GBPJPY",100)]:
        df=load_tv_export(df_file)
        sd=str(df.index[0].date()); ed=str(df.index[-1].date())
        log(f"\n{'#'*80}")
        log(f"  {pair} ({len(df):,} bars, {qty} units)")
        log(f"{'#'*80}")

        board = []

        # --- 1. GC Reversal ---
        t0=time()
        best=None
        for p in [70,90,110]:
            for m in [3.0,4.0,6.0]:
                for vk in [(False,False,"base"),(True,True,"+VK")]:
                    ds=strat_gc(df,p,m,120,vk[0],vk[1])
                    for tp in [0.0,0.15]:
                        for sl in [0.0,0.10]:
                            r=ex(run_bt(ds,sd,ed,tp,sl,qty))
                            if r["tr"]>=15 and (not best or r["pf"]>best["pf"]):
                                best={**r,"p":p,"m":m,"f":vk[2],"tp":tp,"sl":sl}
        if best:
            ds=strat_gc(df,best["p"],best["m"],120,"V" in best["f"],"K" in best["f"])
            oos=ex(run_bt(ds,TS,ed,best["tp"],best["sl"],qty))
            log(f"\n  1. GC Reversal  ({time()-t0:.0f}s)")
            log(f"     P={best['p']} M={best['m']} {best['f']} TP={best['tp']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("GC Reversal",best,oos))

        # --- 2. VWAP Bands ---
        t0=time()
        best=None
        for bm in [1.5,2.0,2.5,3.0]:
            for cd in [30,60,100]:
                for rf in [False,True]:
                    ds=strat_vwap(df,bm,cd,rf)
                    for tp in [0.0,0.15]:
                        for sl in [0.0,0.10]:
                            r=ex(run_bt(ds,sd,ed,tp,sl,qty))
                            if r["tr"]>=15 and (not best or r["pf"]>best["pf"]):
                                best={**r,"bm":bm,"cd":cd,"rf":rf,"tp":tp,"sl":sl}
        if best:
            ds=strat_vwap(df,best["bm"],best["cd"],best["rf"])
            oos=ex(run_bt(ds,TS,ed,best["tp"],best["sl"],qty))
            log(f"\n  2. VWAP Bands  ({time()-t0:.0f}s)")
            log(f"     Band={best['bm']}σ CD={best['cd']} RSI={best['rf']} TP={best['tp']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("VWAP Bands",best,oos))

        # --- 3. Supertrend ---
        t0=time()
        best=None
        for p in [7,10,14,20]:
            for m in [2.0,3.0,4.0]:
                for vf in [False,True]:
                    ds=strat_st(df,p,m,10,vf)
                    for sl in [0.0,0.10,0.15]:
                        r=ex(run_bt(ds,sd,ed,0,sl,qty))
                        if r["tr"]>=15 and (not best or r["pf"]>best["pf"]):
                            best={**r,"p":p,"m":m,"vf":vf,"sl":sl}
        if best:
            ds=strat_st(df,best["p"],best["m"],10,best["vf"])
            oos=ex(run_bt(ds,TS,ed,0,best["sl"],qty))
            log(f"\n  3. Supertrend  ({time()-t0:.0f}s)")
            log(f"     P={best['p']} M={best['m']} VWAP={best['vf']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("Supertrend",best,oos))

        # --- 4. RSI Reversal ---
        t0=time()
        best=None
        for p in [7,14,21]:
            for ob in [70,75,80]:
                for cd in [20,60]:
                    for vf in [False,True]:
                        ds=strat_rsi(df,p,100-ob,ob,cd,vf)
                        for sl in [0.0,0.10]:
                            r=ex(run_bt(ds,sd,ed,0,sl,qty))
                            if r["tr"]>=10 and (not best or r["pf"]>best["pf"]):
                                best={**r,"p":p,"ob":ob,"cd":cd,"vf":vf,"sl":sl}
        if best:
            ds=strat_rsi(df,best["p"],100-best["ob"],best["ob"],best["cd"],best["vf"])
            oos=ex(run_bt(ds,TS,ed,0,best["sl"],qty))
            log(f"\n  4. RSI Reversal  ({time()-t0:.0f}s)")
            log(f"     P={best['p']} OB/OS={best['ob']}/{100-best['ob']} CD={best['cd']} VWAP={best['vf']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("RSI Reversal",best,oos))

        # --- 5. Session Breakout ---
        t0=time()
        best=None
        for cd in [30,60,100]:
            for vf in [False,True]:
                ds=strat_sess(df,cd,vf)
                for sl in [0.0,0.10,0.15]:
                    r=ex(run_bt(ds,sd,ed,0,sl,qty))
                    if r["tr"]>=5 and (not best or r["pf"]>best["pf"]):
                        best={**r,"cd":cd,"vf":vf,"sl":sl}
        if best:
            ds=strat_sess(df,best["cd"],best["vf"])
            oos=ex(run_bt(ds,TS,ed,0,best["sl"],qty))
            log(f"\n  5. Session Breakout  ({time()-t0:.0f}s)")
            log(f"     CD={best['cd']} VWAP={best['vf']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("Session BRK",best,oos))

        # Leaderboard
        log(f"\n  {'='*75}")
        log(f"  LEADERBOARD — {pair} (sorted by OOS PF)")
        log(f"  {'='*75}")
        log(f"  {'#':>3} {'Strategy':>16}  {'FullPF':>6} {'FullTr':>6} {'FullDD':>7}  {'OOS_PF':>6} {'OOSTr':>5} {'OOS_DD':>7}")
        log(f"  {'-'*75}")
        for rank, (name, full, oos) in enumerate(sorted(board, key=lambda x: x[2]["pf"], reverse=True), 1):
            fpf = f"{full['pf']:.2f}" if full['pf'] < 100 else "INF"
            opf = f"{oos['pf']:.2f}" if oos['pf'] < 100 else "INF"
            log(f"  {rank:>3} {name:>16}  {fpf:>6} {full['tr']:>6} {full['dd']:>6.1f}%  {opf:>6} {oos['tr']:>5} {oos['dd']:>6.1f}%")

    out.close()
    print(f"\n  Done! -> strategies/multi_indicator_results.txt")


if __name__ == "__main__":
    main()
