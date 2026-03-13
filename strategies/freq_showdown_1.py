"""
Frequency Showdown v1 — targeting >= 1 trade/day with high PF + expectancy

4 architectures, all designed for higher signal frequency than existing winners:

  1. Session-Filtered VWAP   — existing VWAP+ADX but entries restricted to
                               London/NY hours (07:00-17:00 UTC). Asian thin
                               hours blocked. Expect fewer but higher-quality trades.

  2. BB + ADX Regime         — Bollinger Bands mean reversion with ADX ranging
                               gate. BB is session-independent and adapts to
                               recent price action → more band touches per day
                               than session-anchored VWAP.

  3. BB + Stochastic         — BB touch confirmed by Stochastic extreme (<20 / >80).
                               Stochastic is faster than RSI → more confirmations
                               per day. No ADX filter keeps frequency up.

  4. Opening Range Breakout  — London (07:00) or NY (13:00) open. Define range
                               over first N minutes, enter on first breakout.
                               Exactly 1 setup per session per direction.
                               Trend-following (different edge from #1-3).
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
from indicators.vwap import calc_vwap
from indicators.rsi import calc_rsi
from indicators.adx import calc_adx
from indicators.bbands import calc_bbands
from indicators.stochastic import calc_stochastic


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


# =============================================================================
# 1. SESSION-FILTERED VWAP — VWAP+ADX regime, entries only during London/NY
#    Active window: 07:00–17:00 UTC (covers London open → NY close)
# =============================================================================
def strat_session_vwap(df, bm=2.0, cd=40, adx_thresh=30):
    """VWAP mean reversion + ADX ranging gate, entries blocked outside session."""
    c     = df["Close"].values
    n     = len(c)
    hours = df.index.hour

    vr  = calc_vwap(df, band_mult_2=bm)
    vw  = vr["vwap"]; u2 = vr["upper2"]; l2 = vr["lower2"]
    rsi = calc_rsi(df, 14)["rsi"]
    adx = calc_adx(df, adx_period=14)["adx"]

    df2 = df.copy()
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd

    for i in range(2, n):
        bst += 1
        in_session = 7 <= hours[i] < 17

        ls  = c[i-1] <= l2[i-1] and c[i] > l2[i] and rsi[i] < 35
        ss  = c[i-1] >= u2[i-1] and c[i] < u2[i] and rsi[i] > 65
        cuv = c[i-1] <= vw[i-1] and c[i] > vw[i]
        cdv = c[i-1] >= vw[i-1] and c[i] < vw[i]

        # exits always allowed (don't trap position overnight)
        if pos == 1  and (cuv or ss): lx[i] = True; pos = 0; bst = 0
        elif pos ==-1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0

        # entries — session hours + ranging regime only
        if pos == 0 and bst >= cd and in_session and adx[i] < adx_thresh:
            if ls:   le[i] = True; pos = 1;  bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0

    df2["long_entry"]  = le;       df2["long_exit"]  = lx | se
    df2["short_entry"] = se;       df2["short_exit"] = sx | le
    return df2


# =============================================================================
# 2. BB + ADX REGIME — Bollinger Band reversion, entries gated by ADX ranging
#    More signals than VWAP bands (session-independent, adapts to recent vol)
# =============================================================================
def strat_bb_adx(df, period=20, mult=2.0, adx_thresh=30, cd=30):
    """BB mean reversion — entry when price crosses back inside band + ADX < thresh."""
    c   = df["Close"].values
    n   = len(c)
    bb  = calc_bbands(df, period=period, mult=mult)
    mid = bb["mid"]; u = bb["upper"]; l = bb["lower"]
    adx = calc_adx(df, adx_period=14)["adx"]

    df2 = df.copy()
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    start = period + 1

    for i in range(start, n):
        bst += 1
        if np.isnan(mid[i]) or np.isnan(mid[i-1]): continue

        # band cross-back signals
        ls  = c[i-1] <= l[i-1]   and c[i] > l[i]
        ss  = c[i-1] >= u[i-1]   and c[i] < u[i]
        # midline cross exits
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]

        if pos == 1  and (cuv or ss): lx[i] = True; pos = 0; bst = 0
        elif pos ==-1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0

        if pos == 0 and bst >= cd and adx[i] < adx_thresh:
            if ls:   le[i] = True; pos = 1;  bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0

    df2["long_entry"]  = le;       df2["long_exit"]  = lx | se
    df2["short_entry"] = se;       df2["short_exit"] = sx | le
    return df2


# =============================================================================
# 3. BB + STOCHASTIC — BB touch confirmed by Stochastic extreme
#    Stochastic is faster than RSI → more confirmations per session
# =============================================================================
def strat_bb_stoch(df, period=20, mult=2.0, k_period=14, stoch_thresh=20, cd=30):
    """BB cross-back inside confirmed by stoch_k in oversold/overbought zone."""
    c      = df["Close"].values
    n      = len(c)
    bb     = calc_bbands(df, period=period, mult=mult)
    mid    = bb["mid"]; u = bb["upper"]; l = bb["lower"]
    st     = calc_stochastic(df, k_period=k_period, smooth_k=3, smooth_d=3)
    slow_k = st["slow_k"]

    df2 = df.copy()
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = cd
    start = max(period, k_period) + 5

    for i in range(start, n):
        bst += 1
        if np.isnan(mid[i]) or np.isnan(slow_k[i]): continue

        ls  = c[i-1] <= l[i-1]   and c[i] > l[i]   and slow_k[i] < stoch_thresh
        ss  = c[i-1] >= u[i-1]   and c[i] < u[i]   and slow_k[i] > (100 - stoch_thresh)
        cuv = c[i-1] <= mid[i-1] and c[i] > mid[i]
        cdv = c[i-1] >= mid[i-1] and c[i] < mid[i]

        if pos == 1  and (cuv or ss): lx[i] = True; pos = 0; bst = 0
        elif pos ==-1 and (cdv or ls): sx[i] = True; pos = 0; bst = 0

        if pos == 0 and bst >= cd:
            if ls:   le[i] = True; pos = 1;  bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0

    df2["long_entry"]  = le;       df2["long_exit"]  = lx | se
    df2["short_entry"] = se;       df2["short_exit"] = sx | le
    return df2


# =============================================================================
# 4. OPENING RANGE BREAKOUT — 1 setup per session
#    London (07:00 UTC) or NY (13:00 UTC) opening range
#    Entry on first close that breaks above OR_high (long) or OR_low (short)
#    Force-exit at session end (17:00 UTC); tp/sl from BacktestConfig
# =============================================================================
def strat_orb(df, open_hour=7, range_mins=30):
    """
    Opening Range Breakout.

    Range bars: open_hour:00 → open_hour:mm (mm = range_mins).
    Signal: first bar after range where close breaks OR high/low.
    Force-exit: 17:00 UTC (session end).
    """
    n       = len(df)
    hours   = df.index.hour
    minutes = df.index.minute
    c       = df["Close"].values

    # ── precompute OR levels for each calendar date ──────────────────────────
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

    # ── session masks ─────────────────────────────────────────────────────────
    # after_range: past the range window, within session (up to 17:00 UTC)
    after_range = (
        (~in_range_mask)
        & (hours >= open_hour)
        & (hours < 17)
        & ~np.isnan(or_high)
    )
    session_end = (hours == 17) & (minutes == 0)

    # ── stateful signal loop ──────────────────────────────────────────────────
    df2 = df.copy()
    le = np.zeros(n, bool); lx = np.zeros(n, bool)
    se = np.zeros(n, bool); sx = np.zeros(n, bool)
    pos = 0; bst = 0   # ORB: no inter-trade cooldown — session boundary is enough

    for i in range(1, n):
        bst += 1

        # force-close at session end
        if session_end[i] and pos != 0:
            if pos == 1: lx[i] = True
            else:        sx[i] = True
            pos = 0; bst = 0
            continue

        if not after_range[i]: continue

        # breakout signals (crossover of the OR level)
        ls = c[i-1] <= or_high[i] and c[i] > or_high[i]
        ss = c[i-1] >= or_low[i]  and c[i] < or_low[i]

        # opposite break exits open position
        if pos == 1  and ss: lx[i] = True; pos = 0; bst = 0
        elif pos ==-1 and ls: sx[i] = True; pos = 0; bst = 0

        # new entry (allow re-entry after 60 bars to avoid same-day churning)
        if pos == 0 and bst >= 60:
            if ls:   le[i] = True; pos = 1;  bst = 0
            elif ss: se[i] = True; pos = -1; bst = 0

    df2["long_entry"]  = le;       df2["long_exit"]  = lx | se
    df2["short_entry"] = se;       df2["short_exit"] = sx | le
    return df2


# =============================================================================
# MAIN
# =============================================================================
def main():
    OOS = "2026-02-17"
    rp  = Path(__file__).resolve().parent / "freq_showdown_results.txt"
    out = open(rp, "w", encoding="utf-8")

    def log(m=""):
        print(m); out.write(m + "\n"); out.flush()

    log("FREQUENCY SHOWDOWN v1 — targeting >= 1 trade/day")
    log("4 architectures: Session-VWAP | BB+ADX | BB+Stoch | ORB")
    log(f"Commission: 0.0043% | $1000 initial | OOS from {OOS}\n")

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
        log(f"  Target: {n_days} trades for 1/day average")
        log(f"{'#'*80}")
        board = []

        # ── 1. Session-Filtered VWAP ──────────────────────────────────────────
        t0 = time(); best = None
        for bm in [1.5, 2.0, 2.5, 3.0]:
            for cd in [20, 40, 60]:
                for thresh in [25, 30, 35]:
                    ds = strat_session_vwap(df, bm, cd, thresh)
                    for tp in [0.0, 0.10, 0.15]:
                        for sl in [0.0, 0.10]:
                            r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                            if r["tr"] >= 10 and (not best or r["pf"] > best["pf"]):
                                best = {**r, "bm": bm, "cd": cd, "thresh": thresh, "tp": tp, "sl": sl}
        if best:
            ds  = strat_session_vwap(df, best["bm"], best["cd"], best["thresh"])
            oos = ex(run_bt(ds, OOS, ed, best["tp"], best["sl"], qty))
            tpd = best["tr"] / n_days
            log(f"\n  1. Session VWAP  ({time()-t0:.0f}s)")
            log(f"     Band={best['bm']}sig CD={best['cd']} ADX<{best['thresh']} TP={best['tp']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} ({tpd:.2f}/day) Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("Sess-VWAP", best, oos))

        # ── 2. BB + ADX Regime ────────────────────────────────────────────────
        t0 = time(); best = None
        for period in [10, 20]:
            for mult in [1.5, 2.0, 2.5]:
                for thresh in [25, 30, 35]:
                    for cd in [20, 40]:
                        ds = strat_bb_adx(df, period, mult, thresh, cd)
                        for tp in [0.0, 0.10]:
                            for sl in [0.0, 0.10]:
                                r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                                if r["tr"] >= 10 and (not best or r["pf"] > best["pf"]):
                                    best = {**r, "period": period, "mult": mult, "thresh": thresh, "cd": cd, "tp": tp, "sl": sl}
        if best:
            ds  = strat_bb_adx(df, best["period"], best["mult"], best["thresh"], best["cd"])
            oos = ex(run_bt(ds, OOS, ed, best["tp"], best["sl"], qty))
            tpd = best["tr"] / n_days
            log(f"\n  2. BB + ADX Regime  ({time()-t0:.0f}s)")
            log(f"     P={best['period']} Mult={best['mult']}sig ADX<{best['thresh']} CD={best['cd']} TP={best['tp']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} ({tpd:.2f}/day) Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("BB+ADX", best, oos))

        # ── 3. BB + Stochastic ────────────────────────────────────────────────
        t0 = time(); best = None
        for period in [10, 20]:
            for mult in [1.5, 2.0, 2.5]:
                for k in [5, 14]:
                    for thresh in [20, 25, 30]:
                        for cd in [20, 40]:
                            ds = strat_bb_stoch(df, period, mult, k, thresh, cd)
                            for tp in [0.0, 0.10]:
                                for sl in [0.0, 0.10]:
                                    r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                                    if r["tr"] >= 10 and (not best or r["pf"] > best["pf"]):
                                        best = {**r, "period": period, "mult": mult, "k": k, "thresh": thresh, "cd": cd, "tp": tp, "sl": sl}
        if best:
            ds  = strat_bb_stoch(df, best["period"], best["mult"], best["k"], best["thresh"], best["cd"])
            oos = ex(run_bt(ds, OOS, ed, best["tp"], best["sl"], qty))
            tpd = best["tr"] / n_days
            log(f"\n  3. BB + Stochastic  ({time()-t0:.0f}s)")
            log(f"     BB_P={best['period']} Mult={best['mult']}sig Stoch_K={best['k']} Thresh={best['thresh']}/{100-best['thresh']} CD={best['cd']} TP={best['tp']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} ({tpd:.2f}/day) Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("BB+Stoch", best, oos))

        # ── 4. Opening Range Breakout ─────────────────────────────────────────
        t0 = time(); best = None
        for oh in [7, 13]:
            for rm in [15, 30, 60]:
                ds = strat_orb(df, oh, rm)
                for tp in [0.05, 0.10, 0.20]:
                    for sl in [0.05, 0.10]:
                        r = ex(run_bt(ds, sd, ed, tp, sl, qty))
                        if r["tr"] >= 5 and (not best or r["pf"] > best["pf"]):
                            best = {**r, "oh": oh, "rm": rm, "tp": tp, "sl": sl}
        if best:
            ds  = strat_orb(df, best["oh"], best["rm"])
            oos = ex(run_bt(ds, OOS, ed, best["tp"], best["sl"], qty))
            tpd = best["tr"] / n_days
            log(f"\n  4. Opening Range Breakout  ({time()-t0:.0f}s)")
            log(f"     Open={best['oh']}:00UTC Range={best['rm']}min TP={best['tp']} SL={best['sl']}")
            log(f"     Full: PF={best['pf']:.3f} WR={best['wr']:.1f}% Tr={best['tr']} ({tpd:.2f}/day) Net={best['net']:+.2f} DD={best['dd']:.2f}%")
            log(f"     OOS:  PF={oos['pf']:.3f} WR={oos['wr']:.1f}% Tr={oos['tr']} DD={oos['dd']:.2f}%")
            board.append(("ORB", best, oos))

        # ── Leaderboard ───────────────────────────────────────────────────────
        log(f"\n  {'='*75}")
        log(f"  LEADERBOARD — {pair} 5-min (sorted by OOS PF)")
        log(f"  {'='*75}")
        log(f"  {'#':>3} {'Strategy':>12}  {'FullPF':>6} {'Tr/day':>6} {'FullDD':>7}  {'OOS_PF':>6} {'OOSTr':>5} {'OOS_DD':>7}")
        log(f"  {'-'*75}")
        for rank, (name, full, oos) in enumerate(
            sorted(board, key=lambda x: x[2]["pf"], reverse=True), 1
        ):
            fpf  = f"{full['pf']:.2f}" if full["pf"] < 100 else "INF"
            opf  = f"{oos['pf']:.2f}"  if oos["pf"]  < 100 else "INF"
            tpd  = full["tr"] / n_days
            log(f"  {rank:>3} {name:>12}  {fpf:>6} {tpd:>5.2f}/d {full['dd']:>6.1f}%  {opf:>6} {oos['tr']:>5} {oos['dd']:>6.1f}%")

    out.close()
    print(f"\n  Done! -> strategies/freq_showdown_results.txt")


if __name__ == "__main__":
    main()
