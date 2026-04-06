#!/usr/bin/env python3
"""
btc_psar_mk_sweep.py -- PSAR + Momentum King parameter sweep on BTC $150 Renko

Tests MK parameter variations with PSAR gate. Long only, exit on first down brick.

Usage:
    python renko/btc_psar_mk_sweep.py
"""

import contextlib
import io
import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------------------
LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
OOS_DAYS   = 170

# -- Per-process cache ---------------------------------------------------------
_w = {}


def _init():
    if "df" in _w:
        return
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators

    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)

    _w["df"] = df
    _w["n"] = len(df)
    _w["brick_up"] = df["brick_up"].values
    _w["psar_dir"] = df["psar_dir"].values


def _run_bt(entry, exit_, start, end):
    from engine import BacktestConfig, run_backtest

    df2 = _w["df"].copy()
    df2["long_entry"] = entry
    df2["long_exit"] = exit_
    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
    }


def _run_one(combo):
    _init()
    from indicators.momentum_king import calc_momentum_king

    df = _w["df"]
    n = _w["n"]
    brick_up = _w["brick_up"]
    psar_dir = _w["psar_dir"]

    mk = calc_momentum_king(
        df,
        ema_length=combo["ema"],
        volatility_factor=combo["vol_f"],
        strength_threshold=combo["st"],
        base_neutral=combo["bn"],
    )
    mk_smoothed = mk["smoothed_momentum"]
    mk_nz = mk["neutral_zone_width"]

    # MK bull gate (pre-shifted — use [i-1])
    mk_bull = np.zeros(n, dtype=bool)
    for i in range(1, n):
        mk_bull[i] = mk_smoothed[i - 1] > mk_nz[i - 1]

    # PSAR gate (already pre-shifted in indicators)
    psar_ok = np.isnan(psar_dir) | (psar_dir > 0)

    # Signal gen: long only, exit first down brick
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_entry = -999
    cd = combo["cd"]

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if up and psar_ok[i] and mk_bull[i] and (i - last_entry >= cd):
            entry[i] = True
            in_pos = True
            last_entry = i

    is_r = _run_bt(entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(entry, exit_, OOS_START, OOS_END)

    label = f"ema{combo['ema']}_vf{combo['vol_f']}_st{combo['st']}_bn{combo['bn']}_cd{combo['cd']}"
    return {"label": label, **combo, "is": is_r, "oos": oos_r}


def _build_combos():
    combos = []
    for ema in [7, 10, 14, 21, 28]:
        for vol_f in [0.3, 0.5, 1.0, 1.5, 2.0]:
            for st in [0.3, 0.5, 0.6, 0.8]:
                for bn in [0.01, 0.05, 0.1, 0.5]:
                    for cd in [3, 5, 10]:
                        combos.append({
                            "ema": ema, "vol_f": vol_f, "st": st, "bn": bn, "cd": cd,
                        })
    return combos


if __name__ == "__main__":
    combos = _build_combos()
    print(f"PSAR + MK Sweep: {len(combos)} combos on {MAX_WORKERS} workers")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run_one, c): c for c in combos}
        done = 0
        for f in as_completed(futs):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(combos)}...")
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  ERROR: {e}")

    # Filter: OOS trades >= 50
    viable = [r for r in results if r["oos"]["trades"] >= 50]
    viable.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]), reverse=True)

    print(f"\nTop 20 by OOS WR (trades >= 50):")
    hdr = f"{'Config':<45} {'IS PF':>7} {'IS T':>5} {'IS WR':>6} {'OOS PF':>7} {'OOS T':>6} {'t/d':>5} {'OOS WR':>7} {'Decay':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in viable[:20]:
        tpd = r["oos"]["trades"] / OOS_DAYS
        decay = r["oos"]["wr"] - r["is"]["wr"]
        pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        print(
            f"{r['label']:<45} {r['is']['pf']:>7.1f} {r['is']['trades']:>5} "
            f"{r['is']['wr']:>5.1f}% {pf_s:>7} {r['oos']['trades']:>6} "
            f"{tpd:>5.1f} {r['oos']['wr']:>6.1f}% {decay:>+5.1f}"
        )

    print(f"\nTotal: {len(results)} run, {len(viable)} viable (>=50t OOS)")
    print(f"\nBaseline comparison:")
    print(f"  BTC007v3 TV OOS: PF=27.18, 182t (1.1/d), WR=68.7%")
    print(f"  BTC011   TV OOS: PF=35.64, 134t (0.8/d), WR=73.1%")

    # Save results
    out_path = ROOT / "ai_context" / "btc_psar_mk_results.json"
    with open(out_path, "w") as f:
        json.dump({"combos": len(combos), "viable": len(viable), "top20": viable[:20], "all": results}, f, indent=2)
    print(f"\nSaved to {out_path}")
