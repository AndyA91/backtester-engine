#!/usr/bin/env python3
"""
mym_psar_mk_sweep.py -- PSAR + Momentum King parameter sweep on MYM ETH (Extended Hours)

Tests MK parameter variations with PSAR gate + optional ADX/RelVol gates.
Both brick 14 and 15 ETH. Long + Short, exit on first opposing brick.

Usage:
    python renko/mym_psar_mk_sweep.py
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------------------
BRICKS = {
    14: {
        "file": "CBOT_MINI_MYM1!, 1S renko 14 ETH.csv",
        "is_start": "2025-06-15", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-31",
        "oos_days": 90,
    },
    15: {
        "file": "CBOT_MINI_MYM1!, 1S renko 15 eth.csv",
        "is_start": "2025-04-30", "is_end": "2025-12-31",
        "oos_start": "2026-01-01", "oos_end": "2026-03-31",
        "oos_days": 90,
    },
}
COMMISSION = 0.00475
CAPITAL = 1000.0
QTY_VALUE = 0.50

# -- Per-process cache ---------------------------------------------------------
_w = {}


def _load_eth(filepath):
    """Load ETH Renko CSV preserving extra columns (Volume, RelVol, ADX)."""
    df = pd.read_csv(filepath)
    dt = pd.to_datetime(df["time"], unit="s")
    df["Date"] = dt
    df = df.set_index("Date")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "Relative Volume Ratio": "tv_relvol", "ADX": "tv_adx",
    })
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.iloc[:-1]  # drop last (may be unfinished)
    df["brick_up"] = df["Close"] > df["Open"]
    return df


def _init(brick_size):
    key = f"df_{brick_size}"
    if key in _w:
        return
    from renko.indicators import add_renko_indicators

    cfg = BRICKS[brick_size]
    filepath = ROOT / "data" / "MYM" / cfg["file"]
    df = _load_eth(filepath)
    add_renko_indicators(df)

    # Pre-shift TV columns for [i-1] access
    if "tv_relvol" in df.columns:
        df["tv_relvol_prev"] = df["tv_relvol"].shift(1)
    if "tv_adx" in df.columns:
        df["tv_adx_prev"] = df["tv_adx"].shift(1)

    _w[key] = df
    _w[f"n_{brick_size}"] = len(df)
    _w[f"brick_up_{brick_size}"] = df["brick_up"].values
    _w[f"psar_dir_{brick_size}"] = df["psar_dir"].values
    _w[f"adx_{brick_size}"] = df["adx"].values
    _w[f"tv_adx_{brick_size}"] = df["tv_adx_prev"].values if "tv_adx_prev" in df.columns else np.full(len(df), np.nan)
    _w[f"tv_relvol_{brick_size}"] = df["tv_relvol_prev"].values if "tv_relvol_prev" in df.columns else np.full(len(df), np.nan)

    up = df["brick_up"].sum()
    print(f"  Loaded MYM brick {brick_size} ETH: {len(df)} bars, {up} up / {len(df)-up} down")


def _run_bt(brick_size, entry_l, exit_l, entry_s, exit_s, start, end):
    from engine import BacktestConfig, run_backtest

    df2 = _w[f"df_{brick_size}"].copy()
    df2["long_entry"] = entry_l
    df2["long_exit"] = exit_l
    df2["short_entry"] = entry_s
    df2["short_exit"] = exit_s
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
    brick_size = combo["brick"]
    _init(brick_size)
    from indicators.momentum_king import calc_momentum_king

    df = _w[f"df_{brick_size}"]
    n = _w[f"n_{brick_size}"]
    brick_up = _w[f"brick_up_{brick_size}"]
    psar_dir = _w[f"psar_dir_{brick_size}"]
    adx = _w[f"adx_{brick_size}"]
    tv_adx = _w[f"tv_adx_{brick_size}"]
    tv_relvol = _w[f"tv_relvol_{brick_size}"]

    mk = calc_momentum_king(
        df,
        ema_length=combo["ema"],
        volatility_factor=combo["vol_f"],
        strength_threshold=combo["st"],
        base_neutral=combo["bn"],
    )
    mk_smoothed = mk["smoothed_momentum"]
    mk_nz = mk["neutral_zone_width"]

    # MK bull/bear gate (pre-shifted)
    mk_bull = np.zeros(n, dtype=bool)
    mk_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        mk_bull[i] = mk_smoothed[i - 1] > mk_nz[i - 1]
        mk_bear[i] = mk_smoothed[i - 1] < -mk_nz[i - 1]

    # PSAR gate (already pre-shifted in indicators)
    psar_long_ok = np.isnan(psar_dir) | (psar_dir > 0)
    psar_short_ok = np.isnan(psar_dir) | (psar_dir < 0)

    # ADX gate — use TV-provided ADX if available, else our computed one
    adx_thresh = combo["adx_t"]
    if adx_thresh > 0:
        adx_vals = tv_adx if not np.all(np.isnan(tv_adx)) else adx
        adx_ok = np.where(np.isnan(adx_vals), False, adx_vals >= adx_thresh)
    else:
        adx_ok = np.ones(n, dtype=bool)

    # RelVol gate — filter out low-volume bricks
    rv_thresh = combo.get("rv_t", 0)
    if rv_thresh > 0:
        rv_ok = np.where(np.isnan(tv_relvol), False, tv_relvol >= rv_thresh)
    else:
        rv_ok = np.ones(n, dtype=bool)

    # Signal gen
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    pos_dir_arr = np.zeros(n, dtype=np.int8)
    in_pos = False
    pos_dir = 0
    last_entry = -999
    cd = combo["cd"]

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if (pos_dir == 1 and not up) or (pos_dir == -1 and up):
                exit_[i] = True
                in_pos = False
            continue
        if i - last_entry < cd:
            continue
        if up and psar_long_ok[i] and mk_bull[i] and adx_ok[i] and rv_ok[i]:
            entry[i] = True
            pos_dir_arr[i] = 1
            in_pos = True
            pos_dir = 1
            last_entry = i
        elif not up and psar_short_ok[i] and mk_bear[i] and adx_ok[i] and rv_ok[i]:
            entry[i] = True
            pos_dir_arr[i] = -1
            in_pos = True
            pos_dir = -1
            last_entry = i

    long_entry = entry & (pos_dir_arr == 1)
    long_exit = exit_ & ~brick_up
    short_entry = entry & (pos_dir_arr == -1)
    short_exit = exit_ & brick_up

    bcfg = BRICKS[brick_size]
    is_r = _run_bt(brick_size, long_entry, long_exit, short_entry, short_exit, bcfg["is_start"], bcfg["is_end"])
    oos_r = _run_bt(brick_size, long_entry, long_exit, short_entry, short_exit, bcfg["oos_start"], bcfg["oos_end"])

    label = f"b{brick_size}_ema{combo['ema']}_vf{combo['vol_f']}_st{combo['st']}_bn{combo['bn']}_cd{combo['cd']}_adx{combo['adx_t']}_rv{combo.get('rv_t', 0)}"
    return {"label": label, **combo, "is": is_r, "oos": oos_r}


def _build_combos():
    combos = []
    for brick in [14, 15]:
        for ema in [7, 10, 14, 21]:
            for vol_f in [0.5, 1.0, 1.5, 2.0]:
                for st in [0.3, 0.6]:
                    for bn in [0.05, 0.5]:
                        for cd in [3, 5, 10, 20, 30, 45]:
                            for adx_t in [0, 25, 30, 40, 50]:
                                for rv_t in [0, 0.5, 1.0]:
                                    combos.append({
                                        "brick": brick, "ema": ema,
                                        "vol_f": vol_f, "st": st,
                                        "bn": bn, "cd": cd,
                                        "adx_t": adx_t, "rv_t": rv_t,
                                    })
    return combos


if __name__ == "__main__":
    combos = _build_combos()
    print(f"PSAR + MK Sweep (MYM ETH b14+b15): {len(combos)} combos on {MAX_WORKERS} workers")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_run_one, c): c for c in combos}
        done = 0
        for f in as_completed(futs):
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(combos)}...", flush=True)
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

    # Filter: OOS trades >= 10
    viable = [r for r in results if r["oos"]["trades"] >= 10]
    viable.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]), reverse=True)

    oos_days_map = {14: 90, 15: 90}

    print(f"\nTop 40 by OOS WR (trades >= 10):")
    hdr = f"{'Config':<65} {'IS PF':>7} {'IS T':>5} {'IS WR':>6} {'OOS PF':>7} {'OOS T':>6} {'t/d':>5} {'OOS WR':>7} {'Decay':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in viable[:40]:
        tpd = r["oos"]["trades"] / oos_days_map[r["brick"]]
        decay = r["oos"]["wr"] - r["is"]["wr"]
        pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        is_pf = "inf" if r["is"]["pf"] == float("inf") else f"{r['is']['pf']:.1f}"
        print(
            f"{r['label']:<65} {is_pf:>7} {r['is']['trades']:>5} "
            f"{r['is']['wr']:>5.1f}% {pf_s:>7} {r['oos']['trades']:>6} "
            f"{tpd:>5.1f} {r['oos']['wr']:>6.1f}% {decay:>+5.1f}"
        )

    # Top by PF
    viable_pf = sorted(viable, key=lambda r: r["oos"]["pf"], reverse=True)
    print(f"\nTop 20 by OOS PF (trades >= 10):")
    print(hdr)
    print("-" * len(hdr))
    for r in viable_pf[:20]:
        tpd = r["oos"]["trades"] / oos_days_map[r["brick"]]
        decay = r["oos"]["wr"] - r["is"]["wr"]
        pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        is_pf = "inf" if r["is"]["pf"] == float("inf") else f"{r['is']['pf']:.1f}"
        print(
            f"{r['label']:<65} {is_pf:>7} {r['is']['trades']:>5} "
            f"{r['is']['wr']:>5.1f}% {pf_s:>7} {r['oos']['trades']:>6} "
            f"{tpd:>5.1f} {r['oos']['wr']:>6.1f}% {decay:>+5.1f}"
        )

    # Per-brick summary
    for brick in [14, 15]:
        bv = [r for r in viable if r["brick"] == brick]
        if bv:
            best = max(bv, key=lambda r: r["oos"]["wr"])
            print(f"\nBrick {brick} best OOS WR: {best['label']} -> {best['oos']['wr']:.1f}% WR, PF={best['oos']['pf']:.1f}, {best['oos']['trades']}t")

    print(f"\nTotal: {len(results)} run, {len(viable)} viable (>=10t OOS)")
    print(f"Baseline: MYM001 RTH OOS: PF=104.95, 43t (0.6/d), WR=86.0%")

    # Save
    out_path = ROOT / "ai_context" / "mym_psar_mk_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "combos": len(combos), "viable": len(viable),
            "top40": viable[:40], "all": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")
