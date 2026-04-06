#!/usr/bin/env python3
"""
btc_hf_stack_sweep.py -- BTC HF Phase 2: Stacked Multi-Signal Sweep (Long Only)

Phase 1 found the best individual signals. This phase stacks them together
for higher frequency while maintaining WR/PF. All long-only.

Entry signals (any fires = enter, subject to gate + cooldown):
    R001    N consecutive up bricks (momentum)
    R002    N down bricks then up (reversal)
    ST      Supertrend flips bullish
    MACD    MACD histogram crosses positive
    KAMA    KAMA slope turns positive
    BB_B    BB %B oversold bounce (mean reversion)
    RSI_OV  RSI oversold + up brick

Stacks tested:
    A: R001 + R002                    (brick patterns only)
    B: R001 + R002 + ST               (+ trend flip)
    C: R001 + R002 + MACD             (+ momentum)
    D: R001 + R002 + ST + MACD        (full momentum)
    E: R001 + R002 + KAMA             (+ adaptive)
    F: R001 + R002 + ST + MACD + KAMA (kitchen sink)
    G: ST + MACD + KAMA               (indicators only, no brick count)
    H: R001 + R002 + BB_B             (+ mean reversion)
    I: R001 + R002 + RSI_OV           (+ oversold bounce)
    J: ALL signals combined            (max frequency)

Gate modes: none, psar, adx20, psar_adx20
Cooldowns: [3, 5, 8]
N-bricks: [2, 3]

Usage:
    python renko/btc_hf_stack_sweep.py
"""

import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.config import MAX_WORKERS

# -- Instrument config ---------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20

# -- Grid ----------------------------------------------------------------------

COOLDOWNS  = [3, 5, 8]
GATE_MODES = ["none", "psar", "adx20", "psar_adx20"]
N_BRICKS   = [2, 3]

STACKS = {
    "A_r001_r002":           ["r001", "r002"],
    "B_r001_r002_st":        ["r001", "r002", "st_flip"],
    "C_r001_r002_macd":      ["r001", "r002", "macd_flip"],
    "D_r001_r002_st_macd":   ["r001", "r002", "st_flip", "macd_flip"],
    "E_r001_r002_kama":      ["r001", "r002", "kama_turn"],
    "F_full_momentum":       ["r001", "r002", "st_flip", "macd_flip", "kama_turn"],
    "G_indicators_only":     ["st_flip", "macd_flip", "kama_turn"],
    "H_r001_r002_bb":        ["r001", "r002", "bb_bounce"],
    "I_r001_r002_rsi":       ["r001", "r002", "rsi_oversold"],
    "J_all":                 ["r001", "r002", "st_flip", "macd_flip", "kama_turn", "bb_bounce", "rsi_oversold"],
}


# -- Data loading ---------------------------------------------------------------

def _load_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    return df


# -- Gate computation -----------------------------------------------------------

def _compute_gate(df, gate_mode):
    n = len(df)
    gate = np.ones(n, dtype=bool)
    if "psar" in gate_mode:
        psar = df["psar_dir"].values
        gate &= (np.isnan(psar) | (psar > 0))
    if "adx20" in gate_mode:
        adx = df["adx"].values
        gate &= (np.isnan(adx) | (adx >= 20))
    return gate


# -- Backtest runner ------------------------------------------------------------

def _run_bt(df, entry, exit_, start, end):
    from engine import BacktestConfig, run_backtest
    df2 = df.copy()
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
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# -- Stacked signal generator ---------------------------------------------------

def _gen_stacked(df, cooldown, gate, n_bricks, signals):
    """Generate combined entry signal from multiple signal types."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)

    # Pre-extract indicator arrays
    st_dir = df["st_dir"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    bb_pct = df["bb_pct_b"].values
    rsi = df["rsi"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 35

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit: first down brick
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        # Cooldown + gate
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if not up:
            continue

        # Check each signal type — ANY firing = entry
        fired = False

        if "r001" in signals:
            # N consecutive up bricks including current
            all_up = True
            for j in range(n_bricks):
                if not brick_up[i - j]:
                    all_up = False
                    break
            if all_up:
                fired = True

        if not fired and "r002" in signals:
            # N down bricks then up (current is up)
            all_down = True
            for j in range(1, n_bricks + 1):
                if brick_up[i - j]:
                    all_down = False
                    break
            if all_down:
                fired = True

        if not fired and "st_flip" in signals:
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True

        if not fired and "macd_flip" in signals:
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True

        if not fired and "kama_turn" in signals:
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True

        if not fired and "bb_bounce" in signals:
            if not np.isnan(bb_pct[i]):
                if bb_pct[i] <= 0.25:  # use 0.25 threshold
                    fired = True

        if not fired and "rsi_oversold" in signals:
            if not np.isnan(rsi[i]):
                if rsi[i] < 40:  # use 40 threshold
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# -- Combo builder --------------------------------------------------------------

def _build_combos():
    combos = []
    for stack_name, signals in STACKS.items():
        for cd in COOLDOWNS:
            for gmode in GATE_MODES:
                for nb in N_BRICKS:
                    combos.append({
                        "stack": stack_name,
                        "signals": signals,
                        "cooldown": cd,
                        "gate_mode": gmode,
                        "n_bricks": nb,
                    })
    return combos


# -- Worker ---------------------------------------------------------------------

_w = {}


def _run_one(combo):
    if "df" not in _w:
        _w["df"] = _load_data()
        _w["gates"] = {gm: _compute_gate(_w["df"], gm) for gm in GATE_MODES}

    df = _w["df"]
    gate = _w["gates"][combo["gate_mode"]]

    entry, exit_ = _gen_stacked(
        df, combo["cooldown"], gate, combo["n_bricks"], combo["signals"],
    )

    is_r = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(df, entry, exit_, OOS_START, OOS_END)

    return combo, is_r, oos_r


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    oos_days = 170

    print(f"\n{'='*120}")
    print("  BTC HF Phase 2 — Stacked Multi-Signal Sweep (Long Only)")
    print(f"  Target: 1+ trade/day ({oos_days}+ OOS trades)")
    print(f"{'='*120}")

    # -- Per stack best (by net, T >= 100) --
    print(f"\n  {'Stack':<25} {'Gate':<12} {'cd':>3} {'n':>2} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*118}")

    for sname in STACKS:
        subset = [r for r in all_results
                  if r["stack"] == sname and r["oos_trades"] >= 50 and r["oos_net"] > 0]
        if not subset:
            print(f"  {sname:<25} (no viable results)")
            continue
        best = max(subset, key=lambda r: r["oos_net"])
        pf_i = "INF" if math.isinf(best["is_pf"]) else f"{best['is_pf']:.2f}"
        pf_o = "INF" if math.isinf(best["oos_pf"]) else f"{best['oos_pf']:.2f}"
        tpd = best["oos_trades"] / oos_days
        print(f"  {sname:<25} {best['gate_mode']:<12} {best['cooldown']:>3} {best['n_bricks']:>2} | "
              f"{pf_i:>7} {best['is_trades']:>5} {best['is_wr']:>5.1f}% | "
              f"{pf_o:>8} {best['oos_trades']:>5} {tpd:>4.1f} {best['oos_wr']:>5.1f}% "
              f"{best['oos_net']:>9.2f} {best['oos_dd']:>6.2f}%")

    # -- HF viable (T >= oos_days) sorted by net --
    hf = [r for r in all_results
          if r["oos_trades"] >= oos_days and r["oos_net"] > 0]
    hf.sort(key=lambda r: r["oos_net"], reverse=True)

    print(f"\n{'='*120}")
    print(f"  HIGH FREQ (T>={oos_days}, net>0): {len(hf)} configs")
    print(f"{'='*120}")
    _print_header()
    for i, r in enumerate(hf[:30]):
        _print_row(r, oos_days, i+1)

    # -- Best WR (T >= 150) --
    hw = [r for r in all_results
          if r["oos_trades"] >= 150 and r["oos_net"] > 0]
    hw.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

    print(f"\n{'='*120}")
    print(f"  BEST WR (T>=150, net>0): {len(hw)} configs")
    print(f"{'='*120}")
    _print_header()
    for i, r in enumerate(hw[:30]):
        _print_row(r, oos_days, i+1)

    # -- Best PF (T >= 150) --
    hp = [r for r in all_results
          if r["oos_trades"] >= 150 and r["oos_net"] > 0]
    hp.sort(key=lambda r: (
        r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
        r["oos_net"],
    ), reverse=True)

    print(f"\n{'='*120}")
    print(f"  BEST PF (T>=150, net>0): {len(hp)} configs")
    print(f"{'='*120}")
    _print_header()
    for i, r in enumerate(hp[:30]):
        _print_row(r, oos_days, i+1)

    # -- Balanced: T >= 170, WR >= 55%, sorted by net --
    bal = [r for r in all_results
           if r["oos_trades"] >= oos_days and r["oos_wr"] >= 55.0
           and r["oos_net"] > 0]
    bal.sort(key=lambda r: (r["oos_net"], r["oos_wr"]), reverse=True)

    print(f"\n{'='*120}")
    print(f"  BALANCED (T>={oos_days}, WR>=55%, net>0): {len(bal)} configs")
    print(f"{'='*120}")
    _print_header()
    for i, r in enumerate(bal[:30]):
        _print_row(r, oos_days, i+1)

    # -- Gate effectiveness --
    print(f"\n  --- Gate Effectiveness (avg OOS, trades >= 50) ---")
    for gm in GATE_MODES:
        rows = [r for r in all_results if r["gate_mode"] == gm and r["oos_trades"] >= 50]
        if rows:
            finite_pf = [r["oos_pf"] for r in rows if not math.isinf(r["oos_pf"]) and r["oos_pf"] > 0]
            avg_pf = np.mean(finite_pf) if finite_pf else 0
            avg_t = np.mean([r["oos_trades"] for r in rows])
            avg_wr = np.mean([r["oos_wr"] for r in rows])
            avg_net = np.mean([r["oos_net"] for r in rows])
            print(f"    {gm:<15} avg PF={avg_pf:>7.2f} | avg T={avg_t:>6.1f} | "
                  f"avg WR={avg_wr:>5.1f}% | avg Net={avg_net:>7.2f} | N={len(rows)}")


def _print_header():
    print(f"  {'#':>3} {'Stack':<25} {'Gate':<12} {'cd':>3} {'n':>2} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*120}")


def _print_row(r, oos_days, rank):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / oos_days
    print(f"  {rank:>3} {r['stack']:<25} {r['gate_mode']:<12} {r['cooldown']:>3} {r['n_bricks']:>2} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


# -- Main -----------------------------------------------------------------------

def main():
    combos = _build_combos()
    total = len(combos)

    print(f"\n{'='*70}")
    print(f"BTC HF Phase 2 — Stacked Multi-Signal Sweep (Long Only)")
    print(f"  Stacks     : {len(STACKS)} combos")
    print(f"  Cooldowns  : {COOLDOWNS}")
    print(f"  Gates      : {GATE_MODES}")
    print(f"  N-bricks   : {N_BRICKS}")
    print(f"  Total runs : {total} ({total*2} backtests)")
    print(f"  Workers    : {MAX_WORKERS}")
    print(f"  IS period  : {IS_START} -> {IS_END}")
    print(f"  OOS period : {OOS_START} -> {OOS_END}")
    print(f"{'='*70}\n")

    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "stack":      combo["stack"],
                    "cooldown":   combo["cooldown"],
                    "gate_mode":  combo["gate_mode"],
                    "n_bricks":   combo["n_bricks"],
                    "is_pf":      is_r["pf"],
                    "is_trades":  is_r["trades"],
                    "is_wr":      is_r["wr"],
                    "is_net":     is_r["net"],
                    "is_dd":      is_r["dd"],
                    "oos_pf":     oos_r["pf"],
                    "oos_trades": oos_r["trades"],
                    "oos_wr":     oos_r["wr"],
                    "oos_net":    oos_r["net"],
                    "oos_dd":     oos_r["dd"],
                }
                results.append(row)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

            done += 1
            if done % 40 == 0 or done == total:
                print(f"  [{done:>4}/{total}]", flush=True)

    # Save
    out_path = ROOT / "ai_context" / "btc_hf_stack_results.json"
    out_path.parent.mkdir(exist_ok=True)
    serializable = []
    for r in results:
        sr = dict(r)
        for k in ("is_pf", "oos_pf"):
            if math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(results)} results -> {out_path}")

    _summarize(results)


if __name__ == "__main__":
    main()
