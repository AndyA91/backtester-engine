#!/usr/bin/env python3
"""
btc_phase5_hybrid_sweep.py -- BTC Phase 5: Extended Hybrid Entry Sweep (Long Only)

Builds on Phase 4 winner (R007 + FLIP_supertrend) by adding NEW entry triggers
not previously tested, AND varying the HTF ADX threshold to address trade frequency.

New entry triggers (Phase 4 already tested: escgo/stoch/macd_hist/kama_slope cross,
psar/supertrend/ema_cross flip, pullback depth 1-3):
    rsi_50       RSI crosses above 50 from below (momentum regime change)
    stoch_20     Stoch %K crosses above 20 from below (oversold bounce)
    bb_break     Close > upper BB on an up brick (breakout)
    squeeze_fire Squeeze releases (sq_on -> not sq_on) with positive momentum
    cmf_cross    CMF crosses above 0 (money inflow)
    obv_cross    OBV crosses above OBV_EMA (accumulation confirmed)
    cci_cross    CCI crosses above 0 (momentum flip)
    ichi_break   Ichimoku position flips to +1 (above cloud)
    double_rev   2+ down bricks then 2+ up bricks (extended reversal W-pattern)

Sweep structure:
    Block A: R007+FLIP baseline @ HTF [0, 30, 35, 40]
    Block B: R007+FLIP+[each new entry] @ HTF [0, 30, 35, 40]
    Block C: R007+FLIP+[top 2-3 new entries stacked] @ HTF [0, 30, 35, 40]

Uses ProcessPoolExecutor -- one worker per HTF threshold.

Usage:
    python renko/btc_phase5_hybrid_sweep.py
    python renko/btc_phase5_hybrid_sweep.py --no-parallel
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from renko.config import MAX_WORKERS
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# -- Instrument config ---------------------------------------------------------

LTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 150.csv"
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
VOL_MAX    = 1.5
ADX_THRESH = 30

# -- Param grid ----------------------------------------------------------------

PARAM_GRID = {
    "n_bricks": [2, 3, 5],
    "cooldown": [20, 30],
}

HTF_ADX_THRESHOLDS = [0, 30, 35, 40]

NEW_ENTRIES = [
    "rsi_50", "stoch_20", "bb_break", "squeeze_fire",
    "cmf_cross", "obv_cross", "cci_cross", "ichi_break", "double_rev",
]


# -- Data loading --------------------------------------------------------------

def _load_ltf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)
    return df


def _load_htf_data():
    sys.path.insert(0, str(ROOT))
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


# -- Gate computation ----------------------------------------------------------

def _compute_gates(df_ltf, df_htf, htf_adx_thresh):
    """Compute combined long gate: PSAR + ADX>=30 + vol<=1.5 + optional HTF ADX."""
    sys.path.insert(0, str(ROOT))
    from renko.phase6_sweep import _compute_gate_arrays

    n = len(df_ltf)
    gate = np.ones(n, dtype=bool)

    # PSAR direction
    p6_long, _ = _compute_gate_arrays(df_ltf, "psar_dir")
    gate &= p6_long

    # LTF ADX
    adx = df_ltf["adx"].values
    adx_nan = np.isnan(adx)
    gate &= (adx_nan | (adx >= ADX_THRESH))

    # Vol ratio
    vr = df_ltf["vol_ratio"].values
    vr_nan = np.isnan(vr)
    gate &= (vr_nan | (vr <= VOL_MAX))

    # HTF ADX (skip if threshold == 0)
    if htf_adx_thresh > 0:
        htf_adx = df_htf["adx"].values
        htf_adx_nan = np.isnan(htf_adx)
        htf_gate = htf_adx_nan | (htf_adx >= htf_adx_thresh)

        htf_frame = pd.DataFrame({
            "t": df_htf.index.values,
            "g": htf_gate.astype(float),
        }).sort_values("t")
        ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
        merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
        g = merged["g"].values
        htf_aligned = np.where(np.isnan(g), True, g > 0.5).astype(bool)
        gate &= htf_aligned

    return gate


# -- New entry trigger arrays --------------------------------------------------

def _compute_new_entry(df, trigger_name):
    """Return bool array: True on bars where trigger fires (long direction)."""
    n = len(df)
    sig = np.zeros(n, dtype=bool)

    if trigger_name == "rsi_50":
        rsi = df["rsi"].values
        for i in range(1, n):
            if not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]):
                sig[i] = rsi[i] > 50 and rsi[i-1] <= 50

    elif trigger_name == "stoch_20":
        k = df["stoch_k"].values
        for i in range(1, n):
            if not np.isnan(k[i]) and not np.isnan(k[i-1]):
                sig[i] = k[i] > 20 and k[i-1] <= 20

    elif trigger_name == "bb_break":
        close = df["Close"].values.astype(float)
        bb_u = df["bb_upper"].values
        for i in range(1, n):
            if not np.isnan(bb_u[i]) and not np.isnan(bb_u[i-1]):
                sig[i] = close[i] > bb_u[i] and close[i-1] <= bb_u[i-1]

    elif trigger_name == "squeeze_fire":
        sq_on = df["sq_on"].values
        sq_mom = df["sq_momentum"].values
        for i in range(1, n):
            if (not np.isnan(sq_on[i]) and not np.isnan(sq_on[i-1])
                    and not np.isnan(sq_mom[i])):
                sig[i] = (not sq_on[i]) and sq_on[i-1] and sq_mom[i] > 0

    elif trigger_name == "cmf_cross":
        cmf = df["cmf"].values
        for i in range(1, n):
            if not np.isnan(cmf[i]) and not np.isnan(cmf[i-1]):
                sig[i] = cmf[i] > 0 and cmf[i-1] <= 0

    elif trigger_name == "obv_cross":
        obv = df["obv"].values
        obv_ema = df["obv_ema"].values
        for i in range(1, n):
            if (not np.isnan(obv[i]) and not np.isnan(obv_ema[i])
                    and not np.isnan(obv[i-1]) and not np.isnan(obv_ema[i-1])):
                sig[i] = obv[i] > obv_ema[i] and obv[i-1] <= obv_ema[i-1]

    elif trigger_name == "cci_cross":
        cci = df["cci"].values
        for i in range(1, n):
            if not np.isnan(cci[i]) and not np.isnan(cci[i-1]):
                sig[i] = cci[i] > 0 and cci[i-1] <= 0

    elif trigger_name == "ichi_break":
        ichi = df["ichi_pos"].values
        for i in range(1, n):
            if not np.isnan(ichi[i]) and not np.isnan(ichi[i-1]):
                sig[i] = ichi[i] > 0 and ichi[i-1] <= 0

    elif trigger_name == "double_rev":
        # 2+ down bricks followed by 2+ up bricks (current bar is 2nd up)
        brick_up = df["brick_up"].values
        for i in range(3, n):
            if (brick_up[i] and brick_up[i-1]
                    and not brick_up[i-2] and not brick_up[i-3]):
                sig[i] = True

    return sig


# -- Supertrend flip array (from Phase 4) ------------------------------------

def _compute_st_flip(df):
    """Supertrend flips bullish."""
    n = len(df)
    st = df["st_dir"].values
    flip = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(st[i]) and not np.isnan(st[i-1]):
            flip[i] = st[i] > 0 and st[i-1] <= 0
    return flip


# -- Combined signal generator ------------------------------------------------

def _gen_combined(df, n_bricks, cooldown, gate, st_flip, extra_triggers):
    """R007 + FLIP_supertrend + optional extra triggers. Long only."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)

    in_pos = False
    last_r001 = -999_999
    last_flip = -999_999
    last_extra = -999_999
    warmup = max(n_bricks + 1, 200)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
        if in_pos:
            continue
        if not gate[i]:
            continue

        triggered = False

        # R002: reversal (no cooldown)
        prev = brick_up[i - n_bricks : i]
        prev_all_down = bool(not np.any(prev))
        if prev_all_down and up:
            triggered = True

        # R001: momentum (cooldown)
        if not triggered and (i - last_r001) >= cooldown:
            window = brick_up[i - n_bricks + 1 : i + 1]
            if bool(np.all(window)):
                triggered = True
                last_r001 = i

        # FLIP: supertrend flip on up brick (own cooldown)
        if not triggered and up and st_flip[i] and (i - last_flip) >= cooldown:
            triggered = True
            last_flip = i

        # Extra triggers: any fires on an up brick (shared cooldown)
        if not triggered and up and (i - last_extra) >= cooldown:
            for trig_arr in extra_triggers:
                if trig_arr[i]:
                    triggered = True
                    last_extra = i
                    break

        if triggered:
            entry[i] = True
            in_pos = True

    return entry, exit_


# -- Backtest runner -----------------------------------------------------------

def _run_backtest(df, entry, exit_, start, end):
    sys.path.insert(0, str(ROOT))
    from engine import BacktestConfig, run_backtest

    df_sig = df.copy()
    df_sig["long_entry"] = entry
    df_sig["long_exit"] = exit_

    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION, slippage_ticks=0,
        qty_type="cash", qty_value=QTY_VALUE, pyramiding=1,
        start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest(df_sig, cfg)

    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# -- Worker: sweep one HTF threshold ------------------------------------------

def _sweep_htf_threshold(htf_thresh):
    """Run all entry configs for a single HTF ADX threshold."""
    label = f"HTF{htf_thresh}"
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()
    gate = _compute_gates(df_ltf, df_htf, htf_thresh)

    # Precompute triggers
    st_flip = _compute_st_flip(df_ltf)
    new_arrays = {name: _compute_new_entry(df_ltf, name) for name in NEW_ENTRIES}

    keys = list(PARAM_GRID.keys())
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]

    results = []
    total_configs = []

    # Block A: R007+FLIP baseline
    for pc in param_combos:
        total_configs.append(("r007+flip_st", pc, []))

    # Block B: R007+FLIP+[each new entry]
    for ename in NEW_ENTRIES:
        for pc in param_combos:
            total_configs.append((f"r007+flip_st+{ename}", pc, [ename]))

    # Block C: R007+FLIP+[2 or 3 stacked new entries]
    # Test promising 2-entry combos (diversity: momentum + volume + pattern)
    STACK_2 = [
        ("rsi_50", "squeeze_fire"),
        ("rsi_50", "obv_cross"),
        ("stoch_20", "cmf_cross"),
        ("stoch_20", "squeeze_fire"),
        ("cci_cross", "obv_cross"),
        ("bb_break", "cmf_cross"),
        ("ichi_break", "squeeze_fire"),
        ("double_rev", "rsi_50"),
        ("double_rev", "cmf_cross"),
    ]
    for e1, e2 in STACK_2:
        for pc in param_combos:
            total_configs.append((f"r007+flip_st+{e1}+{e2}", pc, [e1, e2]))

    # Test 3-entry stacks
    STACK_3 = [
        ("rsi_50", "squeeze_fire", "obv_cross"),
        ("stoch_20", "cmf_cross", "squeeze_fire"),
        ("rsi_50", "cmf_cross", "double_rev"),
        ("cci_cross", "squeeze_fire", "obv_cross"),
        ("bb_break", "rsi_50", "cmf_cross"),
    ]
    for e1, e2, e3 in STACK_3:
        for pc in param_combos:
            total_configs.append((f"r007+flip_st+{e1}+{e2}+{e3}", pc, [e1, e2, e3]))

    total = len(total_configs)
    print(f"  [{label}] {total} configs to run | gate pass rate: "
          f"{gate.sum()}/{len(gate)} ({gate.sum()/len(gate)*100:.1f}%)", flush=True)

    done = 0
    for config_name, pc, extra_names in total_configs:
        extra_arrs = [new_arrays[e] for e in extra_names]

        e, x = _gen_combined(
            df_ltf, pc["n_bricks"], pc["cooldown"], gate,
            st_flip, extra_arrs,
        )

        is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
        oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

        is_pf = is_r["pf"]
        oos_pf = oos_r["pf"]
        decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

        results.append({
            "config":     config_name,
            "htf_thresh": htf_thresh,
            "n_bricks":   pc["n_bricks"],
            "cooldown":   pc["cooldown"],
            "is_pf":      is_pf,
            "is_trades":  is_r["trades"],
            "is_wr":      is_r["wr"],
            "oos_pf":     oos_pf,
            "oos_trades": oos_r["trades"],
            "oos_wr":     oos_r["wr"],
            "oos_net":    oos_r["net"],
            "decay_pct":  decay,
        })

        done += 1
        if done % 60 == 0 or done == total:
            print(f"  [{label}] {done:>4}/{total} | {config_name:<45} "
                  f"n={pc['n_bricks']} cd={pc['cooldown']:>2} | "
                  f"OOS PF={oos_pf:>7.2f} T={oos_r['trades']:>4}", flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    print(f"\n{'='*95}")
    print("  BTC Phase 5 -- Extended Hybrid Entry Sweep (Long Only)")
    print(f"{'='*95}")

    # -- Per HTF threshold: baseline vs best additive entry --
    for htf in HTF_ADX_THRESHOLDS:
        subset = [r for r in all_results if r["htf_thresh"] == htf]
        if not subset:
            continue

        baselines = [r for r in subset if r["config"] == "r007+flip_st"]
        additive = [r for r in subset if r["config"] != "r007+flip_st"]

        # Best baseline
        bl_viable = [r for r in baselines if r["oos_trades"] >= 10]
        bl_best = max(bl_viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6) if bl_viable else None

        # Best additive
        add_viable = [r for r in additive if r["oos_trades"] >= 10]
        add_best = max(add_viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6) if add_viable else None

        # Most trades with PF >= 5
        high_trade = [r for r in subset if r["oos_pf"] >= 5]
        ht_best = max(high_trade, key=lambda r: r["oos_trades"]) if high_trade else None

        print(f"\n  --- HTF ADX >= {htf} {'(off)' if htf == 0 else ''} ---")
        if bl_best:
            pf_s = f"{bl_best['oos_pf']:.2f}" if not math.isinf(bl_best['oos_pf']) else "inf"
            print(f"    Baseline best:  n={bl_best['n_bricks']} cd={bl_best['cooldown']} | "
                  f"OOS PF={pf_s} T={bl_best['oos_trades']} WR={bl_best['oos_wr']:.1f}%")
        if add_best:
            pf_s = f"{add_best['oos_pf']:.2f}" if not math.isinf(add_best['oos_pf']) else "inf"
            print(f"    Best additive:  {add_best['config']:<40} n={add_best['n_bricks']} cd={add_best['cooldown']} | "
                  f"OOS PF={pf_s} T={add_best['oos_trades']} WR={add_best['oos_wr']:.1f}%")
        if ht_best:
            pf_s = f"{ht_best['oos_pf']:.2f}" if not math.isinf(ht_best['oos_pf']) else "inf"
            print(f"    Most trades:    {ht_best['config']:<40} n={ht_best['n_bricks']} cd={ht_best['cooldown']} | "
                  f"OOS PF={pf_s} T={ht_best['oos_trades']} WR={ht_best['oos_wr']:.1f}%")

    # -- Per new entry: avg OOS PF lift over baseline --
    print(f"\n  --- New Entry Effectiveness (avg OOS PF, trades >= 10) ---")
    print(f"  {'Entry':<20} {'Avg PF':>8} {'Avg T':>6} {'N':>3} | {'vs Baseline':>11}")
    print(f"  {'-'*60}")

    # Baseline avg for comparison
    baselines_all = [r for r in all_results if r["config"] == "r007+flip_st" and r["oos_trades"] >= 10]
    bl_avg_pf = np.mean([r["oos_pf"] for r in baselines_all if not math.isinf(r["oos_pf"])]) if baselines_all else 0

    entry_stats = []
    for ename in NEW_ENTRIES:
        rows = [r for r in all_results
                if r["config"] == f"r007+flip_st+{ename}" and r["oos_trades"] >= 10]
        if rows:
            finite = [r["oos_pf"] for r in rows if not math.isinf(r["oos_pf"])]
            avg_pf = np.mean(finite) if finite else 0
            avg_t = np.mean([r["oos_trades"] for r in rows])
            delta = avg_pf - bl_avg_pf
            entry_stats.append((ename, avg_pf, avg_t, len(rows), delta))

    entry_stats.sort(key=lambda x: x[1], reverse=True)
    for ename, avg_pf, avg_t, cnt, delta in entry_stats:
        sign = "+" if delta >= 0 else ""
        print(f"  {ename:<20} {avg_pf:>8.2f} {avg_t:>6.1f} {cnt:>3} | {sign}{delta:>9.2f}")

    # -- Overall top 20 --
    all_viable = [r for r in all_results if r["oos_trades"] >= 10]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n{'='*95}")
    print("  Overall Top 20 (OOS trades >= 10)")
    print(f"{'='*95}")
    print(f"  {'Config':<45} {'HTF':>3} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
    print(f"  {'-'*95}")
    for r in all_viable[:20]:
        pf_s = f"{r['oos_pf']:>8.2f}" if not math.isinf(r['oos_pf']) else "     inf"
        is_pf_s = f"{r['is_pf']:>7.2f}" if not math.isinf(r['is_pf']) else "    inf"
        print(f"  {r['config']:<45} {r['htf_thresh']:>3} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{is_pf_s} {r['is_trades']:>4} | "
              f"{pf_s} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    # -- Top 20 by trade count (PF >= 5) --
    high_trade = [r for r in all_results if r["oos_pf"] >= 5 and r["oos_trades"] >= 20]
    high_trade.sort(key=lambda r: r["oos_trades"], reverse=True)

    print(f"\n{'='*95}")
    print("  Top 20 by Trade Count (OOS PF >= 5, trades >= 20)")
    print(f"{'='*95}")
    print(f"  {'Config':<45} {'HTF':>3} {'n':>2} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
    print(f"  {'-'*95}")
    for r in high_trade[:20]:
        pf_s = f"{r['oos_pf']:>8.2f}" if not math.isinf(r['oos_pf']) else "     inf"
        is_pf_s = f"{r['is_pf']:>7.2f}" if not math.isinf(r['is_pf']) else "    inf"
        print(f"  {r['config']:<45} {r['htf_thresh']:>3} {r['n_bricks']:>2} {r['cooldown']:>3} | "
              f"{is_pf_s} {r['is_trades']:>4} | "
              f"{pf_s} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase5_results.json"
    out_path.parent.mkdir(exist_ok=True)

    total_configs_per_htf = (
        len(list(itertools.product(*PARAM_GRID.values())))  # baseline
        + len(NEW_ENTRIES) * len(list(itertools.product(*PARAM_GRID.values())))  # singles
        + 9 * len(list(itertools.product(*PARAM_GRID.values())))  # 2-stacks
        + 5 * len(list(itertools.product(*PARAM_GRID.values())))  # 3-stacks
    )
    total_runs = total_configs_per_htf * len(HTF_ADX_THRESHOLDS)

    print("BTC Phase 5: Extended Hybrid Entry Sweep (Long Only)")
    print(f"  Base         : R007 + FLIP_supertrend (Phase 4 winner)")
    print(f"  New entries  : {NEW_ENTRIES}")
    print(f"  HTF thresholds: {HTF_ADX_THRESHOLDS}")
    print(f"  Params       : {PARAM_GRID}")
    print(f"  Total runs   : {total_runs} ({total_configs_per_htf} per HTF x {len(HTF_ADX_THRESHOLDS)} HTF)")
    print(f"  IS period    : {IS_START} -> {IS_END}")
    print(f"  OOS period   : {OOS_START} -> {OOS_END}")
    print()

    all_results = []

    if args.no_parallel:
        for htf in HTF_ADX_THRESHOLDS:
            all_results.extend(_sweep_htf_threshold(htf))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_sweep_htf_threshold, htf): htf
                for htf in HTF_ADX_THRESHOLDS
            }
            for future in as_completed(futures):
                htf = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [HTF{htf}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [HTF{htf}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
