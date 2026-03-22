#!/usr/bin/env python3
"""
btc_phase11_htf_entry_sweep.py -- HTF Entry Signal Deep Dive (Long Only)

Phase 10 discovered that using HTF ($300) indicator crossovers as ENTRY signals
(not just gates) produces PF 101.52 with 86% WR. This sweep digs deeper:

Part A: Individual HTF crossover signals (7 signals)
    htf_st_flip     HTF supertrend flips bullish
    htf_ema_cross   HTF EMA9 crosses above EMA21
    htf_macd_flip   HTF MACD histogram crosses positive
    htf_rsi_cross   HTF RSI crosses above 50
    htf_kama_turn   HTF KAMA slope turns positive
    htf_psar_flip   HTF PSAR direction flips bullish
    htf_di_cross    HTF +DI crosses above -DI

Part B: HTF signal stacks (combine multiple for more trades)
    htf_2sig        st_flip OR ema_cross (Phase 10 original)
    htf_3sig        st_flip OR ema_cross OR macd_flip
    htf_4sig        + rsi_cross
    htf_5sig        + kama_turn
    htf_all7        all 7 signals (any fires)

Part C: HTF signal + LTF confirmation (filter HTF with LTF state)
    htf2+ltf_rsi    htf_2sig + require LTF RSI > 50
    htf2+ltf_macd   htf_2sig + require LTF MACD_hist > 0
    htf2+ltf_conf2  htf_2sig + require 2-of-3 (RSI>50, MACD>0, KAMA_up)
    htf3+ltf_rsi    htf_3sig + require LTF RSI > 50
    htf3+ltf_conf2  htf_3sig + require 2-of-3

Part D: HTF + BTC003 hybrid (stack HTF entries on top of brick counting)
    btc003+htf2     BTC003 entries + htf_2sig as additional trigger
    btc003+htf3     BTC003 entries + htf_3sig
    btc003+htf_all  BTC003 entries + all 7 HTF signals

Each tested: gates [no_gates, gates_only, gates+htf35] x cooldowns [15, 30, 50]

Usage:
    python renko/btc_phase11_htf_entry_sweep.py
    python renko/btc_phase11_htf_entry_sweep.py --no-parallel
"""

import argparse
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

# -- Config --------------------------------------------------------------------

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

COOLDOWNS = [15, 30, 50]
GATE_CONFIGS = [
    (False, 0,  "no_gates"),
    (True,  0,  "gates_only"),
    (True,  35, "gates+htf35"),
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

def _compute_gates(df_ltf, df_htf, use_gates, htf_adx_thresh):
    n = len(df_ltf)
    gate = np.ones(n, dtype=bool)

    if use_gates:
        sys.path.insert(0, str(ROOT))
        from renko.phase6_sweep import _compute_gate_arrays
        p6_long, _ = _compute_gate_arrays(df_ltf, "psar_dir")
        gate &= p6_long
        adx = df_ltf["adx"].values
        gate &= (np.isnan(adx) | (adx >= ADX_THRESH))
        vr = df_ltf["vol_ratio"].values
        gate &= (np.isnan(vr) | (vr <= VOL_MAX))

    if htf_adx_thresh > 0:
        htf_adx = df_htf["adx"].values
        htf_gate = np.isnan(htf_adx) | (htf_adx >= htf_adx_thresh)
        htf_frame = pd.DataFrame({"t": df_htf.index.values, "g": htf_gate.astype(float)}).sort_values("t")
        ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
        merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
        g = merged["g"].values
        gate &= np.where(np.isnan(g), True, g > 0.5).astype(bool)

    return gate


# -- HTF alignment helper -----------------------------------------------------

def _align_htf_to_ltf(df_ltf, df_htf, htf_col):
    htf_vals = df_htf[htf_col].values
    htf_frame = pd.DataFrame({"t": df_htf.index.values, "v": htf_vals.astype(float)}).sort_values("t")
    ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    return merged["v"].values


# -- Precompute all HTF crossover arrays on LTF timeline -----------------------

def _precompute_htf_crossovers(df_ltf, df_htf):
    """Precompute all 7 HTF crossover bool arrays aligned to LTF bars."""
    n = len(df_ltf)
    crossovers = {}

    # Align HTF indicators to LTF
    htf_st   = _align_htf_to_ltf(df_ltf, df_htf, "st_dir")
    htf_ema9 = _align_htf_to_ltf(df_ltf, df_htf, "ema9")
    htf_ema21= _align_htf_to_ltf(df_ltf, df_htf, "ema21")
    htf_macd = _align_htf_to_ltf(df_ltf, df_htf, "macd_hist")
    htf_rsi  = _align_htf_to_ltf(df_ltf, df_htf, "rsi")
    htf_kama = _align_htf_to_ltf(df_ltf, df_htf, "kama_slope")
    htf_psar = _align_htf_to_ltf(df_ltf, df_htf, "psar_dir")
    htf_pdi  = _align_htf_to_ltf(df_ltf, df_htf, "plus_di")
    htf_mdi  = _align_htf_to_ltf(df_ltf, df_htf, "minus_di")

    def _cross_above_val(arr, val):
        out = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(arr[i]) and not np.isnan(arr[i-1]):
                out[i] = arr[i] > val and arr[i-1] <= val
        return out

    def _cross_above_arr(a, b):
        out = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(a[i]) and not np.isnan(b[i]) and not np.isnan(a[i-1]) and not np.isnan(b[i-1]):
                out[i] = a[i] > b[i] and a[i-1] <= b[i-1]
        return out

    crossovers["htf_st_flip"]   = _cross_above_val(htf_st, 0)
    crossovers["htf_ema_cross"] = _cross_above_arr(htf_ema9, htf_ema21)
    crossovers["htf_macd_flip"] = _cross_above_val(htf_macd, 0)
    crossovers["htf_rsi_cross"] = _cross_above_val(htf_rsi, 50)
    crossovers["htf_kama_turn"] = _cross_above_val(htf_kama, 0)
    crossovers["htf_psar_flip"] = _cross_above_val(htf_psar, 0)
    crossovers["htf_di_cross"]  = _cross_above_arr(htf_pdi, htf_mdi)

    return crossovers


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


# ==============================================================================
# Signal generators
# ==============================================================================

def _gen_htf_crossover(df, htf_signals, signal_names, cooldown, gate, ltf_confirm=None):
    """
    Enter on LTF up brick when any of the named HTF crossover signals fires.
    Optional LTF confirmation filter.

    ltf_confirm: None, "rsi", "macd", "conf2" (2-of-3 RSI>50/MACD>0/KAMA_up)
    """
    n = len(df)
    brick_up = df["brick_up"].values

    # Combine HTF signal arrays (any fires)
    combined = np.zeros(n, dtype=bool)
    for sname in signal_names:
        combined |= htf_signals[sname]

    # LTF confirmation arrays
    ltf_rsi = df["rsi"].values if ltf_confirm else None
    ltf_macd = df["macd_hist"].values if ltf_confirm else None
    ltf_kama = df["kama_slope"].values if ltf_confirm else None

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 60

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue

        if not (up and combined[i]):
            continue

        # LTF confirmation
        if ltf_confirm == "rsi":
            if np.isnan(ltf_rsi[i]) or ltf_rsi[i] <= 50:
                continue
        elif ltf_confirm == "macd":
            if np.isnan(ltf_macd[i]) or ltf_macd[i] <= 0:
                continue
        elif ltf_confirm == "conf2":
            confirms = 0
            if not np.isnan(ltf_rsi[i]) and ltf_rsi[i] > 50:
                confirms += 1
            if not np.isnan(ltf_macd[i]) and ltf_macd[i] > 0:
                confirms += 1
            if not np.isnan(ltf_kama[i]) and ltf_kama[i] > 0:
                confirms += 1
            if confirms < 2:
                continue

        entry[i] = True
        in_pos = True
        last_bar = i

    return entry, exit_


def _gen_btc003_plus_htf(df, htf_signals, htf_signal_names, cooldown, gate):
    """BTC003 entries (R001+R002+FLIP+BBRK) + HTF crossover entries stacked."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    n_bricks = 2

    st_dir = df["st_dir"].values
    bb_u = df["bb_upper"].values

    # Combine HTF signals
    htf_any = np.zeros(n, dtype=bool)
    for sname in htf_signal_names:
        htf_any |= htf_signals[sname]

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_r001 = -999_999
    last_flip = -999_999
    last_bb = -999_999
    last_htf = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i]:
            continue

        triggered = False

        # R002: reversal (no cooldown)
        prev = brick_up[i - n_bricks: i]
        if bool(not np.any(prev)) and up:
            triggered = True

        # R001: momentum
        if not triggered and (i - last_r001) >= cooldown:
            window = brick_up[i - n_bricks + 1: i + 1]
            if bool(np.all(window)):
                triggered = True
                last_r001 = i

        # FLIP: supertrend flip
        if not triggered and up and (i - last_flip) >= cooldown:
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    triggered = True
                    last_flip = i

        # BBRK: BB breakout
        if not triggered and up and (i - last_bb) >= cooldown:
            if not np.isnan(bb_u[i]) and not np.isnan(bb_u[i-1]):
                if close[i] > bb_u[i] and close[i-1] <= bb_u[i-1]:
                    triggered = True
                    last_bb = i

        # HTF crossover entries (independent cooldown)
        if not triggered and up and htf_any[i] and (i - last_htf) >= cooldown:
            triggered = True
            last_htf = i

        if triggered:
            entry[i] = True
            in_pos = True

    return entry, exit_


# ==============================================================================
# Configuration: all sweep configs
# ==============================================================================

INDIVIDUAL_HTF = [
    "htf_st_flip", "htf_ema_cross", "htf_macd_flip",
    "htf_rsi_cross", "htf_kama_turn", "htf_psar_flip", "htf_di_cross",
]

STACKED_HTF = {
    "htf_2sig":  ["htf_st_flip", "htf_ema_cross"],
    "htf_3sig":  ["htf_st_flip", "htf_ema_cross", "htf_macd_flip"],
    "htf_4sig":  ["htf_st_flip", "htf_ema_cross", "htf_macd_flip", "htf_rsi_cross"],
    "htf_5sig":  ["htf_st_flip", "htf_ema_cross", "htf_macd_flip", "htf_rsi_cross", "htf_kama_turn"],
    "htf_all7":  INDIVIDUAL_HTF,
}

CONFIRMED_HTF = {
    "htf2+ltf_rsi":   (["htf_st_flip", "htf_ema_cross"], "rsi"),
    "htf2+ltf_macd":  (["htf_st_flip", "htf_ema_cross"], "macd"),
    "htf2+ltf_conf2": (["htf_st_flip", "htf_ema_cross"], "conf2"),
    "htf3+ltf_rsi":   (["htf_st_flip", "htf_ema_cross", "htf_macd_flip"], "rsi"),
    "htf3+ltf_conf2": (["htf_st_flip", "htf_ema_cross", "htf_macd_flip"], "conf2"),
}

BTC003_HYBRID = {
    "btc003+htf2":  ["htf_st_flip", "htf_ema_cross"],
    "btc003+htf3":  ["htf_st_flip", "htf_ema_cross", "htf_macd_flip"],
    "btc003+htf_all": INDIVIDUAL_HTF,
}


# ==============================================================================
# Worker: sweep one config group
# ==============================================================================

def _sweep_group(group_name):
    """Run a named group of configs."""
    label = group_name.upper()
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()

    htf_signals = _precompute_htf_crossovers(df_ltf, df_htf)

    gates = {}
    for use_g, htf_t, gname in GATE_CONFIGS:
        gates[gname] = _compute_gates(df_ltf, df_htf, use_g, htf_t)

    results = []

    # Determine which configs belong to this group
    configs = []  # list of (config_name, part, gen_fn_args)

    if group_name.startswith("ind_"):
        # Individual HTF signal
        sig_name = group_name[4:]  # strip "ind_"
        for _, htf_t, gname in GATE_CONFIGS:
            for cd in COOLDOWNS:
                configs.append((sig_name, "A_individual", gname, cd, [sig_name], None, False))

    elif group_name.startswith("stack_"):
        stack_name = group_name[6:]
        sig_names = STACKED_HTF[stack_name]
        for _, htf_t, gname in GATE_CONFIGS:
            for cd in COOLDOWNS:
                configs.append((stack_name, "B_stacked", gname, cd, sig_names, None, False))

    elif group_name.startswith("conf_"):
        conf_name = group_name[5:]
        sig_names, ltf_conf = CONFIRMED_HTF[conf_name]
        for _, htf_t, gname in GATE_CONFIGS:
            for cd in COOLDOWNS:
                configs.append((conf_name, "C_confirmed", gname, cd, sig_names, ltf_conf, False))

    elif group_name.startswith("hybrid_"):
        hybrid_name = group_name[7:]
        sig_names = BTC003_HYBRID[hybrid_name]
        for _, htf_t, gname in GATE_CONFIGS:
            for cd in COOLDOWNS:
                configs.append((hybrid_name, "D_hybrid", gname, cd, sig_names, None, True))

    total = len(configs)
    done = 0

    for config_name, part, gname, cd, sig_names, ltf_conf, is_hybrid in configs:
        gate = gates[gname]

        if is_hybrid:
            e, x = _gen_btc003_plus_htf(df_ltf, htf_signals, sig_names, cd, gate)
        else:
            e, x = _gen_htf_crossover(df_ltf, htf_signals, sig_names, cd, gate, ltf_conf)

        is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
        oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

        is_pf = is_r["pf"]
        oos_pf = oos_r["pf"]
        decay = ((oos_pf - is_pf) / is_pf * 100) \
                if is_pf > 0 and not math.isinf(is_pf) else float("nan")

        results.append({
            "part":       part,
            "config":     config_name,
            "htf_sigs":   "+".join(sig_names),
            "ltf_conf":   ltf_conf or "none",
            "gates":      gname,
            "cooldown":   cd,
            "is_pf":      is_pf,
            "is_trades":  is_r["trades"],
            "is_wr":      is_r["wr"],
            "oos_pf":     oos_pf,
            "oos_trades": oos_r["trades"],
            "oos_wr":     oos_r["wr"],
            "oos_net":    oos_r["net"],
            "oos_dd":     oos_r["dd"],
            "decay_pct":  decay,
        })

        done += 1
        pf_s = f"{oos_pf:.2f}" if not math.isinf(oos_pf) else "inf"
        print(f"  [{label}] {done:>2}/{total} | {config_name:<22} {gname:<15} cd={cd:>2} | "
              f"OOS PF={pf_s:>8} T={oos_r['trades']:>4} WR={oos_r['wr']:>5.1f}%",
              flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    parts = {
        "A_individual": "Part A: Individual HTF Crossover Signals",
        "B_stacked": "Part B: HTF Signal Stacks",
        "C_confirmed": "Part C: HTF + LTF Confirmation",
        "D_hybrid": "Part D: BTC003 + HTF Hybrid",
    }

    for part_key, part_title in parts.items():
        subset = [r for r in all_results if r["part"] == part_key]
        if not subset:
            continue

        print(f"\n{'='*110}")
        print(f"  {part_title}")
        print(f"{'='*110}")

        # Best per config
        configs = sorted(set(r["config"] for r in subset))
        print(f"\n  {'Config':<22} {'Gates':<15} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
              f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
        print(f"  {'-'*95}")

        for cname in configs:
            viable = [r for r in subset if r["config"] == cname and r["oos_trades"] >= 5]
            if not viable:
                print(f"  {cname:<22} (< 5 OOS trades in all configs)")
                continue
            best = max(viable, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
            pf_s = f"{best['oos_pf']:.2f}" if not math.isinf(best['oos_pf']) else "inf"
            is_pf_s = f"{best['is_pf']:.2f}" if not math.isinf(best['is_pf']) else "inf"
            print(f"  {cname:<22} {best['gates']:<15} {best['cooldown']:>3} | "
                  f"{is_pf_s:>7} {best['is_trades']:>4} | "
                  f"{pf_s:>8} {best['oos_trades']:>4} {best['oos_wr']:>5.1f}% "
                  f"{best['oos_net']:>8.2f}")

    # Grand top 25
    all_viable = [r for r in all_results if r["oos_trades"] >= 5]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n{'='*110}")
    print("  GRAND TOP 25 — All Configs (OOS trades >= 5)")
    print(f"{'='*110}")
    print(f"  {'Part':<4} {'Config':<22} {'LTF':<6} {'Gates':<15} {'cd':>3} | "
          f"{'IS PF':>7} {'T':>4} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
    print(f"  {'-'*105}")
    for r in all_viable[:25]:
        pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
        is_pf_s = f"{r['is_pf']:.2f}" if not math.isinf(r['is_pf']) else "inf"
        part_tag = r["part"].split("_")[0]
        print(f"  {part_tag:<4} {r['config']:<22} {r['ltf_conf']:<6} {r['gates']:<15} {r['cooldown']:>3} | "
              f"{is_pf_s:>7} {r['is_trades']:>4} | "
              f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    # High frequency
    high_trade = [r for r in all_results if r["oos_pf"] >= 10 and r["oos_trades"] >= 30]
    high_trade.sort(key=lambda r: r["oos_trades"], reverse=True)
    if high_trade:
        print(f"\n  Sweet Spot: OOS PF >= 10, trades >= 30:")
        print(f"  {'Part':<4} {'Config':<22} {'LTF':<6} {'Gates':<15} {'cd':>3} | "
              f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
        print(f"  {'-'*90}")
        for r in high_trade[:15]:
            pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
            part_tag = r["part"].split("_")[0]
            print(f"  {part_tag:<4} {r['config']:<22} {r['ltf_conf']:<6} {r['gates']:<15} {r['cooldown']:>3} | "
                  f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    print(f"\n  --- Reference ---")
    print(f"  BTC003 OOS: PF=49.27, 76t, WR=76.3%")
    print(f"  Phase 10 HTF_ENTRY gates_only cd=35: PF=101.52, 29t, WR=86.2%")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase11_htf_entry_results.json"
    out_path.parent.mkdir(exist_ok=True)

    # Build worker groups
    groups = []
    for sig in INDIVIDUAL_HTF:
        groups.append(f"ind_{sig}")
    for stack in STACKED_HTF:
        groups.append(f"stack_{stack}")
    for conf in CONFIRMED_HTF:
        groups.append(f"conf_{conf}")
    for hybrid in BTC003_HYBRID:
        groups.append(f"hybrid_{hybrid}")

    runs_per_group = len(GATE_CONFIGS) * len(COOLDOWNS)
    total = len(groups) * runs_per_group

    print("BTC Phase 11: HTF Entry Signal Deep Dive (Long Only)")
    print(f"  Worker groups: {len(groups)} ({len(INDIVIDUAL_HTF)} individual + "
          f"{len(STACKED_HTF)} stacked + {len(CONFIRMED_HTF)} confirmed + "
          f"{len(BTC003_HYBRID)} hybrid)")
    print(f"  Per group    : {runs_per_group} runs (3 gates x {len(COOLDOWNS)} cooldowns)")
    print(f"  Total runs   : {total}")
    print(f"  IS period    : {IS_START} -> {IS_END}")
    print(f"  OOS period   : {OOS_START} -> {OOS_END}")
    print()

    all_results = []

    if args.no_parallel:
        for g in groups:
            all_results.extend(_sweep_group(g))
    else:
        with ProcessPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_sweep_group, g): g for g in groups}
            for future in as_completed(futures):
                g = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{g.upper()}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{g.upper()}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
