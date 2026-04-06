#!/usr/bin/env python3
"""
btc_hf_phase5_sweep.py -- BTC HF Phase 5: HTF-Gated Best-of-Breed (Long Only)

Combines Phase 4 winners with HTF $300 ADX gating:

  Part A — HTF-gated combined entries (best-of-breed + HTF):
    trio_stoch:       ST_flip + MACD_flip + KAMA_turn + stoch_cross
    trio_stoch_wpr:   ST_flip + MACD_flip + KAMA_turn + stoch_cross + wpr_cross
    trio_stoch_escgo: ST_flip + MACD_flip + KAMA_turn + stoch_cross + escgo_cross
    full_quality:     ST_flip + MACD_flip + KAMA_turn + stoch_cross + wpr_cross + escgo_cross + cci_cross
    r001_trio:        R001 + ST_flip + MACD_flip + KAMA_turn
    r001_trio_r002:   R001 + R002 + ST_flip + MACD_flip + KAMA_turn

  Part B — HTF-gated new P6 signals (best individuals from P4A):
    WPR + HTF, ESCGO + HTF, DDL + HTF

  Part C — Regime-adaptive: use ADX/Chop to pick entry mode
    High ADX (trending): use R001/ST_flip/MACD_flip
    Low ADX (ranging):   use stoch_cross/wpr_cross/bb_bounce

  All with PSAR gate, HTF ADX thresholds [0, 25, 30, 35]
  Cooldowns: [3, 5]

Usage:
    python renko/btc_hf_phase5_sweep.py
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
HTF_FILE   = "OANDA_BTCUSD.SPOT.US, 1S renko 300.csv"
IS_START   = "2024-06-04"
IS_END     = "2025-09-30"
OOS_START  = "2025-10-01"
OOS_END    = "2026-03-19"
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20
OOS_DAYS   = 170


def _load_ltf_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators
    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df)
    return df


def _load_htf_data():
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(HTF_FILE)
    add_renko_indicators(df)
    return df


def _compute_psar_gate(df):
    n = len(df)
    gate = np.ones(n, dtype=bool)
    psar = df["psar_dir"].values
    gate &= (np.isnan(psar) | (psar > 0))
    return gate


def _align_htf_gate(df_ltf, df_htf, htf_gate_arr):
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values,
        "g": htf_gate_arr.astype(float),
    }).sort_values("t")
    ltf_frame = pd.DataFrame({
        "t": df_ltf.index.values,
    }).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    g = merged["g"].values
    return np.where(np.isnan(g), True, g > 0.5).astype(bool)


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


# =============================================================================
# Part A — HTF-gated combined entries with per-signal cooldowns
# =============================================================================

def _gen_combined(df, gate, signals, cd_map):
    """Combined entry generator with per-signal cooldowns."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    st_dir = df["st_dir"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    stoch_k = df["stoch_k"].values
    bb_pct = df["bb_pct_b"].values
    rsi = df["rsi"].values
    cci = df["cci"].values
    wpr = df["wpr"].values
    escgo_f = df["escgo_fast"].values
    escgo_s = df["escgo_slow"].values
    adx = df["adx"].values
    chop = df["chop"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    warmup = 60
    last_bar = {s: -999_999 for s in signals}

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up:
            continue

        fired = False

        # R001: 2 consecutive up bricks
        if not fired and "r001" in signals and (i - last_bar["r001"]) >= cd_map.get("r001", 5):
            if brick_up[i-1]:
                fired = True
                last_bar["r001"] = i

        # R002: 2 down then up
        if not fired and "r002" in signals and (i - last_bar.get("r002", -999_999)) >= cd_map.get("r002", 0):
            if i >= 3 and not brick_up[i-1] and not brick_up[i-2]:
                fired = True
                last_bar["r002"] = i

        # ST flip
        if not fired and "st_flip" in signals and (i - last_bar["st_flip"]) >= cd_map.get("st_flip", 3):
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True
                    last_bar["st_flip"] = i

        # MACD flip
        if not fired and "macd_flip" in signals and (i - last_bar["macd_flip"]) >= cd_map.get("macd_flip", 3):
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True
                    last_bar["macd_flip"] = i

        # KAMA turn
        if not fired and "kama_turn" in signals and (i - last_bar["kama_turn"]) >= cd_map.get("kama_turn", 3):
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True
                    last_bar["kama_turn"] = i

        # Stoch cross (K crosses above 25)
        if not fired and "stoch_cross" in signals and (i - last_bar["stoch_cross"]) >= cd_map.get("stoch_cross", 3):
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                    fired = True
                    last_bar["stoch_cross"] = i

        # WPR cross (-70)
        if not fired and "wpr_cross" in signals and (i - last_bar["wpr_cross"]) >= cd_map.get("wpr_cross", 3):
            if not np.isnan(wpr[i]) and not np.isnan(wpr[i-1]):
                if wpr[i] >= -70 and wpr[i-1] < -70:
                    fired = True
                    last_bar["wpr_cross"] = i

        # ESCGO cross
        if not fired and "escgo_cross" in signals and (i - last_bar["escgo_cross"]) >= cd_map.get("escgo_cross", 3):
            if not np.isnan(escgo_f[i]) and not np.isnan(escgo_s[i]):
                if not np.isnan(escgo_f[i-1]) and not np.isnan(escgo_s[i-1]):
                    if escgo_f[i] > escgo_s[i] and escgo_f[i-1] <= escgo_s[i-1]:
                        fired = True
                        last_bar["escgo_cross"] = i

        # CCI cross (-100)
        if not fired and "cci_cross" in signals and (i - last_bar["cci_cross"]) >= cd_map.get("cci_cross", 3):
            if not np.isnan(cci[i]) and not np.isnan(cci[i-1]):
                if cci[i] >= -100 and cci[i-1] < -100:
                    fired = True
                    last_bar["cci_cross"] = i

        # BB bounce
        if not fired and "bb_bounce" in signals and (i - last_bar["bb_bounce"]) >= cd_map.get("bb_bounce", 3):
            if not np.isnan(bb_pct[i]):
                if bb_pct[i] <= 0.20:
                    fired = True
                    last_bar["bb_bounce"] = i

        if fired:
            entry[i] = True
            in_pos = True

    return entry, exit_


# =============================================================================
# Part C — Regime-adaptive entries
# =============================================================================

def _gen_regime_adaptive(df, gate, cd, adx_threshold=25, chop_threshold=50):
    """
    Regime-adaptive: different signals based on market state.
    High ADX (trending): R001 + ST_flip + MACD_flip (momentum signals)
    Low ADX (ranging):   stoch_cross + wpr_cross + bb_bounce (MR signals)
    """
    n = len(df)
    brick_up = df["brick_up"].values
    st_dir = df["st_dir"].values
    macd_h = df["macd_hist"].values
    stoch_k = df["stoch_k"].values
    wpr = df["wpr"].values
    bb_pct = df["bb_pct_b"].values
    adx = df["adx"].values
    chop = df["chop"].values

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

        if not gate[i] or not up or (i - last_bar) < cd:
            continue

        fired = False
        is_trending = (not np.isnan(adx[i]) and adx[i] >= adx_threshold)

        if is_trending:
            # Momentum signals in trending markets
            # R001
            if brick_up[i-1]:
                fired = True
            # ST flip
            if not fired and not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True
            # MACD flip
            if not fired and not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True
        else:
            # MR signals in ranging markets
            # Stoch cross
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                    fired = True
            # WPR cross
            if not fired and not np.isnan(wpr[i]) and not np.isnan(wpr[i-1]):
                if wpr[i] >= -70 and wpr[i-1] < -70:
                    fired = True
            # BB bounce
            if not fired and not np.isnan(bb_pct[i]):
                if bb_pct[i] <= 0.20:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# =============================================================================
# Combo builders
# =============================================================================

SIGNAL_SETS = {
    "trio_stoch":       ["st_flip", "macd_flip", "kama_turn", "stoch_cross"],
    "trio_stoch_wpr":   ["st_flip", "macd_flip", "kama_turn", "stoch_cross", "wpr_cross"],
    "trio_stoch_escgo": ["st_flip", "macd_flip", "kama_turn", "stoch_cross", "escgo_cross"],
    "full_quality":     ["st_flip", "macd_flip", "kama_turn", "stoch_cross", "wpr_cross", "escgo_cross", "cci_cross"],
    "r001_trio":        ["r001", "st_flip", "macd_flip", "kama_turn"],
    "r001_trio_r002":   ["r001", "r002", "st_flip", "macd_flip", "kama_turn"],
    "r001_trio_stoch":  ["r001", "st_flip", "macd_flip", "kama_turn", "stoch_cross"],
    "r001_full":        ["r001", "r002", "st_flip", "macd_flip", "kama_turn", "stoch_cross", "wpr_cross", "escgo_cross"],
}

HTF_THRESHOLDS = [0, 25, 30, 35]  # 0 = no HTF gate
COOLDOWNS = [3, 5]

CD_FAST = {s: 3 for s in ["st_flip", "macd_flip", "kama_turn", "stoch_cross",
           "wpr_cross", "escgo_cross", "cci_cross", "bb_bounce", "r001", "r002"]}

CD_MIXED = dict(CD_FAST)
CD_MIXED.update({"r001": 5, "st_flip": 5})


def _build_part_a():
    combos = []
    for set_name, sigs in SIGNAL_SETS.items():
        for htf_t in HTF_THRESHOLDS:
            for cd_name, cd_map in [("fast", CD_FAST), ("mixed", CD_MIXED)]:
                combos.append({
                    "part": "A",
                    "set_name": set_name,
                    "signals": sigs,
                    "cd_map": {s: cd_map.get(s, 3) for s in sigs},
                    "cd_name": cd_name,
                    "htf_thresh": htf_t,
                    "label": f"{set_name}_{cd_name}_htf{htf_t}",
                })
    return combos


def _build_part_b():
    """HTF-gated individual new P6 signals."""
    combos = []
    for sig, gen_name in [("wpr_cross", "wpr"), ("escgo_cross", "escgo"), ("ddl_cross", "ddl")]:
        for htf_t in HTF_THRESHOLDS:
            for cd in COOLDOWNS:
                combos.append({
                    "part": "B",
                    "signal": sig,
                    "cooldown": cd,
                    "htf_thresh": htf_t,
                    "label": f"{gen_name}_htf{htf_t}_cd{cd}",
                })
    return combos


def _build_part_c():
    """Regime-adaptive entries."""
    combos = []
    for adx_t in [20, 25, 30]:
        for htf_t in HTF_THRESHOLDS:
            for cd in COOLDOWNS:
                combos.append({
                    "part": "C",
                    "adx_thresh": adx_t,
                    "cooldown": cd,
                    "htf_thresh": htf_t,
                    "label": f"regime_adx{adx_t}_htf{htf_t}_cd{cd}",
                })
    return combos


# =============================================================================
# Individual signal generators for Part B
# =============================================================================

def _gen_single_signal(df, gate, signal, cd):
    n = len(df)
    brick_up = df["brick_up"].values
    wpr = df["wpr"].values
    escgo_f = df["escgo_fast"].values
    escgo_s = df["escgo_slow"].values
    ddl = df["ddl_diff"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 30

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not up or (i - last_bar) < cd:
            continue

        fired = False
        if signal == "wpr_cross":
            if not np.isnan(wpr[i]) and not np.isnan(wpr[i-1]):
                if wpr[i] >= -70 and wpr[i-1] < -70:
                    fired = True
        elif signal == "escgo_cross":
            if not np.isnan(escgo_f[i]) and not np.isnan(escgo_s[i]):
                if not np.isnan(escgo_f[i-1]) and not np.isnan(escgo_s[i-1]):
                    if escgo_f[i] > escgo_s[i] and escgo_f[i-1] <= escgo_s[i-1]:
                        fired = True
        elif signal == "ddl_cross":
            if not np.isnan(ddl[i]) and not np.isnan(ddl[i-1]):
                if ddl[i] > 0 and ddl[i-1] <= 0:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# =============================================================================
# Worker
# =============================================================================

_w = {}

def _init_worker():
    if "df" not in _w:
        _w["df"] = _load_ltf_data()
        _w["df_htf"] = _load_htf_data()
        _w["psar_gate"] = _compute_psar_gate(_w["df"])
        # Pre-compute HTF gates
        _w["htf_gates"] = {0: np.ones(len(_w["df"]), dtype=bool)}
        df_htf = _w["df_htf"]
        adx_htf = df_htf["adx"].values
        adx_nan = np.isnan(adx_htf)
        for thresh in [25, 30, 35]:
            htf_arr = adx_nan | (adx_htf >= thresh)
            _w["htf_gates"][thresh] = _align_htf_gate(_w["df"], df_htf, htf_arr)


def _run_one(combo):
    _init_worker()
    df = _w["df"]
    psar_gate = _w["psar_gate"]
    htf_gate = _w["htf_gates"][combo["htf_thresh"]]
    gate = psar_gate & htf_gate

    part = combo["part"]

    if part == "A":
        entry, exit_ = _gen_combined(df, gate, combo["signals"], combo["cd_map"])
    elif part == "B":
        entry, exit_ = _gen_single_signal(df, gate, combo["signal"], combo["cooldown"])
    elif part == "C":
        entry, exit_ = _gen_regime_adaptive(df, gate, combo["cooldown"],
                                             adx_threshold=combo["adx_thresh"])
    else:
        return combo, {"pf": 0}, {"pf": 0}

    is_r = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    return combo, is_r, oos_r


# =============================================================================
# Summary
# =============================================================================

def _print_header():
    print(f"  {'#':>3} {'Part':>4} {'Label':<40} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*120}")


def _print_row(r, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / OOS_DAYS if r["oos_trades"] > 0 else 0
    prefix = f"  {rank:>3}" if rank else "  "
    print(f"{prefix} {r['part']:>4} {r['label']:<40} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


def _summarize(all_results):
    for part_name, part_title in [
        ("A", "Part A — HTF-Gated Combined Entries"),
        ("B", "Part B — HTF-Gated New P6 Signals"),
        ("C", "Part C — Regime-Adaptive Entries"),
    ]:
        subset = [r for r in all_results if r["part"] == part_name]
        viable = [r for r in subset if r["oos_trades"] >= 10 and r["oos_net"] > 0]
        viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)

        print(f"\n{'='*130}")
        print(f"  {part_title} — Top by WR ({len(viable)} viable / {len(subset)} total)")
        print(f"{'='*130}")
        _print_header()
        for i, r in enumerate(viable[:20]):
            _print_row(r, rank=i+1)

        # HF subset
        hf = [r for r in subset if r["oos_trades"] >= OOS_DAYS and r["oos_net"] > 0]
        hf.sort(key=lambda r: r["oos_net"], reverse=True)
        if hf:
            print(f"\n  HF subset (>= 1/day, net > 0): {len(hf)} configs — by net")
            _print_header()
            for i, r in enumerate(hf[:15]):
                _print_row(r, rank=i+1)

    # Global rankings
    for title, filt_fn, sort_key in [
        ("GLOBAL TOP 20 by Net (T>=50)",
         lambda r: r["oos_trades"] >= 50 and r["oos_net"] > 0,
         lambda r: r["oos_net"]),
        ("GLOBAL BEST WR (T>=100)",
         lambda r: r["oos_trades"] >= 100 and r["oos_net"] > 0,
         lambda r: (r["oos_wr"], r["oos_net"])),
        ("BEST BALANCED (T>=170, WR>=58%)",
         lambda r: r["oos_trades"] >= OOS_DAYS and r["oos_wr"] >= 58.0 and r["oos_net"] > 0,
         lambda r: (r["oos_net"], r["oos_wr"])),
    ]:
        subset = sorted([r for r in all_results if filt_fn(r)], key=sort_key, reverse=True)
        print(f"\n{'='*130}")
        print(f"  {title}: {len(subset)} configs")
        print(f"{'='*130}")
        _print_header()
        for i, r in enumerate(subset[:20]):
            _print_row(r, rank=i+1)


# =============================================================================
# Main
# =============================================================================

def main():
    combos_a = _build_part_a()
    combos_b = _build_part_b()
    combos_c = _build_part_c()
    all_combos = combos_a + combos_b + combos_c
    total = len(all_combos)

    print(f"\n{'='*70}")
    print(f"BTC HF Phase 5 — HTF-Gated Best-of-Breed")
    print(f"  Part A (HTF combined):     {len(combos_a)} combos")
    print(f"  Part B (HTF new signals):  {len(combos_b)} combos")
    print(f"  Part C (regime-adaptive):  {len(combos_c)} combos")
    print(f"  Total runs: {total} ({total*2} backtests)")
    print(f"  Workers:    {MAX_WORKERS}")
    print(f"{'='*70}\n")

    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in all_combos}
        for fut in as_completed(futures):
            try:
                combo, is_r, oos_r = fut.result()
                row = {
                    "part":       combo["part"],
                    "label":      combo["label"],
                    "htf_thresh": combo["htf_thresh"],
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
            if done % 30 == 0 or done == total:
                print(f"  [{done:>4}/{total}]", flush=True)

    # Save
    out_path = ROOT / "ai_context" / "btc_hf_phase5_results.json"
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
