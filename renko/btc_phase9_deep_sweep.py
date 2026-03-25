#!/usr/bin/env python3
"""
btc_phase9_deep_sweep.py -- BTC Phase 9: Confluence Deep Dive + 10 More Strategies

Part A: TRIPLE_CONFIRM deep dive
    Tests 9 different indicator pool combos (5-set and 7-set) with variable
    thresholds (how many must agree). Finds the optimal indicator mix.

Part B: 10 more standalone strategy concepts using indicators not yet tested:
    OBV_CROSS      OBV crosses above OBV_EMA (accumulation confirmed)
    CMF_BULL       CMF crosses above 0 (money inflow)
    MFI_BOUNCE     MFI crosses above 20 from oversold
    CHOP_TREND     Choppiness drops below 38.2 (trending regime)
    CCI_CROSS      CCI crosses above 0 (momentum flip)
    ICHI_BREAK     Ichimoku position flips bullish (+1)
    WPR_BOUNCE     Williams %R crosses above -80 from oversold
    DDL_BULL       DDL diff crosses above 0 (defense line bullish)
    MOTN_BULL      MOTN DX crosses above 0
    DONCH_BREAK    Close crosses above Donchian midpoint

Usage:
    python renko/btc_phase9_deep_sweep.py
    python renko/btc_phase9_deep_sweep.py --no-parallel
"""

import argparse
import contextlib
import io
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

COOLDOWNS = [25, 40]
GATE_CONFIGS = [
    (True, 0,  "gates_only"),
    (True, 35, "gates+htf35"),
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
# PART A: TRIPLE_CONFIRM — Confluence indicator pool variations
# ==============================================================================

# Boolean indicator definitions: each returns a bool array (True = bullish)
def _bool_indicators(df):
    """Compute all boolean indicator arrays. Returns dict name -> bool array."""
    n = len(df)
    close = df["Close"].values.astype(float)
    indicators = {}

    def _safe_gt(arr, val):
        out = np.zeros(n, dtype=bool)
        valid = ~np.isnan(arr)
        out[valid] = arr[valid] > val
        return out

    def _safe_cross_above(a, b):
        """a crosses above b: a[i] > b[i] and a[i-1] <= b[i-1]."""
        out = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(a[i]) and not np.isnan(b[i]) and not np.isnan(a[i-1]) and not np.isnan(b[i-1]):
                out[i] = a[i] > b[i] and a[i-1] <= b[i-1]
        return out

    # Level-based (state, not crossover)
    indicators["rsi_50"]      = _safe_gt(df["rsi"].values, 50)
    indicators["macd_h_pos"]  = _safe_gt(df["macd_hist"].values, 0)
    indicators["stoch_50"]    = _safe_gt(df["stoch_k"].values, 50)
    indicators["kama_up"]     = _safe_gt(df["kama_slope"].values, 0)
    indicators["st_bull"]     = _safe_gt(df["st_dir"].values, 0)
    indicators["ema9_21"]     = np.zeros(n, dtype=bool)
    e9, e21 = df["ema9"].values, df["ema21"].values
    valid = ~(np.isnan(e9) | np.isnan(e21))
    indicators["ema9_21"][valid] = e9[valid] > e21[valid]

    indicators["cmf_pos"]     = _safe_gt(df["cmf"].values, 0)
    indicators["mfi_50"]      = _safe_gt(df["mfi"].values, 50)
    indicators["psar_bull"]   = _safe_gt(df["psar_dir"].values, 0)
    indicators["cci_pos"]     = _safe_gt(df["cci"].values, 0)

    indicators["obv_above"]   = np.zeros(n, dtype=bool)
    obv, obv_e = df["obv"].values, df["obv_ema"].values
    valid_o = ~(np.isnan(obv) | np.isnan(obv_e))
    indicators["obv_above"][valid_o] = obv[valid_o] > obv_e[valid_o]

    indicators["escgo_bull"]  = np.zeros(n, dtype=bool)
    ef, es = df["escgo_fast"].values, df["escgo_slow"].values
    valid_e = ~(np.isnan(ef) | np.isnan(es))
    indicators["escgo_bull"][valid_e] = ef[valid_e] > es[valid_e]

    return indicators


# Named indicator sets for confluence testing
CONFLUENCE_SETS = {
    # 5-indicator sets
    "C5_orig":     (["rsi_50", "macd_h_pos", "stoch_50", "kama_up", "st_bull"], [2, 3, 4]),
    "C5_trend":    (["ema9_21", "kama_up", "st_bull", "psar_bull", "macd_h_pos"], [2, 3, 4]),
    "C5_momentum": (["rsi_50", "stoch_50", "cci_pos", "mfi_50", "macd_h_pos"], [2, 3, 4]),
    "C5_volume":   (["cmf_pos", "obv_above", "mfi_50", "macd_h_pos", "rsi_50"], [2, 3, 4]),
    "C5_mixed1":   (["rsi_50", "cmf_pos", "st_bull", "escgo_bull", "kama_up"], [2, 3, 4]),
    "C5_mixed2":   (["macd_h_pos", "stoch_50", "psar_bull", "ema9_21", "obv_above"], [2, 3, 4]),
    "C5_diverse":  (["rsi_50", "macd_h_pos", "st_bull", "cmf_pos", "cci_pos"], [2, 3, 4]),
    # 7-indicator sets
    "C7_wide":     (["rsi_50", "macd_h_pos", "stoch_50", "kama_up", "st_bull", "cmf_pos", "ema9_21"], [3, 4, 5]),
    "C7_all_osc":  (["rsi_50", "stoch_50", "cci_pos", "mfi_50", "macd_h_pos", "escgo_bull", "cmf_pos"], [3, 4, 5]),
}


def _gen_confluence(df, ind_dict, indicator_names, threshold, cooldown, gate):
    """N-of-M confluence entry on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    arrays = [ind_dict[name] for name in indicator_names]
    num_indicators = len(arrays)

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
        if not gate[i]:
            continue
        if (i - last_bar) < cooldown:
            continue

        confirms = sum(1 for arr in arrays if arr[i])
        if up and confirms >= threshold:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# PART B: 10 More standalone strategies
# ==============================================================================

def _gen_obv_cross(df, cooldown, gate):
    """OBV crosses above OBV_EMA on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    obv = df["obv"].values
    obv_ema = df["obv_ema"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(30, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(obv[i]) or np.isnan(obv_ema[i]) or np.isnan(obv[i-1]) or np.isnan(obv_ema[i-1]):
            continue
        if up and obv[i] > obv_ema[i] and obv[i-1] <= obv_ema[i-1]:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_cmf_bull(df, cooldown, gate):
    """CMF crosses above 0 on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    cmf = df["cmf"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(25, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(cmf[i]) or np.isnan(cmf[i-1]):
            continue
        if up and cmf[i] > 0 and cmf[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_mfi_bounce(df, cooldown, gate):
    """MFI crosses above 20 from oversold on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    mfi = df["mfi"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(20, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(mfi[i]) or np.isnan(mfi[i-1]):
            continue
        if up and mfi[i] > 20 and mfi[i-1] <= 20:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_chop_trend(df, cooldown, gate):
    """Choppiness drops below 38.2 (trending regime) on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    chop = df["chop"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(20, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(chop[i]) or np.isnan(chop[i-1]):
            continue
        if up and chop[i] < 38.2 and chop[i-1] >= 38.2:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_cci_cross(df, cooldown, gate):
    """CCI crosses above 0 on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    cci = df["cci"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(25, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(cci[i]) or np.isnan(cci[i-1]):
            continue
        if up and cci[i] > 0 and cci[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_ichi_break(df, cooldown, gate):
    """Ichimoku position flips to +1 (above cloud) on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    ichi = df["ichi_pos"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(60, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(ichi[i]) or np.isnan(ichi[i-1]):
            continue
        if up and ichi[i] > 0 and ichi[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_wpr_bounce(df, cooldown, gate):
    """Williams %R crosses above -80 from oversold on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    wpr = df["wpr"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(20, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(wpr[i]) or np.isnan(wpr[i-1]):
            continue
        if up and wpr[i] > -80 and wpr[i-1] <= -80:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_ddl_bull(df, cooldown, gate):
    """DDL diff crosses above 0 on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    ddl = df["ddl_diff"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(30, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(ddl[i]) or np.isnan(ddl[i-1]):
            continue
        if up and ddl[i] > 0 and ddl[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_motn_bull(df, cooldown, gate):
    """MOTN DX crosses above 0 on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    motn = df["motn_dx"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(30, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(motn[i]) or np.isnan(motn[i-1]):
            continue
        if up and motn[i] > 0 and motn[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_donch_break(df, cooldown, gate):
    """Close crosses above Donchian midpoint on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    donch = df["donch_mid"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    for i in range(25, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(donch[i]) or np.isnan(donch[i-1]):
            continue
        if up and close[i] > donch[i] and close[i-1] <= donch[i-1]:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


STANDALONE_GENERATORS = {
    "obv_cross":   _gen_obv_cross,
    "cmf_bull":    _gen_cmf_bull,
    "mfi_bounce":  _gen_mfi_bounce,
    "chop_trend":  _gen_chop_trend,
    "cci_cross":   _gen_cci_cross,
    "ichi_break":  _gen_ichi_break,
    "wpr_bounce":  _gen_wpr_bounce,
    "ddl_bull":    _gen_ddl_bull,
    "motn_bull":   _gen_motn_bull,
    "donch_break": _gen_donch_break,
}


# ==============================================================================
# Workers
# ==============================================================================

def _sweep_confluence_set(set_name):
    """Worker: run one confluence indicator set across thresholds/gates/cooldowns."""
    label = set_name.upper()
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()

    ind_dict = _bool_indicators(df_ltf)
    indicator_names, thresholds = CONFLUENCE_SETS[set_name]

    gates = {}
    for use_g, htf_t, gname in GATE_CONFIGS:
        gates[gname] = _compute_gates(df_ltf, df_htf, use_g, htf_t)

    results = []
    total = len(thresholds) * len(GATE_CONFIGS) * len(COOLDOWNS)
    done = 0

    for thresh in thresholds:
        for _, htf_t, gname in GATE_CONFIGS:
            gate = gates[gname]
            for cd in COOLDOWNS:
                e, x = _gen_confluence(df_ltf, ind_dict, indicator_names, thresh, cd, gate)

                is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
                oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

                is_pf = is_r["pf"]
                oos_pf = oos_r["pf"]
                decay = ((oos_pf - is_pf) / is_pf * 100) \
                        if is_pf > 0 and not math.isinf(is_pf) else float("nan")

                results.append({
                    "part":       "A_confluence",
                    "strategy":   set_name,
                    "indicators": "+".join(indicator_names),
                    "n_ind":      len(indicator_names),
                    "threshold":  thresh,
                    "gates":      gname,
                    "htf_thresh": htf_t,
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
                print(f"  [{label}] {done:>2}/{total} | {thresh}-of-{len(indicator_names)} "
                      f"{gname:<15} cd={cd:>2} | OOS PF={pf_s:>8} T={oos_r['trades']:>4}",
                      flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


def _sweep_standalone(strategy_name):
    """Worker: run one standalone strategy across gates/cooldowns."""
    label = strategy_name.upper()
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()

    gen_fn = STANDALONE_GENERATORS[strategy_name]

    # Also test no_gates for Part B
    all_gate_configs = [
        (False, 0,  "no_gates"),
        (True,  0,  "gates_only"),
        (True,  35, "gates+htf35"),
    ]

    gates = {}
    for use_g, htf_t, gname in all_gate_configs:
        gates[gname] = _compute_gates(df_ltf, df_htf, use_g, htf_t)

    results = []
    total = len(all_gate_configs) * len(COOLDOWNS)
    done = 0

    for _, htf_t, gname in all_gate_configs:
        gate = gates[gname]
        for cd in COOLDOWNS:
            e, x = gen_fn(df_ltf, cd, gate)

            is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
            oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

            is_pf = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay = ((oos_pf - is_pf) / is_pf * 100) \
                    if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "part":       "B_standalone",
                "strategy":   strategy_name,
                "indicators": strategy_name,
                "n_ind":      1,
                "threshold":  1,
                "gates":      gname,
                "htf_thresh": htf_t,
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
            print(f"  [{label}] {done:>2}/{total} | {gname:<15} cd={cd:>2} | "
                  f"OOS PF={pf_s:>8} T={oos_r['trades']:>4}", flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    part_a = [r for r in all_results if r["part"] == "A_confluence"]
    part_b = [r for r in all_results if r["part"] == "B_standalone"]

    # ── Part A: Confluence ──
    print(f"\n{'='*110}")
    print("  PART A: Confluence Deep Dive — Best Config per Indicator Set")
    print(f"{'='*110}")
    print(f"  {'Set':<16} {'Indicators':<55} {'T':>2} {'Gates':<15} {'cd':>3} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6}")
    print(f"  {'-'*108}")

    for sname in CONFLUENCE_SETS:
        subset = [r for r in part_a if r["strategy"] == sname and r["oos_trades"] >= 5]
        if not subset:
            print(f"  {sname:<16} (no viable results)")
            continue
        best = max(subset, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        pf_s = f"{best['oos_pf']:.2f}" if not math.isinf(best['oos_pf']) else "inf"
        inds = best["indicators"][:53]
        print(f"  {sname:<16} {inds:<55} {best['threshold']:>2} {best['gates']:<15} {best['cooldown']:>3} | "
              f"{pf_s:>8} {best['oos_trades']:>4} {best['oos_wr']:>5.1f}%")

    # Confluence top 15
    a_viable = [r for r in part_a if r["oos_trades"] >= 5]
    a_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n  Top 15 Confluence Configs (OOS trades >= 5):")
    print(f"  {'Set':<16} {'T':>2} {'Gates':<15} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
    print(f"  {'-'*90}")
    for r in a_viable[:15]:
        pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
        is_pf_s = f"{r['is_pf']:.2f}" if not math.isinf(r['is_pf']) else "inf"
        print(f"  {r['strategy']:<16} {r['threshold']:>2} {r['gates']:<15} {r['cooldown']:>3} | "
              f"{is_pf_s:>7} {r['is_trades']:>4} | "
              f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    # ── Part B: Standalone ──
    print(f"\n{'='*110}")
    print("  PART B: 10 More Standalone Strategies — Best Config Each")
    print(f"{'='*110}")
    print(f"  {'Strategy':<16} {'Best Gates':<15} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
    print(f"  {'-'*90}")

    for sname in STANDALONE_GENERATORS:
        subset = [r for r in part_b if r["strategy"] == sname and r["oos_trades"] >= 5]
        if not subset:
            print(f"  {sname:<16} (no viable results)")
            continue
        best = max(subset, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        pf_s = f"{best['oos_pf']:.2f}" if not math.isinf(best['oos_pf']) else "inf"
        is_pf_s = f"{best['is_pf']:.2f}" if not math.isinf(best['is_pf']) else "inf"
        print(f"  {sname:<16} {best['gates']:<15} {best['cooldown']:>3} | "
              f"{is_pf_s:>7} {best['is_trades']:>4} | "
              f"{pf_s:>8} {best['oos_trades']:>4} {best['oos_wr']:>5.1f}% {best['oos_net']:>8.2f}")

    # ── Grand top 20 across both parts ──
    all_viable = [r for r in all_results if r["oos_trades"] >= 5]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n{'='*110}")
    print("  GRAND TOP 20 — All Strategies (OOS trades >= 5)")
    print(f"{'='*110}")
    print(f"  {'Part':<4} {'Strategy':<16} {'T':>2} {'Gates':<15} {'cd':>3} | "
          f"{'IS PF':>7} {'T':>4} | {'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
    print(f"  {'-'*95}")
    for r in all_viable[:20]:
        pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
        is_pf_s = f"{r['is_pf']:.2f}" if not math.isinf(r['is_pf']) else "inf"
        part_tag = "A" if r["part"] == "A_confluence" else "B"
        thresh_s = f"{r['threshold']}" if r["part"] == "A_confluence" else "-"
        print(f"  {part_tag:<4} {r['strategy']:<16} {thresh_s:>2} {r['gates']:<15} {r['cooldown']:>3} | "
              f"{is_pf_s:>7} {r['is_trades']:>4} | "
              f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    # Top by trade count
    high_trade = [r for r in all_results if r["oos_pf"] >= 5 and r["oos_trades"] >= 20]
    high_trade.sort(key=lambda r: r["oos_trades"], reverse=True)
    if high_trade:
        print(f"\n  Top 15 by Trade Count (OOS PF >= 5, trades >= 20):")
        print(f"  {'Part':<4} {'Strategy':<16} {'T':>2} {'Gates':<15} {'cd':>3} | "
              f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
        print(f"  {'-'*80}")
        for r in high_trade[:15]:
            pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
            part_tag = "A" if r["part"] == "A_confluence" else "B"
            thresh_s = f"{r['threshold']}" if r["part"] == "A_confluence" else "-"
            print(f"  {part_tag:<4} {r['strategy']:<16} {thresh_s:>2} {r['gates']:<15} {r['cooldown']:>3} | "
                  f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    print(f"\n  --- Reference: BTC003 OOS: PF=49.27, 76t, WR=76.3% ---")
    print(f"  --- Phase 8 best: TRIPLE_CONFIRM gates+htf35 cd=40: PF=34.75, 44t, WR=70.5% ---")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase9_deep_results.json"
    out_path.parent.mkdir(exist_ok=True)

    # Count runs
    a_runs = sum(
        len(thresholds) * len(GATE_CONFIGS) * len(COOLDOWNS)
        for _, (_, thresholds) in CONFLUENCE_SETS.items()
    )
    b_runs = len(STANDALONE_GENERATORS) * 3 * len(COOLDOWNS)  # 3 gate configs
    total = a_runs + b_runs

    print("BTC Phase 9: Confluence Deep Dive + 10 More Strategies (Long Only)")
    print(f"  Part A: {len(CONFLUENCE_SETS)} confluence sets = {a_runs} runs")
    print(f"  Part B: {len(STANDALONE_GENERATORS)} standalone strategies = {b_runs} runs")
    print(f"  Total : {total} runs")
    print(f"  IS    : {IS_START} -> {IS_END}")
    print(f"  OOS   : {OOS_START} -> {OOS_END}")
    print()

    # Build task list: (worker_fn, arg)
    tasks = []
    for sname in CONFLUENCE_SETS:
        tasks.append(("confluence", sname, _sweep_confluence_set))
    for sname in STANDALONE_GENERATORS:
        tasks.append(("standalone", sname, _sweep_standalone))

    all_results = []

    if args.no_parallel:
        for part, sname, fn in tasks:
            all_results.extend(fn(sname))
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for part, sname, fn in tasks:
                futures[pool.submit(fn, sname)] = (part, sname)

            for future in as_completed(futures):
                part, sname = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  [{sname.upper()}] finished -- {len(results)} records")
                except Exception as exc:
                    import traceback
                    print(f"  [{sname.upper()}] FAILED: {exc}")
                    traceback.print_exc()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results -> {out_path}")

    _summarize(all_results)


if __name__ == "__main__":
    main()
