#!/usr/bin/env python3
"""
btc_hf_phase4_sweep.py -- BTC HF Phase 4: New Entries + HTF Gating + Exit Optimization (Long Only)

Four sweeps in one script:

  Part A — New Phase 6 entry signals (untested on BTC HF):
    CCI_CROSS    CCI crosses from <-100 to >=-100 on up brick
    WPR_CROSS    WPR crosses from <-80 to >=-80 on up brick
    ESCGO_CROSS  ESCGO fast crosses above slow on up brick
    ICHI_BREAK   ichi_pos flips to +1 on up brick
    DONCH_BREAK  Close crosses above donch_mid on up brick
    DDL_CROSS    ddl_diff crosses from <=0 to >0 on up brick

  Part B — HTF $300 ADX gating on best entries from Phase 1-3:
    Top entries: R001(n=2), indicator_trio (ST+MACD+KAMA), stoch_cross
    HTF ADX thresholds: 25, 30, 35, 40

  Part C — Exit optimization on best entries:
    exit_1down:    first down brick (baseline)
    exit_2down:    2 consecutive down bricks
    exit_st_flip:  Supertrend flips bearish
    exit_psar_flip: PSAR flips bearish

  Part D — Best-of-breed with per-signal cooldowns:
    Stack highest-WR individual signals with independent cooldowns

Usage:
    python renko/btc_hf_phase4_sweep.py
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

# -- Data loading ---------------------------------------------------------------

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


def _align_htf_gate(df_ltf, df_htf, htf_gate_arr):
    """Backward-fill HTF gate onto LTF timestamps via merge_asof."""
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
# Part A — New Phase 6 entry signals
# =============================================================================

def _gen_cci_cross(df, cooldown, gate, cci_thresh=-100):
    """CCI crosses from below threshold to above threshold on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    cci = df["cci"].values
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
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(cci[i]) or np.isnan(cci[i-1]):
            continue
        if up and cci[i] >= cci_thresh and cci[i-1] < cci_thresh:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_wpr_cross(df, cooldown, gate, wpr_thresh=-80):
    """WPR crosses from below threshold to above on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    wpr = df["wpr"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 20

    for i in range(warmup, n):
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
        if up and wpr[i] >= wpr_thresh and wpr[i-1] < wpr_thresh:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_escgo_cross(df, cooldown, gate):
    """ESCGO fast crosses above ESCGO slow on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    fast = df["escgo_fast"].values
    slow = df["escgo_slow"].values
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
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(fast[i]) or np.isnan(slow[i]) or np.isnan(fast[i-1]) or np.isnan(slow[i-1]):
            continue
        if up and fast[i] > slow[i] and fast[i-1] <= slow[i-1]:
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
        if np.isnan(ichi[i]) or np.isnan(ichi[i-1]):
            continue
        if up and ichi[i] > 0 and ichi[i-1] <= 0:
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
    warmup = 30

    for i in range(warmup, n):
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


def _gen_ddl_cross(df, cooldown, gate):
    """DDL diff crosses from <=0 to >0 on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
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
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(ddl[i]) or np.isnan(ddl[i-1]):
            continue
        if up and ddl[i] > 0 and ddl[i-1] <= 0:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


# =============================================================================
# Part B — HTF ADX gating on top entries
# =============================================================================

def _gen_r001_psar(df, cooldown, gate, n_bricks=2):
    """R001: N consecutive up bricks + PSAR gate."""
    n = len(df)
    brick_up = df["brick_up"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = max(n_bricks + 5, 20)

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        # Check N consecutive up bricks
        all_up = True
        for j in range(n_bricks):
            if not brick_up[i - j]:
                all_up = False
                break
        if all_up:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_indicator_trio(df, cooldown, gate):
    """ST flip + MACD flip + KAMA turn — any fires long."""
    n = len(df)
    brick_up = df["brick_up"].values
    st_dir = df["st_dir"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 35

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        fired = False
        # ST flip
        if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
            if st_dir[i] > 0 and st_dir[i-1] <= 0:
                fired = True
        # MACD flip
        if not fired and not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
            if macd_h[i] > 0 and macd_h[i-1] <= 0:
                fired = True
        # KAMA turn
        if not fired and not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
            if kama_s[i] > 0 and kama_s[i-1] <= 0:
                fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_stoch_cross(df, cooldown, gate, stoch_thresh=25):
    """Stoch K crosses up from below threshold on up brick."""
    n = len(df)
    brick_up = df["brick_up"].values
    stoch_k = df["stoch_k"].values
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 20

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(stoch_k[i]) or np.isnan(stoch_k[i-1]):
            continue
        if up and stoch_k[i] > stoch_thresh and stoch_k[i-1] <= stoch_thresh:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


# =============================================================================
# Part C — Exit optimization
# =============================================================================

def _gen_with_exit(df, cooldown, gate, signal_type, exit_type, **kwargs):
    """
    Generate entry signals with various exit strategies.

    signal_type: "r001", "indicator_trio", "stoch_cross"
    exit_type:   "1down", "2down", "st_flip", "psar_flip"
    """
    n = len(df)
    brick_up = df["brick_up"].values
    st_dir = df["st_dir"].values
    psar_dir = df["psar_dir"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values
    stoch_k = df["stoch_k"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    down_count = 0
    warmup = 40
    stoch_thresh = kwargs.get("stoch_thresh", 25)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # -- Exit logic --
        if in_pos:
            should_exit = False
            if exit_type == "1down":
                should_exit = not up
            elif exit_type == "2down":
                if not up:
                    down_count += 1
                    if down_count >= 2:
                        should_exit = True
                else:
                    down_count = 0
            elif exit_type == "st_flip":
                # Exit when ST flips bearish (or first down brick as safety)
                if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                    if st_dir[i] < 0 and st_dir[i-1] >= 0:
                        should_exit = True
                # Safety: also exit on sustained down moves (3+ down)
                if not up:
                    down_count += 1
                    if down_count >= 3:
                        should_exit = True
                else:
                    down_count = 0
            elif exit_type == "psar_flip":
                if not np.isnan(psar_dir[i]):
                    if psar_dir[i] < 0:
                        should_exit = True
                if not up:
                    down_count += 1
                    if down_count >= 3:
                        should_exit = True
                else:
                    down_count = 0

            if should_exit:
                exit_[i] = True
                in_pos = False
                down_count = 0
            continue

        # -- Entry logic --
        if not gate[i] or (i - last_bar) < cooldown:
            continue

        fired = False

        if signal_type == "r001":
            if up and brick_up[i-1]:
                fired = True
        elif signal_type == "indicator_trio":
            if not up:
                continue
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True
            if not fired and not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True
            if not fired and not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True
        elif signal_type == "stoch_cross":
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if up and stoch_k[i] > stoch_thresh and stoch_k[i-1] <= stoch_thresh:
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i
            down_count = 0

    return entry, exit_


# =============================================================================
# Part D — Best-of-breed with per-signal cooldowns
# =============================================================================

def _gen_best_of_breed(df, gate, signals, cd_per_signal):
    """
    Stack multiple independent signals, each with its own cooldown.

    signals: list of signal names to include
    cd_per_signal: dict of signal_name -> cooldown
    """
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

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    warmup = 60

    # Per-signal cooldown trackers
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
        if "r001" in signals and (i - last_bar["r001"]) >= cd_per_signal.get("r001", 5):
            if brick_up[i-1]:
                fired = True
                last_bar["r001"] = i

        # ST flip
        if not fired and "st_flip" in signals and (i - last_bar["st_flip"]) >= cd_per_signal.get("st_flip", 3):
            if not np.isnan(st_dir[i]) and not np.isnan(st_dir[i-1]):
                if st_dir[i] > 0 and st_dir[i-1] <= 0:
                    fired = True
                    last_bar["st_flip"] = i

        # MACD flip
        if not fired and "macd_flip" in signals and (i - last_bar["macd_flip"]) >= cd_per_signal.get("macd_flip", 3):
            if not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1]):
                if macd_h[i] > 0 and macd_h[i-1] <= 0:
                    fired = True
                    last_bar["macd_flip"] = i

        # KAMA turn
        if not fired and "kama_turn" in signals and (i - last_bar["kama_turn"]) >= cd_per_signal.get("kama_turn", 3):
            if not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1]):
                if kama_s[i] > 0 and kama_s[i-1] <= 0:
                    fired = True
                    last_bar["kama_turn"] = i

        # Stoch cross
        if not fired and "stoch_cross" in signals and (i - last_bar["stoch_cross"]) >= cd_per_signal.get("stoch_cross", 5):
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if stoch_k[i] > 25 and stoch_k[i-1] <= 25:
                    fired = True
                    last_bar["stoch_cross"] = i

        # Band bounce
        if not fired and "bb_bounce" in signals and (i - last_bar["bb_bounce"]) >= cd_per_signal.get("bb_bounce", 3):
            if not np.isnan(bb_pct[i]):
                if bb_pct[i] <= 0.20:
                    fired = True
                    last_bar["bb_bounce"] = i

        # CCI cross
        if not fired and "cci_cross" in signals and (i - last_bar["cci_cross"]) >= cd_per_signal.get("cci_cross", 3):
            if not np.isnan(cci[i]) and not np.isnan(cci[i-1]):
                if cci[i] >= -100 and cci[i-1] < -100:
                    fired = True
                    last_bar["cci_cross"] = i

        # WPR cross
        if not fired and "wpr_cross" in signals and (i - last_bar["wpr_cross"]) >= cd_per_signal.get("wpr_cross", 3):
            if not np.isnan(wpr[i]) and not np.isnan(wpr[i-1]):
                if wpr[i] >= -80 and wpr[i-1] < -80:
                    fired = True
                    last_bar["wpr_cross"] = i

        # ESCGO cross
        if not fired and "escgo_cross" in signals and (i - last_bar["escgo_cross"]) >= cd_per_signal.get("escgo_cross", 3):
            if not np.isnan(escgo_f[i]) and not np.isnan(escgo_s[i]):
                if not np.isnan(escgo_f[i-1]) and not np.isnan(escgo_s[i-1]):
                    if escgo_f[i] > escgo_s[i] and escgo_f[i-1] <= escgo_s[i-1]:
                        fired = True
                        last_bar["escgo_cross"] = i

        # R002: reversal (N down then up)
        if not fired and "r002" in signals and (i - last_bar.get("r002", -999_999)) >= cd_per_signal.get("r002", 0):
            if brick_up[i] and i >= 3:
                if not brick_up[i-1] and not brick_up[i-2]:
                    fired = True
                    last_bar["r002"] = i

        if fired:
            entry[i] = True
            in_pos = True

    return entry, exit_


# =============================================================================
# Combo builders
# =============================================================================

COOLDOWNS = [3, 5, 8, 12]
GATE_MODES = ["none", "psar", "psar_adx20"]

def _build_part_a():
    """New Phase 6 entry signals."""
    combos = []
    for sig in ["cci_cross", "wpr_cross", "escgo_cross", "ichi_break", "donch_break", "ddl_cross"]:
        for gm in GATE_MODES:
            for cd in COOLDOWNS:
                c = {"part": "A", "signal": sig, "gate_mode": gm, "cooldown": cd}
                # Extra params for CCI / WPR threshold variants
                if sig == "cci_cross":
                    for t in [-100, -80]:
                        combos.append({**c, "thresh": t, "label": f"{sig}_t{t}"})
                elif sig == "wpr_cross":
                    for t in [-80, -70]:
                        combos.append({**c, "thresh": t, "label": f"{sig}_t{t}"})
                else:
                    combos.append({**c, "thresh": 0, "label": sig})
    return combos


def _build_part_b():
    """HTF ADX gating on top entries."""
    combos = []
    for sig in ["r001", "indicator_trio", "stoch_cross"]:
        for htf_thresh in [25, 30, 35, 40]:
            for cd in [3, 5, 8]:
                combos.append({
                    "part": "B", "signal": sig, "gate_mode": "psar",
                    "cooldown": cd, "htf_adx_thresh": htf_thresh,
                    "label": f"{sig}_htf{htf_thresh}",
                })
    return combos


def _build_part_c():
    """Exit optimization on best entries."""
    combos = []
    for sig in ["r001", "indicator_trio", "stoch_cross"]:
        for exit_type in ["1down", "2down", "st_flip", "psar_flip"]:
            for cd in [3, 5]:
                combos.append({
                    "part": "C", "signal": sig, "gate_mode": "psar",
                    "cooldown": cd, "exit_type": exit_type,
                    "label": f"{sig}_{exit_type}",
                })
    return combos


def _build_part_d():
    """Best-of-breed combos with per-signal cooldowns."""
    combos = []
    # Combo sets to try
    signal_sets = {
        "trio_r001":     ["st_flip", "macd_flip", "kama_turn", "r001"],
        "trio_stoch":    ["st_flip", "macd_flip", "kama_turn", "stoch_cross"],
        "trio_r001_r002": ["st_flip", "macd_flip", "kama_turn", "r001", "r002"],
        "mr_trio":       ["stoch_cross", "bb_bounce", "cci_cross", "wpr_cross"],
        "full_6":        ["st_flip", "macd_flip", "kama_turn", "stoch_cross", "bb_bounce", "r001"],
        "full_8":        ["st_flip", "macd_flip", "kama_turn", "stoch_cross", "bb_bounce", "cci_cross", "wpr_cross", "r001"],
        "escgo_trio":    ["st_flip", "macd_flip", "escgo_cross", "r001"],
        "quality_4":     ["st_flip", "stoch_cross", "cci_cross", "wpr_cross"],
    }

    # Cooldown configs for per-signal
    cd_configs = {
        "fast_all":   {s: 3 for s in ["st_flip", "macd_flip", "kama_turn", "stoch_cross",
                       "bb_bounce", "cci_cross", "wpr_cross", "r001", "r002", "escgo_cross"]},
        "trend5_mr3": {s: 3 for s in ["stoch_cross", "bb_bounce", "cci_cross", "wpr_cross",
                       "escgo_cross", "r002"]},
        "mixed":      {},
    }
    cd_configs["trend5_mr3"].update({s: 5 for s in ["st_flip", "macd_flip", "kama_turn", "r001"]})
    cd_configs["mixed"] = {s: 5 for s in ["r001"]}
    cd_configs["mixed"].update({s: 3 for s in ["st_flip", "macd_flip", "kama_turn",
                               "stoch_cross", "bb_bounce", "cci_cross", "wpr_cross", "r002", "escgo_cross"]})

    for set_name, sigs in signal_sets.items():
        for cd_name, cd_map in cd_configs.items():
            for gm in ["psar", "none"]:
                combos.append({
                    "part": "D", "signal": set_name, "gate_mode": gm,
                    "cooldown": 0, "cd_config_name": cd_name,
                    "signals": sigs,
                    "cd_per_signal": {s: cd_map.get(s, 3) for s in sigs},
                    "label": f"{set_name}_{cd_name}",
                })
    return combos


# =============================================================================
# Worker
# =============================================================================

_w = {}

def _init_worker():
    if "df" not in _w:
        _w["df"] = _load_ltf_data()
        _w["df_htf"] = _load_htf_data()
        _w["gates"] = {gm: _compute_gate(_w["df"], gm) for gm in GATE_MODES}
        # Pre-compute HTF ADX gates
        _w["htf_adx_gates"] = {}
        df_htf = _w["df_htf"]
        adx_htf = df_htf["adx"].values
        adx_nan = np.isnan(adx_htf)
        for thresh in [25, 30, 35, 40]:
            htf_gate_arr = adx_nan | (adx_htf >= thresh)
            _w["htf_adx_gates"][thresh] = _align_htf_gate(_w["df"], df_htf, htf_gate_arr)


def _run_one(combo):
    _init_worker()
    df = _w["df"]
    part = combo["part"]

    if part == "A":
        gate = _w["gates"][combo["gate_mode"]]
        sig = combo["signal"]
        cd = combo["cooldown"]
        t = combo.get("thresh", 0)

        if sig == "cci_cross":
            entry, exit_ = _gen_cci_cross(df, cd, gate, cci_thresh=t)
        elif sig == "wpr_cross":
            entry, exit_ = _gen_wpr_cross(df, cd, gate, wpr_thresh=t)
        elif sig == "escgo_cross":
            entry, exit_ = _gen_escgo_cross(df, cd, gate)
        elif sig == "ichi_break":
            entry, exit_ = _gen_ichi_break(df, cd, gate)
        elif sig == "donch_break":
            entry, exit_ = _gen_donch_break(df, cd, gate)
        elif sig == "ddl_cross":
            entry, exit_ = _gen_ddl_cross(df, cd, gate)
        else:
            return combo, {"pf": 0}, {"pf": 0}

    elif part == "B":
        # Combine PSAR gate with HTF ADX gate
        psar_gate = _w["gates"]["psar"]
        htf_gate = _w["htf_adx_gates"][combo["htf_adx_thresh"]]
        gate = psar_gate & htf_gate
        cd = combo["cooldown"]
        sig = combo["signal"]

        if sig == "r001":
            entry, exit_ = _gen_r001_psar(df, cd, gate)
        elif sig == "indicator_trio":
            entry, exit_ = _gen_indicator_trio(df, cd, gate)
        elif sig == "stoch_cross":
            entry, exit_ = _gen_stoch_cross(df, cd, gate)
        else:
            return combo, {"pf": 0}, {"pf": 0}

    elif part == "C":
        gate = _w["gates"]["psar"]
        entry, exit_ = _gen_with_exit(
            df, combo["cooldown"], gate,
            signal_type=combo["signal"],
            exit_type=combo["exit_type"],
        )

    elif part == "D":
        gate = _w["gates"][combo["gate_mode"]]
        entry, exit_ = _gen_best_of_breed(
            df, gate,
            signals=combo["signals"],
            cd_per_signal=combo["cd_per_signal"],
        )

    else:
        return combo, {"pf": 0}, {"pf": 0}

    is_r = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_r = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    return combo, is_r, oos_r


# =============================================================================
# Summary
# =============================================================================

def _print_header():
    print(f"  {'#':>3} {'Part':>4} {'Label':<30} {'Gate':<12} {'CD':>3} {'Extra':<12} | "
          f"{'IS PF':>7} {'T':>5} {'WR%':>6} | "
          f"{'OOS PF':>8} {'T':>5} {'t/d':>5} {'WR%':>6} {'Net':>9} {'DD%':>7}")
    print(f"  {'-'*140}")


def _print_row(r, rank=None):
    pf_i = "INF" if math.isinf(r["is_pf"]) else f"{r['is_pf']:.2f}"
    pf_o = "INF" if math.isinf(r["oos_pf"]) else f"{r['oos_pf']:.2f}"
    tpd = r["oos_trades"] / OOS_DAYS if r["oos_trades"] > 0 else 0
    prefix = f"  {rank:>3}" if rank else "  "
    extra = r.get("extra", "")
    print(f"{prefix} {r['part']:>4} {r['label']:<30} {r['gate_mode']:<12} {r.get('cooldown', 0):>3} {extra:<12} | "
          f"{pf_i:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{pf_o:>8} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>9.2f} {r['oos_dd']:>6.2f}%")


def _summarize(all_results):
    for part_name, part_title in [
        ("A", "Part A — New Phase 6 Entry Signals"),
        ("B", "Part B — HTF $300 ADX Gating"),
        ("C", "Part C — Exit Optimization"),
        ("D", "Part D — Best-of-Breed Combos"),
    ]:
        subset = [r for r in all_results if r["part"] == part_name]
        if not subset:
            continue

        # Filter to viable
        viable = [r for r in subset if r["oos_trades"] >= 10 and r["oos_net"] > 0]
        viable.sort(key=lambda r: (
            r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6,
            r["oos_net"],
        ), reverse=True)

        print(f"\n{'='*150}")
        print(f"  {part_title} — {len(viable)} viable / {len(subset)} total")
        print(f"{'='*150}")
        _print_header()
        for i, r in enumerate(viable[:25]):
            _print_row(r, rank=i+1)

        # Also show HF subset (1+ trade/day)
        hf = [r for r in subset if r["oos_trades"] >= OOS_DAYS and r["oos_net"] > 0]
        hf.sort(key=lambda r: r["oos_net"], reverse=True)
        if hf:
            print(f"\n  HF subset (>= 1/day, net > 0): {len(hf)} configs — sorted by net")
            _print_header()
            for i, r in enumerate(hf[:15]):
                _print_row(r, rank=i+1)

    # Global top 20 across all parts
    viable_all = [r for r in all_results if r["oos_trades"] >= 50 and r["oos_net"] > 0]
    viable_all.sort(key=lambda r: r["oos_net"], reverse=True)
    print(f"\n{'='*150}")
    print(f"  GLOBAL TOP 20 by OOS Net (T>=50, net>0)")
    print(f"{'='*150}")
    _print_header()
    for i, r in enumerate(viable_all[:20]):
        _print_row(r, rank=i+1)

    # Global best WR (T >= 100)
    wr_all = [r for r in all_results if r["oos_trades"] >= 100 and r["oos_net"] > 0]
    wr_all.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'='*150}")
    print(f"  GLOBAL BEST WR (T>=100, net>0)")
    print(f"{'='*150}")
    _print_header()
    for i, r in enumerate(wr_all[:20]):
        _print_row(r, rank=i+1)


# =============================================================================
# Main
# =============================================================================

def main():
    combos_a = _build_part_a()
    combos_b = _build_part_b()
    combos_c = _build_part_c()
    combos_d = _build_part_d()
    all_combos = combos_a + combos_b + combos_c + combos_d

    total = len(all_combos)

    print(f"\n{'='*70}")
    print(f"BTC HF Phase 4 — New Entries + HTF + Exits + Best-of-Breed")
    print(f"  Part A (new P6 entries):   {len(combos_a)} combos")
    print(f"  Part B (HTF ADX gating):   {len(combos_b)} combos")
    print(f"  Part C (exit optimization):{len(combos_c)} combos")
    print(f"  Part D (best-of-breed):    {len(combos_d)} combos")
    print(f"  Total runs: {total} ({total*2} backtests)")
    print(f"  Workers:    {MAX_WORKERS}")
    print(f"  IS:  {IS_START} -> {IS_END}")
    print(f"  OOS: {OOS_START} -> {OOS_END}")
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
                    "signal":     combo.get("signal", ""),
                    "gate_mode":  combo.get("gate_mode", ""),
                    "cooldown":   combo.get("cooldown", 0),
                    "extra":      "",
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
                # Add extra info based on part
                if combo["part"] == "A":
                    row["extra"] = f"t={combo.get('thresh', 0)}"
                elif combo["part"] == "B":
                    row["extra"] = f"htf>={combo['htf_adx_thresh']}"
                elif combo["part"] == "C":
                    row["extra"] = combo["exit_type"]
                elif combo["part"] == "D":
                    row["extra"] = combo.get("cd_config_name", "")
                results.append(row)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done:>4}/{total}]", flush=True)

    # Save
    out_path = ROOT / "ai_context" / "btc_hf_phase4_results.json"
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
