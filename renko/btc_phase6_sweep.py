#!/usr/bin/env python3
"""
btc_phase6_sweep.py -- BTC Phase 6 + Oscillator Confluence Sweep (Long Only)

Tests Phase 6 indicators (CCI, Ichi, WPR, Donch, ESCGO, DDL, MOTN) as entries
and gates on BTC $150 Renko. Also tests multi-oscillator confluence entries.

  Part A — 10 new standalone entry signals (90 combos)
  Part B — Phase 6 gates on BTC007 v3 quartet (24 combos)
  Part C — Multi-oscillator confluence (19 combos)
  Part D — BTC007 v3 + extra entry signals (16 combos)
  Part E — Stacked gate pairs on BTC007 v3 (16 combos)
  Part F — LuxAlgo + Phase 6 combined gates (7 combos)

Baseline: BTC007 v3 (MACD+KAMA+Stoch+RSI50 + PSAR + chop60, cd=3)
  TV OOS: PF=27.18, 182t (1.1/d), WR=68.7%

Usage:
    python renko/btc_phase6_sweep.py
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


# -- Data loading (per-process caching) ----------------------------------------

_w = {}


def _init():
    if "df" in _w:
        return
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators

    df = load_renko_export(LTF_FILE)
    add_renko_indicators(df)
    add_phase6_indicators(df)

    n = len(df)
    brick_up = df["brick_up"].values

    # PSAR gate
    psar = df["psar_dir"].values
    psar_nan = np.isnan(psar)
    psar_ok = psar_nan | (psar > 0)

    # Chop gate
    chop = df["chop"].values

    # Phase 6 gate arrays (all pre-shifted, check at [i] directly)
    ichi = df["ichi_pos"].values
    cci = df["cci"].values
    wpr = df["wpr"].values
    escgo_f = df["escgo_fast"].values
    escgo_s = df["escgo_slow"].values
    ddl = df["ddl_diff"].values
    donch = df["donch_mid"].values
    motn_dx = df["motn_dx"].values
    motn_zx = df["motn_zx"].values
    adx = df["adx"].values
    close = df["Close"].values.astype(float)

    # Standard indicators for BTC007 quartet + oscillator confluence
    rsi = df["rsi"].values
    stoch_k = df["stoch_k"].values
    macd_h = df["macd_hist"].values
    kama_s = df["kama_slope"].values

    _w["df"] = df
    _w["n"] = n
    _w["brick_up"] = brick_up
    _w["psar_ok"] = psar_ok
    _w["chop"] = chop
    _w["ichi"] = ichi
    _w["cci"] = cci
    _w["wpr"] = wpr
    _w["escgo_f"] = escgo_f
    _w["escgo_s"] = escgo_s
    _w["ddl"] = ddl
    _w["donch"] = donch
    _w["motn_dx"] = motn_dx
    _w["motn_zx"] = motn_zx
    _w["adx"] = adx
    _w["close"] = close
    _w["rsi"] = rsi
    _w["stoch_k"] = stoch_k
    _w["macd_h"] = macd_h
    _w["kama_s"] = kama_s


def _init_lux():
    """Lazy load LuxAlgo (only needed for Part F)."""
    if "lux_loaded" in _w:
        return
    from renko.luxalgo_indicators import add_luxalgo_indicators
    df = _w["df"]
    add_luxalgo_indicators(df, include_knn=True, svm_vol_weight=0.0)
    _w["lux_loaded"] = True


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
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ==============================================================================
# Gate builder helpers
# ==============================================================================

def _build_gate(gate_name):
    """Build boolean gate array from gate name."""
    n = _w["n"]
    ones = np.ones(n, dtype=bool)

    if gate_name == "none":
        return ones
    elif gate_name == "psar":
        return _w["psar_ok"]
    elif gate_name == "psar_chop":
        chop = _w["chop"]
        psar = _w["psar_ok"]
        chop_ok = np.isnan(chop) | (chop <= 60)
        return psar & chop_ok
    else:
        return ones


def _build_p6_gate(gate_name):
    """Build Phase 6 indicator gate (checks [i] — already pre-shifted)."""
    n = _w["n"]
    ones = np.ones(n, dtype=bool)

    if gate_name == "ichi_pos":
        v = _w["ichi"]
        return np.isnan(v) | (v > 0)
    elif gate_name == "cci_pos":
        v = _w["cci"]
        return np.isnan(v) | (v > 0)
    elif gate_name == "cci_50":
        v = _w["cci"]
        return np.isnan(v) | (v > 50)
    elif gate_name == "wpr_bull":
        v = _w["wpr"]
        return np.isnan(v) | (v > -50)
    elif gate_name == "wpr_bull30":
        v = _w["wpr"]
        return np.isnan(v) | (v > -30)
    elif gate_name == "escgo_bull":
        f, s = _w["escgo_f"], _w["escgo_s"]
        return (np.isnan(f) | np.isnan(s)) | (f > s)
    elif gate_name == "ddl_bull":
        v = _w["ddl"]
        return np.isnan(v) | (v > 0)
    elif gate_name == "donch_above":
        c, d = _w["close"], _w["donch"]
        return np.isnan(d) | (c > d)
    elif gate_name == "motn_dx_pos":
        v = _w["motn_dx"]
        return np.isnan(v) | (v > 0)
    elif gate_name == "motn_zx_pos":
        v = _w["motn_zx"]
        return np.isnan(v) | (v > 0)
    elif gate_name == "adx_25":
        v = _w["adx"]
        return np.isnan(v) | (v >= 25)
    elif gate_name == "adx_30":
        v = _w["adx"]
        return np.isnan(v) | (v >= 30)
    else:
        return ones


# ==============================================================================
# Part A — 10 New Standalone Entry Signals
# ==============================================================================

def _gen_standalone(signal_name, gate, cooldown):
    """Generate entry/exit for a single Phase 6 indicator signal."""
    n = _w["n"]
    brick_up = _w["brick_up"]
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999

    ichi = _w["ichi"]
    donch = _w["donch"]
    close = _w["close"]
    ddl = _w["ddl"]
    motn_dx = _w["motn_dx"]
    motn_zx = _w["motn_zx"]
    wpr = _w["wpr"]
    cci = _w["cci"]
    escgo_f = _w["escgo_f"]
    escgo_s = _w["escgo_s"]

    for i in range(60, n):
        up = bool(brick_up[i])

        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        fired = False

        if signal_name == "ichi_flip":
            if (not np.isnan(ichi[i]) and not np.isnan(ichi[i-1])
                    and ichi[i] > 0 and ichi[i-1] <= 0):
                fired = True

        elif signal_name == "donch_cross":
            if (not np.isnan(donch[i]) and not np.isnan(donch[i-1])
                    and not np.isnan(close[i]) and not np.isnan(close[i-1])
                    and close[i] > donch[i] and close[i-1] <= donch[i-1]):
                fired = True

        elif signal_name == "ddl_cross":
            if (not np.isnan(ddl[i]) and not np.isnan(ddl[i-1])
                    and ddl[i] > 0 and ddl[i-1] <= 0):
                fired = True

        elif signal_name == "motn_dx_cross":
            if (not np.isnan(motn_dx[i]) and not np.isnan(motn_dx[i-1])
                    and motn_dx[i] > 0 and motn_dx[i-1] <= 0):
                fired = True

        elif signal_name == "motn_zx_cross":
            if (not np.isnan(motn_zx[i]) and not np.isnan(motn_zx[i-1])
                    and motn_zx[i] > 0 and motn_zx[i-1] <= 0):
                fired = True

        elif signal_name == "wpr_cross_70":
            if (not np.isnan(wpr[i]) and not np.isnan(wpr[i-1])
                    and wpr[i] > -70 and wpr[i-1] <= -70):
                fired = True

        elif signal_name == "wpr_cross_80":
            if (not np.isnan(wpr[i]) and not np.isnan(wpr[i-1])
                    and wpr[i] > -80 and wpr[i-1] <= -80):
                fired = True

        elif signal_name == "cci_cross_neg50":
            if (not np.isnan(cci[i]) and not np.isnan(cci[i-1])
                    and cci[i] > -50 and cci[i-1] <= -50):
                fired = True

        elif signal_name == "escgo_ddl":
            # Compound: ESCGO bullish AND DDL bullish on up brick
            if (not np.isnan(escgo_f[i]) and not np.isnan(escgo_s[i])
                    and not np.isnan(ddl[i])
                    and escgo_f[i] > escgo_s[i] and ddl[i] > 0):
                fired = True

        elif signal_name == "motn_ichi":
            # Compound: MOTN DX positive AND Ichi above cloud on up brick
            if (not np.isnan(motn_dx[i]) and not np.isnan(ichi[i])
                    and motn_dx[i] > 0 and ichi[i] > 0):
                fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part B/D/E — BTC007 v3 Quartet with custom gates
# ==============================================================================

def _gen_btc007v3(gate, cooldown, extra_signal=None):
    """BTC007 v3 quartet (MACD flip + KAMA turn + stoch cross 25 + RSI cross 50)
    with PSAR + Chop60 + custom gate. Optionally add extra entry signal."""
    n = _w["n"]
    brick_up = _w["brick_up"]
    macd_h = _w["macd_h"]
    kama_s = _w["kama_s"]
    stoch_k = _w["stoch_k"]
    rsi = _w["rsi"]
    psar_ok = _w["psar_ok"]
    chop = _w["chop"]

    # Extra signal arrays (lazy)
    ichi = _w["ichi"]
    donch = _w["donch"]
    close = _w["close"]
    ddl = _w["ddl"]
    motn_dx = _w["motn_dx"]
    wpr = _w["wpr"]
    cci = _w["cci"]
    escgo_f = _w["escgo_f"]
    escgo_s = _w["escgo_s"]

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

        if not gate[i] or not up or (i - last_bar) < cooldown:
            continue

        # PSAR gate
        if not psar_ok[i]:
            continue

        # Chop gate
        if not np.isnan(chop[i]) and chop[i] > 60:
            continue

        fired = False

        # MACD flip
        if not fired:
            if (not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1])
                    and macd_h[i] > 0 and macd_h[i-1] <= 0):
                fired = True

        # KAMA turn
        if not fired:
            if (not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1])
                    and kama_s[i] > 0 and kama_s[i-1] <= 0):
                fired = True

        # Stoch cross 25
        if not fired:
            if (not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1])
                    and stoch_k[i] > 25 and stoch_k[i-1] <= 25):
                fired = True

        # RSI cross 50
        if not fired:
            if (not np.isnan(rsi[i]) and not np.isnan(rsi[i-1])
                    and rsi[i] > 50 and rsi[i-1] <= 50):
                fired = True

        # Extra signal (Part D)
        if not fired and extra_signal is not None:
            if extra_signal == "ichi_flip":
                if (not np.isnan(ichi[i]) and not np.isnan(ichi[i-1])
                        and ichi[i] > 0 and ichi[i-1] <= 0):
                    fired = True
            elif extra_signal == "donch_cross":
                if (not np.isnan(donch[i]) and not np.isnan(donch[i-1])
                        and not np.isnan(close[i]) and not np.isnan(close[i-1])
                        and close[i] > donch[i] and close[i-1] <= donch[i-1]):
                    fired = True
            elif extra_signal == "ddl_cross":
                if (not np.isnan(ddl[i]) and not np.isnan(ddl[i-1])
                        and ddl[i] > 0 and ddl[i-1] <= 0):
                    fired = True
            elif extra_signal == "motn_dx_cross":
                if (not np.isnan(motn_dx[i]) and not np.isnan(motn_dx[i-1])
                        and motn_dx[i] > 0 and motn_dx[i-1] <= 0):
                    fired = True
            elif extra_signal == "wpr_cross_70":
                if (not np.isnan(wpr[i]) and not np.isnan(wpr[i-1])
                        and wpr[i] > -70 and wpr[i-1] <= -70):
                    fired = True
            elif extra_signal == "cci_cross_neg50":
                if (not np.isnan(cci[i]) and not np.isnan(cci[i-1])
                        and cci[i] > -50 and cci[i-1] <= -50):
                    fired = True
            elif extra_signal == "escgo_cross":
                if (not np.isnan(escgo_f[i]) and not np.isnan(escgo_s[i])
                        and not np.isnan(escgo_f[i-1]) and not np.isnan(escgo_s[i-1])
                        and escgo_f[i] > escgo_s[i] and escgo_f[i-1] <= escgo_s[i-1]):
                    fired = True
            elif extra_signal == "wpr_cross_50":
                if (not np.isnan(wpr[i]) and not np.isnan(wpr[i-1])
                        and wpr[i] > -50 and wpr[i-1] <= -50):
                    fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part C — Multi-Oscillator Confluence
# ==============================================================================

def _count_osc_bullish(i):
    """Count how many of 6 oscillators are in bullish state at bar i."""
    count = 0
    rsi = _w["rsi"]
    stk = _w["stoch_k"]
    cci = _w["cci"]
    wpr = _w["wpr"]
    ef = _w["escgo_f"]
    es = _w["escgo_s"]
    ddl = _w["ddl"]

    if not np.isnan(rsi[i]) and rsi[i] > 50:
        count += 1
    if not np.isnan(stk[i]) and stk[i] > 50:
        count += 1
    if not np.isnan(cci[i]) and cci[i] > 0:
        count += 1
    if not np.isnan(wpr[i]) and wpr[i] > -50:
        count += 1
    if not np.isnan(ef[i]) and not np.isnan(es[i]) and ef[i] > es[i]:
        count += 1
    if not np.isnan(ddl[i]) and ddl[i] > 0:
        count += 1
    return count


def _gen_confluence_btc007(min_osc, cooldown):
    """BTC007 v3 quartet entries, but only fire when >= min_osc oscillators bullish.
    PSAR gate active."""
    n = _w["n"]
    brick_up = _w["brick_up"]
    macd_h = _w["macd_h"]
    kama_s = _w["kama_s"]
    stoch_k = _w["stoch_k"]
    rsi = _w["rsi"]
    psar_ok = _w["psar_ok"]
    chop = _w["chop"]

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

        if not psar_ok[i] or not up or (i - last_bar) < cooldown:
            continue

        if not np.isnan(chop[i]) and chop[i] > 60:
            continue

        # Oscillator confluence check
        if _count_osc_bullish(i) < min_osc:
            continue

        fired = False

        # MACD flip
        if (not np.isnan(macd_h[i]) and not np.isnan(macd_h[i-1])
                and macd_h[i] > 0 and macd_h[i-1] <= 0):
            fired = True
        # KAMA turn
        if not fired:
            if (not np.isnan(kama_s[i]) and not np.isnan(kama_s[i-1])
                    and kama_s[i] > 0 and kama_s[i-1] <= 0):
                fired = True
        # Stoch cross 25
        if not fired:
            if (not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1])
                    and stoch_k[i] > 25 and stoch_k[i-1] <= 25):
                fired = True
        # RSI cross 50
        if not fired:
            if (not np.isnan(rsi[i]) and not np.isnan(rsi[i-1])
                    and rsi[i] > 50 and rsi[i-1] <= 50):
                fired = True

        if fired:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_pure_confluence(min_osc, cooldown):
    """Enter when oscillator count rises to min_osc on up brick. PSAR gate."""
    n = _w["n"]
    brick_up = _w["brick_up"]
    psar_ok = _w["psar_ok"]

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

        if not psar_ok[i] or not up or (i - last_bar) < cooldown:
            continue

        # Count rises to threshold
        cur = _count_osc_bullish(i)
        prev = _count_osc_bullish(i - 1)
        if cur >= min_osc and prev < min_osc:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


# ==============================================================================
# Part F — LuxAlgo + Phase 6 Combined Gates
# ==============================================================================

def _build_lux_gate(gate_name):
    """Build LuxAlgo directional gate array."""
    df = _w["df"]
    n = _w["n"]
    ones = np.ones(n, dtype=bool)

    if gate_name == "svm_trend":
        v = df["lux_svm_trend"].values
        return np.isnan(v) | (v > 0)
    elif gate_name == "knn_trend":
        v = df["lux_knn_bullish"].values.astype(float)
        return np.isnan(v) | (v > 0.5)
    elif gate_name == "inertial_cross":
        k = df["lux_inertial_k"].values
        d = df["lux_inertial_d"].values
        return (np.isnan(k) | np.isnan(d)) | (k > d)
    elif gate_name == "breakout_bias":
        bp = df["lux_breakout_bull"].values
        br = df["lux_breakout_bear"].values
        return (np.isnan(bp) | np.isnan(br)) | (bp > br)
    elif gate_name == "rollseg_trend":
        v = df["lux_rollseg_trend"].values
        return np.isnan(v) | (v > 0)
    else:
        return ones


# ==============================================================================
# Combo builders
# ==============================================================================

STANDALONE_SIGNALS = [
    "ichi_flip", "donch_cross", "ddl_cross", "motn_dx_cross", "motn_zx_cross",
    "wpr_cross_70", "wpr_cross_80", "cci_cross_neg50", "escgo_ddl", "motn_ichi",
]

P6_GATES = [
    "ichi_pos", "cci_pos", "cci_50", "wpr_bull", "wpr_bull30", "escgo_bull",
    "ddl_bull", "donch_above", "motn_dx_pos", "motn_zx_pos", "adx_25", "adx_30",
]

EXTRA_SIGNALS = [
    "ichi_flip", "donch_cross", "ddl_cross", "motn_dx_cross",
    "wpr_cross_70", "cci_cross_neg50", "escgo_cross", "wpr_cross_50",
]

GATE_PAIRS = [
    ("ichi_pos", "escgo_bull"),
    ("ichi_pos", "cci_pos"),
    ("escgo_bull", "ddl_bull"),
    ("cci_pos", "wpr_bull"),
    ("adx_25", "ichi_pos"),
    ("adx_25", "escgo_bull"),
    ("ddl_bull", "motn_dx_pos"),
    ("wpr_bull", "donch_above"),
]

LUX_P6_COMBOS = [
    ("svm_trend", "ichi_pos"),
    ("svm_trend", "escgo_bull"),
    ("knn_trend", "ichi_pos"),
    ("knn_trend", "escgo_bull"),
    ("inertial_cross", "cci_pos"),
    ("breakout_bias", "wpr_bull"),
    ("rollseg_trend", "ddl_bull"),
]


def _build_combos():
    combos = []

    # Part A — 10 signals × 3 cd × 3 gates = 90
    for sig in STANDALONE_SIGNALS:
        for cd in [3, 5, 10]:
            for gate in ["none", "psar", "psar_chop"]:
                combos.append({
                    "part": "A", "signal": sig, "cooldown": cd, "gate": gate,
                    "label": f"{sig}_cd{cd}_{gate}",
                })

    # Part B — 12 gates × 2 cd = 24
    for pg in P6_GATES:
        for cd in [3, 5]:
            combos.append({
                "part": "B", "p6_gate": pg, "cooldown": cd,
                "label": f"btc007v3+{pg}_cd{cd}",
            })

    # Part C1 — BTC007 + oscillator confluence: N=[2,3,4,5,6] × cd=[3,5] = 10
    for min_osc in [2, 3, 4, 5, 6]:
        for cd in [3, 5]:
            combos.append({
                "part": "C1", "min_osc": min_osc, "cooldown": cd,
                "label": f"btc007v3_osc{min_osc}_cd{cd}",
            })

    # Part C2 — Pure confluence: N=[3,4,5] × cd=[3,5,10] = 9
    for min_osc in [3, 4, 5]:
        for cd in [3, 5, 10]:
            combos.append({
                "part": "C2", "min_osc": min_osc, "cooldown": cd,
                "label": f"pure_osc{min_osc}_cd{cd}",
            })

    # Part D — BTC007 v3 + extra signal: 8 × cd=[3,5] = 16
    for sig in EXTRA_SIGNALS:
        for cd in [3, 5]:
            combos.append({
                "part": "D", "extra_signal": sig, "cooldown": cd,
                "label": f"btc007v3+{sig}_cd{cd}",
            })

    # Part E — Stacked gate pairs: 8 × cd=[3,5] = 16
    for g1, g2 in GATE_PAIRS:
        for cd in [3, 5]:
            combos.append({
                "part": "E", "gate_pair": (g1, g2), "cooldown": cd,
                "label": f"btc007v3+{g1}+{g2}_cd{cd}",
            })

    # Part F — LuxAlgo + Phase 6 combined: 7 × cd=3 = 7
    for lux_g, p6_g in LUX_P6_COMBOS:
        combos.append({
            "part": "F", "lux_gate": lux_g, "p6_gate": p6_g, "cooldown": 3,
            "label": f"btc007v3+{lux_g}+{p6_g}_cd3",
        })

    return combos


# ==============================================================================
# Worker
# ==============================================================================

def _run_one(combo):
    _init()
    n = _w["n"]
    ones = np.ones(n, dtype=bool)

    part = combo["part"]
    cd = combo["cooldown"]

    if part == "A":
        gate = _build_gate(combo["gate"])
        ent, ext = _gen_standalone(combo["signal"], gate, cd)

    elif part == "B":
        p6_gate = _build_p6_gate(combo["p6_gate"])
        ent, ext = _gen_btc007v3(p6_gate, cd)

    elif part == "C1":
        ent, ext = _gen_confluence_btc007(combo["min_osc"], cd)

    elif part == "C2":
        ent, ext = _gen_pure_confluence(combo["min_osc"], cd)

    elif part == "D":
        ent, ext = _gen_btc007v3(ones, cd, extra_signal=combo["extra_signal"])

    elif part == "E":
        g1, g2 = combo["gate_pair"]
        pg1 = _build_p6_gate(g1)
        pg2 = _build_p6_gate(g2)
        combined = pg1 & pg2
        ent, ext = _gen_btc007v3(combined, cd)

    elif part == "F":
        _init_lux()
        lux_gate = _build_lux_gate(combo["lux_gate"])
        p6_gate = _build_p6_gate(combo["p6_gate"])
        combined = lux_gate & p6_gate
        ent, ext = _gen_btc007v3(combined, cd)

    is_r = _run_bt(ent, ext, IS_START, IS_END)
    oos_r = _run_bt(ent, ext, OOS_START, OOS_END)
    return combo, is_r, oos_r


# ==============================================================================
# Reporting
# ==============================================================================

def _print_header():
    print(f"  {'#':>3} {'Pt':>2} {'Label':<48} | "
          f"{'IS PF':>7} {'IS T':>5} {'IS WR':>6} | "
          f"{'OOS PF':>7} {'OOS T':>5} {'t/d':>5} {'OOS WR':>6} "
          f"{'Net':>10} {'DD%':>8}")
    print("  " + "-" * 120)


def _print_row(r, rank):
    is_pf = f"{r['is_pf']:.2f}" if r['is_pf'] < 9999 else "inf"
    oos_pf = f"{r['oos_pf']:.2f}" if r['oos_pf'] < 9999 else "inf"
    tpd = r['oos_trades'] / OOS_DAYS if OOS_DAYS > 0 else 0
    print(f"  {rank:>3} {r['part']:>2} {r['label']:<48} | "
          f"{is_pf:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
          f"{oos_pf:>7} {r['oos_trades']:>5} {tpd:>4.1f} {r['oos_wr']:>5.1f}% "
          f"{r['oos_net']:>10.2f} {r['oos_dd']:>7.3f}%")


def _show_part(results, part, title, n=15):
    subset = [r for r in results if r["part"] == part
              and r["oos_trades"] >= 5 and r["oos_net"] > 0]
    subset.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'='*65}")
    print(f"  {title} ({len(subset)} viable / {sum(1 for r in results if r['part']==part)} total)")
    print(f"{'='*65}")
    if subset:
        _print_header()
        for i, r in enumerate(subset[:n]):
            _print_row(r, i + 1)


# ==============================================================================
# Main
# ==============================================================================

def main():
    combos = _build_combos()
    total = len(combos)
    print(f"BTC Phase 6 Sweep: {total} combos ({total*2} backtests) on {MAX_WORKERS} workers")

    all_results = []
    done = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, c): c for c in combos}
        for fut in as_completed(futures):
            combo = futures[fut]
            try:
                combo_ret, is_r, oos_r = fut.result()
                row = {
                    "part":       combo_ret["part"],
                    "label":      combo_ret["label"],
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
                for k in combo_ret:
                    if k not in row:
                        row[k] = combo_ret[k]
                all_results.append(row)
            except Exception as e:
                print(f"  ERROR: {combo.get('label', '?')}: {e}")
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}]")

    # Show per-part results
    _show_part(all_results, "A", "Part A — New Standalone Entry Signals")
    _show_part(all_results, "B", "Part B — Phase 6 Gates on BTC007 v3")
    _show_part(all_results, "C1", "Part C1 — BTC007 + Oscillator Confluence")
    _show_part(all_results, "C2", "Part C2 — Pure Oscillator Confluence")
    _show_part(all_results, "D", "Part D — BTC007 v3 + Extra Entry Signals")
    _show_part(all_results, "E", "Part E — Stacked Gate Pairs on BTC007 v3")
    _show_part(all_results, "F", "Part F — LuxAlgo + Phase 6 Combined Gates")

    # Global top 20
    viable = [r for r in all_results if r["oos_trades"] >= 5 and r["oos_net"] > 0]
    viable.sort(key=lambda r: (r["oos_wr"], r["oos_net"]), reverse=True)
    print(f"\n{'='*65}")
    print(f"  GLOBAL TOP 20 ({len(viable)} viable / {len(all_results)} total)")
    print(f"{'='*65}")
    if viable:
        _print_header()
        for i, r in enumerate(viable[:20]):
            _print_row(r, i + 1)

    # Combined IS+OOS top 15 + decay
    for r in all_results:
        r["total_trades"] = r["is_trades"] + r["oos_trades"]
        r["total_net"] = r["is_net"] + r["oos_net"]
        if r["total_trades"] > 0:
            is_w = r["is_trades"] * r["is_wr"] / 100
            oos_w = r["oos_trades"] * r["oos_wr"] / 100
            r["total_wr"] = (is_w + oos_w) / r["total_trades"] * 100
        else:
            r["total_wr"] = 0

    viable2 = [r for r in all_results if r["oos_trades"] >= 5 and r["oos_net"] > 0]
    viable2.sort(key=lambda r: (r["total_wr"], r["total_net"]), reverse=True)
    print(f"\n{'='*130}")
    print(f"  COMBINED IS+OOS TOP 15 + DECAY")
    print(f"{'='*130}")
    hdr = (f"  {'#':>3} {'Pt':>2} {'Label':<48} | "
           f"{'IS PF':>7} {'IS T':>5} {'IS WR':>6} | "
           f"{'OOS PF':>7} {'OOS T':>5} {'OOS WR':>6} {'WR chg':>7} | "
           f"{'ALL T':>6} {'ALL WR':>6} {'ALL Net':>10}")
    print(hdr)
    print("  " + "-" * 126)
    for i, r in enumerate(viable2[:15]):
        is_pf = f"{r['is_pf']:.2f}" if r['is_pf'] < 9999 else "inf"
        oos_pf = f"{r['oos_pf']:.2f}" if r['oos_pf'] < 9999 else "inf"
        wr_chg = r['oos_wr'] - r['is_wr']
        print(f"  {i+1:>3} {r['part']:>2} {r['label']:<48} | "
              f"{is_pf:>7} {r['is_trades']:>5} {r['is_wr']:>5.1f}% | "
              f"{oos_pf:>7} {r['oos_trades']:>5} {r['oos_wr']:>5.1f}% {wr_chg:>+6.1f}% | "
              f"{r['total_trades']:>6} {r['total_wr']:>5.1f}% {r['total_net']:>10.2f}")

    # Save JSON
    out_path = ROOT / "ai_context" / "btc_phase6_sweep_results.json"
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items()
              if k not in ("total_trades", "total_net", "total_wr", "gate_pair")}
        for k in ("is_pf", "oos_pf"):
            if isinstance(sr.get(k), float) and math.isinf(sr[k]):
                sr[k] = "inf"
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(all_results)} results -> {out_path}")


if __name__ == "__main__":
    main()
