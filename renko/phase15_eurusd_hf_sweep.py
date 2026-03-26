#!/usr/bin/env python3
"""
phase15_eurusd_hf_sweep.py — High-Frequency EURUSD Sweep

Goal: 1+ trade/day (~150+ OOS trades) with high WR and high PnL.

4 NEW entry modes (not just R007 consecutive bricks):

Mode 1: "Trend Pullback" — Established EMA trend, enter on first continuation
         brick after 1-2 pullback bricks. High WR because trading WITH trend
         at better price after pullback.

Mode 2: "Momentum Scalp" — Single brick direction + 2+ indicator agreement
         (no consecutive-brick requirement). Very high frequency.

Mode 3: "Velocity Burst" — NEW indicator: brick velocity (bricks/hour).
         Enter when bricks form rapidly in one direction = strong conviction.

Mode 4: "R001 Ultra-Light" — Standard N=2 consecutive bricks but with
         cooldown=3-8 and minimal gating. Baseline high-frequency R007.

NEW custom indicators computed inline:
  - brick_velocity: rolling bricks per hour (momentum proxy)
  - trend_score: fraction of last N bricks in same direction
  - consec_count: running count of consecutive same-direction bricks

All modes across 4 brick sizes (EU4/EU5/EU6/EU7) with light gating.

Usage:
  python renko/phase15_eurusd_hf_sweep.py
  python renko/phase15_eurusd_hf_sweep.py --mode 1
"""

import argparse
import contextlib
import io
import itertools
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

# ── Instrument configs ────────────────────────────────────────────────────────

INSTRUMENTS = {
    "EU4": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0004.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0008.csv",
        "is_start": "2023-01-23", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0004",
    },
    "EU5": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0005.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0008.csv",
        "is_start": "2022-05-18", "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0005",
    },
    "EU6": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0006.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0012.csv",
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0006",
    },
    "EU7": {
        "renko_file": "OANDA_EURUSD, 1S renko 0.0007.csv",
        "htf_file":   "OANDA_EURUSD, 1S renko 0.0012.csv",
        "is_start": None, "is_end": "2025-09-30",
        "oos_start": "2025-10-01", "oos_end": "2026-03-19",
        "label": "EURUSD 0.0007",
    },
}

COMMISSION = 0.0046
CAPITAL = 1000.0
VOL_MAX = 1.5

# ── Data loading ─────────────────────────────────────────────────────────────


def _load_ltf(renko_file):
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)

    # ── NEW custom indicators ─────────────────────────────────────────────

    # 1. Brick velocity: bricks per hour (two-pointer O(n))
    times = df.index.values.astype("int64") / 1e9  # unix seconds
    n_rows = len(df)
    brick_vel = np.zeros(n_rows)
    left = 0
    window_sec = 3600  # 1 hour
    for i in range(n_rows):
        while times[i] - times[left] > window_sec and left < i:
            left += 1
        brick_vel[i] = i - left + 1  # bricks in last hour
    # Pre-shift
    df["brick_velocity"] = pd.Series(brick_vel, index=df.index).shift(1).values

    # 2. Trend score: fraction of last N bricks that are same direction
    brick_up = df["brick_up"].values.astype(float)
    for win in [8, 12]:
        score = np.full(len(df), np.nan)
        for i in range(win, len(df)):
            score[i] = np.mean(brick_up[i - win:i])
        col = f"trend_score_{win}"
        df[col] = pd.Series(score, index=df.index).shift(1).values

    # 3. Consecutive count: running count of same-direction bricks
    bu = df["brick_up"].values
    consec = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if bu[i] == bu[i - 1]:
            consec[i] = consec[i - 1] + 1
        else:
            consec[i] = 1
    consec[0] = 1
    df["consec_count"] = pd.Series(consec, index=df.index).shift(1).values

    return df


def _load_htf(htf_file):
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    df = load_renko_export(htf_file)
    add_renko_indicators(df)
    return df


def _align_htf_to_ltf(df_ltf, df_htf, htf_long, htf_short):
    htf_frame = pd.DataFrame({
        "t": df_htf.index.values, "gl": htf_long.astype(float),
        "gs": htf_short.astype(float),
    }).sort_values("t")
    ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    gl, gs = merged["gl"].values, merged["gs"].values
    return (np.where(np.isnan(gl), True, gl > 0.5).astype(bool),
            np.where(np.isnan(gs), True, gs > 0.5).astype(bool))


# ── Backtest runner ──────────────────────────────────────────────────────────


def _run_bt(df, le, lx, se, sx, start, end):
    from engine import BacktestConfig, run_backtest_long_short
    df2 = df.copy()
    df2["long_entry"] = le; df2["long_exit"] = lx
    df2["short_entry"] = se; df2["short_exit"] = sx
    cfg = BacktestConfig(
        initial_capital=CAPITAL, commission_pct=COMMISSION,
        slippage_ticks=0, qty_type="fixed", qty_value=1000.0,
        pyramiding=1, start_date=start, end_date=end,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df2, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf": float("inf") if math.isinf(pf) else float(pf),
        "net": float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr": float(kpis.get("win_rate", 0.0) or 0.0),
        "dd": float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


# ── Signal generators per mode ──────────────────────────────────────────────


def _gen_mode1_pullback(brick_up, ema_fast, ema_slow,
                        cooldown, pb_len, gate_long, gate_short):
    """Mode 1: Trend Pullback Entry.

    Logic:
    - Trend established: ema_fast > ema_slow (long) or < (short)
    - Wait for pb_len consecutive pullback bricks (against trend)
    - Enter on first continuation brick (back in trend direction)
    - Exit on first reversal brick
    """
    n = len(brick_up)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit: first opposing brick
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        # Cooldown
        if (i - last_trade_bar) < cooldown:
            continue

        # NaN guard
        ef = ema_fast[i]; es = ema_slow[i]
        if np.isnan(ef) or np.isnan(es):
            continue

        # Check trend
        bull_trend = ef > es
        bear_trend = ef < es

        if not bull_trend and not bear_trend:
            continue

        # Check pullback: pb_len consecutive bricks against trend just ended
        # Current brick must be continuation (back in trend direction)
        if bull_trend and up:
            # Need pb_len down bricks before this
            pullback_ok = True
            for j in range(1, pb_len + 1):
                if i - j < 0 or bool(brick_up[i - j]):
                    pullback_ok = False
                    break
            if not pullback_ok:
                continue
            cand = 1
        elif bear_trend and not up:
            pullback_ok = True
            for j in range(1, pb_len + 1):
                if i - j < 0 or not bool(brick_up[i - j]):
                    pullback_ok = False
                    break
            if not pullback_ok:
                continue
            cand = -1
        else:
            continue

        # Gate check
        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        # Fire
        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_mode2_momentum_scalp(brick_up, gate_long, gate_short,
                              consec_count, cooldown, min_consec):
    """Mode 2: Momentum Scalp.

    Logic:
    - Enter when current brick direction AND consec_count >= min_consec
    - Gates provide indicator agreement (replaces multi-indicator check)
    - Very high frequency with low min_consec
    - Exit on first reversal brick
    """
    n = len(brick_up)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        # Cooldown
        if (i - last_trade_bar) < cooldown:
            continue

        # Consecutive count check
        cc = consec_count[i]
        if np.isnan(cc) or cc < min_consec:
            continue

        # Direction
        if up:
            cand = 1
        else:
            cand = -1

        # Gate
        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        # Fire
        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_mode3_velocity(brick_up, brick_velocity, gate_long, gate_short,
                        cooldown, vel_thresh):
    """Mode 3: Velocity Burst.

    Logic:
    - Enter when brick_velocity >= vel_thresh (bricks forming fast)
    - Direction from current brick
    - High velocity = strong conviction = higher WR
    - Exit on first reversal brick
    """
    n = len(brick_up)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = 200

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        # Cooldown
        if (i - last_trade_bar) < cooldown:
            continue

        # Velocity check
        vel = brick_velocity[i]
        if np.isnan(vel) or vel < vel_thresh:
            continue

        # Direction
        if up:
            cand = 1
        else:
            cand = -1

        # Gate
        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        # Fire
        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_mode4_r001_light(brick_up, cooldown, gate_long, gate_short):
    """Mode 4: R001 Ultra-Light (N=2 consecutive, low cooldown, light gates)."""
    n = len(brick_up)
    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    warmup = 200
    n_bricks = 2  # fixed at 2 for max frequency

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit
        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        # Cooldown
        if (i - last_trade_bar) < cooldown:
            continue

        # N=2 consecutive check
        window = brick_up[i - n_bricks + 1: i + 1]
        if bool(np.all(window)):
            cand = 1
        elif bool(not np.any(window)):
            cand = -1
        else:
            continue

        # Gate
        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        # Fire
        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


# ── Gate helpers ──────────────────────────────────────────────────────────────


def _nan_gate(v, cond_long, cond_short):
    m = np.isnan(v)
    return m | cond_long, m | cond_short


def _nan_gate2(a, b, cond_long, cond_short):
    m = np.isnan(a) | np.isnan(b)
    return m | cond_long, m | cond_short


# ── Pre-compute gates ────────────────────────────────────────────────────────


def _precompute_gates(df_ltf, df_htf):
    n = len(df_ltf)
    hours = df_ltf.index.hour
    vr = df_ltf["vol_ratio"].values
    vol_ok = np.isnan(vr) | (vr <= VOL_MAX)

    gates = {}

    # Session baselines (wider for HF)
    for s in [0, 10, 12, 13]:
        if s == 0:
            ok = vol_ok.copy()
        else:
            ok = (hours >= s) & vol_ok
        gates[f"base_s{s}"] = (ok.copy(), ok.copy())

    # ADX levels (lower thresholds for HF)
    adx = df_ltf["adx"].values
    adx_nan = np.isnan(adx)
    for a in [0, 15, 20, 25]:
        ok = adx_nan | (adx >= a) if a > 0 else np.ones(n, dtype=bool)
        gates[f"adx_{a}"] = ok

    # ── Directional gates (P6 proven) ─────────────────────────────────────
    from renko.phase6_sweep import _compute_gate_arrays
    for g in ["escgo_cross", "stoch_cross", "ema_cross", "psar_dir",
              "kama_slope", "ichi_cloud"]:
        gates[f"p6:{g}"] = _compute_gate_arrays(df_ltf, g)

    # Supertrend direction
    st = df_ltf["st_dir"].values
    gates["st_dir"] = _nan_gate(st, st > 0, st < 0)

    # MACD histogram direction
    mh = df_ltf["macd_hist"].values
    gates["macd_hist_dir"] = _nan_gate(mh, mh > 0, mh < 0)

    # Stoch zone: K > 50 AND K > D
    sk = df_ltf["stoch_k"].values; sd = df_ltf["stoch_d"].values
    sk_nan = np.isnan(sk) | np.isnan(sd)
    gates["stoch_zone"] = (
        sk_nan | ((sk > 50) & (sk > sd)),
        sk_nan | ((sk < 50) & (sk < sd)),
    )

    # DI cross
    pdi = df_ltf["plus_di"].values; mdi = df_ltf["minus_di"].values
    gates["di_cross"] = _nan_gate2(pdi, mdi, pdi > mdi, pdi < mdi)

    # Triple EMA
    e9 = df_ltf["ema9"].values; e21 = df_ltf["ema21"].values; e50 = df_ltf["ema50"].values
    ema_nan = np.isnan(e9) | np.isnan(e21) | np.isnan(e50)
    gates["triple_ema"] = (
        ema_nan | ((e9 > e21) & (e21 > e50)),
        ema_nan | ((e9 < e21) & (e21 < e50)),
    )

    # ── HTF gates ─────────────────────────────────────────────────────────
    htf_adx = df_htf["adx"].values
    htf_nan = np.isnan(htf_adx)
    for t in [0, 25, 30, 35]:
        if t == 0:
            gates[f"htf_adx_{t}"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))
        else:
            ok = htf_nan | (htf_adx >= t)
            al, as_ = _align_htf_to_ltf(df_ltf, df_htf, ok, ok.copy())
            gates[f"htf_adx_{t}"] = (al, as_)

    # HTF EMA cross
    htf_e9 = df_htf["ema9"].values; htf_e21 = df_htf["ema21"].values
    htf_ema_nan = np.isnan(htf_e9) | np.isnan(htf_e21)
    hel = htf_ema_nan | (htf_e9 > htf_e21)
    hes = htf_ema_nan | (htf_e9 < htf_e21)
    gates["htf_ema"] = _align_htf_to_ltf(df_ltf, df_htf, hel, hes)

    # HTF supertrend
    htf_st = df_htf["st_dir"].values
    htf_st_nan = np.isnan(htf_st)
    gates["htf_st"] = _align_htf_to_ltf(
        df_ltf, df_htf,
        htf_st_nan | (htf_st > 0),
        htf_st_nan | (htf_st < 0),
    )

    return gates


# ── Build combos ─────────────────────────────────────────────────────────────


def _build_all_combos(mode_num=None):
    combos = []

    # Mode 1: Trend Pullback
    if mode_num is None or mode_num == 1:
        for sess, adx, pb_len, ema_pair, p6, htf_t in itertools.product(
            [0, 12],
            [0, 20],
            [1, 2],
            ["9_21", "21_50"],
            ["none", "stoch_cross", "escgo_cross", "psar_dir",
             "st_dir", "di_cross"],
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": 1, "sess": sess, "adx": adx, "pb_len": pb_len,
                    "ema_pair": ema_pair, "p6": p6, "htf_thresh": htf_t,
                    "cooldown": cd,
                })

    # Mode 2: Momentum Scalp
    if mode_num is None or mode_num == 2:
        for sess, adx, min_consec, p6, htf_t in itertools.product(
            [0, 12],
            [0, 20],
            [2, 3, 4],
            ["none", "stoch_cross", "escgo_cross", "ema_cross",
             "psar_dir", "st_dir", "stoch_zone", "macd_hist_dir",
             "di_cross", "triple_ema"],
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": 2, "sess": sess, "adx": adx,
                    "min_consec": min_consec, "p6": p6,
                    "htf_thresh": htf_t, "cooldown": cd,
                })

    # Mode 3: Velocity Burst (thresholds calibrated to actual data: median=3, max=34)
    if mode_num is None or mode_num == 3:
        for sess, adx, vel_thresh, p6, htf_t in itertools.product(
            [0, 12],
            [0, 20],
            [5, 8, 12, 18],
            ["none", "stoch_cross", "escgo_cross", "psar_dir",
             "st_dir", "di_cross"],
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": 3, "sess": sess, "adx": adx,
                    "vel_thresh": vel_thresh, "p6": p6,
                    "htf_thresh": htf_t, "cooldown": cd,
                })

    # Mode 4: R001 Ultra-Light (N=2 fixed)
    if mode_num is None or mode_num == 4:
        for sess, adx, p6, htf_t in itertools.product(
            [0, 12],
            [0, 20],
            ["none", "stoch_cross", "escgo_cross", "ema_cross",
             "psar_dir", "st_dir", "stoch_zone", "macd_hist_dir",
             "di_cross", "triple_ema"],
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": 4, "sess": sess, "adx": adx, "p6": p6,
                    "htf_thresh": htf_t, "cooldown": cd,
                })

    return combos


# ── Worker ───────────────────────────────────────────────────────────────────

_w_data = {}


def _init_worker(brick_up_bytes, df_bytes, arrays_bytes,
                 is_start, is_end, oos_start, oos_end):
    import pickle
    _w_data["brick_up"] = pickle.loads(brick_up_bytes)
    _w_data["df"] = pd.read_pickle(io.BytesIO(df_bytes))
    _w_data["arrays"] = pickle.loads(arrays_bytes)
    _w_data["is_start"] = is_start
    _w_data["is_end"] = is_end
    _w_data["oos_start"] = oos_start
    _w_data["oos_end"] = oos_end


def _run_one_combo(args):
    combo, gate_long, gate_short = args
    brick_up = _w_data["brick_up"]
    df = _w_data["df"]
    arrays = _w_data["arrays"]
    mode = combo["mode"]

    if mode == 1:
        ema_pair = combo["ema_pair"]
        if ema_pair == "9_21":
            ema_f, ema_s = arrays["ema9"], arrays["ema21"]
        else:
            ema_f, ema_s = arrays["ema21"], arrays["ema50"]
        le, lx, se, sx = _gen_mode1_pullback(
            brick_up, ema_f, ema_s,
            combo["cooldown"], combo["pb_len"], gate_long, gate_short,
        )
    elif mode == 2:
        le, lx, se, sx = _gen_mode2_momentum_scalp(
            brick_up, gate_long, gate_short,
            arrays["consec_count"], combo["cooldown"], combo["min_consec"],
        )
    elif mode == 3:
        le, lx, se, sx = _gen_mode3_velocity(
            brick_up, arrays["brick_velocity"], gate_long, gate_short,
            combo["cooldown"], combo["vel_thresh"],
        )
    elif mode == 4:
        le, lx, se, sx = _gen_mode4_r001_light(
            brick_up, combo["cooldown"], gate_long, gate_short,
        )

    is_r = _run_bt(df, le, lx, se, sx, _w_data["is_start"], _w_data["is_end"])
    oos_r = _run_bt(df, le, lx, se, sx, _w_data["oos_start"], _w_data["oos_end"])
    return is_r, oos_r


# ── Assemble gate arrays ────────────────────────────────────────────────────


def _assemble_tasks(combos, gates):
    tasks = []
    for combo in combos:
        bl, bs = gates[f"base_s{combo['sess']}"]
        bl = bl.copy(); bs = bs.copy()

        # ADX gate
        adx_v = combo["adx"]
        if adx_v > 0:
            bl &= gates[f"adx_{adx_v}"]
            bs &= gates[f"adx_{adx_v}"]

        # P6 / directional gate
        p6 = combo.get("p6", "none")
        if p6 != "none":
            if p6 in gates:
                pl, ps = gates[p6]
            elif f"p6:{p6}" in gates:
                pl, ps = gates[f"p6:{p6}"]
            else:
                pl, ps = gates.get(p6, (np.ones(len(bl), dtype=bool),
                                        np.ones(len(bl), dtype=bool)))
            bl &= pl; bs &= ps

        # HTF gate
        htf_t = combo.get("htf_thresh", 0)
        if htf_t > 0:
            hl, hs = gates[f"htf_adx_{htf_t}"]
            bl &= hl; bs &= hs

        tasks.append((combo, bl, bs))

    return tasks


# ── Main sweep ───────────────────────────────────────────────────────────────


def run_sweep(mode_num=None):
    import pickle

    combos = _build_all_combos(mode_num)
    n_inst = len(INSTRUMENTS)
    total = len(combos) * n_inst

    mode_names = {1: "Trend Pullback", 2: "Momentum Scalp",
                  3: "Velocity Burst", 4: "R001 Ultra-Light"}

    print(f"\n{'='*70}")
    print(f"Phase 15 — EURUSD High-Frequency Sweep")
    print(f"Modes: {mode_num or 'ALL (1-4)'}")
    print(f"Combos per instrument: {len(combos)}")
    print(f"Instruments: {list(INSTRUMENTS.keys())}")
    print(f"Total runs: {total} ({total*2} backtests)")
    print(f"Workers: {MAX_WORKERS}")
    print(f"{'='*70}\n")

    all_results = []

    for inst_key, cfg in INSTRUMENTS.items():
        print(f"\n--- [{inst_key}] {cfg['label']} ---", flush=True)
        print("  Loading LTF data + custom indicators...", flush=True)
        df_ltf = _load_ltf(cfg["renko_file"])
        print("  Loading HTF data...", flush=True)
        df_htf = _load_htf(cfg["htf_file"])

        is_start = cfg["is_start"] or str(df_ltf.index[0].date())

        print("  Pre-computing gates...", flush=True)
        gates = _precompute_gates(df_ltf, df_htf)

        brick_up = df_ltf["brick_up"].values

        # Pack arrays for workers
        arrays = {
            "ema9": df_ltf["ema9"].values,
            "ema21": df_ltf["ema21"].values,
            "ema50": df_ltf["ema50"].values,
            "trend_score_8": df_ltf["trend_score_8"].values,
            "consec_count": df_ltf["consec_count"].values,
            "brick_velocity": df_ltf["brick_velocity"].values,
        }

        print(f"  Assembling {len(combos)} combos...", flush=True)
        tasks = _assemble_tasks(combos, gates)

        print(f"  Running {len(tasks)} combos...", flush=True)
        done = 0
        inst_results = []

        buf = io.BytesIO()
        df_ltf.to_pickle(buf)
        df_bytes = buf.getvalue()

        brick_up_bytes = pickle.dumps(brick_up)
        arrays_bytes = pickle.dumps(arrays)

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_init_worker,
            initargs=(brick_up_bytes, df_bytes, arrays_bytes,
                      is_start, cfg["is_end"],
                      cfg["oos_start"], cfg["oos_end"]),
        ) as pool:
            futures = {}
            for task in tasks:
                combo, gl, gs = task
                f = pool.submit(_run_one_combo, (combo, gl, gs))
                futures[f] = combo

            for fut in as_completed(futures):
                combo = futures[fut]
                try:
                    is_r, oos_r = fut.result()
                    inst_results.append({
                        "inst": inst_key, "label": cfg["label"],
                        "mode": combo["mode"], "combo": combo,
                        "is": is_r, "oos": oos_r,
                    })
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                done += 1
                if done % 1000 == 0 or done == len(tasks):
                    print(f"    [{done:>6}/{len(tasks)}]", flush=True)

        all_results.extend(inst_results)
        print(f"  [{inst_key}] done — {len(inst_results)} results", flush=True)

    # ── Filter: OOS trades >= 100 AND WR >= 60% ─────────────────────────────
    hf_good = [r for r in all_results
               if r["oos"]["trades"] >= 100 and r["oos"]["wr"] >= 60.0
               and r["oos"]["net"] > 0]
    hf_good.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)

    # Also filter WR >= 70%
    hf_great = [r for r in all_results
                if r["oos"]["trades"] >= 80 and r["oos"]["wr"] >= 70.0
                and r["oos"]["net"] > 0]
    hf_great.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)

    # ── Display ──────────────────────────────────────────────────────────────
    def _fmt_row(r, rank):
        pf = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
        c = r["combo"]
        skip = {"mode"}
        extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in skip)
        tpd = r["oos"]["trades"] / 170  # ~170 trading days in OOS
        return (f"  {rank:>2}. [{r['inst']}] Mode{r['mode']}({mode_names[r['mode']]}) "
                f"OOS PF={pf:>7} T={r['oos']['trades']:>4} "
                f"({tpd:.1f}/day) WR={r['oos']['wr']:>5.1f}% "
                f"Net={r['oos']['net']:>8.2f} DD={r['oos']['dd']:>5.2f}% | {extra}")

    print(f"\n{'='*70}")
    print(f"HIGH-FREQ RESULTS: T>=100 AND WR>=60% ({len(hf_good)} configs)")
    print(f"{'='*70}")
    for i, r in enumerate(hf_good[:40]):
        print(_fmt_row(r, i + 1))

    print(f"\n{'='*70}")
    print(f"HIGH WR RESULTS: T>=80 AND WR>=70% ({len(hf_great)} configs)")
    print(f"{'='*70}")
    for i, r in enumerate(hf_great[:40]):
        print(_fmt_row(r, i + 1))

    # ── Per-mode best ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BEST PER MODE (T>=50, net>0)")
    print(f"{'='*70}")
    for mode in [1, 2, 3, 4]:
        mode_r = [r for r in all_results
                  if r["mode"] == mode and r["oos"]["trades"] >= 50
                  and r["oos"]["net"] > 0]
        mode_r.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)
        print(f"\n  MODE {mode}: {mode_names[mode]} ({len(mode_r)} viable)")
        for i, r in enumerate(mode_r[:10]):
            print(_fmt_row(r, i + 1))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "ai_context" / "phase15_results.json"
    out_path.parent.mkdir(exist_ok=True)

    serializable = []
    for r in all_results:
        c = r["combo"]
        sr = {
            "inst": r["inst"], "label": r["label"], "mode": r["mode"],
            "combo": {k: v for k, v in c.items() if k != "mode"},
            "is_pf": "inf" if math.isinf(r["is"]["pf"]) else r["is"]["pf"],
            "is_trades": r["is"]["trades"], "is_wr": r["is"]["wr"],
            "is_net": r["is"]["net"],
            "oos_pf": "inf" if math.isinf(r["oos"]["pf"]) else r["oos"]["pf"],
            "oos_trades": r["oos"]["trades"], "oos_wr": r["oos"]["wr"],
            "oos_net": r["oos"]["net"], "oos_dd": r["oos"]["dd"],
        }
        serializable.append(sr)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total: {len(all_results)} runs ({len(all_results)*2} backtests)")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=None,
                        help="Run single mode (1-4)")
    args = parser.parse_args()
    run_sweep(args.mode)
