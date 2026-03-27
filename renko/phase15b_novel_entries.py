#!/usr/bin/env python3
"""
phase15b_novel_entries.py — Genuinely Novel Entry Methods for EURUSD HF

NOT R001/R007 variants. Entirely new entry logic + new custom indicators.

NEW CUSTOM INDICATORS (computed inline):
  - momentum_score: weighted composite of RSI, Stoch, MACD, EMA alignment (-4 to +4)
  - flow_composite: CMF + norm(MFI) + OBV direction — money flow consensus
  - brick_efficiency: close-to-close distance / total brick range over N bricks

5 NOVEL ENTRY METHODS:

Mode A: "Confluence Score" — Enter when momentum_score crosses threshold
        AND brick direction agrees. Pure weight-of-evidence entry.

Mode B: "Squeeze Breakout" — Enter when BB squeeze releases (sq_on→sq_off)
        with brick direction. Volatility compression → expansion.

Mode C: "EMA Cross Event" — Enter on the actual EMA9/EMA21 cross bar
        (not as a gate, but as the trigger). Fresh regime change entry.

Mode D: "Exhaustion Reversal" — After 6+ consecutive same-direction bricks,
        enter OPPOSITE on first reversal brick. Mean reversion play.

Mode E: "Flow Thrust" — Enter when flow_composite crosses above/below
        threshold AND brick direction confirms. Smart money entry.

Usage:
  python renko/phase15b_novel_entries.py
  python renko/phase15b_novel_entries.py --mode A
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

# ── Data loading + NEW indicators ────────────────────────────────────────────


def _load_ltf(renko_file):
    from renko.data import load_renko_export
    from renko.indicators import add_renko_indicators
    from renko.phase6_enrichment import add_phase6_indicators

    df = load_renko_export(renko_file)
    add_renko_indicators(df)
    add_phase6_indicators(df, include_mk=False)
    n = len(df)

    # ── 1. Momentum Score (novel composite) ─────────────────────────────
    # Score from -4 to +4: sum of individual directional signals
    rsi = df["rsi"].values
    stoch_k = df["stoch_k"].values
    stoch_d = df["stoch_d"].values
    macd_hist = df["macd_hist"].values
    ema9 = df["ema9"].values
    ema21 = df["ema21"].values

    mom_score = np.full(n, np.nan)
    for i in range(n):
        if any(np.isnan(x) for x in [rsi[i], stoch_k[i], stoch_d[i],
                                      macd_hist[i], ema9[i], ema21[i]]):
            continue
        s = 0.0
        # RSI contribution: >55 bullish, <45 bearish
        if rsi[i] > 55: s += 1
        elif rsi[i] < 45: s -= 1
        # Stoch: K > D bullish
        if stoch_k[i] > stoch_d[i]: s += 1
        else: s -= 1
        # MACD hist direction
        if macd_hist[i] > 0: s += 1
        elif macd_hist[i] < 0: s -= 1
        # EMA alignment
        if ema9[i] > ema21[i]: s += 1
        elif ema9[i] < ema21[i]: s -= 1
        mom_score[i] = s
    # Already pre-shifted (all inputs are pre-shifted)
    df["momentum_score"] = mom_score

    # ── 2. Flow Composite (novel: money flow consensus) ──────────────────
    cmf = df["cmf"].values
    mfi = df["mfi"].values
    obv = df["obv"].values
    obv_ema = df["obv_ema"].values

    flow_comp = np.full(n, np.nan)
    for i in range(n):
        if any(np.isnan(x) for x in [cmf[i], mfi[i], obv[i], obv_ema[i]]):
            continue
        fc = 0.0
        # CMF > 0 = buying, < 0 = selling
        fc += np.clip(cmf[i] * 2, -1, 1)  # scale CMF to -1..+1
        # MFI: normalize to -1..+1 (50 = neutral)
        fc += (mfi[i] - 50) / 50
        # OBV above its EMA = bullish
        if obv[i] > obv_ema[i]: fc += 1
        else: fc -= 1
        flow_comp[i] = fc
    df["flow_composite"] = flow_comp

    # ── 3. Consecutive count (running) ───────────────────────────────────
    bu = df["brick_up"].values
    consec = np.zeros(n, dtype=int)
    for i in range(1, n):
        if bu[i] == bu[i - 1]:
            consec[i] = consec[i - 1] + 1
        else:
            consec[i] = 1
    consec[0] = 1
    df["consec_count"] = pd.Series(consec, index=df.index).shift(1).values

    # ── 4. Squeeze state (already in df from indicators) ─────────────────
    # sq_on, sq_momentum already available and pre-shifted

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


# ── NOVEL signal generators ─────────────────────────────────────────────────


def _gen_modeA_confluence(brick_up, momentum_score, cooldown, score_thresh,
                          gate_long, gate_short):
    """Mode A: Confluence Score Entry.

    Enter when momentum_score >= score_thresh (long) or <= -score_thresh (short)
    AND brick direction agrees. Pure weight-of-evidence.
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

        if (i - last_trade_bar) < cooldown:
            continue

        ms = momentum_score[i]
        if np.isnan(ms):
            continue

        # Confluence: score must reach threshold AND brick must agree
        if ms >= score_thresh and up:
            cand = 1
        elif ms <= -score_thresh and not up:
            cand = -1
        else:
            continue

        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_modeB_st_flip(brick_up, st_dir, cooldown, gate_long, gate_short):
    """Mode B: Supertrend Flip Entry.

    Enter when Supertrend direction flips (+1→-1 or -1→+1).
    Brick must confirm the new direction. Clean regime-change signal.
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

        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        st_now = st_dir[i]
        st_prev = st_dir[i - 1]
        if np.isnan(st_now) or np.isnan(st_prev):
            continue

        # Supertrend just flipped
        if st_prev < 0 and st_now > 0 and up:
            cand = 1  # flip to bullish + up brick
        elif st_prev > 0 and st_now < 0 and not up:
            cand = -1  # flip to bearish + down brick
        else:
            continue

        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_modeF_band_bounce(brick_up, bb_pct_b, cooldown, band_thresh,
                           gate_long, gate_short):
    """Mode F: Bollinger Band Bounce Entry.

    Mean reversion: enter when price at band extreme AND brick reverses.
    bb_pct_b < band_thresh (lower band) + up brick → long
    bb_pct_b > (1-band_thresh) (upper band) + down brick → short
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

        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        pct = bb_pct_b[i]
        if np.isnan(pct):
            continue

        # Band extreme + reversal brick
        if pct <= band_thresh and up:
            cand = 1  # at lower band, brick up → mean reversion long
        elif pct >= (1.0 - band_thresh) and not up:
            cand = -1  # at upper band, brick down → mean reversion short
        else:
            continue

        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_modeC_ema_cross(brick_up, ema_fast, ema_slow, cooldown,
                         gate_long, gate_short):
    """Mode C: EMA Cross Event Entry.

    Enter on the bar where EMA fast crosses above/below EMA slow
    (actual cross, not just "above"). Brick must confirm direction.
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

        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        ef_now = ema_fast[i]; es_now = ema_slow[i]
        ef_prev = ema_fast[i - 1]; es_prev = ema_slow[i - 1]

        if any(np.isnan(x) for x in [ef_now, es_now, ef_prev, es_prev]):
            continue

        # Bullish cross: fast was below slow, now above
        bull_cross = ef_prev <= es_prev and ef_now > es_now
        # Bearish cross: fast was above slow, now below
        bear_cross = ef_prev >= es_prev and ef_now < es_now

        if bull_cross and up:
            cand = 1
        elif bear_cross and not up:
            cand = -1
        else:
            continue

        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_modeD_exhaustion(brick_up, consec_count, cooldown, exhaust_len,
                          gate_long, gate_short):
    """Mode D: Exhaustion Reversal Entry.

    After exhaust_len+ consecutive same-direction bricks, the move is
    likely exhausted. Enter OPPOSITE direction on first reversal brick.
    Mean reversion play — high WR because extended runs correct.
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

        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        cc = consec_count[i]
        if np.isnan(cc):
            continue

        # Current brick is the REVERSAL after exhaust_len+ consecutive
        # consec_count[i] was computed on shifted data, so it reflects
        # the run that JUST ended. We need: previous run was long AND
        # current brick reverses.
        # Since consec_count is pre-shifted, at bar i it tells us the
        # consecutive count as of bar i-1. If cc >= exhaust_len, the run
        # was exhausted. We check if current brick is opposite to prev.
        if i < 1:
            continue

        prev_up = bool(brick_up[i - 1])

        # Was there an exhaustion run that just ended?
        if cc < exhaust_len:
            continue

        # Current brick must be a reversal (opposite of the run)
        if prev_up and not up:
            # Long run exhausted → enter short (mean reversion)
            cand = -1
        elif not prev_up and up:
            # Down run exhausted → enter long
            cand = 1
        else:
            continue  # continuation, not reversal

        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

        if cand == 1:
            long_entry[i] = True
        else:
            short_entry[i] = True
        in_position = True
        trade_dir = cand
        last_trade_bar = i

    return long_entry, long_exit, short_entry, short_exit


def _gen_modeE_flow(brick_up, flow_composite, cooldown, flow_thresh,
                    gate_long, gate_short):
    """Mode E: Flow Thrust Entry.

    Enter when flow_composite crosses above flow_thresh (long) or
    below -flow_thresh (short) AND brick direction confirms.
    Smart money following entry.
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

        if in_position:
            is_opp = (trade_dir == 1 and not up) or (trade_dir == -1 and up)
            long_exit[i] = is_opp and trade_dir == 1
            short_exit[i] = is_opp and trade_dir == -1
            if is_opp:
                in_position = False
                trade_dir = 0
        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        fc = flow_composite[i]
        if np.isnan(fc):
            continue

        # Flow thrust: strong money flow + brick confirmation
        if fc >= flow_thresh and up:
            cand = 1
        elif fc <= -flow_thresh and not up:
            cand = -1
        else:
            continue

        if cand == 1 and not gate_long[i]:
            continue
        if cand == -1 and not gate_short[i]:
            continue

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


def _precompute_gates(df_ltf, df_htf):
    n = len(df_ltf)
    hours = df_ltf.index.hour
    vr = df_ltf["vol_ratio"].values
    vol_ok = np.isnan(vr) | (vr <= VOL_MAX)

    gates = {}

    for s in [0, 12]:
        if s == 0:
            ok = vol_ok.copy()
        else:
            ok = (hours >= s) & vol_ok
        gates[f"base_s{s}"] = (ok.copy(), ok.copy())

    adx = df_ltf["adx"].values
    adx_nan = np.isnan(adx)
    for a in [0, 20]:
        ok = adx_nan | (adx >= a) if a > 0 else np.ones(n, dtype=bool)
        gates[f"adx_{a}"] = ok

    # P6 gates
    from renko.phase6_sweep import _compute_gate_arrays
    for g in ["escgo_cross", "stoch_cross", "ema_cross", "psar_dir",
              "kama_slope"]:
        gates[f"p6:{g}"] = _compute_gate_arrays(df_ltf, g)

    st = df_ltf["st_dir"].values
    gates["st_dir"] = _nan_gate(st, st > 0, st < 0)

    pdi = df_ltf["plus_di"].values; mdi = df_ltf["minus_di"].values
    gates["di_cross"] = _nan_gate2(pdi, mdi, pdi > mdi, pdi < mdi)

    # HTF gates
    htf_adx = df_htf["adx"].values
    htf_nan = np.isnan(htf_adx)
    for t in [0, 30]:
        if t == 0:
            gates[f"htf_adx_{t}"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))
        else:
            ok = htf_nan | (htf_adx >= t)
            al, as_ = _align_htf_to_ltf(df_ltf, df_htf, ok, ok.copy())
            gates[f"htf_adx_{t}"] = (al, as_)

    return gates


# ── Build combos ─────────────────────────────────────────────────────────────


def _build_all_combos(mode=None):
    combos = []

    p6_options = ["none", "escgo_cross", "stoch_cross", "psar_dir",
                  "st_dir", "di_cross"]

    # Mode A: Confluence Score
    if mode is None or mode == "A":
        for sess, adx, score_thresh, p6, htf_t in itertools.product(
            [0, 12], [0, 20],
            [2, 3, 4],             # momentum_score threshold
            p6_options,
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": "A", "sess": sess, "adx": adx,
                    "score_thresh": score_thresh, "p6": p6,
                    "htf_thresh": htf_t, "cooldown": cd,
                })

    # Mode B: Supertrend Flip
    if mode is None or mode == "B":
        for sess, adx, p6, htf_t in itertools.product(
            [0, 12], [0, 20],
            p6_options,
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": "B", "sess": sess, "adx": adx,
                    "p6": p6, "htf_thresh": htf_t, "cooldown": cd,
                })

    # Mode F: Band Bounce (mean reversion)
    if mode is None or mode == "F":
        for sess, adx, band_thresh, p6, htf_t in itertools.product(
            [0, 12], [0, 20],
            [0.05, 0.1, 0.15, 0.2],
            p6_options,
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": "F", "sess": sess, "adx": adx,
                    "band_thresh": band_thresh, "p6": p6,
                    "htf_thresh": htf_t, "cooldown": cd,
                })

    # Mode C: EMA Cross Event
    if mode is None or mode == "C":
        for sess, adx, ema_pair, p6, htf_t in itertools.product(
            [0, 12], [0, 20],
            ["9_21", "9_50", "21_50"],
            p6_options,
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": "C", "sess": sess, "adx": adx,
                    "ema_pair": ema_pair, "p6": p6,
                    "htf_thresh": htf_t, "cooldown": cd,
                })

    # Mode D: Exhaustion Reversal
    if mode is None or mode == "D":
        for sess, adx, exhaust_len, p6, htf_t in itertools.product(
            [0, 12], [0, 20],
            [4, 6, 8, 10, 12],     # consecutive bricks before exhaustion
            p6_options,
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": "D", "sess": sess, "adx": adx,
                    "exhaust_len": exhaust_len, "p6": p6,
                    "htf_thresh": htf_t, "cooldown": cd,
                })

    # Mode E: Flow Thrust
    if mode is None or mode == "E":
        for sess, adx, flow_thresh, p6, htf_t in itertools.product(
            [0, 12], [0, 20],
            [0.5, 1.0, 1.5, 2.0],  # flow_composite threshold
            p6_options,
            [0, 30],
        ):
            for cd in [3, 5, 8]:
                combos.append({
                    "mode": "E", "sess": sess, "adx": adx,
                    "flow_thresh": flow_thresh, "p6": p6,
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

    if mode == "A":
        le, lx, se, sx = _gen_modeA_confluence(
            brick_up, arrays["momentum_score"],
            combo["cooldown"], combo["score_thresh"],
            gate_long, gate_short,
        )
    elif mode == "B":
        le, lx, se, sx = _gen_modeB_st_flip(
            brick_up, arrays["st_dir"],
            combo["cooldown"], gate_long, gate_short,
        )
    elif mode == "C":
        ep = combo["ema_pair"]
        if ep == "9_21":
            ef, es = arrays["ema9"], arrays["ema21"]
        elif ep == "9_50":
            ef, es = arrays["ema9"], arrays["ema50"]
        else:
            ef, es = arrays["ema21"], arrays["ema50"]
        le, lx, se, sx = _gen_modeC_ema_cross(
            brick_up, ef, es, combo["cooldown"],
            gate_long, gate_short,
        )
    elif mode == "D":
        le, lx, se, sx = _gen_modeD_exhaustion(
            brick_up, arrays["consec_count"],
            combo["cooldown"], combo["exhaust_len"],
            gate_long, gate_short,
        )
    elif mode == "E":
        le, lx, se, sx = _gen_modeE_flow(
            brick_up, arrays["flow_composite"],
            combo["cooldown"], combo["flow_thresh"],
            gate_long, gate_short,
        )

    elif mode == "F":
        le, lx, se, sx = _gen_modeF_band_bounce(
            brick_up, arrays["bb_pct_b"],
            combo["cooldown"], combo["band_thresh"],
            gate_long, gate_short,
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

        adx_v = combo["adx"]
        if adx_v > 0:
            bl &= gates[f"adx_{adx_v}"]
            bs &= gates[f"adx_{adx_v}"]

        p6 = combo.get("p6", "none")
        if p6 != "none":
            key = p6 if p6 in gates else f"p6:{p6}"
            if key in gates:
                pl, ps = gates[key]
                bl &= pl; bs &= ps

        htf_t = combo.get("htf_thresh", 0)
        if htf_t > 0:
            hl, hs = gates[f"htf_adx_{htf_t}"]
            bl &= hl; bs &= hs

        tasks.append((combo, bl, bs))

    return tasks


# ── Main sweep ───────────────────────────────────────────────────────────────


def run_sweep(mode=None):
    import pickle

    combos = _build_all_combos(mode)
    n_inst = len(INSTRUMENTS)
    total = len(combos) * n_inst

    mode_names = {
        "A": "Confluence Score", "B": "Supertrend Flip",
        "C": "EMA Cross Event", "D": "Exhaustion Reversal",
        "E": "Flow Thrust", "F": "Band Bounce",
    }

    print(f"\n{'='*70}")
    print(f"Phase 15b — Novel Entry Methods Sweep")
    print(f"Modes: {mode or 'ALL (A-E)'}")
    print(f"Combos per instrument: {len(combos)}")
    print(f"Instruments: {list(INSTRUMENTS.keys())}")
    print(f"Total runs: {total} ({total*2} backtests)")
    print(f"Workers: {MAX_WORKERS}")
    print(f"{'='*70}\n")

    all_results = []

    for inst_key, cfg in INSTRUMENTS.items():
        print(f"\n--- [{inst_key}] {cfg['label']} ---", flush=True)
        print("  Loading data + novel indicators...", flush=True)
        df_ltf = _load_ltf(cfg["renko_file"])
        print("  Loading HTF data...", flush=True)
        df_htf = _load_htf(cfg["htf_file"])

        is_start = cfg["is_start"] or str(df_ltf.index[0].date())

        print("  Pre-computing gates...", flush=True)
        gates = _precompute_gates(df_ltf, df_htf)

        brick_up = df_ltf["brick_up"].values

        arrays = {
            "momentum_score": df_ltf["momentum_score"].values,
            "flow_composite": df_ltf["flow_composite"].values,
            "consec_count": df_ltf["consec_count"].values,
            "st_dir": df_ltf["st_dir"].values,
            "bb_pct_b": df_ltf["bb_pct_b"].values,
            "ema9": df_ltf["ema9"].values,
            "ema21": df_ltf["ema21"].values,
            "ema50": df_ltf["ema50"].values,
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
                if done % 500 == 0 or done == len(tasks):
                    print(f"    [{done:>5}/{len(tasks)}]", flush=True)

        all_results.extend(inst_results)
        print(f"  [{inst_key}] done — {len(inst_results)} results", flush=True)

    # ── Display results ──────────────────────────────────────────────────────
    def _fmt(r, rank):
        pf = "INF" if math.isinf(r["oos"]["pf"]) else f"{r['oos']['pf']:.2f}"
        c = r["combo"]
        skip = {"mode"}
        extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in skip)
        tpd = r["oos"]["trades"] / 170
        return (f"  {rank:>2}. [{r['inst']}] Mode{r['mode']}({mode_names[r['mode']]}) "
                f"OOS PF={pf:>7} T={r['oos']['trades']:>4} ({tpd:.1f}/day) "
                f"WR={r['oos']['wr']:>5.1f}% Net={r['oos']['net']:>8.2f} "
                f"DD={r['oos']['dd']:>5.2f}% | {extra}")

    # High-freq + profitable
    hf = [r for r in all_results
          if r["oos"]["trades"] >= 80 and r["oos"]["net"] > 0]
    hf.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)

    print(f"\n{'='*70}")
    print(f"PROFITABLE + T>=80 ({len(hf)} configs)")
    print(f"{'='*70}")
    for i, r in enumerate(hf[:30]):
        print(_fmt(r, i + 1))

    # High WR
    hw = [r for r in all_results
          if r["oos"]["trades"] >= 50 and r["oos"]["wr"] >= 65.0
          and r["oos"]["net"] > 0]
    hw.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["net"]), reverse=True)

    print(f"\n{'='*70}")
    print(f"HIGH WR (>=65%, T>=50, net>0): {len(hw)} configs")
    print(f"{'='*70}")
    for i, r in enumerate(hw[:30]):
        print(_fmt(r, i + 1))

    # Per-mode best
    print(f"\n{'='*70}")
    print(f"BEST PER MODE (T>=30, net>0)")
    print(f"{'='*70}")
    for m in ["A", "B", "C", "D", "E", "F"]:
        mr = [r for r in all_results
              if r["mode"] == m and r["oos"]["trades"] >= 30
              and r["oos"]["net"] > 0]
        mr.sort(key=lambda r: (r["oos"]["net"], r["oos"]["pf"]), reverse=True)
        print(f"\n  MODE {m}: {mode_names[m]} ({len(mr)} viable)")
        for i, r in enumerate(mr[:8]):
            print(_fmt(r, i + 1))

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = ROOT / "ai_context" / "phase15b_results.json"
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
    parser.add_argument("--mode", type=str, default=None,
                        choices=["A", "B", "C", "D", "E", "F"],
                        help="Run single mode")
    args = parser.parse_args()
    run_sweep(args.mode)
