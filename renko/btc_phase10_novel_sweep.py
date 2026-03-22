#!/usr/bin/env python3
"""
btc_phase10_novel_sweep.py -- BTC Phase 10: Novel Strategy Concepts (Long Only)

10 genuinely different approaches not yet tested:

    COMBO_STACK    BTC003 entries + C5_trend confluence combined (max trades)
    REGIME_ADAPT   Chop < 38.2 → momentum entry; Chop > 61.8 → oversold bounce
    ACCEL          MACD hist rising 2+ bars AND RSI slope positive (acceleration)
    HTF_ENTRY      HTF supertrend flip or HTF EMA9>21 cross triggers LTF entry
    DIP_BUY        RSI < 40 but EMA50 rising → buy on up brick (dip in uptrend)
    SCORE          Weighted sum of 7 indicators > threshold (nuanced scoring)
    COMPRESS       BB bandwidth at 20-bar low + close > BB_upper (squeeze breakout)
    W_BOTTOM       2+ down, 1+ up, 1+ down, current up (W reversal pattern)
    DI_SPREAD      (+DI - -DI) > threshold on up brick (directional intensity)
    MTF_CONFIRM    LTF RSI>50 AND HTF RSI>50 AND up brick (both TFs agree)

Each tested with gates [no_gates, gates_only, gates+htf35] x cooldowns [20, 35].

Usage:
    python renko/btc_phase10_novel_sweep.py
    python renko/btc_phase10_novel_sweep.py --no-parallel
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

COOLDOWNS = [20, 35]

STRATEGIES = [
    "combo_stack", "regime_adapt", "accel", "htf_entry", "dip_buy",
    "score", "compress", "w_bottom", "di_spread", "mtf_confirm",
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


# -- HTF indicator alignment helper --------------------------------------------

def _align_htf_to_ltf(df_ltf, df_htf, htf_col):
    """Align an HTF indicator column to LTF timestamps via merge_asof."""
    htf_vals = df_htf[htf_col].values
    htf_frame = pd.DataFrame({"t": df_htf.index.values, "v": htf_vals.astype(float)}).sort_values("t")
    ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
    merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
    return merged["v"].values


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
# Strategy generators
# ==============================================================================

def _gen_combo_stack(df, df_htf, cooldown, gate):
    """
    BTC003 entries (R001+R002+FLIP+BBRK) + C5_trend confluence entries.
    Two independent entry systems combined for maximum trade frequency.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    n_bricks = 2

    # BTC003 indicators
    st_dir = df["st_dir"].values
    bb_u = df["bb_upper"].values

    # C5_trend: ema9>21, kama_up, st_bull, psar_bull, macd_h_pos
    ema9 = df["ema9"].values
    ema21 = df["ema21"].values
    kama_s = df["kama_slope"].values
    psar_d = df["psar_dir"].values
    macd_h = df["macd_hist"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_r001 = -999_999
    last_flip = -999_999
    last_bb = -999_999
    last_conf = -999_999
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

        # -- BTC003 entries --
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

        # -- C5_trend: 3-of-5 confluence (independent cooldown) --
        if not triggered and up and (i - last_conf) >= cooldown:
            confirms = 0
            valid = 0
            if not np.isnan(ema9[i]) and not np.isnan(ema21[i]):
                valid += 1
                if ema9[i] > ema21[i]:
                    confirms += 1
            if not np.isnan(kama_s[i]):
                valid += 1
                if kama_s[i] > 0:
                    confirms += 1
            if not np.isnan(st_dir[i]):
                valid += 1
                if st_dir[i] > 0:
                    confirms += 1
            if not np.isnan(psar_d[i]):
                valid += 1
                if psar_d[i] > 0:
                    confirms += 1
            if not np.isnan(macd_h[i]):
                valid += 1
                if macd_h[i] > 0:
                    confirms += 1
            if valid >= 4 and confirms >= 3:
                triggered = True
                last_conf = i

        if triggered:
            entry[i] = True
            in_pos = True

    return entry, exit_


def _gen_regime_adapt(df, df_htf, cooldown, gate):
    """
    Adaptive regime-based entry:
    - Chop < 38.2 (trending) → enter on 2 consecutive up bricks (momentum)
    - Chop > 61.8 (choppy) → enter on stoch_k < 20 crossing above 20 (oversold bounce)
    """
    n = len(df)
    brick_up = df["brick_up"].values
    chop = df["chop"].values
    stoch_k = df["stoch_k"].values

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
        if np.isnan(chop[i]):
            continue

        triggered = False

        # Trending regime: momentum entry
        if chop[i] < 38.2:
            if up and bool(brick_up[i-1]):
                triggered = True

        # Choppy regime: oversold bounce
        elif chop[i] > 61.8:
            if not np.isnan(stoch_k[i]) and not np.isnan(stoch_k[i-1]):
                if up and stoch_k[i] > 20 and stoch_k[i-1] <= 20:
                    triggered = True

        if triggered:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_accel(df, df_htf, cooldown, gate):
    """
    Momentum acceleration: MACD histogram rising for 2+ bars
    AND RSI current > RSI 2 bars ago, on an up brick.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    macd_h = df["macd_hist"].values
    rsi = df["rsi"].values

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
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if (np.isnan(macd_h[i]) or np.isnan(macd_h[i-1]) or np.isnan(macd_h[i-2])
                or np.isnan(rsi[i]) or np.isnan(rsi[i-2])):
            continue

        # MACD histogram rising for 2 consecutive bars
        macd_rising = macd_h[i] > macd_h[i-1] and macd_h[i-1] > macd_h[i-2]
        # RSI accelerating (current > 2 bars ago)
        rsi_accel = rsi[i] > rsi[i-2]

        if up and macd_rising and rsi_accel:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_htf_entry(df, df_htf, cooldown, gate):
    """
    HTF signals as entry triggers (not just gates):
    - HTF supertrend flips bullish → enter on LTF up brick
    - HTF EMA9 crosses above EMA21 → enter on LTF up brick
    """
    n = len(df)
    brick_up = df["brick_up"].values

    # Align HTF indicators to LTF
    htf_st = _align_htf_to_ltf(df, df_htf, "st_dir")
    htf_ema9 = _align_htf_to_ltf(df, df_htf, "ema9")
    htf_ema21 = _align_htf_to_ltf(df, df_htf, "ema21")

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

        triggered = False

        # HTF supertrend flip bullish
        if not np.isnan(htf_st[i]) and not np.isnan(htf_st[i-1]):
            if htf_st[i] > 0 and htf_st[i-1] <= 0 and up:
                triggered = True

        # HTF EMA9 crosses above EMA21
        if not triggered:
            if (not np.isnan(htf_ema9[i]) and not np.isnan(htf_ema21[i])
                    and not np.isnan(htf_ema9[i-1]) and not np.isnan(htf_ema21[i-1])):
                if htf_ema9[i] > htf_ema21[i] and htf_ema9[i-1] <= htf_ema21[i-1] and up:
                    triggered = True

        if triggered:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_dip_buy(df, df_htf, cooldown, gate):
    """
    Buy dips in confirmed uptrends:
    RSI < 40 (short-term weakness) but EMA50 rising (long-term trend intact),
    on an up brick (reversal confirmation).
    """
    n = len(df)
    brick_up = df["brick_up"].values
    rsi = df["rsi"].values
    ema50 = df["ema50"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 55

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(rsi[i]) or np.isnan(ema50[i]) or np.isnan(ema50[i-1]):
            continue

        ema_rising = ema50[i] > ema50[i-1]
        oversold = rsi[i] < 40

        if up and oversold and ema_rising:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_score(df, df_htf, cooldown, gate):
    """
    Weighted indicator score > threshold.
    Weights based on Phase 8/9 effectiveness:
        ST_bull:   2.0  (strongest trend filter)
        MACD_h>0:  1.5  (strong momentum)
        EMA9>21:   1.5  (reliable trend)
        KAMA_up:   1.0  (adaptive)
        RSI>50:    1.0  (momentum regime)
        CMF>0:     0.5  (volume confirm)
        PSAR_bull: 0.5  (direction)
    Total possible: 8.0. Threshold: 5.0
    """
    n = len(df)
    brick_up = df["brick_up"].values
    st_dir = df["st_dir"].values
    macd_h = df["macd_hist"].values
    ema9 = df["ema9"].values
    ema21 = df["ema21"].values
    kama_s = df["kama_slope"].values
    rsi = df["rsi"].values
    cmf = df["cmf"].values
    psar_d = df["psar_dir"].values

    THRESHOLD = 5.0

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

        score = 0.0
        if not np.isnan(st_dir[i]) and st_dir[i] > 0:
            score += 2.0
        if not np.isnan(macd_h[i]) and macd_h[i] > 0:
            score += 1.5
        if not np.isnan(ema9[i]) and not np.isnan(ema21[i]) and ema9[i] > ema21[i]:
            score += 1.5
        if not np.isnan(kama_s[i]) and kama_s[i] > 0:
            score += 1.0
        if not np.isnan(rsi[i]) and rsi[i] > 50:
            score += 1.0
        if not np.isnan(cmf[i]) and cmf[i] > 0:
            score += 0.5
        if not np.isnan(psar_d[i]) and psar_d[i] > 0:
            score += 0.5

        if up and score >= THRESHOLD:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_compress(df, df_htf, cooldown, gate):
    """
    Volatility compression breakout:
    BB bandwidth at 20-bar low AND close > BB upper on up brick.
    Catches explosive moves after tight consolidation.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)
    bb_bw = df["bb_bw"].values
    bb_u = df["bb_upper"].values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 45

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(bb_bw[i]) or np.isnan(bb_u[i]):
            continue

        # Check if BB bandwidth is at 20-bar low
        bw_window = bb_bw[max(0, i-19):i+1]
        valid_bw = bw_window[~np.isnan(bw_window)]
        if len(valid_bw) < 10:
            continue
        bw_is_low = bb_bw[i] <= np.min(valid_bw) * 1.05  # within 5% of minimum

        if up and bw_is_low and close[i] > bb_u[i]:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_w_bottom(df, df_htf, cooldown, gate):
    """
    W-bottom reversal pattern in brick space:
    2+ down bricks, then 1+ up, then 1+ down (higher low), then current up.
    Classic double-bottom / W-pattern.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values.astype(float)

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 10

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if in_pos:
            if not up:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue

        # Pattern: ...DD U D U (current)
        # i = current up, i-1 = down, i-2 = up, i-3,i-4 = down
        if not up:
            continue
        if i < 5:
            continue

        # Minimal W: down, down, up, down, up(current)
        if (not brick_up[i-4] and not brick_up[i-3]
                and brick_up[i-2] and not brick_up[i-1] and up):
            # Higher low check: second trough (i-1 close) > first trough (i-3 close)
            if close[i-1] > close[i-4]:
                entry[i] = True
                in_pos = True
                last_bar = i

    return entry, exit_


def _gen_di_spread(df, df_htf, cooldown, gate):
    """
    Directional intensity: (+DI - -DI) > threshold on up brick.
    Not just a crossover — requires strong directional separation.
    Tests threshold = 15 (strong trend).
    """
    n = len(df)
    brick_up = df["brick_up"].values
    plus_di = df["plus_di"].values
    minus_di = df["minus_di"].values
    DI_THRESHOLD = 15.0

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
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            continue

        spread = plus_di[i] - minus_di[i]
        if up and spread > DI_THRESHOLD:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


def _gen_mtf_confirm(df, df_htf, cooldown, gate):
    """
    Multi-timeframe confirmation:
    LTF RSI > 50 AND HTF RSI > 50 AND LTF supertrend bullish, on up brick.
    Both timeframes must agree on bullish momentum.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    ltf_rsi = df["rsi"].values
    ltf_st = df["st_dir"].values

    # Align HTF RSI to LTF
    htf_rsi = _align_htf_to_ltf(df, df_htf, "rsi")

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
        if np.isnan(ltf_rsi[i]) or np.isnan(htf_rsi[i]) or np.isnan(ltf_st[i]):
            continue

        if up and ltf_rsi[i] > 50 and htf_rsi[i] > 50 and ltf_st[i] > 0:
            entry[i] = True
            in_pos = True
            last_bar = i

    return entry, exit_


GENERATORS = {
    "combo_stack":  _gen_combo_stack,
    "regime_adapt": _gen_regime_adapt,
    "accel":        _gen_accel,
    "htf_entry":    _gen_htf_entry,
    "dip_buy":      _gen_dip_buy,
    "score":        _gen_score,
    "compress":     _gen_compress,
    "w_bottom":     _gen_w_bottom,
    "di_spread":    _gen_di_spread,
    "mtf_confirm":  _gen_mtf_confirm,
}


# -- Worker --------------------------------------------------------------------

def _sweep_strategy(strategy_name):
    label = strategy_name.upper()
    print(f"  [{label}] Loading data...", flush=True)

    df_ltf = _load_ltf_data()
    df_htf = _load_htf_data()

    gen_fn = GENERATORS[strategy_name]

    gate_configs = [
        (False, 0,  "no_gates"),
        (True,  0,  "gates_only"),
        (True,  35, "gates+htf35"),
    ]

    gates = {}
    for use_g, htf_t, gname in gate_configs:
        gates[gname] = _compute_gates(df_ltf, df_htf, use_g, htf_t)

    results = []
    total = len(gate_configs) * len(COOLDOWNS)
    done = 0

    for _, htf_t, gname in gate_configs:
        gate = gates[gname]
        for cd in COOLDOWNS:
            e, x = gen_fn(df_ltf, df_htf, cd, gate)

            is_r  = _run_backtest(df_ltf, e, x, IS_START, IS_END)
            oos_r = _run_backtest(df_ltf, e, x, OOS_START, OOS_END)

            is_pf = is_r["pf"]
            oos_pf = oos_r["pf"]
            decay = ((oos_pf - is_pf) / is_pf * 100) \
                    if is_pf > 0 and not math.isinf(is_pf) else float("nan")

            results.append({
                "strategy":   strategy_name,
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
                  f"OOS PF={pf_s:>8} T={oos_r['trades']:>4} WR={oos_r['wr']:>5.1f}%",
                  flush=True)

    print(f"  [{label}] Done -- {len(results)} results", flush=True)
    return results


# -- Summary -------------------------------------------------------------------

def _summarize(all_results):
    print(f"\n{'='*105}")
    print("  BTC Phase 10: Novel Strategy Concepts (Long Only)")
    print(f"{'='*105}")

    print(f"\n  {'Strategy':<16} {'Best Gates':<15} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8} {'DD%':>7}")
    print(f"  {'-'*100}")

    for sname in STRATEGIES:
        subset = [r for r in all_results if r["strategy"] == sname and r["oos_trades"] >= 5]
        if not subset:
            print(f"  {sname:<16} (no viable results)")
            continue
        best = max(subset, key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6)
        pf_s = f"{best['oos_pf']:.2f}" if not math.isinf(best['oos_pf']) else "inf"
        is_pf_s = f"{best['is_pf']:.2f}" if not math.isinf(best['is_pf']) else "inf"
        print(f"  {sname:<16} {best['gates']:<15} {best['cooldown']:>3} | "
              f"{is_pf_s:>7} {best['is_trades']:>4} | "
              f"{pf_s:>8} {best['oos_trades']:>4} {best['oos_wr']:>5.1f}% "
              f"{best['oos_net']:>8.2f} {best['oos_dd']:>6.2f}%")

    # Full results table
    all_viable = [r for r in all_results if r["oos_trades"] >= 5]
    all_viable.sort(key=lambda r: r["oos_pf"] if not math.isinf(r["oos_pf"]) else 1e6, reverse=True)

    print(f"\n{'='*105}")
    print("  All Results Ranked (OOS trades >= 5)")
    print(f"{'='*105}")
    print(f"  {'Strategy':<16} {'Gates':<15} {'cd':>3} | {'IS PF':>7} {'T':>4} | "
          f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8} {'DD%':>7}")
    print(f"  {'-'*100}")
    for r in all_viable[:30]:
        pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
        is_pf_s = f"{r['is_pf']:.2f}" if not math.isinf(r['is_pf']) else "inf"
        print(f"  {r['strategy']:<16} {r['gates']:<15} {r['cooldown']:>3} | "
              f"{is_pf_s:>7} {r['is_trades']:>4} | "
              f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% "
              f"{r['oos_net']:>8.2f} {r['oos_dd']:>6.2f}%")

    # High frequency
    high_trade = [r for r in all_results if r["oos_pf"] >= 5 and r["oos_trades"] >= 20]
    high_trade.sort(key=lambda r: r["oos_trades"], reverse=True)
    if high_trade:
        print(f"\n  Top by Trade Count (PF >= 5, trades >= 20):")
        print(f"  {'Strategy':<16} {'Gates':<15} {'cd':>3} | "
              f"{'OOS PF':>8} {'T':>4} {'WR%':>6} {'Net':>8}")
        print(f"  {'-'*75}")
        for r in high_trade[:15]:
            pf_s = f"{r['oos_pf']:.2f}" if not math.isinf(r['oos_pf']) else "inf"
            print(f"  {r['strategy']:<16} {r['gates']:<15} {r['cooldown']:>3} | "
                  f"{pf_s:>8} {r['oos_trades']:>4} {r['oos_wr']:>5.1f}% {r['oos_net']:>8.2f}")

    print(f"\n  --- Reference ---")
    print(f"  BTC003 OOS: PF=49.27, 76t, WR=76.3%")
    print(f"  Phase 9 C5_mixed1 2-of-5 htf35: PF=36.21, 60t, WR=73.3%")
    print(f"  Phase 9 C5_trend 3-of-5 htf35: PF=35.86, 59t, WR=74.6%")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / "ai_context" / "btc_phase10_novel_results.json"
    out_path.parent.mkdir(exist_ok=True)

    total = len(STRATEGIES) * 3 * len(COOLDOWNS)
    print("BTC Phase 10: Novel Strategy Concepts (Long Only)")
    print(f"  Strategies : {STRATEGIES}")
    print(f"  Gate combos: no_gates, gates_only, gates+htf35")
    print(f"  Cooldowns  : {COOLDOWNS}")
    print(f"  Total runs : {total}")
    print(f"  IS period  : {IS_START} -> {IS_END}")
    print(f"  OOS period : {OOS_START} -> {OOS_END}")
    print()

    all_results = []

    if args.no_parallel:
        for sname in STRATEGIES:
            all_results.extend(_sweep_strategy(sname))
    else:
        with ProcessPoolExecutor(max_workers=min(len(STRATEGIES), 6)) as pool:
            futures = {pool.submit(_sweep_strategy, s): s for s in STRATEGIES}
            for future in as_completed(futures):
                sname = futures[future]
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
