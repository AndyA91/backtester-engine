#!/usr/bin/env python3
"""
wizard_btc_sweep.py -- Test scraped Pine Wizard strategies on BTC $150 Renko

Ported strategies (long only, first-down-brick exit):
    ALPHA_TREND     ATR + RSI adaptive band crossover (KivancOzbilgic, 6.7k boosts)
    SSL_CHANNEL     SMA(high) vs SMA(low) direction flip (vdubus, 3.9k boosts)
    HHLL            Bollinger band offset breakout (HPotter, 2.5k boosts)
    MACD_RELOAD     MACD histogram zero cross with EMA (KivancOzbilgic, 7.5k boosts)
    WILDER_VOL      ATR-based SAR system (LucF/Wilder, 0.9k boosts)
    HALFTREND       Adaptive trend band with ATR channels (everget, 12.1k boosts)
    NRTR            Nick Rypock trailing reverse (everget, 9.9k boosts)

Gate combos: none, psar, adx25, psar_adx25
Cooldowns: 3, 5, 10, 20
Exit: first down brick (universal)

Usage:
    python renko/wizard_btc_sweep.py
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
OOS_DAYS   = 170
COMMISSION = 0.0046
CAPITAL    = 1000.0
QTY_VALUE  = 20

COOLDOWNS  = [3, 5, 10, 20]
GATE_MODES = ["none", "psar", "adx25", "psar_adx25"]


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
    if "adx25" in gate_mode:
        adx = df["adx"].values
        gate &= (np.isnan(adx) | (adx >= 25))
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


# ==============================================================================
# SIGNAL GENERATORS — ported from Pine Wizard strategies
# All long-only. Exit = first down brick. Gate applied externally.
# ==============================================================================

def _gen_alpha_trend(df, cooldown, gate, period=14, coeff=1.0):
    """
    AlphaTrend by KivancOzbilgic (6,722 boosts)
    Pine: ATR adaptive band. If RSI >= 50 -> upT (low - ATR*coeff), ratchets up.
          If RSI < 50 -> downT (high + ATR*coeff), ratchets down.
    Signal: AlphaTrend crosses above AlphaTrend[2] -> long.
    BTC has no volume, so we use RSI mode (novolumedata=true).
    """
    n = len(df)
    brick_up = df["brick_up"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    # Compute ATR as SMA of True Range (Pine's ta.sma(ta.tr, period))
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_sma = pd.Series(tr).rolling(period, min_periods=1).mean().values

    # RSI (use pre-computed, pre-shifted)
    rsi = df["rsi"].values

    # AlphaTrend computation
    alpha_trend = np.zeros(n)
    for i in range(1, n):
        upT = low[i] - atr_sma[i] * coeff
        downT = high[i] + atr_sma[i] * coeff

        rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50.0

        if rsi_val >= 50:
            alpha_trend[i] = max(upT, alpha_trend[i-1])
        else:
            alpha_trend[i] = min(downT, alpha_trend[i-1])

    # Signal: alpha_trend[i] crosses above alpha_trend[i-2]
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = max(period + 5, 30)

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        # Crossover: current > [i-2] and previous <= [i-3]
        if i >= 3 and alpha_trend[i] > alpha_trend[i-2] and alpha_trend[i-1] <= alpha_trend[i-3]:
            if brick_up[i]:
                entry[i] = True
                in_pos = True
                last_bar = i
    return entry, exit_


def _gen_ssl_channel(df, cooldown, gate, period=13):
    """
    SSL Channel by vdubus (3,939 boosts)
    Pine: smaHigh = SMA(high, period), smaLow = SMA(low, period)
          Hlv = close > smaHigh ? 1 : close < smaLow ? -1 : Hlv[1]
          Signal: Hlv flips from -1 to +1 -> long.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    sma_high = pd.Series(high).rolling(period, min_periods=1).mean().values
    sma_low = pd.Series(low).rolling(period, min_periods=1).mean().values

    hlv = np.zeros(n)
    for i in range(1, n):
        if close[i] > sma_high[i]:
            hlv[i] = 1
        elif close[i] < sma_low[i]:
            hlv[i] = -1
        else:
            hlv[i] = hlv[i-1]

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = period + 5

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        # SSL flip: Hlv goes from <= 0 to 1
        if hlv[i] == 1 and hlv[i-1] <= 0 and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_hhll(df, cooldown, gate, bb_len=29):
    """
    HHLL by HPotter (2,494 boosts)
    Pine: BB(hlc3, 29, 2) -> xHH, xLL. movevalue = (xHH - xLL)/2
          xLLM = xLL - movevalue. Long when low < xLLM[1] (mean reversion bounce).
    """
    n = len(df)
    brick_up = df["brick_up"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    hlc3 = (high + low + close) / 3.0
    s = pd.Series(hlc3)
    mid = s.rolling(bb_len, min_periods=1).mean().values
    std = s.rolling(bb_len, min_periods=1).std(ddof=0).values

    xHH = mid + 2.0 * std
    xLL = mid - 2.0 * std
    move = (xHH - xLL) / 2.0
    xLLM = xLL - move  # Extended lower band

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = bb_len + 5

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        # Long when price breaks below extended lower band (mean reversion bounce)
        if low[i] < xLLM[i-1] and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_macd_reload(df, cooldown, gate):
    """
    MACD ReLoaded by KivancOzbilgic (7,478 boosts)
    Pine: Uses various MA types for MACD. Core signal = histogram crosses above 0.
    We use standard EMA MACD (already pre-computed).
    Signal: macd_hist crosses from negative to positive -> long.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    hist = df["macd_hist"].values  # pre-shifted

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = 40

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        if np.isnan(hist[i]) or np.isnan(hist[i-1]):
            continue
        # Histogram zero cross: was negative, now positive
        if hist[i] > 0 and hist[i-1] <= 0 and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_wilder_vol(df, cooldown, gate, atr_len=9, arc_factor=1.8):
    """
    Volatility System by Wilder (LucF, 895 boosts)
    Pine: Arc = ATR(atr_len) * arc_factor
          SarHi = lowest(close, atr_len) + Arc
          Long when close crosses above SarHi[1]
    """
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    # Compute ATR manually (can't reuse pre-computed — different period)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(atr_len, min_periods=1).mean().values

    # SarHi = lowest(close, atr_len) + ATR * arc_factor
    lowest_close = pd.Series(close).rolling(atr_len, min_periods=1).min().values

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = atr_len + 5

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        arc = atr[i] * arc_factor
        sar_hi = lowest_close[i-1] + arc
        arc_prev = atr[i-1] * arc_factor if not np.isnan(atr[i-1]) else arc
        sar_hi_prev = lowest_close[max(0, i-2)] + arc_prev
        # Crossover: close crosses above SarHi
        if close[i] > sar_hi and close[i-1] <= sar_hi_prev and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_halftrend(df, cooldown, gate, amplitude=2):
    """
    HalfTrend by everget (12,106 boosts)
    Pine: Adaptive trend using ATR channels. Trend flips when price breaks
          opposite channel boundary.
    Simplified: track highest low / lowest high with ATR/2 buffer.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    atr_val = df["atr"].values  # pre-shifted

    trend = np.zeros(n, dtype=int)  # 0=up, 1=down
    ht_line = np.zeros(n)

    for i in range(amplitude + 1, n):
        atr_i = atr_val[i] if not np.isnan(atr_val[i]) else 0.0
        half_atr = atr_i / 2.0

        prev_trend = trend[i-1]

        if prev_trend == 0:  # Was up-trend
            max_low = np.max(low[max(0, i-amplitude):i+1])
            ht_line[i] = max(ht_line[i-1], max_low)
            if close[i] < ht_line[i] - half_atr:
                trend[i] = 1  # Flip to downtrend
                ht_line[i] = np.min(high[max(0, i-amplitude):i+1])
            else:
                trend[i] = 0
        else:  # Was down-trend
            min_high = np.min(high[max(0, i-amplitude):i+1])
            ht_line[i] = min(ht_line[i-1], min_high)
            if close[i] > ht_line[i] + half_atr:
                trend[i] = 0  # Flip to uptrend
                ht_line[i] = np.max(low[max(0, i-amplitude):i+1])
            else:
                trend[i] = 1

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = amplitude + 10

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        # Trend flips from down (1) to up (0)
        if trend[i] == 0 and trend[i-1] == 1 and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


def _gen_nrtr(df, cooldown, gate, period=40, mult=2.0):
    """
    NRTR (Nick Rypock Trailing Reverse) by everget (9,887 boosts)
    Pine: Tracks trend using highest/lowest close with channel-based reverse.
    Long when trend flips bullish.
    """
    n = len(df)
    brick_up = df["brick_up"].values
    close = df["Close"].values

    nrtr_trend = np.zeros(n, dtype=int)  # 1=up, -1=down
    hp = close[0]
    lp = close[0]

    for i in range(1, n):
        hi_n = np.max(close[max(0, i-period+1):i+1])
        lo_n = np.min(close[max(0, i-period+1):i+1])
        channel = (hi_n - lo_n) * mult / 100.0 if hi_n > lo_n else 0.0

        prev = nrtr_trend[i-1]

        if prev >= 0:  # Was bullish
            hp = max(hp, close[i])
            reverse = hp - channel
            if close[i] <= reverse:
                nrtr_trend[i] = -1
                lp = close[i]
            else:
                nrtr_trend[i] = 1
        else:  # Was bearish
            lp = min(lp, close[i])
            reverse = lp + channel
            if close[i] >= reverse:
                nrtr_trend[i] = 1
                hp = close[i]
            else:
                nrtr_trend[i] = -1

    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    in_pos = False
    last_bar = -999_999
    warmup = period + 5

    for i in range(warmup, n):
        if in_pos:
            if not brick_up[i]:
                exit_[i] = True
                in_pos = False
            continue
        if not gate[i] or (i - last_bar) < cooldown:
            continue
        # Trend flips from -1 to 1
        if nrtr_trend[i] == 1 and nrtr_trend[i-1] == -1 and brick_up[i]:
            entry[i] = True
            in_pos = True
            last_bar = i
    return entry, exit_


# ==============================================================================
# STRATEGY REGISTRY
# ==============================================================================

STRATEGIES = {
    "ALPHA_TREND": {
        "fn": _gen_alpha_trend,
        "params": {"period": [10, 14, 20], "coeff": [0.5, 1.0, 1.5, 2.0]},
        "desc": "AlphaTrend ATR+RSI adaptive band (KivancOzbilgic 6.7k)",
    },
    "SSL_CHANNEL": {
        "fn": _gen_ssl_channel,
        "params": {"period": [8, 13, 20, 30]},
        "desc": "SSL Channel SMA high/low flip (vdubus 3.9k)",
    },
    "HHLL": {
        "fn": _gen_hhll,
        "params": {"bb_len": [15, 20, 29, 40]},
        "desc": "HHLL BB offset breakout (HPotter 2.5k)",
    },
    "MACD_RELOAD": {
        "fn": _gen_macd_reload,
        "params": {},
        "desc": "MACD ReLoaded histogram zero cross (KivancOzbilgic 7.5k)",
    },
    "WILDER_VOL": {
        "fn": _gen_wilder_vol,
        "params": {"atr_len": [7, 9, 14], "arc_factor": [1.5, 1.8, 2.5, 3.0]},
        "desc": "Wilder Volatility System ATR SAR (LucF 0.9k)",
    },
    "HALFTREND": {
        "fn": _gen_halftrend,
        "params": {"amplitude": [2, 3, 5]},
        "desc": "HalfTrend adaptive trend band (everget 12.1k)",
    },
    "NRTR": {
        "fn": _gen_nrtr,
        "params": {"period": [20, 40, 60], "mult": [1.5, 2.0, 3.0]},
        "desc": "NRTR trailing reverse (everget 9.9k)",
    },
}


# -- Worker function -----------------------------------------------------------

_worker_cache = {}

def _worker_init():
    _worker_cache["df"] = _load_data()

def _run_single(task):
    df = _worker_cache["df"]
    strat_name = task["strategy"]
    strat = STRATEGIES[strat_name]
    fn = strat["fn"]
    cd = task["cooldown"]
    gate_mode = task["gate"]
    params = task.get("params", {})

    gate = _compute_gate(df, gate_mode)

    try:
        entry, exit_ = fn(df, cooldown=cd, gate=gate, **params)
    except Exception as e:
        return {**task, "error": str(e)}

    is_kpis = _run_bt(df, entry, exit_, IS_START, IS_END)
    oos_kpis = _run_bt(df, entry, exit_, OOS_START, OOS_END)
    oos_kpis["tpd"] = round(oos_kpis["trades"] / OOS_DAYS, 2)

    return {**task, "is": is_kpis, "oos": oos_kpis}


# -- Build task list -----------------------------------------------------------

def _build_tasks():
    import itertools
    tasks = []
    for strat_name, strat in STRATEGIES.items():
        param_grid = strat["params"]
        if param_grid:
            keys = list(param_grid.keys())
            combos = list(itertools.product(*[param_grid[k] for k in keys]))
        else:
            keys = []
            combos = [()]

        for combo in combos:
            params = dict(zip(keys, combo))
            for cd in COOLDOWNS:
                for gate in GATE_MODES:
                    tasks.append({
                        "strategy": strat_name,
                        "cooldown": cd,
                        "gate": gate,
                        "params": params,
                    })
    return tasks


# -- Main sweep ----------------------------------------------------------------

def main():
    tasks = _build_tasks()
    print(f"Wizard BTC Sweep: {len(tasks)} combos across {len(STRATEGIES)} strategies")
    print(f"Strategies: {', '.join(STRATEGIES.keys())}")
    print(f"Workers: {MAX_WORKERS}\n")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(_run_single, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            results.append(r)
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} complete...")

    # -- Sort by OOS PF, filter to min 10 trades --------------------------------
    valid = [r for r in results if "error" not in r and r["oos"]["trades"] >= 10]
    valid.sort(key=lambda x: x["oos"]["pf"], reverse=True)

    # -- Print top 30 -------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"TOP 30 BY OOS PF (min 10 trades)")
    print(f"{'='*100}")
    print(f"{'Strategy':<16} {'Params':<30} {'Gate':<12} {'CD':>3}  "
          f"{'IS_PF':>7} {'IS_T':>5} {'IS_WR':>6}  "
          f"{'OOS_PF':>7} {'OOS_T':>5} {'OOS_WR':>6} {'t/d':>5} {'OOS_Net':>9}")
    print("-" * 100)

    for r in valid[:30]:
        p = r["params"]
        param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
        print(f"{r['strategy']:<16} {param_str:<30} {r['gate']:<12} {r['cooldown']:>3}  "
              f"{r['is']['pf']:>7.2f} {r['is']['trades']:>5} {r['is']['wr']:>5.1f}%  "
              f"{r['oos']['pf']:>7.2f} {r['oos']['trades']:>5} {r['oos']['wr']:>5.1f}% "
              f"{r['oos']['tpd']:>5.1f} {r['oos']['net']:>9.2f}")

    # -- Save all results ----------------------------------------------------------
    out_file = ROOT / "ai_context" / "wizard_btc_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_file}")

    # -- Summary per strategy -------------------------------------------------------
    print(f"\n{'='*80}")
    print("BEST CONFIG PER STRATEGY (by OOS PF, min 10 trades)")
    print(f"{'='*80}")
    for strat_name in STRATEGIES:
        strat_results = [r for r in valid if r["strategy"] == strat_name]
        if strat_results:
            best = strat_results[0]
            p = best["params"]
            param_str = " ".join(f"{k}={v}" for k, v in p.items()) if p else "-"
            print(f"  {strat_name:<16} {param_str:<25} gate={best['gate']:<12} cd={best['cooldown']:<3} "
                  f"OOS: PF={best['oos']['pf']:.2f} T={best['oos']['trades']} "
                  f"WR={best['oos']['wr']:.1f}% t/d={best['oos']['tpd']:.1f}")
        else:
            print(f"  {strat_name:<16} -- no qualifying results --")


if __name__ == "__main__":
    main()
