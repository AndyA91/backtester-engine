"""
Phase 19 — EURUSD 0.0006 Trend Continuation Methods (3 angles, big sweep)

Purpose
-------
Phase 17 + 18 conclusively killed STREAK-BASED trend entries on EURUSD 0.0006
(naked streak, band-walk filtered streak, bb_mid slope filter on R029 — all
HOLDOUT WR 31-44%). The user correctly pointed out this only kills ONE family
of trend mechanism — direct brick-direction reading. Other mechanisms have
NOT been tested:

  - Pullback entries (mean-reversion-style entry inside a confirmed trend)
  - Structure breakouts (Donchian-style price-level breaks, not streak)
  - KAMA crossovers (continuous-variable trigger, not direction-counting)

Phase 19 tests all three in one big sweep on the same brick (0.0006), same
exit (first opposing brick), same 3-way TRAIN/VALIDATE/HOLDOUT discipline.
Each angle is structurally distinct from Phase 17/18 mechanisms.

ANGLE A — Pullback to bb_mid in confirmed trend
    Logic: confirm uptrend (bb_mid rising AND price above bb_mid for N bricks),
    wait for a brief pullback (M consecutive opposing bricks), enter on the
    first resumption brick. Symmetric for downtrends.
    Hypothesis: this USES the brick's MR-friendliness instead of fighting it
    — enters during the temporary MR move that constitutes a pullback inside
    a larger trend.

ANGLE B — Range Breakout (Donchian-style)
    Logic: enter when current close exceeds the highest close (or high) over
    the last N bricks. No streak requirement. Single-bar trigger.
    Hypothesis: structural price-level breaks fire on the FIRST brick that
    breaches a prior level — fundamentally different from streak's
    "wait for N consecutive same-direction bricks" logic.

ANGLE C — KAMA Crossover
    Logic: enter when close crosses above/below KAMA, optionally filtered by
    KAMA slope direction. KAMA's adaptive smoothing is supposed to ignore
    noise and fire on real moves.
    Hypothesis: continuous-variable entry trigger (position relative to
    smoothed line) is mechanistically different from event-counting (streak).

Data splits (per R22)
---------------------
  TRAIN     2023-01-02 -> 2025-09-30
  VALIDATE  2025-10-01 -> 2025-12-31
  HOLDOUT   2026-01-01 -> 2026-03-25  (sealed during sweep)

NOTE: This is the THIRD sweep using the same HOLDOUT window. Family-wise
error rate is now non-trivial. If this sweep produces a "passes everything"
result, treat it as a candidate-for-validation, not a deployable strategy
— and refresh data + re-test on a fresh window before any deploy.

Decision rules (locked BEFORE viewing HOLDOUT)
----------------------------------------------
Standalone strategy (any angle):
  - HOLDOUT PF >= 4.0
  - HOLDOUT WR >= 50%
  - HOLDOUT trades >= 20
  - HOLDOUT net profit > 0

Finalist filter (TRAIN -> VALIDATE):
  - VALIDATE trades >= 12  (lower than Phase 18 due to expected high selectivity)
  - WR delta in [-5, +15]pp

Compute budget
--------------
  Angle A: 4 × 3 × 4 × 4 × 3 = 576 combos
  Angle B: 6 × 2 × 3 × 2     = 72 combos
  Angle C: 2 × 4 × 3         = 24 combos
  Total: 672 combos × 2 windows = 1,344 backtests + baseline + holdouts
  Wallclock estimate: ~70s on 20 workers

Usage
-----
    python renko/phase19_eurusd_trend_methods.py
"""

import contextlib
import io
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Force UTF-8 stdout on Windows so box-drawing chars don't crash the print
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.config import MAX_WORKERS

RENKO_FILE = "OANDA_EURUSD, 1S renko 0.0006.csv"

# ── Data splits (3-way per R22) ──────────────────────────────────────────────
TRAIN_START    = "2023-01-02"
TRAIN_END      = "2025-09-30"
VALIDATE_START = "2025-10-01"
VALIDATE_END   = "2025-12-31"
HOLDOUT_START  = "2026-01-01"
HOLDOUT_END    = "2026-03-26"

# ── Backtest config (matches R029 Pine for clean comparability) ─────────────
BACKTEST_CONFIG = BacktestConfig(
    initial_capital=1000.0,
    commission_pct=0.0046,
    slippage_ticks=0,
    qty_type="fixed",
    qty_value=1000.0,
    pyramiding=1,
    start_date="2000-01-01",
    end_date="2099-12-31",
    take_profit_pct=0.0,
    stop_loss_pct=0.0,
)

OUTPUT_FILE = ROOT / "ai_context" / "phase19_eurusd_trend_methods_results.json"

# ── R029 baseline params (for context — same as Phase 18) ───────────────────
R029_PARAMS = {
    "band_thresh": 0.20,
    "cooldown":    2,
    "rsi_thresh":  45,
    "adx_max":     25,
    "vol_max":     1.5,
}

# ── Param grids ──────────────────────────────────────────────────────────────

ANGLE_A_GRID = {
    "trend_min_bricks": [5, 10, 15, 20],
    "slope_lookback":   [10, 20, 40],
    "slope_threshold":  [0.0, 0.0003, 0.0006, 0.0010],
    "pullback_min":     [1, 2, 3, 5],
    "cooldown":         [5, 10, 20],
}

ANGLE_B_GRID = {
    "breakout_lookback": [10, 15, 20, 30, 40, 60],
    "use_high_low":      [False, True],   # False = close-break, True = high/low-break
    "cooldown":          [5, 10, 20],
    "require_brick_dir": [False, True],
}

ANGLE_C_GRID = {
    "entry_type":   ["cross", "above_with_slope"],
    "slope_filter": [0.0, 0.00005, 0.0001, 0.0002],
    "cooldown":     [5, 10, 20],
}

# ── Filters / thresholds ─────────────────────────────────────────────────────
MIN_TRADES_PER_WINDOW       = 5
MIN_VALIDATE_TRADES_FINAL   = 12
MAX_TRAIN_VAL_WR_DROP       = 5.0
MAX_TRAIN_VAL_WR_JUMP       = 15.0
TOP_N_FINALISTS             = 5

# ── Holdout decision rules (locked BEFORE viewing holdout) ──────────────────
RULES = {
    "min_pf":         4.0,
    "min_wr":         50.0,
    "min_trades":     20,
    "min_net_profit": 0.0,
}


# ─── R029 baseline generator (same code as Phase 18, for context only) ─────


def gen_r029_baseline(df):
    """R029 with locked Pine params, no slope filter. For context only."""
    brick_up   = df["brick_up"].values
    pct_b      = df["bb_pct_b"].values
    rsi        = df["rsi"].values
    adx        = df["adx"].values
    vol_ratio  = df["vol_ratio"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = 30
    last_trade_bar = -999_999
    pos = 0

    upper_band = 1.0 - R029_PARAMS["band_thresh"]
    lower_band = R029_PARAMS["band_thresh"]
    rsi_long_t  = R029_PARAMS["rsi_thresh"]
    rsi_short_t = 100.0 - R029_PARAMS["rsi_thresh"]
    cd          = R029_PARAMS["cooldown"]
    adx_max     = R029_PARAMS["adx_max"]
    vol_max     = R029_PARAMS["vol_max"]

    for i in range(warmup, n):
        up = bool(brick_up[i])
        if pos == 1 and not up:
            long_exit[i] = True; pos = 0
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0
        if pos != 0:
            continue
        if (i - last_trade_bar) < cd:
            continue
        if not np.isnan(vol_ratio[i]) and vol_ratio[i] > vol_max:
            continue
        if not np.isnan(adx[i]) and adx[i] > adx_max:
            continue
        if np.isnan(pct_b[i]):
            continue
        if pct_b[i] <= lower_band and up:
            if not np.isnan(rsi[i]) and rsi[i] >= rsi_long_t:
                continue
            long_entry[i] = True; pos = 1; last_trade_bar = i
        elif pct_b[i] >= upper_band and not up:
            if not np.isnan(rsi[i]) and rsi[i] <= rsi_short_t:
                continue
            short_entry[i] = True; pos = -1; last_trade_bar = i

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ─── ANGLE A — Pullback to bb_mid in confirmed trend ───────────────────────


def gen_pullback(df, trend_min_bricks, slope_lookback, slope_threshold,
                 pullback_min, cooldown):
    """Pullback entry: confirm trend, wait for pullback, enter on resumption.

    Trend up: bb_mid[i] - bb_mid[i-slope_lookback] > slope_threshold
              AND close > bb_mid for at least `trend_min_bricks` of the
              last `trend_min_bricks` bars
    Pullback (in uptrend): at least `pullback_min` consecutive down bricks
    Entry: first up brick after the pullback while uptrend still confirmed
    Exit: first opposing brick (R1)
    """
    brick_up = df["brick_up"].values
    close    = df["Close"].values
    bb_mid   = df["bb_mid"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(slope_lookback + trend_min_bricks + 5, 30)
    last_exit_bar = -999_999
    pos = 0
    # Pullback state: number of consecutive opposing bricks just observed
    pullback_count = 0
    pullback_dir = 0  # +1 if pullback inside uptrend, -1 inside downtrend

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # Exit on first opposing brick
        if pos == 1 and not up:
            long_exit[i] = True; pos = 0; last_exit_bar = i
            pullback_count = 0; pullback_dir = 0
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0; last_exit_bar = i
            pullback_count = 0; pullback_dir = 0

        if pos != 0:
            continue

        # Trend confirmation (need lookback)
        mid_now  = bb_mid[i]
        mid_prev = bb_mid[i - slope_lookback]
        if np.isnan(mid_now) or np.isnan(mid_prev):
            continue
        slope = mid_now - mid_prev

        # Count how many of the last trend_min_bricks bars had close > or < bb_mid
        recent_close = close[i - trend_min_bricks:i]
        recent_mid   = bb_mid[i - trend_min_bricks:i]
        if np.any(np.isnan(recent_mid)):
            continue
        above_count = int(np.sum(recent_close > recent_mid))
        below_count = int(np.sum(recent_close < recent_mid))

        uptrend_confirmed = (slope > slope_threshold and
                             above_count >= trend_min_bricks - 1)
        downtrend_confirmed = (slope < -slope_threshold and
                               below_count >= trend_min_bricks - 1)

        # Track pullback state
        if uptrend_confirmed:
            if not up:
                # Down brick during uptrend = building pullback
                if pullback_dir == 1:
                    pullback_count += 1
                else:
                    pullback_count = 1
                    pullback_dir = 1
            else:
                # Up brick during uptrend
                if pullback_dir == 1 and pullback_count >= pullback_min:
                    # Resumption — enter long
                    if (i - last_exit_bar) >= cooldown:
                        long_entry[i] = True; pos = 1; last_exit_bar = i
                    pullback_count = 0; pullback_dir = 0
                else:
                    pullback_count = 0; pullback_dir = 0
        elif downtrend_confirmed:
            if up:
                # Up brick during downtrend = building pullback
                if pullback_dir == -1:
                    pullback_count += 1
                else:
                    pullback_count = 1
                    pullback_dir = -1
            else:
                # Down brick during downtrend
                if pullback_dir == -1 and pullback_count >= pullback_min:
                    if (i - last_exit_bar) >= cooldown:
                        short_entry[i] = True; pos = -1; last_exit_bar = i
                    pullback_count = 0; pullback_dir = 0
                else:
                    pullback_count = 0; pullback_dir = 0
        else:
            # No trend — reset pullback state
            pullback_count = 0; pullback_dir = 0

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ─── ANGLE B — Range Breakout (Donchian-style) ─────────────────────────────


def gen_breakout(df, breakout_lookback, use_high_low, cooldown,
                 require_brick_dir):
    """Range breakout: enter when current close (or high/low) exceeds the
    extreme of the last `breakout_lookback` bricks.

    LONG  : close[i] > max(close[i-breakout_lookback:i])  (or high if use_high_low)
    SHORT : symmetric
    Optional: require entry brick direction to match (up brick for long)
    Exit: first opposing brick (R1)
    """
    brick_up = df["brick_up"].values
    close    = df["Close"].values
    high     = df["High"].values
    low      = df["Low"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = max(breakout_lookback + 2, 30)
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if pos == 1 and not up:
            long_exit[i] = True; pos = 0; last_exit_bar = i
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0; last_exit_bar = i

        if pos != 0:
            continue
        if (i - last_exit_bar) < cooldown:
            continue

        if use_high_low:
            upper_extreme = float(np.max(high[i - breakout_lookback:i]))
            lower_extreme = float(np.min(low[i - breakout_lookback:i]))
            current_up = high[i]
            current_dn = low[i]
        else:
            upper_extreme = float(np.max(close[i - breakout_lookback:i]))
            lower_extreme = float(np.min(close[i - breakout_lookback:i]))
            current_up = close[i]
            current_dn = close[i]

        long_break  = current_up > upper_extreme
        short_break = current_dn < lower_extreme

        if require_brick_dir:
            long_break  = long_break  and up
            short_break = short_break and (not up)

        if long_break:
            long_entry[i] = True; pos = 1; last_exit_bar = i
        elif short_break:
            short_entry[i] = True; pos = -1; last_exit_bar = i

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ─── ANGLE C — KAMA Crossover ──────────────────────────────────────────────


def gen_kama(df, entry_type, slope_filter, cooldown):
    """KAMA crossover entry.

    entry_type:
      "cross"            : LONG when close[i] > kama[i] AND close[i-1] <= kama[i-1]
      "above_with_slope" : LONG when close > kama AND brick_up AND kama_slope > 0
    slope_filter > 0     : also require |kama_slope| >= slope_filter
    Exit: first opposing brick (R1)
    """
    brick_up   = df["brick_up"].values
    close      = df["Close"].values
    kama       = df["kama"].values
    kama_slope = df["kama_slope"].values
    n = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = 30
    last_exit_bar = -999_999
    pos = 0

    for i in range(warmup, n):
        up = bool(brick_up[i])

        if pos == 1 and not up:
            long_exit[i] = True; pos = 0; last_exit_bar = i
        elif pos == -1 and up:
            short_exit[i] = True; pos = 0; last_exit_bar = i

        if pos != 0:
            continue
        if (i - last_exit_bar) < cooldown:
            continue

        k_now = kama[i]; k_prev = kama[i - 1]
        c_now = close[i]; c_prev = close[i - 1]
        ks = kama_slope[i]
        if np.isnan(k_now) or np.isnan(k_prev) or np.isnan(ks):
            continue

        if entry_type == "cross":
            long_sig  = (c_now > k_now) and (c_prev <= k_prev)
            short_sig = (c_now < k_now) and (c_prev >= k_prev)
        else:  # above_with_slope
            long_sig  = (c_now > k_now) and up and (ks > 0)
            short_sig = (c_now < k_now) and (not up) and (ks < 0)

        if slope_filter > 0:
            if long_sig  and ks <  slope_filter: long_sig  = False
            if short_sig and ks > -slope_filter: short_sig = False

        if long_sig:
            long_entry[i] = True; pos = 1; last_exit_bar = i
        elif short_sig:
            short_entry[i] = True; pos = -1; last_exit_bar = i

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df


# ── Backtest harness ────────────────────────────────────────────────────────


def run_window(df_signals):
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_signals, BACKTEST_CONFIG)
    n_trades = kpis.get("total_trades", 0)
    if n_trades < MIN_TRADES_PER_WINDOW:
        return None
    trades = kpis.get("trades", [])
    closed = [t for t in trades if t.exit_date is not None and t.pnl is not None]
    worst_loss = min((t.pnl for t in closed), default=0.0)
    n_long  = sum(1 for t in trades if t.direction == "long"  and t.exit_date is not None)
    n_short = sum(1 for t in trades if t.direction == "short" and t.exit_date is not None)
    return {
        "trades":     n_trades,
        "wr":         round(kpis.get("win_rate", 0), 2),
        "pf":         round(kpis.get("profit_factor", 0), 2),
        "net_profit": round(kpis.get("net_profit", 0), 2),
        "max_dd":     round(kpis.get("max_drawdown", 0), 2),
        "avg_trade":  round(kpis.get("avg_trade", 0), 2),
        "avg_wl":     round(kpis.get("avg_win_loss_ratio", 0), 2),
        "worst_loss": round(worst_loss, 2),
        "n_long":     n_long,
        "n_short":    n_short,
    }


def _angle_a_worker(params, df_train_pkl, df_val_pkl):
    import pickle
    df_train = pickle.loads(df_train_pkl)
    df_val   = pickle.loads(df_val_pkl)
    tr = run_window(gen_pullback(df_train, **params))
    vl = run_window(gen_pullback(df_val,   **params))
    if tr is None or vl is None:
        return None
    return {"params": params, "train": tr, "validate": vl}


def _angle_b_worker(params, df_train_pkl, df_val_pkl):
    import pickle
    df_train = pickle.loads(df_train_pkl)
    df_val   = pickle.loads(df_val_pkl)
    tr = run_window(gen_breakout(df_train, **params))
    vl = run_window(gen_breakout(df_val,   **params))
    if tr is None or vl is None:
        return None
    return {"params": params, "train": tr, "validate": vl}


def _angle_c_worker(params, df_train_pkl, df_val_pkl):
    import pickle
    df_train = pickle.loads(df_train_pkl)
    df_val   = pickle.loads(df_val_pkl)
    tr = run_window(gen_kama(df_train, **params))
    vl = run_window(gen_kama(df_val,   **params))
    if tr is None or vl is None:
        return None
    return {"params": params, "train": tr, "validate": vl}


# ── Finalist / holdout ──────────────────────────────────────────────────────


def pick_finalists(results):
    eligible = []
    for r in results:
        tr = r["train"]; vl = r["validate"]
        if vl["trades"] < MIN_VALIDATE_TRADES_FINAL:
            continue
        wr_delta = vl["wr"] - tr["wr"]
        if wr_delta < -MAX_TRAIN_VAL_WR_DROP:
            continue
        if wr_delta > MAX_TRAIN_VAL_WR_JUMP:
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r["validate"]["pf"], reverse=True)
    return eligible[:TOP_N_FINALISTS]


def evaluate(holdout):
    failures = []
    if holdout["trades"] < RULES["min_trades"]:
        failures.append(f"T {holdout['trades']} < {RULES['min_trades']}")
    if holdout["pf"] < RULES["min_pf"]:
        failures.append(f"PF {holdout['pf']} < {RULES['min_pf']}")
    if holdout["wr"] < RULES["min_wr"]:
        failures.append(f"WR {holdout['wr']}% < {RULES['min_wr']}%")
    if holdout["net_profit"] < RULES["min_net_profit"]:
        failures.append(f"Net {holdout['net_profit']} < {RULES['min_net_profit']}")
    return (len(failures) == 0, failures)


def run_sweep(name, grid, worker_fn, gen_fn, df_train, df_validate, df_holdout,
              df_train_pkl, df_val_pkl):
    """Run one angle's sweep + finalists + holdout. Returns (results, holdouts)."""
    keys = list(grid.keys())
    vals = list(grid.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    print(f"\n{'=' * 120}")
    print(f"  {name}")
    print(f"{'=' * 120}")
    print(f"  {len(combos)} combos x 2 windows = {len(combos)*2} backtests")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(worker_fn, p, df_train_pkl, df_val_pkl): p for p in combos}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)
    print(f"  {len(results)} viable in {time.time()-t0:.1f}s")

    if not results:
        return [], []

    # Top by VAL PF
    sorted_pf = sorted(results, key=lambda r: r["validate"]["pf"], reverse=True)[:15]
    print(f"\n  TOP 15 BY VALIDATE PF:")
    for r in sorted_pf:
        p = r["params"]; tr = r["train"]; vl = r["validate"]
        params_str = " ".join(f"{k}={v}" for k, v in p.items())
        print(f"    [{params_str}]")
        print(f"      TR PF={tr['pf']:6.2f} T={tr['trades']:4} WR={tr['wr']:5.1f}% Net=${tr['net_profit']:8.2f} | "
              f"VL PF={vl['pf']:6.2f} T={vl['trades']:3} WR={vl['wr']:5.1f}% Net=${vl['net_profit']:7.2f}")

    # Top by VAL net
    sorted_net = sorted(results, key=lambda r: r["validate"]["net_profit"], reverse=True)[:10]
    print(f"\n  TOP 10 BY VALIDATE NET PROFIT:")
    for r in sorted_net:
        p = r["params"]; vl = r["validate"]
        params_str = " ".join(f"{k}={v}" for k, v in p.items())
        print(f"    Net=${vl['net_profit']:7.2f} T={vl['trades']:3} WR={vl['wr']:5.1f}% PF={vl['pf']:5.2f}  [{params_str}]")

    finalists = pick_finalists(results)
    print(f"\n  FINALISTS: {len(finalists)}  (filter: VAL T>={MIN_VALIDATE_TRADES_FINAL}, "
          f"WR delta in [-{MAX_TRAIN_VAL_WR_DROP},+{MAX_TRAIN_VAL_WR_JUMP}]pp)")

    holdouts = []
    if finalists:
        for f in finalists:
            h = run_window(gen_fn(df_holdout, **f["params"]))
            if h is None:
                h = {"trades": 0, "wr": 0, "pf": 0, "net_profit": 0,
                     "max_dd": 0, "avg_trade": 0, "avg_wl": 0, "worst_loss": 0,
                     "n_long": 0, "n_short": 0}
            holdouts.append({"params": f["params"], "train": f["train"],
                             "validate": f["validate"], "holdout": h})
        print(f"\n  HOLDOUT RESULTS (decision rules: PF>=4, WR>=50%, T>=20, Net>0):")
        for i, r in enumerate(holdouts, 1):
            p = r["params"]; h = r["holdout"]
            ok, fails = evaluate(h)
            verdict = "PASS" if ok else "REJECT"
            params_str = " ".join(f"{k}={v}" for k, v in p.items())
            print(f"  {i}. {verdict} [{params_str}]")
            print(f"     PF={h['pf']:6.2f} T={h['trades']:3} WR={h['wr']:5.1f}% "
                  f"Net=${h['net_profit']:8.2f} L/S={h['n_long']}/{h['n_short']} "
                  f"WLoss=${h['worst_loss']:6.2f}")
            for fmsg in fails:
                print(f"      x {fmsg}")

    return results, holdouts


def main():
    import pickle

    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    print(f"  {len(df):,} bricks loaded, {df.index[0]} -> {df.index[-1]}")

    print("Computing indicators...")
    add_renko_indicators(df)

    df_train    = df.loc[TRAIN_START:TRAIN_END].copy()
    df_validate = df.loc[VALIDATE_START:VALIDATE_END].copy()
    df_holdout  = df.loc[HOLDOUT_START:HOLDOUT_END].copy()
    print(f"  TRAIN:    {len(df_train):>6,} bricks  ({df_train.index[0]} -> {df_train.index[-1]})")
    print(f"  VALIDATE: {len(df_validate):>6,} bricks  ({df_validate.index[0]} -> {df_validate.index[-1]})")
    print(f"  HOLDOUT:  {len(df_holdout):>6,} bricks  ({df_holdout.index[0]} -> {df_holdout.index[-1]})  [SEALED]")

    df_train_pkl = pickle.dumps(df_train)
    df_val_pkl   = pickle.dumps(df_validate)

    # ── R029 baseline (context only — for comparison) ────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  R029 BASELINE (locked Pine params, context only)")
    print(f"{'=' * 120}")
    base_train    = run_window(gen_r029_baseline(df_train))
    base_validate = run_window(gen_r029_baseline(df_validate))
    base_holdout  = run_window(gen_r029_baseline(df_holdout))
    for label, k in [("TRAIN", base_train), ("VALIDATE", base_validate), ("HOLDOUT", base_holdout)]:
        if k is None:
            print(f"  {label:<10} (too few trades)")
        else:
            print(f"  {label:<10} T={k['trades']:>4}  WR={k['wr']:>5.1f}%  PF={k['pf']:>6.2f}  "
                  f"Net=${k['net_profit']:>9.2f}  L/S={k['n_long']}/{k['n_short']}")

    # ── Run all three angles ─────────────────────────────────────────────────
    results_a, holdouts_a = run_sweep(
        "ANGLE A — PULLBACK TO BB_MID IN CONFIRMED TREND",
        ANGLE_A_GRID, _angle_a_worker, gen_pullback,
        df_train, df_validate, df_holdout, df_train_pkl, df_val_pkl)

    results_b, holdouts_b = run_sweep(
        "ANGLE B — RANGE BREAKOUT (DONCHIAN-STYLE)",
        ANGLE_B_GRID, _angle_b_worker, gen_breakout,
        df_train, df_validate, df_holdout, df_train_pkl, df_val_pkl)

    results_c, holdouts_c = run_sweep(
        "ANGLE C — KAMA CROSSOVER",
        ANGLE_C_GRID, _angle_c_worker, gen_kama,
        df_train, df_validate, df_holdout, df_train_pkl, df_val_pkl)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  PHASE 19 SUMMARY")
    print(f"{'=' * 120}")
    a_pass = [r for r in holdouts_a if evaluate(r["holdout"])[0]]
    b_pass = [r for r in holdouts_b if evaluate(r["holdout"])[0]]
    c_pass = [r for r in holdouts_c if evaluate(r["holdout"])[0]]
    print(f"  Angle A (Pullback to bb_mid):       {len(a_pass)} PASS / {len(holdouts_a)} finalists")
    print(f"  Angle B (Range Breakout):           {len(b_pass)} PASS / {len(holdouts_b)} finalists")
    print(f"  Angle C (KAMA Crossover):           {len(c_pass)} PASS / {len(holdouts_c)} finalists")

    if a_pass or b_pass or c_pass:
        print(f"\n  WINNERS:")
        for label, lst in [("A-Pullback", a_pass), ("B-Breakout", b_pass), ("C-KAMA", c_pass)]:
            for w in lst:
                p = w["params"]; h = w["holdout"]
                params_str = " ".join(f"{k}={v}" for k, v in p.items())
                print(f"    {label}: HOLDOUT PF={h['pf']:.2f} T={h['trades']} WR={h['wr']:.1f}% "
                      f"Net=${h['net_profit']:.2f}  [{params_str}]")
        print(f"\n  -> Candidate(s) for Stage 2 combine with R029")
        print(f"  -> Multiple-comparison risk: this is the 3rd sweep on the same HOLDOUT.")
        print(f"     Refresh data through April 7+ and re-run holdout BEFORE any deploy.")
    else:
        print(f"\n  ALL THREE ANGLES FAIL holdout. Combined with Phase 17 + 18, this is now")
        print(f"  6 different trend mechanisms on EURUSD 0.0006 with zero passes:")
        print(f"    1. Naked brick streak")
        print(f"    2. Band-walk filtered streak")
        print(f"    3. bb_mid slope filter on R029 MR")
        print(f"    4. Pullback to bb_mid in confirmed trend")
        print(f"    5. Range breakout (Donchian)")
        print(f"    6. KAMA crossover")
        print(f"  This is strong cross-mechanism evidence the brick is fundamentally MR-only.")
        print(f"  Strong recommend: pivot to a different brick (0.0008/0.0010), different")
        print(f"  instrument (GBPJPY 0.05), or accept R029 as the locked design.")
    print(f"{'=' * 120}")
    print(f"\nNOTE: data ends 2026-03-25. April 7 trends NOT in this holdout.")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "renko_file": RENKO_FILE,
                "train_window":    [TRAIN_START, TRAIN_END],
                "validate_window": [VALIDATE_START, VALIDATE_END],
                "holdout_window":  [HOLDOUT_START, HOLDOUT_END],
                "angle_a_grid": ANGLE_A_GRID,
                "angle_b_grid": ANGLE_B_GRID,
                "angle_c_grid": ANGLE_C_GRID,
                "rules": RULES,
                "finalist_filter": {
                    "min_validate_trades": MIN_VALIDATE_TRADES_FINAL,
                    "max_train_val_wr_drop": MAX_TRAIN_VAL_WR_DROP,
                    "max_train_val_wr_jump": MAX_TRAIN_VAL_WR_JUMP,
                },
            },
            "baseline": {"train": base_train, "validate": base_validate, "holdout": base_holdout},
            "angle_a": {"results": results_a, "holdout": holdouts_a},
            "angle_b": {"results": results_b, "holdout": holdouts_b},
            "angle_c": {"results": results_c, "holdout": holdouts_c},
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
