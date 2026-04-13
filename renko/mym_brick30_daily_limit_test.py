"""
MYM Brick 30 — daily loss limit test for the 5 finalists from the brick 30 sweep.

Tests each finalist (streak, cooldown) against 8 daily loss limit values on the
HOLDOUT window (2025-10-01 -> 2026-04-08).

Day boundary: UTC midnight (resets daily P&L tracker when calendar date changes).

Engine config uses CORRECTED MYM scaling:
  qty_value=0.5, commission_pct=0.004
This matches MYM's $0.50/point multiplier and ~$1.90 RT commission, so all
output dollars are in LIVE-equivalent units (no 2x scaling).

The signal generator now tracks running daily P&L per UTC day. When the
accumulated daily P&L drops to -daily_loss_limit, new entries are blocked
for the rest of that day. Existing positions continue to exit normally on
the first opposing brick.

NOT a sweep — focused test of 5 finalists x 8 limits = 40 backtests. ~10s.
"""

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export

RENKO_FILE = "CBOT_MINI_MYM1!, 1S ETH renko 30.csv"

# ── Holdout window ───────────────────────────────────────────────────────────
HOLDOUT_START = "2025-10-01"
HOLDOUT_END   = "2026-04-08"  # exclusive — captures all of April 7

# ── CORRECTED MYM scaling (workaround until engine gets contract_multiplier) ─
# qty_value=0.5 applies the $0.50/point MYM contract multiplier
# commission_pct=0.004 keeps RT commission ~$1.88 at MYM prices (~$1.90 live)
BACKTEST_CONFIG = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.004,
    qty_type="fixed",
    qty_value=0.5,
    start_date="2000-01-01",
    end_date="2099-12-31",
)
QTY_FOR_PNL_CALC = 0.5      # must match BACKTEST_CONFIG.qty_value
COMM_RATE        = 0.004 / 100  # commission_pct as a decimal multiplier

OUTPUT_FILE = ROOT / "ai_context" / "mym_brick30_daily_limit_test_results.json"

# ── 5 finalists from the brick 30 sweep (streak, cooldown) ──────────────────
FINALISTS = [
    {"min_brick_streak":  8, "cooldown": 75},   # Holdout REJECT (PF<10, WR<55, Apr7<-50)
    {"min_brick_streak": 13, "cooldown": 30},   # Holdout REJECT (WR<55)
    {"min_brick_streak": 15, "cooldown": 25},   # Holdout DEPLOY ✓
    {"min_brick_streak": 13, "cooldown": 40},   # Holdout REJECT (WR<55)
    {"min_brick_streak": 13, "cooldown":  0},   # Holdout REJECT (WR<55)
]

# ── Daily loss limits to test (LIVE-equivalent dollars) ─────────────────────
DAILY_LIMITS = [0, 25, 50, 75, 100, 125, 150, 200]   # 0 = off


def generate_signals_with_daily_limit(df, min_brick_streak, cooldown, daily_loss_limit):
    """Pure brick streak entry, brick flip exit, cooldown bars between trades.

    NEW: tracks running daily P&L (UTC midnight reset). When the accumulated
    daily P&L hits -daily_loss_limit, blocks new entries for the rest of that
    UTC day. Existing positions still exit normally.

    daily_loss_limit=0 disables the limit (matches no-limit baseline).

    Returns the augmented DataFrame plus a metadata dict (entries_blocked,
    days_limit_triggered) for diagnostics.
    """
    brick_up    = df["brick_up"].values
    closes      = df["Close"].values
    timestamps  = df.index
    n           = len(df)

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    warmup = min_brick_streak + 2
    last_exit_bar = -999_999
    pos = 0
    entry_price = 0.0

    daily_pnl = 0.0
    current_day = None  # pd.Timestamp.date() of the current UTC day

    entries_blocked = 0
    days_with_block = set()

    for i in range(warmup, n):
        bar_date = pd.Timestamp(timestamps[i]).date()
        if current_day != bar_date:
            current_day = bar_date
            daily_pnl   = 0.0

        # ── Brick exit (always allowed, even after limit hit) ───────────────
        if pos == 1 and not brick_up[i]:
            gross = (closes[i] - entry_price) * QTY_FOR_PNL_CALC
            comm  = (entry_price + closes[i]) * QTY_FOR_PNL_CALC * COMM_RATE
            net   = gross - comm
            daily_pnl += net
            long_exit[i] = True
            pos = 0
            last_exit_bar = i
        elif pos == -1 and brick_up[i]:
            gross = (entry_price - closes[i]) * QTY_FOR_PNL_CALC
            comm  = (entry_price + closes[i]) * QTY_FOR_PNL_CALC * COMM_RATE
            net   = gross - comm
            daily_pnl += net
            short_exit[i] = True
            pos = 0
            last_exit_bar = i

        # ── Daily limit gate (block new entries if hit) ─────────────────────
        limit_hit = (daily_loss_limit > 0) and (daily_pnl <= -daily_loss_limit)

        # ── Entry: pure brick streak ────────────────────────────────────────
        if pos == 0 and (i - last_exit_bar) >= cooldown:
            last_n = brick_up[i - min_brick_streak:i]
            prev_n = brick_up[i - min_brick_streak - 1:i - 1]
            brk_long_ok  = bool(np.all(last_n)) and not bool(np.all(prev_n))
            brk_short_ok = bool(not np.any(last_n)) and not bool(not np.any(prev_n))

            if brk_long_ok or brk_short_ok:
                if limit_hit:
                    entries_blocked += 1
                    days_with_block.add(bar_date)
                else:
                    if brk_long_ok:
                        long_entry[i] = True
                        pos = 1
                        entry_price = closes[i]
                    else:
                        short_entry[i] = True
                        pos = -1
                        entry_price = closes[i]

    df = df.copy()
    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit

    return df, {
        "entries_blocked":     entries_blocked,
        "days_limit_triggered": len(days_with_block),
    }


def run_one(df_holdout, params, daily_loss_limit):
    df, meta = generate_signals_with_daily_limit(
        df_holdout,
        min_brick_streak=params["min_brick_streak"],
        cooldown=params["cooldown"],
        daily_loss_limit=daily_loss_limit,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df, BACKTEST_CONFIG)

    trades = kpis.get("trades", [])
    closed = [t for t in trades if t.exit_date is not None and t.pnl is not None]
    worst_loss = min((t.pnl for t in closed), default=0.0)

    # April 7 specifically
    apr7_start = pd.Timestamp("2026-04-07 00:00:00")
    apr7_end   = pd.Timestamp("2026-04-08 00:00:00")
    apr7 = [t for t in trades
            if t.entry_date is not None
            and apr7_start <= pd.Timestamp(t.entry_date) < apr7_end
            and t.pnl is not None]
    apr7_pnl    = sum(t.pnl for t in apr7)
    apr7_count  = len(apr7)

    # Worst single day net P&L
    daily_pnls = {}
    for t in closed:
        if t.exit_date is None:
            continue
        d = pd.Timestamp(t.exit_date).date()
        daily_pnls[d] = daily_pnls.get(d, 0.0) + t.pnl
    worst_day_pnl = min(daily_pnls.values(), default=0.0)

    return {
        "params":       params,
        "limit":        daily_loss_limit,
        "trades":       kpis.get("total_trades", 0),
        "wr":           round(kpis.get("win_rate", 0), 2),
        "pf":           round(kpis.get("profit_factor", 0), 2),
        "net_profit":   round(kpis.get("net_profit", 0), 2),
        "max_dd":       round(kpis.get("max_drawdown", 0), 2),
        "worst_loss":   round(worst_loss, 2),
        "worst_day":    round(worst_day_pnl, 2),
        "apr7_pnl":     round(apr7_pnl, 2),
        "apr7_count":   apr7_count,
        "blocked":      meta["entries_blocked"],
        "days_blocked": meta["days_limit_triggered"],
    }


def main():
    print(f"Loading {RENKO_FILE}...")
    df = load_renko_export(RENKO_FILE)
    print(f"  {len(df):,} bricks loaded, {df.index[0]} -> {df.index[-1]}")

    df_holdout = df.loc[HOLDOUT_START:HOLDOUT_END].copy()
    print(f"  HOLDOUT: {len(df_holdout):,} bricks  ({df_holdout.index[0]} -> {df_holdout.index[-1]})\n")
    print(f"  Engine config: qty_value=0.5, commission_pct=0.004 (MYM live-equivalent scaling)\n")

    print(f"  Testing {len(FINALISTS)} finalists x {len(DAILY_LIMITS)} limits = "
          f"{len(FINALISTS) * len(DAILY_LIMITS)} backtests\n")

    t0 = time.time()
    all_results = []
    for params in FINALISTS:
        for limit in DAILY_LIMITS:
            r = run_one(df_holdout, params, limit)
            all_results.append(r)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s\n")

    # ── Per-finalist tables ─────────────────────────────────────────────────
    for params in FINALISTS:
        rows = [r for r in all_results
                if r["params"]["min_brick_streak"] == params["min_brick_streak"]
                and r["params"]["cooldown"] == params["cooldown"]]
        baseline = next((r for r in rows if r["limit"] == 0), None)
        if baseline is None:
            continue
        base_net = baseline["net_profit"]

        print(f"{'=' * 110}")
        print(f"  FINALIST: streak={params['min_brick_streak']}, cooldown={params['cooldown']}")
        print(f"{'=' * 110}")
        print(f"  {'Limit':>6} | {'PF':>6} {'T':>4} {'WR':>5} {'Net$':>10} {'MaxDD':>9} "
              f"{'WLoss':>8} {'WDay':>9} {'Apr7':>8} {'Blkd':>5} {'Days':>5} | {'Net d vs off':>14}")
        print(f"  {'-'*6} | {'-'*70} | {'-'*12}")
        for r in rows:
            limit_str = "off" if r["limit"] == 0 else f"${r['limit']}"
            net_delta = r["net_profit"] - base_net
            delta_pct = (net_delta / base_net * 100) if base_net != 0 else 0.0
            print(f"  {limit_str:>6} | "
                  f"{r['pf']:>6.2f} {r['trades']:>4} {r['wr']:>5.1f} "
                  f"{r['net_profit']:>10.2f} {r['max_dd']:>9.2f} "
                  f"{r['worst_loss']:>8.2f} {r['worst_day']:>9.2f} "
                  f"{r['apr7_pnl']:>8.2f} {r['blocked']:>5} {r['days_blocked']:>5} | "
                  f"{net_delta:>+8.2f} ({delta_pct:>+4.0f}%)")
        print()

    # ── Cross-finalist summary at $50 / $100 / $150 limits ──────────────────
    print(f"{'=' * 110}")
    print(f"  CROSS-FINALIST COMPARISON AT KEY LIMIT VALUES")
    print(f"{'=' * 110}")
    for limit in [0, 50, 100, 150]:
        print(f"\n  Limit = {'off' if limit == 0 else f'${limit}'}:")
        print(f"    {'Finalist':>15} | {'PF':>6} {'T':>4} {'WR':>5} {'Net$':>10} {'WDay':>9} {'Apr7':>8}")
        for params in FINALISTS:
            r = next((x for x in all_results
                      if x["params"]["min_brick_streak"] == params["min_brick_streak"]
                      and x["params"]["cooldown"] == params["cooldown"]
                      and x["limit"] == limit), None)
            if r:
                tag = f"s{params['min_brick_streak']}_cd{params['cooldown']}"
                marker = " *" if (params["min_brick_streak"] == 15 and params["cooldown"] == 25) else "  "
                print(f"    {tag + marker:>15} | "
                      f"{r['pf']:>6.2f} {r['trades']:>4} {r['wr']:>5.1f} "
                      f"{r['net_profit']:>10.2f} {r['worst_day']:>9.2f} {r['apr7_pnl']:>8.2f}")

    # ── Save full results ───────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "config": {
                "renko_file":      RENKO_FILE,
                "holdout_window":  [HOLDOUT_START, HOLDOUT_END],
                "qty_value":       0.5,
                "commission_pct":  0.004,
                "finalists":       FINALISTS,
                "daily_limits":    DAILY_LIMITS,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
