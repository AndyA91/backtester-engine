"""
Compare entry modes: fresh_only vs resume_1 vs resume (unlimited).
Shows whether capping re-entries to 1 per trend keeps the edge.
"""

import contextlib
import io
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.strategies.r024_kama_ribbon_pullback import generate_signals, _KAMA_CACHE

import os

# ── Config ──────────────────────────────────────────────────────────────────
RENKO_FILE = os.environ.get("RENKO_FILE", "OANDA_EURUSD, 1S renko 0.0004.csv")
IS_START = "2023-01-23"
IS_END   = "2025-09-30"

BASE_PARAMS = dict(
    ribbon="3L_5_13_34",
    cooldown=3,
    adx_gate=0,
    exit_on_brick=True,
)

MODES = ["fresh_only", "resume_1", "resume"]

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading renko data...")
df = load_renko_export(RENKO_FILE)
add_renko_indicators(df)

cfg = BacktestConfig(
    initial_capital=1000.0,
    commission_pct=0.0046,
    slippage_ticks=0,
    qty_type="fixed",
    qty_value=1000.0,
    pyramiding=1,
    start_date=IS_START,
    end_date=IS_END,
    take_profit_pct=0.0,
    stop_loss_pct=0.0,
)


def run_mode(mode):
    _KAMA_CACHE.clear()
    params = {**BASE_PARAMS, "entry_mode": mode}
    df_sig = generate_signals(df.copy(), **params)
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)
    trades = [t for t in kpis.get("trades", []) if t.exit_price is not None]
    return trades


# ── Run all modes ───────────────────────────────────────────────────────────
results = {}
for mode in MODES:
    trades = run_mode(mode)
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    pf = total_win / total_loss if total_loss > 0 else float("inf")
    results[mode] = {
        "count": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "avg_pnl": sum(pnls) / len(trades) if trades else 0,
        "net_pnl": sum(pnls),
        "pf": pf,
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
    }

# ── Print comparison table ──────────────────────────────────────────────────
print(f"\nR024 KAMA Ribbon Pullback — entry mode comparison")
print(f"Period: {IS_START} to {IS_END}")
print(f"Params: ribbon={BASE_PARAMS['ribbon']}, cooldown={BASE_PARAMS['cooldown']}, "
      f"adx_gate={BASE_PARAMS['adx_gate']}, exit_on_brick={BASE_PARAMS['exit_on_brick']}")
print()

header = f"{'Metric':<14}"
for mode in MODES:
    header += f" {mode:>14}"
print(header)
print("-" * len(header))

for key, label in [
    ("count",    "Count"),
    ("win_rate", "Win rate %"),
    ("avg_pnl",  "Avg PnL"),
    ("net_pnl",  "Net PnL"),
    ("pf",       "PF"),
    ("avg_win",  "Avg Win"),
    ("avg_loss", "Avg Loss"),
]:
    row = f"{label:<14}"
    for mode in MODES:
        v = results[mode][key]
        if key == "count":
            row += f" {v:>14d}"
        elif key == "win_rate":
            row += f" {v:>13.1f}%"
        elif key == "pf":
            row += f" {v:>14.2f}" if not math.isinf(v) else f" {'INF':>14}"
        else:
            row += f" {v:>14.4f}"
    print(row)
