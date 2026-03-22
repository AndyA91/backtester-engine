"""
Targeted IS/OOS decay analysis for ea002 gate sweep.

Runs:
 1. OOS top-5 configs against IS (to get their IS PF — missing from saved JSON)
 2. IS baseline (no gates, session=0) against OOS (to get OOS baseline PF)
 3. Key gate combos across all n/cd against OOS for pattern check

Saves: ai_context/ea002_decay_table.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import contextlib
import io
import math

from engine import BacktestConfig, run_backtest_long_short
from renko.data import load_renko_export
from renko.strategies.ea002_gate_sweep import generate_signals, RENKO_FILE, COMMISSION_PCT, INITIAL_CAPITAL

IS_START = "2023-07-20"
IS_END   = "2025-09-30"
OOS_START = "2025-10-01"
OOS_END   = "2026-03-17"
MIN_TRADES = 30


def run_one(df, params, start, end):
    df_sig = generate_signals(df.copy(), **params)
    cfg = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        commission_pct=COMMISSION_PCT,
        slippage_ticks=0,
        qty_type="fixed",
        qty_value=1000.0,
        pyramiding=1,
        start_date=start,
        end_date=end,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        kpis = run_backtest_long_short(df_sig, cfg)
    pf = kpis.get("profit_factor", 0.0) or 0.0
    return {
        "pf":     float("inf") if math.isinf(pf) else float(pf),
        "net":    float(kpis.get("net_profit", 0.0) or 0.0),
        "trades": int(kpis.get("total_trades", 0) or 0),
        "wr":     float(kpis.get("win_rate", 0.0) or 0.0),
        "dd":     float(kpis.get("max_drawdown_pct", 0.0) or 0.0),
    }


print("Loading renko data...")
_df = load_renko_export(RENKO_FILE)

# ── OOS top-5 configs (need IS values to compute decay) ──────────────────────
oos_top5_params = [
    dict(n_bricks=5, cooldown=30, session_start=13, raff_gate=False, vp_gate=True,  div_gate=True, do_gate=False),
    dict(n_bricks=4, cooldown=30, session_start=13, raff_gate=False, vp_gate=True,  div_gate=True, do_gate=False),
    dict(n_bricks=5, cooldown=20, session_start=13, raff_gate=False, vp_gate=True,  div_gate=True, do_gate=False),
    dict(n_bricks=4, cooldown=30, session_start=13, raff_gate=True,  vp_gate=True,  div_gate=True, do_gate=False),
    dict(n_bricks=4, cooldown=20, session_start=13, raff_gate=False, vp_gate=True,  div_gate=True, do_gate=False),
]

# OOS PF already known from saved JSON
oos_pf_known = [10.6244, 9.8445, 9.8187, 9.2895, 9.1120]

print("\n=== Computing IS PF for OOS top-5 configs ===")
decay_rows = []
for i, p in enumerate(oos_top5_params):
    is_r  = run_one(_df, p, IS_START, IS_END)
    label = f"n={p['n_bricks']},cd={p['cooldown']},ss={p['session_start']},raff={p['raff_gate']},vp={p['vp_gate']},div={p['div_gate']},do={p['do_gate']}"
    oos_pf = oos_pf_known[i]
    decay  = (oos_pf - is_r["pf"]) / is_r["pf"] * 100 if is_r["pf"] > 0 else None
    print(f"  {label}")
    print(f"    IS: PF={is_r['pf']:.4f} T={is_r['trades']} | OOS: PF={oos_pf:.4f} | Decay={decay:+.1f}%")
    decay_rows.append({
        "label":    label,
        "params":   p,
        "is":       is_r,
        "oos_pf":   oos_pf,
        "decay_pct": decay,
    })

# ── Baseline (no gates, session=0) across all n/cd for IS and OOS ────────────
print("\n=== Baseline IS->OOS for all n/cd combos ===")
baseline_rows = []
for n in [2, 3, 4, 5]:
    for cd in [10, 20, 30]:
        p_base = dict(n_bricks=n, cooldown=cd, session_start=0, raff_gate=False, vp_gate=False, div_gate=False, do_gate=False)
        is_r  = run_one(_df, p_base, IS_START, IS_END)
        oos_r = run_one(_df, p_base, OOS_START, OOS_END)
        decay = (oos_r["pf"] - is_r["pf"]) / is_r["pf"] * 100 if is_r["pf"] > 0 else None
        print(f"  n={n},cd={cd}: IS PF={is_r['pf']:.4f} T={is_r['trades']} | OOS PF={oos_r['pf']:.4f} T={oos_r['trades']} | Decay={decay:+.1f}%")
        baseline_rows.append({
            "n_bricks": n, "cooldown": cd,
            "is": is_r, "oos": oos_r, "decay_pct": decay,
        })

# ── Primary gated combo (ss=13,vp=T,div=T,raff=F,do=F) across all n/cd ──────
print("\n=== Primary gate (ss=13,vp=T,div=T,raff=F,do=F) across all n/cd ===")
primary_rows = []

for n in [2, 3, 4, 5]:
    for cd in [10, 20, 30]:
        p = dict(n_bricks=n, cooldown=cd, session_start=13, raff_gate=False, vp_gate=True, div_gate=True, do_gate=False)
        is_r  = run_one(_df, p, IS_START, IS_END)
        oos_r = run_one(_df, p, OOS_START, OOS_END)
        decay = (oos_r["pf"] - is_r["pf"]) / is_r["pf"] * 100 if is_r["pf"] > 0 else None
        print(f"  n={n},cd={cd}: IS PF={is_r['pf']:.4f} T={is_r['trades']} | OOS PF={oos_r['pf']:.4f} T={oos_r['trades']} | Decay={decay:+.1f}%")
        primary_rows.append({
            "n_bricks": n, "cooldown": cd,
            "is": is_r, "oos": oos_r, "decay_pct": decay,
        })

# ── Also check div-only combo (ss=13,div=T,rest=F) across all n/cd ───────────
print("\n=== div-only gate (ss=13,div=T,vp=F,raff=F,do=F) across all n/cd ===")
div_only_rows = []
for n in [2, 3, 4, 5]:
    for cd in [10, 20, 30]:
        p = dict(n_bricks=n, cooldown=cd, session_start=13, raff_gate=False, vp_gate=False, div_gate=True, do_gate=False)
        is_r  = run_one(_df, p, IS_START, IS_END)
        oos_r = run_one(_df, p, OOS_START, OOS_END)
        decay = (oos_r["pf"] - is_r["pf"]) / is_r["pf"] * 100 if is_r["pf"] > 0 else None
        print(f"  n={n},cd={cd}: IS PF={is_r['pf']:.4f} T={is_r['trades']} | OOS PF={oos_r['pf']:.4f} T={oos_r['trades']} | Decay={decay:+.1f}%")
        div_only_rows.append({
            "n_bricks": n, "cooldown": cd,
            "is": is_r, "oos": oos_r, "decay_pct": decay,
        })

out = {
    "oos_top5_decay": decay_rows,
    "baseline_is_oos": baseline_rows,
    "primary_gate_is_oos": primary_rows,
    "div_only_gate_is_oos": div_only_rows,
}
out_path = Path(__file__).parent / "ea002_decay_table.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}")
