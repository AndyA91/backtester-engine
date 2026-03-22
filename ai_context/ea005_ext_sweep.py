"""
EA005 Extended Parameter Sweep — 216 combos
pvt_length=[10,15,20], va_pct=[0.60,0.70,0.80],
n_inside=[1,2,3], cooldown=[5,10,20,30], session_start=[0,13]

IS:  2023-07-20 → 2025-09-30
OOS: 2025-10-01 → 2026-03-17

Run from project root:
  python ai_context/ea005_ext_sweep.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "renko" / "strategies"))

from renko.runner import sweep

IS_START  = "2023-07-20"
IS_END    = "2025-09-30"
OOS_START = "2025-10-01"
OOS_END   = "2026-03-17"

OUT_DIR = Path(__file__).parent

print("=== EA005 Extended IS Sweep ===")
is_results, mod = sweep(
    "ea005_va_breakout",
    start=IS_START,
    end=IS_END,
)

# Cache is now warm — OOS sweep is fast (only signal gen, no VP recompute)
print("\n=== EA005 Extended OOS Sweep ===")
oos_results, _ = sweep(
    "ea005_va_breakout",
    start=OOS_START,
    end=OOS_END,
)

with open(OUT_DIR / "ea005_ext_is_results.json", "w") as f:
    json.dump(is_results, f, indent=2)

with open(OUT_DIR / "ea005_ext_oos_results.json", "w") as f:
    json.dump(oos_results, f, indent=2)

print(f"\nSaved: {OUT_DIR / 'ea005_ext_is_results.json'}")
print(f"Saved: {OUT_DIR / 'ea005_ext_oos_results.json'}")

# ---------------------------------------------------------------------------
# Quick summary — density winners (OOS trades > 60, PF > 8.0)
# ---------------------------------------------------------------------------
print("\n=== OOS Density Winners (trades>60, PF>8.0) ===")
oos_by_params = {
    str(r["params"]): r for r in oos_results
}

density_winners = [
    r for r in oos_results
    if r["trades"] > 60 and r["pf"] >= 8.0
]
density_winners.sort(key=lambda r: (-r["pf"], -r["trades"]))

if density_winners:
    for r in density_winners[:10]:
        print(
            f"  PF={r['pf']:.2f} T={r['trades']:>4} WR={r['win_rate']:>5.1f}% "
            f"Net={r['net']:>7.2f} | {r['params']}"
        )
else:
    print("  None found — lowering threshold to PF>6.0")
    fallback = [r for r in oos_results if r["trades"] > 60 and r["pf"] >= 6.0]
    fallback.sort(key=lambda r: (-r["pf"], -r["trades"]))
    for r in fallback[:10]:
        print(
            f"  PF={r['pf']:.2f} T={r['trades']:>4} WR={r['win_rate']:>5.1f}% "
            f"Net={r['net']:>7.2f} | {r['params']}"
        )

# ---------------------------------------------------------------------------
# Top 10 OOS overall (any trade count)
# ---------------------------------------------------------------------------
print("\n=== Top 10 OOS (all trade counts) ===")
oos_sorted = sorted(oos_results, key=lambda r: (-r["pf"], -r["net"]))
for r in oos_sorted[:10]:
    print(
        f"  PF={r['pf']:.2f} T={r['trades']:>4} WR={r['win_rate']:>5.1f}% "
        f"Net={r['net']:>7.2f} | {r['params']}"
    )
