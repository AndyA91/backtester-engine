"""Cross-phase analysis for MYM futures strategies."""
import json, math
from collections import defaultdict

# Load all three phases
with open("ai_context/mym_sweep_results.json") as f:
    v1 = json.load(f)
with open("ai_context/mym_sweep_v2_results.json") as f:
    v2 = json.load(f)
with open("ai_context/mym_sweep_v3_results.json") as f:
    v3 = json.load(f)

all_data = v1 + v2 + v3
print(f"Total results: v1={len(v1)}, v2={len(v2)}, v3={len(v3)}, combined={len(all_data)}")

# Best per brick across all phases (OOS trades >= 15 for robustness)
MIN_T = 15
print(f"\n=== ALL-TIME BEST PER BRICK (OOS trades >= {MIN_T}) ===")
for brick in [11, 12, 13, 14, 15]:
    viable = [r for r in all_data
              if r["brick"] == brick and r["oos_trades"] >= MIN_T
              and not math.isinf(r["oos_pf"])]
    viable.sort(key=lambda r: r["oos_pf"], reverse=True)
    if viable:
        b = viable[0]
        print(f"  Brick {brick}: OOS PF={b['oos_pf']:.2f} T={b['oos_trades']} "
              f"WR={b['oos_wr']:.1f}% Net=${b['oos_net']:.2f} "
              f"| {b['stack']} n={b['n_bricks']} cd={b['cooldown']}")

# Find configs that work well across MULTIPLE bricks (robustness check)
print(f"\n=== CROSS-BRICK ROBUST CONFIGS (OOS trades >= {MIN_T}) ===")
print("  Configs that appear in top-50 of 3+ brick sizes:")

# For each brick, get top-50 configs
top_configs = defaultdict(list)
for brick in [11, 12, 13, 14, 15]:
    viable = [r for r in all_data
              if r["brick"] == brick and r["oos_trades"] >= MIN_T
              and not math.isinf(r["oos_pf"])]
    viable.sort(key=lambda r: r["oos_pf"], reverse=True)
    for r in viable[:50]:
        # Create a "config signature" (gate stack + params)
        sig = f"{r['stack']}_n{r['n_bricks']}_cd{r['cooldown']}"
        top_configs[sig].append({
            "brick": brick, "oos_pf": r["oos_pf"], "oos_trades": r["oos_trades"],
            "oos_wr": r["oos_wr"], "oos_net": r["oos_net"]
        })

# Show configs that appear in 3+ brick sizes
multi_brick = {k: v for k, v in top_configs.items() if len(v) >= 3}
multi_brick_sorted = sorted(multi_brick.items(),
                           key=lambda x: sum(e["oos_pf"] for e in x[1]) / len(x[1]),
                           reverse=True)

for config, entries in multi_brick_sorted[:20]:
    avg_pf = sum(e["oos_pf"] for e in entries) / len(entries)
    avg_t = sum(e["oos_trades"] for e in entries) / len(entries)
    bricks = sorted([e["brick"] for e in entries])
    print(f"  {config}")
    print(f"    avg PF={avg_pf:.2f} avg T={avg_t:.0f} bricks={bricks}")
    for e in sorted(entries, key=lambda x: x["brick"]):
        print(f"      B{e['brick']}: PF={e['oos_pf']:.2f} T={e['oos_trades']} "
              f"WR={e['oos_wr']:.1f}% Net=${e['oos_net']:.2f}")

# Best config per brick with >= 30 OOS trades (more conservative)
print(f"\n=== CONSERVATIVE PICK (OOS trades >= 30) ===")
for brick in [11, 12, 13, 14, 15]:
    viable = [r for r in all_data
              if r["brick"] == brick and r["oos_trades"] >= 30
              and not math.isinf(r["oos_pf"])]
    viable.sort(key=lambda r: r["oos_pf"], reverse=True)
    if viable:
        b = viable[0]
        print(f"  Brick {brick}: OOS PF={b['oos_pf']:.2f} T={b['oos_trades']} "
              f"WR={b['oos_wr']:.1f}% Net=${b['oos_net']:.2f} "
              f"| {b['stack']} n={b['n_bricks']} cd={b['cooldown']}")

# OOS PF vs trades tradeoff for top configs
print(f"\n=== TOP 10 BY NET PROFIT (any OOS trades >= 20) ===")
viable = [r for r in all_data
          if r["oos_trades"] >= 20 and not math.isinf(r["oos_pf"])]
viable.sort(key=lambda r: r["oos_net"], reverse=True)
for r in viable[:10]:
    print(f"  Brick {r['brick']}: PF={r['oos_pf']:.2f} T={r['oos_trades']} "
          f"WR={r['oos_wr']:.1f}% Net=${r['oos_net']:.2f} "
          f"| {r['stack']} n={r['n_bricks']} cd={r['cooldown']}")
