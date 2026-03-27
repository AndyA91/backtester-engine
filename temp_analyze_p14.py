#!/usr/bin/env python3
"""Analyze Phase 14 results."""
import json

with open("ai_context/phase14_results.json") as f:
    data = json.load(f)

print(f"Total results: {len(data)}")
insts = set(r["inst"] for r in data)
print(f"Instruments: {insts}")

# Filter WR >= 75% with trades >= 8
wr75 = [r for r in data if r["oos_wr"] >= 75.0 and r["oos_trades"] >= 8]

def safe_pf(pf):
    return 1e12 if pf == "inf" else float(pf)

wr75.sort(key=lambda r: (r["oos_net"], safe_pf(r["oos_pf"])), reverse=True)

print(f"WR >= 75% configs: {len(wr75)}")

print()
print("=" * 90)
print("TOP 30 BY NET PROFIT (WR >= 75%, trades >= 8)")
print("=" * 90)
for i, r in enumerate(wr75[:30]):
    pf = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.2f}"
    is_pf = "INF" if r["is_pf"] == "inf" else f"{r['is_pf']:.2f}"
    c = r["combo"]
    keys_skip = {"idea", "n_bricks", "cooldown"}
    extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
    print(f"  {i+1:>2}. [{r['inst']}] Idea{r['idea']} "
          f"IS PF={is_pf:>7} T={r['is_trades']:>4} | "
          f"OOS PF={pf:>7} T={r['oos_trades']:>3} "
          f"WR={r['oos_wr']:>5.1f}% Net={r['oos_net']:>8.2f} "
          f"DD={r['oos_dd']:>5.2f}% | n={c['n_bricks']} cd={c['cooldown']} {extra}")

print()
print("=" * 90)
print("TOP 30 BY PF (WR >= 75%, trades >= 8)")
print("=" * 90)
wr75_pf = sorted(wr75, key=lambda r: (safe_pf(r["oos_pf"]), r["oos_net"]), reverse=True)
for i, r in enumerate(wr75_pf[:30]):
    pf = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.2f}"
    is_pf = "INF" if r["is_pf"] == "inf" else f"{r['is_pf']:.2f}"
    c = r["combo"]
    keys_skip = {"idea", "n_bricks", "cooldown"}
    extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
    print(f"  {i+1:>2}. [{r['inst']}] Idea{r['idea']} "
          f"IS PF={is_pf:>7} T={r['is_trades']:>4} | "
          f"OOS PF={pf:>7} T={r['oos_trades']:>3} "
          f"WR={r['oos_wr']:>5.1f}% Net={r['oos_net']:>8.2f} "
          f"DD={r['oos_dd']:>5.2f}% | n={c['n_bricks']} cd={c['cooldown']} {extra}")

# Breakdown by idea
print()
print("=" * 90)
print("BREAKDOWN BY IDEA (WR >= 75%, trades >= 8)")
print("=" * 90)
idea_names = {1: "Fisher Regime", 2: "DI Dominance", 3: "Vol Conviction",
              4: "Triple Align", 5: "Mega Stack"}
for idea in sorted(set(r["idea"] for r in wr75)):
    idea_r = [r for r in wr75 if r["idea"] == idea]
    idea_r.sort(key=lambda r: r["oos_net"], reverse=True)
    print(f"\n  Idea {idea} ({idea_names[idea]}): {len(idea_r)} configs")
    for r in idea_r[:8]:
        pf = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.2f}"
        is_pf = "INF" if r["is_pf"] == "inf" else f"{r['is_pf']:.2f}"
        c = r["combo"]
        keys_skip = {"idea", "n_bricks", "cooldown"}
        extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
        print(f"    IS PF={is_pf:>7} T={r['is_trades']:>4} | "
              f"OOS PF={pf:>7} T={r['oos_trades']:>3} WR={r['oos_wr']:>5.1f}% "
              f"Net={r['oos_net']:>8.2f} | n={c['n_bricks']} cd={c['cooldown']} {extra}")

# Overall top 20 regardless of WR
print()
print("=" * 90)
print("OVERALL TOP 20 BY PF (trades >= 8, any WR)")
print("=" * 90)
viable = [r for r in data if r["oos_trades"] >= 8]
viable.sort(key=lambda r: (safe_pf(r["oos_pf"]), r["oos_net"]), reverse=True)
for i, r in enumerate(viable[:20]):
    pf = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.2f}"
    is_pf = "INF" if r["is_pf"] == "inf" else f"{r['is_pf']:.2f}"
    c = r["combo"]
    keys_skip = {"idea", "n_bricks", "cooldown"}
    extra = " ".join(f"{k}={v}" for k, v in c.items() if k not in keys_skip)
    marker = " ***" if r["oos_wr"] >= 75.0 else ""
    print(f"  {i+1:>2}. [{r['inst']}] Idea{r['idea']} "
          f"IS PF={is_pf:>7} T={r['is_trades']:>4} | "
          f"OOS PF={pf:>7} T={r['oos_trades']:>3} "
          f"WR={r['oos_wr']:>5.1f}% Net={r['oos_net']:>8.2f} "
          f"DD={r['oos_dd']:>5.2f}% | n={c['n_bricks']} cd={c['cooldown']} {extra}{marker}")
