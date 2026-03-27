import json

with open("ai_context/phase13_results.json") as f:
    data = json.load(f)

print(f"Total results: {len(data)}")

idea_names = {
    1: "Gap Filler (escgo+HTF)",
    2: "Squeeze Breakout",
    3: "Triple Cascade",
    4: "ADX Momentum",
}


def sk(r):
    viable = r["oos_trades"] >= 10
    pf = r["oos_pf"] if r["oos_pf"] != "inf" else 1e12
    return (viable, pf, r["oos_net"])


# Per idea per instrument
for idea in [1, 2, 3, 4]:
    print(f"\n{'='*70}")
    print(f"IDEA {idea}: {idea_names[idea]}")
    print(f"{'='*70}")
    idea_data = [r for r in data if r["idea"] == idea]

    for inst in ["EU4", "EU5", "EU6", "EU7"]:
        inst_data = [r for r in idea_data if r["inst"] == inst]
        inst_data.sort(key=sk, reverse=True)
        viable = [r for r in inst_data if r["oos_trades"] >= 10]
        label = inst_data[0]["label"] if inst_data else "?"
        print(f"\n  [{inst}] {label} — {len(inst_data)} runs, {len(viable)} viable")
        for r in inst_data[:3]:
            pf_oos = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.2f}"
            pf_is = "INF" if r["is_pf"] == "inf" else f"{r['is_pf']:.2f}"
            c = r["combo"]
            print(
                f"    IS PF={pf_is:>7} T={r['is_trades']:>4} | "
                f"OOS PF={pf_oos:>7} T={r['oos_trades']:>3} "
                f"WR={r['oos_wr']:>5.1f}% Net={r['oos_net']:>7.2f} "
                f"DD={r['oos_dd']:>5.2f}% | {c}"
            )

# Overall top 20
print(f"\n{'='*70}")
print("OVERALL TOP 20 (>=10 OOS trades)")
print(f"{'='*70}")
viable_all = [r for r in data if r["oos_trades"] >= 10]
viable_all.sort(key=sk, reverse=True)
for i, r in enumerate(viable_all[:20]):
    pf_oos = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.2f}"
    c = r["combo"]
    print(
        f"  {i+1:>2}. [{r['inst']}] Idea{r['idea']} "
        f"OOS PF={pf_oos:>7} T={r['oos_trades']:>3} "
        f"WR={r['oos_wr']:>5.1f}% Net={r['oos_net']:>7.2f} "
        f"DD={r['oos_dd']:>5.2f}% | {c}"
    )

# Summary stats per idea
print(f"\n{'='*70}")
print("SUMMARY: Average OOS PF by idea (viable configs only)")
print(f"{'='*70}")
for idea in [1, 2, 3, 4]:
    viable = [r for r in data if r["idea"] == idea and r["oos_trades"] >= 10
              and r["oos_pf"] != "inf"]
    if viable:
        avg_pf = sum(r["oos_pf"] for r in viable) / len(viable)
        avg_trades = sum(r["oos_trades"] for r in viable) / len(viable)
        avg_wr = sum(r["oos_wr"] for r in viable) / len(viable)
        max_pf = max(r["oos_pf"] for r in viable)
        print(f"  Idea {idea} ({idea_names[idea]}): "
              f"avg PF={avg_pf:.2f}, max PF={max_pf:.2f}, "
              f"avg T={avg_trades:.0f}, avg WR={avg_wr:.1f}%, "
              f"viable={len(viable)}")
    else:
        print(f"  Idea {idea}: no viable results")

# Best per instrument across all ideas
print(f"\n{'='*70}")
print("BEST PER INSTRUMENT (all ideas)")
print(f"{'='*70}")
for inst in ["EU4", "EU5", "EU6", "EU7"]:
    inst_viable = [r for r in data if r["inst"] == inst and r["oos_trades"] >= 10]
    inst_viable.sort(key=sk, reverse=True)
    if inst_viable:
        r = inst_viable[0]
        pf_oos = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.2f}"
        print(f"  {inst}: Idea{r['idea']} OOS PF={pf_oos} T={r['oos_trades']} "
              f"WR={r['oos_wr']:.1f}% Net={r['oos_net']:.2f} | {r['combo']}")
