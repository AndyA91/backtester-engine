import json, math

with open("ai_context/phase13_results.json") as f:
    data = json.load(f)

def pf_val(r):
    return r["oos_pf"] if r["oos_pf"] != "inf" else 1e6

def pf_str(r):
    return "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.1f}"

def balanced(r):
    pf = min(pf_val(r), 500)
    return pf * (r["oos_trades"] ** 0.5) * (r["oos_wr"] / 100)

def params_str(r):
    c = r["combo"]
    i = r["idea"]
    if i == 1:
        return f"escgo htf={c['htf_thresh']} osc={c.get('osc','none')} s={c['sess']} a={c['adx']} n={c['n_bricks']} cd={c['cooldown']}"
    elif i == 3:
        return f"cascade {c['ltf_p6']} htf={c['htf_thresh']} s={c['sess']} a={c['adx']} n={c['n_bricks']} cd={c['cooldown']}"
    elif i == 4:
        return f"adxrise lb={c['adx_rise_lb']} min={c['adx_min']} p6={c['p6']} htf={c['htf_thresh']} s={c['sess']} n={c['n_bricks']} cd={c['cooldown']}"
    return str(c)

print("CURRENT LIVE: R016 (EURUSD 0.0005)")
print("  TV: PF=22.91, T=26, WR=69.2%, Net=$52.14")
print("  Python OOS: PF=27.09, T=21, WR=66.7%")
print()
print("DROPPED: R015 (EURUSD 0.0004)")
print("  Python OOS: PF=51.71, T=15, WR=86.7%")
print()

print("=" * 85)
print("PHASE 13 vs LIVE — MULTI-DIMENSIONAL COMPARISON")
print("=" * 85)

for inst in ["EU4", "EU5", "EU6", "EU7"]:
    viable = [r for r in data if r["inst"] == inst and r["oos_trades"] >= 10]
    if not viable:
        print(f"\n--- {inst}: no viable configs ---")
        continue

    by_balanced = sorted(viable, key=balanced, reverse=True)
    high_pf = [r for r in viable if pf_val(r) >= 15]
    by_trades = sorted(high_pf, key=lambda r: r["oos_trades"], reverse=True) if high_pf else []
    by_net = sorted(viable, key=lambda r: r["oos_net"], reverse=True)
    enough_t = [r for r in viable if r["oos_trades"] >= 15]
    by_wr = sorted(enough_t, key=lambda r: r["oos_wr"], reverse=True) if enough_t else []

    label = viable[0]["label"]
    print(f"\n--- {inst} ({label}) ---")

    print("  BALANCED (PF * sqrt(T) * WR):")
    for r in by_balanced[:3]:
        print(f"    I{r['idea']} PF={pf_str(r):>6} T={r['oos_trades']:>3} WR={r['oos_wr']:>5.1f}% "
              f"Net=${r['oos_net']:>6.2f} DD={r['oos_dd']:>5.2f}% | {params_str(r)}")

    print("  MOST TRADES (PF>=15):")
    for r in (by_trades or by_balanced)[:3]:
        print(f"    I{r['idea']} PF={pf_str(r):>6} T={r['oos_trades']:>3} WR={r['oos_wr']:>5.1f}% "
              f"Net=${r['oos_net']:>6.2f} DD={r['oos_dd']:>5.2f}% | {params_str(r)}")

    print("  HIGHEST NET PROFIT:")
    for r in by_net[:3]:
        print(f"    I{r['idea']} PF={pf_str(r):>6} T={r['oos_trades']:>3} WR={r['oos_wr']:>5.1f}% "
              f"Net=${r['oos_net']:>6.2f} DD={r['oos_dd']:>5.2f}% | {params_str(r)}")

    if by_wr:
        print("  HIGHEST WIN RATE (T>=15):")
        for r in by_wr[:3]:
            print(f"    I{r['idea']} PF={pf_str(r):>6} T={r['oos_trades']:>3} WR={r['oos_wr']:>5.1f}% "
                  f"Net=${r['oos_net']:>6.2f} DD={r['oos_dd']:>5.2f}% | {params_str(r)}")

# Direct head-to-head
print()
print("=" * 85)
print("HEAD-TO-HEAD: Best Phase 13 candidates vs R016 live")
print("=" * 85)
print()
print(f"{'Config':<55} {'PF':>6} {'T':>4} {'WR':>6} {'Net':>8} {'DD':>7}")
print("-" * 85)
print(f"{'R016 LIVE (EU5 stoch+HTF35)':<55} {'22.9':>6} {'26':>4} {'69.2%':>6} {'$52.14':>8} {'—':>7}")
print(f"{'R015 DROP (EU4 ema+HTF40)':<55} {'51.7':>6} {'15':>4} {'86.7%':>6} {'—':>8} {'—':>7}")
print()

# Pick the most promising candidates
candidates = []
for inst in ["EU4", "EU5", "EU6", "EU7"]:
    viable = [r for r in data if r["inst"] == inst and r["oos_trades"] >= 10]
    by_b = sorted(viable, key=balanced, reverse=True)
    if by_b:
        candidates.append(by_b[0])
    # Also add highest trade count with PF>=15
    high_pf = [r for r in viable if pf_val(r) >= 15]
    by_t = sorted(high_pf, key=lambda r: r["oos_trades"], reverse=True)
    if by_t and by_t[0] != (by_b[0] if by_b else None):
        candidates.append(by_t[0])

for r in candidates:
    name = f"I{r['idea']} {r['inst']} {params_str(r)[:40]}"
    net_s = f"${r['oos_net']:.2f}"
    print(f"{name:<55} {pf_str(r):>6} {r['oos_trades']:>4} {r['oos_wr']:.1f}% {net_s:>8} {r['oos_dd']:>6.2f}%")

# IS→OOS decay for top candidates
print()
print("=" * 85)
print("IS → OOS DECAY (top balanced per instrument)")
print("=" * 85)
for inst in ["EU4", "EU5", "EU6", "EU7"]:
    viable = [r for r in data if r["inst"] == inst and r["oos_trades"] >= 10]
    by_b = sorted(viable, key=balanced, reverse=True)
    if by_b:
        r = by_b[0]
        is_pf = r["is_pf"] if r["is_pf"] != "inf" else 999
        oos_pf = pf_val(r)
        oos_pf_cap = min(oos_pf, 999)
        if is_pf > 0:
            decay = (1 - oos_pf_cap / is_pf) * 100
        else:
            decay = 0
        print(f"  {inst}: IS PF={r['is_pf'] if r['is_pf'] != 'inf' else 'INF':>7} → OOS PF={pf_str(r):>7} "
              f"(decay {decay:+.0f}%) T_IS={r['is_trades']} T_OOS={r['oos_trades']}")
