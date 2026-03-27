"""Analyze MYM sweep v1 results to plan v2 refinement."""
import json, math

with open("ai_context/mym_sweep_results.json") as f:
    data = json.load(f)

viable = [r for r in data if r["oos_trades"] >= 10]
print(f"Total results: {len(data)}, Viable (OOS T>=10): {len(viable)}")

print("\n=== BEST PER BRICK (OOS PF) ===")
for brick in [11,12,13,14,15]:
    bv = sorted([r for r in viable if r["brick"] == brick and not math.isinf(r["oos_pf"])],
                key=lambda r: r["oos_pf"], reverse=True)
    if not bv: continue
    b = bv[0]
    print(f"  Brick {brick}: OOS PF={b['oos_pf']:.2f} T={b['oos_trades']} "
          f"WR={b['oos_wr']:.1f}% Net=${b['oos_net']:.2f} "
          f"| {b['stack']} n={b['n_bricks']} cd={b['cooldown']}")

print("\n=== AVG OOS PF BY N_BRICKS ===")
for nb in [3,5,7]:
    nv = [r for r in viable if r["n_bricks"] == nb and not math.isinf(r["oos_pf"])]
    if nv:
        print(f"  n={nb}: avg PF={sum(r['oos_pf'] for r in nv)/len(nv):.2f}, "
              f"avg T={sum(r['oos_trades'] for r in nv)/len(nv):.0f}, N={len(nv)}")

print("\n=== AVG OOS PF BY COOLDOWN ===")
for cd in [10,20,30]:
    cv = [r for r in viable if r["cooldown"] == cd and not math.isinf(r["oos_pf"])]
    if cv:
        print(f"  cd={cd}: avg PF={sum(r['oos_pf'] for r in cv)/len(cv):.2f}, "
              f"avg T={sum(r['oos_trades'] for r in cv)/len(cv):.0f}, N={len(cv)}")

print("\n=== AVG OOS PF BY ADX ===")
for at in [20,25,30]:
    av = [r for r in viable if r["adx_thresh"] == at and not math.isinf(r["oos_pf"])]
    if av:
        print(f"  ADX>={at}: avg PF={sum(r['oos_pf'] for r in av)/len(av):.2f}, "
              f"avg T={sum(r['oos_trades'] for r in av)/len(av):.0f}, N={len(av)}")

print("\n=== AVG OOS PF BY P6 GATE ===")
for pg in ["none","stoch_cross","escgo_cross","mk_regime","ema_cross","psar_dir"]:
    pv = [r for r in viable if r["p6_gate"] == pg and not math.isinf(r["oos_pf"])]
    if pv:
        print(f"  {pg:16s} avg PF={sum(r['oos_pf'] for r in pv)/len(pv):.2f}, "
              f"avg T={sum(r['oos_trades'] for r in pv)/len(pv):.0f}, N={len(pv)}")

print("\n=== AVG OOS PF BY OSC ===")
for osc in ["none","sto_tso","macd_lc"]:
    ov = [r for r in viable if r["osc"] == osc and not math.isinf(r["oos_pf"])]
    if ov:
        print(f"  {osc:16s} avg PF={sum(r['oos_pf'] for r in ov)/len(ov):.2f}, "
              f"avg T={sum(r['oos_trades'] for r in ov)/len(ov):.0f}, N={len(ov)}")

print("\n=== AVG OOS PF BY SESSION ===")
for ss in [0,10,11]:
    sv = [r for r in viable if r["sess_start"] == ss and not math.isinf(r["oos_pf"])]
    if sv:
        label = "no filter" if ss == 0 else f"sess>={ss}ET"
        print(f"  {label:16s} avg PF={sum(r['oos_pf'] for r in sv)/len(sv):.2f}, "
              f"avg T={sum(r['oos_trades'] for r in sv)/len(sv):.0f}, N={len(sv)}")

# Per-brick breakdown of n_bricks winner
print("\n=== N_BRICKS WINNER PER BRICK ===")
for brick in [11,12,13,14,15]:
    for nb in [3,5,7]:
        bv = [r for r in viable if r["brick"] == brick and r["n_bricks"] == nb
              and not math.isinf(r["oos_pf"])]
        if bv:
            avg = sum(r["oos_pf"] for r in bv) / len(bv)
            print(f"  Brick {brick} n={nb}: avg PF={avg:.2f} (N={len(bv)})")
    print()

# IS vs OOS PF distribution for top configs
print("\n=== IS→OOS DECAY (top 20 overall) ===")
top = sorted([r for r in viable if not math.isinf(r["oos_pf"])],
             key=lambda r: r["oos_pf"], reverse=True)[:20]
for r in top:
    dec = r["decay_pct"] if not math.isnan(r["decay_pct"]) else 0
    print(f"  Brick {r['brick']:2d} IS_PF={r['is_pf']:>7.2f} OOS_PF={r['oos_pf']:>7.2f} "
          f"Decay={dec:>+7.1f}% T_IS={r['is_trades']:>4} T_OOS={r['oos_trades']:>4} "
          f"| {r['stack']} n={r['n_bricks']} cd={r['cooldown']}")
