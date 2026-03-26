import json

with open("ai_context/phase13_results.json") as f:
    data = json.load(f)

eu4_i4 = [r for r in data if r["inst"] == "EU4" and r["idea"] == 4
          and r["combo"].get("htf_thresh", 0) == 0 and r["oos_trades"] >= 10]

def bal(r):
    pf = r["oos_pf"] if r["oos_pf"] != "inf" else 500
    pf = min(pf, 500)
    return pf * (r["oos_trades"] ** 0.5) * (r["oos_wr"] / 100)

eu4_i4.sort(key=bal, reverse=True)
print("Top 15 EU4 Idea4 no-HTF by balanced score:")
for r in eu4_i4[:15]:
    c = r["combo"]
    pf_oos = "INF" if r["oos_pf"] == "inf" else f"{r['oos_pf']:.1f}"
    pf_is = "INF" if r["is_pf"] == "inf" else f"{r['is_pf']:.1f}"
    print(f"  OOS PF={pf_oos:>6} T={r['oos_trades']:>3} WR={r['oos_wr']:>5.1f}% "
          f"Net=${r['oos_net']:>6.2f} DD={r['oos_dd']:>5.2f}% | "
          f"IS PF={pf_is:>6} T={r['is_trades']:>4} | "
          f"rise={c['adx_rise_lb']} min={c['adx_min']} p6={c['p6']} "
          f"s={c['sess']} n={c['n_bricks']} cd={c['cooldown']}")
