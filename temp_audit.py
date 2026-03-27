import json

with open('ai_context/ea005_ext_is_results.json', 'r') as f:
    is_data = json.load(f)

with open('ai_context/ea005_ext_oos_results.json', 'r') as f:
    oos_data = json.load(f)

winners = [
    {"pvt_length": 10, "va_pct": 0.6, "n_inside": 2, "cooldown": 5, "session_start": 13},
    {"pvt_length": 10, "va_pct": 0.6, "n_inside": 1, "cooldown": 5, "session_start": 13},
    {"pvt_length": 10, "va_pct": 0.6, "n_inside": 1, "cooldown": 10, "session_start": 13},
    {"pvt_length": 10, "va_pct": 0.6, "n_inside": 2, "cooldown": 10, "session_start": 13}, # Near miss
]

def find_result(data, params):
    for r in data:
        if all(r['params'][k] == v for k, v in params.items()):
            return r
    return None

print("| pvt | va | n | cd | ss | IS_PF | IS_T | OOS_PF | OOS_T | Decay |")
print("|---|---|---|---|---|---|---|---|---|---|")
for w in winners:
    is_r = find_result(is_data, w)
    oos_r = find_result(oos_data, w)
    if is_r and oos_r:
        decay = (oos_r['pf'] - is_r['pf']) / is_r['pf'] * 100
        print(f"| {w['pvt_length']} | {w['va_pct']} | {w['n_inside']} | {w['cooldown']} | {w['session_start']} | {is_r['pf']:.2f} | {is_r['trades']} | {oos_r['pf']:.2f} | {oos_r['trades']} | {decay:+.1f}% |")

# Neighbor Analysis for Anchor: pvt=10, va=0.6, n=2, cd=5, ss=13
anchor = {"pvt_length": 10, "va_pct": 0.6, "n_inside": 2, "cooldown": 5, "session_start": 13}
print("\nNeighbor Analysis:")
neighbors = [
    # Vary pvt
    {"pvt_length": 15, "va_pct": 0.6, "n_inside": 2, "cooldown": 5, "session_start": 13},
    # Vary va_pct
    {"pvt_length": 10, "va_pct": 0.7, "n_inside": 2, "cooldown": 5, "session_start": 13},
    # Vary n_inside
    {"pvt_length": 10, "va_pct": 0.6, "n_inside": 1, "cooldown": 5, "session_start": 13}, # already in winners
    {"pvt_length": 10, "va_pct": 0.6, "n_inside": 3, "cooldown": 5, "session_start": 13}, 
]

for n in neighbors:
    is_r = find_result(is_data, n)
    oos_r = find_result(oos_data, n)
    if is_r and oos_r:
        decay = (oos_r['pf'] - is_r['pf']) / is_r['pf'] * 100
        print(f"Neighbor {n}: IS_PF={is_r['pf']:.2f}, OOS_PF={oos_r['pf']:.2f}, OOS_T={oos_r['trades']}, Decay={decay:+.1f}%")

# Parameter Effects: pvt_length
print("\nParameter Effects: pvt_length (OOS Avg PF and T)")
for pvt in [10, 15, 20]:
    pvt_results = [r for r in oos_data if r['params']['pvt_length'] == pvt]
    avg_pf = sum(r['pf'] for r in pvt_results) / len(pvt_results)
    avg_t = sum(r['trades'] for r in pvt_results) / len(pvt_results)
    print(f"pvt={pvt}: Avg PF={avg_pf:.2f}, Avg T={avg_t:.1f}")

# Parameter Effects: va_pct
print("\nParameter Effects: va_pct (OOS Avg PF and T)")
for va in [0.6, 0.7, 0.8]:
    va_results = [r for r in oos_data if r['params']['va_pct'] == va]
    avg_pf = sum(r['pf'] for r in va_results) / len(va_results)
    avg_t = sum(r['trades'] for r in va_results) / len(va_results)
    print(f"va={va}: Avg PF={avg_pf:.2f}, Avg T={avg_t:.1f}")
