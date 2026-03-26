import json

with open('ai_context/ea005_ext_oos_results.json', 'r') as f:
    oos_data = json.load(f)

print("Parameter Effects: session_start (OOS Avg PF and T)")
for ss in [0, 13]:
    ss_results = [r for r in oos_data if r['params']['session_start'] == ss]
    avg_pf = sum(r['pf'] for r in ss_results) / len(ss_results)
    avg_t = sum(r['trades'] for r in ss_results) / len(ss_results)
    # Count how many have PF > 8.0 and T > 60
    winners_count = len([r for r in ss_results if r['pf'] > 8.0 and r['trades'] > 60])
    print(f"ss={ss}: Avg PF={avg_pf:.2f}, Avg T={avg_t:.1f}, High-Performers (PF>8, T>60)={winners_count}")

# Check the best OOS PF for each session_start
best_ss0 = max([r['pf'] for r in oos_data if r['params']['session_start'] == 0])
best_ss13 = max([r['pf'] for r in oos_data if r['params']['session_start'] == 13])
print(f"Best OOS PF (ss=0): {best_ss0:.2f}")
print(f"Best OOS PF (ss=13): {best_ss13:.2f}")
