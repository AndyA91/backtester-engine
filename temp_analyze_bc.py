import json
import pandas as pd

def analyze_sweep(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Average OOS PF per instrument and gate
    summary = df.groupby(['instrument', 'gate'])['oos_pf'].agg(['mean', 'max', 'count']).unstack(0)
    
    # Identify gates that beat instrument avg on 2+ pairs
    instr_avg = df.groupby('instrument')['oos_pf'].mean()
    print("Instrument Baselines (Avg OOS PF):")
    print(instr_avg)
    print("\n" + "="*50 + "\n")
    
    # Check each gate vs mean
    gates = df['gate'].unique()
    for gate in gates:
        gate_data = df[df['gate'] == gate]
        perf = gate_data.groupby('instrument')['oos_pf'].mean()
        
        wins = []
        for inst in instr_avg.index:
            if inst in perf and perf[inst] > instr_avg[inst]:
                wins.append(inst)
        
        if len(wins) >= 2:
            print(f"GATE: {gate.upper()} [CONSENSUS WINNER]")
            print(f"Wins on: {', '.join(wins)}")
            print(perf)
            print("-" * 30)

if __name__ == "__main__":
    analyze_sweep(r'c:\Users\float\Desktop\VS code\backtester-engine\ai_context\bc_sweep_results.json')
