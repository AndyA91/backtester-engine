#!/usr/bin/env python3
"""
euraud_htf_sweep.py -- EURAUD targeted sweep with HTF ADX gating

Tests all signal types + LTF gates + HTF ADX gates to find if we can beat
EA021 (escgo_cross + HTF ADX>=30, TV WR=68.4%).

Cross-asset sweep found R007 n3 + adx40 = 76.5% WR without HTF.
This adds HTF to see if we can push higher.

Usage:
    python renko/euraud_htf_sweep.py
"""

import contextlib
import io
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from renko.data import load_renko_export
from renko.indicators import add_renko_indicators
from renko.phase6_enrichment import add_phase6_indicators
from engine import BacktestConfig, run_backtest

# Config
LTF_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"
HTF_FILES = {
    "0.0012": "OANDA_EURAUD, 1S renko 0.0012.csv",
    "0.0018": "OANDA_EURAUD, 1S renko 0.0018.csv",
}
IS_START = "2023-07-20"
IS_END = "2025-09-30"
OOS_START = "2025-10-01"
OOS_END = "2026-03-17"
OOS_DAYS = 168
COMMISSION = 0.009
CAPITAL = 1000.0


def main():
    # Load data
    print("Loading LTF...")
    df_ltf = load_renko_export(LTF_FILE)
    add_renko_indicators(df_ltf)
    add_phase6_indicators(df_ltf, include_mk=False)

    htf_dfs = {}
    for label, hfile in HTF_FILES.items():
        print(f"Loading HTF {label}...")
        df_h = load_renko_export(hfile)
        add_renko_indicators(df_h)
        htf_dfs[label] = df_h

    n = len(df_ltf)
    brick_up = df_ltf["brick_up"].values

    # -- Signals --
    signals = {}

    for nb in [3, 5, 9]:
        r001_l = np.zeros(n, dtype=bool)
        r001_s = np.zeros(n, dtype=bool)
        r002_l = np.zeros(n, dtype=bool)
        r002_s = np.zeros(n, dtype=bool)
        for i in range(nb + 1, n):
            all_up = all(brick_up[i - j] for j in range(nb))
            all_dn = all(not brick_up[i - j] for j in range(nb))
            prev_dn = all(not brick_up[i - 1 - j] for j in range(nb))
            prev_up = all(brick_up[i - 1 - j] for j in range(nb))
            if all_up:
                r001_l[i] = True
            if all_dn:
                r001_s[i] = True
            if prev_dn and brick_up[i]:
                r002_l[i] = True
            if prev_up and not brick_up[i]:
                r002_s[i] = True
        signals[f"R001_n{nb}"] = (r001_l, r001_s)
        signals[f"R002_n{nb}"] = (r002_l, r002_s)
        signals[f"R007_n{nb}"] = (r001_l | r002_l, r001_s | r002_s)

    # P6 signals
    psar_dir = df_ltf["psar_dir"].values
    stk = df_ltf["stoch_k"].values

    for sig_name, make_l, make_s in [
        ("psar_flip",
         lambda i: np.isnan(psar_dir[i]) or psar_dir[i] > 0,
         lambda i: np.isnan(psar_dir[i]) or psar_dir[i] < 0),
        ("stoch_cross",
         lambda i: not np.isnan(stk[i]) and stk[i] < 25,
         lambda i: not np.isnan(stk[i]) and stk[i] > 75),
    ]:
        sl = np.zeros(n, dtype=bool)
        ss = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if brick_up[i] and make_l(i):
                sl[i] = True
            if not brick_up[i] and make_s(i):
                ss[i] = True
        signals[sig_name] = (sl, ss)

    # ESCGO
    if "escgo_fast" in df_ltf.columns:
        ef = df_ltf["escgo_fast"].values
        es = df_ltf["escgo_slow"].values
        el = np.zeros(n, dtype=bool)
        ess = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(ef[i]) and not np.isnan(es[i]):
                if brick_up[i] and ef[i] > es[i]:
                    el[i] = True
                if not brick_up[i] and ef[i] < es[i]:
                    ess[i] = True
        signals["escgo_cross"] = (el, ess)

    # EMA cross
    ema9 = df_ltf["ema9"].values
    ema21 = df_ltf["ema21"].values
    eml = np.zeros(n, dtype=bool)
    ems = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(ema9[i]) and not np.isnan(ema21[i]):
            if brick_up[i] and ema9[i] > ema21[i]:
                eml[i] = True
            if not brick_up[i] and ema9[i] < ema21[i]:
                ems[i] = True
    signals["ema_cross"] = (eml, ems)

    # -- LTF Gates --
    adx = df_ltf["adx"].values
    psar_l = np.array([np.isnan(psar_dir[i]) or psar_dir[i] > 0 for i in range(n)])
    psar_s = np.array([np.isnan(psar_dir[i]) or psar_dir[i] < 0 for i in range(n)])

    ltf_gates = {}
    ltf_gates["none"] = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))
    ltf_gates["psar"] = (psar_l, psar_s)
    for at in [25, 30, 40, 50]:
        ok = np.array([np.isnan(adx[i]) or adx[i] >= at for i in range(n)])
        ltf_gates[f"adx{at}"] = (ok, ok)
    ltf_gates["psar+adx30"] = (psar_l & ltf_gates["adx30"][0], psar_s & ltf_gates["adx30"][1])
    ltf_gates["psar+adx40"] = (psar_l & ltf_gates["adx40"][0], psar_s & ltf_gates["adx40"][1])

    # -- HTF Gates --
    def align_htf(df_h, threshold):
        h_adx = df_h["adx"].values
        h_nan = np.isnan(h_adx)
        h_ok = h_nan | (h_adx >= threshold)
        htf_frame = pd.DataFrame({
            "t": df_h.index.values, "ok": h_ok.astype(float)
        }).sort_values("t")
        ltf_frame = pd.DataFrame({"t": df_ltf.index.values}).sort_values("t")
        merged = pd.merge_asof(ltf_frame, htf_frame, on="t", direction="backward")
        vals = merged["ok"].values
        return np.where(np.isnan(vals), True, vals > 0.5).astype(bool)

    htf_gates = {"no_htf": np.ones(n, dtype=bool)}
    for hlabel, df_h in htf_dfs.items():
        for ht in [25, 30, 35, 40, 45]:
            htf_gates[f"htf_{hlabel}_adx{ht}"] = align_htf(df_h, ht)

    print(f"Signals: {len(signals)}, LTF gates: {len(ltf_gates)}, HTF gates: {len(htf_gates)}")

    # -- Run all combos --
    cooldowns = [3, 10, 20, 30, 45]
    total = len(signals) * len(ltf_gates) * len(htf_gates) * len(cooldowns)
    print(f"Total combos: {total}", flush=True)

    results = []
    done = 0
    for sig_name, (sig_l, sig_s) in signals.items():
        for gate_name, (gate_l, gate_s) in ltf_gates.items():
            for htf_name, htf_ok in htf_gates.items():
                for cd in cooldowns:
                    gl = sig_l & gate_l & htf_ok
                    gs = sig_s & gate_s & htf_ok

                    # Apply cooldown + exit
                    le = np.zeros(n, dtype=bool)
                    lx = np.zeros(n, dtype=bool)
                    se = np.zeros(n, dtype=bool)
                    sx = np.zeros(n, dtype=bool)
                    in_pos = False
                    pos_dir = 0
                    last_entry = -999

                    for i in range(1, n):
                        up = bool(brick_up[i])
                        if in_pos:
                            if (pos_dir == 1 and not up) or (pos_dir == -1 and up):
                                if pos_dir == 1:
                                    lx[i] = True
                                else:
                                    sx[i] = True
                                in_pos = False
                            continue
                        if i - last_entry < cd:
                            continue
                        if gl[i] and up:
                            le[i] = True
                            in_pos = True
                            pos_dir = 1
                            last_entry = i
                        elif gs[i] and not up:
                            se[i] = True
                            in_pos = True
                            pos_dir = -1
                            last_entry = i

                    # Backtest IS + OOS
                    row = {}
                    for period, start, end in [("is", IS_START, IS_END), ("oos", OOS_START, OOS_END)]:
                        df2 = df_ltf.copy()
                        df2["long_entry"] = le
                        df2["long_exit"] = lx
                        df2["short_entry"] = se
                        df2["short_exit"] = sx
                        cfg = BacktestConfig(
                            initial_capital=CAPITAL, commission_pct=COMMISSION,
                            slippage_ticks=0, qty_type="cash", qty_value=20,
                            pyramiding=1, start_date=start, end_date=end,
                            take_profit_pct=0.0, stop_loss_pct=0.0,
                        )
                        with contextlib.redirect_stdout(io.StringIO()):
                            kpis = run_backtest(df2, cfg)
                        pf = kpis.get("profit_factor", 0) or 0
                        row[period] = {
                            "pf": float("inf") if math.isinf(pf) else round(float(pf), 2),
                            "trades": int(kpis.get("total_trades", 0) or 0),
                            "wr": round(float(kpis.get("win_rate", 0) or 0), 1),
                        }

                    label = f"{sig_name}_{gate_name}_{htf_name}_cd{cd}"
                    results.append({"label": label, "signal": sig_name,
                                    "gate": gate_name, "htf": htf_name,
                                    "cd": cd, **row})
                    done += 1
                    if done % 5000 == 0:
                        print(f"  {done}/{total}...", flush=True)

    # -- Results --
    viable = [r for r in results if r["oos"]["trades"] >= 10]
    viable.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]), reverse=True)

    print(f"\nTop 30 by OOS WR (trades >= 10):")
    hdr = f"{'Config':<75} {'IS PF':>7} {'IS WR':>6} {'OOS PF':>7} {'OOS T':>5} {'t/d':>5} {'OOS WR':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in viable[:30]:
        tpd = r["oos"]["trades"] / OOS_DAYS
        pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        is_pf = "inf" if r["is"]["pf"] == float("inf") else f"{r['is']['pf']:.1f}"
        print(f"{r['label']:<75} {is_pf:>7} {r['is']['wr']:>5.1f}% "
              f"{pf_s:>7} {r['oos']['trades']:>5} {tpd:>5.1f} {r['oos']['wr']:>6.1f}%")

    # Show HTF vs no-HTF for best signal
    print(f"\n--- HTF Impact on R007_n3 + adx40 ---")
    r007_configs = [r for r in viable if "R007_n3" in r["signal"] and "adx40" in r["gate"]]
    r007_configs.sort(key=lambda r: (r["oos"]["wr"], r["oos"]["pf"]), reverse=True)
    for r in r007_configs[:15]:
        tpd = r["oos"]["trades"] / OOS_DAYS
        pf_s = "inf" if r["oos"]["pf"] == float("inf") else f"{r['oos']['pf']:.1f}"
        print(f"  {r['label']:<70} PF={pf_s:>6} T={r['oos']['trades']:>4} WR={r['oos']['wr']:>5.1f}%")

    print(f"\nBaseline: EA021 escgo+HTF = TV PF=14.21, 19t, WR=68.4%")
    print(f"Total: {len(results)} run, {len(viable)} viable (>=10t OOS)")

    out = ROOT / "ai_context" / "euraud_htf_sweep_results.json"
    with open(out, "w") as f:
        json.dump({"total": len(results), "viable": len(viable),
                    "top30": viable[:30]}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
