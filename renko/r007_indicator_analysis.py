"""
R007 Indicator Analysis — correlate Renko indicators at entry with P&L / duration.

Loads the Renko brick dataframe (with full indicator set), joins to the R007
TV trade export by timestamp, then reports which indicators best predict
winner vs loser and long vs short trade duration.

Run from repo root:
  python renko/r007_indicator_analysis.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renko.data       import load_renko_export
from renko.indicators import add_renko_indicators

TV_CSV   = ROOT / "tvresults" / "R007_Combined_R001+R002_[Renko]_OANDA_EURUSD_2026-03-06.csv"
DATA_CSV = ROOT / "data" / "OANDA_EURUSD, 1S renko 0.0004.csv"
OUT_PATH = ROOT / "tvresults" / "r007_indicator_analysis.txt"

IS_START = "2023-01-23"
IS_END   = "2025-09-30"

# Indicators to evaluate (all are pre-shifted — safe to read at entry bar)
INDICATORS = [
    ("adx",         "ADX(14)               "),
    ("chop",        "CHOP(14)              "),
    ("rsi",         "RSI(14)               "),
    ("bb_bw",       "BB Bandwidth(20)      "),
    ("bb_pct_b",    "BB %B(20)             "),
    ("st_dir",      "Supertrend dir(10,3)  "),
    ("sq_on",       "Squeeze active        "),
    ("sq_momentum", "Squeeze momentum      "),
    ("kama_slope",  "KAMA slope(10)        "),
    ("macd_hist",   "MACD histogram        "),
    ("vol_ratio",   "Volume ratio(20)      "),
    ("mfi",         "MFI(14)               "),
    ("stoch_k",     "Stoch %K(14,3)        "),
]


def load_tv_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    entry = df[df["Type"].str.contains("Entry", case=False)].copy()
    exit_ = df[df["Type"].str.contains("Exit",  case=False)].copy()
    entry = entry.rename(columns={"Date and time": "entry_time", "Type": "entry_type",
                                   "Price USD": "entry_price"})
    exit_ = exit_.rename(columns={"Date and time": "exit_time",  "Type": "exit_type",
                                   "Price USD": "exit_price",    "Net P&L USD": "pnl",
                                   "Favorable excursion USD": "mfe",
                                   "Adverse excursion USD": "mae"})
    trades = entry[["Trade #", "entry_time", "entry_type", "entry_price"]].merge(
        exit_[["Trade #", "exit_time", "exit_type", "exit_price", "pnl", "mfe", "mae"]],
        on="Trade #",
    )
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
    trades["duration_h"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
    trades["is_winner"]  = trades["pnl"] > 0
    trades["is_long"]    = trades["entry_type"].str.contains("long", case=False)
    return trades


def calc_pf(sub: pd.DataFrame) -> float:
    g = sub[sub["pnl"] > 0]["pnl"].sum()
    l = abs(sub[sub["pnl"] < 0]["pnl"].sum())
    return g / l if l > 0 else float("inf")


def section(title: str) -> str:
    return f"\n{'='*64}\n  {title}\n{'='*64}"


def main():
    # ── Load Renko data with indicators ───────────────────────────────────────
    print("Loading Renko data + indicators...")
    df = load_renko_export(str(DATA_CSV))
    df = add_renko_indicators(df)
    df_is = df[(df.index >= IS_START) & (df.index <= IS_END)].copy()
    print(f"  Renko IS bars: {len(df_is)}")

    # ── Load TV trades ────────────────────────────────────────────────────────
    trades = load_tv_trades(str(TV_CSV))
    # Keep only IS trades (entry within IS period)
    trades = trades[
        (trades["entry_time"] >= pd.Timestamp(IS_START)) &
        (trades["entry_time"] <= pd.Timestamp(IS_END))
    ].copy()
    print(f"  TV IS trades: {len(trades)}")

    # ── Join trades to Renko bars by entry timestamp ──────────────────────────
    # Renko bars are indexed by time; TV timestamps are in the same timezone.
    # merge_asof: for each trade, find the last Renko bar at or before entry_time.
    df_is_reset = df_is.reset_index()
    # Index name after load_renko_export is "time"
    time_col = df_is_reset.columns[0]  # whatever the index was named
    df_is_reset = df_is_reset.rename(columns={time_col: "bar_time"})
    df_is_reset["bar_time"] = pd.to_datetime(df_is_reset["bar_time"]).astype("datetime64[ns]")
    trades_sorted = trades.sort_values("entry_time")
    trades_sorted["entry_time"] = trades_sorted["entry_time"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        trades_sorted,
        df_is_reset[["bar_time"] + [c for c, _ in INDICATORS]],
        left_on="entry_time",
        right_on="bar_time",
        direction="backward",
    )

    n_joined = merged.dropna(subset=["adx"]).shape[0]
    print(f"  Joined: {n_joined} / {len(trades)} trades matched to Renko bars")
    merged = merged.dropna(subset=["adx"])

    lines = []
    out = lines.append

    out(f"R007 Indicator Analysis at Entry  ({len(merged)} IS trades)")
    out(f"Winners: {merged['is_winner'].sum()}  Losers: {(~merged['is_winner']).sum()}")
    out(f"Baseline PF: {calc_pf(merged):.3f}  WR: {merged['is_winner'].mean()*100:.1f}%")

    # ── Correlation table ─────────────────────────────────────────────────────
    out(section("Correlations with P&L and Duration at Entry"))
    out(f"{'Indicator':<26} {'r(P&L)':>8} {'r(dur)':>8}  {'WIN avg':>9}  {'LOS avg':>9}")
    results = []
    for col, label in INDICATORS:
        if col not in merged.columns:
            continue
        r_pnl = merged[col].corr(merged["pnl"])
        r_dur = merged[col].corr(merged["duration_h"])
        w_avg = merged[merged["is_winner"]][col].mean()
        l_avg = merged[~merged["is_winner"]][col].mean()
        results.append((col, label, r_pnl, r_dur, w_avg, l_avg))
        out(f"{label} {r_pnl:>+8.3f} {r_dur:>+8.3f}  {w_avg:>9.3f}  {l_avg:>9.3f}")

    # Sort by abs correlation with P&L
    results.sort(key=lambda x: abs(x[2]), reverse=True)

    # ── CHOP threshold sweep ───────────────────────────────────────────────────
    out(section("CHOP(14) Threshold Sweep  — skip if chop >= threshold"))
    out(f"{'Threshold':>10} {'Trades':>7} {'Skipped':>8} {'Net $':>8} {'WR':>6} {'PF':>6}")
    for thresh in [50, 55, 60, 65, 70, 75]:
        sub = merged[merged["chop"] < thresh]
        if len(sub) == 0:
            continue
        skipped = len(merged) - len(sub)
        out(f"{thresh:>10} {len(sub):>7} {skipped:>8} {sub['pnl'].sum():>8.2f} "
            f"{sub['is_winner'].mean()*100:>5.1f}% {calc_pf(sub):>6.2f}")

    # ── ADX threshold sweep ────────────────────────────────────────────────────
    out(section("ADX(14) Threshold Sweep  — skip if adx < threshold"))
    out(f"{'Threshold':>10} {'Trades':>7} {'Skipped':>8} {'Net $':>8} {'WR':>6} {'PF':>6}")
    for thresh in [0, 15, 20, 25, 30, 35]:
        sub = merged[merged["adx"] >= thresh]
        if len(sub) == 0:
            continue
        skipped = len(merged) - len(sub)
        out(f"{thresh:>10} {len(sub):>7} {skipped:>8} {sub['pnl'].sum():>8.2f} "
            f"{sub['is_winner'].mean()*100:>5.1f}% {calc_pf(sub):>6.2f}")

    # ── Supertrend direction gate ──────────────────────────────────────────────
    out(section("Supertrend Direction Gate  — only trade WITH st_dir"))
    sub_agree = merged[
        ((merged["is_long"]) & (merged["st_dir"] == 1)) |
        ((~merged["is_long"]) & (merged["st_dir"] == -1))
    ]
    sub_oppose = merged[
        ((merged["is_long"]) & (merged["st_dir"] == -1)) |
        ((~merged["is_long"]) & (merged["st_dir"] == 1))
    ]
    out(f"  Agree with ST : {len(sub_agree)} trades  Net ${sub_agree['pnl'].sum():.2f}  "
        f"WR {sub_agree['is_winner'].mean()*100:.1f}%  PF {calc_pf(sub_agree):.3f}")
    out(f"  Against ST    : {len(sub_oppose)} trades  Net ${sub_oppose['pnl'].sum():.2f}  "
        f"WR {sub_oppose['is_winner'].mean()*100:.1f}%  PF {calc_pf(sub_oppose):.3f}")

    # ── Squeeze gate ───────────────────────────────────────────────────────────
    out(section("Squeeze Gate  — skip trades entered while squeeze is active"))
    sub_nosq = merged[~merged["sq_on"].astype(bool)]
    sub_sq   = merged[merged["sq_on"].astype(bool)]
    out(f"  No squeeze (trade): {len(sub_nosq)} trades  Net ${sub_nosq['pnl'].sum():.2f}  "
        f"WR {sub_nosq['is_winner'].mean()*100:.1f}%  PF {calc_pf(sub_nosq):.3f}")
    out(f"  Squeeze active     : {len(sub_sq)} trades  Net ${sub_sq['pnl'].sum():.2f}  "
        f"WR {sub_sq['is_winner'].mean()*100:.1f}%  PF {calc_pf(sub_sq):.3f}")

    # ── BB Bandwidth sweep ─────────────────────────────────────────────────────
    out(section("BB Bandwidth  — skip if bb_bw < threshold (tight = choppy)"))
    bw_vals = merged["bb_bw"].dropna()
    out(f"  bb_bw range: {bw_vals.min():.4f} – {bw_vals.max():.4f}  "
        f"median {bw_vals.median():.4f}  p25 {bw_vals.quantile(0.25):.4f}")
    out(f"{'Threshold':>10} {'Trades':>7} {'Skipped':>8} {'Net $':>8} {'WR':>6} {'PF':>6}")
    for pct in [10, 20, 30, 40]:
        thresh = float(bw_vals.quantile(pct / 100))
        sub = merged[merged["bb_bw"] >= thresh]
        skipped = len(merged) - len(sub)
        out(f"  p{pct} ({thresh:.4f}) {len(sub):>7} {skipped:>8} {sub['pnl'].sum():>8.2f} "
            f"{sub['is_winner'].mean()*100:>5.1f}% {calc_pf(sub):>6.2f}")

    # ── Combined: best indicators ──────────────────────────────────────────────
    out(section("Combined: CHOP < 61.8 AND ADX >= 20"))
    sub = merged[(merged["chop"] < 61.8) & (merged["adx"] >= 20)]
    skipped = len(merged) - len(sub)
    out(f"  {len(sub)} trades, {skipped} skipped  Net ${sub['pnl'].sum():.2f}  "
        f"WR {sub['is_winner'].mean()*100:.1f}%  PF {calc_pf(sub):.3f}")

    out(section("Combined: CHOP < 61.8 AND Supertrend agree"))
    sub = merged[
        (merged["chop"] < 61.8) &
        (((merged["is_long"]) & (merged["st_dir"] == 1)) |
         ((~merged["is_long"]) & (merged["st_dir"] == -1)))
    ]
    skipped = len(merged) - len(sub)
    out(f"  {len(sub)} trades, {skipped} skipped  Net ${sub['pnl'].sum():.2f}  "
        f"WR {sub['is_winner'].mean()*100:.1f}%  PF {calc_pf(sub):.3f}")

    out("")

    result = "\n".join(lines)
    print(result)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[Saved -> {OUT_PATH}]")


if __name__ == "__main__":
    main()
