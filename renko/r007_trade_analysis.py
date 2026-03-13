"""
R007 Trade Analysis — Winners vs Losers

Loads the TradingView trade export CSV and breaks P&L down by:
  - Hour of entry (UTC)        — session effects
  - Session bucket             — Asian / London / NY overlap
  - Trade duration             — quick whipsaws vs real trends
  - Day of week                — Monday / Friday effects
  - Direction                  — Long vs Short bias

Run from repo root:
  python renko/r007_trade_analysis.py
"""

import os
import pandas as pd

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "tvresults",
    "R007_Combined_R001+R002_[Renko]_OANDA_EURUSD_2026-03-06.csv",
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "tvresults",
    "r007_trade_analysis.txt",
)


def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # Each trade # has two rows: one Entry row, one Exit row.
    entry_rows = df[df["Type"].str.contains("Entry", case=False)].copy()
    exit_rows  = df[df["Type"].str.contains("Exit",  case=False)].copy()

    entry_rows = entry_rows.rename(columns={
        "Date and time": "entry_time",
        "Type":          "entry_type",
        "Price USD":     "entry_price",
    })
    exit_rows = exit_rows.rename(columns={
        "Date and time":          "exit_time",
        "Type":                   "exit_type",
        "Price USD":              "exit_price",
        "Net P&L USD":            "pnl",
        "Favorable excursion USD":"mfe",
        "Adverse excursion USD":  "mae",
    })

    trades = entry_rows[["Trade #", "entry_time", "entry_type", "entry_price"]].merge(
        exit_rows[["Trade #", "exit_time", "exit_type", "exit_price", "pnl", "mfe", "mae"]],
        on="Trade #",
    )

    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
    trades["duration_h"] = (
        (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
    )
    trades["entry_hour"] = trades["entry_time"].dt.hour      # UTC
    trades["entry_dow"]  = trades["entry_time"].dt.day_name()
    trades["is_long"]    = trades["entry_type"].str.contains("long", case=False)
    trades["is_winner"]  = trades["pnl"] > 0

    return trades


def session_label(hour: int) -> str:
    if hour < 7:
        return "Asian     (00–07)"
    elif hour < 13:
        return "London    (07–13)"
    elif hour < 22:
        return "Lon+NY    (13–22)"
    else:
        return "NY close  (22–24)"


def calc_pf(sub: pd.DataFrame) -> float:
    g = sub[sub["pnl"] > 0]["pnl"].sum()
    l = abs(sub[sub["pnl"] < 0]["pnl"].sum())
    return g / l if l > 0 else float("inf")


def section(title: str) -> str:
    return f"\n{'='*62}\n  {title}\n{'='*62}"


def main():
    trades = load_trades(CSV_PATH)

    total       = len(trades)
    n_win       = trades["is_winner"].sum()
    n_los       = total - n_win
    net_total   = trades["pnl"].sum()
    pf_total    = calc_pf(trades)
    avg_win     = trades[trades["is_winner"]]["pnl"].mean()
    avg_los     = trades[~trades["is_winner"]]["pnl"].mean()

    lines = []
    out = lines.append  # shorthand

    out(f"R007 IS Trade Analysis  (n=3, cd=10)  |  {total} trades")
    out(f"  Winners : {n_win} ({n_win/total*100:.1f}%)   Losers : {n_los} ({n_los/total*100:.1f}%)")
    out(f"  Net P&L : ${net_total:.2f}   PF : {pf_total:.3f}")
    out(f"  Avg win : ${avg_win:.3f}   Avg loss : ${avg_los:.3f}")
    out(f"  Expectancy: ${net_total/total:.4f} / trade")

    # ── Hour of entry ──────────────────────────────────────────────────
    out(section("P&L by Entry Hour (UTC)"))
    out(f"{'Hr':>3} {'Count':>6} {'Net $':>9} {'WR':>6} {'PF':>6} {'AvgP&L':>8}")
    for h in range(24):
        sub = trades[trades["entry_hour"] == h]
        if len(sub) == 0:
            continue
        net = sub["pnl"].sum()
        wr  = sub["is_winner"].mean() * 100
        pf  = calc_pf(sub)
        avg = sub["pnl"].mean()
        flag = " ◄" if net < 0 else ""
        out(f"{h:>3} {len(sub):>6} {net:>9.2f} {wr:>5.1f}% {pf:>6.2f} {avg:>8.4f}{flag}")

    # ── Session buckets ────────────────────────────────────────────────
    out(section("P&L by Session Bucket"))
    trades["session"] = trades["entry_hour"].apply(session_label)
    sess_order = ["Asian     (00–07)", "London    (07–13)", "Lon+NY    (13–22)", "NY close  (22–24)"]
    out(f"{'Session':<22} {'Count':>6} {'Net $':>9} {'WR':>6} {'PF':>6}")
    for s in sess_order:
        sub = trades[trades["session"] == s]
        if len(sub) == 0:
            continue
        flag = "  ◄ NEGATIVE" if sub["pnl"].sum() < 0 else ""
        out(f"{s:<22} {len(sub):>6} {sub['pnl'].sum():>9.2f} {sub['is_winner'].mean()*100:>5.1f}% {calc_pf(sub):>6.2f}{flag}")

    # ── Trade duration ─────────────────────────────────────────────────
    out(section("Trade Duration — Winners vs Losers"))
    w = trades[trades["is_winner"]]["duration_h"]
    l = trades[~trades["is_winner"]]["duration_h"]
    out(f"  Winners  — mean {w.mean():.1f}h  median {w.median():.1f}h  p10 {w.quantile(0.1):.1f}h  p90 {w.quantile(0.9):.1f}h")
    out(f"  Losers   — mean {l.mean():.1f}h  median {l.median():.1f}h  p10 {l.quantile(0.1):.1f}h  p90 {l.quantile(0.9):.1f}h")

    bins   = [0, 0.5, 1, 2, 4, 8, 16, 9999]
    labels = ["<0.5h", "0.5–1h", "1–2h", "2–4h", "4–8h", "8–16h", "16h+"]
    trades["dur_bucket"] = pd.cut(trades["duration_h"], bins=bins, labels=labels)
    out(f"\n{'Duration':>8} {'Count':>6} {'Net $':>9} {'WR':>6} {'PF':>6}")
    for b in labels:
        sub = trades[trades["dur_bucket"] == b]
        if len(sub) == 0:
            continue
        flag = "  ◄ NEGATIVE" if sub["pnl"].sum() < 0 else ""
        out(f"{b:>8} {len(sub):>6} {sub['pnl'].sum():>9.2f} {sub['is_winner'].mean()*100:>5.1f}% {calc_pf(sub):>6.2f}{flag}")

    # ── MFE / MAE ─────────────────────────────────────────────────────
    out(section("MFE / MAE (Favorable / Adverse Excursion)"))
    out(f"  Winners — avg MFE ${trades[trades['is_winner']]['mfe'].mean():.3f}  avg MAE ${trades[trades['is_winner']]['mae'].mean():.3f}")
    out(f"  Losers  — avg MFE ${trades[~trades['is_winner']]['mfe'].mean():.3f}  avg MAE ${trades[~trades['is_winner']]['mae'].mean():.3f}")
    # Correlation of MFE/MAE with outcome
    corr_mfe = trades["mfe"].corr(trades["pnl"])
    corr_mae = trades["mae"].corr(trades["pnl"])
    out(f"  Corr(MFE, P&L)={corr_mfe:.3f}   Corr(MAE, P&L)={corr_mae:.3f}")

    # ── Day of week ────────────────────────────────────────────────────
    out(section("P&L by Day of Week (entry day)"))
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out(f"{'Day':<12} {'Count':>6} {'Net $':>9} {'WR':>6} {'PF':>6}")
    for d in dow_order:
        sub = trades[trades["entry_dow"] == d]
        if len(sub) == 0:
            continue
        flag = "  ◄ NEGATIVE" if sub["pnl"].sum() < 0 else ""
        out(f"{d:<12} {len(sub):>6} {sub['pnl'].sum():>9.2f} {sub['is_winner'].mean()*100:>5.1f}% {calc_pf(sub):>6.2f}{flag}")

    # ── Long vs Short ──────────────────────────────────────────────────
    out(section("Long vs Short"))
    for is_long, label in [(True, "Long"), (False, "Short")]:
        sub = trades[trades["is_long"] == is_long]
        out(f"  {label:<6}: {len(sub)} trades  Net ${sub['pnl'].sum():.2f}  WR {sub['is_winner'].mean()*100:.1f}%  PF {calc_pf(sub):.3f}")

    # ── Key correlations ───────────────────────────────────────────────
    out(section("Correlations with P&L"))
    for col, label in [
        ("entry_hour", "Entry hour (UTC)"),
        ("duration_h", "Trade duration (h)"),
        ("mfe",        "MFE (favorable excursion)"),
        ("mae",        "MAE (adverse excursion)"),
    ]:
        r = trades[col].corr(trades["pnl"])
        out(f"  {label:<30} r = {r:+.3f}")

    # ── Worst hours summary ────────────────────────────────────────────
    out(section("Worst Hours (net negative)"))
    by_hour_net = trades.groupby("entry_hour")["pnl"].sum()
    neg_hours   = by_hour_net[by_hour_net < 0].sort_values()
    if len(neg_hours) == 0:
        out("  None — all hours net positive!")
    else:
        for h, net in neg_hours.items():
            sub = trades[trades["entry_hour"] == h]
            out(f"  Hour {h:02d}  Net ${net:.2f}  {len(sub)} trades  WR {sub['is_winner'].mean()*100:.0f}%")

    out("")  # trailing newline

    result = "\n".join(lines)
    print(result)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[Saved -> {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
