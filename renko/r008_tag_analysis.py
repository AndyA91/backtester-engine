"""
R008 Tag Analysis — parse indicator snapshots from TV trade export comment field.

Each entry row Signal column contains a tag like:
  [R001L 5A:31 bA:28 R:58 V:1.15 B:0.74 M:+ ST:U]

Parses these directly — no need to reload Renko data.
Key addition vs R007 analysis: signal-type breakdown (R001 vs R002, L vs S).

Run from repo root:
  python renko/r008_tag_analysis.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

TV_CSV   = ROOT / "tvresults" / "R008_R007_+_5m_Candle_ADX_Gate_[Renko]_OANDA_EURUSD_2026-03-07 (1).csv"
OUT_PATH = ROOT / "tvresults" / "r008_tag_analysis_vol15.txt"

TAG_RE = re.compile(
    r"\[(?P<sig>R\d{3}[LS])"
    r"\s+5A:(?P<adx5>[\d.]+)"
    r"\s+bA:(?P<adx_b>[\d.]+)"
    r"\s+R:(?P<rsi>[\d.]+)"
    r"\s+V:(?P<vol>[\d.]+)"
    r"\s+B:(?P<bb>[\d.]+)"
    r"\s+M:(?P<macd>[+\-])"
    r"\s+ST:(?P<st>[UD])"
    r"\]"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def calc_pf(sub: pd.DataFrame) -> float:
    g = sub[sub["pnl"] > 0]["pnl"].sum()
    l = abs(sub[sub["pnl"] < 0]["pnl"].sum())
    return g / l if l > 0 else float("inf")


def stats(sub: pd.DataFrame, label: str) -> str:
    n   = len(sub)
    wr  = sub["is_winner"].mean() * 100 if n > 0 else 0
    net = sub["pnl"].sum()
    pf  = calc_pf(sub)
    return f"  {label:<28} {n:>5}t  Net ${net:>8.2f}  WR {wr:>4.1f}%  PF {pf:>6.2f}"


def section(title: str) -> str:
    return f"\n{'='*64}\n  {title}\n{'='*64}"


# ── Load + parse ──────────────────────────────────────────────────────────────

def load_tagged_trades(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]

    entry = raw[raw["Type"].str.contains("Entry", case=False)].copy()
    exit_ = raw[raw["Type"].str.contains("Exit",  case=False)].copy()

    entry = entry.rename(columns={
        "Date and time": "entry_time",
        "Type":          "entry_type",
        "Price USD":     "entry_price",
        "Signal":        "signal",
    })
    exit_ = exit_.rename(columns={
        "Date and time":          "exit_time",
        "Net P&L USD":            "pnl",
        "Favorable excursion USD":"mfe",
        "Adverse excursion USD":  "mae",
    })

    trades = entry[["Trade #", "entry_time", "entry_type", "entry_price", "signal"]].merge(
        exit_[["Trade #", "exit_time", "pnl", "mfe", "mae"]],
        on="Trade #",
    )
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
    trades["duration_h"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
    trades["is_winner"]  = trades["pnl"] > 0
    trades["is_long"]    = trades["entry_type"].str.contains("long", case=False)
    trades["hour"]       = trades["entry_time"].dt.hour
    trades["dow"]        = trades["entry_time"].dt.day_name()

    # ── Parse tag fields ──────────────────────────────────────────────────────
    parsed = trades["signal"].str.extract(TAG_RE)
    for col in ["adx5", "adx_b", "rsi", "vol", "bb"]:
        parsed[col] = pd.to_numeric(parsed[col], errors="coerce")

    trades = pd.concat([trades, parsed], axis=1)

    # Derived fields
    trades["sig_type"]   = trades["sig"].str[:4]   # R001 / R002
    trades["sig_dir"]    = trades["sig"].str[4]    # L / S
    trades["macd_bull"]  = trades["macd"] == "+"
    trades["st_up"]      = trades["st"]   == "U"
    trades["aligned"]    = (trades["is_long"] & trades["st_up"]) | (~trades["is_long"] & ~trades["st_up"])

    return trades


def main():
    print(f"Loading {TV_CSV.name}...")
    t = load_tagged_trades(TV_CSV)
    tagged = t.dropna(subset=["sig"]).copy()
    untagged = t[t["sig"].isna()]
    print(f"  Total trades: {len(t)}  Tagged: {len(tagged)}  Untagged: {len(untagged)}")

    # Date range
    print(f"  Date range: {tagged['entry_time'].min().date()} to {tagged['entry_time'].max().date()}")

    lines = []
    out = lines.append

    out(f"R008 Tag Analysis  ({len(tagged)} tagged trades)")
    out(f"Date: {tagged['entry_time'].min().date()} to {tagged['entry_time'].max().date()}")
    out(f"Winners: {tagged['is_winner'].sum()}  Losers: {(~tagged['is_winner']).sum()}")
    out(f"Baseline PF: {calc_pf(tagged):.3f}  WR: {tagged['is_winner'].mean()*100:.1f}%")
    out(f"Net P&L: ${tagged['pnl'].sum():.2f}")

    # ── Signal type breakdown ─────────────────────────────────────────────────
    out(section("Signal Type Breakdown"))
    out(f"  {'Signal':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        for d, label in [("L", "Long"), ("S", "Short")]:
            sub = tagged[(tagged["sig_type"] == sig) & (tagged["sig_dir"] == d)]
            if len(sub) == 0:
                continue
            out(stats(sub, f"{sig} {label}"))
    out("")
    for sig in ["R001", "R002"]:
        sub = tagged[tagged["sig_type"] == sig]
        out(stats(sub, f"{sig} total"))

    # ── 5m ADX range within filtered trades ───────────────────────────────────
    out(section("5m ADX(14) Distribution  (gate already applied >= threshold)"))
    adx5_vals = tagged["adx5"].dropna()
    out(f"  Range: {adx5_vals.min():.0f} – {adx5_vals.max():.0f}  "
        f"median {adx5_vals.median():.0f}  mean {adx5_vals.mean():.0f}")
    out(f"  {'Bucket':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    buckets = [(25,30), (30,35), (35,40), (40,50), (50,70), (70,100)]
    for lo, hi in buckets:
        sub = tagged[(tagged["adx5"] >= lo) & (tagged["adx5"] < hi)]
        if len(sub) == 0:
            continue
        out(stats(sub, f"5A [{lo}–{hi})"))

    # ── 5m ADX by signal type ─────────────────────────────────────────────────
    out(section("5m ADX by Signal Type  (does gate quality differ?)"))
    out(f"  {'Slice':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        for lo, hi in [(25, 35), (35, 50), (50, 100)]:
            sub = tagged[(tagged["sig_type"] == sig) & (tagged["adx5"] >= lo) & (tagged["adx5"] < hi)]
            if len(sub) == 0:
                continue
            out(stats(sub, f"{sig} 5A [{lo}–{hi})"))

    # ── Brick ADX sweep ────────────────────────────────────────────────────────
    out(section("Brick ADX(14) Threshold Sweep  — skip if bA < threshold"))
    out(f"  {'Threshold':>10} {'Trades':>7} {'Skipped':>8} {'Net $':>8} {'WR':>6} {'PF':>6}")
    for thresh in [0, 20, 25, 30, 35, 40]:
        sub = tagged[tagged["adx_b"] >= thresh]
        skipped = len(tagged) - len(sub)
        out(f"  {thresh:>10} {len(sub):>7} {skipped:>8} {sub['pnl'].sum():>8.2f} "
            f"{sub['is_winner'].mean()*100:>5.1f}% {calc_pf(sub):>6.2f}")

    # ── RSI buckets ────────────────────────────────────────────────────────────
    out(section("RSI(14) Buckets at Entry"))
    out(f"  {'Bucket':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    rsi_buckets = [(0,30,"<30 oversold"), (30,40,"30–40"), (40,50,"40–50"),
                   (50,60,"50–60"), (60,70,"60–70"), (70,100,">70 overbought")]
    for lo, hi, label in rsi_buckets:
        sub = tagged[(tagged["rsi"] >= lo) & (tagged["rsi"] < hi)]
        if len(sub) == 0:
            continue
        out(stats(sub, f"RSI {label}"))

    # ── RSI by signal type ────────────────────────────────────────────────────
    out(section("RSI by Signal Type  (reversal vs momentum context)"))
    out(f"  {'Slice':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        for lo, hi, label in rsi_buckets:
            sub = tagged[(tagged["sig_type"] == sig) & (tagged["rsi"] >= lo) & (tagged["rsi"] < hi)]
            if len(sub) == 0:
                continue
            out(stats(sub, f"{sig} RSI {label}"))

    # ── Volume ratio ──────────────────────────────────────────────────────────
    out(section("Volume Ratio  (vol / EMA20 of vol)"))
    out(f"  vol_ratio range: {tagged['vol'].min():.2f} – {tagged['vol'].max():.2f}  "
        f"median {tagged['vol'].median():.2f}")
    out(f"  {'Bucket':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    vol_buckets = [(0, 0.5, "<0.5 quiet"), (0.5, 1.0, "0.5–1.0"),
                   (1.0, 1.5, "1.0–1.5"), (1.5, 2.5, "1.5–2.5"), (2.5, 999, ">2.5 spike")]
    for lo, hi, label in vol_buckets:
        sub = tagged[(tagged["vol"] >= lo) & (tagged["vol"] < hi)]
        if len(sub) == 0:
            continue
        out(stats(sub, f"V {label}"))

    # ── BB %B ─────────────────────────────────────────────────────────────────
    out(section("BB %B(20,2) at Entry  (0=lower, 0.5=mid, 1=upper)"))
    out(f"  {'Bucket':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    bb_buckets = [(0, 0.2, "<0.2 near lower"), (0.2, 0.4, "0.2–0.4"),
                  (0.4, 0.6, "0.4–0.6 mid"), (0.6, 0.8, "0.6–0.8"),
                  (0.8, 1.01, ">0.8 near upper")]
    for lo, hi, label in bb_buckets:
        sub = tagged[(tagged["bb"] >= lo) & (tagged["bb"] < hi)]
        if len(sub) == 0:
            continue
        out(stats(sub, f"B {label}"))

    # ── BB %B by signal type ──────────────────────────────────────────────────
    out(section("BB %B by Signal Type  (where in band do each signal fire?)"))
    out(f"  {'Slice':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        for lo, hi, label in bb_buckets:
            sub = tagged[(tagged["sig_type"] == sig) & (tagged["bb"] >= lo) & (tagged["bb"] < hi)]
            if len(sub) == 0:
                continue
            out(stats(sub, f"{sig} B {label}"))

    # ── MACD alignment ────────────────────────────────────────────────────────
    out(section("MACD Histogram Sign  — momentum alignment at entry"))
    out(f"  {'Slice':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        for bull, label in [(True, "M:+ bull"), (False, "M:- bear")]:
            sub = tagged[(tagged["sig_type"] == sig) & (tagged["macd_bull"] == bull)]
            if len(sub) == 0:
                continue
            out(stats(sub, f"{sig} {label}"))

    # ── Supertrend alignment ──────────────────────────────────────────────────
    out(section("Supertrend Alignment  (trade WITH vs AGAINST ST direction)"))
    out(f"  {'Slice':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        sub_with    = tagged[(tagged["sig_type"] == sig) &  tagged["aligned"]]
        sub_against = tagged[(tagged["sig_type"] == sig) & ~tagged["aligned"]]
        out(stats(sub_with,    f"{sig} WITH ST"))
        out(stats(sub_against, f"{sig} AGAINST ST"))

    # ── Session breakdown ─────────────────────────────────────────────────────
    out(section("P&L by Session Bucket  (entry UTC hour)"))
    out(f"  {'Session':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    sessions = [
        ("Asian     (00–07)", (0,  7)),
        ("London    (07–13)", (7,  13)),
        ("Lon+NY    (13–22)", (13, 22)),
        ("NY close  (22–24)", (22, 24)),
    ]
    for label, (h0, h1) in sessions:
        sub = tagged[(tagged["hour"] >= h0) & (tagged["hour"] < h1)]
        if len(sub) == 0:
            continue
        out(stats(sub, label))

    # ── Session by signal type ────────────────────────────────────────────────
    out(section("Session by Signal Type"))
    out(f"  {'Slice':<28} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        for label, (h0, h1) in sessions:
            sub = tagged[(tagged["sig_type"] == sig) & (tagged["hour"] >= h0) & (tagged["hour"] < h1)]
            if len(sub) == 0:
                continue
            out(stats(sub, f"{sig} {label[:9]}"))

    # ── Best combination candidates ───────────────────────────────────────────
    out(section("Combination Candidates  (additive filter exploration)"))
    out(f"  {'Filter':<36} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")

    combos = [
        ("R001 only",
         tagged["sig_type"] == "R001"),
        ("R002 only",
         tagged["sig_type"] == "R002"),
        ("R001 + London/Lon+NY session",
         (tagged["sig_type"] == "R001") & (tagged["hour"] >= 7) & (tagged["hour"] < 22)),
        ("R002 + London/Lon+NY session",
         (tagged["sig_type"] == "R002") & (tagged["hour"] >= 7) & (tagged["hour"] < 22)),
        ("R001 + WITH ST",
         (tagged["sig_type"] == "R001") & tagged["aligned"]),
        ("R001 + AGAINST ST",
         (tagged["sig_type"] == "R001") & ~tagged["aligned"]),
        ("R002 + WITH ST",
         (tagged["sig_type"] == "R002") & tagged["aligned"]),
        ("R002 + AGAINST ST",
         (tagged["sig_type"] == "R002") & ~tagged["aligned"]),
        ("R001 + bA>=30",
         (tagged["sig_type"] == "R001") & (tagged["adx_b"] >= 30)),
        ("R002 + bA>=30",
         (tagged["sig_type"] == "R002") & (tagged["adx_b"] >= 30)),
        ("R002 + B<0.3 (oversold range)",
         (tagged["sig_type"] == "R002") & (tagged["bb"] < 0.3)),
        ("R002 + B>0.7 (overbought range)",
         (tagged["sig_type"] == "R002") & (tagged["bb"] > 0.7)),
        ("R001 + M:+ (momentum aligned)",
         (tagged["sig_type"] == "R001") & tagged["macd_bull"]),
        ("R002 + V>1.5 (volume spike)",
         (tagged["sig_type"] == "R002") & (tagged["vol"] > 1.5)),
    ]
    for label, mask in combos:
        sub = tagged[mask]
        if len(sub) == 0:
            out(f"  {label:<36}  — no trades")
            continue
        out(stats(sub, label))

    out("")
    result = "\n".join(lines)
    print(result)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[Saved -> {OUT_PATH}]")


if __name__ == "__main__":
    main()
