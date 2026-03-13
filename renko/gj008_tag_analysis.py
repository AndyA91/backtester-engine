"""
GJ008 Tag Analysis — entry AND exit indicator snapshots from TV live-mode export.

Tag format (14 indicators):
  Entry: [R001L 5A:31 bA:28 R:58 V:1.15 B:0.74 M:+ ST:U CH:42 CF:+ MI:55 SK:67 PS:U SQ:0]
  Exit:  [ExitL 5A:31 bA:28 R:58 V:1.15 B:0.74 M:+ ST:U CH:42 CF:+ MI:55 SK:67 PS:U SQ:0]

Run from repo root:
  python renko/gj008_tag_analysis.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
TV_CSV   = ROOT / "tvresults" / "GJ008_GJ007_+_5m_Candle_ADX_Gate_[Renko_GBPJPY]_OANDA_GBPJPY_2026-03-12 (1).csv"
OUT_PATH = ROOT / "tvresults" / "gj008_tag_analysis.txt"

TAG_RE = re.compile(
    r"\[(?P<sig>\w+)"
    r"\s+5A:(?P<adx5>[\d.]+)"
    r"\s+bA:(?P<adx_b>[\d.]+)"
    r"\s+R:(?P<rsi>[\d.]+)"
    r"\s+V:(?P<vol>[\d.]+)"
    r"\s+B:(?P<bb>[\d.]+)"
    r"\s+M:(?P<macd>[+\-])"
    r"\s+ST:(?P<st>[UD])"
    r"\s+CH:(?P<chop>[\d.]+)"
    r"\s+CF:(?P<cmf>[+\-])"
    r"\s+MI:(?P<mfi>[\d.]+)"
    r"\s+SK:(?P<stoch>[\d.]+)"
    r"\s+PS:(?P<psar>[UD])"
    r"\s+SQ:(?P<squeeze>[01])"
    r"\]"
)


def calc_pf(sub: pd.DataFrame) -> float:
    g = sub[sub["pnl"] > 0]["pnl"].sum()
    l = abs(sub[sub["pnl"] < 0]["pnl"].sum())
    return g / l if l > 0 else float("inf")


def stats(sub: pd.DataFrame, label: str) -> str:
    n   = len(sub)
    wr  = sub["is_winner"].mean() * 100 if n > 0 else 0
    net = sub["pnl"].sum()
    pf  = calc_pf(sub)
    return f"  {label:<32} {n:>5}t  Net ${net:>8.2f}  WR {wr:>4.1f}%  PF {pf:>6.2f}"


def section(title: str) -> str:
    return f"\n{'='*68}\n  {title}\n{'='*68}"


def parse_tag(series: pd.Series) -> pd.DataFrame:
    parsed = series.str.extract(TAG_RE)
    for col in ["adx5", "adx_b", "rsi", "vol", "bb", "chop", "mfi", "stoch"]:
        parsed[col] = pd.to_numeric(parsed[col], errors="coerce")
    parsed["squeeze"] = parsed["squeeze"].map({"0": False, "1": True}).fillna(False).astype(bool)
    return parsed


def load_trades(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]

    entry = raw[raw["Type"].str.contains("Entry", case=False)].copy()
    exit_ = raw[raw["Type"].str.contains("Exit",  case=False)].copy()

    entry = entry.rename(columns={
        "Date and time": "entry_time",
        "Signal":        "entry_sig",
    })
    exit_ = exit_.rename(columns={
        "Date and time":           "exit_time",
        "Net P&L USD":             "pnl",
        "Favorable excursion USD": "mfe",
        "Adverse excursion USD":   "mae",
        "Signal":                  "exit_sig",
    })

    trades = entry[["Trade #", "entry_time", "entry_sig"]].merge(
        exit_[["Trade #", "exit_time", "pnl", "mfe", "mae", "exit_sig"]],
        on="Trade #",
    )
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
    trades["duration_h"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
    trades["is_winner"]  = trades["pnl"] > 0
    trades["hour"]       = trades["entry_time"].dt.hour
    trades["dow"]        = trades["entry_time"].dt.day_name()

    # Parse entry tag
    ep = parse_tag(trades["entry_sig"])
    ep.columns = ["e_" + c for c in ep.columns]
    ep["e_sig_type"] = ep["e_sig"].str[:4]   # R001 / R002
    ep["e_sig_dir"]  = ep["e_sig"].str[4]    # L / S
    ep["e_is_long"]  = ep["e_sig_dir"] == "L"
    ep["e_st_up"]    = ep["e_st"] == "U"
    ep["e_aligned"]  = (ep["e_is_long"] & ep["e_st_up"]) | (~ep["e_is_long"] & ~ep["e_st_up"])
    ep["e_psar_u"]   = ep["e_psar"] == "U"
    ep["e_psar_aligned"] = (ep["e_is_long"] & ep["e_psar_u"]) | (~ep["e_is_long"] & ~ep["e_psar_u"])
    ep["e_macd_bull"]= ep["e_macd"] == "+"
    ep["e_cmf_pos"]  = ep["e_cmf"] == "+"

    # Parse exit tag
    xp = parse_tag(trades["exit_sig"])
    xp.columns = ["x_" + c for c in xp.columns]
    xp["x_st_up"]   = xp["x_st"] == "U"
    xp["x_psar_u"]  = xp["x_psar"] == "U"

    trades = pd.concat([trades, ep, xp], axis=1)
    return trades


def main():
    print(f"Loading {TV_CSV.name}...")
    t = load_trades(TV_CSV)
    df = t.dropna(subset=["e_sig"]).copy()
    print(f"  Total trades: {len(t)}  Tagged: {len(df)}  Untagged: {len(t) - len(df)}")
    print(f"  Period: {df['entry_time'].min().date()} to {df['exit_time'].max().date()}")
    print(f"  Winners: {df['is_winner'].sum()}  Losers: {(~df['is_winner']).sum()}")
    print(f"  Baseline PF: {calc_pf(df):.3f}  WR: {df['is_winner'].mean()*100:.1f}%  Net: ${df['pnl'].sum():.2f}")

    lines = []
    out = lines.append

    out(f"GJ008 Tag Analysis  ({len(df)} tagged trades, live mode — no date filter)")
    out(f"Period: {df['entry_time'].min().date()} to {df['exit_time'].max().date()}")
    out(f"Baseline  PF {calc_pf(df):.3f}  WR {df['is_winner'].mean()*100:.1f}%  Net ${df['pnl'].sum():.2f}")
    out(f"Duration: winners {df[df['is_winner']]['duration_h'].mean():.1f}h avg  |  losers {df[~df['is_winner']]['duration_h'].mean():.1f}h avg")

    # ── Signal type breakdown ──────────────────────────────────────────────────
    out(section("ENTRY — Signal Type Breakdown"))
    out(f"  {'Signal':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        for d, label in [("L", "Long"), ("S", "Short")]:
            sub = df[(df["e_sig_type"] == sig) & (df["e_sig_dir"] == d)]
            if len(sub):
                out(stats(sub, f"{sig} {label}"))
    out("")
    for sig in ["R001", "R002"]:
        sub = df[df["e_sig_type"] == sig]
        out(stats(sub, f"{sig} total"))

    # ── Session ────────────────────────────────────────────────────────────────
    out(section("ENTRY — Session (entry hour UTC, live mode so all hours present)"))
    out(f"  {'Session':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    sessions = [
        ("Asian     (00–07)", 0,  7),
        ("London    (07–13)", 7,  13),
        ("Lon+NY    (13–22)", 13, 22),
        ("NY close  (22–24)", 22, 24),
    ]
    for label, h0, h1 in sessions:
        sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        if len(sub):
            out(stats(sub, label))
    out("")
    for sig in ["R001", "R002"]:
        for label, h0, h1 in sessions:
            sub = df[(df["e_sig_type"] == sig) & (df["hour"] >= h0) & (df["hour"] < h1)]
            if len(sub):
                out(stats(sub, f"{sig} {label[:9]}"))

    # ── Supertrend alignment ───────────────────────────────────────────────────
    out(section("ENTRY — Supertrend Alignment"))
    out(f"  {'Slice':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        sub_w = df[(df["e_sig_type"] == sig) &  df["e_aligned"]]
        sub_a = df[(df["e_sig_type"] == sig) & ~df["e_aligned"]]
        out(stats(sub_w, f"{sig} WITH ST"))
        out(stats(sub_a, f"{sig} AGAINST ST"))

    # ── PSAR alignment ─────────────────────────────────────────────────────────
    out(section("ENTRY — PSAR Alignment (entry WITH vs AGAINST PSAR)"))
    out(f"  {'Slice':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        sub_w = df[(df["e_sig_type"] == sig) &  df["e_psar_aligned"]]
        sub_a = df[(df["e_sig_type"] == sig) & ~df["e_psar_aligned"]]
        out(stats(sub_w, f"{sig} WITH PSAR"))
        out(stats(sub_a, f"{sig} AGAINST PSAR"))

    # ── CHOP ───────────────────────────────────────────────────────────────────
    out(section("ENTRY — CHOP Index (<38.2 trending | >61.8 choppy)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    chop_buckets = [
        ("<38.2 trending",  0,    38.2),
        ("38.2–50",         38.2, 50),
        ("50–61.8",         50,   61.8),
        (">61.8 choppy",    61.8, 200),
    ]
    for label, lo, hi in chop_buckets:
        sub = df[(df["e_chop"] >= lo) & (df["e_chop"] < hi)]
        if len(sub):
            out(stats(sub, f"CH {label}"))

    # ── MFI ────────────────────────────────────────────────────────────────────
    out(section("ENTRY — MFI(14) (vol-weighted RSI)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    mfi_buckets = [("<30", 0, 30), ("30–50", 30, 50), ("50–70", 50, 70), (">70", 70, 100)]
    for label, lo, hi in mfi_buckets:
        sub = df[(df["e_mfi"] >= lo) & (df["e_mfi"] < hi)]
        if len(sub):
            out(stats(sub, f"MI {label}"))

    # ── Stochastic ─────────────────────────────────────────────────────────────
    out(section("ENTRY — Stochastic %K(14,3,3)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    stoch_buckets = [("<20 oversold", 0, 20), ("20–40", 20, 40), ("40–60", 40, 60), ("60–80", 60, 80), (">80 overbought", 80, 101)]
    for label, lo, hi in stoch_buckets:
        sub = df[(df["e_stoch"] >= lo) & (df["e_stoch"] < hi)]
        if len(sub):
            out(stats(sub, f"SK {label}"))

    # ── Squeeze ────────────────────────────────────────────────────────────────
    out(section("ENTRY — Squeeze (BB inside KC = coiling)"))
    out(f"  {'Slice':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        sub_sq  = df[(df["e_sig_type"] == sig) &  df["e_squeeze"]]
        sub_nsq = df[(df["e_sig_type"] == sig) & ~df["e_squeeze"]]
        out(stats(sub_sq,  f"{sig} SQ=1 coiling"))
        out(stats(sub_nsq, f"{sig} SQ=0 free"))

    # ── BB %B ──────────────────────────────────────────────────────────────────
    out(section("ENTRY — BB %B(20,2)  (0=lower, 0.5=mid, 1=upper)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    bb_b = [(0, 0.2, "<0.2 near lower"), (0.2, 0.4, "0.2–0.4"),
            (0.4, 0.6, "0.4–0.6 mid"), (0.6, 0.8, "0.6–0.8"),
            (0.8, 1.1, ">0.8 near upper")]
    for lo, hi, label in bb_b:
        sub = df[(df["e_bb"] >= lo) & (df["e_bb"] < hi)]
        if len(sub):
            out(stats(sub, f"B {label}"))

    # ── CMF ────────────────────────────────────────────────────────────────────
    out(section("ENTRY — CMF(20) Sign (buying vs selling pressure)"))
    out(f"  {'Slice':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        sub_pos = df[(df["e_sig_type"] == sig) &  df["e_cmf_pos"]]
        sub_neg = df[(df["e_sig_type"] == sig) & ~df["e_cmf_pos"]]
        out(stats(sub_pos, f"{sig} CF:+ buying"))
        out(stats(sub_neg, f"{sig} CF:- selling"))

    # ── MACD ───────────────────────────────────────────────────────────────────
    out(section("ENTRY — MACD Histogram Sign"))
    out(f"  {'Slice':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for sig in ["R001", "R002"]:
        sub_b = df[(df["e_sig_type"] == sig) &  df["e_macd_bull"]]
        sub_e = df[(df["e_sig_type"] == sig) & ~df["e_macd_bull"]]
        out(stats(sub_b, f"{sig} M:+ bull"))
        out(stats(sub_e, f"{sig} M:- bear"))

    # ── Volume ─────────────────────────────────────────────────────────────────
    out(section("ENTRY — Volume Ratio (brick vol / EMA20)"))
    out(f"  vol range: {df['e_vol'].min():.2f}–{df['e_vol'].max():.2f}  median {df['e_vol'].median():.2f}")
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    vol_b = [(0, 0.5, "<0.5 quiet"), (0.5, 1.0, "0.5–1.0"), (1.0, 1.5, "1.0–1.5"),
             (1.5, 2.5, "1.5–2.5"), (2.5, 999, ">2.5 spike")]
    for lo, hi, label in vol_b:
        sub = df[(df["e_vol"] >= lo) & (df["e_vol"] < hi)]
        if len(sub):
            out(stats(sub, f"V {label}"))

    # ── 5m ADX ─────────────────────────────────────────────────────────────────
    out(section("ENTRY — 5m ADX(14) (gate already >= 25)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    adx_b = [(25, 30), (30, 35), (35, 40), (40, 50), (50, 70), (70, 200)]
    for lo, hi in adx_b:
        sub = df[(df["e_adx5"] >= lo) & (df["e_adx5"] < hi)]
        if len(sub):
            out(stats(sub, f"5A [{lo}–{hi})"))

    # ── EXIT TAG ANALYSIS ──────────────────────────────────────────────────────
    out(section("EXIT — Supertrend at Exit (trade still WITH ST when we close?)"))
    out(f"  Note: for long trades, ST:U at exit = trend intact; ST:D = trend reversed")
    out(f"  {'Slice':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    # For long: exit ST:U = exiting with trend (trend intact); exit ST:D = trend reversed
    long_trades = df[df["e_is_long"]]
    short_trades = df[~df["e_is_long"]]
    out(stats(long_trades[long_trades["x_st_up"]],  "Long, exit ST:U (trend intact)"))
    out(stats(long_trades[~long_trades["x_st_up"]], "Long, exit ST:D (trend reversed)"))
    out(stats(short_trades[~short_trades["x_st_up"]], "Short, exit ST:D (trend intact)"))
    out(stats(short_trades[short_trades["x_st_up"]], "Short, exit ST:U (trend reversed)"))

    out(section("EXIT — PSAR at Exit"))
    out(f"  {'Slice':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    out(stats(long_trades[long_trades["x_psar_u"]],  "Long, exit PSAR:U (price above)"))
    out(stats(long_trades[~long_trades["x_psar_u"]], "Long, exit PSAR:D (price below)"))
    out(stats(short_trades[~short_trades["x_psar_u"]], "Short, exit PSAR:D (price below)"))
    out(stats(short_trades[short_trades["x_psar_u"]], "Short, exit PSAR:U (price above)"))

    out(section("EXIT — 5m ADX at Exit  (is HTF trend still running when we exit?)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for lo, hi in adx_b:
        sub = df[(df["x_adx5"] >= lo) & (df["x_adx5"] < hi)]
        if len(sub):
            out(stats(sub, f"Exit 5A [{lo}–{hi})"))

    out(section("EXIT — Volume Ratio at Exit (high vol at exit = reversal strength?)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    for lo, hi, label in vol_b:
        sub = df[(df["x_vol"] >= lo) & (df["x_vol"] < hi)]
        if len(sub):
            out(stats(sub, f"Exit V {label}"))

    out(section("EXIT — RSI at Exit  (exhaustion signal?)"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    rsi_b = [(0,30,"<30 oversold"), (30,40,"30–40"), (40,50,"40–50"),
             (50,60,"50–60"), (60,70,"60–70"), (70,100,">70 overbought")]
    for lo, hi, label in rsi_b:
        sub = df[(df["x_rsi"] >= lo) & (df["x_rsi"] < hi)]
        if len(sub):
            out(stats(sub, f"Exit R {label}"))

    # ── Duration breakdown ─────────────────────────────────────────────────────
    out(section("DURATION — Hold time distribution"))
    out(f"  {'Bucket':<32} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    dur_b = [(0, 1, "<1h"), (1, 4, "1–4h"), (4, 12, "4–12h"), (12, 24, "12–24h"), (24, 9999, ">24h")]
    for lo, hi, label in dur_b:
        sub = df[(df["duration_h"] >= lo) & (df["duration_h"] < hi)]
        if len(sub):
            out(stats(sub, f"Hold {label}"))

    # ── Combination candidates ─────────────────────────────────────────────────
    out(section("COMBINATION CANDIDATES  (high-PF slices for further gating research)"))
    out(f"  {'Filter':<40} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}")
    combos = [
        ("R001 only",
         df["e_sig_type"] == "R001"),
        ("R002 only",
         df["e_sig_type"] == "R002"),
        ("R001 + Lon+NY (13-22)",
         (df["e_sig_type"] == "R001") & (df["hour"] >= 13) & (df["hour"] < 22)),
        ("R002 + Lon+NY (13-22)",
         (df["e_sig_type"] == "R002") & (df["hour"] >= 13) & (df["hour"] < 22)),
        ("R001 + WITH ST",
         (df["e_sig_type"] == "R001") & df["e_aligned"]),
        ("R002 + WITH ST",
         (df["e_sig_type"] == "R002") & df["e_aligned"]),
        ("R001 + AGAINST ST",
         (df["e_sig_type"] == "R001") & ~df["e_aligned"]),
        ("R002 + AGAINST ST",
         (df["e_sig_type"] == "R002") & ~df["e_aligned"]),
        ("R001 + WITH PSAR",
         (df["e_sig_type"] == "R001") & df["e_psar_aligned"]),
        ("R002 + WITH PSAR",
         (df["e_sig_type"] == "R002") & df["e_psar_aligned"]),
        ("R001 + AGAINST PSAR",
         (df["e_sig_type"] == "R001") & ~df["e_psar_aligned"]),
        ("R002 + AGAINST PSAR",
         (df["e_sig_type"] == "R002") & ~df["e_psar_aligned"]),
        ("R001 + CH<38.2 trending",
         (df["e_sig_type"] == "R001") & (df["e_chop"] < 38.2)),
        ("R001 + CH>61.8 choppy",
         (df["e_sig_type"] == "R001") & (df["e_chop"] > 61.8)),
        ("R002 + CH<38.2 trending",
         (df["e_sig_type"] == "R002") & (df["e_chop"] < 38.2)),
        ("R002 + CH>61.8 choppy",
         (df["e_sig_type"] == "R002") & (df["e_chop"] > 61.8)),
        ("R001 + SQ=1 coiling",
         (df["e_sig_type"] == "R001") &  df["e_squeeze"]),
        ("R002 + SQ=1 coiling",
         (df["e_sig_type"] == "R002") &  df["e_squeeze"]),
        ("All + SQ=0 (free)",
         ~df["e_squeeze"]),
        ("R001 + SK<20 oversold entry",
         (df["e_sig_type"] == "R001") & (df["e_stoch"] < 20)),
        ("R001 + SK>80 overbought entry",
         (df["e_sig_type"] == "R001") & (df["e_stoch"] > 80)),
        ("R001 + V<0.5 quiet",
         (df["e_sig_type"] == "R001") & (df["e_vol"] < 0.5)),
        ("R002 + V<0.5 quiet",
         (df["e_sig_type"] == "R002") & (df["e_vol"] < 0.5)),
        ("R001 + WITH ST + Lon+NY",
         (df["e_sig_type"] == "R001") & df["e_aligned"] & (df["hour"] >= 13) & (df["hour"] < 22)),
        ("R002 + WITH ST + Lon+NY",
         (df["e_sig_type"] == "R002") & df["e_aligned"] & (df["hour"] >= 13) & (df["hour"] < 22)),
    ]
    for label, mask in combos:
        sub = df[mask]
        if len(sub) == 0:
            out(f"  {label:<40}  — no trades")
            continue
        out(f"  {label:<40} {len(sub):>5}t  Net ${sub['pnl'].sum():>8.2f}  WR {sub['is_winner'].mean()*100:>4.1f}%  PF {calc_pf(sub):>6.2f}")

    out("")
    result = "\n".join(lines)
    print(result)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[Saved -> {OUT_PATH}]")


if __name__ == "__main__":
    main()
