"""
R008 Exit Tag Analysis — what conditions exist when we close trades?

Exit tags have the same format as entry tags:
  [ExitL 5A:35 bA:35 R:33 V:1.86 B:0.17 M:- ST:D]

Key questions:
  1. Exit indicator profile: winners vs losers
  2. Early exit detection: high MFE / small PnL → left money on table?
  3. RSI / ADX at exit — trending-out vs mean-reverting?
  4. Vol ratio at exit — spike exits vs quiet exits

Run from repo root:
  python renko/r008_exit_analysis.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

TV_CSV   = ROOT / "tvresults" / "R008_R007_+_5m_Candle_ADX_Gate_[Renko]_OANDA_EURUSD_2026-03-07 (5).csv"
OUT_PATH = ROOT / "tvresults" / "r008_exit_analysis.txt"

_TAIL = (
    r"\s+CH:(?P<{p}ch>[\d.]+)"
    r"\s+CF:(?P<{p}cf>[+\-])"
    r"\s+MI:(?P<{p}mi>[\d.]+)"
    r"\s+SK:(?P<{p}sk>[\d.]+)"
    r"\s+PS:(?P<{p}ps>[UD])"
    r"\s+SQ:(?P<{p}sq>[01])"
    r"\]"
)

ENTRY_RE = re.compile(
    r"\[(?P<sig>R\d{3}[LS])"
    r"\s+5A:(?P<adx5>[\d.]+)"
    r"\s+bA:(?P<adx_b>[\d.]+)"
    r"\s+R:(?P<rsi>[\d.]+)"
    r"\s+V:(?P<vol>[\d.]+)"
    r"\s+B:(?P<bb>[-\d.]+)"
    r"\s+M:(?P<macd>[+\-])"
    r"\s+ST:(?P<st>[UD])"
    + _TAIL.format(p="e_")
)

EXIT_RE = re.compile(
    r"\[(?P<xsig>Exit[LS])"
    r"\s+5A:(?P<x5a>[\d.]+)"
    r"\s+bA:(?P<xba>[\d.]+)"
    r"\s+R:(?P<xrsi>[\d.]+)"
    r"\s+V:(?P<xvol>[\d.]+)"
    r"\s+B:(?P<xbb>[-\d.]+)"
    r"\s+M:(?P<xmacd>[+\-])"
    r"\s+ST:(?P<xst>[UD])"
    + _TAIL.format(p="x_")
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def calc_pf(sub: pd.DataFrame) -> float:
    g = sub[sub["pnl"] > 0]["pnl"].sum()
    l = abs(sub[sub["pnl"] < 0]["pnl"].sum())
    return g / l if l > 0 else float("inf")


def stats(sub: pd.DataFrame, label: str) -> str:
    n   = len(sub)
    if n == 0:
        return f"  {label:<34} — no trades"
    wr  = sub["is_winner"].mean() * 100
    net = sub["pnl"].sum()
    pf  = calc_pf(sub)
    avg_mfe = sub["mfe"].mean()
    avg_pnl = sub["pnl"].mean()
    capture = avg_pnl / avg_mfe * 100 if avg_mfe > 0 else 0
    return (f"  {label:<34} {n:>5}t  Net ${net:>8.2f}  WR {wr:>4.1f}%  PF {pf:>6.2f}"
            f"  MFE {avg_mfe:>5.2f}  capture {capture:>4.0f}%")


def section(title: str) -> str:
    return f"\n{'='*72}\n  {title}\n{'='*72}"


def median_str(series: pd.Series, label: str) -> str:
    return f"  {label}: median={series.median():.1f}  mean={series.mean():.1f}  p25={series.quantile(.25):.1f}  p75={series.quantile(.75):.1f}"


# ── Load ──────────────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]

    entry = raw[raw["Type"].str.contains("Entry", case=False)].copy()
    exit_ = raw[raw["Type"].str.contains("Exit",  case=False)].copy()

    entry = entry.rename(columns={"Date and time": "entry_time", "Signal": "entry_sig"})
    exit_ = exit_.rename(columns={
        "Date and time":           "exit_time",
        "Signal":                  "exit_sig",
        "Net P&L USD":             "pnl",
        "Favorable excursion USD": "mfe",
        "Adverse excursion USD":   "mae",
    })

    t = entry[["Trade #", "entry_time", "entry_sig"]].merge(
        exit_[["Trade #", "exit_time", "exit_sig", "pnl", "mfe", "mae"]],
        on="Trade #",
    )
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["exit_time"]  = pd.to_datetime(t["exit_time"])
    t["duration_h"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 3600
    t["is_winner"]  = t["pnl"] > 0
    t["hour"]       = t["entry_time"].dt.hour

    # Parse entry tag
    ep = t["entry_sig"].str.extract(ENTRY_RE)
    for col in ["adx5", "adx_b", "rsi", "vol", "bb", "e_ch", "e_mi", "e_sk"]:
        if col in ep.columns:
            ep[col] = pd.to_numeric(ep[col], errors="coerce")
    ep = ep.rename(columns=lambda c: "e_" + c if not c.startswith("e_") else c)
    t = pd.concat([t, ep], axis=1)

    # Parse exit tag
    xp = t["exit_sig"].str.extract(EXIT_RE)
    for col in ["x5a", "xba", "xrsi", "xvol", "xbb", "x_ch", "x_mi", "x_sk"]:
        if col in xp.columns:
            xp[col] = pd.to_numeric(xp[col], errors="coerce")
    t = pd.concat([t, xp], axis=1)

    t["sig_type"] = t["e_sig"].str[:4]    # R001 / R002
    t["is_long"]  = t["entry_sig"].str.contains("long", case=False)
    return t


def main():
    print(f"Loading {TV_CSV.name}...")
    t = load(TV_CSV)
    print(f"  Total trades: {len(t)}  date range: {t['entry_time'].min().date()} to {t['exit_time'].max().date()}")

    # Only keep trades with both entry AND exit tags parsed
    tagged = t.dropna(subset=["e_sig", "xsig"]).copy()
    print(f"  Fully tagged (entry+exit): {len(tagged)}")

    # Compute derived columns before slicing W/L
    tagged["st_still_aligned"] = (
        (tagged["is_long"] & (tagged["xst"] == "U")) |
        (~tagged["is_long"] & (tagged["xst"] == "D"))
    )
    tagged["rsi_delta"] = tagged["xrsi"] - tagged["e_rsi"]

    W = tagged[tagged["is_winner"]]
    L = tagged[~tagged["is_winner"]]

    lines = []
    out = lines.append

    out(f"R008 Exit Tag Analysis  ({len(tagged)} trades with exit tags)")
    out(f"Date: {tagged['entry_time'].min().date()} to {tagged['exit_time'].max().date()}")
    out(f"Winners: {len(W)}  Losers: {len(L)}  WR: {len(W)/len(tagged)*100:.1f}%")
    out(f"Overall PF: {calc_pf(tagged):.3f}  Net: ${tagged['pnl'].sum():.2f}")
    out(f"\n  Avg PnL W: ${W['pnl'].mean():.3f}  L: ${L['pnl'].mean():.3f}")
    out(f"  Avg MFE  W: ${W['mfe'].mean():.3f}  L: ${L['mfe'].mean():.3f}")
    out(f"  Avg MAE  W: ${W['mae'].mean():.3f}  L: ${L['mae'].mean():.3f}")
    out(f"  Avg dur  W: {W['duration_h'].mean():.1f}h  L: {L['duration_h'].mean():.1f}h")

    # ── MFE capture ───────────────────────────────────────────────────────────
    out(section("MFE Capture (avg PnL / avg MFE) — how much profit do we keep?"))
    out(f"  {'Slice':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for label, mask in [
        ("All trades",  slice(None)),
        ("Winners",     tagged["is_winner"]),
        ("Losers",      ~tagged["is_winner"]),
        ("R001",        tagged["sig_type"] == "R001"),
        ("R002",        tagged["sig_type"] == "R002"),
    ]:
        sub = tagged[mask] if not isinstance(mask, slice) else tagged
        out(stats(sub, label))

    # ── Exit RSI ──────────────────────────────────────────────────────────────
    out(section("RSI at Exit — trending further or reverting?"))
    out(f"  Winner  : {median_str(W['xrsi'], 'RSI')}")
    out(f"  Loser   : {median_str(L['xrsi'], 'RSI')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    rsi_buckets = [(0,30,"<30"), (30,40,"30–40"), (40,50,"40–50"),
                   (50,60,"50–60"), (60,70,"60–70"), (70,100,">70")]
    for lo, hi, label in rsi_buckets:
        sub = tagged[(tagged["xrsi"] >= lo) & (tagged["xrsi"] < hi)]
        out(stats(sub, f"exit RSI {label}"))

    # ── Exit ADX ──────────────────────────────────────────────────────────────
    out(section("5m ADX at Exit — exiting in trending vs flat market?"))
    out(f"  Winner  : {median_str(W['x5a'], '5mADX')}")
    out(f"  Loser   : {median_str(L['x5a'], 'Loser 5mADX')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi in [(0,20),(20,30),(30,40),(40,55),(55,100)]:
        sub = tagged[(tagged["x5a"] >= lo) & (tagged["x5a"] < hi)]
        out(stats(sub, f"exit 5mADX [{lo}–{hi})"))

    # ── Brick ADX at exit ─────────────────────────────────────────────────────
    out(section("Brick ADX at Exit — momentum still strong or fading?"))
    out(f"  Winner  : {median_str(W['xba'], 'brickADX')}")
    out(f"  Loser   : {median_str(L['xba'], 'Loser brickADX')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi in [(0,20),(20,30),(30,40),(40,55),(55,100)]:
        sub = tagged[(tagged["xba"] >= lo) & (tagged["xba"] < hi)]
        out(stats(sub, f"exit brickADX [{lo}–{hi})"))

    # ── Exit vol ──────────────────────────────────────────────────────────────
    out(section("Vol Ratio at Exit — quiet vs spike brick triggers exit?"))
    out(f"  Winner  : {median_str(W['xvol'], 'vol')}")
    out(f"  Loser   : {median_str(L['xvol'], 'Loser vol')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(0,0.5,"<0.5 quiet"),(0.5,1.0,"0.5–1.0"),
                          (1.0,1.5,"1.0–1.5"),(1.5,2.5,"1.5–2.5"),(2.5,99,">2.5 spike")]:
        sub = tagged[(tagged["xvol"] >= lo) & (tagged["xvol"] < hi)]
        out(stats(sub, f"exit vol {label}"))

    # ── Exit BB %B ────────────────────────────────────────────────────────────
    out(section("BB %B at Exit — price location when kicked out"))
    out(f"  Winner  : {median_str(W['xbb'], 'BB%B')}")
    out(f"  Loser   : {median_str(L['xbb'], 'Loser BB%B')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(-.5,0.2,"<0.2 near lower"),(0.2,0.4,"0.2–0.4"),
                          (0.4,0.6,"0.4–0.6 mid"),(0.6,0.8,"0.6–0.8"),
                          (0.8,1.5,">0.8 near upper")]:
        sub = tagged[(tagged["xbb"] >= lo) & (tagged["xbb"] < hi)]
        out(stats(sub, f"exit BB {label}"))

    # ── Exit Supertrend ───────────────────────────────────────────────────────
    out(section("Supertrend at Exit — ST flipped vs ST still going?"))
    out(f"\n  {'Slice':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    out(stats(tagged[tagged["st_still_aligned"]],  "ST still agrees at exit"))
    out(stats(tagged[~tagged["st_still_aligned"]], "ST flipped at exit"))
    out(stats(W[W["st_still_aligned"]],   "W: ST still agrees"))
    out(stats(W[~W["st_still_aligned"]],  "W: ST flipped"))
    out(stats(L[L["st_still_aligned"]],   "L: ST still agrees"))
    out(stats(L[~L["st_still_aligned"]],  "L: ST flipped"))

    # ── Entry→Exit RSI delta ──────────────────────────────────────────────────
    out(section("RSI Delta (exit RSI - entry RSI) — did momentum build or fade?"))
    out(f"  Winner rsi_delta: mean={W['rsi_delta'].mean():+.1f}  median={W['rsi_delta'].median():+.1f}")
    out(f"  Loser  rsi_delta: mean={L['rsi_delta'].mean():+.1f}  median={L['rsi_delta'].median():+.1f}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(-100,-20,"RSI fell >20"),(- 20,-5,"RSI fell 5–20"),
                          (-5,5,"RSI flat ±5"),(5,20,"RSI rose 5–20"),(20,100,"RSI rose >20")]:
        sub = tagged[(tagged["rsi_delta"] >= lo) & (tagged["rsi_delta"] < hi)]
        out(stats(sub, f"dRSI {label}"))

    # ── Duration vs exit RSI ──────────────────────────────────────────────────
    out(section("Duration Buckets — do long-held trades exit differently?"))
    out(f"  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(0,1,"<1h"),(1,4,"1–4h"),(4,12,"4–12h"),(12,48,"12–48h"),(48,9999,">48h")]:
        sub = tagged[(tagged["duration_h"] >= lo) & (tagged["duration_h"] < hi)]
        out(stats(sub, f"dur {label}"))

    # ── CHOP at exit and entry ─────────────────────────────────────────────────
    out(section("CHOP(14) at Exit — trending vs choppy regime at close?"))
    out(f"  Winner  : {median_str(W['x_ch'], 'CHOP')}")
    out(f"  Loser   : {median_str(L['x_ch'], 'Loser CHOP')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(0,38.2,"<38.2 trending"),(38.2,61.8,"38.2–61.8 transition"),(61.8,100,">61.8 choppy")]:
        sub = tagged[(tagged["x_ch"] >= lo) & (tagged["x_ch"] < hi)]
        out(stats(sub, f"exit CHOP {label}"))

    out(section("CHOP(14) at Entry — regime we entered into"))
    out(f"  Winner  : {median_str(W['e_ch'], 'CHOP')}")
    out(f"  Loser   : {median_str(L['e_ch'], 'Loser CHOP')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(0,38.2,"<38.2 trending"),(38.2,61.8,"38.2–61.8 transition"),(61.8,100,">61.8 choppy")]:
        sub = tagged[(tagged["e_ch"] >= lo) & (tagged["e_ch"] < hi)]
        out(stats(sub, f"entry CHOP {label}"))

    # ── CMF sign ──────────────────────────────────────────────────────────────
    out(section("CMF(20) Sign — Chaikin money flow direction"))
    out(f"\n  {'Slice':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    out(stats(tagged[tagged["x_cf"] == "+"], "exit CMF positive"))
    out(stats(tagged[tagged["x_cf"] == "-"], "exit CMF negative"))
    out(stats(tagged[tagged["e_cf"] == "+"], "entry CMF positive"))
    out(stats(tagged[tagged["e_cf"] == "-"], "entry CMF negative"))
    cmf_entry_aligned = (
        (tagged["is_long"]  & (tagged["e_cf"] == "+")) |
        (~tagged["is_long"] & (tagged["e_cf"] == "-"))
    )
    out(stats(tagged[cmf_entry_aligned],  "entry CMF aligned with trade"))
    out(stats(tagged[~cmf_entry_aligned], "entry CMF opposing trade"))

    # ── MFI at exit ───────────────────────────────────────────────────────────
    out(section("MFI(14) at Exit — vol-weighted RSI levels"))
    out(f"  Winner  : {median_str(W['x_mi'], 'MFI')}")
    out(f"  Loser   : {median_str(L['x_mi'], 'Loser MFI')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(0,20,"<20 oversold"),(20,40,"20–40"),(40,60,"40–60"),(60,80,"60–80"),(80,101,">80 overbought")]:
        sub = tagged[(tagged["x_mi"] >= lo) & (tagged["x_mi"] < hi)]
        out(stats(sub, f"exit MFI {label}"))

    # ── Stochastic %K at exit ─────────────────────────────────────────────────
    out(section("Stochastic %K at Exit — momentum oscillator"))
    out(f"  Winner  : {median_str(W['x_sk'], 'Stoch%K')}")
    out(f"  Loser   : {median_str(L['x_sk'], 'Loser Stoch%K')}")
    out(f"\n  {'Bucket':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    for lo, hi, label in [(0,20,"<20 oversold"),(20,50,"20–50"),(50,80,"50–80"),(80,101,">80 overbought")]:
        sub = tagged[(tagged["x_sk"] >= lo) & (tagged["x_sk"] < hi)]
        out(stats(sub, f"exit Stoch {label}"))

    # ── PSAR direction ────────────────────────────────────────────────────────
    out(section("PSAR Direction — SAR aligned with or against trade"))
    out(f"\n  {'Slice':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    psar_entry_aligned = (
        (tagged["is_long"]  & (tagged["e_ps"] == "U")) |
        (~tagged["is_long"] & (tagged["e_ps"] == "D"))
    )
    out(stats(tagged[psar_entry_aligned],  "entry PSAR aligned with trade"))
    out(stats(tagged[~psar_entry_aligned], "entry PSAR opposing trade"))
    psar_exit_aligned = (
        (tagged["is_long"]  & (tagged["x_ps"] == "U")) |
        (~tagged["is_long"] & (tagged["x_ps"] == "D"))
    )
    out(stats(tagged[psar_exit_aligned],  "exit PSAR still aligned"))
    out(stats(tagged[~psar_exit_aligned], "exit PSAR flipped"))

    # ── Squeeze ───────────────────────────────────────────────────────────────
    out(section("Squeeze (BB inside KC) — coiling vs free"))
    tagged["sq_entry"] = tagged["e_sq"] == "1"
    tagged["sq_exit"]  = tagged["x_sq"] == "1"
    out(f"\n  {'Slice':<34} {'Trades':>5}   {'Net $':>8}  {'WR':>5}   {'PF':>6}  {'avgMFE':>7}  {'capture':>7}")
    out(stats(tagged[tagged["sq_entry"]],  "entry squeeze ON (coiling)"))
    out(stats(tagged[~tagged["sq_entry"]], "entry squeeze OFF (free)"))
    out(stats(tagged[tagged["sq_exit"]],   "exit squeeze ON"))
    out(stats(tagged[~tagged["sq_exit"]],  "exit squeeze OFF"))

    result = "\n".join(lines)
    print(result)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n[Saved -> {OUT_PATH}]")


if __name__ == "__main__":
    main()
