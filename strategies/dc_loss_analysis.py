"""Analyze loss profile of Donchian Mean Reversion TV results."""
import pandas as pd

tv = pd.read_csv("tvresults/Donchian_Mean_Reversion_v1_OANDA_EURUSD_2026-03-03.csv")
exits = tv[tv["Type"].str.startswith("Exit")].copy()
entries = tv[tv["Type"].str.startswith("Entry")].copy()

losers = exits[exits["Net P&L USD"] < 0].copy()
winners = exits[exits["Net P&L USD"] > 0].copy()

print("=== LOSS DISTRIBUTION ===")
print(f"Avg Win:  ${winners['Net P&L USD'].mean():+.2f}")
print(f"Avg Loss: ${losers['Net P&L USD'].mean():+.2f}")
print(f"Median Loss: ${losers['Net P&L USD'].median():+.2f}")
print()
print("Losses by size:")
ranges = [(0, -0.5, "tiny < $0.50"), (-0.5, -1.0, "small $0.50-1"),
          (-1.0, -2.0, "med $1-2"), (-2.0, -4.0, "large $2-4"), (-4.0, -99, "huge > $4")]
for hi, lo, label in ranges:
    mask = (losers["Net P&L USD"] <= hi) & (losers["Net P&L USD"] > lo)
    count = mask.sum()
    total = losers.loc[mask, "Net P&L USD"].sum()
    print(f"  {label:20s}: {count:>2} trades, total ${total:+.2f}")

print()
print("=== WORST 10 LOSERS ===")
worst = losers.nsmallest(10, "Net P&L USD")
for _, row in worst.iterrows():
    tn = row["Trade #"]
    entry_row = entries[entries["Trade #"] == tn].iloc[0]
    et = entry_row["Date and time"]
    xt = row["Date and time"]
    adv = row["Adverse excursion USD"]
    fav = row["Favorable excursion USD"]
    pnl = row["Net P&L USD"]
    print(f"  #{tn:>2}: {pnl:+.2f}  MFE=+{fav:.2f}  MAE=-{adv:.2f}  {et} -> {xt}")

print()
print("=== MFE / MAE ANALYSIS ===")
print(f"Winners avg MFE: ${winners['Favorable excursion USD'].mean():.2f}  avg MAE: ${winners['Adverse excursion USD'].mean():.2f}")
print(f"Losers  avg MFE: ${losers['Favorable excursion USD'].mean():.2f}  avg MAE: ${losers['Adverse excursion USD'].mean():.2f}")
print()

# How many losers would have been cut with various SL levels
print("=== WHAT-IF: STOP LOSS SCENARIOS ===")
for sl_usd in [0.5, 1.0, 1.5, 2.0, 3.0]:
    cut = losers[losers["Adverse excursion USD"] > sl_usd]
    saved = cut["Net P&L USD"].sum() + len(cut) * (-sl_usd)  # would lose SL amt instead
    stopped_wins = winners[winners["Adverse excursion USD"] > sl_usd]
    lost_profit = stopped_wins["Net P&L USD"].sum()
    print(f"  SL=${sl_usd:.1f}: Would cut {len(cut):>2}/{len(losers)} losers, save ~${-saved:+.2f} | BUT stops {len(stopped_wins)} winners (lose ${lost_profit:.2f} profit)")

# Max bars in trade (rough proxy from entry/exit times)
print()
print("=== TRADE DURATION ===")
for idx, (_, erow) in enumerate(entries.iterrows()):
    tn = erow["Trade #"]
    xrow = exits[exits["Trade #"] == tn].iloc[0]
    entry_t = pd.Timestamp(erow["Date and time"])
    exit_t = pd.Timestamp(xrow["Date and time"])
    dur = (exit_t - entry_t).total_seconds() / 60  # minutes
    exits.loc[exits["Trade #"] == tn, "duration_min"] = dur

losers_dur = exits[(exits["Net P&L USD"] < 0)]
winners_dur = exits[(exits["Net P&L USD"] > 0)]
print(f"Winners avg duration: {winners_dur['duration_min'].mean():.0f} min ({winners_dur['duration_min'].mean()/5:.0f} bars)")
print(f"Losers  avg duration: {losers_dur['duration_min'].mean():.0f} min ({losers_dur['duration_min'].mean()/5:.0f} bars)")
print(f"Losers  median dur:   {losers_dur['duration_min'].median():.0f} min ({losers_dur['duration_min'].median()/5:.0f} bars)")
print()

# What if we cap hold time?
print("=== WHAT-IF: MAX HOLD TIME ===")
for max_min in [60, 120, 240, 480]:
    long_losers = losers_dur[losers_dur["duration_min"] > max_min]
    long_winners = winners_dur[winners_dur["duration_min"] > max_min]
    print(f"  MaxHold={max_min}min ({max_min//5} bars): {len(long_losers)} losers + {len(long_winners)} winners exceed limit")
