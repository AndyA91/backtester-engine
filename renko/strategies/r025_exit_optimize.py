"""
R025: Exit Optimization on R001 Winners

Takes the best R001 entry (n_bricks=2, cooldown=30 — IS PF 15.7) and sweeps
alternative exit methods to improve PF and win rate:

Exit modes:
  0 = First opposing brick (R001 baseline)
  1 = N-brick trailing: stay until N opposing bricks from peak
  2 = Min-hold: ignore exits for first N bricks, then first opposing
  3 = Min-hold + trailing combined
  4 = Supertrend exit: exit when Supertrend flips against position
  5 = KAMA slope exit: exit when KAMA slope reverses

The R009 exit study showed 0-1h trades are 100% losers on R008.
Applying the same exit improvements to R001 (the strongest entry signal)
should yield even bigger gains since R001 has more trades to filter.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "R001 best entry + swept exit methods (trail, min-hold, ST, KAMA)"

HYPOTHESIS = (
    "R001 n=2 cd=30 is PF 15.7 with first-opposing exit. R009 proved that "
    "trailing and min-hold exits improve PF on R008. Applying to R001 should "
    "improve further — R001 has 560 trades (more data) and the entry signal "
    "is stronger. Adding Supertrend and KAMA exits tests whether trend-following "
    "exits outperform brick-counting exits."
)

PARAM_GRID = {
    "n_bricks":        [2, 3],
    "cooldown":        [20, 30],
    "exit_mode":       [0, 1, 2, 3, 4, 5],
    "trail_n":         [2, 3],
    "min_hold_bricks": [3, 5, 8],
}
# 2 × 2 × 6 × 2 × 3 = 144 combinations
# exit_mode 0: ignores trail_n and min_hold_bricks
# exit_mode 1: uses trail_n only
# exit_mode 2: uses min_hold_bricks only
# exit_mode 3: uses both
# exit_mode 4: ignores trail_n and min_hold_bricks (uses Supertrend)
# exit_mode 5: ignores trail_n and min_hold_bricks (uses KAMA slope)


def generate_signals(
    df: pd.DataFrame,
    n_bricks: int = 2,
    cooldown: int = 30,
    exit_mode: int = 0,
    trail_n: int = 2,
    min_hold_bricks: int = 5,
) -> pd.DataFrame:
    n = len(df)
    brick_up = df["brick_up"].values
    closes = df["Close"].values
    st_dir = df["st_dir"].values
    kama_slope = df["kama_slope"].values

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    in_position = False
    trade_dir = 0
    last_trade_bar = -999_999
    entry_bar = -1
    peak_close = 0.0
    trail_count = 0
    warmup = max(n_bricks + 1, 30)

    for i in range(warmup, n):
        up = bool(brick_up[i])

        # ── Exit logic ─────────────────────────────────────────────────
        if in_position:
            bricks_held = i - entry_bar

            if exit_mode == 0:
                do_exit = (trade_dir == 1 and not up) or (trade_dir == -1 and up)

            elif exit_mode == 1:
                # N-brick trailing
                if trade_dir == 1:
                    if closes[i] > peak_close:
                        peak_close = closes[i]
                        trail_count = 0
                    if not up:
                        trail_count += 1
                    else:
                        trail_count = 0
                    do_exit = trail_count >= trail_n
                else:
                    if closes[i] < peak_close:
                        peak_close = closes[i]
                        trail_count = 0
                    if up:
                        trail_count += 1
                    else:
                        trail_count = 0
                    do_exit = trail_count >= trail_n

            elif exit_mode == 2:
                # Min-hold then first opposing
                if bricks_held < min_hold_bricks:
                    do_exit = False
                else:
                    do_exit = (trade_dir == 1 and not up) or (trade_dir == -1 and up)

            elif exit_mode == 3:
                # Min-hold then trailing
                if bricks_held < min_hold_bricks:
                    do_exit = False
                else:
                    if trade_dir == 1:
                        if closes[i] > peak_close:
                            peak_close = closes[i]
                            trail_count = 0
                        if not up:
                            trail_count += 1
                        else:
                            trail_count = 0
                        do_exit = trail_count >= trail_n
                    else:
                        if closes[i] < peak_close:
                            peak_close = closes[i]
                            trail_count = 0
                        if up:
                            trail_count += 1
                        else:
                            trail_count = 0
                        do_exit = trail_count >= trail_n

            elif exit_mode == 4:
                # Supertrend exit: exit when ST flips against position
                sd = st_dir[i]
                if np.isnan(sd):
                    do_exit = False
                else:
                    do_exit = (trade_dir == 1 and sd < 0) or (trade_dir == -1 and sd > 0)

            else:
                # exit_mode 5: KAMA slope exit
                ks = kama_slope[i]
                if np.isnan(ks):
                    do_exit = False
                else:
                    do_exit = (trade_dir == 1 and ks < 0) or (trade_dir == -1 and ks > 0)

            if do_exit:
                long_exit[i] = trade_dir == 1
                short_exit[i] = trade_dir == -1
                in_position = False
                trade_dir = 0

        if in_position:
            continue

        if (i - last_trade_bar) < cooldown:
            continue

        # ── R001 entry: N consecutive same-direction bricks ─────────────
        window = brick_up[i - n_bricks + 1: i + 1]
        all_up = bool(np.all(window))
        all_down = bool(not np.any(window))

        if all_up:
            long_entry[i] = True
            in_position = True
            trade_dir = 1
            last_trade_bar = i
            entry_bar = i
            peak_close = closes[i]
            trail_count = 0

        elif all_down:
            short_entry[i] = True
            in_position = True
            trade_dir = -1
            last_trade_bar = i
            entry_bar = i
            peak_close = closes[i]
            trail_count = 0

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
