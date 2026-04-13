"""
MYM Dual Smoothed Heikin Ashi v2 — Stage 2: Gates + Exits

Builds on v1 dual SHA with:
- Slow streak filter: slow SHA must hold color for N bars before entry
- Session filter: RTH only, or skip first N minutes of session
- ATR-based stop loss / profit target (alternative exit to fast SHA flip)
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Dual SHA v2 with slow streak, session filter, ATR stop/target"

HYPOTHESIS = (
    "Stage 1 showed 30-34% WR with 2.1x W/L — marginal edge. "
    "Slow streak avoids transitions. Session filter removes overnight chop. "
    "ATR exits let winners run further and cut losers sooner than SHA flip."
)


def _ema(src: np.ndarray, length: int) -> np.ndarray:
    """EMA matching Pine's ta.ema()."""
    out = np.empty_like(src, dtype=float)
    out[0] = src[0]
    k = 2.0 / (length + 1)
    for i in range(1, len(src)):
        out[i] = src[i] * k + out[i - 1] * (1 - k)
    return out


def _smoothed_ha(o, h, l, c, length):
    """Compute smoothed HA. Returns (ha_open, ha_close, is_bull)."""
    s_o = _ema(o, length)
    s_h = _ema(h, length)
    s_l = _ema(l, length)
    s_c = _ema(c, length)

    n = len(o)
    ha_open = np.empty(n, dtype=float)
    ha_close = (s_o + s_h + s_l + s_c) / 4.0

    ha_open[0] = (s_o[0] + s_c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    is_bull = ha_close >= ha_open
    return ha_open, ha_close, is_bull


def _atr(h, l, c, length):
    """ATR calculation."""
    n = len(h)
    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return _ema(tr, length)


def _slow_streak(is_bull, i):
    """Count consecutive bars of same color ending at bar i."""
    streak = 0
    color = is_bull[i]
    j = i
    while j >= 0 and is_bull[j] == color:
        streak += 1
        j -= 1
    return streak


def generate_signals(
    df: pd.DataFrame,
    fast_len: int = 3,
    slow_len: int = 14,
    cooldown: int = 30,
    min_slow_streak: int = 0,
    session_mode: str = "none",
    exit_mode: str = "sha_flip",
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
    atr_len: int = 14,
) -> pd.DataFrame:
    """
    Generate dual SHA v2 signals.

    Args:
        df: DataFrame with Open, High, Low, Close + datetime index.
        fast_len: Fast SHA smoothing length.
        slow_len: Slow SHA smoothing length.
        cooldown: Min bars between exit and next entry.
        min_slow_streak: Slow SHA must hold color for N bars before entry. 0=disabled.
        session_mode: "none" = 24h, "rth" = 09:30-15:45 ET only,
                      "rth_skip15" = RTH but skip first 15min (09:45-15:45).
        exit_mode: "sha_flip" = fast SHA flip (v1 behavior),
                   "atr_only" = ATR stop/target only (no SHA exit),
                   "sha_or_atr" = whichever triggers first.
        atr_sl_mult: Stop loss = entry +/- atr_sl_mult * ATR. 0=disabled.
        atr_tp_mult: Take profit = entry +/- atr_tp_mult * ATR. 0=disabled.
        atr_len: ATR lookback period.
    """
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    n = len(df)

    _, _, f_bull = _smoothed_ha(o, h, l, c, fast_len)
    _, _, s_bull = _smoothed_ha(o, h, l, c, slow_len)
    atr_vals = _atr(h, l, c, atr_len)

    # Pre-compute slow streak counts
    slow_streaks = np.zeros(n, dtype=int)
    count = 1
    for i in range(1, n):
        if s_bull[i] == s_bull[i - 1]:
            count += 1
        else:
            count = 1
        slow_streaks[i] = count

    # Session filter: compute hour/minute from index
    use_session = session_mode != "none"
    in_session = np.ones(n, dtype=bool)
    if use_session:
        # Convert to ET (UTC-4 for EDT, approximate)
        idx = df.index
        et_hour = np.array([(t.hour - 4) % 24 for t in idx])
        et_min = np.array([t.minute for t in idx])
        hhmm = et_hour * 100 + et_min

        if session_mode == "rth":
            in_session = (hhmm >= 930) & (hhmm < 1545)
        elif session_mode == "rth_skip15":
            in_session = (hhmm >= 945) & (hhmm < 1545)
        elif session_mode == "eth":
            # Extended: 08:00-16:00 ET
            in_session = (hhmm >= 800) & (hhmm < 1600)

    use_atr_exit = exit_mode in ("atr_only", "sha_or_atr")
    use_sha_exit = exit_mode in ("sha_flip", "sha_or_atr")

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)

    warmup = max(fast_len, slow_len, atr_len) + 2
    last_exit_bar = -999_999
    pos = 0  # +1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0

    for i in range(warmup, n):
        fast_bull_flip = f_bull[i] and not f_bull[i - 1]
        fast_bear_flip = not f_bull[i] and f_bull[i - 1]

        # ── Exits ──
        if pos == 1:
            sha_exit = use_sha_exit and fast_bear_flip
            atr_exit = False
            if use_atr_exit and entry_atr > 0:
                sl = entry_price - atr_sl_mult * entry_atr if atr_sl_mult > 0 else -1e18
                tp = entry_price + atr_tp_mult * entry_atr if atr_tp_mult > 0 else 1e18
                atr_exit = (l[i] <= sl) or (h[i] >= tp)

            if sha_exit or atr_exit:
                long_exit[i] = True
                pos = 0
                last_exit_bar = i

        elif pos == -1:
            sha_exit = use_sha_exit and fast_bull_flip
            atr_exit = False
            if use_atr_exit and entry_atr > 0:
                sl = entry_price + atr_sl_mult * entry_atr if atr_sl_mult > 0 else 1e18
                tp = entry_price - atr_tp_mult * entry_atr if atr_tp_mult > 0 else -1e18
                atr_exit = (h[i] >= sl) or (l[i] <= tp)

            if sha_exit or atr_exit:
                short_exit[i] = True
                pos = 0
                last_exit_bar = i

        # ── Session close: force exit at end of session ──
        if use_session and pos != 0 and not in_session[i]:
            if pos == 1:
                long_exit[i] = True
            elif pos == -1:
                short_exit[i] = True
            pos = 0
            last_exit_bar = i

        # ── Entries ──
        if pos == 0 and (i - last_exit_bar) >= cooldown and in_session[i]:
            streak_ok = (min_slow_streak == 0) or (slow_streaks[i] >= min_slow_streak)

            if streak_ok and fast_bull_flip and s_bull[i]:
                long_entry[i] = True
                pos = 1
                entry_price = c[i]  # approximate — engine fills at next bar open
                entry_atr = atr_vals[i]
            elif streak_ok and fast_bear_flip and not s_bull[i]:
                short_entry[i] = True
                pos = -1
                entry_price = c[i]
                entry_atr = atr_vals[i]

    df["long_entry"] = long_entry
    df["long_exit"] = long_exit
    df["short_entry"] = short_entry
    df["short_exit"] = short_exit
    return df
