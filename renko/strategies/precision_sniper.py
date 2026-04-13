"""
Precision Sniper [Renko Port]

Faithful Renko-brick port of the "Precision Sniper [WillyAlgoTrader]"
TradingView indicator (v1.1.0). 10-component confluence score + EMA-cross
entry trigger + ATR-based SL with TP1/TP2/TP3 R-multiple ladder and
trail-to-BE/TP1/TP2 on each TP hit.

Designed for EURAUD 0.0006 brick research but works on any Renko export.

Fidelity vs the original Pine indicator
---------------------------------------
Translation table (8 exact / 2 proxied):

  # | Pine component                    | Brick-space mapping              | Status
  --+-----------------------------------+----------------------------------+-------
  1 | emaFast > emaSlow                 | ema_fast > ema_slow              | exact
  2 | close > emaTrend                  | close_prev > ema_trend           | exact
  3 | rsi > 50 and rsi < 75             | same                             | exact
  4 | macdHist > 0                      | macd_hist > 0                    | exact
  5 | macdLine > macdSig                | macd > macd_sig                  | exact
  6 | close > vwap          (1.0)       | close_prev > bb_mid (SMA20)      | proxy
  7 | volume > volSma * 1.2             | vol_ratio > 1.2                  | exact
  8 | adx > 20 and +di > -di            | same                             | exact
  9 | HTF emaFast > emaSlow (1.5)       | ema50 > ema200 same brick stream | proxy
 10 | close > emaFast       (0.5)       | close_prev > ema_fast            | exact

Entry trigger (EMA cross + momentum filter + RSI extreme guard + score
threshold + direction-flip lockout) is exact to the source.

SL/TP/trail ladder is implemented inline. Behavioural notes:
  - Single full-flatten on trail hit (the engine has no partial-fill support).
    A trail-to-BE-then-stopped-out trade closes the WHOLE position at the
    BE level — there is no "took 33% off at TP1, runner closed at BE"
    behaviour. The Pine indicator's tracker counts these as wins; ours
    realises them as approximately breakeven.
  - Bug-for-bug fidelity: the SL check on each bar uses the PRE-update
    trail (matches Pine). On a bar where the brick high hits TP1 and the
    brick low wicks back below the original SL, TP1 is logged AND the trade
    stops at the original SL — the trail-to-BE update only takes effect on
    the next bar.
  - SL/TP triggers are detected against brick high/low exactly. The realised
    fill happens at brick CLOSE of the trigger bar (engine convention), so
    there is some intra-brick slippage between trigger and fill.

Custom-length core indicators (ema_fast/slow/trend, RSI, ATR) are computed
inline so the preset lengths from the original indicator can be swept.
The HTF proxy (ema50/ema200) and VWAP proxy (bb_mid) come from the
add_renko_indicators baseline at fixed lengths.

State-gating caveat (Pitfall #2)
--------------------------------
This strategy mutates position state (in_pos, last_dir, trail_price) inside
generate_signals across the FULL dataframe — without a bar_in_range gate.
For visual exploration on a single instrument this is fine; the warmup
period before IS_START produces enough state churn that by the time we
reach IS_START the position state is well-aligned with what the engine
sees. For PRODUCTION sweeping with multiple start_date variants, add an
explicit start_date param and gate all state mutations on it.
"""

import numpy as np
import pandas as pd

DESCRIPTION = "Precision Sniper port: 10-component confluence + ATR SL/TP ladder"

HYPOTHESIS = (
    "Multi-factor trend confluence score + EMA-cross entry + R-multiple TP "
    "ladder with trail-to-BE on TP1 captures trend continuation while limiting "
    "downside via fixed-R stops. Confluence weighting heavily favours macro "
    "trend alignment (HTF proxy = ema50/ema200, weight 1.5 of 10), so signals "
    "fire mostly with the dominant brick trend. The hypothesis to falsify: "
    "does the 10-component gating add any uplift over a naive EMA-cross "
    "baseline on EURAUD 0.0006 bricks?"
)

# Default file for `python runner.py precision_sniper`
RENKO_FILE = "OANDA_EURAUD, 1S renko 0.0006.csv"

PARAM_GRID = {
    # Defaults match the Pine "Default" preset.
    "ema_fast_len":  [9],
    "ema_slow_len":  [21],
    "ema_trend_len": [55],
    "rsi_len":       [13],
    "atr_len":       [14],
    "min_score":     [5.0],
    "sl_atr_mult":   [1.5],
    "tp1_r":         [1.0],
    "tp2_r":         [2.0],
    "tp3_r":         [3.0],
    "use_trail":     [True],
}


# ── Inline indicator helpers ─────────────────────────────────────────────────

def _ema(arr, length):
    s = pd.Series(arr)
    return s.ewm(span=length, adjust=False, min_periods=length).mean().values


def _rsi(close, length):
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values


def _atr(high, low, close, length):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return atr.values


# ── Signal generator ─────────────────────────────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    ema_fast_len: int = 9,
    ema_slow_len: int = 21,
    ema_trend_len: int = 55,
    rsi_len: int = 13,
    atr_len: int = 14,
    min_score: float = 5.0,
    sl_atr_mult: float = 1.5,
    tp1_r: float = 1.0,
    tp2_r: float = 2.0,
    tp3_r: float = 3.0,
    use_trail: bool = True,
) -> pd.DataFrame:
    n = len(df)
    h_arr = df["High"].values
    l_arr = df["Low"].values
    c_arr = df["Close"].values

    # Custom-length core indicators — shifted by 1 for [i]-direct use.
    ema_fast  = pd.Series(_ema(c_arr, ema_fast_len)).shift(1).values
    ema_slow  = pd.Series(_ema(c_arr, ema_slow_len)).shift(1).values
    ema_trend = pd.Series(_ema(c_arr, ema_trend_len)).shift(1).values
    rsi_arr   = pd.Series(_rsi(c_arr, rsi_len)).shift(1).values
    atr_arr   = pd.Series(_atr(h_arr, l_arr, c_arr, atr_len)).shift(1).values

    # Pre-shifted indicators from add_renko_indicators (already shifted).
    macd      = df["macd"].values
    macd_sig  = df["macd_sig"].values
    macd_hist = df["macd_hist"].values
    adx       = df["adx"].values
    plus_di   = df["plus_di"].values
    minus_di  = df["minus_di"].values
    bb_mid    = df["bb_mid"].values        # VWAP proxy
    ema50     = df["ema50"].values         # HTF bias proxy (vs ema200)
    ema200    = df["ema200"].values
    vol_ratio = df["vol_ratio"].values

    # Pre-shifted brick close for confluence comparisons (= prior bar's close).
    close_prev = pd.Series(c_arr).shift(1).values

    long_entry  = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)

    # Position state for the SL/TP/trail ladder
    in_pos       = 0       # +1 long, -1 short, 0 flat
    entry_price  = 0.0
    sl_price     = 0.0
    tp1_price    = 0.0
    tp2_price    = 0.0
    tp3_price    = 0.0
    trail_price  = 0.0
    tp1_hit      = False
    tp2_hit      = False
    tp3_hit      = False
    last_dir     = 0       # direction-flip lockout (Pine: lastDirection)

    warmup = max(ema_trend_len, 200) + 5  # ema200 dominates the warmup

    for i in range(warmup, n):
        # NaN guard on every signal input
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(ema_trend[i]) or
            np.isnan(rsi_arr[i]) or np.isnan(atr_arr[i]) or
            np.isnan(macd[i]) or np.isnan(macd_sig[i]) or np.isnan(macd_hist[i]) or
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(bb_mid[i]) or np.isnan(ema50[i]) or np.isnan(ema200[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(close_prev[i])):
            continue

        # ── 1. TP detection + SL/trail check on any open position ─────────────
        # Bug-for-bug fidelity: SL check uses PRE-update trail (matches Pine).
        if in_pos == 1:
            bh = h_arr[i]
            bl = l_arr[i]
            pre_trail = trail_price

            if bh >= tp1_price and not tp1_hit:
                tp1_hit = True
                if use_trail:
                    trail_price = entry_price
            if bh >= tp2_price and not tp2_hit:
                tp2_hit = True
                if use_trail:
                    trail_price = tp1_price
            if bh >= tp3_price and not tp3_hit:
                tp3_hit = True
                if use_trail:
                    trail_price = tp2_price

            if bl <= pre_trail:
                long_exit[i] = True
                in_pos = 0
                tp1_hit = tp2_hit = tp3_hit = False

        elif in_pos == -1:
            bh = h_arr[i]
            bl = l_arr[i]
            pre_trail = trail_price

            if bl <= tp1_price and not tp1_hit:
                tp1_hit = True
                if use_trail:
                    trail_price = entry_price
            if bl <= tp2_price and not tp2_hit:
                tp2_hit = True
                if use_trail:
                    trail_price = tp1_price
            if bl <= tp3_price and not tp3_hit:
                tp3_hit = True
                if use_trail:
                    trail_price = tp2_price

            if bh >= pre_trail:
                short_exit[i] = True
                in_pos = 0
                tp1_hit = tp2_hit = tp3_hit = False

        # ── 2. Confluence scoring ─────────────────────────────────────────────
        cp = close_prev[i]
        bull = 0.0
        bull += 1.0 if ema_fast[i] > ema_slow[i] else 0.0
        bull += 1.0 if cp > ema_trend[i] else 0.0
        bull += 1.0 if (rsi_arr[i] > 50.0 and rsi_arr[i] < 75.0) else 0.0
        bull += 1.0 if macd_hist[i] > 0.0 else 0.0
        bull += 1.0 if macd[i] > macd_sig[i] else 0.0
        bull += 1.0 if cp > bb_mid[i] else 0.0                 # VWAP proxy
        bull += 1.0 if vol_ratio[i] > 1.2 else 0.0
        bull += 1.0 if (adx[i] > 20.0 and plus_di[i] > minus_di[i]) else 0.0
        bull += 1.5 if ema50[i] > ema200[i] else 0.0           # HTF proxy
        bull += 0.5 if cp > ema_fast[i] else 0.0

        bear = 0.0
        bear += 1.0 if ema_fast[i] < ema_slow[i] else 0.0
        bear += 1.0 if cp < ema_trend[i] else 0.0
        bear += 1.0 if (rsi_arr[i] < 50.0 and rsi_arr[i] > 25.0) else 0.0
        bear += 1.0 if macd_hist[i] < 0.0 else 0.0
        bear += 1.0 if macd[i] < macd_sig[i] else 0.0
        bear += 1.0 if cp < bb_mid[i] else 0.0
        bear += 1.0 if vol_ratio[i] > 1.2 else 0.0
        bear += 1.0 if (adx[i] > 20.0 and minus_di[i] > plus_di[i]) else 0.0
        bear += 1.5 if ema50[i] < ema200[i] else 0.0
        bear += 0.5 if cp < ema_fast[i] else 0.0

        # ── 3. Entry trigger ──────────────────────────────────────────────────
        # ema_fast[i] is shifted (= computed through bar i-1). A "fresh" cross
        # at bar i-1 in real space → ema_fast[i] > ema_slow[i] AND
        # ema_fast[i-1] <= ema_slow[i-1].
        bull_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        bear_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]

        bull_mom = cp > ema_fast[i] and cp > ema_slow[i]
        bear_mom = cp < ema_fast[i] and cp < ema_slow[i]

        rsi_not_ob = rsi_arr[i] < 75.0
        rsi_not_os = rsi_arr[i] > 25.0

        raw_buy  = bull_cross and bull_mom and rsi_not_ob and bull >= min_score
        raw_sell = bear_cross and bear_mom and rsi_not_os and bear >= min_score

        # Direction-flip lockout (Pine: lastDirection != 1 / -1)
        buy_ok  = raw_buy  and last_dir != 1
        sell_ok = raw_sell and last_dir != -1

        # Both fire same bar → prefer buy (matches Pine source)
        if buy_ok and sell_ok:
            sell_ok = False

        if buy_ok:
            # Stop-and-reverse: close any open short on the same bar.
            if in_pos == -1:
                short_exit[i] = True
            in_pos = 1
            entry_price = c_arr[i]
            risk = atr_arr[i] * sl_atr_mult
            sl_price = entry_price - risk
            trade_risk = abs(entry_price - sl_price)
            tp1_price = entry_price + trade_risk * tp1_r
            tp2_price = entry_price + trade_risk * tp2_r
            tp3_price = entry_price + trade_risk * tp3_r
            trail_price = sl_price
            tp1_hit = tp2_hit = tp3_hit = False
            long_entry[i] = True
            last_dir = 1

        elif sell_ok:
            if in_pos == 1:
                long_exit[i] = True
            in_pos = -1
            entry_price = c_arr[i]
            risk = atr_arr[i] * sl_atr_mult
            sl_price = entry_price + risk
            trade_risk = abs(entry_price - sl_price)
            tp1_price = entry_price - trade_risk * tp1_r
            tp2_price = entry_price - trade_risk * tp2_r
            tp3_price = entry_price - trade_risk * tp3_r
            trail_price = sl_price
            tp1_hit = tp2_hit = tp3_hit = False
            short_entry[i] = True
            last_dir = -1

    df["long_entry"]  = long_entry
    df["long_exit"]   = long_exit
    df["short_entry"] = short_entry
    df["short_exit"]  = short_exit
    return df
