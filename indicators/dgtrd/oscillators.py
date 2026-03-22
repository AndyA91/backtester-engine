"""
Python translation of:
  Oscillators Overlay w/ Divergencies/Alerts by DGT
  https://www.tradingview.com/script/0c9x2Jam-Oscillators-Overlay-w-Divergencies-Alerts-by-DGT/

Implements all 15 oscillators from the indicator plus oscillator-based
divergence detection (regular and hidden, bull and bear).

Output columns (prefix defaults to 'osc_')
-------------------------------------------
  {p}value        Oscillator main line
  {p}signal       Signal line (NaN if oscillator has no signal line)
  {p}histogram    osc - signal (NaN if no signal)
  {p}bull_div     Regular Bullish divergence flag  (osc HL + price LL)
  {p}bear_div     Regular Bearish divergence flag  (osc LH + price HH)
  {p}hbull_div    Hidden  Bullish divergence flag  (osc LL + price HL)
  {p}hbear_div    Hidden  Bearish divergence flag  (osc HH + price LH)

Supported oscillator types (osc_type parameter)
------------------------------------------------
  "AO"           Awesome Oscillator
  "Chaikin"      Chaikin Oscillator
  "CCI"          Commodity Channel Index
  "Distance"     Distance Oscillator
  "ElderRay"     Elder-Ray Bear/Bull Power
  "EWO"          Elliott Wave Oscillator
  "Klinger"      Klinger Oscillator
  "MFI"          Money Flow Index
  "MACD"         MACD
  "ROC"          Rate Of Change
  "RSI"          Relative Strength Index
  "Stoch"        Stochastic
  "StochRSI"     Stochastic RSI
  "VolumeOsc"    Volume Oscillator
  "WaveTrend"    Wave Trend [LazyBear]

Usage
-----
  from indicators.dgtrd.oscillators import oscillators_overlay

  df = oscillators_overlay(df, osc_type="MACD")
  df = oscillators_overlay(df, osc_type="RSI", rsi_length=14, prefix="rsi_")
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MA helpers (matching Pine's ma() switch)
# ---------------------------------------------------------------------------

def _sma(s: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(s), np.nan)
    for i in range(n - 1, len(s)):
        out[i] = np.mean(s[i - n + 1 : i + 1])
    return out


def _ema(s: np.ndarray, n: int) -> np.ndarray:
    """EMA matching Pine ta.ema(): seed = SMA of first n valid values."""
    out = np.full(len(s), np.nan)
    mult = 2.0 / (n + 1)
    count = 0; start = -1
    valid = ~np.isnan(s)
    for i in range(len(s)):
        if valid[i]:
            count += 1
            if count == n:
                start = i - n + 1
                break
        else:
            count = 0
    if start < 0:
        return out
    seed = start + n - 1
    out[seed] = np.mean(s[start : seed + 1])
    for i in range(seed + 1, len(s)):
        out[i] = s[i] * mult + out[i - 1] * (1 - mult) if not np.isnan(s[i]) else out[i - 1]
    return out


def _rma(s: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RMA = EMA with alpha=1/n. Seed = SMA of first n valid values."""
    out = np.full(len(s), np.nan)
    alpha = 1.0 / n
    count = 0; start = -1
    valid = ~np.isnan(s)
    for i in range(len(s)):
        if valid[i]:
            count += 1
            if count == n:
                start = i - n + 1
                break
        else:
            count = 0
    if start < 0:
        return out
    seed = start + n - 1
    out[seed] = np.mean(s[start : seed + 1])
    for i in range(seed + 1, len(s)):
        v = s[i] if not np.isnan(s[i]) else 0.0
        out[i] = alpha * v + (1 - alpha) * out[i - 1]
    return out


def _wma(s: np.ndarray, n: int) -> np.ndarray:
    """Weighted MA matching Pine ta.wma()."""
    out = np.full(len(s), np.nan)
    weights = np.arange(1, n + 1, dtype=float)
    denom = weights.sum()
    for i in range(n - 1, len(s)):
        window = s[i - n + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = np.dot(window, weights) / denom
    return out


def _vwma(s: np.ndarray, vol: np.ndarray, n: int) -> np.ndarray:
    """Volume-Weighted MA matching Pine ta.vwma()."""
    out = np.full(len(s), np.nan)
    for i in range(n - 1, len(s)):
        sw = s  [i - n + 1 : i + 1]
        vw = vol[i - n + 1 : i + 1]
        if np.any(np.isnan(sw)) or vw.sum() == 0:
            continue
        out[i] = np.dot(sw, vw) / vw.sum()
    return out


def _ma(s: np.ndarray, n: int, ma_type: str, vol: np.ndarray = None) -> np.ndarray:
    """Dispatcher matching Pine ma() switch."""
    t = ma_type.upper()
    if t == "SMA":
        return _sma(s, n)
    if t == "EMA":
        return _ema(s, n)
    if t == "RMA":
        return _rma(s, n)
    if t == "WMA":
        return _wma(s, n)
    if t == "VWMA" and vol is not None:
        return _vwma(s, vol, n)
    raise ValueError(f"Unknown MA type: {ma_type}")


# ---------------------------------------------------------------------------
# Oscillator implementations
# ---------------------------------------------------------------------------

def _accdist(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             vol: np.ndarray) -> np.ndarray:
    """Pine ta.accdist — cumulative Accumulation/Distribution."""
    hl = high - low
    mfm = np.where(hl == 0, 0.0, (2 * close - high - low) / hl)
    return np.nancumsum(mfm * vol)


def _rsi(src: np.ndarray, n: int) -> np.ndarray:
    """RSI using Wilder's RMA (matches Pine ta.rsi)."""
    delta = np.diff(src, prepend=np.nan)
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = _rma(gain, n)
    avg_l = _rma(loss, n)
    rs    = np.where(avg_l == 0, np.inf, avg_g / avg_l)
    return 100.0 - 100.0 / (1.0 + rs)


def _stoch(close: np.ndarray, high: np.ndarray, low: np.ndarray,
           n: int) -> np.ndarray:
    """Pine ta.stoch(close, high, low, n)."""
    out = np.full(len(close), np.nan)
    for i in range(n - 1, len(close)):
        lo = np.min(low [i - n + 1 : i + 1])
        hi = np.max(high[i - n + 1 : i + 1])
        r  = hi - lo
        out[i] = 100.0 * (close[i] - lo) / r if r != 0 else 50.0
    return out


def _mfi(src: np.ndarray, vol: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.mfi(src, n) — MFI based on flow direction."""
    raw_mf = src * vol
    change = np.diff(src, prepend=np.nan)
    pos_mf = np.where(change > 0, raw_mf, 0.0)
    neg_mf = np.where(change < 0, raw_mf, 0.0)

    out = np.full(len(src), np.nan)
    for i in range(n - 1, len(src)):
        p = np.sum(pos_mf[i - n + 1 : i + 1])
        q = np.sum(neg_mf[i - n + 1 : i + 1])
        total = p + q
        out[i] = 100.0 * p / total if total > 0 else 50.0
    return out


def _cci(src: np.ndarray, n: int) -> np.ndarray:
    """CCI using mean absolute deviation (Pine ta.dev)."""
    out = np.full(len(src), np.nan)
    for i in range(n - 1, len(src)):
        window = src[i - n + 1 : i + 1]
        m   = np.mean(window)
        mad = np.mean(np.abs(window - m))
        if mad == 0:
            continue
        out[i] = (src[i] - m) / (0.015 * mad)
    return out


# ---------------------------------------------------------------------------
# Public: compute a single oscillator
# ---------------------------------------------------------------------------

def compute_oscillator(
    df: pd.DataFrame,
    osc_type: str = "MACD",
    # CCI
    cci_length: int = 20,
    cci_smoothing: bool = False,
    cci_ma_type: str = "SMA",
    cci_smooth_length: int = 5,
    # Distance
    pma_length: int = 20,
    pma_ma_type: str = "SMA",
    pma_signal_length: int = 9,
    pma_signal_ma: str = "EMA",
    # EWO
    ewo_signal: bool = True,
    ewo_signal_ma_type: str = "SMA",
    ewo_signal_length: int = 5,
    # Chaikin
    chaikin_short: int = 3,
    chaikin_long: int = 10,
    # MFI
    mfi_length: int = 14,
    # MACD
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_src_ma: str = "EMA",
    macd_sig_ma: str = "EMA",
    # ROC
    roc_length: int = 9,
    # RSI
    rsi_length: int = 14,
    rsi_smoothing: bool = True,
    rsi_ma_type: str = "EMA",
    rsi_ma_length: int = 14,
    # Stoch
    stoch_k: int = 14,
    stoch_smooth_k: int = 1,
    stoch_d: int = 3,
    # StochRSI
    stochrsi_smooth_k: int = 3,
    stochrsi_d: int = 3,
    stochrsi_rsi_length: int = 14,
    stochrsi_stoch_length: int = 14,
    # VolumeOsc
    vol_short: int = 5,
    vol_long: int = 10,
    # WaveTrend
    wave_ch: int = 10,
    wave_avg: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the selected oscillator.

    Returns (osc, signal, histogram) as numpy arrays.
    `signal` and `histogram` are NaN-filled arrays when the oscillator has no
    signal line (matches Pine's na return).
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = np.nan_to_num(df["Volume"].to_numpy(dtype=float), nan=0.0)

    hl2  = (h + l) / 2
    hlc3 = (h + l + c) / 3
    nan  = np.full(len(c), np.nan)

    t = osc_type.strip()

    if t == "AO":
        osc  = _ma(hl2, 5,  "SMA") - _ma(hl2, 34, "SMA")
        sig  = nan.copy()
        hist = osc.copy()

    elif t == "Chaikin":
        ad   = _accdist(h, l, c, v)
        osc  = _ma(ad, chaikin_short, "EMA") - _ma(ad, chaikin_long, "EMA")
        sig  = nan.copy()
        hist = nan.copy()

    elif t == "CCI":
        osc  = _cci(hlc3, cci_length)
        sig  = _ma(osc, cci_smooth_length, cci_ma_type) if cci_smoothing else nan.copy()
        hist = nan.copy()

    elif t == "Distance":
        raw  = (c / _ma(c, pma_length, pma_ma_type) - 1) * 100
        osc  = raw
        sig  = _ma(osc, pma_signal_length, pma_signal_ma)
        hist = osc - sig

    elif t == "ElderRay":
        ema13 = _ma(c, 13, "EMA")
        osc   = h - ema13                   # bull power
        sig   = ema13 - l                   # bear power (Pine: -(low - EMA))
        hist  = nan.copy()

    elif t == "EWO":
        osc  = _ma(c, 5, "SMA") - _ma(c, 35, "SMA")
        sig  = _ma(osc, ewo_signal_length, ewo_signal_ma_type) if ewo_signal else nan.copy()
        hist = osc.copy()

    elif t == "Klinger":
        change_hlc3 = np.diff(hlc3, prepend=np.nan)
        sv   = np.where(change_hlc3 >= 0, v, -v)
        osc  = _ma(sv, 34, "EMA") - _ma(sv, 55, "EMA")
        sig  = _ma(osc, 13, "EMA")
        hist = nan.copy()

    elif t == "MFI":
        osc  = _mfi(hlc3, v, mfi_length)
        sig  = nan.copy()
        hist = nan.copy()

    elif t == "MACD":
        osc  = _ma(c, macd_fast, macd_src_ma) - _ma(c, macd_slow, macd_src_ma)
        sig  = _ma(osc, macd_signal, macd_sig_ma)
        hist = osc - sig

    elif t == "ROC":
        src_shift = np.full(len(c), np.nan)
        src_shift[roc_length:] = c[:len(c) - roc_length]
        osc  = 100.0 * (c - src_shift) / src_shift
        sig  = nan.copy()
        hist = nan.copy()

    elif t == "RSI":
        osc  = _rsi(c, rsi_length)
        sig  = _ma(osc, rsi_ma_length, rsi_ma_type) if rsi_smoothing else nan.copy()
        hist = nan.copy()

    elif t == "Stoch":
        raw_k = _stoch(c, h, l, stoch_k)
        osc   = _ma(raw_k, stoch_smooth_k, "SMA")
        sig   = _ma(osc, stoch_d, "SMA")
        hist  = nan.copy()

    elif t == "StochRSI":
        rsix  = _rsi(c, stochrsi_rsi_length)
        stk   = _stoch(rsix, rsix, rsix, stochrsi_stoch_length)
        osc   = _ma(stk, stochrsi_smooth_k, "SMA")
        sig   = _ma(osc, stochrsi_d, "SMA")
        hist  = nan.copy()

    elif t == "VolumeOsc":
        ema_s = _ma(v, vol_short, "EMA")
        ema_l = _ma(v, vol_long,  "EMA")
        osc   = 100.0 * (ema_s - ema_l) / ema_l
        sig   = nan.copy()
        hist  = nan.copy()

    elif t == "WaveTrend":
        esa  = _ma(hlc3, wave_ch, "EMA")
        d    = _ma(np.abs(hlc3 - esa), wave_ch, "EMA")
        ci   = (hlc3 - esa) / (0.015 * np.where(d == 0, 1e-10, d))
        osc  = _ma(ci, wave_avg, "EMA")
        sig  = _ma(osc, 4, "SMA")
        hist = osc - sig

    else:
        raise ValueError(f"Unknown oscillator type: '{osc_type}'. "
                         "Choose from: AO, Chaikin, CCI, Distance, ElderRay, "
                         "EWO, Klinger, MFI, MACD, ROC, RSI, Stoch, StochRSI, "
                         "VolumeOsc, WaveTrend")

    return osc, sig, hist


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

def _detect_osc_pivots(
    osc: np.ndarray,
    lb_l: int,
    lb_r: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find pivot highs/lows in the oscillator series.
    Returns (ph_detection_bars, pl_detection_bars) — integer arrays of bar
    indices where a pivot is detected (= pivot_bar + lb_r), with NaN elsewhere.
    ph_val[detect_bar] = osc value at the pivot bar.
    """
    n = len(osc)
    ph_val  = np.full(n, np.nan)   # value at detection bar
    pl_val  = np.full(n, np.nan)

    for i in range(lb_l, n - lb_r):
        window = osc[i - lb_l : i + lb_r + 1]
        if np.any(np.isnan(window)):
            continue
        detect = i + lb_r
        if detect >= n:
            continue
        # strict max (no tie) — matches MEMORY.md §6
        if osc[i] == np.max(window) and np.sum(window == osc[i]) == 1:
            ph_val[detect] = osc[i]
        if osc[i] == np.min(window) and np.sum(window == osc[i]) == 1:
            pl_val[detect] = osc[i]

    return ph_val, pl_val


def detect_divergences(
    osc: np.ndarray,
    price_high: np.ndarray,
    price_low: np.ndarray,
    lb_l: int = 5,
    lb_r: int = 5,
    range_lower: int = 5,
    range_upper: int = 60,
) -> dict[str, np.ndarray]:
    """
    Detect regular and hidden oscillator divergences against price.

    Matches Pine's divergence logic (lines 236-268).
    Detection bars are `lb_r` bars after the oscillator pivot bar.

    Returns a dict with keys: bull, bear, hbull, hbear — bool arrays.

    Regular Bullish  : osc Higher Low  + price Lower Low
    Regular Bearish  : osc Lower High  + price Higher High
    Hidden  Bullish  : osc Lower Low   + price Higher Low
    Hidden  Bearish  : osc Higher High + price Lower High
    """
    n = len(osc)
    ph_val, pl_val = _detect_osc_pivots(osc, lb_l, lb_r)

    bull  = np.zeros(n, dtype=bool)
    bear  = np.zeros(n, dtype=bool)
    hbull = np.zeros(n, dtype=bool)
    hbear = np.zeros(n, dtype=bool)

    # Collect ordered lists of (detect_bar, osc_val, price_val) for lows/highs
    pl_events: list[tuple[int, float, float]] = []
    ph_events: list[tuple[int, float, float]] = []

    for i in range(n):
        if not np.isnan(pl_val[i]):
            pbar = i - lb_r                        # actual pivot bar
            pl_events.append((i, pl_val[i], price_low[pbar]))
        if not np.isnan(ph_val[i]):
            pbar = i - lb_r
            ph_events.append((i, ph_val[i], price_high[pbar]))

    # Regular & Hidden Bullish (from pivot lows)
    for k in range(1, len(pl_events)):
        d_cur, osc_cur, px_cur   = pl_events[k]
        d_prv, osc_prv, px_prv   = pl_events[k - 1]

        bar_gap = d_cur - d_prv
        if not (range_lower <= bar_gap <= range_upper):
            continue

        if osc_cur > osc_prv and px_cur < px_prv:   # regular bull: osc HL + price LL
            bull[d_cur] = True
        if osc_cur < osc_prv and px_cur > px_prv:   # hidden bull: osc LL + price HL
            hbull[d_cur] = True

    # Regular & Hidden Bearish (from pivot highs)
    for k in range(1, len(ph_events)):
        d_cur, osc_cur, px_cur   = ph_events[k]
        d_prv, osc_prv, px_prv   = ph_events[k - 1]

        bar_gap = d_cur - d_prv
        if not (range_lower <= bar_gap <= range_upper):
            continue

        if osc_cur < osc_prv and px_cur > px_prv:   # regular bear: osc LH + price HH
            bear[d_cur] = True
        if osc_cur > osc_prv and px_cur < px_prv:   # hidden bear: osc HH + price LH
            hbear[d_cur] = True

    return {"bull": bull, "bear": bear, "hbull": hbull, "hbear": hbear}


# ---------------------------------------------------------------------------
# Umbrella function
# ---------------------------------------------------------------------------

def oscillators_overlay(
    df: pd.DataFrame,
    osc_type: str = "MACD",
    prefix: str   = "osc_",
    lb_l: int = 5,
    lb_r: int = 5,
    range_lower: int = 5,
    range_upper: int = 60,
    **osc_kwargs,
) -> pd.DataFrame:
    """
    Compute oscillator + divergence columns and attach to df.

    Parameters
    ----------
    df          DataFrame with OHLCV columns.
    osc_type    Oscillator type string (see module docstring).
    prefix      Column name prefix (default 'osc_').
    lb_l/lb_r   Pivot lookback left/right for divergence (Pine default 5/5).
    range_lower/range_upper  In-range bar gap for divergence (Pine 5/60).
    **osc_kwargs  Passed to compute_oscillator().

    Returns
    -------
    df copy with oscillator and divergence columns.
    """
    osc, sig, hist = compute_oscillator(df, osc_type=osc_type, **osc_kwargs)

    divs = detect_divergences(
        osc,
        df["High"].to_numpy(dtype=float),
        df["Low"].to_numpy(dtype=float),
        lb_l=lb_l, lb_r=lb_r,
        range_lower=range_lower, range_upper=range_upper,
    )

    df = df.copy()
    df[f"{prefix}value"]     = osc
    df[f"{prefix}signal"]    = sig
    df[f"{prefix}histogram"] = hist
    df[f"{prefix}bull_div"]  = divs["bull"]
    df[f"{prefix}bear_div"]  = divs["bear"]
    df[f"{prefix}hbull_div"] = divs["hbull"]
    df[f"{prefix}hbear_div"] = divs["hbear"]

    return df
