"""Unit tests for indicator functions — correctness and TV-matching behaviour."""

import math
import numpy as np
import pandas as pd
import pytest
from engine.engine import (
    calc_ema, calc_smma, calc_sma, calc_wma, calc_hma, calc_ehma, calc_thma,
    calc_gaussian, calc_rsi, calc_macd, calc_atr,
    calc_highest, calc_lowest, calc_donchian, calc_obv, calc_ichimoku,
    detect_crossover, detect_crossunder, get_source,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def prices():
    """20-bar price series with known values."""
    return pd.Series([
        10, 11, 12, 11, 13, 14, 12, 15, 16, 14,
        13, 12, 11, 10, 11, 13, 15, 17, 16, 18,
    ], dtype=float)


@pytest.fixture
def ohlcv():
    """20-bar OHLCV DataFrame."""
    close = [10, 11, 12, 11, 13, 14, 12, 15, 16, 14,
             13, 12, 11, 10, 11, 13, 15, 17, 16, 18]
    high =  [11, 12, 13, 12, 14, 15, 13, 16, 17, 15,
             14, 13, 12, 11, 12, 14, 16, 18, 17, 19]
    low =   [ 9, 10, 11, 10, 12, 13, 11, 14, 15, 13,
             12, 11, 10,  9, 10, 12, 14, 16, 15, 17]
    opn =   [10, 10, 11, 12, 11, 13, 14, 12, 15, 16,
             14, 13, 12, 11, 10, 11, 13, 15, 17, 16]
    volume = [100]*20
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, dtype=float)


# ── calc_sma ─────────────────────────────────────────────────────────────

class TestSMA:
    def test_basic_sma(self, prices):
        sma = calc_sma(prices, 5)
        # First 4 bars should be NaN
        assert sma.iloc[:4].isna().all()
        # Bar 4 (index 4): mean of [10,11,12,11,13] = 11.4
        assert sma.iloc[4] == pytest.approx(11.4)

    def test_sma_length_1(self, prices):
        sma = calc_sma(prices, 1)
        pd.testing.assert_series_equal(sma, prices, check_names=False)


# ── calc_ema ─────────────────────────────────────────────────────────────

class TestEMA:
    def test_seed_is_sma(self, prices):
        """EMA seed (first value) = SMA of first `length` bars."""
        ema = calc_ema(prices, 5)
        expected_seed = prices.iloc[:5].mean()  # SMA of first 5
        assert ema.iloc[4] == pytest.approx(expected_seed)

    def test_nan_before_seed(self, prices):
        ema = calc_ema(prices, 5)
        assert ema.iloc[:4].isna().all()

    def test_ema_formula(self, prices):
        """Verify EMA recursive formula: ema[i] = val*k + ema[i-1]*(1-k)."""
        length = 5
        k = 2.0 / (length + 1)
        ema = calc_ema(prices, length)
        # Check bar 5 (index 5, value=14)
        expected = prices.iloc[5] * k + ema.iloc[4] * (1 - k)
        assert ema.iloc[5] == pytest.approx(expected)

    def test_nan_leading_input(self):
        """EMA with NaN-leading series finds seed from first valid window."""
        s = pd.Series([np.nan, np.nan, 10, 11, 12, 13, 14, 15])
        ema = calc_ema(s, 3)
        # First valid window of 3: indices 2,3,4 → seed = mean(10,11,12) = 11
        assert ema.iloc[4] == pytest.approx(11.0)
        # Should be NaN before seed
        assert ema.iloc[:4].isna().all()

    def test_nan_carry_forward(self):
        """NaN in the middle carries forward previous EMA value."""
        s = pd.Series([10.0, 11.0, 12.0, np.nan, 14.0])
        ema = calc_ema(s, 3)
        # Seed at index 2
        val_at_2 = ema.iloc[2]
        # Index 3 is NaN → carry forward
        assert ema.iloc[3] == pytest.approx(val_at_2)

    def test_not_enough_data(self):
        s = pd.Series([10.0, 11.0])
        ema = calc_ema(s, 5)
        assert ema.isna().all()


# ── calc_smma ────────────────────────────────────────────────────────────

class TestSMMA:
    def test_seed_is_sma(self, prices):
        smma = calc_smma(prices, 5)
        expected_seed = prices.iloc[:5].mean()
        assert smma.iloc[4] == pytest.approx(expected_seed)

    def test_smma_formula(self, prices):
        """smma[i] = (smma[i-1] * (length-1) + val) / length."""
        length = 5
        smma = calc_smma(prices, length)
        expected = (smma.iloc[4] * (length - 1) + prices.iloc[5]) / length
        assert smma.iloc[5] == pytest.approx(expected)

    def test_nan_before_seed(self, prices):
        smma = calc_smma(prices, 5)
        assert smma.iloc[:4].isna().all()


# ── calc_wma ─────────────────────────────────────────────────────────────

class TestWMA:
    def test_basic_wma(self):
        s = pd.Series([10.0, 20.0, 30.0])
        wma = calc_wma(s, 3)
        # weights [1,2,3], sum=6
        expected = (10*1 + 20*2 + 30*3) / 6
        assert wma.iloc[2] == pytest.approx(expected)

    def test_nan_warmup(self, prices):
        wma = calc_wma(prices, 5)
        assert wma.iloc[:4].isna().all()
        assert not np.isnan(wma.iloc[4])


# ── calc_hma ─────────────────────────────────────────────────────────────

class TestHMA:
    def test_hma_returns_series(self, prices):
        hma = calc_hma(prices, 9)
        assert isinstance(hma, pd.Series)
        assert len(hma) == len(prices)

    def test_hma_formula(self, prices):
        """HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
        length = 16
        s = pd.Series(np.random.RandomState(42).random(50) * 100)
        half = length // 2
        sqrt = round(math.sqrt(length))
        diff = 2 * calc_wma(s, half) - calc_wma(s, length)
        expected = calc_wma(diff, sqrt)
        result = calc_hma(s, length)
        # Compare non-NaN values
        mask = ~expected.isna() & ~result.isna()
        pd.testing.assert_series_equal(result[mask], expected[mask], check_names=False)


# ── calc_ehma / calc_thma ───────────────────────────────────────────────

class TestEHMA:
    def test_ehma_uses_ema(self, prices):
        """EHMA = EMA(2*EMA(n/2) - EMA(n), sqrt(n))."""
        length = 16
        s = pd.Series(np.random.RandomState(42).random(50) * 100)
        half = length // 2
        sqrt = round(math.sqrt(length))
        diff = 2 * calc_ema(s, half) - calc_ema(s, length)
        expected = calc_ema(diff, sqrt)
        result = calc_ehma(s, length)
        mask = ~expected.isna() & ~result.isna()
        pd.testing.assert_series_equal(result[mask], expected[mask], check_names=False)


class TestTHMA:
    def test_thma_formula(self):
        """THMA = WMA(3*WMA(n/3) - WMA(n/2) - WMA(n), n)."""
        length = 9
        s = pd.Series(np.random.RandomState(42).random(50) * 100)
        len3 = length // 3
        half = length // 2
        inner = 3 * calc_wma(s, len3) - calc_wma(s, half) - calc_wma(s, length)
        expected = calc_wma(inner, length)
        result = calc_thma(s, length)
        mask = ~expected.isna() & ~result.isna()
        pd.testing.assert_series_equal(result[mask], expected[mask], check_names=False)


# ── calc_gaussian ────────────────────────────────────────────────────────

class TestGaussian:
    def test_poles_1_is_ema(self, prices):
        g1 = calc_gaussian(prices, 5, poles=1)
        ema = calc_ema(prices, 5)
        pd.testing.assert_series_equal(g1, ema, check_names=False)

    def test_poles_2_is_double_ema(self, prices):
        g2 = calc_gaussian(prices, 5, poles=2)
        expected = calc_ema(calc_ema(prices, 5), 5)
        mask = ~expected.isna() & ~g2.isna()
        pd.testing.assert_series_equal(g2[mask], expected[mask], check_names=False)

    def test_poles_clamped(self, prices):
        """Poles clamped to 1-4; poles=0 → 1, poles=10 → 4."""
        g0 = calc_gaussian(prices, 5, poles=0)
        g1 = calc_gaussian(prices, 5, poles=1)
        pd.testing.assert_series_equal(g0, g1, check_names=False)

        g10 = calc_gaussian(prices, 5, poles=10)
        g4 = calc_gaussian(prices, 5, poles=4)
        mask = ~g10.isna() & ~g4.isna()
        pd.testing.assert_series_equal(g10[mask], g4[mask], check_names=False)


# ── calc_rsi ─────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_range(self, prices):
        rsi = calc_rsi(prices, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_uses_smma(self, prices):
        """RSI should use SMMA (Wilder's) not SMA for avg gain/loss."""
        rsi = calc_rsi(prices, 5)
        # Manually compute to verify
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = calc_smma(gain, 5)
        avg_loss = calc_smma(loss, 5)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        expected = 100 - (100 / (1 + rs))
        mask = ~expected.isna() & ~rsi.isna()
        pd.testing.assert_series_equal(rsi[mask], expected[mask], check_names=False)

    def test_monotone_up_rsi_is_nan(self):
        """Monotonically increasing prices → zero losses → avg_loss=0 → RSI=NaN.

        This is a known edge case: the implementation does replace(0, NaN)
        on avg_loss to avoid division by zero, which produces NaN instead of 100.
        """
        s = pd.Series(range(1, 31), dtype=float)
        rsi = calc_rsi(s, 14)
        # avg_loss is 0 → rs = inf → NaN propagation
        assert np.isnan(rsi.iloc[-1])


# ── calc_macd ────────────────────────────────────────────────────────────

class TestMACD:
    def test_returns_three_series(self, prices):
        macd_line, signal, histogram = calc_macd(prices, 5, 10, 3)
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)

    def test_macd_is_fast_minus_slow(self, prices):
        macd_line, _, _ = calc_macd(prices, 5, 10, 3)
        expected = calc_ema(prices, 5) - calc_ema(prices, 10)
        mask = ~expected.isna() & ~macd_line.isna()
        pd.testing.assert_series_equal(macd_line[mask], expected[mask], check_names=False)

    def test_histogram_is_macd_minus_signal(self, prices):
        macd_line, signal, histogram = calc_macd(prices, 5, 10, 3)
        expected = macd_line - signal
        mask = ~expected.isna() & ~histogram.isna()
        pd.testing.assert_series_equal(histogram[mask], expected[mask], check_names=False)


# ── calc_atr ─────────────────────────────────────────────────────────────

class TestATR:
    def test_atr_positive(self, ohlcv):
        atr = calc_atr(ohlcv, 5)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_true_range_formula(self, ohlcv):
        """ATR = SMMA of true range. True range = max(H-L, |H-prevC|, |L-prevC|)."""
        tr = pd.concat([
            ohlcv["High"] - ohlcv["Low"],
            (ohlcv["High"] - ohlcv["Close"].shift(1)).abs(),
            (ohlcv["Low"] - ohlcv["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        expected = calc_smma(tr, 5)
        result = calc_atr(ohlcv, 5)
        mask = ~expected.isna() & ~result.isna()
        pd.testing.assert_series_equal(result[mask], expected[mask], check_names=False)


# ── calc_highest / calc_lowest ───────────────────────────────────────────

class TestHighestLowest:
    def test_highest(self, prices):
        h = calc_highest(prices, 3)
        # Index 2: max(10,11,12) = 12
        assert h.iloc[2] == 12.0
        # Index 7: max(14,12,15) = 15
        assert h.iloc[7] == 15.0

    def test_lowest(self, prices):
        lo = calc_lowest(prices, 3)
        # Index 2: min(10,11,12) = 10
        assert lo.iloc[2] == 10.0

    def test_nan_warmup(self, prices):
        h = calc_highest(prices, 5)
        assert h.iloc[:4].isna().all()
        assert not np.isnan(h.iloc[4])


# ── calc_donchian ────────────────────────────────────────────────────────

class TestDonchian:
    def test_donchian_components(self, ohlcv):
        upper, lower, mid = calc_donchian(ohlcv["High"], ohlcv["Low"], 5)
        # mid = (upper + lower) / 2
        mask = ~upper.isna()
        expected_mid = (upper[mask] + lower[mask]) / 2
        pd.testing.assert_series_equal(mid[mask], expected_mid, check_names=False)

    def test_upper_is_highest_high(self, ohlcv):
        upper, _, _ = calc_donchian(ohlcv["High"], ohlcv["Low"], 5)
        expected = calc_highest(ohlcv["High"], 5)
        pd.testing.assert_series_equal(upper, expected, check_names=False)

    def test_lower_is_lowest_low(self, ohlcv):
        _, lower, _ = calc_donchian(ohlcv["High"], ohlcv["Low"], 5)
        expected = calc_lowest(ohlcv["Low"], 5)
        pd.testing.assert_series_equal(lower, expected, check_names=False)


# ── calc_obv ─────────────────────────────────────────────────────────────

class TestOBV:
    def test_obv_basic(self):
        close = pd.Series([10, 11, 10, 12, 12], dtype=float)
        volume = pd.Series([100, 200, 150, 300, 250], dtype=float)
        obv = calc_obv(close, volume)
        # Bar 0: 0
        # Bar 1: close up → +200 = 200
        # Bar 2: close down → -150 = 50
        # Bar 3: close up → +300 = 350
        # Bar 4: close equal → 350
        expected = [0, 200, 50, 350, 350]
        for i, exp in enumerate(expected):
            assert obv.iloc[i] == pytest.approx(exp)

    def test_obv_starts_at_zero(self, ohlcv):
        obv = calc_obv(ohlcv["Close"], ohlcv["Volume"])
        assert obv.iloc[0] == 0.0


# ── calc_ichimoku ────────────────────────────────────────────────────────

class TestIchimoku:
    def test_returns_all_keys(self, ohlcv):
        result = calc_ichimoku(ohlcv["High"], ohlcv["Low"],
                               conversion_periods=3, base_periods=5,
                               lagging_span2_periods=10, displacement=5)
        expected_keys = {"conversion", "base", "lead_a", "lead_b",
                         "displaced_lead_a", "displaced_lead_b"}
        assert set(result.keys()) == expected_keys

    def test_conversion_is_donchian_mid(self, ohlcv):
        result = calc_ichimoku(ohlcv["High"], ohlcv["Low"],
                               conversion_periods=3, base_periods=5,
                               lagging_span2_periods=10, displacement=5)
        _, _, expected = calc_donchian(ohlcv["High"], ohlcv["Low"], 3)
        pd.testing.assert_series_equal(result["conversion"], expected, check_names=False)

    def test_displaced_is_shifted(self, ohlcv):
        disp = 5
        result = calc_ichimoku(ohlcv["High"], ohlcv["Low"],
                               conversion_periods=3, base_periods=5,
                               lagging_span2_periods=10, displacement=disp)
        # displaced_lead_a should be lead_a shifted by displacement
        expected = result["lead_a"].shift(disp)
        pd.testing.assert_series_equal(
            result["displaced_lead_a"], expected, check_names=False)


# ── detect_crossover / detect_crossunder ─────────────────────────────────

class TestCrossDetection:
    def test_crossover_basic(self):
        fast = pd.Series([1, 2, 5, 4])
        slow = pd.Series([3, 3, 3, 3])
        cross = detect_crossover(fast, slow)
        # Bar 0: no prev → False
        # Bar 1: fast(2)<=slow(3) prev, fast(2)<=slow(3) now → False (not above)
        # Bar 2: fast_prev(2)<=slow_prev(3) and fast(5)>slow(3) → True
        # Bar 3: fast_prev(5)>slow_prev(3) → False (was already above)
        assert cross.iloc[2] == True
        assert cross.iloc[3] == False

    def test_crossunder_basic(self):
        fast = pd.Series([5, 4, 2, 3])
        slow = pd.Series([3, 3, 3, 3])
        cross = detect_crossunder(fast, slow)
        # Bar 2: fast_prev(4)>=slow_prev(3) and fast(2)<slow(3) → True
        assert cross.iloc[2] == True
        assert cross.iloc[3] == False

    def test_crossover_equal_prior_bar(self):
        """Equal values on prior bar: fast_prev <= slow_prev is True → cross valid."""
        fast = pd.Series([3, 3, 5])
        slow = pd.Series([3, 3, 3])
        cross = detect_crossover(fast, slow)
        assert cross.iloc[2] == True

    def test_crossunder_equal_prior_bar(self):
        fast = pd.Series([3, 3, 1])
        slow = pd.Series([3, 3, 3])
        cross = detect_crossunder(fast, slow)
        assert cross.iloc[2] == True

    def test_no_cross_returns_all_false(self):
        fast = pd.Series([1, 1, 1, 1])
        slow = pd.Series([3, 3, 3, 3])
        cross = detect_crossover(fast, slow)
        assert not cross.any()


# ── get_source ───────────────────────────────────────────────────────────

class TestGetSource:
    def test_close(self, ohlcv):
        pd.testing.assert_series_equal(get_source(ohlcv, "close"), ohlcv["Close"])

    def test_open(self, ohlcv):
        pd.testing.assert_series_equal(get_source(ohlcv, "Open"), ohlcv["Open"])

    def test_hl2(self, ohlcv):
        expected = (ohlcv["High"] + ohlcv["Low"]) / 2
        pd.testing.assert_series_equal(get_source(ohlcv, "hl2"), expected)

    def test_hlc3(self, ohlcv):
        expected = (ohlcv["High"] + ohlcv["Low"] + ohlcv["Close"]) / 3
        pd.testing.assert_series_equal(get_source(ohlcv, "hlc3"), expected)

    def test_ohlc4(self, ohlcv):
        expected = (ohlcv["Open"] + ohlcv["High"] + ohlcv["Low"] + ohlcv["Close"]) / 4
        pd.testing.assert_series_equal(get_source(ohlcv, "ohlc4"), expected)

    def test_case_insensitive(self, ohlcv):
        pd.testing.assert_series_equal(
            get_source(ohlcv, "CLOSE"), get_source(ohlcv, "close"))

    def test_unknown_source_raises(self, ohlcv):
        with pytest.raises(ValueError, match="Unknown source"):
            get_source(ohlcv, "vwap")
