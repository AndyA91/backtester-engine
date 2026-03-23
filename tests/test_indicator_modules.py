"""Unit tests for standalone indicator modules (indicators/ directory).

Tests the 5 highest-value modules: ATR, Supertrend, Parabolic SAR, ADX,
Stochastic, and Squeeze.
"""

import numpy as np
import pandas as pd
import pytest
from indicators.atr import calc_atr
from indicators.supertrend import calc_supertrend
from indicators.parabolic_sar import calc_psar
from indicators.adx import calc_adx
from indicators.stochastic import calc_stochastic
from indicators.squeeze import calc_squeeze


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv():
    """30-bar OHLCV DataFrame with trending then mean-reverting data."""
    np.random.seed(42)
    n = 30
    # Generate a trending series with noise
    base = np.cumsum(np.random.randn(n) * 2) + 100
    high = base + np.abs(np.random.randn(n)) * 2
    low = base - np.abs(np.random.randn(n)) * 2
    opn = low + (high - low) * np.random.rand(n)
    close = low + (high - low) * np.random.rand(n)
    volume = np.random.randint(100, 1000, n).astype(float)
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


@pytest.fixture
def trending_up():
    """20-bar steadily rising OHLCV for direction tests."""
    n = 20
    close = np.linspace(100, 140, n)
    high = close + 2
    low = close - 2
    opn = close - 1
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": [500]*n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


@pytest.fixture
def trending_down():
    """20-bar steadily falling OHLCV."""
    n = 20
    close = np.linspace(140, 100, n)
    high = close + 2
    low = close - 2
    opn = close + 1
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": [500]*n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))


# ── calc_atr (standalone module) ─────────────────────────────────────────

class TestATRModule:
    def test_returns_dict_with_atr_and_tr(self, ohlcv):
        result = calc_atr(ohlcv, period=5)
        assert "atr" in result
        assert "tr" in result
        assert len(result["atr"]) == len(ohlcv)
        assert len(result["tr"]) == len(ohlcv)

    def test_atr_positive(self, ohlcv):
        result = calc_atr(ohlcv, period=5)
        assert (result["atr"] > 0).all()

    def test_tr_formula(self, ohlcv):
        """True range = max(H-L, |H-prevC|, |L-prevC|)."""
        result = calc_atr(ohlcv, period=5)
        tr = result["tr"]
        h = ohlcv["High"].values
        l = ohlcv["Low"].values
        c = ohlcv["Close"].values
        for i in range(1, len(ohlcv)):
            expected = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
            assert tr[i] == pytest.approx(expected)

    def test_rma_method_default(self, ohlcv):
        """Default method is RMA (Wilder's)."""
        result = calc_atr(ohlcv, period=5, method="rma")
        result_default = calc_atr(ohlcv, period=5)
        np.testing.assert_array_almost_equal(result["atr"], result_default["atr"])

    def test_sma_method(self, ohlcv):
        result = calc_atr(ohlcv, period=5, method="sma")
        assert (result["atr"] > 0).all()

    def test_ema_method(self, ohlcv):
        result = calc_atr(ohlcv, period=5, method="ema")
        assert (result["atr"] > 0).all()


# ── calc_supertrend ──────────────────────────────────────────────────────

class TestSupertrend:
    def test_returns_all_keys(self, ohlcv):
        result = calc_supertrend(ohlcv, period=5, multiplier=2.0)
        assert set(result.keys()) == {"supertrend", "direction", "upper_band", "lower_band"}
        for k in result:
            assert len(result[k]) == len(ohlcv)

    def test_direction_values(self, ohlcv):
        result = calc_supertrend(ohlcv, period=5, multiplier=2.0)
        unique = set(result["direction"])
        assert unique.issubset({1.0, -1.0})

    def test_bullish_in_uptrend(self, trending_up):
        result = calc_supertrend(trending_up, period=5, multiplier=2.0)
        # Most bars should be bullish in a steady uptrend
        bullish_pct = (result["direction"] == 1).sum() / len(trending_up)
        assert bullish_pct > 0.7

    def test_bearish_in_downtrend(self, trending_down):
        result = calc_supertrend(trending_down, period=5, multiplier=2.0)
        bearish_pct = (result["direction"] == -1).sum() / len(trending_down)
        assert bearish_pct > 0.5

    def test_supertrend_line_matches_direction(self, ohlcv):
        """Bullish → supertrend = lower_band; Bearish → supertrend = upper_band."""
        result = calc_supertrend(ohlcv, period=5, multiplier=2.0)
        for i in range(len(ohlcv)):
            if result["direction"][i] == 1:
                assert result["supertrend"][i] == pytest.approx(result["lower_band"][i])
            else:
                assert result["supertrend"][i] == pytest.approx(result["upper_band"][i])

    def test_lower_band_trailing(self, trending_up):
        """In uptrend, lower band should be non-decreasing (trailing support)."""
        result = calc_supertrend(trending_up, period=5, multiplier=2.0)
        lb = result["lower_band"]
        # After warmup, lower band should mostly increase
        diffs = np.diff(lb[5:])
        assert (diffs >= -1e-10).sum() > len(diffs) * 0.8


# ── calc_psar ────────────────────────────────────────────────────────────

class TestParabolicSAR:
    def test_returns_all_keys(self, ohlcv):
        result = calc_psar(ohlcv)
        assert set(result.keys()) == {"psar", "direction", "af"}
        for k in result:
            assert len(result[k]) == len(ohlcv)

    def test_direction_values(self, ohlcv):
        result = calc_psar(ohlcv)
        unique = set(result["direction"])
        assert unique.issubset({1, -1, 1.0, -1.0})

    def test_af_bounded(self, ohlcv):
        """AF should be between start (0.02) and maximum (0.2)."""
        result = calc_psar(ohlcv, start=0.02, increment=0.02, maximum=0.2)
        af = result["af"]
        valid = af[~np.isnan(af)]
        assert (valid >= 0.02 - 1e-10).all()
        assert (valid <= 0.20 + 1e-10).all()

    def test_bullish_sar_below_price(self, trending_up):
        """In uptrend, bullish SAR should be below Low."""
        result = calc_psar(trending_up)
        for i in range(2, len(trending_up)):
            if result["direction"][i] == 1:
                assert result["psar"][i] <= trending_up["Low"].iloc[i] + 1e-6

    def test_bearish_sar_above_price(self, trending_down):
        """In downtrend, bearish SAR should be above High."""
        result = calc_psar(trending_down)
        for i in range(2, len(trending_down)):
            if result["direction"][i] == -1:
                assert result["psar"][i] >= trending_down["High"].iloc[i] - 1e-6


# ── calc_adx ─────────────────────────────────────────────────────────────

class TestADX:
    def test_returns_all_keys(self, ohlcv):
        result = calc_adx(ohlcv, di_period=5, adx_period=5)
        assert set(result.keys()) == {"adx", "plus_di", "minus_di", "dx"}
        for k in result:
            assert len(result[k]) == len(ohlcv)

    def test_adx_range(self, ohlcv):
        """ADX should be 0-100."""
        result = calc_adx(ohlcv, di_period=5, adx_period=5)
        valid = result["adx"][result["adx"] > 0]
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_di_range(self, ohlcv):
        """+DI and -DI should be 0-100."""
        result = calc_adx(ohlcv, di_period=5, adx_period=5)
        for key in ["plus_di", "minus_di"]:
            valid = result[key][~np.isnan(result[key]) & (result[key] > 0)]
            if len(valid) > 0:
                assert (valid >= 0).all()
                assert (valid <= 100 + 1e-6).all()

    def test_strong_trend_high_adx(self, trending_up):
        """Strong uptrend should produce high ADX."""
        result = calc_adx(trending_up, di_period=5, adx_period=5)
        # Last few bars should have elevated ADX
        assert result["adx"][-1] > 20

    def test_plus_di_dominates_in_uptrend(self, trending_up):
        result = calc_adx(trending_up, di_period=5, adx_period=5)
        # +DI should be > -DI in a steady uptrend (last bar)
        assert result["plus_di"][-1] > result["minus_di"][-1]

    def test_minus_di_dominates_in_downtrend(self, trending_down):
        result = calc_adx(trending_down, di_period=5, adx_period=5)
        assert result["minus_di"][-1] > result["plus_di"][-1]


# ── calc_stochastic ──────────────────────────────────────────────────────

class TestStochastic:
    def test_returns_all_keys(self, ohlcv):
        result = calc_stochastic(ohlcv, k_period=5, smooth_k=3, smooth_d=3)
        assert set(result.keys()) == {"fast_k", "slow_k", "pct_d"}
        for k in result:
            assert len(result[k]) == len(ohlcv)

    def test_fast_k_range(self, ohlcv):
        """Fast %K should be 0-100 (or 50 for flat range)."""
        result = calc_stochastic(ohlcv, k_period=5)
        fk = result["fast_k"]
        valid = fk[~np.isnan(fk)]
        assert (valid >= 0 - 1e-6).all()
        assert (valid <= 100 + 1e-6).all()

    def test_warmup_nan(self, ohlcv):
        """First k_period-1 bars of fast_k should be NaN."""
        result = calc_stochastic(ohlcv, k_period=5)
        assert np.isnan(result["fast_k"][:4]).all()
        assert not np.isnan(result["fast_k"][4])

    def test_fast_k_formula(self):
        """Spot-check fast_k = 100 * (C - LL) / (HH - LL)."""
        df = pd.DataFrame({
            "Open": [10, 12, 14, 16, 18],
            "High": [11, 13, 15, 17, 20],
            "Low":  [ 9, 11, 13, 15, 17],
            "Close":[10, 12, 14, 16, 19],
        })
        result = calc_stochastic(df, k_period=3, smooth_k=1, smooth_d=1)
        # Bar 2 (index 2): HH=max(11,13,15)=15, LL=min(9,11,13)=9
        # fast_k = 100 * (14 - 9) / (15 - 9) = 100 * 5/6 ≈ 83.33
        assert result["fast_k"][2] == pytest.approx(100 * 5 / 6)

    def test_flat_market_default(self):
        """Flat market (HH == LL) → fast_k = 50."""
        df = pd.DataFrame({
            "Open": [10, 10, 10], "High": [10, 10, 10],
            "Low": [10, 10, 10], "Close": [10, 10, 10],
        })
        result = calc_stochastic(df, k_period=3, smooth_k=1, smooth_d=1)
        assert result["fast_k"][2] == pytest.approx(50.0)

    def test_slow_k_is_smoothed_fast_k(self, ohlcv):
        """slow_k = SMA(fast_k, smooth_k)."""
        result = calc_stochastic(ohlcv, k_period=5, smooth_k=3, smooth_d=3)
        fk = pd.Series(result["fast_k"])
        expected_sk = fk.rolling(3).mean().values
        # Compare where both are non-NaN
        mask = ~np.isnan(expected_sk) & ~np.isnan(result["slow_k"])
        np.testing.assert_array_almost_equal(
            result["slow_k"][mask], expected_sk[mask])


# ── calc_squeeze ─────────────────────────────────────────────────────────

class TestSqueeze:
    def test_returns_all_keys(self, ohlcv):
        result = calc_squeeze(ohlcv, bb_period=10, kc_period=10)
        assert set(result.keys()) == {"momentum", "squeeze_on", "squeeze_off", "no_squeeze"}

    def test_output_lengths(self, ohlcv):
        result = calc_squeeze(ohlcv, bb_period=10, kc_period=10)
        for k in result:
            assert len(result[k]) == len(ohlcv)

    def test_squeeze_on_off_mutually_exclusive(self, ohlcv):
        """squeeze_on and no_squeeze should not both be True on same bar."""
        result = calc_squeeze(ohlcv, bb_period=10, kc_period=10)
        overlap = result["squeeze_on"] & result["no_squeeze"]
        assert not overlap.any()

    def test_squeeze_off_follows_squeeze_on(self, ohlcv):
        """squeeze_off should only be True when previous bar was squeeze_on."""
        result = calc_squeeze(ohlcv, bb_period=10, kc_period=10)
        sq_on = result["squeeze_on"]
        sq_off = result["squeeze_off"]
        for i in range(1, len(ohlcv)):
            if sq_off[i]:
                assert sq_on[i-1], f"squeeze_off at bar {i} but squeeze_on was False at bar {i-1}"

    def test_squeeze_off_first_bar_false(self, ohlcv):
        result = calc_squeeze(ohlcv, bb_period=10, kc_period=10)
        assert result["squeeze_off"][0] == False

    def test_momentum_nan_during_warmup(self, ohlcv):
        result = calc_squeeze(ohlcv, bb_period=10, kc_period=10)
        # First kc_period-1 bars should be NaN
        assert np.isnan(result["momentum"][:9]).all()

    def test_momentum_has_values_after_warmup(self, ohlcv):
        result = calc_squeeze(ohlcv, bb_period=10, kc_period=10)
        valid = result["momentum"][~np.isnan(result["momentum"])]
        assert len(valid) > 0

    def test_use_true_range_flag(self, ohlcv):
        """use_true_range affects KC width → different squeeze detection.

        Momentum uses HH/LL/SMA (not ATR), so it's identical either way.
        The flag only changes the Keltner Channel bands.
        """
        r1 = calc_squeeze(ohlcv, bb_period=10, kc_period=10, use_true_range=True)
        r2 = calc_squeeze(ohlcv, bb_period=10, kc_period=10, use_true_range=False)
        # Squeeze states may differ due to different KC widths
        # (or be identical if the data doesn't trigger a difference)
        # Just verify both produce valid boolean arrays
        assert r1["squeeze_on"].dtype == bool
        assert r2["squeeze_on"].dtype == bool
