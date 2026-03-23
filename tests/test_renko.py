"""Tests for Renko data loading and indicator enrichment pipeline.

Priority 5: renko/data.py (load_renko_export) and
renko/indicators.py (add_renko_indicators).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── Renko data loading ──────────────────────────────────────────────────

class TestLoadRenkoExport:

    @pytest.fixture
    def csv_path(self, tmp_path, monkeypatch):
        """Create a minimal Renko-style CSV with fractional timestamps."""
        import renko.data as renko_data

        # 10 bricks with fractional timestamps (multiple per second)
        base_ts = 1704067200.0  # 2024-01-01 00:00:00
        rows = []
        for i in range(10):
            ts = base_ts + i * 0.001  # fractional seconds
            o = 100 + i
            c = 101 + i  # up bricks
            h = max(o, c) + 0.5
            l = min(o, c) - 0.5
            rows.append(f"{ts},{o},{h},{l},{c}")

        csv_content = "time,open,high,low,close\n" + "\n".join(rows) + "\n"
        csv_file = tmp_path / "test_renko.csv"
        csv_file.write_text(csv_content)
        monkeypatch.setattr(renko_data, "_DATA_DIR", tmp_path)
        return "test_renko.csv"

    def test_loads_correct_brick_count(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        # 10 bricks minus 1 dropped = 9
        assert len(df) == 9

    def test_fractional_timestamps_preserved(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        # All indices should be unique (no duplicates from truncation)
        assert not df.index.duplicated().any()

    def test_column_names(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        for col in ["Open", "High", "Low", "Close", "Volume", "brick_up"]:
            assert col in df.columns

    def test_brick_up_column(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        # All bricks in our fixture are up bricks (Close > Open)
        assert df["brick_up"].all()

    def test_brick_down(self, tmp_path, monkeypatch):
        """Down brick: Close < Open → brick_up = False."""
        import renko.data as renko_data
        base_ts = 1704067200.0
        rows = []
        for i in range(5):
            ts = base_ts + i * 0.001
            o = 110 - i  # descending open
            c = 109 - i  # close below open = down
            h = max(o, c) + 0.5
            l = min(o, c) - 0.5
            rows.append(f"{ts},{o},{h},{l},{c}")
        csv_content = "time,open,high,low,close\n" + "\n".join(rows) + "\n"
        csv_file = tmp_path / "down_renko.csv"
        csv_file.write_text(csv_content)
        monkeypatch.setattr(renko_data, "_DATA_DIR", tmp_path)

        df = renko_data.load_renko_export("down_renko.csv")
        assert not df["brick_up"].any()

    def test_drops_last_brick(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        assert df.index[-1] < pd.Timestamp("2024-01-01 00:00:00.009")

    def test_adds_dummy_volume(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        assert "Volume" in df.columns

    def test_file_not_found(self, monkeypatch):
        import renko.data as renko_data
        monkeypatch.setattr(renko_data, "_DATA_DIR", Path("/nonexistent"))
        with pytest.raises(FileNotFoundError, match="Renko export not found"):
            renko_data.load_renko_export("nope.csv")

    def test_too_few_bricks(self, tmp_path, monkeypatch):
        import renko.data as renko_data
        ts = 1704067200.0
        csv_content = f"time,open,high,low,close\n{ts},100,101,99,100.5\n"
        csv_file = tmp_path / "one_brick.csv"
        csv_file.write_text(csv_content)
        monkeypatch.setattr(renko_data, "_DATA_DIR", tmp_path)
        with pytest.raises(ValueError, match="only 1 brick"):
            renko_data.load_renko_export("one_brick.csv")

    def test_datetime_index(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"

    def test_sorted_by_index(self, csv_path):
        from renko.data import load_renko_export
        df = load_renko_export(csv_path)
        assert df.index.is_monotonic_increasing


# ── Renko indicator enrichment ───────────────────────────────────────────

class TestAddRenkoIndicators:
    """Tests for add_renko_indicators() pipeline."""

    @pytest.fixture
    def renko_df(self):
        """Create a synthetic 250-bar Renko-like DataFrame.

        Enough bars for all indicators to warm up (longest: EMA200).
        """
        np.random.seed(42)
        n = 250
        # Simulate alternating up/down bricks with a slight upward drift
        base = np.cumsum(np.random.randn(n) * 0.5) + 100
        brick_size = 1.0
        close = base
        opn = close - np.where(np.random.rand(n) > 0.4, brick_size, -brick_size)
        high = np.maximum(close, opn) + np.abs(np.random.randn(n)) * 0.3
        low = np.minimum(close, opn) - np.abs(np.random.randn(n)) * 0.3
        volume = np.random.randint(5, 50, n).astype(float)

        # Use fractional timestamps to simulate Renko
        base_ts = 1704067200.0
        timestamps = [base_ts + i * 0.001 for i in range(n)]
        dates = pd.to_datetime(timestamps, unit="s")

        df = pd.DataFrame({
            "Open": opn, "High": high, "Low": low, "Close": close,
            "Volume": volume,
            "brick_up": close > opn,
        }, index=dates)
        df.index.name = "Date"
        return df

    # Expected columns from the docstring (32 total)
    EXPECTED_COLUMNS = [
        "adx", "plus_di", "minus_di", "rsi",
        "macd", "macd_sig", "macd_hist",
        "ema9", "ema21", "ema50", "ema200",
        "atr", "vol_ema", "vol_ratio",
        "chop", "st_dir",
        "bb_upper", "bb_lower", "bb_mid", "bb_bw", "bb_pct_b",
        "kama", "kama_slope",
        "cmf", "mfi", "obv", "obv_ema",
        "psar_dir", "stoch_k", "stoch_d",
        "sq_momentum", "sq_on",
    ]

    def test_all_columns_added(self, renko_df):
        from renko.indicators import add_renko_indicators
        result = add_renko_indicators(renko_df)
        for col in self.EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_column_count(self, renko_df):
        from renko.indicators import add_renko_indicators
        original_cols = set(renko_df.columns)
        result = add_renko_indicators(renko_df)
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) == len(self.EXPECTED_COLUMNS)

    def test_all_columns_pre_shifted(self, renko_df):
        """Row 0 of every indicator column should be NaN (shifted by 1)."""
        from renko.indicators import add_renko_indicators
        result = add_renko_indicators(renko_df)
        for col in self.EXPECTED_COLUMNS:
            val = result[col].iloc[0]
            assert pd.isna(val), f"Column '{col}' row 0 is {val}, expected NaN (pre-shift)"

    def test_row_count_unchanged(self, renko_df):
        from renko.indicators import add_renko_indicators
        original_len = len(renko_df)
        result = add_renko_indicators(renko_df)
        assert len(result) == original_len

    def test_no_nan_after_warmup(self, renko_df):
        """After sufficient warmup (bar 201+), core indicators should have values."""
        from renko.indicators import add_renko_indicators
        result = add_renko_indicators(renko_df)
        # EMA200 is the longest warmup; check bar 210+
        check_cols = ["adx", "rsi", "ema9", "ema21", "atr", "chop", "st_dir"]
        for col in check_cols:
            val = result[col].iloc[210]
            assert not pd.isna(val), f"Column '{col}' still NaN at bar 210"

    def test_returns_same_dataframe(self, renko_df):
        """Function modifies in-place and returns the same df."""
        from renko.indicators import add_renko_indicators
        result = add_renko_indicators(renko_df)
        assert result is renko_df

    def test_direction_columns_valid(self, renko_df):
        """st_dir and psar_dir should be +1 or -1 (after warmup)."""
        from renko.indicators import add_renko_indicators
        result = add_renko_indicators(renko_df)
        for col in ["st_dir", "psar_dir"]:
            valid = result[col].dropna()
            unique = set(valid.unique())
            assert unique.issubset({1.0, -1.0}), f"{col} has values: {unique}"

    def test_rsi_bounded(self, renko_df):
        """RSI should be 0-100."""
        from renko.indicators import add_renko_indicators
        result = add_renko_indicators(renko_df)
        valid = result["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_adx_bounded(self, renko_df):
        """ADX should be 0-100."""
        from renko.indicators import add_renko_indicators
        result = add_renko_indicators(renko_df)
        valid = result["adx"].dropna()
        assert (valid >= 0).all() and (valid <= 100 + 1e-6).all()
