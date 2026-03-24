"""Unit tests for data loading utilities (engine/data.py)."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from engine.data import (
    _parse_date_range, _fmt_date, load_tv_export,
    _normalize_tf, _parse_symbol, _DATA_DIR,
)


# ── _parse_date_range ────────────────────────────────────────────────────

class TestParseDateRange:
    def test_em_dash_separator(self):
        """Standard TV format with em-dash."""
        start, end = _parse_date_range("Jan 02, 2018 \u2014 Feb 17, 2026")
        assert start == "2018-01-02"
        assert end == "2026-02-17"

    def test_hyphen_fallback(self):
        """Falls back to hyphen when no em-dash present."""
        start, end = _parse_date_range("Jan 02, 2018 - Feb 17, 2026")
        assert start == "2018-01-02"
        assert end == "2026-02-17"

    def test_intraday_timestamps(self):
        """Intraday dates get H:M format."""
        start, end = _parse_date_range("Jan 02, 2018 09:30 \u2014 Feb 17, 2026 16:00")
        assert start == "2018-01-02 09:30"
        assert end == "2026-02-17 16:00"

    def test_daily_no_time_component(self):
        """Daily dates use date-only format (no H:M)."""
        start, end = _parse_date_range("Mar 01, 2020 \u2014 Dec 31, 2025")
        assert ":" not in start
        assert ":" not in end


# ── _fmt_date ────────────────────────────────────────────────────────────

class TestFmtDate:
    def test_date_only(self):
        ts = pd.Timestamp("2024-03-15")
        assert _fmt_date(ts) == "2024-03-15"

    def test_with_time(self):
        ts = pd.Timestamp("2024-03-15 14:30")
        assert _fmt_date(ts) == "2024-03-15 14:30"

    def test_midnight_is_date_only(self):
        ts = pd.Timestamp("2024-03-15 00:00")
        assert _fmt_date(ts) == "2024-03-15"


# ── load_tv_export ───────────────────────────────────────────────────────

class TestLoadTvExport:
    """Tests using a synthetic CSV in a temp directory."""

    @pytest.fixture
    def csv_path(self, tmp_path, monkeypatch):
        """Create a minimal TV-style CSV and point _DATA_DIR at it."""
        import engine.data as data_mod

        # Build a 5-bar CSV (last bar will be dropped → 4 bars returned)
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")
        unix_ts = [int(t.timestamp()) for t in timestamps]
        rows = []
        for i, ts in enumerate(unix_ts):
            rows.append(f"{ts},{100+i},{105+i},{95+i},{102+i}")
        csv_content = "time,open,high,low,close\n" + "\n".join(rows) + "\n"

        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text(csv_content)

        # Monkeypatch _DATA_DIR to point to tmp_path
        monkeypatch.setattr(data_mod, "_DATA_DIR", tmp_path)
        return "test_data.csv"

    def test_loads_correct_bars(self, csv_path):
        df = load_tv_export(csv_path)
        # 5 bars minus 1 dropped = 4
        assert len(df) == 4

    def test_column_names(self, csv_path):
        df = load_tv_export(csv_path)
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns

    def test_drops_last_bar(self, csv_path):
        df = load_tv_export(csv_path)
        # Last bar (Jan 5) should be dropped; last remaining is Jan 4
        assert df.index[-1] == pd.Timestamp("2024-01-04")

    def test_index_is_datetime(self, csv_path):
        df = load_tv_export(csv_path)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_sorted_by_date(self, csv_path):
        df = load_tv_export(csv_path)
        assert df.index.is_monotonic_increasing

    def test_adds_dummy_volume(self, csv_path):
        """CSV without Volume column gets a zero-filled Volume."""
        df = load_tv_export(csv_path)
        assert (df["Volume"] == 0).all()

    def test_file_not_found(self, monkeypatch):
        import engine.data as data_mod
        monkeypatch.setattr(data_mod, "_DATA_DIR", Path("/nonexistent"))
        with pytest.raises(FileNotFoundError, match="TV export not found"):
            load_tv_export("does_not_exist.csv")

    def test_too_few_bars(self, tmp_path, monkeypatch):
        """CSV with only 1 bar should raise ValueError."""
        import engine.data as data_mod
        ts = int(pd.Timestamp("2024-01-01").timestamp())
        csv_content = f"time,open,high,low,close\n{ts},100,105,95,102\n"
        csv_file = tmp_path / "one_bar.csv"
        csv_file.write_text(csv_content)
        monkeypatch.setattr(data_mod, "_DATA_DIR", tmp_path)
        with pytest.raises(ValueError, match="only 1 bar"):
            load_tv_export("one_bar.csv")

    def test_scaled_epoch_auto_correction(self, tmp_path, monkeypatch):
        """Scaled epochs (1970-era) should be auto-upscaled by 1000."""
        import engine.data as data_mod

        # Use small timestamps that parse to 1970s
        # 1704067200 = 2024-01-01, scaled = 1704067
        real_ts = [1704067200 + i * 86400 for i in range(5)]
        scaled_ts = [t // 1000 for t in real_ts]

        rows = []
        for i, ts in enumerate(scaled_ts):
            rows.append(f"{ts},{100+i},{105+i},{95+i},{102+i}")
        csv_content = "time,open,high,low,close\n" + "\n".join(rows) + "\n"
        csv_file = tmp_path / "scaled.csv"
        csv_file.write_text(csv_content)
        monkeypatch.setattr(data_mod, "_DATA_DIR", tmp_path)

        df = load_tv_export("scaled.csv")
        # After upscaling, dates should be in ~2024, not 1970
        # (integer division loses precision, so may be late 2023)
        assert df.index[0].year >= 2023


# ── _normalize_tf ────────────────────────────────────────────────────────

class TestNormalizeTf:
    def test_tv_notation(self):
        assert _normalize_tf("240") == "4h"
        assert _normalize_tf("1D") == "1d"
        assert _normalize_tf("D") == "1d"
        assert _normalize_tf("W") == "1w"

    def test_ccxt_notation_passthrough(self):
        assert _normalize_tf("4h") == "4h"
        assert _normalize_tf("1d") == "1d"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            _normalize_tf("99x")


# ── _parse_symbol ────────────────────────────────────────────────────────

class TestParseSymbol:
    def test_exchange_prefix(self):
        exch, base, quote = _parse_symbol("BINANCE:BTCUSDT")
        assert exch == "binance"
        assert base == "BTC"
        assert quote == "USDT"

    def test_slash_format(self):
        exch, base, quote = _parse_symbol("BTC/USDT")
        assert exch is None
        assert base == "BTC"
        assert quote == "USDT"

    def test_concatenated_pair(self):
        exch, base, quote = _parse_symbol("BTCUSDT")
        assert exch is None
        assert base == "BTC"
        assert quote == "USDT"

    def test_usd_pair(self):
        exch, base, quote = _parse_symbol("BTCUSD")
        assert base == "BTC"
        assert quote == "USD"

    def test_base_only(self):
        exch, base, quote = _parse_symbol("SOL")
        assert exch is None
        assert base == "SOL"
        assert quote is None

    def test_case_insensitive_exchange(self):
        exch, base, quote = _parse_symbol("binance:ethusdt")
        assert exch == "binance"
        assert base == "ETH"
        assert quote == "USDT"
