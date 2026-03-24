"""Integration tests for run_backtest() — long-only engine."""

import numpy as np
import pandas as pd
import pytest
from engine.engine import run_backtest, BacktestConfig


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_df(bars, start="2024-01-01"):
    """Build OHLCV+signal DataFrame from list of dicts.

    Each dict: {O, H, L, C, entry, exit} (entry/exit are bools).
    Optional keys: entry_qty, tp_price, sl_price, tp_offset, sl_offset.
    """
    dates = pd.date_range(start, periods=len(bars), freq="D")
    rows = []
    for b in bars:
        row = {
            "Open": b["O"], "High": b["H"], "Low": b["L"], "Close": b["C"],
            "long_entry": b.get("entry", False),
            "long_exit": b.get("exit", False),
        }
        for opt in ("entry_qty", "tp_price", "sl_price", "tp_offset", "sl_offset"):
            if opt in b:
                row[opt] = b[opt]
        rows.append(row)
    return pd.DataFrame(rows, index=dates)


def _cfg(**kwargs):
    defaults = dict(initial_capital=1000.0, commission_pct=0.0)
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


# ── 1. Simple 1-trade scenario ──────────────────────────────────────────

class TestSimpleTrade:
    """Entry signal on bar N → fills at bar N+1 Open. Exit same pattern."""

    def test_one_trade_pnl(self):
        bars = [
            {"O": 100, "H": 105, "L": 95,  "C": 100, "entry": True},   # signal
            {"O": 100, "H": 110, "L": 98,  "C": 105},                   # fill entry at 100
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},     # signal exit
            {"O": 110, "H": 115, "L": 108, "C": 112},                   # fill exit at 110
        ]
        result = run_backtest(_make_df(bars), _cfg())
        assert result["total_trades"] == 1
        trades = result["trades"]
        t = trades[0]
        # Entry at bar 1 Open=100, exit at bar 3 Open=110
        assert t.entry_price == 100.0
        assert t.exit_price == 110.0
        # qty = 1000/100 = 10, pnl = 10 * (110-100) = 100
        assert t.pnl == pytest.approx(100.0)

    def test_entry_fills_next_bar_open(self):
        bars = [
            {"O": 50, "H": 55, "L": 48, "C": 52, "entry": True},
            {"O": 53, "H": 58, "L": 51, "C": 55},  # fills at 53
            {"O": 55, "H": 60, "L": 54, "C": 58, "exit": True},
            {"O": 58, "H": 62, "L": 56, "C": 60},
        ]
        result = run_backtest(_make_df(bars), _cfg())
        assert result["trades"][0].entry_price == 53.0

    def test_exit_fills_next_bar_open(self):
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105},
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},
            {"O": 112, "H": 115, "L": 108, "C": 112},
        ]
        result = run_backtest(_make_df(bars), _cfg())
        assert result["trades"][0].exit_price == 112.0


# ── 2. PnL with commissions ─────────────────────────────────────────────

class TestCommissions:
    def test_pnl_with_commission(self):
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105},
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},
            {"O": 110, "H": 115, "L": 108, "C": 112},
        ]
        # 0.1% commission
        result = run_backtest(_make_df(bars), _cfg(commission_pct=0.1))
        t = result["trades"][0]
        comm_rate = 0.001
        # trade_value = 1000 / (1+0.001) ≈ 999.001
        trade_value = 1000.0 / (1 + comm_rate)
        qty = trade_value / 100.0
        entry_comm = trade_value * comm_rate
        exit_value = qty * 110.0
        exit_comm = exit_value * comm_rate
        expected_pnl = qty * (110 - 100) - entry_comm - exit_comm
        assert t.pnl == pytest.approx(expected_pnl, rel=1e-6)


# ── 3. process_orders_on_close ───────────────────────────────────────────

class TestProcessOnClose:
    def test_fills_at_close(self):
        """process_orders_on_close=True → entry fills at same bar's Close."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 102, "entry": True},
            {"O": 103, "H": 110, "L": 101, "C": 108, "exit": True},
            {"O": 108, "H": 112, "L": 106, "C": 110},
        ]
        result = run_backtest(_make_df(bars), _cfg(process_orders_on_close=True))
        t = result["trades"][0]
        # Entry fills at bar 0 Close=102, exit fills at bar 1 Close=108
        assert t.entry_price == 102.0
        assert t.exit_price == 108.0


# ── 4. Position sizing ──────────────────────────────────────────────────

class TestPositionSizing:
    def test_percent_of_equity_100(self):
        """Default 100% equity → full capital invested."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105},
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},
            {"O": 110, "H": 115, "L": 108, "C": 112},
        ]
        result = run_backtest(_make_df(bars), _cfg())
        t = result["trades"][0]
        # qty = 1000 / 100 = 10
        assert t.entry_qty == pytest.approx(10.0)

    def test_cash_sizing(self):
        """cash=500 → qty = 500 / close_at_signal_bar."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},  # signal, close=100
            {"O": 100, "H": 110, "L": 98, "C": 105},
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},
            {"O": 110, "H": 115, "L": 108, "C": 112},
        ]
        result = run_backtest(_make_df(bars), _cfg(qty_type="cash", qty_value=500))
        t = result["trades"][0]
        # qty = 500 / 100 (close at signal bar) = 5
        assert t.entry_qty == pytest.approx(5.0)

    def test_fixed_sizing(self):
        """fixed=2.5 → qty always 2.5."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105},
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},
            {"O": 110, "H": 115, "L": 108, "C": 112},
        ]
        result = run_backtest(_make_df(bars), _cfg(qty_type="fixed", qty_value=2.5))
        t = result["trades"][0]
        assert t.entry_qty == pytest.approx(2.5)


# ── 5. Date filtering ───────────────────────────────────────────────────

class TestDateFiltering:
    def test_trade_outside_range_ignored(self):
        """Signals outside start/end date should not generate trades."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},  # Jan 1
            {"O": 100, "H": 110, "L": 98, "C": 105},                  # Jan 2
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},   # Jan 3
            {"O": 110, "H": 115, "L": 108, "C": 112},                  # Jan 4
        ]
        # Window starts Jan 5 → no signals in range
        result = run_backtest(
            _make_df(bars),
            _cfg(start_date="2024-01-05", end_date="2024-01-10"),
        )
        assert result["error"] == "No trades executed"


# ── 6. TP/SL interaction ────────────────────────────────────────────────

class TestTPSL:
    def test_tp_exit(self):
        """TP at 5% → entry at 100, TP level=105, hit on bar with High >= 105."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 102, "L": 98, "C": 101},   # fill entry, no TP hit
            {"O": 101, "H": 106, "L": 99, "C": 104},   # TP hit (H=106 >= 105)
        ]
        result = run_backtest(_make_df(bars), _cfg(take_profit_pct=5.0))
        t = result["trades"][0]
        assert t.exit_price == pytest.approx(105.0)

    def test_sl_exit(self):
        """SL at 3% → entry at 100, SL level=97, hit on bar with Low <= 97."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 102, "L": 98, "C": 101},   # fill entry, no SL hit
            {"O": 101, "H": 103, "L": 96, "C": 98},     # SL hit (L=96 <= 97)
        ]
        result = run_backtest(_make_df(bars), _cfg(stop_loss_pct=3.0))
        t = result["trades"][0]
        assert t.exit_price == pytest.approx(97.0)

    def test_tpsl_skipped_on_entry_bar(self):
        """TP/SL should NOT trigger on the same bar as entry fill.

        Entry fills on bar 1 (entry_bar_idx=1). TP check requires i > entry_bar_idx,
        so bar 1 is skipped. Bar 2 is the first bar where TP can fire.
        """
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            # Bar 1: entry fills at Open=100. High=110 would hit 5% TP=105
            # but i == entry_bar_idx → skipped.
            {"O": 100, "H": 110, "L": 90, "C": 105},
            # Bar 2: Open=103 (< TP=105), High=104 (< TP=105) → no TP hit, exit signal fires
            {"O": 103, "H": 104, "L": 100, "C": 102, "exit": True},
            {"O": 102, "H": 106, "L": 100, "C": 104},
        ]
        result = run_backtest(_make_df(bars), _cfg(take_profit_pct=5.0))
        t = result["trades"][0]
        # Exited via signal at bar 3 Open=102, NOT via TP on bar 1
        assert t.exit_price == 102.0


# ── 7. Pyramiding ───────────────────────────────────────────────────────

class TestPyramiding:
    def test_pyramiding_multiple_entries(self):
        """pyramiding=2 allows two sub-positions."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},   # signal 1
            {"O": 100, "H": 110, "L": 98, "C": 105, "entry": True},   # fill 1, signal 2
            {"O": 105, "H": 112, "L": 103, "C": 110},                  # fill 2
            {"O": 110, "H": 115, "L": 108, "C": 112, "exit": True},   # signal exit
            {"O": 112, "H": 118, "L": 110, "C": 115},                  # fill exit
        ]
        result = run_backtest(_make_df(bars), _cfg(pyramiding=2))
        closed = [t for t in result["trades"] if t.exit_date is not None]
        assert len(closed) == 2

    def test_pyramiding_1_blocks_second_entry(self):
        """pyramiding=1 (default) blocks second entry signal."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105, "entry": True},
            {"O": 105, "H": 112, "L": 103, "C": 110},
            {"O": 110, "H": 115, "L": 108, "C": 112, "exit": True},
            {"O": 112, "H": 118, "L": 110, "C": 115},
        ]
        result = run_backtest(_make_df(bars), _cfg(pyramiding=1))
        closed = [t for t in result["trades"] if t.exit_date is not None]
        assert len(closed) == 1

    def test_exit_closes_all_pyramids(self):
        """long_exit closes ALL open sub-positions."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105, "entry": True},
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},
            {"O": 110, "H": 115, "L": 108, "C": 112},
        ]
        result = run_backtest(_make_df(bars), _cfg(pyramiding=3))
        closed = [t for t in result["trades"] if t.exit_date is not None]
        # Both should be closed
        assert len(closed) == 2
        assert all(t.exit_price == 110.0 for t in closed)


# ── 8. Entry and exit on same bar ────────────────────────────────────────

class TestSameBarSignals:
    def test_exit_then_reentry(self):
        """Exit on one bar, new entry on next bar — two sequential trades."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105},                   # fill entry 1
            {"O": 105, "H": 112, "L": 103, "C": 110, "exit": True},    # signal exit
            {"O": 110, "H": 115, "L": 108, "C": 112, "entry": True},   # fill exit, signal entry 2
            {"O": 112, "H": 118, "L": 110, "C": 115},                   # fill entry 2
            {"O": 115, "H": 120, "L": 113, "C": 118, "exit": True},
            {"O": 118, "H": 122, "L": 116, "C": 120},                   # fill exit 2
        ]
        result = run_backtest(_make_df(bars), _cfg())
        closed = [t for t in result["trades"] if t.exit_date is not None]
        assert len(closed) == 2
        assert closed[0].entry_price == 100.0
        assert closed[0].exit_price == 110.0
        assert closed[1].entry_price == 112.0
        assert closed[1].exit_price == 118.0


# ── 9. Missing columns ──────────────────────────────────────────────────

class TestValidation:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Open": [1], "Close": [1]},
                          index=pd.date_range("2024-01-01", periods=1))
        with pytest.raises(ValueError, match="missing required columns"):
            run_backtest(df, _cfg())


# ── 10. Open trade at end of data ────────────────────────────────────────

class TestOpenTradeAtEnd:
    def test_open_trade_recorded(self):
        """Position still open at end of data is recorded with no exit."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "entry": True},
            {"O": 100, "H": 110, "L": 98, "C": 105},  # fill, no exit signal
        ]
        result = run_backtest(_make_df(bars), _cfg())
        open_trades = [t for t in result["trades"] if t.exit_date is None]
        assert len(open_trades) == 1
        assert open_trades[0].entry_price == 100.0
