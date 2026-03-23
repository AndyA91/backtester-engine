"""Integration tests for run_backtest_long_short() — long+short engine."""

import numpy as np
import pandas as pd
import pytest
from engine.engine import run_backtest_long_short, BacktestConfig


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_df(bars, start="2024-01-01"):
    """Build OHLCV+signal DataFrame from list of dicts.

    Each dict: {O, H, L, C, le, lx, se, sx}
    (long_entry, long_exit, short_entry, short_exit).
    """
    dates = pd.date_range(start, periods=len(bars), freq="D")
    rows = []
    for b in bars:
        row = {
            "Open": b["O"], "High": b["H"], "Low": b["L"], "Close": b["C"],
            "long_entry": b.get("le", False),
            "long_exit": b.get("lx", False),
            "short_entry": b.get("se", False),
            "short_exit": b.get("sx", False),
        }
        rows.append(row)
    return pd.DataFrame(rows, index=dates)


def _cfg(**kwargs):
    defaults = dict(initial_capital=1000.0, commission_pct=0.0)
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


# ── 1. Long entry → long exit → short entry → short exit ────────────────

class TestBasicSequence:
    def test_long_then_short(self):
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "le": True},       # long signal
            {"O": 100, "H": 110, "L": 98, "C": 105},                    # fill long at 100
            {"O": 105, "H": 112, "L": 103, "C": 110, "lx": True},      # long exit signal
            {"O": 110, "H": 115, "L": 108, "C": 112, "se": True},      # fill exit at 110, short signal
            {"O": 112, "H": 114, "L": 106, "C": 108},                   # fill short at 112
            {"O": 108, "H": 110, "L": 100, "C": 102, "sx": True},      # short exit signal
            {"O": 102, "H": 105, "L": 100, "C": 103},                   # fill short exit at 102
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg())
        closed = [t for t in result["trades"] if t.exit_date is not None]
        assert len(closed) == 2

        # Long: entry 100, exit 110 → pnl = qty*(110-100)
        long_t = closed[0]
        assert long_t.direction == "long"
        assert long_t.entry_price == 100.0
        assert long_t.exit_price == 110.0
        assert long_t.pnl == pytest.approx(long_t.entry_qty * 10)

        # Short: entry 112, exit 102 → pnl = qty*(112-102)
        short_t = closed[1]
        assert short_t.direction == "short"
        assert short_t.entry_price == 112.0
        assert short_t.exit_price == 102.0
        assert short_t.pnl == pytest.approx(short_t.entry_qty * 10)


# ── 2. Reversal: long → short_entry closes long, opens short ────────────

class TestReversal:
    def test_short_entry_reverses_long(self):
        """long_exit + short_entry on same bar → closes long, opens short at next Open."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "le": True},
            {"O": 100, "H": 110, "L": 98, "C": 105},                          # fill long at 100
            {"O": 105, "H": 112, "L": 103, "C": 110, "lx": True, "se": True}, # reversal signal
            {"O": 110, "H": 115, "L": 108, "C": 112},                          # close long at 110, open short at 110
            {"O": 112, "H": 114, "L": 100, "C": 102, "sx": True},
            {"O": 102, "H": 105, "L": 100, "C": 103},
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg())
        closed = [t for t in result["trades"] if t.exit_date is not None]
        assert len(closed) == 2

        # Long closed via reversal at 110
        assert closed[0].direction == "long"
        assert closed[0].exit_price == 110.0

        # Short opened at same bar's Open=110
        assert closed[1].direction == "short"
        assert closed[1].entry_price == 110.0

    def test_long_entry_reverses_short(self):
        """short_exit + long_entry on same bar → closes short, opens long."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "se": True},
            {"O": 100, "H": 102, "L": 90, "C": 92},                           # fill short at 100
            {"O": 92,  "H": 95,  "L": 88, "C": 90, "sx": True, "le": True},  # reversal signal
            {"O": 90,  "H": 98,  "L": 88, "C": 95},                           # close short at 90, open long at 90
            {"O": 95,  "H": 100, "L": 93, "C": 98, "lx": True},
            {"O": 98,  "H": 102, "L": 96, "C": 100},
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg())
        closed = [t for t in result["trades"] if t.exit_date is not None]
        assert len(closed) == 2

        # Short: entry 100, exit 90 → profit
        assert closed[0].direction == "short"
        assert closed[0].entry_price == 100.0
        assert closed[0].exit_price == 90.0
        assert closed[0].pnl > 0

        # Long: entry 90
        assert closed[1].direction == "long"
        assert closed[1].entry_price == 90.0


# ── 3. Short PnL formula ────────────────────────────────────────────────

class TestShortPnL:
    def test_short_profit(self):
        """Short PnL = qty * (entry - exit) - commissions."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "se": True},
            {"O": 100, "H": 102, "L": 90, "C": 92},                    # fill short at 100
            {"O": 92,  "H": 95,  "L": 88, "C": 90, "sx": True},
            {"O": 90,  "H": 95,  "L": 88, "C": 92},                    # fill exit at 90
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg())
        t = result["trades"][0]
        assert t.direction == "short"
        # qty * (100 - 90) = qty * 10
        assert t.pnl == pytest.approx(t.entry_qty * 10)

    def test_short_loss(self):
        """Short loss when price goes up."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "se": True},
            {"O": 100, "H": 112, "L": 98, "C": 110},                   # fill short at 100
            {"O": 110, "H": 115, "L": 108, "C": 112, "sx": True},
            {"O": 112, "H": 115, "L": 110, "C": 113},                  # fill exit at 112
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg())
        t = result["trades"][0]
        # qty * (100 - 112) = negative
        assert t.pnl == pytest.approx(t.entry_qty * (100 - 112))
        assert t.pnl < 0

    def test_short_pnl_with_commission(self):
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "se": True},
            {"O": 100, "H": 102, "L": 90, "C": 92},
            {"O": 92,  "H": 95,  "L": 88, "C": 90, "sx": True},
            {"O": 90,  "H": 95,  "L": 88, "C": 92},
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg(commission_pct=0.1))
        t = result["trades"][0]
        comm_rate = 0.001
        # PnL = qty * (entry - exit) - entry_comm - exit_comm
        gross = t.entry_qty * (100 - 90)
        expected = gross - t.entry_commission - t.exit_commission
        assert t.pnl == pytest.approx(expected, rel=1e-6)


# ── 4. Missing columns validation ───────────────────────────────────────

class TestValidation:
    def test_missing_short_columns_raises(self):
        df = pd.DataFrame({
            "Open": [1], "High": [2], "Low": [0.5], "Close": [1],
            "long_entry": [False], "long_exit": [False],
        }, index=pd.date_range("2024-01-01", periods=1))
        with pytest.raises(ValueError, match="missing required columns"):
            run_backtest_long_short(df, _cfg())


# ── 5. No simultaneous long+short ────────────────────────────────────────

class TestNoSimultaneous:
    def test_reversal_closes_before_opening(self):
        """Engine never holds long+short simultaneously.
        long_exit + short_entry on same bar → implicit reversal."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "le": True},
            {"O": 100, "H": 110, "L": 98, "C": 105, "lx": True, "se": True},  # reversal
            {"O": 105, "H": 112, "L": 103, "C": 110},                          # close long, open short
            {"O": 110, "H": 114, "L": 100, "C": 102, "sx": True},
            {"O": 102, "H": 105, "L": 100, "C": 103},
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg())
        closed = [t for t in result["trades"] if t.exit_date is not None]
        # Should have long (reversed out) + short (exited)
        assert len(closed) == 2
        assert closed[0].direction == "long"
        assert closed[1].direction == "short"


# ── 6. process_orders_on_close with long-short ──────────────────────────

class TestProcessOnCloseLS:
    def test_fills_at_close(self):
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 102, "le": True},
            {"O": 103, "H": 110, "L": 101, "C": 108, "lx": True},
            {"O": 108, "H": 112, "L": 106, "C": 110},
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg(process_orders_on_close=True))
        t = result["trades"][0]
        assert t.entry_price == 102.0   # bar 0 Close
        assert t.exit_price == 108.0    # bar 1 Close


# ── 7. TP/SL with shorts ────────────────────────────────────────────────

class TestTPSLShort:
    def test_short_tp_hit(self):
        """Short TP at 5% → entry at 100, TP level=95. Low <= 95 triggers."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "se": True},
            {"O": 100, "H": 102, "L": 98, "C": 99},     # fill short at 100
            {"O": 99,  "H": 101, "L": 94, "C": 96},     # TP hit (Low=94 <= 95)
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg(take_profit_pct=5.0))
        t = [t for t in result["trades"] if t.exit_date is not None][0]
        assert t.exit_price == pytest.approx(95.0)

    def test_short_sl_hit(self):
        """Short SL at 3% → entry at 100, SL level=103. High >= 103 triggers."""
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "se": True},
            {"O": 100, "H": 102, "L": 98, "C": 99},     # fill short at 100
            {"O": 99,  "H": 104, "L": 97, "C": 101},    # SL hit (High=104 >= 103)
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg(stop_loss_pct=3.0))
        t = [t for t in result["trades"] if t.exit_date is not None][0]
        assert t.exit_price == pytest.approx(103.0)


# ── 8. Open trade at end of data ────────────────────────────────────────

class TestOpenTradeAtEnd:
    def test_open_short_recorded(self):
        bars = [
            {"O": 100, "H": 105, "L": 95, "C": 100, "se": True},
            {"O": 100, "H": 102, "L": 90, "C": 92},  # fill short, no exit
        ]
        result = run_backtest_long_short(_make_df(bars), _cfg())
        open_trades = [t for t in result["trades"] if t.exit_date is None]
        assert len(open_trades) == 1
        assert open_trades[0].direction == "short"
