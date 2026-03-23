"""Unit tests for compute_kpis() — KPI math verification."""

import math
import pandas as pd
import pytest
from dataclasses import dataclass
from engine.engine import compute_kpis, Trade, BacktestConfig


# ── Helpers ──────────────────────────────────────────────────────────────

def _cfg(capital=1000.0):
    return BacktestConfig(initial_capital=capital)


def _trade(entry_date="2024-01-01", entry_price=100.0, entry_qty=1.0,
           direction="long", exit_date="2024-01-02", exit_price=110.0,
           pnl=None, pnl_pct=None,
           entry_commission=0.0, exit_commission=0.0):
    """Build a Trade with sensible defaults. Auto-computes pnl if not given."""
    ed = pd.Timestamp(entry_date)
    xd = pd.Timestamp(exit_date) if exit_date else None
    if pnl is None and exit_price is not None:
        if direction == "long":
            pnl = entry_qty * (exit_price - entry_price) - entry_commission - exit_commission
        else:
            pnl = entry_qty * (entry_price - exit_price) - entry_commission - exit_commission
    if pnl_pct is None and pnl is not None:
        pnl_pct = (pnl / (entry_price * entry_qty)) * 100
    return Trade(
        entry_date=ed, entry_price=entry_price, entry_qty=entry_qty,
        direction=direction, exit_date=xd, exit_price=exit_price,
        pnl=pnl, pnl_pct=pnl_pct,
        entry_commission=entry_commission, exit_commission=exit_commission,
    )


def _equity_df(final_equity):
    """Minimal equity DataFrame."""
    return pd.DataFrame({"equity": [final_equity]})


# ── 1. No trades → error ────────────────────────────────────────────────

class TestNoTrades:
    def test_empty_list(self):
        result = compute_kpis([], _equity_df(1000), _cfg())
        assert result == {"error": "No trades executed"}


# ── 2. Basic known trades ───────────────────────────────────────────────

class TestBasicKPIs:
    """3 trades with known outcomes: 2 winners, 1 loser."""

    @pytest.fixture
    def trades_and_kpis(self):
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0,
                   exit_date="2024-01-02"),  # +10
            _trade(entry_price=100, exit_price=120, entry_qty=1.0,
                   entry_date="2024-01-03", exit_date="2024-01-04"),  # +20
            _trade(entry_price=100, exit_price=90, entry_qty=1.0,
                   entry_date="2024-01-05", exit_date="2024-01-06"),  # -10
        ]
        # final equity = 1000 + 10 + 20 - 10 = 1020
        kpis = compute_kpis(trades, _equity_df(1020), _cfg(1000))
        return kpis

    def test_net_profit(self, trades_and_kpis):
        assert trades_and_kpis["net_profit"] == pytest.approx(20.0)

    def test_net_profit_pct(self, trades_and_kpis):
        assert trades_and_kpis["net_profit_pct"] == pytest.approx(2.0)

    def test_gross_profit(self, trades_and_kpis):
        assert trades_and_kpis["gross_profit"] == pytest.approx(30.0)

    def test_gross_loss(self, trades_and_kpis):
        assert trades_and_kpis["gross_loss"] == pytest.approx(-10.0)

    def test_profit_factor(self, trades_and_kpis):
        # 30 / 10 = 3.0
        assert trades_and_kpis["profit_factor"] == pytest.approx(3.0)

    def test_total_trades(self, trades_and_kpis):
        assert trades_and_kpis["total_trades"] == 3

    def test_win_rate(self, trades_and_kpis):
        # 2/3 = 66.67%
        assert trades_and_kpis["win_rate"] == pytest.approx(200 / 3)

    def test_num_winning_losing(self, trades_and_kpis):
        assert trades_and_kpis["num_winning"] == 2
        assert trades_and_kpis["num_losing"] == 1

    def test_avg_trade(self, trades_and_kpis):
        # 20 / 3
        assert trades_and_kpis["avg_trade"] == pytest.approx(20 / 3)

    def test_avg_winning(self, trades_and_kpis):
        # (10 + 20) / 2 = 15
        assert trades_and_kpis["avg_winning"] == pytest.approx(15.0)

    def test_avg_losing(self, trades_and_kpis):
        # -10 / 1
        assert trades_and_kpis["avg_losing"] == pytest.approx(-10.0)

    def test_avg_win_loss_ratio(self, trades_and_kpis):
        # |15 / -10| = 1.5
        assert trades_and_kpis["avg_win_loss_ratio"] == pytest.approx(1.5)

    def test_largest_winning(self, trades_and_kpis):
        assert trades_and_kpis["largest_winning"] == pytest.approx(20.0)

    def test_largest_losing(self, trades_and_kpis):
        assert trades_and_kpis["largest_losing"] == pytest.approx(-10.0)

    def test_total_pnl_no_open(self, trades_and_kpis):
        # No open trades → total_pnl == net_profit
        assert trades_and_kpis["total_pnl"] == pytest.approx(20.0)

    def test_open_profit_zero(self, trades_and_kpis):
        assert trades_and_kpis["open_profit"] == pytest.approx(0.0)


# ── 3. All winners → profit_factor = inf ────────────────────────────────

class TestAllWinners:
    def test_profit_factor_inf(self):
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0),
            _trade(entry_price=100, exit_price=105, entry_qty=1.0,
                   entry_date="2024-01-03", exit_date="2024-01-04"),
        ]
        kpis = compute_kpis(trades, _equity_df(1015), _cfg(1000))
        assert kpis["profit_factor"] == float("inf")
        assert kpis["gross_loss"] == 0.0
        assert kpis["num_losing"] == 0

    def test_avg_win_loss_ratio_inf(self):
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0),
        ]
        kpis = compute_kpis(trades, _equity_df(1010), _cfg(1000))
        assert kpis["avg_win_loss_ratio"] == float("inf")


# ── 4. All losers ───────────────────────────────────────────────────────

class TestAllLosers:
    def test_all_losers(self):
        trades = [
            _trade(entry_price=100, exit_price=95, entry_qty=1.0),
            _trade(entry_price=100, exit_price=90, entry_qty=1.0,
                   entry_date="2024-01-03", exit_date="2024-01-04"),
        ]
        kpis = compute_kpis(trades, _equity_df(985), _cfg(1000))
        assert kpis["gross_profit"] == 0.0
        assert kpis["profit_factor"] == pytest.approx(0.0)
        assert kpis["win_rate"] == 0.0
        assert kpis["num_winning"] == 0
        assert kpis["largest_winning"] == 0  # default from max(..., default=0)


# ── 5. Consecutive wins / losses ────────────────────────────────────────

class TestConsecutive:
    def test_consecutive_pattern(self):
        # Pattern: W W L L L W → max_consec_wins=2, max_consec_losses=3
        pnls = [10, 5, -3, -7, -2, 8]
        trades = []
        for i, p in enumerate(pnls):
            ep = 100
            xp = ep + p if p > 0 else ep + p
            trades.append(_trade(
                entry_price=ep, exit_price=xp, entry_qty=1.0,
                entry_date=f"2024-01-{2*i+1:02d}",
                exit_date=f"2024-01-{2*i+2:02d}",
            ))
        kpis = compute_kpis(trades, _equity_df(1011), _cfg(1000))
        assert kpis["max_consec_wins"] == 2
        assert kpis["max_consec_losses"] == 3

    def test_single_trade_win(self):
        trades = [_trade(entry_price=100, exit_price=110, entry_qty=1.0)]
        kpis = compute_kpis(trades, _equity_df(1010), _cfg(1000))
        assert kpis["max_consec_wins"] == 1
        assert kpis["max_consec_losses"] == 0

    def test_single_trade_loss(self):
        trades = [_trade(entry_price=100, exit_price=90, entry_qty=1.0)]
        kpis = compute_kpis(trades, _equity_df(990), _cfg(1000))
        assert kpis["max_consec_wins"] == 0
        assert kpis["max_consec_losses"] == 1


# ── 6. Commission handling ──────────────────────────────────────────────

class TestCommissions:
    def test_total_commission(self):
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0,
                   entry_commission=0.10, exit_commission=0.11),
            _trade(entry_price=100, exit_price=95, entry_qty=1.0,
                   entry_date="2024-01-03", exit_date="2024-01-04",
                   entry_commission=0.10, exit_commission=0.095),
        ]
        kpis = compute_kpis(trades, _equity_df(1000), _cfg(1000))
        assert kpis["total_commission"] == pytest.approx(0.10 + 0.11 + 0.10 + 0.095)

    def test_commission_reduces_pnl(self):
        # Trade: buy at 100, sell at 110, qty=1, commissions=0.5 each
        # pnl = 1*(110-100) - 0.5 - 0.5 = 9.0
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0,
                   entry_commission=0.5, exit_commission=0.5),
        ]
        kpis = compute_kpis(trades, _equity_df(1009), _cfg(1000))
        assert kpis["net_profit"] == pytest.approx(9.0)


# ── 7. Open trades ──────────────────────────────────────────────────────

class TestOpenTrades:
    def test_open_trade_not_in_closed_stats(self):
        """Open trade should NOT count in total_trades or win/loss stats."""
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0),  # closed, +10
            _trade(entry_price=100, exit_price=None, exit_date=None,
                   entry_qty=1.0, entry_date="2024-01-03",
                   pnl=None, pnl_pct=None),  # open
        ]
        # Final equity includes unrealized: say +5 unrealized
        kpis = compute_kpis(trades, _equity_df(1015), _cfg(1000))
        assert kpis["total_trades"] == 1  # only closed
        assert kpis["num_winning"] == 1
        assert kpis["num_losing"] == 0
        assert kpis["net_profit"] == pytest.approx(10.0)  # closed only

    def test_open_profit_computed(self):
        """open_profit = final_equity - initial_capital - net_profit."""
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0),  # closed, +10
            _trade(entry_price=100, exit_price=None, exit_date=None,
                   entry_qty=1.0, entry_date="2024-01-03",
                   pnl=None, pnl_pct=None),  # open
        ]
        # final equity = 1000 + 10 (closed) + 5 (unrealized) = 1015
        kpis = compute_kpis(trades, _equity_df(1015), _cfg(1000))
        assert kpis["open_profit"] == pytest.approx(5.0)
        assert kpis["total_pnl"] == pytest.approx(15.0)  # 10 + 5
        assert kpis["total_pnl_pct"] == pytest.approx(1.5)  # 15/1000 * 100


# ── 8. Drawdown passthrough ─────────────────────────────────────────────

class TestDrawdown:
    def test_drawdown_values_passed_through(self):
        """max_drawdown and max_drawdown_pct come from caller args, not computed."""
        trades = [_trade(entry_price=100, exit_price=110, entry_qty=1.0)]
        kpis = compute_kpis(
            trades, _equity_df(1010), _cfg(1000),
            max_intrabar_dd=42.5, max_intrabar_dd_pct=4.25,
        )
        assert kpis["max_drawdown"] == 42.5
        assert kpis["max_drawdown_pct"] == 4.25

    def test_drawdown_defaults_zero(self):
        trades = [_trade(entry_price=100, exit_price=110, entry_qty=1.0)]
        kpis = compute_kpis(trades, _equity_df(1010), _cfg(1000))
        assert kpis["max_drawdown"] == 0.0
        assert kpis["max_drawdown_pct"] == 0.0


# ── 9. Order dates ──────────────────────────────────────────────────────

class TestOrderDates:
    def test_first_and_last_order_dates(self):
        trades = [
            _trade(entry_date="2024-01-01", exit_date="2024-01-05",
                   entry_price=100, exit_price=110, entry_qty=1.0),
            _trade(entry_date="2024-01-10", exit_date="2024-01-15",
                   entry_price=100, exit_price=105, entry_qty=1.0),
        ]
        kpis = compute_kpis(trades, _equity_df(1015), _cfg(1000))
        assert kpis["first_order_date"] == pd.Timestamp("2024-01-01")
        assert kpis["last_order_date"] == pd.Timestamp("2024-01-15")

    def test_last_order_is_latest_entry_or_exit(self):
        """Last order date = max of all entry and exit dates."""
        trades = [
            _trade(entry_date="2024-01-01", exit_date="2024-01-05",
                   entry_price=100, exit_price=110, entry_qty=1.0),
            _trade(entry_date="2024-01-20", exit_date=None,
                   entry_price=100, exit_price=None, entry_qty=1.0,
                   pnl=None, pnl_pct=None),  # open, latest entry
        ]
        kpis = compute_kpis(trades, _equity_df(1010), _cfg(1000))
        # Jan 20 entry > Jan 5 exit
        assert kpis["last_order_date"] == pd.Timestamp("2024-01-20")


# ── 10. Short trades ────────────────────────────────────────────────────

class TestShortTrades:
    def test_short_pnl(self):
        """Short PnL = qty * (entry - exit) - commissions."""
        trades = [
            _trade(entry_price=100, exit_price=90, entry_qty=2.0,
                   direction="short"),  # pnl = 2*(100-90) = 20
        ]
        kpis = compute_kpis(trades, _equity_df(1020), _cfg(1000))
        assert kpis["net_profit"] == pytest.approx(20.0)
        assert kpis["num_winning"] == 1

    def test_short_loss(self):
        trades = [
            _trade(entry_price=100, exit_price=110, entry_qty=1.0,
                   direction="short"),  # pnl = 1*(100-110) = -10
        ]
        kpis = compute_kpis(trades, _equity_df(990), _cfg(1000))
        assert kpis["net_profit"] == pytest.approx(-10.0)
        assert kpis["num_losing"] == 1


# ── 11. Zero PnL trade counts as loss ───────────────────────────────────

class TestZeroPnl:
    def test_zero_pnl_is_losing(self):
        """pnl == 0 → counted as losing (pnl <= 0 branch)."""
        trades = [
            _trade(entry_price=100, exit_price=100, entry_qty=1.0),  # pnl=0
        ]
        kpis = compute_kpis(trades, _equity_df(1000), _cfg(1000))
        assert kpis["num_winning"] == 0
        assert kpis["num_losing"] == 1
        assert kpis["win_rate"] == 0.0
