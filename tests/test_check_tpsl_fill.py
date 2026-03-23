"""Unit tests for _check_tpsl_fill() — the TP/SL fill-price resolver."""

import pytest
from engine.engine import _check_tpsl_fill


# ── 1. Gap-through at Open ──────────────────────────────────────────────

class TestGapThrough:
    """Bar opens past TP or SL → fill at Open price."""

    def test_long_tp_gap_through(self):
        # TP level = 105, bar opens at 108 → fill at Open
        result = _check_tpsl_fill(
            bar_open=108, bar_high=110, bar_low=106,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (108.0, "tp")

    def test_long_tp_gap_exact(self):
        # TP level = 105, bar opens exactly at 105
        result = _check_tpsl_fill(
            bar_open=105, bar_high=110, bar_low=95,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (105.0, "tp")

    def test_long_sl_gap_through(self):
        # SL level = 97, bar opens at 95 → fill at Open
        result = _check_tpsl_fill(
            bar_open=95, bar_high=96, bar_low=90,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (95.0, "sl")

    def test_long_sl_gap_exact(self):
        # SL level = 97, bar opens exactly at 97
        result = _check_tpsl_fill(
            bar_open=97, bar_high=99, bar_low=95,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (97.0, "sl")

    def test_short_tp_gap_through(self):
        # Short TP level = 95, bar opens at 92 → fill at Open
        result = _check_tpsl_fill(
            bar_open=92, bar_high=94, bar_low=90,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (92.0, "tp")

    def test_short_tp_gap_exact(self):
        # Short TP level = 95, bar opens exactly at 95
        result = _check_tpsl_fill(
            bar_open=95, bar_high=98, bar_low=90,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (95.0, "tp")

    def test_short_sl_gap_through(self):
        # Short SL level = 103, bar opens at 106 → fill at Open
        result = _check_tpsl_fill(
            bar_open=106, bar_high=110, bar_low=105,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (106.0, "sl")

    def test_short_sl_gap_exact(self):
        # Short SL level = 103, bar opens exactly at 103
        result = _check_tpsl_fill(
            bar_open=103, bar_high=106, bar_low=101,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (103.0, "sl")


# ── 2. Single intrabar TP hit ───────────────────────────────────────────

class TestSingleTP:
    """Only TP touched intrabar, SL not reached."""

    def test_long_tp_hit(self):
        # TP=105, SL=97. High=106 hits TP, Low=98 misses SL.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=106, bar_low=98,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (105.0, "tp")

    def test_long_tp_exact_touch(self):
        # High exactly at TP level
        result = _check_tpsl_fill(
            bar_open=100, bar_high=105, bar_low=98,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (105.0, "tp")

    def test_short_tp_hit(self):
        # Short TP=95, SL=103. Low=94 hits TP, High=102 misses SL.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=102, bar_low=94,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (95.0, "tp")

    def test_short_tp_exact_touch(self):
        # Low exactly at short TP level
        result = _check_tpsl_fill(
            bar_open=100, bar_high=102, bar_low=95,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (95.0, "tp")


# ── 3. Single intrabar SL hit ───────────────────────────────────────────

class TestSingleSL:
    """Only SL touched intrabar, TP not reached."""

    def test_long_sl_hit(self):
        # TP=105, SL=97. High=104 misses TP, Low=96 hits SL.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=104, bar_low=96,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (97.0, "sl")

    def test_long_sl_exact_touch(self):
        # Low exactly at SL level
        result = _check_tpsl_fill(
            bar_open=100, bar_high=104, bar_low=97,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (97.0, "sl")

    def test_short_sl_hit(self):
        # Short SL=103. High=104 hits SL, Low=96 doesn't reach TP=95.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=104, bar_low=96,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (103.0, "sl")

    def test_short_sl_exact_touch(self):
        # High exactly at short SL level
        result = _check_tpsl_fill(
            bar_open=100, bar_high=103, bar_low=96,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (103.0, "sl")


# ── 4. Both hit — TV heuristic ──────────────────────────────────────────

class TestBothHitHeuristic:
    """TP and SL both touched intrabar. Open proximity to favourable
    extreme decides which was hit first (TradingView behaviour)."""

    def test_long_open_closer_to_high_tp_wins(self):
        # TP=105, SL=97. Bar: O=103 H=106 L=96.
        # high-open=3, open-low=7 → 3 <= 7 → TP wins.
        result = _check_tpsl_fill(
            bar_open=103, bar_high=106, bar_low=96,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (105.0, "tp")

    def test_long_open_closer_to_low_sl_wins(self):
        # TP=105, SL=97. Bar: O=98 H=106 L=96.
        # high-open=8, open-low=2 → 8 > 2 → SL wins.
        result = _check_tpsl_fill(
            bar_open=98, bar_high=106, bar_low=96,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (97.0, "sl")

    def test_long_open_equidistant_tp_wins(self):
        # Equal distance → TP wins (<=).
        # O=101, H=106, L=96. high-open=5, open-low=5.
        result = _check_tpsl_fill(
            bar_open=101, bar_high=106, bar_low=96,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (105.0, "tp")

    def test_short_open_closer_to_low_tp_wins(self):
        # Short: TP=95, SL=103. Bar: O=97 H=104 L=94.
        # open-low=3, high-open=7 → 3 <= 7 → TP wins.
        result = _check_tpsl_fill(
            bar_open=97, bar_high=104, bar_low=94,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (95.0, "tp")

    def test_short_open_closer_to_high_sl_wins(self):
        # Short: TP=95, SL=103. Bar: O=102 H=104 L=94.
        # open-low=8, high-open=2 → 8 > 2 → SL wins.
        result = _check_tpsl_fill(
            bar_open=102, bar_high=104, bar_low=94,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (103.0, "sl")

    def test_short_open_equidistant_tp_wins(self):
        # Equal distance → TP wins (<=).
        # O=99, H=104, L=94. open-low=5, high-open=5.
        result = _check_tpsl_fill(
            bar_open=99, bar_high=104, bar_low=94,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (95.0, "tp")


# ── 5. Price level priority: absolute > offset > percentage ─────────────

class TestPricePriority:
    """tp_price/sl_price > tp_offset/sl_offset > tp_pct/sl_pct."""

    def test_absolute_overrides_offset_and_pct(self):
        # tp_price=106 should win over tp_offset=2 and tp_pct=5
        result = _check_tpsl_fill(
            bar_open=100, bar_high=107, bar_low=98,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
            tp_price=106, sl_price=0,
            tp_offset=2, sl_offset=0,
        )
        assert result == (106.0, "tp")

    def test_offset_overrides_pct(self):
        # tp_offset=3 → tp_level=103 (long: entry+offset).
        # tp_pct=5 would give 105, but offset wins.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=104, bar_low=98,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
            tp_offset=3, sl_offset=0,
        )
        assert result == (103.0, "tp")

    def test_sl_absolute_overrides_offset_and_pct(self):
        # sl_price=98 should win over sl_offset and sl_pct
        result = _check_tpsl_fill(
            bar_open=100, bar_high=104, bar_low=97,
            entry_price=100, position_side="long",
            tp_pct=50.0, sl_pct=3.0,
            sl_price=98, sl_offset=1,
        )
        assert result == (98.0, "sl")

    def test_sl_offset_overrides_pct(self):
        # sl_offset=1 → sl_level=99 (long: entry-offset).
        # sl_pct=3 would give 97, but offset wins.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=104, bar_low=98,
            entry_price=100, position_side="long",
            tp_pct=50.0, sl_pct=3.0,
            sl_offset=1,
        )
        assert result == (99.0, "sl")

    def test_short_offset_direction(self):
        # Short: tp_offset=3 → tp_level = entry - offset = 97
        result = _check_tpsl_fill(
            bar_open=100, bar_high=102, bar_low=96,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
            tp_offset=3, sl_offset=0,
        )
        assert result == (97.0, "tp")

    def test_short_sl_offset_direction(self):
        # Short: sl_offset=2 → sl_level = entry + offset = 102
        result = _check_tpsl_fill(
            bar_open=100, bar_high=103, bar_low=96,
            entry_price=100, position_side="short",
            tp_pct=50.0, sl_pct=3.0,
            sl_offset=2,
        )
        assert result == (102.0, "sl")


# ── 6. Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Disabled levels, no hit, boundary conditions."""

    def test_no_tp_no_sl_returns_no_fill(self):
        result = _check_tpsl_fill(
            bar_open=100, bar_high=110, bar_low=90,
            entry_price=100, position_side="long",
            tp_pct=0.0, sl_pct=0.0,
        )
        assert result == (None, "")

    def test_only_tp_set_no_sl(self):
        result = _check_tpsl_fill(
            bar_open=100, bar_high=106, bar_low=90,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=0.0,
        )
        assert result == (105.0, "tp")

    def test_only_sl_set_no_tp(self):
        result = _check_tpsl_fill(
            bar_open=100, bar_high=110, bar_low=96,
            entry_price=100, position_side="long",
            tp_pct=0.0, sl_pct=3.0,
        )
        assert result == (97.0, "sl")

    def test_no_hit_returns_empty(self):
        # Both set but neither touched.
        # TP=105, SL=97. High=104, Low=98 — misses both.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=104, bar_low=98,
            entry_price=100, position_side="long",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (None, "")

    def test_short_no_hit_returns_empty(self):
        # Short: TP=95, SL=103. Low=96, High=102 — misses both.
        result = _check_tpsl_fill(
            bar_open=100, bar_high=102, bar_low=96,
            entry_price=100, position_side="short",
            tp_pct=5.0, sl_pct=3.0,
        )
        assert result == (None, "")
