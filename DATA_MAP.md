# Data Map

Quick reference for where all results, data, and documentation live in this repo.

## Results Hierarchy

```
Raw sweep JSONs (ai_context/)
  → MD summaries (RENKO_LEADERBOARD.md, RESEARCH_LOG.md)
    → Live portfolio (ai_context/live_portfolio.json + MEMORY.md table)
```

## File Locations

### Live Portfolio (production — 5 strategies as of 2026-03-20)
| File | Contents |
|------|----------|
| `ai_context/live_portfolio.json` | Live strategy configs, OOS metrics, Pine file paths |
| `ai_context/portfolio_analysis.json` | Correlation matrix, optimal subsets, capital allocation |
| `MEMORY.md` | Live portfolio summary table (manually synced) |

### Research Summaries
| File | Contents |
|------|----------|
| `RENKO_LEADERBOARD.md` | Top 33 qualified Renko strategies (IS period, 60+ trades) |
| `RESEARCH_LOG.md` | R001-R012 sweep results, global top 10, per-instrument bests |
| `strategy_details.md` | EURUSD research archive (phases 1-2) |
| `phase_details.md` | Phases 3-7 archive (moved from MEMORY.md) |

### Raw Sweep Data (`ai_context/`)
| Pattern | Instruments | Count |
|---------|-------------|-------|
| `phase2_results.json` … `phase11_results.json` | EURUSD, GBPJPY, EURAUD | 10 files |
| `btc_phase1_results.json` … `btc_phase11_*.json` | BTCUSD | 11 files |
| `ea*_results.json` (IS/OOS pairs) | EURAUD | ~20 files |
| `gj010_*.json`, `r012_*.json` | GBPJPY, EURUSD | 4 files |
| `bc_sweep_*.json` | Multi-instrument brick-count | 2 files |
| `mym_sweep_*.json` | MYM micro futures | 3 files |

### Sweep Runner Scripts (`ai_context/`)
| Script | Purpose |
|--------|---------|
| `big_batch_runner.py` | Master batch runner for large-scale sweeps |
| `phase2_backtest_sweep.py` … `phase5_backtest_sweep.py` | Phase-specific sweeps |
| `bc_sweep_analysis.py` | Brick-count sweep analysis |
| `ea002_decay_analysis.py`, `ea005_ext_sweep.py`, `ea011v2_sweep.py` | Strategy-specific sweeps |

### TradingView Validation (`tvresults/`)
50 CSV files — trade-by-trade exports from TradingView OOS validation.

| Instrument | Files | Date Range |
|------------|------:|------------|
| EURUSD | 29 | 2026-03-02 to 2026-03-20 |
| GBPJPY | 10 | 2026-03-11 to 2026-03-20 |
| BTCUSD | 4 | 2026-03-21 |
| EURAUD | 4 | 2026-03-17 to 2026-03-19 |
| GBPUSD | 1 | 2026-03-20 |
| USDJPY | 1 | 2026-03-20 |
| MYM | 1 | 2026-03-20 |

Naming convention: `[Strategy]_[Broker]_[Instrument]_[Date].csv`

### Market Data (`data/`)
- `OANDA_*renko*.csv` — Renko brick data (EURUSD, GBPJPY, GBPUSD, USDJPY, EURAUD, BTCUSD; various brick sizes)
- `OANDA_*1D*.csv` — Daily OHLC data
- `SYNTH_*.csv` — Synthetic data for engine tests

### Strategy Code
| Path | Contents |
|------|----------|
| `renko/strategies/` | 50+ production Renko strategies (BTC, GJ, R, EA, UJ, GU series) |
| `strategies/` | OHLC-based strategies (Gaussian, KAMA, Donchian, etc.) |
| `strategies/research/` | Research strategies |
| `strategies/research/renko/` | R001-R012 Renko research series |

### Documentation
| File | Purpose |
|------|---------|
| `MEMORY.md` | **READ FIRST** — rules, conventions, Pine sanitization checklist |
| `BACKTESTING.md` | Engine mechanics, fill logic, TV quirks |
| `PROJECT_MAP.md` | Directory structure & core components |
| `CLAUDE.md` | AI assistant entry point (points to MEMORY.md) |
| `DATA_MAP.md` | This file |
| `CODEX_RESEARCH_HANDOFF.md` | Autonomous research loop instructions |
| `CODEX_OOS_HANDOFF.md` | OOS testing handoff context |
| `OPTIMIZATION_PLAN_MTF_KAMA.md` | MTF KAMA optimization plan |

## Adding New Results

Follow this checklist so nothing gets lost:

1. **Raw sweep output** → save JSON to `ai_context/` with descriptive name (e.g., `phase12_results.json`, `btc_phase12_results.json`)
2. **Summarize best results** → update `RENKO_LEADERBOARD.md` (if Renko) or `RESEARCH_LOG.md`
3. **TV validation exports** → save CSV to `tvresults/` using naming convention above
4. **Promoted to live** → update `ai_context/live_portfolio.json` AND `MEMORY.md` portfolio table
5. **New sweep scripts** → save to `ai_context/` alongside their result JSONs
6. **New strategy code** → save to `renko/strategies/` (Renko) or `strategies/` (OHLC)
7. **Commit and push** — all `.md` and `ai_context/` files are auto-tracked by git (no gitignore exceptions needed)
