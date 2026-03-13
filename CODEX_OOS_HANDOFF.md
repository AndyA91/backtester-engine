# OOS Validation Handoff

## Context

Two strategies are currently under validation on HistData EURUSD 5m/1h/1d data.
This handoff now uses the same split protocol as the research loop:

- IS (already used for development/selection): **2024-01-01 to 2025-09-30**
- OOS (sealed holdout): **2025-10-01 to 2026-02-28**

Data files:
- `data/HISTDATA_EURUSD_5m.csv` - 161k bars
- `data/HISTDATA_EURUSD_1h.csv` - 13.4k bars
- `data/HISTDATA_EURUSD_1d.csv` - 630 bars (22:00 UTC session boundary)

## Validated IS Winners

### Strategy 1: MTF KAMA Dual v4
- **File**: `strategies/mtf_kama_dual_v4.py`
- **Winner params**: `adx_threshold=30, slope1_min=0.00005, cooldown=90, use_session_filter=True, use_kama_slope_filter=False`

### Strategy 2: MTF KAMA Pivot v1
- **File**: `strategies/mtf_kama_pivot_v1.py`
- **Winner params (Candidate B)**: `pivot_len=3, near_pivot_atr=0.5` + same v4 baseline above

## Your Task

Run OOS validation on the sealed holdout period only.
Use `--data histdata` and gate with `--start` / `--end`.

```bash
python strategies/mtf_kama_dual_v4.py --data histdata --start 2025-10-01 --end 2026-02-28
python strategies/mtf_kama_pivot_v1.py --data histdata --start 2025-10-01 --end 2026-02-28
```

Each run writes a results file to `strategies/`. The full sweep still runs, but report the fixed IS-winner params listed above.

## What to Report

For each strategy, extract the winner-params row and report:

| Period | Strategy | Trades | PF | Net $ | Max DD% | Win Rate% |
|--------|----------|--------|----|-------|---------|-----------|
| 2025-10-01 to 2026-02-28 | v4 | ... | ... | ... | ... | ... |
| 2025-10-01 to 2026-02-28 | Pivot v1 | ... | ... | ... | ... | ... |

Also note:
- Winner-param rank in the OOS sweep (1st, 5th, etc.)
- Trades > 0 (0 trades is a red flag)
- PF > 1.0 OOS (minimum bar for "not broken")
- PF > 2.0 OOS (strong for trend-following FX)

## Key Notes

- Do not re-optimize on OOS data.
- The scripts run a full param sweep; only extract/report the fixed winner row.
- `data/` is at repo root, one level above `strategies/`; engine resolves it automatically.
- Commission is 0.0043% per side (already set in BacktestConfig).
