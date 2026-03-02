# Ridge Position Sizer — Backtesting-Centric Sizing Review (Updated)

## Scope correction

This review is now explicitly based on **`ridge_position_sizer.py` backtesting logic** (especially `run_oof_grid_backtest`) rather than Stage-40 optimiser protocol.

---

## What the ridge backtesting path does today

### 1) Score preparation and ranking basis
- Uses `sizer_score_oof` as the primary signal in OOF backtesting.
- De-means score by 7-day asset rolling average (`dev = sizer_score_oof - week_avg`) and ranks cross-sectionally per timestamp (`rank_pct`).
- Entry universe is gated by top quantiles via `q in {0.30, 0.10, 0.05}` (i.e., top 30%, 10%, 5%).

### 2) Current sizing choices under test
In the grid, only two sizing modes are currently compared:
- `fixed_10`: constant 10% size.
- `linear_5_15`: linear interpolation from 5% to 15% over the selected rank band.

### 3) Other grid axes that interact with sizing
Current OOF backtest also sweeps:
- `entry_offset_mode in {optimizer, fixed_0_15}`
- `tp_sl_ratio in {2:1, 3:2, 4:2}`
- `quantile in {0.30, 0.10, 0.05}`

So sizing comparisons are already embedded in a multi-axis setup where interaction effects matter.

### 4) Return/PnL construction used in backtest
- If path arrays exist (`future_highs/lows/closes` + policy params), net utility is simulated via policy utility path.
- Otherwise, fallback proxy uses `fwd_ret_H4 - offset - fee_roundtrip`.
- PnL is computed as `start_equity * position_fraction * net`.

### 5) Metrics emitted by this backtest
Per configuration, output includes:
- `net_pnl`, `sortino`, `maxdd`, `ulcer`,
- `trades_per_day`, `expectancy_per_trade`, `win_rate`.

---

## Backtesting-specific suggestions for alternative score→size mappings

Below are alternatives designed to plug into the **same `run_oof_grid_backtest` structure** as a new `sizing_mode` family.

### A) Piecewise deadzone + ramps
**Rationale:** enforce no/low risk in marginal ranks and concentrate only where rank edge is stronger.

- Example within selected band (`u` in [0,1]):
  - `u < 0.25` → `size=0.00`
  - `0.25<=u<0.70` → ramp to `0.08`
  - `u>=0.70` → ramp to `0.15`
- Better aligned to “take fewer weak top-quantile names”.

### B) Convex power ramp
**Rationale:** if PnL contribution is top-heavy, convexity captures more from extreme ranks.

- `size = s_min + (s_max - s_min) * u^p`, `p > 1`.
- Start with `p in {1.25, 1.75, 2.25}`.
- Compare vs `linear_5_15` (`p=1`) directly.

### C) Concave power ramp
**Rationale:** reduce sizing volatility if top-rank estimates are noisy.

- Same form, `p < 1` (e.g., `0.6, 0.8`).
- Expect lower variance / smoother DD profile.

### D) Quantile step buckets inside selected set
**Rationale:** robust and governance-friendly mapping; lower sensitivity to tiny rank jitter.

- Example for selected names:
  - bottom third: `0.05`
  - middle third: `0.09`
  - top third: `0.15`
- Simple to monitor and explain in production.

### E) Soft-capped exponential map
**Rationale:** stronger top allocation while hard-capping leverage/size.

- `size = s_min + (s_max - s_min) * (1 - exp(-beta*u)) / (1 - exp(-beta))`
- Tune `beta` for curvature; monotone and bounded.

### F) Rank-to-size with shrinkage toward bucket mean
**Rationale:** reduce overreaction to timestamp-level noisy cross-sectional ranks.

- `size = lambda * map(rank) + (1-lambda) * mean_size_selected`
- Tune `lambda` in `[0.5, 1.0]`.
- Should reduce size-churn and turnover-like instability.

---

## How to test these alternatives in ridge backtest terms

1. **Keep current non-sizing axes unchanged initially**
   - Preserve quantile/offset/tp-sl grids while adding sizing families.
   - This isolates sizing impact in existing backtest regime.

2. **Promote by interaction-robustness, not single best row**
   - Require candidate sizing to outperform baseline across several combinations of:
     - quantile (`0.30/0.10/0.05`),
     - offset mode,
     - tp/sl ratio.

3. **Primary/secondary selection criteria**
   - Primary: `net_pnl` with non-degraded `maxdd`/`ulcer` profile.
   - Secondary: `sortino`, `expectancy_per_trade`, `win_rate`.

4. **Stability checks to add in report table**
   - Mean + std of metrics across non-sizing axes for each sizing family.
   - “Win count vs baseline” (how many grid cells beat `linear_5_15`).

---

## First 3 experiments to run (ridge-backtest aligned)

1. **`linear_5_15` vs convex power ramps** (`p=1.25/1.75/2.25`)
   - Tests whether performance is top-rank concentrated.

2. **`linear_5_15` vs piecewise deadzone**
   - Tests whether dropping weak selected names improves DD-adjusted outcomes.

3. **Best from #1/#2 vs quantile step buckets**
   - Tests robustness and simplicity tradeoff.

---

## Decision rule

Adopt a new default only if it:
1) beats `linear_5_15` on aggregate `net_pnl`,
2) does not materially worsen `maxdd` and `ulcer`, and
3) wins in a broad subset of grid interactions (not just one narrow configuration).

