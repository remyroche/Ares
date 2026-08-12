# Strict-R3 robust EV-mapping ablation — long-only

Date: 2026-08-12  
Status: completed causal research ablation; no automatic production promotion

## Decision

Keep the exact-producer reserve map as the executable control. Promote
`cell_day_trim_15pct` to the preferred **shadow successor** for the next frozen,
untouched forward window.

The reason is objective-dependent:

- exact reserve maximises 2026 trade quality and Sortino;
- cell-day trim 15% materially improves throughput, total profit, active-week
  coverage, and the combined 2025–July 2026 profit/drawdown ratio;
- Bayesian and residual-state maps improve continuity but give up too much
  per-trade quality and shock-week protection;
- protecting the most recent five days from bottom-tail trimming does not
  improve the 15% arm.

This experiment used 2025 and 2026 evidence for selection. It therefore cannot
authorize live replacement until a later frozen window confirms the choice.

## Frozen matched contract

Every arm used the same:

- long-only target-free candidate population;
- strict prequential R3 upstream and conversion bundles;
- frozen October–December 2024 Geometry/K9 bundle;
- exact same-producer 42-day calibration reserves excluded from supervised fits;
- SimplePolicyOptimiser outcome: entry at signal close + one hour, SL
  4.152000643 ATR, trailing activation 2.326224920 ATR, giveback
  0.102371990 ATR, H12 timeout, and 100 bps cost exactly once;
- 20 score cells and +50 bps admission floor for point-estimate arms;
- portfolio auction: two new entries per 15-minute bar, eight concurrent, one
  position per asset, 80% margin cap, 7x leverage, 10% wallet margin slots;
- same-producer `final_score` only as a deterministic secondary tie-break when
  mapped EV is exactly equal.

All held-day updates use `policy_label_available_ts < held day 00:00 UTC`.
No held-window percentiles, future-path-qualified candidates, or cross-vintage
raw-score pooling are used.

## Arms

### Whole-day robust residual filtering

Fit the provisional same-producer reserve map, aggregate its calibration error
by UTC day, and filter complete producer days before refitting:

- robust-z symmetric trim: 10%, 15%, 20%, 25%;
- ordinary standard-deviation filters: 1.0, 1.5, 2.0 sigma.

These arms do not solve the June/July drought. Their best 2026 result is the
1-sigma arm at +83.14 bps/trade versus +83.13 for control, with no June or July
admissions. They do not advance.

### Causal residual state

The reserve seeds a causal 21-day daily residual state. Arms use EV level,
3d-versus-14d trend, slope, standard deviation, sign entropy, or their
combination. They use only labels resolved before each held UTC day.

The state arms are unstable across eras. For example, the combined state arm is
+151.65 bps/trade in 2025 Jan–Mar, +67.71 in 2025 Apr–Jul, but only +38.00 in
2026. They do not advance as admission maps.

### Equal-weight cell-day estimator

Each exact producer's reserve fixes twenty score cells. The estimator first
computes:

```text
EV(cell, day) = mean(policy_net_bps for trades in that score cell and UTC day)
```

The map then averages cell-day observations, so one shock day has one day's
influence rather than influence proportional to its candidate count. Symmetric
trimming removes the top and bottom 10%, 15%, 20%, or 25% cell-day observations
inside the causal rolling 42-day window. Curves are made monotone across score
cells.

The 15% arm has the strongest portable profit-oriented result.

### Bayesian cell-day estimator

The empirical-Bayes arms treat cell-day means as observations. Each cell
posterior shrinks toward an equal blend of:

1. the producer-wide equal-day EV; and
2. the frozen same-producer model-family prior.

Admission uses `P(mu_cell > 0)` at 0.70, 0.80, or 0.90. Prior strength is 3,
7, or 14 equivalent days. The most conservative tested arm (`k=7`, `p=0.90`)
is portable and continuous, but its total-profit/risk result remains below the
15% cell-day arm.

### Recent-five-day bottom protection

The bottom-tail filter was prohibited from removing observations in the five
calendar days immediately preceding the held day. Top-tail trimming remained
active over the full causal window.

This did not improve the 15% arm:

- 2025 Jan–Mar: +130.25 versus +132.06 bps/trade;
- 2025 Apr–Jul: +83.38 versus +86.22;
- 2026 Jan–Jul: identical +52.96.

The rolling window already provides sufficient reactivity. This variant does
not advance.

## Portfolio-constrained results

| Period | Arm | Trades | Trades/day | Net bps/trade | Sum net bps | Sortino | Max DD | Worst week | Mean margin use |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 Jan–Mar | Exact reserve | 1,883 | 20.92 | +165.65 | 311,914 | 0.347 | -79.9% | +8.1% | 87.6% |
| 2025 Jan–Mar | Cell-day trim 15% | 1,981 | 22.01 | +156.84 | 310,698 | 0.347 | -77.0% | -8.9% | 89.1% |
| 2025 Apr–Jul | Exact reserve | 1,145 | 9.39 | +145.95 | 167,109 | 0.411 | -63.4% | +4.3% | 82.1% |
| 2025 Apr–Jul | Cell-day trim 15% | 2,039 | 16.71 | +147.02 | 299,766 | 0.392 | -43.9% | -22.6% | 82.9% |
| 2026 Jan–Jul | Exact reserve | 1,374 | 6.48 | +138.46 | 190,241 | 0.635 | -33.6% | +2.8% | 83.5% |
| 2026 Jan–Jul | Cell-day trim 15% | 2,875 | 13.56 | +119.20 | 342,713 | 0.435 | -62.9% | -22.5% | 87.9% |
| 2026 Jan–Jul | Bayesian k7 / p90 | 3,379 | 15.94 | +104.72 | 353,849 | 0.367 | -63.5% | -41.7% | 88.0% |

### Combined evidence

| Arm | Trades | Net bps/trade | Sum net bps | Active weeks | Negative active weeks | Sum log wallet growth | Worst block DD | Log growth / worst DD | Weighted Sortino |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact reserve | 4,402 | +152.04 | 669,265 | 43 | 2 | 38.48 | -79.9% | 48.16 | 0.454 |
| Cell-day trim 15% | 6,895 | +138.24 | 953,177 | 56 | 4 | 53.76 | -77.0% | 69.82 | 0.397 |
| Bayesian k7 / p90 | 7,430 | +127.31 | 945,914 | 61 | 6 | 53.08 | -77.0% | 68.95 | 0.364 |

Relative to exact reserve, cell-day trim 15% produces:

- 56.6% more trades;
- 42.4% more summed net outcome;
- 39.7% more cumulative log-wallet growth;
- 45.0% better log-growth/worst-block-drawdown ratio;
- 9.1% lower net EV per trade;
- 12.5% lower trade-weighted Sortino;
- two additional negative active weeks and a worse single shock week.

## 2026 monthly cell-day 15% portfolio result

| Month | Trades | Net bps/trade |
|---|---:|---:|
| January | 646 | +107.08 |
| February | 457 | +95.39 |
| March | 544 | +156.63 |
| April | 534 | +183.02 |
| May | 624 | +77.73 |
| June | 69 | -26.40 |
| July | 1 | +321.38 |

The arm mitigates but does not eliminate the regime shock. July remains almost
a drought under the fixed +50 bps point-estimate threshold. The Bayesian p90
arm restores 266 July portfolio trades at +33.5 bps/trade when used as an
exact-primary timestamp fallback, but its aggregate Sortino and worst-week
risk are materially inferior; it remains diagnostic only.

## Promotion rule

Run all three maps in shadow from the same live score and resolved-outcome
ledger:

1. exact reserve control;
2. cell-day trim 15%;
3. Bayesian k7 / p90 diagnostic.

Do not let shadow arms alter live admission, sizing, or the auction. Promote
cell-day trim 15% only after a later untouched period confirms:

- higher total net outcome and log-growth/drawdown efficiency;
- no material degradation in worst week or loss clustering;
- no candidate-identity, score-vintage, cost, or replay/inference parity fault.

## Artifacts and scripts

Primary utilities:

- `scripts/ablate_strict_r3_robust_ev_mapping.py`
- `scripts/ablate_strict_r3_cell_day_bayesian_ev_mapping.py`
- `scripts/combine_strict_r3_ev_map_fallbacks.py`
- `scripts/replay_strict_r3_tail_health_portfolio.py`

2026 artifacts:

- `data_perp/artifacts/strict_r3_robust_exactreserve_ev_map_ablation_long_2026_janjul_20260812_v2`
- `data_perp/artifacts/strict_r3_cell_day_bayesian_ev_map_ablation_long_2026_janjul_20260812_v2`
- `data_perp/artifacts/strict_r3_ev_map_timestamp_fallbacks_long_2026_janjul_20260812_v1`
- `data_perp/artifacts/strict_r3_cell_day_bayesian_portfolio_long_2026_janjul_*_20260812_v1`

Matched 2025 artifacts use the analogous `2025_janmar` and `2025_aprjul`
directories. Each immutable run manifest records source paths and hashes.
