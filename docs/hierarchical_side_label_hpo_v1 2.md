# Hierarchical Side Label HPO v1

## Purpose

Optimize causal base labels independently for long and short opportunity streams
without coupling geometry, target shape, archetype specialization, and full model
parameters in one high-dimensional search.

The search objective is OOS base ranking by gross executable EV. Net EV under
the fixed cost contract is always reported and remains a promotion guard, but is
not optimized directly.

## Non-Negotiable Contracts

- `decision_ts = signal_ts + signal_timeframe`.
- `first_path_timestamp >= decision_ts`.
- Geometry is evaluated from path bar zero at or after `decision_ts`.
- Timeout return uses the actual final available close.
- Incomplete paths are censored and excluded, not converted to losses.
- Costs are not subtracted in the gross-EV objective. The corresponding net EV
  is computed once from the documented cost contract.
- Long and short studies, models, rankings, and diagnostics remain separate.
- Archetypes must be observable at decision time. Sparse archetypes fall back
  exactly to their side parent.
- April through June 2026 is untouched by label-HPO selection. July is reported
  separately while incomplete.

## Stage 0: Immutable Path Primitive Cache

Create a chunked, side-separated cache keyed by:

```text
label source hash
market data source hash
symbol and signal timestamp
signal timeframe
decision latency
path timeframe and maximum horizon
entry-model version
```

Persist compact `float32` arrays, with metrics accumulated in `float64`:

```text
signal_ts, decision_ts, first_path_ts
symbol, side, archetype_label_family
entry_open, barrier
side-relative high-return path
side-relative low-return path
side-relative close-return path
valid/censored path mask
```

Use memory-mapped chunks rather than one wide pandas frame. A Numba kernel
evaluates each proposed geometry and emits:

```text
gross_return
hit, stop, timeout, same_bar_both
bars_to_resolution
MFE and MAE before resolution
MFE-before-MAE ordering
post-activation/post-win MFE
underwater bars, fraction, and area
```

The kernel is validated against the scalar first-touch and trailing replay for
both sides. Delayed entry must use a true path beginning at the delayed entry
timestamp; it must never combine a delayed open with earlier intrabar extremes.

## Stage 1: Side-Parent Geometry

Run independent Optuna studies for long and short. Target-shape parameters are
fixed in this stage.

| Parameter | Long | Short |
|---|---:|---:|
| trailing activation | 0.60-1.60 R | 0.50-1.80 R |
| trailing gap | 0.20-0.70 R | 0.15-0.75 R |
| hard stop | 0.60-1.25 R | 0.50-1.15 R |
| maximum hold | 3-12 h | 2-10 h |

Constraints:

```text
trailing_gap <= 0.80 * trailing_activation
hard_stop >= 0.50 R
```

Default budget is 80 trials per side with 24 startup trials and multivariate
TPE. Each trial uses a fixed, deliberately shallow L2 proxy:

```text
max_depth=3
num_leaves=15
min_child_samples=250
learning_rate=0.04
n_estimators=500 maximum
reg_lambda=4.0
```

Use the final 20% of each allowed training fold as an inner chronological early
stopping set, with patience 50. The OOS fold is never used for tree early
stopping.

## Stage 2: Continuous Side Target

Keep the best geometry candidates fixed and optimize target shape separately.
The target contains no discrete stop, timeout, or dirty-path caps.

```python
gross_r = realized_gross_return / stop_distance
ev_term = gross_r / ev_temperature
path_term = (1.0 - mae_to_stop_r) / mae_temperature
speed_term = (max_hold_bars - bars_to_resolution) / time_temperature
post_win_term = (
    post_activation_mfe_r - post_win_offset
) / post_win_temperature

target_soft = sigmoid(
    ev_term
    + w_path * path_term
    + w_speed * speed_term
    + w_post_win * post_win_term
)
```

`w_ev` is fixed at 1.0. Search ranges:

| Parameter | Long | Short |
|---|---:|---:|
| EV temperature | 0.35-1.40 R | 0.30-1.50 R |
| MAE temperature | 0.15-0.60 R | 0.10-0.55 R |
| time temperature | 0.20-0.90 x hold | 0.15-0.75 x hold |
| post-win offset | 0.00-0.60 R | 0.00-0.75 R |
| post-win temperature | 0.15-0.80 R | 0.15-0.90 R |
| path weight | 0.00-0.75 | 0.00-0.90 |
| speed weight | 0.00-0.50 | 0.00-0.65 |
| post-win weight | 0.00-0.35 | 0.00-0.30 |

Default budget is 96 trials per side. Reject a target before model fitting when:

```text
target IQR < 0.10
more than 5% of rows occupy one non-boundary exact value
target top-10 mean is not above target top-30 mean
finite target coverage is below 99.9% of valid paths
```

## Stage 3: Target-Strength Sample Weight

Run this study only after each side's geometry and continuous target have been
frozen. Search:

```text
target_exponent in {1.00, 1.25, 1.50, 1.75, 2.00}
weight_range_ratio ~ log_uniform(3.00, 12.00)
```

For fitted training rows only:

```python
strength = clip(target_soft, 0.0, 1.0) ** exponent
raw_weight = strength

# After timestamp/archetype corrections, winsorize on train only.
raw_weight = minimum(raw_weight, train_quantile(raw_weight, 0.99))

# Find c by bisection so the bounded weights have mean exactly one.
weight_min = 1.0 / sqrt(weight_range_ratio)
weight_max = sqrt(weight_range_ratio)
sample_weight = clip(c * raw_weight, weight_min, weight_max)
solve c such that mean(sample_weight) == 1.0
```

The p99 cap and centering constant are fitted on the permitted training rows and
reused unchanged inside that fold. No OOS target distribution may set them.
Within the same balancing group, target `1.0` receives maximum opportunity
weight. Target `0.0` receives the selected `weight_min` floor instead of
disappearing from the training loss. `weight_range_ratio` is exactly the
largest-to-smallest permitted weight multiple. Its log-symmetric derivation
avoids searching redundant floor and ceiling parameters independently. Search
the exponent and ratio only after the side target is frozen. In Optuna use
`suggest_float("weight_range_ratio", 3.0, 12.0, log=True)`.

### Rebalancing Without Erasing Opportunity Strength

Because models are already side-specific, do not add a side-balance factor.
Apply two bounded corrections before the final mean-one solve:

```python
# Prevent timestamps with more listed assets from dominating.
timestamp_factor = inverse_rows_at_timestamp
timestamp_factor /= mean(timestamp_factor)

# Tempered archetype balancing, not full inverse frequency.
archetype_factor = (median_archetype_support / archetype_support) ** gamma
archetype_factor = clip(archetype_factor, 0.80, 1.25)

raw_weight *= timestamp_factor
raw_weight *= archetype_factor
```

Use `gamma=0.25` initially. A diagnostic ablation may test `0.0`, `0.25`, and
`0.50`, but exponent and gamma should not be searched jointly in the first run.
This keeps high-target rows dominant inside each archetype while preventing a
large archetype or dense timestamp from consuming the loss.

Reject a weighting arm when any of these occur:

```text
effective sample size / row count < 0.60
top target decile receives > 30% of total weight
any supported archetype has effective sample size < 150
monthly effective-weight share differs by > 2x from row share
top-10 gross EV improves only through one month or archetype
```

Evaluate exponents using the same chronological OOS proxy folds and gross-EV
objective as target HPO. Report net EV, target buckets, effective sample size,
weight share by month/archetype, and score dispersion as diagnostics.

An optional later contrastive arm may upweight OOF high-score/low-target hard
negatives. Do not add it to this first simple exponent study because it requires
a prior OOF model and would confound target-strength weighting.

## Stage 4: Side x Archetype Shrinkage

Only six local multipliers are eligible:

```text
activation, trailing gap, hard stop, maximum hold,
MAE temperature, time temperature
```

Each is bounded to 0.80-1.20 of the side parent. A local study requires:

```text
at least three prior development folds
n_eff >= 150 * 6 = 900
at least 150 OOS top-10 candidates across prior folds
positive side-parent gross-EV support
```

Shrink local parameters toward the side parent:

```python
support = min(1.0, n_eff / (150.0 * 6.0))
stability = clip(mean_local_fold_ev / max(mean_parent_fold_ev, eps), 0.0, 1.0)
dispersion = 1.0 / (
    1.0 + std_local_fold_ev / (abs(mean_parent_fold_ev) + eps)
)
local_weight = clip(support * stability * dispersion, 0.0, 0.85)

final_param = (
    local_weight * archetype_param
    + (1.0 - local_weight) * side_parent_param
)
```

## Development Splits and Purging

Default label-HPO development folds:

```text
train through 2025-09 -> validate 2025-10
train through 2025-10 -> validate 2025-11
train through 2025-11 -> validate 2025-12
train through 2025-12 -> validate 2026-01
train through 2026-01 -> validate 2026-02
train through 2026-02 -> validate 2026-03
```

Purge training rows whose outcome path overlaps validation. Embargo by the
maximum candidate holding horizon plus one signal bar.

## Objective, Early Stopping, and Pruning

Rank top 10%, 20%, and 30% independently within side. For each complete OOS
fold:

```python
monotonicity = (
    min(0.0, gross_ev_top10 - gross_ev_top20)
    + min(0.0, gross_ev_top20 - gross_ev_top30)
)

fold_score = (
    0.35 * gross_ev_top10
    + 0.20 * gross_ev_top20
    + 0.10 * gross_ev_top30
    + 0.15 * worst_week_gross_ev_top10
    + 0.15 * worst_month_gross_ev_top10
    + 0.05 * monotonicity
)

objective = (
    mean(fold_scores)
    - 0.50 * std(fold_scores)
    + 0.25 * min(fold_scores)
)
```

Call `trial.report(partial_objective, fold_number)` after every OOS fold. Use:

```text
TPESampler(multivariate=True, group=True, n_startup_trials=24)
MedianPruner(n_startup_trials=20, n_warmup_steps=2, interval_steps=1)
```

After two folds, also prune when a trial's partial upper confidence bound is
below the incumbent's lower confidence bound. Record every pruning decision and
the folds available at that decision.

## Proxy Funnel

1. Numba economic precheck on all cached paths. Reject invalid geometry,
   insufficient coverage, and clearly dominated gross-return distributions.
2. Shallow fixed L2 proxy on a beginning/middle/end sample and three expanding
   development folds. Optuna pruning is active.
3. Medium proxy on up to 150k rows and all six development folds for the top 12
   candidates per side.
4. Full LightGBM evaluation only for the top two parent configurations per side
   plus the incumbent.

Finalists rerun train-only feature selection, use L2 loss and
`W7_timestamp_balanced`, and fit separate long and short models. April-June is
evaluated once after finalist selection.

## Required Artifacts

```text
path_primitives_manifest.json
geometry_trials_long.parquet
geometry_trials_short.parquet
target_trials_long.parquet
target_trials_short.parquet
archetype_shrinkage_long.parquet
archetype_shrinkage_short.parquet
fold_metrics.parquet
finalist_predictions.parquet
label_policy_side_archetype.json
label_target_contract.json
leakage_audit.json
search_breadth.json
runtime_memory_profile.json
```

## Required Tests

- Numba geometry equals scalar replay for long, short, trailing, stop, timeout,
  same-bar conflict, and actual-close timeout.
- Target is finite, bounded, continuous, and free of non-boundary atoms.
- Gross and net reporting reconcile exactly to one cost deduction.
- Proxy early stopping uses inner-train rows only.
- Pruning reads no future development or final-holdout fold.
- Archetype shrinkage uses prior resolved rows only.
- Unsupported archetypes reproduce the side parent exactly.
- Top-k ranking is per-side.
- Finalist paths satisfy `first_path_timestamp >= decision_ts`.
