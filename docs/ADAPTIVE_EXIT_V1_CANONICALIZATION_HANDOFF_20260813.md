# Adaptive Exit V1 — Canonicalization Handoff

## Objective

Promote `F4_disagreement_abstain_p80` as the canonical long-only adaptive-exit
controller around the existing Strict-R3/A5 admission and portfolio stack.

Do not promote any winner-extension arm. The completed winner-extension funnel
left Frozen V1 as the incumbent.

## Implementation receipt — 2026-08-13

The canonicalization implementation is complete in research/shadow mode.
Round 2 was discarded and is not part of this contract.

The sealed deployable bundle is:

```text
data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1
bundle id: 09ae898a734351911ac5
model bundle SHA-256: bd415ec0e32bae9701fc771621ab5d568931bdb1929ed9b4546f660bce28be13
```

It was fitted on 40,000 deterministic equal-month states from
2025-11-01 through 2026-07-31 12:00 UTC, after the required 12-hour label
purge. The inner prior-only gate contains 13,646 states. The frozen
training-only disagreement p80 cutoff is 1.3607286844944153 ATR. Final F1 and
F4 refits use 698 and 371 trees, respectively.

Runtime implementation:

```text
extreme_price_movements/adaptive_exit_v1.py
scripts/build_adaptive_exit_v1_canonical_bundle.py
scripts/validate_adaptive_exit_v1_bundle.py
extreme_price_movements/strict_r3_shadow_portfolio.py
scripts/run_strict_r3_shadow_cycle.py
```

The active hourly orchestrator already selects:

```text
config/strict_r3_inference_bundle_long_20260801_v5_homogeneous28_a5_b10_exactpolicy_v3_oi_continuity.json
```

That sealed schema-v5 contract now requires and hashes both V1 models and the
V1 manifest. It verifies 25 immutable artifacts and 31 runtime-code hashes.
Open-position state is schema v3 and persists the chosen activation, entry
context, causal score history, model decomposition, and effective timestamp.
The prior activation is used to process the just-completed bars; the new
decision can affect only the next 15-minute bar. Missing state or contract
inputs fall back to the frozen base activation.

The serialization receipt is:

```text
data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1/correctness_test_report.json
```

It reproduces F1, F4, and selected activation at zero maximum error on 254
sealed reference states. Focused unit/contract tests pass 21/21. This is not
an untouched trading interval and does not promote live authority:

```text
RESEARCH_CANONICAL = true
LIVE_CANONICAL = false
```

Because runtime code and portfolio-state schema changed, pre-existing
untouched-shadow checkpoints remain evidence for the preceding static-exit
seal. V1 requires a new chained untouched shadow interval before live
promotion.

## Canonical identity

```text
canonical name: adaptive_exit_v1_f4_disagreement_abstain_p80
side: long only
controller authority: trailing activation only
decision clock: after each completed 1-hour bar
decision effective: next 15-minute bar
source path: exact/complete 15-minute OHLC
fallback source: frozen hourly F1 controller only where the authoritative
                 historical contract explicitly uses the hourly proxy
cost: 100 bps exactly once
timeout: 12 hours / 48 complete 15-minute bars
```

Frozen SimplePolicyOptimiser geometry:

```text
SL:                         4.15200064332387 ATR
base trailing activation:   2.326224919759605 ATR
fixed trailing giveback:    0.10237198997143725 ATR
```

The controller must never change the stop, fixed giveback, trailing power,
timeout, entry, cost, or fill convention.

## Controller definition

Two LightGBM quantile regressors predict:

```text
target = remaining_favorable_from_entry_atr
objective = quantile
alpha = 0.65
```

- F1 uses the 28-field short causal path contract.
- F4 uses F1 plus rich causal path state, archetype summaries, entry trust and
  score-evolution fields. The ordered contract is stored in the authoritative
  run manifest.

For each live decision state:

```python
base = 2.326224919759605
core = clip(
    base + 0.75 * (max(f1_prediction, 0.0) - base),
    0.50 * base,
    1.25 * base,
)

disagreement = abs(f1_prediction - f4_prediction)

selected_activation = (
    base
    if disagreement > training_only_disagreement_p80
    else core
)
```

The selected activation persists until the next completed-hour decision.
Changing it cannot loosen an already protected trailing lock.

## Training protocol

- Outer folds: three-month chronological OOF blocks from April 2025 through
  July 2026.
- Training history: rolling nine months, bounded below by 2025-01-01.
- Purge: training timestamps end at least 12 hours before held-period start.
- Training cap: 40,000 states, sampled with deterministic equal-month support.
- Inner split: earliest 65% trains the opportunity models; latest 35%, after a
  12-hour gap, estimates F1/F4 disagreement and the p80 abstention cutoff.
- Final outer predictions: refit F1 and F4 on the full eligible outer-training
  population and score only the held block.
- All preprocessing medians are fit on training rows only.
- For a production activation, refit the same contract entirely on data and
  fully resolved labels strictly before the activation timestamp.

LightGBM parameters for both F1 and F4:

```text
n_estimators:       700 ceiling
learning_rate:      0.03
max_depth:          4
num_leaves:         15
min_child_samples:  max(100, 1% of fit rows)
subsample:          0.75
subsample_freq:     1
colsample_bytree:   0.75
lambda_l2:          10.0
early_stopping:     30 rounds
seed:               20260813
```

## Required deployable bundle

The research run does not serialize sufficient inference state. Produce one
immutable bundle containing:

1. fitted F1 LightGBM booster;
2. fitted F4 LightGBM booster;
3. ordered F1 and F4 feature names;
4. training-only imputation medians for each model;
5. training-only F1/F4 disagreement p80 cutoff;
6. base activation, shrink and bounds;
7. frozen SimplePolicyOptimiser geometry;
8. training start/end, label cutoff, purge, row cap and sampled-row hashes;
9. feature-source and model-code hashes;
10. canonical 15-minute entry/fill/cost/timeout contract;
11. a bundle ID and SHA-256 manifest.

Recommended bundle schema:

```json
{
  "schema": "strict_r3_adaptive_exit_v1_bundle_v1",
  "side": "long",
  "controller": "F4_disagreement_abstain_p80",
  "target": "remaining_favorable_from_entry_atr",
  "objective": "quantile_0.65",
  "decision_clock": "completed_hourly_bar",
  "effective_from": "next_15m_bar",
  "activation_only": true,
  "base_activation_atr": 2.326224919759605,
  "activation_shrink": 0.75,
  "activation_lower_ratio": 0.5,
  "activation_upper_ratio": 1.25,
  "disagreement_rule": "abs(F1-F4) > train_p80 => base activation",
  "sl_atr": 4.15200064332387,
  "fixed_giveback_atr": 0.10237198997143725,
  "timeout_hours": 12,
  "round_trip_cost_bps": 100
}
```

## Runtime wiring

Add an `adaptive_exit_v1` section to the active Strict-R3 inference bundle and
load it in the hourly shadow/live cycle after entry admission and portfolio
acceptance.

At each completed hour for every open long position:

1. materialize the exact causal path state from bars ending at that hour;
2. enforce the ordered F1/F4 feature contracts and training medians;
3. fail closed to the base activation if any contract/parity check fails;
4. score F1 and F4;
5. apply the frozen disagreement rule and continuous activation formula;
6. publish the decision only for the next 15-minute bar;
7. preserve the current stop/giveback/power and any earned trailing lock;
8. store the complete feature, model-output and decision decomposition.

The runtime currently does not contain this controller. Do not declare
canonical deployment complete until the active shadow cycle consumes the
serialized bundle and produces matched decisions.

## Mandatory tests

- Training and runtime feature vectors match exactly in name, order, values and
  missing-value treatment.
- Appending future bars cannot alter an earlier state or decision.
- Decision at hour `h` is unavailable before `h` closes and cannot affect a bar
  before the next 15-minute open.
- F1/F4 disagreement cutoff is fit from prior training data only.
- Held-window percentiles are never used.
- Stop, giveback, power, timeout and cost remain bit-identical to the frozen
  policy.
- Costs are deducted exactly once.
- Missing or invalid features fail closed to base activation.
- Replaying the sealed bundle on the stored OOF/reference inputs reproduces the
  stored selected activations within tolerance and the same exit bars.
- A no-controller replay reproduces the canonical baseline exactly.
- Historical hybrid replay retains authoritative frozen outcomes where exact
  path reconstruction is explicitly unavailable; live inference must use
  complete decision-time 15-minute history.

Required parity gate:

```text
net outcome max absolute error <= 0.01 bps
exit-bar identity = 100%
raw prediction tolerance <= 1e-6
selected activation tolerance <= 1e-6 ATR
```

## Evidence and metrics

Fine-path development winner:

```text
arm: F4_disagreement_abstain_p80
2025 rows: 18,532
baseline: +93.96 net bps/trade
adaptive: +131.13 net bps/trade
uplift: +37.17 bps/trade
```

Authoritative matched fixed-trade replay:

```text
trades: 8,453
baseline: +163.09 net bps/trade
adaptive: +183.63 net bps/trade
uplift: +20.55 bps/trade
adaptive-supported fraction: 80.44%
```

Dynamic-capacity portfolio replay:

```text
baseline: 8,453 trades, 14.72/day, +163.09 net bps/trade
V1:       8,663 trades, 15.09/day, +181.61 net bps/trade
uplift:   +18.52 bps/trade
positive rate: 68.22%
worst week: -11.67%
Sortino: 0.540
max drawdown: -76.53%
```

The drawdown reflects the existing aggressive portfolio sizing. V1 is the exit
controller; it does not authorize a change to portfolio leverage or sizing.

## Authoritative files and hashes

```text
data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4/run_manifest.json
sha256 afab6381cfd10a1a63c3e2eee908e7988db82e53ec4ea3f5dfacef3290484f96

data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4/oof_replay.parquet
sha256 bf60e2278529cac188e6aefe25bfbc29a76ed204a1dc509d492b881f731a192b

data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4/run_manifest.json
sha256 48578b96d98eb8b47a59f95d83a45e07f2e363ee1f4c0ccaf6eb8811399af0bd

data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json
sha256 2dc9a145766ae383a4ab7c33e8a9f9e358175597e05582300ff0a05732673603

scripts/run_canonical_a5_15m_adaptive_exit_funnel.py
sha256 b42328a8716124a265deb44f5134a860cd68ad1d96af79e2b6d3a271f5f77380

scripts/reconcile_authoritative_a5_adaptive_exit.py
sha256 e832ea0f320caa543e2afd503eb7c0415d3116004b3f931e0ed8af7d52e0f5a5
```

Relevant implementation:

- `extreme_price_movements/path_based_exit_optimisation.py`
  - `build_hourly_exit_feature_cache`
  - `build_hourly_exit_path_states`
  - `sequential_continuous_activation_replay`
- `scripts/run_hourly_exit_parameter_ablation.py`
  - `_fit_regressor`
- `scripts/run_canonical_a5_15m_adaptive_exit_funnel.py`
- `scripts/reconcile_authoritative_a5_adaptive_exit.py`

## Documentation updates

Update `docs/TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md` to state:

- Adaptive Exit V1 is the canonical long-only exit controller.
- `F4_disagreement_abstain_p80` is the exact fine-path controller.
- The authority is activation-only.
- Winner Extension W1–W6 is not canonical.
- Top-level baseline exit geometry remains the pre-2025
  SimplePolicyOptimiser winner shown above.
- Historical OOF evidence and deployable serialized bundle are distinct.
- Live activation is allowed only after sealed-bundle replay parity and one
  untouched shadow interval.

## Promotion status

Use two separate states:

```text
RESEARCH_CANONICAL = true
LIVE_CANONICAL = false until serialized-bundle/runtime parity passes
```

Once the runtime parity gate and untouched shadow interval pass, change only
`LIVE_CANONICAL` to true. Do not rerun selection or alter the controller while
performing that promotion.
