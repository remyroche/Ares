# Adaptive Exit V1 — Live Promotion and Monitoring Handoff

## Objective

Prepare the already selected Adaptive Exit V1 controller for live inference,
then monitor its first live-shadow and live-controlled decisions for exact
training/replay/runtime parity.

Adaptive Exit V1 is an **activation-only overlay on the frozen
SimplePolicyOptimiser policy**, never a standalone exit component. Every
position starts with the optimiser's stop, activation, giveback, timeout,
entry convention, and cost. V1 may subsequently replace only the activation
multiple after a completed hourly bar; missing inputs or abstention retain the
optimiser activation.

This is an implementation and operational-validation task. It is **not** a
new model-selection funnel. Do not tune, replace, or reselect the controller.

Canonical controller:

```text
adaptive_exit_v1_f4_disagreement_abstain_p80
side: long only
research canonical: true
live canonical: false until the gates below pass
```

## Frozen evidence

The historical evidence is already sufficient to justify V1 as the intended
live controller:

| Evidence | Result |
|---|---:|
| Fine-path 2025 baseline | +93.96 net bps/trade |
| Fine-path 2025 V1 | +131.13 net bps/trade |
| Fine-path uplift | +37.17 bps/trade |
| Authoritative fixed-trade baseline | +163.09 net bps/trade |
| Authoritative fixed-trade V1 | +183.63 net bps/trade |
| Fixed-trade uplift | +20.55 bps/trade |
| Dynamic-capacity baseline | 8,453 trades; 14.72/day; +163.09 bps/trade |
| Dynamic-capacity V1 | 8,663 trades; 15.09/day; +181.61 bps/trade |
| Dynamic-capacity uplift | +18.52 bps/trade |
| V1 positive-trade rate | 68.22% |
| V1 worst week | -11.67% |
| V1 Sortino | 0.540 |
| V1 max drawdown | -76.53% |

The drawdown reflects the existing aggressive portfolio sizing. It is not an
exit-controller promotion blocker by itself and must not be repaired by
changing V1 in this workstream.

## Frozen controller contract

Two LightGBM quantile regressors predict:

```text
target: remaining_favorable_from_entry_atr
objective: quantile
alpha: 0.65
```

- F1 uses the ordered 28-field short causal path contract.
- F4 uses the ordered rich path, archetype, entry-trust and score-evolution
  contract sealed in the bundle.

Controller formula:

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
    if disagreement > 1.3607286844944153
    else core
)
```

V1 may change only the trailing activation. The following remain immutable:

```text
stop loss:                 4.15200064332387 ATR
base trailing activation: 2.326224919759605 ATR
fixed trailing giveback:  0.10237198997143725 ATR
timeout:                   12 hours / 48 complete 15-minute bars
entry:                     first 15-minute open at signal close + 1 hour
cost:                      100 bps exactly once
decision clock:            after a completed 1-hour bar
effective from:            next 15-minute bar
side:                      long only
```

The selected activation persists until the next completed-hour decision. A
new decision cannot affect any already-started bar and cannot loosen an earned
trailing lock.

## Current sealed artifacts

Deployable V1 bundle:

```text
data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1
bundle id: 09ae898a734351911ac5
model bundle SHA-256: bd415ec0e32bae9701fc771621ab5d568931bdb1929ed9b4546f660bce28be13
```

Active Strict-R3 inference bundle:

```text
config/strict_r3_inference_bundle_long_20260801_v6_robust21_mc1_d2_adaptive_exit_v1.json
```

The active bundle seals 28 immutable artifacts and 34 runtime-code hashes,
including the V1 model, manifest, builder, validator and runtime code.

Upstream execution order is now frozen as:

```text
Robust-21 telemetry
-> exact frozen MC1_d2 absolute EV
-> +50-bps admission
-> final_score auction
-> SimplePolicyOptimiser parent exit
-> Adaptive Exit V1 activation overlay
```

MC1's exact champion-config SHA-256 is
`b1485219617884dfb1cb9bc7b58bf8faf3c8b1dfa87fa1e38786c2384b0ca8bc`;
integration does not retune it. Robust-21 is not numerically blended into
MC1. It remains causal control/fallback telemetry.

Training receipt:

```text
training window: 2025-11-01 through 2026-07-31 12:00 UTC
purge: 12 hours
eligible states: 80,070
equal-month capped states: 40,000
inner disagreement-gate states: 13,646
F1 final trees: 698
F4 final trees: 371
sampled-row SHA-256: e4322c95e1bb3ca6036225c7dec3e3b1c6d27567d892dab6387d972b9bca41e9
```

Primary implementation:

```text
extreme_price_movements/adaptive_exit_v1.py
extreme_price_movements/strict_r3_shadow_portfolio.py
extreme_price_movements/strict_r3_inference_bundle.py
scripts/build_adaptive_exit_v1_canonical_bundle.py
scripts/validate_adaptive_exit_v1_bundle.py
scripts/run_strict_r3_shadow_cycle.py
scripts/run_strict_r3_hourly_shadow.py
```

Documentation and receipts:

```text
docs/ADAPTIVE_EXIT_V1_CANONICALIZATION_HANDOFF_20260813.md
docs/TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807.md
data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1/run_manifest.json
data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1/correctness_test_report.json
```

Research evidence:

```text
data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4
data_perp/artifacts/authoritative_a5_adaptive_exit_reconciliation_20260813_v4
```

## Work already completed

- The research controller was extracted into a reusable inference module.
- F1/F4 models, feature order, preprocessing medians and the disagreement
  cutoff are serialized.
- The controller is loaded by the hourly shadow cycle.
- Per-position portfolio state persists the active activation, entry context,
  causal score history and latest V1 decomposition.
- The prior activation processes the just-completed bars.
- A decision at hour `h` is stamped effective at `h + 15 minutes`.
- Missing inputs or feature-contract failures fall back to base activation.
- The active Strict-R3 inference validator checks V1 identity, hashes,
  activation-only authority and policy-geometry equality.
- Serialization parity is exact on 254 sealed reference states:
  F1 error 0, F4 error 0, selected-activation error 0.
- The focused tests pass 21/21.
- The Round-2 funnel was discarded and must not be restored.

## Required implementation before live authority

### 1. Add an independent runtime-parity recorder

For every open position at every completed-hour decision, persist one row
containing:

```text
candidate_id
symbol
entry_ts
decision_ts
effective_from
position age
ordered F1 feature vector
ordered F4 feature vector
feature-contract hashes
feature coverage and imputation flags
F1 raw prediction
F4 raw prediction
absolute disagreement
frozen disagreement cutoff
core activation
selected activation
abstention flag
fallback flag and reason
prior activation
trailing armed state
maximum favourable excursion
earned trailing lock before decision
earned trailing lock after decision
bundle ID and hashes
```

The record must be append-only and immutable per hourly checkpoint.

### 2. Add an independent replay comparator

For every completed shadow hour, reconstruct each open-position state from the
same causal 15-minute bars and frozen entry information using a separately
invoked replay path. Compare it with the runtime decision.

Required tolerances:

```text
ordered feature names: exact
feature values: <= 1e-10 absolute error where finite
missing-value mask: exact
F1 raw prediction: <= 1e-6
F4 raw prediction: <= 1e-6
selected activation: <= 1e-6 ATR
effective timestamp: exact
exit-bar identity: 100%
exit reason: exact
net outcome: <= 0.01 bps
```

Do not compare against a reconstruction that imports the live decision output.
The comparator must rebuild inputs from bars and sealed position state.

### 3. Verify open-position state continuity

Every hourly checkpoint must consume exactly the preceding checkpoint's
`next_portfolio_state.json`.

Audit:

```text
prior-state SHA-256
next-state SHA-256
wallet before/after
open candidate IDs before/after
prior and new activation per surviving position
next unprocessed 15-minute timestamp
trailing armed state
MFE/protected lock
entry and timeout timestamps
realized exits and released margin
```

No cycle may fall back to an initial empty/static state after the chain starts.

### 4. Add live feature-availability gates

Report separately for F1 and F4:

```text
rows scored
complete rows
rows using training-median imputation
missing fields
non-finite fields
fallback rows
fallback reasons
worst-field availability
```

Promotion requirement:

```text
F1/F4 contract availability >= 99% of eligible open-position decisions
unexplained fallback count = 0
```

A declared market-data outage may create a fail-closed fallback, but it must be
reported and the affected hour cannot count toward the clean parity streak.

### 5. Preserve portfolio semantics

V1 is an exit-only controller. It must not directly change:

- candidate generation;
- score ranking;
- EV admission;
- A5 trust decisions;
- new-entry auction ordering;
- leverage or margin sizing;
- maximum concurrent positions;
- stop, giveback, timeout or cost.

It may indirectly change later entry count only by changing exit timestamps
and therefore releasing portfolio capacity earlier or later. Any such change
must be attributed explicitly to capacity release.

### 6. Add a controlled authority switch

Implement an explicit mode with exactly these values:

```text
static_baseline
adaptive_shadow
adaptive_live
```

Semantics:

- `static_baseline`: calculate neither adaptive authority nor adaptive exits.
- `adaptive_shadow`: calculate and log V1, but execute the frozen base
  activation.
- `adaptive_live`: execute the V1 activation for the next 15-minute bar.

The default remains `adaptive_shadow` until promotion. Unknown or missing mode
must fail closed to `static_baseline` or stop the cycle; it must never imply
live authority.

Changing to `adaptive_live` requires an explicit, versioned config update and
must not modify model artifacts or thresholds.

## Pre-live shadow gate

Run a new untouched chained shadow interval after the V1 runtime seal.
Pre-existing static-exit shadow checkpoints do not count.

Minimum operational gate:

```text
48-72 consecutive hourly checkpoints
>= 20 open-position V1 decisions
>= 5 resolved exits from V1-observed positions
>= 99% F1/F4 feature-contract availability
0 unexplained fallbacks
0 state-chain discontinuities
0 future-bar reads
0 pre-effective-bar interventions
0 policy-geometry drift
0 cost double application
100% exit-bar identity on independently replayed exits
<= 0.01 bps maximum net-outcome discrepancy
<= 1e-6 raw-prediction and activation discrepancy
```

If natural admission produces too few positions, do **not** lower the EV gate
or inject trades into the canonical portfolio. Continue the shadow interval.
A separate synthetic/parity harness may exercise additional sealed historical
positions, but it cannot replace the minimum real chained observations.

## Live promotion decision

When every pre-live gate passes, update only the authority state:

```text
RESEARCH_CANONICAL = true
LIVE_CANONICAL = true
mode = adaptive_live
```

Do not retrain, retune, change the p80 cutoff, modify features, change the
policy, or rerun winner selection during promotion.

Produce a signed promotion receipt containing:

```text
bundle ID and SHA-256
active inference-bundle SHA-256
runtime-code hashes
shadow interval start/end
hour count
position-decision count
resolved-exit count
feature-availability statistics
fallback audit
prediction/activation parity statistics
exit and net-outcome parity statistics
state-chain audit
explicit promotion timestamp
```

Promotion still requires explicit user approval after the receipt is reviewed.

## Hourly monitoring after live activation

At every completed hour, verify and persist:

### Input parity

- Latest source bar is complete and no future bar is present.
- The feature vector has the exact frozen order.
- Runtime and replay values match within tolerance.
- Entry context remains the frozen entry-time context.
- Score evolution uses only values available by the decision timestamp.

### Model parity

- F1/F4 bundle hashes match the promoted receipt.
- Raw predictions match the independent replay within `1e-6`.
- Disagreement and p80 abstention are identical.
- Selected activation matches within `1e-6 ATR`.
- Decision becomes effective only on the next 15-minute bar.

### Policy parity

- Stop and giveback are unchanged.
- Earned trailing lock never decreases.
- Timeout remains exactly H12.
- Cost is booked exactly once at exit.
- Exit bar, reason and price agree with independent replay.

### Stack isolation

- Candidate, base, consensus, reliability, EV map and A5 outputs remain
  unchanged by the exit controller.
- Admission and initial position sizing are unchanged.
- Any later difference in accepted-entry count is explained solely by capacity
  released by different exit timing.

### Operational metrics

Report cumulatively and for the latest hour/day/week:

```text
open positions observed
V1 decisions
intervention rate
abstention rate
fallback rate
activation distribution
F1/F4 disagreement distribution
exit counts by reason
baseline-counterfactual versus V1 exit time
baseline-counterfactual versus V1 net bps
wallet PnL
drawdown
capacity released/consumed
new trades enabled by changed exit timing
```

## Automatic fail-closed rules

Immediately revert affected positions to the base activation, and prevent new
adaptive authority for the hour, if any of these occurs:

- model or manifest hash mismatch;
- feature-order mismatch;
- missing mandatory source history;
- future timestamp in an input;
- non-hourly decision timestamp;
- effective timestamp other than next 15-minute bar;
- prediction or activation parity breach;
- state-chain discontinuity;
- stop/giveback/timeout/cost mismatch;
- attempted loosening of an earned lock;
- unavailable independent replay comparator;
- unexplained fallback-rate breach.

Do not close positions merely because V1 is unavailable. Continue them under
the frozen base activation and policy unless the existing risk system requires
another action.

After any breach:

```text
LIVE_CANONICAL remains the selected model identity
live authority mode becomes adaptive_shadow or static_baseline
the incident hour is excluded from the clean streak
the clean monitoring gate restarts after repair
```

## Required tests

Retain the existing 21 focused tests and add tests for:

1. independent feature reconstruction on a real sealed position;
2. future-bar mutation invariance;
3. exact entry-context persistence;
4. exact score-history as-of behavior;
5. prior activation governing the just-completed interval;
6. new activation governing only subsequent bars;
7. earned-lock monotonicity when activation rises;
8. fallback to base on F1 failure;
9. fallback to base on F4 failure;
10. fallback to base on bundle-hash drift;
11. state continuity across at least three hourly cycles;
12. exit-bar and net-bps parity for stop, trailing and timeout exits;
13. cost applied once;
14. controller cannot alter admission or entry sizing;
15. adaptive-shadow and adaptive-live produce identical logged decisions;
16. adaptive-shadow executes the baseline activation;
17. adaptive-live executes the selected activation;
18. unknown mode cannot receive authority.

## Reusable validation commands

Validate the V1 serialization:

```bash
python3 scripts/validate_adaptive_exit_v1_bundle.py \
  --bundle-dir data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1 \
  --out data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1/correctness_test_report.json
```

Validate the active Strict-R3 bundle:

```bash
python3 scripts/validate_strict_r3_inference_bundle.py \
  --bundle config/strict_r3_inference_bundle_long_20260801_v5_homogeneous28_a5_b10_exactpolicy_v3_oi_continuity.json \
  --decision-ts <UTC_DECISION_TIMESTAMP> \
  --out <OUTPUT_AUDIT_JSON>
```

Run the focused tests:

```bash
python3 -m pytest -q \
  tests/test_adaptive_exit_v1.py \
  tests/test_strict_r3_shadow_portfolio.py \
  tests/test_strict_r3_inference_bundle.py
```

The existing hourly orchestrator remains:

```bash
python3 scripts/run_strict_r3_hourly_shadow.py --help
```

Use its immutable per-hour output convention and chain each cycle from the
preceding `next_portfolio_state.json`.

## Required artifacts

Produce:

```text
adaptive_exit_runtime_decisions.parquet
adaptive_exit_independent_replay.parquet
adaptive_exit_feature_parity.parquet
adaptive_exit_exit_parity.parquet
adaptive_exit_state_chain_audit.parquet
adaptive_exit_hourly_monitoring.parquet
adaptive_exit_daily_monitoring.parquet
adaptive_exit_shadow_gate_report.json
adaptive_exit_live_promotion_receipt.json
ADAPTIVE_EXIT_V1_LIVE_PROMOTION_REPORT.md
```

Every artifact must include the controller bundle ID, active inference-bundle
hash, decision timestamp and source-data cutoff.

## Terminal decisions

Use exactly one of:

```text
ADAPTIVE_EXIT_V1_SHADOW_GATE_IN_PROGRESS
ADAPTIVE_EXIT_V1_SHADOW_GATE_PASSED_AWAITING_APPROVAL
ADAPTIVE_EXIT_V1_LIVE_PROMOTED
ADAPTIVE_EXIT_V1_FAIL_CLOSED_RUNTIME_PARITY
ADAPTIVE_EXIT_V1_FAIL_CLOSED_DATA_AVAILABILITY
ADAPTIVE_EXIT_V1_FAIL_CLOSED_STATE_CONTINUITY
```

Passing the automated gates does not authorize promotion by itself. The agent
must report `ADAPTIVE_EXIT_V1_SHADOW_GATE_PASSED_AWAITING_APPROVAL` and wait
for explicit user approval before setting live authority.
