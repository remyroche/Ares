# Strict-R3 D2 + Conditional Consensus + MC1_d2 + Adaptive Exit V1

## Canonical long-only inference handover

**Updated:** 2026-08-15  
**Executable schema:** `strict_r3_inference_bundle_v6_robust21_mc1_d2_adaptive_exit_v1`  
**Scope:** Kraken perpetuals, long side only  
**Readiness:** live, Kraken Futures long-only; every fresh hour remains fail-closed  
**Canonical bundle:** `config/strict_r3_inference_bundle_long_20260801_v40_top30_canonical_monitor.json`  
**Canonical bundle SHA-256:** `8efb3edefbd3014e278a7ddbd7b393988113d860a4bfc173683182eb44c9d3b2`

This document is the operational and research entry point for the current
stack. It describes only the active chain, its frozen contracts, its measured
contribution, and the files needed to reproduce or inspect it. Historical
ablations are intentionally not reproduced here.

## 1. Status and authority

The stack is authorized for causal long-only live execution. Every decision is
first produced in an exchange-free deterministic stage and handed to the
hash-bound Kraken executor only after its primary current-hour audit and
runtime checkpoint pass. Independent feature/score/admission and exit replays
are offline audit jobs: they require exact identities and at most 0.01%
numerical delta, but are not allowed to delay a current executable decision.
An offline failure blocks subsequent new entries until repaired and resealed.

The user explicitly authorized a manual promotion override on 2026-08-14.
`config/strict_r3_kraken_live_execution_v18_v40_canonical_monitor.json` is the
order-capable execution contract. It binds the exact v40 inference bundle,
policy, runtime code, promotion receipt, and
`config/strict_r3_kraken_live_activation_authorization_20260815_v7_v40_monitor.json`.
The override covers only the three incomplete evidence-count gates
(continuous hours, accepted entries, and resolved exits). It does not override
freshness, point-in-time feature identity, current admission
identity, long-only scope, policy identity, leverage/margin limits, protective
stop installation, or exit-replay parity. Signals preceding the authorization,
including the 09:00 ID/USD:USD shadow proposal, are explicitly ineligible for
later submission.

The operational clock begins at the UTC hour boundary. The current-hour score and portfolio
decision must finish inside the frozen 900-second entry window. The two
read-only parity audits may finish inside a separate 1,200-second audit window;
they cannot authorize, add, remove, or resize an entry and therefore do not
widen the 900-second execution authority. Delayed archive data is never
appended and used to score an expired decision retrospectively. Missing
primitives reject only the affected timestamp x symbol. If all rows are
unavailable, the hour is persisted as `no_actionable_rows_fail_closed`: it
creates no new entry, while existing SimplePolicyOptimiser/Adaptive Exit V1
positions, exits, and portfolio state continue to advance. This zero-entry
contract passed an end-to-end smoke with two realized exits and an exact
next-state transition.

The orchestration manifest seals start time, completion time, decision age,
the 900-second entry limit, and `completed_within_live_decision_window`. Live
operation requires `--enforce-live-wall-clock` on the orchestration and the
primary current-hour audit. Independent feature/score/admission replay and
exit replay are deliberately outside the live critical path. They run from
persisted point-in-time inputs after execution, require exact row identities
and no more than 0.01% numerical delta, and block subsequent new entries if
they fail. They cannot authorize or delay the current decision. The
source-hole/statebridge attempts through 08:00 UTC remain reconciliation
evidence only. New live decisions use the sealed v40 top-30 contract; earlier
schemas remain immutable historical reconciliation evidence.

The first real schema-v13 execution completed for the 2026-08-14 12:00 UTC
decision. It opened long PIXEL and BB positions on Kraken Futures and installed
an exchange-side reduce-only protective stop for each before committing live
state. The immutable receipts are:

- decision: `data_perp/artifacts/strict_r3_schema_v13_top30_live_chain_20260814T120000Z_v2_append_only_activation`;
- live-hour audit: `data_perp/artifacts/strict_r3_schema_v13_top30_live_hour_audit_20260814T120000Z_v1.json`;
- independent replay: `data_perp/artifacts/strict_r3_schema_v13_top30_current_replay_audit_20260814T120000Z_v1.json`;
- exchange execution: `data_perp/artifacts/strict_r3_schema_v13_top30_live_execution_20260814T120000Z_v1.json`.

After activation, `scripts/bridge_strict_r3_live_to_shadow_state.py` preserves
the adaptive-policy context and unit-wallet portfolio representation but
anchors path-dependent exit state to actual Kraken fills: entry price, ATR,
entry/timeout timestamps, MFE, trailing state, and leverage. Execution v18
hash-binds this bridge. This prevents a hypothetical shadow fill from changing
Adaptive Exit V1 or parent-policy decisions for a real position. When an exit
is proposed, `scripts/audit_strict_r3_shadow_exit_replay.py` must receive that
same immutable bridge through `--entry-state`. It verifies predecessor lineage,
the live-fill overlays, and the exact portfolio-state hash used by the hourly
run before independently replaying frozen 15-minute bars. This replay is an
offline audit: exact exit identity, reason, timestamp and policy plus at most
0.01% numeric delta are required; a failure blocks subsequent new entries but
does not delay the current canonical exit.

Recurring hours invoke `scripts/run_strict_r3_hourly_shadow.py` with
`--portfolio-state-reconciliation`. Unlike one-time activation, this mode uses
the supplied actual-fill bridge, requires its exact predecessor hash and live
ledger hash, and retains the predecessor only for immutable
candidate/feature/prediction history. The live ledger was migrated without
changing either open position or any processed decision; the receipt is
`data_perp/artifacts/strict_r3_live_state_migration_v13_to_v14_20260814_v1.json`.
That receipt is retained as historical lineage; the active state is now the
flat v18/v40 successor state.

New entries additionally require executable-price parity with the replay
contract. Execution v18 fetches the contemporaneous bid/ask and visible ask
book for the intended contract count before submission. It computes:

```text
execution_adjusted_EV
  = frozen_MC1_expected_net_bps
  - adverse_midpoint_gap_since_decision_open
  - max(0, live_round_trip_microstructure_bps - 100)
```

`live_round_trip_microstructure_bps` is the expected entry impact from the
quote midpoint, including ask-book depth for the intended size, plus an exit
half-spread proxy. The frozen MC1 target already deducts 100 bps exactly once,
so only microstructure cost above that baseline is charged again. A favourable
execution-delay gap is not credited. The adjusted EV must remain at least +50
bps, and the live full spread must remain at most 100 bps.

Separately, expected ask-book VWAP and the final market fill may be at most 50
adverse bps above the frozen decision-open price. A worse preflight is a
row-local no-order rejection; if the final fill breaches either the slippage
cap or adjusted-EV floor after a passing preflight, the executor immediately
flattens it and never commits the position. Favorable slippage is retained.
These gates do not retrospectively alter existing positions, whose exit state
is anchored to their actual fills.

PIXEL on the first live hour exposed the missing predecessor of this contract:
the frozen decision open was `0.004545`, while the eventual Kraken fill was
`0.004759`, an adverse gap of `470.85 bps`. The first scoring attempt had
failed at append-only resolved-ledger assembly on a conflicting historical
`final_score`; a second full attempt was then run, and submission occurred at
12:11 UTC. The models themselves took roughly three minutes per attempt, but
the duplicated attempt turned the end-to-end delay into roughly eleven
minutes. Execution v18 would reject the PIXEL quote before order submission.
PIXEL later closed through the legacy reconciler, which had no close-email
hook. The active monitor routes confirmed protective fills and canonical
timeout/trailing closes through the shared close-email producer. BB was closed
by that path on 2026-08-15 and its close email was delivered.

The active authority hierarchy is:

```text
point-in-time Kraken universe and features
→ strict-R3 D2 three-class base
→ timestamp-local top-30% compute route
→ ten policy-residual LambdaRank heads
→ 75% base rank + 25% conditional-consensus rank
→ correctness demotion and same-model prior-28-day CDF
→ final_score
→ Robust-21 control telemetry
→ frozen MC1_d2 absolute expected-net map
→ admit when MC1 EV ≥ +50 bps
→ auction admitted rows by final_score
→ constrained portfolio
→ live spread + order-book impact + execution-delay EV adjustment
→ require adjusted EV ≥ +50 bps and fill drift ≤50 bps
→ SimplePolicyOptimiser parent exit
→ Adaptive Exit V1 activation-only overlay
```

Robust-21 and MC1_d2 are not blended. MC1_d2 owns admission; Robust-21 remains
visible as a causal control and diagnostic. `final_score`, not mapped EV, owns
the auction ordering after admission.

R5, A5, Severe-200, and ordinary-consensus outputs remain available as shadow
diagnostics or controller context. They do not own admission, ranking, sizing,
or the parent stop/giveback geometry.

## 2. Decision and causality clock

For a signal hour ending at `t`:

1. Candidate identity and cross-sectional features are built from the complete
   contemporaneously available frozen universe.
2. Eligibility uses only decision-time instrument availability, official
   spread, executable entry, and feature coverage.
3. The decision is taken at `t + 1h`.
4. Entry is the first exact 15-minute open at `t + 1h`.
5. Training or calibration outcomes are usable only after their path has fully
   resolved and `policy_label_available_ts <= decision_ts`.
6. Current candidate outcomes are never opened by scoring or admission.
7. The position times out after 12 hours if no earlier policy exit occurs.

Future-path completeness is not a candidate criterion. Current spread is
limited to 100 bps. Cross-sectional features are computed before the spread
filter, from the point-in-time market universe. The spread gate is rechecked
against the live executable quote, and execution-delay/impact costs are applied
after the portfolio auction but before any order is sent. Missing required
inputs or insufficient visible ask depth fail closed for that row.

## 3. Layer-by-layer contract

### 3.0 Training, refitting, and calibration schedule

The executable stack deliberately mixes refitted and permanently frozen
components. A new bundle must preserve the schedule below; a component that is
not listed as refitted must not be silently trained again.

| Component | Training or fit population | Schedule | Holdout / purge | Runtime role |
|---|---|---|---|---|
| Strict-R3 D2 base | Most recent resolved pre-cutoff history, capped at 240,000 rows | One lockstep upstream fit for each declared 28-day deployment window | Preceding 28 calendar days excluded from all supervised upstream fits | Opportunity probabilities and `base_score` |
| Base score reference | The new base scores the excluded preceding 28-day target-free reserve | Rebuilt with every upstream fit | Same fitted base as held rows; no held-window ranks | `base_rank42` wire alias; semantically prior-28 rank |
| Policy-net anchor map | Earlier prequential base predictions with resolved SimplePolicy outcomes | Rebuilt with every upstream fit | Reserve and held outcomes excluded; 20-bin isotonic map | `base_anchor_bps` and residual definition |
| Ten residual LambdaRank heads | Valid prequential `policy_net_bps - base_anchor_bps` rows | Rebuilt in the same lockstep upstream bundle | Same preceding 28-day reserve excluded | Conditional consensus |
| Geometry/K9 | 2024-10-01 through 2024-12-31 only | **Never refitted** | January 2025 onward is out of geometry-definition sample | Stable geometry semantics, support and OOD |
| Rule/path reliability and correctness | Resolved pre-cutoff conversion history; active bundle starts 2026-02-01 | One conversion fit for each declared 28-day deployment window | Preceding 28 days excluded; frozen Geometry/K9 reused unchanged | Correctness demotion and final score |
| Final-score reference | New conversion bundle scores its own preceding 28-day reserve | Rebuilt with every conversion fit | Same conversion and upstream models as held rows | Prior-28 CDF of demoted score |
| Robust-21 | Prior-resolved policy outcomes only | Updated causally at every decision | 21-day cell/day window; current unresolved outcomes ignored | Control telemetry only |
| MC1_d2 static mapper | 50,000 deterministic day-balanced rows from 1,231,050 causal-history rows | Frozen champion; not refitted inside schema v6 | Fit cutoff 2026-08-01; six-field order and model hash sealed | Absolute-EV admission authority |
| MC1 recent-global shift | Prior-resolved MC1 residual history | Updated causally at every decision | Trailing 21 days, 10% day-tail trim; `label_available_ts <= decision_ts` | Immediate temporal EV adjustment |
| R5/A5 trust context | Frozen cutoff-matched bundles | Frozen in schema v6 | Hash and cutoff checked | Adaptive context/telemetry only; no admission authority |
| SimplePolicyOptimiser | Pre-2025 policy-development population | Frozen | No live retuning | Parent stop, activation, giveback and timeout |
| Adaptive Exit V1 | 2025-11-01 through 2026-07-31 12:00 UTC; 40,000 equal-month sample | Frozen model/controller | 12-hour purge before 2026-08-01 activation | Activation-only modulation; parent fallback |

The upstream and conversion models are therefore lockstep **window-refit**
components, not models refitted every inference hour. A new fit receives a new
cutoff, expiry, hashes, its own preceding 28-day reserve replay, and a new
immutable bundle. Geometry/K9, MC1_d2, SimplePolicyOptimiser, and Adaptive Exit
V1 remain frozen until an explicitly versioned research promotion replaces
them.

#### Operational refit procedure

`scripts/run_strict_r3_canonical_lockstep_walkforward.py` is the authoritative
base/conversion refit producer; `scripts/train_strict_r3_canonical.py` is the
single-bundle training entry point used by that orchestration. For each new
declared cutoff the operator must:

1. freeze the cutoff and following 28-day deployment interval;
2. materialize target-free candidates/features and only labels resolved before
   the cutoff;
3. reserve `[cutoff - 28 days, cutoff)` from every supervised upstream and
   conversion fit;
4. train the strict-R3 base, policy-net map, ten residual heads, and conversion
   model at the same cutoff while reusing the unchanged Geometry/K9 bundle;
5. score that reserve with the new models to create their own rank/CDF
   references before the first deployment decision;
6. validate identities, feature order, label availability, policy cost, and all
   artifact/runtime hashes, then seal a new immutable inference bundle; and
7. activate it only at its declared boundary. The preceding bundle remains the
   rollback artifact and is never mixed into the new bundle's reference ranks.

Robust-21 telemetry and the MC1 recent-global shift update at decision time
from fully resolved historical policy outcomes; they are calibration-state
updates, not model retraining. MC1_d2, Geometry/K9, SimplePolicyOptimiser, and
Adaptive Exit V1 require an explicit researched, versioned promotion to
change. A failed refit or incomplete reserve fails closed and does not extend
or mutate the outgoing bundle.

The resolved calibration ledger has one immutable vintage per UTC day. The
first cycle at `00:00 UTC` rebuilds it from labels available strictly before
that day. Every later hourly cycle copies the preceding ledger byte-for-byte;
late source repairs cannot change Robust-21, the MC1 global shift, IC, or
correctness telemetry intraday. This rule was sealed after the 2026-08-14
01:00 reconciliation audit detected 144 late-recovered historical AR rows in
an otherwise causal rebuild. That failed checkpoint is retained as evidence
but cannot extend the promotion chain; the repaired chain restarts from a new
immutable hourly version and the promotion clock resets.

### 3.1 Candidate and feature spine

The feature materializer produces the exact frozen 120-field long contract in
the declared order. The fields cover market and asset price state, breadth,
volatility, order-book liquidity, open interest, funding, support/resistance,
cross-asset dependence, liquidation/recovery, and transition context.

The authoritative ordered list is in:

- `config/strict_r3_canonical_v2_feature_contract.json`
- the `base_fields` stored in the monthly upstream bundle
- `data_perp/artifacts/strict_r3_lockstep_successor28_homogeneous28_long_aug1_7_20260813_v1/bundles/cutoff=20260801/upstream/monthly_upstream_bundle.joblib`

Runtime gates:

| Gate | Requirement |
|---|---:|
| Per-row finite feature fraction | at least 90% |
| Complete-row fraction per cycle | at least 90% |
| Per-field finite fraction | at least 90% |
| Reference-window variance | every field must vary |
| Missing/noncausal inputs | fail closed |

The frozen bundle's August audit covered 25,794 identity-matched rows: all-120
completeness was 98.60%, row-level ≥90% coverage was 99.61%, minimum per-field
coverage was 98.65%, and no deployed field was constant.

Relevant code:

- `scripts/materialize_strict_r3_target_free_hourly_grid_v2.py`
- `scripts/materialize_strict_r3_forward_features.py`
- `scripts/backfill_kraken_frozen_contract_inputs.py`
- `extreme_price_movements/features.py`
- `extreme_price_movements/features_oi.py`
- `extreme_price_movements/config.py`

### 3.2 Frozen Geometry/K9 representation

Geometry/K9 is fitted once and is never refitted monthly. This preserves the
meaning of every downstream geometry-derived output.

| Item | Frozen value |
|---|---|
| Definition window | 2024-10-01 through 2024-12-31 UTC |
| Complete warm-up rows | 126,638 |
| Encoder fit rows | 126,638 |
| K9 fit rows | 100,000, equal-month sample |
| Clusters | 9 |
| Encoder | 64-tree target-free geometry encoder |
| Temperature scale in conversion | 0.25 |
| Runtime geometry identity | `dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638` |

The frozen artifact contains imputation medians, leaf categories and support,
cluster centers and order, temperature, input order, dates, seeds, and hashes.
Raw K9 memberships are not fed to the consensus or correctness models because
their direct historical ablation was less portable. Stable aggregates such as
entropy, margin, support, OOD, and path reliability remain available where the
declared contracts use them.

Artifact:

- `data_perp/artifacts/strict_r3_schema_v2_geometry_k9_long_octdec2024_k9weighted_20260811_v1/frozen_geometry_k9.joblib`
- `data_perp/artifacts/strict_r3_schema_v2_geometry_k9_long_octdec2024_k9weighted_20260811_v1/run_manifest.json`

### 3.3 Strict-R3 D2 base

The base predicts a three-state R3 outcome:

| Class | Meaning |
|---|---|
| adverse | meaningful adverse path wins |
| weak | neither robust clear nor adverse resolution |
| clear | robust economic clear before adverse movement |

The live base score is:

```text
base_score = P(clear) − 0.5 × P(adverse)
```

The D2 curriculum gives 1.5× weight to robust-clear rows in the prior
strict-prequential teacher's global top 20%. It does not alter labels or sides.
Weights are projected to mean one and bounded to `[0.25, 4.0]`. The current fit
uses 240,000 resolved rows and reserves the preceding 28 days from every active
upstream supervised fit.

Frozen model:

| Parameter | Value |
|---|---:|
| Model | LightGBM three-class classifier |
| Trees | 220 |
| Learning rate | 0.035 |
| Max depth | 5 |
| Leaves | 24 |
| Minimum child rows | 2,400 |
| Feature fraction | 0.85 |
| L2 | 20 |
| Seed | 20260817 |
| Training cap | 240,000 rows |

Measured base contribution on the frozen research comparison:

| Period | Arm | Rank IC | Log loss | Brier | Top-30 recall | Top-40 recall | Top-5 clear uplift |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026 Jan-Jul | D0 base | 0.1999 | 1.0780 | 0.6553 | 32.01% | 42.56% | 8.06 pp |
| 2026 Jan-Jul | **D2 base** | **0.2004** | **1.0719** | **0.6507** | **32.64%** | **43.16%** | **9.69 pp** |

This is a modest but transported improvement. The base is an opportunity
ranker, not the absolute-EV admission model.

### 3.4 Timestamp-local compute route

Every eligible row receives the base score. Only the top 30% of `base_score`
within the current decision timestamp proceeds to consensus, correctness, and
MC1. Ties use ascending `candidate_id`. The complete point-in-time population
is retained for shared geometry and cross-sectional computation.

Rows below the route are explicitly base-only and fail closed for admission.
The gate saves downstream compute; it is not a historical full-window
percentile and does not change any base prediction.

### 3.5 Causal base anchor

`base_score` is converted to `base_rank42` against the same fitted model's
prior 28-day reserve. Despite the legacy wire name, the active reserve is 28
days. A 20-bin monotonic policy-net map fitted only on earlier prequential
predictions and resolved optimized-policy outcomes produces:

```text
base_anchor_bps = map(base_rank42)
policy_residual_bps = policy_net_bps − base_anchor_bps
```

The reserve is excluded from supervised base/map/consensus fitting, then
rescored by the new bundle. Calibration is therefore available from the first
hour of a refit without mixing score vintages.

### 3.6 Ten conditional policy-residual heads

Ten complementary LambdaRank heads model ordinalized policy residual. Their
target edges are `[-150, -50, +50, +150]` bps, giving five ordered grades.
Each head has a frozen feature subset drawn from the 120-field contract and a
declared query geometry. Six use exact timestamp × side; four use 4-hour UTC ×
side. Feature caps/subsets range from 15 to 120 fields and use ordinary or
equal-month weighting.

Common ranker parameters:

| Parameter | Value |
|---|---:|
| Objective | native LambdaRank / NDCG |
| Trees | 120 |
| Learning rate | 0.035 |
| Max depth / leaves | 5 / 31 |
| Minimum child rows | 300 |
| Feature / bag fraction | 0.82 / 0.82 |
| L1 / L2 | 0.02 / 2.0 |
| Max bin | 127 |
| Label gains | `[0, 0.25, 1, 3, 7]` |
| Truncation | 10 |

Each head is ranked against its own resolved prequential training-score
distribution. Their median is `conditional_consensus_rank`. The upstream blend
is:

```text
upstream = 0.75 × base_rank42 + 0.25 × conditional_consensus_rank
```

Matched full-stack ranking contribution:

| Period | Stack | Top 1% | Top 2% | Top 5% | Worst Top-2 month | Positive Top-2 months |
|---|---|---:|---:|---:|---:|---:|
| 2025 Jan-Jul | ordinary consensus | +127.38 | +108.38 | +74.96 | +56.38 | 7/7 |
| 2025 Jan-Jul | **conditional ten-head** | **+147.35** | **+123.77** | **+88.02** | **+66.93** | **7/7** |
| 2026 Jan-Jul | ordinary consensus | +129.38 | +88.99 | +39.84 | **+44.08** | 7/7 |
| 2026 Jan-Jul | **conditional ten-head** | **+133.73** | **+93.72** | **+42.11** | +32.30 | **7/7** |

These are pooled global-tail ranking diagnostics, not executable admissions.
The conditional heads improve pooled Top-1/2/5 in both periods; their 2026
worst-month result is weaker, which is why downstream calibration and control
telemetry remain important.

The exact head definitions live in the upstream bundle and the frozen
conditional-consensus contract loaded by
`extreme_price_movements/strict_r3_canonical_current.py`.

### 3.7 Correctness demotion and final score

The correctness head asks whether optimized-policy residual exceeds +100 bps.
It is trained only on the top 30% of the prequential upstream training score,
with 4-hour UTC × side LambdaRank queries and 64 declared causal fields.

| Parameter | Value |
|---|---:|
| Trees | 120 |
| Learning rate | 0.035 |
| Depth / leaves | 4 / 15 |
| Minimum child rows | 1,889 |
| Feature / bag fraction | 0.80 / 0.82 |
| L1 / L2 | 0.05 / 5.0 |

For rows above its training-domain floor, it can only demote:

```text
multiplier = 0.25 + 0.75 × correctness_rank
raw_correctness_demote = upstream × multiplier
```

Outside that domain the multiplier is one. `final_score` is the CDF of the
demoted score against the same conversion model's prior 28-day reference.
There are no held-window percentiles.

Severe-200 remains a shadow diagnostic (`exact H12 TP6/SL4 net <= -200 bps`)
and does not affect `final_score` in schema v6.

### 3.8 Robust-21 telemetry and frozen MC1_d2 admission

Robust-21 computes a causal 21-day cell/day robust expected-net control. It is
reported for monitoring but has no numerical authority in the active decision.

The schema-v6 JSON still seals the prior schema-v5 EV-bridge file for artifact
continuity, but declares it
`inactive_schema_v5_compatibility_not_used_by_schema_v6`. Its legacy score
family does not match the timestamp-local routed score domain and it is never called
by the schema-v6 admission runner. MC1_d2 plus its causal recent-global shift is
the executable calibration authority.

MC1_d2 maps six already-causal score and agreement fields directly to expected
optimized-policy net bps:

1. `final_score`
2. `base_rank42`
3. `conditional_consensus_rank`
4. `upstream`
5. `ordinary_shadow_consensus_rank`
6. `correctness_rank`

Frozen MC1 model:

| Parameter | Value |
|---|---:|
| Model | HistGradientBoostingRegressor |
| Depth | 2 |
| Iterations | 80 |
| Learning rate | 0.04 |
| L2 | 20 |
| Minimum leaf | 100 |
| Seed | 1729 |
| Training rows | 50,000 deterministic day-balanced sample |
| Causal history rows | 1,231,050 |

It adds a causal recent-global residual shift computed over 21 days, trimming
10% of days from each tail. A label enters that shift only when its policy
outcome is resolved by the decision timestamp.

```text
admit = mc1_d2_expected_net_bps >= +50
auction order = final_score descending
```

MC1 mapper contribution in the 2026 constrained frozen-rank replay:

| Arm | Trades | Trades/day | Net bps/trade | Net sum bps | Positive weeks | Worst week | Sortino | Max MTM DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Robust-21 control | 2,761 | 13.22 | +127.48 | +351,981 | 24/26 | −56.4 | 0.460 | −65.0% |
| **MC1_d2** | **3,855** | **18.19** | **+155.15** | **+598,095** | **31/31** | **+1.3** | **0.755** | **−38.5%** |
| Delta | +1,094 | +4.97 | **+27.67** | **+246,114** | +7 weeks | +57.7 | +0.295 | +26.5 pp |

Admission provenance strengthens the mechanism:

| Cohort | Rows | Net bps/trade | Total net bps |
|---|---:|---:|---:|
| Shared MC1 + Robust-21 | 10,242 | +179.62 | +1,839,670 |
| MC1-only additions | 8,492 | +144.87 | +1,230,269 |
| Robust-21-only omissions | 18,486 | +13.05 | +241,232 |

Agreement remains informative within frozen-score bands: its mean
agreement-to-EV Spearman is approximately +0.76, positive in 9/10 score bands.
The result survives depth/seed/leaf-floor and leave-one-month-out checks, but
2026 participated in champion selection. A later frozen forward period remains
the required production-promotion evidence.

Frozen files:

- `config/strict_r3_mc1_d2_research_champion_20260813_v1.json`
- `data_perp/artifacts/strict_r3_mc1_d2_canonical_long_20260801_v1/mc1_d2.joblib`
- `data_perp/artifacts/strict_r3_mc1_d2_canonical_long_20260801_v1/run_manifest.json`

Champion config SHA-256:
`b1485219617884dfb1cb9bc7b58bf8faf3c8b1dfa87fa1e38786c2384b0ca8bc`.

Model SHA-256:
`6558d8c33a72feb4d06bd8145a3b40b10ed171ac16a22987116a4d280afb17c8`.

### 3.9 Portfolio auction and sizing

Only MC1-admitted rows enter the auction. The auction is long-only and ordered
by `final_score`.

| Constraint | Value |
|---|---:|
| Maximum open positions | 8 |
| Maximum per symbol | 1 |
| Maximum new entries per hour | 2 |
| Maximum total margin | 80% of wallet |
| Margin per slot | 10% of wallet |
| Leverage | 7× |
| Minimum notional | 1 |

The portfolio state is append-only and passed exactly from one hourly cycle to
the next. Positions are advanced through completed 15-minute bars before new
entries are auctioned.

Contract:

- `config/strict_r3_robust21_mc1_d2_portfolio_v1.json`
- `extreme_price_movements/strict_r3_shadow_portfolio.py`

### 3.10 SimplePolicyOptimiser parent exit

The canonical parent policy was selected before the active forward period by
`simple_policy_optimiser`. It is not a hard-coded TP/SL replay.

| Element | Value |
|---|---:|
| Stop loss | 4.1520006433 ATR |
| Trailing activation | 2.3262249198 ATR |
| Trailing giveback | 0.1023719900 ATR |
| Timeout | 12 hours |
| Cost | 100 bps round trip, once |
| Entry | first 15-minute open at signal close + 1 hour |
| Bar ordering | stop; prior-bar trailing; current-bar MFE update |

Artifact:

- `data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json`

### 3.11 Adaptive Exit V1

Adaptive Exit V1 is **not a standalone exit policy**. It sits on top of the
SimplePolicyOptimiser policy and has authority over trailing activation only.
Stop, giveback, timeout, entry, and 100-bps cost remain unchanged.

At each completed hourly bar it uses the position's causal 15-minute path state,
entry-time stack context, and recent prediction state. Its selected activation
becomes effective from the next 15-minute bar. The F4 controller abstains when
its disagreement exceeds the frozen p80 threshold; missing or unsupported
inputs fall back exactly to the parent activation.

| Item | Value |
|---|---|
| Target | remaining favorable excursion from entry, ATR-normalized |
| Loss | quantile 0.65 |
| Training window | 2025-11-01 through 2026-07-31 12:00 UTC |
| Eligible/sample rows | 80,070 / 40,000 equal-month |
| Purge | 12 hours |
| Controller | F4 disagreement-abstain p80 |
| Activation shrink | 0.75 |
| Allowed activation ratio | 0.50× to 1.25× parent |
| Disagreement p80 | 1.3607286845 |

Matched constrained outcome contribution, on the same MC1-selected sources:

| Period | Exit | Trades | Net bps/trade | Delta | Net bps/day | Delta | Sortino | Delta | Max DD | Ulcer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025–Jul 2026 | SimplePolicy parent | 10,895 | +171.58 | — | +3,239.75 | — | 0.632 | — | −85.76% | 9.96 |
| 2025–Jul 2026 | **Parent + Adaptive V1** | **11,028** | **+179.48** | **+7.90** | **+3,430.33** | **+190.58** | **0.676** | **+0.044** | −85.76% | **8.81** |
| 2025 | Parent + Adaptive V1 | 7,117 | +188.47 | +8.05 | — | — | 0.754 | +0.036 | −85.76% | 9.52 |
| 2026 Jan-Jul | Parent + Adaptive V1 | 3,911 | +163.11 | +7.70 | — | — | 0.573 | +0.051 | −38.47% | 7.44 |

Only 14,771 of 66,190 MC1-admitted historical candidates (22.32%) had exact
OOF Adaptive state coverage. Unsupported rows retain the parent-policy result
exactly. The uplift is therefore conservative but remains research evidence,
not untouched forward proof.

Artifacts and code:

- `data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1`
- `extreme_price_movements/adaptive_exit_v1.py`
- `docs/ADAPTIVE_EXIT_V1_LIVE_PROMOTION_HANDOFF_20260813.md`

## 4. Inference readiness verification

### 4.1 Frozen bundle validation

`StrictR3InferenceBundle.validate` at 2026-08-13 12:00 UTC verified:

| Check | Result |
|---|---:|
| Immutable artifact hashes | 28 / 28 |
| Runtime code hashes | 34 / 34 |
| Frozen Geometry/K9 identity | matched |
| Scope | long-only shadow |
| Exchange I/O | disabled |

Focused inference tests: **27 passed, 0 failed**.

### 4.2 Full saved-market rehearsal

The schema-v6 chain was replayed end to end on the saved real 2026-08-13 12:00
UTC snapshot:

| Rehearsal measure | Result |
|---|---:|
| Current candidate rows | 103 |
| Complete 120-feature rows | 103 / 103 |
| Base top-20% routed/mapped rows | 21 |
| MC1 admissions | 1 |
| Portfolio acceptances | 1 |
| New open positions | 1 |
| Next-state timestamp | 2026-08-13 13:00 UTC |
| Exchange calls | 0 |
| Runtime invariant checks | 16 / 16 passed |

The rehearsal confirmed target-free scoring, no held percentiles, same-model
reference replay, frozen geometry, prior-resolved labels only, current outcomes
absent, cost applied once, complete MC1 mapping for routed rows, and exact
portfolio-state persistence.

The timestamp-local compute gate exposed and led to a repaired resolved-ledger edge case:
historical rows that were previously fully scored now retain their immutable
resolved values when a prefix replay stops them after the base. Recomputed
routed rows must still match score and lineage exactly; base-only rows cannot
enter calibration with a null score.

Receipt:

- `agents/receipts/20260814_schema_v6_inference_readiness.json`

### 4.3 Readiness matrix

| Component | Present | Frozen/versioned | Runtime-usable | Fail-closed |
|---|---:|---:|---:|---:|
| Frozen universe and candidate rules | yes | yes | yes | yes |
| 120-field feature contract | yes | yes | yes | yes |
| Strict-R3 D2 base | yes | yes | yes | yes |
| Same-model 28-day reserve | yes | yes | yes | yes |
| Policy-net base map | yes | yes | yes | yes |
| Ten conditional residual heads | yes | yes | yes | yes |
| Frozen Geometry/K9 | yes | yes | yes | yes |
| Correctness demotion | yes | yes | yes | yes |
| Robust-21 telemetry | yes | yes | yes | yes |
| Frozen MC1_d2 authority | yes | yes | yes | yes |
| +50-bps admission | yes | yes | yes | yes |
| Portfolio policy/state | yes | yes | yes | yes |
| SimplePolicy parent exit | yes | yes | yes | yes |
| Adaptive Exit V1 | yes | yes | yes | parent fallback |
| Exchange order submission | **yes** | execution v18 / bundle v40 | **yes** | cost-adjusted, primary-audit-gated and fail-closed |
| Persistent position monitor | **yes** | execution v18 / bundle v40 | **yes** | minute wake-up; canonical completed-15-minute policy state |

Conclusion: the complete chain is authorized for reproducible inference and
Kraken Futures long-only execution. The decision producer remains
exchange-free by design; only the separately hash-bound executor can submit
orders after the primary current-hour audit passes inside the freshness
window. Independent replay receipts are offline verification evidence and do
not sit in front of a live order.

### 4.4 Matched May–July 2026 executable-contract replay

The canonical replay was regenerated after enforcing the inference-only
timestamp-local top-20 base route. The older source-aligned replay let every
candidate reach downstream computation and is retained only as a research
comparison. The matched replay uses:

```text
top 20% base score per timestamp
→ frozen MC1_d2 EV >= +50 bps
→ final_score auction
→ 8 positions / 2 entries per hour / 80% margin / 7x
→ frozen SimplePolicyOptimiser
→ Adaptive Exit V1 where exact historical controller state exists
→ parent policy otherwise
```

Artifact:

- `data_perp/artifacts/strict_r3_mc1_adaptive_exit_canonical_route20_mayjul2026_20260814_v4_currentbound`
- independent verification replay:
  `data_perp/artifacts/strict_r3_mc1_adaptive_exit_canonical_route20_mayjul2026_20260814_v5_independent_verification`

This replay is cryptographically bound to the active schema-v6 inference
bundle. Before replay it validates all 28 artifact hashes and all 39 sealed
runtime-code hashes, then verifies that the parent exit policy and Adaptive
Exit model/manifest are the exact files named by the bundle. Its manifest
records the active bundle SHA, upstream/conversion/Geometry identities, and
the 28-day same-model plus 21-day MC1 calibration contracts. All nine data
outputs are byte-identical to the preceding matched replay; only the manifest
was strengthened.

The current-bound replay regenerated all nine outputs byte-identically to the
preceding `v3_bound` replay while binding the manifest to schema-v6 bundle SHA
`85739214ca748ae1d1fe040b67f35d434c382e1a2941ea5c46fe3832dd1ae509`.
The bundle reseal therefore changed provenance receipts only, not candidates,
scores, admission, exits, portfolio decisions, equity, or metrics.

The independent `v5` verification reran the executable route from the sealed
inputs rather than copying the prior artifact. It again reproduced all nine
`v4` data outputs byte-for-byte. The bundle validator verified 28 artifact
hashes and 39 runtime-code hashes. A separate reserve audit found 444,445
May--July calibration-reference rows; every row was inside its declared
preceding 28-day window and marked out-of-sample to every active supervised
fit. Targeted inference-bundle, wiring, cell/day-admission, and forward-
admission tests passed 18/18. No patch to model, calibration, policy, or replay
logic was required.

The active August model is not retroactively applied to May–July. Those rows
retain their causal historical 28-day walk-forward vintages: each calendar
month intersects two upstream and two conversion bundles, while all three
months retain one identical frozen Geometry/K9 identity. The active-bundle
binding proves the current runtime and shared frozen policy/mapper/exit
artifacts; the historical ledger proves the period-appropriate model lineage.

| Month | Exit arm | Trades | Trades/day | Net bps/trade | Net bps/day | Sortino | Max MTM DD | Worst week EV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| May | SimplePolicy parent | 623 | 20.10 | +107.02 | +2,150.77 | 0.395 | -39.46% | +15.24 |
| May | Parent + Adaptive V1 | 635 | 20.48 | +109.01 | +2,232.99 | 0.396 | -39.46% | +15.24 |
| June | SimplePolicy parent | 363 | 12.10 | +30.79 | +372.51 | 0.217 | -26.15% | -7.91 |
| June | Parent + Adaptive V1 | 364 | 12.13 | +32.17 | +390.32 | 0.226 | -26.15% | -7.91 |
| July | SimplePolicy parent | 449 | 14.48 | +141.31 | +2,046.75 | 0.258 | -25.22% | +101.60 |
| July | Parent + Adaptive V1 | 449 | 14.48 | +141.31 | +2,046.75 | 0.258 | -25.22% | +101.60 |
| **May–July** | **SimplePolicy parent** | **1,435** | **15.60** | **+98.47** | **+1,535.85** | **0.325** | **-39.46%** | **-7.91** |
| **May–July** | **Parent + Adaptive V1** | **1,448** | **15.74** | **+99.71** | **+1,569.36** | **0.327** | **-39.46%** | **-7.91** |

Relative to the former ungated research replay, the executable gate changes
monthly Adaptive net EV/trade by `-1.25` bps in May, `+2.32` bps in June, and
`-14.22` bps in July. It removes 9, 12, and 28 accepted trades respectively.
The delta is explained by the newly canonical compute route, not by a model,
calibration, policy, cost, or portfolio change. The three-month result remains
strongly positive, with no zero-trade day and one day below five trades.

## 5. How to run and inspect the chain

### Hourly orchestration

- Every decision hour starts with one mandatory, atomic public-source refresh
  for the frozen 170-symbol manifest. The cycle may not start unless all three
  refreshes complete:
  1. `scripts/download_kraken_15m_hf.py` through the exact decision open;
  2. `scripts/backfill_kraken_frozen_contract_inputs.py` through the completed
     signal hour, including official trade, mark, OI and order-book analytics;
  3. `scripts/backfill_kraken_historical_funding_rates_api.py` through the
     decision timestamp.
  The funding step is independent of the mark/OI backfill. Omitting it can
  exhaust the two-bar causal funding carry and make
  `post_liquidation_rebound_score` unavailable across the full cross-section.
  Missing data then skips only the affected timestamp×symbol row; a
  market-wide missing frozen field fails the cycle closed.
- Entry availability uses the timestamp-exact Kraken trade-candle open. The
  raw 15-minute cache is authoritative, the shared 15-minute cache fills only
  missing timestamps, and the official Kraken one-hour trade-candle `open`
  is the final missing-cell fallback. The fallback consumes no high, low,
  close, volume, completed-hour, or future-path value and never carries an
  open from another timestamp. On the 2026-08-14 05:00 reconciliation hour,
  the lagging 15-minute endpoint exposed only 10 of 170 opens while the
  official hourly endpoint exposed all 170; the 10 overlapping opens matched
  exactly. Policy ATR uses signal-time Wilder14. Kraken-omitted no-trade
  15-minute intervals are causally materialized as flat OHLC bars from the
  last observed close through signal time; this rule never applies to
  spreads. The final six-hour v12 replay retained 160–163 fully feature-ready
  rows per hour. Persistent spread unavailability is confined to six
  delisted contracts; other rows are rejected only for the exact current-hour
  spread cap or frozen-feature availability.
- `scripts/run_strict_r3_hourly_shadow.py` — materializes point-in-time inputs
  and calls one immutable cycle.
- `scripts/run_strict_r3_shadow_cycle.py` — canonical score, label-ledger,
  admission, auction, state, and adaptive-exit composition.

### Scoring and conversion

- `scripts/score_strict_r3_forward.py` — thin CLI over the canonical scorer.
- `extreme_price_movements/strict_r3_canonical_current.py` — strict-R3 base,
  maps, consensus, geometry, correctness, and final-score implementation.
- `scripts/assemble_strict_r3_runtime_resolved_ledger.py` — append-only,
  prior-resolved calibration state.
- `scripts/materialize_strict_r3_frozen_policy_labels_v2.py` — exact resolved
  optimized-policy labels, never current outcomes.

### Admission and portfolio

- `scripts/admit_strict_r3_mc1_forward.py` — Robust-21 telemetry and MC1_d2
  authority.
- `extreme_price_movements/strict_r3_mc1_mapper.py` — frozen MC1 mapper.
- `extreme_price_movements/strict_r3_cell_day_admission.py` — Robust-21 control.
- `extreme_price_movements/strict_r3_shadow_portfolio.py` — portfolio auction,
  state transitions, parent exit, and Adaptive V1 integration.

### Validation and reporting

- `scripts/replay_strict_r3_mc1_adaptive_exit.py`
- `scripts/finalize_strict_r3_mc1_adaptive_exit_report.py`
- `scripts/audit_strict_r3_schema_v6_live_hour.py` — exact hourly feature,
  layer-output, policy, admission, append-only, and daily-calibration audit;
  includes rolling 3/7/14/28-day IC and hit-rate telemetry.
- `scripts/report_strict_r3_live_candle.py` and
  `scripts/run_strict_r3_live_hourly_report_loop.sh` — independent read-only
  `xx:10` UTC per-candle funnel, source, execution, and position-monitor
  reports.  They initiate, but never replace, the root-cause repair protocol
  in `docs/STRICT_R3_LIVE_OPERATIONS_REMEDIATION_PROTOCOL.md`.
- `scripts/audit_strict_r3_schema_v6_current_replay.py` — hourly fast mode
  consumes the immutable point-in-time 120-feature artifact, independently
  re-scores the complete frozen K9/R3/consensus prefix, and independently
  re-runs trust plus Robust-21/MC1 admission. Every identity must match and
  every numeric output must be within 0.01%. Full source-to-feature
  reconstruction is deliberately opt-in with `--rebuild-inputs`; it is used
  for explicit feature-generation audits rather than duplicated every hour.
  This is an offline audit. A failure blocks subsequent new entries until the
  discrepancy is repaired and resealed, but never delays the current live
  decision.
- `scripts/audit_strict_r3_shadow_exit_replay.py` — independently rebuilds
  each exit from the immutable predecessor state and frozen 15-minute path;
  it is likewise offline and observational.
- `scripts/run_strict_r3_live_position_monitor.py` — persistent unattended
  exchange-writing monitor. It wakes every minute, advances positions only on
  newly completed 15-minute bars through the same canonical policy engine,
  confirms protective fills, tightens protective stops, submits reduce-only
  canonical exits, persists state, and sends close notifications.
- `scripts/run_strict_r3_live_position_monitor_loop.sh` — stable launcher for
  the persistent monitor; imports and immutable contracts are initialized once
  rather than rebuilt every minute.
- `scripts/audit_strict_r3_schema_v6_promotion_chain.py` with
  `config/strict_r3_schema_v6_live_promotion_20260814_v10_audit1200.json` —
  cumulative live-review evidence, the distinct 900-second inference and
  1,200-second audit deadlines, and explicit promotion gates.
- `extreme_price_movements/inference/canonical_stack_reporting.py` — single
  reporting-only field contract and display semantics shared by trade emails,
  persisted entry context, and the daily report.
- `extreme_price_movements/inference/run_inference.py` — plain-text and HTML
  trade-open/trade-close emails.
- `extreme_price_movements/inference/daily_reporter.py` — daily canonical-layer
  coverage, admission, trust, and exit-fallback recap.
- `tests/test_strict_r3_inference_bundle.py`
- `tests/test_run_strict_r3_shadow_cycle.py`
- `extreme_price_movements/tests/test_canonical_stack_reporting.py`
- `extreme_price_movements/tests/test_daily_reporter.py`
- `tests/test_assemble_strict_r3_runtime_resolved_ledger.py`
- `tests/test_strict_r3_shadow_portfolio.py`
- `tests/test_adaptive_exit_v1.py`
- `tests/test_strict_r3_live_auditors.py`

The hourly heartbeat is active under
`strict-r3-canonical-live-hourly-verification`. It may submit only fresh
current-hour portfolio-accepted longs after the primary audit passes. Offline
parity runs after the live decision from persisted inputs. The persistent
minute monitor is independently active under execution v18 and is limited to
managing already-open positions. The normal 168-hour/30-entry/20-exit
statistics continue to be recorded as evidence even though the user explicitly
overrode their minimum counts for activation.

The exchange adapter is protected by the hash-bound v7/v40 manual-override gate.
Its canonical value is `order_submission_authorized: true`, but that flag alone
cannot authorize a trade: an order-capable run also requires the exact manual
authorization, fresh inference and audit receipts, and an explicitly
initialized live state. A shadow receipt cannot grant exchange authority or
transfer hypothetical shadow positions. The reviewed authorization binds the
inference bundle, exit policy, and promotion-audit receipt. Only then may
`scripts/initialize_strict_r3_kraken_live_state.py` verify that the Kraken
account is flat and create a new state containing those three identities. Live
execution rejects a missing, stale, differently authorized, or pre-authorization
state.

The active v40 runtime consumes only its hash-bound persisted feature state,
the same 111,710-row reference-score coordinate, append-only frozen-K9
history, and rolling 3/7/14/28-day telemetry. Full source reconstruction is an
explicit offline audit mode; it is not duplicated in the live critical path.

An order-capable hourly invocation must provide the already-completed primary
current-hour live audit. The executor verifies run identity, bundle identity,
the admitted set, zero outcome consumption, and the runtime checkpoint before
contacting Kraken. Independent full-stack and exit replay audits run offline
from the persisted point-in-time contract. They require exact identities and
at most 0.01% numerical delta; a discrepancy suspends later new entries rather
than withholding or delaying the current canonical action.

For accepted entries, live sizing preserves the exact canonical auction margin
fraction and verifies that the persisted gross-notional-to-margin ratio equals
the frozen 7x leverage. That fraction is applied to current Kraken wallet
equity and may only be reduced by remaining capacity under the 80% margin cap;
it cannot be replaced by an unrelated fixed notional.

An exchange-side protective stop may fill before the following hourly cycle.
That normal event is reconciled only when the same audited hour contains the
matching canonical exit, Kraken confirms the position's bound stop order is
fully filled, and absolute fill-versus-replay slippage is at most 50 bps. The
state is then closed without submitting a duplicate market exit. A missing
position without all three proofs still fails closed.

Kraken order sizing is contract-aware: requested contracts equal live quote
notional divided by `price × contractSize`, and persisted gross notional uses
the actual fill, filled contracts, and that same contract size. A read-only
2026-08-14 market audit found all 280 active Kraken perpetuals currently use
`contractSize = 1.0`; the generalized calculation prevents a future metadata
change from silently changing exposure.

Before an entry order is placed, Kraken must confirm that the requested
leverage operation succeeded at exactly the frozen 7x value. An unavailable
leverage setter, rejected request, market-cap fallback, or lower confirmed
leverage fails closed; the generic executor's best-effort fallback is not
accepted by this canonical adapter.

The entry response is enriched from Kraken and must be terminal with a finite,
positive fill before state/notional construction. If that check fails after an
order was submitted, the adapter cancels any remainder, refreshes the filled
amount, reduce-closes that amount, and fails the cycle rather than leaving an
untracked or unprotected partial position.

### Operational email and daily-report contract

Every trade-open and trade-close email now contains six explicit layer
sections sourced from the persisted causal decision record:

1. R3 base score, same-model rank, policy-net anchor, route state, and feature
   coverage;
2. frozen K9 entropy/margin, OOD, active support, timestamp covariance and
   correlation breaks, and the geometry bundle identity;
3. conditional and ordinary consensus ranks, correctness state, upstream blend,
   and final score;
4. rule/path/model support and OOD plus the posterior trust/risk outputs;
5. Robust-21 control EV/support and frozen MC1_d2 EV, global shift, availability,
   +50-bps admission, and auction rank;
6. the SimplePolicyOptimiser stop, activation, giveback, timeout, and one-time
   cost, followed by the Adaptive Exit V1 decision and bundle identity.

Adaptive Exit V1 is always described as an activation-only modulator of the
SimplePolicyOptimiser parent. If its inputs or bundle are unavailable, if it
abstains, or if it has not yet been evaluated, the email explicitly identifies
the SimplePolicyOptimiser fallback; no adaptive value is inferred or silently
imputed.

The daily report summarizes layer-field coverage, MC1 expected-EV dispersion,
Robust-21 and MC1 admission counts, trust corroboration, and Adaptive Exit
evaluation/fallback counts. The hourly decision producer preserves these fields
before the entry record is created, and entry provenance carries them to the
close email. This reporting module is observational only and cannot alter a
score, admission, size, or exit.

## 6. Runtime invariants and rejection reasons

Every hourly cycle must fail closed if any of these are false:

- the inference bundle is inside its declared activation window;
- every immutable artifact and runtime code hash matches;
- the candidate population is target-free;
- cross-sectional features are point-in-time;
- at least 90% of the cycle has the complete base contract;
- same fitted upstream/conversion bundles score held and reference rows;
- the Geometry/K9 identity equals the frozen hash;
- no held-window percentile operation is present;
- resolved calibration labels predate the current UTC day;
- current outcomes are absent from predictions and admission;
- every routed row receives MC1 expected net;
- the 100-bps cost is applied exactly once;
- portfolio state timestamp and identity match the prior cycle;
- Adaptive Exit uses completed bars and becomes effective only on the next bar;
- exchange calls and order submission remain zero in shadow mode.

Common explicit rejection states are: incomplete base contract, below top-30%
base route, missing MC1 component, MC1 EV below +50 bps, duplicate symbol,
entry-rate limit, concurrency limit, margin cap, missing exact entry bar, or
missing Adaptive inputs with fallback to the parent exit.

## 7. Evidence classification and research rules

The metrics in this document have different evidentiary roles:

- base and consensus tails are matched pooled ranking diagnostics;
- MC1 2026 was final validation/model-selection evidence, not untouched proof;
- adaptive-exit metrics are constrained matched replays with conservative OOF
  state coverage;
- the 2026-08-13 schema-v6 rehearsal proves executability and causality, not
  economic promotion;
- a later frozen forward period is required before exchange authorization.

Research on one layer must preserve all downstream contracts unless the new
experiment is explicitly versioned. In particular, do not silently change the
120-field order, Geometry/K9 bundle, residual target, consensus head identities,
MC1 feature order, +50-bps threshold, auction score, portfolio constraints, or
parent exit while comparing another component.

Useful focused research entry points:

| Question | Primary artifact/code |
|---|---|
| Base target or curriculum | `extreme_price_movements/strict_r3_canonical_current.py`, monthly upstream bundle |
| Feature contract | `config/strict_r3_canonical_v2_feature_contract.json` |
| Residual-head complementarity | conditional head specs in monthly upstream bundle |
| Geometry/support/OOD | frozen Geometry/K9 artifact and conversion bundle |
| Correctness demotion | four-week conversion bundle |
| Absolute-EV calibration | frozen MC1 bundle and `strict_r3_mc1_mapper.py` |
| Admission control comparison | `strict_r3_cell_day_admission.py` |
| Portfolio/sizing | `strict_r3_robust21_mc1_d2_portfolio_v1.json` |
| Exit behavior | SimplePolicy winner and Adaptive Exit bundle |

## 8. Canonical identities

| Component | Identity |
|---|---|
| Active inference bundle v40 | `8efb3edefbd3014e278a7ddbd7b393988113d860a4bfc173683182eb44c9d3b2` |
| Upstream bundle | `8d8139b166dc0af69815247e2abab1999a6670b0d7cb5552a485dd7ea0006a0e` |
| Conversion bundle | `094b26e5fe9a18b0696d444f553b318be8b3cea6b1c0f5c43a01e84347a08fe7` |
| Runtime Geometry/K9 | `dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638` |
| Feature contract | `12672f92789107fab4c9ab76a20c0c6504e8adce215b4a7f3fc83171dc5705c4` |
| MC1 champion config | `b1485219617884dfb1cb9bc7b58bf8faf3c8b1dfa87fa1e38786c2384b0ca8bc` |
| MC1 model | `6558d8c33a72feb4d06bd8145a3b40b10ed171ac16a22987116a4d280afb17c8` |
| Adaptive Exit model | `bd415ec0e32bae9701fc771621ab5d568931bdb1929ed9b4546f660bce28be13` |
| SimplePolicy winner | `2dc9a145766ae383a4ab7c33e8a9f9e358175597e05582300ff0a05732673603` |
| Portfolio policy | `cffb9e9fe3ff1f3509fe421227a0cd04944e12303049ca276f481f93b8b13cf4` |
| Kraken execution contract v18 | `e92cc14b85a511dd049100c1924a3fe0b7eb7489a41f6afb8ca304b34861965f` (order authority true) |
| Live authorization v7/v40 | `21a1ba0754136c0ac7088d25ae7c1dac57b93f21d26c7d0fe8af10d648ec2c24` |

These identities, plus the immutable monthly bundle and state hashes emitted by
each run manifest, define the reproducible inference contract.
