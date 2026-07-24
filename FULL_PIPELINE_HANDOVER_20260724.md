# Ares Pipeline Handover

Date: 2026-07-24
Repository root: `/Users/remyroche/Documents/Ares`
Storage/model timezone: UTC
Display timezone: Europe/Paris (CEST in summer)

## 1. Executive Summary

This handover captures the exact state of the Ares research pipeline after all
long-running work was stopped for a laptop migration.

The current research objective is to preserve the causal base and residual-meta
alpha stack, then add three downstream sources of execution information:

1. Five side-aware LightGBM auxiliary path heads.
2. A CatBoost future-path archetype classifier.
3. A direct-versus-residual execution-EV model that combines alpha, auxiliary
   predictions, and CatBoost outputs.

The base model, top-30/top-40 handoffs, residual-alpha OOF predictions, path
labels, execution-EV labels, and alpha-to-execution OOF stream exist.

The downstream work is not complete:

- The five full auxiliary-head models and their OOF predictions do not exist.
- The final seven-class CatBoost classifier and its OOF predictions do not
  exist.
- The strict joined execution-EV handoff does not exist.
- The execution-EV direct/residual ablation has not been run.
- None of the new path/execution models has been promoted to replay, policy, or
  production inference.

All training processes were stopped. No CatBoost, auxiliary-head, or related
test process should be assumed to be running.

### Critical Migration Warning

A Git clone is not sufficient to resume this work.

The worktree contains a large number of modified files and the new path,
CatBoost, and execution-EV implementation is currently untracked by Git. The
feature store, fitted models, OOF predictions, label artifacts, HPO studies, and
checkpoints are also outside Git.

Transfer the complete working tree and the required `data_perp` paths listed in
Section 7. Preserve the same absolute path if possible:

```text
/Users/remyroche/Documents/Ares
```

Several manifests and checkpoints contain this absolute path. If the repository
moves, migrate those paths deliberately before resuming. Do not use broad text
replacement inside binary joblib, pickle, SQLite, or Parquet files.

## 2. Current Status

### 2.1 Status by Pipeline Layer

| Layer | Status | Current artifact | Production-ready? |
|---|---|---|---|
| Shared feature store | Materialized | `data_perp/features/20260711_070000` | Required input, not a model |
| Frozen cycle AE/GMM | Materialized and used by base | Base run `_feature_selection_phase/ae_gmm_states` | Yes for its associated base cycle |
| Base alpha model | Per-side Pack-B models trained with historical OOS windows and final refit | `s59_h5_signalclose_causal_stagec_packb_wf30_20260721_v1` | Preserve as historical evidence; regenerate locked DEC-09 OOF |
| Shared-store reference/handoff source | Trained, monthly OOS generated, final refit saved | `s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1` | Historical comparator and current materialized handoff source |
| Top-30 meta handoff | Materialized | Shared-store reference run `meta_handoff_top30` | Valid existing input; regenerate from per-side base when downstream pipeline is resumed |
| Top-40 path-head handoff | Materialized | Shared-store reference run `meta_handoff_top40` | Existing CatBoost/aux population; regenerate per side for the final contract |
| Residual meta alpha | Trained with expanding OOS and final refit | `...20260722_residual_only_hpo150_wf30_v1` | Research-ready, not marked canonical |
| 12h path-archetype labels | Materialized | `20260722_path_archetype_labels_v8...` | Label-only |
| CatBoost feature selection/HPO | Complete | `catboost_path_shape_base_top40_fs75_hpo...v3` | Contract only, no final classifier |
| CatBoost geometry search | Complete enough to select geometry | `catboost_path_geometry_4m4m_train70k...v2` | Evidence only |
| Seven-class CatBoost final refit | Stopped before persistence | Empty `...geometry_e33_raw7_uniform...v1` | No |
| Five auxiliary target labels | Materialized | `20260723_s59_h5_path_aux_targets_v11...` | Label-only |
| Five auxiliary LGBM heads | Partial selection checkpoints for 3/5 | `path_auxiliary_lgbm_full...v18` | No |
| 12h execution-EV labels | Materialized | `execution_ev_12h_labels_p90spread_fee30bps...v3` | Label-only |
| Alpha execution-EV OOF stream | Materialized | `execution_ev_alpha_oof_20260722_v2...` | Historical benchmark; canonical Pack-B lineage is not proven |
| Auxiliary/CatBoost execution OOF streams | Not materialized | None | No |
| Strict joined execution-EV handoff | Not materialized | None | No |
| Direct/residual execution-EV ablation | Not run | None | No |
| Entry-timing model | Code only; intentionally later | None | No |
| Policy/inference integration | Not done for new heads | Existing alpha policy remains separate | No promotion |

### 2.2 Current Base Model

Root:

```text
data_perp/reports/
s59_h5_signalclose_causal_stagec_packb_wf30_20260721_v1
```

Key contract:

- 4,500,666 source rows.
- 242 symbols.
- Coverage from 2025-01-01 00:00 UTC through 2026-07-10 21:00 UTC.
- Seven recorded OOS folds plus final refit.
- The directional target resolves over 96 causal 15-minute path bars: 24 hours
  after the one-hour-delayed decision timestamp.
- One frozen cycle AE/GMM state.
- Manifest records `model_side_scope=per_side`.
- Independent selected-feature contracts: 55 long and 37 short features.
- Independent long and short fitted models are stored in the model bundle.
- Feature selection and HPO use the Pack-B side-local training path.

The exact per-side feature lists and parameters are authoritative in this run's
manifest and model directories; do not replace them with the shared-store
run's shared HPO parameter block.

The saved seven fold models do not match the locked DEC-09 four-fold calendar,
and their manifests do not prove the required train cutoffs and row-level label
resolution. They may be rescored only as a historical comparator.

The recovered Pack-B fitted-state contracts are also historical-only:

- feature selection and HPO both used calibration fold
  `2026-06-30_2026-07-30`;
- the recovered AE/GMM state used cycle reference fold
  `2026-06-26_2026-07-26`, ending `2026-06-25 23:00 UTC`;
- DEC-09 requires every feature-selection/HPO label to resolve strictly before
  `2026-03-01 00:00 UTC`.

Consequently, the 55-long/37-short feature lists, promoted parameters, and
recovered AE/GMM state cannot be frozen into canonical April–July OOF. They
remain useful comparator evidence only. Canonical R3 requires a fresh,
side-local pre-March reference process that independently fits AE/GMM state,
feature selection, and HPO for long and short, freezes those artifacts, and
then produces the April, May, June, and July 1–11 half-open OOF folds. For this
Pack-B target:

```text
decision_timestamp = signal_timestamp + 1 hour
base_label_end = decision_timestamp + 24 hours
eligible_train_signal < validation_start - 25 hours
```

Use the label manifest or causal path audit as the authoritative shard
inventory. Do not glob every Parquet file in the labels directory: the current
directory contains an overlapping stale `train_global_short_7.parquet` file
that is absent from the 38-file causal audit. Canonical preflight must reject
unlisted or missing shards and duplicate candidate IDs before fitting.

The downstream path, auxiliary, execution-EV, and timing targets remain
12-hour contracts. Every manifest must bind the horizon of its own target
rather than inheriting a generic pipeline horizon.

The later shared-store run remains useful as a historical comparator and as the
source of already materialized top-30/top-40 research handoffs, but it is not
the directional-base architecture to promote. Regenerate final downstream
handoffs independently from the per-side Pack-B streams. A common top-40
percentage may be used for handoff accounting only when computed within each
side's prediction stream.

Do not compare its corrected-target HPO diagnostics to older pre-causal metrics
without rejoining identical rows, costs, labels, and top-k definitions.

### 2.3 Current Residual Meta Alpha

Root:

```text
data_perp/reports/
s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_residual_only_hpo150_wf30_v1
```

Architecture:

```text
base backbone + side-local residual EV experts + hierarchical EV map
```

Residual target:

```text
ev_after_1pct
- train_only_hierarchical_expected_ev(base_score, side, archetype)
```

The final mapped score is a common expected-net-EV unit after the 1% alpha-layer
cost contract. The execution-EV labels later use a separate p90-spread plus
30-bps fee contract; these must not be mixed or subtracted twice.

Feature selection:

- Side-local.
- Archetype-aware pre-screening.
- Beginning/middle/end sample.
- Fit ends before 2026-03-01.
- 45,000 rows for selection and 45,000 for HPO.
- 150 HPO trials requested per side.
- Five protected base anchors:
  - `score`
  - `base_score_rank_pct_train_prior`
  - `base_margin_to_cutoff`
  - `base_margin_to_cutoff_z`
  - `base_signal_zscore_within_archetype`
- Final contracts contain 22 features for long and 22 for short.
- Both contracts retain a DAE latent feature.

The model generated four monthly OOS folds for April, May, June, and July
1-10, then a final refit through 2026-07-10 21:00 UTC. The final refit contains
1,294,469 training rows and is explicitly excluded from the reported OOS
predictions.

Important status:

- The manifest says `canonical: false`.
- It is nevertheless the current alpha input for the new execution-EV work.
- Preserve it unchanged until the execution-EV ablation is complete.

### 2.4 Causal Timing Contract

The corrected causal contract is:

```text
decision_ts = signal_ts + signal_timeframe
first_path_timestamp >= decision_ts
```

For the current hourly research path:

- Signal timeframe: 1 hour.
- Decision delay: 1 hour.
- Path horizon for path heads and execution-EV: 12 hours.
- No pre-entry outcome may be included as a model input.
- Delayed/slipped executable entry is a separate execution adjustment, not a
  replacement for the mandatory one-bar causal offset.

Any future label, replay, or inference path must retain an invariant test for
the first path timestamp.

### 2.5 CatBoost Future-Path Archetypes

The intended inference taxonomy is now seven classes:

| Archetype | Economic sign | Meaning |
|---|---|---|
| `immediate_adverse_path` | Adverse | Adverse excursion develops before a usable favorable path |
| `early_mfe_full_reversal` | Adverse | Favorable excursion appears but is subsequently lost |
| `fast_realization_winner` | Favorable | Merged fast-clean and fast-early-drawdown winners |
| `late_breakout` | Favorable | Weak early realization followed by later expansion |
| `slow_grinder` | Favorable | Gradual favorable accumulation |
| `noisy_timeout_usable_mfe` | Neutral | Usable excursion exists but conversion is noisy/late |
| `dead_timeout` | Adverse | Insufficient usable favorable excursion |

`fast_realization_winner` intentionally merges the former:

- `fast_clean_winner`
- `fast_winner_early_drawdown`

The final CatBoost output contract must contain:

- One probability per class.
- Maximum probability.
- Normalized entropy.
- Top-2 probability margin.
- Probability mass on adverse archetypes.
- Probability mass on favorable archetypes.

The abandoned refinements must not return:

- No temperature/vector/isotonic probability transform inside this classifier.
- No centroid-derived soft memberships.
- No ambiguity sample weighting.
- No economic sample weighting.

Raw reliability diagnostics such as ECE, Brier score, and log loss remain
allowed because they measure probabilities without transforming them.

### 2.6 CatBoost Feature Selection, HPO, and Geometry

Classifier FS/HPO artifact:

```text
data_perp/reports/
catboost_path_shape_base_top40_fs75_hpo_classsupported_20260723_v3
```

State:

- Feature selection complete.
- HPO complete.
- 75 selected features.
- The completed selection/HPO contract is shared across sides; it is therefore
  a benchmark only under the new side-local requirement.
- No final reusable seven-class classifier bundle.
- No complete CatBoost OOF stream.

Frozen effective parameters:

```text
auto_class_weights: None
bootstrap_type: Bayesian
bagging_temperature: 0.5695602692
border_count: 64
depth: 5
grow_policy: SymmetricTree
iterations: 3000
l2_leaf_reg: 9.6614074
learning_rate: 0.01779270117
od_wait: 150
random_strength: 0.6978987671
rsm: 0.8258801553
```

The HPO study's best objective, 1.72157, was obtained before the final class
merge. The feature and model-parameter contract is retained, while the final
model must be refit on the merged seven-class taxonomy.

Geometry artifact:

```text
data_perp/reports/
catboost_path_geometry_4m4m_train70k_classsupported_20260723_v2
```

Selected geometry ID:

```text
geometry_e33b290e324f3182
```

Evaluation contract:

- Four months training.
- Four months OOS.
- 70,000 deterministic, chronologically spread, side/archetype-stratified
  training-row cap.
- Every OOS row retained.

The completed geometry evidence pooled long and short rows while stratifying
and reporting by side. It must be rerun as two independent long/short geometry
searches before final CatBoost training.

Previously reported provisional leader diagnostics:

```text
OOS rows: 334,546
OOS log loss: approximately 1.44
Weighted F1: approximately 0.30
RPS: approximately 0.193
Predicted economic separation: approximately 0.167
Stability: approximately 0.969
```

The checkpoint includes several metric variants and older eight-class support
rows. Treat the values above as geometry-selection evidence, not final
seven-class classifier metrics. Recompute the full report after the seven-class
OOF refit.

The attempted final export directory is empty:

```text
data_perp/reports/
catboost_path_geometry_e33_raw7_uniform_inference_20260723_v1
```

It may be deleted or reused as an output destination. It is not a model.

### 2.7 Auxiliary Path Heads

Canonical auxiliary target artifact:

```text
data_perp/artifacts/
20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr
```

Target data:

- Long rows: 1,688,396.
- Short rows: 1,688,396.
- Coverage: 2025-02-01 through 2026-07-21 06:00 UTC.
- One-hour decision delay.
- Twelve-hour path horizon.
- Meaningful MFE threshold: at least 1.5 ATR under the current target contract.
- Targets are resolved path labels and must never enter inference features.

Five required heads:

1. `peak_mfe_12h_atr`
2. `time_to_first_meaningful_mfe`
3. `mae_before_meaningful_mfe_atr`
4. `bars_before_price_stops_decreasing`
5. `future_slope_atr_per_hour`

Head-specific intent and downstream use:

| Head | Primary question | Target treatment | Intended execution-EV contribution |
|---|---|---|---|
| `peak_mfe_12h_atr` | How much favorable excursion remains over the causal 12h path? | ATR-normalized peak MFE; log/robust treatment and capped tail diagnostics | Opportunity magnitude, reachable profit geometry, ranking |
| `time_to_first_meaningful_mfe` | How long until the path first reaches meaningful MFE? | Unreached rows capped at 12h; meaningful MFE is the current 1.5-ATR-based event | Realization speed, timeout risk, whether opportunity is too slow |
| `mae_before_meaningful_mfe_atr` | How much adverse excursion occurs before the useful favorable move? | ATR-normalized pre-event MAE; reached-event and unreached-path supportive labels | Stop/adverse-path risk, entry quality, geometry tolerance |
| `bars_before_price_stops_decreasing` | How long until the adverse move forms and confirms its trough? | Bars to confirmed adverse trough plus recovery/supportive events | Early adverse timing, whether immediate entry is premature |
| `future_slope_atr_per_hour` | How efficiently does favorable excursion accumulate? | Signed ATR/hour slope through the defined favorable-path fraction, with 2/4/8/12h support | Path efficiency, continuation strength, timing complement to peak MFE |

All five heads:

- Use pre-entry base and meta feature candidates from the configured feature
  universe.
- Receive observable pre-CatBoost base-archetype encodings.
- Are trained only on the base model's top-40% candidate population.
- Must emit OOF predictions before they can be used by execution EV.
- Must persist side-specific features, parameters, fold models, OOF predictions,
  final refits, and target/support-label provenance.
- Do not consume CatBoost's realized class label as an input.

The user deliberately chose `time_to_first_meaningful_mfe` rather than a raw
time-to-peak head because it is more stable. Peak maturation remains available
as supportive labels such as time to 80%/90% of peak.

Current partial run:

```text
data_perp/reports/
path_auxiliary_lgbm_full_20260723_v18_auxcv6m_min300_strict
```

Only feature-selection checkpoints exist for:

- `time_to_first_meaningful_mfe`
- `peak_mfe_12h_atr`
- `mae_before_meaningful_mfe_atr`

No HPO, OOS predictions, or final model exists for those heads. The other two
heads have not reached a saved selection checkpoint.

Current checkpoint fingerprint:

```text
8c0bfcc4a939a18690c394652571a00944a5d9a183762362e568a16828c7629e
```

Do not resume if the label source, feature store, top-40 candidate identity, or
selection/HPO reference changes. Start a new output directory instead.

### 2.8 Execution-EV Inputs

Execution-EV labels:

```text
data_perp/reports/
execution_ev_12h_labels_p90spread_fee30bps_20260723_v3
```

Key facts:

- 1,034,990 output rows.
- 12-hour causal path.
- 95.59% total source coverage.
- Fee: 30 bps round trip.
- Spread: asset p90 spread, applied once according to the label manifest.
- This cost contract is distinct from the residual-alpha model's 1% mapping.

Alpha OOF stream:

```text
data_perp/reports/
execution_ev_alpha_oof_20260722_v2_basearchetypes
```

Key facts:

- 239,966 rows.
- Exact one-to-one UTC join on timestamp, symbol, and side.
- Four OOS folds:
  - April: 78,584 rows.
  - May: 76,874 rows.
  - June: 63,440 rows.
  - July 1-10: 21,068 rows.
- Includes base/residual alpha predictions, uncertainty/support context, and
  pre-entry base-archetype one-hot features.
- Contains no outcome-derived path archetype labels.

Missing execution-EV inputs:

- Auxiliary-head OOF stream.
- CatBoost seven-class OOF stream.
- Strict joined handoff across labels, alpha, auxiliary heads, and CatBoost.

### 2.9 Meta Execution-EV Head

This is the next meta layer to train. It is distinct from the existing residual
meta alpha head.

Pipeline position:

```text
base alpha
-> residual meta alpha + CatBoost path probabilities + five auxiliary predictions
-> execution-EV head
-> admission/policy
```

The residual-alpha, CatBoost, and five auxiliary branches are parallel
consumers of the matching side-local base stream. CatBoost and the auxiliary
heads are not downstream of residual alpha. Their OOF outputs are joined with
base and residual-alpha OOF outputs only at the execution-EV handoff.

The implementation supports two targets:

Direct:

```text
realized causal 12h execution EV
```

Residual:

```text
realized causal 12h execution EV
- train-only expected execution EV from the alpha stack
```

The direct/residual ablation has not been run. Therefore:

- There is no selected meta execution-EV architecture.
- There is no trained final execution-EV bundle.
- There are no valid execution-EV uplift metrics.
- No policy artifact currently consumes this new head.

The head must combine:

- Base score/rank/margin anchors.
- Current side-local residual-alpha score and hierarchical EV map.
- Observable base archetypes.
- Five auxiliary OOF predictions and their uncertainty/support fields.
- Seven CatBoost probabilities, max probability, entropy, top-2 margin,
  favorable mass, and adverse mass.
- Observable AE/GMM, OOD, support, leaf, market, and drift context surviving
  the strict handoff.

Its primary objective is executable economic ranking, not generic probability
calibration. The final output must nevertheless be mapped to a common expected
EV unit so long and short candidates can compete in one portfolio auction.

### 2.10 Meta Entry-Timing Head

This is a separate model after execution EV. It is not one of the five
auxiliary heads.

Question:

```text
Given the estimated execution EV, should the system enter now or wait for a
better price, balancing adverse-movement risk and achievable net-EV improvement
against the cost and probability of losing the opportunity?
```

Planned architecture:

- Primary: shallow side-local LightGBM.
- Calibration: isotonic mapping.
- Baselines: ridge/logistic and a fixed decision grid.
- Inputs include the execution-EV estimate, entry friction/spread, auxiliary
  timing/adverse predictions, archetype context, and observable market state.
- Objective is cost- and spread-aware.
- Each delayed action estimates fill probability, conditional net-EV change
  versus enter-now, and adverse-first probability. Its expected utility must
  subtract both missed-positive-EV loss when it does not fill and adverse-first
  risk when it does fill.
- The action grid includes passive adverse-limit actions expressed as a
  side-relative ATR offset. The ML head estimates action values; it does not
  place or round an order.
- A separate deterministic layer above the ML head converts a selected
  adverse-limit action into a suggested entry price:
  `decision_price - ATR_offset * ATR` for long and
  `decision_price + ATR_offset * ATR` for short. It must then apply
  conservative tick rounding plus cost, liquidity, staleness, expiry,
  marketability, and fill-feasibility gates.

Current state:

- Core module and runner code exist.
- No canonical timing labels/handoff/model/OOF result is promoted.
- It must remain deferred until the execution-EV head has a stable OOF stream.

Implementation verification:

- Counterfactual labels compare enter-now, delayed market entry, and bounded
  passive adverse-limit entry on the same causal one-minute future path and
  frozen side-local execution geometry.
- Better-entry benefit is represented by conditional post-fill executable net
  EV and `filled_delta_ev_vs_now`, so price improvement is credited only when
  it improves the full post-fill policy outcome.
- Losing the opportunity is represented by the no-fill probability and a
  missed-opportunity loss based on positive enter-now EV; skipping a negative
  enter-now opportunity is not penalized.
- Adverse movement is represented by the probability of adverse-first movement
  and an explicit adverse-first penalty, with OOF diagnostics for post-entry
  MAE, MAE reduction, retained MFE, missed profitable trades, and regret.
- Fee and spread accounting is action-aware and fail-closed: fee, entry spread,
  and exit spread are applied once, and ambiguous passive-limit touch bars do
  not receive optimistic intrabar ordering.
- Model, calibration, and decision-policy evidence must be inner OOF/outer OOF
  and side-local. These properties are implemented, but they are not promotion
  evidence until the stable execution-EV OOF stream exists and the canonical
  four-fold timing run passes the gates below.

### 2.10A Side-Local Head Audit

| Head | Current implementation | Final requirement |
|---|---|---|
| Directional/alpha base | Per-side Pack-B models and feature contracts | Retain independent long/short FS, HPO, models, OOF and final refits |
| Residual alpha experts | Side-local long/short experts | Retain side-local; refit against the matching side-local base stream |
| CatBoost path archetypes | Shared classifier and pooled geometry evidence | Two complete side-local pipelines, including geometry and class balancing |
| Five auxiliary heads | Side-local models/HPO are supported, but pre-screening is currently global | Move pre-screening to each side; enforce separate MDA, HPO, OOF and final models |
| Execution-EV head | Side-local models/HPO, but pooled calibration and no independent FS | Add per-side FS and separate long/short calibration maps |
| Entry-timing head | Side-local models/maps, but model and decision HPO currently precede side split | Move FS, model HPO, action-grid HPO and calibration fully inside each side |
| Exit geometry/admission/sizing | Side and side x archetype aware | Keep side-local until portfolio competition |
| Portfolio manager | Global auction | The only layer where long and short opportunity streams join |

### 2.11 Existing Policy and `simple_policy_optimiser`

The new CatBoost/auxiliary/execution-EV work has not yet produced a replacement
policy. The current executable policy artifact remains:

```text
data_perp/artifacts/
s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2
```

The most current executable policy contract is:

```text
simple_policy_optimiser/deployment/best_policy_params.json
```

Policy name:

```text
s52_v9_tail95_ev70_jointtrailing1m_rawbayes_v1
```

Exit/sizing pathway:

```text
joint_trailing_total_mfe_raw_bayesian_v1
```

#### Score and EV Chain

The currently materialized policy chain is:

```text
base/meta alpha
-> V9 forced local tail-95 predecessor overlay
-> hierarchical side x archetype expected-EV map
-> causal recent side x archetype EV correction
-> fixed corrected-EV admission
-> side/archetype exit geometry
-> Bayesian sizing
-> global portfolio auction
```

The market-state MLP and regime-EV calibrator have been removed from the
executable artifact and future policy-generation path. The V9 overlay now feeds
a standalone monotonic side x archetype EV map directly. The old eight-day
hit-rate-rank smoother is also disabled.

#### Side x Archetype EV Mapping

The historical expected-EV mapping is side and archetype specific. It is not a
single global score-to-EV curve.

Admission uses:

```text
corrected_expected_ev
= side_archetype_mapped_expected_ev
+ causal_21d_robust_trimmed_side_archetype_recent_ev_residual
```

The recent correction:

- Uses the previous 21 causal days.
- Is computed per side x archetype with fallback when support is insufficient.
- Robustly normalizes daily residuals using median/IQR.
- Symmetrically trims the top and bottom 10% of daily residual observations.
- Uses train/reference support rather than future outcomes.

Admission condition:

```text
corrected_expected_ev >= 0.007
```

That is a fixed target of +0.70% net EV under the policy's stored cost
definition.

Policy ID:

```text
side_archetype_hier_ev_fixed70_trim10_21d_v1
```

Policy file:

```text
policy_params/threshold_basis_policy_sidearch_ev70_trim10_21d.json
```

Important distinction: this is recent realized EV residual correction, not the
older 8-day hit-rate-priority rank modulation. The artifact records:

```text
threshold_basis_hr_rank50: false
archetype_hit_surprise_enabled: false
rolling_8d_modulator_enabled: false
```

#### Archetypes in the Existing Policy

The current policy uses the existing observable base/policy archetype keys for:

- Hierarchical EV mapping.
- Recent 21-day EV residual correction.
- Side/archetype exit geometry.
- Bayesian sizing priors/fallback.
- Loss/cooldown diagnostics.

The new seven CatBoost future-path archetypes are not yet part of this policy.
They may enter only after the execution-EV OOF ablation demonstrates value.

#### Exit Geometry

Current executable exit contract:

- Exact one-minute replay.
- 1,440-minute maximum horizon.
- Joint trailing-only geometry.
- Trailing activation/evolution based on total MFE.
- Raw ATR scaling: `atr_power=1`, `atr_multiplier=1`.
- Capital-preservation exits disabled.
- Frozen long-side adverse-exit guard.
- Short-side adverse-exit guard disabled.
- Side x policy-archetype geometry with side-parent fallback.

Source handoff:

```text
data_perp/reports/
simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/
PRODUCTION_HANDOFF.md
```

#### Sizing

Current sizing ID:

```text
raw_bayesian_v1
```

Sizing combines:

- Rank-based base size.
- Expected-EV quality.
- Uncertainty penalty.
- OOD penalty.
- Train-fitted side/archetype x quality-bin beta-binomial lower confidence.
- Side-prior fallback.

The stored pathway uses bounded multipliers and a training-fitted normalizer.
There is no rolling OOS renormalization.

#### Portfolio Auction

Current portfolio policy:

```text
global_auction_v1
```

Key stored constraints:

- At most two new entries per bar.
- One concurrent position per symbol.
- No enforced long/short position-count cap.
- Gross pre-leverage wallet allocation capped at 70%.
- Capital/occupancy pressure raises the effective admission burden as wallet
  usage increases.
- Long and short candidates compete globally using expected EV/rank and sizing
  quality.
- The operational `64` position value is an emergency bound, not an optimized
  active count constraint.

Inference friction rebasing stored in the policy:

- 20 bps fixed round-trip component.
- 1.5 times observed spread.
- This inference adjustment must not be subtracted again if the EV input has
  already been rebased to the same contract.

#### Retired MLP/Regime Contract

The promoted manifest and executable policy now agree:

```text
policy_id: meta_residual_v9_tail95_hier_ev_v1
V9 tail-95: enabled
standalone hierarchical side x archetype EV mapping: enabled
market-state MLP: removed
regime-EV calibration: removed
```

The standalone mapping is:

```text
policy_params/side_archetype_ev_mapping.json
```

It was refit on `237,334` V9-only OOS rows. The matching 21-day causal
admission reference is:

```text
policy_params/threshold_basis_reference_sidearch_v9_ev70_trim10_21d.parquet
```

#### Policy Status Relative to New Research

The existing policy is historical context and a downstream benchmark. The new
pipeline must not overwrite it until:

1. The execution-EV head improves OOS results.
2. The entry-timing head is either validated or explicitly omitted.
3. `simple_policy_optimiser` is rerun using the new common EV output.
4. Side/archetype geometry, sizing, and portfolio constraints are revalidated.
5. Replay/inference parity passes.

### 2.12 Validation Completed

The focused execution-EV/handoff suite passed with the signed Python executable:

```text
54 passed
```

Covered:

- CatBoost OOF reporting.
- CatBoost-to-execution adapter.
- Strict joined execution-EV handoff.
- Execution-EV runner.
- Execution-EV model logic.

The following broader work remains:

- Full CatBoost classifier test suite after the seven-class refit changes.
- Full auxiliary-head suite.
- Full path-label and geometry suite.
- End-to-end joined-handoff smoke.
- OOS economic validation of the direct/residual execution-EV models.

## 3. What Is Left To Do

### Blocking Work

1. Secure the worktree and all required artifacts on the new laptop.
2. Validate the migrated environment and absolute paths.
3. Validate and preserve the existing per-side directional base and
   residual-alpha foundations.
4. Complete the final side-local seven-class CatBoost classifiers, per-side
   geometry, per-side HPO, class-balance mini-HPO, and OOF predictions.
5. Complete all five auxiliary-head selections, HPO, monthly OOS predictions,
   and final refits.
6. Materialize the auxiliary and CatBoost execution OOF streams per side.
7. Build strict long and short execution-EV handoffs.
8. Train and compare direct versus residual execution-EV models per side.
9. Run input-family ablations.
10. Select winners using side-local OOS execution-EV, ranking, and stability
    evidence.
11. Only then join sides in portfolio management and consider policy/replay
    integration.

### Non-Blocking but Important Work

- Persist exact command/provenance in every future checkpoint.
- Add a repository-level stage manifest joining all hashes and row identities.
- Commit or otherwise archive the currently untracked implementation.
- Recompute complete seven-class CatBoost probability/economic diagnostics.
- Add a clear common-unit mapping for execution-EV predictions if required by
  the downstream portfolio auction.
- Keep entry-timing work separate until execution EV is available.

### Explicitly Deferred

- Policy geometry optimization using the new models.
- Portfolio optimization using the new models.
- Live inference integration.
- Entry wait/delay optimization.
- Any predictive regime model not required by the current execution-EV goal.
- Promoting the side-local base or residual-alpha replacements over the current
  benchmark before identical-row OOS validation is complete.

## 4. Relevant Artifacts and Data

### 4.1 Must Copy

| Priority | Path | Approximate size | Why |
|---|---|---:|---|
| P0 | Entire dirty repository source tree | Source-dependent | New implementation is not fully tracked |
| P0 | `data_perp/features/20260711_070000` | 24 GB | Canonical shared feature store |
| P0 | Per-side Pack-B base `...stagec_packb_wf30_20260721_v1` | Size varies | Current long/short base models, features, OOS and final refit |
| P0 | Shared-store reference `...20260722_v1` | 8.3 GB | Historical comparator, frozen AE/GMM and existing top-30/top-40 handoffs |
| P0 | Residual run `...residual_only_hpo150_wf30_v1` | 78 MB | Current residual alpha, OOS, final refit |
| P0 | `20260722_path_archetype_labels_v8...` | 578 MB | CatBoost/path labels |
| P0 | `20260723_s59_h5_path_aux_targets_v11...` | 489 MB | Five auxiliary targets |
| P0 | CatBoost FS/HPO v3 | 6.3 MB | Frozen feature/model contract and study |
| P0 | CatBoost geometry v2 | 78 MB | Winning geometry and evidence |
| P0 | Auxiliary v18 checkpoint | 424 KB | Resume state for three selections |
| P0 | Execution-EV labels v3 | 22 MB | Canonical execution target |
| P0 | Alpha execution OOF v2 | 4.8 MB | Alpha input to final handoff |
| P0 | Current policy artifact `...20260717_v2` | 3.2 GB | EV map, 21-day admission, V9 overlay, policy state, historical bundle |
| P0 | One-minute policy handoff `...20260718_v1` | Small | Canonical exit and Bayesian sizing evidence |
| P1 | Relevant logs under `logs/` | Small | Operational provenance |
| P1 | `.git` directory | Varies | Preserve local history and dirty-state comparison |

### 4.2 Base Artifact Paths

```text
data_perp/reports/s59_h5_signalclose_causal_stagec_packb_wf30_20260721_v1/manifest.json
data_perp/reports/s59_h5_signalclose_causal_stagec_packb_wf30_20260721_v1/models/final_all_rows/base_model.joblib
data_perp/reports/s59_h5_signalclose_causal_stagec_packb_wf30_20260721_v1/models/final_all_rows/columns.json

# Historical shared-store comparator and materialized handoff source
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/manifest.json
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/best_oos_scored_ledger.parquet
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/models/final_all_rows/base_model.joblib
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/models/final_all_rows/columns.json
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/_feature_selection_phase/ae_gmm_states/cycle__global_state.pkl
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/_feature_selection_phase/ae_gmm_states/cycle__global_manifest.json
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/_frozen_ae_gmm_outputs/selected_outputs.parquet
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/_feature_selection_bme_sample/selection_result.json
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/topk_lgbm_hpo_best.json
```

Candidate handoffs:

```text
.../meta_handoff_top30/train_meta_regime_handoff.parquet
.../meta_handoff_top30/contract.json
.../meta_handoff_top40/train_meta_regime_handoff.parquet
.../meta_handoff_top40/contract.json
```

### 4.3 Residual Meta Paths

```text
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_residual_only_hpo150_wf30_v1/manifest.json
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_residual_only_hpo150_wf30_v1/staged_selection_hpo_manifest.json
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_residual_only_hpo150_wf30_v1/oos_predictions.parquet
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_residual_only_hpo150_wf30_v1/final_side_residual_expert.joblib
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_residual_only_hpo150_wf30_v1/feature_selection_importance.csv
data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_residual_only_hpo150_wf30_v1/metrics.csv
```

### 4.4 Label and Model-Development Paths

```text
data_perp/artifacts/20260722_path_archetype_labels_v8_base_top40_costaware_dense12h/
data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/
data_perp/reports/catboost_path_shape_base_top40_fs75_hpo_classsupported_20260723_v3/
data_perp/reports/catboost_path_geometry_4m4m_train70k_classsupported_20260723_v2/
data_perp/reports/path_auxiliary_lgbm_full_20260723_v18_auxcv6m_min300_strict/
data_perp/reports/execution_ev_12h_labels_p90spread_fee30bps_20260723_v3/
data_perp/reports/execution_ev_alpha_oof_20260722_v2_basearchetypes/
```

### 4.5 Current Policy Paths

```text
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/manifest.json
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/policy_params/promoted_policy_manifest.json
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/policy_params/threshold_basis_policy_sidearch_ev70_trim10_21d.json
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/policy_params/side_archetype_expected_ev_policy_manifest.json
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/policy_params/optimized_portfolio_policy_config.json
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/simple_policy_optimiser/deployment/best_policy_params.json
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/simple_policy_optimiser/deployment/best_policy_params_perps.json
data_perp/artifacts/s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2/simple_policy_optimiser/rank_reference/manifest.json
data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/PRODUCTION_HANDOFF.md
```

### 4.6 Historical Bundle Warning

This older packaged policy/model bundle may be useful for historical
comparison and remains the current policy source. It is not the deployment
target for the unfinished CatBoost/auxiliary/execution-EV pipeline:

```text
data_perp/artifacts/
s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2
```

Do not mistake its complete packaging for evidence that the new downstream
research has been promoted.

## 5. Relevant Code

### 5.1 Canonical Existing Pipeline

```text
extreme_price_movements/run_pipeline.py
extreme_price_movements/lgbm_pipeline.py
extreme_price_movements/lgbm_archetype_features.py
extreme_price_movements/alternative_meta_residual_bundle.py
extreme_price_movements/simple_policy_optimiser.py
extreme_price_movements/portfolio_manager.py
extreme_price_movements/regime_ev_calibration.py
extreme_price_movements/inference/run_inference.py
```

### 5.2 New Path and Auxiliary Modules

```text
extreme_price_movements/path_auxiliary_targets.py
extreme_price_movements/path_auxiliary_lgbm.py
extreme_price_movements/path_archetype_labels.py
extreme_price_movements/path_archetype_support.py
extreme_price_movements/path_archetype_geometry_search.py
extreme_price_movements/path_archetype_economic_ic.py
extreme_price_movements/catboost_archetype_classifier.py
```

### 5.3 New Execution Modules

```text
extreme_price_movements/execution_ev_labels.py
extreme_price_movements/execution_ev_meta.py
extreme_price_movements/execution_ev_model_ablation.py
extreme_price_movements/execution_entry_timing_meta.py
extreme_price_movements/execution_timing_risk_meta.py
```

### 5.4 Stage Scripts

Labels and candidates:

```text
scripts/materialize_path_archetype_candidates.py
scripts/materialize_path_archetype_labels.py
scripts/materialize_path_auxiliary_targets.py
scripts/materialize_execution_ev_12h_labels.py
```

Auxiliary models:

```text
scripts/run_path_auxiliary_lgbm_models.py
```

CatBoost:

```text
scripts/run_catboost_path_archetype_classifier.py
scripts/run_catboost_path_archetype_geometry_search.py
scripts/report_catboost_path_archetype_oof.py
```

Execution-EV handoff:

```text
scripts/materialize_execution_ev_alpha_oof.py
scripts/materialize_execution_ev_auxiliary_oof.py
scripts/materialize_execution_ev_catboost_refinement_oof.py
scripts/materialize_execution_ev_joined_handoff.py
```

The `catboost_refinement` filename is legacy. Its current contract should adapt
the canonical raw seven-class CatBoost output, not reintroduce abandoned
probability refinements.

Execution-EV fitting and reporting:

```text
scripts/run_execution_ev_meta.py
scripts/run_execution_ev_model_ablation.py
scripts/report_execution_ev_meta_oof.py
```

Later entry-timing stage:

```text
scripts/materialize_execution_entry_timing_1m_paths.py
scripts/materialize_execution_entry_timing_handoff.py
scripts/run_execution_entry_timing_meta.py
```

### 5.5 Tests

Auxiliary:

```text
tests/test_path_auxiliary_targets.py
tests/test_training_path_auxiliary_targets.py
tests/test_path_auxiliary_lgbm.py
tests/test_run_path_auxiliary_lgbm_models.py
```

CatBoost and geometry:

```text
tests/test_path_archetype_labels.py
tests/test_path_archetype_support.py
tests/test_path_archetype_geometry_search.py
tests/test_path_archetype_economic_ic.py
tests/test_catboost_archetype_classifier.py
tests/test_materialize_path_archetype_labels.py
tests/test_report_catboost_path_archetype_oof.py
```

Execution EV:

```text
tests/test_execution_ev_labels.py
tests/test_execution_ev_meta.py
tests/test_execution_ev_model_ablation.py
tests/test_materialize_execution_ev_alpha_oof.py
tests/test_materialize_execution_ev_auxiliary_oof.py
tests/test_materialize_execution_ev_catboost_refinement_oof.py
tests/test_materialize_execution_ev_joined_handoff.py
tests/test_run_execution_ev_meta.py
```

Entry timing:

```text
tests/test_execution_entry_timing_meta.py
tests/test_materialize_execution_entry_timing_1m_paths.py
tests/test_materialize_execution_entry_timing_handoff.py
tests/test_run_execution_entry_timing_meta.py
```

## 6. Environment and Resume Preconditions

### 6.1 Python

On the source laptop, the Python executable that worked around macOS dynamic
library signing restrictions was:

```text
/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python
```

Do not assume the same workaround is needed or available on the new laptop.
Validate imports for:

- NumPy
- pandas
- PyArrow
- LightGBM
- CatBoost
- scikit-learn
- Optuna
- joblib
- Numba

### 6.2 Storage

Before resuming:

1. Confirm the 24-GB feature store is complete.
2. Confirm Parquet footers are readable.
3. Confirm the AE/GMM pickle and joblib bundles load.
4. Confirm artifact hashes in manifests.
5. Confirm enough free space for OOF predictions and temporary feature
   matrices.
6. Keep training core counts constrained by available RAM.

### 6.3 Time and Join Contract

All storage and joins must use UTC. Legacy naive timestamps are interpreted as
UTC. CEST is display-only.

Every downstream OOF join must be exact and one-to-one on:

```text
__ts__, __symbol__, side_name
```

Reject:

- Duplicate keys.
- Missing expected rows without an attrition report.
- Outcome columns entering inference inputs.
- OOF rows scored by a model trained on their resolved path.

## 7. Detailed Roadmap for the Remaining Pipeline

### End-to-End Side-Local Architecture

All remaining predictive and execution layers must be side-local:

```text
LONG
long base alpha
-> long residual alpha + long CatBoost archetypes + five long auxiliary heads
-> long execution-EV head
-> optional long entry-timing head
-> long policy stream
                         \
                          -> global portfolio auction
                         /
SHORT
short base alpha
-> short residual alpha + short CatBoost archetypes + five short auxiliary heads
-> short execution-EV head
-> optional short entry-timing head
-> short policy stream
```

The `+` denotes parallel branches fed by the same side-local base stream, not a
serial dependency. Each branch performs its own side-local FS, HPO, fitting,
OOF scoring, and final refit as applicable.

Long and short rows must not share a fitted feature selector, HPO study, model,
probability calibrator, EV curve, geometry search, admission estimate, or sizing
model. A side indicator inside one shared model does not satisfy this contract.
The two streams may use the same code and candidate feature registry, but each
side must independently decide which features and parameters survive.

Same-side upstream outputs can be joined inside the corresponding side's
execution-EV and entry-timing handoffs. Long and short candidate streams are
joined only after they have been mapped into comparable expected-EV units, at
the global portfolio-management/auction layer.

### Phase 0: Secure and Verify the Migration

Goal: establish that the new laptop has the same source and artifact state.

Steps:

1. Copy the entire working tree, including hidden files and `.git`.
2. Copy all P0 paths from Section 4.1.
3. Generate checksums on both machines for:
   - base manifest and final model;
   - frozen AE/GMM state;
   - top-30 and top-40 handoffs;
   - residual OOF and final expert;
   - path labels;
   - auxiliary labels;
   - CatBoost HPO contract and geometry checkpoint;
   - execution-EV labels and alpha OOF.
4. Compare checksums.
5. Inspect absolute paths in JSON checkpoints.
6. Validate the Python environment.
7. Verify that no training process is active.

Gate:

- All required hashes match.
- Every required artifact opens.
- No checkpoint points to a missing source.
- The code diff/untracked files are preserved.

### Phase 1: Make the Code State Durable

Goal: prevent loss of the new implementation.

Steps:

1. Review the dirty worktree in logical groups.
2. Separate generated/runtime state from source changes.
3. Stage or archive the new path, CatBoost, auxiliary, and execution-EV source
   files and tests.
4. Record a source revision or immutable source archive.
5. Add a pipeline-stage manifest containing source revision, artifact hashes,
   feature-store signature, and command provenance.

Do not perform unrelated cleanup or revert user changes.

Gate:

- The implementation can be recovered independently of the source laptop.
- Every active artifact points to a recorded source revision.

### Phase 2: Revalidate Deterministic Contracts

Goal: prove that migration did not change semantics.

Run focused tests first, then broader suites.

Minimum focused suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <PYTHON> -m pytest \
  -p no:cacheprovider -q \
  tests/test_path_auxiliary_targets.py \
  tests/test_path_auxiliary_lgbm.py \
  tests/test_path_archetype_labels.py \
  tests/test_path_archetype_support.py \
  tests/test_catboost_archetype_classifier.py \
  tests/test_execution_ev_labels.py \
  tests/test_execution_ev_meta.py \
  tests/test_execution_ev_model_ablation.py \
  tests/test_materialize_execution_ev_joined_handoff.py \
  tests/test_run_execution_ev_meta.py
```

Also verify:

- `first_path_timestamp >= signal_timestamp + signal_timeframe`.
- The final taxonomy has exactly seven classes.
- Class order is preserved from model to Parquet to execution-EV adapter.
- Favorable/adverse probability masses use the intended classes.
- No abandoned calibration/centroid/weighting fields enter the classifier.
- Costs are applied exactly once.

Gate:

- All deterministic contract tests pass.
- Class schema and hashes agree across scripts.

### Phase 2A: Validate the Existing Side-Local Alpha Foundations

Goal: preserve and validate the existing independent long and short
directional opportunity streams before completing downstream heads.

Steps:

1. Use the current Pack-B per-side directional base as the active alpha
   foundation; keep the shared-store base only as a historical comparator.
2. Verify that its saved long and short contracts retain independent feature
   selection, HPO parameters, fitted models, OOF streams, and final refits.
   This verification must inspect the serialized model bundle and manifests,
   not infer side locality from filenames:
   - `model_side_scope` must equal `per_side`;
   - the bundle must contain distinct `long` and `short` fitted models;
   - long and short selected-feature contracts and parameter records must be
     persisted separately;
   - each OOF row must be scored by the model for its own `side_name`;
   - model, feature-contract, parameter, and source hashes must be recorded for
     each side.
3. Reuse or regenerate separate leakage-safe long and short growing-window OOS
   streams only when required by the corrected downstream handoff.
4. Select the top 40% independently within each side for CatBoost, auxiliary,
   and residual-alpha training.
   Selection must be performed per UTC timestamp and side from the matching
   per-side directional model:

   ```text
   long row  -> long base score  -> rank within timestamp x long  -> long top 40%
   short row -> short base score -> rank within timestamp x short -> short top 40%
   ```

   The handoff manifest must identify the exact long/short base model and
   feature-contract hashes used to create each score. Recompute the integer
   ranks from the source OOF ledger and require exact agreement with the saved
   selected mask, including deterministic tie handling.
5. Refit the residual-alpha experts independently:
   - long residual expert receives only long base OOF predictions;
   - short residual expert receives only short base OOF predictions;
   - selection, HPO, residual target construction, and EV mapping remain
     side-local.
   Verify that each residual expert's training keys are a subset of its
   matching directional side stream and that no long row enters the short
   expert or vice versa.
6. Preserve a common expected-EV unit only as an output contract so portfolio
   management can compare sides later.

Gate:

- Both existing per-side streams remain reproducible and beat or remain close
  to the historical shared-base benchmark on identical rows.
- No side borrows fitted selectors, parameters, priors, calibrators, or OOF
  outcomes from the other.
- Base and residual bundles pass an explicit side-local provenance audit.
- Top-40 handoff support and attrition are reported independently by side.
- The top-40 mask is reproduced exactly from the per-side directional OOF
  scores within every UTC timestamp.
- A top-40 artifact sourced from a shared base model, including the currently
  materialized shared-store handoff, is not accepted as the final auxiliary
  training population.
- The residual-alpha stage improves each matching side's base stream OOS.

### Phase 3: Finish the Seven-Class CatBoost Classifier

Goal: produce leakage-safe OOF predictions and an inference-ready final model.

Inputs:

- Path labels v8.
- Base top-40 candidate population.
- Frozen base-cycle AE/GMM outputs.
- Existing CatBoost feature/HPO evidence as a starting benchmark, not as a
  shared production contract.
- Geometry `geometry_e33b290e324f3182`.

The current shared CatBoost fit is not the final target. Build two independent
pipelines:

```text
long CatBoost:  long-only feature selection -> long-only geometry sweep
                -> long-only model HPO -> long-only class-balance mini-HPO

short CatBoost: short-only feature selection -> short-only geometry sweep
                -> short-only model HPO -> short-only class-balance mini-HPO
```

Required side-local sequence:

1. Run feature eligibility, redundancy pruning, staged selection, and automatic
   MDA stopping independently for long and short.
2. Run the archetype geometry sweep independently per side. Geometry class
   thresholds and support gates may differ by side.
3. Run CatBoost HPO independently per side using the selected side contract and
   geometry.
4. After geometry selection, run a small class-balancing HPO independently per
   side. This mini-HPO must compare bounded class-weighting strength rather than
   blindly enabling full inverse-frequency balancing.
5. Refit only the winning per-side configuration on the larger confirmation
   sample, then generate side-local OOF predictions and final models.

Class-balance mini-HPO requirements:

- Include no balancing as the control.
- Search mild-to-moderate frequency correction, such as class-frequency
  exponents from `0.25` through `0.75`, with bounded maximum class-weight ratios.
- Keep the minimum support contract per side: each class should represent at
  least 1% overall and at least 0.5% per month, or be merged before fitting.
- Reject an arm if predicted probabilities or hard predictions collapse onto
  one class, even when weighted loss improves.
- Do not reintroduce centroid memberships, soft rule memberships, ambiguity
  weighting, or economic sample weighting. This mini-HPO is specifically for
  class balance after the side-local geometry is fixed.

Select the best balancing arm using both predictive and economic OOF evidence.
The scoring report must include:

- ML quality: multiclass log loss, macro/weighted F1, RPS, Brier/ECE,
  per-class precision/recall, minimum class recall, confusion distance,
  predicted class shares, max probability, normalized entropy, and top-2
  margin.
- Collapse diagnostics: dominant predicted-class share, missing predicted
  classes, prediction/target class-share divergence, fold/month class support,
  and probability entropy by side.
- Economic quality: economically weighted confusion, standardized separation
  in net EV after costs, MFE, MAE, stop probability, time to realization,
  retention and trailing conversion, plus probability-weighted IC to each
  outcome.
- Stability: fold, month, symbol, base-archetype, and score-tail stability,
  including worst-month economic separation.

No arm is promotable unless it improves or preserves the combined ML/economic
objective and passes every no-collapse support gate.

Required outputs:

- Separate long and short fold models.
- OOF probabilities.
- OOF class prediction.
- Max probability.
- Normalized entropy.
- Top-2 margin.
- Adverse probability mass.
- Favorable probability mass.
- Separate final long and short all-rows models.
- Per-side feature lists, geometry, HPO parameters, class-balance parameters,
  and manifests.
- Class-order hash.
- Source and row-identity hashes.
- OOF metrics per side first, then a reporting-only combined view, including
  month and base archetype.

Required diagnostics:

- Log loss.
- Macro and weighted F1.
- Ranked probability score.
- Raw ECE and Brier score.
- Probability entropy, max probability, and top-2 margin.
- Standard and economically weighted confusion matrices.
- Predicted-versus-realized economic separation.
- IC to net EV, MFE, MAE, time, and stop probability:
  - pooled;
  - per timestamp;
  - monthly;
  - long/short;
  - symbol-neutral.

Gate:

- OOF only for model assessment.
- Every class has acceptable support after the merge on each side.
- Neither side collapses to one dominant predicted class.
- Feature selection, geometry, model HPO, and class-balance HPO are independently
  fitted for long and short.
- No class order drift.
- Final refit is marked as excluded from OOF metrics.

### Phase 4: Finish the Five Auxiliary LGBM Heads

Goal: produce OOF predictions and final models for all five targets.

Resume candidate:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. <PYTHON> -u \
  scripts/run_path_auxiliary_lgbm_models.py \
  --labels-path \
  data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels \
  --archetype-context-path \
  data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1/meta_handoff_top40/train_meta_regime_handoff.parquet \
  --selection-hpo-reference-end 2026-03-01T00:00:00Z \
  --label-resolution-column __label_end_ts__ \
  --feature-dir data_perp/features/20260711_070000 \
  --output-dir \
  data_perp/reports/path_auxiliary_lgbm_full_20260723_v18_auxcv6m_min300_strict \
  --n-trials 75 \
  --seed 42 \
  --purge-hours 13 \
  --start 2025-02-01T00:00:00Z \
  --end 2026-07-21T06:00:00Z \
  --selection-rows 45000 \
  --hpo-rows 45000
```

Before using this command, run `--help` and verify the current CLI still
matches. The checkpoint stores absolute paths and a run fingerprint.

Selection contract:

- Train only on the base top-40 population.
- Before training, verify that this population was generated from the proper
  per-side directional OOF models using top-40 ranking within each UTC
  timestamp and side. Reject a handoff whose source manifest identifies the
  shared-store base or cannot resolve separate long/short model hashes.
- Separate long and short training populations and model bundles.
- Univariate, Relief, correlation pruning, MDA, automatic stopping, and HPO are
  all fitted independently per side.
- Refactor the current global pre-screen before resuming; a global pre-screen or
  global selected-feature union does not satisfy this contract.
- No global selected-feature union and no shared fitted selector.
- Lower redundancy threshold than the alpha head.
- No hard-coded feature count.
- Automatic MDA stop.
- Preserve observable base archetype encodings.
- HPO after feature selection.
- Up to 75 trials, with early stopping/pruning.

Target-specific sample weights remain bounded between 0.5 and 2.0.

Required metrics:

- Regression MAE/RMSE/Huber loss as appropriate.
- Rank IC computed independently per side.
- Top-1/5/10 economic diagnostics.
- Monthly and side stability.
- Support and missingness.
- Calibration of any event-probability supportive output.

Gate:

- Ten OOF model streams exist on exact candidate keys: five targets x two sides.
- Ten final models, with per-side feature lists, parameters, and hashes, exist.
- No supportive realized label is present at inference.

### Phase 5: Materialize Execution-EV Inputs

Goal: create aligned OOF features for the execution model.

Steps:

1. Keep the existing alpha OOF stream unchanged.
2. Run `materialize_execution_ev_auxiliary_oof.py`.
3. Run `materialize_execution_ev_catboost_refinement_oof.py` using the raw
   seven-class contract.
4. Validate every stream separately.
5. Run `materialize_execution_ev_joined_handoff.py`.

Strict handoff requirements:

- Exact UTC keys.
- One-to-one joins.
- Same OOF fold identity.
- No final-refit prediction used in OOF evaluation.
- No path outcome in the input features.
- Explicit attrition report.
- Identical cost/entry/horizon contract.

Expected joined groups:

- Base score/rank/margin anchors.
- Residual-alpha predictions and common EV mapping.
- Base archetype identities.
- Frozen AE/GMM and support/OOD context already present in alpha handoff.
- Five auxiliary OOF predictions and uncertainty.
- Seven CatBoost class probabilities and aggregate confidence/risk fields.
- Execution-EV label and metadata, used only as targets/reporting.

Build this handoff separately for long and short. Do not concatenate sides
before execution-EV fitting.

Gate:

- Row identity and fold provenance are complete.
- Joined rows reconcile to the intersection expected by the manifests.

### Phase 6: Train Direct and Residual Execution-EV Models

Goal: determine whether execution EV should be predicted directly or as a
residual correction to alpha EV.

Direct target:

```text
realized 12h causal execution EV
```

Residual target:

```text
realized 12h causal execution EV
- train-only mapped alpha expected EV
```

Run:

```text
scripts/run_execution_ev_meta.py
scripts/run_execution_ev_model_ablation.py
scripts/report_execution_ev_meta_oof.py
```

Core evaluation:

- Net EV/trade.
- Gross EV/trade.
- Top 1%, 5%, 10%, 20%, and 30%.
- Global top-k as primary.
- Long and short separately.
- Monthly and weekly stability.
- Worst week and worst month.
- Win rate and profit factor.
- MFE/MAE conversion.
- Timeout/stop/adverse shares.
- IC and rank stability.
- Calibration in a common EV unit.
- Side/archetype comparability.
- Residual mean and signed residual autocorrelation.
- Positive and negative hit-rate surprise.

Required input-family ablations:

1. Alpha only.
2. Alpha plus auxiliary heads.
3. Alpha plus CatBoost.
4. Alpha plus auxiliary heads plus CatBoost.
5. Remove timing features.
6. Remove adverse-path features.
7. Remove AE/GMM/OOD/support context.
8. Direct versus residual target.

All comparisons must use identical rows and costs.

Training contract:

- Direct and residual execution-EV feature selection, HPO, early stopping,
  model fitting, and monotonic EV calibration are independent for long and
  short.
- Add an explicit side-local feature-selection stage; the current code's lack
  of an independent selector is not the final contract.
- The long execution-EV model consumes only long alpha, CatBoost, and auxiliary
  OOF outputs; the short model consumes only short outputs.
- Replace the current pooled `__global__` hierarchical calibration mapper with
  separately fitted long and short maps. The two maps must emit the same
  documented expected-EV unit without pooling their fit samples.
- Do not fit a shared backbone, selector, or calibration map.

Promotion gate:

- Top-10 execution EV improves over the alpha-only baseline.
- Worst week/month do not materially degrade unless the average gain exceeds
  the allowed trade-off.
- Long and short are both reported.
- Gains are not isolated to one month/archetype.
- Final model emits a comparable expected-EV unit.

### Phase 7: Final Refit and Bundle

Only after an OOF winner is selected:

1. Refit the selected execution-EV model on all resolved eligible rows.
2. Persist:
   - model;
   - feature contract;
   - target contract;
   - costs;
   - entry/horizon contract;
   - source hashes;
   - final-fit cutoff;
   - common EV mapping;
   - CatBoost class schema;
   - auxiliary model provenance.
3. Mark the final refit as excluded from OOF metrics.
4. Create one inference bundle manifest linking every component.

The bundle may have one manifest, but it must contain distinct long and short
models, selectors, parameters, calibrators, and hashes.

Gate:

- Bundle loads in a clean process.
- Frozen features produce deterministic scores.
- All expected model inputs are present and finite under the declared
  missing-value contract.

### Phase 8: Policy and Replay Integration

This phase is intentionally downstream of execution-EV validation.

Steps:

1. Add the execution-EV winner to `simple_policy_optimiser`.
2. Preserve side and archetype reporting.
3. Re-optimize geometry, sizing, and portfolio allocation only if the
   execution-EV score changes ranking/admission materially.
4. Run causal replay with identical entry, spread, fee, and path contracts.
5. Compare before/after:
   - alpha only;
   - execution-EV score before policy;
   - optimized geometry;
   - sizing;
   - portfolio constraints.
6. Keep every cost component reconciled once.

Do not use the new model live until replay/inference parity is separately
proven.

### Phase 9: Entry-Timing Head

The entry-timing model comes after execution EV because expected EV is a major
input.

Question:

```text
Given current expected execution EV, should the system enter now or wait for a
more favorable price, balancing adverse-movement risk and achievable net-EV
improvement against the cost and probability of losing the trade?
```

Planned primary model:

- Shallow side-local LightGBM.
- Isotonic mapping.
- Ridge/logistic and fixed-grid baselines.
- Cost- and spread-aware decision objective.

Feature selection, HPO, action-value calibration, and final fitting must also
be independent per side. A shared timing model with side as a feature is not an
acceptable fallback.

The current implementation runs model HPO and decision-policy HPO before the
side partition and has no independent feature-selection stage. Move feature
selection, model HPO, action-grid HPO, isotonic calibration, and final fitting
inside the long and short loops before treating this head as side-local.

Required action-value decomposition for every delayed-market or passive-limit
candidate:

```text
expected timing utility
= P(fill) * (enter-now execution EV + conditional net-EV delta after fill)
- (1 - P(fill)) * missed-opportunity penalty * max(enter-now execution EV, 0)
- P(fill) * P(adverse-first | fill) * adverse-first penalty
```

The conditional net-EV delta must include the full effect of the changed entry
price and remaining executable path under the frozen side-local geometry.
Costs must use the same unit as execution EV, with fee, entry spread, and exit
spread reconciled exactly once.

The timing head may recommend an action of the form “wait up to N minutes for
an adverse move of K ATR.” A separate deterministic target-price layer above
the ML model must translate it into:

```text
long suggested limit  = decision price - K * decision-time ATR
short suggested limit = decision price + K * decision-time ATR
```

That layer must round conservatively to the venue tick, reject crossed or
immediately marketable limits unless explicitly intended, verify expected net
benefit after incremental costs and queue/fill assumptions, enforce liquidity
and staleness limits, attach a fixed expiry, and emit the exact suggested price,
expiry, fallback action, and reason codes. The target-price formula and gates
are deterministic policy logic, not extra fitted ML outputs.

Promotion is relative to the enter-now baseline on paired OOF rows. It requires
positive risk-adjusted utility uplift, controlled missed-positive-EV and
adverse-first rates, acceptable worst-fold/week/month behavior, and calibration
of fill, adverse-first, and action-value components. If it fails, retain
enter-now; failure of this optional layer does not reject a stable execution-EV
winner.

Do not begin this phase before the execution-EV winner and its OOF stream are
stable.

## 8. Do Not Do

- Do not replace the current per-side Pack-B directional base with the older
  shared-store benchmark.
- Do not refit AE/GMM independently for downstream folds.
- Do not use final-refit predictions as OOF evidence.
- Do not reuse the pooled CatBoost geometry as the final contract. Run the
  required independent long and short geometry searches.
- Do not restore the abandoned CatBoost calibration/centroid/economic-weight
  refinements.
- Do not feed realized path labels to inference models.
- Do not subtract the alpha 1% cost and execution label fee/spread twice.
- Do not compare metrics from different rows, costs, horizons, or top-k bases.
- Do not promote the empty raw-seven CatBoost output directory.
- Do not treat partial auxiliary selection checkpoints as trained models.
- Do not integrate the new models into live inference before OOF validation and
  replay parity.

## 9. Immediate Restart Checklist

Use this order on the new laptop:

1. Verify copied hashes and free disk.
2. Verify source/untracked files are present.
3. Verify JSON absolute paths.
4. Load the feature store, AE/GMM state, base model, residual model, and label
   Parquets in a read-only smoke.
5. Run deterministic focused tests.
6. Audit the base and residual-alpha bundles for true side locality, including
   independent models, feature contracts, parameters, OOF scoring and hashes.
7. Regenerate and verify top-40 selection per timestamp from the matching
   per-side directional OOF streams.
8. Finish long and short CatBoost OOF/final models, including the
   post-geometry class-balance mini-HPO.
9. Resume and finish all five auxiliary heads independently per side.
10. Materialize CatBoost and auxiliary execution OOF streams per side.
11. Build strict long and short handoffs.
12. Run direct/residual execution-EV training and ablations per side.
13. Produce the full OOS report.
14. Select winners or explicitly reject the new layer.
15. Only then move to final refit, side-local policy optimization, global
    portfolio auction, replay, and inference.

## 10. Definition of Done

This research objective is complete only when:

- The existing per-side Pack-B base and residual-alpha streams remain
  reproducible with leakage-safe OOF predictions and final bundles.
- Serialized base and residual artifacts prove independent long/short models,
  feature contracts, parameters, scoring provenance, and hashes.
- The auxiliary/CatBoost top-40 population is exactly reproducible from the
  appropriate long or short directional OOF score within each UTC timestamp;
  no shared-base score is used.
- All five auxiliary targets have separate long/short OOF predictions and final
  bundles.
- Separate long and short seven-class CatBoost models have leakage-safe OOF
  probabilities, no class collapse, and final bundles.
- CatBoost feature selection, geometry, model HPO, and class-balance mini-HPO
  are independently selected per side using ML and economic metrics.
- The execution-EV handoff has exact row/fold provenance.
- Direct and residual execution-EV models are compared on identical OOS rows
  independently per side.
- Input-family ablations identify where uplift comes from.
- OOS top-10 execution EV improves without unacceptable week/month or side
  instability.
- A complete two-sided inference bundle is serialized.
- Long and short are joined only at the global portfolio-management layer.
- Policy/replay integration is validated separately.
- Live inference remains blocked until feature, prediction, admission, sizing,
  and exit parity pass.

Until those conditions hold, the correct status is:

```text
Per-side directional base: available
Side-local residual alpha: available; verify it consumes the matching side
stream
Path/auxiliary research: partially materialized
Execution-EV model: not yet trained
New policy: not yet optimized
New live deployment: blocked
```

## 11. Final Roadmap Update: Side-Local Until Portfolio

This is the final architecture decision for the remaining pipeline:

```text
LONG
long base alpha
-> long residual alpha + long CatBoost archetypes + five long auxiliary heads
-> long execution-EV head
-> optional long entry-timing head
-> long EV calibration/admission/geometry/sizing
                                      \
                                       -> global portfolio manager
                                      /
SHORT
short base alpha
-> short residual alpha + short CatBoost archetypes + five short auxiliary heads
-> short execution-EV head
-> optional short entry-timing head
-> short EV calibration/admission/geometry/sizing
```

Here too, `residual alpha + CatBoost + five auxiliary heads` are parallel
branches from the base model. None of those three branches is an upstream input
to either of the other two.

The global portfolio manager is the first and only fitted decision layer allowed
to combine long and short candidates. Before that point, every selector, HPO
study, model, calibration map, geometry, threshold, and sizing estimate must be
trained independently by side. Comparable expected-EV units are an interface
contract, not a reason to share a fitted model.

The current implementation audit is:

| Component | Side-local now? | Roadmap action |
|---|---:|---|
| Directional/alpha base | Historical model is per-side, but its FS/HPO/AE provenance is post-cutoff/pooled | Preserve as comparator; rebuild pre-March side-local AE, FS, HPO, models, and OOF |
| Residual alpha experts | Yes | Refit each expert on its matching side-local base OOF stream |
| CatBoost archetypes | No | Replace with independent long/short feature selection, geometry sweep, model HPO, and class-balance mini-HPO |
| Five auxiliary heads | Partial | Move global pre-screening inside each side; require ten final models and ten OOF streams |
| Execution-EV head | Partial | Add per-side FS and replace pooled calibration with independent long/short maps |
| Entry-timing head | Partial | Move FS, model HPO, action HPO and isotonic calibration inside each side |
| Admission, geometry, sizing | Side-aware | Keep independent side and side x archetype estimates |
| Portfolio manager | Global | Join the calibrated long and short streams here only |

For CatBoost specifically, the geometry sweep must happen independently per
side. Once each side's geometry is fixed, run a small per-side class-balancing
HPO with an unweighted control and bounded mild-to-moderate balancing arms. The
winner must be chosen using both:

- predictive evidence: log loss, macro and weighted F1, RPS, Brier/ECE,
  per-class precision/recall, confusion distance, confidence, normalized
  entropy, top-2 margin, and fold/month stability;
- economic evidence: economically weighted confusion, separation in net EV,
  MFE, MAE, stop probability, realization time, retention, trailing conversion,
  and probability-weighted outcome IC.

An arm is rejected if it collapses hard predictions or probability mass toward
one class, drops a supported class, violates the per-side monthly support gates,
or improves aggregate loss by sacrificing economically important minority
classes. CatBoost outputs and reports remain separate for long and short; a
combined table is reporting only.
