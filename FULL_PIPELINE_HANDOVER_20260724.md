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

The strict side-local 31/8 base OOF stream, its canonical top-40 handoff,
regenerated path labels, and the rebuilt residual-alpha OOF stream now exist.
Historical execution-EV labels and alpha-to-execution streams also exist, but
they are not yet canonical for this lineage.

The downstream work is not complete:

- The five full auxiliary-head models and their OOF predictions do not exist.
- The final seven-class CatBoost classifier and its OOF predictions do not
  exist.
- The strict joined execution-EV handoff does not exist.
- The execution-EV direct/residual ablation has not been run.
- None of the new path/execution models has been promoted to replay, policy, or
  production inference.

The residual-alpha rebuild completed on 2026-07-24. No CatBoost,
auxiliary-head, execution-EV, or timing training process should be assumed to
be running unless a later checkpoint explicitly says otherwise.

### Critical Migration Warning

A Git clone is not sufficient to resume this work.

The strict Pack-B base, top-40, path-label, and residual roadmap changes are
committed. The feature store, fitted models, OOF predictions, label artifacts,
HPO studies, and checkpoints remain outside Git and must still be transferred
with the working tree.

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
| Base alpha model | Strict per-side Pack-B outer OOF plus separate final refits | `packb_side_local_outer_oof_20260724_v1_31_8` | Canonical downstream base stream |
| Shared-store reference/handoff source | Trained, monthly OOS generated, final refit saved | `s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_v1` | Historical comparator and current materialized handoff source |
| Top-30 meta handoff | Materialized | Shared-store reference run `meta_handoff_top30` | Valid existing input; regenerate from per-side base when downstream pipeline is resumed |
| Top-40 path-head handoff | Materialized from strict 31/8 outer OOF | `packb_side_local_top40_20260724_v1_31_8` | Canonical downstream population |
| Residual meta alpha | Strict side-local May-July OOF plus separate final refits | `packb_side_local_residual_oof_20260724_v1_31_8` | Promoted independently for both sides |
| 12h path-archetype labels | Regenerated from canonical 31/8 top-40 IDs | `20260724_path_archetype_labels_v9_packb31_8_top40` | Canonical label-only artifact |
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

The recovered AE/GMM state and post-cutoff promoted parameters cannot be frozen
into canonical April–July OOF. The user approved one narrow exception on
2026-07-24 for future runs: the 55-long/37-short feature lists may be selected
once on the 2026-05-31 through 2026-06-30 largest fold and reused backward.
This exception applies only to the feature names. AE/GMM fitting, HPO, model
fitting, calibration, and reported OOF predictions remain subject to their
normal chronological cutoffs. Canonical R3 must therefore refit strict
December-February HPO for the frozen 55/37 lists and compare them with the
fresh 31/8 pair on identical outer rows and costs. Promote 31/8 only if it has
higher paired metrics. For this Pack-B target:

```text
decision_timestamp = signal_timestamp + 1 hour
base_label_end = decision_timestamp + 24 hours
eligible_train_signal < validation_start - 25 hours
```

R3 execution update on 2026-07-24:

- Fresh side-local AE/GMM state is frozen at
  `data_perp/artifacts/packb_side_local_ae_20260724_v1`.
- The recent-winner selector process was rerun independently by side on the
  legal pre-March population. It produced 31 long and 36 short features; the
  selector used correlation-first pruning, archetype univariate and Relief
  screens, iterative MDA, and a strict forward-only 180-day burn-in adapted to
  the locked 11-month history.
- Each side then completed 150 HPO arms across the fixed December, January,
  and February folds without fallback.
- Long promotes the fresh 31-feature v2 contract (`trial_141`): it beats the
  cutoff-legal 16-feature comparator on the primary mean cost-aware objective,
  although its worst fold is weaker.
- Short reuses the cutoff-legal eight-feature v1 contract (`trial_084`). A
  separate 41-feature union common-cohort gate refit the eight- and 36-feature
  winners on identical rows and seeds. The eight-feature model won mean
  objective `0.351842` versus `0.316326`, worst-fold objective `0.312212`
  versus `0.288474`, and executable net-return lift in all three folds.
- The small short list is therefore an evidence-based economic promotion, not
  a fixed-count shortcut. The 36-feature short result remains a diagnostic
  predictive comparator.
- The authoritative routed promotion contract is
  `docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json`.
  These are inner model-selection results, not outer OOF performance claims.
- Subsequent user direction supersedes that v1 routing as the final choice:
  55/37 is the default feature-list route under the explicit selection-timing
  exception. Regenerate strict pre-March HPO for those names, then run a paired
  55/37-versus-31/8 outer-OOF gate. Retain 31/8 only if it wins the paired
  cost-aware metrics. Do not reuse the historical post-cutoff HPO parameters.
- That regeneration is complete at
  `data_perp/artifacts/packb_side_local_fs_hpo_20260724_v3_hist55_37`.
  Long evaluated 150 trials over three folds and selected `trial_003`
  (mean objective `0.233185`, worst fold `0.190258`); short selected
  `trial_049` (mean objective `0.185734`, worst fold `0.168530`). All 55 and
  37 requested features were admitted with the frozen 95% per-feature
  coverage floor and LightGBM native missing-value handling; no imputation or
  post-cutoff parameter reuse occurred.
- The frozen default-pending-gate route is
  `docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v2.json`.
  It is not a final performance promotion: the v2 55/37 stream and v1 31/8
  stream must still be trained on the four outer folds and compared on the
  exact candidate-ID intersection with identical cost labels.
- The 95% floor applies to inner HPO dataset admission. Outer OOF must not
  discard or impute genuine future feature-availability drift: it admits
  label-complete rows with LightGBM native missing values, records
  per-feature coverage for every fold, and includes coverage deterioration in
  the promotion audit. The bounded smoke exposed material July AE/GMM
  coverage drift, so final promotion requires both paired economic metrics
  and an explicit coverage-risk disposition.

The locked inner calendar is fixed before this new search:

- side-local AE/GMM reference: authorized beginning/middle/end samples from
  January 1 through November 1, 2025, with every 24-hour label resolution
  strictly before November 1;
- feature-selection validation: November 1 through December 1, 2025;
- HPO validation: December 2025, January 2026, and February 2026 as three
  chronological folds;
- every inner training row must resolve strictly before its validation start;
- no silent feature-selection or model fallback is allowed.

Use the label manifest or causal path audit as the authoritative shard
inventory. Do not glob every Parquet file in the labels directory: the current
directory contains an overlapping stale `train_global_short_7.parquet` file
that is absent from the 38-file causal audit. Canonical preflight must reject
unlisted or missing shards and duplicate candidate IDs before fitting.

R3 subsequently reconciled the declared July tail append and completed a new
streaming audit of the current 38 monthly shards:

- 4,552,934 rows scanned;
- zero duplicate candidate IDs or timing, path, cost, side, or validity
  failures;
- `train_global_short_7.parquet` is explicitly classified as an excluded
  historical monolithic shard;
- strict pre-March authorization retained 1,711,400 long rows and 1,718,388
  short rows, both ending at label resolution `2026-02-28 23:00 UTC`.

Evidence is under `docs/pipeline_roadmap/20260724/r3/`. This authorizes the
candidate populations. The immutable production population is now materialized
under `data_perp/artifacts/packb_pre_march_population_20260724_v1` with
3,429,788 authorized rows and 18 fixed-calendar side ledgers. The compact
evidence record is
`docs/pipeline_roadmap/20260724/r3/pre_march_population_materialization_v1.json`.
Its guarded run peaked at 463,814,656 bytes RSS, retained more than 10 GiB
available RAM, and produced a 43 MiB artifact. AE/GMM, feature selection, HPO,
and OOF fitting remain unstarted.

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
- It is historical evidence only and is not the current execution-EV input.
- Preserve it only as a comparator until the execution-EV ablation is complete.

R3 canonical replacement, completed 2026-07-24:

- Root:
  `data_perp/artifacts/packb_side_local_residual_oof_20260724_v1_31_8`.
- Input: the exact canonical 31/8 side-local top-40 population.
- April is development-only. OOF scoring covers May, June, and July 1-10 with
  prior-resolved-label cutoffs.
- Long and short use independent feature selection, HPO, baseline isotonic EV
  maps, residual models, correction strengths, folds, and final refits.
- Both side promotion gates passed on 195,931 unique OOF candidate IDs.
- Long cost-aware objective improved from `0.197479` to `0.324685`; weighted
  rank IC improved from `0.126060` to `0.168594`; top-10 net-return lift
  improved from `43.22` to `92.53` bps.
- Short cost-aware objective improved from `0.213851` to `0.380132`; weighted
  rank IC improved from `0.139669` to `0.155049`; top-10 net-return lift
  improved from `43.80` to `107.56` bps.
- Long HPO stopped after 35 trials and short after 39 under patience 20, so the
  configured 75-trial ceiling did not add unnecessary completed trials.
- The guarded run peaked at 3,462,103,040 bytes RSS, retained at least
  10,080,534,528 bytes available RAM and 82,979,606,528 bytes free disk, and
  recorded no guard violations.
- The authoritative evidence is
  `docs/pipeline_roadmap/20260724/r3/packb_residual_rebuild_contract_v1.json`.

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

Historical auxiliary target artifact (comparison only; do not use for the
current 31/8 production route):

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

Canonical 31/8 target and pre-entry context as of 2026-07-25:

```text
data_perp/artifacts/
packb_path_auxiliary_targets_20260725_v1_31_8/targets.parquet

data_perp/artifacts/
packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm/context.parquet
```

- Exact canonical population: 300,315 rows, comprising 140,768 long and
  159,547 short rows.
- Target geometry: decision at signal +1h, contiguous 12h path, resolution at
  signal +13h.
- Meaningful MFE is `max(1.5 * signal ATR, 1.5% entry return)`. The explicit
  `__meaningful_mfe_reached_12h__` event must be used by hurdle/survival roles;
  `__mfe_ge_1_5atr__` is not an equivalent substitute.
- The context contains 63 outputs from each side's frozen, outcome-free
  pre-March AE/GMM state plus `gmm_representation_available`; no outcome
  columns were added.
- Candidate identity is unchanged:
  `1b7442bfd5a75c040869d156b6ed102423eff8876b8fca3b8a0781b7fcdaef55`.
- Context artifact SHA-256:
  `ecedbd3424ceb3b14c91179fa4518fe9966e42a66567d9f7e95b8c784069c837`.

Frozen-representation availability:

| Side/month | Jointly finite AE/GMM inputs |
|---|---:|
| Long April/May/June/July | 100.00% each |
| Short April | 99.97% |
| Short May | 95.35% |
| Short June | 75.41% |
| Short July | 71.01% |
| Short aggregate | 89.71% |

The June short slice contains 41,454 candidates; 10,192 have no jointly finite
frozen representation. A targeted source audit found that the 30 symbols with
the largest unavailable-candidate counts all have internal hourly OHLCV gaps
during May-June. Those 30 symbols alone are missing 6,917 source hours. Examples
include `OPEN` (234 missing hours, maximum gap 28h), `SPK` (228, 10h),
`CAKE` (273, 17h), and `GRIFFAIN` (335, 36h). By contrast, a high-availability
control sample was generally complete or had only one isolated 10h gap.

**2026-07-25 recoverability assessment:** all 41,454 June short candidates have
a complete decision-time OHLCV candle. The 10,192 unavailable representations
instead overlap earlier raw gaps: 10,190 have at least one gap in the preceding
24 hours and all 10,192 do within 48 hours. The direct 24-hour causal surface
contains 15,962 distinct gaps across 125 symbols. This overlap identifies the
missingness mechanism but is not evidence that the gaps are recoverable.

The 30 symbols with the most missing June short candidates account for 4,227
of the 10,192 unavailable candidates (41.47%) and 6,917 locally absent
May-June source hours. A read-only query of Kraken Futures' exact trade-candle
endpoint recovered only 94 valid candles across 17 symbols, or 1.36% of the
audited source-hour deficit. All 94 occur in one May 25 07:00-16:00 UTC gap;
the pass recovered **zero** valid June candles despite covering 5,907 June
source gaps in the 30 worst symbols. The remaining endpoint rows are
overwhelmingly linked zero-volume carry candles rejected by canonical
ingestion. This is strong evidence that a broad repeat backfill would mostly
manufacture continuity from non-trade carry rows rather than restore missing
trades.

The one-pass source audit is now reproducible. The immutable network-response
stage is:

```text
data_perp/artifacts/
kraken_futures_exact_source_repair_20260725_v1
```

Its first-pass ledger contains all 6,917 endpoint timestamps and must **not** be
applied: 6,839 are flat zero-volume rows, mostly linked carry candles. Offline
series-level validation using the same carry filter as canonical Kraken
ingestion produced the only eligible patch:

```text
data_perp/artifacts/
kraken_futures_exact_source_repair_20260725_v1_revalidated_carry_filtered_v2
```

The validated patch contains exactly 94 candles: 78 positive-volume trade
candles and 16 isolated legitimate no-trade candles. It rejects 6,823 scoped
linked carry candles and made no additional network calls. Both artifacts are
`NOT_APPLIED`; the baseline raw, feature, and context artifacts remain
unchanged.

The complete read-only reconciliation artifact is:

```text
data_perp/artifacts/
kraken_futures_june_representation_gap_audit_20260725_v1
```

Its status is `READ_ONLY_NO_BROAD_BACKFILL_RECOMMENDED`; it made no network
calls, recomputed no feature/model, and mutated no baseline artifact.

**Decision / go-no-go:** do not perform a broad or repeated historical
backfill, and do not block the pipeline on the 94-row challenger. The validated
May patch cannot make the June windows continuous because every unavailable
June row still has another gap within 48 hours, and all but two do within 24
hours. In addition, the historical feature-store run did not preserve enough
configuration/state provenance to guarantee a patch-disabled byte-parity
recompute; the copy-on-write planner therefore remains
`PLANNED_NOT_COMPUTED_NO_BASELINE_MUTATION`.

Retain the 94-row ledger and planner as integrity evidence, not as promoted
data. A future bounded repair is allowed only if a materially independent exact
source is available, the sample and carry rejection are predeclared, and a
patch-disabled clean-row parity run succeeds before any patched output is
published. Never impute, interpolate, or forward-fill these gaps.

Proceed with the immutable baseline. Keep `gmm_representation_available` as an
explicit feature, retain native missing handling, and report OOF economics
separately for available and unavailable rows. If the unavailable June short
slice is materially worse, the next remedy is a newly trained gap-aware
representation contract with missingness/staleness masks or a smaller robust
input set—not repeated downloads or synthetic filling.

The complete-label June short audit confirms that the missingness is
economically non-random. Among 31,284 joined rows, the 5,230 representation-
unavailable rows averaged `-0.006555` final return net of 1% cost versus
`-0.000107` for the 26,054 available rows (a `-64.48` bp gap). Their mean
adverse excursion before meaningful MFE was `2.84R` versus `1.90R`, and their
meaningful-MFE rate was `70.74%` versus `73.46%`. Treat these as descriptive
outcome-slice diagnostics, never as feature-selection evidence. They make the
availability flag and separate OOF reporting mandatory, while the 1.36%
source-hour recoverability estimate still argues against repeated backfill.

Do **not** median-fill, zero-fill, forward-fill, or interpolate these missing
AE/GMM inputs into the frozen state. Every candidate has an exact feature-store
row; the missing state comes from sparse individual rolling inputs, not missing
candidate rows. In June, the largest contributors are
`downside_deceleration_8h_rz` (13.24%),
`downside_deceleration_4h_rz` (11.27%), `prog_eff_24` (11.15%),
`efficiency_ratio_20` (10.53%), and `ker_16` (8.95%). The maturity contract
intentionally masks rolling features after insufficient history or internal
hourly gaps. A synthetic fill would violate the frozen training transform.

Backfill policy:

1. Audit raw OHLCV continuity and earlier history for the affected
   symbol/timestamp windows.
2. Recompute a missing feature only when the exact causal source history exists
   and the current store is demonstrably incomplete or stale.
3. Preserve NaN plus `gmm_representation_available=0` when the lookback is
   genuinely immature or crosses a source gap.
4. If missing-state OOF performance is materially worse, refit a new
   training-time representation contract with explicit missingness masks or a
   smaller robust input set; never retrofit an imputer into the current frozen
   state.
5. Report every auxiliary head by side and representation-available versus
   representation-missing rows. No production promotion is allowed if the
   missing-state slice collapses materially.

Five required heads:

1. `peak_mfe_12h_atr`
2. `time_to_first_meaningful_mfe`
3. `mae_before_meaningful_mfe_atr`
4. `bars_before_price_stops_decreasing`
5. `future_slope_atr_per_hour`

Head-specific intent and downstream use:

| Head | Primary question | Target treatment | Intended execution-EV contribution |
|---|---|---|---|
| `peak_mfe_12h_atr` | How much favorable excursion remains over the causal 12h path? | One exact meaningful-event probability plus natural-unit conditional mean and q80 models; the canonical target is capped at 10 ATR | Opportunity magnitude, reachable profit geometry, ranking |
| `time_to_first_meaningful_mfe` | How long until the path first reaches meaningful MFE? | Side-local 2/4/8/12h CDF with one jointly tuned parameter contract; unreached rows are right-censored at 12h and meaningful MFE is `max(1.5 ATR, 1.5% entry return)` | Realization speed, timeout risk, whether opportunity is too slow |
| `mae_before_meaningful_mfe_atr` | How much adverse excursion occurs before the useful favorable move? | Shared exact event probability plus separate natural-unit hit and no-hit conditional risks; the canonical target is capped at 10 ATR | Stop/adverse-path risk, entry quality, geometry tolerance |
| `bars_before_price_stops_decreasing` | How long until the adverse move forms and confirms its trough? | Persist both the legacy adverse-extreme clock and confirmed-trough clock; choose neither from target loss alone | Early adverse timing, whether immediate entry is premature |
| `future_slope_atr_per_hour` | How efficiently does favorable excursion accumulate? | Non-negative favorable-path ATR/hour slope capped at 8; diagnostic-only until an incremental execution-EV ablation passes | Path efficiency, continuation strength, timing complement to peak MFE |

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

1. Preserve the now-validated canonical 31/8 base, top-40, path-label, and
   residual-alpha artifacts unchanged.
2. Complete the final side-local seven-class CatBoost classifiers, per-side
   geometry, per-side HPO, class-balance mini-HPO, and OOF predictions.
3. Complete all five auxiliary-head selections, HPO, monthly OOS predictions,
   and final refits.
4. Materialize the auxiliary and CatBoost execution OOF streams per side.
5. Build strict long and short execution-EV handoffs.
6. Train and compare direct versus residual execution-EV models per side.
7. Run input-family ablations.
8. Select winners using side-local OOS execution-EV, ranking, and stability
   evidence.
9. Train the optional cost-aware entry-timing head strictly from OOF upstream
   predictions, including adverse-movement risk, better-price benefit,
   opportunity-loss risk, and the separate suggested-wait-price layer.
10. Only then join sides in portfolio management and integrate the winning
   execution-EV outputs into `simple_policy_optimiser` and replay.

### Non-Blocking but Important Work

- Persist exact command/provenance in every future checkpoint.
- Add a repository-level stage manifest joining all hashes and row identities.
- Commit each new stage implementation before fitting production artifacts so
  every manifest binds a clean source revision.
- Recompute complete seven-class CatBoost probability/economic diagnostics.
- Add a clear common-unit mapping for execution-EV predictions if required by
  the downstream portfolio auction.
- Keep entry-timing work separate until execution EV is available.

### Explicitly Deferred

- Policy geometry optimization using the new models.
- Portfolio optimization using the new models.
- Live inference integration.
- Any predictive regime model not required by the current execution-EV goal.

## 4. Relevant Artifacts and Data

### 4.1 Must Copy

| Priority | Path | Approximate size | Why |
|---|---|---:|---|
| P0 | Entire repository source tree including `.git` | Source-dependent | Required to preserve exact manifest-bound revisions |
| P0 | `data_perp/features/20260711_070000` | 24 GB | Canonical shared feature store |
| P0 | `packb_side_local_outer_oof_20260724_v1_31_8` | Size varies | Canonical strict long/short base OOF and final refits |
| P0 | `packb_side_local_top40_20260724_v1_31_8` | Size varies | Canonical downstream candidate population |
| P0 | `packb_side_local_residual_oof_20260724_v1_31_8` | 15 MB | Canonical strict residual OOF and final refits |
| P0 | `20260724_path_archetype_labels_v9_packb31_8_top40` | Size varies | Canonical path-label identities |
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

Architecture verification gate:

- Reject any implementation or diagram in which residual alpha is an input to
  CatBoost or to the five auxiliary heads.
- Require the residual-alpha, CatBoost, and five auxiliary-head branches to
  consume the same causally eligible side-local base handoff in parallel.
- Join their strictly OOF/OOF-equivalent predictions only at the side-local
  execution-EV handoff. Final-refit predictions are inference artifacts, not
  admissible training inputs for execution EV or entry timing.

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

1. Use the approved feature-selection-timing exception to freeze the historical
   55-long/37-short names. Verify exact causal point-in-time and inference
   availability for every name, then rerun HPO independently per side on the
   strict December, January, and February folds. Do not reuse the historical
   post-cutoff HPO parameters.
2. Generate fresh April, May, June, and July 1–11 outer OOF base streams from
   both 55/37 and the fresh 31/8 comparator contracts on identical admitted
   rows. Every fold must refit using only labels resolved before its cutoff;
   final-refit predictions are forbidden from OOF metrics. Promote 55/37 by
   default; retain 31/8 only if its paired cost-aware metrics are higher.
3. Verify that the resulting long and short contracts retain independent feature
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
4. Reuse or regenerate separate leakage-safe long and short growing-window OOS
   streams only when required by the corrected downstream handoff.
5. Select the top 40% independently within each side for CatBoost, auxiliary,
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
6. Refit the residual-alpha experts independently:
   - long residual expert receives only long base OOF predictions;
   - short residual expert receives only short base OOF predictions;
   - selection, HPO, residual target construction, and EV mapping remain
     side-local.
   Verify that each residual expert's training keys are a subset of its
   matching directional side stream and that no long row enters the short
   expert or vice versa.
7. Preserve a common expected-EV unit only as an output contract so portfolio
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

Execution status on 2026-07-24:

- The approved feature-name timing exception was applied to the historical
  55-long/37-short lists. Their post-cutoff HPO parameters were not reused:
  both sides received fresh 150-trial HPO on the fixed December-February
  calendar, followed by full April-July strict outer OOF regeneration.
- The historical route and fresh 31-long/8-short route were compared on exact
  candidate-ID intersections with identical side, timestamp, symbol, fold,
  target, sample weight, and cost-aware net-return labels. Final-refit
  predictions were excluded.
- The canonical route is **31 long / 8 short**. On 348,527 paired long rows it
  beats 55 features on objective (`0.329001` versus `0.229413`), weighted rank
  IC (`0.186051` versus `0.138630`), top-10 net-return lift (`0.006473` versus
  `0.004120`), and relative RMSE gain (`0.023329` versus `0.015353`). On
  395,724 paired short rows it beats 37 features on the same four metrics:
  `0.311772` versus `0.195566`, `0.187839` versus `0.125387`, `0.005859`
  versus `0.003187`, and `0.021747` versus `0.015739`.
- The 55/37 route wins the July objective narrowly on both sides but loses the
  aggregate objective in the other three folds and loses every aggregate
  promotion metric. It remains reproducible comparison evidence, not the
  production route.
- The authoritative final contract is
  `docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_final_promotion_v3.json`;
  the reproducible paired gate is
  `docs/pipeline_roadmap/20260724/r3/packb_55_37_vs_31_8_outer_oof_gate_v1.json`.

Immediate next action:

1. Source both side streams from
   `data_perp/artifacts/packb_side_local_outer_oof_20260724_v1_31_8`.
2. **Completed:** deterministic ranks were recomputed within UTC timestamp and
   side and the independent long and short top-40 masks were materialized at
   `data_perp/artifacts/packb_side_local_top40_20260724_v1_31_8`. The saved
   300,315-row mask reproduces exactly: 140,768 long and 159,547 short.
3. **Completed:** the handoff is bound to the canonical prediction,
   feature-contract, HPO, AE/GMM, fold-model, hashed training-ledger, calendar,
   population, and source evidence. Every observed maximum label-resolution
   timestamp precedes its validation boundary. The production evidence contract
   is `docs/pipeline_roadmap/20260724/r3/packb_side_local_top40_production_v1.json`.
4. **Next:** refit and validate the residual-alpha experts independently on
   these matching side-local identities before starting CatBoost or auxiliary
   models.

Residual rebuild decision:

- Do not invoke `run_meta_v9_ev_mapped_side_residual_ablation.py` unchanged.
  Its current handoff, joint helper paths, and historical calendar are tied to
  the former shared-base population.
- The old residual bundle overlaps only 64,182 canonical long identities
  (45.6%) and 57,355 canonical short identities (35.9%); it is not an
  admissible training source or canonical fallback.
- Because the final side-local base OOF stream begins in April, April is the
  residual development warm-up. The first honest residual OOF fold is May,
  trained only on prior resolved April rows; June and July then expand on prior
  resolved canonical rows. April must be explicitly marked
  `base_passthrough_warmup` or excluded from residual uplift claims.
- Feature selection, HPO, baseline EV calibration, residual model, correction
  strength, and final refit are all independent by side. The exact frozen
  implementation and memory contract is
  `docs/pipeline_roadmap/20260724/r3/packb_residual_rebuild_contract_v1.json`.

### Phase 3: Finish the Seven-Class CatBoost Classifier

Goal: produce leakage-safe OOF predictions and an inference-ready final model.

Inputs:

- Canonical path labels v9 at
  `data_perp/artifacts/20260724_path_archetype_labels_v9_packb31_8_top40`,
  regenerated from the final 31/8 top-40 handoff. The former v8 labels overlap
  only 47.8% of the new long population and 39.7% of the new short population,
  so they are historical comparison evidence only.
- Base top-40 candidate population.
- Frozen base-cycle AE/GMM outputs.
- Existing CatBoost feature/HPO evidence as a starting benchmark, not as a
  shared production contract.
- Geometry `geometry_e33b290e324f3182`.

Current label status:

- 222,490 exact candidate identities were materialized from the canonical
  top-40 population: 105,949 long and 116,541 short.
- All 222,490 regenerated path-label rows match their candidate ID, timestamp,
  symbol, and side exactly; none falls outside the canonical candidate input.
- 208,133 rows have a complete 24-hour path. Incomplete rows remain explicit
  support evidence and must not be silently treated as complete training
  targets.
- The binding and hashes are frozen in
  `docs/pipeline_roadmap/20260724/r3/packb_path_labels_production_v1.json`.

The current shared CatBoost fit is not the final target. Build two independent
pipelines:

```text
long CatBoost:  long-only feature selection -> long-only geometry sweep
                -> long-only structural model HPO
                -> April-only fixed-parameter four-arm balance OOF sweep
                -> frozen May/June/July outer OOF

short CatBoost: short-only feature selection -> short-only geometry sweep
                -> short-only structural model HPO
                -> April-only fixed-parameter four-arm balance OOF sweep
                -> frozen May/June/July outer OOF
```

Required side-local sequence:

1. Run feature eligibility, redundancy pruning, staged selection, and automatic
   MDA stopping independently for long and short.
2. Run the archetype geometry sweep independently per side. Geometry class
   thresholds and support gates may differ by side.
3. Run CatBoost HPO independently per side using the selected side contract and
   geometry. Keep three purged chronological HPO folds: the third fold is
   required stability evidence and must not be dropped merely to shorten a
   run. Retain the 75-trial ceiling, but stop after 15 consecutive terminal
   trials without improvement. This bound preserves every improvement in the
   two completed historical CatBoost traces (their longest pre-winner gaps were
   15 and 12 trials; winners were trials 56 and 51) while avoiding their
   unnecessary 18- and 23-trial post-winner tails.
4. Freeze the winning structural HPO parameters, then run a separate
   class-balance mini-sweep independently per side. Evaluate every declared
   balance arm on the exact same purged chronological April-development OOF
   folds and class order; do not let Optuna jointly confound structural
   parameters with the balance choice. No May-July label or outcome may select
   the arm.
5. Select a balance arm only from matched OOF ML and economic evidence. Then
   rematerialize that arm's bounded class weights from the full final
   side-local training labels immediately before the final refit. Never reuse
   fold-local or HPO-subsample weight values in the final model.
6. Freeze feature, geometry, structural-HPO, and balance choices before
   2026-05-01. Generate exact May, June, and partial-July outer OOF folds. Each
   fold trains only on earlier decisions whose labels resolved before the
   validation month, with an exact 24-hour embargo; fold class weights are
   rematerialized from that fold's training labels only.
7. Only after outer OOF is complete, refit the winning per-side configuration
   on all complete labels, persist the final model, and retain the untouched
   May-July predictions as the canonical OOF stream.

Canonical invocation pattern, run one side and one stage at a time:

```bash
SIDE=long  # repeat independently with SIDE=short
CLASSIFIER_ROOT=data_perp/reports/catboost_path_archetype_packb31_8_structural_balance_20260725_v1
GEOMETRY_ROOT=data_perp/reports/catboost_path_archetype_geometry_packb31_8_20260725_v1
LABELS=data_perp/artifacts/20260724_path_archetype_labels_v9_packb31_8_top40/path_archetype_labels.parquet
CONTEXT_ROOT=data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm
CANDIDATE_ROOT=data_perp/artifacts/packb_side_local_top40_20260724_v1_31_8

python3 scripts/run_catboost_path_archetype_classifier.py \
  --input "$LABELS" --side "$SIDE" \
  --output-dir "$CLASSIFIER_ROOT/side=$SIDE" --stage selection_only \
  --feature-dir data_perp/features/20260711_070000 \
  --canonical-candidate-path "$CANDIDATE_ROOT/base_candidate_population.parquet" \
  --canonical-candidate-manifest "$CANDIDATE_ROOT/manifest.json" \
  --canonical-context-path "$CONTEXT_ROOT/context.parquet" \
  --canonical-context-manifest "$CONTEXT_ROOT/manifest.json" \
  --frozen-ae-gmm-sidecar "$CONTEXT_ROOT/context.parquet" \
  --frozen-ae-gmm-manifest "$CONTEXT_ROOT/manifest.json" \
  --discovery-end 2026-05-01T00:00:00Z \
  --development-end 2026-05-01T00:00:00Z \
  --resource-min-free-ram-gib 2 --resource-max-process-rss-gib 12 \
  --resource-min-free-disk-gib 10

python3 scripts/run_catboost_path_archetype_geometry_search.py \
  --input "$LABELS" --side "$SIDE" \
  --output-dir "$GEOMETRY_ROOT" \
  --geometry-prerequisite "$CLASSIFIER_ROOT/side=$SIDE/geometry_prerequisite.json" \
  --canonical-context-manifest "$CONTEXT_ROOT/manifest.json" \
  --feature-dir data_perp/features/20260711_070000 \
  --frozen-ae-gmm-sidecar "$CONTEXT_ROOT/context.parquet" \
  --evaluation-mode short_history_purged_april_v1 \
  --resource-min-free-ram-gib 2 --resource-max-process-rss-gib 12 \
  --resource-min-free-disk-gib 10

python3 scripts/run_catboost_path_archetype_classifier.py \
  --input "$LABELS" --side "$SIDE" \
  --output-dir "$CLASSIFIER_ROOT/side=$SIDE" --stage model_hpo_final \
  --feature-dir data_perp/features/20260711_070000 \
  --canonical-candidate-path "$CANDIDATE_ROOT/base_candidate_population.parquet" \
  --canonical-candidate-manifest "$CANDIDATE_ROOT/manifest.json" \
  --canonical-context-path "$CONTEXT_ROOT/context.parquet" \
  --canonical-context-manifest "$CONTEXT_ROOT/manifest.json" \
  --frozen-ae-gmm-sidecar "$CONTEXT_ROOT/context.parquet" \
  --frozen-ae-gmm-manifest "$CONTEXT_ROOT/manifest.json" \
  --geometry-contract "$GEOMETRY_ROOT/side=$SIDE/geometry_contract.json" \
  --discovery-end 2026-05-01T00:00:00Z \
  --development-end 2026-05-01T00:00:00Z \
  --hpo-trials 75 --hpo-folds 3 --hpo-no-improvement-trials 15 \
  --embargo-hours 24 \
  --resource-min-free-ram-gib 2 --resource-max-process-rss-gib 12 \
  --resource-min-free-disk-gib 10
```

Class-balance mini-HPO requirements:

- Include no balancing as the control.
- Evaluate the fixed arms `uniform`, `frequency_power_0.25`,
  `frequency_power_0.50`, and `frequency_power_0.75`, with the declared maximum
  class-weight ratio cap. Production coverage is incomplete unless all four
  arms finish.
- Fit each fold's balance weights from that fold's training labels only. OOF
  labels and realized path outcomes are evaluation-only.
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

Use a conservative lexicographic promotion gate rather than an opaque weighted
score:

1. All four arms must complete on identical folds and pass the aggregate and
   per-fold no-collapse guards; otherwise select `uniform` and mark the sweep
   non-promotable.
2. Relative to `uniform`, aggregate log loss, macro Brier, and RPS must not
   worsen, and macro F1 must not decline. Retain paired fold deltas.
3. The candidate must produce strictly positive aggregate uplift in realized
   `path_arch_final_return_net_1pct` within the predeclared top 20% of its OOF
   predicted EV, with non-positive EV-MAE change. At least three of four folds
   and at least half of adequately supported UTC months must have non-negative
   top-20% EV uplift.
4. Worst-supported-month top-20% net EV and the worst paired monthly delta must
   be no worse than `uniform`.
5. Ties or any failed gate select `uniform`. Among survivors, break ties by
   larger aggregate top-20% EV uplift, lower EV MAE, lower log loss, then lower
   frequency exponent.

For the economic comparison, estimate class-to-outcome priors exclusively from
each exact OOF fold's training rows and apply the same frozen priors to every
arm. Use canonical v9 outcomes without double-counting costs:
`path_arch_final_return_net_1pct`, `path_arch_peak_mfe_atr`,
`path_arch_mae_12h_r`, `path_arch_mae_before_meaningful_mfe_r`,
`path_arch_stop_before_meaningful_mfe`, `path_arch_reaches_meaningful_mfe`,
conditional `path_arch_time_to_first_meaningful_mfe_h`,
`path_arch_peak_retention_ratio`, and finite
`path_arch_time_to_trailing_h`. Keep R and ATR quantities separate, keep
unreached timing rows censored rather than assigning a fake horizon time, and
report unsupported monthly slices explicitly rather than silently pooling
them.

Persist `class_balance_mini_sweep_report.json` before applying any promotion
gate, then persist `class_balance_economic_oof_report.json`. Bind both reports,
the complete economic-selector configuration, the structural-HPO contract,
the selected-feature contract, the side-local geometry contract, and the exact
ordered candidate identities by deterministic digests in the selected-arm and
final-weight provenance. Fail closed before the final refit when an arm has no
complete OOF result, identities or folds differ, the input contains more than
one side, or any required digest is absent or inconsistent. A fully covered
uniform control selected because no non-uniform arm passes is a valid
production selection; a smoke or incomplete-coverage uniform fallback is not.

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

#### 2026-07-25 long-side CatBoost result

The long-side pipeline is complete on the canonical 31/8 handoff. It selected
75 features, completed an independent geometry sweep, structural HPO,
four-arm April balance sweep, fixed May/June/partial-July outer OOF, and final
refit. The fully covered uniform arm was retained because no non-uniform arm
passed the strict matched ML and economic promotion gates.

The untouched outer OOF contains 64,504 unique long candidate IDs. Aggregate
log loss is `1.704679`, RPS is `0.208295`, macro Brier is `0.114178`, macro F1
is `0.120990`, and weighted F1 is `0.201114`. Against fold-local train-prior
baselines, log loss improves by `0.004798` in May, `0.003263` in June, and
`0.013944` in partial July; RPS improves in all three folds as well.

This is a classification pass with an explicit concentration warning, not a
standalone economic promotion. `dead_timeout` receives 58.0% of aggregate hard
predictions and three low-support classes never win argmax. More importantly,
the probability-weighted top 20% from fold-train-only economic priors has
negative realized net EV in May (`-0.006782`), June (`-0.016409`), and partial
July (`-0.000707`). CatBoost probabilities may therefore enter execution-EV
only as context/risk inputs. The CatBoost branch must show positive paired OOF
uplift over alpha plus auxiliaries before policy admission.

The authoritative validation record, including hashes and monthly prior
comparisons, is
`docs/pipeline_roadmap/20260724/r3/catboost_long_validation_20260725_v1.json`.

#### 2026-07-25 short-side CatBoost result

The short-side pipeline is also complete on the canonical 31/8 handoff. It
selected 75 features, completed its own geometry sweep, structural HPO,
four-arm April balance sweep, fixed May/June/partial-July outer OOF, and final
refit. HPO stopped safely after 40 of the requested 75 trials when the
15-trial no-improvement rule was met; trial 24 remained the winner. The
uniform balance arm was retained because no non-uniform arm passed the strict
matched ML and economic promotion gates.

The untouched outer OOF contains 69,272 unique short candidate IDs. Aggregate
log loss is `1.665006`, RPS is `0.195542`, macro Brier is `0.111815`, macro F1
is `0.143173`, and weighted F1 is `0.233746`. Against fold-local train-prior
baselines, log loss improves by `0.040401` in May, `0.041905` in June, and
`0.027617` in partial July; RPS improves by `0.009196`, `0.007182`, and
`0.007972`, respectively. These improvements validate the new side-local
75-feature model; there is no metric-based reason to fall back to an earlier
feature list.

This is likewise a classification pass with a concentration warning, not a
standalone economic promotion. `dead_timeout` receives 45.9% of aggregate
hard predictions and three low-support classes never win argmax. The
probability-weighted top 20% from fold-train-only economic priors has negative
realized net EV in May (`-0.008606`), June (`-0.002989`), and partial July
(`-0.018316`). Short CatBoost probabilities may enter execution-EV only as
context/risk inputs and must prove positive paired OOF incremental value.

The authoritative validation record, including hashes and monthly prior
comparisons, is
`docs/pipeline_roadmap/20260724/r3/catboost_short_validation_20260725_v1.json`.

### Phase 4: Finish the Five Auxiliary LGBM Heads

Goal: produce OOF predictions and final models for all five targets.

Canonical runner:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. <PYTHON> -u \
  scripts/run_path_auxiliary_role_bundles.py \
  --labels-path \
  data_perp/artifacts/packb_path_auxiliary_targets_20260725_v1_31_8/targets.parquet \
  --context-path \
  data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm/context.parquet \
  --selection-hpo-reference-end 2026-05-01T00:00:00Z \
  --feature-dir data_perp/features/20260711_070000 \
  --output-dir \
  data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8 \
  --n-trials 40 \
  --hpo-patience 12 \
  --seed 42 \
  --purge-hours 13 \
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
- Feature selection uses a purged April validation tail with only resolved
  December-March rows available for training. The base selector's 365-day
  burn-in is replaced by its chronological short-history fallback for binary
  roles; shuffled fallback is forbidden. Regression roles use one April
  validation month rather than the impossible six-month default and require at
  least 200 validation rows per selector fold. The canonical top-40 label
  stream begins in April, so feature selection uses strictly chronological
  train-before-validation splits within the April development window; May,
  June, and July remain untouched outer OOF evaluation months.
- Refactor the current global pre-screen before resuming; a global pre-screen or
  global selected-feature union does not satisfy this contract.
- No global selected-feature union and no shared fitted selector.
- Lower redundancy threshold than the alpha head.
- No hard-coded feature count.
- Automatic MDA stop.
- Preserve observable base archetype encodings.
- HPO after feature selection.
- Up to 40 trials per unique role study, with stale-trial stopping after 12.

Target-specific sample weights remain bounded between 0.5 and 2.0.

Required metrics:

- Regression MAE/RMSE/Huber loss as appropriate.
- Rank IC computed independently per side.
- Top-1/5/10 economic diagnostics.
- Monthly and side stability.
- Support and missingness.
- Calibration of any event-probability supportive output.

Gate:

- Eleven reusable role streams exist for both sides, including the shared
  meaningful-MFE event/timing family; five composed head bundles exist on the
  exact candidate keys.
- Side-local final models, feature lists, parameters, and hashes exist for
  every role.
- No supportive realized label is present at inference.

#### 2026-07-25 canonical auxiliary result and promotion audit

The canonical run is structurally complete and leakage-safe:

- 300,315 exact candidates and 195,931 identical May-July OOF IDs across all
  five heads.
- May/June/July OOF support is 99,992 / 73,211 / 22,728 respectively; July is
  only July 1-10 and must not be described as a full validation month.
- All outer training labels resolve before validation starts, the 13-hour
  decision/label purge is present on every fold, and final-refit models are
  excluded from OOF predictions.
- All 22 side-role contracts use independent long/short selection and HPO.
- The 514 resource checkpoints all passed. Peak process RSS was 4.744 GiB,
  minimum available RAM was 9.243 GiB, and minimum free disk was 76.865 GiB.

Artifact-integrity completion is **not** production promotion. The first three
heads may enter an identical-row execution-EV OOF ablation, but none is
production-approved yet:

| Head/component | May-July OOF finding | Current action |
|---|---|---|
| Peak conditional mean | Learnable: monthly R² long `0.438/0.131/0.274`; short representation-available `0.513/0.262/0.188` | Keep as an execution-EV ablation candidate |
| Peak conditional q80 | Invalid as a range estimate: long empirical coverage `1.000/1.000/0.905` instead of 0.80; pooled long coverage `0.989` and negative pinball skill | Withhold q80 from downstream inputs. For the next training cycle, raise/remove the 10-ATR target ceiling or use a lower identifiable quantile, then require per-side monthly calibration |
| Timing CDF | Two-hour AUC is useful (`0.596-0.664` long, `0.617-0.658` short available) and the projected CDF has zero monotonicity violations | Pass the horizon probabilities, especially 2h/4h, as separate ablation features; do not rely only on expected censored time |
| Expected censored time | Weak: monthly R² long `0.018/-0.002/-0.030`, short available `0.013/-0.071/0.031` | Diagnostic until it adds cost-aware execution value |
| Expected MAE mixture | Not learnable as currently composed: monthly R² long `-0.008/-0.142/-0.206`, short available `0.010/-0.092/0.003` | Do not promote. Replace/augment it with economically defined tail probabilities such as stop-before-meaningful-MFE and MAE-above-0.5/1.0R, then re-test |
| Confirmed adverse trough | R² is approximately `-0.021` to `+0.020`, IC only `0.06-0.15`, and target support is incomplete | Model trough occurrence/competing risk first, then conditional timing; keep blocked from policy use |
| Favorable slope | Positive but weak R²/IC; July is partial | Retain as diagnostic only and require incremental cost-aware EV plus two later full months |

Individual-head improvement assessment:

1. **Peak MFE:** keep the conditional mean, whose pooled OOF IC is `0.557`
   long and `0.535` short. Remove the current long q80 output from model inputs:
   its empirical coverage is `98.9%`, IC is `0.070`, and its `10 ATR` ceiling
   has turned it into an almost constant upper bound. In the next side-local
   training cycle compare an uncapped/wider-cap q70, q75 and q80, calibrate the
   selected quantile on training-only OOF residuals, and require monthly
   coverage within five percentage points of its nominal quantile. Also add
   economically aligned probabilities of clearing the deployed take-profit
   and trailing-activation levels.
2. **Time to meaningful MFE:** retain the monotone CDF representation. The 2h
   classifier is the strongest component (`0.634` AUC on both sides), while
   discrimination weakens at longer horizons. Replace the four independent
   binary fits with one discrete-time hazard model or a jointly constrained
   ordinal CDF; add policy-relevant 1h, 6h and 12h/24h checkpoints only where
   the exit policy can still be active. Select horizons by paired
   execution-EV contribution, not target AUC alone.
3. **MAE before meaningful MFE:** the conditional-hit rank signal is useful on
   long (`0.277` IC) but weak on short (`0.119`); no-hit IC is only
   `0.080/0.133`. Do not use the current conditional-mean mixture as the main
   risk feature. Train side-local competing-risk outputs for
   stop-before-meaningful-MFE and probabilities of crossing the actual
   side/archetype stop, 0.5R and 1.0R before a useful move. Add conditional
   q80/q90 adverse depth and calibrate these probabilities OOF.
4. **Bars before price stops decreasing:** neither unconditional clock is
   adequate. The confirmed-trough IC is `0.112/0.096`; the legacy clock has
   higher IC (`0.200/0.229`) but measures a different, less actionable event.
   First predict whether a confirmed trough occurs before stop, meaningful MFE
   or timeout; then fit its interval/hazard conditionally. Compare confirmation
   rules of one, two and three bars and 25%/50% adverse recovery on identical
   OOF rows. The winner is chosen only through entry-timing or execution-EV
   economics.
5. **Future slope:** current OOF IC is positive but weak and nearly identical
   by side (`0.160/0.159`). Replace the single endpoint-like 12h regression
   with robust multi-horizon favorable-path efficiency: median/Huber slopes at
   2h, 4h, 8h and the remaining policy horizon, plus time-to-50%/80% peak.
   Orthogonalize candidate slope features against peak MFE and timing CDF on
   each training fold, and promote only if the residual slope signal improves
   aggregate and worst-month execution EV.

These are five separate challenger studies. Each inherits the existing
side-local feature selection/HPO and exact OOF folds, changes one head at a
time, and is compared with its current head on identical candidate IDs. A
better target metric alone is insufficient: the final gate is incremental
cost-aware execution EV after the 21-day admission correction.

The machine-readable assessment is
`docs/pipeline_roadmap/20260724/r3/execution_ev_aux_head_individual_assessment_20260725_v1.json`.

Short representation-missing rows are 16,400 of 106,987 OOF rows (15.3%);
June alone is 10,192 of 41,454 (24.6%). Every execution-EV ablation must report
this slice separately. A head can be materialized for a research OOF ablation
without being production-promoted; production promotion requires positive
aggregate and worst-month identical-row economics, acceptable side/month
calibration, and later full-month confirmation.

#### 2026-07-25 meaningful-MFE event-classifier ablation

The conditional peak-size head cannot be made unconditional by multiplying it
by a weak or ambiguously defined event probability. The implementation now
keeps two separate probabilities:

1. `P(reach meaningful MFE within 12h)`, the literal opportunity event used by
   the existing peak mixture; and
2. `P(reach meaningful MFE before an ATR-normalized adverse barrier)`, a
   stricter clean/capturable-opportunity event for execution risk.

The research runner is
`scripts/run_meaningful_mfe_event_classifier_ablation.py`; deterministic label
and fold primitives are in
`extreme_price_movements/meaningful_mfe_event_ablation.py`. The verified
artifact is:

`data_perp/artifacts/meaningful_mfe_event_classifier_ablation_20260725_v1`

The soft triple-barrier baseline is side-normalized and uses a 12-hour timeout,
an upper barrier of `max(1.5 ATR, 1.5% entry return)`, a 1.0-ATR adverse
barrier, continuous favorable-versus-adverse barrier progress, and a bonus for
an earlier clean favorable hit. If the hourly target store says both barriers
were touched in the meaningful-hit hour, the label conservatively assigns the
adverse outcome because intrabar order is not observable. Baseline support is
283,904 valid labels: 107,187 favorable-first, 139,013 adverse-first/conflict,
and 37,704 timeouts.

All models are fit separately by side. Hyperparameters are selected only on
April 22-30 after requiring every training label to resolve before April 22.
May, June, and July 1-10 are expanding outer OOF folds with the same
label-resolution rule. The 179,520 scored OOF rows and the 127,777
exact-policy economic rows are identical across arms. July remains previously
inspected partial-month research evidence, not an untouched final test.

The model-family ablation compared weighted logistic regression, LightGBM, and
CatBoost under the same soft target. Label ablations on the April-selected
LightGBM family compared literal hard reach, hard triple barrier, 0.5/1.0/1.5R
adverse barriers, and removal of the early-hit bonus. April selected LightGBM
for both sides; therefore CatBoost's later pooled OOF result is a research
challenger and cannot be selected retroactively.

| Arm | Literal reach AUC / AP / Brier | Literal top-10 precision | Clean triple-barrier AUC / AP | Clean top-10 precision | Exact-policy global / timestamp-side top-10 |
|---|---:|---:|---:|---:|---:|
| Existing event classifier | `0.5567 / 0.5268 / 0.2493` | `54.67%` | `0.5453 / 0.4166` | `43.70%` | `-123.10 / -94.20 bps` |
| Logistic, soft TB | `0.4988 / 0.4922 / 0.2797` | `50.08%` | `0.5036 / 0.3964` | `41.87%` | `-131.43 / -88.24 bps` |
| LightGBM, soft TB | `0.5179 / 0.5030 / 0.2620` | `51.05%` | `0.5274 / 0.4096` | `42.74%` | `-123.54 / -82.98 bps` |
| CatBoost, soft TB | `0.5332 / 0.5201 / 0.2559` | `55.35%` | `0.5418 / 0.4266` | `46.84%` | `-99.11 / -83.60 bps` |
| LightGBM, hard literal reach | `0.5468 / 0.5262 / 0.2641` | `55.56%` | `0.5335 / 0.4123` | `43.52%` | `-114.35 / -95.65 bps` |
| LightGBM, hard TB | `0.5457 / 0.5233 / 0.2739` | `54.36%` | `0.5411 / 0.4187` | `44.15%` | `-124.59 / -91.05 bps` |
| LightGBM, soft TB 0.5R lower | `0.5161 / 0.5020 / 0.2674` | `50.78%` | `0.5293 / 0.4125` | `43.33%` | `-113.01 / -76.35 bps` |
| LightGBM, soft TB 1.5R lower | `0.5218 / 0.5059 / 0.2622` | `51.28%` | `0.5270 / 0.4086` | `42.45%` | `-134.39 / -83.96 bps` |
| LightGBM, soft TB without time bonus | `0.5194 / 0.5045 / 0.2616` | `50.96%` | `0.5290 / 0.4116` | `42.83%` | `-123.69 / -77.94 bps` |

No arm is promotable. The existing classifier remains best on literal reach
AUC, Brier loss, and log loss. Hard literal training improves top-decile
precision by only 0.89 percentage point while degrading its full-distribution
metrics. CatBoost improves clean-event average precision by 1.00 percentage
point and top-decile precision by 3.14 points relative to the incumbent; it
also improves the exact-policy global tail by 23.99 bps. Those returns remain
negative, and CatBoost was not the April-selected family. The 0.5R soft arm has
the least-negative timestamp-side tail, but its literal-event classification
is materially worse.

The next event-head study is therefore an explicit feature study, not another
uncontrolled estimator sweep:

1. Redo feature selection separately by side for the literal event and clean
   competing-risk event; the present run deliberately reuses the frozen
   55-long/67-short event features to isolate label and model effects.
2. Add only causal path-risk inputs available at decision time: volatility
   regime and ATR stability, spread/cost regime, distance to policy profit and
   stop levels in ATR, liquidity, recent gap/jump state, trend/path efficiency,
   and calibrated base/residual uncertainty. No realized auxiliary target may
   enter as a feature.
3. Fit and OOF-calibrate `P(reach)`, `P(adverse first)`, and `P(timeout)` as a
   side-local competing-risk system. Compare this with separate binary heads;
   require probability coherence and report every side/month.
4. Use `P(reach)` to uncondition peak size. Pass `P(clean reach)` separately to
   execution EV and timing. Do not substitute the latter for the literal
   event probability.
5. Select the challenger on training-only folds, then require improvement in
   literal AUC/AP/Brier, top-10 precision and recall, aggregate and worst-month
   exact-policy net EV, and both pre/post-21-day-admission tails. Preserve one
   later full month as untouched confirmation.

#### 2026-07-25 CatBoost v2 and config base-residual ablations

The proposed CatBoost improvements are now implemented and evaluated in two
leakage-safe research runners:

- `scripts/run_meaningful_mfe_catboost_v2_ablation.py`
- `scripts/run_meaningful_mfe_base_residual_catboost_ablation.py`

Their verified artifacts are:

- `data_perp/artifacts/meaningful_mfe_catboost_v2_ablation_20260725_v1`
- `data_perp/artifacts/meaningful_mfe_base_residual_catboost_ablation_20260725_v1`

Both studies use April 22-30 for feature/HPO/architecture selection, require
all training labels to resolve before April 22, and freeze their choices for
expanding May, June, and partial-July OOF. Rolling Platt calibration is causal:
May uses April held-out predictions, June adds May OOF, and July adds June OOF.
July 1-10 remains inspected research evidence, not an untouched test.

CatBoost v2 implements:

- separate hard clean-event probability and soft-score models;
- a three-class favorable-first/adverse-first/timeout competing-risk model;
- a conditional-quality regressor trained only on favorable-first rows;
- a 0.35-weight treatment for 27,217 order-ambiguous adverse rows;
- separate long/short task-specific feature selection from 325/315
  role-feature pools;
- frozen versus task-top-40 versus task-top-80 feature ablations;
- four CatBoost geometries including depths 4/6/8 and Lossguide;
- three-seed hard-event ensemble, lower-confidence-bound score, native
  archetype context, and causal rolling calibration;
- probability × conditional-quality and probability-gated-quality
  compositions.

Both sides selected task-top-80 plus seven mandatory features. Long selected
depth 8 symmetric trees; short selected depth 8 Lossguide. The main results
are:

| V2 arm | Clean AUC / AP / Brier | Clean top-10 | Literal AUC | Global / timestamp-side top-10 |
|---|---:|---:|---:|---:|
| V1 CatBoost soft TB | `0.5418 / 0.4266 / 0.2506` | `46.84%` | `0.5332` | `-99.11 / -83.60 bps` |
| Hard single | `0.5632 / 0.4344 / 0.2357` | `46.76%` | `0.5649` | `-121.05 / -85.56 bps` |
| Hard 3-seed ensemble | `0.5639 / 0.4377 / 0.2353` | `47.70%` | `0.5666` | `-112.91 / -86.25 bps` |
| Hard ensemble + rolling Platt | `0.5704 / 0.4394 / 0.2358` | `47.69%` | `0.5732` | `-97.94 / -86.25 bps` |
| Ambiguity-weighted hard | `0.5579 / 0.4309 / 0.2372` | `46.17%` | `0.5672` | `-126.88 / -90.12 bps` |
| Competing-risk favorable probability | `0.5669 / 0.4362 / 0.2346` | `47.24%` | `0.5649` | `-110.81 / -88.67 bps` |
| Native categorical context | `0.5593 / 0.4327 / 0.2364` | `46.86%` | `0.5618` | `-116.90 / -84.58 bps` |
| Soft score with new selection/HPO | `0.5487 / 0.4305 / 0.2464` | `47.37%` | `0.5371` | `-94.41 / -80.70 bps` |
| Calibrated probability × quality | `0.5743 / 0.4439 / 0.2432` | `48.41%` | `0.5758` | `-94.46 / -84.81 bps` |
| Calibrated probability-gated quality | `0.5745 / 0.4472 / 0.2532` | `47.81%` | `0.5760` | `-95.06 / -84.26 bps` |

Task-specific selection, expanded geometry, ensembling, and separation of
probability from conditional quality all work statistically. Ambiguity
downweighting and native categorical context do not. The soft-score arm has the
least-negative timestamp-side tail, while the probability × quality arms have
the strongest classification. No arm has positive exact-policy economics, so
none is promotable.

The additional alpha-like architecture resolves `config.py` feature-family
aliases into distinct configured pools, then applies:

```text
configured base features
  -> side-local CatBoost clean-event probability
  -> expanding cross-fitted base OOF probability
configured meta features + base OOF probability
  -> CatBoost probability residual
  -> April-selected residual shrinkage
  -> rolling Platt
  -> strongest v2 conditional-quality composition
```

The available config pools contain 547/564 base features for long/short and
795 meta features. Training-only screening selects 80 base features per side
against the clean event and 80 meta features per side against the cross-fitted
base residual. Every residual-training row uses a base prediction from a model
which did not train on that row. April selected residual shrinkage 0.50 long
and 0.25 short.

| Config architecture | Clean AUC / AP / Brier | Clean top-10 | Literal AUC | Global / timestamp-side top-10 | Worst side-month |
|---|---:|---:|---:|---:|---:|
| Base only, rolling Platt | `0.5629 / 0.4329 / 0.2349` | `45.79%` | `0.5755` | `-101.05 / -101.56 bps` | `-184.99 bps` |
| Base + residual, raw | `0.5614 / 0.4346 / 0.2345` | `46.94%` | `0.5709` | `-97.68 / -102.66 bps` | `-176.69 bps` |
| Base + residual, rolling Platt | `0.5640 / 0.4360 / 0.2351` | `46.39%` | `0.5743` | `-98.53 / -102.66 bps` | `-176.69 bps` |
| Base + residual probability × quality | `0.5707 / 0.4420 / 0.2437` | `47.37%` | `0.5790` | `-94.54 / -98.43 bps` | `-186.08 bps` |
| V2 probability × quality comparator | `0.5743 / 0.4439 / 0.2432` | `48.41%` | `0.5758` | `-94.46 / -84.81 bps` | `-195.60 bps` |

The residual learner is genuinely selected and improves base-only clean AP,
top-decile precision, global policy tail, and worst side-month. It does not
improve clean AUC/Brier consistently, and its timestamp-side economics are
materially worse. When combined with conditional quality it produces the best
literal-event AUC (`0.5790`) and a slightly better worst side-month than v2,
but loses clean ranking and timestamp-side economics. It remains a useful
literal-opportunity challenger, not the strongest execution-policy score.

The causal first-21-day isotonic admission diagnostic does not rescue these
scores. It admits too few rows to support a conclusion: 0 rows for v2
probability × quality, 4 for v2 gated quality, 19 for config base-residual
probability, and only 1-2 for its probability ×/gated quality compositions.
Some resulting top-10 means are positive, but supports of 1-19 trades are
economically meaningless and fail the admission support gate. This confirms
that an event probability is not an execution-EV admission score.

Next action:

1. Keep v2 calibrated hard probability and conditional quality as separate OOF
   inputs; retain config base-residual probability as a third challenger.
2. Train the execution-EV head on the three probabilities
   `P(reach)`, `P(clean favorable first)`, and `P(adverse first)`, conditional
   peak/MAE, and quality. Do not select a trade on any event score alone.
3. Train the execution-EV head, then run the full exit policy and its 21-day
   admission ablation on identical rows. Require adequate admitted support plus
   positive aggregate, timestamp-side, and worst-month net EV.
4. Materialize exact higher-resolution paths before testing different upper
   barriers or shorter timeouts; the current store cannot support those
   geometries without approximation.

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
- Only promotion-audited auxiliary OOF candidates. Initially this means peak
  conditional mean/expected peak and separate timing-horizon probabilities.
  Expected MAE, confirmed adverse-trough timing, favorable slope, and peak q80
  remain diagnostic or withheld until their stated gates pass.
- Seven CatBoost class probabilities and aggregate confidence/risk fields.
- Execution-EV label and metadata, used only as targets/reporting.

Build this handoff separately for long and short. Do not concatenate sides
before execution-EV fitting.

Gate:

- Row identity and fold provenance are complete.
- Joined rows reconcile to the intersection expected by the manifests.

#### 2026-07-25 canonical alpha execution-OOF bridge

The canonical alpha branch is now ready for the strict joined handoff:

- `195,931` exact residual-alpha OOF rows: `88,944` long and `106,987`
  short.
- Fold support is `99,992` May, `73,211` June, and `22,728` partial July,
  matching the auxiliary OOF calendar.
- Every row is bound to the original candidate, frozen-context, residual
  model, per-side feature/HPO, row-identity, and fold/cutoff hashes.
- The bridge is non-mutating. It emits compact supplemental ledgers and
  manifests; it does not rewrite the canonical Pack-B, context, or residual
  artifacts.
- Residual predictions preserve their observed availability at
  `signal timestamp + 1h`, which is the execution decision timestamp required
  by the label contract. The input validators now reject only values arriving
  after that execution decision, rather than incorrectly rejecting valid
  close-of-hour predictions.
- The historical residual-alpha value remains in its proven 1%-cost basis.
  The joined handoff must still reconcile it once to the canonical
  p90-spread-plus-fee execution-label cost before any direct/residual
  comparison.

Artifacts:

```text
data_perp/artifacts/execution_ev_canonical_alpha_inputs_20260725_v1
data_perp/artifacts/execution_ev_alpha_oof_20260725_v1
```

The alpha OOF Parquet SHA-256 is
`0b30798e8dbdc228bf434b33b2aee0227e01d23fc55dad4c670824aaa6268a4d`.
This completes only the alpha branch; CatBoost, promotion-audited auxiliary
streams, labels, and their exact intersection still have to pass the joined
handoff gate.

#### 2026-07-25 canonical auxiliary execution-OOF adapters

All five auxiliary streams are now normalized on the same `195,931` exact
May-July OOF identities as alpha:

```text
data_perp/artifacts/execution_ev_auxiliary_oof_20260725_v1
```

Their signed Parquet hashes are:

| Stream | SHA-256 | Initial model-input status |
|---|---|---|
| Expected peak MFE | `e34f6825a49f8a3e7b07b8e725a4f267c34e45007185af522623da89de6ec356` | Admissible research ablation |
| Timing scalar + 2h/4h/8h/12h CDF | `d5a903de33f062f106d981754b08cf7aca3c61d9548c1290018c84e77f6174c3` | CDF probabilities admissible; weak scalar is fallback/diagnostic |
| Expected MAE mixture | `2840b9b94efcbf95e0185c73ad2d9e96eae34f9447410ba999cb1a81ecf5f157` | Withheld after learnability audit |
| Confirmed adverse-trough clock | `377b24905816666e61f9eaa6f3303b190b25d17220d17d8f263d5460ddf31f55` | Explicit research-only override |
| Favorable slope | `991a3b87a823fd7fd7a1f6a605a9854b8df5c93969a803fef12a0680efa09b1d` | Explicit research-only override |

The joined-handoff provenance now marks MAE, adverse-turn, slope, and the weak
scalar timing expectation as non-model inputs whenever the signed timing-CDF
vector is present. The execution trainer ignores all `model_input=false`
features and predeclares paired staged arms:

1. alpha only;
2. alpha plus frozen context;
3. alpha/context plus peak and timing CDF;
4. alpha/context plus CatBoost;
5. all admissible features.

This directly tests whether CatBoost adds value over alpha plus the admissible
auxiliary heads. Feature selection remains side-local and train-only inside
each outer fold; model HPO is performed once per direct/residual target and
side, then the named arms inherit the frozen selection/HPO contract so the
comparison is paired and does not multiply search cost.

#### 2026-07-25 execution-label lineage repair

The historical execution-label artifact is not a valid canonical join source:
it was built from an older candidate universe and matched only `63,595` of the
`195,931` current alpha OOF identities. No fuzzy or timestamp-only join is
allowed.

Fresh labels were therefore rematerialized from the current Pack-B path-label
population under the unchanged frozen policy and cost geometry:

```text
data_perp/reports/
execution_ev_12h_labels_packb31_8_p90spread_fee30bps_20260725_v1
```

The run produced `201,685` labels from `222,490` complete current path rows.
It preserves the one-hour signal-to-decision delay, 12-hour horizon, frozen
long/short policy geometry, 30 bp round-trip fee, full per-symbol p90 spread,
and explicit missing-spread drop policy.

**Superseded for promotion:** the 2026-07-25 exit-policy audit proved that this
is not the executable policy outcome. It is an hourly 12-hour approximation
with side-parent geometry. The deployed contract is an exact one-minute,
1,440-minute replay using observable side x policy-archetype geometry with
side-parent fallback, the `joint_trailing_total_mfe_raw_bayesian_v1` pathway,
and the deployed cost/executable-price handling. Keep the existing labels and
models as diagnostics only. Do not promote, integrate with
`simple_policy_optimiser`, or start entry-timing training from this target.

Before CatBoost intersection, the fresh labels match `132,644` of the
`195,931` alpha/auxiliary OOF rows. The remaining rows are mostly candidates
without a complete CatBoost/path target and therefore cannot enter a paired
CatBoost execution-EV test. On the completed long CatBoost stream, `61,618`
of `64,504` rows have execution labels; the `2,886` excluded rows are
concentrated in symbols outside the frozen eligible-spread map. Preserve and
report this attrition; do not invent spreads or join across different
candidate IDs.

### Phase 6: Train Direct and Residual Execution-EV Models

Goal: determine whether execution EV should be predicted directly or as a
residual correction to alpha EV.

The runner now has an explicit `--production` profile. It uses
`execution_decision_utc` rather than the signal timestamp for availability,
splitting, and training; admits the full May-July OOF span; defaults to three
outer folds, 40 HPO trials, 1,500 LightGBM estimators, 100-round early stopping,
and three worker threads; and retains override flags for bounded reruns.
The timing-risk companion is disabled unless explicitly requested, so timing
work cannot silently run before the execution-EV winner is stable.

Direct target after the mandatory exit-label repair:

```text
realized causal net EV from the exact deployed exit-policy replay
```

Residual target after the mandatory exit-label repair:

```text
realized causal net EV from the exact deployed exit-policy replay
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
- Global top-k as primary: pool all identical shared outer-OOF rows across
  timestamps and sides; do not compute a per-timestamp top decile and average
  it.
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
2. Alpha plus each auxiliary head independently.
3. Alpha plus CatBoost.
4. Alpha plus all five auxiliary heads.
5. Alpha plus opportunity auxiliaries: peak, timing, and favorable slope.
6. Alpha plus adverse-risk auxiliaries: MAE-before-MFE and adverse-turn.
7. Leave each auxiliary head out of the all-five bundle, one at a time.
8. Alpha plus auxiliary heads plus CatBoost.
9. Remove timing features.
10. Remove adverse-path features.
11. Remove AE/GMM/OOD/support context.
12. Direct versus residual target.
13. For each auxiliary head independently, replace its continuous target with
    an economically anchored soft-binary target while holding the other four
    heads, rows, feature contract, and HPO budget fixed.
14. Route timing and MAE features to the entry-timing meta head only: compare
    execution EV without those two heads against the shared-routing incumbent,
    then test their add-one and joint value inside timing. Do not infer this
    routing result from an execution-EV leave-one-out arm.
15. Replace the continuous execution-net-EV regression target with a
    cost-aware soft-binary utility target, with a train-only mapping back to a
    comparable expected-net-EV unit.

All comparisons must use identical rows and costs.

Soft-binary targets are probabilities in `[0, 1]`, not hard signs and not
globally normalized future outcomes. Their economic center and transition
band must be fixed from the deployed policy geometry or estimated inside each
side-local training fold only:

- **Peak MFE:** soft probability that peak opportunity clears the deployed
  trailing-activation/profit-conversion threshold, with separate 0.5R and 1R
  diagnostics.
- **Time to meaningful MFE:** soft probability of realization by each
  2h/4h/8h/12h horizon; near-boundary times receive graded credit, while
  right-censored paths remain explicit rather than being treated as long
  regressions.
- **MAE before meaningful MFE:** soft adverse-first probabilities around 0.5R
  and 1R, preserving `P(stop 1R) <= P(adverse 0.5R first)`.
- **Bars before price stops decreasing:** soft probability that adverse
  movement stabilizes before a decision deadline and with sufficient remaining
  stop headroom. Timeout/no-turn paths are a competing outcome, not an
  arbitrary maximum bar count.
- **Future favorable slope:** soft probability that favorable accumulation
  clears a cost- and horizon-aware rate threshold; keep a separate probability
  for failing to recover entry friction.

For execution EV, the primary soft target challenger is
`sigmoid(execution_net_ev_12h / tau_side)`, centered at after-cost break-even.
`tau_side` is selected only from each outer-fold training population and is
frozen before validation. Its OOF probability must be mapped monotonically
back to expected net EV using training rows only, so it can be compared with
the regression head and consumed by policy/timing layers in the same return
unit. Evaluate log loss/Brier/calibration as diagnostics, but promotion still
depends on realized net EV of the global top 10% after the causal 21-day
admission calibrator.

The routing ablation has four explicit arms:

1. execution EV receives neither timing nor MAE; timing meta receives neither;
2. execution EV receives both; timing meta receives neither;
3. execution EV receives neither; timing meta receives both;
4. both meta heads receive both.

Arm 3 is the proposed specialization. Timing and MAE may be retained there
even if they fail execution-EV add-one tests, but only if the timing layer
improves cost-aware realized entry utility, adverse movement, missed-
opportunity rate, and global post-admission top-10 economics on identical OOF
rows.

The implementation now treats heads rather than raw feature families as the
unit of auxiliary attribution. This matters because the timing head emits both
a censored scalar and 2h/4h/8h/12h CDF probabilities. Research-only auxiliary
outputs may enter only arms prefixed `research__`; they remain excluded from
the promotable `all_features` arm. Add-one-head arms force the tested head to
remain present after the frozen side-local selector. All diagnostic arms reuse
the matching all-feature fold/side tuned LightGBM parameters instead of
silently falling back to defaults.

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
- Promotion top-10 is ranked only by expected EV **after** the causal 21-day
  side x archetype admission correction. Raw model and train-only isotonic
  scores are diagnostics and cannot win the promotion leaderboard.
- Every promotion row must declare
  `ranking_scope=global_shared_outer_oof` and
  `ranking_stage=after_causal_21d_admission_calibrator`. If no observable
  correction route is available, promotion fails closed.
- Worst week/month do not materially degrade unless the average gain exceeds
  the allowed trade-off.
- Long and short are both reported.
- Gains are not isolated to one month/archetype.
- Final model emits a comparable expected-EV unit.

#### 2026-07-25 corrected execution-EV run and exit-policy audit

The per-side minimum-row defect is fixed. The corrected run at
`data_perp/reports/execution_ev_meta_packb31_8_20260725_v2_side_min5000`
contains `116,244` OOF predictions. Every direct and residual outer fold now
has at least 5,000 training rows independently for long and short; all 12
fold/side studies completed 40-trial HPO and every fold/side calibration map
was fitted.

On the same OOF rows, the best diagnostic arm was direct `alpha_context`:
top-10 mean net EV `0.0018965` versus `0.0013651` for the existing alpha.
That apparent gain is not stable enough to promote even on its own target:
long contributes only `0.0001304`, the worst week is `-0.0048698`, and partial
July is `-0.0034329`. Adding the currently eligible auxiliary bundle produces
`-0.0020741`; adding CatBoost produces `-0.0021575`.

More importantly, the target itself fails the executable exit-policy contract:

| Contract | Current diagnostic labels | Required deployed policy |
|---|---|---|
| Replay resolution | 1h | exact 1m |
| Maximum horizon | 720 minutes | 1,440 minutes |
| Geometry | side-parent only | side x observable policy archetype, with side-parent fallback |
| Simulator | reduced `simulate_execution_ev_12h` | canonical `simple_policy_optimiser.simulate_and_score` |
| Costs | 30 bp fee + full p90 spread | deployed one-percent round-trip fee and executable-price spread handling |

Therefore the corrected run is **non-promotable diagnostic evidence**. Its
negative auxiliary/CatBoost aggregate tails must not be used to reject those
branches permanently; they must be rerun against repaired labels. Production
execution-EV runs now require `--exit-policy-json`, and the joined target must
carry an exact `exit_policy_contract` matching replay resolution, horizon,
geometry scope, pathway, simulator, trailing curve, and policy SHA-256.

Required next sequence:

1. Preserve the existing immutable one-minute store and target-backfill only
   the missing candidate windows needed for the exact replay. The current
   joined population has `99,518 / 127,777` complete 1m x 1,440-minute paths
   (`77.88%`): long `79.48%`, short `76.40%`; May `69.35%`, June `86.72%`,
   and partial July `84.80%`. The `28,259` missing execution paths warrant a
   candidate-window backfill attempt. This is separate from the earlier
   representation gap and does not authorize broad representation backfill,
   synthetic candles, or manufactured continuity.
2. Materialize exact deployed-policy labels with one-minute paths and a
   1,440-minute label-resolution horizon.
3. Use the observable policy archetype where available and record every
   side-parent fallback; report coverage by side, month, archetype, and
   representation-availability slice.
4. Apply deployed fees and entry/exit spread handling exactly once.
5. Rebuild the strict joined handoff with the signed exit-policy contract.
6. Rerun direct/residual OOF using the expanded five-head add-one, grouped,
   all-five, and leave-one-head-out auxiliary matrix.
7. Reapply aggregate, side, month, week, and worst-period promotion gates.

The authoritative audit is
`docs/pipeline_roadmap/20260724/r3/execution_ev_exit_policy_audit_20260725_v1.json`.

#### 2026-07-25 exact-policy repair and post-admission result

The bounded source repair and independent rescan completed successfully:

- 106/106 affected symbols repaired with zero failed or incomplete symbols.
- 3,076,080/3,076,080 required merged candidate-window minutes covered.
- 127,777/127,777 exact 1m x 1,440-minute candidate paths complete.
- Coverage is 100% for long, short, May, June, and partial July.
- 1,251,475 source rows were fetched in 1,807 bounded requests.

The signed exact-policy labels are:

```text
data_perp/artifacts/execution_ev_policy_labels_20260725_v1
```

They contain 127,777 unique rows, no nulls or duplicate candidate identities,
and use the canonical `simple_policy_optimiser.simulate_and_score` replay,
policy SHA-256
`aed39b3474f06a2134ed814bccaf41e0a3fd54bd8194108dfa251f6abcdce301`,
the deployed 1% round-trip fee, and the signed spread baseline. Gross minus fee
equals net on every row; spread drag is already embedded in executable gross
return. All rows use the policy's documented side-parent geometry fallback
because the observable rank-decile archetypes in the current 31/8 handoff do
not match the deployed local-geometry taxonomy. This is recorded row by row
and is not silently represented as local geometry.

The repaired joined handoff is:

```text
data_perp/artifacts/execution_ev_joined_handoff_policy_labels_20260725_v2
```

The production OOF run is:

```text
data_perp/reports/execution_ev_meta_policy1m_20260725_v1
```

It has 115,121 identical model OOF rows. Its standard leaderboard is explicitly
pre-admission diagnostic only. The required causal post-processing report is:

```text
data_perp/reports/execution_ev_meta_policy1m_20260725_v1/post_admission_21d
```

The post-processing contract is now verified:

- daily UTC snapshots;
- only OOF outcomes resolved before the snapshot;
- prior 21 causal days;
- symmetric 10% trimming of daily realized-minus-mapped EV residuals;
- side x OOF-CatBoost-predicted-archetype correction with side/global fallback;
- one globally pooled top 10% across the 115,121 shared OOF rows;
- ranking stage
  `after_causal_21d_admission_calibrator`;
- fixed corrected-EV admission diagnostic at `+0.007`.

No arm passes. All 67 post-calibrator global top-10 means are negative, and all
67 fixed-threshold admitted subsets have negative realized mean EV. The best
admitted mean is still `-0.0010765` on only 253 rows.

The best global top-10 arm is residual all-five auxiliary features **without**
`time_to_first_meaningful_mfe`:

```text
gross EV/trade  +0.005010
net EV/trade    -0.005015
selected rows   11,513
long net EV     -0.006060
short net EV    -0.004675
May net EV      -0.007269
June net EV     -0.001658
partial July    -0.010660
```

The model finds positive gross opportunity, but it is approximately 50 bps per
trade short of covering the deployed fee and spread contract. It therefore
cannot be integrated into `simple_policy_optimiser` admission. Retain the
existing policy/admission stream unchanged and do not start entry timing from
this execution head.

Individual auxiliary-head economics after the same 21-day correction, measured
as global top-10 net-EV change versus the matching alpha-context arm:

| Head | Direct add-one delta | Residual add-one delta | Residual all-five leave-one finding |
|---|---:|---:|---|
| Peak MFE | `-9.19 bps` | `-20.33 bps` | Removing peak improves by `+1.31 bps` |
| Time to meaningful MFE | `-25.69 bps` | `-15.93 bps` | Removing timing improves by `+15.78 bps`; largest harmful interaction |
| MAE before meaningful MFE | `-4.30 bps` | `-23.01 bps` | Removing MAE improves by `+0.95 bps` |
| Bars before adverse trough | `-15.03 bps` | `-22.60 bps` | Removing the clock improves by `+0.25 bps` |
| Favorable slope | `-9.62 bps` | `-6.15 bps` | Removing slope improves by `+2.31 bps` |

Direct all-five interactions are inconsistent: each leave-one removal makes
that already inferior all-five arm worse, while every direct add-one is
negative and the full direct bundle is also negative. Treat this as unstable
feature interaction, not evidence that any head is independently valuable.
The residual evidence is more coherent: every add-one is negative, every
leave-one improves the bundle, and timing is the largest detractor.

Consequences:

1. Reject the current execution-EV head and all five current auxiliary
   representations for production admission.
2. Keep the exact labels and repaired handoff as the canonical challenger
   benchmark.
3. Run the five head-specific target improvements listed in the individual
   assessment one at a time. Timing must use a jointly constrained hazard/CDF;
   MAE and adverse-trough heads must become competing-risk/event-probability
   tasks; the peak q80 ceiling must be repaired; slope must be residualized.
4. Require each improved head to beat alpha-context on this identical
   post-calibrator global top-10 contract before testing interactions.
5. Entry timing remains blocked because there is no positive stable
   execution-EV winner to optimize.

#### 2026-07-25 timing-head discrete-hazard challenger

The first one-head-at-a-time improvement is complete. The incumbent timing
family already used side-local feature selection, one shared side-local HPO
study over 2h/4h/8h/12h, strict May-July OOF folds, and post-prediction
isotonic projection. Its remaining architectural weakness was that four
independent binary classifiers supplied the CDF.

The challenger replaces them with one pooled at-risk discrete-time hazard
model per side. Each candidate contributes interval rows only while at risk;
the model predicts conditional event hazards and constructs
`CDF(t) = 1 - cumulative_product(1 - hazard)`. Monotonicity is therefore
structural, not repaired from OOF labels. It reuses the frozen side/horizon
feature-selection contracts, takes their side-local union, adds deterministic
interval indicators, runs a new 12-trial pooled-hazard HPO independently per
side, and retains the same causal May-July outer folds.

Artifacts:

```text
data_perp/artifacts/path_auxiliary_timing_hazard_challenger_20260725_v1
data_perp/artifacts/execution_ev_joined_handoff_timing_hazard_20260725_v1
data_perp/reports/execution_ev_meta_timing_hazard_policy1m_20260725_v1
data_perp/reports/execution_ev_meta_timing_hazard_policy1m_20260725_v1/post_admission_21d
```

Coverage is exactly the incumbent's 195,931 timing OOF rows, with 179,520
valid target rows. Relative to the incumbent on identical target rows, the
hazard challenger improves 12h log loss by `0.003520`, Brier by `0.001707`,
AUC by `0.00507`, and ECE by `0.01030`. It weakens log loss at 2h by
`0.005657`, 4h by `0.003066`, and 8h by `0.002977`; early-horizon
discrimination therefore remains the next timing-specific modeling problem.

On the identical exact-policy, post-21-day global ranking contract, its
within-run incremental economics are:

| Timing challenger comparison | Global top-10 net-EV delta |
|---|---:|
| Direct add-one versus matching direct alpha-context | `-13.31 bps` |
| Residual add-one versus matching residual alpha-context | `+17.53 bps` |
| Residual all-five minus residual all-five without timing | `+0.24 bps` |

This is a material improvement over the incumbent's residual add-one
`-15.93 bps` and residual interaction `-15.78 bps`. However, it is not a
production winner: the best complete challenger-run arm remains negative at
`-0.003798` net EV per globally selected top-10 trade, and every fixed `+0.007`
admitted subset remains negative.

Absolute cross-run changes in arms that exclude timing also moved materially
because the full runner retuned every arm. Do not attribute those absolute
changes to the timing representation without a current-code incumbent control
rerun. The paired within-run add-one and leave-one comparisons above are the
valid attribution evidence.

Decision:

1. Retain the discrete-hazard timing model as the research incumbent.
2. Do not promote it into policy admission or start entry timing.
3. Improve its early intervals next, preferably using policy-relevant
   1h/6h/12h/24h hazards while each exit remains active and training-only
   calibration.
4. Proceed to the next one-head challenger: MAE competing risks and actual
   stop/0.5R/1R crossing probabilities, with direct and residual add-one tests
   before any interaction search.

#### 2026-07-25 MAE competing-risk challenger

The second one-head challenger is complete. It replaces the weak expected-MAE
mixture with side-local probabilities tied to the deployed stop geometry:

- meaningful MFE before 0.5R adverse movement;
- 0.5R adverse movement before meaningful MFE;
- neither event before the path horizon;
- stop at 1R before meaningful MFE.

The first three outcomes share one multiclass model and sum to one. Stop
severity is modeled conditionally and exported as
`P(adverse 0.5R first) * P(stop 1R | adverse 0.5R first)`, so
`P(stop 1R) <= P(adverse 0.5R first)` by construction. Long and short use
their deployed side-parent stops (`4.0 ATR` and `3.525840973 ATR`) because the
current observable archetypes all follow the signed side-parent fallback.
Feature contracts are the side-local union of the incumbent event, MAE-if-hit,
and MAE-if-no-hit selectors; multiclass and severity HPO are independent by
side. OOF folds remain the strict May-July calendar.

Artifacts:

```text
data_perp/artifacts/path_auxiliary_mae_competing_risk_20260725_v1
data_perp/artifacts/execution_ev_joined_handoff_mae_competing_risk_20260725_v1
data_perp/reports/execution_ev_meta_mae_competing_risk_policy1m_20260725_v1
data_perp/reports/execution_ev_meta_mae_competing_risk_policy1m_20260725_v1/post_admission_21d
```

Coverage is 195,931 OOF prediction rows and 179,520 valid target-metric rows.
The probability constraints have zero violations. Predictive quality is
uneven:

| Metric | Long | Short |
|---|---:|---:|
| Three-outcome macro OVR AUC | `0.5382` | `0.5943` |
| 0.5R-adverse-first AUC | `0.4968` | `0.5557` |
| 1R-stop-before-MFE AUC | `0.5594` | `0.5678` |

Aggregate multiclass log loss is `1.04333` versus `1.05419` for the constant
class-prior reference. Most of that improvement comes from short. Long
0.5R-adverse discrimination is effectively random overall and falls to
`0.4767` in June, so the head does not meet the learnability/stability gate.

On the identical exact-policy, post-21-day, globally pooled top-10 contract:

| MAE representation | Direct add-one | Residual add-one | Direct all-five interaction | Residual all-five interaction |
|---|---:|---:|---:|---:|
| Incumbent expected-MAE mixture | `+1.63 bps` | `+22.66 bps` | `-3.58 bps` | `+0.98 bps` |
| Competing-risk challenger | `-5.20 bps` | `+5.49 bps` | `-4.34 bps` | `-0.69 bps` |

The matching alpha-context and all-five-without-MAE control arms are identical
across these two runs, so this is valid paired attribution rather than an
absolute cross-run comparison. The challenger is worse in both add-one modes
and both interaction modes. Its best complete-run global top-10 arm is still
negative at `-0.003798`, and every fixed `+0.007` admitted subset has negative
realized mean EV.

Decision:

1. Reject this challenger as an execution-EV input and retain it only as
   research evidence.
2. Keep the structurally coherent competing-risk target geometry.
3. Next test policy-anchored soft-binary transition bands around 0.5R and 1R,
   with the softness fitted inside each side-local training fold.
4. Test the resulting MAE probabilities in the entry-timing meta head only,
   as required by routing arm 3 above. This research ablation is allowed even
   while production entry-timing promotion remains blocked.
5. Require material improvement in long-side and June adverse-risk
   discrimination before repeating an execution-EV interaction search.

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

The evaluation must keep the three competing effects separate rather than
hiding them in one aggregate timing score:

1. **Better-price benefit:** conditional executable net-EV improvement after a
   fill, including the changed entry price and remaining path.
2. **Adverse-movement risk:** probability and severity of adverse-first
   movement after entry, plus post-entry MAE and stop-proximity diagnostics.
3. **Lost-opportunity risk:** probability of no fill or expiry while enter-now
   EV was positive, missed profitable trades, and regret versus entering now.

Report all three by side, action, outer fold, week, month, liquidity regime,
and volatility regime. A wait action is eligible only when its conservative
OOF utility remains above enter-now after costs and uncertainty while
missed-opportunity and adverse-risk limits both pass.

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

For every accepted wait action, persist and replay:

- decision price, decision-time ATR, selected `K`, raw target price, rounded
  suggested limit, maximum wait, expiry timestamp, and fallback action;
- calibrated fill probability, conditional better-price EV, adverse-first
  probability/severity, missed-opportunity penalty, all cost components, and
  final conservative action utility;
- a reason-coded fallback to enter-now or skip when the suggested price becomes
  stale, unfillable, uneconomic, crossed, or unsupported by current liquidity.

Search `K`, maximum-wait, and fallback rules per side using inner training data
only. The outer OOF row receives the frozen action and deterministic suggested
price; it must never help choose its own price offset or expiry.

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
Path/auxiliary research: five side-local OOF/final bundles complete; individual
head promotion remains gated; the discrete-hazard timing challenger is retained
for research but rejected for production economics
Execution-EV model: exact-policy OOF and causal post-21d global top-10 complete;
all current and timing-hazard challenger arms rejected because every best
net top-10 and admitted subset remains negative
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
| CatBoost archetypes | Yes, OOF/final models are side-local; economic admission blocked | Retain as context/risk inputs and re-test incremental value against repaired exit labels |
| Five auxiliary heads | Yes for side-local OOF/final research artifacts; only peak mean and timing CDF are currently promotion-eligible inputs | Run the expanded five-head research ablation matrix against repaired exit labels; retain economic gates per head |
| Execution-EV head | Side-local FS/HPO/calibration implemented; current fitted result is diagnostic only | Rematerialize the exact deployed-policy target, rerun OOF, and require the signed exit-policy contract |
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

## 12. Fixed 4m -> 6m Base/Residual Label Ablation (2026-07-25)

The dedicated research runner is now implemented in
`scripts/run_base_residual_label_ablation.py`, with chronology, label recipes,
ranking, and economic evaluation owned by
`extreme_price_movements/base_residual_label_ablation.py`. The verified result
is:

`data_perp/artifacts/base_residual_label_ablation_20260725_v2`

The fixed UTC calendar is:

```text
base fit:       2025-09-01 through 2025-12-30 22:59:59
base purge:     25 hours before 2026-01-01
base OOS:       2026-01-01 through 2026-06-30
meta fit:       first three base-OOS months, 2026-01 through 2026-03
meta final OOS: final three base-OOS months, 2026-04 through 2026-06
```

The base model is fitted once per recipe and side, then frozen for all six OOS
months. Residual label promotion uses only February-March residual OOF. The
21-day admission calibrator is fitted only on the first 21 days of that
residual OOF stream. No April-June row selects a label recipe, threshold,
calibrator, feature, parameter, or side mapping.

All arms use the same paired rows, the same 31-long/8-short approved feature
contracts, the same top-40 residual handoff, and the same realized
`__first_touch_capture_net__` economic field. That field already embeds the
full 1% round-trip cost; the ablation does not subtract cost again. The current
31/8 feature reuse is the previously approved feature-selection exception and
is recorded explicitly in the artifact. It is not a claim that the feature
selection itself was redone inside the four-month window.

The label matrix contains:

- exact incumbent 24-hour soft-label replay;
- a 12-hour timeout/path-quality soft label;
- a time-aware 12-hour label combining cost-aware meaningful MFE, pre-MFE MAE,
  time to meaningful/80%-peak MFE, an early two-hour path proxy, and 12-hour
  slope;
- nine deterministic per-side label-HPO recipes over the MFE, MAE, timing,
  early-path, slope, threshold, and softening weights. Side/sign is never an
  HPO dimension.

The full-population auxiliary store does not contain exact 15-minute closes for
the next two or three bars. Therefore the implemented early-path component is
honestly named `early_path_2h_proxy`: it uses early MFE and adverse-trough
targets. It must not be represented as the requested exact next-2/3-bar
non-flat/non-adverse label. Materializing that exact keyed target from canonical
15-minute bars remains a follow-up task.

Only about 51% of the canonical calendar rows have complete paired 12-hour path
targets after all required fields are enforced (long 51.53%, short 51.26%).
Every arm uses that same paired subset, so arm comparisons are internally fair,
but the result does not yet prove full-universe economics.

### Final untouched April-June results

All returns below are already net of the embedded 1% cost:

| Arm | Raw global top-10 | Raw timestamp x side top-10 | Post-21d admitted global top-10 | Post-21d admitted timestamp x side top-10 |
|---|---:|---:|---:|---:|
| incumbent 24h | -28.87 bps | -28.72 bps | +17.48 bps | +10.07 bps |
| 12h timeout | -26.68 bps | -29.77 bps | +23.28 bps | +13.01 bps |
| time-aware 12h | -25.14 bps | -29.70 bps | +24.22 bps | +8.19 bps |
| side-local HPO winner | -29.19 bps | -29.89 bps | +35.73 bps | +18.83 bps |

The training-only label HPO selected `hpo_01` for long and retained the
incumbent 24-hour label for short. The long winning weights are approximately
51.6% 12-hour execution, 23.6% MFE, 15.6% clean MAE, 3.6% timing, 1.6% early
path, and 4.1% slope, with threshold 0.42 and temperature 0.12.

The admitted side-local winner keeps 3,457 of 149,487 final top-40 residual
rows. Its global admitted top-10 contains 346 rows. Although global admitted
top-10 is positive in April, May, and June, the timestamp x side admitted
top-10 falls to -5.57 bps in June. Raw top-10 is negative under both ranking
scopes in every final month. This instability and the small admitted subset
block promotion.

The requested “remove assets currently excluded at inference” arm has zero row
impact: none of the paired training/evaluation rows belongs to the current
139-symbol spread blacklist. This confirms existing static universe parity for
this dataset, not causal historical parity. The current blacklist is derived
from June-July 2026 spread observations and is therefore future-derived for the
September-March training period. It remains a diagnostic-only check. A valid
historical ablation requires a point-in-time row-level average-spread
eligibility sidecar with strictly prior observations; current spread history
does not cover the 2025 training window.

### Decision and next actions

Do not promote any label arm into the production base/residual chain yet.

1. Materialize exact next-2/3 15-minute-bar favorable/adverse/flat targets and a
   true 12-hour executable timeout utility on the full candidate universe.
2. Backfill or regenerate complete 12-hour target support so paired coverage is
   high enough for a full-universe claim.
3. Add point-in-time average-spread history before retrying the training
   exclusion arm; do not backfill historical rows from the July `latest` file.
4. Repeat the fixed 4m -> 6m matrix with those targets, keeping April-June
   untouched.
5. Require positive raw top-10 or a materially broader, stable admitted set,
   including positive worst-month timestamp x side top-10, before promotion.

Current status:

```text
Fixed-window ablation infrastructure: implemented and tested
Label HPO: completed per side on residual OOF
12h/time-aware challengers: evaluated; not promoted
21-day admission: aggregate uplift, but too narrow and June-unstable
Inference-spread training exclusion: no-op on current paired rows; causal arm blocked by missing history
Production base/residual label: unchanged
```

## 13. Exact-Policy Replay and Head/Ablation Ledger (2026-07-25)

### Frozen winner extension and coverage

The side-local label-HPO selection remains:

- long: `hpo_01`, a soft 12-hour composite with normalized weights 51.57%
  executable 12-hour outcome, 23.59% peak MFE, 15.56% clean pre-MFE MAE,
  3.56% timing, 1.65% early-path proxy, and 4.08% future slope; soft threshold
  0.42 and temperature 0.12;
- short: `baseline_24h`, the incumbent 24-hour soft execution target.

`scripts/extend_label_hpo_winner_for_policy_replay.py` deterministically refits
the two missing frozen base boosters, loads the original residual boosters, EV
maps, and 21-day admission calibrators unchanged, and refuses July scores
unless April-June parity is exact. Both sides passed with maximum absolute
difference `0.0` for base prediction, residual score, and calibrated EV.

The requested replay endpoint was July 23, but a full July-23 claim is not
possible from the current local stores:

- canonical source labels stop on July 20 at 16:00 UTC;
- the frozen feature-complete paired universe used by this model stops on July
  17 at 12:00 UTC;
- after 21-day admission and timestamp-side top-10 ranking, the last
  replay-ready candidate is July 15 at 13:00 UTC.

The artifact therefore records requested and effective cutoffs separately:

`data_perp/artifacts/label_hpo_policy_replay_20260725_v1`

July results below are partial through that effective candidate cutoff, not
July-to-date through the 23rd. Backfill July 18-23 raw inputs, regenerate frozen
features and both label stores, then rerun the parity-gated extension to obtain
the requested complete window.

### Simple-policy optimizer and constrained portfolio replay

The simple-policy optimizer used April 1-30 only for HPO. May 1 through the
effective July cutoff was excluded from every Optuna trial and replayed only
after the policies were frozen. The canonical one-minute
`joint_trailing_total_mfe_raw_bayesian_v1` path, 96-bar/24-hour horizon, delayed
entry, side x archetype shrinkage, and exactly 1% round-trip cost were used.
Twelve trials were run per fitted policy. The candidate handoff already
contained only rows admitted by the causal 21-day calibrator and the
timestamp-side top 10% of that admitted set.

April itself did not support a profitable exit geometry. The parent-policy
diagnostics were negative for both sides, and the frozen May-July holdout
remained negative:

| Frozen holdout policy | Candidates with valid paths | Mean net return/trade |
|---|---:|---:|
| long parent | 402 | -1.393% |
| short parent | 819 | -1.344% |

The global portfolio auction then fitted only its hierarchical EV curve on
April candidates and replayed the frozen May-July policy candidates with the
current constraint shape: maximum two new entries per bar, one concurrent
position per symbol, 70% pre-leverage wallet cap, rank sizing, dynamic
thresholds, and the non-enforced 64-position emergency bound. This is a
constraint replay, not a claim that the old model's portfolio parameters were
retuned for the new label-HPO model.

| Period | Accepted trades | Mean net return/trade | Positive-trade rate |
|---|---:|---:|---:|
| May | 270 | -1.487% | 8.89% |
| June | 265 | -1.266% | 14.72% |
| July partial | 109 | -1.313% | 11.01% |
| Combined | 644 | -1.367% | — |

The constrained replay compounded to -93.69% from the normalized $10,000
wallet, with -1.419% notional-weighted net return, -93.69% maximum drawdown,
34.32% full-stop exits, and 56.21% timeouts. Constraints rejected 493 rows for
symbol cooldown, 74 because the symbol was already open, two at the per-bar
entry cap, and eight below the dynamic threshold. They did not rescue the
negative local policy economics. **The side-local label-HPO winner is therefore
not promotable into the execution policy.**

Primary replay outputs are:

- `simple_policy_optimizer/side_parent_policy_summary.csv`;
- `simple_policy_optimizer/holdout_side_parent_policy_metrics.csv`;
- `simple_policy_optimizer/holdout_side_archetype_policy_metrics.csv`;
- `portfolio_replay/summary.json`;
- `portfolio_replay/portfolio_decisions.parquet`.

### Per-head rules, evidence, and verdicts

All predictive metrics in this table are side-local outer OOF. Economic deltas
are paired only within the stated execution-EV experiment. They must not be
compared as absolute returns with the exact one-minute policy replay above:
the current execution-EV research label used one-hour candles, a 12-hour
side-parent geometry, and a different friction contract.

| Head | Rule / representation tried | Predictive metrics | Paired economic evidence | Verdict |
|---|---|---|---|---|
| CatBoost path archetype, long | Per-side multiclass path-role probabilities; context/risk input only | 64,504 OOF rows; logloss 1.705; RPS 0.208; monthly prior-logloss gain 0.0033-0.0139; `dead_timeout` is 58.0% of argmax and three classes never win | Standalone fold-train-only top-20 net EV -0.858%; every outer month negative | Context-only; require strict exact-policy EV uplift |
| CatBoost path archetype, short | Same, separately selected/HPO'd per side | 69,272 OOF rows; logloss 1.665; RPS 0.196; prior-logloss gain 0.0276-0.0419 | Standalone top-20 net EV -0.793%; every month negative | Context-only |
| `peak_mfe_12h_atr` | Conditional mean plus quantile output, per side | Mean Spearman IC long 0.557 / short 0.535; long q80 coverage 98.9% and IC 0.070, so long q80 is invalid | No valid identical-row add-one execution-EV result | Keep conditional mean as OOF research input; recalibrate/retest q70-q80 and policy-level crossing probabilities |
| `time_to_first_meaningful_mfe` | 12-hour censored pooled discrete hazard; hit-by-horizon probabilities | AUC at 2h 0.634/0.634 and at 8h 0.554/0.604 long/short; versus incumbent logloss delta +0.00566/+0.00307/+0.00298 at 2/4/8h and -0.00352 at 12h | Direct add-one -13.31 bps; residual add-one +17.53 bps; all-five interaction +0.24 bps; best global top-10 still -0.380% and every fixed-threshold admitted arm negative | Research timing representation only |
| `mae_before_meaningful_mfe_atr` | Competing-risk first-event probabilities; per-side conditional depth | If-hit IC 0.277/0.119; no-hit IC 0.080/0.133; macro AUC 0.538/0.594; long adverse-0.5R AUC 0.497 | Challenger direct/residual add-one -5.20/+5.49 bps versus incumbent +1.63/+22.66 bps; challenger all-five -4.34 bps | Reject challenger; test soft policy-anchored bands and timing-only routing |
| `bars_before_price_stops_decreasing` | Confirmed-trough and legacy clocks | Confirmed-trough IC 0.112/0.096; legacy clock IC 0.200/0.229 | No valid same-ID per-side execution-EV ablation | Blocked; reformulate as trough-before-stop/MFE/timeout hazard |
| `future_slope_atr_per_hour` | Per-side continuous 12-hour slope | IC 0.160/0.159 | No valid incremental add-one result | Diagnostic only; use robust multi-horizon efficiency and orthogonalize to peak/timing |

### Execution-EV and requested ablation ledger

| Ablation / routing rule | What was tried | Result | Status |
|---|---|---|---|
| Execution-EV direct alpha context | Base/residual alpha plus context, strict OOF | top-10 +0.190%; MAE 0.01941; RMSE 0.02751; Spearman 0.023 | Best research comparator only |
| Execution-EV direct all features | Add CatBoost and auxiliaries | top-10 +0.131%; MAE 0.01917; RMSE 0.02732; Spearman 0.004; -5.83 bps versus direct alpha context | Did not work |
| Alpha plus all auxiliaries | Direct and residual routes | direct top-10 -0.207% (-39.71 bps versus direct context); residual -0.073% | Did not work |
| Alpha plus CatBoost | Direct and residual routes | direct top-10 -0.216% (-40.54 bps); residual +0.081%, below useful comparator | Did not work |
| Remove one execution-EV feature family | Leave-one-family-out direct/residual matrix | several residual arms stayed slightly positive, but none repaired the invalid exit target or beat the valid exact-policy gate | Diagnostic only |
| 24h versus 12h base label | Same paired rows and frozen chronology | admitted global top-10: +17.48 bps for 24h, +23.28 bps for 12h; raw top-10 remains negative | 12h improved admission economics but not promotion-stable |
| Time-aware 12h base label | Add MFE, MAE, timing, early-path proxy, slope | admitted global top-10 +24.22 bps; timestamp-side +8.19 bps | Improved aggregate, not selected |
| Side-local soft-label HPO | Select recipe on Feb-Mar residual OOF only | final admitted global top-10 +35.73 bps and timestamp-side +18.83 bps; raw top-10 -29.19/-29.89 bps; June timestamp-side -5.57 bps | Selected research winner; exact-policy replay failed |
| Remove currently spread-excluded assets | Static current blacklist | removed zero paired rows; blacklist is not point-in-time for training | No-op; causal arm not yet tested |
| Soft-binary labels for auxiliary heads | Five side-local policy-anchored soft-label recalibration heads, four-model inner HPO, expanding June/July OOF with a 25-hour purge | Complete matrix; only soft timing improved global top-10 versus alpha context (+2.40 bps), but remained negative at -66.27 bps | Completed; no head promoted |
| Route timing and MAE only to timing meta head | Paired exact-policy enter/skip router: alpha only, original timing+MAE, and soft timing+MAE; timing and MAE excluded from the execution-EV non-timing arm | Soft exclusive router was -148.08/-115.28 bps global/timestamp-side top-10, -54.19 bps global versus the original router | Completed; did not work |
| Soft-binary execution-EV meta head | Exact one-minute policy net return transformed by threshold/temperature HPO; thresholds 0/25/50/75 bps, temperatures 30/50/100 bps, four tree geometries; June selection only | Best July alpha-context comparator -68.67/-113.21 bps global/timestamp-side top-10; all-aux -112.83/-113.60 bps | Completed; no promotable arm |

### Exact-policy soft-binary ablation matrix (2026-07-25)

The final paired artifact is
`data_perp/artifacts/exact_policy_soft_binary_ablations_20260725_v4`.
It uses all 127,777 exact candidate identities from the deployed-policy
one-minute replay. The source manifest reports 100% path coverage, a
1,440-minute horizon, spread-aware executable fills, and the strategy's 1%
fee contract. This supersedes the earlier 99,518/127,777 coverage audit:
there is no remaining targeted path-backfill blocker for this candidate set.

The five auxiliary challengers are **side-local OOF recalibration layers over
the existing frozen OOF head outputs**, not native raw-feature refits. June
predictions train only before June and July predictions only before July, both
with a 25-hour purge. Four LightGBM geometries are selected on the purged
trailing 20% of each outer-training set. This produces 64,426 complete
June-July OOF rows. The execution-EV soft-label recipe and model are selected
on June only; the 15,167 July rows are untouched final evaluation.
The available exact-policy July window is July 1-10, not July 1-23.

| Soft auxiliary head | Soft-label rule | July OOF Brier long/short | July OOF Spearman long/short | Exact-policy global top-10: original -> soft | Soft verdict |
|---|---|---:|---:|---:|---|
| `peak_mfe_12h_atr` | Probability that peak MFE clears the larger of 1.5 ATR and the 1% cost hurdle; 0.25 ATR transition | 0.23379 / 0.22851 | -0.052 / -0.005 | -69.83 -> -81.76 bps (-11.93) | Worse and non-discriminating |
| `time_to_first_meaningful_mfe` | Hit-within-12h probability weighted toward arrival before 4h; 1.5h transition | 0.10042 / 0.08351 | 0.111 / 0.166 | -81.41 -> -66.27 bps (+15.13); +2.40 bps versus alpha context | Best individual challenger, still negative |
| `mae_before_meaningful_mfe_atr` | Probability pre-MFE MAE stays below 0.5 ATR; 0.15 ATR transition | 0.17562 / 0.15861 | -0.134 / 0.083 | -113.51 -> -122.63 bps (-9.12) | Worse; keep out of execution EV |
| `bars_before_price_stops_decreasing` | Early confirmed adverse trough around four bars, multiplied by clean-MAE probability; missing trough censored at 12h | 0.02883 / 0.02240 | 0.012 / 0.058 | -76.71 -> -89.58 bps (-12.86) | Low ranking signal; did not work |
| `future_slope_atr_per_hour` | Probability favorable slope is positive; 0.15 ATR/hour transition | 0.02291 / 0.02339 | 0.111 / 0.106 | -132.12 -> -80.93 bps (+51.19) | Large relative repair, still below comparator |

Every economic number above is July exact-policy net return after the stated
cost contract. Global top-10 uses 1,517 rows. Timestamp-side top-10 is also
reported because opportunity admission is local to each decision timestamp
and side:

| Execution-EV / routing arm | Included head rule | Global top-10 | Timestamp-side top-10 | Result |
|---|---|---:|---:|---|
| Alpha context | Existing alpha EV, uncertainty, support, and observable archetype context | -68.67 bps | -113.21 bps | Comparator only |
| Non-timing auxiliaries | Peak, adverse-turn, and slope; timing and MAE excluded | -87.65 bps | -108.64 bps | +25.18 bps over all-aux globally, but below comparator |
| All auxiliaries | All five original auxiliary streams | -112.83 bps | -113.60 bps | -44.16 bps versus comparator |
| Timing router, alpha only | Enter/skip gate from alpha context | -137.16 bps | -121.53 bps | Did not work |
| Timing router, original | Alpha plus original timing and MAE | -93.89 bps | -127.16 bps | Did not work |
| Timing router, exclusive soft | Alpha plus only soft timing and soft MAE | -148.08 bps | -115.28 bps | -54.19 bps globally versus original; did not work |

Fixed score admission at 0.5 produced very small, unstable subsets (zero to
592 rows depending on arm), so isolated positive means such as the one-row
soft-peak comparator or three-row soft timing router are not promotion
evidence. No arm clears either global or timestamp-side top-10 economics.
This fixed-threshold diagnostic is not a new 21-day admission-calibrator fit;
the earlier 21-day-calibrated base-label results remain separately reported in
the ledger and must not be conflated with this July exact-policy matrix.

The timing experiment tests exact-policy **enter versus skip** only. It does
not yet generate or replay counterfactual wait-market, wait-limit, or target
price actions. Those require the separate action layer and fill/missed-
opportunity simulation specified above.

## 14. Repaired Event Heads and Canonical Global-Top-K EV Matrix (2026-07-25)

This section supersedes any interpretation above that treats timestamp-side
ranking as the trading policy. The production research contract is:

1. generate strict side-local outer-OOF scores;
2. apply the causal resolved-before-snapshot 21-day recent-EV correction;
3. pool rows across timestamps and sides;
4. trade the global top `k`.

Timestamp-side top-decile metrics remain diagnostics only.

### Dedicated literal meaningful-MFE classifier

The repaired base-to-residual CatBoost classifier for the literal question
“does the path reach meaningful MFE within 12 hours?” is healthy as a head:
literal AUC 0.5715, average precision 0.5503, Brier 0.2462, ECE 0.0328,
and top-decile precision 0.6026. It beats the incumbent on discrimination,
calibration, and top-decile precision. The probability is therefore retained
as an OOF input candidate, but it is not a standalone trading score.

### Exact 1m policy add-one results

The following results use 127,777 exact candidate identities, three purged
outer folds, side-local CatBoost fits, the canonical 21-day recent-EV
correction, and one global pooled top-10% ranking. Values are mean exact-policy
net return per selected trade after the signed 1% cost contract.

| EV arm | Direct | Residual | Incremental verdict |
|---|---:|---:|---|
| Frozen baseline, all features | -57.85 bps | -67.16 bps | Comparator |
| + literal reach probability | -68.58 bps | -66.55 bps | Direct worse; residual neutral (+0.61 bps) |
| + clean favorable-before-adverse probability | -57.01 bps | -57.13 bps | Useful, especially residual (+10.03 bps) |
| + competing favorable/adverse/timeout probabilities | -63.94 bps | -55.13 bps | Best residual add-one (+12.03 bps) |
| + conditional path quality | -59.46 bps | -66.65 bps | Neutral-to-negative |
| + probability x conditional magnitude economics | -69.98 bps | -63.76 bps | Spearman improves to about 0.07, but the global tail worsens |
| + clean and competing probabilities together | -68.59 bps | -64.22 bps | Interaction is destructive |

MDA does not convert the repaired heads into positive economics. The frozen
baseline direct-MDA arm remains best at -49.09 bps/trade. Clean-event MDA is
-51.45 bps and competing-risk MDA is -52.87 bps. Clean-event MDA does improve
rank correlation and top-k positive-trade rate, but not mean net EV.

A fixed diagnostic clean-probability gate improves the baseline to -46.32
bps/trade at global top-10%, still negative. Global `k` fractions from 0.1%
through 20% were also tested and none became positive. The best predicted
tails fail to average the 1% cost hurdle in gross return.

### Exit-policy and horizon audit

The exact label simulator is the intended
`joint_trailing_total_mfe_raw_bayesian_v1` one-minute simple-policy pathway.
Two contract issues are now explicit:

- despite the `*_12h` column names, the previously canonical labels use a
  signed 1,440-minute / 24-hour exit horizon;
- all current base-rank-decile archetypes fall back to side-parent geometry,
  because none matches the champion policy's older named contextual
  archetypes.

A genuine 720-minute timeout-only replay was materialized with identical
geometry and costs. It is worse: direct global top-10 falls from -57.85 to
-88.96 bps/trade, residual from -67.16 to -99.96 bps/trade, and timeout exits
rise from 15,818 to 38,998. Keep the 24-hour exit horizon for this policy
family; keep 12 hours only for the meaningful-MFE auxiliary target.

The next required policy ablation is a leakage-safe re-optimization of exit
geometry for the archetypes observable in the current stream (the base-rank
deciles), with side-parent fallback as the control. Do not promote any EV/head
architecture until that geometry is frozen on prior folds and the recent-EV
mapped global-top-`k` replay is positive after portfolio constraints.

### Architecture direction

Proceed with:

`base alpha -> residual alpha + per-side path context/heads -> execution EV`

where execution EV uses separate candidate channels rather than concatenating
every head:

- direct EV candidate: baseline plus clean-event probability;
- residual EV candidate: baseline plus competing-risk probabilities;
- literal reach probability: calibration/gating diagnostic unless it proves
  incremental EV;
- probability x magnitude: separate economics diagnostic or regularizer, not
  a raw add-one block;
- timing and MAE: timing/action layer only;
- optional wait/target-price action layer above the ML score, with explicit
  adverse-move, improved-price, missed-opportunity, fill, and cost labels.

Screen new feature families with the common all-features arm. Run MDA and HPO
only once an add-one improves canonical global-tail EV; this preserves the
paired question and shortens future runs materially.

## 15. Alpha-Candidate Context Repair (2026-07-25)

The previous negative matrix omitted causal geometry from the frozen alpha
candidate stream. The execution-EV handoff now includes strict OOF
`base_oof_score`, cutoff margin, margin z-score, timestamp-side score/rank
context, within-archetype score z-score, group size, and base rank decile.
These fields are model context only. They do **not** change admission into a
per-timestamp policy: every result below is ranked once, globally, after the
causal 21-day recent-EV mapper.

Fixed-parameter screening on the common 100-day / three-fold window has now
been decomposed exactly. Every row below uses the same 116,712 outer-OOF rows
and one pooled global top decile after the causal 21-day admission correction.
There is no per-timestamp selection quota.

| Candidate-context family, tested alone | Best target | Global top-10 net EV/trade |
|---|---|---:|
| Raw base OOF alpha score | Residual | +23.11 bps |
| Cutoff margin and cutoff-margin z-score | Direct | **+28.95 bps** |
| Timestamp-relative score/rank | Residual | +23.91 bps |
| Archetype-relative score z-score | Direct | +22.91 bps |
| Rank decile and candidate-group size | Residual | +15.42 bps |

The compact paired combinations are:

| Compact context | Best target | Global top-10 net EV/trade |
|---|---|---:|
| Raw score + cutoff margin/z | Residual | **+28.91 bps** |
| Cutoff margin/z + archetype z | Direct | +26.24 bps |
| Raw score + cutoff margin/z + archetype z | Residual | +23.35 bps |
| Raw score + cutoff margin/z + timestamp-relative context | Residual | +27.40 bps |

Therefore retain raw OOF alpha score plus cutoff margin/z as the compact
candidate context. Timestamp-relative rank is useful alone but is not
incremental to the compact core. Archetype z, rank decile, and group size are
also excluded from the winner.

Add-one tests inside that compact residual architecture give:

| Add-one block | Global top-10 net EV/trade | Verdict |
|---|---:|---|
| Clean favorable-event probability | **+30.55 bps** | Retain; fixed research winner |
| Competing favorable/adverse/timeout probabilities | +19.63 bps | Reject |
| Clean + competing probabilities | +16.74 bps | Reject interaction |
| DAE bottleneck | +16.59 bps | Reject |
| GMM distance/Mahalanobis geometry | +24.48 bps | Reject |

GMM posterior probabilities and compact representation risk summaries remain
excluded; they are not promoted merely because the representation exists.
Timing, MAE, target-price, fill, missed-opportunity, and wait outputs remain in
a separate action layer and are not execution-EV inputs.

The aggregate winner has positive-EV AUC 0.5417, Spearman 0.0569, and +30.55
bps mean exact-policy net EV across 11,672 pooled global-top-decile OOF rows.
Its incumbent 21-day recent-EV score is positive in all three outer folds:
+36.21, +32.40, and +6.03 bps. Causal robust-z normalization is rejected for
this winner because fold 2 becomes -0.54 bps.

Month-local global-top-decile diagnostics are +29.90 bps in May, +48.51 bps in
June, and -28.51 bps for July 1-10. More importantly, the one pooled global
auction selects only five July rows, so the apparent July tail is both negative
and inadequately covered. This concentration must not be repaired by
per-timestamp quotas.

Side-local post-screening HPO and MDA are complete. Residual-only MDA falls to
+3.72 bps and is rejected. The eight-trial residual-only HPO challenger reaches
+29.62 bps versus +25.37 bps for its paired incumbent, but loses latest-fold
stability after temporal normalization and is not promoted.

The required architecture is:

`base alpha -> residual alpha + per-side CatBoost archetype head + five per-side auxiliary heads (parallel) -> compact-context residual execution EV using clean favorable probability -> optional timing/MAE/target-price/wait action layer -> causal recent-EV mapping -> pooled global top-k -> portfolio constraints`

Peak-MFE magnitude and future slope remain useful auxiliary information, but
the evidence favors using them through separately validated risk/action
features or regularization rather than concatenating every head into the
execution-EV ranker.

### Narrowed context, mapper, and constrained-policy result

The frozen compact-context + clean-probability winner was replayed through the
existing portfolio policy on one immutable pooled global-top-decile book. The
baseline constraint set accepts 1,179 trades, returns +18.71 bps
notional-weighted per accepted trade, compounds the normalized wallet by
+60.86%, and has -18.15% maximum drawdown.

One-factor constraint ablations give:

| Constraint arm | Accepted | Net return/trade | Compounded | Max drawdown | Latest-fold net/trade |
|---|---:|---:|---:|---:|---:|
| Existing baseline | 1,179 | +18.71 bps | +60.86% | -18.15% | +18.64 bps |
| Maximum 8 concurrent | 651 | +29.24 bps | +61.19% | -14.07% | +2.83 bps |
| Maximum 40% wallet allocation | 1,161 | +19.13 bps | +34.45% | -12.12% | +11.22 bps |
| Maximum 2 per symbol | 1,345 | +11.46 bps | +41.58% | -23.48% | -20.47 bps |
| Maximum 4 per side | 433 | +19.68 bps | +25.03% | -12.91% | -42.18 bps |
| Maximum 1 new entry per bar | 759 | +16.22 bps | +36.68% | -19.09% | +15.05 bps |

No one-factor challenger is promoted. The concurrency cap improves aggregate
efficiency and drawdown but nearly removes latest-fold edge. The wallet cap is
a defensible risk-budget choice, not an alpha improvement. Relaxing per-symbol
limits and imposing the tested per-side cap both damage latest-fold economics.

The pooled book's calendar-July coverage remains the blocker: only five July
candidates enter the global top decile and only three survive the baseline
portfolio constraints; those three average -424.29 bps
notional-weighted. Latest outer-fold coverage is broader (85 accepted baseline
trades at +18.64 bps), so both latest-fold and latest-month coverage/economics
must be reported and must pass. Aggregate EV alone is no longer a promotion
criterion.

`execution_ev_model_ablation` now records, for every post-calibrator arm:

- latest-fold and latest-month local top-decile economics;
- the number and share of the pooled global top decile coming from each latest
  period;
- a stability gate requiring non-negative economics and minimum coverage in
  both periods.

An arm can remain a research winner while failing this promotion gate. Current
July source data stop at the July 20 16:00 UTC signal; a July-23 conclusion
still requires upstream data backfill.

### July-19 extension result (2026-07-26)

The extension is now large enough to replace the earlier 39-row diagnostic,
but it does **not** rescue the winner:

- base and residual OOF streams extend unchanged through July 19 15:00 UTC;
  every pre-extension prediction matches exactly;
- the clean/competing event study adds 11,528 rows with zero prediction drift
  on all 283,904 older rows;
- the clean hard ensemble remains preferable to the competing-risk
  probability. July long AUC is 0.5233 versus 0.5076 and short AUC is 0.5465
  versus 0.5394. Competing net probability is below chance in aggregate July;
- Peak MFE and path-CatBoost were frozen-scored separately, never relabelled
  as OOF. Their strict joint information boundary leaves 7,112 forward rows
  from July 11 22:00 through July 19 15:00;
- the frozen execution-EV winner uses Peak MFE, path-CatBoost
  probabilities/entropy, alpha/context and clean probability only. Timing,
  MAE, adverse-turn, target-price and wait features remain outside this EV
  model in the action layer.

The exact causal recent-EV correction reproduces all 116,712 historical OOF
scores with maximum absolute delta 0.0. Across the combined 123,824 historical
OOF plus frozen-forward rows, the pooled global top decile is still +26.58
bps. This aggregate is misleading: it selects **zero** of the 7,112 new
forward rows. The new forward period's own global top decile is -80.21 bps,
and the July 1-19 month-local top decile is -31.46 bps. This is a latest-month
coverage and economics failure, not a reason to introduce per-timestamp
quotas.

The immutable pooled book was replayed through the same constraints. Because
no new forward row enters global top-`k`, the July portfolio tail still
contains only three older July trades and remains negative:

| Constraint arm | Accepted | Net return/trade, notional weighted | Compounded | Max drawdown |
|---|---:|---:|---:|---:|
| Existing baseline | 1,249 | +17.32 bps | +57.99% | -20.80% |
| Maximum 8 concurrent | 682 | +29.35 bps | +66.33% | -14.27% |
| Maximum 40% wallet allocation | 1,219 | +16.76 bps | +30.92% | -12.24% |
| Maximum 2 per symbol | 1,412 | +12.65 bps | +49.01% | -23.68% |
| Maximum 4 per side | 445 | +22.22 bps | +29.32% | -11.52% |
| Maximum 1 new entry per bar | 809 | +15.31 bps | +36.30% | -18.02% |

No constraint arm is promoted. The concurrency cap improves aggregate
efficiency, but it cannot cure zero latest-forward admission; the latest
period must first pass the score/coverage gate.

## 16. Regime Learnability and Specialist-Model Workstream (IN PROGRESS)

The June/July behavior may reflect either score-scale drift, missing regime
features, a target mismatch, or genuinely irreconcilable conditional response.
Do not assume specialists are required until the following diagnosis is
complete.

### Executed fixed-winner diagnosis (2026-07-25)

The diagnosis runner now reproduces the fixed research-winner architecture:
CatBoost with the frozen no-HPO geometry, residual target over frozen alpha,
the exact compact context + clean probability inputs, and fixed CatBoost
archetype one-hots. Forward fits purge unresolved labels. Reversed fits are
explicitly non-OOS and non-promotable. All metrics use one global top decile
over the full evaluation month.

| Training direction / window | Evaluation | Training rows | Global top-10 net EV/trade | Positive rate | Spearman |
|---|---|---:|---:|---:|---:|
| May -> June | June | 62,738 | +19.64 bps | 74.90% | 0.1189 |
| Matched May subset -> June | June | 15,230 | +4.21 bps | 71.92% | 0.1059 |
| July future-trained diagnostic -> June | June | 15,230 | -19.65 bps | 62.25% | 0.0885 |
| June -> partial July | July 1-10 | 48,360 | -29.99 bps | 62.44% | 0.0214 |
| May+June -> partial July | July 1-10 | 111,663 | -33.87 bps | 63.49% | 0.0177 |
| May+June, 30-day half-life -> partial July | July 1-10 | 111,663 | -32.94 bps | 62.51% | 0.0277 |
| May+June, 14-day half-life -> partial July | July 1-10 | 111,663 | -29.74 bps | 63.30% | 0.0275 |

The matched swap does **not** support the simple claim that July is a later
regime whose samples learn June better: the future-trained July model is worse
than the equal-size May control on June. Longer history also degrades July, and
causal recency weighting recovers only 0.25 bps versus the one-month control.
Do not introduce specialists from this evidence alone.

The feature audit nevertheless identifies a material conditional shift from
June to partial July:

- `existing_alpha_ev` Spearman changes by -0.1921 and flips negative;
- `base_oof_score` changes by -0.1833 and flips negative;
- clean favorable probability changes by -0.1567 and flips negative;
- CatBoost archetype probabilities `p_2` and `p_5` shift by -2.03 and -1.56
  pooled standard deviations, respectively;
- alpha uncertainty, CatBoost `p_2`, `p_3`, `p_5`, slow-grinder probability,
  and cutoff margin also change economic sign.

This points first to a shared base/label/execution failure plus head-distribution
shift, not a cleanly learnable specialist regime. The next single-model
ablations should therefore add causal drift/trust composites: recent unlabeled
alpha-score distribution shift, alpha-versus-clean-head disagreement,
archetype-probability shift, uncertainty x margin, and clean probability x
liquidity/cost. Only if those fail on an extended July window should the
specialist gates below be activated.

### Expanded July-19 fixed-winner diagnosis (2026-07-26)

The same fixed residual-CatBoost architecture was rerun on the expanded
May-through-July-19 input:

| Training direction / window | Evaluation | Training rows | Global top-10 net EV/trade | Positive rate | Spearman |
|---|---|---:|---:|---:|---:|
| May -> June | June | 62,738 | +7.30 bps | 72.04% | 0.0865 |
| Matched May subset -> June | June | 22,342 | +8.61 bps | 72.45% | 0.0951 |
| July future-trained diagnostic -> June | June | 22,342 | -5.32 bps | 64.26% | 0.0649 |
| June -> July 1-19 | July | 48,360 | **-36.63 bps** | 56.69% | 0.1405 |

The larger reverse-month test still rejects the simple learnable-regime
story: July-trained data perform worse on June than the equal-size causal May
control. July rank correlation is not absent, but its cost-adjusted top tail
is negative, so rank IC alone is not sufficient.

June-to-July drift remains concentrated in economically relevant inputs.
Clean favorable probability changes target correlation by -0.168 and flips
sign; existing alpha and base OOF score change by -0.136 and -0.134 and flip
sign. CatBoost `p_5` shifts -1.11 pooled standard deviations and flips
economic sign; Peak MFE shifts +0.81 standard deviations while losing 0.067
target correlation. The diagnosis therefore continues to favor causal
trust/drift features and label/execution repair before regime specialists.

### Within-July learnability and leaf transfer (2026-07-26)

**Superseded as economic evidence by the exact-policy lineage repair below.**
This run trained on the hourly approximation through July 10 and the exact
one-minute replay from July 11 onward. Its chronology was purged correctly,
but its target definition changed inside the month. Retain the results below
only as historical diagnostics; do not use them to justify a regime model.

July is learnable cross-sectionally but is not yet economically learnable in
strict forward time. Two purged expanding July folds produce 12,483 forward
predictions:

| Evidence | Global top-10 net EV/trade | AUC | Spearman | Status |
|---|---:|---:|---:|---|
| Strict earlier-July -> later-July | **-17.31 bps** | 0.6336 | 0.2065 | Valid forward OOS |
| Random July cross-fit | +36.69 bps | 0.7249 | 0.3420 | Diagnostic; chronology broken |
| July in-sample | +56.30 bps | 0.7497 | 0.3901 | Diagnostic; not OOS |

Both valid folds are negative: -17.41 bps for the July 8 boundary and -66.79
bps for the July 15 boundary. The forward model improves the unconditional
July population by +49.32 bps, but its selected tail is still negative. The
day-block 95% interval for forward top-10 EV is -77.46 to +15.60 bps, and the
improvement-versus-frozen-alpha interval also crosses zero. Long is nearly
breakeven at -9.30 bps; short is decisively worse at -65.89 bps.

A narrower leaf-transfer test explains why the non-chronological diagnostics
look attractive. A July 1-7 model reaches +5.28 bps on July 8-10 versus
-18.65 bps unconditional, but falls to -96.73 bps on July 11-19. Mean
per-tree leaf-distribution JS divergence rises from 0.089 to 0.207 long and
from 0.083 to 0.203 short. Unseen-leaf fractions remain only 3.0% and 4.2%,
so the failure is not mainly unseen geometry: familiar leaves change
frequency and economic meaning. Every late-July leaf-signature cluster is
negative. Leaf support is inconsistent across sides and is not a safe hard
gate.

Therefore:

- do not promote a July-only expert or leaf cluster as alpha;
- retain leaf occupancy, JS drift, unseen/low-support fraction and expert
  disagreement as candidate **trust/abstention** features;
- estimate these features from a frozen reference model and trailing
  unlabeled feature/leaf distributions so they are available causally;
- require repeated forward windows before any leaf cluster receives an
  economic prior.

### Exact-policy lineage repair and corrected within-July test (2026-07-27)

The target-provenance audit found that the preceding regime input joined two
different outcome contracts:

- the superseded hourly 12-hour approximation for the historical 127,777
  rows; and
- the exact one-minute deployed-policy replay for the 7,112 forward rows.

This is not a harmless accounting difference. On the identical 127,777 rows,
replacing the hourly approximation with the exact replay changes every target,
reduces net EV by 85.79 bps on average, and has a maximum absolute difference
of 5,721.71 bps. The old label source had already been marked non-promotable in
the exit-policy audit, but later regime experiments accidentally retained it.

`scripts/build_canonical_exact_policy_regime_input.py` now removes every
realized outcome from the feature frame before an exact one-to-one join to the
single canonical one-minute policy ledger. Gross minus cost equals net with
zero reconciliation error. All 134,889 feature rows receive the canonical
target, and all receive a causal decision-time ATR from the July-20 path
source.

The fixed within-July CatBoost experiment was then rerun unchanged: same
features, geometry, weekly boundaries, 12-hour decision purge, label-resolution
gate, global top decile and day-block uncertainty.

| Evidence | Global top-10 net EV/trade | AUC | Spearman | Status |
|---|---:|---:|---:|---|
| Strict earlier-July -> later-July | **-107.55 bps** | 0.5527 | 0.0712 | Valid forward OOS |
| Random July cross-fit | -35.09 bps | 0.6873 | 0.2706 | Diagnostic; chronology broken |
| July in-sample | -3.07 bps | 0.7278 | 0.3548 | Diagnostic; not OOS |

Both valid folds are negative under the one target contract: -103.80 bps for
the July 8 boundary and -81.12 bps for the July 15 boundary. The pooled
day-block 95% interval is -151.30 to -69.29 bps. Long is -108.12 bps and short
is -112.03 bps. The exact replay therefore strengthens the rejection of a
July specialist: the weak forward rank signal does not approach the cost
hurdle.

The opportunity/capture decomposition uses the same 12,483 valid forward rows,
one pooled global top 10%, and reconciles:

```text
net = path MFE - (path MFE - exact-policy gross) - exact cost
```

| Selection | Path MFE | Gross | Cost | Net | MFE-to-gross gap | Favorable/adverse first |
|---|---:|---:|---:|---:|---:|---:|
| Within-July model | 200.84 bps | -7.59 bps | 99.96 bps | -107.55 bps | 208.43 bps | 39.6% / 48.3% |
| Frozen alpha | 151.18 bps | +5.21 bps | 100.03 bps | -94.82 bps | 145.97 bps | 38.8% / 39.8% |

Relative to frozen alpha, the model selects +49.66 bps more path MFE but adds
62.46 bps of MFE-to-gross gap; cost is unchanged by -0.06 bps. The exact
identity is therefore -12.74 = +49.66 - 62.46 - (-0.06) bps. It finds more
volatile opportunity, but adverse-first frequency, reversal and exit-policy
capture deteriorate faster than opportunity improves. The added 1,199 rows
average -109.97 bps versus -96.71 bps for the 1,199 dropped rows.

This result changes the next target design:

- predict executable gross capture or MFE-to-gross gap jointly with
  opportunity, not meaningful MFE alone;
- keep favorable-first and adverse-first as competing risks;
- train a severe-loss/capture-failure head and make the final target explicitly
  clear the approximately 100 bp realized cost;
- treat whole-horizon MFE/MAE and their ratios as hindsight diagnostics, not
  executable exit counterfactuals;
- do not alter the exit policy from this analysis. Any capture-policy change
  requires a separately frozen simple-policy optimization and forward replay.

A matching corrected barrier grid now covers all 134,889 rows and all four
12h/24h x 1.5/2.0-ATR cells with exact contiguous hourly paths. It copies the
canonical exact-policy gross/net values and reconciles them exactly; barrier
events remain supporting labels and do not replay execution.

Canonical artifacts:

- `data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/`
- `data_perp/artifacts/execution_ev_exact_policy_within_july_learnability_20260727_v1/`
- `data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/`
- `data_perp/artifacts/within_july_exact_policy_opportunity_capture_diagnosis_20260727_v2/`

### Mixed-period remedies tried

**Economic metrics in this subsection are superseded.** These remedies were
trained/evaluated on the hourly approximation or the mixed historical/forward
target. Their relative behavior is diagnostic only. Any surviving remedy must
be rerun on the canonical exact-policy input before it can inform architecture.

The remedies have now been rerun unchanged on the single exact one-minute
policy ledger. Primary metrics below are after the causal recent-EV mapping
and one pooled global top decile:

| Exact-policy remedy | May -> June | Later July |
|---|---:|---:|
| Uniform available history | -63.04 bps | -105.66 bps |
| May+June only | n/a | -140.16 bps |
| Early July weighted 3x | -63.04 bps | -105.10 bps |
| 14-day recency half-life | -83.36 bps | -105.71 bps |
| Calendar x archetype balancing | -77.11 bps | -110.01 bps |
| Causal trust composites | -86.98 bps | -103.31 bps |
| Global + 0.5 recent residual correction | -63.33 bps | **-85.52 bps** |

No arm clears costs even in the May-to-June retention control. The recent
residual remains the least-bad later-July remedy and preserves more rank signal
(AUC 0.588, Spearman 0.116), but its -85.52 bps book is not close to
tradable. May+June-only training is worst in later July, confirming some
non-transfer, but the more important corrected diagnosis is broader: the
current model/target architecture fails under the deployed exit economics in
both regimes. Do not spend the next search budget on regime routing.

Corrected artifact:
`data_perp/artifacts/execution_ev_exact_policy_mixed_period_remedies_20260727_v1/`.

Seven fixed remedies were tested on a strict later-July window while retaining
a May-to-June control:

| Remedy | Later-July top-10 before recent mapping | After mapping | May->June after mapping |
|---|---:|---:|---:|
| Uniform available history | -101.19 bps | -85.47 bps | +11.14 bps |
| May+June only | -119.68 bps | -86.88 bps | n/a |
| Early July weighted 3x | -99.36 bps | -93.48 bps | +11.14 bps |
| 14-day recency half-life | -103.02 bps | -90.39 bps | **+31.40 bps** |
| Calendar x archetype balancing | -96.15 bps | -110.16 bps | -10.12 bps |
| Causal trust composites | -100.50 bps | -83.07 bps | -3.51 bps |
| Global + 0.5 recent residual correction | **-80.18 bps** | **-81.19 bps** | +17.14 bps |

No remedy makes later July tradable. The shrunk recent residual correction is
the most promising direction because it gains about 21 bps before mapping
while retaining positive May-to-June economics. Its final temporal
correction-OOF mapper improves July by another 2.36 bps and June by 0.72 bps
versus the initial mapping, but -81.19 bps remains deeply negative. Simple
recency, month/archetype balancing, and early-July oversampling do not solve
the problem.

The next implementation order is:

1. build a causal trust head targeting whether the global model adds positive
   cost-adjusted utility versus abstaining, using leaf-distribution drift,
   expert disagreement, recent score/feature distribution shifts, liquidity
   and cost;
2. train a global model plus a strongly shrunk rolling residual adapter.
   Choose shrinkage from training-only effective sample size and drift
   confidence; include an explicit zero-correction fallback;
3. define causal regimes precisely and add their stable geometry as residual/
   trust inputs plus transition and persistence as supporting labels. Do not
   use calendar or regime-similarity sample weights;
4. add labels for executable positive net EV, severe-loss probability,
   opportunity prevalence, path efficiency and signal/expert utility delta;
5. validate abstention and adapter arms on multiple future weekly blocks with
   one global mapped top-`k`, latest-week coverage, exact costs and portfolio
   constraints;
6. introduce softly gated specialists only if recurring leaf/feature regimes
   show stable, different economic mappings across several forward episodes.

### Causal-regime, trust and adapter ablations (2026-07-26)

The implemented regime contract is deliberately a market/candidate **state**,
not a profitable-regime label. It is fitted independently by side on prior
decision-time alpha, base-score, Peak-MFE and CatBoost geometry. K is selected
without outcomes from assignment stability and minimum occupancy. Outcomes,
calendar labels and regime weights are prohibited. Safe reusable inputs are
permutation-invariant state entropy, top-two margin, nearest-centroid distance
and frozen-training empirical distance percentile/exceedance. Numeric state
IDs are diagnostic only. Posterior coordinates are fold-local and may only be
used inside one frozen state fit because K and centroid ordering can change.

The supporting labels are `causal_regime_change_within_6h` and its complement
`causal_regime_persistence_6h`, with an explicit `decision + 6h` resolution
timestamp. They are never same-row inputs. The weekly generator scores a
post-week six-hour buffer with the frozen state model so end-of-week labels
are either correctly resolved or null.

The late-July environment is not mainly a new mixture of coarse states:

| Forward week / side | State-distribution JS | Mean frozen-train distance | State-change rate | Unconditional net EV |
|---|---:|---:|---:|---:|
| July 13-19 long | 0.052 | 12.97 train-MAD z | 48.3% | -92.4 bps |
| July 13-19 short | 0.062 | 7.19 train-MAD z | 46.8% | -124.3 bps |

State occupancy remains close to training while observations move far from
the centroids and switch state rapidly. Every late-July state is negative.
This is a continuous within-state extrapolation and transition problem, not
evidence for a profitable hard-routed July specialist. The MAD z-score is
diagnostic only because tiny training MAD can make it unstable; models must
use the bounded empirical distance percentile/exceedance.

A strict July 13-19 residual-head input ablation trained only on resolved
May-June labels found:

| Scope | Baseline top-10 | + causal state inputs | Spearman change |
|---|---:|---:|---:|
| Long | -55.00 bps | -27.25 bps | 0.128 -> 0.177 |
| Short | -65.10 bps | -62.58 bps | 0.183 -> 0.189 |
| Global | -86.91 bps | -91.84 bps | 0.076 -> 0.083 |

The inputs contain useful side-local information, especially long, but raw
cross-side score comparability worsens the global book. Do not promote them
without repeated weekly confirmation and side-local OOF calibration followed
by one pooled global rank.

The rolling residual shrinkage sweep also fails to create a tradable July
tail. Fixed shrink 0.50 is the retention-constrained research winner:

| Residual adapter | June pooled weekly top-10 | July pooled weekly top-10 | July 13-19 |
|---|---:|---:|---:|
| Zero correction | +7.79 bps | -62.79 bps | -82.61 bps |
| Fixed 0.50 | +8.87 bps | -64.97 bps | -69.91 bps |
| Full 1.00 | -9.74 bps | -62.20 bps | -61.57 bps |
| ESS/drift adaptive | +13.62 bps | -67.36 bps | -75.70 bps |

The apparent positive June result above is a weekly diagnostic. The required
single global top-10 policy replay changes it: shrink 0.50 produces -4.92 bps
in June, -70.97 bps in July and admits zero July 13-19 rows. A total
concurrency cap of eight reduces notional-weighted results to +0.65 bps in
June and -27.92 bps in July, but the combined portfolio still compounds
-4.09% and latest-week admission stays zero. Portfolio constraints mitigate
losses; they do not repair score transfer or admission.

A side-local trust head was trained on nine purged expanding weekly folds.
Its target is cost-adjusted net utility versus abstaining, using hard-positive,
50-bps logistic-soft and 200-bps clipped-soft labels. It uses the immutable
21-day recent-mapped ranking score, causal pre-entry spread proxies, expert
disagreement and trailing unlabeled shifts. It does not use realized cost,
calendar regimes or regime weights.

Weekly-selection diagnostics make hard-positive trust-as-rank look marginally
positive (+2.56 bps versus -3.73 bps baseline), but its day-block delta
interval crosses zero and its latest week is -102.06 bps. Under the required
single pooled global top-10 selection, the baseline is +35.01 bps overall
because it is dominated by earlier rows, but its 223 selected July rows lose
-80.62 bps and it selects no July 13-19 rows. Trust challengers select only
5-18 July rows; those rows lose between -164.87 and -543.49 bps, reduce the
overall result to +33.24 to +35.22 bps, and still select no latest-week rows.
The current trust head is therefore rejected as a gate or ranking modifier.

Required next steps:

1. Extend the causal state source with actual decision-time market geometry:
   volatility and volatility-of-volatility, ATR term structure, trend/range
   efficiency, cross-asset breadth/dispersion/correlation, spread/depth,
   funding/basis, open-interest and liquidation-pressure features. The current
   common handoff contains mostly model/head geometry and cannot fully name
   the July market environment.
2. Fit side-local OOF calibrators after adding state inputs, then rank one
   globally comparable score. Keep state distance and transition probability
   continuous; do not hard-route on cluster IDs.
3. Train the 6h transition/persistence auxiliary head strictly OOF and ablate
   it as an input to the residual and trust heads. Add horizon variants
   1h/3h/12h and a soft transition-intensity target; retain only horizons that
   improve exact global-tail economics.
4. Separate opportunity prevalence from conditional magnitude: predict
   executable positive-net-EV probability, severe-loss probability and
   favorable magnitude conditional on a clean event, then combine them in
   common EV units. July has usable rank but a negative selected tail.
5. Re-run adjacent weekly July transfer with an expanding July adapter and
   explicit zero fallback. Promotion requires positive pooled global top-k,
   positive latest-week admission, May-June retention and constrained
   portfolio viability. Do not select shrink or trust thresholds on the
   evaluation week.
6. Only after recurring causal states show stable, different mappings should
   specialists be tested. Start with a global model plus a strongly shrunk
   state-conditional residual and uncertainty-weighted blending; never use
   calendar-month specialists or regime-similarity sample weights.

### A. Reversed-month diagnosis

Run a paired month-swap matrix:

1. canonical direction: train on earlier months and evaluate June/July;
2. reversed diagnostic: train on June/July and evaluate earlier months;
3. adjacent rolling controls: train through month `t-1`, evaluate month `t`;
4. matched-size controls so any difference is not just more/fewer rows.

The reversed experiment is **diagnostic, not OOS promotion evidence**, because
it trains on the future relative to the earlier evaluation months. Compare
base, residual, execution-EV, mapped global-top-`k`, calibration, head metrics,
feature drift, label prevalence, and exact-policy economics. If June/July-
trained models recover June/July but fail earlier periods, that supports a
regime-specific conditional relationship. If both directions fail, first
suspect labels, execution geometry, or insufficient features.

### B. Regime features and composites

Test point-in-time features at base and residual/meta layers:

- volatility level, volatility-of-volatility, ATR term structure, jump/gap
  intensity, and realized downside/upside semivariance;
- trend strength and persistence over multiple horizons, range efficiency,
  breakout compression/expansion, and mean-reversion speed;
- cross-sectional breadth, dispersion, correlation concentration, beta and
  residual-return dispersion;
- volume, turnover, liquidity, causal spread/slippage, funding/basis, open
  interest change, liquidation pressure, and market-impact proxies;
- change-point probability, distance from the training distribution, AE
  reconstruction error, GMM uncertainty/OOD, and recent head disagreement;
- composites such as trend x liquidity, volatility x breadth, funding x OI,
  alpha margin x OOD, clean-path probability x liquidity, and predicted
  opportunity magnitude x adverse-risk probability.

Features must be available at the decision time and selected per side. First
test whether they improve a single model's worst-fold/global-tail economics.
Also train a residual **trust head** whose target is whether each underlying
signal or expert adds positive cost-adjusted utility relative to the frozen
base score.

### C. Supporting-label audit

In addition to the existing event/path heads, ablate strict OOF per-side labels
for:

- cost-adjusted probability that a trade is executable and profitable;
- downside tail/severe-loss probability and expected loss conditional on loss;
- expected favorable magnitude conditional on reaching meaningful MFE;
- opportunity-loss probability if skipped or delayed;
- signal/expert utility delta versus the base model, including a soft
  `trust_this_signal` label;
- path efficiency, reversal risk after MFE, and time-discounted realized
  utility;
- regime-transition/change-point horizon and persistence;
- policy-action labels: enter now, wait-market, wait-limit/target price, or
  skip, including fill and missed-opportunity costs.

For every label, compare learnability, calibration, top-tail lift, stability,
and incremental exact-policy EV. A label is not retained merely because its IC
is high.

### D. Conditional specialists / mixture of experts

Only introduce regime specialists if:

1. the month-swap and rolling controls identify recurring regimes with
   materially different feature-to-utility relationships;
2. a single model with regime features, interactions, and reweighting cannot
   make all important forward folds economically viable;
3. regime membership can be inferred causally with adequate confidence and
   persistence.

Then:

1. define regimes from point-in-time features, not return labels available only
   after entry;
2. train per-side base/residual/meta experts with causal regime posteriors,
   distance and transition-risk predictions as inputs/interactions; do not
   use regime or calendar sample weights;
3. keep a global expert and shrink every specialist toward it according to
   effective sample size;
4. combine experts with a calibrated causal gating network or posterior
   mixture, including hysteresis/minimum-duration controls and an
   uncertain-regime fallback;
5. ablate hard routing, soft probability-weighted blending, and
   global-plus-specialist residual correction;
6. evaluate switching costs, calibration, expert utilization, worst-regime
   economics, mapped global-top-`k`, and constrained portfolio replay.

The preferred first specialist architecture is
`global residual + shrunk regime-specific residual corrections`, not fully
independent end-to-end models. It degrades gracefully when the gate is
uncertain and preserves shared statistical strength.

## 2026-07-26 actual-market-state, transition and July-transfer ablations

Status: **implemented and evaluated; no new trading arm is promotable**.

This work closes the six concrete follow-ups above. All supervised heads are
side-local and strict weekly OOF. All market-state joins use the latest
completed hourly bar with `source_timestamp <= decision_timestamp - 1h`,
maximum 90-minute staleness, and missing values rather than future or stale
substitution. Calibration is fitted per side on earlier resolved OOF rows,
then the policy performs one pooled global top-10% rank.

### 1. Decision-time market state

The new point-in-time source adds 28 train-covered inputs:

- volatility-of-volatility, ATR compression, slope/change and market ATR
  expansion;
- trend/range/path efficiency;
- breadth, dispersion and correlation concentration;
- funding level, dispersion, change and tail concentration;
- open-interest change/flush breadth; and
- explicitly named OI/funding/price/volume liquidation-pressure **proxies**.

`breakout_efficiency_4h` is excluded because train-only coverage is below the
predeclared 95% threshold. Historical funding and OI cover the full 71,586-row
May 31-July 19 candidate audit. A raw order-book sidecar exists for 99.79% of
rows, but only 904 rows/1.26% are true Kraken L2; the rest are local OHLCV
proxies. There is no historical observed-liquidation series. Consequently,
real spread/depth and observed liquidations are explicitly unavailable and
are not fabricated or mislabeled as measured inputs.

Implementation:

- `extreme_price_movements/execution_ev_market_state.py`
- `scripts/run_execution_ev_market_state_transition_heads.py`
- `tests/test_execution_ev_market_state.py`

### 2. Strict OOF transition/persistence heads

The merged result contains 716,916 predictions across six forward weeks,
June 8 through July 19, at 1h/3h/6h/12h. A label is admitted into training
only when its horizon has resolved before the evaluation-week boundary.

| Horizon | Combined AUC long | Combined AUC short | Latest-week AUC long | Latest-week AUC short |
|---:|---:|---:|---:|---:|
| 1h | 0.833 | 0.828 | 0.592 | 0.585 |
| 3h | 0.805 | 0.772 | 0.572 | 0.544 |
| 6h | 0.781 | 0.766 | 0.414 | 0.448 |
| 12h | 0.814 | 0.757 | 0.456 | 0.654 |

Market-state-only performance is essentially equal to combined performance
and materially above old geometry alone, whose aggregate AUC is 0.516-0.687.
The transition label is therefore learnable from actual market state.
However, the latest-week reversal at longer horizons proves that the mapping
is itself regime-sensitive. Every latest-week transition-probability top
decile also has negative net EV. Do not use this head directly for admission
or ranking. Retain 1h/3h probabilities, calibration/error and horizon
disagreement as candidate context/risk inputs; only retain 6h/12h if a
downstream strict-OOF ablation improves global-tail economics.

Artifact:
`data_perp/artifacts/execution_ev_raw_market_state_transition_heads_20260726_v2/`.

### 3. Decomposed execution EV

The side-local outer-OOF ablation trains:

- `P(net EV > 0)` and `E(net EV | net EV > 0)`;
- exact favorable-first `P(tb_hard_label = clean)` and
  `E(max(net EV, 0) | clean)`;
- any-loss and severe-loss probability plus their conditional magnitudes; and
- direct `E(net EV)` as the comparator.

The exact sign partition is
`P(pos) E(net|pos) - P(nonpos) E(-net|nonpos)`. The clean score is explicitly
a partial-risk score, not a complete EV identity:
`P(clean) E(max(net,0)|clean) - P(severe) E(-net|severe)`.
Every quantity is already in cost-adjusted net-return units; cost is applied
exactly once.

On the full 124-feature OOF population, probability-head learnability is
modest:

| Head | AUC | Average precision | Brier | Prevalence |
|---|---:|---:|---:|---:|
| Positive net EV | 0.582 | 0.649 | 0.243 | 0.585 |
| Clean favorable-first | 0.550 | 0.429 | 0.243 | 0.386 |
| Severe loss | 0.531 | 0.274 | 0.197 | 0.258 |

Calibrated aggregate pooled global top-10 EV is +9.56 bps direct, +0.83 bps
for the complete sign partition and +4.87 bps for clean partial-risk.
All three fail July: -36.57, -40.52 and -39.79 bps respectively.

The compact actual-market extension through July 19 confirms the conclusion.
Raw / side-isotonic / isotonic-plus-causal-21d direct EV is -38.26 / -29.13 /
-12.00 bps aggregate and -94.30 / -146.59 / -41.02 bps in July. Isotonic
alone is harmful; the recent correction mitigates the level shift but does
not restore a tradable tail. Decomposed heads improve interpretation and may
serve as supporting inputs, but neither decomposition replaces direct EV.

Artifacts:

- `data_perp/artifacts/execution_ev_decomposition_calibration_20260726_v5/`
- `data_perp/artifacts/execution_ev_decomposition_compact_market_july19_20260726_v1/`

### 4. Side-local calibration before one global rank

The compact ablation cleanly separates:

1. raw score;
2. temporal side-local isotonic fitted only on earlier OOF rows; and
3. isotonic plus causal 21-day side x predicted-archetype residual correction.

The July isotonic failure is not a small calibration defect: it selects an
all-short latest-week book. Even after the recent correction all 602 selected
July 13-19 rows are short. The next calibrator must therefore include
cross-side anchoring in common net-EV units and a no-current/future-row
contract; side-local monotonicity alone cannot guarantee global
comparability.

### 5. Adjacent-week July transfer with zero fallback

Blocks are July 1-5, July 6-12 and July 13-19. Each adapter sees only the
immediately previous block's labels that resolved before the next cutoff.
The first block and under-supported sides receive an exact zero correction.
Shrinkage is frozen at 0/0.25/0.50/1.00 and evaluated with one pooled global
top-10 selection; weekly top-10 is diagnostic only.

| Adapter inputs / shrink | Pooled July top-10 | July 13-19 weekly diagnostic |
|---|---:|---:|
| Zero correction | -31.46 bps | -74.92 bps |
| Model geometry, 0.25 | -41.19 bps | -122.15 bps |
| Model geometry, 0.50 | -38.93 bps | -112.80 bps |
| Model geometry, 1.00 | -39.84 bps | -103.08 bps |
| + actual market state, 0.25 | -46.10 bps | -97.16 bps |
| + actual market state, 0.50 | -62.34 bps | -76.60 bps |
| + actual market state, 1.00 | -61.55 bps | -72.08 bps |

The adjacent-week correction does not transfer reliably even within July.
Actual market state improves transition classification but does not yet tell
the residual learner how to change its economic mapping. Zero correction
remains the research control; no adapter goes to portfolio replay because
every challenger is dominated before constraints.

Artifacts:

- `data_perp/artifacts/adjacent_july_state_adapter_ablation_20260726_v1/`
- `data_perp/artifacts/adjacent_july_market_state_adapter_ablation_20260726_v2/`

### 6. Specialist eligibility

The predeclared gate requires two prior blocks, recurring states, state-effect
rank correlation at least 0.50, sign consistency at least 0.75 and at least
20 bps effect range. It rejects both state bases:

- model-geometry states: long rank correlation is -1 with 0.50 sign
  consistency; short sign consistency is 0.33 and effect range only 12.1 bps;
- actual-market states: only one recurring state per side, so there is no
  stable state differentiation to exploit.

The specialist arm therefore returns the exact baseline score. Do not train
calendar-month or hard-routed July experts. The evidence supports a
continuous, rapidly changing state/context problem, not stable recurring
specialists.

### Decision and next experiments

What worked:

1. causal actual-market features predict state transition much better than
   model geometry;
2. strict side-local OOF infrastructure, horizon-aware resolution, coverage
   gating and zero fallback behave as intended;
3. the causal 21-day EV-level correction partially mitigates the July level
   shift; and
4. explicit EV decomposition exposes whether failure is prevalence,
   magnitude or downside risk.

What did not work:

1. transition probability as an economic ranking signal;
2. side-isotonic calibration without cross-side anchoring;
3. adjacent-week residual adaptation, with or without actual market state;
4. decomposed scores as replacements for direct EV; and
5. state specialists under the current recurrence/stability gate.

Required next ablations, in order:

1. Add true historical L2 spread/depth and observed-liquidation feeds before
   claiming those families have been tested. Until then keep causal proxies
   explicitly named as proxies.
2. Feed 1h/3h transition probabilities, their OOF calibration residuals and
   horizon disagreement into direct EV as continuous interactions. Compare
   input, uncertainty penalty and abstention-only uses; never rank the raw
   probability.
3. Replace plain side isotonic with a hierarchical calibration objective:
   side-local monotonic calibration plus a pooled common-unit anchor, guarded
   by minimum effective sample size and exact zero fallback.
4. Model **mapping change**, not just state: train a strict-OOF
   `trust_base_signal` / expected residual-utility head using transition
   probability, market-state velocity, feature-distance percentile, model
   disagreement and cost/liquidity interactions. Judge it by paired global
   top-10 delta and latest-week admission.
5. Use multi-task shared representation for positive opportunity, clean path,
   conditional magnitude and severe loss, but retain direct EV as the primary
   head. Add monotonic/coherence penalties such as `P(clean) <= P(pos)` only
   when the exact targets logically imply them.
6. Repeat rolling weekly/month-swap tests over more recurring episodes.
   Specialists remain blocked until the same causal state recurs in at least
   three OOS blocks with stable, economically different mappings.

The current preferred architecture is:

`base + residual alpha + CatBoost + auxiliary heads`
`-> direct execution EV with decomposed risk/opportunity support`
`-> hierarchical side calibration into common EV units`
`-> one pooled global top-k`
`-> separate timing / target-price / wait action layer`
`-> portfolio constraints`.

Transition/state features enter as continuous context, uncertainty and trust
inputs. They do not select a specialist, override the global rank, or enter
the timing action layer as a raw trade score.

## 2026-07-26 transition-use, trust, hierarchical and multi-task follow-up

Status: **implemented and evaluated; direct EV remains the production
primary, and no new arm is promotion-eligible**.

All comparisons below preserve the strict prior-OOF/resolution contract and
one pooled global top-10 rank. Weekly selections are diagnostics only.

### Transition context: rejected downstream

The transition overlay uses the strict OOF 1h/3h/6h/12h probabilities only as:

- continuous direct-EV interactions;
- probability uncertainty/range features;
- a bounded same-unit residual correction; and
- a prior-resolved side-local adverse-utility penalty.

It never ranks a raw transition probability. June 8 has an observable exact
zero fallback because there are no earlier transition-OOF labels.

| Arm | Pooled global top-10 | July 13-19 weekly diagnostic |
|---|---:|---:|
| Frozen direct EV | +14.45 bps | -74.92 bps |
| Transition context/uncertainty/risk overlay | -15.06 bps | -97.56 bps |

The overlay is decisively rejected. Transition remains a diagnostic/context
family until a narrower interaction proves incremental on repeated forward
blocks.

Implementation/artifact:

- `scripts/run_execution_ev_transition_context_overlay.py`
- `data_perp/artifacts/execution_ev_transition_context_overlay_20260726_v1/`

### Strict-OOF `trust_base_signal`: learnable, not incremental

The new side-local trust workstream predicts:

1. residual economic utility =
   `realised execution_net_ev_12h - frozen mapped EV`;
2. expected absolute mapping error, in net-return units; and
3. soft trust = positive realised utility multiplied by mapping reliability.

Trust is used only as a residual interaction, uncertainty penalty or
abstention input. It is never substituted for EV as the raw ranking score.

Head learnability:

| Diagnostic | Aggregate |
|---|---:|
| Trust positive-utility AUC | 0.587 |
| Expected absolute mapping-error Spearman | 0.261 |
| Residual-utility Spearman | -0.021 |

The mapping-error head learns something, but the correction direction does
not. Under one pooled global top-10:

| Arm | Aggregate | July selected tail | Latest-week admission |
|---|---:|---:|---:|
| Frozen baseline | +35.01 bps | -80.62 bps | 0 |
| Trust-residual interaction | +30.35 bps | -441.72 bps on 6 rows | 0 |
| Combined trust/mapping gate | +26.75 bps | -84.48 bps on 32 rows | 0 |
| Mapping-uncertainty penalty | +7.33 bps | -543.49 bps on 5 rows | 0 |

The trust head is rejected as a score correction or gate. Expected mapping
error may remain a monitoring/position-sizing input, but not an admission
feature without new forward evidence.

Implementation/artifact:

- `scripts/run_trust_base_signal_mapping_ablation.py`
- `data_perp/artifacts/execution_ev_trust_base_signal_mapping_20260726_v1/`

### Hierarchical calibration: keep as research winner

Calibration stages are now:

1. raw;
2. temporal side-local isotonic;
3. nested hierarchical calibration, where early prior OOF rows fit the two
   side maps and a disjoint later prior-OOF segment fits one pooled common-EV
   anchor; and
4. hierarchical calibration plus the causal 21-day side x predicted
   archetype residual correction.

The disjoint split prevents the pooled anchor from being trained on its own
side-mapped outcomes. Current-fold rows/outcomes are excluded from both
layers.

On the compact Jul19 v2 run, direct-EV top-10 economics are:

| Stage | Aggregate | July | July 13-19 |
|---|---:|---:|---:|
| Raw | -38.26 bps | -94.30 bps | -111.34 bps |
| Side isotonic | -29.13 bps | -146.59 bps | -124.00 bps |
| Hierarchical pooled anchor | -31.32 bps | -99.90 bps | -99.84 bps |
| Hierarchical + causal 21d | -1.59 bps | -22.20 bps | -63.92 bps |

The pooled anchor repairs most of the catastrophic side-only comparability
failure. The recent correction supplies most economic recovery. Keep
hierarchical-plus-recent as the calibration research default, but do not
promote it while latest-week economics remain negative.

Artifact:
`data_perp/artifacts/execution_ev_hierarchical_multitask_compact_july19_20260726_v2/`.

### Multi-task decomposition: direct EV remains primary

Two strict prior-outer-OOF multi-task variants were evaluated:

- a pooled Ridge combiner of direct EV plus decomposed probability/magnitude
  heads; and
- a genuine side-local shared-trunk MLP. The shared loss has four repeated
  standardized direct-EV outputs and five auxiliary outputs: soft positive
  utility, clean event, soft severe loss, positive magnitude and loss
  magnitude. Only the averaged direct outputs become the trading score.

The Ridge blend dominates direct EV at the raw stage across aggregate,
May/June/July and latest week, but every raw tail remains negative. In the v3
within-run comparison:

| Raw score | Aggregate | July | July 13-19 |
|---|---:|---:|---:|
| Direct EV | -33.27 bps | -95.95 bps | -109.20 bps |
| Prior-OOF Ridge auxiliary blend | -15.95 bps | -69.59 bps | -77.98 bps |
| Shared multi-task direct head | -22.12 bps | -64.18 bps | -127.45 bps |

After hierarchical-plus-recent calibration, aggregate EV becomes +2.55 bps
direct, +11.55 bps Ridge and +11.46 bps shared multi-task. However, latest
week is -86.70 bps direct and -88.09 bps for both auxiliary variants. The
Ridge and shared variants also fail strict non-inferiority after the intended
calibration. Direct EV therefore remains the production primary; the Ridge
blend is a raw-score research challenger only.

Artifact:
`data_perp/artifacts/execution_ev_hierarchical_shared_multitask_compact_july19_20260726_v3/`.

### Rolling recurrence and older-history audit

Six expanding weekly raw-market-state gates were evaluated from June 8
through July 19. Each side's state geometry is fitted outcome-free before the
evaluation week; economic residuals must resolve before the cutoff. Only the
short side before June 22 independently passes recurrence/stability. Long
fails, so the global specialist gate remains exact zero fallback. Every other
week fails. No July specialist is eligible.

Older history was explicitly audited:

| Historical population | Raw-state PIT | Stored strict-OOF score | Same 12h execution-EV target | Eligible |
|---|---:|---:|---:|---:|
| Dec-Feb canonical, 766,758 rows | 100% | No | No; first-touch path up to 96 bars | No |
| April PackB OOF, 258,670 rows | 100% | Yes | No; first-touch/capture up to 96 bars | No |
| May 5-Jul19 frozen EV, 123,824 rows | Yes | Yes | Yes | Yes |

Raw-state history is not the blocker: all 181 symbols pass the point-in-time
probe, with 28/29 features above 95% coverage. The missing requirement is a
comparable strict-OOF direct/base execution-EV score and 12h execution-EV
target. Mixing the 96-bar first-touch target into the 12h residual gate would
answer a different question. The maximum honest existing recurrence window
therefore remains May 5-July 19.

Artifacts:

- `data_perp/artifacts/raw_market_state_backward_recurrence_20260726_v1/`
- `data_perp/artifacts/historical_raw_state_recurrence_join_audit_20260726_v1/`

### Updated decision

Keep:

1. direct EV as the primary production objective;
2. nested hierarchical side calibration plus pooled common-unit anchor as the
   calibration research default;
3. the causal 21-day correction;
4. decomposed heads and transition probabilities as supporting context; and
5. exact zero fallback whenever prior support or recurrence fails.

Reject for promotion:

1. transition context/uncertainty overlay;
2. current trust/mapping correction and abstention gate;
3. shared multi-task or Ridge output as the production score; and
4. regime specialists.

The next specialist experiment requires backfilling a canonical deployed-policy 12h
execution-EV outcome and strict-OOF direct score into Dec-Feb/April. Without
that backfill, more historical clustering can diagnose raw-state recurrence
but cannot validate recurrence of the economic mapping.

## 2026-07-27 historical 12h, drift, multi-task and portfolio closure

Status: **implemented and audited; no challenger is promotion-eligible**.

### Historical 12h fee-only comparator panel

The historical runner rebuilds the current side-parent 12h geometry directly
from completed Kraken hourly paths under a fixed 1% fee-only approximation. It
uses a one-hour signal-to-decision delay and a 12-hour
decision-to-resolution interval. It never substitutes the archived
96-bar/first-touch target, but it is not deployed-policy economic replay:
historical executable entry/exit spread and slippage are absent.

| Population | Rows |
|---|---:|
| Archived causal candidates, Mar-2025–Apr-2026 | 1,059,363 |
| Complete hourly fee-only current-geometry diagnostic labels | 945,922 |
| Per-side expanding-month OOF direct scores, May-2025–Apr-2026 | 816,564 |
| Monthly complete-path coverage | 86.2–93.4% |

Every OOF training cutoff precedes its decision; every label resolves 12 hours
after the delayed decision; gross minus the one policy cost reconciles to net.
This reconciles fixed-fee accounting only; spread/slippage economics are
absent. The nine score/context inputs contain no outcome-derived reliability
rates.

The following limitations are promotion-material:

1. the archived incumbent candidate/base-score stream still begins in March
   2025. Pre-March evidence therefore requires a new fold-local raw/PIT base
   score and must not be described as exact incumbent parity;
2. the source 55-feature representation was selected on the July-2026 largest
   fold and reused backward under the explicitly approved feature-selection
   exception. Row scores remain prior-row-only OOS, but the representation is
   future-selected, so this panel is for diagnosis/recurrence research rather
   than untouched strict-OOS promotion;
3. deep one-minute history exists for only a subset of symbols in 2024; global
   store bounds do not certify the point-in-time candidate universe.
   Authoritative funding begins only on 2026-04-22 and trustworthy historical
   L2 spread/depth is absent. Late-2024 must remain an hourly comparator unless
   candidate-level one-minute completeness is separately proven.

A machine-verifiable reconstruction audit confirms the boundary rather than
inferring it from missing output files:

| Required source | Files / rows | Earliest timestamp |
|---|---:|---:|
| Canonical execution-one-minute partitions | 33,882 / 82,711,683 | 2022-07-20 19:22 UTC, but not full-universe coverage |
| Raw/PIT feature files | 249 / 6,585,893 | 2022-06-11 02:00 UTC |
| Compatible source candidate/feature ledgers | 39 / 4,987,993 | 2025-01-01 00:00 UTC |
| Archived causal candidate folds | 26 / 1,219,076 | 2025-03-01 00:00 UTC |

Therefore:

- January–February 2025 is reconstructible with exact one-minute 12h paths by
  fitting a new per-side raw/PIT base model. January needs a nested warm-up;
  February can be fully forward OOS;
- the old55/current score cannot be reproduced honestly: all six
  `meta_sel_ood_*` fields and seven `rel_*` fields are absent and must not be
  copied backward. Every pre-March imputer, feature selector and supervised
  model is fitted inside its permitted prior fold;
- late-2024 can support a separately reported hourly-path comparator, never a
  pooled exact-one-minute policy/timing result;
- a future-trained frozen backcast, random within-month fold or all-symbol
  hourly population must not be presented as the missing OOF evidence.

The exact-one-minute-path, fee-only Jan-Feb reconstruction is now complete:

| Exact-one-minute tier | Result |
|---|---:|
| Source candidates | 436,497 |
| Exact 720-minute path coverage | 100% in January; 99.992–99.993% in February |
| Exact labels after causal 14h ATR availability | 420,386 |
| Nested two-layer strict OOF rows, Jan 15–Feb 28 | 326,328 |
| Single pooled global top-10 fee-only net EV, diagnostic | -19.06 bps |
| Same global book, January slice | -60.04 bps |
| Same global book, February slice | +31.82 bps |
| Same global book, long / short | -113.85 / +10.87 bps |

The base layer uses 99 raw numeric PIT inputs, fold-local top-40 Spearman
selection and a cost-aware soft-positive 12h target. The execution-EV layer
uses only inner weekly base OOF scores plus cutoff, z-score, timestamp-rank and
candidate-group context. Strict two-layer scoring begins January 15; every
February row is forward OOS. The positive February/short result is useful
score-order and raw-state diagnostic evidence only, not deployed-policy
economic regime evidence. January and long-side failure reject promotion in
either case.

Implementation/artifact:

- `scripts/backfill_historical_execution_ev_12h_oof.py`
- `data_perp/artifacts/historical_comparable_execution_ev_12h_oof_20260726_v3/`
- `scripts/reconstruct_janfeb2025_execution_ev_12h_oof.py`
- `data_perp/artifacts/janfeb2025_execution_ev_exact1m_two_layer_oof_20260727_v2/`
- `scripts/audit_late2024_execution_ev_reconstruction_readiness.py`
- `data_perp/artifacts/late2024_execution_ev_reconstruction_readiness_20260727_v3/`

The late-2024 comparator is also complete. It builds the candidate universe
only from cached feature rows physically present at each timestamp, uses
July–September as nested OOF warm-up and reports October–December only. It
keeps the hourly approximation in a separate artifact and prohibits pooling
with the exact-one-minute tier.

| Hourly-only late-2024 tier | Result |
|---|---:|
| PIT feature rows, July–December | 862,916 |
| Current-geometry hourly labels, both sides | 1,286,042 |
| Strict two-layer OOF rows, October–December | 696,236 |
| Single pooled global top-10 net EV | -41.81 bps |
| October / November / December global-book slices | -53.91 / -63.59 / -29.06 bps |
| Long / short global-book slices | -45.20 / -35.86 bps |

All three months and both sides are negative. This rejects a stable profitable
late-2024 recurrence under the comparable raw/PIT architecture. The result is
diagnostic only: the configured raw feature family is later-selected, cached
historical transforms have not been bitwise recomputed from truncated raw
history, path resolution is hourly, and historical L2/spread is unavailable.

Implementation/artifact:

- `scripts/reconstruct_late2024_execution_ev_hourly_comparator.py`
- `data_perp/artifacts/late2024_execution_ev_hourly_comparator_20260727_v2/`

### Corrected causal calibration, EV drift and asymmetric uncertainty

The first drift artifact (`v1`) is **invalid and superseded**: a broad prefix
filter admitted post-decision `h1/h3/h6/h12` state fields. The final `v10`
runner loads only explicitly authorized pre-entry predictions and market-state
fields ending in `h0`; a fail-closed test verifies that later state horizons
are excluded.

All drift/uncertainty heads are side-local and trained only on earlier,
resolved outer-OOF rows. Calibration remains a side monotonic map followed by
a disjoint pooled common-EV anchor. Ranking is one pooled global top-10 after
mapping.

| Causal score | Aggregate | July | Latest week |
|---|---:|---:|---:|
| Direct EV | -33.27 bps | -95.95 bps | -109.20 bps |
| Best calibration-only: hierarchical 21d uniform | -25.49 bps | -95.95 bps | -109.37 bps |
| Signed residual EV-drift correction | **+19.07 bps** | -25.84 bps | -61.13 bps |
| Asymmetric overestimate LCB, lambda 0.5 | -15.53 bps | -78.33 bps | -91.32 bps |

The signed drift head is useful diagnostic evidence that EV mapping changes,
but it remains negative in July/latest and its aggregate global book has
negative long economics. It is not promotable. Overestimate uncertainty is
directionally better than generic/symmetric uncertainty, but fold-2 head
quality turns weak: overestimate Spearman is 0.147 then -0.039, and signed
residual Spearman is 0.119 then 0.030. Abstention is not incremental; sizing
only reduces exposure.

Direct-score standard deviation also compresses sharply across outer folds:
84.7, 108.8 and 53.2 bps, while realized standard deviation stays near 261,
294 and 238 bps. Cross-fold score-scale transport therefore remains a
first-order calibration problem.

Implementation/artifact:

- `scripts/run_execution_ev_calibration_drift_uncertainty_ablation.py`
- `data_perp/artifacts/execution_ev_calibration_drift_uncertainty_july19_20260726_v10/`

### Strengthened direct-primary multi-task architecture

The side-specific shared-trunk stacker now tests:

1. direct-only;
2. all five economic auxiliaries with 2x/4x/8x direct loss;
3. each auxiliary removed individually;
4. a +/-3 training-standard-deviation clipped direct target; and
5. residual-to-frozen-direct learning.

The auxiliaries are soft positive utility, clean favorable path, soft severe
loss, positive magnitude and loss magnitude. Direct net EV is the sole score
output. Every fit uses only prior resolved outer-OOF component predictions and
falls back exactly to frozen direct EV when history is insufficient.

The complete diagnostic grid improves older aggregate rows but not the latest
period:

| Arm | Aggregate | July | Latest week |
|---|---:|---:|---:|
| Frozen direct | -33.27 bps | -95.95 bps | -108.59 bps |
| All auxiliaries, direct loss 2x | **-15.38 bps** | -75.24 bps | -108.85 bps |
| Drop positive magnitude | -15.99 bps | **-39.38 bps** | -72.54 bps |
| Drop positive event | -16.26 bps | -52.61 bps | **-71.34 bps** |

No auxiliary subset is stable across periods. Dropping severe loss or loss
magnitude is generally worse, but the best omission changes between July and
the latest week.

A full-capacity confirmation then refit the four plausible variants using up
to 30,000 prior rows per side/fold and 28 iterations rather than the
8,000-row/12-iteration diagnostic cap. It does not rescue the architecture:
aggregate results range from -16.16 to -20.85 bps and latest-seven-day results
from -106.47 to -131.01 bps. The rejection is therefore not merely diagnostic
underfitting.

Implementation/artifacts:

- `scripts/run_execution_ev_direct_primary_multitask_timescale_ablation.py`
- `data_perp/artifacts/execution_ev_direct_primary_multitask_timescale_20260726_v1/`
- `data_perp/artifacts/execution_ev_direct_primary_multitask_full_capacity_20260727_v1/`

### Separate opportunity and environment timescales

The action layer does not alter EV rank:

- 1h/3h OOF transition and persistence forecasts produce a
  `wait_or_reprice` recommendation;
- 6h/12h persistence, transition, horizon disagreement, expected causal state
  age and causal raw-state velocity feed a weekly side-local probability that
  realized EV undershoots frozen direct EV by at least 50 bps;
- one pooled global top-10 is selected first; action diagnostics operate only
  on that admitted book.

The 6h/12h uncertainty head is weak and unstable (weekly AUC mostly
0.34–0.56). Fixed 0.50/0.60/0.70 risk thresholds do not improve frozen direct:
the overlap control is -25.66 bps versus -27.94/-26.89/-25.81 bps.

The 1h/3h timing rule does isolate a worse cohort: 850 flagged frozen-direct
candidates are -59.82 bps, while the remaining 5,125 are -19.99 bps. This is
evidence for abstention/timing research only. It is not evidence that waiting
earns a better fill because executable delayed-entry prices, missed-opportunity
outcomes and a refreshed post-wait 12h exit path do not yet exist.

### Common promotion and portfolio confirmation

The common audit reports aggregate and globally admitted latest-fold/month/7d
economics, both sides, fold score coverage, side/fold/month composition and
per-fold side calibration deciles. Coverage gates are promotion checks, never
selection quotas. Ranking remains one pooled global top-10.

No evaluated score passes. Important examples:

| Score | Aggregate | Globally admitted latest fold | Latest 7d | Latest-fold rows |
|---|---:|---:|---:|---:|
| Signed drift h0 | **+19.07 bps** | -33.18 bps | -62.57 bps | 1,235 |
| Asymmetric LCB 0.5 | -15.53 bps | -60.91 bps | -60.91 bps | 53 |
| Diagnostic multi-task all aux 2x | -15.38 bps | no rows | no rows | 0 |
| Hierarchical 21d uniform | -25.49 bps | +321.52 bps on only 5 rows | no rows | 5 |

This distinguishes real recent failure from scale-collapse artifacts. Sparse
positive values based on one to five recent admissions are not evidence.

The signed-drift challenger was nevertheless replayed diagnostically through
the exact frozen policy and canonical portfolio constraints. The baseline
portfolio accepts 1,323 trades at -57.81 bps mean net return and compounds
-80.53%; July accepted trades average -66.15 bps. The least-bad constraint,
four concurrent positions per side, still compounds -65.37%. Constraints
cannot rescue the score.

Implementation/artifacts:

- `scripts/audit_execution_ev_promotion_candidates.py`
- `data_perp/artifacts/execution_ev_common_promotion_audit_20260727_v2/`
- `data_perp/artifacts/execution_ev_signed_drift_portfolio_constraints_20260727_v1/`

### Historical forward/reverse transfer diagnosis

A fixed side-local CatBoost residual control uses the historical OOF direct
score plus the nine causal score/support fields, three prior months and the
exact 12h net target. It evaluates one global top-10 within each side/month;
future-to-past fits are diagnostic and never promotion evidence.

Forward top-10 economics from August 2025 through April 2026:

| Side | Positive months | Mean monthly top-10 | Only positive month |
|---|---:|---:|---:|
| Long | 0 / 9 | -46.29 bps | none |
| Short | 1 / 9 | -31.00 bps | October 2025, +75.31 bps |

The October short effect is also positive under its matched future-trained
diagnostic (+53.59 bps), whereas every other forward month is negative.
This supports one narrow October short opportunity, not a stable recurring
expert. It does not satisfy the three-independent-episode specialist gate.

Artifacts:

- `data_perp/artifacts/historical_12h_regime_transfer_long_20260727_v1/`
- `data_perp/artifacts/historical_12h_regime_transfer_short_20260727_v1/`
- `data_perp/artifacts/historical_12h_regime_transfer_long_latest_20260727_v1/`
- `data_perp/artifacts/historical_12h_regime_transfer_short_latest_20260727_v1/`

### Actionable decision

Keep the production architecture unchanged:

`base + residual alpha + CatBoost + five auxiliary heads`
`-> direct execution EV as primary`
`-> nested hierarchical side calibration into pooled common EV units`
`-> one pooled global top-k`
`-> separate timing / target-price / wait action layer`
`-> portfolio constraints`.

Next work, in order:

1. build true 1h/3h delayed-entry and missed-opportunity labels, including
   refreshed exit geometry after waiting, before testing wait/reprice actions;
2. fit cross-fold EV scale transport using only prior resolved rows, with
   dynamic identity shrinkage and explicit fold/month admission diagnostics;
3. extend the signed residual-drift head over the historical panel, using only
   feature families available causally in both eras, then freeze it for a
   genuinely later OOS block;
4. refit raw-feature-level multi-task models only after identifying auxiliaries
   that improve at least two independent recent OOS blocks;
5. keep asymmetric overestimate risk for monitoring/sizing research, not
   admission, until its AUC and economic threshold transfer forward;
6. do not create regime specialists unless one causal state recurs profitably
   in at least three independent OOS episodes with stable side-specific
   economic effects.

### Workstream completion audit

The requested implementation workstream is complete; this is not a model
promotion claim.

| Requirement | Authoritative evidence | Completion result |
|---|---|---|
| Fee-only December–April OOF diagnostic history and extension through 2025 | March-2025–April-2026 hourly panel, including December-2025–April-2026; 816,564 strict OOF rows | Implemented and hash-verified; invalid for deployed-policy economics |
| January–February 2025 extension | Exact-one-minute-path, fee-only nested two-layer panel; 326,328 OOF rows from January 15, all February forward OOS | Implemented and hash-verified; invalid for deployed-policy economics |
| Late-2024 extension | Separate October–December hourly comparator; 696,236 strict OOF rows; no metric pooling or 1m/L2/timing parity claim | Implemented and hash-verified |
| Hierarchical calibration and explicit EV drift | Seven hierarchical calibration arms plus signed residual drift, fitted from prior resolved OOF rows; final causal runner admits only `h0` market state | Implemented; no arm passes recent promotion gates |
| Asymmetric uncertainty | Overestimate/downside LCB, abstention and sizing arms | Implemented; diagnostic improvement is not recent-period stable |
| Strengthened multi-task architecture | Direct-only, all-five-auxiliary loss weights, every leave-one-head-out arm, clipping, residual-to-direct and full-capacity confirmation | Implemented; none is promotion-eligible |
| Opportunity/environment timescales | Separate 1h/3h wait/reprice diagnostics and 6h/12h environment-risk heads | Implemented; timing isolates a bad cohort, delayed-entry benefit remains unproven |
| Policy and portfolio confirmation | One frozen pooled global top-10 book replayed through concurrency, wallet, per-side, per-symbol and entry-rate constraints | Implemented; no constraint rescues negative economics |

Completion verification:

- all artifact payload hashes and canonical manifest hashes pass for the three
  historical panels;
- the final drift feature inventory contains no post-decision market-state
  horizon: every admitted `mkt_state__*` field ends in `__h0`;
- the common promotion manifest has zero eligible challengers and uses one
  pooled global top-k across sides;
- the portfolio manifest freezes that supplied global book and contains
  baseline, concurrency, wallet, per-side, per-symbol and entry-rate arms;
- 39 focused tests covering labels, purging, inner base OOF provenance,
  calibration/drift causality, asymmetric uncertainty, multi-task fallback,
  global admission, policy constraints and historical reconstruction pass;
- `git diff --check` passes.

Production remains unchanged because completion of the research infrastructure
does not override the negative OOS promotion evidence.

## 2026-07-27 next-step decision after the July transfer audit

Status: **diagnosis complete; prioritize executable action labels and a frozen
forward confirmation, not another execution-EV architecture sweep**.

The equal-size July audit does not support a July specialist. The two valid
chronological comparisons are July 1--7 to July 8--14 and July 8--14 to July
15--19. The pooled expanding-forward cohort has 12,483 rows, Spearman 0.2065
and AUC 0.6336, but its global top-decile EV is -17.31 bps with a day-block
95% interval of -77.46 to +15.60 bps. The learner finds some ordering signal;
it does not establish positive, stable economics. Random cross-fit and
in-sample results remain diagnostics and cannot override this conclusion.

The other proposed remedies are now closed unless genuinely new data changes
the evidence:

| Proposed remedy | Strict evidence | Decision |
|---|---|---|
| Equal-size within-July learner | Positive rank statistics, negative and uncertain forward top-decile EV | Do not promote |
| July leaf/state specialist | Later-July top-decile EV -96.7 bps; all adjacent-state adapters worse than baseline | Do not build a mixture of experts |
| Mixed-period reweighting/trust composites | Best later-July arm remains -81.2 bps | Stop reweighting sweep |
| Trust/environment residual | Trust target is learnable, but every gate/residual degrades July economics | Monitoring only |
| Hierarchical scale transport | Best causal drift arm improves aggregate EV but remains negative in July/latest week | Keep identity/fail-closed production behavior |
| Probability/magnitude decomposition and multi-task auxiliaries | Some aggregate improvement; no stable recent-period pass | Do not promote |
| Portfolio constraints | Least-bad constrained replay still compounds materially negative | Downstream risk control only |

Long-side evidence isolates a separate failure. On the exact January--February
one-minute panel, long base-top-decile rows are +50.5 bps gross/-49.5 bps net,
whereas direct-EV-top-decile rows are -12.2 bps gross/-112.2 bps net. The
execution-EV ranking therefore destroys useful base ordering before cost; the
1% round-trip cost is an additional viability hurdle. Geometry or portfolio
constraints cannot be treated as a substitute for fixing that ranking.

### Required next ablations

1. **Executable timing/action value -- first priority.**
   Materialize exact fixed 720-minute paths and refreshed post-fill policy
   outcomes for `enter_now`, waited market entry at 60 and 180 minutes, and
   adverse-limit entries at 0.25 and 0.50 decision-time ATR with 60/180-minute
   expiry. Charge fee, entry spread and exit spread exactly once. Report fill
   probability, conditional filled EV, missed-opportunity loss, action utility,
   action regret versus enter-now, adverse-first risk, exit reasons and global
   admitted-book EV. Select the global top-k book first; timing acts only on
   admitted rows and never changes EV rank.

   The leakage-safe implementation already exists in
   `extreme_price_movements/execution_entry_timing_meta.py`,
   `scripts/materialize_execution_entry_timing_1m_paths.py`,
   `scripts/materialize_execution_entry_timing_handoff.py`, and
   `scripts/run_execution_entry_timing_meta.py`. No completed signed timing
   artifact exists yet. The current signed execution labels end on July 10,
   the canonical one-minute store was observed through July 21, and no source
   parquet currently contains the complete identity + ATR + decomposed
   fee/spread input contract. Do not run the timing model on a partial inner
   join. First rebuild that candidate handoff, refresh one-minute data through
   the terminal 12-hour window, and require 100% month/symbol coverage without
   `--allow-subset`.

2. **Long ranking versus exit geometry -- second priority and identical rows.**
   On exact one-minute OOF rows, freeze candidate IDs and compare base rank,
   direct-EV side rank, and the long contribution to pooled global top-k.
   Report gross/net EV, correlations and calibration deciles, MFE/MAE, exit
   causes, and weekly/monthly stability. Only after that decomposition, replay
   the identical base-selected and EV-selected rows through a small
   one-at-a-time long geometry grid: stop scale 0.9/1.0/1.1, trailing
   activation 0.9/1.0/1.1, and giveback 0.9/1.0/1.1. Keep fees, scores and
   portfolio admission fixed. Any selected geometry must be confirmed on a
   later untouched block; January--February cannot be tuned repeatedly.

3. **Frozen forward confirmation -- mandatory stop condition.**
   Freeze production identity/fail-closed execution EV, the signed-drift
   challenger, the timing action grid and all gates before the next unused
   block. Use no calendar or regime quota: admission remains one pooled global
   top-k after causal EV mapping. Require positive cost-adjusted EV
   simultaneously in aggregate, the latest complete fold, both sides with
   adequate coverage, and the constrained portfolio replay. A sparse positive
   slice or improved AUC/IC is insufficient.

### Explicitly deferred

- Do not add a July expert, GMM/DAE gate, or new trust composite until a causal
  state has recurring economic effects in at least three independent forward
  episodes.
- Do not launch another broad label/HPO sweep before the executable timing and
  identical-row long decompositions identify which error is being corrected.
- Keep target-price suggestions in the action layer. The existing
  `adverse_limit` action is a pre-entry price suggestion; the live safety
  take-profit target is post-entry exit management and must not be relabelled
  as an entry-timing target.

## 2026-07-27 executable entry-timing ablation

Status: **implemented and evaluated; retain enter-now-only**.

The previously missing timing infrastructure is now complete:

- a signed 201,685-row source candidate contract reconstructs causal ATR
  fraction, 30 bps round-trip fee and per-symbol p90 spread from the signed
  execution-EV target;
- the broad source-universe path audit finds 154,593/201,685 exact paths.
  Almost the entire gap is April: 47,025 missing April rows versus only 67
  missing May--July rows. This broad artifact is coverage evidence only;
- the actual downstream universe is the exact execution-EV OOF handoff, not
  the April-heavy source superset. All 127,777 rows in that universe have an
  exact contiguous 720-minute path;
- the trainable timing handoff contains 116,244 rows where both direct and
  residual execution-EV predictions are finite, true outer-OOF. Exact path
  coverage is 100% inside this declared OOF universe;
- counterfactual labels are now cached as a fingerprinted, train-only
  813,708-row ledger so feature/HPO retries do not repeat policy simulation.

The decision-time path contract was corrected during the audit. The signed
execution-EV manifest and one-minute materializer define the first executable
bar at `__decision_ts__`; the label builder previously expected
`__decision_ts__ + 1m`. It now uses the signed boundary exactly, with a
720-minute path ending at decision + 719 minutes and conservative label
resolution at decision + 12 hours.

The strict action grid is:

1. enter now;
2. wait-market for 60 or 180 minutes;
3. adverse-limit at 0.25 or 0.50 decision-time ATR, expiring after 60 or
   180 minutes.

Every filled action refreshes the frozen 12-hour exit policy from its own fill.
Unfilled limits receive explicit missed-opportunity loss. Fee, entry spread
and exit spread are charged exactly once.

### Head learnability

The action components are too weak for unrestricted routing:

| LGBM action head | Fill AUC | Adverse AUC if filled | Delta-EV Spearman if filled | Utility Spearman |
|---|---:|---:|---:|---:|
| Limit 60m / 0.25 ATR | 0.574 | 0.547 | 0.199 | -0.003 |
| Limit 60m / 0.50 ATR | 0.541 | 0.622 | 0.181 | -0.009 |
| Limit 180m / 0.25 ATR | 0.563 | 0.557 | 0.234 | -0.003 |
| Limit 180m / 0.50 ATR | 0.546 | 0.625 | 0.214 | approximately 0 |
| Wait-market 60m | deterministic fill | 0.625 | -0.003 | 0.039 |
| Wait-market 180m | deterministic fill | 0.630 | 0.010 | 0.043 |

The current `adverse_first` threshold is also poorly discriminating: roughly
90% of enter-now paths hit 0.25 ATR adverse movement before the meaningful-MFE
condition. It is useful as a path descriptor but not as the primary action
gate.

### Economics after the required global admission

Timing is evaluated only after causal recent side-isotonic EV mapping and one
pooled global top-10% selection across all timestamps and both sides. There is
no per-timestamp quota and timing never reranks EV. The mapped intersection is
114,096 OOF rows and the admitted book contains 11,410 rows.

| Policy | Action EV | Enter-now control | Delta |
|---|---:|---:|---:|
| LGBM unrestricted | -47.21 bps | -8.18 bps | **-39.03 bps** |
| Fixed grid unrestricted | -52.63 bps | -8.18 bps | -44.45 bps |
| Ridge/logistic unrestricted | -43.79 bps | -8.18 bps | -35.61 bps |

LGBM selects adverse limits on 46.4% of the admitted book, fills only 68.6%,
and misses 22.0% of profitable enter-now trades. The short contribution is
especially poor: -64.41 bps versus its enter-now control.

Frozen safety-gate ablations do not rescue the layer. The best aggregate
result, LGBM with at least 50 bps predicted delta, at least 80% fill
probability and at most 15 bps expected missed EV, improves by only +0.18 bps:
+0.37 bps in May, -0.16 bps in June and no July actions. This is noise and
fails the independent-period gate. Enter-now-only remains the frozen action.

Implementation/artifacts:

- `scripts/materialize_execution_entry_timing_candidates.py`
- `scripts/materialize_execution_entry_timing_1m_paths.py`
- `scripts/materialize_execution_entry_timing_handoff.py`
- `scripts/run_execution_entry_timing_meta.py`
- `scripts/evaluate_execution_entry_timing_global_topk.py`
- `scripts/ablate_execution_entry_timing_action_gates.py`
- `data_perp/artifacts/execution_entry_timing_candidates_20260727_v1/`
- `data_perp/artifacts/execution_entry_timing_1m_paths_oof_universe_20260727_v1/`
- `data_perp/artifacts/execution_entry_timing_handoff_20260727_v1/`
- `data_perp/artifacts/execution_entry_timing_meta_60m_180m_20260727_v2/`
- `data_perp/artifacts/execution_entry_timing_global_topk_20260727_v1/`
- `data_perp/artifacts/execution_entry_timing_action_gates_20260727_v1/`

Twenty-one focused timing/provenance tests pass and all changed Python modules
compile. `git diff --check` passes.

### Required next timing architecture

Do not tune the current unrestricted router further. The next timing learner
must predict pairwise action value relative to enter-now:

1. a soft `P(action utility > enter-now utility)` target plus conditional
   positive delta magnitude;
2. a separate fill hurdle for limit actions and explicit missed-opportunity
   magnitude, rather than reconstructing utility from several weakly
   calibrated components;
3. a less saturated adverse target, selected from training-only prevalence
   and economic severity, such as severe pre-MFE MAE or loss-tail delta;
4. uncertainty-aware lower-confidence-bound routing with enter-now as the
   mandatory fallback;
5. initial restriction to wait-market 60m and at most one sufficiently
   learnable limit action; expand the grid only after positive forward action
   delta in at least two independent OOS blocks.

These changes remain in the separate action layer. They must not add timing or
MAE fields back into execution-EV admission.

The surviving configuration is frozen for the next genuinely later block in
`data_perp/artifacts/execution_ev_frozen_forward_confirmation_20260727_v1/manifest.json`.
The earliest eligible data begins July 24 UTC, but evaluation waits for at
least 14 complete UTC days, 5,000 scored rows and 500 pooled global top-10
capacity rows. Economically admitted rows may be zero. The incumbent,
signed-drift monitor, conservative timing monitor, mapping, global top-k rule
and promotion gates are immutable for that block.

## 2026-07-27 economic-failure, regime-observability and label-grid workstream

Status: **implemented and evaluated; no new production arm promoted**.

### Identical-row economic-failure decomposition

`scripts/diagnose_execution_ev_economic_failure.py` now joins the frozen raw
residual, causal 21-day EV maps, canonical ascending base rank and exact-policy
path outcomes on the immutable
`(__ts__, __symbol__, side_name, candidate_id)` identity. It hard-fails on
duplicates, conflicting outcomes and gross/cost/net non-reconciliation.
Score orientation is explicit: base rank is lower-is-better; EV scores are
higher-is-better.

The canonical diagnostic contains 114,096 jointly finite rows. Its primary
admission is one pooled global top 10% across all timestamps and both sides.
Month-local diagnostics also rank only once across all timestamps and sides;
there is never a timestamp or side quota.

| Score | Pooled top-10 gross | Cost | Net | Positive-net | MFE / MAE |
|---|---:|---:|---:|---:|---:|
| Canonical base rank | 77.47 bps | 75.71 bps | +1.76 bps | 58.1% | 205 / 106 bps |
| Raw residual | 39.46 bps | 71.86 bps | -32.40 bps | 54.5% | 196 / 132 bps |
| 21d global isotonic EV | 71.00 bps | 67.88 bps | +3.12 bps | 61.8% | 205 / 127 bps |
| 21d side-isotonic EV | 59.11 bps | 68.02 bps | -8.91 bps | 59.9% | 203 / 134 bps |

The global 21-day map replaces 6,378 of the raw residual's 11,410 selected
rows. Added rows average -1.70 bps versus -65.25 bps for dropped rows, improving
the selected book by +35.52 bps. The side map improves the raw book by
+23.49 bps but remains negative.

The failure is not only calibration. In the month-local pooled-global top 10%,
the raw residual's gross opportunity falls from 75.56 bps in May and 70.47 bps
in June to only 18.36 bps in July, below the 54.96 bps realized cost. July net
EV is therefore -36.61 bps even though MFE remains 171 bps and 64.7% of rows
finish positive net. The global and side 21-day maps are also negative in July
(-40.86 and -55.59 bps). The July problem is a combination of weaker realized
capture/opportunity economics, score-scale transport and ranking error; another
calibrator alone cannot repair it.

Artifact:
`data_perp/artifacts/execution_ev_economic_failure_diagnosis_20260727_v2/`.

### Strict chronological regime observability

`scripts/run_chronological_regime_observability.py` creates two separate
diagnostics:

1. train-reference-only robust feature shift, with no evaluation-fitted
   transform; and
2. a fixed logistic week-outcome classifier. A week is positive only when the
   frozen 21-day score's single pooled-global top-10 book is profitable. Each
   fold trains only on earlier complete weeks whose final 12-hour label has
   resolved strictly before the evaluation-week boundary. Each week receives
   equal total loss weight, regardless of candidate count.

Only five forward evaluation weeks are available and only one is profitable.
This is the effective sample size; 47,697 candidate rows do not turn five weeks
into 47,697 independent regime observations.

| Feature family | Row AUC | Week AUC | Week Brier | Interpretation |
|---|---:|---:|---:|---|
| Volatility | 0.778 | 1.00 | 0.171 | strongest signal, but only 1 positive / 4 negative weeks |
| Open interest | 0.572 | 0.75 | 0.181 | suggestive |
| Trend/range | 0.671 | 0.75 | 0.183 | suggestive |
| Breadth | 0.548 | 0.50 | 0.195 | no demonstrated value |
| Head context | 0.292 | 0.25 | 0.233 | inverse/non-transfer |
| All market h0 | 0.273 | 0.25 | 0.274 | over-combined and unstable |
| Head + market h0 | 0.261 | 0.25 | 0.293 | worse |

The late-July shift is real, with funding mean z-score the largest robust shift
on July 13--19. Coefficients repeatedly associate downside correlation
concentration, funding-tail concentration and post-liquidation rebound state
with adverse week probability, but five weeks are insufficient for a live
gate.

The corresponding safe architecture test uses train-only, outcome-free
volatility-state balancing through robust scaling and fixed MiniBatchKMeans.
It improves the May-to-June control from +11.14 to +15.23 bps, but worsens the
later-July book from -85.47 to -106.84 bps. This arm is rejected. No hard
regime gate, July expert or state specialist is authorized.

Artifacts:

- `data_perp/artifacts/chronological_regime_observability_20260727_v1/`
- `data_perp/artifacts/execution_ev_observable_volatility_balancing_20260727_v3/`

### Exact 12h/24h meaningful-MFE label grid

The previous studies already covered the canonical 12-hour soft barrier and
clean versus competing-risk models. The missing test was horizon and economic
threshold sensitivity. New infrastructure materializes exact, contiguous
hourly paths for the full 127,777-row execution-EV OOF universe:

- 12h and 24h horizons;
- 1.5 ATR and 2.0 ATR favorable barriers, each retaining the 1.5% economic
  return floor;
- 1.0 ATR adverse barrier;
- conservative adverse assignment for same-hour conflicts;
- an explicit 1% round-trip cost margin;
- time to 80% MFE, first-three-bar clean/non-flat quality and directional
  close slope as supporting labels.

All four cells have 100% exact path coverage.

| Grid | Favorable first | Adverse first | Timeout |
|---|---:|---:|---:|
| 12h / 1.5 ATR | 37.9% | 53.3% | 8.8% |
| 12h / 2.0 ATR | 32.2% | 55.8% | 12.0% |
| 24h / 1.5 ATR | 41.3% | 56.8% | 1.9% |
| 24h / 2.0 ATR | 36.7% | 60.5% | 2.7% |

The classifier ablation freezes the prior April-authorized per-side features
and CatBoost geometries; it performs no new HPO. June and July are expanding
OOF with each cell's true 12h/24h resolution and purge.

| Grid / best statistical arm | AUC | AP | Global top-10 precision | Superseded hourly-target net EV |
|---|---:|---:|---:|---:|
| 12h / 1.5 ATR, competing risk | 0.535 | 0.409 | 42.8% | -15.63 bps |
| 12h / 2.0 ATR, soft | 0.502 | 0.325 | 34.8% | **-1.71 bps** |
| 24h / 1.5 ATR, competing risk | 0.512 | 0.425 | 43.4% | -7.25 bps |
| 24h / 2.0 ATR, competing risk | 0.497 | 0.356 | 34.6% | -23.16 bps |

The near-neutral 12h/2.0-ATR soft result is not a winner: AUC is effectively
random. The net-EV column above came from the now-superseded hourly execution
target and must not be read as exact-policy economics. The statistical result
still shows that extending to 24 hours does not repair event learnability. Do
not add these event scores to EV admission.

Artifacts/implementation:

- `extreme_price_movements/meaningful_mfe_label_grid.py`
- `scripts/materialize_meaningful_mfe_label_grid.py`
- `scripts/run_meaningful_mfe_label_grid_ablation.py`
- `data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/`
- `data_perp/artifacts/meaningful_mfe_label_grid_ablation_20260727_v1/`

### Supporting-label heads: learnability and incremental July value

The corrected 12h/1.5-ATR grid was also used to test three supporting targets:

- first-three-bar clean/non-flat path quality;
- time quality to the fixed economic barrier (not time to 80% of eventual
  MFE, which was confounded by larger trends peaking later); and
- clipped directional close slope transformed through a fixed sigmoid.

Each head uses the frozen April-authorized per-side soft-head features and
CatBoost geometry. There is no supporting-label HPO. Predictions are expanding
monthly OOF with exact label-resolution checks and a 12-hour purge. June OOF
predictions train a fixed per-side logistic stack; July is scored once.

| Supporting head | Pooled Spearman | Long | Short | Head-only pooled top-10 net |
|---|---:|---:|---:|---:|
| Early path quality | 0.079 | 0.066 | 0.063 | -96.02 bps |
| Economic-barrier time quality | 0.076 | 0.131 | 0.024 | -115.91 bps |
| Slope quality | -0.076 | -0.045 | -0.064 | -120.65 bps |

The direct July stack is one pooled global top 10% across all timestamps and
both sides. It is an event-stack diagnostic, not the production recent-EV
mapping or a portfolio replay.

| July event stack | AUC | Favorable-first at top 10% | Exact net EV | Change vs event-only |
|---|---:|---:|---:|---:|
| Event scores only | 0.476 | 33.7% | -153.56 bps | — |
| + early path | 0.467 | 32.9% | -152.26 bps | +1.30 bps |
| + economic-barrier time | 0.478 | 35.3% | -142.03 bps | +11.53 bps |
| + slope | 0.474 | 32.0% | -162.82 bps | -9.26 bps |
| + all three | 0.467 | 33.4% | -149.74 bps | +3.82 bps |

Barrier time replaces 185 of 1,517 rows. Its added rows lose 111.96 bps versus
206.51 bps for the rows it drops, so the improvement is genuine replacement
value, but the resulting book remains deeply negative. Early path and the
combined stack make only small improvements; slope is destructive. Side
diagnostics do not rescue the result: every support arm remains negative for
both sides.

Decision:

- do not add any of these predictions directly to EV admission;
- drop slope quality from the next admission experiment;
- retain economic-barrier time, and secondarily early-path quality, only as
  candidate auxiliary training losses in the future capture/regret model;
- require any future use to improve the final score after causal recent-EV
  mapping, pooled global top-k and constrained portfolio replay. The present
  event-stack result is not promotion evidence.

Artifact:
`data_perp/artifacts/meaningful_mfe_supporting_heads_exact_policy_ablation_20260727_v1/`.

At this checkpoint, thirty-five focused tests across this workstream pass.
The changed runners compile. Later capture-specific and observable-geometry
tests are recorded in their own subsections below.

### Exact-policy opportunity/capture hurdle result

The bounded cost-aware hurdle was implemented with fixed per-side CatBoost
geometry and no HPO. Every head is trained and calibrated through temporal
OOF predictions only:

1. soft opportunity is
   `sigmoid((12h path MFE - exact row cost) / 25 bps)`;
2. capture is `P(exact net > 0 | path opportunity clears exact row cost)`;
3. positive magnitude is the conditional log positive exact net;
4. capture guard is conditional exact gross divided by path MFE; and
5. every composite is mapped back to exact net EV using earlier OOF outcomes,
   then receives the same causal recent-EV mapping and one pooled global top
   decile.

| Arm | May -> June | Later July | July delta vs direct | Interpretation |
|---|---:|---:|---:|---|
| Direct exact-net residual | -60.69 bps | -106.54 bps | baseline | Existing fixed model |
| Opportunity only | -130.85 bps | -114.42 bps | -7.88 bps | Reject |
| Opportunity x capture probability | -63.98 bps | -104.24 bps | +2.30 bps | Small July gain, fails June |
| Opportunity x capture x positive magnitude | -88.11 bps | -107.85 bps | -1.31 bps | Reject |
| Opportunity x capture x capture guard | -109.77 bps | **-100.18 bps** | **+6.36 bps** | Best July challenger, large June failure |

The last arm raises selected July MFE by 31.73 bps relative to direct net and
reduces the MFE-to-gross gap by only 25.33 bps; cost is unchanged. Its July
gain is real under the exact accounting identity, but all arms remain deeply
negative and the best July arm loses 49.08 bps versus direct net in the June
control. This rejects the current hurdle as an admission model and confirms
that a raw MFE opportunity head is not sufficient. The useful residue is the
conditional capture probability: keep it as a candidate auxiliary/abstention
signal, not a standalone ranker.

Artifact:
`data_perp/artifacts/exact_policy_capture_hurdle_ablation_20260727_v1/`.

### Executable pre-exit capture support and mapping diagnosis

The next study removes whole-horizon MFE from the predictive target. Exact
one-minute candles are re-read from the immutable deployed-policy lineage only
through the actual policy exit, inclusive. The materializer covers all
134,889 canonical rows and keeps the conservative full 12-hour
`label_resolution_utc`. It emits:

- pre-exit MFE and MAE;
- favorable-before-adverse ordering at the exact row cost and half-ATR
  thresholds, with same-minute OHLC ambiguity explicit;
- MFE-to-exact-gross gap and gross capture ratio;
- close-path give-back after 50% and 80% of pre-exit MFE;
- exact positive-net and severe-loss events.

None of these realized fields is a decision-time feature. The ablation trains
fixed per-side CatBoost heads through temporal OOF predictions, then tests only
bounded support around the direct exact-net model. The residual meta layer is
nested temporal OOF. No geometry or label choice is tuned on either forward
evaluation block.

The net-positive capture head is the first support target in this workstream
that transfers in rank quality:

| Head | June AUC / Spearman | Later-July AUC / Spearman | Raw-head top-10 net |
|---|---:|---:|---:|
| `P(exact net > 0)` | 0.621 / 0.206 | 0.648 / 0.216 | -63.38 / -66.59 bps |
| severe loss | 0.587 / 0.138 | 0.557 / 0.087 | -69.52 / -93.78 bps |
| favorable-before-adverse | 0.573 / 0.126 | 0.537 / 0.056 | -59.06 / -90.60 bps |
| capture ratio | n/a / 0.103 | n/a / 0.086 | -86.23 / -89.20 bps |
| give-back magnitude | n/a / 0.342 | n/a / 0.255 | -110.81 / -92.27 bps |

June is shown first in the last column. Capture probability is therefore
learnable across both environments; conditional economics, not event
classification alone, remains the bottleneck.

Under the original archetype-local 21-day recent-EV correction:

| Arm | June | Later July | Decision |
|---|---:|---:|---|
| direct exact-net | -60.69 bps | -106.54 bps | incumbent research baseline |
| capture only | **-57.79** | -102.39 | small improvement in both |
| direct + capture residual | -160.13 | **-70.12** | non-transferable; reject |
| low-20% capture abstention | -60.69 | -79.54 | useful July gate, no June change |
| severe-loss high-20% veto | -60.62 | -98.56 | weak improvement in both |
| bounded full support | -111.21 | -73.20 | non-transferable; reject |

Soft exact-net labels at 25/50/100-bp temperatures and hard +50/+100-bp
events are learnable but do not clear the economics gate after the original
recent mapping. The most revealing case is the 25-bp soft label: -9.65 bps in
June before recent correction becomes -101.59 bps afterward. Recent
calibration must therefore be fitted and retained as part of each score
architecture, not treated as an interchangeable downstream component.

A causal mapping-scope ablation uses identical 21-day resolved-only outcomes:

| Score and causal mapping | June | Later July |
|---|---:|---:|
| direct + archetype-local correction | -60.69 | -106.54 |
| capture + global correction | **-49.06** | -86.31 |
| distributional net + global correction | -71.93 | **-71.70** |
| distributional net, no recent correction diagnostic | -69.97 | -70.08 |

The clean distributional score is
`P(win) * E(positive net | win) - P(loss) * E(loss magnitude | loss)`.
It is the most regime-stable score tested so far, but remains approximately
-72 bps in both blocks. The capture/global configuration dominates the direct
incumbent in both blocks, while the distributional/global configuration has
the better worst-period result. Neither is tradeable after the exact
approximately 100-bp cost.

The global capture selection also lacks robust latest-day economics: only 29%
of selected June days and 20% of selected later-July days are positive, and
the later-July selection has effectively no admission coverage after July 15.
No arm qualifies for simple-policy or constrained-portfolio replay.

Canonical artifacts:

- `data_perp/artifacts/exact_policy_capture_labels_20260727_v1/`
- `data_perp/artifacts/exact_policy_capture_support_ablation_20260727_v8/`

#### OOF-only abstention and shared-magnitude follow-up

Capture abstention is now selected only from temporal OOF training rows.
Candidate rejection percentiles are 0/10/20/30/40%. Each candidate first
receives the same causal global 21-day resolved-only mapping used at
evaluation, then is judged by the minimum of:

- pooled global OOF top-decile exact net; and
- exact net among the globally selected rows in the latest seven training
  days.

At least 100 latest rows are required. If no candidate has that support, the
optimizer fails closed to percentile zero.

For May-trained -> June, direct EV selects a 20% capture veto, but the rejected
rows do not alter the mapped global top decile. Capture-only and the separate
distributional score select no veto. For the May-through-July-11 training
window, every base score has only 7-17 globally selected OOF rows in its latest
seven days. All therefore fail coverage and freeze no gate. This is useful
negative evidence: an apparently better fixed July veto cannot be selected
causally from the available score history.

A shared CatBoost `MultiRMSE` head was also tested with positive-net and
negative-net contributions as its two outputs. It is rejected:

| Score + global 21-day mapping | June | Later July |
|---|---:|---:|
| Separate conditional distributional score | -71.93 bps | -71.70 bps |
| Shared two-output contribution score | -69.68 bps | -120.52 bps |

The shared representation marginally helps June and collapses in July. Keep
the separate probability and conditional-magnitude formulation.

#### Side-local feature-input screen

The exact capture heads previously received only 39 downstream context
fields. The frozen alpha winners use 31 long-side and 8 short-side features.
Those exact lists were point-looked up from the immutable causal feature store
for all 134,889 rows, producing 70 long / 47 short inputs when the long
winner's DAE/GMM fields are included and 68 / 47 for raw-only. Coverage is
100% across 156 symbols.

This is a strict add-one screen, not capture-outcome feature selection:

| Score + global mapping | Core 39 | + raw 31/8 | + raw 31/8 and frozen representation |
|---|---:|---:|---:|
| Capture only, June | **-49.06** | -63.35 | -48.35 |
| Capture only, later July | **-86.31** | -97.79 | -93.97 |
| Distributional, June | -71.93 | -72.17 | **-70.63** |
| Distributional, later July | **-71.70** | -90.48 | -88.40 |

Capture AUC is 0.621/0.648 for the core, 0.624/0.630 with raw 31/8, and
0.624/0.639 with representation; June/later-July are shown in that order.
The alpha-selected raw features therefore do not transfer to the capture
task. DAE/GMM recover part of the raw-only July loss but remain worse than the
core. Reject both expansions and do not run MDA/HPO on these lists.

Artifacts:

- `data_perp/artifacts/exact_policy_capture_feature_expansion_20260727_v1/`
- `data_perp/artifacts/exact_policy_capture_feature_expansion_raw31_8_20260727_v1/`
- `data_perp/artifacts/exact_policy_capture_support_expanded31_8_ablation_20260727_v1/`
- `data_perp/artifacts/exact_policy_capture_support_expanded_raw31_8_ablation_20260727_v1/`

### Decision and narrowed next step

Keep the production architecture and frozen confirmation unchanged. Current
evidence rejects:

- more calibration-only work;
- observable-volatility state balancing;
- hard regime gates or July specialists;
- the 24-hour meaningful-MFE event target;
- promoting any label-grid event head;
- directly adding the tested supporting-head predictions to EV admission;
- broad HPO of the existing residual or timing router.

The exact-policy intra-July test, opportunity/capture decomposition, first
cost-aware hurdle, executable capture support, soft-label grid and
distributional score are now complete. The capture event transfers, but no
tested score clears costs. The next model experiment is still not a regime
router. It must improve conditional economic magnitude and admission:

1. retain `P(exact net > 0)` as a support head and the global 21-day mapping as
   its current mapping challenger;
2. keep separate conditional win and loss magnitude heads; the shared
   multi-task contribution head is rejected;
3. keep the OOF-only capture-abstention optimizer fail-closed; the present
   later training window cannot support a nonzero gate;
4. require non-sparse latest-block coverage and positive exact net EV in both
   June and later July before portfolio replay or regime specialization; and
5. the capture-specific, train-only side-local raw-feature screen is now
   complete and rejected, as documented below. Do not launch feature MDA/HPO;
   the add-one failed the required dual-forward-block gate; and
6. return upstream to candidate admission and the frozen
   exit-policy research lane. A downstream meta head cannot manufacture the
   missing approximately 50-70 bps needed to clear costs.

Economic-barrier time and early-path quality may be auxiliary losses for the
capture component, but not direct admission scores. Only after this global
model shows forward value should a small uncertainty-bounded observable-regime
correction be tested. Any correction must fall back to the global score, use
no calendar identity, and beat it in two independent forward blocks after
causal 21-day EV mapping, one pooled global top-k and constrained portfolio
replay. Until that support exists, regime probability remains a monitor rather
than a trade gate.

### Capture-specific causal feature screen

The requested capture-specific selector is complete. The canonical causal
feature universe is the frozen loader contract's 256 decision-time fields, not
the broader 1,352-column schema inventory. The first 1,352-column
materialization was diagnostic only and is not used by the result. The
canonical materialization is:

`data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/`

It covers all 134,889 exact-policy rows and all 156 symbols. Of 256 frozen
causal fields, 249 have at least 99% coverage on both sides and over the full
period.

Feature selection is repeated inside every side-local temporal fit. It
requires:

- at least 99% fit-row coverage;
- the same signed Spearman direction against exact positive net in early and
  late fit halves;
- positive top-decile lift in both halves;
- no more than 24 fields in total or four per feature family; and
- correlation pruning above 0.95.

No evaluation row, later fold, or realized field enters selection. The
selected families are mainly leverage, market, liquidity, volatility and
microstructure, but the exact fields and even family counts change materially
between the May and later-July fits.

The fixed capture classifier uses no HPO. Comparable raw probability and
economic results are:

| Forward block | Arm | Raw AUC | Raw top-10 net | Global 21-day mapped top-10 net |
|---|---|---:|---:|---:|
| May -> June | compact core | 0.627 | -52.30 bps | **-44.94 bps** |
| May -> June | capture-selected raw | 0.597 | -63.79 bps | -46.37 bps |
| later July | compact core | **0.649** | **-71.54 bps** | **-84.32 bps** |
| later July | capture-selected raw | 0.647 | -89.92 bps | -106.53 bps |

The selected raw add-one loses 1.43 bps in June and 22.21 bps in later July
after the canonical global mapping. It therefore fails the explicit
both-forward-block gate. Do not run feature MDA, geometry HPO, or a larger
capture feature search.

Canonical result:
`data_perp/artifacts/capture_specific_feature_screen_20260727_v2/`.

### Observable-stream one-axis exit-geometry result

The earlier scalar geometry sweep had already tested parent, side-only and
side x base-rank-decile scales from 0.8 to 1.2. It remained negative and the
decile overlay did not help. The bounded follow-up now tests the missing
geometry dimensions independently:

- stop multiplier at 0.9 and 1.1 of the champion;
- trailing activation at 0.9 and 1.1; and
- giveback beta at 0.9 and 1.1.

The parent is the frozen one-minute champion. Fees and spread-aware executable
fills are unchanged. Admission is frozen as one pooled global top 10% inside
each outer OOF fold, with deterministic identity tie-breaking. Each fold's
side or side x observable-decile geometry is selected only from selected rows
in earlier folds, using a one-standard-error fallback to the parent. There are
no side, timestamp, calendar or regime quotas.

Parent replay parity is exact: maximum and mean absolute net-return deltas are
both zero. Gross minus cost equals net for every arm and row. The artifact
also records candidate-level cost, MFE, MAE, exit reason/hour, fold selection
identity hashes, side metrics, and daily-block confidence intervals.

| Nested policy | Later-fold rows | Mean net | Worst fold | Latest fold |
|---|---:|---:|---:|---:|
| side parent | 7,079 | -91.97 bps | -98.77 bps | -84.48 bps |
| side-only | 7,079 | **-91.91 bps** | -100.64 bps | **-82.28 bps** |
| side x base-rank decile | 7,079 | -92.73 bps | -100.31 bps | -84.38 bps |

Side-only gains only 0.67 bps in aggregate and 2.20 bps in the latest fold,
while worsening the worst fold by 1.87 bps. Side x decile is worse in
aggregate. The best fixed perturbation, 0.9 giveback beta, is -90.02 bps over
the later folds; all seven fixed arms remain negative. The latest fold has
3,367 selected observations, so this failure is not caused by the sparse
latest-fold coverage of the earlier across-all-folds cutoff.

No exit-geometry challenger is promoted. Do not launch a joint or broad
geometry HPO: the bounded axes show only one-to-two-basis-point repairs against
an approximately 90-basis-point deficit. The next upstream question is whether
the admitted candidates contain attainable net opportunity under any small
frozen executable policy family. Materialize that counterfactual opportunity
and exit-regret decomposition before changing the execution head again.

Canonical result:
`data_perp/artifacts/observable_exit_geometry_ablation_20260727_v2/`.

Six additional focused tests cover the corrected 256-field universe, nested
capture selection, fold-local global top-k admission, one-standard-error
fallback, and one-axis geometry isolation. Together with the earlier 35, the
workstream now has 41 focused passing tests.

### Executable opportunity, wider exit family, and pairwise timing follow-up

The identical fold-local global-top-10 rows were decomposed into path
opportunity, executable gross capture, cost, and policy regret. The
counterfactual family oracle is the per-row hindsight maximum over the seven
frozen nearby executable policies. It is diagnostic only and is never used as
a causal score.

| Later-fold component | Mean |
|---|---:|
| full-path MFE | +400.05 bps |
| path MFE minus parent cost | +300.01 bps |
| parent gross / net | +8.07 / -91.97 bps |
| nearby-family oracle gross / net | +50.47 / -49.78 bps |
| recoverable regret inside nearby family | +42.19 bps |
| path-MFE to family-gross gap | +349.58 bps |

The latest fold is consistent: +408.47 bps path MFE, +308.39 bps after the
cost hurdle, but only +62.62/-37.70 bps family-oracle gross/net over 3,367
rows. Even a hindsight arm choice in the nearby family cannot clear costs.
The candidates contain large favorable excursions, but those excursions are
not captured cleanly by small stop/activation/giveback changes.

Artifact:
`data_perp/artifacts/exit_opportunity_regret_decomposition_20260727_v1/`.

Two bounded wider exit families test whether the gap is caused by the
trailing-only family:

1. hard take-profit at 1.25%, 1.50% or 2.00%, 0.5x trailing activation, 0.5x
   giveback beta, or a 1.50% activation cap;
2. fixed target/stop brackets from 1.50%/1.00% through 3.00%/2.00%.

The hard-target arms raise the target-hit rate to 51-72% but make mean
economics worse because the remaining adverse exits are too large. The best
fixed wider arm is 0.5x giveback at -83.80 bps in later folds, an 8.18-bps
repair that remains far below zero. Nested side-only selection is -89.84 bps
overall and -79.99 bps in the latest fold. Every target/stop bracket is worse
than the parent (-107.16 to -119.39 bps in later folds), so both nested
selectors correctly choose the parent everywhere.

Artifacts:

- `data_perp/artifacts/observable_economic_exit_family_ablation_20260727_v1/`
- `data_perp/artifacts/observable_bracket_exit_family_ablation_20260727_v1/`

Do not expand exit-policy HPO. The large MFE is frequently reached only after
adverse movement that defeats a bounded stop, which makes entry timing and
clean favorable-before-adverse ordering the remaining causal questions.

The required pairwise timing architecture was therefore tested on the already
signed 116,244-row timing handoff and 813,708-row counterfactual action
ledger. It uses the unchanged 18 frozen pre-entry inputs, separate side-local
action models, three expanding outer folds, 12-hour purge and embargo, and
inner-OOF residual lower-confidence bounds. Admission remains the causal
recent side-isotonic EV score followed by one pooled global top 10%; timing
does not rerank it.

The primary action is wait-market 60m versus enter-now. The model directly
learns:

- a soft `P(action utility > enter-now utility)` target;
- conditional positive and negative action-delta magnitudes;
- a separate fill hurdle and missed-opportunity magnitude; and
- an inner-OOF residual-q10 lower confidence bound with enter-now fallback.

One 180m/0.25-ATR limit is retained only as a secondary diagnostic and cannot
change the primary wait-only verdict.

| Pairwise action | Better-action rate | OOF AUC | Expected-delta Spearman | Positive LCB |
|---|---:|---:|---:|---:|
| wait-market 60m | 41.56% | 0.510 | 0.003 | 0.0% |
| limit 180m / 0.25 ATR | 63.66% | 0.522 | 0.075 | 0.0% |

The mapped identity intersection has 114,002 rows and 11,401 admitted rows.
Enter-now is -8.30 bps. Every 0/25/50-bps wait-only or wait-plus-limit
lower-confidence policy selects enter-now on every row, so its delta is
exactly zero. This is a valid fail-closed result, not a positive timing
finding. The first outer fold has insufficient inner OOF support and also
fails closed; later folds have large negative residual-q10 offsets, consistent
with the near-random wait-value discrimination.

Artifact:
`data_perp/artifacts/pairwise_entry_timing_ablation_20260727_v1/`.

Retain enter-now. Do not add another wait horizon, limit offset, timing HPO, or
timing/MAE admission feature. The next learning task must improve clean
favorable-before-adverse candidate admission itself; neither nearby/wider exit
geometry nor the separate wait/reprice layer can recover the current tail.

Three additional focused tests cover opportunity/regret accounting and the
pairwise soft-target reconstruction. The workstream total is now 44 focused
passing tests, in addition to the separate 21-test timing/provenance suite.

### Meaningful-MFE admission screen and July leaf-recurrence decision

The next admission experiment is complete. It corrects an important target
distinction: the exact-cost capture label asks whether price clears the
round-trip-cost hurdle before reversing and before the parent policy exits.
That is not the requested meaningful-MFE question. The canonical screen uses
the fixed `h12_u1p5atr` label instead:

- twelve-hour complete post-decision path;
- favorable barrier `max(1.5 x entry ATR, 1.5%)`;
- adverse barrier `1.0 x entry ATR`;
- same-hour favorable/adverse conflict assigned conservatively to adverse; and
- the existing canonical soft label as the CatBoost training target.

The target geometry is frozen and is not part of feature selection or HPO.
The experiment joins the exact 134,889-row label grid to the outcome-free,
frozen 256-field causal universe by exact candidate identity. Models and
feature selection are side-local. Selection is repeated inside every
temporally purged fit and requires at least 99% fit coverage, stable signed
hard-event IC and positive event top-decile lift in both halves of the fit.
No evaluation outcome enters selection. The fixed arms are compact context,
all 256 causal fields, and nested top-64/top-128 screens. There is no HPO.

All scores are evaluated raw and after side-local train-OOF isotonic
exact-net mapping plus the causal global 21-day correction. The primary
economic metric remains one pooled global top 10% across sides and timestamps.

| Forward block | Arm | Raw event AUC | Raw top-10 net | 21-day mapped top-10 net |
|---|---|---:|---:|---:|
| May -> June | compact context | 0.554 | -67.68 bps | -57.76 bps |
| May -> June | all 256 | 0.585 | **-33.07 bps** | **-48.75 bps** |
| May -> June | top 64 | 0.585 | -66.59 bps | -73.81 bps |
| May -> June | top 128 | **0.587** | -65.85 bps | -72.70 bps |
| May+June -> later July | compact context | **0.555** | -104.03 bps | -129.63 bps |
| May+June -> later July | all 256 | 0.494 | -156.01 bps | -152.32 bps |
| May+June -> later July | top 64 | 0.492 | -100.12 bps | -97.21 bps |
| May+June -> later July | top 128 | 0.483 | **-92.90 bps** | **-93.94 bps** |

The final nested selector retains 52/63 long/short fields for the June fit and
58/51 for the later-July fit. Selection improves later-July mapped economics
by 55-58 bps versus all 256, but worsens June by 24-25 bps; every arm remains
negative. Latest-seven-day selected coverage is 708 rows in later July, so the
failure is not caused by sparse coverage.

The manifest now computes the promotion decision rather than merely describing
it. A selected arm may enter MDA/HPO only if it has positive exact net,
improves all 256, and has adequate latest coverage in both June and later July.
Neither arm qualifies. Do not run clean-event MDA or HPO.

Canonical result:
`data_perp/artifacts/meaningful_mfe_clean_event_feature_screen_20260727_v1/`.

The July-only residual leaf diagnostic explains why a July specialist is not
yet justified. A model trained on purged, resolved July 1-7 rows produces a
small positive raw top-decile result on July 8-10:

- 5,303 pooled rows, AUC 0.579, Spearman 0.155;
- raw pooled top-10 net `+5.28 bps`.

The relationship does not survive later July:

- 7,112 pooled rows, AUC 0.565, Spearman 0.060;
- raw pooled top-10 net `-96.73 bps`;
- mean per-tree leaf-occupancy JS drift rises from about 0.086 in the early
  holdout to about 0.205 later; and
- every long and short leaf-signature cluster has negative later-July mean
  economics.

Reverse transfer also fails asymmetrically: the July model's raw top decile is
`+5.29 bps` on May but `-30.54 bps` on June. These are retrospective
reverse-transfer diagnostics, not promotion evidence. They confirm that a
calendar-specific model can find a transient local relationship but that no
recurring profitable July leaf state has yet been demonstrated.

The causal weekly observability diagnostic points to volatility,
trend/range and open-interest geometry as the best state descriptions.
Volatility has perfect week-level ranking in the tiny strict OOS sample, but
that sample contains only five evaluation weeks and one positive week. It
must remain a monitor, not a trade gate.

The next implementation order is therefore:

1. backfill the identical exact-policy labels, costs and frozen causal fields
   far enough to obtain at least 30-50 completed weekly regimes and multiple
   profitable/adverse cycles;
2. train a decomposed execution hurdle with separate clean-favorable
   probability, conditional favorable magnitude, adverse probability and
   conditional adverse magnitude, alongside a direct-net challenger;
3. test recurrence of frozen, outcome-free market-state prototypes and leaf
   path signatures across historical weeks, never raw calendar labels;
4. only if at least two recurring states preserve conditional-economic sign in
   multiple forward blocks, test a small uncertainty-bounded residual
   specialist that shrinks exactly to zero when state support is inadequate;
5. keep the causal 21-day mapping, one pooled global top-k and portfolio
   constraints unchanged for every comparison; and
6. keep timing, MAE, target-price and wait actions in their separate
   fail-closed action layer.

The target architecture remains:

`base -> residual alpha + CatBoost + five auxiliary heads -> decomposed
execution EV hurdle -> causal 21-day EV mapping -> one pooled global top-k ->
portfolio constraints -> separate timing/action layer`.

A regime component, if eventually supported, is only an additive shrunk
residual on execution EV. It is not a hard router and must fall back exactly
to the global model under low support or OOD conditions.

### Decomposed execution hurdle and four-month exact-history extension

The fixed decomposed-EV hurdle is complete. It adds, per side and inside the
same purged temporal OOF contract:

- the canonical soft probability of `h12_u1p5atr` favorable-first;
- a separate adverse-first probability;
- a calibrated three-class timeout/adverse/favorable competing-risk model;
- conditional exact-net regressions for each of those three outcomes;
- binary and competing-risk probability-weighted EV compositions; and
- fixed 50/50 blends with the direct-net residual model.

Every composite is subsequently fit to exact net using train-OOF isotonic
mapping, receives the same causal 21-day correction, and is ranked as one
pooled global top 10%. No probability, blend or model geometry HPO is used.

| Forward block | Arm | Mapped top-10 net | AUC positive net | Spearman |
|---|---|---:|---:|---:|
| May -> June | direct net | **-59.69 bps** | 0.545 | 0.103 |
| May -> June | hurdle probability | -64.30 bps | **0.557** | **0.115** |
| May -> June | competing clean probability | -69.60 bps | 0.532 | 0.085 |
| May -> June | competing decomposed EV | -96.39 bps | 0.529 | 0.083 |
| May -> June | direct/competing 50/50 | -69.43 bps | 0.535 | 0.088 |
| later July | direct net | -106.52 bps | 0.546 | 0.063 |
| later July | hurdle probability | -104.44 bps | 0.566 | 0.098 |
| later July | competing clean probability | **-92.96 bps** | **0.568** | 0.078 |
| later July | competing decomposed EV | -99.56 bps | 0.521 | 0.073 |
| later July | direct/competing 50/50 | -94.70 bps | 0.557 | 0.081 |

Competing-risk probability improves later-July direct net by 13.56 bps, but
loses 9.91 bps in June and remains deeply negative. Conditional magnitude
composition does not improve either block. The result therefore rejects
decomposed-EV HPO on the current May-July training history. Keep the separate
outcome heads as diagnostics/support targets; do not promote their composite
as admission.

Canonical result:
`data_perp/artifacts/exact_policy_decomposed_hurdle_ablation_20260727_v2/`.

Historical reconstruction infrastructure now accepts an explicit inclusive
month range and has a candidate-level coverage-only mode. It writes path and
ATR coverage separately and fails closed if any side-month has less than 99%
complete 720-minute one-minute paths. It does not infer coverage from global
store bounds.

The March-April preflight passes:

| Month | Exact 1m path coverage | Causal ATR coverage | Exact label coverage |
|---|---:|---:|---:|
| March 2025 | 100.000% | 94.926% | 94.926% |
| April 2025 | 99.264% | 93.361% | 92.707% |

Preflight artifact:
`data_perp/artifacts/marapr2025_execution_ev_exact1m_coverage_preflight_20260727_v1/`.

The first bounded January-April exact-one-minute comparator reconstruction is
also complete:

- 910,933 causal candidates;
- 865,552 exact one-minute fee-only comparator labels;
- 818,554 inner base OOF predictions;
- 771,494 strict two-layer direct-EV OOF predictions;
- strict scoring from January 15 through April 30; and
- 99 raw PIT fields with fold-local top-40 selection, side-local weekly
  expanding fits and resolved-label purging.

Month-local pooled global top-10 diagnostics are:

| Month | Mean exact net |
|---|---:|
| January 2025 after warm-up | -58.11 bps |
| February 2025 | +8.42 bps |
| March 2025 | -25.33 bps |
| April 2025 | -62.69 bps |

These month-local rows rerank within each month and are comparator diagnostics,
not the production admission rule. The single across-period global book is
-40.72 bps. Neither result is economically comparable to May-July: the
historical runner uses `simulate_execution_ev_12h`, with fee-only accounting
and side-parent geometry, whereas the canonical current target uses
`simple_policy_optimiser.simulate_and_score`, deployed asset spread-aware
entry/exit fills and side-by-policy-archetype geometry with parent fallback.
The artifact proves exact-one-minute path coverage, causal input availability
and two-layer OOF feasibility only. Its returns must not be used for state
economics, sample weighting, score residuals, promotion or pooling with the
current deployed-policy target.

Comparator result:
`data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_20260727_v1/`.

Do not extend all 19 months in one in-memory run. Continue with bounded
coverage-first cohorts, then materialize an immutable exact-label panel and
consume it lazily for whole-history OOF. Late 2024 remains an hourly-comparator
tier and must never be pooled with exact-one-minute policy evidence.

The first artifact built from those returns,
`data_perp/artifacts/exact_history_state_recurrence_20260727_v1/`, is therefore
invalid for economic recurrence. Its fixed historical state geometry and
occupancy tables may be retained as exploratory outcome-free geometry, but:

- its state economics, recurrence gates and specialist conclusions are void;
- its 26-field basis is not live-aligned enough for a canonical rerun:
  `cumulative_delta_stall`, `dist_ema200_atr` and `leverage_build_score` must
  be excluded from the first historical/live state basis; and
- a canonical rerun must use expanding past-only state fits rather than one
  KMeans fit on all January-April rows.

Before any economic recurrence work resumes, rebuild January-April with the
same policy JSON, policy-archetype normalization, barrier input, causal ATR
input, deployed spread baseline and `simulate_and_score` pathway used by the
current label materializer. Prove input and target parity on overlapping
May-July candidate identities, then regenerate one common side-local expanding
OOF score lineage. Until that gate passes, historical states are monitors only.

### Historical target-parity repair and bounded current-spread replay

The target-lineage repair is now implemented. The previous January-April
fee-only artifacts remain immutable diagnostics and have explicit
`ECONOMIC_INVALIDATION.json` sidecars. The replacement pathway never silently
changes their targets.

`scripts/materialize_historical_execution_ev_policy_inputs.py` builds the
three immutable inputs required by the canonical policy-label materializer. It
fails closed unless all of the following hold:

- exact candidate identity and `decision = signal + 1h`;
- bit-exact agreement between the archived canonical path barrier and its raw
  historical source witness;
- finite positive archived canonical path ATR;
- the policy hash equals the current canonical label artifact;
- the current reference artifact proves 100% side-parent fallback;
- every admitted symbol has an explicit positive p90 value in the frozen
  spread baseline; and
- every side-month clears the configured canonical path-input coverage gate.

The old historical policy-archetype key is deliberately not reused. It has 0%
textual parity with the current base-rank-decile context. This is safe only
because the referenced current label artifact proves that all 156,202 audited
current rows resolve to `long__parent` or `short__parent`; any future active
side-archetype geometry makes the historical adapter fail closed.

The archived canonical path-input ledger begins on February 1, 2025. For the
February-April cohort:

- 688,160 raw causal candidates;
- 510,996 candidates with exact archived canonical barrier/ATR inputs;
- 100% barrier bit parity on the admitted rows;
- 100% exact-one-minute path coverage in February and March;
- 99.350% exact-one-minute path coverage in April;
- 509,868 completed spread-aware 12h labels; and
- 100% side-parent policy-geometry resolution.

The labels use the same `simple_policy_optimiser.simulate_and_score` code,
policy JSON and frozen per-asset p90 spread-baseline hash as the current
May-July artifact. They are therefore a common **current-spread
counterfactual** on historical paths. They are not factual 2025 execution
costs because contemporaneous historical L2/spread observations are
unavailable.

The historical OOF runner now accepts an immutable external canonical-label
panel and independently verifies simulator, policy, spread, horizon and
geometry hashes before fitting. The February-April replacement produces:

- 509,868 exact-path current-spread labels;
- 470,220 side-local weekly expanding base OOF rows;
- 430,464 strict two-layer direct-EV OOF rows from February 15;
- the same fold-local top-40 raw-PIT selection and resolved-label purge; and
- one pooled global top 10% with no side, timestamp or calendar quota.

| Book | Current-spread counterfactual top-10 net |
|---|---:|
| Single pooled February 15-April 30 book | -120.66 bps |
| February month-local diagnostic | -83.28 bps |
| March month-local diagnostic | -116.46 bps |
| April month-local diagnostic | -149.28 bps |

The previous fee-only positive-February result is rejected. It does not survive
the common spread-aware simulator and must not be used as profitable-regime
evidence.

Canonical research artifacts:

- `data_perp/artifacts/febapr2025_execution_ev_deployed_policy_inputs_20260727_v1/`;
- `data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_stage_20260727_v1/`;
- `data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/`;
- `data_perp/artifacts/febapr2025_execution_ev_current_spread_two_layer_oof_20260727_v2/`.

The next May-July 2025 cohort fails the exact-one-minute data gate before
simulation:

| Month | Exact 12h one-minute path coverage after canonical input join |
|---|---:|
| May 2025 | 60.231% |
| June 2025 | 24.590% |
| July 2025 | 24.419% |

Its 342,974 missing candidate windows are staged in
`data_perp/artifacts/mayjul2025_execution_ev_current_spread_12h_stage_20260727_v1/`.
Do not materialize, fit or interpret this cohort until targeted one-minute
backfill raises every side-month to at least 99%. January 2025 is also excluded
from the replacement panel: its archived canonical barrier/ATR ledger does not
exist, and the raw-Wilder ATR reconstruction is not parity-equivalent to the
current auxiliary input. Neither gap may be filled by tolerance matching or
silent approximation.

### Common-universe and outcome-free state diagnosis

The May-July coverage failure is strongly universe-structured. Exactly 30
symbols have 100% complete 12-hour one-minute paths in every side-month. They
are frozen in `configs/historical_exact1m_common_universe_2025_v1.txt`. This is
an outcome-free availability filter, not a selected profitable universe.

Those 30 symbols represent 23.53% of the current May-July exact-policy rows.
The bounded common-universe replay has:

- 132,480 May-July 2025 exact-path current-spread labels with 100% path
  coverage;
- 260,640 February-July labels after joining the February-April history on the
  same frozen universe;
- 230,400 weekly expanding base OOF predictions; and
- 200,160 strict two-layer EV OOF predictions.

The six-month common-universe model remains negative:

| Month-local pooled top 10% | Net EV |
|---|---:|
| March 2025 | -113.05 bps |
| April 2025 | -88.54 bps |
| May 2025 | -108.81 bps |
| June 2025 | -134.13 bps |
| July 2025 | -145.37 bps |
| Single global book | -109.39 bps |

Thus severe deterioration is not unique to July 2026. The same common
side-local model and current-spread cost contract fail across multiple 2025
months, with June-July 2025 also worse. This weakens the hypothesis that a
single novel July-2026 regime is the primary cause.

The current May-July arms were then reranked within the identical 30-symbol
universe, still as one pooled global top 10%:

| Arm | May->June all | May->June common 30 | Later July all | Later July common 30 |
|---|---:|---:|---:|---:|
| direct net | -60.69 | -65.14 | -106.54 | **-64.51** |
| hurdle probability | -63.98 | **-53.38** | -104.24 | **-66.51** |
| competing clean probability | -74.43 | -106.00 | -93.34 | **-65.75** |

The stable universe recovers roughly 38-42 bps in later July for these three
arms, so asset-universe shift is material. It does not solve the economics:
the best later-July common-universe gross return is only about +35 bps against
about 100 bps realized cost. The diagnosis is therefore:

1. asset-universe shift explains part of the July collapse;
2. simple observable state OOD does not explain most of it;
3. opportunity capture remains below the fixed cost hurdle even on the stable
   universe; and
4. a hard July specialist or regime router remains unjustified.

Artifacts:

- `data_perp/artifacts/mayjul2025_execution_ev_common30_labels_20260727_v2/`;
- `data_perp/artifacts/febjul2025_execution_ev_common30_two_layer_oof_20260727_v3/`;
- `data_perp/artifacts/execution_ev_common30_current_universe_ablation_20260727_v1/`.

The replacement outcome-free state diagnostic is also complete:
`scripts/diagnose_historical_state_recurrence_v2.py` uses only the 23 verified
live-aligned fields and per-side expanding past-only weekly
`CausalRegimeStateModel` fits. It excludes every score, target, outcome,
calendar field, weight and policy action.

| Evaluation period | Mean weekly occupancy JS drift | Mean OOD above training p99 |
|---|---:|---:|
| June 2026 | 0.00071 | 0.044% |
| July 2026 | 0.00482 | 1.83% |

July is observably more shifted, with a 2.94% OOD peak in the partial July 14
long block, but remains mostly inside prior observable support. The selected
three-state geometry is highly concentrated within individual weekly blocks
and has near-zero posterior entropy. Treat it as a coarse state monitor, not a
tradable regime taxonomy.

Canonical state-observability artifact:
`data_perp/artifacts/historical_state_recurrence_observability_20260727_v2/`.

### Common-lineage long-horizon transfer result

The frozen 30-symbol current-spread panel is now extended through every fully
covered cohort from February-November 2025 and joined directly to the current
April-July 2026 exact-policy labels. December 2025-March 2026 remains an
explicit zero-label gap; the runner neither trains on partial months nor
creates synthetic labels.

All five label panels independently pass the same simulator, policy hash,
spread-baseline hash, 12h horizon and universal side-parent-fallback checks.
The source reader filters the frozen universe at parquet scan time, avoiding
an unsafe all-history in-memory load. The final common lineage contains:

- 759,209 raw source candidates in the frozen universe;
- 473,068 exact current-spread labels;
- 442,828 expanding weekly base OOF rows; and
- 412,588 strict two-layer direct-EV OOF rows.

The month-local one-global-top-10 results are:

| Month | Net EV |
|---|---:|
| March 2025 | -113.05 bps |
| April 2025 | -88.54 bps |
| May 2025 | -108.81 bps |
| June 2025 | -134.13 bps |
| July 2025 | -145.37 bps |
| August 2025 | -124.55 bps |
| September 2025 | -120.71 bps |
| October 2025 | -127.21 bps |
| November 2025 | -80.49 bps |
| May 2026 | -150.11 bps |
| June 2026 | -103.93 bps |
| July 2026 through July 17 | -127.30 bps |

The single across-panel global top-10 book is -111.88 bps. More importantly,
none of its 50 selected weekly slices is positive. The best week,
November 17-23 2025, is still -1.34 bps.

This changes the diagnosis:

1. the model does not merely fail to transfer into or out of July;
2. under one common target, feature, OOF, universe and cost contract, it fails
   to clear costs in every tested month and week;
3. July 2026 is worse than June 2026, but is one instance of a broader
   learnability/capture/cost failure; and
4. observable OOD and asset-universe expansion amplify the problem but are not
   its root cause.

The prerequisite for economic state recurrence fails: at least three
independent profitable weekly episodes were required, and there are zero.
Within-state economic mining, state-specialist HPO and hard regime routing are
therefore not authorized. Mining states after observing 50 uniformly negative
books would be retrospective selection, not evidence of recurrence.

Canonical transfer artifact:
`data_perp/artifacts/feb2025_jul2026_execution_ev_common30_transfer_oof_20260727_v4/`.
Its `STATE_SPECIALIST_GATE.json` records the fail-closed decision.

The next ablations should target the actual bottleneck:

1. predict gross opportunity, expected capture and cost separately, with an
   explicit fail-closed admission rule when predicted gross capture does not
   exceed predicted cost plus uncertainty margin;
2. ablate the stable 30-symbol universe, current inference universe, and
   spread/liquidity buckets without allowing universe-specific top-k quotas;
3. train the clean favorable/adverse/timeout probabilities as supporting
   tasks, but retain direct exact net as the primary common-unit score;
4. test whether restricting training to assets eligible under the inference
   spread filter improves the full current book, as a causal universe
   definition rather than a hindsight profitability filter;
5. diagnose why selected gross capture stays near +35 bps while round-trip
   cost is about 100 bps, including entry/exit action-layer opportunity and
   false-positive clean-event admission; and
6. revisit state residuals only after a globally positive baseline exists in
   at least three independent weeks.

### Corrected inference spread-universe replay

The spread-exclusion fields in
`base_residual_label_ablation_20260725_v2` are invalidated only for that
diagnostic by `SPREAD_EXCLUSION_DIAGNOSTIC_INVALIDATION.json`. The original
comparison matched zero rows because candidate symbols used underscores while
the exclusion set used slashes. Its label-HPO and non-spread results remain
unchanged.

`scripts/ablate_execution_ev_inference_spread_universe.py` repairs the
comparison with the exact `universe.py` normalization and
`average_spread_bps > 70` blacklist. It compares the unrestricted pooled
global top 10%, the eligible slice of that original book, and a globally
reranked eligible top 10%.

All 155 symbols in the current mapped cohort are baseline-covered and
eligible; zero rows are excluded. Consequently the three books are identical:
direct net is -60.69/-106.54 bps, hurdle probability is -63.98/-104.24 bps,
and competing clean probability is -74.43/-93.34 bps for May-June/later July.
This is not evidence that filtering is ineffective: it proves that the
available candidate ledger was already filtered upstream. A causal training
exclusion test requires a pre-universe ledger and a point-in-time eligibility
snapshot. The June-July 2026 baseline is non-PIT for historical decisions and
cannot support promotion.

Artifact:
`data_perp/artifacts/execution_ev_inference_spread_universe_ablation_20260727_v2/`.

### Fail-closed variable admission on the existing OOS scores

`scripts/evaluate_execution_ev_variable_admission.py` now evaluates the
existing strict-OOF/forward hurdle predictions without fitting or tuning
another model. For each window and arm it preserves one pooled global 10%
capacity, then applies fixed mapped predicted-net floors of 0, 25 and 50 bps.
The rule may select fewer than capacity, including zero; there are no
timestamp, side or asset quotas.

The result is diagnostically decisive:

| Window | Arm | Rule | Rows | Realized net |
|---|---|---|---:|---:|
| May-June control | direct net | forced global top 10% | 4,925 | -60.69 bps |
| May-June control | direct net | predicted net > 0 | 1,544 | -94.53 bps |
| May-June control | hurdle capture guard | predicted net > 50 bps | 47 | +78.95 bps |
| Later July | direct net | predicted net > 0 | 0 | no trade |
| Later July | hurdle probability | predicted net > 0 | 0 | no trade |
| Later July | hurdle EV | predicted net > 0 | 0 | no trade |
| Later July | hurdle capture guard | predicted net > 0 | 2 | -190.27 bps |

The 47-row May-June result is a fixed-threshold diagnostic, not a winner: it
does not recur in later July and must not be threshold-selected after seeing
these outcomes. The appropriate behavior of the current mapped scores in
later July is therefore abstention, not a mechanically filled top-k book.
No arm qualifies for portfolio replay or promotion.

Artifact:
`data_perp/artifacts/execution_ev_variable_admission_20260727_v1/`.

### Ex-post opportunity/capture ceiling under the frozen exit policy

`scripts/diagnose_execution_ev_oracle_ceiling.py` now answers the prior
question before further model HPO: whether the exact 12-hour labels contain a
large enough *realized* opportunity tail to clear the frozen policy cost.  It
is intentionally non-tradable: it ranks candidates by their future realized
`execution_net_ev_12h`, so it is neither a backtest nor an admissible score,
threshold, HPO objective or promotion metric.

The runner fails closed before loading outcomes unless the canonical label
hashes, the 720-minute horizon, `simple_policy_optimiser.simulate_and_score`,
the frozen policy hash, current-spread lineage, gross-cost-net reconciliation
and universal side-parent fallback all match.  It separately verifies the five
source panels behind the common-30 transfer ledger.  The inference-universe
slice uses the exact `universe.py` symbol normalization and the frozen
70-bps-average-spread blacklist; a missing baseline is ineligible.

Two independent panels passed those checks:

| Panel | Exact rows | Realized net > 0 | >25 bps | >50 bps | One global oracle top 10 net |
|---|---:|---:|---:|---:|---:|
| Current May 1-Jul 10 2026 all-universe panel | 127,777 | 34.00% | 27.63% | 22.54% | **+294.50 bps** |
| Common-30 Feb 2025-Jul 17 2026 transfer ledger | 473,068 | 32.28% | 25.71% | 20.56% | **+266.65 bps** |

The corresponding oracle top-5/top-2 values are +405.08/+572.79 bps in the
current panel and +365.58/+509.85 bps in the common lineage.  At variable
admission, the realized-net-positive sets average +127.43 and +118.34 bps per
trade respectively.  This is an upper ceiling only: the future outcomes make
the selection impossible at decision time.

The ceiling persists across time rather than being an isolated month.  Each of
the 11 current and 56 common-lineage weeks has a positive *week-local oracle*
top-10; their weakest mean net values are +166.47 and +81.89 bps.  Current
month-local oracle top-10 is +270.52 bps in May, +346.19 in June and +196.21
in the partial July.  Candidate-level net-positive opportunity is lower but
material in every current month: 30.01%, 40.43% and 29.77% respectively.

All rows in both diagnostic panels are currently inference-eligible under the
frozen average-spread baseline, with zero missing baseline mappings.  The
candidate panels occupy only the <=10, 10-25 and 25-70 bps buckets.  Thus the
static spread blacklist cannot raise this particular label-level ceiling;
universe filtering must still be tested against the *predicted* global book,
not treated as an economic solution.

This changes the next learning question precisely.  There is substantial
ex-post gross opportunity after the approximately 100-bps policy cost, while
the actual OOS selected books remain negative.  The priority is therefore
recovering and calibrating the favorable-event/capture tail (and rejecting
false positives), rather than assuming the frozen exit policy has no
opportunity.  The oracle does **not** prove that this tail is predictable.

Artifact:
`data_perp/artifacts/execution_ev_oracle_opportunity_ceiling_20260727_v1/`.

### Gross opportunity less exact known cost: fixed ablation rejected

The prior decomposed-hurdle arms did **not** test a distinct model of
`execution_gross_ev_12h - execution_cost_return`: `direct_net`, the positive
magnitude head and the three outcome heads all learn net return, while gross
was only an accounting diagnostic. The narrow v3 extension in
`scripts/run_exact_policy_capture_hurdle_ablation.py` adds two fixed per-side
heads without changing labels, features, geometry, mapping or pooled-global
top-10 ranking:

1. `direct_gross_minus_exact_cost`: residual CatBoost prediction of exact gross
   return, then minus the row known `execution_cost_return` exactly once;
2. `capture_gross_mixture_minus_exact_cost`:
   `P(opportunity) × P(positive capture | opportunity) × E[gross | positive capture]`
   minus that same row cost.

Gross includes executable spread drag; the known deterministic fee is neither
learnt as a target nor subtracted twice. The existing direct-net and net
decomposition arms remain unchanged controls. Raw head OOF is temporal and
side-local; the unchanged mapping is fit only before each forward evaluation
block. There is no HPO, blend or threshold search. Four focused contract tests
pass.

| Forward block | Direct net control | Direct gross − exact cost | Capture-gross mixture − exact cost |
|---|---:|---:|---:|
| May → June | **-60.69 bps** | -103.14 bps | -130.88 bps |
| Later July | **-106.54 bps** | -115.14 bps | -105.34 bps |

The gross-first regression loses 42.46 bps against direct net in June and
8.60 bps in later July. The capture-gross mixture gains only 1.20 bps in July
and loses 70.19 bps in June. Its un-mapped June score selects +30.71 bps, but
the mandated train-derived causal mapping selects -130.88 bps; that scale
does not survive calibration and is not usable as an admission score. Neither
arm clears the latest-fold or economics gate, so do not HPO, blend or replay
either one. The missing gross-minus-known-cost hypothesis is now directly
tested and rejected on the current two forward blocks.

Artifact:
`data_perp/artifacts/exact_policy_decomposed_hurdle_ablation_20260727_v3/`.

The fixed fail-closed floors were also applied to these gross-first arms in
`execution_ev_variable_admission_20260727_v2`. Both admit zero later-July
rows above predicted net zero. Their few positive-score May-June rows are
severely miscalibrated and negative. `PORTFOLIO_REPLAY_GATE.json` therefore
records an explicit reject decision: there is no qualifying arm to send
through simple-policy or constrained-portfolio replay.

### OOS oracle-tail recovery: mapped scores do not retrieve enough realized opportunity

`scripts/diagnose_execution_ev_oracle_recovery.py` measures the recovery gap
using persisted, causally mapped OOS scores from the capture-support and
decomposed-hurdle runs against the exact canonical 1-minute policy target. It
is diagnostic only: future net is revealed only after score ranking, and is
never used for action selection, HPO, threshold choice or promotion.

The runner fails closed on canonical target/accounting hash, prediction hashes,
source canonical-input hashes, finite scores, arm identity uniqueness and equal
OOS coverage. It covers 30 arms on 56,315 candidates: 49,244 in the May-June
control and 7,071 in later July. Each top-k is one pooled global book within
its independent forward block, with no timestamp, asset, side or week quota.
Side/week/latest-week figures are breakdowns of that unchanged book. The latest
later-July week (2026-07-13--19) has 5,929 covered candidates.

For global top-10/top-5/top-2, the artifact reports exact-net event precision,
recall and lift at `>0`, `>25`, `>50` and `>100` bps; false-positive rate and
shortfall severity; missed-winner economics; and same-k future-oracle
overlap/recall/Jaccard. The top-10 headline is decisive:

| Forward block | Best selected exact-net arm | Net | Best same-k future-oracle recall | Best `net > 0` recall | False-positive rate at that recovery best |
|---|---|---:|---:|---:|---:|
| May-June control | severe high-20 veto | -59.57 bps | 16.04% | hurdle probability: 13.26% | 46.42% |
| Later July | direct + capture residual | -71.23 bps | capture-low20 abstain: 17.66% | capture-low20 abstain: 15.70% | 64.12% |

For `net > 100 bps`, the best top-10 recall is only 14.64% in May-June and
19.29% in later July, while 73.22% and 86.86% of the selected book respectively
remain false positives for that event. Even the best top-5 book is -56.01 bps
in the control and -52.50 bps in later July; the best top-2 book is -40.21 and
-62.32 bps. Reducing k alone therefore does not solve the recovery problem. No
arm is eligible for action-layer work, replay or promotion.

Next ablation priority: improve favorable-event/capture discrimination and
false-positive rejection against these fixed recovery metrics before trying
more score blends or regime specialists.

Artifact:
`data_perp/artifacts/execution_ev_oracle_tail_recovery_20260727_v1/`.

### Frozen June false-positive versus missed-winner feature diagnostic

`scripts/diagnose_execution_ev_false_positive_features.py` performs a bounded,
no-fit diagnosis on the existing current mapped `direct_net` and
`capture_only` scores. It joins them to the canonical live-aligned meta input
and exact current-policy target, then applies one pooled global direct top-10%
book. The exact 50-bps surplus boundary is used only after the decision to
form an exhaustive diagnostic partition: selected true positives, selected
false positives, missed high-surplus winners and true negatives. It is never
an input, score, action or selected trading threshold.

May-to-June forward OOS is the sole discovery/control panel. Feature signs,
robust centring/scales and retained-feature medians are frozen before later
July is read. The inventory is restricted to the current canonical numeric
live meta inputs plus contemporaneous direct/capture score context (including
timestamp-local ranks and score disagreement). It excludes exact outcomes,
support-label materializations, exit/action/portfolio fields and calendar
shortcuts. There is no model fit or HPO.

A field had to have at least 100 pooled true/false selected control rows, at
least 20 in every side/class cell, 95% selected-book coverage, a pooled
true-positive-minus-false-positive standardized effect of at least 0.12, and
matching side signs. At most one field per family and four families could be
frozen. Four passed the *control* contract:

| Frozen June field | Family | Control effect | Later-July effect | July net lift vs screen complement |
|---|---|---:|---:|---:|
| `catboost_p_2` | CatBoost geometry | +0.473 | +0.476 | -8.10 bps |
| direct timestamp rank | Candidate-score context | +0.213 | -0.013 | -35.64 bps |
| `base_margin_to_cutoff_z` | Base candidate context | +0.185 | +0.265 | **+54.70 bps** |
| `alpha_prediction_uncertainty` | Alpha confidence/support | +0.171 | -0.158 | -12.83 bps |

The equal-weight four-field composite preserves class separation in July
(+0.431) and high-surplus lift, but loses 3.45 bps versus its complement. It
is not economically stable. `catboost_p_2` also preserves the 50-bps class
contrast but loses 8.10 bps in July; class discrimination is therefore not a
sufficient economic gate. Only `base_margin_to_cutoff_z` retains the sign,
high-surplus lift, and exact-net lift in both windows (+35.69 bps control,
+54.70 bps later July). This is a locked diagnostic candidate only, not a
promotion or an authorization to add a hard gate: the June global selected
book is extremely short-skewed (55 long selections, with only 32/23 long
true/false positives), so independent longer and side-balanced validation is
required.

The more important structural result is the collapse in direct-book precision:
June has 1,844/4,925 50-bps true positives (37.44%), while later July has only
81/708 (11.44%). Missed high-surplus winners remain numerous (11,860 June;
834 later July). The next predeclared use is a narrow, side-aware ablation of
the already-live `base_margin_to_cutoff_z` as an *input* or soft interaction in
the capture/event model, with its June-derived transformation frozen and a new
untouched later period; do not apply it as a post-hoc trading filter.

Artifact:
`data_perp/artifacts/execution_ev_false_positive_feature_diagnosis_20260727_v2/`.

### Frozen base-margin screen: independent common-lineage recurrence

`scripts/diagnose_base_margin_historical_transfer.py` applies the already
frozen `base_margin_to_cutoff_z >= 0.593431` direction and threshold unchanged
to the independent common-30 strict two-layer OOF lineage. It does not fit,
retune or select a threshold on historical outcomes. The primary assessment
is one pooled global top 10% across the full lineage; month-local global books
are recurrence diagnostics, never timestamp or side quotas. Because the
screen was chosen on the current panel and assessed backward in time, this is
reverse-transfer evidence, not chronological promotion evidence.

The frozen screen improves the full-lineage selected-book complement by only
+9.96 bps (-103.94 versus -113.90 bps). Its sign is positive in 9 of 12
month diagnostics, including June and July 2026 (+38.52 and +28.17 bps), but
reverses in July and November 2025 and May 2026. Every retained month remains
negative; the best retained book is June 2026 at -75.97 bps.

Decision: the base margin has recurring false-positive information and is
eligible for a predeclared soft interaction/rejector input. It is not a hard
gate, a standalone score, or promotion evidence. Because the field is already
present in the current core model, the next test must alter how it interacts
with capture probability or score confidence and must be frozen before a
genuinely new forward block.

Artifact:
`data_perp/artifacts/base_margin_historical_transfer_20260727_v1/`.

### Frozen soft base-margin × capture-confidence challenger (reused-OOS only)

The narrow follow-up is now specified in
`scripts/run_exact_policy_capture_support_ablation.py` and scored without a
new fit by `scripts/score_frozen_base_margin_capture_interaction.py`.  It uses
the immutable June screen (`base_margin_to_cutoff_z`, positive direction,
threshold **0.5934305191**, robust scale **0.7888193689**) and a fixed small
weight of **0.25**.  Its formula is:

```text
z_direct + 0.25 × (2 × sigmoid((margin_z − 0.5934305191) / 0.7888193689) − 1)
         × max(0, 0.5 × (z_direct + z_capture))
```

`z_direct` and `z_capture` are side-local temporal-OOF standardizations in the
full runner.  The `max(0, ...)` term is important: a low margin can softly
discount a jointly confident candidate, but cannot turn two low-confidence
head outputs into a reward.  There is no evaluation HPO, refit, hard margin
gate, timestamp quota, or side quota.  The integrated runner maps the arm with
the existing per-side OOF isotonic mapping followed by the causal recent-EV
correction and then selects one pooled global top 10%.

For a bounded current check, the diagnostic scorer reuses the byte-verified
v8 strict side-local OOF capture heads and its already-causal direct-EV map;
it applies only the frozen rank perturbation.  It fails closed if the v8
manifest, head predictions, direct predictions, joined source input, feature
lineage, or frozen screen hashes differ.  This is explicitly not a new OOS
test: June and later July have already been used in diagnosis, and the
diagnostic's block-wide confidence standardization must not be used as live
calibration.

| Reused global top-10 book | Direct mapped EV | Frozen soft interaction | Difference |
|---|---:|---:|---:|
| May→June control | -60.69 bps | -60.18 bps | +0.51 bps |
| Later July | -106.54 bps | -105.32 bps | +1.22 bps |
| Both reused blocks pooled | -65.30 bps | -63.36 bps | +1.95 bps |

The small positive diagnostic movement is neither economic recovery nor
promotion evidence: every book remains negative and it is much smaller than
the known cost/capture deficit.  The **next genuinely new forward block** is
the only decision-evidence block.  Before it is opened, freeze a successor
source contract (new input and feature hashes) and run the fully integrated
strict side-local temporal-OOF arm exactly once; do not alter this formula or
its constants after seeing that block.

Artifact:
`data_perp/artifacts/frozen_base_margin_capture_interaction_diagnostic_20260727_v1/`.

The artifact's `NEXT_GENUINE_FORWARD_GATE.json` freezes the formula, constants,
source hashes and evaluation contract. The current canonical joined input ends
at decision time `2026-07-19 16:00 UTC`; no later exact-policy decision cohort
is presently available in this lineage. Only candidates strictly after that
timestamp may open the gate.

## 2026-07-28 successor forward lock and compact capture closure

Status: **complete inference infrastructure; waiting for genuinely future
point-in-time candidates, not for another research refit**.

### Authoritative successor source lock

The first forward lock was incomplete because it did not authenticate the
new Pack-B base/residual future scorer or the supporting-head pre-entry
materializer. Its `contract.json` remains byte-for-byte unchanged and a
separate `SUPERSEDED.json` records the reason and successor.

The authoritative v5 lock passes the source-lock audit with no blockers. All
earlier contracts remain byte-unchanged but are explicitly superseded: v2
contradicted zero-trade abstention, v3 retained an ambiguous window-completion
label, and v4 did not yet bind the coverage/scored schema or mandatory seal
semantics into the lock fingerprint:

- path:
  `data_perp/artifacts/execution_ev_forward_source_lock_20260728_v5/contract.json`;
- contract SHA-256:
  `a2ca0f8fe443ec8ac9935f20972e4ee3c28877f6919d6d515ae00a0bdea1c647`;
- fingerprint:
  `5d9068ca1ee95526cb17e31fb0bb5bbd017087d75222fb450d1f6c24e2fec460`;
- first eligible decision:
  strictly after `2026-07-27 23:59:59.999999 UTC`;
- requested last decision:
  `2026-08-10 23:59:59.999999 UTC`;
- exact label horizon: 12 hours;
- fixed-window coverage gate: at least 5,000 scored rows, 500 members of the
  single pooled global top-10% capacity, and 14 complete UTC days.

The lock pins the final per-side base and residual scorers, clean-event/Peak/
path pre-entry materializer, final direct and capture heads, training-lineage
proof, causal calibrator seed, frozen policy/spread inputs, policy-label and
canonical-join code, population scorer, and readiness auditor. Admission is
one pooled global top 10% after causal recent side-isotonic mapping. There is
no timestamp, side or asset quota, and zero trades are permitted.

The v5 pre-outcome audit has one root blocker represented by two required
artifacts: `missing_file:scored_population` and its mandatory
`missing_file:preoutcome_seal`. This is intentional. Current live-stage
feature matrices end no later than July 20 and cannot enter a block whose
source lock begins after July 27. No OOF or earlier live matrix may be
substituted.

Capacity and economic admission are now separate. The auditor reconstructs
top-10 membership from mapped scores with candidate-ID tie-breaking and then
checks the stored flags. It separately verifies
`capacity AND mapped EV > 0`. A block with 500 capacity rows and zero
positive-score trades is ready for evaluation as a valid abstention but cannot
be promoted as a profitable strategy.

`scripts/run_execution_ev_forward_preoutcome.py` is the only authorized
publisher. It authenticates the v5 lock, binds candidates to hashed raw-source
coverage, requires complete resolved-only exact-policy update coverage, rejects
unknown/duplicate identities and score mismatches, runs every frozen scorer,
and atomically publishes only a readiness-passing population. Its seal binds
candidate identities, intermediate/output hashes, coverage and update
provenance, and the lock fingerprint.

The v5 fingerprint now also binds the 12-hour horizon, fixed-window coverage
thresholds, identity/availability contract, capacity/admission fields and
daily-coverage schema. The readiness auditor rejects any pre-outcome table
containing execution outcomes or label/target fields and independently
recomputes the seal fingerprint, identity hash, output hashes, global capacity
and economic admission.

### Expanded exact-policy event/capture matrix

`scripts/run_exact_policy_capture_hurdle_ablation.py` is now v4. It adds the
two genuinely missing arms to the existing per-side, fixed-CatBoost,
temporal-OOF comparison:

1. an ATR-soft mutually exclusive target distribution. The favorable mass is
   the canonical `h12_u1p5atr` soft label; non-favorable mass remains adverse
   on adverse-first rows and timeout otherwise. Three train-only calibrated
   regressors predict timeout/adverse/favorable mass, which is normalized to
   a probability simplex;
2. an explicit cost-aware score:

```text
P(opportunity) × P(capture | opportunity) × E[positive net | capture]
− P(adverse first) × E[loss | adverse first].
```

All arms use identical rows, side-local model fitting, exact deployed-policy
gross/cost/net outcomes, causal 21-day mapping, and one pooled global top-10%
book. There is no HPO, evaluation threshold tuning, side quota or
timestamp-local selection.

| Causal mapped global top-10 arm | May → June | Later July | Verdict |
|---|---:|---:|---|
| Direct net control | -59.69 bps | -106.52 bps | Frozen comparator |
| Hurdle probability | -64.30 bps | -104.44 bps | Reject |
| Competing clean probability | -69.60 bps | **-92.96 bps** | Latest-period improvement does not retain control |
| ATR-soft favorable probability | -76.48 bps | -151.14 bps | Reject |
| Competing decomposed EV | -96.39 bps | -99.56 bps | Reject |
| ATR-soft decomposed EV | -91.89 bps | -102.32 bps | Reject |
| Capture upside minus adverse loss | -109.82 bps | -104.95 bps | Reject |

The latest-period least-bad event head is still negative, and every arm fails
the required positive economics plus control/latest recurrence gate. None is
eligible for the action layer, portfolio replay or the frozen forward
challenger.

Artifact:
`data_perp/artifacts/exact_policy_capture_hurdle_ablation_20260728_v4/`.

### Identical-row long ranking decomposition

`scripts/diagnose_execution_ev_long_ranking.py` freezes the 114,096 exact
candidate IDs from the canonical economic-failure panel and compares base
rank, raw execution EV, causal global 21-day EV, and causal side 21-day EV.
Base rank is explicitly lower-is-better; all EV scores are
higher-is-better. Gross minus exact row cost equals net before any ranking.

The primary selection remains one pooled global top 10% across timestamps and
sides. Long results below are contributions to that unchanged book, not a
long quota. A separate long-only top-decile is diagnostic only.

| Score | Whole pooled book | Long selected | Long contribution | Long-only diagnostic |
|---|---:|---:|---:|---:|
| Base rank | +1.76 bps | 5,956 | -5.53 bps | -8.24 bps |
| Raw execution EV | -32.40 bps | 9,774 | -43.24 bps | -6.25 bps |
| Causal global 21d EV | +3.12 bps | 6,579 | +8.50 bps | +9.36 bps |
| Causal side 21d EV | -8.91 bps | 6,246 | +9.55 bps | +15.31 bps |

Raw execution EV replaces almost the entire base-selected book: its Jaccard
overlap with base rank is only 0.069 and it admits 9,774 long rows into the
11,410-row pooled book. This is the clearest current evidence that the raw
execution layer damages long allocation. Causal mapping repairs long ordering
on this reused May-July panel, but the whole-book result changes sign between
global and side mapping. The positive long slices are diagnostics, not fresh
promotion evidence.

The bounded exit-geometry evidence remains closed. Side-only geometry improves
the later-fold parent by only 0.67 bps, the best wider fixed arm by 8.18 bps,
and even the per-row hindsight best nearby policy remains -49.78 bps net.
Another exit HPO cannot repair the upstream capture and allocation deficit.

Artifact:
`data_perp/artifacts/execution_ev_identical_row_long_ranking_20260728_v1/`.

### Next executable action

Do not refit or modify the locked interaction. Once point-in-time inputs after
the lock boundary exist:

1. materialize the complete hourly candidate feature matrix;
2. score base/residual and supporting heads with the locked final refits;
3. score direct/capture and roll the 21-day calibrator only after each 12-hour
   outcome resolves;
4. require 14 complete days, 500 admitted rows, exact schema parity and exact
   one-minute paths;
5. open the forward outcomes once;
6. send an arm to timing and portfolio replay only if exact net EV is positive
   in aggregate, latest complete fold, both sides with adequate coverage, and
   the unconstrained pooled global book.

### Failure-first transition and destination infrastructure

`scripts/run_failure_first_regime_pipeline.py` now implements the shared
failure-first design end to end and fails closed before unsupported model
fitting. Its source contract requires the explicit strict-OOF mapped-score
flag, finite exact gross/net outcomes, exact outcome availability, unique
candidate identity and a model/evaluation origin. The 21-day global admission
frontier and the resolved residual reference both reset across an origin
boundary.

The runner publishes:

- `decision_health_6h.parquet`;
- `candidate_membership_expost.parquet`;
- `failure_episodes.parquet`;
- `episode_row_membership.parquet`;
- `hourly_observable_state.parquet`;
- `episode_window_state.parquet`;
- `episode_window_outcomes.parquet`;
- `failure_episode_profiles_expost.parquet`;
- `excluded_incomplete_episodes.parquet`;
- source, content-hash, forward-coverage and sufficiency manifests.

The observable state has 32 current-time fields: 20 market fields and 12 model
health fields. The failure profile is capped at 40 onset/change descriptors.
The supervised contract is capped at 40 fields including three causal BOCPD
features. BOCPD requires one pre-aggregated row per origin/hour, so candidate
row ordering within an hour cannot leak into its change probability.

When the support gate passes, the same runner freezes GMM and KMeans
taxonomies separately on failures resolved before the first detector origin.
The primary default is GMM with five to eight states and at least five episodes
per state. It then constructs labels for transition within three hours, active
transition, current state and destination state. A single CatBoost classifier
family fits all four heads in expanding, label-availability-purged OOF folds.
Destination failure probability sums an explicit persisted failure-state
vocabulary; it does not assume that every non-`stable` string is automatically
a failure. Constant heads invalidate the detector rather than becoming a
promotion candidate.

The economic report compares the mapped control with the predeclared
failure-trust score:

`mapped_EV - P(failure destination within 3h) * abs(mapped_EV)`.

Both are evaluated as one global top 10% across timestamps and sides, in
aggregate and in the latest complete month. This diagnostic does not alter the
separate timing/action layer.

The authoritative v6 run uses 121,208 strict model-OOS rows from May 7 through
July 19: 114,096 outer-OOF rows plus 7,112 previously opened and resolved
frozen-forward rows retired into later-detector training history. Their
original provenance flags and separate evaluation origins are preserved.
Retired rows are forbidden from evaluating any detector fitted on this
history. The run produces three primary or
catastrophic failure bins and two grouped episodes:

| Episode onset | Trigger bins | Severity | Admitted rows | Net sum | Residual sum | Complete nine-window state |
|---|---:|---|---:|---:|---:|---|
| 2026-06-08 06:00 UTC | 1 | catastrophic | 66 | -1.9857 | -2.3664 | no |
| 2026-06-11 00:00 UTC | 2 | primary failure | 116 | -2.2168 | -2.7998 | yes |

The declared minimum is 40 resolved and complete episodes, at least 40 failure
bins, 180 calendar-span days, 180 actually observed UTC days, no six-hour-bin
gap longer than 21 days, plus detector class-support gates. The extended
current panel has 74 observed UTC days, a maximum gap of 1.25 days and only two
episodes over a 73.5-day span. It is insufficient. Accordingly:

- taxonomy status: `SKIPPED_INSUFFICIENT_SUPPORT`;
- detector status: `SKIPPED_INSUFFICIENT_SUPPORT`;
- taxonomy/detector model files written: zero.

Retired forward raw-H0 joinability is exact at 7,112/7,112 rows. Earlier OOF
raw-H0 coverage is only 52,631/114,096 candidate rows and begins on June 8,
which makes the first episode's -48-hour window incomplete.

Artifact:
`data_perp/artifacts/failure_first_regime_pipeline_20260726_v6/`.

Focused verification:

`python3 -m pytest -q
tests/test_audit_failure_first_current_extension_readiness.py
tests/test_failure_first_binary.py
tests/test_failure_first_hourly.py
tests/test_failure_first_detector.py tests/test_failure_first_health.py
tests/test_failure_first_pipeline.py
tests/test_materialize_failure_first_current_history.py`

passes 32 tests, including the direct binary challenger, explicit forward
provenance evaluation and multiclass report
coverage. Remaining work is data, not permission to weaken the contract:
extend the continuous point-in-time raw-H0 market state and the same
current-model mapped-score/exact-policy ledger until the frozen taxonomy and
chronological detector gates can be evaluated legitimately.

### Current-history retirement and extension readiness

`materialize_failure_first_current_history.py` binds the already-opened
July 11-19 forward evaluation report to its exact source hash, preserves the
original OOF/forward flags, and adds
`failure_first_score_is_strict_model_oos` only after all scores, exact 12h
gross/net outcomes and label timestamps are finite. It materializes:

- 114,096 outer-OOF rows;
- 7,112 retired resolved forward-OOS rows;
- 121,208 total strict model-OOS rows through July 19 16:00 UTC.

The runner now accepts an explicit score-valid flag. Default behavior remains
strict OOF. A combined flag must come from a separate materialized provenance
contract; the runner records whether retired forward history is included.

`audit_failure_first_current_extension_readiness.py` checks the current gate
and every available later-July source. No legal extension exists after July 19
16:00 UTC. Exact remaining minimum deficits are:

| Requirement | Remaining |
|---|---:|
| Observed UTC days | 106 |
| Failure episodes | 38 |
| Complete fixed-window episodes | 39 |
| Failure bins | 37 |

The July-20 base/residual/alpha and exact-policy artifacts stop at the same
current cutoff or lack mapped execution EV. The later raw label tail contains
1,088 rows but no current-model score and only 16 complete one-minute paths.
Legacy Pack-B/V9 and hybrid artifacts lack immutable candidate IDs, have poor
tuple/raw-H0 overlap and use different model, mapping or 8h policy contracts.
None may be appended.

Artifacts:

- `data_perp/artifacts/failure_first_current_strict_model_oos_history_20260726_v1/`;
- `data_perp/artifacts/failure_first_current_extension_readiness_20260726_v1/`.

### Failure-first historical comparator and transfer result

The historical comparator is explicitly separated from current-model OOF. Its
v3 materialization contains 440,560 common-30 strict two-layer OOF rows from
January 15 through November 30, 2025, 307 actually observed UTC days and a
maximum 14-day gap at the March generation boundary. The refreshed continuity
gate therefore passes its 180-observed-day and 21-day-maximum-gap requirements.
It uses exact one-minute current-policy gross/net/cost outcomes, causal
side-local 21-day mapping reset by generation and 28 canonical raw-H0 fields.

The conservative 40-episode gate still fails: there are 51 severe bins and 35
complete episodes. A predeclared lower-bound 30-episode research run was
allowed only as a diagnostic. Five-to-eight GMM/KMeans states are unsupported;
four- and five-state variants create singleton/outlier clusters. Robust
clipping plus a two-state diagonal GMM is the only supported taxonomy, with 22
training episodes in a correlation-fragmentation family and eight in a mixed
observable/model-state family. KMeans still fails the five-episode
minimum-cluster-support gate.

The resulting four-head CatBoost detector is not promotable:

- transition-within-three-hours AUC: 0.420;
- active-transition AUC: 0.513;
- latest OOF transition/active positives: 27/10, below the required 50;
- mapped pooled-global top-10: -108.70 bps;
- destination-risk adjusted pooled-global top-10: -108.89 bps.

Artifact:
`data_perp/artifacts/failure_first_regime_pipeline_historical_20260726_v12/`.

Minimum cluster size does not imply taxonomy stability. A frozen 100-repeat
bootstrap audit gives median adjusted Rand index 0.113, q10 -0.009 and median
minimum resampled cluster size two, versus required ARI thresholds 0.80/0.50.
The two broad families are therefore unstable descriptive summaries, not
reproducible routing states. Taxonomy stability is now an explicit mandatory
promotion criterion.

### Superseded historical model-health/failure v1 report -- do not use

The following v1 results are retained only as an audit trail. They used
12 observed rows as a proxy for 12 elapsed hours and averaged hourly means
without candidate-count weighting. Small source-hour gaps therefore violated
the stated before/after horizon, and variable candidate counts made the label
less economically faithful. Artifacts and conclusions in this superseded
section must not be used. The exact-hour, candidate-weighted v3 result below
is authoritative.

The older exact-lineage support experiment is now materialized at
`data_perp/artifacts/historical_exact_model_health_failure_20260729_v1/`.
This artifact is deliberately separate from the May--July 2026
current-execution-EV lineage and declares `current_lineage=false`; it must
never be substituted for current-lineage health.

Its contract is:

- canonical February--April 2025 raw-alpha/exact-policy lineage;
- one pooled global top 10% selected across both sides and all timestamps,
  with deterministic candidate-ID ties;
- 2,113 hourly rows and 50,444 selected candidates;
- 28 compact model-health features, using decision-time score/book context
  plus only strictly matured realized outcomes; and
- exact failure labels comparing `pre[-12h,0)` with `post[0,+12h)`, requiring
  negative post net, causal mapped-residual degradation and two-of-next-three
  persistence. Target availability follows all post-window outcomes.

The broad residual-z threshold produces 61 merged failure episodes; the
strict threshold produces 41. The broad label therefore crosses the requested
60-event floor for binary research, but neither label supports a detailed
failure taxonomy. The script and focused test are
`scripts/materialize_historical_exact_model_health.py` and
`tests/test_historical_exact_model_health.py`.

The grouped-OOF feature ablation is
`data_perp/artifacts/historical_exact_model_failure_ablation_20260729_v1/`.
Every failure run and its +/-12-hour context is kept in one indivisible
validation group; remaining controls are calendar-week blocks. This is
research OOF, not chronological promotion evidence.

| Failure label | Market only AP / Brier | Market + health AP / Brier | Latest-April AP | Event recall at 2 false alerts/30d |
|---|---:|---:|---:|---:|
| Broad | 0.3785 / 0.1996 | **0.3887 / 0.1969** | 0.3020 | 0.1148 vs 0.0656 market-only |
| Strict | 0.2852 / 0.1268 | **0.3087 / 0.1197** | 0.3478 | 0.1220 vs 0.0732 market-only |

The health block is incremental but uncertain: event/block bootstrap median
AP improvements are about +0.011 broad and +0.023 strict, while both 5th
percentiles remain negative. Health-only and active-transition-only models
are worse than market-only. Adding chronological active-transition
probability to `market + health` also worsens aggregate AP, and explicit
`active risk × health` interactions worsen it further. April broad is the
one exception, where adding active risk raises AP from 0.302 to 0.387; this is
a regime-specific hypothesis, not an aggregate winner.

Only 4/61 broad and 5/41 strict failure episodes lie within +/-6 hours of a
canonical market transition. This reinforces the required separation:
market-transition risk and economic-model-failure risk are mostly different
events. Do not merge their labels or make active-transition probability a
generic failure veto.

The global highest-risk deciles are economically adverse at candidate level,
but the classifier is not yet a clean ordinal severity rank. For the broad
winner the risk tail has a 46.2% failure rate and -71.6 bps post-12h net; for
the strict winner it has a 35.4% failure rate and -64.4 bps. Nevertheless,
the accepted portfolio subset inside labeled failure hours can remain
profitable. The correct policy interpretation is therefore a
context-dependent alpha threshold, not blanket abstention or exposure
reduction.

### Frozen failure-risk portfolio gate

The exact-policy runner is
`scripts/run_model_failure_risk_policy_sweep.py`. It consumes the official
grouped-OOF `market_plus_health` probabilities, retains the pooled global
top-k contract, applies the shared concurrency/exposure/asset constraints,
and persists selected, accepted, equity and replacement-attribution ledgers.
All reports use economic-failure terminology; internal transition aliases are
not exposed.

February--March is the development cohort. For both broad and strict labels,
the best arm is `threshold_increase`, lambda 1, and it improves PnL in each
development month:

| Feb--Mar policy | Baseline PnL | Frozen-candidate PnL | Delta | Sortino | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Broad threshold lambda 1 | +$365.37 | **+$1,837.49** | **+$1,472.12** | 0.0417 vs 0.0092 | -16.05% vs -17.33% |
| Strict threshold lambda 1 | +$365.37 | **+$784.75** | **+$419.38** | 0.0185 vs 0.0092 | -16.11% vs -17.33% |

The untouched April `prior_frozen` artifacts are:

- `model_failure_risk_policy_broad_frozen_april2025_20260729_v1`;
- `model_failure_risk_policy_strict_frozen_april2025_20260729_v1`.

| April policy | Trades | Net PnL | Delta | Sortino | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Baseline | 413 | +$3,281.72 | -- | 0.1474 | -12.88% |
| Broad threshold lambda 1 | 342 | **+$3,679.76** | **+$398.05** | 0.1665 | -11.10% |
| Strict threshold lambda 1 | 377 | **+$3,580.52** | **+$298.81** | 0.1610 | -12.53% |

The broad arm improves accepted PnL during labeled failure episodes by about
$255.50 and outside them by $142.55. The strict arm improves those two
partitions by about $151.15 and $147.66. At candidate level, the broad arm
removes 4,048 candidates averaging -80.46 bps, including 1,197 failure-hour
rows; the strict arm removes 2,352 candidates averaging -79.62 bps,
including 607 failure-hour rows. Neither arm adds replacement candidates:
the mechanism is a stricter context-dependent admission floor.

This result is materially stronger than blanket exposure reduction, which
lost money in development, and stronger in failure attribution than the
active-transition threshold arm, whose April gain occurred outside true
transition hours. It is still **not promotion eligible**:

1. the probability model is grouped OOF, not chronological OOS;
2. it uses the older raw-alpha lineage, not current execution-EV health;
3. broad/strict policies are highly correlated threshold variants rather than
   independent confirmations;
4. April contains only 21 broad and 10 strict failure events; and
5. the selected portfolio can be profitable during a label-defined aggregate
   failure, so the label-to-policy semantics require a current-lineage audit.

Next steps are to materialize the same 28-feature health contract on the
current lineage, train chronological failure probabilities with labels fully
resolved before each fold, freeze the broad threshold arm on a prior block,
and require later-block improvement overall, during true damaging episodes,
outside them, and after replacement attribution. Active-transition
probability should be tested as a conditional modifier only after the
current-lineage failure model transfers; the present aggregate interaction is
negative.

The combined transition, health, failure-ablation and failure-policy focused
suite passes 32/32 tests.

### Superseding exact-hour, candidate-weighted failure workstream

The canonical label implementation now reindexes to an exact hourly calendar.
`pre[-12h,0)` and `post[0,+12h)` are elapsed-time windows, not 12 observed
rows. A missing source hour makes the affected window ineligible rather than
silently lengthening it. Net EV and mapping residual are aggregated as sums
divided by selected candidate rows, so a sparse hour no longer receives the
same weight as a dense hour. Zero-selection but otherwise observed hours
contribute zero rows and require no artificial outcome wait. Every target
availability time remains later than all outcomes in its post window.

The authoritative older-lineage artifact is
`data_perp/artifacts/historical_exact_model_health_failure_20260729_v3/`.
It contains 2,113 hours, the same 50,444 pooled-global raw-alpha top-10
candidates and 28 causal health fields, but the corrected labels now produce
60 broad and 46 strict episodes. Versions v1 and v2 are invalidated.

The current execution-EV lineage is independently materialized at
`data_perp/artifacts/current_lineage_exact_failure_labels_20260729_v1/`.
Its admission book is exactly the requested one pooled global top 10% after
`causal_recent_side_isotonic_ev`, never per timestamp or side. All 114,096
candidate scores are strict OOF; 11,410 are selected. The attached 29-field
current-health panel spans 1,556 hours from May 6 through July 10, 2026.
Only 874 hours (56.17%) have complete exact before/after support, yielding
12 broad and 8 strict failure episodes. This proves current-lineage label
parity but remains insufficient for a stable standalone failure model.

The resolved frozen-forward extension is now also materialized:

- `current_lineage_extended_model_health_20260729_v1`;
- `current_lineage_exact_failure_labels_resolved_july19_20260729_v1`.

It preserves 114,096 outer-OOF rows and adds 7,112 already-opened,
resolved frozen-forward OOS rows through July 19. The 29-field health panel
now spans 1,705 hours and 121,208 candidates; its global mapped top-10 book
contains 12,121 candidates. This adds only one broad and two strict episodes,
for totals of 13 broad and 10 strict. The extension therefore does not solve
the support blocker. Forward rows may train only a later detector and are
forbidden from evaluating any detector fitted on this combined history.

The superseding grouped-OOF ablation is
`data_perp/artifacts/historical_exact_model_failure_ablation_20260729_v3/`.
Failure episodes plus +/-12-hour context remain indivisible validation groups.

| Label / feature block | AP | ROC-AUC | Brier | April AP |
|---|---:|---:|---:|---:|
| Broad market only | **0.4418** | **0.6898** | **0.1901** | 0.3903 |
| Broad market + health | 0.3767 | 0.6298 | 0.2106 | 0.2815 |
| Broad market + health + active interactions | 0.4298 | 0.6568 | 0.1995 | **0.4135** |
| Strict market only | 0.2867 | 0.6858 | 0.1440 | **0.3095** |
| Strict market + health | 0.2987 | **0.7124** | 0.1423 | 0.2847 |
| Strict market + health + active interactions | **0.3175** | 0.7080 | **0.1359** | 0.2338 |

The corrected interpretation is materially different:

- broad health is not incremental; `market + health` loses 0.0651 AP versus
  market-only, and its event/bootstrap delta is negative with 99.1%
  probability;
- strict health/active interactions improve aggregate AP by 0.0308, but the
  bootstrap 5th percentile remains -0.0328 and April reverses sharply;
- active risk still does not justify a generic veto;
- only 5/60 broad and 5/46 strict episodes occur within +/-6 hours of a
  canonical transition; and
- at two false alerts per 30 days, event recall remains only 5.0% broad for
  market-only and 6.5% strict for the interaction winner.

The highest-risk deciles remain economically adverse: broad market-only
records a 53.8% failure rate, -58.8 bps post-12h net and -84.0 bps residual
change; the strict interaction arm records 36.8%, -60.1 bps and -72.9 bps.
These are useful ranking diagnostics, but alert recall is too low for a hard
gate.

The corrected policy artifacts use the corresponding aggregate classifier
winners:

- broad: `market_only`;
- strict: `market_plus_health_plus_active_interactions`.

Both February--March grids again choose `threshold_increase`, lambda 1:

| Corrected development arm | Baseline PnL | Challenger PnL | Delta |
|---|---:|---:|---:|
| Broad market-only | +$365.37 | **+$1,964.40** | **+$1,599.03** |
| Strict interaction | +$365.37 | **+$932.61** | **+$567.24** |

The policies were frozen and evaluated once on April:

| Corrected April arm | Trades | Net PnL | Delta | Sortino | Max DD |
|---|---:|---:|---:|---:|---:|
| Baseline | 413 | +$3,281.72 | -- | 0.1474 | -12.88% |
| Broad market-only threshold 1 | 343 | **+$3,697.92** | **+$416.20** | 0.1682 | -11.10% |
| Strict interaction threshold 1 | 378 | **+$3,581.31** | **+$299.59** | 0.1605 | -12.56% |

These are real untouched-month aggregate improvements, but neither validates
failure protection. For broad, accepted PnL during true failure episodes
falls from $2,393.70 to $2,271.03 (-$122.67), while outside-failure PnL rises
by $538.87. For strict, failure-episode PnL falls by $23.47, while
outside-failure PnL rises by $323.06. The surviving high-alpha trades during
failure hours are highly profitable; the threshold acts as a general
contextual admission filter.

Therefore:

1. retain the broad market-only and strict interaction scores as research
   comparators, not controls;
2. do not claim incremental current model-health value from the historical
   experiment;
3. do not promote the frozen threshold as failure protection;
4. extend the exact current-lineage OOF ledger until at least 60--100
   current-lineage failure episodes resolve;
5. train chronological current-lineage probabilities only after that support
   exists; and
6. require future gains during damaging episodes, outside them and overall,
   with replacement attribution, before any control is eligible.

The combined transition, exact-hour labels, current-lineage and
resolved-forward labels, failure-ablation and policy suite passes 39/39 tests.

The four-head report now includes the previously missing current-state and
destination-state classification metrics. Both heads predict only `stable`:
balanced accuracy is 0.500, macro F1 is 0.483/0.484 and recall for the
correlation-fragmentation failure state is 0.000. The high raw accuracy
0.934/0.939 is therefore class imbalance, not useful state discrimination.

The frozen historical detector was then transferred without refitting to the
current 1,556-hour panel under an exact 28-feature cross-era contract. It
identified the June 8 catastrophic break near the 99th risk percentile three
hours before onset, but only weakly identified the June 11 failure. The max of
transition and destination risk improves aggregate pooled-global top-10 by
4.42 bps, from -5.89 to -1.47 bps. It improves June by 1.68 bps, leaves May
unchanged, and worsens July by 0.15 bps to -55.16 bps. Every adjusted
aggregate and July result remains negative.

Artifact:
`data_perp/artifacts/failure_first_detector_current_transfer_20260726_v6/`.

Decision: infrastructure completion is proven; empirical model completion is
not. Retain transition/destination probabilities as research uncertainty
features only. The next bounded model ablation is a direct supervised binary
failure/transition detector with class weighting or focal loss, plus explicit
market/transition/model-health block ablations. A same-model chronological OOF
history satisfying the frozen support gates and a sealed later forward block
remain mandatory before admission or action-layer integration.

### Direct binary failure/transition ablation

`failure_first_binary.py` and
`scripts/run_failure_first_binary_ablation.py` implement the predeclared
alternative to forcing unsupported taxonomy states. The targets are:

- failure onset within the next three fully observed hours;
- failure active now or reached within the next three hours.

They are constructed directly from strict-OOF economic health bins, reset at
evaluation-origin boundaries and carry exact label-resolution timestamps.
There is no taxonomy dependency. The historical runner first compares four
fixed CatBoost geometries, including unweighted and balanced-class arms, then
uses the winning geometry for block ablations:

- market state only;
- model health only;
- market plus model health;
- each of those with 1h/3h causal deltas and BOCPD;
- full without BOCPD;
- full without causal deltas;
- full 40-field contract.

The HPO research winner is balanced CatBoost with depth six, 160 iterations,
0.04 learning rate and L2 8. The feature research winner contains 20
model-health/transition fields. It covers 5,519 chronological OOF hours:

| Metric | Aggregate | November 2025 |
|---|---:|---:|
| Failure-onset AUC | 0.513 | 0.553 |
| Active-or-within-3h risk AUC | 0.544 | 0.589 |
| Positive onset labels | 87 | 21 |
| Positive risk labels | 351 | 69 |
| Mapped pooled-global top-10 | -145.43 bps | -121.76 bps |
| Risk-adjusted pooled-global top-10 | -138.28 bps | -137.98 bps |
| Increment | +7.15 bps | -16.23 bps |

No HPO or feature arm passes the joint discrimination/latest/economic gate.
Model health only is the most useful reduced classifier family but still
worsens latest economics by 3.29 bps. BOCPD is not incremental inside the full
architecture: removing it improves aggregate/latest risk AUC from 0.556/0.360
to 0.561/0.382 and reduces latest damage from -8.00 to -5.02 bps. Causal
deltas have mixed classification effects and no stable economic contribution.

The frozen historical binary winner was transferred without refitting to the
current 1,556-hour panel:

| Metric | Current aggregate | July |
|---|---:|---:|
| Failure-onset AUC | 0.339 | unavailable: zero positive labels |
| Active-or-within-3h risk AUC | 0.323 | unavailable: zero positive labels |
| Mapped pooled-global top-10 | -5.89 bps | -55.02 bps |
| Risk-adjusted pooled-global top-10 | +1.61 bps | -54.25 bps |
| Increment | +7.50 bps | +0.76 bps |

The positive reused aggregate does not validate the detector: current
failure-label discrimination is inverse, July remains strongly negative, and
historical latest-period economics deteriorate. It is likely exploiting a
ranking correlation unrelated to transferable failure recognition. Promotion,
portfolio integration and action-layer routing remain forbidden.

Artifact:
`data_perp/artifacts/failure_first_binary_ablation_20260726_v1/`.

The frozen binary detector was also evaluated on the separately flagged
frozen-final-fit July 11-19 forward-OOS cohort. Its 7,112 rows have exact
resolved outcomes and raw-H0 coverage; the scorer preserves the forward flag
and never substitutes it for the strict-OOF flag. It covers 149 hourly states.
There are zero positive direct failure labels, so onset/risk AUC cannot be
estimated. The economic result is unambiguously negative:

| Forward score | Pooled-global top-10 |
|---|---:|
| Mapped control | -163.71 bps |
| Failure-onset adjusted | -163.53 bps |
| Active-or-within-3h risk adjusted | -169.54 bps |

The risk overlay worsens the frozen forward book by 5.83 bps. The small 0.18
bps onset difference is noise relative to the loss scale. This forward result
overrides any temptation to promote the +1.61 bps reused-current aggregate.

Artifact:
`data_perp/artifacts/failure_first_binary_forward_july19_20260726_v1/`.

## 2026-07-26 pooled symmetric transition implementation

The regime-transition research infrastructure has been rebuilt around native
hourly, symmetric before/after targets. Do not reuse
`build_hourly_state_transition_labels` or the six-hour-expanded failure
targets for this workstream.

Canonical implementation and artifacts:

- `extreme_price_movements/regime_transition_research.py`;
- `scripts/materialize_regime_transition_research.py`;
- `data_perp/artifacts/regime_transition_research_20260726_v3/`;
- `scripts/run_regime_transition_classifier_ablation.py`;
- `scripts/run_regime_transition_lightgbm_hpo.py`;
- `scripts/run_regime_transition_active_head.py`;
- `scripts/materialize_regime_transition_model_health.py`;
- `scripts/run_regime_transition_model_health_ablation.py`.

The market panel has 30,931 hours from January 2023 through July 12, 2026,
split at the one missing-hour boundary. It retains 58 level fields, all 28
pre-existing market transition fields and 120 new exact-lag/short-long shift
fields. Source time `t` maps to decision time `t+1h`; all lags reset at gaps.

The state contract is:

- origin `[-12h,-3h)`;
- lead `[-3h,0h)`;
- active transition from onset to the first three-hour-persistent destination,
  capped at `+6h`;
- destination from the settled `[+6h,+12h)` state;
- exact event snapshots at `-48,-24,-12,-6,-3,0,+3,+6,+12h`.

The pooled five-state geometry passes the relaxed 1% minimum-state-share
research gate with silhouette 0.582 and yields 151 durable events. Five-fold
event/control-block grouped validation—not random rows and intentionally not
walk-forward—produces:

| Head | Primary results |
|---|---|
| onset within 3h | PR-AUC 0.133; ROC-AUC 0.799; 9.06x base-rate lift |
| active transition | PR-AUC 0.340; ROC-AUC 0.959 |
| settled destination | balanced accuracy 0.612; macro-F1 0.596 |
| seven-class phase | balanced accuracy 0.329; macro-F1 0.257 |

At threshold 0.25, the 3h onset head recalls 21.9% of events at 2.84 false
alert episodes per 30 days. Therefore:

- active and destination probabilities are valid research context;
- 3h onset has signal but is not reliable enough for a hard trade gate;
- the monolithic phase head should be abandoned in favor of separate binary
  onset/active plus conditional destination heads.

The native hourly economic overlay compares `[-12h,0)` versus `[0,+12h)` and
materializes 9,090 hours and 25 adverse episodes. Only three canonical market
transitions fall within six hours of those failures. This is insufficient for
a stable five-to-eight economic failure taxonomy.

The outcome-free old55 model-health overlay covers 11,921 hours and 62
distribution features. On the shorter 47-event overlap, adding it raises
PR-AUC from 0.032 to 0.040 and ROC-AUC from 0.723 to 0.740. Treat this as
incremental sensitivity evidence only: it is old55 lineage, not current-model
parity.

Next: multi-horizon hazard/time-to-onset auxiliaries, one compact
change-point-family ablation, current-lineage health aggregation, extension
through July 21, December 2025 exact-one-minute completion, and substantially
more economic failure episodes before taxonomy fitting.

## 2026-07-27 cumulative hazard and event-impact continuation

The independent-horizon onset design now has a grouped discrete-time survival
challenger:

- `extreme_price_movements/regime_transition_hazard.py`;
- `scripts/run_regime_transition_hazard_challenger.py`;
- `data_perp/artifacts/regime_transition_hazard_challenger_20260727_v1/`.

The risk set excludes onset and post-onset rows, censors at segment ends and
internal gaps, and uses intervals `(0,1]`, `(1,3]`, `(3,6]`, `(6,12]`.
Event groups and seven-day control blocks never cross train/validation folds.
The cumulative incidence is structural
`1 - cumulative_product(1 - interval_hazard)`, so the saved predictions obey
`P(1h) <= P(3h) <= P(6h) <= P(12h)`.

| Horizon | PR-AUC | ROC-AUC | Brier | Prevalence |
|---|---:|---:|---:|---:|
| 1h | 0.0583 | 0.8387 | 0.00501 | 0.00516 |
| 3h | 0.0992 | 0.7919 | 0.01470 | 0.01549 |
| 6h | 0.1486 | 0.7650 | 0.02878 | 0.03078 |
| 12h | 0.1709 | 0.7061 | 0.05193 | 0.05550 |

For the 3h head, event recall is 11.92%, 16.56% and 22.52% at fixed budgets
of one, two and four false-alert episodes per 30 days. At two false alerts,
abrupt-event recall is 18.95% and gradual-event recall is 12.50%. The hazard
contract is better structured than independent probabilities but does not beat
the earlier direct 3h classifier's PR-AUC and remains unsuitable for a hard
veto. Severity weighting is event-level and optional; weighted probabilities
must not be interpreted as ordinary incidence calibration.

Origin-to-destination impact attribution is materialized at
`data_perp/artifacts/regime_transition_event_impact_20260727_v1/` by
`scripts/attribute_regime_transition_event_impact.py`. It reports native
mapped/base EV and realized net EV before, during and after each event,
damage, recovery, duration, severity, coverage and explicit economic-failure
links. The 151-event transition catalog supports only 30
event-by-evaluation-origin economic rows and 18 severe-or-damaging selections.
Stop rate, calibration, rank IC and shrinkage remain unavailable rather than
being filled with proxies. Pair-level economics are therefore descriptive
only; a stable damage router still requires substantially more common-lineage
events.

The proposed active-transition portfolio sweep must not use
`exact_history_state_recurrence_20260727_v1`: that artifact has an explicit
`ECONOMIC_INVALIDATION.json` because its historical fee-only simulator is not
the deployed spread-aware target. The sweep runner now refuses any source with
that marker before creating an output directory. The exact current
replay-compatible OOF overlap has 52,631 rows from June 8 through July 11 but
zero realized active-transition rows; its conditional economic comparison is
undefined. In addition, the active-head grouped OOF is shuffled across
2023--2026 and is research-only rather than chronological policy-OOS evidence.

Consequently, do not run or promote trust-discount, threshold-increase or
exposure-reduction arms until a common target-lineage cohort supplies exact
entry/exit paths and includes actual active-transition episodes. The valid
next step is cohort repair, not tuning a risk multiplier on mismatched labels.

That cohort repair is now materialized at
`data_perp/artifacts/active_transition_exact_policy_cohort_20260727_v2/`.
It joins 114,096 strict candidate-OOF causal-global-21-day-EV rows from May 6
through July 10 to exact one-minute policy entries, actual exit timestamps and
prices, gross/cost/net returns, exit reasons and grouped-OOF active
probabilities. Its frozen pooled global top 10% contains 11,410 candidates;
the shared constrained replay accepts 313 and produces -47.12% compounded
return, -47.30% maximum drawdown, -0.0619 Sortino and -0.9654% mean net return
per accepted trade.

The repaired cohort confirms the evidence limitation rather than solving it:
only one transition event overlaps, represented by 255 candidates across
three active hours, and none of those candidates belongs to the frozen global
top-10 book. Thus the integration mechanics and exact replay schema are valid,
but the protection effect is unidentifiable and no lambda sweep should be run.
`evidence_gate.json` records this distinction machine-readably. A
promotion-valid test still requires more exact-policy history containing
multiple active events plus chronological active probabilities.

The compact current-lineage health panel is also complete:

- `extreme_price_movements/regime_transition_current_model_health.py`;
- `scripts/materialize_regime_transition_current_model_health.py`;
- `data_perp/artifacts/regime_transition_current_model_health_20260727_v1/`;
- `data_perp/artifacts/regime_transition_current_model_health_ablation_20260727_v1/`.

It contains 29 contemporaneously observable hourly fields over 1,556 hours
and 114,096 exactly matched current-lineage candidates from May 6 through
July 10. Old55 is excluded. Families cover candidate/asset breadth and side
balance, mapped-EV distribution and entropy, global-versus-side disagreement,
base-score distribution, residual magnitude and dispersion,
base-versus-residual rank conflict, cutoff margin, score coverage, CatBoost
entropy, alpha uncertainty, and strictly-prior three-day-decayed resolved net
EV, hit rate, mapping error, cost and support. Outcomes resolving at the
decision timestamp are excluded before each snapshot.

No incremental transition metric is reported: the common panel contains only
three positive lead rows from one independent onset event, fewer than the five
requested grouped folds. The ablation records
`INSUFFICIENT_INDEPENDENT_EVENTS_FOR_GROUPED_OOF` and leaves PR-AUC, ROC-AUC
and fixed-alert recall null. Do not substitute row-level splits, in-sample
metrics or old55 history. Current-lineage OOF coverage must span at least five
independent transitions before this add-on can be judged.

The one allowed change-point-family ablation is complete at
`data_perp/artifacts/regime_transition_changepoint_ablation_20260727_v2/`.
It uses a bounded univariate Normal-Inverse-Gamma BOCPD over six observable
market signals, summarized into only four context fields. Each segment resets
the posterior; the first 30 days provide fixed robust scaling and are
unscored; the posterior at `t` sees observations only through `t`. This is an
additive context test, not another state taxonomy.

| Arm | Fields | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| transition baseline | 212 | 0.1418 | **0.8095** | 0.01415 |
| BOCPD only | 4 | 0.0157 | 0.5006 | 0.14581 |
| baseline + BOCPD | 216 | **0.1481** | 0.7868 | **0.01411** |

Alert thresholds are frozen from the first chronological 25% of OOF scores
using score frequency only; the later era contains 97 events. At the nominal
two-alerts-per-30-day cutoff, the additive arm records 14.43% event recall at
0.435 measured false alerts per 30 days versus 13.40% at 0.714 for the
baseline. Gradual recall rises from 12.12% to 18.18%, while abrupt recall
falls from 14.06% to 12.50%. At the one- and four-alert cutoffs the additive
arm has lower total recall, although it also produces fewer measured false
alerts. BOCPD alone is essentially non-predictive.

This is narrow incremental evidence for gradual-transition context: keep the
four-field block available for a future independent-event confirmation, but
do not promote it, create a hard gate, or run another change-point family now.
The supervised folds remain shuffled grouped research OOF, and a production
alert cutoff would need to be frozen on an earlier completed deployment-era
calibration set.

## 2026-07-27 historical exact-policy repair and score-lineage readiness

The deployed-label reconstruction contract is now proven rather than inferred:

- `scripts/audit_deployed_policy_label_parity.py`;
- `data_perp/artifacts/deployed_policy_label_parity_20260727_v1/`;
- `scripts/build_historical_exact_policy_readiness.py`;
- `data_perp/artifacts/historical_exact_policy_readiness_20260727_v2/`.

A deterministic 96-candidate May--July replay, balanced across side and month,
matches the canonical label artifact with zero identity or field mismatches,
zero maximum numerical delta and zero accounting error. The comparison covers
entry/exit prices, gross/cost/net EV, exit time and reason, MFE/MAE, spread and
geometry. The actual deployed resolution is 100% side-parent fallback: the
policy contains side-by-archetype strategies, but no observable candidate in
this lineage maps to them. Historical materialization preserves that fact
rather than fabricating archetype assignments.

Historical readiness is:

| Period | Candidate rows | Canonical path rows | Exact 1m labels | Decision |
|---|---:|---:|---:|---|
| January 2025 | 222,773 | 0 | 0 | forbidden: no joinable canonical path input |
| February--April 2025 | 688,160 | 510,996 | 509,868 | accepted only on frozen common identities |
| December 2025 | 265,976 | 187,612 | 42,404 | diagnostic only: 22.60% exact coverage |

February--April is the usable historical block. It has 99.78% exact-one-minute
coverage on its admitted canonical subset, 430,464 matching historical strict
two-layer OOF score rows, 13 active transition events, 47 active hours and
11,220 active candidate rows. The labels use the deployed spread-aware
12-hour `simple_policy_optimiser` contract, but historical spreads are the
current frozen per-asset counterfactual, not contemporaneous factual spreads.
Do not compare its restricted historical book with an unrestricted current
book.

The historical score rows are not a substitute for the current 31/8
execution-EV lineage. `scripts/audit_current_lineage_score_extension.py`
materializes the current strict panel at
`data_perp/artifacts/current_lineage_score_extension_readiness_20260727_v2/`:
114,096 exact candidates from May 6 through July 10 with canonical base,
residual, alpha, raw execution-EV and causal mapped EV. It overlaps only the
single June transition.

The point-in-time feature store nevertheless covers all 164 historical
symbols, every canonical raw 31/8 base feature and the full residual feature
surface; frozen side-local AE/GMM and HPO contracts also exist. Therefore a
fresh chronological current-lineage reconstruction is feasible. With the
deployed February--April labels now supplied, the remaining ordered work is:

1. rebuild fresh side-local canonical 31/8 base OOF;
2. rebuild the side-local top-40 population and residual-alpha OOF;
3. train historical per-side CatBoost and conditional peak/event heads OOF;
4. train fold-local execution-EV OOF and causal recent mapping;
5. join transition probabilities and require at least five independent events;
6. only then rerun health incrementality and active-risk portfolio policies.

Never backcast the frozen 2026 execution bundle, use in-sample scores, or
resurrect the invalidated fee-only target. February--April alone provides 13
events, so it is the minimum valid reconstruction block.

The event-by-event readiness map is canonical at
`data_perp/artifacts/regime_transition_coverage_readiness_20260727_v2/`.
It intersects all 151 events with score lineage, exact one-minute path,
entry/exit replay, deployed geometry, health inputs, active probability and
economic-failure linkage, with source hashes and stable event fingerprints.

All 151 events have active grouped-OOF coverage. Twenty-seven events from
February through November 2025 already have complete archival common evidence:
strict historical OOF score identity, exact path and execution replay,
deployed geometry, archived health and active probability. Their monthly
distribution is February 2, March 1, April 6, May 2, June 2, July 2, August 1,
September 4, October 4 and November 3. The earliest five-event reconstruction
set is February 3, February 4, March 29, April 7 and April 8. Higher-priority
validation anchors by recorded severity/readiness are October 10, February 4,
October 11, November 4 and July 2.

Every one of the 27 still lacks the newly defined current score and
current-health lineage. Therefore the minimum path is not more raw path or
policy reconstruction: backfill current canonical score and health on these
already exact event windows. The queue reason is
`BACKFILL_CURRENT_SCORE_AND_HEALTH_ON_ARCHIVAL_EXACT_COHORT`.

The historical failure catalog contains 35 native economic-failure episodes:
25 short of the lower 60-event target and 65 short of 100. Only three fall
within six hours of a canonical transition, eight within 12 hours and 11
within 24 hours. Do not infer an event count merely from extending calendar
coverage; materialize and count native failures explicitly.

The first current-lineage reconstruction gate is now complete:

- `data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/`;
- `extreme_price_movements/canonical_318_historical_calendar.py`;
- `scripts/run_febapr2025_canonical_base_oof.py`;
- `data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/`.

The frozen population has 509,868 exact identities, 254,934 per side, with
13 transition events. The base reconstruction uses the native
`__first_touch_target_soft__`, native `__w__` weight and first-touch economic
diagnostic. Deployed `execution_net_ev_12h` is never substituted for base
supervision; it is joined only for held-out economics.

Three expanding monthly folds per side produce six fits. January native
base-label history supplies the first training block; later folds expand
through resolved history. Every training label resolves before validation
start. Each side/fold uses an outcome-free, calendar-month-stratified,
candidate-ID-hash cap of 100,000 rows. All 509,868 frozen identities receive
exactly one OOF prediction. Point-in-time feature keys match 100%; frozen
31-long/8-short features, `trial_141`/`trial_084`, and side-specific AE/GMM
states are hashed in each shard. Outer OOF retains label-complete rows with
LightGBM native missing values and records per-feature coverage; no imputation
or outcome-defined event sampling occurs.

| Base-only diagnostic | Rows | Base-target Spearman | Global top-10 exact execution EV |
|---|---:|---:|---:|
| all sides/months | 509,868 | 0.1752 | -62.55 bps |
| April latest month | 172,450 | 0.1948 | -58.35 bps |
| long | 254,934 | 0.1801 | -63.69 bps |
| short | 254,934 | 0.1755 | -48.14 bps |

The identity, feature, provenance and OOF gate passes. Base-only execution
economics fail, as expected for an alpha target that is not the final
cost-aware ranker. This result neither rejects the base nor authorizes
portfolio replay. The next bounded step is side-local top-40 materialization
and residual-alpha OOF on this exact lineage, followed by an identical-row
base-versus-residual economic gate.

That residual gate is now complete:

- `data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/`;
- `data_perp/artifacts/febapr2025_canonical_top40_residual_readiness_20260727_v3/`;
- `scripts/run_febapr2025_canonical_residual_oof.py`;
- `data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/`.

The timestamp-by-side top-40 handoff contains 205,194 rows, 102,597 per side.
All selected identities, ranks, group sizes and rank percentiles reproduce
exactly from the fresh base OOF with zero mismatches. Target
`__first_touch_capture_net__`, soft target and weights are finite; weights are
strictly positive. February has no prior resolved top-40 base OOF support and
is retained as 64,512-row warm-up metadata only with
`residual_is_oof=false` and no residual-EV value. It is excluded from every
residual metric.

March and April supply four strict side-by-month residual shards and 140,682
identical OOF rows. Each shard uses the frozen 69-field side contract, HPO
parameters and correction alpha, a fold-fitted side-local isotonic base-EV
map, only prior resolved labels, 100% point-in-time key matches and at least
95% finite coverage for every selected field. No in-sample base score enters
training.

| Strict March--April top-10 diagnostic | Base EV map | Residual EV | Change |
|---|---:|---:|---:|
| one pooled global book | -39.47 bps | **-26.59 bps** | +12.89 bps |
| long-only diagnostic | -27.67 bps | -48.44 bps | -20.77 bps |
| short-only diagnostic | -41.56 bps | **-10.48 bps** | +31.08 bps |
| March pooled | -29.56 bps | -26.45 bps | +3.11 bps |
| April pooled | -51.87 bps | **-24.32 bps** | +27.55 bps |

Native-target Spearman rises from 0.1562 to 0.1869 globally, from 0.1671 to
0.1810 long and from 0.1519 to 0.2169 short. Thus the residual learns its
native target and improves the actual pooled global ranking in both strict
months, but remains negative after exact execution costs. Its transfer is
side-asymmetric: short improves materially while long execution ranking
degrades. Preserve both predictions for downstream identical-row ablations;
do not blindly promote residual on each side or replay a portfolio yet.

The base-only transition diagnostic at
`data_perp/artifacts/febapr2025_base_oof_transition_diagnostic_20260727_v1/`
confirms complete score coverage for all 13 event windows and all 47 active
hours. Median per-event score-to-target IC is 0.239 before, 0.222 during,
0.136 after and 0.129 on active hours. Median exact 12-hour EV is respectively
-1.376%, -1.378%, -1.153% and -1.358%. Active long EV is -2.572% versus
-0.052% short. This is descriptive evidence of side-asymmetric transition
damage, not a routing rule.

### Base-target IC versus execution-EV divergence is a required workstream

Do not dismiss the gap between improving native-target IC and negative
execution economics merely because the base is an alpha model. It is now a
dedicated identical-row diagnostic at
`data_perp/artifacts/febapr2025_base_oof_economic_divergence_20260727_v1/`.
The audit reproduces the long February/March/April pooled rank IC of
0.155/0.162/0.226. The signal is not only cross-asset composition:
timestamp-local IC is 0.233/0.214/0.235 and symbol-neutral IC is
0.129/0.144/0.197.

The economic decomposition finds a real magnitude-and-horizon problem. Long
top-decile gross execution return is +40.8/+8.7/+61.9 bps, but the exact
spread-aware cost is +100.2/+100.0/+100.3 bps, producing
-59.4/-91.3/-38.5 bps net. Short gross is +80.2/+54.3/+17.8 bps against
approximately 100 bps cost, producing -20.2/-46.0/-82.3 bps net. Thus April
long improves in both IC and net EV relative to March; the important
divergence is not a simple monotonic IC-up/EV-down sequence. It is that higher
average rank agreement with the native target does not reliably select a tail
whose gross magnitude clears costs, with especially clear February-to-March
long and March-to-April short failures.

The native soft target resolves 24 hours after decision while exact execution
EV resolves under the deployed 12-hour exit policy. This is not a row-join
error. The native target is economically relevant—native-target to gross/net
rank IC remains approximately 0.59--0.61—but 18.7%--27.4% of top-decile rows
have a positive native label and non-positive execution net, and 49%--60% of
the selected top decile has non-positive execution net. Adjacent-hour
top-decile symbol turnover of approximately 77%--84% also makes a static alpha
cutoff unsuitable.

Add the following controlled experiments to the active workstream:

1. Preserve the current 24-hour native target as the directional baseline,
   but train a matched 12-hour native-label challenger on the same rows,
   features, folds and side-local HPO. Report pooled, timestamp-local and
   symbol-neutral IC plus exact gross, cost and net top-decile economics.
2. Separate direction from magnitude: ablate a gross-opportunity head, a
   cost-hurdle probability (`gross > realized cost + margin`) and a direct
   exact-net-EV head. Tune the margin only inside training folds.
3. Measure tail recall explicitly: overlap with the native-target top decile,
   exact-gross top decile and exact-net oracle top decile, together with
   precision for positive net EV. Aggregate IC cannot substitute for these
   metrics.
4. Attribute failures by month, side, asset, exit reason, MFE/MAE/time path,
   transition phase and cost bucket. Require latest-month coverage and do not
   accept an aggregate win that hides a losing side or latest month.
5. Test whether the execution layer should consume both the 24-hour
   directional score and a separately learned 12-hour opportunity score.
   CatBoost/path and auxiliary heads remain candidates for the latter; timing,
   MAE, target-price and wait actions remain in the separate action layer.
6. Select and calibrate on the actual downstream rule: causal recent EV
   mapping followed by one pooled global top-`k` book and the frozen portfolio
   constraints. Timestamp-local ranks remain diagnostics/context features,
   not the selection policy.
7. Treat rising base-target rank IC alongside falling execution EV as an
   independent failure mode, not as an expected consequence of the
   two-layer architecture. On identical rows, decompose each month and side
   into:
   - score-to-native-target IC, score-to-exact-gross IC and score-to-exact-net
     IC, including decile monotonicity rather than correlation alone;
   - top-decile gross return, realized cost, net return, positive-net
     precision, exact-gross/net oracle overlap and score/rank stability;
   - selected-universe composition by asset, score concentration, candidate
     count, exit reason, MFE/MAE, time to meaningful MFE and failure horizon;
   - the corrected exact-time regime/transition fields from
     `febapr2025_strict_residual_gross_regime_context_20260729_v3`, with
     separate within-regime ranking and between-regime allocation effects.

   This diagnostic must distinguish at least four hypotheses: the base ranks
   the 24-hour alpha target correctly but not the deployed 12-hour payoff; it
   ranks direction but not magnitude above the approximately 1% hurdle; its
   top tail changes asset/regime composition while aggregate IC improves; or
   the exit policy converts similar latent opportunities differently across
   regimes. Compare matched score quantiles and fixed-size pooled global
   top-`k` selections. Do not use per-timestamp selection, ex-post transition
   labels, or any regime context that was not known at decision time.

The strict gross/hurdle grid is frozen at
`data_perp/artifacts/historical_execution_ev_gross_hurdle_decomposition_20260729_v2/`.
Its 2,110 planned and completed fits compare direct gross, direct net, hard
hurdle, soft hurdle and probability-times-conditional-magnitude targets across
base+residual, risk, peak and six-class add/drop arms. Every fit uses March
development only with side-local purging; April's 69,258 rows remain untouched.
Runner, source and all 11 output hashes verify, and the focused gross/context
suite passes 10/10 tests.

This grid provides an additional warning that the IC-versus-EV failure is
real. The highest raw April net-rank IC is 0.1529 for the
`plus_risk/soft_hurdle` arm, but its raw pooled-global top-decile loses 43.17
bps and its common-unit selection loses 27.26 bps. The best economic result is
instead `plus_risk_peak/direct_net` after cross-side common-unit
standardization: 93.99 bps gross against 100.47 bps cost, or **-6.48 bps net**,
with 53.52% positive-net precision and only 20.23% exact-net-oracle recall.
Raw-unit, March-only-mapped and causal-online-21-day variants are also all
negative. Thus no portfolio replay is justified until the diagnostic
identifies a transferable admission signal rather than another
correlation-only gain.

The first strict March/April decomposition is recorded at
`data_perp/artifacts/historical_base_ic_ev_divergence_20260729_v1/`. It uses
the repaired pre-entry context v3 and reproduces the same qualitative
failure: base score-to-net rank IC is +0.0976 in March and +0.0528 in April,
while pooled-global top-decile net is respectively -78.08 and -91.28 bps.
Oracle recall is only 16.62%/18.51%, despite positive-net precision of
55.79%/56.29%; the negative mean therefore comes from payoff magnitude and
the adverse left tail, not merely binary hit rate. High `range_24h_pct` is
less bad, while a sharp three-hour increase in range is particularly poor.
Treat those strata as challenger hypotheses, not routing rules. A complete
February-to-April identical-row v2, including decile monotonicity, composition,
path attribution and the original native-target IC sequence, remains required
before closing this diagnostic.

The architectural conclusion is unchanged but sharper: retain the base for
directional alpha only if its transfer and tail-recall diagnostics remain
useful; never treat its raw score as an admission score. The execution layer
must learn exact 12-hour gross magnitude and the approximately 1% cost hurdle,
then be judged through causal global mapping and constrained replay.

### Historical top-40 exact one-minute path gate

The path-data blocker for historical CatBoost and path-head reconstruction is
removed:

- `data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1/`;
- 205,194 requested and emitted candidates;
- 64,512 February, 71,424 March and 69,258 April rows;
- 720 one-minute OHLC observations beginning at each decision timestamp;
- 100% complete coverage in every month and zero missing windows;
- exact frozen top-40 identities, historical path ATR/barrier inputs,
  accepted spread/fee fields, immutable one-minute store hashes and the
  signed-v2 source wrapper are bound in the manifest.

This artifact is future outcome data and may be used only to construct labels
and held-out path diagnostics. It must never enter the pre-entry feature
matrix. The next steps are now executable rather than data-blocked:

1. construct the cost-aware path-archetype labels and report class support,
   ambiguity and economic separation per side/month;
2. prove parity between independently reconstructed five auxiliary targets
   and the historical v6 labels;
3. freeze historical pre-entry context and run feature selection, HPO and
   geometry independently per side and role;
4. generate only purged/OOS head predictions for the March--April execution
   layer, using February solely when the corresponding 12-hour labels resolved
   before the validation boundary.

The two historical path-head input surfaces are now fully materialized.

`data_perp/artifacts/febapr2025_top40_exact1m_path_head_labels_20260727_v1/`
contains 42 atomic target shards covering exactly 205,194 of 205,194 identities
with no gap or overlap. The root index verifies every shard hash, source offset
and row count and aggregates side/month support, archetype support and
triple-barrier ambiguity without reloading the one-minute JSON paths. It emits
the five primary auxiliary targets plus their event/validity fields. It also
records an important scope distinction: these are 12-hour
execution-compatible labels using the v6 rule family; frozen CatBoost v6 used
24-hour paths, so the archetype labels are not falsely claimed to be bitwise
equivalent to the old v6 target.

`data_perp/artifacts/febapr2025_historical_path_head_context_20260727_v1/`
contains the matching causal pre-entry surface in 122 resumable symbol shards:
205,194 exact candidate identities, 188,361 unique symbol-by-signal keys,
102,597 rows per side, and 64,512/71,424/69,258 rows in
February/March/April. Its compact identity index matches the frozen population
hash, every shard has the same frozen 256-raw-feature schema, and a forbidden
column scan confirms that no future path, execution outcome, auxiliary target
or target-derived weight entered the context.

Therefore historical path-head training is no longer blocked by data or
feature-context materialization. Remaining gates are model gates:

1. define an economically coherent, adequately supported side-local merge of
   the sparse 12-hour archetype classes;
2. run CatBoost feature selection, HPO and geometry independently per side;
3. run auxiliary feature selection and HPO independently per role and side,
   retaining the meaningful-MFE event hurdle and competing-risk challenger as
   distinct models;
4. generate purged March--April OOF predictions and assess learnability,
   calibration and identical-row incremental execution economics before any
   head is admitted downstream.

The matched 12-hour-versus-24-hour **base-label** experiment is now complete.
The result and its exact held-out execution-EV gate are recorded below. Exact
execution EV was forbidden from native-label construction and model training;
it was joined only after all OOF scores froze.

The native recipe parity part of that unblock sequence is now complete at
`data_perp/artifacts/febapr2025_native_first_touch_24h_replay_parity_20260729_v1/`.
A deterministic 48-row audit balanced by side and February/March/April replays
the archived native label from the canonical 15-minute store with signal+1h
entry, 96 bars, archived per-row geometry, trailing-profit logic, explicit
same-bar ordering and 1% round-trip cost. Maximum absolute deltas are
2.72e-8 for the soft target and 1.64e-9 for capture net; hit, stop, timeout and
first-touch bar match on all 48 rows. The recipe is therefore frozen and
reproducible.

Full-universe path materialization is also complete at
`data_perp/artifacts/febapr2025_native_first_touch_full_12h_paths_20260729_v2/`:
509,868 candidate inputs equal 509,868 shard inputs and 509,868 unique output
identities across 180 deterministic symbol/month shards. The completion gate
reverifies every input/output hash, reports no overlap, missing or extra ID,
and confirms that every missing-window manifest is empty. The payload is
native future OHLC only; execution-EV and policy-exit fields are explicitly
absent. The parity-proven recipe has now been applied and the matched base
retrain is complete.

The distinct native 12-hour labels are now complete at
`data_perp/artifacts/febapr2025_native_first_touch_full_12h_labels_20260729_v1/`:
180 of 180 label shards, 509,868 exact unique identities, no gap or overlap,
finite soft targets bounded in [0,1], and resolution exactly 12 hours after
decision for every row. No execution-EV field enters their construction.

The horizon change is material rather than a relabelled copy. Against the
archived 24-hour native target, Pearson/Spearman correlation is 0.8550/0.8724,
mean absolute soft-label change is 0.0784, 15.96% of rows change by at least
0.10 and 3.68% change outcome state. Mean target falls from 0.3034 to 0.2314.
Long changes more than short: 18.6%--18.9% of monthly long rows move by at
least 0.10 versus 12.5%--13.9% short. This confirms that a matched retrain is
necessary; rescoring the 24-hour model is not the requested ablation.

#### Final matched native-12h challenger and held-out divergence gate

The completed full-universe path/label artifacts are
`data_perp/artifacts/febapr2025_native_first_touch_full_12h_paths_20260729_v2/`
and `data_perp/artifacts/febapr2025_native_first_touch_full_12h_labels_20260729_v1/`.
They cover exactly 509,868 identities across 180 shards with no gap/overlap,
and every native label resolves at decision+12h. The 24-hour recipe parity
proof remains
`data_perp/artifacts/febapr2025_native_first_touch_24h_replay_parity_20260729_v1/`.

February is a legal native-12h fold. January was backfilled from the archival
ledger through the immutable 1m store at
`data_perp/artifacts/january2025_native_first_touch_full_12h_paths_20260729_v1/`
and `data_perp/artifacts/january2025_native_first_touch_full_12h_labels_20260729_v1/`:
222,773 identities in 79 shards, all decision+12h. Only January labels resolved
before 2025-02-01 train February. The frozen fold outputs are
`data_perp/artifacts/feb2025_native12h_base_oof_20260729_v1/` and
`data_perp/artifacts/febapr2025_native12h_partial_marapr_base_oof_20260729_v1/`.
All folds reuse frozen side-local 31/8 features and HPO and never read
execution-EV training fields.

The identical-row score comparison is
`data_perp/artifacts/febapr2025_native12h_matched_score_divergence_20260729_v1/`.
On all 509,868 February--April rows, the new 12h score improves its intended
native-12h rank metrics: pooled Pearson/Spearman IC 0.1989/0.1683 versus
0.1952/0.1609 for the old score; timestamp-local Spearman 0.1742 versus
0.1725; symbol-neutral Spearman 0.1703 versus 0.1642. Conversely, against the
native-24h target, the old score retains Spearman 0.1752 versus 0.1632 for the
new score. This is a horizon-specific improvement, not a universal score win.

For one pooled global top-10% book, the new score improves native-12h tail
economics slightly: 121.45 bps gross, 100 bps native cost and +21.45 bps net,
versus 119.28/100/+19.28 bps for the old score. Native-12h oracle-tail
overlap/recall is 20.67% versus 20.24%, and positive-net precision is 48.10%
versus 47.93%. The artifact records these IC and tail metrics for every
month/side slice as well as the pooled book.

Only after that OOF freeze, the exact signed execution-EV labels were joined at
`data_perp/artifacts/febapr2025_native12h_execution_ev_divergence_20260729_v1/`.
The immutable provenance contract is
`data_perp/artifacts/febapr2025_native12h_divergence_provenance_20260729_v2/`:
it records SHA256 hashes for the native identical-row and held-out-EV sources,
the 509,868-ID hash (`fe6dfe0fd4054fa83b25178af1ccc8e45b2d247a0c92264cf64eccd51bb41daa`),
exact identity equality and the passed assertion
`gross - cost == net` at 1e-12 tolerance. Publication is atomic.
The global top-10% result reverses the native-tail result:

| Held-out exact execution-EV, global top 10% | Old 24h score | New 12h score |
|---|---:|---:|
| Gross | +37.63 bps | +33.12 bps |
| Cost | 100.19 bps | 100.17 bps |
| Net | **-62.56 bps** | **-67.05 bps** |
| Positive net fraction | **46.51%** | 44.50% |

Exact-EV top-decile net by month/side (new/old) is: February long
-71.83/-59.35 bps, February short -16.92/-20.13 bps; March long
-97.11/-91.33 bps, March short -42.44/-46.01 bps; April long
-51.23/-38.39 bps, April short -80.80/-82.33 bps. Short slices sometimes
improve, but the pooled global book and every long slice degrade. Do not
post-select a side/month subset after observing these outcomes.

**Decision: reject / no promotion.** The 12h retrain is a useful IC-versus-EV
diagnosis, but its global held-out exact net EV is 4.49 bps/trade worse. Retain
both frozen scores for the required workstream; do not replace the 24h base,
admit the 12h score to policy, or use either raw score as execution admission.
The next candidate must learn the gross opportunity/cost hurdle and be judged
through causal global mapping and one pooled constrained book.

#### Frozen-score failure attribution (diagnostic only)

The required attribution gate is now materialised at
`data_perp/artifacts/febapr2025_native12_execution_ev_failure_attribution_20260729_v4/`.
It joins only after the 509,868-score freeze: causal archived signal-time
features for feature contrasts, and transition/path fields strictly as ex-post
diagnostics. It preserves the single pooled global top-10% selection for each
score; it does not define a new policy. The manifest records all input hashes,
the same candidate-ID hash as the frozen divergence artifact
(`fe6dfe0fd4054fa83b25178af1ccc8e45b2d247a0c92264cf64eccd51bb41daa`), SHA256
for every Parquet output, and a detached SHA256 for the manifest. It also
asserts frozen-score versus population gross/cost/net equality and
`gross - cost == net`, each at 1e-12.

The paired top-decile table isolates the loss: 36,279 rows are selected by
both scores, 14,708 only by the new score, and 14,708 only by the old score.
The new-only rows average -110.42 bps net with 38.48% positive outcomes;
old-only averages -94.82 bps with 45.46% positive outcomes. Thus the observed
new-only minus old-only displacement is -15.60 bps net/trade (-15.68 bps
gross; cost is effectively unchanged at -0.08 bps; positive rate -6.98pp).
This explains the pooled deterioration as score replacement, not a cost change.

Failures are concentrated in full stops (100% non-positive for both scores)
and timeouts (new: 83.45% non-positive), while the new score also worsens
trailing-exit failures (26.74% versus 24.01%). Its deterioration is present in
every month and both sides, largest in April (-9.94 bps) and long (-8.25 bps).
The active-transition cohort is worse for the new score (-137.12 versus -93.68
bps), but it selects *fewer* of those rows (833 versus 1,244), so it cannot
explain the aggregate degradation.
Among observable pre-entry candidates, the cleanest replacement contrast is
lower `range_24h_pct` for new-only than old-only (-0.219 standard deviations),
then lower volatility z-score (-0.096 SD) and higher jump intensity (+0.074
SD). Within the new selection, failures likewise have lower range (-0.368 SD),
lower volatility z-score (-0.108 SD), and lower `trend_r2_24` (-0.094 SD) than
successes. `spread_proxy_abs_return_bps_robust_z` is numerically pathological
(means around 1e11) and must not be recommended from this analysis despite its
small standardized difference. The robust contrasts are range, volatility and
trend. These are hypotheses for future pre-entry features/regime gates, not
post-hoc admission rules: transition phase/event, exit reason, MFE, MAE, and
exit time remain diagnostic-only.

#### Strict historical execution-EV add/drop and mapping gate

The first strict historical execution-EV add/drop gate is frozen at
`data_perp/artifacts/historical_execution_ev_add_drop_gate_20260729_v6/`.
V3 and the interrupted V4 partial are invalid; V5 score ledgers are valid but
its candidate-ID turnover field is superseded. V6 is canonical. It verifies
140,682 exact residual identities, develops only on March rows whose labels
resolve before each feature-selection/HPO/inner-OOF cutoff, and leaves all
69,258 April rows untouched. Every one of 27 arms persists a checksummed
47,136-row chronological March inner-OOF calibration ledger and a
69,258-row April prediction ledger. The target assertion
`gross - realized cost == net` passes at 1e-10. Timing, MAE, target-price and
wait outputs are absent.

All reported books below are one pooled global April top 10% across both
sides, never a per-timestamp quota:

| Raw April arm | Gross | Cost | Net | Positive net | Decision |
|---|---:|---:|---:|---:|---|
| base only | -12.69 bps | 99.94 bps | -112.63 bps | 51.7% | reject |
| residual only | +67.20 bps | 100.34 bps | **-33.14 bps** | 45.5% | strongest core, still negative |
| base + residual | +67.13 bps | 100.34 bps | -33.20 bps | 47.4% | retain comparator |
| base + residual + competing risk | +69.74 bps | 100.35 bps | **-30.61 bps** | 48.5% | best raw mean; not reliable after mapping |
| base + residual + peak starter | +67.39 bps | 100.34 bps | -32.95 bps | 47.9% | near-neutral add-one only |
| base + residual + six-class | +38.80 bps | 100.19 bps | -61.40 bps | 46.1% | reject add-one |
| all non-timing inputs | +13.14 bps | 100.06 bps | -86.92 bps | 40.9% | reject interaction bundle |

The requested context-field ablations are also explicit. Relative to
base+residual, raw cutoff margin improves only +1.04 bps and its z-score
+0.53 bps. Timestamp z-score (-3.08 bps), timestamp rank (-2.25),
archetype-relative z-score (-4.99), rank decile (-2.08), and especially
candidate-group size (-59.93) degrade the raw tail. The full context bundle is
not additive. These are April diagnostic deltas, not promotion estimates.

The material conclusion is a gross-opportunity shortfall, not a variable-cost
problem. The strongest raw arms recover about 67--70 bps gross against a
stable realized hurdle near 100 bps. They improve on the all-candidate April
mean (-104.44 bps net), but none clears costs.

The companion mapping repair is immutable at
`data_perp/artifacts/historical_execution_ev_mapping_repair_20260729_v4/`.
It consumes only hash-verified V6 March inner-OOF and April ledgers, persists
all mapping scores/selection flags, and reports adjacent-hour/day selected
asset turnover rather than the invalid timestamp-unique candidate-ID
turnover. March raw-score-to-net Pearson reliability is below the diagnostic
0.02 threshold on long for every tested arm; short is only 0.027--0.047.
Pooled isotonic/ridge and side-local mappings usually collapse side capacity
and economics. Common-unit/hierarchical mapping restores cross-side capacity
but remains negative: base+residual -26.83 bps, peak -27.67, six-class
-30.57, residual-only -49.57, and competing risk -133.12. Reliability-gated
variants also remain negative (-35.82 to -59.03 bps).

**Decision: no execution-EV winner and no portfolio replay yet.** No mapping
is both cross-side comparable and economically non-degrading, and no raw or
mapped global book is positive after cost. Do not interpret the absence of a
constrained replay as missing evaluation: the predeclared admission gate
failed. The next active experiment must model gross opportunity, the
probability of clearing realized cost plus a train-only margin, soft hurdle
labels, and conditional win/loss magnitude separately. Only a positive,
reliable April winner may advance to the frozen portfolio constraints.

The first historical CatBoost taxonomy gate also produced an actionable
result. The inherited merged seven-class taxonomy is rejected because
`noisy_timeout_usable_mfe` has only 107 long and 68 short observations
(0.104%/0.066%), below the frozen support floors. The fast-clean plus
fast-early-drawdown merge itself is economically coherent: source-class exact
EV means are +2.337%/+2.046% long and +2.739%/+1.970% short.

A versioned historical-only six-class contract is now accepted at
`data_perp/artifacts/febapr2025_historical_catboost_six_class_gate_20260729_v1/`.
It predeclares the semantically aligned merge of
`noisy_timeout_usable_mfe` and `early_mfe_full_reversal` into
`mfe_reversal_or_timeout`: all 11,688 source rows reached at least 0.5R MFE
and ended non-positive. The explicit class order is immediate adverse, fast
realization winner, late breakout, slow grinder, MFE reversal/timeout and dead
timeout. Both side/month support gates pass. This mapping was selected by
path semantics, not held-out EV; EV is audit-only. The global production
seven-class contract remains unchanged, and the separate three-class
competing-risk challenger is not an alias for this taxonomy.

Historical CatBoost now requires a bounded adapter because the production
runner hardcodes the global seven-class order and a monolithic context. Do not
modify that production contract. The adapter must consume the sealed label and
PIT-context shards, preserve the six-class order, and run feature selection,
HPO, geometry and March--April OOF independently per side.

That historical adapter is now complete at
`data_perp/artifacts/febapr2025_historical_six_class_catboost_20260729_v3/`.
The earlier v1/v2 directories are invalid pilots and must not be used: their
geometry thresholds did not change any labels and/or their HPO was capped
before convergence. V3 uses:

- train-only two-fold chronological importance plus validation-permutation
  stability, selecting 48 features independently per side;
- a discriminating, support-gated 1.5/2.0/3.0R geometry sweep per side;
- six bounded 128-tree HPO arms per side;
- converged trial 4 on both sides (long best iteration 102, short 72);
- inner early stopping only on already resolved prior rows, followed by refit
  on all prior rows; outer March/April labels never choose iterations;
- atomic side-by-month OOF checkpoints and a combined 140,682-row OOF report.

Both sides select the 3.0R geometry. Strict combined OOF multiclass logloss is
1.6278, accuracy 0.3384, multiclass Brier 0.7679 and top-confidence ECE 0.0671.
Calibration is the principal failure: immediate-adverse probability is
underpredicted by 25.86 percentage points, while fast, reversal/timeout and
other rarer classes are materially overpredicted. This blocks direct action
use.

As a diagnostic only, raw probability of the three nominally actionable
classes (fast + late + slow) raises positive exact-EV rate from 39.01% on all
rows to 51.25% in the pooled global top 10%, and moves the median to +5.8 bps,
but mean net EV remains -74.52 bps. This is one pooled global book across both
sides/months, not per-timestamp selection. It is not a deployed EV mapping or
a portfolio result.

Keep the six-class probabilities as strict OOF research features, subject to
calibration and identical-row add/drop tests. Next compare the separate
three-class soft triple-barrier competing-risk classifier on the same context;
do not merge that challenger into the six-class taxonomy. Neither CatBoost arm
may enter the execution layer until causal mapping and exact global-top-`k`
economics are positive.

The separate competing-risk challenger is now complete at
`data_perp/artifacts/febapr2025_historical_competing_risk_catboost_20260729_v2/`.
Its discarded v1 selection accidentally admitted stored `__soft_tb_*` outcome
fields; this was detected before HPO or OOF, and v2 explicitly excludes and
tests against every such field. V2 uses the exact ATR-normalized 12-hour
contract: upper `max(1.5 ATR, 1.5%)`, lower `1.0 ATR`, timeout as the third
class, same-bar conflicts assigned adverse, and recorded ambiguous rows
retained at weight 0.35. Feature selection/MDA, HPO and OOF are per side and
causally purged.

On the same 140,682 strict rows, competing-risk logloss is 1.0168, accuracy
0.4562, macro-F1 0.4151, balanced accuracy 0.4685, Brier 0.6186 and ECE
0.0513. Favorable-first discrimination is modest (ROC-AUC 0.5615, AP 0.4550).
Timeout is easier to rank (ROC-AUC 0.7432, AP 0.2734) but badly miscalibrated:
mean predicted probability 0.2855 versus actual share 0.0792.

Raw pooled global top-10% favorable-first probability produces -71.46 bps mean
net EV, -16.97 bps median and 43.93% positive outcomes. On identical rows the
six-class actionable score produces -74.52 bps mean, +5.80 bps median and
51.25% positive outcomes. Score correlation is only 0.3983. Thus the
competing-risk arm is somewhat distinct but weaker in the selected tail.
Retain it as a separate calibrated context ablation; do not replace or merge
the six-class head, and do not admit either raw probability to policy.

The causal calibration/context ablation is complete at
`data_perp/artifacts/historical_path_head_causal_calibration_context_20260729_v1/`.
Its status is deliberately
`research_only_prior_internal_oof_non_nested_hpo`: the calibration data is
chronologically prior to each March/April outer fold, but the February
FS/HPO/geometry contract overlaps that internal calibration era. It is causal
with respect to the outer outcomes, not promotion-valid nested OOF. A promoted
version needs nested model selection or an earlier disjoint calibration era.

Temperature scaling improves probability metrics but not economics:

| Stream | Logloss raw -> calibrated | Brier raw -> calibrated | ECE raw -> calibrated | Global top-10 mean EV raw -> calibrated |
|---|---:|---:|---:|---:|
| six-class | 1.6278 -> 1.6002 | 0.7679 -> 0.7556 | 0.0671 -> 0.0243 | -74.52 -> -74.63 bps |
| competing risk | 1.0168 -> 1.0045 | 0.6186 -> 0.6148 | 0.0513 -> 0.0432 | -71.46 -> -69.37 bps |

Six/risk score correlation is 0.3983 and their pooled top-decile Jaccard is
only 0.1040, so the challenger is incremental in a statistical sense.
Nevertheless the diagnostic prior-OOF Ridge maps remain negative: the best
risk-only map is approximately -52.23 bps and the combined-context maps are
-66.80/-71.24 bps with worse hit rate. Do not add both merely because their
candidate sets differ. Calibration repairs confidence, not economic
magnitude or the cost hurdle.

The first two strict auxiliary-role OOF streams are now canonical at
`data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2/`.
V1 is machine-invalidated because decision-month validation admitted 36
boundary rows outside the exact residual population. V2 reuses only fitted
model objects after asserting identical training cutoffs, reference row counts,
selected features and HPO contracts; it discards every V1 prediction and
rescores the exact 140,682 residual identities. Its candidate-ID hash matches
the strict residual OOF hash.

| Starter role | Strict rows/support | Aggregate OOF result | Interpretation |
|---|---:|---:|---|
| meaningful-MFE probability (`peak_mfe_12h_atr.p_hit`) | 140,682 | ROC-AUC 0.5489, logloss 0.6883, Brier 0.2476, ECE 0.0086 | calibrated but weak event discrimination |
| peak MFE conditional mean | 74,123 hit rows | Spearman IC 0.5148, MAE 1.700 ATR, RMSE 2.169 ATR | strong conditional magnitude learning |

The fold pattern matters. P-hit AUC is about 0.534 in March and 0.563 in April;
April short reaches about 0.607, whereas March short is approximately 0.509.
Conditional peak IC is approximately 0.495 in March and 0.541 in April, with
side/fold values around 0.49--0.57. The architecture can learn how large the
move becomes after a meaningful event; the event classifier remains the
bottleneck for unconditional trade selection.

All five auxiliary-head families are now materialized as 13 decomposed role
models under the same exact-identity, per-role, per-side FS/HPO and purged OOF
contract. Seven roles cover the timing CDF and conditional hit/no-hit MAE.
Four final roles cover peak q80, legacy adverse-extreme timing, confirmed
adverse-trough timing and future slope; the original meaningful-MFE event
classifier and conditional peak mean complete the set. Completion means
strict OOF predictions exist, not that any role is promoted. No auxiliary role
is promoted without incremental exact execution economics.

The seven-role timing/MAE OOF panel is immutable at
`data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae/`.
It contains exactly 140,682 strict residual identities, and every output,
runner, source and fitted-fold checkpoint hash verifies. The separate
action-layer diagnostic is immutable at
`data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae_diagnostics/`.
It joins realized outcomes for evaluation only; those labels are explicitly
forbidden as model inputs.

| Timing/MAE role | Aggregate strict OOF result | Fold warning / use |
|---|---:|---|
| meaningful MFE by 2h | AUC 0.6107, Brier 0.1295, ECE 0.0103 | strongest action classifier; test only after global EV admission |
| meaningful MFE by 4h | AUC 0.5910, Brier 0.2000, ECE 0.0176 | useful CDF interval context, not a stand-alone admission score |
| meaningful MFE by 8h | AUC 0.5582, Brier 0.2439, ECE 0.0178 | weak incremental horizon |
| meaningful MFE by 12h | AUC 0.5486, Brier 0.2502, ECE 0.0412 | short-April AUC 0.6155 but short-March only 0.5160 |
| MAE event probability | AUC 0.5383, Brier 0.2483, ECE 0.0097 | short-April 0.6017; short-March 0.4918 |
| MAE conditional on hit | IC 0.1040, MAE 0.722 ATR | March-long IC 0.0132 is effectively absent |
| MAE conditional on no hit | IC 0.1970, MAE 2.163 ATR | more learnable and more useful for abstention/risk sizing |

The four independently trained timing probabilities violated cumulative
ordering on 14,493 rows (10.30%). Row-wise PAVA projection fixes every
violation and produces five coherent non-negative interval masses summing to
one. The correction is small on average (mean absolute change 0.000934) and
does not materially change AUC. Expected-MAE mixtures remain weak and
regime-dependent: using the projected 12h timing probability gives IC
0.0517/0.1176 on March/April long but -0.0333/0.0016 on March/April short.
Therefore retain the projected timing CDF, conditional no-hit MAE and fast-hit
score as **action-layer candidates only**. Do not feed timing, MAE, target-price
or wait actions into the execution-EV head.

The final four-role panel is immutable at
`data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_remaining_roles/`.
It contains the same 140,682 unique strict residual identities (70,341 per
side); all four prediction columns are finite, the output/runner/context/
checkpoint hashes verify and all eight role-month checkpoints are present.

| Final role | Aggregate strict OOF result | Side/month stability | Decision |
|---|---:|---|---|
| peak MFE conditional q80 | IC 0.4988; pinball 0.6753; 76.61% empirical coverage | IC 0.492--0.562; coverage 73.2--81.9% | strong magnitude ranker; recalibrate q80 coverage before use |
| legacy adverse extreme | IC 0.1226; MAE 3.699 bars | IC 0.081/0.110 in March and 0.161/0.233 in April L/S | benchmark only; semantics mix an extreme with stabilization |
| confirmed adverse trough | IC 0.0887; MAE 2.679 bars on 84,682 valid rows | IC 0.061/0.098 in March and 0.153/0.051 in April L/S | economically cleaner target but unstable; action-layer challenger only |
| future slope | IC 0.1842; MAE 0.434 ATR/hour | IC 0.150/0.209 in March and 0.162/0.223 in April L/S | robust supporting label; test as payoff/persistence context, not direct admission |

Peak q80 confirms that conditional opportunity magnitude is highly learnable,
but its aggregate 76.6% coverage is below the nominal 80% and the
long/short calibration differs. Pair it with the event probability and
calibrate the quantile causally before any expected-value product. Future slope
is the most stable of the final unconditional regressions and should support
the payoff-scale/exit-conversion decomposition. The confirmed trough target is
conceptually preferable to the legacy extreme, but its lower and
side-dependent IC means it has not yet earned replacement. Compare them by
incremental wait/abstention economics after global EV admission; never select
the legacy version merely for its higher aggregate IC.

### Exact native-base IC versus execution-EV divergence (Feb--Apr 2025)

The superseding immutable paired audit is
`data_perp/artifacts/historical_base_ic_ev_divergence_20260729_v4/`. It uses
the 509,868-ID frozen native-score/exact-12h-execution attribution panel and,
where present, the repaired v3 pre-entry gross/regime panel. Its manifest
verifies runner, source and report hashes. The primary selection diagnostics
are pooled global top-`k` within month, never per timestamp; any per-side tail
is explicitly labelled diagnostic-only and is not a side quota.

The quoted native-target IC sequence is exactly reproduced as the **long-side**
native-24h-target Spearman: 0.154984 / 0.161872 / 0.225896 in Feb / Mar / Apr.
The pooled two-side values differ (0.171 / 0.163 / 0.195); that pooling
difference caused the earlier apparent discrepancy. The audit asserts and
records both side-local and pooled metrics.

Despite positive native-target rank IC, the pooled global top-decile exact
execution result is negative: approximately -50.9 bps in February, -83.0 bps
in March and -58.4 bps in April. The audit persists native-target/gross/net
decile monotonicity, fixed global top-1/5/10/20% economics, side/asset mix,
exit/MFE/MAE/exit-horizon composition, positive-net precision and gross/net
oracle overlap. The failure is economic magnitude and adverse path conversion,
not merely absent rank correlation. In the selected long tail, exit shares are
nearly unchanged from Feb to Mar (trailing 50.6% -> 50.2%, timeout 27.6% ->
27.5%, full SL 21.8% -> 22.3%), but trailing payoff compresses +179.6 ->
+120.3 bps, timeout loss worsens -143.6 -> -177.7 bps, and trailing MFE falls
565 -> 462 bps. April improves through 59.0% trailing, 14.8% full SL and
+132.3 bps trailing payoff: a payoff-scale/exit-conversion regime shift.

Gross, cost and net Spearman can be nearly identical here without implying
cost predictability: Spearman only measures ranks, while realized cost is near
constant/monotone over much of the population and net is gross minus that cost.
Use cost dispersion and selected gross-versus-cost means to interpret economics,
not the apparent cost rank IC.

The repaired pre-entry regime panel covers no February rows and about 40% of
the broader March/April native-score population, so February has no imputed
regime strata and March/April strata are explicitly coverage-limited. Invalid
v1/v2 transition panels must never be reused. A separate full paired diagnostic
may supersede this compact audit, but must preserve the same source/hash and
global-selection contracts.

This divergence is now a first-class workstream, with a narrower hypothesis
than “the alpha model stopped working.” The evidence says ordering survives
while the economic conversion of a high alpha rank changes. The next matched
tests must therefore:

1. decompose each base-score decile into meaningful-MFE probability,
   conditional peak magnitude, adverse-path/MAE, exit type, time to exit,
   realized gross, cost and net;
2. measure month-to-month changes both at fixed score quantiles and after a
   causal score map, so rank deterioration is not confused with payoff-scale
   drift;
3. test payoff-scale and exit-conversion context as interactions with the
   frozen base/residual scores, not as replacements for them;
4. use the decomposed form
   `P(opportunity) × conditional gross magnitude − adverse/timeout risk − cost`
   and compare it against direct net regression on identical rows;
5. require improvement in global top-10% net EV, positive-net precision,
   oracle-tail recall and latest-fold coverage together. Higher rank IC alone
   is not a winning criterion;
6. diagnose the deployed exit rule at fixed alpha rank. If the optimal exit
   family differs by payoff-scale regime, expose an OOF exit-policy value
   feature or router candidate; do not silently change the realized-label exit
   policy; and
7. rerun the same paired attribution on older materialized months before
   assigning the effect to a one-off February/March transition.

The non-monotonic month relationship is itself a required investigation, not
an architectural assumption. Long-side base-target rank IC rises from
`0.154984` in February to `0.161872` in March and `0.225896` in April, while
the corresponding selected execution economics remain negative and do not
move monotonically with IC (approximately `-59`, `-91`, and `-38` bps in the
quoted direct execution-EV top deciles). “The base predicts alpha rather than
execution EV” explains why the levels need not match, but it does **not**
explain why better alpha discrimination coincides with worse economic
conversion. The causal conversion workstream must explicitly resolve this
paradox.

Add the following paired diagnostics on identical candidate IDs:

1. **Target-horizon bridge.** Measure the frozen base score against the native
   24-hour alpha target, exact 12-hour MFE, 12-hour gross policy return, cost,
   and 12-hour net policy return. Report rank IC and decile response curves for
   every bridge target. This distinguishes a 24h-to-12h horizon mismatch from
   an exit-policy or payoff-scale failure.
2. **Opportunity versus realization.** Within every fixed base-score decile,
   decompose expected net into opportunity incidence, conditional favorable
   magnitude, trailing conversion, timeout/full-stop/adverse probabilities,
   their conditional payoffs, and cost. A higher native-target IC accompanied
   by lower `P(opportunity)`, lower conditional payoff, or worse conversion
   must be visible directly.
3. **Rank-preserving month counterfactual.** Apply February, March and April
   causal score-rank-to-component maps to the same frozen monthly score
   percentiles. Compare (a) each month's ordering with a fixed reference
   conversion map and (b) a fixed reference ordering with each month's
   conversion map. This separates ordering quality from the changing
   economics attached to a rank.
4. **Selected-book composition control.** Recompute the pooled global top
   1/5/10/20% with deterministic candidate-ID ties, then hold side, asset,
   score-decile and candidate-group composition fixed by reweighting. Report
   how much of each month-to-month EV change is composition versus within-cell
   payoff change.
5. **Exit-policy conversion check.** On the same selected IDs, compare the
   frozen deployed exit policy with diagnostic fixed-time and oracle
   counterfactuals. These counterfactuals may identify unrealized alpha but
   must not replace the canonical realized label or silently change the
   policy.
6. **Formal change attribution.** For February-to-March and
   March-to-April, attribute the top-decile net-EV delta to opportunity
   prevalence, favorable magnitude, exit mixture, payoff conditional on exit,
   cost, and book composition. Persist the interaction/remainder rather than
   forcing an additive explanation.
7. **Learnability test.** For every component that explains a material share
   of the divergence, evaluate whether it is predictable from strictly
   decision-time inputs, including payoff-scale, exit-conversion and regime
   transition features. Explanatory hindsight variables are not eligible
   model inputs.

Required conclusion states are: horizon mismatch, rank-to-opportunity mapping
drift, opportunity-to-exit conversion drift, payoff-scale drift, selected-book
composition drift, cost drift, or a quantified mixture. Do not close this
workstream with “IC and EV measure different things”; it closes only when the
month-to-month economic delta is attributed and the material causal component
is either made learnable or shown not to transfer.

The most useful new labels/features are accordingly pre-entry estimates of
opportunity probability, conditional peak q50/q80, early adverse trough,
fast-hit probability, no-hit MAE and recent payoff-scale/exit-conversion
regime. Timing and target-price actions still remain downstream of EV
admission. This workstream succeeds only if those estimates recover economic
tail quality on a later untouched month; explaining the divergence is not
itself a promotion.

The completed `febapr2025_incremental_gross_regime_context_ablation_20260729_v3`
is a **Ridge diagnostic**, not the strongest CatBoost gross-hurdle continuation.
Its context winner is selected by inner MSE and its B3 risk+peak result is
-112.01 bps; it must not be substituted for the frozen CatBoost strongest
baselines (risk+peak direct-net common-unit -6.48 bps and risk direct-gross
common-unit -7.82 bps). The required next experiment is a separate CatBoost,
March-purged, economics-aware incremental-context ablation on precisely those
two baselines, with untouched April raw/common-unit/March-only/online-causal
mapping ledgers and no replay unless a reliable mapped or common-unit result is
positive.

### Bounded incremental gross/regime-context ablation

The corrected strict context panel has been tested incrementally at
`data_perp/artifacts/febapr2025_incremental_gross_regime_context_ablation_20260729_v3/`.
V1 is invalid because it omitted the frozen base OOF score from the base
architecture; v2 is invalid because it failed to hash the compact report.
Both carry machine-readable invalidation markers. V3 binds the corrected
design, runner, seven source files, exact 140,682-row identity, 18 prediction
ledgers and report. Its detached manifest hash verifies and the exact-lag,
fold-purge, pooled-global-selection and April-holdout suite passes 5/5 tests.

The bounded runner uses per-side March feature ranking and Ridge-alpha
selection, a March-resolved causal recent isotonic map and one pooled-global
April top decile. Timing, MAE, target price and wait actions remain excluded.

| Arm | April gross | Cost | Net | Positive-net |
|---|---:|---:|---:|---:|
| Base + residual | 27.54 bps | 100.14 | **-72.60** | 39.72% |
| + static core | 22.15 | 100.11 | -77.96 | 46.16% |
| + static regime | 24.82 | 100.12 | -75.31 | 46.81% |
| + exact 3h core deltas | 21.79 | 100.11 | -78.32 | 45.47% |
| + exact 12h core deltas | 22.28 | 100.11 | -77.83 | 45.51% |
| + compact transition-regime block | 23.69 | 100.12 | -76.42 | 47.34% |
| + competing risk | -3.33 | 99.98 | -103.31 | 40.25% |
| + peak probability/magnitude | 19.54 | 100.10 | -80.56 | 45.12% |
| + risk + peak | -12.07 | 99.94 | -112.01 | 37.24% |

No arm is positive or reliable, so portfolio replay is correctly `NOT_RUN`.
Static and transition context can raise binary positive-net rate while
worsening payoff magnitude, another instance of classification/rank quality
not clearing the economic hurdle. This is a rejection of these additions
inside this bounded Ridge architecture, not proof that the 20 causal context
fields can never help a cost-aware nonlinear head. The inner choice is based
on MSE, not the final global-top-`k` economic objective; do not use it to close
the broader gross/hurdle/context research question.

### Economics-aware CatBoost context continuation

The required strongest-setup continuation is complete at
`data_perp/artifacts/historical_execution_ev_catboost_context_continuation_20260729_v1/`.
It extends, without modifying, the two best frozen gross/hurdle-v2
configurations:

- competing risk + peak, direct net; and
- competing risk, direct gross.

For each it reruns an identical no-context control and incrementally adds
static core, compact static regime, exact 3h deltas, exact 12h deltas and a
compact level-plus-transition regime block. All 360 planned and actual fits
use side-local March-purged feature selection/HPO with the same economics-aware
objective as the frozen runner. April remains untouched. Raw, common-unit,
March-only and online causal-21-day ledgers are persisted, and selection is
one pooled global top decile. The sealed context, strict identity, accounting,
frozen-v2 sources, runner and all 25 outputs verify. The focused contract suite
also recomputes the global selected tail from the frozen April ledger.

| CatBoost continuation | No context | Static core | Static regime | Exact 3h | Exact 12h | Transition regime |
|---|---:|---:|---:|---:|---:|---:|
| Risk + peak / direct net, common-unit April net | **-12.33** | -25.39 | -22.39 | -25.12 | -42.36 | -40.00 |
| Risk / direct gross, common-unit April net | **-8.55** | -27.41 | -9.28 | -11.79 | -26.85 | -22.48 |

No context arm improves its matched no-context control, no raw or mapped
variant is positive and no mapping is both reliable and non-degrading.
Online causal mapping remains negative for every arm; latest-week common-unit
economics are also negative. Consequently `eligible_arms=[]` and portfolio
replay is correctly not run.

The rerun controls differ modestly from the larger frozen-v2 grid
(-12.33/-8.55 bps here versus -6.48/-7.82 bps there) because this bounded
continuation is a fresh stochastic refit. Context deltas must therefore be
read only against their matched rerun controls, not against the earlier
point estimates. On that valid paired basis, the tested pre-entry
level/transition context is rejected for these two strongest CatBoost
architectures. Do not respond by adding all 57 context fields or tuning on
April.

## 2026-07-29 causal alpha-rank-to-economics conversion and IC/EV attribution

Status: **implemented, source-separated and audited; the IC/EV paradox is
attributed, and no conversion map is promotion-eligible**.

### Materialized conversion evidence

The immutable input bundle is
`data_perp/artifacts/historical_score_economics_conversion_ledgers_20260729_v1/`.
It contains 2,934,844 rows across five ledgers that must never be pooled:

| Source family | Rows | Path/cost evidence | Role |
|---|---:|---|---|
| Canonical frozen base | 509,868 | exact 1m; current-spread counterfactual exact-policy cost | sole base promotion tier |
| Canonical frozen residual | 140,682 | exact 1m; same cost contract; strict residual OOF only | sole residual promotion tier |
| Reconstructed Jan--Apr 2025 | 771,494 | exact 1m; fee-only reconstructed two-layer OOF | diagnostic bridge |
| Reconstructed Oct--Dec 2024 | 696,236 | hourly; fee-only reconstructed two-layer OOF | older recurrence diagnostic |
| Historical May 2025--Apr 2026 | 816,564 | hourly; fee-only old55/current comparator | longer recurrence diagnostic |

Every row has a unique candidate identity, `signal + 1h` decision,
`decision + 12h` resolution, finite score/economics and exact
`gross - cost = net` reconciliation. Only outcomes with
`execution_label_end_utc < UTC-day snapshot` enter a map. The materializer
normalizes source `full_sl` to canonical `full_stop` while preserving the raw
exit reason. This repair is material: it restores 107,446 canonical-base and
28,187 canonical-residual stop rows to mutually exclusive exit/adverse-risk
accounting. Exactly one of trailing, timeout, full stop or adverse exit is
required on every row.

Implementation:

- `scripts/materialize_historical_score_economics_conversion_ledger.py`
- `scripts/run_causal_score_economics_conversion_mapping.py`
- `scripts/run_base_ic_execution_ev_change_attribution.py`
- `scripts/summarize_historical_causal_score_economics_conversion.py`
- `tests/test_historical_score_economics_conversion.py`

The focused semantic, causality, global-rank and attribution suite passes 7/7.

### Causal component map

The mapper uses the previous 21 resolved days and a side-local causal score
percentile/decile, then shrinks side x decile estimates toward side and global
priors. It produces:

- `P(gross > cost)` and the +25 bps hurdle sensitivity;
- conditional opportunity gross q50/q80 with support and fallback level;
- trailing, timeout, full-stop and adverse-exit probabilities;
- adverse and timeout risk;
- expected gross, cost, MFE, MAE and direct net; and
- an exit-mixture net explanation.

Direct expected net is the only possible admission challenger. Opportunity
q50/q80 and exit-mixture scores are diagnostics until separately validated.
All evaluation uses one pooled global top 1/5/10/20% with candidate-ID
tie-breaking, never a timestamp, side or asset quota.

The deduplicated verified summary is
`data_perp/artifacts/historical_causal_score_economics_conversion_summary_20260729_v2/`.
Eight distinct source/score experiments pass their causal and hash audits:

| Source / score | Raw pooled top-10 | Causal direct map | Delta | Opportunity AUC |
|---|---:|---:|---:|---:|
| Canonical base alpha | -59.58 bps | -74.18 bps | -14.60 | 0.537 |
| Canonical residual EV | -27.09 | -54.90 | -27.81 | 0.505 |
| Reconstructed exact-1m base | -36.33 | -61.09 | -24.76 | 0.555 |
| Reconstructed exact-1m direct EV | -40.68 | -74.40 | -33.72 | 0.499 |
| Late-2024 hourly base | -3.66 | -26.42 | -22.76 | 0.585 |
| Late-2024 hourly direct EV | -41.85 | -28.56 | +13.29 | 0.545 |
| Historical hourly base | -43.26 | -35.28 | +7.98 | 0.518 |
| Historical hourly direct EV | -42.90 | -41.56 | +1.34 | 0.509 |

The few aggregate improvements are not reliable. The May-2025--Apr-2026 base
map improves five months and degrades seven; the direct map improves one and
degrades eleven. Both convert every raw positive month into a negative mapped
month. Several mapped monthly books are 100% one side. No canonical map is
positive pooled or by month, both canonical maps hit a 100% monthly side
share, and their opportunity AUC is below 0.55. Therefore
`survivors=[]`; portfolio replay is correctly not run.

### Why improving base IC coexists with worse EV

The exact paired attribution is
`data_perp/artifacts/historical_base_ic_execution_ev_change_attribution_20260729_v1/`.
It compares identical canonical IDs and the same top-10/global/cost/exit
contract.

The target bridge shows that the long-side native 24h-target IC improves, but
the conversion to the exact 12h policy target is much weaker:

| Long-side bridge | Feb | Mar | Apr |
|---|---:|---:|---:|
| Native 24h alpha-target IC | 0.1550 | 0.1619 | 0.2259 |
| Exact 12h MFE IC | 0.1051 | 0.1270 | 0.1878 |
| Exact 12h gross/net IC | 0.0904 | 0.0935 | 0.1432 |
| `P(gross > cost)` label IC | 0.0959 | 0.0946 | 0.1355 |
| Pooled-global raw top-10 net | -50.87 bps | -83.03 bps | -58.35 bps |

This is not solely a 24h-versus-12h horizon mismatch: 12h MFE and gross rank
IC also remain positive and improve into April. Rank correlation says which
rows tend to be better; it does not preserve the opportunity prevalence or
payoff attached to a rank.

The 100-bin rank-preserving counterfactual attributes February-to-March as:

- ordering/book-composition effect: **+0.34 bps**;
- rank-to-economics conversion effect: **-32.75 bps**;
- modeled change: -32.41 bps versus -32.17 bps realized.

Thus the ordering actually helps slightly while conversion collapses.
Three independent exact Shapley reconciliations of the same -32.17 bps
realized change show:

| Feb -> Mar lens | Mix/prevalence | Conditional payoff | Cost | Total |
|---|---:|---:|---:|---:|
| Opportunity | -36.72 | +4.39 | +0.16 | -32.17 |
| Exit | -5.14 | -27.19 | +0.16 | -32.17 |
| Side book | +0.28 | -32.61 | +0.16 | -32.17 |

March loses mainly because fewer selected rows clear the economic opportunity
hurdle and because payoff conditional on the realized exit compresses. Side
mix and the current-spread-counterfactual cost are negligible explanations.

March-to-April reverses for the same economic reasons:

- ordering/book effect: +1.56 bps;
- rank-to-economics conversion: +22.89 bps;
- opportunity prevalence/payoff/cost: +13.55 / +11.25 / -0.12 bps;
- exit mix/payoff/cost: +25.57 / -0.76 / -0.12 bps; and
- side mix/within-side payoff/cost: +1.36 / +23.45 / -0.12 bps.

April's recovery is primarily better opportunity/exit conversion and
within-side payoff. Its higher IC is useful, but it is not the cause of the
month-level EV recovery.

#### Global-tail follow-up: the IC/EV divergence is genuine

The required tail-depth investigation is now materialized at
`data_perp/artifacts/canonical_base_ic_ev_tail_diagnostic_20260729_v1/`.
It reads the audited full canonical panel
`canonical_opportunity_payoff_trust_panel_20260729_v2` and selects one pooled
global monthly top 1/5/10/20% with candidate-ID tie-breaking. It does not
rerank by timestamp or side.

Within the actually selected global top 10%, native-target rank IC improves
from 0.098 in February to 0.123 in March and 0.150 in April. Exact-net rank IC
also improves, from 0.040 to 0.068 and 0.089. Nevertheless:

| Month | Global top-10 net | `gross > cost` precision | Lift over month | Long share |
|---|---:|---:|---:|---:|
| February | -50.87 bps | 50.11% | 1.236x | 73.66% |
| March | -83.03 bps | 42.68% | 1.181x | 73.29% |
| April | -58.35 bps | 45.82% | 1.302x | 68.11% |

Thus this is not an artefact of quoting full-population IC while only the
tail deteriorates: ordering within the selected tail also improves. What
changes is the economic value attached to that order. The 100-bin
rank-preserving bridge is consistent at every tested depth:

| Global depth | Feb -> Mar ordering/composition | Feb -> Mar conversion |
|---|---:|---:|
| Top 1% | +6.88 bps | -18.90 bps |
| Top 5% | +0.87 bps | -29.83 bps |
| Top 10% | +0.34 bps | -32.75 bps |
| Top 20% | -1.62 bps | -19.68 bps |

The score-to-economics deciles remain strongly monotone: pooled net-decile
Spearman is 0.988/1.000/1.000 in February/March/April and opportunity-decile
Spearman is 1.000 in every month. Tail non-monotonicity is therefore not the
main failure. Score dispersion, candidate-group size, timestamp
concentration and side share are also broadly stable across February and
March. Asset concentration rises only modestly. The March loss is a
lower opportunity base rate and worse conditional payoff/exit conversion
throughout the high-score region, not a broken ordering at one cutoff or a
materially different selected-book composition.

One narrow result is economically positive but not sufficient for promotion:
April raw-base global top 1% is +16.62 bps, while February and March are
-8.90 and -20.46 bps. This is another regime-dependent tail result, not a
stable final admission policy.

Implementation and verification:

- `scripts/materialize_canonical_opportunity_payoff_trust_panel.py`;
- `scripts/run_canonical_base_ic_ev_tail_diagnostic.py`;
- `tests/test_materialize_canonical_opportunity_payoff_trust_panel.py`;
- `tests/test_canonical_base_ic_ev_tail_diagnostic.py`.

The canonical panel binds all 509,868 exact IDs to their retained same-run
31-long/8-short base validation matrices, exact 12-hour economics, pre-entry
state, and exact-gap 3h/12h regime transitions. Side-local minimum finite
coverage is 99.91% long and 99.99% short; transition coverage is at least
99.98%. Categorical causal-map fallback provenance is audit-only and is
represented by explicit Boolean indicators for model use. The superseded v1
panel is explicitly invalidated.

### Architecture decision and next ablations

Do not promote the current side-decile 21-day map as the final admission
score. Retain its component estimates, residuals, support, fallback level and
mapping error as recent context. The preferred next architecture is:

`base + residual alpha + CatBoost + auxiliary heads (parallel)`
`-> multi-task direct execution EV with opportunity and exit-payoff support`
`-> causal mapping-trust / residual-utility head`
`-> hierarchical side calibration with a pooled common-EV anchor`
`-> one pooled global top-k`
`-> separate timing / target-price / wait action layer`
`-> portfolio constraints`.

Required next experiments, in order:

1. Treat the improving-IC/worsening-EV divergence as an explicit open
   diagnostic, even though the first attribution already locates the
   February-to-March loss in rank-to-economics conversion. On identical
   candidate IDs and with one frozen global-top-10 policy, measure by
   month/side:
   - full-population rank IC versus top-tail IC, precision, lift and realized
     net EV;
   - score-decile monotonicity for MFE, `gross > cost`, gross, exit class and
     net, with uncertainty and effective support;
   - score dispersion/compression, cutoff turnover, timestamp/asset/side
     concentration and candidate-group-size effects;
   - fixed-order/current-economics and current-order/fixed-economics
     counterfactuals at top 1/5/10/20%, not only the existing 100-bin top-10
     bridge; and
   - whether the divergence remains after matching months on opportunity
     prevalence, payoff scale, volatility/regime, exit-policy conversion and
     available candidate mix.
   The purpose is to distinguish a benign global-rank improvement outside the
   tradable tail from tail non-monotonicity, opportunity-base-rate drift,
   payoff/exit conversion drift, or selection-composition drift. Improving
   aggregate IC must not be treated as evidence that admission EV improved.
2. Train a strict OOF per-side opportunity classifier on the exact labels
   `gross > cost` and `gross > cost + 25 bps`, using decision-time base,
   residual, regime-transition, market-state, leaf-drift and score-context
   interactions. The current score-decile opportunity AUC of 0.505--0.537 is
   the canonical bottleneck.
3. Condition separate gross-magnitude and exit-payoff heads on that event,
   retaining direct net as the primary head. Compare multi-task shared
   representation versus separate heads on identical IDs.
4. Target **mapping change/trust**, not just state: predict the incremental
   utility of trusting base/residual rank using recent causal opportunity
   prevalence, exit-conditional payoff, transition probability/calibration
   error, horizon disagreement, market-state velocity and feature/leaf
   distance. Apply it as a residual correction or abstention score.
5. Replace independent side maps with hierarchical common-unit calibration.
   Hard-fail or fall back to the raw/common anchor when effective support is
   low, the latest-week residual is adverse, or predicted monthly side share
   exceeds 95%.
6. Test q50/q80 only as magnitude/support inputs. Their empirical coverage is
   close to nominal, but ranking them directly consistently fails; quantile
   learnability does not solve event-probability or cross-side calibration.
7. Add OOF exit-policy value/counterfactual features to determine whether high
   alpha is being lost by the deployed exit family. Keep the realized exact
   exit label canonical and keep routing downstream of EV admission.
8. Require pooled-global top-10 positive net, positive latest month/week,
   opportunity calibration, side balance and improvement over raw score
   together before portfolio replay. Aggregate EV improvement alone is
   insufficient.

Older exact/hourly families may confirm recurrence and train diagnostic
hypotheses, but must remain separate from canonical promotion evidence.

### Opportunity/payoff/trust ablation result: event learning improved, economics did not

The next two experiments in the IC/EV-divergence branch are now complete.
They use exact 12-hour labels, one pooled global top-k, and keep timing, MAE,
target-price and wait decisions outside the EV layer.

The residual-tier experiment is materialized at
`data_perp/artifacts/historical_execution_ev_opportunity_payoff_trust_ablation_20260729_v1/`.
It compares matched direct net, 0-bps and 25-bps opportunity
probability-times-signed-magnitude, four-exit probability-times-payoff,
direct-primary OOF stacking, a causal trust overlay and the frozen control.
Its per-side opportunity classifiers are materially better than the old
score-decile proxy:

| Exact April event | ROC-AUC | Average precision | Brier | Log loss | Mean calibration error |
|---|---:|---:|---:|---:|---:|
| `gross > cost` | 0.643 | 0.508 | 0.223 | 0.637 | -0.006 |
| `gross > cost + 25 bps` | 0.655 | 0.462 | 0.206 | 0.599 | -0.008 |

This resolves one question: decision-time features can predict whether a
candidate clears the economic opportunity hurdle. It does **not** resolve
admission. On untouched April, raw/common-unit global top-10 net is -6.48
bps for the frozen residual control, versus -50.26 for matched direct net,
-53.54/-66.19 for the 0/25-bps probability-times-magnitude arms, -55.36 for
the four-exit mixture, -80.61 for the OOF stack and -77.25 for the trust
overlay. Hierarchical calibration is worse and frequently collapses the
selected book toward one side. Classifier-only ranking is also negative.
Only the AUC gate passes; global economics, latest-week, side-balance and
beat-control gates fail. Portfolio replay is correctly skipped.

The full-base experiment is materialized at
`data_perp/artifacts/canonical_full_base_opportunity_ablation_20260729_v1/`.
It trains side-local CatBoost on 334,298 exact February--March development
rows, using five blocked complement folds with exact path non-overlap, and
holds out all 172,450 April rows. The execution-label cutoff is based on the
12-hour `execution_label_end_utc`, not the frozen base model's old 24-hour
label-resolution field. It tests:

- base score only (`S0`);
- score context (`S1`);
- compact regime levels and exact 3h/12h transitions (`R`);
- retained same-run 31-long/8-short base inputs (`B`);
- combined arms and a no-DAE/GMM sensitivity;
- hard 0-bps opportunity, hard 25-bps opportunity, existing soft labels and
  direct net; and
- fixed, compact and deep geometries, with HPO restricted to OOF-selected
  finalists.

An audit after the run found a nested-cross-fitting defect in the development
expected-net map. For held fold `h`, the mapper used predictions from every
other complement-CV fold; those other models can have been trained with fold
`h` outcomes. Mapped development economics were then used for arm and geometry
selection. `INVALIDATION.json` therefore invalidates the mapped development
metrics, the resulting winner selection and any promotion interpretation of
that winner. The per-row base-model OOF predictions and predeclared fixed-arm
April diagnostics remain usable, but April is no longer an untouched
confirmation of an independently selected winner.

With that qualification, no tested fixed arm passes the April economic gate.
The best raw challenger inspected is the 25-bps `S0` deep model at -65.18 bps,
still worse than the frozen raw base at -58.35 bps. Score context helps direct
net in the fixed diagnostic: raw top-10 improves from -79.58 bps for `S0` to
-69.15 bps for `S1`. Adding the compact regime block gives -69.81 bps, so it
is not incremental in that configuration. These are diagnostic effect sizes,
not promotion evidence.

The important failure is not merely weak event classification. Some regime
and base-feature arms raise top-10 opportunity precision to approximately
58%, yet lose 85--101 bps per selected trade. They identify movement without
identifying favorable payoff magnitude, loss severity or exit-policy
conversion. DAE/GMM effects change sign by target and are non-robust.
Development-to-April top-10 transfer across configurations is weak
(Spearman approximately 0.44), with many selected configurations losing
20--35 bps in April. Because development mapping was not nested, treat this as
supporting evidence of conversion/regime non-transfer rather than a clean
transfer estimate.

The architecture decision is therefore:

- retain base/residual alpha as the ranking spine;
- retain the improved opportunity classifier only as a supporting task or
  feature;
- reject the current probability-times-magnitude, four-exit mixture,
  independent hierarchical isotonic mapping, OOF stack and trust-overlay
  implementations;
- keep score context because it is incrementally useful for direct net, but
  do not promote it alone;
- do not admit DAE/GMM posterior or compact risk-summary fields unless a
  later paired add/drop test shows stable incremental economics; and
- do not run portfolio replay until pooled global tail, latest period,
  side-balance and beat-control gates pass together.

The next economic model should jointly estimate opportunity incidence,
conditional favorable payoff, conditional adverse payoff/loss severity and
exit-policy conversion, and optimize their common-unit net value in the
global tail. Regime-transition features should predict changes in those
conversion components, not only the probability that price moves.

### Active-transition economic policy gate (corrected exact-policy replay)

The superseding research artifacts are:

- `data_perp/artifacts/active_transition_canonical_exact_policy_sweep_20260729_v2/`;
- `data_perp/artifacts/active_transition_canonical_event_impacts_20260729_v2/`.

They cover 504,440 mapped-eligible February--April 2025 candidates, 2,113
source hours, 47 active-transition hours and 13 transition events. Selection
is one pooled global top 10% per score stream. The replay enforces eight
concurrent positions, one position per asset, two new entries per bar and 75%
maximum wallet allocation. Exact canonical gross, cost, net, exit minute and
exit class are used. The canonical net already contains the cost, so
`expected_friction_bps` is explicitly zero. V1 is superseded: its non-zero
expected-friction field affected replay ordering even though it was not
subtracted from net.

For raw alpha, the frozen baseline remains economically positive after
portfolio constraints: 1,146 trades, +8.16 bps mean trade return, +$4,235.86
net PnL, +42.36% compounded return, 0.0529 Sortino and -18.66% interpolated
maximum drawdown. Its 27 accepted active-transition trades span eight events
and average +107.98 bps, versus +5.75 bps outside transitions. Therefore an
active transition is not equivalent to model failure. Blanket exposure
reduction is rejected on this lineage.

The best observed raw-alpha threshold arm at lambda 1 removes 1,284 frozen
top-decile candidates and produces +$506.13 versus baseline, but it retains
the same 27 accepted active-transition trades across the same eight events.
Its improvement is primarily book composition outside accepted transition
trades, not demonstrated transition protection. The small raw risk-premium
lambda 0.25 improvement (+$222.32) is likewise same-cohort exploratory
evidence. Every arm now persists kept/removed/newly-added attribution so these
effects cannot be mislabelled.

The mapped-direct-net stream is an unstable negative control, not a promotion
candidate. Correcting expected friction materially changes its replay:
baseline is 175 trades and -$870.21, with 15 active trades from only two events
at -436.79 bps per trade. Full active-probability exposure reduction turns the
same cohort to +$496.82, and risk-premium lambda 0.25 to +$88.33, but the
evidence is only two accepted events and the grouped-OOF probability/lambda
grid share the evaluation cohort. These arms must not be selected or
generalized from this result.

The event-centred report freezes each global book before slicing `[-12h,
onset)`, `[onset, end)` and `[end, end+12h)`. Raw alpha has six economically
damaging events among 13; mapped direct has three among the eight events with
non-empty comparable windows. Mean damage is +164.3 bps for raw and +220.6
bps for mapped, but 90% event-bootstrap intervals for mean damage cross zero:
about [-13.6, +430.0] and [-76.4, +682.4] bps respectively. Destination
accuracy is 84.6%, rather than confidence alone.

Active-head threshold operating points on this overlap are:

| Threshold | All-event recall | Severe-event recall | False episodes / 30d | Raw damaging-event recall | Mapped damaging-event recall |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 69.2% | 75.0% | 8.57 | 83.3% | 100.0% |
| 0.50 | 46.2% | 75.0% | 3.84 | 50.0% | 66.7% |
| 0.75 | 38.5% | 75.0% | 1.56 | 50.0% | 66.7% |

All conclusions remain research-only. Active and destination scores are
grouped OOF, not chronological policy OOS; policy lambdas are assessed on the
same cohort; only 13 exact-policy events overlap; and the normalized-price
ledger lacks the exact intratrade path, so MTM drawdown and Sortino use the
shared replay interpolation.

Next transition-policy gates are:

1. generate chronological active/destination OOS probabilities and freeze
   lambdas before the final event block;
2. use leave-one-event and calendar-block evaluation with event bootstrap,
   latest-month coverage and a declared false-alert budget;
3. keep transition risk separate from current model-health failure and test
   their interaction explicitly;
4. advance raw-alpha trust/threshold challengers only if gains occur on
   detected damaging events and survive replacement attribution;
5. retain mapped-direct exposure reduction only as a diagnostic of
   lineage-dependent transition sensitivity; and
6. keep timing, MAE, target-price and wait actions in the downstream action
   layer.

The IC-versus-execution-EV divergence remains a linked first-class workstream,
not a resolved architectural slogan. The paired audit and change attribution
above already show that February-to-March deterioration is almost entirely
rank-to-economics conversion (-32.75 bps) rather than ordering/book composition
(+0.34 bps), while March-to-April recovery is again conversion-led. Continue
the specified horizon bridge, opportunity/realization decomposition,
rank-preserving month counterfactual, exit-policy conversion check and
decision-time learnability tests. Transition features should be evaluated as
predictors of opportunity prevalence, payoff scale and exit conversion; a
higher base-target IC is never sufficient evidence of better tradable EV.

### Chronological active-transition validation and frozen April policy gate

The grouped-OOF limitation has now been narrowed materially. The superseding
active-head artifact is
`data_perp/artifacts/regime_transition_active_head_chronological_oos_20260729_v2/`.
It trains an expanding model for every evaluation month from January 2024
through July 2026 using the same 212 decision-time feature columns and model
configuration as the grouped head. A training row is legal only when
`max(source+12h, target__available_utc)` is strictly earlier than the
evaluation-month boundary. The panel therefore contributes 2023 history to
the first fold and expands thereafter.

Across 22,171 chronological OOS hours and 93 active-transition events:

| Metric | Chronological OOS |
|---|---:|
| Prevalence | 1.664% |
| PR-AUC | 0.3553 |
| ROC-AUC | 0.9632 |
| Brier | 0.01489 |
| F1 at 0.5 | 0.3650 |

This is at least as strong as the earlier grouped result (PR-AUC 0.3404),
which materially strengthens the claim that active transitions transfer
through time. It is still research-only because the upstream five-state
geometry and source research panel are pooled rather than production-causal.
The first chronological artifact,
`regime_transition_active_head_chronological_oos_20260729_v1`, has an explicit
`INVALIDATION.json`: its predictions and fold metrics are byte-identical, but
its event recall allowed a high approach-window score to count as active
detection. Only v2 event operating metrics are legal.

The v2 chronological operating curve is:

| Threshold | Active-event recall | False episodes / 30d |
|---:|---:|---:|
| 0.25 | 66.7% | 7.44 |
| 0.50 | 49.5% | 3.57 |
| 0.75 | 29.0% | 1.27 |

On the exact February--April economic overlap, the same chronological scores
reach PR-AUC 0.4554, ROC-AUC 0.9702 and Brier 0.01746 over 2,136 hours and 47
active rows. The event-centred mixed-validation artifact is
`data_perp/artifacts/active_transition_canonical_event_impacts_chronological_oos_20260729_v2/`.
At thresholds 0.25 / 0.50 / 0.75 it detects 76.9% / 61.5% / 38.5% of the 13
events and 83.3% / 66.7% / 33.3% of the six raw-alpha economically damaging
events. Severe-event recall is 100% / 100% / 75%. Destination accuracy remains
84.6%. The mixed grouped-destination report is superseded below by a
chronological destination report.

The corrected all-period chronological-score grid is
`data_perp/artifacts/active_transition_canonical_exact_policy_sweep_chronological_oos_20260729_v1/`.
It confirms the earlier qualitative result: raw-alpha active trades are
profitable, so blanket exposure reduction loses money. The raw threshold
lambda 1 arm improves aggregate constrained PnL by $582, but keeps almost the
same accepted active-transition economics; its gain is not proven transition
protection.

To remove same-cohort lambda selection, the policy runner now supports explicit
date windows, monthly economics and `prior_frozen` execution. The development
artifact
`data_perp/artifacts/active_transition_policy_development_febmar2025_20260729_v1/`
uses only February--March and selects `threshold_increase`, lambda 1:

| Feb--Mar raw-alpha policy | Net PnL | Compounded | Sortino | Max DD |
|---|---:|---:|---:|---:|
| Baseline | +$365.37 | +3.65% | 0.0092 | -17.33% |
| Frozen candidate | +$773.36 | +7.73% | 0.0174 | -17.36% |

The candidate is positive in both development months. The untouched
prior-frozen evaluation is
`data_perp/artifacts/active_transition_policy_frozen_april2025_20260729_v1/`;
it contains only baseline and the predeclared challenger:

| April raw-alpha policy | Trades | Net PnL | Compounded | Sortino | Max DD |
|---|---:|---:|---:|---:|---:|
| Baseline | 413 | +$3,281.72 | +32.82% | 0.1474 | -12.88% |
| Frozen threshold lambda 1 | 396 | +$3,549.58 | +35.50% | 0.1602 | -12.83% |
| Delta | -17 | **+$267.86** | **+2.68 pp** | **+0.0128** | **+0.04 pp** |

This is a genuine untouched-month aggregate improvement, but it does **not**
validate transition protection. Baseline active trades are already profitable:
19 trades across five events, +$474.42 and +108.97 bps/trade. The challenger
keeps 19 active trades and produces +$461.996 there; its improvement comes
outside true active hours (+$280.29), while active PnL declines by $12.43.
At candidate level it removes 1,020 rows averaging -78.23 bps, including 260
active rows, but portfolio capacity replaces their effect. It also misses 26
baseline accepted trades, including 12 profitable trades. Therefore the arm
is an economically useful **general risk-threshold candidate**, not evidence
that active-transition probability should control exposure.

Production promotion remains blocked by:

1. pooled upstream state geometry;
2. only six April events and five with accepted active trades;
3. no independent evidence that gains concentrate in economically damaging
   detected transitions; and
4. current-lineage model-health overlap still containing only one independent
   onset event.

Next work, in order:

1. extend the current-lineage 29-feature model-health panel backward until at
   least five transition events for a first grouped test and preferably
   60--100 economic-failure episodes for failure taxonomy;
2. test the interaction `active risk × current model-health deterioration`
   separately from either marginal signal;
3. freeze any next policy on an earlier event block and require improvement
   during detected damaging events, outside transitions, and overall;
4. retain active risk as a continuous premium/threshold context, never a
   blanket veto; and
5. keep cumulative hazard as a supporting onset score: its coherent grouped
   3h PR-AUC of 0.0992 did not beat the direct onset classifier and has not
   earned a hard gate.

### Chronological conditional-destination head and abstention

The superseding destination artifact is
`data_perp/artifacts/regime_transition_destination_chronological_oos_20260729_v1/`.
It uses expanding monthly folds over the older 2023--2026 panel. Training
labels must resolve before the evaluation-month boundary, and every event ID
present in evaluation is removed wholesale from training. It retains the
frozen 206-feature winner and five-state CatBoost contract from the grouped
ablation.

Across 573 chronological OOS rows from 81 independent evaluation events:

| Destination metric | Chronological OOS | Earlier grouped OOF |
|---|---:|---:|
| Balanced accuracy | 0.6060 | 0.612 |
| Macro-F1 | 0.5996 | 0.596 |
| Log loss | 0.7593 | 0.774 |
| Accuracy | 73.82% | not primary |

The near-identical grouped and chronological results establish that
conditional destination classification transfers through time on the pooled
research geometry. On the exact February--April overlap, 86 rows across all 13
events achieve balanced accuracy 0.9107, macro-F1 0.7885, accuracy 82.56% and
log loss 0.5005.

Confidence abstention is strongly monotone on the full chronological ledger:

| Minimum confidence | Row coverage | Accuracy | Macro-F1 | Covered events |
|---:|---:|---:|---:|---:|
| 0.00 | 100.0% | 73.82% | 0.5996 | 81 |
| 0.50 | 72.25% | 85.99% | 0.6874 | 73 |
| 0.60 | 61.78% | 89.83% | 0.7147 | 71 |
| 0.70 | 50.26% | 93.40% | 0.7494 | 61 |
| 0.80 | 31.76% | 97.25% | 0.7800 | 47 |

The fully chronological-component event report is
`data_perp/artifacts/active_transition_canonical_event_impacts_chronological_active_destination_20260729_v2/`.
It selects the highest-confidence prediction available no later than onset.
On its 13 events, raw destination accuracy is 84.62%. A 0.70 confidence floor
accepts 11/13 events and is correct on all 11; a 0.80 floor accepts 10/13 and
is also 100% accurate. Both errors have confidence below 0.65. The 0.70 floor
is therefore the leading abstention candidate, but must be frozen on an
earlier block before a later event evaluation; 13 same-cohort events are not
promotion evidence.

The focused chronological active/destination/policy/event suite now passes
23/23 tests. Both transition heads are chronological and label-purged at the
model level. The remaining shared blocker is the pooled upstream state
geometry, plus limited independent economic-event support—not time leakage in
the active or destination estimators themselves.

### Open investigation: improving base IC with deteriorating execution EV

Do not treat the February--April divergence as an expected or harmless
consequence of the layered architecture. The long-side base-target rank IC
improves from 0.155 in February to 0.162 in March and 0.226 in April, while
the corresponding **long-local top decile of the frozen base score**, measured
against exact execution economics, remains negative at approximately -59,
-91 and -38 bps. These are not direct execution-EV score results and are not
the pooled-global production selection rule; every later comparison must
declare score stream, target, side universe, population/tail, selection rule
and cost/exit contract. The base is intended to learn alpha rather than final
cost-aware admission, but that distinction explains only why the two metrics
have different levels. It does not explain why improved discrimination can
coexist with a sharply worse economic conversion.

The paired audit already narrows the February-to-March change: frozen ordering
and book composition contribute approximately +0.34 bps, whereas the
rank-to-economics conversion contributes -32.75 bps. The workstream remains
open because the causal, decision-time-predictable source of that conversion
loss has not yet been isolated. It must be investigated jointly with the
regime-transition work, while keeping transition risk and model-health failure
as separate hypotheses.

Required next diagnostics and ablations:

1. On identical candidate IDs, report by month, side and pooled global
   top-1/5/10/20%: native-target IC, 12-hour MFE IC, gross/net-policy IC,
   opportunity precision and lift, realized gross, cost and net EV.
   Split the rank evidence into full-population IC, top-20% and top-10%
   partial/tail IC, pairwise concordance inside the eventual admitted set, and
   the overlap/stability of the selected IDs. This directly tests whether the
   reported IC gain occurs mostly in economically irrelevant middle/lower
   ranks while ordering inside the tradeable global tail deteriorates.
2. Decompose each score quantile into `P(meaningful MFE)`, conditional MFE,
   early MAE, time-to-MFE, timeout/full-stop/trailing probabilities and
   conditional exit payoff. Attribute changes with uncertainty, not only
   point estimates.
   Also report the native alpha target's scale and dispersion within each
   quantile, plus its concordance with opportunity incidence and exact net
   return. This distinguishes better ranking of the native target from better
   ranking of economically realizable outcomes.
3. Hold ordering fixed while swapping monthly economic-conversion maps, then
   hold conversion fixed while swapping ordering. Repeat after matching or
   reweighting side, asset, volatility, candidate-group size, payoff-scale
   regime and transition state.
4. Test whether the deployed exit policy destroys otherwise valid alpha by
   comparing it, on the same selected IDs, with diagnostic fixed-horizon and
   oracle exits. These are explanatory counterfactuals only and must not
   replace the canonical realized label.
5. Test decision-time predictors of the conversion components, especially
   opportunity base rate, conditional payoff scale and exit-conversion
   quality. Include the materialized regime-transition features, active-risk
   probability and compact model-health features, with explicit interactions
   against the frozen base/residual score.
6. Separate calibration failure from ranking failure: compare raw scores,
   causal recent EV maps and common-unit mappings, and inspect whether mapping
   support or fallback changes around the economic deterioration.
7. Run the same attribution on older months and on July-only OOF blocks. A
   component is eligible for the model only if it transfers to a later
   untouched month or is routed by a causally identifiable regime.
8. Add a formal **IC-gain/EV-loss bridge attribution** for each adjacent
   month pair. Starting from the earlier month, replace in turn: score
   ordering, native-target distribution, alpha-to-opportunity conversion,
   opportunity-to-exit conversion, payoff conditional on exit, cost, and book
   composition. Report bootstrap uncertainty and the interaction remainder.
   The investigation is unresolved unless it identifies which replacement
   creates the sign/magnitude gap between the improving IC and deteriorating
   global-tail EV.

The leading hypothesis is opportunity/exit-conversion drift, not a collapse
of alpha ordering. Promotion nevertheless requires a model or causal router
that improves later-month pooled global top-k net EV, positive-net precision
and damaging-transition economics. Better rank IC by itself is never a
promotion criterion.

### First causal test of the base-IC/EV conversion hypothesis

The required before/after conversion data and decision-time context are now
materialized:

- `canonical_economic_conversion_transition_labels_20260729_v1`: 85,440 H3
  and H12 outcome rows at global hour x side x frozen score decile, including
  opportunity prevalence, conditional favorable payoff, adverse severity,
  exit incidence/payoffs and reconciled/direct mean net;
- `canonical_economic_conversion_transition_context_20260729_v1`: 42,720
  unique anchor cohorts with 47 whitelisted causal features covering frozen
  score context, market levels, exact 3h/12h pre-entry deltas and compact
  regime-transition composites; and
- `canonical_economic_conversion_transition_head_ablation_20260729_v1`:
  292,800 OOF predictions from five expanding chronological folds, purged by
  each row's actual target-availability time, with fixed model geometry and no
  HPO or feature selection.

For H=12h, opportunity-change IC/AUC/AP are 0.455/0.707/0.691 and model MAE is
0.1704 versus 0.1924 for the constant baseline. Adverse-severity change reaches
0.400/0.688/0.661 with MAE 0.00913 versus 0.00990. Direct/exit-mixture net
change is strongest at 0.473/0.716/0.707 with MAE 0.01216 versus 0.01382.
Favorable-payoff scale is weak: IC 0.144, AUC 0.580 and only a negligible MAE
gain. Direct mean net and reconciled exit-mixture net are identical by
construction here and must not be double-counted as independent heads.

This supports the hypothesis that the economic meaning of an alpha rank
changes with observable market context. It does **not** yet solve the
February--April divergence. Latest-fold performance weakens materially:
opportunity remains useful (IC 0.305, AUC 0.627 and MAE 0.1299 versus 0.1348),
but direct/exit net falls to IC 0.248/AUC 0.636 and loses the MAE baseline
(0.00829 versus 0.00809); adverse-severity MAE also loses its baseline.
Because the router is most needed in the latest regime, no transition head is
authorized for admission, trust gating or portfolio replay.

The investigation therefore continues with explicit feature-group ablations,
support-aware/smoothed and shared H3/H12 targets, a robust redesign of the
favorable-payoff label, and a joined counterfactual that holds frozen alpha
ordering fixed while applying OOF-predicted opportunity/adverse/exit
conversion. Success requires explaining and improving monthly pooled-global
top-k net economics—not merely maintaining aggregate transition IC. Only an
OOF prediction of conversion change may enter later as a bounded
score-by-conversion interaction; realized transition labels, static side
preferences and duplicate direct/exit heads remain prohibited.

### Conversion feature/target ablations and frozen-tail verdict

The feature-group experiment
`canonical_economic_conversion_transition_feature_group_ablation_20260729_v1`
contains 936,960 OOF predictions across eight fixed causal arms. It retains
side/score-decile identity controls in every arm and separates score context,
market state/deltas, regime levels and regime-transition deltas. No arm passes
the joint opportunity/direct-net gate. Market plus regime is strongest for
direct ranking (aggregate IC 0.484; latest IC/AUC 0.264/0.648) but still loses
latest MAE to the constant baseline (0.00822 versus 0.00809). Score plus regime
repairs latest MAE to 0.00795, but latest IC/AUC fall to 0.197/0.603 and
aggregate IC to 0.240. This is a real calibration/ranking trade-off, not a
geometry winner.

The immutable contribution artifact
`canonical_economic_conversion_contribution_labels_20260729_v1` adds:

- raw and robust unconditional upside contribution;
- raw and robust unconditional loss contribution; and
- soft net-positive rate.

Raw upside minus raw loss reconciles direct mean net on all 85,400 resolved
rows. In `canonical_economic_conversion_transition_target_ablation_20260729_v1`,
the old conditional favorable head remains weak (development/confirmation IC
0.147/0.092). Support weighting gives only 0.149/0.098, and empirical-Bayes
smoothing weakens development IC to 0.107. By contrast, raw unconditional
upside reaches 0.388/0.231 and robust unconditional upside reaches 0.391/0.246;
both pass the predeclared component gate. Unconditional loss reaches
development/confirmation IC 0.480/0.219 but fails confirmation MAE.

The soft net-positive-rate target is exactly equal to the existing
opportunity-0bps target on every complete H3 and H12 window. It is a useful
reconciliation check but must not be fitted, counted or reported as a second
independent head.

`canonical_base_conversion_prediction_attribution_20260729_v1` then joins
seven OOF conversion predictions to the unchanged pooled-global monthly base
top-1/5/10/20 books. Cohort identity is exact UTC hour x side x frozen base
score decile; no candidate is reranked. Within the base top 10%, candidate net
IC is near zero for every head, all March-defined high-versus-low daily-block
95% intervals cross zero, and the March-to-April +23.26 bps recovery is not
explained by movement into predicted states. Predicted-state composition
accounts for at most 4.54 bps in absolute value, while 22.45--27.80 bps remains
within-state conversion.

The immutable decision summary is
`canonical_conversion_transition_workstream_summary_20260729_v1`:

- zero feature groups advance;
- genuine passing supporting targets are raw and robust unconditional upside;
- zero high/low economic intervals exclude zero;
- no admission interaction and no portfolio replay are authorized.

The architecture implication is sharper than before. Broad conversion change
is learnable, but the current timestamp-side-decile target geometry is too
coarse for the candidates that enter the global alpha tail. Continue with:

1. shared H3/H12 auxiliary learning for opportunity, direct net, adverse
   severity and robust upside, with training-only scale normalization;
2. causal recent common-unit score/EV bands and high-alpha-tail contribution
   labels that match the global-book geometry without selecting per timestamp
   or side;
3. absolute mapping-support/admission-distance context; and
4. older pre-February plus later July exact history for a genuine later-period
   confirmation.

Only a head that significantly stratifies the unchanged frozen global tail may
advance as one bounded score-by-conversion interaction. The separate action
layer continues to own timing, MAE, target-price and wait decisions.

### Shared H3/H12 auxiliary learning

The bounded test is complete at
`canonical_conversion_shared_horizon_ablation_20260729_v1`. It uses
training-only per-horizon median/MAD scaling, equal H3/H12 loss mass, actual
target-availability purging and only a known horizon indicator. H3 outcomes
are auxiliary training rows; no same-anchor H3 label or prediction is exposed
to H12 inference.

No component passes the strict development-plus-confirmation gate. The result
is nevertheless informative:

- direct-net confirmation MAE improves from 0.00831 to 0.00784 and now beats
  the 0.00809 constant; IC improves 0.249→0.291 and AUC 0.636→0.663;
- opportunity confirmation MAE/IC/AUC improve
  0.12992→0.12744, 0.305→0.312 and 0.627→0.638;
- adverse confirmation IC/AUC improve 0.127→0.172 and 0.564→0.586, but MAE
  0.00667 still misses the 0.00659 constant; and
- robust-upside confirmation MAE/IC improve slightly, while AUC worsens.

The pooled model pays for these latest-fold gains with small development
MAE/IC deterioration and materially worse sign calibration; direct-net
development ECE rises from 0.021 to 0.051. Do not promote a fully shared head.
The only justified follow-up is a hybrid: shared H3/H12 continuous regression
with an H12-only sign/calibration classifier, using a bounded H3 loss-weight
grid chosen on development folds before one frozen confirmation. This remains
secondary to materializing a conversion label geometry that matches the
causal recent common-unit global book.

### Cross-era economic-failure transfer diagnostic

The immutable cross-era result is
`data_perp/artifacts/cross_era_execution_failure_transfer_20260729_v1/`,
implemented by
`scripts/run_cross_era_execution_failure_transfer.py`. It preserves the
lineage boundary explicitly:

- March--April 2025 is a reconstruction of the historical execution
  architecture with exact labels for that architecture, not a backcast of the
  current execution-EV model;
- May--July 2026 is the exact current execution-EV lineage; and
- only historical-to-current is a causal temporal transfer test.
  Current-to-historical is persisted solely as a non-causal reverse
  diagnostic. Within-era grouped OOF is research context, not walk-forward or
  promotion evidence.

The panels share 27 semantically comparable model-health fields.
`health__alpha_uncertainty_mean` is unavailable historically, and raw
CatBoost entropy is not comparable because the historical architecture has
six classes versus seven currently; both are excluded. A parallel health block
uses a strictly prior, within-era 21-day robust normalization with 72 hours of
minimum history. The feature grid compares market-only, raw health, normalized
health, market plus either health block, chronological active probability and
explicit active-by-health interactions.

Exact joined coverage is 1,064 reconstructed historical hours from 2025-03-12
through 2025-04-30 and 897 current hours from 2026-05-07 through 2026-07-07.
The current extended health panel reaches July 19, but the authoritative
global-top-10 exact-label contract has only four complete July hours in this
joined panel. This is **not July economic evidence**. The combined unjoined
label sources contain only 34 broad and 28 strict events, still below the
required 60--100 failure episodes.

Historical-to-current results:

| Failure label | Current prevalence | Market-only AP | Best health/context AP | Delta | Best block |
|---|---:|---:|---:|---:|---|
| Broad | 0.1360 | 0.2611 | 0.2698 | +0.0088 | market + raw common health |
| Strict | 0.0713 | 0.0967 | 0.1282 | +0.0315 | market + causal-normalized health |

The apparent gains are weak evidence:

- broad Brier changes only from 0.12248 to 0.12240;
- strict Brier slightly worsens from 0.07221 to 0.07230;
- active probability and active-by-health interactions do not win either
  label;
- within-current grouped OOF selects market-only for both labels, so the
  forward health improvement is not internally stable; and
- at one/two/four false alerts per 30 days, broad event recall does not improve
  over market-only. Strict recall at two alerts rises from 0% to 30%, but this
  is only three of ten events and cannot support a policy.

Reverse transfer is asymmetric. Normalized health improves broad AP from
0.2381 to 0.3004 when training on current and testing backward, but strict AP
does not improve. Because this direction uses future training data it is
diagnostic only. Together with the within-era results, it indicates that some
broad deterioration geometry recurs, but there is no stable, bidirectional
strict-failure classifier and no evidence for a hard regime router.

Consequences:

1. Keep market-transition risk and model-failure risk separate. Their
   interaction has now been tested and is not generally incremental.
2. Do not promote the small AP winners or use them as a veto. At most they are
   candidates for a continuous trust/threshold premium after independent
   event support exists.
3. Extend exact current economic labels—not only health rows—through a period
   with meaningful July global-top-k support. The existing July extension does
   not satisfy this.
4. Materialize older reconstructed periods with the same exact-hour,
   candidate-weighted contract until at least 60--100 broad/strict episodes
   are available. Preserve an era indicator and never pool reconstructed and
   current rows as one exact lineage.
5. Freeze the next failure model on earlier eras and require later-block event
   recall, calibration, risk-tail economics and overall/outside-failure
   economics together. Aggregate AP alone is insufficient.
6. Continue the IC-versus-EV work through pre-entry opportunity incidence and
   exit/payoff conversion features. The cross-era health result does not yet
   make that conversion regime observable.

The integrated active/destination/policy/model-health/exact-label/cross-era
suite passes 44/44 focused tests.

### July exact-failure support audit

The support limitation is now materialized rather than inferred at
`data_perp/artifacts/current_failure_label_support_audit_20260729_v1/` by
`scripts/audit_current_failure_label_support.py`. The audit does not change
the canonical policy.

There are 121,208 mapped strict model-OOS candidates in the combined
May--July history. The one pooled global top-10 selection has a mapped-EV
cutoff of +33.58 bps and selects 12,121 rows:

- 12,119 are outer OOF;
- only two are retired resolved forward OOS; and
- the cutoff itself is an isotonic plateau containing 366 rows, of which 249
  enter under the frozen candidate-ID tie break.

This is not missing-outcome censoring. The retired forward source contains
7,112 fully resolved candidates from July 11--19. Their mean mapped EV is
-59.96 bps and mean realized 12-hour net is -102.83 bps, so almost all
correctly rank below the combined global cutoff. Their mapped-score geometry
is also coarse: only 50 unique values across 7,112 rows.

A deliberately noncanonical upper-bound diagnostic selects the top 10% only
inside the retired-forward role. It is never eligible for training,
promotion, or policy comparison because it changes the combined global book.
Even that relaxed diagnostic produces only:

- 712 selected candidates;
- 23 complete exact hourly label windows, from July 12 10:00 through July 14
  11:00 UTC;
- one broad failure episode; and
- zero strict episodes.

Therefore the July evidence gap cannot be repaired safely by changing the
label materializer, selecting per timestamp/side, or applying calendar/regime
quotas. The current policy is mostly abstaining from the economically poor
forward cohort. More chronological strict model-OOS resolved history is
required. The large isotonic plateaus should be monitored and an economically
meaningful decision-time secondary tie-break may be ablated in future, but
changing the canonical candidate-ID tie-break is a policy change and is not
authorized by this audit.

The support audit adds two focused tests; the integrated workstream now passes
46/46 focused tests.

### Matched-control cumulative-hazard challenger

The remaining onset recommendation—learn impending transitions against
observably similar stable controls—is now tested at
`data_perp/artifacts/regime_transition_hazard_matched_controls_20260729_v1/`
by `scripts/run_regime_transition_hazard_matched_controls.py`.

Within every grouped training fold, all pre-onset rows are retained and stable
controls are matched on:

- exact current geometric state;
- a +/-90-day calendar neighbourhood;
- state age;
- peer-volatility decoupling;
- breadth dispersion;
- negative-breadth direction; and
- BTC-versus-ETH dominance momentum.

Matching uses labels only to construct the training sample. Validation remains
the full untouched grouped fold. Selected stable controls are reweighted
within state to preserve the full stable-row mass, so the challenger does not
redefine the class prior. Event groups never cross folds.

| 3h hazard | PR-AUC | ROC-AUC | Brier | Event recall @ 1/2/4 false alerts per 30d |
|---|---:|---:|---:|---:|
| Full-control baseline | 0.09923 | 0.79195 | 0.01470 | 11.92% / 16.56% / 22.52% |
| Matched-control training | 0.10202 | 0.77397 | 0.01477 | 13.25% / 16.56% / 19.21% |

At the primary two-alert budget, abrupt recall remains 18.95% and gradual
recall 12.50% for both arms. Matching modestly improves 1h/3h PR-AUC and the
tight one-alert operating point, but worsens ROC-AUC, calibration, the 12h
hazard and the four-alert operating point. It does not resolve the onset
bottleneck and does not authorize a veto or risk router.

The negative result is useful: generic control imbalance is not the main
reason the cumulative hazard trails the direct 3h classifier. Keep the
structurally coherent hazard as supporting context, but spend the next onset
budget on genuinely new pre-transition observables or independent
chronological event evidence—not further control-sampling HPO.

The matched-control challenger adds two focused tests; the integrated
workstream now passes 48/48 focused tests.

### Raw-feature shared-trunk direct execution-utility ablation

The requested direct multi-task execution-utility experiment is complete at
`data_perp/artifacts/canonical_raw_feature_direct_utility_multitask_20260729_v1/`.
Its verified decision summary is
`data_perp/artifacts/canonical_raw_feature_direct_utility_multitask_summary_20260729_v1/`.
The implementation and focused contract suite are:

- `scripts/run_canonical_raw_feature_direct_utility_multitask.py`;
- `scripts/summarize_canonical_raw_feature_direct_utility_multitask.py`; and
- `tests/test_run_canonical_raw_feature_direct_utility_multitask.py`.

The focused suite passes 15/15 tests. Both artifacts have detached manifest
hashes, and the recorded runner hash exactly equals the current runner.

This is the first experiment in this workstream that is simultaneously:

- a genuine raw-feature shared representation rather than a stack over frozen
  component-head predictions;
- side-local at model fit;
- trained on the exact 205,194-row February--April top-40/residual-gated
  population;
- primary-targeted on exact one-minute deployed-policy
  `execution_net_ev_12h`;
- purged by `execution_label_end_utc < cutoff`;
- selected on March only and refit before one April diagnostic score;
- ranked after the causal 21-day side-isotonic EV map with one pooled global
  top-`k`, never per timestamp or side; and
- scored only by its direct-net head. Auxiliary predictions are diagnostics
  and are never composed algebraically.

The exact calendar contract is:

| Role | Rows |
|---|---:|
| February pre-Feb-15 OOF training | 31,008 |
| February OOF prediction history | 32,256 |
| February resolved training for March | 63,264 |
| March predictions | 71,424 |
| March resolved arm-selection rows | 70,176 |
| February+March resolved final training | 134,688 |
| April reused diagnostic | 69,258 |

The raw matrix contains the frozen 237-field config-derived PIT universe plus
seven candidate-score context fields. Fixed add/drop blocks add chronological
active probability, sparse destination probabilities with availability,
four causal BOCPD fields, the matching 2025 raw-alpha health contract and
four predeclared transition-by-health interactions. The grouped cumulative
hazard artifact is explicitly excluded because its pooled upstream geometry
is not a chronological execution-model input. Current 2026 repaired-head
health is also excluded from this 2025 raw-alpha experiment.

The primary direct loss has weight 4.0. Exact economic auxiliaries use fixed
low weights: opportunity 0.25, favorable magnitude 0.20, adverse magnitude
0.20, deployed-policy MFE-to-exit conversion loss 0.20 and timeout 0.15.
The original hourly five-head family is tested only in one separately declared
0.05-weight cross-resolution regularizer arm. Those hourly targets are not
described or used as exact-policy EV components.

Twelve bounded configurations were compared: direct-only; full economic
multi-task; five economic add-one-outs; an hourly path-regularizer add-on; and
transition, health, transition+health and explicit interaction add/drop arms.
March selection uses the mean causal-mapped pooled-global top-5/10/20 net
return with a 5%--95% side-share support gate.

March selected `base_transition_health + economic_multitask`. The selection
table is:

| March mapped selection arm | Mean top-5/10/20 | Delta vs direct-only | Delta vs base economic core |
|---|---:|---:|---:|
| transition + health + economic tasks | -37.71 bps | +61.19 | +27.44 |
| health + economic tasks | -39.39 | +59.50 | +25.75 |
| economic tasks without favorable magnitude | -44.06 | +54.83 | +21.08 |
| economic tasks without conversion loss | -50.19 | +48.71 | +14.96 |
| transition + economic tasks | -53.66 | +45.23 | +11.49 |
| transition + health + explicit interactions | -64.39 | +34.51 | +0.76 |
| base economic multi-task | -65.14 | +33.75 | reference |
| without adverse magnitude | -72.31 | +26.58 | -7.17 |
| economic + hourly path regularizers | -77.55 | +21.34 | -12.41 |
| without timeout | -78.57 | +20.33 | -13.42 |
| without opportunity | -95.22 | +3.67 | -30.08 |
| direct-only | -98.89 | reference | -33.75 |

This gives useful task-role evidence, but every March aggregate remains
negative. Opportunity is the dominant useful regularizer; timeout and adverse
magnitude are also incremental. Favorable-magnitude and conversion-loss
regularization hurt the March direct tail under this shared geometry despite
their targets being individually learnable. The hourly path package also
hurts; it is not admitted to the direct execution model.

The exact winner-head diagnostics transfer from March to April as follows:

| Head | March | April |
|---|---:|---:|
| direct net | IC 0.0876; MAE 0.02579 | IC 0.0802; MAE 0.02617 |
| opportunity | AUC 0.5993; AP 0.4973; Brier 0.2339 | AUC 0.5996; AP 0.4581; Brier 0.2361 |
| favorable magnitude | IC 0.1060; MAE 0.01325 | IC 0.1525; MAE 0.01140 |
| adverse magnitude | IC 0.3101; MAE 0.01844 | IC 0.2477; MAE 0.01812 |
| exit-conversion loss | IC 0.2019; MAE 0.01961 | IC 0.2438; MAE 0.01848 |
| timeout | AUC 0.5722; AP 0.3172; Brier 0.2095 | AUC 0.5595; AP 0.3277; Brier 0.2015 |

Thus the auxiliary questions remain partly learnable and several transfer
cleanly, but the shared direct representation does not convert that
learnability into a robust admission tail. This is negative transfer/task
interference or an inadequate direct-utility geometry, not absence of signal
in every auxiliary.

The untouched April result rejects the March winner:

| Identical 69,258-row April global top-10 score | Mean net |
|---|---:|
| frozen residual expected EV control | **-24.32 bps** |
| frozen raw base score | -33.94 |
| frozen base expected EV | -54.30 |
| joint winner raw direct | -87.62 |
| joint winner causal-recent map | -88.97 |

The winner loses to the frozen residual by 64.65 bps. Its raw direct IC is
0.0802, while the recent mapped score has approximately zero IC (-0.0010).
The causal mapper is therefore harmful on this score stream in April; mapping
is not a harmless normalization layer.

Latest coverage confirms the failure. On the globally admitted April top-10,
the winner's final partial week is -245.85 bps versus -79.52 bps for the
frozen residual. The winner is negative in four of five April week
attributions and its only positive week is an all-long book. It fails the
positive-April, residual-non-inferiority and latest-week gates, so portfolio
replay is correctly not run.

Decision: **do not promote** the new shared-trunk winner, any context block,
the hourly path regularizer or the causal mapping on this score stream.
Transition plus matching-lineage health is a useful March diagnostic, not a
validated router.

The next bounded work should not reopen generic transition HPO. It should:

1. keep opportunity, adverse magnitude and timeout as the leading shared
   regularizers;
2. test favorable magnitude and conversion loss in task-specific adapters,
   detached-gradient branches or gradient-conflict control rather than forcing
   all tasks through the same trunk;
3. compare raw direct versus causal mapping with a frozen mapping-trust
   fallback, because mapping destroyed April rank information;
4. diagnose whether the direct loss needs tail-aware economics without
   outcome-derived sample weights, using February development and March
   selection only;
5. retain transition and health as add/drop context, but omit the explicit
   hand-built interactions that added no March value;
6. score the frozen next architecture on another untouched chronological
   period or older source-separated lineage before any policy replay; and
7. continue prospective economic-opportunity event packets until 60--100
   strict episodes exist. No supervised hard failure router is authorized at
   current support.

## 2026-07-29 exact global-book conversion workstream

The raw side/score-decile transition geometry has now been superseded for this
question by an exact causal common-unit global-book materialization:

- labels:
  `data_perp/artifacts/canonical_global_book_conversion_transition_labels_20260729_v1/`;
- decision-time context:
  `data_perp/artifacts/canonical_global_book_conversion_context_20260729_v1/`;
- direct global-book and broad EV-band heads:
  `data_perp/artifacts/canonical_global_book_conversion_head_ablation_20260729_v1/`;
- reconciled selected-book components:
  `data_perp/artifacts/canonical_global_book_reconciled_component_ablation_20260729_v1/`;
- paired day-block uncertainty:
  `data_perp/artifacts/canonical_global_book_component_bootstrap_20260729_v1/`.

The label artifact contains 21,130 exact pooled-global book rows and 20,890
causal EV-band rows over H3/H12. The source is the 504,440-row mapped-eligible
historical population. Selection is one sort across all sides, timestamps and
assets within each before/after H-window on `mapped_direct_net`, with
`candidate_id` ascending as the deterministic tie-break. There is no side,
timestamp or asset quota. The primary task is H12/global 10%; the 1/5/20/100%
fractions remain separately keyed diagnostics and are never model inputs.

The context artifact has 81 global-book and 56 EV-band decision-time features.
It contains causal mapped-score distribution, prior-21-day p90 margin,
coordinate availability, EV-band mass, market/regime-transition state and
trailing 3h/12h geometry. It contains no outcome, exit, MFE/MAE, timing,
target-price or wait-action field. `book_fraction` and `horizon_hours` are
keys, not features. An initially published timestamp-local top-k draft was
rejected and preserved only at
`canonical_global_book_conversion_context_20260729_v1_rejected_timestamp_local_draft`;
it must never be used. The canonical replacement contains no timestamp-local
selection fields and carries exact label-completeness/availability audit
columns.

Every OOF fit requires both complete windows, positive population/selected
support and actual `after_target_available_utc < validation_start_utc`.
The H12/global-10% task has 2,090 complete rows. Full validation folds contain
336 hourly anchors each; the final 86-row April-27--May-2 fold is truncated
and diagnostic only. No feature selection or HPO was used.

### Direct book versus broad EV-band result

The broad band-state problem is learnable in aggregate, but the exact global
book correction is not:

| H12 target/geometry | OOF rows | Model MAE | Constant MAE | IC | Sign AUC | Latest result |
|---|---:|---:|---:|---:|---:|---|
| Broad B0--B4 band conversion residual | 5,927 | 0.00950 | 0.01019 | **0.362** | **0.677** | full fold 3 IC 0.193 but MAE 0.00706 versus 0.00670 constant; truncated fold IC -0.027 |
| Exact global 10% book conversion residual, initial compact head | 1,430 | 0.01110 | 0.01043 | 0.008 | 0.515 | fails aggregate and latest gates |
| Exact global 10% book conversion residual, same-geometry depth-3 control | 1,430 | 0.01093 | 0.01040 | 0.044 | diagnostic only; still worse than zero/constant |

The broad band predictions are not additive selected-book contributions and
must not be summed into a book correction. They may become supporting context
only through nested chronological cross-fitting after a component
architecture earns that experiment.

### Exact reconciled selected-book components

For each H12/global-10% label, B1--B4 selected-book residual contributions
reconcile to the global conversion residual with maximum absolute error
`3.30e-17`. B0 has zero selected support/contribution and is not fitted.
Independent fixed depth-3/64-tree heads were trained for B1--B4 on global plus
matching band context, and their OOF predictions were summed.

| H12 model | MAE | Zero MAE | IC | Latest full-fold MAE | Latest zero MAE | Latest IC |
|---|---:|---:|---:|---:|---:|---:|
| Same-geometry global direct residual | 0.01093 | 0.01041 | 0.044 | 0.00911 | 0.00767 | 0.041 |
| Reconciled B1--B4 component sum | 0.01108 | 0.01041 | 0.053 | 0.00871 | 0.00767 | 0.032 |

Thus decomposition slightly improves some ranking/local stability but does not
beat the zero-correction baseline. B1 is particularly zero-dominated; B3/B4
carry most of the useful ranking signal. No reconciled sign probability is
claimed: averaging independent component sign probabilities is invalid.

The paired 2,000-draw UTC-day-block bootstrap confirms the failure rather than
treating overlapping hourly H12 rows as independent:

- on full development folds 0--3, the direct head is worse than zero by
  `+5.54` bps MAE-equivalent (`95% CI +0.83 to +10.12` bps) and the component
  sum by `+7.12` bps (`+0.66 to +13.15` bps);
- direct IC is 0.054 (`95% CI -0.055 to 0.170`) and component IC 0.058
  (`-0.069 to 0.190`); neither is established;
- in March the direct/component ICs are 0.104/0.014, while in April they are
  -0.004/0.140. The April component-minus-direct IC gain is 0.144 but its
  block interval still crosses zero (`-0.034 to 0.315`);
- April component top-minus-bottom exact direct-net spread is +47.32 bps, with
  a `-12.65 to +115.46` bps interval. This is a useful regime-sensitive clue,
  not promotion evidence.

### Decision and required next ablations

**Do not promote a global-book correction, component sum, broad-band stacker,
admission rule or portfolio replay.** The zero correction remains the
economic baseline. The evidence says broad EV-band state changes are
predictable, but the changing mixture and zero-inflated contribution of those
bands inside the exact globally selected book is the bottleneck.

The next bounded work is:

1. replace raw B1--B4 regression with a two-part component model:
   probability of non-zero selected-book contribution, then signed/conditional
   magnitude, with an explicit zero fallback and B1 allowed to remain zero;
2. ablate global-only versus matching-band context per B1--B4 and report
   regime-transition, mapped-geometry, base-score and market-context
   leave-one-family-out effects. Select no common feature family merely
   because it helps one band;
3. test shrinkage/partial pooling across B2--B4 and a reconciled direct
   residual head trained on component OOF predictions only through nested
   chronological cross-fitting. Do not use in-sample band predictions;
4. diagnose the March-to-April architecture reversal explicitly. Determine
   which B3/B4 regimes and transition features make decomposition useful in
   April but not March, and link this to the open base-IC/EV bridge
   attribution;
5. repeat on older source-separated materialized history and extend the exact
   current lineage through July. The current 56 development days and one
   truncated final fold are insufficient for a regime router;
6. retain day/non-overlapping-12h block uncertainty and the zero, resolved
   mean and causal-persistence baselines for every challenger; and
7. keep timing, MAE, target-price and wait actions in the separate downstream
   action layer.

### Zero-inflated component and regime-state follow-up

The two-part B1--B4 follow-up is complete at
`data_perp/artifacts/canonical_global_book_component_hurdle_ablation_20260729_v1/`,
with paired day-block uncertainty at
`canonical_global_book_hurdle_bootstrap_20260729_v1/`.

Two H12 component forms were tested with fixed depth-3/64-tree geometry:

- `P(nonzero) × E(signed contribution | nonzero)`; and
- `P(nonzero) × (2P(positive | nonzero)-1) ×
  E(abs(contribution) | nonzero)`.

Each uses global-only, matching-band-only or combined context. Insufficient
component/conditional support falls back exactly to zero. No HPO, feature
selection, broad-band stacking, action routing or replay was used.

| Reconciled H12/global-10% arm | MAE | Zero MAE | IC | Latest full MAE | Latest full IC | Full folds with positive IC/spread |
|---|---:|---:|---:|---:|---:|---:|
| Raw B1--B4 regression | 0.01108 | 0.01041 | 0.053 | 0.00871 | 0.032 | 4 / 2 |
| Band-only hurdle signed mean | **0.01052** | 0.01041 | **0.119** | 0.00839 | 0.021 | **4 / 3** |
| Combined hurdle sign+magnitude | 0.01052 | 0.01041 | 0.075 | **0.00802** | **0.067** | 3 / 3 |

The hurdle design repairs much of the raw-regression damage. On full folds
0--3, band-only improves MAE over raw by 5.93 bps (`95% day-block CI 0.87 to
10.79` bps), has IC 0.129 (`0.007 to 0.242`) and a +53.97 bps exact
direct-net quintile spread (`+5.19 to +93.94` bps). It nevertheless remains
1.18 bps worse than zero on MAE, with an interval crossing zero.

The context result changes with period:

- March favors band-only hurdle context: IC 0.172 and +86.20 bps direct-net
  spread.
- April favors the combined sign+magnitude arm: IC 0.084 and +58.90 bps
  direct-net spread, with MAE approximately zero-neutral.
- On the latest full fold, combined context is directionally more stable than
  band-only, but neither beats zero reliably.

The exact six-family confirmation is
`canonical_global_book_hurdle_feature_family_ablation_20260729_v2/`. Its v1
predecessor mixed current geometry with score/mapping state and is
superseded. V2 partitions the 55 band-only / 136 combined features into:
current geometry, score-and-mapping, core market level, transition dynamics,
regime composites and trailing geometry.

Only **band-local core market level** meets the strict incremental criterion
for the band-only hurdle: removing it worsens MAE and IC in all four full
folds and reduces exact direct-net spread in three. Mean fold effects from
dropping it are +2.77 bps MAE, -0.0546 IC and -30.36 bps direct-net spread.
March is especially dependent on this family: IC falls 0.172 -> 0.070 and
direct-net spread +86.20 -> +15.18 bps.

Transition dynamics are important for the combined hurdle—removal worsens MAE
in all four folds and IC in three—but miss the spread-consistency gate.
Several other combined families are substitutable or harmful in aggregate.
Removing score/mapping state improves aggregate MAE and IC even though it
helps isolated April spread slices. This is interaction/negative-transfer
evidence, not permission to select a post-hoc compact feature list.

The fixed pre-March unsupervised audit is
`canonical_global_book_hurdle_market_state_diagnosis_20260729_v2/`. V1 omitted
pre-validation assignments and is superseded. The state uses only 20
contemporaneous band-local market fields: five core market features across
B1--B4. Median imputation, robust scaling and `K=3` centroids are frozen
before the first OOF validation. No timestamp/month/fold, outcome, residual,
mapping support count, exit, MFE/MAE or recent performance enters the state.

The states are:

- `S0`, ordinary/lower range-volatility: 515 February fit anchors and 1,261
  later anchors;
- `S1`, high range/volatility/jump/chop: 88 February anchors and 180 later
  anchors across March and April; and
- `S2`, extreme range with elevated trend: 21 hours on one February day and
  **zero** later anchors.

`S2` is a non-recurrent training regime: it has an 80.95% self-transition
probability during its single episode and then disappears. `S1` is
economically interesting: band-only beats zero by 4.80 bps MAE overall, by
8.20 bps in March and by 0.15 bps in April. It still fails the support gate:
full-fold supports are 88, 16, 68 and 3 anchors. `S0` dominates support, where
both learned corrections remain worse than zero overall.

Decision: **no state router is authorized.** The separation is useful for
data collection, not control. Materialize older and July exact-book episodes,
measure whether `S1` and an `S2`-like extreme state recur, and rerun the same
frozen support/economics table. Only after at least three days per state in
each supporting fold and positive incremental performance in both earlier and
later periods may a nested soft router over `{zero, band-only, combined}` be
attempted.

## 2026-07-29 frozen transition stack, opportunity packets and completed IC/EV bridge

### Transition research is now version-frozen

The transition stack is frozen by
`configs/frozen_transition_research_stack_20260729_v1.json` and verified at
`data_perp/artifacts/frozen_transition_research_stack_audit_20260729_v1/`.
All seven registered source hashes and declared field contracts pass. The
registry preserves:

- chronological active-transition probability;
- chronological destination probabilities, confidence and entropy;
- exactly four BOCPD context fields;
- the grouped cumulative-hazard output as descriptive evidence only;
- separate historical and current compact health catalogs; and
- six pooled-geometry origin-state fields for descriptive packets only.

The consumer contract is fail-closed: transition signals are context rather
than controls; generic exposure reduction and admission vetoes are forbidden;
historical and current health lineages cannot mix; outcome/target fields
cannot enter context; and direct execution utility remains the sole rank
target. Generic onset HPO, more change-point families or state counts,
monolithic phase models, hard cross-era failure classifiers, manufactured
July support and algebraic probability-times-magnitude EV composition are
paused. The grouped hazard and pooled origin-state geometry are explicitly
not model-input eligible.

Implementation:

- `scripts/audit_frozen_transition_research_stack.py`;
- `tests/test_frozen_transition_research_stack.py`.

The fail-closed focused suite passes 3/3.

### Economic opportunity states and frozen event packets

The requested descriptive packet system is materialized at
`data_perp/artifacts/economic_opportunity_state_packets_20260729_v1/` by
`scripts/materialize_economic_opportunity_state_packets.py`.

It consolidates broad and strict anchors whose active windows overlap or are
separated by no more than six hours, but never merges historical and current
policy lineages. Only strict-containing incidents enter the packet index.
Each packet contains:

- frozen pooled-global selected-book composition;
- base, residual/direct-EV and mapped-score distributions where the lineage
  genuinely provides them;
- opportunity, favorable payoff, adverse payoff, timeout, MFE-to-exit
  conversion and cost components;
- active-transition, destination, BOCPD and compact health trajectories;
- descriptive origin-state metadata;
- a causal prior-only 30-day reference requiring 168 resolved hours;
- recovery observation for up to 72 hours; and
- a publication time after every included outcome has resolved.

The materializer recovered current MFE, MAE, exit and score fields by an exact
candidate-ID join to the matching current and retired-forward rich handoffs.
No proxy fields were manufactured. Every packet is resolution-frozen and all
source/output hashes verify.

| Packet support | Historical raw-alpha lineage | Current execution-EV lineage |
|---|---:|---:|
| Original strict anchors | 46 | 10 |
| Strict-containing consolidated incidents | 41 | 10 |
| Adequate causal reference | 37 | 7 |
| Insufficient early reference | 4 | 3 |
| Recovered within 72h | 41 | 9 |

The fixed multilabel taxonomy currently finds:

| State | Historical incidents | Current incidents |
|---|---:|---:|
| Sparse opportunity | 0 | 0 |
| Favorable-payoff compression | 0 | 0 |
| High opportunity / poor conversion | 0 | 0 |
| Adverse-payoff expansion | 1 | 1 |
| Timeout degradation | 0 | 0 |
| Exit-conversion failure | 0 | 1 |
| Execution/liquidity impairment | 0 | 1 |
| Mixed | 0 | 1 |
| Normal opportunity under fixed thresholds | 32 | 4 |
| Unclassified / inadequate reference | 8 | 4 |

This is an informative rejection, not a reason to optimize the thresholds.
The existing strict mapping-residual failure anchors do not map cleanly onto
large component-level opportunity shocks. In particular, many strict anchors
still look normal under the prior-resolved opportunity/payoff thresholds.
Therefore the current failure label cannot be silently relabelled as an
opportunity-state target. Keep continuous packet measures, inspect recurrence
prospectively, and revisit the event definition only with a predeclared
economic contract. Do not fit a supervised router from these 41/10 separate
lineage incidents or pool them into a fictitious 51-event exact lineage.

Implementation:

- `scripts/materialize_economic_opportunity_state_packets.py`;
- `tests/test_materialize_economic_opportunity_state_packets.py`.

The focused packet suite passes 6/6. The artifact explicitly authorizes no
classifier, transition control, admission veto or policy replay.

### Completed same-ID base-IC versus execution-EV diagnostic

The missing paired diagnostics are now unified at
`data_perp/artifacts/base_ic_execution_ev_paired_completion_20260729_v1/`,
implemented by
`scripts/complete_base_ic_execution_ev_paired_diagnostic.py`.
It uses all 509,868 canonical February--April IDs and freezes one
pooled-global monthly base-score top 10% with candidate-ID ties:

- February: 15,938 selected;
- March: 17,805 selected;
- April: 17,245 selected.

The unified bridge reports native 24h alpha, exact 12h MFE, gross, cost and
net IC plus decile response curves on the same IDs. The long-side sequence is
reproduced exactly:

| Long-side bridge | February | March | April |
|---|---:|---:|---:|
| Native 24h alpha IC | 0.1550 | 0.1619 | 0.2259 |
| Exact 12h MFE IC | 0.1051 | 0.1270 | 0.1878 |
| Exact 12h gross/net IC | 0.0904 | 0.0935 | 0.1432 |

This again rejects a simple horizon-mismatch explanation: the base score
improves against exact 12h opportunity and policy payoff too.

#### Same-book exit-policy counterfactual

Exact native 720x1m paths were joined to every one of the 50,988 selected
candidate IDs. Fixed-time and oracle results are diagnostic only; they do not
replace the deployed canonical label.

| Global top-10 exit | February | March | April |
|---|---:|---:|---:|
| Deployed policy | -50.87 bps | -83.03 bps | -58.35 bps |
| Fixed 1h | -57.15 | -62.64 | -67.38 |
| Fixed 2h | -52.25 | -54.42 | -55.93 |
| Fixed 4h | -43.07 | -53.75 | -44.72 |
| Fixed 8h | **-34.46** | -54.20 | -26.25 |
| Fixed 12h | -37.07 | -60.79 | **-6.55** |
| Oracle MFE less actual cost | +265.52 | +185.19 | +197.09 |

March is unusual: every fixed exit from 1h through 12h beats the deployed
policy, although none becomes positive. April strongly favors longer holding;
fixed 12h improves the deployed result by 51.80 bps and nearly reaches
breakeven. February favors roughly 8h. The appropriate exit horizon or exit
conversion is therefore state-dependent. This is diagnostic evidence for a
separate OOF action/exit-value layer, not permission to change the canonical
training label retrospectively.

#### Composition versus within-book economics

Joint reweighting holds side, top-20/other asset, score decile and fixed
candidate-group-size strata together:

| Change | Composition effect | Within-cell payoff effect | Common support |
|---|---:|---:|---:|
| February -> March | +0.03 bps | **-33.50 bps** | 100.0% / 99.1% |
| March -> April | +4.92 bps | **+19.90 bps** | 99.9% / 100.0% |

This closes the remaining composition objection. February-to-March failure is
not side, asset, score-decile or candidate-group drift; it is worsening payoff
inside nearly identical book cells.

#### Unified exact change attribution

The exact Shapley identity uses book composition, opportunity prevalence,
exit mixture, favorable/adverse exit-conditional gross payoff and cost, with
an explicit fallback/interaction remainder:

| February -> March contribution | bps |
|---|---:|
| Opportunity prevalence | **-35.90** |
| Favorable/exit payoff | **-22.27** |
| Exit mixture | +11.53 |
| Adverse/exit payoff | +12.62 |
| Composition | +0.54 |
| Cost | +0.05 |
| Fallback/interaction remainder | +1.26 |
| Realized total | **-32.17** |

| March -> April contribution | bps |
|---|---:|
| Opportunity prevalence | **+14.23** |
| Exit mixture | **+9.26** |
| Composition | +2.30 |
| Adverse/exit payoff | +2.24 |
| Favorable/exit payoff | -2.92 |
| Cost | +0.00 |
| Fallback/interaction remainder | -0.43 |
| Realized total | **+24.68** |

The paradox is now resolved descriptively. Better base ordering coexists with
worse trading when the opportunity base rate and favorable payoff attached to
that order collapse. April recovers through opportunity prevalence and exit
mixture, not cost. Rank IC is real but economically incomplete.

The existing decision-time conversion heads do not yet make this changing
surface tradable. Across their unchanged global-top-10 tails, mean net-rank
IC is between approximately -0.037 and +0.016, and no daily-block
high-versus-low interval excludes zero in every month. All remain
promotion-ineligible.

This remains an **active predictive-conversion workstream**, rather than a
closed explanation or an architectural excuse. The next experiments must test
whether decision-time information can predict the conditional conversion from
base rank to economic outcome:

1. fit separate OOF support targets for meaningful-MFE incidence,
   opportunity-conditional payoff scale, early adverse excursion and
   exit-policy conversion, then combine them only after their individual
   calibration and tail lift are established;
2. interact those support predictions with frozen base rank, cutoff margin,
   global candidate-group context and the causal regime-transition features;
3. compare a decomposed probability-times-payoff construction with a direct
   cost-aware EV residual, using identical rows and the actual pooled-global
   recent-mapped top-k selection rule;
4. use one shared regime-aware residual expert: remove prior-resolved broad
   regime conversion shifts with a soft, hierarchically-shrunk residual prior;
   train the candidate correction on invariant features, restricted causal
   regime interactions and regime-relative features; select it on worst-era
   transport; then apply only a small side x soft-regime common-bps calibration
   correction. Month-local/era-local experts and hard routing are superseded;
   and
5. require later-month net EV, positive-net precision, latest-month coverage
   and portfolio replay. An IC improvement without conversion lift cannot
   promote any model.

The first strict predictive-conversion implementation is now complete at
`data_perp/artifacts/strict_oof_ic_ev_conversion_ablation_20260730_v2/`,
implemented by
`scripts/run_strict_oof_ic_ev_conversion_ablation.py`. It freezes the same
exact-policy rows and 39 approved decision-time context fields, uses
per-side chronological OOF support heads with support-label availability
gates, and ranks one pooled global top 10% only after the causal recent-EV
mapping. Timing, MAE, target-price and wait actions remain excluded.

The result does **not** resolve the bridge:

| Pooled-global top 10% | May -> June pre-map | Later July pre-map | Later July after recent map |
|---|---:|---:|---:|
| Direct exact-net residual | -53.79 bps | -99.93 bps | -93.00 bps |
| Incidence/capture/adverse diagnostic | -71.91 | -86.24 | -105.29 |
| Complete four-state exit-policy EV | -68.65 | **-79.77** | -97.88 |
| 50/50 direct + complete blend | -57.87 | **-79.16** | **-89.03** |

The decomposition is incremental in later July before mapping, improving the
direct tail by approximately 20 bps, but it degrades the May-to-June control
and every arm remains negative after approximately 100 bps realized cost.
The current recent-EV mapping also materially worsens the May-to-June control,
so mapping is part of the bridge diagnosis rather than a fixed repair.

Support-head OOF metrics localize the remaining bottleneck:

- meaningful-MFE incidence ROC-AUC is only 0.517 in the May-to-June control
  and 0.557 in later July; later-July long is 0.529;
- conditional capture given meaningful MFE is materially stronger at 0.661
  and 0.746 pooled, respectively; and
- later-July favorable-first/adverse-first ROC-AUC is 0.569/0.546 pooled, but
  only 0.513/0.493 long.

Therefore the next bridge ablation must prioritize opportunity incidence and
long-side regime interactions, not merely add another conditional-payoff
regressor. It must also compare the recent mapping with identity/no-map and
causally re-estimated mapping arms on the identical global book. This artifact
is research evidence only and authorizes no promotion.

### July 20--23 exact frozen-stack completion

The previously missing July 20--23 population is now materialized end to end
under
`data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/`.
This is a retrospective, research-only completion and is not forward/OOS
promotion evidence.

The immutable lineage is:

1. 14,400 hourly candidates over the strict 75-symbol four-source universe,
   split 7,200/side;
2. the exact frozen 256-input side-specific AE/GMM states, whose 63 generated
   representation fields reproduce 177,394 historical contiguous
   symbol-hours with zero mismatches;
3. 5,760 Pack-B top-40 rows, split 2,880/side;
4. frozen clean-event, conditional Peak-MFE and seven-class CatBoost heads,
   followed by the frozen direct/capture final heads and causal 21-day recent
   EV mapping;
5. explicit side-parent decision geometry from the signed deployment policy,
   using the exact prior-hour raw Wilder ATR barrier; and
6. exact one-minute policy outcomes for every Pack-B row.

The one-minute store repair was append-only. For the 12-hour timeout arm,
402,540 required minutes are now covered across all 75 symbols; 361,427
previously absent rows were fetched, with zero failed or incomplete symbols.
The exact-label join admits all 5,760 rows with no subset and no imputation.
Costs are not duplicated: spread is embedded in gross execution and the
stored fee bridge is applied once to obtain net.

The hourly ATR audit exposed 1,211 historical missing hours across 55 symbols.
No selected signal or prior-hour decision bar is missing. These historical
gaps are retained rather than interpolated or as-of filled; Wilder ATR runs
chronologically over observed canonical bars so the first post-gap true range
includes the observed close jump. The geometry adapter fails closed if the
signal or prior-hour candle is absent. Its focused and downstream contract
suites pass 18/18.

Implementation and lineage:

- `scripts/materialize_execution_ev_frozen_decision_geometry.py`;
- `scripts/materialize_execution_ev_retrospective_geometry.py`;
- `scripts/materialize_execution_ev_policy_labels.py`;
- `data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/frozen_decision_geometry/`;
- `data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/geometry/`;
- `data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/labels_12h/`.

#### Exact score and portfolio economics

The canonical exact economics report is
`data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/exact_economics_report_v2/`,
implemented by `scripts/report_execution_ev_july_exact_economics.py`.
It reconstructs the one pooled-global top 10% with the frozen candidate-ID
tie break, uses actual policy exit times, and replays the signed deployment
portfolio constraints without adding friction to already-net outcomes.

| Frozen July 20--23 cohort | Rows | Mean exact net | Positive-net precision |
|---|---:|---:|---:|
| All Pack-B | 5,760 | -126.20 bps | 16.98% |
| Global mapped-EV top 10% | 576 | -156.38 | 20.66% |
| Mapped EV > 0 bps | 15 | -130.70 | 53.33% |
| Mapped EV > 25 bps | 9 | -16.86 | 55.56% |
| Mapped EV > 50 bps | 6 | -43.93 | 66.67% |

The >0 cohort is sharply side-asymmetric: ten long rows average -8.53 bps,
whereas five short rows average -375.06 bps. This is not adequate support for
promotion, but it is strong evidence that the current short transfer and
absolute calibration fail more severely than long ordering.

The config-faithful portfolio replay accepts 32/576 global-top-10 rows at
-160.84 bps mean net and -14.76% compounded return. It accepts 12/15 of the
positive-floor rows at -109.14 bps, 8/9 of the 25-bps-floor rows at -19.50
bps, and all 6 of the 50-bps-floor rows at -43.93 bps. The signed policy's
five-loss archetype guard is applied at side-parent granularity because the
retrospective geometry contains 100% explicit side-parent fallback; 262/576
top-10 rows are rejected by that guard. This is a faithful mechanical replay,
but it is not evidence for finer archetype-local cooldown behavior.

The mapped score is materially quantized: only 71 values occur. The global
top-10 cutoff is -80.35 bps; 338 rows are strictly above it and 384 are tied,
with the frozen candidate-ID rule selecting 238 tied rows. Outcome-only
best/worst tie assignments span -100.07 to -192.62 bps, and a deterministic
1,000-draw tie bootstrap has p05/median/p95 of
-155.05/-150.49/-145.57 bps. Every admissible tie assignment remains
negative, but the 92.55-bps range means the raw full-top-10 headline is not a
stable model metric. The 15 positive-floor rows are strictly above the cutoff
and are unaffected. A future mapped rank must preserve continuous
within-isotonic-bin ordering or use a predeclared decision-time secondary
score.

The first strictly historical tie-repair ablation is complete at
`data_perp/artifacts/mapped_ev_historical_oof_tie_repair_20260730_v3/`,
implemented by `scripts/ablate_mapped_ev_historical_tie_repair.py`. It selects
the secondary recipe only on 69,277 complete pre-July-20 temporal-OOF rows,
then evaluates the frozen choice once on July 20--23. Worst-month, mean-month
and pooled exact-net selection chooses `base_alpha` as the within-bin
secondary order. It does not change mapped EV levels, the -80.35-bps cutoff,
the 338 rows strictly above it, or the 15 positive-floor admissions.

Within the 384-row cutoff tie, it swaps 88 selected rows each way and improves
the July top-10 mean from -156.38 to -150.01 bps, a modest +6.37-bps lift.
The signed-policy portfolio replay changes from 32 accepted rows/-160.84 bps
to 30/-148.42 bps. This remains materially negative and is not a promotion
candidate. Keep continuous within-bin ranking in the architecture, but do not
expect tie repair alone to solve the conversion failure.

#### Exact per-head diagnosis

The per-head audit is
`data_perp/artifacts/july_exact_preentry_head_audit_20260730_v2/`,
implemented by `scripts/audit_july_exact_preentry_heads.py`. It uses the exact
raw Wilder ATR and barrier, defines meaningful MFE as at least
`max(1.5 ATR, 1.5%)`, and does not manufacture unavailable seven-class path
truth.

| Frozen score/head | Exact diagnostic |
|---|---|
| Base | pooled net IC +0.031; global top 10% -125.14 bps |
| Residual delta | pooled net IC -0.009; top 10% -151.73 bps |
| Base + residual | pooled net IC +0.036; top 10% -132.05 bps |
| Direct EV | pooled net IC +0.061; long +0.136, short -0.028 |
| Capture probability | positive-net AUC 0.639; long 0.666, short 0.588 |
| Recent mapped EV | pooled net IC +0.059; long +0.134, short -0.027 |

Conditional Peak-MFE magnitude remains learnable among actual meaningful-MFE
hits: IC is 0.356 pooled, 0.160 long and 0.469 short. The event classifier is
the bottleneck: AUC is only 0.532 and it predicts 45.45% incidence versus
35.23% realized. The resulting unconditional Peak construction reverses
sign, with exact Peak IC -0.099, net IC -0.102 and top-decile net
-156.95 bps. Conditional magnitude is therefore useful only behind a repaired
incidence and adverse-risk gate; it must not be treated as an unconditional
ranking feature.

The seven-class CatBoost surface is also non-transferable in this window.
Predicted favorable mass has meaningful-event AUC 0.464, while predicted
adverse mass has adverse-barrier AUC 0.413. The
`immediate_adverse_path` probability is positively related to realized net
economics, which is consistent with semantic inversion or severe regime
drift, not usable class fidelity. Retrain/recalibrate it per side on compatible
exact path labels or exclude it from the EV stack until the class semantics
recover.

No frozen head produces a positive pooled-global top-decile book on any of
the four decision days. Direct EV and capture preserve useful long ordering,
but they do not calibrate absolute EV; short direct/mapped ordering fails.
The next challenger should therefore be long-first, preserve the conditional
Peak magnitude head, replace its incidence/risk gate, and keep short and
seven-class CatBoost out until separately repaired.

The historical-to-current incidence challenger is complete at
`data_perp/artifacts/historical_to_july_meaningful_mfe_gate_challenger_20260730_v2/`,
implemented by
`scripts/run_historical_to_july_meaningful_mfe_gate_challenger.py`. It uses
134,880 historical rows whose labels resolve strictly before July 20 00:00,
249/249 causal fields shared with the current Pack-B surface, and side-local
nested temporal-OOF feature selection, HPO, class weighting and calibration.
All 16 final side models, admission thresholds and the adverse-risk weight
were persisted before the current exact outcomes were opened. The incomplete
`v1` directory is superseded and must not be used.

| Incidence gate on untouched July 20--23 | ROC-AUC | PR-AUC | Brier | Global top-10 net |
|---|---:|---:|---:|---:|
| Frozen clean probability | 0.532 | 0.378 | 0.240 | -129.71 bps |
| CatBoost hard clean-first | **0.599** | **0.439** | **0.234** | **-96.93** |
| CatBoost hard meaningful-MFE | 0.561 | 0.406 | 0.238 | -102.99 |
| LightGBM hard meaningful-MFE | 0.559 | 0.411 | 0.238 | -106.13 |
| LightGBM soft triple barrier | 0.518 | 0.374 | 0.238 | -120.05 |
| CatBoost soft triple barrier | 0.503 | 0.360 | 0.238 | -141.11 |

The hard clean-first classifier improves incidence ranking on both sides
(AUC 0.609 long/0.614 short), meaningful-MFE precision in its top decile to
52.95%, and exact net by +32.77 bps versus the frozen gate. It remains
miscalibrated high (mean probability 45.1% versus 35.2% incidence), and its
top decile is negative on every decision day. Hard clean-first is therefore
the correct target direction for the next gate, not a deployable admission
model.

Soft triple-barrier labels do not transfer here. All three soft-label model
families are weaker than the hard clean-first target; short logistic soft
reverses below chance at AUC 0.462. Keep soft labels as an optional
regularizer/secondary target, not the primary incidence label.

Every probability-times-current-conditional-Peak construction is worse than
incidence alone. The best risk-adjusted product remains -113.31 bps and raw
products range roughly -130 to -141 bps. The separately trained adverse-1ATR
gate is also non-transferable: AUC is 0.525 pooled, 0.521 long and 0.464
short. Its historical OOF weight degrades current economics. Therefore:

1. retain the conditional Peak magnitude model only as detached supporting
   information;
2. promote hard clean-first as the next incidence target candidate;
3. train a separate adverse-tail/payoff-conversion gate with explicit
   side-drift and minimum-coverage safeguards;
4. calibrate admission to absolute exact net economics rather than event
   probability; and
5. do not multiply the current Peak magnitude and incidence predictions.

Historically selected direct admission also drifts almost entirely to long.
CatBoost hard meaningful admits 137 rows (136 long/one short) at -60.86 bps
mean net and 41.61% positive; its best 14-row tail is still -21.10 bps.
The three-row positive LightGBM tail is too small and its full 22-row
admission averages -137.30 bps. No arm is promotion eligible.

The complementary drift/reliability audit is authoritative at
`data_perp/artifacts/two_stage_clean_first_drift_reliability_20260730_v2/`,
implemented by `scripts/audit_two_stage_clean_first_drift_reliability.py`.
Historical rule selection uses 134,354 complete pre-July-20 temporal-OOF rows
and evaluates the frozen result once on July.

The short side fails minimum coverage before any reliability overlay:
clean-first historical admission selects only 1/2,880 current short rows
(0.035%) versus the predeclared 1% minimum. That row realizes -431.2 bps,
does not reach meaningful MFE and does reach the adverse barrier. Long
coverage is 154/2,880 (5.35%) but still averages -104.4 bps. An abstention
rule cannot repair a missing base admission surface.

The historical-OOF reliability grid tests high adverse probability, low clean
probability, 249-feature OOD, and their blend at at least 70% monthly
coverage. It selects `none`: every abstention rule worsens historical
worst-month/mean economics. Do not add a deployed reliability veto from this
audit.

Short July outcome shifts versus historical OOF are large: meaningful
incidence -12.24 percentage points, positive-net incidence -18.69 points,
adverse incidence +2.24 points, mean net -23.96 bps and positive-payoff scale
-65.89 bps. Clean-probability calibration residual worsens by 7.15 points.
The 64-feature CatBoost leaf mix shifts more on short
(mean/max tree JS 0.205/0.568 versus 0.118/0.376 long), while unseen leaves
remain negligible. This is distribution/leaf-mixture drift, not novel-tree
support.

The dominant raw drift fields are breadth/high-volatility state,
`eth_btc_ret_24h`, negative/4h market breadth and cross-sectional
trade-size/depth liquidity. Use these as candidate regime inputs for
retraining or stratified experts. For now:

- require at least 1% per-side base admission coverage before any reliability
  rule is considered;
- shadow-monitor short adverse probability, raw 249-feature OOD fraction,
  clean probability and leaf-mix JS/support; and
- do not deploy a hard abstention rule until temporal-OOF economics supports
  one.

The two-stage absolute-net challenger is complete at
`data_perp/artifacts/two_stage_absolute_net_conversion_challenger_20260730_v2/`,
implemented by
`scripts/run_two_stage_absolute_net_conversion_challenger.py`. It consumes
the immutable clean-first stage-one predictions, trains side-local
pre-July-20 temporal-OOF positive/timeout hurdle and conditional
favorable/adverse/timeout/other payoff components plus a direct residual, and
selects the blend by historical worst-month then mean-month economics. The
conditional Peak head is explicitly excluded. The incomplete `v1` artifact
is marked failed and must not be used.

Historical selection chooses pure hurdle EV with zero direct-residual weight,
but even its historical June/July worst-month/mean top-decile economics are
-110.64/-99.97 bps. On untouched July 20--23:

- the positive-net classifier transfers statistically
  (AUC 0.655 pooled, 0.596 long, 0.678 short; Brier 0.147), but overpredicts
  26.4% positives versus 17.0% realized;
- its probability-only top decile is -141.51 bps despite 31.6% positive-net
  precision;
- hurdle absolute EV improves that to -131.59 bps, still worse than the
  clean-first incidence-only -96.93-bps book;
- the safeguarded surface contains only 16 long rows because short fails the
  1% coverage rule, and averages -201.68 bps; and
- absolute predicted EV > 0 admits one row at -3.31 bps, with no rows above
  25 or 50 bps.

Classification is no longer the only bottleneck: favorable-versus-loss payoff
scale and adverse-tail conversion remain unlearned. Do not promote the
two-stage hurdle, do not restore the current Peak product, and do not infer
economic quality from positive-event AUC. The next payoff work must be
tail-aware and regime/side conditional, with enough absolute-EV admission
support before portfolio replay. This artifact correctly skips portfolio
replay because it has candidate-local overlapping outcomes but no new causal
fill/order ledger.

The July-specific alpha-to-economics decomposition is persisted at
`data_perp/artifacts/july_alpha_economics_conversion_diagnostic_20260730_v1/`,
implemented by `scripts/diagnose_july_alpha_economics_conversion.py`.
It confirms that better within-window ordering can coexist with negative
economics because opportunity incidence, favorable/adverse payoff scale and
exit mixture move independently of rank IC. This reinforces the active
February--April conversion workstream; it does not close it.

#### Identical-row 12-hour versus production 24-hour timeout control

The full signed 24-hour policy horizon is also materialized on the identical
5,760 rows at
`data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/labels_24h/`.
The geometry and score selection are frozen; only the timeout is extended.
The hash-bound paired report is
`data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/exact_policy_timeout_ablation_12h_vs_24h_v1/`,
implemented by `scripts/report_execution_ev_timeout_ablation.py`.

Across the full population, 24 hours changes mean net by only -0.35 bps
relative to 12 hours. It worsens long by -12.41 bps and improves short by
11.71 bps. Positive precision rises from 16.98% to 20.50%, with 224
loss-to-win and 21 win-to-loss flips, but the opposing side effects cancel in
the mean. The frozen global top 10% improves by +2.12 bps but remains -154.27
bps; it contains 13 loss-to-win and two win-to-loss flips.

The small positive mapped cohorts look better at 24 hours:

| Frozen mapped cohort | 12h mean net | 24h mean net | Paired delta |
|---|---:|---:|---:|
| EV > 0 bps, 15 rows | -130.70 | -92.68 | +38.02 |
| EV > 25 bps, 9 rows | -16.86 | +46.51 | +63.37 |
| EV > 50 bps, 6 rows | -43.93 | +51.12 | +95.05 |

This is not a replicated horizon win. In the nine-row 25-bps cohort, eight
trades are unchanged; the entire improvement comes from one SNX long changing
from a -420.57-bps 12-hour timeout to a +149.72-bps trailing exit at 20.88
hours. Treat the apparent positive 24-hour tail as a single-trade result and
require older compatible periods plus a causal horizon/action head before
changing the deployed timeout.

The config-faithful portfolio comparison is also mixed. The global-top-10
accepted set grows from 32 to 47 rows; mean accepted net improves from
-160.84 to -111.51 bps, but compounded return worsens from -14.76% to -15.45%
because the longer horizon creates more exposure and a different loss/risk
trajectory. The 25-bps-floor replay accepts eight rows at both horizons and
changes from -19.50 bps/-0.48% compounded to +51.78 bps/+1.30%, again solely
because of the single SNX trade. Do not promote a 24-hour horizon from this
support.

The corrected older-period recurrence control is authoritative at
`data_perp/artifacts/historical_exact_policy_timeout_recurrence_may_jul10_20260730_v4/`,
implemented by `scripts/report_historical_exact_policy_timeout_recurrence.py`.
It binds 114,096 finite strict-OOF
`causal_recent_side_isotonic` rows from May 7 through July 10 and verifies the
same exact-policy 12-hour gross/cost/net targets row-for-row. The 12-hour and
24-hour policy files, canonical policy core, full strategy list and all 11
individual strategies are hash-verified.

SNX-style late recovery is recurrent in the broad population, not a unique
path: 5,936 rows over 117 assets and 2,754 symbol-days; the historical global
top 10% contains 705 such rows over 105 assets/352 symbol-days with +291.58
bps mean event delta. But longer holding has an opposing timeout-to-full-stop
tail, and its economic effect changes by cohort and month:

| Historical frozen cohort | Rows | 12h net | 24h net | Paired delta |
|---|---:|---:|---:|---:|
| All strict-OOF mapped rows | 114,096 | -109.36 bps | -108.53 | +0.83 |
| Global mapped top 10% | 11,410 | -100.61 | -98.80 | +1.81 |
| Mapped EV > 0 bps | 1,284 | | | **-14.96** |
| Mapped EV > 25 bps | 71 | | | **-13.37** |
| Mapped EV > 50 bps | 35 | | | **-43.02** |

For the global top 10%, May favors 24 hours by +10.64 bps, June disfavors it
by -13.41 bps, and the compatible early-July mapped support contains only five
rows. Most importantly, every positive mapped-EV floor is worse at 24 hours
historically. The one-trade current-July floor improvement therefore fails
replication. Keep 12 hours as the current timeout baseline and pursue only a
separate causal regime/path-conditioned action head that predicts late
recovery versus timeout-to-stop risk.

That action-head challenger is now complete at
`data_perp/artifacts/timeout_action_head_ablation_20260730_v3/`, implemented
by `scripts/run_timeout_action_head_ablation.py`. It uses 13 causal features
and side-local temporal-OOF models for 12h-versus-24h outcome delta, late
recovery and timeout-to-full-stop risk. None passes the historical OOF
economic gate:

| Router | Historical action coverage | Historical paired delta |
|---|---:|---:|
| Late-recovery classifier | 2.01% | -0.544 bps |
| Delta regression | 8.01% | -0.678 bps |
| Classifier/regression blend | 3.51% | -0.412 bps |
| Always use 24h | 100% | -11.53 bps |

On the frozen July global top 10%, the classifier changes only 7/576 rows and
adds +0.16 bps; regression and the blend degrade the result. The apparent
portfolio-wallet improvement for the classifier/blend is path-dependent:
only one accepted trade has its timeout changed, while the later accepted set
changes through cooldown and loss-streak state. It is not evidence of learned
timeout skill. Keep the 12-hour baseline and do not promote any router.

### Exact clean-first semantic repair and cross-era payoff materialization

The clean-event workstream previously mixed two questions in its current-July
audit: the model was trained on favorable-first ordering, but the reported
current AUC used eventual meaningful-MFE incidence because exact first-touch
ordering was not yet materialized. That substitution is now removed.

`scripts/materialize_july_exact1m_clean_first_labels.py` reconstructs the
`h12_u1p5atr` event directly from immutable one-minute paths. It applies the
same executable entry/exit half-spread adjustment as the February--April path
labels, uses the favorable barrier `max(1.5 ATR, 1.5%)`, the adverse barrier
`1.0 ATR`, and resolves same-minute OHLC conflicts adverse-first. It performs
no interpolation or as-of fill.

The current artifact
`data_perp/artifacts/july_exact1m_clean_first_labels_20260730_v1/` covers
5,760/5,760 rows. Exact favorable-first prevalence is only 19.38% long and
21.91% short; adverse-first/conflict prevalence is 73.99%/69.27%. The
identical-row probability audit at
`data_perp/artifacts/july_exact_clean_first_probability_audit_20260730_v1/`
changes the short conclusion:

| Scope | Exact clean-first AUC | PR AUC | Predicted mean | Exact prevalence | Score top-decile net |
|---|---:|---:|---:|---:|---:|
| Pooled | 0.577 | 0.256 | 45.1% | 20.6% | -96.93 bps |
| Short | 0.589 | 0.276 | 45.3% | 21.9% | -100.33 bps |

The previously reported short AUC 0.614 was against eventual meaningful MFE,
not the trained clean-first event, and must not be quoted as clean-first
performance. On the exact short target, Brier is 0.223, ten-bin ECE is 0.234
and top-decile clean precision is only 31.60%. The score retains weak ordering
but is severely overconfident and economically non-actionable.

The same exact-one-minute target has been materialized for all 134,889
May--July 19 training candidates at
`data_perp/artifacts/harmonized_mayjul19_exact1m_clean_first_labels_20260730_v1/`
using 156 atomic, identity-validated symbol shards. This repair is material:
the old hourly grid called 37.98% of rows clean versus 26.64% under executable
one-minute ordering; it created 15,419 false clean events and missed only 124
exact clean events. Future clean-event training and calibration must use the
harmonized target.

The hash-bound cross-era payoff input is
`data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3/`, implemented
by `scripts/materialize_cross_era_tail_payoff_dataset.py`. It joins:

- 205,194 February--April 2025 candidates;
- 134,889 May--July 19 2026 candidates;
- the common frozen 256-column causal pre-entry contract;
- exact spread-aware 12-hour gross, cost and net economics; and
- measurement-matched executable one-minute favorable/adverse/timeout events.

The resulting 340,083 identities are unique and every input/output hash,
feature value, economic label and resolution timestamp verifies. Candidate
group geometry still differs by era, so raw rank, raw cutoff margin and group
size are prohibited pooled inputs. Use percentile rank and explicit era/domain
interactions only.

The corresponding coherent payoff challenger is complete at
`data_perp/artifacts/cross_era_tail_payoff_challenger_20260730_v2/`,
implemented by `scripts/run_cross_era_tail_payoff_challenger.py`. It fits,
per side:

- one mutually exclusive four-class economic distribution: positive,
  adverse-negative, timeout-negative and other-negative;
- positive q25/q50 payoff;
- adverse q50/q85 loss;
- pooled-and-shrunk timeout q75 loss; and
- other-negative q75 loss.

All probabilities are calibrated from side- and era-local inner chronological
training blocks and renormalized to a simplex. Three feature arms compare raw
features, stable alpha context, and stable context plus explicit regime
composites under two bounded capacities. Five OOF blocks cover March/April
2025 and May/June/July 2026. Arm/HPO selection uses only historical causal
21-day-mapped pooled-global top-decile economics, then worst/latest month and
tail loss. Current scores and the model state were hash-sealed before July
20--23 outcomes were opened.

The historical winner is
`raw_context_regime__regularised_40`, but it is not profitable:

| Historical OOF selection metric | Result |
|---|---:|
| Global mapped top-10 net EV | -76.42 bps |
| Positive-net precision | 40.62% |
| Selected-tail CVaR5 | -772.24 bps |
| Worst/latest month | -171.56 bps |
| Four-class log loss | 1.144 |

The explicit regime arm is only modestly incremental: +2.17 bps versus the
best raw-only arm and +0.93 bps versus the best stable-context arm on the
aggregate historical top decile. It improves the latest historical month
relative to the comparable raw shallow arm, but July remains -171.56 bps.
This supports regime context as a small conditional modifier, not a hard
router or primary score.

The untouched current result rejects the challenger:

| July 20--23 exact result | Value |
|---|---:|
| Global top-10 rows | 576 |
| Net EV | -126.91 bps |
| Positive precision | 19.44% |
| CVaR5 | -430.97 bps |
| Side allocation | 576 long / 0 short |

The frozen per-head audit is materialized at
`data_perp/artifacts/cross_era_tail_payoff_challenger_head_audit_20260730_v1/`
by `scripts/audit_cross_era_tail_payoff_challenger_heads.py`. It does not refit,
rescore or recalibrate the challenger. On the exact July 20--23 labels:

- positive q25/q50 and adverse q50 have negative pinball skill on both sides;
- adverse q85 retains only +0.013 long / +0.181 short pinball skill;
- short timeout q75 is inverted (rank IC -0.193, pinball skill -0.564);
- long positive probability is 36.0% predicted versus 20.6% realized, while
  long adverse probability is 48.7% predicted versus 66.1% realized; and
- short positive probability is 11.4% versus 13.4% realized, while short
  adverse probability is 74.9% versus 64.7% realized.

The conditional payoff heads were learnable on parts of the historical OOF
set, but their transfer breaks by month and side. Positive/adverse central
payoff and short timeout magnitude must not be trusted as direct utility
terms; adverse q85 is the only current tail-magnitude component with residual
skill on both sides and remains a risk feature, not a standalone admission
score.

The causal mapping and allocation audit is frozen at
`data_perp/artifacts/cross_era_tail_payoff_mapping_flip_audit_20260730_v4/`,
implemented by `scripts/diagnose_cross_era_tail_payoff_mapping_flip.py`.
Historical OOF alone selects the mapping; July 20--23 is evaluated once using
only pre-July-20 resolved OOF labels.

| Historical-only mapping arm | Aggregate top-10 EV | Worst/latest month | CVaR5 |
|---|---:|---:|---:|
| Pooled isotonic (historical winner) | -74.64 bps | -136.06 / -136.06 | -596.92 |
| Side-shrunk K=2,000 | -76.42 | -205.63 / -136.06 | -772.24 |
| Raw tail score | -82.04 | -119.43 / -119.43 | -512.55 |

| Frozen-current arm | Allocation | Exact top-10 EV | Diagnosis |
|---|---:|---:|---|
| Raw tail score | 576 long / 0 short | -127.16 bps | All-long exists before mapping. |
| Side-shrunk K=2,000 | 576 / 0 | -126.91 | Side map amplifies long scores. |
| Pooled map | 308 / 268 | -122.11 | More balanced, but selects poor shorts. |

The side-shrunk map adds 69.14 bps on average to current long candidates and
-8.58 bps to shorts; its selected-long contribution is +234.12 bps. Support
is not sparse (20.8k--25.4k pooled reference rows/day and 0.83--0.87 side
weights), but 7--15% of current long raw scores exceed side-map support while
no shorts do. Endpoint clipping therefore amplifies an already broken raw
cross-side scale. The pooled map avoids that side shift but creates a
1,425-candidate cutoff plateau for only 576 slots, making candidate-ID
allocation unstable.

Historically unselected pooled-percentile and robust-z secondary ordering are
marginally worse historically (-74.66/-74.72 bps versus -74.64). Their much
better-looking single-current results (-41.41/-93.64 bps) are sensitivity
diagnostics only and cannot select a rule retrospectively. Every future
challenger must report raw and mapped global-tail economics, side allocation,
calibration, extrapolation/support and plateau/tie stability together. Use a
predeclared continuous secondary order and explicit out-of-support behavior;
do not treat stronger side shrinkage or current-selected tie-breaking as a
repair.

The all-long allocation is not learned short abstention. Raw and mapped
tail-EV top deciles are both approximately -127 bps and all long. Historical
side allocation also swings from mostly short in March/April, to mostly long
in May, to almost/all short in June/early July, then all long currently.
Absolute cross-side score scale and conditional-payoff transfer are unstable.

Current positive-event classification retains some sign information
(pooled AUC 0.636; long 0.623; short 0.645), but ranking directly by that
probability still loses -143.82 bps. Adverse-class AUC is only 0.519 pooled.
Most importantly, raw tail-EV rank IC is -0.020 long and -0.106 short on the
untouched current rows. The conditional magnitude/tail composition, not the
probability simplex implementation, is a binding failure, but raw cross-side
score scale, side-calibration inversion and causal-map extrapolation also
materially drive allocation; none can be waived as an implementation detail.

**Decision: reject / no promotion / no portfolio replay.** All coherent-mixture
score and mapping variants fail. The historical,
latest-month and current candidate-local gates all fail before the portfolio
gate. Preserve the coherent-mixture infrastructure and exact labels, but do
not deploy its score, side allocation or mapping.

### Direct exact-net quantile and severe-loss challenger

The isolated direct challenger is materialized at
`data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1/`
by `scripts/run_cross_era_direct_net_quantile_challenger.py`. It removes the
four-class payoff mixture and fits side-local exact-net q10/q25/q50/q75,
severe-loss probabilities at -100/-200/-400 bps, and conditional q75 loss
magnitudes for disjoint 100--200, 200--400 and 400+ bps buckets. It compares
raw, stable alpha-context and context-plus-regime inputs under two bounded
capacities and four historically selected score forms: q10, q25, q50 and
q50 minus severe expected loss. Quantiles and severe thresholds are projected
to valid row-wise order before scoring. Timing, MAE, wait and target-price
actions remain outside this admission head.

The historical OOF winner is `raw_context__shallow_24__q25`. It improves the
coherent mixture's aggregate result but still fails every economic gate:

| Direct q25 gate | Exact result |
|---|---:|
| Historical global top-10 net EV | -48.95 bps |
| Historical positive precision | 48.29% |
| Historical CVaR5 | -813.30 bps |
| Latest/worst historical month | -187.65 bps |
| Long-local historical top-10 | -56.45 bps |
| Short-local historical top-10 | -63.39 bps |

All five validation months are negative: March 2025 -37.24, April -108.23,
May 2026 -109.93, June -61.71 and July 1--19 -187.65 bps. Context is useful
relative to the other tested direct arms, but does not produce an admissible
tail. The severe-penalized median never beats q25, so subtracting the learned
severe expectation does not repair the direct utility score.

| Best historical trial per score form | Global top-10 EV | Latest month | CVaR5 |
|---|---:|---:|---:|
| q25 | -48.95 bps | -187.65 | -813.30 |
| q10 | -68.30 | -166.10 | -752.78 |
| q50 minus severe expectation | -70.01 | -175.64 | -781.18 |
| q50 | -80.86 | -118.19 | -879.68 |

The fail-closed post-run audit is at
`data_perp/artifacts/cross_era_direct_net_quantile_challenger_gate_audit_20260730_v1/`,
implemented by
`scripts/audit_cross_era_direct_net_quantile_challenger.py`. It verifies every
source hash and complete one-to-one current coverage: 5,760 predictions,
5,760 exact labels, zero duplicates, missing IDs or extra IDs.

| July 20--23 direct q25 | Allocation | Exact top-10 EV | Precision | CVaR5 |
|---|---:|---:|---:|---:|
| Raw q25 | 1 long / 575 short | -136.83 bps | 6.42% | -483.86 |
| Side-shrunk mapped q25 | 576 long / 0 short | -148.43 | 11.11% | -478.86 |
| Mapped long-local | 288 long | -127.36 | 11.46% | -379.82 |
| Mapped short-local | 288 short | -210.51 | 7.29% | -691.09 |

Every current decision day is negative (-59.76, -110.85, -104.33 and
-159.26 bps). Unlike the coherent-mixture score, whose raw book was already
all long, direct raw q25 is almost all short and causal side mapping reverses
it to all long. This proves that raw cross-side non-comparability and
side-local mapping can create opposite allocation failures; neither raw score
nor the present mapping is a safe fallback.

The dedicated mapping/allocation audit is frozen at
`data_perp/artifacts/cross_era_direct_q25_mapping_allocation_audit_20260730_v1/`
by `scripts/audit_cross_era_direct_q25_mapping_allocation.py`. Its historical
selection and current mappings were persisted before exact current labels
were loaded
(`frozen_before_current_evaluation.json`, SHA-256
`b9ece14f8e7abfac814000e0d86f258b6bf18c2b04750f9c58a11e78207b4697`).
The stored side-shrunk scores reproduce exactly on history and current data.

| Direct q25 mapping arm | Historical top-10 EV | Frozen current EV | Current allocation |
|---|---:|---:|---:|
| Side-shrunk (historical winner) | -48.95 bps | -148.43 bps | 576 long / 0 short |
| Pooled causal | -74.14 | -122.11 | 308 / 268 |
| Pooled + q25-percentile plateau order | -74.01 | -174.74 | 6 / 570 |
| Pooled + side robust-z plateau order | -74.02 | -152.89 | 160 / 416 |
| Raw q25 | -76.47 | -136.83 | 1 / 575 |

The pooled map is less negative on the single current period but was
materially worse historically and cannot be chosen retrospectively. It also
places 1,440 candidates on the 576-slot cutoff plateau. The historically
preferred continuous plateau key is only a marginal historical improvement
over plain pooled mapping and degrades current economics to -174.74 bps.
Predeclared continuous secondary order therefore solves deterministic
allocation, not payoff transfer. No mapping arm is promotable and none
authorizes portfolio replay.

Current q25 rank IC is -0.101 long / -0.054 short. Median q50 retains only
+0.013/+0.021 and does not recover tail economics. Severe-loss calibration
also transfers asymmetrically:

| Current severe event | Long predicted / actual | Short predicted / actual |
|---|---:|---:|
| net <= -100 bps | 28.9% / 51.5% | 46.5% / 35.1% |
| net <= -200 bps | 20.2% / 33.2% | 38.2% / 24.5% |
| net <= -400 bps | 4.7% / 6.1% | 7.6% / 7.4% |

The moderate-loss heads understate long risk and overstate short risk; only
the extreme -400-bps rate is approximately calibrated. The frozen v1 source
persisted calibrated probabilities only, so exact raw-versus-calibrated
probability attribution is unavailable for this run. The strengthened runner
now persists both, fails closed on identity coverage, hash-binds current
inputs, emits per-month/day/side economics and documents that its final
all-history probability refit is not exactly model-matched to the pre-July
calibrator. Future runs must either preserve prediction-model parity or
explicitly retain that approximation as an ablation.

**Decision: reject / no portfolio replay.** Historical aggregate, every
historical month, latest month, current global and both current side-local
gates are negative. The focused runner and gate-audit suites pass 10/10. The
next direct-utility experiment must first solve common-unit cross-side
calibration and moderate-loss transfer; a larger HPO or stronger side-local
isotonic map is not justified by this result.

The metric distinction is mandatory: the approximately -59/-91/-38 bps
figures are exact execution outcomes of the long-local top decile ranked by
the frozen base score. They are not the performance of a direct execution-EV
head.

The IC/EV divergence is nevertheless a first-class unresolved conversion
problem, not an expected result to waive away. The paired diagnostic at
`data_perp/artifacts/july_alpha_economics_conversion_diagnostic_20260730_v1/`
shows that the long base score improves not only against its own alpha target,
but also against exact downstream labels:

| Month | Base-target rank IC | Exact MFE rank IC | Exact net rank IC | Base-ranked top-decile exact net |
|---|---:|---:|---:|---:|
| February | 0.155 | 0.105 | 0.090 | -50.87 bps |
| March | 0.162 | 0.127 | 0.094 | -83.03 bps |
| April | 0.226 | 0.188 | 0.143 | -58.35 bps |

Therefore the contradiction is not that the base has no economically relevant
ordering. It is that modest monotone rank information does not identify a
profitable extreme tail after asymmetric adverse outcomes, deployed exits and
costs. The February-to-March deterioration is driven chiefly by lower
opportunity incidence and weaker favorable payoff; improvements in exit
mixture and adverse payoff only partly offset them. April has the best exact
net IC but still a negative selected tail, which also implicates tail
calibration/selection rather than global rank alone.

The workstream must now measure, by month and side: opportunity incidence,
conditional favorable and adverse payoff, cost and exit mixture, calibration
within the selected tail, selection stability under ties, and raw-versus-mapped
ordering. Require paired tail economics alongside IC for every challenger.
An IC improvement is not a promotion criterion unless the globally selected
tail improves after the exact policy and costs.

Implementation:

- `scripts/complete_base_ic_execution_ev_paired_diagnostic.py`;
- `tests/test_complete_base_ic_execution_ev_paired_diagnostic.py`.

The focused completion suite passes 6/6. Together with the packet and frozen
stack suites, the new work passes 15/15 focused tests. No admission model,
exit policy or portfolio constraint has been changed.

### Updated bounded next work

1. Keep the transition stack frozen and consume it only through the versioned
   registry.
2. Preserve opportunity and extreme adverse magnitude as leading shared
   regularizers. Revalidate timeout by side before reuse: current short
   timeout magnitude is inverted, so it is not an eligible shared utility
   term.
3. Isolate favorable-payoff and conversion gradients with task-specific
   adapters, detached gradients or conflict control.
4. Treat the completed q10/q25/q50/severe-loss experiment as a negative
   baseline. Before another direct-loss HPO, run a side/month/era and
   transition-state error attribution for q25, q50 and the -100/-200-bps
   severe heads. Test whether causal regime/transition context, balanced
   month/domain weights, side-specific adapters or a small reliability head
   can recover within-side rank and moderate-loss calibration on a held-out
   domain. Do not ask mapping to repair a negative raw within-side IC.
5. Train an OOF action/exit-value challenger over fixed 1/2/4/8/12h and the
   deployed exit, conditioned on opportunity/payoff scale and transition/
   health context. It remains downstream of global EV admission.
6. Continue immutable opportunity packets prospectively. Do not tune the
   state thresholds on the current packets, and do not train a hard router
   before 60--100 independent incidents within a compatible lineage.
7. Score the next frozen direct architecture on a genuinely untouched or
   source-separated period before portfolio replay.
8. Keep the rising-base-IC/falling-execution-EV paradox as an explicit active
   diagnostic. On identical candidate IDs and separately by side/month, test
   whether the failure comes from horizon mismatch, opportunity prevalence,
   conditional favorable/adverse payoff, exit conversion, costs, score
   calibration, global-tail instability, or changing asset/side/regime
   composition. Compare full-sample rank IC with top-1/5/10/20% response
   curves, tail precision/recall, CVaR and deterministic selection stability;
   then run rank-preserving month counterfactuals and fixed-composition
   reweighting. Do not accept “the base learns alpha, not execution EV” as the
   conclusion: close this item only after the IC-to-EV delta is quantitatively
   attributed and any material decision-time component is tested OOF as an
   EV-head feature or calibration input.

   This investigation is mandatory precisely because the direction is odd,
   not because the architecture permits the base to optimize a different
   target.  April improves on base-target, exact-MFE and exact-net rank IC yet
   still produces a negative selected tail.  Test whether broad IC is hiding
   a non-monotone or unstable upper-tail response, month-varying calibration,
   adverse-tail concentration, candidate-composition drift, exit-policy
   conversion loss or cost drag.  Report top-1/5/10/20 response curves,
   conditional favorable/adverse payoff, loss rate, CVaR, exit mixture,
   composition, cutoff ties and uncertainty on identical rows, then use
   rank-preserving month counterfactuals and fixed-composition reweighting to
   attribute the monthly EV delta.

   Feature/target/geometry selection must be scored after the same strictly
   causal recent-EV mapping and pooled-global top-k used by the live policy.
   Raw challenger ranking may be retained only as a diagnostic.  A March
   validation fold may use only predictions with outcomes resolved before
   that fold to fit its mapping, with unsupported warm-up rows excluded.
   The first conversion-residual draft violated this selection-policy
   alignment by choosing its March configuration on raw scores and applying
   score-specific causal mapping only in April; its data integrity and April
   mapping remain diagnostic, but its chosen configuration is not an
   authoritative mapped-policy winner.
   Execute this as a fixed waterfall rather than a narrative explanation:
   (a) base target -> exact 12h MFE/opportunity; (b) opportunity -> attainable
   gross payoff under each frozen exit; (c) gross payoff -> deployed-exit
   realized payoff; and (d) realized payoff -> exact net after spread, fees
   and other policy costs. At every step report full rank IC and global
   top-1/5/10/20% economics on the same rows, plus tail precision, loss rate,
   CVaR and support. Then run score-rank-preserving month swaps of opportunity
   incidence, favorable payoff, adverse payoff, exit mixture, costs and
   candidate composition to quantify each contribution to the monthly EV
   delta. The primary comparison is February versus March and April, where
   base-target IC rises while the base-ranked exact-execution tail remains
   negative. Repeat it for both sides and for raw base, residual and direct-EV
   scores without substituting mapped scores.

   Treat metric naming as part of the audit contract: the approximately
   -59/-91/-38-bps monthly values are base-score-ranked exact execution
   outcomes, not direct-EV-head top deciles. Materialize both series side by
   side so that a base-to-economics conversion failure cannot be mistaken for
   a trained EV-head failure. If exact-net IC rises but tail EV falls, inspect
   score compression, quantile response non-monotonicity, extreme-loss
   concentration and deterministic cutoff/tie sensitivity before attributing
   the gap to costs or exits.
9. Keep cross-side mapping as an audit layer until raw within-side ordering
   survives the latest-domain gates. When it does, compare pooled common-unit
   calibration with explicit tail extrapolation against side-shrunk mapping,
   using a historically frozen continuous plateau order. The completed q25
   audit shows that deterministic tie resolution alone does not repair payoff
   transfer and that side mapping can reverse an almost-all-short raw book
   into an all-long mapped book.

### Frozen active-transition interaction audit

The remaining narrow transition-interaction recommendation is tested at
`data_perp/artifacts/frozen_transition_opportunity_interaction_audit_20260729_v1/`
by `scripts/audit_frozen_transition_opportunity_interactions.py`.

The audit binds the March-selected
`base_transition_health + economic_multitask` model, uses only 70,176 March
rows whose 12-hour outcomes resolve before April, freezes side-local March
80th-percentile risk thresholds, and scores all 69,258 April rows once. The
fixed no-HPO ridge arms are:

1. direct-score calibration only;
2. direct score plus active probability and economic-risk main effects; and
3. the same fields plus active probability crossed with low predicted
   opportunity, adverse magnitude, exit-conversion loss, timeout, negative
   recent health, recent mapping error, recent cost, low mapping support and
   destination uncertainty.

All top-k metrics remain one pooled global April book with candidate-ID ties.
There is no timestamp/side quota, transition veto, exposure change, April
selection, portfolio replay or promotion claim.

| April score | Full IC | Global top-10 net |
|---|---:|---:|
| Frozen raw direct score | **0.0802** | **-87.62 bps** |
| March side-local direct calibration | 0.0320 | -101.42 |
| Economic-context main effects | 0.0092 | -131.34 |
| Active × economic-risk interactions | 0.0522 | -110.53 |

Explicit active interactions recover some IC versus main effects and improve
the extreme top 1% from -220.34 to -124.32 bps, but they still degrade the
unchanged top-10 book by 22.91 bps versus the raw direct score and remain
negative at every tested depth. They do not justify a trust head or control.

The frozen April 2x2 descriptive contrasts are directionally informative:

- active × high predicted exit-conversion risk: -51.14 bps interaction;
- active × high predicted adverse magnitude: -38.98 bps;
- active × high recent mapping error: -18.09 bps;
- active × negative recent health: -14.43 bps.

These are conditional diagnostics, not causal estimates or thresholds.
Timeout, low-opportunity and recent-cost contrasts do not have the expected
negative interaction sign. Low-map-support interaction is unidentified, and
destination uncertainty has only 386 active/high rows with an empty
inactive/high comparison cell, so its difference-in-differences is
unavailable.

Decision: **freeze the negative result.** Active-transition probability may
remain contextual evidence, especially alongside predicted conversion and
adverse risk, but the fixed interaction stack is not incrementally tradable.
Do not reopen a larger interaction search on April. Reassess only after a new
direct-utility geometry and independent chronological evidence exist.

Implementation:

- `scripts/audit_frozen_transition_opportunity_interactions.py`;
- `tests/test_frozen_transition_opportunity_interactions.py`.

The focused interaction suite passes 4/4.

## 2026-07-29 cross-era exact transition materialisation and grouped diagnosis

This continuation implements the requested older-data/current-data transition
research without requiring walk-forward validation.  It also corrects a
material target-lineage mismatch caught during the source audit.

### Current exact-policy causal mapping correction

The published May--July recent-EV map was calibrated on the older v7 outcome
ledger.  The corrected exact-policy ledger differs by approximately -85.79
bps on average and therefore cannot be joined to that old map as if the map
were still in common exact-policy EV units.

The immutable corrected source is:

`data_perp/artifacts/current_exact_policy_global_book_mapping_source_20260729_v1/`

implemented by:

`scripts/materialize_current_global_book_mapping_source.py`.

It keeps the frozen raw CatBoost residual score and its model provenance but
refits only the daily 21-day score-to-EV map against corrected exact-policy
outcomes whose `execution_label_end_utc < snapshot`.  It contains:

- 123,824 candidate rows;
- 121,208 causally mapped exact-policy rows;
- 2,616 explicit early-May warm-up exclusions;
- 116,712 strict outer-OOF raw-score rows, of which 114,096 have a mapped
  score;
- 7,112 frozen, non-promotable forward-OOS rows;
- decision coverage from 2026-05-07 00:00 UTC through
  2026-07-19 16:00 UTC.

The mapper uses one pooled global ranking universe after the causal common-unit
map.  It never creates a timestamp, side, asset or regime quota.  Gross, cost
and net use the corrected spread-aware exact policy; `full_sl` is explicitly
normalised to `full_stop`.

The matching before/after labels are immutable at:

`data_perp/artifacts/current_exact_policy_global_book_conversion_transition_labels_20260729_v1/`.

They contain 17,050 global-book rows and 16,810 causal mapped-EV-band rows
over exact 3h and 12h before `[s-H,s)` / after `[s,s+H)` windows.  All
33,515 eligible book phases and 18,590 eligible band phases reconcile.

### Older exact-1m transition arm

The source-separated older arm is immutable at:

`data_perp/artifacts/reconstructed_exact1m_global_book_conversion_transition_labels_20260729_v1/`.

It uses the existing strict-OOF January--April 2025 direct-EV score and causal
21-day map:

- 771,494 candidate rows;
- 764,984 mapped rows;
- 6,510 explicit mapping warm-up rows;
- 25,210 global-book transition rows;
- 24,970 mapped-EV-band rows.

This is exact one-minute path evidence but fee-only reconstructed economics.
It remains prohibited from pooled PnL comparison, sample weighting or
promotion with the spread-aware canonical/current arms.  A dtype-safe
`1e-7` accounting tolerance was added to the common label materializer because
the old ledger is float32; the observed maximum reconciliation difference is
only `5.96e-8` return (approximately 0.000596 bps).  Economically meaningful
accounting differences still fail closed.

### Final common research panel

Use only:

`data_perp/artifacts/cross_era_global_book_transition_research_panel_20260729_v3/`.

The earlier v1/v2 drafts are superseded.  The final panel has:

- 12,418 complete source-separated transition rows;
- 311 permitted decision-time features;
- 32 raw, soft, persistent, onset and mechanism targets;
- 1,302 optional fixed-UTC non-overlapping anchors;
- exact strict-OOF versus frozen-forward provenance for the current arm.

The shared feature surface contains only:

1. the 26 raw point-in-time market fields expressed as side-local
   cross-sectional medians and IQRs, their long/short gaps and exact 1h/3h/12h
   past deltas;
2. causal mapped-EV coordinate geometry at the decision plus trailing 3h/12h
   support, score distribution, causal percentile, p90 margin and EV-band
   shares.

Existing full-history KMeans `state_id`, state distance and OOD fields are
excluded.  They were outcome-free but are not a stable cross-era geometry:
64--65% of May--July observations were OOD and no historical state passed the
economic recurrence gate.  Month, year, source family, economics tier and
OOF/forward role are metadata only, never routing features.

The primary adverse state is predeclared as:

```text
after conversion residual <= -50 bps
AND delta conversion residual <= -75 bps
AND delta direct net EV <= -75 bps
```

`active_adverse` requires at least two positives across the current and next
two hourly anchors.  `adverse_onset` requires no active state during the prior
six hours.  `adverse_onset_within_3h` asks whether an onset occurs at the
current or next two anchors.  Every derived target carries the exact maximum
availability time of all raw labels on which it depends.  The 50/75/100-bps
sensitivity variants are materialised separately and are not classifier HPO.

H12 primary support is:

| Source-separated H12 arm | Rows | Raw adverse | Active adverse | Onsets | Onset-within-3h positives |
|---|---:|---:|---:|---:|---:|
| Reconstructed fee-only Jan--Apr 2025 | 2,498 | 611 | 607 | 56 | 168 |
| Canonical spread-aware Feb--Apr 2025 | 2,090 | 443 | 441 | 45 | 135 |
| Current exact spread-aware May--Jul 2026 | 1,546 | 357 | 352 | 33 | 97 |

### Non-walk-forward grouped classifier ablation

The final diagnostic is immutable at:

`data_perp/artifacts/cross_era_regime_transition_classifier_ablation_20260729_v3/`,

implemented by:

`scripts/run_cross_era_regime_transition_classifier_ablation.py`.

The design is not walk-forward:

- shuffled five-fold `StratifiedGroupKFold` over seven-day UTC groups;
- a two-sided 36h embargo removes every training anchor near a held-out
  label-dependency envelope;
- all imputation, scaling and modelling are fitted inside each training fold;
- feature arms are coordinates only, raw state only, past transitions only,
  and coordinates plus raw state;
- logistic and ExtraTrees challengers are compared with the fold prior;
- nested three-fold grouped/36h-purged predictions choose probability
  shrinkage toward the outer-training prevalence;
- fee-only, canonical spread-aware and current spread-aware results remain
  separate.  The combined spread arm also reports each source separately.

All 160 model/feature/target arms were evaluable; no arm was skipped.

#### What is learnable

Active adverse-state ranking contains real but regime-dependent signal:

| H12 active-adverse arm | Best raw ranking PR-AUC | Fold-prior PR-AUC | Raw ROC-AUC | Best calibrated Brier | Fold-prior Brier |
|---|---:|---:|---:|---:|---:|
| Reconstructed fee-only | 0.381 | 0.215 | 0.649 | 0.1840 | 0.1854 |
| Canonical spread-aware | 0.382 | 0.196 | 0.643 | 0.1665 | 0.1674 |
| Current exact spread-aware | 0.297 | 0.220 | 0.568 | 0.1773 | 0.1784 |
| Combined spread-aware | 0.318 | 0.207 | 0.597 | 0.1705 | 0.1717 |

Canonical active-state ranking is strongest with coordinates plus raw state.
The current arm ranks best with recent raw transition deltas.  Thus mapping
geometry and raw market transitions are complementary, but their incremental
importance changes by era.

Nested calibration is deliberately conservative.  Current fold-local
shrinkage weights are usually `0.0--0.1`, showing that only a small portion
of the raw probability variation survives an honest calibration objective.

On current strict mapped-OOF rows:

- active-adverse calibrated Brier improves from 0.17347 prior to 0.17229,
  with PR-AUC 0.230 versus 0.215 prior;
- onset-within-3h calibrated Brier improves from 0.06089 to 0.06068,
  with PR-AUC 0.0837 versus 0.0613 prior and ROC-AUC 0.5656;
- at the fixed 10% alert budget, onset event recall is only 21.9%, median lead
  is 2h and false alerts are approximately 66.5 per 30 days.

Only 30 complete H12 active-state rows fall in the frozen forward-OOS segment,
and the onset evaluation has no useful forward event support.  It cannot
validate July transfer.

#### Decision

The workstream has materialised the correct data and demonstrated that broad
active-regime state is partly learnable.  It has **not** produced a reliable
actionable transition classifier.  The onset head's calibration gain is tiny,
its alert precision is inadequate and its false-alert burden is too high.
No admission veto, trust router, portfolio constraint, exit policy, timing
action or production score changes.

### Bounded next work

1. Run seven-day block bootstrap intervals on the predeclared calibrated
   active/onset challengers; do not select a winner from point estimates.
2. Run the source-to-source transfer matrix for raw state, coordinates and
   combined features.  This tests whether the May--July transition surface
   is genuinely distinct rather than merely lower support.
3. Ablate the predeclared 50/75/100-bps adverse-label sensitivity grid and
   separate upside-collapse from loss-expansion mechanisms.
4. Add fold-local feature selection and HPO only after the transfer result,
   including raw feature composites for volatility/range expansion,
   liquidity/spread impairment, directional crowding, OI/flow stress and
   mapped-score concentration changes.
5. Materialise new candidate, exact-policy label, causal-map and raw-context
   rows for 2026-07-20 through 2026-07-23.  No existing candidate-level
   artifact covers those dates; names containing `july20` are run names and
   stop at July 19.
6. Keep timing, MAE, target-price and wait actions in the downstream action
   layer.  A transition probability, if eventually admitted, remains context
   for the direct execution-EV/action architecture rather than a standalone
   trade side or hard gate.

Implementation and verification:

- `scripts/materialize_current_global_book_mapping_source.py`;
- `scripts/materialize_canonical_global_book_conversion_transition_labels.py`;
- `scripts/materialize_cross_era_global_book_transition_research_panel.py`;
- `scripts/run_cross_era_regime_transition_classifier_ablation.py`;
- `scripts/bootstrap_cross_era_transition_classifier.py`;
- five focused new/extended test modules.

The integrated focused suite passes 14/14.  All six new immutable artifact
manifests and every recorded output checksum have been independently
reverified.

#### Paired seven-day block-bootstrap gate

The requested uncertainty gate is complete at:

`data_perp/artifacts/cross_era_transition_classifier_bootstrap_20260729_v1/`.

It uses 2,000 paired resamples of the same UTC seven-day groups and compares
each frozen calibrated challenger with its exact fold-prior predictions.
Intervals are exploratory because the arms were frozen after reviewing the
ablation; they do not correct for that prior selection.

| Frozen calibrated arm | Delta Brier, 95% CI | Delta PR-AUC, 95% CI |
|---|---:|---:|
| Canonical active | -0.00088 [-0.00392, +0.00216] | +0.0879 [+0.0319, +0.1737] |
| Canonical onset | -0.00038 [-0.00118, +0.00010] | +0.0307 [-0.0001, +0.0882] |
| Current strict-OOF active | -0.00117 [-0.00459, +0.00158] | +0.0155 [-0.0282, +0.0704] |
| Current strict-OOF onset | -0.00021 [-0.00050, +0.00000] | +0.0224 [+0.0097, +0.0311] |
| Reconstructed active | -0.00141 [-0.00274, +0.00009] | +0.0453 [+0.0172, +0.0853] |
| Reconstructed onset | 0.00000 [0.00000, 0.00000] | 0.0000 [0.0000, 0.0000] |

The result reinforces the promotion rejection:

- active-state ranking is real in the canonical and reconstructed arms, but
  calibrated Brier improvement still crosses zero;
- current strict-OOF active ranking and calibration are not established;
- current onset ordering has a positive exploratory PR-AUC interval and a
  positive ROC-AUC interval, but its Brier interval touches zero and the
  operational alert burden remains unacceptable;
- the reconstructed onset calibrator collapses fully to the prior.

Therefore no transition head is admitted.  The source-transfer matrix and
label-sensitivity/mechanism ablations remain the next model diagnostics.

#### Source-to-source transition transfer matrix

The predeclared transfer diagnostic is complete at:

`data_perp/artifacts/cross_era_transition_source_transfer_20260729_v1/`.

It trains on exactly one source and scores a different source once.  Current
training and evaluation use only strict mapped-OOF rows.  Shrinkage is chosen
from nested grouped, 36-hour-purged OOF predictions inside the training source;
the destination source never participates in model or calibration selection.
All six ordered source pairs, two targets, four fixed feature families and two
model families ran without skipped arms.  This produced 192 metric rows and
194,368 model-prediction rows.  Fee-only and spread-aware outcomes remain
separate; reverse-time arms are symmetry diagnostics and never promotion
evidence.

Best Brier-selected active-state result for each ordered pair:

| Train -> evaluation | Feature/model | Shrink | Delta Brier | Delta PR-AUC | Delta ROC-AUC |
|---|---|---:|---:|---:|---:|
| Reconstructed fee-only -> canonical spread | past transitions / logistic | 0.20 | -0.00752 | +0.1864 | +0.1649 |
| Canonical spread -> reconstructed fee-only | raw state / logistic | 0.10 | -0.00470 | +0.1640 | +0.1667 |
| Canonical spread -> current strict OOF | coordinates / ExtraTrees | 0.05 | -0.00039 | +0.0824 | +0.1096 |
| Current strict OOF -> canonical spread | coordinates / logistic | 0.05 | -0.00135 | +0.0742 | +0.0755 |
| Reconstructed fee-only -> current strict OOF | prior fallback | 0.00 | 0.00000 | 0.0000 | 0.0000 |
| Current strict OOF -> reconstructed fee-only | coordinates / ExtraTrees | 0.10 | -0.00068 | +0.0199 | +0.0255 |

The broad adverse-state geometry therefore transfers strongly between the two
older 2025 sources and weakly but bidirectionally between the canonical
spread-aware and current spread-aware sources.  The fee-only reconstruction
does not transfer forward into the current era after honest training-source
calibration.  This is evidence of a changed economic/label surface, not proof
that the raw market geometry itself is unique to July.

Onset-within-three-hours does not transfer reliably.  Reconstructed-to-
canonical, canonical-to-reconstructed, reconstructed-to-current and
canonical-to-current all shrink completely to the prior.  Current-to-
canonical has only -0.000016 Brier, +0.0106 PR-AUC and +0.0275 ROC-AUC;
current-to-reconstructed is weaker still.  These are descriptive point
estimates without a frozen paired transfer bootstrap.

Decision:

- retain coordinate geometry as the leading cross-era active-risk feature
  family, with raw state/past transitions as source-dependent additions;
- do not promote an onset head, transition veto, trust router or portfolio
  control;
- do not pool fee-only and spread-aware economics to enlarge support;
- run the 50/75/100-bps adverse-label and upside-collapse/loss-expansion
  mechanism ablations next; and
- require paired group-bootstrap uncertainty on any transfer arm selected for
  further use.

The runner and its deterministic, model-independent constant-prior top-10 tie
break are covered by
`tests/test_run_cross_era_transition_source_transfer.py` (2/2 passing).  The
artifact manifest sidecar and all three recorded output hashes have been
reverified.

#### July 20--23 raw-input repair progress

The first raw-data blocker identified by the July extension audit has been
partially repaired through the required label-resolution cutoff
`2026-07-24 12:00 UTC`.

The supported append-only hourly OHLCV repair inspected the trailing 14 days
for all 237 verified Kraken perpetual instruments.  It appended 15,382
previously missing rows across 227 instruments with zero fetch failures.  The
supported open-interest repair was then run in four disjoint symbol
partitions; in total it fetched 23,207 missing quote-notional OI rows, updated
223 instruments and reported zero failures.  Existing partitions were
preserved and only missing timestamps were appended.

Historical funding was subsequently repaired for all 237 verified
instruments using the exchange API. The merge added 115,564 non-null funding
observations with 237 successful instruments, zero empty responses and zero
failures. Over the July 20--23 decision interval, 209 instruments now have all
96 hourly funding observations; seven have none and must remain explicit
coverage exclusions rather than being silently forward-filled.

The derived hourly order-book proxies were then rebuilt from the repaired
native OHLCV history for all 237 verified instruments, with 237 successful
rebuilds and zero skips. Over July 20--23, 77 instruments have all 96 proxy
hours, 87 have at least 95%, 104 have at least 80%, and seven have none. The
latest proxy timestamp is `2026-07-24 12:00 UTC`. This proxy coverage follows
the underlying OHLCV availability and defines another explicit admissibility
filter for the July point-in-time surface.

A direct post-repair audit over `[2026-07-20 00:00, 2026-07-24 12:00]`
finds the same effective timestamp coverage for OHLCV and usable
price-converted OI:

| Raw-input coverage | Instruments |
|---|---:|
| 100% of 109 required hours | 76 |
| At least 95% | 87 |
| At least 80% | 101 |
| No OHLCV rows | 7 |
| No usable OI rows | 8 |

The common latest timestamp is now `2026-07-24 12:00 UTC`.  The incomplete
tail is concentrated in thin, inactive or unsupported contracts; it must
remain an explicit raw-coverage exclusion.  No forward feature builder may
neutral-fill those gaps or claim a 237-asset complete universe.

This does **not** yet make the July extension runnable.  The remaining input
blockers are:

1. materialize a point-in-time candidate ledger and the exact frozen feature
   surface for July 20--23; the persisted July-20 snapshots still fail the
   31/8 base, residual, auxiliary and CatBoost feature contracts;
2. use that candidate ledger to download exact one-minute execution windows
   through decision plus 12 hours;
3. verify funding, spread/order-book and every decision-time feature
   availability field; and
4. create a July-specific retrospective/non-promotable source lock unless an
   earlier genuinely frozen model bundle can be recovered.

The later July-27 frozen-forward lock remains prohibited for July 20--23
because several of its models use labels resolved inside that interval.

The dedicated fail-closed adapter is now implemented at
`scripts/materialize_execution_ev_july_retrospective_candidates.py`, with
four focused tests passing.  It derives and hash-binds the frozen per-side
base, residual, support-head and CatBoost contracts; requires deterministic
candidate identity, UTC decision/availability timestamps and finite raw
features; emits hourly/daily coverage; and writes no candidates unless the
entire requested surface passes.  It never reads outcomes or the later
July-27 confirmation lock.

The authoritative full-source preflight is
`data_perp/artifacts/execution_ev_july_retrospective_candidates_preflight_20260730_v4/`.
It supersedes the one-symbol v3 preflight.  V4 scans the latest full
249-symbol canonical static surface and reports:

- 96 requested signal hours from July 20 00:00 through July 23 23:00;
- 23,904 expected timestamp x symbol source rows;
- zero source rows and zero complete contract hours in that interval; and
- `candidates_written=false`,
  `status=blocked_incomplete_point_in_time_static_surface`.

All three coverage payload hashes recorded by the v4 source manifest have
been reverified.  This is now a materialized, executable blocker: rebuilding
the canonical PIT static surface is the next required step.  Raw OHLCV/OI
repair alone cannot substitute for feature generation.

#### Adverse-label sensitivity and conditional-mechanism ablation

The v4 cross-era panel and bounded classifier shards are materialized at:

- `data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4/`;
- `data_perp/artifacts/cross_era_regime_transition_classifier_ablation_20260730_v4_active{50,75,100}/`;
- `data_perp/artifacts/cross_era_regime_transition_classifier_ablation_20260730_v4_current_onset{50,75,100}/`;
- `data_perp/artifacts/cross_era_regime_transition_classifier_ablation_20260730_v4_current_{upside,loss}/`; and
- the four current-primary-75 feature shards ending in
  `_coordinates`, `_state`, `_past` and `_current_active75`.

The panel now materializes independent 50/75/100-bps raw, persistent-active,
onset and onset-within-three-hour labels, each with its exact derived
availability timestamp.  The 75-bps target remains the legacy-compatible
primary alias.  Upside-collapse and loss-expansion are truly conditional:
inactive rows are null rather than false negatives, and mechanism availability
is inherited from the fully resolved primary active-state label.  The
classifier refuses any derived target without its matching availability field.
CV remains source-separated grouped seven-day CV with a two-sided 36-hour
purge.

Combined-context ExtraTrees active-state results:

| Source | 50 bps AP / top-decile lift | 75 bps AP / lift | 100 bps AP / lift |
|---|---:|---:|---:|
| Reconstructed fee-only | 0.347 / 1.42x | 0.357 / 1.95x | 0.302 / 1.93x |
| Canonical spread | 0.375 / 2.06x | 0.382 / 2.33x | 0.259 / 2.07x |
| Current exact spread | 0.305 / 1.46x | 0.285 / 1.73x | 0.242 / 1.60x |

On the 1,506 current strict mapped-OOF rows, 75 bps has AP 0.284, ROC-AUC
0.572 and top-decile lift 1.83x.  The 50-bps target has higher AP because it is
less rare, but 75 bps gives the strongest high-score concentration and remains
the preferred research active-state definition.

The current 75-bps feature ablation is incremental:

| Feature surface | AP | Top-decile lift |
|---|---:|---:|
| Causal coordinates only | 0.273 | 1.12x |
| Raw state only | 0.265 | 1.51x |
| Past transitions only | 0.257 | 1.51x |
| Coordinates + raw state/transitions | **0.285** | **1.73x** |

The conditional mechanism result is more useful than the onset result:

- **Upside collapse** is learnable on current strict OOF: 331 active rows,
  37.2% prevalence, AP 0.599, ROC-AUC 0.757 and 1.84x top-decile lift.  Retain
  it as a conditional supporting-head candidate.
- **Loss expansion** is not a useful classifier under the current +50-bps
  definition: prevalence is 94.6%, AP 0.936 only because the event is nearly
  universal, ROC-AUC is 0.454 and lift is 0.97x.  Redefine it using severity,
  a higher threshold, a source-relative quantile or a continuous/ordinal
  conditional loss target before another model run.
- **Onset within three hours** remains unusable at every threshold.  Strict-OOF
  AP is 0.062/0.072/0.080, ROC-AUC 0.446/0.478/0.498, event recall
  17.6%/21.9%/22.6%, and false alerts remain approximately 63--66 per 30 days.

Decision:

- retain combined coordinate + state/transition context and the 75-bps active
  definition for research;
- advance conditional upside-collapse only as a supporting feature/head;
- redesign loss expansion and the onset target before spending HPO budget;
- keep all transition outputs out of timing, wait, admission, portfolio and
  production policy; and
- require later untouched or chronological evidence plus operational alert
  precision before any promotion.

The integrated panel/classifier/source-transfer/bootstrap focused suite passes
9/9.  The v4 panel and all twelve bounded shard manifests, sidecars and
recorded outputs have been independently hash-verified.

### Separate Common-30 historical opportunity-support extension

The older March--July 2025 Common-30 source has now been materialized as a
strictly separate lineage at:

`data_perp/artifacts/common30_opportunity_support_extension_20260729_v1/`

This corrects an important lineage ambiguity. The source provides 200,160
strict expanding-OOF two-layer direct-EV scores and exact one-minute 12-hour
outcomes, but it is **not** an incumbent/current-policy replay. It uses:

- a frozen 30-symbol universe;
- a distinct raw-PIT/two-layer model contract;
- exact 12-hour labels;
- a 100-bps counterfactual cost;
- no incumbent admission calibrator;
- no concurrency, exposure, asset-limit or portfolio replay.

It is therefore named
`historical_2025_common30_12h_cost100bps_direct_ev_oof`, is permanently
`promotion_eligible=false`, and must never be pooled with either the
full-universe historical raw-alpha lineage or the current 2026 execution-EV
lineage. Its permitted use is within-lineage retrospective descriptive
recurrence and failure-packet research.

The materializer replaces any convenience outcome copies in the OOF score
table from the frozen exact-label ledger, verifies the signal+1h decision and
decision+12h resolution contract, verifies gross minus cost equals net, and
requires every direct-model training cutoff to be no later than signal time
and strictly earlier than execution decision time. The source summary,
coverage preflight and both base/direct fold audits are hash-bound in the
manifest.

Selection is recomputed rather than inherited from the archived retrospective
raw-score report:

1. fit a 21-day side-shrunk isotonic score-to-net-EV map at each UTC-day
   snapshot;
2. admit only references whose exact 12-hour outcomes resolved strictly before
   the snapshot;
3. select one pooled global top 10% across the entire eligible lineage by
   causal mapped EV, with candidate-ID tie breaking;
4. never re-rank by timestamp, month or side.

This is a causal mapped-score diagnostic selection, not the incumbent
21-day admission-plus-portfolio policy. That limitation is explicit in the
manifest.

Materialized support:

| Common-30 support field | Result |
|---|---:|
| Strict OOF candidates | 200,160 |
| Causal-map eligible | 198,780 |
| Frozen mapped global top-10 candidates | 19,878 |
| Broad economic-failure events | 35 |
| Strict economic-failure events | 24 |
| Frozen strict-containing packets | 23 |
| Packets with adequate causal reference | 18 |
| Recovered within 72h | 18 |
| Median recovery time | 13.5h |

The 23 packets occur in March/April/May/June/July as 3/4/5/1/10. All are
resolution-frozen. Five packets lack the required causal state-reference
history and remain unclassified for relative-state purposes.

Within-lineage taxonomy:

| State | Packets |
|---|---:|
| Normal opportunity | 12 |
| Unclassified | 9 |
| Adverse-payoff expansion | 2 |
| High opportunity / poor conversion | 1 |
| Exit-conversion failure | 1 |
| Mixed | 1 |
| Sparse opportunity | 0 |
| Favorable-payoff compression | 0 |
| Timeout degradation | 0 |
| Execution/liquidity impairment | 0 |

The mixed March packet is simultaneously high-opportunity/poor-conversion,
adverse-expansion and exit-conversion failure. Adverse expansion recurs once
in March and once in late July, but all other non-normal labels have only a
single occurrence. Most strict mapping-residual failures are economically
normal or below the taxonomy's pre-frozen state thresholds. This is useful
evidence that a generic strict failure label is not equivalent to a stable,
actionable opportunity state.

The recomputed causal map does not rescue this lineage:

| Month | Selected rows | Net EV | Gross EV | Long share |
|---|---:|---:|---:|---:|
| March | 3,558 | -133.82 bps | -33.99 bps | 79.5% |
| April | 4,992 | -155.08 | -55.36 | 65.5% |
| May | 3,300 | -166.03 | -66.36 | 84.2% |
| June | 373 | -136.77 | -36.96 | 59.5% |
| July | 7,655 | -141.80 | -42.01 | 90.3% |
| Pooled | 19,878 | **-147.63** | approximately -47.8 | approximately 82% |

The source's archived future-informed raw global top-10 diagnostic was
-109.39 bps. The causal mapped selection is therefore 38.24 bps worse, while
remaining negative in every month. This adds another independent example in
which an unconstrained recent isotonic EV map damages the economically
relevant tail.

Active-transition probability is close to zero for most strict packets. One
April packet has active probability 0.567 yet is classified as a normal
opportunity, and the March mixed conversion/adverse packet has active
probability below 0.001. This independently reinforces the frozen conclusion:
active transition is neither necessary nor sufficient for economic failure.

Decision:

- retain the 23 packets as a separate descriptive historical lineage;
- do not add them to the current/prospective 60--100 incident promotion gate;
- do not fit a router from them;
- use the repeated March/July adverse-expansion examples only as hypotheses
  for future packet matching;
- require future/current packets under one frozen incumbent selection and
  portfolio contract before promotion-grade classification.

Implementation:

- `scripts/materialize_common30_opportunity_support_extension.py`;
- `tests/test_common30_opportunity_support_extension.py`;
- backward-compatible mapped-selection support in
  `scripts/materialize_historical_exact_model_health.py`.

The integrated execution-utility, IC/EV, frozen-transition, packet and
Common-30 suite passes 54/54 tests.

### Fail-closed prospective incident registry and corrected support count

The packet snapshot was subsequently audited for true prospective,
append-only behavior. The earlier phrase "10 current incidents" needs a
precise qualification:

- all ten packets are strict model-OOS research-book incidents;
- all ten anchors occur between 8 May and 29 June 2026;
- their selected event-window candidates are outer OOF;
- seven have adequate causal state-reference history;
- none comes from the resolved-forward July segment;
- none is bound to an immutable incumbent admission-plus-portfolio decision
  ledger.

The correct status is therefore:

| Support scope | Count |
|---|---:|
| Current-model strict research incidents | 10 |
| Current-model taxonomy-usable incidents | 7 |
| Resolved-forward prospective research incidents | **0** |
| Incumbent-portfolio-parity prospective incidents | **0** |
| Minimum required | 60 |
| Target support | 100 |

The previous snapshot materializers are immutable outputs but are not
append-only ledgers. Extending their input can change the all-period global
top-10 cutoff, robust failure-label normalization, prior event anchors and
ordinal incident IDs. In the observed July extension, an earlier broad anchor
moved and a new June incident appeared. Refusing to overwrite an artifact
directory is therefore not sufficient prospective protection.

A dedicated fail-closed gate is now implemented:

- `configs/prospective_opportunity_incident_gate_20260729_v1.json`;
- `scripts/enforce_prospective_opportunity_incident_gate.py`;
- `tests/test_prospective_opportunity_incident_gate.py`;
- `data_perp/artifacts/prospective_opportunity_incident_gate_20260729_v1/`;
- chained verification snapshot
  `data_perp/artifacts/prospective_opportunity_incident_gate_20260729_v2/`.

The gate:

1. hash-verifies the packet snapshots, Common-30 quarantine, current selected
   candidates, portfolio replay and the complete frozen-transition registry;
2. keeps historical raw-alpha, Common-30, current research and future
   incumbent-policy contracts separate;
3. classifies current packets as retrospective outer OOF, resolved-forward
   research, or mixed/unknown provenance;
4. requires packet availability by the declared as-of time and a frozen
   resolution state;
5. rejects unconsolidated overlapping incidents and source events appearing
   in more than one packet;
6. persists a content hash for every packet;
7. rejects removal or mutation of a prior frozen packet;
8. rejects a new packet inserted behind a lineage's frozen anchor watermark;
9. chains every new registry to the prior registry hash;
10. exposes an enforcement mode that refuses authorization below the numerical
    gate.

The first registry contains 74 quarantined packets:

- 41 historical raw-alpha;
- 10 current strict-model-OOS research;
- 23 Common-30 counterfactual.

These are not summed for any promotion decision. The enforced current-model
gap is 50 incidents to the minimum, while the incumbent-portfolio gap is the
full 60 because no exact prospective portfolio lineage is yet bound.

The unchanged second snapshot successfully validates against the first
registry and records its parent hash. Unit tests additionally prove that a
mixed OOF/forward packet is ineligible, a frozen packet rewrite fails, a
retroactive insert fails, and ten incidents cannot open the 60-event gate.

Current enforced decisions:

- supervised failure-detector training: **not authorized**;
- opportunity-state router: **not authorized**;
- incumbent-portfolio promotion: **not authorized**;
- transition/opportunity admission or exposure controls: **not authorized**;
- cross-lineage pooling: **forbidden**.

The next prospective collector must first bind a content-addressed incumbent
policy descriptor containing model/feature hashes, causal mapping and
admission contract, universe and tie rule, portfolio/concurrency/exposure/
asset limits, exact exit/cost/label geometry, and event construction rules.
It must then store immutable decision-time candidate and portfolio decisions
before outcomes resolve. Future incidents can be appended only after the
12-hour post window, six-hour consolidation uncertainty and 72-hour recovery
or censor horizon are closed.

### Content-addressed policy and immutable decision-ledger contract

The policy-binding format required by the prospective collector is now
implemented and exercised on the available global-top-10 portfolio replay:

- `configs/research_execution_policy_decision_contract_20260729_v1.json`;
- `scripts/freeze_opportunity_policy_decision_ledger.py`;
- `tests/test_freeze_opportunity_policy_decision_ledger.py`;
- `data_perp/artifacts/research_execution_policy_decision_ledger_20260729_v1/`.

The frozen descriptor binds:

- score family and side-local model contract;
- 21-day causal side-shrunk isotonic mapping geometry;
- one pooled global top-10 selection and candidate-ID tie rule;
- no timestamp or side quotas;
- baseline global-auction concurrency, per-symbol, entry-rate and wallet caps;
- signal+1h decision, 12-hour outcome, cost and exit contracts;
- before/after failure windows, persistence, consolidation and recovery rules;
- source data, portfolio configuration and mapping/portfolio/label/packet
  runner hashes;
- the exact materialized evaluation universe.

Its canonical policy ID is:

`737819e4fd4a17a5b7be94494dd2b5756668ea5f79cf984608570223b9326048`

The immutable ledger contains:

| Decision-ledger field | Result |
|---|---:|
| Global-top-10 candidate decisions | 12,383 |
| Portfolio accepted | 1,249 |
| First decision | 5 May 2026 20:00 UTC |
| Last decision | 10 July 2026 12:00 UTC |
| Prospective forward decisions | **0** |

Each decision receives a deterministic ID from policy ID, candidate ID and
decision timestamp. Candidate identity, signal/decision time, mapped rank,
evaluation fold, exact candidate outcome, portfolio acceptance/rejection,
position size, concurrency state, wallet state and portfolio outcome are
persisted together. The builder requires exactly one portfolio decision for
every selected candidate and rejects duplicate candidates or a signal/decision
timing mismatch.

This artifact is deliberately
`RESEARCH_SNAPSHOT_NOT_INCUMBENT` and `promotion_eligible=false`. It proves
that the required content-addressed policy and decision-ledger machinery
works, but it cannot seed the prospective gate because its single global
evaluation book ends before the forward segment and contains zero prospective
decisions. A future incumbent contract must use the same schema with
`FROZEN_PROSPECTIVE_INCUMBENT`, record decisions before outcomes resolve, and
then remain unchanged while packets accumulate.

The live `inference_trades.csv` ledger was also checked as a possible
prospective source. It ends on 19 July 2026 and records the separate
`s52_meta_threshold_handoff` live policy family, not the frozen execution-EV
research policy above. It therefore cannot be relabeled or imported into this
policy lineage. No post-19-July same-policy decision source is currently
materialized in Ares.

### Formal workstream completion audit

The controlling brief is now audited requirement by requirement at:

`data_perp/artifacts/execution_utility_workstream_completion_audit_20260729_v1/`

The hash-bound matrix reports:

| Audit classification | Requirements |
|---|---:|
| Proved complete | 6 |
| Completed negative result | 2 |
| Correctly not authorized | 1 |
| Partial, research-only | 1 |
| Incomplete external support | 1 |

The two completed negative results are:

- the direct multi-task model does not beat the frozen residual control on
  April: causal-mapped -88.97 versus residual -24.32 bps;
- frozen transition/health interactions are not incrementally tradable and
  authorize no control.

The partial item is the policy ledger: its schema and content addressing are
complete, but the only materialized instance is deliberately research-only
with zero prospective decisions.

The single unmet end state is now explicit and narrow: an unchanged
prospective incumbent execution-EV policy feed plus 60--100 fully resolved,
compatible incidents. Current evidence is 10 current-model OOF incidents,
seven taxonomy-usable, zero prospective-forward and zero incumbent-portfolio
incidents.

Audit status:

`IMPLEMENTATION_COMPLETE_PROSPECTIVE_ACCUMULATION_OPEN`

The audit states `objective_complete=false` and confirms that no further model
or HPO work is authorized on the unchanged evidence. This prevents the
implementation work from being mistaken for completion while also preventing
another search from manufacturing apparent progress from the same incidents.

Implementation:

- `scripts/audit_execution_utility_workstream_completion.py`;
- `tests/test_audit_execution_utility_workstream_completion.py`.

### 30 July cross-era transition attribution and raw transfer ablation

This section supersedes the older statement above that no additional bounded
model work was authorized. The later roadmap explicitly authorized causal
domain-balancing, side-local adapter and reliability ablations using the
newly materialized cross-era exact-12-hour dataset, followed by a frozen
July 20--23 evaluation.

#### Transition lineage and error attribution

The authoritative cross-era input remains:

`data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3/`

It contains 340,083 unique candidate identities and 256 frozen pre-entry
features. The July 20--23 pack contains 5,760 identities with one-to-one
exact labels and `feature_available_at <= execution_decision_utc` throughout.

The frozen active-transition/BOCPD registry is **not** usable on that current
window: its dense sources end on 12 July at 19:00 UTC and have 0% July 20--23
coverage. Destination probabilities remain excluded because their declared
availability field is absent and their support is sparse. The eight static
regime fields in the cross-era feature contract are complete across both eras
and current, so attribution uses them while representing missing active
transition probability explicitly.

The cross-era entropy, stability and volatility-of-volatility columns are
already transformed/winsorized coordinates, not physical probabilities.
Therefore the legacy expression
`entropy48 * (1 - clip(stability24, 0, 1))` is retained only for lineage
comparison. New correction layers use:

- `transition_pressure_z = entropy48 - stability24`;
- `entropy_acceleration_z = entropy12 - entropy48`;
- `entropy_vov_interaction_z = entropy48 * volatility_of_volatility_48`.

The completed attribution artifact is:

`data_perp/artifacts/cross_era_direct_head_transition_attribution_audit_20260730_v2/`

It establishes:

- July 20--23 q25 raw IC is -0.101 long and -0.054 short; q50 retains only
  +0.013 long and +0.021 short;
- raw q25 global top-10 is -136.83 bps and the frozen mapped q25 result is
  -148.43 bps;
- state-local current tails are negative across all sufficiently supported
  transformed transition, entropy, stability and volatility cells;
- the failure is therefore broad rather than an isolated regime that can be
  repaired by a hard router;
- severe-loss calibration shifts in opposite directions by side: long
  moderate-loss risk is systematically underpredicted while short risk is
  generally overpredicted.

Decision: do not train a state veto or reopen mapping. Test side-specific
adapters and reliability only as raw-score challengers.

#### Frozen transfer-adapter experiment

Implementation:

- `scripts/run_cross_era_direct_net_transfer_adapter_ablation.py`;
- `scripts/score_cross_era_direct_net_transfer_adapter_ablation.py`;
- `scripts/audit_cross_era_direct_net_transfer_adapter_ablation.py`;
- the three matching focused test modules.

The final usable freeze is:

`data_perp/artifacts/cross_era_direct_net_transfer_adapter_ablation_20260730_v2/`

The earlier `..._v1/` research tables reproduce the same metrics, but its
model bundle was serialized with a script-local class and cannot be loaded by
the separate label-free scorer. It is invalid for deployment/scoring evidence
and must not be cited as the frozen model. V2 uses a cross-process-loadable
plain bundle; its label-free score artifact is:

`data_perp/artifacts/cross_era_direct_net_transfer_adapter_current_score_20260730_v2/`

and its exact post-label audit is:

`data_perp/artifacts/cross_era_direct_net_transfer_adapter_current_audit_20260730_v2/`

The bounded search tests one fixed `raw_context_shallow_24` parent, three
train-only weight profiles and four raw-score arms:

- parent q25;
- side-local clipped CatBoost q25 residual adapter;
- one-way q25-coverage reliability penalty;
- adapter plus reliability.

All parent heads, adapters and reliability models are side-local. Correction
models see only cross-fitted parent predictions. Every outer and inner fit
requires `label_resolution_utc < validation_start`. Reliability features are
standardized and the fit fails closed on non-convergence; the final long and
short fits converged in 30 and 27 iterations. No score mapping, GMM/DAE
posterior, compact risk summary, candidate-group geometry or current outcome
enters selection. Current scoring occurs in a separate process with no label
argument.

Weight results:

| Weight profile | Aggregate raw top-10 | Latest July 1--19 | Severe calibration versus uniform | Eligible |
|---|---:|---:|---|---|
| Uniform | -76.91 bps | -117.06 bps | reference | yes |
| Era-balanced | -79.03 bps | -120.60 bps | no worse | yes |
| Era-month-balanced | -75.57 bps | -118.89 bps | worse | no |

Uniform is retained because selection ranks exact economics only among
month-covered, nonnegative aggregate/latest side-IC and non-degraded severe
calibration profiles.

Architecture results under uniform weighting:

| Raw arm | Aggregate top-10 | Latest July 1--19 | Worst month | Minimum aggregate side IC |
|---|---:|---:|---:|---:|
| Parent | -73.31 bps | -116.98 bps | -116.98 bps | +0.044 |
| Adapter | **-61.98 bps** | -119.82 bps | -119.82 bps | +0.028 |
| Reliability | -74.04 bps | -121.44 bps | -121.44 bps | +0.043 |
| Adapter + reliability | -63.72 bps | -124.61 bps | -124.61 bps | +0.027 |

The adapter is only a **research-only relative winner**. It improves aggregate
OOF economics by 11.33 bps versus the parent but degrades the latest month,
remains negative overall, and has negative raw IC in long May and short March.
Reliability is not incremental.

The explicit transfer cards reject domain generalization in both directions:

| Training -> evaluation | Parent top-10 | Adapter top-10 | Adapter + reliability |
|---|---:|---:|---:|
| February--April 2025 -> May--July 2026 | -90.70 bps | -83.83 bps | -82.62 bps |
| May--July 2026 -> February--April 2025, diagnostic only | -71.80 bps | -46.56 bps | -43.45 bps |

Every transfer arm also fails at least one raw side/month IC gate. The reverse
row is deliberately diagnostic and not causal policy evidence.

Frozen July 20--23 results:

| Raw arm | Global top-10 exact net | Allocation |
|---|---:|---:|
| Parent | -155.29 bps | 1 long / 575 short |
| Adapter | **-161.09 bps** | 0 long / 576 short |
| Reliability | -155.85 bps | 1 long / 575 short |
| Adapter + reliability | -160.65 bps | 0 long / 576 short |

For the selected adapter, raw current IC is -0.119 long and -0.140 short.
Positive precision is 7.29% and CVaR5 is -558.93 bps. Short severe-loss
discrimination also fails (`p100` ROC-AUC 0.419 and `p200` 0.471); long is
only near random at 0.518/0.521. Identity coverage is exactly 5,760/5,760.

**Decision: reject and do not replay portfolio constraints.** Historical
aggregate, latest month, forward transfer, current global, current side-local
and current raw-IC gates all fail. Mapping remains unauthorized. The result
also rejects a simple transition-context residual correction as the solution
to July transfer.

#### Required next diagnosis

1. Complete the fixed IC-to-EV waterfall already specified in updated bounded
   work item 8. Join immutable base, residual and direct-EV predictions to
   exact opportunity/MFE, attainable gross payoff, deployed-exit payoff,
   explicit costs and exact net on identical identities. Do not expand the
   transfer runner's frozen feature contract to do this.
2. Quantify February -> March -> April and May -> June -> July changes at
   top-1/5/10/20%, by side, including opportunity prevalence, favorable and
   adverse conditional payoff, exit mixture, cost drag, score compression,
   non-monotonic response and cutoff/tie sensitivity.
3. Run rank-preserving month-component swaps and fixed-composition
   reweighting. The goal is to attribute the EV delta, not merely restate that
   the base target differs from exact execution utility.
4. Treat the negative bidirectional transfer as evidence for a domain
   representation problem. Before another model search, materialize causal
   transition features through the full July evaluation window from the same
   definition in both eras. Do not impute the absent active transition
   probability or destination state.
5. If the waterfall identifies a decision-time component, test it OOF as an
   EV-head feature, side adapter input or reliability input. If it identifies
   irreconcilable domains, predeclare soft regime/domain experts and a causal
   gating diagnostic; do not train a hard router from the current negative
   state cells.

#### Source-separated IC-to-EV waterfall completed for existing ledgers

The first three diagnosis items above are now materialized for every existing
standardized historical score/economics ledger at:

`data_perp/artifacts/source_separated_ic_ev_waterfall_20260730_v2/`

V2 is canonical. The numerically identical v1 tables predate the explicit
score-role correction and must not be used as the provenance manifest.

Implementation:

- `scripts/materialize_source_separated_ic_ev_waterfall.py`;
- `tests/test_materialize_source_separated_ic_ev_waterfall.py`.

The audit hash-verifies and evaluates five sources independently:

- canonical exact-1m February--April base;
- canonical exact-1m March--April strict residual subset;
- reconstructed exact-1m January--April base/direct, diagnostic fee-only;
- reconstructed hourly October--December 2024 base/direct, diagnostic;
- old-55 hourly May 2025--April 2026 base/direct recurrence, diagnostic.

It never pools evidence tiers or performs a new mapping. Score roles are
explicit: raw base alpha, raw/OOF direct execution EV, residual delta
component, or upstream expected-EV stream. Every output uses the full
four-field identity, deterministic candidate-ID cutoff ordering, month and
pooled-global/side scopes, and global or side-local top 1/5/10/20% books.

Outputs include:

- full-sample target/MFE/gross/cost/net rank IC;
- tail support, precision, loss rate, CVaR5 and within-tail IC;
- MFE-ceiling -> deployed-gross -> explicit-cost -> exact-net levels;
- score compression and 20-bin response monotonicity;
- deterministic cutoff-tie support plus explicitly outcome-aware best/worst
  diagnostic bounds;
- adjacent-month fixed-composition decomposition for net-positive rate, MFE
  ceiling, deployed gross, explicit cost and exact net using side ×
  deterministic side-local rank decile × fixed asset cells.

Important semantic limits are enforced rather than filled:

- the base target is legacy native-24-hour alpha while exact economics use a
  12-hour policy replay;
- MFE is an upper-bound ceiling, not a canonical attainable-gross label;
- `opportunity_gross_above_cost_0bps` is exactly the `net > 0` alias in these
  ledgers and is not an independent opportunity event;
- canonical spread drag is embedded in gross and the explicit gross-to-net
  gap is fee cost; fee-only/old-hourly sources are not exact-policy parity;
- gross - cost = net is checked with a 1e-7 return tolerance solely to cover
  observed float32 rounding no larger than 5.9604645e-8.

The original long-side IC values reproduce exactly:

| Month | Legacy base-target IC | 12h MFE IC | Exact net IC |
|---|---:|---:|---:|
| February | 0.155 | 0.105 | 0.090 |
| March | 0.162 | 0.127 | 0.093 |
| April | 0.226 | 0.188 | 0.143 |

Thus base ordering becomes more relevant to both MFE and exact net. The
remaining failure is tail level, width and conversion:

| Month | Global top 1% net | Top 5% | Top 10% | Top 20% |
|---|---:|---:|---:|---:|
| February | -8.90 bps | -40.48 | -50.87 | -67.92 |
| March | -20.46 bps | -67.76 | -83.03 | -89.12 |
| April | **+16.62 bps** | -39.93 | -58.35 | -76.59 |

April therefore contains a positive extreme 1% base tail, but that signal
does not remain profitable at the traded 5--20% widths. It is not a pure
global-IC contradiction: useful rank information is too weak relative to the
approximately 100-bps explicit policy cost outside a very thin tail.

Top-10 waterfall:

| Month | MFE ceiling | Deployed gross | Explicit cost | Exact net | Positive precision | CVaR5 |
|---|---:|---:|---:|---:|---:|---:|
| February | 365.77 bps | 49.38 | 100.25 | -50.87 | 50.11% | -892.35 |
| March | 285.27 bps | 17.05 | 100.09 | -83.03 | 42.68% | -769.52 |
| April | 297.30 bps | 41.86 | 100.21 | -58.35 | 45.82% | -705.21 |

February -> March is not primarily a candidate-composition change. At fixed
rank/side/asset cells, the top-10 exact-net delta is:

- composition: +0.36 bps;
- within-cell payoff: **-32.47 bps**;
- MFE within-cell change: -80.98 bps;
- deployed-gross within-cell change: -32.64 bps;
- explicit-cost within-cell change: -0.16 bps.

March -> April recovery combines +6.53 bps composition and +18.10 bps
within-cell payoff. Explicit cost again contributes only about +0.12 bps;
the recovery is gross-payoff conversion, not cheaper trading.

Side-local top-10 exact net further identifies changing side quality:

| Month | Long | Short |
|---|---:|---:|
| February | -59.39 bps | -20.16 bps |
| March | -91.31 bps | -45.99 bps |
| April | -38.45 bps | -82.31 bps |

April's aggregate improvement is long-led while short deteriorates sharply.
This supports a side/domain reliability problem rather than a single pooled
calibration repair.

Twenty-bin response curves remain broadly increasing
(pooled bin-to-net IC 0.982/0.986/0.983), but contain 3/5/2 adjacent
monotonicity violations. Raw base cutoff ties are absent at every tested
width, so deterministic tie handling does not explain its negative tails.
By contrast, the upstream `base_expected_ev` stream has material plateaus:
March global top-1 tie sensitivity is 90.47 bps and top-10 is 18.67 bps.

On the strict March--April canonical residual subset, the combined
`residual_expected_ev` stream is the best tested canonical top-10:

| Score | March | April |
|---|---:|---:|
| Raw base alpha | -65.71 bps | -33.94 bps |
| Upstream base expected EV | -31.88 | -54.30 |
| Residual delta component | -42.74 | -53.63 |
| Combined residual expected EV | **-26.45** | **-24.32** |

The residual layer improves tail economics but does not cross zero. The
reconstructed direct-EV source is not exact-policy parity: its fee-only top-10
is +8.46 bps in February, -25.33 in March and -62.69 in April. Treat that as
diagnostic evidence of direct-head instability, never as a promotion result.

Remaining waterfall materialization is deliberately source-separated:

1. ~~Join strict March--April residual identities to the raw direct-q OOF
   source for an exact 140,682-row all-score comparison.~~ Completed below.
2. Build a May--early-July exact-policy subset using the 127,777 rows that
   have raw MFE/exit data; do not replace later missing MFE with ATR units.
3. Build the complete 5,760-row July 20--23 forward-score bridge, flagged as
   retrospective/non-OOS performance evidence.
4. Define a named executable target-price/barrier realization rule before
   adding `attainable_gross`; until then retain MFE only as a ceiling.

#### Exact March--April all-score bridge completed

The exact 140,682-row base/residual/direct-q comparison is materialized at:

`data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/`

Implementation:

- `scripts/materialize_marapr2025_all_score_ic_ev_waterfall.py`;
- `tests/test_materialize_marapr2025_all_score_ic_ev_waterfall.py`.

The bridge hash-verifies both upstream sources, uses the complete four-field
identity, and now additionally requires exact equality of realized net
outcome and label-resolution timestamp. The direct source is read with an
explicit raw-column projection, so `mapped_q25_bps` is neither read nor
emitted. The focused source-separated and all-score waterfall suites pass
11/11.

This comparison separates two issues that were previously conflated.

First, the base-score paradox is real for February -> March, but not for
March -> April. In the broader canonical February--April ledger, base-target
IC rises from 0.155 to 0.162 while top-10 exact net falls from -50.87 to
-83.03 bps. The fixed-composition waterfall attributes almost none of that
drop to candidate mix or explicit costs: within-cell MFE falls by 80.98 bps
and deployed gross by 32.64 bps. The base preserves modest broad ordering,
but the favorable-payoff distribution inside nominally similar high-rank
cells deteriorates.

On the exact strict March--April all-score subset, the base score then
improves in both full exact-net IC and tail economics:

| Raw base alpha | March | April |
|---|---:|---:|
| Exact-net rank IC | 0.067 | 0.112 |
| Global top 1% exact net | -1.58 bps | +63.81 bps |
| Global top 5% exact net | -42.50 bps | -7.44 bps |
| Global top 10% exact net | -65.71 bps | -33.94 bps |
| Global top 20% exact net | -74.37 bps | -53.56 bps |

Thus April is not evidence that higher base IC causes lower EV. It is
evidence that useful broad ordering is concentrated in an extremely thin
tail and is not strong enough to clear the approximately 100-bps explicit
cost at the deployed 5--20% widths.

Second, the raw direct q25 head has a distinct and more severe April failure:

| Raw direct q25 | March | April |
|---|---:|---:|
| Exact-net rank IC | 0.093 | 0.086 |
| Global top 1% exact net | +107.57 bps | -56.96 bps |
| Global top 5% exact net | +8.74 bps | -112.45 bps |
| Global top 10% exact net | -21.76 bps | -93.24 bps |
| Global top 20% exact net | -49.03 bps | -77.38 bps |
| Top-10 within-tail net IC | +0.184 | **-0.092** |
| Top-10 positive precision | 50.0% | 42.2% |
| Top-10 CVaR5 | -723.73 bps | **-916.65 bps** |

The direct head therefore retains weak positive global IC in April while its
selected tail becomes internally anti-ranked and much more adversely skewed.
This is a tail-transfer/calibration failure, not a base-alpha conversion
failure. It is also overwhelmingly short-side: April direct-q top-10 is
+5.02 bps long versus -155.02 bps short. A pooled mapper must not be used to
hide that side-local breakdown.

The combined residual expected-EV stream is less unstable:

| Combined residual expected EV | March | April |
|---|---:|---:|
| Exact-net rank IC | 0.079 | 0.095 |
| Global top 1% exact net | +68.92 bps | +9.31 bps |
| Global top 5% exact net | +12.21 bps | -5.37 bps |
| Global top 10% exact net | -26.45 bps | -24.32 bps |
| Global top 20% exact net | -52.95 bps | -50.19 bps |

It improves over raw base at top 5--10%, but remains economically
insufficient. This supports the intended layered architecture only as a
diagnostic: it does not promote the residual stream or establish that the
current EV head transfers.

Required continuation:

1. ~~Extend the same exact identities and metrics through May--early July and
   the July 20--23 bridge.~~ Completed below for raw scores. Preserve pooled
   global top-k selection after the recent causal EV mapping as a separate
   source-bound policy comparison; side-local results remain attribution.
2. Add top-tail recall against ex-post profitable candidates and conditional
   payoff distributions, not only precision. Test whether February ->
   March high-base-rank candidates retain opportunity but lose capture, or
   lose the opportunity event itself under a separately defined barrier.
3. For the direct head, isolate April short-side tail failure by score
   decile, asset, transition state, liquidity/spread bucket, predicted-loss
   probability and residual/base disagreement. Require latest-month coverage
   and side-local raw IC before considering mapping.
4. Use the eventual named attainable-gross rule to split MFE opportunity from
   executable capture. Until that rule exists, do not interpret the large
   MFE-to-gross gap as a realizable trading gain.
5. Promote no score from IC alone. Every challenger must improve exact
   globally selected top-k net, positive precision and adverse CVaR at the
   intended book width, with portfolio replay only after those gates pass.

#### Exact May--July 10 all-score waterfall completed

The strict 127,777-row May--July 10 product is:

`data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/`

Implementation:

- `scripts/materialize_mayjul2026_exact_allscore_ic_ev_waterfall.py`;
- `tests/test_materialize_mayjul2026_exact_allscore_ic_ev_waterfall.py`.

The first draft was intentionally discarded before materialization because it
joined the 1,440-minute deployed-policy ledger while this workstream requires
the signed 720-minute before/after horizon. The canonical product now uses:

- exact one-minute, decision-plus-12-hour MFE/MAE/exit/gross/cost/net labels;
- strict prior-resolved side-local base OOF;
- strict residual OOF;
- the raw q25/q50 direct challenger;
- the separately trained transfer parent, adapter and reliability arms.

The residual learner's own legacy fixed-1%-cost target resolves 12 hours later
than the exact 12-hour execution endpoint on every row. This is not hidden or
coerced: residual outputs are evaluated only as OOF score arms, never as
same-target calibration.

The direct/cross-era sources encode `BCH/USD:USD` as `BCH_USD:USD`. The
materializer permits that source-local repair only after asserting that the
candidate-ID symbol, timestamp, `1h` timeframe and side all match the stored
fields. Coverage is exactly 127,777/127,777 and direct/adapter realized net
and label endpoints match the exact-policy anchor on every row. No
`mapped_*` column is read or emitted.

Raw exact-net rank IC:

| Score | May | June | July 1--10 |
|---|---:|---:|---:|
| Base alpha | 0.039 | 0.065 | **-0.118** |
| Residual expected EV | 0.047 | **0.096** | **-0.104** |
| Direct q25 challenger | 0.027 | 0.080 | **-0.100** |
| Transfer parent | 0.043 | 0.081 | **-0.092** |
| Transfer adapter | 0.037 | 0.076 | **-0.088** |

Raw pooled-global top-10 exact net:

| Score | May | June | July 1--10 |
|---|---:|---:|---:|
| Base alpha | -76.76 bps | -99.75 | -143.89 |
| Base expected EV | -74.93 | -80.21 | -142.93 |
| Residual expected EV | **-64.08** | -75.79 | **-133.55** |
| Direct q25 challenger | -100.64 | -42.91 | -152.92 |
| Transfer parent | -91.05 | **-38.18** | **-133.05** |
| Transfer adapter | -99.45 | -43.27 | -133.49 |

Every raw arm and month remains negative at the traded top-10 width. June is
the least-bad direct/transfer month, then July inverts the ranking surface.
For the base score, July top-10 precision is 19.2%, mean MFE is 151.44 bps,
deployed gross is -44.11 bps and exact net is -143.89 bps. For direct q25,
July precision is 19.6%, gross is -53.19 bps and net is -152.92 bps.

Twenty-bin pooled response confirms a regime inversion rather than only one
bad cutoff:

| Score | May bin-to-net IC | June | July 1--10 |
|---|---:|---:|---:|
| Base alpha | 0.773 | 0.826 | **-0.929** |
| Residual expected EV | 0.277 | 0.983 | **-0.862** |
| Direct q25 | 0.869 | 0.817 | **-0.468** |
| Transfer adapter | 0.839 | 0.881 | **-0.439** |

The top-10 fixed-composition decomposition localizes the June -> July drop:

| Score | Composition effect | Within-cell net effect | Within-cell MFE | Within-cell gross | Cost effect |
|---|---:|---:|---:|---:|---:|
| Base alpha | +23.34 bps | **-68.65** | -137.58 | -69.00 | -0.34 |
| Residual expected EV | +11.23 | **-74.90** | -167.12 | -75.28 | -0.38 |
| Direct q25 | -6.70 | **-116.56** | -245.15 | -117.15 | -0.59 |
| Transfer adapter | -4.73 | **-94.78** | -257.23 | -95.26 | -0.48 |

July degradation is therefore predominantly a within-cell opportunity and
capture failure. Candidate composition partly helps the base/residual arms,
and explicit fee cost is nearly unchanged. This reinforces the domain/
conversion diagnosis: July changes the economic meaning of familiar score
cells.

These are deliberately **pre-map raw-score diagnostics**. The trading policy
remains one pooled global top-k after the causal recent-EV mapping. A mapped
score may only be evaluated through its own causal, resolved-before-snapshot
lineage; it must not be reconstructed by fitting a new map on these outcomes.

#### July 20--23 exact retrospective all-score bridge completed

The complete bridge is:

`data_perp/artifacts/july20_23_retrospective_allscore_bridge_20260730_v1/`

Implementation:

- `scripts/materialize_july20_23_retrospective_allscore_bridge.py`;
- `tests/test_materialize_july20_23_retrospective_allscore_bridge.py`.

It binds 5,760 exact identities, 2,880 per side, across frozen Pack-B,
pre-entry heads, final direct/capture heads, the raw q25/q50 challenger,
transfer-adapter scores and exact one-minute 12-hour outcomes. All persisted
score-stage availability timestamps are at or before decision. The direct-q
and adapter score files do not contain their own availability timestamps, so
they inherit decision lineage only through the exact hash-bound Pack-B join.

Every output is stamped
`RETROSPECTIVE_NONPROMOTABLE_NOT_OOS_RAW_SCORES_ONLY`. The candidate surface
was retrospectively materialized; the term `base_oof_score` is model-lineage
nomenclature and does not make July 20--23 OOS evidence. Mapped EV, admission
flags and global rank are excluded because the frozen mapped-policy report is
a separate evidence source.

Primary raw results:

| Score | Exact-net IC | Top 1% | Top 5% | Top 10% | Top 20% |
|---|---:|---:|---:|---:|---:|
| Base alpha | 0.031 | -111.97 | -125.40 | **-125.14** | -114.07 |
| Existing alpha EV | 0.036 | -144.70 | -122.08 | -132.05 | -125.91 |
| Final direct net | **0.061** | -122.48 | -149.72 | **-144.86** | -145.34 |
| Margin/capture interaction | **0.062** | -194.67 | -145.11 | -153.57 | -147.30 |
| Direct q25 challenger | -0.025 | -207.03 | -143.80 | -136.83 | -142.80 |
| Transfer parent | -0.043 | -201.40 | -162.92 | -155.29 | -147.07 |
| Transfer adapter | -0.052 | -174.91 | -156.61 | -161.09 | -145.98 |

The final direct head is another concrete IC-to-EV warning: it has the best
positive aggregate exact-net IC, yet its top-10 loses 144.86 bps. Its selected
tail has 25.2% positive precision, -661.46-bps CVaR5, 174.74-bps MFE,
-45.09-bps deployed gross and approximately 99.78-bps explicit cost.

No primary raw score has a positive pooled-global top-10 on any individual
day from July 20 through July 23. Base top-10 ranges from -100.4 to -148.9
bps; final direct from -186.9 to -96.6; direct q25 from -227.5 to -74.3; and
the transfer adapter from -262.2 to -63.1. The result is broad, not one bad
day.

Side attribution remains unstable. Final-direct top-10 is -86.93 bps long
versus -166.70 short; clean-favorable probability reverses that pattern at
-184.62 long versus -92.19 short. Neither is a side-local promotion case, and
no side quota may replace the policy's pooled-global ranking.

#### Named attainable-gross barrier rule

The minimum executable target-price diagnostic is now formalized in
`extreme_price_movements/execution_entry_timing_meta.py` as:

```text
AGBR-L60-K0.25-v1
action_id = adverse_limit_60m_0.2500atr
```

At decision price `P0`, decision ATR `A` and side sign `s` (`+1` long,
`-1` short):

```text
raw_limit = P0 - s * 0.25 * A
expiry = decision + 60 minutes, inclusive
```

A long fills when exact-1m low touches the limit; a short fills when high
touches it. The fill is the stated limit, never an optimistic intrabar price.
Because OHLC cannot order the touch and same-bar excursion, protected exit
replay starts on the next minute while retaining the fill price. Post-fill
gross comes from the frozen re-anchored exit-policy simulator, with executable
spread embedded and fee deducted once.

The barrier-realized label is:

```text
filled AND post_fill_gross_ev >= fee_return + target_net_buffer
```

The initial fixed buffer is zero. MFE is intentionally not an input. A no-fill
utility is `-max(enter_now_net_ev, 0)`, so avoiding a trade cannot be labelled
as a free success when it loses a profitable opportunity.

The rule does not change the frozen `enter_now` policy or default action grid.
The existing 30-bp/12-hour timing study selected adverse limits frequently but
lost -39.03 bps versus enter-now; the only positive narrow gate was +0.18 bps,
did not recur and selected no adverse limits. Any new use must rematerialize
AGBR under the exact signed policy SHA, fee, spread, geometry and horizon.
Touch-fill OHLC labels also do not model exchange queue priority, so the rule
is research-only and may not emit live limit orders without tick/rounding and
queue/fill infrastructure.

Focused validation across the timing, waterfall, exact-policy causal mapping
meaningful-MFE tail-recall and frozen-book attribution stack passes 41/41.

#### Remaining IC-to-EV continuation

1. **Completed:** publish and compare the existing causal recent-EV mapped
   score streams against identity/no-map on the same exact-policy population,
   without fitting on evaluated outcomes. The primary policy metric remains
   one pooled global top-k. Results and the no-promotion decision are recorded
   below.
2. Investigate the apparent base-alpha IC versus execution-EV contradiction
   as a first-class workstream, rather than accepting it as an architectural
   consequence. The February -> March example is especially important:
   base-target rank IC rises from 0.155 to 0.162 while the long-side
   **base-score-ranked top-decile exact-policy net** falls from approximately
   -59 to -91 bps. These figures are not direct-EV-head performance. For every
   score/month transition:

   - bind both statistics to the same exact candidate identities and report
     base-target IC, exact-policy gross/net IC and top 1/5/10/20%;
   - separate target and horizon mismatch (base alpha target versus exact
     12-hour policy net), score calibration/compression and cutoff ties;
   - report tail-conditional IC, monotonic response-bin shape and whether the
     IC gain occurs outside the globally traded top-k;
   - decompose the tail change into candidate composition, within-cell
     meaningful-MFE incidence, MFE size conditional on incidence, exit
     capture/gross, explicit spread/fee cost and downside/CVaR;
   - attribute reversals by month, causal transition/regime state, side,
     asset, liquidity/spread bucket and base/direct disagreement;
   - test whether stronger base alpha is selecting larger theoretical
     excursions that the frozen exit policy cannot monetize, or merely
     reordering many small/low-value observations;
   - require bootstrap uncertainty and minimum tail counts before calling an
     IC or EV change material.

   This investigation is diagnostic. It must not alter the frozen exit policy,
   refit mappings on evaluated outcomes, introduce per-timestamp top-k or
   promote a score because aggregate IC alone improved.
3. **Completed:** extend the decomposition with independently defined
   meaningful-MFE barrier incidence, clean competing-risk incidence,
   row-cost-aware path opportunity and global-tail recall. The result is
   recorded below. The old `net > 0` opportunity alias is not used as an
   opportunity event.
4. **Completed:** attribute the July direct-head failure by causal transition
   features,
   liquidity/spread bucket, asset, side, base/direct disagreement and
   predicted adverse-loss probability. Do not use active transition
   probabilities after their source coverage ends. The older strict-OOF
   transition overlay was inspected but excluded: it has no per-row
   availability timestamp, is tied to the older correction/mapping lineage
   and its persisted execution outcome differs from the signed exact-policy
   anchor. The static spread-universe baseline was also excluded because it is
   explicitly non-PIT June--July metadata. The completed attribution and
   source boundary are recorded below.
5. Test AGBR only as a paired action label after an EV arm passes raw and
   mapped global admission. Require exact 1m paths, the same enter-now and
   post-fill policy engine, recurrence, missed-opportunity cost and portfolio
   replay before any action promotion.

#### Corrected exact-policy causal mapping comparison completed

The canonical exact-policy publisher is:

`data_perp/artifacts/current_exact_policy_global_book_mapping_source_20260730_v3/`

It persists the raw execution score, causal 21-day global isotonic map and
causal 21-day side-shrunk isotonic map together with exact signed-policy
gross, cost, net, MFE, MAE and exit metadata. A mapping snapshot may use only
labels whose resolution timestamp is strictly before the snapshot. There are
114,096 strict mapped-OOF rows, 7,112 frozen-forward rows and 2,616 warm-up
rows; the waterfall below evaluates only the strict mapped-OOF population and
does not refit a mapping.

The canonical comparison is:

`data_perp/artifacts/causal_mapping_ic_ev_waterfall_20260730_v1/`

Pooled-global top-10 exact-policy net:

| Score | May | June | July |
|---|---:|---:|---:|
| Causal global map | -84.83 bps | -84.92 | -131.26 |
| Causal side-shrunk map | -102.91 | -109.23 | -137.96 |
| Raw execution score | -115.68 | **-69.06** | **-105.09** |

The global map repairs May relative to raw by 30.84 bps, but loses 15.87 bps
in June and 26.17 bps in July. The side map is worse than the global map in
all three months. Every arm and month remains negative at top-10; alternative
1/5/20% widths are also negative. The mapping therefore does not solve July
and can erase the relatively better raw June ranking. Neither global nor side
mapping is promoted.

An earlier apparent positive mapped result was traced to
`execution_ev_economic_failure_diagnosis_20260727_v2/diagnostic_rows.parquet`.
All 114,096 outcomes differed from the signed 12-hour exact-policy anchor,
including materially different gross/net paths and a lower mean cost. That
artifact was an obsolete target/cost lineage, not evidence of profitable
mapping. The invalid comparison was recoverably quarantined at:

`data_perp/artifacts/causal_mapping_ic_ev_waterfall_20260730_v1_INVALID_OBSOLETE_TARGET/`

It must not be cited or used for model selection.

#### Independent meaningful-MFE incidence and global-tail recall completed

The canonical diagnostic is:

`data_perp/artifacts/meaningful_mfe_tail_recall_20260730_v1/`

Implementation:

- `scripts/materialize_meaningful_mfe_tail_recall.py`;
- `tests/test_materialize_meaningful_mfe_tail_recall.py`.

It joins all 127,777 canonical May--July 10 score identities to each of four
predeclared meaningful-MFE grids with exact decision-time and exact-policy-net
parity. The primary grid is `h12_u1p5atr`; `h12_u2p0atr` is threshold
sensitivity, while both 24-hour cells are sensitivity-only because they see
12 hours beyond the deployed execution target.

Four outcomes remain deliberately distinct:

1. **Any meaningful-MFE touch:** peak favorable return reaches the actual
   `max(1.5 ATR, 1.5%)` upper barrier at any time in 12 hours.
2. **Clean favorable-first:** that upper barrier is reached before the
   1-ATR adverse barrier; same-hour conflict is adverse.
3. **Exact-cost path opportunity:** exact 12-hour MFE exceeds that row's
   exact execution cost.
4. **Captured positive net:** the frozen exact-policy replay finishes above
   zero net.

The hourly barrier labels do not replay the exit policy and therefore cannot
be called executable EV. Conversely, captured-positive net is not relabelled
as opportunity. All selection remains month-level pooled global top
1/5/10/20; side views are attribution only.

Primary-grid pooled-global top-10:

| Score | Month | Any-touch rate / recall lift | Clean-first rate / recall lift | Positive net given touch: population -> tail | MFE / gross / net |
|---|---|---:|---:|---:|---:|
| Base alpha | May | 55.9% / 1.18x | 48.3% / 1.30x | 60.8% -> 67.0% | 247.6 / 23.4 / -76.8 bps |
| Base alpha | June | 56.2% / 1.08x | 46.8% / 1.19x | 72.1% -> 69.4% | 281.0 / 0.3 / -99.7 |
| Base alpha | July | **41.5% / 0.88x** | **33.9% / 0.93x** | **61.1% -> 45.6%** | **151.4 / -44.1 / -143.9** |
| Residual expected EV | May | 55.3% / 1.17x | 48.0% / 1.29x | 60.8% -> 74.5% | 282.0 / 36.1 / -64.1 |
| Residual expected EV | June | 57.4% / 1.10x | 47.3% / 1.20x | 72.1% -> 73.4% | 341.9 / 24.3 / -75.8 |
| Residual expected EV | July | **42.6% / 0.90x** | **34.9% / 0.96x** | **61.1% -> 56.5%** | **170.9 / -33.7 / -133.5** |
| Direct q25 | June | 65.1% / 1.25x | 52.2% / 1.32x | 72.1% -> 76.5% | 387.0 / 57.4 / -42.9 |
| Direct q25 | July | **40.7% / 0.86x** | **29.1% / 0.80x** | **61.1% -> 46.5%** | **148.8 / -53.2 / -152.9** |
| Transfer parent | June | 64.7% / 1.24x | 51.9% / 1.32x | 72.1% -> 77.6% | 390.4 / 62.1 / -38.2 |
| Transfer parent | July | **44.8% / 0.94x** | **33.0% / 0.90x** | **61.1% -> 46.5%** | **160.7 / -33.2 / -133.1** |

July therefore has two distinct failures:

- **opportunity recognition reverses:** every main score selects any-touch
  and clean-first events at or below random recall in the global top decile;
- **conversion also degrades:** conditional positive-net capture inside
  selected opportunity rows falls below the July population rate, sharply
  for base, direct-q and transfer.

This rules out the simple explanation that alpha still finds the same
opportunities and only the fee or exit policy deteriorates. The score surface
selects fewer/lower-quality opportunities in July, and the exit policy
monetizes those selected opportunities less effectively. The next attribution
must model these as separate dependent variables: opportunity incidence,
adverse competing risk and capture conditional on opportunity.

The artifact is diagnostic-only, fits no model, reads no mapped score and is
not promotion evidence. Its source grid is the newer exact-policy grid; the
older `meaningful_mfe_label_grid_ablation_20260727_v1` used a different grid
and its classifier metrics must not be reused as exact-policy evidence.

#### May--July frozen-book failure attribution completed

The canonical attribution is:

`data_perp/artifacts/mayjul_failure_attribution_20260730_v1/`

Its `attribution_rows.parquet` is the exact 127,777-row handoff for the next
bounded ablations. It carries immutable identities, separate opportunity/
competing-risk/capture targets and their resolution timestamps, strict-OOF
risk predictions and availability, causal transition context with an explicit
coverage mask, raw score arms, same-decision rank disagreement and frozen
monthly-global top-10 membership flags. Targets and context coexist in this
research table; training consumers must select feature columns explicitly and
enforce the recorded target-resolution cutoff.

Implementation:

- `scripts/materialize_mayjul_failure_attribution.py`;
- `tests/test_materialize_mayjul_failure_attribution.py`.

Each raw score arm is selected once as one month-level pooled global top
decile with deterministic candidate-ID tie breaking. Side, asset, archetype,
risk and transition bands are evaluated only after that book is frozen; no
local reranking or quota is introduced.

The diagnostic binds:

- the 127,777-row exact all-score ledger and primary `h12_u1p5atr` opportunity
  grid;
- the full-coverage side-local strict-OOF MAE competing-risk predictions,
  asserting `train_decision_cutoff < decision` and
  `available_at <= decision` on every row;
- only fields catalogued as `decision_time_feature` in the v4 cross-era
  transition panel. Current strict-OOF context is observed at anchor minus one
  hour. Missing hours are an explicit `UNAVAILABLE` slice and never backfilled;
- pre-May, outcome-free transition-panel anchors to freeze all transition-band
  thresholds.

The causal liquidity view is the decision-time global spread-proxy context.
The current static per-asset spread baseline is not point-in-time and is
excluded. Asset results are therefore outcome attribution by symbol, not a
historical liquidity feature.

Frozen global-book summary:

| Raw score | June top-10 | July top-10 | June -> July | July 95% day-bootstrap interval |
|---|---:|---:|---:|---:|
| Base alpha | -99.75 bps | -143.89 | -44.14 | [-185.30, -112.41] |
| Residual expected EV | -75.79 | -133.55 | -57.76 | [-179.88, -92.02] |
| Direct q25 | -42.91 | -152.92 | **-110.01** | [-214.80, -103.40] |
| Transfer parent | -38.18 | -133.05 | **-94.88** | [-191.07, -89.07] |

The exact fixed-slice decomposition shows that July is overwhelmingly a
**within-state failure**, not a change in observable composition:

| Direct-q25 axis | Composition | Within slice | Total |
|---|---:|---:|---:|
| Side | +12.79 bps | **-122.79** | -110.01 |
| Archetype | -5.57 | **-104.44** | -110.01 |
| Predicted adverse-risk band | -4.79 | **-105.22** | -110.01 |
| Base/direct disagreement | +3.27 | **-113.28** | -110.01 |
| ATR fraction | -2.88 | **-107.13** | -110.01 |
| ATR-compression context | -9.69 | **-100.32** | -110.01 |
| Directional-risk-skew context | -3.39 | **-106.61** | -110.01 |
| Memory-asymmetry context | -1.31 | **-108.70** | -110.01 |
| Global spread-proxy context | -4.15 | **-105.85** | -110.01 |
| Leverage-build context | +5.65 | **-115.66** | -110.01 |

The conclusion recurs for base, residual and transfer-parent arms. Observable
composition usually explains little and sometimes offsets the decline. A
gating rule based on one current transition band is therefore unlikely to
repair July; the economics changed inside familiar states.

Important localized findings:

- raw direct-q25 top-10 changes from 90.0% short in June to 100% short in
  July. This is a side-scale/calibration warning for that raw challenger, but
  not the explanation for July: under June side economics the composition
  shift contributes +12.79 bps, while within-short deterioration contributes
  -122.79 bps;
- the direct-q25 `dead_timeout` archetype grows from 26.4% to 36.4%, while its
  net falls from -48.37 to -205.42 bps, any-touch incidence from 56.5% to
  20.7% and conditional positive-net capture from 72.7% to 32.5%;
- the historically defined memory-asymmetry B2 band is the largest supported
  July shortfall cell: 31.4% of the direct book at -220.36 bps, with only
  24.7% any-touch incidence;
- all major transition bands deteriorate between June and July. For example,
  ATR-compression B2 moves from +23.76 to -114.20 bps and
  directional-risk-skew B1 from -17.59 to -119.89 bps;
- the strict-OOF adverse-risk proxy rises only modestly in the direct book
  from 0.341 to 0.385 and does not separate July net: its two populated broad
  bands are both approximately -153 bps. Its target uses a different
  0.5R competing-risk geometry, so this is an association test, not a
  calibration comparison against the 1ATR/1.5ATR grid;
- asset exposure is broad rather than dominated by one token: direct q25 has
  107 July assets, a 2.9% largest-asset share and HHI 0.0149. Individual bad
  assets exist but no single asset explains the inversion.

Direct/base overlap is also low and unstable: Jaccard is 0.130 in May, 0.072
in June and 0.108 in July. The direct-only replacement cohort improves on
base-only in June (-41.30 versus -107.01 bps) but loses that advantage in July
(-148.86 versus -137.65). The common July cohort is worse still at -169.74
bps. This is a regime-wide loss of ranking meaning, not merely a handful of
bad replacements.

All results remain raw-score diagnostics. The deployed policy still ranks one
global book after causal recent-EV mapping. No raw side composition, asset
slice or transition cell is itself a promotion or trading rule.

#### Required next exact-label regime ablations

The attribution rejects a simple static regime gate. The next work must test
whether the July relationship is learnable and transferable when opportunity
and capture are modelled separately:

0. **Base-IC-to-execution-tail bridge (mandatory cross-cutting diagnostic).**
   Do not treat improving base-target IC with deteriorating exact-policy tail
   EV as an expected architectural consequence. The approximately
   `-59/-91/-38`-bps February/March/April figures are the exact-policy outcomes
   of the **long-side base-score-ranked** top decile, not direct-EV-head
   performance; every report must materialize those two score lineages side by
   side and name them unambiguously.

   On identical candidate IDs, by side and month, carry each frozen base,
   residual and direct-EV score through the same bridge:

   ```text
   native base target
   -> exact 12h any-touch / clean-first incidence
   -> conditional MFE magnitude
   -> attainable gross under each frozen exit
   -> deployed-exit gross
   -> exact net after row-specific cost
   ```

   At every step report full-sample rank IC, global top-1/5/10/20% economics,
   event precision/recall, positive-net precision, loss rate, CVaR, support,
   score compression, response-bin monotonicity and cutoff-tie sensitivity.
   Decompose each month-to-month tail delta into candidate composition versus
   within-cell changes in opportunity prevalence, favorable magnitude,
   adverse/timeout payoff, exit-family mixture, gross capture and explicit
   cost. Run rank-preserving month swaps and fixed-composition reweighting to
   quantify these components, with bootstrap intervals.

   This item is resolved only when the IC-to-EV delta is quantitatively
   attributed. Any material decision-time explanation must then be tested OOF
   as an event, capture, reliability or calibration input. Aggregate IC alone
   cannot promote a model, and the frozen exit policy must not be changed
   inside this diagnostic.

   The apparent improvement in base-target IC must remain an explicit
   falsification target, even though the first waterfall below already shows
   that correlation and economic level can diverge. For each adjacent month
   and side, add the following fixed tests to every subsequent event/simplex
   ablation:

   - measure the same frozen base score against the native base target, exact
     12-hour MFE, deployed gross and exact net on identical rows, then report
     both full-population IC and IC restricted to the globally selected
     1/5/10/20% tails;
   - separate a change in ordering from a change in payoff level by reporting
     response-bin slopes and intercepts, opportunity prevalence, conditional
     favorable magnitude, adverse/timeout magnitude and positive-net
     precision. A higher rank IC with a lower response intercept is a
     conversion/domain failure, not improved executable alpha;
   - freeze February rank cells and reweight March into them, then perform the
     reverse swap. Repeat for March/April. Report composition, within-cell MFE,
     within-cell deployed gross, within-cell exact net and explicit-cost
     contributions with day-block bootstrap intervals;
   - replay the already frozen exit-policy alternatives on the same selected
     paths only as attribution. If MFE remains strong but every valid exit
     loses capture, classify the failure as exit conversion; if independent
     any-touch/clean-first incidence falls, classify it as opportunity
     recognition. Do not tune or select an exit on the evaluated month;
   - test score saturation and tail-width sensitivity explicitly. The work
     must distinguish broad IC gains driven below the admission cutoff from a
     genuinely stronger extreme tail, and must quantify whether the
     economically viable tail is materially thinner than the configured
     global top-k;
   - fit no corrective mapper inside this analysis. Any proposed transition,
     reliability, opportunity or capture feature discovered by the
     attribution must be evaluated later with train-only fitting and
     source-forward availability.

   The current leading diagnosis is therefore a hypothesis to challenge, not
   a closed explanation: February -> March preserves slightly better broad
   ordering while the favorable-payoff distribution and deployed gross fall
   inside comparable high-rank cells; March -> April partly recovers through
   long-side gross conversion, while the economically positive signal remains
   concentrated in a much thinner tail than the configured global top 10%.

1. **Exact event-model reset.** Retrain the meaningful-MFE family against the
   new exact-policy grid; do not reuse metrics from the older grid. Compare
   logistic, LightGBM and CatBoost per side for:

   - any upper-barrier touch;
   - clean favorable-first versus adverse competing risk;
   - positive-net capture conditional on any touch;
   - positive-net capture conditional on clean favorable-first.

   Primary is `h12_u1p5atr`; `h12_u2p0atr` is threshold sensitivity. Twenty-
   four-hour cells are support-label sensitivity only and cannot replace the
   12-hour deployed target.

2. **Transfer matrix and within-July learnability.** Report May->June,
   June->July, July->June and grouped-day OOF within July. Walk-forward is not
   required for this diagnosis, but rows from one UTC day must remain in one
   fold. If July grouped OOF is weak, there is no evidence for a learnable
   July-specific expert. If July OOF is strong but both cross-period
   directions fail, proceed to regime-conditioned experts.

3. **Base -> residual opportunity architecture.** Match the existing
   `config.py` base-versus-meta feature contracts:

   - base event learners use the side-specific base feature sets;
   - residual/meta learners receive strictly OOF base logits/probabilities,
     score/margin context, strict-OOF auxiliary probabilities and the
     decision-time transition feature family;
   - transition context is added to the residual/meta layer first, not mixed
     into every base learner by default;
   - compare monolithic, base-only and base->residual arms with identical
     folds and candidate identities.

4. **Opportunity/capture EV composition.** Compare:

   ```text
   P(any touch) × E(net | touch)
   P(clean first) × E(net | clean first)
   full competing-risk state simplex × conditional net payoffs
   direct exact-net residual
   ```

   Payoffs are already exact net; no second cost deduction is permitted. A
   blend may be considered only if it improves the same global tail in every
   required period.

5. **Train-only label geometry/economic HPO.** Within grouped training folds,
   test the 1.5/2.0-ATR upper grid, adverse threshold sensitivity and a new
   per-row exact-cost-plus-buffer label. Keep event reachability separate from
   exit capture. Select geometry on AP/Brier plus globally selected top-10
   exact net, not AUC alone.

6. **Side-scale/global-book calibration.** Because raw direct-q25 becomes
   100% short in July, compare no map, causal global map and causal side-
   calibrated-to-global map on identical OOF predictions. Side normalization
   may correct incomparable raw scales but may not impose a side quota. The
   final selection remains one pooled global top-k.

7. **Promotion sequence.** A candidate must pass, in order:

   - grouped OOF learnability and latest-month coverage for its own target;
   - cross-period opportunity and conditional-capture diagnostics;
   - causal recent-EV mapping with no evaluated-outcome refit;
   - one pooled global top-1/5/10/20 exact-policy economics, uncertainty and
     recurrence;
   - frozen simple-policy and portfolio replay with concurrency, exposure and
     asset limits.

Only after an EV arm passes those gates may the separate timing/MAE/
target-price action layer and AGBR paired-action experiment resume.

#### Primary exact-grid event/capture reset completed (2026-07-30)

The primary `h12_u1p5atr` reset is complete:

- runner:
  `scripts/run_meaningful_mfe_exact_grid_reset.py`;
- tests:
  `tests/test_run_meaningful_mfe_exact_grid_reset.py`;
- artifact:
  `data_perp/artifacts/meaningful_mfe_exact_grid_reset_20260730_v2/`.

The v2 primary reproduces every v1 parquet hash exactly and additionally binds
the current runner SHA; v1 is therefore superseded without any metric change.

The runner hash-binds the restored 134,889-row, 249-feature point-in-time
universe to the exact label grid one-to-one. It independently proves decision
time, ATR, gross and net parity between the grid and economics anchor,
`label_resolution = decision + 12h`, and `gross - exact row cost = net`.
Any-touch is derived as
`peak_mfe_atr * oof_entry_atr_fraction >= upper_return`; it is never
substituted by clean favorable-first. Conditional capture is trained only on
the declared touch or clean population. All output hashes verify and the
focused exact-grid/label/fold suite passes 15/15.

Logistic, LightGBM and CatBoost are fitted separately by side for any-touch,
clean-first, positive-net capture conditional on touch, positive-net capture
conditional on clean, and the ATR-normalized soft triple-barrier comparator.
Geometry selection uses only the purged May 1--24 -> May 25--31 inner split.
Conditional heads are not selected on their uncomposed economic ranking.
Every later score is raw and unmapped; side-scale calibration remains an
explicit next ablation.

The exact transfer economics reject promotion:

| Diagnostic | Best valid pooled global top-10 score | Exact net |
|---|---|---:|
| May -> June | logistic clean-first | -40.37 bps |
| June -> July 1--10 | LightGBM soft triple barrier | -92.62 bps |
| grouped-day July OOF | logistic any-touch | -62.48 bps |
| July -> June 1--10 reverse-time | logistic soft triple barrier | +3.76 bps |
| July -> full June reverse-time | logistic any-touch | -51.76 bps |

The reverse-time positive is duration-local and permanently non-promotable:
it disappears on full June. Population exact net is -104.40 bps in June and
-116.67 bps in July, so some heads improve the unconditional population but
none crosses costs.

July is only partly learnable:

- grouped-day July any-touch AUC is below chance for long CatBoost
  (`0.483`) but reaches `0.644` for short logistic;
- grouped-day clean-first AUC is only about `0.55--0.57` on both sides;
- conditional capture is substantially more learnable, with grouped-day AUC
  roughly `0.60--0.71`, but it is conditional on an opportunity event that is
  not reliably ranked;
- the best grouped-July composed score is still negative:
  logistic clean x conditional-capture is `-88.50` bps and logistic
  touch x conditional-capture is `-90.64` bps;
- June -> July clean-first transfer is especially weak on short
  (`0.483/0.502/0.463` AUC for CatBoost/LightGBM/logistic).

This localizes the bottleneck: conditional conversion has learnable structure,
but opportunity incidence, cross-period transfer and global side scale do not.
It also rejects a July specialist at present: one side/one event is learnable,
not the complete pooled opportunity-to-net decision. Raw global scaling is
materially unstable; for example the June-trained LightGBM soft score's July
global top decile is 100% short. This is evidence for causal side-to-global
calibration testing, not for a side quota.

The next exact steps are:

1. **Completed:** config-routed base -> residual, monolithic and disjoint-meta
   sensitivity on the identical transfers and grouped July folds;
2. **Completed:** `h12_u2p0atr` and exact row-cost-plus-buffer label
   sensitivities without replacing the deployed 12-hour primary target;
3. **In progress:** materialize the full-horizon one-minute, row-cost-aware
   clean/adverse/timeout simplex and fit conditional payoff magnitudes,
   because a positive-net classifier alone does not encode loss size;
4. **Completed:** train-only calibration ledger plus no-map, causal-global and
   causal-side-calibrated-to-global comparison on identical rows;
5. do not run portfolio replay unless an arm first produces recurrent positive
   pooled global exact-policy economics.

#### Exact 2.0-ATR threshold sensitivity completed

The fixed-process threshold sensitivity is:

- artifact:
  `data_perp/artifacts/meaningful_mfe_exact_grid_reset_h12_u2p0atr_20260730_v1/`;
- grid: exact `h12_u2p0atr`;
- same candidates, 12-hour horizon, costs, folds, model families, feature
  process and May-only HPO contract as the 1.5-ATR primary.

Every output and runner hash verifies. Raising the upper barrier reduces
event prevalence as expected:

| Evaluation | 1.5-ATR any-touch / clean | 2.0-ATR any-touch / clean |
|---|---:|---:|
| June | 52.02% / 39.37% | 42.39% / 32.26% |
| July 1--10 | 47.39% / 36.43% | 39.59% / 30.47% |

It makes conversion conditional on an event easier to learn, especially long:
grouped-July long conditional-capture AUC rises from about `0.66--0.71` at
1.5 ATR to `0.75--0.77` at 2 ATR. It does not repair reachability:

- grouped-July long any-touch remains approximately chance (`0.499--0.505`);
- short logistic any-touch remains the only strong event head (`0.642`);
- clean-first is weak (`0.526--0.532` long and `0.494--0.582` short).

Best valid pooled global top-10 economics:

| Diagnostic | 1.5 ATR | 2.0 ATR |
|---|---:|---:|
| May -> June | -40.37 bps | -56.49 bps |
| June -> July | -92.62 bps | -88.43 bps |
| grouped-July OOF | -62.48 bps | -67.13 bps |
| July -> June 1--10 reverse-time | +3.76 bps | +13.35 bps |
| July -> full June reverse-time | -51.76 bps | -43.02 bps |

The best grouped-July composed score improves modestly from `-88.50` to
`-84.07` bps, but remains deeply negative. The larger barrier therefore
sharpens conditional conversion and the narrow reverse diagnostic without
improving recurrent current-period opportunity ranking. Keep 1.5 ATR as the
primary event geometry; use 2.0 ATR only as a supporting payoff-scale target
or conditional-magnitude feature. Do not replace the deployed target or
promote a 2.0-ATR gate.

#### Exact row-cost-plus-buffer sensitivity completed

The remaining bounded label-geometry sensitivity is complete:

- runner:
  `scripts/run_meaningful_mfe_exact_grid_cost_buffer_sensitivity.py`;
- tests:
  `tests/test_run_meaningful_mfe_exact_grid_cost_buffer_sensitivity.py`;
- artifact:
  `data_perp/artifacts/meaningful_mfe_exact_grid_cost_buffer_sensitivity_20260730_v1/`.

Every output and the executed runner hash verify; the focused cost-buffer and
exact-reset suite passes 13/13. The run preserves the 134,889 exact identities,
249 signed point-in-time features, 12-hour resolution, frozen primary-v2
model geometry, side-local fitting and train-only target-specific feature
selection. No new HPO dimension was introduced.

For buffer `b` in 0/25/50/100 bps, the distinct labels are:

```text
path opportunity = execution MFE > exact row cost + b
capture            = exact policy net > b
composed score      = P(path opportunity) x P(capture | path opportunity)
```

Capture implies path opportunity on every authoritative row. Exact gross
minus the row cost equals exact net; cost is never subtracted a second time.
The conditional capture probability is never ranked alone. All books remain
one pooled global top-k; side figures are diagnostics only.

Global support is adequate but shifts sharply:

| Buffer | Path opportunity | Capture | Capture given opportunity |
|---:|---:|---:|---:|
| 0 bps | 64.03% | 33.42% | 52.19% |
| 25 bps | 57.88% | 27.10% | 46.82% |
| 50 bps | 52.32% | 22.04% | 42.12% |
| 100 bps | 42.45% | 13.98% | 32.94% |

Best pooled-global top-10 exact net within this sensitivity:

| Diagnostic | Best fixed buffer/family | Exact net |
|---|---|---:|
| May -> June | 100-bps LightGBM | -56.24 bps |
| June -> July 1--10 | 50-bps LightGBM | -99.30 bps |
| grouped-day July OOF | 50-bps logistic | -87.78 bps |
| July -> June 1--10 reverse-time | 25-bps CatBoost | -8.02 bps |
| July -> full June reverse-time | 50-bps LightGBM | -52.31 bps |

The preferred threshold is not stable across periods, and every forward,
grouped-July and reverse diagnostic remains negative. Relative to the primary
exact-grid reset, the best cost-buffer score is worse in May -> June
(-56.24 versus -40.37 bps), June -> July (-99.30 versus -92.62 bps) and
grouped July (-87.78 versus -62.48 bps).

Grouped-July head quality localizes the failure:

- long path-opportunity AUC remains only about 0.515--0.549 across buffers;
- short opportunity is materially learnable, peaking at 0.697 for the
  50-bps logistic head;
- conditional capture is moderate at low buffers but deteriorates as the
  required payoff increases; at 100 bps it is about 0.52--0.55 long and
  0.58--0.60 short;
- the best grouped-July top decile still loses 87.78 bps with CVaR5 of
  -813.84 bps.

A fixed exact-cost buffer therefore does not repair opportunity-to-capture
transfer and must not be selected as an admission threshold. Preserve these
heads only as supporting event/capture targets for the next jointly modelled
competing-risk/payoff architecture. No arm qualifies for causal mapping
promotion, action-layer work or portfolio replay.

#### Full-horizon one-minute cost-aware competing-risk labels completed

The missing target infrastructure for the next simplex/payoff comparison is
now materialized:

- materializer:
  `scripts/materialize_execution_ev_cost_aware_competing_risk_labels.py`;
- tests:
  `tests/test_materialize_execution_ev_cost_aware_competing_risk_labels.py`;
- primary meaningful-floor artifact:
  `data_perp/artifacts/execution_ev_cost_aware_competing_risk_1m_labels_20260730_v1/`;
- supporting no-1.5%-return-floor artifact:
  `data_perp/artifacts/execution_ev_cost_aware_competing_risk_1m_labels_nofloor_20260730_v1/`.

Both artifacts contain 624,808 rows: every one of 156,202 exact-policy
candidate identities at 0/25/50/100 bps. Coverage is 156,202/156,202 complete
12-hour paths. Every path contains exactly 720 finite, positive one-minute
OHLC bars with no fill, interpolation or as-of repair. All source, output and
executed-runner hashes verify; the focused suite passes 11/11.
The current 134,889-row, 249-feature point-in-time model universe joins
one-to-one to both label artifacts at all four buffers: 539,556/539,556
identity-buffer rows with no missing identity.

The labels ignore the candidate's actual policy exit and always inspect the
entire decision-to-decision+12h path from the frozen executable entry price.
For the primary geometry:

```text
upper = max(1.5 x signal-time ATR fraction,
            1.5% return,
            exact row fee cost + buffer)
lower = 1.0 x signal-time ATR fraction
```

The three hard classes are mutually exclusive timeout, adverse-first and
clean-economic-favorable-first. A same-minute favorable/adverse conflict is
adverse-first. First-touch minute/timestamp, endpoint signed margins and the
upper-barrier driver are persisted. A separate timeout-only soft viability
simplex uses endpoint barrier margins; observed hit classes remain hard
one-hot outcomes and never receive fractional contradictory labels.

The primary buffer grid exposes substantial nominal redundancy:

| Buffer versus primary 0 bps | Upper changed | Hard class changed |
|---:|---:|---:|
| 25 bps | 0 / 156,202 | 0 / 156,202 |
| 50 bps | 49,584 / 156,202 | 42 / 156,202 |
| 100 bps | 117,105 / 156,202 | 6,555 / 156,202 |

The no-1.5%-floor supporting geometry is materially more sensitive:

| Buffer versus no-floor 0 bps | Upper changed | Hard class changed |
|---:|---:|---:|
| 25 bps | 66,289 / 156,202 | 2,226 / 156,202 |
| 50 bps | 86,565 / 156,202 | 4,981 / 156,202 |
| 100 bps | 117,105 / 156,202 | 11,496 / 156,202 |

Do not train redundant primary 25-bps models, and treat primary 50 bps as a
parity/geometry audit rather than an independent target. The bounded model
comparison should use primary 0 and 100 bps plus the genuinely distinct
no-floor 0/25/50/100-bps sensitivities. Buffer/geometry selection remains
train-only. These are target artifacts, not decision-time features or model
evidence.

The primary 0-bps hard classes are economically ordered in every month and
side, which validates them as supporting labels:

| Month / side | Clean favorable mean net | Adverse-first mean net | Timeout mean net |
|---|---:|---:|---:|
| May long / short | +93.2 / +66.2 bps | -200.9 / -203.6 bps | -87.6 / -64.2 bps |
| June long / short | +109.8 / +105.8 bps | -261.5 / -204.6 bps | -84.7 / -48.6 bps |
| July long / short | +65.4 / +31.2 bps | -194.6 / -197.3 bps | -83.3 / -69.8 bps |

The July short clean-favorable payoff falls from +105.8 bps in June to only
+31.2 bps even though the event remains economically positive on average.
This is direct evidence that event discrimination and conditional payoff
transfer must be modelled and evaluated separately.

Exact `execution_cost_return` is the signed policy fee component. Executable
entry spread is embedded in the frozen entry price and spread-aware exit drag
is reflected in exact gross/capture outcomes. The first-touch label is
therefore a path-opportunity ceiling; it is not executable net EV. The
downstream conditional-gross composition must retain exact exit conversion
and subtract the row fee exactly once.

#### Exact config-routed base -> residual sensitivity completed

The completed authoritative artifact is:

- runner:
  `scripts/run_meaningful_mfe_exact_grid_base_residual_sensitivity.py`;
- tests:
  `tests/test_run_meaningful_mfe_exact_grid_base_residual_sensitivity.py`;
- artifact:
  `data_perp/artifacts/meaningful_mfe_exact_grid_base_residual_sensitivity_20260730_v2/`.

The earlier v1 artifact is superseded: v2 adds the required monolithic
comparator and binds the current runner SHA. All v2 output/report hashes and
the runner hash verify. Focused exact-grid and architecture tests pass 12/12.

The signed 249-feature universe intersects the current `config.py` routing as:

- base: 112 long / 119 short features;
- meta: 148 features;
- base/meta overlap: 40 per side;
- disjoint-meta sensitivity: 108 features per side.

This is the exact signed-universe intersection, not the full legacy
591/608-base and 1,248-meta loader surface. Within each side, target and outer
split, base-only and both residual arms share the identical fitted base and
identical strictly chronological cross-fitted base probabilities. The
residual receives no in-sample base score. Shrinkage is frozen from the prior
architecture study at `0.50` long and `0.25` short. The monolithic arm uses
the direct union of available base and meta features.

Pooled global top-10 exact-policy economics:

| Transfer / target | Base only | Monolithic union | Config residual | Disjoint residual |
|---|---:|---:|---:|---:|
| May -> June any-touch | -78.32 | -83.47 | **-68.00** | -71.61 |
| May -> June clean-first | -71.14 | -82.53 | **-62.63** | -74.95 |
| June -> July any-touch | **-120.64** | -132.63 | -127.70 | -135.21 |
| June -> July clean-first | **-115.78** | -136.77 | -129.11 | -128.17 |
| grouped-July OOF any-touch | **-101.40** | -137.48 | -123.14 | -122.78 |
| grouped-July OOF clean-first | **-83.07** | -122.46 | -90.95 | -84.82 |
| July -> June 1--10 any-touch | -61.64 | -59.16 | -39.06 | **-38.29** |
| July -> June 1--10 clean-first | -104.55 | **-27.80** | -56.80 | -52.10 |

The architecture conclusion is negative but useful:

1. residual context helps the forward May -> June period by about 5.8--10.3
   bps and helps the reverse early-June diagnostic, but it loses 7.1--16.6
   bps versus base-only on June -> July and 1.8--21.7 bps in grouped July;
2. more direct capacity is not the repair: monolithic union is the worst
   current-July arm for both targets;
3. the configured 40-feature overlap is not the cause. Removing it does not
   produce recurrent economic improvement and often worsens the forward
   transfer;
4. short-side July event rank is partly learnable: grouped-July short
   any-touch AUC rises from `0.609` base to `0.615` disjoint residual and
   clean-first from `0.568` to `0.579`, but pooled tail economics deteriorate.
   Better average classification/calibration is therefore not sufficient;
5. July-trained monolithic clean-first transfers back to early June much
   better than it learns July OOF itself. This is another representation/
   relationship non-transfer result, not evidence for a deployable July
   specialist.

No architecture is promotable and no portfolio replay is authorized. Use the
side-specific base event model as the bounded current opportunity baseline;
retain residual and monolithic results as regime/reliability diagnostics.
The next architecture work should add an explicit OOF reliability/trust
adapter or softly weighted recurring-state expert only after a causal state
recurs. Do not spend the next budget on a larger undifferentiated union model
or on removing base/meta overlap.

#### Exact event-score causal global/side mapping completed

The missing calibration infrastructure is now materialized:

- June OOF ledger:
  `scripts/materialize_meaningful_mfe_exact_grid_june_calibration_oof.py`;
- ledger tests:
  `tests/test_materialize_meaningful_mfe_exact_grid_june_calibration_oof.py`;
- ledger artifact:
  `data_perp/artifacts/meaningful_mfe_exact_grid_june_calibration_oof_20260730_v1/`;
- mapping evaluator:
  `scripts/evaluate_meaningful_mfe_exact_grid_causal_mapping.py`;
- mapping tests:
  `tests/test_evaluate_meaningful_mfe_exact_grid_causal_mapping.py`;
- mapping artifact:
  `data_perp/artifacts/meaningful_mfe_exact_grid_causal_mapping_20260730_v1/`.

The ledger contains 33,153 strict June OOF rows over June 10--30 plus the
15,167 unchanged June-trained July-forward rows. It evaluates 15 predeclared
score streams: any-touch, clean-first and soft triple-barrier probabilities
for all three model families, plus touch x capture and clean x capture
compositions. Raw conditional-capture probabilities remain forbidden.

Every OOF row proves:

- training label resolution precedes its validation decision;
- training decision is purged by 12 hours;
- prediction availability is the row's decision time;
- label resolution is decision + 12 hours;
- exact gross - row cost = exact net;
- the frozen winner plus exact-reset runner recipe hash matches the July
  score recipe.

The mapper fits a new global isotonic and side-shrunk-to-global isotonic map
at each decision day using only exact outcomes resolved before that day's
snapshot and within the preceding 21 days. Side shrinkage is
`n_side / (n_side + 500)`. Raw, global and side-to-global arms use identical
mapped-eligible identities and then rank one pooled global book over the whole
July evaluation. No side/day/timestamp/asset quota is used.

Coverage is complete for all scores: 15,167/15,167 July rows and 100% latest-
day coverage. Daily reference support is 31,295--32,921 total rows, with at
least 15,456 long and 15,833 short references. Thus failure is not caused by
map warm-up or sparse side support.

Representative pooled global top-10 results on identical rows:

| Score | Raw | Causal global | Side -> global |
|---|---:|---:|---:|
| LightGBM soft triple barrier | **-92.62** | -127.10 | -117.69 |
| logistic touch x capture | **-97.54** | -111.97 | -170.89 |
| LightGBM touch x capture | **-98.96** | -101.79 | -137.47 |
| CatBoost touch x capture | **-101.10** | -101.66 | -124.74 |
| logistic any-touch | -132.76 | **-122.01** | -126.21 |

Across the predeclared surface, the best descriptive top-10 result is:

- raw: `-92.62` bps;
- causal global: `-101.66` bps;
- side-to-global: `-106.89` bps.

The global map repairs only weak individual streams such as logistic
any-touch, not the competitive tail. Side calibration is generally worse,
which means the June side-to-EV relationship itself does not transfer to
July; it is not merely a raw unit mismatch. At top 1%, side-to-global improves
LightGBM touch x capture from `-60.50` to `-49.21` bps, but remains negative
and worsens CVaR from about `-763` to `-1,099` bps. This narrow result is not
a candidate.

The separate `mapped EV > 0` admission diagnostic is too sparse:

- depending on score/map, only 0--131 of 15,167 rows are admitted;
- top-10% after admission therefore selects 0--14 rows;
- apparent positive examples are often single observations;
- CatBoost touch x capture under the global map admits 77 rows and selects
  eight at +117.80 bps, but CVaR is -388.89 bps and the side-to-global version
  is negative.

These sparse gates cannot support recurrence, portfolio replay or promotion.
They are not a valid claim that the 21-day admission calibrator repairs July.
May -> June remains unavailable for an exactly model-matched causal map
because late-May outcomes selected the frozen HPO recipe; July -> June remains
reverse-time and forbidden from the causal mapping table.

Decision: retain raw scores as diagnostic baselines only. Causal global and
side-to-global mapping do not repair the event architecture, and no positive-
admission arm has adequate support. The next useful work is reliability/
relationship-shift detection or recurring-state conditioning before ranking,
not another monotone score map.

### 2024 frozen-transition exact-1m research extension staged

The older-data extension is now materialized as a separate, explicitly
non-promotable lineage:

- frozen candidate backcast:
  `data_perp/reports/failure_2024_transition_exact1m_candidate_backcast_20260730_v1/`;
- legacy request-stage snapshots retained for byte provenance:
  `data_perp/artifacts/failure_2024_transition_exact1m_stage_20260730_v1/`
  and
  `data_perp/artifacts/failure_2024q1_transition_exact1m_stage_20260730_v1/`;
- strict source-native request stages:
  `data_perp/artifacts/failure_2024_transition_exact1m_request_stage_20260730_v2/`
  and
  `data_perp/artifacts/failure_2024q1_transition_exact1m_request_stage_20260730_v2/`;
- frozen Q1 Kraken product map:
  `data_perp/artifacts/failure_2024q1_kraken_product_map_20260730_v1/`;
- causal Q1 replay inputs (corrected side-qualified geometry):
  `data_perp/artifacts/failure_2024q1_exact1m_label_inputs_20260730_v2/`;
- candidate-level Q1 coverage proof:
  `data_perp/artifacts/failure_2024q1_exact1m_candidate_coverage_20260730_v1/`;
- corrected Q1 policy replay:
  `data_perp/artifacts/failure_2024q1_exact1m_policy_labels_20260730_v2/`;
- serialized Q1 exact paths:
  `data_perp/artifacts/failure_2024q1_exact1m_paths_20260730_v1/`;
- final source-separated Q1 labels:
  `data_perp/artifacts/failure_2024q1_exact1m_multitask_labels_20260730_v1/`;
- staging implementation:
  `scripts/materialize_historical_backcast_exact1m_stage.py`;
- product-lineage implementation:
  `scripts/materialize_kraken_historical_product_map.py`;
- label-input implementation:
  `scripts/materialize_historical_backcast_exact1m_label_inputs.py`;
- candidate-level final coverage gate:
  `scripts/audit_historical_exact1m_candidate_coverage.py`;
- source-separated physical-path/policy-label implementation:
  `scripts/materialize_historical_backcast_exact1m_execution_path_labels.py`.

The 2024 backcast contains 543,816 top-30 rows and 190,398 admitted-monitor
rows across all 366 days. It is a frozen base-only diagnostic backcast with
515 observable features, not OOF. The strict v2 **request** stage freezes
190,398 source-native candidate identities (182,718 unique decision/symbol
pairs) across 141 symbols, shifts each signal by exactly one hour to the
decision timestamp and defines the required half-open 720-minute path. It
does not call a request population an exact-path artifact.

The v2 identity is
`sha256(source-shard hash | source-row number | signal | symbol | side |
archetype | barrier)`. Logical collisions fail closed; no row is silently
deduplicated. The stage manifest separately records effective source shards,
pre/post filters, `candidate_path_map.parquet`, and exact output hashes.

The no-download canonical-store preflight found 43,660,200 required minute
buckets and only 48,780 already present. All 141 symbols therefore require
repair. This is a genuine full-year backfill, not a small gap fill.

Q1 is the first bounded download slice: 36,326 candidate identities, 34,992
unique decision/symbol pairs and 82 symbols. Every symbol is frozen to an
official USD-settled linear `PF_*` Kraken Charts product and boundary-probed
at both ends of its staged history. Inverse `PI_*` products are rejected.
The independently frozen product IDs have zero mismatches against the exact
PF IDs used by the append-only download across all 82 symbols.

The causal label-input artifact already covers all 36,326 rows:

- finite exact signal-time reconstructed Wilder ATR(14): 36,326/36,326;
- uninterrupted prior 90-day hourly-history diagnostic: 34,563/36,326
  (95.147%);
- hourly source parts hash-bound: 167;
- decision/path contract: `[signal+1h, signal+1h+12h)`.

The Q1 1m collection is complete. Two disjoint symbol partitions wrote
immutable append-only parts. The strict frozen-product verification proves
8,592,840/8,592,840 required merged minutes across 82/82 symbols. The
independent candidate-level audit proves exactly 720/720 UTC minutes for
36,326/36,326 candidate identities, with zero missing paths, duplicate
conflicts or product-lineage mismatches. The final coverage manifest hashes
every canonical-store part.

The downstream label pipeline is now explicit:

```text
strict v2 candidate/request identity
+ frozen PF product map
+ causal signal-time ATR
+ exact immutable 720-minute path
-> candidate-local frozen-exit replay
-> exact physical path labels
-> exact-identity joined multi-task labels
```

The direct primary target is `execution_net_ev_12h`. Auxiliary tasks include
opportunity occurrence, favorable payoff, adverse competing risk,
exit-conversion loss, timeout, the five established per-side path heads, the
supporting soft labels, and the ATR-normalized soft triple barrier. Physical
path labels use the unadjusted decision-open OHLC path. Policy economics use
the separately signed current-spread counterfactual replay. The joined
artifact is convenience for multi-task learning, not permission to conflate
the two source contracts.

The corrected Q1 replay resolves 36,297 rows with the exact side-local
archetype geometry and only 29 rows through the signed side-parent fallback.
The earlier
`failure_2024q1_exact1m_label_inputs_20260730_v1` and
`failure_2024q1_exact1m_policy_labels_20260730_v1` artifacts are superseded:
their raw, non-side-qualified archetype key forced all rows through the
side-parent fallback. They must not be used downstream.

Q1 aggregate current-spread-counterfactual economics are:

- mean/median net return: -1.714% / -1.449%;
- positive-return rate: 33.91%;
- exits: 23,681 timeout, 9,764 trailing, 2,869 full stop and 12 adverse exit;
- long opportunity/adverse-first/timeout rates:
  45.39% / 59.09% / 67.80%;
- short opportunity/adverse-first/timeout rates:
  38.20% / 60.46% / 62.58%;
- mean peak MFE: 2.195 ATR long and 1.872 ATR short.

These negative direct economics are a research result, not an endorsement of
the frozen backcast ranker. They reinforce the need to learn execution
utility separately from base alpha and to diagnose opportunity incidence,
conversion and loss magnitude as distinct tasks.

The reusable exact-label recurrence diagnostic is now:

- runner:
  `scripts/diagnose_historical_exact1m_recurrence.py`;
- focused tests:
  `tests/test_diagnose_historical_exact1m_recurrence.py`;
- Q1 report:
  `data_perp/reports/failure_2024q1_exact1m_base_recurrence_20260730_v3/`;
- older 2022--2023 report:
  `data_perp/reports/failure_2022_2023_pf_exact1m_base_recurrence_20260730_v2/`.

It refuses incomplete label bundles, verifies the request-stage/source/label
and 720/720 coverage hashes, binds its own runner hash, and selects one pooled
global top 1/5/10/20% per month with candidate-ID tie-breaking. It never
reranks by timestamp or imposes a side quota. Physical opportunity/MFE targets
remain separate from policy gross/cost/net economics. Transition targets are
diagnostic strata only and are excluded from the pre-entry association screen.
The transition input is repeatable: the older run must supply both the frozen
2022 extension and the schema-identical 2023+ panel. Timestamp overlaps or
schema differences fail closed. The real panels verify at 227/227 identical
columns, zero overlapping decision timestamps and an exact one-hour boundary
from 2023-01-01 00:00 to 01:00 UTC. The earlier Q1 v1 report is superseded
because it did not bind the runner bytes. Q1 v2 and older v1 are also
superseded: the runner previously hard-coded the text `2022 unobserved` even
when the frozen 2022 panel was supplied and joined completely. The corrected
runner derives the interpretation from measured per-year coverage. Q1 v3 and
older v2 bind the corrected runner hash; older v2 records 100% transition
context coverage in 2022, 2023 and the 16 horizon-boundary decisions on
2024-01-01.

The Q1 global base-score top-10 net economics are -227.81, -135.34 and
-152.62 bps for January, February and March. Transition-active selected rows
are only 23, 20 and 28 respectively. Their active/inactive net comparisons
are -225.74/-227.85, -248.94/-133.26 and -161.87/-152.42 bps. Every active
cell fails the frozen 50-row support gate. This is consistent with the
governing result: active transition is not a universal veto, while the
February interaction is a hypothesis requiring recurrence rather than a
routing rule.

Interpretation is intentionally narrow:

- evidence scope is `frozen_backcast_diagnostic_not_oof`;
- `execution_parity_claim=false` and `promotion_eligible=false`;
- exact means the one-minute OHLC path envelope, not tick/order-book replay;
- same-minute favourable/adverse barrier conflicts require conservative
  adverse-first treatment;
- historical bid/ask, depth and queue information are unavailable, so
  economics are frozen/current-spread counterfactual only;
- pre-2025 candidates do not carry the bit-exact deployed-policy
  `__path_auxiliary_atr_fraction__` geometry. They can support transition
  mechanism and label-learnability research, but not deployed-policy
  comparison or the prospective 60--100 incident gate.

The 2022--2023 extension now has a completed base-only candidate backcast:

- backcast:
  `data_perp/reports/failure_2022_2023_pf_baseonly_backcast_20260730_v1/`;
- strict PF request stage:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_request_stage_20260730_v1/`;
- frozen product map:
  `data_perp/artifacts/failure_2022_2023_pf_kraken_product_map_20260730_v1/`;
- corrected causal label inputs:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2/`.
- frozen August--December 2022 transition-market extension:
  `data_perp/artifacts/regime_transition_research_2022augdec_frozen_v1/`.
- candidate-level 720/720 coverage proof:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_candidate_coverage_20260730_v1/`;
- current frozen-exit counterfactual policy labels:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_policy_labels_20260730_v1/`;
- timing candidate handoff:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_timing_candidates_20260730_v1/`;
- serialized exact paths:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_paths_20260730_v1/`;
- final source-separated direct/auxiliary labels:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1/`;
- exact base-score/economics recurrence report:
  `data_perp/reports/failure_2022_2023_pf_exact1m_base_recurrence_20260730_v2/`.

The backcast contains 310,088 rows and 118,734 selected-monitor candidates
over 17 months and 489 covered days. Observable signals begin on 2022-08-30;
the unsupported partial July 2022 run is excluded because it has no
resolvable frozen policy archetype. The strict stage contains 118,734
candidate identities, 112,746 unique decision/symbol paths and 74 frozen
official PF products. All 118,734 candidates have a finite causal signal-time
ATR; 111,086 (93.56%) also have an uninterrupted prior 90-day hourly-history
diagnostic.

The older exact-1m download and all downstream labels are complete. Two
disjoint product-bound workers completed 74/74 products with zero failures.
The independent strict verify-only pass fetched no additional rows and
proved:

- 25,353,060/25,353,060 required merged minutes;
- 118,734/118,734 candidate identities with exactly 720/720 path minutes;
- zero incomplete candidates, missing timestamps, duplicate conflicts or
  immutable duplicate rows;
- exactly 59,367 long and 59,367 short candidate identities;
- one-to-one candidate-ID joins through request stage, candidate coverage,
  causal label inputs, policy replay, timing handoff, exact paths, physical
  labels and joined multi-task labels;
- zero signal, decision, symbol or side mismatches across those joins;
- 11,409/11,409 manifest-declared file hashes independently verified.

Core immutable output hashes are:

- candidate coverage parquet:
  `7d47c4a0c31973a52a14488dadf1b789680ec67c2a1feefee608c20cedf49023`;
- policy labels:
  `d26e16e7acf176820f950d9ca25fd0e3968cecf9dd6819019623b9a8499c04dd`;
- exact paths:
  `25a341eabbefa7ca940c3fd7ec2d2f95730849c3d4f8b97d88e3e57cb885ed3e`;
- physical path labels:
  `5ed29dbdb889343d69b5a25c81e3cabd9e2a9321ae05ed0675cabbdc1b832f42`;
- joined multi-task labels:
  `8a004e2bd7cacbb89854e1535fec17157ba62c7e123cc422fb0119e93a29fdb8`.

The path artifact's legacy explicitly configured file
`missing_paths.csv` is JSON despite its extension. It is not a table of 558
missing paths: its JSON payload reports 118,734 complete rows, zero incomplete
rows and full by-month/by-symbol coverage. Future runs should retain the
materializer's default `.missing.json` name.

The frozen PF/base-top-30 candidate population begins at 2022-08-30 10:00
UTC because no earlier frozen, policy-resolvable shards exist. That statement
applies only to that population lineage. It does **not** mean January--July
Kraken history is unavailable. A separate inverse-PI paired market-grid
population has now been materialized for January--July 2022, as documented
below. It must not be pooled silently with the later USD-linear PF population.
The final 16 2023-12-31 23:00 PF signals correctly decide at 2024-01-01 00:00
and use paths through 12:00 under the fixed +60-minute
decision/+720-minute path contract.

### January--July 2022 separate inverse-PI candidate lineage

The approved earlier-period exception is implemented as a new, versioned
candidate population rather than a relabelled copy of the later frozen
population. The authoritative chain is:

- fixed paired acquisition population:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_grid_source_20260730_v1/`;
- acquisition request stage and exact PI product map:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_exact1m_stage_20260730_v1/`
  and
  `data_perp/artifacts/jan_jul_2022_inverse_pi_product_map_20260730_v1/`;
- exact-minute acquisition/verification manifests:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_download_20260730_v1/`;
- final causal feature population:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_features_20260730_v3/`;
- final feature-preserving exact request stage and product map:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_exact1m_stage_20260730_v1/`
  and
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_product_map_20260730_v1/`;
- final no-download strict verification and independent candidate coverage:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_download_verify_20260730_v1/`
  and
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_candidate_coverage_20260730_v1/`;
- causal ATR and explicit side-parent label inputs:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_label_inputs_20260730_v1/`;
- candidate-local deployed-exit counterfactual labels:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_policy_labels_20260730_v1/`;
- timing candidates and serialized exact paths:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_timing_candidates_20260730_v1/`
  and
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_paths_20260730_v1/`;
- authoritative source-separated physical/direct/auxiliary labels:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_causal_multitask_labels_20260730_v2/`.

The population contains 50,880 candidates: every hour from 2022-01-01
through 2022-07-31, five continuously available Kraken inverse perpetuals,
and paired long/short rows. The exact product bindings are `PI_XBTUSD`,
`PI_ETHUSD`, `PI_LTCUSD`, `PI_XRPUSD` and `PI_BCHUSD`. All five were
boundary-probed before the request was accepted. No future-informed
candidate screen is applied.

The acquisition downloaded and re-verified 349,260 exact minutes per product,
including the 30-day pre-January causal feature warm-up. The final strict
path-only pass proves 1,529,700/1,529,700 merged minutes, fetched no
additional rows, and the independent audit proves 50,880/50,880 candidate
identities with exactly 720/720 UTC path minutes. There are zero incomplete
candidates, missing timestamps, conflicting duplicates or product-binding
mismatches. All 50,880 rows also have an uninterrupted 30-day hourly history
and finite signal-time Wilder ATR(14).

Hourly features are right-labelled: the row stamped `t` uses only
`[t-1h,t)`, the execution decision is `t+1h`, and the label path is
`[t+1h,t+13h)`. The final stage preserves 69 causal pre-entry features:
44 asset/market fields plus 25 transition-dynamic fields covering volatility,
dispersion, breadth, cross-asset correlation and BTC-versus-alt rotation.
This is a transition-research feature set for the separate market grid; it is
not represented as the already-frozen 227-column later detector.

Because this grid has no causal archetype classifier, policy geometry is
bound explicitly to `long__parent` and `short__parent`. All 50,880 policy
labels resolve through those deployed side-parent geometries. The earlier
acquisition-only 2% barrier is never used as an economic target. Final
barriers are `clip(1.5 * causal ATR14 fraction, 0.5%, 5%)`.

The economic contract remains deliberately narrow:

- returns are quote-notional price returns, not inverse-collateral ROE;
- historical L2 spreads, depth and queue state are unavailable;
- current spread/fee accounting is a counterfactual, with the spread baseline
  falling back to its frozen cross-sectional estimate for these inverse
  symbols;
- the population is non-OOF, non-promotable and cannot count toward the
  prospective incident gate;
- it must remain separate from the USD-linear PF/base-top-30 lineage in
  training reports, calibration and promotion decisions.

The direct side-parent replay has mean net return -109.21 bps, mean gross
return -9.26 bps and approximately 99.95 bps mean cost. The simple
market-grid momentum proxy is diagnostic only, not the production base
alpha. Its pooled global monthly top-10 net return remains negative in every
month: -105.90, -121.07, -85.66, -121.19, -74.35, -121.41 and -78.69 bps
from January through July. Gross top-10 is positive in March (+14.41 bps),
May (+25.78 bps) and July (+21.42 bps), but the current-cost counterfactual
eliminates that edge. This makes the new lineage useful for cost-conversion,
opportunity-state and transition-mechanism research, not as evidence that
the grid itself is tradable.

Monthly opportunity incidence spans 40.43%--49.14%, adverse-first incidence
55.78%--58.70%, and timeout incidence 51.33%--59.21%. The direction that is
easier varies materially by month: for example April long mean net is -144
bps versus -83 bps short, while July long is -71 bps versus -135 bps short.
This is exactly the kind of regime/side interaction that the direct utility
architecture should test without turning transition state into a universal
veto.

The initial causal feature artifacts
`jan_jul_2022_inverse_pi_causal_features_20260730_v1` and `_v2` are
superseded because their shard provenance did not preserve the final causal
population/product binding. The initial multi-task label artifact
`jan_jul_2022_inverse_pi_causal_multitask_labels_20260730_v1` is also
superseded because its manifest incorrectly hard-coded the later frozen
lineage; its label bytes match v2, but only v2 has the correct signed
provenance. The materializers now fail closed on these mismatches. The
integrated lineage/product/feature/coverage/timing/path suite passes 41
focused tests.

The current frozen-exit replay resolves 118,664 rows (99.941%) with exact
side/archetype geometry and only 70 through the signed side-parent fallback.
The primary target remains direct `execution_net_ev_12h`; all opportunity,
payoff, adverse-risk, conversion, timeout, soft-triple-barrier and established
path targets remain auxiliaries and are source separated from policy
economics.

The older recurrence result is a strong negative control for the frozen base
ranker:

- pooled global monthly top-10 net EV is negative for every adequately
  supported month; the only positive month is the partial August 2022 cell
  at +44.31 bps on only eight selected rows;
- September--December 2022 top-10 net EV is -131.16, -145.72, -145.19 and
  -170.03 bps;
- January--December 2023 ranges from -127.07 to -223.87 bps, with no positive
  month;
- cost is stable near 100 bps, so the time variation is not a cost-drift
  explanation; top-10 gross EV itself is negative in every adequately
  supported month;
- conditional favorable payoff remains sizeable, but opportunity prevalence,
  adverse competing risk, timeout and exit-conversion loss jointly prevent
  profitable conversion.

This also resolves the apparently paradoxical result that base-target IC can
rise while direct execution EV falls. Within side, the base score retains
weak relative ordering: median monthly rank IC is 0.035/0.049 for opportunity,
0.042/0.040 for peak MFE and 0.010/0.024 for policy net on long/short. In
2023 the highest score decile beats the lowest by a median 0.090 opportunity
rate, 0.643 ATR peak MFE and 27.5 bps net return. But this is a relative
improvement from an unprofitable absolute surface, not positive direct
utility. Moreover, pooled global selection becomes almost entirely long from
March 2023 onward (for example 1,185/1,191 selected rows in July and
1,190/1,191 in December), exposing a frozen cross-side score-scale drift.
This is precisely why the direct execution-utility layer needs causal
cross-side/global calibration and why base IC cannot be treated as final
trading EV.

The transition interaction remains heterogeneous rather than a veto:

- June 2023 active-transition support reaches 103 selected rows and is less
  negative than inactive (-93.54 versus -193.55 bps);
- July reaches exactly 50 active rows and is more negative than inactive
  (-261.37 versus -174.85 bps);
- every other active monthly cell has fewer than 50 rows.

Active transition must therefore remain an interaction context for
opportunity, health, conversion and liquidity, not a direct exposure rule.
The source and transition association screens are univariate research
diagnostics only; they are not fitted models, feature-selection evidence or
promotion claims.

These rows remain source-separated historical research only: historical L2,
spread and bit-exact pre-2025 deployed geometry cannot be reconstructed. The
bundle is explicitly non-OOF, non-promotable and does not claim execution
parity.

The frozen transition research contract has also been extended backward from
its prior January-2023 start through 2022-08-30:

- runner:
  `scripts/materialize_frozen_regime_transition_extension.py`;
- output: 2,976 uninterrupted hourly rows through 2022-12-31;
- schema: exactly the same 227 columns and order as frozen v3;
- geometry: the exact existing v3 state transform and hash, with no refit,
  feature selection or new imputation rule;
- support: six transition events, 129 labelled event-window rows and 54
  validated event snapshots;
- timing: one-hour source-to-decision shift, 24-hour causal input buffer,
  12-hour forward target buffer and anchor-plus-13-hour target availability.

Three frozen geometry inputs have zero raw coverage in this interval:
`btc_ex_eth_oi_dominance_z_ratio`, `btc_oi_dominance_z_ratio` and
`btc_over_eth_dominance_roc`. The artifact explicitly records this and reuses
the already-fitted v3 `SimpleImputer`; it does not synthesize values or fit a
new fill rule. This makes the extension valid for frozen-geometry transition
research, but the three-field limitation must be an explicit sensitivity
when economic outcomes are joined. No 2022 economics overlay was
manufactured; the exact-policy label path above is the required source.

The combined recurrence/transition/exact-lineage suite passes 45/45 focused
tests and all new output hashes verify independently.

The prospective gate is unchanged: older downloaded data can enlarge
source-separated research and falsify transfer hypotheses, but it cannot be
retroactively counted as an incumbent-policy incident. Current prospective
portfolio-grade count remains zero.

## 2026-07-30 exact competing-risk simplex and soft-label result

The row-cost-aware three-class architecture is now implemented and evaluated
on the exact 134,889-row current-lineage panel:

```text
P(timeout, adverse-first, clean-economic-favourable-first)
× side/class-conditional deployed-gross payoff
- row-specific execution cost exactly once
```

It is a standalone PIT-feature diagnostic. It does **not** implement or
replace the final config-routed `base -> residual/context -> execution EV`
architecture. The frozen `base_oof_score` is persisted only for the
identical-row IC-to-EV bridge and is excluded from model features.

### Infrastructure and audit repairs

The primary implementation is:

- runner:
  `scripts/run_execution_ev_competing_risk_simplex_ablation.py`;
- tests:
  `tests/test_run_execution_ev_competing_risk_simplex_ablation.py`;
- primary 100-bps artifact:
  `data_perp/artifacts/execution_ev_competing_risk_simplex_primary100_20260730_v1/`.

An independent pre-run audit caught and repaired three material issues:

1. Inverse-prevalence class weights were removed. Weighted class probabilities
   cannot be interpreted as natural `P(class | x)` in an expected-value sum,
   and scalar temperature cannot undo three different prior shifts.
2. Calibration now reports raw, scalar-temperature and regularized
   offset-plus-temperature arms. The latter fits one common temperature plus
   two anchored class offsets on chronological, 12-hour-purged train-only
   calibration rows. Soft targets use fractional cross-entropy. Inadequate
   support falls back explicitly to raw.
3. Geometry selection now searches the complete bounded composition:
   two classifier candidates times two independently selected candidates for
   each of the three conditional gross heads, or 16 combinations per
   family/side. Each head has train-only feature selection, winsorization and
   task-appropriate payoff parameters. The direct-net control has its own
   independent feature and geometry selection.

Grouped-July predictions retain their fold identity, and economics are
published both for the combined grouped-OOF population and each two-day
holdout. The runner also persists the frozen-base MFE/gross/net/event IC
bridge, global top 1/5/10/20%, positive precision, opportunity/clean recall,
CVaR, score compression, cutoff ties and side composition.

The focused primary suite passes 14/14. The primary report hash, runner hash,
all eight output hashes and every upstream feature/label/materializer hash
verify.

### Primary 100-bps result

The primary label uses
`max(1.5 ATR, 1.5%, row cost + 100 bps)` as the clean favourable barrier,
1 ATR adverse, full decision-plus-720-minute paths and adverse-first
same-minute ties.

Pooled-global top-10 exact net, bps/trade:

| Evaluation | Best composed | Best direct-net | Frozen base | Revealed-class oracle |
|---|---:|---:|---:|---:|
| May -> June | **-60.97** | -78.12 | -99.75 | **+62.84** |
| June -> July | -123.73 | **-113.14** | -143.89 | **+24.01** |
| Grouped July OOF | -95.25 | **-93.82** | -143.89 | **+60.75** |
| July -> June matched, reverse diagnostic | **-36.36** | -94.07 | -124.62 | **+93.51** |
| July -> June full, reverse diagnostic | **-99.27** | -104.79 | -99.75 | **+62.84** |

No predictive arm is recurrently positive. The only positive predictive
cells are isolated 1% reverse-time results; they are non-promotable and
disappear by 5%. Every source-forward and grouped-July top-5/10/20 result is
negative.

The same grouped-July logistic direct-net arm is negative in all five
holdouts at top 10%:

`-64.33, -41.56, -58.59, -105.67, -64.56 bps`.

The best aggregate composed arm, logistic offset-temperature, is also
negative in every fold:

`-43.67, -22.74, -70.69, -199.23, -77.24 bps`.

Calibration is useful but insufficient. At top 10%, logistic
offset-temperature improves May -> June from -74.82 raw to -60.97 and
grouped July from -110.95 to -95.25. Tree-family calibration can instead
degrade the tail. Calibration is therefore retained as an ablation dimension,
not promoted as a mapping repair.

The grouped-July classifier has modest learnability:

- long best-NLL arm, logistic offset-temperature:
  NLL/RPS `0.923/0.154`; clean AUC/AP `0.589/0.379`, adverse `0.583/0.641`,
  timeout `0.541/0.127`;
- short best-NLL arm, raw CatBoost:
  NLL/RPS `0.802/0.127`; clean AUC/AP `0.619/0.306`, adverse `0.611/0.777`,
  timeout `0.553/0.108`.

Conditional clean-payoff magnitude is the strongest supporting regression:
May -> June IC is `0.280--0.285` long and `0.408--0.439` short; June -> July
is `0.229--0.285` long and `0.233--0.289` short. Adverse magnitude is weak
and unstable, especially June -> July short (`-0.080--0.027`). Timeout
magnitude is modestly learnable. The July short clean-class net payoff also
falls from approximately +113 bps in June to +44 bps in July, so correct
class assignment alone faces a material payoff-transfer shift.

The true-class oracle is deliberately nonpredictive: it reveals the
evaluation class, applies only train-side/class gross means, and subtracts
row cost once. Its positive top-10 results prove that the primary taxonomy has
an economic ceiling at the intended width. Its negative top-1 cells show that
the taxonomy does not rank within class. The remaining bottlenecks are
predictive class assignment, conditional-payoff transfer and within-class
tail ranking.

The IC-to-EV bridge also makes the failure explicit:

- grouped-July logistic direct-net has MFE IC `+0.118` and exact-net IC
  `+0.071`, but selects only +6.22 bps gross against 100.03 bps cost, hence
  -93.82 bps net;
- May -> June best composed reaches +39.23 bps gross against 100.20 bps cost;
- June -> July best predictive gross is already -13.20 bps;
- the matched reverse best reaches +63.96 bps gross but still loses
  -36.36 bps after cost.

This is the same general mechanism seen in the older recurrence ledger:
positive relative ordering or event learnability can coexist with an
unprofitable response level at the configured global tail.

#### Mandatory investigation: why improving base IC does not convert to EV

Do not close this issue with the statement that the base learns alpha rather
than final execution ranking. That architecture explains why the two metrics
need not coincide, but it does not explain the observed month-to-month
movement. In particular, long native-target rank IC rises from `0.155` in
February to `0.162` in March and `0.226` in April while the base-score-ranked
exact-policy top-decile net is approximately `-59`, `-91` and `-38` bps. The
February-to-March deterioration despite improving IC is a required
falsification case. Also preserve the terminology: these three economic
figures are exact execution outcomes ranked by the frozen **base** score, not
the performance of a separately trained direct execution-EV head. Any direct
head must be reported as a separate score stream on identical rows.

Add this investigation to the active base -> residual/context workstream and
run it before interpreting another aggregate-IC improvement as progress:

1. On identical candidate identities, by month and side, compare the frozen
   base, residual and direct-EV scores against the native 24-hour target,
   matched 12-hour target, exact MFE, deployed gross and exact net. Report
   full-sample and globally selected top-1/5/10/20% IC, economics, positive-net
   precision, loss rate and CVaR; never substitute per-timestamp top-k.
2. Split rank quality from payoff level. For fixed score ventiles/deciles,
   report response slopes and intercepts, meaningful-MFE incidence,
   conditional MFE, MFE-to-realized-gross capture, adverse/timeout incidence,
   exit reason, row cost and net payoff. Test explicitly whether IC improves
   mostly outside the traded global tail or while the tail response intercept
   falls.
3. Run February/March and March/April fixed-composition reweighting and
   rank-cell swaps, with day-block bootstrap intervals. Attribute each
   top-decile EV delta to candidate composition, within-cell opportunity
   incidence, conditional favorable magnitude, exit capture, adverse payoff,
   timeout payoff and cost. The approximately 100-bp cost level is stable in
   the existing audit, so gross/capture changes must not be hidden by a generic
   “cost-aware” explanation.
4. Test the horizon/target hypothesis with a matched 12-hour base-label model
   using the same rows, features, folds, feature selection and side-local HPO
   as the 24-hour model. Then cross-score 24-hour and 12-hour predictions on
   both targets and on the same frozen exact exit policy. This distinguishes a
   horizon mismatch from a regime or execution-capture failure.
5. Measure tail membership stability and cutoff geometry: overlap and recall
   versus the exact-MFE, gross and net oracle tails; score compression and
   cutoff ties; asset, liquidity, spread, transition/regime and exit-family
   composition. Quantify whether a small rank improvement is simply
   reordering economically interchangeable rows.
6. Perform the monetisation counterfactuals without changing model ranking:
   replay the already frozen exit-policy alternatives on the same selected
   identities and calculate gross-capture efficiency (`realized gross / MFE`)
   by month, side and regime. This is diagnostic only; it must not select a new
   exit policy on evaluation outcomes.
7. Promote any explanation only if its decision-time feature or supporting
   label improves recurrent OOF/global-tail economics in the subsequent
   execution layer. Candidate outputs include a 12-hour opportunity score,
   cost-hurdle probability, capture/reliability score and regime-conditioned
   trust feature. Ex-post attribution alone is not a routing rule.

The deliverable is a quantitative waterfall explaining the February -> March
and March -> April IC/EV deltas, plus the same bridge for the current
May/June/July lineage. Until that exists, improving base-target IC is evidence
of better relative alpha ordering only, not evidence that the architecture is
converting it into executable value.

### No-floor 50-bps and soft-label challenger

The first no-floor run failed closed before publishing outputs because the
arbitrary v1 250-row conditional-payoff minimum exceeded the real fold-4
timeout support. The empty directory is retained as:

`data_perp/artifacts/execution_ev_competing_risk_simplex_nofloor50_20260730_v1_FAILED_INSUFFICIENT_TIMEOUT_SUPPORT/`.

The completed challenger uses a separately versioned runner so the
hash-bound primary runner remains immutable:

- runner:
  `scripts/run_execution_ev_competing_risk_simplex_ablation_v2.py`;
- additional test:
  `tests/test_run_execution_ev_competing_risk_simplex_ablation_v2.py`;
- artifact:
  `data_perp/artifacts/execution_ev_competing_risk_simplex_nofloor50_20260730_v2/`.

V2 predeclares a 200-row minimum after auditing all grouped folds; the actual
minimum is 212 rows. The combined focused suite passes 15/15, and the primary
v1 runner still matches its published hash. The no-floor report, runner and
all output hashes verify.

The no-floor label uses `max(1.5 ATR, row cost + 50 bps)` without the fixed
1.5% return floor. Its soft logistic target keeps observed first touches
one-hot and replaces only genuine timeouts with their terminal
clean/adverse/timeout viability simplex.

Pooled-global top-10 exact net:

| Evaluation | Best hard composition | Best soft composition | Direct-net control | Revealed-class oracle |
|---|---:|---:|---:|---:|
| May -> June | **-60.88** | -61.84 | -78.12 | +34.39 |
| June -> July | **-118.79** | -146.64 | -113.14 | **-1.47** |
| Grouped July OOF | **-109.71** | -111.48 | -93.82 | +43.74 |
| July -> June matched | **-41.42** | -42.20 | -94.07 | +71.91 |
| July -> June full | **-107.33** | -111.63 | -104.79 | +34.39 |

Soft supervision slightly improves some proper scores but does not improve
execution ranking. In grouped July, long soft offset-temperature NLL is
0.897 versus 0.906 for hard logistic, and short is 0.820 versus 0.840; top-10
economics nevertheless worsen from -109.71 hard to -111.48 soft. The soft
arm is negative in every grouped fold, including -241.50 bps in fold 3.

Removing the fixed floor is also non-incremental. Relative to primary 100,
the no-floor label is essentially tied in May, worse in forward July and
worse in grouped-July. Its revealed-class oracle falls from +24.01 to
-1.47 bps in June -> July, showing that the alternative taxonomy itself has
less economic separation there.

**Decision:** reject the no-floor 50-bps geometry and timeout-soft simplex as
execution-EV admissions. Do not run no-floor 100 because it is exactly
equivalent to primary 100. Do not run primary 25/50 because they are inert or
near-inert under the fixed 1.5% floor. Neither completed predictive
architecture passes the raw global-tail gate, so there is no causal mapping,
action-layer, simple-policy or portfolio-constraint replay.

### Config-routed context screen and transition-sidecar materialisation (2026-07-30)

The exact Primary100 candidate-context sidecar is now materialised at
`data_perp/artifacts/primary100_exact_context_sidecar_20260730_v1/`. It binds
all 134,889 frozen candidate identities to outcome-free base-score geometry,
candidate-relative fields, DAE geometry, permitted GMM distances and raw
transition entropies. Representation geometry is available on 131,011 rows;
the remaining 3,878 short rows stay explicitly unavailable rather than being
filled.

The config-routed add/drop runner is
`scripts/run_execution_ev_competing_risk_context_add_drop.py`; its bounded
20-iteration diagnostic is
`data_perp/artifacts/execution_ev_competing_risk_context_add_drop_screen20_20260730_v1/`.
It trains the event/payoff channels only inside each outer training sample,
passes only their OOF predictions to the side-local direct-net context head,
uses the configured side-local base and meta feature pools, and evaluates a
single deterministic pooled-global top 1/5/10/20%. Timing, MAE, wait and
target-price fields are rejected. GMM posteriors and compact risk summaries
remain excluded.

Top-10 exact net, bps/trade, for the main controls and add/drop blocks:

| Arm | May -> June | June -> July | Grouped July OOF | July -> June matched |
|---|---:|---:|---:|---:|
| Frozen base | -99.75 | -143.89 | -143.89 | -124.62 |
| Direct meta, no alpha | -52.12 | -129.28 | -122.80 | -44.18 |
| Direct meta + alpha | -95.71 | -128.35 | -126.96 | -12.88 |
| Clean probability | -92.15 | -124.87 | **-117.69** | -9.74 |
| Competing risk | **-40.70** | -125.01 | -123.75 | **+19.95** |
| Clean value + rank | -68.05 | -138.36 | -134.27 | -1.22 |
| Candidate context, joint | -57.70 | -129.21 | -131.31 | -117.20 |
| DAE geometry | -68.86 | -133.18 | -124.37 | -0.79 |
| GMM distance/Mahalanobis geometry | -100.52 | -133.33 | -131.66 | -17.67 |
| Raw transition entropy | -70.71 | -129.84 | -134.79 | -3.10 |

No arm is recurrently positive. The apparent reverse-time competing-risk win
is non-promotable and is accompanied by a 0% long share at top 10%. In
grouped July the best aggregate arm, clean probability, is still -117.69 bps
and no single arm wins all five folds. Several heads collapse to one side in
individual folds. Therefore do **not** spend the next run on 260-iteration
confirmation or HPO of these exact blocks. They have failed the bounded
screen's economics/coverage gate.

Two strict transition sidecars are also materialised:

- current Primary100:
  `data_perp/artifacts/primary100_current_transition_feature_sidecar_20260730_v1/`;
  134,889 rows and 311 manifest-whitelisted decision-time fields, with exact
  decision-time-to-anchor matches on 111,712 rows and 23,177 unavailable rows;
- historical exact-label bundle:
  `data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_sidecar_20260730_v1/`;
  118,734 rows and 212 decision-time fields, but transition coverage on only
  5,932 rows (full August--December 2022 coverage, effectively no 2023
  continuation).

Both sidecars fail closed on future/target/outcome namespaces, use exact
timestamp joins, and perform no as-of fill. The combined focused suite passes
22/22. The historical result identifies a material data task rather than a
modelling result: reconstruct the same transition feature contract through
2023 before using the older labels to judge cross-era transition
classification.

### Pooled transition lifecycle screen and historical-readiness audit

The next lifecycle diagnostic is materialized at
`data_perp/artifacts/pooled_transition_lifecycle_diagnostic_20260730_v1/`,
implemented by `scripts/run_pooled_transition_lifecycle_diagnostic.py`.
It is a non-walk-forward, research-only screen over the existing immutable
source-separated H12 global-book panel.  It uses shuffled grouped seven-day
CV, a two-sided 36-hour embargo, fold-local preprocessing, and nested
grouped/embargoed Brier shrinkage.  Source family is never a model feature:
it is retained as a domain flag for reporting and inverse-source-frequency
weights balance training/calibration mass.  The current source is restricted
to strict mapped-OOF rows.

The screen retains the existing exact `onset within 3h` target and adds:

- **recovery within 3h**, conditional on an active adverse state now and three
  exact subsequent inactive active-state anchors; and
- **reversal after recovery within 3h**, requiring three active anchors, a
  fully observed four-anchor inactive recovery including the present anchor,
  then a fresh active state within the next three anchors.

Each derived label's availability is the maximum availability of every active
state dependency.  Thus recovery has no outcome information not resolved by
the same declared H12 before/after lineage plus its exact three-anchor state
extension.

Results reject a lifecycle head at this stage:

| Target | Source-balanced rows / positives | Best pooled ROC-AUC / AP | Top-decile lift | Calibration finding |
|---|---:|---:|---:|---|
| Onset within 3h | 6,058 / 400 | 0.463 / 0.061 | 0.83x | ExtraTrees shrinks fully to prior; logistic retains at most 0.05 |
| Conditional recovery within 3h | 1,377 / 143 | 0.477 / 0.101 | 0.76x | ExtraTrees shrinks fully to prior; logistic retains at most 0.05 |
| Reversal after recovery within 3h | 6,054 / 6 | not fit | not fit | Fail closed: insufficient grouped binary support |

The new reversal definition has only 0 canonical, 2 current strict-OOF and 4
reconstructed positives.  It is correctly recorded as two skipped arms rather
than a fitted point estimate.  The pooled global top-decile alert book also
selects no current-source rows for either ExtraTrees lifecycle score, an
additional domain-coverage warning.  No onset, recovery or reversal output is
an admission veto, trust router, timing/wait input, portfolio control or
production feature.

The exact historical readiness audit is materialized at
`data_perp/artifacts/pooled_transition_classification_readiness_20260730_v2/`,
implemented by `scripts/materialize_pooled_transition_classification_readiness.py`.
It verifies the sidecar and label hashes, records monthly/side coverage and
fails closed with status
`BLOCKED_MISSING_2023_TRANSITION_FEATURE_AND_COMMON_GLOBAL_BOOK_LABELS`.
It identifies four required derivations before a valid historical/current
pooled classifier exists:

1. exact decision-time transition context through all 2023 candidate hours
   (not a forward fill of the 2022 sidecar);
2. a versioned semantic common feature mapping—the current v4's 311 fields
   and the historical frozen sidecar's 212 fields have zero exact names in
   common;
3. the historical raw score, causal 21-day map, map support and pooled global
   top-10 membership at each anchor; and
4. immutable causal global-book `before [s-H,s)` / `after [s,s+H)` label
   aggregates with exact availability times, from which active/onset/recovery/
   reversal can be derived.

The forthcoming historical transition reconstruction can be checked without
changing this conclusion by running the readiness audit with its sidecar root:

```text
python3 scripts/materialize_pooled_transition_classification_readiness.py \
  --older-context <reconstructed-sidecar-root> \
  --older-labels <historical-label-root> \
  --current-panel <current-v4-panel-root> \
  --output-dir <new-immutable-readiness-artifact>
```

It must not be described as pooled classifier support until all four
requirements are actually `ready=true`; a richer 2023 sidecar alone repairs
only the first item.

### Strict common-geometry historical/current transition classifier

The preceding v2 readiness blocker is now superseded by
`data_perp/artifacts/pooled_transition_classification_readiness_20260730_v7/`,
which records `READY_FOR_POOLED_TRANSITION_CLASSIFICATION` and
`all_requirements_ready=true`.  The completed prerequisites are the 2023
context continuation, historical causal mapping, canonical global-book
transition labels and the strict semantic common geometry.

The new immutable panel is
`data_perp/artifacts/pooled_historical_current_transition_panel_20260730_v1/`,
implemented by
`scripts/materialize_pooled_historical_current_transition_panel.py`.  It has
17,320 exact H12 anchors:

| Source | Rows | Provenance |
|---|---:|---|
| Historical 2022--2023 backcast | 11,186 | non-OOF, frozen-spread counterfactual, diagnostic only |
| Reconstructed Jan--Apr 2025 | 2,498 | strict OOF, fee-only source-separated |
| Canonical Feb--Apr 2025 | 2,090 | strict OOF, spread aware |
| Current May--Jul 2026 | 1,546 | strict-OOF and frozen-forward provenance retained separately |

Every row uses exactly the 90 features formed from nine semantically common
raw fields x side median/IQR state summaries x state mean/long-short gap and
exact 1h/3h/12h past deltas.  The signal context is anchor minus one hour.
There is no as-of join, resample, interpolation or fill.  All active, onset,
conditional recovery and reversal labels are rebuilt uniformly after pooling
from one pooled-global top-10 H12 before/after labels, and every derived target
retains the exact maximum availability of its dependencies.

The bounded classifier artifact is
`data_perp/artifacts/pooled_historical_current_transition_classifier_20260730_v1/`,
implemented by
`scripts/run_pooled_historical_current_transition_classifier.py`.  Current
frozen-forward rows are excluded from fitting.  Grouped OOF uses shuffled
seven-day groups, a two-sided 36-hour purge, fold-local preprocessing,
source-balanced sample weights and nested grouped Brier shrinkage.  Source,
domain, calendar and provenance fields are weighting/reporting metadata only.

Pooled/common-geometry conclusions:

| Target/model | Pooled ROC-AUC / AP | Current strict-OOF ROC-AUC / AP | Decision |
|---|---:|---:|---|
| Active adverse / logistic | 0.505 / 0.223 | 0.503 / 0.220 | no useful recurrent ranking |
| Active adverse / ExtraTrees | 0.477 / 0.213 | 0.524 / 0.238 | weak current-only hint, not pooled transfer |
| Onset within 3h / logistic | 0.491 / 0.066 | 0.521 / 0.066 | AP is essentially prevalence |
| Conditional recovery / logistic | 0.475 / 0.106 | 0.471 / 0.101 | not learnable |
| Reversal after recovery | 21 positives pooled; 2 current | not reliable | too sparse; no routing use |

The source-transfer matrix is also non-promotable.  Historical 2022--2023 to
current active risk has ROC-AUC 0.461 and tie-aware top-decile lift 0.994;
historical to current onset has ROC-AUC 0.506 and lift 0.722.  Thus the older
backcast does not teach the current regime.  Reconstructed-2025 to current is
the only modest hint: active ROC-AUC 0.541/AP 0.238 and onset ROC-AUC
0.535/AP 0.071.  Canonical-2025 to current active is neutral at ROC-AUC 0.505.
Reverse current-to-older active transfers shrink completely to the prior.
Strong reconstructed/canonical transfers occur over overlapping 2025 market
times and are within-era semantic consistency, not evidence of cross-regime
generalization.

#### Mandatory tie-aware correction

The frozen classifier's timestamp tie-break is deterministic, but a constant
or cutoff-tied probability does not define an economic ranking.  The immutable
no-refit audit is
`data_perp/artifacts/pooled_historical_current_transition_score_tie_audit_20260730_v1/`,
implemented by
`scripts/audit_pooled_historical_current_transition_score_ties.py`.  It
reports score dispersion, zero-shrink share, cutoff plateau size, and expected
top-decile precision with exact best/worst plateau bounds.  Fifteen of 32 arms
are non-ranking or cutoff-ambiguous.

Material corrections include:

- current-trained active transfer shrinks to a constant for all three older
  destinations.  Its previously displayed lifts of 2.06x, 1.33x and 1.28x
  become exactly 1.00x in expectation and are not model evidence;
- canonical-to-current active has only 19 unique scores and a 1,168-row
  cutoff plateau.  The timestamp-tie lift of 1.72x becomes 1.013x expected;
- canonical-to-current onset is constant; its 1.24x timestamp-tie lift becomes
  1.00x;
- reconstructed-to-current active remains only a weak hint after correction:
  expected lift 1.156x with a 673-row cutoff plateau; and
- reconstructed-to-current onset has expected lift 1.360x, but a 271-row
  plateau and exact precision bounds from 0 to 0.161, so it is not a reliable
  alert head.

The tie-aware audit supersedes every interpretation of v1 timestamp-tie-broken
top-decile lift.  AUC/AP and tie-aware expected precision/bounds are the valid
diagnostic evidence.  No active, onset, recovery or reversal head is admitted
as a hard veto, trust router, timing action, portfolio control or production
feature.

### Required continuation

1. Preserve the primary barrier taxonomy as a supporting multi-task target,
   not an admission winner. Retain clean probability and clean-payoff
   magnitude as separate candidate channels; adverse/timeout probabilities
   remain reliability/risk context.
2. Test class assignment and payoff magnitude inside the config-routed
   `base -> residual/context -> execution EV` architecture. The standalone
   PIT simplex intentionally excluded base score, residual score, cutoff
   margin/z and candidate-relative context, so it does not answer their
   incremental interaction.
3. Add an explicit within-clean-class ranking head. The positive top-10
   class oracle and negative top-1 oracle show that a three-class mean alone
   cannot resolve the economically extreme tail.
4. Diagnose the July short clean-payoff collapse using causal transition,
   relationship-shift, volatility/range/trend and liquidity context. Treat
   transition features as interactions, never a hard veto.
5. Use the 118,734-row older exact-label bundle to test label learnability and
   regime mechanisms with source/domain flags and historical sample weights.
   Do not mix its frozen-current-spread counterfactual economics with current
   exact execution parity or count it toward the prospective incident gate.
6. Require a challenger to be positive at pooled-global top 10% in both
   source-forward transfers and all five grouped-July folds before causal
   mapping. Only after that gate may the timing/action layer, simple policy
   optimiser and portfolio constraints run.
7. Complete the mandatory IC-to-EV waterfall above. The already measured
   February -> March split attributes only approximately +0.34 bps to changed
   100-bin ordering while rank-to-economics conversion loses approximately
   32.75 bps; opportunity prevalence (-35.9 bps) and favourable payoff
   (-22.27 bps) are the largest observed components, with cost roughly flat.
   Reproduce this on identical rows with confidence intervals and extend it
   through April and May/June/July before choosing new labels or trust
   features.
8. The pooled common-geometry transition diagnosis is complete and negative.
   Next, retain transition state only as a bounded interaction with base/
   residual trust and opportunity/capture channels.  Any new challenger must
   improve current strict-OOF and grouped-July economics, retain tie-aware
   score-dispersion gates, and report the historical non-OOF source separately.

### January--July 2022 inverse-PI direct-utility ablation result

The user explicitly approved a different candidate population before the
frozen PF population begins.  That scope is now implemented as a separate,
non-pooled lineage rather than treated as missing coverage.

The hash-bound exact-ID research panel is:

- `data_perp/artifacts/jan_jul_2022_inverse_pi_exact_id_research_panel_20260730_v1/`;
- 50,880 unique candidates and 69 decision-time features;
- 29 asset, 15 market and 25 transition-dynamic fields;
- no outcome, action or future field in the feature contract.

The authoritative fixed-geometry result is:

- `data_perp/artifacts/jan_jul_2022_inverse_pi_direct_utility_multitask_ablation_20260730_v2/`;
- runner: `scripts/run_inverse_pi_direct_utility_multitask_ablation.py`;
- reusable matched evaluator:
  `extreme_price_movements/inverse_fixed_geometry_evaluation.py`;
- 14/14 focused panel/runner/evaluator tests pass.

The `_v1` ablation report is superseded only because ten July-31 signals have
an August-1 execution timestamp and were initially displayed as a tiny August
book.  `_v2` retains the true execution timestamp but forms the seven monthly
research books by candidate signal month.

The diagnostic uses five equal contiguous signal-time blocks, a two-sided
12-hour path-overlap purge, side-local neural fits, one frozen
64 -> 32 geometry, no HPO, and pooled plus side-shrunk isotonic mapping.
Because each held-out block can train on both earlier and later complementary
blocks, mapping status is explicitly
`out_of_block_train_only_noncausal`.  This is research-only, not walk-forward
or promotion evidence.  All arms use identical candidate IDs/outcomes and one
pooled-global cross-side/cross-timestamp top-k book per signal month.

At pooled-global top 10%:

| Arm | Mean gross bps | Mean net bps | Worst month net bps | Net delta vs direct-only |
|---|---:|---:|---:|---:|
| Market/asset context, direct only | -1.58 | -101.57 | -163.60 | baseline |
| Same context + economic heads | -6.61 | -106.57 | -164.06 | -5.00 |
| Transition dynamics only + economic heads | -21.75 | -121.64 | -179.22 | -20.08 |
| Levels + transition dynamics + economic heads | -14.89 | -114.82 | -177.14 | -13.25 |
| Levels + five bounded base-score x transition-z72 interactions + economic heads | **+1.48** | **-98.53** | **-148.84** | **+3.04** |
| Previous arm + all five path heads | -2.80 | -102.79 | -152.06 | -1.22 |

The one-percent cost is approximately 100 bps per trade across all arms, so
no arm is economically tradable in this counterfactual inverse-product
population.  The result is still mechanistically useful:

1. transition state is not useful as a standalone model input or by broad
   concatenation;
2. compact score-by-transition interactions are incremental and improve the
   worst month by about 14.8 bps;
3. economic auxiliaries alone do not improve the direct head;
4. adding all five path heads together destroys the bounded interaction
   gain, so path heads require individual/low-weight ablations rather than a
   bundle.

Per-head out-of-block learnability is weak.  On the bounded-interaction
economic arm, opportunity AUC is 0.516, adverse-first AUC 0.498, favorable
rank IC 0.044, adverse-magnitude rank IC 0.107, conversion-loss rank IC 0.077
and timeout AUC 0.592 pooled.  The pooled timeout number contains side/base
rate structure: its side-local AUCs are only 0.531 long and 0.558 short.
When all five path heads are added, rank IC is 0.002 peak-MFE, 0.006
time-to-MFE, 0.005 MAE-before-MFE, -0.017 bars-decreasing and 0.060 future
slope.  Only future slope shows a modest recurring signal in this lineage.

The requested base-IC/EV waterfall cannot reproduce the production
base-target IC anomaly here: `score_meta_base_soft_label` is null on all
50,880 rows and `base_score` is the explicitly declared simple grid-momentum
control, not production alpha.  Its monthly gross/net rank IC ranges from
-0.015 to +0.059; gross top-10 is positive only in March (+14.41 bps), May
(+25.78) and July (+21.42), while the fixed approximately 100-bps cost makes
every net month negative.  The production February-to-July IC-to-EV waterfall
therefore remains a separate required workstream on the canonical alpha rows.

Required continuation from this result:

1. carry only the five bounded base-score x transition-z72 interactions into
   the canonical alpha population as a frozen challenger;
2. ablate future slope alone, then each other path head alone at lower weights;
3. redesign opportunity/adverse event classification before treating those
   heads as useful regularizers;
4. test cross-lineage representation transfer without sharing EV calibration
   or economics;
5. keep timing, MAE, target-price and wait actions outside this direct utility
   score layer.

### 2026-07-30 completion: identical-row IC-to-EV waterfall

Required-continuation item 7 above is now complete. The immutable diagnostic
is `data_perp/artifacts/mandatory_ic_ev_waterfall_20260730_v1/`, implemented by
`scripts/run_mandatory_ic_ev_waterfall.py`. It freezes monthly pooled-global
top 1/5/10/20% books, never maps scores, never selects per timestamp or side,
and reports day-block bootstrap intervals without reselecting candidates.

The primary pooled-global base-alpha top-10 result is:

| Month | Gross bps | Explicit cost bps | Net bps | Positive-net share |
|---|---:|---:|---:|---:|
| February 2025 | +49.38 | 100.25 | -50.87 | 50.11% |
| March 2025 | +17.05 | 100.09 | -83.03 | 42.68% |
| April 2025 | +41.86 | 100.21 | -58.35 | 45.82% |
| May 2026 | -- | -- | -76.76 | -- |
| June 2026 | -- | -- | -99.75 | -- |
| July 2026 | -- | -- | -143.89 | -- |

The long-side figures that motivated the diagnostic are also reproduced:
native base-target rank IC is 0.155/0.162/0.226 in February/March/April, exact
12-hour net rank IC is 0.090/0.093/0.143, and direct top-decile net is
-59.39/-91.31/-38.45 bps. Improving native-label IC therefore does carry
some exact-net ordering information, but not enough to overcome changing
opportunity prevalence, payoff scale and the approximately 100-bps cost.

The February -> March pooled top-10 deterioration is -32.17 bps, with a
day-block 95% interval of [-89.89,+16.70]. Holding side x score-ventile
composition fixed explains only +0.26 bps. The leading exact attribution is
positive-net payoff -21.88 bps, positive-net prevalence -21.70 bps,
full-stop prevalence -3.17 bps, timeout prevalence -2.87 bps and cost
+0.16 bps. This is a response-distribution/conversion change, not a loss of
base rank geometry. March -> April improves +24.68 bps, interval
[-9.71,+62.60], led by full-stop prevalence +12.12 bps, positive prevalence
+7.43 bps and composition +3.98 bps, partly offset by positive payoff
-3.17 bps.

The same failure strengthens in the current period. May -> June is driven
mainly by adverse payoff (-22.80 bps), timeout payoff (-14.84 bps) and timeout
prevalence (-6.55 bps), partly offset by positive prevalence (+10.67 bps).
June -> July is driven by positive prevalence (-45.16 bps), adverse
prevalence (-27.85 bps) and positive payoff (-16.00 bps), while rank-cell
composition contributes +23.61 bps. July's top-10 interval is
[-188.22,-113.06] bps. No residual or direct stream is recurrently positive.

Interpretation and workstream impact:

1. `base -> residual/context -> execution EV` remains the right separation,
   but rising base IC is not accepted as evidence of improving tradability.
2. The execution layer must model the conditional response distribution:
   opportunity probability, favorable payoff scale, adverse-loss severity,
   timeout/stop mixture and exit-policy conversion, with direct net EV still
   the primary target.
3. Base-score rank and margin should be retained as context/interactions; a
   corrective mapper cannot repair a period in which positive prevalence and
   payoff collapse.
4. The next controlled tests must match/reweight months on side, asset,
   spread, volatility, candidate-group size and score ventile, then test
   whether conversion targets remain different. If they do, add causally
   observable transition interactions and conditional experts; do not route
   on latent or future regime identity.
5. The legacy native alpha target is 24 hours while the exact execution
   outcome is 12 hours. A native-12h alpha-label parity ablation remains
   required before attributing all of the IC/EV gap to market conversion.

### 2026-07-30 completion: older transition materialisation and pooled diagnosis

Required-continuation item 8 above is also complete as a diagnostic
workstream. It does not produce a promotable transition head.

Materialised contracts:

- `failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1`:
  118,734/118,734 exact candidates, 212 decision-time fields, no fill;
- `failure_2022_2023_pf_exact1m_causal_global_book_mapping_20260730_v1`:
  117,700 causally mapped rows after an honest 1,034-row warm-up, using the
  frozen 21-day pooled-global mapper and minimum eligible support 1,008;
- `failure_2022_2023_pf_exact1m_global_book_transition_labels_20260730_v1`:
  exact before `[s-H,s)` and after `[s,s+H)` labels at 1/5/10/20/100% depth,
  including 22,418 exact H3/H12 top-10 rows and declared availability times;
- `historical_current_common_transition_geometry_20260730_v1`: nine exact
  raw concepts expanded to 90 common robust level/gap/delta fields, with
  118,734 historical candidates, 11,726 historical hourly contexts and 4,176
  current-v4 contexts, all exact signal+1h and no as-of fill;
- `pooled_transition_classification_readiness_20260730_v7`: all four
  transition-context, semantic-contract, causal-mapping and exact-label gates
  pass. Earlier v2/v6 readiness artifacts are superseded, not overwritten;
- `pooled_historical_current_transition_panel_20260730_v1`: 17,320 exact H12
  pooled-global top-10 rows, 90 common fields and 50 uniformly rebuilt target
  variants across 2022-23 historical, January-April 2025 reconstructed,
  February-April 2025 canonical and May-July 2026 current sources.

The first strict classifier screen is
`pooled_historical_current_transition_classifier_20260730_v1`. It uses
five grouped seven-day folds, a two-sided 36-hour purge, fold-local
preprocessing, source-balanced weights and nested Brier shrinkage. Current
fitting/reporting retains strict OOF rows only; 2022-23 remains explicitly
non-OOF diagnostic data.

| Target | Rows / positives | Best pooled AUC | Best pooled AP | Best top-10 lift |
|---|---:|---:|---:|---:|
| Active adverse | 17,274 / 3,628 | 0.505 logistic | 0.223 | 1.13x |
| Adverse onset within 3h | 17,234 / 1,240 | 0.491 logistic | 0.066 | 1.07x ExtraTrees |
| Recovery within 3h | 3,623 / 452 | 0.484 ExtraTrees | 0.109 | 1.17x |
| Reversal after recovery within 3h | 17,229 / 21 | 0.342 ExtraTrees | 0.0009 | 0.00x |

Current strict-OOF source metrics are likewise weak: active adverse reaches
AUC 0.524 and 1.14x top-decile lift with ExtraTrees; onset reaches AUC 0.521
but only 0.78x top-decile lift with logistic; recovery is at chance and only
35 positives; reversal has two positives. These figures do not support an
admission veto, regime router, timing action or portfolio control.

Source transfer is asymmetric. Reconstructed January-April 2025 -> canonical
February-April active adverse reaches AUC 0.675 and 2.31x top-decile lift;
canonical -> reconstructed reaches AUC 0.663 and 1.92x. The same reconstructed
source -> current result is much weaker: AUC 0.541, AP 0.238 and tie-aware
expected lift 1.156x for active; AUC 0.535, AP 0.071 and expected lift 1.360x
for onset, whose cutoff plateau leaves precision bounds of 0--0.161.
Historical 2022-23 -> current is below chance for active (AUC 0.461, lift
0.994x) and provides no useful onset tail (AUC 0.506, lift 0.722x). This
indicates a limited early-2025 recurrence, not a stable cross-era transition
classifier.

The immutable no-refit correction is
`pooled_historical_current_transition_score_tie_audit_20260730_v1`. It finds
15/32 arms non-ranking or cutoff-ambiguous. In particular, current-trained
active transfer scores shrink to a constant prior, so their previously
printed 2.06x/1.33x/1.28x timestamp-tie lifts all become exactly 1.00x.
Canonical -> current active has only 19 distinct scores and a 1,168-row cutoff
plateau; its 1.72x printed lift becomes 1.013x expected. Any future top-k
evaluator must require non-zero prediction dispersion and report boundary-tie
mass, expected tie allocation and precision bounds.

### Updated next ablations after the completed materialisation

1. Rebuild native alpha labels at the exact 12-hour execution horizon and run
   the fixed-row waterfall against the legacy 24-hour target. Add target IC,
   exact gross IC, exact net IC, opportunity recall and conditional payoff by
   score ventile.
2. Fit a hurdle/distributional execution model on identical strict-OOF rows:
   `P(gross > cost)`, favorable magnitude, adverse magnitude, timeout/stop
   mixture and direct net. Compare joint loss, stopped-gradient support heads
   and direct-net-only; keep side-local fitting but one pooled common-unit map
   and one global top-k selection.
3. Diagnose conversion shift with matched/reweighted month pairs. Decompose
   covariate shift from conditional label shift and test base-score x compact
   transition interactions one family at a time. The five bounded
   score-by-transition interactions remain the first frozen challenger.
4. For transition classification, replace broad 90-field concatenation with
   sparse mechanism-specific groups (volatility compression/release, trend
   acceleration, leverage build and memory asymmetry), their short deltas and
   base-score interactions. Test active/onset first; recovery/reversal remain
   under-supported.
5. Add source-held-out/domain-generalization objectives, source-specific
   intercepts, invariant-risk penalties and importance weighting. Report
   within-source grouped OOF separately from every source-transfer direction;
   never pool them into one headline.
6. Repair top-k evaluation for constant/tied scores. Require prediction
   dispersion, threshold tie mass, alert rate, precision/recall, false alerts
   per day, month/week coverage and side/asset concentration.
7. Test July-local grouped OOF and adjacent-week transfer using the same
   labels. Only if July is learnable locally should leaf/context clustering
   be used to extract a causally observable July state and train weighted
   experts. A regime expert must improve an untouched later block and degrade
   gracefully when regime probability is uncertain.
8. Do not replay the simple policy optimiser or portfolio constraints until a
   frozen execution challenger is positive at pooled-global top 10%, covers
   the latest month/week, beats base/control, remains side-balanced and passes
   causal mapping. Timing, MAE, target-price and wait actions remain a
   separate action layer.

### 2026-07-30 completed continuation: horizon parity, conversion shift and hurdle architecture

The first three post-waterfall ablations are complete.  They use sealed,
immutable artifacts and do not authorize policy or portfolio replay.

#### Exact 12-hour versus legacy 24-hour base-label parity

`febapr2025_exact12h_legacy24h_base_label_parity_20260730_v1` compares the
already frozen legacy-24h and native-12h OOF base scores on exactly 509,868
February--April candidates.  Decision timestamps and candidate IDs are
identical; the legacy target resolves at decision+24h, while the new native
target and exact execution outcome resolve at decision+12h.  Selection is one
pooled-global top 1/5/10/20% book; side rows are attribution only.

| All February--April | Legacy-24h score | Native-12h score |
|---|---:|---:|
| IC to legacy native target | 0.1752 | 0.1632 |
| IC to native-12h target | 0.1609 | **0.1683** |
| IC to exact-12h gross/net | **0.1199** | 0.0913 |
| Global top-10 gross | **+37.64 bps** | +33.11 bps |
| Global top-10 cost | 100.19 bps | 100.17 bps |
| Global top-10 net | **-62.55 bps** | -67.05 bps |
| Opportunity precision / recall | **46.51% / 12.51%** | 44.50% / 11.96% |

Monthly old/new top-10 net is -50.87/-52.56 bps in February,
-83.03/-83.26 bps in March and -58.35/-67.93 bps in April.  Both long and
short exact-net IC fall under the 12h score.  Therefore horizon alignment
improves the intended native label but moves the base farther from the actual
exit-policy economics.  Do not replace the legacy score; the execution layer
must learn the cost-aware conversion explicitly.

#### Matched month-pair conversion shift

`matched_month_pair_conversion_shift_20260730_v1` freezes each raw-base
monthly pooled-global top 10% book and propensity-reweights the earlier month
toward the later month using only causal side, asset, score-ventile,
candidate-group-size, liquidity/spread, volatility/range/trend and transition
context.  It reports support coverage, effective sample size and balance and
fails closed where overlap is inadequate.

| Pair | Support gate | Raw net change | Composition | Conditional response | Support restriction |
|---|---|---:|---:|---:|---:|
| February -> March | Fail: target coverage 34.9% | -32.17 bps | not interpretable | not interpretable | not interpretable |
| March -> April | Pass | +24.68 bps | +5.95 bps | **+32.38 bps** | -13.65 bps |
| May -> June | Fail: ESS ratio 0.071, max SMD 0.656 | -22.99 bps | not interpretable | not interpretable | not interpretable |
| June -> July | Pass | -44.14 bps | +5.45 bps | **-53.11 bps** | +3.52 bps |

For the supported June -> July comparison, covariate composition would
slightly improve the book.  The deterioration is conditional: opportunity
falls 14.33 percentage points, favorable gross payoff falls 41.59 bps,
full-stop incidence rises 6.47 points and timeout incidence rises 4.69
points, while cost improves 0.27 bps.  This confirms that July is primarily a
change in `P(opportunity)` and conditional path/payoff conversion, not a
different mix of observable candidates.

#### Direct-primary hurdle/distributional execution ablation

The authoritative result is
`exact_strict_oof_hurdle_distributional_ablation_20260730_v3`.  `_v1` is
explicitly invalidated because it was unsealed; the unpublished `_v2` staging
was removed.  `_v3` trains every model per side on strict resolved rows,
applies the causal common-unit recent-EV map, uses deterministic candidate-ID
ties and evaluates one pooled-global top 1/5/10/20% book.  Timing, wait,
target-price and MAE actions are absent.

Tested arms are direct net, `P(gross > cost)` times conditional gain/loss,
actual full-stop/timeout/other-exit mixture, direct/exit blend, CatBoost
MultiRMSE with the direct residual repeated three times, and a stopped-gradient
Ridge residual using only earlier OOF support-head predictions.

| Mapped global top 10% | May -> June | Later July |
|---|---:|---:|
| Direct net control | -124.82 bps | -93.00 bps |
| **Gross-cost hurdle EV** | **-81.35 bps** | **-71.55 bps** |
| Exit-policy mixture | -169.27 bps | -74.37 bps |
| Direct/exit blend | -133.22 bps | -79.06 bps |
| Direct-primary joint MultiRMSE | -121.96 bps | -86.65 bps |
| Stopped-gradient residual | -147.88 bps | -111.62 bps |

The gross-cost hurdle is the only arm that improves materially over direct in
both windows.  It is now the frozen **research comparison control**, not a
promotion candidate.  It remains negative at every mapped global depth in
both windows, negative in the latest week, and side-unstable (11.6% long in
May--June, 68.5% long in later July).  Later-July opportunity and timeout
heads are learnable (AUC 0.639/0.642), but May full-stop AUC is 0.428.  The
joint and stopped-gradient designs are rejected.

### 2026-07-30 completed continuation: sparse transition and July-local diagnosis

`pooled_historical_current_sparse_transition_mechanism_ablation_20260730_v1`
compares the 90-field control with seven fixed groups: compression/release,
EMA/trend acceleration, leverage build, memory/range recurrence, sparse state
levels, 1h/3h deltas and their compact union.  It uses grouped/purged pooled
OOF, separate current strict-OOF fitting and reconstructed-2025/historical
source transfer into current.  Every top-decile metric is tie-aware.

The broad all-90 model is not rescued:

- current-local active adverse is below chance for every clean arm; apparent
  state/compact lifts are cutoff-ambiguous;
- current-local onset has one clean but small signal:
  compression/release AUC 0.538, AP 0.070 versus 0.065 prevalence, top-10
  precision 0.094 and lift 1.443x;
- reconstructed January--April 2025 -> current active has the best clean
  transfer cells: sparse state levels AUC 0.557/lift 1.356x and memory/range
  AUC 0.583/AP 0.276/lift 1.296x;
- 2022--23 -> current does not confirm those effects; and
- reconstructed onset transfer is either weak or cutoff-ambiguous.

These are mechanism-monitoring leads only.  The same active features do not
learn within current, so no transition veto, trust router or expert is
authorized.  Compression/onset and memory/state active predictions may next
be tested as uncertainty or interaction context, never raw admission scores.

`july_local_exact_h12_transition_diagnosis_20260730_v1` then separates local
learnability from cross-period transfer:

- strict-OOF July active has only 198 rows/31 positives over two UTC weeks;
  strict onset has 188/10.  Three-fold grouped OOF correctly fails closed;
- adding resolved frozen-forward rows for diagnosis yields active 228/52
  across three weeks.  Grouped/purged logistic AUC is 0.218, AP 0.145 versus
  0.228 prevalence, and tie-aware top-10 lift is zero;
- adjacent-week active transfer is also negative: AUC 0.205, AP 0.131 versus
  0.204 prevalence and zero top-10 lift; and
- onset arms mostly skip because fewer than ten purged positives remain.

The July 20--23 retrospective cannot legally extend this panel yet.  It has
5,760 exact candidate-economic rows, but its frozen bridge explicitly lacks
causal recent-EV mapping coordinates, causal pooled-global before/after
top-10 labels and the strict 90-field decision-time geometry.  Do not call
those observations transition-classifier coverage until those three inputs
are materialised.

### Architecture and executable queue after this tranche

The next architecture remains:

`base -> residual alpha + CatBoost + auxiliaries -> execution EV -> separate action layer`

Within execution EV:

1. Keep direct net as the primary training target.
2. Retain the opportunity hurdle, conditional favorable/adverse magnitude
   and timeout/stop heads as support; use the frozen gross-cost hurdle EV as
   the comparison control.
3. Do not use the current joint MultiRMSE or stopped-gradient stack.
4. Keep transition probabilities out of raw ranking.  Compression/onset and
   memory/state may enter only as bounded uncertainty/interactions after
   source-forward replication.
5. Keep the recent-EV map pooled and causal, but repair cross-side common-unit
   anchoring before interpreting a globally ranked book.
6. Keep timing, MAE, target price and wait actions in the separate action
   layer.

Next experiments, in order:

1. Materialise July 20--23 causal mapping coordinates, pooled-global
   before/after labels and strict common transition geometry, then rerun
   July-local and adjacent-week tests.
2. On frozen strict rows, ablate direct/hurdle blends at
   0/25/50/75/100% hurdle weight.  Choose weight using earlier OOF only,
   freeze it, and evaluate both forward windows once.
3. Compare the current mapping with a pooled global anchor plus side-residual
   shrinkage and exact zero fallback.  Require latest-week coverage and a
   non-collapsed side share without imposing side quotas.
4. Add only the clean compression/onset and memory/state predictions as
   uncertainty penalties or bounded score-by-hurdle interactions.  Report
   mechanism-conditioned economics; do not rank on the probabilities.
5. Extend exact incidents before February and after July.  No conditional
   expert is permitted until the regime mechanism is locally learnable and
   improves an untouched later block.
6. Continue to block simple-policy and portfolio replay until a mapped
   pooled-global challenger is positive, latest-period positive,
   side-balanced and better than the frozen controls.

### Explicit active item: why improving base IC does not reliably convert to EV

The February--April observation remains an active, first-class workstream.
On identical canonical rows, native base-target rank IC improves
`0.155 -> 0.162 -> 0.226`, and exact 12-hour net rank IC also improves
`0.090 -> 0.093 -> 0.143`, while the base-ranked pooled-global top-decile
exact execution outcome is `-59.39 / -91.31 / -38.45 bps`. This is not
explained away by the architecture statement that the base learns alpha: the
conversion failure itself must be measured and, if learnable, modelled.

The current evidence says that global-average rank quality and tail economics
are separating. February -> March loses 32.17 bps at top 10%; fixed
side-by-score-ventile composition explains only +0.26 bps, while positive
payoff size and positive-net prevalence deteriorate materially. March ->
April then recovers 24.68 bps through stop/prevalence changes. Native 12-hour
label parity does not repair the issue, and supported matched-month tests
point to conditional-response shift rather than candidate composition alone.

The corrected frozen numeric-band diagnostic adds a second mechanism.  A
February top-decile threshold admits only 7.283% of March and moves from
-50.87 to -76.18 bps; the frozen top ventile contracts from 5.00% to 3.06%
and its within-band response worsens by 17.17 bps.  A March top-decile
threshold, however, admits 9.634% of April, and the highest-ventile
within-band response improves by 30.49 bps.  Therefore February -> March
combines score-scale/cutoff migration with conditional payoff deterioration,
whereas March -> April is mainly an economic-response recovery at broadly
stable score scale.  The aggregate IC increase is real but is not sufficient
to locate either effect.

Add the following to the live `base -> residual/context -> execution EV`
workstream:

1. tail-local IC and calibration at global top 1/5/10/20%, with opportunity
   precision/recall, favorable/adverse magnitude and stop/timeout incidence;
   decompose aggregate IC into admitted-tail, near-cutoff and non-admitted
   contributions;
2. an identical-candidate target bridge from native alpha through exact gross,
   deployed-exit gross, explicit cost and net;
3. frozen source-month score-ventile transition matrices into the next month,
   comparing fixed numeric thresholds with fixed-quantile books to separate
   score compression from response drift;
4. older-data support repair for the currently unsupported February -> March
   matched decomposition;
5. fixed-book cost-hurdle sensitivity, without reselecting at each cost;
6. exact-row comparisons of raw base, residual alpha, direct EV and causal
   mapped EV to locate the layer that changes tail precision or calibration;
7. bounded causal score-by-conversion interactions for opportunity,
   favorable contribution, adverse risk/severity, exit mixture and the
   retained compact transition mechanisms; and
8. conditional experts only if within-state learnability and decision-time
   state recognition are first demonstrated on untouched blocks.

All evaluation remains pooled-global top-k after the recent causal EV map;
side and asset results are attribution and concentration checks, never
separate selectors. Close the item only after the IC-to-EV delta is
quantitatively reconciled on identical rows and a frozen challenger is
positive, latest-period covered, side-balanced, tie-safe and better than the
direct/base controls.

### Completed frozen blend and common-unit mapping controls

`frozen_hurdle_blend_ablation_20260730_v1` performs no model refit and
selects one direct/hurdle weight from resolved May development OOF only.
Pooled-global top-10 net for 0/25/50/75/100% hurdle is
`-112.18/-115.65/-118.07/-122.48/-128.04 bps`; the frozen winner is direct
only.  Forward mapped results are:

| Window | 0% | 25% | 50% | 75% | 100% hurdle |
|---|---:|---:|---:|---:|---:|
| May -> June | -124.82 | -121.96 | -105.91 | -91.53 | -81.35 |
| Later July | -93.00 | -80.24 | -74.97 | -69.07 | -71.55 |

These are diagnostics after the freeze; every arm is negative and they
cannot be used to change the chosen weight.  Pre-map direct ordering is
better in May but hurdle ordering is better in July, confirming period
instability rather than a transferable blend.  Manifest seal:
`870e9df0ebe6e371e1d8348e1dd1cf99986c8258b3e0beccce13e0ca9c2078e9`.

`hurdle_cross_side_common_unit_mapping_20260730_v1` then compares the
canonical mapper with a strictly causal 21-day pooled anchor and
side-residual shrinkage with exact-zero weak-support fallback.  Development
OOF freezes shrinkage at 4,000.  Later-July top-10 improves from `-71.55` to
`-57.64 bps`, but is still negative and cutoff-tie ambiguous.  May -> June
falls to `-112.30 bps` with 0% long selection.  Thus one fixed cross-side
residual mapping does not transfer and all promotion gates fail.  Manifest
seal:
`b663c5264966a038b03678ab690baea3e3acbc201f3609e925ed7a4ec0f741a4`.

No simple-policy or portfolio replay is authorized.  Direct net remains the
primary execution score; hurdle components remain support/control heads.
The next work is July 20--23 causal materialisation, the identical-row
IC-to-EV tail bridge, and bounded conversion interactions without side
quotas or forward-window selection.

### Completed July 20--23 prerequisite with strict provenance separation

`july20_23_retrospective_allscore_transition_readiness_20260730_v1` audits
the original 5,760-row raw-score bridge and fails closed.  The bridge
explicitly excludes mapped execution EV/global admission and does not bind a
causal mapping state or strict common 90-field geometry.  Its 2,880 rows per
side cannot legally extend the July-local OOF/adjacent-week panel.

A separate retrospective extension,
`july20_23_exact_h12_transition_inputs_20260730_v1`, uses the authoritative
v2 frozen scorer without pretending to be the original bridge.  It preserves
the same 5,760 identities and materialises 5,760 mapped candidates, 4,380
honest coordinate rows after causal warm-up, 96 exact H12 label anchors, 84
strict-geometry anchors and 73 complete transition-panel rows.  It is
explicitly non-OOF/non-promotable.  Panel SHA:
`07250e40fa5e252662c002148ae856f7e6c72cd8d57c64671f6b2cfc9f8e894f`;
manifest SHA:
`c7cab6a48990348d54b465d95065f9a323fb5c2265aa913d2f94c145e94a6ea4`.

The provenance-separated rerun
`july_local_exact_h12_transition_diagnosis_20260730_v2` leaves the conclusion
unchanged.  Active-adverse grouped diagnostic OOF has 299 rows/54 positives,
AUC 0.301, AP 0.124 versus 0.181 prevalence and zero expected top-10 lift.
Adjacent-week active transfer has 179/24, AUC 0.190, AP 0.082 versus 0.134
prevalence and zero lift.  The July 20--23 block alone has only 71 active
rows/two positives: AUC 0.580, AP 0.048 and zero lift.  Onset has only 64
usable diagnostic predictions/three positives, AUC 0.383 and zero lift.
No transition router, veto, action or portfolio control is authorized.

### Completed identical-row four-score IC-to-EV diagnostic

`identical_row_four_layer_ic_ev_diagnostic_20260730_v1` joins raw base,
residual expected EV, direct q25 EV and causal mapped base-score economics on 140,682 exact
common March--April rows.  All four identity fields and exact
net/gross/cost/MFE/MAE/exit/opportunity labels match with 100% mapped
coverage.  May--July fails closed because no causal mapped EV score exists on
the same canonical-alpha identities; no substitute population is used.

| Pooled-global March--April top 10% | Full net IC | Tail net IC | Opp. precision/recall | Gross/cost/net bps |
|---|---:|---:|---:|---:|
| Raw base alpha | 0.0884 | 0.1050 | 46.96% / 12.04% | +50.30 / 100.25 / -49.95 |
| Residual expected EV | 0.0869 | 0.0948 | 46.22% / 11.85% | +73.78 / 100.37 / -26.59 |
| Direct EV q25 | 0.0899 | 0.0305 | 46.59% / 11.94% | +44.77 / 100.22 / -55.45 |
| Causal mapped base-alpha economics | 0.0519 | 0.0079 | 43.22% / 11.08% | +34.35 / 100.17 / -65.82 |

The residual book improves economics by selecting larger favorable payoffs
even though its aggregate IC is slightly lower.  The q25 direct challenger
has much weaker pooled tail-local ordering.  However, the mapped comparator
is not downstream of q25: historical `mapped_direct_net` is an unfortunate
alias for a causal economics map whose bound raw input is canonical
`score_base_alpha`.  It therefore shows that this mapped-base comparator has
lower tail IC/opportunity precision than raw base, but it cannot establish
raw-direct -> mapped-direct degradation.  Fixed-book
zero-cost/deployed-cost results prove that cost makes all arms negative but
does not explain their ordering.

Treat v1 as a four-score identical-row comparator, not a sequential layer
attribution.  A lineage-explicit v2 must rename the mapped-base comparator,
add month-level results and assert the raw-score source/model hash.  A true
direct mapping must be materialised separately before evaluating whether
mapping repairs or damages direct-tail calibration.

Manifest seal:
`c21c7d00c99ef0563ed3034431efc82cd506feda1136f47457b5c8bde4754aca`.
The artifact also reports pooled-global top 1/5/10/20, calibration where
units permit, favorable/adverse magnitude, MFE/MAE, exit mixture,
cutoff ties and separate side/asset attribution.

### Bounded direct-tail repair: v1 invalidated, causal v2 authoritative

`bounded_direct_tail_repair_20260730_v1` must not be used: its initial April
fit included late-March labels that were not resolved before the first April
decisions.  The separately sealed correction is
`bounded_direct_tail_repair_20260730_v1_correction_20260730_v1`, seal
`a45c94d7b919167e3e5f936cbf1d1db2b451b4d17a78d9f6680960a10f72fbe5`.

The corrected `bounded_direct_tail_repair_20260730_v2` starts April
confirmation at 2025-04-01 01:00 UTC and enforces maximum fit/calibration
label end 2025-04-01 00:00 UTC.  It uses per-side fixed-geometry
histogram-gradient models over the frozen base/residual/direct score
lineages.  March chronological OOF selects the residual-interaction weight
from the bounded `{0,.25,.5}` grid once; the frozen value is `.5`.  Timing,
MAE, target-price and wait fields are excluded.

| Raw global top 10% | March matched OOF | April confirmation |
|---|---:|---:|
| Incumbent direct q25 | -83.71 bps | -93.24 bps |
| Tail-weighted direct | -67.49 bps | -41.79 bps |
| Robust decomposed | -74.81 bps | **-33.49 bps** |
| Residual x conversion | -72.42 bps | -72.13 bps |

The March figures cover only the common chronological-OOF evaluation subset,
so the incumbent `-83.71 bps` is not a contradiction of the earlier
full-March `-21.76 bps` diagnostic.  Tail weighting and robust decomposition
improve both matched development and untouched confirmation versus the
incumbent, but remain negative.

`bounded_direct_tail_repair_20260730_v2_supplement_20260730_v1` completes the
reporting contract under seal
`67fb02db6712b181b929d4cf0dbd7567010ffa1e801775e9d92791e3b71dbff0`.
For robust decomposition, raw April top-10 is -33.49 bps, long attribution is
+2.96 bps and short attribution is -50.49 bps; the latest confirmation block
falls to -91.14 bps.  Raw calibration MAE is about 222 bps.  Its causal map
is -32.87 bps deterministically and -33.17 bps under random cutoff-tie
allocation, with best/worst bounds -10.03/-61.04 bps and a 2,029-row cutoff
plateau.  Tail-weighted mapping has a 1,601-row plateau.

No raw or mapped arm passes positive top-10/latest gates.  The retained
research direction is robust favorable-minus-adverse decomposition plus
tail-aware training, with the next ablation focused on short-side/latest
failure and strict-OOF meaningful-MFE/peak contribution and future-slope
support.  No side quota or policy/portfolio replay is authorized.

### Current exact direct-q25 causal mapping lineage

`mayjul_identical_four_layer_mapping_readiness_20260730_v1` first failed
closed because the exact q25 output did not persist per-row score availability
or fold/fit cutoffs.  This was a provenance gap, not a missing score: all
127,777 waterfall rows were bit-identical to `q25_net_bps`.

`mayjul_exact_direct_q25_causal_mapping_20260730_v1` reconstructs that
provenance without model refit or score mutation.  May has 63,351 uniquely
assigned OOF rows, June 49,259 and July 15,167.  Each bound feature timestamp
precedes decision time, each maximum training-label resolution precedes its
fold cutoff and candidate decision, and exact H12 identities/labels match.
The original dataset, runner, frozen state, final model and direct-score
output hashes are recorded.  Because historical fold binaries were not
persisted, the artifact binds the exact OOF output plus recipe/config/state;
it does not claim the frozen final binary is identical to each fold binary.

The causal map is applied only to the exact direct q25 lineage.  It uses a
21-day prior-resolved window, 500-row minimum support and side shrinkage 500.
It maps 125,551 rows and retains 2,226 honest warm-up rows.  All 72 daily
snapshots prove maximum reference-label end before the snapshot.

| Exact common pooled-global | Top 1% net | Top 10% net | Top-10 positive rate |
|---|---:|---:|---:|
| Raw base | -49.49 bps | -94.00 bps | 36.20% |
| Residual expected EV | -46.92 bps | **-77.63 bps** | 41.88% |
| Raw direct q25 | -44.05 bps | -89.75 bps | 36.63% |
| Causal mapped direct q25 | **-11.58 bps** | -89.57 bps | 38.26% |

The pooled top-1 change is not stable across months.  Raw -> mapped direct
top-10 is `-100.76 -> -115.25` in May, `-42.91 -> -55.10` in June and
`-152.92 -> -178.16 bps` in July.  At top 1%, mapping worsens May and June
but improves July from -163.47 to -88.52 bps; changed cross-month allocation
drives the pooled improvement.  Mapping also creates cutoff plateaus and
severe side shifts, including zero-long cells.

No promotion gate changes.  The exact direct mapping lineage is now
available for diagnosis, but neither raw nor mapped direct is positive at
global top 10%, latest/month coverage is not stable, and the residual
comparator is also negative.  Direct-tail target/representation repair must
precede another mapper or policy replay.

Manifest seal:
`d64c37b01f06333e2243a5d66571ab188d1c16585396737fbd73ddf5752cd038`.

### Exact forward transition geometry and bounded interaction verdict

The v1 retained-mechanism extension failed closed at 81.57% later-July
coverage and did not rank a biased subset.  The missing prerequisite is now
fully materialised in `forward_exact_transition_geometry_20260730_v1`.  It
joins every candidate's `(symbol, signal timestamp)` exactly against 75
hash-bound feature shards, applies the existing nine-concept -> 90-field
per-side median/IQR/gap/exact-lag constructor, verifies
decision = signal+1h, performs no as-of join/fill and preserves genuine
missingness.  Coverage is 49,244/49,244 May -> June and 7,071/7,071 later
July.  Seal:
`752c78a39c0feb397f3bc5c19683a32d73a35e0df8c570f8385eeb156450e8b3`.

`causal_retained_transition_mechanism_extension_20260730_v2` binds those
complete joins, seal
`8cac9003ff8850a48241598d9f52e99f52ed4248cc186d3ca912e2b6eaa424db`.

`causal_retained_transition_interactions_20260730_v2` then refits the
retained fixed mechanism recipes causally because the earlier artifact did
not persist model state.  It makes no state-reproduction claim.  The recipes
are compression/onset (10 fields), memory/active (50 fields) and
state/active (9 fields), using labels resolved before each cutoff and the
same temporal Brier shrinkage.  Only bounded `0/.25/.50` uncertainty and
hurdle×conversion weights are searched on earlier strict OOF evidence.

| Global top 10% | May -> June | Later July |
|---|---:|---:|
| Common-unit mapping control | -112.30 bps | -57.64 bps |
| Selected uncertainty penalty | -112.30 bps (0) | -57.64 bps (0) |
| Selected bounded hurdle×conversion | -112.30 bps (0) | **-54.28 bps (.50)** |
| Latest week: control / interaction | -67.23 / -67.23 bps | -55.53 / **-52.98 bps** |

May -> June remains 0% long and selects zero weight for every mechanism.
Later July's `.50` interaction removes the cutoff tie and improves about
3.4 bps, but remains negative in the whole window and latest week.  No
positive/latest/side/control gate passes, so no policy or portfolio replay
runs.  Retained transition probabilities remain bounded diagnostic
uncertainty/interactions only; they are not admission ranks, routers or
vetoes.

Manifest seal:
`2478433ddaac004aff441e25228a87b550a62df04f3c8b724555a89be1a01f68`.

### Closed diagnostic tranche: frozen bands, crowding overlap and support heads

The corrected frozen numeric-band result is
`frozen_month_score_band_transition_diagnostic_20260730_v2`, seal
`8ae47fa93db1b156fee81e63b59de3e190912f1e02e0a6bda3de4334f826b14c`.
The pre-fix v1 omitted source-frozen contributions and is separately
invalidated under seal
`184b8bd65ef36ba869a3f16ab4e67e903f27130ab47dffc531029511f7fab6c2`.
A frozen February top-decile threshold admits 7.283% of March and moves from
-50.87 to -76.18 bps.  Its top ventile contracts from 5.00% to 3.06%, with
within-band net worsening 17.17 bps.  A frozen March top-decile threshold
admits 9.634% of April, while highest-ventile response improves 30.49 bps.
Therefore February -> March mixes score compression with conditional-response
deterioration; March -> April is mainly response recovery.

The primary February -> March matched decomposition remains unsupported.
`febmar_overlap_crowding_sensitivity_20260730_v2`, seal
`38485e1d9d7edf5ec0899c70c8d39449065b74590d6394d2d1b41dd62413916e`,
tests two predeclared outcome-blind estimands:

| Estimand | March coverage | ESS | Max SMD | Conditional net shift |
|---|---:|---:|---:|---:|
| Side + asset + frozen score ventile; omit absolute group size | 99.12% | 14,853 | 0.014 | -36.36 bps [-89.55,+16.79] |
| Same + raw/log candidate-group size | 34.91% | 8,436 | 0.041 | Not computed; fails closed |

The supported group-size-omitted estimand also loses 52.46 bps of favorable
gross [95% day-block interval -90.31,-17.29] and 7.43 percentage points of
opportunity incidence [-15.83,+1.76].  It is diagnostic only.

A direct audit of the field prevents a misleading regime conclusion.
February contains only 236/238 rows per timestamp, exactly 118/119 assets x
two sides; March contains 238/240, exactly 119/120 assets x two sides, after
KAITO enters the universe.  April similarly tracks 120--122 assets x two
sides except for 12 partial 75-asset hours.  Therefore
`candidate_group_rows` is principally eligible-universe cardinality, not
observed signal crowding.  The q0/q2 non-overlap is a real categorical support
failure for that exact field, but it is largely mechanical and must not be
called a market transition.  Older data should not be materialized merely to
match an arbitrary asset count.

The required replacement splits decision-time geometry into eligible-universe
cardinality, count/fraction clearing the base cutoff, side imbalance and score
density near the pooled-global cutoff.  Only the latter three are candidate
market-state variables.  Re-run the overlap audit with this decomposition;
treat universe cardinality as nuisance/provenance and retain the supported
side + asset + frozen-score-ventile estimand as the economically relevant
current comparator.

The corrected auxiliary-support ablation is
`bounded_robust_auxiliary_contribution_ablation_20260730_v2`, seal
`afebfb8ce2108d849dc2f0ff18bdcae6fa3317afe685c5aeb05c762498560e81`;
its provenance/invalidation sidecar is sealed by
`ead17f6393fa312a66222c9e76b440d6009fd68b775660dbef1a62192e521496`,
and the expanded gates by
`f9924d058bb459ca44a8e518dfb7e3c584ebc8ed00b52de1df28f960f4704f17`.
V1 used the wrong q25 control and selected a map on its own fit labels; it is
invalid.  V2 reconstructs the robust-decomposed control bit-identically,
selects weights only on raw March chronological OOF and applies a March-fit
map only to April.

March freezes `future_slope @ .25` at -61.58 bps versus -70.57 for the
control.  On April it improves raw top-10 from -33.49 to -30.44 bps, but the
mapped random-tie expectation is -31.66 bps and the latest mapped top-10 is
-89.05 bps.  Selected long is +1.79 bps mapped, while short is -48.32 bps.
The apparent deterministic mapped top-1 of +0.98 bps is not evidence:
706 tied rows compete for 515 remaining slots and random-tie expected top-1
is -14.59 bps.  Peak contribution is not incrementally useful at its frozen
.25 weight.  Calibration and all economic/latest/side gates fail.  No
portfolio replay ran.

Fifteen focused tests pass across the frozen-band, auxiliary and overlap
artifacts.  Two legacy propensity tests emit an upstream L-BFGS deprecation
warning under the installed SciPy version; the warning does not affect the
sealed numerical results.

The next bounded repair is side-local support composition inside the robust
decomposition: select small per-side slope/favorable/adverse weights on March
chronological OOF, keep one pooled-global selection book, and evaluate April
once.  Add a high-score x true signal-density interaction only after the
causal density decomposition above exists; raw universe cardinality is not a
regime proxy.  Do not use timing, MAE, target-price or wait-action fields.
Replay remains blocked until raw and tie-expected mapped top-10, latest-block,
two-side and calibration gates all pass.

The interpretation correction is sealed at
`febmar_eligible_universe_interpretation_20260730_v1`, manifest
`d408610f44122267ca37cbf9bf4ec24353b3c4067ffd8dd0346b7bb3f470265d`.
It binds the canonical panel and both overlap artifacts, prohibits the
market-crowding claim, and confirms that no causal normalized true
candidate-density field exists in this panel.  A future materializer needs
raw/pre-filter candidate counts plus a decision-time fixed denominator before
density can be tested as a transition feature.

### Corrected side-local support-composition verdict

`bounded_side_local_support_composition_20260730_v1` searches 64 fixed
`{0,.15}` per-side combinations of positive peak contribution, positive
future slope and negative predicted MAE severity.  March chronological OOF
selects one configuration using one pooled-global raw top-decile; there are
no side quotas.  The April control is bit-identical to robust decomposition.

The MAE mixture is a valid adverse-risk support input: its source manifest
and predictions are hash-bound, and 12 role x side x March/April fold proofs
show the training-label resolution precedes validation.  It is used only as
predicted loss severity in ranking, never as timing, target-price, wait or MAE
action.

The v1 tie report computed random-tie expected precision incorrectly, although
its expected net was correct.  Do not use those v1 precision fields.  The
authoritative chain is:

- tie correction manifest
  `ab2779e0d8c2ca3d6ea3bb36105a45d50a897a894e339d9043d8596ac6d70c2f`;
- final wrapper `bounded_side_local_support_composition_20260730_v3_final_seal`,
  manifest
  `bdb32f1f5443a05f4514d5e49d822d9ec9c36136d6e57ee8c56b84e3cf992395`,
  seal
  `4ba4885d827adc3ff905c38b9c2f3ea24bef3051c5ccd23373c0737bd2f9dee1`.

The selected weights are identical on both sides: peak `.15`, slope `.15`,
adverse `0`.  March development top-10 is -61.54 bps.  April results are:

| Pooled-global depth | Raw net | Mapped deterministic | Mapped random-tie expected |
|---|---:|---:|---:|
| 1% | -24.21 | +1.05 | -18.37 |
| 5% | -11.28 | -11.58 | -12.30 |
| 10% | -30.18 | -30.95 | -30.03 |
| 20% | -46.17 | -48.15 | -48.90 |

The deterministic mapped top-1 is invalid for promotion: 770 rows tie at the
cutoff for a 693-row book.  Corrected mapped expected precision at
1/5/10/20% is 0.4387/0.4826/0.4679/0.4473.  Latest raw/mapped top-10 is
-89.27/-85.72 bps.  At top 10%, long is +3.02/+1.32 bps raw/mapped and short
is -47.37/-47.96 bps.  Raw bias/ECE is +84.84 bps; mapped is -44.40 bps.

Relative to the earlier support screen, raw top-10 improves only 0.26 bps
over frozen future-slope `.25` and 3.31 bps over the robust control.  The
adverse weight freezes at zero and the two side recipes collapse to the same
configuration.  Thus no side-local or adverse-support advantage is
demonstrated.  All economics, latest, short-side, mapped-tie and calibration
gates fail; no portfolio replay ran.  Stop additive support-weight sweeps and
move to a short-side conditional payoff/loss-conversion repair.

### Frozen score-signal-density sensitivity

`febmar_true_signal_density_overlap_20260730_v1`, manifest
`9bbdb2eb79723ace9f406ea03d6401cfa1a028041a65b58db431b49bfa8eda1c`,
separates true score density from eligible-universe cardinality.  It freezes
on February only:

- numeric pooled-global top-10 base-score cutoff `0.4877426318`;
- near-cutoff width `q95 - q90 = 0.0633044937`;
- per-hour count/fraction above the cutoff;
- long-short imbalance among above-cutoff rows; and
- fraction inside the frozen near-cutoff band.

All fields use signal-time canonical base OOF scores and identities.  No
outcome, mapping, exit, March normalization or eligible-asset-count covariate
enters support fitting.

| Support/decomposition result | Value |
|---|---:|
| February / March common-support coverage | 98.34% / 98.88% |
| Weighted ESS / ratio | 11,774 / 0.751 |
| Maximum post-weight SMD | 0.030 |
| Conditional net shift | -36.21 bps [-86.24,+11.23] |
| Conditional favorable-gross shift | -47.60 bps [-80.81,-12.43] |
| Conditional opportunity-incidence shift | -7.00 pp [-14.83,+1.94] |

The support pass makes this stronger than the raw-cardinality estimand.
Observable frozen score density does not explain February -> March; the
negative favorable-gross shift remains statistically separated from zero in
the day-block interval.  Net uncertainty still crosses zero, so this is not
a policy or market-regime conclusion.  It does, however, focus the next
execution model on conditional favorable-payoff scale and adverse conversion,
not universe size or another rank-density correction.

The January readiness artifact's original “crowding” terminology is
superseded by
`january_canonical_crowding_readiness_20260730_v1_correction`, manifest
`3eca1030079b005cbb0db6be5066149ed9f9c268a26267dad3ead06a953fa317`.
January is not required merely to match a q2 asset-count segment.  Its
separate readiness blockers remain: no canonical January base-score stream
and no current-spread exact-policy H12 economics; the historical soft-base /
1m-100bps lineage may not be pooled or calibration-bridged.

### Bounded short-side favorable/adverse conversion repair

The strict feature audit is
`short_conditional_payoff_readiness_20260730_v1`, manifest
`d438082b83033d2fb28c29e7953784917135c6d5092fa733f26d61a96eefb55c`.
It joins 140,682 exact March--April identities one-to-one across the
base/residual/direct score triplet and strict-OOF peak, future-slope and MAE
mixture ledgers.  The smallest admissible short heads are:

- `P(net > 0)`: score triplet;
- conditional favorable magnitude: score triplet + expected peak + future
  slope;
- conditional adverse severity: score triplet + expected MAE severity.

Realised MFE/MAE, timing, target-price, wait and mapped-outcome fields are
forbidden.  Compact causal contexts are joinable but excluded from this
smallest baseline and require a separately predeclared arm.

`short_conversion_ablation_readiness_20260730_v1`, manifest
`9dccdd4d4766e3f5e478f616b0a18ba95d7c1819daf5a8692ea3f4dc59152999`,
finds 31,942 positive and 57,081 nonpositive March short rows over 31 day
blocks; the frozen global top-decile contains 2,257/2,499 over all 31 days.
Support is ample.  March is one outer fold, so it cannot be both selection
and confirmation.  The executed experiment uses inner chronological March
OOF only for selection and April as the untouched confirmation.

The authoritative artifact chain is:

- ablation v2 manifest
  `7397e8de679f7222c55c12381c1c4601506af3e90a4e00e3dfd0011c494f3a35`;
- fixed-book day-block supplement
  `d78ca453ee119ae07d4c224579c7d422472a0ab7e3b9b3b6e2067c81f232d8ab`;
- final test-binding wrapper
  `bounded_short_conditional_payoff_ablation_20260730_v3_final_seal`,
  manifest
  `6d63562692fa09c805fe0bcdf96a5c3c124d19af94da7cf11661ead11ac370d2`,
  seal
  `4534480977a88da90a05f1c723a38aa7c5dd663ff401e72c207aa54e9923eb0c`.

Long is fixed to bit-identical robust decomposition.  Short uses fixed
geometry for `P(net>0)`, conditional favorable payoff and conditional loss.
The eight predeclared arms are score-only, +peak/slope, +predicted adverse
severity and all supports at tail weights `{1,2}`.  March OOF freezes
+peak/slope, tail 2, at -53.20 bps versus -54.00 for the score-only control.

| April pooled-global | Raw | Frozen-map expected |
|---|---:|---:|
| Top 1% | +31.32 bps | +31.85 bps |
| Top 5% | -10.32 | -13.65 |
| Top 10% | -24.55 | -25.19 |
| Top 20% | -54.42 | -55.47 |

The top-1 book is raw-tie-safe but statistically and temporally unstable.
Its fixed-book 2,000-draw UTC-day interval is [-52.27,+107.71] bps across 29
days; latest-week top-1 is -86.24 bps.  It contains 499 long rows (72.0%) at
+62.33 bps and 194 short rows at -48.43 bps.  Top-10's interval is
[-59.81,+14.70] bps; long is -4.27 and short -46.40 bps.  Latest raw/mapped
top-10 is -85.47/-86.08 bps.

The short model predicts top-10 `P(net>0)=0.519`, conditional favorable
+201 bps and adverse loss 230 bps, hence -3 bps decomposed EV; realised
short net is -46.4 bps.  It improves relative ranking modestly but misses the
new response level.  Mapped top-10 also has a 30.5%-of-book plateau, and raw
bias/ECE is +27.79/+32.91 bps.  Every economics, latest, side, tie and
calibration gate fails.  No replay ran.

The density attribution must not be misquoted as uniquely short-side:
conditional favorable-gross loss is -45.84 bps
[-75.90,-6.68] long and -53.53 [-110.53,+27.11] short.  The point estimate is
worse short, but only long is clearly separated from zero.  The short repair
targets the current selected-book failure, not a proven universal
February--March short-only mechanism.

That downstream mapping test is now complete.  Weak pooled support is
unmapped warm-up with `NaN`; it is never a tradable raw or zero-EV fallback.
Weak side support retains the pooled anchor with an exact zero residual.

### Sealed short-winner causal recent-EV mapping

Use only `short_winner_causal_recent_ev_mapping_20260730_v5`, manifest SHA
`44a1d4602e5ef1943f402f36f72e3b3fa8eea437f8150e530471aeb27a77606a`.
Versions v2--v4 are explicitly invalidated as incomplete or non-parity
lineages and redirect to v5.

The previously missing candidate-score stream is now materialized:

- 33,408 globally unique March candidate-head OOF rows.  The candidate IDs
  and raw scores are bit-identical to the frozen winner reconstruction
  (`max |delta| = 0`), and raw global top-10 reproduces
  -53.1962560169 bps exactly.
- 69,258 April frozen-forward rows, 34,629 per side.  Raw scores are
  bit-identical to the sealed winner confirmation ledger; there is no April
  ranker refit.
- Every row records the stable candidate key, validation range, model/train
  cutoff, latest resolved training label, score availability, upstream
  outer-OOF provenance and candidate-head OOF/forward status.

The mapper configuration was precommitted without looking at its economics:
2,000 pooled references, 1,000 side references and shrinkage lambda 500.
For each UTC-day snapshot it fits a pooled isotonic map on
`snapshot - 21d <= label_end < snapshot`, then adds
`n_side/(n_side+500) * (side_isotonic - pooled_isotonic)`.  Prior resolved
April outcomes enter only after their label end.  All 31 snapshots have zero
evaluation/reference ID overlap, finite maps and full support.  Pooled
reference support grows 33,408 -> 48,952; short support grows
12,672 -> 24,476.

| April pooled-global | Raw | Frozen-map expected | Causal pooled | Causal pooled + side |
|---|---:|---:|---:|---:|
| Top 1% | +31.32 bps | +31.85 | +23.54 | +8.66 |
| Top 5% | -10.32 | -13.65 | -50.85 | -21.00 |
| Top 10% | -24.55 | -25.19 | -31.16 | -39.87 |
| Top 20% | -54.42 | -55.47 | -66.50 | -64.79 |
| Latest decision-week top 10% | -80.48 | -74.94 | -88.13 | -100.50 |

At top 10%, the side-shrunk map is 45.0% long / 55.0% short at
-6.34 / -67.12 bps.  Its cutoff-tie fraction is 2.48%, maximum asset share
3.13%, prediction bias +12.77 bps and ECE 31.13 bps.  The fixed selected-book
equal-UTC-day interval is [-65.04,-18.33] bps.  Its positive top-1 result is
not promotion evidence: latest decision-week is -68.80 bps and the day-block
interval is [-41.59,+76.40].

Identical-ID top-10 controls are -33.94 bps base, -24.32 residual and
-93.24 direct-q25.  Thus causal mapping does not repair the short response
shift and the side residual makes the pooled result worse.  Causality,
coverage, tie, allocation, concentration and bias gates pass; expected
economics, latest week, both-side positivity, ECE and control-improvement
gates fail.  No simple-policy or portfolio replay ran.

Do not retune this mapper or another additive peak/slope/MAE weight grid.
The next executable branch is a bounded conversion-residual learner with
older same-lineage exact-12h OOF support and compact causal transition
contexts.  It must ablate opportunity, favorable magnitude, adverse
severity, exit mixture, clean versus competing-risk probabilities and
retained score-by-transition interactions on one pooled-global post-map
tail.  April is no longer an untouched final test, so any promotion requires
a new frozen forward block.  Timing, MAE, target-price and wait actions remain
outside this score, and portfolio replay remains blocked.

### Materialized conversion-residual research input

The exact input is now sealed at
`v5_conversion_residual_input_20260730_v1`, manifest SHA
`23c54eb43447ca826d527a9e0b4d3ecfacfb285e6c098108a3570a284a856bd5`.
Its 102,666 unique identities comprise 33,408 March candidate-head OOF rows
and 69,258 April frozen-forward rows.  Side totals are 55,365 long and 47,301
short; April itself is 34,629 per side.

Join identity is `candidate_id + side_name`, with UTC `__ts__` equality as a
hard assertion.  Do not join raw symbol spelling: v5 normalizes symbols such
as `1INCH_USD:USD`, while the exact label source retains
`1INCH/USD:USD`; the stable candidate ID preserves the correct raw identity.

A read-only independent rebuild verifies:

- exact parity with v5 raw/map scores and the canonical base/residual score
  sources;
- strict OOF residual status on every included row;
- exact 12-hour decision/label-end alignment and
  `gross - one cost = net` on every identity;
- bit parity with strict OOF peak, fixed-slope and predicted-MAE sources; and
- complete causal compact context after recomputing side-hour base-score
  deciles on the full canonical population before selecting v5 rows.

The baseline contract has 33 finite inputs: raw/base/residual/direct scores,
meaningful-MFE probability and conditional/expected peak, fixed future slope,
five market-state levels, ten 3h/12h core deltas, eight regime composites,
side sign and frozen score decile.  Four predicted-MAE mixture fields are a
separate optional adverse-risk ablation.  Targets, realised paths, exits,
timing, bars-before-trough, target-price/wait actions, mapped coordinates,
reference counts/cutoffs and universe cardinality are excluded from model
features.  The cohort-context transform is not deployable until inference
implements the identical full-population decile aggregation.

The history audit narrows “use older data” to a real materialization task.
February base and exact labels exist, but residual is a non-OOF passthrough
warm-up and the v5 candidate-head/strict auxiliary stream is absent.  January
lacks both the canonical score stream and compatible current-spread deployed
exact-12h labels.  Broader historical soft-base, old55 and hourly/no-spread
artifacts are forbidden bridges.  They must not be pooled merely to increase
row count.

Run the bounded feature/target grid on March development and April only as a
rediagnostic: score triplet; +peak/slope; +core levels; +core transitions;
+regime composites; all compact; and optional predicted-MAE risk.  Compare
direct residual, favorable/adverse hurdle and clean/competing-risk targets,
then apply the identical causal pooled-global map.  No result can be promoted
from April again.  In parallel, extend genuine history by materializing a
canonical January base stream and compatible exact policy labels, followed
by February strict residual and auxiliary OOF predictions; final confirmation
must use a new forward block.

## 2026-07-30 — full-history regime and transition workstream

This workstream now has complete strict exact-minute 2024 coverage rather
than a sampled or incomplete proxy.  All 141 required products pass the
product-bound verifier (43,660,200 / 43,660,200 required minutes), and all
190,398 2024 candidates have exact 12-hour policy, timing, path and
multitask auxiliary labels.  The reconstructed base-plus-residual calendar
now contains 360,012 rows across every month from January 2022 through
December 2024.

On the complete canonical denominator currently available (39 months and
168 weeks through 2026), rank IC is meaningfully positive in 32 / 39 months
(82.05%) and 127 / 168 weeks (75.60%).  Cost-aware EV is meaningfully
positive in 0 / 39 months and 1 / 168 weeks (0.60%).  This is decisive
evidence that alpha learnability and economic conversion are separate
problems.  The one complete robust-positive week is 2025-04-07; it is not a
promotion basis.  Missing canonical scoring windows in 2025-2026 still need
materialization and must not silently leave the calendar denominator.

The expanded failure diagnostic does not support a hard hand-written regime
gate.  No individual feature, covariance shift or standardized interaction
passes the combined multiple-testing, recurrence and distinguishing gates
on the enlarged 2024 population.  The strongest raw discriminator,
`range_climax_reversal`, has worst-versus-regular robust-z difference
-0.849, BH q 0.0199 and AUC 0.708, but only 0.25 recurrence and therefore
fails the recurrence gate.  Use soft learned context and uncertainty, not a
binary exclusion rule.

### Separate state and transition layers

Current market regime and transition state are not interchangeable:

- the regime layer estimates the current persistent market state;
- the transition layer estimates lifecycle and change dynamics
  (`stable_origin`, `precondition`, `approach`, `acceleration`, `trigger`,
  `active_dislocation`, `confirmation`, `settled`, `failed_transition`,
  `reversal`, `stable_destination`);
- each layer must have its own OOF probabilities, entropy, margin, OOD
  measure, training cutoff, availability timestamp and provenance;
- the mandatory matched arms are baseline, regime only, transition only and
  regime plus transition.

The transition catalogue currently contains 157 transition episodes and 157
matched stable controls.  Its purged event-group OOF stable-versus-transition
classifier has ROC-AUC 0.874, average precision 0.871 and Brier 0.149.
Conditional morphology accuracy is 68.8% on only 32 supported OOF events, so
morphology types remain research-only and fold-local until recurrence,
alignment and support improve.

The exact path-geometry diagnostic now covers 309,132 compatible candidates
and 20,510 decision hours from 2022-08-30 through 2024.  It reports regime
state and transition phase as separate taxonomies, separately by side, with
decision-hour-clustered intervals for opportunity, peak MFE, pre-opportunity
MAE, time-to-MFE, future slope, timeout, exit conversion and frozen
counterfactual net EV.  Transition lifecycle is materially asymmetric: for
example long confirmation has -1.14% mean net EV (95% interval
[-1.39%,-0.90%]) versus long failed-transition -2.49%
([-2.85%,-2.13%]); short approach is -1.19%
([-1.36%,-1.02%]) versus short confirmation -2.45%.  Current regime state
is less sharply separated.  These are descriptive raw-candidate outcomes:
transition phase is ex-post and must not enter a decision-time model directly.
Only causal transition probabilities/features with availability at the
decision may do so.

The causal multiview panel has 33,907 hourly rows and exact
1/3/6/12/24/48/72/168-hour views (plus conditional 15-minute views), with
distribution shifts, dynamics, realized-volatility/vol-of-vol, dependence
and covariance geometry.  The v2 panel now includes a separately attested
exact-timestamp liquidity enrichment: 228 compatible product files, 41
observed source fields, and 656 causal multiview liquidity fields derived
from real Amihud, volume, spread, depth and cross-sectional stress measures.
Market-wide fields and cross-product aggregates remain distinct; there is no
as-of fill across missing timestamps.  The v2 panel contains 14,536 output
features in total.  Thin-compression and fabricated liquidity proxies remain
forbidden.

### Interaction discovery contract

Add regime-by-feature and transition-by-feature interactions with two
independent discovery routes:

1. subsampled tree-derived SHAP interaction values, using an actual
   tree-explainer route and never a main-effect fallback disguised as an
   interaction;
2. regime-conditional permutation importance, computed inside supported
   state/phase strata with fold-local predictors and targets.

Selection and redundancy pruning are training-fold-only, family-balanced and
horizon-preserving.  Regime and transition interaction rankings remain
separate, with a third explicitly combined arm.  Promotion requires stable
sign/rank, sufficient state/phase support, latest-month coverage, monthly and
weekly Q10 economics, and incremental value over both single-layer arms.

The enriched fold-local selection is materialized at
`fold_local_multiview_selection_2022_2026_20260730_v3`.  It has 12 expanding
quarterly folds and 24,379 OOF hours for each task.  Each fold selects 28
fields separately for regime and transition, including eight real-liquidity
fields.  The two heads share 20.83 / 28 fields on average but retain 7.17
task-specific fields each; this is related context, not interchangeable
state.  Cross-asset market-spread changes at 24/48/72/168 hours are the most
stable liquidity selections.  Selection frequency is not economic proof.

Candidate-level soft layers are also materialized.  Full 2024 has
190,398 / 190,398 exact candidates across 12 monthly folds.  The earliest
fail-closed older extension begins 2023-04-01 (after transition targets are
actually resolved) and covers 293,828 candidates through 2024 across 21
monthly folds.  Each row has three regime-state probabilities, seven
transition-phase probabilities and an independent transition-active
probability.  Both simplexes close, both availability timestamps precede the
decision, and the regime and transition training/provenance fields are
independent.

The bounded empirical interaction probe is materialized at
`oof_regime_transition_interactions_2023q4_2024q1_20260730_v2`.  It fits an
actual LightGBM tree on 12,000 deterministic October-December 2023 rows,
evaluates 6,000 deterministic January-March 2024 rows and computes the actual
TreeSHAP interaction tensor on a 512-row held subsample.  The largest regime
interaction is `regime_state_p__2 x score_residual_expected_ev`
(mean absolute SHAP interaction 0.000271).  The leading transition
interactions are `settled_destination x residual score` (0.000199),
transition margin x residual score (0.000142), and transition probability x
12h/6h market-spread IQR (about 0.000109/0.000104).  Conditional permutation
importance confirms the residual score in all three months
(regime-conditioned delta-MSE 2.65e-05; transition-conditioned 1.99e-05;
positive-month fraction 1.0 for both).  Spread-context importance is positive
in only two of three months and has negative monthly Q10, so it remains an
ablation candidate rather than a gate.

The full-2024 matched four-arm economic ablation is sealed at
`full2024_matched_regime_transition_economic_ablation_20260730_v2`.
All arms use the identical 190,398 candidates and a causal trailing-90-day
Ridge EV map by month; January is the same raw-score cold start.  Selection
is exactly one pooled global top 10% over the full candidate population.
Baseline has alpha IC 0.12293, execution IC 0.06593 and -121.61 bps net EV.
Regime-only degrades to -124.84 bps and execution IC 0.06467.  Transition-only
is closest at -120.71 bps (+0.90 bps versus baseline) but execution IC falls
to 0.06525.  The combined layer is -122.35 bps with execution IC 0.06371.
No challenger meets the precommitted >=5 bps net uplift, positive execution
IC uplift and non-worse weekly-Q10 gates.  This rejects these current direct
context additions; it does not reject regime/transition-conditioned
residual trust, interaction features or separate clean/competing-risk heads.
No portfolio replay ran.

Separate clean-opportunity and adverse-competing-risk OOF heads are
materialized at
`clean_competing_risk_probability_oof_2023_2024_20260730_v1`.  The exact
common panel has 293,828 rows; 225,726 prediction rows cover five expanding
quarterly folds, four architectures, two heads and side-local fits.  The
adverse head is consistently learnable: mean global AUC is 0.607 baseline,
0.606 regime-only, 0.606 transition-only and 0.605 combined, with AP about
0.659-0.660 and Brier about 0.232-0.233.  The clean head remains weak:
baseline mean AUC 0.529, regime-only 0.536 (best), transition-only 0.526 and
combined 0.532; regime-only ranges from 0.496 to 0.563 by fold.

At pooled global top 10%, combining the two probabilities as
`p_clean * (1-p_adverse)` improves the economic tail relative to clean-only
for every architecture, but none becomes positive.  Regime-only is best:
-103.59 bps joint versus -120.54 bps clean-only.  Baseline is -108.02 versus
-125.08; transition-only -112.25 versus -132.46; combined -107.75 versus
-125.71.  Treat the adverse probability as useful risk context inside the
next residual/trust learner, not as an independently tradable rank.  The
frozen historical labels make this artifact research-only/non-promotable
even though its new predictions are strictly chronological OOF.

The interaction-conditioned residual-trust learner is sealed at
`interaction_conditioned_residual_trust_oof_2023q4_2024_20260730_v1`.
It evaluates 190,398 matched 2024 candidates across four chronological OOF
quarters.  Baseline remains least bad: alpha IC 0.13031, execution IC
0.08989 and pooled-global post-map top-10 net EV -109.70 bps.  Regime-only
is -111.93 bps; regime plus transition -116.44; combined plus adverse risk
-117.53; combined plus clean probability -120.77; transition-only -123.37.
No context/risk arm is incremental in this fixed low-capacity residual-trust
architecture.  The explicit additive GAM calibration ablation is the next
bounded model test; direct routing or state gates remain forbidden.

That GAM test is now sealed at
`oof_gam_regime_calibrator_2024q2q4_20260730_v1`.  It uses additive cubic
splines plus ridge only—no unrestricted interactions—and strictly prior OOF
scores/labels, so Q1 is training-only and the matched evaluation is 154,072
Q2-Q4 candidates.  The uncalibrated baseline remains best at -104.99 bps
top-10 net EV and execution IC 0.09550.  Baseline spline is -107.78 bps /
0.09364.  The best contextual GAM is combined plus adverse risk at
-112.17 bps / 0.08414.  It materially improves decile calibration MAE versus
its corresponding uncalibrated contextual arm (0.000956 versus 0.003232) but
does not improve ranking or economics.  Regime, transition and combined GAMs
are all worse.  The requested GAM path is therefore tested and rejected in
this form.

The recurring-transition taxonomy audit is sealed at
`recurring_transition_taxonomy_stability_20260730_v1`.  OOF
stable-versus-transition LightGBM has AUC 0.8738, AP 0.8705 and Brier 0.1486
on 314 events/controls, but weakens in 2022 (AUC 0.800) and 2026 (0.785).
The morphology classifier agrees with fold-local GMM labels on 79.5% of 156
non-abstained events.  Twelve fold-local components recur across at least two
eras, but cross-fold semantic alignment is not identified; no global
morphology type or policy routing may be claimed.  GMM, LightGBM and BOCPD
have genuine runs.  KMeans and AE-GMM have limited-scope runs.  HDBSCAN,
categorical HMM and Bayesian Rule List were implementation/dependency-only
at this checkpoint; a dependency-free BRL OOF challenger is now required.

The follow-up support bound is sealed at
`transition_morphology_support_bound_20260730_v1`.  It audits 157 unique OOF
events across five eras, 14 fold-local components, 43 support cells and 14
component-fold calibration/abstention rows.  The existing morphology fold
plan contains zero explicit `role=test` rows, so it cannot substantiate a
held-out-era semantic-matching claim.  No event is duplicated to inflate
support, and the artifact deliberately names zero global transition types
and creates zero gates.  The morphology requirement therefore remains
honestly incomplete until a genuine held-out-era fold plan is materialized.

That missing fold plan is now sealed at
`leave_one_era_out_transition_morphology_20260730_v1`.  Every one of the 157
unique events is assigned exactly once by a three-component GMM trained
without its calendar era: support is 6/58/41/36/16 events for
2022/2023/2024/2025/2026, with zero skipped events.  Posterior probability,
entropy, margin and abstention are retained.  This fixes the held-out-row
defect, but does not by itself close the morphology requirement: component
descriptors remain fold-local, and train-only prototype matching plus
held-era predictive outcome evidence have not yet established global
semantic recurrence.  Global type names and policy gates remain disabled.

Post-event evaluation labels are conservatively bound at
`transition_event_outcome_binding_20260730_v1`.  It produces 118 exact
event-by-source slices from the 2022--2023, 2024 and canonical 2025 sources,
with all 157 events audited and unmatched reasons retained.  Economics from
incompatible source lineages are not pooled.  The subsequent
`nested_morphology_increment_readiness_20260730_v1` proves the remaining
joint test is not currently identified: the assignments retain no train-only
prototype descriptor vectors/matching matrix, the outcome binding lacks the
current-regime/transition-probability baseline and matched 2026 rows, and
support is only 64/41/13 slices for 2022--23/2024/2025.  This is a sealed
statistical-insufficiency result, not permission to match numeric fold IDs.

That BRL gap is now closed in `transition_pattern_catalogue_20260730_v6`.
The native backend is an ordered low-cardinality rule list with explicit
Beta-Binomial posterior probabilities and length/width MAP penalties; it is
honestly labelled `native_beta_binomial_map`, not MCMC BRL.  It uses the
exact same 314 purged OOF event/control rows and five folds as LightGBM.
BRL reaches AUC 0.5999, AP 0.5710 and Brier 0.2646, far below LightGBM
0.8738/0.8705/0.1486.  It is an interpretable negative challenger and must
not become a transition gate or regime-state substitute.

The authoritative signed requirement audit is
`regime_objective_completion_audit_20260730_v6`: 14 of 17 requirements are
proved and 3 remain incomplete.  It references the current catalogue v6,
early supplement v3, all-era recurrence v1 and causal inventory v5, and
contains no stale requests to rerun completed work.  The three open
requirements are cross-era transition-morphology validation with adequate
support, an accepted all-era unsupervised-regime economic solution, and
stable regime-category EV.  V6 also records the negative morphology,
unsupervised-economic, alpha-to-EV and held-out category-stability follow-ups.
This explicitly prevents treating the objective as complete or promoting a
regime-aware policy.

### Final regime/transition evaluation contract

The decisive model assessment is now a strict calendar holdout.  Regime and
transition models are trained and frozen using only causal data/labels
available by 2025-12-31, then assessed once on untouched 2026.  Regime state
and transition state remain separate prediction layers; evaluate
regime-only, transition-only and combined sidecars as three distinct arms.
No 2026 row may influence feature selection, HPO, state semantics,
probability calibration, thresholds or model fitting.

The primary comparable lineage begins on 2022-08-30 and supplies the
2022--2025 training surface.  The separate January--August 2022 inverse-PI
population may be added only if an explicit harmonized feature-definition
contract is proved.  Its local state IDs, probabilities and economics must
never be pooled with or relabelled as the later PF taxonomy.  The 2026
assessment must report probability discrimination/calibration, uncertainty,
state/transition stability, long/short path geometry and exact economic
attribution by month.  Any trading diagnostic still uses one pooled global
top 10% after the arm's own causal EV map; ex-post transition phases remain
attribution-only.

The first strict transition-only baseline is sealed at
`strict_forward_transition_evaluation_20260730_v1`.  It trains on 29,268
resolved hourly rows from 2022-08-30 through 2025-12-31 and scores 4,627
untouched 2026 hours through 12 July.  Overall active-transition AUC is
0.572, AP 0.0209 at 1.43% prevalence, Brier 0.0146 and ECE 0.0058.
Performance is unstable by month: AUC is 0.853 in January and 0.900 in June,
but about 0.52--0.54 in February/March and undefined in April/May/July
because those months contain no positive transition events.  Lifecycle
classification is weak (macro-F1 0.096).  On the fixed pooled-global
top-10 economic book, every transition-risk decile is net-negative.  This v1
baseline is not promotable; a train-only blocked-CV/HPO/class-imbalance
challenger is required before concluding the transition model itself is
exhausted.

That challenger is sealed at
`strict_forward_transition_challenger_20260730_v2`.  Blocked 2022--2025
inner CV selects LightGBM on all causal transition fields, positive weight 5
and train-only Platt calibration.  It does not transfer to untouched 2026:
AP falls from 0.0209 to 0.0177 and AUC from 0.572 to 0.522.  Brier/ECE improve
to 0.0141/0.0032, but lifecycle macro-F1 slips to 0.0935.  Global-top-10 net
also worsens from -97.56 to -99.02 bps (May -117.13, June -71.79, partial
July -108.13).  V2 is rejected.  Better calibration without discrimination
or economics is not sufficient for promotion.

The corresponding failure audit is sealed at
`strict_transition_nontransfer_diagnostic_20260730_v2`.  April, May and
partial July's zero active-transition targets are genuine resolved outcomes,
not coverage gaps.  The 2026 lifecycle mix shifts toward 2,481
`stable_destination` hours and only 66 `active_dislocation` hours.  Large
train-to-test changes include BTC OI-dominance ratios and a covariance sign
flip between breadth-dispersion short/long and downside-breadth short/long
(+0.152 to -0.211).  V1/V2 monthly rankings still correlate about 0.74--0.86,
so the v2 failure is primarily prior/regime non-transfer rather than a wholly
different ranking.  The next strict arms are train-only 1/3/6/12-hour onset
heads, cause-specific competing-risk lifecycle heads, calibrated versus raw
probabilities, and stable-family feature robustness.

The first strict regime-only baseline is sealed at
`strict_forward_regime_only_2022aug_2025_to_2026_20260730_v1`.  It trains on
the same calendar cutoff and scores the same 4,627 untouched 2026 hours, but
uses the older multiview v1 panel and a pure variance screen.  It is not an
acceptable persistent-regime solution: posterior entropy is effectively
zero, all six states appear every month, and hourly state changes run about
40%--54%.  Its selected semantics are dominated by extreme-scale covariance
and robust-z transforms.  Treat v1 as a pathology-revealing baseline only.
The authoritative v2 must use enriched multiview v2, balanced coverage of
volatility/liquidity/dependence/distribution families, robust scaling and a
train-only causal persistence filter; it must report raw versus filtered
dwell/switch behavior without tuning on 2026.

The separate early-2022 supplement is sealed at
`early_2022_inverse_pi_regime_supplement_20260730_v3`.  It covers January
through 30 August with 57,840 exact inverse-PI candidates and 5,784 hourly
OOF state/transition rows.  States use eight leave-month-out GMM fits and the
transition layer is trained separately with onset/active/decay/stable
lifecycle outputs.  The transition classifier remains weak: monthly AUC
ranges from 0.489 to 0.543, so it must not control policy.  January--July
evaluation uses the existing block-OOF raw score; August uses a separately
labelled leave-month-out HGB raw score.  All months use exact 12-hour
economics and one pooled global cross-side top 10%.  Its bridge to the later
PF lineage permits feature-family comparison only and explicitly forbids
treating local state IDs as equivalent.

The August gap was genuinely backfilled rather than accepted as missing.
Five inverse-PI contracts received 208,800 exact one-minute rows with 100%
coverage.  The causal feature, exact-path and label chain then sealed 6,960
August candidates, each with a complete 720-minute path.  The separate
supplement now ends exactly where the later PF ledger begins on 30 August;
there is no remaining chronological date gap, although the two populations
remain non-pooled and non-equivalent.

The all-era worst-period explanatory artifact is sealed at
`all_era_worst_period_multiview_recurrence_20260730_v1`.  Within each
comparable lineage, worst complete weeks are the bottom 25% by net economics.
The diagnostic tests a deterministic causal 64-field compact multiview
surface, per-era BH-controlled feature shifts, bounded covariance and
standardized-product interactions, and regime-conditional permutation
importance.  Across the three comparable tested eras, no effect recurs with
the same direction in at least two separated eras.  Two covariance effects
survive only in the long 2022-August-through-2024 era:
`breakout_confirmation x BTC_decoupling` and
`breakout_efficiency x breakout_retention`.  They are not recurrent and
cannot become gates.  Early 2022 is explicitly `missing_multiview_evidence`,
not a negative result.  The artifact therefore closes the requested bounded
all-era diagnostic, while also proving that the present feature surface does
not yet identify a robust shared failure driver.

The causal feature inventory is sealed at
`causal_regime_feature_inventory_20260730_v5`; earlier v1--v4 revisions are
superseded.  It inventories 16,618 actual fields across the regime ledger,
multiview panel, 249-file historical feature store, separate early-2022
inverse supplement and historical model-health source.  It distinguishes
causal observables, source-unavailable fields, unselected fields and 23
outcome/state fields that are forbidden as inputs.  Seventy-two actual
multiview units were selected across folds.  Selection is concentrated in
liquidity/spread/depth (45 fields; 218 regime and 225 transition fold
selections) and funding/OI/liquidation (17; 76/82), followed by volatility
(8; 29/28) and path geometry (2; 13/1).  The inventory also records 83 exact
source-unavailable liquidity asset-field combinations and seven plausible
missing observable/composite families.  These include executable multi-level
impact/depth, cross-venue funding/OI/liquidation dispersion, shrunk
cross-asset factor-residual networks, volatility-conditional trend
persistence, causal pre-entry fragility, causal score/population drift, and
implied-volatility/skew or basis term structure.  This is an inventory and
research backlog, not feature promotion.

The alpha-to-execution failure decomposition is sealed at
`alpha_execution_ev_gap_diagnostic_20260730_v1`.  It evaluates 619,694 exact
candidates and 61,979 monthly pooled-global top-10 selections.  The primary
failure is poor clean-opportunity capture under an approximately 100 bps
cost hurdle, not sparse recent mapping support.  Selected opportunity rates
are 36.3% for frozen PF 2022--2024, 45.8% for canonical 2025 and 35.2% for
current 2026.  In 2025/2026, selected non-opportunities lose about 258/245
bps while opportunities earn about 191/121 bps.  Positive alpha IC therefore
does not imply tradable execution economics: frozen 2022--2024 and canonical
2025 both have alpha IC about 0.226 while net EV is -92.9 and -52.4 bps.

The recent causal map has ample reference support (median about 34.8k
selected reference rows), but it is not reliably rank-preserving.  Monthly
raw-versus-mapped top-10 overlap ranges from 17% to 75%, and mapping changes
selected net EV by +12.8 to -44.0 bps.  The latest strict months remain
negative despite positive alpha IC: May -64.1 bps and June -75.8 bps; July
is partial at -133.5 bps.  The next bounded tests are a causal cost-aware
opportunity hurdle before EV mapping and a map rank-preservation/shrinkage
sweep.  Regime/transition slices remain descriptive; this audit authorizes
no regime gate.

Those two bounded follow-ups are sealed at
`causal_opportunity_hurdle_mapping_ablation_20260730_v3`; v1/v2 are
superseded by the strict resolution-boundary fix.  Fit/HPO excludes
prior-month rows whose exact 12-hour outcome resolves at or after the
held-out month begins.  Every arm
uses its own causal map, identical candidate rules and one pooled global
monthly top 10%; weekly reporting preserves that monthly book rather than
reranking.  Baseline aggregate monthly net is -103.88 bps.  The opportunity
hurdle is effectively unchanged at -103.90 bps and often admits 100% of the
baseline.  Rank-preservation/support shrinkage improves aggregate net to
-94.05 bps, including April (-24.32 versus -42.49) and June (-97.39 versus
-109.23), but harms partial July (-160.45 versus -159.92).  Latest weekly
results are all negative.  Neither arm passes aggregate-plus-latest or
Q10/Q50 gates, so both are rejected and no portfolio replay ran.

The authoritative common-surface unsupervised economic ablation is
`unsupervised_economic_common_oof_20260730_v2`; v1 is superseded because it
included forbidden GMM posterior/risk-summary fields.  V2 scores 84,963
identical candidates across eight chronological weekly folds from 20 May
through 10 July 2026.  Every side-local arm uses only already-resolved labels,
its own trailing-21-day causal EV map, and one pooled global post-map top 10%.
GMM is geometry-only (OOD, Mahalanobis and expected Mahalanobis); DAE,
failure-destination probability, transition probability and combined
failure-first context are separate arms.  Timing, MAE, target-price and wait
fields are excluded.

The baseline is +11.32 bps aggregate net, but is not promotable: partial July
is -54.95 bps, only 50% of weeks and 33% of months are positive, weekly Q10
is -25.44 bps and monthly Q10 is -44.52 bps.  Every unsupervised arm is worse
in aggregate: GMM geometry -2.69 bps, DAE -23.86, GMM+DAE -23.71,
failure-destination -18.09, transition-only -2.26 and combined failure-first
-8.69.  Transition-only and combined failure improve partial July by about
25 bps versus baseline, but remain near -30 bps and aggregate-negative.
Therefore no arm passes aggregate-plus-latest gates, no portfolio replay ran,
and this May--July slice does not satisfy an all-era or full-July claim.

The full-2024 extension is sealed separately at
`unsupervised_economic_2024_extension_20260730_v1`.  It uses 190,398
identical candidates and 12 monthly folds, with every train label resolved
before its evaluation month.  Fold-local GMM geometry and DAE representations
are rebuilt from a fixed 35-field causal raw subset; precomputed geometry,
GMM posterior/entropy/risk summaries and action-layer fields are excluded.
Baseline net is -136.07 bps.  DAE is the strongest ranking diagnostic at
-106.36 bps (+29.71 bps versus baseline; alpha IC 0.1249 and execution IC
0.1170), followed by GMM+DAE -106.42, failure-destination -114.78,
failure+transition -116.07, GMM geometry -119.28 and transition-only
-125.25.  Only DAE's December is positive (+12.37 bps); no arm passes
aggregate-plus-latest gates.

This 2024 result strengthens the regime-specific non-transfer diagnosis:
DAE materially improves ranking/loss in 2024 but degrades the exact
May--July 2026 common surface.  The 2024 economics are exact one-minute paths
but use a frozen current-spread historical counterfactual, so they remain
diagnostic and cannot support a global representation promotion or portfolio
replay.

The bounded cross-era DAE diagnosis is sealed at
`dae_cross_era_nontransfer_diagnosis_20260730_v2`.  Candidate-level
DAE-minus-baseline score association with net EV is positive in every 2024
month (median rank IC +0.099) but has median -0.017 across May--July 2026
(May -0.052, June +0.022, July -0.028).  A causal
representation-trust signal is not yet identified: the 2024 runner did not
retain its fold-local per-row DAE codes, and the sealed 2024/2026 surfaces
share no comparable causal raw market fields.  Historical precomputed
embedding fields can be compared only as proxies.  The next valid experiment
must persist matched fold-local codes, reconstruction error, representation
age/train support and OOD for both periods, then learn trust on prior months
and lock thresholds before later-month evaluation.  No DAE trust gate is
authorized by the current diagnostic.

Regime-category stability is audited at
`heldout_regime_category_economics_stability_20260730_v1`.  Selection is
fixed first as one pooled global top 10% per lineage-month, then decision-time
state, ex-post transition phase and state-by-phase are attributed as three
separate taxonomies.  Across 617,406 candidates, 61,751 are selected and
50,220 carry both context labels.  No incompatible economic cohorts are
pooled.  There are zero stable-good categories.  Five stable-poor categories
pass both-side, three-era, day-shrunk leave-era-out tests, but only inside the
non-promotable frozen 2022-August-through-2024 spread-counterfactual cohort:
decision-time state 0; ex-post `precondition`; ex-post
`stable_destination`; and the two corresponding state-0 combinations.
These are research diagnostics, not gates.  Exact USD-linear 2025--2026 has
only two comparable eras, below the three-era requirement; inverse-PI 2022 H1
has no compatible state/phase context.  Ex-post phase remains forbidden as a
model input or policy gate.

### Required execution order

1. Causally enrich the hourly panel with real liquidity/market-stress fields.
   **Complete:** `regime_liquidity_enrichment_2022_2026_20260730_v1` and
   `regime_multiview_panel_2022_2026_20260730_v2`.
2. Materialize compact fold-local regime and transition feature panels.
   **Complete:** `fold_local_multiview_selection_2022_2026_20260730_v3`.
3. Produce independent candidate-keyed OOF regime and transition probability
   ledgers with exact/as-of availability checks.
   **Complete through 2024:** full-2024 and 2023-04-through-2024 artifacts.
4. Run the SHAP-interaction and conditional-permutation discovery separately
   for regime and transition, then test the combined layer.
   **Complete as an explanatory diagnostic:** the bounded signed
   2023Q4-to-2024Q1 discovery and the all-era recurrence artifact are sealed.
   No same-direction effect recurs across separated comparable eras.
5. Join the probability/interaction ledgers to identical candidate rows and
   run baseline, regime-only, transition-only and combined base/residual/GAM
   ablations.
   **Complete for the bounded matched architectures:** the direct Ridge-map,
   interaction-conditioned residual-trust and additive GAM ablations have no
   gate winner.
   **Clean/competing-risk slice complete:** adverse risk is learnable and
   improves joint tails, while clean opportunity remains the bottleneck.
6. Evaluate one pooled global top-k after the recent causal EV map.  Do not
   rank per timestamp, side, regime or transition state.
7. Report IC, execution IC, gross/net EV, costs, hit rate, support, weekly and
   monthly Q10/Q50, positive-period fractions, worst periods, state/phase
   stability and long/short path geometry.
8. Replay the frozen winner through concurrency, exposure and asset limits
   only if the matched economic and stability gates pass.

Timing, MAE, target-price and wait recommendations remain a separate action
layer above the score.  Nothing in this regime workstream authorizes those
heads to enter the execution-EV score directly.

### Remaining calendar gaps are upstream score/label work, not raw-data gaps

The signed readiness artifact
`canonical_base_residual_gap_readiness_20260730_v2` verifies all 14 requested
gap months (Jan-Feb 2025, May-Dec 2025, Jan-Apr 2026).  Every month has both
authoritative side shards, valid candidate/path rows, unique candidate IDs
and 100 numeric raw source features.  None is yet fully scoreable because
each lacks canonical base OOF, canonical residual OOF and candidate-local
exact-12h execution economics.  February 2025 has 64,512 verified base-only
warm-up rows, but residual OOF is false and it cannot enter the canonical
calendar.  Do not substitute comparator, pooled or historical score
lineages; execute the three missing stages per month.

The bounded January--February verification is sealed at
`canonical_janfeb2025_gap_closure_20260730_v1`.  February has a 64,512-row
canonical base-only top-40 sidecar, 32,256 rows per side, joined to the
accepted exact-1m deployed-policy 12-hour economics.  Candidate identity,
base score and source hashes agree exactly with the accepted February base
OOF and top-40 artifacts, and gross minus cost equals net.  It remains
explicitly `residual_is_oof=false`: the source residual is a base-passthrough
warm-up, so February cannot enter stack or policy evaluation.  January
remains blocked because the accepted canonical base OOF begins in February
and there is no canonical path-input join.  Native 12-hour labels were not
substituted.  This closes provenance verification, not the missing canonical
stages.

The separately versioned May--June continuation is sealed at
`mayjun2025_canonical_base_continuation_20260730_v1` and
`mayjun2025_canonical_residual_continuation_20260730_v1`.  Both contain
87,840 strict point-in-time OOF rows with exact one-minute 12-hour economics
and pre-month label-resolution cutoffs.  Accepted side-local contracts are
frozen: long trial_141 with 31 features and short trial_084 with 8; neither
feature selection nor HPO is rerun.  This is common-30-universe continuation
evidence, not a replacement for accepted January--April artifacts.

Under one pooled global top 10%, calibrated base is -77.17 bps aggregate and
residual is -83.32 bps.  Residual slightly improves May (-76.83 versus
-80.71 bps) but materially worsens June (-94.40 versus -74.48 bps).
Neither continuation is promotable; timing and wait actions remain outside
the residual layer.

### Authoritative mapped-policy conversion-residual ablation

`v5_conversion_residual_ablation_20260730_v3` is the authoritative bounded
diagnostic (manifest SHA
`40c78dea764e3dc4fb9be9620b4302b8e1b3498acfab485f04138fe1bc04f388`).
V1 is invalidated because March selection used raw scores.  V2 corrected the
selection policy but is superseded by v3's complete development and forward
gate surface; v2 and v3 prediction ledgers are byte-identical.

Selection now matches the live contract exactly: configuration-specific
prior-resolved daily causal mapping, random-tie-expected mapped economics,
one pooled-global top 10%, and score-specific mapped controls.  Unsupported
warm-up rows receive no mapped value.  The 18,432-row March selection ledger
has 13,824 eligible rows (75%); 4,608 early rows remain unmapped because the
compatible short lineage does not provide enough prior support.

Mapped selection chooses the 23-field score + peak/slope + market-level +
regime direct-residual diagnostic leader.  It is not admissible:

- March aggregate / latest / worst top-10 net:
  `-133.21 / -127.93 / -149.08 bps`;
- March stability objective: `-165.73 bps`;
- March aggregate cutoff-tie fraction: `26.03%`, with the worst fold at
  `249.89%` of book size;
- April mapped expected / latest-week net:
  `-92.17 / -144.21 bps`;
- April long / short contribution:
  `-100.25 / -89.09 bps`, at 27.92% / 72.08% share;
- April bias / ECE: `31.82 / 85.16 bps`; and
- best identical-ID mapped control: residual at `-30.39 bps`.

The March target comparison also rejects positive hurdle and five-class
competing risk: their stability objectives are `-194.75` and `-233.30 bps`
versus `-165.73` for direct residual.  April hurdle and competing-risk point
estimates are less negative than the selected arm, but April is reused and
cannot alter the frozen March choice; both still lose to mapped residual.

All material economic, stability, coverage, calibration and mapped-control
gates fail.  No simple-policy or constrained portfolio replay runs.
Causal mapping changes the diagnostic winner but does not repair conversion.
Do not expand HPO.  First extend same-lineage short/January/February OOF
history, finish the fixed-cohort IC-to-EV waterfall, and reserve a new
forward block.  A tie-safe monotone mapper may be tested only as a
predeclared post-extension arm, never selected on reused April.

### Authoritative extended-history conversion-residual v4

The immediate same-lineage history repair is complete.  The sealed
`v5_early_short_oof_extension_20260730_v1` ledger adds 8,064 strict short OOF
candidate scores for March 13--19 and publishes 41,472 March rows, exactly
20,736 per side.  It uses the frozen B/peak-slope/tail2 candidate
configuration, has strict `training_label_resolved_max_utc <
validation_start_utc` at both new cutoffs, and preserves all 33,408 v5
overlap scores bit-for-bit.  This is development history, not new promotion
evidence; January and February canonical gaps remain explicit.

The joined successor is
`v5_conversion_residual_input_20260730_v2`:

- 110,730 unique candidate-side rows;
- 41,472 March OOF rows and 69,258 unchanged April diagnostic rows;
- 55,365 rows per side;
- 16,128 March rows before the calibration cutoff;
- 6,912 March 20--22 calibration rows; and
- 18,432 March 23--30 selection rows.

Every joined March identity has strict canonical base, residual, peak,
future-slope and MAE predictions, exact deployed-exit decision-plus-12-hour
labels, one explicit current-spread cost, and the approved market/regime/
transition context.  Timing, target-price and wait actions remain excluded.

`v5_conversion_residual_ablation_20260730_v4` is now the authoritative
bounded conversion diagnostic (manifest SHA
`9e2195f84322eb704d0cb9c244082a92c043db4c3ee4981b5145363fd076baf7`).
A fresh isolated reproduction generated byte-identical hashes for all 13
published output files.
It supersedes v3 because it removes v3's 75% mapping-coverage limitation.
For each of ten predeclared feature/target configurations, it independently:

1. fits side-local models on prior-resolved March 13--19 history;
2. emits 6,912 March 20--22 calibration OOF predictions;
3. excludes those calibration rows from feature/architecture selection;
4. builds the unchanged daily causal 21-day pooled isotonic plus shrunk-side
   map from prior-resolved score-specific predictions; and
5. evaluates one pooled-global, random-tie-expected top 10% over all 18,432
   March 23--30 selection rows.

All three selection folds have 100% mapped support.  The seven feature arms
are:

| Feature arm, direct residual | March mapped top 10% | Worst fold | Stability |
|---|---:|---:|---:|
| Scores only | -129.11 bps | -177.71 | -188.97 |
| + peak/slope | -130.70 | -185.83 | -220.19 |
| + market levels | -159.81 | -229.68 | -253.41 |
| + levels and transitions | -180.28 | -244.06 | -267.47 |
| + levels and regimes | -158.10 | -221.16 | -259.23 |
| All compact | -173.45 | -223.97 | -261.59 |
| All compact + optional MAE | -188.70 | -228.70 | -232.75 |

Thus the older 23-field v3 leader does not survive once every selection row
has legal score-specific mapping support.  The four raw score inputs are the
least-bad feature set.  Peak/slope, context, transitions, regimes and MAE all
degrade the predeclared March stability objective when directly concatenated
into this learner.

On that fixed four-score feature set, target comparison is:

| Target architecture | March mapped top 10% | Latest fold | Worst fold | Stability |
|---|---:|---:|---:|---:|
| Direct residual | -129.11 bps | -85.88 | -177.71 | -188.97 |
| Positive hurdle | -132.52 | -97.69 | -180.10 | **-185.47** |
| Five-class competing risk | -145.95 | -112.58 | -175.29 | -185.55 |

The positive hurdle wins only the stated mean-minus-volatility-plus-worst
stability formula, by 0.08 bps over competing risk.  It is not an economic
winner.  Its fold cutoff-tie fractions are 12.57%, 80.92% and 167.68% of
book size, despite an aggregate tie fraction of 4.12%.

April remains reused diagnostic evidence:

| April mapped top 10% | Expected net | Latest week |
|---|---:|---:|
| Positive hurdle | -56.40 bps | -95.75 |
| Competing risk | -54.76 | -102.14 |
| Direct residual, scores only | -61.33 | -119.87 |
| Residual control | **-31.43** | -89.57 |
| Base control | -48.16 | -52.96 |
| Frozen v5 control | -54.19 | -100.50 |
| Direct-q25 control | -79.69 | -135.72 |

The hurdle book is 23.27% long / 76.73% short, with -10.08 / -70.43 bps
side contributions.  Bias is 21.01 bps and ECE 37.62 bps.  It fails March
aggregate/latest/worst economics, fold tie safety, April aggregate/latest,
side-share, both-side, mapped-control and calibration gates.  April is also
not untouched.  No simple-policy or constrained-portfolio replay ran.

The integrated mapping/extension/input/ablation suite passes 15/15 focused
tests.  The next conversion experiment must not expand this HPO.  Continue
the mandatory base-IC/EV waterfall, matched 12-hour-label comparison and
regime-conditioned reliability/trust work.  Route timing, MAE, target-price
and wait recommendations through the separate action layer.  Supporting
peak/slope/regime signals may be retested through detached reliability or
interaction adapters, but their direct-concatenation ablations are now
negative.

### Corrected full-stop target and repaired full-base selection

The supporting-label audit found one exact target defect in
`v5_conversion_residual_input_20260730_v2`: `target_stop_exit` searched the
exit-reason string for `"stop"`, but the canonical deployed-policy value is
`"full_sl"`.  V2 therefore emitted zero stop targets despite 22,406 exact
full-stop rows.  `INVALIDATION.json` marks only that field invalid.

`v5_conversion_residual_input_20260730_v3` is the corrected sealed successor
(manifest SHA
`cac676ae44816fd1fead2c9c69d48893cfa0ca2ae881f5896719d65ad56f0a05`).
It derives stop and timeout targets from the canonical mutually exclusive
flags and asserts reason parity.  Counts are now 22,406 full stops and 30,471
timeouts.  The v4 conversion ablation is unaffected: the field was neither a
feature nor a fitted target, and its competing-risk classes were derived
directly from exact exit reason and net outcome.

The invalidated full-base v1 model-selection result is now repaired at
`canonical_full_base_opportunity_ablation_20260730_v2` (manifest SHA
`4e7c295467b14635abe527b63b18b5924fab887ac7aa43f68db3a3b19d6f8a26`).
The repair does not reuse invalid static mappers.  It hash-verifies the source
artifact, reuses only valid raw OOF bytes, performs feature/target/geometry
selection on raw random-tie-expected pooled-global top-10 economics, fits the
four previously missing April configurations per side, freezes all choices,
and only then applies the fixed causal daily 21-day recent-EV mapper.

The raw OOF selections are:

| Target | Arm | Geometry | Raw top-10 net |
|---|---|---|---:|
| Direct net | S0 | compact d4 | -52.41 bps |
| Direct net | S1+B | compact d4 | -65.78 |
| Hard 0-bps opportunity | S0 | compact d4 | -64.39 |
| Hard 0-bps opportunity | S1+B | fixed d5 | -69.28 |
| Hard 25-bps opportunity | S1+B | fixed d5 | -72.22 |
| Hard 25-bps opportunity | S0 | deep d6 | -73.48 |
| Existing soft target | S0 | compact d4 | -57.29 |
| Existing soft target | S1+B | compact d4 | -64.65 |

April is reused diagnostic evidence, never promotion evidence.  Under the
primary pooled recent-EV map:

| April global top 10% | Expected net | Latest week |
|---|---:|---:|
| Frozen base control | **-68.93 bps** | -87.30 |
| Hard-25 S0 / deep d6 | -85.72 | -93.37 |
| Soft S0 / compact d4 | -88.18 | -144.03 |
| Direct S0 / compact d4 | -88.99 | -106.64 |
| Hard-0 S0 / compact d4 | -95.11 | -115.49 |
| Direct S1+B / compact d4 | -108.77 | -121.24 |
| Soft S1+B / compact d4 | -120.61 | -131.08 |
| Hard-25 S1+B / fixed d5 | -133.54 | -171.66 |
| Hard-0 S1+B / fixed d5 | -137.28 | -144.99 |

Every arm has 100% April coverage and 31/31 legal/supported causal mapping
snapshots.  None beats the identically mapped base control.  Direct S0 is
86.39% short; soft S0 is 85.57% short.  S1+B frequently improves opportunity
precision while worsening exact net, so the side-specific frozen 31/8 inputs
still do not solve conditional payoff, loss severity or exit capture.
Side-residual mapping also fails to repair the economics.

No promotion or replay is authorized.  The repaired predictions are legal
support sidecars for the next bounded reliability architecture, not
standalone admission models.  The next materialization must join them to the
exact top-40 ATR triple-barrier labels and v3 conversion identities, then
test meaningful-MFE, clean-binary and three-way competing-risk support
separately.  Proper pre-exit capture and fixed severe-loss targets remain
missing; full-12-hour MFE-minus-gross is not a valid capture substitute
because it can include post-exit opportunity.  Timing, MAE, target-price and
wait actions remain downstream in the separate action layer.

### Sealed reliability input and mandatory IC-to-EV decoupling diagnosis

That materialization is complete at
`data_perp/artifacts/canonical_execution_reliability_input_20260730_v2`.
The sealed panel has 110,730 exact identities: 41,472 March rows, 69,258
April rows and 55,365 rows per side.  It joins the corrected v3 conversion
outcomes, all eight repaired full-base raw support sidecars, the four v4
scores, individual candidate context, five regime levels, ten 3h/12h
transitions, eight regime composites and the exact decision-to-12-hour ATR
triple-barrier shards.  All identities and UTC decision/end timestamps match;
all 44 default EV inputs are finite and complete; and exact
`gross - row_cost = net` has zero numerical error.

Target support is sufficient for bounded comparison: 57,552 meaningful-MFE
events, 44,858 clean favorable-first events, 56,875 adverse-first/conflict
events, 8,997 triple-barrier timeouts, 40,542 positive-net rows, 45,644
losses of at least 100 bps, 22,406 deployed full stops and 30,471 deployed
policy timeouts.  The full-horizon gross/MFE ratio is explicitly
diagnostic-only; it is not a valid pre-exit capture target.

The frozen contract is
`configs/canonical_execution_reliability_workstream_20260730_v1.json`.  It
requires the v4 four-score control, detached repaired support sidecars,
meaningful-MFE then capture, clean favorable-first, three-way competing risk,
and only then five bounded base-score-by-transition interactions.  It
forbids timing/MAE/target-price/wait fields from the EV layer and keeps April
non-promotable reused evidence.

The February--April IC/EV divergence is a mandatory output of every run.
The base's native-target rank IC improving from `0.155` to `0.162` to
`0.226` while pooled-global direct execution top-decile EV remains
`-59.39/-91.31/-38.45 bps` cannot be dismissed as “base is alpha, EV is
execution.”  The required diagnosis uses identical candidate cohorts and
must separate:

1. native-target versus exact 12h MFE/gross/deployed-exit gross/net rank
   quality;
2. aggregate IC from top-1/5/10/20% tail-local IC, calibration, opportunity
   precision/recall and conditional payoff magnitude;
3. score-scale/cutoff migration from within-fixed-band response drift;
4. stop, timeout, side, asset and regime composition from within-cell payoff
   changes;
5. target-horizon mismatch through matched 12h versus 24h labels; and
6. raw base, residual, direct-EV and causal-mapped-EV ordering on the same
   rows, including fixed cost-hurdle counterfactuals without reselection.

The work item is not closed by higher IC.  It closes only after each
month-to-month IC/EV delta is quantitatively reconciled and a frozen
causally mapped challenger is positive overall and in the latest period,
both-side viable, tie-safe and better than the frozen controls.

## 2026-07-30 — strict 2022–2025 regime/transition freeze and 2026 assessment

The calendar boundary is mandatory and machine-checked.  Every regime or
transition feature choice, preprocessing transform, geometry choice, class
weight, calibration method, persistence parameter, semantic assignment and
decision threshold must be learned using 2022–2025 only.  The 2026 rows are
an untouched assessment set and may be used only for metrics and
attribution.  Current-regime and transition-onset probabilities remain
separate inputs; neither is a substitute for the other.

The strict multihorizon transition ablation is sealed at
`strict_transition_v3_multihorizon_competing_risk_20260730_v2`.  Its four
blocked 2022–2025 folds select feature family, positive-class weight and
calibration independently for 1/3/6/12-hour onset heads before one evaluation
on 4,627 fully resolved 2026 hours.  The 1-hour head reaches AP 0.00680 at
0.00346 prevalence and AUC 0.618.  The 3-hour head is the strongest
transferable horizon, with AP 0.02122 at 0.01037 prevalence and AUC 0.687.
The 6-hour and 12-hour heads do not transfer (AUC 0.468/0.498).  The balanced
competing-risk lifecycle head is not usable (macro-F1 0.093), and the few
apparently discriminative individual causes have only 15–17 positive test
examples.  The high-risk global-top-10 economic slices remain negative in
May, June and partial July.  Retain 1h/3h probabilities as diagnostic
context; reject 6h/12h and lifecycle/cause actions.

The corrected strict regime-only result is sealed at
`strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3`.  It has
29,280 training hours from 30 August 2022 through 31 December 2025 and 4,627
untouched 2026 assessment hours.  Train-only processing includes
family-balanced feature selection (16 each for volatility, liquidity,
dependence/covariance and distribution dynamics), 0.5/99.5% winsorisation,
robust scaling, diagonal-GMM BIC selection, low-redundancy screening and a
final-block persistence sweep.  Transition counts never cross calendar gaps
and the filter resets at each gap.

This is a valid holdout but a failed regime architecture.  BIC selects six
states; posterior entropy remains almost zero, filtered state-change rates
are 28.3%–35.3% per month, and median filtered dwell is only two hours
(mean 3.09).  The selected sticky prior barely changes raw emissions.
Therefore v3 is authoritative evidence that a diagonal GMM over these
features does not produce a persistent economic regime taxonomy; it is not a
policy gate.  V2 is superseded because its persistence objective calculated
likelihood after normalisation and did not correctly separate temporal
switching/gap resets.

January–August 2022 remains a separate inverse-PI candidate population.  It
is valid older evidence but cannot be silently pooled into the later
feature/model lineage.  To satisfy literal January-2022 coverage in a single
fit, first materialise a common market-only feature contract on both
populations, then rerun this same pre-2026 freeze.  Until then, the honest
comparable training span is 30 August 2022–December 2025, with the early-2022
supplement reported separately.

## 2026-07-30 — sealed reliability ablation v2 and decision

Use this lineage, which supersedes the stale v2-input/v1-contract references
above:

- input:
  `data_perp/artifacts/canonical_execution_reliability_input_20260730_v4`;
- contract:
  `configs/canonical_execution_reliability_workstream_20260730_v2.json`;
- full result:
  `data_perp/artifacts/canonical_execution_reliability_ablation_20260730_v2`;
- compact decision:
  `data_perp/artifacts/canonical_execution_reliability_ablation_summary_20260730_v1`.

V4 changes feature authorization only.  Its panel hash is
`d6ecbe3a70116c7bb9c303a3dc7a2c8217aecd0abf85b269f53ae8d83946ba51`,
byte-identical to v3; `frozen_base_score_decile` becomes an approved context
input.  The ablation manifest is
`7a2a2a88571081a5cfd15748da749681e93aa3f120d8267689b56d224a3b5d02`.
All source/output hashes, the current runner/config hashes and both linked
IC-to-EV evidence artifacts verify.

The full artifact contains 21 configurations and 1,986,642 score rows.  Each
configuration has 25,344 March OOF candidates over one map-calibration and
three selection folds, plus 69,258 April frozen-forward candidates.  There
are no duplicate config/candidate/side identities.  All selection-fold map
snapshots satisfy the prior-resolved 21-day contract.  Exact
`execution_gross_ev_12h - execution_cost_return =
execution_net_ev_12h` has zero error.  Global-book attribution reconciles
exactly by side, asset, frozen March execution-risk quintile and realised
exit; no slice is reranked.

Frozen choices:

1. support: `score4+support_S0+support_S1B`;
2. context: `base_rank_pct_timestamp_side +
   base_score_z_timestamp_side`;
3. target architecture: A2 meaningful-MFE, conditional positive capture,
   conditional gain and adverse magnitude;
4. final research challenger: A2 plus exactly five fold/side-train
   standardized, clipped base-score-by-12h-transition interactions.

The final name is
`A5__A2__context__timestamp_side_relative`.  Its March selection mean,
latest, worst and stability objective are
`-64.12 / -50.69 / -81.54 / -90.96 bps`.  The mapped residual control is
better at `-60.13 / -55.07 / -68.52 / -80.25 bps`.

Promotion ledger:

| Gate | Observed | Result |
|---|---:|---|
| March mapped aggregate top 10% | -93.54 bps | fail |
| March latest fold | -50.69 bps | fail |
| March worst fold | -81.54 bps | fail |
| April mapped reused diagnostic | -59.12 bps | fail |
| April latest seven days | -131.09 bps | fail |
| March long / short contribution | -42.51 / -51.03 bps | fail |
| Aggregate March/April tie share | 2.07% / 0.32% | pass |
| Maximum selection-fold tie share | 23.27% | fail |
| Delta versus residual control, March aggregate | -22.10 bps | fail |
| Delta versus residual control objective | -10.70 bps | fail |
| New untouched forward evidence | absent | fail |

The correct state is
`SEALED_RESEARCH_DECISION_NO_PROMOTION_NO_PORTFOLIO_REPLAY`.

Mechanism:

- meaningful-MFE classification remains weak (AUC approximately 0.57,
  AP approximately 0.35, ECE approximately 0.12);
- conditional capture classification is stronger (AUC 0.65--0.67,
  AP approximately 0.94) but has approximately 90% prevalence;
- positive/favorable payoff magnitude has rank IC approximately 0.22--0.24,
  and timeout payoff approximately 0.36;
- the adverse loss head is sparse (159--513 fold/side training rows) and
  has rank IC near zero;
- transition interactions materially improve A2 stability and capture
  discrimination, but not the weak MFE or loss channels;
- April A5 trailing exits contribute +95.31 bps, overwhelmed by full-stop
  and timeout contributions of -106.04 and -48.39 bps;
- all five frozen execution-risk quintiles remain negative.

Next implementation must target the conversion bottleneck, not expand
generic HPO: add cost-aware pre-exit opportunity, successful-trailing,
deployed-full-stop and timeout heads; replace sparse loss regression with a
probability/severity hurdle; split broad competing risks by deployed exit
outcome; retain only timestamp-side context and bounded transitions; and
test a causal rank-preserving alternative to flat isotonic maps.  Extend
strict same-lineage history before March and reserve untouched forward data
before any replay.  Timing, MAE, target-price and wait/reprice actions remain
in the separate downstream action layer.

## 2026-07-30 — corrected causal mapping ablation v2

The authoritative artifact is
`data_perp/artifacts/canonical_execution_reliability_mapping_ablation_20260730_v2`.
The prior v1 artifact now contains `INVALIDATION.json` and must not be cited:
it omitted the residual control, allowed an invalid M2 specification, used a
noncanonical latest-seven-day interval, used nonfractional attribution and
departed from the predeclared M3 design.

V2 evaluates the final A5 reliability challenger and the identically eligible
residual control under:

1. the frozen baseline mapper;
2. M1 strict pooled, within-snapshot rank-preserving mapping;
3. M2 positive-slope robust mapping;
4. M3 pooled-only timestamp-percentile 20-bin mapping with pseudo-count 200,
   PAVA and no side residual.

All 40 common eligible days per configuration are causal. Reference labels
are resolved before each score snapshot, scores are available at the snapshot,
and reference/evaluation identity overlap is zero. M1/M2/M3 each record zero
within-snapshot inversions and zero plateaus. Side, asset, frozen risk-state
and realised-exit fractional attribution reconciles to the single global book
with maximum absolute error `1.78e-14` bps. Five focused tests pass; the
manifest, seal, every output hash and the current runner hash independently
verify.

| March selection objective | Baseline | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| Residual control | -80.25 bps | -104.67 | -101.68 | -106.34 |
| A5 challenger | -90.96 bps | -159.21 | -163.71 | -144.47 |

A5 global top-10% April aggregate/latest-seven-day economics are respectively:
baseline `-59.28/-131.09`, M1 `-73.73/-102.86`, M2
`-83.43/-93.20`, and M3 `-64.53/-88.94` bps. A5 loses to the
same-mapper residual control on aggregate March and April evidence for all four
mappers. M2 and M3 improve A5 relative to the residual only in the narrow
April latest-seven-day comparison; both remain strongly negative and fail the
March and April aggregate gates.

**Decision:** `SEALED_RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY`.
Calibration is not the primary explanation for the improving-alpha-IC /
negative-execution-EV divergence. Preserve the causal mapping contract, but
move the active diagnosis to cost-clearing opportunity incidence, realised
exit outcomes, signed conditional payoff branches and regime transfer. Do
not spend another ablation on a more flexible mapper until a pre-mapping score
has positive, stable gross-to-net tail separation.

## 2026-07-30 — sealed A-grade cost-clearing conversion v5

The authoritative restart-safe artifact is
`data_perp/artifacts/a_grade_cost_clearing_conversion_ablation_20260730_v5`.
It supersedes v4 for execution evidence because v5 binds seven immutable
14-day fold checkpoints. Five are scored and two are warm-up; maximum fold
wall time is 9.84 seconds. Six focused tests pass. Every input/output hash,
the detached manifest hash and the current runner hash independently verify.

Exact common intersections are:

- historical: 110,730 of 140,682 strict rows, 2025-03-12 through 2025-04-30;
- current: 52,295 of 127,777 strict rows, 2026-06-08 through 2026-07-10.

The residual control is compared with a cost-clearing alpha hurdle:
class-balanced standardized logistic `P(execution_net_ev_12h > 0)` using only
residual EV, base alpha and side, multiplied through train-only,
side-conditional positive/negative payoff estimates. Within-lineage
diagnostics add regime, transition, or both. All train labels resolve before
their 14-day test block; per-arm isotonic maps use only earlier resolved OOF
score/outcome rows. Selection is one pooled global top 10% per lineage/month,
with no side quota and no weekly reranking.

Strict forward scoring fits, selects and maps on 2025 only, then applies to
2026 without 2026 labels:

| Arm | Aggregate monthly net EV | July net EV |
|---|---:|---:|
| Residual control | -96.16 bps | -126.02 bps |
| Alpha cost-clearing hurdle | -106.21 bps | -114.36 bps |

The hurdle recovers 11.66 bps in July but loses 10.05 bps in aggregate and
remains negative. It also collapses diversification to 10--13 selected
assets, versus about 98--113 for the residual control. Positive-net recall
moves from 8.22% to 9.42% in July but falls from 11.42% to 10.50% in June.

The context arms are diagnostic only. April 2025 regime-plus-transition is
`-103.19` bps versus residual `-37.91`; July within-lineage is `-86.45`
versus residual `-96.62`. The latter is not evidence of transfer: 2025 and
2026 context sidecars have incompatible feature semantics. The runner
therefore correctly publishes zero strict-forward context scores and marks
all three context arms
`fail_closed_noncomparable_2025_2026_context_feature_contract`.

**Decision:** `SEALED_DIAGNOSTIC_NON_PROMOTION`; no portfolio replay. A
generic cost-clearing binary hurdle is insufficient, and the mapper is already
ruled out by mapping v2. The remaining causal mechanism is whether
semantically identical, pre-entry regime/transition state can identify when
high-alpha candidates will convert into cost-clearing opportunity and safe
capture. Build that common cross-era feature contract before retesting the
hurdle. Continue the separate explicit opportunity/exit-outcome decomposition
to distinguish missed opportunity, successful trailing, full stop and timeout.

## 2026-07-30 — nonlinear alpha-tail cost-clearing verdict

The sealed artifact is
`data_perp/artifacts/nonlinear_alpha_tail_cost_clearing_hurdle_20260730_v1`.
Three focused tests pass; the detached manifest, all 15 recorded outputs and
the current runner hash independently verify.

The arm extends the v5 linear hurdle with fixed timestamp-by-side alpha
percentile, 20 ventile indicators, 80/90/95 tail hinges and side interactions.
There is no HPO. Ranking ties are deterministic before fitting; selected
cutoff ties are allocated fractionally. All three arms share 50,826 April OOF
IDs and 52,295 forward IDs. Every 2025 training/map label resolves before its
fold, and 2026 uses only the frozen 2025 fit and 2025 blocked-OOF map.

| Frozen 2025 fit/map -> 2026 global top 10% | June | July | Average |
|---|---:|---:|---:|
| Residual control | -66.45 bps | -131.46 bps | -98.96 bps |
| V5 linear alpha hurdle | -101.59 bps | -116.53 bps | -109.06 bps |
| Nonlinear alpha-tail hurdle | -101.50 bps | -117.89 bps | -109.70 bps |

The nonlinear arm has forward long/short net contributions of
`-113.96/-98.66` bps. It fails positive aggregate/latest/worst economics,
improvement over both controls and both-side positivity. Its causal, identity,
pooled-global selection and frozen-forward gates pass.

It also fails the tie gate for a substantive reason: the 2025 OOF mapper
places 35.99% of April candidates on the cutoff, and the frozen-forward map
places 99.997% of June and 99.940% of July candidates there. Fractional
selection preserves exact global mass, but the score has essentially no
transported selection resolution.

**Decision:** `SEALED_DIAGNOSTIC_NON_PROMOTION`; no portfolio replay. Fixed
alpha-tail nonlinearity is eliminated as the missing repair. The remaining
tests are the explicit cost-aware opportunity/exit-outcome hierarchy and a
cost-clearing hurdle using the sealed common-semantic cross-era transition
geometry. Do not spend another run on additional alpha bins or mapper
flexibility unless a new independent cohort supplies a different hypothesis.

## 2026-07-30 — common-semantic transition conversion verdict

The sealed artifact is
`data_perp/artifacts/common_semantic_transition_cost_clearing_ablation_20260730_v1`.
Three focused tests pass. The common-geometry manifest, every input/output
hash, detached manifest seal and current runner hash independently verify.

The run uses the already sealed 90-field cross-era semantic geometry: 36
state-level fields and 54 strict 1/3/12h lag/delta fields derived from the
same nine raw fields. Exact `__ts__ == signal_context_utc` joins are used with
no fill. Historical coverage is 110,610/110,730 rows (two missing timestamps);
forward coverage is 51,279/52,295 after seven missing timestamps and 426
feature-incomplete rows are excluded. The final arm-local-map intersection
contains identical 50,706 blocked-2025 OOF and 51,279 forward IDs for every
arm.

All train labels end before their fold. The 2025 full fit and 2025 blocked-OOF
map are frozen before scoring 2026. Selection is one pooled fractional global
top-k with no side quota.

| Forward mapped global top 10% | Aggregate | June | July |
|---|---:|---:|---:|
| Residual control | -82.62 bps | -66.45 | -132.75 |
| V5 linear hurdle | -105.74 bps | -101.59 | -116.69 |
| State context | -105.12 bps | -100.83 | -116.89 |
| Transition deltas | -105.74 bps | -101.59 | -116.69 |
| State + transition | -105.37 bps | -101.55 | -115.68 |

Causal legality, identity parity, pooled-global selection and 2026-label
isolation pass. All economic-improvement, both-side-positive and tie gates
fail. The mapped challenger cutoff-tie fractions are approximately
80.5--100%, so their calibrated scores have little or no selection resolution.

The unmapped transition-only score is directionally better: aggregate raw
top-10 net EV is `-72.82` bps versus `-84.42` for the raw residual control,
and it improves both June and July by about 14 bps. The result remains
negative and therefore cannot justify bypassing the required recent EV map.
It shows only that the common transition geometry may be useful as a bounded
interaction after a viable economic component exists.

**Decision:** `SEALED_DIAGNOSTIC_NON_PROMOTION`; no portfolio replay. Reject
the 90-field state/transition expansion as a standalone binary cost-clearing
head. Wait for the explicit opportunity/exit-outcome hierarchy. If that
produces a transferable component, the sole admissible follow-up is a bounded
residual-plus-transition interaction under a strictly order-preserving
common-unit map, compared on the identical frozen-forward global book.

## 2026-07-30 — checkpointed explicit exit/outcome hierarchy

The authoritative artifact is
`data_perp/artifacts/canonical_execution_reliability_exit_hurdle_ablation_20260730_v1`,
with persistent immutable checkpoints at the sibling `_checkpoints` directory.
The original all-or-nothing fit terminated without an artifact. Before the
rerun, the runner was hardened with identity-bound side/fold/head checkpoints,
an immutable root contract, current-runner/config/input/target/parent
fingerprints, safe resume, duplicate-run locking and a JSONL progress log.

The successful clean rerun records 290 START and 290 COMPLETE events, one
RUN_COMPLETE, zero ERROR/REUSE events and process exit 0. Nine focused tests
pass. Every checkpoint prediction hash, all eight published output hashes,
the manifest seal and current runner hash independently verify.

Architectures:

1. H1: `P(pre-exit MFE > row cost + {0,25,50}bps)`, conditional successful
   trailing and gain, plus signed opportunity-failure and no-opportunity
   payoffs;
2. H2: four-class successful-trailing/trailing-nonpositive/hard-adverse/timeout
   probability with class-conditional payoff;
3. H3: successful-trailing hurdle followed by the competing-risk outcome and
   conditional payoff branches;
4. H4: severe-loss probability, conditional log severity and signed
   non-severe payoff.

The H1 signed branches are present in the executed runner and checkpoint
contract. This corrects the earlier conceptual error of representing every
non-success branch as a positive loss magnitude.

| Mapped global top 10% | March | April frozen | Latest 7 decision days |
|---|---:|---:|---:|
| H0 A0 residual control | -71.44 bps | -30.21 bps | -81.23 bps |
| H0 A5 reliability challenger | -93.54 | -59.12 | -130.26 |
| H1 opportunity 0bps | -98.27 | -62.40 | -90.24 |
| H1 opportunity 25bps | -120.76 | -64.69 | -91.01 |
| H1 opportunity 50bps | -119.57 | -73.42 | -106.00 |
| H2 four-class | -119.57 | -66.50 | -99.40 |
| H3 hierarchical | -115.36 | -57.23 | -79.61 |
| H4 severe hurdle | -81.51 | -63.71 | -72.19 |

H1 0bps is the frozen learned selection winner: March fold top-10 net is
`-109.92/-108.17/-52.66` bps and its mean/latest/worst/objective is
`-90.25/-52.66/-109.92/-131.03`. It still loses to the residual control.
March H1 gross is only +1.74 bps against 100.01 bps cost. Long/short
contributions are `-3.64/-94.64` bps, with 91.9% of selected mass short.
Its aggregate cutoff-tie selected share is 22.57%, and the mapping fold
reaches 93.51%.

All seven promotion gates fail: March aggregate, latest/worst, April/latest7,
both-side contribution, tie safety, control improvement and untouched-forward
evidence. The correct status is
`RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY`.

Head evidence explains why a more elaborate replacement score is not the next
move. Opportunity and success event AUCs are only about 0.60--0.61; H2/H3
competing-risk macro AUC is about 0.58--0.59; severe-loss AUC is 0.54.
Some conditional payoff heads do rank outcomes—hard-adverse payoff rank IC is
about 0.521 and non-severe payoff about 0.281—but their probability/payoff
composition does not produce cost-covering selection.

**Decision:** reject H1--H4 as replacement execution-EV architectures and do
not replay. The sole next component use is a bounded adverse-risk overlay on
the unchanged residual control, with identical causal mapping/global books.
Timing, MAE, target-price and wait/reprice remain in the separate action layer.

## 2026-07-30 — bounded adverse-risk overlay and branch closure

The sealed component-use artifact is
`data_perp/artifacts/bounded_adverse_risk_overlay_ablation_20260730_v1`.
Three focused tests pass independently. The frozen config, runner, parent
artifacts, checkpoint contract, all 40 reconstructed H2/H4 prediction payloads,
every output hash and the manifest seal verify.

The run adds only bounded signed adverse penalties to the unchanged residual
score:

- H2: `P(hard_adverse) * clipped conditional hard-adverse payoff`;
- H4: `P(severe_100bps) * clipped conditional severe-loss magnitude`;
- combined H2+H4.

All components are side/fold OOF or April frozen. Class order is verified
before extracting hard-adverse probability; penalty signs are non-positive;
clipping bounds come from that fold/side's training outcomes. Fixed lambdas
0.25/0.5/1.0 are compared under the same causal 21-day mapper and pooled
fractional global books.

The March selection winner is `h2_lambda_0.25`:

| Evidence | Residual | Winner |
|---|---:|---:|
| March mean/latest/worst/objective | -60.13/-55.07/-68.52/-80.25 | -59.81/-55.10/-67.79/-79.59 |
| March mapped aggregate top 10% | -71.44 | -72.10 |
| April mapped aggregate top 10% | -30.21 | -30.65 |
| April latest seven days | -81.23 | -81.17 |

The 0.66-bps stability-objective improvement and 0.06-bps latest-seven-day
improvement do not survive aggregate evidence. Both March side contributions
are negative. Tie safety passes, but aggregate/latest/worst positivity,
both-side, beat-control and untouched-forward gates fail.

**Decision:** `RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY`. Do not use the
adverse heads as replacement scores, direct penalties or portfolio inputs on
this cohort.

This closes the current downstream meta-ranking hypothesis set:

1. causal mapping alternatives do not repair A5;
2. a linear positive-net hurdle fails;
3. fixed nonlinear alpha-tail structure fails;
4. common-semantic state/transition expansion fails after mapping;
5. H1--H4 explicit opportunity/exit compositions fail;
6. the strongest isolated adverse component is nonincremental as a bounded
   residual overlay.

The unresolved IC-to-EV gap is therefore upstream of this family of meta
rankers. The mandatory next evidence is an identical-ID exit/opportunity
counterfactual on residual-selected global top 1/5/10/20 books: 12h MFE
oracle, fixed-horizon return, deployed-exit capture ratio, cost-clearing
opportunity prevalence, full-stop regret and timeout regret, split by
month/side. If achievable gross remains below cost, redesign the candidate
target/horizon/universe rather than fit another meta head. If the oracle is
adequate but deployed capture is not, move the repair to the separate
timing/exit action layer.

## 2026-07-30 — identical-ID exit/opportunity diagnosis

The required diagnostic is complete and sealed at
`data_perp/artifacts/residual_selected_exit_opportunity_counterfactual_20260730_v3`.
`v1` is explicitly invalidated; it used a native base-label path source for
the fixed-close field.  `v2` uses
`febapr2025_top40_exact1m_paths_20260727_v1/paths.parquet`, verifies exact
candidate/side/symbol/signal/decision/path-end parity, and applies the
canonical row cost once. `v2` is also explicitly invalidated: its primary
economics were sound, but it called an exit-minute-inclusive path field
strictly pre-exit and failed to mask that auxiliary metric to policy-path
parity-valid rows.

The run preserves the sealed H0 A0 causal mapped score and selects one
fractional-tie pooled-global monthly top 1/5/10/20 book.  All counterfactuals
use those identical IDs and weights.  Three focused tests pass; all eight
March/April × top-fraction deployed economics rows reproduce the sealed
control to below `1e-9` bps.

Global top-10 result:

| Metric | March | April |
|---|---:|---:|
| Deployed net | -71.44 bps | -30.21 bps |
| MFE oracle net | +145.15 | +188.18 |
| Exact fixed-12h close net | -21.80 | +53.89 |
| Through-exit-minute MFE net, parity-valid rows | +42.19 | +69.50 |
| Opportunity > cost + 25bps | 41.99% | 48.39% |
| Full stop / timeout share | 21.96% / 25.76% | 15.76% / 27.38% |

Day-cluster 95% intervals make the main contrast robust: deployed net remains
below zero in both months, while MFE net remains above zero.  Fixed-12h is
negative/inconclusive in March and positive in April, so simply extending
every trade to 12h would not solve the problem.

Exit-conditional top-10 evidence:

- full-stop MFE-oracle net is `-59.77/-50.24` bps and timeout MFE-oracle net
  is `-46.86/-64.30` bps: these rows mostly lacked cost-clearing opportunity;
- trailing MFE-oracle net is `+326.56/+376.18` bps, but deployed trailing net
  is only `+107.76/+136.04` bps;
- full-stop and timeout oracle-regret contributions sum to about
  `102.21/81.90` bps;
- March long deployed/fixed-12h is `-105.61/-87.43` bps, while March short
  fixed-12h is `+51.46` bps; April fixed-12h is positive on both sides.

This resolves the immediate IC/EV question more precisely.  There is real
path opportunity on the residual-selected books, but it is mixed with
no-opportunity full-stop/timeout admissions, and the deployed trailing policy
leaves material favorable movement uncaptured.  March-long opportunity is
short-lived rather than a durable 12h move.

The next implementation target is therefore a separate, OOF-trained action
layer on unchanged rankings.  First materialize exact 1/2/4/8/12h returns,
time-to-MFE, early MAE/flatness, time-under-water and post-MFE giveback.
Then replay a deliberately small causal exit family (fixed horizon, partial
profit plus trailing remainder, time stop, trailing-width/decay variants)
before training `trade/skip/wait/reprice` and
`hold/partial/exit/tighten/loosen` decisions.  Timing, MAE and target-price
outputs remain forbidden inputs to the EV ranker.

No portfolio replay or simple-policy optimization is authorized until one
frozen action policy is positive by month and side, improves both deployed and
fixed-time controls, clears clustered uncertainty, and preserves the pooled
global selection contract.

## 2026-07-30 — action targets, fixed controls and stateful exits

The action target pack is complete and sealed at
`data_perp/artifacts/execution_action_target_pack_20260730_v2`.  It covers
110,730 canonical rows using exact contiguous 720x1m execution paths and
contains side-relative fixed 1/2/3/4/8/12h returns, MFE/MAE, slope,
underwater, peak/timing, cost-clear, early 2--3h path-quality and giveback
targets.  Canonical cost is deducted once from each fixed-close net target.
Every target has an explicit availability horizon and is forbidden as an
inference feature.  The v1 target pack is invalidated because zero-MFE rows
were assigned finite fraction-of-peak timing; v2 censors and masks those
fields correctly.

The unchanged-book fixed controls are sealed at
`data_perp/artifacts/fixed_horizon_action_ablation_20260730_v2`.  They retain
the exact v3 pooled-global monthly top-1/5/10/20 IDs and fractional weights,
apply the canonical cost once and use 2,000 paired UTC-day bootstrap draws.
Deployed parity is below `1.5e-14` bps.

| Global top-10 arm | March | April |
|---|---:|---:|
| Deployed | -71.44 bps | -30.21 bps |
| Fixed 1h | -74.70 | -60.14 |
| Fixed 2h | -63.07 | -39.73 |
| Fixed 4h | -57.43 | -15.26 |
| Fixed 8h | -46.90 | +22.08 |
| Fixed 12h | **-21.80** | **+53.89** |

Fixed 12h improves deployed by `+49.64/+84.10` bps, but its March interval is
`[-69.06,+39.78]` and March long remains `-87.43` bps.  Hence no fixed hold
is promoted.  Fixed-ablation v1 is invalidated only for a hard-coded
March/April capacity assertion; v2 derives capacity from the sealed parent.

The first live-state variants are sealed at
`data_perp/artifacts/frozen_exit_state_action_ablation_20260730_v4`.
v1 is the complete pre-partial phase; v2 is superseded because it did not
materialize P50's exact two-exit fee robustness table.  v3 is invalidated for
P50 only: its partial ran after same-bar intrabar exit checks even though
activation was known at the open.  v4 gives the open partial first priority,
then preserves the exact remainder state and exit order.
The exact simulator replay has zero mismatches on all 18,107 deployed
controls across gross/net, exit hour/reason, executable entry/exit prices,
spread fields and geometry.  Each variant reuses the sealed row cost:

- `T4` keeps deployed stop/trail state for 240 minutes and then uses the
  spread-aware simulator timeout fill;
- `D2` decays activation after minute 120 with a 120-minute half-life toward
  a 50% floor;
- `W75` tightens the active trailing width to 75%.
- `P50` takes 50% at the causally executable next-bar open when trailing
  activation first becomes known and leaves the remainder in unchanged state.

| Global top-10 arm | March | April | Delta vs deployed |
|---|---:|---:|---:|
| `T4` | -121.16 bps | -88.83 bps | -49.72 / -58.62 |
| `D2` | -67.94 | -27.56 | +3.50 / +2.65 |
| `W75` | -72.48 | -32.34 | -1.04 / -2.13 |
| `P50` | -80.43 | -41.27 | -8.98 / -11.06 |

`T4` shows why the raw fixed-4h close is not a deployable time-stop proxy:
the actual stateful/spread-aware timeout is materially worse.  `D2` is the
only directionally helpful arm, changes outcomes on about 17.7% of rows, but
its paired 95% improvement intervals cross zero in both months
(`[-2.19,+10.16]`; `[-0.87,+6.14]`), all global books remain negative and
three of four month-side books remain negative.  `W75` is nonincremental and
worse in April.

P50 activates on 56.38%/61.75% of top-10 rows at a weighted mean
4.07h/3.68h.  Its paired degradation intervals exclude zero in both months
(`[-12.74,-6.00]`; `[-13.51,-8.81]`).  Exact weighted two-exit fees change
the primary canonical-cost result by only +0.045/+0.055 bps, so accounting
robustness does not alter the rejection.  A universal first-activation
partial exit gives away more later payoff than it protects.

Every strict month, side, uncertainty and fixed-12h gate
fails.  No portfolio replay is authorized.

Remaining work is deliberately narrow:

1. train OOF/OOF-equivalent pre-entry `trade/skip/wait/reprice` and post-entry
   `hold/partial/exit/tighten/loosen` heads from the sealed target pack;
2. retain D2/P50 only as conditional action choices; do not HPO wider
   decay/fraction grids on the reused diagnostic months;
3. keep timing, MAE and target-price outputs out of the EV ranking head;
4. require positive month and side net, positive paired clustered lower
   bounds, and improvement over deployed plus fixed controls before any
   simple-policy or portfolio replay.

## 2026-07-30 — frozen no-reranking wait10 action

The sealed action-learning input is
`data_perp/artifacts/frozen_entry_action_handoff_20260730_v2`:

- 18,107 unchanged pooled-global March/April identities and fractional
  top-1/5/10/20 weights;
- 45 authorised pre-entry model inputs with explicit roles;
- exact targets and 720x1m paths kept target-only;
- exact deployed barrier/archetype inputs kept replay-only;
- normalized-symbol plus exact timestamp assertions for the historical
  `/`-versus-`_` representation mismatch.

Do not use the older `execution_entry_timing_meta` simulator for this frozen
book.  It did not reproduce the current exit policy (about 315 bps maximum
sample difference).  The canonical runner is
`scripts/run_frozen_preentry_wait_action_ablation.py`; it uses the current
`simple_policy_optimiser`, requires exact enter-now parity, and never reranks
or backfills.

The sealed result is
`data_perp/artifacts/frozen_preentry_wait10_action_ablation_20260730_v2`.
Wait10 enters at decision plus 10 minutes and simulates exactly the remaining
710 bars to the original deadline with freshly recomputed costs.  Control
parity is zero-mismatch across gross/net, exit hour/reason, MFE/MAE,
entry/exit prices, spread fields and geometry.  All March training labels
resolve strictly before validation.  v1 is invalidated for a
boundary-inclusive label-resolution timestamp.

At global top 10%, March chronological OOF is `-73.04` bps enter-now versus
`-84.97` always-wait; the hindsight oracle is `-66.58` (`+6.46`).  The best
learned March diagnostic is full-soft at `-73.48` (`-0.44`,
95% `[-1.01,+0.22]`).  April frozen March-forward is `-30.21` enter-now,
`-40.58` always-wait and `-25.47` oracle (`+4.74`,
95% `[+3.50,+5.87]`).  The best learned April diagnostic, full-soft, is
`-32.83` (`-2.62`, 95% `[-3.73,-1.64]`).  No learned policy passes.

The better-wait event classifier is nevertheless informative: April AUC is
`0.738/0.689` long/short for compact inputs and `0.746/0.709` for the full
authorised set.  Wait is truly better on only `5.0%/10.4%` of rows; magnitude
rank correlations are near zero, while learned policies wait on 26--28% of
the top-10 book.  This is a rare-event magnitude/calibration and abstention
failure.  The long top-10 contribution remains `+9.71` bps when direct/soft
abstains; short routing causes the degradation.

Next: extend action labels across older untouched blocks; train calibrated
cost-asymmetric event and magnitude heads with a positive-utility lower-bound
gate; then implement an exact-current-policy adverse-limit action with
fill/adverse/missed-opportunity components.  Portfolio replay remains gated
off.  A learned post-entry router still requires causal prefix-state
materialization.

## 2026-07-30 — older exact-policy Wait10 training and frozen-book result

Two new sealed artifacts implement the first older-data extension:

- `febapr2025_current_policy_wait10_action_20260730_v1`: 205,194 unchanged
  residual-top40 identities, 34 exact PIT inputs, exact enter-now and Wait10
  labels, and zero current-policy parity mismatches;
- `frozen_older_data_wait10_action_ablation_20260730_v1`: side-local older-data
  heads scored on the unchanged March/April global books and weights.

February is valid all-candidate action training, but it does not extend the
same selected-book lineage: it is base-only, residual is not OOF, and no
causal recent-EV mapped score or frozen global weight exists.  No February
book was reconstructed.  Broad-population Wait10-positive rates are
20.36%/23.27% long/short in February, versus only about 5--10% in the frozen
mapped top-10 evaluation tail.

All state/transition inputs materially improve the rare-event classifier.
March AUC is 0.761/0.795 long/short from February training; April is
0.795/0.756 from February-plus-resolved-March.  Magnitude routing remains the
failure.  Direct/expected magnitude scores do not rank the selected tail,
while the expected-delta rule nearly always waits and loses about 10 bps.

Complete-book top-10 results (the earlier -73.04 March number was the
chronological-OOF subset, not this full 4,149-row March frozen book):

| Evaluation | Enter now | Always Wait10 | Oracle Wait10 | Best learned |
|---|---:|---:|---:|---:|
| March complete frozen book | -71.44 bps | -82.86 | -65.12 (+6.32) | -71.44 (+0.004) |
| April frozen forward | -30.21 | -40.58 | -25.47 (+4.74) | -29.93 (+0.28) |

The April best is February-base-rank-top-half/base-only direct delta, waits
on 2.30% of global top-10 weight, leaves long unchanged and adds only
+0.38 bps within short.  Its paired interval is [-0.01,+0.63] bps.  The March
best interval is [-0.005,+0.013].  Neither excludes zero.

A train-only day-cluster positive-lower-bound rule abstains in 23/24
source/feature/side calibrations.  The sole admitted threshold acts on only
0.058% of April top-10 weight and is flat/slightly negative.  This is a
successful safety result but a failed action-alpha result.  No configuration
is promotable and no portfolio replay is authorized.

A resumable 293,828-row Apr-2023--Dec-2024 materializer is now implemented in
`scripts/materialize_2023apr_2024_current_policy_wait10_action.py`.  It uses
held-block OOF base/residual scores, candidate-keyed OOF regime-transition
probabilities, exact 720x1m paths and exact side-archetype policy controls.
It must preserve full enter-now outcome and geometry parity.  It creates
training labels only—never reconstructed historical global-book weights.

## 2026-07-30 — base-IC/execution-EV interpretation correction

The `0.155 -> 0.162 -> 0.226` native-target IC sequence is long-side.  The
negative top-decile values are pooled-global raw-base books.  Always print
pooled, long and short IC beside pooled-global economics rather than implying
that those two quoted series are one matched statistic.

The raw-base mechanism is already sealed.  Pooled-global top-10
gross/cost/net is +49.38/100.25/-50.87 bps in February,
+17.05/100.09/-83.03 in March and +41.86/100.21/-58.35 in April.
February-to-March deterioration is mostly favourable-payoff scale and
positive-net prevalence; rank-cell composition contributes only about
+0.26 bps.  A fixed-12h close helps materially but remains negative, proving
that exit conversion is part—not all—of the gap.

The remaining missing bridge is the deployed selection axis: one sealed,
same-row comparison of raw base, causal-mapped base, residual, true raw
direct-EV and causal-mapped direct-EV with causal map availability and one
pooled-global top-k rule.  Existing February--April historical waterfalls are
raw-base diagnostics and must not be described as deployed mapped-score
performance.

### 2026-07-30 — identical-row causal score-conversion bridge completed

The missing deployed-axis diagnostic is sealed at
`marapr2025_identical_causal_score_bridge_20260730_v1`.  The materializer
reconstructs the historical direct-head folds from the hash-bound cross-era
dataset; proves exact four-field coverage and bit-identical raw base, residual
and true q25 scores; and retains the same exact current-spread 1m
decision-plus-12-hour gross/cost/net labels.

For fairness, it does not reuse the old base map trained on a larger base-only
population.  It fits two fixed, diagnostic-only pooled isotonic calibrators on
the same 140,682-row candidate population—base alpha to exact net and direct
q25 to exact net—with an identical 21-day prior-resolved reference set and
2,000-row minimum.  There is no model fit, HPO, threshold selection, side
quota or action-layer change.  March 1--2 lack enough causal support, leaving
136,074 identical evaluation rows (66,816 March; 69,258 April).  February
cannot enter this comparison because strict residual and true direct-q25 OOF
lineages start in March.

Pooled-global monthly top-10 results:

| Layer | March MFE/gross/cost/net | April MFE/gross/cost/net |
|---|---:|---:|
| Raw base alpha | 288.99 / 33.24 / 100.17 / **-66.93** | 322.42 / 66.39 / 100.33 / **-33.94** |
| Causal-mapped base | 287.75 / 23.97 / 100.12 / **-76.15** | 302.81 / 55.12 / 100.28 / **-45.16** |
| Residual expected EV | 322.08 / 69.20 / 100.35 / **-31.15** | 297.10 / 76.06 / 100.38 / **-24.32** |
| Raw direct q25 | 334.26 / 72.88 / 100.36 / **-27.48** | 288.75 / 6.80 / 100.03 / **-93.24** |
| Causal-mapped direct q25 | 329.10 / 70.37 / 100.35 / **-29.98** | 235.04 / -7.96 / 99.96 / **-107.92** |

The base-IC question is now resolved on a matched axis.  Raw-base pooled
native-target rank IC rises from 0.147 in March to 0.184 in April; exact-net
IC rises from 0.068 to 0.112.  The selected gross return nearly doubles while
cost stays near 100 bps, and net improves by 32.99 bps.  Better base rank does
convert into better economics, but the opportunity scale remains below cost.
The earlier long-side 0.155/0.162/0.226 series and pooled-global book values
must still not be quoted as one statistic.

The residual is the most stable conversion layer: it improves raw base by
+35.78 bps in March and +9.62 in April.  Day-block bootstrap intervals for
those deltas still cross zero, and neither month becomes profitable, so this
is incumbent selection evidence rather than promotion evidence.

The direct q25 layer exposes the larger non-transfer problem.  Its March tail
is competitive, but April falls to -93.24 bps.  In the April global book,
74.2% of rows are short; short contributes -101.75 bps at a -137.05 bps
within-side mean, while long contributes +8.51 bps at +33.05 bps.  This is a
short conditional-payoff/adverse-conversion failure, not a universal direct
head failure.

Neither causal map repairs its source:

- base mapping changes net by -9.23/-11.22 bps in March/April;
- direct mapping changes net by -2.50/-14.69 bps.

The maps collapse scores to about 0.8--1.2% unique levels and materially
change calendar allocation.  April mapped-direct top-10 uses only 19 days and
places 66.4% of selected rows in five days, versus 54.0% for raw direct.  The
failure is therefore a combination of plateau/tie compression and unstable
cross-day level calibration under the global top-k protocol.

No configuration is promotable and portfolio replay is unauthorized.  Next:

1. keep residual expected EV as the candidate-ranking incumbent;
2. compare causal rank-preserving affine/spline and shrink-to-raw maps against
   isotonic, with explicit plateau and calendar-concentration penalties;
3. repair April short conditional payoff/adverse-risk conversion using causal
   regime-transition context, without imposing a fixed side quota;
4. confirm any frozen winner on a later untouched month before replaying
   concurrency, exposure or asset limits.

## 2026-07-30 — cross-era Wait10 regime reversal

`2023apr_2024_current_policy_wait10_action_20260730_v1` is now complete and
sealed.  It contains 293,828 all-candidate training rows with held-block OOF
base/residual scores, candidate-keyed OOF transition context and exact
current-policy action labels.  Full numeric, exit-reason and side-archetype
geometry parity is exact.  It creates no historical book weights.

Always delaying entry by ten minutes is mildly beneficial in the older broad
population: +1.11/+2.41 bps long/short in Apr--Dec 2023 and +4.22/+2.24 in
2024.  It is materially harmful in February--April 2025.  This is a real
action-label regime reversal, not merely a model transfer failure.

The common-feature transfer artifact
`cross_era_wait10_transition_ablation_20260730_v1` uses the same raw causal
hourly transition calendar on both sides of the era boundary.  Best frozen
top-10 improvement is +0.20 bps March and +0.08 bps April; neither lower
confidence bound is positive.  Current raw score scales are 5--7 historical
IQRs out of range, so raw score-level transfer is invalid as a final test.

The high-entropy/low-persistence transition cell is itself non-stationary:
historical Wait10 delta is +3.70 bps in that cell, but every March/April 2025
row falls into the same cell and averages -7.17/-5.62 bps.  The coarse
transition category identifies instability but not the economic subtype.

`cross_era_wait10_transition_ablation_20260730_v2` adds the causal fields that
distinguish those subtypes: BTC-alt resilience, breadth dispersion and
downside intensity, compression quality, recent short-damage state, funding
change and state age.  April-long event AUC improves to 0.678.  Best frozen
top-10 lifts become +0.39 bps March (95% [-2.14,+2.96]) and +0.15 bps April
([-0.08,+0.41]).  The subtype fields are directionally incremental, but the
economics remain too small and uncertain for promotion or portfolio replay.

Next action-head work:

1. replace cross-era raw score levels with complete-group timestamp/side ranks
   and robust z-scores;
2. model transition subtype explicitly, then fit positive/negative action
   magnitude within subtype rather than assuming one high-entropy state;
3. require stable March and April economics with paired lower bounds above
   zero before any portfolio constraints are replayed.

### 2026-07-30 — complete-group rank normalization does not repair Wait10 transfer

`cross_era_wait10_rank_normalized_ablation_20260730_v1` implements the required
score normalization without using the frozen selected subset to define its
coordinates.  Descending percentile ranks and z-scores for base and residual,
plus residual-minus-base rank, are computed inside every complete
timestamp-side candidate group in both eras.  Only then are the unchanged
March/April frozen identities and weights joined.

This removes the raw score-level comparison failure, but the action result
remains non-promotable:

| Evaluation | Best frozen top-10 lift | Action weight | Paired 95% interval |
|---|---:|---:|---:|
| March 2025 | +0.164 bps | 2.44% | [0.000, +0.339] |
| April 2025 | +0.094 bps | 0.217% | [0.000, +0.328] |

Both confidence intervals touch zero.  The best event AUC is about 0.645 and
does not exceed the expanded transition-only April-long AUC of 0.678.  No
route is authorized for portfolio replay.

The audit also finds a second domain shift: historical complete
timestamp-side groups contain roughly 7--15 candidates, while current groups
contain about 48.  Current rows remain outside the historical 1st--99th
percentile range at mean rates of about 20.9% for base rank, 29.8% for
residual rank and 16.0% for residual z.  Complete-group ranks remove raw scale
as the dominant confounder, but candidate-group geometry and the economic
meaning of the transition subtype still fail to transfer.

Next action-layer work should use quantile/decile coordinates with explicit
group-size, cutoff-margin and candidate-density context; match or weight
historical rows on that geometry; and fit positive/negative action magnitude
inside the retained causal transition subtypes.  Promotion still requires a
strictly positive paired lower bound in both March and April.

### 2026-07-30 — mapping repair abstains; direct-EV break starts inside March

The pre-registered 2x2 score-mapping experiment is sealed at
`marapr2025_causal_mapping_repair_ablation_20260730_v1`.  It compares pooled
21-day causal isotonic mapping, raw-score plateau resolution, a fixed 25%
positive-Huber shrinkage component, and the combination.  Scores, labels,
costs, candidate identities, one pooled-global top-k rule and the separate
action layer all remain frozen.

Neither source has a passing mapping arm.  For base, raw top-10 net is
-31.07/-112.72/-33.94 bps on March 3--19, March 20--31 and April.  The best
mapped values are -43.10/-102.45/-48.57.  For direct q25, raw is
+20.97/-96.44/-93.24 bps; isotonic is -0.29/-98.69/-109.10 and the
isotonic--Huber blend is -3.16/-97.91/-114.45.  Residual control is
-9.57/-60.43/-24.32 and remains the strongest confirmation incumbent.

Raw-score tie-breaking fixes arbitrary ordering inside exact isotonic
plateaus, while the Huber blend restores more than 91% unique scores.  Those
mechanical repairs do not fix the book: mapped base calendar coverage drops
and concentration rises, and direct mapped arms are 21--24 bps worse than
raw in the selection window with paired intervals below zero.  The sealed
selection decision is `ABSTAIN` for both base and direct q25.  Portfolio
replay and promotion remain disabled.

The new diagnostic boundary is March 20, not April.  Direct raw q25 moves
from +20.97 bps before the boundary to -96.44 bps after it; the latter book
is 85.2% short with -100.93 bps short conditional economics.  April confirms
the failure, but changing side allocation alone does not repair it.  Treat
this as a regime-dependent conditional-payoff/trust problem.

Immediate continuation:

1. join the fixed identical-row bridge to causal pre-entry transition and
   candidate-context fields;
2. explain direct-minus-residual economics across March weeks and the
   March-20 boundary, including capture, adverse movement and costs;
3. pre-register a bounded regime-trust gate that can route to direct q25 only
   when causally supported and otherwise retains residual;
4. fit or select that gate only on earlier/held-out evidence and preserve the
   reused March/April periods as diagnostics, not promotion evidence;
5. keep timing, MAE, target-price and wait actions outside this gate.

Integrity is closed: a clean temporary rerun reproduced all nine output
hashes, the full focused lineage suite passes 24 tests, and the recorded
runner and positive-Huber source hashes match disk.

### 2026-07-30 — March regime boundary is learnable; direct-head trust is not

Two sealed artifacts continue the mapping result:

- `marapr2025_direct_residual_regime_trust_diagnostic_20260730_v1`;
- `marapr2025_direct_residual_regime_break_learnability_20260730_v1`.

The first binds all 136,074 direct/residual identities to 21 causal soft
regime/transition fields at signal `__ts__`, with strict pre-March OOF
provenance.  It excludes OOD, state/cluster/destination IDs, post-entry
geometry and all action fields.  Direct/residual selection overlap is very
low: Jaccard 6.4%, 7.9% and 10.0% in March 3--19, March 20--31 and April.
Direct-only economics change +12.85 -> -100.63 -> -107.96 bps, while
residual-only is -21.85 -> -58.47 -> -23.67.  This is a direct selection
failure on top of a broad market payoff deterioration, not a remapping issue.

The second uses fixed balanced logistic diagnostics with no HPO or feature
selection.  March-boundary recognition uses shuffled UTC-day groups; direct
trust is side-local with seven-day groups.  Trajectory
probability/entropy/margin recognises the March-20 boundary at AUC 0.700
(every fold 0.676--0.894), but the best direct-over-residual trust AUC is only
0.536 long and 0.532 short.  Every short arm's highest-probability decile
still has negative direct advantage.  Current soft regime fields can
recognise the state change; they cannot infer which ranking source will be
economically correct.

Do not build a policy gate from these probabilities.  Keep residual as
incumbent and trajectory as diagnostic transition context.  The next required
dataset is an older identical-row side-local OOF ledger containing both
direct-q25 and residual scores plus inference-parity causal market mechanics.
Use it to train separate incremental-capture, cost-clearing and adverse-loss
heads, then a bounded direct-versus-residual trust output.  Include
score-rank conflict, cutoff margin and candidate-group geometry.  Promotion
still needs a later untouched common cohort; action layers and portfolio
replay remain separate and disabled.  Clean reruns reproduce both artifacts,
and the focused lineage suite passes 33 tests.

### 2026-07-30 — next trust ledger requires an H12 residual rebuild

The May--July readiness audit found 125,551 exact common rows with a genuine
direct-q25 score, not a proxy.  Its May/June/July fold cutoffs are causal and
the score is bit-identical to the sealed q25 challenger output.  Preserve it.

The residual score beside it is not same-target evidence: it is OOF but was
trained on a legacy 24-hour fixed-cost residual target, whose label resolves
12 hours after the exact current-policy H12 endpoint used for evaluation.
Do not train or assess a direct-versus-residual policy gate on that pair.

Next, use the 127,777 raw-score identities, freeze direct q25, and rebuild
only a side-local chronological-OOF residual on exact H12 current-policy net
with the existing May/June/July cutoffs and strict resolved-label audit.
Then attach causal context at signal time and form the exact intersection.
Use 125,551 rows only if mapped q25 is explicitly required.  Apply the frozen
trajectory neutral-fill/availability contract where its 2026 sidecar is
missing.  The result remains diagnostic because May--July has already been
reused; a later untouched common cohort is still necessary.

### 2026-07-30 — exact-H12 rebuild completed; no promotion

`exact_h12_side_local_residual_oof_20260730_v2` replaces v1 as the
authoritative diagnostic because its long/short pair is selected jointly on
the actual pooled-global policy.  It contains 127,777 May--July candidates,
uses resolved February--March 2025 labels for training and April for frozen
selection, and reproduces every row/table/model/map/contract hash.  Long uses
`legacy_capacity_64` at residual blend 0.75; short selects blend 0.0, so no
incremental short residual survives.  Exact-residual global top-10 is
-67.88/-104.26/-148.98 bps in May/June/July and is not promotable.

The earlier rising February--April IC is not evidence that execution EV
should rise monotonically.  It is native 24-hour alpha-target IC.  Exact-H12
net IC is only 0.090/0.093/0.143 long, while raw-base top-10 gross is
+49.38/+17.05/+41.86 bps against approximately 100 bps of cost.  April does
recover gross ordering relative to March, but remains economically negative.
March loses positive-payoff size and prevalence; April direct EV additionally
over-allocates to a severely negative short book.

`exact_h12_residual_regime_transfer_diagnostic_20260730_v1` attaches only
frozen signal-time regime, transition and trajectory context.  Combined
context recognises July at day-group OOF AUC 0.792, but cannot reliably learn
which score to trust: residual-over-base AUC is 0.435/0.521 and
direct-over-residual 0.476/0.577 long/short.  Treat the July signature as
state context for later features or weighting, not as an economic route.

`exact_h12_residual_recent_ev_mapping_20260730_v1` applies the canonical
causal 21-day map before pooled-global top-k.  Mapped exact-residual economics
are -72.82/-108.87/-180.00 bps, and July selection becomes 99.54% short.
Thus neither mapping nor portfolio constraints can rescue the present score.
Do not replay or promote it.

The next material architecture should learn cost clearance, favourable
capture and adverse-loss severity separately per side, supported by
peak-MFE/future-slope signals, before producing a bounded execution-EV
ranking.  It needs older identical-target OOF ledgers or an untouched cohort,
full production feature selection/HPO and candidate-context ablations.  The
production inference route already fails closed on missing/non-finite selected
inputs, but this diagnostic has no final refit or live loader.  A future
exact-H12 package must replay its declared complete-case or named-native-
missing policy before its ordered features, sources, base-score and map hashes
can be bound into that route.  Timing, MAE, target-price and wait decisions
remain outside this ranking head.

### 2026-07-30 — older current-map common ledger

`marapr2025_exact_h12_current_mapping_20260730_v1` materialises 140,682
identical March--April rows across base, strict residual and direct q25, all
with exact-1m current-policy H12 labels. It binds direct OOF provenance to
the original chronological `old_march`/`old_april` fold recipe and publishes
score availability at the decision time; per-fold binaries remain honestly
unavailable but the output, frozen state, final model and recipe hashes bind.
The current 21-day UTC/min-500/side-shrunk map leaves 2,208 warm-up rows
unmapped and never substitutes a raw fallback.

The map is not a repair. At pooled global top-10, residual is
-27.14/-24.32 bps raw versus -13.48/-31.38 mapped in March/April; direct q25
is -20.18/-93.24 raw versus -42.04/-77.38 mapped. Base is
-65.61/-33.94 raw versus -56.32/-43.12 mapped. Each mapping result remains
negative and reallocates side mass, so portfolio replay stays disabled.

March--April direct-model selection was historical/reused, therefore this
artifact is support for component/candidate-context diagnostics only. It is
not untouched validation, a new HPO surface, or promotion evidence.

### 2026-07-30 — older current-map common ledger

`marapr2025_exact_h12_current_mapping_20260730_v1` materialises 140,682
identical March--April rows across base, strict residual and direct q25, all
with exact-1m current-policy H12 labels.  It binds direct OOF provenance to
the original chronological `old_march`/`old_april` fold recipe and publishes
score availability at the decision time; per-fold binaries remain honestly
unavailable but the output, frozen state, final model and recipe hashes bind.
The current 21-day UTC/min-500/side-shrunk map leaves 2,208 warm-up rows
unmapped and never substitutes a raw fallback.

The map is not a repair.  At pooled global top-10, residual is
-27.14/-24.32 bps raw versus -13.48/-31.38 mapped in March/April; direct q25
is -20.18/-93.24 raw versus -42.04/-77.38 mapped.  Base is
-65.61/-33.94 raw versus -56.32/-43.12 mapped.  Each mapping result remains
negative and reallocates side mass, so portfolio replay stays disabled.

March--April direct-model selection was historical/reused, therefore this
artifact is support for component/candidate-context diagnostics only.  It is
not untouched validation, a new HPO surface, or promotion evidence.

### 2026-08-01 — local native-source backfill readiness

Before acquiring more data, the two local orderbook-hourly roots were scanned
with `scripts/audit_native_l2_backfill_readiness.py`. The resulting
`native_l2_backfill_readiness_20260801_v1` artifact inventories 568 files and
10,373,441 rows: 6,928 exact `kraken_futures_l2_snapshot` rows in 73 product
files and 10,366,513 explicit `local_ohlcv_summary` proxy rows. Exact native
coverage starts on 2026-07-11, whereas the declared candidate panels begin on
2026-04-01. The full-window source gate fails closed; no labels, scores,
models, or portfolio outputs are produced.

Proxy orderbook summaries remain excluded from native depth/flow claims. A
future backfill must preserve exact product identity and factual snapshot or
publication timing, then pass the existing backward/as-of overlap audit before
strict OOF labels, HPO, or global-top-k economics are allowed.

### 2026-08-01 — dense raw native-L2 source extension

The broader local source scan found raw per-level native snapshots in
`spread_snapshots/orderbook_history`, in addition to the lower-density hourly
sidecar. `scripts/materialize_native_l2_continuation_from_snapshots.py`
aggregates the ten canonical daily files with a vectorized reducer and keeps
the observed timestamp as the causal availability time. The resulting v3
sidecar has 51,778 aggregated snapshots across 303 products from July 11--23;
50,334 rows have a valid preceding snapshot within the two-hour bound.

The corrected candidate overlap is 3.297% for the canonical handoff and
57.292% for the July 20--23 retrospective bridge, but remains zero for the
May--July exact-H12 and A-grade strict-forward panels. This is still
research-only source coverage. Historical native data before July 11 remains
required before strict OOF labels, HPO, or global-top-k economics can use the
features.

### 2026-08-01 — daily native-L2 coverage closure

The full local source inventory was rerun against `data_perp` using the v3
corrected overlap contract. It scans 71,135 files / 327,133,322 rows and
finds 2,865,522 exact native rows. Those rows occur on only ten UTC days:
July 11--16, July 18, and July 21--23; July 17, 19, and 20 are missing.
Because the candidate requirement starts April 1, the historical backfill
gate remains fail-closed. The detailed artifact is
`data_perp/artifacts/native_l2_backfill_readiness_20260801_v3/`.

Do not use the partial cohort for labels, strict OOF fitting, HPO, global-top-k
economics, or portfolio replay. Acquire longer factual native history, then
rerun the sidecar and exact-product backward as-of overlap before reopening
the execution workstream.

### 2026-08-01 — current-run stop reconciliation

The current-run stop audit is sealed at
`data_perp/artifacts/current_run_stop_audit_20260801_v1/`. A process-table
check found no active Ares training, collector, materializer, or audit
process. Registry PID 1026 was a stale PID reuse belonging to macOS
`imagent`, not the registered Ares collector; the safety wrapper sent no
signal and preserved that unrelated process. The registry entry is marked
`stale_pid_reuse`, and no new roadmap run was started.

### 2026-08-01 — native-L2 acquisition manifest

`data_perp/artifacts/native_l2_backfill_request_20260801_v1/` is the current
fail-closed acquisition handoff. It contains 25,343 candidate product/day
pairs over April 1–July 23, of which 952 are covered by the partial native
sidecar and 24,391 require factual native-L2 backfill. It contains no labels,
scores, costs, portfolio fields, or fitted-model outputs.

## 2026-08-01 — target–feature–execution audit closure

`data_perp/artifacts/target_alignment/alignment_audit_20260801_v2/` is the
canonical reconciliation pack for the cached roadmap. It materializes the
target contract, label dictionary/support report, feature eligibility and
lineage views, fold/OOF manifests, candidate-level target-ablation
predictions, supportive-label policy summary, monthly/side diagnostics, and a
correctness report.

The audit passes 55 of 57 checks. Exact-H12 identity, horizon and availability,
feature causality, cost-once accounting, frozen policy/cost identity,
chronological folds, candidate-level supportive OOF ordering, explicit
execution-feature lineage, aggregate OOF ordering, and pooled global top-k
selection are evidenced. The versioned canonical v3 target pack at
`data_perp/artifacts/root_cause_exact_h12_execution_target_pack_20260801_v3/`
now carries
the explicit supportive-label metadata and dictionary fields. The audit fails
closed only on economics: the best
supportive pooled global top-10% result is -113.44 bps net and the best
exact-H12 target-ablation result is -104.05 bps net. Partial native-L2 history
also cannot satisfy the April–July candidate-window gate; 24,391 product/day
pairs remain to be backfilled.

This is a research audit, not a promotion result. No target, supportive head,
execution model, timing/wait action, portfolio constraint, or partial native
cohort is promoted from it.

## 2026-08-01 — economic headroom audit

`data_perp/artifacts/exact_h12_economic_headroom_diagnostic_20260801_v1/`
closes the remaining ambiguity in the negative economic gate. On the same
exact-H12 candidate population and frozen execution/cost contract, the oracle
pooled global top-10% is **+468.27 bps gross**, **102.34 bps cost**, and
**+365.93 bps net**. The best model book, the frozen base-opportunity control,
is **-4.07 bps gross**, **99.98 bps cost**, and **-104.05 bps net** at top-10%.
At top-1% its gross is **+91.67 bps**, so the population has a narrow positive
tail but the model does not recover enough of it at the required book size.

At a hypothetical zero cost the best model top-10% is still **-4.07 bps**.
This rules out treating the failure as a fee-only problem: the primary
bottleneck is economic-tail ranking from features/labels/model calibration.
The audit now uses the actual policy-selection score
`calibrated_expected_net_bps` with the ablation's descending stable sort and
materialized-row tie rule, and verifies that its top-10 metrics reproduce the
authoritative target-ablation CSV exactly. Raw-score ranking diagnostics are
not used for conclusions.

The result is diagnostic only. It does not promote a target or execution
head: the alignment pack remains 55/57 with both economic checks negative,
and native-L2 backfill remains outstanding for 24,391 April–July
product/day pairs.

## 2026-08-01 — requirement-level roadmap audit

`data_perp/artifacts/updated_roadmap_requirement_audit_20260801_v1/` provides
the direct requirement-to-evidence matrix for the cached roadmap. It records
12 passing requirements, 5 failed economic acceptance gates, and 1 external
native-L2 block. The passing rows cover the exact-H12 contract, shared frozen
policy/cost/geometry, base-versus-execution target separation, five-head
support metadata, label prohibitions, layer-specific feature eligibility,
strict candidate-level OOF timing, one pooled global top-k policy, the
30-cell target/support ablation matrix, and reproducible output hashes.

The five failed rows are not hidden by the valid mechanics: pooled top-10 net
is negative, the latest month and both sides are negative, no paired bootstrap
challenger has a positive lower bound, and supportive OOF predictions do not
improve the exact-net book. This is the authoritative roadmap completion
matrix; it is research-only and performs no new fitting.
