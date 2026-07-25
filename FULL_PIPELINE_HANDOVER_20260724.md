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

All comparisons must use identical rows and costs.

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
head promotion remains gated by repaired-label execution-EV ablations
Execution-EV model: exact-policy OOF and causal post-21d global top-10 complete;
all current arms rejected because every net top-10 and admitted subset is negative
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
