# Ares Full Pipeline — Actionable Roadmap

Date: 2026-07-24
Source: `FULL_PIPELINE_HANDOVER_20260724.md`
Repository: `/Users/remyroche/Documents/Ares`
Storage/model timezone: UTC
Status: execution-EV research incomplete; policy and live promotion blocked

## 1. Purpose

This document is the execution control plane for the handover. The handover
remains the detailed technical and historical reference; this roadmap defines
the order of work, ownership roles, required outputs, hard gates, and stop
conditions.

The target is not “train all unfinished models.” The target is to determine,
with leakage-safe side-local OOS evidence, whether CatBoost path probabilities
and the five auxiliary path heads improve executable top-10% EV over the
existing alpha stack. Only a demonstrated winner advances to bundling, policy,
replay, and inference parity.

## 2. Fixed Architecture and Scope

The following architecture is locked:

```text
LONG:  long base ─┬─> long residual alpha ──────────┐
                  ├─> long CatBoost ────────────────┤
                  └─> 5 long auxiliary heads ──────┤
                                                   └─> long execution EV
                                                       -> optional entry timing
                                                       -> long policy stream ─┐
                                                                             ├─> global auction
SHORT: short base ─┬─> short residual alpha ────────┐                        │
                   ├─> short CatBoost ──────────────┤                        │
                   └─> 5 short auxiliary heads ─────┤                        │
                                                    └─> short execution EV   │
                                                        -> optional entry timing
                                                        -> short policy stream ─┘
```

Residual alpha, CatBoost, and the five auxiliary heads are parallel,
side-local inputs to execution EV. CatBoost and the auxiliary heads are not
downstream of residual alpha.

Until the global portfolio auction, long and short must have independent:

- feature selection;
- HPO studies;
- fitted models;
- OOF streams;
- probability/EV calibration;
- geometry, admission, and sizing estimates;
- manifests and hashes.

A side column in a shared fitted model does not meet this requirement.

### In scope

- Migration/source recovery and deterministic validation.
- Side-local Pack-B and residual-alpha provenance.
- Pack-B-derived top-40 candidate regeneration.
- Two seven-class CatBoost pipelines.
- Ten auxiliary models: five heads × two sides.
- Strict side-local execution-EV handoffs.
- Direct/residual execution-EV and input-family ablations.
- Winner final refit and two-sided inference bundle.
- Policy/replay/parity work only after promotion.

### Deferred or conditional

- Entry timing: start only after stable execution-EV OOF; validate or explicitly
  omit before replacing the current policy.
- Geometry, sizing, portfolio re-optimization: only if the promoted score
  materially changes ranking or admission.
- Live integration: blocked until full replay/inference parity.
- New predictive regime work: out of scope.

## 3. Current Baseline

| Component | Status | Roadmap treatment |
|---|---|---|
| Pack-B directional base | Available, per-side manifest | Historical comparator only; rebuild pre-March AE/FS/HPO state before canonical OOF |
| Shared-store base/top-40 | Available | Benchmark/research source only; invalid as final population |
| Residual alpha | Available, `canonical: false` | Preserve benchmark; audit/refit against matching Pack-B side |
| Path labels | Materialized | Verify causal/schema/hash contract |
| CatBoost FS/HPO and pooled geometry | Benchmark evidence | Rerun FS, geometry, HPO, balancing independently per side |
| Final CatBoost | Missing | Build OOF and final bundles |
| Auxiliary selection checkpoints | Partial, 3/5 only | Reuse only if fingerprint remains valid; otherwise new run |
| Auxiliary OOF/final models | Missing | Build ten OOF streams and ten final models |
| Execution-EV labels | Materialized | Preserve; validate cost and causal contract |
| Existing alpha execution OOF | Materialized | Preserve as benchmark; regenerate if side-local lineage fails |
| Strict execution handoff | Missing | Build separately for long and short |
| Direct/residual execution EV | Not run | Run controlled ablation |
| Current policy | Executable historical benchmark | Do not overwrite |
| New live deployment | Blocked | Remains blocked through parity gate |

Repository recovery risk is high: the current tree contains at least 154
tracked modifications and 552 untracked paths. A clean clone is not a recovery
method.

## 3A. Implementation Verification — 2026-07-24

This audit distinguishes an implemented safeguard from a roadmap intention.
“Partial” means the code has useful foundations but does not yet meet the locked
side-local production contract.

| Requirement | Roadmap coverage | Current implementation | Verdict |
|---|---|---|---|
| Feed execution EV into `simple_policy_optimiser` | P1 names the integration but previously lacked an interface design | Existing runner can rerank an arbitrary candidate score via `--rank-score-col` and can disable legacy regime calibration, but no canonical execution-EV candidate adapter or waterfall exists | PARTIAL — add the P1 integration contract below |
| Execution-EV OOF/OOS assessment | E1 requires OOF comparisons | Outer expanding purged folds; per-fold inner HPO, early stopping, and train-OOF calibration avoid outer validation outcomes | PASS for assessment provenance |
| Execution-EV strictly side-local fitted state | E1 requires it | Models and HPO split by side, but fold calibration combines both sides and the final bundle stores a pooled `__global__` mapper | FAIL — block training/promotion until separate maps exist |
| Execution-EV feature selection and HPO | E1 requires independent FS/HPO | HPO exists for the full-feature arm; there is no explicit FS stage, ablation arms use defaults, and final parameters are inherited from the last outer fold | PARTIAL — implement frozen side-local FS and final authorized HPO |
| Timing OOF/OOS assessment | T1 requires side-local OOF | Strong row-level upstream OOF fold/cutoff validation, expanding purged outer folds, train-only inner isotonic maps, and separate final refit | PASS for assessment provenance |
| Timing strictly side-local fitted state | T1 requires it | Fold models/calibration split by side, but model HPO and action-policy HPO are run once before the side loop | FAIL — move both searches inside each side |
| Timing feature selection and feature contract | T1 requires side-local FS | Feature family/provenance checks are strong; no feature-selection stage exists | PARTIAL — add side-local task-aware FS |
| Timing risk/benefit/cost semantics | T1 must compare waiting with entering now | Filled price improvement enters through conditional delta EV; no-fill opportunity loss and adverse-first risk are explicit; entry/exit costs are reconciled and double-accounting is rejected | PASS for research labels/objective; PARTIAL for live execution because the model emits an action template/ATR offset, not a validated order price |
| Timing seven-class CatBoost contract | T1 assumes final taxonomy | Timing imports legacy eight-class `PATH_SHAPE_TYPES` and requires that probability count | FAIL — replace with explicit seven-class schema/hash from the CatBoost bundle |
| Five auxiliary heads exist and are per-side | A1/A2 require ten streams/models | Five exact targets exist; models, HPO, OOF, and bundles split by side | PASS after selection-stage correction |
| Auxiliary feature selection is per-side | A1/A2 requires all selector stages per-side | MDA is side-local, but the current contract explicitly performs global univariate/Relief pre-screening first | FAIL — move the entire pre-screen into each side |
| Auxiliary targets are learnable/economic | A1/A2 names metrics but previously lacked target QA gates | Targets are ATR-normalized, log transformed, bounded, supported by economic path labels and weighted training loss; mixture/censoring/redundancy risks remain | PARTIAL — add head-specific QA and challenger targets below |
| CatBoost FS/HPO/geometry/final model are per-side | C1/C2 requires two pipelines | Current classifier runner performs one pooled selection, HPO, OOF, and final fit with side-stratified samples; existing geometry evidence is pooled | FAIL — run two independent end-to-end pipelines |

### Verified OOF rules for stacked meta heads

Execution-EV training evidence is acceptable only when every upstream predictive
feature carries row-level OOF/frozen provenance and its source training cutoff
precedes the scored decision. The execution-EV head itself must emit outer-fold
OOF predictions with the fold and train cutoff recorded per row.

The current execution-EV provenance object only asserts `oof_or_frozen`; it does
not verify upstream row-level folds or source cutoffs. Extend it to the stronger
timing-style contract with `oof_fold_col`, `source_train_cutoff_col`,
`available_at_col`, and inference-only `frozen_bundle_id`. Enforce a common
upstream OOF cohort across alpha, CatBoost, and all auxiliary streams, or persist
and validate an explicit fold-correspondence map. Independent declarations that
each field is “OOF” are not sufficient.

Timing training evidence is acceptable only when:

- the protected execution-EV input is the selected execution-EV **OOF**
  prediction, never its all-row final-refit score;
- alpha, residual, CatBoost, and auxiliary inputs carry row-level upstream OOF
  fold and source-train-cutoff columns;
- every source cutoff is no later than the timing outer-fold training cutoff
  and strictly before the scored decision;
- timing labels and counterfactual paths are train/report-only;
- model HPO, action-grid HPO, and isotonic maps use only authorized training or
  inner-OOS rows;
- the final timing refit is excluded from OOF metrics and policy optimization.

“Frozen” is sufficient for inference, not for training-time OOF evidence from a
supervised upstream model. Add tests that reject an upstream final-fit bundle ID
when row-level OOF lineage is required.

### Feature adequacy requirements for execution EV

The existing required families cover the core stack: five path-head predictions,
the complete CatBoost probability vector and entropy, alpha score, base
archetypes, uncertainty, and leaf support. Before feature selection, expand and
validate the candidate registry so it also exposes, where causally available:

- base and residual score, rank, margin, mapped EV, and disagreement;
- all five auxiliary predictions, fold dispersion/uncertainty, support, and
  missingness indicators;
- seven ordered CatBoost probabilities, max probability, normalized entropy,
  top-2 margin, favorable mass, adverse mass, and class-support diagnostics;
- frozen AE/GMM posteriors, entropy, distance, reconstruction error, speed,
  acceleration, OOD and drift;
- side-local archetype reliability, leaf support, and causal recent signed EV
  residual/hit-rate surprise;
- decision-time spread, friction, liquidity, volatility/ATR, and capacity
  context in portable units;
- horizon/entry/cost identity fields as protected context rather than accidental
  numeric features.

For each side and for direct/residual targets:

1. validate live reproducibility before selection;
2. remove constants, near-constants, excessive-missingness fields, duplicates,
   and outcome-like fields;
3. run univariate/Relief screening by supported archetype;
4. correlation/semantic-family prune;
5. run automatic-stopping MDA using global, macro-archetype, worst-supported
   archetype, and top-10 EV objectives;
6. freeze the selected contract on the authorized reference fold;
7. run side-only chronological purged HPO;
8. confirm the selected feature family through identical-row ablations.

The full-feature and ablation arms must either use separately authorized HPO or
a predeclared matched-parameter comparison. The current “HPO full arm versus
default-parameter ablations” is not sufficient attribution evidence.

### Feature adequacy requirements for entry timing

The current strict family contract is a good minimum, but timing selection must
prioritize action value rather than generic fit. Candidate inputs should include:

- selected execution EV and its uncertainty/common-unit mapping;
- alpha/residual disagreement and rank margin;
- five auxiliary predictions, especially time-to-MFE, pre-MFE MAE, adverse
  turn timing, and slope;
- seven CatBoost probabilities and aggregate confidence/risk;
- side/archetype, OOD, leaf support, spread, liquidity, ATR/volatility, and
  short-horizon causal microstructure;
- distance from decision price to candidate adverse-limit offsets;
- time-of-bar/session context only when UTC-causal and inference-portable.

Run side-local FS separately for the fill, adverse-first, and conditional
filled-delta tasks, then test a shared side-local union against task-specific
feature sets. HPO must be side-local and scored on the combined action utility,
fill Brier/log loss, adverse-first calibration, filled-delta error, missed-trade
cost, and worst-period stability. Eight model trials and eight decision trials
are smoke defaults, not adequate final search evidence.

### Entry-timing objective and price-action boundary

The current timing formulation is directionally correct and must be preserved
as a side-local action-value problem. For each approved action template it
estimates:

- fill probability;
- adverse-first probability;
- conditional net execution-EV change versus entering now when filled;
- missed-opportunity loss when the action is not filled.

Its decision objective must retain the explicit decomposition:

```text
risk-adjusted action utility
  = P(fill) × (net EV now + conditional filled delta EV)
  - P(no fill) × missed-opportunity penalty × max(net EV now, 0)
  - P(fill) × P(adverse first) × adverse-risk penalty
```

The filled delta captures the benefit or harm of the later entry price and
subsequent re-anchored execution path. Fees, spread, slippage, and any
maker/taker assumptions must be included exactly once in both labels and
replay. The adverse-risk penalty is a tunable risk preference, not a trading
cost; report the resulting quantity as `risk_adjusted_expected_utility`, not
pure net EV.

The ML layer may rank only a frozen, side-specific grid of `enter_now`,
`wait_market`, and `adverse_limit` templates. It must not directly emit an
executable order price. Add a separate deterministic timing-execution policy
above the model. Using only current pre-entry state, it converts a selected
adverse-limit template into a price:

```text
raw_limit = reference_price - side_sign × offset_atr × ATR
```

where `side_sign = +1` for long and `-1` for short. The policy must then apply
the declared bid/ask/reference-price convention, side-correct tick rounding,
price-band and non-crossing checks, time-in-force/cancel deadline, stale-quote
checks, and exchange/order constraints. It must also reconcile action-specific
maker/taker fees, spread/slippage, latency, queue priority, partial-fill, and
cancel/replace assumptions.

Initially keep the action and offset grid discrete to control overfitting. A
continuous or interpolated price optimizer is a separate challenger and may
advance only on nested OOF evidence. Select a wait or adverse-limit action only
when its calibrated risk-adjusted utility clears `enter_now` by a predeclared
margin and confidence, support, liquidity, fill-feasibility, staleness, and
cost-reconciliation gates pass. Otherwise use `enter_now` or reject/abstain
under the existing admission policy.

Persist for every recommendation the complete scored action table, selected
template, wait/expiry time, ATR offset, computed executable price, reference
price and ATR timestamps, tick rule, cost components, confidence/support
diagnostics, and model/policy versions. This makes the specific price
suggestion reproducible without moving order mechanics into the ML model.

### Head-specific auxiliary target review and improvements

All target QA must be reported independently for long and short, month,
base-archetype, symbol, volatility/spread bucket, and score tail. At minimum
record rows, missingness, zero/reached/censored share, min, p01, p05, median,
p95, p99, max, IQR, unique values, cap-saturation share, and simple-baseline
error/IC. A head is not accepted merely because its model trains.

#### `peak_mfe_12h_atr`

Current target:

- complete 12h favorable excursion;
- values below `max(1.5 ATR, 1.5% return)` become zero;
- ATR-normalized, clipped at 10 ATR, trained as `log1p`.

Strengths: portable magnitude, bounded tails, and direct opportunity relevance.

Required improvements/tests:

- quantify the zero mass and 10-ATR cap saturation per side/month/symbol;
- compare against constant, side/archetype median, and alpha-score baselines;
- add a two-part challenger: probability of meaningful MFE plus conditional
  magnitude for reached rows;
- add median/p80 quantile challengers to represent reachable and upside MFE;
- report IC and realized net EV/retention in predicted top 1/5/10%;
- test whether predicted peak adds value after time-to-MFE and slope.

#### `time_to_first_meaningful_mfe`

Current target:

- first passage of `max(1.5 ATR, 1.5% return)`;
- unreached valid rows are right-censored at 12h;
- trained as `log1p(hours)`.

Strengths: stable first-passage question and direct timeout relevance.

Risk: a true late hit and “never hit” both appear at the 12h boundary.

Required improvements/tests:

- report reached/censored shares and conditional time distributions per side;
- add a hurdle/discrete-hazard challenger:
  `P(reach by 12h)` plus conditional time or hazards by 2/4/8/12h;
- use censor-aware/AFT or survival metrics as a challenger, not ordinary MAE only;
- score Brier/calibration at 2/4/8/12h and economic EV decay by predicted delay;
- test sensitivity to the 1.5-ATR/1.5% threshold without tuning on final OOS.

#### `mae_before_meaningful_mfe_atr`

Current target:

- maximum adverse excursion through the meaningful-hit bar;
- for unreached rows, adverse excursion over the full 12h path;
- ATR-normalized, clipped at 10 ATR, trained as `log1p`.

Strengths: directly relevant to stop tolerance and entry quality.

Risk: reached and unreached rows have different stopping semantics.

Required improvements/tests:

- report reached versus unreached target distributions separately;
- add a hurdle challenger: reach probability plus conditional pre-hit MAE;
- add p50/p80/p90 quantile-loss challengers for adverse-risk control;
- report calibration at 0.25/0.5/0.75/1.0/1.5 ATR thresholds and ordering versus
  the meaningful favorable hit;
- measure incremental value for stop rate, entry timing, and execution EV after
  peak/time predictions;
- quantify cap saturation and tiny-ATR sensitivity.

#### `bars_before_price_stops_decreasing`

Current target:

- one-based bar position of the worst adverse point before the meaningful hit;
- uses the full horizon when the meaningful hit is not reached;
- entry is bar zero; trained as `log1p(bars)`.

Strengths: side-normalized early adverse timing.

Risk: the name implies confirmed trough detection, while the primary target is
the ex-post adverse argmax; hourly bars also make short waiting decisions coarse.

Required improvements/tests:

- rename the primary semantic contract to
  `bars_to_pre_mfe_adverse_trough`, or redefine it using the already
  materialized confirmed-trough label;
- separate `P(trough before meaningful MFE)` from conditional trough time;
- compare ordinal/discrete-hazard classification at 1/2/4/8/12 bars against
  regression;
- report zero mass, integer-class support, confirmation availability, recovery
  shares, and accuracy by action-relevant horizon;
- evaluate a higher-frequency target for 5/10/20-minute timing decisions; do not
  claim hourly-bar precision for minute-level entry actions.

#### `future_slope_atr_per_hour`

Current target:

- 80% of capped eventual MFE divided by time to first reach 80% of that peak;
- ATR/hour, non-negative, clipped at 10, trained as `log1p`.

Strengths: compact realization-speed/magnitude measure.

Risk: it is mathematically derived from future peak and timing, so it may add
little independent information and may overreact to a single extreme wick.

Required improvements/tests:

- measure correlation, conditional mutual information, and ablation value versus
  predicted peak MFE and time-to-MFE;
- report zero mass, cap saturation, and sensitivity to peak clipping;
- compare robust signed side-relative slope/efficiency challengers based on
  cumulative favorable path at fixed 2/4/8/12h horizons;
- compare a residualized slope target after train-only expected slope from
  peak/time context;
- require incremental execution-EV uplift before retaining this head.

### CatBoost side-local verification contract

Side-stratified sampling, a side feature, or side-level reporting does not prove
side-local training. The final CatBoost implementation must invoke two isolated
runs or an equivalent outer side loop. Each side must persist its own:

- candidate-row and top-40 lineage hash;
- seven-class support/month-support report and taxonomy hash;
- selection sample and selected-feature list;
- geometry sweep study and winner;
- CatBoost model HPO study and winner;
- bounded class-balance mini-HPO and no-balance control;
- OOF fold models/probabilities/fold-cutoff ledger;
- final model, class order, feature order, parameters, and source hashes.

Add fail-closed tests that long and short artifact hashes/study IDs differ where
fitted state is expected to differ, that neither manifest names the pooled
geometry as final, and that a combined report contains no fitted object.

## 4. Roles

Assign one named person to each role before starting compute-heavy work. One
person may hold multiple roles.

| Role | Accountability |
|---|---|
| Roadmap owner | Gate decisions, scope control, status and decision log |
| Data/provenance owner | Hashes, row identity, UTC, folds, labels, cost lineage |
| Alpha owner | Pack-B/residual audit and side-local top-40 regeneration |
| CatBoost owner | Per-side FS, geometry, HPO, balancing, OOF and bundles |
| Auxiliary owner | Per-side pre-screen, five heads, OOF and bundles |
| Execution-EV owner | Strict handoff, direct/residual runs, ablations, report |
| Validation owner | Tests, leakage review, identical-row comparisons, parity |
| Policy owner | Policy/replay integration after research promotion |

The validation owner signs each gate independently of the implementing owner.

## 5. Dependency Graph

```text
R0 Migration integrity
  -> R1 Source durability
  -> R2 Deterministic contracts
  -> R3 Side-local alpha and top-40 provenance
       -> C1 CatBoost long  ─┐
       -> C2 CatBoost short ─┤
       -> A1 Auxiliary long ─┤  (parallel)
       -> A2 Auxiliary short ┘
                             -> H1 Strict execution-EV handoffs
                             -> E1 Direct/residual + feature-family ablations
                             -> D1 Promote or reject
                                  -> B1 Final bundle
                                  -> T1 Entry timing: validate or omit
                                  -> P1 Policy/replay
                                  -> L1 Live parity and promotion
```

Critical path:

```text
R0 -> R1 -> R2 -> R3 -> max(C1/C2/A1/A2) -> H1 -> E1 -> D1 -> B1 -> P1 -> L1
```

Do not begin CatBoost or auxiliary final training before R3. Do not begin
execution-EV fitting before every required OOF stream passes H1.

## 6. Blocking Decisions to Lock Before Training

The handover contains qualitative promotion language. Record numeric rules in
an immutable experiment manifest before model selection; never choose them
after viewing final OOS results.

| ID | Decision required | Current state |
|---|---|---|
| DEC-01 | “Remain close” tolerance for Pack-B vs shared-base benchmark | Locked in `config/full_pipeline_decisions_20260724.json` |
| DEC-02 | Minimum residual-alpha improvement per side and required fold consistency | Locked in decision manifest |
| DEC-03 | CatBoost dominant-class, class-share divergence, and entropy collapse limits | Locked in decision manifest |
| DEC-04 | Formula/weights for combined CatBoost ML + economic objective | Locked in decision manifest |
| DEC-05 | Seven-class support failure policy: hard stop or predeclared merge | Locked: hard stop for CatBoost side, no post-hoc merge |
| DEC-06 | Execution-EV minimum top-10 uplift and statistical/economic confidence rule | Locked in decision manifest |
| DEC-07 | Allowed worst-week/worst-month degradation trade-off | Locked in decision manifest |
| DEC-08 | Common EV unit, side-map equivalence test, and global-top-k reporting rule | Locked in decision manifest |
| DEC-09 | Exact fold calendar, embargo, and untouched replay period | Locked in decision manifest |
| DEC-10 | Entry-timing promotion requirement or explicit omission rule | Locked: validate after stable E1; retain `enter_now` on failure |

Hard requirements already fixed and not subject to tuning:

- CatBoost target support is at least 1% overall and 0.5% per month, per side,
  unless DEC-05 defines a pre-fit merge.
- Decision timestamp is signal timestamp plus one signal timeframe.
- First path timestamp must be at or after the decision timestamp.
- The Pack-B directional-base target resolves over 96 × 15-minute bars, or 24
  hours after its decision. Auxiliary, CatBoost, execution-EV, and timing path
  targets use 12 hours. Every stage must bind and hash its actual horizon; a
  generic “current path horizon” is not sufficient provenance.
- With the one-hour decision delay, the locked Pack-B outer-fold signal purge
  is 25 hours: `signal_timestamp < validation_start - 25 hours`, equivalently
  `decision_timestamp < validation_start - 24 hours`. The final accepted
  Pack-B training label must resolve strictly before the validation boundary.
- Pack-B AE/GMM fitting, feature selection, and HPO must use only labels whose
  actual 24-hour resolution timestamp is strictly before
  `2026-03-01 00:00 UTC`. The recovered June/July state, 55/37 feature
  contracts, and promoted parameters are comparator inputs, not canonical
  fitted state.
- Exact join key is `__ts__, __symbol__, side_name`.
- OOF assessment cannot use a final-refit prediction.
- Costs are recorded once.
- Sides cannot share fitted state before the portfolio auction.

## 7. Work Packages

### R0 — Migration Integrity

Owner: Data/provenance
Depends on: none
Can run in parallel: read-only environment checks and artifact inventory

Tasks:

- Copy the entire worktree, hidden files, `.git`, logs, and every P0 artifact.
- Preserve `/Users/remyroche/Documents/Ares` when possible.
- Generate and compare source/destination checksums for every P0 artifact.
- Verify free disk for OOF streams and temporary matrices.
- Inspect JSON/text checkpoints for absolute paths; never broad-replace binary
  pickle, joblib, SQLite, or Parquet content.
- Confirm no training jobs are active.
- Smoke-open Parquet footers, feature store, AE/GMM pickle, and model bundles.
- Validate NumPy, pandas, PyArrow, LightGBM, CatBoost, scikit-learn, Optuna,
  joblib, and Numba.

Deliverables:

- `migration_inventory.json`
- `migration_checksums.sha256`
- `migration_verification.md`
- Read-only smoke log

Gate R0:

- All P0 hashes match.
- Every required artifact opens.
- No checkpoint points to a missing source.
- Dirty and untracked source is preserved.
- Disk and environment checks pass.

Stop if any required hash, source file, or serialized model is missing.

### R1 — Durable Source and Stage Manifest

Owner: Roadmap owner + Data/provenance
Depends on: R0

Tasks:

- Separate source changes from generated/runtime state without unrelated cleanup.
- Commit or create an immutable archive of the new path, CatBoost, auxiliary,
  execution-EV, and timing modules, scripts, and tests.
- Record the exact source revision/archive hash.
- Create a stage manifest schema containing:
  - source revision;
  - command and CLI arguments;
  - Python/package environment;
  - feature-store and upstream artifact hashes;
  - row/key/fold hashes;
  - side;
  - label, entry, horizon, purge, embargo, and cost contracts;
  - selected feature/parameter/model/calibrator hashes;
  - final-fit cutoff and OOF exclusion flag.

Deliverables:

- Recoverable source revision or archive
- Versioned stage-manifest schema and first repository-level stage manifest

Gate R1:

- Active implementation is recoverable without the source laptop.
- Every new run can point to immutable source and input identities.

### R2 — Deterministic Contract Validation

Owner: Validation
Depends on: R1

Tasks:

- Run the focused suite listed in the handover.
- Run broader auxiliary, CatBoost/geometry, label, and execution adapter suites.
- Add or confirm assertions for:
  - UTC and exact causal timestamp invariant;
  - exactly seven CatBoost classes and stable order;
  - correct favorable/adverse probability masses;
  - absence of abandoned transforms, memberships, and weighting fields;
  - exact one-time cost application;
  - strict OOF provenance and no outcome inputs;
  - authorized pre-March side-local AE/GMM input order and serialized-state
    reuse within each side's outer folds;
  - purge and embargo across overlapping label paths.

Deliverables:

- Test log with command, source hash, pass/fail counts
- Contract assertion report

Gate R2:

- All deterministic tests pass.
- Class schema and cost/horizon contracts hash consistently across stages.
- No leakage, OOF, UTC, frozen-state, or cost exception remains unexplained.

### R3 — Side-Local Alpha and Top-40 Provenance

Owner: Alpha + Data/provenance
Depends on: R2

Tasks:

- Inspect the serialized Pack-B bundle, not just filenames:
  - `model_side_scope=per_side`;
  - distinct long/short models;
  - distinct selected features and HPO parameters;
  - own-side OOF scoring;
  - per-side source/model/contract hashes.
- Audit residual-alpha training keys against the matching Pack-B side stream.
- Refit residual experts if they currently inherit the shared-store base stream.
- Regenerate top-40 independently within each UTC timestamp and side from
  matching Pack-B OOF scores.
- Recompute integer rank and selection mask with deterministic ties.
- Produce side-specific support and attrition reports.
- Preserve shared-store top-40 and current alpha execution OOF as historical
  comparators only.
- Regenerate the alpha execution OOF if it cannot prove Pack-B + matching-side
  residual lineage.
- Treat the seven saved monthly Pack-B fold models as a historical comparator
  only. Their windows and train-cutoff evidence do not match DEC-09.
- Preserve the recovered exact AE/GMM state, promoted 55-feature long /
  37-feature short contracts, and parameters only as historical comparator
  evidence. Their June/July selection/reference windows cross the locked
  pre-March resolution cutoff and are ineligible for canonical OOF.
- Build a fresh pre-March authorization ledger. Fit AE/GMM state independently
  by side on the authorized pre-March reference interval, then run feature
  selection and HPO independently for long and short using only labels whose
  actual 24-hour resolution is strictly before `2026-03-01 00:00 UTC`.
- The immutable population and 18 fixed-calendar side-cohort ledgers are now
  materialized under
  `data_perp/artifacts/packb_pre_march_population_20260724_v1`: 3,429,788
  authorized rows, 1,197,582/1,199,653 long/short AE-reference rows, and
  distinct long/short November FS plus December/January/February HPO cohorts.
  Independent checks found no duplicate IDs, wrong-side rows, or timing
  violations. This completes population materialization, not learned fitting.
- Use the immutable DEC-09 inner calendar: fit each side's AE/GMM on
  beginning/middle/end samples from authorized rows before November 1, use
  November as the feature-selection validation interval, and use December,
  January, and February as the three chronological HPO validation intervals.
  Apply the strict 24-hour resolution boundary at every inner split and permit
  no silent selector/model fallback.
- Freeze each side's newly selected feature contract, parameters, and AE/GMM
  state before scoring any outer fold. Regenerate four canonical Pack-B OOF
  folds using the locked half-open April, May, June, and July 1–11 signal
  windows without consulting outer-fold outcomes.
- Resolve the canonical label shard list from the label manifest or 38-file
  causal audit. Reject missing or extra Parquet shards, including the stale
  overlapping `train_global_short_7.parquet`, and reject duplicate candidate
  IDs before any fitting.
- Stream folds, sides, and bounded symbol batches sequentially. Do not persist
  raw or AE-transformed fold caches. Preflight and checkpoint RAM, process RSS,
  and free disk with JSONL telemetry.

Deliverables:

- Pack-B side-local provenance audit
- Residual-alpha side-local provenance audit or replacement OOF/final bundle
- Canonical Pack-B-derived long top-40 handoff
- Canonical Pack-B-derived short top-40 handoff
- Exact rank/mask reconciliation and hashes
- Four-fold DEC-09 Pack-B OOF ledger with row-level signal, decision,
  24-hour label-resolution, fold, cutoff, side-model, feature-contract,
  parameter, AE/GMM-state, source, and score hashes
- Pre-March per-side AE/GMM, feature-selection, and HPO authorization ledger,
  including resolved-label keys, reference/inner folds, search breadth, and
  immutable artifact hashes
- Authoritative label-shard inventory and duplicate-key audit

Gate R3:

- No fitted selector, parameter, model, prior, calibrator, or OOF outcome crosses sides.
- Every AE/GMM, selector, and HPO input is authorized by an actual label
  resolution strictly before the DEC-09 pre-March cutoff.
- No unlisted label shard or duplicate candidate ID enters the run.
- Top-40 masks reproduce exactly from the corresponding source OOF ledger.
- Current shared-store top-40 is absent from canonical downstream manifests.
- Residual alpha meets locked DEC-02 per side.
- Benchmark comparison meets locked DEC-01 on identical rows and costs.

### C1/C2 — Seven-Class CatBoost, Independently per Side

Owner: CatBoost
Depends on: R3
Can run in parallel: long and short; auxiliary work packages

For each side:

1. Run feature eligibility, redundancy pruning, staged selection, and automatic
   MDA stopping.
2. Run a side-only geometry search; pooled geometry
   `geometry_e33b290e324f3182` is a benchmark/seed only.
3. Run side-only model HPO.
4. Run a bounded class-balance mini-HPO:
   - unweighted control;
   - predeclared mild/moderate frequency correction;
   - bounded maximum weight ratios;
   - no centroid membership, ambiguity weighting, economic sample weighting,
     or probability transform.
5. Generate fold OOF predictions.
6. Produce complete predictive, economic, concentration, and stability reports.
7. Refit the selected configuration on eligible resolved rows.

Required OOF fields:

- seven ordered probabilities;
- predicted class;
- maximum probability;
- normalized entropy;
- top-2 margin;
- favorable and adverse probability mass;
- side, fold, exact keys, class-order hash, row/source/model hashes.

Required reports:

- log loss, macro/weighted F1, RPS, Brier, raw ECE;
- class precision/recall, confusion distance, predicted shares;
- max probability, entropy, top-2 margin;
- net-EV, MFE, MAE, stop, realization, retention and trailing separation;
- probability-weighted outcome IC;
- fold/month/symbol/base-archetype/score-tail stability and worst month.

Gate C1/C2:

- Each class meets fixed support or the predeclared DEC-05 path is followed.
- No missing supported predicted class or collapse under DEC-03.
- FS, geometry, model HPO, and balance HPO are side-specific.
- Class order is unchanged through the execution adapter.
- Assessment uses OOF only; final refit is explicitly excluded.

### A1/A2 — Five Auxiliary Heads, Independently per Side

Owner: Auxiliary
Depends on: R3
Can run in parallel: long/short; five targets; CatBoost

Targets:

1. `peak_mfe_12h_atr`
2. `time_to_first_meaningful_mfe`
3. `mae_before_meaningful_mfe_atr`
4. `bars_before_price_stops_decreasing`
5. `future_slope_atr_per_hour`

Tasks per side and target:

- Move the current global pre-screen fully inside the side loop.
- Use only the canonical Pack-B-derived side top-40 population.
- Run univariate/Relief, correlation pruning, MDA automatic stopping, then HPO.
- Keep target-specific weights bounded to `[0.5, 2.0]`.
- Generate monthly growing-window OOF predictions and uncertainty/support fields.
- Report regression error, rank IC, top-1/5/10 economics, monthly stability,
  support/missingness, and supportive-event calibration.
- Final-refit and persist side-specific model, features, parameters, provenance,
  and hashes.

Checkpoint rule:

- The legacy v18 fingerprint is
  `8c0bfcc4a939a18690c394652571a00944a5d9a183762362e568a16828c7629e`.
- Changing top-40 identity to the required Pack-B stream changes an upstream
  identity. Therefore do not resume v18 unless the full fingerprint
  unexpectedly still matches after the canonical handoff is built.
- The handover’s sample resume command points at shared-store top-40 and must
  not be run as written. Confirm current CLI with `--help` and use a new output
  directory for the corrected side-local run.

Gate A1/A2:

- Ten exact-key OOF streams exist: five targets × two sides.
- Ten final bundles exist with independent features, parameters, models, hashes.
- No resolved/supportive outcome label appears in inference features.
- OOF rows were not scored by a model trained on their resolved label path.

### H1 — Strict Execution-EV Handoffs

Owner: Execution-EV + Data/provenance
Depends on: C1, C2, A1, A2, canonical alpha OOF

Tasks:

- Adapt raw CatBoost OOF; the legacy `catboost_refinement` filename must not
  restore abandoned refinements.
- Materialize auxiliary OOF execution inputs.
- Validate each stream separately.
- Join separately for long and short.
- Reconcile key intersection and attrition against every input manifest.

Required inputs:

- base score/rank/margin anchors;
- matching-side residual-alpha output and mapped EV;
- observable base archetypes and frozen AE/GMM/OOD/support context;
- five auxiliary predictions plus uncertainty/support;
- seven CatBoost probabilities plus aggregate confidence/risk fields;
- execution label and metadata as targets/report-only fields.

Gate H1:

- Exact one-to-one keys on `__ts__, __symbol__, side_name`.
- Same OOF fold identity and eligible resolution contract.
- No final-refit prediction or path outcome in inference inputs.
- Entry/horizon/cost contracts match or have an explicit, non-duplicative map.
- Expected intersection, missing rows, and attrition reconcile exactly.
- The execution-EV OOF writer emits, for every finite prediction:
  `execution_ev_oof_fold`, source training cutoff, prediction availability
  timestamp, model/feature/calibrator hashes, and a true-OOF flag.
- A signed side-local mapping stage emits `execution_ev_map_oof` (or the single
  canonical equivalent name), its common-unit/cost contract, and the exact
  mapping source/hash required by timing.
- Warm-up/non-OOF rows remain explicit and are excluded before any downstream
  training join; they are never silently filled.

### E1 — Direct/Residual Execution-EV and Feature-Family Ablations

Owner: Execution-EV
Depends on: H1, decisions DEC-06 through DEC-09 locked

Implementation prerequisites:

- Add an explicit side-local feature-selection stage and freeze its result before
  side-local HPO.
- Fit fold calibration only on same-side inner OOF rows; remove the pooled
  fold calibrator.
- Persist `calibration["long"]` and `calibration["short"]`; remove the final
  `calibration["__global__"]` mapper.
- Select final parameters through a declared all-authorized-training procedure,
  not implicitly by taking the last outer fold’s parameters.
- Make ablation attribution fair: use authorized arm-specific HPO or freeze a
  matched parameter contract across arms.
- Require explicit row-level source fold and source-train-cutoff provenance for
  every supervised upstream prediction family.
- Expand the named ablation plan to include alpha + auxiliary, alpha +
  CatBoost, alpha + both, and removal of AE/GMM/OOD/support as explicit arms;
  do not rely only on one-family-at-a-time generic removal.
- Add gross EV, trade support/rate, notional and bankroll PnL, drawdown, cost
  sensitivity, stop/timeout conversion, signed residual autocorrelation, and
  signed causal hit-rate surprise to the report.

Train independently per side:

- Direct target: realized causal 12h execution EV.
- Residual target: realized causal 12h execution EV minus a train-only mapped
  alpha expected EV.
- Independent feature selection, HPO, early stopping, model, and monotonic EV
  calibration.
- Replace any pooled `__global__` mapper with independently fitted long and
  short maps emitting the documented common unit.

Required identical-row ablations:

1. Alpha only.
2. Alpha + auxiliary.
3. Alpha + CatBoost.
4. Alpha + auxiliary + CatBoost.
5. Remove timing features.
6. Remove adverse-path features.
7. Remove AE/GMM/OOD/support context.
8. Direct versus residual target.

Required reporting:

- gross/net EV per trade at top 1/5/10/20/30%;
- per-side primary views and reporting-only comparable global top-k;
- weekly/monthly stability, worst week/month;
- win rate, profit factor, MFE/MAE conversion, timeout/stop/adverse shares;
- IC/rank stability and common-unit calibration;
- side/archetype contribution and concentration;
- signed residual mean/autocorrelation;
- positive and negative causal hit-rate surprise.

Gate E1:

- Every arm uses identical eligible rows, folds, labels, horizon, costs, and top-k.
- Automated tests prove that changing short-side outcomes cannot change the
  long selector, HPO result, model, calibrator, or OOF predictions, and vice
  versa.
- No `__global__` fitted calibrator exists in a selected execution-EV bundle.
- Every finite execution-EV OOF prediction has an outer fold, validation
  interval, and training cutoff, and all supervised upstream source cutoffs are
  compatible with that fold.
- Top-10 improvement clears DEC-06.
- Worst-period behavior clears DEC-07.
- Both sides contribute; uplift is not confined to one month/archetype.
- Independent maps pass the DEC-08 common-unit equivalence test.

### D1 — Promote or Reject

Owner: Roadmap owner + Validation
Depends on: E1

Produce a signed decision record with one outcome:

- `PROMOTE_FULL`: alpha + auxiliary + CatBoost execution EV;
- `PROMOTE_PARTIAL`: a simpler winning family;
- `RETAIN_ALPHA`: no execution-EV layer clears the gate;
- `RERUN`: only for a predeclared contract failure, never because results are
  disappointing.

The decision record must cite exact run IDs, hashes, rows, folds, metrics, and
all failed/passed gates. A rejection is a valid completed research result.

Selection must be side-local under fixed DEC-06/07/09 rules. The current
aggregate all-side top-10 winner helper is reporting-only and cannot select the
production pair. The timing handoff must consume the actual selected long and
short mode/arm, not hard-code `direct__all_features`.

### B1 — Final Refit and Two-Sided Bundle

Owner: Execution-EV + Validation
Depends on: D1 promotion

Tasks:

- Refit only the selected side-specific configurations on resolved eligible rows.
- Persist models, selectors, features, parameters, calibrators, target/cost/
  entry/horizon contracts, final-fit cutoffs, class schema, upstream provenance,
  common-unit contract, and hashes.
- Create one bundle manifest linking distinct long and short components.
- Load in a clean process and run deterministic score and missing-value tests.

Gate B1:

- Bundle loads without development state.
- Required inputs are present and finite under the declared missing-value policy.
- Frozen features reproduce deterministic scores.
- Final refits are marked excluded from all OOF claims.

### T1 — Entry Timing Decision

Owner: Roadmap + Policy
Depends on: stable E1 OOF

Choose and record one:

- `OMIT`: execution-EV proceeds directly to policy, with rationale and scope;
- `VALIDATE`: run fully side-local timing FS, model HPO, action-grid HPO,
  isotonic calibration, OOF assessment, and final fit.

Before `VALIDATE`:

- replace legacy `PATH_SHAPE_TYPES` usage with the exact seven-class order and
  hash supplied by the selected CatBoost bundle;
- add independent side-local feature selection for fill, adverse-first, and
  conditional filled-delta tasks;
- move model HPO and decision/action-policy HPO inside the long and short loops;
- persist separate allowed-action grids, penalties, parameters, selected
  features, and isotonic maps by side;
- preserve separate outputs for fill probability, adverse-first probability,
  and conditional filled delta EV; do not collapse the adverse-risk preference
  into a value presented as pure net EV;
- require the selected execution-EV OOF score and its row-level source
  fold/cutoff as the protected timing input;
- implement and sign the missing mapped execution-EV OOF producer required by
  the timing handoff;
- reconcile the 1-minute path interval so materialization and label building
  agree exactly on the first executable minute and the inclusive/exclusive 12h
  endpoint;
- increase final HPO breadth beyond smoke defaults and record the full search;
- implement the deterministic timing-execution policy described above, including
  side-correct conversion of an approved ATR offset into a tick-valid limit
  price, expiry/cancel rules, action-specific costs, and fail-closed checks.

Gate T1 when validating:

- every recommendation used for timing metrics is outer-fold OOF;
- every upstream supervised prediction is row-level OOF-compatible with the
  timing outer fold;
- no final-refit prediction enters timing training, HPO, calibration, action
  selection, or policy optimization;
- long outcomes cannot affect any short fitted timing state and vice versa;
- all seven CatBoost probabilities and the class-order hash match the selected
  CatBoost bundles;
- the timing handoff rejects rows lacking finite execution-EV OOF prediction,
  true OOF flag, prediction availability, source cutoff, and signed mapped-EV
  provenance;
- the first timing path bar is strictly the first executable minute after the
  decision and the last bar matches the declared 12h endpoint, with an exact
  720-bar invariant for one-minute paths;
- the LGBM timing arm beats ridge/logistic, fixed-grid, and enter-now baselines
  on identical rows after spread, fees, missed opportunity, and adverse-first
  costs;
- uplift is stable by side, week/month, archetype, spread, liquidity, volatility,
  and action type;
- fill and adverse-first probabilities are calibrated side-locally, and
  conditional filled-delta error is stable on supported rows;
- report realized risk-adjusted utility and pure net EV separately, plus regret
  versus enter-now and an OOF oracle, fill/no-fill rate, missed positive-EV
  opportunity, adverse-first rate, and entry-price improvement in bps and ATR;
- every price recommendation is reproducible from logged pre-entry inputs,
  respects tick/market/order constraints, and has replay/live parity for
  fill, partial-fill, expiry, cancel/replace, fee, spread, and slippage rules;
- a wait/limit recommendation cannot pass unless its conservative calibrated
  utility clears enter-now by the declared margin and all confidence, support,
  liquidity, staleness, fill-feasibility, and cost gates pass.

Do not let entry-timing experimentation silently block a valid execution-EV
decision. Do not promote a replacement policy without an explicit T1 outcome.

### P1 — Policy and Replay Integration

Owner: Policy + Validation
Depends on: B1 and T1 decision

#### P1.1 Candidate adapter

Build a strict adapter from the selected execution-EV OOF ledger to the existing
policy candidate schema. It must preserve exact candidate identity and add:

- `execution_ev_oof` in the documented common net-return unit;
- `execution_ev_rank_pct`, computed within the predeclared ranking scope;
- `execution_ev_side_calibration_id`;
- execution-EV outer fold, training cutoff, model/feature/calibrator hashes;
- upstream alpha/CatBoost/auxiliary lineage hashes;
- target horizon, decision/entry timestamps, and cost-basis ID;
- optional timing OOF action/value fields when T1 is validated;
- explicit `policy_score_source` and `policy_cost_reconciliation_id`.

Reject duplicate keys, final-refit predictions, missing OOF lineage, mixed cost
bases, or any row outside the policy replay path intersection.

The existing side/archetype policy runner can provide a first shadow integration
because it accepts an arbitrary `--rank-score-col`. For execution-EV candidates:

- set the rank source to `execution_ev_oof` or the adapter’s frozen
  `execution_ev_rank_pct`;
- use the explicitly chosen `side`, `timestamp_side`, or post-common-unit
  `global` rank scope from DEC-08;
- disable the legacy regime-EV calibrator because execution EV is already
  calibrated to the common unit;
- do not reuse the current `0.007` admission threshold until its old cost/unit
  contract is reconciled to the execution-EV target;
- do not run the legacy S52 strategy-mask path as the final global architecture.
  Use it only as an adapter/proof-of-interface or matched benchmark.

#### P1.2 Integration modes

Evaluate three increasingly invasive modes:

1. **Shadow/rank-only:** execution EV supplies candidate ordering while current
   admission, exit geometry, sizing, and portfolio settings remain frozen.
2. **EV admission:** execution EV supplies ordering and a train-selected
   same-unit admission threshold; current geometry/sizing/portfolio remain
   frozen.
3. **Full challenger:** execution EV supplies ordering/admission and policy
   geometry, sizing, and portfolio parameters are re-optimized on authorized
   policy folds.

If timing is validated, add a fourth mode in which timing chooses
an approved enter-now/wait/adverse-limit template before exit replay, and the
separate deterministic timing-execution policy converts any adverse-limit
template into a logged, tick-valid price. The comparison without timing must
remain present.

#### P1.3 OOF-only policy waterfall

Use execution-EV OOF—and timing OOF when applicable—for every policy optimization
and research replay row. Final-refit scores are inference-only. Run this
identical-row waterfall:

1. Current alpha/policy benchmark.
2. Execution-EV OOF rank + frozen current admission/geometry/sizing/portfolio.
3. Execution-EV OOF rank/admission + frozen geometry/sizing/portfolio.
4. Step 3 + re-optimized geometry.
5. Step 4 + re-optimized sizing.
6. Step 5 + re-optimized portfolio auction.
7. Optional timing OOF action layer.

At every step report the marginal change in candidate count, selected identity,
net EV/trade, total notional return, bankroll PnL, turnover, exposure, drawdown,
stops, timeouts, concentration, and worst week/month. Freeze each selected
policy before scoring the next non-training fold.

#### P1.4 Cost and score contract

Choose one canonical expected-net-EV unit before policy optimization. Record:

- whether p90 spread and the 30-bps round trip are already embedded;
- whether inference friction rebasing is required;
- which recent-EV correction, if any, is applied;
- whether the threshold is a raw EV value or a percentile;
- how sizing consumes EV magnitude, uncertainty, OOD, and support.

Never feed the new execution EV through the retired regime calibrator, subtract
the old 1% alpha mapping again, apply the p90 spread/30-bps label cost again, or
stack the old `corrected_expected_ev` adjustment without an explicit matched
ablation.

The new execution score is already net of its declared label costs. In the
first integration, use it for admission and auction ordering only. Do not route
it into the existing holding-pressure/redeployment fields that subtract
round-trip friction from an expected-EV input. That path requires a separate
gross-versus-net conversion contract and ablation; otherwise costs are charged
twice.

#### P1.5 Artifact safety

- Write every challenger to a new artifact/run directory.
- Keep the current champion immutable and record it as rollback.
- Persist policy-fold cutoffs, selected thresholds/geometry/sizing/portfolio
  parameters, candidate/score hashes, and cost reconciliation.
- Create a thin inference adapter only after the research/policy winner is
  frozen; it must consume final execution-EV/timing bundles with the same field
  and unit contract as the OOF adapter.

Gate P1:

- Candidate adapter achieves exact one-to-one key and path reconciliation.
- Policy optimization and comparisons use execution-EV/timing OOF only.
- Rank-only, EV-admission, and full-challenger modes are all reported on
  identical rows; optional timing has a no-timing control.
- Legacy calibration/correction/cost layers are either disabled or individually
  reconciled and ablated.
- Predicted net execution EV is not supplied to friction-subtracting
  holding-pressure/redeployment fields unless an explicit gross/net conversion
  test proves costs are applied once.
- New policy clears predeclared economic and stability gates.
- Side/archetype reporting is complete.
- Replay reproduces research score/admission behavior under the same contracts.
- Current policy remains recoverable.

### L1 — Inference Parity and Promotion

Owner: Validation + Policy
Depends on: P1

Required parity:

- feature values/order/missingness;
- frozen AE/GMM outputs;
- base, residual, CatBoost, auxiliary, and execution-EV predictions;
- common EV calibration;
- admission, sizing, portfolio allocation, and exits;
- timestamps, entry delay, horizon, spread, and fees.

Gate L1:

- Replay/inference parity passes on frozen fixtures and a clean process.
- Operational rollback target is recorded.
- Live promotion is explicit; absence of this sign-off means live remains blocked.

## 8. Global Stop Conditions

Stop the affected run immediately for:

- source/input/fingerprint drift;
- non-UTC or duplicate handoff keys;
- failed causal path invariant, purge, or embargo;
- outcome/support labels in inference inputs;
- same-fold upstream predictions or final-refit predictions in OOF evidence;
- AE/GMM refit outside the authorized pre-March side-local cycle contract;
- shared fitted selector/model/calibrator before the portfolio layer;
- class-order drift or CatBoost collapse;
- unexplained handoff attrition;
- cost double-counting or incompatible cost comparisons;
- comparisons on different rows, folds, horizons, labels, or top-k bases;
- tuning against final OOS/replay results;
- an attempt to treat the empty CatBoost directory or partial auxiliary
  checkpoints as trained models.

## 9. Artifact Naming and Status Rules

Every output directory must be new unless an exact resume fingerprint matches.
Each stage manifest must use one status:

- `DRAFT`
- `RUNNING`
- `FAILED_CONTRACT`
- `OOF_COMPLETE`
- `REJECTED`
- `SELECTED`
- `FINAL_REFIT`
- `REPLAY_VALIDATED`
- `INFERENCE_PARITY_VALIDATED`
- `PROMOTED`

Never use `canonical`, `production`, or `promoted` for an OOF-only research run.
Never overwrite the existing executable policy while validating a challenger.

## 10. Weekly/Run Control Board

Track this table in the repository and update it only with linked evidence.

| WP | Status | Owner | Blocking issue | Evidence/run ID | Gate signed by |
|---|---|---|---|---|---|
| R0 Migration | IN PROGRESS | Data/provenance | No comparison baseline; six active Pack-B lineage paths missing | `docs/pipeline_roadmap/20260724/r0/`; `r0_missing_path_triage.md` |  |
| R1 Source durability | IN PROGRESS | Roadmap + Data/provenance | Remote recovery requires explicit publication authorization | `config/pipeline_stage_manifest_repository_20260724.json`; schema v1 |  |
| R2 Contracts | IN PROGRESS | Validation | P0 artifact-level hash reconciliation and stage-specific horizon reconciliation pending | `docs/pipeline_roadmap/20260724/r2_test_log.md`: 138 focused + 149 broader pass; DEC-01…10 locked |  |
| R3 Alpha/top-40 | IN PROGRESS | Alpha + Data/provenance | Side-local AE/FS/HPO fitting and four-fold OOF still required; population materialization is complete | Current 38-shard audit PASS on 4,552,934 rows; immutable 3,429,788-row population + 18 side-cohort ledgers; historical `short_7` explicitly excluded; `docs/pipeline_roadmap/20260724/r3/pre_march_population_materialization_v1.json` |  |
| C1 CatBoost long | BLOCKED BY R3 |  |  |  |  |
| C2 CatBoost short | BLOCKED BY R3 |  |  |  |  |
| A1 Auxiliary long | BLOCKED BY R3 |  |  |  |  |
| A2 Auxiliary short | BLOCKED BY R3 |  |  |  |  |
| H1 Handoffs | BLOCKED |  |  |  |  |
| E1 Execution EV | BLOCKED |  |  |  |  |
| D1 Decision | BLOCKED |  |  |  |  |
| B1 Bundle | BLOCKED |  |  |  |  |
| T1 Timing decision | BLOCKED |  |  |  |  |
| P1 Policy/replay | BLOCKED |  |  |  |  |
| L1 Live parity | BLOCKED |  |  |  |  |

## 11. Immediate Next Actions

Execute in this order:

1. Keep DEC-01 through DEC-10 frozen; bind the actual stage-specific label
   horizon in every run manifest.
2. Complete R0 checksums, read-only loads, disk/environment, and process audit.
3. Make the dirty/untracked source recoverable under R1.
4. Run R2 deterministic and broader contract suites.
5. Fit the distinct long and short outcome-free AE/GMM states from the now
   materialized pre-November cohorts, then run side-local feature selection
   and fixed three-fold HPO. The inventory, duplicate-key, timing,
   fixed-calendar, population-ledger, and post-fit evidence validators pass;
   no learned artifact has yet been fitted.
6. Run the four canonical Pack-B OOF folds sequentially only after the
   pre-March artifact ledger and measured memory/disk envelope pass, then
   derive top-40 per timestamp × side.
7. Decide whether the existing alpha execution OOF can prove canonical lineage;
   regenerate it if not.
8. Start C1/C2 and A1/A2 in parallel using new output directories.
9. Reconverge only when all twelve required model streams are OOF-complete:
   two CatBoost streams plus ten auxiliary streams.
10. Build H1, run E1, and make the D1 promote/reject decision before any
    policy, timing, replay, or live work.

## 12. Research Definition of Done

The execution-EV research objective is complete when either:

1. a direct or residual side-local execution-EV configuration clears every OOS
   promotion gate and has a clean two-sided final bundle; or
2. all predeclared ablations are complete on identical OOS rows and the signed
   decision is `RETAIN_ALPHA`.

Policy completion and live completion are separate milestones. A research
winner is not a live-ready model.

## 13. 2026-07-27 Inference Spread-Universe Replay

The static spread-exclusion fields in
`base_residual_label_ablation_20260725_v2` are invalidated **only for that
diagnostic** by
`data_perp/artifacts/base_residual_label_ablation_20260725_v2/SPREAD_EXCLUSION_DIAGNOSTIC_INVALIDATION.json`.
The original comparison was a slash/underscore normalization no-op: both sides
matched zero rows. Its label-HPO and non-spread results are not invalidated.

`execution_ev_inference_spread_universe_ablation_20260727_v2` repairs the
measurement using the exact live `universe.py` normalizer and
`average_spread_bps > 70` blacklist rule. It evaluates the three current mapped
arms (`direct_net`, `hurdle_prob`, `competing_clean_probability`) as one pooled
global top-10% book, the eligible slice of that book, and a reranked eligible
book, with exact frozen-policy gross/cost/net targets.

This particular current mapped cohort already contains 155 baseline-covered,
inference-eligible symbols and **zero** symbols excluded by the current static
contract. Therefore all three books are identical; this is not evidence that
the spread universe is ineffective, only that the candidate ledger was already
filtered upstream. The replay is explicitly `non_PIT_retrospective_only`:
the baseline was produced in June--July 2026 and cannot support a causal
historical training-exclusion or promotion claim. A removal-effect ablation
requires an earlier pre-universe candidate ledger plus a decision-time frozen
eligibility snapshot.

## 14. Fail-Closed Global Admission

`execution_ev_variable_admission_20260727_v1` applies fixed mapped
predicted-net floors of 0, 25 and 50 bps to the existing strict-OOF/forward
scores. It fits no model and preserves one pooled global 10% capacity; an arm
may select fewer rows or no rows.

The current direct, hurdle-probability and hurdle-EV arms admit zero later-July
trades above a zero predicted-net floor. The hurdle capture guard admits two
and realizes -190.27 bps. In the May-June control, the only positive fixed
slice is 47 hurdle-capture-guard rows above 50 predicted bps at +78.95 bps,
but it fails the later-July recurrence gate. It is not a threshold winner.

Decision: do not force top-k when every mapped score is negative, and do not
run portfolio promotion replay for any of these arms. Continue with the
oracle opportunity/capture ceiling and exit-policy diagnosis.

## 15. Opportunity Ceiling and Gross-First Closure

The exact-policy oracle ceiling is strongly positive but non-tradable:
one-global oracle top-10 net is +294.50 bps on the 127,777-row current panel
and +266.65 bps on the 473,068-row common-30 lineage. Every tested week has a
positive week-local oracle top-10. The negative learned books are therefore a
recoverability/false-positive problem, not an absence-of-opportunity result.

The missing gross-first arms are complete and rejected:

| Arm | May → June | Later July |
|---|---:|---:|
| Direct exact net | -60.69 bps | -106.54 bps |
| Direct predicted gross − exact cost | -103.14 bps | -115.14 bps |
| Capture-gross mixture − exact cost | -130.88 bps | -105.34 bps |

Both gross-first arms admit zero later-July trades above a fixed zero
predicted-net floor. `execution_ev_variable_admission_20260727_v2/`
therefore contains an explicit `PORTFOLIO_REPLAY_GATE.json` rejection. Do not
spend portfolio-optimizer degrees of freedom on a signal that fails the
unconstrained economic and recurrence gates.

## 16. False-Positive Recovery Gate

Across 30 current mapped arms, the best learned global top-10 book remains
-59.57 bps in May-June and -71.23 bps in later July. Same-k future-oracle
recall peaks at only 16.04% and 17.66%; smaller top-5/top-2 books remain
negative. The system is missing most high-surplus rows while retaining severe
false positives.

A June-frozen decision-time screen found only
`base_margin_to_cutoff_z` economically direction-stable in later July.
Applying its exact frozen threshold backward to the common-30 strict OOF
lineage yields a positive keep-versus-drop lift in 9 of 12 month diagnostics,
but every retained month remains negative. Authorize it only as a soft
capture/confidence interaction in a future-frozen challenger. Do not deploy it
as a hard gate, and do not reuse later July as the deciding OOS block.

## 17. 2026-07-28 Compact Capture Matrix

The corrected v4 hurdle runner now compares all requested event/capture
formulations per side on the same purged rows, exact costs, causal 21-day
mapping, and one pooled global top-10% book. It adds the two previously missing
arms:

- a mutually exclusive ATR-soft timeout/adverse/favorable target simplex; and
- explicit captured upside minus adverse-first loss.

Neither is promoted:

| Arm | May → June | Later July | Decision |
|---|---:|---:|---|
| Direct net baseline | -59.69 bps | -106.52 bps | Frozen control |
| Competing clean probability | -69.60 bps | **-92.96 bps** | Latest improves, control degrades; reject |
| ATR-soft favorable probability | -76.48 bps | -151.14 bps | Reject |
| ATR-soft decomposed EV | -91.89 bps | -102.32 bps | Reject |
| Capture upside − adverse loss | -109.82 bps | -104.95 bps | Reject |

The competing-risk classifier remains the least-bad latest-period event arm,
but it does not satisfy joint recurrence or positive-EV gates. Do not add any
of these heads to the frozen forward challenger.

Artifact:
`data_perp/artifacts/exact_policy_capture_hurdle_ablation_20260728_v4/`.

## 18. Identical-Row Long Ranking

The 114,096-row exact-policy diagnostic now reports the long contribution to
each unchanged pooled global book and a separate long-only diagnostic. The
base, raw execution-EV, and mapped scores use identical candidate IDs and exact
gross/cost/net outcomes.

| Pooled score | Whole-book net | Long rows in book | Long contribution |
|---|---:|---:|---:|
| Base rank | +1.76 bps | 5,956 | -5.53 bps |
| Raw execution EV | -32.40 bps | 9,774 | -43.24 bps |
| Causal global 21d EV | +3.12 bps | 6,579 | +8.50 bps |
| Causal side 21d EV | -8.91 bps | 6,246 | +9.55 bps |

The raw execution-EV layer strongly over-admits long candidates and damages
their economics. The causal maps repair long ordering on this reused May-July
panel, but the global book is not stable across mapping scopes. Retain mapping
as calibration; do not treat the positive reused long slice as new promotion
evidence.

The bounded exit-policy workstream is closed: nearby geometry changes repair
only 1-8 bps against an approximately 90-bps later-fold deficit, and even the
per-row hindsight best nearby policy remains negative after cost. No further
exit HPO is authorized before a new forward block.

Artifact:
`data_perp/artifacts/execution_ev_identical_row_long_ranking_20260728_v1/`.

## 19. Frozen Forward Package

The complete final-refit inference chain is now implemented and pinned:

1. Pack-B 31/8 per-side base scoring and deterministic top-40% candidate
   context;
2. per-side residual alpha;
3. clean-event, Peak MFE and seven-class path supporting heads;
4. persisted direct-net and capture heads;
5. causal 21-day calibrator state;
6. frozen base-margin × capture-confidence interaction;
7. one pooled global top-10% admission, with zero trades allowed.

The earlier source locks are explicitly superseded without modifying their
contracts. The authoritative successor is
`execution_ev_forward_source_lock_20260728_v5`, fingerprint
`5d9068ca1ee95526cb17e31fb0bb5bbd017087d75222fb450d1f6c24e2fec460`.
It freezes decisions after 2026-07-27 23:59:59.999999 UTC through August 10,
with exact 12-hour outcomes.

Source-lock readiness passes. Pre-outcome readiness fails only because the
genuinely future scored population does not exist yet. Open the block only
after 14 complete UTC days, at least 5,000 scored rows and 500 members of the
single pooled global top-10% evaluation capacity. Economically admitted rows
may be zero when all mapped scores are non-positive; this is a valid
abstention, not a successful promotion. Do not substitute the available
July-20 live matrices or any reused OOF cohort.

Run the block only through
`scripts/run_execution_ev_forward_preoutcome.py`. It validates raw coverage
and resolved-update manifests, exact candidate/score lineage, deterministic
candidate-ID tie-breaking, and every frozen model/code hash before atomically
publishing a sealed population. The seal is mandatory and binds the source
lock, candidate identities, raw coverage, causal updates, intermediate hashes
and final output hashes. Outcome/label columns are forbidden pre-outcome.

## 20. Failure-First Regime Pipeline

The failure-first infrastructure is implemented in:

- `extreme_price_movements/unsupervised_regime_learning/failure_first_health.py`;
- `extreme_price_movements/unsupervised_regime_learning/failure_first_hourly.py`;
- `extreme_price_movements/unsupervised_regime_learning/failure_first_pipeline.py`;
- `extreme_price_movements/unsupervised_regime_learning/failure_first_detector.py`;
- `extreme_price_movements/unsupervised_regime_learning/failure_first_binary.py`;
- `scripts/run_failure_first_regime_pipeline.py`;
- `scripts/run_failure_first_binary_ablation.py`.

The canonical runner starts with failures in the causally mapped, pooled
global stream. It uses a trailing `[t-21d,t)` global score q90 as a causal
shadow admission frontier, exact 12-hour net outcomes, six-hour health bins,
and prior-only q05/q10/q20 residual tails. Admission and residual histories
reset at every model/evaluation-origin boundary. The retrospective full-panel
top decile is forbidden for episode discovery.

Each failure episode is described at:

`-48h, -24h, -12h, -6h, -3h, 0h, +3h, +6h, +12h`.

Decision-time state and ex-post outcomes are stored separately. The compact
taxonomy profile contains exactly 40 descriptors: observable onset state plus
predeclared -12h-to-onset changes. H1/H3/H6/H12 market-state fields, outcomes,
MFE, MAE, exit fields, GMM posteriors, DAE fields and retrospective regime IDs
are forbidden detector inputs.

If support passes, the runner:

1. freezes failure-only GMM and KMeans taxonomies separately before detector
   evaluation, defaulting to a predeclared five-to-eight-state GMM taxonomy;
2. rejects any taxonomy with fewer than five resolved episodes in a cluster;
3. builds availability-explicit current-state, active-transition,
   transition-within-three-hours and destination-state targets;
4. adds one causal BOCPD signal family as continuous features, never as a hard
   decision rule;
5. fits one compact CatBoost four-head detector in purged chronological OOF;
6. rejects any fold that would require a constant-class head;
7. reports aggregate and latest-month classification metrics plus one pooled
   global top-10% economic overlay. No per-timestamp or per-side quota is used.

Taxonomy and detector training require all of:

- at least 40 resolved failure episodes;
- at least 40 complete fixed-window episodes;
- at least 40 failure bins;
- at least 180 calendar days;
- at least 180 actually observed UTC days and no gap longer than 21 days;
- at least 1,000 complete detector rows;
- at least 50 transition positives, 50 active-transition positives and 50
  rows in every current/destination class.

The authoritative run is
`data_perp/artifacts/failure_first_regime_pipeline_20260726_v6/`. It contains
121,208 strict model-OOS rows: 114,096 outer-OOF rows plus 7,112 previously
opened, resolved and explicitly retired frozen-forward rows. Original
provenance flags and evaluation origins remain intact; retired rows may train
only a later detector and can never evaluate one fitted on this history.
The ledger spans 73.5 days, has 74 observed UTC days and a maximum six-hour-bin
gap of 1.25 days, but only three primary/catastrophic bins
merged into two episodes. Only one episode has complete market-state coverage
at all nine offsets because raw H0 state begins on June 8. The runner therefore
publishes descriptive health, membership, episode, window and outcome
artifacts, then correctly returns `INSUFFICIENT_SUPPORT`. It writes no taxonomy
or detector model.

The extension-readiness audit finds no legal local source after July 19 16:00
UTC. Remaining minimum deficits are 106 observed days, 38 failure episodes, 39
complete fixed-window episodes and 37 failure bins. July-20 upstream
base/residual/alpha artifacts stop at the same cutoff or lack mapped execution
EV; the later 1,088-row raw label tail has no current score and only 16 complete
one-minute paths. Legacy Pack-B/V9 artifacts lack immutable IDs and use
different models, mapping and 8h policy mechanics. Do not substitute them or
relax the gate.

Artifacts:

- `failure_first_current_strict_model_oos_history_20260726_v1`;
- `failure_first_current_extension_readiness_20260726_v1`.

Timing, MAE, target-price and wait actions remain in the separate action layer
and are not inputs to this failure/trust detector.

### Historical comparator result

The available strict historical two-layer OOF lineage was materialized
separately rather than represented as current-model history:

- `failure_first_historical_backfill_20260726_v3`;
- 440,560 common-30 rows;
- January 15 through November 30, 2025, with a March warm-up boundary;
- 307 observed days;
- maximum six-hour-bin gap of 14 days, below the frozen 21-day ceiling;
- exact one-minute current-policy gross/net/cost outcomes;
- causal side-local 21-day EV correction reset by model generation;
- 28 canonical raw-H0 market-state fields joined from the feature store with
  completed-bar delay and at least 90% field coverage.

The historical pipeline finds 51 severe bins and 35 complete episodes. The
conservative 40-episode gate remains unmet. A lower-bound 30-episode research
run was therefore permitted as diagnostic only.

The proposed five-to-eight-state taxonomy does **not** fit this sample. Both
GMM and KMeans create unsupported singleton/outlier clusters even at four or
five states. Robust clipping plus a two-state diagonal GMM is the only
supported configuration:

| State | Frozen train episodes | Dominant descriptor |
|---|---:|---|
| correlation fragmentation, elevated | 22 | onset breadth dispersion |
| mixed observable/model state, elevated | 8 | onset historical base score |

KMeans still fails the minimum five-episode cluster gate. Thus the evidence
supports two broad failure families, not five-to-eight stable regimes.

The two-state CatBoost detector completes purged chronological OOF, but is
rejected:

- transition-within-3h AUC: **0.420**;
- active-transition AUC: **0.513**;
- latest OOF transition positives: 27, below the required 50;
- mapped global top-10%: **-108.70 bps**;
- destination-risk adjusted: **-108.89 bps**.

Artifact:
`data_perp/artifacts/failure_first_regime_pipeline_historical_20260726_v12/`.

The support-passing two-state GMM also fails the frozen bootstrap-stability
gate: median adjusted Rand index is 0.113 and q10 is -0.009, versus required
0.80/0.50. Its median minimum resampled cluster contains only two episodes.
The two families are therefore descriptive, not reproducible regimes.

The frozen historical detector was then scored, without refitting, on the
current 114,096-row strict-OOF panel using an exact 28-field cross-era contract.
This is cross-model transfer diagnosis, not current detector OOF.

- The June 8 catastrophic episode is already at the 99.10th percentile of
  destination risk three hours before onset and the 98.91st percentile one
  hour before onset.
- The June 11 primary episode is only at the 78.60th percentile three hours
  before onset and falls to the 8.61st percentile at onset.
- Destination-risk adjustment improves June by **+2.00 bps**.
- The max of transition and destination risk improves the aggregate pooled
  global top-10% by **+4.42 bps**, from -5.89 to -1.47 bps.
- It changes May by 0.00 bps and worsens July by **-0.15 bps**.
- Every adjusted aggregate and July book remains negative.

Artifact:
`data_perp/artifacts/failure_first_detector_current_transfer_20260726_v6/`.

Decision: the failure-first idea has partial cross-era signal, particularly for
the June 8 catastrophic break, but no stable five-to-eight regime taxonomy and
no promotable detector. Retain the transition/destination probabilities as
research features only. The next model ablation should use direct supervised
binary failure/transition targets with class weighting or focal loss, rather
than forcing additional unsupervised states. Current-model chronological OOF
history and a fresh later block remain mandatory before any trust overlay can
enter admission.

### Direct binary failure detector ablation

The next bounded ablation removes the unsupported multi-state taxonomy from
the supervised target. It predicts failure onset within three hours and
failure active now or reached within three hours directly from health bins.
Every label retains exact outcome availability and evaluation-origin
boundaries.

The historical chronological OOF grid compares balanced versus unweighted
CatBoost geometries, then ablates market state, model health, exact 1h/3h
causal deltas and BOCPD separately. The full feature contract is capped at 40.
The HPO winner is the balanced depth-six arm. The research feature winner is a
20-field model-health plus transition-context arm:

| Metric | Aggregate | Latest historical month |
|---|---:|---:|
| Failure-onset AUC | 0.513 | 0.553 |
| Active-or-within-3h risk AUC | 0.544 | 0.589 |
| Mapped global top-10 | -145.43 bps | -121.76 bps |
| Risk-adjusted global top-10 | -138.28 bps | -137.98 bps |
| Increment versus mapped | +7.15 bps | -16.23 bps |

No feature block passes. Model health alone has better latest economics than
the selected classifier arm but still worsens the latest book by 3.29 bps.
BOCPD is not incremental in the full architecture: removing it improves both
aggregate/latest risk AUC and reduces the latest economic damage. Causal
deltas improve some latest classification metrics but not stable economics.

The frozen historical binary detector transfers to the current panel as a
diagnostic:

- current failure-onset/risk AUC: 0.339/0.323, inverse to the desired label;
- mapped aggregate global top-10: -5.89 bps;
- risk-adjusted aggregate: +1.61 bps, a +7.50 bps delta;
- mapped July: -55.02 bps;
- risk-adjusted July: -54.25 bps, a +0.76 bps delta.

The positive reused aggregate is not failure-detector validation: the model is
anti-discriminative on the two current failures, July contains no positive
failure labels, and historical latest-period economics worsen. The arm remains
research-only and cannot enter admission.

Artifact:
`data_perp/artifacts/failure_first_binary_ablation_20260726_v1/`.

The same frozen binary detector was then scored on the separately provenance-
flagged July 11-19 frozen-final-fit forward-OOS cohort. All 7,112 rows have
resolved exact 12-hour outcomes and raw-H0 coverage; none enters fitting,
feature selection, HPO or failure discovery. The 149 scored hours contain no
positive direct failure labels, so classification AUC is unavailable.
Economically, the mapped pooled-global top-10 is -163.71 bps, the onset
adjustment is effectively unchanged at -163.53 bps, and the active/near-term
risk adjustment worsens it by 5.83 bps to -169.54 bps. This resolved forward
block rejects the overlay.

Artifact:
`data_perp/artifacts/failure_first_binary_forward_july19_20260726_v1/`.

## 2026-07-26 pooled symmetric regime-transition research

Status: **data/label/model infrastructure implemented; active-state and
destination heads are useful research signals, but advance onset timing is
not reliable enough to gate trading**.

This work replaces the structurally mismatched six-hour-bin targets for the
regime-transition research workstream. It deliberately follows the approved
research exception: older and newer periods are pooled and walk-forward
validation is not required. It remains research-only and is not production
promotion evidence.

### Corrected data and label contract

The canonical market spine is one causal market row per hour from the compact
transition store. It covers January 1, 2023 through July 12, 2026:

- 30,931 hourly rows;
- one internal missing-hour boundary, represented as two exact-lag segments;
- 58 market-state level fields;
- all 28 pre-existing regime-transition fields;
- 120 newly materialized exact 3h/6h/12h/24h velocity and short-versus-long
  robust shift fields.

Each source bar at `t` is assigned to a decision at `t+1h`. Exact lags reset at
gaps. No outcome, taxonomy or future field is used as a model input.

The replacement target is symmetric and phase-aware:

- origin state: `[-12h,-3h)`;
- approach: `[-12h,-6h)`;
- acceleration: `[-6h,-3h)`;
- immediate lead: `[-3h,0h)`;
- active transition: from `0h` until the first three-hour-persistent
  destination, capped at `+6h`;
- early destination: transition end through `+6h`;
- settled destination: `[+6h,+12h)`.

The exact `-48,-24,-12,-6,-3,0,+3,+6,+12h` event-study grid is materialized.
`onset_within_3h` fires only for a future onset; recovery never counts as an
onset. Destination is the persistent settled state, never the first changed
hour.

A pooled outcome-free KMeans comparison over five to eight states selects
five states: silhouette 0.582 and minimum state share 1.22%. It yields 151
durable transitions. The earlier six-state sensitivity produces 526 noisier
events, negative silhouette and an unsupported minimum state share; it is not
the canonical result.

Artifacts:

- `data_perp/artifacts/regime_transition_research_20260726_v3/`;
- implementation in `extreme_price_movements/regime_transition_research.py`;
- runner `scripts/materialize_regime_transition_research.py`.

### Native hourly economic overlay

The economic overlay is also rebuilt without six-hour replication. It uses
the admitted candidate membership at each actual decision hour, compares
`[-12h,0)` with `[0,+12h)`, and records the maximum exact outcome-resolution
timestamp. The available exact OOF generations contribute:

- 9,090 hourly economic rows;
- 25 persistent adverse economic episodes;
- three of the canonical 151 market transitions within six hours of an
  economic failure.

This support is enough to materialize and audit the relationship, but not
enough to train a stable multi-class economic failure taxonomy. Do not force
five-to-eight economic failure states from 25 episodes.

### Grouped pooled classifier results

All reported metrics use five-fold stratified grouped validation. Every
transition window remains in one fold; benign controls are grouped by
calendar week. This is deliberately non-walk-forward, but it is not random
row validation.

The nested LightGBM HPO and feature-count sweep selects the expressive
212-field market model. Feature selection is performed inside each training
fold. For onset timing:

| Horizon | Prevalence | PR-AUC | Lift/base rate | ROC-AUC |
|---:|---:|---:|---:|---:|
| 1h | 0.49% | 0.063 | 13.00x | 0.853 |
| 3h | 1.46% | 0.133 | 9.06x | 0.799 |
| 6h | 2.93% | 0.152 | 5.18x | 0.756 |
| 12h | 5.86% | 0.211 | 3.61x | 0.720 |

At the most useful conservative 3h operating point tested, event recall is
21.9% with 2.84 false alert episodes per 30 days and three-hour median lead.
This is genuine early-warning signal, but insufficient as a standalone
admission or trust gate.

Once the transition has started, recognition is materially stronger:

- active-transition PR-AUC 0.340 at 1.94% prevalence;
- active-transition ROC-AUC 0.959;
- active-transition F1 at 0.5: 0.353.

Settled-destination prediction across five states is also useful:

- balanced accuracy 0.612;
- macro-F1 0.596;
- multiclass log loss 0.774;
- 1,054 lead/active rows from 151 held-out event groups.

The seven-class phase head is not ready: balanced accuracy 0.329 and macro-F1
0.257. Keep separate binary onset/active heads plus the conditional
destination head; do not use one monolithic phase classifier.

Artifacts:

- `data_perp/artifacts/regime_transition_classifier_ablation_20260726_v2/`;
- `data_perp/artifacts/regime_transition_lightgbm_hpo_20260726_v1/`;
- `data_perp/artifacts/regime_transition_active_head_20260726_v1/`.

### Feature conclusions

What works:

- causal current-state geometry—state age, switch count, nearest-centroid
  distance and margin—is the strongest compact risk context;
- market levels plus full transition history outperform transition transforms
  alone;
- nested LightGBM HPO improves 3h PR-AUC from 0.103 to 0.133 and ROC-AUC from
  0.764 to 0.799;
- model-health distributions are incremental on the shorter old55 period.

What does not work:

- the 28 pre-existing transition transforms alone are nearly uninformative
  for exact 3h onset (CatBoost PR-AUC 0.018, ROC-AUC 0.521);
- adding the new velocity block without market levels/state context is weak;
- forcing a single seven-class phase model;
- using advance-onset risk as a hard trading gate at current event recall.

The outcome-free old55 model-health overlay covers 11,921 hours from March
2025 through July 10, 2026 and contains 62 hourly distribution features. On
the same 47-event subset, market plus model health improves PR-AUC from 0.032
to 0.040 and ROC-AUC from 0.723 to 0.740. That is a 26% relative PR-AUC
improvement, but the small event count and old55 lineage make it sensitivity
evidence only.

Artifact:
`data_perp/artifacts/regime_transition_model_health_ablation_20260726_v1/`.

### Next required ablations

1. Treat the active head and destination head as supporting context, not an
   admission override.
2. Improve the onset head with explicit multi-horizon auxiliary training:
   jointly learn 1h/3h/6h/12h hazards and time-to-onset, then test whether
   cross-horizon disagreement improves the exact 3h operating curve.
3. Add a compact causal change-point block (one BOCPD/PELT family only) and
   retain it only if event recall improves at fixed false alerts per 30 days.
4. Materialize current-lineage model-health distributions. The old55 overlay
   proves possible incremental value but cannot establish current parity.
5. Extend the compact market spine through July 21 from the full feature
   store; do not bridge the July 11 missing hour.
6. Materialize the 92.69%-complete December 2025 exact one-minute outcome
   paths. This is the highest-value missing economic month.
7. Expand economic failures materially beyond 25 episodes before attempting
   failure archetype clustering or an economic transition head.
8. Report event recall and destination confusion by source state and year.
   The pooled result may conceal rare state-pair failures even without a
   walk-forward requirement.

## Base-rank IC versus execution-EV conversion workstream

Status: active. The paired canonical audit confirms that improving base-target
rank IC does not imply improving cost-aware execution value. Within the
selected pooled global top 10%, native-target IC improves
0.098 -> 0.123 -> 0.150 from February through April, while exact net is
-50.87 -> -83.03 -> -58.35 bps. February-to-March ordering contributes only
+0.34 bps; rank-to-economics conversion contributes -32.75 bps.

Completed infrastructure and evidence:

- canonical 509,868-row exact panel with frozen 31-long/8-short inputs,
  12-hour opportunity/payoff labels and causal 3h/12h regime transitions:
  `canonical_opportunity_payoff_trust_panel_20260729_v2`;
- pooled global top-1/5/10/20 tail attribution:
  `canonical_base_ic_ev_tail_diagnostic_20260729_v1`;
- residual-tier opportunity/payoff/direct-net/trust ablation:
  `historical_execution_ev_opportunity_payoff_trust_ablation_20260729_v1`;
- full-base side-local CatBoost target/context/regime/base-feature ablation:
  `canonical_full_base_opportunity_ablation_20260729_v1`.

Important qualification: the full-base v1 artifact now contains
`INVALIDATION.json`. Its leave-fold-out development mapper was not nested with
respect to the held fold, and mapped development economics influenced
arm/geometry selection. Do not use its winner or mapped development metrics as
OOF promotion evidence. Predeclared fixed-arm April diagnostics and the broad
failure result remain informative.

What the completed ablations establish:

1. The economic opportunity event is learnable: untouched-April ROC-AUC is
   0.643 for `gross > cost` and 0.655 for `gross > cost + 25 bps`.
2. Opportunity probability is insufficient for trading. Current
   probability-times-magnitude, exit-mixture, stack and trust arms all
   underperform the frozen controls at pooled global top 10%.
3. Higher opportunity precision can coexist with worse net EV. Several
   regime/base-feature arms reach approximately 58% precision but lose
   85--101 bps, implicating conditional payoff asymmetry, adverse-loss
   severity and exit conversion.
4. Raw score context helps direct net by about 10 bps at top 10%, but remains
   negative and below the raw base.
5. The current hierarchical mapping is harmful and can collapse the selected
   book toward one side.
6. DAE/GMM effects are target-dependent and non-robust.
7. Diagnostic configuration ranking transfers weakly from development to April
   (top-10 Spearman approximately 0.44), but the affected development mapping
   was not nested. Repair selection before treating this as a clean transfer
   estimate. No portfolio replay is authorized.

Next required ablations, in order:

0. **Explain the base-IC / execution-EV divergence on a fixed cohort.** The
   new February--April observation is not merely an architectural
   justification: base-target rank IC rises from 0.155 (February) to 0.162
   (March) to 0.226 (April), while the corresponding books ranked by the
   base score have negative exact 12-hour execution outcomes (-59, -91 and
   -38 bps) at global top decile. These are not results from a separately
   trained direct execution-EV head. On exactly the
   same eligible candidate IDs, global top-k rule and realised 12-hour paths,
   publish a monthly/side/base-score-decile waterfall:
   base-target rank IC -> gross opportunity/payoff -> exit conversion ->
   explicit costs -> net execution EV. Then separately hold the base ordering
   fixed and test (a) base-score calibration/common-unit mapping, (b) changing
   candidate composition and side/asset share, and (c) conditional loss-tail
   severity. The diagnostic must report raw-score and rank-only top-k controls,
   mapping coverage/reference support, turnover/concurrency, opportunity rate,
   favorable-payoff scale, MAE/stop incidence, cost and net EV. This distinguishes
   an alpha-label mismatch from a selection/mapping artifact or a real
   deterioration in cost/exit conversion; it must precede any claim that a
   higher base IC is economically beneficial.

   Treat the apparently contradictory direction as a falsifiable workstream,
   not as an expected consequence of the layered architecture.  In
   particular, April improves not only on the native base target but also on
   exact-net rank IC while its selected execution tail remains negative.  The
   next diagnostic must therefore distinguish:

   - broad monotone ordering that does not survive in the extreme tail;
   - score compression or month-varying score calibration near the admission
     cutoff;
   - changing side, asset, spread, volatility, opportunity and regime
     composition;
   - favorable-payoff prevalence/scale versus adverse-tail frequency and
     severity;
   - base-label horizon/shape mismatch and censoring;
   - deployed-exit conversion and explicit cost drag; and
   - instability caused by ties or small score perturbations at global top-k.

   For each month and side, publish full-sample IC together with top
   1/5/10/20% response curves, gross opportunity rate, conditional favorable
   payoff, conditional adverse payoff, stop/timeout/target exit shares, cost,
   net EV, loss rate, CVaR and bootstrap uncertainty.  Then perform
   rank-preserving month swaps and fixed-composition reweighting so the
   February-to-March and March-to-April EV deltas are attributed numerically.
   Repeat the comparison for base, residual and execution-EV scores on the
   same rows.

   Selection evidence must use the policy actually traded: strictly causal
   recent-EV mapping followed by one pooled-global top-k.  Raw-score
   selection is a diagnostic arm only.  Any fold used to select feature
   groups, targets or geometry must build its map solely from outcomes whose
   labels resolved before that fold; warm-up rows without adequate reference
   support are not admitted.  This closes the discovered mismatch in the
   first conversion-residual draft, which selected its March winner on raw
   challenger scores even though April was ranked after causal mapping.
1. Train a joint economics decomposition on identical strict-OOF rows:
   `P(opportunity)`, conditional favorable payoff, conditional adverse
   payoff/loss severity, and exit-policy conversion. Keep direct net as the
   primary task and use the decomposed heads as support.
2. Compare expected-net composition with an economics-aware global-tail
   ranking objective. Selection and evaluation must be one pooled global
   top-k after a common-unit map, never per timestamp or per side.
3. Replace independent side isotonic maps with a pooled causal anchor plus
   side residual shrinkage. Add abstention when recent conversion support is
   weak or the predicted side share breaches the balance gate.
4. Train regime/transition targets for opportunity base rate, payoff scale,
   loss severity and exit-conversion change. Test score-by-regime
   interactions and conditional experts only after the conversion targets
   are shown to be observable.
5. Reweight/match months on side, asset, volatility, candidate-group size,
   opportunity prevalence and payoff scale to distinguish covariate shift
   from conditional-conversion shift.
6. Repeat on older exact months and July-only OOF blocks. A conditional
   expert is allowed only where the regime can be identified causally at
   decision time and improves a later untouched period.
7. Investigate the regime-dependent raw-base top 1% result
   (April +16.62 bps, February -8.90, March -20.46) as a possible abstention
   or depth-control signal; do not promote it without stable later-period
   evidence.
8. Replay through portfolio constraints only after pooled global tail,
   latest-month/week, side-balance, calibration and beat-control gates all
   pass.
9. Repair the full-base experiment: use raw OOF target scores for bounded
   arm/geometry selection, or generate nuisance predictions nested with
   respect to each mapped validation fold. Fit the final pooled common-unit
   mapper only after model selection is frozen.

Completion update (2026-07-30): priority 0 is now materialised as
`mandatory_ic_ev_waterfall_20260730_v1`. The February -> March top-10 change
is -32.17 bps with only +0.26 bps attributable to rank-cell composition;
positive-payoff scale and positive prevalence contribute -21.88 and -21.70
bps. March -> April recovers +24.68 bps mainly through lower full-stop
prevalence and higher positive prevalence. May -> July deteriorates further,
culminating at -143.89 bps in July. Therefore mark the measurement task
complete but keep the causal diagnosis active: exact-12h versus legacy-24h
base labels, matched/reweighted month pairs, conditional payoff/exit heads and
compact score-by-transition interactions are the next experiments.

Older-data transition readiness is also complete. Exact 2022-23 context,
causal global mapping, before/after labels and a 90-field common semantic
geometry now feed a 17,320-row pooled panel. The first grouped/purged screen
is negative: pooled AUC is 0.505 active, 0.491 onset, 0.484 recovery and 0.342
reversal; current strict-OOF performance is at chance or weak. Cross-source
transfer is asymmetric and 2022-23 does not transfer to current. Next test
sparse mechanism groups, compact base-score interactions, source-held-out
objectives and July-local/adjacent-week recurrence. Treat constant-score
top-decile cells as invalid and add an explicit tie-aware evaluation gate.

### 2026-07-30 next-tranche completion update

The exact horizon-parity, matched-shift, direct-primary hurdle, sparse
transition and July-local tasks are complete:

- `febapr2025_exact12h_legacy24h_base_label_parity_20260730_v1`;
- `matched_month_pair_conversion_shift_20260730_v1`;
- `exact_strict_oof_hurdle_distributional_ablation_20260730_v3`;
- `pooled_historical_current_sparse_transition_mechanism_ablation_20260730_v1`;
- `july_local_exact_h12_transition_diagnosis_20260730_v1`.

Decisions:

1. Reject the native-12h base replacement.  It improves native-12h IC but
   worsens exact execution IC and global top-10 net.
2. Treat the supported June -> July failure as conditional response shift:
   -53.11 bps after matching, versus +5.45 bps from composition.
3. Freeze gross-cost hurdle EV as the next research comparison control.  It
   improves mapped top-10 versus direct in both forward windows but remains
   -81.35/-71.55 bps, latest-negative and side-unstable.
4. Reject the current joint MultiRMSE and stopped-gradient architectures.
5. Retain only compression/onset and memory/state active mechanisms for
   narrow uncertainty/interaction follow-up.  No transition head is a veto
   or router.
6. July-local transition learning is not established.  Strict OOF has only
   two weeks; the broader diagnostic is below chance.

Next executable queue:

1. build July 20--23 causal map coordinates, pooled-global before/after labels
   and the strict common transition geometry;
2. run frozen direct/hurdle blend weights and pooled-anchor plus side-residual
   mapping repair;
3. test the two retained transition mechanisms only as bounded
   hurdle/uncertainty interactions;
4. extend exact incidents and require untouched later-period confirmation;
5. keep simple-policy and portfolio replay blocked until mapped global,
   latest-period, side-balance and beat-control gates all pass.

### Completed chronological repair and joint decomposition

The replacement experiment is complete:

- `canonical_full_base_joint_economics_decomposition_20260729_v1`;
- `canonical_full_base_joint_economics_summary_20260729_v1`;
- `canonical_full_base_joint_economics_pooled_common_mapping_ablation_20260729_v1`.

It uses 334,298 pre-April-resolved development rows and 254,894 chronological
OOF rows. The 3,120 nominal late-March validation rows whose 12-hour outcomes
resolve after the April freeze are excluded. Feature arms and geometry are
frozen; no selection or HPO consumes April.

Results:

1. S1+B materially improves opportunity AUC and conditional favorable/adverse
   rank IC, including in April.
2. It does not improve pooled global economics. Every development and April
   global top-10 score is negative.
3. The useful side changes by period. Development S1+B shorts can be positive
   while longs are strongly negative; April S0 longs become positive while
   shorts are strongly negative.
4. Static side residual calibration collapses April toward one side.
5. Prior-OOF side normalization improves S0 April top-10 by about 21.7 bps but
   worsens development and the latest week.
6. No arm passes the global, latest-week, balance and beat-control gates. No
   portfolio replay is authorized.

Architecture implication: retain the component heads as support, but do not
rank directly on their current absolute scales. The next admissible feature is
an OOF prediction of *conversion change*, not a realized transition label or a
static side preference.

### Materialized conversion-transition targets

`canonical_economic_conversion_transition_labels_20260729_v1` now provides
85,440 outcome-only cohort rows for H=12h primary and H=3h auxiliary windows.
Each row is global hour × side × frozen causal base-score decile and contains
before/after/delta values, support and actual target-availability timestamps
for:

- opportunity prevalence;
- favorable payoff scale;
- adverse loss severity;
- four-class exit incidence and conditional payoff;
- exit-mixture expected net; and
- direct mean net.

### Completed causal context and transition-head learnability

The first causal observability test is complete:

- `canonical_economic_conversion_transition_context_20260729_v1` contains
  42,720 unique anchor-time cohorts and 47 whitelisted decision-time features:
  frozen score context, five compact market levels, eighteen exact 3h/12h
  pre-entry deltas, eight regime composites, side and score decile;
- `canonical_economic_conversion_transition_head_ablation_20260729_v1`
  contains 292,800 chronological OOF predictions from five expanding folds.
  Training rows are admitted only when their actual after-target availability
  precedes the validation boundary. The experiment uses fixed geometry and no
  feature selection or HPO.

The H=12h aggregate results show that conversion change is observable:

| Target change | Rank IC | Sign AUC | Sign AP | Model MAE | Constant MAE |
|---|---:|---:|---:|---:|---:|
| Opportunity prevalence | 0.455 | 0.707 | 0.691 | 0.1704 | 0.1924 |
| Adverse severity | 0.400 | 0.688 | 0.661 | 0.00913 | 0.00990 |
| Direct/exit-mixture net | 0.473 | 0.716 | 0.707 | 0.01216 | 0.01382 |
| Favorable payoff scale | 0.144 | 0.580 | 0.567 | 0.00474 | 0.00479 |

Direct mean net and reconciled exit-mixture net are the same accounting
quantity in this materialization; do not count them as two independent tasks.
The H=3h sensitivity preserves useful opportunity/direct ranking, but
favorable payoff magnitude is essentially unlearned.

Stability is not yet adequate. In the latest H=12 fold, opportunity remains
above baseline (IC 0.305, AUC 0.627 and MAE 0.1299 versus 0.1348), while
direct/exit net falls to IC 0.248 and AUC 0.636 and its MAE is worse than the
constant baseline (0.00829 versus 0.00809). Adverse-severity MAE also loses to
the constant baseline. This is precisely the regime where a router is needed,
so aggregate learnability is insufficient.

Updated next actions:

1. Run feature-group ablations: score/context only; market levels plus exact
   deltas; regime composites; and their bounded combinations. Require
   latest-fold improvement, not only aggregate IC/AUC.
2. Test support-aware weighting, target smoothing and shared H3/H12
   multi-task training. Keep actual target-availability purging.
3. Redesign favorable payoff scale with robust log/quantile/range targets and
   higher-support cohorts; drop it if it still fails the constant baseline.
4. Diagnose the base-IC/EV divergence explicitly by joining each OOF
   conversion prediction back to frozen score deciles and measuring whether
   predicted opportunity, adverse severity or exit-mixture change explains
   the monthly collapse after holding the alpha ordering fixed.
5. Add the transition predictions to admission only as one grouped
   score-by-conversion interaction after a later-fold stability gate passes.
   Do not attach realized transitions, duplicate direct/exit targets, a static
   side preference or a hard regime veto.
6. Repeat on older exact months and July-only OOF blocks. A conditional expert
   or portfolio replay remains prohibited until pooled-global top-k,
   latest-fold/week, side-balance and beat-control gates pass.

### Completed feature, target and frozen-tail attribution ablations

The next three immutable experiments are complete:

- `canonical_economic_conversion_transition_feature_group_ablation_20260729_v1`;
- `canonical_economic_conversion_contribution_labels_20260729_v1` and
  `canonical_economic_conversion_transition_target_ablation_20260729_v1`; and
- `canonical_base_conversion_prediction_attribution_20260729_v1`, summarized
  by `canonical_conversion_transition_workstream_summary_20260729_v1`.

The feature experiment compares eight fixed H12 arms on 936,960 OOF
prediction rows. Every arm retains side and frozen score-decile controls; the
matrix separates score geometry, generic market levels/deltas, regime levels
and regime-transition deltas. No arm passes both the opportunity and
direct-net gates:

1. Market plus regime context carries most of the signal. For direct-net
   change it reaches aggregate IC 0.484 and latest IC/AUC 0.264/0.648, but
   latest MAE remains worse than the constant baseline (0.00822 versus
   0.00809) and both recent folds do not beat the baseline.
2. Score plus regime repairs latest direct-net MAE (0.00795 versus 0.00809),
   but loses too much discrimination: latest IC/AUC fall to 0.197/0.603 and
   aggregate IC falls to 0.240.
3. Full context remains the only arm passing the opportunity gate. Score-only
   and regime-transition-only arms are weak; transition deltas are not a
   sufficient standalone explanation.
4. No feature arm advances to a frozen admission interaction or portfolio
   replay.

The target redesign resolves the weak favorable-payoff label:

| H12 target | Development IC | Confirmation IC | Development MAE gain | Confirmation MAE gain | Gate |
|---|---:|---:|---:|---:|---|
| Old conditional favorable mean | 0.147 | 0.092 | 1.1% | +0.000028 | Fail |
| Support-weighted conditional mean | 0.149 | 0.098 | 1.2% | +0.000029 | Fail |
| Empirical-Bayes conditional mean | 0.107 | 0.140 | 0.3% | +0.000040 | Fail |
| Raw unconditional upside contribution | 0.388 | 0.231 | 7.1% | +0.000102 | Pass |
| Robust unconditional upside contribution | 0.391 | 0.246 | 7.6% | +0.000126 | Pass |
| Unconditional loss contribution | 0.480 | 0.219 | 11.3% | -0.000262 | Diagnostic; latest MAE fail |

Unconditional upside is `mean(max(net,0))`; unconditional loss is
`mean(max(-net,0))`. The raw pair reconciles direct mean net exactly on 85,400
resolved transition rows. This is economically cleaner and avoids the
low-support difference of conditional positive means. The robust upside arm
is the leading supporting label, but it is not an admission score.

The proposed soft net-positive-rate label is **not independent evidence**. On
every complete H3 and H12 window it is exactly equal to the existing
`opportunity_probability_0bps` label. Keep one head only.

The frozen-tail attribution joins seven OOF transition predictions to the
unchanged March/April pooled-global base top-1/5/10/20 books by exact
hour/side/frozen score decile. It does not rerank candidates. Results reject
an immediate score interaction:

- within the actual base top 10%, candidate-level net IC is approximately
  -0.02 to +0.03 in March and -0.02 to +0.01 in April across the tested heads;
- every predicted-high versus predicted-low daily-block 95% interval crosses
  zero;
- March-to-April top-10 net improves by +23.26 bps, but fixed predicted-state
  composition explains at most 4.54 bps in absolute value; 22.45--27.80 bps
  remains within-state conversion;
- therefore the heads learn broad cohort transition labels but do not yet
  identify the economic conversion change inside the traded alpha tail.

Revised next work:

1. Replace timestamp-side score deciles with a label geometry closer to the
   traded book: causal recent common-unit EV/score bands and an explicit
   high-alpha-tail contribution target. Do not select per timestamp or side;
   the evaluation remains one pooled-global top-k after the recent mapping.
2. Add absolute score/mapping-support and causal admission-distance context so
   the environment head can distinguish the candidates that actually reach
   the global book, rather than assigning one transition value to every member
   of a broad relative decile.
3. Extend exact history before February and through July, freeze candidate
   designs on development folds, and require a later-month confirmation. The
   present March/April attribution cannot test February because 28 days of
   prior resolved history are required.
4. Only after a head significantly stratifies frozen-tail economics should it
   enter as one bounded score-by-conversion interaction. Timing, MAE,
   target-price and wait actions remain in the separate action layer.

### Completed shared H3/H12 auxiliary-learning test

`canonical_conversion_shared_horizon_ablation_20260729_v1` compares an H12-only
model with a pooled H3/H12 model for opportunity, direct net, adverse severity
and robust unconditional upside. Both use the same 47 causal features, folds
and fixed CatBoost geometry. Pooled targets use fold-local training median/MAD
normalization and equal total H3/H12 loss mass. The only added input is the
known horizon indicator; no same-anchor H3 outcome or prediction is an H12
feature.

No target passes the strict all-metric shared-horizon gate:

| Target | Development effect | Confirmation effect | Verdict |
|---|---|---|---|
| Direct net | MAE +0.000063, IC -0.0023, ECE 0.021→0.051 | MAE 0.00831→0.00784, IC 0.249→0.291, AUC 0.636→0.663 | Latest regression repaired, development calibration fails |
| Opportunity | MAE +0.00173, IC -0.0083 | MAE 0.12992→0.12744, IC 0.305→0.312, AUC 0.627→0.638 | Latest improves, development non-inferiority fails |
| Adverse severity | MAE +0.000053, IC -0.0100 | MAE 0.00689→0.00667, IC 0.127→0.172, AUC 0.564→0.586 | Better latest, still misses constant MAE 0.00659 |
| Robust upside | MAE +0.000028, IC +0.0004 | MAE 0.002270→0.002262, IC 0.244→0.257; AUC worsens | Mixed |

This rejects one fully shared head, but supports a narrower next ablation:

1. share H3/H12 structure for the continuous direct-net regression only;
2. keep the sign/calibration classifier H12-only;
3. choose a bounded H3 loss weight from development folds only (0%, 10%, 25%,
   50%), freeze it, and use the truncated final fold once for confirmation;
4. require the shared regression to beat the confirmation constant without
   worsening development MAE/IC beyond the frozen non-inferiority margin.

This hybrid-head ablation is secondary to repairing the global-tail label
geometry. Better latest transition regression still does not overcome the
failed frozen-tail economic attribution.

## 2026-07-30 January--July 2022 separate-population extension

The earlier-history data/label prerequisite is complete. The final population
is a paired hourly long/short grid over five continuously available Kraken
inverse perpetuals, not a backward relabelling of the August-2022+ frozen PF
population.

Completed and gated:

1. 50,880 causal candidates, 25,440 per side, from January 1 through July 31.
2. Exact PI product binding and two-boundary availability probes.
3. 30-day pre-January minute warm-up and 720-minute post-decision paths.
4. Independent 50,880/50,880 candidate-level 720/720 coverage proof.
5. Forty-four causal asset/market features plus 25 transition-dynamic fields.
6. Explicit deployed `long__parent` / `short__parent` exit geometry.
7. Direct 12-hour policy net target, physical path labels, soft triple
   barrier, five established auxiliary heads and supporting path labels.
8. Exact provenance propagation through stage, product map, coverage, policy,
   timing, paths and final joined labels.

Hard boundaries:

- inverse quote-notional returns are not USD-linear PF returns or inverse
  collateral ROE;
- current spread/fee economics are counterfactual because historical L2 is
  unavailable;
- this population is non-OOF and non-promotable;
- do not pool its calibration or headline economics with later PF candidates;
- do not count it toward the prospective incident gate.

Required next ablations, in order:

1. Build one hash-bound research panel by exact candidate ID from the
   feature-preserving stage and the v2 multi-task labels. Keep a categorical
   population-lineage indicator whenever later populations are evaluated in
   the same report.
2. Run fixed-geometry non-walk-forward calendar-block diagnostics inside this
   lineage. Use month blocks, side-local fits and a pooled-global top-k
   evaluation after mapping. These are learnability/recurrence diagnostics,
   not promotion folds.
3. Primary architecture:
   direct `execution_net_ev_12h` regression plus sign/calibration classifier.
   Auxiliaries are representation regularizers only: opportunity incidence,
   unconditional upside contribution, adverse competing risk/severity,
   conversion loss, timeout and the five path heads.
4. Transition interaction ablation:
   no transition fields; 25 transition dynamics only; asset/market levels;
   levels plus transition dynamics; bounded score-by-transition interactions.
   Require latest-month economics and side coverage, not aggregate IC alone.
5. Base-IC/EV divergence:
   measure within-side rank IC, pooled-global top-k gross/net conversion,
   cost hurdle, opportunity incidence and conversion loss on identical rows.
   The simple grid momentum score is only a control; do not call it the
   production alpha base.
6. Cross-lineage transfer:
   train on January--July inverse-grid research and test only representation/
   relationship recurrence on later separately reported PF rows, then reverse
   the direction. Never use a shared monotone EV calibration across the two
   contract/population lineages.
7. If transition interactions recur, freeze a compact context recipe and test
   it on later untouched months. Keep timing, MAE, target-price and wait
   actions in the separate action layer.

Do not start another broad transition HPO. The purpose of this extension is
to identify stable opportunity/conversion mechanisms and regime-transition
interactions under a deliberately different candidate population, not to
manufacture a larger pseudo-OOF promotion set.

### Status update: separate-population ablation completed

The different January--July 2022 candidate population is accepted and fully
materialised.  Exact-ID panel construction and the fixed-geometry diagnostic
are complete:

- panel:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_exact_id_research_panel_20260730_v1/`;
- authoritative result:
  `data_perp/artifacts/jan_jul_2022_inverse_pi_direct_utility_multitask_ablation_20260730_v2/`;
- `_v1` result is superseded by `_v2` because monthly selection is now pinned
  to signal month while retaining true execution timestamps;
- 14/14 focused tests pass.

Decision from the six matched arms:

- **retain as challenger:** market/asset levels plus exactly five bounded
  `base_score x transition z72` interactions and the six economic auxiliaries;
- **reject:** transition-only context;
- **reject:** unrestricted level-plus-transition concatenation;
- **reject:** adding the five path heads as one bundle;
- **do not promote:** every top-10 net result remains negative after the
  approximately 100-bps cost hurdle.

The bounded interaction arm improves top-10 net by +3.04 bps and worst-month
net by +14.76 bps versus direct-only, but remains -98.53 bps net
(+1.48 bps gross).  This is a representation clue, not a trading result.

Next executable queue:

1. Freeze the five z72 interactions; do not HPO a wider transition namespace.
2. On the canonical alpha candidates, compare direct-only versus the frozen
   interaction challenger on identical global books and causal recent EV
   maps.
3. Within the interaction arm, add path auxiliaries one at a time.  Start
   with future slope because it is the only path head with material pooled
   out-of-block rank IC here (0.060); use lower weights than the failed
   five-head bundle.
4. Rebuild the opportunity/reach-meaningful-MFE classifier with the approved
   ATR-normalised soft triple-barrier alternatives.  Current AUC is only
   0.516 at best in this lineage.
5. Run the canonical production base-target IC -> gross EV -> cost -> net EV
   waterfall on identical rows.  It cannot be answered by this older panel
   because its base soft target is entirely unavailable and its score is only
   the simple momentum control.
6. Perform bidirectional cross-lineage transfer only at the representation
   level.  Keep separate mappings, economic contracts and reports.

## 2026-07-30 explicit continuation: rising base IC versus weak execution EV

Treat the February--April result as a first-class conversion investigation,
not as a reason to change the intended architecture.  On identical canonical
rows, native base-target rank IC rises `0.155 -> 0.162 -> 0.226`, while exact
12-hour net rank IC also rises `0.090 -> 0.093 -> 0.143`.  Yet the
base-ranked global top-decile exact execution outcome is
`-59.39 / -91.31 / -38.45 bps` for February/March/April.  Thus the anomaly is
not a loss of average monotone ordering.  It is a failure of that ordering to
convert reliably in the admitted extreme tail after the deployed exit policy
and costs.

The completed identical-row waterfall already narrows the diagnosis:

- February -> March top-10 net deteriorates by 32.17 bps.  Holding
  side-by-score-ventile composition fixed explains only +0.26 bps; falling
  positive-payoff size and positive-net prevalence are the dominant terms.
- March -> April recovers by 24.68 bps despite a further IC increase.  The
  recovery is led by stop/prevalence changes, not simply by stronger alpha
  rank.
- Exact 12-hour native-label retraining improves native-horizon IC but worsens
  exact gross/net ordering and top-10 economics, so horizon mismatch alone is
  not the repair.
- Matched reweighting finds conditional-response shift after observable
  composition is controlled where overlap is adequate; February -> March
  still fails the overlap gate and requires more historical support.
- The corrected frozen-score-band diagnostic
  `frozen_month_score_band_transition_diagnostic_20260730_v2` shows that a
  fixed February top-decile threshold admits only 7.283% of March rather than
  10%; its net falls from -50.87 to -76.18 bps.  February's frozen top
  ventile also contracts from 5.00% to 3.06%, while its within-band net
  deteriorates by 17.17 bps.  By contrast, March's frozen top-decile threshold
  admits 9.634% of April and the highest-ventile within-band response improves
  by 30.49 bps.  The divergence therefore contains both score-scale/cutoff
  migration and a real change in conditional economic response; neither
  aggregate IC nor a local quantile cut can distinguish them.

Required executable continuation:

1. Report tail-local diagnostics on the same rows: quantile-local rank IC,
   top-1/5/10/20 opportunity precision/recall, favorable and adverse payoff
   scale, stop/timeout mixture, and calibration by month and side.  Side is
   attribution only; selection remains one pooled-global top-k after causal
   recent EV mapping.  Decompose each month's aggregate IC into admitted
   top-tail, near-cutoff and non-admitted contributions so an IC gain driven
   by the middle of the distribution cannot be credited to the traded tail.
2. Build a candidate-level target bridge from the native alpha label to
   exact 12-hour gross, exit-policy gross, explicit cost and net.  Attribute
   disagreement to horizon, exit conversion, cost hurdle and tail
   extrapolation separately.
3. Freeze score ventiles in the source month and measure their destination
   transition matrix, opportunity incidence and payoff distribution in the
   next month.  Run both fixed-numeric-threshold and fixed-quantile books so
   score compression/calibration is separated from rank preservation and
   payoff non-stationarity.
4. Repeat the matched month-pair analysis with older data and improved common
   support for February -> March.  Fail closed on coverage, ESS or balance;
   do not interpret unsupported propensity decompositions.
5. Run cost-hurdle sensitivity at fixed books, including zero cost and the
   deployed cost, to determine whether the tail fails before costs or only
   after costs.  Never reselect the book for each cost assumption.
6. Compare raw base, residual alpha, direct EV and mapped EV on the exact same
   candidate IDs and frozen books.  Measure where incremental layers improve
   average IC but damage tail precision, side balance or calibration.
7. Test only causally observable conversion interactions that can explain
   conditional response: base-score/margin x opportunity probability,
   favorable-contribution scale, adverse competing risk/severity,
   stop/timeout mixture and the frozen compact transition interactions.
   Timing, MAE, target-price and wait actions remain outside this score.
8. If one global conversion model still cannot transfer, test bounded
   uncertainty-weighted experts only after demonstrating within-period
   learnability and identifying the state with decision-time features.  No
   future regime labels, static side routing or per-timestamp selection.

Completion gate: the investigation closes only when the monthly change from
base rank to global top-k net EV is quantitatively reconciled on identical
rows, including latest-month/week coverage and tie-aware uncertainty.  A
model may advance only if its frozen mapped global top-10 is positive,
beats the direct/base controls, remains side-balanced and survives an
untouched later block.  Until then, rising base IC is evidence of alpha
learnability, not evidence of executable EV.

## 2026-07-30 completed controls: hurdle blending and cross-side mapping

Two previously queued no-refit controls are complete and sealed.

`frozen_hurdle_blend_ablation_20260730_v1` selects a blend weight exactly
once from 35,644 resolved May development-OOF rows.  Pooled-global top-10 net
worsens monotonically from `-112.18 bps` at 0% hurdle to `-128.04 bps` at
100%, so the frozen winner is **0% hurdle**.  Forward May -> June and later
July diagnostics show less-negative post-hoc arms at higher hurdle weights,
but every arm remains negative and those windows cannot alter the frozen
choice.  Pre-map ordering also reverses by period: direct is better in May,
while hurdle is better in July.  Therefore no stable direct/hurdle blend has
been demonstrated.

`hurdle_cross_side_common_unit_mapping_20260730_v1` compares the canonical
map with a causal 21-day pooled anchor and side-residual shrinkage.  The
shrinkage strength is frozen at 4,000 from pre-June OOF.  It improves later
July top-10 from `-71.55` to `-57.64 bps`, but remains negative and
cutoff-tie ambiguous.  In May -> June it falls to `-112.30 bps` and selects
0% long.  A single fixed cross-side residual map therefore does not transfer
and is rejected.

Consequences:

1. Keep direct net as the primary execution score.  The hurdle remains a
   support/control head, not a blended primary score.
2. Do not select the apparently better 75% July blend from forward outcomes.
3. Do not use side quotas to repair the mapping collapse.
4. Require a causal common-unit mapping to pass both forward windows, latest
   week, tie and side-balance gates before portfolio replay.
5. Continue next with July 20--23 materialisation, the identical-row
   IC-to-EV tail bridge, and bounded conversion interactions.  Portfolio and
   simple-policy replay remain blocked by negative mapped economics.

## 2026-07-30 completed July prerequisite and identical-row layer bridge

The original 5,760-row July 20--23 raw-score bridge correctly fails closed.
`july20_23_retrospective_allscore_transition_readiness_20260730_v1` proves
that its sealed contract excludes mapped EV/global admission, has no
hash-bound causal map state and lacks the strict common 90-field geometry.
It cannot legally extend July OOF or adjacent-week evaluation.

A separate, explicitly retrospective/non-OOF extension was materialised from
the authoritative v2 frozen scorer:
`july20_23_exact_h12_transition_inputs_20260730_v1`.  It preserves the same
5,760 candidate IDs, produces 4,380 honest causal-coordinate rows after
warm-up, 96 H12 anchors and 73 complete 90-field transition rows.  The
provenance-separated `july_local_exact_h12_transition_diagnosis_20260730_v2`
does **not** improve the learnability conclusion:

- active-adverse diagnostic grouped OOF: 299 rows/54 positives, AUC 0.301,
  AP 0.124 versus 0.181 prevalence and zero expected top-10 lift;
- active adjacent-week transfer: 179/24, AUC 0.190, AP 0.082 versus 0.134
  prevalence and zero lift;
- July 20--23 active alone: 71/2, AUC 0.580 but AP 0.048 and zero lift;
- onset diagnostic: only 64/3 usable predictions, AUC 0.383 and zero lift.

Therefore July transition routing remains unauthorized.  The extension is a
diagnostic incident only and does not increase strict OOF support.

`identical_row_four_layer_ic_ev_diagnostic_20260730_v1` then compares the
available layers on 140,682 exact common March--April canonical rows.  All
candidate identities and net/gross/cost/MFE/MAE/exit/opportunity labels are
identical; May--July fails closed because there is no causal mapped score on
the same canonical-alpha identities.

| Global March--April top 10% | Full net IC | Tail net IC | Opportunity precision | Gross bps | Cost bps | Net bps |
|---|---:|---:|---:|---:|---:|---:|
| Raw base alpha | 0.0884 | 0.1050 | 46.96% | +50.30 | 100.25 | -49.95 |
| Residual expected EV | 0.0869 | 0.0948 | 46.22% | **+73.78** | 100.37 | **-26.59** |
| Direct EV q25 | 0.0899 | 0.0305 | 46.59% | +44.77 | 100.22 | -55.45 |
| Causal mapped **base-alpha economics** | 0.0519 | 0.0079 | 43.22% | +34.35 | 100.17 | -65.82 |

Lineage correction: `mapped_direct_net` in the historical canonical panel is
an unfortunately named alias for the causal economics map of
`score_base_alpha`; it is **not** the q25 direct score after mapping.  These
are four identical-row score comparators, not four sequential pipeline
layers.  The residual comparator improves gross payoff scale despite slightly
lower IC.  The q25 direct challenger has weak pooled tail ordering, while the
mapped-base comparator has still lower tail IC and opportunity precision.
The approximately 100-bps cost makes all books negative, but it is not the
source of the between-score ordering because zero-cost results use the same
frozen books.  Raw-direct -> mapped-direct degradation is not yet measured.

Revised executable priority:

1. Preserve base and residual ordering as controls; do not infer that a
   higher full-distribution IC warrants replacing the residual book.
2. Repair the direct head against high-alpha-tail contribution, favorable
   payoff scale and adverse competing risk, with calibration assessed inside
   the frozen global tail.
3. Materialise a true causal mapping of the exact raw direct score, binding
   the raw-score column/model hash and mapping lineage.  Only then require
   that mapping is non-degrading versus raw direct on identical rows.
4. Materialise a comparable causal mapped score on current canonical-alpha
   identities before claiming May--July layer attribution.
5. Continue the compact transition mechanisms only as bounded uncertainty
   interactions; the July classifier cannot be a router or veto.
6. Supersede the v1 four-layer interpretation with a lineage-explicit v2;
   tests must fail if a mapped-score alias cannot prove its raw input score.

## 2026-07-30 bounded direct-tail repair: causal v2 result

The first repair artifact, `bounded_direct_tail_repair_20260730_v1`, is
invalid for April confirmation because it allowed late-March labels that
resolved after the first April decisions into model fitting/calibration.
`bounded_direct_tail_repair_20260730_v1_correction_20260730_v1` records that
invalidation under seal
`a45c94d7b919167e3e5f936cbf1d1db2b451b4d17a78d9f6680960a10f72fbe5`.

The authoritative result is `bounded_direct_tail_repair_20260730_v2`, seal
`fa57d8fe2947a20ae75c67a0de69001672a1c818537dd0c091880fcba9b4c362`.
It uses March chronological OOF for development, starts confirmation at
2025-04-01 01:00 UTC and requires every fitting/calibration label to end no
later than 2025-04-01 00:00 UTC.  Models are per-side, but all selection is
one pooled-global book.  No broad HPO or action-layer feature is used.

| Raw pooled-global top 10% | March matched OOF subset | April untouched confirmation |
|---|---:|---:|
| Incumbent direct q25 | -83.71 bps | -93.24 bps |
| Tail-weighted direct | **-67.49 bps** | -41.79 bps |
| Robust favorable-minus-adverse decomposition | -74.81 bps | **-33.49 bps** |
| Residual x conversion interaction | -72.42 bps | -72.13 bps |

The March column is the matched chronological-OOF subset used by all arms,
not the full-March q25 diagnostic; it must not be compared directly with the
earlier full-month `-21.76 bps` figure.  Both tail weighting and robust
decomposition improve the incumbent in development and confirmation, so the
repair direction is real, but no arm is positive.

The sealed reporting supplement,
`bounded_direct_tail_repair_20260730_v2_supplement_20260730_v1`, seal
`67fb02db6712b181b929d4cf0dbd7567010ffa1e801775e9d92791e3b71dbff0`,
adds tie-aware, calibration, side and concentration gates:

- robust decomposed raw April top-10 is -33.49 bps; its selected long
  contribution is +2.96 bps but short is -50.49 bps;
- its latest confirmation block is -91.14 bps, so aggregate improvement does
  not transfer to the latest regime;
- mapped decomposed deterministic/expected/best/worst top-10 is
  -32.87/-33.17/-10.03/-61.04 bps;
- the mapped decomposed cutoff plateau contains 2,029 rows (29.3% of the
  selected book); weighted has 1,601 rows (23.1%);
- raw decomposed calibration MAE is approximately 222 bps, far too large for
  threshold interpretation.

All promotion gates fail and no policy/portfolio replay runs.

Next actions:

1. Retain tail weighting and robust favorable-minus-adverse decomposition as
   challengers, not winners.
2. Diagnose the short-side and latest-block failure first; do not add a side
   quota.
3. Ablate strict-OOF meaningful-MFE/peak contribution and future slope
   support one at a time and together at bounded weights.  Keep timing, MAE,
   target-price and wait outside EV.
4. Replace plateau-heavy isotonic mapping only after the raw score is
   positive; mapping must report expected/best/worst tie allocation.
5. Require confirmation and latest-block positivity, stable side
   contribution and calibration non-inferiority before replay.

## 2026-07-30 current exact raw-direct -> mapped-direct lineage completed

The initial readiness audit,
`mayjul_identical_four_layer_mapping_readiness_20260730_v1`, correctly failed
closed because the persisted q25 rows lacked per-row score availability and
fold/fit provenance.  Those fields were then deterministically reconstructed
from the bound v3 dataset, original chronological-fold recipe, original
runner, frozen state/config and exact score output without refitting or
changing any q25 value.

The authoritative artifact is
`mayjul_exact_direct_q25_causal_mapping_20260730_v1`, seal
`d64c37b01f06333e2243a5d66571ab188d1c16585396737fbd73ddf5752cd038`.
It proves:

- all 127,777 `q25_net_bps` values are bit-identical to the waterfall's
  `score_direct_q25_challenger_bps`;
- every row has one reconstructed May/June/July fold, point-in-time features,
  `score_available_at = decision`, and maximum training-label resolution
  before its fold cutoff and decision;
- exact H12 labels and all four identity fields match;
- the 21-day map uses only labels resolved before each UTC-day snapshot;
- 125,551 rows map successfully after an honest 2,226-row May-1 warm-up.

Historical fold-model binaries were not persisted.  The successor binds the
exact OOF output, recipe/config/state and final-model lineage, but does not
claim that the frozen final binary is identical to each historical fold
binary.

| 125,551-row pooled-global book | Base | Residual | Raw direct q25 | Causal mapped direct q25 |
|---|---:|---:|---:|---:|
| Top 1% net | -49.49 | -46.92 | -44.05 | **-11.58** |
| Top 10% net | -94.00 | **-77.63** | -89.75 | -89.57 |
| Top 10% positive rate | 36.20% | 41.88% | 36.63% | 38.26% |

The aggregate top-1 improvement is composition-sensitive, not a stable
monthly repair.  At top 10%, raw -> mapped direct is:

- May: `-100.76 -> -115.25 bps`;
- June: `-42.91 -> -55.10 bps`;
- July: `-152.92 -> -178.16 bps`.

At top 1%, mapping worsens May and June but improves July
`-163.47 -> -88.52 bps`; the pooled `-44.05 -> -11.58 bps` therefore reflects
a changed cross-month allocation.  It must not be reported as universal tail
compression.  Mapping also creates cutoff ties and strong side shifts
including zero long selections in some June/July cells.

Decision:

1. The raw-direct -> mapped-direct lineage is now measurable and the earlier
   alias ambiguity is closed.
2. The current q25 model and map both fail global top-10/latest/month gates.
3. Do not promote the pooled top-1 improvement; require within-month/week
   confirmation and non-collapsed side coverage.
4. The residual score remains the best current pooled top-10 comparator, but
   is still -77.63 bps and non-promotable.
5. Continue direct-tail target/representation repair before another mapping
   design; no policy or portfolio replay.

## 2026-07-30 completed retained-transition interaction screen

The first joinability artifact correctly stopped at 81.57% later-July
coverage.  It is superseded by the completed exact-geometry chain:

- `forward_exact_transition_geometry_20260730_v1`, seal
  `752c78a39c0feb397f3bc5c19683a32d73a35e0df8c570f8385eeb156450e8b3`;
- `causal_retained_transition_mechanism_extension_20260730_v2`, seal
  `8cac9003ff8850a48241598d9f52e99f52ed4248cc186d3ca912e2b6eaa424db`;
- `causal_retained_transition_interactions_20260730_v2`, seal
  `2478433ddaac004aff441e25228a87b550a62df04f3c8b724555a89be1a01f68`.

The geometry builder exact-joins frozen candidates to 75 hash-bound feature
shards, reconstructs the same nine raw concepts into 90 per-side
median/IQR/gap/exact-1/3/12h-delta fields, preserves genuine missingness and
uses no fill or as-of join.  Coverage is now exactly 49,244/49,244 for
May -> June and 7,071/7,071 for later July.

Because prior fitted mechanism state was not persisted, the screen explicitly
performs a causal refit rather than claiming model-state reproduction.  The
three retained fixed recipes are compression/onset (10 fields),
memory/active (50) and state/active (9).  Every training label resolves before
the forward cutoff; temporal Brier shrinkage and the bounded
`0/.25/.50` interaction/penalty grids are selected from earlier strict OOF
evidence only.

| Pooled-global top 10% | May -> June | Later July |
|---|---:|---:|
| Common-unit control | -112.30 bps | -57.64 bps |
| Selected uncertainty penalty | -112.30 bps (weight 0) | -57.64 bps (weight 0) |
| Selected bounded hurdle x conversion | -112.30 bps (weight 0) | **-54.28 bps (weight .50)** |
| Latest-week control / interaction | -67.23 / -67.23 bps | -55.53 / **-52.98 bps** |

The July interaction removes its cutoff tie and adds about 3.4 bps, but it
remains negative.  May -> June still selects 0% long and all transition
weights freeze at zero.  No mechanism passes positive/latest/side/control
gates.

Decision:

1. Exact transition coverage is no longer a blocker.
2. Compression/onset and memory/state remain monitoring/uncertainty features,
   not admission scores, routers or vetoes.
3. Retain the `.50` July interaction only as a diagnostic mechanism effect;
   do not promote or choose it globally.
4. Stop this interaction branch until the raw direct score becomes positive
   and a mechanism repeats on another untouched forward block.
5. No policy/portfolio replay.

## 2026-07-30 closure of frozen-band, overlap and auxiliary-support tranche

Three linked diagnostics are now sealed and narrow the next repair.

1. `frozen_month_score_band_transition_diagnostic_20260730_v1` is explicitly
   invalidated under seal
   `184b8bd65ef36ba869a3f16ab4e67e903f27130ab47dffc531029511f7fab6c2`.
   The corrected v2 seal is
   `8ae47fa93db1b156fee81e63b59de3e190912f1e02e0a6bda3de4334f826b14c`.
   February -> March contains both score-scale compression and worse
   within-band response; March -> April keeps nearly the same numeric tail
   mass while the response improves.  This closes the first frozen-band
   diagnostic, not the IC-to-EV workstream.
2. `febmar_overlap_crowding_sensitivity_20260730_v2`, seal
   `38485e1d9d7edf5ec0899c70c8d39449065b74590d6394d2d1b41dd62413916e`,
   proves that side + asset + frozen-score-ventile matching is supported
   (99.12% March coverage, ESS 14,853, max SMD 0.014), but its
   group-size-omitted conditional net shift is still -36.36 bps with a
   [-89.55,+16.79] day-block interval.  Restoring raw/log candidate-group
   size collapses March support to 34.91%, so that primary estimand fails
   closed.  A direct cardinality audit changes the interpretation: February
   has only 236/238 rows per hour (118/119 assets x two sides), while March
   has 238/240 (119/120 assets x two sides) after KAITO enters the universe.
   Thus this field is predominantly eligible-universe size, not demonstrated
   market crowding.  The categorical support failure is mechanically real,
   but must not be called a regime transition or used to demand an economic
   crowding model.
3. The corrected robust auxiliary contribution chain is sealed by
   `afebfb8ce2108d849dc2f0ff18bdcae6fa3317afe685c5aeb05c762498560e81`
   with expanded gates under
   `f9924d058bb459ca44a8e518dfb7e3c584ebc8ed00b52de1df28f960f4704f17`.
   March OOF freezes `future_slope @ .25`; raw April top-10 improves from
   -33.49 to -30.44 bps, but mapped expected top-10 is -31.66 bps, latest
   mapped top-10 is -89.05 bps and short is -48.32 bps.  The deterministic
   mapped top-1 of +0.98 bps is invalidated by its cutoff plateau; random-tie
   expectation is -14.59 bps.  Peak contribution is not incremental at its
   frozen .25 weight.  No replay is authorized.

Fifteen focused tests pass across these artifacts.  Two inherited
scikit-learn/SciPy L-BFGS deprecation warnings remain in legacy propensity
test paths; they do not alter the sealed outputs.

Next executable repair:

1. Replace the ambiguous group-size field with a causal decomposition:
   eligible-universe cardinality, number/fraction of candidates actually
   clearing the base cutoff, side imbalance and score density near the global
   cutoff.  Universe cardinality is a nuisance/provenance field; only the
   latter three may represent market crowding.  Re-run overlap before asking
   older data to supply a mechanically different asset count.
2. Retain future slope as a bounded support input, but do not promote it as
   an admission score.  Test side-local bounded contribution weights selected
   on March chronological OOF and evaluated once on April, while selection
   remains one pooled-global top-k.
3. In the short branch, add decomposed favorable contribution and adverse
   loss severity separately.  Test a high-score x high-signal-density
   interaction only after the true density variables above exist causally;
   do not use raw universe cardinality as a regime proxy.  This directly
   targets the observed short and latest failures; timing, MAE, target-price
   and wait actions remain excluded.
4. Require positive raw and mapped expected top-10, positive latest block,
   stable two-side contribution, tie-safe selection and calibration
   non-inferiority before policy or portfolio replay.

The eligible-universe interpretation is now sealed in
`febmar_eligible_universe_interpretation_20260730_v1`, manifest
`d408610f44122267ca37cbf9bf4ec24353b3c4067ffd8dd0346b7bb3f470265d`.
It confirms there is no causal normalized true candidate-density field in the
canonical panel.  The next materializer must persist raw/pre-filter candidate
counts and an explicit decision-time denominator; no outcome or density
ablation is authorized until that field exists.

The bounded side-local support composition is also complete.  Use the final
wrapper `bounded_side_local_support_composition_20260730_v3_final_seal`,
manifest
`bdb32f1f5443a05f4514d5e49d822d9ec9c36136d6e57ee8c56b84e3cf992395`
and seal
`4ba4885d827adc3ff905c38b9c2f3ea24bef3051c5ccd23373c0737bd2f9dee1`.
Its v2 correction invalidates only v1 random-tie **precision** fields; all
net-EV values are unchanged.  The strict adverse-severity input has 12
role/side/month fold proofs with training-label resolution before validation.
It is ranking-risk support, not a timing or MAE action.

March global OOF selects the same weights on both sides:
peak `.15`, slope `.15`, adverse `0`.  Therefore the search finds no
side-specific composition and no incremental adverse-severity contribution.
April raw top-10 is -30.18 bps versus -30.44 for the earlier frozen slope arm
and -33.49 for the robust control.  Mapped random-tie expected top-10 is
-30.03 bps; latest raw/mapped top-10 is -89.27/-85.72 bps.  Long remains
slightly positive (+3.02/+1.32 bps raw/mapped) while short remains
-47.37/-47.96 bps.  The marginal aggregate improvement does not clear any
economic, latest, side, tie or calibration gate.  Stop additive support-weight
sweeps here; the next model change must repair short conditional payoff/loss
conversion rather than retune these three components.

The labels-free score-density sensitivity is complete at
`febmar_true_signal_density_overlap_20260730_v1`, manifest
`9bbdb2eb79723ace9f406ea03d6401cfa1a028041a65b58db431b49bfa8eda1c`.
It freezes February's numeric global top-10 base-score cutoff and q95-q90
near-cutoff width, then computes hourly above-threshold count/fraction,
long-short imbalance and near-cutoff fraction from decision-time scores.
Eligible asset count is not a covariate.  Support passes at 98.34% February /
98.88% March, ESS 11,774 and max post-weight SMD 0.030.  Conditional March
minus February net remains -36.21 bps [-86.24,+11.23], favorable gross
-47.60 bps [-80.81,-12.43] and opportunity incidence -7.00 points
[-14.83,+1.94].  Thus neither mechanical universe size nor observable frozen
score density explains the loss; conditional favorable-payoff conversion is
the strongest measured failure.

January older-data readiness is retained only for the wider IC/EV extension,
not to match an asset-count bin.  The terminology correction manifest is
`3eca1030079b005cbb0db6be5066149ed9f9c268a26267dad3ead06a953fa317`.
January still lacks the canonical base-score stream and current-spread exact
12-hour economics, and `historical_base_soft_oof` remains prohibited as an
unproven bridge.  Materialize those only if the next conversion model needs
genuinely older same-lineage economic support.

## 2026-07-30 bounded short conditional-conversion result

Two readiness audits precede the fit:

- `short_conditional_payoff_readiness_20260730_v1`, manifest
  `d438082b83033d2fb28c29e7953784917135c6d5092fa733f26d61a96eefb55c`,
  proves complete exact-ID score/peak/slope/MAE joins and provides the strict
  feature whitelist.  Timing, target-price, wait, realised path labels and
  mapped outcomes are forbidden.  Compact context is excluded from this
  baseline.
- `short_conversion_ablation_readiness_20260730_v1`, manifest
  `9dccdd4d4766e3f5e478f616b0a18ba95d7c1819daf5a8692ea3f4dc59152999`,
  shows ample March short support.  March alone cannot both select and
  confirm a challenger; the executed design therefore uses inner
  chronological March OOF for selection and untouched April exactly once.

The authoritative result is
`bounded_short_conditional_payoff_ablation_20260730_v3_final_seal`, manifest
`6d63562692fa09c805fe0bcdf96a5c3c124d19af94da7cf11661ead11ac370d2`,
seal
`4534480977a88da90a05f1c723a38aa7c5dd663ff401e72c207aa54e9923eb0c`.
Long remains the exact robust control.  Short predicts `P(net>0)`,
conditional favorable payoff and conditional adverse loss.  Among the
predeclared score-only, peak+slope, adverse and all-support arms with tail
weights `{1,2}`, March OOF freezes peak+slope with weight 2:
-53.20 bps versus -54.00 for the control.

April confirmation:

| Global depth | Raw net | Frozen mapped expected net |
|---|---:|---:|
| 1% | +31.32 bps | +31.85 bps |
| 5% | -10.32 | -13.65 |
| 10% | -24.55 | -25.19 |
| 20% | -54.42 | -55.47 |

The raw top-1 result is not recurrent evidence.  Its fixed-book 95% UTC-day
interval is [-52.27,+107.71] bps over 29 days and latest-week top-1 is
-86.24 bps.  It is 72.0% long at +62.33 bps; its short contribution is
-48.43 bps.  At top 10%, long is -4.27 and short -46.40 bps; latest raw /
mapped is -85.47/-86.08 bps.  Short predicted probability/favorable/loss are
`.519 / +201 / -230 bps`, implying -3 bps, versus -46.4 bps realised:
conditional-response calibration remains the dominant failure.

All economic, latest, side, mapped-tie and calibration gates fail.  Peak and
slope improve the March control by only 0.80 bps; predicted MAE severity is
not selected.  No replay runs.

This mapping step is completed by the sealed v5 result below.  Weak pooled
support is treated as unmapped warm-up with `NaN`, never as a tradable raw or
zero-EV fallback.  Weak side support retains the pooled anchor with an exact
zero side residual.

## 2026-07-30 sealed causal recent-EV mapping result

The authoritative artifact is
`short_winner_causal_recent_ev_mapping_20260730_v5`, manifest SHA
`44a1d4602e5ef1943f402f36f72e3b3fa8eea437f8150e530471aeb27a77606a`.
Versions v2, v3 and v4 are explicitly invalidated and redirect to v5.

The materialized score contract now closes the earlier lineage blocker:

- 33,408 unique March candidate-head OOF scores, exactly matching the frozen
  winner's ID set, raw scores and -53.196256 bps OOF global top-10;
- 69,258 April frozen-forward scores, bit-identical to the sealed winner
  ledger, with no April ranker refit;
- candidate keys, score availability, validation interval, training cutoff,
  maximum resolved training-label time and OOF/forward flags on every row;
- one precommitted mapper: at least 2,000 pooled and 1,000 side references,
  with side residual weight `n_side / (n_side + 500)`; and
- 31 daily UTC snapshots using only
  `snapshot - 21d <= label_end < snapshot`, with score availability before
  the snapshot and zero evaluation/reference identity overlap.

Every snapshot is legal and mapped.  Pooled support ranges from 33,408 to
48,952 rows as prior resolved April outcomes enter; short support ranges from
12,672 to 24,476.  Selection remains one pooled-global top-k after mapping,
with no timestamp, side, asset or archetype quota.

| April global book | Raw | Frozen March map expected | Causal pooled | Causal pooled + side residual |
|---|---:|---:|---:|---:|
| Top 1% net | +31.32 bps | +31.85 | +23.54 | +8.66 |
| Top 5% net | -10.32 | -13.65 | -50.85 | -21.00 |
| Top 10% net | -24.55 | -25.19 | -31.16 | -39.87 |
| Top 20% net | -54.42 | -55.47 | -66.50 | -64.79 |
| Latest decision-week top 10% | -80.48 | -74.94 | -88.13 | -100.50 |

At top 10%, the side-shrunk map selects 45.0% long / 55.0% short.  Long
contributes -6.34 bps and short -67.12 bps.  The pooled-only map is less bad
at -1.75 / -61.69 bps, but still fails.  The side-shrunk cutoff-tie fraction
is 2.48%, maximum asset share is 3.13%, bias is +12.77 bps and ECE is
31.13 bps.  Its equal-day fixed-book interval is
[-65.04,-18.33] bps at top 10%.  The positive top-1 point estimate is also
non-recurrent: latest decision-week is -68.80 bps and its interval is
[-41.59,+76.40] bps.

Identical-ID top-10 controls are -33.94 bps base, -24.32 residual and
-93.24 direct-q25.  The side-shrunk causal map therefore loses to the raw
winner and residual control.  It passes causality, coverage, top-10 tie,
side-share, asset-concentration and bias gates, but fails expected economics,
latest week, both-side positivity, ECE and control improvement.  No simple
policy or portfolio replay is authorized.

Required continuation:

1. Stop retuning the 21-day mapper and additive peak/slope/MAE weights.  They
   do not repair the conditional response level.
2. Extend the candidate-head OOF ledger with older same-lineage exact-12h
   rows and the compact causal transition contexts.  Development may pool
   older periods; any promotion claim requires a new forward block because
   April has now been inspected more than once.
3. Train a bounded conversion-residual layer on the score triplet plus
   opportunity, favorable magnitude, adverse severity, exit-mixture and
   retained regime-transition interactions.  Compare clean and
   competing-risk probabilities separately.  Optimize the pooled-global
   post-map top tail, not a per-timestamp or per-side book.
4. Report the full IC-to-EV waterfall alongside the conversion challenger so
   gains in average ordering cannot hide loss of tail payoff prevalence,
   favorable scale or cost-aware calibration.
5. Keep timing, MAE, target-price and wait actions in their separate action
   layer.  Continue to block portfolio replay until a frozen mapped top-10 is
   positive, latest-period positive, both-side positive, tie-safe,
   calibrated and better than the identical-ID controls.

## 2026-07-30 conversion-residual input materialized

`v5_conversion_residual_input_20260730_v1` is the sealed research input for
the next bounded learner.  Manifest SHA:
`23c54eb43447ca826d527a9e0b4d3ecfacfb285e6c098108a3570a284a856bd5`.
An independent rebuild and hash audit found no defect.

The panel contains 102,666 unique candidate-side identities:

| Period/status | Long | Short | Total |
|---|---:|---:|---:|
| March candidate-head OOF development | 20,736 | 12,672 | 33,408 |
| April frozen-forward rediagnostic | 34,629 | 34,629 | 69,258 |
| Total | 55,365 | 47,301 | 102,666 |

All joins use `candidate_id + side_name` and assert UTC `__ts__` equality.
Raw symbol is deliberately not a join key because normalized v5 symbols and
the exact label ledger have different spellings while candidate identity is
stable.  The materializer proves one-to-one parity for:

- frozen v5 raw and causal-map scores;
- canonical base and strict residual OOF scores;
- exact deployed-exit decision+12h gross, one explicit current-spread cost
  and net labels;
- strict OOF meaningful-MFE probability/conditional peak, fixed-geometry
  future slope and predicted MAE mixture; and
- compact causal market levels, 3h/12h transition deltas and eight regime
  composites.

Score deciles are recomputed on the full canonical hourly side population
before the 102,666 rows are selected, then joined to the immutable
`timestamp x side x decile` context.  All 33 baseline and four optional
adverse-risk inputs are finite and complete.  MAE predictions are optional
risk-support ablations only.  Realised paths, outcomes, exit fields, timing,
bars-before-trough, target-price/wait actions, mapped coordinates, map
references and universe cardinality are never model features.  The aggregate
decile-context transform is still research-only until strict live inference
implements the identical operation.

The older-data request is fail-closed rather than silently relaxed:

- February has compatible base scores and exact labels, but its residual is
  a non-OOF passthrough warm-up and it lacks the v5 candidate-head and strict
  auxiliary score stream.
- January has neither the frozen canonical score lineage nor a compatible
  current-spread deployed-policy exact-12h join.
- `historical_base_soft_oof`, old55 and hourly/no-spread sources remain
  prohibited bridges.

Therefore the current panel supports a March-development / April
rediagnostic ablation, not promotion.  In parallel, genuine history extension
requires materializing the canonical January base stream, compatible exact
policy labels, a February strict residual OOF and February auxiliary OOF
predictions.  Do not substitute a nominally larger incompatible dataset.

Next execution:

1. Freeze a small feature-group grid: score triplet; +peak/slope; +core
   market levels; +3h/12h transitions; +eight regime composites; all compact;
   and all compact + optional predicted-MAE risk.
2. Compare direct conversion residual, positive/favorable/adverse hurdle and
   clean-versus-competing-risk formulations.  Select on chronological March
   OOF only and report April once as a reused diagnostic.
3. Apply the same precommitted pooled-global recent-EV map to every frozen
   challenger and report global top 1/5/10/20, latest week, both sides,
   calibration, ties and concentration.
4. No challenger from this reused April window may be promoted.  A genuine
   promotion test requires the compatible history extension and a new
   forward block.

## 2026-07-30 actionable regime + transition extension

Status: infrastructure and historical labels are materialized; candidate-level
OOF layers and matched economic ablations are still in progress.

Completed:

- [x] Strict exact-minute 2024 backfill: 141 / 141 products and
  43,660,200 / 43,660,200 required minutes.
- [x] Exact 12-hour policy/path/timing/multitask labels for all 190,398 2024
  candidates.
- [x] Reconstructed 2022-2024 base-plus-residual population: 360,012 rows.
- [x] Complete available performance denominator: 39 months, 168 weeks.
- [x] Causal 1/3/6/12/24/48/72/168-hour multiview market panel.
- [x] Adaptive transition lifecycle catalogue with purged event-group OOF
  stable-versus-transition probabilities.
- [x] Fold-local family-balanced feature-selection contract.
- [x] Separate regime and transition OOF namespaces/provenance contract.
- [x] Subsampled exact tree-SHAP interaction and conditional-permutation
  discovery infrastructure.

Mandatory next tasks:

- [x] Enrich the hourly panel with actual causal spread, volume, Amihud and
  cross-sectional stress fields from the historical feature store:
  `regime_liquidity_enrichment_2022_2026_20260730_v1` feeds
  `regime_multiview_panel_2022_2026_20260730_v2` by exact timestamp.
- [x] Compact the hourly panel inside each training fold; retain horizon and
  feature-family coverage without using held-out rows.
- [x] Materialize candidate-keyed regime probabilities and transition
  probabilities separately.  Each must carry its own train cutoff,
  availability, entropy, margin, OOD score and source hashes.  Complete for
  293,828 candidates from 2023-04 through 2024, including full 2024.
- [x] Preserve four matched arms on identical candidates:
  `baseline`, `regime_only`, `transition_only`,
  `regime_plus_transition`.
- [x] Discover `regime x feature` and `transition x feature` interactions
  independently using subsampled exact tree SHAP interactions.
- [x] Run regime-conditional and transition-phase-conditional permutation
  importance with explicit support and recurrence gates.
- [x] Test clean and competing-risk probabilities separately inside each
  context arm.
- [x] Join exact path geometry and report each state/phase by side:
  opportunity rate, peak MFE, MAE, time-to-MFE, future slope, timeout,
  exit conversion and valid net EV.  Completed in
  `regime_transition_path_geometry_diagnostic_20260730_v1` for 309,132
  candidates; transition phase is ex-post diagnostic only.
- [x] Run the bounded matched direct-context, interaction-conditioned
  residual-trust and additive GAM ablations with one pooled global top-k
  after causal recent-EV mapping.  The uncontextualized baselines remain
  best; no regime/transition context arm passes.
- [ ] Require aggregate and latest-month economics plus weekly/monthly
  Q10/Q50, positive-period fractions and worst-period stability.
- [ ] Replay concurrency, exposure and asset constraints only for a frozen
  winner that passes the preceding gates.
- [ ] Fill the remaining canonical scoring windows in 2025-2026; missing
  periods must remain explicit until genuinely backfilled.  The signed v2
  readiness ledger proves raw inputs exist for all 14 months; canonical base
  OOF, residual OOF and candidate-local exact-12h economics remain the three
  explicit missing stages.
- [x] Seal the January--February 2025 provenance check.  February has 64,512
  verified base-only top-40 rows with exact deployed-policy 12-hour
  economics, but remains `residual_is_oof=false` and is ineligible for stack
  evaluation.  January still has no accepted canonical base OOF or canonical
  path-input join.  Do not substitute native labels or comparator scores.
- [x] Seal the separately versioned May--June 2025 common-30 continuation.
  Base and residual each have 87,840 strict PIT OOF rows with exact 12-hour
  economics and frozen 31-long/8-short feature contracts.  Base is -77.17
  bps aggregate; residual is -83.32 bps, improving May slightly but degrading
  June materially.  Neither is promotable or a replacement for Jan--April.

Interpretation guard: current regime describes persistent state; transition
state describes change/lifecycle.  Neither may substitute for the other.
Their combination is a distinct ablation, not a merged label.  Timing, MAE,
target-price and wait actions remain outside the execution-EV score in a
separate action layer.

First four-arm direct-context result:

- [x] Full-2024 matched causal Ridge-map probe on 190,398 candidates.
- [x] One pooled global top 10%, with no timestamp/side/state-local ranking.
- [x] Baseline -121.61 bps; regime-only -124.84; transition-only -120.71;
  combined -122.35.
- [x] Zero gate winners; no portfolio replay.
- [ ] Continue with interaction-conditioned residual trust and the
  base/residual/GAM challenger rather than promoting direct state additions.

Residual-trust result:

- [x] Four-quarter chronological OOF interaction-conditioned trust learner.
- [x] Six identical-candidate arms, 190,398 2024 candidates.
- [x] Baseline remains best at -109.70 bps top-10 net EV.
- [x] Regime, transition, combined, adverse-risk and clean additions all
  degrade this low-capacity learner.
- [x] Complete the separate additive GAM regime calibration ablation.
  Calibration improves, but the uncalibrated baseline remains best:
  -104.99 bps versus -112.17 bps for the best contextual GAM.

Remaining objective audit:

- [x] Authoritative signed 17-requirement audit v6: 14 proved, 3 incomplete.
  The open requirements are adequate cross-era morphology support, an
  accepted all-era unsupervised-regime economic solution, and stable
  regime-category EV.  V6 incorporates the latest negative morphology,
  unsupervised, alpha-to-EV and category-stability evidence, with no stale
  completed task in its next-work list.
- [ ] Train and freeze the current-regime model on causal comparable-lineage
  data from 2022-08-30 through 2025-12-31; assess once on untouched 2026.
  Keep all feature selection, HPO, state semantics and calibration inside
  the training interval.  The v1 baseline is sealed but pathological:
  near-zero entropy and 40%--54% hourly switching.  Authoritative v2 must use
  enriched multiview v2, family-balanced features and train-only causal
  persistence filtering.
- [ ] Train and freeze the transition model on the same 2022--2025 cutoff;
  assess once on untouched 2026 with probability, uncertainty and lifecycle
  outputs.  Do not substitute current-regime state for transition state.
  The strict v1 baseline is sealed: AUC 0.572, AP 0.0209 at 1.43%
  prevalence, lifecycle macro-F1 0.096, and every economic risk decile is
  negative.  The train-only blocked-CV/HPO v2 improves Brier/ECE but worsens
  AP/AUC, lifecycle macro-F1 and net EV (-99.02 versus -97.56 bps), so it is
  rejected.
- [ ] Run train-only strict 1/3/6/12-hour transition-onset heads,
  cause-specific competing-risk lifecycle heads, calibrated/raw comparison,
  and stable-feature-family robustness.  The non-transfer audit proves the
  zero-event 2026 months are genuine and identifies OI/breadth covariance
  shifts, not label-coverage holes.
- [ ] Evaluate `regime_only`, `transition_only` and
  `regime_plus_transition` as three distinct 2026 arms on identical rows,
  with one pooled global post-map top 10% and exact monthly economics.
- [ ] Add January--August 2022 inverse-PI rows only if identical/harmonized
  feature definitions are proved.  Never pool its local taxonomy IDs or
  economics with the later PF lineage.
- [x] Integrate the accepted alternate Jan-Aug 2022 population into a
  separately labelled soft regime/transition supplement.  The v3 supplement
  has 57,840 exact candidates and 5,784 leave-month-out OOF hourly rows.
  Its transition classifier is weak (monthly AUC 0.489--0.543), and its local
  state IDs are explicitly non-equivalent to the later PF lineage.
- [x] Backfill 2022-08-01 through 2022-08-29.  Five inverse contracts now
  have 208,800 exact one-minute rows, and all 6,960 August candidates have
  complete 720-minute paths.  The chronological gap is closed; population
  non-equivalence remains mandatory.
- [x] Execute a Bayesian Rule List transition challenger on the same OOF
  target; do not count implementation-only code as evidence.
  Native Beta-Binomial MAP BRL is materially weaker than LightGBM
  (AUC 0.600 versus 0.874) and is rejected.
- [x] Extend interaction/covariance recurrence beyond the bounded
  2023Q4-to-2024Q1 probe.  Across three comparable eras, zero
  BH-controlled feature, covariance, interaction or regime-conditional
  effects recur in the same direction across at least two separated eras.
  Early 2022 remains explicitly missing comparable multiview evidence.
- [x] Materialize the causal feature/gap inventory.  The sealed v5 artifact
  inventories 16,618 actual fields, 72 selected multiview units, 23 forbidden
  outcome/state fields and 83 exact source-unavailable liquidity
  asset-field combinations.  It records seven economically plausible missing
  observable/composite families without promoting any of them.
- [x] Diagnose positive alpha IC versus negative execution EV on exact
  economics.  Across 619,694 candidates, the dominant failure is low
  clean-opportunity capture under an approximately 100 bps cost hurdle.
  Sparse recent mapping support is not supported; rank-changing map behavior
  remains material.
- [x] Ablate a causal cost-aware opportunity hurdle before EV mapping and a
  rank-preservation/support-shrinkage map.  The hurdle is non-incremental
  (-103.90 versus -103.88 bps baseline).  Rank/support improves aggregate net
  to -94.05 bps but fails latest July (-160.45 bps) and all stability gates.
  Both are rejected; no portfolio replay.
- [x] Run the matched common-surface GMM/DAE/failure-first economic ablation.
  The authoritative v2 excludes GMM posterior and compact risk summaries.
  Baseline is +11.32 bps aggregate but fails latest/stability gates; all six
  context arms are aggregate-negative.  Transition-only and combined
  failure improve partial July by about 25 bps but remain negative.
- [ ] Extend the common unsupervised economic surface beyond 20 May--10 July
  2026 before making an all-era or full-July conclusion.  Full 2024 is now
  sealed separately: DAE improves net by +29.71 bps versus baseline and
  raises IC, but remains -106.36 bps and helps 2024 while hurting 2026.
  This is evidence of representation non-transfer, not promotion.
- [ ] Persist matched fold-local DAE codes, reconstruction error,
  representation age/train support and OOD for 2024 and 2026, then test a
  prior-month-trained trust layer on locked later months.  The current
  cross-era diagnostic cannot identify trust because 2024 codes were not
  retained and the sealed surfaces share no comparable raw market fields.
- [x] Run held-out regime-category economic stability attribution with
  current regime, ex-post transition phase and their combination kept
  separate.  No stable-good category exists.  Five stable-poor categories
  recur only in the non-promotable frozen 2022--2024 counterfactual cohort;
  exact 2025--2026 has only two comparable eras.  No gate is authorized.
- [ ] Establish cross-fold morphology alignment before naming global
  recurrent transition types.  The sealed support bound has 157 unique
  events across five eras and 14 fold-local components.  A strict
  leave-one-era-out follow-up now assigns all 157 events exactly once with
  zero skips, fixing the missing held-out-row defect.  Train-only cross-fold
  prototype matching and held-era predictive outcome evidence are still
  required before naming global types or creating gates.  The exact binding
  currently supplies only 118 event-source slices (64/41/13 in
  2022--23/2024/2025), no matched 2026 slice, no retained prototype matching
  matrix, and no current-regime/transition-probability baseline.  The nested
  increment test is therefore sealed as statistically insufficient.

## 2026-07-30 mapped-policy-aligned conversion-residual result

The authoritative diagnostic is
`v5_conversion_residual_ablation_20260730_v3`, manifest SHA
`40c78dea764e3dc4fb9be9620b4302b8e1b3498acfab485f04138fe1bc04f388`.
V1 is invalid because it selected its March configuration on raw challenger
scores.  V2 corrected selection but is superseded because its promotion
table omitted explicit March coverage, aggregate/latest/worst-fold economics
and development tie gates.  V3 preserves the v2 predictions byte-for-byte
and supplies the complete gate surface.

Every configuration now follows the traded decision contract:

- side-local models produce chronological March OOF predictions;
- each configuration receives its own daily 21-day pooled-isotonic anchor
  plus shrunk side residual, using only prior-resolved OOF predictions;
- weak pooled support is `NaN` and ineligible, never raw/zero fallback;
- feature and target selection use random-tie-expected mapped net EV under
  one pooled-global top 10%, never per timestamp or per side; and
- base, residual, direct-q25 and v5 controls each receive their own
  score-specific causal mapping before comparison.

March has 18,432 selection rows, of which 13,824 (75%) are causally mappable.
The first fold has 2,304 mapped rows; the remaining 4,608 early rows are
fail-closed warm-up caused by insufficient prior short-side lineage.  The
mapped stability leader changes from the raw-selection score-only arm to the
23-field score + peak/slope + market-level + regime arm, but every result is
economically negative:

| March mapped diagnostic | Aggregate top 10% | Latest fold | Worst fold | Stability objective |
|---|---:|---:|---:|---:|
| Selected direct residual | -133.21 bps | -127.93 | -149.08 | -165.73 |
| Positive hurdle objective | -197.26 | -101.79 | -198.78 | -194.75 |
| Competing-risk objective | -183.93 | -164.04 | -243.00 | -233.30 |

The selected arm's isotonic cutoff plateaus are also inadmissible: 26.03% of
the aggregate book is tied at the cutoff, while fold-level tie mass reaches
67.53%, 50.00% and 249.89% of book size.  Random-tie expectation prevents row
ordering from fabricating performance, but it does not make this tradeable.

April remains a reused diagnostic, never promotion evidence:

| April mapped global top 10% | Expected net | Latest week |
|---|---:|---:|
| Selected direct residual | -92.17 bps | -144.21 |
| Best challenger seen in April, positive hurdle | -79.77 | -132.22 |
| Residual control | -30.39 | -89.57 |
| Frozen v5 control | -39.87 | -100.50 |
| Base control | -46.01 | -52.96 |
| Direct-q25 control | -80.90 | -135.72 |

The selected April book is 27.92% long / 72.08% short.  Long contributes
-100.25 bps and short -89.09 bps.  Bias is 31.82 bps and ECE is 85.16 bps.
It therefore fails development coverage, aggregate/latest/worst development
economics, development tie safety, April economics/latest week, both-side
positivity, bias, calibration and mapped-control improvement.  No simple
policy or portfolio replay is authorized.

Actionable interpretation:

1. Causal mapping alignment changes which feature arm looks least bad, but
   does not rescue the conversion learner.
2. Regime/level context has weak diagnostic value in March; its April
   deterioration and extreme mapped ties prohibit treating it as a stable
   gain.
3. Positive hurdle and competing-risk targets do not beat direct residual
   during mapped OOF selection.  Their less-negative April point estimates
   are post-selection diagnostics only.
4. Peak/slope, transition, compact-context and optional-MAE additions do not
   produce positive mapped economics.  Do not enlarge HPO on this panel.

Required continuation, in order:

1. Materialize earlier same-lineage short OOF scores and genuine
   January/February base, residual, auxiliary and exact-policy rows so
   development mapping has full causal coverage without lowering support
   thresholds.
2. Complete the fixed-cohort base-IC-to-execution-EV waterfall and
   rank-preserving month counterfactuals.  The current learner result does
   not explain why improving broad IC fails in the selected tail.
3. Re-run the bounded learner only after history extension.  Freeze the
   current grid and add one predeclared tie-safe monotone mapping challenger;
   never tune it on reused April.
4. Confirm any survivor on a genuinely untouched forward block with positive
   aggregate, latest-period, worst-period and both-side economics before
   portfolio replay.

## 2026-07-30 extended-history conversion v4 result

The early-March short OOF gap has been repaired without changing the frozen
candidate-head architecture or lowering mapping-support thresholds:

- `v5_early_short_oof_extension_20260730_v1` adds 8,064 strict short OOF
  scores over March 13--19.  Its 33,408 overlapping v5 rows retain
  bit-identical raw scores.
- `v5_conversion_residual_input_20260730_v2` joins 41,472 balanced March OOF
  rows and the unchanged 69,258 April diagnostic rows to the exact base,
  residual, auxiliary, transition-context and deployed-policy label sources.
- `v5_conversion_residual_ablation_20260730_v4`, manifest SHA
  `9e2195f84322eb704d0cb9c244082a92c043db4c3ee4981b5145363fd076baf7`,
  supersedes v3 as the authoritative bounded conversion diagnostic.
  A fresh isolated rerun reproduced all 13 published output files
  byte-for-byte.

Every one of the ten predeclared feature/target configurations now emits its
own 6,912 March 20--22 calibration-OOF predictions using only labels resolved
before March 20.  Those rows are excluded from configuration scoring and
supply only prior-resolved score-specific map history.  All 18,432 March
23--30 selection rows are consequently mapped under the unchanged causal
21-day pooled-plus-shrunk-side contract.  Selection coverage is 100%, versus
75% in v3.

The added history changes the diagnostic conclusion but does not produce an
admissible model:

| Frozen March selection arm | Mapped top 10% | Worst fold | Stability objective |
|---|---:|---:|---:|
| Four scores, direct residual | -129.11 bps | -177.71 | -188.97 |
| Four scores + peak/slope | -130.70 | -185.83 | -220.19 |
| Scores + peak/slope + levels | -159.81 | -229.68 | -253.41 |
| Scores + peak/slope + levels + transitions | -180.28 | -244.06 | -267.47 |
| Scores + peak/slope + levels + regimes | -158.10 | -221.16 | -259.23 |
| All compact context | -173.45 | -223.97 | -261.59 |
| All compact + optional MAE | -188.70 | -228.70 | -232.75 |

On the frozen four-score feature winner, the positive hurdle narrowly leads
the stability objective at -185.47 bps, versus -185.55 for competing risk
and -188.97 for direct residual.  Its aggregate March top-10 remains
-132.52 bps, its latest fold -97.69 bps and its worst fold -180.10 bps.
The final fold has an extreme isotonic plateau whose cutoff-tie mass is
167.68% of book size, so the apparent narrow objective lead is not robust.

April is reused diagnostic evidence only.  The mapped hurdle arm is
-56.40 bps overall and -95.75 bps in the latest week.  Its book is 23.27%
long at -10.08 bps and 76.73% short at -70.43 bps.  It loses to the
score-specific mapped residual control at -31.43 bps, and its ECE is
37.62 bps.  Aggregate/latest/worst March economics, fold tie safety, April
economics, side balance, both-side positivity, mapped-control improvement
and calibration therefore fail.  No simple-policy or portfolio replay is
authorized.

Actionable decision:

1. Mark the March mapping-coverage repair complete; do not lower the
   2,000 pooled / 1,000 side support guards.
2. Treat direct addition of peak/slope, market levels, transitions, regimes
   and optional MAE to this low-capacity conversion learner as negative
   ablations.  Their standalone learnability does not make them incremental
   in this architecture.
3. Do not expand HPO around the hurdle's marginal stability-objective lead.
   It is economically negative, short-heavy and tie-unstable.
4. Continue the base-IC-to-execution-EV attribution, matched 12-hour target
   and regime-conditioned trust/reliability workstreams.  Test supporting
   heads through detached reliability, interaction or action-layer routes,
   not by repeating the rejected direct concatenation.
5. Reserve portfolio replay for a future frozen configuration that passes
   mapped aggregate, latest, worst-fold, both-side, calibration and tie gates
   on a new forward block.

## 2026-07-30 full-base raw-OOF repair and stop-target correction

A supporting-label audit found that
`v5_conversion_residual_input_20260730_v2.target_stop_exit` was identically
zero: the materializer searched the canonical exit reason for `"stop"`, while
the exact deployed-policy full-stop value is `"full_sl"`.  The affected v2
field is partially invalidated.  The corrected sealed successor is
`v5_conversion_residual_input_20260730_v3`, manifest SHA
`cac676ae44816fd1fead2c9c69d48893cfa0ca2ae881f5896719d65ad56f0a05`.
It copies `target_stop_exit` from `exit_is_full_stop` and asserts parity with
`execution_exit_reason == "full_sl"`: 22,406 stop targets and 30,471 timeout
targets now reconcile exactly.  V4's predictions and conclusions are
unaffected because it never used `target_stop_exit`; its competing classes
come directly from exact exit reason and net outcome.

The invalid full-base opportunity experiment has also been repaired without
repeating its approximately 450 already-valid CatBoost OOF fits.  The
authoritative successor is
`canonical_full_base_opportunity_ablation_20260730_v2`, manifest SHA
`4e7c295467b14635abe527b63b18b5924fab887ac7aa43f68db3a3b19d6f8a26`.
It:

- hash-verifies the invalidated v1 source and reuses only the unaffected
  334,298 raw side-local OOF predictions;
- selects the two fixed arms per target and their geometry using raw,
  random-tie-expected pooled-global top-10 economics only;
- fits the four missing April configurations, eight side-local models total;
- freezes every target/feature/geometry choice before mapping;
- applies separate score-specific causal 21-day pooled maps, with the
  shrunk-side residual reported only as a secondary diagnostic; and
- labels all 172,450 April rows reused diagnostic evidence, with promotion
  and portfolio replay unconditionally disabled.

Raw OOF selection reproduces:

| Target | Selected arm/geometry | Raw development top 10% |
|---|---|---:|
| Direct net | S0 / compact d4 | -52.41 bps |
| Direct net | S1+B / compact d4 | -65.78 |
| Gross > cost | S0 / compact d4 | -64.39 |
| Gross > cost | S1+B / fixed d5 | -69.28 |
| Gross > cost +25 bps | S1+B / fixed d5 | -72.22 |
| Gross > cost +25 bps | S0 / deep d6 | -73.48 |
| Existing soft economic label | S0 / compact d4 | -57.29 |
| Existing soft economic label | S1+B / compact d4 | -64.65 |

Every repaired April challenger is negative and worse than the identically
mapped base control:

| Causal pooled map, April global top 10% | Net | Latest week |
|---|---:|---:|
| Frozen base control | **-68.93 bps** | -87.30 |
| Hard-25 S0 / deep d6 | -85.72 | -93.37 |
| Soft S0 / compact d4 | -88.18 | -144.03 |
| Direct S0 / compact d4 | -88.99 | -106.64 |
| Hard-0 S0 / compact d4 | -95.11 | -115.49 |
| Best S1+B arm | -108.77 | -121.24 |

All eight configurations have 100% map coverage and 31/31 causally legal,
supported daily snapshots.  The score-only heads remain side-unstable:
direct S0 is 86.39% short, while the soft S0 arm is 85.57% short.  S1+B
raises opportunity precision but worsens net payoff, confirming that the
frozen 31/8 base inputs can identify movement without reliably identifying
capture, adverse severity or executable payoff.  Shrunk-side mapping does
not repair the result.

Decision:

1. Mark the invalid mapped-development selection repaired; preserve v1 only
   as a hash-bound raw-prediction source.
2. Reject the repaired hard-0, hard-25, soft and direct full-base heads as
   standalone admission rankings.  Do not expand their HPO.
3. Use the repaired OOF/forward predictions only as detached support
   sidecars in the next meaningful-MFE/clean/competing-risk/capture
   reliability experiment.
4. Materialize a proper 2025 pre-exit capture target and a fixed severe-loss
   target.  Do not use full-12-hour MFE minus gross as exit capture because it
   can include favorable movement after the deployed exit.
5. Keep hit timing, MAE path timing, target price and wait/reprice actions in
   the separate action layer.

## 2026-07-30 sealed execution-reliability input and IC-to-EV requirement

The next reliability workstream now has a sealed canonical input:
`canonical_execution_reliability_input_20260730_v2`.  It contains 110,730
exact identities (41,472 March and 69,258 April; 55,365 per side), exact
decision-to-12-hour ATR triple-barrier labels, exact deployed-policy
gross/cost/net outcomes, the corrected stop/timeout labels, the four v4
scores, eight repaired full-base support sidecars, candidate-context fields,
five regime levels, ten 3h/12h transitions and eight regime composites.
There are no missing or non-finite default inputs.  Gross minus one explicit
row cost equals net exactly.

The frozen experiment contract is
`configs/canonical_execution_reliability_workstream_20260730_v1.json`.
It predeclares the v4 control; detached repaired supports;
meaningful-MFE, clean favorable-first and three-way competing-risk routes;
and one bounded transition-interaction extension.  Timing, MAE, target-price
and wait/reprice outputs remain in the separate action layer.  April is
reused diagnostic evidence, so no result from this lineage is promotion
evidence.

The apparently improving base-target rank IC
(`0.155 / 0.162 / 0.226` in February/March/April) alongside negative direct
execution-EV top-decile outcomes (`-59.39 / -91.31 / -38.45 bps`) is now a
mandatory gate in that contract.  Every reliability run must emit an
identical-cohort waterfall from native target through exact 12-hour MFE,
deployed-exit gross, explicit cost and net; global-tail IC/calibration and
opportunity recall at top 1/5/10/20%; fixed-threshold versus fixed-quantile
migration; stop/timeout and payoff-magnitude attribution; and matched 12h
versus 24h labels.  Aggregate IC improvement alone is not model progress.
The item closes only when the month-to-month IC/EV deltas are quantitatively
reconciled and a frozen causally mapped challenger passes economics,
latest-period, both-side, tie and control-improvement gates.

## 2026-07-30 strict regime/transition split

Hard rule: train and freeze regime and transition models on 2022–2025;
assess once on 2026.  No 2026 feature selection, HPO, calibration, state
semantics, persistence, thresholding or contextual-score fitting is allowed.

Completed:

1. `strict_transition_v3_multihorizon_competing_risk_20260730_v2`:
   retain the modestly transferable 1h/3h onset probabilities as diagnostic
   context; reject 6h/12h, lifecycle and cause-specific action use.
2. `strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3`:
   valid strict holdout, but reject the diagonal-GMM taxonomy because its
   median dwell is two hours and monthly switching remains 28%–35%.
3. Preserve regime-only, transition-only and combined arms on identical 2026
   rows.  Evaluate one pooled global monthly top 10% only after each arm's
   own causal EV map; never rank per timestamp or per state.

Next:

1. Materialise a common market-only feature panel for January–August 2022
   and the later PF lineage, using the accepted different candidate
   populations but identical feature definitions.  Then rerun the frozen
   model on literal January 2022–December 2025.
2. Replace the rejected diagonal GMM with persistent-state challengers:
   an HMM/sticky-HMM and a duration-aware semi-Markov model.  Choose state
   count and persistence only inside blocked 2022–2025 validation.
3. Freeze candidate-context coefficients on the exact available pre-2026 OOF
   panel, then score untouched 2026 baseline, regime-only, transition-only
   and combined arms.  Missing July–December 2025 compatible candidate OOF
   rows must be disclosed, not imputed or treated as evaluation.
4. Do not authorize policy/portfolio replay unless a frozen arm improves
   aggregate, latest-month and worst-period net EV, retains both-side
   contribution, and passes calibration/tie/concentration gates.

## 2026-07-30 execution-reliability ablation v2: completed, not promotable

The reliability input and experiment references above are superseded by the
sealed v4/v2 lineage:

- `canonical_execution_reliability_input_20260730_v4` keeps the 110,730-row
  v3 panel and capture support byte-identical, but formally authorizes
  `frozen_base_score_decile` for the requested context ablation.  Its
  group-size alias is documented and excluded as a duplicate.
- `configs/canonical_execution_reliability_workstream_20260730_v2.json`
  binds the exact outer folds, feature arms, head targets, HPO, PVC feature
  selection, causal mapper, pooled-global book and promotion gates.
- `canonical_execution_reliability_ablation_20260730_v2` contains 21
  evaluated score configurations, 1,986,642 score rows, 224 fold/side/head
  metric rows, full global-book attribution and frozen pre-April recipes.
- `canonical_execution_reliability_ablation_summary_20260730_v1` is the
  authoritative compact arm, head and gate ledger.

The independent audit passes all integrity and research contracts: every
listed hash and manifest seal verifies; March has 25,344 exact outer-OOF
rows per configuration; April has 69,258 frozen-forward rows; all three
selection folds have complete causal-map coverage; score availability is at
decision time; exact `gross - cost = net` has zero error; and side, asset,
regime and realised-exit attribution reconciles to one pooled-global book
without group reranking.

The frozen architecture path and mapped March selection economics are:

| Stage | Frozen choice | Mean | Latest fold | Worst fold | Stability objective |
|---|---|---:|---:|---:|---:|
| Mapped control | residual expected EV | -60.13 | -55.07 | -68.52 | **-80.25** |
| A1 support | S0 + S1B | -106.56 | -50.45 | -168.65 | -172.94 |
| A1 context | timestamp-side rank + z | -97.45 | -36.62 | -162.78 | -163.95 |
| A2 | meaningful-MFE -> capture -> gain/loss | -82.74 | -42.45 | -146.63 | -142.24 |
| A3 | clean favorable-first mixture | -104.45 | -61.08 | -133.00 | -153.29 |
| A4 | favorable/adverse/timeout mixture | -106.89 | -38.15 | -167.17 | -175.19 |
| A5 | A2 + five bounded transition interactions | **-64.12** | **-50.69** | **-81.54** | **-90.96** |

A5 is the research winner among learned architectures because the transition
interactions materially repair A2 fold stability.  It is not the production
winner and must not be described as such: the identically mapped residual
control retains the better objective by 10.70 bps.

Global pooled-book economics fail:

| Frozen global top 10% | March mapped | April mapped diagnostic | April latest seven days |
|---|---:|---:|---:|
| Residual control | -71.44 bps | -30.21 bps | -81.01 bps |
| Final A5 challenger | **-93.54 bps** | **-59.12 bps** | **-131.09 bps** |

A5 March folds are `-60.12 / -81.54 / -50.69 bps`.  Its March long and
short contributions are both negative (`-42.51 / -51.03 bps`), as are
April (`-19.18 / -39.94 bps`).  The aggregate March/April cutoff-tie shares
are safe at 2.07%/0.32%, but selection fold 1 is 23.27%, so fold-level tie
safety fails.  No untouched forward evidence exists.  Portfolio replay
therefore remains forbidden.

What the head ablations establish:

1. The meaningful-MFE event remains the bottleneck.  A2/A5 OOF ROC-AUC is
   only about 0.57, AP about 0.35 and ECE about 0.12.
2. Conditional positive capture is more learnable (AUC about 0.65--0.67,
   AP about 0.94), but its prevalence is about 0.90, so it offers limited
   selectivity after conditioning on valid meaningful opportunity.
3. Conditional gain magnitude is learnable (rank IC about 0.22), as are
   favorable payoff (about 0.24) and timeout payoff (about 0.36).
4. Adverse/loss magnitude is not reliable.  The A2/A5 loss head has only
   159--513 training examples by fold/side and rank IC near zero.
5. Timestamp-side rank/z is the only candidate context with useful
   incremental March stability.  Cutoff margin, timestamp-global context
   and combined side/global context hurt.  Global group size is exactly
   nonincremental; rank decile modestly improves the worst fold but loses
   to timestamp-side context.  Raw OOF alpha is already the byte-identical
   base-alpha anchor.  Archetype-relative z remains unavailable and was not
   replaced with DAE/GMM geometry.
6. The five transition interactions improve A2 stability and capture
   discrimination, but do not repair meaningful-MFE or loss prediction.

The failure is economic, not merely classificatory.  A5 gross capture is
only +6.49 bps in March and +41.08 bps in April against approximately
100 bps cost.  Every frozen execution-risk quintile is negative.  In April,
successful trailing exits contribute +95.31 bps, but deployed full stops
contribute -106.04 bps and timeouts -48.39 bps.  The current meta layer
raises some useful conditional metrics without selecting enough gross
movement or avoiding enough stop/timeout loss to clear cost.

### Required next ablations

1. Replace the weak generic meaningful-MFE gate with explicit pre-entry,
   cost-aware event heads: `pre_exit_mfe > row_cost + buffer`, successful
   trailing/capture, deployed full-stop, and deployed timeout.  Keep exact
   exit-policy targets and one cost.
2. Replace the sparse continuous adverse-loss regressor with a hurdle:
   full-stop/severe-loss probability followed by conditional severity only
   where support is sufficient.  Keep gain magnitude regression, which is
   demonstrably learnable.
3. Split A4's broad adverse/conflict class into realised execution outcomes
   aligned with the attribution failure: successful trailing, full stop,
   timeout and other adverse exit.  These are targets only, never inputs.
4. Retain timestamp-side rank/z and the five bounded transition
   interactions as the bounded candidate architecture.  Drop cutoff,
   timestamp-global and group-size context unless new evidence makes them
   incremental.
5. Ablate a causal rank-preserving EV calibrator against isotonic mapping.
   The current isotonic plateaus create 23.27% selected boundary share in
   one fold.  Any alternative must remain score-specific, 21-day,
   prior-resolved and pooled-global.
6. Extend genuine same-lineage base, residual, support and label history
   before March and reserve new untouched forward evidence.  February can
   currently support only the base IC-to-EV waterfall, not a strict
   residual/direct-head comparison.
7. Do not train regime-specialist models from this result: all five
   execution-risk quintiles are negative.  First prove that a pre-entry
   stop/timeout classifier and cost-aware opportunity head transfer across
   periods.
8. Re-run the identical-cohort IC-to-EV waterfall with each new challenger.
   Rising aggregate IC remains insufficient while global selected-tail net
   EV is negative.

### 2026-07-30 causal mapping ablation v2: completed, negative

The mapping-only diagnosis is complete and independently verified at
`data_perp/artifacts/canonical_execution_reliability_mapping_ablation_20260730_v2`.
The earlier v1 artifact is explicitly invalidated because it omitted the
same-cohort residual control and did not satisfy the frozen mapping,
latest-window or attribution contracts.

V2 compares the final A5 challenger and the residual control under four
identical mappings: the frozen baseline, strict pooled rank-preserving
mapping (M1), positive-slope robust mapping (M2), and pooled timestamp-
percentile fixed-bin mapping with shrinkage/PAVA (M3).  All 80
configuration/day audit rows are causal, with zero reference/evaluation
identity overlap.  Every alternative mapper has zero within-snapshot
inversions and zero plateaus.  Fractional global-book attribution reconciles
within `1.78e-14` bps.

| March selection objective | Baseline | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| Residual control | -80.25 | -104.67 | -101.68 | -106.34 |
| A5 challenger | -90.96 | -159.21 | -163.71 | -144.47 |

A5 April global top-10% / latest-seven-day EV is
`-59.28 / -131.09` bps under the baseline,
`-73.73 / -102.86` under M1,
`-83.43 / -93.20` under M2, and
`-64.53 / -88.94` under M3.  The residual control remains better on the
aggregate March and April evidence under every same mapper.  Some alternative
maps reduce A5's latest-seven-day loss, but none repairs aggregate economics
or passes promotion.

**Decision:** close generic mapping repair as the primary explanation of the
IC-to-EV failure.  Do not replay a portfolio and do not tune another flexible
calibrator on this cohort.  Continue with the already frozen cost-aware
opportunity/exit decomposition, signed conditional payoff branches, older
same-lineage history and untouched-forward evidence.  Mapping remains a
required causal transport layer, but it is not the current bottleneck.

### 2026-07-30 A-grade cost-clearing conversion v5: completed, negative

The restart-safe exact test is sealed at
`data_perp/artifacts/a_grade_cost_clearing_conversion_ablation_20260730_v5`.
It contains seven immutable 14-day checkpoints (five scored, two warm-up),
and all six focused tests, input/output hashes, the manifest hash and current
runner hash verify.

The test uses one common exact-identity cohort per lineage, a pooled global
top 10% after arm-local causal mapping, and no side quota. The alpha hurdle
predicts `execution_net_ev_12h > 0` from residual EV, base alpha and side,
then combines that probability with train-only side-conditional positive and
negative payoffs. All fold labels resolve before the test block. Strict
2025-to-2026 forward scoring fits and maps on 2025 only.

| Strict 2025-fit -> 2026 global top 10% | Aggregate monthly EV | July EV |
|---|---:|---:|
| Residual control | -96.16 bps | -126.02 bps |
| Alpha cost-clearing hurdle | -106.21 bps | -114.36 bps |

The hurdle improves July by 11.66 bps but degrades the aggregate by 10.05 bps;
both remain uneconomic. It also concentrates the selected book into only
10--13 assets, versus roughly 98--113 for the residual control. This is not
a viable conversion repair.

The 2025 within-lineage regime-plus-transition diagnostic is also negative
(`-103.19` bps versus residual `-37.91` in April). The July within-lineage
diagnostic improves to `-86.45` versus `-96.62`, but it is not forward
evidence: the historical and current context sidecars have incompatible
feature semantics, so all regime/transition arms correctly fail closed in
the strict 2025-to-2026 test.

**Required continuation:** materialise one semantically identical, pre-entry
regime-transition feature contract across the older and current lineages.
Then rerun the same frozen cost-clearing hurdle with 2025-only fit/mapping and
2026 forward application. The next hypothesis is conditional conversion
under transferable state/transition context—not another map or an unconstrained
positive-net classifier. Preserve the exit-outcome decomposition as the
parallel mechanism test.

### 2026-07-30 nonlinear alpha-tail hurdle: completed, rejected

The fixed no-HPO test is sealed at
`data_perp/artifacts/nonlinear_alpha_tail_cost_clearing_hurdle_20260730_v1`.
It adds deterministic timestamp-by-side alpha percentiles, 20 ventiles and
80/90/95% tail hinges to the v5 cost-clearing hurdle. Stable ties use alpha
descending, symbol ascending and candidate ID ascending. Training is blocked
2025 OOF; the fit and blocked-OOF map are frozen before 2026. All arms share
identical eligible IDs and use one pooled fractional global top 10%.

| Frozen-forward global top 10% | June | July | Monthly average |
|---|---:|---:|---:|
| Residual control | -66.45 bps | -131.46 bps | -98.96 bps |
| V5 linear hurdle | -101.59 bps | -116.53 bps | -109.06 bps |
| Nonlinear alpha-tail hurdle | -101.50 bps | -117.89 bps | -109.70 bps |

The nonlinear arm is worse than both controls in aggregate. Its long and
short forward contributions are `-113.96 / -98.66` bps. Causal legality,
identity parity, global-selection and frozen-forward gates pass, but economic,
both-side and tie gates fail. The mapped score is effectively flat:
99.997%/99.940% of June/July candidates share the cutoff. This is exact
fractional allocation, not a hidden deterministic tie break, but it has no
usable selection resolution.

**Decision:** reject the hypothesis that a fixed piecewise alpha-tail response
alone repairs the July conversion failure. Do not add more alpha bins, knots
or flexible calibration on this cohort. Proceed to the already materialized
common-semantic regime-transition geometry and the explicit exit-outcome
decomposition.

### 2026-07-30 common-semantic transition conversion: completed, rejected

The cross-era test is sealed at
`data_perp/artifacts/common_semantic_transition_cost_clearing_ablation_20260730_v1`.
It binds the hash-verified 90-field
`historical_current_common_transition_geometry_20260730_v1` contract and
splits it a priori into 36 state-level and 54 strict 1/3/12h lag/delta fields.
No source, calendar, outcome, provenance or state-ID field enters a model.

Exact timestamp joining with no fill yields 110,610 historical and 51,279
forward complete-case rows before the common map-eligibility intersection.
The final identical comparison cohort contains 50,706 blocked-2025 OOF and
51,279 forward IDs for all five arms.

| Frozen-forward mapped global top 10% | Aggregate net EV |
|---|---:|
| Residual control | -82.62 bps |
| V5 linear alpha hurdle | -105.74 bps |
| Common state hurdle | -105.12 bps |
| Common transition-delta hurdle | -105.74 bps |
| Common state + transition hurdle | -105.37 bps |

State and combined context make only sub-basis-point improvements over the
linear hurdle and remain far behind the residual control. All challenger
sides remain negative. Cutoff ties are also severe: 80.5--100% by challenger
on the forward mapped books.

There is one diagnostic nuance. Before mapping, the transition-only hurdle
has `-72.82` bps aggregate top-10 EV versus `-84.42` for the raw residual
control, with roughly 14 bps improvements in both June and July. The causal
map collapses this arm to one score and removes its selection resolution.
This is insufficient for promotion because even the raw tail remains
negative, but it supports retaining the common transition score as a bounded
candidate interaction in a later architecture rather than as a standalone
admission head.

**Decision:** reject state/transition feature expansion inside the generic
binary cost-clearing hurdle. Do not promote or replay. Await the explicit
exit-outcome hierarchy. If that identifies a viable component, test one
bounded residual-plus-transition interaction with a strictly order-preserving
common-unit map; do not run another unconstrained 90-feature admission model.

### 2026-07-30 explicit exit/outcome hierarchy: completed, rejected

The final artifact is sealed at
`data_perp/artifacts/canonical_execution_reliability_exit_hurdle_ablation_20260730_v1`.
The runner now has immutable identity-bound side/fold/head checkpoints, a
single-run lock, explicit progress/error logging, default safe resume and
fail-closed runner/config/input/target/parent fingerprints. The clean rerun
sealed 290/290 units with zero errors; nine focused tests and every checkpoint,
output, manifest and current-runner hash verify.

The learned architectures are:

- H1: cost-buffer opportunity probability, successful-trailing probability,
  conditional gain, and signed no-opportunity/opportunity-failure payoffs;
- H2: four-class realised exit probability plus class-conditional payoff;
- H3: successful-trailing hurdle followed by competing-risk class/payoff;
- H4: severe-loss probability, conditional severity and signed non-severe
  payoff.

The corrected H1 formula retains positive and negative signed failure branches;
it does not silently treat every non-success as an adverse magnitude.

| Mapped global top 10% | March | April | April latest 7d |
|---|---:|---:|---:|
| Residual control H0 A0 | -71.44 bps | -30.21 bps | -81.23 bps |
| H1 0bps selection winner | -98.27 bps | -62.40 bps | -90.24 bps |
| H2 four-class | -119.57 bps | -66.50 bps | -99.40 bps |
| H3 hierarchical | -115.36 bps | -57.23 bps | -79.61 bps |
| H4 severe-loss hurdle | -81.51 bps | -63.71 bps | -72.19 bps |

H1 0bps wins the learned March stability objective at `-131.03` bps, with
folds `-109.92/-108.17/-52.66`; it is still materially worse than the
control. Its March book is 91.9% short by selected mass, with long/short net
contributions of `-3.64/-94.64` bps, and a 22.57% aggregate cutoff-tie
selected share. All seven promotion gates fail.

The event heads are only moderately learnable: H1 opportunity AUC is about
0.605--0.613, H1 conditional-success AUC about 0.602--0.613, H2 macro AUC
0.587, H3 success AUC 0.596/risk macro AUC 0.584, and H4 severe-loss AUC
0.540. Conditional payoff ranking is stronger in places—hard-adverse payoff
rank IC is about 0.521 and non-severe payoff about 0.281—but the complete
probability/payoff compositions do not clear cost.

**Decision:** reject all four as replacement execution-EV heads; no portfolio
replay. Preserve the useful adverse conditional predictions only for one
bounded residual-score risk-overlay ablation. Do not add timing, MAE,
target-price or wait/reprice outputs to this layer.

### 2026-07-30 bounded adverse-risk overlay: completed, rejected

The final component-use test is sealed at
`data_perp/artifacts/bounded_adverse_risk_overlay_ablation_20260730_v1`.
It reconstructs 40 exact H2/H4 checkpoint payloads with checkpoint fingerprint,
train/validation identity, class-order and payload-hash validation. No model is
refit and no checkpoint is modified.

The overlays preserve the residual EV score and add only clipped, economically
signed H2 expected hard-adverse payoff and/or H4 expected severe-loss penalties.
Fold/side clipping bounds use training outcomes only. Fixed
`lambda={0.25,0.5,1.0}` arms are selected on the frozen March stability
objective, then evaluated on aggregate March and frozen April.

`h2_lambda_0.25` is the selection winner:

| Mapped global top 10% | Residual control | H2 risk overlay |
|---|---:|---:|
| March stability objective | -80.25 bps | -79.59 bps |
| March aggregate | -71.44 bps | -72.10 bps |
| April aggregate | -30.21 bps | -30.65 bps |
| April latest seven days | -81.23 bps | -81.17 bps |

The overlay improves the stability objective by only 0.66 bps and latest-seven
days by 0.06 bps, while losing both aggregate comparisons. Both March side
contributions remain negative. Tie safety passes, but all economic and
promotion gates fail.

**Decision:** adverse predictions are not incrementally useful even as a
bounded overlay on this cohort. Close the current meta-ranking branch: mapping,
linear/nonlinear cost-clear hurdles, common transition context, explicit exit
hierarchies and bounded adverse penalties have all failed the identical global
book. Do not run a portfolio replay.

The next diagnosis must move upstream to opportunity creation and exit-policy
capture on the unchanged residual-selected IDs. Quantify the 12h MFE oracle,
fixed-horizon returns, deployed-exit capture ratio, full-stop/timeout regret
and cost-clearing opportunity prevalence by month/side/global top 1/5/10/20%.
Only if that counterfactual proves adequate gross opportunity should another
selector be trained; otherwise change the base/residual candidate target,
holding period, entry universe or cost/exit design.

### 2026-07-30 identical-ID opportunity/exit counterfactual: completed

The replacement sealed diagnostic is
`data_perp/artifacts/residual_selected_exit_opportunity_counterfactual_20260730_v3`.
The first `v1` artifact is explicitly invalidated because its fixed-close arm
used the native base-label path lineage.  `v2` uses the exact execution-path
materialization, asserts one-to-one candidate, side, normalized-symbol, signal,
decision and minute-719 endpoint parity, and subtracts the canonical row cost
exactly once.  `v2` is also invalidated: its primary economics were correct,
but it described MFE through the exit minute as strictly pre-exit and did not
mask that auxiliary metric to exact policy-path-parity rows.

Selection is unchanged: causal mapped H0 A0 residual score, one pooled-global
book within month, fractional expected boundary-tie membership, global top
1/5/10/20%.  Counterfactual outcomes are joined only after selection; there is
no per-timestamp or per-side ranking and no reranking.

At global top 10%:

| Metric | March | April frozen |
|---|---:|---:|
| Deployed net | -71.44 bps | -30.21 bps |
| 12h MFE oracle net of canonical cost | +145.15 | +188.18 |
| Fixed minute-720 close net of canonical cost | -21.80 | +53.89 |
| Through-exit-minute MFE net, parity-valid rows | +42.19 | +69.50 |
| 25-bps-buffer opportunity prevalence | 41.99% | 48.39% |
| Full-stop rate | 21.96% | 15.76% |
| Timeout rate | 25.76% | 27.38% |
| Full-stop oracle-regret contribution | +75.86 bps | +56.03 bps |
| Timeout oracle-regret contribution | +26.36 bps | +25.87 bps |

UTC-day clustered 95% intervals confirm that the top-10 deployed book is
negative (`[-86.19,-49.29]` March; `[-55.71,-5.67]` April) while the MFE
oracle is positive (`[109.31,188.43]`; `[138.14,245.28]`).  The fixed 12h
close is inconclusive/negative in March (`[-70.56,33.13]`) but positive in
April (`[10.77,106.09]`).  Therefore a blind 12h hold is not the repair.

The exit attribution is sharper:

- full-stop rows have negative oracle net even with hindsight
  (`-59.77/-50.24` bps March/April top 10%), so these are primarily bad
  admissions, not merely premature exits;
- timeout rows also have negative oracle net (`-46.86/-64.30` bps), again
  pointing to no-opportunity admissions;
- trailing rows contain strong oracle opportunity (`+326.56/+376.18` bps)
  but deploy only `+107.76/+136.04` bps, leaving substantial capture
  shortfall;
- March long is the critical persistence failure: deployed/fixed-12h are
  `-105.61/-87.43` bps, whereas March short fixed-12h is `+51.46` bps.
  April fixed-12h is positive on both sides.

**Diagnosis:** the selector does find economically meaningful path
opportunity, so the negative EV is not explained by a complete absence of
gross movement.  The failure is a mixture of (a) admitting full-stop/timeout
rows with no cost-clearing opportunity and (b) under-capturing profitable
trailing rows.  The March-long path does not persist to 12h, so exit repair
must be side/path/regime aware rather than a universal longer hold.

**Next bounded workstream — separate action layer, no selector change:**

1. On the same selected IDs, materialize fixed 1/2/4/8/12h, time-to-MFE,
   post-MFE giveback, early 2--3-bar MAE/flatness and time-under-water targets.
2. Replay a small frozen exit family: deployed control; fixed horizons;
   partial profit plus trailing remainder; time stop; trailing-width/decay
   variants.  Apply identical row costs and one pooled-global book.
3. Split the action problem explicitly:
   - pre-entry `trade/skip/wait/reprice` for predicted no-opportunity,
     full-stop and timeout risk;
   - post-entry `hold/partial/exit/tighten/loosen` for opportunity capture.
4. Train every action head on OOF/OOF-equivalent exact paths only.  Keep
   timing, MAE and target-price fields out of the execution-EV ranking head.
5. Require positive month and side economics, day-cluster lower bounds,
   bounded turnover/cost, and improvement over both deployed and fixed-time
   controls before portfolio replay.

Do not reopen the failed downstream meta-ranking arms or run the simple policy
optimizer yet.  First determine whether a causal, bounded exit/action family
can capture the demonstrated opportunity without hindsight.

### 2026-07-30 action-target pack and fixed-horizon controls: completed

The target-only action dataset is sealed at
`data_perp/artifacts/execution_action_target_pack_20260730_v2`.  It contains
110,730 canonical rows backed by exact contiguous 720x1m execution paths.
All returns, MFE, MAE, timing, slope, underwater and giveback fields are
side-relative.  Fixed-close targets are available at 1/2/3/4/8/12h and deduct
the canonical row cost exactly once; target availability is declared at the
corresponding horizon.  Full-path peak/timing/giveback and cost-clear labels
resolve only at 12h.  Early 2--3h clean/non-flat, adverse and cost-positive
flags are included.  No selection, score, rank or weight field is present,
and every label is forbidden as an inference feature.

`execution_action_target_pack_20260730_v1` is
`INVALIDATED_DO_NOT_USE`: zero-MFE paths received spuriously finite
fraction-of-peak timing.  `v2` marks peak timing invalid, censors the
time-to-event fields at 12h and leaves giveback undefined on those rows.

The paired fixed-control artifact is
`data_perp/artifacts/fixed_horizon_action_ablation_20260730_v2`.  It evaluates
deployed and forced minute-60/120/240/480/720 closes on the unchanged v3
fractional pooled-global monthly top-1/5/10/20 books, reusing the canonical
cost once and drawing 2,000 paired UTC-day bootstrap samples.  Deployed
control parity is below `1.5e-14` bps.

At global top 10%:

| Arm | March net | April net |
|---|---:|---:|
| Deployed | -71.44 bps | -30.21 bps |
| Fixed 1h | -74.70 | -60.14 |
| Fixed 2h | -63.07 | -39.73 |
| Fixed 4h | -57.43 | -15.26 |
| Fixed 8h | -46.90 | +22.08 |
| Fixed 12h | **-21.80** | **+53.89** |

Fixed 12h has a March 95% interval of `[-69.06,+39.78]` bps and April
`[+8.70,+107.58]`.  It improves deployed by `+49.64/+84.10` bps, but March
remains negative and March long is `-87.43` bps.  A universal fixed hold is
therefore rejected.  `fixed_horizon_action_ablation_20260730_v1` is
`INVALIDATED_DO_NOT_USE` only because its capacity assertion hard-coded the
March/April populations; `v2` derives them from the sealed parent selection.

### 2026-07-30 frozen state-machine exit family: completed, rejected

The exact replay is sealed at
`data_perp/artifacts/frozen_exit_state_action_ablation_20260730_v4`.
The earlier v1 is the complete three-arm phase before partial exits; v2 added
P50 but did not materialize its exact two-exit fee robustness table.  v3 is
explicitly invalidated for P50 only: it applied intrabar stop/target/adverse
checks before an activation already known at the bar open.  v4 executes the
partial at that open first and then runs the unchanged remainder state.
The canonical simulator reproduces all 18,107 selected deployed outcomes
exactly: zero mismatches in gross/net, exit hour/reason, executable
entry/exit prices, spread fields and geometry key.  IDs, scores, fractional
weights and global books remain frozen.  Variant gross returns reuse the
sealed deployed row cost once rather than recomputing a return-dependent fee.

The predeclared stateful arms are:

- `T4`: deployed stops/trailing on the first 240 observed minutes, then the
  normal spread-aware timeout fill for survivors;
- `D2`: after minute 120, decay the trailing-activation threshold with a
  120-minute half-life toward 50% of its original level;
- `W75`: tighten the active trailing-gap parameter to 75%, preserving the
  simulator's minimum-gap floor;
- `P50`: when trailing activation first becomes causally known from the path
  through minute `j-1`, take 50% at minute `j`'s executable open with the
  deployed spread/gap proxy, and keep the remaining 50% in the unchanged
  state machine.

Global top-10 results:

| Arm | March net | April net | March delta vs deployed | April delta vs deployed |
|---|---:|---:|---:|---:|
| `T4` | -121.16 bps | -88.83 bps | -49.72 | -58.62 |
| `D2` | -67.94 | -27.56 | +3.50 | +2.65 |
| `W75` | -72.48 | -32.34 | -1.04 | -2.13 |
| `P50` | -80.43 | -41.27 | -8.98 | -11.06 |

`T4` is not equivalent to the optimistic raw fixed-4h control: it preserves
the live state machine and uses the deployed adverse close-fill/spread
contract.  It is robustly worse.  `D2` changes gross outcomes on about 17.7%
of rows and is the only directionally helpful state arm, but its paired
95% delta interval crosses zero in both months (`[-2.19,+10.16]` March;
`[-0.87,+6.14]` April), every global month remains negative, and the
March-long/short and April-short books remain negative.  `W75` is
nonincremental and is significantly worse in April.

`P50` activates on 56.38%/61.75% of the March/April top-10 book at a
weighted mean 4.07h/3.68h.  It is significantly worse than deployed:
the paired delta intervals are `[-12.74,-6.00]` and `[-13.51,-8.81]` bps.
The exact two-exit fee computation changes canonical-cost net by only
+0.045/+0.055 bps, leaving the rejection unchanged.  Unconditional
profit-taking at first activation therefore sacrifices more subsequent
payoff than it protects.

All strict positivity,
side, uncertainty and fixed-12h comparison gates fail; no arm is promotable
and no portfolio replay is authorized.

The remaining bounded action work is:

1. use the sealed target pack for OOF/OOF-equivalent action heads, beginning
   with `trade/skip/wait/reprice` for no-opportunity risk and then
   `hold/partial/exit/tighten/loosen` for capture;
2. treat `D2` and `P50` only as possible conditional actions.  Do not tune a
   wider decay or partial-fraction grid on reused March/April; a learned
   action must decide OOF when each mechanism is useful;
3. retain timing, MAE and target-price outputs exclusively in the separate
   action layer;
4. require positive global and side economics in each untouched month,
   positive paired day-cluster lower bounds, and improvement over deployed
   plus fixed-time controls before simple-policy or portfolio replay.

### 2026-07-30 frozen pre-entry wait action: infrastructure complete, first learner rejected

The no-reranking learning handoff is sealed at
`data_perp/artifacts/frozen_entry_action_handoff_20260730_v2`.  It preserves
all 18,107 frozen pooled-global identities and fractional top-1/5/10/20
weights, joins the 45 authorised decision-time inputs, exact future action
targets, exact 720x1m paths, and the deployed barrier/archetype replay inputs.
Path symbols use `/` while canonical identities use `_`; the materializer
joins by candidate and side and then requires timestamp plus normalized-symbol
parity.  Paths, targets, mapped score, weights and policy-replay inputs are
separate non-feature roles.

The historical `execution_entry_timing_meta` simulator was explicitly
rejected for this test: on a control sample it differed from the current
deployed exit policy by as much as about 315 bps.  The new runner instead uses
the exact current `simple_policy_optimiser` pathway and refuses to produce an
action result unless enter-now reproduces the deployed control.

The canonical result is sealed at
`data_perp/artifacts/frozen_preentry_wait10_action_ablation_20260730_v2`.
Its wait action has no position during minutes 0--9, enters at the raw
minute-10 open through the normal spread/slippage pathway, keeps the original
barrier and side strategy, and runs on the remaining 710 minutes to the
original absolute 12h deadline.  It recomputes action-specific costs once.
Enter-now has zero mismatches on gross/net, exit hour/reason, MFE/MAE,
entry/exit prices, spread fields and geometry key.  The frozen identity/weight
digest is unchanged.  March action labels resolve strictly before every
chronological OOF validation boundary.  v1 is invalidated because its maximum
training label-resolution timestamp equalled validation start.

Global top-10 economics:

| Evaluation | Enter now | Always wait 10m | Oracle wait 10m | Best learned diagnostic |
|---|---:|---:|---:|---:|
| March chronological OOF | -73.04 bps | -84.97 | -66.58 (`+6.46`) | full soft: -73.48 (`-0.44`) |
| April frozen March-forward | -30.21 bps | -40.58 | -25.47 (`+4.74`) | full soft: -32.83 (`-2.62`) |

The April oracle improvement interval is `[+3.50,+5.87]` bps, proving a
small conditional wait opportunity exists, but every learned April policy is
significantly worse than enter-now.  Full-soft has an April interval of
`[-3.73,-1.64]` bps; full-direct is `[-4.44,-1.74]`.  March full-soft is
statistically inconclusive at `[-1.01,+0.22]`, but still not positive.

The better-wait classifier has useful discrimination despite the economic
failure: April AUC is `0.738/0.689` long/short with the compact inputs and
`0.746/0.709` with all authorised inputs.  The event is rare, however:
wait improves only `5.0%` of long and `10.4%` of short rows.  Delta magnitude
rank correlations are approximately zero, and the learned policies wait on
roughly 26--28% of the top-10 book versus a 5.1% oracle action rate.  Thus the
current bottleneck is cost-asymmetric magnitude/calibration and abstention,
not basic event separability.  April top-10 long remains `+9.71` bps when the
direct/soft policy abstains completely; the harmful learned actions are
concentrated in short.

Consequences:

1. Do not portfolio-replay or promote wait10; every learned global policy
   fails economics and uncertainty.
2. Extend action training to older strict OOF/forward blocks.  Fit the rare
   better-wait event separately from positive/negative magnitude, use
   cost-asymmetric sample weights, calibrate on inner chronological OOF, and
   require a positive expected-utility lower bound before waiting.
3. Add a current-policy adverse-limit action next, with explicit fill,
   adverse-first and missed-opportunity heads.  Do not reuse the obsolete
   timing simulator or reselect candidates.
4. Materialize causal post-entry prefix states before any learned
   hold/exit/P50/D2 router.  Future aggregate path targets remain labels only.

### 2026-07-30 older-data Wait10 extension: event learning improves, economic routing does not

The exact all-candidate training bridge is sealed at
`data_perp/artifacts/febapr2025_current_policy_wait10_action_20260730_v1`.
It contains all 205,194 canonical residual-top40 rows from February--April,
34 exact signal-time inputs, exact current-policy enter-now labels and
current-policy Wait10 counterfactuals.  Full-row enter-now parity is exact for
gross, cost, net, exit time/reason, MFE/MAE, prices, spreads and geometry.
Wait10 retains the original absolute deadline and recomputes action costs once.
No historical global-book membership is inferred or reconstructed.

This distinction matters.  Wait10 is better on the broad residual-top40
population at rates of 20.36%/23.27% long/short in February,
18.30%/22.73% in March and 18.97%/22.73% in April.  On the frozen mapped
top-10 book the event is only about 5--10%.  Older all-candidate training is
therefore an action-conditionality experiment, not evidence that February
extends the same residual/recent-EV selected-book lineage.  February has OOF
base scores but no OOF residual score, mapped score or frozen global weights.

The frozen evaluation is sealed at
`data_perp/artifacts/frozen_older_data_wait10_action_ablation_20260730_v1`.
It compares:

- February all-candidate versus February base-rank-top-half training;
- March-all-resolved and February-plus-March-resolved training for April;
- base-only, base-plus-transition and all 34 state/transition inputs;
- event, economically weighted event, direct delta, conditional magnitude,
  q25 lower-bound and soft-delta heads;
- fixed rules plus a train-only day-cluster lower-bound abstention rule.

The enlarged context improves event classification.  With all state and
transition inputs, March AUC reaches 0.761 long / 0.795 short from February
training.  April reaches 0.795 long / 0.756 short from
February-plus-resolved-March training.  This does not solve value routing:
direct and expected-delta magnitude predictions are effectively constant or
non-ranking in the selected tail, and the expected-delta rule waits on about
98% of the book and loses roughly 10 bps.

At global top 10%, the complete frozen March book is -71.44 bps enter-now;
the Wait10 oracle is -65.12 (+6.32) and always-wait is -82.86 (-11.41).
The best non-oracle result is February/base-only q25 at +0.004 bps with a
95% interval of approximately [-0.005,+0.013].  The complete frozen April
book is -30.21 bps; the best diagnostic is February-base-rank-top-half,
base-only direct delta at -29.93 (+0.28), waiting on 2.30%.  The gain is
short-side only (+0.38 bps within short; long abstains), and its paired
interval [-0.01,+0.63] crosses zero.  It is not promotable.

The train-only positive-lower-bound gate abstains in 23 of 24
source/feature/side calibrations.  Its one admitted threshold changes only
0.058% of the April top-10 weight and is economically flat/slightly negative.
This is correct safety behavior and confirms that event AUC is not the
bottleneck: conditional magnitude and selected-tail calibration remain the
bottlenecks.  Portfolio replay stays unauthorized.

Next gates:

1. finish the 293,828-row Apr-2023--Dec-2024 all-candidate action ledger
   using held-block OOF base/residual scores, candidate-keyed OOF
   regime-transition context and exact side-archetype replay parity;
2. train cross-era event/magnitude/lower-tail heads on that ledger and score
   only the unchanged March/April book; do not manufacture older book weights;
3. implement the causal mapped-score conversion bridge before making any
   production claim about rising base IC versus deployed global top-k EV;
4. proceed to adverse-limit fill/adverse-first/missed-opportunity labels only
   after the exact current-policy replay contract is preserved.

### 2026-07-30 cross-era Wait10 regime diagnosis: the same coarse state changes sign

The larger exact training ledger is now sealed at
`data_perp/artifacts/2023apr_2024_current_policy_wait10_action_20260730_v1`:
293,828 Apr-2023--Dec-2024 rows, 21 held-block OOF score/regime-transition
inputs, exact 720x1m paths, exact side-archetype geometry and zero enter-now
parity mismatches.  A persisted-archetype normalization trap was caught before
materialization: replay must remove the already-added `policy_archetype_`
prefix before the canonical resolver adds it once.  Otherwise geometry silently
falls back to side-parent even when gross/net happen to match.

Wait10 economics themselves change regime:

| Training era | Long mean delta | Short mean delta |
|---|---:|---:|
| Apr--Dec 2023 | +1.11 bps | +2.41 bps |
| 2024 | +4.22 | +2.24 |
| February 2025 | -5.44 | -9.87 |
| March 2025 | -7.86 | -6.49 |
| April 2025 | -6.32 | -4.92 |

The first common-feature transfer result is sealed at
`cross_era_wait10_transition_ablation_20260730_v1`.  It trains on all
Apr-2023--Dec-2024, 2024 only, or Q4-2024; evaluates only the unchanged
March/April 2025 frozen books; and compares the two common score fields,
25 raw transition fields from the same sealed hourly calendar, and their
combination.  Best global top-10 lifts are only +0.20 bps in March and
+0.08 bps in April, with intervals touching zero.  Event AUC is at most about
0.625.  Raw base/residual scores are also badly out of range: current base
score medians are roughly 5--7 historical IQRs higher and 84--93% of current
base scores lie outside historical 1st--99th percentiles.  Raw score-level
transfer is therefore rejected; a timestamp/side rank-normalized arm is
required next.

More importantly, historical terciles expose a taxonomy failure.  Every March
and April 2025 row is in the historical `high transition entropy / low
persistence` cell.  That exact historical cell has +3.70 bps mean Wait10
delta, while the 2025 population loses -7.17 bps in March and -5.62 bps in
April.  Entropy/persistence identify a broad transition state but cannot
identify the economically opposite subtype.

The preregistered subtype expansion is sealed at
`cross_era_wait10_transition_ablation_20260730_v2`.  It adds BTC-versus-alt
resilience, breadth dispersion/intensity, compression quality, recent
short-damage structure, funding change and state age.  These fields improve
April long event AUC to 0.678 and raise the best diagnostic lifts to
+0.39 bps March and +0.15 bps April.  The paired intervals remain
[-2.14,+2.96] and [-0.08,+0.41] bps.  No route is promotable.

Interpretation and next ablations:

1. transition subtype features are incremental for event detection, but still
   do not learn stable action magnitude;
2. score-level non-transfer must be removed with timestamp/side rank and
   robust-z coordinates computed from the complete candidate groups in each
   era, not from the frozen selected subset;
3. add an explicit era-invariant transition-subtype classifier using
   BTC-alt resilience gap, breadth dispersion/intensity, compression quality
   and recent damage/funding structure, then test calibration on held months;
4. split event probability from conditional positive/negative magnitude and
   train the magnitude heads within matched transition subtypes;
5. keep portfolio replay disabled until a frozen learned route has a positive
   paired lower bound in both March and April.

#### Complete-group rank-normalized cross-era result

The required rank-normalized arm is sealed at
`data_perp/artifacts/cross_era_wait10_rank_normalized_ablation_20260730_v1`.
Ranks and z-scores are computed within each **complete timestamp-side candidate
group** before the unchanged March/April frozen identities are joined.  The
comparison uses base rank/z, residual rank/z and residual-minus-base rank,
alone and with the expanded causal transition-subtype fields.

This removes the catastrophic raw-score scale mismatch but does not rescue
cross-era action transfer:

| Evaluation | Best frozen top-10 lift | Action weight | Paired 95% interval |
|---|---:|---:|---:|
| March 2025 | +0.164 bps | 2.44% | [0.000, +0.339] |
| April 2025 | +0.094 bps | 0.217% | [0.000, +0.328] |

Both intervals touch zero.  Event AUC reaches about 0.645, below the 0.678
April-long result from the expanded transition-only arm.  No configuration is
promotable and portfolio replay remains disabled.

The remaining shift is partly candidate-population geometry.  Historical
timestamp-side groups contain roughly 7--15 candidates, versus about 48 in the
current complete panel.  Consequently, even after complete-group
normalization, current rows remain out of historical support: mean
outside-historical-1st--99th-percentile rates are approximately 20.9% for
base rank, 29.8% for residual rank and 16.0% for residual z.  Rank
normalization therefore proves that raw scale was a confounder, not the cause
of the sign reversal.

Next cross-era action work must:

1. use within-group quantile/decile coordinates plus group size, cutoff margin
   and candidate-density context, rather than assuming percentile rank alone
   is era invariant;
2. align or match training rows on candidate-group geometry before fitting
   conditional action magnitude;
3. retain the expanded transition subtype fields, but require held-month
   economic calibration and a strictly positive paired lower bound in both
   March and April before portfolio replay.

### Base IC versus EV: resolved descriptive mechanism and remaining mapping gap

Do not rerun the raw base-score waterfall.  The sealed
`historical_base_ic_execution_ev_change_attribution_20260729_v1` and
`mandatory_ic_ev_waterfall_20260730_v1` already establish the mechanism on
identical exact-policy rows.  The quoted `0.155 -> 0.162 -> 0.226` sequence is
the **long-side native-24h target IC**, while the quoted top-decile economics
are a **pooled-global raw-base book**; they must never be presented as the same
axis.  Long exact-net IC also rises `0.090 -> 0.093 -> 0.143`, yet pooled
global top-10 gross/cost/net is:

| Month | Gross | Cost | Net |
|---|---:|---:|---:|
| February | +49.38 bps | 100.25 bps | -50.87 bps |
| March | +17.05 | 100.09 | -83.03 |
| April | +41.86 | 100.21 | -58.35 |

February-to-March deterioration is payoff/conversion, not primarily rank-cell
composition: favourable-payoff scale contributes about -21.88 bps and
positive-net prevalence -21.70 bps, while score-cell composition is only
+0.26 bps.  Fixed-12h close improves the same raw-base books to
-37.07/-60.79/-6.55 bps, so exit conversion explains material loss but does
not make the book reliably profitable.

The remaining missing experiment is narrower: freeze one exact
February--April table comparing raw base, causal-mapped base, residual, true
raw direct-EV and causal-mapped direct-EV under the same mapping availability
and one pooled-global top-k rule.  Existing historical waterfalls diagnose
base alpha; they do not prove the deployed causal-mapped selection protocol.
Do not fit a new model for this bridge.

#### Completed identical-row causal score-conversion bridge

The missing comparison is now sealed at
`data_perp/artifacts/marapr2025_identical_causal_score_bridge_20260730_v1`.
It fits no alpha, residual or direct model and performs no HPO.  It reconstructs
the exact OOF membership of the true direct q25 head, proves bit-identical base,
residual and direct scores on all four identity fields, and fits two
predeclared diagnostic calibrators on the **same** candidate population:

- raw base alpha -> exact 12h net;
- raw direct q25 -> exact 12h net.

Both use one pooled 21-day isotonic map, only labels resolved before the UTC
day snapshot, and the same 2,000-row minimum support.  March 1--2 have
insufficient causal support, so every arm is restricted to the identical
136,074 rows: 66,816 in March and 69,258 in April.  February remains
unavailable because strict residual and true direct-q25 OOF lineages begin in
March.  Selection is one pooled-global monthly top-k after mapping, never per
timestamp or side.

Identical-row pooled-global top-10 economics are:

| Layer | March net | April net |
|---|---:|---:|
| Raw base alpha | -66.93 bps | -33.94 bps |
| Causal-mapped base | -76.15 | -45.16 |
| Residual expected EV | **-31.15** | **-24.32** |
| Raw direct q25 | -27.48 | -93.24 |
| Causal-mapped direct q25 | -29.98 | -107.92 |

This resolves the apparent base-IC/EV contradiction more precisely.  On the
same pooled rows, raw-base native-target IC rises 0.147 -> 0.184 and exact-net
IC rises 0.068 -> 0.112.  Its top-10 gross also rises +33.24 -> +66.39 bps,
while explicit cost remains about 100 bps.  The improved rank therefore does
translate into better gross and net economics; it simply does not clear the
cost hurdle.

The downstream findings are stricter:

1. the residual layer is the only arm that improves the raw-base tail in both
   months (+35.78 bps March, +9.62 April), although both improvements have
   day-bootstrap intervals crossing zero and net remains negative;
2. the direct q25 head is a regime-transfer failure: it is competitive in
   March and collapses in April.  The April global book allocates 74.2% to
   short; short averages -137.05 bps while long is +33.05 bps;
3. causal mapping does not repair either source.  It worsens base by
   -9.23/-11.22 bps and direct q25 by -2.50/-14.69 bps in March/April;
4. isotonic compression is material: mapped scores retain only about
   0.8--1.2% unique levels.  April mapped-direct top-10 is concentrated in
   19 days, with 66.4% of rows in five days.  The global map is changing
   cross-day allocation, not merely calibrating within-timestamp ranks.

No layer clears cost, no mapped route is promotable, and portfolio replay
remains disabled.  The next conversion work must use residual expected EV as
the ranking incumbent; ablate rank-preserving/shrunk causal mappings against
isotonic plateaus; and repair the April short conditional payoff/adverse-risk
conversion before any further direct-head expansion.

#### Mapping-repair ablation: sealed negative result and sharper regime boundary

The fixed mapping-only comparison is sealed at
`data_perp/artifacts/marapr2025_causal_mapping_repair_ablation_20260730_v1`.
It changes no alpha, residual or direct model, labels, costs, identities,
top-k rule or action layer.  March 3--19 is the only arm-selection window;
March 20--31 and April are separately reported reused-month confirmations.
The four predeclared arms isolate isotonic mapping, raw-score tie-breaking,
25% shrinkage toward a positive-slope robust Huber/tanh map, and their
combination.  Every daily fit uses only prior-resolved labels from the same
pooled 21-day reference set.

Pooled-global top-10 net economics are:

| Source / arm | Mar 3--19 | Mar 20--31 | April |
|---|---:|---:|---:|
| Base raw | -31.07 bps | -112.72 bps | -33.94 bps |
| Base isotonic | -46.92 | -105.93 | -48.58 |
| Base isotonic + raw tie-break | -46.57 | -106.18 | -48.60 |
| Base 75/25 isotonic--Huber | -43.10 | -102.45 | -48.57 |
| Direct q25 raw | **+20.97** | **-96.44** | **-93.24** |
| Direct q25 isotonic | -0.29 | -98.69 | -109.10 |
| Direct q25 isotonic + raw tie-break | +0.01 | -98.99 | -109.66 |
| Direct q25 75/25 isotonic--Huber | -3.16 | -97.91 | -114.45 |
| Residual control | -9.57 | -60.43 | **-24.32** |

Tie-breaking removes candidate-ID dependence at exact plateaus, and the
Huber blend restores score uniqueness (about 94% for base and 92% for direct
in the selection window).  Neither restores calendar allocation or
economics.  Mapped base effective days fall from 13.91 raw to about
10.6--10.9 and top-three-day share rises from 29.7% to 42--43%.  Direct
mapping loses about 21--24 bps versus raw during the selection window; the
paired bootstrap intervals are strictly negative.  Both source decisions
therefore correctly resolve to `ABSTAIN`; no promotion or portfolio replay is
authorized.

The decisive result is temporal rather than map-specific.  Raw direct q25
changes from +20.97 bps on March 3--19 to -96.44 bps on March 20--31, before
the April failure.  The March confirmation book is 85.2% short and its short
rows average -100.93 bps.  In April, reducing short allocation does not
rescue the Huber arm because both conditional side economics deteriorate.
The failure is therefore not just score compression or a fixed side-mix
problem: the conditional payoff attached to the direct score changes regime.

Next conversion work is constrained accordingly:

1. stop expanding mapping HPO on these reused months;
2. retain residual expected EV as the ranking incumbent;
3. materialise a causal pre-entry transition panel around the March 20 break
   and attribute direct-minus-residual economics by state, side, week,
   score/cutoff context and adverse/capture components;
4. test the direct head only as a bounded conditional expert whose trust is
   learned from strictly earlier or held-out regime evidence, with residual
   as the fail-closed route;
5. require positive confirmation economics, latest-period coverage,
   both-side safety and week/month tail gates before any portfolio replay;
6. keep timing, MAE, target-price and wait actions in their separate action
   layer.

The artifact was reproduced from scratch with all nine parquet hashes
identical to the sealed manifest.  The complete focused lineage suite passes
24/24 tests, and runner/source hashes match the manifest.

#### March-20 causal regime diagnosis: boundary detectable, direct trust not learnable

The identical-row causal context panel is sealed at
`data_perp/artifacts/marapr2025_direct_residual_regime_trust_diagnostic_20260730_v1`.
All 136,074 bridge identities join exactly at candidate signal `__ts__`—not
execution time—to nine authoritative BOCPD regime fields, eight authoritative
transition fields and trajectory probability/entropy/margin plus availability.
Every March/April row is blocked OOF for calendar era 2025: the regime and
transition fits end before 2025-01-01 with labels resolved by
2024-12-31 23:00Z, and the trajectory model holds out 2025 using 2022--24.
OOD values, state/destination/cluster IDs, post-entry paths and action fields
are excluded.

The selection attribution sharpens the failure:

| Period | Shared book | Direct-only net | Residual-only net | Direct-minus-residual |
|---|---:|---:|---:|---:|
| Mar 3--19 | 470 rows / 6.4% Jaccard | +12.85 bps | -21.85 bps | +30.54 bps |
| Mar 20--31 | 404 / 7.9% | **-100.63** | -58.47 | **-36.00** |
| April | 1,263 / 10.0% | **-107.96** | -23.67 | **-68.92** |

The shared rows also change from +80.52 bps before March 20 to -71.91 bps
afterward, so there is a broad payoff deterioration.  But the direct-only
collapse proves that most incremental damage is source-specific selection,
not merely a market-wide cost shift.  Direct calendar concentration also
worsens much more than residual.

The causal context does register a state change.  Relative to its fixed
pre-March reference, post-March-20 trajectory and BOCPD onset probabilities
fall while run length rises: the post-break environment looks more stable,
not more transition-active.  The fixed grouped-OOF learnability study is
sealed at
`marapr2025_direct_residual_regime_break_learnability_20260730_v1`:

| Task | Regime | Transition | Trajectory | Combined |
|---|---:|---:|---:|---:|
| Recognise Mar 20 boundary, AUC | 0.569 | 0.571 | **0.700** | 0.591 |
| Long direct-over-residual trust, AUC | **0.536** | 0.454 | 0.527 | 0.530 |
| Short direct-over-residual trust, AUC | 0.505 | 0.472 | 0.523 | **0.532** |

Trajectory boundary recognition is stable across its UTC-day-group folds
(AUC 0.676--0.894).  Economic trust is not: its best AUC is only 0.536 long
and 0.532 short under seven-day-group OOF.  The highest predicted short-trust
decile still has negative direct advantage for every arm.  The combined
context also underperforms trajectory alone for boundary recognition, so
adding correlated weak probabilities is harmful.

This rules out a direct-q25 gate based only on current regime probabilities.
The correct architecture and next materialisation are now:

1. retain residual expected EV as the fail-closed ranking incumbent;
2. keep the sparse trajectory detector as a **transition-state feature**, not
   an economic trust score;
3. reconstruct an older identical-row, side-local OOF direct-q25 and residual
   score ledger; without it, March/April can diagnose but cannot train a
   promotable trust head;
4. materialise inference-parity causal market mechanics—BTC/alt resilience,
   breadth and correlation structure, funding/OI change, spread/liquidity,
   compression and recent short damage—before using them; the existing
   inventory alone does not authorize a join;
5. train separate side-local conditional-payoff heads for direct-only
   incremental capture, cost-clearing probability and adverse loss, plus a
   bounded direct-versus-residual trust output;
6. interact those heads with score-rank conflict, cutoff margin and
   candidate-group geometry, then require grouped OOF stability and a later
   untouched common direct/residual cohort;
7. keep timing, MAE, target-price and wait actions in the separate action
   layer and keep portfolio replay disabled.

Both new artifacts reproduce exactly from clean temporary runs.  The
expanded focused lineage suite now passes 33/33 tests.

#### Fresh matched-trust readiness: direct q25 exists; residual must be rebuilt

The follow-on lineage audit changes the next materialisation precisely.
`mayjul_exact_direct_q25_causal_mapping_20260730_v1/four_layer_common_rows.parquet`
contains 125,551 unique exact rows from 2026-05-01 23:00 through
2026-07-10 23:00 UTC, with 61,125 May, 49,259 June and 15,167 July rows.
Its direct score is the true exact-H12 q25 output, bit-identical to the
challenger's `q25_net_bps`, with causal fold cutoffs:

| Fold | Rows | Fit cutoff | Maximum training-label resolution |
|---|---:|---:|---:|
| recent May | 61,125 | 2026-05-01 | 2025-05-01 12:00Z |
| recent June | 49,259 | 2026-06-01 | 2026-05-31 23:00Z |
| recent July | 15,167 | 2026-07-01 | 2026-06-30 23:00Z |

Do **not** retrain or remap this direct score.  Its per-fold binaries were not
persisted, so it is recipe/output/frozen-state bound rather than binary
replayable, but its row-level OOF chronology and exact score identity are
sealed.

The existing residual in that panel is not a valid same-target comparator.
Although side-local and prior-resolved OOF, it was trained on the legacy
fixed-1%-cost **24-hour** residual target; its label resolves 12 hours later
than the panel's exact current-policy H12 endpoint.  Using it for a trust
policy would conflate model trust with target-horizon mismatch.

The next executable stage is therefore:

1. retain all 127,777 raw-score May--July identities and the frozen true q25
   output; the 125,551 subset is required only if a mapped-q25 coordinate is
   explicitly tested;
2. rebuild **only** residual expected EV, per side, on the exact
   current-policy H12 net target with the same May/June/July fold cutoffs;
3. require row-level `max_train_label_end < fit_cutoff <= decision`, and bind
   feature list, target hash, training identities, fold and score outputs;
4. attach authoritative regime/transition context at signal `__ts__`;
   trajectory missingness must use the already frozen neutral-fill plus
   availability contract, never row dropping or an ex-post fill;
5. materialise the exact common panel and repeat side-local direct-versus-
   residual trust learnability and month-transfer tests;
6. keep the result diagnostic: these 2026 months have been reused in the
   broader research programme and cannot alone authorize promotion.

No usable fresh 2023--25 same-target population already exists.  The
2023--24 ledgers lack true q25; late-2024 is hourly/non-poolable; and the
January--February exact-1m direct comparator is invalidated.  Reconstructing
older q25 remains a later expansion, not a shortcut around the H12 residual
rebuild.

### 2026-07-30 — exact-H12 rebuild, July context, and policy-parity mapping

The required exact-H12 residual rebuild is complete.  The authoritative
artifact is `exact_h12_side_local_residual_oof_20260730_v2`; v1 is
superseded for model selection because it selected each side independently
while the production decision is a pooled-global book.  V2 selects the
long/short pair jointly on April weekly pooled-global top-10 stability, using
only February--March 2025 resolved training labels and April validation, then
freezes the pair for May--July 2026 OOF scoring.  Row identity, label
resolution, costs, folds, model/map contracts and output hashes reproduce.

The selected long arm is `legacy_capacity_64` with residual blend 0.75.  The
selected short blend is 0.0: the short residual correction is rejected and
the short route remains the mapped base.  Even the development winner is
negative (-39.56 bps April top-10; -41.24 bps mean week; -100.59 bps worst
week).  May--July exact-residual global top-10 is respectively -67.88,
-104.26 and -148.98 bps.  Relative to the exact mapped base it adds +8.48,
-40.42 and -0.27 bps.  This is diagnostic evidence, not a promotable model.

The IC/EV discrepancy is now explicitly decomposed.  The previously quoted
0.155/0.162/0.226 February--April IC is long base score versus the legacy
native 24-hour alpha target.  Against exact H12 net execution, long IC is
only 0.090/0.093/0.143.  Raw-base pooled-global top-10 gross/cost/net is
+49.38/100.25/-50.87 bps in February, +17.05/100.09/-83.03 in March, and
+41.86/100.21/-58.35 in April.  April's better rank relationship therefore
partly improves gross selection relative to March, but no month clears the
cost hurdle.  The March deterioration is mainly lower positive payoff and
prevalence, not candidate-cell composition; April direct-EV failure is also
a short-side allocation failure.

Causal July context is useful for state recognition, not routing.  Frozen
regime/transition/trajectory fields identify July with grouped-OOF AUC 0.792
in combination, but residual-over-base trust AUC is only 0.435 long and
0.521 short; direct-over-residual trust is 0.476/0.577.  July differs most in
trajectory availability, transition onset/run-length, trajectory margin and
model entropy.  These fields may support causal features, sample weights and
later regime specialists, but must not become a gate without independent
economic-trust evidence.

The canonical 21-day recent-EV mapping was then applied before pooled-global
top-k, with exact-H12 label resolution before every snapshot.  It does not
repair the model: mapped exact-residual top-10 is -72.82/-108.87/-180.00 bps
in May/June/July, and July becomes 99.54% short.  Mapping can change
cross-side allocation and create plateaus; it is not a substitute for a
cost-clearing target.

Next gates:

1. stop generic calibrator, probability-map and portfolio-replay sweeps;
2. build side-specific heads for `P(gross > realised cost)`, clean favourable
   capture magnitude and adverse-loss severity, with peak-MFE/future-slope
   support and exact inference-parity features;
3. use regime-transition context as causal state information and test
   pre-registered weighting/specialist mechanisms only with older identical-
   target ledgers or a new untouched cohort;
4. if residual learning is revisited, restore the full standard selection
   stack (univariate, Relief, MDA and archetype-aware checks), jointly tune
   the pooled-global policy objective, and individually ablate candidate-
   context fields;
5. retain the existing strict production feature-parity gate; before any
   future exact-H12 package can enter it, final-refit and replay its explicit
   complete-case or named-native-missing policy, then bind ordered feature,
   source, base-score and map hashes.  Keep timing, MAE, target-price and wait
   actions in their separate action layer.

### 2026-07-30 — older common exact-H12 ledger under the current map

`marapr2025_exact_h12_current_mapping_20260730_v1` now supplies the required
older common score surface: 140,682 identical March--April 2025 candidates
with exact-1m current-policy H12 gross/cost/net labels, canonical base OOF,
strict residual OOF, and direct-q25 OOF. It reconstructs direct-q25 score
availability and `old_march`/`old_april` fold chronology from the bound source
recipe, feature dataset, frozen state and model. All source hashes bind; no
model is refit. The current map uses 21 UTC days, `label_end < snapshot`, a
500-row pooled minimum and 500-row side shrinkage. It keeps 2,208 warm-up
rows unavailable rather than backfilling them.

This makes mapping effects comparable across the two eras, but not positive.
At pooled-global top-10, March residual changes from -27.14 raw to -13.48 bps
mapped, then April worsens from -24.32 to -31.38 bps. Direct q25 changes from
-20.18 to -42.04 bps in March and from -93.24 to -77.38 in April. Base is
-65.61/-33.94 raw and -56.32/-43.12 mapped. All books remain negative;
mapping shifts side allocation materially and no replay is authorized.

The ledger is diagnostic only: the direct head used March--April during its
historical selection. Its value is to support an older identical-ID,
candidate-context and component-head experiment—not to justify a mapper
search or promotion.

### 2026-07-30 — older common exact-H12 ledger under the current map

`marapr2025_exact_h12_current_mapping_20260730_v1` now supplies the required
older common score surface: 140,682 identical March--April 2025 candidates
with exact-1m current-policy H12 gross/cost/net labels, canonical base OOF,
strict residual OOF, and direct-q25 OOF.  It reconstructs direct-q25 score
availability and `old_march`/`old_april` fold chronology from the bound source
recipe, feature dataset, frozen state and model.  All source hashes bind; no
model is refit.  The current map uses 21 UTC days, `label_end < snapshot`, a
500-row pooled minimum and 500-row side shrinkage.  It keeps 2,208 warm-up
rows unavailable rather than backfilling them.

This makes mapping effects comparable across the two eras, but not positive.
At pooled-global top-10, March residual changes from -27.14 raw to -13.48 bps
mapped, then April worsens from -24.32 to -31.38 bps.  Direct q25 changes from
-20.18 to -42.04 bps in March and from -93.24 to -77.38 in April.  Base is
-65.61/-33.94 raw and -56.32/-43.12 mapped.  All books remain negative;
mapping shifts side allocation materially and no replay is authorized.

The ledger is diagnostic only: the direct head used March--April during its
historical selection.  Its value is to support an older identical-ID,
candidate-context and component-head experiment—not to justify a mapper
search or promotion.

### 2026-07-30 — long-split cost-clearing base/residual target ablation: reject retargeting adapter

`long_base_residual_h12_target_ablation_20260730_v3` evaluates the requested
non-walk-forward chronology: 12 months of base fitting (2023-04--2024-03),
eight months of frozen base predictions (2024-04--11), four months of meta
fit (2024-04--07), and four untouched meta prediction months (2024-08--11).
All labels are exact H12 gross/cost/net; the causal map requires the
materialized label-availability timestamp before its UTC-day snapshot, which
is stricter than endpoint-only eligibility.  The decision is one pooled
global top-k across both sides and all timestamps.  Monthly results only
decompose this fixed selected book; they never rerank locally.

The three base targets all enforce cost clearance (0 bps, 25 bps, and
cost-clearing upside).  The four meta targets are direct net, net conversion
residual, globally top-decile-weighted net residual, and a 25-bps
cost-clearance classifier.  The latter is the most policy-focused arm.
Controls replay frozen base alpha and frozen residual expected EV under the
identical causal map and global-book evaluator.

At the global top 10%, the frozen residual control is -106.01 bps.  The best
new configuration—25-bps base opportunity plus 25-bps meta classifier—is
-178.87 bps; frozen base alpha is -215.92 bps.  All twelve new configurations
are negative and worse than the residual control.  The frozen residual book
is negative in every selected month (August--November: -161.30, -111.49,
-136.26, -35.95 bps).  No target, map, or portfolio promotion follows.

This is an honest **base retargeting adapter** result, not a raw-base-refit
claim: the continuous historical panel has frozen base alpha plus causal
regime/transition context, not a compatible raw feature matrix over the full
20-month span.  It rules out simply replacing the native alpha target with a
cost-clearing direct target.  The next prerequisite is to materialize a
feature-complete, exact-H12/current-policy historical raw-base panel.  On that
panel, test a decomposed architecture: base learns clean cost-clearing
opportunity; residual learns conversion into realised H12 policy net; separate
adverse, timeout and capture heads remain supporting components rather than
being collapsed into a direct-net target.

### 2026-07-30 — raw-feature panel materialised; not a full-universe refit substrate

`long_exact_h12_raw_base_panel_20260730_v2` materialises 272,686 April 2023
through November 2024 candidate rows, with 380 strict observable raw features
and exact H12 current-policy gross/cost/net plus path support labels.  The
calendar is complete and exactly supports the 12-month base / 8-month base-OOS
/ 4+4-month residual split.  Label endpoint and availability are bound per
row.  Frozen base-attribution, threshold/selection, DAE, GMM and regime-source
fields are physically excluded from the raw feature contract.  The predecessor
v1 is invalid for model input because it retained those unsupported fields.

This is still **not** a promotable or full-universe base-refit substrate:

1. historical labels exist only for old selected-top30/monitor candidates,
   selected by a frozen proxy score, so the population is candidate-conditioned;
2. the historical labels are current-spread counterfactuals and the source
   manifests explicitly set `execution_parity_claim=false`;
3. historical L2 and bit-exact pre-2025 path-geometry parity are unavailable.

It may support a strictly labelled research ablation of target/component
learnability on that conditioned population.  It must not be presented as a
production base result, used for promotion, or used to infer a full-universe
base-selection policy.  A genuine raw-base refit still requires exact H12
labels for every base candidate plus a bit-exact historical execution geometry
and economics contract.

### 2026-07-30 — raw base + residual cost-clearing target ablation: no promotion

`long_raw_base_residual_h12_ablation_20260730_v1` is the requested genuine
raw-feature research refit on the preceding conditioned panel.  It fits the
base for 12 months (2023-04--2024-03), produces eight frozen OOS months,
fits the residual on the first four (2024-04--07), and evaluates the untouched
final four (2024-08--11).  Base selection is side-local and base-train-only:
each target/side retains the top 64 gain-screened observable raw fields.
Residual inputs are that frozen side-local feature set plus frozen base
expected net.  Base target-to-net calibration and residual map-history scores
are chronological blocked OOF.  Walk-forward validation was intentionally
not required for this diagnostic.

Every target is net of row cost with a further +25 bp post-cost hurdle.  The
base arms are: soft net-hurdle opportunity, adverse/MAE-risk-penalised net,
and timely clean net.  The residual arms are: net residual, globally
top-decile-weighted residual, soft post-cost-clear probability, and clean
globally-top-decile-weighted residual.  The global tail is calculated over
the whole training book across sides and timestamps; it is never a
per-timestamp or per-side rank.  Final selection is likewise one pooled
global top-k across all timestamps and both sides after a 21-day causal pooled
isotonic map.  Side/month tables are only membership decompositions.

At the global top 10%, the best arm is soft net-hurdle base plus clean
global-tail-weighted residual: **-129.54 bps/trade** (7,520 selected of
75,200), versus **-175.10 bps** for the frozen-base-score control.  It is an
improvement in this bounded diagnostic, but remains economically negative.
All top-k cutoffs are negative: its top 1/5/10/20% values are -46.25,
-122.95, -129.54 and -146.17 bps.  The immediate cause is short allocation:
the selected long subset is -34.31 bps while the selected short subset is
-155.26 bps.  No result is eligible for model, mapping, policy, or portfolio
promotion.  This does support the narrower hypothesis that cost-clearing,
clean-tail residual emphasis carries more useful ranking signal than direct
soft-clear probability on this population; it does not establish a usable
trading policy.

### 2026-08-01 — strict semantic OOF and native continuation-information audit

The semantic-support infrastructure is complete as a diagnostic: 132,930 OOF
predictions across 27 heads, with the chronology
`base_train=2023-04..2024-03`, `meta_train=2024-04..2024-07`, and
`meta_oos=2024-08..2024-11`. All conditional heads score every candidate and
apply their validity masks only to the target metric, avoiding the earlier
selection-bias failure mode. The strongest learnability signals are
opportunity reach AUC 0.719 (rank IC 0.297), conditional peak-MFE rank IC
0.626, adverse AUC 0.707, persistence rank IC 0.371, and early opportunity
hazard AUC 0.770. These are learnability results, not entry economics.

The pooled-global top-k economic diagnostic is negative for every usable
composition C0--C4. At top-10%, development C0--C4 are respectively
-129.95, -163.95, -203.26, -204.82 and -177.45 bps; final OOS values are
-117.91, -139.88, -174.48, -171.36 and -183.27 bps. The latest final month is
also negative. Two-hundred UTC-day bootstrap replicates put the probability of
a positive result at zero for every control. Positive O1--O3 oracle books use
future outcomes and are retained only to show the opportunity ceiling. The
root-cause disposition is therefore `economic_translation_and_composition_bottleneck`,
not a head-lineage or learnability failure.

To test the prescribed next step, a strict native-L2 continuation generator
was added. It accepts only `kraken_futures_l2_snapshot` rows and emits causal
spread/depth/imbalance/shape and snapshot-change fields. The available exact
native cohort is only 6,928 rows across 73 products from July 11--23, 2026;
5,981,302 `local_ohlcv_summary` rows were excluded as proxies. The sidecar is
research-only and is not appended to production base/meta keys. The raw native
trade-count/notional/flow fields are all zero, so no aggressor-flow feature is
claimed; bounded prior-snapshot change fields are available on 6,144 rows.

A label-free candidate-overlap audit performs an exact-product backward as-of
join with no future fill. At a two-hour staleness bound, only 195 of 311,843
canonical handoff rows (0.063%; 32 products, three days) match. The July
20--23 retrospective bridge and the May--July exact-H12 residual/strict-
forward panels have zero overlap. Consequently no native-L2 OOF model,
feature selection, HPO, or economic result is admissible yet. The next
concrete prerequisite is longer timestamped native-L2 history (or an
equivalent factual native feed), followed by the same as-of join and a strict
global-top-k `retain | clear` OOF test.

### 2026-08-01 — native-L2 backfill readiness contract

`scripts/audit_native_l2_backfill_readiness.py` now materializes a source
inventory before any backfill or model work. It scans both local
orderbook-hourly roots using only parquet metadata plus source, product-key,
and timestamp columns; labels, scores, costs, and portfolio fields are not
loaded. The authoritative artifact is
`data_perp/artifacts/native_l2_backfill_readiness_20260801_v1/`.

The inventory contains 568 files and 10,373,441 rows: 6,928 exact
`kraken_futures_l2_snapshot` rows in 73 product files and 10,366,513 explicit
`local_ohlcv_summary` proxy rows. Exact native coverage starts at
2026-07-11T11:00:00Z, while the declared candidate panels begin on
2026-04-01. The full-window gate therefore fails closed:
`historical_native_backfill_required=true`, `candidate_joined=false`,
`model_fitted=false`, and `promotion_eligible=false`.

This closes the local-source search for the current roots without treating
proxy OHLCV as native depth. A backfill is admissible only when it preserves
exact product identity, factual snapshot/publication timing, and the existing
two-hour bounded as-of contract. After backfill, rerun the native sidecar and
candidate-overlap audit before reading labels or fitting any model.

### 2026-08-01 — dense raw native-L2 history discovered and rechecked

The broader local scan found a second native source under
`data_perp/exchanges/krakenfutures/spread_snapshots/orderbook_history/`.
It is raw per-level Kraken L2, not an OHLCV proxy. A vectorized aggregator now
materializes `native_l2_continuation_sidecar_20260801_v3` from the ten
canonical daily history files, retaining `observed_ts` as availability and
deduplicating exact product/snapshot keys.

The v3 sidecar contains 51,778 aggregated native snapshots across 303 exact
products from 2026-07-11 through 2026-07-23; 50,334 rows have a bounded prior
snapshot within two hours (97.21%). The corrected v3 overlap audit raises
coverage to 10,282/311,843 (3.297%) in the canonical handoff and
3,300/5,760 (57.292%) in the July 20--23 retrospective bridge. The exact-H12
May--July and A-grade strict-forward panels remain at zero overlap, so this is
still a source-readiness result, not an OOF/model result.

The v3 report is authoritative for current-period native coverage;
`native_l2_continuation_sidecar_20260801_v2` and its v2 overlap audit remain
valid but lower-density hourly-source diagnostics. Historical backfill before
2026-07-11 is still required before the roadmap can enter strict OOF economic
testing.

### 2026-08-01 — native-L2 readiness daily-coverage recheck

The fail-closed inventory was rerun over the complete local `data_perp` tree,
using the corrected v3 overlap manifest. It scanned 71,135 parquet files and
327,133,322 rows without loading labels, scores, costs, or portfolio fields.
The exact native source contributes 2,865,522 tagged rows, but only on ten UTC
calendar days: 2026-07-11--16, 2026-07-18, and 2026-07-21--23. The observed
native calendar gaps are 2026-07-17, 2026-07-19, and 2026-07-20; the declared
candidate window still begins on 2026-04-01. The authoritative readiness
artifact is `native_l2_backfill_readiness_20260801_v3`.

The daily coverage accounting strengthens, rather than changes, the decision:
`historical_native_backfill_required=true`, `candidate_joined=false`,
`model_fitted=false`, and `promotion_eligible=false`. No model, HPO, label,
or economic result may be derived from the partial native cohort.

### 2026-08-01 — current-run stop and registry reconciliation

The current-run stop audit is sealed at
`data_perp/artifacts/current_run_stop_audit_20260801_v1/`. An escalated
process-table check found zero active Ares training, collector, materializer,
or audit processes. The only apparently live registry PID (1026) had been
reused by macOS `imagent` (working directory `/`, parent PID 1), not the
registered Ares collector. The safety wrapper correctly sent no signal to
that unrelated process, and the registry entry is marked
`stale_pid_reuse`. No new roadmap run was started.

The subsequent registry cleanup marked all other dead entries `exited`; the
active job listing is empty. PID 1026 remains stale and was never signalled.

`scripts/codex_job_control.py list --active` now requires both a live PID and
registry status `running`, so stale PID reuse cannot be presented as an active
Ares run. This is an observability and safety repair only; it does not change
any model, target, feature, policy, or promotion decision.

### 2026-08-01 — native-L2 backfill request manifest

The next data-acquisition step is now explicit in
`data_perp/artifacts/native_l2_backfill_request_20260801_v1/`. The request
builder reads only candidate product identity/time columns and the native
sidecar's product/timestamp columns; labels, scores, costs, portfolio fields,
and model outputs are not loaded.

The manifest covers 336 products and 25,343 candidate product/day pairs from
2026-04-01 through 2026-07-23. Only 952 pairs currently have a native
snapshot, leaving 24,391 pairs to backfill. The provider contract remains
exact product identity, factual observed/publication timestamps, the native
source allow-list, and the two-hour as-of bound; OHLCV proxies are excluded.
This is a request/readiness artifact only: `model_fitted=false` and
`promotion_eligible=false`.

### 2026-08-01 — canonical target–feature–execution alignment audit

The cached target–feature–execution roadmap is now materialized as one
fail-closed audit pack at
`data_perp/artifacts/target_alignment/alignment_audit_20260801_v2/`.
It joins the exact-H12 target contract, materialized primary/supportive
labels, layer-specific feature eligibility, chronological fold/OOF manifests,
the exact target ablation, and supportive-label global-tail economics without
retraining or silently promoting a research arm.

The pack records 55 passed and 2 failed checks. The passed checks prove exact
12-hour timing, label availability after the horizon, causal feature cutoff,
one-time gross-minus-row-cost accounting, frozen policy/cost IDs, unique
candidate identity, chronological folds, aggregate OOF fit-end ordering, and
one pooled global top-k selection. The original v2 source pack has now been
superseded for research by the versioned canonical v3 target pack, whose
supportive labels and dictionary contain all explicit valid/condition-met/
censored/support-count fields. The remaining failed checks are economic: the
best supportive global top-10% net is -113.44 bps and the best exact-H12
target-ablation global top-10% net is -104.05 bps. The canonical execution
feature manifest binds 30 candidate-level OOF score features with fit-end and
generated-time lineage. Native-L2
history is separately still blocked by 24,391 missing product/day pairs in
`native_l2_backfill_request_20260801_v1`.

The resulting status is `FAIL_CLOSED_RESEARCH_ONLY` and
`promotion_eligible=false`. Timing, MAE, target-price, wait, and portfolio
actions remain separate layers.

### 2026-08-01 — economic headroom and ranking bottleneck

The remaining economic gate is now separated into an oracle ceiling, model
ranking, and cost sensitivity in
`data_perp/artifacts/exact_h12_economic_headroom_diagnostic_20260801_v1/`.
The oracle pooled global top-10% has **+468.27 bps gross**, **102.34 bps
cost**, and **+365.93 bps net**, so the fixed policy/cost contract does contain
an economically viable tail. The best model top-10% is the frozen
`CONTROL_base_opportunity`: **-4.07 bps gross**, **99.98 bps cost**, and
**-104.05 bps net**. Its top-1% gross is **+91.67 bps**, showing that a small
positive tail exists but is not ranked broadly enough for the 10% book.

The zero-cost counterfactual remains **-4.07 bps** for the best model top-10%,
so fee reduction or hurdle changes alone cannot repair the current result. The
binding issue is ranking/feature/label conversion, not absence of economic
headroom. The diagnostic now uses the same selection contract as the target
ablation: `calibrated_expected_net_bps`, descending stable row-order ties, and
one pooled global top-10% book. Its reproduction check is exact; the previous
raw-score ranking view was not authoritative and is superseded.

This does not reopen promotion. The alignment audit remains 55/57 checks
passed, with the exact-H12 and supportive economic checks failed; native-L2
history is also incomplete for 24,391 April–July product/day requirements.
The next experiment should target ranking conversion (clean/competing-risk
probabilities, cost-aware magnitude, and genuinely causal continuation
features) before any auxiliary/action/portfolio layer is added.

### 2026-08-01 — requirement-level audit

The roadmap is now checked requirement-by-requirement in
`data_perp/artifacts/updated_roadmap_requirement_audit_20260801_v1/` rather
than inferred from the aggregate alignment score. It records **12 PASS**,
**5 FAIL**, and **1 BLOCKED_EXTERNAL** items. Contract, target separation,
support metadata, feature eligibility, strict candidate-level OOF lineage,
pooled-global policy, ablation coverage, and reproducible manifests pass.

The failed acceptance gates are explicit: no positive pooled top-10% net arm,
negative latest month, negative long and short books, no positive paired
bootstrap lower bound, and no incremental value from supportive OOF scores.
The separate external continuation prerequisite has 24,391 missing native-L2
symbol/day pairs. This matrix is evidence-only and does not launch another
run.
