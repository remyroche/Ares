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
| Pack-B directional base | Available, per-side manifest | Preserve; audit serialized side-local provenance |
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
  `decision_timestamp <= validation_start - 24 hours`. The final accepted
  Pack-B training label must resolve no later than the validation boundary.
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
  - frozen AE/GMM input order and serialized-state reuse;
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
- Regenerate four canonical Pack-B OOF folds using the locked half-open April,
  May, June, and July 1–11 signal windows. Freeze the recovered exact AE/GMM
  state and promoted 55-feature long / 37-feature short contracts; perform no
  new FS or HPO in this recovery run.
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

Gate R3:

- No fitted selector, parameter, model, prior, calibrator, or OOF outcome crosses sides.
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
- AE/GMM refit outside the frozen cycle contract;
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
| R3 Alpha/top-40 | IN PROGRESS | Alpha + Data/provenance | Saved seven-fold Pack-B models are historical-only; canonical four-fold regeneration required | Exact recovered AE/GMM state `6521f981…`; resource guards committed in `ac6a116305` |  |
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
5. Finish the four-fold DEC-09 Pack-B regeneration runner and smoke it under
   the fail-closed resource guard.
6. Run the canonical Pack-B OOF regeneration sequentially only if the measured
   memory preflight is safe, then derive top-40 per timestamp × side.
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
