# Stage-I MDA/HPO speedup implementation — 2026-08-03

## Implemented contract

- Long and short can run concurrently as two isolated processes, capped at
  four LightGBM threads each. A deterministic available-memory preflight falls
  back to sequential execution before either worker is launched.
- Each side has its own checkpoint root, log and immutable request hash.
  Changed inputs, schedules or commands reject stale resumes.
- HPO searches of at least nine trials use deterministic successive halving:
  every proposal receives cohort 1; the top `ceil(n/3)` receive cohorts 1–2;
  final survivors receive every HPO cohort. The winner is then regenerated on
  the original full strict-OOF fold plan.
- LightGBM training matrices reuse exact-key Dataset/bin objects across HPO
  trials. Reuse is permitted only for byte-identical features, rows, targets,
  weights, classifier mode and `max_bin`.
- Stage-I MDA retains one model seed. Every evaluable feature/group is tested
  in each of the three disjoint chronological cohorts. Within a cohort, only a
  harmful confidence interval wholly below zero or a useful interval wholly
  above the frozen phantom threshold may stop at the minimum repeat count.
  Null, borderline and unstable evidence receives the full repeat budget.
- Conclusively harmful correlated groups skip redundant individual member
  permutations. Members remain review-only until cross-cohort evidence exists.
- Future selector runs use `stage_i_tiered_round_pruning_v2`. Above 200 active
  features, one completed MDA round targets 70% survival. From 200 down to the
  hard coarse floor of 120, each completed round targets 80% survival. Covered
  examples are 250 -> 175, 190 -> 152, and 123 -> 120.
- Each selection checkpoint uses evidence aggregated across its chronological
  MDA folds, never a separate feature list per CV fold. While above 120, a
  checkpoint that continues removes at least five evidence-assessed,
  nonprotected, non-warm-up features; the persisted minimum is explicitly
  named `minimum_evidenced_removals_per_selection_checkpoint`. The quota is
  capped by distance to 120, so 123 -> 120 removes three. No fold-specific
  feature contract is created.
- Required handoff/trust fields, causal warm-up fields, and exact declared
  regime/archetype/support context retain their existing protection. Untested
  fields remain, target-specific zero-use evidence still requires all three
  chronological eras, and complete equal-evidence boundary ties are retained.
  If those gates make the applicable minimum impossible, the round persists a
  shortfall and stops unchanged; it never removes by name or declaration
  order. At or below 120, conservative CI hard-drop-only behavior resumes with
  no forced quota and no coarse removal below 120.

## Versioning and in-flight compatibility

New project runs default to selector schema
`stage_i_grouped_stability_mda_v6`, pruning schema
`stage_i_tiered_round_pruning_v2`, and bounded orchestrator schema v2. Base and
meta manifests persist the full pruning contract and its canonical SHA-256;
resume checks bind both. The orchestrator request hash also binds the new
selector and pruning schemas.

The selector process already running when this policy was added continues with
the v5/v1 Python objects imported at process start. A completed immutable
v5/orchestrator-v1 artifact is accepted only as a no-op resume after its old
request hash, child SHA-256, selector lineage, v5 schema, and exact v1 pruning
history validate. It is never reinterpreted as v6. Partial v5 checkpoints fail
closed and must be restarted as a new future run.

The direct base and meta CLIs run an all-requested-sides legacy preflight
before entering either per-side execution loop. If any requested cell is v5,
every requested side must already be present, complete, same-side, immutable
v5, and carry the exact shared v1 pruning history. Missing, incomplete, or
mixed v5/v6 roots reject atomically, so v5 compatibility can only be an
all-complete no-op. The orchestrator uses the same shared history validator,
including exact mapping and legacy mode checks.

## Feature-transition audit ledger

Every future iterative MDA run now emits a literal, side-scoped membership
ledger alongside its per-round MDA evidence:

- `iterative_mda_feature_transition_ledger.csv`
- `iterative_mda_feature_transition_ledger.json`

The ledger is produced once the selector chooses its final prefix and records
the complete candidate, retained and discarded lists at the initial
post-prescreen checkpoint, every iterative pruning transition, and final
prefix selection.  It carries explicit `selection_side` and
`selection_layer` (`base` or `meta`) metadata; it therefore does not require
reviewers to infer the cell from a shared parent directory.  The JSON retains
ordered feature lists for reproducible review; the CSV has one feature per
step with `retained`/`discarded` status for joins and aggregate audits.  Each
transition also links to its source `mda_feature_audit.csv`.

For declared single-side Stage-I cells, the completed side manifest also
surfaces both ledger paths and the transition count.  This makes the ledger
part of the immutable experiment receipt rather than an output that can only
be found by directory convention.

The same files are refreshed atomically after every completed pruning round.
Until final one-SE prefix selection, they state `selection_complete: false`,
leave `final_selected_features` empty, and expose the non-promotable
`checkpoint_active_features` instead.  Thus an interrupted run preserves its
literal history without claiming a provisional prefix is a frozen contract.

This is audit-only: a failure to write the ledger cannot change selection, but
is made explicit in the run metrics and log.  A focused correctness test
asserts that complete membership is preserved across transitions.  Processes
already running when this was introduced cannot retroactively produce the
ledger because their Python module was loaded before the implementation.

## Expected compute reduction

For the production 60-trial, four-fold HPO request, full-budget evaluation
would require 240 fold fits. The frozen `[1, 2, 4]` successive-halving schedule
requires at most 128 fold fits: **46.7% fewer HPO fold fits**, before cache
benefits. Winner full-OOF regeneration is unchanged.

## Microbenchmark

On a deterministic 20,000-row × 40-feature, four-trial, 120-tree multiclass
benchmark using four threads:

| Training path | Seconds |
|---|---:|
| Rebuild sklearn LightGBM Dataset/bins each fit | 2.353 |
| Reuse exact-key native LightGBM Dataset/bins | 2.046 |

Observed Dataset/bin reuse speedup: **1.15×**. Predictions were byte-identical
to the sklearn fit in the parity test. Production gains may differ with feature
count and row width.

## Verification

The updated focused Stage-I suite passed **144 tests** covering:

- HPO chronology, multiclass simplex and target adapters;
- deterministic halving schedule and full-evidence winner eligibility;
- cached-bin prediction parity;
- clear harmful/useful/borderline and warm-up adaptive-depth decisions;
- 70% above 200, 80% down to 120, and ≤120 conservative pruning;
- aggregated-round minimum-five enforcement, floor capping, protected and
  equal-evidence shortfalls, and no coarse transition below 120;
- immutable completed-v5 no-op compatibility and v6 manifest/resume hashing;
- direct base/meta all-side rejection for mixed, missing, and incomplete legacy
  roots, plus canonical pruning-contract hashes in selector test doubles;
- side-process isolation, four-thread cap, memory fallback and stale-resume
  rejection;
- existing grouped MDA and Stage-I feature-selection behavior.

No expensive experiment was started, and the active selector materialization
or its output was not modified.
