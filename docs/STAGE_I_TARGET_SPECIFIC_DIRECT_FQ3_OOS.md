# Target-specific R3/S/O → direct FQ3 OOS path

`stage_i_target_specific_oos.py` is the strict evaluation route for a
target-v2 winner with one of the base targets `soft_scalar_S`,
`cumulative_ordinal5_O`, or a completed frozen strict-OOF R3 base, and the
meta target `fold_quantile_residual3`.

Its purpose is deliberately narrower than the legacy R3/Huber production
runner: it evaluates a direct-base correctness meta model without letting a
pre-mapped bps score leak into the meta input.

## Semantics

For each side and chronological fold:

1. Fit the frozen S/O base model using only labels available before the fold,
   or reuse the immutable R3 strict-OOF ledger without refitting it.
2. Produce its direct score and full `base_state_p*` simplex.
   Scalar S is represented as `[1 - S, S]`; ordinal O retains all five states.
   R3 retains its three-state simplex and exact native
   `P(clear)-P(adverse)` score in `[-1,1]`.
3. Construct the FQ3 target from the **same-side direct base score**.  The
   realised exact-net outcome is transformed only with the fold-training
   empirical outcome CDF, expressed on the base score's native coordinate.
   `outcome_coordinate - base_direct_score` is split
   at training-only q33/q67 into `overestimating`, `approximately_right`, and
   `underestimating`.  It must satisfy `q33 < 0 <= q67`.
4. Score the FQ3 model with direct base score, states, and selected causal
   regime/context/trust fields.  The meta score is a bounded direct-score
   correction, not a bps correction.
5. After both sides are scored, fit the canonical prior-resolved 21-day
   pooled-parent / side-shrunk rank map separately for base and meta scores.
   This is the first and only conversion to common expected-net bps.
6. Apply the 50-bps admission floor, then rank once across the pooled sides.
   There is no timestamp-local ranking and no raw global ranking.

The model is thus answering “is the direct base score too optimistic, about
right, or too pessimistic?” rather than predicting an already-calibrated bps
residual.

## Required winner and source contracts

The route fails before a fit if any of the following is absent or mismatched:

- exact long and short v2 winner cells;
- S, O, or frozen strict-OOF R3 base plus FQ3 meta family, with identical geometry;
- evaluation target contracts bound to the input contract ledger, with target
  semantics identical to the selector's frozen training contracts, including
  winner gross and net economics (`gross - 100 bps == net`);
- `meta_target_semantics = same_side_direct_base_output_correctness_q33_v1`;
- `base_input_semantics = same_side_direct_base_output_without_bps_conversion_v1`;
- exact selector manifest byte hashes, selected feature contracts, and
  independent non-empty base and meta correlation-policy lineage;
- `base_raw_score`, all required `base_state_p*`, and all three direct trust
  fields in the selected meta contract;
- explicitly declared non-empty selected regime, context, and trust feature
  requirements in the meta target metadata;
- exact candidate/signal-time/symbol alignment of feature and contract files;
- close → +1h entry/decision → H12 label availability, and complete valid
  labels only;
- source `features.parquet` and `contract.parquet` SHA-256 values recorded in
  the source manifest.

The source manifest must additionally bind two content-hashed contracts:

- `causal_feature_role_contract` (`..._v1`) lists the approved raw causal
  base and meta fields separately.  A selected field outside its proper list
  fails closed.  Target/economic/path/map/state namespaces are forbidden;
  `base_raw_score`, `base_state_p*`, and direct trust fields are created only
  inside the strict evaluator and are rejected if a source parquet supplies
  them.
- `evaluation_month_contract` declares all 36 months from 2024-01 through
  2026-12 and marks each as source-available or as an explicit gap with a
  reason.  A declared available month with zero candidates, valid labels,
  base OOF, or meta OOF is a promotion failure unless a named zero-coverage
  exception and reason are present.

Base and meta feature selection may have different winning correlation
policies.  Both are hash-bound and reported independently; the evaluator does
not force them to be equal.

Selected meta fields also need at least 90% finite OOF coverage and variation
over scored rows.  A field that is absent, constant, or mostly null fails
closed rather than being filled or silently dropped.

The existing scalar-S base-selector writer now emits `base_state_p0` and
`base_state_p1` from the same strict scalar OOF output.  This is a future-run
contract change only; no existing artifact is rewritten.

## Materializing input safely

Use the model-free materializer rather than assembling the files by hand:

```bash
python3 scripts/materialize_stage_i_target_specific_inputs.py \
  --selector-dir data_perp/artifacts/stage_i_selector_sample_20260803_v5 \
  --base-selector-dir /immutable/base_selector \
  --meta-selector-dir /immutable/direct_fq3_meta_selector \
  --winner-bundle /immutable/stage_i_adapter_winner.json \
  --target-winner-dir /immutable/scalar_or_ordinal_winner_bundle \
  --output-dir /immutable/target_specific_inputs
```

`--target-winner-dir` is required for scalar/ordinal bases. It is omitted for
the frozen R3 control, where the materializer verifies and references each
side's immutable `selector_base_oof.parquet`. Materialization performs no
model fit, scoring, target tuning, or map fitting.

The selector target contract and evaluation target contract are intentionally
different content contracts. The first binds the exact rows used for
selection/HPO. The second binds the exact rows in `contract.parquet`. The
evaluator requires their family, layer, name, geometry, columns, weights, and
metadata to be identical, while requiring row-dependent hashes to be freshly
bound. A genuine later OOS population cannot reproduce training-row hashes.

For each side the materializer:

- joins features and labels by candidate/timestamp/symbol identity, preserving
  the authoritative target order;
- writes only the union of selected raw base and meta fields;
- proves those fields came from the proper config-derived layer universe;
- rejects future, label, path, economic, map, and generated base-state fields
  from the source feature file;
- binds target validity, entry timing, winner-geometry gross/net economics,
  the single 100-bps cost application, and both weight vectors;
- emits an availability/reason entry for every month from 2024-01 through
  2026-12. Historical, intervening, later, and future gaps are explicit rather
  than being presented as successful zero-row coverage;
- records byte hashes for emitted Parquets and every selector, winner,
  source-feature, and optional frozen-R3 dependency.

Run the evaluator with `--preflight` immediately afterward. It reopens the
published files and independently verifies all contracts without fitting a
model.

## CLI

```bash
python3 scripts/run_stage_i_target_specific_oos.py \
  --winner-bundle /path/winner.json \
  --input-root /path/target_specific_inputs \
  --base-selector-dir /path/base_selection \
  --meta-selector-dir /path/meta_selection \
  --preflight
```

Replace `--preflight` with `--output-dir /new/immutable/artifact` to execute.
The input root must have the following per-side files:

```text
long/features.parquet
long/contract.parquet
long/manifest.json
short/features.parquet
short/contract.parquet
short/manifest.json
```

Each source manifest names `base_target_column`, `meta_target_column`, and an
`artifact_sha256` mapping containing the byte hashes of the two parquet files.
For R3 it additionally names `frozen_base_oof_path`; the referenced file hash
must equal `selector_base_oof_sha256` in the byte-bound completed selector
manifest, whose regenerated-fold audit must be strictly prior-resolved.
The CLI never overwrites an output directory.

## Selector/HPO migration boundary

`scripts/run_stage_i_adapter_meta_feature_selection.py` is now fail-closed by
default because it implements the older mapped-expected-net residual target.
It can be reopened only with
`--allow-legacy-premapped-residual-control`, which labels its output as a
non-promotable historical control. The winner-bundle builder independently
rejects that semantics and any selected
`prequential_base_expected_net_bps` feature.

The reusable `fit_direct_fq3_estimator(...)` and
`direct_fq3_selector_fit_context(...)` boundaries fit outcome percentiles,
terciles, and class locations from each internal training slice only. Generic
Stage-I MDA/HPO now has a separate native-score offset contract: it accepts
exact net as economic supervision and native same-side `base_raw_score` as the
ranking offset, while rejecting bps offsets and pre-mapped expected-net fields.
`base_raw_score` and the state/trust handoff are protected from removal; target
labels are derived before any validation permutation and only from each
training slice.

Use the bounded side orchestrator for promotable meta selection:

```bash
python3 scripts/run_stage_i_sides_bounded.py \
  --layer meta --meta-mode direct_fq3 \
  --selector-dir /path/selector_sample \
  --base-selection-dir /path/completed_base_selection \
  --output-dir /new/direct_fq3_meta_selection \
  --required-regime-feature REGIME_FIELD \
  --required-context-feature CONTEXT_FIELD
```

The default meta training stream contains every finite base-OOF row, one row
per candidate. MDA still optimises side-global top-10 economics, never a
per-timestamp tail. An optional predeclared `--base-candidate-fraction` uses a
stable side-global base-score membership gate, but the default is `1.0` to
match the per-row meta requirement. Long and short remain isolated bounded
workers with the established memory-based sequential fallback and immutable
resume checks. Final evaluation remains pooled-global only after causal
side-local common-bps mapping.

## Outputs

- `strict_oof_predictions.parquet`: the 2024--2026 evaluation ledger with raw
  direct base/meta scores, all base states, FQ3 states, and output-only 21-day
  mapping coordinates.  `full_history_strict_oof_predictions.parquet` is kept
  separately only to document the earlier strict fold/map warm-up.
- `fold_provenance.parquet`: strict prior-resolved fold proof and q33/q67
  correctness states.
- `per_side_month_base_meta_metrics.parquet`: raw side/month diagnostics plus
  per-side/month attribution of the mapped pooled-global books, with and
  without the 21-day admission floor.
- `worst_period_diagnostics.parquet`: worst month for each layer, selection
  mode, and global tail.
- `joint_stack_promotion_score.parquet`: the candidate gate from the
  reconstructed/meta stack only; it is never derived from base-only metrics.
- `2024_2026_side_month_coverage_audit.parquet`: all 36 months × both sides ×
  base/meta, with candidate, valid, strict-OOF, mapped, admitted counts and
  the declared source-gap contract.
- `base_causal_21d_map_audit.parquet` and
  `meta_causal_21d_map_audit.parquet`: exact prior-resolved map support.
- `manifest.json`: winner, source, target, correlation, timing, mapping, and
  file-hash lineage.

Raw score summaries are intentionally unranked diagnostics.  Every decision
metric is based on the mapped common-bps score and one pooled-global book.

## Promotion and finalist comparison

Base-only economics are diagnostic only: they cannot promote or reject a
target.  The evaluator writes a single-stack joint gate candidate, and
`compare_target_specific_finalists(...)` compares R3/S/O finalist ledgers
only when their full 2024--26 candidate identities, decision timestamps,
target-valid flags, and strict-meta availability are identical.  It
ranks only the reconstructed/meta 21-day common-bps score, reports pooled
top-1/5/10/20 and worst-month top-10 economics, and emits one frozen
joint-stack promotion score/gate.  It refuses to intersect non-identical rows
or substitute a base score.

Use `scripts/compare_stage_i_target_specific_finalists.py` with repeated
`--finalist NAME=/immutable/artifact` arguments to materialise that comparison
without retraining any finalist.  The comparator verifies the manifest-recorded
SHA-256 values for both the strict OOF ledger and 2024--26 coverage audit,
requires a complete causal 21-day common-bps mapping/admission contract, and
requires all 72 meta side/month coverage cells (the native evaluator also
publishes the 72 base cells).

An older frozen R3 stack may enter this comparison only through
`scripts/normalize_frozen_r3_finalist.py`.  It consumes an existing strict OOF
ledger, its immutable manifest, a separately frozen 21-day admission ledger
and manifest, and a frozen coverage audit.  The adapter verifies byte hashes
and requires each source manifest to declare the exact strict/admission ledger
hash under `files` or legacy `artifacts` (relative paths and basenames are
accepted; absent, unrelated, ambiguous, or mismatched declarations fail). It
then verifies exact candidate identity and copies/renames the already-mapped output
into `meta_causal_21d_expected_net_bps` and `meta_causal_21d_admitted`.  It
does no fitting, map refit, threshold change, target change, or tuning.

R3/S/O finalists must have the same candidate identities, decision timestamps,
target-valid flags and strict-meta availability. They do not have to have the
same realised gross/net paths: different declared target/exit geometries are
legitimate. Each artifact is instead checked independently for its declared
geometry and for `gross - 100 bps == net` before policy economics are compared.
