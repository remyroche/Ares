# Action-layer infrastructure audit — 2026-08-07

## Scope

This audit records the current action-layer inputs and the remaining
implementation gap. It does not promote an action, alter the execution-EV
ranking, or authorize portfolio replay.

## Sealed inputs

### Pre-entry handoff

`data_perp/artifacts/frozen_entry_action_handoff_20260730_v2/handoff.parquet`

- 18,107 unchanged selected-book identities.
- 45 authorised pre-entry model/context inputs, described by
  `feature_roles.json`.
- Side, timestamp, score provenance, residual/base outputs, causal regime and
  transition context are present.
- Future path/target columns are retained only for target/replay parity and
  are not admissible feature inputs.
- The handoff is frozen and must not be reranked or backfilled.

### Outcome-only action target pack

`data_perp/artifacts/execution_action_target_pack_20260730_v2/labels.parquet`

- 110,730 canonical rows.
- Exact contiguous 720-bar one-minute paths.
- Side-relative fixed 1/2/3/4/8/12-hour returns, MFE, MAE, slope,
  underwater, cost-clear, timing, early-path-quality and giveback targets.
- Every target has an explicit availability timestamp.
- The pack is target-only and cannot be used as an inference feature source.

### Older pre-entry action data

`data_perp/artifacts/2023apr_2024_current_policy_wait10_action_20260730_v1`

- 293,828 rows with 25 pre-entry fields.
- Exact enter-now versus wait-10-minute labels and OOF provenance.
- The existing `run_frozen_preentry_wait_action_ablation.py` supports the
  wait-10 diagnostic only.

## Current implementation status

Present:

- exact enter-now parity checks;
- side-local wait-10 event/magnitude heads;
- fixed controls and causal OOF/forward diagnostics;
- stateful exit-action simulator with exact policy parity.

Missing:

- one reusable multi-action trainer for pre-entry `trade/skip/wait/reprice`;
- one reusable prefix-state trainer for post-entry
  `hold/partial/exit/tighten/loosen`;
- a complete causal post-entry prefix-state feature materialization;
- positive-utility lower-bound gating and action-specific promotion reports.

## Required implementation contract

1. Fit long and short action heads independently.
2. Train only on rows whose target has matured before the validation timestamp.
3. Keep action labels, path outcomes and replay-only geometry out of inference
   features.
4. Preserve the frozen global-book identities and weights; action heads may
   change actions, not the ranking population.
5. Compare against enter-now, fixed-horizon and always-wait controls on the
   same rows, with one cost application.
6. Require positive net by side and month, a positive clustered lower bound,
   and improvement over both deployed and fixed controls before promotion.
7. Keep timing, MAE, target-price and wait actions outside the execution-EV
   ranking head.

Native-L2-derived action features remain blocked until the factual historical
backfill is complete. The current request covers 24,391 missing April--July
symbol/day pairs; the partial July sidecar is not sufficient for strict OOF
training or promotion.

## Decision

The target and replay substrate is complete and sealed. The reusable
multi-action and post-entry prefix-state trainers remain implementation work;
no action or policy is promoted from the current diagnostics.

## First reusable-run validation

`scripts/run_action_heads_oof.py` was run against the matched older dataset

`data_perp/artifacts/2023apr_2024_current_policy_wait10_action_20260730_v1`

with 21 causal pre-entry fields, side-local models, monthly expanding folds,
and the explicit `execution_label_end_utc < held_month_start` maturity rule.
It produced 567,496 OOF predictions across 20 held month-side folds for two
heads:

| Head | Mean rank IC | Mean top-10 wait-delta | Positive month-side tails |
|---|---:|---:|---:|
| `wait_better` classifier | 0.0555 | +5.58 bps | 33/40 |
| `wait_delta` regressor | 0.0603 | +8.61 bps | 38/40 |

These are ranked `wait10_net - enter_now_net` diagnostics, not realized
policy results. They do not override the sealed March/April current-book
result, where the learned wait policy failed its paired lower-bound and
current-policy gates. The runner is therefore infrastructure-complete but
promotion remains closed.

Evidence is under
`/Users/remyroche/Documents/Codex/artifacts/action_heads_oof_20260807_v1`.

## Current-book role-bound validation

The first current-book action-head run used a generic numeric/name filter and
was diagnostic only. It has been rerun with the frozen role registry bound
explicitly:

`data_perp/artifacts/frozen_entry_action_handoff_20260730_v2/feature_roles.json`

The registry's 45-field `model_inputs` list is now authoritative; selection
weights, target-only fields, execution-only fields, and other handoff columns
are not silently added. The runner records the registry hash and contract mode
in its manifest. The role-bound artifacts are under

`/Users/remyroche/Documents/Codex/artifacts/action_heads_current_book_20260807_v1/oof_roles_v2`

This matched current-book subset contains 18,107 rows from March--April 2025.
Only April is a held fold (March is the initial training month), yielding
27,916 side/head OOF rows. Results are therefore diagnostic, not a promotion
test:

| Head | Side | Held month | Rank IC | Top-10 utility | Pool utility |
|---|---|---|---:|---:|---:|
| `trade_positive_12h` | long | 2025-04 | 0.0035 | +1.33 bps | +0.66 bps |
| `trade_positive_12h` | short | 2025-04 | 0.1356 | +0.59 bps | −0.24 bps |
| `cost_clear_25bps` | long | 2025-04 | 0.0851 | +2.00 bps | +0.66 bps |
| `cost_clear_25bps` | short | 2025-04 | 0.0704 | +0.27 bps | −0.24 bps |

The 45-field contract is now verified in the manifest, but this two-month
sample is too short to establish monthly robustness or action promotion. The
role-bound run also confirms that action-head evaluation must not rely on the
generic name filter when a frozen handoff contract exists.
