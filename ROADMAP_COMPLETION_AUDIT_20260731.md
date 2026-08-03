# Candidate-level exact-H12 roadmap completion audit

## Decision

`STAGE_B_NO_EXECUTION_TARGET_ADVANCES` remains the correct terminal decision for this roadmap run.

This is not a score selection. It is a layer diagnosis: existing models preserve some opportunity and path-risk information, but no exact-H12 execution formulation produces a positive, calibrated, globally comparable candidate tail under the frozen policy. Therefore no model output is promoted into a new entry graph.

## Requirement-by-requirement evidence

| Roadmap requirement | Evidence | Result |
|---|---|---|
| One frozen candidate-level execution policy, H12 endpoint, exact gross-minus-cost accounting | Exact target manifests and assertions in `exact_h12_target_purity_ablation_20260731_v*`; all historical extensions assert `gross - cost = net`. | Pass |
| Identical rows / OOS upstream base / pooled-global tails before side split | 75,196 matched Aug--Nov 2024 candidates in the exact target suite; 2022--24 historical audits use unique candidate identity and deterministic pooled-global selection. | Pass |
| E0 direct net, E1 residual net, E2 event decomposition, E3 hurdle evaluated before tail weighting | `exact_h12_target_purity_ablation_20260731_v4` and later exact path extensions. | Complete, no arm passes |
| Supportive path labels remain separate from entry EV until a viable execution target exists | Exact reachability, adverse and persistence labels were materialised and tested as target diagnostics; none is supplied to the entry inference graph. | Pass |
| Entry threshold must be based on expected net, not period-wide top-k | Causal 21-day maps in the 2024 suite and 90-day prior-resolved monthly maps in the historical state test. All positive threshold findings lack stable support; no threshold selected. | Pass: no threshold selected |
| Older existing data must be used | 360,012 2022--24 OOF candidates audited; 118,734 rich 2022H2--2023 causal transition candidates used for feature, transition and causal calibration tests. Inverse 2022H1 is kept separate from linear-PF economics. | Pass |
| Do not require walk-forward validation for historical regime diagnosis | Symmetric calendar-block OOF target learnability and weekly grouped-OOF transition-identifiability studies are explicitly labelled research-only. | Pass |
| Identify whether transition regimes are reliably classifiable | Active transition: ROC-AUC 0.951 / 7.98x top-decile lift; onset within 3h: ROC-AUC 0.778 / PR-AUC 0.397. | Pass: diagnostic controller only |

## Execution-target decision tree

| Layer / formulation | What was learned | Disposition |
|---|---|---|
| Frozen base opportunity score | Retains the strongest available reference ordering, but its selected exact-H12 tail is negative. | Reference only |
| Direct exact net and residual net | Terminal post-cost regression is too noisy and loses useful base structure. | Rejected |
| Generic / post-cost competing risk and hurdle decomposition | Path semantics and adverse-risk distinctions are real, but recomposed entry EV is still negative at the global tail. | Diagnostic target research only |
| Exact reachability | More learnable than durable retention; insufficient as entry EV because givebacks are large. | Diagnostic head only |
| Retention/giveback, flat, hierarchical, soft and target-selected variants | Retention conditional on reachability is weakly identifiable from the entry snapshot and does not repair selection. | Action-layer research only |
| Side score bridge | Identifies a real comparability problem but does not improve short economics enough. | Diagnostic calibration issue |
| Score-map plateau tie repair | Changes top-10 by at most 0.25 bps. | Deterministic implementation detail |

## Existing-history extension

The expanded historical work is important because it rejects the easy explanation that the 2024 target failure was a short-panel anomaly.

| Test | Compatible support | Main result | Decision |
|---|---:|---|---|
| Reconstructed residual OOF stack | 309,132 linear-PF rows, 2022H2--2024 | Net rank IC 0.097; global top-10 -98.5 bps. | No viable target tail |
| Transition context, all fields | 118,734 rows, 2022H2--2023 | IC rises 0.120→0.123 but top-10 worsens -80.0→-95.6 bps. | Reject bulk context in EV |
| Six compact transition mechanisms | Same rows and protocol | Every group has negative top-10 and threshold; static state increases IC but loses economics. | Reject as entry features |
| Transition active/onset classifier | 11,736 causal hourly rows, 2022H2--2023 | Active state is reliably detectable; onset is moderately predictable but noisy. | Diagnostic/controller only |
| Causal state 2/3 calibration | 112,802 2023 evaluation rows | Common top-10 -99.1; interaction top-10 -103.1; threshold becomes negative. | Reject calibration interaction |

## Resulting architecture

There is no justified production entry graph beyond the current opportunity reference. The only evidence-supported structure is intentionally incomplete:

```text
causal decision-time features
        ↓
base opportunity model
        ↓ strict OOF opportunity score
        ├── transition-active / onset monitor (diagnostic controller only)
        └── exact-H12 execution research heads (not promoted)
```

The missing edge—`execution model → calibrated_expected_net_bps → enter if > threshold`—is **not** installed, because Stage B does not meet its economic and calibration gate.

## Explicitly not selected

- no new base or residual target;
- no auxiliary-head feature stack;
- no state gate, state quota, side quota, or state-specific threshold;
- no candidate threshold;
- no portfolio constraints, sizing, concurrency, exposure, or exit-policy optimisation;
- no factual historical L2/flow persistence claim.

These are not deferred implementation omissions. The roadmap makes them conditional on a viable exact-H12 execution target, which has not been demonstrated.

## Next admissible work

1. Add genuinely new timestamped decision-time continuation information (L2 depth/imbalance change, spread/depth resilience, aggressor flow, liquidation impulse, liquidity-cluster distance) and test it first on `retain | clear` under strict OOF.
2. Treat the transition monitor as a separate controller study: choose an alert budget and calibrate onset probabilities causally before any downstream action use.
3. Only if a new feature group produces a positive causal expected-net rule and global top-10 on the fixed policy should the roadmap reopen at Stage B; then proceed sequentially to base target, supportive-head, final calibration and threshold selection.

## Primary artifacts

- `data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v11/`
- `data_perp/artifacts/reconstructed_stack_all_eras_audit_20260731_v1/`
- `data_perp/artifacts/historical_transition_target_learnability_20260731_v2/`
- `data_perp/artifacts/historical_transition_identifiability_20260731_v1/`
- `data_perp/artifacts/historical_causal_state_calibration_ablation_20260731_v1/`
- `TARGET_AUDIT_20260731_EXACT_PERSISTENCE.md`

## 2026-08-01 continuation-information follow-up

The first admissible implementation of the next-step feature request is now
materialised as a **research-only native-L2 sidecar**. The generator accepts
only rows tagged `kraken_futures_l2_snapshot`; the 5,981,302 rows tagged
`local_ohlcv_summary` were explicitly excluded as OHLCV-derived proxies. The
corrected v2 sidecar contains 6,928 snapshots across 73 exact product identities from
2026-07-11 11:00:00Z through 2026-07-23 20:00:03Z. It includes causal spread,
top/depth imbalance, depth shape, snapshot-gap and change features, with no
labels, scores, ranks, or portfolio fields. It is registered in config only
as `NATIVE_L2_CONTINUATION_FEATURE_KEYS`; production base/meta feature lists
were not changed. The raw native trade-count/notional/flow fields are all
zero, so no aggressor-flow feature is claimed; bounded prior-snapshot change
fields are available on 6,144 rows.

The candidate-overlap audit then performed a backward as-of join by exact
product identity, never using a future snapshot or forward fill. At the
declared two-hour staleness bound:

| candidate panel | rows | matched rows | coverage | lag-ready rows |
|---|---:|---:|---:|---:|
| canonical candidate handoff | 311,843 | 195 | 0.063% | 41 |
| July 20--23 retrospective bridge | 5,760 | 0 | 0.000% | 0 |
| exact-H12 side-local residual OOF | 127,777 | 0 | 0.000% | 0 |
| A-grade strict-forward scores | 104,590 | 0 | 0.000% | 0 |

The 195 handoff matches span only three days and 32 products. This is not a
training cohort and cannot support strict OOF feature selection, HPO,
economics, or a policy claim. The detailed, label-free audit is
`data_perp/artifacts/native_l2_candidate_overlap_audit_20260801_v2/` (v1 is
fail-closed because its lag-indicator missingness contract was repaired).

The practical conclusion is therefore unchanged but now evidence-backed: the
roadmap still has no positive execution target. The native-L2 feature request
is implemented, but its historical source coverage must be extended before it
can enter a candidate-level OOF experiment. In the meantime, the transition
classifier remains a diagnostic controller rather than an entry gate, and
timing, wait, target-price and portfolio layers remain out of scope.

## 2026-08-01 source/backfill readiness closure

The local native-source search is now recorded in
`data_perp/artifacts/native_l2_backfill_readiness_20260801_v1/`. The scan
covered both local orderbook-hourly roots and found 568 parquet files with
10,373,441 rows. Only 6,928 rows are tagged
`kraken_futures_l2_snapshot` (73 product-file identities); 10,366,513 rows are
explicitly tagged `local_ohlcv_summary` proxies and remain excluded.

The exact native window is 2026-07-11T11:00:00Z through
2026-07-23T20:00:03Z, but the declared candidate requirement begins
2026-04-01. The readiness gate consequently remains fail-closed and records
`historical_native_backfill_required=true`, `candidate_joined=false`,
`model_fitted=false`, and `promotion_eligible=false`. This is a source
coverage result, not a model result and not evidence that proxy rows can be
used as native L2.

The next admissible transition is: acquire/materialize longer factual native
L2 history; rerun the existing causal sidecar; rerun exact-product backward
as-of overlap; then, only if coverage passes, build strict OOF `retain | clear`
labels and evaluate one pooled-global top-k book. No portfolio or action layer
is reopened by this readiness artifact.

## 2026-08-01 dense native-source extension

The initial hourly-source audit did not exhaust the local native feed. The
broader source inventory found raw per-level native snapshots in
`data_perp/exchanges/krakenfutures/spread_snapshots/orderbook_history/`.
`scripts/materialize_native_l2_continuation_from_snapshots.py` now aggregates
those rows with a vectorized, source-restricted reducer and retains observed
timestamps as feature availability. The authoritative current-period artifact
is `data_perp/artifacts/native_l2_continuation_sidecar_20260801_v3/`.

It contains 51,778 aggregated snapshots across 303 products, with 50,334
bounded-lag-ready rows (97.21%) from July 11--23. The corresponding corrected
as-of overlap audit is
`data_perp/artifacts/native_l2_candidate_overlap_audit_20260801_v3/`: it
matches 10,282/311,843 canonical handoff rows (3.297%) and 3,300/5,760
July-20--23 bridge rows (57.292%), while the exact-H12 May--July and A-grade
strict-forward panels remain at zero. The full historical gate therefore still
fails: no labels, OOF model, HPO, economic result, or promotion decision is
derived from this cohort.

## 2026-08-01 daily native-source coverage recheck

The full local `data_perp` tree was re-inventoried with the corrected v3
overlap manifest. The scan covers 71,135 parquet files and 327,133,322 rows;
2,865,522 rows are tagged `kraken_futures_l2_snapshot`. Native observations
exist on ten UTC days only (July 11--16, 18, and 21--23), with calendar gaps
on July 17, 19, and 20. The authoritative artifact is
`data_perp/artifacts/native_l2_backfill_readiness_20260801_v3/`.

This does not pass the roadmap gate: the candidate window begins April 1,
native history begins July 11, and exact-H12 May--July overlap remains zero.
The artifact remains source-readiness evidence only; labels, OOF fitting,
HPO, economics, and promotion remain disabled.

## 2026-08-01 current-run stop and registry reconciliation

The current-run stop audit is sealed at
`data_perp/artifacts/current_run_stop_audit_20260801_v1/`. An escalated
process-table check found zero active Ares processes. Registered PID 1026 was
verified as a reused macOS `imagent` process rather than the Ares collector;
no signal was sent to the unrelated process, and the registry entry is now
`stale_pid_reuse`. No new roadmap run was started.

## 2026-08-01 native-L2 backfill request

The actionable acquisition manifest is
`data_perp/artifacts/native_l2_backfill_request_20260801_v1/`. It was built
from identity/timestamp columns only and requests 24,391 missing product/day
pairs across the April 1–July 23 candidate window. The partial local sidecar
currently covers 952 of 25,343 required pairs. No labels, models, HPO,
economics, or promotion decision is derived from this request.

## 2026-08-01 target–feature–execution alignment audit

The cached roadmap has been reconciled into
`data_perp/artifacts/target_alignment/alignment_audit_20260801_v2/`.
The audit is fail-closed: 55 checks pass and 2 fail. The exact-H12 target
contract is mechanically sound (720-minute horizon, labels available only at
the horizon, causal feature cutoff, frozen policy/cost IDs, and gross minus
row cost equals net exactly once), the fold and aggregate OOF manifests are
chronological, and economics use one pooled global top-k book.

The unresolved items are material, not cosmetic. The canonical v3 target pack
at `data_perp/artifacts/root_cause_exact_h12_execution_target_pack_20260801_v3/`
now carries the full explicit supportive validity/condition/censoring/
support-count metadata requested by the roadmap. The remaining economic gates
are negative: the best supportive top-10% result is -113.44 bps net; the best
exact-H12 target arm is -104.05 bps net; and factual native-L2
history is missing for 24,391 April–July product/day requirements. The pack
therefore remains `FAIL_CLOSED_RESEARCH_ONLY` with
`promotion_eligible=false`.

## 2026-08-01 economic headroom/ranking audit

The economic failure is not explained by cost alone. The versioned diagnostic
`data_perp/artifacts/exact_h12_economic_headroom_diagnostic_20260801_v1/`
reports an oracle pooled global top-10% of **+468.27 bps gross**, **102.34 bps
cost**, **+365.93 bps net**, while the best model top-10% (frozen
`CONTROL_base_opportunity`) is **-4.07 bps gross**, **99.98 bps cost**,
**-104.05 bps net**. The model top-1% gross is **+91.67 bps**. Therefore the
candidate population contains a positive economic tail, but current feature/
label/ranking conversion does not recover it at 10% global selection.

The cost-sensitivity check gives **-4.07 bps** at zero hypothetical cost for
the best model top-10%; changing the fee assumption cannot by itself clear
the gate. Ranking diagnostics were corrected to use the actual selection
score (`calibrated_expected_net_bps`) and the stable materialized-row tie
rule, and an exact score-alignment artifact proves that the diagnostic
reproduces the authoritative target-ablation metrics. The former raw-score
view is superseded.

This strengthens the next-step recommendation: improve cost-aware economic
tail ranking (clean/competing-risk reachability, conditional magnitude,
continuation information, and calibration) before adding supportive heads,
timing/wait actions, or portfolio constraints. Promotion remains blocked by
the two negative economic checks and by incomplete native-L2 history (24,391
missing April–July product/day requirements).

## 2026-08-01 requirement-level completion matrix

The cached roadmap is now audited directly by
`data_perp/artifacts/updated_roadmap_requirement_audit_20260801_v1/`. The
matrix reports **12 PASS**, **5 FAIL**, and **1 BLOCKED_EXTERNAL**. It confirms
the exact target contract, target-layer separation, supportive metadata,
feature eligibility, strict candidate OOF lineage, pooled-global selection,
ablation coverage, and manifest reproducibility.

It also makes the unresolved acceptance gates explicit: pooled top-10 net,
latest-month net, both-side net, paired-bootstrap uncertainty, and
supportive-OOF incremental value all fail. Native-L2 remains a separate
external acquisition block with 24,391 missing symbol/day pairs. No model or
policy is promoted from this matrix.
