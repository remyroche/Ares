# R2 Contract Assertion Report

Date: 2026-07-24
Status: `RUNNING`

| Contract | Test evidence | Artifact evidence | Verdict |
|---|---|---|---|
| UTC and causal decision/path timestamps | Focused and broader suites pass | P0-wide assertion report not yet generated | PARTIAL |
| Exactly seven CatBoost classes and stable order | Path support and CatBoost tests pass | Cross-stage class-order/hash reconciliation pending | PARTIAL |
| Favorable/adverse probability masses | CatBoost and execution-EV tests pass | P0 CatBoost-to-handoff values pending | PARTIAL |
| No abandoned transforms/memberships/weights in inference inputs | Classifier contract tests pass | Final selected feature manifests pending | PARTIAL |
| Costs applied exactly once | Label, execution-EV, timing, and candidate tests pass | P0 label/replay cost hash reconciliation pending | PARTIAL |
| Strict OOF provenance and no outcome inputs | Execution-EV handoff tests pass | R3 audit found existing Pack-B/residual lineage insufficient | FAIL |
| Frozen AE/GMM input order and state reuse | Broader frozen-sidecar tests pass | P0 state/input-order hash reconciliation pending | PARTIAL |
| Purge and embargo for overlapping paths | Execution-EV model tests pass | Stage-wide fold calendar/hash reconciliation pending DEC-09 | PARTIAL |

## Blocking findings

1. Existing Pack-B OOF and residual artifacts do not prove the required
   row-level, own-side lineage. R3 regeneration or recovery is required.
2. The P0 artifact-wide hash reconciliation has not yet been emitted.
3. DEC-09 must freeze the fold calendar, purge, embargo, and untouched replay
   period before final training.

R2 remains open even though both deterministic code suites pass. No downstream
training may treat the existing shared-store top-40 or alpha execution OOF as
canonical evidence.
