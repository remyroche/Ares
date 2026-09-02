# BCF Same-Lineage Challenger — 2026-08-24

Status: challenger only.  This document is not a promotion or a live-execution
authorization.  The hourly entry producer remains stopped; the existing
position monitor remains the sole live exchange-facing process.

## Purpose

The former August BCF model could not be used because its serialized bundle
was absent and the legacy prequential feature surface did not reproduce the
feature values produced by inference.  This challenger rebuilds the BCF base,
residual, Severe, reference CDF, and BCF-native MC1 components from one
target-free 170-symbol feature-state lineage.

## Immutable inputs and outputs

| Item | Evidence |
|---|---|
| Target-free training surface | 738,310 rows, 4,343 hourly decisions, 170 symbols; SHA `9ed862875b19f9fc1fc95fda12d7244f3e067414d33e8a927992cd405329a992` |
| Label source | Strict prequential ledger SHA `2ef06dde238b0ce0c8c3f6cb2bec305a2a03190f113fddb974097e407792f682`; labels were joined only after feature materialization |
| Rebuilt prequential handoff | 375,360 rows, April–July 2026, 42-day same-model references |
| Frozen Geometry/K9 | BCF semantic hash `7a602dfb5f10bef3791fd869b17dcfaeb53f96264fa8983c01ef5fd79681191c`; never refit |
| BCF monthly bundle | `554155fd64537da03dfbf12ab9cd5cf9f57d792cb792d5807dc60a2e60c100e2` |
| BCF-native MC1 challenger | `f7910b6949c7f6dcb57945184b5f234d74192f9b143fa560c0042e4d20fc23f8` |

The BCF monthly bundle trained 240,000 base rows, 240,000 rank-map/residual
rows, and 231,135 Severe rows.  The BCF-native MC1 challenger trained on
48,826 equal-day sampled examples drawn from 90,780 resolved, BCF-OOS August
score/outcome pairs.

## Causality and parity receipts

1. A July 31 cold start using a source panel truncated before the decision
   exactly reproduced the training surface: 170 identities, timestamps, and
   all 120 model fields matched with maximum absolute delta 0.0.
2. The August scorer used its own 171,360-row June–July same-model 42-day
   reference, then scored 92,990 target-free August candidates.  It used zero
   held-window percentile operations and no outcome column.
3. Re-running the same scorer from its hash-bound reference cache produced
   92,990/92,990 identical identities and a maximum numerical delta of 0.0.
4. The BCF MC1 runtime smoke test used only label timestamps prior to its
   2026-08-24 cutoff and had 82,620 21-day support rows across 21 days.  It is
   a runtime test, not a fresh forward result.

Relevant artifacts:

- `data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/features/`
- `data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/prequential/`
- `data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/bcf_bundle/`
- `data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/bcf_scores/`
- `data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/bcf_mc1_bundle/`
- `data_perp/artifacts/strict_r3_bcf_same_lineage_challenger_20260824_v1/parity_probe_jul31/parity_audit/`

## Required before any reseal

- Materialize a genuinely fresh post-2026-08-24 decision with the new
  target-free source/state lineage and score it with the challenger.
- Run an independent feature/score/admission parity replay on that decision.
- Rebuild the current-v5 half of the dual-admission comparison from the same
  fresh decision and verify the common portfolio state and auction.
- Complete a no-order recovery from the last verified live state into the new
  state lineage; do not bridge historical state files.
- Require a frozen forward-validation period for the newly trained August
  BCF/MC1 pair before approving live entry use.

Until all five conditions pass, no successor seal or live-entry restart is
allowed.
