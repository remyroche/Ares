# P8U CMI / Under–Over–Magnitude causal research — 2026-08-29

## Decision

**No Under, Over, or Magnitude head advances.**  Each candidate was fitted and
evaluated with strict chronological out-of-fold lineage on the only common
full-1400-feature windows currently available (October–December 2025).  Every
candidate degraded the predeclared downstream score-stability objective.  The
requested 2026 validation was therefore deliberately not run: it was gated on
a positive, portable 2025 result rather than used to rescue a failing head.

This is a negative result, not evidence that the base score has no economic
information.  It says that the present 1,400-feature contract adds no usable
incremental correction through these three head definitions after conditioning
on the base explanation in the top 15% of base predictions.

## Scope and provenance

The external ChatGPT shared specification was not retrievable in this session.
This implementation therefore follows the closest local, versioned P8U
pipeline contract and records that limitation explicitly.  No result below is
claimed as a bit-for-bit reproduction of an inaccessible external text.

Two distinct pieces of evidence were created.

1. **2024–2025 upstream recovery.**  Target-free F72/base and Router feature
   panels were recovered for July 2024–February 2025 from decision-time inputs,
   and strict target-free base scores were produced for December 2024–February
   2025.  This demonstrates the early historical substrate and its causal
   identity handling, but it is not a full-1400 meta-head experiment.
2. **Full-1400 CMI / UOM test.**  The native full feature universe contains
   1,407 candidate fields; 1,093 passed the established coverage/variance
   gate.  Conditional-MI evidence was fitted only inside the timestamp-local
   top 15% of the base score.  Separate frozen 80-field contracts were built
   for Under, Over, and Magnitude, then scored chronologically OOF in
   October–December 2025 before policy outcomes were joined downstream.

At the time of the completed head decision, the full-1400 test could not
honestly be extended earlier because existing early feature panels did not
share the compatible 1,407-field contract.  The early substrate has documented
coverage holes, especially a small set of cross-asset beta/residual fields,
and was therefore retained as diagnostic evidence rather than silently
substituted for the full contract.  A separate bounded re-materialisation of
the true full contract for December 2024–February 2025 is now in progress
under a corrected source-precedence contract.  It is not part of the decision
table until its identity, coverage, strict source-horizon, and OOF audits
complete.

### Superseded early-materialisation attempts

The first bounded early roots
`strict_r3_p8u_full1400_meta_features_dec24_feb25_batched64_20260829_v2` and
`strict_r3_p8u_full1400_meta_features_nov24_batched48_20260829_v1` are
**invalid for downstream use**.  They constructed cross-sectional fields from
the Router-50 candidate subset rather than the complete contemporaneous market
universe.  A direct shared-ID F72 parity audit detected material differences
in eight fields.  They are retained only as diagnostic receipts.

Their first replacements used the frozen 141-symbol raw market-context list
from the prior target-free feature manifest:

- `strict_r3_p8u_full1400_meta_features_dec24_feb25_completecontext_batched64_20260829_v3`
- `strict_r3_p8u_full1400_meta_features_nov24_completecontext_batched48_20260829_v2`

The feature engine now distinguishes `candidate_symbols` (output rows) from
`context_symbols` (raw market inputs).  Those partial roots are not usable:
they were stopped before completion after a second, more fundamental source
audit found a horizon-sensitive raw-source fallback.

### Source-precedence repair (pending full all-symbol receipt)

The audit found two variants of the same bug: source selection was affected by
whether a later portion of the requested horizon happened to be complete.

1. A later missing 15-minute bar could cause the canonical cache to be opened
   for an earlier row.
2. A partial official hourly archive that began in a later month could suppress
   an older legacy-hourly fallback for earlier rows.  KAIA in December 2024
   exposed this: extending the source cutoff to March 2025 erased 658 prior
   hourly observations even though no December source value had changed.

The canonical panel builder now merges every source **cell-locally** with the
declared precedence `downloaded_15m → canonical cache → official hourly →
legacy hourly`.  The two source-precedence regression tests pass.  The
all-symbol real raw-prefix audit is running under
`strict_r3_p8u_f72_horizon_invariance_dec24_20260829_v9`; only an exact result
permits the corrected full-feature builds to start.  New materialisation
manifests also bind the panel-builder hash and
`cell_local_15m_cache_official_legacy_v2` source contract, preventing a resume
from mixing the earlier semantics.

## Causality contract

- Score panels are target-free when written; policy outcomes are joined only
  after scoring.
- CMI is conditioned on `BASE_EXPLANATION_V1` and calculated only on rows in
  the base timestamp-local top 15%, never on a future global rank.
- `BASE_EXPLANATION_V1` comprises the base score/rank and timestamp-local
  explanatory statistics: `base_score`, `base_rank_ts`, query count/mean/std/
  range, score z-score, top gap, and top-two gap.
- Each head uses an independently selected frozen 80-field causal contract.
- Head training and the policy conversion comparison are chronological OOF;
  the held month and its outcomes never enter the corresponding fit.
- No live model, MC1 calibration, portfolio configuration, or execution
  contract was modified.

## Full-1400 strict-OOF results

`SStableMeta` is the predeclared score-stability objective (higher is better).
The top-2 substitution statistic is an economic diagnostic, not a live
admission rule.

| Head | Objective family | CMI(meta, policy \| base) | Mean top-2 substitution EV | Mean admission substitution utility | SStableMeta | Worst-week SStableMeta | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Under | `under_bps100__timestamp` | +0.1291 | -47.03 bps | +1.46 bps | **-0.1589** | -0.1624 | reject |
| Over | `over_bps100__timestamp` | +0.0735 | -17.22 bps | -2.13 bps | **-0.1518** | -0.1877 | reject |
| Magnitude | `magnitude_bps__base_band` | +0.0794 | -50.02 bps | -0.07 bps | **-0.2238** | -0.3093 | reject |

The Under target has measurable conditional association, but that association
does not convert to useful downstream selection.  The separation is important:
statistical incremental information is not sufficient for an executable
correction head.

### Under monthly OOF diagnostics

| Held month | SStableMeta | Top-2 substitution EV |
|---|---:|---:|
| 2025-10 | -0.1237 | -36.83 bps |
| 2025-11 | -0.1625 | -51.01 bps |
| 2025-12 | -0.1905 | -53.26 bps |

Magnitude is negative in all three months (October -0.4287, November -0.2106,
December -0.0321).  No candidate has the required consistently positive,
portable economic result to justify touching 2026.

## Early recovered base evidence (not a meta-head promotion test)

The following pre-repair F72 figures remain historical diagnostics only.  They
are not selection, promotion, or strict-causal evidence until the patched
source-horizon audit and the rebuilt early feature ledger confirm equivalence:

| Held month | Top 1% | Top 2% | Top 5% | Top 10% | Top 15% |
|---|---:|---:|---:|---:|---:|
| 2024-12 | +207.40 | +195.56 | +156.94 | +104.73 | +65.91 |
| 2025-01 | +235.41 | +204.23 | +160.99 | +133.11 | +106.63 |
| 2025-02 | +198.51 | +163.23 | +141.06 | +108.61 | +79.43 |

All values are net bps/trade.  These results use a two-month startup window
because the earlier policy ledger does not supply a full three-month warm-up;
they are **diagnostic only** and must not be used to select or promote an
inference model.

## Artifacts

### CMI selection

- `data_perp/artifacts/strict_r3_p8u_meta_uom_top15_cmi_under_octdec25_20260829_v1`
- `data_perp/artifacts/strict_r3_p8u_meta_uom_top15_cmi_over_octdec25_20260829_v1`
- `data_perp/artifacts/strict_r3_p8u_meta_uom_top15_cmi_magnitude_octdec25_20260829_v1`

### Frozen contracts

- `data_perp/artifacts/strict_r3_p8u_meta_uom_top15_contract_under80_octdec25_20260829_v1/contract.json`
- `data_perp/artifacts/strict_r3_p8u_meta_uom_top15_contract_over80_octdec25_20260829_v1/contract.json`
- `data_perp/artifacts/strict_r3_p8u_meta_uom_top15_contract_magnitude80_octdec25_20260829_v1/contract.json`

### Strict-OOF scores and metrics

- `data_perp/artifacts/strict_r3_p8u_meta_uom_under80_octdec25_score_20260829_v1`
- `data_perp/artifacts/strict_r3_p8u_meta_uom_over80_octdec25_score_20260829_v1`
- `data_perp/artifacts/strict_r3_p8u_meta_uom_magnitude80_octdec25_score_20260829_v1`

### Early target-free recovery

- `data_perp/artifacts/strict_r3_p8u_f72_router_early_jul_oct24_identityonly_20260829_v1`
- `data_perp/artifacts/strict_r3_p8u_f72_router_early_nov24_identityonly_20260829_v1`
- `data_perp/artifacts/strict_r3_p8u_tail125_base_history_dec24_feb25_partialwarmup_20260829_v1`

### Reusable code

- `scripts/select_strict_r3_p8u_meta_fullfeatures_v1.py`
- `scripts/build_strict_r3_p8u_cmi_meta_contracts_v1.py`
- `scripts/materialize_strict_r3_f72_early_router_features_v1.py`
- `scripts/audit_strict_r3_source_horizon_invariance_v1.py`
- `scripts/run_strict_r3_p8u_precision_preservation_weight_funnel_v1.py`

## Next decision

Keep the current canonical stack unchanged.  A future meta research arm needs
one of the following before revisiting this family:

1. a common full-1400 causal feature ledger across a longer 2024–2025 span;
2. a target that has both conditional association **and** a positive downstream
substitution/utility result in development; or
3. a separately predeclared architecture that uses the features as a
conservative risk demoter rather than as a rank correction.
