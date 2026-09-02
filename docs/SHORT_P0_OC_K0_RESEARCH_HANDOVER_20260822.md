# Short P0 → O → C → K0 research handover

Status: frozen **research** winner. It is not a live or canonical trading
authority. The work uses short-side target-free candidates and strict
prequential outcomes only.

## Frozen portable research stack

```text
P0 target-free candidates
  → O: P(MFE within 6h > 250 bps)
  → C: conditional normalized-regret conversion quality
  → K0: p(O) × μ1(C) + (1 − p(O)) × μ0(P0 anchor)
  → admit expected policy net ≥ +75 bps
```

O is the frozen O250/H6 binary LightGBM: 45 stable fields, uniform weights,
and Platt probability calibration. C is a five-state ordinalized normalized
regret target trained only on true O-positive rows. The portable C contract has
59 fields: it is the prior C60 contract without
`ob_trade_size_to_l1_depth_z_24h`, whose 2025–26 coverage was only 77.8%.

K0 is analytic, not another prediction head:

```text
K0 expected net = p(O) × μ1(C) + (1 − p(O)) × μ0(P0 anchor)
```

`μ1` is an isotonic conditional policy-net map. `μ0` is a five-bin map of the
existing P0 anchor, empirically shrunk to its global policy-net prior with
`k=500`. The live-style admission candidate for this research stack is
`K0 expected policy net >= +75 bps`.

Excluded by contract: MC1, a trust/risk head, consensus, a separate mapper,
and any live/canonical authority.

## Frozen diagnostics and MC1 assessment

The frozen diagnostic scorecard was run after the contract was fixed.  It
reports K0 and opportunity bands separately by era, the `p(O) × C` matrix,
component-neutralisation counterfactuals, the full `25…200 bps` threshold
curve, nearby historical O-definition controls, and an execution-margin
sensitivity curve.  None of these diagnostics changes O, C, K0, or the
`+75 bps` admission threshold.

The exact requested MC1-equivalent was also tested as an isolated,
strict-prequential challenger using only P0, O250/H6, C3/C59 and K0-derived
target-free inputs.  It was a fixed shallow HGBR expected-policy-net mapper;
each held-month model used only earlier outer-OOS rows whose labels had
resolved before that month.  It did not advance:

| Matched strict-prequential arm | Net bps/trade | Total net bps | CVaR10 |
| --- | ---: | ---: | ---: |
| Native frozen K0 | +171.82 | +146,902 | −508.56 |
| MC0: K0 only | +172.26 | +138,323 | −519.79 |
| MC1: P0 + O + C + K0 | +161.72 | +127,275 | −550.84 |

MC1's 2026 mean was stronger (`+184.72 bps/trade`) but its 2025 mean,
combined contribution and downside all deteriorated.  It remains excluded;
no canonical or live artifact was changed.  The scorecard and the immutable
MC1 receipt are under
`data_perp/artifacts/strict_r3_short_p0_oc_k0_frozen_diagnostics_mc1_20260822_v2`.
The v2 rerun reproduces the v1 component ledger and MC1 prediction ledger
exactly; it is the current verification receipt.

## Why C59 rather than C60

C60 was development-strong, but one high-ranking C feature had a material
availability shift after 2024. It was present in 100% of development rows, but
only 77.8% of 2025–26 valid rows. C59 removes that one feature and avoids a
training-versus-inference imputation regime shift.

The final O/C coverage audit contains 104 fields (45 O + 59 C); all have at
least 96.8% valid-row coverage and nonzero variance over the audited history.

## Strict-OOS final economics

The portable C59 final stack begins in February 2025, after three months and
at least 500 opportunity-positive prior-resolved rows become available for K0.

| Stack | 2025 net bps/trade | 2026 net bps/trade | Mean | Total net bps | Known trades | Worst month | Mean CVaR10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Portable C59 + K0 anchor5/k500 + abs75 | +166.39 | +167.12 | +166.53 | +218,652 | 1,313 | −0.79 | −343.61 |
| C60 high-total control + K0 anchor5/k50 + abs50 | +156.49 | +148.19 | +154.43 | +227,315 | 1,472 | +6.46 | −387.67 |

C59 is selected under the roadmap’s priority—highest mean EV among arms
meeting the two-era and participation gates—while the C60 control retains
3.8% more total bps. This is a research trade-off, not a promotion claim.

## Causality and parity evidence

- Every historical candidate is scored target-free; 3,703 invalid/unresolved
  rows are excluded from label fitting and economics, never encoded as loss.
- For every held month, K0 calibration uses only earlier outer-OOS rows with
  `label_available_at < held_month_start`.
- The independent final reconstruction matches all 13,081 supported candidate
  IDs and every published score component exactly (maximum delta `0.0`).
- K0’s reconstructed analytic formula differs only by float32 serialization:
  maximum absolute delta `2.44e-05` bps.
- The 411 selected but unresolved rows remain predictions only; they are not
  assigned retrospective outcomes.

## Research sequence retained as evidence

1. `run_strict_r3_short_p0_oc_k0_round1.py` — jointly consistent opportunity
   threshold/horizon definitions.
2. `run_strict_r3_short_p0_oc_k0_round2.py` — O-specific stability-MDA,
   feature caps, weights, calibration, and HPO.
3. `run_strict_r3_short_p0_oc_k0_round3_c_targets.py` — C target funnel;
   normalized regret advanced.
4. `run_strict_r3_short_p0_oc_k0_round3_c_refinement.py` — C-specific MDA
   and C weighting; C60/uniform advanced before portability repair.
5. `run_strict_r3_short_p0_oc_k0_round3_c_hpo.py` — C HPO and three-seed
   check; neither beat C60/uniform.
6. `run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair.py` — matched
   single-feature portability repair.
7. `run_strict_r3_short_p0_oc_k0_round4_k0_refinement.py` — K0 μ1/μ0/
   admission funnel.
8. `audit_strict_r3_short_p0_oc_k0_final.py` — independent final audit.

Primary audited artifacts:

- `data_perp/artifacts/strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair_20260822_v1`
- `data_perp/artifacts/strict_r3_short_p0_oc_k0_round4_k0_refinement_c59_20260822_v1`
- `data_perp/artifacts/strict_r3_short_p0_oc_k0_final_audit_c59_20260822_v2`

The early, partial `...final_audit_c59_20260822_v1` directory is intentionally
not a result: report rendering failed after computation. Use the immutable v2
audit, which recomputed and validated all scores.

## Frozen O45/C59 portability receipt — no feature reselection

`audit_strict_r3_short_p0_oc_k0_feature_contract.py` now provides the
separate target-free robustness receipt at
`data_perp/artifacts/strict_r3_short_p0_oc_k0_feature_contract_portability_20260822_v2`.
It does **not** alter the selected C59 stack, fit a new MDA, or promote a new
feature contract.  It audits finite coverage, month/era drift, PSI, range and
outlier shifts, target-free redundancy, source-tier coverage, and the already
completed MDA stability.  Any diagnostic using labels is labelled as such.

The earlier statement that all final-contract fields had at least 96.8%
coverage is superseded by this stricter monthly/era audit.  It blacklists
seven O and three C fields for a *future portability challenger only*: two
O and one C fields have sustained later-era coverage below 90%, while the
remaining seven repeatedly become near-constant.  The frozen O45/C59 model
continues to use its original contract; these findings do not retroactively
change it.

The audit predeclares, without economic selection, O35/O30 and C50/C40
portable/redundancy representatives for a later untouched evaluation.  O40
is deliberately unavailable—only 38 O fields survive the stated target-free
rules, and the audit will not pad a nominal 40-field contract with a
blacklisted field.  The family-dropout receipt is strict chronological and
head-isolated: an O-family dropout keeps the source C score exactly fixed;
a C-family dropout keeps the source O score exactly fixed; K0 is then replayed
strict-prequentially.  It is attribution evidence, not a selection mechanism.
`O60` is intentionally not emitted: expanding above frozen O45 would be a new
supervised feature-generation/selection exercise, which this audit explicitly
does not perform.

### Current portability receipt: v3

Use
`data_perp/artifacts/strict_r3_short_p0_oc_k0_feature_contract_portability_20260822_v3`.
It supersedes v2 only as a diagnostic receipt: it separates OI, funding and
leverage in the family dropout rather than pooling them as one broad family,
and it replaces the ambiguous “source reliability proxy” label with an
explicit source-lineage table.  The short stack has **no sealed short live
inference/source-receipt contract**, therefore no feature is claimed to have
verified live-source reliability.  Historical source availability and
symbol-tier coverage are reported separately; they are not a live SLO or a
promotion basis.

The refined, strict-prequential family evidence is directionally clear but
remains diagnostic only: O loses 34.90 bps/trade without leverage and 22.52
bps/trade without OI positioning; C loses 23.01 without session/time, 13.93
without volatility transition, and 13.34 without the spectral state field.
Removing O liquidity or O volatility happens to improve the exhausted-period
diagnostic, which is precisely why no dropout is used for feature pruning.

## Pre-registered untouched feature-contract experiment

The next evidence block must use the immutable 2×2 registry at
`data_perp/artifacts/strict_r3_short_p0_oc_k0_untouched_2x2_preregistration_20260822_v1`.
It is a preregistration only—no model fit, score, K0 replay, or 2025–26
policy-economic result was generated for these arms.

| Arm | O contract | C contract | Role |
| --- | --- | --- | --- |
| A0 | O45 | C59 | frozen research control |
| A1 | O30 | C59 | compact-O challenger |
| A2 | O45 | C40 | compact-C challenger |
| A3 | O30 | C40 | fully compact challenger |

All arms lock the same P0 candidate contract, O250/H6 target, C3 normalized
regret target, uniform weights, frozen LightGBM parameters/seeds, Platt O
calibration, K0 `isotonic μ1 + anchor5/k500 μ0`, and the `+75 bps` absolute
expected-policy-net admission.  The evaluator must reject a held start before
`2026-08-01 UTC`, so the contracts cannot be selected on the evidence used to
create them.  It must report EV, trades, CVaR, worst month/week, O calibration,
within-volatility lift, and C score/MFE-bucket monotonicity.  No automatic
promotion exists; a compact head must be economically non-inferior **and**
improve portability/stability on untouched data.

`O60` is not a fifth arm: it would expand beyond frozen O45 and therefore
requires a later supervised feature-generation study, not this pruning test.

### Short source-readiness gate

The same preregistration materialises one source-readiness row for every O/C
field.  Because there is no sealed short current-hour runtime source and
training/inference-parity receipt, all currently non-blacklisted fields are
grade C (historically available, live-unproven) and hard-portability failures
are grade D.  There are presently no A/B fields, therefore all four short arms
are **research-only and fail closed for production**.  Historical coverage is
not treated as a live SLO.
