# P8U Meta HPO selection confirmation — 2026-08-31

## Decision

No new Meta challenger advances.  The frozen Under-F120 incumbent remains the
only retained Meta score contract.

This was a strict offline evaluation of the new HPO-selection workflow, not a
live or execution change.  The HPO proxy chose a bounded set of challengers;
separate strict-prequential MC1 maps, the dual +50-bps admission gate, and one
chronological constrained portfolio were the only advancement authority.

## What the HPO goal is

The correct cheap goal is **not** a universal estimate of downstream PnL.
It is a challenger funnel:

```text
fixed incumbent
  + proxy Top-3 new challengers
  + one uncertainty control
  + one descriptor-diverse control
→ fresh strict MC1 confirmation
→ constrained portfolio
→ promotion or rejection
```

The incumbent is never proxy-ranked out.  A proxy score has no live,
admission, portfolio, or promotion authority.

The historical challenger-relative fit is deliberately still inactive: its
three historic banks provided 14 challenger rows but zero rows satisfying the
frozen full `BeatIncumbent` tolerances.  A single-class fit is rejected
fail-closed.  The existing frozen GateProxy P0 is used only as a Top-3
shortlist reducer.

## Matched contract

All entries below use exactly:

- frozen P8U F72 target-free Base coordinate;
- six complete calendar months of strictly prior MC1 training;
- separate Current and BCF MC1 packages;
- prior-21-day resolved residual shift;
- dual Current and BCF expected EV `>= +50 bps` admission;
- a common chronological constrained portfolio;
- canonical rich policy net labels, joined only after target-free scores;
- held evaluation **February–July 2026**.  August 2025–January 2026 provide
  the score/MC1 history and are not treated as held performance.

The incumbent was rerun under this same Base, policy, months, MC1, threshold,
and portfolio contract; it is not a stored headline comparison.

| Contract | Entries | Net bps/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Fixed Under-F120 incumbent | 5,135 | 129.58 | 665,399.76 | 63.80 | 42.12 | -23.02% |

## F120 GateProxy shortlist confirmation

| Candidate | Proposal role | Entries | Δ entries | Net bps/trade | Δ bps/trade | Total Δ bps | Worst-month Δ | Worst-week Δ | Drawdown Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `u125ts__h1_shallow_sparse` | Top-1 | 5,056 | -79 | 129.97 | +0.39 | -8,286.52 | -1.19 | -2.72 | -15.14pp |
| `u125ts__h0_inherited_under` | Top-2 | 5,046 | -89 | 128.07 | -1.52 | -19,181.54 | +0.25 | -0.24 | -8.82pp |
| `u125ts__h2_compact_tail` | Top-3 | 4,954 | -181 | 129.32 | -0.26 | -24,760.58 | -0.71 | -1.98 | -9.79pp |
| `u125b21__h0_inherited_under` | Diverse control | 5,295 | +160 | 126.72 | -2.86 | +5,605.80 | +1.71 | +3.33 | -24.24pp |
| `u75ts__h1_shallow_sparse` | Uncertainty control | 5,102 | -33 | 128.96 | -0.62 | -7,432.60 | +0.65 | -0.39 | -5.28pp |

The Top-1 variant slightly improves per-trade EV, but loses total utility and
materially worsens drawdown and both worst-period measures.  The only
candidate with a small total-bps increase (`u125b21`) achieves it by accepting
more lower-quality entries and has the worst drawdown deterioration.  Neither
is a promotion candidate.

## F123 paired additive check

F123 appends only the three frozen strict-OOF SHAP-stable fields to the same
F120 contract:

1. `shap_f72_signed_balance`
2. `shap_f72_positive_total`
3. `shap_f72_contrib__mark_perp_dislocation`

The three Top-3 F120 HPO geometries were replayed without retuning under the
same strict downstream contract.

| Candidate | Entries | Net bps/trade | Δ bps/trade | Total Δ bps | Worst-month Δ | Worst-week Δ | Drawdown Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `u125ts__h0_inherited_under__f123` | 5,032 | 129.43 | -0.15 | -14,092.13 | -0.52 | +0.41 | -5.16pp |
| `u125ts__h1_shallow_sparse__f123` | 5,020 | 127.62 | -1.96 | -24,725.16 | -1.76 | -8.36 | -9.65pp |
| `u125ts__h2_compact_tail__f123` | 5,024 | 129.12 | -0.46 | -16,704.66 | +0.98 | -0.39 | -5.63pp |

The additive SHAP-stable overlay does not improve the nominated variants.  It
is retained as an investigated causal feature receipt, not added to the
incumbent.

## Integrity evidence

Every candidate continuation passed:

- exact F72 Base identity and zero Base-rank delta;
- target-free held score persistence before any policy/path join;
- prior-resolved labels for all score and MC1 training;
- exactly six complete calendar months for each MC1 fit;
- separate Current/BCF maps and prior-resolved 21-day shift;
- deterministic serialized MC1 score parity;
- no live or exchange mutation.

Primary artifacts:

- [F120 confirmation plan](../data_perp/artifacts/strict_r3_p8u_meta_relative_gateproxy_targetquery_bank_20260831_v1/hpo_mc1_confirmation_plan_f120_v3/)
- [F120 candidate confirmations](../data_perp/artifacts/strict_r3_p8u_meta_relative_gateproxy_targetquery_bank_20260831_v1/hpo_mc1_confirmation_f120_v1/)
- [matched incumbent control](../data_perp/artifacts/strict_r3_p8u_meta_relative_gateproxy_targetquery_bank_20260831_v1/hpo_mc1_incumbent_f120_matched_v1/)
- [F123 trial receipt](../config/strict_r3_p8u_meta_relative_hpo_trials_under125_timestamp_f123_20260831_v1.json)
- [F123 target-free score roots](../data_perp/artifacts/strict_r3_p8u_meta_relative_gateproxy_targetquery_bank_20260831_v1/hpo_under125_timestamp_f123_v1/)

## Next independent bank

Keep the incumbent as the compulsory control.  Apply the frozen proxy only to
shortlist new challengers, then run matched downstream confirmation.  Do not
fit or tune the challenger-relative proxy until a future independent bank
contains both qualified wins and losses under the frozen `BeatIncumbent`
contract.
