# V2 State/Reliability Additive Subspace Screen — 2026-08-29

## Scope

This is an offline, long-only research screen. Every candidate retains the
current frozen 120-field parent Meta contract; state/reliability fields are
strictly additive. No live, canonical, MC1, admission, portfolio, or exchange
contract was changed by this work.

The V2 state fields are target-free at inference. Feature eligibility was
frozen from the 2025 selection receipt. The 2026 January–July results below
are a development confirmation, not untouched promotion evidence.

## Direct additive-arm audit

The direct arms below all retain the same F120 parent fields. `M8` means the
complete additive reliability bundle; its historical receipt was written as
`m7_full_reliability` before the producer’s arm-name correction. Future
materialisations use the distinct `m8_full_reliability` ID.

| Arm | Additive family | SStableMeta | Mean Top-2 substitution EV | Admission utility |
|---|---|---:|---:|---:|
| M0 | Parent control | -0.09056 | -44.05 bps | +5.10 bps |
| M1 | Deviation fields | -0.06041 | -37.28 bps | +5.43 bps |
| M2 | Frozen episode geometry | **-0.05701** | -37.33 bps | +5.48 bps |
| M3 | Error variance | -0.07014 | -39.79 bps | +5.37 bps |
| M4 | Failure likelihood | -0.09149 | -44.84 bps | +5.54 bps |
| M5 | Authority context | -0.08082 | -41.33 bps | +5.46 bps |
| M6 | Deviations + failure | -0.07102 | -42.36 bps | +5.20 bps |
| M7 | Episodes + failure + variance | -0.09293 | -43.27 bps | +5.37 bps |
| M8 | Full reliability bundle | -0.08053 | -41.80 bps | +5.48 bps |

M1/M2 were advanced because they provided the only coherent strict-OOF
Meta-screen improvement. Their prior downstream results were non-promoting;
the additional p056 subspace was the final bounded synergy probe.

## Screen

64 deterministic shallow subspaces were assessed across seven chronological
held months. The parent-control stability score was -0.07848. The best
screened subspace was `p056`, with a stability score of -0.05239, and it beat
the parent-control stability score in every held month.

`p056` adds nine fields:

- `v2_direct_delta_control__correlation`
- `v2_innovation_z__breadth`
- `v2_innovation_z__oi_effective_rank`
- `v2_regime_second_distance`
- `v2_transition_abs1_breadth`
- `v2_transition_positive_breadth`
- `v2_transition_z__execution_spread_level`
- `v2_transition_z__liquidity_depth`
- `v2_transition_z__spectral_lambda1_share`

The smaller `p023` contract did not survive the detailed frozen-model rerun.

## Detailed strict-OOF Meta result

| Contract | SStableMeta | Weekly robust average | Weekly lower tail | Mean Top-2 substitution EV | Admission utility |
|---|---:|---:|---:|---:|---:|
| Parent M0 | -0.09056 | — | — | -44.05 bps | +5.10 bps |
| `p023` compact | -0.09060 | -0.03202 | -0.11717 | -40.83 bps | +5.40 bps |
| `p056` additive | -0.05816 | -0.01225 | -0.09182 | -36.78 bps | +5.41 bps |

`p056` is a credible upstream signal candidate, but upstream screening is not
sufficient for advancement.

## Matched downstream test

The downstream comparison reuses the exact parent Base source, independent
strict-prequential dual MC1 maps, 50-bps dual admission, and one chronological
portfolio. Evaluation is April–July 2026 after the required three-month MC1
warm-up.

| Arm | Accepted trades | Admitted candidates | Net EV/trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent M0 control | 3,634 | 14,074 | +119.21 | +433,200.57 | +68.28 | +50.44 | -21.86% |
| Existing M2 episode reference | 3,678 | 14,818 | +117.86 | +433,476.44 | +67.60 | +48.36 | -22.45% |
| `p056` additive | 3,659 | 14,281 | +116.63 | +426,760.94 | +65.37 | +44.16 | -26.76% |

## Decision

Do not advance `p023` or `p056` to the canonical or live stack. `p056` shows
that several lower-standard but recurring state features can improve the Meta
objective when combined with the parent features. It fails the required
downstream test, however: it lowers EV/trade by 2.57 bps, total net bps by
6,439.63 bps, worsens the worst week by 6.27 bps, and worsens maximum drawdown
by 4.91 percentage points versus the parent M0 control.

All artifacts are research receipts only:

- state screen: `data_perp/artifacts/strict_r3_p8u_state_reliability_subspaces_janjul26_20260829_v2`
- p023 detailed run: `data_perp/artifacts/strict_r3_p8u_state_reliability_p023_objective_janjul26_20260829_v1`
- p056 detailed run: `data_perp/artifacts/strict_r3_p8u_state_reliability_p056_objective_janjul26_20260829_v1`
- p056 downstream run: `data_perp/artifacts/strict_r3_p8u_state_reliability_downstream_p056_janjul26_20260829_v1`
