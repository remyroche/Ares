# P8U Meta — enriched-feature / GateProxy HPO final decision

## Decision

The enriched Meta pipeline completed without changing a live, inference, or exchange artifact. The retained research control remains the existing **Under-F120 LightGBM rank-XENDCG depth-4 sparse model** (`lgbm_hpo_05_depth4_sparse`). No newly screened regular-HPO candidate cleared its strict-MC1 constrained-portfolio comparison.

- Retain the existing Under control for future research comparisons.
- Do not promote a new HPO configuration or reopen CatBoost.
- Reject the Over and Magnitude target families for this stack.
- Retain State only as a diagnostic family, not a successor.

This is a completed research decision, not a live-stack change.

## Causal comparison contract

All candidates use the frozen F72 Base coordinate, canonical reconciled rich-policy outcomes, strict target-free score materialisation, fixed dual MC1 maps, a `>= +50 bps` admission gate, and the normal shared chronological portfolio constraints. Score prehistory is August–December 2025; MC1 and portfolio evaluation is February–July 2026.

The new receipts prove that score unions were target-free before outcomes opened, Meta/Base identities and Base ranks matched exactly, historical/evaluation months were disjoint, Base and policy contracts were fixed, and no live or exchange mutation occurred.

Key receipts:

- retained Under confirmation: `data_perp/artifacts/strict_r3_p8u_meta_crossmodel_fullf120_selected_mc1_under100_fullprehistory_aug25jul26_20260830_v2/`;
- final Under HPO confirmation: `data_perp/artifacts/strict_r3_p8u_meta_final_lgbm_hpo_under100_selected_mc1_fullprehistory_aug25jul26_20260830_v1/`;
- final State HPO confirmation: `data_perp/artifacts/strict_r3_p8u_meta_final_lgbm_hpo_state_weighted_selected_mc1_fullprehistory_aug25jul26_20260830_v1/`.

## Enriched feature selection

These are surviving fields, not the raw candidate universe. Each target-specific 120-field contract was selected by randomized shallow strict-OOF ranker subspaces with cross-era inclusion, gain, and tail-SHAP evidence.

| Target contract | Kalman | Transition | Fast–slow | Synergy | Innovation / Mahalanobis | SHAP-derived |
|---|---:|---:|---:|---:|---:|---:|
| Under | 2 | 4 | 8 | 1 | 0 | 9 |
| Magnitude | 4 | 6 | 12 | 2 | 0 | 8 |
| State | 2 | 4 | 8 | 2 | 0 | 9 |
| Over | 2 | 4 | 8 | 1 | 0 | 14 |

Innovation and Mahalanobis features were considered but did not meet the cross-era bar. Inclusion-uplift is training-only evidence, never an inference feature. Kalman uses predeclared causal fast/slow pairs and was tuned through strict cross-fold feature/pair/block selection, not held-period fitting.

Contracts:

- Under: `data_perp/artifacts/strict_r3_p8u_meta_fullfeatures_successor_under100_20260830_v2/subspace_contract.json`;
- Magnitude: `data_perp/artifacts/strict_r3_p8u_meta_fullfeatures_successor_magnitude_20260830_v2/subspace_contract.json`;
- State: `data_perp/artifacts/strict_r3_p8u_meta_fullfeatures_successor_state_20260830_v2/subspace_contract.json`;
- Over: `data_perp/artifacts/strict_r3_p8u_meta_fullfeatures_successor_over50_20260830_v2/subspace_contract.json`.

## Model and weighting decisions

Model-family confirmation retained LightGBM rank-XENDCG. In the Under family, the established depth-4 sparse configuration reached `+135.73 bps/trade`; matched CatBoost QueryRMSE and YetiRank candidates reached `+131.16` and `+125.34`. CatBoost has no successor authority.

Selected weighting before final HPO: Under and Magnitude use uniform weights; State uses `w4_top20_class_balance_strong`; Over failed its economic screen. The final regular LightGBM bank contained nine predeclared configurations. GateProxy only nominated Top-3 plus uncertainty plus diverse controls; it did not select a winner.

## Final strict-MC1 results

All figures are constrained portfolios over February–July 2026.

| Family / candidate | Entries | Net EV / trade (bps) | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| **Retained Under control: depth-4 sparse** | **5,006** | **+135.73** | **+679,479** | **+68.26** | **+43.83** | **−32.75%** |
| New Under: depth-6 guarded | 4,906 | +132.94 | +652,221 | +66.13 | +43.57 | −29.00% |
| New Under: depth-5 sparse | 4,973 | +132.69 | +659,875 | +62.64 | +42.06 | −30.00% |
| New Under: depth-4 balanced | 4,989 | +130.77 | +652,409 | +63.41 | +41.08 | −28.00% |
| New Under: sparse control | 5,142 | +130.75 | +672,341 | +65.13 | +44.31 | −28.00% |
| New Under: depth-3 sparse | 5,068 | +128.38 | +650,621 | +63.94 | +42.86 | −33.00% |
| Retained State control: depth-5 capacity | 5,101 | +130.64 | +666,383 | +64.90 | +44.48 | −32.85% |
| New State: depth-5 capacity | 5,066 | +131.54 | +666,387 | +63.23 | +43.38 | −31.00% |
| New State: depth-4 wide | 5,152 | +129.80 | +668,720 | +63.84 | +39.61 | −27.00% |

The best new Under candidate trails the retained Under control by `−2.79 bps/trade` and `−27,258 total bps`, with weaker worst-period performance. State gains only `+0.90 bps/trade` against its own prior control, essentially no total net bps, and loses worst-month/week stability. Neither advances.

## GateProxy and reproducibility

The support-qualified proxy is `GateProxy_P0_Ridge`. It is a cheap strict-OOF estimate of downstream MC1 usefulness and has no promotion authority. It nominated five trials per family: three high-score candidates, one high-uncertainty probe, and one descriptor-diverse control. Exact plans:

- `data_perp/artifacts/strict_r3_p8u_meta_final_lgbm_hpo_under100_mc1_plan_20260830_v1/`;
- `data_perp/artifacts/strict_r3_p8u_meta_final_lgbm_hpo_state_weighted_mc1_plan_20260830_v1/`.

Focused regression coverage passed for the final-LGBM-HPO guard, descriptor source/identity propagation, and immutable confirmation-plan materialisation.

## Next permitted work

Do not retune on these February–July results. A successor requires a new predeclared candidate family or feature contract and a later untouched evaluation block. The retained Under control is the required comparator.
