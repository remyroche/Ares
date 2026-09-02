# P8U Archetype Recovery — Router Recall Research

## Decision

**Do not promote.** The experiment leaves the current Router, Base, Meta, MC1, admission, portfolio, policy, and live stack unchanged. The representations are structurally valid, but their soft-membership probe does not beat the matched generic multiseed probe consistently enough in 2025 development. The frozen 2026 period was therefore not opened for model selection.

## Causal contract

This is a Router-only research path. It first seals a target-free candidate population and computes structural geometry from causal fields only. Structural selection has no access to Router scores/ranks, labels, realised policy outcomes, future-path data, or symbol identifiers. Labels are joined only after the structural representation and feature contract are frozen.

Rows with less than 90% source coverage are excluded rather than imputed. The available contracts retained 75 structural fields and 141 predictive fields. Training-only percentile scaling, winsorisation, correlation collapse and stability checks are refit within the permitted pre-held period only.

Probe targets are decision-time-ATR-normalised policy utility, either Huber regression or a six-class ordinal target: `<= -1`, `(-1, 0]`, `(0, 0.5]`, `(0.5, 1]`, `(1, 2]`, `> 2 ATR`. Timestamp LambdaRank was ablation-only. All reported Router-recall and economics remain realised policy-net **bps**.

## Frozen structural selections

| Held fold | Primary | Max mass | Minimum ESS share | 5% ESS gate | Median second membership | Effective components | Stability | Reconstruction MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 Q2 | NMF K5 | 23.1% | 62.4% | pass | 28.2% | 3.69 | 0.974 | 0.295 |
| 2025 Q3 | NMF K4 | 27.8% | 64.7% | pass | 30.0% | 2.98 | 0.831 | 0.293 |
| 2025 Q4 | NMF K4 | 27.6% | 65.4% | pass | 30.0% | 3.00 | 0.617 | 0.289 |

These pass the predeclared anti-collapse, overlap, structural-distinctness and adjacent-period stability gates. The constrained diagonal-GMM controls were also frozen from the same target-free discovery artifacts: K7/K8/K5 for Q2/Q3/Q4. They are controls only and cannot replace the NMF primary.

## Matched 10% rescue evaluation

The table holds the Router allocation at 50% and uses the probe to refill the other 50% of the same 10% selected budget. `Recall >50` is recall of realised policy outcomes above +50 bps; economic-mass recall is recall of total positive policy-net bps.

| Fold | Arm | Mean net bps | CVaR10 bps | Recall >50 | Recall >100 | Economic-mass recall |
|---|---|---:|---:|---:|---:|---:|
| Q2 | Primary soft NMF probe | +95.39 | -555.37 | 13.54% | 15.78% | 20.71% |
|  | C0 generic multiseed | +96.03 | -568.36 | **13.69%** | **15.95%** | **21.04%** |
|  | C4 constrained GMM | +79.60 | -578.17 | 12.99% | 14.96% | 19.94% |
| Q3 | Primary soft NMF probe | **+31.72** | -561.77 | 17.63% | 22.64% | **25.06%** |
|  | C0 generic multiseed | +19.67 | -590.28 | **18.11%** | **22.68%** | 24.76% |
|  | C4 constrained GMM | +28.86 | -541.81 | 16.59% | 21.44% | 24.06% |
| Q4 | Primary soft NMF probe | **+36.60** | -595.01 | 17.57% | 22.43% | 25.62% |
|  | C0 generic multiseed | +22.11 | -600.00 | **18.22%** | 22.33% | 24.84% |
|  | C4 constrained GMM | +32.87 | -590.07 | 17.13% | 21.76% | 24.99% |

The primary representation beats the constrained GMM control in every held fold, but C0 has higher `Recall >50` in Q2, Q3 and Q4. Q3 and Q4 improve mean economics versus C0, yet they do not establish the predeclared portable Router-recall uplift. That is insufficient to justify an outcome-selected 2026 test.

## Interpretation

The result falsifies the specific claim that causal structural archetypes add enough *incremental Router recall* beyond a generic high-capacity probe. It does not say that the new feature families are useless: the generic probe can exploit them, while membership-only performance is weak, showing that the soft archetype itself is not an adequate substitute for the predictive fields.

The appropriate next step is not a wider archetype search. Improve or constrain the generic probe under a newly predeclared development protocol, then test whether archetype membership adds value on top of that fixed baseline. Do not change or promote the live stack from this result.

## Reproducibility

- Frozen configuration: `config/strict_r3_p8u_archetype_recovery_20260830_v1.json`
- Q2 result: `data_perp/artifacts/strict_r3_p8u_archetype_recovery_development_q2_20260830_v5_frozengmmcontrol`
- Q3 result: `data_perp/artifacts/strict_r3_p8u_archetype_recovery_development_q3_20260830_v2_frozengmmcontrol`
- Q4 result: `data_perp/artifacts/strict_r3_p8u_archetype_recovery_development_q4_20260830_v2_frozengmmcontrol`
- Structural sources: `data_perp/artifacts/strict_r3_p8u_archetype_recovery_structural_q2_20260830_v4_overlapgate` and `data_perp/artifacts/strict_r3_p8u_archetype_recovery_structural_q3q4_20260830_v2_overlapgate`
- Unit and compilation checks: 17 tests passed; only synthetic NMF convergence warnings were emitted.

Earlier primary-only development runs are superseded by the listed C4-complete runs; they are retained as immutable research evidence but must not be used for the promotion decision.
