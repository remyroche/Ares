# P8U Meta GateProxy — 88-trial support-qualified successor

## Status

This is the successor **offline HPO/feature-screening** binding after the
frozen P3 Pairwise batch was confirmed through actual matched six-month MC1
replays.  It does not change Router50, F72 Base, Under F120, MC1 admission,
the portfolio auction, execution, or any live artifact.

The active successor binding is
[`strict_r3_p8u_meta_hpo_objective_binding_allcontracts88_20260830_v1.json`](../config/strict_r3_p8u_meta_hpo_objective_binding_allcontracts88_20260830_v1.json).
It resolves to `GateProxy_P0_Ridge`, SHA-256
`3d8d07385d72ac6002e5b0c7e5f3800b6a6cef363dfbfd7251bfedbfcb03adb3`.

## What was confirmed

The frozen all-contracts-81 P3 binding ranked a strict-OOF, descriptor-only
75-trial candidate bank.  Its exact proposal contained three highest predicted
gate-value trials, one high-surrogate-disagreement probe, and one
descriptor-diverse control.  All five then received independent target-free
score unions and separate Current/BCF six-complete-month MC1 maps under the
same dual `>= +50` bps gate and constrained portfolio contract.

The three new independently specified F128 trials were also scored and
confirmed.  F128 consequently reaches six labelled trials, the predeclared
minimum at which it may participate in feature-contract portability selection.
This is a support threshold, not a claim that F128 is universally superior.

| Contract | MC1-labelled trials |
|---|---:|
| Frozen Under F120 | 54 |
| SHAP-stable F123 | 19 |
| SHAP-context F125 | 9 |
| SHAP-combined F128 | 6 |
| **Joint ledger** | **88** |

The joint label receipt recomputes robust normalisation and the 1,000-draw
weekly-block bootstrap across all 88 trials:
[`strict_r3_p8u_meta_proxy_downstream_labels_parent54_append27_gateproxy5_f128support2_joint88_20260830_v1`](../data_perp/artifacts/strict_r3_p8u_meta_proxy_downstream_labels_parent54_append27_gateproxy5_f128support2_joint88_20260830_v1/).

## Successor model choice

Model choice remains the predeclared lexicographic rule: maximize the minimum
support-qualified grouped-holdout Spearman, then the supported mean across
target family, loss, feature contract, and era.  On the 88-trial refit, Ridge
wins maximin portability even though P3 Pairwise retains a higher mean.

| GateProxy | Minimum supported Spearman | Supported mean Spearman | Era Spearman | Feature-contract Spearman |
|---|---:|---:|---:|---:|
| **P0 Ridge — selected** | **0.520** | 0.607 | **0.525** | 0.649 |
| P3 Pairwise | 0.462 | **0.633** | 0.462 | **0.714** |
| P1 ElasticNet | 0.455 | 0.532 | 0.492 | 0.572 |
| P2 depth-2 GBDT | 0.092 | 0.428 | 0.473 | 0.707 |

The support-aware choice receipt is
[`strict_r3_p8u_meta_gateproxy_grouped_portability_allcontracts88_20260830_v1`](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_grouped_portability_allcontracts88_20260830_v1/).

## Required use

`GateProxy_P0_Ridge` may rank new Meta HPO and feature-contract candidates
into the usual Top-3 + uncertainty + diverse-control MC1 proposal.  It may not
promote a candidate automatically.  Every candidate still requires a fresh
strict target-free score receipt followed by a matched six-month dual-MC1
replay; only that evidence can advance the candidate.

The frozen P3 binding remains the historical authority for the candidate batch
it already selected.  It is not retroactively rewritten.

## Audit

The one-file completion receipt is
[`strict_r3_p8u_meta_gateproxy_completion_audit_20260830_v1`](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_completion_audit_20260830_v1/).  It verifies the 75-trial bank, exact five-trial P3 proposal, five completed MC1 receipts, 88 unique joint labels, three new F128 confirmations, six F128 total labelled trials, and the successor binding hash.
