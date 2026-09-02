# P8U learned downstream-value objective for Meta HPO

> **Historical P3 record.** The P3 Pairwise binding in this document selected
> and confirmed its August-30 batch.  For **new** Meta HPO and feature-contract
> screening, use the support-qualified 88-trial P0 Ridge successor documented
> in [P8U_META_HPO_GATEPROXY_88TRIAL_SUCCESSOR_20260830.md](P8U_META_HPO_GATEPROXY_88TRIAL_SUCCESSOR_20260830.md).
> P3 remains immutable historical evidence; neither binding has live authority.

## Decision

The Meta research objective is now a learned **GateProxy**, not a hand-weighted collection of IC, CMI, or Meta-native economic metrics.

```text
Meta HPO / feature-contract trial
→ target-free strict-OOF Meta scores
→ 16 resolved-history descriptor diagnostics
→ GateProxy_P3_Pairwise
→ small MC1 confirmation proposal
→ fresh matched six-month dual-MC1 replay
→ normal advancement decision
```

It answers one narrow, useful question: **which Meta trial is most likely to improve the downstream dual-MC1 admission gate?** It does not score live candidates, change the BCF priority coordinate, replace MC1, or promote a trial automatically.

The versioned objective is [strict_r3_p8u_meta_hpo_gateproxy_objective_allcontracts81_20260830_v2.json](../config/strict_r3_p8u_meta_hpo_gateproxy_objective_allcontracts81_20260830_v2.json). New Meta HPO and feature-contract work must use the explicit [v2 objective binding](../config/strict_r3_p8u_meta_hpo_objective_binding_allcontracts81_20260830_v2.json); the parent-54 ElasticNet remains historical evidence only.

## Ground truth

The training/falsification ledger contains 81 deliberately varied, strict-OOF Meta trials. Every trial was passed through the same fixed downstream contract:

- separate Current and BCF MC1 mappings trained on six complete prior months;
- prior-21-day shifts built only from resolved labels;
- real dual Current/BCF expected-policy-net gate of at least +50 bps;
- BCF mapping as the fixed auction priority coordinate;
- candidate policy outcomes joined only after target-free score receipts were sealed.

The Gate target, `dgate_shrunk`, combines the matched differences in admitted EV, utility per timestamp, >50/>100-bps precision, admission volume, and weekly downside. A 1,000-draw weekly-block bootstrap estimates uncertainty and reliability-shrinks noisy trial labels. Constrained-portfolio metrics are retained for confirmation, not used as the target.

| Feature contract | MC1-labelled trials | Status |
|---|---:|---|
| Frozen Under-F120 | 54 | portable test group |
| SHAP-stable F123 | 16 | portable test group |
| SHAP-context F125 | 8 | portable test group |
| SHAP-combined F128 | 3 | exploratory only; too small for selection evidence |

The parent and append ledgers were re-normalized jointly before fitting, so their downstream labels share one robust scale. See [joint label ledger](../data_perp/artifacts/strict_r3_p8u_meta_proxy_downstream_labels_parent54_append27_joint81_20260830_v1/).

## Descriptor vector

`GateProxy_P3_Pairwise` uses 16 target-free strict-OOF Meta diagnostics:

1. residual IC;
2. CMI conditional on Base;
3. IC in Base 5–10%, 10–20%, and 20–30% bands;
4. Meta Top-1 and Top-2 policy economics;
5. Top-1 and Top-2 candidate-only substitution economics;
6. false- and useful-upgrade economics;
7. Base/Meta rank correlation and median correction magnitude;
8. weekly Q10; and
9. fixed shallow-probe Top-2 and admitted-utility deltas.

The descriptors describe a trial after its held outcomes resolve; they are used only to choose later research trials, never to score a live opportunity.

The P3 coefficients are descriptive—not causal feature importance—but the strongest standardized signals in the completed 81-trial fit are Base-band 5–10% IC, Base-band 10–20% IC, false-upgrade economics, Base-band 20–30% IC, correction magnitude, Base/Meta rank relationship, and weekly Q10. This is useful direction for Meta research, not a license to replace the learned proxy with a new hand-weighted objective.

## Selected surrogate

The final shortlist surrogate is a strongly regularized pairwise logistic ranker, `P3_pairwise`, trained on reliability-weighted `dgate_shrunk`. It optimizes trial ordering, rather than pretending it can estimate an exact downstream bps delta.

Model artifact: [dgate_shrunk__P3_pairwise.joblib](../data_perp/artifacts/strict_r3_p8u_meta_downstream_proxy_allcontracts81_20260830_v2/models/dgate_shrunk__P3_pairwise.joblib), SHA-256 `248dd1c73a9934300af444ef9fd51410fc14fa3c3a52f9122d5b84077ef6a475`.

`PriorityProxy` remains diagnostic-only. Its era portability is weak and it has no HPO authority. This is consistent with the architecture: Meta changes Current/gate membership, while BCF remains the frozen auction-priority coordinate.

## Portability evidence

Model choice used grouped strict-OOF validation and maximized the worst *mean* Spearman across the four required axes, after excluding individual holdout groups with fewer than six trials.

| GateProxy P3 grouped holdout | Mean Spearman | Worst supported group Spearman |
|---|---:|---:|
| Target family | 0.818 | 0.725 |
| Loss family | 0.476 | 0.427 |
| Feature contract (F120/F123/F125) | 0.725 | 0.377 |
| Era | 0.474 | 0.304 |

The independent 14-trial parent holdout provides secondary corroboration: P3 Spearman was 0.574, its Top-3 contained the downstream winner, and regret at three was zero. It was not selected before that old holdout, so this supports the result but does not turn it into a new predeclared test.

The F128 contract is not treated as a valid generalization claim: its three trials are explicitly excluded from model selection, then retained only as an exploratory OOD condition.

## Required use

1. Create a new Meta trial and its target-free strict-OOF score ledger.
2. Once its outcomes resolve, build the standard descriptor receipt.
3. Run [score_strict_r3_p8u_meta_gateproxy_objective_v2.py](../scripts/score_strict_r3_p8u_meta_gateproxy_objective_v2.py).
4. Send the Top-3 GateProxy trials to a fresh matched six-month MC1 replay. When capacity permits, add the one high-uncertainty and one diverse-control proposal to improve the proxy’s training coverage.
5. Advance only using actual MC1/admission/portfolio evidence and the normal promotion gates.

For Meta feature selection, use the same sequence with `Δ GateProxy` to screen additions, removals, and swaps. Feature contracts may never be accepted solely from proxy scores.

## Artifacts

- [81-trial proxy fit](../data_perp/artifacts/strict_r3_p8u_meta_downstream_proxy_allcontracts81_20260830_v2/)
- [support-aware model choice](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_grouped_portability_allcontracts81_20260830_v3/)
- [reusable GateProxy scorer](../scripts/score_strict_r3_p8u_meta_gateproxy_objective_v2.py)
- [append-contract MC1 receipts](../data_perp/artifacts/strict_r3_p8u_meta_proxy_selected_mc1_append27_20260830_v1/)
- [support-aware selector](../scripts/select_strict_r3_p8u_meta_gateproxy_from_grouped_portability_v2.py)
- [completion audit](../data_perp/artifacts/strict_r3_p8u_meta_hpo_objective_completion_audit_20260830_v2/completion_audit.json)

## Non-authority boundaries

- No proxy score is a live feature, trade-admission signal, portfolio signal, or execution signal.
- No proxy score can promote a Meta trial.
- No proxy score replaces MC1, its dual gate, or a constrained portfolio replay.
- The learned objective is an offline research screening tool; MC1 is the ground-truth evaluator.
