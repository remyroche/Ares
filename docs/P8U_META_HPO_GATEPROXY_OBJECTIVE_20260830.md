# P8U Meta-HPO GateProxy Objective

## Purpose

This is the research objective for choosing which Meta HPO trials deserve an expensive full downstream MC1 replay. It is not a trading model, a new Meta head, an admission model, or a replacement for the full-stack promotion test.

The frozen relationship is:

```text
Meta HPO trial
→ strict-OOF cheap diagnostics
→ predicted downstream dual-MC1 gate benefit
→ Top-3 costly matched MC1 confirmation
→ fixed full-stack promotion decision
```

The versioned contract is [strict_r3_p8u_meta_hpo_gateproxy_objective_parent54_20260830_v1.json](../config/strict_r3_p8u_meta_hpo_gateproxy_objective_parent54_20260830_v1.json).

## Chosen objective

`GateProxy_P1_ElasticNet` predicts `dgate_shrunk`: the reliability-weighted benefit of a Meta replacement to the fixed dual MC1 `+50 bps` gate. It uses a strongly regularized ElasticNet model and 16 strict-OOF descriptors:

- residual IC and conditional MI given Base;
- Meta residual IC in the Base 5–10, 10–20, and 20–30% bands;
- Meta Top-1/Top-2 economics and candidate-only substitution economics;
- useful and false upgrade economics;
- Base/Meta rank relationship and correction magnitude;
- weekly downside stability and shallow-probe deltas.

The model was chosen before the unseen-trial test by maximizing its minimum supported grouped-holdout Spearman across target, loss, and era families. The artifact is [dgate_shrunk__P1_elastic_net.joblib](../data_perp/artifacts/strict_r3_p8u_meta_downstream_proxy_parent54_final_20260830_v2/models/dgate_shrunk__P1_elastic_net.joblib), SHA-256 `ccad070d07fd30c10497e6285eef13e7340c1e68b4f047d25cc808fb9512220a`.

The pre-holdout model-choice receipt is [here](../data_perp/artifacts/strict_r3_p8u_meta_gateproxy_preholdout_selection_parent40_20260830_v1/preholdout_gateproxy_choice.json).

`dgate_shrunk` is built from matched real dual-gate measures: admitted EV, utility per decision timestamp, >50/>100-bps precision, admitted volume, and weekly downside. The 1,000-draw bootstrap uses weekly blocks and applies reliability shrinkage. Weekly utility and volume are explicitly normalized per decision timestamp, matching the aggregate label scale.

## Why no PriorityProxy

The Meta replacement changes the Current MC1 coordinate and thus which rows pass the dual gate. The BCF coordinate remains the fixed priority coordinate. Accordingly, a separate PriorityProxy did not transport: its grouped and 14-trial unseen results were unreliable. It has zero HPO authority.

This is not a loss of useful scope. The proxy is used for the component Meta can actually affect: gate quality. Once a trial is shortlisted, full MC1 and the normal constrained portfolio replay determine the final economics.

## Evidence

Initial training used 40 diverse frozen-parent trials and six months of matched strict MC1 replays. The Gate ElasticNet achieved mean Spearman:

| Grouped holdout | Gate ElasticNet Spearman |
|---|---:|
| Target family | 0.662 |
| Loss family | 0.495 |
| Era | 0.504 |

The 14 remaining frozen-parent trials were predeclared as a disjoint holdout. Predictions were sealed before their MC1 outputs were joined.

| Unseen 14-trial result | Gate ElasticNet |
|---|---:|
| Spearman | 0.512 |
| Top-3 precision | 0.667 |
| True downstream winner in proxy Top-3 | Yes |
| Regret@3 | 0 |

After that test, the final research-only ElasticNet was refit on all 54 frozen-parent trials. Its all-data grouped CV is descriptive only; the 14-trial test above is the independent validation.

## Required use

1. Generate a candidate Meta trial’s strict-OOF target-free scores.
2. Build the same 16 descriptors after the held outcomes resolve.
3. Rank trials by `GateProxy_P1_ElasticNet`.
4. Send only the Top-3 to the fixed full MC1 replay.
5. Apply the existing full-stack promotion gates to those actual MC1 results.

No direct Meta metric, proxy threshold, or portfolio backtest alone may promote a trial.

## Scope limitation

This objective is validated only for the frozen `current_frozen` Under-F120 parent contract. Historic strict prehistory for the stable/context SHAP append contracts is not available, so leave-feature-contract-out validation is explicitly unsupported. Those feature contracts must remain exploratory and receive actual MC1 confirmation until compatible historical lineage is materialized.

## Core artifacts

- [Parent-40 training labels](../data_perp/artifacts/strict_r3_p8u_meta_proxy_downstream_labels_parent40_20260829_v2/)
- [Parent-14 sealed holdout predictions](../data_perp/artifacts/strict_r3_p8u_meta_proxy_parent14_holdout_scores_20260830_v1/)
- [Parent-14 holdout MC1 labels](../data_perp/artifacts/strict_r3_p8u_meta_proxy_parent14_holdout_labels_20260830_v1/)
- [Parent-14 falsification metrics](../data_perp/artifacts/strict_r3_p8u_meta_proxy_parent14_holdout_evaluation_20260830_v1/)
- [Final 54-trial proxy fit](../data_perp/artifacts/strict_r3_p8u_meta_downstream_proxy_parent54_final_20260830_v2/)

All of these artifacts are offline research only and have no live score, admission, execution, or portfolio authority.
