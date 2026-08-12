# TP6/SL4 co-firing/economic clusters and cluster-specific varying-coefficient GAM

## Decision

The requested two-stage experiment is implemented and run on the extended
long-side TP6/SL4 panel.  It fixes the two known defects:

1. clusters are selected from co-firing/economic structure rather than generic
   silhouette alone;
2. the cluster GAM target is the ordinary signed residual, while membership is
   only a sample weight/exposure variable.

The contract is balanced and differentiated on the development validation
block, but transport to 2025 is sparse.  The best GAM blend improves the
pooled top-5 net result from +2.57 to +7.15 bps/trade, while worsening the
monthly portability score and failing the broad promotion gate.  Keep it as a
research diagnostic; do not promote it yet.

## Data and causal schedule

- Long side only.
- Extended panel: April–November 2024 plus January–December 2025.
- 100 canonical meta structural-family fields were available.
- 313 causal/meta/regime context fields were available before cluster-specific
  selection.
- Discovery: April–September 2024.
- Development validation: October–November 2024.
- Frozen contract: refit on all pre-2025 rows after the development selection.
- OOF evaluation: January–December 2025, 10,224 rows.
- Each held month uses only prior rows whose labels have matured before the
  held-month start.

## Stage 1 — co-firing/economic contract

For each family pair, the similarity combines:

- activation co-firing Jaccard;
- normalized pointwise mutual information (NPMI);
- contribution-profile coherence;
- train-only economic coherence of conditional residual effects.

Candidate clusterings were evaluated for:

- within-cluster compactness;
- contribution-mass balance;
- held-out active/inactive residual differentiation;
- validation support and temporal transport;
- silhouette only as a secondary geometric diagnostic.

The old all-family attempt was rejected: its nominal best K=4 cluster held
88.1% of development mass and the other clusters had zero support in the
validation block.

The corrected contract first restricted the input to 13/100 families with at
least 5% activation in both discovery and validation.  It selected K=4:

| Candidate K | Silhouette | Max mass | Min mass | Balance | Validation support | Valid |
|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 0.116 | 38.6% | 13.5% | 0.924 | 100% | yes |
| 5 | 0.109 | 38.6% | 4.6% | 0.849 | 100% | yes |
| 6 | 0.114 | 38.6% | 4.6% | 0.893 | 100% | yes |
| 7 | 0.033 | 25.8% | 4.6% | 0.948 | 85.7% | yes |
| 8 | 0.010 | 25.8% | 4.6% | 0.940 | 87.5% | yes |
| 9 | 0.008 | 24.1% | 1.7% | 0.918 | 88.9% | no minimum-mass gate |

The selected four clusters contain 2, 4, 5, and 2 family fields.  Mean
within-cluster similarity is 0.405–0.487; economic coherence is 0.903–0.973.
Development-validation active-minus-inactive residual differentiation is:

| Cluster index | Active rows | Mean membership | Active-minus-inactive residual |
|---:|---:|---:|---:|
| 0 | 414 | 0.175 | +11.10 bps |
| 1 | 694 | 0.191 | −11.42 bps |
| 2 | 112 | 0.024 | −25.30 bps |
| 3 | 1,217 | 0.461 | −3.40 bps |

This demonstrates that the contract is balanced and economically
differentiated on the development validation block.  It does not guarantee
that the same clusters will recur in 2025.

## Stage 2 — cluster-specific varying-coefficient GAM

For cluster `k`, the model is:

```text
R_i = exact_net_bps_i - base_expected_bps_i
Rhat_ik = alpha_k + exposure_ik × (beta_k + delta_k(X_i))
```

where:

- `exposure_ik` is the cluster's absolute path/contribution mass;
- `membership_ik` is used as the sample weight;
- `delta_k(X)` is an additive spline over up to 12 causal context fields;
- the target is ordinary signed residual bps, never membership × residual;
- no generic base-score-only smooth terms are added to the GAM.

Each cluster's context fields are selected from the 313-field causal/meta pool
using prior training rows and weighted binned MI.  The model is a 3-knot,
degree-2 spline basis with Ridge(alpha=20).  Predictions are combined by the
held-row membership-weighted average of cluster residual predictions.

Tested correction strengths:

```text
base_expected_bps + gamma × cluster_vcgam_residual
gamma ∈ {0.25, 0.50, 1.00}
```

## OOF cluster support and own metrics

The frozen contract's 2025 transport is weak:

| Cluster | Active 2025 months | Mean active rows/month | Mean active net | Mean active residual IC |
|---|---:|---:|---:|---:|
| 0 | 0/12 | 0 | unavailable | unavailable |
| 1 | 4/12 | 182 | −54.07 bps | −0.005 |
| 2 | 4/12 | 164 | −52.08 bps | +0.057 |
| 3 | 2/12 | 123 | −55.22 bps | +0.029 |

Only 10 of 48 cluster-month cells had meaningful held-row support.  The GAM
therefore has a legitimate cluster-specific target and causal fit, but the
structural exposures themselves do not recur reliably across 2025.

## Pooled global economics

| Arm | Top 0.5% | Top 1% | Top 2% | Top 5% | Top 10% | Rank IC |
|---|---:|---:|---:|---:|---:|---:|
| Base expected-bps control | −98.12 | −106.56 | −32.76 | **+2.57** | −8.94 | 0.064 |
| GAM gamma 0.25 | −71.70 | −91.09 | −50.47 | **+7.15** | **−1.78** | 0.068 |
| GAM gamma 0.50 | −67.82 | −69.93 | −55.21 | −5.19 | −5.89 | 0.070 |
| GAM gamma 1.00 | **−8.16** | **−18.47** | −63.51 | −16.60 | −17.41 | 0.070 |

Monthly top-5 stability:

| Arm | Mean | Median | Positive months | Worst month | Portability score |
|---|---:|---:|---:|---:|---:|
| Base | +2.58 | +6.45 | 6/12 | −167.81 | −208.48 |
| GAM gamma 0.25 | −1.26 | +9.95 | 7/12 | −167.81 | −216.10 |
| GAM gamma 0.50 | −2.29 | −6.99 | 5/12 | −167.81 | −231.85 |
| GAM gamma 1.00 | −1.07 | −29.19 | 5/12 | −167.81 | −264.12 |

The gamma-0.25 arm is the only one improving pooled top-5 and top-10, but its
mean monthly EV and portability score are worse than the base control.  The
full-strength GAM has attractive narrow-tail results but poor broad-tail
ranking.

## Correctness checks

- 10,224 OOF rows scored.
- Candidate IDs unique; no duplicates.
- All output scores finite.
- Contract frozen before 2025 evaluation.
- Held 2025 outcomes not used for contract or context selection.
- Membership is not multiplied into the target.
- Membership is used only as exposure/sample weight and aggregation weight.
- Target-like context fields selected: zero.
- Global ranking occurs after score generation.

## Decision

The architecture is now correctly separated:

1. co-firing/economic clustering produces a frozen structural contract;
2. only that contract's exposures enter the cluster-specific varying GAM;
3. the GAM predicts ordinary residual economics around the cluster contribution;
4. membership is a weight/exposure variable, not supervision.

The experiment does not yet pass the transport/promotion gate.  The next
repair should address the source path contract—why only 10/48 cluster-months
have usable 2025 support—before further GAM tuning.  More GAM flexibility would
otherwise be fitting sparse or absent structural states.

