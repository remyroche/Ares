# Regime execution-plan result — 2026-08-03

## Terminal decision

```text
REGIME_INFRASTRUCTURE_VALID
PRIMARY_K3_K4_K5_NO_INTRINSICALLY_FEASIBLE_GENERATOR
REGIME_DIRECT_FEATURE_USE_NOT_VALIDATED
REGIME_PRIOR_NOT_PROMOTED
REGIME_TRUST_SHRINKAGE_REJECTED
```

This is a controlled negative result, not a selection of the least-negative
arm.  No generator or utilization mode advances into the residual stack.

## Implemented contract

The primary regime generator now supports the predeclared funnel:

| Generator | Contract |
|---|---|
| G0 | fixed K=5 |
| G1 | fixed K=4 |
| G2 | fixed K=3 |
| G3 | K=5, then a train-only merge of the lowest-soft-occupancy component into its nearest centroid when it is below 2% support |

All models are fitted before their chronological OOF quarter.  Coordinates are
causally matched to the previous frozen fold only when effective K is the
same.  Each row carries the standard soft posterior and phase outputs plus
compact within-state position fields: distance to every centroid, assigned
centroid distance, radius percentile, boundary margin, and a forward-only
centroid-distance velocity.  Raw state IDs remain diagnostics; only aligned
soft memberships are considered in the membership control/prior.

The four uses tested are:

| Mode | Implementation |
|---|---|
| U0 | direct causal residual mapping with primary, transition, position and/or leverage bundles |
| U1 | side × aligned-soft-state/phase residual prior, hierarchically shrunk to the pooled residual prior |
| U2 | monotonic trust shrinkage of the **base-to-residual correction** back toward the causal base score |
| U3 | U1 followed by U2 |

The trust implementation specifically does not shrink a final expected-net
score toward a side mean.  It preserves the base score and shrinks only the
residual delta.

## Intrinsic generator result

No candidate clears all label-free support and alignment gates.

| Generator | Effective K | Mean stability objective | Minimum occupancy | Intrinsically feasible |
|---|---:|---:|---:|---|
| G0 K=5 | 5 | 0.678 | 0.000 | no |
| G1 K=4 | 4 | 0.717 | 0.000 | no |
| G2 K=3 | 3 | 0.752 | 0.002 | no |
| G3 K=5 merge | 4–5 | 0.682 | 0.003 | no |

The 2% floor is not being applied mechanically to a genuine recurring crisis
state.  The explicit rare-state audit finds no qualifying exception: the rare
components have zero independent six-hour episodes in their rare folds.  G3
also changes effective K from four to five across folds, so its posterior
coordinates cannot retain aligned semantic meaning.  The K=3/4/5 split audit
shows the extra components repeatedly create children with negligible support
or no durable episode, not a repeated additional market mechanism.

## Matched OOF economics

The population is 237,246 exact residual-OOF candidates from 2023-09 through
2024-12.  Evaluation is one pooled global top-k after each causal common-bps
map—not a per-timestamp or per-side selection.  The top-10 baseline is
−107.95 net bps/trade.

The best pooled direct arm is G3 `primary + transition` at −99.35 bps, a
+8.59 bps uplift with +0.0010 net rank-IC.  It is **not valid** because G3 is
intrinsically infeasible and its worst transport result is −10.62 bps below
its own baseline; its worst month is −68.16 bps below baseline.  It must not
be promoted.

The valid-looking low-K alternatives do not repair this:

- G2 `primary + transition`: −104.48 bps, +3.47 bps pooled, but −0.00066 IC
  and −12.72 bps on its worst transport split.
- G1 `transition`: −106.06 bps, +1.89 bps pooled, but slightly lower IC and
  a −2.04 bps worst transport result.
- Leverage-only is non-incremental: −108.23 bps, despite a tiny IC increase.

Conditional bundle removal confirms that leverage is not a robust third
bundle: from the G3 full direct arm, removing leverage improves top-10 net by
1.70 bps; from G2, it improves by 2.19 bps.  The combined stack therefore
does not beat its strongest conditional subset reliably.

The held-out conditional group-permutation MDA reaches the same conclusion.
Permuting the primary bundle costs 5.1–19.0 top-10 net bps and permuting the
transition bundle costs 1.5–11.7 bps, so both contain conditional information
inside the fitted full model.  Permuting leverage changes net by between
−0.65 and +0.47 bps and slightly *improves* IC in every generator.  Leverage
is redundant or harmful conditional on primary and transition context.  This
is descriptive only: the full model still fails the intrinsic and transport
gates, so MDA does not rescue it.

## Prior, trust, source, and membership controls

### U1: residual prior

The G1 prior is the closest diagnostic to useful transport:

| Transport | Baseline net bps | G1 prior net bps |
|---|---:|---:|
| 2023-Q4 → 2024 | −117.00 | −114.80 |
| 2024-H1 → H2 | −75.33 | −73.06 |

But pooled top-10 is only −107.14 bps (+0.81 bps) with a small IC decline, and
the worst month is −61.98 bps below baseline.  This is evidence that a
regime-conditioned **calibration prior** is a more plausible role than direct
reranking, but not evidence for promotion while the states themselves fail
support.

### U2/U3: trust shrinkage

Both trust modes are decisively harmful (about −155 to −158 net bps top-10).
The simple uncertainty product is therefore not a usable trust controller.
It should not receive threshold or HPO work until there is an intrinsically
stable state representation and an independently learnable residual-error
target.

### Continuous inputs versus memberships

Sixteen supported, nonconstant source dimensions have 100% candidate coverage;
one constant source field was explicitly excluded.  Continuous source
dimensions are approximately baseline economics (−107.65 bps), while soft
memberships do not give a consistent gain.  Combining them is not incremental.

The descriptive era-classifier result explains why: for G0, soft memberships
have 0.340 rank correlation with the calendar-era label versus 0.086 for the
source dimensions.  The states are largely an era-compression device, not a
transportable within-era ordering representation.

## Next admissible work

Do not add more latent systems or tune the direct arms.  The first repair is
intrinsic: replace the diagonal-GMM split mechanism with a support-constrained
model that either has a stable K=2/3 state set or encodes a verified rare state
through explicit episode criteria.  Only after one generator passes all
support/alignment gates should the G1-style side-soft-state residual prior be
retested.  The direct ranker, leverage bundle, and monotonic trust shrinkage
should remain frozen as rejected controls.

## Artifacts

- `data_perp/artifacts/regime_execution_g{0,1,2,3}_*_20260803_v1/`
- `data_perp/artifacts/regime_execution_plan_20260803_v1/`
- `data_perp/artifacts/regime_generator_state_support_20260803_v1/`
- `data_perp/artifacts/regime_bundle_mda_20260803_v1/`

The artifact manifests contain source hashes, the causal train-end/availability
contract, per-month results, per-side and per-phase top-k attribution,
within-state IC/spread/composition, transport, support/episode diagnostics,
and K-split diagnostics.
