# Side-local prior-mapped residual audit — 2026-08-05

## Executive result

The repaired stack is implemented and replayed on three untouched transport
folds. The side-local, prior-resolved map is the better default. The residual
learner is not production-ready: it improves some broad tails relative to the
no-op map, but remains negative after the 100-bps cost and loses at the most
important top-1% tail. The correct current policy is therefore **use the
side-local mapped base score and reject the residual correction unless a later
portability gate passes**.

## What changed

- Base score is mapped separately for long and short with a monotone PAVA map,
  using only prior-resolved labels and the strict boundary
  `label_available_ts < decision_timestamp`.
- Residual target is exact H12 net bps minus that side-local causal map.
- The residual LambdaRank model is fitted per row and side, with a fixed
  residual contract, OOF residual calibration, an OOF lambda sweep
  `{0, .125, .25, .5, .75, 1}`, and region gates.
- Lambda/gates are selected on validation top-tail net economics against the
  no-op map; residual MSE is secondary. A correction is disabled when it does
  not beat the no-op in its validation region.
- Version 3 additionally joins a fixed store-backed regime/context ledger to
  the residual contract. The contract has 100 fields: seven frozen specialist
  outputs, base/map trust fields, six selected context fields, and 80 causal
  regime/transition/funding/OI/liquidity/volatility fields.

Artifacts:

- `data_perp/artifacts/gated_prior_mapped_residual_20260805_v3/`
- `manifest.json`, `base_map_manifest.json`, `residual_feature_contract.json`
- `predictions.parquet`, `metrics.parquet`, `gate_audit.parquet`

The residual contract hash is
`843ebd68761ea0e9f0ee09dd17c205d908367106cd0f758a6997c37b24e85906`.

## OOS pooled global results

All values are bps/trade; net subtracts the 100-bps cost exactly once.

| System | Top 1% gross / net | Top 5% gross / net | Top 10% gross / net |
|---|---:|---:|---:|
| No-op side-local map | +80.92 / **−19.08** | −3.58 / **−103.58** | +17.07 / **−82.93** |
| Regime-aware gated residual | +57.59 / **−42.41** | +21.22 / **−78.78** | +26.18 / **−73.82** |

The regime-aware residual therefore improves the no-op by +24.80 bps at top-5
and +9.10 bps at top-10, but loses −23.33 bps at top-1. None of the pooled
tails clears costs.

## Fold and side stability

| Fold | System | Top 1% net | Top 5% net | Top 10% net |
|---|---|---:|---:|---:|
| Jul–Aug | no-op | −64.05 | −134.19 | −104.61 |
| Jul–Aug | gated residual | −101.89 | −71.93 | −83.42 |
| Sep–Oct | no-op | −44.39 | −108.26 | −107.90 |
| Sep–Oct | gated residual | −47.91 | −109.45 | −107.88 |
| Nov partial | no-op | +131.00 | −27.89 | +16.31 |
| Nov partial | gated residual | +97.18 | −28.96 | +18.54 |

The November shift is decisive: the no-op captures a strong long-side regime,
while the residual correction gives most of it back. Side-local behaviour is
also asymmetric; the short side remains the main source of pooled negative
tails.

## Why the meta layer is worse than the base

1. **The residual is a harder target than opportunity.** It asks the model to
   predict conversion error after subtracting a noisy estimated value. The
   residual has much lower signal-to-noise than the base clear-vs-adverse
   ordering, so useful base rank information is easily destroyed by residual
   rank inversions.
2. **The pre-existing map was unstable across side/regime.** The repaired map
   removes the worst mapping mismatch, but the residual still has to transport
   across July, September/October and November. It does not.
3. **More fields add variance, not independent conversion information.** The
   specialist outputs and context variables are correlated, regime-shifted,
   and often weakly varying relative to the residual. LambdaRank can use their
   fold-specific noise to reorder the global top tail. Earlier frozen-input
   arms confirm this: `specialists_plus_selected_context` had pooled net
   top-5 −77.76 versus `specialist_heads_only` −90.69, while its rank IC was
   −0.025; adding context did not create stable positive ordering.
4. **Global top-k magnifies a few bad inversions.** Evaluation ranks globally
   after mapping, not per timestamp. A small number of high residual scores in
   the wrong side/regime can displace many reasonable base candidates.
5. **Validation-gated improvement is not enough.** The OOF gate improved v1
   substantially, but the November holdout still rejects portability. It is a
   valid negative result, not evidence that the residual is economically useful.

## Specialist target audit

The frozen specialists currently use `exact_h12_net_bps_gt_50`. Its prevalence
is not stable:

- long: ~21–27% in Jul–Oct, 40% in November;
- short: ~21–30% in Jul–Oct, 25% in November;
- mean net bps also shifts from roughly −70 to −110 in earlier long folds and
  to +2 in November, while November short is about −198 bps.

This target is therefore a candidate target-repair item. Before reusing the
specialists, compare it against a cost-aware, side/regime-transportable target
(for example a soft robust-clear margin or a small ordinal net-margin target)
and require target prevalence, oracle monotonicity and transport stability.

## Decision

- **Advance:** side-local prior-resolved base mapping.
- **Do not advance:** the residual correction or its full-input variants as a
  production policy.
- **Next required ablation:** repair/retest specialist targets, then test a
  deliberately small residual contract with shrinkage and a leave-one-regime-
  out gate. Keep the no-op map as the mandatory control.

