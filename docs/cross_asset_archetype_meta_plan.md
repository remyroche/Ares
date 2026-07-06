# Cross-Asset Archetype Models Feeding the Meta Layer

## Objective

Build cross-asset representation models whose outputs feed the existing meta model layer.

The goal is not to maximize global average utility directly. The goal is to generate latent market dimensions that are useful across the archetype system: a latent can be accepted if it materially helps at least one sufficiently supported archetype x side cell, even if its global-average effect is muted.

The final system should answer:

- Which live-predictable market dimensions explain archetype-conditional outcomes?
- Which archetypes become more learnable when these dimensions are available?
- Which dimensions help de-risk dirty-positive, bad-MAE, timeout, or lower-tail states?
- Which dimensions improve missed-opportunity detection or score calibration by side?
- Which dimensions are stable enough to feed train_meta?

Candidate outputs are meta features only. They are not hard gates and not standalone trading policies. Promote only if they improve OOF train_meta diagnostics and downstream replay versus the non-representation baseline.

## Core Principle: Shared Latents, Archetype-Conditional Acceptance

Do not start with separate full models per archetype. Many archetype x side cells are small, and standalone specialists risk overfit, incomparable scores, and unstable allocation.

Instead:

1. Train global/shared representation models on broad cross-market features.
2. Export shared latent dimensions, OOD/error features, and risk scores.
3. Evaluate each latent/output by archetype x side.
4. Accept a latent if it has strong, stable, economically meaningful value in at least one supported archetype x side cell.
5. Feed accepted latents to the global meta model.
6. Only later test lightweight archetype-specific residual heads or calibration layers where support and OOF evidence justify them.

This is not a global-average filter. A dimension can be useful if it identifies, for example:

- long dirty-positive states,
- short liquidity-stress risk,
- high timeout regimes,
- cross-asset fragility pockets,
- missed opportunity pockets,
- June-like adverse-path states.

## Safe Archetype Contract

Only live-predictable archetypes may be model inputs:

- GMM posteriors
- GMM entropy
- Mahalanobis distance
- AE reconstruction error
- cluster speed / acceleration
- feature-derived regime families
- long mixed / run-entry splits built from pre-entry AE/GMM/context features

Hard GMM cluster id is live-predictable, but it should be treated carefully:

- use hard cluster id freely for diagnostics, grouping, and reliability tables;
- prefer posteriors, entropy, distance, reconstruction error, speed, and acceleration as model inputs;
- if hard cluster id is fed to meta, make it an explicit ablation against a soft-only variant.

Outcome/path archetypes are targets and diagnostics only. They must not be used as decision-time inputs.

## Cross-Market Feature Universe

Use cross-market and cross-asset features already present in `extreme_price_movements/`.

Candidate prefixes include:

```text
q_tail_
width_
tail_
asym_
iqr_
pct_assets_
cs_
btc_
eth_
eth_btc_
xs_dispersion_
state_spectral_
xasset_
mkt_
eig_
market_breadth_
market_dispersion_
xasset_mkt_
market_index_
cross_asset_
median_asset_
top_decile_asset_
cross_asset_correlation_
avg_pairwise_corr_
```

Also include feature families from `extreme_price_movements/unsupervised_regime_learning/`, provided they are decision-time safe.

## Shared Preprocessing Contract

Before feeding features into any LGBM, NN, AE, or control model:

1. Fit robust scaler on train fold only.
2. Transform validation/live with the frozen scaler.
3. Clip standardized values, default `[-8, 8]`.
4. Add missingness indicators.
5. Preserve feature-family metadata.
6. Downcast to `float32` where safe.
7. Cache preprocessed matrices and fold splits.
8. Avoid repeated pandas operations inside HPO loops.

Reject a candidate feature set if:

- NaN/inf remains after preprocessing.
- Latent dimensions collapse.
- Fold scaler behavior is unstable.
- Feature coverage is too sparse.
- Feature is not decision-time safe.
- Validation/live uses a non-frozen scaler.

## Initial Model Families

Start with two representation families, not five.

### Model A: Cross-Market LGBM Challenger

Purpose:

Measure whether cross-market features add incremental predictive value beyond base/meta context.

Inputs:

- cross-market features
- base model predictions
- side
- live-predictable archetype/GMM/AE features

Targets:

Use train_meta-compatible labels and weights, with extra reporting on:

- utility / exec margin
- clean exec
- full-path bad-MAE
- timeout
- dirty-positive
- lower-tail utility
- missed oracle opportunity
- path-order quality

V1 exports:

```text
cross_lgbm_exec_margin_score
cross_lgbm_bad_mae_score
cross_lgbm_timeout_score
cross_lgbm_dirty_positive_score
```

Later exports, only if V1 proves value:

```text
cross_lgbm_residual_pred
cross_lgbm_rank_score
cross_lgbm_tail_risk_score
cross_lgbm_missed_oracle_score
cross_lgbm_feature_importance_by_fold
```

If Model A includes base predictions, training rows must use OOF base predictions and validation/live rows must use frozen prior-fold or final frozen base predictions.

### Model B: Compact Group-Wise Denoising AE

Purpose:

Learn market-state/OOD/fragility representations from cross-market features.

Architecture:

```text
family_encoder_g:
    input_dim_g -> hidden_g -> family_z_g

global_encoder:
    concat(family_z_g) -> hidden_global -> market_z

decoder:
    market_z + family_z_g -> reconstruct family_g
```

V1 exports:

```text
market_z_0...market_z_k
market_ae_recon_error
market_ae_recon_error_pct
market_ae_mahalanobis_diag
ae_tail_error_pct
ae_breadth_error_pct
ae_dispersion_error_pct
ae_btc_eth_error_pct
ae_corr_spectral_error_pct
ae_xasset_error_pct
```

Later exports:

```text
market_ae_nearest_train_percentile
market_ae_entropy_or_uncertainty
market_z_speed
market_z_acceleration
family_error_speed
family_error_acceleration
```

AE candidates advance only if the exported features improve downstream residual/risk proxies. Reconstruction loss alone is not promotable evidence.

## OOF / Frozen Representation Contract

Representation outputs fed to train_meta must be OOF or prior-fold:

- train rows receive representation predictions from models that did not fit those rows;
- validation rows receive predictions from models fit only on earlier/train folds;
- scalers, feature selection, AE weights, LGBM weights, clusterers, and thresholds are frozen before scoring validation rows;
- final live inference uses frozen artifacts and the same feature list/preprocessing contract.

In-sample representation outputs may be used for diagnostics or final fitting after validation is complete, but not as evidence of meta improvement.

## Archetype-Conditional Acceptance

Evaluate every candidate output by:

```text
archetype x side
GMM cluster x side
month x archetype x side
source_family x side
bad-MAE environment
timeout environment
lower-tail environment
cross-sectional fragility state
```

For each latent/output, compute:

- top10/top20/top30 clean precision lift
- exec margin / EV lift
- full-path bad-MAE reduction
- timeout reduction
- dirty-positive reduction
- MFE-before-MAE-1R lift
- MAE-1R-before-MFE-1R reduction
- max adverse before MFE reduction
- underwater bars / underwater fraction reduction
- CUSUM-good-first lift if available
- CUSUM-bad-first reduction if available
- policy-simulated EV lift
- lower-tail utility lift
- oracle recall lift
- score monotonicity
- selected-row concentration
- support count and month coverage

Accept a latent if:

```text
supported_cells_with_positive_value >= 1
and effect_size_in_best_cell >= threshold
and month/fold stability passes
and random-control-adjusted score > margin
and no catastrophic degradation in other major cells
```

Recommended minimum support:

```text
cell rows >= 100 train
cell rows >= 30 validation
cell appears in >= 2 months or folds
train clean/positive examples >= 10
validation clean/positive examples >= 5
max single-asset share <= 80%
max single-week share <= 80%
```

For smaller cells, keep the latent shadow-only unless the effect is very large and replicated.

## Promotion Objective

Use a cell-aware promotion score, not a global average:

```text
LatentPromotionScore =
    + 0.20 * best_supported_cell_value
    + 0.15 * weighted_positive_cell_value
    + 0.15 * path_order_value
    + 0.15 * tail_risk_detection_score
    + 0.10 * missed_opportunity_score
    + 0.10 * stability_score
    + 0.10 * useful_novelty_score
    + 0.05 * interpretability_score
    - 0.20 * worst_major_cell_degradation
    - complexity_penalty
    - random_control_penalty
```

Where:

- `best_supported_cell_value` captures strong value in at least one archetype x side cell.
- `weighted_positive_cell_value` rewards broader usefulness without requiring global average dominance.
- `path_order_value` captures MFE-before-MAE improvement, MAE-before-MFE reduction, underwater duration reduction, and CUSUM-good-first improvement where available.
- `worst_major_cell_degradation` blocks latents that help one tiny pocket while damaging large cells.
- `useful_novelty_score` requires novelty to be predictive or risk-useful, not merely different.

## Random-Control Requirement

Every latent/output must beat matched controls:

```text
Z_perm
Z_block_perm
Z_random_walk
Z_noise_ar1
```

Control-adjusted score:

```text
control_adjusted_score =
    candidate_score
  - median(control_scores)
  - 0.5 * std(control_scores)
```

If a latent does not beat controls, keep it shadow-only.

## HPO Staging

Use staged HPO to avoid overfitting and downstream retraining cost.

### Stage 0: Sanity

Reject candidates with:

- leakage
- NaN/inf
- collapsed latent variance
- unstable scaler
- excessive autocorrelation without useful transition sensitivity
- insufficient feature coverage

### Stage 1: Intrinsic + Tail Separability

For AE:

- reconstruction stability
- latent variance
- latent fold stability
- OOD/tail separability
- transition sensitivity

For LGBM:

- top-k clean precision
- bad-MAE PR-AUC/lift
- timeout lift
- residual IC

Keep top 25-40%.

### Stage 2: Archetype-Conditional Proxies

Run cell-aware diagnostics:

- archetype x side top-k lift
- bad-MAE reduction
- timeout reduction
- MFE-before-MAE lift
- MAE-before-MFE reduction
- underwater duration reduction
- policy-simulated EV lift
- missed opportunity lift
- random-control comparison

Keep top 10-15%.

### Stage 3: Cheap Portfolio Proxy

Run:

- rank-weighted exec margin
- top-bottom bucket spread
- de-risking utility
- turnover penalty
- month/fold stability
- side x archetype diagnostics

Only after Stage 3 should outputs be considered for train_meta ablation.

## Meta-Model Ablation

Train meta with ablations:

```text
Baseline:
    existing meta features only

A:
    baseline + accepted cross_lgbm outputs

B:
    baseline + accepted AE/OOD outputs

C:
    baseline + accepted outputs from A+B

D:
    baseline + accepted outputs + lightweight residual/calibration heads where justified
```

Do not train standalone archetype-specific meta models initially.

Specialist logic is limited to:

- side-specific calibration
- archetype-specific residual correction
- threshold adjustment features
- small veto/agreement heads for high-support cells

Promote a specialist only if:

- support is sufficient
- OOF improves
- bad-MAE or timeout decreases
- lower-tail utility improves
- score monotonicity improves
- improvement is stable across folds/months

## Versioned Implementation

### V1: Representation Acceptance Framework

Inputs:

- cross-market feature block
- side
- OOF/frozen base score and base rank features
- AE/GMM soft descriptors
- source/archetype reporting buckets
- labels/outcomes for training and diagnostics only

Models:

- Model A: cross-market LGBM risk/exec challenger
- Model B: compact group-wise denoising AE

Candidate outputs:

```text
cross_lgbm_exec_margin_score
cross_lgbm_bad_mae_score
cross_lgbm_timeout_score
cross_lgbm_dirty_positive_score
market_z_0...market_z_k
market_ae_recon_error
market_ae_mahalanobis
family_recon_error_tail
family_recon_error_breadth
family_recon_error_dispersion
family_recon_error_btc_eth
family_recon_error_corr_spectral
family_recon_error_xasset
```

Diagnostics:

- archetype x side
- month x archetype x side
- source_family x side
- spread bucket x side
- bad-MAE environment
- timeout environment

V1 goal: prove the acceptance machinery works and identifies at least one useful shared latent/output.

### V2: Meta Ablation

Run:

```text
M0: baseline meta
M1: baseline + accepted cross_lgbm outputs
M2: baseline + accepted AE/OOD outputs
M3: baseline + accepted cross_lgbm + AE/OOD outputs
```

Evaluate:

- OOF train_meta metrics
- top10/top20/top30 EV and clean precision
- bad-MAE
- timeout
- lower-tail utility
- oracle recall
- path-order metrics
- side split
- archetype x side split
- monthly stability

Promote only if an ablation improves the actual meta system, not merely proxy diagnostics.

### V3: Lightweight Residual / Calibration Layers

Only after V2 proves value, test:

- side-specific calibration
- archetype-specific residual correction
- small agreement/veto features for high-support cells

Do not train standalone per-archetype meta models initially.

## Final Done Criteria

The plan succeeds only if at least one representation family improves the meta system after promotion.

Pass criteria:

1. Candidate outputs beat random/autocorrelated controls.
2. Candidate outputs improve frozen-teacher residual proxies.
3. Candidate outputs improve at least one supported archetype x side cell.
4. Candidate outputs do not catastrophically degrade major cells.
5. Candidate outputs improve OOF train_meta performance.
6. Candidate outputs improve portfolio replay.
7. Candidate outputs reduce bad-MAE, timeout, lower-tail risk, adverse path-order metrics, or improve oracle recall.
8. Long and short behavior is reported separately.
9. No feature leakage is detected.
10. Complexity is justified by downstream value.

Fail if:

- representation loss is good but downstream value is absent
- improvement is only in-sample
- improvement is dominated by one asset/month without replication
- candidate features do not beat controls
- candidate worsens lower-tail or bad-MAE materially
- candidate is redundant with existing features
- candidate is too expensive relative to marginal gain

Main rule:

```text
Promote shared latent market dimensions only when they create OOF downstream value in the meta layer.
They do not need to improve the global average first, but they must help at least one supported archetype x side cell without damaging the rest of the book.
```
