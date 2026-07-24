# Leakage Prevention Rules

## 1. Temporal And Alignment Leakage

Features use only data observable at decision time. Targets, barrier paths, MFE,
MAE, timeout, stop, and realized utility begin after that point. Use causal as-of
joins for 15m, hourly, funding, OI, orderbook, and market data.

Purge train rows whose future label intervals overlap validation/OOS intervals.
Apply an embargo based on the maximum target/execution horizon.

## 2. Fitted Transform Leakage

Fit imputers, encoders, calibration maps, cluster priors, and drift references
on the rows permitted by their training contract. OOS rows receive frozen
transforms.

The production AE/GMM representation is fitted exactly once per model cycle,
at feature-selection/HPO time, from sampled beginning/middle/end rows of the
designated cycle reference period. The exact scaler/AE/GMM state and input order
are reused for all growing base/meta windows, final refits, replay, and
inference. This may expose the unsupervised representation to later covariate
distributions, so it is representation-selection leakage and must be disclosed.
It may not use outcomes, labels, or future-derived weights. Do not describe
those rows as an untouched final test of representation discovery.

## 3. Feature Selection And HPO Leakage

Feature selection and HPO may use their designated validation samples, but those
rows are no longer an untouched final test. In the standard pipeline, run them
once on the largest authorized training fold using beginning/middle/end samples,
then freeze features and parameters for later OOS windows.

Target-derived sample weights are allowed only in training loss. They may not
become row features or affect OOS rank normalization through future outcomes.

## 4. Archetype And Meta Leakage

Outcome signatures may define training archetypes or meta targets. At inference,
use only cluster assignments or classifiers computable from pre-entry features.
Train-derived outcome priors must be frozen and support-weighted.

Meta training must consume OOF/frozen base scores. Same-fold fitted base scores
are not valid meta inputs.

## 5. Cross-Asset Leakage

Use only the point-in-time available universe. Future listings, full-period
liquidity screens, and normalization over future rows are prohibited.

## 6. Policy And Portfolio Leakage

Optimize thresholds, geometry, sizing, calibration, recent-performance rules,
and portfolio limits on training/validation folds only. Compile metrics from
each fold's non-training rows. Freeze the selected policy before replaying the
next period.

Recent hit-rate or EV surprise at timestamp `t` may use only outcomes resolved
before `t`.

Residual autocorrelation and hit-rate surprise must be generated sequentially.
For each row, exclude its own outcome and every still-open/unresolved prior row.
Expected hit-rate and EV baselines must come from frozen train-derived or
causally expanding history. Preserve both positive and negative surprise.

## 7. Required Audit Statement

For every reported result state:

- training, feature-selection/HPO, policy-fit, and evaluation dates
- model OOF versus policy OOS versus frozen replay status
- purge/embargo horizon
- base prediction provenance used by meta
- whether archetype/calibration priors were frozen
- whether costs and thresholds were selected on the reported period
- residual definition, lag/window, ordering, and OOS provenance
- hit-rate surprise half-life, support, and resolved-outcome cutoff
