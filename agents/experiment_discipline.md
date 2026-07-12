# ML Experiment Discipline

Experiments must isolate the proposed change and produce reproducible,
time-ordered economic evidence.

## 1. Fixed Contracts

Record run ID, dataset/feature store, label and geometry manifest, universe,
costs, model objective, selected features, parameters, seeds, train/eval dates,
and code revision. Keep all unrelated contracts fixed in an ablation.

When comparing meta arms, use the same base predictions and candidate rows.
When comparing policy arms, use the same base/meta scores and replay paths.

## 2. Time-Based Evaluation

Use expanding or rolling walk-forward splits. Purge overlapping label intervals
and apply an embargo based on the maximum future path used by the target.

Distinguish:

- training rows
- feature-selection/HPO validation rows
- model OOS rows
- policy-optimization validation rows
- untouched/frozen replay rows

Rows used for feature selection or HPO are not untouched final-test rows.

## 3. Feature Selection And HPO

For the current LGBM path:

- Run univariate and Relief pre-screening against each relevant archetype and
  retain a feature when it passes for at least one supported archetype.
- When MDA is enabled and archetype identity is available, score permutation
  value with the global, macro-archetype, and worst-supported-archetype top-k
  objectives. Record any global-only fallback in the selector diagnostics.
- Run later selection and HPO with explicit side handling.
- Run feature selection and HPO once on the designated largest training fold,
  using time-spread samples from its beginning, middle, and end.
- Freeze selected features and parameters for later growing OOS windows unless
  the experiment explicitly studies retraining them.
- Do not hard-code a feature count when the MDA pipeline can select it from its
  stopping/importance contract.

## 4. AE/GMM State

Fit AE/GMM once on authorized pre-OOS data and freeze it across subsequent folds.
Sample across multiple training subperiods. The current intended scale is about
15k AE rows and up to 100k GMM rows when available. Record actual support.

Do not select cluster count or semantics using the OOS months later reported as
evidence.

Base and meta experiments must preserve archetype identity. Meta experiments
must compare archetype-aware arms against a matched baseline meta model on the
same base top-30 candidate rows.

## 5. Metrics And Objectives

Prioritize the metrics used for trading:

- top-k net EV and precision, especially top 10/20/30%
- worst-week and worst-month top-k EV
- clean-positive versus dirty-positive separation in the selected tail
- stop, timeout, concentration, and side/archetype stability
- total net PnL, trades/day, and portfolio drawdown
- signed probability/economic residual mean and autocorrelation
- signed 3d/7d/14d hit-rate surprise with effective support

AUC is diagnostic, not the primary promotion metric.

Positive surprise and favorable residual structure are useful opportunity
signals, not values to discard. Negative surprise and persistent adverse
residuals are degradation signals. Report the full signed distributions.

## 6. Search Discipline

Report search breadth and all arms, not only the winner. Prefer hierarchical or
staged searches to unconstrained joint grids. Confirm winners on fixed contracts
and require improvement across relevant folds, not only aggregate PnL.

Do not repeatedly tune on the same final months and continue calling them an
untouched test. Promote only after frozen replay or a later unseen window.
