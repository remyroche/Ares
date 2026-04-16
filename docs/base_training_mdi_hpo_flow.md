# Base Training MDI + HPO Flow

This note documents how `extreme_price_movements/run_pipeline.py base_training`
handles feature selection (MDI) and hyper-parameter optimization (HPO).

## Stage order

The base pipeline stage is:

1. MDI feature selection
2. ExtraTrees base HPO
3. Final base model training

Implementation reference: `run_base_hpo_step` in
`extreme_price_movements/pipeline_steps.py`.

## Scope granularity

MDI and HPO are run per concrete base scope:

- `strategy_id`
- `horizon`
- optional historical dataset suffix variants (`""`, `_tight`, `_wide`) when
  present

## MDI selection behavior

`mdi_feature_selection_v3` is used for base selection. The selector combines
multiple quality signals, including:

- support in top-ranked subsets
- global importance
- fold stability
- frequency support
- interaction support

A final weighted score is then used to rank and keep features.

## HPO objective (base ExtraTrees path)

Base HPO uses `run_base_extratrees_hpo`.

Per trial:

- runs purged cross-validation
- computes top-30% IC per fold from model probabilities
- aggregates fold IC via mean and standard deviation

Optimization target:

- maximize `mean_ic - std_ic`

This favors high signal at the decision tail while penalizing unstable
configurations.

## Persistence and reuse

Selected features and HPO outputs are persisted under artifacts. During base
training, persisted HPO-selected feature sets are loaded when available;
otherwise MDI is recomputed.
