# Ares Agent Guide

This file is the entry point for automated work in this repository. Read the
relevant contract under `agents/` before changing data, models, validation, or
execution behavior.

## Current Pipeline

The production research path is:

1. Causal, point-in-time feature generation.
2. Train-only AE/GMM state discovery, frozen before OOS assignment.
3. Global, archetype-aware base models with explicit long/short handling.
4. Top-30% base candidate handoff to the meta model.
5. A side- and archetype-aware meta model trained on the base candidate stream.
6. Top-10% policy admission after side/archetype EV calibration and recent
   hit-rate adjustment.
7. Side and side-by-archetype execution geometry, position sizing, capital
   pressure, and portfolio allocation.
8. Live inference with the same feature, score, calibration, cost, and policy
   contracts used by replay.

Do not introduce `strategy_id` masks into this global path unless an experiment
explicitly tests that change.

## Archetype Awareness

- Base models use observable label/state archetypes during pre-screening and may
  consume frozen AE/GMM IDs, posteriors, entropy, distance, reconstruction error,
  speed, acceleration, and train-derived reliability context.
- Meta models retain the base archetype identity and may add meta-layer regimes,
  side x archetype reliability, residual, drift, and recent-performance context.
- Keep archetype effects continuous/probabilistic where possible. Do not convert
  them into hard gates without stable leakage-safe OOS evidence.
- Report every model layer overall and by side, archetype, and side x archetype.

## Canonical Code

- Pipeline CLI: `extreme_price_movements/run_pipeline.py`
- Feature registration/generation: `config.py`, `features.py`, `features_oi.py`
- Feature parity: `feature_transform_contract.py`, `inference/feature_parity.py`
- Base/meta training and feature selection: `lgbm_pipeline.py`
- AE/GMM state: `lgbm_archetype_features.py`, `evm_latent_state_discovery.py`
- Policy optimization: `simple_policy_optimiser.py`
- Regime/archetype EV calibration: `regime_ev_calibration.py`
- Portfolio and live policy: `portfolio_manager.py`, `inference/portfolio_policy.py`
- Live inference: `inference/run_inference.py`

## Non-Negotiable Contracts

- Preserve temporal ordering, purging, and embargo where label paths overlap.
- Fit scalers, AE/GMM, feature selection, HPO, calibration, and priors only on
  rows permitted by the relevant training/validation contract.
- Keep AE/GMM state frozen across growing OOS windows so cluster semantics do
  not change between folds.
- Keep long and short rows separate in diagnostics and side-aware stages.
- Treat archetypes as context unless a leakage-safe OOS test supports gating.
- Compare models on identical rows, periods, costs, labels, and top-k basis.
- Report both notional return per trade and bankroll/portfolio PnL.
- Report signed residual mean/autocorrelation and signed hit-rate surprise. Keep
  both favorable and adverse deviations; do not clip these metrics at zero.
- Record costs once. Never subtract the same fee or spread in both labels and
  replay without an explicit reconciliation.
- Preserve existing artifacts unless deletion or overwrite is explicitly
  requested.

## Detailed Guidance

- `agents/dataset_contract.md`
- `agents/feature_pipeline_rules.md`
- `agents/leakage_prevention.md`
- `agents/model_validation_protocol.md`
- `agents/experiment_discipline.md`
- `agents/backtest_protocol.md`
