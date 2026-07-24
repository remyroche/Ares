# Ares Agent Guide

This file is the entry point for automated work in this repository. Read the
relevant contract under `agents/` before changing data, models, validation, or
execution behavior.

## Sub-Agent Delegation

Use sub-agents proactively whenever a task has relevant work that can proceed
independently. Do not wait for the user to request delegation explicitly. Before
delegating, identify the immediate critical-path task and keep that work in the
main agent; delegate bounded sidecars that can run in parallel without blocking
the next local action.

Use `gpt-5.6-terra` and select reasoning effort according to the work:

- **Terra Medium**: targeted repository searches, artifact inventories, test-log
  inspection, small isolated fixes, and other well-bounded routine work.
- **Terra High**: cross-module audits, implementation of meaningful isolated
  components, model/policy metric analysis, parity investigations, and test
  design where several contracts interact.
- **Terra XHigh**: difficult root-cause analysis, leakage-sensitive model or
  feature design, production inference/replay discrepancies, major architectural
  changes, and high-stakes quantitative validation.

Prefer multiple parallel sub-agents when there are genuinely independent
questions or disjoint write scopes. Every delegated task must have a concrete
deliverable and, for code edits, explicit file ownership. Tell workers that the
worktree is shared and that they must preserve and accommodate existing edits.
Do not delegate the same investigation twice, do not delegate a task whose
result is required before the main agent can make its next move, and do not use
sub-agents for trivial single-command work.

Use delegation to reduce main-thread context and total token use, not merely to
move work elsewhere:

- Give each sub-agent the smallest self-contained prompt and file/artifact scope
  needed for its task. Do not fork the full conversation context unless the task
  genuinely depends on it.
- Ask for compact, evidence-first results: conclusions, exact paths/lines,
  commands/tests run, metrics, and changed files. Avoid long narrative recaps.
- Let sub-agents read large logs, reports, and artifact trees, then return only
  the relevant findings so those raw contents do not enter the main context.
- Reuse an existing sub-agent for closely related follow-up work instead of
  spawning another agent and repeating context.
- Do not delegate when the prompt, synchronization, and review overhead is
  likely to exceed the work or token cost of handling the task locally.
- Preserve essential sub-agent findings in a short main-thread summary before
  closing the agent; do not copy its full output unless it is required evidence.

Continue useful local work while sub-agents run. Review their evidence or code
before relying on it, integrate only changes consistent with the contracts in
this file, and close completed agents promptly.

## Current Pipeline

The production research path is:

1. Causal, point-in-time feature generation.
2. Cycle-reference AE/GMM state discovery, fitted once with sampled
   beginning/middle/end rows at feature-selection/HPO time and then frozen.
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

- Use UTC for storage, joins, model features, labels, replay, inference, and
  artifact timestamps. Legacy naive timestamps are interpreted as UTC; never
  infer host-local time. Europe/Paris/CEST is display, email, and UI only.
- Preserve temporal ordering, purging, and embargo where label paths overlap.
- Fit feature selection/HPO, calibration, priors, and supervised models only on
  rows permitted by the relevant training/validation contract.
- Fit scaler/AE/GMM exactly once per model cycle on the designated sampled
  beginning/middle/end reference period. Reuse that exact serialized state for
  every base/meta growing window, final refit, replay, and inference. This
  reference fit is an explicitly disclosed representation-selection exception;
  it must never consume outcomes or justify an untouched-OOS claim.
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
