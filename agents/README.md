# Ares Research Contracts

Start with the repository-level [`AGENTS.md`](../AGENTS.md), then load the
contract relevant to the task:

- [`dataset_contract.md`](dataset_contract.md): timestamp, row, side, label,
  handoff, universe, and dtype semantics.
- [`feature_pipeline_rules.md`](feature_pipeline_rules.md): causal feature
  generation, cross-asset rules, AE/GMM inputs, parity, and memory constraints.
- [`leakage_prevention.md`](leakage_prevention.md): leakage checks across base,
  meta, archetypes, calibration, and policy optimization.
- [`model_validation_protocol.md`](model_validation_protocol.md): time-aware
  model validation and promotion evidence.
- [`experiment_discipline.md`](experiment_discipline.md): reproducible ablation,
  feature-selection, HPO, and walk-forward procedure.
- [`backtest_protocol.md`](backtest_protocol.md): execution, cost, portfolio,
  and economic metric requirements.

These documents define contracts. Current parameter values and promoted policy
IDs must still be read from the run manifest and saved policy artifacts.
