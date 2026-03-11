# Agents Repository Guidelines

This directory contains key guidelines and protocols that govern backtesting, datasets, experiments, feature pipelines, and leakage prevention in this repository.

Here is a summary of the available documents:

* **[backtest_protocol.md](backtest_protocol.md)**: Defines the economic evaluation framework and rules for simulating realistic trading conditions in backtests. It details requirements for transaction costs, timing, and proper out-of-sample evaluation metrics.
* **[dataset_contract.md](dataset_contract.md)**: Establishes data semantics, timestamp alignment rules, feature causality, missing data policies, and the usage of memory-efficient data types for all datasets.
* **[experiment_discipline.md](experiment_discipline.md)**: Specifies the minimum protocol for ML experiments, emphasizing statistical validity, out-of-sample robustness, correct time-based data splitting, walk-forward evaluation, and strict reproducibility.
* **[feature_pipeline_rules.md](feature_pipeline_rules.md)**: Outlines the rules governing feature generation to prevent lookahead bias and data leakage. Key areas include causality, rolling window rules, global normalization constraints, cross-asset isolation, and deterministic execution.
* **[leakage_prevention.md](leakage_prevention.md)**: Focuses explicitly on preventing various forms of data leakage (temporal, normalization, cross-sectional, label leakage) and mandates strict validation checks for pipeline integrity.
