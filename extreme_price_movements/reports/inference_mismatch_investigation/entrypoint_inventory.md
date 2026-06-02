# Entrypoint Inventory

Status: initial repository-verified inventory, 2026-06-01.

## Training And Artifact Generation

| Entrypoint | Purpose | Important notes |
| --- | --- | --- |
| `python3 extreme_price_movements/run_pipeline.py download --market-mode perps --exchange kraken` | Fetch scoped Kraken perp data. | Supports env-controlled symbol ordering, partitioning, stride, freshness gates, 15m, 1m, and microdata flags. |
| `python3 extreme_price_movements/run_pipeline.py labels --market-mode perps --exchange kraken --run-id <run_id>` | Build labels for runtime horizons. | Uses artifact/source overrides such as `EPM_ARTIFACT_SOURCE_RUN_ID`. |
| `python3 extreme_price_movements/run_pipeline.py features --market-mode perps --exchange kraken --run-id <run_id>` | Build feature frames and health reports. | Perp mode enables perp feature keys and scoped exchange roots. |
| `EPM_MODEL_BACKEND=lgbm_pipeline python3 extreme_price_movements/run_pipeline.py train_base --market-mode perps --exchange kraken --run-id <run_id>` | Train base heads. | Uses `lgbm_pipeline.py`; selected features and LGBM reference artifacts are saved. |
| `EPM_MODEL_BACKEND=lgbm_pipeline python3 extreme_price_movements/run_pipeline.py train_meta --market-mode perps --exchange kraken --run-id <run_id>` | Train meta heads. | Writes `meta_oof`, `meta_feature_contract.json`, `meta_head_metrics.json`, and meta model packages. |
| `python3 -m extreme_price_movements.simple_policy_optimiser ...` or import-level `run_simple_policy_optimisation(...)` | Current simple policy deployment optimizer. | Uses OOS/policy meta rows, delayed-entry 1m proxy, per-strategy rank threshold discovery, Stage B CV, portfolio replay, and rank-reference export. |

## Inference And Monitoring

| Entrypoint | Purpose | Important notes |
| --- | --- | --- |
| `python3 -m extreme_price_movements.inference.run_inference --market-mode perps --exchange kraken ...` | Live/live-test inference loop. | Loads policy artifacts, computes candidate masks/features, scores alpha/meta heads, applies policy-rank gates, writes prediction ledger, and routes market/stop execution. |
| `python3 scripts/replay_live_signal_predictions.py ...` | Recompute live ledger decisions. | Filters by `policy_rank_reference_percentile`, compares live vs replay base/meta/score/rank values. |
| `python3 scripts/historical_inference_parity.py ...` | Replay historical OOS/policy rows through inference code. | Samples either `meta_oof` or saved policy rank-reference rows. |
| `python3 scripts/verify_live_ohlcv_parity.py ...` | Compare live-fetched OHLCV against historical endpoint data. | Recently created/used for Kraken live-vs-history candle parity checks. |

## Key Modules

| Module | Role |
| --- | --- |
| `extreme_price_movements/lgbm_pipeline.py` | LGBM base/meta training, selected feature contracts, OOF metrics, rank-bin OOF meta features, reference diagnostics. |
| `extreme_price_movements/slice_plan_store.py` | Materialized train/OOS/policy stage views and usage limits. |
| `extreme_price_movements/feature_transform_contract.py` | Feature transform contract persistence. |
| `extreme_price_movements/simple_policy_optimiser.py` | Simple policy OOS execution simulation, threshold discovery, rank reference persistence, deployment payloads, candidate export. |
| `extreme_price_movements/inference/policy_rank_reference.py` | Saved policy rank CDF lookup and rank-percentile gate. |
| `extreme_price_movements/inference/model_orchestrator.py` | Alpha and meta scoring, feature contract validation, drift/leaf diagnostics extraction. |
| `extreme_price_movements/inference/feature_generator.py` | Live feature computation/cache/materialization for model inputs. |
| `extreme_price_movements/inference/run_inference.py` | Live loop, candidate gating, portfolio auction, entry diagnostics, prediction ledger, active position monitoring. |
| `extreme_price_movements/inference/prediction_ledger.py` | Durable prediction ledger writer. |
| `extreme_price_movements/inference/trade_executor.py` | Exchange execution abstraction and live/shadow order handling. |
| `extreme_price_movements/inference/dynamic_strategy_performance.py` | Reads ledger/trade history to assess realized performance and drift availability. |

## Existing Tests To Run First

Use these before adding deeper regression tests:

```bash
python3 -m pytest tests/test_policy_rank_reference.py -q
python3 -m pytest tests/test_live_feature_universe_parity.py -q
python3 -m pytest tests/test_replay_live_signal_predictions.py -q
python3 -m pytest extreme_price_movements/tests/test_inference_alignment.py -q
python3 -m pytest extreme_price_movements/tests/test_inference_step_parity.py -q
```

## Pipeline-Specific Corrections To The Generic Spec

- Replace generic score-adjustment wording with rank-normalization in new investigation outputs. Runtime compatibility columns may still use legacy transformed-score names.
- Remove passive-limit execution language. Market/stop execution, marketable-entry wrappers, rejected orders, stale quotes, spread/slippage, and stop-fill realism remain in scope.
- Keep incomplete latest bars in scope. That is a data availability issue, not an execution assumption.
- The most important top-k reconciliation is policy-rank CDF parity and cross-strategy auction rank parity, not raw probability sorting.
