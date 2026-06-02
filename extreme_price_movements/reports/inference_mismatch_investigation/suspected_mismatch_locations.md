# Suspected Mismatch Locations

Status: initial evidence-backed list, 2026-06-01. These are investigation targets, not confirmed root causes.

## P0: Score Naming And Rank-Normalization Semantics

Evidence:

- `simple_policy_optimiser.py` writes a transformed score, then computes `rank_pct` and persists policy rank references.
- `inference/policy_rank_reference.py` maps the live transformed score into saved policy-rank percentiles.
- `tests/test_policy_rank_reference.py` verifies live gates use `policy_rank_reference_percentile` and not `meta_train_rank_pct`.

Risk:

- Because the code still uses legacy score-transform names, an audit can accidentally compare raw score thresholds instead of the actual rank-normalized deployment contract.

Next check:

- For the active run, load any legacy score-transform artifact if it exists. Quantify whether it changes score ordering. If it does, rank-normalized parity must include that transform; if it does not, call it a legacy naming layer.

## P0: Historical Replay Through Actual Inference Path

Evidence:

- `scripts/historical_inference_parity.py` exists and can sample `meta_oof` or policy rank-reference rows.
- `scripts/replay_live_signal_predictions.py` exists for ledger rows.

Risk:

- Good OOS could be computed from OOS/meta rows that do not exactly match live feature construction or live model package inputs.

Next check:

- Run historical parity for each of the four active strategy ids using policy rank-reference rows, not only raw `meta_oof` rows.
- Require equality or tight tolerance at these layers: feature matrix, alpha prediction, meta prediction, score transform, policy rank percentile, and auction rank percentile.

## P0: Deployment Strategy Mask Contract Missing

Evidence:

- `strategy_for_inference.json` for `20260525_010004_nopenalty` selects six strategies but all six rows have `lgbm_regime_mask={}` and `regime_mask_source=missing_lgbm_mask_contract`.
- A run-scoped shadow live cycle loaded the six-strategy portfolio contract, then filtered the legacy four-row perps mask CSV to zero selected rows and failed closed before scoring.
- The legacy perps mask CSV contains four older rule strategies, not the current six selected policy/model heads.
- `policy_oos_retrain_strategy_source_perps.csv` calls the selected model/head ids `canonical_key`, but sampled ids are not parseable by the canonical rule parser and do not expose feature/operator/slot structure.

Risk:

- Live inference cannot evaluate the required pre-base LGBM regime masks for the selected deployment. It therefore correctly becomes monitor-only/fail-closed. This can look like "no masks pass" or "no live positions" but is actually an artifact contract failure before candidate scoring.

Patch:

- Policy deployment writers now pass `market_mode` to the mask-contract loader.
- LGBM deployments now reject otherwise-profitable strategies with missing mask contracts when `EPM_MODEL_BACKEND=lgbm_pipeline` or `EPM_REQUIRE_LGBM_REGIME_MASK_CONTRACTS=1`.

Next check:

- Regenerate or link valid mask contracts for the selected six by preserving the original generated-rule `base_event_trigger` through the retrain/strategy-selection handoff, then rerun the run-scoped shadow live cycle and compare fresh ledger rows against replay decisions.

## P0: Execution Delay And Adverse Entry Gap

Evidence:

- `simple_policy_optimiser.py` applies `_apply_delayed_entry_execution_model(...)` and writes delay rejection reports.
- `run_inference.py` writes `decision_to_entry_seconds`, `signal_to_entry_seconds`, `entry_delay_effect_bps`, `adverse_signal_gap_bps`, `expected_total_entry_friction_bps`, and spread/slippage fields to the prediction ledger.
- Previous live observation showed one large adverse signal gap that would materially change EV.

Risk:

- OOS edge may be true at signal time but not after live data/fetch/feature/execution delay. This is especially important if the live cycle takes several minutes after the hourly close.

Next check:

- Compare OOS delayed-entry proxy distributions against live ledger delay/gap/friction distributions.
- Verify policy optimiser now uses t+10 candles as requested and reports rejection simulations at 0.5%, 1.0%, and 1.5% adverse move thresholds.

## P0: Feature Universe And Cross-Sectional Features

Evidence:

- `tests/test_live_feature_universe_parity.py` verifies full tradable universe feature computation.
- User-observed feature diffs included cross-sectional residuals and market-wide features.
- `inference/feature_generator.py` has live cache and symbol universe logic.

Risk:

- If live feature computation uses a smaller or drifting universe, peer residuals, ranks, covariance, market-equity features, and relative-volume features can diverge from OOS/training.

Next check:

- For identical timestamp/symbol rows, diff feature values by family and classify diffs as source data, live cache, universe, stale microdata, or feature implementation.
- Specifically revisit market-equity mask feature being outside range for all four masks and determine whether that is market regime, feature bug, or range contract problem.

## P0: Deployment LGBM Regime-Mask Handoff

Status: confirmed and repaired, 2026-06-02.

Evidence:

- Run-scoped shadow live failed closed before scoring because all six selected strategies in `20260525_010004_nopenalty` had `regime_mask_source=missing_lgbm_mask_contract`.
- The fallback perps mask source contained only stale/older strategies and filtered to zero for the selected six.
- The active `policy_oos_retrain_strategy_source_perps.csv` had safe ids in `canonical_key`, not parseable LGBM rule expressions.
- The corresponding original parseable rules were still available in `strategy_selection/candidate_strategies.json`.

Fix:

- Preserve explicit `strategy_id` as deployment/model identity in the mask loader.
- Use `base_event_trigger` as the parseable LGBM mask rule when present.
- Fail deployment creation for LGBM strategies missing mask contracts.
- Backfill the current policy-source CSV and deployment JSON copies with embedded parseable mask contracts.

Verification:

- Inference loader selected cores: `6`.
- Embedded aliases loaded: `12`.
- Runtime coverage guard passed.
- Focused deployment/params-store/inference parity tests passed.

## P1: Artifact Contract Integrity

Evidence:

- `model_orchestrator.py` validates selected feature contracts and extracts LGBM diagnostic ledger keys.
- `lgbm_pipeline.py` saves selected feature counts, selected features, OOF score/rank stats, rank-bin stats, and feature/importances.
- `parity.py` loads deployment strategy filters from policy artifacts.

Risk:

- The inference loop could load the wrong artifact run, a stale policy payload, or a model package whose selected feature contract is compatible by name but not by transform/source window.

Next check:

- Build `artifact_manifest.json` for `20260525_010004_nopenalty`, recording model run id, policy run id, feature contract hash, transform contract hash, meta feature contract, selected features, policy rank reference row counts, and deployment thresholds.

## P1: Market/Stop Execution Logging

Evidence:

- `run_inference.py` logs many entry fields and stop lifecycle events.
- The execution code also has marketable limit-price fields.

Risk:

- If marketable limit wrappers are treated like passive limit orders in diagnostics, execution cost assumptions will be misread. The investigation must measure market/stop execution outcomes directly.

Next check:

- Verify ledger/trade logs include enough live data to tune market entry gap, spread, slippage, stop trigger/fill gap, funding, and fees.
- Remove passive-limit execution metrics from new reports unless the exchange API returns a rejection/no-fill state that matters operationally.

## P1: Live Data Source Parity

Evidence:

- Recent sampled live-vs-historical OHLCV check matched 7/8 sampled symbols exactly and showed ETH/USD:USD as an outlier.
- User requested a more reliable test.

Risk:

- If live cached bars differ from historical bars for the same closed candle, training/OOS/live parity is undermined.

Next check:

- Extend candle parity sampling across more symbols, multiple recent hourly windows, and both high-liquidity and low-liquidity assets.
- Report exact mismatches by OHLCV field and data path.

## P2: Model Decay Versus Regime Shift

Evidence:

- The user observed recent months were more bullish and long/short behavior looked asymmetric.
- Regime adaptors are disabled for the active run.

Risk:

- Poor live results could be genuine regime shift even after software parity is proven.

Next check:

- Compare OOS versus live distributions for score ranks, feature drift PSI, covariance drift, rare leaf fraction, leaf count/support, and contribution drift.
- Segment performance by market direction, 5d/10d/15d/30d trend, volatility, volume, OI, funding, and liquidity regimes.
