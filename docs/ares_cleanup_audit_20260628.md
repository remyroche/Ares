# Ares Cleanup Audit - 2026-06-28

This audit reviews local generated artifacts in `/Users/remyroche/Documents/Ares` after the C3el / reliability-blend / market-state experiments. It records the cleanup that was actually performed and the remaining operational risks after deleting large generated stores.

## Current Size

Total repository footprint after the 2026-06-28 cleanup pass:

| Path | Size | Notes |
| --- | ---: | --- |
| `.` | 83G | Full workspace |
| `data_perp` | 68G | Main generated-data footprint |
| `data_perp/features` | 39G | Remaining generated feature stores |
| `data_perp/artifacts` | 25G | Model/artifact runs |
| `data_perp/reports` | 1.3G | Experiment reports and ledgers |

## Deleted In This Cleanup Pass

The following generated outputs were removed after review.

| Target | Approx size | Reason |
| --- | ---: | --- |
| `.pytest_cache`, `.mplconfig`, `**/__pycache__` | small | Reproducible local caches |
| `logs/*` | ~139M | Reproducible logs from completed/debug runs |
| Deprecated `performance_market_state_modulator*` reports | ~28.5G before cleanup | Negative global/per-head modulator line; not promoted |
| Old market-state/direct-suppression reports | several GB | Superseded by later T1/static-baseline conclusion |
| Superseded exact-state C3el smoke/sweep reports | several GB | Failed/intermediate runs; current exact panel and selected-head reports retained |
| `data_perp/artifacts/low_perf_specialist_*` | ~1.2G | Low-performance specialist line not promoted |
| `data_perp/artifacts/unsupervised_regime_learning_poc` | ~1.8G | Superseded PoC artifact |
| `data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit` large contents | ~21G before cleanup | Old full-fit artifact contents were intentionally removed; a small `live_state` stub may be recreated by a live process |
| `data_perp/reports/finalfit_broad_candidate_regen_20260627` | ~2.5G | Dynamic-HR/T1 follow-up artifact no longer needed |
| `data_perp/features/20260605_070000` | ~36G | Deleted by explicit user request |
| `data_perp/features/20260622_230000` | ~18G | Deleted by explicit user request |

## Keep For Current Work

Do not delete these in a default cleanup pass. They are referenced by source defaults, config, current manifests, or active analysis.

| Target | Reason |
| --- | --- |
| `data_perp/artifacts/20260618_081800_current4_final_fit` | Remaining full-fit model artifact with real model contents. |
| `data_perp/artifacts/meta_featureselect_recentguard_20260622_0119` | Default meta artifact dir for q_fail/contextual/reliability scripts. Large, but still referenced by defaults. |
| `data_perp/features/20260627_120000` | Referenced by `config/reliability_blend_production_stack.json` as active feature store. |
| `data_perp/artifacts/20260627_120000` | Recent active artifact run. |
| `data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625` | Baseline/A0 policy manifest source for current C3el inputs. |
| `data_perp/reports/exact_state_size_action_learning_20260626_8fold_train720_eval120_c3dq_c3dr_c3ds_unfiltered_recall_reuse_e50_m25` | Contains the current exact-state size-action panel used by C3el learner runs. |
| `data_perp/reports/exact_state_size_action_learning_20260628_last4w_c3el_inputs` | Current broad/deployable candidate inputs for last-4-week C3el runs. |
| Latest `c3el_*` summaries | Tiny, useful provenance for conclusions and next steps. |

## Remaining Feature Stores

| Target | Size | Notes |
| --- | ---: | --- |
| `data_perp/features/20260617_090000` | 397M | Small retained feature store |
| `data_perp/features/20260627_010000` | 19G | Retained live/latest feature store |
| `data_perp/features/20260627_120000` | 20G | Current retained live/latest feature store |

## Stale Historical Defaults

The active T1/live-facing defaults were patched to stop using deleted feature stores. The remaining references to `20260605_070000`, `20260622_230000`, and `20260617_090000_no_mkt4_labelhpo_final_fit` are primarily historical/research rerun scripts. They should be treated as non-runnable without explicit replacement paths or regenerated artifacts.

Notable remaining stale-default families:

- old contextual meta/q_fail/reliability ablation scripts defaulting to `data_perp/features/20260605_070000`;
- old market-state controller/research scripts that now require explicit historical feature stores after cleanup;
- old low-performance specialist and recent-failure diagnostic scripts referencing the removed June 17 artifact lineage;
- one test fixture preserving the old June 17 run id as historical sample data.

Do not interpret these historical scripts as current production defaults.

## Config-Referenced Market-State Reports To Keep Or Rewrite First

`config/reliability_blend_production_stack.json` references several market-state/direct-suppression report folders. Do not delete these unless you also update the config or accept broken artifact links:

- `data_perp/reports/market_state_controller_bundle_score_globalrank_no_backfill_shadow_allstates_20260627_120000_20260627_13_20260627_15`
- `data_perp/reports/market_state_direct_suppression_controller_training_globalrank_no_backfill_combined_20260627_v7_with_jun26_partial_strategy_diagnostics`
- `data_perp/reports/market_state_direct_suppression_ledger_globalrank_no_backfill_combined_20260627_v4_with_jun26_partial_strategy_diagnostics`
- `data_perp/reports/market_state_direct_suppression_controller_training_globalrank_no_backfill_pruned_20260627_v2`
- `data_perp/reports/market_state_direct_suppression_ledger_globalrank_no_backfill_pruned_20260627_v2`

Given the current decision to stop active market-state suppression, these may become deletable after the production stack config is simplified or archived.

## Suggested Future Deletion Order

1. Delete caches and empty dirs.
2. Delete `performance_market_state_modulator*` report folders.
3. Delete old market-state report folders except the config-referenced keep-list above and final review summaries.
4. Prune exact-state C3el sweeps, keeping the current exact panel, current inputs, latest selected-head runs, and final c3el summary reports.
5. Only then consider the two remaining large feature stores after verifying no active live/replay job depends on them.

## Not Safe Without Follow-Up

These are large or operationally sensitive; deleting them can break scripts or reproducibility:

- `data_perp/artifacts/20260618_081800_current4_final_fit`
- `data_perp/artifacts/meta_featureselect_recentguard_20260622_0119`
- `data_perp/features/20260627_120000`
- `data_perp/features/20260627_010000`
- Raw/live market data under `data_perp/exchanges`, `data_perp/ohlcv`, `data_perp/orderbook_hourly`, `data_perp/execution_1m`

## Estimated Low-Risk Reclaim

The completed cleanup reduced the workspace from roughly 210G to 83G, reclaiming about 127G.
