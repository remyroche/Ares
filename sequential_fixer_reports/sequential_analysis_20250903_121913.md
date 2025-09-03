# Sequential Code Analysis Report

**Target:** `/workspace/src`  
**Timestamp:** 2025-09-03T12:19:12.378662  
**Total files:** 487  
**Files analyzed:** 487  

## Summary

| Metric | Count |
|--------|-------|
| Clean files | 262 |
| Syntax errors | 155 |
| Indentation errors | 0 |
| Other errors | 0 |

## Top 10 Most Problematic Files

| File | Issues |
|------|--------|
| `tactician/dynamic_barrier_calculator.py` | 2 |
| `training/steps/step17_final_parameters_optimization_new.py` | 2 |
| `training/steps/step9_5_multi_timeframe_hmm_ensemble_validator.py` | 2 |
| `training/steps/step1/validate_and_fix_aggtrades_format.py` | 2 |
| `training/steps/step1/aggtrades_validator.py` | 2 |
| `training/steps/step1/step1_orchestrator.py` | 2 |
| `analyst/enhanced_regime_predictor.py` | 2 |
| `analyst/predictive_ensembles/regime_ensembles/base_ensemble.py` | 2 |
| `ares_pipeline.py` | 1 |
| `config.py` | 1 |

## Syntax Errors

- **monitoring/performance_dashboard.py** (line 11): from __future__ imports must occur at the beginning of the file
- **monitoring/performance_monitor.py** (line 10): from __future__ imports must occur at the beginning of the file
- **launcher/enhanced_trading_launcher.py** (line 81): invalid syntax
- **interfaces/enhanced_event_bus.py** (line 29): invalid syntax
- **supervisor/enhanced_prediction_service.py** (line 257): expected 'except' or 'finally' block
- **supervisor/supervisor.py** (line 630): unexpected indent
- **pipelines/live_trading_pipeline.py** (line 18): unexpected indent
- **pipelines/improved_pipeline_executor.py** (line 159): unexpected indent
- **pipelines/components/monitoring_manager.py** (line 28): '(' was never closed
- **pipelines/components/lifecycle_manager.py** (line 18): unexpected indent
- **pipelines/components/data_manager.py** (line 22): unexpected indent
- **integration/paper_trading_integration.py** (line 15): invalid syntax
- **tactician/position_sizer.py** (line 16): invalid syntax
- **tactician/position_division_strategy.py** (line 13): unexpected indent
- **tactician/tactician.py** (line 1138): unterminated triple-quoted string literal (detected at line 1146)
- **tactician/enhanced_execution_manager.py** (line 47): unexpected indent
- **tactician/position_monitor.py** (line 590): expected 'except' or 'finally' block
- **tactician/sr_data_integration.py** (line 32): expected 'except' or 'finally' block
- **tactician/sr_breakout_predictor.py** (line 917): expected 'except' or 'finally' block
- **tactician/ml_target_validator.py** (line 3): invalid syntax

... and 135 more syntax errors

