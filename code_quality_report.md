# Code Quality Analysis Report

## Summary

- **Debug Statements**: 166
- **Type Ignores**: 120
- **Broad Exceptions**: 4503
- **Unused Imports**: 1392
- **Todo Comments**: 744

## Detailed Issues

### test_advanced_features.py

#### Broad Exceptions

Line 113: `except Exception as e:`

#### Unused Imports

Line 9: `from datetime import datetime`
Line 9: `from datetime import timedelta`

### test_feature_output_validation.py

#### Broad Exceptions

Line 235: `except Exception as e:`

### test_comprehensive_sr_features.py

#### Broad Exceptions

Line 174: `except Exception as e:`

#### Unused Imports

Line 13: `from datetime import datetime`
Line 13: `from datetime import timedelta`
Line 21: `from src.training.steps.step2_feature_engineering import run_step`
Line 23: `from src.utils.logger import system_logger`
Line 157: `from src.training.steps.step2_feature_engineering import run_step`

### kelly_criterion_fix.py

#### Broad Exceptions

Line 85: `except Exception as e:`
Line 150: `except Exception as e:`

### create_regime_splits.py

#### Debug Statements

Line 26: `print(f"🔍 Creating regime splits file: {output_file}")`

#### Broad Exceptions

Line 125: `except Exception as e:`

#### Unused Imports

Line 9: `from pathlib import Path`

### test_step1_5_edge_cases.py

#### Broad Exceptions

Line 141: `except Exception as e:`
Line 194: `except Exception as e:`
Line 270: `except Exception as e:`
Line 298: `except Exception as e:`

### create_30m_hmm_artifacts.py

#### Broad Exceptions

Line 71: `except Exception as e:`

### test_enhanced_hmm_features.py

#### Debug Statements

Line 26: `print("🔍 Testing Enhanced HMM Feature Selection")`
Line 248: `print(f"\n🔍 Testing with Real Data from {meta_file}")`

#### Broad Exceptions

Line 281: `except Exception as e:`

#### Unused Imports

Line 12: `from src.training.steps.step1_7_hmm_regime_discovery import _name_liquidity_states`
Line 12: `from src.training.steps.step1_7_hmm_regime_discovery import _generate_archetype_descriptions`

### enhanced_validation_logging.py

#### Debug Statements

Line 210: `print(f"\\n🔍 ISSUE BREAKDOWN:")`

#### Unused Imports

Line 7: `import json`
Line 9: `from typing import Dict`
Line 9: `from typing import List`
Line 9: `from typing import Any`
Line 10: `from pathlib import Path`

### test_step1_7_simple.py

#### Debug Statements

Line 26: `print("🔍 Testing function existence...")`

#### Broad Exceptions

Line 39: `except Exception as e:`

### simulate_regime_merging_from_existing_data.py

#### Debug Statements

Line 139: `print("🔍 Loading existing HMM data...")`
Line 230: `print("\n🔍 CLOSEST TO TARGET (75%):")`
Line 286: `print("🔍 Running full parameter sweep...")`

#### Broad Exceptions

Line 114: `except Exception as e:`
Line 196: `except Exception as e:`

#### Unused Imports

Line 10: `from collections import defaultdict`

### test_individual_parquet.py

#### Broad Exceptions

Line 65: `except Exception as e:`
Line 125: `except Exception as e:`

#### Unused Imports

Line 9: `from datetime import datetime`

### test_hmm_fix.py

#### Broad Exceptions

Line 106: `except Exception as e:`

### test_sr_features_simple.py

#### Broad Exceptions

Line 43: `except Exception as e:`
Line 71: `except Exception as e:`
Line 110: `except Exception as e:`
Line 143: `except Exception as e:`

#### Unused Imports

Line 80: `from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor`

### test_step1_7.py

#### Broad Exceptions

Line 40: `except Exception as e:`

### test_enhanced_step1_7.py

#### Broad Exceptions

Line 59: `except Exception as e:`

### test_hmm_state_naming_fix.py

#### Debug Statements

Line 35: `print("🔍 Testing HMM State Naming Fixes")`

#### Broad Exceptions

Line 68: `except Exception as e:`
Line 100: `except Exception as e:`

#### Unused Imports

Line 12: `from src.training.steps.step1_7_hmm_regime_discovery import _name_momentum_states`
Line 12: `from src.training.steps.step1_7_hmm_regime_discovery import _name_volatility_states`
Line 12: `from src.training.steps.step1_7_hmm_regime_discovery import _name_liquidity_states`
Line 12: `from src.training.steps.step1_7_hmm_regime_discovery import _name_microstructure_states`

### test_step1_5_simple.py

#### Broad Exceptions

Line 41: `except Exception as e:`

#### Unused Imports

Line 4: `import os`
Line 26: `from src.config import CONFIG`
Line 30: `from src.utils.logger import setup_logging`
Line 30: `from src.utils.logger import system_logger`
Line 35: `from src.training.steps.step1_5_data_converter import UnifiedDataConverter`

### test_shap_timeout.py

#### Debug Statements

Line 62: `print("🔍 Starting feature filtering with timeout protection...")`

#### Broad Exceptions

Line 82: `except Exception as e:`

#### Unused Imports

Line 7: `import os`

### test_sr_criteria_loosening.py

#### Debug Statements

Line 102: `print("\n🔍 Testing with S/R levels...")`
Line 122: `print("\n🔍 Testing fallback case (no S/R levels)...")`

#### Unused Imports

Line 11: `from datetime import datetime`
Line 11: `from datetime import timedelta`

### run_30m_hmm_step.py

#### Broad Exceptions

Line 87: `except Exception as e:`
Line 110: `except Exception as e:`

#### Unused Imports

Line 9: `import os`

### ares_launcher.py

#### Debug Statements

Line 1974: `print(f"🔍 DEBUG: Executing command: {args.command}")`
Line 1975: `print(f"🔍 DEBUG: Symbol: {args.symbol}, Exchange: {args.exchange}")`
Line 2058: `print(f"🔍 DEBUG: Found command handler for '{args.command}'")`
Line 2060: `print(f"🔍 DEBUG: Command execution result: {success}")`

#### Broad Exceptions

Line 516: `except Exception as e:`
Line 655: `except Exception as e:`
Line 730: `except Exception as e:`
Line 864: `except Exception as e:`
Line 929: `except Exception as e:`
Line 1000: `except Exception as e:`
Line 1076: `except Exception as e:`
Line 1147: `except Exception as e:`
Line 1224: `except Exception as e:`
Line 1288: `except Exception as e:`
... and 4 more

#### Todo Comments

Line 1974: `print(f"🔍 DEBUG: Executing command: {args.command}")`
Line 1975: `print(f"🔍 DEBUG: Symbol: {args.symbol}, Exchange: {args.exchange}")`
Line 2058: `print(f"🔍 DEBUG: Found command handler for '{args.command}'")`
Line 2060: `print(f"🔍 DEBUG: Command execution result: {success}")`

### extract_feature_details.py

#### Unused Imports

Line 8: `import re`

### sr_weight_optimization_backtest.py

#### Debug Statements

Line 576: `print("\n🔍 Key Findings:")`

#### Broad Exceptions

Line 78: `except Exception as e:`
Line 133: `except Exception as e:`
Line 162: `except Exception as e:`
Line 195: `except Exception as e:`
Line 229: `except Exception as e:`
Line 292: `except Exception as e:`
Line 346: `except Exception as e:`
Line 417: `except Exception as e:`
Line 465: `except Exception as e:`
Line 492: `except Exception as e:`

#### Unused Imports

Line 16: `from datetime import timedelta`
Line 20: `from typing import List`

### debug_low_variance_features.py

#### Unused Imports

Line 10: `from pathlib import Path`
Line 11: `import json`
Line 12: `from typing import List`

#### Todo Comments

Line 3: `Debug script to analyze low variance features in the autoencoder feature generator.`
Line 105: `def generate_debug_report(`
Line 109: `Generate a comprehensive debug report for low variance features.`
Line 120: `report.append("LOW VARIANCE FEATURES DEBUG REPORT")`
Line 205: `logger.info("🚀 Starting low variance features debug analysis...")`
Line 210: `logger.info("   2. Call generate_debug_report(features_df)")`
Line 256: `report = generate_debug_report(dummy_df)`
Line 259: `logger.info("✅ Debug analysis complete!")`

### implement_feature_specific_validation.py

#### Unused Imports

Line 7: `import json`
Line 8: `import re`
Line 11: `from typing import Tuple`
Line 13: `from pathlib import Path`

### analyze_strict_thresholds.py

#### Unused Imports

Line 7: `import json`
Line 8: `import re`
Line 11: `from typing import List`
Line 13: `from pathlib import Path`

### check_existing_data.py

#### Debug Statements

Line 10: `print("🔍 CHECKING EXISTING UNIFIED DATA:")`

#### Broad Exceptions

Line 34: `except Exception as e:`

#### Unused Imports

Line 4: `import glob`

### test_optimized_feature_selection.py

#### Debug Statements

Line 255: `print(f"\n🔍 Matrix VIF vs Iterative VIF Test")`

#### Broad Exceptions

Line 291: `except Exception as e:`
Line 372: `except Exception as e:`

#### Unused Imports

Line 8: `import asyncio`
Line 11: `import json`

### test_nan_detection.py

#### Debug Statements

Line 22: `print("🔍 Creating test data with known NaN values...")`
Line 87: `print(f"\n🔍 Manual NaN detection results:")`
Line 221: `print(f"\n🔍 Analyzing NaN ranges for column: {col}")`

#### Unused Imports

Line 7: `import os`
Line 17: `from datetime import datetime`
Line 17: `from datetime import timedelta`

### test_feature_reduction_implementation.py

#### Broad Exceptions

Line 247: `except Exception as e:`

#### Unused Imports

Line 11: `from pathlib import Path`

### test_state_naming.py

#### Unused Imports

Line 5: `import os`

### feature_analysis_script.py

#### Debug Statements

Line 135: `print(f"\n🔍 ISSUE BREAKDOWN:")`

#### Unused Imports

Line 10: `from typing import List`

### test_data_quality_decorators.py

#### Broad Exceptions

Line 165: `except Exception as e:`
Line 178: `except Exception as e:`
Line 187: `except Exception as e:`
Line 198: `except Exception as e:`
Line 207: `except Exception as e:`
Line 217: `except Exception as e:`
Line 230: `except Exception as e:`
Line 240: `except Exception as e:`

### test_incremental.py

#### Debug Statements

Line 10: `print("🔍 TESTING INCREMENTAL PROCESSING LOGIC")`

#### Broad Exceptions

Line 39: `except Exception as e:`
Line 81: `except Exception as e:`

#### Unused Imports

Line 6: `from datetime import datetime`
Line 6: `from datetime import timedelta`
Line 6: `from datetime import timezone`

### test_enhanced_validation.py

#### Debug Statements

Line 102: `print("\n🔍 Running enhanced validation...")`
Line 141: `print(f"\n🔍 DETAILED ISSUES:")`

### test_exclude_recent_days.py

#### Broad Exceptions

Line 161: `except Exception as e:`

### test_simple.py

#### Debug Statements

Line 26: `print("🔍 Getting manager...")`
Line 31: `print("🔍 Testing file paths...")`
Line 38: `print("🔍 Testing file existence...")`

#### Broad Exceptions

Line 44: `except Exception as e:`

### debug_hmm_combinations.py

#### Debug Statements

Line 42: `print(f"🔍 Found {len(state_cols)} state columns: {state_cols}")`
Line 140: `print("🔍 HMM Combinations Debug Analysis")`

#### Todo Comments

Line 3: `Debug HMM Combinations Script`
Line 140: `print("🔍 HMM Combinations Debug Analysis")`

### test_print.py

#### Unused Imports

Line 7: `import sys`
Line 11: `import os`

### consolidate_data.py

#### Broad Exceptions

Line 55: `except Exception as e:`

#### Unused Imports

Line 12: `from datetime import timedelta`
Line 13: `from pathlib import Path`

### analyze_validation_issues.py

#### Unused Imports

Line 11: `from typing import List`
Line 13: `from pathlib import Path`

### test_step1_5_converter.py

#### Broad Exceptions

Line 106: `except Exception as e:`

### cleanup_script.py

#### Type Ignores

Line 27: `r'# type: ignore',`
Line 28: `r'# noqa',`
Line 210: `"        content = re.sub(r'\\s*# type: ignore.*\\n', '\\n', content)",`

#### Broad Exceptions

Line 31: `r'except Exception:',`
Line 32: `r'except Exception as e:',`
Line 33: `r'except:',`
Line 93: `except Exception as e:`
Line 218: `"    except Exception as e:",`

#### Unused Imports

Line 11: `from typing import Set`

#### Todo Comments

Line 20: `self.debug_patterns = [`
Line 21: `r'print\(.*DEBUG.*\)',`
Line 23: `r'print\(.*debug.*\)',`
Line 24: `r'print\(.*Debug.*\)',`
Line 51: `'debug_statements': [],`
Line 55: `'todo_comments': [],`
Line 63: `# Check for debug statements`
Line 65: `for pattern in self.debug_patterns:`
Line 67: `issues['debug_statements'].append((i, line.strip()))`
Line 82: `# Check for TODO comments`
... and 9 more

### debug_clustering.py

#### Broad Exceptions

Line 72: `except Exception as e:`
Line 86: `except Exception as e:`

#### Todo Comments

Line 3: `Debug script to test HMM clustering functionality.`

### run_fixed_hmm_regime_discovery.py

#### Broad Exceptions

Line 54: `except Exception as e:`
Line 67: `except Exception as e:`

#### Unused Imports

Line 11: `import os`

### test_step1_5_granularity.py

#### Broad Exceptions

Line 140: `except Exception as e:`

#### Unused Imports

Line 14: `from datetime import datetime`
Line 15: `from pathlib import Path`

### convert_csv_to_parquet.py

#### Broad Exceptions

Line 45: `except Exception as e:`

### diagnose_regime_data.py

#### Debug Statements

Line 24: `print(f"🔍 Loading test data for {exchange}_{symbol} ({days} days)...")`
Line 228: `print("\n🔍 Analyzing data quality...")`
Line 264: `print(f"🔍 NaN values per column:")`
Line 310: `print(f"🔍 Key feature statistics:")`
Line 348: `print("🔍 Testing HMM state interpretation...")`
Line 415: `print(f"\n🔍 Threshold analysis:")`
Line 549: `print("🔍 Regime Classification Data Processing Diagnostic")`

#### Broad Exceptions

Line 69: `except Exception as e:`
Line 82: `except Exception as e:`
Line 146: `except Exception as e:`
Line 162: `except Exception as e:`
Line 165: `except Exception as e:`
Line 221: `except Exception as e:`
Line 327: `except Exception as e:`
Line 387: `except Exception as e:`

#### Unused Imports

Line 11: `from datetime import datetime`
Line 11: `from datetime import timedelta`
Line 12: `import json`
Line 19: `from src.utils.logger import system_logger`

### test_regime_concentration.py

#### Broad Exceptions

Line 60: `except Exception as e:`

### test_enhanced_data_quality_fixes.py

#### Unused Imports

Line 9: `from datetime import timedelta`
Line 36: `from src.training.steps.raw_data_quality_checker import validate_raw_data_quality`
Line 36: `from src.training.steps.raw_data_quality_checker import fix_irregular_intervals_automatically`

### test_enhanced_data_quality_fix.py

#### Debug Statements

Line 78: `print("\n🔍 Analyzing interval issues...")`
Line 255: `print(f"🔍 Analyzing patterns for {symbol} on {exchange}")`

#### Broad Exceptions

Line 329: `except Exception as e:`

### test_hmm_manager_only.py

#### Debug Statements

Line 27: `print("🔍 Testing loading non-existent HMM composite clusters...")`
Line 37: `print("🔍 Testing file existence check...")`

#### Broad Exceptions

Line 79: `except Exception as e:`

### simulate_regime_merging_optimization.py

#### Debug Statements

Line 112: `print("🔍 Loading regime data...")`
Line 190: `print("\n🔍 CLOSEST TO TARGET (75%):")`

#### Broad Exceptions

Line 87: `except:`
Line 156: `except Exception as e:`

### test_enhanced_data_quality_simple.py

#### Debug Statements

Line 30: `print(f"[DEBUG] {msg}")`

#### Broad Exceptions

Line 290: `except Exception as e:`

#### Unused Imports

Line 7: `import sys`
Line 8: `import os`
Line 10: `from typing import Dict`
Line 10: `from typing import List`
Line 10: `from typing import Callable`
Line 10: `from typing import Union`
Line 10: `from typing import Tuple`
Line 11: `from datetime import datetime`
Line 11: `from datetime import timedelta`

#### Todo Comments

Line 29: `def debug(self, msg):`
Line 30: `print(f"[DEBUG] {msg}")`

### run_pipeline_simple.py

#### Broad Exceptions

Line 63: `except Exception as e:`

### test_sr_feature_categorization.py

#### Broad Exceptions

Line 152: `except Exception as e:`

### diagnose_feature_pipeline.py

#### Debug Statements

Line 39: `print("\n🔍 ROOT CAUSE ANALYSIS:")`
Line 79: `print("\n🔧 DEBUGGING STEPS:")`

#### Unused Imports

Line 10: `from pathlib import Path`
Line 11: `import json`
Line 12: `from typing import List`
Line 12: `from typing import Dict`
Line 12: `from typing import Any`

#### Todo Comments

Line 79: `print("\n🔧 DEBUGGING STEPS:")`

### test_feature_artifacts.py

#### Broad Exceptions

Line 62: `except Exception as e:`
Line 83: `except Exception as e:`
Line 110: `except Exception as e:`
Line 131: `except Exception as e:`

#### Unused Imports

Line 16: `from datetime import datetime`
Line 24: `from src.training.steps.step2_feature_engineering import run_step`

### scripts/verify_timeframe_data.py

#### Broad Exceptions

Line 135: `except Exception as e:`
Line 182: `except Exception as e:`
Line 314: `except Exception as e:`

#### Unused Imports

Line 11: `import os`
Line 13: `from typing import List`
Line 15: `from datetime import timedelta`

### scripts/update_entire_repository_logging.py

#### Debug Statements

Line 415: `print(f"🔍 Found {len(python_files)} Python files to process")`

#### Broad Exceptions

Line 291: `except Exception as e:`
Line 380: `except Exception as e:`
Line 401: `except Exception as e:`

#### Unused Imports

Line 9: `import os`
Line 13: `from typing import Set`
Line 19: `from src.utils.warning_symbols import error`
Line 19: `from src.utils.warning_symbols import critical`
Line 19: `from src.utils.warning_symbols import problem`
Line 19: `from src.utils.warning_symbols import failed`
Line 19: `from src.utils.warning_symbols import invalid`
Line 19: `from src.utils.warning_symbols import missing`
Line 19: `from src.utils.warning_symbols import timeout`
Line 19: `from src.utils.warning_symbols import connection_error`
... and 3 more

### scripts/diagnose_feature_quality.py

#### Debug Statements

Line 480: `print("🔍 Analyzing feature quality...")`

#### Broad Exceptions

Line 504: `except Exception as e:`

#### Unused Imports

Line 14: `from typing import Tuple`
Line 23: `from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering`
Line 26: `from src.training.steps.step1_7_hmm_regime_discovery import _select_block_features`
Line 26: `from src.training.steps.step1_7_hmm_regime_discovery import _transform_for_hmm`

### scripts/verify_feature_calculations.py

#### Debug Statements

Line 537: `print("🔍 Verifying momentum features...")`
Line 540: `print("🔍 Verifying volatility features...")`
Line 543: `print("🔍 Verifying liquidity features...")`
Line 546: `print("🔍 Verifying variance thresholds...")`

#### Broad Exceptions

Line 558: `except Exception as e:`

#### Unused Imports

Line 14: `from typing import Tuple`

### scripts/launch_advanced_monitoring.py

#### Broad Exceptions

Line 144: `except Exception as e:`
Line 167: `except Exception as e:`
Line 195: `except Exception as e:`
Line 199: `except Exception as e:`
Line 274: `except Exception as e:`
Line 325: `except Exception as e:`
Line 340: `except Exception as e:`
Line 361: `except Exception as e:`
Line 378: `except Exception as e:`

#### Unused Imports

Line 21: `from src.utils.warning_symbols import critical`
Line 21: `from src.utils.warning_symbols import problem`
Line 21: `from src.utils.warning_symbols import invalid`
Line 21: `from src.utils.warning_symbols import missing`
Line 21: `from src.utils.warning_symbols import timeout`
Line 21: `from src.utils.warning_symbols import connection_error`
Line 21: `from src.utils.warning_symbols import validation_error`
Line 21: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 86: `"enable_debug_aggregation": True,`

### scripts/validate_fix_simple.py

#### Debug Statements

Line 19: `print("🔍 Validating multicollinearity fix...")`
Line 91: `print("🔍 Checking feature selection configuration...")`
Line 154: `print("\n🔍 Next steps:")`

#### Broad Exceptions

Line 83: `except Exception as e:`
Line 125: `except Exception as e:`

### scripts/run_enhanced_training.py

#### Unused Imports

Line 29: `from src.utils.warning_symbols import error`
Line 29: `from src.utils.warning_symbols import warning`
Line 29: `from src.utils.warning_symbols import critical`
Line 29: `from src.utils.warning_symbols import problem`
Line 29: `from src.utils.warning_symbols import invalid`
Line 29: `from src.utils.warning_symbols import missing`
Line 29: `from src.utils.warning_symbols import timeout`
Line 29: `from src.utils.warning_symbols import connection_error`
Line 29: `from src.utils.warning_symbols import validation_error`
Line 29: `from src.utils.warning_symbols import initialization_error`
... and 1 more

### scripts/test_nan_fix.py

#### Broad Exceptions

Line 133: `except Exception as e:`
Line 138: `except Exception as e:`
Line 199: `except Exception as e:`
Line 236: `except Exception as e:`

### scripts/setup_challenger_model.py

#### Broad Exceptions

Line 58: `except Exception as e:`
Line 68: `except Exception as e:`
Line 118: `except Exception as e:`
Line 138: `except Exception as e:`

#### Unused Imports

Line 25: `from src.utils.warning_symbols import warning`
Line 25: `from src.utils.warning_symbols import critical`
Line 25: `from src.utils.warning_symbols import problem`
Line 25: `from src.utils.warning_symbols import failed`
Line 25: `from src.utils.warning_symbols import invalid`
Line 25: `from src.utils.warning_symbols import timeout`
Line 25: `from src.utils.warning_symbols import connection_error`
Line 25: `from src.utils.warning_symbols import validation_error`
Line 25: `from src.utils.warning_symbols import initialization_error`
Line 25: `from src.utils.warning_symbols import execution_error`

### scripts/training_cli.py

#### Broad Exceptions

Line 178: `except Exception as e:`
Line 216: `except Exception as e:`
Line 377: `except Exception as e:`
Line 680: `except Exception as e:`
Line 711: `except Exception as e:`

### scripts/test_sr_level_fix.py

#### Broad Exceptions

Line 200: `except Exception as e:`

### scripts/train_multi_timeframe_hmm_ensemble.py

#### Broad Exceptions

Line 55: `except Exception as e:`
Line 137: `except Exception as e:`
Line 282: `except Exception as e:`

#### Unused Imports

Line 15: `from typing import Any`
Line 23: `from datetime import datetime`
Line 23: `from datetime import timedelta`
Line 31: `from src.config import CONFIG`

### scripts/assess_data_quality.py

#### Debug Statements

Line 549: `print("\n🔍 Performing data quality assessment...")`
Line 562: `print("\n🔍 Creating sample labeled dataset for enhanced analysis...")`
Line 586: `print("\n🔍 Analyzing multicollinearity...")`
Line 592: `print("\n🔍 Analyzing label distribution...")`
Line 691: `print(f"   3. 🔍 MONITOR RESULTS:")`

#### Broad Exceptions

Line 173: `except Exception as e:`
Line 299: `except Exception as e:`
Line 501: `except Exception as e:`

#### Unused Imports

Line 16: `import os`
Line 26: `from typing import Tuple`
Line 28: `from sklearn.feature_selection import mutual_info_classif`

### scripts/validate_multicollinearity_fix.py

#### Debug Statements

Line 38: `print("🔍 Validating multicollinearity fix...")`
Line 94: `print("🔍 Checking for perfect correlations...")`
Line 127: `print("🔍 Checking specific problematic features...")`

#### Broad Exceptions

Line 150: `except Exception as e:`

#### Unused Imports

Line 13: `import os`

### scripts/compare_agg_trades_formats.py

#### Broad Exceptions

Line 98: `except Exception as e:`
Line 203: `except Exception as e:`

#### Unused Imports

Line 20: `from src.utils.warning_symbols import critical`
Line 20: `from src.utils.warning_symbols import problem`
Line 20: `from src.utils.warning_symbols import invalid`
Line 20: `from src.utils.warning_symbols import timeout`
Line 20: `from src.utils.warning_symbols import connection_error`
Line 20: `from src.utils.warning_symbols import validation_error`
Line 20: `from src.utils.warning_symbols import initialization_error`
Line 20: `from src.utils.warning_symbols import execution_error`

### scripts/test_critical_fixes.py

#### Broad Exceptions

Line 219: `except Exception as e:`

### scripts/fix_multicollinearity_issue.py

#### Broad Exceptions

Line 80: `except Exception as e:`

#### Unused Imports

Line 13: `import os`

#### Todo Comments

Line 5: `This script fixes the critical bug where all multi-timeframe price_change and volume_change`

### scripts/download_missing_timeframes.py

#### Broad Exceptions

Line 69: `except Exception as e:`
Line 192: `except Exception as e:`

#### Unused Imports

Line 11: `import os`
Line 14: `from typing import List`

### scripts/update_logging_warnings.py

#### Debug Statements

Line 215: `print(f"🔍 Found {len(python_files)} Python files in training steps directory")`

#### Broad Exceptions

Line 141: `except Exception as e:`
Line 199: `except Exception as e:`

#### Unused Imports

Line 9: `import os`
Line 13: `from typing import List`
Line 19: `from src.utils.warning_symbols import error`
Line 19: `from src.utils.warning_symbols import critical`
Line 19: `from src.utils.warning_symbols import problem`
Line 19: `from src.utils.warning_symbols import failed`
Line 19: `from src.utils.warning_symbols import invalid`
Line 19: `from src.utils.warning_symbols import timeout`
Line 19: `from src.utils.warning_symbols import connection_error`
Line 19: `from src.utils.warning_symbols import validation_error`
... and 2 more

### scripts/launch_with_monitoring.py

#### Unused Imports

Line 7: `from src.utils.warning_symbols import error`
Line 7: `from src.utils.warning_symbols import critical`
Line 7: `from src.utils.warning_symbols import problem`
Line 7: `from src.utils.warning_symbols import failed`
Line 7: `from src.utils.warning_symbols import invalid`
Line 7: `from src.utils.warning_symbols import missing`
Line 7: `from src.utils.warning_symbols import timeout`
Line 7: `from src.utils.warning_symbols import connection_error`
Line 7: `from src.utils.warning_symbols import validation_error`
Line 7: `from src.utils.warning_symbols import initialization_error`
... and 1 more

### scripts/blank_training_run.py

#### Debug Statements

Line 230: `print("🔍 Running data quality validation...")`

#### Broad Exceptions

Line 348: `except Exception as e:`
Line 382: `except Exception as e:`
Line 421: `except Exception as e:`

#### Unused Imports

Line 30: `from src.utils.warning_symbols import warning`
Line 30: `from src.utils.warning_symbols import problem`
Line 30: `from src.utils.warning_symbols import invalid`
Line 30: `from src.utils.warning_symbols import timeout`
Line 30: `from src.utils.warning_symbols import connection_error`
Line 30: `from src.utils.warning_symbols import validation_error`
Line 30: `from src.utils.warning_symbols import initialization_error`
Line 30: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 389: `# Log additional debugging information`

### scripts/show_timeframe_config.py

#### Unused Imports

Line 11: `from src.utils.warning_symbols import error`
Line 11: `from src.utils.warning_symbols import warning`
Line 11: `from src.utils.warning_symbols import critical`
Line 11: `from src.utils.warning_symbols import problem`
Line 11: `from src.utils.warning_symbols import failed`
Line 11: `from src.utils.warning_symbols import invalid`
Line 11: `from src.utils.warning_symbols import timeout`
Line 11: `from src.utils.warning_symbols import connection_error`
Line 11: `from src.utils.warning_symbols import validation_error`
Line 11: `from src.utils.warning_symbols import initialization_error`
... and 1 more

### scripts/analyze_timeframe_incremental.py

#### Broad Exceptions

Line 215: `except Exception as e:`
Line 888: `except Exception as e:`
Line 905: `except Exception as e:`

### scripts/fix_consolidated_data.py

#### Broad Exceptions

Line 52: `except Exception as e:`

#### Unused Imports

Line 8: `from src.utils.warning_symbols import error`
Line 8: `from src.utils.warning_symbols import critical`
Line 8: `from src.utils.warning_symbols import problem`
Line 8: `from src.utils.warning_symbols import invalid`
Line 8: `from src.utils.warning_symbols import missing`
Line 8: `from src.utils.warning_symbols import timeout`
Line 8: `from src.utils.warning_symbols import connection_error`
Line 8: `from src.utils.warning_symbols import validation_error`
Line 8: `from src.utils.warning_symbols import initialization_error`
Line 8: `from src.utils.warning_symbols import execution_error`

### scripts/run_scans.py

#### Broad Exceptions

Line 212: `except Exception as e:`

#### Unused Imports

Line 11: `from src.utils.warning_symbols import critical`
Line 11: `from src.utils.warning_symbols import problem`
Line 11: `from src.utils.warning_symbols import invalid`
Line 11: `from src.utils.warning_symbols import missing`
Line 11: `from src.utils.warning_symbols import connection_error`
Line 11: `from src.utils.warning_symbols import validation_error`
Line 11: `from src.utils.warning_symbols import initialization_error`
Line 11: `from src.utils.warning_symbols import execution_error`

### scripts/fix_corrupted_data.py

#### Debug Statements

Line 204: `print(f"\n🔍 Found {len(pkl_files)} existing pickle files")`

#### Broad Exceptions

Line 163: `except Exception as e:`
Line 221: `except Exception as e:`

#### Unused Imports

Line 23: `from src.utils.warning_symbols import error`
Line 23: `from src.utils.warning_symbols import critical`
Line 23: `from src.utils.warning_symbols import problem`
Line 23: `from src.utils.warning_symbols import failed`
Line 23: `from src.utils.warning_symbols import invalid`
Line 23: `from src.utils.warning_symbols import timeout`
Line 23: `from src.utils.warning_symbols import connection_error`
Line 23: `from src.utils.warning_symbols import validation_error`
Line 23: `from src.utils.warning_symbols import initialization_error`
Line 23: `from src.utils.warning_symbols import execution_error`

### scripts/database_migration.py

#### Debug Statements

Line 74: `print(f"🔍 Checksum: {checksum}")`
Line 140: `print("🔍 File Validation Results:")`

#### Broad Exceptions

Line 85: `except Exception as e:`
Line 124: `except Exception as e:`
Line 166: `except Exception as e:`
Line 190: `except Exception as e:`
Line 229: `except Exception as e:`
Line 249: `except Exception as e:`

#### Unused Imports

Line 32: `from src.utils.logger import system_logger`
Line 33: `from src.utils.warning_symbols import critical`
Line 33: `from src.utils.warning_symbols import problem`
Line 33: `from src.utils.warning_symbols import invalid`
Line 33: `from src.utils.warning_symbols import missing`
Line 33: `from src.utils.warning_symbols import timeout`
Line 33: `from src.utils.warning_symbols import connection_error`
Line 33: `from src.utils.warning_symbols import validation_error`
Line 33: `from src.utils.warning_symbols import initialization_error`
Line 33: `from src.utils.warning_symbols import execution_error`

### scripts/consolidate_binance_15m.py

#### Broad Exceptions

Line 119: `except Exception as e:`

#### Unused Imports

Line 19: `from src.utils.warning_symbols import critical`
Line 19: `from src.utils.warning_symbols import problem`
Line 19: `from src.utils.warning_symbols import failed`
Line 19: `from src.utils.warning_symbols import timeout`
Line 19: `from src.utils.warning_symbols import connection_error`
Line 19: `from src.utils.warning_symbols import validation_error`
Line 19: `from src.utils.warning_symbols import initialization_error`
Line 19: `from src.utils.warning_symbols import execution_error`

### scripts/simple_data_loader.py

#### Broad Exceptions

Line 104: `except Exception as e:`

#### Unused Imports

Line 18: `from src.utils.warning_symbols import warning`
Line 18: `from src.utils.warning_symbols import critical`
Line 18: `from src.utils.warning_symbols import problem`
Line 18: `from src.utils.warning_symbols import invalid`
Line 18: `from src.utils.warning_symbols import missing`
Line 18: `from src.utils.warning_symbols import timeout`
Line 18: `from src.utils.warning_symbols import connection_error`
Line 18: `from src.utils.warning_symbols import validation_error`
Line 18: `from src.utils.warning_symbols import initialization_error`
Line 18: `from src.utils.warning_symbols import execution_error`

### scripts/configure_optimization_settings.py

#### Debug Statements

Line 221: `print(f"\n🔍 Sampling Strategy: {bayesian_config['sampling_strategy']}")`
Line 228: `print("\n🔍 Search Spaces:")`
Line 350: `print("\n🔍 Example 2: Bayesian Optimization")`

#### Broad Exceptions

Line 87: `except Exception as e:`
Line 342: `except Exception as e:`
Line 365: `except Exception as e:`
Line 387: `except Exception as e:`
Line 485: `except Exception as e:`

#### Unused Imports

Line 10: `from src.utils.warning_symbols import critical`
Line 10: `from src.utils.warning_symbols import problem`
Line 10: `from src.utils.warning_symbols import invalid`
Line 10: `from src.utils.warning_symbols import missing`
Line 10: `from src.utils.warning_symbols import timeout`
Line 10: `from src.utils.warning_symbols import connection_error`
Line 10: `from src.utils.warning_symbols import validation_error`
Line 10: `from src.utils.warning_symbols import execution_error`

### scripts/run_enhanced_backtesting.py

#### Broad Exceptions

Line 111: `except Exception as e:`

#### Unused Imports

Line 25: `from src.utils.warning_symbols import error`
Line 25: `from src.utils.warning_symbols import warning`
Line 25: `from src.utils.warning_symbols import critical`
Line 25: `from src.utils.warning_symbols import problem`
Line 25: `from src.utils.warning_symbols import invalid`
Line 25: `from src.utils.warning_symbols import missing`
Line 25: `from src.utils.warning_symbols import timeout`
Line 25: `from src.utils.warning_symbols import connection_error`
Line 25: `from src.utils.warning_symbols import validation_error`
Line 25: `from src.utils.warning_symbols import initialization_error`
... and 1 more

### scripts/analyze_timeframe.py

#### Broad Exceptions

Line 176: `except Exception as e:`
Line 527: `except Exception as e:`
Line 892: `except Exception as e:`
Line 909: `except Exception as e:`

### scripts/analyze_hmm_regimes.py

#### Debug Statements

Line 1373: `print("🔍 HMM REGIME ANALYSIS SUMMARY")`
Line 1724: `print(f"🔍 Analyzing HMM regimes for {exchange}_{symbol}_{timeframe}...")`

#### Broad Exceptions

Line 413: `except Exception as e:`
Line 1793: `except Exception as e:`
Line 1869: `except Exception as e:`
Line 1915: `except Exception as e:`
Line 1981: `except Exception as e:`
Line 2067: `except Exception as e:`
Line 2127: `except Exception as e:`
Line 2222: `except Exception as e:`

#### Unused Imports

Line 10: `import os`
Line 19: `from datetime import timedelta`
Line 1888: `from matplotlib.patches import Circle`

### scripts/regenerate_pickle_files.py

#### Broad Exceptions

Line 136: `except Exception as e:`
Line 202: `except Exception as e:`

#### Unused Imports

Line 11: `from src.utils.warning_symbols import error`
Line 11: `from src.utils.warning_symbols import critical`
Line 11: `from src.utils.warning_symbols import problem`
Line 11: `from src.utils.warning_symbols import failed`
Line 11: `from src.utils.warning_symbols import invalid`
Line 11: `from src.utils.warning_symbols import timeout`
Line 11: `from src.utils.warning_symbols import connection_error`
Line 11: `from src.utils.warning_symbols import validation_error`
Line 11: `from src.utils.warning_symbols import initialization_error`
Line 11: `from src.utils.warning_symbols import execution_error`

### scripts/fix_multicollinearity.py

#### Debug Statements

Line 223: `print("\n🔍 ROOT CAUSE:")`

#### Unused Imports

Line 13: `import sys`
Line 14: `from pathlib import Path`

### scripts/test_vif_fixes.py

#### Broad Exceptions

Line 195: `except:`
Line 221: `except:`
Line 270: `except Exception as e:`

### scripts/download_mexc_agg_trades.py

#### Broad Exceptions

Line 169: `except Exception as e:`

#### Unused Imports

Line 22: `from src.utils.warning_symbols import critical`
Line 22: `from src.utils.warning_symbols import problem`
Line 22: `from src.utils.warning_symbols import invalid`
Line 22: `from src.utils.warning_symbols import timeout`
Line 22: `from src.utils.warning_symbols import connection_error`
Line 22: `from src.utils.warning_symbols import validation_error`
Line 22: `from src.utils.warning_symbols import initialization_error`
Line 22: `from src.utils.warning_symbols import execution_error`

### scripts/migrate_parquet_datasets.py

#### Unused Imports

Line 17: `from __future__ import annotations`

### scripts/run_event_bus_example.py

#### Broad Exceptions

Line 45: `except Exception as e:`

#### Unused Imports

Line 14: `from src.utils.warning_symbols import error`
Line 14: `from src.utils.warning_symbols import critical`
Line 14: `from src.utils.warning_symbols import problem`
Line 14: `from src.utils.warning_symbols import failed`
Line 14: `from src.utils.warning_symbols import invalid`
Line 14: `from src.utils.warning_symbols import missing`
Line 14: `from src.utils.warning_symbols import timeout`
Line 14: `from src.utils.warning_symbols import connection_error`
Line 14: `from src.utils.warning_symbols import validation_error`
Line 14: `from src.utils.warning_symbols import initialization_error`
... and 1 more

### scripts/resume_training.py

#### Unused Imports

Line 28: `from src.training.steps.step1_data_collection import run_step`
Line 32: `from src.utils.warning_symbols import warning`
Line 32: `from src.utils.warning_symbols import critical`
Line 32: `from src.utils.warning_symbols import problem`
Line 32: `from src.utils.warning_symbols import invalid`
Line 32: `from src.utils.warning_symbols import missing`
Line 32: `from src.utils.warning_symbols import timeout`
Line 32: `from src.utils.warning_symbols import connection_error`
Line 32: `from src.utils.warning_symbols import validation_error`
Line 32: `from src.utils.warning_symbols import initialization_error`
... and 1 more

### scripts/run_feature_diagnostic.py

#### Broad Exceptions

Line 704: `except Exception as e:`

#### Unused Imports

Line 14: `from typing import Tuple`
Line 23: `from src.utils.data_quality_validator import DataQualityValidator`

### scripts/bot_monitor.py

#### Broad Exceptions

Line 49: `except Exception as e:`
Line 58: `except Exception as e:`
Line 122: `except Exception as e:`
Line 156: `except Exception as e:`
Line 223: `except Exception as e:`

#### Unused Imports

Line 19: `from src.utils.warning_symbols import critical`
Line 19: `from src.utils.warning_symbols import problem`
Line 19: `from src.utils.warning_symbols import failed`
Line 19: `from src.utils.warning_symbols import invalid`
Line 19: `from src.utils.warning_symbols import missing`
Line 19: `from src.utils.warning_symbols import timeout`
Line 19: `from src.utils.warning_symbols import connection_error`
Line 19: `from src.utils.warning_symbols import validation_error`
Line 19: `from src.utils.warning_symbols import initialization_error`
Line 19: `from src.utils.warning_symbols import execution_error`

### scripts/fix_multicollinearity_simple.py

#### Debug Statements

Line 91: `print("\n🔍 Next steps:")`

#### Broad Exceptions

Line 76: `except Exception as e:`

#### Unused Imports

Line 12: `import os`

#### Todo Comments

Line 5: `This script fixes the critical bug where all multi-timeframe price_change and volume_change`

### scripts/check_notifications.py

#### Debug Statements

Line 30: `print("🔍 Checking for ARES Bot notifications...")`

#### Broad Exceptions

Line 57: `except Exception as e:`
Line 75: `except Exception as e:`
Line 115: `except Exception as e:`

#### Unused Imports

Line 7: `from src.utils.warning_symbols import error`
Line 7: `from src.utils.warning_symbols import critical`
Line 7: `from src.utils.warning_symbols import problem`
Line 7: `from src.utils.warning_symbols import failed`
Line 7: `from src.utils.warning_symbols import invalid`
Line 7: `from src.utils.warning_symbols import timeout`
Line 7: `from src.utils.warning_symbols import connection_error`
Line 7: `from src.utils.warning_symbols import validation_error`
Line 7: `from src.utils.warning_symbols import initialization_error`
Line 7: `from src.utils.warning_symbols import execution_error`

### scripts/rename_data_files.py

#### Broad Exceptions

Line 102: `except Exception as e:`

#### Unused Imports

Line 8: `from src.utils.warning_symbols import error`
Line 8: `from src.utils.warning_symbols import critical`
Line 8: `from src.utils.warning_symbols import problem`
Line 8: `from src.utils.warning_symbols import invalid`
Line 8: `from src.utils.warning_symbols import timeout`
Line 8: `from src.utils.warning_symbols import connection_error`
Line 8: `from src.utils.warning_symbols import validation_error`
Line 8: `from src.utils.warning_symbols import initialization_error`
Line 8: `from src.utils.warning_symbols import execution_error`

### scripts/investigate_regime_calculations.py

#### Broad Exceptions

Line 303: `except Exception:`
Line 326: `except Exception:`
Line 350: `except Exception:`
Line 498: `except Exception as e:`

#### Unused Imports

Line 14: `from typing import Tuple`

### backtesting/ares_data_downloader_clean.py

#### Debug Statements

Line 188: `print("🔍 DEBUG: No existing aggtrades files found")`
Line 204: `print(f"🔍 DEBUG: Cannot read {file_path}")`
Line 250: `print(f"🔍 DEBUG: Error reading {file_path}: {e}")`
Line 259: `print("🔍 DEBUG: No valid timestamps found in existing files")`
Line 263: `print(f"🔍 DEBUG: Error finding latest timestamp: {e}")`
Line 275: `print(f"🔍 DEBUG: Found latest aggtrades timestamp: {latest_timestamp}")`
Line 354: `print(f"🔍 DEBUG: Found latest timestamp: {latest_timestamp}")`
Line 358: `print(f"🔍 DEBUG: Starting download from: {start_date}")`
Line 375: `print(f"🔍 DEBUG: Starting daily period loop from {current} to {end_date}")`
Line 381: `print(f"🔍 DEBUG: Checking file: {filename}")`
... and 3 more

#### Broad Exceptions

Line 164: `except Exception as e:`
Line 176: `except Exception as e:`
Line 249: `except Exception as e:`
Line 262: `except Exception as e:`
Line 659: `except Exception as e:`
Line 741: `except Exception as e:`
Line 815: `except Exception as e:`
Line 852: `except Exception as e:`
Line 890: `except Exception as e:`
Line 894: `except Exception as e:`
... and 2 more

#### Unused Imports

Line 37: `from src.utils.logger import setup_logging`
Line 38: `from src.utils.warning_symbols import connection_error`
Line 38: `from src.utils.warning_symbols import error`
Line 38: `from src.utils.warning_symbols import execution_error`
Line 38: `from src.utils.warning_symbols import initialization_error`
Line 38: `from src.utils.warning_symbols import invalid`
Line 38: `from src.utils.warning_symbols import missing`
Line 38: `from src.utils.warning_symbols import problem`
Line 38: `from src.utils.warning_symbols import timeout`
Line 38: `from src.utils.warning_symbols import validation_error`
... and 1 more

#### Todo Comments

Line 188: `print("🔍 DEBUG: No existing aggtrades files found")`
Line 204: `print(f"🔍 DEBUG: Cannot read {file_path}")`
Line 250: `print(f"🔍 DEBUG: Error reading {file_path}: {e}")`
Line 255: `f"🔍 DEBUG: Latest timestamp found: {latest_timestamp} from {latest_file}"`
Line 259: `print("🔍 DEBUG: No valid timestamps found in existing files")`
Line 263: `print(f"🔍 DEBUG: Error finding latest timestamp: {e}")`
Line 275: `print(f"🔍 DEBUG: Found latest aggtrades timestamp: {latest_timestamp}")`
Line 354: `print(f"🔍 DEBUG: Found latest timestamp: {latest_timestamp}")`
Line 358: `print(f"🔍 DEBUG: Starting download from: {start_date}")`
Line 375: `print(f"🔍 DEBUG: Starting daily period loop from {current} to {end_date}")`
... and 5 more

### backtesting/ares_data_downloader_optimized.py

#### Debug Statements

Line 334: `print(f"🔍 DEBUG: Exchange name: {self.config.exchange.lower()}")`
Line 335: `print(f"🔍 DEBUG: ExchangeFactory available: {ExchangeFactory is not None}")`
Line 336: `print(f"🔍 DEBUG: ExchangeFactory methods: {dir(ExchangeFactory)}")`
Line 343: `print("🔍 DEBUG: Exchange client created successfully")`
Line 344: `print(f"🔍 DEBUG: Exchange client type: {type(self.exchange_client)}")`
Line 345: `print(f"🔍 DEBUG: Exchange client methods: {dir(self.exchange_client)}")`
Line 353: `print(f"🔍 DEBUG: Failed to create exchange client: {e}")`
Line 354: `print(f"🔍 DEBUG: Error type: {type(e)}")`
Line 421: `print("🔍 DEBUG: No existing aggtrades files found")`
Line 472: `print(f"🔍 DEBUG: Force mode: {getattr(self.config, 'force', False)}")`
... and 23 more

#### Type Ignores

Line 615: `)  # type: ignore[arg-type]`

#### Broad Exceptions

Line 172: `except Exception:`
Line 186: `except Exception:`
Line 310: `except Exception as e:`
Line 352: `except Exception as e:`
Line 382: `except Exception as e:`
Line 410: `except Exception:`
Line 454: `except Exception:`
Line 461: `except Exception:`
Line 466: `except Exception:`
Line 488: `except Exception as e:`
... and 11 more

#### Unused Imports

Line 56: `from src.utils.error_handler import handle_file_operations`
Line 56: `from src.utils.error_handler import handle_network_operations`
Line 61: `from src.utils.warning_symbols import error`
Line 61: `from src.utils.warning_symbols import warning`
Line 61: `from src.utils.warning_symbols import problem`
Line 61: `from src.utils.warning_symbols import invalid`
Line 61: `from src.utils.warning_symbols import connection_error`
Line 61: `from src.utils.warning_symbols import validation_error`
Line 61: `from src.utils.warning_symbols import initialization_error`
Line 61: `from src.utils.warning_symbols import execution_error`
... and 3 more

#### Todo Comments

Line 334: `print(f"🔍 DEBUG: Exchange name: {self.config.exchange.lower()}")`
Line 335: `print(f"🔍 DEBUG: ExchangeFactory available: {ExchangeFactory is not None}")`
Line 336: `print(f"🔍 DEBUG: ExchangeFactory methods: {dir(ExchangeFactory)}")`
Line 343: `print("🔍 DEBUG: Exchange client created successfully")`
Line 344: `print(f"🔍 DEBUG: Exchange client type: {type(self.exchange_client)}")`
Line 345: `print(f"🔍 DEBUG: Exchange client methods: {dir(self.exchange_client)}")`
Line 353: `print(f"🔍 DEBUG: Failed to create exchange client: {e}")`
Line 354: `print(f"🔍 DEBUG: Error type: {type(e)}")`
Line 421: `print("🔍 DEBUG: No existing aggtrades files found")`
Line 472: `print(f"🔍 DEBUG: Force mode: {getattr(self.config, 'force', False)}")`
... and 47 more

### src/tasks.py

#### Broad Exceptions

Line 73: `except Exception as e:`

### src/ares_pipeline.py

#### Debug Statements

Line 614: `print("   🔍 Executing market analysis...")`

#### Broad Exceptions

Line 141: `except Exception:`
Line 175: `except Exception as e:`
Line 200: `except Exception as e:`
Line 220: `except Exception as e:`
Line 233: `except Exception as e:`
Line 250: `except Exception as e:`
Line 267: `except Exception as e:`
Line 284: `except Exception as e:`
Line 301: `except Exception as e:`
Line 314: `except Exception as e:`
... and 23 more

#### Unused Imports

Line 38: `from src.utils.warning_symbols import execution_error`
Line 38: `from src.utils.warning_symbols import initialization_error`
Line 38: `from src.utils.warning_symbols import problem`
Line 208: `from exchange.factory import ExchangeFactory`

### src/config.py

#### Broad Exceptions

Line 211: `except Exception as e:`
Line 228: `except Exception as e:`
Line 254: `except Exception:`
Line 276: `except Exception as e:`
Line 293: `except Exception:`
Line 326: `except Exception:`
Line 354: `except Exception:`
Line 370: `except Exception:`
Line 392: `except Exception:`
Line 406: `except Exception:`
... and 2 more

#### Unused Imports

Line 11: `from src.config import CONFIG`
Line 11: `from src.config import AresConfig`
Line 11: `from src.config import get_lookback_window`
Line 69: `from src.config.environment import get_environment_settings`

### src/paper_trader.py

#### Broad Exceptions

Line 88: `except Exception as e:`
Line 118: `except Exception as e:`
Line 159: `except Exception as e:`
Line 186: `except Exception as e:`
Line 279: `except Exception as e:`
Line 379: `except Exception as e:`
Line 428: `except Exception as e:`
Line 450: `except Exception as e:`
Line 463: `except Exception as e:`
Line 479: `except Exception as e:`
... and 6 more

### src/monitoring/error_detection_system.py

#### Broad Exceptions

Line 330: `except Exception as e:`
Line 355: `except Exception:`
Line 445: `except Exception:`
Line 482: `except Exception:`
Line 558: `except Exception:`
Line 572: `except Exception:`
Line 605: `except Exception:`
Line 633: `except Exception:`
Line 666: `except Exception:`
Line 684: `except Exception:`
... and 27 more

### src/monitoring/monitoring_integration_example.py

#### Broad Exceptions

Line 80: `except Exception:`
Line 133: `except Exception as e:`
Line 223: `except Exception:`
Line 254: `except Exception:`
Line 287: `except Exception:`
Line 394: `except Exception:`
Line 471: `except Exception:`
Line 563: `except Exception:`
Line 639: `except Exception as e:`
Line 732: `except Exception:`
... and 4 more

### src/monitoring/tracking_system.py

#### Broad Exceptions

Line 227: `except Exception:`
Line 250: `except Exception:`
Line 265: `except Exception:`
Line 280: `except Exception:`
Line 295: `except Exception:`
Line 310: `except Exception:`
Line 325: `except Exception:`
Line 362: `except Exception:`
Line 378: `except Exception:`
Line 393: `except Exception:`
... and 12 more

#### Todo Comments

Line 459: `self.logger.debug(f"Tracked ensemble decision: {decision.decision_id}")`
Line 497: `self.logger.debug(f"Tracked regime analysis: {analysis.analysis_id}")`
Line 550: `self.logger.debug("Tracked feature importance")`
Line 581: `self.logger.debug(f"Tracked decision path: {path.path_id}")`
Line 619: `self.logger.debug("Tracked model behavior")`

### src/monitoring/correlation_manager.py

#### Broad Exceptions

Line 103: `except Exception:`
Line 144: `except Exception:`
Line 200: `except Exception:`
Line 261: `except Exception:`
Line 290: `except Exception:`
Line 308: `except Exception:`
Line 323: `except Exception:`
Line 351: `except Exception:`

#### Todo Comments

Line 142: `self.logger.debug(f"Tracking correlation request: {correlation_id}")`
Line 198: `self.logger.debug(f"Tracked correlation response: {correlation_id}")`

### src/monitoring/report_scheduler.py

#### Broad Exceptions

Line 129: `except Exception:`
Line 177: `except Exception:`
Line 198: `except Exception:`
Line 215: `except Exception:`
Line 257: `except Exception:`
Line 280: `except Exception as e:`
Line 296: `except Exception:`
Line 321: `except Exception:`
Line 336: `except Exception:`
Line 357: `except Exception:`
... and 7 more

### src/monitoring/enhanced_ml_tracker.py

#### Broad Exceptions

Line 338: `except Exception as e:`
Line 364: `except Exception as e:`
Line 441: `except Exception as e:`
Line 470: `except Exception as e:`
Line 483: `except Exception as e:`
Line 502: `except Exception as e:`
Line 585: `except Exception as e:`
Line 609: `except Exception as e:`
Line 701: `except Exception as e:`
Line 725: `except Exception as e:`
... and 18 more

#### Unused Imports

Line 21: `from src.utils.warning_symbols import critical`
Line 21: `from src.utils.warning_symbols import problem`
Line 21: `from src.utils.warning_symbols import invalid`
Line 21: `from src.utils.warning_symbols import missing`
Line 21: `from src.utils.warning_symbols import timeout`
Line 21: `from src.utils.warning_symbols import connection_error`
Line 21: `from src.utils.warning_symbols import validation_error`
Line 21: `from src.utils.warning_symbols import initialization_error`
Line 21: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 579: `self.logger.debug(`
Line 697: `self.logger.debug(f"Tracked ensemble performance {ensemble_id}")`
Line 779: `self.logger.debug(f"Recorded outcome for prediction {prediction_id}")`
Line 1156: `self.logger.debug("Completed periodic performance analysis")`
Line 1172: `self.logger.debug("Completed periodic model comparison")`

### src/monitoring/performance_dashboard.py

#### Broad Exceptions

Line 205: `except Exception:`
Line 251: `except Exception:`
Line 283: `except Exception as e:`
Line 333: `except Exception:`
Line 401: `except Exception as e:`
Line 409: `except Exception as e:`
Line 499: `except Exception:`
Line 534: `except Exception:`
Line 569: `except Exception:`
Line 603: `except Exception as e:`
... and 2 more

#### Todo Comments

Line 201: `self.logger.debug(`

### src/monitoring/integration_manager.py

#### Broad Exceptions

Line 114: `except Exception as e:`
Line 162: `except Exception:`
Line 180: `except Exception:`
Line 200: `except Exception:`
Line 226: `except Exception:`
Line 262: `except Exception:`
Line 278: `except Exception:`
Line 330: `except Exception:`
Line 359: `except Exception:`
Line 380: `except Exception:`
... and 4 more

### src/monitoring/performance_monitor.py

#### Broad Exceptions

Line 170: `except Exception:`
Line 211: `except Exception:`
Line 244: `except Exception:`
Line 254: `except Exception:`
Line 264: `except Exception:`
Line 298: `except Exception:`
Line 314: `except Exception:`
Line 331: `except Exception:`
Line 374: `except Exception:`
Line 394: `except Exception:`
... and 19 more

### src/monitoring/metrics_dashboard.py

#### Broad Exceptions

Line 210: `except Exception:`
Line 230: `except Exception:`
Line 246: `except Exception:`
Line 262: `except Exception:`
Line 280: `except Exception:`
Line 306: `except Exception:`
Line 340: `except Exception as e:`
Line 361: `except Exception:`
Line 388: `except Exception as e:`
Line 405: `except Exception:`
... and 12 more

### src/monitoring/csv_exporter.py

#### Broad Exceptions

Line 143: `except Exception as e:`
Line 190: `except Exception as e:`
Line 235: `except Exception as e:`
Line 287: `except Exception as e:`
Line 340: `except Exception as e:`
Line 386: `except Exception as e:`
Line 435: `except Exception as e:`
Line 467: `except Exception as e:`
Line 487: `except Exception as e:`
Line 505: `except Exception as e:`
... and 5 more

#### Unused Imports

Line 17: `from src.utils.warning_symbols import critical`
Line 17: `from src.utils.warning_symbols import problem`
Line 17: `from src.utils.warning_symbols import failed`
Line 17: `from src.utils.warning_symbols import invalid`
Line 17: `from src.utils.warning_symbols import missing`
Line 17: `from src.utils.warning_symbols import timeout`
Line 17: `from src.utils.warning_symbols import connection_error`
Line 17: `from src.utils.warning_symbols import validation_error`
Line 17: `from src.utils.warning_symbols import initialization_error`
Line 17: `from src.utils.warning_symbols import execution_error`

### src/monitoring/regime_sr_tracker.py

#### Broad Exceptions

Line 388: `except Exception as e:`
Line 413: `except Exception as e:`
Line 494: `except Exception as e:`
Line 520: `except Exception as e:`
Line 545: `except Exception as e:`
Line 565: `except Exception as e:`
Line 671: `except Exception as e:`
Line 704: `except Exception as e:`
Line 729: `except Exception as e:`
Line 832: `except Exception as e:`
... and 21 more

#### Unused Imports

Line 25: `from src.utils.warning_symbols import warning`
Line 25: `from src.utils.warning_symbols import critical`
Line 25: `from src.utils.warning_symbols import problem`
Line 25: `from src.utils.warning_symbols import invalid`
Line 25: `from src.utils.warning_symbols import missing`
Line 25: `from src.utils.warning_symbols import timeout`
Line 25: `from src.utils.warning_symbols import connection_error`
Line 25: `from src.utils.warning_symbols import validation_error`
Line 25: `from src.utils.warning_symbols import initialization_error`
Line 25: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 665: `self.logger.debug(`
Line 726: `self.logger.debug(f"Fetching market data for {symbol} {timeframe}")`
Line 1154: `self.logger.debug(`

### src/monitoring/advanced_tracer.py

#### Broad Exceptions

Line 363: `except Exception as e:`
Line 415: `except Exception as e:`
Line 444: `except Exception as e:`
Line 500: `except Exception:`
Line 528: `except Exception as e:`
Line 546: `except Exception:`
Line 567: `except Exception:`
Line 593: `except Exception:`

#### Todo Comments

Line 6: `of the Ares trading bot with correlation IDs for debugging and performance analysis.`
Line 31: `DEBUG = "debug"`

### src/monitoring/__init__.py

#### Unused Imports

Line 10: `from src.utils.warning_symbols import connection_error`
Line 10: `from src.utils.warning_symbols import critical`
Line 10: `from src.utils.warning_symbols import error`
Line 10: `from src.utils.warning_symbols import execution_error`
Line 10: `from src.utils.warning_symbols import failed`
Line 10: `from src.utils.warning_symbols import initialization_error`
Line 10: `from src.utils.warning_symbols import invalid`
Line 10: `from src.utils.warning_symbols import missing`
Line 10: `from src.utils.warning_symbols import problem`
Line 10: `from src.utils.warning_symbols import timeout`
... and 9 more

### src/monitoring/trade_conditions_monitor.py

#### Broad Exceptions

Line 366: `except Exception as e:`
Line 392: `except Exception:`
Line 491: `except Exception:`
Line 504: `except Exception:`
Line 518: `except Exception:`
Line 562: `except Exception:`
Line 631: `except Exception:`
Line 679: `except Exception:`
Line 716: `except Exception:`
Line 742: `except Exception:`
... and 12 more

#### Todo Comments

Line 284: `and execution for ML model improvement and debugging.`
Line 639: `"""Log detailed decision information for debugging."""`
Line 780: `self.logger.debug(`
Line 1035: `self.logger.debug(f"Fetching {timeframe} data for {symbol} at {timestamp}")`

### src/monitoring/ml_monitor.py

#### Broad Exceptions

Line 198: `except Exception:`
Line 229: `except Exception:`
Line 245: `except Exception:`
Line 263: `except Exception:`
Line 281: `except Exception:`
Line 317: `except Exception:`
Line 333: `except Exception:`
Line 348: `except Exception:`
Line 363: `except Exception:`
Line 378: `except Exception:`
... and 12 more

### src/backtesting/enhanced_backtester.py

#### Broad Exceptions

Line 118: `except Exception as e:`
Line 146: `except Exception as e:`
Line 175: `except Exception as e:`
Line 197: `except Exception as e:`
Line 217: `except Exception as e:`
Line 295: `except Exception:`
Line 317: `except Exception:`
Line 321: `except Exception as e:`
Line 367: `except Exception as e:`
Line 453: `except Exception as e:`
... and 8 more

#### Unused Imports

Line 23: `from src.utils.warning_symbols import warning`
Line 23: `from src.utils.warning_symbols import critical`
Line 23: `from src.utils.warning_symbols import problem`
Line 23: `from src.utils.warning_symbols import missing`
Line 23: `from src.utils.warning_symbols import timeout`
Line 23: `from src.utils.warning_symbols import connection_error`
Line 23: `from src.utils.warning_symbols import validation_error`
Line 23: `from src.utils.warning_symbols import execution_error`

### src/launcher/enhanced_trading_launcher.py

#### Broad Exceptions

Line 105: `except Exception as e:`
Line 132: `except Exception:`
Line 164: `except Exception:`
Line 212: `except Exception:`
Line 259: `except Exception:`
Line 319: `except Exception:`
Line 379: `except Exception:`
Line 395: `except Exception:`
Line 409: `except Exception:`
Line 427: `except Exception:`
... and 4 more

#### Todo Comments

Line 253: `# TODO: Initialize live trading components`
Line 369: `# TODO: Implement live trading execution`
Line 391: `# TODO: Implement live trading metrics`

### src/components/modular_analyst.py

#### Broad Exceptions

Line 91: `except Exception:`
Line 123: `except Exception:`
Line 164: `except Exception:`
Line 194: `except Exception:`
Line 216: `except Exception:`
Line 239: `except Exception:`
Line 261: `except Exception:`
Line 284: `except Exception:`
Line 342: `except Exception:`
Line 381: `except Exception:`
... and 25 more

### src/components/modular_tactician.py

#### Broad Exceptions

Line 178: `except Exception:`
Line 201: `except Exception:`
Line 222: `except Exception:`
Line 243: `except Exception:`
Line 266: `except Exception:`
Line 339: `except Exception:`
Line 390: `except Exception:`
Line 448: `except Exception:`
Line 506: `except Exception:`
Line 523: `except Exception:`
... and 22 more

### src/components/modular_supervisor.py

#### Broad Exceptions

Line 100: `except Exception:`
Line 134: `except Exception:`
Line 175: `except Exception:`
Line 205: `except Exception:`
Line 230: `except Exception:`
Line 255: `except Exception:`
Line 276: `except Exception:`
Line 301: `except Exception:`
Line 368: `except Exception:`
Line 419: `except Exception:`
... and 31 more

### src/components/__init__.py

#### Unused Imports

Line 3: `from src.utils.warning_symbols import connection_error`
Line 3: `from src.utils.warning_symbols import critical`
Line 3: `from src.utils.warning_symbols import error`
Line 3: `from src.utils.warning_symbols import execution_error`
Line 3: `from src.utils.warning_symbols import failed`
Line 3: `from src.utils.warning_symbols import initialization_error`
Line 3: `from src.utils.warning_symbols import invalid`
Line 3: `from src.utils.warning_symbols import missing`
Line 3: `from src.utils.warning_symbols import problem`
Line 3: `from src.utils.warning_symbols import timeout`
... and 6 more

### src/components/modular_strategist.py

#### Broad Exceptions

Line 183: `except Exception:`
Line 204: `except Exception:`
Line 225: `except Exception:`
Line 246: `except Exception:`
Line 269: `except Exception:`
Line 344: `except Exception:`
Line 395: `except Exception:`
Line 453: `except Exception:`
Line 511: `except Exception:`
Line 569: `except Exception:`
... and 22 more

### src/interfaces/event_bus.py

#### Broad Exceptions

Line 91: `except Exception:`
Line 107: `except Exception:`
Line 125: `except Exception:`
Line 140: `except Exception:`
Line 158: `except Exception:`
Line 182: `except Exception:`
Line 208: `except Exception as e:`
Line 228: `except Exception:`
Line 242: `except Exception:`
Line 260: `except Exception:`
... and 3 more

### src/interfaces/__init__.py

#### Unused Imports

Line 3: `from base_interfaces import IAnalyst`
Line 3: `from base_interfaces import IEventBus`
Line 3: `from base_interfaces import IExchangeClient`
Line 3: `from base_interfaces import IModelManager`
Line 3: `from base_interfaces import IPerformanceReporter`
Line 3: `from base_interfaces import IStateManager`
Line 3: `from base_interfaces import IStrategist`
Line 3: `from base_interfaces import ISupervisor`
Line 3: `from base_interfaces import ITactician`
Line 14: `from event_bus import Event`
... and 2 more

### src/interfaces/enhanced_event_bus.py

#### Broad Exceptions

Line 191: `except Exception as e:`
Line 239: `except Exception as e:`
Line 263: `except Exception as e:`
Line 290: `except Exception as e:`
Line 362: `except Exception as e:`
Line 409: `except Exception as e:`
Line 481: `except Exception as e:`
Line 504: `except Exception as e:`
Line 527: `except Exception as e:`
Line 549: `except Exception as e:`
... and 13 more

#### Unused Imports

Line 17: `from src.utils.warning_symbols import critical`
Line 17: `from src.utils.warning_symbols import problem`
Line 17: `from src.utils.warning_symbols import missing`
Line 17: `from src.utils.warning_symbols import timeout`
Line 17: `from src.utils.warning_symbols import connection_error`
Line 17: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 188: `self.logger.debug(f"Saved event {event.metadata.event_id} to {event_file}")`
Line 258: `self.logger.debug(`
Line 668: `self.logger.debug(`
Line 792: `self.logger.debug(`

### src/trading/live_wavelet_analyzer.py

#### Broad Exceptions

Line 124: `except Exception as e:`
Line 165: `except Exception as e:`
Line 174: `except Exception:`
Line 239: `except Exception as e:`
Line 265: `except Exception as e:`
Line 291: `except Exception as e:`
Line 353: `except Exception as e:`
Line 402: `except Exception as e:`
Line 439: `except Exception as e:`

#### Unused Imports

Line 21: `from src.utils.warning_symbols import critical`
Line 21: `from src.utils.warning_symbols import problem`
Line 21: `from src.utils.warning_symbols import failed`
Line 21: `from src.utils.warning_symbols import invalid`
Line 21: `from src.utils.warning_symbols import missing`
Line 21: `from src.utils.warning_symbols import connection_error`
Line 21: `from src.utils.warning_symbols import validation_error`
Line 21: `from src.utils.warning_symbols import execution_error`

### src/trading/live_wavelet_demo.py

#### Broad Exceptions

Line 61: `except Exception as e:`
Line 82: `except Exception as e:`
Line 132: `except Exception as e:`
Line 178: `except Exception as e:`
Line 199: `except Exception as e:`
Line 221: `except Exception as e:`
Line 244: `except Exception as e:`
Line 257: `except Exception as e:`
Line 291: `except Exception as e:`
Line 316: `except Exception as e:`

#### Unused Imports

Line 17: `from src.utils.warning_symbols import critical`
Line 17: `from src.utils.warning_symbols import problem`
Line 17: `from src.utils.warning_symbols import invalid`
Line 17: `from src.utils.warning_symbols import missing`
Line 17: `from src.utils.warning_symbols import timeout`
Line 17: `from src.utils.warning_symbols import connection_error`
Line 17: `from src.utils.warning_symbols import validation_error`
Line 17: `from src.utils.warning_symbols import execution_error`

### src/trading/live_wavelet_integration.py

#### Broad Exceptions

Line 77: `except Exception as e:`
Line 145: `except Exception:`
Line 172: `except Exception:`
Line 188: `except Exception:`
Line 210: `except Exception:`
Line 252: `except Exception:`
Line 283: `except Exception:`
Line 308: `except Exception:`
Line 344: `except Exception:`
Line 380: `except Exception:`

### src/supervisor/performance_monitor.py

#### Broad Exceptions

Line 74: `except Exception:`
Line 90: `except Exception:`
Line 108: `except Exception:`
Line 127: `except Exception:`
Line 147: `except Exception:`
Line 167: `except Exception:`
Line 197: `except Exception:`
Line 211: `except Exception:`
Line 354: `except Exception as e:`
Line 391: `except Exception as e:`

### src/supervisor/global_portfolio_manager.py

#### Broad Exceptions

Line 106: `except Exception as e:`
Line 146: `except Exception:`
Line 188: `except Exception:`
Line 224: `except Exception:`
Line 247: `except Exception:`
Line 270: `except Exception:`
Line 291: `except Exception:`
Line 312: `except Exception:`
Line 335: `except Exception:`
Line 407: `except Exception:`
... and 31 more

### src/supervisor/performance_reporter.py

#### Broad Exceptions

Line 61: `except Exception:`
Line 93: `except Exception:`
Line 127: `except Exception:`
Line 151: `except Exception:`
Line 177: `except Exception:`
Line 207: `except Exception:`
Line 219: `except Exception:`
Line 231: `except Exception:`
Line 241: `except Exception:`
Line 252: `except Exception:`
... and 40 more

#### Todo Comments

Line 642: `self.logger.debug("Real-time report updated")`

### src/supervisor/exchange_ab_tester.py

#### Broad Exceptions

Line 86: `except Exception:`
Line 124: `except Exception:`
Line 197: `except Exception as e:`
Line 240: `except Exception:`
Line 261: `except Exception:`
Line 315: `except Exception:`
Line 344: `except Exception:`
Line 364: `except Exception as e:`
Line 378: `except Exception:`
Line 395: `except Exception:`

### src/supervisor/multi_exchange_ab_tester.py

#### Broad Exceptions

Line 160: `except Exception as e:`
Line 183: `except Exception:`
Line 205: `except Exception:`
Line 229: `except Exception:`
Line 248: `except Exception:`
Line 306: `except Exception:`
Line 331: `except Exception:`
Line 468: `except Exception as e:`
Line 504: `except Exception:`
Line 529: `except Exception:`
... and 17 more

#### Unused Imports

Line 23: `from src.utils.warning_symbols import failed`

### src/supervisor/dynamic_weighter.py

#### Broad Exceptions

Line 95: `except Exception:`
Line 129: `except Exception:`
Line 171: `except Exception:`
Line 205: `except Exception:`
Line 226: `except Exception:`
Line 249: `except Exception:`
Line 270: `except Exception:`
Line 293: `except Exception:`
Line 316: `except Exception:`
Line 387: `except Exception:`
... and 31 more

### src/supervisor/enhanced_model_monitor.py

#### Broad Exceptions

Line 193: `except Exception as e:`
Line 226: `except Exception:`
Line 247: `except Exception:`
Line 265: `except Exception:`
Line 283: `except Exception:`
Line 312: `except Exception:`
Line 329: `except Exception:`
Line 344: `except Exception:`
Line 359: `except Exception:`
Line 374: `except Exception:`
... and 12 more

#### Todo Comments

Line 418: `self.logger.debug("🔍 Drift detection completed")`
Line 551: `self.logger.debug("📊 Performance snapshots captured")`
Line 603: `self.logger.debug("📈 Feature drift analysis completed")`
Line 673: `self.logger.debug("🎯 Ensemble performance monitored")`

### src/supervisor/exchange_volume_adapter.py

#### Broad Exceptions

Line 122: `except Exception as e:`
Line 145: `except Exception:`
Line 167: `except Exception:`
Line 191: `except Exception:`
Line 265: `except Exception:`
Line 276: `except Exception:`
Line 291: `except Exception:`
Line 338: `except Exception:`
Line 384: `except Exception as e:`
Line 399: `except Exception as e:`
... and 3 more

#### Unused Imports

Line 17: `from src.utils.warning_symbols import failed`

### src/supervisor/monitoring.py

#### Broad Exceptions

Line 54: `except Exception:`
Line 70: `except Exception:`
Line 88: `except Exception:`
Line 107: `except Exception:`
Line 127: `except Exception:`
Line 146: `except Exception:`
Line 160: `except Exception:`
Line 174: `except Exception:`
Line 213: `except Exception as e:`

### src/supervisor/main.py

#### Broad Exceptions

Line 316: `except Exception as e:`
Line 356: `except Exception:`
Line 372: `except Exception:`
Line 390: `except Exception:`
Line 409: `except Exception:`
Line 427: `except Exception:`
Line 441: `except Exception:`
Line 474: `except Exception as e:`

#### Todo Comments

Line 241: `self.logger.debug(f"Updated account equity: ${current_equity:,.2f}")`
Line 296: `self.logger.debug(`

### src/supervisor/ab_tester.py

#### Broad Exceptions

Line 97: `except Exception:`
Line 140: `except Exception:`
Line 200: `except Exception:`
Line 221: `except Exception:`
Line 241: `except Exception:`
Line 281: `except Exception:`
Line 327: `except Exception:`
Line 379: `except Exception:`

### src/supervisor/risk_allocator.py

#### Broad Exceptions

Line 67: `except Exception:`
Line 83: `except Exception:`
Line 101: `except Exception:`
Line 120: `except Exception:`
Line 140: `except Exception:`
Line 159: `except Exception:`
Line 178: `except Exception:`
Line 192: `except Exception:`
Line 232: `except Exception:`
Line 268: `except Exception:`
... and 5 more

### src/supervisor/pnl_loss_functions.py

#### Broad Exceptions

Line 157: `except Exception:`
Line 187: `except Exception:`
Line 229: `except Exception:`
Line 263: `except Exception:`
Line 284: `except Exception:`
Line 305: `except Exception:`
Line 326: `except Exception:`
Line 347: `except Exception:`
Line 370: `except Exception:`
Line 439: `except Exception as e:`
... and 31 more

#### Unused Imports

Line 4: `from keras import backend`

### src/supervisor/__init__.py

#### Unused Imports

Line 4: `from src.utils.warning_symbols import connection_error`
Line 4: `from src.utils.warning_symbols import critical`
Line 4: `from src.utils.warning_symbols import error`
Line 4: `from src.utils.warning_symbols import execution_error`
Line 4: `from src.utils.warning_symbols import failed`
Line 4: `from src.utils.warning_symbols import initialization_error`
Line 4: `from src.utils.warning_symbols import invalid`
Line 4: `from src.utils.warning_symbols import missing`
Line 4: `from src.utils.warning_symbols import problem`
Line 4: `from src.utils.warning_symbols import timeout`
... and 7 more

### src/supervisor/supervisor.py

#### Broad Exceptions

Line 49: `except Exception:`
Line 86: `except Exception:`
Line 122: `except Exception:`
Line 241: `except Exception:`
Line 261: `except Exception:`
Line 285: `except Exception:`
Line 312: `except Exception:`
Line 337: `except Exception:`
Line 353: `except Exception:`
Line 370: `except Exception:`
... and 21 more

### src/supervisor/model_behavior_tracker.py

#### Broad Exceptions

Line 147: `except Exception as e:`
Line 173: `except Exception:`
Line 196: `except Exception:`
Line 214: `except Exception:`
Line 232: `except Exception:`
Line 256: `except Exception:`
Line 273: `except Exception:`
Line 351: `except Exception:`
Line 373: `except Exception:`
Line 394: `except Exception:`
... and 13 more

#### Todo Comments

Line 349: `self.logger.debug("📊 Behavior snapshots captured")`

### src/supervisor/optimizer.py

#### Broad Exceptions

Line 58: `except Exception:`
Line 74: `except Exception:`
Line 92: `except Exception:`
Line 111: `except Exception:`
Line 131: `except Exception:`
Line 150: `except Exception:`
Line 165: `except Exception:`
Line 179: `except Exception:`
Line 271: `except Exception as e:`
Line 292: `except Exception:`
... and 1 more

### src/pipelines/live_trading_pipeline.py

#### Broad Exceptions

Line 107: `except Exception:`
Line 201: `except Exception:`
Line 251: `except Exception:`
Line 272: `except Exception:`
Line 313: `except Exception as e:`
Line 318: `except Exception:`
Line 339: `except Exception:`
Line 395: `except Exception:`
Line 434: `except Exception:`
Line 478: `except Exception:`
... and 24 more

#### Unused Imports

Line 11: `from src.tactician.enhanced_order_manager import OrderRequest`
Line 11: `from src.tactician.enhanced_order_manager import OrderSide`
Line 11: `from src.tactician.enhanced_order_manager import OrderType`
Line 206: `from exchange.factory import ExchangeFactory`
Line 302: `from exchange.factory import ExchangeFactory`

### src/pipelines/base_pipeline.py

#### Broad Exceptions

Line 286: `except Exception:`
Line 307: `except Exception:`
Line 328: `except Exception:`
Line 351: `except Exception:`
Line 374: `except Exception:`
Line 432: `except Exception:`
Line 471: `except Exception:`
Line 517: `except Exception:`
Line 567: `except Exception:`
Line 615: `except Exception:`
... and 22 more

### src/pipelines/__init__.py

#### Unused Imports

Line 8: `from src.utils.warning_symbols import connection_error`
Line 8: `from src.utils.warning_symbols import critical`
Line 8: `from src.utils.warning_symbols import error`
Line 8: `from src.utils.warning_symbols import execution_error`
Line 8: `from src.utils.warning_symbols import failed`
Line 8: `from src.utils.warning_symbols import initialization_error`
Line 8: `from src.utils.warning_symbols import invalid`
Line 8: `from src.utils.warning_symbols import missing`
Line 8: `from src.utils.warning_symbols import problem`
Line 8: `from src.utils.warning_symbols import timeout`
... and 7 more

### src/pipelines/components/monitoring_manager.py

#### Broad Exceptions

Line 102: `except Exception:`
Line 136: `except Exception:`
Line 177: `except Exception:`
Line 207: `except Exception:`
Line 230: `except Exception:`
Line 253: `except Exception:`
Line 276: `except Exception:`
Line 297: `except Exception:`
Line 359: `except Exception:`
Line 400: `except Exception:`
... and 25 more

### src/pipelines/components/__init__.py

#### Unused Imports

Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import execution_error`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import initialization_error`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import timeout`
... and 9 more

### src/pipelines/components/lifecycle_manager.py

#### Broad Exceptions

Line 108: `except Exception:`
Line 140: `except Exception:`
Line 181: `except Exception:`
Line 211: `except Exception:`
Line 234: `except Exception:`
Line 257: `except Exception:`
Line 280: `except Exception:`
Line 303: `except Exception:`
Line 367: `except Exception:`
Line 408: `except Exception:`
... and 25 more

### src/pipelines/components/data_manager.py

#### Broad Exceptions

Line 92: `except Exception:`
Line 120: `except Exception:`
Line 161: `except Exception:`
Line 191: `except Exception:`
Line 212: `except Exception:`
Line 233: `except Exception:`
Line 254: `except Exception:`
Line 275: `except Exception:`
Line 331: `except Exception:`
Line 370: `except Exception:`
... and 25 more

### src/sentinel/sentinel.py

#### Broad Exceptions

Line 87: `except Exception:`
Line 114: `except Exception:`
Line 148: `except Exception:`
Line 189: `except Exception:`
Line 222: `except Exception:`
Line 241: `except Exception:`
Line 264: `except Exception:`
Line 304: `except Exception:`
Line 340: `except Exception:`
Line 372: `except Exception:`
... and 9 more

#### Todo Comments

Line 455: `self.logger.debug(f"Alert callback {i+1} executed successfully")`

### src/sentinel/health_integration.py

#### Debug Statements

Line 304: `print("\n🔍 Component Health Details:")`

#### Broad Exceptions

Line 65: `except Exception:`
Line 101: `except Exception:`
Line 156: `except Exception as e:`
Line 217: `except Exception as e:`
Line 275: `except Exception as e:`

#### Todo Comments

Line 74: `logger.debug("🔍 Running periodic health checks...")`
Line 269: `"health_score_history": [],  # TODO: Implement trending`

### src/sentinel/health_checker.py

#### Broad Exceptions

Line 76: `except Exception as e:`
Line 256: `except Exception as e:`
Line 290: `except Exception as e:`
Line 339: `except Exception as e:`
Line 391: `except Exception as e:`
Line 455: `except Exception as e:`
Line 515: `except Exception as e:`

### src/transition/seq2seq_trainer.py

#### Type Ignores

Line 17: `pl = None  # type: ignore`

#### Broad Exceptions

Line 16: `except Exception:  # pragma: no cover`
Line 24: `except Exception:`
Line 42: `except Exception:`
Line 448: `except Exception:`
Line 466: `except Exception:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 5: `from dataclasses import dataclass`

### src/transition/combined_features_builder.py

#### Broad Exceptions

Line 132: `except Exception as e:`
Line 138: `except Exception as e:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 6: `from typing import Dict`

### src/transition/baseline_rf.py

#### Type Ignores

Line 15: `import shap  # type: ignore`
Line 17: `shap = None  # type: ignore`

#### Broad Exceptions

Line 16: `except Exception:  # pragma: no cover`
Line 109: `except Exception:`
Line 136: `except Exception as e:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 11: `from sklearn.model_selection import train_test_split`

### src/transition/inference_combiner.py

#### Broad Exceptions

Line 62: `except Exception as e:`

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/transition/rolling_inference.py

#### Broad Exceptions

Line 67: `except Exception as e:`
Line 115: `except Exception:`
Line 125: `except Exception:`
Line 150: `except Exception:`
Line 162: `except Exception:`
Line 174: `except Exception:`
Line 181: `except Exception:`
Line 201: `except Exception:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 8: `import os`

### src/transition/event_trigger_indexer.py

#### Broad Exceptions

Line 169: `except Exception as e:`
Line 175: `except Exception as e:`
Line 181: `except Exception as e:`
Line 187: `except Exception as e:`
Line 193: `except Exception as e:`
Line 203: `except Exception as e:`
Line 214: `except Exception as e:`

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/transition/multitask_rf.py

#### Broad Exceptions

Line 89: `except Exception:`
Line 147: `except Exception:`
Line 187: `except Exception:`
Line 247: `except Exception:`
Line 249: `except Exception:`
Line 290: `except Exception:`
Line 333: `except Exception as e:`
Line 343: `except Exception as e:`
Line 349: `except Exception as e:`
Line 354: `except Exception as e:`
... and 5 more

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 15: `from sklearn.model_selection import train_test_split`

### src/transition/rolling_window_dataset.py

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/transition/path_targets.py

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/transition/state_sequence_builder.py

#### Broad Exceptions

Line 53: `except Exception:`
Line 68: `except Exception as e:`
Line 90: `except Exception:`
Line 145: `except Exception:`
Line 148: `except Exception as e:`

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/transition/event_window_dataset.py

#### Broad Exceptions

Line 135: `except Exception:`
Line 250: `except Exception:`
Line 323: `except Exception:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 6: `from typing import Iterable`

### src/custom_types/config_types.py

#### Todo Comments

Line 76: `log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]`
Line 85: `debug_mode: bool`

### src/custom_types/__init__.py

#### Unused Imports

Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import execution_error`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import initialization_error`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import timeout`
... and 2 more

### src/reports/paper_trading_reporter.py

#### Broad Exceptions

Line 323: `except Exception as e:`
Line 382: `except Exception as e:`
Line 410: `except Exception as e:`
Line 438: `except Exception as e:`
Line 507: `except Exception as e:`
Line 560: `except Exception as e:`
Line 609: `except Exception as e:`
Line 662: `except Exception as e:`
Line 720: `except Exception as e:`
Line 745: `except Exception as e:`
... and 4 more

#### Unused Imports

Line 22: `from src.utils.warning_symbols import warning`
Line 22: `from src.utils.warning_symbols import critical`
Line 22: `from src.utils.warning_symbols import problem`
Line 22: `from src.utils.warning_symbols import failed`
Line 22: `from src.utils.warning_symbols import invalid`
Line 22: `from src.utils.warning_symbols import timeout`
Line 22: `from src.utils.warning_symbols import connection_error`
Line 22: `from src.utils.warning_symbols import validation_error`
Line 22: `from src.utils.warning_symbols import initialization_error`
Line 22: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 192: `self.logger.debug(f"CWD: {cwd}")`
Line 193: `self.logger.debug(f"Report directory (abs): {self.report_directory}")`
Line 634: `self.logger.debug(`
Line 635: `f"[DEBUG] Generating {report_type} report with formats: {export_formats}",`
Line 638: `f"[DEBUG] Generating {report_type} report with formats: {export_formats}",`
Line 655: `self.logger.debug(f"[DEBUG] Exporting report as {format_type}")`
Line 656: `self.logger.info(f"[DEBUG] Exporting report as {format_type}")`
Line 659: `self.logger.debug(f"[DEBUG] Generated {report_type} report")`
Line 664: `self.logger.debug(f"[DEBUG] Error generating report: {e}")`
Line 767: `self.logger.debug(f"Exported JSON report: {filepath}")`
... and 2 more

### src/integration/paper_trading_integration.py

#### Broad Exceptions

Line 107: `except Exception as e:`
Line 135: `except Exception:`
Line 235: `except Exception:`
Line 245: `except Exception:`
Line 260: `except Exception:`
Line 295: `except Exception:`
Line 306: `except Exception:`
Line 328: `except Exception:`
Line 350: `except Exception:`
Line 402: `except Exception:`
... and 2 more

### src/examples/di_usage_example.py

#### Broad Exceptions

Line 48: `except Exception:`
Line 81: `except Exception:`
Line 116: `except Exception:`
Line 154: `except Exception:`
Line 193: `except Exception:`
Line 209: `except Exception:`
Line 230: `except Exception as e:`

### src/examples/enhanced_event_bus_example.py

#### Broad Exceptions

Line 397: `except Exception:`

### src/examples/type_safety_example.py

#### Broad Exceptions

Line 215: `except Exception:`
Line 281: `except Exception:`

### src/tactician/position_sizer.py

#### Type Ignores

Line 39: `self.print = _shim_print  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 36: `except Exception:`
Line 125: `except Exception:`
Line 232: `except Exception:`
Line 284: `except Exception:`
Line 341: `except Exception:`
Line 363: `except Exception:`
Line 435: `except Exception:`
Line 479: `except Exception:`
Line 503: `except Exception as e:`
Line 541: `except Exception:`
... and 2 more

### src/tactician/sr_weight_optimizer.py

#### Broad Exceptions

Line 90: `except Exception as e:`
Line 148: `except Exception as e:`
Line 185: `except Exception as e:`
Line 217: `except Exception as e:`
Line 251: `except Exception as e:`
Line 290: `except Exception as e:`
Line 310: `except Exception as e:`
Line 340: `except Exception as e:`
Line 398: `except Exception as e:`
Line 437: `except Exception as e:`
... and 4 more

#### Unused Imports

Line 3: `import asyncio`
Line 6: `from typing import Tuple`
Line 7: `from datetime import timedelta`
Line 11: `import os`

### src/tactician/position_closing.py

#### Broad Exceptions

Line 106: `except Exception:`
Line 148: `except Exception:`
Line 232: `except Exception:`
Line 313: `except Exception as e:`
Line 340: `except Exception:`
Line 371: `except Exception:`
Line 404: `except Exception:`
Line 443: `except Exception:`
Line 483: `except Exception:`
Line 493: `except Exception:`
... and 4 more

### src/tactician/ml_target_updater.py

#### Broad Exceptions

Line 205: `except Exception as e:`
Line 258: `except Exception as e:`
Line 320: `except Exception as e:`
Line 394: `except Exception as e:`
Line 777: `except Exception as e:`
Line 789: `except Exception as e:`
Line 812: `except Exception as e:`

#### Unused Imports

Line 15: `from src.utils.warning_symbols import critical`
Line 15: `from src.utils.warning_symbols import problem`
Line 15: `from src.utils.warning_symbols import invalid`
Line 15: `from src.utils.warning_symbols import missing`
Line 15: `from src.utils.warning_symbols import timeout`
Line 15: `from src.utils.warning_symbols import connection_error`
Line 15: `from src.utils.warning_symbols import validation_error`
Line 15: `from src.utils.warning_symbols import initialization_error`
Line 15: `from src.utils.warning_symbols import execution_error`

### src/tactician/position_division_strategy.py

#### Broad Exceptions

Line 213: `except Exception as e:`
Line 236: `except Exception as e:`
Line 289: `except Exception as e:`
Line 347: `except Exception:`
Line 523: `except Exception as e:`
Line 618: `except Exception:`
Line 682: `except Exception:`
Line 720: `except Exception:`
Line 841: `except Exception:`
Line 968: `except Exception:`
... and 7 more

#### Todo Comments

Line 214: `self.logger.debug(f"Could not load from {path}: {e}")`
Line 303: `self.logger.debug("🔍 Validating position division configuration...")`
Line 319: `self.logger.debug(`
Line 327: `self.logger.debug(`
Line 628: `self.logger.debug(`
Line 660: `self.logger.debug(`
Line 676: `self.logger.debug(`
Line 693: `self.logger.debug(`
Line 700: `self.logger.debug(`
Line 708: `self.logger.debug(`
... and 44 more

### src/tactician/tactician.py

#### Broad Exceptions

Line 89: `except Exception:`
Line 121: `except Exception:`
Line 159: `except Exception:`
Line 203: `except Exception:`
Line 237: `except Exception:`
Line 274: `except Exception:`
Line 305: `except Exception:`
Line 384: `except Exception:`
Line 408: `except Exception as e:`

### src/tactician/tactics_orchestrator.py

#### Broad Exceptions

Line 176: `except Exception as e:`
Line 295: `except Exception as e:`
Line 360: `except Exception as e:`
Line 366: `except Exception as e:`
Line 373: `except Exception:`
Line 415: `except Exception:`
Line 462: `except Exception:`
Line 497: `except Exception:`
Line 642: `except Exception:`
Line 685: `except Exception:`
... and 25 more

#### Unused Imports

Line 4: `from typing import Optional`
Line 5: `import os`

#### Todo Comments

Line 786: `self.logger.debug("S/R opportunities disabled")`
Line 801: `self.logger.debug(`
Line 1245: `self.logger.debug(`
Line 1269: `self.logger.debug(`
Line 1308: `self.logger.debug(`
Line 1332: `self.logger.debug(`
Line 1365: `# TODO: Integrate with actual order execution system`

### src/tactician/position_monitor.py

#### Type Ignores

Line 600: `pos = self.active_positions.get(assessment.position_id, {})  # type: ignore[attr-defined]`
Line 633: `pos = self.active_positions.get(assessment.position_id, {})  # type: ignore[attr-defined]`
Line 664: `pos = self.active_positions.get(assessment.position_id, {})  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 136: `except Exception as e:`
Line 159: `except Exception as e:`
Line 222: `except Exception as e:`
Line 230: `except Exception as e:`
Line 304: `except Exception as e:`
Line 402: `except Exception as e:`
Line 431: `except Exception as e:`
Line 456: `except Exception as e:`
Line 487: `except Exception as e:`
Line 558: `except Exception as e:`
... and 9 more

#### Unused Imports

Line 21: `from src.utils.warning_symbols import error`
Line 21: `from src.utils.warning_symbols import warning`
Line 21: `from src.utils.warning_symbols import critical`
Line 21: `from src.utils.warning_symbols import problem`
Line 21: `from src.utils.warning_symbols import failed`
Line 21: `from src.utils.warning_symbols import invalid`
Line 21: `from src.utils.warning_symbols import missing`
Line 21: `from src.utils.warning_symbols import timeout`
Line 21: `from src.utils.warning_symbols import connection_error`
Line 21: `from src.utils.warning_symbols import validation_error`
... and 2 more

#### Todo Comments

Line 185: `self.logger.debug(`
Line 711: `self.logger.debug(`

### src/tactician/enhanced_order_manager.py

#### Type Ignores

Line 1154: `open_orders = await self.exchange_client.get_open_orders()  # type: ignore[attr-defined]`
Line 1245: `)  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 263: `except Exception:`
Line 266: `except Exception:`
Line 288: `except Exception:`
Line 350: `except Exception:`
Line 374: `except Exception as e:`
Line 395: `except Exception as e:`
Line 410: `except Exception:`
Line 540: `except Exception:`
Line 586: `except Exception:`
Line 611: `except Exception:`
... and 20 more

#### Todo Comments

Line 562: `self.logger.debug(`
Line 573: `self.logger.debug("Volume confirmation not met")`
Line 581: `self.logger.debug("Momentum confirmation not met")`
Line 751: `self.logger.debug(`
Line 758: `self.logger.debug(`
Line 770: `self.logger.debug("Order size exceeds liquidity limits")`

### src/tactician/leverage_sizer.py

#### Type Ignores

Line 38: `self.print = _shim_print  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 35: `except Exception:`
Line 126: `except Exception:`
Line 242: `except Exception:`
Line 305: `except Exception:`
Line 338: `except Exception:`
Line 374: `except Exception:`
Line 393: `except Exception:`
Line 437: `except Exception:`
Line 461: `except Exception:`
Line 492: `except Exception:`

### src/tactician/sr_breakout_predictor.py

#### Broad Exceptions

Line 219: `except Exception as e:`
Line 266: `except Exception as e:`
Line 352: `except Exception as e:`
Line 524: `except Exception as e:`
Line 638: `except Exception as e:`
Line 711: `except Exception as e:`
Line 748: `except Exception as e:`
Line 894: `except Exception as e:`
Line 1015: `except Exception as e:`
Line 1046: `except Exception as e:`
... and 36 more

#### Unused Imports

Line 3: `from datetime import datetime`
Line 4: `from typing import Dict`
Line 4: `from typing import List`
Line 4: `from typing import Tuple`
Line 16: `from src.utils.warning_symbols import error`
Line 16: `from src.utils.warning_symbols import invalid`
Line 16: `from src.utils.warning_symbols import missing`
Line 16: `from src.utils.warning_symbols import warning`

#### Todo Comments

Line 1175: `# Log confidence breakdown for debugging`
Line 1176: `self.logger.debug(f"SR Confidence Components: {confidence_components}")`
Line 1177: `self.logger.debug(f"Final SR Confidence Score: {final_confidence:.3f}")`
Line 1832: `self.logger.debug(`
Line 1921: `self.logger.debug(f"Error loading trained model: {e}")`

### src/tactician/__init__.py

#### Unused Imports

Line 4: `from src.utils.warning_symbols import connection_error`
Line 4: `from src.utils.warning_symbols import critical`
Line 4: `from src.utils.warning_symbols import error`
Line 4: `from src.utils.warning_symbols import execution_error`
Line 4: `from src.utils.warning_symbols import failed`
Line 4: `from src.utils.warning_symbols import initialization_error`
Line 4: `from src.utils.warning_symbols import invalid`
Line 4: `from src.utils.warning_symbols import missing`
Line 4: `from src.utils.warning_symbols import problem`
Line 4: `from src.utils.warning_symbols import timeout`
... and 6 more

### src/tactician/ml_target_validator.py

#### Broad Exceptions

Line 62: `except Exception:`
Line 78: `except Exception:`
Line 96: `except Exception:`
Line 115: `except Exception:`
Line 133: `except Exception:`
Line 155: `except Exception:`
Line 174: `except Exception:`
Line 193: `except Exception:`
Line 212: `except Exception:`
Line 226: `except Exception:`
... and 2 more

### src/tactician/async_order_executor.py

#### Broad Exceptions

Line 188: `except Exception:`
Line 208: `except Exception:`
Line 227: `except Exception:`
Line 255: `except Exception:`
Line 275: `except Exception:`
Line 303: `except Exception:`
Line 309: `except Exception:`
Line 360: `except Exception:`
Line 365: `except Exception as e:`
Line 376: `except Exception:`
... and 17 more

### src/tactician/ml_tactics_manager.py

#### Broad Exceptions

Line 79: `except Exception:`
Line 106: `except Exception:`
Line 122: `except Exception:`
Line 207: `except Exception:`
Line 243: `except Exception:`
Line 300: `except Exception:`
Line 359: `except Exception as e:`
Line 409: `except Exception:`
Line 460: `except Exception:`
Line 511: `except Exception:`
... and 6 more

### src/tracking/trade_tracker.py

#### Broad Exceptions

Line 300: `except Exception:`
Line 394: `except Exception:`

### src/training/model_trainer.py

#### Broad Exceptions

Line 166: `except Exception:`
Line 200: `except Exception:`
Line 240: `except Exception:`
Line 264: `except Exception:`
Line 380: `except Exception as e:  # ImportError or dependency issues`
Line 439: `except Exception:`
Line 444: `except Exception:`
Line 481: `except Exception:`
Line 526: `except Exception:`
Line 572: `except Exception:`
... and 8 more

#### Unused Imports

Line 18: `from sklearn.model_selection import train_test_split`

#### Todo Comments

Line 57: `debug_training_step,`
Line 304: `@debug_training_step(`
Line 306: `save_debug_artifacts=True,`

### src/training/enhanced_training_manager.py

#### Type Ignores

Line 25: `import pyarrow.parquet as pq  # type: ignore`
Line 27: `pq = None  # type: ignore`
Line 2119: `self._summarize_calibration(calibration_results)  # type: ignore[name-defined]`

#### Broad Exceptions

Line 78: `except Exception:`
Line 97: `except Exception:`
Line 312: `except Exception as e:`
Line 322: `except Exception as e:`
Line 565: `except Exception:`
Line 605: `except Exception as e:`
Line 638: `except Exception as e:`
Line 662: `except Exception as e:`
Line 696: `except Exception as e:`
Line 757: `except Exception as e:`
... and 48 more

#### Todo Comments

Line 508: `)  # "info" or "debug"`
Line 751: `# Log failed prerequisites for debugging`
Line 969: `if self.verbosity == "debug":`
Line 970: `self.logger.debug(`
Line 973: `self.logger.debug(`
Line 1466: `# Non-fatal if validator is missing, but log for debugging.`
Line 3870: `self.logger.debug(f"   🗑️ Cleared: {file_path}")`
Line 3879: `self.logger.debug(f"   ℹ️ No artifacts found for {step_name}")`

### src/training/data_access_utils.py

#### Broad Exceptions

Line 73: `except Exception as e:`
Line 121: `except Exception as e:`
Line 147: `except Exception as e:`
Line 174: `except Exception as e:`
Line 211: `except Exception as e:`
Line 250: `except Exception as e:`
Line 277: `except Exception as e:`
Line 321: `except Exception as e:`

### src/training/regularization.py

#### Broad Exceptions

Line 81: `except Exception as e:`
Line 116: `except Exception as e:`
Line 213: `except Exception as e:`

### src/training/feature_integration.py

#### Broad Exceptions

Line 91: `except Exception as e:`
Line 159: `except Exception as e:`
Line 196: `except Exception as e:`
Line 250: `except Exception as e:`
Line 302: `except Exception as e:`
Line 308: `except Exception as e:`
Line 335: `except Exception as e:`
Line 383: `except Exception as e:`

#### Unused Imports

Line 18: `from src.utils.warning_symbols import warning`
Line 18: `from src.utils.warning_symbols import critical`
Line 18: `from src.utils.warning_symbols import problem`
Line 18: `from src.utils.warning_symbols import failed`
Line 18: `from src.utils.warning_symbols import invalid`
Line 18: `from src.utils.warning_symbols import missing`
Line 18: `from src.utils.warning_symbols import timeout`
Line 18: `from src.utils.warning_symbols import connection_error`
Line 18: `from src.utils.warning_symbols import validation_error`
Line 18: `from src.utils.warning_symbols import execution_error`

### src/training/wavelet_integration_demo.py

#### Broad Exceptions

Line 89: `except Exception as e:`
Line 144: `except Exception as e:`
Line 192: `except Exception as e:`
Line 247: `except Exception as e:`
Line 303: `except Exception as e:`
Line 360: `except Exception as e:`
Line 433: `except Exception as e:`
Line 473: `except Exception as e:`
Line 527: `except Exception as e:`

#### Unused Imports

Line 31: `from src.utils.warning_symbols import warning`
Line 31: `from src.utils.warning_symbols import critical`
Line 31: `from src.utils.warning_symbols import invalid`
Line 31: `from src.utils.warning_symbols import missing`
Line 31: `from src.utils.warning_symbols import timeout`
Line 31: `from src.utils.warning_symbols import connection_error`
Line 31: `from src.utils.warning_symbols import validation_error`
Line 31: `from src.utils.warning_symbols import initialization_error`
Line 31: `from src.utils.warning_symbols import execution_error`

### src/training/step_orchestrator.py

#### Broad Exceptions

Line 91: `except Exception as e:`
Line 221: `except Exception as e:`

### src/training/enhanced_training_manager_optimized.py

#### Type Ignores

Line 21: `import pyarrow as pa  # type: ignore`
Line 22: `import pyarrow.parquet as pq  # type: ignore`
Line 23: `import pyarrow.dataset as ds  # type: ignore`
Line 25: `pa = None  # type: ignore`
Line 26: `pq = None  # type: ignore`
Line 27: `ds = None  # type: ignore`
Line 418: `import pyarrow as pa  # type: ignore`
Line 419: `import pyarrow.parquet as pq_mod  # type: ignore`

#### Broad Exceptions

Line 146: `except Exception as e:`
Line 155: `except Exception as e:`
Line 204: `except Exception as e:`
Line 223: `except Exception as e:`
Line 278: `except Exception:`
Line 359: `except Exception as e:`
Line 378: `except Exception as e:`
Line 431: `except Exception as e:`
Line 459: `except Exception:`
Line 469: `except Exception:`
... and 18 more

#### Unused Imports

Line 3: `import asyncio`
Line 10: `import warnings`
Line 11: `from concurrent.futures import ThreadPoolExecutor`
Line 12: `from datetime import datetime`
Line 14: `from typing import Union`
Line 28: `from sklearn.base import BaseEstimator`
Line 30: `from src.utils.error_handler import handle_errors`
Line 32: `from src.utils.validator_orchestrator import validator_orchestrator`

#### Todo Comments

Line 565: `# Reduce noise: use debug and include basic shape`
Line 567: `self.logger.debug(f"Optimized DataFrame memory usage: shape={df.shape}")`
Line 1069: `# Reduce log volume, but keep trial-level info at debug, step-level at info`
Line 1070: `self.logger.debug(`

### src/training/data_sharing_manager.py

#### Broad Exceptions

Line 95: `except Exception:`

#### Unused Imports

Line 3: `import asyncio`
Line 5: `import os`
Line 7: `from datetime import datetime`
Line 7: `from datetime import timedelta`
Line 9: `from pathlib import Path`

#### Todo Comments

Line 25: `debug_training_step,`
Line 137: `"""String representation for debugging."""`
Line 206: `@debug_training_step(`
Line 208: `save_debug_artifacts=True,`

### src/training/memory_profiler.py

#### Broad Exceptions

Line 313: `except Exception as e:`

#### Todo Comments

Line 177: `"flags": gc.get_debug(),`

### src/training/integration_guide.py

#### Broad Exceptions

Line 101: `except Exception:`
Line 210: `except Exception:`

### src/training/unified_data_orchestrator.py

#### Debug Statements

Line 127: `print(f"🔍 [INIT] Quality validation config:")`
Line 1242: `print(f"🔍 [VALIDATE] Starting data validation: {request_id}")`
Line 1258: `print(f"🔍 [VALIDATE] Step 1: Checking data size...")`
Line 1270: `print(f"🔍 [VALIDATE] Step 2: Checking for missing values...")`
Line 1294: `print(f"🔍 [VALIDATE] Step 3: Checking for duplicates...")`
Line 1318: `print(f"🔍 [VALIDATE] Step 4: Checking timestamp issues...")`
Line 1326: `print(f"🔍 [VALIDATE] Step 5: Checking price anomalies...")`
Line 1641: `print(f"🔍 [FIND_FILES] Searching for raw data files...")`

#### Broad Exceptions

Line 157: `except Exception:`
Line 248: `except Exception as e:`
Line 327: `except Exception as e:`
Line 592: `except Exception as e:`
Line 876: `except Exception as e:`
Line 1063: `except Exception as e:`
Line 1126: `except Exception as e:`
Line 1169: `except Exception as e:`
Line 1343: `except Exception as e:`
Line 1370: `except Exception as e:`
... and 9 more

#### Unused Imports

Line 21: `import os`
Line 25: `from datetime import timedelta`
Line 27: `from typing import Tuple`
Line 27: `from typing import Union`
Line 27: `from typing import Iterator`
Line 28: `from functools import lru_cache`
Line 29: `import warnings`
Line 35: `from src.utils.error_handler import handle_specific_errors`
Line 36: `from src.utils.warning_symbols import error`
Line 36: `from src.utils.warning_symbols import warning`
... and 2 more

#### Todo Comments

Line 45: `debug_training_step,`
Line 189: `@debug_training_step(`
Line 191: `save_debug_artifacts=True,`
Line 271: `@debug_training_step(`
Line 273: `save_debug_artifacts=True,`
Line 368: `@debug_training_step(`
Line 370: `save_debug_artifacts=True,`
Line 635: `@debug_training_step(`
Line 637: `save_debug_artifacts=True,`
Line 910: `@debug_training_step(`
... and 6 more

### src/training/dual_model_system.py

#### Type Ignores

Line 68: `self.print = _shim_print  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 65: `except Exception:`
Line 180: `except Exception:`
Line 238: `except Exception as e:`
Line 291: `except Exception as e:`
Line 398: `except Exception:`
Line 426: `except Exception as e:`
Line 454: `except Exception as e:`
Line 565: `except Exception as e:`
Line 668: `except Exception as e:`
Line 731: `except Exception as e:`
... and 24 more

#### Unused Imports

Line 16: `from src.utils.warning_symbols import failed`
Line 16: `from src.utils.warning_symbols import invalid`

#### Todo Comments

Line 32: `debug_training_step,`
Line 494: `@debug_training_step(`
Line 496: `save_debug_artifacts=True,`

### src/training/performance_comparison.py

#### Broad Exceptions

Line 82: `except Exception as e:`
Line 161: `except Exception as e:`
Line 216: `except Exception as e:`
Line 275: `except Exception as e:`
Line 337: `except Exception as e:`
Line 412: `except Exception as e:`
Line 584: `except Exception as e:`
Line 747: `except Exception as e:`

### src/training/launcher_integration_patch.py

#### Broad Exceptions

Line 62: `except Exception as e:`
Line 296: `except Exception as e:`
Line 322: `except Exception:`
Line 446: `except Exception as e:`

### src/training/optimized_feature_selection_manager.py

#### Broad Exceptions

Line 189: `except Exception as e:`
Line 270: `except Exception as e:`
Line 300: `except Exception:`
Line 320: `except:`
Line 419: `except Exception as e:`
Line 791: `except Exception as e:`

#### Unused Imports

Line 3: `import asyncio`
Line 7: `import os`
Line 9: `from typing import Optional`
Line 9: `from typing import Set`
Line 11: `from pathlib import Path`
Line 15: `from sklearn.feature_selection import mutual_info_regression`
Line 15: `from sklearn.feature_selection import f_classif`
Line 15: `from sklearn.feature_selection import f_regression`
Line 15: `from sklearn.feature_selection import SelectKBest`
Line 15: `from sklearn.feature_selection import SelectFromModel`
... and 7 more

### src/training/feature_selection_manager.py

#### Broad Exceptions

Line 124: `except Exception as e:`
Line 433: `except Exception as e:`

#### Unused Imports

Line 3: `import asyncio`
Line 7: `import os`
Line 8: `from typing import Optional`
Line 10: `from pathlib import Path`
Line 12: `from sklearn.feature_selection import mutual_info_regression`
Line 12: `from sklearn.feature_selection import f_classif`
Line 12: `from sklearn.feature_selection import f_regression`
Line 12: `from sklearn.feature_selection import SelectKBest`
Line 12: `from sklearn.feature_selection import SelectFromModel`
Line 12: `from sklearn.feature_selection import VarianceThreshold`
... and 6 more

### src/training/progress_manager.py

#### Broad Exceptions

Line 80: `except Exception as e:`
Line 116: `except Exception as e:`
Line 141: `except Exception as e:`
Line 166: `except Exception as e:`
Line 201: `except Exception as e:`

### src/training/wavelet_feature_selection_demo.py

#### Broad Exceptions

Line 67: `except Exception as e:`
Line 90: `except Exception as e:`
Line 158: `except Exception as e:`
Line 184: `except Exception as e:`
Line 213: `except Exception as e:`
Line 368: `except Exception as e:`
Line 422: `except Exception as e:`
Line 442: `except Exception as e:`

#### Unused Imports

Line 26: `from src.utils.warning_symbols import error`
Line 26: `from src.utils.warning_symbols import critical`
Line 26: `from src.utils.warning_symbols import problem`
Line 26: `from src.utils.warning_symbols import invalid`
Line 26: `from src.utils.warning_symbols import missing`
Line 26: `from src.utils.warning_symbols import timeout`
Line 26: `from src.utils.warning_symbols import connection_error`
Line 26: `from src.utils.warning_symbols import validation_error`
Line 26: `from src.utils.warning_symbols import initialization_error`
Line 26: `from src.utils.warning_symbols import execution_error`

### src/training/calibration_manager.py

#### Broad Exceptions

Line 86: `except Exception:`
Line 116: `except Exception:`
Line 143: `except Exception as e:`
Line 212: `except Exception:`
Line 257: `except Exception:`
Line 301: `except Exception:`
Line 345: `except Exception:`
Line 414: `except Exception as e:`
Line 447: `except Exception:`
Line 478: `except Exception:`
... and 4 more

### src/training/ensemble_creator_simple.py

#### Broad Exceptions

Line 113: `except Exception as e:`
Line 153: `except Exception as e:`
Line 245: `except Exception as e:`
Line 295: `except Exception as e:`
Line 343: `except Exception as e:`
Line 380: `except Exception as e:`
Line 409: `except Exception as e:`
Line 435: `except Exception as e:`
Line 463: `except Exception as e:`
Line 494: `except Exception as e:`
... and 3 more

### src/training/training_orchestrator.py

#### Broad Exceptions

Line 75: `except Exception as e:`
Line 115: `except Exception as e:`
Line 157: `except Exception as e:`
Line 206: `except Exception as e:`
Line 245: `except Exception:`
Line 320: `except Exception:`
Line 343: `except Exception:`
Line 394: `except Exception:`
Line 420: `except Exception as e:`

### src/training/ensemble_creator.py

#### Broad Exceptions

Line 154: `except Exception:`
Line 171: `except Exception:`
Line 191: `except Exception as e:`
Line 209: `except Exception:`
Line 248: `except Exception as e:`
Line 424: `except Exception as e:`
Line 474: `except Exception as e:`
Line 563: `except Exception:`
Line 584: `except Exception as e:`
Line 605: `except Exception as e:`
... and 16 more

#### Todo Comments

Line 41: `debug_training_step,`
Line 290: `@debug_training_step(`
Line 292: `save_debug_artifacts=True,`

### src/training/model_specific_pruning.py

#### Broad Exceptions

Line 138: `except Exception as e:`
Line 208: `except Exception as e:`
Line 273: `except Exception as e:`
Line 320: `except Exception as e:`
Line 374: `except Exception as e:`
Line 418: `except Exception as e:`
Line 461: `except Exception as e:`

#### Unused Imports

Line 3: `import asyncio`
Line 6: `import json`
Line 7: `import os`
Line 8: `from typing import Optional`
Line 9: `from datetime import datetime`
Line 10: `from pathlib import Path`
Line 12: `from sklearn.feature_selection import mutual_info_regression`
Line 12: `from sklearn.feature_selection import f_classif`
Line 12: `from sklearn.feature_selection import f_regression`
Line 12: `from sklearn.feature_selection import SelectKBest`
... and 8 more

### src/training/training_manager.py

#### Broad Exceptions

Line 100: `except Exception:`
Line 130: `except Exception as e:`
Line 173: `except Exception as e:`
Line 208: `except Exception as e:`
Line 221: `except Exception as e:`
Line 244: `except Exception as e:`
Line 267: `except Exception as e:`
Line 290: `except Exception as e:`
Line 313: `except Exception:`
Line 461: `except Exception as e:`
... and 24 more

### src/training/enhanced_coarse_optimizer.py

#### Broad Exceptions

Line 127: `except Exception:`
Line 256: `except Exception:`
Line 336: `except Exception as e:`
Line 381: `except Exception:`
Line 423: `except Exception:`
Line 478: `except Exception:`
Line 495: `except Exception:`
Line 544: `except Exception:`
Line 589: `except Exception as e:`
Line 740: `except Exception as e:`
... and 9 more

#### Unused Imports

Line 19: `from sklearn.model_selection import train_test_split`

### src/training/ensemble_manager.py

#### Broad Exceptions

Line 84: `except Exception as e:`
Line 112: `except Exception as e:`
Line 138: `except Exception as e:`
Line 220: `except Exception as e:`
Line 265: `except Exception as e:`
Line 331: `except Exception as e:`
Line 385: `except Exception as e:`
Line 433: `except Exception as e:`
Line 483: `except Exception as e:`
Line 531: `except Exception as e:`
... and 5 more

### src/training/model_training_integrator.py

#### Broad Exceptions

Line 127: `except Exception as e:`
Line 151: `except Exception as e:`
Line 178: `except Exception as e:`
Line 187: `except Exception as e:`
Line 257: `except Exception as e:`
Line 348: `except Exception as e:`
Line 392: `except Exception as e:`
Line 409: `except Exception as e:`
Line 439: `except Exception as e:`
Line 478: `except Exception as e:`
... and 7 more

#### Unused Imports

Line 21: `from sklearn.model_selection import train_test_split`

### src/training/di_training_manager.py

#### Broad Exceptions

Line 89: `except Exception as e:`
Line 118: `except Exception as e:`
Line 162: `except Exception as e:`
Line 190: `except Exception:`
Line 255: `except Exception as e:`
Line 307: `except Exception as e:`
Line 346: `except Exception as e:`
Line 383: `except Exception as e:`
Line 418: `except Exception as e:`

### src/training/wavelet_feature_selection_workflow.py

#### Broad Exceptions

Line 172: `except Exception as e:`
Line 227: `except Exception as e:`
Line 353: `except Exception as e:`
Line 449: `except Exception as e:`
Line 542: `except Exception as e:`
Line 604: `except Exception as e:`
Line 716: `except Exception as e:`
Line 795: `except Exception as e:`
Line 1024: `except Exception as e:`

#### Unused Imports

Line 30: `from sklearn.model_selection import train_test_split`

### src/training/data_efficiency_optimizer.py

#### Broad Exceptions

Line 240: `except Exception as e:`
Line 253: `except Exception:`
Line 299: `except Exception as e:`
Line 338: `except Exception:`
Line 351: `except Exception as e:`
Line 366: `except Exception:`
Line 409: `except Exception:`
Line 440: `except Exception:`
Line 456: `except Exception as e:`
Line 497: `except Exception:`
... and 4 more

#### Todo Comments

Line 183: `self.logger.debug(f"Current memory usage: {memory_percent:.2f}%")`

### src/training/optimization_manager.py

#### Broad Exceptions

Line 90: `except Exception as e:`
Line 120: `except Exception as e:`
Line 146: `except Exception as e:`
Line 224: `except Exception as e:`
Line 269: `except Exception as e:`
Line 343: `except Exception as e:`
Line 390: `except Exception as e:`
Line 443: `except Exception as e:`
Line 506: `except Exception as e:`
Line 547: `except Exception as e:`
... and 4 more

### src/training/tpsl_optimizer.py

#### Type Ignores

Line 9: `import pandas_ta as ta  # noqa: F401 - ensure .ta accessor is registered`

#### Broad Exceptions

Line 150: `except Exception as e:`
Line 166: `except Exception as e:`
Line 176: `except Exception as e:`
Line 437: `except Exception as e:`

### src/training/enhanced_optimization_orchestrator.py

#### Broad Exceptions

Line 128: `except Exception as e:`
Line 148: `except Exception as e:`
Line 159: `except Exception as e:`
Line 170: `except Exception as e:`

### src/training/data_manager.py

#### Broad Exceptions

Line 144: `except Exception as e:`
Line 555: `except Exception:`
Line 604: `except Exception:`
Line 767: `except Exception as e:`
Line 792: `except Exception:`

#### Unused Imports

Line 11: `from src.utils.warning_symbols import validation_error`
Line 11: `from src.utils.warning_symbols import warning`

### src/training/wavelet_caching_workflow.py

#### Broad Exceptions

Line 42: `except Exception as e:`
Line 83: `except Exception as e:`
Line 148: `except Exception as e:`
Line 171: `except Exception:`
Line 235: `except Exception as e:`
Line 254: `except Exception:`
Line 291: `except Exception as e:`
Line 331: `except Exception as e:`
Line 413: `except Exception as e:`

#### Unused Imports

Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import warning`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import timeout`
Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import validation_error`
... and 2 more

### src/training/optimized_backtester.py

#### Broad Exceptions

Line 305: `except Exception as e:`

### src/training/optimization/cached_optimizer.py

#### Broad Exceptions

Line 69: `except Exception:`
Line 84: `except Exception:`
Line 130: `except Exception:`
Line 157: `except Exception:`
Line 194: `except Exception:`
Line 225: `except Exception:`
Line 269: `except Exception:`
Line 343: `except Exception:`
Line 376: `except Exception:`
Line 399: `except Exception:`

### src/training/optimization/progressive_optimizer.py

#### Broad Exceptions

Line 167: `except Exception:`
Line 248: `except Exception:`
Line 318: `except Exception:`
Line 382: `except Exception:`
Line 427: `except Exception:`
Line 469: `except Exception:`
Line 511: `except Exception:`
Line 553: `except Exception:`
Line 591: `except Exception:`

### src/training/optimization/parallel_optimizer.py

#### Broad Exceptions

Line 97: `except Exception:`
Line 140: `except Exception:`
Line 183: `except Exception:`
Line 226: `except Exception:`
Line 282: `except Exception:`
Line 334: `except Exception:`
Line 360: `except Exception:`
Line 386: `except Exception:`
Line 412: `except Exception:`

### src/training/optimization/rollback_manager.py

#### Broad Exceptions

Line 90: `except Exception:`
Line 153: `except Exception:`
Line 188: `except Exception:`
Line 227: `except Exception as e:`
Line 234: `except Exception:`
Line 270: `except Exception:`
Line 297: `except Exception:`
Line 363: `except Exception:`
Line 410: `except Exception:`
Line 446: `except Exception:`
... and 6 more

### src/training/optimization/computational_optimization_manager.py

#### Broad Exceptions

Line 428: `except Exception as e:`
Line 440: `except Exception:`
Line 871: `except Exception:`
Line 976: `except Exception as e:`
Line 1009: `except Exception as e:`
Line 1064: `except Exception as e:`

#### Unused Imports

Line 27: `from src.utils.warning_symbols import warning`
Line 27: `from src.utils.warning_symbols import critical`
Line 27: `from src.utils.warning_symbols import problem`
Line 27: `from src.utils.warning_symbols import invalid`
Line 27: `from src.utils.warning_symbols import missing`
Line 27: `from src.utils.warning_symbols import timeout`
Line 27: `from src.utils.warning_symbols import connection_error`
Line 27: `from src.utils.warning_symbols import validation_error`
Line 27: `from src.utils.warning_symbols import initialization_error`
Line 27: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 221: `self.logger.debug(f"Cache hit for parameters: {cache_key[:8]}")`
Line 325: `self.logger.debug(`
Line 501: `self.logger.debug(f"Using cached model for key: {model_key[:8]}")`
Line 514: `self.logger.debug(f"Training new model for key: {model_key[:8]}")`
Line 567: `self.logger.debug("Using light complexity model")`
Line 570: `self.logger.debug("Using medium complexity model")`
Line 572: `self.logger.debug("Using heavy complexity model")`
Line 679: `self.logger.debug("Using cached feature selection")`
Line 868: `# Reduce noise: move to debug and include shape`
Line 870: `self.logger.debug(f"Optimized DataFrame memory usage: shape={df.shape}")`

### src/training/optimization/adaptive_trial_allocator.py

#### Broad Exceptions

Line 99: `except Exception:`
Line 133: `except Exception:`
Line 149: `except Exception as e:`
Line 169: `except Exception as e:`
Line 242: `except Exception:`
Line 264: `except Exception:`
Line 307: `except Exception:`
Line 363: `except Exception:`
Line 387: `except Exception:`
Line 408: `except Exception:`
... and 1 more

### src/training/optimization/__init__.py

#### Unused Imports

Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import execution_error`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import initialization_error`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import timeout`
... and 3 more

### src/training/tests/test_regime_change_prediction.py

#### Unused Imports

Line 3: `import asyncio`
Line 6: `from datetime import datetime`
Line 6: `from datetime import timedelta`
Line 7: `from typing import Any`
Line 7: `from typing import Dict`

### src/training/steps/multi_timeframe_hmm_ensemble.py

#### Broad Exceptions

Line 216: `except Exception as e:`
Line 257: `except Exception as e:`
Line 273: `except Exception as e:`
Line 359: `except Exception as e:`
Line 382: `except Exception as e:`
Line 391: `except Exception as e:`
Line 435: `except Exception as e:`
Line 467: `except Exception as e:`
Line 569: `except Exception as e:`
Line 608: `except Exception as e:`
... and 6 more

#### Unused Imports

Line 25: `import warnings`
Line 28: `from concurrent.futures import ThreadPoolExecutor`
Line 28: `from concurrent.futures import ProcessPoolExecutor`
Line 29: `import gc`
Line 38: `from sklearn.metrics import mean_absolute_error`

### src/training/steps/update_steps_for_unified_data.py

#### Unused Imports

Line 10: `import os`
Line 11: `from typing import List`
Line 95: `from src.config.constants import FULL_TRAINING_LOOKBACK_DAYS`
Line 95: `from src.config.constants import SHORT_BLANK_LOOKBACK_DAYS`

### src/training/steps/step11_confidence_calibration_validator.py

#### Broad Exceptions

Line 140: `except Exception:`
Line 225: `except Exception as e:`
Line 317: `except Exception as e:`

### src/training/steps/step9_5_hmm_lm_generalist_training.py

#### Broad Exceptions

Line 158: `except Exception as e:`
Line 214: `except Exception as e:`
Line 278: `except Exception as e:`
Line 393: `except Exception as e:`
Line 459: `except Exception as e:`
Line 512: `except Exception as e:`
Line 558: `except Exception as e:`
Line 592: `except Exception as e:`
Line 960: `except Exception as e:`

#### Unused Imports

Line 4: `import concurrent.futures`
Line 7: `import pickle`
Line 19: `from sklearn.preprocessing import StandardScaler`
Line 19: `from sklearn.preprocessing import LabelEncoder`
Line 20: `from sklearn.model_selection import TimeSeriesSplit`
Line 25: `from src.utils.warning_symbols import error`
Line 25: `from src.utils.warning_symbols import failed`
Line 25: `from src.utils.warning_symbols import success`
Line 26: `from src.utils.decorators import guard_dataframe_nulls`
Line 26: `from src.utils.decorators import with_tracing_span`

### src/training/steps/step6_hmm_based_training.py

#### Broad Exceptions

Line 134: `except Exception as e:`
Line 267: `except Exception as e:`
Line 320: `except Exception as e:`
Line 554: `except Exception as e:`
Line 617: `except Exception as e:`
Line 631: `except Exception as e:`
Line 674: `except Exception as e:`
Line 773: `except Exception as e:`
Line 804: `except Exception as e:`
Line 863: `except Exception as e:`
... and 61 more

#### Unused Imports

Line 3: `import asyncio`
Line 19: `from sklearn.model_selection import TimeSeriesSplit`
Line 31: `from src.utils.warning_symbols import error`
Line 31: `from src.utils.warning_symbols import failed`
Line 31: `from src.utils.warning_symbols import success`
Line 32: `from src.utils.decorators import guard_dataframe_nulls`
Line 32: `from src.utils.decorators import with_tracing_span`

#### Todo Comments

Line 817: `self.logger.debug(`
Line 836: `self.logger.debug(`
Line 841: `self.logger.debug(f"No split files found for {timeframe}")`
Line 884: `self.logger.debug(`
Line 889: `self.logger.debug(f"No legacy pickle files found for {timeframe}")`
Line 1787: `self.logger.debug(`
Line 3698: `self.logger.debug(`
Line 3829: `self.logger.debug(`
Line 4800: `self.logger.debug(f"Error processing S/R sample {idx}: {e}")`
Line 4929: `debug_training_step,`
... and 2 more

### src/training/steps/step5_hmm_based_training_validator.py

#### Broad Exceptions

Line 253: `except Exception as e:`
Line 352: `except Exception as e:`
Line 437: `except Exception as e:`
Line 553: `except Exception as e:`
Line 560: `except Exception as e:`
Line 620: `except Exception as e:`
Line 668: `except Exception:`

### src/training/steps/step8_tactician_labeling_validator.py

#### Broad Exceptions

Line 164: `except Exception:`
Line 242: `except Exception:`
Line 244: `except Exception:`
Line 251: `except Exception:`
Line 337: `except Exception:`
Line 412: `except Exception:`
Line 414: `except Exception:`
Line 421: `except Exception:`
Line 465: `except Exception:`
Line 505: `except:`
... and 2 more

### src/training/steps/hmm_feature_enhancer.py

#### Broad Exceptions

Line 52: `except Exception as e:`
Line 81: `except Exception as e:`
Line 115: `except Exception as e:`
Line 149: `except Exception as e:`
Line 203: `except Exception as e:`
Line 233: `except Exception as e:`

#### Unused Imports

Line 5: `from typing import Dict`
Line 5: `from typing import List`
Line 5: `from typing import Tuple`
Line 5: `from typing import Optional`

### src/training/steps/step1_5_data_converter.py

#### Debug Statements

Line 2048: `print(f"🔍 Looking for klines at: {parquet_path}")`

#### Broad Exceptions

Line 88: `except Exception:`
Line 100: `except Exception:`
Line 157: `except Exception:`
Line 169: `except Exception:`
Line 181: `except Exception:`
Line 183: `except Exception:`
Line 195: `except Exception:`
Line 292: `except Exception:`
Line 306: `except Exception:`
Line 367: `except Exception:`
... and 56 more

#### Unused Imports

Line 6: `import pickle`
Line 45: `from src.utils.warning_symbols import error`
Line 45: `from src.utils.warning_symbols import warning`
Line 45: `from src.utils.warning_symbols import critical`
Line 45: `from src.utils.warning_symbols import problem`
Line 45: `from src.utils.warning_symbols import failed`
Line 45: `from src.utils.warning_symbols import invalid`
Line 45: `from src.utils.warning_symbols import missing`
Line 45: `from src.utils.warning_symbols import timeout`
Line 45: `from src.utils.warning_symbols import connection_error`
... and 3 more

#### Todo Comments

Line 307: `# Leave column as-is if conversion fails; log at debug level`
Line 309: `self.logger.debug(`
Line 489: `self.logger.debug("Manifest update skipped")`
Line 577: `self.logger.debug(`
Line 2145: `self.logger.debug(`
Line 2303: `self.logger.debug(`

### src/training/steps/step7_analyst_ensemble_creation_validator.py

#### Broad Exceptions

Line 76: `except Exception as e:`
Line 112: `except Exception as e:`
Line 191: `except Exception as e:`

#### Unused Imports

Line 5: `import logging`
Line 6: `from typing import List`
Line 6: `from typing import Tuple`
Line 12: `from src.utils.decorators import guard_dataframe_nulls`
Line 12: `from src.utils.decorators import with_tracing_span`

### src/training/steps/step14_monte_carlo_validation.py

#### Broad Exceptions

Line 31: `except Exception as e:`
Line 130: `except Exception:`
Line 168: `except Exception:`
Line 186: `except Exception as e:`
Line 313: `except Exception as e:`

#### Unused Imports

Line 6: `import pickle`
Line 11: `from src.utils.warning_symbols import validation_error`
Line 15: `from src.training.steps.unified_data_loader import get_unified_data_loader`

#### Todo Comments

Line 198: `debug_training_step,`
Line 248: `@debug_training_step(`
Line 250: `save_debug_artifacts=True,`

### src/training/steps/step12_final_parameters_optimization_validator.py

#### Broad Exceptions

Line 155: `except Exception:`
Line 221: `except Exception as e:`
Line 311: `except Exception as e:`
Line 394: `except Exception as e:`

### src/training/steps/feature_artifact_loader.py

#### Broad Exceptions

Line 108: `except Exception as e:`
Line 193: `except Exception as e:`
Line 200: `except Exception as e:`
Line 282: `except Exception as e:`
Line 312: `except Exception as e:`
Line 396: `except Exception as e:`
Line 470: `except Exception as e:`
Line 544: `except Exception as e:`
Line 631: `except Exception as e:`
Line 721: `except Exception as e:`
... and 2 more

#### Unused Imports

Line 13: `from typing import Optional`

#### Todo Comments

Line 23: `debug_training_step,`
Line 61: `@debug_training_step(`
Line 63: `save_debug_artifacts=False,`
Line 105: `logger.debug(f"Generated artifact paths for {exchange}_{symbol}: {list(paths.keys())}")`
Line 142: `@debug_training_step(`
Line 144: `save_debug_artifacts=False,`
Line 183: `logger.debug(f"Missing artifact file: {paths[file_type]}")`
Line 234: `@debug_training_step(`
Line 236: `save_debug_artifacts=False,`
Line 346: `@debug_training_step(`
... and 14 more

### src/training/steps/data_downloader.py

#### Type Ignores

Line 30: `model_training_cfg: dict[str, Any] | None = CONFIG.get("MODEL_TRAINING")  # type: ignore[assignment]`
Line 34: `lookback_years = int(model_training_cfg["lookback_years"])  # type: ignore[arg-type]`

#### Broad Exceptions

Line 35: `except Exception:`
Line 54: `except Exception as e:`
Line 74: `except Exception as e:`

#### Unused Imports

Line 1: `from __future__ import annotations`
Line 41: `from backtesting.ares_data_downloader_optimized import DownloadConfig`
Line 61: `from backtesting.ares_data_downloader_clean import DownloadConfig`

### src/training/steps/step4_processing_labeling_validator.py

#### Broad Exceptions

Line 144: `except Exception as e:`
Line 150: `except Exception as e:`
Line 204: `except Exception as e:`
Line 223: `except Exception as e:`
Line 271: `except Exception as e:`

#### Unused Imports

Line 10: `import pickle`
Line 13: `from typing import List`

### src/training/steps/step2_market_regime_classification.py

#### Broad Exceptions

Line 62: `except Exception as e:`
Line 200: `except Exception:`
Line 283: `except Exception as e:`
Line 546: `except Exception as e:`
Line 654: `except Exception as e:`

#### Unused Imports

Line 114: `from src.config.constants import FULL_TRAINING_LOOKBACK_DAYS`
Line 114: `from src.config.constants import SHORT_BLANK_LOOKBACK_DAYS`

#### Todo Comments

Line 558: `debug_training_step,`
Line 595: `@debug_training_step(`
Line 597: `save_debug_artifacts=True,`

### src/training/steps/step3_hmm_regime_discovery.py

#### Type Ignores

Line 59: `import hdbscan  # type: ignore`

#### Broad Exceptions

Line 62: `except Exception:`
Line 103: `except Exception:`
Line 106: `except Exception as e:`
Line 194: `except Exception as e:`
Line 261: `except Exception as e:`
Line 276: `except Exception as e:`
Line 458: `except Exception as e:`
Line 469: `except Exception as e:`
Line 489: `except:`
Line 515: `except Exception as e:`
... and 33 more

#### Unused Imports

Line 5: `import math`
Line 6: `import warnings`
Line 7: `import sys`
Line 9: `import contextlib`
Line 13: `from typing import Union`
Line 17: `from concurrent.futures import ThreadPoolExecutor`
Line 17: `from concurrent.futures import ProcessPoolExecutor`
Line 19: `import psutil`
Line 29: `from src.utils.error_handler import safe_division`
Line 29: `from src.utils.error_handler import clean_dataframe`
... and 7 more

#### Todo Comments

Line 256: `elif level.lower() == "debug":`
Line 257: `logger.debug(log_data)`
Line 863: `f"✅ Model and scaler files saved successfully", "debug", "ModelCache"`
Line 874: `"debug",`
Line 1173: `# Debug: log all available features and their block assignments`
Line 2860: `# DEBUG: Log the timeframes_to_process variable`
Line 2861: `logger.info(f"🔍 DEBUG: timeframes_to_process = {timeframes_to_process}")`
Line 2862: `logger.info(f"🔍 DEBUG: TIMEFRAMES = {TIMEFRAMES}")`
Line 2863: `logger.info(f"🔍 DEBUG: data_dir = {data_dir}")`
Line 2864: `logger.info(f"🔍 DEBUG: exchange = {exchange}, symbol = {symbol}")`
... and 9 more

### src/training/steps/step2_feature_engineering.py

#### Broad Exceptions

Line 101: `except Exception:`
Line 220: `except Exception:`
Line 563: `except Exception:`
Line 594: `except Exception:`
Line 637: `except Exception as e:`
Line 719: `except Exception as e:`
Line 733: `except Exception:`
Line 745: `except Exception as e:`
Line 786: `except Exception as e:`
Line 905: `except Exception:`
... and 14 more

#### Unused Imports

Line 13: `import shutil`
Line 1003: `from statsmodels.stats.outliers_influence import variance_inflation_factor`

#### Todo Comments

Line 32: `debug_training_step,`
Line 68: `@debug_training_step(`
Line 70: `save_debug_artifacts=False,`
Line 135: `@debug_training_step(`
Line 137: `save_debug_artifacts=False,`
Line 190: `@debug_training_step(`
Line 192: `save_debug_artifacts=False,`
Line 255: `@debug_training_step(`
Line 257: `save_debug_artifacts=False,`
Line 309: `@debug_training_step(`
... and 20 more

### src/training/steps/step9_tactician_specialist_training_validator.py

#### Broad Exceptions

Line 162: `except Exception:`
Line 255: `except Exception as e:`
Line 351: `except Exception as e:`
Line 475: `except Exception:`
Line 482: `except Exception as e:`
Line 520: `except Exception:`

### src/training/steps/vectorized_advanced_feature_engineering.py

#### Broad Exceptions

Line 97: `except Exception:`
Line 146: `except Exception:`
Line 239: `except Exception as e:`
Line 281: `except Exception as e:`
Line 292: `except Exception as e:`
Line 330: `except Exception as e:`
Line 364: `except Exception as e:`
Line 403: `except Exception as e:`
Line 407: `except Exception as e:`
Line 476: `except Exception as e:`
... and 95 more

#### Unused Imports

Line 11: `import pywt`
Line 13: `from typing import List`
Line 13: `from typing import Optional`
Line 14: `from datetime import timedelta`
Line 16: `import random`
Line 24: `from src.training.steps.raw_data_quality_checker import validate_raw_data_quality`
Line 25: `from src.utils.data_quality_decorators import validate_microstructure_data_quality`
Line 44: `from src.utils.parallel_processing_optimizer import parallel_feature_engineering`

#### Todo Comments

Line 53: `debug_training_step,`
Line 714: `# Debug: Check what columns are available`
Line 787: `# Debug: Check feature values - only show features with >0.1% NaN values`
Line 1771: `# Debug: Log input data structure`
Line 1807: `# Debug: Log output data structure`
Line 1960: `self.logger.debug(f"✅ Generated {len(features)} simple features for {timeframe} timeframe")`
Line 2062: `self.logger.debug(f"Comprehensive method failed: {e1}")`
Line 2069: `self.logger.debug(f"Basic method failed: {e2}")`
Line 2076: `self.logger.debug(f"Inline method failed: {e3}")`
Line 2116: `self.logger.debug(`
... and 57 more

### src/training/steps/step15_ab_testing_validator.py

#### Broad Exceptions

Line 157: `except Exception:`
Line 247: `except Exception as e:`
Line 349: `except Exception as e:`
Line 433: `except Exception as e:`

### src/training/steps/precompute_wavelet_features.py

#### Broad Exceptions

Line 82: `except Exception:`
Line 130: `except Exception:`
Line 169: `except Exception:`
Line 200: `except Exception:`
Line 248: `except Exception:`
Line 291: `except Exception:`
Line 315: `except Exception:`
Line 335: `except Exception:`
Line 375: `except Exception:`
Line 423: `except Exception:`
... and 3 more

#### Todo Comments

Line 367: `self.logger.debug(f"💾 Cached batch {batch_idx + 1}/{total_batches}")`

### src/training/steps/step1_data_collection_validator.py

#### Broad Exceptions

Line 77: `except Exception:`
Line 100: `except Exception as e:`
Line 209: `except Exception as e:`

#### Unused Imports

Line 6: `import pickle`
Line 13: `from src.utils.warning_symbols import validation_error`

### src/training/steps/step4_processing_labeling.py

#### Broad Exceptions

Line 60: `except Exception:`
Line 121: `except Exception:`
Line 133: `except Exception:`
Line 137: `except Exception as e:`
Line 263: `except Exception:`
Line 301: `except Exception as e:`
Line 322: `except Exception as e:`
Line 335: `except Exception as e:`
Line 340: `except Exception as e:`

#### Unused Imports

Line 10: `from src.utils.logger import system_logger`
Line 18: `from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering`
Line 21: `from src.training.enhanced_training_manager_optimized import MemoryEfficientDataManager`
Line 175: `from src.config.constants import FULL_TRAINING_LOOKBACK_DAYS`
Line 175: `from src.config.constants import SHORT_BLANK_LOOKBACK_DAYS`

### src/training/steps/step5_5_unified_regime_intelligence.py

#### Type Ignores

Line 1781: `import optuna  # type: ignore`

#### Broad Exceptions

Line 304: `except Exception as e:`
Line 369: `except Exception as e:`
Line 461: `except Exception as e:`
Line 537: `except Exception as e:`
Line 611: `except Exception as e:`
Line 633: `except Exception as e:`
Line 659: `except Exception as e:`
Line 684: `except Exception as e:`
Line 714: `except Exception as e:`
Line 759: `except Exception as e:`
... and 22 more

#### Unused Imports

Line 23: `import asyncio`
Line 30: `from typing import Tuple`
Line 30: `from typing import Union`
Line 41: `from sklearn.ensemble import RandomForestClassifier`
Line 41: `from sklearn.ensemble import RandomForestRegressor`
Line 42: `from sklearn.model_selection import TimeSeriesSplit`
Line 47: `from src.utils.warning_symbols import success`
Line 48: `from src.utils.decorators import guard_dataframe_nulls`
Line 48: `from src.utils.decorators import with_tracing_span`

#### Todo Comments

Line 1942: `debug_training_step,`
Line 1990: `@debug_training_step(`
Line 1992: `save_debug_artifacts=True,`
Line 2028: `# Log step parameters for debugging`

### src/training/steps/step2_feature_engineering_validator.py

#### Broad Exceptions

Line 178: `except Exception as e:`
Line 188: `except Exception as e:`
Line 290: `except Exception as e:`
Line 300: `except Exception as e:`
Line 358: `except Exception as e:`
Line 432: `except Exception as e:`
Line 441: `except Exception as e:`
Line 523: `except Exception as e:`
Line 552: `except Exception as e:`
Line 591: `except Exception as e:`
... and 1 more

#### Unused Imports

Line 13: `from typing import List`

### src/training/steps/step6_analyst_enhancement_validator.py

#### Broad Exceptions

Line 92: `except Exception as e:`
Line 111: `except Exception as e:`
Line 132: `except Exception as e:`
Line 151: `except Exception as e:`
Line 171: `except Exception as e:`
Line 276: `except Exception as e:`
Line 365: `except Exception as e:`
Line 428: `except Exception as e:`
Line 455: `except Exception:`
Line 473: `except Exception as e:`
... and 1 more

### src/training/steps/unified_data_loader.py

#### Broad Exceptions

Line 90: `except Exception:`
Line 147: `except Exception as e:`
Line 162: `except Exception as e:`
Line 220: `except Exception as e:`
Line 298: `except Exception as e:`
Line 377: `except Exception as e:`
Line 427: `except Exception as e:`
Line 453: `except Exception as e:`
Line 537: `except Exception as e:`
Line 559: `except Exception as e:`
... and 12 more

#### Unused Imports

Line 10: `from typing import Union`
Line 10: `from typing import Iterator`
Line 36: `from src.utils.warning_symbols import error`
Line 36: `from src.utils.warning_symbols import warning`
Line 36: `from src.utils.warning_symbols import failed`
Line 36: `from src.utils.warning_symbols import missing`
Line 45: `from src.utils.training_pipeline_decorators import monitor_pipeline_step`
Line 45: `from src.utils.training_pipeline_decorators import validate_pipeline_input`
Line 45: `from src.utils.training_pipeline_decorators import monitor_pipeline_performance`
Line 45: `from src.utils.training_pipeline_decorators import PipelineValidationLevel`

### src/training/steps/step12_final_parameters_optimization.py

#### Type Ignores

Line 130: `return len(obj)  # type: ignore[arg-type]`

#### Broad Exceptions

Line 112: `except Exception:`
Line 131: `except Exception:`
Line 141: `except Exception:`
Line 167: `except Exception:`
Line 197: `except Exception:`
Line 213: `except Exception:`
Line 229: `except Exception:`
Line 239: `except Exception as e:`
Line 263: `except Exception:`
Line 285: `except Exception:`
... and 29 more

#### Unused Imports

Line 24: `from src.training.steps.unified_data_loader import get_unified_data_loader`

#### Todo Comments

Line 1628: `debug_training_step,`
Line 1678: `@debug_training_step(`
Line 1680: `save_debug_artifacts=True,`

### src/training/steps/vectorized_labelling_orchestrator.py

#### Broad Exceptions

Line 33: `except Exception:`
Line 40: `except Exception:`
Line 65: `except Exception:`
Line 182: `except Exception as e:`
Line 196: `except Exception:`
Line 237: `except Exception as e:`
Line 250: `except Exception as e:`
Line 269: `except Exception as e:`
Line 328: `except Exception:`
Line 362: `except Exception:`
... and 70 more

#### Unused Imports

Line 14: `import pywt`
Line 15: `from typing import Dict`
Line 15: `from typing import Tuple`
Line 16: `from datetime import timedelta`

#### Todo Comments

Line 147: `# Debug snapshots for logging`
Line 148: `self._debug_raw_ohlcv: pd.DataFrame | None = None`
Line 149: `self._debug_price_returns: pd.DataFrame | None = None`
Line 161: `if self._debug_raw_ohlcv is not None:`
Line 163: `if c in self._debug_raw_ohlcv.columns and c not in sample.columns:`
Line 164: `sample[c] = self._debug_raw_ohlcv[c]`
Line 165: `if self._debug_price_returns is not None:`
Line 166: `for c in self._debug_price_returns.columns:`
Line 168: `sample[c] = self._debug_price_returns[c]`
Line 359: `self._debug_raw_ohlcv = price_data[`
... and 10 more

### src/training/steps/step2_market_regime_classification_validator.py

#### Broad Exceptions

Line 143: `except Exception:`
Line 150: `except Exception:`
Line 164: `except Exception:`
Line 233: `except Exception as e:`
Line 329: `except Exception as e:`
Line 390: `except Exception as e:`
Line 466: `except Exception as e:`

### src/training/steps/step9_tactician_specialist_training.py

#### Broad Exceptions

Line 40: `except Exception as e:`
Line 63: `except Exception as e:`
Line 185: `except Exception as e:`
Line 228: `except Exception as e:`
Line 472: `except Exception:`
Line 474: `except Exception:`
Line 479: `except Exception:`
Line 503: `except Exception:`
Line 512: `except Exception:`
Line 661: `except Exception:`
... and 14 more

#### Unused Imports

Line 19: `from src.training.steps.unified_data_loader import get_unified_data_loader`

#### Todo Comments

Line 186: `self.logger.debug(`
Line 1473: `debug_training_step,`
Line 1523: `@debug_training_step(`
Line 1525: `save_debug_artifacts=True,`

### src/training/steps/backtesting_with_cached_features.py

#### Broad Exceptions

Line 99: `except Exception as e:`
Line 150: `except Exception as e:`
Line 238: `except Exception as e:`
Line 285: `except Exception as e:`
Line 368: `except Exception as e:`
Line 414: `except Exception as e:`
Line 439: `except Exception:`
Line 448: `except Exception:`
Line 461: `except Exception as e:`
Line 473: `except Exception as e:`
... and 3 more

#### Unused Imports

Line 24: `from src.utils.warning_symbols import warning`
Line 24: `from src.utils.warning_symbols import critical`
Line 24: `from src.utils.warning_symbols import problem`
Line 24: `from src.utils.warning_symbols import failed`
Line 24: `from src.utils.warning_symbols import invalid`
Line 24: `from src.utils.warning_symbols import missing`
Line 24: `from src.utils.warning_symbols import timeout`
Line 24: `from src.utils.warning_symbols import connection_error`
Line 24: `from src.utils.warning_symbols import validation_error`
Line 24: `from src.utils.warning_symbols import initialization_error`
... and 1 more

### src/training/steps/step11_confidence_calibration.py

#### Type Ignores

Line 814: `def fit(self, X, y):  # noqa: D401`

#### Broad Exceptions

Line 25: `except Exception:  # pragma: no cover`
Line 111: `except Exception as e:`
Line 124: `except Exception:`
Line 152: `except Exception:`
Line 217: `except Exception as e:`
Line 226: `except Exception:`
Line 269: `except Exception:`
Line 291: `except Exception:`
Line 307: `except Exception:`
Line 325: `except Exception:`
... and 18 more

#### Unused Imports

Line 7: `from datetime import datetime`
Line 21: `from src.training.steps.unified_data_loader import get_unified_data_loader`

#### Todo Comments

Line 851: `debug_training_step,`
Line 900: `@debug_training_step(`
Line 902: `save_debug_artifacts=True,`

### src/training/steps/step5_5_unified_regime_intelligence_validator.py

#### Broad Exceptions

Line 82: `except Exception as e:`
Line 109: `except Exception as e:`
Line 213: `except Exception as e:`
Line 269: `except Exception as e:`
Line 296: `except Exception as e:`
Line 377: `except Exception as e:`
Line 427: `except Exception:`
Line 457: `except Exception as e:`
Line 507: `except Exception as e:`
Line 535: `except Exception as e:`
... and 4 more

#### Unused Imports

Line 9: `import asyncio`
Line 14: `from typing import List`
Line 14: `from typing import Optional`
Line 14: `from typing import Tuple`
Line 14: `from typing import Union`
Line 20: `from sklearn.preprocessing import StandardScaler`
Line 24: `from src.utils.warning_symbols import error`
Line 24: `from src.utils.warning_symbols import failed`
Line 24: `from src.utils.warning_symbols import success`
Line 24: `from src.utils.warning_symbols import warning`
... and 4 more

### src/training/steps/step16_saving_validator.py

#### Broad Exceptions

Line 154: `except Exception:`
Line 241: `except Exception as e:`
Line 292: `except Exception:`
Line 317: `except Exception:`
Line 337: `except Exception:`
Line 344: `except Exception:`
Line 379: `except Exception:`
Line 484: `except Exception as e:`

### src/training/steps/sr_outcome_model_trainer.py

#### Broad Exceptions

Line 114: `except Exception as e:`
Line 162: `except Exception as e:`
Line 200: `except Exception as e:`
Line 259: `except Exception as e:`
Line 275: `except Exception as e:`
Line 396: `except Exception as e:`
Line 428: `except Exception as e:`
Line 473: `except Exception as e:`
Line 488: `except Exception as e:`
Line 560: `except Exception as e:`
... and 11 more

#### Unused Imports

Line 10: `import asyncio`
Line 14: `from datetime import timedelta`
Line 31: `from src.utils.warning_symbols import error`
Line 31: `from src.utils.warning_symbols import failed`
Line 31: `from src.utils.warning_symbols import success`
Line 32: `from src.utils.data_quality_decorators import validate_data_quality`
Line 32: `from src.utils.data_quality_decorators import ValidationLevel`

#### Todo Comments

Line 260: `self.logger.debug(f"Error labeling sample {idx}: {e}")`

### src/training/steps/step1_data_collection.py

#### Debug Statements

Line 138: `print(f"🔍 Looking for files matching pattern: {pattern}")`
Line 139: `print(f"🔍 Searching in directory: data_cache")`
Line 140: `print(f"🔍 Current working directory: {os.getcwd()}")`
Line 1048: `print("🔍 Checking existing consolidated file...")`
Line 1106: `print("🔍 Looking for aggtrades files with patterns:")`
Line 1436: `print("🔍 Running data collection quality analysis...")`

#### Broad Exceptions

Line 153: `except Exception as e:`
Line 207: `except Exception as e:`
Line 277: `except Exception as e:`
Line 351: `except Exception as e:`
Line 437: `except Exception as e:`
Line 609: `except Exception as e:`
Line 688: `except Exception as e:`
Line 1430: `except Exception as e:`
Line 1473: `except Exception as e:`
Line 1481: `except Exception as e:`

#### Todo Comments

Line 843: `debug_training_step,`
Line 875: `@debug_training_step(`
Line 877: `save_debug_artifacts=True,`

### src/training/steps/step3_hmm_regime_discovery_validator.py

#### Broad Exceptions

Line 111: `except Exception as e:`
Line 127: `except Exception as e:`
Line 176: `except Exception as e:`

#### Unused Imports

Line 10: `import os`
Line 14: `from typing import List`
Line 14: `from typing import Optional`

### src/training/steps/raw_data_quality_checker.py

#### Broad Exceptions

Line 129: `except Exception as e:`
Line 172: `except Exception as e:`
Line 196: `except Exception as e:`
Line 215: `except Exception as e:`
Line 386: `except Exception as e:`
Line 475: `except Exception as e:`
Line 515: `except Exception as e:`
Line 687: `except Exception as e:`
Line 696: `except Exception as e:`
Line 805: `except Exception as e:`
... and 12 more

#### Unused Imports

Line 10: `from typing import Tuple`
Line 21: `from src.utils.warning_symbols import error`
Line 21: `from src.utils.warning_symbols import warning`
Line 22: `from src.utils.error_handler import handle_errors`

#### Todo Comments

Line 803: `self.logger.debug(f"⚠️ No data found in {file_path} for gap period")`
Line 806: `self.logger.debug(f"⚠️ Error loading {file_path}: {e}")`
Line 1135: `self.logger.debug(f"⚠️ Failed to parse {col}: {e}")`
Line 1149: `self.logger.debug(f"⚠️ Failed to parse existing index: {e}")`
Line 1232: `self.logger.debug(f"⚠️ Error estimating timeframe: {e}")`

### src/training/steps/step13_walk_forward_validation.py

#### Broad Exceptions

Line 31: `except Exception as e:`
Line 109: `except Exception:`
Line 142: `except Exception:`
Line 155: `except Exception as e:`
Line 282: `except Exception:`

#### Unused Imports

Line 6: `import pickle`
Line 15: `from src.training.steps.unified_data_loader import get_unified_data_loader`

#### Todo Comments

Line 167: `debug_training_step,`
Line 217: `@debug_training_step(`
Line 219: `save_debug_artifacts=True,`

### src/training/steps/step7_analyst_ensemble_creation.py

#### Broad Exceptions

Line 86: `except Exception as e:`
Line 114: `except Exception as e:`
Line 121: `except Exception as e:`
Line 182: `except Exception as e:`
Line 208: `except Exception as e:`
Line 240: `except Exception as e:`
Line 272: `except Exception as e:`
Line 302: `except Exception as e:`

#### Unused Imports

Line 5: `import logging`
Line 6: `from typing import List`
Line 9: `from sklearn.ensemble import VotingClassifier`
Line 10: `from sklearn.model_selection import cross_val_score`
Line 15: `from src.utils.warning_symbols import error`
Line 15: `from src.utils.warning_symbols import failed`
Line 15: `from src.utils.warning_symbols import success`
Line 15: `from src.utils.warning_symbols import warning`
Line 16: `from src.utils.decorators import guard_dataframe_nulls`
Line 16: `from src.utils.decorators import with_tracing_span`

### src/training/steps/step14_monte_carlo_validation_validator.py

#### Broad Exceptions

Line 162: `except Exception as e:`
Line 250: `except Exception as e:`
Line 366: `except Exception as e:`
Line 460: `except Exception as e:`

### src/training/steps/step16_saving.py

#### Type Ignores

Line 207: `import mlflow  # type: ignore`

#### Broad Exceptions

Line 77: `except Exception:`
Line 98: `except Exception:`
Line 138: `except Exception as e:`
Line 188: `except Exception as e:`
Line 254: `except Exception as e:`
Line 316: `except Exception as e:`
Line 442: `except Exception as e:`

#### Unused Imports

Line 13: `from src.utils.warning_symbols import error`
Line 17: `from src.training.steps.unified_data_loader import get_unified_data_loader`

#### Todo Comments

Line 328: `debug_training_step,`
Line 377: `@debug_training_step(`
Line 379: `save_debug_artifacts=True,`

### src/training/steps/step5_regime_data_splitting_validator.py

#### Broad Exceptions

Line 65: `except Exception as e:`

#### Unused Imports

Line 7: `import pickle`
Line 14: `from src.utils.warning_symbols import error`
Line 14: `from src.utils.warning_symbols import validation_error`
Line 23: `from src.config import CONFIG`

### src/training/steps/step13_walk_forward_validation_validator.py

#### Broad Exceptions

Line 163: `except Exception as e:`
Line 255: `except Exception as e:`
Line 331: `except Exception as e:`
Line 403: `except Exception as e:`

### src/training/steps/step4_regime_data_splitting.py

#### Broad Exceptions

Line 108: `except Exception as e:`
Line 125: `except Exception as e:`

#### Unused Imports

Line 7: `from typing import Optional`
Line 13: `from src.utils.warning_symbols import failed`
Line 37: `from src.config.constants import FULL_TRAINING_LOOKBACK_DAYS`
Line 37: `from src.config.constants import SHORT_BLANK_LOOKBACK_DAYS`

#### Todo Comments

Line 160: `debug_training_step,`
Line 208: `@debug_training_step(`
Line 210: `save_debug_artifacts=True,`

### src/training/steps/step6_analyst_enhancement.py

#### Type Ignores

Line 75: `_NP_ORIGINAL_BITGEN_CTOR = None  # type: ignore[var-annotated]`
Line 78: `def _normalized_numpy_bitgen_ctor(bit_generator_name, state=None, *args, **kwargs):  # type: ignore[override]`
Line 92: `return _NP_ORIGINAL_BITGEN_CTOR(name_candidate, effective_state)  # type: ignore[misc]`
Line 95: `return _NP_ORIGINAL_BITGEN_CTOR(name_candidate)  # type: ignore[misc]`
Line 96: `except Exception as ctor_exc:  # noqa: BLE001`
Line 103: `import numpy.random._mt19937 as _mt  # type: ignore[attr-defined]`
Line 121: `import numpy.random._pickle as np_random_pickle  # type: ignore[attr-defined]`
Line 129: `np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor  # type: ignore[attr-defined]`
Line 133: `except Exception as _shim_exc:  # noqa: BLE001`

#### Broad Exceptions

Line 87: `except Exception:`
Line 106: `except Exception:`
Line 110: `except Exception:`
Line 206: `except Exception as e:`
Line 227: `except Exception as e:`
Line 267: `except Exception:`
Line 285: `except Exception:`
Line 293: `except Exception:`
Line 308: `except Exception:`
Line 324: `except Exception as e:`
... and 61 more

#### Unused Imports

Line 22: `from sklearn.metrics import f1_score`
Line 48: `from src.utils.warning_symbols import critical`
Line 48: `from src.utils.warning_symbols import problem`
Line 48: `from src.utils.warning_symbols import invalid`
Line 48: `from src.utils.warning_symbols import missing`
Line 48: `from src.utils.warning_symbols import connection_error`
Line 48: `from src.utils.warning_symbols import validation_error`
Line 48: `from src.utils.warning_symbols import initialization_error`
Line 48: `from src.utils.warning_symbols import execution_error`
Line 63: `from src.utils.data_quality_decorators import validate_feature_engineering_with_lookahead_bias_detection`
... and 3 more

#### Todo Comments

Line 3768: `debug_training_step,`
Line 3817: `@debug_training_step(`
Line 3819: `save_debug_artifacts=True,`
Line 3869: `# Log step parameters for debugging`

### src/training/steps/optimized_step_executor.py

#### Broad Exceptions

Line 215: `except Exception as e:`
Line 258: `except Exception as e:`
Line 311: `except Exception as e:`
Line 340: `except Exception as e:`
Line 389: `except Exception as e:`
Line 445: `except Exception as e:`
Line 500: `except Exception as e:`
Line 545: `except Exception as e:`
Line 567: `except Exception as e:`
Line 791: `except Exception as e:`

#### Unused Imports

Line 7: `from concurrent.futures import as_completed`
Line 8: `from pathlib import Path`
Line 9: `from typing import List`
Line 9: `from typing import Optional`
Line 9: `from typing import Tuple`
Line 15: `from src.utils.error_handler import handle_errors`
Line 15: `from src.utils.error_handler import handle_specific_errors`
Line 18: `from src.utils.data_quality_decorators import validate_data_quality`
Line 18: `from src.utils.data_quality_decorators import ValidationLevel`

### src/training/steps/step15_ab_testing.py

#### Broad Exceptions

Line 32: `except Exception as e:`
Line 98: `except Exception:`
Line 195: `except Exception as e:`
Line 321: `except Exception as e:`

#### Unused Imports

Line 16: `from src.training.steps.unified_data_loader import get_unified_data_loader`

#### Todo Comments

Line 209: `debug_training_step,`
Line 259: `@debug_training_step(`
Line 261: `save_debug_artifacts=True,`

### src/training/steps/step8_tactician_labeling.py

#### Broad Exceptions

Line 201: `except Exception as e:`
Line 231: `except Exception:`
Line 250: `except Exception:`
Line 272: `except Exception as e:`
Line 408: `except Exception:`
Line 424: `except Exception:`
Line 426: `except Exception:`
Line 453: `except Exception:`
Line 467: `except Exception:`
Line 469: `except Exception:`
... and 1 more

#### Unused Imports

Line 14: `from src.utils.warning_symbols import error`
Line 170: `from src.config.constants import FULL_TRAINING_LOOKBACK_DAYS`
Line 170: `from src.config.constants import SHORT_BLANK_LOOKBACK_DAYS`

#### Todo Comments

Line 486: `debug_training_step,`
Line 535: `@debug_training_step(`
Line 537: `save_debug_artifacts=True,`

### src/training/steps/step12_final_parameters_optimization/evaluation_engine.py

#### Broad Exceptions

Line 148: `except Exception:`
Line 243: `except Exception:`
Line 353: `except Exception:`
Line 368: `except Exception:`
Line 385: `except Exception:`
Line 399: `except Exception:`
Line 414: `except Exception:`
Line 430: `except Exception:`
Line 451: `except Exception:`
Line 475: `except Exception:`
... and 3 more

### src/training/steps/step12_final_parameters_optimization/optimized_optuna_optimization.py

#### Broad Exceptions

Line 241: `except Exception:`

#### Unused Imports

Line 14: `from sklearn.model_selection import train_test_split`

### src/training/steps/step12_final_parameters_optimization/efficiency_optimizer.py

#### Broad Exceptions

Line 223: `except Exception as e:`
Line 262: `except Exception as e:`
Line 291: `except Exception as e:`
Line 331: `except Exception as e:`
Line 371: `except Exception as e:`
Line 424: `except Exception as e:`
Line 481: `except Exception as e:`
Line 518: `except Exception as e:`
Line 529: `except Exception as e:`
Line 538: `except Exception as e:`
... and 12 more

#### Unused Imports

Line 23: `from src.utils.warning_symbols import error`
Line 23: `from src.utils.warning_symbols import warning`
Line 23: `from src.utils.warning_symbols import critical`
Line 23: `from src.utils.warning_symbols import problem`
Line 23: `from src.utils.warning_symbols import failed`
Line 23: `from src.utils.warning_symbols import invalid`
Line 23: `from src.utils.warning_symbols import missing`
Line 23: `from src.utils.warning_symbols import timeout`
Line 23: `from src.utils.warning_symbols import connection_error`
Line 23: `from src.utils.warning_symbols import validation_error`
... and 2 more

### src/training/steps/step12_final_parameters_optimization/__init__.py

#### Unused Imports

Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import execution_error`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import initialization_error`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import timeout`
... and 6 more

### src/training/steps/step4_analyst_labeling_feature_engineering_components/optimized_triple_barrier_labeling.py

#### Type Ignores

Line 13: `import numba  # type: ignore`
Line 15: `numba = None  # type: ignore`

#### Broad Exceptions

Line 14: `except Exception:  # pragma: no cover`
Line 147: `except Exception:`
Line 158: `except Exception:`
Line 193: `except Exception:`

#### Unused Imports

Line 3: `from datetime import timedelta`
Line 9: `from src.utils.logger import system_logger`

#### Todo Comments

Line 122: `# Debug`

### src/training/steps/step4_analyst_labeling_feature_engineering_components/__init__.py

#### Unused Imports

Line 12: `from src.utils.warning_symbols import connection_error`
Line 12: `from src.utils.warning_symbols import critical`
Line 12: `from src.utils.warning_symbols import error`
Line 12: `from src.utils.warning_symbols import execution_error`
Line 12: `from src.utils.warning_symbols import failed`
Line 12: `from src.utils.warning_symbols import initialization_error`
Line 12: `from src.utils.warning_symbols import invalid`
Line 12: `from src.utils.warning_symbols import missing`
Line 12: `from src.utils.warning_symbols import problem`
Line 12: `from src.utils.warning_symbols import timeout`
... and 3 more

### src/training/steps/multi_timeframe_training/multi_timeframe_training_manager.py

#### Broad Exceptions

Line 124: `except Exception as e:`
Line 170: `except Exception as e:`
Line 218: `except Exception:`
Line 250: `except Exception as e:`
Line 273: `except Exception as e:`
Line 300: `except Exception as e:`
Line 327: `except Exception as e:`
Line 354: `except Exception as e:`
Line 387: `except Exception as e:`
Line 455: `except Exception:`
... and 28 more

#### Unused Imports

Line 18: `from src.utils.warning_symbols import initialization_error`

#### Todo Comments

Line 277: `# Log specific error details for debugging`
Line 304: `# Log specific error details for debugging`
Line 331: `# Log specific error details for debugging`
Line 358: `# Log specific error details for debugging`
Line 391: `# Log specific error details for debugging`
Line 1279: `# Log failed prerequisites for debugging`

### src/training/steps/multi_timeframe_training/__init__.py

#### Unused Imports

Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import execution_error`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import initialization_error`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import timeout`
... and 3 more

### src/training/steps/analyst_training_components/regime_specific_tpsl_optimizer.py

#### Broad Exceptions

Line 244: `except Exception as e:`
Line 266: `except Exception as e:`
Line 286: `except Exception:`
Line 300: `except Exception:`
Line 381: `except Exception as e:`
Line 454: `except Exception:`
Line 526: `except Exception:`
Line 648: `except Exception as e:`

### src/training/steps/analyst_training_components/__init__.py

#### Unused Imports

Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import execution_error`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import initialization_error`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import timeout`
... and 3 more

### src/training/steps/data_preparation_components/aggtrades_data_formatting.py

#### Broad Exceptions

Line 49: `except Exception as e:`
Line 80: `except Exception as e:`
Line 116: `except Exception as e:`
Line 150: `except Exception as e:`
Line 188: `except Exception as e:`
Line 222: `except Exception as e:`
Line 501: `except Exception as e:`

### src/training/steps/data_preparation_components/training_validation_config.py

#### Debug Statements

Line 401: `print(f"🔍 DEBUG: blank_mode detected: {blank_mode}")`
Line 402: `print(f"🔍 DEBUG: Available memory: {memory.available / (1024**3):.1f}GB")`

#### Broad Exceptions

Line 377: `except Exception as e:`

#### Todo Comments

Line 397: `# Debug logging`
Line 399: `f"🔍 DEBUG: BLANK_TRAINING_MODE environment variable: {os.getenv('BLANK_TRAINING_MODE', 'not set')}",`
Line 401: `print(f"🔍 DEBUG: blank_mode detected: {blank_mode}")`
Line 402: `print(f"🔍 DEBUG: Available memory: {memory.available / (1024**3):.1f}GB")`
Line 410: `f"🔍 DEBUG: Using blank mode requirements: {min_memory_gb}GB RAM, {min_disk_gb}GB disk, {min_cpu_cores} CPU cores",`
Line 418: `f"🔍 DEBUG: Using production requirements: {min_memory_gb}GB RAM, {min_disk_gb}GB disk, {min_cpu_cores} CPU cores",`

### src/training/steps/data_preparation_components/__init__.py

#### Unused Imports

Line 9: `from src.utils.warning_symbols import connection_error`
Line 9: `from src.utils.warning_symbols import critical`
Line 9: `from src.utils.warning_symbols import error`
Line 9: `from src.utils.warning_symbols import execution_error`
Line 9: `from src.utils.warning_symbols import failed`
Line 9: `from src.utils.warning_symbols import initialization_error`
Line 9: `from src.utils.warning_symbols import invalid`
Line 9: `from src.utils.warning_symbols import missing`
Line 9: `from src.utils.warning_symbols import problem`
Line 9: `from src.utils.warning_symbols import timeout`
... and 4 more

### src/training/examples/optimized_training_example.py

#### Broad Exceptions

Line 160: `except Exception:`

#### Unused Imports

Line 32: `from src.training.memory_profiler import profile_memory_usage`

### src/training/core/stage_registry.py

#### Broad Exceptions

Line 93: `except Exception:`
Line 123: `except Exception:`
Line 164: `except Exception:`
Line 194: `except Exception:`
Line 215: `except Exception:`
Line 238: `except Exception:`
Line 259: `except Exception:`
Line 280: `except Exception:`
Line 338: `except Exception:`
Line 377: `except Exception:`
... and 25 more

### src/training/core/checkpoint_manager.py

#### Broad Exceptions

Line 104: `except Exception:`
Line 138: `except Exception:`
Line 179: `except Exception:`
Line 209: `except Exception:`
Line 232: `except Exception:`
Line 255: `except Exception:`
Line 278: `except Exception:`
Line 301: `except Exception:`
Line 365: `except Exception:`
Line 406: `except Exception:`
... and 25 more

### src/training/core/stage_context.py

#### Broad Exceptions

Line 88: `except Exception:`
Line 120: `except Exception:`
Line 161: `except Exception:`
Line 191: `except Exception:`
Line 212: `except Exception:`
Line 235: `except Exception:`
Line 256: `except Exception:`
Line 279: `except Exception:`
Line 343: `except Exception:`
Line 382: `except Exception:`
... and 25 more

### src/training/core/pipeline_base.py

#### Broad Exceptions

Line 130: `except Exception:`
Line 158: `except Exception:`
Line 199: `except Exception:`
Line 229: `except Exception:`
Line 250: `except Exception:`
Line 271: `except Exception:`
Line 292: `except Exception:`
Line 313: `except Exception:`
Line 369: `except Exception:`
Line 408: `except Exception:`
... and 25 more

### src/training/core/pipeline_orchestrator.py

#### Broad Exceptions

Line 106: `except Exception as e:`
Line 140: `except Exception:`
Line 181: `except Exception:`
Line 211: `except Exception:`
Line 232: `except Exception:`
Line 255: `except Exception:`
Line 278: `except Exception:`
Line 301: `except Exception:`
Line 365: `except Exception:`
Line 406: `except Exception:`
... and 25 more

### src/training/core/__init__.py

#### Unused Imports

Line 8: `from src.utils.warning_symbols import connection_error`
Line 8: `from src.utils.warning_symbols import critical`
Line 8: `from src.utils.warning_symbols import error`
Line 8: `from src.utils.warning_symbols import execution_error`
Line 8: `from src.utils.warning_symbols import failed`
Line 8: `from src.utils.warning_symbols import initialization_error`
Line 8: `from src.utils.warning_symbols import invalid`
Line 8: `from src.utils.warning_symbols import missing`
Line 8: `from src.utils.warning_symbols import problem`
Line 8: `from src.utils.warning_symbols import timeout`
... and 7 more

### src/exchange/binance.py

#### Broad Exceptions

Line 172: `except Exception:`
Line 197: `except Exception:`
Line 231: `except Exception:`
Line 278: `except Exception:`
Line 335: `except Exception:`
Line 444: `except Exception:`
Line 528: `except Exception:`
Line 536: `except Exception:`
Line 546: `except Exception:`
Line 554: `except Exception:`
... and 8 more

#### Todo Comments

Line 543: `# TODO: Implement Binance user data stream listenKey + ws connect`

### src/core/di_integration.py

#### Broad Exceptions

Line 90: `except Exception as e:`
Line 254: `except Exception as e:`
Line 286: `except Exception as e:`
Line 312: `except Exception as e:`
Line 347: `except Exception as e:`

#### Unused Imports

Line 27: `from src.utils.warning_symbols import failed`

#### Todo Comments

Line 217: `self.logger.debug(f"Initialized {component_name}")`

### src/core/injectable_base.py

#### Type Ignores

Line 49: `self.print = _shim_print  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 43: `except Exception as e:`

#### Todo Comments

Line 54: `self.logger.debug(f"Component {self.__class__.__name__} configured")`

### src/core/di_launcher.py

#### Broad Exceptions

Line 75: `except Exception as e:`
Line 109: `except Exception as e:`
Line 142: `except Exception as e:`
Line 170: `except Exception as e:`
Line 268: `except Exception:`
Line 288: `except Exception as e:`
Line 311: `except Exception as e:`
Line 336: `except Exception as e:`

#### Todo Comments

Line 284: `self.logger.debug(f"Started {component_name}")`
Line 307: `self.logger.debug(f"Initialized {component_name}")`
Line 331: `self.logger.debug(f"Stopped {component_name}")`

### src/core/enhanced_factories.py

#### Broad Exceptions

Line 81: `except Exception:`
Line 121: `except Exception:`
Line 156: `except Exception:`
Line 199: `except Exception:`
Line 242: `except Exception:`
Line 317: `except Exception:`

### src/core/generic_base.py

#### Broad Exceptions

Line 131: `except Exception:`

### src/core/service_registry.py

#### Unused Imports

Line 142: `from exchange.factory import ExchangeFactory`

### src/core/dependency_injection.py

#### Broad Exceptions

Line 214: `except Exception as e:`
Line 256: `except Exception as e:`
Line 279: `except Exception as e:`
Line 293: `except Exception as e:`
Line 374: `except Exception as e:`
Line 426: `except Exception as e:`

#### Unused Imports

Line 21: `from src.utils.warning_symbols import failed`

#### Todo Comments

Line 96: `self.logger.debug(`
Line 121: `self.logger.debug(`
Line 137: `self.logger.debug(`
Line 146: `self.logger.debug(f"Entered scope: {scope_id}")`
Line 154: `self.logger.debug(f"Exited scope: {scope_id}")`
Line 163: `self.logger.debug(f"Set config: {key} = {value}")`

### src/core/config_service.py

#### Broad Exceptions

Line 25: `except Exception:`
Line 120: `except Exception:`
Line 200: `except Exception:`
Line 208: `except Exception:`
Line 257: `except Exception as e:`
Line 302: `except Exception:`
Line 335: `except Exception:`
Line 398: `except Exception:`
Line 420: `except Exception:`
Line 442: `except Exception:`
... and 11 more

#### Todo Comments

Line 414: `# arg_config["debug"] = True  # Example`

### src/core/__init__.py

#### Unused Imports

Line 3: `from dependency_injection import ComponentFactory`
Line 3: `from dependency_injection import DependencyContainer`
Line 3: `from dependency_injection import ModularTradingSystem`
Line 3: `from dependency_injection import ServiceRegistration`

### src/config/environment.py

#### Broad Exceptions

Line 165: `except Exception as e:`

### src/config/label_model_mapping.py

#### Type Ignores

Line 225: `import xgboost as xgb  # type: ignore`
Line 239: `import lightgbm as lgb  # type: ignore`
Line 254: `from catboost import CatBoostClassifier  # type: ignore`
Line 307: `from hmmlearn.hmm import GaussianHMM  # type: ignore`

#### Broad Exceptions

Line 347: `except Exception:`
Line 350: `except Exception:`

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/config/multi_timeframe_hmm_ensemble_config.py

#### Broad Exceptions

Line 224: `except Exception:`

### src/config/enhanced_reporting_config.py

#### Broad Exceptions

Line 417: `except Exception as e:`

### src/config/validation.py

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/config/__init__.py

#### Unused Imports

Line 248: `from src.config.trading import get_position_sizing_config`
Line 261: `from src.config.trading import get_position_closing_config`
Line 274: `from src.config.trading import get_position_monitoring_config`
Line 283: `from src.config.training import get_enhanced_training_config`

### src/utils/lookahead_bias_detector_example.py

#### Debug Statements

Line 96: `print("🔍 Enhanced Lookahead Bias Detector Demonstration")`
Line 148: `print(f"\n🔍 Suspicious Features: {len(results['suspicious_features'])}")`
Line 201: `print(f"\n🔍 Enhanced detector analysis:")`

#### Unused Imports

Line 13: `from datetime import datetime`
Line 13: `from datetime import timedelta`

### src/utils/observability.py

#### Broad Exceptions

Line 42: `except Exception:  # pragma: no cover`
Line 68: `except Exception:  # pragma: no cover`

#### Unused Imports

Line 1: `from __future__ import annotations`
Line 54: `from opentelemetry import _logs`

### src/utils/step_dependency_validator.py

#### Broad Exceptions

Line 129: `except Exception as e:`
Line 227: `except Exception as e:`
Line 283: `except Exception as e:`
Line 289: `except Exception as e:`
Line 334: `except Exception as e:`
Line 364: `except Exception as e:`

#### Unused Imports

Line 6: `import asyncio`
Line 7: `from typing import Optional`
Line 12: `from src.utils.warning_symbols import error`
Line 12: `from src.utils.warning_symbols import warning`
Line 12: `from src.utils.warning_symbols import critical`

#### Todo Comments

Line 168: `self.logger.debug(f"✅ {step_name} completed successfully")`
Line 216: `self.logger.debug(f"✅ {step_name} completed successfully (from centralized progress)")`
Line 219: `self.logger.debug(f"✅ {step_name} was skipped (from centralized progress)")`
Line 222: `self.logger.debug(f"⚠️ {step_name} status: {step_status.get('status', 'unknown')} (from centralized progress)")`
Line 224: `self.logger.debug(f"⚠️ No status mapping found for {step_name} in centralized progress")`
Line 272: `self.logger.debug(f"✅ {step_name} completed successfully (from centralized progress)")`
Line 275: `self.logger.debug(f"✅ {step_name} was skipped (from centralized progress)")`
Line 278: `self.logger.debug(f"⚠️ {step_name} status: {step_status.get('status', 'unknown')} (from centralized progress)")`
Line 280: `self.logger.debug(f"⚠️ No status mapping found for {step_name} in centralized progress")`

### src/utils/data_quality_validator.py

#### Todo Comments

Line 277: `# For wavelet features, only log as debug info`
Line 279: `self.logger.debug(`

### src/utils/state_manager.py

#### Broad Exceptions

Line 139: `except Exception:`
Line 273: `except Exception:`
Line 296: `except Exception:`
Line 320: `except Exception:`
Line 362: `except Exception:`
Line 410: `except Exception as e:`

### src/utils/structured_logging.py

#### Type Ignores

Line 14: `from pythonjsonlogger import jsonlogger  # type: ignore`
Line 16: `jsonlogger = None  # type: ignore`
Line 69: `def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - filter is required API`
Line 104: `class CorrelationIdMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]`
Line 111: `async def dispatch(self, request: Request, call_next):  # type: ignore[override]`

#### Broad Exceptions

Line 15: `except Exception:  # pragma: no cover - optional dependency`
Line 73: `except Exception:`
Line 123: `except Exception:`

#### Unused Imports

Line 1: `from __future__ import annotations`

### src/utils/async_utils.py

#### Broad Exceptions

Line 85: `except Exception:`
Line 111: `except Exception:`
Line 140: `except Exception:`
Line 181: `except Exception:`
Line 222: `except Exception:`
Line 249: `except Exception:`
Line 283: `except Exception:`
Line 311: `except Exception:`
Line 326: `except Exception:`
Line 358: `except Exception:`
... and 12 more

#### Todo Comments

Line 305: `self.logger.debug(f"Removed {oldest_key} from cache")`
Line 309: `self.logger.debug(f"Added {file_path} to cache")`

### src/utils/enhanced_data_quality_decorators.py

#### Broad Exceptions

Line 110: `except Exception:`
Line 288: `except Exception as e:`

#### Unused Imports

Line 9: `import inspect`
Line 10: `from typing import List`
Line 10: `from typing import Callable`
Line 10: `from typing import Union`
Line 10: `from typing import Tuple`
Line 11: `from datetime import datetime`
Line 11: `from datetime import timedelta`
Line 15: `import asyncio`
Line 17: `import warnings`
Line 22: `from src.utils.warning_symbols import error`
... and 6 more

#### Todo Comments

Line 124: `self.logger.debug(f"❌ [CACHE] Cache miss for {method_name}")`
Line 144: `self.logger.debug(f"🗑️ [CACHE] Clearing data quality cache")`
Line 147: `self.logger.debug(f"✅ [CACHE] Cache cleared ({cache_size} entries removed)")`

### src/utils/time_utils.py

#### Broad Exceptions

Line 32: `except Exception:`
Line 40: `except Exception:`
Line 63: `except Exception:`

#### Unused Imports

Line 1: `from __future__ import annotations`

### src/utils/data_preprocessing.py

#### Broad Exceptions

Line 98: `except Exception as e:`
Line 149: `except Exception as e:`
Line 193: `except Exception as e:`

### src/utils/feature_output_validator.py

#### Debug Statements

Line 131: `print(f"🔍 [FEATURE OUTPUT VALIDATION] Starting validation for {method_name}")`

#### Broad Exceptions

Line 476: `except Exception as e:`
Line 606: `except Exception as e:`

#### Unused Imports

Line 16: `from src.utils.warning_symbols import error`
Line 16: `from src.utils.warning_symbols import warning`

### src/utils/intelligent_feature_cache.py

#### Broad Exceptions

Line 206: `except Exception as e:`
Line 234: `except Exception as e:`
Line 335: `except Exception as e:`

#### Unused Imports

Line 12: `import os`
Line 15: `from typing import Dict`
Line 15: `from typing import List`
Line 406: `import inspect`

#### Todo Comments

Line 204: `logger.debug(f"💾 Saved to disk cache: {cache_key}")`
Line 231: `logger.debug(f"📂 Loaded from disk cache: {cache_key}")`

### src/utils/parallel_processing_optimizer.py

#### Broad Exceptions

Line 76: `except:`

#### Unused Imports

Line 15: `from typing import Tuple`

#### Todo Comments

Line 120: `logger.debug(f"📦 Split DataFrame into {len(chunks)} chunks of ~{chunk_size} rows each")`
Line 139: `logger.debug(f"🔗 Merged {len(chunks)} chunks into DataFrame with {len(merged_df)} rows")`
Line 161: `logger.debug("📊 Dataset too small for parallel processing, using sequential")`
Line 302: `"""Log system information for debugging."""`
Line 344: `logger.debug(f"⏭️ Skipping parallel processing for async function: {func.__name__}")`

### src/utils/decorators.py

#### Type Ignores

Line 41: `from pydantic import validate_call as _pydantic_validate_call  # type: ignore`
Line 43: `_pydantic_validate_call = None  # type: ignore`
Line 46: `from beartype import beartype as _beartype  # type: ignore`
Line 48: `_beartype = None  # type: ignore`
Line 51: `from typeguard import typechecked as _typechecked  # type: ignore`
Line 53: `_typechecked = None  # type: ignore`
Line 56: `import pandera as pa  # type: ignore`
Line 58: `pa = None  # type: ignore`
Line 73: `def decorator(func: F) -> F:  # type: ignore[override]`
Line 93: `def decorator(func: F) -> F:  # type: ignore[override]`
... and 20 more

#### Broad Exceptions

Line 42: `except Exception:  # pragma: no cover`
Line 47: `except Exception:  # pragma: no cover`
Line 52: `except Exception:  # pragma: no cover`
Line 57: `except Exception:  # pragma: no cover`
Line 351: `except Exception:`
Line 431: `except Exception:`
Line 517: `except Exception:  # pragma: no cover`
Line 524: `except Exception:  # pragma: no cover`
Line 628: `except Exception:`
Line 685: `except Exception:`
... and 1 more

#### Unused Imports

Line 12: `from __future__ import annotations`
Line 41: `from pydantic import validate_call`
Line 46: `from beartype import beartype`
Line 51: `from typeguard import typechecked`

#### Todo Comments

Line 614: `Masks values of known sensitive keys. Keeps structure to aid debugging.`

### src/utils/data_loader.py

#### Broad Exceptions

Line 107: `except Exception as e:`
Line 161: `except Exception as e:`
Line 193: `except Exception as e:`
Line 227: `except Exception as e:`
Line 241: `except Exception as e:`
Line 276: `except Exception as e:`
Line 301: `except Exception as e:`
Line 333: `except:`
Line 341: `except Exception as e:`
Line 350: `except Exception as e:`

#### Unused Imports

Line 11: `from typing import Union`

### src/utils/training_pipeline_decorators.py

#### Debug Statements

Line 711: `print(f"🔍 [PIPELINE STEP] Pre-execution checks for {step_name}")`
Line 724: `print(f"🔍 [PIPELINE STEP] Running data quality validation for {step_name}")`
Line 778: `print(f"🔍 [PIPELINE STEP] Post-execution checks for {step_name}")`
Line 914: `print(f"🔍 [PIPELINE INPUT] Validating input for {method_name}")`
Line 1035: `print(f"🔍 [PIPELINE INPUT] Validating input data for {method_name}")`

#### Broad Exceptions

Line 262: `except Exception:`
Line 265: `except Exception:`
Line 276: `except Exception:`
Line 300: `except Exception:`
Line 329: `except Exception:`
Line 337: `except Exception:`
Line 368: `except Exception:`
Line 395: `except Exception:`
Line 755: `except Exception as e:`
Line 807: `except Exception as e:`
... and 2 more

#### Unused Imports

Line 9: `import traceback`
Line 10: `from typing import Union`
Line 11: `from datetime import timedelta`
Line 20: `from src.utils.warning_symbols import warning`
Line 20: `from src.utils.warning_symbols import critical`

#### Todo Comments

Line 151: `def debug_training_step(`
Line 153: `save_debug_artifacts: bool = True,`
Line 157: `"""Decorator for debugging training steps."""`

### src/utils/optimization_integration_test.py

#### Unused Imports

Line 12: `import time`
Line 14: `import pytest`
Line 19: `from src.utils.training_pipeline_decorators import validate_step_output`
Line 30: `from src.utils.data_quality_decorators import validate_wavelet_data_quality`
Line 30: `from src.utils.data_quality_decorators import validate_microstructure_data_quality`
Line 30: `from src.utils.data_quality_decorators import validate_multi_timeframe_data_quality`
Line 30: `from src.utils.data_quality_decorators import validate_klines_data_quality`
Line 30: `from src.utils.data_quality_decorators import validate_feature_engineering_with_lookahead_bias_detection`
Line 30: `from src.utils.data_quality_decorators import ValidationLevel`

#### Todo Comments

Line 25: `debug_training_step,`
Line 90: `@debug_training_step`

### src/utils/config_loader.py

#### Broad Exceptions

Line 50: `except Exception:`
Line 167: `except Exception:`
Line 195: `except Exception:`
Line 261: `except Exception:`

### src/utils/error_handler.py

#### Broad Exceptions

Line 100: `except Exception as e:`
Line 163: `except Exception:`
Line 200: `except Exception:`
Line 306: `except Exception as e:`
Line 437: `except Exception as e:`
Line 465: `except Exception as e:`
Line 500: `except Exception as e:`
Line 626: `except Exception as e:`
Line 639: `except Exception as e:`
Line 800: `except Exception as e:`
... and 21 more

#### Todo Comments

Line 88: `logger.debug(`

### src/utils/data_validation.py

#### Broad Exceptions

Line 57: `except Exception as e:`
Line 109: `except Exception as e:`
Line 190: `except Exception as e:`
Line 254: `except Exception as e:`

### src/utils/parquet_utils.py

#### Broad Exceptions

Line 63: `except Exception as e:`
Line 66: `except Exception as e:`
Line 102: `except Exception as e:`
Line 115: `except Exception as e:`
Line 130: `except Exception as e:`
Line 168: `except Exception as e:`

#### Unused Imports

Line 5: `import logging`

### src/utils/comprehensive_logger.py

#### Broad Exceptions

Line 217: `except Exception:`
Line 267: `except Exception:`
Line 278: `except Exception:`
Line 289: `except Exception:`
Line 370: `except Exception:`

#### Unused Imports

Line 10: `import logging.handlers`

#### Todo Comments

Line 202: `if root_logger.level > logging.DEBUG:`
Line 203: `root_logger.setLevel(logging.DEBUG)`
Line 298: `level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)`

### src/utils/validator_orchestrator.py

#### Broad Exceptions

Line 67: `except Exception:`
Line 96: `except Exception:`
Line 125: `except Exception as e:`
Line 147: `except Exception:`
Line 270: `except Exception:`

#### Unused Imports

Line 5: `import asyncio`
Line 20: `from src.utils.warning_symbols import error`

#### Todo Comments

Line 56: `# Debug-level context for troubleshooting`
Line 58: `self.logger.debug(`
Line 98: `self.logger.debug(`
Line 127: `# Log full stack trace for debugging`
Line 148: `self.logger.debug(`

### src/utils/hmm_composite_manager.py

#### Broad Exceptions

Line 152: `except Exception as e:`
Line 223: `except Exception as e:`
Line 275: `except Exception as e:`
Line 328: `except Exception as e:`
Line 382: `except Exception as e:`
Line 464: `except Exception as e:`
Line 546: `except Exception as e:`
Line 606: `except Exception as e:`

#### Unused Imports

Line 14: `import asyncio`
Line 17: `from typing import List`
Line 20: `from pathlib import Path`
Line 436: `from src.training.steps.step3_hmm_regime_discovery import run_step_enhanced`

#### Todo Comments

Line 99: `self.logger.debug(`

### src/utils/signal_handler.py

#### Broad Exceptions

Line 100: `except Exception:`
Line 127: `except Exception:`
Line 151: `except Exception:`
Line 180: `except Exception:`
Line 204: `except Exception:`
Line 228: `except Exception:`
Line 252: `except Exception:`
Line 293: `except Exception:`
Line 305: `except Exception:`
Line 337: `except Exception:`
... and 8 more

### src/utils/data_optimizer.py

#### Broad Exceptions

Line 91: `except Exception:`
Line 102: `except Exception:`
Line 140: `except Exception:`
Line 162: `except Exception:`
Line 189: `except Exception:`
Line 226: `except Exception:`
Line 237: `except Exception:`
Line 260: `except Exception:`
Line 275: `except Exception:`
Line 295: `except Exception:`
... and 10 more

#### Unused Imports

Line 11: `from typing import Optional`

### src/utils/enhanced_data_quality_validator.py

#### Broad Exceptions

Line 261: `except:`
Line 265: `except:`

#### Unused Imports

Line 12: `import logging`

### src/utils/data_quality_decorators.py

#### Broad Exceptions

Line 54: `except Exception:`
Line 108: `except Exception as e:`
Line 114: `except:`
Line 202: `except Exception as e:`
Line 448: `except Exception as e:`
Line 638: `except Exception as e:`
Line 779: `except Exception as e:`
Line 831: `except Exception as e:`

#### Unused Imports

Line 9: `import inspect`
Line 10: `from typing import List`
Line 10: `from typing import Union`
Line 11: `from datetime import datetime`
Line 11: `from datetime import timedelta`
Line 18: `from src.utils.warning_symbols import error`
Line 20: `from src.utils.feature_output_validator import validate_feature_output`
Line 21: `from src.utils.lookahead_bias_detector import detect_lookahead_bias`
Line 21: `from src.utils.lookahead_bias_detector import apply_feature_lagging`

#### Todo Comments

Line 68: `self.logger.debug(f"❌ [CACHE] Cache miss for {method_name}")`
Line 90: `self.logger.debug(f"🗑️ [CACHE] Clearing data quality cache")`
Line 93: `self.logger.debug(f"✅ [CACHE] Cache cleared ({cache_size} entries removed)")`
Line 135: `logger.debug(f"🔍 [DATA EXTRACTION] Extracting data for {method_name}")`
Line 153: `logger.debug(f"   📊 Found {param} in kwargs")`
Line 157: `logger.debug(f"   ✅ Valid DataFrame found: {param} with shape {data.shape}")`
Line 160: `logger.debug(f"   ✅ DataFrame has datetime index: {data.index.min()} to {data.index.max()}")`
Line 162: `logger.debug(f"   ⚠️ DataFrame does not have datetime index")`
Line 165: `logger.debug(f"   ⚠️ {param} found but DataFrame is empty")`
Line 167: `logger.debug(f"   ⚠️ {param} found but not a DataFrame: type={type(data)}")`
... and 28 more

### src/utils/mlflow_utils.py

#### Broad Exceptions

Line 36: `except Exception as e:`
Line 76: `except Exception as e:`
Line 106: `except Exception as e:`

#### Unused Imports

Line 10: `from src.utils.warning_symbols import failed`

### src/utils/base_validator.py

#### Broad Exceptions

Line 92: `except Exception as e:`
Line 128: `except Exception as e:`
Line 194: `except Exception as e:`
Line 327: `except Exception as e:`
Line 369: `except Exception as e:`

#### Unused Imports

Line 10: `from src.utils.warning_symbols import warning`

### src/utils/confidence.py

#### Broad Exceptions

Line 31: `except Exception:`

### src/utils/prometheus_metrics.py

#### Type Ignores

Line 18: `Counter = Gauge = Histogram = None  # type: ignore[assignment]`
Line 19: `generate_latest = None  # type: ignore[assignment]`
Line 20: `start_http_server = None  # type: ignore[assignment]`
Line 262: `return generate_latest()  # type: ignore[return-value]`

#### Broad Exceptions

Line 17: `except Exception as e:  # pragma: no cover - optional dependency fallback`
Line 156: `except Exception:`
Line 163: `except Exception as e:`

### src/utils/model_manager.py

#### Type Ignores

Line 32: `_NP_ORIGINAL_BITGEN_CTOR = None  # type: ignore[var-annotated]`
Line 35: `def _normalized_numpy_bitgen_ctor(bit_generator_name, state=None, *args, **kwargs):  # type: ignore[override]`
Line 49: `return _NP_ORIGINAL_BITGEN_CTOR(name_candidate, effective_state)  # type: ignore[misc]`
Line 52: `return _NP_ORIGINAL_BITGEN_CTOR(name_candidate)  # type: ignore[misc]`
Line 53: `except Exception as ctor_exc:  # noqa: BLE001`
Line 60: `import numpy.random._mt19937 as _mt  # type: ignore[attr-defined]`
Line 78: `import numpy.random._pickle as np_random_pickle  # type: ignore[attr-defined]`
Line 86: `np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor  # type: ignore[attr-defined]`
Line 90: `except Exception as _shim_exc:  # noqa: BLE001`

#### Broad Exceptions

Line 44: `except Exception:`
Line 63: `except Exception:`
Line 67: `except Exception:`
Line 101: `except Exception:`
Line 173: `except Exception:`
Line 204: `except Exception:`
Line 238: `except Exception:`
Line 264: `except Exception:`
Line 313: `except Exception:`
Line 386: `except Exception:`
... and 8 more

### src/utils/domain_errors.py

#### Unused Imports

Line 5: `from __future__ import annotations`

### src/utils/enhanced_error_handler.py

#### Broad Exceptions

Line 530: `except Exception as e:`

#### Unused Imports

Line 20: `from functools import wraps`
Line 21: `from typing import Union`

### src/utils/logger.py

#### Type Ignores

Line 38: `def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]`
Line 250: `def handleError(self, record: logging.LogRecord) -> None:  # type: ignore[override]`
Line 825: `.astype(str)  # type: ignore[operator]`
Line 826: `.value_counts()  # type: ignore[attr-defined]`
Line 827: `.to_dict()  # type: ignore[attr-defined]`
Line 839: `df[columns_list[:10]].isnull().sum().to_dict()  # type: ignore[index]`
Line 852: `preview = sample.to_dict(orient="records")  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 47: `except Exception:`
Line 73: `except Exception:`
Line 79: `except Exception:`
Line 84: `except Exception:`
Line 150: `except Exception:`
Line 186: `except Exception as e:`
Line 221: `except Exception as e:`
Line 259: `except Exception:`
Line 263: `except Exception:`
Line 267: `except Exception:`
... and 25 more

#### Unused Imports

Line 9: `import logging.handlers`
Line 520: `import concurrent.futures`

#### Todo Comments

Line 82: `# Also set TF CPP log level to suppress INFO/DEBUG C++ logs`
Line 198: `valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]`
Line 409: `level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)`
Line 418: `valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]`
Line 587: `# Temporarily set logging to INFO level for debugging`
Line 853: `logger.debug(f"🔎 {df_name} sample: {preview}")`

### src/utils/lookahead_bias_detector.py

#### Broad Exceptions

Line 108: `except Exception as e:`
Line 129: `except Exception:`
Line 181: `except Exception as e:`
Line 564: `except Exception as e:`
Line 682: `except Exception as e:`

#### Unused Imports

Line 10: `from typing import List`
Line 10: `from typing import Tuple`
Line 11: `from datetime import datetime`
Line 11: `from datetime import timedelta`
Line 12: `import warnings`
Line 14: `from src.utils.warning_symbols import warning`
Line 14: `from src.utils.warning_symbols import error`
Line 14: `from src.utils.warning_symbols import critical`

### src/utils/data_type_optimizer.py

#### Broad Exceptions

Line 64: `except:`
Line 175: `except:`
Line 184: `except Exception as e:`

#### Unused Imports

Line 10: `from typing import Any`
Line 10: `from typing import List`
Line 10: `from typing import Optional`

#### Todo Comments

Line 185: `logger.debug(f"Could not optimize {col} to {dtype}: {e}")`

### src/strategist/volatility_targeting_strategy.py

#### Todo Comments

Line 97: `self.logger.debug(f"Calculated position multiplier: {multiplier:.3f}")`

### src/strategist/strategist.py

#### Type Ignores

Line 56: `self.print = _shim_print  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 53: `except Exception:`
Line 198: `except Exception:`
Line 241: `except Exception:`
Line 331: `except Exception:`
Line 420: `except Exception as e:`
Line 553: `except Exception:`
Line 622: `except Exception:`
Line 672: `except Exception:`
Line 751: `except Exception:`
Line 810: `except Exception:`
... and 9 more

#### Unused Imports

Line 12: `from src.utils.warning_symbols import invalid`

### src/validation/critical_path_validators.py

#### Broad Exceptions

Line 69: `except Exception as e:`
Line 128: `except Exception as e:`
Line 171: `except Exception as e:`
Line 204: `except Exception as e:`
Line 349: `except Exception:`

### src/analyst/order_book_analyzer.py

#### Broad Exceptions

Line 37: `except Exception as e:`
Line 93: `except Exception as e:`
Line 126: `except Exception as e:`
Line 168: `except Exception as e:`

### src/analyst/advanced_feature_engineering.py

#### Broad Exceptions

Line 68: `except Exception as e:`
Line 123: `except Exception as e:`
Line 150: `except Exception as e:`
Line 679: `except Exception as e:`
Line 820: `except Exception as e:`
Line 879: `except Exception as e:`
Line 916: `except Exception as e:`
Line 965: `except Exception as e:`
Line 1036: `except Exception as e:`
Line 1105: `except Exception as e:`
... and 46 more

#### Unused Imports

Line 17: `from src.utils.warning_symbols import critical`
Line 17: `from src.utils.warning_symbols import problem`
Line 17: `from src.utils.warning_symbols import failed`
Line 17: `from src.utils.warning_symbols import invalid`
Line 17: `from src.utils.warning_symbols import missing`
Line 17: `from src.utils.warning_symbols import timeout`
Line 17: `from src.utils.warning_symbols import connection_error`
Line 17: `from src.utils.warning_symbols import validation_error`
Line 17: `from src.utils.warning_symbols import execution_error`
Line 2339: `import os`
... and 1 more

### src/analyst/live_regime_calculations.py

#### Unused Imports

Line 3: `from __future__ import annotations`

### src/analyst/transition_regime_handler.py

#### Broad Exceptions

Line 174: `except Exception as e:`
Line 196: `except Exception as e:`

#### Unused Imports

Line 10: `from typing import List`

### src/analyst/multi_timeframe_regime_integration.py

#### Broad Exceptions

Line 169: `except Exception as e:`
Line 231: `except Exception:`
Line 271: `except Exception as e:`
Line 371: `except Exception as e:`
Line 578: `except Exception as e:`
Line 603: `except Exception:`

#### Unused Imports

Line 15: `import os`
Line 35: `from src.utils.warning_symbols import initialization_error`
Line 35: `from src.utils.warning_symbols import invalid`

### src/analyst/feature_engineering_orchestrator.py

#### Broad Exceptions

Line 182: `except Exception:`
Line 228: `except Exception:`
Line 265: `except Exception:`
Line 306: `except Exception:`
Line 366: `except Exception:`
Line 401: `except Exception:`
Line 438: `except Exception as e:`
Line 478: `except Exception as e:`
Line 517: `except Exception:`
Line 544: `except Exception:`
... and 6 more

### src/analyst/meta_labeling_system.py

#### Broad Exceptions

Line 248: `except Exception as e:`
Line 254: `except Exception as e:`
Line 296: `except Exception as e:`
Line 344: `except Exception as e:`
Line 408: `except Exception as e:`
Line 414: `except Exception as e:`
Line 420: `except Exception as e:`
Line 426: `except Exception as e:`
Line 432: `except Exception as e:`
Line 438: `except Exception as e:`
... and 58 more

#### Unused Imports

Line 18: `from src.utils.warning_symbols import error`
Line 18: `from src.utils.warning_symbols import warning`
Line 18: `from src.utils.warning_symbols import critical`
Line 18: `from src.utils.warning_symbols import problem`
Line 18: `from src.utils.warning_symbols import failed`
Line 18: `from src.utils.warning_symbols import invalid`
Line 18: `from src.utils.warning_symbols import missing`
Line 18: `from src.utils.warning_symbols import timeout`
Line 18: `from src.utils.warning_symbols import connection_error`
Line 18: `from src.utils.warning_symbols import validation_error`
... and 2 more

#### Todo Comments

Line 155: `# Debug/performance logging controls`
Line 156: `self.debug_logging: bool = bool(self.labeling_config.get("debug_logging", True))`
Line 2449: `if self.debug_logging:`
Line 2472: `if self.debug_logging:`
Line 2492: `if self.debug_logging:`
Line 2511: `if self.debug_logging:`
Line 2661: `if self.debug_logging:`

### src/analyst/autoencoder_feature_generator.py

#### Type Ignores

Line 22: `from tensorflow.keras import Model, layers, regularizers  # type: ignore[import-not-found]`
Line 27: `)  # type: ignore[import-not-found]`
Line 32: `tf = None  # type: ignore`
Line 33: `TFKerasPruningCallback = object  # type: ignore`
Line 35: `object  # type: ignore`
Line 2154: `def on_train_begin(self, logs=None):  # type: ignore[override]`
Line 2157: `def on_train_end(self, logs=None):  # type: ignore[override]`
Line 2160: `def on_epoch_begin(self, epoch, logs=None):  # type: ignore[override]`
Line 2163: `def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]`

#### Broad Exceptions

Line 48: `except Exception:`
Line 109: `except Exception:`
Line 112: `except Exception:`
Line 187: `except Exception:`
Line 287: `except Exception as e:`
Line 350: `except Exception:`
Line 863: `except Exception as e:`
Line 1045: `except Exception as e:`
Line 1315: `except Exception as e:`
Line 1703: `except Exception:`
... and 15 more

#### Unused Imports

Line 55: `from src.utils.warning_symbols import error`
Line 17: `import shap`
Line 45: `from src.utils.warning_symbols import missing`
Line 1421: `import threading`

#### Todo Comments

Line 518: `# Safety check: if we have very few engineered features, log all available columns for debugging`

### src/analyst/predictive_ensembles.py

#### Broad Exceptions

Line 104: `except Exception:`
Line 136: `except Exception:`
Line 178: `except Exception:`
Line 212: `except Exception:`
Line 233: `except Exception:`
Line 254: `except Exception:`
Line 275: `except Exception:`
Line 298: `except Exception:`
Line 319: `except Exception:`
Line 382: `except Exception:`
... and 31 more

### src/analyst/regime_runtime.py

#### Broad Exceptions

Line 19: `except Exception as e:`
Line 181: `except Exception:`
Line 213: `except Exception as e:`
Line 231: `except Exception as e:`
Line 235: `except Exception as e:`

#### Unused Imports

Line 4: `import json`
Line 5: `from typing import Tuple`

### src/analyst/liquidation_risk_model.py

#### Broad Exceptions

Line 105: `except Exception as e:`
Line 150: `except Exception as e:`
Line 228: `except Exception:`
Line 278: `except Exception as e:`
Line 317: `except Exception as e:`
Line 342: `except Exception as e:`
Line 365: `except Exception as e:`
Line 393: `except Exception as e:`
Line 430: `except Exception as e:`
Line 460: `except Exception as e:`
... and 1 more

### src/analyst/multi_timeframe_feature_engineering.py

#### Broad Exceptions

Line 377: `except Exception:`
Line 411: `except Exception as e:`
Line 475: `except Exception as e:`
Line 583: `except Exception as e:`
Line 635: `except Exception as e:`
Line 680: `except Exception as e:`
Line 729: `except Exception as e:`
Line 787: `except Exception as e:`
Line 836: `except Exception as e:`
Line 859: `except Exception:`
... and 2 more

### src/analyst/dynamic_regime_mapper.py

#### Broad Exceptions

Line 58: `except Exception as e:`
Line 91: `except Exception as e:`
Line 161: `except Exception as e:`
Line 334: `except Exception as e:`
Line 353: `except Exception as e:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 7: `from typing import Optional`
Line 7: `from typing import Tuple`

### src/analyst/decision_aggregator.py

#### Broad Exceptions

Line 18: `except Exception:`
Line 121: `except Exception:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 5: `import os`

### src/analyst/data_utils.py

#### Broad Exceptions

Line 99: `except Exception:`
Line 131: `except Exception:`
Line 172: `except Exception:`
Line 202: `except Exception:`
Line 225: `except Exception:`
Line 246: `except Exception:`
Line 267: `except Exception:`
Line 290: `except Exception:`
Line 352: `except Exception:`
Line 393: `except Exception:`
... and 29 more

#### Todo Comments

Line 1130: `f"[DEBUG] load_klines_data: type={type(df)}, shape={df.shape}, columns={df.columns.tolist()}",`

### src/analyst/unified_regime_intelligence_runtime.py

#### Broad Exceptions

Line 124: `except Exception as e:`
Line 158: `except Exception as e:`
Line 178: `except Exception as e:`
Line 200: `except Exception as e:`
Line 238: `except Exception as e:`
Line 324: `except Exception as e:`
Line 380: `except Exception as e:`
Line 442: `except Exception as e:`
Line 560: `except Exception as e:`
Line 614: `except Exception as e:`
... and 7 more

#### Unused Imports

Line 7: `import asyncio`
Line 12: `from typing import List`
Line 12: `from typing import Tuple`
Line 13: `import logging`

### src/analyst/regime_expert_orchestrator.py

#### Broad Exceptions

Line 105: `except Exception as e:`
Line 154: `except Exception as e:`
Line 193: `except Exception as e:`
Line 238: `except Exception as e:`
Line 304: `except Exception as e:`
Line 365: `except Exception as e:`
Line 411: `except Exception as e:`
Line 484: `except Exception as e:`
Line 555: `except Exception as e:`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 6: `import os`

### src/analyst/meta_label_relevance.py

#### Type Ignores

Line 86: `import shap  # type: ignore`
Line 87: `from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore`

#### Unused Imports

Line 3: `from __future__ import annotations`
Line 5: `from typing import Iterable`

### src/analyst/ml_confidence_predictor.py

#### Broad Exceptions

Line 297: `except Exception as e:`
Line 573: `except Exception as e:`
Line 604: `except Exception as e:`
Line 612: `except Exception as e:`
Line 639: `except Exception as e:`
Line 674: `except Exception as e:`
Line 739: `except Exception as e:`
Line 766: `except Exception as e:`
Line 793: `except Exception as e:`
Line 829: `except Exception as e:`
... and 46 more

#### Unused Imports

Line 9: `from lightgbm import LGBMClassifier`
Line 10: `from sklearn.model_selection import StratifiedKFold`
Line 23: `from src.utils.warning_symbols import critical`
Line 23: `from src.utils.warning_symbols import problem`
Line 23: `from src.utils.warning_symbols import invalid`
Line 23: `from src.utils.warning_symbols import timeout`
Line 23: `from src.utils.warning_symbols import connection_error`

#### Todo Comments

Line 1147: `self.logger.debug(f"Analyst model not found: {model_key}")`
Line 1172: `self.logger.debug(f"Tactician model not found: {model_key}")`
Line 1190: `self.logger.debug("No ensemble models available")`
Line 1204: `self.logger.debug("No calibrated models available")`
Line 1220: `self.logger.debug("No regime models available")`
Line 1240: `self.logger.debug("No multi-timeframe models available")`

### src/analyst/di_analyst.py

#### Broad Exceptions

Line 88: `except Exception:`
Line 164: `except Exception:`
Line 235: `except Exception:`
Line 254: `except Exception:`
Line 290: `except Exception:`
Line 314: `except Exception:`
Line 338: `except Exception:`

#### Todo Comments

Line 126: `self.logger.debug("Event subscriptions set up")`
Line 144: `self.logger.debug(f"Analyzing market data for {market_data.symbol}")`

### src/analyst/__init__.py

#### Unused Imports

Line 4: `from live_regime_calculations import LiveRegimeCalculator`
Line 4: `from live_regime_calculations import RegimeSummary`

### src/analyst/unified_regime_classifier.py

#### Type Ignores

Line 221: `import numpy.random._pickle as np_random_pickle  # type: ignore[attr-defined]`
Line 259: `import numpy.random._mt19937 as _mt  # type: ignore[attr-defined]`
Line 270: `np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor  # type: ignore[attr-defined]`

#### Broad Exceptions

Line 206: `except Exception as e:`
Line 244: `except Exception:`
Line 262: `except Exception:`
Line 266: `except Exception:`
Line 876: `except Exception as e:`
Line 1008: `except Exception as e:`
Line 1099: `except Exception as e:`
Line 1267: `except Exception as e:`
Line 1315: `except Exception as e:`
Line 1402: `except Exception as e:`
... and 7 more

#### Unused Imports

Line 4: `from typing import Union`
Line 4: `from typing import cast`
Line 15: `from src.utils.error_handler import create_fallback_strategy`
Line 15: `from src.utils.error_handler import handle_type_conversions`
Line 23: `from src.utils.warning_symbols import error`
Line 23: `from src.utils.warning_symbols import failed`

### src/analyst/simple_regime_rules.py

#### Unused Imports

Line 1: `from __future__ import annotations`

### src/analyst/analyst.py

#### Broad Exceptions

Line 183: `except Exception:`
Line 241: `except Exception:`
Line 263: `except Exception:`
Line 282: `except Exception as e:`
Line 408: `except Exception:`
Line 462: `except Exception:`
Line 523: `except Exception:`
Line 549: `except Exception:`
Line 580: `except Exception:`
Line 596: `except Exception:`
... and 11 more

#### Unused Imports

Line 20: `from src.utils.warning_symbols import invalid`
Line 20: `from src.utils.warning_symbols import missing`

### src/analyst/predictive_ensembles/multi_timeframe_ensemble.py

#### Broad Exceptions

Line 215: `except Exception:`
Line 269: `except Exception:`
Line 311: `except Exception:`
Line 333: `except Exception:`
Line 356: `except Exception:`
Line 431: `except Exception:`
Line 474: `except Exception:`
Line 541: `except Exception:`
Line 597: `except Exception:`
Line 674: `except Exception:`
... and 5 more

#### Todo Comments

Line 366: `self.logger.debug("🔧 Preparing features and target...")`
Line 428: `self.logger.debug(f"📊 Features shape: {X.shape}, Target shape: {y.shape}")`
Line 462: `self.logger.debug(`
Line 553: `self.logger.debug("🔧 Preparing meta-learner data...")`
Line 621: `self.logger.debug(`
Line 631: `self.logger.debug(f"📊 Getting prediction for {timeframe}...")`
Line 639: `self.logger.debug(`
Line 645: `self.logger.debug("🧠 Combining predictions with meta-learner...")`
Line 723: `self.logger.debug("🧠 Combining predictions with meta-learner...")`
Line 740: `self.logger.debug(f"📊 Meta-features: {meta_features}")`
... and 6 more

### src/analyst/predictive_ensembles/enhanced_ensemble_orchestrator.py

#### Broad Exceptions

Line 324: `except Exception:`
Line 515: `except Exception as e:`
Line 592: `except Exception as e:`

#### Todo Comments

Line 221: `self.logger.debug(f"📊 Initialized {regime_key} ensemble")`
Line 241: `self.logger.debug(f"📊 Collecting from {ensemble_key}...")`
Line 251: `self.logger.debug(`
Line 287: `self.logger.debug(`
Line 361: `self.logger.debug(f"📊 Processing {regime_key} regime...")`
Line 367: `self.logger.debug(f"📊 Getting prediction from {ensemble_key}...")`
Line 407: `self.logger.debug(`
Line 483: `self.logger.debug("🧠 Using global meta-learner for final prediction...")`
Line 510: `self.logger.debug(`
Line 530: `self.logger.debug("🔧 Preparing enhanced meta-features...")`
... and 6 more

### src/analyst/predictive_ensembles/ensemble_orchestrator.py

#### Broad Exceptions

Line 293: `except Exception as e:`
Line 584: `except Exception as e:`
Line 606: `except Exception as e:`
Line 756: `except Exception as e:`
Line 773: `except Exception as e:`

#### Todo Comments

Line 671: `self.logger.debug(f"Mapped cluster_id {cluster_id} to regime {regime}")`

### src/analyst/predictive_ensembles/two_tier_integration.py

#### Todo Comments

Line 177: `self.logger.debug("🔧 Extracting Tier 1 direction from ensemble...")`
Line 185: `self.logger.debug(`
Line 191: `self.logger.debug(`
Line 197: `self.logger.debug(f"📊 Medium confidence ({base_confidence:.3f}), holding")`
Line 206: `self.logger.debug(f"📊 Strategy classified as: {strategy}")`
Line 225: `self.logger.debug(`
Line 255: `self.logger.debug(`
Line 268: `self.logger.debug(f"⏰ Getting Tier 2 timing for strategy: {strategy}")`
Line 276: `self.logger.debug(f"📊 Timing signal: {timing_signal:.3f}")`
Line 280: `self.logger.debug(`
... and 24 more

### src/analyst/predictive_ensembles/__init__.py

#### Unused Imports

Line 3: `from src.utils.warning_symbols import connection_error`
Line 3: `from src.utils.warning_symbols import critical`
Line 3: `from src.utils.warning_symbols import error`
Line 3: `from src.utils.warning_symbols import execution_error`
Line 3: `from src.utils.warning_symbols import failed`
Line 3: `from src.utils.warning_symbols import initialization_error`
Line 3: `from src.utils.warning_symbols import invalid`
Line 3: `from src.utils.warning_symbols import missing`
Line 3: `from src.utils.warning_symbols import problem`
Line 3: `from src.utils.warning_symbols import timeout`
... and 3 more

### src/analyst/predictive_ensembles/regime_ensembles/sideways_range_ensemble.py

#### Broad Exceptions

Line 31: `except Exception:`
Line 104: `except Exception:`

### src/analyst/predictive_ensembles/regime_ensembles/volatile_regime_ensemble.py

#### Broad Exceptions

Line 117: `except Exception as e:`
Line 144: `except Exception as e:`
Line 164: `except Exception as e:`
Line 206: `except Exception as e:`
Line 257: `except Exception as e:`
Line 273: `except Exception as e:`
Line 287: `except Exception as e:`
Line 361: `except Exception as e:`

#### Unused Imports

Line 2: `from src.utils.warning_symbols import warning`
Line 2: `from src.utils.warning_symbols import critical`
Line 2: `from src.utils.warning_symbols import problem`
Line 2: `from src.utils.warning_symbols import invalid`
Line 2: `from src.utils.warning_symbols import missing`
Line 2: `from src.utils.warning_symbols import timeout`
Line 2: `from src.utils.warning_symbols import connection_error`
Line 2: `from src.utils.warning_symbols import validation_error`
Line 2: `from src.utils.warning_symbols import initialization_error`
Line 2: `from src.utils.warning_symbols import execution_error`

### src/analyst/predictive_ensembles/regime_ensembles/bull_trend_ensemble.py

#### Broad Exceptions

Line 113: `except Exception:`
Line 150: `except Exception:`
Line 325: `except Exception:`
Line 367: `except Exception:`

### src/analyst/predictive_ensembles/regime_ensembles/bear_trend_ensemble.py

#### Broad Exceptions

Line 107: `except Exception:`
Line 269: `except Exception:`
Line 284: `except Exception:`

### src/analyst/predictive_ensembles/regime_ensembles/__init__.py

#### Unused Imports

Line 4: `from src.utils.warning_symbols import connection_error`
Line 4: `from src.utils.warning_symbols import critical`
Line 4: `from src.utils.warning_symbols import error`
Line 4: `from src.utils.warning_symbols import execution_error`
Line 4: `from src.utils.warning_symbols import failed`
Line 4: `from src.utils.warning_symbols import initialization_error`
Line 4: `from src.utils.warning_symbols import invalid`
Line 4: `from src.utils.warning_symbols import missing`
Line 4: `from src.utils.warning_symbols import problem`
Line 4: `from src.utils.warning_symbols import timeout`
... and 7 more

### src/analyst/predictive_ensembles/regime_ensembles/base_ensemble.py

#### Broad Exceptions

Line 516: `except Exception as e:`
Line 597: `except Exception:`
Line 840: `except Exception as e:`
Line 852: `except Exception as e:`
Line 934: `except Exception as e:`
Line 964: `except Exception as e:`
Line 998: `except Exception as e:`
Line 1060: `except Exception:`
Line 1109: `except Exception:`
Line 1319: `except Exception as e:`
... and 6 more

#### Unused Imports

Line 11: `from src.utils.error_handler import handle_specific_errors`
Line 17: `from src.utils.lookahead_bias_detector import detect_lookahead_bias`
Line 17: `from src.utils.lookahead_bias_detector import apply_feature_lagging`

### src/database/efficient_features_database.py

#### Broad Exceptions

Line 74: `except Exception as e:`
Line 167: `except Exception as e:`
Line 180: `except Exception as e:`
Line 199: `except Exception:`
Line 218: `except Exception as e:`
Line 302: `except Exception as e:`
Line 341: `except Exception as e:`
Line 435: `except Exception as e:`
Line 515: `except Exception as e:`
Line 604: `except Exception as e:`
... and 2 more

#### Unused Imports

Line 13: `from src.utils.warning_symbols import critical`
Line 13: `from src.utils.warning_symbols import problem`
Line 13: `from src.utils.warning_symbols import failed`
Line 13: `from src.utils.warning_symbols import invalid`
Line 13: `from src.utils.warning_symbols import timeout`
Line 13: `from src.utils.warning_symbols import connection_error`
Line 13: `from src.utils.warning_symbols import validation_error`
Line 13: `from src.utils.warning_symbols import initialization_error`
Line 13: `from src.utils.warning_symbols import execution_error`

### src/database/precomputed_features_manager.py

#### Broad Exceptions

Line 99: `except Exception as e:`
Line 218: `except Exception:`
Line 305: `except Exception:`
Line 358: `except Exception:`
Line 390: `except Exception:`
Line 401: `except Exception:`
Line 453: `except Exception:`
Line 527: `except Exception:`

### src/database/migration_utils.py

#### Broad Exceptions

Line 92: `except Exception as e:`
Line 125: `except Exception as e:`
Line 179: `except Exception as e:`
Line 220: `except Exception as e:`
Line 289: `except Exception as e:`
Line 306: `except Exception as e:`
Line 322: `except Exception as e:`
Line 363: `except Exception as e:`

#### Unused Imports

Line 13: `from src.utils.warning_symbols import error`
Line 13: `from src.utils.warning_symbols import warning`
Line 13: `from src.utils.warning_symbols import critical`
Line 13: `from src.utils.warning_symbols import problem`
Line 13: `from src.utils.warning_symbols import invalid`
Line 13: `from src.utils.warning_symbols import timeout`
Line 13: `from src.utils.warning_symbols import connection_error`
Line 13: `from src.utils.warning_symbols import validation_error`
Line 13: `from src.utils.warning_symbols import initialization_error`
Line 13: `from src.utils.warning_symbols import execution_error`

### src/database/firestore_manager.py

#### Todo Comments

Line 193: `self.logger.debug(`
Line 229: `self.logger.debug(`
Line 248: `self.logger.debug(f"Document {doc_id} retrieved from {collection_name}.")`
Line 266: `self.logger.debug(`
Line 300: `self.logger.debug(`
Line 340: `self.logger.debug(`

### src/database/sqlite_manager.py

#### Broad Exceptions

Line 207: `except Exception:`
Line 239: `except Exception:`
Line 282: `except Exception:`
Line 304: `except Exception:`
Line 353: `except Exception:`
Line 428: `except Exception:`
Line 493: `except Exception:`
Line 570: `except Exception:`
Line 632: `except Exception:`
Line 675: `except Exception:`
... and 12 more

### GUI/api_server.py

#### Broad Exceptions

Line 159: `except Exception:`
Line 209: `except Exception:`
Line 533: `except Exception:`
Line 574: `except Exception as e:`
Line 584: `except Exception as e:`
Line 600: `except Exception as e:`
Line 625: `except Exception as e:`
Line 644: `except Exception as e:`
Line 701: `except Exception as e:`
Line 741: `except Exception as e:`
... and 20 more

### examples/enhanced_step1_7_usage_example.py

#### Broad Exceptions

Line 80: `except Exception as e:`
Line 121: `except Exception as e:`

### examples/enhanced_nan_analysis_example.py

#### Debug Statements

Line 85: `print("\n🔍 Analyzing NaN patterns in trading data...")`

#### Unused Imports

Line 11: `import os`
Line 21: `from datetime import timedelta`

### examples/regime_expert_usage_example.py

#### Unused Imports

Line 10: `from typing import Dict`
Line 10: `from typing import Any`

### examples/enhanced_reporting_example.py

#### Broad Exceptions

Line 380: `except Exception as e:`

#### Unused Imports

Line 10: `from src.utils.warning_symbols import error`
Line 10: `from src.utils.warning_symbols import critical`
Line 10: `from src.utils.warning_symbols import problem`
Line 10: `from src.utils.warning_symbols import invalid`
Line 10: `from src.utils.warning_symbols import missing`
Line 10: `from src.utils.warning_symbols import timeout`
Line 10: `from src.utils.warning_symbols import connection_error`
Line 10: `from src.utils.warning_symbols import validation_error`
Line 10: `from src.utils.warning_symbols import initialization_error`
Line 10: `from src.utils.warning_symbols import execution_error`

### examples/data_quality_example.py

#### Debug Statements

Line 67: `print("🔍 Data Quality Assessment Example")`
Line 101: `print("\n🔍 Performing data quality assessment...")`
Line 182: `print("🔍 REAL DATA INTEGRATION EXAMPLE")`

#### Broad Exceptions

Line 210: `except Exception as e:`

#### Unused Imports

Line 12: `from datetime import datetime`
Line 12: `from datetime import timedelta`

### exchange/mexc_optimized.py

#### Debug Statements

Line 83: `print(f"🔍 DEBUG: HTTP error {response.status}: {text[:200]}")`
Line 134: `print(f"🔍 DEBUG: Error fetching hour data: {e}")`
Line 145: `print(f"🔍 DEBUG: Fetching {len(hour_ranges)} hours concurrently")`
Line 158: `print(f"🔍 DEBUG: Hour {i+1} failed: {result}")`
Line 161: `print(f"🔍 DEBUG: Hour {i+1} completed: {len(result)} trades")`
Line 177: `print("🔍 DEBUG: Optimized MEXC _get_historical_agg_trades_raw called")`
Line 192: `print(f"🔍 DEBUG: Processing {len(hour_ranges)} hours concurrently")`
Line 203: `print(f"🔍 DEBUG: Error in optimized _get_historical_agg_trades_raw: {e}")`
Line 217: `print("🔍 DEBUG: Optimized MEXC get_historical_agg_trades called")`
Line 234: `print(f"🔍 DEBUG: Returning {len(result)} trades")`

#### Broad Exceptions

Line 32: `except Exception as e:`
Line 133: `except Exception as e:`
Line 202: `except Exception as e:`

#### Todo Comments

Line 83: `print(f"🔍 DEBUG: HTTP error {response.status}: {text[:200]}")`
Line 134: `print(f"🔍 DEBUG: Error fetching hour data: {e}")`
Line 145: `print(f"🔍 DEBUG: Fetching {len(hour_ranges)} hours concurrently")`
Line 158: `print(f"🔍 DEBUG: Hour {i+1} failed: {result}")`
Line 161: `print(f"🔍 DEBUG: Hour {i+1} completed: {len(result)} trades")`
Line 177: `print("🔍 DEBUG: Optimized MEXC _get_historical_agg_trades_raw called")`
Line 179: `f"🔍 DEBUG: Time range: {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}",`
Line 192: `print(f"🔍 DEBUG: Processing {len(hour_ranges)} hours concurrently")`
Line 198: `f"🔍 DEBUG: Successfully collected {len(all_trades)} aggregated trades",`
Line 203: `print(f"🔍 DEBUG: Error in optimized _get_historical_agg_trades_raw: {e}")`
... and 4 more

### exchange/okx.py

#### Broad Exceptions

Line 92: `except Exception as e:`
Line 146: `except Exception as e:`
Line 175: `except Exception as e:`
Line 189: `except Exception as e:`
Line 205: `except Exception as e:`
Line 218: `except Exception as e:`
Line 231: `except Exception as e:`
Line 244: `except Exception as e:`
Line 306: `except Exception as e:`
Line 328: `except Exception as e:`
... and 15 more

#### Unused Imports

Line 20: `from src.utils.warning_symbols import warning`
Line 20: `from src.utils.warning_symbols import critical`
Line 20: `from src.utils.warning_symbols import problem`
Line 20: `from src.utils.warning_symbols import invalid`
Line 20: `from src.utils.warning_symbols import missing`
Line 20: `from src.utils.warning_symbols import timeout`
Line 20: `from src.utils.warning_symbols import connection_error`
Line 20: `from src.utils.warning_symbols import validation_error`
Line 20: `from src.utils.warning_symbols import initialization_error`
Line 20: `from src.utils.warning_symbols import execution_error`

### exchange/gateio.py

#### Broad Exceptions

Line 94: `except Exception as e:`
Line 147: `except Exception as e:`
Line 190: `except Exception as e:`
Line 283: `except Exception as e:`
Line 415: `except Exception as e:`
Line 428: `except Exception as e:`
Line 538: `except Exception as e:`
Line 553: `except Exception as e:`
Line 559: `except Exception as e:`
Line 590: `except Exception as e:`
... and 14 more

#### Unused Imports

Line 22: `from src.utils.warning_symbols import warning`
Line 22: `from src.utils.warning_symbols import critical`
Line 22: `from src.utils.warning_symbols import problem`
Line 22: `from src.utils.warning_symbols import invalid`
Line 22: `from src.utils.warning_symbols import missing`
Line 22: `from src.utils.warning_symbols import timeout`
Line 22: `from src.utils.warning_symbols import connection_error`
Line 22: `from src.utils.warning_symbols import validation_error`
Line 22: `from src.utils.warning_symbols import initialization_error`
Line 22: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 367: `# Log first few trades for debugging`

### exchange/mexc.py

#### Debug Statements

Line 394: `print("🔍 DEBUG: MEXC get_historical_agg_trades called")`

#### Broad Exceptions

Line 94: `except Exception as e:`
Line 146: `except Exception as e:`
Line 201: `except Exception as e:`
Line 373: `except Exception as e:`
Line 380: `except Exception as e:`
Line 436: `except Exception as e:`
Line 553: `except Exception as e:`
Line 566: `except Exception as e:`
Line 597: `except Exception as e:`
Line 611: `except Exception as e:`
... and 16 more

#### Unused Imports

Line 22: `from src.utils.warning_symbols import critical`
Line 22: `from src.utils.warning_symbols import problem`
Line 22: `from src.utils.warning_symbols import invalid`
Line 22: `from src.utils.warning_symbols import missing`
Line 22: `from src.utils.warning_symbols import timeout`
Line 22: `from src.utils.warning_symbols import connection_error`
Line 22: `from src.utils.warning_symbols import validation_error`
Line 22: `from src.utils.warning_symbols import initialization_error`
Line 22: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 394: `print("🔍 DEBUG: MEXC get_historical_agg_trades called")`
Line 396: `f"🔍 DEBUG: Parameters: symbol={symbol}, start_time_ms={start_time_ms}, end_time_ms={end_time_ms}, limit={limit}",`

### exchange/binance.py

#### Broad Exceptions

Line 207: `except Exception as e:`
Line 359: `except Exception:`
Line 387: `except Exception:`
Line 436: `except Exception as e:`
Line 489: `except Exception as e:`
Line 588: `except Exception as e:`
Line 655: `except Exception as e:`
Line 673: `except Exception as e:`
Line 690: `except Exception as e:`
Line 704: `except Exception as e:`
... and 39 more

#### Unused Imports

Line 31: `from src.utils.warning_symbols import critical`
Line 31: `from src.utils.warning_symbols import problem`
Line 31: `from src.utils.warning_symbols import missing`
Line 31: `from src.utils.warning_symbols import timeout`
Line 31: `from src.utils.warning_symbols import connection_error`
Line 31: `from src.utils.warning_symbols import validation_error`
Line 31: `from src.utils.warning_symbols import initialization_error`
Line 31: `from src.utils.warning_symbols import execution_error`

#### Todo Comments

Line 784: `system_logger.debug(`
Line 974: `# Debug: Check what we're getting from the API`
Line 977: `f"CCXT Debug: First trade timestamp: {trades[0].get('T')}",`
Line 980: `f"CCXT Debug: Last trade timestamp: {trades[-1].get('T')}",`
Line 983: `f"CCXT Debug: Number of unique timestamps: {len({t.get('T') for t in trades})}",`

### exchange/base_exchange.py

#### Broad Exceptions

Line 324: `except Exception:`
Line 341: `except Exception:`
Line 350: `except Exception:`
Line 367: `except Exception:`
Line 466: `except Exception:`
Line 499: `except Exception:`
Line 515: `except Exception:`
Line 525: `except Exception:`
