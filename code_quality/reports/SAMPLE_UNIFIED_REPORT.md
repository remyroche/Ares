# Code Quality Unified Report

**Generated:** 2025-01-15T14:30:00
**Project Root:** /workspace/src

## Overall Summary
- **Total Files Analyzed:** 480
- **Total Directories:** 45
- **Total Issues Found:** 3,245
- **Issues Fixed:** 2,876

### Issue Breakdown
- **Syntax Errors:** 125
- **Import Issues:** 892
- **Async Issues:** 234
- **Type Issues:** 1,456
- **Function Issues:** 378
- **Circular Imports:** 12
- **Security Issues:** 45
- **Performance Issues:** 103

## Critical Files (Most Issues)
| File | Total Issues | Fixed |
|------|--------------|-------|
| data_manager.py | 145 | 132 |
| ares_pipeline.py | 134 | 118 |
| config.py | 98 | 89 |
| paper_trader.py | 87 | 76 |
| indicators.py | 76 | 65 |
| ares_model_v6.py | 72 | 68 |
| backtester_v3.py | 69 | 61 |
| optimizer_v2.py | 58 | 52 |
| utils.py | 54 | 49 |
| logger.py | 48 | 42 |

## Directory Summary
| Directory | Files | Files with Issues | Total Issues | Fixed |
|-----------|-------|-------------------|--------------|-------|
| models | 45 | 38 | 523 | 467 |
| utils | 23 | 19 | 312 | 285 |
| data | 18 | 15 | 245 | 218 |
| trading | 32 | 28 | 398 | 356 |
| config | 12 | 10 | 167 | 149 |
| pipelines | 8 | 7 | 134 | 121 |
| indicators | 15 | 13 | 198 | 176 |
| backtesting | 22 | 19 | 267 | 238 |
| optimization | 17 | 14 | 223 | 195 |
| reporting | 11 | 8 | 145 | 128 |

## File Details (Top 20)

### data_manager.py
**Path:** `/workspace/src/data_manager.py`
**Lines of Code:** 1,234
**Total Issues:** 145 (Fixed: 132)

**Syntax Errors:**
- Missing colon at line 234
- Incorrect indentation at line 567
- Unclosed parenthesis at line 890

**Import Issues:**
- Missing import for pandas
- Missing import for numpy
- Circular import with config.py
- ... and 12 more

**Async Issues:**
- Missing await for async function call at line 345
- Async function called without await at line 678
- ... and 8 more

**Type Issues:**
- Missing type hint for parameter 'df' in function 'process_data'
- Missing return type for function 'calculate_indicators'
- Missing type hint for parameter 'config' in function '__init__'
- ... and 45 more

### ares_pipeline.py
**Path:** `/workspace/src/ares_pipeline.py`
**Lines of Code:** 2,156
**Total Issues:** 134 (Fixed: 118)

**Import Issues:**
- Missing import for AresModelV6
- Missing import for DataManager
- Unused import 'os'
- ... and 23 more

**Function Issues:**
- Undefined function 'run_full_training' called at line 456
- Parameter mismatch for function 'initialize' at line 234
- Missing required parameter 'config' at line 789
- ... and 18 more

**Async Issues:**
- Missing await for 'run_async' at line 123
- Async function 'initialize' called without await at line 345
- ... and 12 more

## Clean Files
**156 files with no issues found**