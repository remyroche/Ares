# Code Quality Scripts

This directory contains all the specialized scripts for maintaining and improving code quality.

## Scripts Overview

### 1. Import Management
- **`fix_missing_imports.py`** - Analyzes and fixes missing imports using AST parsing
- **`safe_import_fixer.py`** - Safer regex-based import fixing for files with syntax errors

### 2. Async/Await Management
- **`fix_async_await.py`** - Identifies and fixes missing await statements

### 3. Circular Import Detection
- **`detect_circular_imports.py`** - Analyzes and reports circular import dependencies

### 4. Type Hint Management
- **`add_type_hints.py`** - Analyzes type hint coverage and suggests improvements

### 5. Code Analysis
- **`extract_interactions.py`** - Extracts code interaction patterns from validation reports
- **`interaction_summary.py`** - Generates summaries of code interactions
- **`simple_interaction_mapper.py`** - Maps code interactions using existing tools

### 6. Master Scripts
- **`apply_all_fixes.py`** - Coordinates all fixes in the correct order
- **`final_code_fixes.py`** - Applies final touches and generates reports

## Usage

### Run All Fixes (Dry Run)
```bash
python3 /workspace/code_quality/scripts/apply_all_fixes.py
```

### Run All Fixes (Apply)
```bash
python3 /workspace/code_quality/scripts/apply_all_fixes.py --apply
```

### Individual Scripts

Fix imports:
```bash
python3 /workspace/code_quality/scripts/safe_import_fixer.py --fix
```

Fix async/await:
```bash
python3 /workspace/code_quality/scripts/fix_async_await.py --fix
```

Check circular imports:
```bash
python3 /workspace/code_quality/scripts/detect_circular_imports.py
```

Analyze type hints:
```bash
python3 /workspace/code_quality/scripts/add_type_hints.py --analyze
```

## Reports

All reports are saved in `/workspace/code_quality/reports/` directory.