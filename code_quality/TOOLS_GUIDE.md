# Code Quality Tools Guide

A comprehensive guide to all code quality tools created for maintaining and improving the codebase.

## Table of Contents
1. [Master Control Tool](#master-control-tool)
2. [Import Management Tools](#import-management-tools)
3. [Syntax Error Fixers](#syntax-error-fixers)
4. [Async/Await Fixers](#asyncawait-fixers)
5. [Type Hint Tools](#type-hint-tools)
6. [Code Analysis Tools](#code-analysis-tools)
7. [Utility Modules](#utility-modules)

---

## Master Control Tool

### `master_code_quality.py`
**Purpose**: Unified interface for all code quality operations

**Usage**:
```bash
# Show quality dashboard
python3 /workspace/code_quality/scripts/master_code_quality.py --dashboard

# Analyze current state
python3 /workspace/code_quality/scripts/master_code_quality.py --analyze

# Fix specific issues (dry run)
python3 /workspace/code_quality/scripts/master_code_quality.py --fix syntax imports async types

# Fix all issues (dry run)
python3 /workspace/code_quality/scripts/master_code_quality.py --fix all

# Apply fixes (actual changes)
python3 /workspace/code_quality/scripts/master_code_quality.py --fix all --apply

# Generate comprehensive report
python3 /workspace/code_quality/scripts/master_code_quality.py --report
```

**Features**:
- Quality score calculation (0-100)
- Coordinated execution of all tools
- Progress tracking
- Historical comparison

---

## Import Management Tools

### `fix_missing_imports.py`
**Purpose**: Analyzes and fixes missing imports using AST parsing

**Usage**:
```bash
# Dry run (see what would be fixed)
python3 /workspace/code_quality/scripts/fix_missing_imports.py --project-root /workspace/src

# Apply fixes
python3 /workspace/code_quality/scripts/fix_missing_imports.py --project-root /workspace/src --fix

# Custom exclusions
python3 /workspace/code_quality/scripts/fix_missing_imports.py --exclude "*/tests/*" "*/docs/*"
```

**Fixes**:
- Missing imports for pandas, numpy, datetime, etc.
- Import ordering
- Duplicate imports

### `safe_import_fixer.py`
**Purpose**: Safer regex-based import fixing for files with syntax errors

**Usage**:
```bash
# Dry run
python3 /workspace/code_quality/scripts/safe_import_fixer.py --project-root /workspace/src

# Apply fixes
python3 /workspace/code_quality/scripts/safe_import_fixer.py --fix
```

**Features**:
- Works on files with syntax errors
- Pattern-based import detection
- Safe insertion without breaking code

---

## Syntax Error Fixers

### `advanced_syntax_fixer.py`
**Purpose**: Fixes complex syntax errors automatically

**Usage**:
```bash
# Analyze syntax errors
python3 /workspace/code_quality/scripts/advanced_syntax_fixer.py --project-root /workspace/src

# Fix syntax errors
python3 /workspace/code_quality/scripts/advanced_syntax_fixer.py --fix
```

**Fixes**:
- Missing colons after function/class definitions
- Try blocks without except/finally
- Indentation errors
- Invalid syntax patterns

### `fix_common_syntax_patterns.py`
**Purpose**: Quick fixes for common syntax patterns

**Usage**:
```bash
# Run on entire codebase
python3 /workspace/code_quality/scripts/fix_common_syntax_patterns.py
```

**Current Patterns**:
- `:->` to `->` (type hint syntax)
- More patterns can be added easily

---

## Async/Await Fixers

### `fix_async_await.py`
**Purpose**: Basic async/await pattern fixing

**Usage**:
```bash
# Dry run
python3 /workspace/code_quality/scripts/fix_async_await.py --project-root /workspace/src

# Apply fixes
python3 /workspace/code_quality/scripts/fix_async_await.py --fix
```

### `robust_async_fixer.py`
**Purpose**: Advanced async/await fixing with context awareness

**Usage**:
```bash
# Dry run
python3 /workspace/code_quality/scripts/robust_async_fixer.py --project-root /workspace/src

# Apply fixes
python3 /workspace/code_quality/scripts/robust_async_fixer.py --fix
```

**Features**:
- Detects async context accurately
- Maintains list of known async functions
- Fixes various await patterns
- Handles nested async contexts

---

## Type Hint Tools

### `add_type_hints.py`
**Purpose**: Analyzes type hint coverage and suggests improvements

**Usage**:
```bash
# Analyze type hint coverage
python3 /workspace/code_quality/scripts/add_type_hints.py --analyze

# Suggest type hints for specific file
python3 /workspace/code_quality/scripts/add_type_hints.py --suggest /path/to/file.py

# Create stub file
python3 /workspace/code_quality/scripts/add_type_hints.py --create-stub /path/to/module.py
```

**Features**:
- Coverage analysis
- Type inference from names
- Stub file generation

### `enhanced_type_hints.py`
**Purpose**: Intelligent type hint enhancement to reach 90%+ coverage

**Usage**:
```bash
# Analyze and improve to 90% coverage
python3 /workspace/code_quality/scripts/enhanced_type_hints.py --target 0.9

# Custom target coverage
python3 /workspace/code_quality/scripts/enhanced_type_hints.py --target 0.95
```

**Features**:
- Smart parameter type inference
- Return type pattern matching
- Automatic import management
- Bulk processing

---

## Code Analysis Tools

### `detect_circular_imports.py`
**Purpose**: Detects and reports circular import dependencies

**Usage**:
```bash
# Analyze project for circular imports
python3 /workspace/code_quality/scripts/detect_circular_imports.py --project-root /workspace/src

# Custom output location
python3 /workspace/code_quality/scripts/detect_circular_imports.py --output /path/to/report.json
```

**Reports**:
- Circular dependency chains
- Import depth analysis
- Highly imported modules
- Suggested fixes

### `extract_interactions.py`
**Purpose**: Extracts code interaction patterns from validation reports

**Usage**:
```bash
# Extract interactions from validation report
python3 /workspace/code_quality/scripts/extract_interactions.py
```

**Analyzes**:
- Function call relationships
- Module dependencies
- Undefined functions
- Import patterns

### `interaction_summary.py`
**Purpose**: Generates human-readable interaction summaries

**Usage**:
```bash
# Generate interaction summary
python3 /workspace/code_quality/scripts/interaction_summary.py
```

### `simple_interaction_mapper.py`
**Purpose**: Maps code interactions using existing tools

**Usage**:
```bash
# Map interactions for entire project
python3 /workspace/code_quality/scripts/simple_interaction_mapper.py --project-root /workspace

# Custom output directory
python3 /workspace/code_quality/scripts/simple_interaction_mapper.py --output-dir /path/to/output
```

---

## Utility Modules

### `common_operations.py`
**Location**: `/workspace/src/utils/common_operations.py`

**Purpose**: Provides 50+ commonly used operations to reduce undefined function errors

**Usage in Code**:
```python
from src.utils.common_operations import (
    # DateTime operations
    get_current_datetime,
    get_today,
    format_datetime,
    parse_datetime,
    
    # DataFrame operations
    create_empty_dataframe,
    safe_fillna,
    safe_rolling,
    
    # Numeric operations
    safe_mean,
    safe_std,
    
    # File operations
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    
    # Async operations
    safe_sleep,
    safe_gather,
    create_async_task,
    
    # And many more...
)
```

**Categories**:
- DateTime utilities
- DataFrame operations
- Numeric calculations
- File handling
- Async/await helpers
- Collection operations
- String manipulation
- Logging utilities
- Argument parsing
- Type conversions
- Validation helpers
- Memory optimization

---

## Batch Processing Tools

### `apply_all_fixes.py`
**Purpose**: Original script to coordinate all fixes

**Usage**:
```bash
# Dry run mode
python3 /workspace/code_quality/scripts/apply_all_fixes.py

# Apply all fixes
python3 /workspace/code_quality/scripts/apply_all_fixes.py --apply
```

### `final_code_fixes.py`
**Purpose**: Applies final touches and generates reports

**Usage**:
```bash
# Run final fixes and generate report
python3 /workspace/code_quality/scripts/final_code_fixes.py
```

---

## Quick Reference Commands

### Daily Maintenance
```bash
# Check current quality
python3 /workspace/code_quality/scripts/master_code_quality.py --dashboard

# Fix all issues (dry run first)
python3 /workspace/code_quality/scripts/master_code_quality.py --fix all
python3 /workspace/code_quality/scripts/master_code_quality.py --fix all --apply
```

### Specific Fixes
```bash
# Fix syntax errors only
python3 /workspace/code_quality/scripts/advanced_syntax_fixer.py --fix

# Fix imports only
python3 /workspace/code_quality/scripts/safe_import_fixer.py --fix

# Fix async/await only
python3 /workspace/code_quality/scripts/robust_async_fixer.py --fix

# Improve type hints
python3 /workspace/code_quality/scripts/enhanced_type_hints.py --target 0.9
```

### Analysis Only
```bash
# Check circular imports
python3 /workspace/code_quality/scripts/detect_circular_imports.py

# Analyze code interactions
python3 /workspace/code_quality/scripts/simple_interaction_mapper.py

# Generate summary report
python3 /workspace/code_quality/scripts/master_code_quality.py --report
```

---

## Reports Location

All reports are saved in `/workspace/code_quality/reports/`:
- `syntax_fixes_report.json`
- `import_fixes_report.json`
- `async_fixes_report.json`
- `type_hints_report.json`
- `circular_imports_report.json`
- `quality_summary_*.json`
- `quality_history.json`

---

## Best Practices

1. **Always run in dry-run mode first** to see what changes will be made
2. **Check quality dashboard** before and after fixes to measure improvement
3. **Fix syntax errors first** - they block other tools from working
4. **Use the master script** for coordinated fixes
5. **Review reports** to understand patterns and prioritize fixes
6. **Backup important files** before applying bulk fixes
7. **Run tests** after applying fixes to ensure nothing broke

---

## Troubleshooting

### Tools not working?
1. Check Python version (3.7+ required)
2. Ensure you're in the correct directory
3. Check file permissions
4. Look for syntax errors preventing parsing

### Fixes not applying?
1. Some syntax errors require manual intervention
2. Complex patterns may not be auto-fixable
3. Check the reports for specific error details

### Quality score not improving?
1. Focus on high-impact files first
2. Fix syntax errors before other issues
3. Some improvements take time to reflect

---

## Adding New Patterns

To add new fix patterns, edit the relevant tool:

1. For syntax patterns: Edit `fix_common_syntax_patterns.py`
2. For import patterns: Edit `COMMON_IMPORTS` in `fix_missing_imports.py`
3. For async functions: Update the list in `robust_async_fixer.py`
4. For type patterns: Edit `param_type_patterns` in `enhanced_type_hints.py`