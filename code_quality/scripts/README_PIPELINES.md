# Code Quality Scripts - Pipeline Organization

This directory contains code quality tools organized into category-based and unified pipelines.

## Directory Structure

### Individual Tools
- **Syntax & Import Fixes**
  - `advanced_syntax_fixer.py` - Fixes complex syntax errors
  - `safe_import_fixer.py` - Fixes missing and incorrect imports
  - `detect_circular_imports.py` - Detects circular import dependencies

- **Async & Type Enhancements**
  - `robust_async_fixer.py` - Fixes missing await statements (comprehensive)
  - `enhanced_type_hints.py` - Adds type hints with advanced inference

- **Code Analysis**
  - `simple_interaction_mapper.py` - Maps code interactions and dependencies
  - `extract_interactions.py` - Extracts interaction patterns from reports

### Category-Based Pipelines

1. **`pipeline_syntax_imports.py`** - Syntax and Import Pipeline
   - Runs: syntax fixes, import fixes, circular import detection
   - Usage: `python pipeline_syntax_imports.py --project-root /workspace/src`

2. **`pipeline_async_types.py`** - Async and Type Hints Pipeline
   - Runs: async/await fixes, type hint enhancements
   - Usage: `python pipeline_async_types.py --project-root /workspace/src`

3. **`pipeline_analysis.py`** - Code Analysis Pipeline
   - Runs: function validation, interaction mapping, comprehensive review
   - Usage: `python pipeline_analysis.py --project-root /workspace/src`

### Unified Pipelines

1. **`pipeline_unified_standalone.py`** - Standalone Version (No Imports)
   - Runs all tools using subprocess calls
   - Better for isolation and avoiding import conflicts
   - Usage: `python pipeline_unified_standalone.py --project-root /workspace/src`
   - Options:
     - `--categories syntax_imports async_types` - Run specific categories
     - `--tool syntax_fixer` - Run a single tool
     - `--timeout 600` - Set timeout per tool

2. **`pipeline_unified_integrated.py`** - Integrated Version (Direct Imports)
   - Directly imports and uses code quality modules
   - Better performance, tighter integration
   - Usage: `python pipeline_unified_integrated.py --project-root /workspace/src`

## Report Output

All reports are now saved with datetime stamps in the `code_quality/reports/` directory:
- Format: `{tool_name}_{YYYYMMDD_HHMMSS}.{json|txt}`
- Example: `syntax_fixes_20250115_143022.json`

## Redundant Tools (Deprecated)

The following tools have overlapping functionality and should be replaced by their enhanced versions:
- `fix_async_await.py` → Use `robust_async_fixer.py`
- `add_type_hints.py` → Use `enhanced_type_hints.py`
- `fix_common_syntax_patterns.py` → Use `advanced_syntax_fixer.py`

## Quick Start

For a complete code quality check, run:
```bash
cd /workspace/code_quality/scripts
python pipeline_unified_standalone.py --project-root /workspace/src
```

For specific fixes only:
```bash
# Fix syntax and imports only
python pipeline_syntax_imports.py --project-root /workspace/src

# Fix async and add type hints only  
python pipeline_async_types.py --project-root /workspace/src

# Run analysis only
python pipeline_analysis.py --project-root /workspace/src
```

## Best Practices

1. Always run syntax fixes before other tools
2. Use the standalone pipeline for production runs to avoid import conflicts
3. Check reports in `code_quality/reports/` for detailed results
4. Run tools in dry-run mode first to preview changes