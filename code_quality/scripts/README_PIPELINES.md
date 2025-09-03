# Code Quality Scripts - Tool Organization

This directory contains individual code quality tools. The pipelines that orchestrate these tools have been moved to `code_quality/pipelines/` for better organization.

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

## Pipelines

All pipelines have been moved to `code_quality/pipelines/`. See the [Pipelines README](../pipelines/README.md) for detailed information about available pipelines.

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
cd /workspace/code_quality/pipelines
python pipeline_unified_standalone.py --project-root /workspace/src
```

For specific fixes only:
```bash
cd /workspace/code_quality/pipelines

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