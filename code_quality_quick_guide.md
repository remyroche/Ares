# Code Quality Tools - Quick Guide

## Daily Code Quality Check

```bash
# Check current quality status
python3 code_quality/scripts/master_code_quality.py --dashboard

# Fix all issues (dry run first)
python3 code_quality/scripts/master_code_quality.py --fix all

# Apply fixes
python3 code_quality/scripts/master_code_quality.py --fix all --apply
```

## Specific Fixes

### Fix Syntax Errors
```bash
python3 code_quality/scripts/advanced_syntax_fixer.py --project-root /workspace/src --fix
```

### Fix Import Issues
```bash
python3 code_quality/scripts/safe_import_fixer.py --project-root /workspace/src --fix
```

### Fix Async/Await Issues
```bash
python3 code_quality/scripts/robust_async_fixer.py --project-root /workspace/src --fix
```

### Improve Type Hints
```bash
# Target 90% coverage
python3 code_quality/scripts/enhanced_type_hints.py --project-root /workspace/src --target 0.9
```

## Analysis Tools

### Function Validation
```bash
python3 code_quality/function_validator.py --project-root /workspace/src --output reports/function_validation.json
```

### Circular Import Detection
```bash
python3 code_quality/scripts/detect_circular_imports.py --project-root /workspace/src
```

### Comprehensive Code Review
```bash
python3 code_quality/comprehensive_code_review.py --project-root /workspace/src
```

## Custom Fix Script

For specific syntax patterns not caught by standard tools:
```bash
python3 /workspace/fix_syntax_errors.py
```

## Reports Location

All reports are saved in: `/workspace/code_quality/reports/`

Key reports:
- `quality_summary_*.json` - Overall quality metrics
- `syntax_fixes_report_*.json` - Syntax error details
- `import_fixes_report_*.json` - Import issues
- `async_fixes_report_*.json` - Async/await issues
- `type_hints_report_*.json` - Type hint coverage

## Best Practices

1. Always run tools in dry-run mode first
2. Review reports before applying fixes
3. Back up critical files before bulk fixes
4. Run tests after applying fixes
5. Use the dashboard to track progress

## Integration

Add to your development workflow:
```bash
# Pre-commit hook
python3 code_quality/scripts/master_code_quality.py --dashboard

# CI/CD pipeline
python3 code_quality/run_validation.py --mode both
```