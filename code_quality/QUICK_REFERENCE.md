# Code Quality Quick Reference

## Common Operations Module

Import commonly used functions:
```python
from src.utils.common_operations import (
    get_current_datetime,  # Get current datetime
    safe_fillna,          # Fill NaN values safely
    safe_mean,            # Calculate mean with empty handling
    ensure_directory,     # Create directory if not exists
    safe_json_dump,       # Save JSON safely
    create_argument_parser # Create argument parser
)
```

## Running Code Quality Tools

1. Check for issues:
   ```bash
   python3 /workspace/code_quality/apply_all_fixes.py
   ```

2. Fix imports:
   ```bash
   python3 /workspace/code_quality/safe_import_fixer.py --fix
   ```

3. Check circular imports:
   ```bash
   python3 /workspace/code_quality/detect_circular_imports.py
   ```

4. Analyze type hints:
   ```bash
   python3 /workspace/code_quality/add_type_hints.py --analyze
   ```

## Best Practices

1. Always import required modules at the top of files
2. Use `await` with async function calls
3. Add type hints to function signatures
4. Use the common_operations module for frequently used utilities
5. Run code quality checks before committing
