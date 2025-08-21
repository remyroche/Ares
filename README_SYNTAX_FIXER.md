# Universal Python Syntax Fixer

This script automatically fixes common Python syntax errors that were identified during the cleanup of the `src/utils/` directory. It can be used on any Python file or directory to fix similar issues.

## What It Fixes

The script addresses the following common syntax issues:

### 1. Import Statement Fixes
- Malformed typing imports: `from typing import Any = Dict` → `from typing import Any, Dict`
- Custom module imports: `from src.utils import func1 = func2` → `from src.utils import func1, func2`

### 2. Function Signature Fixes
- Incorrect parameter syntax: `def func(param): type)` → `def func(param: type)`
- Complex parameter patterns: `def func(self, param): type = default)` → `def func(self, param: type = default)`
- Async function signatures: `async def func(param): type)` → `async def func(param: type)`

### 3. Exception Handling Fixes
- Exception tuples: `ValueError = AttributeError` → `ValueError, AttributeError`
- Error handling patterns: `(False = "message")` → `(False, "message")`

### 4. Assignment vs Comparison Fixes
- `isinstance()` calls: `isinstance(obj = type)` → `isinstance(obj, type)`
- For loops: `for key = value in items:` → `for key, value in items:`
- Function calls: `func(param = value)` → `func(param, value)`

### 5. Return Statement Fixes
- Multiple returns: `return val1 = val2` → `return val1, val2`
- Specific patterns: `return None = None` → `return None, None`

### 6. Variable Declaration Fixes
- Type annotations: `dict[key = value]` → `dict[key, value]`
- Function parameters: `param: str | None = None = )` → `param: str | None = None)`

### 7. Integration Configuration Fixes
- Sentry integrations: `integrations=[sentry = AioHttp()]` → `integrations=[sentry, AioHttp()]`

### 8. Try/Except Block Fixes
- Missing try statements: `AVAILABLE , True` → `try:\n    AVAILABLE = True`

### 9. Indentation and Structure Fixes
- Empty code blocks: Adds `pass` statements where needed
- Inconsistent indentation: Normalizes indentation levels
- Function definitions in classes: Proper indentation

## Usage

### Basic Usage
```bash
# Fix a single file
python universal_syntax_fixer.py path/to/file.py

# Fix all Python files in a directory
python universal_syntax_fixer.py path/to/directory/

# Fix all Python files recursively in a directory
python universal_syntax_fixer.py src/
```

### Examples
```bash
# Fix the utils directory
python universal_syntax_fixer.py src/utils/

# Fix the analyst directory
python universal_syntax_fixer.py src/analyst/

# Fix a specific file
python universal_syntax_fixer.py src/utils/logger.py

# Fix the entire src directory
python universal_syntax_fixer.py src/
```

## Output

The script provides detailed feedback:

```
🔍 Found 45 Python files in 'src/utils/'
==================================================
✅ Fixed: src/utils/async_utils.py
✅ Fixed: src/utils/data_loader.py
⏭️  No changes needed: src/utils/logger.py
❌ Error fixing src/utils/problematic_file.py: invalid syntax
==================================================
📊 Summary: Fixed 42 out of 45 files
```

## Safety Features

- **Backup**: The script modifies files in place, so consider backing up your code before running
- **Error Handling**: Individual file errors don't stop the entire process
- **Validation**: Only Python files (`.py` extension) are processed
- **Recursive**: Automatically finds all Python files in subdirectories

## When to Use

Use this script when you encounter:
- Syntax errors related to the patterns listed above
- Files that won't compile due to malformed syntax
- Large codebases with similar syntax issues
- After code generation or automated editing that introduced syntax errors

## Limitations

The script focuses on common, pattern-based syntax errors. It may not fix:
- Complex logical errors
- Semantic issues
- Custom syntax patterns not covered
- Deep structural problems requiring manual analysis

## Best Practices

1. **Backup First**: Always backup your code before running the script
2. **Test Incrementally**: Start with a small subset of files
3. **Review Changes**: Check the modified files to ensure the fixes are correct
4. **Run Tests**: After fixing, run your test suite to ensure functionality is preserved
5. **Version Control**: Commit your changes to version control for easy rollback

## Example Before/After

### Before (Broken Syntax)
```python
from typing import Any = Dict , List = Optional, Tuple

def validate_data(self, data): pd.DataFrame) -> bool:
    if isinstance(data = pd.DataFrame):
        for key = value in data.items():
            return True = False
    return None = None
```

### After (Fixed Syntax)
```python
from typing import Any, Dict, List, Optional, Tuple

def validate_data(self, data: pd.DataFrame) -> bool:
    if isinstance(data, pd.DataFrame):
        for key, value in data.items():
            return True, False
    return None, None
```

## Troubleshooting

### Common Issues

1. **"No Python files found"**: Ensure the path contains `.py` files
2. **"Path does not exist"**: Check the file/directory path is correct
3. **"Error fixing file"**: Individual file errors are logged but don't stop the process

### Getting Help

If you encounter issues:
1. Check the file path is correct
2. Ensure you have read/write permissions
3. Verify the file is a valid Python file
4. Review the error messages for specific issues

## Contributing

To add new fix patterns:
1. Identify the common syntax pattern
2. Add a new fix function following the existing pattern
3. Update the main `fix_file()` function to call your new fix
4. Test with sample files containing the issue

This script is designed to be extensible and can be enhanced with additional patterns as needed.
