# Syntax Error Fix Action Plan

## Overview
37 core source files contain syntax errors that prevent code execution. This document provides a systematic approach to fix these errors.

## Error Categories

### 1. Indentation Errors (7 files)
**Error Type**: "unindent does not match any outer indentation level" or "unexpected indent"

**Affected Files**:
- `src/supervisor/global_portfolio_manager.py` (line 253)
- `src/training/steps/step14_tactician_labeling.py` (line 413)
- `src/training/enhanced_training_manager.py` (line 1774)
- `src/training/step_orchestrator.py` (line 306)
- `src/training/steps/step16_confidence_calibration.py` (line 43)
- `src/monitoring/portfolio_tracker.py` (line 1012)
- `src/monitoring/base_dashboard.py` (line 2228)

**Fix Strategy**:
```python
# Before fixing, check:
# 1. Mixed tabs and spaces
# 2. Incorrect indentation level
# 3. Missing or extra indentation

# Example fix:
# Convert all tabs to spaces
# Ensure consistent 4-space indentation
# Align with parent block structure
```

### 2. Missing Exception Blocks (11 files)
**Error Type**: "expected 'except' or 'finally' block"

**Affected Files**:
- `src/tactician/sr_weight_optimizer.py` (line 90)
- `src/tactician/sr_breakout_predictor.py` (line 1201)
- `src/training/model_trainer.py` (line 557)
- `src/training/steps/step21_saving.py` (line 237)
- `src/monitoring/risk_tracker.py` (line 1017)
- `src/monitoring/alert_manager.py` (line 228)
- `src/data_management/fetcher.py` (line 251)
- `src/data_management/kline_aggregator.py` (line 1134)
- `src/utils/helper_argparse.py` (line 55)
- `src/utils/data_utils.py` (line 1251)
- `src/utils/array_utils.py` (line 3329)

**Fix Strategy**:
```python
# Template for fixing:
try:
    # existing code
    pass
except Exception as e:
    # Add appropriate exception handling
    logger.error(f"Error occurred: {e}")
    raise
finally:
    # Add cleanup code if needed
    pass
```

### 3. Invalid Syntax (19 files)
**Error Type**: "invalid syntax"

**Common Causes**:
- Missing colons after if/for/while/def statements
- Unclosed parentheses, brackets, or quotes
- Invalid operators or keywords
- Malformed f-strings

**Fix Strategy**:
1. Check lines before the reported error (often the real issue is earlier)
2. Verify all brackets/parentheses are balanced
3. Check for missing colons
4. Validate string formatting

## Automated Fix Script

```python
#!/usr/bin/env python3
"""
Automated syntax error fixer for the project.
Run this script to attempt automatic fixes for common syntax errors.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Tuple

class SyntaxErrorFixer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.errors_fixed = 0
        self.files_processed = 0
        
    def fix_indentation(self, content: str) -> str:
        """Fix common indentation issues."""
        # Convert tabs to spaces
        content = content.replace('\t', '    ')
        
        # Fix mixed indentation
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Ensure consistent indentation
            if line.strip():
                indent_match = re.match(r'^(\s*)', line)
                if indent_match:
                    indent = indent_match.group(1)
                    # Ensure indentation is multiple of 4
                    indent_level = len(indent) // 4
                    fixed_indent = '    ' * indent_level
                    line = fixed_indent + line.lstrip()
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def fix_try_except(self, content: str) -> str:
        """Add missing except blocks after try statements."""
        lines = content.split('\n')
        fixed_lines = []
        in_try_block = False
        try_indent = 0
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # Detect try block
            if re.match(r'^(\s*)try\s*:', line):
                in_try_block = True
                try_indent = len(re.match(r'^(\s*)', line).group(1))
            
            # Check if we need to add except block
            elif in_try_block and i < len(lines) - 1:
                next_line = lines[i + 1]
                next_indent = len(re.match(r'^(\s*)', next_line).group(1)) if next_line.strip() else 0
                
                # If next line has same or less indentation, try block is ending
                if next_indent <= try_indent and not re.match(r'^(\s*)except', next_line):
                    # Add generic except block
                    fixed_lines.append(' ' * try_indent + 'except Exception as e:')
                    fixed_lines.append(' ' * (try_indent + 4) + 'logger.error(f"Error: {e}")')
                    fixed_lines.append(' ' * (try_indent + 4) + 'raise')
                    in_try_block = False
                    
        return '\n'.join(fixed_lines)
    
    def validate_syntax(self, content: str) -> List[str]:
        """Validate Python syntax and return errors."""
        try:
            ast.parse(content)
            return []
        except SyntaxError as e:
            return [f"Line {e.lineno}: {e.msg}"]
    
    def fix_file(self, filepath: Path) -> Tuple[bool, List[str]]:
        """Fix syntax errors in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply fixes
            fixed_content = self.fix_indentation(content)
            fixed_content = self.fix_try_except(fixed_content)
            
            # Validate
            errors = self.validate_syntax(fixed_content)
            
            if not errors:
                # Create backup
                backup_path = filepath.with_suffix('.bak')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Write fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return True, []
            else:
                return False, errors
                
        except Exception as e:
            return False, [str(e)]
    
    def run(self, target_files: List[str]):
        """Run the fixer on specified files."""
        for file_path in target_files:
            filepath = self.project_root / file_path
            if filepath.exists():
                success, errors = self.fix_file(filepath)
                self.files_processed += 1
                
                if success:
                    self.errors_fixed += 1
                    print(f"✅ Fixed: {file_path}")
                else:
                    print(f"❌ Failed: {file_path}")
                    for error in errors:
                        print(f"   {error}")
            else:
                print(f"⚠️  Not found: {file_path}")
        
        print(f"\nSummary: Fixed {self.errors_fixed}/{self.files_processed} files")

# Files with syntax errors (from the report)
FILES_TO_FIX = [
    "src/supervisor/global_portfolio_manager.py",
    "src/tactician/sr_weight_optimizer.py",
    "src/tactician/sr_breakout_predictor.py",
    # ... add all 37 files
]

if __name__ == "__main__":
    fixer = SyntaxErrorFixer("/workspace")
    fixer.run(FILES_TO_FIX)
```

## Manual Fix Guidelines

### For Each Error Type:

1. **Indentation Errors**:
   - Use an editor with visible whitespace
   - Enable "show indentation guides"
   - Use automatic indentation fixing tools
   - Check for mixed tabs/spaces

2. **Missing Exception Handling**:
   - Always add at least a basic except block
   - Log errors appropriately
   - Consider specific exception types
   - Add finally blocks for cleanup

3. **Invalid Syntax**:
   - Check the line before the error
   - Verify all brackets match
   - Check string quotes
   - Validate function definitions

## Testing After Fixes

1. **Syntax Validation**:
   ```bash
   python -m py_compile <filename>
   ```

2. **Import Test**:
   ```python
   import sys
   sys.path.append('/workspace/src')
   import module_name  # Should not raise SyntaxError
   ```

3. **AST Validation**:
   ```python
   import ast
   with open('file.py', 'r') as f:
       ast.parse(f.read())  # Should not raise SyntaxError
   ```

## Prevention Strategies

1. **Pre-commit Hooks**:
   ```yaml
   # .pre-commit-config.yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.4.0
       hooks:
         - id: check-ast
         - id: check-syntax
   ```

2. **IDE Configuration**:
   - Enable syntax checking
   - Use Python linters
   - Configure auto-formatting

3. **CI/CD Integration**:
   - Add syntax checking to CI pipeline
   - Fail builds on syntax errors
   - Generate reports

## Next Steps

1. Run the automated fixer script
2. Manually fix remaining errors
3. Test all fixed files
4. Implement prevention strategies
5. Set up continuous monitoring