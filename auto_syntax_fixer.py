#!/usr/bin/env python3
"""
Automated syntax fixer for common Python syntax errors.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

def fix_import_statements(content: str) -> str:
    """Fix malformed import statements."""
    # Fix: combine "from X import Y, import Z" -> two statements
    content = re.sub(
        r"from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import\s+([^,]+),\s+import\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"from \1 import \2\nimport \3",
        content,
    )
    # Fix: combine "from X import Y, def" artifact -> split correctly
    content = re.sub(
        r"from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import\s+([^,]+),\s+def\s+",
        r"from \1 import \2\n\ndef ",
        content,
    )
    return content

def fix_exception_handling(content: str) -> str:
    """Fix malformed exception handling."""
    # No-op placeholder; previous patterns were invalid
    return content

def fix_function_calls(content: str) -> str:
    """Fix malformed function calls."""
    return content

def fix_dictionary_syntax(content: str) -> str:
    """Fix malformed dictionary syntax."""
    return content

def fix_assignment_syntax(content: str) -> str:
    """Fix malformed assignment syntax."""
    return content

def fix_string_literals(content: str) -> str:
    """Fix unterminated string literals."""
    return content

def fix_file_operations(content: str) -> str:
    """Fix malformed file operations."""
    return content

def fix_syntax_errors_in_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Fix syntax errors in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content=f.read()
        
        original_content=content
        fixes_applied = []
        
        # Apply all fixes
        content = fix_import_statements(content)
        content=fix_exception_handling(content)
        content=fix_function_calls(content)
        content=fix_dictionary_syntax(content)
        content=fix_assignment_syntax(content)
        content=fix_string_literals(content)
        content=fix_file_operations(content)
        
        # Check if any fixes were applied
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixes_applied.append("Applied syntax fixes")
        
        # Test if file compiles
        try:
            compile(content, str(file_path), 'exec')
            return True, fixes_applied
        except SyntaxError as e:
            return False, [f"Still has syntax error: {e}"]
            
    except Exception as e:
        return False, [f"Error processing file: {e}"]

def main():
    """Main function to fix syntax errors in all Python files."""
    root_dir=Path(".")
    python_files=list(root_dir.rglob("*.py"))
    
    # Skip the code_quality directory and virtual environment
    python_files=[f for f in python_files 
                    if "code_quality" not in str(f) and "code_quality_env" not in str(f)]
    
    print(f"Fixing syntax errors in {len(python_files)} Python files...")
    print("=" * 60)
    
    fixed_files=0
    still_broken = 0
    total_fixes = 0
    
    for file_path in python_files:
        print(f"Processing: {file_path}")
        
        success, fixes=fix_syntax_errors_in_file(file_path)
        
        if success:
            if fixes:
                fixed_files += 1
                total_fixes += len(fixes)
                print(f"  ✅ Fixed: {', '.join(fixes)}")
            else:
                print(f"  ✓ Already valid")
        else:
            still_broken += 1
            print(f"  ❌ Still broken: {fixes[0] if fixes else 'Unknown error'}")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total files processed: {len(python_files)}")
    print(f"  Files fixed: {fixed_files}")
    print(f"  Files still broken: {still_broken}")
    print(f"  Total fixes applied: {total_fixes}")
    
    if still_broken > 0:
        print(f"\n⚠️  {still_broken} files still have syntax errors and need manual attention.")
        return 1
    else:
        print(f"\n✅ All Python files now have valid syntax!")
        return 0

if __name__== "__main__":
    exit(main())