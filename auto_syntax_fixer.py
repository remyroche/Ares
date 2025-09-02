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
    # Fix: from typing import Any
import argparse -> from typing import Any; import argparse
    content = re.sub(r'from\s+(\w+(?:\.\w+')*)\s+import\s+([^,]+),\s+import\s+(\w+)',
        r'from \1 import \2\nimport \3',
        content
    )
    
    # Fix: from pathlib import Path

def function -> from pathlib import Path\n\ndef function
    content = re.sub(r'from\s+(\w+(?:\.\w+')*)\s+import\s+([^,]+),\s+def\s+',
        r'from \1 import \2\n\ndef ',
        content
    )
    
    # Fix: from sklearn.metrics.pairwise import cosine_similarity
import json
    content = re.sub(r'from\s+(\w+(?:\.\w+')*)\s+import\s+([^,]+),\s+import\s+(\w+)',
        r'from \1 import \2\nimport \3',
        content
    )
    
    return content

def fix_exception_handling(content: str) -> str:
    """Fix malformed exception handling."""
    # Fix: except (ValueError, TypeError, KeyError) -> except (ValueError, TypeError, KeyError)
    content = re.sub(r'except\s*\(\s*(\w+')\s*=\s*(\w+)',
        r'except (\1, \2',
        content
    )
    
    return content

def fix_function_calls(content: str) -> str:
    """Fix malformed function calls."""
    # Fix: function(param=value) -> function(param=value)
    content = re.sub(r'(\w+')\s*=\s*([^,)]+)(?=\s*[,)])',
        r'\1=\2',
        content
    )
    
    # Fix: function(param=param) -> function(param=param)
    content = re.sub(r'(\w+')\s*=\s*(?=\s*[,)])',
        r'\1=\1',
        content
    )
    
    return content

def fix_dictionary_syntax(content: str) -> str:
    """Fix malformed dictionary syntax."""
    # Fix: "key": value=} -> "key": value}
    content = re.sub(r'([^,{]\s*=\s*')(?=\s*})',
        r'\1',
        content
    )
    
    # Fix: "key": value, -> "key": value,
    content = re.sub(r'([^,{]\s*')\s+,\s*',
        r'\1, ',
        content
    )
    
    return content

def fix_assignment_syntax(content: str) -> str:
    """Fix malformed assignment syntax."""
    # Fix: return existing_files, missing_files -> return existing_files, missing_files
    content = re.sub(r'return\s+(\w+')\s*=\s*(\w+)',
        r'return \1, \2',
        content
    )
    
    # Fix: variable=value = -> variable = value
    content = re.sub(r'(\w+')\s*=\s*([^=]+)\s*=\s*(?=\s*[,)])',
        r'\1=\2',
        content
    )
    
    return content

def fix_string_literals(content: str) -> str:
    """Fix unterminated string literals."""
    # Fix: content=re.sub(r'from pathlib import Path -> content = re.sub(r'from pathlib import Path'
    content = re.sub(
        r"content\s*=\s*re\.sub\s*\(\s*r'([^']*?)(?=\s*\)|$)",
        r"content=re.sub(r'\1'",
        content
    )
    
    return content

def fix_file_operations(content: str) -> str:
    """Fix malformed file operations."""
    # Fix: with open(file, "w") -> with open(file, "w")
    content = re.sub(r'with\s+open\s*\(\s*(\w+')\s*=\s*([^)]+)\s*\)',
        r'with open(\1, \2)',
        content
    )
    
    # Fix: json.dump(data, f, indent=2) -> json.dump(data, f, indent=2)
    content = re.sub(r'json\.dump\s*\(\s*(\w+')\s*=\s*(\w+)\s*,\s*(\w+)',
        r'json.dump(\1, \2, \3',
        content
    )
    
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