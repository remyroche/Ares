#!/usr/bin/env python3
"""
Fix syntax errors in Python files.
This script was used to fix IndentationError in placeholder_finder.py
"""

import os
import re

def fix_indentation_error(file_path):
    """Fix the IndentationError in placeholder_finder.py"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the specific indentation error around line 53-54
        # The issue was with the try-except block structure
        fixed_content = content.replace(
            '        try:\n            pass\n        except Exception as e:\n            print(f"Error analyzing {filepath}: {e}")\n            return {}',
            '        try:\n            with open(filepath, \'r\', encoding=\'utf-8\') as f:\n                content = f.read()\n\n            issues = {\n                \'pass_statements\': self._find_pass_statements(content, filepath),\n                \'todo_comments\': self._find_todo_comments(content, filepath),\n                \'raise_notimplemented\': self._find_raise_notimplemented(content, filepath),\n                \'placeholder_functions\': self._find_placeholder_functions(content, filepath)\n            }\n\n            self.stats[\'files_analyzed\'] += 1\n            self.stats[\'pass_statements\'] += len(issues[\'pass_statements\'])\n            self.stats[\'todo_comments\'] += len(issues[\'todo_comments\'])\n            self.stats[\'raise_notimplemented\'] += len(issues[\'raise_notimplemented\'])\n            self.stats[\'placeholder_functions\'] += len(issues[\'placeholder_functions\'])\n\n            total_issues = sum(len(issues[key]) for key in issues)\n            self.stats[\'total_placeholders\'] += total_issues\n\n            if total_issues > 0:\n                self.placeholders[filepath] = issues\n\n            return issues\n\n        except (UnicodeDecodeError, PermissionError) as e:\n            print(f"Error analyzing {filepath}: {e}")\n            return {}'
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed indentation error in {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix syntax errors"""
    target_file = "code_quality/tools/placeholder_finder.py"
    
    if os.path.exists(target_file):
        success = fix_indentation_error(target_file)
        if success:
            print("🎉 Syntax errors fixed successfully!")
        else:
            print("💥 Failed to fix syntax errors")
    else:
        print(f"❌ File not found: {target_file}")

if __name__ == "__main__":
    main()