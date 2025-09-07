#!/usr/bin/env python3
"""Comprehensive syntax fixer for common Python syntax errors."""

import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any

class ComprehensiveSyntaxFixer:
    """Fixes common Python syntax errors automatically."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        self.errors_fixed = {}
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_applied = 0
            
            # Fix 1: Empty try blocks - add pass
            content = self._fix_empty_try_blocks(content)
            if content != original_content:
                fixes_applied += 1
                original_content = content
            
            # Fix 2: Missing except/finally blocks
            content = self._fix_missing_except_finally(content)
            if content != original_content:
                fixes_applied += 1
                original_content = content
            
            # Fix 3: Malformed import statements
            content = self._fix_malformed_imports(content)
            if content != original_content:
                fixes_applied += 1
                original_content = content
            
            # Fix 4: Unexpected indentation
            content = self._fix_unexpected_indentation(content)
            if content != original_content:
                fixes_applied += 1
                original_content = content
            
            # Fix 5: Invalid syntax in function definitions
            content = self._fix_invalid_syntax(content)
            if content != original_content:
                fixes_applied += 1
                original_content = content
            
            # Fix 6: Arguments cannot follow var-keyword argument
            content = self._fix_var_keyword_arguments(content)
            if content != original_content:
                fixes_applied += 1
                original_content = content
            
            if fixes_applied > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Verify the fix worked
                result = subprocess.run(
                    ['python3', '-m', 'py_compile', str(file_path)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.fixes_applied += fixes_applied
                    self.errors_fixed[str(file_path)] = fixes_applied
                    print(f"✅ Fixed {fixes_applied} issues in {file_path}")
                    return True
                else:
                    print(f"❌ Fixes didn't resolve syntax errors in {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False
    
    def _fix_empty_try_blocks(self, content: str) -> str:
        """Fix empty try blocks by adding pass."""
        # Pattern: try:\n    except
        pattern = r'try:\s*\n(\s*)except'
        replacement = r'try:\n\1    pass\n\1except'
        return re.sub(pattern, replacement, content)
    
    def _fix_missing_except_finally(self, content: str) -> str:
        """Fix missing except/finally blocks."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for try without except/finally
            if re.match(r'(\s*)try:\s*$', line):
                indent = re.match(r'(\s*)', line).group(1)
                fixed_lines.append(line)
                i += 1
                
                # Look for the next non-empty line
                while i < len(lines) and not lines[i].strip():
                    fixed_lines.append(lines[i])
                    i += 1
                
                # If next line is not except/finally, add pass
                if i < len(lines):
                    next_line = lines[i]
                    if not re.match(r'\s*(except|finally)', next_line):
                        fixed_lines.append(f"{indent}    pass")
                        # Don't increment i, process the current line again
                    else:
                        fixed_lines.append(next_line)
                        i += 1
                else:
                    # End of file, add pass
                    fixed_lines.append(f"{indent}    pass")
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_malformed_imports(self, content: str) -> str:
        """Fix malformed import statements."""
        lines = content.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for malformed import like "from module import (\nimport other"
            if re.match(r'(\s*)from\s+\w+.*import\s*\(\s*$', line):
                indent = re.match(r'(\s*)', line).group(1)
                fixed_lines.append(line)
                i += 1
                
                # Collect import lines until we find the closing )
                import_lines = []
                while i < len(lines):
                    current_line = lines[i]
                    if current_line.strip() == ')':
                        break
                    elif current_line.strip().startswith('import '):
                        # This is a malformed import, move it outside
                        import_lines.append(current_line.strip())
                    else:
                        import_lines.append(current_line)
                    i += 1
                
                # Add the import lines before the from import
                for import_line in import_lines:
                    if import_line.strip().startswith('import '):
                        fixed_lines.insert(-1, import_line)
                    else:
                        fixed_lines.append(import_line)
                
                # Add the closing )
                if i < len(lines):
                    fixed_lines.append(lines[i])
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)
    
    def _fix_unexpected_indentation(self, content: str) -> str:
        """Fix unexpected indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix lines that have unexpected indentation
            if re.match(r'^\s{1,3}[^\s]', line) and not re.match(r'^\s{4}', line):
                # This line has 1-3 spaces but should probably be unindented
                if line.strip() and not line.strip().startswith('#'):
                    # Remove leading whitespace
                    fixed_lines.append(line.lstrip())
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_invalid_syntax(self, content: str) -> str:
        """Fix common invalid syntax patterns."""
        # Fix function definitions with invalid syntax
        content = re.sub(r'def\s+(\w+)\s*\(\s*\)\s*:\s*$', r'def \1():\n    pass', content, flags=re.MULTILINE)
        
        # Fix class definitions with invalid syntax
        content = re.sub(r'class\s+(\w+)\s*\(\s*\)\s*:\s*$', r'class \1:\n    pass', content, flags=re.MULTILINE)
        
        return content
    
    def _fix_var_keyword_arguments(self, content: str) -> str:
        """Fix 'arguments cannot follow var-keyword argument' errors."""
        # This is a complex fix that would require parsing the function signature
        # For now, we'll just add a comment to mark the issue
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if '**kwargs' in line and ',' in line:
                # Check if there are arguments after **kwargs
                parts = line.split('**kwargs')
                if len(parts) > 1 and parts[1].strip().startswith(','):
                    # This is likely the issue - move **kwargs to the end
                    before_kwargs = parts[0]
                    after_kwargs = parts[1]
                    
                    # Extract the arguments after **kwargs
                    after_args = after_kwargs.strip().lstrip(',').strip()
                    if after_args:
                        # Move **kwargs to the end
                        new_line = before_kwargs + after_args + ', **kwargs'
                        fixed_lines.append(new_line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_directory(self, directory: Path) -> Dict[str, Any]:
        """Fix syntax errors in all Python files in a directory."""
        print(f"🔧 Starting comprehensive syntax fixing in {directory}")
        
        python_files = list(directory.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            self.files_processed += 1
            self.fix_file(file_path)
        
        return {
            "files_processed": self.files_processed,
            "total_fixes_applied": self.fixes_applied,
            "files_with_fixes": len(self.errors_fixed),
            "fixes_by_file": self.errors_fixed
        }

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Python syntax fixer")
    parser.add_argument("--directory", type=str, default=".", help="Directory to fix")
    parser.add_argument("--file", type=str, help="Single file to fix")
    
    args = parser.parse_args()
    
    fixer = ComprehensiveSyntaxFixer()
    
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            fixer.fix_file(file_path)
        else:
            print(f"File not found: {args.file}")
    else:
        directory = Path(args.directory)
        results = fixer.fix_directory(directory)
        
        print(f"\n📊 Summary:")
        print(f"  Files processed: {results['files_processed']}")
        print(f"  Total fixes applied: {results['total_fixes_applied']}")
        print(f"  Files with fixes: {results['files_with_fixes']}")

if __name__ == "__main__":
    main()
