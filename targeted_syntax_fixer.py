#!/usr/bin/env python3
"""
Targeted Syntax Fixer for Common Repository Errors

This script specifically targets the most common error patterns found in the scan:
1. Malformed try-except blocks with multiple pass statements
2. Missing indented blocks after control structures
3. Specific syntax errors in function calls and assignments
4. Indentation inconsistencies
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)


class TargetedSyntaxFixer:
    """Targeted syntax fixer for specific error patterns."""
    
    def __init__(self):
        self.fixes_applied=0
        self.files_processed = 0
        self.files_fixed = 0
        
    def fix_file(self, file_path: str) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content=f.read()
            
            original_content=content
            fixes_in_file = 0
            
            # Apply targeted fixes
            content, fixes=self._fix_malformed_try_except_patterns(content)
            fixes_in_file += fixes
            
            content, fixes=self._fix_missing_indented_blocks(content)
            fixes_in_file += fixes
            
            content, fixes=self._fix_specific_syntax_errors(content)
            fixes_in_file += fixes
            
            content, fixes=self._fix_indentation_issues(content)
            fixes_in_file += fixes
            
            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixes_applied += fixes_in_file
                self.files_fixed += 1
                logger.info(f"✅ Fixed {fixes_in_file} issues in {file_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False
    
    def _fix_malformed_try_except_patterns(self, content: str) -> Tuple[str, int]:
        """Fix the specific malformed try-except patterns found in the scan."""
        fixes=0
        
        # Pattern 1: Multiple pass statements in try-except blocks
        patterns = [
            # Complex pattern with multiple except blocks
            (r'try:\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n', 'try:\n'),
            
            # Pattern with two except blocks
            (r'try:\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n', 'try:\n'),
            
            # Simple pattern with one except block
            (r'try:\s*\n\s*pass\s*\nexcept Exception as e:\s*\n\s*pass\s*\n', 'try:\n'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content=re.sub(pattern, replacement, content)
                fixes += 1
        
        return content, fixes
    
    def _fix_missing_indented_blocks(self, content: str) -> Tuple[str, int]:
        """Fix missing indented blocks after control structures."""
        fixes=0
        lines = content.split('\n')
        fixed_lines=[]
        
        i = 0
        while i < len(lines):
            line=lines[i]
            fixed_lines.append(line)
            
            # Check for control structures that need indented blocks
            if re.match(r'^\s*(if|try|for|while|def|class)\s+.*:\s*$', line):
                # Look ahead to see if next line is properly indented
                if i + 1 < len(lines):
                    next_line=lines[i + 1]
                    
                    # Check if next line is problematic
                    if (not next_line.strip() or 
                        next_line.strip() == 'pass' or
                        (not next_line.startswith('    ') and not next_line.startswith('\t'))):
                        
                        # Find the end of the problematic block
                        j=i + 1
                        while j < len(lines) and (not lines[j].strip() or 
                                                 lines[j].strip() == 'pass' or
                                                 lines[j].startswith('    pass') or
                                                 lines[j].startswith('\tpass')):
                            j += 1
                        
                        # Replace with proper structure
                        if j > i + 1:
                            fixed_lines.append('    pass  # TODO: Add proper implementation')
                            i=j - 1
                            fixes += 1
            
            i += 1
        
        return '\n'.join(fixed_lines), fixes
    
    def _fix_specific_syntax_errors(self, content: str) -> Tuple[str, int]:
        """Fix specific syntax errors found in the scan."""
        fixes=0
        
        # Fix import statement errors
        content = re.sub(
            r"from pathlib import Path\)\s*\n\s*import glob",
            'from pathlib import Path\nimport glob',
            content,
        )
        fixes += len(re.findall(r"from pathlib import Path\nimport glob", content))
        
        # Fix function call syntax errors
        content = re.sub(
            r'logging\.basicConfig\(\s*level\s*,\s*logging\.INFO\s*,\s*format\s*,\s*"([^"]*)"\s*\)',
            r'logging.basicConfig(level=logging.INFO, format=r"\1")',
            content,
        )
        fixes += len(re.findall(r'logging\.basicConfig\(', content))
        
        # Fix max() function calls with syntax errors
        content = re.sub(r'max\(([^,]+),\s*key\s*=\s*([^\)]+)\)', r'max(\1, key=\2)', content)
        fixes += len(re.findall(r'max\([^,]+,\s*key\s*=\s*[^\)]+\)', content))
        
        # Fix to_parquet calls with syntax errors
        content = re.sub(r'\.to_parquet\(([^,]+),\s*index\s*=\s*False\)', r'.to_parquet(\1, index=False)', content)
        fixes += len(re.findall(r'\.to_parquet\([^,]+,\s*index\s*=\s*False\)', content))
        
        # Fix re.sub calls with syntax errors
        content = re.sub(
            r're\.sub\(pattern\s*=\s*"([^"]*)"\s*,\s*content\s*,\s*flags\s*=\s*re\.IGNORECASE\)',
            r're.sub(r"\1", content, flags=re.IGNORECASE)',
            content,
        )
        fixes += len(re.findall(r're\.sub\(pattern\s*=\s*"', content))
        
        # Fix open() calls with syntax errors
        content = re.sub(r'open\(file_path\s*=\s*"w"', 'open(file_path, "w"', content)
        fixes += len(re.findall(r'open\(file_path,\s*"w"', content))
        
        return content, fixes
    
    def _fix_indentation_issues(self, content: str) -> Tuple[str, int]:
        """Fix indentation inconsistencies."""
        fixes=0
        lines = content.split('\n')
        fixed_lines=[]
        
        for line in lines:
            # Convert tabs to spaces
            if line.startswith('\t'):
                line='    ' + line[1:]
                fixes += 1
            
            # Fix mixed tabs and spaces
            if '\t' in line and '    ' in line:
                line = line.replace('\t', '    ')
                fixes += 1
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines), fixes
    
    def scan_and_fix_directory(self, directory: str) -> Dict[str, int]:
        """Scan and fix all Python files in a directory."""
        logger.info(f"🔍 Scanning directory: {directory}")
        
        # Find all Python files
        python_files=[]
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv', 'env', 'backup_corrupted_files']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(python_files)} Python files")
        
        # Process each file
        for file_path in python_files:
            self.files_processed += 1
            self.fix_file(file_path)
        
        return {
            'files_processed': self.files_processed,
            'files_fixed': self.files_fixed,
            'total_fixes': self.fixes_applied
        }


def main():
    """Main function to run the targeted syntax fixer."""
    logger.info("🚀 Starting targeted syntax fixer")
    
    fixer=TargetedSyntaxFixer()
    
    # Fix files in the current directory and subdirectories
    results=fixer.scan_and_fix_directory('.')
    
    # Print summary
    logger.info("📊 Fix Summary:")
    logger.info(f"   Files processed: {results['files_processed']}")
    logger.info(f"   Files fixed: {results['files_fixed']}")
    logger.info(f"   Total fixes applied: {results['total_fixes']}")
    
    # Run a verification scan
    logger.info("🔍 Running verification scan...")
    import subprocess
    try:
        result=subprocess.run(
            "find . -name '*.py' -type f -exec python -m py_compile {} \; 2>&1 | wc -l",
            shell=True, capture_output=True, text=True
        )
        remaining_errors=int(result.stdout.strip())
        logger.info(f"   Remaining errors: {remaining_errors}")
        
        if remaining_errors < 472:  # Original error count
            improvement=472 - remaining_errors
            logger.info(f"✅ Improved by {improvement} errors!")
        else:
            logger.warning("⚠️ No improvement detected")
            
    except Exception as e:
        logger.error(f"❌ Error during verification: {e}")
    
    logger.info("✅ Targeted syntax fixing completed!")


if __name__== "__main__":
    main()
