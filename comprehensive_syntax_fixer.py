#!/usr/bin/env python3
"""
Comprehensive Syntax Fixer for Ares Repository

This script automatically fixes common Python syntax errors:
1. Unmatched parentheses, brackets, and braces
2. Indentation errors
3. Unterminated string literals
4. Missing colons after control structures
5. Invalid syntax patterns
6. Await outside async functions
"""

import os
import re
import ast
import tokenize
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SyntaxFixer:
    """Comprehensive syntax error fixer."""

    def __init__(self):
        self.fixes_applied = 0
        self.files_fixed = 0
        self.errors_fixed = 0

    def fix_unmatched_parentheses(self, content: str) -> str:
        """Fix unmatched parentheses, brackets, and braces."""
        original_content = content
        
        # Count opening and closing characters
        open_paren = content.count('(')
        close_paren = content.count(')')
        open_bracket = content.count('[')
        close_bracket = content.count(']')
        open_brace = content.count('{')
        close_brace = content.count('}')
        
        # Fix parentheses
        if open_paren > close_paren:
            content += ')' * (open_paren - close_paren)
        elif close_paren > open_paren:
            # Remove extra closing parentheses from the end
            lines = content.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                if close_paren > open_paren and lines[i].strip().endswith(')'):
                    lines[i] = lines[i].rstrip(')')
                    close_paren -= 1
            content = '\n'.join(lines)
        
        # Fix brackets
        if open_bracket > close_bracket:
            content += ']' * (open_bracket - close_bracket)
        elif close_bracket > open_bracket:
            lines = content.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                if close_bracket > open_bracket and lines[i].strip().endswith(']'):
                    lines[i] = lines[i].rstrip(']')
                    close_bracket -= 1
            content = '\n'.join(lines)
        
        # Fix braces
        if open_brace > close_brace:
            content += '}' * (open_brace - close_brace)
        elif close_brace > open_brace:
            lines = content.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                if close_brace > open_brace and lines[i].strip().endswith('}'):
                    lines[i] = lines[i].rstrip('}')
                    close_brace -= 1
            content = '\n'.join(lines)
        
        if content != original_content:
            self.fixes_applied += 1
            logger.info("Fixed unmatched parentheses/brackets/braces")
        
        return content

    def fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated string literals."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Check for unterminated strings
            quote_count = line.count('"') + line.count("'")
            if quote_count % 2 != 0:
                # Find the last quote and add a matching one
                last_single = line.rfind("'")
                last_double = line.rfind('"')
                
                if last_single > last_double:
                    # Single quote is last
                    lines[i] = line + "'"
                else:
                    # Double quote is last
                    lines[i] = line + '"'
                
                self.fixes_applied += 1
                logger.info(f"Fixed unterminated string on line {i+1}")
        
        return '\n'.join(lines)

    def fix_missing_colons(self, content: str) -> str:
        """Fix missing colons after control structures."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for missing colons after control structures
            if (stripped.startswith(('if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ', 'try:', 'except', 'finally')) and 
                not stripped.endswith(':') and 
                not stripped.endswith(':')):
                
                # Don't add colon if line ends with comment or has a colon later
                if '#' not in stripped and ':' not in stripped:
                    lines[i] = line + ':'
                    self.fixes_applied += 1
                    logger.info(f"Added missing colon on line {i+1}")
        
        return '\n'.join(lines)

    def fix_indentation_errors(self, content: str) -> str:
        """Fix common indentation errors."""
        lines = content.split('\n')
        fixed_lines = []
        indent_stack = [0]  # Track indentation levels
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:  # Empty line
                fixed_lines.append(line)
                continue
            
            # Calculate current indentation
            current_indent = len(line) - len(line.lstrip())
            
            # Check for common indentation issues
            if stripped.startswith(('def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ')):
                # These should be at the current indent level or start a new block
                expected_indent = indent_stack[-1]
                if current_indent < expected_indent:
                    # Fix indentation
                    lines[i] = ' ' * expected_indent + stripped
                    self.fixes_applied += 1
                    logger.info(f"Fixed indentation on line {i+1}")
            
            # Update indent stack for next line
            if stripped.endswith(':'):
                indent_stack.append(indent_stack[-1] + 4)
            elif stripped in ['pass', 'break', 'continue', 'return']:
                # These don't increase indentation
                pass
            else:
                # Check if we should decrease indentation
                if current_indent < indent_stack[-1]:
                    indent_stack.pop()
        
        return '\n'.join(lines)

    def fix_await_outside_async(self, content: str) -> str:
        """Fix 'await' outside async function by wrapping in async function."""
        if 'await ' in content and 'async def' not in content:
            # Simple fix: wrap the file content in an async function
            lines = content.split('\n')
            fixed_lines = ['async def main():']
            
            for line in lines:
                if line.strip() and not line.strip().startswith('#'):
                    fixed_lines.append('    ' + line)
                else:
                    fixed_lines.append(line)
            
            fixed_lines.append('')
            fixed_lines.append('if __name__ == "__main__":')
            fixed_lines.append('    import asyncio')
            fixed_lines.append('    asyncio.run(main())')
            
            self.fixes_applied += 1
            logger.info("Fixed await outside async function")
            return '\n'.join(fixed_lines)
        
        return content

    def fix_positional_after_keyword(self, content: str) -> str:
        """Fix positional arguments after keyword arguments."""
        # This is a complex fix that requires parsing
        # For now, we'll just log the issue
        if re.search(r'=\w+\s+\w+\s*[^=]', content):
            logger.warning("Found potential positional after keyword argument - manual review needed")
        
        return content

    def fix_invalid_syntax_patterns(self, content: str) -> str:
        """Fix common invalid syntax patterns."""
        original_content = content
        
        # Fix common patterns
        content = re.sub(r'==\s*=\s*', '== ', content)  # Fix ===
        content = re.sub(r'!=\s*=\s*', '!= ', content)   # Fix !==
        content = re.sub(r'&&', 'and', content)           # Fix && to and
        content = re.sub(r'\|\|', 'or', content)         # Fix || to or
        
        # Fix common typos
        content = re.sub(r'\bTrue\s*=\s*', 'True == ', content)
        content = re.sub(r'\bFalse\s*=\s*', 'False == ', content)
        content = re.sub(r'\bNone\s*=\s*', 'None == ', content)
        
        if content != original_content:
            self.fixes_applied += 1
            logger.info("Fixed invalid syntax patterns")
        
        return content

    def fix_file(self, file_path: str) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply fixes
            content = self.fix_unterminated_strings(content)
            content = self.fix_unmatched_parentheses(content)
            content = self.fix_missing_colons(content)
            content = self.fix_invalid_syntax_patterns(content)
            content = self.fix_await_outside_async(content)
            content = self.fix_positional_after_keyword(content)
            
            # Try to fix indentation errors
            try:
                content = self.fix_indentation_errors(content)
            except Exception as e:
                logger.warning(f"Could not fix indentation in {file_path}: {e}")
            
            # Verify the fix worked by trying to parse
            try:
                ast.parse(content)
                # If we get here, the syntax is valid
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.files_fixed += 1
                    logger.info(f"✅ Fixed syntax errors in {file_path}")
                    return True
                else:
                    logger.info(f"ℹ️  No fixes needed for {file_path}")
                    return False
            except SyntaxError as e:
                logger.warning(f"⚠️  Could not fix all syntax errors in {file_path}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False

    def scan_and_fix_directory(self, directory: str) -> Dict:
        """Scan and fix all Python files in a directory."""
        logger.info(f"🔧 Starting syntax fixes in: {directory}")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv', 'env', 'backup_']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"📁 Found {len(python_files)} Python files")
        
        # Fix each file
        for file_path in python_files:
            self.fix_file(file_path)
        
        return {
            'files_processed': len(python_files),
            'files_fixed': self.files_fixed,
            'fixes_applied': self.fixes_applied
        }


def main():
    """Main function to run the syntax fixer."""
    logger.info("🚀 Starting comprehensive syntax fixer")
    
    fixer = SyntaxFixer()
    
    # Fix files in current directory
    results = fixer.scan_and_fix_directory('.')
    
    # Print summary
    logger.info("📊 Fix Summary:")
    logger.info(f"   Files processed: {results['files_processed']}")
    logger.info(f"   Files fixed: {results['files_fixed']}")
    logger.info(f"   Total fixes applied: {results['fixes_applied']}")
    
    logger.info("✅ Syntax fixing completed!")


if __name__ == "__main__":
    main()