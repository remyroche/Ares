#!/usr/bin/env python3
"""
Comprehensive script to fix all silent failures by replacing bare except clauses
with proper error handling using tprint logging.
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Tuple

def find_bare_except_clauses(content: str) -> List[Tuple[int, int, str]]:
    """Find all bare except clauses in content."""
    bare_except_pattern = r'\bexcept:\s*\n'
    matches = []
    
    for match in re.finditer(bare_except_pattern, content):
        start = match.start()
        end = match.end()
        
        # Get the line with the except clause
        line_start = content.rfind('\n', 0, start) + 1
        line_end = content.find('\n', end)
        if line_end == -1:
            line_end = len(content)
        
        except_line = content[line_start:line_end].strip()
        
        # Check if this is truly a bare except (not followed by specific exception type)
        context_before = content[max(0, start-100):start]
        if not re.search(r'except\s+(Exception|ImportError|ValueError|TypeError|RuntimeError|OSError|IOError|KeyError|IndexError|AttributeError|NameError|ZeroDivisionError|FileNotFoundError|ConnectionError|TimeoutError|PermissionError|MemoryError|NotImplementedError)\s+as', context_before + 'except:'):
            matches.append((start, end, except_line))
    
    return matches

def add_tprint_import(content: str) -> str:
    """Add tprint import if not already present."""
    tprint_import_pattern = r'from src\.utils\.tprint import.*'
    
    if re.search(tprint_import_pattern, content):
        return content
    
    # Find the last import line
    import_pattern = r'(^import.*|^from.*import.*)'
    lines = content.split('\n')
    
    last_import_idx = -1
    for i, line in enumerate(lines):
        if re.match(import_pattern, line):
            last_import_idx = i
    
    if last_import_idx >= 0:
        # Add tprint import after the last import
        tprint_import = '''

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)'''
        lines.insert(last_import_idx + 1, tprint_import)
        return '\n'.join(lines)
    
    return content

def fix_bare_except_clause(content: str, start: int, end: int, except_line: str) -> str:
    """Fix a single bare except clause."""
    # Get context around the except clause
    context_start = max(0, start - 100)
    context_end = min(len(content), end + 100)
    context = content[context_start:context_end]
    
    # Find the exact except clause with more context
    except_start = context.find('except:')
    if except_start >= 0:
        # Get the line with the except clause
        line_start = context[:except_start].rfind('\n') + 1
        line_end = context.find('\n', except_start)
        if line_end == -1:
            line_end = len(context)
        
        except_line = context[line_start:line_end].strip()
        
        # Determine the appropriate logging level and message
        if 'pass' in context[except_start:except_start+50]:
            # If it's just pass, use debug logging
            replacement = except_line.replace('except:', 'except Exception as e:') + '\n' + ' ' * (except_start - line_start) + '                    tprint_debug(f"🔍 Operation failed: {e}")\n' + ' ' * (except_start - line_start) + '                    pass'
        else:
            # If it has other code, use warning logging
            replacement = except_line.replace('except:', 'except Exception as e:') + '\n' + ' ' * (except_start - line_start) + '                    tprint_warning(f"⚠️ Operation failed: {e}")'
        
        return content.replace(except_line, replacement)
    
    return content

def fix_file_silent_failures(file_path: str) -> Tuple[bool, int]:
    """Fix silent failures in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all bare except clauses
        bare_matches = find_bare_except_clauses(content)
        
        if not bare_matches:
            return False, 0
        
        print(f"🔍 Found {len(bare_matches)} bare except clauses in {file_path}")
        
        # Add tprint import if not present
        content = add_tprint_import(content)
        
        # Fix each bare except clause
        for start, end, except_line in bare_matches:
            content = fix_bare_except_clause(content, start, end, except_line)
        
        # Write the fixed content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {len(bare_matches)} bare except clauses in {file_path}")
        return True, len(bare_matches)
        
    except Exception as e:
        from src.utils.tprint import tprint_error
        tprint_error(f"❌ Error processing {file_path}: {e}")
        return False, 0

def fix_all_silent_failures():
    """Fix all silent failures in the codebase."""
    # Get all Python files
    python_files = []
    
    # Search in common directories
    search_dirs = [
        "/workspace/src",
        "/workspace/code_quality",
        "/workspace/core",
        "/workspace/tests",
        "/workspace/examples"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
    
    # Also check root directory
    for file in os.listdir("/workspace"):
        if file.endswith('.py'):
            python_files.append(os.path.join("/workspace", file))
    
    files_fixed = 0
    exceptions_fixed = 0
    
    print(f"🔍 Scanning {len(python_files)} Python files for silent failures...")
    
    for file_path in python_files:
        try:
            fixed, count = fix_file_silent_failures(file_path)
            if fixed:
                files_fixed += 1
                exceptions_fixed += count
        except Exception as e:
            from src.utils.tprint import tprint_error
            tprint_error(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n🎉 Fixed {exceptions_fixed} bare except clauses in {files_fixed} files")
    
    # Verify no more bare except clauses remain
    print("\n🔍 Verifying fixes...")
    remaining_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            bare_matches = find_bare_except_clauses(content)
            if bare_matches:
                remaining_files.append((file_path, len(bare_matches)))
        except Exception:
            pass
    
    if remaining_files:
        print(f"⚠️ {len(remaining_files)} files still have bare except clauses:")
        for file_path, count in remaining_files[:10]:  # Show first 10
            print(f"  - {file_path}: {count} bare except clauses")
        if len(remaining_files) > 10:
            print(f"  ... and {len(remaining_files) - 10} more files")
    else:
        print("✅ No bare except clauses found! All silent failures have been fixed.")

if __name__ == "__main__":
    fix_all_silent_failures()