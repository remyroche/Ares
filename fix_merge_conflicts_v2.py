#!/usr/bin/env python3
"""
Fix merge conflicts from decorator migration - improved version.
"""

import re
from pathlib import Path

def fix_file_imports(file_path):
    """Fix imports and merge conflicts in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # First, handle merge conflicts
    if '<<<<<<< HEAD' in content:
        content = resolve_conflicts(content)
    
    # Then fix import issues
    content = fix_imports(content)
    
    # Clean up any syntax issues
    content = clean_syntax(content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def resolve_conflicts(content):
    """Resolve merge conflicts intelligently."""
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        if lines[i].startswith('<<<<<<< HEAD'):
            # Find the conflict boundaries
            head_start = i + 1
            equals_line = None
            merge_end = None
            
            for j in range(i + 1, len(lines)):
                if lines[j].startswith('======='):
                    equals_line = j
                elif lines[j].startswith('>>>>>>>'):
                    merge_end = j
                    break
            
            if equals_line and merge_end:
                # Extract sections
                head_section = lines[head_start:equals_line]
                merge_section = lines[equals_line + 1:merge_end]
                
                # Decide which section to keep
                resolved = resolve_section(head_section, merge_section)
                result.extend(resolved)
                
                i = merge_end + 1
                continue
        
        result.append(lines[i])
        i += 1
    
    return '\n'.join(result)

def resolve_section(head_section, merge_section):
    """Intelligently resolve a conflict section."""
    # Check if this is an import conflict
    head_text = '\n'.join(head_section)
    merge_text = '\n'.join(merge_section)
    
    # If HEAD has new core decorators, keep it
    if 'from src.core.decorators import' in head_text or 'from src.core.errors import' in head_text:
        return head_section
    
    # If merge has old decorators, replace with new
    if 'from src.utils.centralized_decorators import' in merge_text or \
       'from src.utils.error_handler import' in merge_text:
        return ['from src.core.decorators import (', 
                '    handles_errors,',
                '    validates,',
                '    cached,',
                '    traced,',
                '    log_execution_time,',
                ')']
    
    # Otherwise, prefer HEAD
    return head_section

def fix_imports(content):
    """Fix import statements."""
    lines = content.split('\n')
    fixed_lines = []
    skip_next = 0
    
    for i, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue
            
        # Fix broken multi-line imports
        if 'from src.core.decorators import' in line and 'import' in line and '(' not in line:
            # Single line import - ensure it's properly formatted
            match = re.search(r'from src\.core\.decorators import (.+)', line)
            if match:
                imports = match.group(1).strip()
                fixed_lines.append(f'from src.core.decorators import {imports}')
            else:
                fixed_lines.append(line)
        elif line.strip() and not line.strip().startswith(('from ', 'import ')) and \
             i > 0 and 'import' in lines[i-1] and '(' in lines[i-1] and ')' not in lines[i-1]:
            # This might be a continuation of a broken import
            # Skip malformed lines that aren't proper import continuations
            if not re.match(r'\s*([\w_]+,?\s*)+\)?$', line):
                continue
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def clean_syntax(content):
    """Clean up syntax issues."""
    # Remove duplicate imports
    content = remove_duplicate_imports(content)
    
    # Fix orphaned import elements
    lines = content.split('\n')
    cleaned = []
    
    for i, line in enumerate(lines):
        # Skip orphaned decorator names
        if line.strip() in ['PerformanceLevel,', 'handle_errors,', 'handle_specific_errors,',
                           'memory_efficient,', 'performance_monitor,', 'pipeline_checkpoint,',
                           'resource_monitor,', ')']:
            # Check if this is part of a proper import statement
            if i > 0 and ('from ' in lines[i-1] or ',' in lines[i-1]):
                cleaned.append(line)
            # Otherwise skip it
        else:
            cleaned.append(line)
    
    return '\n'.join(cleaned)

def remove_duplicate_imports(content):
    """Remove duplicate import statements."""
    lines = content.split('\n')
    seen_imports = {}
    result = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.strip().startswith('from src.core.decorators import'):
            # Collect all decorators from this import
            decorators = set()
            
            if '(' in line and ')' not in line:
                # Multi-line import
                j = i
                while j < len(lines) and ')' not in lines[j]:
                    if j > i:
                        dec_match = re.findall(r'(\w+)(?:,|$)', lines[j])
                        decorators.update(dec_match)
                    j += 1
                
                # Merge with existing if found
                key = 'src.core.decorators'
                if key in seen_imports:
                    seen_imports[key].update(decorators)
                    i = j + 1
                    continue
                else:
                    seen_imports[key] = decorators
                    for k in range(i, j + 1):
                        if k < len(lines):
                            result.append(lines[k])
                    i = j + 1
                    continue
            else:
                # Single line import
                dec_match = re.search(r'from src\.core\.decorators import (.+)', line)
                if dec_match:
                    decs = [d.strip() for d in dec_match.group(1).split(',')]
                    decorators.update(decs)
                    
                key = 'src.core.decorators'
                if key in seen_imports:
                    seen_imports[key].update(decorators)
                    i += 1
                    continue
                else:
                    seen_imports[key] = decorators
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)

def main():
    # Find all Python files with potential issues
    import subprocess
    
    # Find files with merge conflicts or import issues
    cmd = 'find /workspace/src -name "*.py" -type f | xargs grep -l -E "<<<<<<< HEAD|from src.core.decorators import.*from src.core.decorators import" | sort | uniq'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    files_to_fix = result.stdout.strip().split('\n') if result.stdout.strip() else []
    
    print(f"Found {len(files_to_fix)} files to fix\n")
    
    fixed_count = 0
    for file_path in files_to_fix:
        if file_path:
            print(f"Processing {file_path}...")
            if fix_file_imports(file_path):
                fixed_count += 1
                print(f"  ✓ Fixed")
            else:
                print(f"  - No changes needed")
    
    print(f"\nFixed {fixed_count} files.")

if __name__ == '__main__':
    main()