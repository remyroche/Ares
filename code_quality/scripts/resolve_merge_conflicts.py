#!/usr/bin/env python3
"""
Script to resolve merge conflicts from code quality improvements.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Tuple


def get_conflicted_files() -> List[Path]:
    """Get list of files with merge conflicts."""
    result = subprocess.run(
        ['git', 'diff', '--name-only', '--diff-filter=U'],
        capture_output=True,
        text=True
    )
    return [Path(f) for f in result.stdout.strip().split('\n') if f]


def resolve_import_conflicts(content: str) -> str:
    """
    Resolve import conflicts by keeping both imports.
    
    Pattern: Our branch added code quality imports (asyncio, copy, datetime)
    while main may have added other imports (logging, etc).
    """
    lines = content.split('\n')
    resolved_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Detect conflict start
        if '<<<<<<< HEAD' in line:
            # Find the conflict boundaries
            head_start = i
            middle = i + 1
            while middle < len(lines) and '=======' not in lines[middle]:
                middle += 1
            
            main_start = middle + 1
            end = main_start
            while end < len(lines) and '>>>>>>> origin/main' not in lines[end]:
                end += 1
            
            if middle < len(lines) and end < len(lines):
                # Extract both versions
                head_content = lines[head_start + 1:middle]
                main_content = lines[main_start:end]
                
                # Check if this is an import conflict
                head_imports = [l for l in head_content if l.strip().startswith('import ') or l.strip().startswith('from ')]
                main_imports = [l for l in main_content if l.strip().startswith('import ') or l.strip().startswith('from ')]
                
                if head_imports or main_imports:
                    # This is an import conflict - keep both
                    # Add main imports first (to preserve original)
                    for imp in main_content:
                        if imp.strip():
                            resolved_lines.append(imp)
                    # Then add our imports if they're different
                    for imp in head_content:
                        if imp.strip() and imp not in main_content:
                            resolved_lines.append(imp)
                else:
                    # For non-import conflicts, prefer our version (HEAD)
                    # since we've been fixing code quality
                    resolved_lines.extend(head_content)
                
                # Skip to after the conflict
                i = end + 1
                continue
        
        # Normal line - keep it
        resolved_lines.append(line)
        i += 1
    
    return '\n'.join(resolved_lines)


def resolve_file(file_path: Path) -> bool:
    """Resolve conflicts in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '<<<<<<< HEAD' not in content:
            return False  # No conflicts
        
        # Resolve conflicts
        resolved = resolve_import_conflicts(content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(resolved)
        
        print(f"✓ Resolved conflicts in {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error resolving {file_path}: {e}")
        return False


def main():
    """Resolve all merge conflicts."""
    print("Resolving merge conflicts...")
    print("=" * 60)
    
    # Get conflicted files
    conflicted_files = get_conflicted_files()
    print(f"Found {len(conflicted_files)} files with conflicts")
    
    # Resolve each file
    resolved_count = 0
    for file_path in conflicted_files:
        if resolve_file(file_path):
            resolved_count += 1
    
    print(f"\nResolved {resolved_count} files")
    
    # Stage resolved files
    if resolved_count > 0:
        print("\nStaging resolved files...")
        for file_path in conflicted_files:
            subprocess.run(['git', 'add', str(file_path)])
        print("✓ All resolved files staged")
    
    # Check remaining conflicts
    remaining = subprocess.run(
        ['git', 'diff', '--name-only', '--diff-filter=U'],
        capture_output=True,
        text=True
    )
    
    if remaining.stdout.strip():
        print(f"\n⚠️  Still have conflicts in: {remaining.stdout}")
    else:
        print("\n✓ All conflicts resolved!")
        print("You can now commit the merge with: git commit")


if __name__ == '__main__':
    main()