#!/usr/bin/env python3
"""
Script to resolve merge conflicts by intelligently merging decorator imports.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple, Optional


def extract_conflict_sections(content: str) -> List[Tuple[str, str, str]]:
    """Extract conflict sections from file content."""
    conflicts = []
    pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]+'
    
    for match in re.finditer(pattern, content, re.DOTALL):
        ours = match.group(1)
        theirs = match.group(2)
        full_match = match.group(0)
        conflicts.append((full_match, ours, theirs))
    
    return conflicts


def merge_imports(ours: str, theirs: str) -> str:
    """Intelligently merge import statements."""
    # Extract all imports from both sections
    our_imports = set()
    their_imports = set()
    
    # Pattern to find imports
    import_pattern = r'from\s+([\w.]+)\s+import\s+(?:\(([\s\S]*?)\)|(\w+))'
    
    # Extract our imports
    for match in re.finditer(import_pattern, ours, re.MULTILINE | re.DOTALL):
        module = match.group(1)
        if match.group(2):  # Multiline import
            imports = re.findall(r'(\w+)', match.group(2))
            for imp in imports:
                our_imports.add((module, imp))
        else:  # Single import
            our_imports.add((module, match.group(3)))
    
    # Extract their imports
    for match in re.finditer(import_pattern, theirs, re.MULTILINE | re.DOTALL):
        module = match.group(1)
        if match.group(2):  # Multiline import
            imports = re.findall(r'(\w+)', match.group(2))
            for imp in imports:
                their_imports.add((module, imp))
        else:  # Single import
            their_imports.add((module, match.group(3)))
    
    # Merge imports, prioritizing new decorator system
    merged_imports = {}
    
    # Process our imports (new decorator system)
    for module, imp in our_imports:
        if module not in merged_imports:
            merged_imports[module] = set()
        merged_imports[module].add(imp)
    
    # Add any additional imports from theirs that aren't decorators
    old_decorator_modules = [
        'src.utils.centralized_decorators',
        'src.utils.training_pipeline_decorators',
        'src.utils.validation_decorators',
        'src.utils.enhanced_validation_decorators',
        'src.utils.decorators',
        'src.utils.error_handler',
    ]
    
    for module, imp in their_imports:
        if module not in old_decorator_modules:
            if module not in merged_imports:
                merged_imports[module] = set()
            merged_imports[module].add(imp)
    
    # Build merged import statements
    import_lines = []
    
    # Sort modules, with core.decorators and core.domain first
    sorted_modules = sorted(merged_imports.keys(), key=lambda x: (
        0 if x == 'src.core.decorators' else
        1 if x == 'src.core.domain' else
        2 if x.startswith('src.core') else
        3
    ))
    
    for module in sorted_modules:
        imports = sorted(list(merged_imports[module]))
        if len(imports) == 1:
            import_lines.append(f"from {module} import {imports[0]}")
        else:
            import_lines.append(f"from {module} import (")
            for imp in imports[:-1]:
                import_lines.append(f"    {imp},")
            import_lines.append(f"    {imports[-1]}")
            import_lines.append(")")
    
    return '\n'.join(import_lines)


def resolve_file_conflicts(filepath: Path) -> bool:
    """Resolve conflicts in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '<<<<<<< HEAD' not in content:
            return False
        
        conflicts = extract_conflict_sections(content)
        
        for full_match, ours, theirs in conflicts:
            # Check if this is an import conflict
            if 'import' in ours or 'import' in theirs:
                # Merge imports intelligently
                merged = merge_imports(ours, theirs)
                content = content.replace(full_match, merged)
            else:
                # For non-import conflicts, prefer our changes (new decorator usage)
                # but check if there's new functionality in theirs
                if '@' in ours or '@' in theirs:
                    # Decorator usage - use ours
                    content = content.replace(full_match, ours)
                else:
                    # Other code - try to preserve new functionality
                    # For now, just use theirs if it's longer (likely has new code)
                    if len(theirs.strip()) > len(ours.strip()):
                        content = content.replace(full_match, theirs)
                    else:
                        content = content.replace(full_match, ours)
        
        # Write resolved content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error resolving {filepath}: {str(e)}")
        return False


def main():
    """Main function to resolve all conflicts."""
    print("Resolving merge conflicts...")
    
    # Get list of conflicted files
    result = os.popen('git diff --name-only --diff-filter=U').read()
    conflicted_files = [Path(f.strip()) for f in result.strip().split('\n') if f.strip()]
    
    print(f"Found {len(conflicted_files)} conflicted files")
    
    resolved_count = 0
    for filepath in conflicted_files:
        if resolve_file_conflicts(filepath):
            resolved_count += 1
            print(f"✓ Resolved: {filepath}")
            # Add the resolved file
            os.system(f'git add "{filepath}"')
        else:
            print(f"✗ Failed to resolve: {filepath}")
    
    print(f"\nResolved {resolved_count}/{len(conflicted_files)} files")
    
    # Check remaining conflicts
    remaining = os.popen('git diff --name-only --diff-filter=U').read().strip()
    if remaining:
        print("\nRemaining conflicts:")
        print(remaining)
        print("\nThese files need manual resolution.")
    else:
        print("\nAll conflicts resolved!")
        print("You can now commit the merge.")


if __name__ == "__main__":
    main()