#!/usr/bin/env python3
"""
Script to migrate from centralized_decorators to new domain decorators.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Update patterns to replace TODO imports with proper domain imports
UPDATE_PATTERNS = [
    # Pattern to find TODO comment blocks with centralized_decorators imports
    (
        r'# TODO: These decorators need to be migrated to core decorators or removed\s*\n'
        r'from src\.utils\.centralized_decorators import \(([\s\S]*?)\)',
        'process_todo_import'
    ),
]


def process_todo_import(match_text: str, imports_text: str) -> str:
    """Process TODO import blocks and replace with domain decorators."""
    # Extract individual imports
    imports = re.findall(r'(\w+)(?:\s*,)?', imports_text)
    
    # Map imports to new locations
    core_imports = []
    domain_imports = []
    
    for imp in imports:
        imp = imp.strip()
        if not imp:
            continue
            
        # Check if it's a core decorator or domain decorator
        if imp in ['ValidationLevel', 'PerformanceLevel', 'PipelineStage', 'PipelineValidationLevel']:
            domain_imports.append(imp)
        elif imp.startswith('validate_') or imp.startswith('monitor_') or imp.startswith('secure_'):
            domain_imports.append(imp)
        elif imp in ['prevent_data_leakage', 'ensure_data_integrity', 'quality_gate']:
            domain_imports.append(imp)
        elif imp in ['comprehensive_validation', 'artifact_versioning', 'deterministic_seed']:
            domain_imports.append(imp)
        elif imp in ['idempotent_step', 'time_budget_watchdog', 'smart_validation_cache']:
            domain_imports.append(imp)
        elif imp == 'enforce_ndarray':
            domain_imports.append(imp)
        else:
            # Unknown decorator - keep in comments
            pass
    
    # Build new import statements
    import_lines = []
    
    if domain_imports:
        if len(domain_imports) == 1:
            import_lines.append(f"from src.core.domain import {domain_imports[0]}")
        else:
            import_lines.append("from src.core.domain import (")
            for imp in sorted(set(domain_imports[:-1])):
                import_lines.append(f"    {imp},")
            import_lines.append(f"    {domain_imports[-1]}")
            import_lines.append(")")
    
    return '\n'.join(import_lines)


def process_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Process a single file and return whether it was modified and any issues."""
    issues = []
    modified = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Process each pattern
        for pattern, processor_name in UPDATE_PATTERNS:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
            
            # Process matches in reverse order to maintain positions
            for match in reversed(matches):
                if processor_name == 'process_todo_import':
                    imports_text = match.group(1)
                    replacement = process_todo_import(match.group(0), imports_text)
                    
                    # Replace the entire match
                    start, end = match.span()
                    content = content[:start] + replacement + content[end:]
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            modified = True
            
    except Exception as e:
        issues.append(f"Error processing {filepath}: {str(e)}")
    
    return modified, issues


def main():
    """Main migration function."""
    print("Starting migration to domain decorators...")
    
    # Find all Python files that have TODO comments about decorators
    files_to_update = []
    
    result = os.popen('grep -r "TODO: These decorators need to be migrated" src/ --include="*.py" -l').read()
    if result:
        files_to_update = [Path(f.strip()) for f in result.strip().split('\n') if f.strip()]
    
    print(f"Found {len(files_to_update)} files to update...")
    
    modified_count = 0
    all_issues = []
    
    for filepath in files_to_update:
        modified, issues = process_file(filepath)
        if modified:
            modified_count += 1
            print(f"✓ Updated: {filepath}")
        if issues:
            all_issues.extend(issues)
    
    print(f"\n{'='*60}")
    print(f"Migration complete!")
    print(f"Files modified: {modified_count}")
    
    if all_issues:
        print(f"\nIssues encountered:")
        for issue in all_issues:
            print(f"  - {issue}")
    
    print(f"\nNext steps:")
    print(f"1. Review the changes to ensure correctness")
    print(f"2. Run tests to verify functionality")
    print(f"3. Remove any remaining unused imports")


if __name__ == "__main__":
    main()