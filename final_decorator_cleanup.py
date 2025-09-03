#!/usr/bin/env python3
"""
Final cleanup script to update all remaining centralized_decorators imports.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


def extract_decorator_imports(import_block: str) -> List[str]:
    """Extract individual decorator names from import block."""
    # Handle multiline imports
    imports = re.findall(r'(\w+)(?:\s*,)?', import_block)
    return [imp.strip() for imp in imports if imp.strip()]


def categorize_imports(imports: List[str]) -> Tuple[List[str], List[str]]:
    """Categorize imports into core and domain decorators."""
    core_imports = []
    domain_imports = []
    
    # Domain decorator patterns
    domain_patterns = [
        'ValidationLevel', 'PerformanceLevel', 'PipelineStage', 'PipelineValidationLevel',
        'validate_', 'monitor_', 'secure_', 'prevent_data_leakage', 'ensure_data_integrity',
        'quality_gate', 'comprehensive_validation', 'artifact_versioning', 'deterministic_seed',
        'idempotent_step', 'time_budget_watchdog', 'smart_validation_cache', 'enforce_ndarray',
        'optimize_memory_usage', 'comprehensive_data_validation'
    ]
    
    for imp in imports:
        is_domain = False
        for pattern in domain_patterns:
            if pattern.endswith('_') and imp.startswith(pattern):
                is_domain = True
                break
            elif imp == pattern:
                is_domain = True
                break
        
        if is_domain:
            domain_imports.append(imp)
        else:
            # Unknown imports - add to domain for now
            domain_imports.append(imp)
    
    return core_imports, domain_imports


def process_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Process a single file and return whether it was modified and any issues."""
    issues = []
    modified = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Find all centralized_decorators imports
        pattern = r'from src\.utils\.centralized_decorators import \(([\s\S]*?)\)'
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
        
        # Process matches in reverse order
        for match in reversed(matches):
            imports_text = match.group(1)
            imports = extract_decorator_imports(imports_text)
            
            if not imports:
                continue
            
            core_imports, domain_imports = categorize_imports(imports)
            
            # Build replacement
            replacement_lines = []
            
            if domain_imports:
                if len(domain_imports) == 1:
                    replacement_lines.append(f"from src.core.domain import {domain_imports[0]}")
                else:
                    replacement_lines.append("from src.core.domain import (")
                    for imp in sorted(set(domain_imports[:-1])):
                        replacement_lines.append(f"    {imp},")
                    replacement_lines.append(f"    {domain_imports[-1]}")
                    replacement_lines.append(")")
            
            replacement = '\n'.join(replacement_lines)
            
            # Replace the import
            start, end = match.span()
            content = content[:start] + replacement + content[end:]
        
        # Also handle single-line imports
        pattern = r'from src\.utils\.centralized_decorators import\s+(\w+(?:\s*,\s*\w+)*)'
        matches = list(re.finditer(pattern, content))
        
        for match in reversed(matches):
            imports_text = match.group(1)
            imports = [imp.strip() for imp in imports_text.split(',')]
            
            core_imports, domain_imports = categorize_imports(imports)
            
            if domain_imports:
                replacement = f"from src.core.domain import {', '.join(domain_imports)}"
                start, end = match.span()
                content = content[:start] + replacement + content[end:]
        
        # Clean up any duplicate imports
        lines = content.split('\n')
        seen_imports = set()
        cleaned_lines = []
        
        for line in lines:
            if line.strip().startswith('from src.core.domain import'):
                if line in seen_imports:
                    continue  # Skip duplicate
                seen_imports.add(line)
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            modified = True
            
    except Exception as e:
        issues.append(f"Error processing {filepath}: {str(e)}")
    
    return modified, issues


def main():
    """Main cleanup function."""
    print("Starting final decorator cleanup...")
    
    # Find all Python files that still import from centralized_decorators
    files_to_update = []
    
    for root, dirs, files in os.walk('src/'):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                # Skip the centralized_decorators file itself
                if 'centralized_decorators' in str(filepath):
                    continue
                
                # Check if file contains old imports
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if 'from src.utils.centralized_decorators import' in f.read():
                            files_to_update.append(filepath)
                except:
                    pass
    
    print(f"Found {len(files_to_update)} files to update...")
    
    modified_count = 0
    all_issues = []
    
    for filepath in sorted(files_to_update):
        modified, issues = process_file(filepath)
        if modified:
            modified_count += 1
            print(f"✓ Updated: {filepath}")
        if issues:
            all_issues.extend(issues)
    
    print(f"\n{'='*60}")
    print(f"Cleanup complete!")
    print(f"Files modified: {modified_count}")
    
    if all_issues:
        print(f"\nIssues encountered:")
        for issue in all_issues:
            print(f"  - {issue}")
    
    print(f"\nFinal steps:")
    print(f"1. Review all changes")
    print(f"2. Run comprehensive tests")
    print(f"3. Remove old decorator modules from src/utils/")


if __name__ == "__main__":
    main()