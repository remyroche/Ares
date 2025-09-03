#!/usr/bin/env python3
"""
Script to migrate from old decorators to new core decorators.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Mapping of old decorator imports to new ones
DECORATOR_MAPPING = {
    # Error handling
    "handle_errors": "handles_errors",
    "handle_file_operations": "handles_errors",
    "handle_specific_errors": "handles_errors",
    
    # Validation
    "validate_data_quality": "validates",
    "validate_step_prerequisites": "validates",
    "validate_step_output": "validates",
    "validate_dataframe_schema": "validate_dataframe",
    "validate_data_structure": "validate_dataframe",
    "validate_call_or_runtime_types": "validates",
    
    # Retry and resilience
    "circuit_breaker_protection": "circuit_breaker",
    "retry_with_circuit_breaker": "retry_with_circuit_breaker",
    
    # Logging
    "debug_training_step": "log_call",
    "resource_monitor": "log_execution_time",
    
    # Caching
    "intelligent_caching": "cached",
    "cache_feature_engineering": "cached",
    
    # Security and data processing
    "secure_data_processing": "validates",  # Will need to combine with validation
    "prevent_data_leakage": "validates",
    
    # Performance
    "memory_efficient": "cached",  # Memory efficiency through caching
    "optimize_memory_usage": "cached",
    
    # Tracing
    "with_tracing_span": "traced",
    
    # Guards
    "guard_dataframe_nulls": "validates",
    "guard_array_nan_inf": "validates",
    "nan_inf_and_constant_guard": "validates",
}

# Import statement mappings
OLD_IMPORT_PATTERNS = [
    (r'from src\.utils\.centralized_decorators import \((.*?)\)', 'centralized_decorators'),
    (r'from src\.utils\.training_pipeline_decorators import \((.*?)\)', 'training_pipeline_decorators'),
    (r'from src\.utils\.validation_decorators import \((.*?)\)', 'validation_decorators'),
    (r'from src\.utils\.enhanced_validation_decorators import \((.*?)\)', 'enhanced_validation_decorators'),
    (r'from src\.utils\.decorators import \((.*?)\)', 'decorators'),
    (r'from src\.utils\.enhanced_data_quality_decorators import \((.*?)\)', 'enhanced_data_quality_decorators'),
    (r'from src\.utils\.advanced_decorators import \((.*?)\)', 'advanced_decorators'),
]

# New core imports
NEW_CORE_IMPORTS = {
    'decorators': 'from src.core.decorators import (',
    'errors': 'from src.core.errors import (',
}


def extract_imports_from_multiline(content: str, pattern: str) -> Set[str]:
    """Extract imports from multiline import statements."""
    imports = set()
    
    # Handle multiline imports
    multiline_pattern = pattern.replace(r'\((.*?)\)', r'\(([\s\S]*?)\)')
    matches = re.finditer(multiline_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        import_block = match.group(1)
        # Extract individual imports
        items = re.findall(r'(\w+)(?:\s+as\s+\w+)?', import_block)
        imports.update(items)
    
    return imports


def map_decorator_to_new(old_name: str) -> Tuple[str, str]:
    """Map old decorator name to new decorator and its module."""
    if old_name in DECORATOR_MAPPING:
        new_name = DECORATOR_MAPPING[old_name]
        # Determine which module it belongs to
        if new_name in ['AppError', 'ValidationError', 'ErrorCode']:
            return new_name, 'errors'
        else:
            return new_name, 'decorators'
    return old_name, 'decorators'  # Default to decorators module


def process_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Process a single file and return whether it was modified and any issues."""
    issues = []
    modified = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Track what decorators we need to import
        needed_decorators = set()
        needed_errors = set()
        
        # Extract all old imports
        all_old_imports = set()
        for pattern, module_name in OLD_IMPORT_PATTERNS:
            imports = extract_imports_from_multiline(content, pattern)
            all_old_imports.update(imports)
        
        # Map to new decorators
        for old_import in all_old_imports:
            new_name, module = map_decorator_to_new(old_import)
            if module == 'errors':
                needed_errors.add(new_name)
            else:
                needed_decorators.add(new_name)
        
        # Remove old import statements
        for pattern, _ in OLD_IMPORT_PATTERNS:
            multiline_pattern = pattern.replace(r'\((.*?)\)', r'\(([\s\S]*?)\)')
            content = re.sub(multiline_pattern + r'\s*\n', '', content, flags=re.MULTILINE | re.DOTALL)
        
        # Update decorator usage in code
        for old_name, new_name in DECORATOR_MAPPING.items():
            # Replace @old_name with @new_name
            content = re.sub(rf'@{old_name}\b', f'@{new_name}', content)
            # Replace old_name( with new_name(
            content = re.sub(rf'\b{old_name}\(', f'{new_name}(', content)
        
        # Add new imports at the top of the file (after initial comments/docstrings)
        if needed_decorators or needed_errors:
            # Find where to insert imports
            import_match = re.search(r'^((?:.*\n)*?)(?:import|from)', content, re.MULTILINE)
            if import_match:
                prefix = import_match.group(1)
                rest = content[len(prefix):]
                
                new_imports = []
                if needed_decorators:
                    decorator_imports = sorted(list(needed_decorators))
                    new_imports.append(f"from src.core.decorators import (\n    " + 
                                     ",\n    ".join(decorator_imports) + "\n)")
                
                if needed_errors:
                    error_imports = sorted(list(needed_errors))
                    new_imports.append(f"from src.core.errors import (\n    " + 
                                     ",\n    ".join(error_imports) + "\n)")
                
                content = prefix + "\n".join(new_imports) + "\n\n" + rest
        
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
    print("Starting decorator migration...")
    
    # Find all Python files that need updating
    files_to_update = []
    for root, dirs, files in os.walk('src/training/steps'):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                files_to_update.append(filepath)
    
    print(f"Found {len(files_to_update)} Python files to check...")
    
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
    print(f"2. Update any custom decorator usage patterns")
    print(f"3. Test the updated code")
    print(f"4. Remove old decorator modules if no longer needed")


if __name__ == "__main__":
    main()