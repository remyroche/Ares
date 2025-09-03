#!/usr/bin/env python3
"""
Improved script to migrate from old decorators to new core decorators.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Mapping of old decorator imports to new ones
DECORATOR_MAPPING = {
    # Error handling
    "handle_errors": ("handles_errors", "decorators"),
    "handle_file_operations": ("handles_errors", "decorators"),
    "handle_specific_errors": ("handles_errors", "decorators"),
    "handles_errors": ("handles_errors", "decorators"),  # Already correct
    
    # Validation
    "validate_step_prerequisites": ("validates", "decorators"),
    "validate_step_output": ("validates", "decorators"),
    "validate_dataframe_schema": ("validate_dataframe", "decorators"),
    "validate_data_structure": ("validate_dataframe", "decorators"),
    "validate_call_or_runtime_types": ("validates", "decorators"),
    "guard_dataframe_nulls": ("validates", "decorators"),
    "guard_array_nan_inf": ("validates", "decorators"),
    "nan_inf_and_constant_guard": ("validates", "decorators"),
    
    # Retry and resilience
    "circuit_breaker_protection": ("circuit_breaker", "decorators"),
    "retry_with_circuit_breaker": ("retry_with_circuit_breaker", "decorators"),
    
    # Logging
    "debug_training_step": ("log_call", "decorators"),
    "resource_monitor": ("log_execution_time", "decorators"),
    
    # Caching
    "intelligent_caching": ("cached", "decorators"),
    "cache_feature_engineering": ("cached", "decorators"),
    "memory_efficient": ("cached", "decorators"),
    "optimize_memory_usage": ("cached", "decorators"),
    
    # Tracing
    "with_tracing_span": ("traced", "decorators"),
    
    # Performance monitoring  
    "performance_monitor": ("log_execution_time", "decorators"),
}

# Decorators that need custom handling or don't have direct mappings
CUSTOM_DECORATORS = {
    # These are domain-specific and should be kept temporarily
    "validate_data_quality",
    "validate_feature_engineering_with_lookahead_bias_detection",
    "validate_klines_data_quality",
    "validate_multi_timeframe_data_quality",
    "validate_ohlcv_data_quality",
    "validate_wavelet_data_quality",
    "secure_data_processing",
    "prevent_data_leakage",
    "quality_gate",
    "validate_feature_engineering_pipeline",
    "validate_hmm_data_requirements",
    "validate_hmm_regime_discovery",
    "comprehensive_data_validation",
    "validate_data_completeness",
    "validate_datetime_index",
    "validate_constant_features",
    "validate_low_variance_features",
    "validate_memory_optimized_data_quality",
    "validate_multi_timeframe_alignment",
    "validate_multi_timeframe_processing",
    "validate_enhanced_validation",
    "model_validation",
    "pipeline_checkpoint",
    "adaptive_resource_allocation",
    "comprehensive_validation",
    "artifact_versioning",
    "artifact_write_lock",
    "deterministic_seed",
    "idempotent_step",
    "time_budget_watchdog",
}

# Import statement patterns
OLD_IMPORT_PATTERNS = [
    (r'from src\.utils\.centralized_decorators import', 'centralized_decorators'),
    (r'from src\.utils\.training_pipeline_decorators import', 'training_pipeline_decorators'),
    (r'from src\.utils\.validation_decorators import', 'validation_decorators'),
    (r'from src\.utils\.enhanced_validation_decorators import', 'enhanced_validation_decorators'),
    (r'from src\.utils\.decorators import', 'decorators'),
    (r'from src\.utils\.enhanced_data_quality_decorators import', 'enhanced_data_quality_decorators'),
    (r'from src\.utils\.advanced_decorators import', 'advanced_decorators'),
    (r'from src\.utils\.error_handler import', 'error_handler'),
]


def extract_imports_from_block(content: str, start_pattern: str) -> Tuple[Optional[str], Set[str]]:
    """Extract imports from an import block and return the full block and individual imports."""
    imports = set()
    
    # Find the import statement
    pattern = start_pattern + r'\s*\(([\s\S]*?)\)'
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    
    if match:
        full_block = match.group(0)
        import_block = match.group(1)
        # Extract individual imports
        items = re.findall(r'(\w+)(?:\s+as\s+\w+)?', import_block)
        imports.update(items)
        return full_block, imports
    
    # Try single line import
    pattern = start_pattern + r'\s+(\w+(?:\s*,\s*\w+)*)'
    match = re.search(pattern, content)
    if match:
        full_block = match.group(0)
        import_items = match.group(1).split(',')
        imports.update(item.strip() for item in import_items)
        return full_block, imports
    
    return None, set()


def process_file(filepath: Path, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """Process a single file and return whether it was modified and any issues."""
    issues = []
    modified = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Track what we need to import
        new_decorators_needed = set()
        new_errors_needed = set()
        custom_decorators_needed = set()
        
        # Process each old import pattern
        for pattern, module_name in OLD_IMPORT_PATTERNS:
            full_block, imports = extract_imports_from_block(content, pattern)
            
            if full_block and imports:
                # Map each import
                for old_name in imports:
                    if old_name in CUSTOM_DECORATORS:
                        custom_decorators_needed.add(old_name)
                    elif old_name in DECORATOR_MAPPING:
                        new_name, module = DECORATOR_MAPPING[old_name]
                        if module == 'errors':
                            new_errors_needed.add(new_name)
                        else:
                            new_decorators_needed.add(new_name)
                        
                        # Update usage in code
                        if old_name != new_name:
                            # Replace @old_name with @new_name
                            content = re.sub(rf'@{old_name}\b', f'@{new_name}', content)
                            # Replace old_name( with new_name(
                            content = re.sub(rf'(?<!\w){old_name}\(', f'{new_name}(', content)
                    else:
                        # Unknown decorator - keep for now but log
                        custom_decorators_needed.add(old_name)
                        issues.append(f"Unknown decorator: {old_name}")
                
                # Remove the old import block
                content = content.replace(full_block, '')
        
        # Clean up extra newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Build new import statements
        import_lines = []
        
        if new_decorators_needed:
            decorators = sorted(list(new_decorators_needed))
            if len(decorators) == 1:
                import_lines.append(f"from src.core.decorators import {decorators[0]}")
            else:
                import_lines.append("from src.core.decorators import (")
                for dec in decorators[:-1]:
                    import_lines.append(f"    {dec},")
                import_lines.append(f"    {decorators[-1]}")
                import_lines.append(")")
        
        if new_errors_needed:
            errors = sorted(list(new_errors_needed))
            if len(errors) == 1:
                import_lines.append(f"from src.core.errors import {errors[0]}")
            else:
                import_lines.append("from src.core.errors import (")
                for err in errors[:-1]:
                    import_lines.append(f"    {err},")
                import_lines.append(f"    {errors[-1]}")
                import_lines.append(")")
        
        # Keep custom decorators temporarily with a TODO comment
        if custom_decorators_needed:
            import_lines.append("")
            import_lines.append("# TODO: These decorators need to be migrated to core decorators or removed")
            decorators = sorted(list(custom_decorators_needed))
            import_lines.append("from src.utils.centralized_decorators import (")
            for dec in decorators[:-1]:
                import_lines.append(f"    {dec},")
            import_lines.append(f"    {decorators[-1]}")
            import_lines.append(")")
        
        # Insert new imports after docstring/comments
        if import_lines and (new_decorators_needed or new_errors_needed or custom_decorators_needed):
            # Find insertion point
            lines = content.split('\n')
            insert_idx = 0
            
            # Skip shebang
            if lines and lines[0].startswith('#!'):
                insert_idx = 1
            
            # Skip module docstring
            if insert_idx < len(lines) and lines[insert_idx].strip().startswith('"""'):
                for i in range(insert_idx + 1, len(lines)):
                    if lines[i].strip().endswith('"""'):
                        insert_idx = i + 1
                        break
            
            # Skip comments and empty lines
            while insert_idx < len(lines) and (not lines[insert_idx].strip() or 
                                              lines[insert_idx].strip().startswith('#')):
                insert_idx += 1
            
            # Insert imports
            lines = lines[:insert_idx] + import_lines + [''] + lines[insert_idx:]
            content = '\n'.join(lines)
        
        # Only write if content changed
        if content != original_content:
            if not dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            modified = True
            
    except Exception as e:
        issues.append(f"Error processing {filepath}: {str(e)}")
    
    return modified, issues


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate decorators to new core system')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--path', default='src/training/steps', help='Path to process')
    args = parser.parse_args()
    
    print("Starting improved decorator migration...")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    # Find all Python files that need updating
    files_to_update = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                files_to_update.append(filepath)
    
    print(f"Found {len(files_to_update)} Python files to check...")
    
    modified_count = 0
    all_issues = []
    
    for filepath in files_to_update:
        modified, issues = process_file(filepath, dry_run=args.dry_run)
        if modified:
            modified_count += 1
            print(f"✓ {'Would update' if args.dry_run else 'Updated'}: {filepath}")
        if issues:
            all_issues.extend(issues)
    
    print(f"\n{'='*60}")
    print(f"Migration complete!")
    print(f"Files {'would be' if args.dry_run else ''} modified: {modified_count}")
    
    if all_issues:
        print(f"\nIssues encountered:")
        unique_issues = sorted(set(all_issues))
        for issue in unique_issues:
            count = all_issues.count(issue)
            print(f"  - {issue} ({count} occurrences)")
    
    print(f"\nNext steps:")
    print(f"1. Review the changes to ensure correctness")
    print(f"2. Migrate custom decorators marked with TODO comments")
    print(f"3. Test the updated code")
    print(f"4. Remove old decorator modules when migration is complete")


if __name__ == "__main__":
    main()