#!/usr/bin/env python3
"""
Comprehensive cleanup script to update ALL old decorator imports to new system.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


# Mapping of old decorators to their new locations
DECORATOR_MAPPINGS = {
    # From centralized_decorators
    'ValidationLevel': ('domain', 'ValidationLevel'),
    'PerformanceLevel': ('domain', 'PerformanceLevel'),
    'validate_data_quality': ('domain', 'validate_data_quality'),
    'validate_feature_engineering_with_lookahead_bias_detection': ('domain', 'validate_feature_engineering_with_lookahead_bias_detection'),
    'validate_klines_data_quality': ('domain', 'validate_klines_data_quality'),
    'validate_multi_timeframe_data_quality': ('domain', 'validate_multi_timeframe_data_quality'),
    'validate_ohlcv_data_quality': ('domain', 'validate_ohlcv_data_quality'),
    'validate_wavelet_data_quality': ('domain', 'validate_wavelet_data_quality'),
    'monitor_step_execution': ('domain', 'monitor_step_execution'),
    'quality_gate': ('domain', 'quality_gate'),
    'secure_data_processing': ('domain', 'secure_data_processing'),
    'prevent_data_leakage': ('domain', 'prevent_data_leakage'),
    'ensure_data_integrity': ('domain', 'ensure_data_integrity'),
    'validate_pipeline_step': ('domain', 'validate_pipeline_step'),
    'secure_step_execution': ('domain', 'secure_step_execution'),
    'monitor_pipeline_performance': ('domain', 'monitor_pipeline_performance'),
    'artifact_versioning': ('domain', 'artifact_versioning'),
    'deterministic_seed': ('domain', 'deterministic_seed'),
    'idempotent_step': ('domain', 'idempotent_step'),
    'time_budget_watchdog': ('domain', 'time_budget_watchdog'),
    'smart_validation_cache': ('domain', 'smart_validation_cache'),
    'enforce_ndarray': ('domain', 'enforce_ndarray'),
    'comprehensive_validation': ('domain', 'comprehensive_validation'),
    'optimize_memory_usage': ('domain', 'optimize_memory_usage'),
    
    # From training_pipeline_decorators
    'circuit_breaker_protection': ('decorators', 'circuit_breaker'),
    'debug_training_step': ('decorators', 'log_call'),
    'memory_efficient': ('decorators', 'cached'),
    'resource_monitor': ('decorators', 'log_execution_time'),
    'validate_step_output': ('decorators', 'validates'),
    'validate_step_prerequisites': ('decorators', 'validates'),
    
    # From validation_decorators
    'validate_enhanced_validation': ('decorators', 'validates'),
    
    # From error_handler
    'handle_errors': ('decorators', 'handles_errors'),
    
    # From decorators
    'guard_dataframe_nulls': ('decorators', 'validates'),
    'with_tracing_span': ('decorators', 'traced'),
}

# Modules to update imports from
OLD_MODULES = [
    'src.utils.centralized_decorators',
    'src.utils.training_pipeline_decorators',
    'src.utils.validation_decorators',
    'src.utils.enhanced_validation_decorators',
    'src.utils.decorators',
    'src.utils.error_handler',
    'src.utils.advanced_decorators',
    'src.utils.enhanced_data_quality_decorators',
]


def extract_imports(import_text: str) -> List[str]:
    """Extract individual imports from import statement."""
    imports = re.findall(r'(\w+)(?:\s+as\s+\w+)?', import_text)
    return [imp for imp in imports if imp and imp != 'import' and imp != 'from']


def process_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Process a single file and update decorator imports."""
    issues = []
    modified = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Track what we need to import
        core_decorators_needed = set()
        domain_decorators_needed = set()
        
        # Process each old module
        for old_module in OLD_MODULES:
            # Find multiline imports
            pattern = rf'from {re.escape(old_module)} import \(([\s\S]*?)\)'
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
            
            for match in reversed(matches):
                imports_text = match.group(1)
                imports = extract_imports(imports_text)
                
                # Map each import
                for imp in imports:
                    if imp in DECORATOR_MAPPINGS:
                        location, new_name = DECORATOR_MAPPINGS[imp]
                        if location == 'decorators':
                            core_decorators_needed.add(new_name)
                        else:  # domain
                            domain_decorators_needed.add(new_name)
                    else:
                        # Unknown decorator - add to domain by default
                        domain_decorators_needed.add(imp)
                
                # Remove the old import
                start, end = match.span()
                content = content[:start] + content[end:]
                if start > 0 and content[start-1:start] == '\n':
                    content = content[:start-1] + content[start:]
            
            # Find single-line imports
            pattern = rf'from {re.escape(old_module)} import\s+([^\n]+)'
            matches = list(re.finditer(pattern, content))
            
            for match in reversed(matches):
                imports_text = match.group(1)
                imports = [imp.strip() for imp in imports_text.split(',')]
                
                for imp in imports:
                    imp = imp.strip()
                    if imp in DECORATOR_MAPPINGS:
                        location, new_name = DECORATOR_MAPPINGS[imp]
                        if location == 'decorators':
                            core_decorators_needed.add(new_name)
                        else:  # domain
                            domain_decorators_needed.add(new_name)
                    else:
                        # Unknown decorator
                        domain_decorators_needed.add(imp)
                
                # Remove the old import
                start, end = match.span()
                content = content[:start] + content[end:]
                if start > 0 and content[start-1:start] == '\n':
                    content = content[:start-1] + content[start:]
        
        # Update decorator usage in the code
        for old_name, (location, new_name) in DECORATOR_MAPPINGS.items():
            if old_name != new_name:
                # Update @decorator usage
                content = re.sub(rf'@{re.escape(old_name)}\b', f'@{new_name}', content)
                # Update decorator() calls
                content = re.sub(rf'(?<![\w.]){re.escape(old_name)}\(', f'{new_name}(', content)
        
        # Add new imports if needed
        if core_decorators_needed or domain_decorators_needed:
            # Find where to insert imports (after docstring and comments)
            lines = content.split('\n')
            insert_idx = 0
            
            # Skip shebang
            if lines and lines[0].startswith('#!'):
                insert_idx = 1
            
            # Skip file-level comments
            while insert_idx < len(lines) and lines[insert_idx].strip().startswith('#'):
                insert_idx += 1
            
            # Skip module docstring
            if insert_idx < len(lines) and lines[insert_idx].strip().startswith('"""'):
                insert_idx += 1
                while insert_idx < len(lines) and not lines[insert_idx].strip().endswith('"""'):
                    insert_idx += 1
                if insert_idx < len(lines):
                    insert_idx += 1
            
            # Skip empty lines
            while insert_idx < len(lines) and not lines[insert_idx].strip():
                insert_idx += 1
            
            # Build import statements
            import_lines = []
            
            if core_decorators_needed:
                decorators = sorted(list(core_decorators_needed))
                if len(decorators) == 1:
                    import_lines.append(f"from src.core.decorators import {decorators[0]}")
                else:
                    import_lines.append("from src.core.decorators import (")
                    for dec in decorators[:-1]:
                        import_lines.append(f"    {dec},")
                    import_lines.append(f"    {decorators[-1]}")
                    import_lines.append(")")
            
            if domain_decorators_needed:
                if import_lines:
                    import_lines.append("")  # Empty line between imports
                decorators = sorted(list(domain_decorators_needed))
                if len(decorators) == 1:
                    import_lines.append(f"from src.core.domain import {decorators[0]}")
                else:
                    import_lines.append("from src.core.domain import (")
                    for dec in decorators[:-1]:
                        import_lines.append(f"    {dec},")
                    import_lines.append(f"    {decorators[-1]}")
                    import_lines.append(")")
            
            if import_lines:
                lines = lines[:insert_idx] + import_lines + [""] + lines[insert_idx:]
                content = '\n'.join(lines)
        
        # Clean up multiple empty lines
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        
        # Write back if modified
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            modified = True
            
    except Exception as e:
        issues.append(f"Error processing {filepath}: {str(e)}")
    
    return modified, issues


def main():
    """Main cleanup function."""
    print("Starting comprehensive decorator cleanup...")
    
    # Find all Python files
    all_files = []
    for root, dirs, files in os.walk('src/'):
        # Skip __pycache__ and old decorator modules
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                
                # Skip old decorator modules
                if any(module.replace('.', '/') in str(filepath) for module in OLD_MODULES):
                    continue
                
                all_files.append(filepath)
    
    # Check which files need updating
    files_to_update = []
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if any(module in content for module in OLD_MODULES):
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
        for issue in all_issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more issues")
    
    print(f"\nDecorator migration is now complete!")
    print(f"All decorators have been migrated to:")
    print(f"  - src.core.decorators (core functionality)")
    print(f"  - src.core.domain (domain-specific decorators)")


if __name__ == "__main__":
    main()