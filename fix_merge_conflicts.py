#!/usr/bin/env python3
"""
Fix merge conflicts from decorator migration.
"""

import re
from pathlib import Path

def fix_merge_conflict(file_path):
    """Fix merge conflicts in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file has conflicts
    if '<<<<<<< HEAD' not in content:
        return False
    
    # Split into lines for processing
    lines = content.split('\n')
    fixed_lines = []
    in_conflict = False
    head_section = []
    merge_section = []
    current_section = None
    
    for line in lines:
        if line.startswith('<<<<<<< HEAD'):
            in_conflict = True
            current_section = 'head'
            head_section = []
            continue
        elif line.startswith('======='):
            current_section = 'merge'
            merge_section = []
            continue
        elif line.startswith('>>>>>>>'):
            in_conflict = False
            # Resolve the conflict
            # For decorator imports, we want to keep the new core decorators (HEAD)
            # and discard the old centralized decorators (merge)
            if any('from src.core.decorators import' in h for h in head_section):
                # Keep HEAD section (new decorators)
                fixed_lines.extend(head_section)
            elif any('from src.utils.centralized_decorators import' in m for m in merge_section):
                # Replace with new decorators
                fixed_lines.append('from src.core.decorators import (')
                fixed_lines.append('    handles_errors,')
                fixed_lines.append('    validates,')
                fixed_lines.append('    cached,')
                fixed_lines.append('    log_execution_time,')
                fixed_lines.append('    traced,')
                fixed_lines.append(')')
            elif any('from src.utils.error_handler import' in m for m in merge_section):
                # Replace with new decorators
                fixed_lines.append('from src.core.decorators import handles_errors, retry, timeout')
            else:
                # For other conflicts, try to merge intelligently
                # If HEAD has the new system, keep it
                if any('src.core.decorators' in h for h in head_section) or \
                   any('src.core.errors' in h for h in head_section):
                    fixed_lines.extend(head_section)
                else:
                    # Otherwise keep merge section
                    fixed_lines.extend(merge_section)
            head_section = []
            merge_section = []
            continue
        
        if in_conflict:
            if current_section == 'head':
                head_section.append(line)
            elif current_section == 'merge':
                merge_section.append(line)
        else:
            fixed_lines.append(line)
    
    # Write back
    fixed_content = '\n'.join(fixed_lines)
    
    # Clean up any duplicate imports
    fixed_content = clean_duplicate_imports(fixed_content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    return True

def clean_duplicate_imports(content):
    """Remove duplicate import statements."""
    lines = content.split('\n')
    seen_imports = set()
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Handle multi-line imports
        if line.startswith('from ') and '(' in line and ')' not in line:
            # Collect the entire import block
            import_block = [lines[i]]
            j = i + 1
            while j < len(lines) and ')' not in lines[j]:
                import_block.append(lines[j])
                j += 1
            if j < len(lines):
                import_block.append(lines[j])
            
            # Create a signature for this import
            import_sig = ' '.join(l.strip() for l in import_block)
            if import_sig not in seen_imports:
                seen_imports.add(import_sig)
                cleaned_lines.extend(import_block)
            
            i = j + 1
        else:
            # Single line import or non-import line
            if line.startswith(('from ', 'import ')):
                if line not in seen_imports:
                    seen_imports.add(line)
                    cleaned_lines.append(lines[i])
            else:
                cleaned_lines.append(lines[i])
            i += 1
    
    return '\n'.join(cleaned_lines)

def main():
    # Files with conflicts
    conflict_files = [
        'src/analyst/enhanced_prediction_integrator.py',
        'src/analyst/liquidation_risk_model.py',
        'src/exchange/binance.py',
        'src/pipelines/base_pipeline.py',
        'src/pipelines/components/data_manager.py',
        'src/pipelines/components/lifecycle_manager.py',
        'src/pipelines/components/monitoring_manager.py',
        'src/pipelines/live_trading_pipeline.py',
        'src/strategist/strategist_backup.py',
        'src/tactician/enhanced_order_manager.py',
        'src/tactician/enhanced_prediction_integrator.py',
        'src/tactician/ml_tactics_manager.py',
        'src/tactician/ml_target_updater.py',
        'src/tactician/ml_target_validator.py',
        'src/tactician/position_division_strategy.py',
        'src/tactician/position_sizer.py',
        'src/tactician/scenario_based_predictor.py',
        'src/training/core/checkpoint_manager.py',
        'src/training/core/pipeline_orchestrator.py',
        'src/training/core/stage_context.py',
        'src/training/feature_engineering.py',
        'src/training/optimization/advanced_surrogate_models.py',
        'src/training/steps/hmm_feature_enhancer.py',
    ]
    
    print("Fixing merge conflicts in decorator migration...\n")
    
    fixed_count = 0
    for file_path in conflict_files:
        full_path = Path('/workspace') / file_path
        if full_path.exists():
            print(f"Processing {file_path}...")
            if fix_merge_conflict(full_path):
                fixed_count += 1
                print(f"  ✓ Fixed conflicts")
            else:
                print(f"  - No conflicts found")
    
    print(f"\nFixed conflicts in {fixed_count} files.")

if __name__ == '__main__':
    main()