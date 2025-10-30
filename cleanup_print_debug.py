#!/usr/bin/env python3
"""
Script to clean up print() debug statements from feature_generation_interaction_generation_step.py
"""

import re
from pathlib import Path

def clean_print_debug(file_path: Path):
    """Remove print() debug statements."""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match print statements with DEBUG
    debug_pattern = r'^\s*print\(.*DEBUG:.*\)\s*$'
    
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        if re.match(debug_pattern, line):
            removed_count += 1
            continue
        cleaned_lines.append(line)
    
    if removed_count > 0:
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
        print(f"✅ Removed {removed_count} print() debug statements")
    else:
        print(f"ℹ️  No print() debug statements found")
    
    return removed_count

if __name__ == '__main__':
    file_path = Path('src/training/steps/pre_training/feature_generation_interaction_generation_step.py')
    clean_print_debug(file_path)

