#!/usr/bin/env python3
"""
Clean up remaining less critical debug statements while keeping important warnings.
"""

import re
from pathlib import Path

def clean_remaining_debug(file_path: Path):
    """Remove less critical debug statements, keep important warnings."""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Patterns for less critical debug statements to remove
    # Keep: ⚠️ DEBUG warnings (important error indicators)
    # Remove: 🔍 DEBUG info (less critical context)
    patterns_to_remove = [
        r'^\s*tprint_info\(f?"🔍 DEBUG:.*columns\[:.*\].*\)\s*.*$',  # Column display statements
        r'^\s*tprint_info\(f?\'🔍 DEBUG:.*columns\[:.*\].*\)\s*.*$',
        r'^\s*tprint_info\(f?"  🔍 DEBUG: About to generate.*\)\s*$',
        r'^\s*tprint_info\(f?\'  🔍 DEBUG: About to generate.*\)\s*$',
        r'^\s*tprint_info\(f?"  🔍 DEBUG: Expected variants.*\)\s*$',
        r'^\s*tprint_info\(f?\'  🔍 DEBUG: Expected variants.*\)\s*$',
        r'^\s*tprint_info\(f?"  🔍 DEBUG: Feature category breakdown.*\)\s*$',
        r'^\s*tprint_info\(f?\'  🔍 DEBUG: Feature category breakdown.*\)\s*$',
    ]
    
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        should_remove = any(re.match(pattern, line) for pattern in patterns_to_remove)
        
        if should_remove:
            removed_count += 1
            continue
        
        cleaned_lines.append(line)
    
    if removed_count > 0:
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
        print(f"✅ Removed {removed_count} additional debug statements")
        print(f"ℹ️  Kept important warning messages (⚠️ DEBUG)")
    else:
        print(f"ℹ️  No additional debug statements to remove")
    
    return removed_count

if __name__ == '__main__':
    file_path = Path('src/training/steps/pre_training/feature_generation_interaction_generation_step.py')
    clean_remaining_debug(file_path)

