#!/usr/bin/env python3
"""
Script to clean up excessive debug logging from feature_generation_interaction_generation_step.py
Removes lines containing '🔍 DEBUG:' to improve readability and performance.
"""

import re
from pathlib import Path

def clean_debug_logging(file_path: Path, dry_run: bool = False):
    """Remove excessive debug logging statements."""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Patterns to remove
    debug_patterns = [
        r'^\s*tprint_info\(f?"🔍 DEBUG:.*\)$',
        r'^\s*tprint_info\(f?\'🔍 DEBUG:.*\)$',
        r'^\s*print\(f?"🔍 DEBUG:.*\)$',
        r'^\s*print\(f?\'🔍 DEBUG:.*\)$',
    ]
    
    cleaned_lines = []
    removed_count = 0
    line_numbers = []
    
    for i, line in enumerate(lines, 1):
        # Check if line matches any debug pattern
        is_debug = any(re.match(pattern, line) for pattern in debug_patterns)
        
        if is_debug:
            removed_count += 1
            line_numbers.append(i)
            # Skip this line (remove it)
            continue
        
        cleaned_lines.append(line)
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Removed {removed_count} debug logging statements")
    
    if removed_count > 0 and not dry_run:
        # Write cleaned content back
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
        print(f"✅ Cleaned file written: {file_path}")
        print(f"📊 Removed lines: {line_numbers[:20]}{'...' if len(line_numbers) > 20 else ''}")
    elif removed_count > 0:
        print(f"📋 Would remove lines: {line_numbers[:20]}{'...' if len(line_numbers) > 20 else ''}")
    
    return removed_count

if __name__ == '__main__':
    import sys
    
    file_path = Path('src/training/steps/pre_training/feature_generation_interaction_generation_step.py')
    
    # First, do a dry run
    print("=" * 80)
    print("DRY RUN - Showing what would be removed")
    print("=" * 80)
    clean_debug_logging(file_path, dry_run=True)
    
    print("\n" + "=" * 80)
    print("ACTUAL RUN - Removing debug statements")
    print("=" * 80)
    
    # Ask for confirmation or run directly if --yes flag
    if '--yes' in sys.argv or '-y' in sys.argv:
        clean_debug_logging(file_path, dry_run=False)
    else:
        response = input("\nProceed with cleanup? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            clean_debug_logging(file_path, dry_run=False)
        else:
            print("❌ Cleanup cancelled")

