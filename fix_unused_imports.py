#!/usr/bin/env python3
"""
Simple script to remove unused imports from specific files
"""

import ast
import sys
from pathlib import Path




# List of files to fix (valid files only)
files_to_fix = [
    './test_multi_output_final_integration.py',
    './test_steps_1_7_compatibility.py', 
    './test_sr_training_integration.py',
    './fix_metadata_and_naming.py',
    './complete_remaining_steps_integration.py',
    './test_sr_optimization_integration.py',
    './test_tactician_multi_outcome_predictions_updated.py',
    './test_enhanced_dynamic_feature_selection.py',
    './targeted_fix.py'
]

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    
    print(f"{'DRY RUN: ' if dry_run else ''}Fixing unused imports...")
    
    for filepath in files_to_fix:
        if Path(filepath).exists():
            remove_unused_imports(filepath, dry_run=dry_run)
        else:
            print(f"File not found: {filepath}")