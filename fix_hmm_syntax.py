#!/usr/bin/env python3
"""
Script to fix syntax issues in the HMM file.
"""

import re

def fix_hmm_syntax_issues(file_path):
    """Fix syntax issues in the HMM file."""
    print(f"Fixing syntax issues in: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove orphaned except blocks that don't have matching try blocks
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is an orphaned except block
        if line.strip().startswith('except ') and not line.startswith('    ') and not line.startswith('\t'):
            # Look backwards for a matching try block
            found_try = False
            for j in range(i - 1, -1, -1):
                if lines[j].strip().startswith('try:'):
                    found_try = True
                    break
                if lines[j].strip() and not lines[j].strip().startswith('#') and not lines[j].strip().startswith('@'):
                    break
            
            if not found_try:
                # This is an orphaned except block, skip it and its content
                print(f"  Removing orphaned except block at line {i + 1}: {line.strip()}")
                while i < len(lines) and (lines[i].strip().startswith('except ') or lines[i].strip() == '' or (lines[i].startswith('    ') and not lines[i].strip().startswith('def ') and not lines[i].strip().startswith('class ') and not lines[i].strip().startswith('@'))):
                    i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Fixed syntax issues in {file_path}")

def main():
    """Main function."""
    files_to_fix = [
        '/workspace/src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery.py',
        '/workspace/src/utils/data_quality_framework.py',
        '/workspace/src/analyst/predictive_ensembles/ensemble_orchestrator.py',
        '/workspace/src/training/steps/market_analysis/cross_timeframe_interaction_features.py'
    ]
    
    for file_path in files_to_fix:
        fix_hmm_syntax_issues(file_path)

if __name__ == "__main__":
    main()