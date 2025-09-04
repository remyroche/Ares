#!/usr/bin/env python3
"""
Script to fix unreachable code issues in the specified files.
"""

import re
import os

def fix_unreachable_code_in_file(file_path):
    """Fix unreachable code patterns in a single file."""
    print(f"Fixing unreachable code in: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: Remove empty lines after except blocks with return statements
    pattern1 = r'(except.*?:\s*\n\s*return.*?\n)\s*\n(\s*def|\s*class|\s*@|\s*[A-Z])'
    content = re.sub(pattern1, r'\1\n\2', content, flags=re.MULTILINE)
    
    # Pattern 2: Remove unreachable code after return statements in try blocks
    pattern2 = r'(try:\s*\n.*?return.*?\n)(.*?)(except.*?:)'
    content = re.sub(pattern2, r'\1\3', content, flags=re.MULTILINE | re.DOTALL)
    
    # Pattern 3: Remove unreachable code after return statements in if blocks
    pattern3 = r'(if.*?:\s*\n.*?return.*?\n)\s*\n(\s*[a-zA-Z].*?\n)'
    matches = re.findall(pattern3, content, flags=re.MULTILINE)
    if matches:
        for match in matches:
            # Only remove if the second part looks like unreachable code
            if not match[1].strip().startswith(('def ', 'class ', '@', 'if ', 'else', 'elif', 'for ', 'while ', 'try:', 'except', 'finally:')):
                content = content.replace(match[0] + '\n' + match[1], match[0])
    
    # Pattern 4: Remove duplicate empty lines
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Pattern 5: Fix fallback functions with unreachable code
    pattern5 = r'(def create_fallback_.*?\(\).*?:\s*\n.*?return.*?\n)\s*(.*?\n)'
    matches = re.findall(pattern5, content, flags=re.MULTILINE | re.DOTALL)
    for match in matches:
        if not match[1].strip().startswith(('def ', 'class ', '@', 'if ', 'else')):
            content = content.replace(match[0] + match[1], match[0])
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Fixed unreachable code patterns in {file_path}")
        return True
    else:
        print(f"  ℹ️  No unreachable code patterns found in {file_path}")
        return False

def main():
    """Main function to fix unreachable code in all target files."""
    files_to_fix = [
        '/workspace/src/training/probabilistic_bayesian_optimizer.py',
        '/workspace/src/training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery.py',
        '/workspace/src/utils/data_quality_framework.py',
        '/workspace/src/analyst/predictive_ensembles/ensemble_orchestrator.py',
        '/workspace/src/training/steps/market_analysis/cross_timeframe_interaction_features.py'
    ]
    
    fixed_files = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_unreachable_code_in_file(file_path):
                fixed_files += 1
        else:
            print(f"  ⚠️  File not found: {file_path}")
    
    print(f"\n🎯 Summary: Fixed unreachable code in {fixed_files}/{len(files_to_fix)} files")

if __name__ == "__main__":
    main()