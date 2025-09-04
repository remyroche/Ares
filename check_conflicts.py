#!/usr/bin/env python3
import os
import glob

# Check for conflict markers in all Python files
conflict_files = []

for py_file in glob.glob('/workspace/src/**/*.py', recursive=True):
    try:
        with open(py_file, 'r') as f:
            content = f.read()
            if '<<<<<<< HEAD' in content or '=======' in content or '>>>>>>>' in content:
                conflict_files.append(py_file)
    except:
        pass

if conflict_files:
    print(f"Found {len(conflict_files)} files with conflicts:")
    for f in conflict_files:
        print(f"  - {f}")
else:
    print("No conflict markers found in Python files")

# Check git directory files
print("\nGit merge files present:")
git_files = ['.git/MERGE_HEAD', '.git/MERGE_MSG', '.git/MERGE_MODE', '.git/AUTO_MERGE']
for gf in git_files:
    if os.path.exists(f'/workspace/{gf}'):
        print(f"  ✓ {gf}")
    else:
        print(f"  ✗ {gf}")