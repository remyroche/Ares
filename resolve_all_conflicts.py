#!/usr/bin/env python3
"""
Script to resolve all merge conflicts by accepting our version.
Our version has the correct imports that were added by the auto-fixer.
"""

import subprocess
import sys


def resolve_conflicts():
    """Resolve all conflicts by accepting our version."""
    try:
        # Get list of conflicted files
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='/workspace')
        
        conflicted_files = []
        for line in result.stdout.split('\n'):
            if line.startswith('UU ') or line.startswith('AA ') or line.startswith('DD '):
                file_path = line[3:].strip()
                conflicted_files.append(file_path)
        
        print(f"Found {len(conflicted_files)} files with conflicts")
        
        # Resolve conflicts by accepting our version
        resolved_count = 0
        for file_path in conflicted_files:
            try:
                result = subprocess.run(['git', 'checkout', '--ours', file_path], 
                                      capture_output=True, text=True, cwd='/workspace')
                if result.returncode == 0:
                    resolved_count += 1
                    print(f"✓ Resolved {file_path}")
                else:
                    print(f"✗ Failed to resolve {file_path}: {result.stderr}")
            except Exception as e:
                print(f"✗ Error resolving {file_path}: {e}")
        
        print(f"\nResolved conflicts in {resolved_count} files")
        
        # Add all resolved files
        if resolved_count > 0:
            subprocess.run(['git', 'add', '.'], cwd='/workspace')
            print("Added all resolved files to staging area")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    resolve_conflicts()