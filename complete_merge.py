#!/usr/bin/env python3
import subprocess
import os
import sys

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/workspace')
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

# Check git status
print("Checking git status...")
stdout, stderr, code = run_command("git status --porcelain")
print(f"Modified files:\n{stdout}")

# Check if we're in a merge
stdout, stderr, code = run_command("git status")
if "All conflicts fixed but you are still merging" in stdout or "You have unmerged paths" in stdout:
    print("\nMerge in progress detected")
    
    # Try to complete the merge
    print("\nCompleting merge...")
    stdout, stderr, code = run_command("git commit --no-edit")
    if code == 0:
        print("✅ Merge completed successfully!")
    else:
        print(f"❌ Error completing merge: {stderr}")
else:
    print("\nNo merge in progress or already completed")
    
# Show final status
stdout, stderr, code = run_command("git status")
print(f"\nFinal status:\n{stdout}")