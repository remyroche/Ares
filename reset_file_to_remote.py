#!/usr/bin/env python3
"""
Reset local file to remote version
"""
import subprocess
import os

def run_git_command(cmd, cwd="/Users/remyroche/Documents/Ares"):
    """Run a git command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"❌ Error running command: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        print(f"✅ Success: {cmd}")
        if result.stdout.strip():
            print(f"Output: {result.stdout}")
        return True
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout running command: {cmd}")
        return False
    except Exception as e:
        print(f"❌ Exception running command: {cmd}")
        print(f"Exception: {e}")
        return False

def main():
    """Main function to reset file to remote version."""
    print("🔄 Resetting compare_tbm_parameters.py to remote version...")
    
    # Reset file to remote version
    print("\n📥 Checking out remote version...")
    file_path = "extreme_price_movements/offline_optimisers/compare_tbm_parameters.py"
    if not run_git_command(f"git checkout origin/main -- {file_path}"):
        return False
    
    # Check status
    print("\n📋 Checking git status...")
    if not run_git_command("git status"):
        return False
    
    print("\n✅ File reset to remote version completed!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 compare_tbm_parameters.py has been reset to the GitHub version!")
    else:
        print("❌ Reset failed. Please check the errors above.")
        exit(1)
