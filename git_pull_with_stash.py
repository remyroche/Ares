#!/usr/bin/env python3
"""
Git pull with stash handling
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
    """Main function to handle git pull with stashing."""
    print("🔄 Starting git pull with stash handling...")
    
    # Stash changes
    print("\n📦 Stashing current changes...")
    if not run_git_command("git stash push -m 'Temporary stash before pull'"):
        print("⚠️ Stash failed, but continuing...")
    
    # Pull changes
    print("\n📥 Pulling changes from remote...")
    if not run_git_command("git pull"):
        print("❌ Pull failed")
        return False
    
    # Unstash changes
    print("\n📦 Restoring stashed changes...")
    if not run_git_command("git stash pop"):
        print("⚠️ Could not restore stash (may be empty)")
    
    print("\n✅ Git pull with stash handling completed!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Git pull completed successfully!")
    else:
        print("❌ Git pull failed. Please check the errors above.")
        exit(1)
