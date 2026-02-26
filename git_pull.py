#!/usr/bin/env python3
"""
Git pull script
"""
import subprocess
import os

def run_git_command(cmd, cwd="/Users/remyroche/Documents/Ares"):
    """Run a git command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60)
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
    """Main function to pull changes."""
    print("🔄 Starting git pull...")
    
    # Pull changes
    print("\n📥 Pulling changes from remote...")
    if not run_git_command("git pull"):
        return False
    
    print("\n✅ Git pull completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Git pull completed successfully!")
    else:
        print("❌ Git pull failed. Please check the errors above.")
        exit(1)
