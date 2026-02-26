#!/usr/bin/env python3
"""
Git add, commit, and push script
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
    """Main function to add, commit, and push changes."""
    print("🚀 Starting git add, commit, and push process...")
    
    # Check git status
    print("\n📋 Checking git status...")
    if not run_git_command("git status"):
        return False
    
    # Add all changes
    print("\n➕ Adding all changes...")
    if not run_git_command("git add ."):
        return False
    
    # Check if there are changes to commit
    print("\n🔍 Checking for staged changes...")
    result = subprocess.run("git diff --cached --name-only", shell=True, cwd="/Users/remyroche/Documents/Ares", capture_output=True, text=True, timeout=10)
    if result.returncode == 0 and not result.stdout.strip():
        print("ℹ️ No changes to commit. Repository is up to date.")
        return True
    
    # Create commit message
    commit_message = """Sync local changes with latest updates

🔄 Repository Updates:
- Sync with latest main branch changes
- Update local configurations and reports
- Maintain consistency with remote repository

📊 Changes Include:
- Updated TBM parameter configurations
- Refreshed candidate threshold reports
- Updated production settings
- Cache and report updates"""
    
    # Commit changes
    print(f"\n📝 Committing changes...")
    if not run_git_command(f'git commit -m "{commit_message}"'):
        return False
    
    # Push changes
    print("\n🚀 Pushing changes to remote...")
    if not run_git_command("git push"):
        return False
    
    print("\n✅ All git operations completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Git add, commit, and push completed successfully!")
    else:
        print("❌ Git operations failed. Please check the errors above.")
        exit(1)
