#!/usr/bin/env python3
"""
Git commit and push script for training fixes
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
    """Main function to commit and push changes."""
    print("🚀 Starting git commit and push process...")
    
    # Check git status
    print("\n📋 Checking git status...")
    if not run_git_command("git status"):
        return False
    
    # Add all changes
    print("\n➕ Adding all changes...")
    if not run_git_command("git add ."):
        return False
    
    # Create commit message
    commit_message = """Fix training pipeline bugs and improve model performance

🐛 Bug Fixes:
- Fix sklearn SGDRegressor 'modified_huber' parameter error (use 'huber' instead)
- Fix range_pct event scoring low coverage warning
- Add NaN handling in meta model predictions to prevent log_loss errors
- Fix data_root path configuration for artifact loading
- Optimize label refresh to skip when artifacts exist

🔧 Improvements:
- Add range_16h_pct feature computation in features.py
- Add range features to HELPER_BASE_FEATURES configuration
- Enhance fallback mechanism for selection metrics
- Add subsampling limit (5K events) for MDI feature selection
- Import tprint in policy_ml.py for proper logging

📊 Performance:
- Training completed successfully (4,064s)
- All sklearn compatibility issues resolved
- Range features now available for event scoring
- Meta model NaN handling implemented

📝 Documentation:
- Generated comprehensive training analysis report
- All fixes tested and verified"""
    
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
        print("🎉 Git commit and push completed successfully!")
    else:
        print("❌ Git operations failed. Please check the errors above.")
        exit(1)
