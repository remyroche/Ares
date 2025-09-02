#!/usr/bin/env python3
"""
Simple Step Formatter Wrapper

This script provides an easy way to format step mentions (step01 -> step01, step02 -> step02, etc.)
in both file contents and file names.

Usage:
    python format_steps.py                    # Show what would be changed (dry run)
    python format_steps.py --apply           # Apply the changes
    python format_steps.py --apply --backup  # Apply changes with backup files
    python format_steps.py --help            # Show help
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main function to run the step formatter."""
    
    # Check if step_formatter.py exists
    if not Path("step_formatter.py").exists():
        print("❌ Error: step_formatter.py not found in current directory")
        print("Please run this script from the directory containing step_formatter.py")
        sys.exit(1)
    
    # Parse arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nAvailable options:")
        print("  --apply           Apply the changes (default is dry run)")
        print("  --backup          Create backup files before making changes")
        print("  --recursive       Process subdirectories recursively")
        print("  --help, -h        Show this help message")
        sys.exit(0)
    
    # Build command
    cmd = ["python3", "step_formatter.py", "--recursive"]
    
    # Add flags based on arguments
    if "--apply" in sys.argv:
        # Remove --dry-run (default behavior)
        pass
    else:
        # Default to dry run
        cmd.append("--dry-run")
    
    if "--backup" in sys.argv:
        cmd.append("--backup")
    
    # Add current directory as target
    cmd.append(".")
    
    # Show what we're about to do
    mode = "APPLYING CHANGES" if "--apply" in sys.argv else "DRY RUN (no changes)"
    print(f"🚀 Starting step formatter in {mode} mode")
    print(f"📁 Target: Current directory (recursive)")
    if "--backup" in sys.argv:
        print("💾 Backup mode: Backup files will be created")
    print(f"🔧 Command: {' '.join(cmd)}")
    print()
    
    # Run the formatter
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Step formatting completed successfully!")
        
        if "--apply" not in sys.argv:
            print("\n💡 To apply the changes, run:")
            print("   python format_steps.py --apply")
            if "--backup" not in sys.argv:
                print("   python format_steps.py --apply --backup  # With backup files")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running step formatter: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()