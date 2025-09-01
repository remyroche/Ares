#!/usr/bin/env python3
"""
Code Quality Cleanup Script
Automates the cleanup process based on code quality analysis findings
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class CodeQualityCleanup:
    """Handles automated code quality cleanup tasks."""
    
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path)
        self.code_quality_dir = self.workspace_path / "code_quality"
        
    def run_import_cleanup(self, dry_run: bool = True) -> bool:
        """Run the batch import cleaner."""
        print("🧹 Running import cleanup...")
        
        try:
            cmd = [
                sys.executable,
                str(self.code_quality_dir / "tools" / "batch_import_cleaner.py"),
                "*.py"
            ]
            
            if dry_run:
                cmd.append("--dry-run")
                print("  (Dry run mode - no changes will be made)")
            
            result = subprocess.run(cmd, cwd=self.workspace_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Import cleanup completed successfully")
                if dry_run:
                    print("  Run without --dry-run to apply changes")
                return True
            else:
                print(f"❌ Import cleanup failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running import cleanup: {e}")
            return False
    
    def run_code_quality_analysis(self, output_file: str = None) -> bool:
        """Run the code quality analyzer."""
        print("🔍 Running code quality analysis...")
        
        try:
            cmd = [
                sys.executable,
                str(self.code_quality_dir / "tools" / "code_quality_analyzer.py"),
                str(self.workspace_path),
                "--exclusions", str(self.code_quality_dir / "exclusions.txt")
            ]
            
            if output_file:
                cmd.extend(["--output", output_file])
            
            result = subprocess.run(cmd, cwd=self.workspace_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Code quality analysis completed successfully")
                if output_file:
                    print(f"  Report saved to: {output_file}")
                return True
            else:
                print(f"❌ Code quality analysis failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running code quality analysis: {e}")
            return False
    
    def run_commented_code_analysis(self, output_file: str = None) -> bool:
        """Run the commented code analyzer."""
        print("📝 Running commented code analysis...")
        
        try:
            cmd = [
                sys.executable,
                str(self.code_quality_dir / "analyze_commented_code.py"),
                str(self.workspace_path),
                "--exclusions", str(self.code_quality_dir / "exclusions.txt")
            ]
            
            if output_file:
                cmd.extend(["--output", output_file])
            
            result = subprocess.run(cmd, cwd=self.workspace_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Commented code analysis completed successfully")
                if output_file:
                    print(f"  Report saved to: {output_file}")
                return True
            else:
                print(f"❌ Commented code analysis failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running commented code analysis: {e}")
            return False
    
    def fix_syntax_errors(self, dry_run: bool = True) -> bool:
        """Attempt to fix common syntax errors."""
        print("🔧 Attempting to fix syntax errors...")
        
        if dry_run:
            print("  (Dry run mode - no changes will be made)")
            return True
        
        # This would require more sophisticated syntax error detection and fixing
        # For now, we'll just report that manual review is needed
        print("⚠️  Syntax error fixing requires manual review")
        print("  Files with syntax errors need individual attention")
        return True
    
    def run_full_cleanup(self, dry_run: bool = True) -> bool:
        """Run all cleanup tasks."""
        print("🚀 Starting full code quality cleanup...")
        print("=" * 50)
        
        success = True
        
        # Run import cleanup
        if not self.run_import_cleanup(dry_run):
            success = False
        
        print()
        
        # Run code quality analysis
        if not self.run_code_quality_analysis("cleanup_quality_report.txt"):
            success = False
        
        print()
        
        # Run commented code analysis
        if not self.run_commented_code_analysis("cleanup_comments_report.txt"):
            success = False
        
        print()
        
        # Attempt syntax error fixing
        if not self.fix_syntax_errors(dry_run):
            success = False
        
        print()
        print("=" * 50)
        
        if success:
            print("✅ Full cleanup completed successfully")
            if dry_run:
                print("  Run with --no-dry-run to apply changes")
        else:
            print("❌ Some cleanup tasks failed")
        
        return success


def main():
    parser = argparse.ArgumentParser(description='Code Quality Cleanup Script')
    parser.add_argument('--clean-imports', action='store_true', help='Clean unused imports')
    parser.add_argument('--analyze', action='store_true', help='Run code quality analysis')
    parser.add_argument('--analyze-comments', action='store_true', help='Analyze commented code')
    parser.add_argument('--full-cleanup', action='store_true', help='Run all cleanup tasks')
    parser.add_argument('--no-dry-run', action='store_true', help='Apply changes (not just preview)')
    parser.add_argument('--workspace', default='.', help='Workspace path (default: current directory)')
    parser.add_argument('--report', action='store_true', help='Generate cleanup report')
    
    args = parser.parse_args()
    
    cleanup = CodeQualityCleanup(args.workspace)
    dry_run = not args.no_dry_run
    
    if args.report:
        report = cleanup.generate_cleanup_report()
        print(report)
        return
    
    if args.clean_imports:
        cleanup.run_import_cleanup(dry_run)
    elif args.analyze:
        cleanup.run_code_quality_analysis("quality_report.txt")
    elif args.analyze_comments:
        cleanup.run_commented_code_analysis("comments_report.txt")
    elif args.full_cleanup:
        cleanup.run_full_cleanup(dry_run)
    else:
        # Default: show help
        parser.print_help()


if __name__ == '__main__':
    main()