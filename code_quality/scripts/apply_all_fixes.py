#!/usr/bin/env python3
"""
Master script to apply all code quality fixes.
This script coordinates the various fix scripts to improve code quality.
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime


class CodeQualityFixer:
    def __init__(self, project_root: str = '/workspace/src'):
        self.project_root = Path(project_root)
        self.reports_dir = Path('/workspace/code_quality/reports')
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_command(self, cmd: str, description: str) -> bool:
        """Run a command and capture output."""
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd='/workspace'
            )
            
            if result.returncode == 0:
                print(f"✓ {description} completed successfully")
                return True
            else:
                print(f"✗ {description} failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"✗ Error running {description}: {e}")
            return False
    
    def fix_imports(self, dry_run: bool = True):
        """Fix missing imports."""
        cmd = f"python3 code_quality/fix_missing_imports.py --project-root {self.project_root}"
        if not dry_run:
            cmd += " --fix"
        
        return self.run_command(cmd, "Fix missing imports")
    
    def fix_async_await(self, dry_run: bool = True):
        """Fix async/await patterns."""
        cmd = f"python3 code_quality/fix_async_await.py --project-root {self.project_root}"
        if not dry_run:
            cmd += " --fix"
        
        return self.run_command(cmd, "Fix async/await patterns")
    
    def check_circular_imports(self):
        """Check for circular imports."""
        cmd = f"python3 code_quality/detect_circular_imports.py --project-root {self.project_root}"
        return self.run_command(cmd, "Check circular imports")
    
    def analyze_type_hints(self):
        """Analyze type hint coverage."""
        cmd = f"python3 code_quality/add_type_hints.py --project-root {self.project_root} --analyze"
        return self.run_command(cmd, "Analyze type hints")
    
    def create_summary_report(self):
        """Create a summary report of all fixes."""
        report = {
            'timestamp': self.timestamp,
            'project_root': str(self.project_root),
            'fixes_applied': [],
            'recommendations': []
        }
        
        # Check import fixes report
        import_report_path = Path('/workspace/code_quality/import_fixes_report.json')
        if import_report_path.exists():
            with open(import_report_path, 'r') as f:
                import_data = json.load(f)
            report['fixes_applied'].append({
                'type': 'imports',
                'files_to_fix': len(import_data.get('imports_by_file', {})),
                'summary': import_data.get('summary', {})
            })
        
        # Check async fixes report
        async_report_path = Path('/workspace/code_quality/async_fixes_report.json')
        if async_report_path.exists():
            with open(async_report_path, 'r') as f:
                async_data = json.load(f)
            report['fixes_applied'].append({
                'type': 'async_await',
                'files_to_fix': len(async_data.get('issues_by_file', {})),
                'total_issues': async_data.get('total_issues', 0)
            })
        
        # Check circular imports report
        circular_report_path = Path('/workspace/code_quality/circular_imports_report.json')
        if circular_report_path.exists():
            with open(circular_report_path, 'r') as f:
                circular_data = json.load(f)
            report['fixes_applied'].append({
                'type': 'circular_imports',
                'cycles_found': circular_data.get('circular_imports', {}).get('count', 0)
            })
        
        # Add recommendations
        report['recommendations'] = [
            "1. Review and apply the import fixes to resolve undefined function errors",
            "2. Fix async/await patterns to ensure proper asynchronous execution",
            "3. Add type hints to improve code readability and catch type errors",
            "4. Use the common_operations utility module for frequently used functions",
            "5. Run these fixes regularly as part of your development workflow"
        ]
        
        # Save summary
        summary_path = self.reports_dir / f"code_quality_summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSummary report saved to: {summary_path}")
        
        # Print summary
        print("\nCODE QUALITY IMPROVEMENT SUMMARY")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Project: {report['project_root']}")
        print("\nFixes Available:")
        for fix in report['fixes_applied']:
            print(f"  - {fix['type']}: ", end="")
            if fix['type'] == 'imports':
                print(f"{fix['files_to_fix']} files need import fixes")
            elif fix['type'] == 'async_await':
                print(f"{fix['total_issues']} async/await issues in {fix['files_to_fix']} files")
            elif fix['type'] == 'circular_imports':
                print(f"{fix['cycles_found']} circular import cycles found")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        return report
    
    def apply_all_fixes(self, dry_run: bool = True):
        """Apply all fixes in sequence."""
        print("COMPREHENSIVE CODE QUALITY FIXES")
        print("="*60)
        print(f"Project root: {self.project_root}")
        print(f"Mode: {'DRY RUN' if dry_run else 'APPLYING FIXES'}")
        
        # Step 1: Check circular imports
        self.check_circular_imports()
        
        # Step 2: Fix imports
        self.fix_imports(dry_run=dry_run)
        
        # Step 3: Fix async/await
        self.fix_async_await(dry_run=dry_run)
        
        # Step 4: Analyze type hints
        self.analyze_type_hints()
        
        # Step 5: Create summary
        self.create_summary_report()
        
        if dry_run:
            print("\n" + "="*60)
            print("DRY RUN COMPLETE")
            print("To apply fixes, run with --apply flag")
        else:
            print("\n" + "="*60)
            print("FIXES APPLIED")
            print("Please review changes and run tests")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply comprehensive code quality fixes')
    parser.add_argument('--project-root', default='/workspace/src',
                       help='Root directory of the project')
    parser.add_argument('--apply', action='store_true',
                       help='Actually apply fixes (default is dry run)')
    
    args = parser.parse_args()
    
    fixer = CodeQualityFixer(args.project_root)
    fixer.apply_all_fixes(dry_run=not args.apply)


if __name__ == '__main__':
    main()