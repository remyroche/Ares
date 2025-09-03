#!/usr/bin/env python3
"""
Master Code Quality Management Script

This script coordinates all code quality improvements and provides
a unified interface for maintaining code quality.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class CodeQualityMaster:
    def __init__(self):
        self.scripts_dir = Path('/workspace/code_quality/scripts')
        self.reports_dir = Path('/workspace/code_quality/reports')
        self.reports_dir.mkdir(exist_ok=True)
        
        self.scripts = {
            'import_fixer': 'safe_import_fixer.py',
            'syntax_fixer': 'advanced_syntax_fixer.py',
            'async_fixer': 'robust_async_fixer.py',
            'type_hints': 'enhanced_type_hints.py',
            'circular_imports': 'detect_circular_imports.py',
            'interaction_mapper': 'simple_interaction_mapper.py'
        }
        
    def run_script(self, script_name: str, args: List[str] = None) -> Dict[str, Any]:
        """Run a code quality script and capture results."""
        if script_name not in self.scripts:
            return {'error': f'Unknown script: {script_name}'}
        
        script_path = self.scripts_dir / self.scripts[script_name]
        cmd = ['python3', str(script_path)]
        
        if args:
            cmd.extend(args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='/workspace'
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of code quality."""
        print("Analyzing current code quality state...")
        print("=" * 60)
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'syntax_errors': 0,
            'import_issues': 0,
            'async_issues': 0,
            'type_coverage': 0.0,
            'circular_imports': 0
        }
        
        # Check syntax errors
        print("\n1. Checking syntax errors...")
        result = self.run_script('syntax_fixer', ['--project-root', '/workspace/src'])
        if 'files with syntax errors' in result.get('stdout', ''):
            # Parse the output
            for line in result['stdout'].split('\n'):
                if 'Found' in line and 'files with syntax errors' in line:
                    state['syntax_errors'] = int(line.split()[1])
        
        # Check import issues
        print("2. Checking import issues...")
        # This would analyze the existing reports
        import_report = self.reports_dir / 'import_fixes_report.json'
        if import_report.exists():
            with open(import_report, 'r') as f:
                data = json.load(f)
                state['import_issues'] = data.get('total_files', 0)
        
        # Check async issues
        print("3. Checking async/await issues...")
        async_report = self.reports_dir / 'async_fixes_report.json'
        if async_report.exists():
            with open(async_report, 'r') as f:
                data = json.load(f)
                state['async_issues'] = data.get('total_issues', 0)
        
        # Check type hint coverage
        print("4. Checking type hint coverage...")
        type_report = self.reports_dir / 'type_hints_report.json'
        if type_report.exists():
            with open(type_report, 'r') as f:
                data = json.load(f)
                state['type_coverage'] = data.get('overall_coverage', 0.0)
        
        # Check circular imports
        print("5. Checking circular imports...")
        circular_report = self.reports_dir / 'circular_imports_report.json'
        if circular_report.exists():
            with open(circular_report, 'r') as f:
                data = json.load(f)
                state['circular_imports'] = data.get('circular_imports', {}).get('count', 0)
        
        return state
    
    def apply_fixes(self, fix_types: List[str], dry_run: bool = True) -> Dict[str, Any]:
        """Apply selected fixes to the codebase."""
        results = {}
        
        mode = "DRY RUN" if dry_run else "APPLYING FIXES"
        print(f"\n{mode}")
        print("=" * 60)
        
        if 'syntax' in fix_types:
            print("\nFixing syntax errors...")
            args = ['--project-root', '/workspace/src']
            if not dry_run:
                args.append('--fix')
            results['syntax'] = self.run_script('syntax_fixer', args)
        
        if 'imports' in fix_types:
            print("\nFixing import issues...")
            args = ['--project-root', '/workspace/src']
            if not dry_run:
                args.append('--fix')
            results['imports'] = self.run_script('import_fixer', args)
        
        if 'async' in fix_types:
            print("\nFixing async/await issues...")
            args = ['--project-root', '/workspace/src']
            if not dry_run:
                args.append('--fix')
            results['async'] = self.run_script('async_fixer', args)
        
        if 'types' in fix_types:
            print("\nImproving type hint coverage...")
            args = ['--project-root', '/workspace/src', '--target', '0.9']
            results['types'] = self.run_script('type_hints', args)
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        print("\nGenerating summary report...")
        
        # Get current state
        current_state = self.analyze_current_state()
        
        # Load historical data if available
        history_file = self.reports_dir / 'quality_history.json'
        history = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        # Add current state to history
        history.append(current_state)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Generate summary
        summary = {
            'current_state': current_state,
            'improvements': {},
            'recommendations': []
        }
        
        # Calculate improvements if we have history
        if len(history) > 1:
            previous = history[-2]
            for key in ['syntax_errors', 'import_issues', 'async_issues']:
                if key in previous and key in current_state:
                    summary['improvements'][key] = previous[key] - current_state[key]
            
            if 'type_coverage' in previous and 'type_coverage' in current_state:
                summary['improvements']['type_coverage'] = (
                    current_state['type_coverage'] - previous['type_coverage']
                )
        
        # Add recommendations
        if current_state['syntax_errors'] > 0:
            summary['recommendations'].append(
                f"Fix {current_state['syntax_errors']} remaining syntax errors"
            )
        
        if current_state['async_issues'] > 0:
            summary['recommendations'].append(
                f"Add await statements to {current_state['async_issues']} async calls"
            )
        
        if current_state['type_coverage'] < 0.9:
            summary['recommendations'].append(
                f"Increase type hint coverage from {current_state['type_coverage']:.1%} to 90%+"
            )
        
        # Save summary
        summary_file = self.reports_dir / f'quality_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_dashboard(self):
        """Print a code quality dashboard."""
        state = self.analyze_current_state()
        
        print("\n" + "=" * 60)
        print("CODE QUALITY DASHBOARD")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Quality metrics
        print("QUALITY METRICS")
        print("-" * 30)
        print(f"Syntax Errors:     {state['syntax_errors']:>6} files")
        print(f"Import Issues:     {state['import_issues']:>6} files")
        print(f"Async Issues:      {state['async_issues']:>6} calls")
        print(f"Type Coverage:     {state['type_coverage']:>6.1%}")
        print(f"Circular Imports:  {state['circular_imports']:>6}")
        print()
        
        # Overall score (simple calculation)
        score = 100
        score -= min(state['syntax_errors'], 50) * 0.5  # -0.5 per syntax error, max -25
        score -= min(state['import_issues'], 100) * 0.2  # -0.2 per import issue, max -20
        score -= min(state['async_issues'], 100) * 0.1  # -0.1 per async issue, max -10
        score -= (1 - state['type_coverage']) * 20  # Max -20 for 0% coverage
        score -= state['circular_imports'] * 5  # -5 per circular import
        
        print(f"OVERALL QUALITY SCORE: {max(0, score):.1f}/100")
        print()
        
        # Recommendations
        print("RECOMMENDATIONS")
        print("-" * 30)
        if state['syntax_errors'] > 0:
            print(f"1. Run syntax fixer to fix {state['syntax_errors']} files")
        if state['import_issues'] > 0:
            print(f"2. Run import fixer to resolve {state['import_issues']} import issues")
        if state['async_issues'] > 0:
            print(f"3. Add await to {state['async_issues']} async function calls")
        if state['type_coverage'] < 0.9:
            print(f"4. Increase type coverage from {state['type_coverage']:.1%} to 90%+")
        
        print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Master Code Quality Management')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze current code quality state')
    parser.add_argument('--fix', nargs='+', 
                       choices=['syntax', 'imports', 'async', 'types', 'all'],
                       help='Apply specific fixes')
    parser.add_argument('--apply', action='store_true',
                       help='Actually apply fixes (default is dry run)')
    parser.add_argument('--dashboard', action='store_true',
                       help='Show code quality dashboard')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    master = CodeQualityMaster()
    
    if args.dashboard or (not any([args.analyze, args.fix, args.report])):
        master.print_dashboard()
    
    if args.analyze:
        state = master.analyze_current_state()
        print("\nCurrent state saved to reports directory")
    
    if args.fix:
        fix_types = args.fix
        if 'all' in fix_types:
            fix_types = ['syntax', 'imports', 'async', 'types']
        
        results = master.apply_fixes(fix_types, dry_run=not args.apply)
        
        if not args.apply:
            print("\nTo apply these fixes, run with --apply flag")
    
    if args.report:
        summary = master.generate_summary_report()
        print(f"\nSummary report generated: {list(summary.keys())}")


if __name__ == '__main__':
    main()