#!/usr/bin/env python3
"""
Syntax and Import Pipeline

This pipeline handles all syntax-related fixes and import management:
1. Advanced syntax fixing
2. Import fixing and management
3. Circular import detection
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.safe_import_fixer import SafeImportFixer
from scripts.detect_circular_imports import CircularImportDetector


class SyntaxImportPipeline:
    """Pipeline for syntax and import-related fixes."""
    
    def __init__(self, project_root: str = '/workspace/src'):
        self.project_root = Path(project_root)
        self.reports_dir = Path('/workspace/code_quality/reports')
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'syntax_fixes': {},
            'import_fixes': {},
            'circular_imports': {},
            'summary': {}
        }
        
    def run_syntax_fixes(self) -> Dict[str, Any]:
        """Run advanced syntax fixes."""
        print("\n" + "="*60)
        print("Running Advanced Syntax Fixes")
        print("="*60)
        
        fixer = AdvancedSyntaxFixer(str(self.project_root))
        fixer.fix_all_files()
        
        result = {
            'fixed_files': fixer.fixed_files,
            'failed_files': fixer.failed_files,
            'syntax_errors': dict(fixer.syntax_errors),
            'total_fixed': len(fixer.fixed_files),
            'total_failed': len(fixer.failed_files)
        }
        
        # Save report
        report_path = self.reports_dir / f"syntax_fixes_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        self.results['syntax_fixes'] = result
        return result
        
    def run_import_fixes(self) -> Dict[str, Any]:
        """Run import fixes."""
        print("\n" + "="*60)
        print("Running Import Fixes")
        print("="*60)
        
        fixer = SafeImportFixer(str(self.project_root))
        fixer.fix_all_files()
        
        result = {
            'fixed_files': fixer.fixed_files,
            'failed_files': fixer.failed_files,
            'import_errors': dict(fixer.import_errors),
            'total_fixed': len(fixer.fixed_files),
            'total_failed': len(fixer.failed_files)
        }
        
        # Save report
        report_path = self.reports_dir / f"import_fixes_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        self.results['import_fixes'] = result
        return result
        
    def detect_circular_imports(self) -> Dict[str, Any]:
        """Detect circular imports."""
        print("\n" + "="*60)
        print("Detecting Circular Imports")
        print("="*60)
        
        detector = CircularImportDetector(str(self.project_root))
        cycles = detector.find_circular_imports()
        
        result = {
            'circular_imports': cycles,
            'total_cycles': len(cycles),
            'affected_modules': list(set(
                module for cycle in cycles 
                for module in cycle
            ))
        }
        
        # Save report
        report_path = self.reports_dir / f"circular_imports_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        self.results['circular_imports'] = result
        return result
        
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete syntax and import pipeline."""
        print("\n" + "="*80)
        print("SYNTAX AND IMPORT PIPELINE")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        
        # Run each step
        syntax_result = self.run_syntax_fixes()
        import_result = self.run_import_fixes()
        circular_result = self.detect_circular_imports()
        
        # Create summary
        self.results['summary'] = {
            'timestamp': self.timestamp,
            'project_root': str(self.project_root),
            'syntax_fixes': {
                'fixed': syntax_result['total_fixed'],
                'failed': syntax_result['total_failed']
            },
            'import_fixes': {
                'fixed': import_result['total_fixed'],
                'failed': import_result['total_failed']
            },
            'circular_imports': {
                'cycles_found': circular_result['total_cycles'],
                'affected_modules': len(circular_result['affected_modules'])
            }
        }
        
        # Save comprehensive report
        report_path = self.reports_dir / f"syntax_import_pipeline_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Syntax fixes: {syntax_result['total_fixed']} fixed, {syntax_result['total_failed']} failed")
        print(f"Import fixes: {import_result['total_fixed']} fixed, {import_result['total_failed']} failed")
        print(f"Circular imports: {circular_result['total_cycles']} cycles found")
        print(f"\nReports saved to: {self.reports_dir}")
        
        return self.results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run syntax and import fixes pipeline')
    parser.add_argument('--project-root', default='/workspace/src',
                        help='Project root directory')
    parser.add_argument('--syntax-only', action='store_true',
                        help='Run only syntax fixes')
    parser.add_argument('--imports-only', action='store_true',
                        help='Run only import fixes')
    parser.add_argument('--circular-only', action='store_true',
                        help='Run only circular import detection')
    
    args = parser.parse_args()
    
    pipeline = SyntaxImportPipeline(args.project_root)
    
    if args.syntax_only:
        pipeline.run_syntax_fixes()
    elif args.imports_only:
        pipeline.run_import_fixes()
    elif args.circular_only:
        pipeline.detect_circular_imports()
    else:
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()