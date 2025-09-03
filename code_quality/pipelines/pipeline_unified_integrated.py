#!/usr/bin/env python3
"""
Unified Code Quality Pipeline - Integrated Version

This version directly imports and uses the code quality modules for
better performance and integration.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all the tools
from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.safe_import_fixer import SafeImportFixer
from scripts.robust_async_fixer import RobustAsyncFixer
from scripts.enhanced_type_hints import TypeHintEnhancer
from scripts.detect_circular_imports import CircularImportDetector
from function_validator import FunctionValidator
from comprehensive_code_review import CodeQualityReviewer
from scripts.simple_interaction_mapper import extract_interactions, generate_report
from utils.report_aggregator import ReportAggregator


class UnifiedIntegratedPipeline:
    """Unified pipeline that directly uses imported code quality modules."""
    
    def __init__(self, project_root: str = '/workspace/src'):
        self.project_root = Path(project_root)
        self.reports_dir = Path('/workspace/code_quality/reports')
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'syntax_imports': {},
            'async_types': {},
            'analysis': {},
            'summary': {}
        }
        self.report_aggregator = ReportAggregator(project_root)
        
    def run_syntax_fixes(self) -> Dict[str, Any]:
        """Run advanced syntax fixes."""
        print("\n" + "="*60)
        print("Running Advanced Syntax Fixes")
        print("="*60)
        
        start_time = time.time()
        fixer = AdvancedSyntaxFixer(str(self.project_root))
        fixer.fix_all_files()
        
        result = {
            'fixed_files': fixer.fixed_files,
            'failed_files': fixer.failed_files,
            'syntax_errors': dict(fixer.syntax_errors),
            'total_fixed': len(fixer.fixed_files),
            'total_failed': len(fixer.failed_files),
            'execution_time': time.time() - start_time
        }
        
        # Save report
        report_path = self.reports_dir / f"syntax_fixes_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_import_fixes(self) -> Dict[str, Any]:
        """Run import fixes."""
        print("\n" + "="*60)
        print("Running Import Fixes")
        print("="*60)
        
        start_time = time.time()
        fixer = SafeImportFixer(str(self.project_root))
        fixer.fix_all_files()
        
        result = {
            'fixed_files': fixer.fixed_files,
            'failed_files': fixer.failed_files,
            'import_errors': dict(fixer.import_errors),
            'total_fixed': len(fixer.fixed_files),
            'total_failed': len(fixer.failed_files),
            'execution_time': time.time() - start_time
        }
        
        # Save report
        report_path = self.reports_dir / f"import_fixes_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def detect_circular_imports(self) -> Dict[str, Any]:
        """Detect circular imports."""
        print("\n" + "="*60)
        print("Detecting Circular Imports")
        print("="*60)
        
        start_time = time.time()
        detector = CircularImportDetector(str(self.project_root))
        cycles = detector.find_circular_imports()
        
        result = {
            'circular_imports': cycles,
            'total_cycles': len(cycles),
            'affected_modules': list(set(
                module for cycle in cycles 
                for module in cycle
            )),
            'execution_time': time.time() - start_time
        }
        
        # Save report
        report_path = self.reports_dir / f"circular_imports_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_async_fixes(self) -> Dict[str, Any]:
        """Run robust async/await fixes."""
        print("\n" + "="*60)
        print("Running Async/Await Fixes")
        print("="*60)
        
        start_time = time.time()
        fixer = RobustAsyncFixer(str(self.project_root))
        fixer.fix_all_files()
        
        result = {
            'fixed_files': fixer.fixed_files,
            'failed_files': fixer.failed_files,
            'total_fixed': len(fixer.fixed_files),
            'total_failed': len(fixer.failed_files),
            'execution_time': time.time() - start_time
        }
        
        # Save report
        report_path = self.reports_dir / f"async_fixes_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_type_hints(self) -> Dict[str, Any]:
        """Run type hint enhancements."""
        print("\n" + "="*60)
        print("Running Type Hint Enhancements")
        print("="*60)
        
        start_time = time.time()
        
        # This would need to be implemented properly
        # For now, return a placeholder
        result = {
            'message': 'Type hints enhancement needs AST implementation',
            'execution_time': time.time() - start_time
        }
        
        return result
        
    def run_function_validation(self) -> Dict[str, Any]:
        """Run function validation checks."""
        print("\n" + "="*60)
        print("Running Function Validation")
        print("="*60)
        
        start_time = time.time()
        validator = FunctionValidator(str(self.project_root))
        validator.validate_all_files()
        
        result = {
            'issues': [issue.__dict__ for issue in validator.issues],
            'total_issues': len(validator.issues),
            'files_analyzed': validator.files_analyzed,
            'total_files': len(validator.files_analyzed),
            'issue_summary': validator.get_issue_summary(),
            'execution_time': time.time() - start_time
        }
        
        # Save report
        report_path = self.reports_dir / f"function_validation_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_comprehensive_review(self) -> Dict[str, Any]:
        """Run comprehensive code quality review."""
        print("\n" + "="*60)
        print("Running Comprehensive Code Review")
        print("="*60)
        
        start_time = time.time()
        reviewer = CodeQualityReviewer()
        reviewer.review_directory(str(self.project_root))
        report = reviewer.generate_report()
        
        result = {
            'issues': report['issues'],
            'total_issues': len(report['issues']),
            'summary': report['summary'],
            'metrics': report.get('metrics', {}),
            'security_issues': report.get('security_issues', []),
            'performance_issues': report.get('performance_issues', []),
            'execution_time': time.time() - start_time
        }
        
        # Save report
        report_path = self.reports_dir / f"comprehensive_review_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_interaction_mapping(self) -> Dict[str, Any]:
        """Run code interaction mapping."""
        print("\n" + "="*60)
        print("Running Code Interaction Mapping")
        print("="*60)
        
        start_time = time.time()
        
        # Use the comprehensive review data
        reviewer = CodeQualityReviewer()
        reviewer.review_directory(str(self.project_root))
        report_data = reviewer.generate_report()
        
        # Extract interactions
        interactions = extract_interactions(report_data)
        
        # Generate readable report
        report_content = generate_report(interactions)
        
        result = {
            'interactions': interactions,
            'module_count': len(interactions['import_graph']),
            'function_count': len(interactions['function_definitions']),
            'undefined_functions': len(interactions['undefined_functions']),
            'async_issues': len(interactions['async_patterns']),
            'execution_time': time.time() - start_time
        }
        
        # Save reports
        json_path = self.reports_dir / f"code_interactions_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        text_path = self.reports_dir / f"code_interactions_{self.timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write(report_content)
            
        return result
        
    def run_all(self) -> Dict[str, Any]:
        """Run all code quality tools."""
        print(f"\n{'='*80}")
        print("UNIFIED CODE QUALITY PIPELINE - INTEGRATED")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        
        total_start = time.time()
        
        # Syntax and Imports
        self.results['syntax_imports'] = {
            'syntax_fixes': self.run_syntax_fixes(),
            'import_fixes': self.run_import_fixes(),
            'circular_imports': self.detect_circular_imports()
        }
        
        # Async and Types
        self.results['async_types'] = {
            'async_fixes': self.run_async_fixes(),
            'type_hints': self.run_type_hints()
        }
        
        # Analysis
        self.results['analysis'] = {
            'function_validation': self.run_function_validation(),
            'comprehensive_review': self.run_comprehensive_review(),
            'interaction_mapping': self.run_interaction_mapping()
        }
        
        # Generate summary
        self.results['summary'] = self._generate_summary(time.time() - total_start)
        
        # Save comprehensive report
        report_path = self.reports_dir / f"unified_integrated_pipeline_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Print summary
        self._print_summary()
        
        return self.results
        
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate summary of all results."""
        summary = {
            'timestamp': self.timestamp,
            'project_root': str(self.project_root),
            'total_execution_time': total_time,
            'categories': {}
        }
        
        for category, tools in self.results.items():
            if category == 'summary':
                continue
                
            category_summary = {}
            for tool_name, result in tools.items():
                if isinstance(result, dict):
                    category_summary[tool_name] = {
                        'execution_time': result.get('execution_time', 0),
                        'issues_fixed': result.get('total_fixed', 0),
                        'issues_found': result.get('total_issues', 0),
                        'files_processed': result.get('total_files', 0)
                    }
                    
            summary['categories'][category] = category_summary
            
        return summary
        
    def _print_summary(self):
        """Print a formatted summary."""
        summary = self.results['summary']
        
        print(f"\n{'='*80}")
        print("PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"Total execution time: {summary['total_execution_time']:.2f} seconds")
        
        for category, tools in summary['categories'].items():
            print(f"\n{category.upper()}:")
            for tool, info in tools.items():
                print(f"  {tool}:")
                print(f"    Execution time: {info['execution_time']:.2f}s")
                if info['issues_fixed']:
                    print(f"    Issues fixed: {info['issues_fixed']}")
                if info['issues_found']:
                    print(f"    Issues found: {info['issues_found']}")
                if info['files_processed']:
                    print(f"    Files processed: {info['files_processed']}")
                    
        print(f"\nReports saved to: {self.reports_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run unified code quality pipeline (integrated version)'
    )
    parser.add_argument('--project-root', default='/workspace/src',
                        help='Project root directory')
    parser.add_argument('--skip-syntax', action='store_true',
                        help='Skip syntax and import fixes')
    parser.add_argument('--skip-async', action='store_true',
                        help='Skip async and type fixes')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip code analysis')
    
    args = parser.parse_args()
    
    pipeline = UnifiedIntegratedPipeline(args.project_root)
    
    # You could implement selective running based on args
    # For now, just run all
    pipeline.run_all()


if __name__ == '__main__':
    main()