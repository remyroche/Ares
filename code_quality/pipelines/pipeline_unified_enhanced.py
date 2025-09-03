#!/usr/bin/env python3
"""
Unified Code Quality Pipeline - Enhanced Version with Unified Reporting

This version provides comprehensive unified reporting with per-file and
per-directory information using the ReportAggregator.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import ast

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all the tools
from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.safe_import_fixer import SafeImportFixer
from scripts.robust_async_fixer import RobustAsyncFixer
from scripts.enhanced_type_hints import TypeHintEnhancer
from scripts.detect_circular_imports import ImportAnalyzer as CircularImportDetector
from function_validator import FunctionValidator
from enhanced_validator import EnhancedValidator
from comprehensive_code_review import CodeQualityReviewer
from scripts.simple_interaction_mapper import extract_interactions, generate_report
from utils.report_aggregator import ReportAggregator
from analyzers.metrics_analyzer import MetricsAnalyzer
from analyzers.test_coverage_analyzer import TestCoverageAnalyzer
from analyzers.code_smell_detector import CodeSmellDetector
from analyzers.documentation_analyzer import DocumentationAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer
from analyzers.configuration_analyzer import ConfigurationAnalyzer
from analyzers.data_flow_analyzer import DataFlowAnalyzer


class UnifiedEnhancedPipeline:
    """Enhanced unified pipeline with comprehensive reporting."""
    
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
        
        # Add to aggregator
        self.report_aggregator.add_syntax_results(result)
        
        # Save individual report
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
        
        # Add to aggregator
        self.report_aggregator.add_import_results(result)
        
        # Save individual report
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
        
        # Add to aggregator
        self.report_aggregator.add_circular_import_results(result)
        
        # Save individual report
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
        
        # Add to aggregator
        self.report_aggregator.add_async_results(result)
        
        # Save individual report
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
        
        # Get all Python files
        python_files = []
        for pattern in ['**/*.py']:
            python_files.extend(self.project_root.glob(pattern))
        
        fixed_files = []
        failed_files = []
        
        for file_path in python_files:
            try:
                enhancer = TypeHintEnhancer()
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse and transform
                tree = ast.parse(content)
                new_tree = enhancer.visit(tree)
                
                if enhancer.changes_made:
                    # Generate new code
                    new_content = ast.unparse(new_tree)
                    
                    # Add necessary imports
                    if enhancer.imports_needed:
                        import_lines = []
                        if any('Path' in imp for imp in enhancer.imports_needed):
                            import_lines.append('from pathlib import Path')
                        if any('Union' in imp or 'Dict' in imp or 'List' in imp or 'Optional' in imp or 'Any' in imp or 'Tuple' in imp 
                               for imp in enhancer.imports_needed):
                            import_lines.append('from typing import Dict, List, Optional, Union, Any, Tuple')
                        
                        # Insert imports after module docstring and other imports
                        lines = new_content.split('\n')
                        insert_pos = 0
                        for i, line in enumerate(lines):
                            if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith('#'):
                                if line.startswith('import ') or line.startswith('from '):
                                    insert_pos = i + 1
                                else:
                                    break
                        
                        for imp in import_lines:
                            lines.insert(insert_pos, imp)
                            insert_pos += 1
                        
                        new_content = '\n'.join(lines)
                    
                    # Write back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    fixed_files.append({
                        'file': str(file_path),
                        'changes': enhancer.changes_made
                    })
                    
            except Exception as e:
                failed_files.append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        result = {
            'fixed_files': fixed_files,
            'failed_files': failed_files,
            'total_fixed': len(fixed_files),
            'total_failed': len(failed_files),
            'execution_time': time.time() - start_time
        }
        
        # Add to aggregator
        self.report_aggregator.add_type_results(result)
        
        # Save individual report
        report_path = self.reports_dir / f"type_hints_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
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
        
        # Add to aggregator
        self.report_aggregator.add_function_validation_results(result)
        
        # Save individual report
        report_path = self.reports_dir / f"function_validation_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_enhanced_validation(self) -> Dict[str, Any]:
        """Run enhanced validation for function arguments and data access."""
        print("\n" + "="*60)
        print("Running Enhanced Validation (Arguments & Data Access)")
        print("="*60)
        
        start_time = time.time()
        validator = EnhancedValidator(str(self.project_root))
        report = validator.validate_project()
        
        result = {
            'issues': report['issues'],
            'total_issues': report['summary']['total_issues'],
            'argument_mismatches': report['summary']['argument_mismatches'],
            'unsafe_data_access': report['summary']['unsafe_data_access'],
            'missing_null_checks': report['summary']['missing_null_checks'],
            'type_inconsistencies': report['summary']['type_inconsistencies'],
            'files_processed': report['summary']['files_processed'],
            'execution_time': time.time() - start_time,
            'data_access_summary': report.get('data_access_summary', {}),
            'function_signatures': len(report.get('function_signatures', {}))
        }
        
        # Add to aggregator
        self.report_aggregator.add_enhanced_validation_results(report)
        
        # Save individual report
        report_path = self.reports_dir / f"enhanced_validation_{self.timestamp}.json"
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
        
        # Add to aggregator
        self.report_aggregator.add_comprehensive_review_results(result)
        
        # Save individual report
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
        
    def run_metrics_analysis(self) -> Dict[str, Any]:
        """Run code metrics analysis."""
        print("\n" + "="*60)
        print("Running Code Metrics Analysis")
        print("="*60)
        
        start_time = time.time()
        analyzer = MetricsAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        result = analyzer.generate_report()
        result['execution_time'] = time.time() - start_time
        
        # Add to aggregator
        self.report_aggregator.file_metrics.update(analyzer.file_metrics)
        
        # Save report
        report_path = self.reports_dir / f"metrics_analysis_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_test_coverage_analysis(self) -> Dict[str, Any]:
        """Run test coverage analysis."""
        print("\n" + "="*60)
        print("Running Test Coverage Analysis")
        print("="*60)
        
        start_time = time.time()
        analyzer = TestCoverageAnalyzer(str(self.project_root))
        result = analyzer.analyze_project()
        result['execution_time'] = time.time() - start_time
        
        # Save report
        report_path = self.reports_dir / f"test_coverage_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_code_smell_detection(self) -> Dict[str, Any]:
        """Run code smell detection."""
        print("\n" + "="*60)
        print("Running Code Smell Detection")
        print("="*60)
        
        start_time = time.time()
        detector = CodeSmellDetector(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            detector.analyze_file(file_path)
            
        result = detector.generate_report()
        result['execution_time'] = time.time() - start_time
        
        # Save report
        report_path = self.reports_dir / f"code_smells_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_documentation_analysis(self) -> Dict[str, Any]:
        """Run documentation quality analysis."""
        print("\n" + "="*60)
        print("Running Documentation Analysis")
        print("="*60)
        
        start_time = time.time()
        analyzer = DocumentationAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        # Analyze README
        analyzer.analyze_readme()
        
        result = analyzer.generate_report()
        result['execution_time'] = time.time() - start_time
        
        # Save report
        report_path = self.reports_dir / f"documentation_analysis_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run performance analysis."""
        print("\n" + "="*60)
        print("Running Performance Analysis")
        print("="*60)
        
        start_time = time.time()
        analyzer = PerformanceAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        result = analyzer.generate_report()
        result['execution_time'] = time.time() - start_time
        
        # Save report
        report_path = self.reports_dir / f"performance_analysis_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_configuration_analysis(self) -> Dict[str, Any]:
        """Run configuration analysis."""
        print("\n" + "="*60)
        print("Running Configuration Analysis")
        print("="*60)
        
        start_time = time.time()
        analyzer = ConfigurationAnalyzer(str(self.project_root))
        result = analyzer.analyze_project()
        result['execution_time'] = time.time() - start_time
        
        # Save report
        report_path = self.reports_dir / f"configuration_analysis_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_data_flow_analysis(self) -> Dict[str, Any]:
        """Run data flow analysis."""
        print("\n" + "="*60)
        print("Running Data Flow Analysis")
        print("="*60)
        
        start_time = time.time()
        analyzer = DataFlowAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        result = analyzer.generate_report()
        result['execution_time'] = time.time() - start_time
        
        # Save report
        report_path = self.reports_dir / f"data_flow_analysis_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_all(self) -> Dict[str, Any]:
        """Run all code quality tools with unified reporting."""
        print(f"\n{'='*80}")
        print("UNIFIED CODE QUALITY PIPELINE - ENHANCED")
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
            'enhanced_validation': self.run_enhanced_validation(),
            'comprehensive_review': self.run_comprehensive_review(),
            'interaction_mapping': self.run_interaction_mapping(),
            'metrics': self.run_metrics_analysis(),
            'test_coverage': self.run_test_coverage_analysis(),
            'code_smells': self.run_code_smell_detection(),
            'documentation': self.run_documentation_analysis(),
            'performance': self.run_performance_analysis(),
            'configuration': self.run_configuration_analysis(),
            'data_flow': self.run_data_flow_analysis()
        }
        
        # Generate summary
        self.results['summary'] = self._generate_summary(time.time() - total_start)
        
        # Save individual pipeline report
        report_path = self.reports_dir / f"unified_pipeline_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Generate and save unified reports
        print("\n" + "="*60)
        print("Generating Unified Reports")
        print("="*60)
        
        json_report, md_report = self.report_aggregator.save_reports(
            self.reports_dir,
            "unified_code_quality_report"
        )
        
        print(f"\nUnified reports saved:")
        print(f"  JSON: {json_report}")
        print(f"  Markdown: {md_report}")
        
        # Print summary
        self._print_summary()
        
        # Print aggregated summary
        self._print_aggregated_summary()
        
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
        print("PIPELINE EXECUTION SUMMARY")
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
                    
    def _print_aggregated_summary(self):
        """Print aggregated report summary."""
        report = self.report_aggregator.generate_unified_report()
        summary = report['overall_summary']
        
        print(f"\n{'='*80}")
        print("UNIFIED CODE QUALITY SUMMARY")
        print(f"{'='*80}")
        print(f"Total Files Analyzed: {summary['total_files']}")
        print(f"Total Directories: {summary['total_directories']}")
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"Issues Fixed: {summary['fixed_issues']}")
        
        # Print enhanced validation specific stats if available
        if 'enhanced_validation' in self.results.get('analysis', {}):
            ev = self.results['analysis']['enhanced_validation']
            print("\nEnhanced Validation Results:")
            print(f"  - Argument Mismatches: {ev.get('argument_mismatches', 0)}")
            print(f"  - Unsafe Data Access: {ev.get('unsafe_data_access', 0)}")
            print(f"  - Missing Null Checks: {ev.get('missing_null_checks', 0)}")
            print(f"  - Type Inconsistencies: {ev.get('type_inconsistencies', 0)}")
        
        print("\nIssue Breakdown:")
        for issue_type, count in summary['issue_breakdown'].items():
            print(f"  {issue_type.replace('_', ' ').title()}: {count}")
            
        if summary['critical_files']:
            print("\nTop Files with Issues:")
            for i, file_info in enumerate(summary['critical_files'][:5]):
                file_name = Path(file_info['file']).name
                print(f"  {i+1}. {file_name}: {file_info['issues']} issues ({file_info['fixed']} fixed)")
                
        print(f"\nClean Files: {len(summary['clean_files'])}")
        print(f"\nReports saved to: {self.reports_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run unified code quality pipeline with enhanced reporting'
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
    
    pipeline = UnifiedEnhancedPipeline(args.project_root)
    
    # You could implement selective running based on args
    # For now, just run all
    pipeline.run_all()


if __name__ == '__main__':
    main()