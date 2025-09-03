#!/usr/bin/env python3
"""
Code Analysis Pipeline

This pipeline handles code analysis and reporting:
1. Function validation
2. Code interaction mapping
3. Comprehensive code review
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from function_validator import FunctionValidator
from scripts.simple_interaction_mapper import extract_interactions, generate_report
from comprehensive_code_review import CodeQualityReviewer
from analyzers.metrics_analyzer import MetricsAnalyzer
from analyzers.test_coverage_analyzer import TestCoverageAnalyzer
from analyzers.code_smell_detector import CodeSmellDetector
from analyzers.documentation_analyzer import DocumentationAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer
from analyzers.configuration_analyzer import ConfigurationAnalyzer
from analyzers.data_flow_analyzer import DataFlowAnalyzer


class AnalysisPipeline:
    """Pipeline for code analysis and reporting."""
    
    def __init__(self, project_root: str = '/workspace/src'):
        self.project_root = Path(project_root)
        self.reports_dir = Path('/workspace/code_quality/reports')
        self.reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'function_validation': {},
            'interactions': {},
            'code_review': {},
            'summary': {}
        }
        
    def run_function_validation(self) -> Dict[str, Any]:
        """Run function validation checks."""
        print("\n" + "="*60)
        print("Running Function Validation")
        print("="*60)
        
        validator = FunctionValidator(str(self.project_root))
        validator.validate_all_files()
        
        result = {
            'issues': [issue.__dict__ for issue in validator.issues],
            'total_issues': len(validator.issues),
            'files_analyzed': validator.files_analyzed,
            'total_files': len(validator.files_analyzed),
            'issue_summary': validator.get_issue_summary()
        }
        
        # Save report
        report_path = self.reports_dir / f"function_validation_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        self.results['function_validation'] = result
        return result
        
    def run_interaction_mapping(self) -> Dict[str, Any]:
        """Run code interaction mapping."""
        print("\n" + "="*60)
        print("Running Code Interaction Mapping")
        print("="*60)
        
        # First run comprehensive review to get data
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
            'async_issues': len(interactions['async_patterns'])
        }
        
        # Save JSON report
        report_path = self.reports_dir / f"code_interactions_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        # Save text report
        text_report_path = self.reports_dir / f"code_interactions_{self.timestamp}.txt"
        with open(text_report_path, 'w') as f:
            f.write(report_content)
            
        self.results['interactions'] = result
        return result
        
    def run_comprehensive_review(self) -> Dict[str, Any]:
        """Run comprehensive code quality review."""
        print("\n" + "="*60)
        print("Running Comprehensive Code Review")
        print("="*60)
        
        reviewer = CodeQualityReviewer()
        reviewer.review_directory(str(self.project_root))
        report = reviewer.generate_report()
        
        result = {
            'issues': report['issues'],
            'total_issues': len(report['issues']),
            'summary': report['summary'],
            'metrics': report.get('metrics', {}),
            'security_issues': report.get('security_issues', []),
            'performance_issues': report.get('performance_issues', [])
        }
        
        # Save report
        report_path = self.reports_dir / f"comprehensive_review_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        self.results['code_review'] = result
        return result
        
    def run_metrics_analysis(self) -> Dict[str, Any]:
        """Run code metrics analysis."""
        print("\n" + "="*60)
        print("Running Code Metrics Analysis")
        print("="*60)
        
        analyzer = MetricsAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        result = analyzer.generate_report()
        
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
        
        analyzer = TestCoverageAnalyzer(str(self.project_root))
        result = analyzer.analyze_project()
        
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
        
        detector = CodeSmellDetector(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            detector.analyze_file(file_path)
            
        result = detector.generate_report()
        
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
        
        analyzer = DocumentationAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        # Analyze README
        analyzer.analyze_readme()
        
        result = analyzer.generate_report()
        
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
        
        analyzer = PerformanceAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        result = analyzer.generate_report()
        
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
        
        analyzer = ConfigurationAnalyzer(str(self.project_root))
        result = analyzer.analyze_project()
        
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
        
        analyzer = DataFlowAnalyzer(str(self.project_root))
        
        # Analyze all Python files
        python_files = list(self.project_root.rglob('*.py'))
        for file_path in python_files:
            analyzer.analyze_file(file_path)
            
        result = analyzer.generate_report()
        
        # Save report
        report_path = self.reports_dir / f"data_flow_analysis_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        print("\n" + "="*80)
        print("CODE ANALYSIS PIPELINE")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        
        # Run each step
        validation_result = self.run_function_validation()
        interaction_result = self.run_interaction_mapping()
        review_result = self.run_comprehensive_review()
        
        # Run new analyzers
        metrics_result = self.run_metrics_analysis()
        test_coverage_result = self.run_test_coverage_analysis()
        code_smell_result = self.run_code_smell_detection()
        documentation_result = self.run_documentation_analysis()
        performance_result = self.run_performance_analysis()
        configuration_result = self.run_configuration_analysis()
        data_flow_result = self.run_data_flow_analysis()
        
        # Create summary
        self.results['summary'] = {
            'timestamp': self.timestamp,
            'project_root': str(self.project_root),
            'function_validation': {
                'issues': validation_result['total_issues'],
                'files': validation_result['total_files']
            },
            'interactions': {
                'modules': interaction_result['module_count'],
                'functions': interaction_result['function_count'],
                'undefined': interaction_result['undefined_functions'],
                'async_issues': interaction_result['async_issues']
            },
            'code_review': {
                'total_issues': review_result['total_issues'],
                'security_issues': len(review_result['security_issues']),
                'performance_issues': len(review_result['performance_issues'])
            },
            'metrics': {
                'avg_complexity': metrics_result['summary']['average_cyclomatic_complexity'],
                'avg_maintainability': metrics_result['summary']['average_maintainability_index'],
                'total_loc': metrics_result['summary']['total_lines_of_code']
            },
            'test_coverage': {
                'overall_coverage': test_coverage_result['summary']['overall_coverage'],
                'untested_files': len(test_coverage_result['untested_files']),
                'test_to_code_ratio': test_coverage_result['summary']['test_to_code_ratio']
            },
            'code_smells': {
                'total_smells': code_smell_result['summary']['total_smells'],
                'high_severity': code_smell_result['summary']['high_severity'],
                'smell_types': code_smell_result['summary']['unique_smell_types']
            },
            'documentation': {
                'overall_coverage': documentation_result['summary']['overall_coverage'],
                'readme_quality': documentation_result['summary']['readme_quality'],
                'total_issues': documentation_result['summary']['total_issues']
            },
            'performance': {
                'total_issues': performance_result['summary']['total_issues'],
                'critical_issues': performance_result['summary']['critical_issues'],
                'high_severity': performance_result['summary']['high_severity']
            },
            'configuration': {
                'hardcoded_secrets': configuration_result['summary']['hardcoded_secrets'],
                'missing_required': configuration_result['summary']['missing_required'],
                'total_issues': configuration_result['summary']['total_issues']
            },
            'data_flow': {
                'total_issues': data_flow_result['summary']['total_issues'],
                'unused_variables': data_flow_result['summary']['unused_variables'],
                'uninitialized_usage': data_flow_result['summary']['uninitialized_usage']
            }
        }
        
        # Save comprehensive report
        report_path = self.reports_dir / f"analysis_pipeline_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Function validation: {validation_result['total_issues']} issues in {validation_result['total_files']} files")
        print(f"Code interactions: {interaction_result['module_count']} modules, {interaction_result['function_count']} functions")
        print(f"Comprehensive review: {review_result['total_issues']} total issues")
        
        print("\n--- New Analyzer Results ---")
        print(f"Code Metrics: Avg complexity: {metrics_result['summary']['average_cyclomatic_complexity']:.2f}, "
              f"Avg maintainability: {metrics_result['summary']['average_maintainability_index']:.2f}")
        print(f"Test Coverage: {test_coverage_result['summary']['overall_coverage']:.1f}% coverage, "
              f"{len(test_coverage_result['untested_files'])} untested files")
        print(f"Code Smells: {code_smell_result['summary']['total_smells']} smells found, "
              f"{code_smell_result['summary']['high_severity']} high severity")
        print(f"Documentation: {documentation_result['summary']['overall_coverage']:.1f}% documented")
        print(f"Performance: {performance_result['summary']['total_issues']} issues, "
              f"{performance_result['summary']['critical_issues']} critical")
        print(f"Configuration: {configuration_result['summary']['hardcoded_secrets']} hardcoded secrets, "
              f"{configuration_result['summary']['total_issues']} total issues")
        print(f"Data Flow: {data_flow_result['summary']['total_issues']} issues, "
              f"{data_flow_result['summary']['unused_variables']} unused variables")
        
        print(f"\nReports saved to: {self.reports_dir}")
        
        return self.results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run code analysis pipeline')
    parser.add_argument('--project-root', default='/workspace/src',
                        help='Project root directory')
    parser.add_argument('--validation-only', action='store_true',
                        help='Run only function validation')
    parser.add_argument('--interactions-only', action='store_true',
                        help='Run only interaction mapping')
    parser.add_argument('--review-only', action='store_true',
                        help='Run only comprehensive review')
    
    args = parser.parse_args()
    
    pipeline = AnalysisPipeline(args.project_root)
    
    if args.validation_only:
        pipeline.run_function_validation()
    elif args.interactions_only:
        pipeline.run_interaction_mapping()
    elif args.review_only:
        pipeline.run_comprehensive_review()
    else:
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()