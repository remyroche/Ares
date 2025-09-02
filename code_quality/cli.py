#!/usr/bin/env python3
"""
Command-line interface for Code Quality Tools.
"""

import sys
import argparse
from pathlib import Path
import os

from .core import get_default_config, load_config
from .fixers.auto_fixer import AutoFixer
from .fixers.sequential_fixer import SequentialFixer
from .analyzers.linter_analyzer import LinterAnalyzer
from .analyzers.call_graph_analyzer import CallGraphAnalyzer
from .analyzers.dependency_analyzer import DependencyAnalyzer
from .analyzers.syntax_validator import SyntaxValidator
from .reporters.quality_reporter import QualityReporter


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Code Quality Tools - Comprehensive Python code analysis and fixing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive analysis
  python -m code_quality.cli analyze --path /path/to/code --output reports/
  
  # Auto-fix code issues
  python -m code_quality.cli fix --path /path/to/code
  
  # Run sequential auto-fix pipeline
  python -m code_quality.cli sequential-fix --target /path/to/code --output reports/
  
  # Validate syntax only
  python -m code_quality.cli validate --path /path/to/code
  
  # Map call graph
  python -m code_quality.cli call-graph --path /path/to/code --visualize
  
  # Analyze dependencies
  python -m code_quality.cli dependencies --path /path/to/code
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Comprehensive analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Run comprehensive code quality analysis')
    analyze_parser.add_argument('--path', required=True, help='Path to directory containing Python files')
    analyze_parser.add_argument('--output', help='Output directory for reports')
    analyze_parser.add_argument('--config', help='Path to configuration file')
    analyze_parser.add_argument('--auto-fix', action='store_true', help='Run auto-fixing before analysis')
    
    # Auto-fix command
    fix_parser = subparsers.add_parser('fix', help='Auto-fix code issues')
    fix_parser.add_argument('--path', required=True, help='Path to directory or file containing Python code')
    fix_parser.add_argument('--config', help='Path to configuration file')
    fix_parser.add_argument('--max-line-length', type=int, default=88, help='Maximum line length')
    fix_parser.add_argument('--aggressive', action='store_true', help='Enable aggressive fixing')
    
    # Sequential fix command
    sequential_parser = subparsers.add_parser('sequential-fix', help='Run sequential auto-fix pipeline')
    sequential_parser.add_argument('--target', required=True, 
                                 help='Path to Python file, directory, or comma-separated list of files')
    sequential_parser.add_argument('--config', help='Path to configuration file')
    sequential_parser.add_argument('--output', help='Output directory for reports')
    sequential_parser.add_argument('--no-backups', action='store_true', help='Disable backup creation')
    
    # Syntax validation command
    validate_parser = subparsers.add_parser('validate', help='Validate Python syntax')
    validate_parser.add_argument('--path', required=True, help='Path to directory containing Python files')
    validate_parser.add_argument('--config', help='Path to configuration file')
    validate_parser.add_argument('--output', help='Output file for validation report (JSON)')
    
    # Linter analysis command
    linter_parser = subparsers.add_parser('lint', help='Run linter analysis')
    linter_parser.add_argument('--path', required=True, help='Path to directory containing Python files')
    linter_parser.add_argument('--config', help='Path to configuration file')
    linter_parser.add_argument('--output', help='Output file for linter results (JSON)')
    
    # Call graph command
    call_graph_parser = subparsers.add_parser('call-graph', help='Analyze call graph and dependencies')
    call_graph_parser.add_argument('--path', required=True, help='Path to directory containing Python files')
    call_graph_parser.add_argument('--config', help='Path to configuration file')
    call_graph_parser.add_argument('--output', help='Output directory for results')
    call_graph_parser.add_argument('--format', choices=['json', 'dot', 'gexf'], default='json', 
                                 help='Output format for graph export')
    call_graph_parser.add_argument('--visualize', action='store_true', help='Generate graph visualization')
    
    # Dependency analysis command
    deps_parser = subparsers.add_parser('dependencies', help='Analyze package dependencies')
    deps_parser.add_argument('--path', required=True, help='Path to directory containing Python files')
    deps_parser.add_argument('--config', help='Path to configuration file')
    deps_parser.add_argument('--output', help='Output directory for results')
    deps_parser.add_argument('--generate-requirements', help='Generate requirements.txt file')
    deps_parser.add_argument('--check-security', action='store_true', help='Check for security vulnerabilities')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    try:
        if args.command == 'analyze':
            return _run_comprehensive_analysis(args, config)
        elif args.command == 'fix':
            return _run_auto_fix(args, config)
        elif args.command == 'sequential-fix':
            return _run_sequential_fix(args, config)
        elif args.command == 'validate':
            return _run_syntax_validation(args, config)
        elif args.command == 'lint':
            return _run_linter_analysis(args, config)
        elif args.command == 'call-graph':
            return _run_call_graph_analysis(args, config)
        elif args.command == 'dependencies':
            return _run_dependency_analysis(args, config)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def _run_comprehensive_analysis(args, config):
    """Run comprehensive code quality analysis."""
    print("Running comprehensive code quality analysis...")
    
    reporter = QualityReporter(config)
    results = reporter.generate_comprehensive_report(
        directory=args.path,
        run_auto_fix=args.auto_fix,
        output_dir=args.output
    )
    
    print("Analysis completed successfully!")
    return 0


def _run_auto_fix(args, config):
    """Run auto-fixing on code."""
    print("Running auto-fixer...")
    
    # Update config with command line arguments
    config.auto_fix.max_line_length = args.max_line_length
    config.auto_fix.aggressive = args.aggressive
    
    fixer = AutoFixer(config)
    
    # Check if path is a file or directory
    if os.path.isfile(args.path):
        results = fixer.fix_file(args.path)
    else:
        results = fixer.fix_all(args.path)
    
    # Print summary
    summary = fixer.get_fix_summary()
    print(f"\nAuto-fix completed!")
    print(f"Tools run: {', '.join(summary['tools_run'])}")
    print(f"Successful: {', '.join(summary['successful_tools'])}")
    print(f"Failed: {', '.join(summary['failed_tools'])}")
    
    return 0 if not summary['failed_tools'] else 1


def _run_sequential_fix(args, config):
    """Run sequential auto-fix pipeline."""
    print("Running sequential auto-fix pipeline...")
    
    # Parse target
    if "," in args.target:
        # Comma-separated list of files
        target = [f.strip() for f in args.target.split(",")]
    else:
        target = args.target
    
    fixer = SequentialFixer(config)
    results = fixer.run_pipeline(
        target=target,
        output_dir=args.output,
        create_backups=not args.no_backups
    )
    
    print("Sequential fix pipeline completed!")
    
    # Return appropriate exit code
    if results["summary"]["overall_status"] == "success":
        return 0
    elif results["summary"]["overall_status"] == "partial":
        return 1
    else:
        return 2


def _run_syntax_validation(args, config):
    """Run syntax validation."""
    print("Running syntax validation...")
    
    validator = SyntaxValidator(config)
    results = validator.validate_directory(args.path)
    
    # Print summary
    validator.print_summary()
    
    # Export results if requested
    if args.output:
        validator.export_report(args.output)
    
    # Return error code if there are syntax errors
    if results["summary"]["total_errors"] > 0:
        return 1
    
    return 0


def _run_linter_analysis(args, config):
    """Run linter analysis."""
    print("Running linter analysis...")
    
    analyzer = LinterAnalyzer(config)
    results = analyzer.analyze_directory(args.path)
    
    # Print summary
    print(f"\nLinter analysis completed!")
    print(f"Total issues: {results['total_issues']}")
    print(f"Files with issues: {results['total_files_with_issues']}")
    print(f"Errors: {results['total_errors']}")
    print(f"Warnings: {results['total_warnings']}")
    
    # Export results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return 0


def _run_call_graph_analysis(args, config):
    """Run call graph analysis."""
    print("Running call graph analysis...")
    
    analyzer = CallGraphAnalyzer(config)
    results = analyzer.analyze_directory(args.path)
    
    # Print summary
    print(f"\nCall graph analysis completed!")
    print(f"Total nodes: {results['total_nodes']}")
    print(f"Total edges: {results['total_edges']}")
    print(f"Potential dead code: {len(results['potential_dead_code'])}")
    print(f"Circular dependencies: {len(results['circular_dependencies'])}")
    
    # Export results if requested
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Export graph
        graph_file = output_dir / f"call_graph.{args.format}"
        analyzer.export_graph(str(graph_file), args.format)
        print(f"Call graph exported to {graph_file}")
        
        # Export analysis results
        results_file = output_dir / "call_graph_analysis.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Analysis results exported to {results_file}")
        
        # Generate visualization
        if args.visualize:
            viz_file = output_dir / "call_graph_visualization.png"
            analyzer.visualize_graph(str(viz_file))
            print(f"Graph visualization saved to {viz_file}")
    
    return 0


def _run_dependency_analysis(args, config):
    """Run dependency analysis."""
    print("Running dependency analysis...")
    
    analyzer = DependencyAnalyzer(config)
    results = analyzer.analyze_directory(args.path)
    
    # Print summary
    print(f"\nDependency analysis completed!")
    print(f"Total dependencies: {results['total_dependencies']}")
    print(f"Installed: {results['installed_dependencies']}")
    print(f"Missing: {results['missing_dependencies']}")
    print(f"Unused: {results['unused_dependencies']}")
    
    # Check security vulnerabilities
    if args.check_security:
        print("\nChecking security vulnerabilities...")
        vulnerabilities = analyzer.check_security_vulnerabilities()
        if vulnerabilities:
            print(f"Found {len(vulnerabilities)} security vulnerabilities:")
            for vuln in vulnerabilities[:5]:
                print(f"  - {vuln['package']}: {vuln['severity']} - {vuln['description']}")
        else:
            print("No security vulnerabilities found.")
    
    # Generate requirements.txt
    if args.generate_requirements:
        analyzer.generate_requirements_txt(args.generate_requirements)
    
    # Export results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Export analysis
        analysis_file = output_dir / "dependency_analysis.json"
        analyzer.export_analysis(str(analysis_file))
        print(f"\nAnalysis results exported to {analysis_file}")
        
        # Generate requirements.txt
        requirements_file = output_dir / "requirements_generated.txt"
        analyzer.generate_requirements_txt(str(requirements_file))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())