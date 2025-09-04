"""
Command Line Interface for Dead Code Mapping Tools

Provides a unified CLI for all dead code detection and mapping capabilities.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add the parent directory to the path to import mapping modules
sys.path.insert(0, str(Path(__file__).parent))

from dead_code import (
    export_dead_code_mapping,
    get_removal_recommendations,
    map_dead_code,
)
from call_graph import (
    analyze_call_complexity,
    export_call_graph_mapping,
    find_orphaned_functions,
    get_function_usage_analysis,
    map_call_graph,
)
from data_flow import (
    analyze_variable_lifecycle,
    export_data_flow_mapping,
    find_data_flow_bottlenecks,
    map_data_flow,
    track_data_dependencies,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dead Code Mapping Tools - Comprehensive dead code detection and analysis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dead code analysis
    dead_code_parser = subparsers.add_parser("dead-code", help="Analyze dead code")
    dead_code_parser.add_argument("path", help="Path to directory or file to analyze")
    dead_code_parser.add_argument("--config", help="Path to configuration file")
    dead_code_parser.add_argument("--output", help="Output file path")
    dead_code_parser.add_argument("--format", choices=["json", "csv", "text"], default="json", help="Output format")
    dead_code_parser.add_argument("--no-deprecated", action="store_true", help="Skip deprecated code detection")
    dead_code_parser.add_argument("--no-dynamic", action="store_true", help="Skip dynamic import analysis")
    dead_code_parser.add_argument("--no-conditional", action="store_true", help="Skip conditional dead code detection")
    dead_code_parser.add_argument("--recommendations", action="store_true", help="Show removal recommendations")
    
    # Call graph analysis
    call_graph_parser = subparsers.add_parser("call-graph", help="Analyze call graph")
    call_graph_parser.add_argument("path", help="Path to directory to analyze")
    call_graph_parser.add_argument("--config", help="Path to configuration file")
    call_graph_parser.add_argument("--output", help="Output file path")
    call_graph_parser.add_argument("--format", choices=["json", "csv", "text"], default="json", help="Output format")
    call_graph_parser.add_argument("--no-dead-code", action="store_true", help="Skip dead code detection")
    call_graph_parser.add_argument("--no-unused-imports", action="store_true", help="Skip unused import detection")
    call_graph_parser.add_argument("--orphaned", action="store_true", help="Find orphaned functions")
    call_graph_parser.add_argument("--complexity", action="store_true", help="Analyze call complexity")
    call_graph_parser.add_argument("--usage", action="store_true", help="Analyze function usage")
    
    # Data flow analysis
    data_flow_parser = subparsers.add_parser("data-flow", help="Analyze data flow")
    data_flow_parser.add_argument("path", help="Path to directory to analyze")
    data_flow_parser.add_argument("--config", help="Path to configuration file")
    data_flow_parser.add_argument("--output", help="Output file path")
    data_flow_parser.add_argument("--format", choices=["json", "csv", "text"], default="json", help="Output format")
    data_flow_parser.add_argument("--no-variables", action="store_true", help="Skip variable tracking")
    data_flow_parser.add_argument("--no-functions", action="store_true", help="Skip function tracking")
    data_flow_parser.add_argument("--no-classes", action="store_true", help="Skip class tracking")
    data_flow_parser.add_argument("--lifecycle", action="store_true", help="Analyze variable lifecycle")
    data_flow_parser.add_argument("--bottlenecks", action="store_true", help="Find data flow bottlenecks")
    data_flow_parser.add_argument("--dependencies", action="store_true", help="Track data dependencies")
    
    # Comprehensive analysis
    comprehensive_parser = subparsers.add_parser("comprehensive", help="Run comprehensive analysis")
    comprehensive_parser.add_argument("path", help="Path to directory to analyze")
    comprehensive_parser.add_argument("--config", help="Path to configuration file")
    comprehensive_parser.add_argument("--output-dir", help="Output directory for all reports")
    comprehensive_parser.add_argument("--format", choices=["json", "csv", "text"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "dead-code":
            return handle_dead_code_command(args)
        elif args.command == "call-graph":
            return handle_call_graph_command(args)
        elif args.command == "data-flow":
            return handle_data_flow_command(args)
        elif args.command == "comprehensive":
            return handle_comprehensive_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_dead_code_command(args) -> int:
    """Handle dead code analysis command."""
    print(f"Analyzing dead code in: {args.path}")
    
    # Run dead code analysis
    result = map_dead_code(
        args.path,
        args.config,
        include_deprecated=not args.no_deprecated,
        include_dynamic_imports=not args.no_dynamic,
        include_conditional=not args.no_conditional,
    )
    
    # Print summary
    summary = result.get("summary", {})
    print(f"\nDead Code Analysis Summary:")
    print(f"  Total Issues: {summary.get('total_issues', 0)}")
    print(f"  Deprecated Code: {summary.get('deprecated_count', 0)}")
    print(f"  High Impact: {summary.get('high_impact_count', 0)}")
    print(f"  Medium Impact: {summary.get('medium_impact_count', 0)}")
    print(f"  Low Impact: {summary.get('low_impact_count', 0)}")
    print(f"  Potential Lines Removed: {summary.get('potential_lines_removed', 0)}")
    
    # Show recommendations if requested
    if args.recommendations:
        print(f"\nRemoval Recommendations:")
        recommendations = get_removal_recommendations(args.path, args.config)
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['priority'].upper()}: {rec['description']}")
            print(f"     File: {rec['file_path']}:{rec['line_number']}")
            print(f"     Action: {rec['action']}")
            print()
    
    # Export if output specified
    if args.output:
        export_dead_code_mapping(args.path, args.output, args.config, args.format)
        print(f"Results exported to: {args.output}")
    
    return 0


def handle_call_graph_command(args) -> int:
    """Handle call graph analysis command."""
    print(f"Analyzing call graph in: {args.path}")
    
    # Run call graph analysis
    result = map_call_graph(
        args.path,
        args.config,
        include_dead_code=not args.no_dead_code,
        include_unused_imports=not args.no_unused_imports,
    )
    
    # Print summary
    summary = result.get("summary", {})
    print(f"\nCall Graph Analysis Summary:")
    print(f"  Total Functions: {summary.get('total_functions', 0)}")
    print(f"  Total Calls: {summary.get('total_calls', 0)}")
    print(f"  Total Imports: {summary.get('total_imports', 0)}")
    print(f"  Circular Dependencies: {summary.get('circular_dependencies_count', 0)}")
    print(f"  Dead Functions: {summary.get('dead_functions_count', 0)}")
    print(f"  Unused Imports: {summary.get('unused_imports_count', 0)}")
    print(f"  Graph Density: {summary.get('graph_density', 0):.3f}")
    print(f"  Is DAG: {summary.get('is_dag', True)}")
    
    # Additional analyses if requested
    if args.orphaned:
        print(f"\nOrphaned Functions:")
        orphaned = find_orphaned_functions(args.path, args.config)
        for func in orphaned[:10]:  # Show top 10
            print(f"  {func['name']} in {func['file_path']}:{func['line']}")
        if len(orphaned) > 10:
            print(f"  ... and {len(orphaned) - 10} more")
    
    if args.complexity:
        print(f"\nCall Complexity Analysis:")
        complexity = analyze_call_complexity(args.path, args.config)
        high_complexity = complexity.get("high_complexity_functions", [])
        print(f"  High Complexity Functions: {len(high_complexity)}")
        for func in high_complexity[:5]:  # Show top 5
            print(f"    {func}")
    
    if args.usage:
        print(f"\nFunction Usage Analysis:")
        usage = get_function_usage_analysis(args.path, args.config)
        patterns = usage.get("patterns", {})
        print(f"  Unused Functions: {len(patterns.get('unused_functions', []))}")
        print(f"  Highly Used Functions: {len(patterns.get('highly_used_functions', []))}")
        print(f"  Complex Unused Functions: {len(patterns.get('complex_unused_functions', []))}")
    
    # Export if output specified
    if args.output:
        export_call_graph_mapping(args.path, args.output, args.config, args.format)
        print(f"Results exported to: {args.output}")
    
    return 0


def handle_data_flow_command(args) -> int:
    """Handle data flow analysis command."""
    print(f"Analyzing data flow in: {args.path}")
    
    # Run data flow analysis
    result = map_data_flow(
        args.path,
        args.config,
        track_variables=not args.no_variables,
        track_functions=not args.no_functions,
        track_classes=not args.no_classes,
    )
    
    # Print summary
    summary = result.get("summary", {})
    print(f"\nData Flow Analysis Summary:")
    print(f"  Total Variables: {summary.get('total_variables', 0)}")
    print(f"  Total Functions: {summary.get('total_functions', 0)}")
    print(f"  Total Classes: {summary.get('total_classes', 0)}")
    print(f"  Dead Variables: {summary.get('dead_variables_count', 0)}")
    print(f"  Unused Parameters: {summary.get('unused_parameters_count', 0)}")
    print(f"  Data Flow Complexity: {summary.get('data_flow_complexity', 0):.3f}")
    
    # Additional analyses if requested
    if args.lifecycle:
        print(f"\nVariable Lifecycle Analysis:")
        lifecycle = analyze_variable_lifecycle(args.path, args.config)
        patterns = lifecycle.get("lifecycle_patterns", {})
        print(f"  Immediate Death: {len(patterns.get('immediate_death', []))}")
        print(f"  Single Use: {len(patterns.get('single_use', []))}")
        print(f"  Multiple Assignments: {len(patterns.get('multiple_assignments', []))}")
    
    if args.bottlenecks:
        print(f"\nData Flow Bottlenecks:")
        bottlenecks = find_data_flow_bottlenecks(args.path, args.config)
        for bottleneck in bottlenecks[:5]:  # Show top 5
            print(f"  {bottleneck['function_name']}: {bottleneck['unused_parameters_count']} unused parameters")
    
    if args.dependencies:
        print(f"\nData Dependencies:")
        dependencies = track_data_dependencies(args.path, args.config)
        dep_summary = dependencies.get("summary", {})
        print(f"  Total Dependencies: {dep_summary.get('total_dependencies', 0)}")
        print(f"  Unused Dependencies: {dep_summary.get('unused_dependencies_count', 0)}")
    
    # Export if output specified
    if args.output:
        export_data_flow_mapping(args.path, args.output, args.config, args.format)
        print(f"Results exported to: {args.output}")
    
    return 0


def handle_comprehensive_command(args) -> int:
    """Handle comprehensive analysis command."""
    print(f"Running comprehensive analysis on: {args.path}")
    
    output_dir = Path(args.output_dir) if args.output_dir else Path("dead_code_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all analyses
    print("\n1. Dead Code Analysis...")
    dead_code_file = output_dir / f"dead_code_analysis.{args.format}"
    export_dead_code_mapping(args.path, str(dead_code_file), args.config, args.format)
    
    print("2. Call Graph Analysis...")
    call_graph_file = output_dir / f"call_graph_analysis.{args.format}"
    export_call_graph_mapping(args.path, str(call_graph_file), args.config, args.format)
    
    print("3. Data Flow Analysis...")
    data_flow_file = output_dir / f"data_flow_analysis.{args.format}"
    export_data_flow_mapping(args.path, str(data_flow_file), args.config, args.format)
    
    # Generate summary report
    print("4. Generating Summary Report...")
    summary_file = output_dir / "analysis_summary.txt"
    generate_summary_report(args.path, str(summary_file), args.config)
    
    print(f"\nComprehensive analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Dead Code: {dead_code_file}")
    print(f"  - Call Graph: {call_graph_file}")
    print(f"  - Data Flow: {data_flow_file}")
    print(f"  - Summary: {summary_file}")
    
    return 0


def generate_summary_report(path: str, output_file: str, config_path: str | None) -> None:
    """Generate a comprehensive summary report."""
    with open(output_file, "w") as f:
        f.write("COMPREHENSIVE DEAD CODE ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Path: {path}\n")
        f.write(f"Generated: {Path(__file__).name}\n\n")
        
        # Dead code summary
        dead_code_result = map_dead_code(path, config_path)
        dead_summary = dead_code_result.get("summary", {})
        f.write("DEAD CODE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Issues: {dead_summary.get('total_issues', 0)}\n")
        f.write(f"Deprecated Code: {dead_summary.get('deprecated_count', 0)}\n")
        f.write(f"High Impact: {dead_summary.get('high_impact_count', 0)}\n")
        f.write(f"Medium Impact: {dead_summary.get('medium_impact_count', 0)}\n")
        f.write(f"Low Impact: {dead_summary.get('low_impact_count', 0)}\n")
        f.write(f"Potential Lines Removed: {dead_summary.get('potential_lines_removed', 0)}\n\n")
        
        # Call graph summary
        call_graph_result = map_call_graph(path, config_path)
        cg_summary = call_graph_result.get("summary", {})
        f.write("CALL GRAPH ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Functions: {cg_summary.get('total_functions', 0)}\n")
        f.write(f"Dead Functions: {cg_summary.get('dead_functions_count', 0)}\n")
        f.write(f"Unused Imports: {cg_summary.get('unused_imports_count', 0)}\n")
        f.write(f"Circular Dependencies: {cg_summary.get('circular_dependencies_count', 0)}\n\n")
        
        # Data flow summary
        data_flow_result = map_data_flow(path, config_path)
        df_summary = data_flow_result.get("summary", {})
        f.write("DATA FLOW ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dead Variables: {df_summary.get('dead_variables_count', 0)}\n")
        f.write(f"Unused Parameters: {df_summary.get('unused_parameters_count', 0)}\n")
        f.write(f"Data Flow Complexity: {df_summary.get('data_flow_complexity', 0):.3f}\n\n")
        
        # Recommendations
        f.write("TOP RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        recommendations = get_removal_recommendations(path, config_path, 10)
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec['priority'].upper()}: {rec['description']}\n")
            f.write(f"   File: {rec['file_path']}:{rec['line_number']}\n")
            f.write(f"   Action: {rec['action']}\n\n")


if __name__ == "__main__":
    sys.exit(main())