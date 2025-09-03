#!/usr/bin/env python3
"""
Enhanced Visualization Script for Code Interactions.

This script generates comprehensive visual representations of code quality metrics,
including dependency graphs, complexity heatmaps, and interactive dashboards.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add visualizers to path
sys.path.insert(0, str(Path(__file__).parent))

from visualizers import (
    DependencyGraphVisualizer,
    ComplexityHeatmapVisualizer,
    InteractionNetworkVisualizer,
    DashboardGenerator
)


def visualize_code_interactions(analysis_results: Dict[str, Any], output_dir: str = None):
    """
    Generate comprehensive visualizations from code analysis results.
    
    Args:
        analysis_results: Complete analysis results from code quality tools
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = "code_quality/visualizations"
    
    print("CODE INTERACTION VISUALIZATION")
    print("=" * 80)
    print()
    
    # Create visualizers
    dep_viz = DependencyGraphVisualizer(output_dir)
    complexity_viz = ComplexityHeatmapVisualizer(output_dir)
    network_viz = InteractionNetworkVisualizer(output_dir)
    dashboard_gen = DashboardGenerator(output_dir)
    
    generated_files = []
    
    # 1. Dependency Visualizations
    if 'dependencies' in analysis_results:
        print("[1/5] Creating dependency visualizations...")
        
        deps = analysis_results['dependencies'].get('modules', {})
        if deps:
            # Main dependency graph
            fig, metadata = dep_viz.create_dependency_graph(deps, "Module Dependencies")
            files = dep_viz.save_figure(fig, "dependency_graph")
            generated_files.extend(files)
            dep_viz.save_metadata("dependency_graph", metadata)
            
            # Circular dependencies
            if metadata.get('circular_dependencies'):
                fig = dep_viz.create_circular_dependency_visualization(
                    metadata['circular_dependencies'],
                    "Circular Dependencies"
                )
                files = dep_viz.save_figure(fig, "circular_dependencies")
                generated_files.extend(files)
            
            # Module hierarchy
            fig = dep_viz.create_module_hierarchy(deps, "Module Hierarchy")
            files = dep_viz.save_figure(fig, "module_hierarchy")
            generated_files.extend(files)
            
            print(f"  ✓ Generated {len(files)} dependency visualizations")
    
    # 2. Complexity Visualizations
    if 'complexity' in analysis_results:
        print("[2/5] Creating complexity visualizations...")
        
        complexity_data = analysis_results['complexity'].get('files', {})
        if complexity_data:
            # Complexity heatmap
            fig, metadata = complexity_viz.create_complexity_heatmap(
                complexity_data,
                "Code Complexity Heatmap"
            )
            files = complexity_viz.save_figure(fig, "complexity_heatmap")
            generated_files.extend(files)
            complexity_viz.save_metadata("complexity_heatmap", metadata)
            
            # Treemap visualization
            fig = complexity_viz.create_treemap_visualization(
                complexity_data,
                'cyclomatic_complexity',
                "Complexity Treemap"
            )
            files = complexity_viz.save_figure(fig, "complexity_treemap")
            generated_files.extend(files)
            
            # Bubble chart
            fig = complexity_viz.create_module_complexity_bubble_chart(
                complexity_data,
                "Module Complexity Overview"
            )
            files = complexity_viz.save_figure(fig, "complexity_bubble_chart")
            generated_files.extend(files)
            
            print(f"  ✓ Generated {len(files)} complexity visualizations")
    
    # 3. Function Call Network
    if 'call_graph' in analysis_results:
        print("[3/5] Creating function call network...")
        
        call_graph = analysis_results['call_graph'].get('functions', {})
        if call_graph:
            # Function call network
            fig, metadata = network_viz.create_function_call_network(
                call_graph,
                "Function Call Network"
            )
            files = network_viz.save_figure(fig, "function_call_network")
            generated_files.extend(files)
            network_viz.save_metadata("function_call_network", metadata)
            
            # Interactive network
            html_file = network_viz.create_interactive_network(
                call_graph,
                title="Interactive Function Network"
            )
            generated_files.append(html_file)
            
            print(f"  ✓ Generated function call visualizations")
    
    # 4. Module Interactions
    if 'imports' in analysis_results:
        print("[4/5] Creating module interaction visualizations...")
        
        imports = analysis_results['imports'].get('files', {})
        interactions = {}
        
        for file, data in imports.items():
            module = file.replace('.py', '').replace('/', '.')
            interactions[module] = [
                imp.get('module', '') for imp in data.get('imports', [])
                if imp.get('module')
            ]
        
        if interactions:
            # Interaction matrix
            fig = network_viz.create_module_interaction_matrix(
                interactions,
                "Module Interaction Matrix"
            )
            files = network_viz.save_figure(fig, "interaction_matrix")
            generated_files.extend(files)
            
            print(f"  ✓ Generated module interaction visualizations")
    
    # 5. Interactive Dashboard
    print("[5/5] Creating interactive dashboard...")
    
    dashboard_file = dashboard_gen.generate_quality_dashboard(
        analysis_results,
        "Code Quality Dashboard"
    )
    generated_files.append(dashboard_file)
    
    print(f"  ✓ Generated interactive dashboard")
    
    # Summary
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Generated {len(generated_files)} visualization files:")
    for file in generated_files:
        print(f"  - {file}")
    
    return generated_files


def create_sample_visualizations():
    """Create sample visualizations with example data."""
    # Sample data for demonstration
    sample_data = {
        "dependencies": {
            "modules": {
                "main": ["config", "utils", "database"],
                "config": ["utils"],
                "database": ["config", "utils"],
                "api": ["database", "utils", "auth"],
                "auth": ["database", "utils"],
                "utils": []
            }
        },
        "complexity": {
            "files": {
                "main.py": {
                    "cyclomatic_complexity": 15,
                    "lines_of_code": 250,
                    "maintainability_index": 65
                },
                "database.py": {
                    "cyclomatic_complexity": 25,
                    "lines_of_code": 500,
                    "maintainability_index": 45
                },
                "api.py": {
                    "cyclomatic_complexity": 10,
                    "lines_of_code": 150,
                    "maintainability_index": 75
                },
                "utils.py": {
                    "cyclomatic_complexity": 5,
                    "lines_of_code": 100,
                    "maintainability_index": 85
                }
            },
            "average_complexity": 13.75
        },
        "call_graph": {
            "functions": {
                "main": ["init_config", "setup_database", "start_api"],
                "init_config": ["load_config", "validate_config"],
                "setup_database": ["connect_db", "migrate_db"],
                "start_api": ["create_app", "run_server"],
                "authenticate": ["check_token", "verify_user"],
                "process_request": ["validate_input", "execute_query", "format_response"]
            }
        },
        "issues": [
            {"file": "database.py", "message": "High cyclomatic complexity", "severity": "High", "line": 145},
            {"file": "main.py", "message": "Missing error handling", "severity": "Medium", "line": 87},
            {"file": "api.py", "message": "Unused import", "severity": "Low", "line": 5}
        ]
    }
    
    return visualize_code_interactions(sample_data)


def main():
    """Main entry point for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Generate visual representations of code quality metrics"
    )
    parser.add_argument(
        "--input", "-i",
        help="JSON file containing analysis results"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for visualizations",
        default="code_quality/visualizations"
    )
    parser.add_argument(
        "--sample", "-s",
        action="store_true",
        help="Generate sample visualizations with example data"
    )
    
    args = parser.parse_args()
    
    if args.sample:
        print("Generating sample visualizations...")
        create_sample_visualizations()
    elif args.input:
        # Load analysis results
        import json
        with open(args.input, 'r') as f:
            analysis_results = json.load(f)
        
        visualize_code_interactions(analysis_results, args.output)
    else:
        print("Error: Please provide either --input FILE or use --sample flag")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
