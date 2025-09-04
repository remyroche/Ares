"""
Data Flow Mapping Tools

Advanced data flow analysis and mapping utilities that track how data moves
through the codebase and identify potential dead code through data flow patterns.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from code_quality.analyzers.data_flow_analyzer import DataFlowAnalyzer
from code_quality.core.config import CodeQualityConfig, get_default_config, load_config


def _load_cq_config(config_path: str | None) -> CodeQualityConfig:
    """Load code quality configuration."""
    if config_path:
        return load_config(config_path)
    return get_default_config()


def map_data_flow(
    path: str,
    config_path: str | None = None,
    track_variables: bool = True,
    track_functions: bool = True,
    track_classes: bool = True,
) -> dict[str, Any]:
    """
    Map data flow through the codebase and identify unused data paths.

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file
        track_variables: Whether to track variable data flow
        track_functions: Whether to track function data flow
        track_classes: Whether to track class data flow

    Returns:
        Dictionary with data flow mapping results
    """
    directory = str(Path(path))
    config = _load_cq_config(config_path)

    analyzer = DataFlowAnalyzer(config)
    analysis = analyzer.analyze_directory(directory)
    
    result = {
        "directory_path": directory,
        "data_flow_analysis": analysis,
        "unused_data_paths": [],
        "dead_variables": [],
        "unused_parameters": [],
    }
    
    # Analyze for unused data paths
    if track_variables:
        result["dead_variables"] = _find_dead_variables(analysis)
    
    if track_functions:
        result["unused_parameters"] = _find_unused_parameters(analysis)
    
    # Generate summary
    result["summary"] = {
        "total_variables": len(analysis.get("variables", {})),
        "total_functions": len(analysis.get("functions", {})),
        "total_classes": len(analysis.get("classes", {})),
        "dead_variables_count": len(result["dead_variables"]),
        "unused_parameters_count": len(result["unused_parameters"]),
        "data_flow_complexity": _calculate_data_flow_complexity(analysis),
    }
    
    return result


def _find_dead_variables(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Find variables that are assigned but never used."""
    dead_variables = []
    variables = analysis.get("variables", {})
    
    for var_name, var_info in variables.items():
        # Check if variable is assigned but never read
        if var_info.get("assignments", 0) > 0 and var_info.get("reads", 0) == 0:
            dead_variables.append({
                "name": var_name,
                "file_path": var_info.get("file_path", ""),
                "line": var_info.get("line", 0),
                "assignments": var_info.get("assignments", 0),
                "reads": var_info.get("reads", 0),
                "scope": var_info.get("scope", ""),
                "type": var_info.get("type", ""),
            })
    
    return dead_variables


def _find_unused_parameters(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Find function parameters that are never used."""
    unused_parameters = []
    functions = analysis.get("functions", {})
    
    for func_name, func_info in functions.items():
        parameters = func_info.get("parameters", [])
        for param in parameters:
            if not param.get("used", False):
                unused_parameters.append({
                    "function_name": func_name,
                    "parameter_name": param.get("name", ""),
                    "file_path": func_info.get("file_path", ""),
                    "line": param.get("line", 0),
                    "parameter_type": param.get("type", ""),
                    "default_value": param.get("default_value", ""),
                })
    
    return unused_parameters


def _calculate_data_flow_complexity(analysis: dict[str, Any]) -> float:
    """Calculate data flow complexity score."""
    variables = analysis.get("variables", {})
    functions = analysis.get("functions", {})
    
    if not variables and not functions:
        return 0.0
    
    # Calculate complexity based on variable usage patterns
    total_variables = len(variables)
    unused_variables = sum(1 for var in variables.values() 
                          if var.get("assignments", 0) > 0 and var.get("reads", 0) == 0)
    
    # Calculate complexity based on function parameter usage
    total_parameters = sum(len(func.get("parameters", [])) for func in functions.values())
    unused_parameters = sum(1 for func in functions.values() 
                           for param in func.get("parameters", []) 
                           if not param.get("used", False))
    
    # Normalize complexity score (0-1, where 1 is most complex)
    variable_complexity = unused_variables / total_variables if total_variables > 0 else 0
    parameter_complexity = unused_parameters / total_parameters if total_parameters > 0 else 0
    
    return (variable_complexity + parameter_complexity) / 2


def export_data_flow_mapping(
    path: str,
    output_file: str,
    config_path: str | None = None,
    format: str = "json",
) -> str:
    """
    Export data flow mapping to a file.

    Args:
        path: Path to directory to analyze
        output_file: Output file path
        config_path: Optional path to configuration file
        format: Output format ("json", "csv", "text")

    Returns:
        Path to the exported file
    """
    mapping_data = map_data_flow(path, config_path)
    
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2, default=str)
    elif format.lower() == "csv":
        _export_data_flow_to_csv(mapping_data, out_path)
    elif format.lower() == "text":
        _export_data_flow_to_text(mapping_data, out_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(out_path)


def _export_data_flow_to_csv(mapping_data: dict[str, Any], output_path: Path) -> None:
    """Export data flow mapping data to CSV format."""
    import csv
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write dead variables
        writer.writerow(["Dead Variables - File", "Variable", "Line", "Assignments", "Reads", "Scope", "Type"])
        for var in mapping_data.get("dead_variables", []):
            writer.writerow([
                var.get("file_path", ""),
                var.get("name", ""),
                var.get("line", ""),
                var.get("assignments", ""),
                var.get("reads", ""),
                var.get("scope", ""),
                var.get("type", ""),
            ])
        
        # Write unused parameters
        writer.writerow([])  # Empty row
        writer.writerow(["Unused Parameters - File", "Function", "Parameter", "Line", "Type", "Default"])
        for param in mapping_data.get("unused_parameters", []):
            writer.writerow([
                param.get("file_path", ""),
                param.get("function_name", ""),
                param.get("parameter_name", ""),
                param.get("line", ""),
                param.get("parameter_type", ""),
                param.get("default_value", ""),
            ])


def _export_data_flow_to_text(mapping_data: dict[str, Any], output_path: Path) -> None:
    """Export data flow mapping data to text format."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("DATA FLOW MAPPING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        summary = mapping_data.get("summary", {})
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Variables: {summary.get('total_variables', 0)}\n")
        f.write(f"Total Functions: {summary.get('total_functions', 0)}\n")
        f.write(f"Total Classes: {summary.get('total_classes', 0)}\n")
        f.write(f"Dead Variables: {summary.get('dead_variables_count', 0)}\n")
        f.write(f"Unused Parameters: {summary.get('unused_parameters_count', 0)}\n")
        f.write(f"Data Flow Complexity: {summary.get('data_flow_complexity', 0):.3f}\n\n")
        
        # Dead variables
        f.write("DEAD VARIABLES\n")
        f.write("-" * 15 + "\n")
        for var in mapping_data.get("dead_variables", []):
            f.write(f"{var.get('name', '')} in {var.get('file_path', '')}:{var.get('line', '')}\n")
            f.write(f"  Assignments: {var.get('assignments', 0)}, Reads: {var.get('reads', 0)}\n")
            f.write(f"  Scope: {var.get('scope', '')}, Type: {var.get('type', '')}\n")
        
        # Unused parameters
        f.write("\n\nUNUSED PARAMETERS\n")
        f.write("-" * 18 + "\n")
        for param in mapping_data.get("unused_parameters", []):
            f.write(f"{param.get('parameter_name', '')} in {param.get('function_name', '')}\n")
            f.write(f"  File: {param.get('file_path', '')}:{param.get('line', '')}\n")
            f.write(f"  Type: {param.get('parameter_type', '')}, Default: {param.get('default_value', '')}\n")


def analyze_variable_lifecycle(
    path: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Analyze variable lifecycle and identify variables with unusual patterns.

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file

    Returns:
        Dictionary with variable lifecycle analysis
    """
    mapping_data = map_data_flow(path, config_path)
    analysis = mapping_data.get("data_flow_analysis", {})
    variables = analysis.get("variables", {})
    
    lifecycle_analysis = {
        "immediate_death": [],  # Variables assigned and never read
        "single_use": [],       # Variables assigned once and read once
        "multiple_assignments": [],  # Variables assigned multiple times
        "unused_assignments": [],    # Variables with unused assignments
    }
    
    for var_name, var_info in variables.items():
        assignments = var_info.get("assignments", 0)
        reads = var_info.get("reads", 0)
        
        if assignments > 0 and reads == 0:
            lifecycle_analysis["immediate_death"].append({
                "name": var_name,
                "file_path": var_info.get("file_path", ""),
                "line": var_info.get("line", 0),
                "assignments": assignments,
                "reads": reads,
            })
        elif assignments == 1 and reads == 1:
            lifecycle_analysis["single_use"].append({
                "name": var_name,
                "file_path": var_info.get("file_path", ""),
                "line": var_info.get("line", 0),
                "assignments": assignments,
                "reads": reads,
            })
        elif assignments > 1:
            lifecycle_analysis["multiple_assignments"].append({
                "name": var_name,
                "file_path": var_info.get("file_path", ""),
                "line": var_info.get("line", 0),
                "assignments": assignments,
                "reads": reads,
            })
    
    return {
        "lifecycle_patterns": lifecycle_analysis,
        "summary": {
            "total_variables": len(variables),
            "immediate_death_count": len(lifecycle_analysis["immediate_death"]),
            "single_use_count": len(lifecycle_analysis["single_use"]),
            "multiple_assignments_count": len(lifecycle_analysis["multiple_assignments"]),
        }
    }


def find_data_flow_bottlenecks(
    path: str,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Find data flow bottlenecks where data is frequently passed but not used.

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file

    Returns:
        List of data flow bottleneck information
    """
    mapping_data = map_data_flow(path, config_path)
    unused_parameters = mapping_data.get("unused_parameters", [])
    
    # Group unused parameters by function
    function_unused_params = {}
    for param in unused_parameters:
        func_name = param.get("function_name", "")
        if func_name not in function_unused_params:
            function_unused_params[func_name] = []
        function_unused_params[func_name].append(param)
    
    # Identify functions with many unused parameters (bottlenecks)
    bottlenecks = []
    for func_name, params in function_unused_params.items():
        if len(params) >= 2:  # Functions with 2+ unused parameters
            bottlenecks.append({
                "function_name": func_name,
                "file_path": params[0].get("file_path", ""),
                "unused_parameters_count": len(params),
                "unused_parameters": params,
                "severity": "high" if len(params) >= 4 else "medium",
            })
    
    return bottlenecks


def track_data_dependencies(
    path: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Track data dependencies and identify unused data paths.

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file

    Returns:
        Dictionary with data dependency tracking
    """
    mapping_data = map_data_flow(path, config_path)
    analysis = mapping_data.get("data_flow_analysis", {})
    
    # Build dependency graph
    dependencies = {}
    variables = analysis.get("variables", {})
    functions = analysis.get("functions", {})
    
    # Track variable dependencies
    for var_name, var_info in variables.items():
        if var_info.get("reads", 0) > 0:  # Only track variables that are read
            dependencies[var_name] = {
                "type": "variable",
                "file_path": var_info.get("file_path", ""),
                "line": var_info.get("line", 0),
                "dependents": [],  # Variables/functions that depend on this
                "dependencies": [],  # Variables/functions this depends on
                "usage_count": var_info.get("reads", 0),
            }
    
    # Track function dependencies
    for func_name, func_info in functions.items():
        dependencies[func_name] = {
            "type": "function",
            "file_path": func_info.get("file_path", ""),
            "line": func_info.get("line", 0),
            "dependents": [],
            "dependencies": [],
            "usage_count": func_info.get("calls", 0),
        }
    
    return {
        "dependency_graph": dependencies,
        "unused_dependencies": [name for name, info in dependencies.items() 
                               if info["usage_count"] == 0],
        "summary": {
            "total_dependencies": len(dependencies),
            "unused_dependencies_count": len([name for name, info in dependencies.items() 
                                            if info["usage_count"] == 0]),
        }
    }