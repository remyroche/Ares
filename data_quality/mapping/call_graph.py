"""
Call Graph Mapping Tools

Advanced call graph analysis and mapping utilities that provide comprehensive
function call relationship analysis and dead code detection through call patterns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_quality.analyzers.call_graph_analyzer import CallGraphAnalyzer
from code_quality.core.config import CodeQualityConfig, get_default_config, load_config


def _load_cq_config(config_path: str | None) -> CodeQualityConfig:
    """Load code quality configuration."""
    if config_path:
        return load_config(config_path)
    return get_default_config()


def map_call_graph(
    path: str,
    config_path: str | None = None,
    include_dead_code: bool = True,
    include_unused_imports: bool = True,
) -> dict[str, Any]:
    """
    Map call graph relationships and identify dead code through call patterns.

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file
        include_dead_code: Whether to include dead code detection
        include_unused_imports: Whether to include unused import detection

    Returns:
        Dictionary with call graph mapping results
    """
    directory = str(Path(path))
    config = _load_cq_config(config_path)

    analyzer = CallGraphAnalyzer(config)
    analysis = analyzer.analyze_directory(directory)
    
    result = {
        "directory_path": directory,
        "call_graph": {
            "total_functions": len(analysis.get("functions", {})),
            "total_calls": len(analysis.get("call_relationships", [])),
            "total_imports": len(analysis.get("import_relationships", [])),
            "circular_dependencies": analysis.get("circular_dependencies", []),
            "graph_metrics": analysis.get("graph_metrics", {}),
        },
        "functions": analysis.get("functions", {}),
        "call_relationships": analysis.get("call_relationships", []),
        "import_relationships": analysis.get("import_relationships", []),
    }
    
    if include_dead_code:
        dead_code = analyzer.find_dead_code()
        result["dead_code"] = {
            "unused_functions": dead_code,
            "count": len(dead_code),
        }
    
    if include_unused_imports:
        unused_imports = analyzer.find_unused_imports()
        result["unused_imports"] = {
            "imports": unused_imports,
            "count": len(unused_imports),
        }
    
    # Generate summary statistics
    result["summary"] = {
        "total_functions": len(analysis.get("functions", {})),
        "total_calls": len(analysis.get("call_relationships", [])),
        "total_imports": len(analysis.get("import_relationships", [])),
        "circular_dependencies_count": len(analysis.get("circular_dependencies", [])),
        "dead_functions_count": len(dead_code) if include_dead_code else 0,
        "unused_imports_count": len(unused_imports) if include_unused_imports else 0,
        "graph_density": analysis.get("graph_metrics", {}).get("density", 0),
        "is_dag": analysis.get("graph_metrics", {}).get("is_dag", True),
    }
    
    return result


def export_call_graph_mapping(
    path: str,
    output_file: str,
    config_path: str | None = None,
    format: str = "json",
) -> str:
    """
    Export call graph mapping to a file.

    Args:
        path: Path to directory to analyze
        output_file: Output file path
        config_path: Optional path to configuration file
        format: Output format ("json", "csv", "text")

    Returns:
        Path to the exported file
    """
    mapping_data = map_call_graph(path, config_path)
    
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2, default=str)
    elif format.lower() == "csv":
        _export_call_graph_to_csv(mapping_data, out_path)
    elif format.lower() == "text":
        _export_call_graph_to_text(mapping_data, out_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(out_path)


def _export_call_graph_to_csv(mapping_data: dict[str, Any], output_path: Path) -> None:
    """Export call graph mapping data to CSV format."""
    import csv
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write call relationships
        writer.writerow(["Caller File", "Caller Function", "Callee File", "Callee Function", "Call Type"])
        for call in mapping_data.get("call_relationships", []):
            caller = call.get("caller", {})
            callee = call.get("callee", {})
            writer.writerow([
                caller.get("file_path", ""),
                caller.get("name", ""),
                callee.get("file_path", ""),
                callee.get("name", ""),
                call.get("call_type", ""),
            ])
        
        # Write dead code
        writer.writerow([])  # Empty row
        writer.writerow(["Dead Code - File", "Function", "Line", "Type", "Module"])
        for dead in mapping_data.get("dead_code", {}).get("unused_functions", []):
            writer.writerow([
                dead.get("file_path", ""),
                dead.get("name", ""),
                dead.get("line", ""),
                dead.get("node_type", ""),
                dead.get("module_path", ""),
            ])


def _export_call_graph_to_text(mapping_data: dict[str, Any], output_path: Path) -> None:
    """Export call graph mapping data to text format."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("CALL GRAPH MAPPING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        summary = mapping_data.get("summary", {})
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Functions: {summary.get('total_functions', 0)}\n")
        f.write(f"Total Calls: {summary.get('total_calls', 0)}\n")
        f.write(f"Total Imports: {summary.get('total_imports', 0)}\n")
        f.write(f"Circular Dependencies: {summary.get('circular_dependencies_count', 0)}\n")
        f.write(f"Dead Functions: {summary.get('dead_functions_count', 0)}\n")
        f.write(f"Unused Imports: {summary.get('unused_imports_count', 0)}\n")
        f.write(f"Graph Density: {summary.get('graph_density', 0):.3f}\n")
        f.write(f"Is DAG: {summary.get('is_dag', True)}\n\n")
        
        # Call relationships
        f.write("CALL RELATIONSHIPS\n")
        f.write("-" * 20 + "\n")
        for call in mapping_data.get("call_relationships", []):
            caller = call.get("caller", {})
            callee = call.get("callee", {})
            f.write(f"{caller.get('name', '')} -> {callee.get('name', '')}\n")
            f.write(f"  {caller.get('file_path', '')}:{caller.get('line', '')} -> {callee.get('file_path', '')}:{callee.get('line', '')}\n")
        
        # Dead code
        f.write("\n\nDEAD CODE (UNUSED FUNCTIONS)\n")
        f.write("-" * 30 + "\n")
        for dead in mapping_data.get("dead_code", {}).get("unused_functions", []):
            f.write(f"{dead.get('name', '')} in {dead.get('file_path', '')}:{dead.get('line', '')}\n")
            f.write(f"  Type: {dead.get('node_type', '')}\n")
            f.write(f"  Module: {dead.get('module_path', '')}\n")


def get_function_usage_analysis(
    path: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Analyze function usage patterns and identify potential issues.

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file

    Returns:
        Dictionary with function usage analysis
    """
    mapping_data = map_call_graph(path, config_path)
    
    functions = mapping_data.get("functions", {})
    call_relationships = mapping_data.get("call_relationships", [])
    
    # Analyze function usage
    usage_stats = {}
    for func_name, func_info in functions.items():
        # Count how many times this function is called
        call_count = sum(1 for call in call_relationships 
                        if call.get("callee", {}).get("name") == func_name)
        
        # Count how many functions this function calls
        calls_made = sum(1 for call in call_relationships 
                        if call.get("caller", {}).get("name") == func_name)
        
        usage_stats[func_name] = {
            "file_path": func_info.get("file_path", ""),
            "line": func_info.get("line", 0),
            "node_type": func_info.get("node_type", ""),
            "times_called": call_count,
            "functions_called": calls_made,
            "is_used": call_count > 0,
            "complexity": func_info.get("complexity", 0),
        }
    
    # Identify patterns
    unused_functions = [name for name, stats in usage_stats.items() if not stats["is_used"]]
    highly_used_functions = [name for name, stats in usage_stats.items() if stats["times_called"] > 5]
    complex_unused_functions = [name for name, stats in usage_stats.items() 
                              if not stats["is_used"] and stats["complexity"] > 5]
    
    return {
        "usage_statistics": usage_stats,
        "patterns": {
            "unused_functions": unused_functions,
            "highly_used_functions": highly_used_functions,
            "complex_unused_functions": complex_unused_functions,
        },
        "summary": {
            "total_functions": len(functions),
            "unused_count": len(unused_functions),
            "highly_used_count": len(highly_used_functions),
            "complex_unused_count": len(complex_unused_functions),
        }
    }


def find_orphaned_functions(
    path: str,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Find functions that are not called by any other function (orphaned).

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file

    Returns:
        List of orphaned function information
    """
    mapping_data = map_call_graph(path, config_path)
    
    functions = mapping_data.get("functions", {})
    call_relationships = mapping_data.get("call_relationships", [])
    
    # Create a set of all called functions
    called_functions = set()
    for call in call_relationships:
        callee_name = call.get("callee", {}).get("name")
        if callee_name:
            called_functions.add(callee_name)
    
    # Find functions that are never called
    orphaned = []
    for func_name, func_info in functions.items():
        if func_name not in called_functions:
            # Skip special methods and main functions
            if not func_name.startswith("__") and func_name != "main":
                orphaned.append({
                    "name": func_name,
                    "file_path": func_info.get("file_path", ""),
                    "line": func_info.get("line", 0),
                    "node_type": func_info.get("node_type", ""),
                    "module_path": func_info.get("module_path", ""),
                    "complexity": func_info.get("complexity", 0),
                })
    
    return orphaned


def analyze_call_complexity(
    path: str,
    config_path: str | None = None,
    complexity_threshold: int = 10,
) -> dict[str, Any]:
    """
    Analyze call complexity and identify functions with high call complexity.

    Args:
        path: Path to directory to analyze
        config_path: Optional path to configuration file
        complexity_threshold: Threshold for high complexity

    Returns:
        Dictionary with call complexity analysis
    """
    mapping_data = map_call_graph(path, config_path)
    
    functions = mapping_data.get("functions", {})
    call_relationships = mapping_data.get("call_relationships", [])
    
    # Calculate call complexity for each function
    call_complexity = {}
    for func_name, func_info in functions.items():
        # Count direct calls made by this function
        direct_calls = sum(1 for call in call_relationships 
                          if call.get("caller", {}).get("name") == func_name)
        
        # Count total calls in the call chain (recursive)
        total_calls = _calculate_total_calls(func_name, call_relationships, set())
        
        call_complexity[func_name] = {
            "file_path": func_info.get("file_path", ""),
            "line": func_info.get("line", 0),
            "direct_calls": direct_calls,
            "total_calls": total_calls,
            "cyclomatic_complexity": func_info.get("complexity", 0),
            "is_high_complexity": total_calls > complexity_threshold,
        }
    
    # Identify high complexity functions
    high_complexity = [name for name, stats in call_complexity.items() 
                      if stats["is_high_complexity"]]
    
    return {
        "call_complexity": call_complexity,
        "high_complexity_functions": high_complexity,
        "summary": {
            "total_functions": len(functions),
            "high_complexity_count": len(high_complexity),
            "average_direct_calls": sum(stats["direct_calls"] for stats in call_complexity.values()) / len(functions) if functions else 0,
            "average_total_calls": sum(stats["total_calls"] for stats in call_complexity.values()) / len(functions) if functions else 0,
        }
    }


def _calculate_total_calls(func_name: str, call_relationships: list, visited: set) -> int:
    """Calculate total calls in the call chain for a function."""
    if func_name in visited:
        return 0  # Avoid infinite recursion
    
    visited.add(func_name)
    total = 0
    
    for call in call_relationships:
        if call.get("caller", {}).get("name") == func_name:
            callee_name = call.get("callee", {}).get("name")
            if callee_name:
                total += 1 + _calculate_total_calls(callee_name, call_relationships, visited.copy())
    
    return total