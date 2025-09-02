"""
Dependency Graph Tools

Thin wrappers around the code_quality ImportAnalyzer to build and visualize
import dependency graphs for a given directory or list of files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from code_quality.core.config import get_default_config, load_config, CodeQualityConfig
from code_quality.analyzers.import_analyzer import ImportAnalyzer


def _load_cq_config(config_path: Optional[str]) -> CodeQualityConfig:
    if config_path:
        return load_config(config_path)
    return get_default_config()


def build_dependency_graph(
    path: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a directory and return an import dependency graph report.

    Returns a dict with summary, issues and a serializable graph structure.
    """
    directory = str(Path(path))
    config = _load_cq_config(config_path)

    analyzer = ImportAnalyzer(config)
    report = analyzer.analyze_directory(directory)
    return report


def visualize_dependency_graph(
    path: str,
    output_path: str,
    config_path: Optional[str] = None,
) -> None:
    """
    Build and save a PNG visualization of the dependency graph for a directory.
    """
    directory = str(Path(path))
    config = _load_cq_config(config_path)
    analyzer = ImportAnalyzer(config)
    analyzer.analyze_directory(directory)
    analyzer.visualize_import_graph(output_path)


def export_dependency_graph_json(
    path: str,
    output_file: str,
    config_path: Optional[str] = None,
) -> str:
    """
    Build the dependency graph and export to a JSON file. Returns the path.
    """
    directory = str(Path(path))
    config = _load_cq_config(config_path)
    analyzer = ImportAnalyzer(config)
    report = analyzer.analyze_directory(directory)

    # Ensure graph is serializable
    graph_payload = {
        "nodes": list(report.get("import_graph", {}).get("nodes", [])),
        "edges": list(report.get("import_graph", {}).get("edges", [])),
        "has_cycles": bool(report.get("import_graph", {}).get("has_cycles", False)),
        "issues": report.get("issues", {}),
        "summary": report.get("summary", {}),
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph_payload, f, indent=2)
    return str(out_path)


def to_networkx_graph(report: Dict[str, Any]) -> nx.DiGraph:
    """
    Convert a dependency graph report to a NetworkX DiGraph.
    """
    g = nx.DiGraph()
    import_graph = report.get("import_graph", {})
    for node in import_graph.get("nodes", []):
        g.add_node(node)
    for edge in import_graph.get("edges", []):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            g.add_edge(edge[0], edge[1])
    return g

