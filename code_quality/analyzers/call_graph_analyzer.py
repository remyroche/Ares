"""
Call Graph Analyzer - Maps function calls, imports, and dependencies between Python files.
"""

import os
import ast
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..core.config import CodeQualityConfig, get_default_config
from ..utils.file_utils import find_python_files, get_file_dependencies


class CallNode:
    """Represents a callable entity (function, class, method) in the call graph."""
    
    def __init__(self, name: str, file_path: str, node_type: str, line: int, 
                 module_path: str = "", is_imported: bool = False):
        self.name = name
        self.file_path = file_path
        self.node_type = node_type  # 'function', 'class', 'method', 'module'
        self.line = line
        self.module_path = module_path
        self.is_imported = is_imported
        self.calls: List[str] = []  # List of function names this node calls
        self.called_by: List[str] = []  # List of function names that call this node
    
    def __repr__(self):
        return f"CallNode({self.name}@{self.file_path}:{self.line})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "node_type": self.node_type,
            "line": line,
            "module_path": self.module_path,
            "is_imported": self.is_imported,
            "calls": self.calls,
            "called_by": self.called_by
        }


class CallGraphAnalyzer:
    """
    Analyzes Python code to build a comprehensive call graph showing function dependencies.
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.nodes: Dict[str, CallNode] = {}
        self.call_graph = nx.DiGraph()
        self.import_graph = nx.DiGraph()
        self.file_dependencies: Dict[str, Dict[str, List[str]]] = {}
        
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """
        Analyze all Python files in a directory to build the call graph.
        
        Args:
            directory: Directory containing Python files to analyze
            
        Returns:
            Dictionary containing call graph analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing call graph for {len(python_files)} Python files...")
        
        # Clear previous results
        self.nodes.clear()
        self.call_graph.clear()
        self.import_graph.clear()
        self.file_dependencies.clear()
        
        # First pass: collect all callable definitions
        for file_path in python_files:
            self._collect_definitions(file_path)
        
        # Second pass: analyze function calls and imports
        for file_path in python_files:
            self._analyze_calls_and_imports(file_path)
        
        # Build the call graph
        self._build_call_graph()
        
        # Analyze the graph
        analysis_results = self._analyze_graph()
        
        return analysis_results
    
    def _collect_definitions(self, file_path: str) -> None:
        """Collect all function, class, and method definitions from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            module_name = Path(file_path).stem
            
            # Add module node
            module_node = CallNode(
                name=module_name,
                file_path=file_path,
                node_type="module",
                line=1,
                module_path=module_name
            )
            self.nodes[f"{file_path}::{module_name}"] = module_node
            
            # Collect function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    node_id = f"{file_path}::{node.name}"
                    call_node = CallNode(
                        name=node.name,
                        file_path=file_path,
                        node_type="function",
                        line=node.lineno,
                        module_path=module_name
                    )
                    self.nodes[node_id] = call_node
                
                elif isinstance(node, ast.ClassDef):
                    class_id = f"{file_path}::{node.name}"
                    class_node = CallNode(
                        name=node.name,
                        file_path=file_path,
                        node_type="class",
                        line=node.lineno,
                        module_path=module_name
                    )
                    self.nodes[class_id] = class_node
                    
                    # Collect method definitions
                    for child in ast.walk(node):
                        if isinstance(child, ast.FunctionDef):
                            method_id = f"{file_path}::{node.name}.{child.name}"
                            method_node = CallNode(
                                name=f"{node.name}.{child.name}",
                                file_path=file_path,
                                node_type="method",
                                line=child.lineno,
                                module_path=module_name
                            )
                            self.nodes[method_id] = method_node
                            
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
    
    def _analyze_calls_and_imports(self, file_path: str) -> None:
        """Analyze function calls and imports in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            module_name = Path(file_path).stem
            
            # Get file dependencies
            self.file_dependencies[file_path] = get_file_dependencies(file_path)
            
            # Analyze function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self._analyze_function_call(node, file_path, module_name)
                
                elif isinstance(node, ast.Import):
                    self._analyze_import(node, file_path, module_name)
                
                elif isinstance(node, ast.ImportFrom):
                    self._analyze_import_from(node, file_path, module_name)
                    
        except Exception as e:
            print(f"Warning: Could not analyze calls in {file_path}: {e}")
    
    def _analyze_function_call(self, node: ast.Call, file_path: str, module_name: str) -> None:
        """Analyze a function call and add it to the call graph."""
        # Find the calling function context
        calling_function = self._find_calling_function(node, file_path)
        if not calling_function:
            return
        
        # Analyze the call target
        if isinstance(node.func, ast.Name):
            # Direct function call: func_name()
            called_function = node.func.id
            self._add_call_relationship(calling_function, called_function, file_path, module_name)
            
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method() or module.function()
            if isinstance(node.func.value, ast.Name):
                # obj.method() or module.function()
                obj_name = node.func.value.id
                method_name = node.func.attr
                
                # Check if it's a method call on self/this
                if obj_name in ['self', 'cls']:
                    # Method call within the same class
                    self._add_call_relationship(calling_function, f"{calling_function}.{method_name}", file_path, module_name)
                else:
                    # External method call
                    self._add_call_relationship(calling_function, f"{obj_name}.{method_name}", file_path, module_name)
    
    def _find_calling_function(self, node: ast.Call, file_path: str) -> Optional[str]:
        """Find the function that contains this call."""
        current = node
        while current.parent:
            if isinstance(current.parent, ast.FunctionDef):
                return f"{file_path}::{current.parent.name}"
            elif isinstance(current.parent, ast.ClassDef):
                # Find the method containing this call
                for child in ast.walk(current.parent):
                    if isinstance(child, ast.FunctionDef) and self._contains_node(child, node):
                        return f"{file_path}::{current.parent.name}.{child.name}"
                return None
            current = current.parent
        return None
    
    def _contains_node(self, container: ast.AST, target: ast.AST) -> bool:
        """Check if a container AST node contains a target node."""
        for child in ast.walk(container):
            if child is target:
                return True
        return False
    
    def _add_call_relationship(self, caller: str, callee: str, file_path: str, module_name: str) -> None:
        """Add a call relationship between two functions."""
        # Find the caller node
        caller_node = None
        for node_id, node in self.nodes.items():
            if node_id == caller:
                caller_node = node
                break
        
        if caller_node:
            caller_node.calls.append(callee)
            
            # Find or create the callee node
            callee_node = None
            for node_id, node in self.nodes.items():
                if node.name == callee or node_id.endswith(f"::{callee}"):
                    callee_node = node
                    break
            
            if callee_node:
                callee_node.called_by.append(caller)
            else:
                # Create a placeholder node for external calls
                external_node = CallNode(
                    name=callee,
                    file_path="external",
                    node_type="external",
                    line=0,
                    module_path="external",
                    is_imported=True
                )
                external_node.called_by.append(caller)
                self.nodes[f"external::{callee}"] = external_node
    
    def _analyze_import(self, node: ast.Import, file_path: str, module_name: str) -> None:
        """Analyze import statements."""
        for alias in node.names:
            imported_name = alias.asname or alias.name
            self._add_import_relationship(file_path, imported_name, alias.name)
    
    def _analyze_import_from(self, node: ast.ImportFrom, file_path: str, module_name: str) -> None:
        """Analyze from ... import statements."""
        module = node.module or ""
        for alias in node.names:
            imported_name = alias.asname or alias.name
            full_name = f"{module}.{alias.name}" if module else alias.name
            self._add_import_relationship(file_path, imported_name, full_name)
    
    def _add_import_relationship(self, file_path: str, imported_name: str, full_name: str) -> None:
        """Add an import relationship."""
        # Add to import graph
        self.import_graph.add_edge(file_path, full_name, name=imported_name)
    
    def _build_call_graph(self) -> None:
        """Build the NetworkX call graph from collected data."""
        for node_id, node in self.nodes.items():
            self.call_graph.add_node(node_id, **node.to_dict())
            
            for callee in node.calls:
                # Find the callee node
                callee_id = None
                for nid, n in self.nodes.items():
                    if n.name == callee or nid.endswith(f"::{callee}"):
                        callee_id = nid
                        break
                
                if callee_id:
                    self.call_graph.add_edge(node_id, callee_id, call_type="function_call")
    
    def _analyze_graph(self) -> Dict[str, Any]:
        """Analyze the built call graph for insights."""
        analysis = {
            "total_nodes": len(self.nodes),
            "total_edges": self.call_graph.number_of_edges(),
            "node_types": defaultdict(int),
            "file_stats": defaultdict(lambda: {"functions": 0, "classes": 0, "methods": 0}),
            "call_relationships": [],
            "import_relationships": [],
            "potential_dead_code": [],
            "circular_dependencies": [],
            "dependency_chains": [],
            "graph_metrics": {}
        }
        
        # Count node types
        for node in self.nodes.values():
            analysis["node_types"][node.node_type] += 1
            
            # Count by file
            if node.node_type == "function":
                analysis["file_stats"][node.file_path]["functions"] += 1
            elif node.node_type == "class":
                analysis["file_stats"][node.file_path]["classes"] += 1
            elif node.node_type == "method":
                analysis["file_stats"][node.file_path]["methods"] += 1
        
        # Find potential dead code (unused functions)
        for node_id, node in self.nodes.items():
            if node.node_type in ["function", "method"] and not node.called_by:
                # Check if it's a main function or special method
                if not (node.name == "main" or node.name.startswith("__")):
                    analysis["potential_dead_code"].append({
                        "name": node.name,
                        "file_path": node.file_path,
                        "line": node.line,
                        "node_type": node.node_type
                    })
        
        # Find circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.call_graph))
            analysis["circular_dependencies"] = cycles
        except nx.NetworkXNoCycle:
            analysis["circular_dependencies"] = []
        
        # Calculate graph metrics
        if self.call_graph.number_of_nodes() > 0:
            analysis["graph_metrics"] = {
                "density": nx.density(self.call_graph),
                "average_clustering": nx.average_clustering(self.call_graph.to_undirected()),
                "connected_components": nx.number_strongly_connected_components(self.call_graph),
                "is_dag": nx.is_directed_acyclic_graph(self.call_graph)
            }
        
        # Build call relationships list
        for edge in self.call_graph.edges(data=True):
            source = self.nodes[edge[0]]
            target = self.nodes[edge[1]]
            analysis["call_relationships"].append({
                "caller": {
                    "name": source.name,
                    "file_path": source.file_path,
                    "line": source.line,
                    "node_type": source.node_type
                },
                "callee": {
                    "name": target.name,
                    "file_path": target.file_path,
                    "line": target.line,
                    "node_type": target.node_type
                },
                "call_type": edge[2].get("call_type", "unknown")
            })
        
        # Build import relationships list
        for edge in self.import_graph.edges(data=True):
            analysis["import_relationships"].append({
                "file_path": edge[0],
                "imported": edge[1],
                "alias": edge[2].get("name", "")
            })
        
        return analysis
    
    def find_dead_code(self) -> List[Dict[str, Any]]:
        """Find potentially dead code (unused functions/methods)."""
        dead_code = []
        
        for node_id, node in self.nodes.items():
            if node.node_type in ["function", "method"]:
                # Skip special methods and main functions
                if node.name.startswith("__") or node.name == "main":
                    continue
                
                # Check if function is called
                if not node.called_by:
                    # Check if it's imported elsewhere
                    is_imported = False
                    for import_rel in self.import_graph.edges(data=True):
                        if import_rel[1] == node.name:
                            is_imported = True
                            break
                    
                    if not is_imported:
                        dead_code.append({
                            "name": node.name,
                            "file_path": node.file_path,
                            "line": node.line,
                            "node_type": node.node_type,
                            "module_path": node.module_path
                        })
        
        return dead_code
    
    def find_unused_imports(self) -> List[Dict[str, Any]]:
        """Find unused imports."""
        unused_imports = []
        
        for edge in self.import_graph.edges(data=True):
            imported_name = edge[2].get("name", "")
            if imported_name:
                # Check if this import is used in function calls
                is_used = False
                for node in self.nodes.values():
                    if imported_name in node.calls or imported_name == node.name:
                        is_used = True
                        break
                
                if not is_used:
                    unused_imports.append({
                        "file_path": edge[0],
                        "imported": edge[1],
                        "alias": imported_name
                    })
        
        return unused_imports
    
    def get_function_dependencies(self, function_name: str) -> Dict[str, Any]:
        """Get all dependencies for a specific function."""
        dependencies = {
            "function": function_name,
            "calls": [],
            "called_by": [],
            "imports": [],
            "file_path": None
        }
        
        # Find the function node
        for node_id, node in self.nodes.items():
            if node.name == function_name:
                dependencies["file_path"] = node.file_path
                dependencies["calls"] = node.calls
                dependencies["called_by"] = node.called_by
                break
        
        # Find imports for this function's file
        for edge in self.import_graph.edges():
            if edge[0] == dependencies["file_path"]:
                dependencies["imports"].append(edge[1])
        
        return dependencies
    
    def export_graph(self, output_path: str, format: str = "json") -> None:
        """Export the call graph to various formats."""
        if format == "json":
            # Export as JSON
            graph_data = {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [{"source": edge[0], "target": edge[1], **edge[2]} 
                         for edge in self.call_graph.edges(data=True)],
                "imports": [{"source": edge[0], "target": edge[1], **edge[2]} 
                           for edge in self.import_graph.edges(data=True)]
            }
            
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
        elif format == "dot":
            # Export as DOT format for Graphviz
            nx.drawing.nx_pydot.write_dot(self.call_graph, output_path)
            
        elif format == "gexf":
            # Export as GEXF format for Gephi
            nx.write_gexf(self.call_graph, output_path)
    
    def visualize_graph(self, output_path: str, max_nodes: int = 100) -> None:
        """Create a visual representation of the call graph."""
        if len(self.nodes) > max_nodes:
            print(f"Warning: Graph has {len(self.nodes)} nodes, limiting visualization to {max_nodes}")
            # Create a subgraph with most connected nodes
            node_degrees = dict(self.call_graph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = self.call_graph.subgraph([node[0] for node in top_nodes])
        else:
            subgraph = self.call_graph
        
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes by type
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node]["node_type"]
            if node_type == "function":
                node_colors.append("lightblue")
            elif node_type == "class":
                node_colors.append("lightgreen")
            elif node_type == "method":
                node_colors.append("lightcoral")
            else:
                node_colors.append("lightgray")
        
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels for important nodes
        labels = {}
        for node in subgraph.nodes():
            if subgraph.degree(node) > 2:  # Only label nodes with more connections
                labels[node] = subgraph.nodes[node]["name"]
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Functions'),
            mpatches.Patch(color='lightgreen', label='Classes'),
            mpatches.Patch(color='lightcoral', label='Methods'),
            mpatches.Patch(color='lightgray', label='Other')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.title("Python Call Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {output_path}")


def main():
    """Command-line interface for the call graph analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Python call graph and dependencies")
    parser.add_argument("--path", required=True, help="Path to directory containing Python files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--format", choices=["json", "dot", "gexf"], default="json", 
                       help="Output format for graph export")
    parser.add_argument("--visualize", action="store_true", help="Generate graph visualization")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        from ..core.config import load_config
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Run call graph analysis
    analyzer = CallGraphAnalyzer(config)
    results = analyzer.analyze_directory(args.path)
    
    # Print summary
    print("\n" + "="*50)
    print("CALL GRAPH ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total nodes: {results['total_nodes']}")
    print(f"Total edges: {results['total_edges']}")
    print(f"Node types: {dict(results['node_types'])}")
    
    print(f"\nPotential dead code: {len(results['potential_dead_code'])} items")
    if results['potential_dead_code']:
        print("Top dead code candidates:")
        for item in results['potential_dead_code'][:5]:
            print(f"  - {item['name']} ({item['node_type']}) in {item['file_path']}:{item['line']}")
    
    print(f"\nCircular dependencies: {len(results['circular_dependencies'])}")
    if results['circular_dependencies']:
        print("Circular dependency chains:")
        for cycle in results['circular_dependencies'][:3]:
            print(f"  - {' -> '.join(cycle)}")
    
    # Export results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Export graph
        graph_file = output_dir / f"call_graph.{args.format}"
        analyzer.export_graph(str(graph_file), args.format)
        print(f"\nCall graph exported to {graph_file}")
        
        # Export analysis results
        results_file = output_dir / "call_graph_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Analysis results exported to {results_file}")
        
        # Generate visualization
        if args.visualize:
            viz_file = output_dir / "call_graph_visualization.png"
            analyzer.visualize_graph(str(viz_file))
            print(f"Graph visualization saved to {viz_file}")


if __name__ == "__main__":
    main()