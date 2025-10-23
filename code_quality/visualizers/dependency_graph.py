from src.utils.tprint import tprint

"""
Dependency Graph Visualizer

Creates visual representations of module and package dependencies.
Enhanced with import verification data for more accurate dependency analysis.
"""

from typing import Optional, Dict, List, Any, Tuple
from .code_visualizer import CodeVisualizer
import numpy as np

try:
    import networkx as nx
    import matplotlib.pyplot as plt

    NETWORKX_AVAILABLE = True
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    nx = None
    plt = None
    tprint("Warning: NetworkX/matplotlib not available - dependency graph visualization will be limited")


class DependencyGraphVisualizer(CodeVisualizer):
    """Visualizes module dependencies as directed graphs with import verification enhancement."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir)
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.import_verification_data = None
        
    def create_dependency_graph(self, dependencies: Dict[str, List[str]], 
                              title: str = "Code Dependencies") -> Tuple[Any, Dict[str, Any]]:
        if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return None, {"error": "NetworkX/matplotlib not available"}
        """
        Create a dependency graph visualization.
        
        Args:
            dependencies: Dict mapping modules to their dependencies
            title: Graph title
            
        Returns:
            Tuple of (figure, metadata)
        """
        # Build the graph
        self.graph.clear()
        for module, deps in dependencies.items():
            if not self.graph.has_node(module):
                self.graph.add_node(module)
            for dep in deps:
                self.graph.add_edge(module, dep)
        
        # Calculate node properties
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        betweenness = nx.betweenness_centrality(self.graph)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Main dependency graph
        pos = self._calculate_layout(self.graph)
        
        # Node sizes based on importance
        node_sizes = [300 + 1000 * betweenness.get(node, 0) for node in self.graph.nodes()]
        
        # Node colors based on in-degree
        node_colors = self.create_color_map([in_degrees.get(node, 0) for node in self.graph.nodes()])
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, ax=ax1, 
                             node_size=node_sizes,
                             node_color=node_colors,
                             alpha=0.8)
        
        nx.draw_networkx_labels(self.graph, pos, ax=ax1,
                              labels={n: self.format_label(n, 15) for n in self.graph.nodes()},
                              font_size=8)
        
        nx.draw_networkx_edges(self.graph, pos, ax=ax1,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=10,
                             alpha=0.5,
                             arrowstyle='->')
        
        ax1.set_title(f"{title} - Dependency Network", fontsize=16)
        ax1.axis('off')
        
        # Dependency statistics
        self._plot_dependency_stats(ax2, in_degrees, out_degrees)
        
        plt.tight_layout()
        
        # Prepare metadata
        metadata = {
            'total_modules': len(self.graph.nodes()),
            'total_dependencies': len(self.graph.edges()),
            'most_dependent': max(in_degrees.items(), key=lambda x: x[1]) if in_degrees else ('', 0),
            'most_dependencies': max(out_degrees.items(), key=lambda x: x[1]) if out_degrees else ('', 0),
            'circular_dependencies': list(nx.simple_cycles(self.graph)),
            'isolated_modules': list(nx.isolates(self.graph)),
            'strongly_connected_components': [list(comp) for comp in nx.strongly_connected_components(self.graph) if len(comp) > 1]
        }
        
        return fig, metadata
    
    def create_enhanced_dependency_graph_with_imports(self, dependencies: Dict[str, List[str]], 
                                                    import_verification_data: Dict[str, Any],
                                                    title: str = "Enhanced Code Dependencies") -> Tuple[Any, Dict[str, Any]]:
        """
        Create an enhanced dependency graph using import verification data.
        
        Args:
            dependencies: Dict mapping modules to their dependencies
            import_verification_data: Results from ImportVerifierAnalyzer
            title: Graph title
            
        Returns:
            Tuple of (figure, metadata)
        """
        if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return None, {"error": "NetworkX/matplotlib not available"}
        
        # Store import verification data for use in other methods
        self.import_verification_data = import_verification_data
        
        # Build the enhanced graph with import verification data
        self.graph.clear()
        import_status = import_verification_data.get("import_status", {})
        
        # Add nodes with import verification metadata
        for module, deps in dependencies.items():
            if not self.graph.has_node(module):
                # Get import verification data for this module
                module_import_data = import_status.get(module, {})
                self.graph.add_node(module, 
                                  is_imported=module_import_data.get("is_imported", False),
                                  import_count=module_import_data.get("import_count", 0),
                                  only_non_production=module_import_data.get("only_imported_by_non_production", False),
                                  imported_by=module_import_data.get("imported_by", []))
            
            for dep in deps:
                if not self.graph.has_node(dep):
                    dep_import_data = import_status.get(dep, {})
                    self.graph.add_node(dep,
                                      is_imported=dep_import_data.get("is_imported", False),
                                      import_count=dep_import_data.get("import_count", 0),
                                      only_non_production=dep_import_data.get("only_imported_by_non_production", False),
                                      imported_by=dep_import_data.get("imported_by", []))
                self.graph.add_edge(module, dep)
        
        # Calculate enhanced node properties
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        betweenness = nx.betweenness_centrality(self.graph)
        
        # Create enhanced figure with import verification insights
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # 1. Main enhanced dependency graph
        self._plot_enhanced_dependency_network(ax1, import_status)
        
        # 2. Import status distribution
        self._plot_import_status_distribution(ax2, import_status)
        
        # 3. Critical dependencies (high import count)
        self._plot_critical_dependencies(ax3, import_status)
        
        # 4. Circular dependencies analysis
        self._plot_circular_dependencies_analysis(ax4, import_verification_data)
        
        plt.tight_layout()
        
        # Prepare enhanced metadata
        metadata = self._generate_enhanced_metadata(import_verification_data, dependencies)
        
        return fig, metadata
    
    def create_circular_dependency_visualization(self, cycles: List[List[str]], 
                                               title: str = "Circular Dependencies") -> Any:
        """
        Visualize circular dependencies.
        
        Args:
            cycles: List of circular dependency cycles
            title: Graph title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if not cycles:
            ax.text(0.5, 0.5, 'No circular dependencies found!', 
                   ha='center', va='center', fontsize=20, color='green')
            ax.axis('off')
            return fig
        
        # Create a graph of just the cycles
        cycle_graph = nx.DiGraph()
        for cycle in cycles:
            for i in range(len(cycle)):
                cycle_graph.add_edge(cycle[i], cycle[(i + 1) % len(cycle)])
        
        # Use circular layout for better visualization
        pos = nx.circular_layout(cycle_graph)
        
        # Draw the cycle graph
        nx.draw_networkx_nodes(cycle_graph, pos, ax=ax,
                             node_color='red',
                             node_size=1000,
                             alpha=0.7)
        
        nx.draw_networkx_labels(cycle_graph, pos, ax=ax,
                              labels={n: self.format_label(n, 20) for n in cycle_graph.nodes()},
                              font_size=10)
        
        nx.draw_networkx_edges(cycle_graph, pos, ax=ax,
                             edge_color='darkred',
                             arrows=True,
                             arrowsize=20,
                             width=2,
                             arrowstyle='->')
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        # Add cycle information
        cycle_text = f"Found {len(cycles)} circular dependencies:\n"
        for i, cycle in enumerate(cycles[:5]):  # Show first 5
            cycle_text += f"{i+1}. {' → '.join(cycle)} → {cycle[0]}\n"
        if len(cycles) > 5:
            cycle_text += f"... and {len(cycles) - 5} more"
        
        ax.text(0.02, 0.98, cycle_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig
    
    def create_module_hierarchy(self, dependencies: Dict[str, List[str]], 
                              title: str = "Module Hierarchy") -> Any:
        """
        Create a hierarchical view of modules.
        
        Args:
            dependencies: Module dependencies
            title: Graph title
            
        Returns:
            Matplotlib figure
        """
        # Build the graph
        self.graph.clear()
        for module, deps in dependencies.items():
            if not self.graph.has_node(module):
                self.graph.add_node(module)
            for dep in deps:
                self.graph.add_edge(module, dep)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Try to create a hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        except:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Calculate levels based on dependency depth
        levels = self._calculate_dependency_levels()
        level_colors = self.create_color_map(list(range(max(levels.values()) + 1)) if levels else [0])
        
        node_colors = [level_colors[levels.get(node, 0)] for node in self.graph.nodes()]
        
        # Draw the hierarchy
        nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                             node_color=node_colors,
                             node_size=800,
                             alpha=0.8)
        
        nx.draw_networkx_labels(self.graph, pos, ax=ax,
                              labels={n: self.format_label(n, 20) for n in self.graph.nodes()},
                              font_size=9)
        
        nx.draw_networkx_edges(self.graph, pos, ax=ax,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=10,
                             alpha=0.3)
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        # Add legend for levels
        if levels:
            from matplotlib.patches import Rectangle
            legend_elements = []
            for level in range(max(levels.values()) + 1):
                color = level_colors[level]
                legend_elements.append(Rectangle((0, 0), 1, 1, fc=color, 
                                               edgecolor='black', linewidth=0.5))
            
            ax.legend(legend_elements, [f'Level {i}' for i in range(len(legend_elements))],
                     loc='upper right', title='Dependency Depth')
        
        return fig
    
    def _calculate_layout(self, graph) -> Dict:
        """Calculate optimal layout for the graph."""
        # Try different layouts and choose the best
        layouts = {
            'spring': lambda: nx.spring_layout(graph, k=2, iterations=50),
            'kamada_kawai': lambda: nx.kamada_kawai_layout(graph),
            'spectral': lambda: nx.spectral_layout(graph)
        }
        
        # Use spring layout as default
        try:
            return layouts['spring']()
        except:
            return nx.random_layout(graph)
    
    def _calculate_dependency_levels(self) -> Dict[str, int]:
        """Calculate dependency levels for hierarchical visualization."""
        levels = {}
        
        # Find nodes with no dependencies (roots)
        roots = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        # BFS to assign levels
        from collections import deque
        queue = deque([(root, 0) for root in roots])
        
        while queue:
            node, level = queue.popleft()
            if node in levels:
                continue
            levels[node] = level
            
            # Add nodes that depend on this one
            for predecessor in self.graph.predecessors(node):
                queue.append((predecessor, level + 1))
        
        # Assign level to any remaining nodes
        for node in self.graph.nodes():
            if node not in levels:
                levels[node] = 0
        
        return levels
    
    def _plot_dependency_stats(self, ax, in_degrees: Dict[str, int], 
                              out_degrees: Dict[str, int]):
        """Plot dependency statistics."""
        # Top 10 most dependent modules
        top_dependent = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        top_dependencies = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create bar plots
        ax.set_title("Dependency Statistics", fontsize=14)
        
        if top_dependent:
            modules = [self.format_label(m[0], 20) for m in top_dependent]
            counts = [m[1] for m in top_dependent]
            
            y_pos = range(len(modules))
            ax.barh(y_pos, counts, color='steelblue', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(modules)
            ax.set_xlabel('Number of modules depending on this')
            ax.invert_yaxis()
        
        ax.grid(True, alpha=0.3)
    
    def _plot_enhanced_dependency_network(self, ax, import_status: Dict[str, Any]) -> None:
        """Plot the main enhanced dependency network with import verification data."""
        if not self.graph.nodes():
            ax.text(0.5, 0.5, 'No dependency data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Calculate layout
        pos = self._calculate_layout(self.graph)
        
        # Node styling based on import verification data
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            is_imported = node_data.get("is_imported", False)
            import_count = node_data.get("import_count", 0)
            only_non_prod = node_data.get("only_non_production", False)
            
            # Color coding based on import status
            if only_non_prod:
                node_colors.append('orange')  # Only imported by non-production
            elif is_imported:
                node_colors.append('green')   # Imported by production code
            else:
                node_colors.append('red')     # Not imported
            
            # Size based on import count
            node_sizes.append(max(100, min(1000, 100 + import_count * 50)))
        
        # Draw network
        nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8,
                             edgecolors='black',
                             linewidths=1)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, ax=ax,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=10,
                             alpha=0.5,
                             arrowstyle='->')
        
        # Draw labels
        labels = {n: self.format_label(n, 15) for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, ax=ax, font_size=8)
        
        ax.set_title("Enhanced Dependency Network", fontsize=14)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='green', s=100, edgecolors='black', label='Imported by production'),
            plt.scatter([], [], c='orange', s=100, edgecolors='black', label='Only non-production'),
            plt.scatter([], [], c='red', s=100, edgecolors='black', label='Not imported')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_import_status_distribution(self, ax, import_status: Dict[str, Any]) -> None:
        """Plot import status distribution."""
        if not import_status:
            ax.text(0.5, 0.5, 'No import data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Count import statuses
        imported_count = sum(1 for status in import_status.values() if status.get("is_imported", False))
        unimported_count = len(import_status) - imported_count
        only_non_prod_count = sum(1 for status in import_status.values() if status.get("only_imported_by_non_production", False))
        prod_imported_count = imported_count - only_non_prod_count
        
        # Create pie chart
        labels = ['Imported by production', 'Only non-production', 'Not imported']
        sizes = [prod_imported_count, only_non_prod_count, unimported_count]
        colors = ['green', 'orange', 'red']
        
        # Remove zero values
        non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
        if non_zero_data:
            labels, sizes, colors = zip(*non_zero_data)
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, alpha=0.8)
            ax.set_title('Import Status Distribution', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No import data', ha='center', va='center')
            ax.axis('off')
    
    def _plot_critical_dependencies(self, ax, import_status: Dict[str, Any]) -> None:
        """Plot critical dependencies (high import count)."""
        if not import_status:
            ax.text(0.5, 0.5, 'No import data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Get top critical dependencies
        critical_deps = []
        for file_path, status in import_status.items():
            import_count = status.get("import_count", 0)
            if import_count > 2:  # Threshold for critical dependency
                critical_deps.append((file_path, import_count))
        
        # Sort by import count
        critical_deps.sort(key=lambda x: x[1], reverse=True)
        top_critical = critical_deps[:10]  # Top 10
        
        if top_critical:
            files, counts = zip(*top_critical)
            file_labels = [self.format_label(Path(f).name, 20) for f in files]
            
            y_pos = range(len(file_labels))
            ax.barh(y_pos, counts, color='coral', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(file_labels)
            ax.set_xlabel('Number of Importers')
            ax.set_title('Critical Dependencies (High Import Count)', fontsize=14)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No critical dependencies found', ha='center', va='center')
            ax.axis('off')
    
    def _plot_circular_dependencies_analysis(self, ax, import_verification_data: Dict[str, Any]) -> None:
        """Plot circular dependencies analysis."""
        advanced_analysis = import_verification_data.get("advanced_analysis", {})
        circular_imports = advanced_analysis.get("circular_imports", [])
        
        if not circular_imports:
            ax.text(0.5, 0.5, '✅ No Circular Dependencies Found!', 
                    ha='center', va='center', fontsize=16, color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax.axis('off')
        else:
            # Show circular dependencies count and details
            cycle_text = f"Found {len(circular_imports)} circular dependencies:\n\n"
            for i, cycle in enumerate(circular_imports[:5], 1):  # Show first 5
                cycle_names = [Path(f).name for f in cycle]
                cycle_text += f"{i}. {' → '.join(cycle_names)} → {cycle_names[0]}\n"
            if len(circular_imports) > 5:
                cycle_text += f"\n... and {len(circular_imports) - 5} more"
            
            ax.text(0.05, 0.95, cycle_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title('Circular Dependencies Analysis', fontsize=14)
            ax.axis('off')
    
    def _generate_enhanced_metadata(self, import_verification_data: Dict[str, Any], 
                                  dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate enhanced metadata including import verification insights."""
        import_status = import_verification_data.get("import_status", {})
        summary = import_verification_data.get("summary", {})
        advanced_analysis = import_verification_data.get("advanced_analysis", {})
        
        # Calculate enhanced metrics
        total_modules = len(self.graph.nodes()) if self.graph else 0
        total_dependencies = len(self.graph.edges()) if self.graph else 0
        
        # Import verification metrics
        imported_modules = sum(1 for status in import_status.values() if status.get("is_imported", False))
        unimported_modules = len(import_status) - imported_modules
        circular_deps = len(advanced_analysis.get("circular_imports", []))
        
        return {
            'total_modules': total_modules,
            'total_dependencies': total_dependencies,
            'import_verification_metrics': {
                'total_files_analyzed': summary.get("total_files", 0),
                'imported_files': summary.get("imported_files", 0),
                'unimported_files': summary.get("unimported_files", 0),
                'import_percentage': summary.get("import_percentage", 0),
                'circular_dependencies': circular_deps
            },
            'enhanced_analysis': {
                'most_dependent': max(import_status.items(), key=lambda x: x[1].get("import_count", 0)) if import_status else ('', 0),
                'most_dependencies': max(dependencies.items(), key=lambda x: len(x[1])) if dependencies else ('', 0),
                'circular_dependencies': advanced_analysis.get("circular_imports", []),
                'critical_paths': advanced_analysis.get("critical_paths", {}),
                'import_depths': advanced_analysis.get("import_depths", {})
            },
            'visualization_type': 'enhanced_dependency_graph_with_imports',
            'enhancement_version': '1.0.0'
        }