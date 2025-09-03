"""
Dependency Graph Visualizer

Creates visual representations of module and package dependencies.
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple
import json
from pathlib import Path
from .code_visualizer import CodeVisualizer


class DependencyGraphVisualizer(CodeVisualizer):
    """Visualizes module dependencies as directed graphs."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir)
        self.graph = nx.DiGraph()
        
    def create_dependency_graph(self, dependencies: Dict[str, List[str]], 
                              title: str = "Code Dependencies") -> Tuple[plt.Figure, Dict[str, Any]]:
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
    
    def create_circular_dependency_visualization(self, cycles: List[List[str]], 
                                               title: str = "Circular Dependencies") -> plt.Figure:
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
                              title: str = "Module Hierarchy") -> plt.Figure:
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
    
    def _calculate_layout(self, graph: nx.DiGraph) -> Dict:
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
    
    def _plot_dependency_stats(self, ax: plt.Axes, in_degrees: Dict[str, int], 
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