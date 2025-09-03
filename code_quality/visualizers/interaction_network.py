"""
Interaction Network Visualizer

Creates network visualizations of code interactions and relationships.
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any, Set
import json
from pathlib import Path
from .code_visualizer import CodeVisualizer


class InteractionNetworkVisualizer(CodeVisualizer):
    """Visualizes code interactions as interactive networks."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir)
        
    def create_function_call_network(self, call_graph: Dict[str, List[str]], 
                                   title: str = "Function Call Network") -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Create a network visualization of function calls.
        
        Args:
            call_graph: Dict mapping functions to their calls
            title: Network title
            
        Returns:
            Tuple of (figure, metadata)
        """
        # Build the graph
        G = nx.DiGraph()
        for func, calls in call_graph.items():
            if not G.has_node(func):
                G.add_node(func)
            for call in calls:
                G.add_edge(func, call)
        
        # Calculate node metrics
        pagerank = nx.pagerank(G)
        betweenness = nx.betweenness_centrality(G)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # Identify node types
        entry_points = [n for n in G.nodes() if in_degree.get(n, 0) == 0]
        exit_points = [n for n in G.nodes() if out_degree.get(n, 0) == 0]
        hub_nodes = [n for n in G.nodes() if betweenness.get(n, 0) > 0.1]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Main network visualization
        pos = self._calculate_hierarchical_layout(G, entry_points)
        
        # Node styling based on type
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node in entry_points:
                node_colors.append('lightgreen')
                node_sizes.append(1000)
            elif node in exit_points:
                node_colors.append('lightcoral')
                node_sizes.append(800)
            elif node in hub_nodes:
                node_colors.append('gold')
                node_sizes.append(1200)
            else:
                node_colors.append('lightblue')
                node_sizes.append(600)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=ax1,
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8,
                             edgecolors='black',
                             linewidths=1)
        
        # Draw labels
        labels = {n: self.format_label(n, 15) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)
        
        # Draw edges with varying styles
        edge_widths = []
        for u, v in G.edges():
            # Thicker lines for more important connections
            importance = pagerank.get(v, 0) * 10
            edge_widths.append(max(0.5, min(3, importance)))
        
        nx.draw_networkx_edges(G, pos, ax=ax1,
                             edge_color='gray',
                             width=edge_widths,
                             alpha=0.5,
                             arrows=True,
                             arrowsize=10,
                             arrowstyle='->')
        
        ax1.set_title(title, fontsize=16)
        ax1.axis('off')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='lightgreen', s=100, edgecolors='black', label='Entry Points'),
            plt.scatter([], [], c='gold', s=100, edgecolors='black', label='Hub Functions'),
            plt.scatter([], [], c='lightblue', s=100, edgecolors='black', label='Regular Functions'),
            plt.scatter([], [], c='lightcoral', s=100, edgecolors='black', label='Exit Points')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        # Function statistics
        self._plot_function_statistics(ax2, G, pagerank, betweenness)
        
        plt.tight_layout()
        
        # Prepare metadata
        metadata = {
            'total_functions': len(G.nodes()),
            'total_calls': len(G.edges()),
            'entry_points': entry_points,
            'exit_points': exit_points,
            'hub_functions': hub_nodes,
            'most_called': max(in_degree.items(), key=lambda x: x[1]) if in_degree else None,
            'most_calling': max(out_degree.items(), key=lambda x: x[1]) if out_degree else None,
            'isolated_functions': list(nx.isolates(G)),
            'strongly_connected': [list(c) for c in nx.strongly_connected_components(G) if len(c) > 1]
        }
        
        return fig, metadata
    
    def create_interactive_network(self, interactions: Dict[str, List[str]], 
                                 node_metadata: Optional[Dict[str, Dict]] = None,
                                 title: str = "Interactive Code Network") -> str:
        """
        Create an interactive network visualization using Plotly.
        
        Args:
            interactions: Dict mapping nodes to their connections
            node_metadata: Optional metadata for each node
            title: Network title
            
        Returns:
            Path to saved HTML file
        """
        # Build the graph
        G = nx.Graph()
        for node, connections in interactions.items():
            if not G.has_node(node):
                G.add_node(node)
            for conn in connections:
                G.add_edge(node, conn)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract edge coordinates
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>"
            if node_metadata and node in node_metadata:
                for key, value in node_metadata[node].items():
                    hover_text += f"{key}: {value}<br>"
            hover_text += f"Connections: {G.degree(node)}"
            node_text.append(hover_text)
            
            # Node styling
            node_color.append(G.degree(node))
            node_size.append(10 + G.degree(node) * 2)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[self.format_label(n, 20) for n in G.nodes()],
            textposition='top center',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        # Save as HTML
        html_file = self.output_dir / f"interactive_network_{title.replace(' ', '_').lower()}.html"
        fig.write_html(str(html_file))
        
        return str(html_file)
    
    def create_module_interaction_matrix(self, interactions: Dict[str, List[str]], 
                                       title: str = "Module Interaction Matrix") -> plt.Figure:
        """
        Create a matrix visualization of module interactions.
        
        Args:
            interactions: Module interactions
            title: Matrix title
            
        Returns:
            Matplotlib figure
        """
        # Get all unique modules
        modules = set(interactions.keys())
        for connections in interactions.values():
            modules.update(connections)
        modules = sorted(list(modules))
        
        # Create interaction matrix
        n = len(modules)
        matrix = np.zeros((n, n))
        module_to_idx = {m: i for i, m in enumerate(modules)}
        
        for module, connections in interactions.items():
            if module in module_to_idx:
                for conn in connections:
                    if conn in module_to_idx:
                        matrix[module_to_idx[module], module_to_idx[conn]] = 1
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        # Interaction matrix
        im = ax1.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels([self.format_label(m, 15) for m in modules], rotation=90)
        ax1.set_yticklabels([self.format_label(m, 15) for m in modules])
        
        # Add grid
        ax1.set_xticks(np.arange(n+1)-0.5, minor=True)
        ax1.set_yticks(np.arange(n+1)-0.5, minor=True)
        ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('Target Module')
        ax1.set_ylabel('Source Module')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Interaction', rotation=270, labelpad=15)
        
        # Clustering analysis
        self._plot_module_clusters(ax2, matrix, modules)
        
        plt.tight_layout()
        return fig
    
    def create_layered_architecture_view(self, layers: Dict[str, List[str]], 
                                       dependencies: Dict[str, List[str]],
                                       title: str = "Layered Architecture") -> plt.Figure:
        """
        Create a layered view of system architecture.
        
        Args:
            layers: Dict mapping layer names to modules in that layer
            dependencies: Module dependencies
            title: View title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Calculate positions
        layer_height = 2
        layer_spacing = 3
        module_width = 2
        module_spacing = 0.5
        
        positions = {}
        layer_y = {}
        
        # Position layers
        for i, (layer_name, modules) in enumerate(layers.items()):
            y = i * layer_spacing
            layer_y[layer_name] = y
            
            # Position modules within layer
            total_width = len(modules) * module_width + (len(modules) - 1) * module_spacing
            start_x = -total_width / 2
            
            for j, module in enumerate(modules):
                x = start_x + j * (module_width + module_spacing) + module_width / 2
                positions[module] = (x, y)
        
        # Draw layers
        for layer_name, y in layer_y.items():
            modules = layers[layer_name]
            if modules:
                min_x = min(positions[m][0] for m in modules) - module_width / 2 - 1
                max_x = max(positions[m][0] for m in modules) + module_width / 2 + 1
                
                # Layer background
                layer_box = FancyBboxPatch((min_x, y - layer_height / 2), 
                                         max_x - min_x, layer_height,
                                         boxstyle="round,pad=0.1",
                                         facecolor='lightgray',
                                         edgecolor='black',
                                         alpha=0.3)
                ax.add_patch(layer_box)
                
                # Layer label
                ax.text(min_x - 1, y, layer_name, fontsize=12, fontweight='bold',
                       va='center', ha='right')
        
        # Draw modules
        for module, (x, y) in positions.items():
            # Module box
            module_box = FancyBboxPatch((x - module_width / 2, y - 0.3), 
                                      module_width, 0.6,
                                      boxstyle="round,pad=0.05",
                                      facecolor='lightblue',
                                      edgecolor='darkblue',
                                      linewidth=2)
            ax.add_patch(module_box)
            
            # Module label
            ax.text(x, y, self.format_label(module, 15), 
                   ha='center', va='center', fontsize=9)
        
        # Draw dependencies
        for module, deps in dependencies.items():
            if module in positions:
                x1, y1 = positions[module]
                for dep in deps:
                    if dep in positions:
                        x2, y2 = positions[dep]
                        
                        # Draw arrow
                        ax.annotate('', xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                                  arrowprops=dict(arrowstyle='->', color='red', 
                                                alpha=0.6, linewidth=1.5))
        
        ax.set_title(title, fontsize=16)
        ax.axis('equal')
        ax.axis('off')
        
        # Set limits
        all_x = [pos[0] for pos in positions.values()]
        all_y = [pos[1] for pos in positions.values()]
        if all_x and all_y:
            ax.set_xlim(min(all_x) - 3, max(all_x) + 3)
            ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        
        return fig
    
    def _calculate_hierarchical_layout(self, G: nx.DiGraph, roots: List[str]) -> Dict:
        """Calculate hierarchical layout for directed graph."""
        if not roots:
            # Find potential roots
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        if not roots:
            # Use spring layout if no clear hierarchy
            return nx.spring_layout(G)
        
        # Calculate levels using BFS
        levels = {}
        from collections import deque
        
        queue = deque([(root, 0) for root in roots])
        while queue:
            node, level = queue.popleft()
            if node not in levels:
                levels[node] = level
                for successor in G.successors(node):
                    queue.append((successor, level + 1))
        
        # Position nodes
        pos = {}
        level_counts = {}
        for node, level in levels.items():
            if level not in level_counts:
                level_counts[level] = 0
            level_counts[level] += 1
        
        level_positions = {level: 0 for level in level_counts}
        
        for node, level in sorted(levels.items(), key=lambda x: (x[1], x[0])):
            x = level_positions[level] - level_counts[level] / 2
            y = -level * 2
            pos[node] = (x, y)
            level_positions[level] += 1
        
        # Add any remaining nodes
        for node in G.nodes():
            if node not in pos:
                pos[node] = (0, 0)
        
        return pos
    
    def _plot_function_statistics(self, ax: plt.Axes, G: nx.DiGraph, 
                                pagerank: Dict, betweenness: Dict):
        """Plot function call statistics."""
        # Top functions by different metrics
        top_n = 10
        
        # Sort by PageRank
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create bar plot
        if top_pagerank:
            functions = [self.format_label(f[0], 20) for f in top_pagerank]
            scores = [f[1] for f in top_pagerank]
            
            y_pos = range(len(functions))
            ax.barh(y_pos, scores, color='steelblue', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(functions)
            ax.set_xlabel('PageRank Score')
            ax.set_title('Most Important Functions', fontsize=14)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
    
    def _plot_module_clusters(self, ax: plt.Axes, matrix: np.ndarray, modules: List[str]):
        """Plot module clustering analysis."""
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        try:
            # Perform hierarchical clustering
            if matrix.shape[0] > 1:
                linkage_matrix = linkage(matrix, method='ward')
                dendrogram(linkage_matrix, labels=[self.format_label(m, 15) for m in modules], 
                          ax=ax, orientation='right')
                ax.set_title('Module Clustering', fontsize=14)
                ax.set_xlabel('Distance')
            else:
                ax.text(0.5, 0.5, 'Not enough data for clustering', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
        except:
            ax.text(0.5, 0.5, 'Clustering analysis not available', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')