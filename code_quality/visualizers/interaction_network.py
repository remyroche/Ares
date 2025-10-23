"""
Interaction Network Visualizer

Creates network visualizations of code interactions and relationships.
Enhanced with import verification data for more accurate interaction analysis.
"""

from typing import Optional, Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
from .code_visualizer import CodeVisualizer


class InteractionNetworkVisualizer(CodeVisualizer):
    """Visualizes code interactions as interactive networks with import verification enhancement."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir)
        self.import_verification_data = None
        
    def create_function_call_network(self, call_graph: Dict[str, List[str]], 
                                   title: str = "Function Call Network") -> Tuple[Any, Dict[str, Any]]:
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
    
    def create_enhanced_interaction_network_with_imports(self, interactions: Dict[str, List[str]], 
                                                       import_verification_data: Dict[str, Any],
                                                       title: str = "Enhanced Interaction Network") -> Tuple[Any, Dict[str, Any]]:
        """
        Create an enhanced interaction network using import verification data.
        
        Args:
            interactions: Dict mapping nodes to their connections
            import_verification_data: Results from ImportVerifierAnalyzer
            title: Network title
            
        Returns:
            Tuple of (figure, metadata)
        """
        # Store import verification data
        self.import_verification_data = import_verification_data
        
        # Build the enhanced graph
        G = nx.DiGraph()
        import_status = import_verification_data.get("import_status", {})
        
        # Add nodes with import verification metadata
        for node, connections in interactions.items():
            if not G.has_node(node):
                # Get import verification data for this node
                node_import_data = import_status.get(node, {})
                G.add_node(node,
                          is_imported=node_import_data.get("is_imported", False),
                          import_count=node_import_data.get("import_count", 0),
                          only_non_production=node_import_data.get("only_imported_by_non_production", False),
                          imported_by=node_import_data.get("imported_by", []))
            
            for conn in connections:
                if not G.has_node(conn):
                    conn_import_data = import_status.get(conn, {})
                    G.add_node(conn,
                              is_imported=conn_import_data.get("is_imported", False),
                              import_count=conn_import_data.get("import_count", 0),
                              only_non_production=conn_import_data.get("only_imported_by_non_production", False),
                              imported_by=conn_import_data.get("imported_by", []))
                G.add_edge(node, conn)
        
        # Calculate enhanced node metrics
        pagerank = nx.pagerank(G)
        betweenness = nx.betweenness_centrality(G)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # Identify enhanced node types using import data
        entry_points = [n for n in G.nodes() if in_degree.get(n, 0) == 0]
        exit_points = [n for n in G.nodes() if out_degree.get(n, 0) == 0]
        hub_nodes = [n for n in G.nodes() if betweenness.get(n, 0) > 0.1]
        
        # Enhanced node classification using import data
        critical_nodes = [n for n in G.nodes() if G.nodes[n].get("import_count", 0) > 5]
        unused_nodes = [n for n in G.nodes() if not G.nodes[n].get("is_imported", False)]
        non_prod_nodes = [n for n in G.nodes() if G.nodes[n].get("only_non_production", False)]
        
        # Create enhanced figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # 1. Main enhanced interaction network
        self._plot_enhanced_interaction_network(ax1, G, import_status, pagerank, betweenness)
        
        # 2. Import-based node classification
        self._plot_import_based_classification(ax2, G, import_status)
        
        # 3. Critical interaction analysis
        self._plot_critical_interactions(ax3, G, import_status, pagerank)
        
        # 4. Interaction patterns analysis
        self._plot_interaction_patterns(ax4, G, import_status, interactions)
        
        plt.tight_layout()
        
        # Prepare enhanced metadata
        metadata = self._generate_enhanced_interaction_metadata(import_verification_data, interactions, G)
        
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
                                       title: str = "Module Interaction Matrix") -> Any:
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
                                       title: str = "Layered Architecture") -> Any:
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
    
    def _plot_enhanced_interaction_network(self, ax, G: nx.DiGraph, import_status: Dict[str, Any], 
                                         pagerank: Dict, betweenness: Dict) -> None:
        """Plot the main enhanced interaction network with import verification data."""
        if not G.nodes():
            ax.text(0.5, 0.5, 'No interaction data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Calculate layout
        pos = self._calculate_hierarchical_layout(G, [])
        
        # Enhanced node styling based on import verification data
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
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
            
            # Size based on PageRank and import count
            pagerank_score = pagerank.get(node, 0)
            size = max(100, min(1000, 100 + pagerank_score * 1000 + import_count * 50))
            node_sizes.append(size)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=ax,
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8,
                             edgecolors='black',
                             linewidths=1)
        
        # Draw edges with varying styles based on importance
        edge_widths = []
        for u, v in G.edges():
            # Thicker lines for more important connections
            importance = pagerank.get(v, 0) * 10
            edge_widths.append(max(0.5, min(3, importance)))
        
        nx.draw_networkx_edges(G, pos, ax=ax,
                             edge_color='gray',
                             width=edge_widths,
                             alpha=0.5,
                             arrows=True,
                             arrowsize=10,
                             arrowstyle='->')
        
        # Draw labels
        labels = {n: self.format_label(n, 15) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        ax.set_title("Enhanced Interaction Network", fontsize=14)
        ax.axis('off')
        
        # Add enhanced legend
        legend_elements = [
            plt.scatter([], [], c='green', s=100, edgecolors='black', label='Imported by production'),
            plt.scatter([], [], c='orange', s=100, edgecolors='black', label='Only non-production'),
            plt.scatter([], [], c='red', s=100, edgecolors='black', label='Not imported')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    
    def _plot_import_based_classification(self, ax, G: nx.DiGraph, import_status: Dict[str, Any]) -> None:
        """Plot import-based node classification."""
        if not G.nodes():
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Classify nodes based on import data
        imported_nodes = [n for n in G.nodes() if G.nodes[n].get("is_imported", False)]
        unimported_nodes = [n for n in G.nodes() if not G.nodes[n].get("is_imported", False)]
        non_prod_nodes = [n for n in G.nodes() if G.nodes[n].get("only_non_production", False)]
        critical_nodes = [n for n in G.nodes() if G.nodes[n].get("import_count", 0) > 5]
        
        # Create classification chart
        categories = ['Imported', 'Unimported', 'Non-Production Only', 'Critical (>5 imports)']
        counts = [len(imported_nodes), len(unimported_nodes), len(non_prod_nodes), len(critical_nodes)]
        colors = ['green', 'red', 'orange', 'purple']
        
        # Remove zero counts
        non_zero_data = [(cat, count, color) for cat, count, color in zip(categories, counts, colors) if count > 0]
        if non_zero_data:
            categories, counts, colors = zip(*non_zero_data)
            
            wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
                                            startangle=90, alpha=0.8)
            ax.set_title('Node Classification by Import Status', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No classification data', ha='center', va='center')
            ax.axis('off')
    
    def _plot_critical_interactions(self, ax, G: nx.DiGraph, import_status: Dict[str, Any], pagerank: Dict) -> None:
        """Plot critical interactions analysis."""
        if not G.nodes():
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Find critical interactions (high PageRank + high import count)
        critical_interactions = []
        for node in G.nodes():
            node_data = G.nodes[node]
            pagerank_score = pagerank.get(node, 0)
            import_count = node_data.get("import_count", 0)
            
            # Critical if high PageRank and high import count
            if pagerank_score > 0.01 and import_count > 2:
                critical_interactions.append((node, pagerank_score, import_count))
        
        # Sort by combined score
        critical_interactions.sort(key=lambda x: x[1] * x[2], reverse=True)
        top_critical = critical_interactions[:10]  # Top 10
        
        if top_critical:
            nodes, pageranks, import_counts = zip(*top_critical)
            node_labels = [self.format_label(Path(n).name, 20) for n in nodes]
            
            # Create scatter plot
            scatter = ax.scatter(pageranks, import_counts, s=100, alpha=0.7, c=range(len(nodes)), cmap='viridis')
            
            # Add labels
            for i, (node, pr, ic) in enumerate(top_critical):
                ax.annotate(node_labels[i], (pr, ic), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('PageRank Score')
            ax.set_ylabel('Import Count')
            ax.set_title('Critical Interactions (PageRank vs Import Count)', fontsize=14)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No critical interactions found', ha='center', va='center')
            ax.axis('off')
    
    def _plot_interaction_patterns(self, ax, G: nx.DiGraph, import_status: Dict[str, Any], interactions: Dict[str, List[str]]) -> None:
        """Plot interaction patterns analysis."""
        if not G.nodes():
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Analyze interaction patterns
        pattern_stats = {
            'high_interaction_high_import': 0,  # Many interactions, many imports
            'high_interaction_low_import': 0,   # Many interactions, few imports
            'low_interaction_high_import': 0,   # Few interactions, many imports
            'low_interaction_low_import': 0     # Few interactions, few imports
        }
        
        for node in G.nodes():
            node_data = G.nodes[node]
            interaction_count = len(interactions.get(node, []))
            import_count = node_data.get("import_count", 0)
            
            # Classify pattern
            if interaction_count > 3 and import_count > 3:
                pattern_stats['high_interaction_high_import'] += 1
            elif interaction_count > 3 and import_count <= 3:
                pattern_stats['high_interaction_low_import'] += 1
            elif interaction_count <= 3 and import_count > 3:
                pattern_stats['low_interaction_high_import'] += 1
            else:
                pattern_stats['low_interaction_low_import'] += 1
        
        # Create bar chart
        patterns = list(pattern_stats.keys())
        counts = list(pattern_stats.values())
        colors = ['darkgreen', 'orange', 'blue', 'gray']
        
        bars = ax.bar(range(len(patterns)), counts, color=colors, alpha=0.8)
        ax.set_xticks(range(len(patterns)))
        ax.set_xticklabels([p.replace('_', '\n') for p in patterns], rotation=45, ha='right')
        ax.set_ylabel('Number of Nodes')
        ax.set_title('Interaction Patterns Analysis', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(count), ha='center', va='bottom')
    
    def _generate_enhanced_interaction_metadata(self, import_verification_data: Dict[str, Any], 
                                              interactions: Dict[str, List[str]], G: nx.DiGraph) -> Dict[str, Any]:
        """Generate enhanced metadata for interaction network with import verification data."""
        import_status = import_verification_data.get("import_status", {})
        summary = import_verification_data.get("summary", {})
        advanced_analysis = import_verification_data.get("advanced_analysis", {})
        
        # Calculate enhanced metrics
        total_nodes = len(G.nodes())
        total_edges = len(G.edges())
        
        # Import verification metrics
        imported_nodes = sum(1 for node in G.nodes() if G.nodes[node].get("is_imported", False))
        unimported_nodes = total_nodes - imported_nodes
        critical_nodes = sum(1 for node in G.nodes() if G.nodes[node].get("import_count", 0) > 5)
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'import_verification_metrics': {
                'total_files_analyzed': summary.get("total_files", 0),
                'imported_nodes': imported_nodes,
                'unimported_nodes': unimported_nodes,
                'critical_nodes': critical_nodes,
                'import_percentage': summary.get("import_percentage", 0)
            },
            'enhanced_analysis': {
                'interaction_patterns': {
                    'high_interaction_high_import': sum(1 for node in G.nodes() 
                                                      if len(interactions.get(node, [])) > 3 and 
                                                         G.nodes[node].get("import_count", 0) > 3),
                    'high_interaction_low_import': sum(1 for node in G.nodes() 
                                                      if len(interactions.get(node, [])) > 3 and 
                                                         G.nodes[node].get("import_count", 0) <= 3),
                    'low_interaction_high_import': sum(1 for node in G.nodes() 
                                                      if len(interactions.get(node, [])) <= 3 and 
                                                         G.nodes[node].get("import_count", 0) > 3),
                    'low_interaction_low_import': sum(1 for node in G.nodes() 
                                                     if len(interactions.get(node, [])) <= 3 and 
                                                        G.nodes[node].get("import_count", 0) <= 3)
                },
                'circular_dependencies': advanced_analysis.get("circular_imports", []),
                'critical_paths': advanced_analysis.get("critical_paths", {})
            },
            'visualization_type': 'enhanced_interaction_network_with_imports',
            'enhancement_version': '1.0.0'
        }