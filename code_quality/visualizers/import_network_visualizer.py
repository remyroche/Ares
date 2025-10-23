#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Import Network Visualizer

Creates advanced visualizations of import relationships and dependencies
using data from the ImportVerifierAnalyzer. This visualizer enhances the
existing code quality pipeline with sophisticated import analysis graphs.
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    tprint("Warning: Plotly not available - interactive visualizations will be limited")

from .code_visualizer import CodeVisualizer


class ImportNetworkVisualizer(CodeVisualizer):
    """Advanced visualizer for import relationships and dependencies."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir)
        self.import_graph = nx.DiGraph()
        self.module_metadata = {}
        
    def create_import_network_from_verifier_data(self, verifier_results: Dict[str, Any], 
                                               title: str = "Import Network Analysis") -> Tuple[Any, Dict[str, Any]]:
        """
        Create comprehensive import network visualization from ImportVerifierAnalyzer results.
        
        Args:
            verifier_results: Results from ImportVerifierAnalyzer
            title: Visualization title
            
        Returns:
            Tuple of (figure, metadata)
        """
        # Extract data from verifier results
        import_status = verifier_results.get("import_status", {})
        summary = verifier_results.get("summary", {})
        advanced_analysis = verifier_results.get("advanced_analysis", {})
        
        # Build the import network graph
        self._build_import_network(import_status)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # 1. Main import network
        self._plot_main_import_network(axes[0, 0], import_status, summary)
        
        # 2. Import depth analysis
        self._plot_import_depth_analysis(axes[0, 1], advanced_analysis)
        
        # 3. Critical path analysis
        self._plot_critical_paths(axes[1, 0], advanced_analysis)
        
        # 4. Import statistics
        self._plot_import_statistics(axes[1, 1], summary, import_status)
        
        plt.tight_layout()
        
        # Prepare comprehensive metadata
        metadata = self._generate_network_metadata(verifier_results)
        
        return fig, metadata
    
    def create_interactive_import_network(self, verifier_results: Dict[str, Any], 
                                        title: str = "Interactive Import Network") -> str:
        """
        Create an interactive import network using Plotly.
        
        Args:
            verifier_results: Results from ImportVerifierAnalyzer
            title: Network title
            
        Returns:
            Path to saved HTML file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations")
        
        import_status = verifier_results.get("import_status", {})
        
        # Build network graph
        G = nx.DiGraph()
        node_data = {}
        
        for file_path, status in import_status.items():
            # Add node with metadata
            G.add_node(file_path, **status)
            node_data[file_path] = status
        
        # Add edges based on import relationships
        for file_path, status in import_status.items():
            imported_by = status.get("imported_by", [])
            for importer in imported_by:
                G.add_edge(importer, file_path)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{Path(edge[0]).name} → {Path(edge[1]).name}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_hovertext = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node styling based on import status
            status = node_data[node]
            is_imported = status.get("is_imported", False)
            import_count = status.get("import_count", 0)
            only_non_prod = status.get("only_imported_by_non_production", False)
            
            # Color coding
            if only_non_prod:
                color = 'orange'  # Only imported by non-production
            elif is_imported:
                color = 'green'   # Imported by production code
            else:
                color = 'red'     # Not imported
            
            node_colors.append(color)
            node_sizes.append(max(10, min(50, 10 + import_count * 2)))
            
            # Hover text
            hover_text = f"<b>{Path(node).name}</b><br>"
            hover_text += f"Full path: {node}<br>"
            hover_text += f"Imported: {'Yes' if is_imported else 'No'}<br>"
            hover_text += f"Import count: {import_count}<br>"
            hover_text += f"Only non-prod: {'Yes' if only_non_prod else 'No'}<br>"
            if status.get("imported_by"):
                hover_text += f"Imported by: {len(status['imported_by'])} files"
            
            node_hovertext.append(hover_text)
            node_text.append(Path(node).name)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hovertext=node_hovertext,
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
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
        
        # Add legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Imported by production',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='orange'),
            name='Only imported by non-production',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Not imported',
            showlegend=True
        ))
        
        # Save as HTML
        html_file = self.output_dir / f"interactive_import_network_{title.replace(' ', '_').lower()}.html"
        fig.write_html(str(html_file))
        
        return str(html_file)
    
    def create_import_heatmap(self, verifier_results: Dict[str, Any], 
                            title: str = "Import Relationship Heatmap") -> Any:
        """
        Create a heatmap showing import relationships between modules.
        
        Args:
            verifier_results: Results from ImportVerifierAnalyzer
            title: Heatmap title
            
        Returns:
            Matplotlib figure
        """
        import_status = verifier_results.get("import_status", {})
        
        # Build import matrix
        files = list(import_status.keys())
        n_files = len(files)
        
        if n_files == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No files to analyze', ha='center', va='center', fontsize=16)
            ax.axis('off')
            return fig
        
        # Create import matrix (file i imports file j)
        import_matrix = np.zeros((n_files, n_files))
        file_to_idx = {f: i for i, f in enumerate(files)}
        
        for file_path, status in import_status.items():
            imported_by = status.get("imported_by", [])
            for importer in imported_by:
                if importer in file_to_idx:
                    import_matrix[file_to_idx[importer], file_to_idx[file_path]] = 1
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Main heatmap
        im = ax1.imshow(import_matrix, cmap='Blues', aspect='auto')
        
        # Set labels
        file_labels = [Path(f).name for f in files]
        ax1.set_xticks(range(n_files))
        ax1.set_yticks(range(n_files))
        ax1.set_xticklabels(file_labels, rotation=90, fontsize=8)
        ax1.set_yticklabels(file_labels, fontsize=8)
        
        ax1.set_title("Import Relationships", fontsize=14)
        ax1.set_xlabel("Imported Files")
        ax1.set_ylabel("Importing Files")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Import Relationship', rotation=270, labelpad=15)
        
        # Import statistics
        import_counts = [status.get("import_count", 0) for status in import_status.values()]
        unimported_count = sum(1 for status in import_status.values() if not status.get("is_imported", False))
        
        # Create bar chart of import counts
        top_imported = sorted(zip(files, import_counts), key=lambda x: x[1], reverse=True)[:10]
        if top_imported:
            top_files, top_counts = zip(*top_imported)
            top_labels = [Path(f).name for f in top_files]
            
            y_pos = range(len(top_labels))
            ax2.barh(y_pos, top_counts, color='steelblue', alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_labels)
            ax2.set_xlabel('Number of Importers')
            ax2.set_title('Most Imported Files', fontsize=14)
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_circular_dependency_analysis(self, verifier_results: Dict[str, Any], 
                                          title: str = "Circular Dependency Analysis") -> Any:
        """
        Create detailed circular dependency analysis visualization.
        
        Args:
            verifier_results: Results from ImportVerifierAnalyzer
            title: Analysis title
            
        Returns:
            Matplotlib figure
        """
        advanced_analysis = verifier_results.get("advanced_analysis", {})
        circular_imports = advanced_analysis.get("circular_imports", [])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        if not circular_imports:
            # No circular dependencies
            ax1.text(0.5, 0.5, '✅ No Circular Dependencies Found!', 
                    ha='center', va='center', fontsize=20, color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax1.axis('off')
            
            ax2.text(0.5, 0.5, 'Your codebase has a clean\nimport structure!', 
                    ha='center', va='center', fontsize=16, color='darkgreen')
            ax2.axis('off')
        else:
            # Visualize circular dependencies
            cycle_graph = nx.DiGraph()
            for cycle in circular_imports:
                for i in range(len(cycle)):
                    cycle_graph.add_edge(cycle[i], cycle[(i + 1) % len(cycle)])
            
            # Use circular layout
            pos = nx.circular_layout(cycle_graph)
            
            # Draw the cycle graph
            nx.draw_networkx_nodes(cycle_graph, pos, ax=ax1,
                                 node_color='red',
                                 node_size=1000,
                                 alpha=0.7)
            
            nx.draw_networkx_labels(cycle_graph, pos, ax=ax1,
                                  labels={n: Path(n).name for n in cycle_graph.nodes()},
                                  font_size=10)
            
            nx.draw_networkx_edges(cycle_graph, pos, ax=ax1,
                                 edge_color='darkred',
                                 arrows=True,
                                 arrowsize=20,
                                 width=2,
                                 arrowstyle='->')
            
            ax1.set_title(f"Circular Dependencies ({len(circular_imports)} found)", fontsize=14)
            ax1.axis('off')
            
            # List circular dependencies
            cycle_text = f"Found {len(circular_imports)} circular dependencies:\n\n"
            for i, cycle in enumerate(circular_imports[:10], 1):  # Show first 10
                cycle_names = [Path(f).name for f in cycle]
                cycle_text += f"{i}. {' → '.join(cycle_names)} → {cycle_names[0]}\n"
            if len(circular_imports) > 10:
                cycle_text += f"\n... and {len(circular_imports) - 10} more"
            
            ax2.text(0.05, 0.95, cycle_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax2.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _build_import_network(self, import_status: Dict[str, Any]) -> None:
        """Build NetworkX graph from import status data."""
        self.import_graph.clear()
        
        for file_path, status in import_status.items():
            # Add node with metadata
            self.import_graph.add_node(file_path, **status)
            
            # Add edges for import relationships
            imported_by = status.get("imported_by", [])
            for importer in imported_by:
                self.import_graph.add_edge(importer, file_path)
    
    def _plot_main_import_network(self, ax, import_status: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """Plot the main import network."""
        if not self.import_graph.nodes():
            ax.text(0.5, 0.5, 'No import data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Calculate layout
        pos = nx.spring_layout(self.import_graph, k=2, iterations=50)
        
        # Node styling based on import status
        node_colors = []
        node_sizes = []
        
        for node in self.import_graph.nodes():
            status = import_status.get(node, {})
            is_imported = status.get("is_imported", False)
            import_count = status.get("import_count", 0)
            only_non_prod = status.get("only_imported_by_non_production", False)
            
            if only_non_prod:
                node_colors.append('orange')
            elif is_imported:
                node_colors.append('green')
            else:
                node_colors.append('red')
            
            node_sizes.append(max(100, min(1000, 100 + import_count * 50)))
        
        # Draw network
        nx.draw_networkx_nodes(self.import_graph, pos, ax=ax,
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8,
                             edgecolors='black',
                             linewidths=1)
        
        # Draw edges
        nx.draw_networkx_edges(self.import_graph, pos, ax=ax,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=10,
                             alpha=0.5,
                             arrowstyle='->')
        
        # Draw labels
        labels = {n: Path(n).name for n in self.import_graph.nodes()}
        nx.draw_networkx_labels(self.import_graph, pos, labels, ax=ax, font_size=8)
        
        ax.set_title("Import Network", fontsize=14)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='green', s=100, edgecolors='black', label='Imported by production'),
            plt.scatter([], [], c='orange', s=100, edgecolors='black', label='Only non-production'),
            plt.scatter([], [], c='red', s=100, edgecolors='black', label='Not imported')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_import_depth_analysis(self, ax, advanced_analysis: Dict[str, Any]) -> None:
        """Plot import depth analysis."""
        import_depths = advanced_analysis.get("import_depths", {})
        
        if not import_depths:
            ax.text(0.5, 0.5, 'No depth data available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Create histogram of import depths
        depths = list(import_depths.values())
        ax.hist(depths, bins=min(20, len(set(depths))), alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Import Depth')
        ax.set_ylabel('Number of Files')
        ax.set_title('Import Depth Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if depths:
            mean_depth = np.mean(depths)
            max_depth = max(depths)
            ax.axvline(mean_depth, color='red', linestyle='--', label=f'Mean: {mean_depth:.1f}')
            ax.axvline(max_depth, color='orange', linestyle='--', label=f'Max: {max_depth}')
            ax.legend()
    
    def _plot_critical_paths(self, ax, advanced_analysis: Dict[str, Any]) -> None:
        """Plot critical path analysis."""
        critical_paths = advanced_analysis.get("critical_paths", {})
        high_impact_files = critical_paths.get("high_impact_files", [])
        
        if not high_impact_files:
            ax.text(0.5, 0.5, 'No critical paths identified', ha='center', va='center')
            ax.axis('off')
            return
        
        # Plot top critical files
        top_files = high_impact_files[:10]  # Top 10
        if top_files:
            files, counts = zip(*top_files)
            file_labels = [Path(f).name for f in files]
            
            y_pos = range(len(file_labels))
            ax.barh(y_pos, counts, color='coral', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(file_labels)
            ax.set_xlabel('Number of Dependents')
            ax.set_title('Critical Files (High Impact)', fontsize=14)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
    
    def _plot_import_statistics(self, ax, summary: Dict[str, Any], import_status: Dict[str, Any]) -> None:
        """Plot import statistics."""
        # Create pie chart of import status
        total_files = summary.get("total_files", 0)
        imported_files = summary.get("imported_files", 0)
        unimported_files = summary.get("unimported_files", 0)
        only_non_prod = summary.get("only_non_production_files", 0)
        
        if total_files == 0:
            ax.text(0.5, 0.5, 'No files to analyze', ha='center', va='center')
            ax.axis('off')
            return
        
        # Calculate categories
        prod_imported = imported_files - only_non_prod
        
        labels = ['Imported by production', 'Only non-production', 'Not imported']
        sizes = [prod_imported, only_non_prod, unimported_files]
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
    
    def _generate_network_metadata(self, verifier_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata for the network visualization."""
        import_status = verifier_results.get("import_status", {})
        summary = verifier_results.get("summary", {})
        advanced_analysis = verifier_results.get("advanced_analysis", {})
        
        # Calculate network metrics
        if self.import_graph.nodes():
            network_metrics = {
                'total_nodes': len(self.import_graph.nodes()),
                'total_edges': len(self.import_graph.edges()),
                'density': nx.density(self.import_graph),
                'average_clustering': nx.average_clustering(self.import_graph.to_undirected()),
                'is_connected': nx.is_weakly_connected(self.import_graph),
                'strongly_connected_components': len(list(nx.strongly_connected_components(self.import_graph))),
                'weakly_connected_components': len(list(nx.weakly_connected_components(self.import_graph)))
            }
        else:
            network_metrics = {}
        
        return {
            'visualization_type': 'import_network_analysis',
            'timestamp': verifier_results.get("pipeline_info", {}).get("timestamp", ""),
            'project_root': verifier_results.get("pipeline_info", {}).get("project_root", ""),
            'summary_statistics': summary,
            'advanced_analysis': advanced_analysis,
            'network_metrics': network_metrics,
            'file_count': len(import_status),
            'visualizer_version': '1.0.0'
        }