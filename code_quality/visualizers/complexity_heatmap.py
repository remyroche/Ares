"""
Complexity Heatmap Visualizer

Creates heatmap visualizations of code complexity metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from .code_visualizer import CodeVisualizer


class ComplexityHeatmapVisualizer(CodeVisualizer):
    """Visualizes code complexity metrics as heatmaps."""
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir)
        
    def create_complexity_heatmap(self, complexity_data: Dict[str, Dict[str, float]], 
                                title: str = "Code Complexity Heatmap") -> Tuple[plt.Figure, Dict[str, Any]]:
        """
        Create a heatmap of code complexity metrics.
        
        Args:
            complexity_data: Dict mapping files to complexity metrics
            title: Heatmap title
            
        Returns:
            Tuple of (figure, metadata)
        """
        # Prepare data for heatmap
        files = list(complexity_data.keys())
        metrics = set()
        for file_metrics in complexity_data.values():
            metrics.update(file_metrics.keys())
        metrics = sorted(list(metrics))
        
        # Create matrix
        matrix = []
        for file in files:
            row = []
            for metric in metrics:
                value = complexity_data[file].get(metric, 0)
                row.append(value)
            matrix.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(matrix, index=[self.format_label(f, 30) for f in files], columns=metrics)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(10, len(files) * 0.3)))
        
        # Main heatmap
        sns.heatmap(df, ax=ax1, cmap='RdYlGn_r', annot=True, fmt='.1f', 
                   cbar_kws={'label': 'Complexity Score'})
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Files', fontsize=12)
        
        # Summary statistics
        self._plot_complexity_distribution(ax2, complexity_data)
        
        plt.tight_layout()
        
        # Calculate metadata
        all_complexities = []
        for file_metrics in complexity_data.values():
            all_complexities.extend(file_metrics.values())
        
        metadata = {
            'total_files': len(files),
            'metrics_tracked': metrics,
            'average_complexity': np.mean(all_complexities) if all_complexities else 0,
            'max_complexity': max(all_complexities) if all_complexities else 0,
            'min_complexity': min(all_complexities) if all_complexities else 0,
            'high_complexity_files': self._identify_high_complexity_files(complexity_data)
        }
        
        return fig, metadata
    
    def create_treemap_visualization(self, complexity_data: Dict[str, Dict[str, float]], 
                                   metric: str = 'cyclomatic_complexity',
                                   title: str = "Code Complexity Treemap") -> plt.Figure:
        """
        Create a treemap visualization of code complexity.
        
        Args:
            complexity_data: Complexity data by file
            metric: Specific metric to visualize
            title: Treemap title
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Extract data for the specific metric
        data = []
        for file, metrics in complexity_data.items():
            if metric in metrics:
                data.append({
                    'file': file,
                    'value': metrics[metric],
                    'path_parts': file.split('/')
                })
        
        if not data:
            ax.text(0.5, 0.5, f'No {metric} data available', 
                   ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        # Sort by value
        data.sort(key=lambda x: x['value'], reverse=True)
        
        # Create treemap using squarify
        try:
            import squarify
            
            values = [d['value'] for d in data]
            labels = [f"{self.format_label(d['file'], 20)}\n{d['value']:.1f}" for d in data]
            colors = self.create_color_map(values)
            
            squarify.plot(sizes=values, label=labels, color=colors, 
                         alpha=0.7, text_kwargs={'fontsize': 9})
            
            ax.set_title(f"{title} - {metric}", fontsize=16)
            ax.axis('off')
            
        except ImportError:
            # Fallback to bar chart if squarify not available
            files = [self.format_label(d['file'], 25) for d in data[:20]]
            values = [d['value'] for d in data[:20]]
            colors = self.create_color_map(values)
            
            y_pos = range(len(files))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(files)
            ax.set_xlabel(metric)
            ax.set_title(f"{title} - Top 20 Files by {metric}", fontsize=16)
            ax.invert_yaxis()
        
        return fig
    
    def create_complexity_timeline(self, historical_data: List[Dict[str, Any]], 
                                 title: str = "Complexity Over Time") -> plt.Figure:
        """
        Create a timeline visualization of complexity metrics.
        
        Args:
            historical_data: List of complexity snapshots with timestamps
            title: Graph title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if not historical_data:
            ax1.text(0.5, 0.5, 'No historical data available', 
                    ha='center', va='center', fontsize=20)
            ax1.axis('off')
            ax2.axis('off')
            return fig
        
        # Extract timeline data
        timestamps = []
        avg_complexities = []
        max_complexities = []
        file_counts = []
        
        for snapshot in historical_data:
            timestamps.append(snapshot.get('timestamp', ''))
            complexities = []
            for file_data in snapshot.get('files', {}).values():
                if 'cyclomatic_complexity' in file_data:
                    complexities.append(file_data['cyclomatic_complexity'])
            
            if complexities:
                avg_complexities.append(np.mean(complexities))
                max_complexities.append(max(complexities))
                file_counts.append(len(complexities))
            else:
                avg_complexities.append(0)
                max_complexities.append(0)
                file_counts.append(0)
        
        # Plot average and max complexity
        x = range(len(timestamps))
        ax1.plot(x, avg_complexities, 'b-o', label='Average Complexity', linewidth=2)
        ax1.plot(x, max_complexities, 'r-s', label='Maximum Complexity', linewidth=2)
        ax1.fill_between(x, avg_complexities, alpha=0.3)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([t[:10] for t in timestamps], rotation=45)
        ax1.set_ylabel('Complexity Score')
        ax1.set_title(f"{title} - Complexity Trends", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot file count
        ax2.bar(x, file_counts, color='green', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([t[:10] for t in timestamps], rotation=45)
        ax2.set_ylabel('Number of Files')
        ax2.set_title('Files Analyzed Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_module_complexity_bubble_chart(self, complexity_data: Dict[str, Dict[str, float]], 
                                            title: str = "Module Complexity Overview") -> plt.Figure:
        """
        Create a bubble chart showing multiple complexity dimensions.
        
        Args:
            complexity_data: Complexity data by file
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Extract data for bubble chart
        files = []
        cyclomatic = []
        lines_of_code = []
        maintainability = []
        
        for file, metrics in complexity_data.items():
            if all(m in metrics for m in ['cyclomatic_complexity', 'lines_of_code', 'maintainability_index']):
                files.append(file)
                cyclomatic.append(metrics['cyclomatic_complexity'])
                lines_of_code.append(metrics['lines_of_code'])
                maintainability.append(metrics['maintainability_index'])
        
        if not files:
            ax.text(0.5, 0.5, 'Insufficient data for bubble chart', 
                   ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        # Create bubble chart
        # Size based on lines of code
        sizes = [loc * 2 for loc in lines_of_code]
        # Color based on maintainability
        colors = self.create_color_map(maintainability, cmap_name='RdYlGn')
        
        scatter = ax.scatter(cyclomatic, maintainability, s=sizes, c=colors, 
                           alpha=0.6, edgecolors='black', linewidth=1)
        
        # Add labels for outliers
        for i, file in enumerate(files):
            if cyclomatic[i] > np.percentile(cyclomatic, 90) or maintainability[i] < np.percentile(maintainability, 10):
                ax.annotate(self.format_label(file, 20), 
                          (cyclomatic[i], maintainability[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Cyclomatic Complexity', fontsize=12)
        ax.set_ylabel('Maintainability Index', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Add size legend
        for size, label in [(100, 'Small'), (500, 'Medium'), (1000, 'Large')]:
            ax.scatter([], [], s=size*2, c='gray', alpha=0.6, 
                      edgecolors='black', linewidth=1, label=f'{label} ({size} LOC)')
        ax.legend(title='File Size', loc='best')
        
        # Add colorbar for maintainability
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                                   norm=plt.Normalize(vmin=min(maintainability), 
                                                     vmax=max(maintainability)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Maintainability Index', fontsize=10)
        
        return fig
    
    def _plot_complexity_distribution(self, ax: plt.Axes, complexity_data: Dict[str, Dict[str, float]]):
        """Plot distribution of complexity metrics."""
        # Collect all complexity values by metric type
        metrics_distribution = {}
        
        for file_metrics in complexity_data.values():
            for metric, value in file_metrics.items():
                if metric not in metrics_distribution:
                    metrics_distribution[metric] = []
                metrics_distribution[metric].append(value)
        
        # Create box plot
        if metrics_distribution:
            data_for_plot = []
            labels = []
            
            for metric, values in sorted(metrics_distribution.items()):
                if values:
                    data_for_plot.append(values)
                    labels.append(metric.replace('_', ' ').title())
            
            if data_for_plot:
                box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
                
                # Color boxes
                colors = self.create_color_map(range(len(data_for_plot)), cmap_name='Set3')
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel('Values')
                ax.set_title('Complexity Metrics Distribution', fontsize=14)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
    
    def _identify_high_complexity_files(self, complexity_data: Dict[str, Dict[str, float]], 
                                      threshold_percentile: float = 80) -> List[Dict[str, Any]]:
        """Identify files with high complexity."""
        high_complexity_files = []
        
        # Calculate thresholds for each metric
        metrics_values = {}
        for file_metrics in complexity_data.values():
            for metric, value in file_metrics.items():
                if metric not in metrics_values:
                    metrics_values[metric] = []
                metrics_values[metric].append(value)
        
        thresholds = {}
        for metric, values in metrics_values.items():
            if values:
                thresholds[metric] = np.percentile(values, threshold_percentile)
        
        # Find files exceeding thresholds
        for file, metrics in complexity_data.items():
            violations = []
            for metric, value in metrics.items():
                if metric in thresholds and value > thresholds[metric]:
                    violations.append({
                        'metric': metric,
                        'value': value,
                        'threshold': thresholds[metric]
                    })
            
            if violations:
                high_complexity_files.append({
                    'file': file,
                    'violations': violations
                })
        
        return high_complexity_files