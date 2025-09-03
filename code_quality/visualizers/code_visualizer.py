"""
Base Code Visualizer

Provides base functionality for all visualization tools.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np


class CodeVisualizer:
    """Base class for code visualization tools."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path("code_quality/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def save_figure(self, fig: plt.Figure, filename: str, formats: List[str] = None) -> List[str]:
        """
        Save a figure in multiple formats.
        
        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
            formats: List of formats to save (default: ['png', 'pdf', 'svg'])
            
        Returns:
            List of saved file paths
        """
        if formats is None:
            formats = ['png', 'pdf', 'svg']
            
        saved_files = []
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
            saved_files.append(str(filepath))
            
        return saved_files
    
    def create_color_map(self, values: List[float], cmap_name: str = 'RdYlGn_r') -> List[str]:
        """
        Create colors for values using a colormap.
        
        Args:
            values: List of numeric values
            cmap_name: Name of the colormap
            
        Returns:
            List of hex color codes
        """
        if not values:
            return []
            
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        
        return [mcolors.to_hex(cmap(norm(v))) for v in values]
    
    def format_label(self, text: str, max_length: int = 30) -> str:
        """
        Format a label for display.
        
        Args:
            text: Label text
            max_length: Maximum length before truncation
            
        Returns:
            Formatted label
        """
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + '...'
    
    def create_legend_entries(self, categories: Dict[str, str]) -> List[Tuple[Rectangle, str]]:
        """
        Create legend entries with colored rectangles.
        
        Args:
            categories: Dict mapping category names to colors
            
        Returns:
            List of (patch, label) tuples for legend
        """
        entries = []
        for category, color in categories.items():
            patch = Rectangle((0, 0), 1, 1, fc=color, edgecolor='black', linewidth=0.5)
            entries.append((patch, category))
        return entries
    
    def save_metadata(self, filename: str, metadata: Dict[str, Any]):
        """
        Save visualization metadata as JSON.
        
        Args:
            filename: Base filename (without extension)
            metadata: Metadata dictionary
        """
        filepath = self.output_dir / f"{filename}_metadata.json"
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def create_summary_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary statistics from data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Summary statistics
        """
        stats = {
            'total_items': len(data),
            'timestamp': str(Path.ctime(Path(__file__))),
            'visualizer': self.__class__.__name__
        }
        return stats