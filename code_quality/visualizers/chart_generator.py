#!/usr/bin/env python3
from src.utils.tprint import tprint

"""Chart generator for code analysis visualizations."""

from typing import Dict, Any, Optional, List
from pathlib import Path


class ChartGenerator:
    """Generates charts and visualizations for code analysis."""
    
    def __init__(self, output_dir: str):
        """Initialize the chart generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dead_code_type_chart(self, dead_code_report) -> Optional[Any]:
        """Create a chart showing dead code issues by type."""
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(dead_code_report, 'issues_by_type') or not dead_code_report.issues_by_type:
                return None
            
            # Prepare data
            types = list(dead_code_report.issues_by_type.keys())
            counts = list(dead_code_report.issues_by_type.values())
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bar chart
            bars = ax.bar(types, counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'])
            
            # Customize chart
            ax.set_title('Dead Code Issues by Type', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Issue Type', fontsize=12)
            ax.set_ylabel('Number of Issues', fontsize=12)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            tprint("  - Matplotlib not available for dead code type chart")
            return None
        except Exception as e:
            tprint(f"  - Error creating dead code type chart: {e}")
            return None
    
    def create_dead_code_severity_chart(self, dead_code_report) -> Optional[Any]:
        """Create a chart showing dead code issues by severity."""
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(dead_code_report, 'issues_by_severity') or not dead_code_report.issues_by_severity:
                return None
            
            # Prepare data
            severities = list(dead_code_report.issues_by_severity.keys())
            counts = [len(issues) for issues in dead_code_report.issues_by_severity.values()]
            
            # Color mapping for severities
            color_map = {'high': '#ff4757', 'medium': '#ffa502', 'low': '#2ed573'}
            colors = [color_map.get(severity, '#747d8c') for severity in severities]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            wedges, texts, autotexts = ax1.pie(counts, labels=severities, colors=colors, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 10})
            ax1.set_title('Dead Code Issues by Severity', fontsize=14, fontweight='bold')
            
            # Bar chart
            bars = ax2.bar(severities, counts, color=colors)
            ax2.set_title('Dead Code Issues Count by Severity', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Severity Level', fontsize=12)
            ax2.set_ylabel('Number of Issues', fontsize=12)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            tprint("  - Matplotlib not available for dead code severity chart")
            return None
        except Exception as e:
            tprint(f"  - Error creating dead code severity chart: {e}")
            return None
    
    def save_figure(self, fig, filename: str) -> List[str]:
        """Save a matplotlib figure to files."""
        if fig is None:
            return []
        
        try:
            import matplotlib.pyplot as plt
            
            saved_files = []
            
            # Save as PNG
            png_file = self.output_dir / f"{filename}.png"
            fig.savefig(png_file, dpi=300, bbox_inches='tight')
            saved_files.append(str(png_file))
            
            # Save as SVG
            svg_file = self.output_dir / f"{filename}.svg"
            fig.savefig(svg_file, format='svg', bbox_inches='tight')
            saved_files.append(str(svg_file))
            
            plt.close(fig)
            return saved_files
            
        except Exception as e:
            tprint(f"  - Error saving figure {filename}: {e}")
            return []
