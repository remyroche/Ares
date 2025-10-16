"""
Visualization utilities for TAS

Visualization tools for tree architecture search including:
- Tree structure visualization
- Architecture comparison
- Search progress visualization
- Performance metrics plotting
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class TreeVisualizer:
    """Tree visualization utilities."""

    def __init__(self):
        self.logger = logging.getLogger("TAS.TreeVisualizer")
        self.figure_size = (12, 8)

    def plot_tree_structure(self, tree_data: Dict[str, Any], save_path: Optional[str] = None):
        """Plot tree structure."""
        try:
            plt.figure(figsize=self.figure_size)

            # Simple tree representation
            if 'nodes' in tree_data:
                self._plot_tree_nodes(tree_data['nodes'])
            else:
                # Fallback: plot as simple text
                plt.text(0.5, 0.5, str(tree_data),
                        ha='center', va='center', fontsize=12)
                plt.title("Tree Structure")

            plt.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Tree structure saved to {save_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"❌ Failed to plot tree structure: {e}")

    def _plot_tree_nodes(self, nodes: List[Dict[str, Any]]):
        """Plot tree nodes recursively."""
        # Simplified implementation
        for i, node in enumerate(nodes):
            x = i * 0.1
            y = 0.5
            plt.scatter(x, y, s=100)
            plt.text(x, y + 0.1, str(node.get('value', i)),
                    ha='center', va='center', fontsize=10)

    def plot_fitness_history(self, fitness_history: List[float], save_path: Optional[str] = None):
        """Plot fitness evolution over generations."""
        try:
            plt.figure(figsize=self.figure_size)

            generations = range(1, len(fitness_history) + 1)
            plt.plot(generations, fitness_history, 'b-', linewidth=2, marker='o')

            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.title('Fitness Evolution')
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Fitness history saved to {save_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"❌ Failed to plot fitness history: {e}")

@dataclass
class TreeArchitectureVisualizer(TreeVisualizer):
    """Visualization for tree architectures."""

    def plot_architecture_comparison(self, architectures: List[Dict[str, Any]],
                                   metrics: List[str] = None, save_path: Optional[str] = None):
        """Compare multiple tree architectures."""
        try:
            plt.figure(figsize=self.figure_size)

            if not metrics:
                metrics = ['fitness', 'complexity', 'accuracy']

            n_architectures = len(architectures)
            n_metrics = len(metrics)

            for i, metric in enumerate(metrics):
                plt.subplot(1, n_metrics, i + 1)

                values = [arch.get(metric, 0) for arch in architectures]
                plt.bar(range(n_architectures), values, alpha=0.7)
                plt.xlabel('Architecture')
                plt.ylabel(metric.title())
                plt.title(f'Architecture {metric.title()} Comparison')
                plt.xticks(range(n_architectures), [f'Arch {j+1}' for j in range(n_architectures)])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Architecture comparison saved to {save_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"❌ Failed to plot architecture comparison: {e}")

@dataclass
class TreeSearchVisualizer(TreeVisualizer):
    """Visualization for tree search process."""

    def plot_search_space(self, search_history: List[Dict[str, Any]], save_path: Optional[str] = None):
        """Plot search space exploration."""
        try:
            plt.figure(figsize=self.figure_size)

            if search_history and len(search_history) > 0:
                # Extract fitness values
                fitness_values = [point.get('fitness', 0) for point in search_history]

                # Simple scatter plot of search points
                plt.scatter(range(len(fitness_values)), fitness_values, alpha=0.6, s=50)

                plt.xlabel('Search Iteration')
                plt.ylabel('Fitness')
                plt.title('Search Space Exploration')
                plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Search space plot saved to {save_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"❌ Failed to plot search space: {e}")

    def plot_convergence_analysis(self, convergence_data: Dict[str, List[float]],
                                save_path: Optional[str] = None):
        """Plot convergence analysis."""
        try:
            plt.figure(figsize=self.figure_size)

            for label, data in convergence_data.items():
                plt.plot(range(len(data)), data, label=label, marker='o')

            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.title('Convergence Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"✅ Convergence analysis saved to {save_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"❌ Failed to plot convergence analysis: {e}")
