"""
Regime Analysis Visualization and Reporting Tools.

This module provides comprehensive visualization and reporting capabilities
for market regime analysis results. It creates publication-quality charts,
interactive dashboards, and detailed reports for regime research.

Key Visualization Features:
- Regime time series plots with market data overlay
- Clustering quality metrics visualization
- Feature importance heatmaps and rankings
- Dimension analysis radar charts
- Performance comparison dashboards
- Interactive regime exploration tools
- Publication-ready report generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# Interactive plotting (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.utils.logger import system_logger


class VisualizationType(Enum):
    """Enumeration of visualization types."""
    REGIME_TIMESERIES = "regime_timeseries"
    CLUSTERING_QUALITY = "clustering_quality"
    FEATURE_IMPORTANCE = "feature_importance"
    DIMENSION_ANALYSIS = "dimension_analysis"
    VALIDATION_METRICS = "validation_metrics"
    PERFORMANCE_COMPARISON = "performance_comparison"
    REGIME_TRANSITIONS = "regime_transitions"
    CORRELATION_MATRIX = "correlation_matrix"
    DASHBOARD = "dashboard"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    # General settings
    style: str = "whitegrid"  # seaborn style
    color_palette: str = "Set2"  # color palette
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    # Interactive settings
    use_plotly: bool = True
    plotly_theme: str = "plotly_white"
    
    # Output settings
    save_plots: bool = True
    output_format: str = "png"  # png, pdf, svg
    output_directory: str = "regime_visualizations"
    
    # Plot-specific settings
    show_regime_labels: bool = True
    show_transitions: bool = True
    show_confidence_intervals: bool = True
    max_regimes_display: int = 10
    
    # Color schemes for regimes
    regime_colors: Optional[List[str]] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.regime_colors is None:
            self.regime_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]


class RegimeVisualization:
    """
    Comprehensive visualization system for regime analysis.
    
    This class provides various visualization methods for regime analysis
    results, including static plots, interactive dashboards, and reports.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the regime visualization system.
        
        Args:
            config: Configuration for visualizations
        """
        self.config = config or VisualizationConfig()
        self.logger = system_logger.getChild('RegimeVisualization')
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_style(self.config.style)
        sns.set_palette(self.config.color_palette)
        
        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        if self.config.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_regime_timeseries(self,
                              market_data: pd.DataFrame,
                              regime_labels: np.ndarray,
                              title: str = "Market Regimes Over Time",
                              show_price: bool = True) -> plt.Figure:
        """
        Plot regime labels over time with market data.
        
        Args:
            market_data: Market data with datetime index
            regime_labels: Regime assignments
            title: Plot title
            show_price: Whether to show price data
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("📊 Creating regime timeseries plot")
        
        fig, axes = plt.subplots(2 if show_price else 1, 1, 
                                figsize=self.config.figure_size,
                                sharex=True)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Prepare data
        if isinstance(market_data.index, pd.DatetimeIndex):
            x_data = market_data.index
        else:
            x_data = range(len(market_data))
        
        # Ensure regime_labels matches data length
        min_len = min(len(market_data), len(regime_labels))
        x_data = x_data[:min_len]
        regime_labels = regime_labels[:min_len]
        
        # Plot price data if requested
        if show_price:
            ax_price = axes[0]
            if 'close' in market_data.columns:
                ax_price.plot(x_data, market_data['close'].iloc[:min_len], 
                            color='black', linewidth=1, alpha=0.7)
                ax_price.set_ylabel('Price')
                ax_price.set_title(f'{title} - Price Chart')
                ax_price.grid(True, alpha=0.3)
        
        # Plot regime labels
        ax_regime = axes[-1]
        
        # Create regime color mapping
        unique_regimes = np.unique(regime_labels)
        regime_color_map = {
            regime: self.config.regime_colors[i % len(self.config.regime_colors)]
            for i, regime in enumerate(unique_regimes)
        }
        
        # Plot regime background colors
        current_regime = regime_labels[0]
        start_idx = 0
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != current_regime or i == len(regime_labels) - 1:
                end_idx = i if i != len(regime_labels) - 1 else i
                
                # Add colored background
                ax_regime.axvspan(x_data[start_idx], x_data[end_idx],
                                alpha=0.3, color=regime_color_map[current_regime],
                                label=f'Regime {current_regime}' if start_idx == 0 else "")
                
                current_regime = regime_labels[i]
                start_idx = i
        
        # Plot regime line
        ax_regime.plot(x_data, regime_labels, color='red', linewidth=2, alpha=0.8)
        ax_regime.set_ylabel('Regime')
        ax_regime.set_xlabel('Time')
        ax_regime.set_title(f'{title} - Regime Labels')
        ax_regime.grid(True, alpha=0.3)
        
        # Format x-axis for datetime
        if isinstance(market_data.index, pd.DatetimeIndex):
            ax_regime.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_regime.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax_regime.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        handles, labels = ax_regime.get_legend_handles_labels()
        if handles:
            ax_regime.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"regime_timeseries.{self.config.output_format}"
            plt.savefig(self.output_dir / filename, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"💾 Saved plot: {filename}")
        
        return fig
    
    def plot_clustering_quality_comparison(self,
                                         clustering_results: Dict[str, Dict[str, float]],
                                         title: str = "Clustering Quality Comparison") -> plt.Figure:
        """
        Plot comparison of clustering quality metrics.
        
        Args:
            clustering_results: Dictionary of clustering results with metrics
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("📊 Creating clustering quality comparison plot")
        
        # Prepare data
        methods = list(clustering_results.keys())
        metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        
        # Create subplot
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract metric values
            values = []
            method_names = []
            
            for method, results in clustering_results.items():
                if metric in results:
                    values.append(results[metric])
                    method_names.append(method)
            
            if values:
                # Create bar plot
                bars = ax.bar(method_names, values, 
                            color=self.config.regime_colors[:len(method_names)])
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel('Score')
                
                # Rotate x-axis labels if needed
                if len(max(method_names, key=len)) > 8:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"clustering_quality_comparison.{self.config.output_format}"
            plt.savefig(self.output_dir / filename, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"💾 Saved plot: {filename}")
        
        return fig
    
    def plot_feature_importance_heatmap(self,
                                      feature_importance_results: Dict[str, Dict[str, float]],
                                      top_n: int = 20,
                                      title: str = "Feature Importance Across Methods") -> plt.Figure:
        """
        Plot feature importance heatmap across different methods.
        
        Args:
            feature_importance_results: Dictionary of feature importance results
            top_n: Number of top features to display
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("📊 Creating feature importance heatmap")
        
        # Prepare data
        all_features = set()
        for method_results in feature_importance_results.values():
            if isinstance(method_results, dict):
                all_features.update(method_results.keys())
        
        # Get top features by average importance
        feature_avg_importance = {}
        for feature in all_features:
            importances = []
            for method_results in feature_importance_results.values():
                if isinstance(method_results, dict) and feature in method_results:
                    importances.append(method_results[feature])
            
            if importances:
                feature_avg_importance[feature] = np.mean(importances)
        
        # Select top N features
        top_features = sorted(feature_avg_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:top_n]
        top_feature_names = [f[0] for f in top_features]
        
        # Create importance matrix
        importance_matrix = []
        method_names = list(feature_importance_results.keys())
        
        for method in method_names:
            method_results = feature_importance_results[method]
            if isinstance(method_results, dict):
                row = [method_results.get(feature, 0) for feature in top_feature_names]
                importance_matrix.append(row)
        
        if importance_matrix:
            # Create heatmap
            fig, ax = plt.subplots(figsize=(max(12, len(top_feature_names) * 0.5), 
                                          max(6, len(method_names) * 0.5)))
            
            sns.heatmap(importance_matrix, 
                       xticklabels=top_feature_names,
                       yticklabels=method_names,
                       annot=True, fmt='.3f',
                       cmap='YlOrRd', ax=ax)
            
            ax.set_title(title)
            ax.set_xlabel('Features')
            ax.set_ylabel('Methods')
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            if self.config.save_plots:
                filename = f"feature_importance_heatmap.{self.config.output_format}"
                plt.savefig(self.output_dir / filename, dpi=self.config.dpi, bbox_inches='tight')
                self.logger.info(f"💾 Saved plot: {filename}")
            
            return fig
        
        # Return empty figure if no data
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.text(0.5, 0.5, 'No feature importance data available', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    def plot_dimension_analysis_radar(self,
                                    dimension_results: Dict[str, Dict[str, float]],
                                    title: str = "Market Dimension Analysis") -> plt.Figure:
        """
        Plot radar chart for market dimension analysis.
        
        Args:
            dimension_results: Dictionary of dimension analysis results
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("📊 Creating dimension analysis radar chart")
        
        # Prepare data
        dimensions = list(dimension_results.keys())
        metrics = ['importance_score', 'stability_score', 'predictive_power', 'regime_discriminability']
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, subplot_kw=dict(projection='polar'))
        
        for i, (dimension, results) in enumerate(dimension_results.items()):
            if i >= self.config.max_regimes_display:
                break
                
            values = []
            for metric in metrics:
                value = results.get(metric, 0)
                values.append(value)
            
            values += values[:1]  # Complete the circle
            
            color = self.config.regime_colors[i % len(self.config.regime_colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=dimension.replace('_', ' ').title(), color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"dimension_analysis_radar.{self.config.output_format}"
            plt.savefig(self.output_dir / filename, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"💾 Saved plot: {filename}")
        
        return fig
    
    def plot_validation_metrics_comparison(self,
                                         validation_results: Dict[str, Dict[str, Any]],
                                         title: str = "Validation Metrics Comparison") -> plt.Figure:
        """
        Plot comparison of validation metrics across methods.
        
        Args:
            validation_results: Dictionary of validation results
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("📊 Creating validation metrics comparison plot")
        
        # Prepare data
        methods = list(validation_results.keys())
        
        # Extract key metrics
        key_metrics = ['silhouette_score', 'temporal_consistency', 'return_separability', 
                      'regime_balance', 'regime_homogeneity']
        
        # Create subplot
        n_metrics = len(key_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            
            values = []
            method_names = []
            colors = []
            
            for j, (method, results) in enumerate(validation_results.items()):
                if metric in results and 'value' in results[metric]:
                    values.append(results[metric]['value'])
                    method_names.append(method)
                    colors.append(self.config.regime_colors[j % len(self.config.regime_colors)])
            
            if values:
                bars = ax.bar(method_names, values, color=colors)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Score')
                ax.set_ylim(0, max(values) * 1.1)
                
                # Rotate labels if needed
                if any(len(name) > 8 for name in method_names):
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"validation_metrics_comparison.{self.config.output_format}"
            plt.savefig(self.output_dir / filename, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"💾 Saved plot: {filename}")
        
        return fig
    
    def plot_regime_transitions(self,
                              regime_labels: np.ndarray,
                              title: str = "Regime Transition Matrix") -> plt.Figure:
        """
        Plot regime transition matrix.
        
        Args:
            regime_labels: Regime assignments
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("📊 Creating regime transition matrix plot")
        
        # Calculate transition matrix
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)
        
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            from_regime = np.where(unique_regimes == regime_labels[i])[0][0]
            to_regime = np.where(unique_regimes == regime_labels[i + 1])[0][0]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, 
                                   out=np.zeros_like(transition_matrix), 
                                   where=row_sums!=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(transition_probs,
                   xticklabels=[f'Regime {r}' for r in unique_regimes],
                   yticklabels=[f'Regime {r}' for r in unique_regimes],
                   annot=True, fmt='.3f',
                   cmap='Blues', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = f"regime_transitions.{self.config.output_format}"
            plt.savefig(self.output_dir / filename, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"💾 Saved plot: {filename}")
        
        return fig
    
    def create_interactive_dashboard(self,
                                   market_data: pd.DataFrame,
                                   regime_labels: np.ndarray,
                                   analysis_results: Dict[str, Any]) -> str:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            market_data: Market data
            regime_labels: Regime assignments
            analysis_results: Analysis results
            
        Returns:
            Path to saved HTML file
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available, skipping interactive dashboard")
            return ""
        
        self.logger.info("📊 Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Price and Regimes', 'Regime Distribution', 
                          'Feature Importance', 'Validation Metrics',
                          'Clustering Quality', 'Dimension Analysis'],
            specs=[[{"secondary_y": True}, {}],
                   [{}, {}],
                   [{}, {}]]
        )
        
        # 1. Price and Regimes
        if 'close' in market_data.columns:
            fig.add_trace(
                go.Scatter(x=market_data.index, y=market_data['close'],
                          name='Price', line=dict(color='black')),
                row=1, col=1
            )
        
        # Add regime background colors
        unique_regimes = np.unique(regime_labels)
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            if np.any(regime_mask):
                fig.add_trace(
                    go.Scatter(x=market_data.index[regime_mask], 
                             y=[regime] * np.sum(regime_mask),
                             mode='markers',
                             name=f'Regime {regime}',
                             marker=dict(color=self.config.regime_colors[regime % len(self.config.regime_colors)])),
                    row=1, col=1, secondary_y=True
                )
        
        # 2. Regime Distribution
        regime_counts = np.bincount(regime_labels)
        fig.add_trace(
            go.Bar(x=[f'Regime {i}' for i in range(len(regime_counts))],
                   y=regime_counts,
                   name='Regime Distribution'),
            row=1, col=2
        )
        
        # 3. Feature Importance (if available)
        if 'feature_importance' in analysis_results:
            importance_data = analysis_results['feature_importance']
            if isinstance(importance_data, dict) and 'consensus_features' in importance_data:
                features, scores, _ = zip(*importance_data['consensus_features'][:10])
                fig.add_trace(
                    go.Bar(x=list(scores), y=list(features), 
                           orientation='h',
                           name='Feature Importance'),
                    row=2, col=1
                )
        
        # 4. Validation Metrics (if available)
        if 'validation_metrics' in analysis_results:
            validation_data = analysis_results['validation_metrics']
            for method, metrics in validation_data.items():
                if isinstance(metrics, dict):
                    metric_names = []
                    metric_values = []
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and 'value' in metric_data:
                            metric_names.append(metric_name.replace('_', ' ').title())
                            metric_values.append(metric_data['value'])
                    
                    if metric_names:
                        fig.add_trace(
                            go.Bar(x=metric_names, y=metric_values,
                                   name=f'{method} Validation'),
                            row=2, col=2
                        )
                        break  # Only show first method to avoid clutter
        
        # 5. Clustering Quality (if available)
        if 'clustering_results' in analysis_results:
            clustering_data = analysis_results['clustering_results']
            if 'all_results' in clustering_data:
                methods = []
                silhouette_scores = []
                
                for method, result in clustering_data['all_results'].items():
                    if 'metrics' in result and 'silhouette_score' in result['metrics']:
                        methods.append(method)
                        silhouette_scores.append(result['metrics']['silhouette_score'])
                
                if methods:
                    fig.add_trace(
                        go.Bar(x=methods, y=silhouette_scores,
                               name='Silhouette Score'),
                        row=3, col=1
                    )
        
        # 6. Dimension Analysis (if available)
        if 'dimension_analysis' in analysis_results:
            dimension_data = analysis_results['dimension_analysis']
            if 'top_dimensions' in dimension_data:
                dimensions = []
                composite_scores = []
                
                for dim_name, dim_data in dimension_data['top_dimensions'][:5]:
                    dimensions.append(dim_name.replace('_', ' ').title())
                    composite_scores.append(dim_data.get('metrics', {}).get('composite_score', 0))
                
                if dimensions:
                    fig.add_trace(
                        go.Bar(x=dimensions, y=composite_scores,
                               name='Dimension Score'),
                        row=3, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title_text="Market Regime Analysis Dashboard",
            showlegend=True,
            height=1200,
            template=self.config.plotly_theme
        )
        
        # Save interactive plot
        if self.config.save_plots:
            filename = "regime_analysis_dashboard.html"
            filepath = self.output_dir / filename
            pyo.plot(fig, filename=str(filepath), auto_open=False)
            self.logger.info(f"💾 Saved interactive dashboard: {filename}")
            return str(filepath)
        
        return ""
    
    def generate_comprehensive_report(self,
                                    market_data: pd.DataFrame,
                                    regime_labels: np.ndarray,
                                    analysis_results: Dict[str, Any],
                                    title: str = "Market Regime Analysis Report") -> str:
        """
        Generate comprehensive visual report.
        
        Args:
            market_data: Market data
            regime_labels: Regime assignments
            analysis_results: Complete analysis results
            title: Report title
            
        Returns:
            Path to saved report
        """
        self.logger.info("📊 Generating comprehensive visual report")
        
        # Create main plots
        plots_created = []
        
        try:
            # 1. Regime timeseries
            fig1 = self.plot_regime_timeseries(market_data, regime_labels)
            plots_created.append("Regime Timeseries")
            plt.close(fig1)
        except Exception as e:
            self.logger.error(f"Failed to create regime timeseries: {e}")
        
        try:
            # 2. Clustering quality comparison
            if 'clustering_results' in analysis_results and 'all_results' in analysis_results['clustering_results']:
                clustering_metrics = {}
                for method, result in analysis_results['clustering_results']['all_results'].items():
                    if 'metrics' in result:
                        clustering_metrics[method] = result['metrics']
                
                if clustering_metrics:
                    fig2 = self.plot_clustering_quality_comparison(clustering_metrics)
                    plots_created.append("Clustering Quality Comparison")
                    plt.close(fig2)
        except Exception as e:
            self.logger.error(f"Failed to create clustering comparison: {e}")
        
        try:
            # 3. Feature importance heatmap
            if 'feature_importance' in analysis_results and 'importance_results' in analysis_results['feature_importance']:
                importance_data = {}
                for method, result in analysis_results['feature_importance']['importance_results'].items():
                    if 'feature_names' in result and 'importance_scores' in result:
                        feature_names = result['feature_names']
                        importance_scores = result['importance_scores']
                        importance_data[method] = dict(zip(feature_names, importance_scores))
                
                if importance_data:
                    fig3 = self.plot_feature_importance_heatmap(importance_data)
                    plots_created.append("Feature Importance Heatmap")
                    plt.close(fig3)
        except Exception as e:
            self.logger.error(f"Failed to create feature importance heatmap: {e}")
        
        try:
            # 4. Dimension analysis radar
            if 'dimension_analysis' in analysis_results and 'dimension_results' in analysis_results['dimension_analysis']:
                dimension_data = analysis_results['dimension_analysis']['dimension_results']
                if dimension_data:
                    fig4 = self.plot_dimension_analysis_radar(dimension_data)
                    plots_created.append("Dimension Analysis Radar")
                    plt.close(fig4)
        except Exception as e:
            self.logger.error(f"Failed to create dimension radar: {e}")
        
        try:
            # 5. Validation metrics comparison
            if 'validation_metrics' in analysis_results:
                fig5 = self.plot_validation_metrics_comparison(analysis_results['validation_metrics'])
                plots_created.append("Validation Metrics Comparison")
                plt.close(fig5)
        except Exception as e:
            self.logger.error(f"Failed to create validation comparison: {e}")
        
        try:
            # 6. Regime transitions
            fig6 = self.plot_regime_transitions(regime_labels)
            plots_created.append("Regime Transitions")
            plt.close(fig6)
        except Exception as e:
            self.logger.error(f"Failed to create regime transitions: {e}")
        
        try:
            # 7. Interactive dashboard
            if PLOTLY_AVAILABLE:
                dashboard_path = self.create_interactive_dashboard(market_data, regime_labels, analysis_results)
                if dashboard_path:
                    plots_created.append("Interactive Dashboard")
        except Exception as e:
            self.logger.error(f"Failed to create interactive dashboard: {e}")
        
        # Generate summary report
        report_content = [
            f"# {title}",
            "=" * len(title),
            "",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Visualizations Created",
            ""
        ]
        
        for i, plot_name in enumerate(plots_created, 1):
            report_content.append(f"{i}. {plot_name}")
        
        report_content.extend([
            "",
            "## Analysis Summary",
            "",
            f"- **Data Points**: {len(market_data)}",
            f"- **Regimes Identified**: {len(np.unique(regime_labels))}",
            f"- **Analysis Components**: {len(analysis_results)}",
            "",
            "## Files Generated",
            ""
        ])
        
        # List generated files
        if self.config.save_plots:
            for file in self.output_dir.glob(f"*.{self.config.output_format}"):
                report_content.append(f"- {file.name}")
            for file in self.output_dir.glob("*.html"):
                report_content.append(f"- {file.name}")
        
        # Save report
        if self.config.save_plots:
            report_path = self.output_dir / "visual_report_summary.md"
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_content))
            
            self.logger.info(f"📊 Generated comprehensive visual report: {len(plots_created)} visualizations")
            return str(report_path)
        
        return '\n'.join(report_content)