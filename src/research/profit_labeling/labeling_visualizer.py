"""
Labeling Visualizer for Multi-Horizon Profit Labeling Research

This module provides comprehensive visualization capabilities for profit labeling
analysis results, similar to the visualization systems used in HMM clustering research.
It creates publication-quality charts and interactive dashboards.

Key Visualization Categories:
1. Heuristic Analysis Visualizations
2. Validation Results Visualizations  
3. Parameter Optimization Visualizations
4. Labeling Quality Dashboards
5. Comparative Analysis Charts
6. Interactive Research Dashboards
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Optional advanced visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, callback
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from src.utils.logger import get_logger
from .heuristic_analyzer import HeuristicAnalysisResult, AnalysisMetric
from .labeling_validator import ValidationResult, ValidationMetric
from .parameter_optimizer import OptimizationResult, OptimizationMethod


class VisualizationType(Enum):
    """Enumeration of visualization types."""
    HEURISTIC_ANALYSIS = "heuristic_analysis"
    VALIDATION_RESULTS = "validation_results"
    OPTIMIZATION_RESULTS = "optimization_results"
    LABELING_QUALITY = "labeling_quality"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    INTERACTIVE_DASHBOARD = "interactive_dashboard"


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    # Output settings
    output_format: str = "png"  # png, pdf, svg, html
    output_dpi: int = 300
    figure_size: Tuple[int, int] = (12, 8)
    
    # Style settings
    style_theme: str = "seaborn-v0_8"  # matplotlib style
    color_palette: str = "husl"        # seaborn palette
    font_size: int = 12
    title_font_size: int = 14
    
    # Chart-specific settings
    show_confidence_intervals: bool = True
    show_statistical_significance: bool = True
    include_annotations: bool = True
    interactive_charts: bool = True
    
    # Dashboard settings
    dashboard_port: int = 8050
    dashboard_debug: bool = False
    
    # File organization
    create_subdirectories: bool = True
    timestamp_files: bool = True


class LabelingVisualizer:
    """
    Comprehensive visualizer for multi-horizon profit labeling research.
    
    This class provides publication-quality visualizations for all aspects
    of profit labeling research, including heuristic analysis, validation
    results, parameter optimization, and comparative studies.
    
    Key Features:
    1. **Static Charts**: High-quality matplotlib/seaborn visualizations
    2. **Interactive Charts**: Plotly-based interactive visualizations  
    3. **Dashboards**: Real-time interactive research dashboards
    4. **Export Options**: Multiple formats (PNG, PDF, SVG, HTML)
    5. **Customization**: Flexible styling and configuration options
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the labeling visualizer."""
        self.config = config or VisualizationConfig()
        self.logger = get_logger('LabelingVisualizer')
        
        # Set up matplotlib style
        plt.style.use(self.config.style_theme)
        sns.set_palette(self.config.color_palette)
        
        # Generated visualizations tracking
        self.generated_charts: Dict[str, Path] = {}
        self.dashboard_apps: Dict[str, Any] = {}
        
        self.logger.info('📊 Labeling Visualizer initialized')
        self.logger.info(f'   → Output format: {self.config.output_format}')
        self.logger.info(f'   → Interactive charts: {self.config.interactive_charts and PLOTLY_AVAILABLE}')
        
    def visualize_heuristic_analysis(self,
                                   analysis_results: Dict[str, HeuristicAnalysisResult],
                                   output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate visualizations for heuristic analysis results."""
        self.logger.info('📈 Generating heuristic analysis visualizations')
        
        output_dir = Path(output_dir)
        if self.config.create_subdirectories:
            output_dir = output_dir / "heuristic_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_charts = {}
        
        # 1. Target/Horizon Effectiveness Chart
        target_results = {k: v for k, v in analysis_results.items() 
                         if 'effectiveness' in k}
        if target_results:
            chart_path = self._create_target_effectiveness_chart(target_results, output_dir)
            generated_charts['target_effectiveness'] = chart_path
        
        # 2. Quality Scoring Analysis
        quality_results = {k: v for k, v in analysis_results.items() 
                          if v.analysis_type == AnalysisMetric.QUALITY_CONSISTENCY}
        if quality_results:
            chart_path = self._create_quality_scoring_chart(quality_results, output_dir)
            generated_charts['quality_scoring'] = chart_path
        
        # 3. Composite Score Analysis
        composite_results = {k: v for k, v in analysis_results.items() 
                           if v.analysis_type == AnalysisMetric.COMPOSITE_COHERENCE}
        if composite_results:
            chart_path = self._create_composite_scores_chart(composite_results, output_dir)
            generated_charts['composite_scores'] = chart_path
        
        # 4. Analysis Summary Dashboard
        if self.config.interactive_charts and PLOTLY_AVAILABLE:
            dashboard_path = self._create_heuristic_analysis_dashboard(
                analysis_results, output_dir
            )
            generated_charts['interactive_dashboard'] = dashboard_path
        
        self.generated_charts.update(generated_charts)
        self.logger.info(f'✅ Generated {len(generated_charts)} heuristic analysis charts')
        
        return generated_charts
    
    def visualize_validation_results(self,
                                   validation_results: Dict[str, ValidationResult],
                                   output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate visualizations for validation results."""
        self.logger.info('🔍 Generating validation results visualizations')
        
        output_dir = Path(output_dir)
        if self.config.create_subdirectories:
            output_dir = output_dir / "validation_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_charts = {}
        
        # 1. Validation Metrics Overview
        chart_path = self._create_validation_overview_chart(validation_results, output_dir)
        generated_charts['validation_overview'] = chart_path
        
        # 2. Statistical Significance Analysis
        significance_results = {k: v for k, v in validation_results.items() 
                              if v.p_value is not None}
        if significance_results:
            chart_path = self._create_significance_analysis_chart(significance_results, output_dir)
            generated_charts['significance_analysis'] = chart_path
        
        # 3. Bias Detection Visualization
        bias_results = {k: v for k, v in validation_results.items() 
                       if v.metric == ValidationMetric.BIAS_DETECTION}
        if bias_results:
            chart_path = self._create_bias_detection_chart(bias_results, output_dir)
            generated_charts['bias_detection'] = chart_path
        
        # 4. Validation Interactive Dashboard
        if self.config.interactive_charts and PLOTLY_AVAILABLE:
            dashboard_path = self._create_validation_dashboard(validation_results, output_dir)
            generated_charts['validation_dashboard'] = dashboard_path
        
        self.generated_charts.update(generated_charts)
        self.logger.info(f'✅ Generated {len(generated_charts)} validation charts')
        
        return generated_charts
    
    def visualize_optimization_results(self,
                                     optimization_results: Dict[str, OptimizationResult],
                                     output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate visualizations for parameter optimization results."""
        self.logger.info('🎯 Generating optimization results visualizations')
        
        output_dir = Path(output_dir)
        if self.config.create_subdirectories:
            output_dir = output_dir / "optimization_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_charts = {}
        
        # 1. Optimization Convergence Charts
        for method_name, result in optimization_results.items():
            chart_path = self._create_convergence_chart(result, method_name, output_dir)
            generated_charts[f'convergence_{method_name}'] = chart_path
        
        # 2. Method Comparison Chart
        if len(optimization_results) > 1:
            chart_path = self._create_method_comparison_chart(optimization_results, output_dir)
            generated_charts['method_comparison'] = chart_path
        
        # 3. Parameter Sensitivity Analysis
        for method_name, result in optimization_results.items():
            if len(result.optimization_history) > 10:
                chart_path = self._create_parameter_sensitivity_chart(
                    result, method_name, output_dir
                )
                generated_charts[f'sensitivity_{method_name}'] = chart_path
        
        # 4. Optimization Interactive Dashboard
        if self.config.interactive_charts and PLOTLY_AVAILABLE:
            dashboard_path = self._create_optimization_dashboard(
                optimization_results, output_dir
            )
            generated_charts['optimization_dashboard'] = dashboard_path
        
        self.generated_charts.update(generated_charts)
        self.logger.info(f'✅ Generated {len(generated_charts)} optimization charts')
        
        return generated_charts
    
    def create_comprehensive_research_dashboard(self,
                                              heuristic_results: Optional[Dict[str, HeuristicAnalysisResult]] = None,
                                              validation_results: Optional[Dict[str, ValidationResult]] = None,
                                              optimization_results: Optional[Dict[str, OptimizationResult]] = None,
                                              market_data: Optional[pd.DataFrame] = None,
                                              labeled_data: Optional[pd.DataFrame] = None,
                                              output_dir: Union[str, Path] = "research_dashboard") -> Optional[str]:
        """Create comprehensive interactive research dashboard."""
        if not (PLOTLY_AVAILABLE and DASH_AVAILABLE):
            self.logger.warning('Plotly and Dash required for interactive dashboard')
            return None
        
        self.logger.info('🚀 Creating comprehensive research dashboard')
        
        # Create Dash app
        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        # Define layout
        app.layout = html.Div([
            html.H1("Multi-Horizon Profit Labeling Research Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            dcc.Tabs(id="main-tabs", value='heuristic-tab', children=[
                dcc.Tab(label='Heuristic Analysis', value='heuristic-tab'),
                dcc.Tab(label='Validation Results', value='validation-tab'),
                dcc.Tab(label='Optimization Results', value='optimization-tab'),
                dcc.Tab(label='Data Explorer', value='data-tab'),
            ]),
            
            html.Div(id='tab-content', style={'padding': '20px'})
        ])
        
        # Define callbacks
        @app.callback(Output('tab-content', 'children'),
                     Input('main-tabs', 'value'))
        def render_content(active_tab):
            if active_tab == 'heuristic-tab':
                return self._create_heuristic_tab_content(heuristic_results)
            elif active_tab == 'validation-tab':
                return self._create_validation_tab_content(validation_results)
            elif active_tab == 'optimization-tab':
                return self._create_optimization_tab_content(optimization_results)
            elif active_tab == 'data-tab':
                return self._create_data_explorer_content(market_data, labeled_data)
            
            return html.Div([html.H3('Select a tab to view content')])
        
        # Store app for potential later use
        self.dashboard_apps['comprehensive'] = app
        
        # Save dashboard as HTML file
        output_path = Path(output_dir) / "comprehensive_dashboard.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate static HTML version
        dashboard_html = self._generate_dashboard_html(
            heuristic_results, validation_results, optimization_results
        )
        
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f'📊 Dashboard saved to {output_path}')
        
        # Return path to dashboard
        return str(output_path)
    
    # Chart creation methods
    def _create_target_effectiveness_chart(self,
                                         target_results: Dict[str, HeuristicAnalysisResult],
                                         output_dir: Path) -> Path:
        """Create target/horizon effectiveness chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Extract data
        targets = []
        hit_rates = []
        predictive_powers = []
        confidence_intervals = []
        
        for key, result in target_results.items():
            if 'effectiveness' in key:
                target_info = result.metadata.get('target', 'unknown')
                horizon_info = result.metadata.get('horizon', 'unknown')
                label = f"{target_info}_{horizon_info}"
                
                targets.append(label)
                hit_rates.append(result.metric_value)
                predictive_powers.append(result.metadata.get('predictive_power', 0))
                confidence_intervals.append(result.confidence_interval)
        
        # Chart 1: Hit Rates
        bars1 = ax1.bar(targets, hit_rates, alpha=0.7, color='steelblue')
        ax1.set_title('Target/Horizon Hit Rates', fontsize=self.config.title_font_size)
        ax1.set_ylabel('Hit Rate')
        ax1.set_xlabel('Target/Horizon Combination')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add confidence intervals if available
        if self.config.show_confidence_intervals and any(ci is not None for ci in confidence_intervals):
            for i, (bar, ci) in enumerate(zip(bars1, confidence_intervals)):
                if ci is not None:
                    lower, upper = ci
                    ax1.errorbar(i, hit_rates[i], yerr=[[hit_rates[i] - lower], [upper - hit_rates[i]]], 
                               fmt='none', color='black', capsize=5)
        
        # Chart 2: Predictive Power
        bars2 = ax2.bar(targets, predictive_powers, alpha=0.7, color='forestgreen')
        ax2.set_title('Predictive Power by Target/Horizon', fontsize=self.config.title_font_size)
        ax2.set_ylabel('Predictive Power')
        ax2.set_xlabel('Target/Horizon Combination')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        output_path = output_dir / f"target_effectiveness.{self.config.output_format}"
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_quality_scoring_chart(self,
                                    quality_results: Dict[str, HeuristicAnalysisResult],
                                    output_dir: Path) -> Path:
        """Create quality scoring analysis chart."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract quality consistency scores
        methods = []
        consistency_scores = []
        
        for key, result in quality_results.items():
            methods.append(key.replace('_', ' ').title())
            consistency_scores.append(result.metric_value)
        
        # Create horizontal bar chart
        bars = ax.barh(methods, consistency_scores, alpha=0.7, color='coral')
        ax.set_title('Quality Scoring Consistency Analysis', fontsize=self.config.title_font_size)
        ax.set_xlabel('Consistency Score')
        ax.set_xlim(0, 1.0)
        
        # Add value labels on bars
        for bar, score in zip(bars, consistency_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=self.config.font_size)
        
        # Add threshold line
        ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Good Threshold')
        ax.legend()
        
        plt.tight_layout()
        
        # Save chart
        output_path = output_dir / f"quality_scoring_analysis.{self.config.output_format}"
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_composite_scores_chart(self,
                                     composite_results: Dict[str, HeuristicAnalysisResult],
                                     output_dir: Path) -> Path:
        """Create composite scores analysis chart."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract data
        score_types = []
        coherence_values = []
        predictive_powers = []
        
        for key, result in composite_results.items():
            score_type = result.metadata.get('score_type', key)
            score_types.append(score_type.replace('_', ' ').title())
            coherence_values.append(result.metric_value)
            predictive_powers.append(result.metadata.get('predictive_power', 0))
        
        # Create scatter plot
        scatter = ax.scatter(coherence_values, predictive_powers, 
                           s=100, alpha=0.7, c=range(len(score_types)), 
                           cmap='viridis')
        
        # Add labels
        for i, score_type in enumerate(score_types):
            ax.annotate(score_type, (coherence_values[i], predictive_powers[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=self.config.font_size-2)
        
        ax.set_xlabel('Coherence Score')
        ax.set_ylabel('Predictive Power')
        ax.set_title('Composite Scores: Coherence vs Predictive Power', 
                    fontsize=self.config.title_font_size)
        
        # Add quadrant lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.75, 0.75, 'High Coherence\nHigh Predictive', 
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        
        # Save chart
        output_path = output_dir / f"composite_scores_analysis.{self.config.output_format}"
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_validation_overview_chart(self,
                                        validation_results: Dict[str, ValidationResult],
                                        output_dir: Path) -> Path:
        """Create validation overview chart."""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Prepare data
        metrics = []
        values = []
        significance = []
        metric_types = []
        
        for key, result in validation_results.items():
            metrics.append(key.replace('_', ' ').title())
            values.append(result.value)
            significance.append(result.is_significant if result.is_significant is not None else False)
            metric_types.append(result.metric.value)
        
        # Chart 1: Validation Scores Overview
        ax1 = fig.add_subplot(gs[0, :])
        colors = ['green' if sig else 'orange' for sig in significance]
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_title('Validation Metrics Overview', fontsize=self.config.title_font_size)
        ax1.set_ylabel('Metric Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add significance indicators
        for bar, sig in zip(bars, significance):
            if sig:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        '✓', ha='center', va='bottom', fontsize=16, color='green')
        
        # Chart 2: Metric Type Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        metric_type_counts = pd.Series(metric_types).value_counts()
        ax2.pie(metric_type_counts.values, labels=metric_type_counts.index, autopct='%1.1f%%')
        ax2.set_title('Validation Metric Types')
        
        # Chart 3: Significance Analysis
        ax3 = fig.add_subplot(gs[1, 1])
        sig_counts = pd.Series(significance).value_counts()
        colors_pie = ['green', 'orange']
        ax3.pie(sig_counts.values, labels=['Significant', 'Not Significant'], 
               colors=colors_pie, autopct='%1.1f%%')
        ax3.set_title('Statistical Significance')
        
        plt.tight_layout()
        
        # Save chart
        output_path = output_dir / f"validation_overview.{self.config.output_format}"
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_convergence_chart(self,
                                result: OptimizationResult,
                                method_name: str,
                                output_dir: Path) -> Path:
        """Create optimization convergence chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size)
        
        # Extract optimization history
        iterations = [entry['iteration'] for entry in result.optimization_history]
        scores = [entry['score'] for entry in result.optimization_history]
        
        # Chart 1: Convergence over iterations
        ax1.plot(iterations, scores, 'b-', alpha=0.7, linewidth=2)
        ax1.scatter(iterations, scores, c=scores, cmap='viridis', s=30, alpha=0.8)
        ax1.set_title(f'{method_name} Optimization Convergence', 
                     fontsize=self.config.title_font_size)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Score')
        ax1.grid(True, alpha=0.3)
        
        # Add best score line
        best_score = result.best_score
        ax1.axhline(y=best_score, color='red', linestyle='--', 
                   label=f'Best Score: {best_score:.4f}')
        ax1.legend()
        
        # Chart 2: Score distribution
        ax2.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(x=best_score, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Score Distribution')
        ax2.set_xlabel('Objective Score')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save chart
        output_path = output_dir / f"convergence_{method_name}.{self.config.output_format}"
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_method_comparison_chart(self,
                                      optimization_results: Dict[str, OptimizationResult],
                                      output_dir: Path) -> Path:
        """Create optimization method comparison chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        methods = list(optimization_results.keys())
        best_scores = [result.best_score for result in optimization_results.values()]
        iterations = [result.convergence_info.get('iterations', 0) 
                     for result in optimization_results.values()]
        
        # Chart 1: Best Scores Comparison
        bars1 = ax1.bar(methods, best_scores, alpha=0.7, color='steelblue')
        ax1.set_title('Best Scores by Optimization Method', fontsize=self.config.title_font_size)
        ax1.set_ylabel('Best Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars1, best_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # Chart 2: Iterations Comparison
        bars2 = ax2.bar(methods, iterations, alpha=0.7, color='forestgreen')
        ax2.set_title('Iterations by Method', fontsize=self.config.title_font_size)
        ax2.set_ylabel('Number of Iterations')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, iter_count in zip(bars2, iterations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{iter_count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save chart
        output_path = output_dir / f"method_comparison.{self.config.output_format}"
        plt.savefig(output_path, dpi=self.config.output_dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # Interactive dashboard creation methods
    def _create_heuristic_analysis_dashboard(self,
                                           analysis_results: Dict[str, HeuristicAnalysisResult],
                                           output_dir: Path) -> Path:
        """Create interactive heuristic analysis dashboard."""
        if not PLOTLY_AVAILABLE:
            return self._create_static_heuristic_dashboard(analysis_results, output_dir)
        
        # Create plotly figures
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=['Target Effectiveness', 'Quality Scoring', 
                          'Composite Scores', 'Analysis Summary'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # Add charts to subplots
        # (Implementation would add specific plotly charts here)
        
        # Save as HTML
        output_path = output_dir / "heuristic_analysis_dashboard.html"
        fig.write_html(str(output_path))
        
        return output_path
    
    def _generate_dashboard_html(self,
                               heuristic_results: Optional[Dict[str, HeuristicAnalysisResult]],
                               validation_results: Optional[Dict[str, ValidationResult]],
                               optimization_results: Optional[Dict[str, OptimizationResult]]) -> str:
        """Generate static HTML dashboard."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Horizon Profit Labeling Research Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                .significant {{ background: #d4edda; }}
                .not-significant {{ background: #f8d7da; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Multi-Horizon Profit Labeling Research Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            {self._generate_heuristic_html_section(heuristic_results) if heuristic_results else ''}
            {self._generate_validation_html_section(validation_results) if validation_results else ''}
            {self._generate_optimization_html_section(optimization_results) if optimization_results else ''}
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_heuristic_html_section(self, results: Dict[str, HeuristicAnalysisResult]) -> str:
        """Generate HTML section for heuristic results."""
        section_html = '<div class="section"><h2>Heuristic Analysis Results</h2>'
        
        for key, result in results.items():
            section_html += f'''
            <div class="metric">
                <h3>{key.replace('_', ' ').title()}</h3>
                <p><strong>Value:</strong> {result.metric_value:.4f}</p>
                <p><strong>Interpretation:</strong> {result.interpretation}</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in result.recommendations)}
                </ul>
            </div>
            '''
        
        section_html += '</div>'
        return section_html
    
    def _generate_validation_html_section(self, results: Dict[str, ValidationResult]) -> str:
        """Generate HTML section for validation results."""
        section_html = '<div class="section"><h2>Validation Results</h2>'
        
        for key, result in results.items():
            significance_class = "significant" if result.is_significant else "not-significant"
            section_html += f'''
            <div class="metric {significance_class}">
                <h3>{key.replace('_', ' ').title()}</h3>
                <p><strong>Value:</strong> {result.value:.4f}</p>
                <p><strong>Significant:</strong> {'Yes' if result.is_significant else 'No'}</p>
                <p><strong>Interpretation:</strong> {result.interpretation}</p>
            </div>
            '''
        
        section_html += '</div>'
        return section_html
    
    def _generate_optimization_html_section(self, results: Dict[str, OptimizationResult]) -> str:
        """Generate HTML section for optimization results."""
        section_html = '<div class="section"><h2>Optimization Results</h2>'
        
        # Best result summary
        best_result = max(results.values(), key=lambda x: x.best_score)
        section_html += f'''
        <div class="metric significant">
            <h3>Best Configuration</h3>
            <p><strong>Method:</strong> {best_result.method.value}</p>
            <p><strong>Score:</strong> {best_result.best_score:.4f}</p>
            <p><strong>Parameters:</strong></p>
            <ul>
                {''.join(f'<li>{k}: {v}</li>' for k, v in best_result.best_params.items())}
            </ul>
        </div>
        '''
        
        section_html += '</div>'
        return section_html
    
    # Dashboard tab content creators
    def _create_heuristic_tab_content(self, heuristic_results):
        """Create heuristic analysis tab content."""
        if not heuristic_results:
            return html.Div([html.H3('No heuristic analysis results available')])
        
        # Create content based on results
        return html.Div([
            html.H3('Heuristic Analysis Results'),
            html.P(f'Analyzed {len(heuristic_results)} heuristic components'),
            # Add more interactive content here
        ])
    
    def _create_validation_tab_content(self, validation_results):
        """Create validation results tab content."""
        if not validation_results:
            return html.Div([html.H3('No validation results available')])
        
        return html.Div([
            html.H3('Validation Results'),
            html.P(f'Validated {len(validation_results)} components'),
            # Add more interactive content here
        ])
    
    def _create_optimization_tab_content(self, optimization_results):
        """Create optimization results tab content."""
        if not optimization_results:
            return html.Div([html.H3('No optimization results available')])
        
        return html.Div([
            html.H3('Optimization Results'),
            html.P(f'Completed {len(optimization_results)} optimization runs'),
            # Add more interactive content here
        ])
    
    def _create_data_explorer_content(self, market_data, labeled_data):
        """Create data explorer tab content."""
        if market_data is None and labeled_data is None:
            return html.Div([html.H3('No data available for exploration')])
        
        return html.Div([
            html.H3('Data Explorer'),
            html.P('Interactive data exploration tools would go here'),
            # Add interactive data exploration components
        ])
    
    def save_all_visualizations(self, output_dir: Union[str, Path]):
        """Save all generated visualizations to specified directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for chart_name, chart_path in self.generated_charts.items():
            try:
                # Copy to output directory if not already there
                if chart_path.parent != output_dir:
                    new_path = output_dir / chart_path.name
                    new_path.write_bytes(chart_path.read_bytes())
                    saved_count += 1
            except Exception as e:
                self.logger.warning(f'Failed to save {chart_name}: {e}')
        
        self.logger.info(f'💾 Saved {saved_count} visualizations to {output_dir}')
    
    def generate_visualization_report(self) -> str:
        """Generate report of all created visualizations."""
        if not self.generated_charts:
            return "No visualizations have been generated yet."
        
        report_lines = [
            "# Multi-Horizon Profit Labeling Visualization Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Summary",
            f"Total visualizations created: {len(self.generated_charts)}",
            ""
        ]
        
        # Group by type
        chart_types = {}
        for chart_name, chart_path in self.generated_charts.items():
            chart_type = chart_name.split('_')[0]
            if chart_type not in chart_types:
                chart_types[chart_type] = []
            chart_types[chart_type].append((chart_name, chart_path))
        
        for chart_type, charts in chart_types.items():
            report_lines.extend([
                f"### {chart_type.title()} Visualizations",
                ""
            ])
            
            for chart_name, chart_path in charts:
                report_lines.append(f"- **{chart_name}**: `{chart_path}`")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


# Convenience functions
def create_profit_labeling_visualizations(heuristic_results: Optional[Dict[str, HeuristicAnalysisResult]] = None,
                                        validation_results: Optional[Dict[str, ValidationResult]] = None,
                                        optimization_results: Optional[Dict[str, OptimizationResult]] = None,
                                        output_dir: Union[str, Path] = "visualizations",
                                        config: Optional[VisualizationConfig] = None) -> Dict[str, Path]:
    """Convenience function to create all profit labeling visualizations."""
    visualizer = LabelingVisualizer(config)
    
    all_charts = {}
    
    if heuristic_results:
        charts = visualizer.visualize_heuristic_analysis(heuristic_results, output_dir)
        all_charts.update(charts)
    
    if validation_results:
        charts = visualizer.visualize_validation_results(validation_results, output_dir)
        all_charts.update(charts)
    
    if optimization_results:
        charts = visualizer.visualize_optimization_results(optimization_results, output_dir)
        all_charts.update(charts)
    
    return all_charts
