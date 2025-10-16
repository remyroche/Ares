"""
Results Visualization and Reporting Dashboard

This module provides comprehensive visualization and reporting capabilities
for A/B/C testing results with interactive dashboards and automated reports.

Key Features:
- Interactive performance dashboards
- Statistical analysis visualizations
- Automated report generation
- Export capabilities (PDF, HTML, Excel)
- Real-time monitoring dashboards
- Custom chart configurations
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import base64
from io import BytesIO

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Report generation
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose, validate_data_quality,
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)

class ChartType(Enum):
    """Types of charts available."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CORRELATION = "correlation"
    CANDLESTICK = "candlestick"
    AREA = "area"
    PIE = "pie"
    RADAR = "radar"

class ReportFormat(Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"

@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    chart_type: ChartType
    title: str
    x_label: str = ""
    y_label: str = ""
    width: int = 800
    height: int = 600
    theme: str = "default"
    colors: List[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    show_legend: bool = True
    show_grid: bool = True
    interactive: bool = True

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str
    subtitle: str = ""
    author: str = "A/B/C Testing Framework"
    output_format: ReportFormat = ReportFormat.HTML
    include_charts: bool = True
    include_statistics: bool = True
    include_recommendations: bool = True
    template_path: Optional[str] = None
    output_path: str = "reports"

class ResultsVisualizer:
    """Comprehensive results visualization system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize results visualizer."""
        self.config = config or {}
        self.logger = logger.getChild('ResultsVisualizer')

        # Check available libraries
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        self.seaborn_available = SEABORN_AVAILABLE
        self.plotly_available = PLOTLY_AVAILABLE
        self.jinja2_available = JINJA2_AVAILABLE
        self.weasyprint_available = WEASYPRINT_AVAILABLE

        # Set default style
        if self.matplotlib_available:
            plt.style.use('default')

        if self.seaborn_available:
            sns.set_style("whitegrid")

        self.logger.info("🚀 ResultsVisualizer initialized")
        self.logger.info(f"📊 Matplotlib available: {self.matplotlib_available}")
        self.logger.info(f"📊 Seaborn available: {self.seaborn_available}")
        self.logger.info(f"📊 Plotly available: {self.plotly_available}")
        self.logger.info(f"📊 Jinja2 available: {self.jinja2_available}")
        self.logger.info(f"📊 WeasyPrint available: {self.weasyprint_available}")

    @traced(span_name='create_performance_dashboard')
    async def create_performance_dashboard(self, model_results: List[Dict[str, Any]],
                                         output_path: str = "dashboard.html") -> str:
        """Create interactive performance dashboard."""
        self.logger.info("📊 Creating performance dashboard...")

        if not self.plotly_available:
            self.logger.warning("⚠️ Plotly not available, creating static dashboard")
            return await self._create_static_dashboard(model_results, output_path)

        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Portfolio Performance', 'Risk Metrics',
                              'Trade Analysis', 'Drawdown Analysis',
                              'Performance Comparison', 'Correlation Matrix'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # 1. Portfolio Performance
            await self._add_portfolio_performance_chart(fig, model_results, row=1, col=1)

            # 2. Risk Metrics
            await self._add_risk_metrics_chart(fig, model_results, row=1, col=2)

            # 3. Trade Analysis
            await self._add_trade_analysis_chart(fig, model_results, row=2, col=1)

            # 4. Drawdown Analysis
            await self._add_drawdown_analysis_chart(fig, model_results, row=2, col=2)

            # 5. Performance Comparison
            await self._add_performance_comparison_chart(fig, model_results, row=3, col=1)

            # 6. Correlation Matrix
            await self._add_correlation_matrix_chart(fig, model_results, row=3, col=2)

            # Update layout
            fig.update_layout(
                title="A/B/C Testing Performance Dashboard",
                height=1200,
                showlegend=True,
                template="plotly_white"
            )

            # Save dashboard
            ensure_directory(Path(output_path).parent)
            fig.write_html(output_path)

            self.logger.info(f"✅ Performance dashboard created: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"❌ Error creating performance dashboard: {e}")
            return await self._create_static_dashboard(model_results, output_path)

    async def _add_portfolio_performance_chart(self, fig, model_results: List[Dict[str, Any]],
                                             row: int, col: int) -> None:
        """Add portfolio performance chart."""
        try:
            for i, result in enumerate(model_results):
                if 'equity_curve' in result and not result['equity_curve'].empty:
                    equity_curve = result['equity_curve']

                    fig.add_trace(
                        go.Scatter(
                            x=equity_curve.index,
                            y=equity_curve['portfolio_value'],
                            mode='lines',
                            name=f"{result['model_name']} Portfolio Value",
                            line=dict(width=2)
                        ),
                        row=row, col=col
                    )

            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Portfolio Value ($)", row=row, col=col)

        except Exception as e:
            self.logger.warning(f"⚠️ Could not add portfolio performance chart: {e}")

    async def _add_risk_metrics_chart(self, fig, model_results: List[Dict[str, Any]],
                                    row: int, col: int) -> None:
        """Add risk metrics chart."""
        try:
            models = [r['model_name'] for r in model_results]
            sharpe_ratios = [r.get('sharpe_ratio', 0) for r in model_results]
            max_drawdowns = [abs(r.get('max_drawdown', 0)) for r in model_results]
            volatilities = [r.get('volatility', 0) for r in model_results]

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=sharpe_ratios,
                    name="Sharpe Ratio",
                    marker_color='lightblue'
                ),
                row=row, col=col
            )

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=max_drawdowns,
                    name="Max Drawdown",
                    marker_color='lightcoral'
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text="Models", row=row, col=col)
            fig.update_yaxes(title_text="Risk Metrics", row=row, col=col)

        except Exception as e:
            self.logger.warning(f"⚠️ Could not add risk metrics chart: {e}")

    async def _add_trade_analysis_chart(self, fig, model_results: List[Dict[str, Any]],
                                      row: int, col: int) -> None:
        """Add trade analysis chart."""
        try:
            models = [r['model_name'] for r in model_results]
            total_trades = [r.get('total_trades', 0) for r in model_results]
            win_rates = [r.get('win_rate', 0) for r in model_results]

            fig.add_trace(
                go.Scatter(
                    x=total_trades,
                    y=win_rates,
                    mode='markers+text',
                    text=models,
                    textposition="top center",
                    name="Trade Performance",
                    marker=dict(size=10, color='green')
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text="Total Trades", row=row, col=col)
            fig.update_yaxes(title_text="Win Rate", row=row, col=col)

        except Exception as e:
            self.logger.warning(f"⚠️ Could not add trade analysis chart: {e}")

    async def _add_drawdown_analysis_chart(self, fig, model_results: List[Dict[str, Any]],
                                         row: int, col: int) -> None:
        """Add drawdown analysis chart."""
        try:
            for i, result in enumerate(model_results):
                if 'equity_curve' in result and not result['equity_curve'].empty:
                    equity_curve = result['equity_curve']

                    # Calculate drawdown
                    peak = equity_curve['portfolio_value'].expanding().max()
                    drawdown = (equity_curve['portfolio_value'] - peak) / peak * 100

                    fig.add_trace(
                        go.Scatter(
                            x=equity_curve.index,
                            y=drawdown,
                            mode='lines',
                            name=f"{result['model_name']} Drawdown",
                            fill='tonexty' if i > 0 else 'tozeroy',
                            line=dict(width=1)
                        ),
                        row=row, col=col
                    )

            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col)

        except Exception as e:
            self.logger.warning(f"⚠️ Could not add drawdown analysis chart: {e}")

    async def _add_performance_comparison_chart(self, fig, model_results: List[Dict[str, Any]],
                                              row: int, col: int) -> None:
        """Add performance comparison chart."""
        try:
            models = [r['model_name'] for r in model_results]
            total_returns = [r.get('total_return', 0) * 100 for r in model_results]
            annualized_returns = [r.get('annualized_return', 0) * 100 for r in model_results]

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=total_returns,
                    name="Total Return (%)",
                    marker_color='lightgreen'
                ),
                row=row, col=col
            )

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=annualized_returns,
                    name="Annualized Return (%)",
                    marker_color='lightblue'
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text="Models", row=row, col=col)
            fig.update_yaxes(title_text="Returns (%)", row=row, col=col)

        except Exception as e:
            self.logger.warning(f"⚠️ Could not add performance comparison chart: {e}")

    async def _add_correlation_matrix_chart(self, fig, model_results: List[Dict[str, Any]],
                                          row: int, col: int) -> None:
        """Add correlation matrix chart."""
        try:
            # Extract daily returns for correlation analysis
            returns_data = {}
            for result in model_results:
                if 'daily_returns' in result and not result['daily_returns'].empty:
                    returns_data[result['model_name']] = result['daily_returns']

            if len(returns_data) > 1:
                # Create correlation matrix
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()

                fig.add_trace(
                    go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=correlation_matrix.round(3).values,
                        texttemplate="%{text}",
                        textfont={"size": 10}
                    ),
                    row=row, col=col
                )

            fig.update_xaxes(title_text="Models", row=row, col=col)
            fig.update_yaxes(title_text="Models", row=row, col=col)

        except Exception as e:
            self.logger.warning(f"⚠️ Could not add correlation matrix chart: {e}")

    async def _create_static_dashboard(self, model_results: List[Dict[str, Any]],
                                     output_path: str) -> str:
        """Create static dashboard using matplotlib."""
        if not self.matplotlib_available:
            self.logger.error("❌ No visualization libraries available")
            return ""

        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('A/B/C Testing Performance Dashboard', fontsize=16)

            # 1. Portfolio Performance
            ax1 = axes[0, 0]
            for result in model_results:
                if 'equity_curve' in result and not result['equity_curve'].empty:
                    equity_curve = result['equity_curve']
                    ax1.plot(equity_curve.index, equity_curve['portfolio_value'],
                            label=result['model_name'], linewidth=2)
            ax1.set_title('Portfolio Performance')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)

            # 2. Risk Metrics
            ax2 = axes[0, 1]
            models = [r['model_name'] for r in model_results]
            sharpe_ratios = [r.get('sharpe_ratio', 0) for r in model_results]
            ax2.bar(models, sharpe_ratios, color='lightblue')
            ax2.set_title('Sharpe Ratios')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.tick_params(axis='x', rotation=45)

            # 3. Performance Comparison
            ax3 = axes[0, 2]
            total_returns = [r.get('total_return', 0) * 100 for r in model_results]
            ax3.bar(models, total_returns, color='lightgreen')
            ax3.set_title('Total Returns')
            ax3.set_ylabel('Return (%)')
            ax3.tick_params(axis='x', rotation=45)

            # 4. Trade Analysis
            ax4 = axes[1, 0]
            total_trades = [r.get('total_trades', 0) for r in model_results]
            win_rates = [r.get('win_rate', 0) for r in model_results]
            ax4.scatter(total_trades, win_rates, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax4.annotate(model, (total_trades[i], win_rates[i]),
                           xytext=(5, 5), textcoords='offset points')
            ax4.set_title('Trade Analysis')
            ax4.set_xlabel('Total Trades')
            ax4.set_ylabel('Win Rate')
            ax4.grid(True)

            # 5. Drawdown Analysis
            ax5 = axes[1, 1]
            max_drawdowns = [abs(r.get('max_drawdown', 0)) * 100 for r in model_results]
            ax5.bar(models, max_drawdowns, color='lightcoral')
            ax5.set_title('Maximum Drawdowns')
            ax5.set_ylabel('Drawdown (%)')
            ax5.tick_params(axis='x', rotation=45)

            # 6. Volatility
            ax6 = axes[1, 2]
            volatilities = [r.get('volatility', 0) * 100 for r in model_results]
            ax6.bar(models, volatilities, color='orange')
            ax6.set_title('Volatility')
            ax6.set_ylabel('Volatility (%)')
            ax6.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Save dashboard
            ensure_directory(Path(output_path).parent)
            plt.savefig(output_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✅ Static dashboard created: {output_path.replace('.html', '.png')}")
            return output_path.replace('.html', '.png')

        except Exception as e:
            self.logger.error(f"❌ Error creating static dashboard: {e}")
            return ""

    @traced(span_name='generate_comprehensive_report')
    async def generate_comprehensive_report(self, abc_results: Dict[str, Any],
                                          config: ReportConfig) -> str:
        """Generate comprehensive A/B/C testing report."""
        self.logger.info("📄 Generating comprehensive report...")

        try:
            # Prepare report data
            report_data = await self._prepare_report_data(abc_results)

            # Generate charts
            charts = await self._generate_report_charts(abc_results, config)

            # Create report content
            if config.output_format == ReportFormat.HTML:
                report_path = await self._generate_html_report(report_data, charts, config)
            elif config.output_format == ReportFormat.PDF:
                report_path = await self._generate_pdf_report(report_data, charts, config)
            elif config.output_format == ReportFormat.EXCEL:
                report_path = await self._generate_excel_report(report_data, config)
            else:
                report_path = await self._generate_json_report(report_data, config)

            self.logger.info(f"✅ Comprehensive report generated: {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"❌ Error generating comprehensive report: {e}")
            raise

    async def _prepare_report_data(self, abc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for report generation."""
        return {
            'test_info': {
                'test_name': abc_results.get('test_name', 'A/B/C Test'),
                'test_description': abc_results.get('test_description', ''),
                'symbol': abc_results.get('symbol', ''),
                'exchange': abc_results.get('exchange', ''),
                'timeframe': abc_results.get('timeframe', ''),
                'start_time': abc_results.get('start_time', ''),
                'end_time': abc_results.get('end_time', ''),
                'total_duration': abc_results.get('total_duration', 0)
            },
            'model_results': abc_results.get('model_results', []),
            'statistical_analysis': abc_results.get('statistical_tests', {}),
            'performance_ranking': abc_results.get('performance_ranking', []),
            'recommendations': abc_results.get('recommendations', []),
            'best_performing_model': abc_results.get('best_performing_model'),
            'most_robust_model': abc_results.get('most_robust_model'),
            'generated_at': datetime.now().isoformat()
        }

    async def _generate_report_charts(self, abc_results: Dict[str, Any],
                                    config: ReportConfig) -> Dict[str, str]:
        """Generate charts for the report."""
        charts = {}

        if not config.include_charts:
            return charts

        try:
            # Create temporary dashboard
            temp_dashboard = await self.create_performance_dashboard(
                abc_results.get('model_results', []),
                "temp_dashboard.html"
            )

            if temp_dashboard:
                charts['dashboard'] = temp_dashboard

        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate report charts: {e}")

        return charts

    async def _generate_html_report(self, report_data: Dict[str, Any],
                                  charts: Dict[str, str], config: ReportConfig) -> str:
        """Generate HTML report."""
        try:
            # HTML template
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { text-align: center; margin-bottom: 40px; }
                    .section { margin: 30px 0; }
                    .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }
                    .recommendation { background-color: #f0f8ff; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ title }}</h1>
                    <h2>{{ subtitle }}</h2>
                    <p>Generated on {{ generated_at }}</p>
                </div>

                <div class="section">
                    <h2>Test Information</h2>
                    <p><strong>Test Name:</strong> {{ test_info.test_name }}</p>
                    <p><strong>Description:</strong> {{ test_info.test_description }}</p>
                    <p><strong>Symbol:</strong> {{ test_info.symbol }}</p>
                    <p><strong>Exchange:</strong> {{ test_info.exchange }}</p>
                    <p><strong>Timeframe:</strong> {{ test_info.timeframe }}</p>
                    <p><strong>Duration:</strong> {{ test_info.total_duration }} seconds</p>
                </div>

                <div class="section">
                    <h2>Model Performance Summary</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Total Return</th>
                            <th>Sharpe Ratio</th>
                            <th>Max Drawdown</th>
                            <th>Win Rate</th>
                            <th>Total Trades</th>
                        </tr>
                        {% for result in model_results %}
                        <tr>
                            <td>{{ result.model_name }}</td>
                            <td>{{ "%.2f"|format(result.total_return * 100) }}%</td>
                            <td>{{ "%.2f"|format(result.sharpe_ratio) }}</td>
                            <td>{{ "%.2f"|format(result.max_drawdown * 100) }}%</td>
                            <td>{{ "%.2f"|format(result.win_rate * 100) }}%</td>
                            <td>{{ result.total_trades }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>

                {% if recommendations %}
                <div class="section">
                    <h2>Recommendations</h2>
                    {% for rec in recommendations %}
                    <div class="recommendation">
                        <h3>{{ rec.title }}</h3>
                        <p><strong>Priority:</strong> {{ rec.priority }}</p>
                        <p><strong>Description:</strong> {{ rec.description }}</p>
                        <p><strong>Action:</strong> {{ rec.action }}</p>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}

                <div class="section">
                    <h2>Key Findings</h2>
                    {% if best_performing_model %}
                    <p><strong>Best Performing Model:</strong> {{ best_performing_model }}</p>
                    {% endif %}
                    {% if most_robust_model %}
                    <p><strong>Most Robust Model:</strong> {{ most_robust_model }}</p>
                    {% endif %}
                </div>
            </body>
            </html>
            """

            # Render template
            if self.jinja2_available:
                template = Template(html_template)
                html_content = template.render(**report_data)
            else:
                # Simple string replacement
                html_content = html_template
                for key, value in report_data.items():
                    html_content = html_content.replace(f"{{ {key} }}", str(value))

            # Save report
            output_path = Path(config.output_path) / f"{config.title.replace(' ', '_')}_report.html"
            ensure_directory(output_path.parent)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            return str(output_path)

        except Exception as e:
            self.logger.error(f"❌ Error generating HTML report: {e}")
            raise

    async def _generate_pdf_report(self, report_data: Dict[str, Any],
                                 charts: Dict[str, str], config: ReportConfig) -> str:
        """Generate PDF report."""
        if not self.weasyprint_available:
            self.logger.warning("⚠️ WeasyPrint not available, generating HTML instead")
            return await self._generate_html_report(report_data, charts, config)

        try:
            # Generate HTML first
            html_path = await self._generate_html_report(report_data, charts, config)

            # Convert to PDF
            pdf_path = html_path.replace('.html', '.pdf')

            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            weasyprint.HTML(string=html_content).write_pdf(pdf_path)

            return pdf_path

        except Exception as e:
            self.logger.error(f"❌ Error generating PDF report: {e}")
            raise

    async def _generate_excel_report(self, report_data: Dict[str, Any],
                                   config: ReportConfig) -> str:
        """Generate Excel report."""
        try:
            output_path = Path(config.output_path) / f"{config.title.replace(' ', '_')}_report.xlsx"
            ensure_directory(output_path.parent)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Model performance summary
                if report_data['model_results']:
                    df_performance = pd.DataFrame(report_data['model_results'])
                    df_performance.to_excel(writer, sheet_name='Performance Summary', index=False)

                # Recommendations
                if report_data['recommendations']:
                    df_recommendations = pd.DataFrame(report_data['recommendations'])
                    df_recommendations.to_excel(writer, sheet_name='Recommendations', index=False)

                # Test information
                test_info_df = pd.DataFrame([report_data['test_info']])
                test_info_df.to_excel(writer, sheet_name='Test Information', index=False)

            return str(output_path)

        except Exception as e:
            self.logger.error(f"❌ Error generating Excel report: {e}")
            raise

    async def _generate_json_report(self, report_data: Dict[str, Any],
                                  config: ReportConfig) -> str:
        """Generate JSON report."""
        try:
            output_path = Path(config.output_path) / f"{config.title.replace(' ', '_')}_report.json"
            ensure_directory(output_path.parent)

            await safe_json_dump(output_path, report_data, indent=2)

            return str(output_path)

        except Exception as e:
            self.logger.error(f"❌ Error generating JSON report: {e}")
            raise

# Convenience function for easy integration
async def create_results_visualizer(config: Optional[Dict[str, Any]] = None) -> ResultsVisualizer:
    """Create a results visualizer instance."""
    return ResultsVisualizer(config)
