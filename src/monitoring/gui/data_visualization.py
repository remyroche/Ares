"""
Data Visualization Components for Monitoring Dashboard

Provides charts and visualizations for monitoring data including
trade performance, regime analysis, and daily summaries.
"""
import warnings
import logging
import time
import typing
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import tk, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ...utils.logger import system_logger

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

class MonitoringVisualization:
    """
    Data visualization components for monitoring dashboard.
    """

    def __init__(self, parent_frame: tk.Widget) -> None:
        """Initialize the visualization component."""
        self.parent_frame = parent_frame
        self.logger = system_logger.getChild('MonitoringVisualization')
        self.trade_data: Optional[pd.DataFrame] = None
        self.daily_summary_data: Optional[pd.DataFrame] = None
        plt.style.use('seaborn-v0_8')
        self.fig = Figure(figsize=(12, 8), dpi = 100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.parent_frame)
        self.canvas.get_tk_widget().pack(fill = tk.BOTH, expand = True)
        self.logger.info('Monitoring visualization initialized')

    def set_trade_data(self, data: pd.DataFrame) -> None:
        """Set trade data for visualization."""
        self.trade_data = data
        self.logger.info(f'Set trade data: {len(data)} records')

    def set_daily_summary_data(self, data: pd.DataFrame) -> None:
        """Set daily summary data for visualization."""
        self.daily_summary_data = data
        self.logger.info(f'Set daily summary data: {len(data)} records')

    def plot_trade_performance(self) -> None:
        """Plot trade performance over time."""
        if self.trade_data is None:
            self._show_no_data_message()
            return
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, hspace = 0.3, wspace = 0.3)
        ax1 = self.fig.add_subplot(gs[0, 0])
        if 'timestamp' in self.trade_data.columns:
            self.trade_data['date'] = pd.to_datetime(self.trade_data['timestamp']).dt.date
            daily_trades = self.trade_data.groupby('date').size()
            ax1.plot(daily_trades.index, daily_trades.values, marker='o', linewidth = 2)
            ax1.set_title('Daily Trade Count')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Trades')
            ax1.tick_params(axis='x', rotation = 45)
        ax2 = self.fig.add_subplot(gs[0, 1])
        if 'action' in self.trade_data.columns:
            action_counts = self.trade_data['action'].value_counts()
            colors = ['green' if action == 'buy' else 'red' if action == 'sell' else 'gray' for action in action_counts.index]
            ax2.pie(action_counts.values, labels = action_counts.index, autopct='%1.1f%%', colors = colors, startangle = 90)
            ax2.set_title('Trade Action Distribution')
        ax3 = self.fig.add_subplot(gs[1, 0])
        if 'overall_confidence' in self.trade_data.columns:
            ax3.hist(self.trade_data['overall_confidence'], bins = 20, alpha = 0.7, color='blue', edgecolor='black')
            ax3.set_title('Confidence Distribution')
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('Frequency')
            ax3.axvline(self.trade_data['overall_confidence'].mean(), color='red', linestyle='--', label = f"Mean: {self.trade_data['overall_confidence'].mean():.3f}")
            ax3.legend()
        ax4 = self.fig.add_subplot(gs[1, 1])
        if 'overall_risk_score' in self.trade_data.columns:
            ax4.hist(self.trade_data['overall_risk_score'], bins = 20, alpha = 0.7, color='orange', edgecolor='black')
            ax4.set_title('Risk Score Distribution')
            ax4.set_xlabel('Risk Score')
            ax4.set_ylabel('Frequency')
            ax4.axvline(self.trade_data['overall_risk_score'].mean(), color='red', linestyle='--', label = f"Mean: {self.trade_data['overall_risk_score'].mean():.3f}")
            ax4.legend()
        self.canvas.draw()

    def plot_regime_analysis(self) -> None:
        """Plot HMM regime analysis."""
        if self.trade_data is None:
            self._show_no_data_message()
            return
        self.fig.clear()
        if 'hmm_regime_id' not in self.trade_data.columns:
            self._show_no_regime_data_message()
            return
        gs = self.fig.add_gridspec(2, 2, hspace = 0.3, wspace = 0.3)
        ax1 = self.fig.add_subplot(gs[0, 0])
        regime_counts = self.trade_data['hmm_regime_id'].value_counts()
        bars = ax1.bar(regime_counts.index, regime_counts.values, color = plt.cm.Set3(np.linspace(0, 1, len(regime_counts))))
        ax1.set_title('Regime Distribution')
        ax1.set_xlabel('Regime ID')
        ax1.set_ylabel('Number of Trades')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom')
        ax2 = self.fig.add_subplot(gs[0, 1])
        if 'hmm_regime_probability' in self.trade_data.columns:
            regime_probs = self.trade_data.groupby('hmm_regime_id')['hmm_regime_probability'].mean()
            bars = ax2.bar(regime_probs.index, regime_probs.values, color = plt.cm.Set2(np.linspace(0, 1, len(regime_probs))))
            ax2.set_title('Average Regime Probability')
            ax2.set_xlabel('Regime ID')
            ax2.set_ylabel('Average Probability')
            ax2.set_ylim(0, 1)
        ax3 = self.fig.add_subplot(gs[1, 0])
        if 'hmm_regime_stability_score' in self.trade_data.columns:
            regime_stability = self.trade_data.groupby('hmm_regime_id')['hmm_regime_stability_score'].mean()
            bars = ax3.bar(regime_stability.index, regime_stability.values, color = plt.cm.viridis(np.linspace(0, 1, len(regime_stability))))
            ax3.set_title('Average Regime Stability')
            ax3.set_xlabel('Regime ID')
            ax3.set_ylabel('Average Stability Score')
            ax3.set_ylim(0, 1)
        ax4 = self.fig.add_subplot(gs[1, 1])
        if 'hmm_regime_duration' in self.trade_data.columns:
            regime_duration = self.trade_data.groupby('hmm_regime_id')['hmm_regime_duration'].mean()
            bars = ax4.bar(regime_duration.index, regime_duration.values, color = plt.cm.plasma(np.linspace(0, 1, len(regime_duration))))
            ax4.set_title('Average Regime Duration')
            ax4.set_xlabel('Regime ID')
            ax4.set_ylabel('Average Duration (periods)')
        self.canvas.draw()

    def plot_daily_summary(self) -> None:
        """Plot daily summary metrics."""
        if self.daily_summary_data is None:
            self._show_no_data_message()
            return
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, hspace = 0.3, wspace = 0.3)
        if 'date' in self.daily_summary_data.columns:
            self.daily_summary_data['date_parsed'] = pd.to_datetime(self.daily_summary_data['date'])
        ax1 = self.fig.add_subplot(gs[0, 0])
        if 'total_pnl' in self.daily_summary_data.columns:
            ax1.plot(self.daily_summary_data['date_parsed'], self.daily_summary_data['total_pnl'], marker='o', linewidth = 2, markersize = 4)
            ax1.axhline(y = 0, color='black', linestyle='--', alpha = 0.5)
            ax1.set_title('Daily PnL')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('PnL')
            ax1.tick_params(axis='x', rotation = 45)
            for i, pnl in enumerate(self.daily_summary_data['total_pnl']):
                color = 'green' if pnl > 0 else 'red'
                ax1.scatter(self.daily_summary_data['date_parsed'].iloc[i], pnl, color = color, s = 30, alpha = 0.7)
        ax2 = self.fig.add_subplot(gs[0, 1])
        if 'total_trades' in self.daily_summary_data.columns:
            ax2.bar(self.daily_summary_data['date_parsed'], self.daily_summary_data['total_trades'], color='skyblue', alpha = 0.7, edgecolor='black')
            ax2.set_title('Daily Trade Count')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Trades')
            ax2.tick_params(axis='x', rotation = 45)
        ax3 = self.fig.add_subplot(gs[1, 0])
        if 'win_rate' in self.daily_summary_data.columns:
            ax3.plot(self.daily_summary_data['date_parsed'], self.daily_summary_data['win_rate'], marker='s', linewidth = 2, color='purple', markersize = 4)
            ax3.set_title('Daily Win Rate')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation = 45)
            ax3.axhline(y = 0.5, color='red', linestyle='--', alpha = 0.5, label='50%')
            ax3.legend()
        ax4 = self.fig.add_subplot(gs[1, 1])
        if 'long_trades' in self.daily_summary_data.columns and 'short_trades' in self.daily_summary_data.columns:
            width = 0.35
            x = np.arange(len(self.daily_summary_data))
            ax4.bar(x - width / 2, self.daily_summary_data['long_trades'], width, label='Long Trades', color='green', alpha = 0.7)
            ax4.bar(x + width / 2, self.daily_summary_data['short_trades'], width, label='Short Trades', color='red', alpha = 0.7)
            ax4.set_title('Long vs Short Trades')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Number of Trades')
            ax4.set_xticks(x)
            ax4.set_xticklabels([d.strftime('%m-%d') for d in self.daily_summary_data['date_parsed']], rotation = 45)
            ax4.legend()
        self.canvas.draw()

    def plot_cumulative_performance(self) -> None:
        """Plot cumulative performance metrics."""
        if self.daily_summary_data is None:
            self._show_no_data_message()
            return
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, hspace = 0.3, wspace = 0.3)
        if 'date' in self.daily_summary_data.columns:
            self.daily_summary_data['date_parsed'] = pd.to_datetime(self.daily_summary_data['date'])
        ax1 = self.fig.add_subplot(gs[0, :])
        if 'total_pnl' in self.daily_summary_data.columns:
            cumulative_pnl = self.daily_summary_data['total_pnl'].cumsum()
            ax1.plot(self.daily_summary_data['date_parsed'], cumulative_pnl, linewidth = 3, color='blue', marker='o', markersize = 4)
            ax1.axhline(y = 0, color='black', linestyle='--', alpha = 0.5)
            ax1.set_title('Cumulative PnL Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative PnL')
            ax1.tick_params(axis='x', rotation = 45)
            ax1.grid(True, alpha = 0.3)
            final_pnl = cumulative_pnl.iloc[-1]
            ax1.annotate(f'Final: {final_pnl:.2f}', xy=(self.daily_summary_data['date_parsed'].iloc[-1], final_pnl), xytext=(10, 10), textcoords='offset points', bbox = dict(boxstyle='round,pad = 0.3', facecolor='yellow', alpha = 0.7), arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad = 0'))
        ax2 = self.fig.add_subplot(gs[1, 0])
        if 'total_trades' in self.daily_summary_data.columns:
            cumulative_trades = self.daily_summary_data['total_trades'].cumsum()
            ax2.plot(self.daily_summary_data['date_parsed'], cumulative_trades, linewidth = 2, color='green', marker='s', markersize = 3)
            ax2.set_title('Cumulative Trade Count')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Trades')
            ax2.tick_params(axis='x', rotation = 45)
            ax2.grid(True, alpha = 0.3)
        ax3 = self.fig.add_subplot(gs[1, 1])
        if 'win_rate' in self.daily_summary_data.columns:
            window_size = min(7, len(self.daily_summary_data))
            rolling_win_rate = self.daily_summary_data['win_rate'].rolling(window = window_size, min_periods = 1).mean()
            ax3.plot(self.daily_summary_data['date_parsed'], rolling_win_rate, linewidth = 2, color='purple', marker='o', markersize = 3)
            ax3.axhline(y = 0.5, color='red', linestyle='--', alpha = 0.5, label='50%')
            ax3.set_title(f'{window_size}-Day Rolling Win Rate')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation = 45)
            ax3.legend()
            ax3.grid(True, alpha = 0.3)
        self.canvas.draw()

    def plot_correlation_matrix(self) -> None:
        """Plot correlation matrix of numerical features."""
        if self.trade_data is None:
            self._show_no_data_message()
            return
        self.fig.clear()
        numerical_cols = self.trade_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            self._show_insufficient_data_message()
            return
        corr_matrix = self.trade_data[numerical_cols].corr()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax = 1)
        ax.set_xticks(range(len(numerical_cols)))
        ax.set_yticks(range(len(numerical_cols)))
        ax.set_xticklabels(numerical_cols, rotation = 45, ha='right')
        ax.set_yticklabels(numerical_cols)
        for i in range(len(numerical_cols)):
            for j in range(len(numerical_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize = 8)
        cbar = self.fig.colorbar(im, ax = ax)
        cbar.set_label('Correlation Coefficient')
        ax.set_title('Feature Correlation Matrix')
        self.canvas.draw()

    def _show_no_data_message(self) -> None:
        """Show message when no data is available."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No data available for visualization', ha='center', va='center', fontsize = 16, transform = ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def _show_no_regime_data_message(self) -> None:
        """Show message when no regime data is available."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No HMM regime data available', ha='center', va='center', fontsize = 16, transform = ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def _show_insufficient_data_message(self) -> None:
        """Show message when insufficient data is available."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Insufficient numerical data for correlation analysis', ha='center', va='center', fontsize = 16, transform = ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def clear_plot(self) -> None:
        """Clear the current plot."""
        self.fig.clear()
        self.canvas.draw()

class VisualizationControlPanel:
    """
    Control panel for visualization options.
    """

    def __init__(self, parent_frame: tk.Widget, visualization: MonitoringVisualization) -> None:
        """Initialize the control panel."""
        self.parent_frame = parent_frame
        self.visualization = visualization
        self.control_frame = ttk.LabelFrame(parent_frame, text='Visualization Controls')
        self.control_frame.pack(fill = tk.X, padx = 5, pady = 5)
        self._create_buttons()

    def _create_buttons(self) -> None:
        """Create control buttons."""
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill = tk.X, padx = 5, pady = 5)
        ttk.Button(button_frame, text='Trade Performance', command = self.visualization.plot_trade_performance).pack(side = tk.LEFT, padx = 2)
        ttk.Button(button_frame, text='Regime Analysis', command = self.visualization.plot_regime_analysis).pack(side = tk.LEFT, padx = 2)
        ttk.Button(button_frame, text='Daily Summary', command = self.visualization.plot_daily_summary).pack(side = tk.LEFT, padx = 2)
        ttk.Button(button_frame, text='Cumulative Performance', command = self.visualization.plot_cumulative_performance).pack(side = tk.LEFT, padx = 2)
        ttk.Button(button_frame, text='Correlation Matrix', command = self.visualization.plot_correlation_matrix).pack(side = tk.LEFT, padx = 2)
        ttk.Button(button_frame, text='Clear', command = self.visualization.clear_plot).pack(side = tk.LEFT, padx = 2)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            self.logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
