#!/usr/bin/env python3
"""
Data Visualization Components for Monitoring Dashboard

Provides charts and visualizations for monitoring data including
trade performance, regime analysis, and daily summaries.
"""


from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .utils.logger import system_logger


class MonitoringVisualization:
    """
    Data visualization components for monitoring dashboard.
    """
    
    def __init__(self, parent_frame: tk.Widget):
        """Initialize the visualization component."""
        self.parent_frame = parent_frame
        self.logger = system_logger.getChild("MonitoringVisualization")
        
        # Data storage
        self.trade_data: Optional[pd.DataFrame] = None
        self.daily_summary_data: Optional[pd.DataFrame] = None
        
        # Matplotlib setup
        plt.style.use('seaborn-v0_8')
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.logger.info("Monitoring visualization initialized")
    
    def set_trade_data(self, data: pd.DataFrame):
        """Set trade data for visualization."""
        self.trade_data = data
        self.logger.info(f"Set trade data: {len(data)} records")
    
    def set_daily_summary_data(self, data: pd.DataFrame):
        """Set daily summary data for visualization."""
        self.daily_summary_data = data
        self.logger.info(f"Set daily summary data: {len(data)} records")
    
    def plot_trade_performance(self):
        """Plot trade performance over time."""
        if self.trade_data is None:
            self._show_no_data_message()
            return
        
        self.fig.clear()
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Trade count over time
        ax1 = self.fig.add_subplot(gs[0, 0])
        if 'timestamp' in self.trade_data.columns:
            self.trade_data['date'] = pd.to_datetime(self.trade_data['timestamp']).dt.date
            daily_trades = self.trade_data.groupby('date').size()
            ax1.plot(daily_trades.index, daily_trades.values, marker='o', linewidth=2)
            ax1.set_title('Daily Trade Count')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Trades')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Action distribution
        ax2 = self.fig.add_subplot(gs[0, 1])
        if 'action' in self.trade_data.columns:
            action_counts = self.trade_data['action'].value_counts()
            colors = ['green' if action == 'buy' else 'red' if action == 'sell' else 'gray' 
                    for action in action_counts.index]
            ax2.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
            ax2.set_title('Trade Action Distribution')
        
        # Plot 3: Confidence distribution
        ax3 = self.fig.add_subplot(gs[1, 0])
        if 'overall_confidence' in self.trade_data.columns:
            ax3.hist(self.trade_data['overall_confidence'], bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax3.set_title('Confidence Distribution')
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('Frequency')
            ax3.axvline(self.trade_data['overall_confidence'].mean(), color='red', 
                    linestyle='--', label=f'Mean: {self.trade_data["overall_confidence"].mean():.3f}')
            ax3.legend()
        
        # Plot 4: Risk score distribution
        ax4 = self.fig.add_subplot(gs[1, 1])
        if 'overall_risk_score' in self.trade_data.columns:
            ax4.hist(self.trade_data['overall_risk_score'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_title('Risk Score Distribution')
            ax4.set_xlabel('Risk Score')
            ax4.set_ylabel('Frequency')
            ax4.axvline(self.trade_data['overall_risk_score'].mean(), color='red', 
                    linestyle='--', label=f'Mean: {self.trade_data["overall_risk_score"].mean():.3f}')
            ax4.legend()
        
        self.canvas.draw()
    
    def plot_regime_analysis(self):
        """Plot HMM regime analysis."""
        if self.trade_data is None:
            self._show_no_data_message()
            return
        
        self.fig.clear()
        
        # Check if regime data exists
        if 'hmm_regime_id' not in self.trade_data.columns:
            self._show_no_regime_data_message()
            return
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Regime distribution
        ax1 = self.fig.add_subplot(gs[0, 0])
        regime_counts = self.trade_data['hmm_regime_id'].value_counts()
        bars = ax1.bar(regime_counts.index, regime_counts.values, 
                    color=plt.cm.Set3(np.linspace(0, 1, len(regime_counts))))
        ax1.set_title('Regime Distribution')
        ax1.set_xlabel('Regime ID')
        ax1.set_ylabel('Number of Trades')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Plot 2: Regime probability distribution
        ax2 = self.fig.add_subplot(gs[0, 1])
        if 'hmm_regime_probability' in self.trade_data.columns:
            regime_probs = self.trade_data.groupby('hmm_regime_id')['hmm_regime_probability'].mean()
            bars = ax2.bar(regime_probs.index, regime_probs.values,
                        color=plt.cm.Set2(np.linspace(0, 1, len(regime_probs))))
            ax2.set_title('Average Regime Probability')
            ax2.set_xlabel('Regime ID')
            ax2.set_ylabel('Average Probability')
            ax2.set_ylim(0, 1)
        
        # Plot 3: Regime stability
        ax3 = self.fig.add_subplot(gs[1, 0])
        if 'hmm_regime_stability_score' in self.trade_data.columns:
            regime_stability = self.trade_data.groupby('hmm_regime_id')['hmm_regime_stability_score'].mean()
            bars = ax3.bar(regime_stability.index, regime_stability.values,
                        color=plt.cm.viridis(np.linspace(0, 1, len(regime_stability))))
            ax3.set_title('Average Regime Stability')
            ax3.set_xlabel('Regime ID')
            ax3.set_ylabel('Average Stability Score')
            ax3.set_ylim(0, 1)
        
        # Plot 4: Regime duration
        ax4 = self.fig.add_subplot(gs[1, 1])
        if 'hmm_regime_duration' in self.trade_data.columns:
            regime_duration = self.trade_data.groupby('hmm_regime_id')['hmm_regime_duration'].mean()
            bars = ax4.bar(regime_duration.index, regime_duration.values,
                        color=plt.cm.plasma(np.linspace(0, 1, len(regime_duration))))
            ax4.set_title('Average Regime Duration')
            ax4.set_xlabel('Regime ID')
            ax4.set_ylabel('Average Duration (periods)')
        
        self.canvas.draw()
    
    def plot_daily_summary(self):
        """Plot daily summary metrics."""
        if self.daily_summary_data is None:
            self._show_no_data_message()
            return
        
        self.fig.clear()
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Convert date column
        if 'date' in self.daily_summary_data.columns:
            self.daily_summary_data['date_parsed'] = pd.to_datetime(self.daily_summary_data['date'])
        
        # Plot 1: Daily PnL
        ax1 = self.fig.add_subplot(gs[0, 0])
        if 'total_pnl' in self.daily_summary_data.columns:
            ax1.plot(self.daily_summary_data['date_parsed'], self.daily_summary_data['total_pnl'], 
                    marker='o', linewidth=2, markersize=4)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax1.set_title('Daily PnL')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('PnL')
            ax1.tick_params(axis='x', rotation=45)
            
            # Color positive/negative values
            for i, pnl in enumerate(self.daily_summary_data['total_pnl']):
                color = 'green' if pnl > 0 else 'red'
                ax1.scatter(self.daily_summary_data['date_parsed'].iloc[i], pnl, 
                        color=color, s=30, alpha=0.7)
        
        # Plot 2: Daily Trade Count
        ax2 = self.fig.add_subplot(gs[0, 1])
        if 'total_trades' in self.daily_summary_data.columns:
            ax2.bar(self.daily_summary_data['date_parsed'], self.daily_summary_data['total_trades'],
                color='skyblue', alpha=0.7, edgecolor='black')
            ax2.set_title('Daily Trade Count')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Trades')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Win Rate Over Time
        ax3 = self.fig.add_subplot(gs[1, 0])
        if 'win_rate' in self.daily_summary_data.columns:
            ax3.plot(self.daily_summary_data['date_parsed'], self.daily_summary_data['win_rate'],
                    marker='s', linewidth=2, color='purple', markersize=4)
            ax3.set_title('Daily Win Rate')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
            ax3.legend()
        
        # Plot 4: Long vs Short Trades
        ax4 = self.fig.add_subplot(gs[1, 1])
        if 'long_trades' in self.daily_summary_data.columns and 'short_trades' in self.daily_summary_data.columns:
            width = 0.35
            x = np.arange(len(self.daily_summary_data))
            ax4.bar(x - width/2, self.daily_summary_data['long_trades'], width, 
                label='Long Trades', color='green', alpha=0.7)
            ax4.bar(x + width/2, self.daily_summary_data['short_trades'], width,
                label='Short Trades', color='red', alpha=0.7)
            ax4.set_title('Long vs Short Trades')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Number of Trades')
            ax4.set_xticks(x)
            ax4.set_xticklabels([d.strftime('%m-%d') for d in self.daily_summary_data['date_parsed']], 
                            rotation=45)
            ax4.legend()
        
        self.canvas.draw()
    
    def plot_cumulative_performance(self):
        """Plot cumulative performance metrics."""
        if self.daily_summary_data is None:
            self._show_no_data_message()
            return
        
        self.fig.clear()
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Convert date column
        if 'date' in self.daily_summary_data.columns:
            self.daily_summary_data['date_parsed'] = pd.to_datetime(self.daily_summary_data['date'])
        
        # Plot 1: Cumulative PnL
        ax1 = self.fig.add_subplot(gs[0, :])
        if 'total_pnl' in self.daily_summary_data.columns:
            cumulative_pnl = self.daily_summary_data['total_pnl'].cumsum()
            ax1.plot(self.daily_summary_data['date_parsed'], cumulative_pnl,
                    linewidth=3, color='blue', marker='o', markersize=4)
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax1.set_title('Cumulative PnL Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative PnL')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add final value annotation
            final_pnl = cumulative_pnl.iloc[-1]
            ax1.annotate(f'Final: {final_pnl:.2f}', 
                        xy=(self.daily_summary_data['date_parsed'].iloc[-1], final_pnl),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot 2: Cumulative Trades
        ax2 = self.fig.add_subplot(gs[1, 0])
        if 'total_trades' in self.daily_summary_data.columns:
            cumulative_trades = self.daily_summary_data['total_trades'].cumsum()
            ax2.plot(self.daily_summary_data['date_parsed'], cumulative_trades,
                    linewidth=2, color='green', marker='s', markersize=3)
            ax2.set_title('Cumulative Trade Count')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Trades')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Win Rate
        ax3 = self.fig.add_subplot(gs[1, 1])
        if 'win_rate' in self.daily_summary_data.columns:
            # Calculate rolling average win rate
            window_size = min(7, len(self.daily_summary_data))  # 7-day rolling average
            rolling_win_rate = self.daily_summary_data['win_rate'].rolling(window=window_size, min_periods=1).mean()
            
            ax3.plot(self.daily_summary_data['date_parsed'], rolling_win_rate,
                    linewidth=2, color='purple', marker='o', markersize=3)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%')
            ax3.set_title(f'{window_size}-Day Rolling Win Rate')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of numerical features."""
        if self.trade_data is None:
            self._show_no_data_message()
            return
        
        self.fig.clear()
        
        # Select numerical columns
        numerical_cols = self.trade_data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            self._show_insufficient_data_message()
            return
        
        # Calculate correlation matrix
        corr_matrix = self.trade_data[numerical_cols].corr()
        
        # Create heatmap
        ax = self.fig.add_subplot(111)
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(numerical_cols)))
        ax.set_yticks(range(len(numerical_cols)))
        ax.set_xticklabels(numerical_cols, rotation=45, ha='right')
        ax.set_yticklabels(numerical_cols)
        
        # Add correlation values
        for i in range(len(numerical_cols)):
            for j in range(len(numerical_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        ax.set_title('Feature Correlation Matrix')
        
        self.canvas.draw()
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No data available for visualization', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
    
    def _show_no_regime_data_message(self):
        """Show message when no regime data is available."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No HMM regime data available', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
    
    def _show_insufficient_data_message(self):
        """Show message when insufficient data is available."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Insufficient numerical data for correlation analysis', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
    
    def clear_plot(self):
        """Clear the current plot."""
        self.fig.clear()
        self.canvas.draw()


class VisualizationControlPanel:
    """
    Control panel for visualization options.
    """
    
    def __init__(self, parent_frame: tk.Widget, visualization: MonitoringVisualization):
        """Initialize the control panel."""
        self.parent_frame = parent_frame
        self.visualization = visualization
        
        # Create control frame
        self.control_frame = ttk.LabelFrame(parent_frame, text="Visualization Controls")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create buttons
        self._create_buttons()
    
    def _create_buttons(self):
        """Create control buttons."""
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Trade Performance", 
                command=self.visualization.plot_trade_performance).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Regime Analysis", 
                command=self.visualization.plot_regime_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Daily Summary", 
                command=self.visualization.plot_daily_summary).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Cumulative Performance", 
                command=self.visualization.plot_cumulative_performance).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Correlation Matrix", 
                command=self.visualization.plot_correlation_matrix).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear", 
                command=self.visualization.clear_plot).pack(side=tk.LEFT, padx=2)