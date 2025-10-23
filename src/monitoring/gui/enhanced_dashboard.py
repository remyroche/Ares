"""
Enhanced Monitoring Dashboard with Visualization

A comprehensive GUI dashboard that combines data display and visualization
for enhanced ML monitoring with HMM regime analysis.
"""
import logging
import typing
from typing import Dict, Any, Optional
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

from ...utils.logger import system_logger
from .monitoring_dashboard import MonitoringDashboard
from .data_visualization import MonitoringVisualization, VisualizationControlPanel

class EnhancedMonitoringDashboard(MonitoringDashboard):
    """
    Enhanced monitoring dashboard with integrated visualization capabilities.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the enhanced monitoring dashboard."""
        super().__init__(config)
        self.logger = system_logger.getChild('EnhancedMonitoringDashboard')
        self.visualization_frame: Optional[ttk.Frame] = None
        self.visualization: Optional[MonitoringVisualization] = None
        self.visualization_controls: Optional[VisualizationControlPanel] = None
        self.logger.info('Enhanced Monitoring Dashboard initialized')

    def _create_trade_decisions_tab(self) -> None:
        """Create the enhanced trade decisions tab with visualization."""
        self.trade_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trade_frame, text='Trade Decisions')
        paned_window = ttk.PanedWindow(self.trade_frame, orient = tk.HORIZONTAL)
        paned_window.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight = 1)
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight = 1)
        toolbar = ttk.Frame(left_frame)
        toolbar.pack(fill = tk.X, padx = 5, pady = 5)
        ttk.Button(toolbar, text='Load Trade Data', command = self._load_trade_data).pack(side = tk.LEFT, padx = 5)
        ttk.Button(toolbar, text='Filter', command = self._show_trade_filter).pack(side = tk.LEFT, padx = 5)
        ttk.Button(toolbar, text='Export', command = self._export_trade_data).pack(side = tk.LEFT, padx = 5)
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.trade_tree = ttk.Treeview(tree_frame)
        v_scrollbar = ttk.Scrollbar(tree_frame, orient = tk.VERTICAL, command = self.trade_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient = tk.HORIZONTAL, command = self.trade_tree.xview)
        self.trade_tree.configure(yscrollcommand = v_scrollbar.set, xscrollcommand = h_scrollbar.set)
        self.trade_tree.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        v_scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
        h_scrollbar.pack(side = tk.BOTTOM, fill = tk.X)
        self.trade_tree['columns'] = ('timestamp', 'token', 'action', 'price', 'confidence', 'regime')
        self.trade_tree['show'] = 'headings'
        for col in self.trade_tree['columns']:
            self.trade_tree.heading(col, text = col.title())
            self.trade_tree.column(col, width = 100)
        self.visualization_frame = ttk.Frame(right_frame)
        self.visualization_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.visualization_controls = VisualizationControlPanel(self.visualization_frame, None)
        viz_container = ttk.Frame(self.visualization_frame)
        viz_container.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.visualization = MonitoringVisualization(viz_container)
        self.visualization_controls.visualization = self.visualization

    def _create_daily_summary_tab(self) -> None:
        """Create the enhanced daily summary tab with visualization."""
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text='Daily Summary')
        paned_window = ttk.PanedWindow(self.summary_frame, orient = tk.HORIZONTAL)
        paned_window.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight = 1)
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight = 1)
        toolbar = ttk.Frame(left_frame)
        toolbar.pack(fill = tk.X, padx = 5, pady = 5)
        ttk.Button(toolbar, text='Load Summary Data', command = self._load_summary_data).pack(side = tk.LEFT, padx = 5)
        ttk.Button(toolbar, text='Date Range', command = self._show_date_range).pack(side = tk.LEFT, padx = 5)
        ttk.Button(toolbar, text='Export', command = self._export_summary_data).pack(side = tk.LEFT, padx = 5)
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.summary_tree = ttk.Treeview(tree_frame)
        v_scrollbar = ttk.Scrollbar(tree_frame, orient = tk.VERTICAL, command = self.summary_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient = tk.HORIZONTAL, command = self.summary_tree.xview)
        self.summary_tree.configure(yscrollcommand = v_scrollbar.set, xscrollcommand = h_scrollbar.set)
        self.summary_tree.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        v_scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
        h_scrollbar.pack(side = tk.BOTTOM, fill = tk.X)
        self.summary_tree['columns'] = ('date', 'total_trades', 'long_trades', 'short_trades', 'dominant_regime', 'total_pnl', 'win_rate', 'profit_factor')
        self.summary_tree['show'] = 'headings'
        for col in self.summary_tree['columns']:
            self.summary_tree.heading(col, text = col.replace('_', ' ').title())
            self.summary_tree.column(col, width = 120)
        viz_container = ttk.Frame(right_frame)
        viz_container.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        viz_controls = VisualizationControlPanel(viz_container, None)
        viz_plot_frame = ttk.Frame(viz_container)
        viz_plot_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.summary_visualization = MonitoringVisualization(viz_plot_frame)
        viz_controls.visualization = self.summary_visualization

    def _create_regime_analysis_tab(self) -> None:
        """Create the enhanced regime analysis tab with visualization."""
        self.regime_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.regime_frame, text='Regime Analysis')
        paned_window = ttk.PanedWindow(self.regime_frame, orient = tk.HORIZONTAL)
        paned_window.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight = 1)
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight = 1)
        toolbar = ttk.Frame(left_frame)
        toolbar.pack(fill = tk.X, padx = 5, pady = 5)
        ttk.Button(toolbar, text='Load Regime Data', command = self._load_regime_data).pack(side = tk.LEFT, padx = 5)
        ttk.Button(toolbar, text='Regime Filter', command = self._show_regime_filter).pack(side = tk.LEFT, padx = 5)
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.regime_tree = ttk.Treeview(tree_frame)
        v_scrollbar = ttk.Scrollbar(tree_frame, orient = tk.VERTICAL, command = self.regime_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient = tk.HORIZONTAL, command = self.regime_tree.xview)
        self.regime_tree.configure(yscrollcommand = v_scrollbar.set, xscrollcommand = h_scrollbar.set)
        self.regime_tree.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
        v_scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
        h_scrollbar.pack(side = tk.BOTTOM, fill = tk.X)
        self.regime_tree['columns'] = ('regime_id', 'regime_name', 'probability', 'stability', 'duration', 'trade_count', 'avg_pnl', 'win_rate')
        self.regime_tree['show'] = 'headings'
        for col in self.regime_tree['columns']:
            self.regime_tree.heading(col, text = col.replace('_', ' ').title())
            self.regime_tree.column(col, width = 120)
        viz_container = ttk.Frame(right_frame)
        viz_container.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        viz_controls = VisualizationControlPanel(viz_container, None)
        viz_plot_frame = ttk.Frame(viz_container)
        viz_plot_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.regime_visualization = MonitoringVisualization(viz_plot_frame)
        viz_controls.visualization = self.regime_visualization

    def _create_statistics_tab(self) -> None:
        """Create the enhanced statistics tab with visualization."""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text='Statistics & Analytics')
        paned_window = ttk.PanedWindow(stats_frame, orient = tk.VERTICAL)
        paned_window.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        top_frame = ttk.Frame(paned_window)
        paned_window.add(top_frame, weight = 1)
        bottom_frame = ttk.Frame(paned_window)
        paned_window.add(bottom_frame, weight = 2)
        stats_text = tk.Text(top_frame, wrap = tk.WORD, font=('Courier', 10))
        stats_scrollbar = ttk.Scrollbar(top_frame, orient = tk.VERTICAL, command = stats_text.yview)
        stats_text.configure(yscrollcommand = stats_scrollbar.set)
        stats_text.pack(side = tk.LEFT, fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        stats_scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
        self.stats_text = stats_text
        viz_container = ttk.Frame(bottom_frame)
        viz_container.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        viz_controls = VisualizationControlPanel(viz_container, None)
        viz_plot_frame = ttk.Frame(viz_container)
        viz_plot_frame.pack(fill = tk.BOTH, expand = True, padx = 5, pady = 5)
        self.stats_visualization = MonitoringVisualization(viz_plot_frame)
        viz_controls.visualization = self.stats_visualization

    def _populate_trade_tree(self) -> None:
        """Populate the trade decisions tree and update visualization."""
        super()._populate_trade_tree()
        if self.visualization:
            if self.current_mode == 'all':
                all_data = []
                for mode, df in self.trade_data.items():
                    all_data.append(df)
                if all_data:
                    df = pd.concat(all_data, ignore_index = True)
                    self.visualization.set_trade_data(df)
                    self.visualization.plot_trade_performance()
            elif self.current_mode in self.trade_data:
                self.visualization.set_trade_data(self.trade_data[self.current_mode])
                self.visualization.plot_trade_performance()

    def _populate_summary_tree(self) -> None:
        """Populate the daily summary tree and update visualization."""
        super()._populate_summary_tree()
        if hasattr(self, 'summary_visualization'):
            if self.current_mode == 'all':
                all_data = []
                for mode, df in self.daily_summary_data.items():
                    all_data.append(df)
                if all_data:
                    df = pd.concat(all_data, ignore_index = True)
                    self.summary_visualization.set_daily_summary_data(df)
                    self.summary_visualization.plot_daily_summary()
            elif self.current_mode in self.daily_summary_data:
                self.summary_visualization.set_daily_summary_data(self.daily_summary_data[self.current_mode])
                self.summary_visualization.plot_daily_summary()

    def _populate_regime_tree(self) -> None:
        """Populate the regime analysis tree and update visualization."""
        super()._populate_regime_tree()
        if hasattr(self, 'regime_visualization'):
            if self.current_mode == 'all':
                all_data = []
                for mode, df in self.trade_data.items():
                    all_data.append(df)
                if all_data:
                    df = pd.concat(all_data, ignore_index = True)
                    self.regime_visualization.set_trade_data(df)
                    self.regime_visualization.plot_regime_analysis()
            elif self.current_mode in self.trade_data:
                self.regime_visualization.set_trade_data(self.trade_data[self.current_mode])
                self.regime_visualization.plot_regime_analysis()

    def _update_statistics(self) -> None:
        """Update the statistics display and visualization."""
        super()._update_statistics()
        if hasattr(self, 'stats_visualization'):
            if self.current_mode == 'all':
                all_trade_data = []
                for mode, df in self.trade_data.items():
                    all_trade_data.append(df)
                if all_trade_data:
                    df = pd.concat(all_trade_data, ignore_index = True)
                    self.stats_visualization.set_trade_data(df)
                all_summary_data = []
                for mode, df in self.daily_summary_data.items():
                    all_summary_data.append(df)
                if all_summary_data:
                    df = pd.concat(all_summary_data, ignore_index = True)
                    self.stats_visualization.set_daily_summary_data(df)
            else:
                if self.current_mode in self.trade_data:
                    self.stats_visualization.set_trade_data(self.trade_data[self.current_mode])
                if self.current_mode in self.daily_summary_data:
                    self.stats_visualization.set_daily_summary_data(self.daily_summary_data[self.current_mode])
            if self.current_mode in self.daily_summary_data or self.current_mode == 'all':
                self.stats_visualization.plot_cumulative_performance()
            elif self.current_mode in self.trade_data or self.current_mode == 'all':
                self.stats_visualization.plot_trade_performance()

    def _load_data_from_file(self, file_path: str) -> None:
        """Load data from a CSV file and update all visualizations."""
        super()._load_data_from_file(file_path)
        if self.trade_data is not None:
            if self.visualization:
                self.visualization.set_trade_data(self.trade_data)
            if hasattr(self, 'regime_visualization'):
                self.regime_visualization.set_trade_data(self.trade_data)
            if hasattr(self, 'stats_visualization'):
                self.stats_visualization.set_trade_data(self.trade_data)
        if self.daily_summary_data is not None:
            if hasattr(self, 'summary_visualization'):
                self.summary_visualization.set_daily_summary_data(self.daily_summary_data)
            if hasattr(self, 'stats_visualization'):
                self.stats_visualization.set_daily_summary_data(self.daily_summary_data)

    def _show_about(self) -> None:
        """Show about dialog with enhanced information."""
        about_text = '\nEnhanced ML Monitoring Dashboard\n\nA comprehensive GUI for monitoring ML model performance\nand trading decisions with HMM regime analysis.\n\nFeatures:\n- Trade decision tracking with HMM regime information\n- Daily summary analysis with PnL and win rate tracking\n- HMM regime monitoring and analysis\n- Real-time data visualization with multiple chart types\n- CSV import/export with detailed breakdowns\n- Interactive charts and correlation analysis\n- Split-pane interface for data and visualization\n\nVersion: 2.0.0\n        '
        messagebox.showinfo('About', about_text)

def create_enhanced_monitoring_dashboard(config: Dict[str, Any]) -> EnhancedMonitoringDashboard:
    """Create and return an enhanced monitoring dashboard instance."""
    return EnhancedMonitoringDashboard(config)
if __name__ == '__main__':
    config = {'monitoring_gui': {'window_width': 1600, 'window_height': 1000, 'refresh_interval_ms': 5000}}
    dashboard = create_enhanced_monitoring_dashboard(config)
    dashboard.run()
