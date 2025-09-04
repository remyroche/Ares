#!/usr/bin/env python3
"""
Monitoring Dashboard GUI

A comprehensive GUI for displaying enhanced ML monitoring data including
trade decisions, daily summaries, and HMM regime information.
"""


from tkinter import ttk, messagebox, filedialog
from pathlib import Path

from .utils.logger import system_logger


class MonitoringDashboard:
    """
    Main monitoring dashboard GUI application.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the monitoring dashboard."""
        self.config = config
        self.logger = system_logger.getChild("MonitoringDashboard")
        
        # GUI Configuration
        self.gui_config = config.get("monitoring_gui", {})
        self.window_width = self.gui_config.get("window_width", 1400)
        self.window_height = self.gui_config.get("window_height", 900)
        self.refresh_interval = self.gui_config.get("refresh_interval_ms", 5000)
        
        # Data storage - separate by trading mode
        self.trade_data: Dict[str, pd.DataFrame] = {}  # mode -> DataFrame
        self.daily_summary_data: Dict[str, pd.DataFrame] = {}  # mode -> DataFrame
        self.current_data_paths: Dict[str, str] = {}  # mode -> path
        self.current_mode: str = "all"  # Currently selected mode
        
        # GUI components
        self.root: Optional[tk.Tk] = None
        self.notebook: Optional[ttk.Notebook] = None
        self.trade_frame: Optional[ttk.Frame] = None
        self.summary_frame: Optional[ttk.Frame] = None
        self.regime_frame: Optional[ttk.Frame] = None
        
        # Widgets
        self.trade_tree: Optional[ttk.Treeview] = None
        self.summary_tree: Optional[ttk.Treeview] = None
        self.regime_tree: Optional[ttk.Treeview] = None
        
        # Status and control
        self.status_var = tk.StringVar()
        self.last_update_var = tk.StringVar()
        self.auto_refresh_var = tk.BooleanVar(value=True)
        
        self.logger.info("Monitoring Dashboard initialized")
    
    def create_gui(self):
        """Create the main GUI interface."""
        try:
            # Create main window
            self.root = tk.Tk()
            self.root.title("Enhanced ML Monitoring Dashboard")
            self.root.geometry(f"{self.window_width}x{self.window_height}")
            self.root.configure(bg='#f0f0f0')
            
            # Create menu bar
            self._create_menu_bar()
            
            # Create main notebook for tabs
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create tabs
            self._create_trade_decisions_tab()
            self._create_daily_summary_tab()
            self._create_regime_analysis_tab()
            self._create_statistics_tab()
            
            # Create status bar
            self._create_status_bar()
            
            # Create control panel
            self._create_control_panel()
            
            # Start auto-refresh if enabled
            if self.auto_refresh_var.get():
                self._start_auto_refresh()
            
            self.logger.info("GUI created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating GUI: {e}")
            messagebox.showerror("Error", f"Failed to create GUI: {e}")
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load CSV Data", command=self._load_csv_data)
        file_menu.add_command(label="Export Current View", command=self._export_current_view)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Auto Refresh", variable=self.auto_refresh_var, 
                                 command=self._toggle_auto_refresh)
        view_menu.add_command(label="Refresh Now", command=self._refresh_data)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_trade_decisions_tab(self):
        """Create the trade decisions tab."""
        self.trade_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trade_frame, text="Trade Decisions")
        
        # Create toolbar
        toolbar = ttk.Frame(self.trade_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Trade Data", command=self._load_trade_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Filter", command=self._show_trade_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Export", command=self._export_trade_data).pack(side=tk.LEFT, padx=5)
        
        # Create treeview with scrollbars
        tree_frame = ttk.Frame(self.trade_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview
        self.trade_tree = ttk.Treeview(tree_frame)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.trade_tree.xview)
        self.trade_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack widgets
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure columns (will be set when data is loaded)
        self.trade_tree['columns'] = ('timestamp', 'token', 'action', 'price', 'confidence', 'regime')
        self.trade_tree['show'] = 'headings'
        
        # Set column headings
        for col in self.trade_tree['columns']:
            self.trade_tree.heading(col, text=col.title())
            self.trade_tree.column(col, width=100)
    
    def _create_daily_summary_tab(self):
        """Create the daily summary tab."""
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Daily Summary")
        
        # Create toolbar
        toolbar = ttk.Frame(self.summary_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Summary Data", command=self._load_summary_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Date Range", command=self._show_date_range).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Export", command=self._export_summary_data).pack(side=tk.LEFT, padx=5)
        
        # Create treeview
        tree_frame = ttk.Frame(self.summary_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_tree = ttk.Treeview(tree_frame)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.summary_tree.xview)
        self.summary_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack widgets
        self.summary_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure columns
        self.summary_tree['columns'] = ('date', 'total_trades', 'long_trades', 'short_trades', 
                                       'dominant_regime', 'total_pnl', 'win_rate', 'profit_factor')
        self.summary_tree['show'] = 'headings'
        
        # Set column headings
        for col in self.summary_tree['columns']:
            self.summary_tree.heading(col, text=col.replace('_', ' ').title())
            self.summary_tree.column(col, width=120)
    
    def _create_regime_analysis_tab(self):
        """Create the regime analysis tab."""
        self.regime_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.regime_frame, text="Regime Analysis")
        
        # Create toolbar
        toolbar = ttk.Frame(self.regime_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Regime Data", command=self._load_regime_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Regime Filter", command=self._show_regime_filter).pack(side=tk.LEFT, padx=5)
        
        # Create treeview
        tree_frame = ttk.Frame(self.regime_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.regime_tree = ttk.Treeview(tree_frame)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.regime_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.regime_tree.xview)
        self.regime_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack widgets
        self.regime_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure columns
        self.regime_tree['columns'] = ('regime_id', 'regime_name', 'probability', 'stability', 
                                      'duration', 'trade_count', 'avg_pnl', 'win_rate')
        self.regime_tree['show'] = 'headings'
        
        # Set column headings
        for col in self.regime_tree['columns']:
            self.regime_tree.heading(col, text=col.replace('_', ' ').title())
            self.regime_tree.column(col, width=120)
    
    def _create_statistics_tab(self):
        """Create the statistics tab."""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")
        
        # Create statistics display
        stats_text = tk.Text(stats_frame, wrap=tk.WORD, font=('Courier', 10))
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=stats_text.yview)
        stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store reference for updates
        self.stats_text = stats_text
    
    def _create_status_bar(self):
        """Create the status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.last_update_var).pack(side=tk.RIGHT, padx=5)
        
        self.status_var.set("Ready")
        self.last_update_var.set("Last update: Never")
    
    def _create_control_panel(self):
        """Create the control panel."""
        control_frame = ttk.LabelFrame(self.root, text="Controls")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Auto refresh control
        ttk.Checkbutton(control_frame, text="Auto Refresh", variable=self.auto_refresh_var,
                       command=self._toggle_auto_refresh).pack(side=tk.LEFT, padx=5)
        
        # Refresh interval
        ttk.Label(control_frame, text="Refresh Interval (ms):").pack(side=tk.LEFT, padx=5)
        self.refresh_interval_var = tk.StringVar(value=str(self.refresh_interval))
        refresh_entry = ttk.Entry(control_frame, textvariable=self.refresh_interval_var, width=10)
        refresh_entry.pack(side=tk.LEFT, padx=5)
        
        # Apply button
        ttk.Button(control_frame, text="Apply", command=self._apply_refresh_interval).pack(side=tk.LEFT, padx=5)
        
        # Trading mode selection
        ttk.Label(control_frame, text="Trading Mode:").pack(side=tk.LEFT, padx=5)
        self.trading_mode_var = tk.StringVar(value="all")
        mode_combo = ttk.Combobox(control_frame, textvariable=self.trading_mode_var, 
                                 values=["all", "backtest", "paper", "live"], 
                                 state="readonly", width=10)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)
        
        # Data path display
        ttk.Label(control_frame, text="Data Path:").pack(side=tk.LEFT, padx=5)
        self.data_path_var = tk.StringVar(value="No data loaded")
        ttk.Label(control_frame, textvariable=self.data_path_var, foreground="blue").pack(side=tk.LEFT, padx=5)
    
    def _load_csv_data(self):
        """Load CSV data from file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self._load_data_from_file(file_path)
    
    def _load_data_from_file(self, file_path: str):
        """Load data from a CSV file."""
        try:
            self.status_var.set("Loading data...")
            self.root.update()
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Determine data type and trading mode based on columns and filename
            if 'decision_id' in df.columns:
                # Trade decisions data
                mode = self._extract_mode_from_filename(file_path, df)
                self.trade_data[mode] = df
                self.current_data_paths[mode] = file_path
                
                # Update current mode if it matches
                if mode == self.current_mode or self.current_mode == "all":
                    self._populate_trade_tree()
                
                self.status_var.set(f"Loaded {len(df)} {mode} trade decisions")
                
            elif 'date' in df.columns and 'total_trades' in df.columns:
                # Daily summary data
                mode = self._extract_mode_from_filename(file_path, df)
                self.daily_summary_data[mode] = df
                self.current_data_paths[mode] = file_path
                
                # Update current mode if it matches
                if mode == self.current_mode or self.current_mode == "all":
                    self._populate_summary_tree()
                
                self.status_var.set(f"Loaded {len(df)} {mode} daily summaries")
            else:
                messagebox.showwarning("Warning", "Unknown CSV format")
                return
            
            self.data_path_var.set(file_path)
            self.last_update_var.set(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.status_var.set("Error loading data")
    
    def _extract_mode_from_filename(self, file_path: str, df: pd.DataFrame) -> str:
        """Extract trading mode from filename or data."""
        try:
            filename = Path(file_path).name.lower()
            
            # Check filename for mode indicators
            if 'backtest' in filename:
                return 'backtest'
            elif 'paper' in filename:
                return 'paper'
            elif 'live' in filename:
                return 'live'
            
            # Check data for trading_mode column
            if 'trading_mode' in df.columns:
                modes = df['trading_mode'].unique()
                if len(modes) == 1:
                    return modes[0]
                else:
                    return 'mixed'
            
            # Default to 'all' if no mode can be determined
            return 'all'
            
        except Exception as e:
            self.logger.error(f"Error extracting mode from filename: {e}")
            return 'all'
    
    def _populate_trade_tree(self):
        """Populate the trade decisions tree."""
        if self.trade_tree is None:
            return
        
        # Clear existing data
        for item in self.trade_tree.get_children():
            self.trade_tree.delete(item)
        
        # Get data for current mode
        if self.current_mode == "all":
            # Combine all modes
            all_data = []
            for mode, df in self.trade_data.items():
                all_data.append(df)
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
            else:
                return
        elif self.current_mode in self.trade_data:
            df = self.trade_data[self.current_mode]
        else:
            return
        
        # Update columns based on data
        columns = list(df.columns)
        self.trade_tree['columns'] = columns
        self.trade_tree['show'] = 'headings'
        
        # Set column headings and widths
        for col in columns:
            self.trade_tree.heading(col, text=col.replace('_', ' ').title())
            self.trade_tree.column(col, width=120, minwidth=80)
        
        # Insert data
        for index, row in df.iterrows():
            values = [str(row[col]) if pd.notna(row[col]) else "" for col in columns]
            self.trade_tree.insert("", tk.END, values=values)
    
    def _populate_summary_tree(self):
        """Populate the daily summary tree."""
        if self.summary_tree is None:
            return
        
        # Clear existing data
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        # Get data for current mode
        if self.current_mode == "all":
            # Combine all modes
            all_data = []
            for mode, df in self.daily_summary_data.items():
                all_data.append(df)
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
            else:
                return
        elif self.current_mode in self.daily_summary_data:
            df = self.daily_summary_data[self.current_mode]
        else:
            return
        
        # Update columns based on data
        columns = list(df.columns)
        self.summary_tree['columns'] = columns
        self.summary_tree['show'] = 'headings'
        
        # Set column headings and widths
        for col in columns:
            self.summary_tree.heading(col, text=col.replace('_', ' ').title())
            self.summary_tree.column(col, width=120, minwidth=80)
        
        # Insert data
        for index, row in df.iterrows():
            values = [str(row[col]) if pd.notna(row[col]) else "" for col in columns]
            self.summary_tree.insert("", tk.END, values=values)
    
    def _populate_regime_tree(self):
        """Populate the regime analysis tree."""
        if self.regime_tree is None:
            return
        
        # Clear existing data
        for item in self.regime_tree.get_children():
            self.regime_tree.delete(item)
        
        # Get data for current mode
        if self.current_mode == "all":
            # Combine all modes
            all_data = []
            for mode, df in self.trade_data.items():
                all_data.append(df)
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
            else:
                return
        elif self.current_mode in self.trade_data:
            df = self.trade_data[self.current_mode]
        else:
            return
        
        # Analyze regime data
        regime_stats = self._analyze_regime_data(df)
        
        # Insert regime statistics
        for regime_id, stats in regime_stats.items():
            values = [
                regime_id,
                stats.get('name', 'Unknown'),
                f"{stats.get('probability', 0):.3f}",
                f"{stats.get('stability', 0):.3f}",
                stats.get('duration', 0),
                stats.get('trade_count', 0),
                f"{stats.get('avg_pnl', 0):.2f}",
                f"{stats.get('win_rate', 0):.3f}"
            ]
            self.regime_tree.insert("", tk.END, values=values)
    
    def _analyze_regime_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze regime data from trade decisions."""
        regime_stats = {}
        
        # Group by regime
        if 'hmm_regime_id' in df.columns:
            for regime_id, group in df.groupby('hmm_regime_id'):
                stats = {
                    'name': regime_id,
                    'trade_count': len(group),
                    'probability': group['hmm_regime_probability'].mean() if 'hmm_regime_probability' in group.columns else 0,
                    'stability': group['hmm_regime_stability_score'].mean() if 'hmm_regime_stability_score' in group.columns else 0,
                    'duration': group['hmm_regime_duration'].mean() if 'hmm_regime_duration' in group.columns else 0,
                    'avg_pnl': 0,  # Would need success_metrics data
                    'win_rate': 0   # Would need success_metrics data
                }
                regime_stats[regime_id] = stats
        
        return regime_stats
    
    def _update_statistics(self):
        """Update the statistics display."""
        if self.stats_text is None:
            return
        
        self.stats_text.delete(1.0, tk.END)
        
        stats_text = f"=== MONITORING STATISTICS ({self.current_mode.upper()}) ===\n\n"
        
        # Trade decisions statistics
        if self.current_mode == "all":
            total_trades = sum(len(df) for df in self.trade_data.values())
            stats_text += f"Total Trade Decisions: {total_trades}\n"
            
            for mode, df in self.trade_data.items():
                stats_text += f"\n{mode.upper()} Mode:\n"
                stats_text += f"  Trades: {len(df)}\n"
                
                if 'action' in df.columns:
                    action_counts = df['action'].value_counts()
                    stats_text += f"  Actions: {dict(action_counts)}\n"
                
                if 'overall_confidence' in df.columns:
                    avg_confidence = df['overall_confidence'].mean()
                    stats_text += f"  Avg Confidence: {avg_confidence:.3f}\n"
                
                if 'hmm_regime_id' in df.columns:
                    regime_counts = df['hmm_regime_id'].value_counts()
                    stats_text += f"  Regime Distribution: {dict(regime_counts)}\n"
        elif self.current_mode in self.trade_data:
            df = self.trade_data[self.current_mode]
            stats_text += f"Trade Decisions: {len(df)}\n"
            
            if 'action' in df.columns:
                action_counts = df['action'].value_counts()
                stats_text += f"Actions: {dict(action_counts)}\n"
            
            if 'overall_confidence' in df.columns:
                avg_confidence = df['overall_confidence'].mean()
                stats_text += f"Average Confidence: {avg_confidence:.3f}\n"
            
            if 'hmm_regime_id' in df.columns:
                regime_counts = df['hmm_regime_id'].value_counts()
                stats_text += f"Regime Distribution: {dict(regime_counts)}\n"
        
        # Daily summary statistics
        if self.current_mode == "all":
            total_summaries = sum(len(df) for df in self.daily_summary_data.values())
            stats_text += f"\nTotal Daily Summaries: {total_summaries}\n"
            
            for mode, df in self.daily_summary_data.items():
                stats_text += f"\n{mode.upper()} Daily Summaries:\n"
                stats_text += f"  Days: {len(df)}\n"
                
                if 'total_pnl' in df.columns:
                    total_pnl = df['total_pnl'].sum()
                    avg_pnl = df['total_pnl'].mean()
                    stats_text += f"  Total PnL: {total_pnl:.2f}\n"
                    stats_text += f"  Avg Daily PnL: {avg_pnl:.2f}\n"
                
                if 'win_rate' in df.columns:
                    avg_win_rate = df['win_rate'].mean()
                    stats_text += f"  Avg Win Rate: {avg_win_rate:.3f}\n"
        elif self.current_mode in self.daily_summary_data:
            df = self.daily_summary_data[self.current_mode]
            stats_text += f"\nDaily Summaries: {len(df)}\n"
            
            if 'total_pnl' in df.columns:
                total_pnl = df['total_pnl'].sum()
                avg_pnl = df['total_pnl'].mean()
                stats_text += f"Total PnL: {total_pnl:.2f}\n"
                stats_text += f"Average Daily PnL: {avg_pnl:.2f}\n"
            
            if 'win_rate' in df.columns:
                avg_win_rate = df['win_rate'].mean()
                stats_text += f"Average Win Rate: {avg_win_rate:.3f}\n"
        
        self.stats_text.insert(1.0, stats_text)
    
    def _start_auto_refresh(self):
        """Start auto-refresh timer."""
        if self.auto_refresh_var.get():
            self._refresh_data()
            self.root.after(self.refresh_interval, self._start_auto_refresh)
    
    def _toggle_auto_refresh(self):
        """Toggle auto-refresh on/off."""
        if self.auto_refresh_var.get():
            self._start_auto_refresh()
        else:
            self.status_var.set("Auto-refresh disabled")
    
    def _apply_refresh_interval(self):
        """Apply new refresh interval."""
        try:
            self.refresh_interval = int(self.refresh_interval_var.get())
            self.status_var.set(f"Refresh interval set to {self.refresh_interval}ms")
        except ValueError:
            messagebox.showerror("Error", "Invalid refresh interval")
    
    def _on_mode_changed(self, event=None):
        """Handle trading mode selection change."""
        try:
            new_mode = self.trading_mode_var.get()
            if new_mode != self.current_mode:
                self.current_mode = new_mode
                self._update_display_for_mode()
                self.status_var.set(f"Switched to {new_mode} mode")
        except Exception as e:
            self.logger.error(f"Error changing trading mode: {e}")
    
    def _update_display_for_mode(self):
        """Update display based on current trading mode."""
        try:
            # Update data path display
            if self.current_mode in self.current_data_paths:
                self.data_path_var.set(self.current_data_paths[self.current_mode])
            else:
                self.data_path_var.set(f"No {self.current_mode} data loaded")
            
            # Update current tab display
            current_tab = self.notebook.tab(self.notebook.select(), "text")
            
            if current_tab == "Trade Decisions":
                self._populate_trade_tree()
            elif current_tab == "Daily Summary":
                self._populate_summary_tree()
            elif current_tab == "Regime Analysis":
                self._populate_regime_tree()
            elif current_tab == "Statistics":
                self._update_statistics()
                
        except Exception as e:
            self.logger.error(f"Error updating display for mode: {e}")
    
    def _refresh_data(self):
        """Refresh data from current files."""
        if self.current_data_paths:
            for mode, path in self.current_data_paths.items():
                self._load_data_from_file(path)
            self._update_statistics()
    
    def _load_trade_data(self):
        """Load trade data specifically."""
        file_path = filedialog.askopenfilename(
            title="Select Trade Decisions CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self._load_data_from_file(file_path)
    
    def _load_summary_data(self):
        """Load daily summary data specifically."""
        file_path = filedialog.askopenfilename(
            title="Select Daily Summary CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self._load_data_from_file(file_path)
    
    def _load_regime_data(self):
        """Load regime data from trade decisions."""
        if self.trade_data is not None:
            self._populate_regime_tree()
        else:
            messagebox.showwarning("Warning", "Please load trade data first")
    
    def _show_trade_filter(self):
        """Show trade filter dialog."""
        messagebox.showinfo("Info", "Trade filter functionality coming soon")
    
    def _show_date_range(self):
        """Show date range selection dialog."""
        messagebox.showinfo("Info", "Date range selection coming soon")
    
    def _show_regime_filter(self):
        """Show regime filter dialog."""
        messagebox.showinfo("Info", "Regime filter functionality coming soon")
    
    def _export_trade_data(self):
        """Export trade data."""
        if self.trade_data is not None:
            file_path = filedialog.asksaveasfilename(
                title="Export Trade Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.trade_data.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Trade data exported to {file_path}")
    
    def _export_summary_data(self):
        """Export summary data."""
        if self.daily_summary_data is not None:
            file_path = filedialog.asksaveasfilename(
                title="Export Summary Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.daily_summary_data.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Summary data exported to {file_path}")
    
    def _export_current_view(self):
        """Export current view data."""
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        
        if current_tab == "Trade Decisions" and self.trade_data is not None:
            self._export_trade_data()
        elif current_tab == "Daily Summary" and self.daily_summary_data is not None:
            self._export_summary_data()
        else:
            messagebox.showwarning("Warning", "No data to export")
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """
Enhanced ML Monitoring Dashboard

A comprehensive GUI for monitoring ML model performance
and trading decisions with HMM regime analysis.

Features:
- Trade decision tracking
- Daily summary analysis
- HMM regime monitoring
- Real-time data visualization
- CSV import/export

Version: 1.0.0
        """
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run the GUI application."""
        try:
            self.create_gui()
            self.logger.info("Starting monitoring dashboard GUI")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error running GUI: {e}")
            messagebox.showerror("Error", f"Failed to run GUI: {e}")


def create_monitoring_dashboard(config: Dict[str, Any]) -> MonitoringDashboard:
    """Create and return a monitoring dashboard instance."""
    return MonitoringDashboard(config)


if __name__ == "__main__":
    # Example usage
    config = {
        "monitoring_gui": {
            "window_width": 1400,
            "window_height": 900,
            "refresh_interval_ms": 5000
        }
    }
    
    dashboard = create_monitoring_dashboard(config)
    dashboard.run()