"""
Advanced Monitoring & Analytics for CVLSA

This module implements comprehensive monitoring and analytics with:
1. Detailed analytics for optimization process
2. Experiment tracking and logging
3. Performance reporting and visualization
4. Real-time monitoring dashboards
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import json
import sqlite3
from pathlib import Path
import threading
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import queue
import weakref

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    # Experiment tracking
    enable_experiment_tracking: bool = True
    experiment_database: str = "./experiments.db"
    auto_save_interval: int = 60  # Seconds
    
    # Analytics
    enable_detailed_analytics: bool = True
    analytics_retention_days: int = 30
    performance_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 'mse', 'mae', 'r2_score'
    ])
    
    # Monitoring
    enable_real_time_monitoring: bool = True
    monitoring_interval: int = 5  # Seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'memory_usage': 0.8,
        'cpu_usage': 0.9,
        'error_rate': 0.1
    })
    
    # Reporting
    enable_auto_reporting: bool = True
    report_frequency: str = 'daily'  # 'hourly', 'daily', 'weekly'
    report_formats: List[str] = field(default_factory=lambda: ['json', 'html'])
    
    # Visualization
    enable_visualization: bool = True
    plot_style: str = 'seaborn'
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100

@dataclass
class Experiment:
    """Experiment tracking data structure."""
    experiment_id: str
    name: str
    description: str
    start_time: float
    end_time: Optional[float] = None
    status: str = 'running'  # 'running', 'completed', 'failed', 'cancelled'
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

class ExperimentTracker:
    """Advanced experiment tracking system."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.db_path = Path(config.experiment_database)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Active experiments
        self.active_experiments: Dict[str, Experiment] = {}
        self._lock = threading.Lock()
        
        # Auto-save thread
        self.auto_save_thread = None
        self.auto_save_active = False
        
        if config.auto_save_interval > 0:
            self.start_auto_save()
        
        logger.info("📊 Experiment tracker initialized")
    
    def _init_database(self):
        """Initialize experiment database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    status TEXT NOT NULL,
                    config TEXT,
                    metrics TEXT,
                    hyperparameters TEXT,
                    results TEXT,
                    artifacts TEXT,
                    tags TEXT,
                    notes TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    component TEXT
                )
            ''')
    
    def start_experiment(self, name: str, description: str = "", 
                        config: Optional[Dict[str, Any]] = None,
                        hyperparameters: Optional[Dict[str, Any]] = None,
                        tags: Optional[List[str]] = None) -> str:
        """Start a new experiment."""
        experiment_id = f"exp_{int(time.time())}_{hash(name) % 10000}"
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            start_time=time.time(),
            config=config or {},
            hyperparameters=hyperparameters or {},
            tags=tags or []
        )
        
        with self._lock:
            self.active_experiments[experiment_id] = experiment
        
        # Save to database
        self._save_experiment(experiment)
        
        logger.info(f"🧪 Started experiment: {name} ({experiment_id})")
        return experiment_id
    
    def log_metric(self, experiment_id: str, metric_name: str, metric_value: float):
        """Log a metric for an experiment."""
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        
        with self._lock:
            experiment = self.active_experiments[experiment_id]
            experiment.metrics[metric_name] = metric_value
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiment_metrics (experiment_id, timestamp, metric_name, metric_value)
                VALUES (?, ?, ?, ?)
            ''', (experiment_id, time.time(), metric_name, metric_value))
    
    def log_system_metric(self, metric_name: str, metric_value: float, component: str = "system"):
        """Log a system metric."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO system_metrics (timestamp, metric_name, metric_value, component)
                VALUES (?, ?, ?, ?)
            ''', (time.time(), metric_name, metric_value, component))
    
    def complete_experiment(self, experiment_id: str, results: Optional[Dict[str, Any]] = None,
                           status: str = 'completed'):
        """Complete an experiment."""
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        
        with self._lock:
            experiment = self.active_experiments[experiment_id]
            experiment.end_time = time.time()
            experiment.status = status
            if results:
                experiment.results = results
        
        # Save to database
        self._save_experiment(experiment)
        
        # Remove from active experiments
        with self._lock:
            del self.active_experiments[experiment_id]
        
        duration = experiment.end_time - experiment.start_time
        logger.info(f"✅ Completed experiment: {experiment.name} ({experiment_id}) in {duration:.2f}s")
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO experiments 
                (experiment_id, name, description, start_time, end_time, status, config, 
                 metrics, hyperparameters, results, artifacts, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment.experiment_id,
                experiment.name,
                experiment.description,
                experiment.start_time,
                experiment.end_time,
                experiment.status,
                json.dumps(experiment.config),
                json.dumps(experiment.metrics),
                json.dumps(experiment.hyperparameters),
                json.dumps(experiment.results),
                json.dumps(experiment.artifacts),
                json.dumps(experiment.tags),
                experiment.notes
            ))
    
    def start_auto_save(self):
        """Start auto-save thread."""
        if self.auto_save_active:
            return
        
        self.auto_save_active = True
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()
        
        logger.info("💾 Auto-save started")
    
    def stop_auto_save(self):
        """Stop auto-save thread."""
        self.auto_save_active = False
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=1.0)
        
        logger.info("💾 Auto-save stopped")
    
    def _auto_save_loop(self):
        """Auto-save loop."""
        while self.auto_save_active:
            try:
                with self._lock:
                    for experiment in self.active_experiments.values():
                        self._save_experiment(experiment)
                
                time.sleep(self.config.auto_save_interval)
                
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
                time.sleep(self.config.auto_save_interval)
    
    def get_experiment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get experiment history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM experiments 
                ORDER BY start_time DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            experiments = []
            
            for row in cursor.fetchall():
                experiment_dict = dict(zip(columns, row))
                
                # Parse JSON fields
                for field in ['config', 'metrics', 'hyperparameters', 'results', 'artifacts', 'tags']:
                    if experiment_dict[field]:
                        experiment_dict[field] = json.loads(experiment_dict[field])
                
                experiments.append(experiment_dict)
            
            return experiments
    
    def get_experiment_metrics(self, experiment_id: str) -> pd.DataFrame:
        """Get metrics for a specific experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, metric_name, metric_value 
                FROM experiment_metrics 
                WHERE experiment_id = ? 
                ORDER BY timestamp
            ''', (experiment_id,))
            
            data = cursor.fetchall()
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=['timestamp', 'metric_name', 'metric_value'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df

class PerformanceAnalytics:
    """Advanced performance analytics system."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.performance_data: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Initialize visualization
        if config.enable_visualization:
            plt.style.use(config.plot_style)
        
        logger.info("📈 Performance analytics initialized")
    
    def record_performance(self, component: str, operation: str, 
                          metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        """Record performance metrics."""
        performance_record = {
            'timestamp': time.time(),
            'component': component,
            'operation': operation,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        with self._lock:
            self.performance_data.append(performance_record)
            
            # Limit data size
            if len(self.performance_data) > 10000:
                self.performance_data = self.performance_data[-5000:]
    
    def generate_performance_report(self, component: Optional[str] = None,
                                  operation: Optional[str] = None,
                                  time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            # Filter data
            filtered_data = self.performance_data
            
            if component:
                filtered_data = [d for d in filtered_data if d['component'] == component]
            
            if operation:
                filtered_data = [d for d in filtered_data if d['operation'] == operation]
            
            if time_range:
                start_time, end_time = time_range
                filtered_data = [d for d in filtered_data if start_time <= d['timestamp'] <= end_time]
            
            if not filtered_data:
                return {'error': 'No data found for the specified criteria'}
            
            # Calculate statistics
            all_metrics = {}
            for record in filtered_data:
                for metric_name, metric_value in record['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
            
            # Generate statistics
            report = {
                'total_records': len(filtered_data),
                'time_range': {
                    'start': min(d['timestamp'] for d in filtered_data),
                    'end': max(d['timestamp'] for d in filtered_data)
                },
                'metrics': {}
            }
            
            for metric_name, values in all_metrics.items():
                if values:
                    report['metrics'][metric_name] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
            
            return report
    
    def create_performance_visualization(self, component: Optional[str] = None,
                                       operation: Optional[str] = None,
                                       save_path: Optional[str] = None) -> str:
        """Create performance visualization."""
        if not self.config.enable_visualization:
            return ""
        
        with self._lock:
            # Filter data
            filtered_data = self.performance_data
            
            if component:
                filtered_data = [d for d in filtered_data if d['component'] == component]
            
            if operation:
                filtered_data = [d for d in filtered_data if d['operation'] == operation]
            
            if not filtered_data:
                return ""
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
            fig.suptitle(f'Performance Analytics - {component or "All Components"}')
            
            # Extract data for plotting
            timestamps = [d['timestamp'] for d in filtered_data]
            all_metrics = {}
            
            for record in filtered_data:
                for metric_name, metric_value in record['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
            
            # Plot 1: Time series of metrics
            if all_metrics:
                metric_names = list(all_metrics.keys())[:3]  # Plot first 3 metrics
                for i, metric_name in enumerate(metric_names):
                    if i < len(axes[0]):
                        axes[0, i].plot(timestamps, all_metrics[metric_name])
                        axes[0, i].set_title(f'{metric_name} Over Time')
                        axes[0, i].set_xlabel('Time')
                        axes[0, i].set_ylabel(metric_name)
            
            # Plot 2: Distribution of metrics
            if all_metrics:
                metric_name = list(all_metrics.keys())[0]
                axes[1, 0].hist(all_metrics[metric_name], bins=20, alpha=0.7)
                axes[1, 0].set_title(f'{metric_name} Distribution')
                axes[1, 0].set_xlabel(metric_name)
                axes[1, 0].set_ylabel('Frequency')
            
            # Plot 3: Component performance comparison
            component_metrics = {}
            for record in filtered_data:
                comp = record['component']
                if comp not in component_metrics:
                    component_metrics[comp] = []
                
                # Use first metric for comparison
                if record['metrics']:
                    first_metric = list(record['metrics'].values())[0]
                    component_metrics[comp].append(first_metric)
            
            if component_metrics:
                components = list(component_metrics.keys())
                means = [np.mean(component_metrics[comp]) for comp in components]
                axes[1, 1].bar(components, means)
                axes[1, 1].set_title('Component Performance Comparison')
                axes[1, 1].set_xlabel('Component')
                axes[1, 1].set_ylabel('Average Performance')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save or return path
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                # Return as base64 string for web display
                import io
                import base64
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=self.config.dpi, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return f"data:image/png;base64,{image_base64}"

class RealTimeMonitor:
    """Real-time monitoring system."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_queue = queue.Queue()
        self.metrics_buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        logger.info("📡 Real-time monitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("📡 Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("📡 Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                with self._lock:
                    self.metrics_buffer.append(metrics)
                    
                    # Limit buffer size
                    if len(self.metrics_buffer) > 1000:
                        self.metrics_buffer = self.metrics_buffer[-500:]
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.config.monitoring_interval * 2)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        import psutil
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_sent_mb': psutil.net_io_counters().bytes_sent / (1024**2),
            'network_recv_mb': psutil.net_io_counters().bytes_recv / (1024**2)
        }
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions."""
        for alert_name, threshold in self.config.alert_thresholds.items():
            if alert_name in metrics:
                if metrics[alert_name] > threshold:
                    alert = {
                        'timestamp': time.time(),
                        'alert_type': alert_name,
                        'value': metrics[alert_name],
                        'threshold': threshold,
                        'severity': 'high' if metrics[alert_name] > threshold * 1.5 else 'medium'
                    }
                    
                    self.alert_queue.put(alert)
                    logger.warning(f"🚨 Alert: {alert_name} = {metrics[alert_name]:.2f} (threshold: {threshold})")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        with self._lock:
            if self.metrics_buffer:
                return self.metrics_buffer[-1]
            return {}
    
    def get_alert_queue(self) -> List[Dict[str, Any]]:
        """Get all pending alerts."""
        alerts = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
            except queue.Empty:
                break
        
        return alerts

class AdvancedMonitoringAnalytics:
    """Main advanced monitoring and analytics system."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        
        # Initialize components
        self.experiment_tracker = ExperimentTracker(self.config)
        self.performance_analytics = PerformanceAnalytics(self.config)
        self.real_time_monitor = RealTimeMonitor(self.config)
        
        logger.info("📊 Advanced monitoring and analytics initialized")
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        if self.config.enable_real_time_monitoring:
            self.real_time_monitor.start_monitoring()
        
        logger.info("📊 All monitoring systems started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.real_time_monitor.stop_monitoring()
        self.experiment_tracker.stop_auto_save()
        
        logger.info("📊 All monitoring systems stopped")
    
    def start_experiment(self, name: str, description: str = "", 
                        config: Optional[Dict[str, Any]] = None,
                        hyperparameters: Optional[Dict[str, Any]] = None,
                        tags: Optional[List[str]] = None) -> str:
        """Start a new experiment."""
        return self.experiment_tracker.start_experiment(
            name, description, config, hyperparameters, tags
        )
    
    def log_metric(self, experiment_id: str, metric_name: str, metric_value: float):
        """Log a metric for an experiment."""
        self.experiment_tracker.log_metric(experiment_id, metric_name, metric_value)
    
    def complete_experiment(self, experiment_id: str, results: Optional[Dict[str, Any]] = None):
        """Complete an experiment."""
        self.experiment_tracker.complete_experiment(experiment_id, results)
    
    def record_performance(self, component: str, operation: str, 
                          metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        """Record performance metrics."""
        self.performance_analytics.record_performance(component, operation, metrics, metadata)
    
    def generate_comprehensive_report(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        report = {
            'timestamp': time.time(),
            'system_status': self._get_system_status(),
            'experiment_summary': self._get_experiment_summary(),
            'performance_summary': self._get_performance_summary(),
            'alerts': self.real_time_monitor.get_alert_queue()
        }
        
        if experiment_id:
            report['experiment_details'] = self._get_experiment_details(experiment_id)
        
        return report
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        current_metrics = self.real_time_monitor.get_current_metrics()
        return {
            'cpu_usage': current_metrics.get('cpu_percent', 0),
            'memory_usage': current_metrics.get('memory_percent', 0),
            'memory_available_gb': current_metrics.get('memory_available_gb', 0),
            'disk_usage': current_metrics.get('disk_usage_percent', 0)
        }
    
    def _get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        experiments = self.experiment_tracker.get_experiment_history(limit=10)
        
        return {
            'total_experiments': len(experiments),
            'active_experiments': len(self.experiment_tracker.active_experiments),
            'recent_experiments': experiments[:5]
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_analytics.generate_performance_report()
    
    def _get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed experiment information."""
        metrics_df = self.experiment_tracker.get_experiment_metrics(experiment_id)
        
        return {
            'experiment_id': experiment_id,
            'metrics_dataframe': metrics_df.to_dict('records') if not metrics_df.empty else [],
            'performance_report': self.performance_analytics.generate_performance_report()
        }
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data for monitoring dashboard."""
        return {
            'current_metrics': self.real_time_monitor.get_current_metrics(),
            'performance_visualization': self.performance_analytics.create_performance_visualization(),
            'experiment_history': self.experiment_tracker.get_experiment_history(limit=20),
            'alerts': self.real_time_monitor.get_alert_queue(),
            'system_status': self._get_system_status()
        }


# Factory functions
def create_advanced_monitoring_analytics(config: Optional[ExperimentConfig] = None) -> AdvancedMonitoringAnalytics:
    """Create advanced monitoring and analytics system."""
    return AdvancedMonitoringAnalytics(config)