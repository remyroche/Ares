"""
Financial Metrics Logger for Ares Trading System

This module provides a dedicated logger for financial metrics with timestamp-based
file naming and specialized formatting for trading and financial data.
"""

import logging
import os
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

# Import the main logger for fallback
try:
    from src.utils.logger import system_logger, get_logger
except ImportError:
    system_logger = None
    get_logger = lambda name: logging.getLogger(name)


@dataclass
class FinancialMetric:
    """Structure for individual financial metrics."""
    timestamp: str
    symbol: str
    exchange: str
    timeframe: str
    metric_name: str
    metric_value: float
    metric_type: str  # 'performance', 'risk', 'return', 'drawdown', 'sharpe', etc.
    step_name: str
    regime_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class TradingPerformanceMetrics:
    """Comprehensive trading performance metrics."""
    timestamp: str
    symbol: str
    exchange: str
    timeframe: str
    step_name: str
    
    # Performance Metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk Metrics
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    
    # Trading Metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Additional Context
    regime_id: Optional[str] = None
    model_version: Optional[str] = None
    confidence_score: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class FinancialMetricsLogger:
    """
    Dedicated logger for financial metrics with timestamp-based file naming.
    
    Features:
    - Timestamp-based log file naming
    - Structured financial data logging
    - CSV export capabilities
    - JSON export for complex metrics
    - Thread-safe operations
    - Integration with main logging system
    """
    
    def __init__(self, 
                 log_dir: str = "logs/financial_metrics",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_csv: bool = True,
                 enable_json: bool = True,
                 max_file_size_mb: int = 50,
                 backup_count: int = 10):
        """
        Initialize the financial metrics logger.
        
        Args:
            log_dir: Directory for financial metrics logs
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_csv: Enable CSV export
            enable_json: Enable JSON export
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_csv = enable_csv
        self.enable_json = enable_json
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize loggers
        self._setup_loggers()
        
        # CSV file handles (one per session)
        self._csv_handles = {}
        self._csv_writers = {}
        
        # JSON file handles
        self._json_handles = {}
        
        # Session tracking
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Track current file path for logging
        self.current_file_path = None
        
        # Fallback to main logger if available
        self.fallback_logger = system_logger.getChild('FinancialMetrics') if system_logger else None
    
    def _setup_loggers(self):
        """Setup the financial metrics loggers."""
        # Main financial metrics logger
        self.logger = logging.getLogger('FinancialMetrics')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with timestamp
        if self.enable_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f'financial_metrics_{timestamp}.log'
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=self.max_file_size, 
                backupCount=self.backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _get_csv_file_path(self, metric_type: str) -> Path:
        """Get CSV file path for a specific metric type."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.log_dir / f'financial_metrics_{metric_type}_{timestamp}.csv'
    
    def _get_json_file_path(self, metric_type: str) -> Path:
        """Get JSON file path for a specific metric type."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.log_dir / f'financial_metrics_{metric_type}_{timestamp}.json'
    
    def log_financial_metric(self, 
                           symbol: str,
                           exchange: str,
                           timeframe: str,
                           metric_name: str,
                           metric_value: float,
                           metric_type: str,
                           step_name: str,
                           regime_id: Optional[str] = None,
                           additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a single financial metric.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric (performance, risk, return, etc.)
            step_name: Training step name
            regime_id: Market regime identifier
            additional_data: Additional context data
        """
        with self._lock:
            try:
                timestamp = datetime.now().isoformat()
                
                # Create metric object
                metric = FinancialMetric(
                    timestamp=timestamp,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metric_type=metric_type,
                    step_name=step_name,
                    regime_id=regime_id,
                    additional_data=additional_data
                )
                
                # Log to console/file
                log_message = (
                    f"💰 {metric_type.upper()} | {symbol} | {step_name} | "
                    f"{metric_name}: {metric_value:.6f}"
                )
                if regime_id:
                    log_message += f" | Regime: {regime_id}"
                
                self.logger.info(log_message)
                
                # Log to CSV if enabled
                if self.enable_csv:
                    self._log_to_csv(metric)
                
                # Log to JSON if enabled
                if self.enable_json:
                    self._log_to_json(metric)
                
                # Fallback logging
                if self.fallback_logger:
                    self.fallback_logger.info(f"Financial metric logged: {metric_name}")
                
            except Exception as e:
                error_msg = f"Failed to log financial metric: {e}"
                self.logger.error(error_msg)
                if self.fallback_logger:
                    self.fallback_logger.error(error_msg)
    
    def log_trading_performance(self, 
                              symbol: str,
                              exchange: str,
                              timeframe: str,
                              step_name: str,
                              performance_data: Dict[str, Any],
                              regime_id: Optional[str] = None,
                              model_version: Optional[str] = None,
                              confidence_score: Optional[float] = None) -> None:
        """
        Log comprehensive trading performance metrics.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            step_name: Training step name
            performance_data: Dictionary containing performance metrics
            regime_id: Market regime identifier
            model_version: Model version identifier
            confidence_score: Confidence score for the metrics
        """
        with self._lock:
            try:
                timestamp = datetime.now().isoformat()
                
                # Create performance metrics object
                metrics = TradingPerformanceMetrics(
                    timestamp=timestamp,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    step_name=step_name,
                    
                    # Performance Metrics
                    total_return=performance_data.get('total_return', 0.0),
                    annualized_return=performance_data.get('annualized_return', 0.0),
                    volatility=performance_data.get('volatility', 0.0),
                    sharpe_ratio=performance_data.get('sharpe_ratio', 0.0),
                    sortino_ratio=performance_data.get('sortino_ratio', 0.0),
                    calmar_ratio=performance_data.get('calmar_ratio', 0.0),
                    
                    # Risk Metrics
                    max_drawdown=performance_data.get('max_drawdown', 0.0),
                    max_drawdown_duration=performance_data.get('max_drawdown_duration', 0),
                    var_95=performance_data.get('var_95', 0.0),
                    cvar_95=performance_data.get('cvar_95', 0.0),
                    
                    # Trading Metrics
                    win_rate=performance_data.get('win_rate', 0.0),
                    profit_factor=performance_data.get('profit_factor', 0.0),
                    avg_win=performance_data.get('avg_win', 0.0),
                    avg_loss=performance_data.get('avg_loss', 0.0),
                    largest_win=performance_data.get('largest_win', 0.0),
                    largest_loss=performance_data.get('largest_loss', 0.0),
                    total_trades=performance_data.get('total_trades', 0),
                    winning_trades=performance_data.get('winning_trades', 0),
                    losing_trades=performance_data.get('losing_trades', 0),
                    
                    # Additional Context
                    regime_id=regime_id,
                    model_version=model_version,
                    confidence_score=confidence_score,
                    additional_metrics=performance_data.get('additional_metrics', {})
                )
                
                # Log summary to console/file
                summary_message = (
                    f"📊 PERFORMANCE | {symbol} | {step_name} | "
                    f"Return: {metrics.total_return:.2%} | "
                    f"Sharpe: {metrics.sharpe_ratio:.2f} | "
                    f"MaxDD: {metrics.max_drawdown:.2%} | "
                    f"Win Rate: {metrics.win_rate:.1%}"
                )
                if regime_id:
                    summary_message += f" | Regime: {regime_id}"
                
                self.logger.info(summary_message)
                
                # Log detailed metrics
                self.logger.info(f"📈 Detailed Performance | {symbol} | {step_name}")
                self.logger.info(f"   Total Return: {metrics.total_return:.4f}")
                self.logger.info(f"   Annualized Return: {metrics.annualized_return:.4f}")
                self.logger.info(f"   Volatility: {metrics.volatility:.4f}")
                self.logger.info(f"   Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
                self.logger.info(f"   Sortino Ratio: {metrics.sortino_ratio:.4f}")
                self.logger.info(f"   Calmar Ratio: {metrics.calmar_ratio:.4f}")
                self.logger.info(f"   Max Drawdown: {metrics.max_drawdown:.4f}")
                self.logger.info(f"   VaR 95%: {metrics.var_95:.4f}")
                self.logger.info(f"   CVaR 95%: {metrics.cvar_95:.4f}")
                self.logger.info(f"   Win Rate: {metrics.win_rate:.4f}")
                self.logger.info(f"   Profit Factor: {metrics.profit_factor:.4f}")
                self.logger.info(f"   Total Trades: {metrics.total_trades}")
                self.logger.info(f"   Winning Trades: {metrics.winning_trades}")
                self.logger.info(f"   Losing Trades: {metrics.losing_trades}")
                
                # Log to CSV if enabled
                if self.enable_csv:
                    self._log_performance_to_csv(metrics)
                
                # Log to JSON if enabled
                if self.enable_json:
                    self._log_performance_to_json(metrics)
                
                # Fallback logging
                if self.fallback_logger:
                    self.fallback_logger.info(f"Trading performance logged for {symbol} in {step_name}")
                
            except Exception as e:
                error_msg = f"Failed to log trading performance: {e}"
                self.logger.error(error_msg)
                if self.fallback_logger:
                    self.fallback_logger.error(error_msg)
    
    def _log_to_csv(self, metric: FinancialMetric) -> None:
        """Log metric to CSV file."""
        try:
            csv_file = self._get_csv_file_path(metric.metric_type)
            
            # Check if file exists to determine if we need headers
            file_exists = csv_file.exists()
            
            # Track file path for logging
            if not file_exists:
                self.current_file_path = csv_file
                self.logger.info(f"📁 Creating new financial metrics CSV file: {csv_file}")
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header if file is new
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'symbol', 'exchange', 'timeframe', 'metric_name',
                        'metric_value', 'metric_type', 'step_name', 'regime_id', 'additional_data'
                    ])
                
                # Write metric data
                writer.writerow([
                    metric.timestamp,
                    metric.symbol,
                    metric.exchange,
                    metric.timeframe,
                    metric.metric_name,
                    metric.metric_value,
                    metric.metric_type,
                    metric.step_name,
                    metric.regime_id or '',
                    json.dumps(metric.additional_data) if metric.additional_data else ''
                ])
                
        except Exception as e:
            self.logger.error(f"Failed to write metric to CSV: {e}")
    
    def _log_to_json(self, metric: FinancialMetric) -> None:
        """Log metric to JSON file."""
        try:
            json_file = self._get_json_file_path(metric.metric_type)
            
            # Load existing data or create new list
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
                # Track file path for logging
                self.current_file_path = json_file
                self.logger.info(f"📁 Creating new financial metrics JSON file: {json_file}")
            
            # Add new metric
            data.append(asdict(metric))
            
            # Write back to file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to write metric to JSON: {e}")
    
    def _log_performance_to_csv(self, metrics: TradingPerformanceMetrics) -> None:
        """Log performance metrics to CSV file."""
        try:
            csv_file = self._get_csv_file_path('performance')
            
            # Check if file exists to determine if we need headers
            file_exists = csv_file.exists()
            
            # Track file path for logging
            if not file_exists:
                self.current_file_path = csv_file
                self.logger.info(f"📁 Creating new performance metrics CSV file: {csv_file}")
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header if file is new
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'symbol', 'exchange', 'timeframe', 'step_name',
                        'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
                        'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'max_drawdown_duration',
                        'var_95', 'cvar_95', 'win_rate', 'profit_factor', 'avg_win', 'avg_loss',
                        'largest_win', 'largest_loss', 'total_trades', 'winning_trades',
                        'losing_trades', 'regime_id', 'model_version', 'confidence_score',
                        'additional_metrics'
                    ])
                
                # Write metrics data
                writer.writerow([
                    metrics.timestamp,
                    metrics.symbol,
                    metrics.exchange,
                    metrics.timeframe,
                    metrics.step_name,
                    metrics.total_return,
                    metrics.annualized_return,
                    metrics.volatility,
                    metrics.sharpe_ratio,
                    metrics.sortino_ratio,
                    metrics.calmar_ratio,
                    metrics.max_drawdown,
                    metrics.max_drawdown_duration,
                    metrics.var_95,
                    metrics.cvar_95,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.avg_win,
                    metrics.avg_loss,
                    metrics.largest_win,
                    metrics.largest_loss,
                    metrics.total_trades,
                    metrics.winning_trades,
                    metrics.losing_trades,
                    metrics.regime_id or '',
                    metrics.model_version or '',
                    metrics.confidence_score or '',
                    json.dumps(metrics.additional_metrics) if metrics.additional_metrics else ''
                ])
                
        except Exception as e:
            self.logger.error(f"Failed to write performance metrics to CSV: {e}")
    
    def _log_performance_to_json(self, metrics: TradingPerformanceMetrics) -> None:
        """Log performance metrics to JSON file."""
        try:
            json_file = self._get_json_file_path('performance')
            
            # Load existing data or create new list
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
                # Track file path for logging
                self.current_file_path = json_file
                self.logger.info(f"📁 Creating new performance metrics JSON file: {json_file}")
            
            # Add new metrics
            data.append(asdict(metrics))
            
            # Write back to file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to write performance metrics to JSON: {e}")
    
    def log_step_start(self, step_name: str, symbol: str, exchange: str, timeframe: str) -> None:
        """Log the start of a training step."""
        with self._lock:
            try:
                timestamp = datetime.now().isoformat()
                message = f"🚀 STEP START | {step_name} | {symbol} | {exchange} | {timeframe} | {timestamp}"
                self.logger.info(message)
                
                if self.fallback_logger:
                    self.fallback_logger.info(f"Financial metrics logging started for {step_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to log step start: {e}")
    
    def log_step_end(self, step_name: str, symbol: str, exchange: str, timeframe: str, 
                    success: bool = True, error_message: Optional[str] = None) -> None:
        """Log the end of a training step."""
        with self._lock:
            try:
                timestamp = datetime.now().isoformat()
                status = "SUCCESS" if success else "FAILED"
                message = f"🏁 STEP END | {step_name} | {symbol} | {exchange} | {timeframe} | {status} | {timestamp}"
                
                if not success and error_message:
                    message += f" | Error: {error_message}"
                
                self.logger.info(message)
                
                if self.fallback_logger:
                    self.fallback_logger.info(f"Financial metrics logging ended for {step_name}: {status}")
                    
            except Exception as e:
                self.logger.error(f"Failed to log step end: {e}")
    
    def log_model_performance(self, 
                            symbol: str,
                            exchange: str,
                            timeframe: str,
                            step_name: str,
                            model_name: str,
                            model_version: str,
                            performance_metrics: Dict[str, float],
                            regime_id: Optional[str] = None) -> None:
        """Log model-specific performance metrics."""
        with self._lock:
            try:
                timestamp = datetime.now().isoformat()
                
                # Log model performance summary
                summary_message = (
                    f"🤖 MODEL PERFORMANCE | {model_name} v{model_version} | {symbol} | {step_name}"
                )
                if regime_id:
                    summary_message += f" | Regime: {regime_id}"
                
                self.logger.info(summary_message)
                
                # Log individual metrics
                for metric_name, metric_value in performance_metrics.items():
                    self.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name=f"{model_name}_{metric_name}",
                        metric_value=metric_value,
                        metric_type="model_performance",
                        step_name=step_name,
                        regime_id=regime_id,
                        additional_data={
                            "model_name": model_name,
                            "model_version": model_version,
                            "original_metric_name": metric_name
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"Failed to log model performance: {e}")
    
    def export_metrics_summary(self, 
                             symbol: Optional[str] = None,
                             step_name: Optional[str] = None,
                             metric_type: Optional[str] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Export a summary of logged metrics.
        
        Args:
            symbol: Filter by symbol
            step_name: Filter by step name
            metric_type: Filter by metric type
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            Dictionary containing summary statistics
        """
        try:
            summary = {
                "export_timestamp": datetime.now().isoformat(),
                "filters": {
                    "symbol": symbol,
                    "step_name": step_name,
                    "metric_type": metric_type,
                    "start_date": start_date,
                    "end_date": end_date
                },
                "summary_statistics": {},
                "file_locations": {
                    "log_directory": str(self.log_dir),
                    "csv_files": [],
                    "json_files": []
                }
            }
            
            # Find relevant files
            csv_files = list(self.log_dir.glob("financial_metrics_*.csv"))
            json_files = list(self.log_dir.glob("financial_metrics_*.json"))
            
            summary["file_locations"]["csv_files"] = [str(f) for f in csv_files]
            summary["file_locations"]["json_files"] = [str(f) for f in json_files]
            
            # Basic statistics
            summary["summary_statistics"] = {
                "total_csv_files": len(csv_files),
                "total_json_files": len(json_files),
                "log_directory_size_mb": sum(f.stat().st_size for f in self.log_dir.glob("*") if f.is_file()) / (1024 * 1024)
            }
            
            self.logger.info(f"📊 Metrics summary exported: {summary['summary_statistics']}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics summary: {e}")
            return {"error": str(e)}
    
    def close(self) -> None:
        """Close the financial metrics logger and clean up resources."""
        with self._lock:
            try:
                # Close CSV handles
                for handle in self._csv_handles.values():
                    if not handle.closed:
                        handle.close()
                
                # Close JSON handles
                for handle in self._json_handles.values():
                    if not handle.closed:
                        handle.close()
                
                # Clear handlers
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
                
                self.logger.info("🔒 Financial metrics logger closed successfully")
                
            except Exception as e:
                if self.fallback_logger:
                    self.fallback_logger.error(f"Error closing financial metrics logger: {e}")


# Global instance
_financial_metrics_logger: Optional[FinancialMetricsLogger] = None


def get_financial_metrics_logger() -> FinancialMetricsLogger:
    """Get the global financial metrics logger instance."""
    global _financial_metrics_logger
    if _financial_metrics_logger is None:
        _financial_metrics_logger = FinancialMetricsLogger()
    return _financial_metrics_logger


def setup_financial_metrics_logging(log_dir: str = "logs/financial_metrics", **kwargs) -> FinancialMetricsLogger:
    """Setup the global financial metrics logger."""
    global _financial_metrics_logger
    _financial_metrics_logger = FinancialMetricsLogger(log_dir=log_dir, **kwargs)
    return _financial_metrics_logger


@contextmanager
def financial_metrics_context(step_name: str, symbol: str, exchange: str, timeframe: str):
    """Context manager for financial metrics logging within a training step."""
    logger = get_financial_metrics_logger()
    
    try:
        logger.log_step_start(step_name, symbol, exchange, timeframe)
        yield logger
        logger.log_step_end(step_name, symbol, exchange, timeframe, success=True)
    except Exception as e:
        logger.log_step_end(step_name, symbol, exchange, timeframe, success=False, error_message=str(e))
        raise


# Convenience functions for common operations
def log_return_metric(symbol: str, exchange: str, timeframe: str, step_name: str, 
                     return_value: float, regime_id: Optional[str] = None) -> None:
    """Log a return metric."""
    logger = get_financial_metrics_logger()
    logger.log_financial_metric(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name="return",
        metric_value=return_value,
        metric_type="return",
        step_name=step_name,
        regime_id=regime_id
    )


def log_risk_metric(symbol: str, exchange: str, timeframe: str, step_name: str,
                   risk_value: float, risk_type: str = "volatility", regime_id: Optional[str] = None) -> None:
    """Log a risk metric."""
    logger = get_financial_metrics_logger()
    logger.log_financial_metric(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name=risk_type,
        metric_value=risk_value,
        metric_type="risk",
        step_name=step_name,
        regime_id=regime_id
    )


def log_sharpe_ratio(symbol: str, exchange: str, timeframe: str, step_name: str,
                    sharpe_value: float, regime_id: Optional[str] = None) -> None:
    """Log a Sharpe ratio metric."""
    logger = get_financial_metrics_logger()
    logger.log_financial_metric(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name="sharpe_ratio",
        metric_value=sharpe_value,
        metric_type="performance",
        step_name=step_name,
        regime_id=regime_id
    )


def log_drawdown_metric(symbol: str, exchange: str, timeframe: str, step_name: str,
                       drawdown_value: float, regime_id: Optional[str] = None) -> None:
    """Log a drawdown metric."""
    logger = get_financial_metrics_logger()
    logger.log_financial_metric(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name="max_drawdown",
        metric_value=drawdown_value,
        metric_type="risk",
        step_name=step_name,
        regime_id=regime_id
    )


# Export main classes and functions
__all__ = [
    'FinancialMetricsLogger',
    'FinancialMetric',
    'TradingPerformanceMetrics',
    'get_financial_metrics_logger',
    'setup_financial_metrics_logging',
    'financial_metrics_context',
    'log_return_metric',
    'log_risk_metric',
    'log_sharpe_ratio',
    'log_drawdown_metric'
]