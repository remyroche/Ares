from ..standardized_parquet_handler import standardized_parquet_handler
"""Enhanced Logging System for Backtesting Pipeline.

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

This module provides comprehensive logging with emojis, progress tracking,
quality assessment, and detailed error reporting for the backtesting pipeline.
"""
import logging
import time
import sys
from pathlib import Path
import traceback
import psutil
import threading
from contextlib import contextmanager
from src.utils.common_operations import format_datetime, get_current_datetime, safe_file_exists, ensure_directory, safe_json_dump, safe_json_load
import json
import numpy as np
import typing

class BacktestingLogger:
    """Enhanced logger for backtesting pipeline with comprehensive monitoring."""
    @log_important_calls

    def __init__(self, name: str, log_dir: str='log', enable_console: bool = True) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self.enable_console = enable_console
        ensure_directory(self.log_dir)
        self.logger = logging.getLogger(f'backtesting.{name}')
        self.logger.setLevel(logging.DEBUG)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'backtesting_{name}_{timestamp}.log'
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%H:%M:%S')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        self.start_time = time.time()
        self.step_times = {}
        self.quality_flags = []
        self.errors = []
        self.warnings = []
        self.progress_data = {}
        self.performance_metrics = {}
        self.monitor_thread = None
        self.monitoring = False
        self.logger.info('🚀 Enhanced Backtesting Logger Initialized')
        self.logger.info(f'📁 Log file: {log_file}')
        self.logger.info(f"🖥️ Console output: {('Enabled' if self.enable_console else 'Disabled')}")

    def start_performance_monitoring(self, interval: float = 5.0) -> None:
        """Start performance monitoring in background thread."""
        if self.monitoring:
            return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target = self._monitor_performance, args=(interval,), daemon = True)
        self.monitor_thread.start()
        self.logger.info(f'📊 Performance monitoring started (interval: {interval}s)')

    def stop_performance_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout = 1.0)
        self.logger.info('📊 Performance monitoring stopped')
    @log_all_calls

    def _monitor_performance(self, interval: float) -> None:
        """Background performance monitoring."""
        while self.monitoring:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                timestamp = time.time()
                self.performance_metrics[timestamp] = {'memory_mb': memory_info.rss / 1024 / 1024, 'cpu_percent': cpu_percent, 'elapsed_time': timestamp - self.start_time}
                if memory_info.rss / 1024 / 1024 > 1000:
                    self.logger.warning(f'⚠️ High memory usage: {memory_info.rss / 1024 / 1024:.1f} MB')
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f'❌ Performance monitoring error: {e}')
                break

    @contextmanager
    def step_timer(self, step_name: str) -> None:
        """Context manager for timing steps."""
        start_time = time.time()
        self.logger.info(f'🔄 Starting step: {step_name}')
        try:
            yield
            elapsed = time.time() - start_time
            self.step_times[step_name] = elapsed
            self.logger.info(f'✅ Step completed: {step_name} ({elapsed:.2f}s)')
        except Exception as e:
            elapsed = time.time() - start_time
            self.step_times[step_name] = elapsed
            self.logger.error(f'❌ Step failed: {step_name} ({elapsed:.2f}s) - {e}')
            raise

    def log_progress(self, step: str, progress: float, message: str='') -> None:
        """Log progress with visual indicator."""
        progress_bar = self._create_progress_bar(progress)
        if message:
            self.logger.info(f'📈 {step}: {progress_bar} {progress:.1f}% - {message}')
        else:
            self.logger.info(f'📈 {step}: {progress_bar} {progress:.1f}%')
        self.progress_data[step] = {'progress': progress, 'message': message, 'timestamp': time.time()}
    @log_all_calls

    def _create_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a visual progress bar."""
        filled = int(width * progress / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f'[{bar}]'

    def log_quality_flag(self, flag_type: str, message: str, severity: str='WARNING') -> None:
        """Log quality flags for issue detection."""
        flag_data = {'type': flag_type, 'message': message, 'severity': severity, 'timestamp': time.time()}
        self.quality_flags.append(flag_data)
        emoji = '⚠️' if severity == 'WARNING' else '❌' if severity == 'ERROR' else 'ℹ️'
        self.logger.warning(f'{emoji} Quality Flag [{flag_type}]: {message}')

    def log_error(self, error: Exception, context: str='') -> None:
        """Log errors with detailed context."""
        error_data = {'type': type(error).__name__, 'message': str(error), 'context': context, 'timestamp': time.time(), 'traceback': traceback.format_exc()}
        self.errors.append(error_data)
        self.logger.error(f'❌ Error in {context}: {error}')
        self.logger.debug(f'📋 Error traceback: {traceback.format_exc()}')

    def log_warning(self, message: str, context: str='') -> None:
        """Log warnings with context."""
        warning_data = {'message': message, 'context': context, 'timestamp': time.time()}
        self.warnings.append(warning_data)
        self.logger.warning(f'⚠️ Warning in {context}: {message}')

    def log_success(self, message: str, context: str='') -> None:
        """Log success messages - only use emoji for step completion."""
        if 'completed' in message.lower() or 'finished' in message.lower():
            self.logger.info(f'✅ {context}: {message}')
        else:
            self.logger.info(f'Success in {context}: {message}')

    def log_info(self, message: str, context: str='') -> None:
        """Log info messages - no emojis for normal operations."""
        if context:
            self.logger.info(f'{context}: {message}')
        else:
            self.logger.info(message)

    def log_debug(self, message: str, context: str='') -> None:
        """Log debug messages - no emojis for normal operations."""
        if context:
            self.logger.debug(f'{context}: {message}')
        else:
            self.logger.debug(message)

    def log_data_quality(self, data_info: Dict[str, Any]) -> None:
        """Log data quality assessment."""
        self.logger.info('Data Quality Assessment:')
        for key, value in data_info.items():
            if isinstance(value, (int, float)):
                self.logger.info(f'   • {key}: {value:,}')
            else:
                self.logger.info(f'   • {key}: {value}')
        if data_info.get('missing_percentage', 0) > 5:
            self.log_quality_flag('DATA_QUALITY', f"High missing data percentage: {data_info.get('missing_percentage', 0):.1f}%", 'WARNING')
        if data_info.get('duplicate_count', 0) > 0:
            self.log_quality_flag('DATA_QUALITY', f"Duplicate records found: {data_info.get('duplicate_count', 0)}", 'WARNING')

    def log_validation_result(self, step: str, passed: bool, details: Dict[str, Any]) -> None:
        """Log validation results."""
        if passed:
            self.logger.info(f'Validation passed: {step}')
        else:
            self.logger.error(f'❌ Validation failed: {step}')
            self.log_quality_flag('VALIDATION', f'Validation failed for {step}', 'ERROR')
        for key, value in details.items():
            self.logger.info(f'   • {key}: {value}')

    def log_backtesting_metrics(self, metrics: Dict[str, Any], regime: str='Overall') -> None:
        """Log backtesting performance metrics."""
        self.logger.info(f'Backtesting Metrics - {regime}:')
        if 'total_return' in metrics:
            self.logger.info(f"   • Total Return: {metrics['total_return']:.2%}")
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            if sharpe < 1.0:
                self.log_quality_flag('PERFORMANCE', f'Low Sharpe ratio: {sharpe:.2f}', 'WARNING')
            self.logger.info(f'   • Sharpe Ratio: {sharpe:.2f}')
        if 'win_rate' in metrics:
            win_rate = metrics['win_rate']
            if win_rate < 0.5:
                self.log_quality_flag('PERFORMANCE', f'Low win rate: {win_rate:.2%}', 'WARNING')
            self.logger.info(f'   • Win Rate: {win_rate:.2%}')
        if 'max_drawdown' in metrics:
            max_dd = metrics['max_drawdown']
            if max_dd > 0.2:
                self.log_quality_flag('PERFORMANCE', f'High max drawdown: {max_dd:.2%}', 'WARNING')
            self.logger.info(f'   • Max Drawdown: {max_dd:.2%}')
        if 'total_trades' in metrics:
            self.logger.info(f"   • Total Trades: {metrics['total_trades']:,}")
        if 'avg_trade_return' in metrics:
            self.logger.info(f"   • Avg Trade Return: {metrics['avg_trade_return']:.2%}")
        if 'profit_factor' in metrics:
            pf = metrics['profit_factor']
            if pf < 1.2:
                self.log_quality_flag('PERFORMANCE', f'Low profit factor: {pf:.2f}', 'WARNING')
            self.logger.info(f'   • Profit Factor: {pf:.2f}')
        if 'volatility' in metrics:
            self.logger.info(f"   • Volatility: {metrics['volatility']:.2%}")
        if 'var_95' in metrics:
            self.logger.info(f"   • VaR (95%): {metrics['var_95']:.2%}")
        if 'calmar_ratio' in metrics:
            self.logger.info(f"   • Calmar Ratio: {metrics['calmar_ratio']:.2f}")

    def log_regime_analysis(self, regime_results: Dict[str, Any]) -> None:
        """Log analysis results for each market regime."""
        self.logger.info('Market Regime Analysis:')
        for regime, results in regime_results.items():
            self.logger.info(f'  {regime}:')
            if 'regime_duration' in results:
                self.logger.info(f"    • Duration: {results['regime_duration']:.1f} days")
            if 'regime_frequency' in results:
                self.logger.info(f"    • Frequency: {results['regime_frequency']:.1%}")
            if 'regime_return' in results:
                regime_return = results['regime_return']
                if regime_return < 0:
                    self.log_quality_flag('REGIME_PERFORMANCE', f'Negative returns in {regime}: {regime_return:.2%}', 'WARNING')
                self.logger.info(f'    • Regime Return: {regime_return:.2%}')
            if 'regime_sharpe' in results:
                regime_sharpe = results['regime_sharpe']
                if regime_sharpe < 0.5:
                    self.log_quality_flag('REGIME_PERFORMANCE', f'Low Sharpe in {regime}: {regime_sharpe:.2f}', 'WARNING')
                self.logger.info(f'    • Regime Sharpe: {regime_sharpe:.2f}')
            if 'regime_trades' in results:
                self.logger.info(f"    • Trades in Regime: {results['regime_trades']:,}")

    def log_model_performance(self, model_results: Dict[str, Any]) -> None:
        """Log model performance metrics."""
        self.logger.info('Model Performance Analysis:')
        for model_name, results in model_results.items():
            self.logger.info(f'  {model_name}:')
            if 'accuracy' in results:
                accuracy = results['accuracy']
                if accuracy < 0.6:
                    self.log_quality_flag('MODEL_PERFORMANCE', f'Low accuracy for {model_name}: {accuracy:.2%}', 'WARNING')
                self.logger.info(f'    • Accuracy: {accuracy:.2%}')
            if 'precision' in results:
                self.logger.info(f"    • Precision: {results['precision']:.2%}")
            if 'recall' in results:
                self.logger.info(f"    • Recall: {results['recall']:.2%}")
            if 'f1_score' in results:
                self.logger.info(f"    • F1 Score: {results['f1_score']:.2%}")
            if 'avg_confidence' in results:
                confidence = results['avg_confidence']
                if confidence < 0.7:
                    self.log_quality_flag('MODEL_PERFORMANCE', f'Low confidence for {model_name}: {confidence:.2%}', 'WARNING')
                self.logger.info(f'    • Avg Confidence: {confidence:.2%}')
            if 'top_features' in results:
                self.logger.info(f"    • Top Features: {', '.join(results['top_features'][:3])}")

    def log_risk_metrics(self, risk_metrics: Dict[str, Any]) -> None:
        """Log comprehensive risk metrics."""
        self.logger.info('Risk Analysis:')
        if 'portfolio_var' in risk_metrics:
            var = risk_metrics['portfolio_var']
            if var > 0.05:
                self.log_quality_flag('RISK', f'High portfolio VaR: {var:.2%}', 'WARNING')
            self.logger.info(f'   • Portfolio VaR: {var:.2%}')
        if 'portfolio_es' in risk_metrics:
            es = risk_metrics['portfolio_es']
            if es > 0.08:
                self.log_quality_flag('RISK', f'High Expected Shortfall: {es:.2%}', 'WARNING')
            self.logger.info(f'   • Expected Shortfall: {es:.2%}')
        if 'concentration_risk' in risk_metrics:
            conc_risk = risk_metrics['concentration_risk']
            if conc_risk > 0.3:
                self.log_quality_flag('RISK', f'High concentration risk: {conc_risk:.2%}', 'WARNING')
            self.logger.info(f'   • Concentration Risk: {conc_risk:.2%}')
        if 'liquidity_risk' in risk_metrics:
            liq_risk = risk_metrics['liquidity_risk']
            if liq_risk > 0.1:
                self.log_quality_flag('RISK', f'High liquidity risk: {liq_risk:.2%}', 'WARNING')
            self.logger.info(f'   • Liquidity Risk: {liq_risk:.2%}')
        if 'correlation_risk' in risk_metrics:
            corr_risk = risk_metrics['correlation_risk']
            if corr_risk > 0.8:
                self.log_quality_flag('RISK', f'High correlation risk: {corr_risk:.2%}', 'WARNING')
            self.logger.info(f'   • Correlation Risk: {corr_risk:.2%}')

    def log_performance_summary(self) -> None:
        """Log performance summary."""
        total_time = time.time() - self.start_time
        self.logger.info('📊 Performance Summary:')
        self.logger.info(f'   • Total execution time: {total_time:.2f}s')
        self.logger.info(f'   • Quality flags: {len(self.quality_flags)}')
        self.logger.info(f'   • Errors: {len(self.errors)}')
        self.logger.info(f'   • Warnings: {len(self.warnings)}')
        if self.step_times:
            self.logger.info('   • Step execution times:')
            for step, time_taken in self.step_times.items():
                self.logger.info(f'     - {step}: {time_taken:.2f}s')
        if self.performance_metrics:
            latest_metrics = max(self.performance_metrics.values(), key=lambda x: x['elapsed_time'])
            self.logger.info(f"   • Peak memory usage: {latest_metrics['memory_mb']:.1f} MB")
            self.logger.info(f"   • Peak CPU usage: {latest_metrics['cpu_percent']:.1f}%")

    def generate_report(self, output_file: Optional[str]=None) -> Dict[str, Any]:
        """Generate comprehensive report."""
        total_time = time.time() - self.start_time
        report = {'execution_summary': {'total_time_seconds': total_time, 'start_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'), 'end_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'), 'logger_name': self.name}, 'step_times': self.step_times, 'progress_data': self.progress_data, 'quality_flags': self.quality_flags, 'errors': self.errors, 'warnings': self.warnings, 'performance_metrics': self.performance_metrics, 'quality_assessment': self._assess_overall_quality()}
        if output_file:
            safe_json_dump(report, output_file, indent = 2)
            self.logger.info(f'📋 Report saved to: {output_file}')
        return report
    @log_all_calls

    def _assess_overall_quality(self) -> Dict[str, Any]:
        """Assess overall quality of the execution."""
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        quality_flag_count = len(self.quality_flags)
        if error_count > 0:
            quality_level = 'POOR'
        elif quality_flag_count > 5 or warning_count > 10:
            quality_level = 'FAIR'
        elif quality_flag_count > 0 or warning_count > 0:
            quality_level = 'GOOD'
        else:
            quality_level = 'EXCELLENT'
        return {'quality_level': quality_level, 'error_count': error_count, 'warning_count': warning_count, 'quality_flag_count': quality_flag_count, 'recommendations': self._generate_recommendations()}
    @log_all_calls

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        if len(self.errors) > 0:
            recommendations.append('🔧 Address all errors before proceeding')
        if len(self.quality_flags) > 5:
            recommendations.append('⚠️ Review quality flags and consider data preprocessing')
        if len(self.warnings) > 10:
            recommendations.append('📊 Review warnings and optimize configuration')
        if self.performance_metrics:
            latest_metrics = max(self.performance_metrics.values(), key=lambda x: x['elapsed_time'])
            if latest_metrics['memory_mb'] > 2000:
                recommendations.append('💾 Consider optimizing memory usage')
            if latest_metrics['cpu_percent'] > 90:
                recommendations.append('⚡ Consider optimizing CPU usage')
        if not recommendations:
            recommendations.append('✅ No issues detected - execution quality is excellent')
        return recommendations
    @log_important_calls

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_performance_monitoring()
        self.logger.info('🧹 Backtesting logger cleanup completed')
_global_logger = None

def get_backtesting_logger(name: str='pipeline', log_dir: str='log') -> BacktestingLogger:
    """Get or create global backtesting logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = BacktestingLogger(name, log_dir)
    return _global_logger

def cleanup_global_logger() -> None:
    """Cleanup global logger."""
    global _global_logger
    if _global_logger:
        _global_logger.cleanup()
        _global_logger = None

"""Enhanced Logging System for Backtesting Pipeline.

This module provides comprehensive logging with emojis, progress tracking,
and structured output for backtesting operations.
"""