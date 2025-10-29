"""
Data Quality Metrics Tracking

Tracks data quality metrics over time for monitoring and alerting.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error

logger = system_logger.getChild('DataQualityMetrics')

@dataclass
class QualityMetric:
    """Individual quality metric."""
    timestamp: datetime
    symbol: str
    quality_score: float
    validation_errors: int
    validation_warnings: int
    failed_rules: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualitySummary:
    """Quality summary for a time period."""
    symbol: str
    period_start: datetime
    period_end: datetime
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    total_errors: int
    total_warnings: int
    error_rate: float
    samples: int


class DataQualityMetricsTracker:
    """
    Track data quality metrics over time.
    
    Provides metrics for monitoring, alerting, and quality analysis.
    """
    
    def __init__(self, max_history: int = 10000) -> None:
        """
        Initialize quality metrics tracker.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        tprint_info(f"🔍 Initializing Data Quality Metrics Tracker (max_history: {max_history})")
        self.max_history: int = max_history
        self.logger = logger.getChild('Tracker')
        
        # Per-symbol metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        
        # Aggregated statistics
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Alert thresholds
        self.quality_threshold_warning: float = 0.7
        self.quality_threshold_error: float = 0.5
        self.error_rate_threshold: float = 0.1  # 10% error rate
        tprint_info("✅ Data Quality Metrics Tracker initialized")
    
    def record_metric(self, metric: QualityMetric) -> None:
        """
        Record a quality metric.
        
        Args:
            metric: Quality metric to record
        """
        try:
            symbol: str = metric.symbol
            self._metrics[symbol].append(metric)
            
            # Update statistics
            self._update_stats(symbol)
            
            # Check for alerts
            self._check_alerts(metric)
            
            tprint_info(f"📊 Recorded quality metric for {symbol}: score={metric.quality_score:.3f}, errors={metric.validation_errors}, warnings={metric.validation_warnings}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to record metric: {e}")
            self.logger.error(f"❌ Failed to record metric: {e}")
    
    def _update_stats(self, symbol: str) -> None:
        """Update statistics for a symbol."""
        metrics: List[QualityMetric] = list(self._metrics[symbol])
        if not metrics:
            return
        
        scores: List[float] = [m.quality_score for m in metrics]
        errors: List[int] = [m.validation_errors for m in metrics]
        warnings: List[int] = [m.validation_warnings for m in metrics]
        
        avg_score: float = sum(scores) / len(scores)
        error_rate: float = sum(errors) / len(metrics) if metrics else 0.0
        
        self._stats[symbol] = {
            'avg_quality_score': avg_score,
            'min_quality_score': min(scores),
            'max_quality_score': max(scores),
            'total_samples': len(metrics),
            'total_errors': sum(errors),
            'total_warnings': sum(warnings),
            'error_rate': error_rate,
            'last_update': datetime.now(timezone.utc)
        }
        
        if error_rate > self.error_rate_threshold:
            tprint_warning(f"⚠️ High error rate for {symbol}: {error_rate:.2%} (threshold: {self.error_rate_threshold:.2%})")
    
    def _check_alerts(self, metric: QualityMetric) -> None:
        """Check if metric triggers alerts."""
        if metric.quality_score < self.quality_threshold_error:
            tprint_error(f"🔴 Low quality score for {metric.symbol}: {metric.quality_score:.2f}")
            self.logger.error(
                f"🔴 Low quality score for {metric.symbol}: {metric.quality_score:.2f}"
            )
        
        if metric.quality_score < self.quality_threshold_warning:
            tprint_warning(f"⚠️ Quality warning for {metric.symbol}: {metric.quality_score:.2f}")
            self.logger.warning(
                f"⚠️ Quality warning for {metric.symbol}: {metric.quality_score:.2f}"
            )
        
        if metric.validation_errors > 0:
            tprint_warning(f"⚠️ Validation errors for {metric.symbol}: {metric.validation_errors}")
            self.logger.warning(
                f"⚠️ Validation errors for {metric.symbol}: {metric.validation_errors}"
            )
    
    def get_summary(
        self,
        symbol: str,
        period_hours: int = 24
    ) -> Optional[QualitySummary]:
        """
        Get quality summary for a symbol over a time period.
        
        Args:
            symbol: Trading symbol
            period_hours: Number of hours to summarize
            
        Returns:
            QualitySummary or None
        """
        metrics = list(self._metrics[symbol])
        if not metrics:
            return None
        
        cutoff_time: datetime = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        recent_metrics: List[QualityMetric] = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            tprint_info(f"⚠️ No metrics found for {symbol} in the last {period_hours} hours")
            return None
        
        scores: List[float] = [m.quality_score for m in recent_metrics]
        errors: List[int] = [m.validation_errors for m in recent_metrics]
        warnings: List[int] = [m.validation_warnings for m in recent_metrics]
        
        tprint_info(f"📊 Generated quality summary for {symbol}: {len(recent_metrics)} samples over {period_hours}h")
        
        return QualitySummary(
            symbol=symbol,
            period_start=cutoff_time,
            period_end=datetime.now(timezone.utc),
            avg_quality_score=sum(scores) / len(scores),
            min_quality_score=min(scores),
            max_quality_score=max(scores),
            total_errors=sum(errors),
            total_warnings=sum(warnings),
            error_rate=sum(errors) / len(recent_metrics) if recent_metrics else 0.0,
            samples=len(recent_metrics)
        )
    
    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for symbol(s).
        
        Args:
            symbol: Trading symbol, or None for all symbols
            
        Returns:
            Dictionary of statistics
        """
        if symbol:
            return self._stats.get(symbol, {})
        return dict(self._stats)
    
    def get_recent_metrics(self, symbol: str, count: int = 100) -> List[QualityMetric]:
        """
        Get recent quality metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            count: Number of metrics to return
            
        Returns:
            List of quality metrics
        """
        metrics = list(self._metrics[symbol])
        return metrics[-count:] if len(metrics) > count else metrics
    
    def get_quality_trend(self, symbol: str, periods: int = 10) -> List[float]:
        """
        Get quality score trend over recent periods.
        
        Args:
            symbol: Trading symbol
            periods: Number of periods to analyze
            
        Returns:
            List of average quality scores per period
        """
        metrics = list(self._metrics[symbol])
        if not metrics:
            return []
        
        # Split into periods
        period_size = max(1, len(metrics) // periods)
        trends = []
        
        for i in range(0, len(metrics), period_size):
            period_metrics = metrics[i:i + period_size]
            avg_score = sum(m.quality_score for m in period_metrics) / len(period_metrics)
            trends.append(avg_score)
        
        return trends
    
    def clear_old_metrics(self, days: int = 30) -> None:
        """
        Clear metrics older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff_time: datetime = datetime.now(timezone.utc) - timedelta(days=days)
        cleared_count: int = 0
        
        for symbol in list(self._metrics.keys()):
            before_count: int = len(self._metrics[symbol])
            self._metrics[symbol] = deque(
                (m for m in self._metrics[symbol] if m.timestamp >= cutoff_time),
                maxlen=self.max_history
            )
            after_count: int = len(self._metrics[symbol])
            cleared_count += (before_count - after_count)
        
        tprint_info(f"🧹 Cleared {cleared_count} metrics older than {days} days")
        self.logger.info(f"🧹 Cleared metrics older than {days} days")
    
    def export_metrics(self, symbol: str, file_path: str) -> None:
        """
        Export metrics to CSV file.
        
        Args:
            symbol: Trading symbol
            file_path: Output file path
        """
        try:
            import pandas as pd
            
            metrics: List[QualityMetric] = list(self._metrics[symbol])
            if not metrics:
                tprint_warning(f"⚠️ No metrics to export for {symbol}")
                self.logger.warning(f"No metrics to export for {symbol}")
                return
            
            data: List[Dict[str, Any]] = []
            for m in metrics:
                data.append({
                    'timestamp': m.timestamp,
                    'symbol': m.symbol,
                    'quality_score': m.quality_score,
                    'validation_errors': m.validation_errors,
                    'validation_warnings': m.validation_warnings,
                    'failed_rules': ','.join(m.failed_rules)
                })
            
            df: pd.DataFrame = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            tprint_success(f"✅ Exported {len(metrics)} metrics for {symbol} to {file_path}")
            self.logger.info(f"✅ Exported {len(metrics)} metrics to {file_path}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to export metrics: {e}")
            self.logger.error(f"❌ Failed to export metrics: {e}")
