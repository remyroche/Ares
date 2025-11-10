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
from src.utils.tprint import tprint

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
        tprint(f"DataQualityMetricsTracker.__init__: max_history={max_history}")
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
        tprint(f"DataQualityMetricsTracker.__init__: Initialized successfully")
    
    def record_metric(self, metric: QualityMetric) -> None:
        """
        Record a quality metric.

        Args:
            metric: Quality metric to record
        """
        symbol: str = metric.symbol
        tprint(f"record_metric: symbol={symbol}, quality_score={metric.quality_score:.3f}")
        try:
            self._metrics[symbol].append(metric)
            tprint(f"record_metric: Appended metric to buffer")

            # Update statistics
            self._update_stats(symbol)

            # Check for alerts
            self._check_alerts(metric)

            tprint(f"record_metric: Recorded metric for {symbol}: score={metric.quality_score:.3f}, errors={metric.validation_errors}, warnings={metric.validation_warnings}")

        except Exception as e:
            tprint(f"record_metric: Failed to record metric: {e}")
            self.logger.error(f"Failed to record metric: {e}")
    
    def _update_stats(self, symbol: str) -> None:
        """Update statistics for a symbol."""
        tprint(f"_update_stats: symbol={symbol}")
        metrics: List[QualityMetric] = list(self._metrics[symbol])
        if not metrics:
            tprint(f"_update_stats: No metrics for {symbol}, returning")
            return

        scores: List[float] = [m.quality_score for m in metrics]
        errors: List[int] = [m.validation_errors for m in metrics]
        warnings: List[int] = [m.validation_warnings for m in metrics]

        avg_score: float = sum(scores) / len(scores)
        error_rate: float = sum(errors) / len(metrics) if metrics else 0.0
        tprint(f"_update_stats: avg_score={avg_score:.3f}, error_rate={error_rate:.3f}")

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
            tprint(f"_update_stats: High error rate for {symbol}: {error_rate:.2%} (threshold: {self.error_rate_threshold:.2%})")
        tprint(f"_update_stats: Stats updated for {symbol}")
    
    def _check_alerts(self, metric: QualityMetric) -> None:
        """Check if metric triggers alerts."""
        tprint(f"_check_alerts: symbol={metric.symbol}, quality_score={metric.quality_score:.2f}")
        if metric.quality_score < self.quality_threshold_error:
            tprint(f"_check_alerts: Low quality score for {metric.symbol}: {metric.quality_score:.2f}")
            self.logger.error(
                f"Low quality score for {metric.symbol}: {metric.quality_score:.2f}"
            )

        if metric.quality_score < self.quality_threshold_warning:
            tprint(f"_check_alerts: Quality warning for {metric.symbol}: {metric.quality_score:.2f}")
            self.logger.warning(
                f"Quality warning for {metric.symbol}: {metric.quality_score:.2f}"
            )

        if metric.validation_errors > 0:
            tprint(f"_check_alerts: Validation errors for {metric.symbol}: {metric.validation_errors}")
            self.logger.warning(
                f"Validation errors for {metric.symbol}: {metric.validation_errors}"
            )
        tprint(f"_check_alerts: Alert check complete for {metric.symbol}")
    
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
        tprint(f"get_summary: symbol={symbol}, period_hours={period_hours}")
        metrics = list(self._metrics[symbol])
        if not metrics:
            tprint(f"get_summary: No metrics for {symbol}, returning None")
            return None

        cutoff_time: datetime = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        recent_metrics: List[QualityMetric] = [m for m in metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            tprint(f"get_summary: No metrics found for {symbol} in the last {period_hours} hours, returning None")
            return None

        scores: List[float] = [m.quality_score for m in recent_metrics]
        errors: List[int] = [m.validation_errors for m in recent_metrics]
        warnings: List[int] = [m.validation_warnings for m in recent_metrics]

        avg_score = sum(scores) / len(scores)
        tprint(f"get_summary: Generated quality summary for {symbol}: {len(recent_metrics)} samples, avg_score={avg_score:.3f}")

        summary = QualitySummary(
            symbol=symbol,
            period_start=cutoff_time,
            period_end=datetime.now(timezone.utc),
            avg_quality_score=avg_score,
            min_quality_score=min(scores),
            max_quality_score=max(scores),
            total_errors=sum(errors),
            total_warnings=sum(warnings),
            error_rate=sum(errors) / len(recent_metrics) if recent_metrics else 0.0,
            samples=len(recent_metrics)
        )
        tprint(f"get_summary: Returning summary for {symbol}")
        return summary
    
    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for symbol(s).

        Args:
            symbol: Trading symbol, or None for all symbols

        Returns:
            Dictionary of statistics
        """
        tprint(f"get_stats: symbol={symbol}")
        if symbol:
            stats = self._stats.get(symbol, {})
            tprint(f"get_stats: Returning stats for {symbol}: {len(stats)} items")
            return stats
        tprint(f"get_stats: Returning stats for all symbols: {len(self._stats)} symbols")
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
        tprint(f"get_recent_metrics: symbol={symbol}, count={count}")
        metrics = list(self._metrics[symbol])
        result = metrics[-count:] if len(metrics) > count else metrics
        tprint(f"get_recent_metrics: Returning {len(result)} metrics for {symbol}")
        return result
    
    def get_quality_trend(self, symbol: str, periods: int = 10) -> List[float]:
        """
        Get quality score trend over recent periods.

        Args:
            symbol: Trading symbol
            periods: Number of periods to analyze

        Returns:
            List of average quality scores per period
        """
        tprint(f"get_quality_trend: symbol={symbol}, periods={periods}")
        metrics = list(self._metrics[symbol])
        if not metrics:
            tprint(f"get_quality_trend: No metrics for {symbol}, returning empty list")
            return []

        # Split into periods
        period_size = max(1, len(metrics) // periods)
        trends = []
        tprint(f"get_quality_trend: Analyzing {len(metrics)} metrics in {periods} periods (period_size={period_size})")

        for i in range(0, len(metrics), period_size):
            period_metrics = metrics[i:i + period_size]
            avg_score = sum(m.quality_score for m in period_metrics) / len(period_metrics)
            trends.append(avg_score)

        tprint(f"get_quality_trend: Returning {len(trends)} trend values for {symbol}")
        return trends
    
    def clear_old_metrics(self, days: int = 30) -> None:
        """
        Clear metrics older than specified days.

        Args:
            days: Number of days to keep
        """
        tprint(f"clear_old_metrics: days={days}")
        cutoff_time: datetime = datetime.now(timezone.utc) - timedelta(days=days)
        cleared_count: int = 0

        for symbol in list(self._metrics.keys()):
            before_count: int = len(self._metrics[symbol])
            self._metrics[symbol] = deque(
                (m for m in self._metrics[symbol] if m.timestamp >= cutoff_time),
                maxlen=self.max_history
            )
            after_count: int = len(self._metrics[symbol])
            cleared_for_symbol = before_count - after_count
            if cleared_for_symbol > 0:
                tprint(f"clear_old_metrics: Cleared {cleared_for_symbol} metrics for {symbol}")
            cleared_count += cleared_for_symbol

        tprint(f"clear_old_metrics: Cleared {cleared_count} metrics older than {days} days")
        self.logger.info(f"Cleared metrics older than {days} days")
    
    def export_metrics(self, symbol: str, file_path: str) -> None:
        """
        Export metrics to CSV file.

        Args:
            symbol: Trading symbol
            file_path: Output file path
        """
        tprint(f"export_metrics: symbol={symbol}, file_path={file_path}")
        try:
            import pandas as pd

            metrics: List[QualityMetric] = list(self._metrics[symbol])
            if not metrics:
                tprint(f"export_metrics: No metrics to export for {symbol}, returning")
                self.logger.warning(f"No metrics to export for {symbol}")
                return

            tprint(f"export_metrics: Preparing {len(metrics)} metrics for export")
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

            tprint(f"export_metrics: Writing DataFrame to CSV")
            df: pd.DataFrame = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            tprint(f"export_metrics: Successfully exported {len(metrics)} metrics for {symbol} to {file_path}")
            self.logger.info(f"Exported {len(metrics)} metrics to {file_path}")

        except Exception as e:
            tprint(f"export_metrics: Failed to export metrics: {e}")
            self.logger.error(f"Failed to export metrics: {e}")
