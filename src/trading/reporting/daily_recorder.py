"""
Daily Trading Recorder

Creates daily summary records with one line per day containing
comprehensive trading metrics, model performance, and key events.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import csv
import json

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.tprint import tprint
from ..monitoring.comprehensive_trade_monitor import DetailedTradeMetrics, TradingSessionMetrics
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler
from ..utils.helpers import format_trading_metrics

logger = system_logger.getChild('DailyRecorder')

class DailyTradingRecord:
    """
    Single day trading record with comprehensive summary.
    """

    def __init__(self, trade_date: date):
        tprint(f"DailyTradingRecord.__init__: Initializing for date={trade_date}")
        self.date = trade_date

        # Basic Trading Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.break_even_trades = 0
        self.win_rate = 0.0

        # Performance Metrics
        self.total_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.profit_factor = 0.0
        self.best_trade = 0.0
        self.worst_trade = 0.0
        self.avg_trade_pnl = 0.0

        # Risk Metrics
        self.max_drawdown = 0.0
        self.avg_portfolio_risk = 0.0
        self.max_portfolio_risk = 0.0
        self.avg_leverage = 0.0
        self.max_leverage = 0.0
        self.sharpe_ratio = 0.0

        # Model Performance
        self.models_used = set()
        self.avg_model_confidence = 0.0
        self.best_model_accuracy = 0.0
        self.worst_model_accuracy = 0.0
        self.model_agreement_score = 0.0

        # Signal Analysis
        self.avg_signal_confidence = 0.0
        self.avg_signal_strength = 0.0
        self.signal_accuracy = 0.0

        # Regime Analysis
        self.primary_regime = "unknown"
        self.regime_changes = 0
        self.avg_regime_confidence = 0.0
        self.regime_stability = 0.0

        # Execution Quality
        self.avg_execution_quality = 0.0
        self.avg_slippage = 0.0
        self.avg_commission = 0.0
        self.execution_success_rate = 0.0
        self.avg_execution_time_ms = 0.0

        # Market Context
        self.market_volatility = 0.0
        self.market_trend = "neutral"
        self.avg_price = 0.0
        self.price_range_pct = 0.0
        self.volume_profile = "normal"

        # Feature Importance (Top 5)
        self.top_features = {}

        # Notable Events
        self.notable_events = []

        # Session Information
        self.sessions_count = 0
        self.total_session_duration_hours = 0.0
        self.avg_session_duration_hours = 0.0

        # System Health
        self.system_uptime_pct = 0.0
        self.error_count = 0
        self.warning_count = 0

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to CSV row format."""
        tprint(f"DailyTradingRecord.to_csv_row: Converting to CSV for date={self.date}")
        result = {
            # Date and Basic Info
            'date': self.date.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'break_even_trades': self.break_even_trades,
            'win_rate': round(self.win_rate, 4),

            # Performance
            'total_pnl': round(self.total_pnl, 2),
            'gross_profit': round(self.gross_profit, 2),
            'gross_loss': round(self.gross_loss, 2),
            'profit_factor': round(self.profit_factor, 4),
            'best_trade': round(self.best_trade, 2),
            'worst_trade': round(self.worst_trade, 2),
            'avg_trade_pnl': round(self.avg_trade_pnl, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 4),

            # Risk
            'max_drawdown': round(self.max_drawdown, 4),
            'avg_portfolio_risk': round(self.avg_portfolio_risk, 4),
            'max_portfolio_risk': round(self.max_portfolio_risk, 4),
            'avg_leverage': round(self.avg_leverage, 2),
            'max_leverage': round(self.max_leverage, 2),

            # Models
            'models_used_count': len(self.models_used),
            'models_used_list': '|'.join(sorted(self.models_used)),
            'avg_model_confidence': round(self.avg_model_confidence, 4),
            'best_model_accuracy': round(self.best_model_accuracy, 4),
            'worst_model_accuracy': round(self.worst_model_accuracy, 4),
            'model_agreement_score': round(self.model_agreement_score, 4),

            # Signals
            'avg_signal_confidence': round(self.avg_signal_confidence, 4),
            'avg_signal_strength': round(self.avg_signal_strength, 4),
            'signal_accuracy': round(self.signal_accuracy, 4),

            # Regime
            'primary_regime': self.primary_regime,
            'regime_changes': self.regime_changes,
            'avg_regime_confidence': round(self.avg_regime_confidence, 4),
            'regime_stability': round(self.regime_stability, 4),

            # Execution
            'avg_execution_quality': round(self.avg_execution_quality, 4),
            'avg_slippage': round(self.avg_slippage, 6),
            'avg_commission': round(self.avg_commission, 2),
            'execution_success_rate': round(self.execution_success_rate, 4),
            'avg_execution_time_ms': round(self.avg_execution_time_ms, 2),

            # Market
            'market_volatility': round(self.market_volatility, 4),
            'market_trend': self.market_trend,
            'avg_price': round(self.avg_price, 2),
            'price_range_pct': round(self.price_range_pct, 4),
            'volume_profile': self.volume_profile,

            # Top Features (JSON string of top 5)
            'top_features': json.dumps(dict(list(self.top_features.items())[:5])),

            # Events
            'notable_events_count': len(self.notable_events),
            'notable_events': '|'.join(self.notable_events),

            # Sessions
            'sessions_count': self.sessions_count,
            'total_session_duration_hours': round(self.total_session_duration_hours, 2),
            'avg_session_duration_hours': round(self.avg_session_duration_hours, 2),

            # System Health
            'system_uptime_pct': round(self.system_uptime_pct, 2),
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }
        tprint(f"DailyTradingRecord.to_csv_row: Returning CSV row with {len(result)} fields")
        return result

class DailyRecorder:
    """
    Daily trading recorder that creates one-line-per-day summaries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tprint(f"DailyRecorder.__init__: Initializing with config keys={list((config or {}).keys())}")
        self.config = config or {}
        self.logger = logger.getChild('DailyRecorder')

        # Configuration constants
        self.default_account_size = self.config.get('default_account_size', 10000.0)

        # File configuration
        self.records_directory = Path(self.config.get('records_directory', 'daily_trading_records'))
        self.records_filename = self.config.get('records_filename', 'daily_trading_log.csv')
        self.backup_enabled = self.config.get('backup_enabled', True)

        # Ensure directory exists
        self.records_directory.mkdir(parents=True, exist_ok=True)

        # Full path to records file
        self.records_file = self.records_directory / self.records_filename

        # Initialize file with headers if it doesn't exist
        self._initialize_records_file()

    def _initialize_records_file(self):
        """Initialize the daily records file with headers."""
        tprint(f"DailyRecorder._initialize_records_file: Checking records file at {self.records_file}")
        try:
            if not self.records_file.exists():
                # Create CSV headers
                headers = [
                    # Date and Basic Info
                    'date', 'total_trades', 'winning_trades', 'losing_trades', 'break_even_trades', 'win_rate',

                    # Performance
                    'total_pnl', 'gross_profit', 'gross_loss', 'profit_factor', 'best_trade', 'worst_trade',
                    'avg_trade_pnl', 'sharpe_ratio',

                    # Risk
                    'max_drawdown', 'avg_portfolio_risk', 'max_portfolio_risk', 'avg_leverage', 'max_leverage',

                    # Models
                    'models_used_count', 'models_used_list', 'avg_model_confidence', 'best_model_accuracy',
                    'worst_model_accuracy', 'model_agreement_score',

                    # Signals
                    'avg_signal_confidence', 'avg_signal_strength', 'signal_accuracy',

                    # Regime
                    'primary_regime', 'regime_changes', 'avg_regime_confidence', 'regime_stability',

                    # Execution
                    'avg_execution_quality', 'avg_slippage', 'avg_commission', 'execution_success_rate',
                    'avg_execution_time_ms',

                    # Market
                    'market_volatility', 'market_trend', 'avg_price', 'price_range_pct', 'volume_profile',

                    # Features and Events
                    'top_features', 'notable_events_count', 'notable_events',

                    # Sessions
                    'sessions_count', 'total_session_duration_hours', 'avg_session_duration_hours',

                    # System Health
                    'system_uptime_pct', 'error_count', 'warning_count'
                ]

                with open(self.records_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

                tprint_success(f"✅ Initialized daily records file: {self.records_file}")
                tprint(f"DailyRecorder._initialize_records_file: Created new file with {len(headers)} columns")
            else:
                tprint_info(f"📄 Using existing daily records file: {self.records_file}")
                tprint(f"DailyRecorder._initialize_records_file: Using existing file")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize records file: {e}")
            tprint(f"DailyRecorder._initialize_records_file: Exception - {e}")

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def record_daily_summary(
        self,
        trades: List[DetailedTradeMetrics],
        sessions: List[TradingSessionMetrics],
        target_date: Optional[date] = None
    ) -> bool:
        """
        Record daily trading summary.

        Args:
            trades: All trades for the day
            sessions: All sessions for the day
            target_date: Date to record (defaults to today)

        Returns:
            True if recording successful
        """
        tprint(f"DailyRecorder.record_daily_summary: Starting for target_date={target_date}, trades_count={len(trades)}, sessions_count={len(sessions)}")
        try:
            record_date = target_date or date.today()
            tprint(f"DailyRecorder.record_daily_summary: Using record_date={record_date}")

            tprint_info(f"📝 Recording daily summary for {record_date}")

            # Filter trades for the specific date
            daily_trades = [
                t for t in trades
                if t.timestamp.date() == record_date
            ]

            # Filter sessions for the specific date
            daily_sessions = [
                s for s in sessions
                if s.start_time.date() == record_date
            ]

            if not daily_trades and not daily_sessions:
                tprint_warning(f"⚠️ No trading activity found for {record_date}")
                tprint(f"DailyRecorder.record_daily_summary: Recording zero-activity day for {record_date}")
                # Still record a zero-activity day
                daily_trades = []
                daily_sessions = []

            # Create daily record
            daily_record = DailyTradingRecord(record_date)

            # Populate record with trade data
            await self._populate_daily_record(daily_record, daily_trades, daily_sessions)

            # Write to CSV file
            await self._write_daily_record(daily_record)

            # Create backup if enabled
            if self.backup_enabled:
                await self._create_backup(daily_record)

            tprint_success(f"✅ Daily summary recorded for {record_date}")
            tprint(f"DailyRecorder.record_daily_summary: Successfully recorded for {record_date}, returning True")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to record daily summary: {e}")
            tprint(f"DailyRecorder.record_daily_summary: Exception - {e}, returning False")
            return False

    async def _populate_daily_record(
        self,
        record: DailyTradingRecord,
        trades: List[DetailedTradeMetrics],
        sessions: List[TradingSessionMetrics]
    ):
        """Populate daily record with comprehensive data."""
        tprint(f"DailyRecorder._populate_daily_record: Populating record for date={record.date}, trades={len(trades)}, sessions={len(sessions)}")
        try:
            # Basic trade statistics
            record.total_trades = len(trades)

            if trades:
                # PnL analysis
                pnl_values = [t.pnl_absolute for t in trades if t.pnl_absolute is not None]

                if pnl_values:
                    record.winning_trades = len([p for p in pnl_values if p > 0])
                    record.losing_trades = len([p for p in pnl_values if p < 0])
                    record.break_even_trades = len([p for p in pnl_values if p == 0])
                    record.win_rate = record.winning_trades / len(pnl_values)

                    record.total_pnl = sum(pnl_values)
                    record.gross_profit = sum(p for p in pnl_values if p > 0)
                    record.gross_loss = abs(sum(p for p in pnl_values if p < 0))
                    # Profit factor: inf when no losses indicates perfect performance
                    if record.gross_loss > 0:
                        record.profit_factor = record.gross_profit / record.gross_loss
                    elif record.gross_profit > 0:
                        record.profit_factor = float('inf')  # Perfect performance - no losses
                    else:
                        record.profit_factor = 0.0  # No trades or no profits

                    record.best_trade = max(pnl_values)
                    record.worst_trade = min(pnl_values)
                    record.avg_trade_pnl = np.mean(pnl_values)

                    # Calculate Sharpe ratio
                    # Convert absolute PnL to percentage returns first
                    if len(pnl_values) > 1:
                        # Calculate returns as percentage changes
                        # For daily returns, we need to normalize by some base value
                        # Using first trade's value as reference, or calculate percentage returns
                        # Since we don't have account value, calculate returns relative to a base
                        base_value = abs(pnl_values[0]) if pnl_values[0] != 0 else self.default_account_size
                        returns = np.array(pnl_values) / base_value  # Convert to relative returns
                        record.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

                    # Calculate max drawdown
                    cumulative_pnl = np.cumsum(pnl_values)
                    peak = np.maximum.accumulate(cumulative_pnl)
                    drawdown = (peak - cumulative_pnl) / (peak + 1e-8)  # Avoid division by zero
                    record.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

                # Risk metrics
                portfolio_risks = [t.portfolio_risk for t in trades if t.portfolio_risk > 0]
                if portfolio_risks:
                    record.avg_portfolio_risk = np.mean(portfolio_risks)
                    record.max_portfolio_risk = max(portfolio_risks)

                leverages = [t.leverage for t in trades if t.leverage > 0]
                if leverages:
                    record.avg_leverage = np.mean(leverages)
                    record.max_leverage = max(leverages)

                # Model analysis
                all_models = set()
                model_confidences = []
                signal_confidences = []
                signal_strengths = []

                for trade in trades:
                    all_models.update(trade.models_used.keys())
                    model_confidences.extend(trade.model_confidences.values())
                    signal_confidences.append(trade.signal_confidence)
                    signal_strengths.append(trade.signal_strength)

                record.models_used = all_models
                record.avg_model_confidence = np.mean(model_confidences) if model_confidences else 0.0
                record.avg_signal_confidence = np.mean(signal_confidences) if signal_confidences else 0.0
                record.avg_signal_strength = np.mean(signal_strengths) if signal_strengths else 0.0

                # Model accuracy analysis
                model_accuracies = await self._calculate_model_accuracies(trades)
                if model_accuracies:
                    record.best_model_accuracy = max(model_accuracies.values())
                    record.worst_model_accuracy = min(model_accuracies.values())
                    # Model agreement: inverse of variance (lower variance = higher agreement)
                    if len(model_accuracies) > 1:
                        accuracy_values = list(model_accuracies.values())
                        variance = np.var(accuracy_values)
                        # Normalize variance to [0, 1] and invert: high variance = low agreement
                        max_variance = 0.25  # Maximum possible variance for accuracy [0, 1] is 0.25
                        normalized_variance = min(variance / max_variance, 1.0)
                        record.model_agreement_score = 1.0 - normalized_variance
                    else:
                        record.model_agreement_score = 1.0  # Single model = perfect agreement

                # Signal accuracy
                correct_signals = 0
                for trade in trades:
                    if trade.pnl_absolute is not None:
                        # Signal was correct if high confidence led to profit
                        if (trade.signal_confidence > 0.7 and trade.pnl_absolute > 0) or \
                           (trade.signal_confidence < 0.5 and trade.pnl_absolute <= 0):
                            correct_signals += 1

                record.signal_accuracy = correct_signals / len(trades) if trades else 0.0

                # Regime analysis
                regimes = [t.regime_type for t in trades]
                regime_confidences = [t.regime_confidence for t in trades if t.regime_confidence > 0]

                if regimes:
                    # Most common regime
                    regime_counts = {}
                    for regime in regimes:
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1
                    record.primary_regime = max(regime_counts.items(), key=lambda x: x[1])[0]

                    # Regime changes - count actual transitions between consecutive trades
                    regime_changes_count = 0
                    if len(trades) > 1:
                        # Sort trades by timestamp to ensure correct order
                        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
                        for i in range(1, len(sorted_trades)):
                            if sorted_trades[i-1].regime_type != sorted_trades[i].regime_type:
                                regime_changes_count += 1
                    record.regime_changes = regime_changes_count

                    # Average regime confidence
                    record.avg_regime_confidence = np.mean(regime_confidences) if regime_confidences else 0.0

                    # Regime stability (less changes = more stable)
                    # Normalize to [0, 1] range: 0 = unstable (many changes), 1 = stable (no changes)
                    if len(trades) > 1:
                        max_possible_changes = len(trades) - 1
                        stability = 1.0 - (regime_changes_count / max_possible_changes) if max_possible_changes > 0 else 1.0
                        record.regime_stability = max(0.0, min(1.0, stability))  # Clamp to [0, 1]
                    else:
                        record.regime_stability = 1.0  # Single trade means stable

                # Execution quality
                execution_qualities = [t.execution_quality for t in trades if t.execution_quality > 0]
                slippages = [t.slippage for t in trades if t.slippage is not None]
                commissions = [t.commission for t in trades if t.commission is not None]
                execution_times = [t.execution_time_ms for t in trades if t.execution_time_ms > 0]

                if execution_qualities:
                    record.avg_execution_quality = np.mean(execution_qualities)
                    record.execution_success_rate = len([q for q in execution_qualities if q > 0.8]) / len(execution_qualities)

                if slippages:
                    record.avg_slippage = np.mean(slippages)

                if commissions:
                    record.avg_commission = np.mean(commissions)

                if execution_times:
                    record.avg_execution_time_ms = np.mean(execution_times)

                # Market context
                prices = [t.price for t in trades]
                if prices:
                    record.avg_price = np.mean(prices)
                    record.price_range_pct = (max(prices) - min(prices)) / np.mean(prices)

                # Market volatility from trades
                volatilities = [t.volatility_estimate for t in trades if t.volatility_estimate > 0]
                if volatilities:
                    record.market_volatility = np.mean(volatilities)

                # Market trend analysis
                if len(trades) > 1:
                    first_price = trades[0].price
                    last_price = trades[-1].price
                    trend_change = (last_price - first_price) / first_price

                    if trend_change > 0.01:
                        record.market_trend = "bullish"
                    elif trend_change < -0.01:
                        record.market_trend = "bearish"
                    else:
                        record.market_trend = "neutral"

                # Feature importance aggregation
                all_features = {}
                feature_counts = {}

                for trade in trades:
                    for feature, importance in trade.feature_importance.items():
                        if feature not in all_features:
                            all_features[feature] = 0.0
                            feature_counts[feature] = 0
                        all_features[feature] += importance
                        feature_counts[feature] += 1

                # Average feature importance
                for feature in all_features:
                    all_features[feature] /= feature_counts[feature]

                # Top 5 features
                sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
                record.top_features = dict(sorted_features[:5])

                # Notable events detection
                await self._detect_notable_events(record, trades)

            # Session analysis
            if sessions:
                record.sessions_count = len(sessions)

                session_durations = []
                for session in sessions:
                    if session.end_time:
                        duration = (session.end_time - session.start_time).total_seconds() / 3600
                        session_durations.append(duration)

                if session_durations:
                    record.total_session_duration_hours = sum(session_durations)
                    record.avg_session_duration_hours = np.mean(session_durations)

            # System health (would be populated from system monitoring)
            record.system_uptime_pct = 95.0  # Placeholder
            record.error_count = 0  # Would be populated from error tracking
            record.warning_count = 2  # Would be populated from warning tracking

            tprint(f"DailyRecorder._populate_daily_record: Completed population - total_trades={record.total_trades}, total_pnl={record.total_pnl:.2f}")

        except Exception as e:
            tprint_error(f"❌ Failed to populate daily record: {e}")
            tprint(f"DailyRecorder._populate_daily_record: Exception during population - {e}")
            raise

    async def _calculate_model_accuracies(self, trades: List[DetailedTradeMetrics]) -> Dict[str, float]:
        """Calculate accuracy for each model."""
        tprint(f"DailyRecorder._calculate_model_accuracies: Calculating for {len(trades)} trades")
        try:
            model_accuracies = {}

            # Get all unique models
            all_models = set()
            for trade in trades:
                all_models.update(trade.models_used.keys())

            # Calculate accuracy for each model
            for model_id in all_models:
                model_trades = [t for t in trades if model_id in t.models_used and t.pnl_absolute is not None]

                if model_trades:
                    correct_predictions = 0
                    for trade in model_trades:
                        model_confidence = trade.model_confidences.get(model_id, 0.0)

                        # Model was "correct" if high confidence led to profit
                        if (model_confidence > 0.7 and trade.pnl_absolute > 0) or \
                           (model_confidence < 0.5 and trade.pnl_absolute <= 0):
                            correct_predictions += 1

                    model_accuracies[model_id] = correct_predictions / len(model_trades)

            tprint(f"DailyRecorder._calculate_model_accuracies: Calculated {len(model_accuracies)} model accuracies")
            return model_accuracies

        except Exception as e:
            tprint_error(f"❌ Failed to calculate model accuracies: {e}")
            tprint(f"DailyRecorder._calculate_model_accuracies: Exception - {e}, returning empty dict")
            return {}

    async def _detect_notable_events(self, record: DailyTradingRecord, trades: List[DetailedTradeMetrics]):
        """Detect notable events for the day."""
        tprint(f"DailyRecorder._detect_notable_events: Detecting events for {len(trades)} trades")
        try:
            events = []

            # Large PnL events
            if record.best_trade > 500:
                events.append(f"LARGE_WIN:{record.best_trade:.0f}")

            if record.worst_trade < -200:
                events.append(f"LARGE_LOSS:{record.worst_trade:.0f}")

            # High performance events
            if record.win_rate > 0.8:
                events.append(f"HIGH_WIN_RATE:{record.win_rate:.1%}")

            if record.profit_factor > 3.0:
                events.append(f"HIGH_PROFIT_FACTOR:{record.profit_factor:.1f}")

            # Risk events
            if record.max_drawdown > 0.1:
                events.append(f"HIGH_DRAWDOWN:{record.max_drawdown:.1%}")

            if record.max_leverage > 5.0:
                events.append(f"HIGH_LEVERAGE:{record.max_leverage:.1f}")

            # Model events
            if record.avg_model_confidence > 0.9:
                events.append(f"HIGH_MODEL_CONFIDENCE:{record.avg_model_confidence:.1%}")

            if record.model_agreement_score < 0.3:
                events.append("LOW_MODEL_AGREEMENT")

            # Regime events
            if record.regime_changes > 5:
                events.append(f"HIGH_REGIME_VOLATILITY:{record.regime_changes}")

            if record.avg_regime_confidence > 0.9:
                events.append(f"HIGH_REGIME_CONFIDENCE:{record.avg_regime_confidence:.1%}")

            # Execution events
            if record.avg_execution_quality < 0.7:
                events.append("POOR_EXECUTION_QUALITY")

            if record.avg_slippage > 0.005:
                events.append(f"HIGH_SLIPPAGE:{record.avg_slippage:.1%}")

            # Trading volume events
            if record.total_trades > 50:
                events.append(f"HIGH_ACTIVITY:{record.total_trades}")
            elif record.total_trades == 0:
                events.append("NO_TRADING")

            record.notable_events = events
            tprint(f"DailyRecorder._detect_notable_events: Detected {len(events)} notable events")

        except Exception as e:
            tprint_error(f"❌ Failed to detect notable events: {e}")
            tprint(f"DailyRecorder._detect_notable_events: Exception - {e}")

    async def _write_daily_record(self, record: DailyTradingRecord):
        """Write daily record to CSV file."""
        tprint(f"DailyRecorder._write_daily_record: Writing record for date={record.date}")
        try:
            # Check if record already exists for this date
            existing_records = await self._read_existing_records()

            # Remove existing record for this date if it exists
            existing_records = [r for r in existing_records if r.get('date') != record.date.isoformat()]

            # Add new record
            new_row = record.to_csv_row()

            # Write all records back to file
            with open(self.records_file, 'w', newline='') as f:
                if existing_records:
                    # Get field names from existing records
                    fieldnames = existing_records[0].keys()
                else:
                    # Use new record field names
                    fieldnames = new_row.keys()

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # Write existing records
                for existing_record in existing_records:
                    writer.writerow(existing_record)

                # Write new record
                writer.writerow(new_row)

            tprint_info(f"📝 Daily record written to {self.records_file}")
            tprint(f"DailyRecorder._write_daily_record: Successfully wrote record for {record.date}")

        except Exception as e:
            tprint_error(f"❌ Failed to write daily record: {e}")
            tprint(f"DailyRecorder._write_daily_record: Exception - {e}")
            raise

    async def _read_existing_records(self) -> List[Dict[str, Any]]:
        """Read existing records from CSV file."""
        tprint(f"DailyRecorder._read_existing_records: Reading from {self.records_file}")
        try:
            if not self.records_file.exists():
                return []

            records = []
            with open(self.records_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))

            tprint(f"DailyRecorder._read_existing_records: Read {len(records)} existing records")
            return records

        except Exception as e:
            tprint_warning(f"⚠️ Failed to read existing records: {e}")
            tprint(f"DailyRecorder._read_existing_records: Exception - {e}, returning empty list")
            return []

    async def _create_backup(self, record: DailyTradingRecord):
        """Create backup of daily record."""
        tprint(f"DailyRecorder._create_backup: Creating backup for date={record.date}")
        try:
            backup_dir = self.records_directory / 'backups'
            backup_dir.mkdir(exist_ok=True)

            # Create monthly backup file
            backup_file = backup_dir / f"daily_records_{record.date.strftime('%Y_%m')}.csv"

            # Append to monthly backup
            file_exists = backup_file.exists()

            with open(backup_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=record.to_csv_row().keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(record.to_csv_row())

            tprint_info(f"💾 Backup created: {backup_file}")
            tprint(f"DailyRecorder._create_backup: Backup successful for {record.date}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to create backup: {e}")
            tprint(f"DailyRecorder._create_backup: Exception - {e}")

    async def get_daily_summary(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Get daily summary for a specific date."""
        tprint(f"DailyRecorder.get_daily_summary: Retrieving summary for date={target_date}")
        try:
            existing_records = await self._read_existing_records()

            for record in existing_records:
                if record.get('date') == target_date.isoformat():
                    tprint(f"DailyRecorder.get_daily_summary: Found summary for {target_date}")
                    return record

            tprint(f"DailyRecorder.get_daily_summary: No summary found for {target_date}")
            return None

        except Exception as e:
            tprint_error(f"❌ Failed to get daily summary: {e}")
            tprint(f"DailyRecorder.get_daily_summary: Exception - {e}, returning None")
            return None

    async def get_historical_summary(self, days: int = 30) -> pd.DataFrame:
        """Get historical daily summaries."""
        tprint(f"DailyRecorder.get_historical_summary: Retrieving last {days} days")
        try:
            existing_records = await self._read_existing_records()

            if not existing_records:
                return pd.DataFrame()

            df = pd.DataFrame(existing_records)
            df['date'] = pd.to_datetime(df['date'])

            # Filter to last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= cutoff_date]

            # Sort by date
            df = df.sort_values('date')

            tprint(f"DailyRecorder.get_historical_summary: Returning {len(df)} records")
            return df

        except Exception as e:
            tprint_error(f"❌ Failed to get historical summary: {e}")
            tprint(f"DailyRecorder.get_historical_summary: Exception - {e}, returning empty DataFrame")
            return pd.DataFrame()

# Global instance
daily_recorder = DailyRecorder()

# Convenience functions
async def record_daily_trading_summary(
    trades: List[DetailedTradeMetrics],
    sessions: List[TradingSessionMetrics],
    target_date: Optional[date] = None
) -> bool:
    """Record daily trading summary."""
    tprint(f"record_daily_trading_summary: Called with {len(trades)} trades, {len(sessions)} sessions, target_date={target_date}")
    result = await daily_recorder.record_daily_summary(trades, sessions, target_date)
    tprint(f"record_daily_trading_summary: Returning {result}")
    return result

async def get_daily_trading_summary(target_date: date) -> Optional[Dict[str, Any]]:
    """Get daily trading summary for specific date."""
    tprint(f"get_daily_trading_summary: Called for target_date={target_date}")
    result = await daily_recorder.get_daily_summary(target_date)
    tprint(f"get_daily_trading_summary: Returning {'summary' if result else 'None'}")
    return result

async def get_trading_history(days: int = 30) -> pd.DataFrame:
    """Get historical daily trading summaries."""
    tprint(f"get_trading_history: Called for days={days}")
    result = await daily_recorder.get_historical_summary(days)
    tprint(f"get_trading_history: Returning DataFrame with {len(result)} rows")
    return result
