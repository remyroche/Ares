#!/usr/bin/env python3
"""
Daily Summary Tracker for Enhanced ML Monitoring

Tracks daily trading statistics including trades, shorts vs longs, HMM regime,
PnL, win rate, and other key metrics for ongoing monitoring.
"""

import collections
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
import typing

@dataclass
class DailyTradeSummary:
    """Daily summary of trading activity."""
    date: date
    trading_mode: str  # "backtest", "paper", "live", or "all"
    total_trades: int
    long_trades: int
    short_trades: int
    hold_trades: int
    
    # HMM Regime Information
    dominant_regime: str
    regime_distribution: Dict[str, int]  # regime -> count
    regime_stability_avg: float
    
    # Performance Metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Trading Statistics
    avg_position_size: float
    avg_confidence: float
    avg_risk_score: float
    total_volume: float
    
    # Model Performance
    model_accuracy_avg: float
    ensemble_consensus_avg: float
    model_disagreement_avg: float
    
    # Risk Metrics
    var_95: float  # Value at Risk 95%
    max_loss: float
    max_gain: float
    
    # Additional Metrics
    execution_time_avg_ms: float
    successful_trades: int
    failed_trades: int
    
    # Timestamps
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    summary_generated_at: datetime = None

@dataclass
class RegimePerformance:
    """Performance metrics by HMM regime."""
    regime_id: str
    regime_name: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    profit_factor: float
    sharpe_ratio: float
    avg_confidence: float
    avg_risk_score: float

class DailySummaryTracker:
    """
    Tracks and generates daily summaries of trading activity and performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the daily summary tracker."""
        self.config = config
        self.logger = system_logger.getChild("DailySummaryTracker")
        
        # Configuration
        self.tracker_config = config.get("daily_summary_tracker", {})
        self.enable_real_time_updates = self.tracker_config.get("enable_real_time_updates", True)
        self.summary_retention_days = self.tracker_config.get("summary_retention_days", 365)
        self.export_directory = self.tracker_config.get("export_directory", "daily_summaries")
        
        # Storage - separate by trading mode
        self.daily_summaries: Dict[Tuple[date, str], DailyTradeSummary] = {}  # (date, mode) -> summary
        self.regime_performances: Dict[str, List[RegimePerformance]] = defaultdict(list)
        self.trade_buffer: List[Any] = []  # Buffer for current day's trades
        
        # Export paths
        self.export_dir = Path(self.export_directory)
        self.export_dir.mkdir(exist_ok = True)
        
        # Current day tracking - separate by mode
        self.current_date = date.today()
        self.current_day_trades: Dict[str, List[Any]] = defaultdict(list)  # mode -> trades
        
        self.logger.info("Daily Summary Tracker initialized")
    
    @handles_errors(default_return = None, context="daily_summary_tracker.add_trade")
    async def add_trade(self, trade_decision: Any) -> None:
        """Add a trade decision to the current day's tracking."""
        try:
            trade_date = trade_decision.timestamp.date()
            trading_mode = trade_decision.trading_mode.value
            
            # Check if we need to process previous day
            if trade_date != self.current_date:
                await self._process_previous_day()
                self.current_date = trade_date
                self.current_day_trades.clear()
            
            # Add trade to current day for this mode
            self.current_day_trades[trading_mode].append(trade_decision)
            
            # Update real-time if enabled
            if self.enable_real_time_updates:
                await self._update_current_day_summary()
            
            self.logger.debug(f"Added {trading_mode} trade to daily tracking: {trade_decision.decision_id}")
            
        except Exception as e:
            self.logger.error(f"Error adding trade to daily tracking: {e}")
    
    async def _process_previous_day(self):
        """Process and generate summary for the previous day."""
        try:
            if not self.current_day_trades:
                return
            
            # Generate summaries for each trading mode
            for trading_mode, trades in self.current_day_trades.items():
                if not trades:
                    continue
                
                # Generate summary for this mode
                summary = await self._generate_daily_summary(self.current_date, trades, trading_mode)
                
                # Store summary with mode key
                self.daily_summaries[(self.current_date, trading_mode)] = summary
                
                # Update regime performances
                await self._update_regime_performances(summary, trades)
                
                # Export daily summary
                await self._export_daily_summary(summary)
                
                self.logger.info(f"Processed {len(trades)} {trading_mode} trades for {self.current_date}")
            
        except Exception as e:
            self.logger.error(f"Error processing previous day: {e}")
    
    async def _update_current_day_summary(self):
        """Update the current day's summary in real-time."""
        try:
            if not self.current_day_trades:
                return
            
            # Update summaries for each trading mode
            for trading_mode, trades in self.current_day_trades.items():
                if not trades:
                    continue
                
                # Generate current summary for this mode
                summary = await self._generate_daily_summary(self.current_date, trades, trading_mode)
                
                # Update stored summary
                self.daily_summaries[(self.current_date, trading_mode)] = summary
            
        except Exception as e:
            self.logger.error(f"Error updating current day summary: {e}")
    
    async def _generate_daily_summary(self, trade_date: date, trades: List[Any], trading_mode: str = "all") -> DailyTradeSummary:
        """Generate comprehensive daily summary from trades."""
        try:
            if not trades:
                return self._create_empty_summary(trade_date)
            
            # Basic trade counts
            total_trades = len(trades)
            long_trades = sum(1 for t in trades if t.action == "buy")
            short_trades = sum(1 for t in trades if t.action == "sell")
            hold_trades = sum(1 for t in trades if t.action == "hold")
            
            # HMM Regime Analysis
            regime_distribution = defaultdict(int)
            regime_stabilities = []
            regime_probabilities = []
            
            for trade in trades:
                if trade.context.hmm_regime_info:
                    regime_id = trade.context.hmm_regime_info.regime_id
                    regime_distribution[regime_id] += 1
                    regime_stabilities.append(trade.context.hmm_regime_info.regime_stability_score)
                    regime_probabilities.append(trade.context.hmm_regime_info.regime_probability)
            
            dominant_regime = max(regime_distribution.items(), key=lambda x: x[1])[0] if regime_distribution else "unknown"
            regime_stability_avg = np.mean(regime_stabilities) if regime_stabilities else 0.0
            
            # Performance Metrics
            pnls = []
            realized_pnls = []
            unrealized_pnls = []
            wins = 0
            
            for trade in trades:
                if trade.success_metrics:
                    pnl = trade.success_metrics.get('profit_loss', 0.0)
                    pnls.append(pnl)
                    
                    if pnl > 0:
                        wins += 1
                        realized_pnls.append(pnl)
                    else:
                        unrealized_pnls.append(pnl)
            
            total_pnl = sum(pnls)
            realized_pnl = sum(realized_pnls)
            unrealized_pnl = sum(unrealized_pnls)
            win_rate = wins / total_trades if total_trades > 0 else 0.0
            
            # Calculate profit factor
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            
            # Calculate Sharpe ratio (simplified)
            if pnls:
                returns = np.array(pnls)
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(pnls)
            
            # Trading Statistics
            position_sizes = [t.position_size for t in trades if t.position_size]
            confidences = [t.overall_confidence for t in trades]
            risk_scores = [t.overall_risk_score for t in trades]
            volumes = [t.context.volume for t in trades]
            
            avg_position_size = np.mean(position_sizes) if position_sizes else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            avg_risk_score = np.mean(risk_scores) if risk_scores else 0.0
            total_volume = sum(volumes)
            
            # Model Performance
            model_accuracies = []
            ensemble_consensus = []
            model_disagreements = []
            
            for trade in trades:
                if trade.ensemble_decision:
                    ensemble_consensus.append(trade.ensemble_decision.consensus_score)
                    model_disagreements.append(trade.ensemble_decision.disagreement_level)
                    
                    for model_decision in trade.ensemble_decision.model_decisions:
                        # Use confidence as proxy for accuracy
                        model_accuracies.append(model_decision.confidence)
            
            model_accuracy_avg = np.mean(model_accuracies) if model_accuracies else 0.0
            ensemble_consensus_avg = np.mean(ensemble_consensus) if ensemble_consensus else 0.0
            model_disagreement_avg = np.mean(model_disagreements) if model_disagreements else 0.0
            
            # Risk Metrics
            var_95 = np.percentile(pnls, 5) if pnls else 0.0  # 5th percentile for VaR
            max_loss = min(pnls) if pnls else 0.0
            max_gain = max(pnls) if pnls else 0.0
            
            # Execution Statistics
            execution_times = [t.execution_time_ms for t in trades if t.execution_time_ms]
            execution_time_avg_ms = np.mean(execution_times) if execution_times else 0.0
            
            successful_trades = sum(1 for t in trades if t.success_metrics and t.success_metrics.get('profit_loss', 0) > 0)
            failed_trades = total_trades - successful_trades
            
            # Timestamps
            first_trade_time = min(t.timestamp for t in trades) if trades else None
            last_trade_time = max(t.timestamp for t in trades) if trades else None
            
            return DailyTradeSummary(
                date = trade_date,
                trading_mode = trading_mode,
                total_trades = total_trades,
                long_trades = long_trades,
                short_trades = short_trades,
                hold_trades = hold_trades,
                dominant_regime = dominant_regime,
                regime_distribution = dict(regime_distribution),
                regime_stability_avg = regime_stability_avg,
                total_pnl = total_pnl,
                realized_pnl = realized_pnl,
                unrealized_pnl = unrealized_pnl,
                win_rate = win_rate,
                profit_factor = profit_factor,
                sharpe_ratio = sharpe_ratio,
                max_drawdown = max_drawdown,
                avg_position_size = avg_position_size,
                avg_confidence = avg_confidence,
                avg_risk_score = avg_risk_score,
                total_volume = total_volume,
                model_accuracy_avg = model_accuracy_avg,
                ensemble_consensus_avg = ensemble_consensus_avg,
                model_disagreement_avg = model_disagreement_avg,
                var_95 = var_95,
                max_loss = max_loss,
                max_gain = max_gain,
                execution_time_avg_ms = execution_time_avg_ms,
                successful_trades = successful_trades,
                failed_trades = failed_trades,
                first_trade_time = first_trade_time,
                last_trade_time = last_trade_time,
                summary_generated_at = datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating daily summary: {e}")
            return self._create_empty_summary(trade_date, trading_mode)
    
    def _create_empty_summary(self, trade_date: date, trading_mode: str = "all") -> DailyTradeSummary:
        """Create an empty summary for a day with no trades."""
        return DailyTradeSummary(
            date = trade_date,
            trading_mode = trading_mode,
            total_trades = 0,
            long_trades = 0,
            short_trades = 0,
            hold_trades = 0,
            dominant_regime="none",
            regime_distribution={},
            regime_stability_avg = 0.0,
            total_pnl = 0.0,
            realized_pnl = 0.0,
            unrealized_pnl = 0.0,
            win_rate = 0.0,
            profit_factor = 0.0,
            sharpe_ratio = 0.0,
            max_drawdown = 0.0,
            avg_position_size = 0.0,
            avg_confidence = 0.0,
            avg_risk_score = 0.0,
            total_volume = 0.0,
            model_accuracy_avg = 0.0,
            ensemble_consensus_avg = 0.0,
            model_disagreement_avg = 0.0,
            var_95 = 0.0,
            max_loss = 0.0,
            max_gain = 0.0,
            execution_time_avg_ms = 0.0,
            successful_trades = 0,
            failed_trades = 0,
            summary_generated_at = datetime.now()
        )
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnls:
            return 0.0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    
    async def _update_regime_performances(self, summary: DailyTradeSummary):
        """Update regime performance tracking."""
        try:
            for regime_id, count in summary.regime_distribution.items():
                if count > 0:
                    # Calculate regime-specific metrics
                    regime_trades = [t for t in self.current_day_trades 
                                if t.context.hmm_regime_info and t.context.hmm_regime_info.regime_id == regime_id]
                    
                    if regime_trades:
                        regime_pnls = []
                        regime_wins = 0
                        regime_confidences = []
                        regime_risks = []
                        
                        for trade in regime_trades:
                            if trade.success_metrics:
                                pnl = trade.success_metrics.get('profit_loss', 0.0)
                                regime_pnls.append(pnl)
                                if pnl > 0:
                                    regime_wins += 1
                            
                            regime_confidences.append(trade.overall_confidence)
                            regime_risks.append(trade.overall_risk_score)
                        
                        regime_performance = RegimePerformance(
                            regime_id = regime_id,
                            regime_name = regime_id,  # Could be enhanced with actual names
                            trade_count = count,
                            win_rate = regime_wins / count if count > 0 else 0.0,
                            avg_pnl = np.mean(regime_pnls) if regime_pnls else 0.0,
                            total_pnl = sum(regime_pnls),
                            profit_factor = 0.0,  # Would need more complex calculation
                            sharpe_ratio = 0.0,   # Would need more complex calculation
                            avg_confidence = np.mean(regime_confidences) if regime_confidences else 0.0,
                            avg_risk_score = np.mean(regime_risks) if regime_risks else 0.0
                        )
                        
                        self.regime_performances[regime_id].append(regime_performance)
                        
        except Exception as e:
            self.logger.error(f"Error updating regime performances: {e}")
    
    @handles_errors(default_return = False, context="daily_summary_tracker._export_daily_summary")
    async def _export_daily_summary(self, summary: DailyTradeSummary) -> bool:
        """Export daily summary to CSV."""
        try:
            # Convert to DataFrame
            summary_data = asdict(summary)
            
            # Convert datetime objects to strings
            for key, value in summary_data.items():
                if isinstance(value, datetime):
                    summary_data[key] = value.isoformat()
                elif isinstance(value, date):
                    summary_data[key] = value.isoformat()
            
            df = pd.DataFrame([summary_data])
            
            # Export to CSV with trading mode in filename
            filename = f"daily_summary_{summary.trading_mode}_{summary.date.strftime('%Y%m%d')}.csv"
            filepath = self.export_dir / filename
            df.to_csv(filepath, index = False)
            
            self.logger.info(f"Exported daily summary for {summary.date} to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting daily summary: {e}")
            return False
    
    @handles_errors(default_return = None, context="daily_summary_tracker.get_daily_summary")
    async def get_daily_summary(self, target_date: Optional[date] = None) -> Optional[DailyTradeSummary]:
        """Get daily summary for a specific date."""
        try:
            if target_date is None:
                target_date = date.today()
            
            return self.daily_summaries.get(target_date)
            
        except Exception as e:
            self.logger.error(f"Error getting daily summary: {e}")
            return None
    
    @handles_errors(default_return = None, context="daily_summary_tracker.get_summary_range")
    async def get_summary_range(self, start_date: date, end_date: date) -> List[DailyTradeSummary]:
        """Get daily summaries for a date range."""
        try:
            summaries = []
            current_date = start_date
            
            while current_date <= end_date:
                if current_date in self.daily_summaries:
                    summaries.append(self.daily_summaries[current_date])
                current_date += timedelta(days = 1)
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error getting summary range: {e}")
            return []
    
    @handles_errors(default_return = None, context="daily_summary_tracker.get_regime_performance")
    async def get_regime_performance(self, regime_id: str, days: int = 30) -> List[RegimePerformance]:
        """Get regime performance for the last N days."""
        try:
            if regime_id not in self.regime_performances:
                return []
            
            performances = self.regime_performances[regime_id]
            return performances[-days:] if len(performances) > days else performances
            
        except Exception as e:
            self.logger.error(f"Error getting regime performance: {e}")
            return []
    
    @handles_errors(default_return = False, context="daily_summary_tracker.export_summary_csv")
    async def export_summary_csv(self, start_date: Optional[date] = None, 
                            end_date: Optional[date] = None) -> bool:
        """Export daily summaries to a comprehensive CSV file."""
        try:
            if start_date is None:
                start_date = min(self.daily_summaries.keys()) if self.daily_summaries else date.today()
            if end_date is None:
                end_date = max(self.daily_summaries.keys()) if self.daily_summaries else date.today()
            
            summaries = await self.get_summary_range(start_date, end_date)
            
            if not summaries:
                self.logger.warning("No summaries to export")
                return False
            
            # Convert to DataFrame
            data = []
            for summary in summaries:
                summary_dict = asdict(summary)
                
                # Convert datetime objects to strings
                for key, value in summary_dict.items():
                    if isinstance(value, datetime):
                        summary_dict[key] = value.isoformat()
                    elif isinstance(value, date):
                        summary_dict[key] = value.isoformat()
                    elif isinstance(value, dict):
                        # Flatten regime distribution
                        if key == 'regime_distribution':
                            for regime, count in value.items():
                                summary_dict[f'regime_{regime}_count'] = count
                            del summary_dict[key]
                
                data.append(summary_dict)
            
            df = pd.DataFrame(data)
            
            # Export to CSV
            filename = f"daily_summaries_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
            filepath = self.export_dir / filename
            df.to_csv(filepath, index = False)
            
            self.logger.info(f"Exported {len(summaries)} daily summaries to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting summary CSV: {e}")
            return False
    
    def get_tracker_stats(self) -> Dict[str, Any]:
        """Get statistics about the daily summary tracker."""
        return {
            'total_days_tracked': len(self.daily_summaries),
            'current_date': self.current_date.isoformat(),
            'trades_today': len(self.current_day_trades),
            'regimes_tracked': len(self.regime_performances),
            'export_directory': str(self.export_dir),
            'summary_retention_days': self.summary_retention_days,
            'enable_real_time_updates': self.enable_real_time_updates
        }