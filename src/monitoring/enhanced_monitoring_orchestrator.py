#!/usr/bin/env python3
"""
Enhanced Monitoring Orchestrator

Comprehensive monitoring system that integrates all monitoring components to provide
detailed tracking and explanations for trading decisions across backtesting, paper trading,
and live trading systems.

Features:
- Context capture (exchange, token, time, price)
- Trade indicators (confidence, risk, etc.)
- Per-ensemble indicators (weight of each ML model)
- Per-ML indicators (confidence, risk, etc.)
- Per-ML decision making (weight of each trading indicator)
- SHAP/LIME explanations for detailed model insights
- Monthly and daily CSV exports
- Integration with all trading modes
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from .enhanced_ml_monitoring import (
    EnhancedMLMonitor, TradeContext, TradingIndicator, MLModelDecision,
    EnsembleDecision, TradeDecision, TradingMode, ModelType,
    ModelPerformanceMetrics, EnsemblePerformanceMetrics
)
from .ensemble_monitor import EnsembleMonitor, ModelContribution
from .daily_summary_tracker import DailySummaryTracker, DailyTradeSummary
from .shap_lime_integration import ExplainabilityIntegrator
from .trading_integration import TradingSystemIntegrator
import logging
import time

@dataclass
class EnhancedMonitoringConfig:
    """Configuration for enhanced monitoring system."""
    # Core monitoring settings
    enable_monitoring: bool = True
    enable_explanations: bool = True
    enable_real_time_tracking: bool = True

    # Export settings
    monthly_export_enabled: bool = True
    daily_export_enabled: bool = True
    export_directory: str = "enhanced_monitoring_exports"

    # Performance tracking
    max_decisions_in_memory: int = 50000
    performance_window_days: int = 30

    # SHAP/LIME settings
    enable_shap: bool = True
    enable_lime: bool = True
    explanation_timeout: int = 30

    # Trading integration
    auto_integrate_trading_systems: bool = True
    capture_all_trading_modes: bool = True

    # Data retention
    data_retention_days: int = 365
    cleanup_frequency_hours: int = 24

@dataclass
class ComprehensiveTradeDecision:
    """Comprehensive trade decision with full context and explanations."""
    # Basic decision info
    decision_id: str
    timestamp: datetime
    trading_mode: TradingMode

    # Context (exchange, token, time, price)
    context: TradeContext

    # Trade indicators (confidence, risk, etc.)
    trading_indicators: List[TradingIndicator]
    overall_confidence: float
    overall_risk_score: float

    # Per-ensemble indicators (weight of each ML model)
    ensemble_decision: EnsembleDecision

    # Per-ML indicators (confidence, risk, etc.)
    individual_model_decisions: List[MLModelDecision]

    # Per-ML decision making (weight of each trading indicator)
    model_indicator_weights: Dict[str, Dict[str, float]]  # model_id -> indicator_name -> weight

    # SHAP/LIME explanations
    shap_explanations: Optional[Dict[str, Any]] = None
    lime_explanations: Optional[Dict[str, Any]] = None

    # Final decision
    action: str  # "buy", "sell", "hold"
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Performance tracking
    execution_time_ms: float = 0.0
    success_metrics: Optional[Dict[str, float]] = None

    # Additional metadata
    market_conditions: Optional[Dict[str, Any]] = None
    regime_analysis: Optional[Dict[str, Any]] = None

@dataclass
class MonthlyReport:
    """Monthly comprehensive monitoring report."""
    report_id: str
    month: str  # YYYY-MM format
    start_date: date
    end_date: date
    generated_at: datetime

    # Summary statistics
    total_decisions: int
    total_trades: int
    trading_mode_distribution: Dict[str, int]

    # Performance metrics
    overall_win_rate: float
    overall_profit_factor: float
    overall_sharpe_ratio: float
    total_pnl: float
    total_pnl_percentage: float

    # Model performance
    model_performance_summary: Dict[str, Dict[str, float]]
    ensemble_performance_summary: Dict[str, Dict[str, float]]

    # HMM regime analysis
    dominant_regimes: Dict[str, int]
    regime_performance: Dict[str, Dict[str, float]]

    # Risk analysis
    max_drawdown: float
    var_95: float
    risk_metrics: Dict[str, float]

    # Export paths
    detailed_decisions_csv: Optional[str] = None
    daily_summaries_csv: Optional[str] = None
    model_performance_csv: Optional[str] = None
    ensemble_analysis_csv: Optional[str] = None

class EnhancedMonitoringOrchestrator:
    """
    Comprehensive monitoring orchestrator that integrates all monitoring components.

    This orchestrator provides:
    1. Context capture (exchange, token, time, price)
    2. Trade indicators (confidence, risk, etc.)
    3. Per-ensemble indicators (weight of each ML model)
    4. Per-ML indicators (confidence, risk, etc.)
    5. Per-ML decision making (weight of each trading indicator)
    6. SHAP/LIME explanations for detailed model insights
    7. Monthly and daily CSV exports
    8. Integration with all trading modes
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced monitoring orchestrator."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedMonitoringOrchestrator")

        # Load configuration
        self.monitor_config = EnhancedMonitoringConfig(**config.get("enhanced_monitoring", {}))

        # Initialize components
        self.enhanced_ml_monitor = EnhancedMLMonitor(config)
        self.ensemble_monitor = EnsembleMonitor(config)
        self.daily_summary_tracker = DailySummaryTracker(config)
        self.explainability_integrator = ExplainabilityIntegrator(config)
        self.trading_integrator = TradingSystemIntegrator(config)

        # Storage
        self.comprehensive_decisions: List[ComprehensiveTradeDecision] = []
        self.monthly_reports: List[MonthlyReport] = []

        # Export directory
        self.export_dir = Path(self.monitor_config.export_directory)
        self.export_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.start_time = datetime.now()
        self.decision_count = 0
        self.last_cleanup = datetime.now()

        self.logger.info("Enhanced Monitoring Orchestrator initialized")

    @handles_errors(default_return=None, context="enhanced_monitoring_orchestrator.record_comprehensive_decision")
    async def record_comprehensive_decision(
        self,
        context: TradeContext,
        trading_mode: TradingMode,
        trading_indicators: List[TradingIndicator],
        ensemble_decision: EnsembleDecision,
        individual_model_decisions: List[MLModelDecision],
        model_indicator_weights: Dict[str, Dict[str, float]],
        action: str,
        position_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
        regime_analysis: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0
    ) -> Optional[ComprehensiveTradeDecision]:
        """Record a comprehensive trade decision with all context and explanations."""
        try:
            start_time = time.time()

            # Generate decision ID
            decision_id = f"{trading_mode.value}_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # Calculate overall metrics
            overall_confidence = self._calculate_overall_confidence(trading_indicators, ensemble_decision, individual_model_decisions)
            overall_risk_score = self._calculate_overall_risk_score(trading_indicators, ensemble_decision, individual_model_decisions)

            # Get SHAP/LIME explanations if enabled
            shap_explanations = None
            lime_explanations = None

            if self.monitor_config.enable_explanations:
                shap_explanations, lime_explanations = await self._get_model_explanations(
                    individual_model_decisions, context
                )

            # Create comprehensive decision
            comprehensive_decision = ComprehensiveTradeDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                trading_mode=trading_mode,
                context=context,
                trading_indicators=trading_indicators,
                overall_confidence=overall_confidence,
                overall_risk_score=overall_risk_score,
                ensemble_decision=ensemble_decision,
                individual_model_decisions=individual_model_decisions,
                model_indicator_weights=model_indicator_weights,
                shap_explanations=shap_explanations,
                lime_explanations=lime_explanations,
                action=action,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                execution_time_ms=execution_time_ms,
                market_conditions=market_conditions,
                regime_analysis=regime_analysis
            )

            # Store decision
            self.comprehensive_decisions.append(comprehensive_decision)
            self.decision_count += 1

            # Maintain memory limit
            if len(self.comprehensive_decisions) > self.monitor_config.max_decisions_in_memory:
                self.comprehensive_decisions = self.comprehensive_decisions[-self.monitor_config.max_decisions_in_memory:]

            # Record in individual components
            await self._record_in_components(comprehensive_decision)

            # Check for exports
            await self._check_and_export_if_needed()

            # Cleanup if needed
            await self._cleanup_if_needed()

            self.logger.info(
                f"Recorded comprehensive decision {decision_id}: "
                f"{action} {context.token} at {context.price} "
                f"(confidence: {overall_confidence:.3f}, risk: {overall_risk_score:.3f})"
            )

            return comprehensive_decision

        except Exception as e:
            self.logger.error(f"Error recording comprehensive decision: {e}")
            return None

    async def _get_model_explanations(
        self,
        model_decisions: List[MLModelDecision],
        context: TradeContext
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Get SHAP and LIME explanations for model decisions."""
        try:
            shap_explanations = {}
            lime_explanations = {}

            for model_decision in model_decisions:
                model_id = model_decision.model_id

                # Get SHAP explanations
                if self.monitor_config.enable_shap and model_decision.shap_values:
                    shap_explanations[model_id] = model_decision.shap_values

                # Get LIME explanations
                if self.monitor_config.enable_lime and model_decision.lime_explanation:
                    lime_explanations[model_id] = model_decision.lime_explanation

            return shap_explanations if shap_explanations else None, lime_explanations if lime_explanations else None

        except Exception as e:
            self.logger.error(f"Error getting model explanations: {e}")
            return None, None

    async def _record_in_components(self, decision: ComprehensiveTradeDecision):
        """Record decision in all monitoring components."""
        try:
            # Convert to standard TradeDecision for compatibility
            standard_decision = TradeDecision(
                decision_id=decision.decision_id,
                context=decision.context,
                trading_mode=decision.trading_mode,
                timestamp=decision.timestamp,
                trading_indicators=decision.trading_indicators,
                overall_confidence=decision.overall_confidence,
                overall_risk_score=decision.overall_risk_score,
                ensemble_decision=decision.ensemble_decision,
                action=decision.action,
                position_size=decision.position_size,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                execution_time_ms=decision.execution_time_ms,
                success_metrics=decision.success_metrics
            )

            # Record in enhanced ML monitor
            await self.enhanced_ml_monitor.record_trade_decision(standard_decision)

            # Record in daily summary tracker
            await self.daily_summary_tracker.add_trade(standard_decision)

            # Record model performance if available
            for model_decision in decision.individual_model_decisions:
                if hasattr(model_decision, 'performance_metrics') and model_decision.performance_metrics:
                    await self.enhanced_ml_monitor.record_model_performance(model_decision.performance_metrics)

            # Record ensemble performance
            if decision.ensemble_decision:
                ensemble_perf = EnsemblePerformanceMetrics(
                    ensemble_id=decision.ensemble_decision.ensemble_id,
                    timestamp=decision.timestamp,
                    accuracy=decision.ensemble_decision.final_confidence,
                    win_rate=0.0,  # Would need historical data
                    profit_factor=1.0,  # Would need historical data
                    sharpe_ratio=0.0,  # Would need historical data
                    model_diversity_score=1.0 - decision.ensemble_decision.disagreement_level,
                    consensus_quality=decision.ensemble_decision.consensus_score,
                    disagreement_impact=decision.ensemble_decision.disagreement_level,
                    weight_stability=1.0,  # Would need historical data
                    model_contributions={}
                )
                await self.enhanced_ml_monitor.record_ensemble_performance(ensemble_perf)

        except Exception as e:
            self.logger.error(f"Error recording in components: {e}")

    def _calculate_overall_confidence(
        self,
        trading_indicators: List[TradingIndicator],
        ensemble_decision: EnsembleDecision,
        model_decisions: List[MLModelDecision]
    ) -> float:
        """Calculate overall confidence from all components."""
        confidence_factors = []

        # Add ensemble confidence
        confidence_factors.append(ensemble_decision.final_confidence)

        # Add trading indicator confidences
        for indicator in trading_indicators:
            confidence_factors.append(indicator.confidence)

        # Add model confidences
        for model_decision in model_decisions:
            confidence_factors.append(model_decision.confidence)

        return np.mean(confidence_factors) if confidence_factors else 0.0

    def _calculate_overall_risk_score(
        self,
        trading_indicators: List[TradingIndicator],
        ensemble_decision: EnsembleDecision,
        model_decisions: List[MLModelDecision]
    ) -> float:
        """Calculate overall risk score from all components."""
        risk_factors = []

        # Add ensemble risk
        risk_factors.append(ensemble_decision.final_risk_score)

        # Add trading indicator risks
        for indicator in trading_indicators:
            risk_factors.append(indicator.risk_score)

        # Add model risks
        for model_decision in model_decisions:
            risk_factors.append(model_decision.risk_score)

        return np.mean(risk_factors) if risk_factors else 0.5

    async def _check_and_export_if_needed(self):
        """Check if it's time to export data."""
        try:
            current_time = datetime.now()

            # Check for monthly export
            if self.monitor_config.monthly_export_enabled:
                if (current_time - self.start_time).days >= 30:
                    await self.export_monthly_report()
                    self.start_time = current_time  # Reset timer

        except Exception as e:
            self.logger.error(f"Error checking export timing: {e}")

    async def _cleanup_if_needed(self):
        """Clean up old data if needed."""
        try:
            current_time = datetime.now()

            if (current_time - self.last_cleanup).total_seconds() >= self.monitor_config.cleanup_frequency_hours * 3600:
                await self._cleanup_old_data()
                self.last_cleanup = current_time

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.monitor_config.data_retention_days)

            # Clean up comprehensive decisions
            original_count = len(self.comprehensive_decisions)
            self.comprehensive_decisions = [
                d for d in self.comprehensive_decisions
                if d.timestamp >= cutoff_date
            ]
            removed_count = original_count - len(self.comprehensive_decisions)

            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old decisions")

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    @handles_errors(default_return=False, context="enhanced_monitoring_orchestrator.export_monthly_report")
    async def export_monthly_report(self) -> bool:
        """Export comprehensive monthly monitoring report."""
        try:
            current_time = datetime.now()
            month_str = current_time.strftime("%Y-%m")
            start_date = current_time.replace(day=1).date()

            # Calculate end date (last day of month)
            if current_time.month == 12:
                end_date = current_time.replace(year=current_time.year + 1, month=1, day=1).date() - timedelta(days=1)
            else:
                end_date = current_time.replace(month=current_time.month + 1, day=1).date() - timedelta(days=1)

            # Get decisions for this month
            month_decisions = [
                d for d in self.comprehensive_decisions
                if start_date <= d.timestamp.date() <= end_date
            ]

            if not month_decisions:
                self.logger.warning(f"No decisions found for month {month_str}")
                return False

            # Generate report
            report = await self._generate_monthly_report(month_str, start_date, end_date, month_decisions)

            # Export detailed data
            await self._export_monthly_data(report, month_decisions)

            # Store report
            self.monthly_reports.append(report)

            self.logger.info(f"Exported monthly report for {month_str}: {len(month_decisions)} decisions")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting monthly report: {e}")
            return False

    async def _generate_monthly_report(
        self,
        month_str: str,
        start_date: date,
        end_date: date,
        decisions: List[ComprehensiveTradeDecision]
    ) -> MonthlyReport:
        """Generate comprehensive monthly report."""
        try:
            # Basic statistics
            total_decisions = len(decisions)
            total_trades = sum(1 for d in decisions if d.action in ["buy", "sell"])

            # Trading mode distribution
            mode_distribution = {}
            for decision in decisions:
                mode = decision.trading_mode.value
                mode_distribution[mode] = mode_distribution.get(mode, 0) + 1

            # Performance metrics
            pnls = []
            wins = 0
            for decision in decisions:
                if decision.success_metrics and 'profit_loss' in decision.success_metrics:
                    pnl = decision.success_metrics['profit_loss']
                    pnls.append(pnl)
                    if pnl > 0:
                        wins += 1

            overall_win_rate = wins / len(pnls) if pnls else 0.0
            total_pnl = sum(pnls) if pnls else 0.0

            # Calculate profit factor
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            overall_profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

            # Calculate Sharpe ratio
            if pnls:
                returns = np.array(pnls)
                overall_sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            else:
                overall_sharpe_ratio = 0.0

            # Calculate total PnL percentage (would need initial capital)
            total_pnl_percentage = 0.0  # Would need initial capital to calculate

            # Model performance summary
            model_performance_summary = self._calculate_model_performance_summary(decisions)

            # Ensemble performance summary
            ensemble_performance_summary = self._calculate_ensemble_performance_summary(decisions)

            # HMM regime analysis
            dominant_regimes, regime_performance = self._calculate_regime_analysis(decisions)

            # Risk analysis
            max_drawdown = self._calculate_max_drawdown(pnls)
            var_95 = np.percentile(pnls, 5) if pnls else 0.0
            risk_metrics = {
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'max_loss': min(pnls) if pnls else 0.0,
                'max_gain': max(pnls) if pnls else 0.0
            }

            return MonthlyReport(
                report_id=f"monthly_report_{month_str}_{uuid.uuid4().hex[:8]}",
                month=month_str,
                start_date=start_date,
                end_date=end_date,
                generated_at=datetime.now(),
                total_decisions=total_decisions,
                total_trades=total_trades,
                trading_mode_distribution=mode_distribution,
                overall_win_rate=overall_win_rate,
                overall_profit_factor=overall_profit_factor,
                overall_sharpe_ratio=overall_sharpe_ratio,
                total_pnl=total_pnl,
                total_pnl_percentage=total_pnl_percentage,
                model_performance_summary=model_performance_summary,
                ensemble_performance_summary=ensemble_performance_summary,
                dominant_regimes=dominant_regimes,
                regime_performance=regime_performance,
                max_drawdown=max_drawdown,
                var_95=var_95,
                risk_metrics=risk_metrics
            )

        except Exception as e:
            self.logger.error(f"Error generating monthly report: {e}")
            raise

    def _calculate_model_performance_summary(self, decisions: List[ComprehensiveTradeDecision]) -> Dict[str, Dict[str, float]]:
        """Calculate model performance summary."""
        model_performance = {}

        for decision in decisions:
            for model_decision in decision.individual_model_decisions:
                model_id = model_decision.model_id

                if model_id not in model_performance:
                    model_performance[model_id] = {
                        'total_predictions': 0,
                        'avg_confidence': 0.0,
                        'avg_risk_score': 0.0,
                        'total_processing_time': 0.0
                    }

                model_performance[model_id]['total_predictions'] += 1
                model_performance[model_id]['avg_confidence'] += model_decision.confidence
                model_performance[model_id]['avg_risk_score'] += model_decision.risk_score
                model_performance[model_id]['total_processing_time'] += model_decision.processing_time_ms

        # Calculate averages
        for model_id in model_performance:
            total = model_performance[model_id]['total_predictions']
            if total > 0:
                model_performance[model_id]['avg_confidence'] /= total
                model_performance[model_id]['avg_risk_score'] /= total

        return model_performance

    def _calculate_ensemble_performance_summary(self, decisions: List[ComprehensiveTradeDecision]) -> Dict[str, Dict[str, float]]:
        """Calculate ensemble performance summary."""
        ensemble_performance = {}

        for decision in decisions:
            ensemble_id = decision.ensemble_decision.ensemble_id

            if ensemble_id not in ensemble_performance:
                ensemble_performance[ensemble_id] = {
                    'total_decisions': 0,
                    'avg_confidence': 0.0,
                    'avg_risk_score': 0.0,
                    'avg_consensus_score': 0.0,
                    'avg_disagreement_level': 0.0
                }

            ensemble_performance[ensemble_id]['total_decisions'] += 1
            ensemble_performance[ensemble_id]['avg_confidence'] += decision.ensemble_decision.final_confidence
            ensemble_performance[ensemble_id]['avg_risk_score'] += decision.ensemble_decision.final_risk_score
            ensemble_performance[ensemble_id]['avg_consensus_score'] += decision.ensemble_decision.consensus_score
            ensemble_performance[ensemble_id]['avg_disagreement_level'] += decision.ensemble_decision.disagreement_level

        # Calculate averages
        for ensemble_id in ensemble_performance:
            total = ensemble_performance[ensemble_id]['total_decisions']
            if total > 0:
                ensemble_performance[ensemble_id]['avg_confidence'] /= total
                ensemble_performance[ensemble_id]['avg_risk_score'] /= total
                ensemble_performance[ensemble_id]['avg_consensus_score'] /= total
                ensemble_performance[ensemble_id]['avg_disagreement_level'] /= total

        return ensemble_performance

    def _calculate_regime_analysis(self, decisions: List[ComprehensiveTradeDecision]) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
        """Calculate HMM regime analysis."""
        regime_counts = {}
        regime_performance = {}

        for decision in decisions:
            if decision.regime_analysis and 'regime_id' in decision.regime_analysis:
                regime_id = decision.regime_analysis['regime_id']
                regime_counts[regime_id] = regime_counts.get(regime_id, 0) + 1

                if regime_id not in regime_performance:
                    regime_performance[regime_id] = {
                        'total_decisions': 0,
                        'avg_confidence': 0.0,
                        'avg_risk_score': 0.0,
                        'total_pnl': 0.0
                    }

                regime_performance[regime_id]['total_decisions'] += 1
                regime_performance[regime_id]['avg_confidence'] += decision.overall_confidence
                regime_performance[regime_id]['avg_risk_score'] += decision.overall_risk_score

                if decision.success_metrics and 'profit_loss' in decision.success_metrics:
                    regime_performance[regime_id]['total_pnl'] += decision.success_metrics['profit_loss']

        # Calculate averages
        for regime_id in regime_performance:
            total = regime_performance[regime_id]['total_decisions']
            if total > 0:
                regime_performance[regime_id]['avg_confidence'] /= total
                regime_performance[regime_id]['avg_risk_score'] /= total

        return regime_counts, regime_performance

    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnls:
            return 0.0

        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    async def _export_monthly_data(self, report: MonthlyReport, decisions: List[ComprehensiveTradeDecision]):
        """Export detailed monthly data to CSV files."""
        try:
            month_dir = self.export_dir / f"monthly_reports_{report.month}"
            month_dir.mkdir(exist_ok=True)

            # Export comprehensive decisions
            decisions_data = []
            for decision in decisions:
                decision_dict = asdict(decision)

                # Convert datetime objects to strings
                for key, value in decision_dict.items():
                    if isinstance(value, datetime):
                        decision_dict[key] = value.isoformat()
                    elif isinstance(value, date):
                        decision_dict[key] = value.isoformat()

                decisions_data.append(decision_dict)

            decisions_df = pd.DataFrame(decisions_data)
            decisions_path = month_dir / f"comprehensive_decisions_{report.month}.csv"
            decisions_df.to_csv(decisions_path, index=False)
            report.detailed_decisions_csv = str(decisions_path)

            # Export daily summaries
            daily_summaries = await self.daily_summary_tracker.get_summary_range(
                report.start_date, report.end_date
            )
            if daily_summaries:
                daily_data = [asdict(summary) for summary in daily_summaries]
                daily_df = pd.DataFrame(daily_data)
                daily_path = month_dir / f"daily_summaries_{report.month}.csv"
                daily_df.to_csv(daily_path, index=False)
                report.daily_summaries_csv = str(daily_path)

            # Export model performance
            model_perf_data = []
            for model_id, perf in report.model_performance_summary.items():
                perf_dict = {'model_id': model_id}
                perf_dict.update(perf)
                model_perf_data.append(perf_dict)

            if model_perf_data:
                model_perf_df = pd.DataFrame(model_perf_data)
                model_perf_path = month_dir / f"model_performance_{report.month}.csv"
                model_perf_df.to_csv(model_perf_path, index=False)
                report.model_performance_csv = str(model_perf_path)

            # Export ensemble analysis
            ensemble_data = []
            for ensemble_id, perf in report.ensemble_performance_summary.items():
                perf_dict = {'ensemble_id': ensemble_id}
                perf_dict.update(perf)
                ensemble_data.append(perf_dict)

            if ensemble_data:
                ensemble_df = pd.DataFrame(ensemble_data)
                ensemble_path = month_dir / f"ensemble_analysis_{report.month}.csv"
                ensemble_df.to_csv(ensemble_path, index=False)
                report.ensemble_analysis_csv = str(ensemble_path)

            # Export report summary
            report_dict = asdict(report)
            for key, value in report_dict.items():
                if isinstance(value, datetime):
                    report_dict[key] = value.isoformat()
                elif isinstance(value, date):
                    report_dict[key] = value.isoformat()

            report_path = month_dir / f"monthly_report_summary_{report.month}.json"
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2)

            self.logger.info(f"Exported monthly data for {report.month} to {month_dir}")

        except Exception as e:
            self.logger.error(f"Error exporting monthly data: {e}")

    @handles_errors(default_return=False, context="enhanced_monitoring_orchestrator.export_daily_ongoing_csv")
    async def export_daily_ongoing_csv(self) -> bool:
        """Export ongoing daily CSV with main metrics."""
        try:
            # Get current date
            current_date = date.today()

            # Get daily summary for today
            daily_summary = await self.daily_summary_tracker.get_daily_summary(current_date)

            if not daily_summary:
                self.logger.warning(f"No daily summary available for {current_date}")
                return False

            # Create ongoing CSV data
            ongoing_data = {
                'date': current_date.isoformat(),
                'exchange': 'all',  # Would need to aggregate from decisions
                'asset': 'all',     # Would need to aggregate from decisions
                'total_trades': daily_summary.total_trades,
                'long_trades': daily_summary.long_trades,
                'short_trades': daily_summary.short_trades,
                'hold_trades': daily_summary.hold_trades,
                'dominant_hmm_clusters': daily_summary.dominant_regime,
                'sharpe_ratio': daily_summary.sharpe_ratio,
                'pnl_absolute': daily_summary.total_pnl,
                'pnl_percentage': 0.0,  # Would need initial capital
                'win_rate': daily_summary.win_rate,
                'profit_factor': daily_summary.profit_factor,
                'max_drawdown': daily_summary.max_drawdown,
                'avg_confidence': daily_summary.avg_confidence,
                'avg_risk_score': daily_summary.avg_risk_score,
                'model_accuracy_avg': daily_summary.model_accuracy_avg,
                'ensemble_consensus_avg': daily_summary.ensemble_consensus_avg
            }

            # Create or append to ongoing CSV
            ongoing_csv_path = self.export_dir / "ongoing_daily_metrics.csv"

            if ongoing_csv_path.exists():
                # Append to existing file
                existing_df = pd.read_csv(ongoing_csv_path)
                new_df = pd.DataFrame([ongoing_data])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(ongoing_csv_path, index=False)
            else:
                # Create new file
                df = pd.DataFrame([ongoing_data])
                df.to_csv(ongoing_csv_path, index=False)

            self.logger.info(f"Exported ongoing daily metrics to {ongoing_csv_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting daily ongoing CSV: {e}")
            return False

    @handles_errors(default_return=False, context="enhanced_monitoring_orchestrator.integrate_trading_systems")
    async def integrate_trading_systems(
        self,
        backtesting_system: Optional[Any] = None,
        paper_trading_system: Optional[Any] = None,
        live_trading_system: Optional[Any] = None
    ) -> bool:
        """Integrate with trading systems for automatic monitoring."""
        try:
            if not self.monitor_config.auto_integrate_trading_systems:
                self.logger.info("Auto-integration disabled")
                return True

            success = True

            # Integrate with backtesting system
            if backtesting_system:
                integration_success = await self.trading_integrator.integrate_backtesting(backtesting_system)
                if not integration_success:
                    self.logger.warning("Failed to integrate with backtesting system")
                    success = False

            # Integrate with paper trading system
            if paper_trading_system:
                integration_success = await self.trading_integrator.integrate_paper_trading(paper_trading_system)
                if not integration_success:
                    self.logger.warning("Failed to integrate with paper trading system")
                    success = False

            # Integrate with live trading system
            if live_trading_system:
                integration_success = await self.trading_integrator.integrate_live_trading(live_trading_system)
                if not integration_success:
                    self.logger.warning("Failed to integrate with live trading system")
                    success = False

            if success:
                self.logger.info("Successfully integrated with all provided trading systems")
            else:
                self.logger.warning("Some trading system integrations failed")

            return success

        except Exception as e:
            self.logger.error(f"Error integrating trading systems: {e}")
            return False

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return {
            'orchestrator_stats': {
                'total_comprehensive_decisions': len(self.comprehensive_decisions),
                'decision_count': self.decision_count,
                'monitoring_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'monthly_reports_generated': len(self.monthly_reports),
                'export_directory': str(self.export_dir)
            },
            'enhanced_ml_monitor_stats': self.enhanced_ml_monitor.get_monitoring_stats(),
            'ensemble_monitor_stats': self.ensemble_monitor.get_ensemble_stats(),
            'daily_summary_tracker_stats': self.daily_summary_tracker.get_tracker_stats(),
            'trading_integrator_stats': self.trading_integrator.get_integration_stats()
        }

    @handles_errors(default_return=False, context="enhanced_monitoring_orchestrator.force_export_all")
    async def force_export_all(self) -> bool:
        """Force export of all monitoring data."""
        try:
            # Export monthly report
            monthly_success = await self.export_monthly_report()

            # Export daily ongoing CSV
            daily_success = await self.export_daily_ongoing_csv()

            # Force export from components
            component_success = await self.enhanced_ml_monitor.force_export()

            return monthly_success and daily_success and component_success

        except Exception as e:
            self.logger.error(f"Error in force export all: {e}")
            return False
