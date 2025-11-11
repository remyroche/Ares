"""
Performance Reporter

Comprehensive performance reporting system for trading operations
with detailed analytics, ML model performance, and risk analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel
from src.utils.tprint import tprint
from ..monitoring.comprehensive_trade_monitor import DetailedTradeMetrics, TradingSessionMetrics
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler
from ..utils.helpers import format_trading_metrics, calculate_sharpe_ratio, calculate_max_drawdown

logger = system_logger.getChild('PerformanceReporter')

class PerformanceReporter:
    """
    Comprehensive performance reporting system for trading operations.

    Generates detailed reports with:
    - Trade-by-trade analysis
    - ML model performance breakdown
    - SHAP/LIME explanation summaries
    - Risk analysis
    - Regime performance analysis
    - Execution quality metrics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tprint(f"PerformanceReporter.__init__: Called")
        tprint(f"PerformanceReporter.__init__: Initializing with config keys={list((config or {}).keys())}")
        self.config = config or {}
        self.logger = logger.getChild('PerformanceReporter')

        # Configuration constants
        self.default_account_size = self.config.get('default_account_size', 10000.0)

        # Report configuration
        self.report_directory = Path(self.config.get('report_directory', 'trading_reports'))
        self.enable_html_reports = self.config.get('enable_html_reports', True)
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)

        # Ensure report directory exists
        self.report_directory.mkdir(parents=True, exist_ok=True)

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def generate_comprehensive_report(
        self,
        trades: List[DetailedTradeMetrics],
        session_metrics: Optional[TradingSessionMetrics] = None,
        report_name: str = "trading_performance"
    ) -> Dict[str, Any]:
"""
        Generate comprehensive trading performance report.

        Args:
            trades: List of detailed trade metrics
            session_metrics: Session-level metrics
            report_name: Name for the report

        Returns:
            Comprehensive report dictionary
        """
        tprint(f"PerformanceReporter.generate_comprehensive_report: Called")
        tprint(f"PerformanceReporter.generate_comprehensive_report: Starting report {report_name} with {len(trades)} trades")
        try:
            tprint_info(f"📊 Generating comprehensive trading report: {report_name}")

            if not trades:
                tprint_warning("⚠️ No trades provided for report generation")
                return {'error': 'No trades available'}

            # Generate report sections
            report = {
                'report_metadata': await self._generate_report_metadata(report_name),
                'executive_summary': await self._generate_executive_summary(trades, session_metrics),
                'trade_analysis': await self._generate_trade_analysis(trades),
                'model_performance': await self._generate_model_performance_analysis(trades),
                'explainability_analysis': await self._generate_explainability_analysis(trades),
                'risk_analysis': await self._generate_risk_analysis(trades),
                'regime_analysis': await self._generate_regime_analysis(trades),
                'execution_analysis': await self._generate_execution_analysis(trades),
                'detailed_trades': [trade.to_dict() for trade in trades]
            }

            # Export report
            await self._export_report(report, report_name)

            tprint_success(f"✅ Generated comprehensive report with {len(trades)} trades")
            tprint(f"PerformanceReporter.generate_comprehensive_report: Report complete, returning {len(report)} sections")
            return report

        except Exception as e:
            tprint_error(f"❌ Failed to generate comprehensive report: {e}")
            return {'error': str(e)}

    async def _generate_report_metadata(self, report_name: str) -> Dict[str, Any]:
"""Generate report metadata."""
        tprint(f"PerformanceReporter._generate_report_metadata: Called")
        return {
            'report_name': report_name,
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
            'generator': 'PerformanceReporter',
            'report_type': 'comprehensive_trading_analysis'
        }

    async def _generate_executive_summary(
        self,
        trades: List[DetailedTradeMetrics],
        session_metrics: Optional[TradingSessionMetrics]
    ) -> Dict[str, Any]:
"""Generate executive summary of trading performance."""
        tprint(f"PerformanceReporter._generate_executive_summary: Called")
        try:
            # Basic statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl_absolute and t.pnl_absolute > 0])
            losing_trades = len([t for t in trades if t.pnl_absolute and t.pnl_absolute < 0])

            # PnL analysis
            total_pnl = sum(t.pnl_absolute for t in trades if t.pnl_absolute is not None)
            pnl_values = [t.pnl_absolute for t in trades if t.pnl_absolute is not None]

            # Performance metrics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            avg_win = np.mean([t.pnl_absolute for t in trades if t.pnl_absolute and t.pnl_absolute > 0]) if winning_trades > 0 else 0.0
            avg_loss = np.mean([t.pnl_absolute for t in trades if t.pnl_absolute and t.pnl_absolute < 0]) if losing_trades > 0 else 0.0
            # Profit factor: inf when no losses indicates perfect performance
            if avg_loss != 0 and losing_trades > 0:
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades))
            elif avg_win > 0 and winning_trades > 0:
                profit_factor = float('inf')  # Perfect performance - no losses
            else:
                profit_factor = 0.0

            # Risk metrics
            # Convert absolute PnL to returns for Sharpe ratio calculation
            if len(pnl_values) > 1:
                # Calculate returns as percentage changes
                # Use a base value to normalize (first value or average)
                base_value = abs(pnl_values[0]) if pnl_values[0] != 0 else np.mean([abs(p) for p in pnl_values if p != 0]) or self.default_account_size
                returns = np.array(pnl_values) / base_value
                sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 1 else 0.0
            else:
                sharpe_ratio = 0.0
            max_drawdown_pct, _, _ = calculate_max_drawdown(np.cumsum(pnl_values)) if pnl_values else (0.0, 0, 0)

            # Model usage summary
            model_usage = {}
            for trade in trades:
                for model_id in trade.models_used.keys():
                    model_usage[model_id] = model_usage.get(model_id, 0) + 1

            # Confidence analysis
            confidence_scores = [t.signal_confidence for t in trades if t.signal_confidence > 0]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

            summary = {
                'performance_overview': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'profit_factor': profit_factor,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown_pct
                },
                'key_metrics': {
                    'average_win': avg_win,
                    'average_loss': avg_loss,
                    'average_confidence': avg_confidence,
                    'best_trade': max(pnl_values) if pnl_values else 0.0,
                    'worst_trade': min(pnl_values) if pnl_values else 0.0
                },
                'model_summary': {
                    'models_used': len(model_usage),
                    'most_used_model': max(model_usage.items(), key=lambda x: x[1]) if model_usage else None,
                    'model_usage_distribution': model_usage
                },
                'trading_period': {
                    'start_time': min(t.timestamp for t in trades).isoformat() if trades else None,
                    'end_time': max(t.timestamp for t in trades).isoformat() if trades else None,
                    'duration_hours': (max(t.timestamp for t in trades) - min(t.timestamp for t in trades)).total_seconds() / 3600 if len(trades) > 1 else 0.0
                }
            }

            return summary

        except Exception as e:
            tprint_error(f"❌ Failed to generate executive summary: {e}")
            return {}

    async def _generate_trade_analysis(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Generate detailed trade-by-trade analysis."""
        tprint(f"PerformanceReporter._generate_trade_analysis: Called")
        try:
            # Trade distribution by action
            action_distribution = {}
            for trade in trades:
                action_distribution[trade.action] = action_distribution.get(trade.action, 0) + 1

            # Trade distribution by symbol
            symbol_distribution = {}
            for trade in trades:
                symbol_distribution[trade.symbol] = symbol_distribution.get(trade.symbol, 0) + 1

            # Performance by action
            performance_by_action = {}
            for action in action_distribution.keys():
                action_trades = [t for t in trades if t.action == action]
                action_pnl = [t.pnl_absolute for t in action_trades if t.pnl_absolute is not None]

                if action_pnl:
                    performance_by_action[action] = {
                        'trade_count': len(action_trades),
                        'total_pnl': sum(action_pnl),
                        'avg_pnl': np.mean(action_pnl),
                        'win_rate': len([p for p in action_pnl if p > 0]) / len(action_pnl),
                        'best_trade': max(action_pnl),
                        'worst_trade': min(action_pnl)
                    }

            # Timing analysis
            timing_analysis = await self._analyze_trade_timing(trades)

            # Size analysis
            size_analysis = await self._analyze_position_sizes(trades)

            return {
                'trade_distribution': {
                    'by_action': action_distribution,
                    'by_symbol': symbol_distribution
                },
                'performance_by_action': performance_by_action,
                'timing_analysis': timing_analysis,
                'position_size_analysis': size_analysis,
                'trade_quality_metrics': await self._calculate_trade_quality_metrics(trades)
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate trade analysis: {e}")
            return {}

    async def _generate_model_performance_analysis(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Generate ML model performance analysis."""
        tprint(f"PerformanceReporter._generate_model_performance_analysis: Called")
        try:
            model_performance = {}

            # Collect all models used
            all_models = set()
            for trade in trades:
                all_models.update(trade.models_used.keys())

            # Analyze each model
            for model_id in all_models:
                model_trades = [t for t in trades if model_id in t.models_used]

                if model_trades:
                    # Basic performance
                    model_pnl = [t.pnl_absolute for t in model_trades if t.pnl_absolute is not None]
                    model_confidences = [t.model_confidences.get(model_id, 0.0) for t in model_trades]
                    model_weights = [t.model_weights.get(model_id, 0.0) for t in model_trades]

                    # Model accuracy (correlation between confidence and success)
                    successful_trades = [t for t in model_trades if t.pnl_absolute and t.pnl_absolute > 0]
                    accuracy = len(successful_trades) / len(model_trades) if model_trades else 0.0

                    # Confidence analysis
                    high_confidence_trades = [t for t in model_trades if t.model_confidences.get(model_id, 0.0) > 0.7]
                    high_conf_success_rate = len([t for t in high_confidence_trades if t.pnl_absolute and t.pnl_absolute > 0]) / len(high_confidence_trades) if high_confidence_trades else 0.0

                    model_performance[model_id] = {
                        'usage_count': len(model_trades),
                        'total_pnl': sum(model_pnl),
                        'avg_pnl': np.mean(model_pnl) if model_pnl else 0.0,
                        'accuracy': accuracy,
                        'avg_confidence': np.mean(model_confidences) if model_confidences else 0.0,
                        'avg_weight': np.mean(model_weights) if model_weights else 0.0,
                        'high_confidence_success_rate': high_conf_success_rate,
                        'confidence_pnl_correlation': (
                            np.corrcoef(model_confidences, model_pnl)[0, 1]
                            if len(model_confidences) > 1
                            and len(model_pnl) > 1
                            and len(model_confidences) == len(model_pnl)
                            else 0.0
                        )
                    }

            # Model comparison
            model_comparison = await self._compare_model_performance(model_performance)

            return {
                'individual_model_performance': model_performance,
                'model_comparison': model_comparison,
                'ensemble_analysis': await self._analyze_ensemble_performance(trades)
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate model performance analysis: {e}")
            return {}

    async def _generate_explainability_analysis(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Generate SHAP/LIME explainability analysis."""
        tprint(f"PerformanceReporter._generate_explainability_analysis: Called")
        try:
            # Feature importance analysis
            all_features = set()
            feature_importance_by_model = {}

            for trade in trades:
                # Collect SHAP explanations
                for model_id, shap_values in trade.shap_explanations.items():
                    if model_id not in feature_importance_by_model:
                        feature_importance_by_model[model_id] = {}

                    for feature, importance in shap_values.items():
                        all_features.add(feature)
                        if feature not in feature_importance_by_model[model_id]:
                            feature_importance_by_model[model_id][feature] = []
                        feature_importance_by_model[model_id][feature].append(abs(importance))

            # Calculate average feature importance
            avg_feature_importance = {}
            for model_id, features in feature_importance_by_model.items():
                avg_feature_importance[model_id] = {
                    feature: np.mean(importances)
                    for feature, importances in features.items()
                }

            # Overall feature importance (across all models)
            overall_feature_importance = {}
            for feature in all_features:
                importances = []
                for model_id in feature_importance_by_model:
                    if feature in feature_importance_by_model[model_id]:
                        importances.extend(feature_importance_by_model[model_id][feature])

                if importances:
                    overall_feature_importance[feature] = np.mean(importances)

            # Top features
            top_features = sorted(overall_feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

            # Model agreement analysis
            model_agreement = await self._analyze_model_agreement(trades)

            return {
                'feature_importance_by_model': avg_feature_importance,
                'overall_feature_importance': overall_feature_importance,
                'top_features': dict(top_features),
                'model_agreement_analysis': model_agreement,
                'explanation_coverage': {
                    'trades_with_shap': len([t for t in trades if t.shap_explanations]),
                    'trades_with_lime': len([t for t in trades if t.lime_explanations]),
                    'total_trades': len(trades),
                    'shap_coverage': len([t for t in trades if t.shap_explanations]) / len(trades) if trades else 0.0,
                    'lime_coverage': len([t for t in trades if t.lime_explanations]) / len(trades) if trades else 0.0
                }
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate explainability analysis: {e}")
            return {}

    async def _generate_risk_analysis(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Generate comprehensive risk analysis."""
        tprint(f"PerformanceReporter._generate_risk_analysis: Called")
        try:
            # Risk metrics aggregation
            portfolio_risks = [t.portfolio_risk for t in trades if t.portfolio_risk > 0]
            var_95_values = [t.var_95 for t in trades if t.var_95 > 0]
            volatility_estimates = [t.volatility_estimate for t in trades if t.volatility_estimate > 0]

            # Position size analysis
            position_sizes = [t.position_size for t in trades if t.position_size > 0]
            leverages = [t.leverage for t in trades if t.leverage > 0]

            # Risk-adjusted performance
            risk_adjusted_returns = []
            for trade in trades:
                if trade.pnl_percentage and trade.portfolio_risk and trade.portfolio_risk > 0:
                    risk_adjusted_return = trade.pnl_percentage / trade.portfolio_risk
                    risk_adjusted_returns.append(risk_adjusted_return)

            # Drawdown analysis
            pnl_series = [t.pnl_absolute for t in trades if t.pnl_absolute is not None]
            if pnl_series:
                cumulative_pnl = np.cumsum(pnl_series)
                max_dd, dd_start, dd_end = calculate_max_drawdown(cumulative_pnl)

                # Drawdown periods
                drawdown_periods = await self._identify_drawdown_periods(cumulative_pnl)
            else:
                max_dd = 0.0
                drawdown_periods = []

            return {
                'risk_metrics_summary': {
                    'avg_portfolio_risk': np.mean(portfolio_risks) if portfolio_risks else 0.0,
                    'max_portfolio_risk': max(portfolio_risks) if portfolio_risks else 0.0,
                    'avg_var_95': np.mean(var_95_values) if var_95_values else 0.0,
                    'avg_volatility_estimate': np.mean(volatility_estimates) if volatility_estimates else 0.0
                },
                'position_analysis': {
                    'avg_position_size': np.mean(position_sizes) if position_sizes else 0.0,
                    'max_position_size': max(position_sizes) if position_sizes else 0.0,
                    'avg_leverage': np.mean(leverages) if leverages else 0.0,
                    'max_leverage': max(leverages) if leverages else 0.0
                },
                'risk_adjusted_performance': {
                    'avg_risk_adjusted_return': np.mean(risk_adjusted_returns) if risk_adjusted_returns else 0.0,
                    'risk_adjusted_sharpe': np.mean(risk_adjusted_returns) / np.std(risk_adjusted_returns) if len(risk_adjusted_returns) > 1 else 0.0
                },
                'drawdown_analysis': {
                    'max_drawdown': max_dd,
                    'drawdown_periods': drawdown_periods,
                    'avg_drawdown_duration': np.mean([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0.0
                }
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate risk analysis: {e}")
            return {}

    async def _generate_regime_analysis(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Generate regime-based performance analysis."""
        tprint(f"PerformanceReporter._generate_regime_analysis: Called")
        try:
            regime_performance = {}

            # Group trades by regime
            for trade in trades:
                regime = trade.regime_type
                if regime not in regime_performance:
                    regime_performance[regime] = {
                        'trades': [],
                        'total_pnl': 0.0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'avg_confidence': 0.0,
                        'avg_regime_confidence': 0.0
                    }

                regime_performance[regime]['trades'].append(trade)

                if trade.pnl_absolute is not None:
                    regime_performance[regime]['total_pnl'] += trade.pnl_absolute
                    if trade.pnl_absolute > 0:
                        regime_performance[regime]['winning_trades'] += 1
                    else:
                        regime_performance[regime]['losing_trades'] += 1

            # Calculate regime statistics
            for regime, data in regime_performance.items():
                trades_in_regime = data['trades']
                total_trades = len(trades_in_regime)

                if total_trades > 0:
                    data['trade_count'] = total_trades
                    data['win_rate'] = data['winning_trades'] / total_trades
                    data['avg_confidence'] = np.mean([t.signal_confidence for t in trades_in_regime])
                    data['avg_regime_confidence'] = np.mean([t.regime_confidence for t in trades_in_regime])
                    data['avg_pnl_per_trade'] = data['total_pnl'] / total_trades

                    # Remove trade objects for JSON serialization
                    del data['trades']

            # Best and worst performing regimes
            regime_pnl = {regime: data['total_pnl'] for regime, data in regime_performance.items()}
            best_regime = max(regime_pnl.items(), key=lambda x: x[1]) if regime_pnl else None
            worst_regime = min(regime_pnl.items(), key=lambda x: x[1]) if regime_pnl else None

            return {
                'regime_performance': regime_performance,
                'regime_summary': {
                    'regimes_traded': len(regime_performance),
                    'best_performing_regime': best_regime,
                    'worst_performing_regime': worst_regime,
                    'regime_distribution': {regime: data['trade_count'] for regime, data in regime_performance.items()}
                }
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate regime analysis: {e}")
            return {}

    async def _generate_execution_analysis(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Generate execution quality analysis."""
        tprint(f"PerformanceReporter._generate_execution_analysis: Called")
        try:
            # Execution metrics
            execution_times = [t.execution_time_ms for t in trades if t.execution_time_ms > 0]
            slippages = [t.slippage for t in trades if t.slippage is not None]
            commissions = [t.commission for t in trades if t.commission is not None]
            execution_qualities = [t.execution_quality for t in trades if t.execution_quality > 0]
            timing_qualities = [t.timing_quality for t in trades if t.timing_quality > 0]

            # Execution success analysis
            successful_executions = len([t for t in trades if t.execution_quality > 0.8])
            execution_success_rate = successful_executions / len(trades) if trades else 0.0

            # Cost analysis
            total_commissions = sum(commissions) if commissions else 0.0
            total_slippage_cost = sum(slippages) if slippages else 0.0

            return {
                'execution_metrics': {
                    'avg_execution_time_ms': np.mean(execution_times) if execution_times else None,
                    'max_execution_time_ms': max(execution_times) if execution_times else None,
                    'avg_slippage': np.mean(slippages) if slippages else None,
                    'max_slippage': max(slippages) if slippages else None,
                    # Return None if no data to distinguish from poor quality (0.0)
                    'avg_execution_quality': np.mean(execution_qualities) if execution_qualities else None,
                    'avg_timing_quality': np.mean(timing_qualities) if timing_qualities else None
                },
                'execution_success': {
                    'success_rate': execution_success_rate,
                    'successful_executions': successful_executions,
                    'total_executions': len(trades)
                },
                'cost_analysis': {
                    'total_commissions': total_commissions,
                    'total_slippage_cost': total_slippage_cost,
                    'avg_commission_per_trade': total_commissions / len(trades) if trades else 0.0,
                    'avg_slippage_per_trade': total_slippage_cost / len(trades) if trades else 0.0
                }
            }

        except Exception as e:
            tprint_error(f"❌ Failed to generate execution analysis: {e}")
            return {}

    async def _analyze_trade_timing(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Analyze trade timing patterns."""
        tprint(f"PerformanceReporter._analyze_trade_timing: Called")
        try:
            # Time distribution analysis
            hour_distribution = {}
            day_distribution = {}

            for trade in trades:
                hour = trade.timestamp.hour
                day = trade.timestamp.strftime('%A')

                hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
                day_distribution[day] = day_distribution.get(day, 0) + 1

            # Performance by time
            performance_by_hour = {}
            for hour in range(24):
                hour_trades = [t for t in trades if t.timestamp.hour == hour]
                if hour_trades:
                    hour_pnl = [t.pnl_absolute for t in hour_trades if t.pnl_absolute is not None]
                    performance_by_hour[hour] = {
                        'trade_count': len(hour_trades),
                        'total_pnl': sum(hour_pnl) if hour_pnl else 0.0,
                        'avg_pnl': np.mean(hour_pnl) if hour_pnl else 0.0
                    }

            return {
                'time_distribution': {
                    'by_hour': hour_distribution,
                    'by_day': day_distribution
                },
                'performance_by_hour': performance_by_hour,
                'best_trading_hours': sorted(performance_by_hour.items(), key=lambda x: x[1]['avg_pnl'], reverse=True)[:3]
            }

        except Exception as e:
            tprint_error(f"❌ Failed to analyze trade timing: {e}")
            return {}

    async def _analyze_position_sizes(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Analyze position sizing patterns and performance."""
        tprint(f"PerformanceReporter._analyze_position_sizes: Called")
        try:
            position_sizes = [t.position_size for t in trades if t.position_size > 0]

            if not position_sizes:
                return {'message': 'No position size data available'}

            # Size distribution
            size_percentiles = {
                'min': min(position_sizes),
                'max': max(position_sizes),
                'mean': np.mean(position_sizes),
                'median': np.median(position_sizes),
                'p25': np.percentile(position_sizes, 25),
                'p75': np.percentile(position_sizes, 75),
                'std': np.std(position_sizes)
            }

            # Performance by size quartiles
            q1 = np.percentile(position_sizes, 25)
            q2 = np.percentile(position_sizes, 50)
            q3 = np.percentile(position_sizes, 75)

            size_performance = {}
            for quartile, (min_size, max_size) in enumerate([
                (0, q1), (q1, q2), (q2, q3), (q3, float('inf'))
            ], 1):
                quartile_trades = [
                    t for t in trades
                    if min_size <= t.position_size < max_size
                ]

                if quartile_trades:
                    quartile_pnl = [t.pnl_absolute for t in quartile_trades if t.pnl_absolute is not None]
                    size_performance[f'quartile_{quartile}'] = {
                        'trade_count': len(quartile_trades),
                        'size_range': [min_size, max_size],
                        'total_pnl': sum(quartile_pnl) if quartile_pnl else 0.0,
                        'avg_pnl': np.mean(quartile_pnl) if quartile_pnl else 0.0,
                        'win_rate': len([p for p in quartile_pnl if p > 0]) / len(quartile_pnl) if quartile_pnl else 0.0
                    }

            return {
                'size_distribution': size_percentiles,
                'performance_by_size_quartile': size_performance,
                'optimal_size_analysis': await self._find_optimal_position_sizes(trades)
            }

        except Exception as e:
            tprint_error(f"❌ Failed to analyze position sizes: {e}")
            return {}

    async def _find_optimal_position_sizes(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Find optimal position sizes based on historical performance."""
        tprint(f"PerformanceReporter._find_optimal_position_sizes: Called")
        try:
            # Group trades by confidence levels and analyze optimal sizes
            confidence_ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

            optimal_sizes = {}
            for i, (min_conf, max_conf) in enumerate(confidence_ranges):
                range_trades = [
                    t for t in trades
                    if min_conf <= t.signal_confidence < max_conf and t.pnl_absolute is not None
                ]

                if range_trades:
                    # Find size that maximizes risk-adjusted returns
                    best_size = 0.0
                    best_performance = -float('inf')

                    for trade in range_trades:
                        if trade.position_size > 0 and trade.portfolio_risk > 0:
                            risk_adjusted_return = trade.pnl_percentage / trade.portfolio_risk
                            if risk_adjusted_return > best_performance:
                                best_performance = risk_adjusted_return
                                best_size = trade.position_size

                    optimal_sizes[f'confidence_{min_conf}_{max_conf}'] = {
                        'optimal_size': best_size,
                        'performance': best_performance,
                        'trade_count': len(range_trades)
                    }

            return optimal_sizes

        except Exception as e:
            tprint_error(f"❌ Failed to find optimal position sizes: {e}")
            return {}

    async def _export_report(self, report: Dict[str, Any], report_name: str):
"""Export comprehensive report to files."""
        tprint(f"PerformanceReporter._export_report: Called")
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Export JSON report
            json_file = self.report_directory / f"{report_name}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            tprint_success(f"✅ Exported JSON report to {json_file}")

            # Export CSV summary
            csv_file = self.report_directory / f"{report_name}_summary_{timestamp}.csv"
            summary_data = self._extract_summary_for_csv(report)
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(csv_file, index=False)

            tprint_success(f"✅ Exported CSV summary to {csv_file}")

            # Export HTML report if enabled
            if self.enable_html_reports:
                await self._generate_html_report(report, report_name, timestamp)

        except Exception as e:
            tprint_error(f"❌ Failed to export report: {e}")

    def _extract_summary_for_csv(self, report: Dict[str, Any]) -> Dict[str, Any]:
"""Extract key metrics for CSV export."""
        tprint(f"PerformanceReporter._extract_summary_for_csv: Called")
        try:
            summary = {}

            # Executive summary metrics
            if 'executive_summary' in report:
                exec_summary = report['executive_summary']
                if 'performance_overview' in exec_summary:
                    for key, value in exec_summary['performance_overview'].items():
                        summary[f'performance_{key}'] = value

                if 'key_metrics' in exec_summary:
                    for key, value in exec_summary['key_metrics'].items():
                        summary[f'metric_{key}'] = value

            # Risk analysis metrics
            if 'risk_analysis' in report and 'risk_metrics_summary' in report['risk_analysis']:
                for key, value in report['risk_analysis']['risk_metrics_summary'].items():
                    summary[f'risk_{key}'] = value

            # Model performance summary
            if 'model_performance' in report and 'individual_model_performance' in report['model_performance']:
                model_perf = report['model_performance']['individual_model_performance']
                for model_id, metrics in model_perf.items():
                    for metric_name, value in metrics.items():
                        summary[f'model_{model_id}_{metric_name}'] = value

            # Add timestamp
            summary['report_timestamp'] = datetime.now().isoformat()

            return summary

        except Exception as e:
            tprint_error(f"❌ Failed to extract summary for CSV: {e}")
            return {}

    async def _generate_html_report(self, report: Dict[str, Any], report_name: str, timestamp: str):
"""Generate HTML report with visualizations."""
        tprint(f"PerformanceReporter._generate_html_report: Called")
        try:
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Performance Report - {report_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .neutral {{ color: blue; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Trading Performance Report</h1>
                    <p><strong>Report:</strong> {report_name}</p>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """

            # Add executive summary
            if 'executive_summary' in report:
                html_content += self._generate_html_executive_summary(report['executive_summary'])

            # Add model performance
            if 'model_performance' in report:
                html_content += self._generate_html_model_performance(report['model_performance'])

            # Add regime analysis
            if 'regime_analysis' in report:
                html_content += self._generate_html_regime_analysis(report['regime_analysis'])

            html_content += """
            </body>
            </html>
            """

            # Save HTML file
            html_file = self.report_directory / f"{report_name}_{timestamp}.html"
            with open(html_file, 'w') as f:
                f.write(html_content)

            tprint_success(f"✅ Generated HTML report: {html_file}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate HTML report: {e}")

    def _generate_html_executive_summary(self, summary: Dict[str, Any]) -> str:
"""Generate HTML for executive summary."""
        tprint(f"PerformanceReporter._generate_html_executive_summary: Called")
        html = '<div class="section"><h2>Executive Summary</h2>'

        if 'performance_overview' in summary:
            perf = summary['performance_overview']
            html += '<div class="metrics">'

            for key, value in perf.items():
                css_class = 'positive' if 'pnl' in key.lower() and value > 0 else 'negative' if 'pnl' in key.lower() and value < 0 else 'neutral'
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                html += f'<div class="metric {css_class}"><strong>{key.replace("_", " ").title()}:</strong> {formatted_value}</div>'

            html += '</div>'

        html += '</div>'
        return html

    def _generate_html_model_performance(self, model_perf: Dict[str, Any]) -> str:
"""Generate HTML for model performance."""
        tprint(f"PerformanceReporter._generate_html_model_performance: Called")
        html = '<div class="section"><h2>Model Performance Analysis</h2>'

        if 'individual_model_performance' in model_perf:
            html += '<table><tr><th>Model ID</th><th>Usage Count</th><th>Total PnL</th><th>Accuracy</th><th>Avg Confidence</th></tr>'

            for model_id, metrics in model_perf['individual_model_performance'].items():
                html += f"""
                <tr>
                    <td>{model_id}</td>
                    <td>{metrics.get('usage_count', 0)}</td>
                    <td class="{'positive' if metrics.get('total_pnl', 0) > 0 else 'negative'}">{metrics.get('total_pnl', 0):.4f}</td>
                    <td>{metrics.get('accuracy', 0):.2%}</td>
                    <td>{metrics.get('avg_confidence', 0):.2%}</td>
                </tr>
                """

            html += '</table>'

        html += '</div>'
        return html

    def _generate_html_regime_analysis(self, regime_analysis: Dict[str, Any]) -> str:
"""Generate HTML for regime analysis."""
        tprint(f"PerformanceReporter._generate_html_regime_analysis: Called")
        html = '<div class="section"><h2>Regime Performance Analysis</h2>'

        if 'regime_performance' in regime_analysis:
            html += '<table><tr><th>Regime</th><th>Trade Count</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th></tr>'

            for regime, metrics in regime_analysis['regime_performance'].items():
                html += f"""
                <tr>
                    <td>{regime.replace('_', ' ').title()}</td>
                    <td>{metrics.get('trade_count', 0)}</td>
                    <td>{metrics.get('win_rate', 0):.2%}</td>
                    <td class="{'positive' if metrics.get('total_pnl', 0) > 0 else 'negative'}">{metrics.get('total_pnl', 0):.4f}</td>
                    <td class="{'positive' if metrics.get('avg_pnl_per_trade', 0) > 0 else 'negative'}">{metrics.get('avg_pnl_per_trade', 0):.4f}</td>
                </tr>
                """

            html += '</table>'

        html += '</div>'
        return html

    async def _compare_model_performance(self, model_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
"""Compare performance across different models."""
        tprint(f"PerformanceReporter._compare_model_performance: Called")
        try:
            if not model_performance:
                return {}
            
            # Rank models by various metrics
            models_by_pnl = sorted(model_performance.items(), key=lambda x: x[1].get('total_pnl', 0), reverse=True)
            models_by_accuracy = sorted(model_performance.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
            models_by_confidence = sorted(model_performance.items(), key=lambda x: x[1].get('avg_confidence', 0), reverse=True)
            
            return {
                'rankings': {
                    'by_pnl': [(m[0], m[1].get('total_pnl', 0)) for m in models_by_pnl],
                    'by_accuracy': [(m[0], m[1].get('accuracy', 0)) for m in models_by_accuracy],
                    'by_confidence': [(m[0], m[1].get('avg_confidence', 0)) for m in models_by_confidence]
                },
                'best_performer': models_by_pnl[0][0] if models_by_pnl else None,
                'most_accurate': models_by_accuracy[0][0] if models_by_accuracy else None,
                'most_confident': models_by_confidence[0][0] if models_by_confidence else None
            }
        except Exception as e:
            tprint_error(f"❌ Failed to compare model performance: {e}")
            return {}

    async def _analyze_ensemble_performance(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Analyze ensemble model performance."""
        tprint(f"PerformanceReporter._analyze_ensemble_performance: Called")
        try:
            if not trades:
                return {}
            
            # Analyze trades where multiple models were used
            ensemble_trades = [t for t in trades if len(t.models_used) > 1]
            single_model_trades = [t for t in trades if len(t.models_used) == 1]
            
            ensemble_pnl = [t.pnl_absolute for t in ensemble_trades if t.pnl_absolute is not None]
            single_pnl = [t.pnl_absolute for t in single_model_trades if t.pnl_absolute is not None]
            
            return {
                'ensemble_vs_single': {
                    'ensemble_trades': len(ensemble_trades),
                    'single_model_trades': len(single_model_trades),
                    'ensemble_avg_pnl': np.mean(ensemble_pnl) if ensemble_pnl else 0.0,
                    'single_avg_pnl': np.mean(single_pnl) if single_pnl else 0.0,
                    'ensemble_win_rate': len([p for p in ensemble_pnl if p > 0]) / len(ensemble_pnl) if ensemble_pnl else 0.0,
                    'single_win_rate': len([p for p in single_pnl if p > 0]) / len(single_pnl) if single_pnl else 0.0
                },
                'ensemble_effectiveness': 'better' if ensemble_pnl and single_pnl and np.mean(ensemble_pnl) > np.mean(single_pnl) else 'similar' if ensemble_pnl and single_pnl else 'unknown'
            }
        except Exception as e:
            tprint_error(f"❌ Failed to analyze ensemble performance: {e}")
            return {}

    async def _calculate_trade_quality_metrics(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Calculate trade quality metrics."""
        tprint(f"PerformanceReporter._calculate_trade_quality_metrics: Called")
        try:
            if not trades:
                return {}
            
            # Quality indicators
            high_confidence_trades = [t for t in trades if t.signal_confidence > 0.7]
            low_confidence_trades = [t for t in trades if t.signal_confidence < 0.5]
            
            high_conf_pnl = [t.pnl_absolute for t in high_confidence_trades if t.pnl_absolute is not None]
            low_conf_pnl = [t.pnl_absolute for t in low_confidence_trades if t.pnl_absolute is not None]
            
            return {
                'high_confidence_metrics': {
                    'count': len(high_confidence_trades),
                    'avg_pnl': np.mean(high_conf_pnl) if high_conf_pnl else 0.0,
                    'win_rate': len([p for p in high_conf_pnl if p > 0]) / len(high_conf_pnl) if high_conf_pnl else 0.0
                },
                'low_confidence_metrics': {
                    'count': len(low_confidence_trades),
                    'avg_pnl': np.mean(low_conf_pnl) if low_conf_pnl else 0.0,
                    'win_rate': len([p for p in low_conf_pnl if p > 0]) / len(low_conf_pnl) if low_conf_pnl else 0.0
                },
                'quality_score': len(high_confidence_trades) / len(trades) if trades else 0.0
            }
        except Exception as e:
            tprint_error(f"❌ Failed to calculate trade quality metrics: {e}")
            return {}

    async def _analyze_model_agreement(self, trades: List[DetailedTradeMetrics]) -> Dict[str, Any]:
"""Analyze model agreement across trades."""
        tprint(f"PerformanceReporter._analyze_model_agreement: Called")
        try:
            if not trades:
                return {}
            
            agreement_scores = []
            for trade in trades:
                if len(trade.model_predictions) > 1:
                    predictions = list(trade.model_predictions.values())
                    # Agreement is inverse of variance
                    variance = np.var(predictions)
                    agreement = 1.0 - min(variance, 1.0)
                    agreement_scores.append(agreement)
            
            return {
                'avg_agreement': np.mean(agreement_scores) if agreement_scores else 0.0,
                'trades_with_multiple_models': len([t for t in trades if len(t.models_used) > 1]),
                'agreement_distribution': {
                    'high': len([s for s in agreement_scores if s > 0.8]),
                    'medium': len([s for s in agreement_scores if 0.5 <= s <= 0.8]),
                    'low': len([s for s in agreement_scores if s < 0.5])
                }
            }
        except Exception as e:
            tprint_error(f"❌ Failed to analyze model agreement: {e}")
            return {}

    async def _identify_drawdown_periods(self, cumulative_pnl: np.ndarray) -> List[Dict[str, Any]]:
"""Identify drawdown periods in cumulative PnL."""
        tprint(f"PerformanceReporter._identify_drawdown_periods: Called")
        try:
            if len(cumulative_pnl) < 2:
                return []
            
            periods = []
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = peak - cumulative_pnl
            
            # Find drawdown periods (where drawdown > 0)
            in_drawdown = False
            start_idx = None
            
            for i, dd in enumerate(drawdown):
                if dd > 0 and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    start_idx = i
                elif dd == 0 and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    if start_idx is not None:
                        periods.append({
                            'start_index': start_idx,
                            'end_index': i - 1,
                            'duration': i - start_idx,
                            'max_drawdown': max(drawdown[start_idx:i])
                        })
            
            # Handle case where drawdown continues to end
            if in_drawdown and start_idx is not None:
                periods.append({
                    'start_index': start_idx,
                    'end_index': len(drawdown) - 1,
                    'duration': len(drawdown) - start_idx,
                    'max_drawdown': max(drawdown[start_idx:])
                })
            
            return periods
        except Exception as e:
            tprint_error(f"❌ Failed to identify drawdown periods: {e}")
            return []

# Global instance
performance_reporter = PerformanceReporter()

# Convenience functions
async def generate_trading_report(
    trades: List[DetailedTradeMetrics],
    session_metrics: Optional[TradingSessionMetrics] = None,
    report_name: str = "trading_performance"
) -> Dict[str, Any]:
"""Generate comprehensive trading performance report."""
    tprint(f"PerformanceReporter.generate_trading_report: Called")
    return await performance_reporter.generate_comprehensive_report(trades, session_metrics, report_name)
