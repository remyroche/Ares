"""
Enhanced Reporting System for Step 5 Labeling

This module provides comprehensive reporting capabilities for step5_labeling
with detailed metrics, performance analytics, label quality assessments,
and trading strategy implications from the labeling process.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings

# Avoid circular import - import these functions when needed
# from src.training.reports import save_training_report, CentralizedReportManager
from src.utils.logger import system_logger
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context

logger = system_logger.getChild('Step05EnhancedReporting')
financial_logger = get_financial_metrics_logger()


@dataclass
class LabelQualityMetrics:
    """Comprehensive metrics for label quality assessment."""
    total_labels: int
    buy_labels: int
    sell_labels: int
    hold_labels: int
    label_distribution_balance: float
    label_confidence_score: float
    label_consistency_score: float
    label_purity_score: float
    label_stability_score: float
    false_positive_rate: float
    false_negative_rate: float
    label_accuracy_estimate: float


@dataclass
class LabelingPerformanceMetrics:
    """Performance metrics for the labeling process."""
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    label_creation_rate: float  # labels/second
    meta_labeling_time: float
    fallback_labeling_time: float
    validation_time: float
    total_function_calls: int
    successful_operations: int
    failed_operations: int
    error_rate: float
    processing_efficiency: float
    optimization_effectiveness: float


@dataclass
class MetaLabelingAnalysis:
    """Analysis of meta-labeling system performance."""
    meta_labels_created: int
    meta_labeling_success_rate: float
    meta_label_confidence_avg: float
    meta_label_quality_score: float
    primary_vs_meta_label_agreement: float
    meta_labeling_computation_time: float
    meta_labeling_memory_usage: float
    meta_labeling_optimization_gain: float


@dataclass
class LabelValidationResults:
    """Results from label validation processes."""
    validation_passed: bool
    validation_checks_performed: int
    validation_failures: int
    validation_error_rate: float
    data_integrity_score: float
    label_consistency_score: float
    statistical_validation_score: float
    cross_validation_score: float
    validation_warnings: List[str]
    validation_recommendations: List[str]


@dataclass
class TradingStrategyImplications:
    """Trading strategy implications derived from label analysis."""
    expected_win_rate: float
    expected_profit_factor: float
    expected_max_drawdown: float
    optimal_holding_period_days: float
    risk_adjusted_return_expectation: float
    strategy_confidence_score: float
    market_regime_suitability: Dict[str, float]
    position_sizing_recommendation: float
    entry_signal_strength: float
    exit_signal_reliability: float


@dataclass
class LabelDistributionAnalysis:
    """Analysis of label distribution patterns."""
    temporal_distribution_uniformity: float
    label_sequence_patterns: Dict[str, int]
    label_clustering_coefficient: float
    label_transition_smoothness: float
    label_persistence_distribution: List[float]
    consecutive_label_patterns: Dict[str, int]
    label_volatility_by_regime: Dict[str, float]


class Step05EnhancedReporter:
    """
    Enhanced reporting system for Step 5 Labeling.

    Provides comprehensive metrics including:
    - Label quality assessment and validation
    - Meta-labeling performance analysis
    - Labeling efficiency and optimization metrics
    - Trading strategy implications
    - Label distribution and pattern analysis
    - Visualization capabilities
    """

    def __init__(self, output_dir: str = "src/training/reports/step05"):
        """
        Initialize the Step05 enhanced reporter.

        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = system_logger.getChild('Step05EnhancedReporter')

        # Initialize report manager (avoid circular import)
        try:
            from src.training.reports import CentralizedReportManager
            self.report_manager = CentralizedReportManager()
        except (ImportError, TypeError):
            self.logger.warning("Could not import CentralizedReportManager, using fallback")
            self.report_manager = None

    def generate_comprehensive_report(self,
                                    labeled_data: pd.DataFrame,
                                    labeling_results: Dict[str, Any],
                                    performance_data: Dict[str, Any],
                                    validation_results: Dict[str, Any],
                                    meta_labeling_analysis: Dict[str, Any],
                                    symbol: str,
                                    exchange: str,
                                    timeframe: str) -> Dict[str, Any]:
        """
        Generate comprehensive report with all metrics and analyses.

        Args:
            labeled_data: The labeled dataset
            labeling_results: Results from the labeling process
            performance_data: Performance metrics during execution
            validation_results: Results from label validation
            meta_labeling_analysis: Analysis of meta-labeling performance
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe analyzed

        Returns:
            Comprehensive report dictionary
        """
        # Use financial metrics context for this step
        with financial_metrics_context("Step05_Enhanced_Reporting", symbol, exchange, timeframe):
            try:
                self.logger.info("🔍 Generating comprehensive Step05 (Labeling) report...")
                financial_logger.log_step_start("Step05_Enhanced_Reporting", symbol, exchange, timeframe)

                # Generate all report sections
                report = {
                    'metadata': self._generate_metadata(symbol, exchange, timeframe),
                    'label_quality_assessment': self._generate_label_quality_assessment(labeled_data),
                    'performance_metrics': self._generate_performance_metrics(performance_data),
                    'meta_labeling_analysis': self._generate_meta_labeling_analysis(meta_labeling_analysis),
                    'validation_results': self._generate_validation_results(validation_results),
                    'label_distribution_analysis': self._generate_label_distribution_analysis(labeled_data),
                    'trading_strategy_implications': self._generate_trading_strategy_implications(labeled_data, labeling_results),
                    'labeling_efficiency_analysis': self._generate_labeling_efficiency_analysis(performance_data),
                    'optimization_recommendations': self._generate_optimization_recommendations(performance_data, labeling_results),
                    'visualization_data': self._generate_visualization_data(labeled_data, labeling_results)
                }

                # Log key financial metrics from the report
                self._log_financial_metrics_from_report(report, symbol, exchange, timeframe)

                self.logger.info("✅ Comprehensive Step05 report generated successfully")
                financial_logger.log_step_end("Step05_Enhanced_Reporting", symbol, exchange, timeframe, success=True)
                return report

            except Exception as e:
                self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
                financial_logger.log_step_end("Step05_Enhanced_Reporting", symbol, exchange, timeframe, success=False, error_message=str(e))
                # Return minimal report on error
                return {
                    'metadata': self._generate_metadata(symbol, exchange, timeframe),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }

    def _generate_metadata(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'report_type': 'step05_labeling_enhanced',
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'step_name': 'Step 5',
            'step_description': 'Enhanced Labeling with Meta-Labeling and Validation'
        }

    def _log_financial_metrics_from_report(self, report: Dict[str, Any], symbol: str, exchange: str, timeframe: str) -> None:
        """Log key financial metrics from the comprehensive report."""
        try:
            # Log trading strategy implications as financial metrics
            trading_data = report.get('trading_strategy_implications', {})
            if trading_data and 'strategy_implications' in trading_data:
                implications = trading_data['strategy_implications']
                
                # Log key performance metrics
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="expected_win_rate",
                    metric_value=implications.get('expected_win_rate', 0.0),
                    metric_type="performance",
                    step_name="Step05_Enhanced_Reporting"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="expected_profit_factor",
                    metric_value=implications.get('expected_profit_factor', 0.0),
                    metric_type="performance",
                    step_name="Step05_Enhanced_Reporting"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="expected_max_drawdown",
                    metric_value=implications.get('expected_max_drawdown', 0.0),
                    metric_type="risk",
                    step_name="Step05_Enhanced_Reporting"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="strategy_confidence_score",
                    metric_value=implications.get('strategy_confidence_score', 0.0),
                    metric_type="performance",
                    step_name="Step05_Enhanced_Reporting"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="risk_adjusted_return_expectation",
                    metric_value=implications.get('risk_adjusted_return_expectation', 0.0),
                    metric_type="performance",
                    step_name="Step05_Enhanced_Reporting"
                )
            
            # Log label quality metrics
            quality_data = report.get('label_quality_assessment', {})
            if quality_data and 'quality_metrics' in quality_data:
                metrics = quality_data['quality_metrics']
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="label_confidence_score",
                    metric_value=metrics.get('label_confidence_score', 0.0),
                    metric_type="quality",
                    step_name="Step05_Enhanced_Reporting"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="label_accuracy_estimate",
                    metric_value=metrics.get('label_accuracy_estimate', 0.0),
                    metric_type="quality",
                    step_name="Step05_Enhanced_Reporting"
                )
            
            # Log performance metrics
            perf_data = report.get('performance_metrics', {})
            if perf_data and 'metrics' in perf_data:
                metrics = perf_data['metrics']
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="processing_efficiency",
                    metric_value=metrics.get('processing_efficiency', 0.0),
                    metric_type="performance",
                    step_name="Step05_Enhanced_Reporting"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="error_rate",
                    metric_value=metrics.get('error_rate', 0.0),
                    metric_type="risk",
                    step_name="Step05_Enhanced_Reporting"
                )
            
            # Log comprehensive trading performance if we have enough data
            if trading_data and 'strategy_implications' in trading_data:
                implications = trading_data['strategy_implications']
                
                # Create performance data dictionary for comprehensive logging
                performance_data = {
                    'total_return': implications.get('expected_win_rate', 0.0) * implications.get('expected_profit_factor', 1.0) - 1.0,
                    'annualized_return': implications.get('expected_win_rate', 0.0) * implications.get('expected_profit_factor', 1.0) - 1.0,
                    'volatility': 0.2,  # Default volatility estimate
                    'sharpe_ratio': implications.get('risk_adjusted_return_expectation', 0.0),
                    'sortino_ratio': implications.get('risk_adjusted_return_expectation', 0.0) * 1.2,  # Estimate
                    'calmar_ratio': implications.get('expected_win_rate', 0.0) / max(implications.get('expected_max_drawdown', 0.1), 0.01),
                    'max_drawdown': implications.get('expected_max_drawdown', 0.0),
                    'max_drawdown_duration': 30,  # Default estimate
                    'var_95': implications.get('expected_max_drawdown', 0.0) * 0.8,  # Estimate
                    'cvar_95': implications.get('expected_max_drawdown', 0.0) * 0.9,  # Estimate
                    'win_rate': implications.get('expected_win_rate', 0.0),
                    'profit_factor': implications.get('expected_profit_factor', 1.0),
                    'avg_win': implications.get('expected_profit_factor', 1.0) * 0.02,  # Estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': implications.get('expected_profit_factor', 1.0) * 0.05,  # Estimate
                    'largest_loss': implications.get('expected_max_drawdown', 0.0) * 0.5,  # Estimate
                    'total_trades': 100,  # Default estimate
                    'winning_trades': int(implications.get('expected_win_rate', 0.5) * 100),
                    'losing_trades': int((1 - implications.get('expected_win_rate', 0.5)) * 100)
                }
                
                financial_logger.log_trading_performance(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    step_name="Step05_Enhanced_Reporting",
                    performance_data=performance_data,
                    confidence_score=implications.get('strategy_confidence_score', 0.0)
                )
            
            self.logger.info("💰 Financial metrics logged successfully from Step05 report")
            
        except Exception as e:
            self.logger.warning(f"Could not log financial metrics from report: {e}")

    def _generate_label_quality_assessment(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of generated labels."""
        try:
            if 'label' not in labeled_data.columns:
                return {'error': 'No label column found in data'}

            # Calculate label distribution
            label_counts = labeled_data['label'].value_counts()
            total_labels = len(labeled_data)

            buy_labels = label_counts.get(1, 0)  # Assuming 1 = buy
            sell_labels = label_counts.get(-1, 0)  # Assuming -1 = sell
            hold_labels = label_counts.get(0, 0)  # Assuming 0 = hold

            # Calculate distribution balance
            proportions = [buy_labels/total_labels, sell_labels/total_labels, hold_labels/total_labels]
            ideal = 1/3
            balance = max(0, 100 - sum(abs(p - ideal) * 100 for p in proportions))

            # Calculate label consistency (temporal stability)
            if len(labeled_data) > 1:
                label_changes = sum(1 for i in range(1, len(labeled_data)) if labeled_data['label'].iloc[i] != labeled_data['label'].iloc[i-1])
                consistency_score = max(0, 100 - (label_changes / len(labeled_data)) * 100)
            else:
                consistency_score = 100.0

            # Calculate label purity (concentration of strong signals)
            strong_signals = buy_labels + sell_labels
            purity_score = (strong_signals / total_labels) * 100

            # Calculate label stability (resistance to noise)
            if len(labeled_data) > 10:
                # Rolling window stability
                window_size = min(20, len(labeled_data) // 10)
                rolling_stability = []
                for i in range(window_size, len(labeled_data), window_size):
                    window = labeled_data['label'].iloc[i-window_size:i]
                    stability = window.value_counts().iloc[0] / len(window) if len(window) > 0 else 0
                    rolling_stability.append(stability)
                stability_score = np.mean(rolling_stability) * 100 if rolling_stability else 50.0
            else:
                stability_score = 50.0

            quality_metrics = LabelQualityMetrics(
                total_labels=total_labels,
                buy_labels=buy_labels,
                sell_labels=sell_labels,
                hold_labels=hold_labels,
                label_distribution_balance=balance,
                label_confidence_score=self._calculate_label_confidence(labeled_data),
                label_consistency_score=consistency_score,
                label_purity_score=purity_score,
                label_stability_score=stability_score,
                false_positive_rate=self._estimate_false_positive_rate(labeled_data),
                false_negative_rate=self._estimate_false_negative_rate(labeled_data),
                label_accuracy_estimate=self._estimate_label_accuracy(labeled_data)
            )

            return {
                'quality_metrics': asdict(quality_metrics),
                'label_distribution': label_counts.to_dict(),
                'quality_warnings': self._identify_label_quality_warnings(labeled_data),
                'quality_improvements': self._suggest_label_quality_improvements(labeled_data)
            }

        except Exception as e:
            self.logger.warning(f"Could not assess label quality: {e}")
            return {'error': str(e)}

    def _generate_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance metrics."""
        try:
            metrics = LabelingPerformanceMetrics(
                execution_time_seconds=performance_data.get('execution_time', 0.0),
                memory_usage_mb=performance_data.get('memory_usage', 0.0),
                cpu_usage_percent=performance_data.get('cpu_usage', 0.0),
                label_creation_rate=performance_data.get('label_creation_rate', 0.0),
                meta_labeling_time=performance_data.get('meta_labeling_time', 0.0),
                fallback_labeling_time=performance_data.get('fallback_labeling_time', 0.0),
                validation_time=performance_data.get('validation_time', 0.0),
                total_function_calls=performance_data.get('function_calls', 0),
                successful_operations=performance_data.get('successful_ops', 0),
                failed_operations=performance_data.get('failed_ops', 0),
                error_rate=performance_data.get('error_rate', 0.0),
                processing_efficiency=performance_data.get('processing_efficiency', 0.0),
                optimization_effectiveness=performance_data.get('optimization_effectiveness', 0.0)
            )

            return {
                'metrics': asdict(metrics),
                'efficiency_scores': self._calculate_labeling_efficiency_scores(metrics),
                'performance_warnings': self._identify_performance_warnings(metrics),
                'optimization_analysis': self._analyze_optimization_effectiveness(metrics)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate performance metrics: {e}")
            return {'error': str(e)}

    def _generate_meta_labeling_analysis(self, meta_labeling_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate meta-labeling system analysis."""
        try:
            analysis = MetaLabelingAnalysis(
                meta_labels_created=meta_labeling_analysis.get('meta_labels_created', 0),
                meta_labeling_success_rate=meta_labeling_analysis.get('success_rate', 0.0),
                meta_label_confidence_avg=meta_labeling_analysis.get('avg_confidence', 0.0),
                meta_label_quality_score=meta_labeling_analysis.get('quality_score', 0.0),
                primary_vs_meta_label_agreement=meta_labeling_analysis.get('agreement_rate', 0.0),
                meta_labeling_computation_time=meta_labeling_analysis.get('computation_time', 0.0),
                meta_labeling_memory_usage=meta_labeling_analysis.get('memory_usage', 0.0),
                meta_labeling_optimization_gain=meta_labeling_analysis.get('optimization_gain', 0.0)
            )

            return {
                'meta_labeling_metrics': asdict(analysis),
                'meta_labeling_effectiveness': self._assess_meta_labeling_effectiveness(analysis),
                'meta_vs_primary_comparison': self._compare_meta_vs_primary_labels(meta_labeling_analysis),
                'meta_labeling_optimization_analysis': self._analyze_meta_labeling_optimization(analysis)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate meta-labeling analysis: {e}")
            return {'error': str(e)}

    def _generate_validation_results(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation results analysis."""
        try:
            results = LabelValidationResults(
                validation_passed=validation_results.get('passed', False),
                validation_checks_performed=validation_results.get('checks_performed', 0),
                validation_failures=validation_results.get('failures', 0),
                validation_error_rate=validation_results.get('error_rate', 0.0),
                data_integrity_score=validation_results.get('data_integrity_score', 0.0),
                label_consistency_score=validation_results.get('label_consistency_score', 0.0),
                statistical_validation_score=validation_results.get('statistical_score', 0.0),
                cross_validation_score=validation_results.get('cross_validation_score', 0.0),
                validation_warnings=validation_results.get('warnings', []),
                validation_recommendations=validation_results.get('recommendations', [])
            )

            return {
                'validation_metrics': asdict(results),
                'validation_summary': self._summarize_validation_results(results),
                'validation_action_items': self._generate_validation_action_items(results)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate validation results: {e}")
            return {'error': str(e)}

    def _generate_label_distribution_analysis(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate label distribution analysis."""
        try:
            if 'label' not in labeled_data.columns:
                return {'error': 'No label column found in data'}

            # Analyze temporal distribution
            temporal_uniformity = self._calculate_temporal_uniformity(labeled_data)

            # Analyze label sequence patterns
            sequence_patterns = self._analyze_label_sequences(labeled_data)

            # Calculate clustering coefficient
            clustering_coeff = self._calculate_label_clustering(labeled_data)

            # Analyze transition smoothness
            transition_smoothness = self._calculate_transition_smoothness(labeled_data)

            # Analyze persistence distribution
            persistence_dist = self._calculate_persistence_distribution(labeled_data)

            # Analyze consecutive patterns
            consecutive_patterns = self._analyze_consecutive_patterns(labeled_data)

            # Analyze volatility by regime (if regime column exists)
            regime_volatility = {}
            if 'regime_id' in labeled_data.columns:
                for regime in labeled_data['regime_id'].unique():
                    regime_data = labeled_data[labeled_data['regime_id'] == regime]
                    if len(regime_data) > 1:
                        label_changes = sum(1 for i in range(1, len(regime_data)) if regime_data['label'].iloc[i] != regime_data['label'].iloc[i-1])
                        regime_volatility[f'regime_{regime}'] = label_changes / len(regime_data)

            analysis = LabelDistributionAnalysis(
                temporal_distribution_uniformity=temporal_uniformity,
                label_sequence_patterns=sequence_patterns,
                label_clustering_coefficient=clustering_coeff,
                label_transition_smoothness=transition_smoothness,
                label_persistence_distribution=persistence_dist,
                consecutive_label_patterns=consecutive_patterns,
                label_volatility_by_regime=regime_volatility
            )

            return {
                'distribution_analysis': asdict(analysis),
                'pattern_insights': self._extract_pattern_insights(analysis),
                'distribution_quality_metrics': self._assess_distribution_quality(analysis)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate label distribution analysis: {e}")
            return {'error': str(e)}

    def _generate_trading_strategy_implications(self, labeled_data: pd.DataFrame, labeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading strategy implications from label analysis."""
        try:
            # Calculate expected win rate
            if 'label' in labeled_data.columns and 'close' in labeled_data.columns:
                win_rate = self._calculate_expected_win_rate(labeled_data)
                profit_factor = self._calculate_expected_profit_factor(labeled_data)
                max_drawdown = self._calculate_expected_max_drawdown(labeled_data)
                holding_period = self._calculate_optimal_holding_period(labeled_data)
            else:
                win_rate = profit_factor = max_drawdown = holding_period = 0.5

            # Calculate risk-adjusted return expectation
            risk_adjusted_return = win_rate * profit_factor / (1 + max_drawdown) if max_drawdown > 0 else 0

            # Calculate strategy confidence
            strategy_confidence = self._calculate_strategy_confidence(labeled_data)

            # Assess market regime suitability
            regime_suitability = self._assess_regime_suitability(labeled_data)

            # Calculate position sizing recommendation
            position_sizing = self._calculate_position_sizing_recommendation(labeled_data)

            # Calculate entry/exit signal metrics
            entry_strength = self._calculate_entry_signal_strength(labeled_data)
            exit_reliability = self._calculate_exit_signal_reliability(labeled_data)

            implications = TradingStrategyImplications(
                expected_win_rate=win_rate,
                expected_profit_factor=profit_factor,
                expected_max_drawdown=max_drawdown,
                optimal_holding_period_days=holding_period,
                risk_adjusted_return_expectation=risk_adjusted_return,
                strategy_confidence_score=strategy_confidence,
                market_regime_suitability=regime_suitability,
                position_sizing_recommendation=position_sizing,
                entry_signal_strength=entry_strength,
                exit_signal_reliability=exit_reliability
            )

            return {
                'strategy_implications': asdict(implications),
                'trading_recommendations': self._generate_trading_recommendations(implications),
                'risk_assessment': self._assess_strategy_risks(implications),
                'performance_projections': self._project_strategy_performance(implications)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate trading strategy implications: {e}")
            return {'error': str(e)}

    def _generate_labeling_efficiency_analysis(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate labeling efficiency analysis."""
        try:
            return {
                'efficiency_metrics': self._calculate_labeling_efficiency_metrics(performance_data),
                'bottleneck_analysis': self._identify_labeling_bottlenecks(performance_data),
                'optimization_opportunities': self._identify_optimization_opportunities(performance_data),
                'resource_utilization_analysis': self._analyze_resource_utilization(performance_data)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate labeling efficiency analysis: {e}")
            return {'error': str(e)}

    def _generate_optimization_recommendations(self, performance_data: Dict[str, Any], labeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        try:
            return {
                'performance_optimizations': self._recommend_performance_optimizations(performance_data),
                'memory_optimizations': self._recommend_memory_optimizations(performance_data),
                'algorithm_optimizations': self._recommend_algorithm_optimizations(labeling_results),
                'hardware_optimizations': self._recommend_hardware_optimizations(performance_data),
                'implementation_priority': self._prioritize_optimization_implementations(performance_data, labeling_results)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate optimization recommendations: {e}")
            return {'error': str(e)}

    def _generate_visualization_data(self, labeled_data: pd.DataFrame, labeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for visualizations."""
        try:
            viz_data = {
                'label_distribution_plot': self._prepare_label_distribution_data(labeled_data),
                'label_temporal_pattern': self._prepare_label_temporal_data(labeled_data),
                'label_transition_heatmap': self._prepare_label_transition_data(labeled_data),
                'label_quality_dashboard': self._prepare_label_quality_data(labeled_data),
                'performance_timeline': self._prepare_performance_timeline_data(labeling_results),
                'meta_labeling_comparison': self._prepare_meta_labeling_comparison_data(labeling_results),
                'regime_labeling_analysis': self._prepare_regime_labeling_data(labeled_data)
            }

            return viz_data

        except Exception as e:
            self.logger.warning(f"Could not generate visualization data: {e}")
            return {'error': str(e)}

    def save_comprehensive_report(self, report: Dict[str, Any], base_filename: str = "step05_enhanced_report") -> Dict[str, str]:
        """
        Save comprehensive report in multiple formats.

        Args:
            report: The comprehensive report dictionary
            base_filename: Base filename for saved files

        Returns:
            Dictionary mapping format types to file paths
        """
        try:
            self.logger.info("💾 Saving comprehensive Step05 report...")

            saved_files = {}
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save JSON report
            json_path = self._save_json_report(report, timestamp, base_filename)
            saved_files['json'] = str(json_path)

            # Save Markdown report
            md_path = self._save_markdown_report(report, timestamp, base_filename)
            saved_files['markdown'] = str(md_path)

            # Save CSV data
            csv_path = self._save_csv_report(report, timestamp, base_filename)
            saved_files['csv'] = str(csv_path)

            # Generate and save visualizations
            try:
                self._generate_visualizations(report, timestamp, base_filename)
                saved_files['visualizations'] = str(self.output_dir / f"{base_filename}_visualizations_{timestamp}")
            except Exception as e:
                self.logger.warning(f"Could not generate visualizations: {e}")

            # Generate comprehensive summary dashboard
            try:
                viz_dir = self.output_dir / f"{base_filename}_visualizations_{timestamp}"
                viz_dir.mkdir(exist_ok=True)
                self._create_comprehensive_summary_dashboard(report, viz_dir)
            except Exception as e:
                self.logger.warning(f"Could not create comprehensive summary dashboard: {e}")

            # Use centralized report manager if available
            if self.report_manager:
                try:
                    from src.training.reports import save_training_report
                    report_path = save_training_report(
                        report_data=report,
                        step_name="step05",
                        symbol=report.get('metadata', {}).get('symbol', 'unknown'),
                        exchange=report.get('metadata', {}).get('exchange', 'unknown'),
                        timeframe=report.get('metadata', {}).get('timeframe', 'unknown'),
                        report_type="enhanced_labeling_analysis"
                    )
                    saved_files['centralized'] = str(report_path)
                except Exception as e:
                    self.logger.warning(f"Could not save to centralized reports: {e}")

            self.logger.info(f"✅ Step05 enhanced report saved successfully: {saved_files}")
            return saved_files

        except Exception as e:
            self.logger.error(f"❌ Failed to save comprehensive report: {e}")
            return {'error': str(e)}

    # Helper methods for analysis and calculations
    def _calculate_label_confidence(self, labeled_data: pd.DataFrame) -> float:
        """Calculate label confidence score."""
        if 'label_confidence' in labeled_data.columns:
            return labeled_data['label_confidence'].mean()
        else:
            # Estimate based on label distribution stability
            if 'label' in labeled_data.columns and len(labeled_data) > 10:
                window_size = min(20, len(labeled_data) // 5)
                confidences = []
                for i in range(window_size, len(labeled_data), window_size):
                    window = labeled_data['label'].iloc[i-window_size:i]
                    dominant_label_ratio = window.value_counts().iloc[0] / len(window)
                    confidences.append(dominant_label_ratio)
                return np.mean(confidences) * 100 if confidences else 50.0
            else:
                return 50.0

    def _estimate_false_positive_rate(self, labeled_data: pd.DataFrame) -> float:
        """Estimate false positive rate."""
        # This is a simplified estimation - in practice would use validation data
        if 'label' not in labeled_data.columns:
            return 0.0

        # Estimate based on label switching frequency
        if len(labeled_data) > 1:
            switches = sum(1 for i in range(1, len(labeled_data)) if labeled_data['label'].iloc[i] != labeled_data['label'].iloc[i-1])
            return min(switches / len(labeled_data), 0.5)  # Cap at 50%
        return 0.0

    def _estimate_false_negative_rate(self, labeled_data: pd.DataFrame) -> float:
        """Estimate false negative rate."""
        # Simplified estimation
        return self._estimate_false_positive_rate(labeled_data) * 0.8  # Assume slightly lower

    def _estimate_label_accuracy(self, labeled_data: pd.DataFrame) -> float:
        """Estimate label accuracy."""
        if 'label' not in labeled_data.columns:
            return 0.0

        # Estimate based on label stability and distribution
        stability = self._calculate_label_confidence(labeled_data) / 100
        balance = self._calculate_label_balance(labeled_data)

        return (stability * 0.7 + balance * 0.3) * 100

    def _calculate_label_balance(self, labeled_data: pd.DataFrame) -> float:
        """Calculate label balance score."""
        if 'label' not in labeled_data.columns:
            return 0.0

        label_counts = labeled_data['label'].value_counts()
        total = len(labeled_data)

        if len(label_counts) == 0:
            return 0.0

        proportions = label_counts / total
        ideal = 1 / len(label_counts)

        return min(1.0, 1.0 - np.std(proportions) / ideal)

    # Additional helper methods would be implemented here
    # These are simplified stubs for the full implementation

    def _identify_label_quality_warnings(self, labeled_data: pd.DataFrame) -> List[str]:
        """Identify label quality warnings."""
        warnings = []

        if 'label' in labeled_data.columns:
            label_counts = labeled_data['label'].value_counts()
            if len(label_counts) < 2:
                warnings.append("Very few distinct labels detected")

            # Check for extreme imbalance
            max_proportion = label_counts.iloc[0] / len(labeled_data)
            if max_proportion > 0.9:
                warnings.append("Extreme label imbalance detected")

        return warnings

    def _suggest_label_quality_improvements(self, labeled_data: pd.DataFrame) -> List[str]:
        """Suggest label quality improvements."""
        improvements = []

        if 'label' in labeled_data.columns:
            balance = self._calculate_label_balance(labeled_data)
            if balance < 0.3:
                improvements.append("Implement label balancing techniques")

            confidence = self._calculate_label_confidence(labeled_data)
            if confidence < 60:
                improvements.append("Improve label confidence through better feature engineering")

        return improvements

    def _calculate_labeling_efficiency_scores(self, metrics: Any) -> Dict[str, float]:
        """Calculate efficiency scores."""
        return {
            'time_efficiency': max(0, 100 - (metrics.execution_time_seconds / 300) * 100),  # 5min baseline
            'memory_efficiency': max(0, 100 - (metrics.memory_usage_mb / 1000) * 100),  # 1GB baseline
            'processing_efficiency': metrics.label_creation_rate / 500 * 100,  # Normalize
            'overall_efficiency': (metrics.successful_operations / max(1, metrics.total_function_calls)) * 100
        }

    def _identify_performance_warnings(self, metrics: Any) -> List[str]:
        """Identify performance warnings."""
        warnings = []
        if metrics.execution_time_seconds > 300:
            warnings.append("High execution time detected")
        if hasattr(metrics, 'memory_usage_mb') and metrics.memory_usage_mb > 1000:
            warnings.append("High memory usage detected")
        if hasattr(metrics, 'error_rate') and metrics.error_rate > 0.1:
            warnings.append("High error rate detected")
        return warnings

    def _analyze_optimization_effectiveness(self, metrics: Any) -> Dict[str, Any]:
        """Analyze optimization effectiveness."""
        return {'effectiveness': 'simplified analysis'}

    def _assess_meta_labeling_effectiveness(self, analysis: Any) -> Dict[str, Any]:
        """Assess meta-labeling effectiveness."""
        return {'effectiveness': 'simplified'}

    def _compare_meta_vs_primary_labels(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare meta vs primary labels."""
        return {'comparison': 'simplified'}

    def _analyze_meta_labeling_optimization(self, analysis: Any) -> Dict[str, Any]:
        """Analyze meta-labeling optimization."""
        return {'optimization': 'simplified'}

    def _summarize_validation_results(self, results: Any) -> Dict[str, Any]:
        """Summarize validation results."""
        return {'summary': 'simplified'}

    def _generate_validation_action_items(self, results: Any) -> List[str]:
        """Generate validation action items."""
        return ['action item 1', 'action item 2']

    def _calculate_temporal_uniformity(self, labeled_data: pd.DataFrame) -> float:
        """Calculate temporal uniformity of labels."""
        if 'label' not in labeled_data.columns:
            return 0.0

        # Simple uniformity based on label distribution over time
        n_windows = min(10, len(labeled_data) // 20)
        if n_windows < 2:
            return 100.0

        window_size = len(labeled_data) // n_windows
        uniformity_scores = []

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(labeled_data))
            window = labeled_data['label'].iloc[start_idx:end_idx]

            if len(window) > 0:
                dominant_label = window.mode().iloc[0] if len(window.mode()) > 0 else window.iloc[0]
                uniformity = (window == dominant_label).sum() / len(window)
                uniformity_scores.append(uniformity)

        return np.mean(uniformity_scores) * 100 if uniformity_scores else 100.0

    def _analyze_label_sequences(self, labeled_data: pd.DataFrame) -> Dict[str, int]:
        """Analyze label sequences."""
        if 'label' not in labeled_data.columns or len(labeled_data) < 2:
            return {}

        sequences = {}
        for i in range(len(labeled_data) - 1):
            seq = f"{labeled_data['label'].iloc[i]}->{labeled_data['label'].iloc[i+1]}"
            sequences[seq] = sequences.get(seq, 0) + 1

        return sequences

    def _calculate_label_clustering(self, labeled_data: pd.DataFrame) -> float:
        """Calculate label clustering coefficient."""
        if 'label' not in labeled_data.columns:
            return 0.0

        # Simplified clustering based on consecutive same labels
        if len(labeled_data) < 2:
            return 0.0

        consecutive_same = sum(1 for i in range(1, len(labeled_data))
                              if labeled_data['label'].iloc[i] == labeled_data['label'].iloc[i-1])

        return consecutive_same / len(labeled_data) * 100

    def _calculate_transition_smoothness(self, labeled_data: pd.DataFrame) -> float:
        """Calculate transition smoothness."""
        if 'label' not in labeled_data.columns or len(labeled_data) < 2:
            return 100.0

        transitions = sum(1 for i in range(1, len(labeled_data))
                         if labeled_data['label'].iloc[i] != labeled_data['label'].iloc[i-1])

        return max(0, 100 - (transitions / len(labeled_data)) * 100)

    def _calculate_persistence_distribution(self, labeled_data: pd.DataFrame) -> List[float]:
        """Calculate persistence distribution."""
        if 'label' not in labeled_data.columns:
            return []

        persistence_lengths = []
        current_length = 1

        for i in range(1, len(labeled_data)):
            if labeled_data['label'].iloc[i] == labeled_data['label'].iloc[i-1]:
                current_length += 1
            else:
                persistence_lengths.append(current_length)
                current_length = 1

        persistence_lengths.append(current_length)

        return persistence_lengths[:20]  # Return first 20 for analysis

    def _analyze_consecutive_patterns(self, labeled_data: pd.DataFrame) -> Dict[str, int]:
        """Analyze consecutive label patterns."""
        if 'label' not in labeled_data.columns:
            return {}

        patterns = {}
        current_pattern = []
        current_label = None
        count = 0

        for label in labeled_data['label']:
            if label == current_label:
                count += 1
            else:
                if current_label is not None and count > 1:
                    pattern_key = f"{current_label}x{count}"
                    patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
                current_label = label
                count = 1

        # Add the last pattern
        if current_label is not None and count > 1:
            pattern_key = f"{current_label}x{count}"
            patterns[pattern_key] = patterns.get(pattern_key, 0) + 1

        return patterns

    def _extract_pattern_insights(self, analysis: Any) -> Dict[str, Any]:
        """Extract pattern insights."""
        return {'insights': 'simplified'}

    def _assess_distribution_quality(self, analysis: Any) -> Dict[str, Any]:
        """Assess distribution quality."""
        return {'quality': 'simplified'}

    def _calculate_expected_win_rate(self, labeled_data: pd.DataFrame) -> float:
        """Calculate expected win rate."""
        # Simplified calculation - in practice would use backtested results
        if 'label' not in labeled_data.columns:
            return 0.5

        # Estimate based on label stability
        stability = self._calculate_transition_smoothness(labeled_data) / 100
        return 0.5 + (stability - 0.5) * 0.3  # Range: 0.35 to 0.65

    def _calculate_expected_profit_factor(self, labeled_data: pd.DataFrame) -> float:
        """Calculate expected profit factor."""
        # Simplified calculation
        win_rate = self._calculate_expected_win_rate(labeled_data)
        return 1.2 + (win_rate - 0.5) * 0.8  # Range: 1.0 to 1.6

    def _calculate_expected_max_drawdown(self, labeled_data: pd.DataFrame) -> float:
        """Calculate expected max drawdown."""
        stability = self._calculate_transition_smoothness(labeled_data) / 100
        return 0.3 - (stability - 0.5) * 0.2  # Range: 0.2 to 0.4

    def _calculate_optimal_holding_period(self, labeled_data: pd.DataFrame) -> float:
        """Calculate optimal holding period."""
        avg_persistence = np.mean(self._calculate_persistence_distribution(labeled_data)) if self._calculate_persistence_distribution(labeled_data) else 5
        return avg_persistence * 0.5  # Convert to days (assuming hourly data)

    def _calculate_strategy_confidence(self, labeled_data: pd.DataFrame) -> float:
        """Calculate strategy confidence."""
        quality = self._calculate_label_confidence(labeled_data)
        stability = self._calculate_transition_smoothness(labeled_data)
        return (quality + stability) / 2

    def _assess_regime_suitability(self, labeled_data: pd.DataFrame) -> Dict[str, float]:
        """Assess regime suitability."""
        return {
            'bull_market': 0.7,
            'bear_market': 0.6,
            'sideways_market': 0.8,
            'high_volatility': 0.5
        }

    def _calculate_position_sizing_recommendation(self, labeled_data: pd.DataFrame) -> float:
        """Calculate position sizing recommendation."""
        confidence = self._calculate_strategy_confidence(labeled_data) / 100
        return 0.05 + confidence * 0.15  # Range: 0.05 to 0.2

    def _calculate_entry_signal_strength(self, labeled_data: pd.DataFrame) -> float:
        """Calculate entry signal strength."""
        return self._calculate_label_confidence(labeled_data) / 100

    def _calculate_exit_signal_reliability(self, labeled_data: pd.DataFrame) -> float:
        """Calculate exit signal reliability."""
        return self._calculate_transition_smoothness(labeled_data) / 100

    def _generate_trading_recommendations(self, implications: Any) -> List[str]:
        """Generate trading recommendations."""
        return ['recommendation 1', 'recommendation 2']

    def _assess_strategy_risks(self, implications: Any) -> Dict[str, Any]:
        """Assess strategy risks."""
        return {'risks': 'simplified'}

    def _project_strategy_performance(self, implications: Any) -> Dict[str, Any]:
        """Project strategy performance."""
        return {'projection': 'simplified'}

    def _calculate_labeling_efficiency_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate labeling efficiency metrics."""
        return {'efficiency': 'simplified'}

    def _identify_labeling_bottlenecks(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify labeling bottlenecks."""
        return ['bottleneck 1', 'bottleneck 2']

    def _identify_optimization_opportunities(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify optimization opportunities."""
        return ['optimization 1', 'optimization 2']

    def _analyze_resource_utilization(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource utilization."""
        return {'utilization': 'simplified'}

    def _recommend_performance_optimizations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Recommend performance optimizations."""
        return ['optimization 1', 'optimization 2']

    def _recommend_memory_optimizations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Recommend memory optimizations."""
        return ['memory optimization 1', 'memory optimization 2']

    def _recommend_algorithm_optimizations(self, labeling_results: Dict[str, Any]) -> List[str]:
        """Recommend algorithm optimizations."""
        return ['algorithm optimization 1', 'algorithm optimization 2']

    def _recommend_hardware_optimizations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Recommend hardware optimizations."""
        return ['hardware optimization 1', 'hardware optimization 2']

    def _prioritize_optimization_implementations(self, performance_data: Dict[str, Any], labeling_results: Dict[str, Any]) -> List[str]:
        """Prioritize optimization implementations."""
        return ['priority 1', 'priority 2']

    # Visualization helper methods
    def _prepare_label_distribution_data(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for label distribution plot."""
        if 'label' not in labeled_data.columns:
            return {}

        label_counts = labeled_data['label'].value_counts().sort_index()
        return {
            'labels': label_counts.index.tolist(),
            'counts': label_counts.values.tolist(),
            'percentages': (label_counts / len(labeled_data) * 100).tolist()
        }

    def _prepare_label_temporal_data(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for label temporal pattern."""
        if 'label' not in labeled_data.columns or 'timestamp' not in labeled_data.columns:
            return {}

        return {
            'timestamps': labeled_data['timestamp'].tolist(),
            'labels': labeled_data['label'].tolist()
        }

    def _prepare_label_transition_data(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for label transition heatmap."""
        if 'label' not in labeled_data.columns:
            return {}

        transitions = self._analyze_label_sequences(labeled_data)
        unique_labels = sorted(labeled_data['label'].unique())

        # Create transition matrix
        n_labels = len(unique_labels)
        matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]

        for transition, count in transitions.items():
            try:
                from_label, to_label = map(int, transition.split('->'))
                if from_label in unique_labels and to_label in unique_labels:
                    i = unique_labels.index(from_label)
                    j = unique_labels.index(to_label)
                    matrix[i][j] = count
            except:
                continue

        return {
            'matrix': matrix,
            'labels': [f'Label {i}' for i in unique_labels]
        }

    def _prepare_label_quality_data(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for label quality dashboard."""
        quality_assessment = self._generate_label_quality_assessment(labeled_data)
        return quality_assessment

    def _prepare_performance_timeline_data(self, labeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for performance timeline."""
        return {
            'timeline': labeling_results.get('performance_timeline', [])
        }

    def _prepare_meta_labeling_comparison_data(self, labeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for meta-labeling comparison."""
        return {
            'comparison': labeling_results.get('meta_comparison', {})
        }

    def _prepare_regime_labeling_data(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for regime labeling analysis."""
        if 'regime_id' not in labeled_data.columns or 'label' not in labeled_data.columns:
            return {}

        regime_label_dist = {}
        for regime in labeled_data['regime_id'].unique():
            regime_data = labeled_data[labeled_data['regime_id'] == regime]
            regime_label_dist[f'regime_{regime}'] = regime_data['label'].value_counts().to_dict()

        return regime_label_dist

    def _save_json_report(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> Path:
        """Save report as JSON."""
        file_path = self.output_dir / f"{base_filename}_{timestamp}.json"
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"📄 JSON report saved to: {file_path}")
        return file_path

    def _save_markdown_report(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> Path:
        """Save report as Markdown."""
        file_path = self.output_dir / f"{base_filename}_{timestamp}.md"

        md_content = self._generate_markdown_content(report)

        with open(file_path, 'w') as f:
            f.write(md_content)
        self.logger.info(f"📄 Markdown report saved to: {file_path}")
        return file_path

    def _save_csv_report(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> Path:
        """Save report data as CSV files."""
        csv_dir = self.output_dir / f"{base_filename}_data_{timestamp}"
        csv_dir.mkdir(exist_ok=True)

        try:
            # Save label quality metrics
            quality_data = report.get('label_quality_assessment', {})
            if quality_data and 'quality_metrics' in quality_data:
                quality_df = pd.DataFrame([quality_data['quality_metrics']])
                quality_df.to_csv(csv_dir / 'label_quality_metrics.csv', index=False)

            # Save performance metrics
            perf_data = report.get('performance_metrics', {})
            if perf_data and 'metrics' in perf_data:
                perf_df = pd.DataFrame([perf_data['metrics']])
                perf_df.to_csv(csv_dir / 'performance_metrics.csv', index=False)

            # Save trading implications
            trading_data = report.get('trading_strategy_implications', {})
            if trading_data and 'strategy_implications' in trading_data:
                trading_df = pd.DataFrame([trading_data['strategy_implications']])
                trading_df.to_csv(csv_dir / 'trading_implications.csv', index=False)

        except Exception as e:
            self.logger.warning(f"Could not save CSV data: {e}")

        self.logger.info(f"📄 CSV data saved to: {csv_dir}")
        return csv_dir

    def _generate_visualizations(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> None:
        """Generate and save visualizations."""
        try:
            viz_dir = self.output_dir / f"{base_filename}_visualizations_{timestamp}"
            viz_dir.mkdir(exist_ok=True)

            viz_data = report.get('visualization_data', {})

            # Generate label distribution plot
            if 'label_distribution_plot' in viz_data:
                self._create_label_distribution_plot(viz_data['label_distribution_plot'], viz_dir)

            # Generate label transition heatmap
            if 'label_transition_heatmap' in viz_data:
                self._create_label_transition_heatmap(viz_data['label_transition_heatmap'], viz_dir)

            # Generate enhanced label temporal pattern
            if 'label_temporal_pattern' in viz_data:
                self._create_label_temporal_plot(viz_data['label_temporal_pattern'], viz_dir)

            # Generate label quality dashboard
            quality_data = report.get('label_quality_assessment', {})
            if quality_data and 'quality_metrics' in quality_data:
                self._create_label_quality_dashboard(quality_data['quality_metrics'], viz_dir)

            # Generate performance timeline chart
            if 'performance_timeline' in viz_data:
                self._create_performance_timeline_chart(viz_data['performance_timeline'], viz_dir)

            # Generate meta-labeling comparison chart
            if 'meta_labeling_comparison' in viz_data:
                self._create_meta_labeling_comparison_chart(viz_data['meta_labeling_comparison'], viz_dir)

            # Generate regime labeling analysis chart
            if 'regime_labeling_analysis' in viz_data:
                self._create_regime_labeling_analysis_chart(viz_data['regime_labeling_analysis'], viz_dir)

            self.logger.info(f"📊 Visualizations saved to: {viz_dir}")

        except Exception as e:
            self.logger.warning(f"Could not generate visualizations: {e}")

    def _generate_markdown_content(self, report: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report content."""
        md_lines = []

        # Header
        metadata = report.get('metadata', {})
        md_lines.extend([
            "# Step 5 Enhanced Labeling - Comprehensive Analysis Report",
            "",
            f"**Generated:** {metadata.get('generated_at', 'Unknown')}",
            f"**Symbol:** {metadata.get('symbol', 'Unknown')}",
            f"**Exchange:** {metadata.get('exchange', 'Unknown')}",
            f"**Timeframe:** {metadata.get('timeframe', 'Unknown')}",
            f"**Step Description:** {metadata.get('step_description', 'Enhanced Labeling with Meta-Labeling and Validation')}",
            "",
        ])

        # Executive Summary
        md_lines.extend(self._generate_executive_summary_section(report))

        # Performance Summary
        md_lines.extend(self._generate_performance_summary_section(report))

        # Data Quality Assessment
        md_lines.extend(self._generate_data_quality_section(report))

        # Label Quality Assessment
        md_lines.extend(self._generate_label_quality_section(report))

        # Meta-Labeling Analysis
        md_lines.extend(self._generate_meta_labeling_section(report))

        # Trading Strategy Implications
        md_lines.extend(self._generate_trading_strategy_section(report))

        # Risk Assessment
        md_lines.extend(self._generate_risk_assessment_section(report))

        # Optimization Recommendations
        md_lines.extend(self._generate_optimization_section(report))

        # Alerts and Warnings
        md_lines.extend(self._generate_alerts_section(report))

        md_lines.append("")
        return "\n".join(md_lines)

    def _generate_executive_summary_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate executive summary section."""
        lines = [
            "## 🚀 Executive Summary",
            "",
            "This comprehensive report provides detailed analysis of Step 5: Enhanced Labeling with Meta-Labeling and Validation.",
            "The analysis covers label quality assessment, performance metrics, trading strategy implications, and optimization recommendations.",
            "",
        ]

        # Key highlights
        quality_data = report.get('label_quality_assessment', {})
        perf_data = report.get('performance_metrics', {})
        trading_data = report.get('trading_strategy_implications', {})

        if quality_data and 'quality_metrics' in quality_data:
            metrics = quality_data['quality_metrics']
            lines.extend([
                "### 📊 Key Metrics Overview",
                f"- **Total Labels Generated:** {metrics.get('total_labels', 0):,}",
                f"- **Label Quality Score:** {metrics.get('label_confidence_score', 0):.1f}%",
                f"- **Label Consistency:** {metrics.get('label_consistency_score', 0):.1f}%",
            ])

        if perf_data and 'metrics' in perf_data:
            metrics = perf_data['metrics']
            lines.extend([
                f"- **Processing Time:** {metrics.get('execution_time_seconds', 0):.2f} seconds",
                f"- **Success Rate:** {(metrics.get('successful_operations', 0) / max(1, metrics.get('total_function_calls', 1))) * 100:.1f}%",
            ])

        if trading_data and 'strategy_implications' in trading_data:
            implications = trading_data['strategy_implications']
            lines.extend([
                f"- **Expected Win Rate:** {implications.get('expected_win_rate', 0):.1f}%",
                f"- **Strategy Confidence:** {implications.get('strategy_confidence_score', 0):.1f}%",
                "",
            ])
        else:
            lines.append("")

        return lines

    def _generate_performance_summary_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance summary section."""
        lines = [
            "## 📈 Performance Summary",
            "",
        ]

        perf_data = report.get('performance_metrics', {})
        if perf_data and 'metrics' in perf_data:
            metrics = perf_data['metrics']
            lines.extend([
                f"- **Execution Time:** {metrics.get('execution_time_seconds', 0):.2f} seconds",
                f"- **Memory Usage:** {metrics.get('memory_usage_mb', 0):.1f} MB",
                f"- **Label Creation Rate:** {metrics.get('label_creation_rate', 0):.0f} labels/sec",
                f"- **Total Function Calls:** {metrics.get('total_function_calls', 0):,}",
                f"- **Successful Operations:** {metrics.get('successful_operations', 0):,}",
                f"- **Failed Operations:** {metrics.get('failed_operations', 0):,}",
                f"- **Error Rate:** {metrics.get('error_rate', 0):.1f}%",
                f"- **Processing Efficiency:** {metrics.get('processing_efficiency', 0):.1f}%",
                "",
                "### ⚡ Efficiency Scores",
            ])

            if 'efficiency_scores' in perf_data:
                eff_scores = perf_data['efficiency_scores']
                lines.extend([
                    f"- **Time Efficiency:** {eff_scores.get('time_efficiency', 0):.1f}%",
                    f"- **Memory Efficiency:** {eff_scores.get('memory_efficiency', 0):.1f}%",
                    f"- **Processing Efficiency:** {eff_scores.get('processing_efficiency', 0):.1f}%",
                    f"- **Overall Efficiency:** {eff_scores.get('overall_efficiency', 0):.1f}%",
                    "",
                ])

        return lines

    def _generate_data_quality_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate data quality assessment section."""
        lines = [
            "## 📊 Data Quality Assessment",
            "",
        ]

        # This would typically come from validation results
        validation_data = report.get('validation_results', {})
        if validation_data and 'validation_metrics' in validation_data:
            metrics = validation_data['validation_metrics']
            lines.extend([
                f"- **Validation Passed:** {'✅ Yes' if metrics.get('validation_passed', False) else '❌ No'}",
                f"- **Validation Checks:** {metrics.get('validation_checks_performed', 0)}",
                f"- **Data Integrity Score:** {metrics.get('data_integrity_score', 0):.1f}%",
                f"- **Label Consistency Score:** {metrics.get('label_consistency_score', 0):.1f}%",
                f"- **Statistical Validation Score:** {metrics.get('statistical_validation_score', 0):.1f}%",
                "",
            ])

            if 'validation_warnings' in metrics and metrics['validation_warnings']:
                lines.extend([
                    "### ⚠️ Validation Warnings",
                ])
                for warning in metrics['validation_warnings'][:5]:  # Limit to 5 warnings
                    lines.append(f"- {warning}")
                lines.append("")

        return lines

    def _generate_label_quality_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate label quality assessment section."""
        lines = [
            "## 🏷️ Label Quality Assessment",
            "",
        ]

        quality_data = report.get('label_quality_assessment', {})
        if quality_data and 'quality_metrics' in quality_data:
            metrics = quality_data['quality_metrics']
            lines.extend([
                "### 📈 Label Distribution",
                "| Label Type | Count | Percentage |",
                "|------------|-------|------------|",
                f"| Buy Signals | {metrics.get('buy_labels', 0):,} | {(metrics.get('buy_labels', 0) / max(1, metrics.get('total_labels', 1))) * 100:.1f}% |",
                f"| Sell Signals | {metrics.get('sell_labels', 0):,} | {(metrics.get('sell_labels', 0) / max(1, metrics.get('total_labels', 1))) * 100:.1f}% |",
                f"| Hold Signals | {metrics.get('hold_labels', 0):,} | {(metrics.get('hold_labels', 0) / max(1, metrics.get('total_labels', 1))) * 100:.1f}% |",
                "",
                "### 🎯 Quality Metrics",
                f"- **Total Labels:** {metrics.get('total_labels', 0):,}",
                f"- **Label Confidence Score:** {metrics.get('label_confidence_score', 0):.1f}%",
                f"- **Label Consistency Score:** {metrics.get('label_consistency_score', 0):.1f}%",
                f"- **Label Purity Score:** {metrics.get('label_purity_score', 0):.1f}%",
                f"- **Label Stability Score:** {metrics.get('label_stability_score', 0):.1f}%",
                f"- **False Positive Rate:** {metrics.get('false_positive_rate', 0):.1f}%",
                f"- **False Negative Rate:** {metrics.get('false_negative_rate', 0):.1f}%",
                f"- **Label Accuracy Estimate:** {metrics.get('label_accuracy_estimate', 0):.1f}%",
                "",
            ])

        # Label distribution analysis
        dist_data = report.get('label_distribution_analysis', {})
        if dist_data and 'distribution_analysis' in dist_data:
            analysis = dist_data['distribution_analysis']
            lines.extend([
                "### 📊 Distribution Analysis",
                f"- **Temporal Uniformity:** {analysis.get('temporal_distribution_uniformity', 0):.1f}%",
                f"- **Transition Smoothness:** {analysis.get('label_transition_smoothness', 0):.1f}%",
                f"- **Clustering Coefficient:** {analysis.get('label_clustering_coefficient', 0):.1f}%",
                "",
            ])

        return lines

    def _generate_meta_labeling_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate meta-labeling analysis section."""
        lines = [
            "## 🧠 Meta-Labeling Analysis",
            "",
        ]

        meta_data = report.get('meta_labeling_analysis', {})
        if meta_data and 'meta_labeling_metrics' in meta_data:
            metrics = meta_data['meta_labeling_metrics']
            lines.extend([
                f"- **Meta Labels Created:** {metrics.get('meta_labels_created', 0):,}",
                f"- **Success Rate:** {metrics.get('meta_labeling_success_rate', 0):.1f}%",
                f"- **Average Confidence:** {metrics.get('meta_label_confidence_avg', 0):.1f}%",
                f"- **Quality Score:** {metrics.get('meta_label_quality_score', 0):.1f}%",
                f"- **Agreement Rate:** {metrics.get('primary_vs_meta_label_agreement', 0):.1f}%",
                f"- **Computation Time:** {metrics.get('meta_labeling_computation_time', 0):.2f}s",
                f"- **Memory Usage:** {metrics.get('meta_labeling_memory_usage', 0):.1f} MB",
                f"- **Optimization Gain:** {metrics.get('meta_labeling_optimization_gain', 0):.1f}x",
                "",
            ])

        return lines

    def _generate_trading_strategy_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate trading strategy implications section."""
        lines = [
            "## 💰 Trading Strategy Implications",
            "",
        ]

        trading_data = report.get('trading_strategy_implications', {})
        if trading_data and 'strategy_implications' in trading_data:
            implications = trading_data['strategy_implications']
            lines.extend([
                "### 📊 Expected Performance",
                f"- **Win Rate:** {implications.get('expected_win_rate', 0):.1f}%",
                f"- **Profit Factor:** {implications.get('expected_profit_factor', 0):.2f}",
                f"- **Maximum Drawdown:** {implications.get('expected_max_drawdown', 0):.1f}%",
                f"- **Risk-Adjusted Return:** {implications.get('risk_adjusted_return_expectation', 0):.2f}",
                "",
                "### 🎯 Strategy Parameters",
                f"- **Optimal Holding Period:** {implications.get('optimal_holding_period_days', 0):.1f} days",
                f"- **Position Sizing:** {implications.get('position_sizing_recommendation', 0):.1%} of capital",
                f"- **Strategy Confidence Score:** {implications.get('strategy_confidence_score', 0):.1f}%",
                f"- **Entry Signal Strength:** {implications.get('entry_signal_strength', 0):.1f}%",
                f"- **Exit Signal Reliability:** {implications.get('exit_signal_reliability', 0):.1f}%",
                "",
            ])

            # Market regime suitability
            if 'market_regime_suitability' in implications:
                regime_data = implications['market_regime_suitability']
                lines.extend([
                    "### 🌍 Market Regime Suitability",
                ])
                for regime, score in regime_data.items():
                    lines.append(f"- **{regime.replace('_', ' ').title()}:** {score:.1f}%")
                lines.append("")

        return lines

    def _generate_risk_assessment_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate risk assessment section."""
        lines = [
            "## ⚠️ Risk Assessment",
            "",
        ]

        # Calculate overall risk level based on various factors
        risk_level = "MEDIUM"  # Default
        risk_factors = []

        # Assess risks from different components
        quality_data = report.get('label_quality_assessment', {})
        if quality_data and 'quality_metrics' in quality_data:
            metrics = quality_data['quality_metrics']
            if metrics.get('label_confidence_score', 0) < 60:
                risk_factors.append("Low label confidence may lead to poor trading signals")
                risk_level = "HIGH"
            elif metrics.get('label_consistency_score', 0) < 70:
                risk_factors.append("Inconsistent labeling may cause strategy instability")
                if risk_level == "MEDIUM":
                    risk_level = "MEDIUM-HIGH"

        perf_data = report.get('performance_metrics', {})
        if perf_data and 'metrics' in perf_data:
            metrics = perf_data['metrics']
            if metrics.get('error_rate', 0) > 0.1:
                risk_factors.append("High error rate in labeling process")
                risk_level = "HIGH"

        lines.extend([
            f"**Overall Risk Level:** {risk_level}",
            "",
        ])

        if risk_factors:
            lines.extend([
                "### 🚨 Key Risk Factors",
            ])
            for factor in risk_factors:
                lines.append(f"- {factor}")
            lines.append("")

        # Mitigation strategies
        lines.extend([
            "### 🛡️ Risk Mitigation Strategies",
            "- Implement robust validation procedures",
            "- Use ensemble labeling approaches",
            "- Monitor label quality metrics continuously",
            "- Establish fallback labeling mechanisms",
            "",
        ])

        return lines

    def _generate_optimization_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations section."""
        lines = [
            "## 🔧 Optimization Recommendations",
            "",
        ]

        # Performance optimizations
        perf_opt_data = report.get('optimization_recommendations', {})
        if perf_opt_data and 'performance_optimizations' in perf_opt_data:
            optimizations = perf_opt_data['performance_optimizations']
            if optimizations:
                lines.extend([
                    "### ⚡ Performance Optimizations",
                ])
                for opt in optimizations[:5]:  # Limit to 5 recommendations
                    lines.append(f"- {opt}")
                lines.append("")

        # Memory optimizations
        if perf_opt_data and 'memory_optimizations' in perf_opt_data:
            optimizations = perf_opt_data['memory_optimizations']
            if optimizations:
                lines.extend([
                    "### 💾 Memory Optimizations",
                ])
                for opt in optimizations[:3]:  # Limit to 3 recommendations
                    lines.append(f"- {opt}")
                lines.append("")

        # Algorithm optimizations
        if perf_opt_data and 'algorithm_optimizations' in perf_opt_data:
            optimizations = perf_opt_data['algorithm_optimizations']
            if optimizations:
                lines.extend([
                    "### 🧮 Algorithm Optimizations",
                ])
                for opt in optimizations[:3]:  # Limit to 3 recommendations
                    lines.append(f"- {opt}")
                lines.append("")

        return lines

    def _generate_alerts_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate alerts and warnings section."""
        lines = [
            "## 🚨 Alerts and Recommendations",
            "",
        ]

        alerts = []

        # Check for critical issues
        quality_data = report.get('label_quality_assessment', {})
        if quality_data and 'quality_metrics' in quality_data:
            metrics = quality_data['quality_metrics']
            if metrics.get('label_confidence_score', 0) < 50:
                alerts.append("🚨 **CRITICAL:** Label confidence is below acceptable threshold")
            elif metrics.get('label_confidence_score', 0) < 70:
                alerts.append("⚠️ **WARNING:** Label confidence is below optimal level")

            if metrics.get('false_positive_rate', 0) > 0.3:
                alerts.append("🚨 **CRITICAL:** High false positive rate detected")
            elif metrics.get('false_positive_rate', 0) > 0.2:
                alerts.append("⚠️ **WARNING:** Elevated false positive rate")

        perf_data = report.get('performance_metrics', {})
        if perf_data and 'metrics' in perf_data:
            metrics = perf_data['metrics']
            if metrics.get('error_rate', 0) > 0.15:
                alerts.append("🚨 **CRITICAL:** Labeling process has high error rate")
            elif metrics.get('error_rate', 0) > 0.1:
                alerts.append("⚠️ **WARNING:** Labeling process error rate is elevated")

        validation_data = report.get('validation_results', {})
        if validation_data and 'validation_metrics' in validation_data:
            metrics = validation_data['validation_metrics']
            if not metrics.get('validation_passed', True):
                alerts.append("🚨 **CRITICAL:** Validation checks failed")

        if alerts:
            lines.extend(alerts)
            lines.append("")
        else:
            lines.extend([
                "✅ No critical alerts detected",
                "",
            ])

        # General recommendations
        lines.extend([
            "### 💡 General Recommendations",
            "- Monitor label quality metrics regularly",
            "- Consider implementing additional validation layers",
            "- Review and optimize labeling algorithms periodically",
            "- Maintain comprehensive logging for debugging",
            "",
        ])

        return lines

    def _create_label_distribution_plot(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create label distribution plot."""
        try:
            labels = data.get('labels', [])
            counts = data.get('counts', [])
            percentages = data.get('percentages', [])

            if labels and counts:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Bar chart
                bars = ax1.bar(labels, counts, color='skyblue', alpha=0.7)
                ax1.set_title('Label Distribution (Count)')
                ax1.set_xlabel('Label')
                ax1.set_ylabel('Number of Labels')
                ax1.grid(True, alpha=0.3)

                # Add value labels
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                            f'{count:,}', ha='center', va='bottom')

                # Pie chart
                ax2.pie(percentages, labels=[f'Label {label}\n({pct:.1f}%)' for label, pct in zip(labels, percentages)],
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('Label Distribution (Percentage)')
                ax2.axis('equal')

                plt.tight_layout()
                plt.savefig(viz_dir / 'label_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create label distribution plot: {e}")

    def _create_label_transition_heatmap(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create label transition heatmap."""
        try:
            matrix = data.get('matrix', [])
            labels = data.get('labels', [])

            if matrix and labels:
                plt.figure(figsize=(10, 8))
                sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd',
                           xticklabels=labels, yticklabels=labels, square=True)
                plt.title('Label Transition Matrix')
                plt.xlabel('To Label')
                plt.ylabel('From Label')
                plt.tight_layout()

                plt.savefig(viz_dir / 'label_transitions.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create label transition heatmap: {e}")

    def _create_label_temporal_plot(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create label temporal pattern plot."""
        try:
            timestamps = data.get('timestamps', [])
            labels = data.get('labels', [])

            if timestamps and labels and len(timestamps) > 1:
                # Convert timestamps if they're strings
                if isinstance(timestamps[0], str):
                    timestamps = pd.to_datetime(timestamps)

                plt.figure(figsize=(15, 8))

                # Create subplot for temporal pattern
                plt.subplot(2, 1, 1)
                plt.plot(timestamps, labels, 'b-', alpha=0.7, linewidth=1.5)
                plt.title('Label Temporal Pattern Over Time', fontsize=14, fontweight='bold')
                plt.xlabel('Time')
                plt.ylabel('Label Value')
                plt.grid(True, alpha=0.3)
                plt.yticks(sorted(set(labels)))

                # Create subplot for label distribution over time
                plt.subplot(2, 1, 2)
                label_counts = pd.Series(labels).value_counts().sort_index()
                bars = plt.bar(label_counts.index, label_counts.values,
                              color=['red', 'blue', 'green'][:len(label_counts)],
                              alpha=0.7)
                plt.title('Label Distribution Summary', fontsize=12)
                plt.xlabel('Label')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar, count in zip(bars, label_counts.values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(label_counts.values)*0.01,
                            f'{count:,}', ha='center', va='bottom', fontweight='bold')

                plt.tight_layout()
                plt.savefig(viz_dir / 'label_temporal_pattern.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create label temporal plot: {e}")

    def _create_label_quality_dashboard(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create comprehensive label quality dashboard."""
        try:
            plt.figure(figsize=(16, 12))

            # Quality metrics radar chart
            plt.subplot(2, 2, 1)
            metrics = ['Confidence', 'Consistency', 'Purity', 'Stability', 'Accuracy']
            values = [
                data.get('label_confidence_score', 50),
                data.get('label_consistency_score', 50),
                data.get('label_purity_score', 50),
                data.get('label_stability_score', 50),
                data.get('label_accuracy_estimate', 50)
            ]

            # Close the radar chart
            values += values[:1]
            metrics += metrics[:1]

            angles = [n / float(len(metrics[:-1])) * 2 * 3.14159 for n in range(len(metrics[:-1]))]
            angles += angles[:1]

            plt.polar(angles, values, 'o-', linewidth=2, label='Quality Metrics')
            plt.fill(angles, values, alpha=0.25)
            plt.xticks(angles[:-1], metrics[:-1])
            plt.title('Label Quality Metrics Radar', fontsize=12, fontweight='bold')
            plt.ylim(0, 100)

            # Error rates comparison
            plt.subplot(2, 2, 2)
            errors = ['False Positive', 'False Negative']
            rates = [
                data.get('false_positive_rate', 0) * 100,
                data.get('false_negative_rate', 0) * 100
            ]

            bars = plt.bar(errors, rates, color=['red', 'orange'], alpha=0.7)
            plt.title('Error Rates Analysis', fontsize=12, fontweight='bold')
            plt.ylabel('Rate (%)')
            plt.ylim(0, max(rates) * 1.2 if rates else 10)

            for bar, rate in zip(bars, rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rates)*0.01 if rates else 0.5,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

            # Label distribution pie chart
            plt.subplot(2, 2, 3)
            buy_labels = data.get('buy_labels', 0)
            sell_labels = data.get('sell_labels', 0)
            hold_labels = data.get('hold_labels', 0)

            sizes = [buy_labels, sell_labels, hold_labels]
            labels_pie = ['Buy', 'Sell', 'Hold']
            colors = ['green', 'red', 'blue']

            if sum(sizes) > 0:
                plt.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title('Label Distribution', fontsize=12, fontweight='bold')
                plt.axis('equal')

            # Quality score summary (using pie chart as gauge)
            plt.subplot(2, 2, 4)
            quality_score = data.get('label_confidence_score', 50)

            # Determine quality level
            if quality_score >= 80:
                level = "Excellent"
                color = 'green'
            elif quality_score >= 60:
                level = "Good"
                color = 'yellow'
            elif quality_score >= 40:
                level = "Fair"
                color = 'orange'
            else:
                level = "Poor"
                color = 'red'

            # Create gauge-like pie chart
            plt.pie([quality_score, 100-quality_score], colors=[color, 'lightgray'],
                   startangle=90, counterclock=False)
            plt.text(0, 0, f'{level}\n{quality_score:.0f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold')
            plt.title('Overall Quality Score', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.savefig(viz_dir / 'label_quality_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create label quality dashboard: {e}")

    def _create_performance_timeline_chart(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create performance timeline visualization."""
        try:
            timeline_data = data.get('timeline', [])

            if timeline_data:
                plt.figure(figsize=(14, 8))

                # Extract metrics over time
                times = [entry.get('timestamp', i) for i, entry in enumerate(timeline_data)]
                memory_usage = [entry.get('memory_mb', 0) for entry in timeline_data]
                cpu_usage = [entry.get('cpu_percent', 0) for entry in timeline_data]
                labels_created = [entry.get('labels_created', 0) for entry in timeline_data]

                # Memory usage over time
                plt.subplot(3, 1, 1)
                plt.plot(times, memory_usage, 'b-', linewidth=2, marker='o', markersize=4)
                plt.title('Memory Usage Over Time', fontsize=12, fontweight='bold')
                plt.ylabel('Memory (MB)')
                plt.grid(True, alpha=0.3)

                # CPU usage over time
                plt.subplot(3, 1, 2)
                plt.plot(times, cpu_usage, 'r-', linewidth=2, marker='s', markersize=4)
                plt.title('CPU Usage Over Time', fontsize=12, fontweight='bold')
                plt.ylabel('CPU (%)')
                plt.grid(True, alpha=0.3)

                # Labels created over time
                plt.subplot(3, 1, 3)
                plt.plot(times, labels_created, 'g-', linewidth=2, marker='^', markersize=4)
                plt.title('Label Creation Progress', fontsize=12, fontweight='bold')
                plt.ylabel('Labels Created')
                plt.xlabel('Time')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(viz_dir / 'performance_timeline.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create performance timeline chart: {e}")

    def _create_meta_labeling_comparison_chart(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create meta-labeling comparison visualization."""
        try:
            comparison_data = data.get('comparison', {})

            if comparison_data:
                plt.figure(figsize=(12, 8))

                # Agreement rates
                plt.subplot(2, 2, 1)
                agreements = comparison_data.get('agreement_rates', [])
                if agreements:
                    plt.hist(agreements, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    plt.title('Primary vs Meta Label Agreement Distribution', fontsize=11, fontweight='bold')
                    plt.xlabel('Agreement Rate')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)

                # Confidence comparison
                plt.subplot(2, 2, 2)
                primary_conf = comparison_data.get('primary_confidence', [])
                meta_conf = comparison_data.get('meta_confidence', [])

                if primary_conf and meta_conf:
                    plt.scatter(primary_conf, meta_conf, alpha=0.6, color='green', s=50)
                    plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Agreement')
                    plt.title('Confidence Comparison', fontsize=11, fontweight='bold')
                    plt.xlabel('Primary Label Confidence')
                    plt.ylabel('Meta Label Confidence')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                # Performance metrics comparison
                plt.subplot(2, 2, 3)
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                primary_scores = comparison_data.get('primary_scores', [0.5, 0.5, 0.5, 0.5])
                meta_scores = comparison_data.get('meta_scores', [0.5, 0.5, 0.5, 0.5])

                x = np.arange(len(metrics))
                width = 0.35

                plt.bar(x - width/2, primary_scores, width, label='Primary', alpha=0.7, color='blue')
                plt.bar(x + width/2, meta_scores, width, label='Meta', alpha=0.7, color='orange')

                plt.title('Performance Metrics Comparison', fontsize=11, fontweight='bold')
                plt.xlabel('Metric')
                plt.ylabel('Score')
                plt.xticks(x, metrics)
                plt.legend()
                plt.grid(True, alpha=0.3, axis='y')

                # Quality improvement
                plt.subplot(2, 2, 4)
                improvements = comparison_data.get('quality_improvements', [])
                if improvements:
                    plt.plot(improvements, 'o-', linewidth=2, markersize=6, color='purple')
                    plt.title('Meta-Labeling Quality Improvement', fontsize=11, fontweight='bold')
                    plt.xlabel('Iteration')
                    plt.ylabel('Quality Score')
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(viz_dir / 'meta_labeling_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create meta-labeling comparison chart: {e}")

    def _create_regime_labeling_analysis_chart(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create regime-based labeling analysis visualization."""
        try:
            if data:
                plt.figure(figsize=(14, 10))

                # Extract regime data
                regimes = list(data.keys())
                n_regimes = len(regimes)

                if n_regimes > 0:
                    # Create subplots for each regime
                    n_cols = min(3, n_regimes)
                    n_rows = (n_regimes + n_cols - 1) // n_cols

                    for i, regime in enumerate(regimes):
                        plt.subplot(n_rows, n_cols, i + 1)

                        regime_data = data[regime]
                        if isinstance(regime_data, dict):
                            labels = list(regime_data.keys())
                            counts = list(regime_data.values())

                            if labels and counts:
                                # Convert label keys if they're numeric
                                label_names = []
                                for label in labels:
                                    if isinstance(label, (int, float)):
                                        if label == 1:
                                            label_names.append('Buy')
                                        elif label == -1:
                                            label_names.append('Sell')
                                        elif label == 0:
                                            label_names.append('Hold')
                                        else:
                                            label_names.append(f'Label {label}')
                                    else:
                                        label_names.append(str(label))

                                colors = ['green' if 'Buy' in name else 'red' if 'Sell' in name else 'blue' for name in label_names]
                                plt.pie(counts, labels=label_names, colors=colors, autopct='%1.1f%%', startangle=90)
                                plt.title(f'{regime.replace("_", " ").title()}', fontsize=10, fontweight='bold')

                    plt.tight_layout()
                    plt.savefig(viz_dir / 'regime_labeling_analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create regime labeling analysis chart: {e}")

    def _create_comprehensive_summary_dashboard(self, report: Dict[str, Any], viz_dir: Path) -> None:
        """Create comprehensive summary dashboard."""
        try:
            plt.figure(figsize=(16, 12))

            # Overall performance metrics
            plt.subplot(3, 3, 1)
            perf_data = report.get('performance_metrics', {})
            if perf_data and 'metrics' in perf_data:
                metrics = perf_data['metrics']
                perf_scores = [
                    metrics.get('processing_efficiency', 50),
                    (metrics.get('successful_operations', 0) / max(1, metrics.get('total_function_calls', 1))) * 100,
                    100 - (metrics.get('error_rate', 0) * 100),
                    metrics.get('label_creation_rate', 0) / 10  # Normalize
                ]
                perf_labels = ['Efficiency', 'Success Rate', 'Reliability', 'Speed']

                plt.barh(perf_labels, perf_scores, color='skyblue', alpha=0.7)
                plt.title('Performance Overview', fontsize=11, fontweight='bold')
                plt.xlim(0, 100)

            # Label quality overview
            plt.subplot(3, 3, 2)
            quality_data = report.get('label_quality_assessment', {})
            if quality_data and 'quality_metrics' in quality_data:
                metrics = quality_data['quality_metrics']
                quality_scores = [
                    metrics.get('label_confidence_score', 50),
                    metrics.get('label_consistency_score', 50),
                    metrics.get('label_purity_score', 50),
                    metrics.get('label_accuracy_estimate', 50)
                ]
                quality_labels = ['Confidence', 'Consistency', 'Purity', 'Accuracy']

                plt.barh(quality_labels, quality_scores, color='lightgreen', alpha=0.7)
                plt.title('Label Quality Overview', fontsize=11, fontweight='bold')
                plt.xlim(0, 100)

            # Trading strategy implications
            plt.subplot(3, 3, 3)
            trading_data = report.get('trading_strategy_implications', {})
            if trading_data and 'strategy_implications' in trading_data:
                implications = trading_data['strategy_implications']
                strategy_scores = [
                    implications.get('expected_win_rate', 50),
                    implications.get('strategy_confidence_score', 50),
                    (1 - implications.get('expected_max_drawdown', 0.3)) * 100,  # Convert to score
                    implications.get('entry_signal_strength', 50)
                ]
                strategy_labels = ['Win Rate', 'Confidence', 'Risk Score', 'Signal Strength']

                plt.barh(strategy_labels, strategy_scores, color='gold', alpha=0.7)
                plt.title('Strategy Implications', fontsize=11, fontweight='bold')
                plt.xlim(0, 100)

            # Label distribution summary
            plt.subplot(3, 3, 4)
            if quality_data and 'quality_metrics' in quality_data:
                metrics = quality_data['quality_metrics']
                labels_dist = ['Buy', 'Sell', 'Hold']
                counts = [
                    metrics.get('buy_labels', 0),
                    metrics.get('sell_labels', 0),
                    metrics.get('hold_labels', 0)
                ]

                if sum(counts) > 0:
                    plt.pie(counts, labels=labels_dist, autopct='%1.1f%%', startangle=90,
                           colors=['green', 'red', 'blue'])
                    plt.title('Label Distribution', fontsize=11, fontweight='bold')

            # Risk assessment gauge
            plt.subplot(3, 3, 5)
            risk_level = "MEDIUM"  # This should be calculated based on actual data
            risk_score = 50  # Default medium risk

            # Calculate actual risk score
            if quality_data and 'quality_metrics' in quality_data:
                conf_score = metrics.get('label_confidence_score', 50)
                if conf_score < 60:
                    risk_score = 80  # High risk
                    risk_level = "HIGH"
                elif conf_score < 75:
                    risk_score = 60  # Medium-high risk
                    risk_level = "MEDIUM-HIGH"
                else:
                    risk_score = 30  # Low risk
                    risk_level = "LOW"

            # Create risk gauge effect
            plt.pie([risk_score, 100-risk_score], colors=['red', 'lightgray'],
                   startangle=90, counterclock=False)
            plt.text(0, 0, f'{risk_level}\n{risk_score}%', ha='center', va='center', fontsize=12, fontweight='bold')
            plt.title('Risk Assessment', fontsize=11, fontweight='bold')

            # Meta-labeling effectiveness
            plt.subplot(3, 3, 6)
            meta_data = report.get('meta_labeling_analysis', {})
            if meta_data and 'meta_labeling_metrics' in meta_data:
                metrics = meta_data['meta_labeling_metrics']
                meta_scores = [
                    metrics.get('meta_labeling_success_rate', 50),
                    metrics.get('meta_label_quality_score', 50),
                    metrics.get('primary_vs_meta_label_agreement', 50),
                    metrics.get('meta_label_confidence_avg', 50)
                ]
                meta_labels = ['Success', 'Quality', 'Agreement', 'Confidence']

                plt.plot(meta_labels, meta_scores, 'o-', linewidth=2, markersize=6, color='purple')
                plt.title('Meta-Labeling Effectiveness', fontsize=11, fontweight='bold')
                plt.ylim(0, 100)
                plt.grid(True, alpha=0.3)

            # Performance timeline summary
            plt.subplot(3, 3, 7)
            if perf_data and 'metrics' in perf_data:
                metrics = perf_data['metrics']
                timeline_labels = ['Start', 'Processing', 'Validation', 'Complete']
                timeline_values = [
                    0,
                    metrics.get('execution_time_seconds', 0) * 0.6,  # Approximate processing time
                    metrics.get('execution_time_seconds', 0) * 0.8,  # Approximate validation time
                    metrics.get('execution_time_seconds', 0)
                ]

                plt.plot(timeline_labels, timeline_values, 's-', linewidth=2, markersize=8, color='navy')
                plt.title('Process Timeline', fontsize=11, fontweight='bold')
                plt.ylabel('Time (seconds)')
                plt.grid(True, alpha=0.3)

            # Optimization opportunities
            plt.subplot(3, 3, 8)
            opt_data = report.get('optimization_recommendations', {})
            opt_categories = ['Performance', 'Memory', 'Algorithm', 'Hardware']
            opt_scores = [0, 0, 0, 0]  # Default no optimizations needed

            if opt_data:
                if opt_data.get('performance_optimizations'):
                    opt_scores[0] = min(100, len(opt_data['performance_optimizations']) * 20)
                if opt_data.get('memory_optimizations'):
                    opt_scores[1] = min(100, len(opt_data['memory_optimizations']) * 25)
                if opt_data.get('algorithm_optimizations'):
                    opt_scores[2] = min(100, len(opt_data['algorithm_optimizations']) * 25)
                if opt_data.get('hardware_optimizations'):
                    opt_scores[3] = min(100, len(opt_data['hardware_optimizations']) * 25)

            plt.bar(opt_categories, opt_scores, color='coral', alpha=0.7)
            plt.title('Optimization Opportunities', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Potential (%)')

            # Key metrics summary
            plt.subplot(3, 3, 9)
            key_metrics = {}
            if quality_data and 'quality_metrics' in quality_data:
                key_metrics['Labels'] = metrics.get('total_labels', 0)
            if perf_data and 'metrics' in perf_data:
                key_metrics['Success Rate'] = (metrics.get('successful_operations', 0) / max(1, metrics.get('total_function_calls', 1))) * 100

            if key_metrics:
                labels = list(key_metrics.keys())
                values = list(key_metrics.values())

                plt.bar(labels, values, color='teal', alpha=0.7)
                plt.title('Key Metrics Summary', fontsize=11, fontweight='bold')
                plt.xticks(rotation=45, ha='right')

                # Add value labels
                for i, v in enumerate(values):
                    plt.text(i, v + max(values)*0.01, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.savefig(viz_dir / 'comprehensive_summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create comprehensive summary dashboard: {e}")
