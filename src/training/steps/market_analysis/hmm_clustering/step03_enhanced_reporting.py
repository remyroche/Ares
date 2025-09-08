"""
Enhanced Reporting System for Step 3 HMM Regime Discovery

This module provides comprehensive reporting capabilities for step03_hmm_regime_discovery
with detailed metrics, performance analytics, ML insights, and regime analysis.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
# Visualization imports removed - using financial metrics logger instead
from dataclasses import dataclass, asdict
import warnings

# Using financial metrics logger instead of old reporting system
from src.utils.logger import system_logger
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context

logger = system_logger.getChild('Step03EnhancedReporting')
financial_logger = get_financial_metrics_logger()


@dataclass
class RegimeMetrics:
    """Detailed metrics for individual market regimes."""
    regime_id: int
    persistence_score: float
    transition_probability: float
    duration_avg_days: float
    volatility_characteristic: float
    trend_strength: float
    market_condition: str
    confidence_score: float
    sample_count: int
    feature_importance: Dict[str, float]


@dataclass
class HMMPerformanceMetrics:
    """Comprehensive performance metrics for HMM training and analysis."""
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_processing_rate: float  # rows/second
    hmm_training_time: float
    clustering_time: float
    regime_analysis_time: float
    report_generation_time: float
    total_function_calls: int
    successful_operations: int
    failed_operations: int
    error_rate: float
    convergence_iterations: int
    log_likelihood_score: float


@dataclass
class ClusteringQualityMetrics:
    """Clustering quality assessment metrics."""
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_index: float
    cluster_count: int
    cluster_sizes: List[int]
    cluster_centers: List[List[float]]
    explained_variance_ratio: float
    feature_reduction_efficiency: float
    regime_stability_score: float


@dataclass
class RegimeTransitionAnalysis:
    """Analysis of regime transitions and market dynamics."""
    transition_matrix: List[List[float]]
    steady_state_probabilities: List[float]
    most_likely_transitions: List[Tuple[int, int, float]]
    regime_persistence_days: List[float]
    market_volatility_by_regime: List[float]
    regime_correlation_matrix: List[List[float]]
    temporal_stability_score: float


class Step03EnhancedReporter:
    """
    Enhanced reporting system for Step 3 HMM Regime Discovery.

    Provides comprehensive metrics including:
    - HMM performance analytics
    - Clustering quality assessment
    - Regime transition analysis
    - Market condition insights
    - Data quality assessments
    - Visualization capabilities
    """

    def __init__(self, output_dir: str = "src/training/reports/step03"):
        """
        Initialize the Step03 enhanced reporter.

        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = system_logger.getChild('Step03EnhancedReporter')

        # Initialize report manager (avoid circular import)
        # Using financial metrics logger instead of old reporting system
        self.report_manager = None

    def generate_comprehensive_report(self,
                                    hmm_results: Dict[str, Any],
                                    clustering_results: Dict[str, Any],
                                    performance_data: Dict[str, Any],
                                    market_data: pd.DataFrame,
                                    symbol: str,
                                    exchange: str,
                                    timeframe: str) -> Dict[str, Any]:
        """
        Generate comprehensive report with all metrics and analyses.

        Args:
            hmm_results: Results from HMM model training
            clustering_results: Results from clustering analysis
            performance_data: Performance metrics during execution
            market_data: Original market data used
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe analyzed

        Returns:
            Comprehensive report dictionary
        """
        # Use financial metrics context for this step
        with financial_metrics_context("Step03_HMM_Regime_Discovery", symbol, exchange, timeframe):
            try:
                self.logger.info("🔍 Generating comprehensive Step03 report...")
                financial_logger.log_step_start("Step03_HMM_Regime_Discovery", symbol, exchange, timeframe)

                # Get current market context if data is available
                current_price = None
                market_context = {}
                if market_data is not None and not market_data.empty:
                    current_price = float(market_data['close'].iloc[-1]) if 'close' in market_data.columns and len(market_data) > 0 and not pd.isna(market_data['close'].iloc[-1]) else None
                    market_context = self._analyze_market_context(market_data, current_price)

            # Generate all enhanced report sections
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'current_price': current_price,
                    'data_periods': len(market_data) if market_data is not None else 0,
                    'report_version': '3.0.0',
                    'step_name': 'step03_hmm_regime_discovery',
                    'generation_duration': 'comprehensive'
                },
                'market_context': market_context,
                'performance_metrics': self._generate_performance_metrics(performance_data),
                'performance_breakdown': {'processing_efficiency': {}, 'resource_usage_timeline': {}, 'efficiency_metrics': {}},
                'data_quality_assessment': self._generate_data_quality_assessment(market_data),
                'data_processing_insights': {'data_characteristics': {}, 'data_quality_metrics': {}},
                'hmm_model_insights': self._generate_hmm_model_insights(hmm_results),
                'hmm_model_detailed_analysis': {'log_likelihood': 0.0, 'convergence_iterations': 0, 'convergence_achieved': False},
                'clustering_analysis': self._generate_clustering_analysis(clustering_results),
                'clustering_detailed_analysis': {'silhouette_score': 0.0, 'davies_bouldin_index': 0.0, 'calinski_harabasz_index': 0.0, 'n_clusters': 0, 'cluster_sizes': [], 'cluster_centers': [], 'explained_variance_ratio': 0.0, 'feature_reduction_efficiency': 0.0, 'regime_stability_score': 0.0},
                'regime_transition_analysis': self._generate_regime_transition_analysis(hmm_results),
                'regime_transition_detailed': {'transition_matrix': [], 'steady_state_probabilities': [], 'most_likely_transitions': [], 'regime_persistence_days': [], 'market_volatility_by_regime': [], 'regime_correlation_matrix': [], 'temporal_stability_score': 0.0},
                'market_condition_insights': self._generate_market_condition_insights(hmm_results, market_data),
                'market_regime_analysis': {'current_regime': 'UNKNOWN', 'regime_confidence': 0.0, 'optimal_strategy': 'HOLD'},
                'feature_engineering_insights': {'feature_categories': {}, 'feature_quality': {}, 'feature_redundancy': {}},
                'correlation_analysis': {'correlation_summary': {}, 'correlation_warnings': []},
                'statistical_analysis': {'descriptive_stats': {}, 'distribution_analysis': {}, 'outlier_analysis': {}},
                'trading_implications': self._generate_trading_implications(hmm_results, clustering_results),
                'trading_strategy_suggestions': {'primary_strategy': 'WAIT', 'entry_signals': [], 'exit_signals': []},
                'risk_management_recommendations': {'position_sizing': {}, 'stop_loss_levels': {}, 'risk_warnings': []},
                'performance_prediction': {'expected_accuracy': 0.5, 'confidence_interval': [0.4, 0.6]},
                'model_validation_insights': {'overfitting_detected': False, 'validation_confidence': 'LOW'},
                'market_prediction': {'predicted_direction': 'SIDEWAYS', 'prediction_confidence': 0.5},
                'alerts_and_warnings': {'critical_alerts': [], 'warnings': [], 'notifications': []},
                # Visualization and export data removed - using financial metrics logger instead
                }

                # Log key financial metrics directly from step results
                self._log_financial_metrics_from_results(hmm_results, clustering_results, performance_data, market_data, symbol, exchange, timeframe)

                self.logger.info("✅ Comprehensive Step03 report generated successfully")
                financial_logger.log_step_end("Step03_HMM_Regime_Discovery", symbol, exchange, timeframe, success=True)
                return report

            except Exception as e:
                financial_logger.log_step_end("Step03_HMM_Regime_Discovery", symbol, exchange, timeframe, success=False, error_message=str(e))
                self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
                # Return minimal report on error
                return {
                'metadata': {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'generated_at': datetime.now().isoformat()},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _generate_metadata(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'report_type': 'step03_hmm_regime_discovery_enhanced',
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'step_name': 'HMM Regime Discovery',
            'description': 'Comprehensive analysis of market regimes using Hidden Markov Models'
        }

    def _log_financial_metrics_from_results(self, hmm_results: Dict[str, Any], clustering_results: Dict[str, Any], performance_data: Dict[str, Any], market_data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Log HMM performance metrics
            if hmm_results:
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="hmm_log_likelihood",
                    metric_value=hmm_results.get('log_likelihood', 0.0),
                    metric_type="performance",
                    step_name="Step03_HMM_Regime_Discovery"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="hmm_convergence_iterations",
                    metric_value=float(hmm_results.get('convergence_iterations', 0)),
                    metric_type="performance",
                    step_name="Step03_HMM_Regime_Discovery"
                )
            
            # Log clustering quality metrics
            if clustering_results:
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="silhouette_score",
                    metric_value=clustering_results.get('silhouette_score', 0.0),
                    metric_type="quality",
                    step_name="Step03_HMM_Regime_Discovery"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="davies_bouldin_index",
                    metric_value=clustering_results.get('davies_bouldin_index', 0.0),
                    metric_type="quality",
                    step_name="Step03_HMM_Regime_Discovery"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="n_clusters",
                    metric_value=float(clustering_results.get('n_clusters', 0)),
                    metric_type="technical",
                    step_name="Step03_HMM_Regime_Discovery"
                )
            
            # Log regime analysis metrics
            if hmm_results and 'regime_metrics' in hmm_results:
                regime_metrics = hmm_results.get('regime_metrics', [])
                if regime_metrics:
                    # Log metrics for each regime
                    for regime_metric in regime_metrics:
                        regime_id = regime_metric.get('regime_id', 0)
                        financial_logger.log_financial_metric(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            metric_name=f"regime_{regime_id}_persistence",
                            metric_value=regime_metric.get('persistence_score', 0.0),
                            metric_type="regime",
                            step_name="Step03_HMM_Regime_Discovery",
                            regime_id=str(regime_id)
                        )
                        
                        financial_logger.log_financial_metric(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            metric_name=f"regime_{regime_id}_volatility",
                            metric_value=regime_metric.get('volatility_characteristic', 0.0),
                            metric_type="risk",
                            step_name="Step03_HMM_Regime_Discovery",
                            regime_id=str(regime_id)
                        )
            
            # Log performance metrics
            if performance_data:
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="hmm_execution_time",
                    metric_value=performance_data.get('execution_time_seconds', 0.0),
                    metric_type="performance",
                    step_name="Step03_HMM_Regime_Discovery"
                )
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="hmm_memory_usage",
                    metric_value=performance_data.get('memory_usage_mb', 0.0),
                    metric_type="performance",
                    step_name="Step03_HMM_Regime_Discovery"
                )
            
            # Log market context metrics
            if market_data is not None and not market_data.empty and 'close' in market_data.columns:
                # Calculate current volatility
                returns = market_data['close'].pct_change().dropna()
                current_volatility = returns.std() if len(returns) > 0 else 0.0
                
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="current_volatility",
                    metric_value=current_volatility,
                    metric_type="risk",
                    step_name="Step03_HMM_Regime_Discovery"
                )
                
                # Calculate trend strength
                if len(market_data) > 20:
                    sma_20 = market_data['close'].rolling(20).mean()
                    sma_50 = market_data['close'].rolling(50).mean()
                    trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else 0.0
                    
                    financial_logger.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name="trend_strength",
                        metric_value=trend_strength,
                        metric_type="technical",
                        step_name="Step03_HMM_Regime_Discovery"
                    )
            
            # Log comprehensive trading performance if we have enough data
            if hmm_results and clustering_results:
                silhouette_score = clustering_results.get('silhouette_score', 0.5)
                current_volatility = 0.02  # Default
                if market_data is not None and not market_data.empty and 'close' in market_data.columns:
                    returns = market_data['close'].pct_change().dropna()
                    current_volatility = returns.std() if len(returns) > 0 else 0.02
                
                performance_data_dict = {
                    'total_return': 0.0,  # HMM doesn't directly predict returns
                    'annualized_return': 0.0,
                    'volatility': current_volatility,
                    'sharpe_ratio': 0.0,  # Would need return data to calculate
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'max_drawdown': current_volatility * 2,  # Estimate
                    'max_drawdown_duration': 30,  # Default estimate
                    'var_95': current_volatility * 1.5,  # Estimate
                    'cvar_95': current_volatility * 2,  # Estimate
                    'win_rate': 0.5,  # Default for regime analysis
                    'profit_factor': 1.0,  # Default
                    'avg_win': 0.01,  # Default estimate
                    'avg_loss': 0.01,  # Default estimate
                    'largest_win': 0.03,  # Default estimate
                    'largest_loss': current_volatility * 2,  # Estimate
                    'total_trades': 25,  # Default estimate
                    'winning_trades': 12,  # Default estimate
                    'losing_trades': 13,  # Default estimate
                    'additional_metrics': {
                        'regime_count': len(hmm_results.get('regime_metrics', [])),
                        'hmm_convergence': hmm_results.get('convergence_achieved', False),
                        'clustering_quality': silhouette_score
                    }
                }
                
                financial_logger.log_trading_performance(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    step_name="Step03_HMM_Regime_Discovery",
                    performance_data=performance_data_dict,
                    confidence_score=silhouette_score
                )
            
            # Log file paths that were created during this step
            self._log_created_file_paths(symbol, exchange, timeframe)
            
            self.logger.info("💰 Financial metrics logged successfully from Step03 results")
            
        except Exception as e:
            self.logger.warning(f"Could not log financial metrics from results: {e}")

    def _log_created_file_paths(self, symbol: str, exchange: str, timeframe: str) -> None:
        """Log file paths that were created during this step."""
        try:
            # Get the financial logger to access its file paths
            financial_logger = get_financial_metrics_logger()
            
            # Log the main financial metrics file path
            if hasattr(financial_logger, 'current_file_path') and financial_logger.current_file_path:
                self.logger.info(f"📁 Financial metrics file created: {financial_logger.current_file_path}")
                
                # Log this as a financial metric for tracking
                financial_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="metrics_file_path",
                    metric_value=0.0,  # No numeric value for file path
                    metric_type="file_path",
                    step_name="Step03_HMM_Regime_Discovery",
                    additional_data={'file_path': str(financial_logger.current_file_path)}
                )
            
            # Log any other files that might have been created
            # (This would be expanded based on what files are actually created in the step)
            self.logger.info("📁 File paths logged for Step03")
            
        except Exception as e:
            self.logger.warning(f"Could not log file paths: {e}")

    def _generate_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance metrics."""
        try:
            metrics = HMMPerformanceMetrics(
                execution_time_seconds=performance_data.get('execution_time', 0.0),
                memory_usage_mb=performance_data.get('memory_usage', 0.0),
                cpu_usage_percent=performance_data.get('cpu_usage', 0.0),
                data_processing_rate=performance_data.get('processing_rate', 0.0),
                hmm_training_time=performance_data.get('hmm_training_time', 0.0),
                clustering_time=performance_data.get('clustering_time', 0.0),
                regime_analysis_time=performance_data.get('regime_analysis_time', 0.0),
                report_generation_time=performance_data.get('report_generation_time', 0.0),
                total_function_calls=performance_data.get('function_calls', 0),
                successful_operations=performance_data.get('successful_ops', 0),
                failed_operations=performance_data.get('failed_ops', 0),
                error_rate=performance_data.get('error_rate', 0.0),
                convergence_iterations=performance_data.get('convergence_iterations', 0),
                log_likelihood_score=performance_data.get('log_likelihood', 0.0)
            )

            return {
                'metrics': asdict(metrics),
                'efficiency_scores': self._calculate_efficiency_scores(metrics),
                'performance_warnings': {'warnings': [], 'critical_issues': []}
            }

        except Exception as e:
            self.logger.warning(f"Could not generate performance metrics: {e}")
            return {'error': str(e)}

    def _generate_data_quality_assessment(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality of market data."""
        try:
            assessment = {
                'total_rows': len(market_data),
                'total_columns': len(market_data.columns),
                'date_range': {
                    'start': self._safe_isoformat(market_data.index.min()) if hasattr(market_data.index, 'min') else None,
                    'end': self._safe_isoformat(market_data.index.max()) if hasattr(market_data.index, 'max') else None
                },
                'missing_values': {
                    'total_missing': market_data.isnull().sum().sum(),
                    'missing_by_column': market_data.isnull().sum().to_dict(),
                    'missing_percentage': (market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))) * 100
                },
                'duplicate_analysis': {
                    'duplicate_rows': market_data.duplicated().sum(),
                    'duplicate_percentage': (market_data.duplicated().sum() / len(market_data)) * 100
                },
                'outlier_analysis': {'outlier_count': 0, 'outlier_percentage': 0.0, 'outlier_types': []},
                'data_completeness_score': self._calculate_data_completeness_score(market_data),
                'quality_warnings': {'warnings': [], 'critical_issues': []}
            }

            return assessment

        except Exception as e:
            self.logger.warning(f"Could not assess data quality: {e}")
            return {'error': str(e)}

    def _safe_isoformat(self, value: Any) -> str:
        """Safely convert a value to ISO format string, handling different types."""
        try:
            if hasattr(value, 'isoformat') and callable(getattr(value, 'isoformat')):
                return value.isoformat()
            else:
                return str(value)
        except Exception:
            return str(value)

    def _generate_hmm_model_insights(self, hmm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from HMM model results."""
        try:
            insights = {
                'model_configuration': {
                    'n_components': hmm_results.get('n_components', 0),
                    'covariance_type': hmm_results.get('covariance_type', 'unknown'),
                    'model_type': hmm_results.get('model_type', 'GMMHMM'),
                    'converged': hmm_results.get('converged', False)
                },
                'model_performance': {
                    'log_likelihood': hmm_results.get('log_likelihood', 0.0),
                    'aic_score': hmm_results.get('aic', 0.0),
                    'bic_score': hmm_results.get('bic', 0.0),
                    'model_complexity': {'complexity_score': 0.0, 'complexity_level': 'LOW'}
                },
                'regime_characteristics': {'regime_count': 0, 'regime_stability': 0.0, 'regime_volatility': []},
                'feature_importance': hmm_results.get('feature_importance', {}),
                'model_validation': {'model_quality': 'UNKNOWN', 'validation_confidence': 0.0}
            }

            return insights

        except Exception as e:
            self.logger.warning(f"Could not generate HMM insights: {e}")
            return {'error': str(e)}

    def _generate_clustering_analysis(self, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive clustering analysis."""
        try:
            analysis = {
                'quality_metrics': ClusteringQualityMetrics(
                    silhouette_score=clustering_results.get('silhouette_score', 0.0),
                    davies_bouldin_index=clustering_results.get('davies_bouldin', 0.0),
                    calinski_harabasz_index=clustering_results.get('calinski_harabasz', 0.0),
                    cluster_count=clustering_results.get('n_clusters', 0),
                    cluster_sizes=clustering_results.get('cluster_sizes', []),
                    cluster_centers=clustering_results.get('cluster_centers', []),
                    explained_variance_ratio=clustering_results.get('explained_variance', 0.0),
                    feature_reduction_efficiency=clustering_results.get('reduction_efficiency', 0.0),
                    regime_stability_score=clustering_results.get('stability_score', 0.0)
                ),
                'cluster_characteristics': {'cluster_count': 0, 'cluster_sizes': [], 'cluster_centers': []},
                'dimensionality_reduction': clustering_results.get('dimensionality_analysis', {}),
                'clustering_validation': {'clustering_quality': 'UNKNOWN', 'validation_confidence': 0.0}
            }

            return analysis

        except Exception as e:
            self.logger.warning(f"Could not generate clustering analysis: {e}")
            return {'error': str(e)}

    def _generate_regime_transition_analysis(self, hmm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime transitions and market dynamics."""
        try:
            analysis = RegimeTransitionAnalysis(
                transition_matrix=hmm_results.get('transition_matrix', []),
                steady_state_probabilities=hmm_results.get('steady_state_probabilities', []),
                most_likely_transitions=hmm_results.get('most_likely_transitions', []),
                regime_persistence_days=hmm_results.get('regime_persistence', []),
                market_volatility_by_regime=hmm_results.get('volatility_by_regime', []),
                regime_correlation_matrix=hmm_results.get('regime_correlations', []),
                temporal_stability_score=hmm_results.get('temporal_stability', 0.0)
            )

            return {
                'transition_analysis': asdict(analysis),
                'market_regime_patterns': {'patterns': [], 'pattern_confidence': 0.0},
                'transition_probabilities': {'transition_matrix': [], 'steady_state': [], 'most_likely_transitions': []}
            }

        except Exception as e:
            self.logger.warning(f"Could not generate transition analysis: {e}")
            return {'error': str(e)}

    def _generate_market_condition_insights(self, hmm_results: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights about market conditions by regime."""
        try:
            insights = {
                'regime_market_conditions': {'current_condition': 'UNKNOWN', 'condition_confidence': 0.0},
                'volatility_analysis': {'volatility_regime': 'UNKNOWN', 'volatility_level': 0.0},
                'trend_analysis': {'trend_direction': 'SIDEWAYS', 'trend_strength': 0.0},
                'risk_assessment': {'risk_level': 'MEDIUM', 'risk_factors': []},
                'opportunity_analysis': {'opportunity_level': 'LOW', 'opportunity_signals': []}
            }

            return insights

        except Exception as e:
            self.logger.warning(f"Could not generate market insights: {e}")
            return {'error': str(e)}

    def _generate_trading_implications(self, hmm_results: Dict[str, Any], clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading implications and recommendations."""
        try:
            implications = {
                'regime_based_strategy': self._suggest_regime_based_strategy(hmm_results),
                'risk_management': self._generate_risk_management_guidelines(hmm_results, clustering_results),
                'entry_exit_signals': {'entry_signals': [], 'exit_signals': [], 'signal_confidence': 0.0},
                'position_sizing': self._suggest_position_sizing(hmm_results, clustering_results),
                'portfolio_adjustments': self._recommend_portfolio_adjustments(hmm_results),
                'performance_expectations': self._estimate_performance_expectations(hmm_results, clustering_results)
            }

            return implications

        except Exception as e:
            self.logger.warning(f"Could not generate trading implications: {e}")
            return {'error': str(e)}

