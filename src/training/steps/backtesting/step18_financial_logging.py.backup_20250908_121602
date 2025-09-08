"""
Financial metrics logging for Step18 Backtesting Main.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step18FinancialLogging')


class Step18FinancialLogger:
    """Independent financial metrics logger for Step18 Backtesting Main."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, backtesting_results: Dict[str, Any], validation_results: Dict[str, Any], 
                          execution_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step18 execution."""
        with financial_metrics_context(
            step_name="Step18_Backtesting_Main",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step18_Backtesting_Main", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(backtesting_results, validation_results, execution_data, performance_metrics)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step18_Backtesting_Main", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step18_Backtesting_Main", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, backtesting_results: Dict[str, Any], validation_results: Dict[str, Any], 
                                          execution_data: Dict[str, Any], performance_metrics: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log backtesting performance metrics
            if backtesting_results:
                if 'total_backtesting_time' in backtesting_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_backtesting_time",
                        metric_value=float(backtesting_results['total_backtesting_time']),
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'execution_efficiency' in backtesting_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="execution_efficiency",
                        metric_value=backtesting_results['execution_efficiency'],
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'parallel_processing_gain' in backtesting_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="parallel_processing_gain",
                        metric_value=backtesting_results['parallel_processing_gain'],
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'memory_utilization' in backtesting_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="memory_utilization",
                        metric_value=backtesting_results['memory_utilization'],
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'data_processing_speed' in backtesting_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="data_processing_speed",
                        metric_value=backtesting_results['data_processing_speed'],
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'regime_processing_coverage' in backtesting_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="regime_processing_coverage",
                        metric_value=backtesting_results['regime_processing_coverage'],
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
            
            # Log walk forward validation metrics
            if validation_results and 'walk_forward' in validation_results:
                wf_data = validation_results['walk_forward']
                
                if 'total_runs' in wf_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_total_runs",
                        metric_value=float(wf_data['total_runs']),
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'efficiency' in wf_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_efficiency",
                        metric_value=wf_data['efficiency'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'oos_performance' in wf_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_out_of_sample_performance",
                        metric_value=wf_data['oos_performance'],
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'overfitting_score' in wf_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_overfitting_score",
                        metric_value=wf_data['overfitting_score'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'stability_score' in wf_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_stability_score",
                        metric_value=wf_data['stability_score'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'decay_analysis' in wf_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="walk_forward_prediction_decay",
                        metric_value=wf_data['decay_analysis'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
            
            # Log Monte Carlo validation metrics
            if validation_results and 'monte_carlo' in validation_results:
                mc_data = validation_results['monte_carlo']
                
                if 'total_simulations' in mc_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="monte_carlo_total_simulations",
                        metric_value=float(mc_data['total_simulations']),
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'significance' in mc_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="monte_carlo_statistical_significance",
                        metric_value=mc_data['significance'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'scenario_coverage' in mc_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="monte_carlo_scenario_coverage",
                        metric_value=mc_data['scenario_coverage'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'robustness' in mc_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="monte_carlo_robustness_score",
                        metric_value=mc_data['robustness'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
            
            # Log A/B testing metrics
            if validation_results and 'ab_testing' in validation_results:
                ab_data = validation_results['ab_testing']
                
                if 'total_tests' in ab_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ab_testing_total_tests",
                        metric_value=float(ab_data['total_tests']),
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'significance' in ab_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ab_testing_statistical_significance",
                        metric_value=ab_data['significance'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'winner_rate' in ab_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ab_testing_winner_detection_rate",
                        metric_value=ab_data['winner_rate'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'false_positive' in ab_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ab_testing_false_positive_rate",
                        metric_value=ab_data['false_positive'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'test_power' in ab_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="ab_testing_power_analysis",
                        metric_value=ab_data['test_power'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
            
            # Log model persistence metrics
            if backtesting_results and 'persistence' in backtesting_results:
                persistence_data = backtesting_results['persistence']
                
                if 'total_saved' in persistence_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_models_saved",
                        metric_value=float(persistence_data['total_saved']),
                        metric_type="performance",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'compression_ratio' in persistence_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="model_compression_ratio",
                        metric_value=persistence_data['compression_ratio'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'save_load_perf' in persistence_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="save_load_performance",
                        metric_value=persistence_data['save_load_perf'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'integrity_score' in persistence_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="persistence_integrity_score",
                        metric_value=persistence_data['integrity_score'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'version_efficiency' in persistence_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="version_control_efficiency",
                        metric_value=persistence_data['version_efficiency'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'reproducibility' in persistence_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="model_reproducibility_score",
                        metric_value=persistence_data['reproducibility'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
            
            # Log quality assessment metrics
            if performance_metrics:
                if 'data_quality' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="data_quality_score",
                        metric_value=performance_metrics['data_quality'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'validation_completeness' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="validation_completeness_score",
                        metric_value=performance_metrics['validation_completeness'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'reproducibility' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="result_reproducibility_score",
                        metric_value=performance_metrics['reproducibility'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'statistical_rigor' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="statistical_rigor_score",
                        metric_value=performance_metrics['statistical_rigor'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'methodological_soundness' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="methodological_soundness_score",
                        metric_value=performance_metrics['methodological_soundness'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
                
                if 'risk_coverage' in performance_metrics:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="risk_assessment_coverage",
                        metric_value=performance_metrics['risk_coverage'],
                        metric_type="trading",
                        step_name="Step18_Backtesting_Main"
                    )
            
            # Log regime backtesting metrics
            if execution_data and 'regimes' in execution_data:
                regimes_data = execution_data['regimes']
                total_regimes = len(regimes_data)
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_regimes_processed",
                    metric_value=float(total_regimes),
                    metric_type="performance",
                    step_name="Step18_Backtesting_Main"
                )
                
                # Log regime-specific performance
                for regime_id, regime_data in regimes_data.items():
                    if isinstance(regime_data, dict):
                        if 'performance' in regime_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"regime_{regime_id}_performance",
                                metric_value=regime_data['performance'],
                                metric_type="trading",
                                step_name="Step18_Backtesting_Main",
                                regime_id=str(regime_id)
                            )
                        
                        if 'adaptability' in regime_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"regime_{regime_id}_adaptability",
                                metric_value=regime_data['adaptability'],
                                metric_type="trading",
                                step_name="Step18_Backtesting_Main",
                                regime_id=str(regime_id)
                            )
            
            # Log risk assessment metrics
            if performance_metrics:
                risk_metrics = {
                    'var_95': performance_metrics.get('var_95', 0.0),
                    'expected_shortfall': performance_metrics.get('expected_shortfall', 0.0),
                    'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
                    'sortino_ratio': performance_metrics.get('sortino_ratio', 0.0),
                    'calmar_ratio': performance_metrics.get('calmar_ratio', 0.0)
                }
                
                for risk_name, risk_value in risk_metrics.items():
                    if risk_value is not None:
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=risk_name,
                            metric_value=float(risk_value),
                            metric_type="risk",
                            step_name="Step18_Backtesting_Main"
                        )
            
            # Log comprehensive trading performance estimation
            if backtesting_results and validation_results and performance_metrics:
                # Extract key metrics for performance estimation
                execution_efficiency = backtesting_results.get('execution_efficiency', 0.5)
                parallel_gain = backtesting_results.get('parallel_processing_gain', 0.5)
                regime_coverage = backtesting_results.get('regime_processing_coverage', 0.5)
                
                # Walk forward validation metrics
                wf_data = validation_results.get('walk_forward', {})
                wf_efficiency = wf_data.get('efficiency', 0.5)
                oos_performance = wf_data.get('oos_performance', 0.5)
                wf_stability = wf_data.get('stability_score', 0.5)
                overfitting = wf_data.get('overfitting_score', 0.2)
                
                # Monte Carlo validation metrics
                mc_data = validation_results.get('monte_carlo', {})
                mc_significance = mc_data.get('significance', 0.5)
                mc_robustness = mc_data.get('robustness', 0.5)
                scenario_coverage = mc_data.get('scenario_coverage', 0.5)
                
                # A/B testing metrics
                ab_data = validation_results.get('ab_testing', {})
                ab_significance = ab_data.get('significance', 0.5)
                winner_rate = ab_data.get('winner_rate', 0.5)
                test_power = ab_data.get('test_power', 0.5)
                
                # Quality metrics
                data_quality = performance_metrics.get('data_quality', 0.5)
                validation_completeness = performance_metrics.get('validation_completeness', 0.5)
                reproducibility = performance_metrics.get('reproducibility', 0.5)
                statistical_rigor = performance_metrics.get('statistical_rigor', 0.5)
                
                # Risk metrics
                sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
                max_drawdown = performance_metrics.get('max_drawdown', 0.1)
                var_95 = performance_metrics.get('var_95', 0.05)
                
                # Calculate combined backtesting quality score
                backtesting_quality = (
                    execution_efficiency + parallel_gain + regime_coverage + 
                    wf_efficiency + oos_performance + wf_stability + 
                    mc_significance + mc_robustness + scenario_coverage + 
                    ab_significance + winner_rate + test_power + 
                    data_quality + validation_completeness + reproducibility + 
                    statistical_rigor
                ) / 15.0
                
                # Adjust for overfitting (penalty)
                backtesting_quality = backtesting_quality * (1.0 - overfitting)
                
                # Estimate trading performance based on backtesting quality
                estimated_return = (backtesting_quality * 0.05) - ((1 - backtesting_quality) * 0.025)
                estimated_volatility = 0.03  # Default estimate
                
                # Estimate trading metrics
                total_regimes = len(execution_data.get('regimes', {})) if execution_data else 3
                wf_runs = wf_data.get('total_runs', 10)
                mc_sims = mc_data.get('total_simulations', 100)
                ab_tests = ab_data.get('total_tests', 5)
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.6) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2.5,  # Estimate
                    'max_drawdown_duration': 35,  # Default estimate
                    'var_95': estimated_volatility * 1.8,  # Estimate
                    'cvar_95': estimated_volatility * 2.2,  # Estimate
                    'win_rate': backtesting_quality,
                    'profit_factor': 1.0 + (backtesting_quality - 0.5) * 3.5,
                    'avg_win': 0.04,  # Default estimate
                    'avg_loss': 0.025,  # Default estimate
                    'largest_win': 0.09,  # Default estimate
                    'largest_loss': estimated_volatility * 2.5,  # Estimate
                    'total_trades': int(total_regimes * wf_runs * 2.5),  # Estimate 2.5 trades per regime per run
                    'winning_trades': int(total_regimes * wf_runs * 2.5 * backtesting_quality),
                    'losing_trades': int(total_regimes * wf_runs * 2.5 * (1 - backtesting_quality)),
                    'additional_metrics': {
                        'backtesting_quality': backtesting_quality,
                        'execution_efficiency': execution_efficiency,
                        'parallel_processing_gain': parallel_gain,
                        'regime_processing_coverage': regime_coverage,
                        'walk_forward_efficiency': wf_efficiency,
                        'out_of_sample_performance': oos_performance,
                        'walk_forward_stability': wf_stability,
                        'overfitting_score': overfitting,
                        'monte_carlo_significance': mc_significance,
                        'monte_carlo_robustness': mc_robustness,
                        'scenario_coverage': scenario_coverage,
                        'ab_testing_significance': ab_significance,
                        'winner_detection_rate': winner_rate,
                        'test_power_analysis': test_power,
                        'data_quality_score': data_quality,
                        'validation_completeness': validation_completeness,
                        'result_reproducibility': reproducibility,
                        'statistical_rigor': statistical_rigor,
                        'total_regimes_processed': total_regimes,
                        'walk_forward_runs': wf_runs,
                        'monte_carlo_simulations': mc_sims,
                        'ab_tests_performed': ab_tests,
                        'total_backtesting_time': backtesting_results.get('total_backtesting_time', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step18_Backtesting_Main",
                    **estimated_performance
                )
            
        except Exception as e:
            logger.error(f"Failed to log financial metrics from results: {e}")
    
    def _log_created_file_paths(self) -> None:
        """Log file paths that were created during this step."""
        try:
            if hasattr(self.financial_logger, 'current_file_path') and self.financial_logger.current_file_path:
                logger.info(f"📁 Financial metrics file created: {self.financial_logger.current_file_path}")
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="metrics_file_path",
                    metric_value=0.0,
                    metric_type="file_path",
                    step_name="Step18_Backtesting_Main",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step18")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")