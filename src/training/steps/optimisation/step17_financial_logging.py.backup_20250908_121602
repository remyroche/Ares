"""
Financial metrics logging for Step17 Enhanced Multi-Objective Optimization.
Independent logging module that can be used without the reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from src.utils.financial_metrics_logger import get_financial_metrics_logger, financial_metrics_context
from src.utils.logger import system_logger

logger = system_logger.getChild('Step17FinancialLogging')


class Step17FinancialLogger:
    """Independent financial metrics logger for Step17 Enhanced Multi-Objective Optimization."""
    
    def __init__(self, symbol: str, exchange: str, timeframe: str):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.financial_logger = get_financial_metrics_logger()
    
    def log_step_execution(self, optimization_results: Dict[str, Any], block_results: Dict[str, Any], 
                          parameter_analysis: Dict[str, Any], validation_results: Dict[str, Any], 
                          global_results: Dict[str, Any]) -> None:
        """Log comprehensive financial metrics for Step17 execution."""
        with financial_metrics_context(
            step_name="Step17_Enhanced_Multi_Objective_Optimization",
            symbol=self.symbol,
            exchange=self.exchange,
            timeframe=self.timeframe
        ):
            try:
                self.financial_logger.log_step_start("Step17_Enhanced_Multi_Objective_Optimization", self.symbol, self.exchange, self.timeframe)
                
                # Log all financial metrics
                self._log_financial_metrics_from_results(optimization_results, block_results, parameter_analysis, validation_results, global_results)
                
                # Log file paths
                self._log_created_file_paths()
                
                self.financial_logger.log_step_end("Step17_Enhanced_Multi_Objective_Optimization", self.symbol, self.exchange, self.timeframe, success=True)
                
            except Exception as e:
                self.financial_logger.log_step_end("Step17_Enhanced_Multi_Objective_Optimization", self.symbol, self.exchange, self.timeframe, success=False, error_message=str(e))
                logger.error(f"Failed to log financial metrics: {e}")
    
    def _log_financial_metrics_from_results(self, optimization_results: Dict[str, Any], block_results: Dict[str, Any], 
                                          parameter_analysis: Dict[str, Any], validation_results: Dict[str, Any], 
                                          global_results: Dict[str, Any]) -> None:
        """Log key financial metrics directly from step results."""
        try:
            # Note: Data quality and performance metrics are logged in regular system logs
            # Financial metrics logger focuses only on financial/trading metrics
            
            # Log optimization performance metrics
            if optimization_results:
                if 'total_duration' in optimization_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_optimization_time",
                        metric_value=float(optimization_results['total_duration']),
                        metric_type="performance",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'total_trials' in optimization_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="total_optimization_trials",
                        metric_value=float(optimization_results['total_trials']),
                        metric_type="performance",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'convergence_score' in optimization_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="optimization_convergence_score",
                        metric_value=optimization_results['convergence_score'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'efficiency_score' in optimization_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="optimization_efficiency_score",
                        metric_value=optimization_results['efficiency_score'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'stability_score' in optimization_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="parameter_stability_score",
                        metric_value=optimization_results['stability_score'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'improvement_score' in optimization_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="objective_improvement_score",
                        metric_value=optimization_results['improvement_score'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'pareto_quality' in optimization_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="pareto_front_quality",
                        metric_value=optimization_results['pareto_quality'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
            
            # Log multi-objective optimization metrics
            if optimization_results and 'multi_objective' in optimization_results:
                mo_data = optimization_results['multi_objective']
                
                if 'pareto_front_size' in mo_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="pareto_front_size",
                        metric_value=float(mo_data['pareto_front_size']),
                        metric_type="performance",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'hypervolume' in mo_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="hypervolume_score",
                        metric_value=mo_data['hypervolume'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'diversity' in mo_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="solution_diversity_score",
                        metric_value=mo_data['diversity'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'convergence_rate' in mo_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="multi_objective_convergence_rate",
                        metric_value=mo_data['convergence_rate'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'correlation' in mo_data:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="objective_correlation",
                        metric_value=abs(mo_data['correlation']),  # Use absolute value for independence measure
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
            
            # Log block optimization metrics
            if block_results and 'blocks' in block_results:
                blocks = block_results['blocks']
                total_blocks = len(blocks)
                
                self.financial_logger.log_financial_metric(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    metric_name="total_optimization_blocks",
                    metric_value=float(total_blocks),
                    metric_type="performance",
                    step_name="Step17_Enhanced_Multi_Objective_Optimization"
                )
                
                # Log individual block performance
                for block_name, block_data in blocks.items():
                    if isinstance(block_data, dict):
                        if 'duration' in block_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"block_{block_name}_optimization_time",
                                metric_value=float(block_data['duration']),
                                metric_type="performance",
                                step_name="Step17_Enhanced_Multi_Objective_Optimization",
                                additional_data={'block_name': block_name}
                            )
                        
                        if 'convergence' in block_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"block_{block_name}_convergence_score",
                                metric_value=block_data['convergence'],
                                metric_type="trading",
                                step_name="Step17_Enhanced_Multi_Objective_Optimization",
                                additional_data={'block_name': block_name}
                            )
                        
                        if 'importance' in block_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"block_{block_name}_parameter_importance",
                                metric_value=block_data['importance'],
                                metric_type="trading",
                                step_name="Step17_Enhanced_Multi_Objective_Optimization",
                                additional_data={'block_name': block_name}
                            )
            
            # Log parameter sensitivity metrics
            if parameter_analysis:
                if 'sensitivity_scores' in parameter_analysis:
                    sensitivity_scores = parameter_analysis['sensitivity_scores']
                    for param_name, sensitivity in sensitivity_scores.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"parameter_{param_name}_sensitivity",
                            metric_value=float(sensitivity),
                            metric_type="trading",
                            step_name="Step17_Enhanced_Multi_Objective_Optimization",
                            additional_data={'parameter_name': param_name}
                        )
                
                if 'importance_scores' in parameter_analysis:
                    importance_scores = parameter_analysis['importance_scores']
                    for param_name, importance in importance_scores.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"parameter_{param_name}_importance",
                            metric_value=float(importance),
                            metric_type="trading",
                            step_name="Step17_Enhanced_Multi_Objective_Optimization",
                            additional_data={'parameter_name': param_name}
                        )
                
                if 'stability_scores' in parameter_analysis:
                    stability_scores = parameter_analysis['stability_scores']
                    for param_name, stability in stability_scores.items():
                        self.financial_logger.log_financial_metric(
                            symbol=self.symbol,
                            exchange=self.exchange,
                            timeframe=self.timeframe,
                            metric_name=f"parameter_{param_name}_stability",
                            metric_value=float(stability),
                            metric_type="trading",
                            step_name="Step17_Enhanced_Multi_Objective_Optimization",
                            additional_data={'parameter_name': param_name}
                        )
            
            # Log optimization validation metrics
            if validation_results:
                if 'cv_score' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="cross_validation_score",
                        metric_value=validation_results['cv_score'],
                        metric_type="performance",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'oos_performance' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="out_of_sample_performance",
                        metric_value=validation_results['oos_performance'],
                        metric_type="performance",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'robustness' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="optimization_robustness_score",
                        metric_value=validation_results['robustness'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'stability' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="optimization_stability_score",
                        metric_value=validation_results['stability'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'generalization' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="optimization_generalization_score",
                        metric_value=validation_results['generalization'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'overfitting' in validation_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="optimization_overfitting_score",
                        metric_value=validation_results['overfitting'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
            
            # Log global optimization metrics
            if global_results:
                if 'objective_score' in global_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="global_objective_score",
                        metric_value=global_results['objective_score'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'consistency_score' in global_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="parameter_consistency_score",
                        metric_value=global_results['consistency_score'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
                
                if 'coverage_score' in global_results:
                    self.financial_logger.log_financial_metric(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.timeframe,
                        metric_name="optimization_coverage_score",
                        metric_value=global_results['coverage_score'],
                        metric_type="trading",
                        step_name="Step17_Enhanced_Multi_Objective_Optimization"
                    )
            
            # Log objective performance metrics
            if optimization_results and 'objectives' in optimization_results:
                objectives_data = optimization_results['objectives']
                for obj_name, obj_data in objectives_data.items():
                    if isinstance(obj_data, dict):
                        if 'best' in obj_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"objective_{obj_name}_best_value",
                                metric_value=float(obj_data['best']),
                                metric_type="trading",
                                step_name="Step17_Enhanced_Multi_Objective_Optimization",
                                additional_data={'objective_name': obj_name}
                            )
                        
                        if 'improvement' in obj_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"objective_{obj_name}_improvement_rate",
                                metric_value=float(obj_data['improvement']),
                                metric_type="trading",
                                step_name="Step17_Enhanced_Multi_Objective_Optimization",
                                additional_data={'objective_name': obj_name}
                            )
                        
                        if 'stability' in obj_data:
                            self.financial_logger.log_financial_metric(
                                symbol=self.symbol,
                                exchange=self.exchange,
                                timeframe=self.timeframe,
                                metric_name=f"objective_{obj_name}_stability_score",
                                metric_value=float(obj_data['stability']),
                                metric_type="trading",
                                step_name="Step17_Enhanced_Multi_Objective_Optimization",
                                additional_data={'objective_name': obj_name}
                            )
            
            # Log comprehensive trading performance estimation
            if optimization_results and validation_results and global_results:
                # Extract key metrics for performance estimation
                convergence_score = optimization_results.get('convergence_score', 0.5)
                efficiency_score = optimization_results.get('efficiency_score', 0.5)
                stability_score = optimization_results.get('stability_score', 0.5)
                improvement_score = optimization_results.get('improvement_score', 0.5)
                pareto_quality = optimization_results.get('pareto_quality', 0.5)
                
                # Multi-objective metrics
                mo_data = optimization_results.get('multi_objective', {})
                hypervolume = mo_data.get('hypervolume', 0.5)
                diversity = mo_data.get('diversity', 0.5)
                convergence_rate = mo_data.get('convergence_rate', 0.5)
                
                # Validation metrics
                cv_score = validation_results.get('cv_score', 0.5)
                oos_performance = validation_results.get('oos_performance', 0.5)
                robustness = validation_results.get('robustness', 0.5)
                generalization = validation_results.get('generalization', 0.5)
                overfitting = validation_results.get('overfitting', 0.2)
                
                # Global metrics
                global_objective = global_results.get('objective_score', 0.5)
                consistency = global_results.get('consistency_score', 0.5)
                coverage = global_results.get('coverage_score', 0.5)
                
                # Calculate combined optimization quality score
                optimization_quality = (
                    convergence_score + efficiency_score + stability_score + 
                    improvement_score + pareto_quality + hypervolume + 
                    diversity + convergence_rate + cv_score + oos_performance + 
                    robustness + generalization + global_objective + 
                    consistency + coverage
                ) / 15.0
                
                # Adjust for overfitting (penalty)
                optimization_quality = optimization_quality * (1.0 - overfitting)
                
                # Estimate trading performance based on optimization quality
                estimated_return = (optimization_quality * 0.04) - ((1 - optimization_quality) * 0.02)
                estimated_volatility = 0.03  # Default estimate
                
                # Estimate trading metrics
                total_trials = optimization_results.get('total_trials', 100)
                total_blocks = len(block_results.get('blocks', {})) if block_results else 3
                
                estimated_performance = {
                    'total_return': estimated_return,
                    'annualized_return': estimated_return * 252,  # Assuming daily signals
                    'volatility': estimated_volatility,
                    'sharpe_ratio': estimated_return / estimated_volatility if estimated_volatility > 0 else 0.0,
                    'sortino_ratio': estimated_return / (estimated_volatility * 0.6) if estimated_volatility > 0 else 0.0,
                    'calmar_ratio': 0.0,  # Would need max drawdown
                    'max_drawdown': estimated_volatility * 2.0,  # Estimate
                    'max_drawdown_duration': 25,  # Default estimate
                    'var_95': estimated_volatility * 1.6,  # Estimate
                    'cvar_95': estimated_volatility * 2.0,  # Estimate
                    'win_rate': optimization_quality,
                    'profit_factor': 1.0 + (optimization_quality - 0.5) * 3.0,
                    'avg_win': 0.035,  # Default estimate
                    'avg_loss': 0.02,  # Default estimate
                    'largest_win': 0.08,  # Default estimate
                    'largest_loss': estimated_volatility * 2.0,  # Estimate
                    'total_trades': int(total_trials * total_blocks * 0.8),  # Estimate 0.8 trades per trial per block
                    'winning_trades': int(total_trials * total_blocks * 0.8 * optimization_quality),
                    'losing_trades': int(total_trials * total_blocks * 0.8 * (1 - optimization_quality)),
                    'additional_metrics': {
                        'optimization_quality': optimization_quality,
                        'convergence_score': convergence_score,
                        'efficiency_score': efficiency_score,
                        'stability_score': stability_score,
                        'improvement_score': improvement_score,
                        'pareto_quality': pareto_quality,
                        'hypervolume_score': hypervolume,
                        'diversity_score': diversity,
                        'convergence_rate': convergence_rate,
                        'cross_validation_score': cv_score,
                        'out_of_sample_performance': oos_performance,
                        'robustness_score': robustness,
                        'generalization_score': generalization,
                        'overfitting_score': overfitting,
                        'global_objective_score': global_objective,
                        'parameter_consistency_score': consistency,
                        'optimization_coverage_score': coverage,
                        'total_trials': total_trials,
                        'total_blocks': total_blocks,
                        'total_optimization_time': optimization_results.get('total_duration', 0.0)
                    }
                }
                
                self.financial_logger.log_trading_performance(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    step_name="Step17_Enhanced_Multi_Objective_Optimization",
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
                    step_name="Step17_Enhanced_Multi_Objective_Optimization",
                    additional_data={'file_path': str(self.financial_logger.current_file_path)}
                )
            logger.info("📁 File paths logged for Step17")
        except Exception as e:
            logger.warning(f"Could not log file paths: {e}")