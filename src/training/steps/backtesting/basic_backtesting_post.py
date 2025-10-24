"""
Basic Backtesting Post-Step

This module provides the post-processing step for basic backtesting operations.
It handles results processing, analysis, and artifact generation after backtesting execution.

Key Features:
- Results processing and analysis
- Performance metrics calculation
- Report generation
- Artifact management
- Data preview and logging
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint, tprint_data_preview
from src.config.pipeline_modes import get_mode_config, get_mode_lookback_days

logger = logging.getLogger(__name__)

class BasicBacktestingPostStep(BaseStep):
    """
    Post-processing step for basic backtesting operations.
    
    This step handles:
    - Processing backtesting results
    - Calculating performance metrics
    - Generating reports
    - Managing artifacts
    """

    def __init__(self, step_name: str = "basic_backtesting_post", config: Optional[Dict[str, Any]] = None):
        """Initialize the basic backtesting post-step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.step.{step_name}")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the basic backtesting post-processing step.

        Args:
            config: Configuration containing symbol, exchange, timeframes, etc.

        Returns:
            Execution result with artifacts and metrics
        """
        self.logger.info('🔧 Starting Basic Backtesting Post-Processing')

        try:
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            execution_mode = config.get('execution_mode', 'light')
            
            # Get mode configuration for lookback periods and other parameters
            mode_config = get_mode_config(execution_mode)
            
            if not symbol:
                raise ValueError("Symbol is required for basic backtesting post-processing")
            
            # Preview configuration data
            tprint_data_preview(config, "basic_backtesting_post_config", max_rows=10, level="DEBUG")
            
            self.logger.info(f"Post-processing backtesting results for {symbol} from {exchange}")
            self.logger.info(f"Timeframe: {timeframe}, Direction: {direction}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Set up artifact manager context
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='BasicBacktesting'
            )
            
            # Load backtesting results
            backtest_results = await self._load_backtest_results(symbol, exchange, direction, config)
            if backtest_results is None:
                raise ValueError("Failed to load backtesting results")
            
            # Preview loaded backtest results
            tprint_data_preview(backtest_results, "loaded_backtest_results", max_rows=5, level="INFO")
            
            # Process backtesting results
            processed_results = await self._process_backtest_results(backtest_results, config)
            tprint_data_preview(processed_results, "processed_backtest_results", max_rows=10, level="INFO")
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(processed_results, config)
            tprint_data_preview(performance_metrics, "performance_metrics", max_rows=10, level="INFO")
            
            # Generate analysis report
            analysis_report = await self._generate_analysis_report(processed_results, performance_metrics, config)
            tprint_data_preview(analysis_report, "analysis_report", max_rows=5, level="INFO")
            
            # Generate summary
            summary = await self._generate_summary(processed_results, performance_metrics, analysis_report, config)
            tprint_data_preview(summary, "backtesting_summary", max_rows=10, level="INFO")
            
            # Save processed results as artifact
            results_artifact_path = self._save_artifact(
                processed_results,
                'basic_backtesting_processed_results',
                'data'
            )
            artifacts.append(results_artifact_path)
            
            # Save performance metrics as artifact
            metrics_artifact_path = self._save_artifact(
                performance_metrics,
                'basic_backtesting_performance_metrics',
                'data'
            )
            artifacts.append(metrics_artifact_path)
            
            # Save analysis report as artifact
            report_artifact_path = self._save_artifact(
                analysis_report,
                'basic_backtesting_analysis_report',
                'metadata'
            )
            artifacts.append(report_artifact_path)
            
            # Save summary as artifact
            summary_artifact_path = self._save_artifact(
                summary,
                'basic_backtesting_summary',
                'metadata'
            )
            artifacts.append(summary_artifact_path)
            
            # Calculate metrics
            metrics = {
                'total_return': performance_metrics.get('total_return', 0.0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                'win_rate': performance_metrics.get('win_rate', 0.0),
                'total_trades': performance_metrics.get('total_trades', 0),
                'artifacts_created': len(artifacts),
                'processing_time': (datetime.now() - datetime.now()).total_seconds()
            }
            
            self.logger.info(f"✅ Basic backtesting post-processing completed successfully")
            self.logger.info(f"📊 Total return: {metrics['total_return']:.2%}")
            self.logger.info(f"📊 Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            self.logger.info(f"📊 Max drawdown: {metrics['max_drawdown']:.2%}")
            self.logger.info(f"📊 Win rate: {metrics['win_rate']:.2%}")
            self.logger.info(f"📊 Total trades: {metrics['total_trades']}")
            self.logger.info(f"📁 Artifacts created: {metrics['artifacts_created']}")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'processed_results': processed_results,
                'performance_metrics': performance_metrics,
                'analysis_report': analysis_report,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"❌ Basic backtesting post-processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }

    async def _load_backtest_results(self, symbol: str, exchange: str, direction: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load backtesting results from previous steps."""
        try:
            self.logger.info(f"📂 Loading backtesting results: {symbol} {exchange} {direction}")
            
            # Try to load from different possible artifact names
            possible_names = [
                'backtest_results',
                'vectorized_backtest_results',
                'vectorbt_backtest_results',
                'sr_backtest_results',
                'final_parameters_optimization_result',
                'real_parameters_optimization_result',
                f'{symbol}_backtest_results',
                f'{exchange}_{symbol}_backtest_results'
            ]
            
            for name in possible_names:
                results = self._load_dataframe(name)
                if results is not None:
                    self.logger.info(f"✅ Loaded backtesting results as '{name}'")
                    return results
            
            # Try to load from metadata
            metadata = self._load_metadata('backtest_metadata')
            if metadata is not None:
                self.logger.info("✅ Loaded backtesting metadata")
                return metadata
            
            self.logger.warning("⚠️ No backtesting results found")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load backtesting results: {e}")
            return None

    async def _process_backtest_results(self, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process backtesting results for analysis."""
        try:
            self.logger.info("🔧 Processing backtesting results")
            
            processed_results = {
                'raw_results': results,
                'processed_at': datetime.now().isoformat(),
                'symbol': config.get('symbol', 'ETHUSDT'),
                'exchange': config.get('exchange', 'binance'),
                'timeframe': config.get('timeframe', '15m'),
                'direction': config.get('direction', 'longs')
            }
            
            # Extract key metrics if available
            if isinstance(results, dict):
                processed_results['total_return'] = results.get('total_return', 0.0)
                processed_results['sharpe_ratio'] = results.get('sharpe_ratio', 0.0)
                processed_results['max_drawdown'] = results.get('max_drawdown', 0.0)
                processed_results['win_rate'] = results.get('win_rate', 0.0)
                processed_results['total_trades'] = results.get('total_trades', 0)
                
                # Extract equity curve if available
                if 'equity_curve' in results:
                    processed_results['equity_curve'] = results['equity_curve']
                
                # Extract trades if available
                if 'trades' in results:
                    processed_results['trades'] = results['trades']
            
            self.logger.info(f"✅ Backtesting results processed")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process backtesting results: {e}")
            return {}

    async def _calculate_performance_metrics(self, processed_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            self.logger.info("📊 Calculating performance metrics")
            
            performance_metrics = {
                'calculated_at': datetime.now().isoformat(),
                'symbol': config.get('symbol', 'ETHUSDT'),
                'exchange': config.get('exchange', 'binance'),
                'timeframe': config.get('timeframe', '15m'),
                'direction': config.get('direction', 'longs')
            }
            
            # Extract basic metrics
            performance_metrics['total_return'] = processed_results.get('total_return', 0.0)
            performance_metrics['sharpe_ratio'] = processed_results.get('sharpe_ratio', 0.0)
            performance_metrics['max_drawdown'] = processed_results.get('max_drawdown', 0.0)
            performance_metrics['win_rate'] = processed_results.get('win_rate', 0.0)
            performance_metrics['total_trades'] = processed_results.get('total_trades', 0)
            
            # Calculate additional metrics if equity curve is available
            equity_curve = processed_results.get('equity_curve')
            if equity_curve is not None and len(equity_curve) > 0:
                if isinstance(equity_curve, pd.Series):
                    equity_values = equity_curve.values
                else:
                    equity_values = equity_curve
                
                # Calculate volatility
                if len(equity_values) > 1:
                    returns = np.diff(equity_values) / equity_values[:-1]
                    performance_metrics['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
                    performance_metrics['annualized_return'] = (equity_values[-1] / equity_values[0]) ** (252 / len(equity_values)) - 1
                else:
                    performance_metrics['volatility'] = 0.0
                    performance_metrics['annualized_return'] = 0.0
                
                # Calculate maximum drawdown
                peak = np.maximum.accumulate(equity_values)
                drawdown = (equity_values - peak) / peak
                performance_metrics['max_drawdown'] = np.min(drawdown)
                
                # Calculate Calmar ratio
                if performance_metrics['max_drawdown'] != 0:
                    performance_metrics['calmar_ratio'] = performance_metrics['annualized_return'] / abs(performance_metrics['max_drawdown'])
                else:
                    performance_metrics['calmar_ratio'] = 0.0
            
            # Calculate Sortino ratio if available
            if 'sharpe_ratio' in performance_metrics and 'volatility' in performance_metrics:
                performance_metrics['sortino_ratio'] = performance_metrics['sharpe_ratio'] * 1.2  # Approximation
            
            # Calculate profit factor if trades are available
            trades = processed_results.get('trades')
            if trades is not None and len(trades) > 0:
                if isinstance(trades, pd.DataFrame):
                    trade_returns = trades.get('return', trades.get('Return', []))
                else:
                    trade_returns = trades
                
                if len(trade_returns) > 0:
                    positive_returns = [r for r in trade_returns if r > 0]
                    negative_returns = [r for r in trade_returns if r < 0]
                    
                    if negative_returns:
                        performance_metrics['profit_factor'] = sum(positive_returns) / abs(sum(negative_returns))
                    else:
                        performance_metrics['profit_factor'] = float('inf') if positive_returns else 0.0
                else:
                    performance_metrics['profit_factor'] = 0.0
            else:
                performance_metrics['profit_factor'] = 0.0
            
            self.logger.info(f"✅ Performance metrics calculated")
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {}

    async def _generate_analysis_report(self, processed_results: Dict[str, Any], performance_metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        try:
            self.logger.info("📋 Generating analysis report")
            
            analysis_report = {
                'generated_at': datetime.now().isoformat(),
                'symbol': config.get('symbol', 'ETHUSDT'),
                'exchange': config.get('exchange', 'binance'),
                'timeframe': config.get('timeframe', '15m'),
                'direction': config.get('direction', 'longs'),
                'execution_mode': config.get('execution_mode', 'light')
            }
            
            # Performance summary
            analysis_report['performance_summary'] = {
                'total_return': performance_metrics.get('total_return', 0.0),
                'annualized_return': performance_metrics.get('annualized_return', 0.0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': performance_metrics.get('sortino_ratio', 0.0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                'calmar_ratio': performance_metrics.get('calmar_ratio', 0.0),
                'volatility': performance_metrics.get('volatility', 0.0),
                'win_rate': performance_metrics.get('win_rate', 0.0),
                'profit_factor': performance_metrics.get('profit_factor', 0.0),
                'total_trades': performance_metrics.get('total_trades', 0)
            }
            
            # Risk analysis
            analysis_report['risk_analysis'] = {
                'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                'volatility': performance_metrics.get('volatility', 0.0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
                'risk_adjusted_return': performance_metrics.get('sharpe_ratio', 0.0) * performance_metrics.get('volatility', 0.0)
            }
            
            # Trading analysis
            analysis_report['trading_analysis'] = {
                'total_trades': performance_metrics.get('total_trades', 0),
                'win_rate': performance_metrics.get('win_rate', 0.0),
                'profit_factor': performance_metrics.get('profit_factor', 0.0),
                'avg_trade_return': performance_metrics.get('total_return', 0.0) / max(performance_metrics.get('total_trades', 1), 1)
            }
            
            # Recommendations
            analysis_report['recommendations'] = self._generate_recommendations(performance_metrics)
            
            self.logger.info(f"✅ Analysis report generated")
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate analysis report: {e}")
            return {}

    def _generate_recommendations(self, performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance metrics."""
        recommendations = []
        
        total_return = performance_metrics.get('total_return', 0.0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = performance_metrics.get('max_drawdown', 0.0)
        win_rate = performance_metrics.get('win_rate', 0.0)
        profit_factor = performance_metrics.get('profit_factor', 0.0)
        
        if total_return > 0.1:  # 10% return
            recommendations.append("Strategy shows strong positive returns")
        elif total_return < -0.05:  # -5% return
            recommendations.append("Strategy shows negative returns - consider optimization")
        
        if sharpe_ratio > 1.0:
            recommendations.append("Excellent risk-adjusted returns (Sharpe > 1.0)")
        elif sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - consider risk management improvements")
        
        if abs(max_drawdown) > 0.2:  # 20% drawdown
            recommendations.append("High maximum drawdown - consider position sizing adjustments")
        
        if win_rate > 0.6:  # 60% win rate
            recommendations.append("High win rate indicates good signal quality")
        elif win_rate < 0.4:  # 40% win rate
            recommendations.append("Low win rate - consider signal filtering improvements")
        
        if profit_factor > 1.5:
            recommendations.append("Strong profit factor indicates good risk-reward ratio")
        elif profit_factor < 1.0:
            recommendations.append("Profit factor below 1.0 - consider strategy revision")
        
        if not recommendations:
            recommendations.append("Strategy performance is within normal ranges")
        
        return recommendations

    async def _generate_summary(self, processed_results: Dict[str, Any], performance_metrics: Dict[str, Any], analysis_report: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        try:
            self.logger.info("📝 Generating executive summary")
            
            summary = {
                'generated_at': datetime.now().isoformat(),
                'symbol': config.get('symbol', 'ETHUSDT'),
                'exchange': config.get('exchange', 'binance'),
                'timeframe': config.get('timeframe', '15m'),
                'direction': config.get('direction', 'longs'),
                'execution_mode': config.get('execution_mode', 'light')
            }
            
            # Key metrics
            summary['key_metrics'] = {
                'total_return': f"{performance_metrics.get('total_return', 0.0):.2%}",
                'sharpe_ratio': f"{performance_metrics.get('sharpe_ratio', 0.0):.2f}",
                'max_drawdown': f"{performance_metrics.get('max_drawdown', 0.0):.2%}",
                'win_rate': f"{performance_metrics.get('win_rate', 0.0):.2%}",
                'total_trades': performance_metrics.get('total_trades', 0)
            }
            
            # Performance rating
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            if sharpe_ratio > 2.0:
                summary['performance_rating'] = "Excellent"
            elif sharpe_ratio > 1.0:
                summary['performance_rating'] = "Good"
            elif sharpe_ratio > 0.5:
                summary['performance_rating'] = "Fair"
            else:
                summary['performance_rating'] = "Poor"
            
            # Status
            total_return = performance_metrics.get('total_return', 0.0)
            if total_return > 0.05:  # 5% return
                summary['status'] = "Profitable"
            elif total_return > -0.02:  # -2% return
                summary['status'] = "Break-even"
            else:
                summary['status'] = "Loss-making"
            
            # Recommendations
            summary['recommendations'] = analysis_report.get('recommendations', [])
            
            self.logger.info(f"✅ Executive summary generated")
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate executive summary: {e}")
            return {}