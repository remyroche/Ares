"""
Unified Walk-Forward Analyzer for Backtesting

This module provides unified walk-forward analysis functionality for
backtesting across TAS, NAS, and hybrid systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Time period parameters
    train_period_days: int = 252  # 1 year
    test_period_days: int = 63    # 1 quarter
    step_size_days: int = 21      # 1 month
    
    # Data parameters
    min_train_periods: int = 100
    min_test_periods: int = 20
    
    # Model parameters
    retrain_frequency: str = "step"  # "step", "fixed", "adaptive"
    model_selection_method: str = "best_performance"
    
    # Validation parameters
    enable_in_sample_validation: bool = True
    enable_out_of_sample_validation: bool = True
    validation_metric: str = "sharpe_ratio"
    
    # Performance parameters
    enable_regime_analysis: bool = True
    enable_risk_analysis: bool = True
    
    # Output parameters
    save_detailed_results: bool = True
    enable_plotting: bool = True


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""
    
    # Configuration
    config: WalkForwardConfig
    n_periods: int
    
    # Overall performance
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Period-wise results
    period_results: List[Dict[str, Any]]
    
    # Stability metrics
    performance_stability: float
    parameter_stability: float
    
    # Regime analysis
    regime_performance: Optional[Dict[str, Dict[str, float]]] = None
    
    # Risk metrics
    risk_metrics: Optional[Dict[str, float]] = None
    
    # Time series data
    equity_curve: Optional[pd.DataFrame] = None
    performance_by_period: Optional[pd.DataFrame] = None
    
    # Metadata
    execution_time: float
    start_date: datetime
    end_date: datetime


class WalkForwardAnalyzer:
    """
    Unified walk-forward analyzer for backtesting.
    
    Provides comprehensive walk-forward analysis for model validation,
    performance evaluation, and stability assessment.
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize the walk-forward analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(
        self,
        model: Any,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]] = None
    ) -> WalkForwardResult:
        """
        Perform walk-forward analysis.
        
        Args:
            model: Trading model or strategy
            data: Historical market data
            regime_info: Regime information (optional)
            
        Returns:
            WalkForwardResult with analysis results
        """
        start_time = datetime.now()
        self.logger.info("Starting walk-forward analysis")
        
        try:
            # Prepare data
            data = self._prepare_data(data)
            
            # Generate walk-forward periods
            periods = self._generate_periods(data)
            self.logger.info(f"Generated {len(periods)} walk-forward periods")
            
            # Run walk-forward analysis
            period_results = []
            equity_curve = []
            cumulative_equity = 100000  # Starting capital
            
            for i, period in enumerate(periods):
                self.logger.info(f"Processing period {i+1}/{len(periods)}")
                
                # Train model on training data
                train_data = data.loc[period['train_start']:period['train_end']]
                test_data = data.loc[period['test_start']:period['test_end']]
                
                # Train model (simplified)
                trained_model = self._train_model(model, train_data, regime_info)
                
                # Test model on test data
                period_result = self._test_model(trained_model, test_data, regime_info)
                period_result['period'] = i + 1
                period_result['train_start'] = period['train_start']
                period_result['train_end'] = period['train_end']
                period_result['test_start'] = period['test_start']
                period_result['test_end'] = period['test_end']
                
                period_results.append(period_result)
                
                # Update equity curve
                period_return = period_result.get('total_return', 0)
                cumulative_equity *= (1 + period_return)
                
                equity_curve.append({
                    'date': period['test_end'],
                    'equity': cumulative_equity,
                    'period_return': period_return
                })
            
            # Calculate overall performance
            overall_performance = self._calculate_overall_performance(period_results)
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(period_results)
            
            # Calculate regime analysis
            regime_analysis = None
            if self.config.enable_regime_analysis and regime_info:
                regime_analysis = self._calculate_regime_analysis(period_results, regime_info)
            
            # Calculate risk metrics
            risk_metrics = None
            if self.config.enable_risk_analysis:
                risk_metrics = self._calculate_risk_metrics(period_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = WalkForwardResult(
                config=self.config,
                n_periods=len(periods),
                total_return=overall_performance['total_return'],
                annualized_return=overall_performance['annualized_return'],
                sharpe_ratio=overall_performance['sharpe_ratio'],
                max_drawdown=overall_performance['max_drawdown'],
                win_rate=overall_performance['win_rate'],
                period_results=period_results,
                performance_stability=stability_metrics['performance_stability'],
                parameter_stability=stability_metrics['parameter_stability'],
                regime_performance=regime_analysis,
                risk_metrics=risk_metrics,
                equity_curve=pd.DataFrame(equity_curve),
                performance_by_period=pd.DataFrame(period_results),
                execution_time=execution_time,
                start_date=periods[0]['train_start'] if periods else data.index.min(),
                end_date=periods[-1]['test_end'] if periods else data.index.max()
            )
            
            self.logger.info(f"Walk-forward analysis completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed: {e}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for walk-forward analysis."""
        # Ensure proper timestamp index
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
        
        # Sort by date
        data = data.sort_index()
        
        # Add returns if not present
        if 'returns' not in data.columns and 'close' in data.columns:
            data['returns'] = data['close'].pct_change()
        
        return data.dropna()
    
    def _generate_periods(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate walk-forward periods."""
        periods = []
        
        start_date = data.index.min()
        end_date = data.index.max()
        
        current_date = start_date
        
        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_period_days)
            
            # Test period
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_period_days)
            
            # Check if we have enough data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]
            
            if (len(train_data) >= self.config.min_train_periods and 
                len(test_data) >= self.config.min_test_periods):
                
                periods.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
            
            # Move to next period
            current_date += timedelta(days=self.config.step_size_days)
            
            # Stop if we don't have enough data for next period
            if current_date + timedelta(days=self.config.train_period_days + self.config.test_period_days) > end_date:
                break
        
        return periods
    
    def _train_model(
        self,
        model: Any,
        train_data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> Any:
        """Train model on training data."""
        # Simplified model training
        # In practice, this would use the actual model training logic
        
        if hasattr(model, 'fit'):
            try:
                # Prepare features and targets
                features = train_data.drop(columns=['returns'] if 'returns' in train_data.columns else [])
                targets = train_data['returns'] if 'returns' in train_data.columns else train_data.iloc[:, -1]
                
                # Train model
                trained_model = model.fit(features, targets)
                return trained_model
            except Exception as e:
                self.logger.warning(f"Model training failed: {e}")
                return model
        else:
            # Return original model if no training method
            return model
    
    def _test_model(
        self,
        trained_model: Any,
        test_data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test model on test data."""
        try:
            # Get predictions
            if hasattr(trained_model, 'predict'):
                features = test_data.drop(columns=['returns'] if 'returns' in test_data.columns else [])
                predictions = trained_model.predict(features)
            else:
                # Use simple strategy as fallback
                predictions = self._simple_strategy(test_data)
            
            # Calculate returns
            if 'returns' in test_data.columns:
                actual_returns = test_data['returns'].values
            else:
                actual_returns = test_data['close'].pct_change().dropna().values
            
            # Align predictions with actual returns
            min_length = min(len(predictions), len(actual_returns))
            predictions = predictions[:min_length]
            actual_returns = actual_returns[:min_length]
            
            # Calculate performance metrics
            total_return = np.prod(1 + actual_returns) - 1
            volatility = np.std(actual_returns) * np.sqrt(252)
            sharpe_ratio = np.mean(actual_returns) / np.std(actual_returns) * np.sqrt(252) if np.std(actual_returns) > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + actual_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Calculate win rate
            win_rate = np.mean(actual_returns > 0)
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'predictions': predictions,
                'actual_returns': actual_returns
            }
            
        except Exception as e:
            self.logger.warning(f"Model testing failed: {e}")
            return {
                'total_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'predictions': [],
                'actual_returns': []
            }
    
    def _simple_strategy(self, data: pd.DataFrame) -> np.ndarray:
        """Simple strategy as fallback."""
        if 'close' in data.columns:
            # Simple moving average strategy
            sma_short = data['close'].rolling(window=10).mean()
            sma_long = data['close'].rolling(window=30).mean()
            signals = np.where(sma_short > sma_long, 1, -1)
            return signals
        else:
            return np.zeros(len(data))
    
    def _calculate_overall_performance(self, period_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        returns = [result['total_return'] for result in period_results]
        
        total_return = np.prod([1 + r for r in returns]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod([1 + r for r in returns])
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate win rate
        win_rate = np.mean([r > 0 for r in returns])
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def _calculate_stability_metrics(self, period_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate stability metrics."""
        returns = [result['total_return'] for result in period_results]
        sharpe_ratios = [result['sharpe_ratio'] for result in period_results]
        
        # Performance stability (coefficient of variation of returns)
        performance_stability = 1 - (np.std(returns) / abs(np.mean(returns))) if np.mean(returns) != 0 else 0
        
        # Parameter stability (coefficient of variation of Sharpe ratios)
        parameter_stability = 1 - (np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios))) if np.mean(sharpe_ratios) != 0 else 0
        
        return {
            'performance_stability': performance_stability,
            'parameter_stability': parameter_stability
        }
    
    def _calculate_regime_analysis(
        self,
        period_results: List[Dict[str, Any]],
        regime_info: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate regime-specific performance."""
        regime_performance = {}
        
        # This is a simplified implementation
        # In practice, you would map each period to its corresponding regime
        
        for period_result in period_results:
            # Get regime for this period (simplified)
            regime = "unknown"  # Would be determined from regime_info
            
            if regime not in regime_performance:
                regime_performance[regime] = {
                    'returns': [],
                    'sharpe_ratios': [],
                    'drawdowns': []
                }
            
            regime_performance[regime]['returns'].append(period_result['total_return'])
            regime_performance[regime]['sharpe_ratios'].append(period_result['sharpe_ratio'])
            regime_performance[regime]['drawdowns'].append(period_result['max_drawdown'])
        
        # Calculate regime statistics
        for regime, metrics in regime_performance.items():
            regime_performance[regime] = {
                'avg_return': np.mean(metrics['returns']),
                'avg_sharpe': np.mean(metrics['sharpe_ratios']),
                'avg_drawdown': np.mean(metrics['drawdowns']),
                'n_periods': len(metrics['returns'])
            }
        
        return regime_performance
    
    def _calculate_risk_metrics(self, period_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk metrics."""
        returns = [result['total_return'] for result in period_results]
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk
        cvar_95 = np.mean([r for r in returns if r <= var_95])
        
        # Downside deviation
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0
        
        # Sortino ratio
        mean_return = np.mean(returns)
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio
        }
    
    def generate_report(self, result: WalkForwardResult) -> str:
        """Generate walk-forward analysis report."""
        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Configuration
        report.append(f"\nCONFIGURATION:")
        report.append(f"Training Period: {result.config.train_period_days} days")
        report.append(f"Test Period: {result.config.test_period_days} days")
        report.append(f"Step Size: {result.config.step_size_days} days")
        report.append(f"Number of Periods: {result.n_periods}")
        
        # Overall performance
        report.append(f"\nOVERALL PERFORMANCE:")
        report.append(f"Total Return: {result.total_return:.2%}")
        report.append(f"Annualized Return: {result.annualized_return:.2%}")
        report.append(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        report.append(f"Max Drawdown: {result.max_drawdown:.2%}")
        report.append(f"Win Rate: {result.win_rate:.2%}")
        
        # Stability metrics
        report.append(f"\nSTABILITY METRICS:")
        report.append(f"Performance Stability: {result.performance_stability:.3f}")
        report.append(f"Parameter Stability: {result.parameter_stability:.3f}")
        
        # Risk metrics
        if result.risk_metrics:
            report.append(f"\nRISK METRICS:")
            report.append(f"VaR (95%): {result.risk_metrics.get('var_95', 0):.2%}")
            report.append(f"CVaR (95%): {result.risk_metrics.get('cvar_95', 0):.2%}")
            report.append(f"Sortino Ratio: {result.risk_metrics.get('sortino_ratio', 0):.3f}")
        
        # Regime analysis
        if result.regime_performance:
            report.append(f"\nREGIME PERFORMANCE:")
            for regime, perf in result.regime_performance.items():
                report.append(f"\n{regime}:")
                report.append(f"  Avg Return: {perf['avg_return']:.2%}")
                report.append(f"  Avg Sharpe: {perf['avg_sharpe']:.3f}")
                report.append(f"  Avg Drawdown: {perf['avg_drawdown']:.2%}")
                report.append(f"  Periods: {perf['n_periods']}")
        
        # Execution info
        report.append(f"\nEXECUTION INFO:")
        report.append(f"Start Date: {result.start_date}")
        report.append(f"End Date: {result.end_date}")
        report.append(f"Execution Time: {result.execution_time:.2f} seconds")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)