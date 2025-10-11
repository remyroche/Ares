"""
VectorBT Performance Metrics

Enhanced performance metrics calculation using VectorBT for improved accuracy and speed.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from dataclasses import dataclass

from .vectorbt_base import VectorBTBase, VectorBTError
from .vectorbt_config import VectorBTConfig

logger = logging.getLogger(__name__)

@dataclass
class MetricsResult:
    """Result from metrics calculation."""
    metrics: Dict[str, Any]
    execution_time: float
    data_points: int

class VectorBTMetrics(VectorBTBase):
    """
    VectorBT Performance Metrics
    
    Provides comprehensive performance metrics calculation using VectorBT.
    """
    
    def __init__(self, config: VectorBTConfig):
        """Initialize VectorBT metrics calculator."""
        super().__init__(config)
        self.logger.info("VectorBT Metrics initialized")
    
    def calculate_comprehensive_metrics(self, 
                                      returns: pd.Series,
                                      equity_curve: Optional[pd.Series] = None,
                                      benchmark_returns: Optional[pd.Series] = None) -> MetricsResult:
        """Calculate comprehensive performance metrics."""
        start_time = time.time()
        
        try:
            # Validate returns
            if returns is None or len(returns) == 0:
                raise ValueError("Returns data is empty or None")
            
            returns = returns.dropna()
            if len(returns) == 0:
                raise ValueError("No valid returns data after cleaning")
            
            # Calculate all metric categories
            basic_metrics = self._calculate_basic_metrics(returns)
            risk_metrics = self._calculate_risk_metrics(returns, equity_curve)
            performance_metrics = self._calculate_performance_metrics(returns, equity_curve)
            drawdown_metrics = self._calculate_drawdown_metrics(returns, equity_curve)
            
            # Calculate benchmark metrics if provided
            benchmark_metrics = {}
            if benchmark_returns is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)
            
            # Combine all metrics
            all_metrics = {
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'performance_metrics': performance_metrics,
                'drawdown_metrics': drawdown_metrics,
                'benchmark_metrics': benchmark_metrics
            }
            
            execution_time = time.time() - start_time
            
            result = MetricsResult(
                metrics=all_metrics,
                execution_time=execution_time,
                data_points=len(returns)
            )
            
            self.log_performance("calculate_comprehensive_metrics", execution_time)
            self.logger.info(f"Comprehensive metrics calculated in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive metrics calculation failed: {e}")
            raise VectorBTError(f"Metrics calculation failed: {e}")
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate basic return metrics."""
        try:
            # Basic statistics
            total_return = (1 + returns).prod() - 1
            mean_return = returns.mean()
            median_return = returns.median()
            std_return = returns.std()
            
            # Annualized metrics
            periods_per_year = 252  # Assuming daily data
            annualized_return = (1 + mean_return) ** periods_per_year - 1
            annualized_volatility = std_return * np.sqrt(periods_per_year)
            
            # Min/Max returns
            min_return = returns.min()
            max_return = returns.max()
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            return {
                'total_return': total_return,
                'mean_return': mean_return,
                'median_return': median_return,
                'std_return': std_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'min_return': min_return,
                'max_return': max_return,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            
        except Exception as e:
            self.logger.error(f"Basic metrics calculation failed: {e}")
            return {}
    
    def _calculate_risk_metrics(self, returns: pd.Series, equity_curve: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate risk metrics."""
        try:
            # Value at Risk (VaR)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # Expected Shortfall (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Upside deviation
            upside_returns = returns[returns > 0]
            upside_deviation = upside_returns.std() if len(upside_returns) > 0 else 0
            
            # Tail ratio
            tail_ratio = upside_deviation / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio components
            if equity_curve is not None:
                peak = equity_curve.cummax()
                drawdown = (equity_curve - peak) / peak
                max_drawdown = abs(drawdown.min())
            else:
                # Calculate from returns
                cumulative = (1 + returns).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
                max_drawdown = abs(drawdown.min())
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'downside_deviation': downside_deviation,
                'upside_deviation': upside_deviation,
                'tail_ratio': tail_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _calculate_performance_metrics(self, returns: pd.Series, equity_curve: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Basic performance
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Annualized metrics
            periods_per_year = 252
            annualized_return = (1 + mean_return) ** periods_per_year - 1
            annualized_volatility = std_return * np.sqrt(periods_per_year)
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            daily_risk_free_rate = risk_free_rate / periods_per_year
            excess_returns = returns - daily_risk_free_rate
            sharpe_ratio = excess_returns.mean() / std_return if std_return > 0 else 0
            sharpe_ratio *= np.sqrt(periods_per_year)  # Annualize
            
            # Sortino ratio
            downside_returns = returns[returns < daily_risk_free_rate]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
            sortino_ratio *= np.sqrt(periods_per_year)  # Annualize
            
            # Calmar ratio
            if equity_curve is not None:
                peak = equity_curve.cummax()
                drawdown = (equity_curve - peak) / peak
                max_drawdown = abs(drawdown.min())
            else:
                cumulative = (1 + returns).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
                max_drawdown = abs(drawdown.min())
            
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Information ratio (if benchmark provided)
            information_ratio = 0  # Will be calculated in benchmark metrics
            
            # Win rate
            win_rate = (returns > 0).mean()
            
            # Profit factor
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            gross_profit = positive_returns.sum() if len(positive_returns) > 0 else 0
            gross_loss = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    def _calculate_drawdown_metrics(self, returns: pd.Series, equity_curve: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        try:
            if equity_curve is not None:
                peak = equity_curve.cummax()
                drawdown = (equity_curve - peak) / peak
            else:
                cumulative = (1 + returns).cumprod()
                peak = cumulative.cummax()
                drawdown = (cumulative - peak) / peak
            
            # Maximum drawdown
            max_drawdown = abs(drawdown.min())
            
            # Average drawdown
            drawdown_periods = drawdown[drawdown < 0]
            avg_drawdown = abs(drawdown_periods.mean()) if len(drawdown_periods) > 0 else 0
            
            # Drawdown duration
            drawdown_start = None
            drawdown_durations = []
            current_duration = 0
            
            for i, dd in enumerate(drawdown):
                if dd < 0:
                    if drawdown_start is None:
                        drawdown_start = i
                    current_duration += 1
                else:
                    if drawdown_start is not None:
                        drawdown_durations.append(current_duration)
                        drawdown_start = None
                        current_duration = 0
            
            # Add final drawdown if it extends to the end
            if drawdown_start is not None:
                drawdown_durations.append(current_duration)
            
            max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0
            avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0
            
            # Recovery time (time to reach new high after drawdown)
            recovery_times = []
            for i, dd in enumerate(drawdown):
                if dd == 0 and i > 0 and drawdown.iloc[i-1] < 0:
                    # Found end of drawdown, calculate recovery time
                    drawdown_start_idx = None
                    for j in range(i-1, -1, -1):
                        if drawdown.iloc[j] < 0:
                            drawdown_start_idx = j
                        else:
                            break
                    
                    if drawdown_start_idx is not None:
                        recovery_time = i - drawdown_start_idx
                        recovery_times.append(recovery_time)
            
            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
            max_recovery_time = max(recovery_times) if recovery_times else 0
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'avg_drawdown_duration': avg_drawdown_duration,
                'max_recovery_time': max_recovery_time,
                'avg_recovery_time': avg_recovery_time,
                'drawdown_count': len(drawdown_durations)
            }
            
        except Exception as e:
            self.logger.error(f"Drawdown metrics calculation failed: {e}")
            return {}
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        try:
            # Align returns and benchmark
            aligned_data = pd.DataFrame({
                'returns': returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) == 0:
                return {}
            
            returns_aligned = aligned_data['returns']
            benchmark_aligned = aligned_data['benchmark']
            
            # Excess returns
            excess_returns = returns_aligned - benchmark_aligned
            
            # Tracking error
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # Information ratio
            information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            information_ratio *= np.sqrt(252)  # Annualize
            
            # Beta
            covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha
            periods_per_year = 252
            risk_free_rate = 0.02 / periods_per_year
            alpha = returns_aligned.mean() - (risk_free_rate + beta * (benchmark_aligned.mean() - risk_free_rate))
            alpha *= periods_per_year  # Annualize
            
            # Correlation
            correlation = returns_aligned.corr(benchmark_aligned)
            
            # R-squared
            r_squared = correlation ** 2
            
            # Relative performance
            total_return = (1 + returns_aligned).prod() - 1
            benchmark_total_return = (1 + benchmark_aligned).prod() - 1
            relative_return = total_return - benchmark_total_return
            
            return {
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'correlation': correlation,
                'r_squared': r_squared,
                'relative_return': relative_return,
                'total_return': total_return,
                'benchmark_total_return': benchmark_total_return
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark metrics calculation failed: {e}")
            return {}
    
    def calculate_rolling_metrics(self, 
                                 returns: pd.Series,
                                 window: int = 252,
                                 metrics: List[str] = None) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        try:
            if metrics is None:
                metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
            
            rolling_data = {}
            
            for metric in metrics:
                if metric == 'sharpe_ratio':
                    rolling_data[metric] = returns.rolling(window).apply(
                        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
                    )
                
                elif metric == 'sortino_ratio':
                    risk_free_rate = 0.02 / 252
                    rolling_data[metric] = returns.rolling(window).apply(
                        lambda x: (x.mean() - risk_free_rate) / x[x < risk_free_rate].std() * np.sqrt(252)
                        if len(x[x < risk_free_rate]) > 0 and x[x < risk_free_rate].std() > 0 else 0
                    )
                
                elif metric == 'max_drawdown':
                    rolling_data[metric] = returns.rolling(window).apply(
                        lambda x: abs(((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())
                    )
                
                elif metric == 'win_rate':
                    rolling_data[metric] = returns.rolling(window).apply(lambda x: (x > 0).mean())
                
                elif metric == 'volatility':
                    rolling_data[metric] = returns.rolling(window).std() * np.sqrt(252)
                
                elif metric == 'return':
                    rolling_data[metric] = returns.rolling(window).apply(
                        lambda x: (1 + x).prod() - 1
                    )
            
            return pd.DataFrame(rolling_data)
            
        except Exception as e:
            self.logger.error(f"Rolling metrics calculation failed: {e}")
            return pd.DataFrame()
    
    def calculate_regime_metrics(self, 
                                returns: pd.Series,
                                regime_labels: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for different market regimes."""
        try:
            regime_metrics = {}
            
            for regime in regime_labels.unique():
                regime_returns = returns[regime_labels == regime]
                
                if len(regime_returns) == 0:
                    continue
                
                # Calculate basic metrics for this regime
                regime_result = self.calculate_comprehensive_metrics(regime_returns)
                
                regime_metrics[f'regime_{regime}'] = {
                    'data_points': len(regime_returns),
                    'total_return': regime_result.metrics['basic_metrics']['total_return'],
                    'sharpe_ratio': regime_result.metrics['performance_metrics']['sharpe_ratio'],
                    'max_drawdown': regime_result.metrics['risk_metrics']['max_drawdown'],
                    'win_rate': regime_result.metrics['performance_metrics']['win_rate'],
                    'volatility': regime_result.metrics['basic_metrics']['annualized_volatility']
                }
            
            return regime_metrics
            
        except Exception as e:
            self.logger.error(f"Regime metrics calculation failed: {e}")
            return {}
    
    def generate_metrics_report(self, metrics_result: MetricsResult) -> Dict[str, Any]:
        """Generate a comprehensive metrics report."""
        try:
            report = {
                'summary': {
                    'data_points': metrics_result.data_points,
                    'execution_time': metrics_result.execution_time,
                    'calculation_timestamp': pd.Timestamp.now().isoformat()
                },
                'key_metrics': {
                    'total_return': metrics_result.metrics['basic_metrics'].get('total_return', 0),
                    'sharpe_ratio': metrics_result.metrics['performance_metrics'].get('sharpe_ratio', 0),
                    'max_drawdown': metrics_result.metrics['risk_metrics'].get('max_drawdown', 0),
                    'win_rate': metrics_result.metrics['performance_metrics'].get('win_rate', 0),
                    'volatility': metrics_result.metrics['basic_metrics'].get('annualized_volatility', 0)
                },
                'risk_assessment': self._assess_risk_level(metrics_result.metrics),
                'performance_grade': self._grade_performance(metrics_result.metrics),
                'recommendations': self._generate_recommendations(metrics_result.metrics)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Metrics report generation failed: {e}")
            return {'error': str(e)}
    
    def _assess_risk_level(self, metrics: Dict[str, Any]) -> str:
        """Assess risk level based on metrics."""
        try:
            volatility = metrics['basic_metrics'].get('annualized_volatility', 0)
            max_drawdown = abs(metrics['risk_metrics'].get('max_drawdown', 0))
            
            if volatility < 0.1 and max_drawdown < 0.05:
                return 'Low'
            elif volatility < 0.2 and max_drawdown < 0.15:
                return 'Medium'
            elif volatility < 0.3 and max_drawdown < 0.25:
                return 'High'
            else:
                return 'Very High'
                
        except Exception:
            return 'Unknown'
    
    def _grade_performance(self, metrics: Dict[str, Any]) -> str:
        """Grade performance based on metrics."""
        try:
            sharpe_ratio = metrics['performance_metrics'].get('sharpe_ratio', 0)
            calmar_ratio = metrics['performance_metrics'].get('calmar_ratio', 0)
            
            if sharpe_ratio > 2 and calmar_ratio > 1:
                return 'A'
            elif sharpe_ratio > 1.5 and calmar_ratio > 0.5:
                return 'B'
            elif sharpe_ratio > 1 and calmar_ratio > 0:
                return 'C'
            elif sharpe_ratio > 0:
                return 'D'
            else:
                return 'F'
                
        except Exception:
            return 'Unknown'
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        try:
            sharpe_ratio = metrics['performance_metrics'].get('sharpe_ratio', 0)
            max_drawdown = abs(metrics['risk_metrics'].get('max_drawdown', 0))
            win_rate = metrics['performance_metrics'].get('win_rate', 0)
            
            if sharpe_ratio < 0.5:
                recommendations.append("Consider improving risk-adjusted returns through better position sizing or strategy refinement")
            
            if max_drawdown > 0.2:
                recommendations.append("High maximum drawdown detected - consider implementing stop-loss or position sizing limits")
            
            if win_rate < 0.4:
                recommendations.append("Low win rate suggests strategy may need optimization for better entry/exit signals")
            
            if sharpe_ratio > 2 and max_drawdown < 0.1:
                recommendations.append("Excellent performance - consider scaling up position sizes gradually")
            
        except Exception:
            recommendations.append("Unable to generate recommendations due to missing metrics")
        
        return recommendations