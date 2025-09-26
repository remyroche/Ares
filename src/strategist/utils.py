from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import logging

"""Strategist utilities."""

class PerformanceOptimizer:
    """Performance optimizer for strategy components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance optimizer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def optimize_strategy_performance(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy performance metrics."""
        try:
            # Extract performance metrics
            metrics = self._extract_performance_metrics(strategy_data)
            
            # Apply optimization rules
            optimized_metrics = self._apply_optimization_rules(metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(optimized_metrics)
            
            return {
                'original_metrics': metrics,
                'optimized_metrics': optimized_metrics,
                'recommendations': recommendations,
                'optimization_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
            return {'error': str(e)}
    
    def _extract_performance_metrics(self, strategy_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from strategy data."""
        metrics = {}
        
        # Extract common metrics
        if 'returns' in strategy_data:
            returns = strategy_data['returns']
            if isinstance(returns, pd.Series):
                metrics['total_return'] = float(returns.sum())
                metrics['sharpe_ratio'] = float(returns.mean() / returns.std() if returns.std() > 0 else 0)
                metrics['max_drawdown'] = float((returns.cumsum() - returns.cumsum().expanding().max()).min())
        
        if 'trades' in strategy_data:
            trades = strategy_data['trades']
            if isinstance(trades, pd.DataFrame) and len(trades) > 0:
                metrics['total_trades'] = len(trades)
                if 'pnl' in trades.columns:
                    winning_trades = trades[trades['pnl'] > 0]
                    metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0
                    metrics['avg_win'] = float(winning_trades['pnl'].mean()) if len(winning_trades) > 0 else 0
        
        return metrics
    
    def _apply_optimization_rules(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Apply optimization rules to metrics."""
        optimized = metrics.copy()
        
        # Risk-adjusted returns
        if 'sharpe_ratio' in optimized and 'total_return' in optimized:
            if optimized['sharpe_ratio'] < 1.0:
                optimized['risk_adjusted_return'] = optimized['total_return'] * 0.8
            else:
                optimized['risk_adjusted_return'] = optimized['total_return'] * 1.2
        
        # Drawdown optimization
        if 'max_drawdown' in optimized:
            if optimized['max_drawdown'] < -0.1:  # More than 10% drawdown
                optimized['drawdown_penalty'] = 0.5
            else:
                optimized['drawdown_penalty'] = 1.0
        
        return optimized
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if metrics.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("Consider improving risk-adjusted returns")
        
        if metrics.get('max_drawdown', 0) < -0.1:
            recommendations.append("Reduce maximum drawdown through better risk management")
        
        if metrics.get('win_rate', 0) < 0.4:
            recommendations.append("Improve trade selection to increase win rate")
        
        if not recommendations:
            recommendations.append("Strategy performance is within acceptable parameters")
        
        return recommendations

class StrategyComponentExtractor:
    """Strategy component extractor for analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize strategy component extractor."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def extract_components(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract strategy components for analysis."""
        try:
            components = {}
            
            # Extract entry signals
            if 'entry_signals' in strategy_data:
                components['entry_signals'] = self._extract_entry_signals(strategy_data['entry_signals'])
            
            # Extract exit signals
            if 'exit_signals' in strategy_data:
                components['exit_signals'] = self._extract_exit_signals(strategy_data['exit_signals'])
            
            # Extract position sizing
            if 'position_sizes' in strategy_data:
                components['position_sizing'] = self._extract_position_sizing(strategy_data['position_sizes'])
            
            # Extract risk management
            if 'risk_management' in strategy_data:
                components['risk_management'] = self._extract_risk_management(strategy_data['risk_management'])
            
            return components
            
        except Exception as e:
            self.logger.error(f"Component extraction failed: {e}")
            return {'error': str(e)}
    
    def _extract_entry_signals(self, entry_data: Any) -> Dict[str, Any]:
        """Extract entry signal components."""
        if isinstance(entry_data, pd.DataFrame):
            return {
                'signal_count': len(entry_data),
                'signal_frequency': len(entry_data) / 1000 if len(entry_data) > 0 else 0,  # per 1000 periods
                'signal_types': entry_data.columns.tolist() if hasattr(entry_data, 'columns') else []
            }
        return {'signal_count': 0}
    
    def _extract_exit_signals(self, exit_data: Any) -> Dict[str, Any]:
        """Extract exit signal components."""
        if isinstance(exit_data, pd.DataFrame):
            return {
                'exit_count': len(exit_data),
                'exit_types': exit_data.columns.tolist() if hasattr(exit_data, 'columns') else []
            }
        return {'exit_count': 0}
    
    def _extract_position_sizing(self, sizing_data: Any) -> Dict[str, Any]:
        """Extract position sizing components."""
        if isinstance(sizing_data, pd.Series):
            return {
                'avg_position_size': float(sizing_data.mean()),
                'max_position_size': float(sizing_data.max()),
                'min_position_size': float(sizing_data.min()),
                'position_size_std': float(sizing_data.std())
            }
        return {'avg_position_size': 0}
    
    def _extract_risk_management(self, risk_data: Any) -> Dict[str, Any]:
        """Extract risk management components."""
        if isinstance(risk_data, dict):
            return {
                'stop_loss_enabled': risk_data.get('stop_loss', False),
                'take_profit_enabled': risk_data.get('take_profit', False),
                'max_position_size': risk_data.get('max_position_size', 1.0),
                'risk_per_trade': risk_data.get('risk_per_trade', 0.02)
            }
        return {'stop_loss_enabled': False}

class ValidationError(Exception):
    """Validation error for strategy data."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value
        self.timestamp = pd.Timestamp.now()

def validate_data_sufficiency(data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
    """Validate data sufficiency for strategy analysis."""
    try:
        if isinstance(data, pd.DataFrame):
            # Check minimum data requirements
            if len(data) < 100:
                raise ValidationError("Insufficient data: need at least 100 rows", "row_count", len(data))
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}", "columns", data.columns.tolist())
            
            # Check for data quality
            if data.isnull().sum().sum() > len(data) * 0.1:  # More than 10% missing data
                raise ValidationError("Too much missing data", "missing_data_ratio", data.isnull().sum().sum() / len(data))
        
        elif isinstance(data, dict):
            # Check for required keys
            required_keys = ['returns', 'trades']
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValidationError(f"Missing required keys: {missing_keys}", "keys", list(data.keys()))
        
        return True
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Data validation failed: {e}")

def validate_strategy_parameters(params: Dict[str, Any]) -> bool:
    """Validate strategy parameters."""
    try:
        # Check for required parameters
        required_params = ['symbol', 'timeframe', 'initial_capital']
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise ValidationError(f"Missing required parameters: {missing_params}", "parameters", list(params.keys()))
        
        # Validate parameter values
        if params.get('initial_capital', 0) <= 0:
            raise ValidationError("Initial capital must be positive", "initial_capital", params.get('initial_capital'))
        
        if params.get('risk_per_trade', 0) <= 0 or params.get('risk_per_trade', 0) > 0.1:
            raise ValidationError("Risk per trade must be between 0 and 0.1", "risk_per_trade", params.get('risk_per_trade'))
        
        return True
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Parameter validation failed: {e}")

def calculate_strategy_metrics(returns: pd.Series, trades: pd.DataFrame = None) -> Dict[str, float]:
    """Calculate comprehensive strategy metrics."""
    try:
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = float(returns.sum())
        metrics['annualized_return'] = float(returns.mean() * 252)  # Assuming daily data
        metrics['volatility'] = float(returns.std() * np.sqrt(252))
        metrics['sharpe_ratio'] = float(metrics['annualized_return'] / metrics['volatility']) if metrics['volatility'] > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = float(drawdown.min())
        metrics['avg_drawdown'] = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0
        
        # Trade metrics (if available)
        if trades is not None and len(trades) > 0:
            if 'pnl' in trades.columns:
                winning_trades = trades[trades['pnl'] > 0]
                losing_trades = trades[trades['pnl'] < 0]
                
                metrics['total_trades'] = len(trades)
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(losing_trades)
                metrics['win_rate'] = len(winning_trades) / len(trades) if len(trades) > 0 else 0
                metrics['avg_win'] = float(winning_trades['pnl'].mean()) if len(winning_trades) > 0 else 0
                metrics['avg_loss'] = float(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
                metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
        
        return metrics
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Strategy metrics calculation failed: {e}")
        return {'error': str(e)}