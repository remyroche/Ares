from typing import Dict, List, Optional, Union, Any, Tuple
from ..utils.logger import system_logger
from .core.decorators import handles_errors
"""
ML-Optimized Barriers with Multi-Objective Optimization
Integrates with existing HMM regime logic and barrier optimization
"""
from dataclasses import dataclass
from scipy.optimize import minimize
from ..utils.logger import system_logger
import pandas as pd
from typing import Any
from typing import Dict
from typing import Optional
from datetime import datetime
import numpy as np

@dataclass
class BarrierOptimizationResult:
    """Result of barrier optimization for a specific regime"""
    regime: str
    optimal_barriers: Dict[str, float]
    optimization_metrics: Dict[str, float]
    optimization_success: bool
    timestamp: datetime
    objective_value: float
    iterations: int

class MLOptimizedBarriers:
    """
    ML-Optimized Barriers with Multi-Objective Optimization.
    Optimizes barriers per HMM regime using PnL 50% + Win Rate 25% + Sharpe 25%.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize ML-Optimized Barriers system.
        
        Args:
            config: Configuration dictionary with optimization settings
        """
        self.config = config
        self.logger = system_logger.getChild('MLOptimizedBarriers')
        self.optimization_config = config.get('barrier_optimization', {})
        self.regime_names = [f'regime_{i:02d}' for i in range(20)]
        self.trading_fee = 0.0008
        self.objective_weights = {'pnl': 0.5, 'win_rate': 0.25, 'sharpe_ratio': 0.25}
        self.optimization_bounds = {'profit_take_multiplier': (0.0013, 0.0058), 'stop_loss_multiplier': (0.001, 0.0038), 'confidence_threshold': (0.4, 0.8)}
        self.optimized_barriers: Dict[str, Dict[str, float]] = {}
        self.optimization_history: Dict[str, List[BarrierOptimizationResult]] = {}
        self.regime_data: Dict[str, pd.DataFrame] = {}
        self.hmm_regime_predictor = None
        self.trade_executor = None

    @handles_errors(exceptions=(ValueError, AttributeError), default_return = False, context='ML barrier optimization initialization')
    async def initialize(self) -> bool:
        """
        Initialize the ML-Optimized Barriers system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info('Initializing ML-Optimized Barriers system...')
            for regime in self.regime_names:
                self.optimization_history[regime] = []
                self.regime_data[regime] = pd.DataFrame()
            await self._load_existing_barriers()
            self.logger.info('✅ ML-Optimized Barriers system initialized successfully')
            return True
        except Exception as e:
            self.logger.error(f'❌ ML-Optimized Barriers initialization failed: {e}')
            return False

    async def _load_existing_barriers(self) -> None:
        """Load existing optimized barriers from storage"""
        try:
            for regime in self.regime_names:
                self.optimized_barriers[regime] = {'profit_take_multiplier': 0.0028, 'stop_loss_multiplier': 0.0018, 'confidence_threshold': 0.6, 'optimization_status': 'default'}
        except Exception as e:
            self.logger.warning(f'Could not load existing barriers: {e}')

    @handles_errors(exceptions=(ValueError, KeyError), default_return = None, context='regime barrier optimization')
    async def optimize_regime_barriers(self, regime: str, historical_data: pd.DataFrame, min_trades: int = 100) -> Optional[BarrierOptimizationResult]:
        """
        Optimize barriers for a specific HMM regime using multi-objective optimization.
        
        Args:
            regime: HMM regime name
            historical_data: Historical trading data for the regime
            min_trades: Minimum number of trades required for optimization
            
        Returns:
            BarrierOptimizationResult: Optimization result
        """
        try:
            if regime not in self.regime_names:
                self.logger.error(f'Invalid regime: {regime}')
                return None
            if len(historical_data) < min_trades:
                self.logger.warning(f'Insufficient data for regime {regime}: {len(historical_data)} < {min_trades}')
                return None
            self.logger.info(f'Starting barrier optimization for regime {regime} with {len(historical_data)} trades')
            optimization_data = self._prepare_optimization_data(historical_data)
            optimization_result = await self._run_multi_objective_optimization(optimization_data, regime)
            if optimization_result['optimization_success']:
                self.optimized_barriers[regime] = optimization_result['optimal_barriers']
                result = BarrierOptimizationResult(regime = regime, optimal_barriers = optimization_result['optimal_barriers'], optimization_metrics = optimization_result['metrics'], optimization_success = True, timestamp = datetime.now(), objective_value = optimization_result['objective_value'], iterations = optimization_result['iterations'])
                self.optimization_history[regime].append(result)
                self.logger.info(f'✅ Successfully optimized barriers for regime {regime}')
                return result
            else:
                self.logger.error(f'❌ Optimization failed for regime {regime}')
                return None
        except Exception as e:
            self.logger.error(f'Error optimizing barriers for regime {regime}: {e}')
            return None

    def _prepare_optimization_data(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for optimization"""
        optimization_data = {'trades': historical_data, 'regime_characteristics': self._extract_regime_characteristics(historical_data), 'market_conditions': self._extract_market_conditions(historical_data)}
        return optimization_data

    def _extract_regime_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract characteristics of the regime"""
        if 'confidence' in data.columns:
            confidence_stats = {'mean_confidence': data['confidence'].mean(), 'std_confidence': data['confidence'].std(), 'min_confidence': data['confidence'].min(), 'max_confidence': data['confidence'].max()}
        else:
            confidence_stats = {'mean_confidence': 0.6, 'std_confidence': 0.1, 'min_confidence': 0.4, 'max_confidence': 0.8}
        if 'pnl' in data.columns:
            pnl_stats = {'mean_pnl': data['pnl'].mean(), 'std_pnl': data['pnl'].std(), 'win_rate': (data['pnl'] > 0).mean(), 'sharpe_ratio': data['pnl'].mean() / data['pnl'].std() if data['pnl'].std() > 0 else 0}
        else:
            pnl_stats = {'mean_pnl': 0.0, 'std_pnl': 0.01, 'win_rate': 0.5, 'sharpe_ratio': 0.0}
        return {**confidence_stats, **pnl_stats}

    def _extract_market_conditions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract market conditions from data"""
        return {'volatility': 0.02, 'trend_strength': 0.0, 'market_regime': 'normal'}

    async def _run_multi_objective_optimization(self, optimization_data: Dict[str, Any], regime: str) -> Dict[str, Any]:
        """Run multi-objective optimization for barrier parameters"""

        def objective_function(barriers: List[Any]) -> None:
            """Multi-objective function: PnL 50% + Win Rate 25% + Sharpe 25%"""
            profit_take = barriers[0]
            stop_loss = barriers[1]
            confidence_threshold = barriers[2]
            metrics = self._calculate_trading_metrics(optimization_data, {'profit_take_multiplier': profit_take, 'stop_loss_multiplier': stop_loss, 'confidence_threshold': confidence_threshold})
            normalized_pnl = self._normalize_pnl(metrics['total_pnl'])
            normalized_win_rate = metrics['win_rate']
            normalized_sharpe = self._normalize_sharpe(metrics['sharpe_ratio'])
            weighted_objective = -(self.objective_weights['pnl'] * normalized_pnl + self.objective_weights['win_rate'] * normalized_win_rate + self.objective_weights['sharpe_ratio'] * normalized_sharpe)
            return weighted_objective
        initial_guess = [(self.optimization_bounds['profit_take_multiplier'][0] + self.optimization_bounds['profit_take_multiplier'][1]) / 2, (self.optimization_bounds['stop_loss_multiplier'][0] + self.optimization_bounds['stop_loss_multiplier'][1]) / 2, (self.optimization_bounds['confidence_threshold'][0] + self.optimization_bounds['confidence_threshold'][1]) / 2]
        constraints = [{'type': 'ineq', 'fun': lambda x: x[0] - x[1]}, {'type': 'ineq', 'fun': lambda x: x[0] - 2 * self.trading_fee}, {'type': 'ineq', 'fun': lambda x: x[1] - 1.5 * self.trading_fee}, {'type': 'ineq', 'fun': lambda x: x[2] - 0.3}, {'type': 'ineq', 'fun': lambda x: 0.9 - x[2]}]
        result = minimize(objective_function, initial_guess, method='SLSQP', bounds = list(self.optimization_bounds.values()), constraints = constraints, options={'maxiter': 1000, 'ftol': 1e-06})
        if result.success:
            optimal_barriers = {'profit_take_multiplier': result.x[0], 'stop_loss_multiplier': result.x[1], 'confidence_threshold': result.x[2], 'optimization_status': 'optimized', 'optimization_timestamp': datetime.now()}
            final_metrics = self._calculate_trading_metrics(optimization_data, optimal_barriers)
            return {'optimization_success': True, 'optimal_barriers': optimal_barriers, 'objective_value': -result.fun, 'iterations': result.nit, 'metrics': final_metrics}
        else:
            fallback_barriers = self.optimized_barriers.get(regime, {'profit_take_multiplier': 0.0028, 'stop_loss_multiplier': 0.0018, 'confidence_threshold': 0.6})
            return {'optimization_success': False, 'optimal_barriers': fallback_barriers, 'objective_value': float('inf'), 'iterations': result.nit, 'metrics': {'error': 'Optimization failed'}}

    def _calculate_trading_metrics(self, optimization_data: Dict[str, Any], barriers: Dict[str, float]) -> Dict[str, float]:
        """Calculate trading metrics for given barrier configuration"""
        trades = self._simulate_trades_with_barriers(optimization_data, barriers)
        if not trades:
            return {'total_pnl': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_trades': 0}
        profits = [trade['pnl'] for trade in trades]
        total_pnl = sum(profits)
        win_rate = sum((1 for p in profits if p > 0)) / len(profits)
        if len(profits) > 1 and np.std(profits) > 0:
            sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252 * 24 * 4)
        else:
            sharpe_ratio = 0
        cumulative_pnl = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        return {'total_pnl': total_pnl, 'win_rate': win_rate, 'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'total_trades': len(trades), 'avg_pnl': np.mean(profits), 'std_pnl': np.std(profits)}

    def _simulate_trades_with_barriers(self, optimization_data: Dict[str, Any], barriers: Dict[str, float]) -> List[Dict[str, Any]]:
        """Simulate trades with given barrier configuration"""
        trades = []
        historical_trades = optimization_data['trades']
        for _, row in historical_trades.iterrows():
            if self._should_enter_trade(row, barriers):
                trade_result = self._simulate_trade_outcome(row, barriers)
                trades.append(trade_result)
        return trades

    def _should_enter_trade(self, row: pd.Series, barriers: Dict[str, float]) -> bool:
        """Determine if trade should be entered based on confidence threshold"""
        confidence = row.get('confidence', 0.6)
        return confidence >= barriers['confidence_threshold']

    def _simulate_trade_outcome(self, row: pd.Series, barriers: Dict[str, float]) -> Dict[str, Any]:
        """Simulate individual trade outcome"""
        profit_take = barriers['profit_take_multiplier']
        stop_loss = barriers['stop_loss_multiplier']
        if 'price_movement' in row:
            price_movement = row['price_movement']
        else:
            regime_volatility = 0.002
            price_movement = np.random.normal(0, regime_volatility)
        if price_movement >= profit_take:
            pnl = (profit_take - self.trading_fee) * 100
            outcome = 'profit_take'
        elif price_movement <= -stop_loss:
            pnl = (-stop_loss - self.trading_fee) * 100
            outcome = 'stop_loss'
        else:
            pnl = (price_movement - self.trading_fee) * 100
            outcome = 'partial'
        return {'pnl': pnl, 'price_movement': price_movement, 'outcome': outcome, 'barriers_used': barriers}

    def _normalize_pnl(self, pnl: float) -> float:
        """Normalize PnL to [0, 1] range using sigmoid function"""
        return 1 / (1 + np.exp(-pnl / 1000))

    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to [0, 1] range"""
        return np.clip((sharpe + 2) / 6, 0, 1)

    @handles_errors(exceptions=(ValueError, KeyError), default_return = None, context='get optimized barriers')
    def get_optimized_barriers(self, regime: str) -> Optional[Dict[str, float]]:
        """
        Get optimized barriers for a specific regime.
        
        Args:
            regime: HMM regime name
            
        Returns:
            Dict: Optimized barrier configuration
        """
        try:
            if regime not in self.optimized_barriers:
                self.logger.warning(f'No optimized barriers found for regime {regime}')
                return None
            return self.optimized_barriers[regime].copy()
        except Exception as e:
            self.logger.error(f'Error getting optimized barriers for regime {regime}: {e}')
            return None

    @handles_errors(exceptions=(ValueError, KeyError), default_return = None, context='update optimization weights')
    def update_optimization_weights(self, new_weights: Dict[str, float]) -> bool:
        """
        Update objective function weights.
        
        Args:
            new_weights: New weight configuration
            
        Returns:
            bool: True if update successful
        """
        try:
            if abs(sum(new_weights.values()) - 1.0) > 1e-06:
                raise ValueError('Weights must sum to 1.0')
            required_keys = ['pnl', 'win_rate', 'sharpe_ratio']
            if not all((key in new_weights for key in required_keys)):
                raise ValueError(f'Missing required weight keys: {required_keys}')
            self.objective_weights = new_weights.copy()
            self.logger.info(f'Updated optimization weights: {new_weights}')
            return True
        except Exception as e:
            self.logger.error(f'Error updating optimization weights: {e}')
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all regime optimizations"""
        summary = {'total_regimes': len(self.regime_names), 'optimized_regimes': len(self.optimized_barriers), 'optimization_weights': self.objective_weights, 'regime_details': {}}
        for regime in self.regime_names:
            if regime in self.optimized_barriers:
                barriers = self.optimized_barriers[regime]
                history = self.optimization_history.get(regime, [])
                summary['regime_details'][regime] = {'optimized': True, 'barriers': barriers, 'optimization_count': len(history), 'last_optimized': history[-1].timestamp if history else None, 'optimization_status': barriers.get('optimization_status', 'unknown')}
            else:
                summary['regime_details'][regime] = {'optimized': False, 'barriers': None, 'optimization_count': 0, 'last_optimized': None, 'optimization_status': 'not_optimized'}
        return summary

    async def optimize_all_regimes(self, regime_data: Dict[str, pd.DataFrame], min_trades: int = 100) -> Dict[str, BarrierOptimizationResult]:
        """
        Optimize barriers for all regimes.
        
        Args:
            regime_data: Historical data for each regime
            min_trades: Minimum trades required per regime
            
        Returns:
            Dict: Optimization results for each regime
        """
        results = {}
        for regime in self.regime_names:
            if regime in regime_data and len(regime_data[regime]) >= min_trades:
                self.logger.info(f'Optimizing barriers for regime {regime}')
                result = await self.optimize_regime_barriers(regime, regime_data[regime], min_trades)
                results[regime] = result
            else:
                self.logger.warning(f'Skipping regime {regime}: insufficient data')
                results[regime] = None
        return results