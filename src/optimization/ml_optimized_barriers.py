"""
ML-Optimized Barriers with Multi-Objective Optimization
Integrates with existing HMM regime logic and barrier optimization
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import stats

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.core.decorators.errors import handles_errors


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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML-Optimized Barriers system.
        
        Args:
            config: Configuration dictionary with optimization settings
        """
        self.config = config
        self.logger = system_logger.getChild('MLOptimizedBarriers')
        
        # Optimization Configuration
        self.optimization_config = config.get('barrier_optimization', {})
        self.regime_names = [f"regime_{i:02d}" for i in range(20)]  # regime_00 to regime_19
        self.trading_fee = 0.0008  # 0.08% trading fee
        
        # Multi-Objective Weights (as specified)
        self.objective_weights = {
            'pnl': 0.50,        # 50% weight on PnL
            'win_rate': 0.25,   # 25% weight on win rate
            'sharpe_ratio': 0.25  # 25% weight on Sharpe ratio
        }
        
        # Optimization Bounds (accounting for 0.08% trading fee)
        self.optimization_bounds = {
            'profit_take_multiplier': (0.0013, 0.0058),  # 0.13% to 0.58% (0.05%+fee to 0.5%+fee)
            'stop_loss_multiplier': (0.0010, 0.0038),    # 0.10% to 0.38% (0.02%+fee to 0.3%+fee)
            'confidence_threshold': (0.4, 0.8)           # 40% to 80%
        }
        
        # Storage
        self.optimized_barriers: Dict[str, Dict[str, float]] = {}
        self.optimization_history: Dict[str, List[BarrierOptimizationResult]] = {}
        self.regime_data: Dict[str, pd.DataFrame] = {}
        
        # Integration with existing systems
        self.hmm_regime_predictor = None  # Will be injected
        self.trade_executor = None  # Will be injected
        
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='ML barrier optimization initialization')
    async def initialize(self) -> bool:
        """
        Initialize the ML-Optimized Barriers system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing ML-Optimized Barriers system...")
            
            # Initialize storage for each regime
            for regime in self.regime_names:
                self.optimization_history[regime] = []
                self.regime_data[regime] = pd.DataFrame()
                
            # Load existing optimized barriers if available
            await self._load_existing_barriers()
            
            self.logger.info("✅ ML-Optimized Barriers system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ML-Optimized Barriers initialization failed: {e}")
            return False
    
    async def _load_existing_barriers(self) -> None:
        """Load existing optimized barriers from storage"""
        try:
            # This would load from your existing barrier storage system
            # For now, initialize with default values
            for regime in self.regime_names:
                self.optimized_barriers[regime] = {
                    'profit_take_multiplier': 0.0028,  # Default 0.28% (0.2% + 0.08% fee)
                    'stop_loss_multiplier': 0.0018,    # Default 0.18% (0.1% + 0.08% fee)
                    'confidence_threshold': 0.6,       # Default 60%
                    'optimization_status': 'default'
                }
        except Exception as e:
            self.logger.warning(f"Could not load existing barriers: {e}")
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='regime barrier optimization')
    async def optimize_regime_barriers(
        self,
        regime: str,
        historical_data: pd.DataFrame,
        min_trades: int = 100
    ) -> Optional[BarrierOptimizationResult]:
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
                self.logger.error(f"Invalid regime: {regime}")
                return None
            
            # Validate data
            if len(historical_data) < min_trades:
                self.logger.warning(f"Insufficient data for regime {regime}: {len(historical_data)} < {min_trades}")
                return None
            
            self.logger.info(f"Starting barrier optimization for regime {regime} with {len(historical_data)} trades")
            
            # Prepare optimization data
            optimization_data = self._prepare_optimization_data(historical_data)
            
            # Run multi-objective optimization
            optimization_result = await self._run_multi_objective_optimization(
                optimization_data, regime
            )
            
            if optimization_result['optimization_success']:
                # Store optimized barriers
                self.optimized_barriers[regime] = optimization_result['optimal_barriers']
                
                # Create result object
                result = BarrierOptimizationResult(
                    regime=regime,
                    optimal_barriers=optimization_result['optimal_barriers'],
                    optimization_metrics=optimization_result['metrics'],
                    optimization_success=True,
                    timestamp=datetime.now(),
                    objective_value=optimization_result['objective_value'],
                    iterations=optimization_result['iterations']
                )
                
                # Store in history
                self.optimization_history[regime].append(result)
                
                self.logger.info(f"✅ Successfully optimized barriers for regime {regime}")
                return result
            else:
                self.logger.error(f"❌ Optimization failed for regime {regime}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error optimizing barriers for regime {regime}: {e}")
            return None
    
    def _prepare_optimization_data(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for optimization"""
        
        # Extract relevant features for optimization
        optimization_data = {
            'trades': historical_data,
            'regime_characteristics': self._extract_regime_characteristics(historical_data),
            'market_conditions': self._extract_market_conditions(historical_data)
        }
        
        return optimization_data
    
    def _extract_regime_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract characteristics of the regime"""
        
        if 'confidence' in data.columns:
            confidence_stats = {
                'mean_confidence': data['confidence'].mean(),
                'std_confidence': data['confidence'].std(),
                'min_confidence': data['confidence'].min(),
                'max_confidence': data['confidence'].max()
            }
        else:
            confidence_stats = {
                'mean_confidence': 0.6,
                'std_confidence': 0.1,
                'min_confidence': 0.4,
                'max_confidence': 0.8
            }
        
        if 'pnl' in data.columns:
            pnl_stats = {
                'mean_pnl': data['pnl'].mean(),
                'std_pnl': data['pnl'].std(),
                'win_rate': (data['pnl'] > 0).mean(),
                'sharpe_ratio': data['pnl'].mean() / data['pnl'].std() if data['pnl'].std() > 0 else 0
            }
        else:
            pnl_stats = {
                'mean_pnl': 0.0,
                'std_pnl': 0.01,
                'win_rate': 0.5,
                'sharpe_ratio': 0.0
            }
        
        return {**confidence_stats, **pnl_stats}
    
    def _extract_market_conditions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract market conditions from data"""
        
        # This would extract market volatility, trend, etc.
        # For now, return default values
        return {
            'volatility': 0.02,  # 2% volatility
            'trend_strength': 0.0,  # Neutral trend
            'market_regime': 'normal'
        }
    
    async def _run_multi_objective_optimization(
        self,
        optimization_data: Dict[str, Any],
        regime: str
    ) -> Dict[str, Any]:
        """Run multi-objective optimization for barrier parameters"""
        
        def objective_function(barriers):
            """Multi-objective function: PnL 50% + Win Rate 25% + Sharpe 25%"""
            
            # Extract barrier parameters
            profit_take = barriers[0]
            stop_loss = barriers[1]
            confidence_threshold = barriers[2]
            
            # Calculate metrics using the barrier configuration
            metrics = self._calculate_trading_metrics(
                optimization_data, {
                    'profit_take_multiplier': profit_take,
                    'stop_loss_multiplier': stop_loss,
                    'confidence_threshold': confidence_threshold
                }
            )
            
            # Normalize metrics to [0, 1] range
            normalized_pnl = self._normalize_pnl(metrics['total_pnl'])
            normalized_win_rate = metrics['win_rate']  # Already 0-1
            normalized_sharpe = self._normalize_sharpe(metrics['sharpe_ratio'])
            
            # Weighted objective (minimize negative for maximization)
            weighted_objective = -(
                self.objective_weights['pnl'] * normalized_pnl +
                self.objective_weights['win_rate'] * normalized_win_rate +
                self.objective_weights['sharpe_ratio'] * normalized_sharpe
            )
            
            return weighted_objective
        
        # Initial guess (current barriers or middle of bounds)
        initial_guess = [
            (self.optimization_bounds['profit_take_multiplier'][0] + 
             self.optimization_bounds['profit_take_multiplier'][1]) / 2,
            (self.optimization_bounds['stop_loss_multiplier'][0] + 
             self.optimization_bounds['stop_loss_multiplier'][1]) / 2,
            (self.optimization_bounds['confidence_threshold'][0] + 
             self.optimization_bounds['confidence_threshold'][1]) / 2
        ]
        
        # Optimization constraints (accounting for trading fee)
        constraints = [
            # Profit take should be greater than stop loss (risk-reward ratio > 1)
            {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},
            # Profit take should be at least 2x trading fee (0.16%) to ensure profitability
            {'type': 'ineq', 'fun': lambda x: x[0] - 2 * self.trading_fee},
            # Stop loss should be at least 1.5x trading fee (0.12%) to limit losses
            {'type': 'ineq', 'fun': lambda x: x[1] - 1.5 * self.trading_fee},
            # Confidence threshold should be reasonable
            {'type': 'ineq', 'fun': lambda x: x[2] - 0.3},  # At least 30%
            {'type': 'ineq', 'fun': lambda x: 0.9 - x[2]}   # At most 90%
        ]
        
        # Run optimization
        result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            bounds=list(self.optimization_bounds.values()),
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            optimal_barriers = {
                'profit_take_multiplier': result.x[0],
                'stop_loss_multiplier': result.x[1],
                'confidence_threshold': result.x[2],
                'optimization_status': 'optimized',
                'optimization_timestamp': datetime.now()
            }
            
            # Calculate final metrics
            final_metrics = self._calculate_trading_metrics(optimization_data, optimal_barriers)
            
            return {
                'optimization_success': True,
                'optimal_barriers': optimal_barriers,
                'objective_value': -result.fun,
                'iterations': result.nit,
                'metrics': final_metrics
            }
        else:
            # Fallback to current barriers if optimization fails (accounting for trading fee)
            fallback_barriers = self.optimized_barriers.get(regime, {
                'profit_take_multiplier': 0.0028,  # 0.28% (0.2% + 0.08% fee)
                'stop_loss_multiplier': 0.0018,    # 0.18% (0.1% + 0.08% fee)
                'confidence_threshold': 0.6
            })
            
            return {
                'optimization_success': False,
                'optimal_barriers': fallback_barriers,
                'objective_value': float('inf'),
                'iterations': result.nit,
                'metrics': {'error': 'Optimization failed'}
            }
    
    def _calculate_trading_metrics(
        self,
        optimization_data: Dict[str, Any],
        barriers: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate trading metrics for given barrier configuration"""
        
        # Simulate trading with given barriers
        trades = self._simulate_trades_with_barriers(optimization_data, barriers)
        
        if not trades:
            return {
                'total_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_trades': 0
            }
        
        # Calculate metrics
        profits = [trade['pnl'] for trade in trades]
        total_pnl = sum(profits)
        win_rate = sum(1 for p in profits if p > 0) / len(profits)
        
        # Sharpe ratio calculation (annualized for high-frequency trading)
        if len(profits) > 1 and np.std(profits) > 0:
            # Annualize based on timeframe frequency (5m = 12 intervals/hour, 15m = 4 intervals/hour, 30m = 2 intervals/hour, 1h = 1 interval/hour)
            # Use 15m as baseline (4 intervals/hour) for consistency
            sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252 * 24 * 4)  # Annualized
        else:
            sharpe_ratio = 0
            
        # Maximum drawdown
        cumulative_pnl = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'avg_pnl': np.mean(profits),
            'std_pnl': np.std(profits)
        }
    
    def _simulate_trades_with_barriers(
        self,
        optimization_data: Dict[str, Any],
        barriers: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Simulate trades with given barrier configuration"""
        
        trades = []
        historical_trades = optimization_data['trades']
        
        for _, row in historical_trades.iterrows():
            # Check if trade conditions are met
            if self._should_enter_trade(row, barriers):
                # Simulate trade outcome
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
        
        # Use actual price movement if available, otherwise simulate
        if 'price_movement' in row:
            price_movement = row['price_movement']
        else:
            # Simulate price movement based on regime characteristics
            regime_volatility = 0.002  # Default 0.2% volatility
            price_movement = np.random.normal(0, regime_volatility)
        
        # Determine trade outcome (accounting for trading fee)
        if price_movement >= profit_take:
            pnl = (profit_take - self.trading_fee) * 100  # Subtract trading fee
            outcome = 'profit_take'
        elif price_movement <= -stop_loss:
            pnl = (-stop_loss - self.trading_fee) * 100  # Subtract trading fee
            outcome = 'stop_loss'
        else:
            pnl = (price_movement - self.trading_fee) * 100  # Subtract trading fee
            outcome = 'partial'
            
        return {
            'pnl': pnl,
            'price_movement': price_movement,
            'outcome': outcome,
            'barriers_used': barriers
        }
    
    def _normalize_pnl(self, pnl: float) -> float:
        """Normalize PnL to [0, 1] range using sigmoid function"""
        return 1 / (1 + np.exp(-pnl / 1000))  # Scale factor of 1000
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to [0, 1] range"""
        # Sharpe ratio typically ranges from -2 to 4
        return np.clip((sharpe + 2) / 6, 0, 1)
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='get optimized barriers')
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
                self.logger.warning(f"No optimized barriers found for regime {regime}")
                return None
            
            return self.optimized_barriers[regime].copy()
            
        except Exception as e:
            self.logger.error(f"Error getting optimized barriers for regime {regime}: {e}")
            return None
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='update optimization weights')
    def update_optimization_weights(
        self,
        new_weights: Dict[str, float]
    ) -> bool:
        """
        Update objective function weights.
        
        Args:
            new_weights: New weight configuration
            
        Returns:
            bool: True if update successful
        """
        try:
            # Validate weights sum to 1.0
            if abs(sum(new_weights.values()) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            
            # Validate required keys
            required_keys = ['pnl', 'win_rate', 'sharpe_ratio']
            if not all(key in new_weights for key in required_keys):
                raise ValueError(f"Missing required weight keys: {required_keys}")
            
            self.objective_weights = new_weights.copy()
            self.logger.info(f"Updated optimization weights: {new_weights}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating optimization weights: {e}")
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all regime optimizations"""
        
        summary = {
            'total_regimes': len(self.regime_names),
            'optimized_regimes': len(self.optimized_barriers),
            'optimization_weights': self.objective_weights,
            'regime_details': {}
        }
        
        for regime in self.regime_names:
            if regime in self.optimized_barriers:
                barriers = self.optimized_barriers[regime]
                history = self.optimization_history.get(regime, [])
                
                summary['regime_details'][regime] = {
                    'optimized': True,
                    'barriers': barriers,
                    'optimization_count': len(history),
                    'last_optimized': history[-1].timestamp if history else None,
                    'optimization_status': barriers.get('optimization_status', 'unknown')
                }
            else:
                summary['regime_details'][regime] = {
                    'optimized': False,
                    'barriers': None,
                    'optimization_count': 0,
                    'last_optimized': None,
                    'optimization_status': 'not_optimized'
                }
        
        return summary
    
    async def optimize_all_regimes(
        self,
        regime_data: Dict[str, pd.DataFrame],
        min_trades: int = 100
    ) -> Dict[str, BarrierOptimizationResult]:
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
                self.logger.info(f"Optimizing barriers for regime {regime}")
                result = await self.optimize_regime_barriers(
                    regime, regime_data[regime], min_trades
                )
                results[regime] = result
            else:
                self.logger.warning(f"Skipping regime {regime}: insufficient data")
                results[regime] = None
        
        return results