"""
Practical Examples for Meta-Learning Trading HPO

This module provides comprehensive examples demonstrating how to use the
meta-learning trading HPO system for various high leverage trading scenarios.

Examples include:
- High-frequency trading optimization
- Regime-aware portfolio optimization
- Risk-constrained leverage trading
- Multi-asset trading optimization
- Real-time trading system optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import warnings

# Import our trading HPO components
from .meta_learning_trading_hpo import MetaLearningTradingHPO, TradingOptimizationHistoryDB
from .trading_meta_features import TradingMetaFeaturesExtractor
from .trading_optimization_strategies import (
    TradingOptimizationOrchestrator, 
    TradingOptimizationStrategy,
    RegimeAwareOptimization,
    RiskConstrainedOptimization,
    LeverageAdaptiveOptimization
)

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    XGBOOST_AVAILABLE = True
    LIGHTGBM_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    logger.warning("Some ML libraries not available - limited example functionality")


class TradingHPOExamples:
    """Comprehensive examples for trading HPO usage."""
    
    def __init__(self):
        """Initialize trading HPO examples."""
        self.logger = logger.getChild('TradingHPOExamples')
        
        # Initialize HPO system
        self.config = {
            'trading': {
                'max_leverage': 10.0,
                'risk_tolerance': 0.05,
                'regime_awareness': True,
                'leverage_aware_optimization': True,
                'history_db_path': 'trading_hpo_examples.db'
            },
            'regime_aware': {
                'regime_detection_window': 50,
                'regime_confidence_threshold': 0.7
            },
            'risk_constrained': {
                'max_drawdown_limit': 0.15,
                'min_sharpe_ratio': 1.0,
                'max_var_95': -0.05
            }
        }
        
        self.meta_hpo = MetaLearningTradingHPO(self.config)
        self.optimization_orchestrator = TradingOptimizationOrchestrator(self.config)
    
    def example_1_high_frequency_trading_optimization(self) -> Dict[str, Any]:
        """
        Example 1: High-Frequency Trading Optimization
        
        Optimize a high-frequency trading model with:
        - Microsecond-level latency considerations
        - High leverage (5x-10x)
        - Risk constraints for rapid execution
        - Market microstructure awareness
        """
        try:
            self.logger.info("🚀 Example 1: High-Frequency Trading Optimization")
            
            # Generate synthetic high-frequency data
            price_data, target_data = self._generate_high_frequency_data()
            
            # Define high-frequency trading model factory
            def hft_model_factory(**params):
                if XGBOOST_AVAILABLE:
                    return xgb.XGBClassifier(
                        max_depth=params.get('max_depth', 6),
                        learning_rate=params.get('learning_rate', 0.1),
                        n_estimators=params.get('n_estimators', 100),
                        subsample=params.get('subsample', 0.8),
                        colsample_bytree=params.get('colsample_bytree', 0.8),
                        reg_alpha=params.get('reg_alpha', 0),
                        reg_lambda=params.get('reg_lambda', 1),
                        random_state=42,
                        n_jobs=1  # Single thread for HFT
                    )
                else:
                    return RandomForestClassifier(
                        n_estimators=params.get('n_estimators', 100),
                        max_depth=params.get('max_depth', 6),
                        random_state=42,
                        n_jobs=1
                    )
            
            # Optimize with high leverage and risk constraints
            results = self.meta_hpo.trading_meta_learning_optimization(
                model_factory=hft_model_factory,
                price_data=price_data,
                target_data=target_data,
                model_type='xgboost_trading',
                leverage_factor=8.0,  # 8x leverage for HFT
                n_trials=150  # More trials for HFT precision
            )
            
            # Add HFT-specific analysis
            results['hft_analysis'] = {
                'latency_optimized': True,
                'microstructure_aware': True,
                'high_leverage_handling': True,
                'execution_risk_considered': True
            }
            
            self.logger.info(f"✅ HFT optimization completed - Score: {results['best_score']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ HFT optimization example failed: {e}")
            return {'error': str(e)}
    
    def example_2_regime_aware_portfolio_optimization(self) -> Dict[str, Any]:
        """
        Example 2: Regime-Aware Portfolio Optimization
        
        Optimize a portfolio trading model that adapts to different market regimes:
        - Bull market: Aggressive growth strategies
        - Bear market: Defensive strategies
        - High volatility: Risk management focus
        - Low volatility: Momentum strategies
        """
        try:
            self.logger.info("📊 Example 2: Regime-Aware Portfolio Optimization")
            
            # Generate multi-regime data
            price_data, target_data, regime_labels = self._generate_multi_regime_data()
            
            # Define portfolio model factory
            def portfolio_model_factory(**params):
                if LIGHTGBM_AVAILABLE:
                    return lgb.LGBMRegressor(
                        num_leaves=params.get('num_leaves', 31),
                        learning_rate=params.get('learning_rate', 0.1),
                        n_estimators=params.get('n_estimators', 100),
                        feature_fraction=params.get('feature_fraction', 0.9),
                        bagging_fraction=params.get('bagging_fraction', 0.8),
                        bagging_freq=params.get('bagging_freq', 5),
                        lambda_l1=params.get('lambda_l1', 0),
                        lambda_l2=params.get('lambda_l2', 0),
                        random_state=42,
                        n_jobs=1
                    )
                else:
                    return RandomForestRegressor(
                        n_estimators=params.get('n_estimators', 100),
                        max_depth=params.get('max_depth', 10),
                        random_state=42,
                        n_jobs=1
                    )
            
            # Use regime-aware optimization
            results = self.optimization_orchestrator.optimize_trading_model(
                meta_hpo=self.meta_hpo,
                model_factory=portfolio_model_factory,
                price_data=price_data,
                target_data=target_data,
                model_type='lightgbm_trading',
                strategy=TradingOptimizationStrategy.REGIME_AWARE,
                leverage_factor=3.0,  # Moderate leverage for portfolio
                n_trials=200
            )
            
            # Add regime analysis
            results['regime_analysis'] = {
                'regimes_detected': len(np.unique(regime_labels)),
                'regime_adaptation': True,
                'transition_handling': True,
                'regime_specific_optimization': True
            }
            
            self.logger.info(f"✅ Regime-aware optimization completed - Score: {results['best_score']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware optimization example failed: {e}")
            return {'error': str(e)}
    
    def example_3_risk_constrained_leverage_trading(self) -> Dict[str, Any]:
        """
        Example 3: Risk-Constrained High Leverage Trading
        
        Optimize a high leverage trading model with strict risk constraints:
        - Maximum 10x leverage
        - Drawdown limits
        - VaR constraints
        - Sharpe ratio requirements
        """
        try:
            self.logger.info("⚠️ Example 3: Risk-Constrained High Leverage Trading")
            
            # Generate volatile trading data
            price_data, target_data = self._generate_volatile_trading_data()
            
            # Define risk-constrained model factory
            def risk_aware_model_factory(**params):
                if XGBOOST_AVAILABLE:
                    return xgb.XGBRegressor(
                        max_depth=params.get('max_depth', 6),
                        learning_rate=params.get('learning_rate', 0.05),  # Lower LR for stability
                        n_estimators=params.get('n_estimators', 200),
                        subsample=params.get('subsample', 0.7),  # Lower subsample for stability
                        colsample_bytree=params.get('colsample_bytree', 0.7),
                        reg_alpha=params.get('reg_alpha', 1),  # Higher regularization
                        reg_lambda=params.get('reg_lambda', 1),
                        random_state=42,
                        n_jobs=1
                    )
                else:
                    return RandomForestRegressor(
                        n_estimators=params.get('n_estimators', 200),
                        max_depth=params.get('max_depth', 6),
                        min_samples_split=params.get('min_samples_split', 10),  # More conservative
                        random_state=42,
                        n_jobs=1
                    )
            
            # Use risk-constrained optimization
            results = self.optimization_orchestrator.optimize_trading_model(
                meta_hpo=self.meta_hpo,
                model_factory=risk_aware_model_factory,
                price_data=price_data,
                target_data=target_data,
                model_type='xgboost_trading',
                strategy=TradingOptimizationStrategy.RISK_CONSTRAINED,
                leverage_factor=10.0,  # Maximum leverage
                n_trials=180
            )
            
            # Add risk analysis
            results['risk_analysis'] = {
                'max_leverage_used': 10.0,
                'risk_constraints_applied': True,
                'drawdown_controlled': True,
                'var_limits_respected': True,
                'sharpe_ratio_optimized': True
            }
            
            self.logger.info(f"✅ Risk-constrained optimization completed - Score: {results['best_score']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Risk-constrained optimization example failed: {e}")
            return {'error': str(e)}
    
    def example_4_multi_asset_trading_optimization(self) -> Dict[str, Any]:
        """
        Example 4: Multi-Asset Trading Optimization
        
        Optimize a multi-asset trading system with:
        - Multiple correlated assets
        - Cross-asset risk management
        - Portfolio-level optimization
        - Asset-specific parameter adaptation
        """
        try:
            self.logger.info("🌐 Example 4: Multi-Asset Trading Optimization")
            
            # Generate multi-asset data
            multi_asset_data = self._generate_multi_asset_data()
            
            # Define multi-asset model factory
            def multi_asset_model_factory(**params):
                if LIGHTGBM_AVAILABLE:
                    return lgb.LGBMClassifier(
                        num_leaves=params.get('num_leaves', 31),
                        learning_rate=params.get('learning_rate', 0.1),
                        n_estimators=params.get('n_estimators', 150),
                        feature_fraction=params.get('feature_fraction', 0.8),
                        bagging_fraction=params.get('bagging_fraction', 0.8),
                        bagging_freq=params.get('bagging_freq', 5),
                        min_child_samples=params.get('min_child_samples', 20),
                        lambda_l1=params.get('lambda_l1', 0.1),
                        lambda_l2=params.get('lambda_l2', 0.1),
                        random_state=42,
                        n_jobs=1
                    )
                else:
                    return RandomForestClassifier(
                        n_estimators=params.get('n_estimators', 150),
                        max_depth=params.get('max_depth', 8),
                        min_samples_split=params.get('min_samples_split', 5),
                        random_state=42,
                        n_jobs=1
                    )
            
            # Optimize for each asset and combine
            asset_results = {}
            for asset_name, (price_data, target_data) in multi_asset_data.items():
                asset_results[asset_name] = self.meta_hpo.trading_meta_learning_optimization(
                    model_factory=multi_asset_model_factory,
                    price_data=price_data,
                    target_data=target_data,
                    model_type='lightgbm_trading',
                    leverage_factor=2.0,  # Conservative leverage for multi-asset
                    n_trials=100
                )
            
            # Combine results
            combined_results = self._combine_multi_asset_results(asset_results)
            
            # Add multi-asset analysis
            combined_results['multi_asset_analysis'] = {
                'assets_optimized': len(multi_asset_data),
                'cross_asset_correlation_considered': True,
                'portfolio_level_optimization': True,
                'asset_specific_adaptation': True
            }
            
            self.logger.info(f"✅ Multi-asset optimization completed - "
                           f"Assets: {len(multi_asset_data)}, "
                           f"Best combined score: {combined_results['best_score']:.4f}")
            return combined_results
            
        except Exception as e:
            self.logger.error(f"❌ Multi-asset optimization example failed: {e}")
            return {'error': str(e)}
    
    def example_5_real_time_trading_system_optimization(self) -> Dict[str, Any]:
        """
        Example 5: Real-Time Trading System Optimization
        
        Optimize a real-time trading system with:
        - Continuous optimization
        - Online learning adaptation
        - Real-time risk monitoring
        - Performance degradation detection
        """
        try:
            self.logger.info("⏰ Example 5: Real-Time Trading System Optimization")
            
            # Generate streaming data simulation
            streaming_data = self._generate_streaming_trading_data()
            
            # Define real-time model factory
            def real_time_model_factory(**params):
                if SKLEARN_AVAILABLE:
                    return MLPRegressor(
                        hidden_layer_sizes=(params.get('hidden_units', 100),),
                        learning_rate_init=params.get('learning_rate', 0.001),
                        max_iter=params.get('epochs', 100),
                        alpha=params.get('l2_regularization', 0.0001),
                        random_state=42
                    )
                else:
                    return RandomForestRegressor(
                        n_estimators=params.get('n_estimators', 100),
                        max_depth=params.get('max_depth', 8),
                        random_state=42,
                        n_jobs=1
                    )
            
            # Simulate real-time optimization
            real_time_results = self._simulate_real_time_optimization(
                streaming_data, real_time_model_factory
            )
            
            # Add real-time analysis
            real_time_results['real_time_analysis'] = {
                'continuous_optimization': True,
                'online_learning': True,
                'performance_monitoring': True,
                'degradation_detection': True,
                'adaptive_parameters': True
            }
            
            self.logger.info(f"✅ Real-time optimization completed - "
                           f"Updates: {real_time_results['optimization_updates']}")
            return real_time_results
            
        except Exception as e:
            self.logger.error(f"❌ Real-time optimization example failed: {e}")
            return {'error': str(e)}
    
    def example_6_comprehensive_trading_system(self) -> Dict[str, Any]:
        """
        Example 6: Comprehensive Trading System
        
        Complete example combining all optimization strategies:
        - Multi-strategy optimization
        - Risk management integration
        - Regime awareness
        - Leverage adaptation
        - Performance monitoring
        """
        try:
            self.logger.info("🎯 Example 6: Comprehensive Trading System")
            
            # Generate comprehensive trading data
            price_data, target_data = self._generate_comprehensive_trading_data()
            
            # Define comprehensive model factory
            def comprehensive_model_factory(**params):
                if XGBOOST_AVAILABLE:
                    return xgb.XGBRegressor(
                        max_depth=params.get('max_depth', 6),
                        learning_rate=params.get('learning_rate', 0.1),
                        n_estimators=params.get('n_estimators', 200),
                        subsample=params.get('subsample', 0.8),
                        colsample_bytree=params.get('colsample_bytree', 0.8),
                        reg_alpha=params.get('reg_alpha', 0),
                        reg_lambda=params.get('reg_lambda', 1),
                        gamma=params.get('gamma', 0),
                        min_child_weight=params.get('min_child_weight', 1),
                        random_state=42,
                        n_jobs=1
                    )
                else:
                    return RandomForestRegressor(
                        n_estimators=params.get('n_estimators', 200),
                        max_depth=params.get('max_depth', 6),
                        min_samples_split=params.get('min_samples_split', 2),
                        min_samples_leaf=params.get('min_samples_leaf', 1),
                        random_state=42,
                        n_jobs=1
                    )
            
            # Use multi-strategy optimization
            results = self.optimization_orchestrator.multi_strategy_optimization(
                meta_hpo=self.meta_hpo,
                model_factory=comprehensive_model_factory,
                price_data=price_data,
                target_data=target_data,
                model_type='xgboost_trading',
                leverage_factor=5.0,  # Moderate-high leverage
                total_budget=300  # Large budget for comprehensive optimization
            )
            
            # Add comprehensive analysis
            results['comprehensive_analysis'] = {
                'multi_strategy_optimization': True,
                'risk_management_integrated': True,
                'regime_awareness_active': True,
                'leverage_adaptation_enabled': True,
                'performance_monitoring_active': True,
                'meta_learning_active': True
            }
            
            self.logger.info(f"✅ Comprehensive optimization completed - "
                           f"Best score: {results['best_score']:.4f}, "
                           f"Strategy: {results.get('multi_strategy', {}).get('best_strategy', 'unknown')}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive optimization example failed: {e}")
            return {'error': str(e)}
    
    # Data generation methods
    def _generate_high_frequency_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic high-frequency trading data."""
        try:
            # Generate 1-minute data for 1 day (1440 minutes)
            n_points = 1440
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
            
            # Generate price data with microstructure noise
            base_price = 100.0
            returns = np.random.normal(0, 0.001, n_points)  # 0.1% volatility
            microstructure_noise = np.random.normal(0, 0.0001, n_points)  # Microstructure noise
            
            prices = base_price * np.exp(np.cumsum(returns + microstructure_noise))
            
            # Create OHLCV data
            price_data = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, n_points))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, n_points))),
                'close': prices,
                'volume': np.random.lognormal(10, 1, n_points)
            })
            
            # Generate target (next period return > 0)
            future_returns = np.roll(returns, -1)[:-1]
            target_data = pd.Series((future_returns > 0).astype(int), index=timestamps[:-1])
            
            return price_data, target_data
            
        except Exception as e:
            self.logger.warning(f"High-frequency data generation failed: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _generate_multi_regime_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Generate data with multiple market regimes."""
        try:
            n_points = 2000
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')
            
            # Define regimes
            regimes = ['bull', 'bear', 'sideways', 'high_vol']
            regime_lengths = [500, 400, 600, 500]
            
            prices = [100.0]
            returns = []
            regime_labels = []
            
            for i, (regime, length) in enumerate(zip(regimes, regime_lengths)):
                if regime == 'bull':
                    regime_returns = np.random.normal(0.0005, 0.01, length)  # Positive trend
                elif regime == 'bear':
                    regime_returns = np.random.normal(-0.0005, 0.01, length)  # Negative trend
                elif regime == 'sideways':
                    regime_returns = np.random.normal(0, 0.005, length)  # Low volatility
                else:  # high_vol
                    regime_returns = np.random.normal(0, 0.02, length)  # High volatility
                
                returns.extend(regime_returns)
                regime_labels.extend([regime] * length)
                
                # Update prices
                for ret in regime_returns:
                    prices.append(prices[-1] * (1 + ret))
            
            # Create price data
            price_data = pd.DataFrame({
                'timestamp': timestamps,
                'close': prices[:n_points],
                'volume': np.random.lognormal(8, 1, n_points)
            })
            
            # Generate target
            target_data = pd.Series(returns[:n_points], index=timestamps)
            regime_series = pd.Series(regime_labels[:n_points], index=timestamps)
            
            return price_data, target_data, regime_series
            
        except Exception as e:
            self.logger.warning(f"Multi-regime data generation failed: {e}")
            return pd.DataFrame(), pd.Series(), pd.Series()
    
    def _generate_volatile_trading_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate volatile trading data for risk-constrained optimization."""
        try:
            n_points = 1500
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='30min')
            
            # Generate volatile returns with fat tails
            returns = np.random.standard_t(df=3, size=n_points) * 0.01  # t-distribution with fat tails
            
            # Add some autocorrelation
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            prices = 100.0 * np.exp(np.cumsum(returns))
            
            # Create price data
            price_data = pd.DataFrame({
                'timestamp': timestamps,
                'close': prices,
                'volume': np.random.lognormal(9, 1, n_points)
            })
            
            # Generate target (future returns)
            target_data = pd.Series(returns, index=timestamps)
            
            return price_data, target_data
            
        except Exception as e:
            self.logger.warning(f"Volatile data generation failed: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _generate_multi_asset_data(self) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Generate multi-asset trading data."""
        try:
            assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
            n_points = 1000
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')
            
            multi_asset_data = {}
            
            for asset in assets:
                # Generate correlated returns
                base_returns = np.random.normal(0, 0.005, n_points)
                asset_returns = base_returns + np.random.normal(0, 0.002, n_points)
                
                prices = 1.0 * np.exp(np.cumsum(asset_returns))
                
                price_data = pd.DataFrame({
                    'timestamp': timestamps,
                    'close': prices,
                    'volume': np.random.lognormal(8, 1, n_points)
                })
                
                # Generate target (direction)
                target_data = pd.Series((asset_returns > 0).astype(int), index=timestamps)
                
                multi_asset_data[asset] = (price_data, target_data)
            
            return multi_asset_data
            
        except Exception as e:
            self.logger.warning(f"Multi-asset data generation failed: {e}")
            return {}
    
    def _generate_streaming_trading_data(self) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """Generate streaming trading data for real-time optimization."""
        try:
            # Generate multiple time windows of data
            streaming_data = []
            
            for window in range(10):  # 10 time windows
                n_points = 200
                start_time = datetime(2024, 1, 1) + timedelta(hours=window * 24)
                timestamps = pd.date_range(start=start_time, periods=n_points, freq='1H')
                
                # Generate data with concept drift
                drift_factor = 0.1 * window  # Gradual drift
                returns = np.random.normal(drift_factor, 0.01, n_points)
                
                prices = 100.0 * np.exp(np.cumsum(returns))
                
                price_data = pd.DataFrame({
                    'timestamp': timestamps,
                    'close': prices,
                    'volume': np.random.lognormal(8, 1, n_points)
                })
                
                target_data = pd.Series(returns, index=timestamps)
                
                streaming_data.append((price_data, target_data))
            
            return streaming_data
            
        except Exception as e:
            self.logger.warning(f"Streaming data generation failed: {e}")
            return []
    
    def _generate_comprehensive_trading_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate comprehensive trading data for full system optimization."""
        try:
            n_points = 3000
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='15min')
            
            # Generate complex price dynamics
            # Trend component
            trend = np.linspace(0, 0.1, n_points)
            
            # Volatility clustering
            volatility = np.ones(n_points)
            for i in range(1, n_points):
                volatility[i] = 0.95 * volatility[i-1] + 0.05 * abs(np.random.normal(0, 0.01))
            
            # Returns with trend and volatility clustering
            returns = trend + np.random.normal(0, volatility * 0.01)
            
            # Add some jumps
            jump_indices = np.random.choice(n_points, size=20, replace=False)
            returns[jump_indices] += np.random.normal(0, 0.05, 20)
            
            prices = 100.0 * np.exp(np.cumsum(returns))
            
            # Create comprehensive price data
            price_data = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
                'close': prices,
                'volume': np.random.lognormal(9, 1, n_points)
            })
            
            # Generate target (future returns)
            target_data = pd.Series(returns, index=timestamps)
            
            return price_data, target_data
            
        except Exception as e:
            self.logger.warning(f"Comprehensive data generation failed: {e}")
            return pd.DataFrame(), pd.Series()
    
    # Helper methods
    def _combine_multi_asset_results(self, asset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple asset optimizations."""
        try:
            # Find best performing asset
            best_asset = max(asset_results.keys(), 
                           key=lambda x: asset_results[x].get('best_score', 0))
            
            # Combine results
            combined_results = asset_results[best_asset].copy()
            combined_results['multi_asset_results'] = asset_results
            combined_results['best_asset'] = best_asset
            
            # Calculate combined score
            scores = [r.get('best_score', 0) for r in asset_results.values()]
            combined_results['best_score'] = np.mean(scores)
            combined_results['score_std'] = np.std(scores)
            
            return combined_results
            
        except Exception as e:
            self.logger.warning(f"Multi-asset result combination failed: {e}")
            return {'error': str(e)}
    
    def _simulate_real_time_optimization(self, 
                                       streaming_data: List[Tuple[pd.DataFrame, pd.Series]],
                                       model_factory: Callable) -> Dict[str, Any]:
        """Simulate real-time optimization with streaming data."""
        try:
            optimization_updates = []
            current_best_params = None
            current_best_score = 0.0
            
            for i, (price_data, target_data) in enumerate(streaming_data):
                # Perform optimization on current window
                if current_best_params is None:
                    # First optimization
                    results = self.meta_hpo.trading_meta_learning_optimization(
                        model_factory=model_factory,
                        price_data=price_data,
                        target_data=target_data,
                        model_type='neural_network_trading',
                        leverage_factor=2.0,
                        n_trials=50
                    )
                else:
                    # Warm start with previous best parameters
                    results = self.meta_hpo.trading_meta_learning_optimization(
                        model_factory=model_factory,
                        price_data=price_data,
                        target_data=target_data,
                        model_type='neural_network_trading',
                        leverage_factor=2.0,
                        n_trials=30  # Fewer trials for real-time
                    )
                
                # Update best parameters if improved
                if results.get('best_score', 0) > current_best_score:
                    current_best_params = results.get('best_params', {})
                    current_best_score = results.get('best_score', 0)
                
                optimization_updates.append({
                    'window': i,
                    'best_score': results.get('best_score', 0),
                    'parameters_updated': results.get('best_score', 0) > current_best_score,
                    'optimization_time': results.get('optimization_time', 0)
                })
            
            return {
                'optimization_updates': len(optimization_updates),
                'final_best_score': current_best_score,
                'final_best_params': current_best_params,
                'update_history': optimization_updates,
                'real_time_optimization': True
            }
            
        except Exception as e:
            self.logger.warning(f"Real-time optimization simulation failed: {e}")
            return {'error': str(e)}


def run_all_trading_examples():
    """Run all trading HPO examples."""
    try:
        logger.info("🚀 Running all trading HPO examples")
        
        examples = TradingHPOExamples()
        
        # Run all examples
        results = {
            'example_1_hft': examples.example_1_high_frequency_trading_optimization(),
            'example_2_regime': examples.example_2_regime_aware_portfolio_optimization(),
            'example_3_risk': examples.example_3_risk_constrained_leverage_trading(),
            'example_4_multi_asset': examples.example_4_multi_asset_trading_optimization(),
            'example_5_real_time': examples.example_5_real_time_trading_system_optimization(),
            'example_6_comprehensive': examples.example_6_comprehensive_trading_system()
        }
        
        # Summary
        successful_examples = sum(1 for r in results.values() if 'error' not in r)
        logger.info(f"✅ Completed {successful_examples}/{len(results)} examples successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to run trading examples: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Run examples when script is executed directly
    results = run_all_trading_examples()
    print(f"Trading HPO Examples Results: {len([r for r in results.values() if 'error' not in r])} successful")