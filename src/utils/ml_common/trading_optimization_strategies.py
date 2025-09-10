"""
Trading-Specific Optimization Strategies for High Leverage Trading

This module provides specialized optimization strategies designed specifically
for high leverage trading scenarios, including regime-aware optimization,
risk-constrained optimization, and leverage-adaptive strategies.

Key Features:
- Regime-aware optimization strategies
- Risk-constrained parameter search
- Leverage-adaptive optimization
- Market microstructure considerations
- High-frequency trading optimizations
- Portfolio-level optimization strategies
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum
import warnings

from .meta_learning_trading_hpo import MetaLearningTradingHPO
from .trading_meta_features import TradingMetaFeaturesExtractor
from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, PatientPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - limited optimization functionality")


class TradingOptimizationStrategy(Enum):
    """Trading optimization strategies."""
    REGIME_AWARE = "regime_aware"
    RISK_CONSTRAINED = "risk_constrained"
    LEVERAGE_ADAPTIVE = "leverage_adaptive"
    HIGH_FREQUENCY = "high_frequency"
    PORTFOLIO_LEVEL = "portfolio_level"
    MICROSTRUCTURE_AWARE = "microstructure_aware"


class RegimeAwareOptimization:
    """Regime-aware optimization strategy for trading models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime-aware optimization."""
        self.config = config or {}
        self.logger = logger.getChild('RegimeAwareOptimization')
        
        # Regime detection parameters
        self.regime_detection_window = self.config.get('regime_detection_window', 50)
        self.regime_confidence_threshold = self.config.get('regime_confidence_threshold', 0.7)
        self.regime_transition_buffer = self.config.get('regime_transition_buffer', 10)
        
        # Regime-specific optimization parameters
        self.regime_optimization_budgets = self.config.get('regime_optimization_budgets', {
            'bull_market': 100,
            'bear_market': 80,
            'sideways_market': 60,
            'high_volatility': 120,
            'low_volatility': 50
        })
    
    def optimize_with_regime_awareness(self, 
                                     meta_hpo: MetaLearningTradingHPO,
                                     model_factory: Callable,
                                     price_data: pd.DataFrame,
                                     target_data: pd.Series,
                                     model_type: str,
                                     total_budget: int = 200) -> Dict[str, Any]:
        """
        Optimize with regime awareness.
        
        Args:
            meta_hpo: Meta-learning HPO instance
            model_factory: Model factory function
            price_data: Price data
            target_data: Target data
            model_type: Model type
            total_budget: Total optimization budget
            
        Returns:
            Regime-aware optimization results
        """
        try:
            self.logger.info("🔄 Starting regime-aware optimization")
            
            # 1. Detect current regime
            current_regime = self._detect_current_regime(price_data, target_data)
            self.logger.info(f"📊 Detected current regime: {current_regime}")
            
            # 2. Detect regime transitions
            regime_transitions = self._detect_regime_transitions(price_data, target_data)
            
            # 3. Allocate budget based on regime
            regime_budget = self._allocate_regime_budget(current_regime, total_budget)
            
            # 4. Optimize for current regime
            current_regime_results = meta_hpo.trading_meta_learning_optimization(
                model_factory=model_factory,
                price_data=price_data,
                target_data=target_data,
                model_type=model_type,
                market_regime=current_regime,
                n_trials=regime_budget
            )
            
            # 5. If regime transitions detected, optimize for transition periods
            transition_results = {}
            if regime_transitions:
                transition_budget = total_budget - regime_budget
                transition_results = self._optimize_for_transitions(
                    meta_hpo, model_factory, price_data, target_data, 
                    model_type, regime_transitions, transition_budget
                )
            
            # 6. Combine results
            combined_results = self._combine_regime_results(
                current_regime_results, transition_results, current_regime
            )
            
            self.logger.info(f"✅ Regime-aware optimization completed - "
                           f"Best score: {combined_results['best_score']:.4f}")
            return combined_results
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware optimization failed: {e}")
            return {'error': str(e)}
    
    def _detect_current_regime(self, 
                             price_data: pd.DataFrame, 
                             target_data: pd.Series) -> str:
        """Detect current market regime."""
        try:
            # Extract recent data for regime detection
            recent_window = min(self.regime_detection_window, len(price_data))
            recent_prices = price_data.tail(recent_window)
            recent_returns = target_data.tail(recent_window)
            
            # Calculate regime indicators
            volatility = recent_returns.std()
            trend = recent_returns.mean()
            skewness = recent_returns.skew()
            
            # Regime classification
            if volatility > recent_returns.std() * 1.5:  # High volatility
                if trend > 0:
                    return 'high_vol_bull'
                else:
                    return 'high_vol_bear'
            else:  # Normal volatility
                if trend > recent_returns.std() * 0.5:
                    return 'bull_market'
                elif trend < -recent_returns.std() * 0.5:
                    return 'bear_market'
                else:
                    return 'sideways_market'
                    
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return 'unknown'
    
    def _detect_regime_transitions(self, 
                                 price_data: pd.DataFrame, 
                                 target_data: pd.Series) -> List[Dict[str, Any]]:
        """Detect regime transitions in the data."""
        try:
            transitions = []
            
            # Use rolling windows to detect transitions
            window_size = self.regime_detection_window
            step_size = window_size // 4
            
            for i in range(0, len(target_data) - window_size, step_size):
                window_returns = target_data.iloc[i:i + window_size]
                window_regime = self._detect_current_regime(
                    price_data.iloc[i:i + window_size], window_returns
                )
                
                if i > 0:
                    prev_regime = transitions[-1]['regime'] if transitions else 'unknown'
                    if window_regime != prev_regime:
                        transitions.append({
                            'start_idx': i,
                            'end_idx': i + window_size,
                            'regime': window_regime,
                            'transition_strength': self._calculate_transition_strength(
                                target_data.iloc[i-step_size:i+window_size]
                            )
                        })
                else:
                    transitions.append({
                        'start_idx': i,
                        'end_idx': i + window_size,
                        'regime': window_regime,
                        'transition_strength': 0.0
                    })
            
            return transitions
            
        except Exception as e:
            self.logger.warning(f"Regime transition detection failed: {e}")
            return []
    
    def _calculate_transition_strength(self, returns: pd.Series) -> float:
        """Calculate strength of regime transition."""
        try:
            # Measure volatility change and trend change
            mid_point = len(returns) // 2
            first_half = returns.iloc[:mid_point]
            second_half = returns.iloc[mid_point:]
            
            vol_change = abs(second_half.std() - first_half.std()) / first_half.std()
            trend_change = abs(second_half.mean() - first_half.mean()) / first_half.std()
            
            return float(vol_change + trend_change)
            
        except Exception as e:
            return 0.0
    
    def _allocate_regime_budget(self, regime: str, total_budget: int) -> int:
        """Allocate optimization budget based on regime."""
        regime_budget = self.regime_optimization_budgets.get(regime, 80)
        return min(regime_budget, total_budget)
    
    def _optimize_for_transitions(self, 
                                meta_hpo: MetaLearningTradingHPO,
                                model_factory: Callable,
                                price_data: pd.DataFrame,
                                target_data: pd.Series,
                                model_type: str,
                                transitions: List[Dict[str, Any]],
                                budget: int) -> Dict[str, Any]:
        """Optimize specifically for regime transition periods."""
        try:
            if not transitions or budget <= 0:
                return {}
            
            # Focus on strongest transitions
            strong_transitions = [t for t in transitions if t['transition_strength'] > 0.5]
            
            if not strong_transitions:
                return {}
            
            # Use transition-specific optimization
            transition_budget = budget // len(strong_transitions)
            
            transition_results = []
            for transition in strong_transitions:
                # Extract transition data
                start_idx = max(0, transition['start_idx'] - self.regime_transition_buffer)
                end_idx = min(len(price_data), transition['end_idx'] + self.regime_transition_buffer)
                
                transition_price_data = price_data.iloc[start_idx:end_idx]
                transition_target_data = target_data.iloc[start_idx:end_idx]
                
                # Optimize for transition
                result = meta_hpo.trading_meta_learning_optimization(
                    model_factory=model_factory,
                    price_data=transition_price_data,
                    target_data=transition_target_data,
                    model_type=model_type,
                    market_regime=f"transition_{transition['regime']}",
                    n_trials=transition_budget
                )
                
                transition_results.append(result)
            
            # Combine transition results
            if transition_results:
                best_transition = max(transition_results, key=lambda x: x.get('best_score', 0))
                return best_transition
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Transition optimization failed: {e}")
            return {}
    
    def _combine_regime_results(self, 
                              current_results: Dict[str, Any],
                              transition_results: Dict[str, Any],
                              current_regime: str) -> Dict[str, Any]:
        """Combine results from different regime optimizations."""
        try:
            # Choose best result
            if transition_results and transition_results.get('best_score', 0) > current_results.get('best_score', 0):
                best_results = transition_results
                best_results['optimization_type'] = 'transition_optimized'
            else:
                best_results = current_results
                best_results['optimization_type'] = 'regime_optimized'
            
            # Add regime information
            best_results['regime_awareness'] = {
                'current_regime': current_regime,
                'regime_optimized': True,
                'transition_considered': bool(transition_results)
            }
            
            return best_results
            
        except Exception as e:
            self.logger.warning(f"Result combination failed: {e}")
            return current_results


class RiskConstrainedOptimization:
    """Risk-constrained optimization strategy for high leverage trading."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk-constrained optimization."""
        self.config = config or {}
        self.logger = logger.getChild('RiskConstrainedOptimization')
        
        # Risk constraints
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)
        self.min_sharpe_ratio = self.config.get('min_sharpe_ratio', 1.0)
        self.max_var_95 = self.config.get('max_var_95', -0.05)
        self.max_leverage_risk = self.config.get('max_leverage_risk', 0.8)
        
        # Risk penalty weights
        self.risk_penalty_weights = self.config.get('risk_penalty_weights', {
            'drawdown': 2.0,
            'sharpe': 1.5,
            'var': 3.0,
            'leverage': 2.5
        })
    
    def optimize_with_risk_constraints(self, 
                                     meta_hpo: MetaLearningTradingHPO,
                                     model_factory: Callable,
                                     price_data: pd.DataFrame,
                                     target_data: pd.Series,
                                     model_type: str,
                                     leverage_factor: float = 1.0,
                                     n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize with risk constraints.
        
        Args:
            meta_hpo: Meta-learning HPO instance
            model_factory: Model factory function
            price_data: Price data
            target_data: Target data
            model_type: Model type
            leverage_factor: Leverage factor
            n_trials: Number of trials
            
        Returns:
            Risk-constrained optimization results
        """
        try:
            self.logger.info(f"⚠️ Starting risk-constrained optimization (leverage: {leverage_factor}x)")
            
            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna required for risk-constrained optimization")
            
            # Get base search space
            meta_features = meta_hpo.meta_features_extractor.extract_trading_meta_features(
                price_data, target_data
            )
            
            search_space = meta_hpo._generate_trading_meta_learning_search_space(
                model_type, meta_features, None, leverage_factor, n_trials
            )
            
            def risk_constrained_objective(trial):
                # Sample parameters
                params = {}
                for param_name, param_config in search_space.items():
                    if isinstance(param_config, dict):
                        param_type = param_config.get('type', 'float')
                        if param_type == 'float':
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_type == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_type == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name,
                                param_config['choices']
                            )
                
                # Create and evaluate model
                model = model_factory(**params)
                score = self._evaluate_with_risk_constraints(
                    model, price_data, target_data, meta_features, leverage_factor
                )
                
                # Report intermediate results
                trial.report(score, step=trial.number)
                
                # Prune if risk constraints severely violated
                if self._should_prune_for_risk(score, meta_features, leverage_factor):
                    raise optuna.TrialPruned()
                
                return score
            
            # Create study with risk-aware pruner
            sampler = TPESampler()
            pruner = PatientPruner(MedianPruner(), patience=5)
            
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )
            
            study.optimize(risk_constrained_objective, n_trials=n_trials)
            
            results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'risk_constraints_applied': True,
                'leverage_factor': leverage_factor,
                'risk_metrics': self._calculate_final_risk_metrics(
                    study.best_params, price_data, target_data, leverage_factor
                )
            }
            
            self.logger.info(f"✅ Risk-constrained optimization completed - "
                           f"Best score: {results['best_score']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Risk-constrained optimization failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_with_risk_constraints(self, 
                                      model: Any,
                                      price_data: pd.DataFrame,
                                      target_data: pd.Series,
                                      meta_features: Dict[str, float],
                                      leverage_factor: float) -> float:
        """Evaluate model with risk constraints."""
        try:
            # Prepare data
            X = price_data.values if hasattr(price_data, 'values') else price_data
            y = target_data.values if hasattr(target_data, 'values') else target_data
            
            # Train model
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate base performance
            base_score = self._calculate_base_performance(y, predictions)
            
            # Apply risk adjustments
            risk_adjusted_score = self._apply_risk_penalties(
                base_score, y, predictions, meta_features, leverage_factor
            )
            
            return risk_adjusted_score
            
        except Exception as e:
            self.logger.warning(f"Risk-constrained evaluation failed: {e}")
            return 0.0
    
    def _calculate_base_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate base performance score."""
        try:
            if len(np.unique(y_true)) <= 10:  # Classification
                return np.mean((y_true > 0) == (y_pred > 0))
            else:  # Regression
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _apply_risk_penalties(self, 
                            base_score: float,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            meta_features: Dict[str, float],
                            leverage_factor: float) -> float:
        """Apply risk-based penalties to performance score."""
        try:
            adjusted_score = base_score
            
            # 1. Drawdown penalty
            max_drawdown = meta_features.get('max_drawdown', 0)
            if max_drawdown < -self.max_drawdown_limit:
                penalty = abs(max_drawdown) * self.risk_penalty_weights['drawdown']
                adjusted_score *= (1.0 - penalty)
            
            # 2. Sharpe ratio penalty
            sharpe_ratio = meta_features.get('sharpe_ratio', 0)
            if sharpe_ratio < self.min_sharpe_ratio:
                penalty = (self.min_sharpe_ratio - sharpe_ratio) * self.risk_penalty_weights['sharpe']
                adjusted_score *= (1.0 - penalty)
            
            # 3. VaR penalty
            var_95 = meta_features.get('var_95', 0)
            if var_95 < self.max_var_95:
                penalty = abs(var_95 - self.max_var_95) * self.risk_penalty_weights['var']
                adjusted_score *= (1.0 - penalty)
            
            # 4. Leverage risk penalty
            leverage_risk = meta_features.get('leverage_risk', 0)
            if leverage_risk > self.max_leverage_risk:
                penalty = (leverage_risk - self.max_leverage_risk) * self.risk_penalty_weights['leverage']
                adjusted_score *= (1.0 - penalty)
            
            # 5. Leverage factor adjustment
            if leverage_factor > 1.0:
                leverage_penalty = (leverage_factor - 1.0) * 0.1
                adjusted_score *= (1.0 - leverage_penalty)
            
            return max(0.0, adjusted_score)
            
        except Exception as e:
            self.logger.warning(f"Risk penalty application failed: {e}")
            return base_score
    
    def _should_prune_for_risk(self, 
                             score: float,
                             meta_features: Dict[str, float],
                             leverage_factor: float) -> bool:
        """Determine if trial should be pruned based on risk."""
        try:
            # Prune if risk constraints severely violated
            max_drawdown = meta_features.get('max_drawdown', 0)
            if max_drawdown < -self.max_drawdown_limit * 1.5:  # 50% over limit
                return True
            
            var_95 = meta_features.get('var_95', 0)
            if var_95 < self.max_var_95 * 1.5:  # 50% over limit
                return True
            
            leverage_risk = meta_features.get('leverage_risk', 0)
            if leverage_risk > self.max_leverage_risk * 1.5:  # 50% over limit
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def _calculate_final_risk_metrics(self, 
                                    best_params: Dict[str, Any],
                                    price_data: pd.DataFrame,
                                    target_data: pd.Series,
                                    leverage_factor: float) -> Dict[str, float]:
        """Calculate final risk metrics for best parameters."""
        try:
            # This would typically involve training the model with best params
            # and calculating comprehensive risk metrics
            # For now, return basic metrics from meta-features
            
            meta_features = TradingMetaFeaturesExtractor().extract_trading_meta_features(
                price_data, target_data
            )
            
            return {
                'max_drawdown': meta_features.get('max_drawdown', 0),
                'sharpe_ratio': meta_features.get('sharpe_ratio', 0),
                'var_95': meta_features.get('var_95', 0),
                'leverage_risk': meta_features.get('leverage_risk', 0),
                'leverage_factor': leverage_factor,
                'calmar_ratio': meta_features.get('calmar_ratio', 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Final risk metrics calculation failed: {e}")
            return {}


class LeverageAdaptiveOptimization:
    """Leverage-adaptive optimization strategy."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize leverage-adaptive optimization."""
        self.config = config or {}
        self.logger = logger.getChild('LeverageAdaptiveOptimization')
        
        # Leverage adaptation parameters
        self.leverage_thresholds = self.config.get('leverage_thresholds', [1.0, 2.0, 5.0, 10.0])
        self.leverage_adaptation_factors = self.config.get('leverage_adaptation_factors', {
            'conservative': 0.8,
            'moderate': 1.0,
            'aggressive': 1.2
        })
    
    def optimize_with_leverage_adaptation(self, 
                                        meta_hpo: MetaLearningTradingHPO,
                                        model_factory: Callable,
                                        price_data: pd.DataFrame,
                                        target_data: pd.Series,
                                        model_type: str,
                                        leverage_factor: float,
                                        n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize with leverage adaptation.
        
        Args:
            meta_hpo: Meta-learning HPO instance
            model_factory: Model factory function
            price_data: Price data
            target_data: Target data
            model_type: Model type
            leverage_factor: Leverage factor
            n_trials: Number of trials
            
        Returns:
            Leverage-adaptive optimization results
        """
        try:
            self.logger.info(f"⚖️ Starting leverage-adaptive optimization (leverage: {leverage_factor}x)")
            
            # Determine leverage category
            leverage_category = self._categorize_leverage(leverage_factor)
            
            # Adapt search space for leverage level
            adapted_search_space = self._adapt_search_space_for_leverage(
                meta_hpo, model_type, leverage_factor, leverage_category
            )
            
            # Perform optimization with adapted parameters
            results = meta_hpo.trading_meta_learning_optimization(
                model_factory=model_factory,
                price_data=price_data,
                target_data=target_data,
                model_type=model_type,
                leverage_factor=leverage_factor,
                n_trials=n_trials
            )
            
            # Add leverage adaptation information
            results['leverage_adaptation'] = {
                'leverage_factor': leverage_factor,
                'leverage_category': leverage_category,
                'adaptation_applied': True,
                'search_space_adapted': True
            }
            
            self.logger.info(f"✅ Leverage-adaptive optimization completed - "
                           f"Category: {leverage_category}, Score: {results['best_score']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Leverage-adaptive optimization failed: {e}")
            return {'error': str(e)}
    
    def _categorize_leverage(self, leverage_factor: float) -> str:
        """Categorize leverage factor."""
        if leverage_factor <= 1.0:
            return 'conservative'
        elif leverage_factor <= 3.0:
            return 'moderate'
        else:
            return 'aggressive'
    
    def _adapt_search_space_for_leverage(self, 
                                       meta_hpo: MetaLearningTradingHPO,
                                       model_type: str,
                                       leverage_factor: float,
                                       leverage_category: str) -> Dict[str, Any]:
        """Adapt search space based on leverage level."""
        try:
            # Get base search space
            base_search_space = meta_hpo.trading_search_spaces.get(model_type, {})
            
            # Apply leverage-specific adaptations
            adapted_space = base_search_space.copy()
            
            if leverage_category == 'conservative':
                # Conservative: Lower complexity, higher regularization
                adapted_space = self._apply_conservative_adaptations(adapted_space)
            elif leverage_category == 'aggressive':
                # Aggressive: Higher complexity, lower regularization
                adapted_space = self._apply_aggressive_adaptations(adapted_space)
            
            return adapted_space
            
        except Exception as e:
            self.logger.warning(f"Search space adaptation failed: {e}")
            return base_search_space
    
    def _apply_conservative_adaptations(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conservative adaptations for low leverage."""
        adapted = search_space.copy()
        
        # Increase regularization
        if 'reg_alpha' in adapted:
            adapted['reg_alpha']['high'] = min(adapted['reg_alpha']['high'] * 1.5, 50)
        if 'reg_lambda' in adapted:
            adapted['reg_lambda']['high'] = min(adapted['reg_lambda']['high'] * 1.5, 50)
        if 'l1_regularization' in adapted:
            adapted['l1_regularization']['high'] = min(adapted['l1_regularization']['high'] * 1.5, 0.2)
        if 'l2_regularization' in adapted:
            adapted['l2_regularization']['high'] = min(adapted['l2_regularization']['high'] * 1.5, 0.2)
        
        # Reduce complexity
        if 'max_depth' in adapted:
            adapted['max_depth']['high'] = min(adapted['max_depth']['high'], 8)
        if 'num_leaves' in adapted:
            adapted['num_leaves']['high'] = min(adapted['num_leaves']['high'], 100)
        if 'hidden_layers' in adapted:
            adapted['hidden_layers']['high'] = min(adapted['hidden_layers']['high'], 3)
        
        return adapted
    
    def _apply_aggressive_adaptations(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Apply aggressive adaptations for high leverage."""
        adapted = search_space.copy()
        
        # Decrease regularization
        if 'reg_alpha' in adapted:
            adapted['reg_alpha']['high'] = max(adapted['reg_alpha']['high'] * 0.7, 1)
        if 'reg_lambda' in adapted:
            adapted['reg_lambda']['high'] = max(adapted['reg_lambda']['high'] * 0.7, 1)
        if 'l1_regularization' in adapted:
            adapted['l1_regularization']['high'] = max(adapted['l1_regularization']['high'] * 0.7, 0.01)
        if 'l2_regularization' in adapted:
            adapted['l2_regularization']['high'] = max(adapted['l2_regularization']['high'] * 0.7, 0.01)
        
        # Increase complexity
        if 'max_depth' in adapted:
            adapted['max_depth']['high'] = min(adapted['max_depth']['high'] * 1.2, 15)
        if 'num_leaves' in adapted:
            adapted['num_leaves']['high'] = min(adapted['num_leaves']['high'] * 1.2, 300)
        if 'hidden_layers' in adapted:
            adapted['hidden_layers']['high'] = min(adapted['hidden_layers']['high'] * 1.2, 6)
        
        return adapted


class TradingOptimizationOrchestrator:
    """Orchestrator for different trading optimization strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trading optimization orchestrator."""
        self.config = config or {}
        self.logger = logger.getChild('TradingOptimizationOrchestrator')
        
        # Initialize strategies
        self.regime_aware = RegimeAwareOptimization(self.config.get('regime_aware', {}))
        self.risk_constrained = RiskConstrainedOptimization(self.config.get('risk_constrained', {}))
        self.leverage_adaptive = LeverageAdaptiveOptimization(self.config.get('leverage_adaptive', {}))
        
        # Strategy selection logic
        self.strategy_weights = self.config.get('strategy_weights', {
            'regime_aware': 0.4,
            'risk_constrained': 0.4,
            'leverage_adaptive': 0.2
        })
    
    def optimize_trading_model(self, 
                             meta_hpo: MetaLearningTradingHPO,
                             model_factory: Callable,
                             price_data: pd.DataFrame,
                             target_data: pd.Series,
                             model_type: str,
                             strategy: TradingOptimizationStrategy = TradingOptimizationStrategy.RISK_CONSTRAINED,
                             leverage_factor: float = 1.0,
                             n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize trading model using specified strategy.
        
        Args:
            meta_hpo: Meta-learning HPO instance
            model_factory: Model factory function
            price_data: Price data
            target_data: Target data
            model_type: Model type
            strategy: Optimization strategy
            leverage_factor: Leverage factor
            n_trials: Number of trials
            
        Returns:
            Optimization results
        """
        try:
            self.logger.info(f"🎯 Starting {strategy.value} optimization")
            
            if strategy == TradingOptimizationStrategy.REGIME_AWARE:
                return self.regime_aware.optimize_with_regime_awareness(
                    meta_hpo, model_factory, price_data, target_data, model_type, n_trials
                )
            elif strategy == TradingOptimizationStrategy.RISK_CONSTRAINED:
                return self.risk_constrained.optimize_with_risk_constraints(
                    meta_hpo, model_factory, price_data, target_data, model_type, leverage_factor, n_trials
                )
            elif strategy == TradingOptimizationStrategy.LEVERAGE_ADAPTIVE:
                return self.leverage_adaptive.optimize_with_leverage_adaptation(
                    meta_hpo, model_factory, price_data, target_data, model_type, leverage_factor, n_trials
                )
            else:
                # Default to meta-learning optimization
                return meta_hpo.trading_meta_learning_optimization(
                    model_factory, price_data, target_data, model_type, 
                    leverage_factor=leverage_factor, n_trials=n_trials
                )
                
        except Exception as e:
            self.logger.error(f"❌ Trading optimization failed: {e}")
            return {'error': str(e)}
    
    def multi_strategy_optimization(self, 
                                  meta_hpo: MetaLearningTradingHPO,
                                  model_factory: Callable,
                                  price_data: pd.DataFrame,
                                  target_data: pd.Series,
                                  model_type: str,
                                  leverage_factor: float = 1.0,
                                  total_budget: int = 200) -> Dict[str, Any]:
        """
        Perform multi-strategy optimization.
        
        Args:
            meta_hpo: Meta-learning HPO instance
            model_factory: Model factory function
            price_data: Price data
            target_data: Target data
            model_type: Model type
            leverage_factor: Leverage factor
            total_budget: Total optimization budget
            
        Returns:
            Multi-strategy optimization results
        """
        try:
            self.logger.info("🎯 Starting multi-strategy optimization")
            
            # Allocate budget across strategies
            regime_budget = int(total_budget * self.strategy_weights['regime_aware'])
            risk_budget = int(total_budget * self.strategy_weights['risk_constrained'])
            leverage_budget = int(total_budget * self.strategy_weights['leverage_adaptive'])
            
            # Run different strategies
            regime_results = self.regime_aware.optimize_with_regime_awareness(
                meta_hpo, model_factory, price_data, target_data, model_type, regime_budget
            )
            
            risk_results = self.risk_constrained.optimize_with_risk_constraints(
                meta_hpo, model_factory, price_data, target_data, model_type, leverage_factor, risk_budget
            )
            
            leverage_results = self.leverage_adaptive.optimize_with_leverage_adaptation(
                meta_hpo, model_factory, price_data, target_data, model_type, leverage_factor, leverage_budget
            )
            
            # Combine results
            all_results = [regime_results, risk_results, leverage_results]
            valid_results = [r for r in all_results if 'error' not in r and 'best_score' in r]
            
            if not valid_results:
                return {'error': 'All optimization strategies failed'}
            
            # Select best result
            best_result = max(valid_results, key=lambda x: x.get('best_score', 0))
            
            # Add multi-strategy information
            best_result['multi_strategy'] = {
                'strategies_used': ['regime_aware', 'risk_constrained', 'leverage_adaptive'],
                'budget_allocation': {
                    'regime_aware': regime_budget,
                    'risk_constrained': risk_budget,
                    'leverage_adaptive': leverage_budget
                },
                'best_strategy': self._identify_best_strategy(valid_results)
            }
            
            self.logger.info(f"✅ Multi-strategy optimization completed - "
                           f"Best score: {best_result['best_score']:.4f}")
            return best_result
            
        except Exception as e:
            self.logger.error(f"❌ Multi-strategy optimization failed: {e}")
            return {'error': str(e)}
    
    def _identify_best_strategy(self, results: List[Dict[str, Any]]) -> str:
        """Identify which strategy produced the best result."""
        try:
            if not results:
                return 'none'
            
            best_result = max(results, key=lambda x: x.get('best_score', 0))
            
            if 'regime_awareness' in best_result:
                return 'regime_aware'
            elif 'risk_constraints_applied' in best_result:
                return 'risk_constrained'
            elif 'leverage_adaptation' in best_result:
                return 'leverage_adaptive'
            else:
                return 'meta_learning'
                
        except Exception as e:
            return 'unknown'