"""
Unified Trading Viability Evaluator

This module provides a unified trading viability evaluation system that combines
the best practices from both TAS and NAS regime detection systems. It evaluates
the practical trading viability of detected regimes considering real-world constraints.

Features:
- Unified trading metrics calculation
- Support for both tree-based and neural-based regime detection
- Position-aware trading analysis
- Real-world trading constraints evaluation
- Configurable viability thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

# Import position-aware trading analyzer
from .position_aware_trading import PositionAwareTradingAnalyzer, PositionAwareConfig

logger = logging.getLogger(__name__)


class TradingViabilityMetricType(Enum):
    """Types of trading viability metrics to evaluate."""
    TRADING_FREQUENCY = "trading_frequency"
    POSITION_DURATION = "position_duration"
    MODEL_CONFIDENCE = "model_confidence"
    RISK_ADJUSTED_RETURNS = "risk_adjusted_returns"
    TRANSACTION_COSTS = "transaction_costs"
    MARKET_LIQUIDITY = "market_liquidity"
    REGIME_STABILITY = "regime_stability"
    EXECUTION_FEASIBILITY = "execution_feasibility"


@dataclass
class TradingViabilityConfig:
    """Configuration for unified trading viability evaluation."""
    
    # Trading viability weights
    trading_frequency_weight: float = 0.20
    position_duration_weight: float = 0.15
    model_confidence_weight: float = 0.20
    risk_adjusted_returns_weight: float = 0.25
    transaction_costs_weight: float = 0.10
    market_liquidity_weight: float = 0.05
    regime_stability_weight: float = 0.05
    
    # Thresholds
    viability_threshold: float = 0.6
    min_trading_frequency: float = 0.1  # Minimum trades per day
    max_trading_frequency: float = 10.0  # Maximum trades per day
    min_position_duration: float = 5.0  # Minimum minutes
    max_position_duration: float = 1440.0  # Maximum minutes (1 day)
    min_model_confidence: float = 0.6
    min_risk_adjusted_return: float = 0.1
    
    # Transaction costs
    transaction_cost_bps: float = 1.0  # 1 basis point
    slippage_bps: float = 0.5  # 0.5 basis points
    market_impact_threshold: float = 0.001  # 0.1%
    
    # Position-aware analysis
    enable_position_aware_analysis: bool = True
    position_aware_config: Optional[PositionAwareConfig] = None
    
    # Advanced features
    enable_liquidity_analysis: bool = True
    liquidity_lookback: int = 20  # Lookback periods for liquidity
    enable_execution_analysis: bool = True
    execution_slippage_threshold: float = 0.002  # 0.2%
    
    # Regime-specific analysis
    enable_regime_specific_analysis: bool = True
    min_regime_samples: int = 50
    regime_stability_threshold: float = 0.7


@dataclass
class TradingViabilityResult:
    """Result from unified trading viability evaluation."""
    
    # Overall scores
    overall_score: float
    viability_level: str  # 'high', 'medium', 'low'
    
    # Individual metric scores
    trading_frequency_score: float
    position_duration_score: float
    model_confidence_score: float
    risk_adjusted_returns_score: float
    transaction_costs_score: float
    market_liquidity_score: float
    regime_stability_score: float
    execution_feasibility_score: float
    
    # Regime-specific analysis
    regime_viability_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime_viability_scores: Dict[str, float] = field(default_factory=dict)
    
    # Trading simulation results
    trading_simulation_results: Optional[Dict[str, Any]] = None
    
    # Position-aware analysis
    position_aware_analysis: Optional[Dict[str, Any]] = None
    
    # Execution analysis
    execution_feasibility_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    data_shape: Tuple[int, int] = (0, 0)
    n_regimes: int = 0
    evaluation_time: float = 0.0


class UnifiedTradingViabilityEvaluator:
    """
    Unified Trading Viability Evaluator.
    
    Combines the best practices from both TAS and NAS regime detection systems
    to provide comprehensive trading viability evaluation.
    """
    
    def __init__(self, config: TradingViabilityConfig):
        """Initialize unified trading viability evaluator.
        
        Args:
            config: Trading viability configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize position-aware analyzer if enabled
        self.position_analyzer = None
        if config.enable_position_aware_analysis:
            if config.position_aware_config is None:
                config.position_aware_config = PositionAwareConfig()
            self.position_analyzer = PositionAwareTradingAnalyzer(config.position_aware_config)
        
        self.logger.info("✅ Unified Trading Viability Evaluator initialized")
        self.logger.info(f"   Position-aware analysis: {config.enable_position_aware_analysis}")
        self.logger.info(f"   Liquidity analysis: {config.enable_liquidity_analysis}")
        self.logger.info(f"   Execution analysis: {config.enable_execution_analysis}")
    
    def evaluate(self, 
                 market_data: Union[pd.DataFrame, np.ndarray], 
                 regime_predictions: np.ndarray,
                 regime_probabilities: Optional[np.ndarray] = None,
                 timestamps: Optional[np.ndarray] = None,
                 regime_metadata: Optional[Dict[str, Any]] = None) -> TradingViabilityResult:
        """
        Evaluate trading viability of regimes using unified approach.
        
        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Regime predictions
            regime_probabilities: Optional regime probabilities
            timestamps: Optional timestamps
            regime_metadata: Optional regime metadata
            
        Returns:
            Comprehensive trading viability result
        """
        start_time = time.time()
        
        try:
            self.logger.info("📈 Starting unified trading viability evaluation...")
            self.logger.info(f"   Data shape: {market_data.shape}")
            self.logger.info(f"   Regimes: {len(np.unique(regime_predictions))}")
            
            # Convert data to numpy array if needed
            if isinstance(market_data, pd.DataFrame):
                data_array = market_data.values
                if timestamps is None and 'timestamp' in market_data.columns:
                    timestamps = market_data['timestamp'].values
            else:
                data_array = market_data
                if timestamps is None:
                    timestamps = np.arange(len(data_array))
            
            # Calculate individual trading viability metrics
            frequency_scores = self._calculate_trading_frequency_viability(data_array, regime_predictions, timestamps)
            duration_scores = self._calculate_position_duration_viability(data_array, regime_predictions, timestamps)
            confidence_scores = self._calculate_model_confidence_viability(regime_probabilities, regime_predictions)
            risk_scores = self._calculate_risk_adjusted_returns_viability(data_array, regime_predictions)
            cost_scores = self._calculate_transaction_costs_viability(data_array, regime_predictions)
            liquidity_scores = self._calculate_market_liquidity_viability(data_array, regime_predictions)
            stability_scores = self._calculate_regime_stability_viability(regime_predictions)
            execution_scores = self._calculate_execution_feasibility_viability(data_array, regime_predictions)
            
            # Calculate weighted overall trading viability
            overall_scores = (
                frequency_scores * self.config.trading_frequency_weight +
                duration_scores * self.config.position_duration_weight +
                confidence_scores * self.config.model_confidence_weight +
                risk_scores * self.config.risk_adjusted_returns_weight +
                cost_scores * self.config.transaction_costs_weight +
                liquidity_scores * self.config.market_liquidity_weight +
                stability_scores * self.config.regime_stability_weight
            )
            
            # Apply viability threshold
            viable_regimes = overall_scores >= self.config.viability_threshold
            
            # Regime-specific analysis
            regime_profiles = {}
            regime_viability = {}
            if self.config.enable_regime_specific_analysis:
                regime_profiles = self._analyze_regime_viability_profiles(data_array, regime_predictions, timestamps)
                regime_viability = self._calculate_regime_viability_scores(regime_predictions, overall_scores)
            
            # Trading simulation
            trading_simulation = None
            if self.config.enable_execution_analysis:
                trading_simulation = self._perform_trading_simulation(data_array, regime_predictions, timestamps)
            
            # Position-aware analysis
            position_analysis = None
            if self.position_analyzer:
                try:
                    df_data = pd.DataFrame(data_array, columns=['open', 'high', 'low', 'close', 'volume'])
                    position_analysis = self.position_analyzer.calculate_position_aware_trading_viability(
                        df_data, regime_predictions
                    )
                except Exception as e:
                    self.logger.warning(f"Position-aware analysis failed: {e}")
            
            # Execution feasibility analysis
            execution_analysis = None
            if self.config.enable_execution_analysis:
                execution_analysis = self._analyze_execution_feasibility(data_array, regime_predictions, timestamps)
            
            # Determine viability level
            mean_score = np.mean(overall_scores)
            if mean_score >= 0.8:
                viability_level = 'high'
            elif mean_score >= 0.6:
                viability_level = 'medium'
            else:
                viability_level = 'low'
            
            execution_time = time.time() - start_time
            
            # Create result
            result = TradingViabilityResult(
                overall_score=mean_score,
                viability_level=viability_level,
                trading_frequency_score=np.mean(frequency_scores),
                position_duration_score=np.mean(duration_scores),
                model_confidence_score=np.mean(confidence_scores),
                risk_adjusted_returns_score=np.mean(risk_scores),
                transaction_costs_score=np.mean(cost_scores),
                market_liquidity_score=np.mean(liquidity_scores),
                regime_stability_score=np.mean(stability_scores),
                execution_feasibility_score=np.mean(execution_scores),
                regime_viability_profiles=regime_profiles,
                regime_viability_scores=regime_viability,
                trading_simulation_results=trading_simulation,
                position_aware_analysis=position_analysis,
                execution_feasibility_analysis=execution_analysis,
                data_shape=data_array.shape,
                n_regimes=len(np.unique(regime_predictions)),
                evaluation_time=execution_time
            )
            
            self.logger.info(f"✅ Unified trading viability evaluation completed in {execution_time:.2f}s")
            self.logger.info(f"   Overall score: {mean_score:.3f}")
            self.logger.info(f"   Viability level: {viability_level}")
            self.logger.info(f"   Viable regimes: {np.sum(viable_regimes)}/{len(regime_predictions)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Unified trading viability evaluation failed: {e}")
            
            return TradingViabilityResult(
                overall_score=0.0,
                viability_level='low',
                trading_frequency_score=0.0,
                position_duration_score=0.0,
                model_confidence_score=0.0,
                risk_adjusted_returns_score=0.0,
                transaction_costs_score=0.0,
                market_liquidity_score=0.0,
                regime_stability_score=0.0,
                execution_feasibility_score=0.0,
                evaluation_time=execution_time
            )
    
    def _calculate_trading_frequency_viability(self, market_data: np.ndarray,
                                            regime_predictions: np.ndarray,
                                            timestamps: Optional[np.ndarray]) -> np.ndarray:
        """Calculate trading frequency viability."""
        try:
            frequency_scores = np.zeros(len(regime_predictions))
            
            # Calculate regime changes (proxy for trading frequency)
            regime_changes = np.diff(regime_predictions) != 0
            change_indices = np.where(regime_changes)[0]
            
            if len(change_indices) == 0:
                # No regime changes - very low frequency
                return np.ones(len(regime_predictions)) * 0.1
            
            # Calculate time between changes
            if timestamps is not None and len(timestamps) > 1:
                time_diffs = np.diff(timestamps[change_indices])
                avg_time_between_changes = np.mean(time_diffs)
                
                # Convert to frequency (changes per day)
                if avg_time_between_changes > 0:
                    frequency_per_day = 1.0 / (avg_time_between_changes / (24 * 60 * 60))  # Assuming seconds
                else:
                    frequency_per_day = 0.0
            else:
                # Estimate frequency based on data length
                total_periods = len(regime_predictions)
                frequency_per_day = len(change_indices) / (total_periods / (24 * 60))  # Assuming minute data
            
            # Calculate viability score based on frequency
            if frequency_per_day < self.config.min_trading_frequency:
                frequency_score = 0.2  # Too low frequency
            elif frequency_per_day > self.config.max_trading_frequency:
                frequency_score = 0.3  # Too high frequency
            else:
                # Optimal frequency range
                optimal_frequency = (self.config.min_trading_frequency + self.config.max_trading_frequency) / 2
                frequency_score = 1.0 - abs(frequency_per_day - optimal_frequency) / optimal_frequency
                frequency_score = max(0.0, min(1.0, frequency_score))
            
            # Apply score to all predictions
            frequency_scores[:] = frequency_score
            
            return frequency_scores
            
        except Exception as e:
            self.logger.warning(f"Trading frequency calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_position_duration_viability(self, market_data: np.ndarray,
                                            regime_predictions: np.ndarray,
                                            timestamps: Optional[np.ndarray]) -> np.ndarray:
        """Calculate position duration viability."""
        try:
            duration_scores = np.zeros(len(regime_predictions))
            
            # Calculate regime durations
            regime_durations = []
            current_regime = regime_predictions[0]
            current_duration = 1
            
            for i in range(1, len(regime_predictions)):
                if regime_predictions[i] == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_regime = regime_predictions[i]
                    current_duration = 1
            
            # Add last duration
            regime_durations.append(current_duration)
            
            if not regime_durations:
                return np.ones(len(regime_predictions)) * 0.5
            
            # Convert to time units if timestamps available
            if timestamps is not None and len(timestamps) > 1:
                avg_time_period = np.mean(np.diff(timestamps))
                regime_durations_minutes = [d * avg_time_period / 60 for d in regime_durations]  # Convert to minutes
            else:
                # Assume minute data
                regime_durations_minutes = regime_durations
            
            # Calculate duration viability
            avg_duration = np.mean(regime_durations_minutes)
            
            if avg_duration < self.config.min_position_duration:
                duration_score = 0.3  # Too short positions
            elif avg_duration > self.config.max_position_duration:
                duration_score = 0.4  # Too long positions
            else:
                # Optimal duration range
                optimal_duration = (self.config.min_position_duration + self.config.max_position_duration) / 2
                duration_score = 1.0 - abs(avg_duration - optimal_duration) / optimal_duration
                duration_score = max(0.0, min(1.0, duration_score))
            
            # Apply score to all predictions
            duration_scores[:] = duration_score
            
            return duration_scores
            
        except Exception as e:
            self.logger.warning(f"Position duration calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_model_confidence_viability(self, regime_probabilities: Optional[np.ndarray],
                                            regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate model confidence viability."""
        try:
            if regime_probabilities is None:
                # No probability information available
                return np.ones(len(regime_predictions)) * 0.5
            
            # Calculate confidence scores
            confidence_scores = np.zeros(len(regime_predictions))
            
            for i in range(len(regime_predictions)):
                predicted_regime = regime_predictions[i]
                
                if i < len(regime_probabilities):
                    # Get probability for predicted regime
                    regime_prob = regime_probabilities[i, predicted_regime] if regime_probabilities.ndim > 1 else regime_probabilities[i]
                    confidence_scores[i] = regime_prob
                else:
                    confidence_scores[i] = 0.5
            
            # Apply confidence threshold
            viable_confidence = confidence_scores >= self.config.min_model_confidence
            confidence_scores = np.where(viable_confidence, confidence_scores, confidence_scores * 0.5)
            
            return confidence_scores
            
        except Exception as e:
            self.logger.warning(f"Model confidence calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_risk_adjusted_returns_viability(self, market_data: np.ndarray,
                                                regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate risk-adjusted returns viability."""
        try:
            if market_data.shape[1] < 4:
                return np.ones(len(regime_predictions)) * 0.5
            
            close_prices = market_data[:, 3]
            risk_scores = np.zeros(len(regime_predictions))
            
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                if not np.any(regime_mask):
                    continue
                
                regime_prices = close_prices[regime_mask]
                
                if len(regime_prices) < 3:
                    continue
                
                # Calculate returns
                returns = np.diff(regime_prices) / regime_prices[:-1]
                
                # Calculate risk-adjusted metrics
                mean_return = np.mean(returns)
                volatility = np.std(returns)
                
                if volatility > 0:
                    sharpe_ratio = mean_return / volatility
                    risk_adjusted_return = sharpe_ratio
                else:
                    risk_adjusted_return = 0.0
                
                # Calculate viability score
                if risk_adjusted_return >= self.config.min_risk_adjusted_return:
                    risk_score = min(risk_adjusted_return / 2.0, 1.0)  # Normalize to 0-1
                else:
                    risk_score = risk_adjusted_return / self.config.min_risk_adjusted_return * 0.5
                
                risk_scores[regime_mask] = max(0.0, min(1.0, risk_score))
            
            return risk_scores
            
        except Exception as e:
            self.logger.warning(f"Risk-adjusted returns calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_transaction_costs_viability(self, market_data: np.ndarray,
                                             regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate transaction costs viability."""
        try:
            cost_scores = np.zeros(len(regime_predictions))
            
            # Calculate regime changes (trading opportunities)
            regime_changes = np.diff(regime_predictions) != 0
            n_trades = np.sum(regime_changes)
            
            if n_trades == 0:
                # No trading - no costs
                return np.ones(len(regime_predictions))
            
            # Estimate transaction costs
            if market_data.shape[1] >= 4:
                close_prices = market_data[:, 3]
                avg_price = np.mean(close_prices)
                
                # Calculate cost per trade
                cost_per_trade = avg_price * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
                total_costs = n_trades * cost_per_trade
                
                # Calculate cost as percentage of average price
                cost_percentage = total_costs / avg_price
                
                # Calculate viability score (lower costs are better)
                if cost_percentage <= self.config.market_impact_threshold:
                    cost_score = 1.0
                else:
                    cost_score = max(0.0, 1.0 - (cost_percentage - self.config.market_impact_threshold) / self.config.market_impact_threshold)
            else:
                # No price data - assume moderate costs
                cost_score = 0.7
            
            # Apply score to all predictions
            cost_scores[:] = cost_score
            
            return cost_scores
            
        except Exception as e:
            self.logger.warning(f"Transaction costs calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_market_liquidity_viability(self, market_data: np.ndarray,
                                            regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate market liquidity viability."""
        try:
            if not self.config.enable_liquidity_analysis or market_data.shape[1] < 5:
                return np.ones(len(regime_predictions)) * 0.5
            
            volumes = market_data[:, 4]  # Volume
            liquidity_scores = np.zeros(len(regime_predictions))
            
            # Calculate liquidity metrics
            avg_volume = np.mean(volumes)
            volume_volatility = np.std(volumes)
            
            # Liquidity score based on volume characteristics
            if avg_volume > 0:
                # Higher average volume is better
                volume_score = min(avg_volume / (avg_volume + volume_volatility), 1.0)
                
                # Lower volatility is better for liquidity
                volatility_penalty = min(volume_volatility / avg_volume, 1.0)
                
                liquidity_score = volume_score * (1.0 - volatility_penalty * 0.3)
            else:
                liquidity_score = 0.1  # Very low liquidity
            
            # Apply score to all predictions
            liquidity_scores[:] = liquidity_score
            
            return liquidity_scores
            
        except Exception as e:
            self.logger.warning(f"Market liquidity calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_regime_stability_viability(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime stability viability."""
        try:
            stability_scores = np.zeros(len(regime_predictions))
            
            if len(regime_predictions) < 2:
                return np.ones(len(regime_predictions)) * 0.5
            
            # Calculate regime changes
            regime_changes = np.sum(np.diff(regime_predictions) != 0)
            total_periods = len(regime_predictions) - 1
            
            # Stability is inverse of change frequency
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            stability = max(0.0, min(1.0, stability))
            
            # Apply stability threshold
            if stability >= self.config.regime_stability_threshold:
                stability_score = stability
            else:
                stability_score = stability * 0.5  # Penalty for low stability
            
            # Apply score to all predictions
            stability_scores[:] = stability_score
            
            return stability_scores
            
        except Exception as e:
            self.logger.warning(f"Regime stability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_execution_feasibility_viability(self, market_data: np.ndarray,
                                                regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate execution feasibility viability."""
        try:
            if not self.config.enable_execution_analysis:
                return np.ones(len(regime_predictions)) * 0.5
            
            execution_scores = np.zeros(len(regime_predictions))
            
            if market_data.shape[1] < 4:
                return np.ones(len(regime_predictions)) * 0.5
            
            # Calculate price volatility (affects execution)
            close_prices = market_data[:, 3]
            price_volatility = np.std(np.diff(close_prices) / close_prices[:-1])
            
            # Calculate execution feasibility
            if price_volatility <= self.config.execution_slippage_threshold:
                execution_score = 1.0  # Low volatility - easy execution
            else:
                execution_score = max(0.0, 1.0 - (price_volatility - self.config.execution_slippage_threshold) / self.config.execution_slippage_threshold)
            
            # Apply score to all predictions
            execution_scores[:] = execution_score
            
            return execution_scores
            
        except Exception as e:
            self.logger.warning(f"Execution feasibility calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _analyze_regime_viability_profiles(self, market_data: np.ndarray,
                                         regime_predictions: np.ndarray,
                                         timestamps: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Analyze viability profiles for each regime."""
        try:
            profiles = {}
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_data = market_data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Viability profile for this regime
                profile = {
                    'regime_id': regime,
                    'duration': len(regime_data),
                    'avg_volume': np.mean(regime_data[:, 4]) if regime_data.shape[1] > 4 else 1.0,
                    'price_volatility': np.std(regime_data[:, 3]) if regime_data.shape[1] > 3 else 0.0,
                    'trading_opportunities': self._count_trading_opportunities(regime_data),
                    'execution_difficulty': self._calculate_execution_difficulty(regime_data)
                }
                
                profiles[f'regime_{regime}'] = profile
            
            return profiles
            
        except Exception as e:
            self.logger.warning(f"Regime viability profile analysis failed: {e}")
            return {}
    
    def _count_trading_opportunities(self, regime_data: np.ndarray) -> int:
        """Count trading opportunities in regime data."""
        try:
            if regime_data.shape[1] < 4:
                return 0
            
            # Simple trading opportunity detection based on price movements
            prices = regime_data[:, 3]
            price_changes = np.diff(prices) / prices[:-1]
            
            # Count significant price movements
            significant_moves = np.abs(price_changes) > 0.001  # 0.1% threshold
            return np.sum(significant_moves)
            
        except Exception:
            return 0
    
    def _calculate_execution_difficulty(self, regime_data: np.ndarray) -> float:
        """Calculate execution difficulty for regime data."""
        try:
            if regime_data.shape[1] < 4:
                return 0.5
            
            prices = regime_data[:, 3]
            price_volatility = np.std(np.diff(prices) / prices[:-1])
            
            # Higher volatility = higher execution difficulty
            return min(price_volatility * 100, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_regime_viability_scores(self, regime_predictions: np.ndarray,
                                         overall_scores: np.ndarray) -> Dict[str, float]:
        """Calculate viability scores for each regime."""
        try:
            regime_scores = {}
            unique_regimes = np.unique(regime_predictions)
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_score = np.mean(overall_scores[regime_mask])
                regime_scores[f'regime_{regime}'] = regime_score
            
            return regime_scores
            
        except Exception as e:
            self.logger.warning(f"Regime viability score calculation failed: {e}")
            return {}
    
    def _perform_trading_simulation(self, market_data: np.ndarray,
                                  regime_predictions: np.ndarray,
                                  timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform trading simulation to assess viability."""
        try:
            if market_data.shape[1] < 4:
                return {}
            
            close_prices = market_data[:, 3]
            
            # Simple trading simulation
            positions = []
            returns = []
            costs = []
            
            current_position = 0
            entry_price = 0
            
            for i in range(len(regime_predictions)):
                regime = regime_predictions[i]
                price = close_prices[i]
                
                # Simple trading logic based on regime changes
                if i > 0 and regime != regime_predictions[i-1]:
                    # Regime change - close current position and open new one
                    if current_position != 0:
                        # Close position
                        trade_return = (price - entry_price) / entry_price * current_position
                        returns.append(trade_return)
                        
                        # Calculate transaction costs
                        cost = price * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
                        costs.append(cost)
                    
                    # Open new position (simplified)
                    current_position = 1 if regime % 2 == 0 else -1
                    entry_price = price
                
                positions.append(current_position)
            
            # Close final position
            if current_position != 0:
                trade_return = (close_prices[-1] - entry_price) / entry_price * current_position
                returns.append(trade_return)
                
                cost = close_prices[-1] * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
                costs.append(cost)
            
            # Calculate simulation results
            if returns:
                total_return = np.sum(returns)
                total_costs = np.sum(costs)
                net_return = total_return - total_costs
                win_rate = np.mean([r > 0 for r in returns])
                avg_return = np.mean(returns)
                return_volatility = np.std(returns)
                
                if return_volatility > 0:
                    sharpe_ratio = avg_return / return_volatility
                else:
                    sharpe_ratio = 0.0
            else:
                total_return = 0.0
                total_costs = 0.0
                net_return = 0.0
                win_rate = 0.0
                avg_return = 0.0
                sharpe_ratio = 0.0
            
            return {
                'total_return': total_return,
                'total_costs': total_costs,
                'net_return': net_return,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'n_trades': len(returns),
                'positions': positions
            }
            
        except Exception as e:
            self.logger.warning(f"Trading simulation failed: {e}")
            return {}
    
    def _analyze_execution_feasibility(self, market_data: np.ndarray,
                                     regime_predictions: np.ndarray,
                                     timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze execution feasibility."""
        try:
            if market_data.shape[1] < 4:
                return {}
            
            close_prices = market_data[:, 3]
            
            # Calculate execution metrics
            price_volatility = np.std(np.diff(close_prices) / close_prices[:-1])
            price_range = (np.max(close_prices) - np.min(close_prices)) / np.mean(close_prices)
            
            # Execution feasibility score
            if price_volatility <= self.config.execution_slippage_threshold:
                feasibility_score = 1.0
            else:
                feasibility_score = max(0.0, 1.0 - (price_volatility - self.config.execution_slippage_threshold) / self.config.execution_slippage_threshold)
            
            return {
                'feasibility_score': feasibility_score,
                'price_volatility': price_volatility,
                'price_range': price_range,
                'execution_difficulty': 1.0 - feasibility_score,
                'slippage_risk': min(price_volatility * 100, 1.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Execution feasibility analysis failed: {e}")
            return {}


# Convenience functions
def create_unified_trading_viability_evaluator(config: Optional[TradingViabilityConfig] = None) -> UnifiedTradingViabilityEvaluator:
    """Create a unified trading viability evaluator."""
    if config is None:
        config = TradingViabilityConfig()
    return UnifiedTradingViabilityEvaluator(config)


def quick_trading_viability_evaluation(market_data: Union[pd.DataFrame, np.ndarray],
                                     regime_predictions: np.ndarray,
                                     config: Optional[TradingViabilityConfig] = None) -> TradingViabilityResult:
    """Quick trading viability evaluation with default settings."""
    evaluator = create_unified_trading_viability_evaluator(config)
    return evaluator.evaluate(market_data, regime_predictions)