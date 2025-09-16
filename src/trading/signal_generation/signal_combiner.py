"""
Signal Combiner

Combines signals from analyst and tactician components with regime-aware weighting
and confidence scoring.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from ..config.regime_config import RegimeType
from ..config.trading_config import TradingConfig

logger = system_logger.getChild('SignalCombiner')

@dataclass
class TradingSignal:
    """Trading signal structure."""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # 'analyst', 'tactician', 'combined'
    regime_weights: Dict[RegimeType, float]
    metadata: Dict[str, Any]

@dataclass
class SignalCombinationResult:
    """Result of signal combination."""
    combined_signal: TradingSignal
    analyst_signal: Optional[TradingSignal]
    tactician_signal: Optional[TradingSignal]
    combination_method: str
    confidence_score: float
    regime_adjustment: float
    metadata: Dict[str, Any]

class SignalCombiner:
    """
    Signal combiner that integrates analyst and tactician signals with regime awareness.
    
    Combines signals using various methods:
    - Weighted average based on historical performance
    - Regime-based weighting
    - Confidence-based weighting
    - Ensemble methods
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger.getChild('SignalCombiner')
        
        # Signal sources
        self.analyst_signals: List[TradingSignal] = []
        self.tactician_signals: List[TradingSignal] = []
        self.combined_signals: List[SignalCombinationResult] = []
        
        # Performance tracking
        self.analyst_performance: Dict[str, float] = {}
        self.tactician_performance: Dict[str, float] = {}
        
        # Combination methods
        self.combination_methods = {
            'weighted_average': self._combine_weighted_average,
            'regime_based': self._combine_regime_based,
            'confidence_weighted': self._combine_confidence_weighted,
            'ensemble': self._combine_ensemble,
            'majority_vote': self._combine_majority_vote
        }
        
        # Default combination method
        self.default_method = 'regime_based'
        
        # Regime-specific weights
        self.regime_weights = {
            RegimeType.TRENDING_UP: {'analyst': 0.6, 'tactician': 0.4},
            RegimeType.TRENDING_DOWN: {'analyst': 0.6, 'tactician': 0.4},
            RegimeType.SIDEWAYS: {'analyst': 0.4, 'tactician': 0.6},
            RegimeType.HIGH_VOLATILITY: {'analyst': 0.7, 'tactician': 0.3},
            RegimeType.LOW_VOLATILITY: {'analyst': 0.5, 'tactician': 0.5},
            RegimeType.BREAKOUT: {'analyst': 0.8, 'tactician': 0.2},
            RegimeType.REVERSAL: {'analyst': 0.3, 'tactician': 0.7},
            RegimeType.MOMENTUM: {'analyst': 0.6, 'tactician': 0.4},
            RegimeType.MEAN_REVERSION: {'analyst': 0.4, 'tactician': 0.6},
        }
        
        # Default weights
        self.default_weights = {'analyst': 0.6, 'tactician': 0.4}
        
    @handles_errors
    @log_execution_time()
    @traced(span_name="combine_signals")
    async def combine_signals(
        self,
        symbol: str,
        analyst_signal: Optional[TradingSignal],
        tactician_signal: Optional[TradingSignal],
        regime_probabilities: Dict[RegimeType, float],
        method: Optional[str] = None
    ) -> SignalCombinationResult:
        """
        Combine analyst and tactician signals.
        
        Args:
            symbol: Trading symbol
            analyst_signal: Signal from analyst component
            tactician_signal: Signal from tactician component
            regime_probabilities: Current regime probabilities
            method: Combination method to use
            
        Returns:
            SignalCombinationResult: Combined signal with metadata
        """
        try:
            if method is None:
                method = self.default_method
            
            if method not in self.combination_methods:
                raise ValueError(f"Unknown combination method: {method}")
            
            # Validate inputs
            if not analyst_signal and not tactician_signal:
                raise ValueError("At least one signal must be provided")
            
            # Get combination function
            combine_func = self.combination_methods[method]
            
            # Combine signals
            result = await combine_func(
                symbol, analyst_signal, tactician_signal, regime_probabilities
            )
            
            # Store result
            self.combined_signals.append(result)
            
            # Maintain history size
            if len(self.combined_signals) > 1000:
                self.combined_signals = self.combined_signals[-1000:]
            
            self.logger.debug(f"Signals combined for {symbol} using {method}: {result.combined_signal.signal_type}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Signal combination failed for {symbol}: {e}")
            raise
    
    async def _combine_weighted_average(
        self,
        symbol: str,
        analyst_signal: Optional[TradingSignal],
        tactician_signal: Optional[TradingSignal],
        regime_probabilities: Dict[RegimeType, float]
    ) -> SignalCombinationResult:
        """Combine signals using weighted average based on historical performance."""
        try:
            # Get performance-based weights
            analyst_weight = self.analyst_performance.get(symbol, 0.5)
            tactician_weight = self.tactician_performance.get(symbol, 0.5)
            
            # Normalize weights
            total_weight = analyst_weight + tactician_weight
            if total_weight > 0:
                analyst_weight /= total_weight
                tactician_weight /= total_weight
            else:
                analyst_weight = tactician_weight = 0.5
            
            # Calculate combined signal
            combined_strength = 0.0
            combined_confidence = 0.0
            signal_type = 'hold'
            
            if analyst_signal:
                combined_strength += analyst_signal.strength * analyst_weight
                combined_confidence += analyst_signal.confidence * analyst_weight
                if analyst_signal.signal_type != 'hold':
                    signal_type = analyst_signal.signal_type
            
            if tactician_signal:
                combined_strength += tactician_signal.strength * tactician_weight
                combined_confidence += tactician_signal.confidence * tactician_weight
                if tactician_signal.signal_type != 'hold' and signal_type == 'hold':
                    signal_type = tactician_signal.signal_type
            
            # Determine final signal type based on strength
            if combined_strength > 0.6:
                if signal_type == 'hold':
                    signal_type = 'buy' if combined_strength > 0 else 'sell'
            elif combined_strength < -0.6:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Create combined signal
            combined_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                strength=combined_strength,
                confidence=combined_confidence,
                source='combined',
                regime_weights=regime_probabilities,
                metadata={
                    'combination_method': 'weighted_average',
                    'analyst_weight': analyst_weight,
                    'tactician_weight': tactician_weight
                }
            )
            
            return SignalCombinationResult(
                combined_signal=combined_signal,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method='weighted_average',
                confidence_score=combined_confidence,
                regime_adjustment=0.0,
                metadata={
                    'analyst_weight': analyst_weight,
                    'tactician_weight': tactician_weight
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Weighted average combination failed: {e}")
            raise
    
    async def _combine_regime_based(
        self,
        symbol: str,
        analyst_signal: Optional[TradingSignal],
        tactician_signal: Optional[TradingSignal],
        regime_probabilities: Dict[RegimeType, float]
    ) -> SignalCombinationResult:
        """Combine signals using regime-based weighting."""
        try:
            # Calculate regime-based weights
            analyst_weight = 0.0
            tactician_weight = 0.0
            
            for regime, probability in regime_probabilities.items():
                regime_weights = self.regime_weights.get(regime, self.default_weights)
                analyst_weight += regime_weights['analyst'] * probability
                tactician_weight += regime_weights['tactician'] * probability
            
            # Normalize weights
            total_weight = analyst_weight + tactician_weight
            if total_weight > 0:
                analyst_weight /= total_weight
                tactician_weight /= total_weight
            else:
                analyst_weight = tactician_weight = 0.5
            
            # Calculate combined signal
            combined_strength = 0.0
            combined_confidence = 0.0
            signal_type = 'hold'
            
            if analyst_signal:
                combined_strength += analyst_signal.strength * analyst_weight
                combined_confidence += analyst_signal.confidence * analyst_weight
                if analyst_signal.signal_type != 'hold':
                    signal_type = analyst_signal.signal_type
            
            if tactician_signal:
                combined_strength += tactician_signal.strength * tactician_weight
                combined_confidence += tactician_signal.confidence * tactician_weight
                if tactician_signal.signal_type != 'hold' and signal_type == 'hold':
                    signal_type = tactician_signal.signal_type
            
            # Determine final signal type
            if combined_strength > 0.6:
                if signal_type == 'hold':
                    signal_type = 'buy'
            elif combined_strength < -0.6:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Create combined signal
            combined_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                strength=combined_strength,
                confidence=combined_confidence,
                source='combined',
                regime_weights=regime_probabilities,
                metadata={
                    'combination_method': 'regime_based',
                    'analyst_weight': analyst_weight,
                    'tactician_weight': tactician_weight
                }
            )
            
            return SignalCombinationResult(
                combined_signal=combined_signal,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method='regime_based',
                confidence_score=combined_confidence,
                regime_adjustment=abs(analyst_weight - tactician_weight),
                metadata={
                    'analyst_weight': analyst_weight,
                    'tactician_weight': tactician_weight,
                    'regime_probabilities': regime_probabilities
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Regime-based combination failed: {e}")
            raise
    
    async def _combine_confidence_weighted(
        self,
        symbol: str,
        analyst_signal: Optional[TradingSignal],
        tactician_signal: Optional[TradingSignal],
        regime_probabilities: Dict[RegimeType, float]
    ) -> SignalCombinationResult:
        """Combine signals using confidence-based weighting."""
        try:
            # Calculate confidence-based weights
            analyst_confidence = analyst_signal.confidence if analyst_signal else 0.0
            tactician_confidence = tactician_signal.confidence if tactician_signal else 0.0
            
            total_confidence = analyst_confidence + tactician_confidence
            if total_confidence > 0:
                analyst_weight = analyst_confidence / total_confidence
                tactician_weight = tactician_confidence / total_confidence
            else:
                analyst_weight = tactician_weight = 0.5
            
            # Calculate combined signal
            combined_strength = 0.0
            combined_confidence = 0.0
            signal_type = 'hold'
            
            if analyst_signal:
                combined_strength += analyst_signal.strength * analyst_weight
                combined_confidence += analyst_signal.confidence * analyst_weight
                if analyst_signal.signal_type != 'hold':
                    signal_type = analyst_signal.signal_type
            
            if tactician_signal:
                combined_strength += tactician_signal.strength * tactician_weight
                combined_confidence += tactician_signal.confidence * tactician_weight
                if tactician_signal.signal_type != 'hold' and signal_type == 'hold':
                    signal_type = tactician_signal.signal_type
            
            # Determine final signal type
            if combined_strength > 0.6:
                if signal_type == 'hold':
                    signal_type = 'buy'
            elif combined_strength < -0.6:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Create combined signal
            combined_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                strength=combined_strength,
                confidence=combined_confidence,
                source='combined',
                regime_weights=regime_probabilities,
                metadata={
                    'combination_method': 'confidence_weighted',
                    'analyst_weight': analyst_weight,
                    'tactician_weight': tactician_weight
                }
            )
            
            return SignalCombinationResult(
                combined_signal=combined_signal,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method='confidence_weighted',
                confidence_score=combined_confidence,
                regime_adjustment=0.0,
                metadata={
                    'analyst_weight': analyst_weight,
                    'tactician_weight': tactician_weight
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Confidence-weighted combination failed: {e}")
            raise
    
    async def _combine_ensemble(
        self,
        symbol: str,
        analyst_signal: Optional[TradingSignal],
        tactician_signal: Optional[TradingSignal],
        regime_probabilities: Dict[RegimeType, float]
    ) -> SignalCombinationResult:
        """Combine signals using ensemble methods."""
        try:
            # Get signals from all methods
            weighted_avg_result = await self._combine_weighted_average(
                symbol, analyst_signal, tactician_signal, regime_probabilities
            )
            regime_based_result = await self._combine_regime_based(
                symbol, analyst_signal, tactician_signal, regime_probabilities
            )
            confidence_weighted_result = await self._combine_confidence_weighted(
                symbol, analyst_signal, tactician_signal, regime_probabilities
            )
            
            # Ensemble weights
            ensemble_weights = {
                'weighted_average': 0.3,
                'regime_based': 0.4,
                'confidence_weighted': 0.3
            }
            
            # Calculate ensemble signal
            combined_strength = 0.0
            combined_confidence = 0.0
            
            for result, weight in ensemble_weights.items():
                if result == 'weighted_average':
                    signal = weighted_avg_result.combined_signal
                elif result == 'regime_based':
                    signal = regime_based_result.combined_signal
                else:
                    signal = confidence_weighted_result.combined_signal
                
                combined_strength += signal.strength * weight
                combined_confidence += signal.confidence * weight
            
            # Determine signal type
            signal_type = 'hold'
            if combined_strength > 0.6:
                signal_type = 'buy'
            elif combined_strength < -0.6:
                signal_type = 'sell'
            
            # Create ensemble signal
            combined_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                strength=combined_strength,
                confidence=combined_confidence,
                source='combined',
                regime_weights=regime_probabilities,
                metadata={
                    'combination_method': 'ensemble',
                    'ensemble_weights': ensemble_weights
                }
            )
            
            return SignalCombinationResult(
                combined_signal=combined_signal,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method='ensemble',
                confidence_score=combined_confidence,
                regime_adjustment=0.0,
                metadata={
                    'ensemble_weights': ensemble_weights,
                    'individual_results': {
                        'weighted_average': weighted_avg_result.combined_signal.strength,
                        'regime_based': regime_based_result.combined_signal.strength,
                        'confidence_weighted': confidence_weighted_result.combined_signal.strength
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble combination failed: {e}")
            raise
    
    async def _combine_majority_vote(
        self,
        symbol: str,
        analyst_signal: Optional[TradingSignal],
        tactician_signal: Optional[TradingSignal],
        regime_probabilities: Dict[RegimeType, float]
    ) -> SignalCombinationResult:
        """Combine signals using majority vote."""
        try:
            # Collect signals
            signals = []
            if analyst_signal and analyst_signal.signal_type != 'hold':
                signals.append(analyst_signal)
            if tactician_signal and tactician_signal.signal_type != 'hold':
                signals.append(tactician_signal)
            
            if not signals:
                # No active signals, return hold
                combined_signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type='hold',
                    strength=0.0,
                    confidence=0.0,
                    source='combined',
                    regime_weights=regime_probabilities,
                    metadata={'combination_method': 'majority_vote'}
                )
                
                return SignalCombinationResult(
                    combined_signal=combined_signal,
                    analyst_signal=analyst_signal,
                    tactician_signal=tactician_signal,
                    combination_method='majority_vote',
                    confidence_score=0.0,
                    regime_adjustment=0.0,
                    metadata={}
                )
            
            # Count votes
            buy_votes = sum(1 for s in signals if s.signal_type == 'buy')
            sell_votes = sum(1 for s in signals if s.signal_type == 'sell')
            
            # Determine majority
            if buy_votes > sell_votes:
                signal_type = 'buy'
                strength = sum(s.strength for s in signals if s.signal_type == 'buy') / buy_votes
            elif sell_votes > buy_votes:
                signal_type = 'sell'
                strength = sum(s.strength for s in signals if s.signal_type == 'sell') / sell_votes
            else:
                signal_type = 'hold'
                strength = 0.0
            
            # Calculate average confidence
            confidence = sum(s.confidence for s in signals) / len(signals)
            
            # Create combined signal
            combined_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                source='combined',
                regime_weights=regime_probabilities,
                metadata={
                    'combination_method': 'majority_vote',
                    'buy_votes': buy_votes,
                    'sell_votes': sell_votes
                }
            )
            
            return SignalCombinationResult(
                combined_signal=combined_signal,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method='majority_vote',
                confidence_score=confidence,
                regime_adjustment=0.0,
                metadata={
                    'buy_votes': buy_votes,
                    'sell_votes': sell_votes
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Majority vote combination failed: {e}")
            raise
    
    def update_performance(self, symbol: str, analyst_performance: float, tactician_performance: float):
        """Update performance metrics for signal sources."""
        try:
            self.analyst_performance[symbol] = analyst_performance
            self.tactician_performance[symbol] = tactician_performance
            
            self.logger.debug(f"Performance updated for {symbol}: analyst={analyst_performance:.3f}, tactician={tactician_performance:.3f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update performance for {symbol}: {e}")
    
    def get_available_methods(self) -> list[str]:
        """Get list of available combination methods."""
        return list(self.combination_methods.keys())
    
    def set_default_method(self, method: str):
        """Set default combination method."""
        if method in self.combination_methods:
            self.default_method = method
            self.logger.info(f"Default combination method set to: {method}")
        else:
            self.logger.warning(f"Unknown combination method: {method}")
    
    def get_combination_history(self, limit: int = 100) -> List[SignalCombinationResult]:
        """Get recent signal combination history."""
        return self.combined_signals[-limit:] if self.combined_signals else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for signal sources."""
        return {
            'analyst_performance': self.analyst_performance.copy(),
            'tactician_performance': self.tactician_performance.copy(),
            'total_combinations': len(self.combined_signals),
            'default_method': self.default_method
        }
    
    async def stop(self):
        """Stop signal combiner."""
        try:
            self.logger.info("🛑 Stopping Signal Combiner...")
            self.logger.info("✅ Signal Combiner stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Signal Combiner: {e}")