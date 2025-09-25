"""
Signal Combiner (Refactored)

This module combines signals from Analyst and Tactician components
using shared utilities for consistent signal processing and combination.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

# Import shared utilities
from ..utils.confidence_calculator import UnifiedConfidenceCalculator, ConfidenceMetrics
from ..utils.fallback_analyzer import UnifiedFallbackAnalyzer, FallbackAnalysisResult

# Import signal types
from .analyst_signals_refactored import AnalystSignal, SignalType, SignalStrength
from .tactician_signals_refactored import TacticianSignal, TimingSignal, TimingConfidence

logger = system_logger.getChild('SignalCombinerRefactored')

class CombinedAction(Enum):
    """Combined trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"
    INCREASE = "increase"

class CombinationMethod(Enum):
    """Signal combination methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    RISK_ADJUSTED = "risk_adjusted"

@dataclass
class CombinedSignal:
    """Combined signal from Analyst and Tactician."""
    timestamp: datetime
    symbol: str
    action: CombinedAction
    confidence: float
    strength: float
    analyst_signal: Optional[AnalystSignal] = None
    tactician_signal: Optional[TacticianSignal] = None
    combination_method: CombinationMethod = CombinationMethod.WEIGHTED_AVERAGE
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    position_sizing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CombinationWeights:
    """Weights for signal combination."""
    analyst_weight: float = 0.6
    tactician_weight: float = 0.4
    confidence_threshold: float = 0.6
    risk_adjustment_factor: float = 0.8
    regime_adjustment_factor: float = 1.0

class SignalCombiner:
    """
    Refactored Signal Combiner using shared utilities.
    
    Combines Analyst and Tactician signals to generate final trading decisions
    with proper weighting and risk management using shared utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the signal combiner with shared utilities.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('SignalCombiner')
        
        # Initialize shared utilities
        self.confidence_calculator = UnifiedConfidenceCalculator(config.get('confidence_config', {}))
        self.fallback_analyzer = UnifiedFallbackAnalyzer(config.get('fallback_config', {}))
        
        # Combination parameters
        self.weights = CombinationWeights(
            analyst_weight=config.get('analyst_weight', 0.6),
            tactician_weight=config.get('tactician_weight', 0.4),
            confidence_threshold=config.get('confidence_threshold', 0.6),
            risk_adjustment_factor=config.get('risk_adjustment_factor', 0.8),
            regime_adjustment_factor=config.get('regime_adjustment_factor', 1.0)
        )
        
        # Combination method
        self.combination_method = CombinationMethod(
            config.get('combination_method', 'weighted_average')
        )
        
        # Performance tracking
        self.combination_history: List[CombinedSignal] = []
        self.max_history = config.get('max_history', 1000)
        
        # Performance metrics
        self.combination_count = 0
        self.successful_combinations = 0
        self.failed_combinations = 0

    async def initialize(self) -> bool:
        """
        Initialize the signal combiner.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("✅ Signal Combiner initialized with shared utilities")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Signal Combiner: {e}")
            return False

    @handles_errors
    @traced(span_name="signal_combination")
    @log_execution_time()
    async def combine_signals(
        self,
        analyst_signal: Optional[AnalystSignal] = None,
        tactician_signal: Optional[TacticianSignal] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Combine Analyst and Tactician signals using shared utilities.
        
        Args:
            analyst_signal: Signal from Analyst component
            tactician_signal: Signal from Tactician component
            additional_context: Additional context for combination
            
        Returns:
            Combined signal result or None if no valid combination
        """
        try:
            tprint_info("🔄 Combining Analyst and Tactician signals...")
            
            # Validate inputs
            if not analyst_signal and not tactician_signal:
                tprint_warning("⚠️ No signals provided for combination")
                return None
            
            # Perform signal combination based on method
            if self.combination_method == CombinationMethod.WEIGHTED_AVERAGE:
                combined_signal = await self._weighted_average_combination(
                    analyst_signal, tactician_signal, additional_context
                )
            elif self.combination_method == CombinationMethod.CONFIDENCE_WEIGHTED:
                combined_signal = await self._confidence_weighted_combination(
                    analyst_signal, tactician_signal, additional_context
                )
            elif self.combination_method == CombinationMethod.HIERARCHICAL:
                combined_signal = await self._hierarchical_combination(
                    analyst_signal, tactician_signal, additional_context
                )
            elif self.combination_method == CombinationMethod.CONSENSUS:
                combined_signal = await self._consensus_combination(
                    analyst_signal, tactician_signal, additional_context
                )
            elif self.combination_method == CombinationMethod.RISK_ADJUSTED:
                combined_signal = await self._risk_adjusted_combination(
                    analyst_signal, tactician_signal, additional_context
                )
            else:
                # Default to weighted average
                combined_signal = await self._weighted_average_combination(
                    analyst_signal, tactician_signal, additional_context
                )
            
            if combined_signal:
                # Store combination in history
                self._store_combination(combined_signal)
                self.combination_count += 1
                
                tprint_success(f"✅ Combined signal generated: {combined_signal.action.value} "
                             f"(confidence: {combined_signal.confidence:.3f})")
                
                # Return as dictionary for compatibility
                return self._signal_to_dict(combined_signal)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Signal combination failed: {e}")
            return None

    async def _weighted_average_combination(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[CombinedSignal]:
        """Combine signals using weighted average method with shared utilities."""
        try:
            # Calculate weighted confidence using shared utilities
            analyst_confidence = analyst_signal.confidence_score if analyst_signal else 0.0
            tactician_confidence = tactician_signal.confidence_score if tactician_signal else 0.0
            
            # Use shared confidence calculator
            confidence_metrics = await self.confidence_calculator.calculate_confidence(
                base_confidence=analyst_confidence,
                enhancement_confidence=tactician_confidence,
                risk_metrics=self._extract_risk_metrics(analyst_signal, tactician_signal),
                regime_metrics=additional_context.get('regime_data') if additional_context else None,
                signal_type="both",
                additional_context=additional_context
            )
            
            # Check confidence threshold
            if confidence_metrics.final_confidence < self.weights.confidence_threshold:
                return None
            
            # Determine action
            action = self._determine_combined_action(analyst_signal, tactician_signal)
            
            # Calculate strength
            strength = self._calculate_combined_strength(analyst_signal, tactician_signal)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(analyst_signal, tactician_signal)
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(tactician_signal)
            
            return CombinedSignal(
                timestamp=datetime.now(),
                symbol=analyst_signal.symbol if analyst_signal else tactician_signal.symbol,
                action=action,
                confidence=confidence_metrics.final_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.WEIGHTED_AVERAGE,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata={
                    'combination_weights': {
                        'analyst_weight': self.weights.analyst_weight,
                        'tactician_weight': self.weights.tactician_weight
                    },
                    'confidence_metrics': confidence_metrics.confidence_components,
                    'additional_context': additional_context or {}
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Weighted average combination failed: {e}")
            return None

    async def _confidence_weighted_combination(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[CombinedSignal]:
        """Combine signals using confidence-weighted method with shared utilities."""
        try:
            # Use confidence scores as weights
            analyst_confidence = analyst_signal.confidence_score if analyst_signal else 0.0
            tactician_confidence = tactician_signal.confidence_score if tactician_signal else 0.0
            
            # Use shared confidence calculator with confidence-based weighting
            confidence_metrics = await self.confidence_calculator.calculate_confidence(
                base_confidence=analyst_confidence,
                enhancement_confidence=tactician_confidence,
                risk_metrics=self._extract_risk_metrics(analyst_signal, tactician_signal),
                regime_metrics=additional_context.get('regime_data') if additional_context else None,
                signal_type="both",
                additional_context=additional_context
            )
            
            # Check confidence threshold
            if confidence_metrics.final_confidence < self.weights.confidence_threshold:
                return None
            
            # Determine action
            action = self._determine_combined_action(analyst_signal, tactician_signal)
            
            # Calculate strength
            strength = self._calculate_combined_strength(analyst_signal, tactician_signal)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(analyst_signal, tactician_signal)
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(tactician_signal)
            
            return CombinedSignal(
                timestamp=datetime.now(),
                symbol=analyst_signal.symbol if analyst_signal else tactician_signal.symbol,
                action=action,
                confidence=confidence_metrics.final_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.CONFIDENCE_WEIGHTED,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata={
                    'confidence_weights': {
                        'analyst_confidence': analyst_confidence,
                        'tactician_confidence': tactician_confidence
                    },
                    'confidence_metrics': confidence_metrics.confidence_components,
                    'additional_context': additional_context or {}
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Confidence weighted combination failed: {e}")
            return None

    async def _hierarchical_combination(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[CombinedSignal]:
        """Combine signals using hierarchical method with shared utilities."""
        try:
            # Analyst determines the overall direction
            if not analyst_signal:
                return None
            
            # Use shared confidence calculator for hierarchical combination
            base_confidence = analyst_signal.confidence_score
            enhancement_confidence = tactician_signal.confidence_score if tactician_signal else base_confidence * 0.8
            
            confidence_metrics = await self.confidence_calculator.calculate_confidence(
                base_confidence=base_confidence,
                enhancement_confidence=enhancement_confidence,
                risk_metrics=self._extract_risk_metrics(analyst_signal, tactician_signal),
                regime_metrics=additional_context.get('regime_data') if additional_context else None,
                signal_type="hierarchical",
                additional_context=additional_context
            )
            
            # Check confidence threshold
            if confidence_metrics.final_confidence < self.weights.confidence_threshold:
                return None
            
            # Determine action
            action = self._determine_combined_action(analyst_signal, tactician_signal)
            
            # Calculate strength
            strength = self._calculate_combined_strength(analyst_signal, tactician_signal)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(analyst_signal, tactician_signal)
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(tactician_signal)
            
            return CombinedSignal(
                timestamp=datetime.now(),
                symbol=analyst_signal.symbol,
                action=action,
                confidence=confidence_metrics.final_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.HIERARCHICAL,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata={
                    'hierarchical_method': 'analyst_direction_tactician_timing',
                    'confidence_metrics': confidence_metrics.confidence_components,
                    'additional_context': additional_context or {}
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Hierarchical combination failed: {e}")
            return None

    async def _consensus_combination(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[CombinedSignal]:
        """Combine signals using consensus method with shared utilities."""
        try:
            # Both signals must be present for consensus
            if not analyst_signal or not tactician_signal:
                return None
            
            # Check if signals agree on direction
            analyst_action = self._map_analyst_signal_to_action(analyst_signal.signal_type)
            tactician_action = self._map_tactician_signal_to_action(tactician_signal.timing_signal)
            
            # Determine if there's consensus
            if analyst_action == tactician_action:
                # Consensus reached - use shared confidence calculator
                confidence_metrics = await self.confidence_calculator.calculate_confidence(
                    base_confidence=analyst_signal.confidence_score,
                    enhancement_confidence=tactician_signal.confidence_score,
                    risk_metrics=self._extract_risk_metrics(analyst_signal, tactician_signal),
                    regime_metrics=additional_context.get('regime_data') if additional_context else None,
                    signal_type="consensus",
                    additional_context=additional_context
                )
                
                action = analyst_action
            else:
                # No consensus - use fallback
                return None
            
            # Check confidence threshold
            if confidence_metrics.final_confidence < self.weights.confidence_threshold:
                return None
            
            # Calculate strength
            strength = self._calculate_combined_strength(analyst_signal, tactician_signal)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(analyst_signal, tactician_signal)
            
            # Calculate position sizing
            position_sizing = self._calculate_position_sizing(tactician_signal)
            
            return CombinedSignal(
                timestamp=datetime.now(),
                symbol=analyst_signal.symbol,
                action=action,
                confidence=confidence_metrics.final_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.CONSENSUS,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata={
                    'consensus_reached': True,
                    'analyst_action': analyst_action.value,
                    'tactician_action': tactician_action.value,
                    'confidence_metrics': confidence_metrics.confidence_components,
                    'additional_context': additional_context or {}
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Consensus combination failed: {e}")
            return None

    async def _risk_adjusted_combination(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[CombinedSignal]:
        """Combine signals using risk-adjusted method with shared utilities."""
        try:
            # Start with weighted average
            base_signal = await self._weighted_average_combination(
                analyst_signal, tactician_signal, additional_context
            )
            
            if not base_signal:
                return None
            
            # Apply risk adjustments using shared utilities
            risk_metrics = self._extract_risk_metrics(analyst_signal, tactician_signal)
            
            # Use shared confidence calculator for risk adjustment
            confidence_metrics = await self.confidence_calculator.calculate_confidence(
                base_confidence=base_signal.confidence,
                enhancement_confidence=base_signal.confidence,
                risk_metrics=risk_metrics,
                regime_metrics=additional_context.get('regime_data') if additional_context else None,
                signal_type="risk_adjusted",
                additional_context=additional_context
            )
            
            # Update signal with risk-adjusted confidence
            base_signal.confidence = confidence_metrics.final_confidence
            base_signal.combination_method = CombinationMethod.RISK_ADJUSTED
            base_signal.metadata['risk_adjustment'] = {
                'risk_adjustment_factor': self.weights.risk_adjustment_factor,
                'confidence_metrics': confidence_metrics.confidence_components,
                'risk_metrics': risk_metrics
            }
            
            return base_signal
            
        except Exception as e:
            self.logger.error(f"❌ Risk adjusted combination failed: {e}")
            return None

    def _extract_risk_metrics(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal]
    ) -> Dict[str, float]:
        """Extract risk metrics from signals."""
        try:
            risk_metrics = {}
            
            # Analyst risk metrics
            if analyst_signal:
                risk_metrics.update({
                    'market_health': analyst_signal.market_health_score,
                    'volatility': analyst_signal.volatility_score,
                    'liquidation_risk': analyst_signal.liquidation_risk_score
                })
            
            # Tactician risk metrics
            if tactician_signal:
                risk_metrics.update(tactician_signal.risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Risk metrics extraction failed: {e}")
            return {}

    def _determine_combined_action(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal]
    ) -> CombinedAction:
        """Determine combined action from both signals."""
        try:
            # Get actions from both signals
            analyst_action = self._map_analyst_signal_to_action(
                analyst_signal.signal_type if analyst_signal else SignalType.HOLD
            )
            tactician_action = self._map_tactician_signal_to_action(
                tactician_signal.timing_signal if tactician_signal else TimingSignal.HOLD
            )
            
            # Combine actions based on priority
            if analyst_action == CombinedAction.BUY and tactician_action in [CombinedAction.BUY, CombinedAction.HOLD]:
                return CombinedAction.BUY
            elif analyst_action == CombinedAction.SELL and tactician_action in [CombinedAction.SELL, CombinedAction.HOLD]:
                return CombinedAction.SELL
            elif analyst_action == CombinedAction.HOLD and tactician_action == CombinedAction.CLOSE:
                return CombinedAction.CLOSE
            else:
                return CombinedAction.HOLD
                
        except Exception as e:
            self.logger.error(f"❌ Action determination failed: {e}")
            return CombinedAction.HOLD

    def _map_analyst_signal_to_action(self, signal_type: SignalType) -> CombinedAction:
        """Map analyst signal type to combined action."""
        mapping = {
            SignalType.BUY: CombinedAction.BUY,
            SignalType.SELL: CombinedAction.SELL,
            SignalType.HOLD: CombinedAction.HOLD,
            SignalType.CLOSE: CombinedAction.CLOSE
        }
        return mapping.get(signal_type, CombinedAction.HOLD)

    def _map_tactician_signal_to_action(self, timing_signal: TimingSignal) -> CombinedAction:
        """Map tactician timing signal to combined action."""
        mapping = {
            TimingSignal.ENTER_LONG: CombinedAction.BUY,
            TimingSignal.ENTER_SHORT: CombinedAction.SELL,
            TimingSignal.EXIT_LONG: CombinedAction.CLOSE,
            TimingSignal.EXIT_SHORT: CombinedAction.CLOSE,
            TimingSignal.HOLD: CombinedAction.HOLD,
            TimingSignal.CLOSE_ALL: CombinedAction.CLOSE
        }
        return mapping.get(timing_signal, CombinedAction.HOLD)

    def _calculate_combined_strength(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal]
    ) -> float:
        """Calculate combined signal strength."""
        try:
            analyst_strength = analyst_signal.signal_strength.value if analyst_signal else 0.0
            tactician_strength = tactician_signal.confidence.value if tactician_signal else 0.0
            
            # Map strength values to numeric
            strength_mapping = {
                'weak': 0.25,
                'low': 0.25,
                'moderate': 0.5,
                'medium': 0.5,
                'strong': 0.75,
                'high': 0.75,
                'very_strong': 1.0,
                'very_high': 1.0
            }
            
            analyst_numeric = strength_mapping.get(analyst_strength, 0.5)
            tactician_numeric = strength_mapping.get(tactician_strength, 0.5)
            
            # Weighted average
            analyst_weight = self.weights.analyst_weight if analyst_signal else 0.0
            tactician_weight = self.weights.tactician_weight if tactician_signal else 0.0
            
            total_weight = analyst_weight + tactician_weight
            if total_weight == 0:
                return 0.5
            
            return (analyst_numeric * analyst_weight + tactician_numeric * tactician_weight) / total_weight
            
        except Exception as e:
            self.logger.error(f"❌ Strength calculation failed: {e}")
            return 0.5

    def _calculate_risk_metrics(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal]
    ) -> Dict[str, float]:
        """Calculate combined risk metrics."""
        try:
            risk_metrics = {}
            
            # Analyst risk metrics
            if analyst_signal:
                risk_metrics.update({
                    'market_health': analyst_signal.market_health_score,
                    'volatility': analyst_signal.volatility_score,
                    'liquidation_risk': analyst_signal.liquidation_risk_score
                })
            
            # Tactician risk metrics
            if tactician_signal:
                risk_metrics.update(tactician_signal.risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Risk metrics calculation failed: {e}")
            return {}

    def _calculate_position_sizing(self, tactician_signal: Optional[TacticianSignal]) -> Dict[str, float]:
        """Calculate position sizing from tactician signal."""
        try:
            if not tactician_signal or not tactician_signal.position_sizing:
                return {
                    'recommended_size': 0.0,
                    'max_size': 0.0,
                    'leverage': 1.0,
                    'risk_per_trade': 0.02
                }
            
            sizing = tactician_signal.position_sizing
            return {
                'recommended_size': sizing.recommended_size,
                'max_size': sizing.max_size,
                'leverage': sizing.leverage,
                'risk_per_trade': sizing.risk_per_trade,
                'kelly_fraction': sizing.kelly_fraction,
                'confidence_multiplier': sizing.confidence_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"❌ Position sizing calculation failed: {e}")
            return {
                'recommended_size': 0.0,
                'max_size': 0.0,
                'leverage': 1.0,
                'risk_per_trade': 0.02
            }

    def _signal_to_dict(self, signal: CombinedSignal) -> Dict[str, Any]:
        """Convert combined signal to dictionary."""
        return {
            'action': signal.action.value,
            'confidence': signal.confidence,
            'strength': signal.strength,
            'symbol': signal.symbol,
            'timestamp': signal.timestamp.isoformat(),
            'combination_method': signal.combination_method.value,
            'risk_metrics': signal.risk_metrics,
            'position_sizing': signal.position_sizing,
            'metadata': signal.metadata
        }

    def _store_combination(self, signal: CombinedSignal):
        """Store combination in history."""
        self.combination_history.append(signal)
        
        # Maintain history size
        if len(self.combination_history) > self.max_history:
            self.combination_history.pop(0)

    def get_combination_history(self, n: int = 100) -> List[CombinedSignal]:
        """Get recent combination history."""
        return self.combination_history[-n:] if len(self.combination_history) >= n else self.combination_history.copy()

    def get_combination_stats(self) -> Dict[str, Any]:
        """Get combination statistics."""
        if self.combination_count == 0:
            return {
                'total_combinations': 0,
                'success_rate': 0.0,
                'action_distribution': {},
                'avg_confidence': 0.0
            }
        
        # Calculate action distribution
        action_distribution = {}
        for signal in self.combination_history:
            action = signal.action.value
            action_distribution[action] = action_distribution.get(action, 0) + 1
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in self.combination_history])
        
        return {
            'total_combinations': self.combination_count,
            'success_rate': self.successful_combinations / self.combination_count if self.combination_count > 0 else 0.0,
            'action_distribution': action_distribution,
            'avg_confidence': avg_confidence,
            'recent_combinations': len(self.combination_history),
            'shared_utilities_stats': {
                'confidence_calculator': self.confidence_calculator.get_performance_metrics(),
                'fallback_analyzer': self.fallback_analyzer.get_performance_metrics()
            }
        }

    def update_combination_performance(self, signal: CombinedSignal, was_successful: bool):
        """Update combination performance tracking."""
        if was_successful:
            self.successful_combinations += 1
        else:
            self.failed_combinations += 1

# Convenience functions
def create_signal_combiner(config: Dict[str, Any]) -> SignalCombiner:
    """Create a configured signal combiner."""
    return SignalCombiner(config)

async def combine_signals(
    signal_combiner: SignalCombiner,
    analyst_signal: Optional[AnalystSignal] = None,
    tactician_signal: Optional[TacticianSignal] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Combine signals with convenience function."""
    return await signal_combiner.combine_signals(
        analyst_signal=analyst_signal,
        tactician_signal=tactician_signal,
        additional_context=additional_context
    )