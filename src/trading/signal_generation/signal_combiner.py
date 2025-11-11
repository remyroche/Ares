"""
Signal Combiner

This module combines signals from Analyst and Tactician components
to generate final trading decisions with proper weighting and risk management.
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
from src.utils.tprint import tprint

# Import signal types
from .analyst_signals import AnalystSignal, SignalType, SignalStrength
from .tactician_signals import TacticianSignal, TimingSignal, TimingConfidence

logger = system_logger.getChild('SignalCombiner')

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
    Signal Combiner that combines Analyst and Tactician signals
    to generate final trading decisions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the signal combiner.

        Args:
            config: Configuration dictionary
        """
        tprint(f"[SIGNAL_COMBINER] __init__: Initializing signal combiner")
        self.config = config
        self.logger = logger.getChild('SignalCombiner')

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

        # Optional gated stacker for expert combination
        self.stacker = config.get('stacker_model')
        self.stacker_artifacts = config.get('stacker_artifacts', {})
        self.stacker_ready = False
        self._initialise_stacker_from_artifacts()
        tprint(f"[SIGNAL_COMBINER] __init__ -> initialized (method={self.combination_method.value}, confidence_threshold={self.weights.confidence_threshold})")

    async def initialize(self) -> bool:
        """
        Initialize the signal combiner.

        Returns:
            bool: True if initialization successful
        """
        tprint(f"[SIGNAL_COMBINER] initialize: Initializing signal combiner")
        try:
            self.logger.info("✅ Signal Combiner initialized")
            tprint(f"[SIGNAL_COMBINER] initialize -> True")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Signal Combiner: {e}")
            tprint(f"[SIGNAL_COMBINER] initialize -> False (error: {e})", color="red")
            return False

    def _initialise_stacker_from_artifacts(self) -> None:
        """Load gating/calibration state for the stacker if provided."""

        if not self.stacker:
            self.stacker_ready = False
            return

        gating_state = None
        calibration_state = None
        if isinstance(self.stacker_artifacts, dict):
            gating_state = (
                self.stacker_artifacts.get('stacker_gating_state')
                or self.stacker_artifacts.get('gating_state')
            )
            calibration_state = (
                self.stacker_artifacts.get('stacker_calibration_state')
                or self.stacker_artifacts.get('calibration_state')
            )

        try:
            if gating_state and hasattr(self.stacker, 'load_gating_state'):
                self.stacker.load_gating_state(gating_state)
            if calibration_state and hasattr(self.stacker, 'load_calibration_state'):
                self.stacker.load_calibration_state(calibration_state)
        except Exception as exc:
            self.logger.warning(
                "⚠️ Failed to hydrate stacker gating/calibration state: %s", exc
            )

        self.stacker_ready = bool(getattr(self.stacker, 'fitted', False))

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
        Combine Analyst and Tactician signals.

        Args:
            analyst_signal: Signal from Analyst component
            tactician_signal: Signal from Tactician component
            additional_context: Additional context for combination

        Returns:
            Combined signal result or None if no valid combination
        """
        tprint(f"[SIGNAL_COMBINER] combine_signals: has_analyst={analyst_signal is not None}, has_tactician={tactician_signal is not None}, method={self.combination_method.value}")
        try:
            tprint_info("🔄 Combining Analyst and Tactician signals...")

            # Validate inputs
            if not analyst_signal and not tactician_signal:
                tprint_warning("⚠️ No signals provided for combination")
                tprint(f"[SIGNAL_COMBINER] combine_signals -> None (no signals)")
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
                result = self._signal_to_dict(combined_signal)
                tprint(f"[SIGNAL_COMBINER] combine_signals -> {combined_signal.action.value} (confidence={combined_signal.confidence:.3f}, strength={combined_signal.strength:.3f})")
                return result

            tprint(f"[SIGNAL_COMBINER] combine_signals -> None (no combined signal)")
            return None

        except Exception as e:
            self.logger.error(f"❌ Signal combination failed: {e}")
            tprint(f"[SIGNAL_COMBINER] combine_signals -> ERROR: {e}", color="red")
            return None

    async def _weighted_average_combination(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[CombinedSignal]:
        """Combine signals using weighted average method."""
        tprint(f"[SIGNAL_COMBINER] _weighted_average_combination: combining signals")
        try:
            analyst_confidence = analyst_signal.confidence_score if analyst_signal else 0.0
            tactician_confidence = tactician_signal.confidence_score if tactician_signal else 0.0

            analyst_weight = self.weights.analyst_weight if analyst_signal else 0.0
            tactician_weight = self.weights.tactician_weight if tactician_signal else 0.0

            tprint(f"[SIGNAL_COMBINER] _weighted_average_combination: analyst_conf={analyst_confidence:.3f}, tactician_conf={tactician_confidence:.3f}")

            gated_output = self._compute_gated_output(
                analyst_signal, tactician_signal, additional_context
            )

            metadata: Dict[str, Any] = {'additional_context': additional_context or {}}
            combined_utility: Optional[float] = None

            if gated_output:
                weights_dict = gated_output['weights']
                analyst_weight = float(weights_dict.get('analyst', [analyst_weight])[0])
                tactician_weight = float(weights_dict.get('tactician', [tactician_weight])[0])
                total_weight = analyst_weight + tactician_weight
                if total_weight > 0:
                    analyst_weight /= total_weight
                    tactician_weight /= total_weight
                combined_confidence = float(gated_output['probability'][0])
                combined_raw_probability = float(gated_output['raw_probability'][0])
                if gated_output.get('utility') is not None:
                    combined_utility = float(gated_output['utility'][0])
                metadata['gated'] = {
                    'probability': combined_confidence,
                    'raw_probability': combined_raw_probability,
                    'weights': {
                        name: float(value[0]) if isinstance(value, np.ndarray) else float(value)
                        for name, value in weights_dict.items()
                    },
                    'expert_probabilities': {
                        name: float(values[0]) if isinstance(values, np.ndarray) else float(values)
                        for name, values in gated_output['expert_probabilities'].items()
                    },
                    'utility': combined_utility,
                }
            else:
                total_weight = analyst_weight + tactician_weight
                if total_weight == 0:
                    return None
                analyst_weight /= total_weight
                tactician_weight /= total_weight
                combined_confidence = (
                    analyst_confidence * analyst_weight
                    + tactician_confidence * tactician_weight
                )

            if combined_confidence < self.weights.confidence_threshold:
                tprint(f"[SIGNAL_COMBINER] _weighted_average_combination -> None (confidence {combined_confidence:.3f} below threshold {self.weights.confidence_threshold:.3f})")
                return None

            action = self._determine_combined_action(analyst_signal, tactician_signal)
            strength = self._calculate_combined_strength(analyst_signal, tactician_signal)
            risk_metrics = self._calculate_risk_metrics(analyst_signal, tactician_signal)
            tprint(f"[SIGNAL_COMBINER] _weighted_average_combination: action={action.value}, strength={strength:.3f}, confidence={combined_confidence:.3f}")
            if combined_utility is not None:
                risk_metrics = dict(risk_metrics or {})
                risk_metrics['combined_utility'] = combined_utility

            position_sizing = self._calculate_position_sizing(tactician_signal)

            metadata['combination_weights'] = {
                'analyst_weight': analyst_weight,
                'tactician_weight': tactician_weight,
            }
            if combined_utility is not None:
                metadata['combined_utility'] = combined_utility
            metadata['combined_confidence'] = combined_confidence

            signal = CombinedSignal(
                timestamp=datetime.now(),
                symbol=analyst_signal.symbol if analyst_signal else tactician_signal.symbol,
                action=action,
                confidence=combined_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.WEIGHTED_AVERAGE,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata=metadata,
            )
            tprint(f"[SIGNAL_COMBINER] _weighted_average_combination -> {action.value} (confidence={combined_confidence:.3f})")
            return signal

        except Exception as e:
            self.logger.error(f"❌ Weighted average combination failed: {e}")
            tprint(f"[SIGNAL_COMBINER] _weighted_average_combination -> ERROR: {e}", color="red")
            return None

    async def _confidence_weighted_combination(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[CombinedSignal]:
        """Combine signals using confidence-weighted method."""
        try:
            # Use confidence scores as weights
            analyst_confidence = analyst_signal.confidence_score if analyst_signal else 0.0
            tactician_confidence = tactician_signal.confidence_score if tactician_signal else 0.0

            total_confidence = analyst_confidence + tactician_confidence
            if total_confidence == 0:
                return None

            # Normalize confidence scores to weights
            analyst_weight = analyst_confidence / total_confidence
            tactician_weight = tactician_confidence / total_confidence

            # Combined confidence is the weighted average
            combined_confidence = (
                analyst_confidence * analyst_weight +
                tactician_confidence * tactician_weight
            )

            # Check confidence threshold
            if combined_confidence < self.weights.confidence_threshold:
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
                confidence=combined_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.CONFIDENCE_WEIGHTED,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata={
                    'confidence_weights': {
                        'analyst_weight': analyst_weight,
                        'tactician_weight': tactician_weight
                    },
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
        """Combine signals using hierarchical method (Analyst decides direction, Tactician decides timing)."""
        try:
            # Analyst determines the overall direction
            if not analyst_signal:
                return None

            # Tactician provides timing and position sizing
            if not tactician_signal:
                # Use analyst signal with default timing
                combined_confidence = analyst_signal.confidence_score * 0.8  # Reduce confidence without tactician
                action = self._map_analyst_signal_to_action(analyst_signal.signal_type)
            else:
                # Combine with tactician timing
                combined_confidence = min(analyst_signal.confidence_score, tactician_signal.confidence_score)
                action = self._determine_combined_action(analyst_signal, tactician_signal)

            # Check confidence threshold
            if combined_confidence < self.weights.confidence_threshold:
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
                confidence=combined_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.HIERARCHICAL,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata={
                    'hierarchical_method': 'analyst_direction_tactician_timing',
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
        """Combine signals using consensus method (both must agree)."""
        try:
            # Both signals must be present for consensus
            if not analyst_signal or not tactician_signal:
                return None

            # Check if signals agree on direction
            analyst_action = self._map_analyst_signal_to_action(analyst_signal.signal_type)
            tactician_action = self._map_tactician_signal_to_action(tactician_signal.timing_signal)

            # Determine if there's consensus
            if analyst_action == tactician_action:
                # Consensus reached
                combined_confidence = (analyst_signal.confidence_score + tactician_signal.confidence_score) / 2
                action = analyst_action
            else:
                # No consensus - default to hold
                combined_confidence = 0.3  # Low confidence for no consensus
                action = CombinedAction.HOLD

            # Check confidence threshold
            if combined_confidence < self.weights.confidence_threshold:
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
                confidence=combined_confidence,
                strength=strength,
                analyst_signal=analyst_signal,
                tactician_signal=tactician_signal,
                combination_method=CombinationMethod.CONSENSUS,
                risk_metrics=risk_metrics,
                position_sizing=position_sizing,
                metadata={
                    'consensus_reached': analyst_action == tactician_action,
                    'analyst_action': analyst_action.value,
                    'tactician_action': tactician_action.value,
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
        """Combine signals using risk-adjusted method."""
        try:
            # Start with weighted average
            base_signal = await self._weighted_average_combination(
                analyst_signal, tactician_signal, additional_context
            )

            if not base_signal:
                return None

            # Apply risk adjustments
            risk_adjustment = self.weights.risk_adjustment_factor

            # Adjust confidence based on risk metrics
            if base_signal.risk_metrics:
                volatility_risk = base_signal.risk_metrics.get('volatility', 0.02)
                liquidation_risk = base_signal.risk_metrics.get('liquidation_risk', 0.1)

                # Reduce confidence for high risk
                risk_factor = 1.0 - (volatility_risk * 2 + liquidation_risk * 0.5)
                risk_factor = max(0.1, min(1.0, risk_factor))

                base_signal.confidence *= risk_factor * risk_adjustment

            # Check confidence threshold after risk adjustment
            if base_signal.confidence < self.weights.confidence_threshold:
                return None

            # Update metadata
            base_signal.combination_method = CombinationMethod.RISK_ADJUSTED
            base_signal.metadata['risk_adjustment'] = {
                'risk_adjustment_factor': risk_adjustment,
                'volatility_risk': base_signal.risk_metrics.get('volatility', 0.0),
                'liquidation_risk': base_signal.risk_metrics.get('liquidation_risk', 0.0)
            }

            return base_signal

        except Exception as e:
            self.logger.error(f"❌ Risk adjusted combination failed: {e}")
            return None

    def _determine_combined_action(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal]
    ) -> CombinedAction:
        """Determine combined action from both signals."""
        tprint(f"[SIGNAL_COMBINER] _determine_combined_action: determining action")
        try:
            # Get actions from both signals
            analyst_action = self._map_analyst_signal_to_action(
                analyst_signal.signal_type if analyst_signal else SignalType.HOLD
            )
            tactician_action = self._map_tactician_signal_to_action(
                tactician_signal.timing_signal if tactician_signal else TimingSignal.HOLD
            )

            tprint(f"[SIGNAL_COMBINER] _determine_combined_action: analyst_action={analyst_action.value}, tactician_action={tactician_action.value}")

            # Combine actions based on priority
            if analyst_action == CombinedAction.BUY and tactician_action in [CombinedAction.BUY, CombinedAction.HOLD]:
                result = CombinedAction.BUY
            elif analyst_action == CombinedAction.SELL and tactician_action in [CombinedAction.SELL, CombinedAction.HOLD]:
                result = CombinedAction.SELL
            elif analyst_action == CombinedAction.HOLD and tactician_action == CombinedAction.CLOSE:
                result = CombinedAction.CLOSE
            else:
                result = CombinedAction.HOLD

            tprint(f"[SIGNAL_COMBINER] _determine_combined_action -> {result.value}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Action determination failed: {e}")
            tprint(f"[SIGNAL_COMBINER] _determine_combined_action -> HOLD (error: {e})", color="red")
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

    def _compute_gated_output(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.stacker or not self.stacker_ready:
            return None
        if analyst_signal is None or tactician_signal is None:
            return None

        base_predictions = self._prepare_base_predictions_for_gater(
            analyst_signal, tactician_signal
        )
        if not base_predictions:
            return None

        regime_features = self._assemble_regime_features_for_gater(
            analyst_signal, tactician_signal, additional_context
        )

        try:
            return self.stacker.combine_outputs(base_predictions, regime_features)
        except Exception as exc:
            self.logger.warning("⚠️ Failed to evaluate gated stacker: %s", exc)
            return None

    def _prepare_base_predictions_for_gater(
        self,
        analyst_signal: AnalystSignal,
        tactician_signal: TacticianSignal,
    ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        analyst_prob = self._extract_probability_from_signal(analyst_signal)
        tactician_prob = self._extract_probability_from_signal(tactician_signal)
        if analyst_prob is None or tactician_prob is None:
            return None

        predictions: Dict[str, Dict[str, np.ndarray]] = {
            'analyst': {'probability': np.array([analyst_prob], dtype=float)},
            'tactician': {'probability': np.array([tactician_prob], dtype=float)},
        }

        analyst_utility = self._extract_utility_from_signal(analyst_signal)
        if analyst_utility is not None:
            predictions['analyst']['utility'] = np.array([analyst_utility], dtype=float)

        tactician_utility = self._extract_utility_from_signal(tactician_signal)
        if tactician_utility is not None:
            predictions['tactician']['utility'] = np.array([tactician_utility], dtype=float)

        return predictions

    def _extract_probability_from_signal(self, signal: Any) -> Optional[float]:
        if signal is None:
            return None

        candidates: List[Any] = []
        if hasattr(signal, 'metadata') and isinstance(signal.metadata, dict):
            metadata = signal.metadata
            candidates.extend(
                metadata.get(key)
                for key in (
                    'probability',
                    'meta_probability',
                    'stacker_probability',
                    'confidence_score',
                )
            )
        if hasattr(signal, 'ml_predictions') and isinstance(signal.ml_predictions, dict):
            ml_preds = signal.ml_predictions
            candidates.extend(
                ml_preds.get(key)
                for key in ('probability', 'meta_probability', 'stacker_probability')
            )
        if hasattr(signal, 'confidence_score'):
            candidates.append(getattr(signal, 'confidence_score'))

        for value in candidates:
            prob = self._safe_probability(value)
            if prob is not None:
                return prob
        return None

    def _extract_utility_from_signal(self, signal: Any) -> Optional[float]:
        if signal is None:
            return None
        sources: List[Any] = []
        if hasattr(signal, 'metadata') and isinstance(signal.metadata, dict):
            metadata = signal.metadata
            sources.extend(
                metadata.get(key)
                for key in ('utility', 'expected_utility', 'expected_reward', 'reward')
            )
        if hasattr(signal, 'risk_metrics') and isinstance(signal.risk_metrics, dict):
            sources.append(signal.risk_metrics.get('expected_utility'))
        for value in sources:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _assemble_regime_features_for_gater(
        self,
        analyst_signal: Optional[AnalystSignal],
        tactician_signal: Optional[TacticianSignal],
        additional_context: Optional[Dict[str, Any]],
    ) -> Dict[str, np.ndarray]:
        required = []
        if self.stacker and hasattr(self.stacker, 'regime_feature_names'):
            required = list(self.stacker.regime_feature_names)

        contexts: List[Dict[str, Any]] = []
        if additional_context:
            contexts.append(additional_context)
        if analyst_signal and isinstance(analyst_signal.metadata, dict):
            contexts.append(analyst_signal.metadata)
        if tactician_signal and isinstance(tactician_signal.metadata, dict):
            contexts.append(tactician_signal.metadata)

        feature_map: Dict[str, np.ndarray] = {}

        volatility = self._resolve_context_value(
            contexts,
            'volatility_level',
            fallback=getattr(analyst_signal, 'volatility_score', None),
        )
        # Trend signal primarily uses the `trend_score` metadata key with graceful fallbacks.
        trend = self._resolve_context_value(
            contexts,
            'trend_score',
            fallback=getattr(analyst_signal, 'market_health_score', None),
        )
        liquidity = self._resolve_context_value(
            contexts,
            'liquidity_z',
            fallback=(
                getattr(tactician_signal, 'risk_metrics', {}).get('liquidity_z')
                if tactician_signal and isinstance(tactician_signal.risk_metrics, dict)
                else None
            ),
        )

        feature_map['volatility_level'] = np.array([
            self._safe_float(volatility)
        ])
        feature_map['trend_score'] = np.array([
            self._safe_float(trend)
        ])
        feature_map['liquidity_z'] = np.array([
            self._safe_float(liquidity)
        ])

        for name in required:
            if name not in feature_map:
                value = self._resolve_context_value(contexts, name)
                feature_map[name] = np.array([self._safe_float(value)])

        return feature_map

    def _resolve_context_value(
        self,
        contexts: List[Dict[str, Any]],
        key: str,
        fallback: Any = None,
    ) -> Any:
        for context in contexts:
            if key in context and context[key] is not None:
                return context[key]
        return fallback

    def _safe_probability(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            prob = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(prob):
            return None
        if prob > 1.0:
            prob = prob / 100.0 if prob <= 100.0 else 1.0
        if prob < 0.0:
            prob = 0.0
        return max(0.0, min(1.0, prob))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _signal_to_dict(self, signal: CombinedSignal) -> Dict[str, Any]:
        """Convert combined signal to dictionary."""
        tprint(f"[SIGNAL_COMBINER] _signal_to_dict: converting signal to dict")
        result = {
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
        tprint(f"[SIGNAL_COMBINER] _signal_to_dict -> dict with {len(result)} keys")
        return result

    def _store_combination(self, signal: CombinedSignal):
        """Store combination in history."""
        tprint(f"[SIGNAL_COMBINER] _store_combination: storing combination (history_size={len(self.combination_history)})")
        self.combination_history.append(signal)

        # Maintain history size
        if len(self.combination_history) > self.max_history:
            self.combination_history.pop(0)
        tprint(f"[SIGNAL_COMBINER] _store_combination -> stored (new_size={len(self.combination_history)})")

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
            'recent_combinations': len(self.combination_history)
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
    tprint(f"[SIGNAL_COMBINER] create_signal_combiner: Creating signal combiner")
    combiner = SignalCombiner(config)
    tprint(f"[SIGNAL_COMBINER] create_signal_combiner -> created")
    return combiner

async def combine_signals(
    signal_combiner: SignalCombiner,
    analyst_signal: Optional[AnalystSignal] = None,
    tactician_signal: Optional[TacticianSignal] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Combine signals with convenience function."""
    tprint(f"[SIGNAL_COMBINER] combine_signals: calling combiner.combine_signals()")
    result = await signal_combiner.combine_signals(
        analyst_signal=analyst_signal,
        tactician_signal=tactician_signal,
        additional_context=additional_context
    )
    tprint(f"[SIGNAL_COMBINER] combine_signals -> {result['action'] if result else None}")
    return result
