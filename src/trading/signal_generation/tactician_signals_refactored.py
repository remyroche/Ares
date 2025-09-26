"""
Tactician Signal Generation (Refactored)

This module integrates the Tactician component to generate timing signals
using shared utilities for feature engineering, confidence calculation,
and fallback analysis.
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
from ..utils.feature_engineering import UnifiedFeatureEngine, FeatureSet
from ..utils.confidence_calculator import UnifiedConfidenceCalculator, ConfidenceMetrics
from ..utils.fallback_analyzer import UnifiedFallbackAnalyzer, FallbackAnalysisResult
from ..utils.signal_enhancer_base import BaseSignalEnhancer, EnhancementResult

# Import TAS components for enhanced signal generation
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_regime_detector import (
    EnhancedTASRegimeDetector, EnhancedTASResult
)
from src.training.steps.market_analysis.tas_regime.core.tas_config import TASConfig

logger = system_logger.getChild('TacticianSignalsRefactored')

class TimingSignal(Enum):
    """Types of timing signals."""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    CLOSE_ALL = "close_all"

class TimingConfidence(Enum):
    """Timing confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class PositionSizing:
    """Position sizing information."""
    recommended_size: float
    max_size: float
    leverage: float
    risk_per_trade: float
    kelly_fraction: float
    confidence_multiplier: float

@dataclass
class TacticianSignal:
    """Tactician-generated timing signal with TAS enhancement."""
    timestamp: datetime
    symbol: str
    timing_signal: TimingSignal
    confidence: TimingConfidence
    confidence_score: float
    position_sizing: PositionSizing
    scenario_predictions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    timing_indicators: Dict[str, float] = field(default_factory=dict)
    # TAS enhancement fields
    tas_prediction: Optional[Dict[str, Any]] = None
    tas_confidence: float = 0.0
    tas_architecture_type: Optional[str] = None
    signal_type: Optional[int] = None  # Analyst signal type (long/short)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TASSignalEnhancer(BaseSignalEnhancer):
    """
    TAS-specific signal enhancer that extends the base signal enhancer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TAS signal enhancer."""
        super().__init__("tas", config)
        self.tas_engine = None
        self.tas_models = {}
        self.tas_architectures = {}
        self.max_model_contributions = int(self.config.get('max_model_contributions', 3))

    async def _load_enhancement_models(self, models: Optional[Dict[str, Any]] = None) -> bool:
        """Load TAS enhancement models."""
        try:
            if models:
                self.tas_models = models
                self.max_model_contributions = int(self.config.get('max_model_contributions', 3))
                self.logger.info(f"✅ Loaded {len(models)} TAS models")
                return True
            
            # Initialize TAS engine
            tas_config = TASConfig(
                n_regimes=8,
                primary_timeframe=self.config.get('tas_timeframe', '1m'),
                enable_tree_ensemble=True,
                enable_boosted_trees=True,
                enable_random_forest=True,
                population_size=30,
                generations=50
            )
            
            self.tas_engine = EnhancedTASRegimeDetector(tas_config)
            self.logger.info("✅ TAS engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load TAS models: {e}")
            return False
    
    async def _generate_enhancement_prediction(
        self,
        features: FeatureSet,
        market_data: pd.DataFrame,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate TAS enhancement prediction."""
        try:
            if not self.tas_engine:
                return None
            
            # Prepare features for TAS prediction
            feature_vector = self.feature_engine.get_feature_vector(features)
            
            # Get signal type from context
            signal_type = additional_context.get('signal_type', 0) if additional_context else 0
            
            # Use TAS model for this signal type if available
            if signal_type in self.tas_models:
                tas_model = self.tas_models[signal_type]

                candidate_predictions = self._collect_candidate_predictions(
                    tas_model,
                    feature_vector.reshape(1, -1)
                )

                tas_result = self.tas_engine.search(
                    train_data=(feature_vector.reshape(1, -1), np.array([0])),
                    validation_data=(feature_vector.reshape(1, -1), np.array([0])),
                    regime_data={'analyst_signals': [signal_type]}
                ) if self.tas_engine else None

                if candidate_predictions:
                    confidences = [pred['confidence'] for pred in candidate_predictions if pred.get('confidence') is not None]
                    combined_confidence = float(np.clip(np.mean(confidences), 0.0, 1.0)) if confidences else 0.5

                    return {
                        'tas_prediction': {
                            'model_contributions': candidate_predictions,
                            'aggregate_prediction': self._aggregate_candidate_predictions(candidate_predictions)
                        },
                        'confidence': combined_confidence,
                        'architecture': getattr(tas_result, 'best_architecture', {}),
                        'signal_type': signal_type,
                        'contribution': 'timing_signals'
                    }

                if tas_result and tas_result.best_score > 0:
                    return {
                        'tas_prediction': tas_result.best_prediction,
                        'confidence': tas_result.best_score,
                        'architecture': tas_result.best_architecture,
                        'signal_type': signal_type,
                        'contribution': 'timing_signals'
                    }

            return None

        except Exception as e:
            self.logger.error(f"❌ TAS prediction failed: {e}")
            return None

    def _collect_candidate_predictions(self, model_container: Any, feature_vector: np.ndarray) -> List[Dict[str, Any]]:
        """Collect predictions from top TAS models for stacked blending."""
        predictions: List[Dict[str, Any]] = []

        for model_name, candidate in self._resolve_model_candidates(model_container):
            if len(predictions) >= self.max_model_contributions:
                break

            try:
                prediction = self._invoke_candidate_model(model_name, candidate, feature_vector)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug(f"⚠️ TAS candidate {model_name} failed: {exc}")
                continue

            if prediction:
                predictions.append(prediction)

        return predictions

    def _resolve_model_candidates(self, model_container: Any) -> List[Tuple[str, Any]]:
        """Resolve candidate TAS models from stored containers."""
        candidates: List[Tuple[str, Any]] = []

        if isinstance(model_container, dict):
            for key in ("model_candidates", "models", "top_models", "ensemble"):
                value = model_container.get(key)
                candidates.extend(self._normalize_candidate_collection(value))

            if "model" in model_container:
                candidates.append((model_container.get("name", "primary"), model_container["model"]))
        elif isinstance(model_container, (list, tuple)):
            for idx, candidate in enumerate(model_container):
                candidates.append((getattr(candidate, "name", f"model_{idx}"), candidate))
        else:
            if hasattr(model_container, "top_models"):
                candidates.extend(self._normalize_candidate_collection(getattr(model_container, "top_models")))

            candidates.append((getattr(model_container, "name", "primary"), model_container))

        return candidates[: self.max_model_contributions]

    def _normalize_candidate_collection(self, value: Any) -> List[Tuple[str, Any]]:
        """Normalize candidate containers to (name, model) tuples."""
        if not value:
            return []

        if isinstance(value, dict):
            return [(str(name), model) for name, model in value.items()]

        if isinstance(value, (list, tuple)):
            return [(getattr(model, "name", f"model_{idx}"), model) for idx, model in enumerate(value)]

        return [(getattr(value, "name", "candidate"), value)]

    def _invoke_candidate_model(self, model_name: str, model: Any, feature_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """Execute a candidate TAS model and standardize its output."""
        prediction_output: Any = None
        confidence: Optional[float] = None

        if isinstance(model, dict):
            prediction_output = model.get("prediction") or model.get("output")
            confidence = model.get("confidence") or model.get("score")
        else:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_vector)
                prediction_output = proba.tolist() if hasattr(proba, "tolist") else proba
                try:
                    confidence = float(np.max(proba))
                except Exception:  # pragma: no cover - safe guard
                    confidence = None
            elif hasattr(model, "predict"):
                pred = model.predict(feature_vector)
                prediction_output = pred.tolist() if hasattr(pred, "tolist") else pred
                if isinstance(pred, np.ndarray):
                    confidence = float(np.clip(np.mean(np.abs(pred)), 0.0, 1.0))
            elif callable(model):
                pred = model(feature_vector)
                prediction_output = pred.tolist() if hasattr(pred, "tolist") else pred

        if confidence is None:
            confidence = 0.5

        return {
            'model_name': model_name,
            'prediction': prediction_output,
            'confidence': float(confidence)
        }

    def _aggregate_candidate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate candidate predictions into a compact summary."""
        if not predictions:
            return {}

        confidences = [pred.get('confidence', 0.0) for pred in predictions]
        aggregate_confidence = float(np.clip(np.mean(confidences), 0.0, 1.0)) if confidences else 0.0

        return {
            'aggregate_confidence': aggregate_confidence,
            'model_names': [pred.get('model_name') for pred in predictions]
        }

class TacticianSignalGenerator:
    """
    Refactored Tactician Signal Generator using shared utilities.
    
    Integrates with the Tactician component and TAS for enhanced timing signal generation,
    using shared feature engineering, confidence calculation, and fallback analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tactician signal generator with shared utilities.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('TacticianSignalGenerator')
        
        # Tactician component (will be injected)
        self.tactician = None
        
        # Initialize shared utilities
        self.feature_engine = UnifiedFeatureEngine(config.get('feature_config', {}))
        self.confidence_calculator = UnifiedConfidenceCalculator(config.get('confidence_config', {}))
        self.fallback_analyzer = UnifiedFallbackAnalyzer(config.get('fallback_config', {}))
        
        # TAS signal enhancer
        self.tas_enhancer = TASSignalEnhancer(config.get('tas_config', {}))
        
        # Signal generation parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.tas_confidence_threshold = config.get('tas_confidence_threshold', 0.7)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        self.max_leverage = config.get('max_leverage', 3.0)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Conservative Kelly
        
        # Confidence thresholds
        self.confidence_thresholds = {
            TimingConfidence.LOW: 0.5,
            TimingConfidence.MEDIUM: 0.65,
            TimingConfidence.HIGH: 0.8,
            TimingConfidence.VERY_HIGH: 0.9
        }
        
        # TAS configuration
        self.enable_tas_enhancement = config.get('enable_tas_enhancement', True)
        self.tas_timeframe = config.get('tas_timeframe', '1m')
        
        # Signal history
        self.signal_history: List[TacticianSignal] = []
        self.max_history = config.get('max_history', 1000)
        
        # Performance tracking
        self.signal_count = 0
        self.successful_signals = 0
        self.failed_signals = 0
        self.tas_enhanced_signals = 0

    async def initialize(self, tactician_component, tas_models: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the signal generator with tactician component and TAS models.
        
        Args:
            tactician_component: Initialized Tactician component
            tas_models: Pre-trained TAS models for per-signal-type timing generation
            
        Returns:
            bool: True if initialization successful
        """
        try:
            self.tactician = tactician_component
            
            # Initialize TAS enhancer
            if self.enable_tas_enhancement:
                await self.tas_enhancer.initialize(tas_models)
            
            self.logger.info("✅ Tactician Signal Generator initialized with shared utilities")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Tactician Signal Generator: {e}")
            return False

    @handles_errors
    @traced(span_name="tactician_signal_generation")
    @log_execution_time()
    async def generate_timing_signal(
        self,
        symbol: str,
        analyst_signal: Dict[str, Any],
        market_data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]] = None,
        account_balance: float = 10000.0,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[TacticianSignal]:
        """
        Generate timing signal using Tactician component with shared utilities.
        
        Args:
            symbol: Trading symbol
            analyst_signal: Signal from Analyst component
            market_data: Market data DataFrame
            current_position: Current position information
            account_balance: Account balance for position sizing
            additional_context: Additional context for signal generation
            
        Returns:
            TacticianSignal or None if no signal generated
        """
        try:
            if not self.tactician:
                self.logger.error("❌ Tactician component not initialized")
                return None

            tprint_info(f"🔄 Generating tactician timing signal for {symbol}")
            
            # Perform timing analysis using Tactician
            timing_analysis = await self._perform_timing_analysis(
                symbol, analyst_signal, market_data, current_position, additional_context
            )
            
            if not timing_analysis:
                tprint_warning(f"⚠️ No timing analysis result for {symbol}")
                return None
            
            # Enhance with TAS if available
            enhanced_result = None
            if self.enable_tas_enhancement:
                # Add signal type to context for TAS
                tas_context = (additional_context or {}).copy()
                tas_context['signal_type'] = self._map_analyst_signal_to_type(analyst_signal)
                
                enhanced_result = await self.tas_enhancer.enhance_signal(
                    timing_analysis, market_data, None, tas_context
                )
            
            # Calculate position sizing
            position_sizing = await self._calculate_position_sizing(
                symbol, analyst_signal, timing_analysis, account_balance
            )
            
            # Generate timing signal with TAS enhancement
            signal = await self._generate_timing_signal_from_analysis(
                symbol, timing_analysis, position_sizing, current_position, enhanced_result
            )
            
            if signal:
                # Store signal in history
                self._store_signal(signal)
                self.signal_count += 1
                
                tprint_success(f"✅ Generated {signal.timing_signal.value} signal for {symbol} "
                             f"(confidence: {signal.confidence_score:.3f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Timing signal generation failed for {symbol}: {e}")
            return None

    async def _perform_timing_analysis(
        self,
        symbol: str,
        analyst_signal: Dict[str, Any],
        market_data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform timing analysis using Tactician component."""
        try:
            # Prepare timing context
            timing_context = {
                'symbol': symbol,
                'analyst_signal': analyst_signal,
                'market_data': market_data,
                'current_position': current_position,
                'additional_context': additional_context or {}
            }
            
            # Call Tactician's timing analysis method
            if hasattr(self.tactician, 'analyze_timing'):
                timing_result = await self.tactician.analyze_timing(timing_context)
            elif hasattr(self.tactician, 'run_timing_analysis'):
                timing_result = await self.tactician.run_timing_analysis(timing_context)
            else:
                # Use fallback analysis
                fallback_result = await self.fallback_analyzer.perform_fallback_analysis(
                    market_data, "tas", current_position, additional_context
                )
                timing_result = self._convert_fallback_to_timing_analysis(fallback_result, analyst_signal)
            
            return timing_result
            
        except Exception as e:
            self.logger.error(f"❌ Timing analysis failed: {e}")
            return None

    def _convert_fallback_to_timing_analysis(
        self, 
        fallback_result: FallbackAnalysisResult, 
        analyst_signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert fallback analysis result to timing analysis format."""
        return {
            'timing_signal': fallback_result.signal_direction,
            'confidence_score': fallback_result.confidence_score,
            'scenario_predictions': {
                'bullish_probability': 0.6 if analyst_signal.get('signal_type') == 'buy' else 0.4,
                'bearish_probability': 0.6 if analyst_signal.get('signal_type') == 'sell' else 0.4,
                'sideways_probability': 0.3
            },
            'risk_metrics': {
                'volatility': fallback_result.volatility_score,
                'momentum': 0.0,  # Default value
                'volume_trend': 1.0  # Default value
            },
            'timing_indicators': fallback_result.technical_indicators,
            'analysis_metadata': fallback_result.analysis_metadata
        }

    async def _calculate_position_sizing(
        self,
        symbol: str,
        analyst_signal: Dict[str, Any],
        timing_analysis: Dict[str, Any],
        account_balance: float
    ) -> PositionSizing:
        """Calculate position sizing based on confidence and risk parameters."""
        try:
            # Get confidence scores
            analyst_confidence = analyst_signal.get('confidence_score', 0.5)
            timing_confidence = timing_analysis.get('confidence_score', 0.5)
            
            # Combined confidence
            combined_confidence = (analyst_confidence + timing_confidence) / 2
            
            # Kelly criterion calculation for position sizing
            win_probability = combined_confidence
            avg_win = 0.005   # 0.5% average win
            avg_loss = 0.003  # 0.3% average loss
            
            kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, self.kelly_fraction))
            
            # Position size calculation
            risk_amount = account_balance * self.risk_per_trade
            confidence_multiplier = combined_confidence
            
            # Base position size
            base_size = risk_amount * confidence_multiplier
            
            # Apply Kelly fraction
            recommended_size = base_size * kelly_fraction
            
            # Maximum position size (10% of account)
            max_size = account_balance * 0.1
            
            # Leverage calculation
            leverage = min(combined_confidence * self.max_leverage, self.max_leverage)
            
            return PositionSizing(
                recommended_size=min(recommended_size, max_size),
                max_size=max_size,
                leverage=leverage,
                risk_per_trade=self.risk_per_trade,
                kelly_fraction=kelly_fraction,
                confidence_multiplier=confidence_multiplier
            )
            
        except Exception as e:
            self.logger.error(f"❌ Position sizing calculation failed: {e}")
            # Return conservative defaults
            return PositionSizing(
                recommended_size=account_balance * 0.01,
                max_size=account_balance * 0.05,
                leverage=1.0,
                risk_per_trade=self.risk_per_trade,
                kelly_fraction=0.1,
                confidence_multiplier=0.5
            )

    def _map_analyst_signal_to_type(self, analyst_signal: Dict[str, Any]) -> int:
        """Map analyst signal to signal type for TAS model selection."""
        signal_type = analyst_signal.get('signal_type', 'hold')
        if signal_type == 'buy':
            return 1  # Long signal
        elif signal_type == 'sell':
            return -1  # Short signal
        else:
            return 0  # Hold signal

    async def _generate_timing_signal_from_analysis(
        self,
        symbol: str,
        timing_analysis: Dict[str, Any],
        position_sizing: PositionSizing,
        current_position: Optional[Dict[str, Any]],
        enhanced_result: Optional[EnhancementResult] = None
    ) -> Optional[TacticianSignal]:
        """Generate timing signal from analysis result with TAS enhancement."""
        try:
            # Extract timing information
            timing_signal_str = timing_analysis.get('timing_signal', 'hold')
            confidence_score = timing_analysis.get('confidence_score', 0.0)
            
            # Use enhanced confidence if available
            if enhanced_result and enhanced_result.success:
                confidence_score = enhanced_result.confidence_metrics.final_confidence
                self.tas_enhanced_signals += 1
            
            # Check confidence threshold
            if confidence_score < self.confidence_threshold:
                return None
            
            # Map timing signal
            timing_signal = self._map_timing_signal(timing_signal_str, current_position)
            
            # Determine confidence level
            confidence = self._determine_confidence_level(confidence_score)
            
            # Create signal with TAS enhancement
            signal = TacticianSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                timing_signal=timing_signal,
                confidence=confidence,
                confidence_score=confidence_score,
                position_sizing=position_sizing,
                scenario_predictions=timing_analysis.get('scenario_predictions', {}),
                risk_metrics=timing_analysis.get('risk_metrics', {}),
                timing_indicators=timing_analysis.get('timing_indicators', {}),
                # TAS enhancement fields
                tas_prediction=enhanced_result.enhanced_signal.get('tas_prediction') if enhanced_result else None,
                tas_confidence=enhanced_result.confidence_metrics.enhanced_confidence if enhanced_result else 0.0,
                tas_architecture_type=enhanced_result.enhanced_signal.get('tas_architecture', {}).get('type') if enhanced_result else None,
                signal_type=enhanced_result.enhanced_signal.get('signal_type') if enhanced_result else None,
                metadata=timing_analysis.get('analysis_metadata', {})
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Timing signal generation from analysis failed: {e}")
            return None

    def _map_timing_signal(self, signal_str: str, current_position: Optional[Dict[str, Any]]) -> TimingSignal:
        """Map timing signal string to TimingSignal enum."""
        if current_position:
            # We have a position
            if signal_str == 'exit':
                position_type = current_position.get('type', 'long')
                return TimingSignal.EXIT_LONG if position_type == 'long' else TimingSignal.EXIT_SHORT
            else:
                return TimingSignal.HOLD
        else:
            # No position
            if signal_str == 'enter':
                # This would need to be determined by analyst signal direction
                # For now, default to long entry
                return TimingSignal.ENTER_LONG
            else:
                return TimingSignal.HOLD

    def _determine_confidence_level(self, confidence_score: float) -> TimingConfidence:
        """Determine confidence level based on score."""
        if confidence_score >= self.confidence_thresholds[TimingConfidence.VERY_HIGH]:
            return TimingConfidence.VERY_HIGH
        elif confidence_score >= self.confidence_thresholds[TimingConfidence.HIGH]:
            return TimingConfidence.HIGH
        elif confidence_score >= self.confidence_thresholds[TimingConfidence.MEDIUM]:
            return TimingConfidence.MEDIUM
        else:
            return TimingConfidence.LOW

    def _store_signal(self, signal: TacticianSignal):
        """Store signal in history."""
        self.signal_history.append(signal)
        
        # Maintain history size
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)

    def get_signal_history(self, n: int = 100) -> List[TacticianSignal]:
        """Get recent signal history."""
        return self.signal_history[-n:] if len(self.signal_history) >= n else self.signal_history.copy()

    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        if self.signal_count == 0:
            return {
                'total_signals': 0,
                'success_rate': 0.0,
                'signal_distribution': {},
                'avg_confidence': 0.0,
                'avg_position_size': 0.0
            }
        
        # Calculate signal distribution
        signal_distribution = {}
        for signal in self.signal_history:
            signal_type = signal.timing_signal.value
            signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1
        
        # Calculate averages
        avg_confidence = np.mean([s.confidence_score for s in self.signal_history])
        avg_position_size = np.mean([s.position_sizing.recommended_size for s in self.signal_history])
        
        return {
            'total_signals': self.signal_count,
            'success_rate': self.successful_signals / self.signal_count if self.signal_count > 0 else 0.0,
            'signal_distribution': signal_distribution,
            'avg_confidence': avg_confidence,
            'avg_position_size': avg_position_size,
            'recent_signals': len(self.signal_history),
            'tas_enhanced_signals': self.tas_enhanced_signals,
            'tas_enhancement_rate': self.tas_enhanced_signals / self.signal_count if self.signal_count > 0 else 0.0,
            'shared_utilities_stats': {
                'feature_engine': self.feature_engine.get_performance_metrics(),
                'confidence_calculator': self.confidence_calculator.get_performance_metrics(),
                'fallback_analyzer': self.fallback_analyzer.get_performance_metrics(),
                'tas_enhancer': self.tas_enhancer.get_enhancement_stats()
            }
        }

    def update_signal_performance(self, signal: TacticianSignal, was_successful: bool):
        """Update signal performance tracking."""
        if was_successful:
            self.successful_signals += 1
        else:
            self.failed_signals += 1

# Convenience functions
def create_tactician_signal_generator(config: Dict[str, Any]) -> TacticianSignalGenerator:
    """Create a configured tactician signal generator."""
    return TacticianSignalGenerator(config)

async def generate_tactician_signal(
    signal_generator: TacticianSignalGenerator,
    symbol: str,
    analyst_signal: Dict[str, Any],
    market_data: pd.DataFrame,
    tactician_component,
    current_position: Optional[Dict[str, Any]] = None,
    account_balance: float = 10000.0
) -> Optional[TacticianSignal]:
    """Generate tactician signal with convenience function."""
    if not signal_generator.tactician:
        await signal_generator.initialize(tactician_component)
    
    return await signal_generator.generate_timing_signal(
        symbol=symbol,
        analyst_signal=analyst_signal,
        market_data=market_data,
        current_position=current_position,
        account_balance=account_balance
    )