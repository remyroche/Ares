"""
Analyst Signal Generation

This module integrates the Analyst component to generate trading signals
based on market analysis, feature engineering, and ML predictions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

# Import NAS components for enhanced signal generation
from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import (
    EnhancedPerfectNASRegimeDetector, EnhancedPerfectNASResult
)
from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType
)

logger = system_logger.getChild('AnalystSignals')

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class AnalystSignal:
    """Analyst-generated trading signal with NAS enhancement."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence_score: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    market_health_score: float = 0.0
    volatility_score: float = 0.0
    liquidation_risk_score: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)
    # NAS enhancement fields
    nas_prediction: Optional[Dict[str, Any]] = None
    nas_confidence: float = 0.0
    nas_architecture_type: Optional[str] = None
    regime_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AnalystSignalGenerator:
    """
    Analyst Signal Generator that integrates with the Analyst component
    and NAS for enhanced trading signal generation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyst signal generator with NAS enhancement.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('AnalystSignalGenerator')

        # Analyst component (will be injected)
        self.analyst = None

        # NAS engine for enhanced signal generation
        self.nas_engine = None
        self.nas_models = {}  # Per-regime NAS models
        self.nas_architectures = {}  # Per-regime NAS architectures

        # Signal generation parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.nas_confidence_threshold = config.get('nas_confidence_threshold', 0.7)
        self.signal_strength_thresholds = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 0.65,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 0.9
        }

        # NAS configuration
        self.enable_nas_enhancement = config.get('enable_nas_enhancement', True)
        self.nas_timeframe = config.get('nas_timeframe', '5m')
        self.regime_timeframe = config.get('regime_timeframe', '15m')

        # Signal history (using deque for efficient O(1) operations)
        self.max_history = config.get('max_history', 1000)
        self.signal_history: deque = deque(maxlen=self.max_history)

        # Performance tracking
        self.signal_count = 0
        self.successful_signals = 0
        self.failed_signals = 0
        self.nas_enhanced_signals = 0

    async def initialize(self, analyst_component, nas_models: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the signal generator with analyst component and NAS models.

        Args:
            analyst_component: Initialized Analyst component
            nas_models: Pre-trained NAS models for per-regime signal generation

        Returns:
            bool: True if initialization successful
        """
        try:
            self.analyst = analyst_component

            # Initialize NAS engine if enhancement is enabled
            if self.enable_nas_enhancement:
                await self._initialize_nas_engine(nas_models)

            self.logger.info("✅ Analyst Signal Generator initialized with NAS enhancement")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Analyst Signal Generator: {e}")
            return False

    async def _initialize_nas_engine(self, nas_models: Optional[Dict[str, Any]] = None):
        """Initialize NAS engine for enhanced signal generation."""
        try:
            # Create NAS configuration
            nas_config = PerfectNASConfig(
                primary_architecture=NeuralArchitectureType.HYBRID,
                n_regimes=8,
                primary_timeframe=self.nas_timeframe,
                enable_neural_odes=True,
                enable_vision_transformers=True,
                enable_state_space_models=True,
                enable_micro_regime_detection=True,
                population_size=30,
                generations=50
            )

            # Initialize NAS engine
            self.nas_engine = EnhancedPerfectNASRegimeDetector(nas_config)

            # Load pre-trained NAS models if provided
            if nas_models:
                self.nas_models = nas_models
                self.logger.info(f"✅ Loaded {len(nas_models)} NAS models for signal generation")
            else:
                self.logger.warning("⚠️ No NAS models provided, using fallback analysis")

            self.logger.info("✅ NAS engine initialized for signal generation")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize NAS engine: {e}")
            self.enable_nas_enhancement = False

    @handles_errors
    @traced(span_name="analyst_signal_generation")
    @log_execution_time()
    async def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        regime_data: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[AnalystSignal]:
        """
        Generate trading signal using Analyst component.

        Args:
            symbol: Trading symbol
            market_data: Market data DataFrame
            regime_data: Current regime information
            additional_context: Additional context for signal generation

        Returns:
            AnalystSignal or None if no signal generated
        """
        try:
            if not self.analyst:
                self.logger.error("❌ Analyst component not initialized")
                return None

            tprint_info(f"🔄 Generating analyst signal for {symbol}")

            # Perform market analysis using Analyst
            analysis_result = await self._perform_market_analysis(
                symbol, market_data, regime_data, additional_context
            )

            if not analysis_result:
                tprint_warning(f"⚠️ No analysis result for {symbol}")
                return None

            # Enhance with NAS prediction if available
            nas_prediction = None
            if self.enable_nas_enhancement and self.nas_engine:
                nas_prediction = await self._generate_nas_prediction(
                    symbol, market_data, regime_data
                )

            # Generate signal based on analysis and NAS prediction
            signal = await self._generate_signal_from_analysis(
                symbol, analysis_result, market_data, nas_prediction
            )

            if signal:
                # Store signal in history
                self._store_signal(signal)
                self.signal_count += 1

                tprint_success(f"✅ Generated {signal.signal_type.value} signal for {symbol} "
                             f"(confidence: {signal.confidence_score:.3f})")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Signal generation failed for {symbol}: {e}")
            return None

    async def _perform_market_analysis(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        regime_data: Optional[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform market analysis using Analyst component."""
        try:
            # Prepare analysis context
            analysis_context = {
                'symbol': symbol,
                'market_data': market_data,
                'regime_data': regime_data,
                'additional_context': additional_context or {}
            }

            # Call Analyst's analyze method
            if hasattr(self.analyst, 'analyze'):
                analysis_result = await self.analyst.analyze(analysis_context)
            elif hasattr(self.analyst, 'run_analysis'):
                analysis_result = await self.analyst.run_analysis(analysis_context)
            else:
                # Fallback to basic analysis
                analysis_result = await self._fallback_analysis(analysis_context)

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Market analysis failed: {e}")
            return None

    async def _fallback_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when Analyst methods are not available."""
        try:
            market_data = context['market_data']

            # Basic technical analysis
            if len(market_data) < 20:
                return None

            # Calculate basic indicators
            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]

            # Simple signal based on price momentum
            recent_returns = returns[-5:].mean()
            volatility = np.std(returns[-20:])

            # Generate basic analysis result
            analysis_result = {
                'signal_direction': 'buy' if recent_returns > 0.001 else 'sell' if recent_returns < -0.001 else 'hold',
                'confidence_score': min(abs(recent_returns) / volatility, 1.0) if volatility > 0 else 0.5,
                'market_health_score': 0.7,  # Default value
                'volatility_score': volatility,
                'liquidation_risk_score': 0.3,  # Default value
                'feature_importance': {},
                'ml_predictions': {},
                'analysis_metadata': {
                    'method': 'fallback',
                    'timestamp': datetime.now().isoformat()
                }
            }

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Fallback analysis failed: {e}")
            return None

    async def _generate_nas_prediction(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        regime_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate NAS prediction for enhanced signal generation."""
        try:
            if not self.nas_engine:
                return None

            # Ensure we have enough data to build a sliding window for regime detection
            if market_data is None or len(market_data) < 10:
                self.logger.debug("⚠️ Insufficient market data for NAS prediction")
                return None

            window_size = min(len(market_data), 240)
            sliding_window = market_data.tail(window_size).copy()

            # Focus on numeric columns (prefer OHLCV if available)
            ohlcv_columns = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in sliding_window.columns]
            if ohlcv_columns:
                nas_input = sliding_window[ohlcv_columns]
            else:
                numeric_cols = sliding_window.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    self.logger.debug("⚠️ No numeric columns available for NAS prediction")
                    return None
                nas_input = sliding_window[numeric_cols]

            nas_input = nas_input.replace([np.inf, -np.inf], np.nan).dropna()
            if len(nas_input) < 8:
                self.logger.debug("⚠️ Sliding window after cleaning is too small for NAS prediction")
                return None

            # Get current regime for NAS model selection
            regime_id = regime_data.get('regime_id', 0) if regime_data else 0
            nas_model = self.nas_models.get(regime_id)

            # Generate NAS prediction for trading signals using recent market window
            nas_result = self.nas_engine.detect_regimes(
                nas_input,
                optimize_architecture=False,  # Use pre-trained model
                enable_meta_learning=False
            )

            if not nas_result.success or nas_result.regime_predictions.size == 0:
                return None

            predicted_regime = int(nas_result.regime_predictions[-1])

            last_probabilities = None
            if isinstance(nas_result.regime_probabilities, np.ndarray) and nas_result.regime_probabilities.size > 0:
                if nas_result.regime_probabilities.ndim == 1:
                    last_probabilities = nas_result.regime_probabilities
                else:
                    last_probabilities = nas_result.regime_probabilities[-1]

            nas_confidence = 0.0
            if last_probabilities is not None and len(last_probabilities) > 0:
                # Validate predicted_regime is within valid bounds
                if isinstance(predicted_regime, (int, np.integer)):
                    if 0 <= predicted_regime < len(last_probabilities):
                        nas_confidence = float(last_probabilities[predicted_regime])
                    else:
                        # Index out of bounds - use max probability as fallback
                        self.logger.warning(
                            f"⚠️ Predicted regime {predicted_regime} out of bounds [0, {len(last_probabilities)}), "
                            f"using max probability as fallback"
                        )
                        nas_confidence = float(np.max(last_probabilities))
                else:
                    # Invalid predicted_regime type - use max probability
                    self.logger.warning(
                        f"⚠️ Invalid predicted_regime type: {type(predicted_regime)}, using max probability"
                    )
                    nas_confidence = float(np.max(last_probabilities))
            else:
                self.logger.debug("⚠️ No valid regime probabilities available for NAS confidence")

            close_direction = 'hold'
            if 'close' in nas_input.columns and len(nas_input['close']) >= 2:
                price_change = nas_input['close'].iloc[-1] - nas_input['close'].iloc[0]
                if price_change > 0:
                    close_direction = 'buy'
                elif price_change < 0:
                    close_direction = 'sell'

            architecture_metadata: Dict[str, Any] = {}
            if nas_model:
                if isinstance(nas_model, dict):
                    architecture = nas_model.get('architecture')
                    if isinstance(architecture, dict):
                        architecture_metadata.update(architecture)
                        if 'type' not in architecture_metadata and 'name' in architecture_metadata:
                            architecture_metadata['type'] = architecture_metadata.get('name')
                    elif architecture is not None:
                        architecture_metadata['type'] = getattr(architecture, 'type', None) or str(architecture)
                    architecture_metadata.setdefault('model_type', nas_model.get('model_type'))
                    architecture_metadata.setdefault('trained', nas_model.get('trained'))
                    if nas_model.get('performance_score') is not None:
                        architecture_metadata.setdefault('performance_score', nas_model.get('performance_score'))
                else:
                    architecture_metadata['type'] = getattr(nas_model, 'architecture_type', None) or str(type(nas_model))

            timestamp = None
            if hasattr(market_data, 'index') and len(market_data.index) > 0:
                last_index = market_data.index[-1]
                if isinstance(last_index, (datetime, np.datetime64, pd.Timestamp)):
                    timestamp = pd.Timestamp(last_index).isoformat()

            nas_prediction_payload = {
                'predicted_regime': predicted_regime,
                'regime_probabilities': last_probabilities.tolist() if last_probabilities is not None else [],
                'direction': close_direction,
                'window_size': len(nas_input),
                'timestamp': timestamp,
                'current_regime_id': regime_id
            }

            return {
                'nas_prediction': nas_prediction_payload,
                'nas_confidence': nas_confidence,
                'nas_architecture': architecture_metadata,
                'regime_id': predicted_regime,
                'nas_contribution': 'trading_signals'
            }

        except Exception as e:
            self.logger.error(f"❌ NAS prediction failed for {symbol}: {e}")
            return None

    def _prepare_nas_features(self, market_data: pd.DataFrame, regime_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """Prepare features for NAS prediction."""
        try:
            # Extract basic features from market data
            features = []

            # Price features
            if len(market_data) >= 20:
                close_prices = market_data['close'].values
                returns = np.diff(close_prices) / close_prices[:-1]

                # Recent returns
                features.extend([
                    returns[-1],  # Latest return
                    returns[-5:].mean(),  # 5-period average return
                    returns[-10:].mean(),  # 10-period average return
                    returns[-20:].mean(),  # 20-period average return
                ])

                # Volatility features
                features.extend([
                    np.std(returns[-5:]),  # 5-period volatility
                    np.std(returns[-10:]),  # 10-period volatility
                    np.std(returns[-20:]),  # 20-period volatility
                ])

                # Price momentum
                features.extend([
                    (close_prices[-1] - close_prices[-5]) / close_prices[-5],  # 5-period momentum
                    (close_prices[-1] - close_prices[-10]) / close_prices[-10],  # 10-period momentum
                    (close_prices[-1] - close_prices[-20]) / close_prices[-20],  # 20-period momentum
                ])
            else:
                # Fallback features
                features = [0.0] * 10

            # Add regime information if available
            if regime_data:
                features.append(regime_data.get('regime_id', 0))
                features.append(regime_data.get('regime_stability', 0.5))
            else:
                features.extend([0, 0.5])

            return np.array(features)

        except Exception as e:
            self.logger.error(f"❌ Feature preparation failed: {e}")
            return np.zeros(12)  # Fallback features

    async def _generate_signal_from_analysis(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        market_data: pd.DataFrame,
        nas_prediction: Optional[Dict[str, Any]] = None
    ) -> Optional[AnalystSignal]:
        """Generate signal from analysis result with NAS enhancement."""
        try:
            # Extract signal information
            signal_direction = analysis_result.get('signal_direction', 'hold')
            confidence_score = analysis_result.get('confidence_score', 0.0)

            # Enhance with NAS prediction if available
            if nas_prediction:
                nas_confidence = nas_prediction.get('nas_confidence', 0.0)
                nas_prediction_value = nas_prediction.get('nas_prediction', {})

                # Combine confidence scores (weighted average: 60% analysis, 40% NAS)
                combined_confidence = (confidence_score * 0.6) + (nas_confidence * 0.4)

                # Use NAS prediction to enhance signal direction if confidence is high
                if nas_confidence >= self.nas_confidence_threshold:
                    nas_direction = nas_prediction_value.get('direction', signal_direction)
                    if nas_direction != signal_direction:
                        # NAS overrides if it's more confident
                        signal_direction = nas_direction
                        self.nas_enhanced_signals += 1
                    # Use combined confidence when NAS is confident
                    confidence_score = combined_confidence
                else:
                    # NAS confidence is low, but still use combined for slight enhancement
                    confidence_score = combined_confidence

            # Check confidence threshold
            if confidence_score < self.confidence_threshold:
                return None

            # Determine signal type
            signal_type = self._map_signal_direction(signal_direction)

            # Determine signal strength
            signal_strength = self._determine_signal_strength(confidence_score)

            # Calculate price targets
            current_price = market_data['close'].iloc[-1]
            price_target, stop_loss = self._calculate_price_targets(
                signal_type, current_price, analysis_result
            )

            # Create signal with NAS enhancement
            signal = AnalystSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                price_target=price_target,
                stop_loss=stop_loss,
                market_health_score=analysis_result.get('market_health_score', 0.0),
                volatility_score=analysis_result.get('volatility_score', 0.0),
                liquidation_risk_score=analysis_result.get('liquidation_risk_score', 0.0),
                feature_importance=analysis_result.get('feature_importance', {}),
                ml_predictions=analysis_result.get('ml_predictions', {}),
                # NAS enhancement fields
                nas_prediction=nas_prediction,
                nas_confidence=nas_prediction.get('nas_confidence', 0.0) if nas_prediction else 0.0,
                nas_architecture_type=nas_prediction.get('nas_architecture', {}).get('type') if nas_prediction else None,
                regime_id=nas_prediction.get('regime_id') if nas_prediction else None,
                metadata=analysis_result.get('analysis_metadata', {})
            )

            return signal

        except Exception as e:
            self.logger.error(f"❌ Signal generation from analysis failed: {e}")
            return None

    def _map_signal_direction(self, direction: str) -> SignalType:
        """Map analysis direction to signal type."""
        direction_map = {
            'buy': SignalType.BUY,
            'sell': SignalType.SELL,
            'hold': SignalType.HOLD,
            'close': SignalType.CLOSE
        }
        return direction_map.get(direction.lower(), SignalType.HOLD)

    def _determine_signal_strength(self, confidence_score: float) -> SignalStrength:
        """Determine signal strength based on confidence score."""
        if confidence_score >= self.signal_strength_thresholds[SignalStrength.VERY_STRONG]:
            return SignalStrength.VERY_STRONG
        elif confidence_score >= self.signal_strength_thresholds[SignalStrength.STRONG]:
            return SignalStrength.STRONG
        elif confidence_score >= self.signal_strength_thresholds[SignalStrength.MODERATE]:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _calculate_price_targets(
        self,
        signal_type: SignalType,
        current_price: float,
        analysis_result: Dict[str, Any]
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate price targets and stop loss."""
        try:
            volatility = analysis_result.get('volatility_score', 0.02)

            if signal_type == SignalType.BUY:
                # For buy signals, target 2x volatility, stop loss at 1x volatility
                price_target = current_price * (1 + 2 * volatility)
                stop_loss = current_price * (1 - volatility)
            elif signal_type == SignalType.SELL:
                # For sell signals, target 2x volatility, stop loss at 1x volatility
                price_target = current_price * (1 - 2 * volatility)
                stop_loss = current_price * (1 + volatility)
            else:
                # For hold/close signals, no price targets
                price_target = None
                stop_loss = None

            return price_target, stop_loss

        except Exception as e:
            self.logger.error(f"❌ Price target calculation failed: {e}")
            return None, None

    def _store_signal(self, signal: AnalystSignal):
        """Store signal in history (deque automatically handles maxlen)."""
        self.signal_history.append(signal)

    def get_signal_history(self, n: int = 100) -> List[AnalystSignal]:
        """Get recent signal history."""
        # Convert deque to list for return
        signal_list = list(self.signal_history)
        return signal_list[-n:] if len(signal_list) >= n else signal_list

    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        if self.signal_count == 0:
            return {
                'total_signals': 0,
                'success_rate': 0.0,
                'signal_distribution': {},
                'avg_confidence': 0.0
            }

        # Calculate signal distribution
        signal_distribution = {}
        for signal in self.signal_history:
            signal_type = signal.signal_type.value
            signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1

        # Calculate average confidence
        avg_confidence = np.mean([s.confidence_score for s in self.signal_history])

        return {
            'total_signals': self.signal_count,
            'success_rate': self.successful_signals / self.signal_count if self.signal_count > 0 else 0.0,
            'signal_distribution': signal_distribution,
            'avg_confidence': avg_confidence,
            'recent_signals': len(self.signal_history)
        }

    def update_signal_performance(self, signal: AnalystSignal, was_successful: bool):
        """Update signal performance tracking."""
        if was_successful:
            self.successful_signals += 1
        else:
            self.failed_signals += 1

# Convenience functions

def create_analyst_signal_generator(config: Dict[str, Any]) -> AnalystSignalGenerator:
    """Create a configured analyst signal generator."""
    return AnalystSignalGenerator(config)

async def generate_analyst_signal(
    signal_generator: AnalystSignalGenerator,
    symbol: str,
    market_data: pd.DataFrame,
    analyst_component,
    regime_data: Optional[Dict[str, Any]] = None
) -> Optional[AnalystSignal]:
    """Generate analyst signal with convenience function."""
    if not signal_generator.analyst:
        await signal_generator.initialize(analyst_component)

    return await signal_generator.generate_signal(
        symbol=symbol,
        market_data=market_data,
        regime_data=regime_data
    )
