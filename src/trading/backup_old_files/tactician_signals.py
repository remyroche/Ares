"""
Tactician Signal Generation

This module integrates the Tactician component to generate timing signals
for trade execution based on position sizing, scenario predictions, and risk management.
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

# Import TAS components for enhanced signal generation
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_regime_detector import (
    EnhancedTASRegimeDetector, EnhancedTASResult
)
from src.training.steps.market_analysis.tas_regime.core.tas_config import TASConfig

logger = system_logger.getChild('TacticianSignals')

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

class TacticianSignalGenerator:
    """
    Tactician Signal Generator that integrates with the Tactician component
    and TAS for enhanced timing signal generation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tactician signal generator with TAS enhancement.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('TacticianSignalGenerator')
        
        # Tactician component (will be injected)
        self.tactician = None
        
        # TAS engine for enhanced signal generation
        self.tas_engine = None
        self.tas_models = {}  # Per-signal-type TAS models
        self.tas_architectures = {}  # Per-signal-type TAS architectures
        
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
            
            # Initialize TAS engine if enhancement is enabled
            if self.enable_tas_enhancement:
                await self._initialize_tas_engine(tas_models)
            
            self.logger.info("✅ Tactician Signal Generator initialized with TAS enhancement")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Tactician Signal Generator: {e}")
            return False
    
    async def _initialize_tas_engine(self, tas_models: Optional[Dict[str, Any]] = None):
        """Initialize TAS engine for enhanced timing signal generation."""
        try:
            # Create TAS configuration
            tas_config = TASConfig(
                n_regimes=8,
                primary_timeframe=self.tas_timeframe,
                enable_tree_ensemble=True,
                enable_boosted_trees=True,
                enable_random_forest=True,
                population_size=30,
                generations=50
            )
            
            # Initialize TAS engine
            self.tas_engine = EnhancedTASRegimeDetector(tas_config)
            
            # Load pre-trained TAS models if provided
            if tas_models:
                self.tas_models = tas_models
                self.logger.info(f"✅ Loaded {len(tas_models)} TAS models for timing signal generation")
            else:
                self.logger.warning("⚠️ No TAS models provided, using fallback analysis")
            
            self.logger.info("✅ TAS engine initialized for timing signal generation")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TAS engine: {e}")
            self.enable_tas_enhancement = False

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
        Generate timing signal using Tactician component.
        
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
            
            # Enhance with TAS prediction if available
            tas_prediction = None
            if self.enable_tas_enhancement and self.tas_engine:
                tas_prediction = await self._generate_tas_prediction(
                    symbol, analyst_signal, market_data, current_position
                )
            
            # Calculate position sizing
            position_sizing = await self._calculate_position_sizing(
                symbol, analyst_signal, timing_analysis, account_balance
            )
            
            # Generate timing signal with TAS enhancement
            signal = await self._generate_timing_signal_from_analysis(
                symbol, timing_analysis, position_sizing, current_position, tas_prediction
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
                # Fallback to basic timing analysis
                timing_result = await self._fallback_timing_analysis(timing_context)
            
            return timing_result
            
        except Exception as e:
            self.logger.error(f"❌ Timing analysis failed: {e}")
            return None

    async def _fallback_timing_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback timing analysis when Tactician methods are not available."""
        try:
            market_data = context['market_data']
            analyst_signal = context['analyst_signal']
            current_position = context.get('current_position')
            
            if len(market_data) < 20:
                return None
            
            # Basic timing indicators
            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Calculate timing metrics
            recent_volatility = np.std(returns[-10:])
            price_momentum = returns[-5:].mean()
            volume_trend = market_data['volume'].iloc[-5:].mean() / market_data['volume'].iloc[-20:-5].mean()
            
            # Determine timing signal based on analyst signal and market conditions
            analyst_direction = analyst_signal.get('signal_type', 'hold')
            confidence_score = analyst_signal.get('confidence_score', 0.5)
            
            # Timing logic
            if current_position:
                # We have a position, consider exit timing
                if confidence_score < 0.4 or recent_volatility > 0.05:
                    timing_signal = 'exit'
                    timing_confidence = min(confidence_score + 0.2, 1.0)
                else:
                    timing_signal = 'hold'
                    timing_confidence = confidence_score
            else:
                # No position, consider entry timing
                if analyst_direction in ['buy', 'sell'] and confidence_score > 0.6:
                    timing_signal = 'enter'
                    timing_confidence = confidence_score
                else:
                    timing_signal = 'hold'
                    timing_confidence = 0.5
            
            # Generate timing analysis result
            timing_result = {
                'timing_signal': timing_signal,
                'confidence_score': timing_confidence,
                'scenario_predictions': {
                    'bullish_probability': 0.6 if analyst_direction == 'buy' else 0.4,
                    'bearish_probability': 0.6 if analyst_direction == 'sell' else 0.4,
                    'sideways_probability': 0.3
                },
                'risk_metrics': {
                    'volatility': recent_volatility,
                    'momentum': price_momentum,
                    'volume_trend': volume_trend
                },
                'timing_indicators': {
                    'rsi': 50.0,  # Default value
                    'macd': 0.0,  # Default value
                    'bollinger_position': 0.5  # Default value
                },
                'analysis_metadata': {
                    'method': 'fallback',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            return timing_result
            
        except Exception as e:
            self.logger.error(f"❌ Fallback timing analysis failed: {e}")
            return None

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
            # Note: These values represent actual trading outcomes, not model training targets
            win_probability = combined_confidence
            avg_win = 0.005   # 0.5% average win (realistic trading outcome expectation)
            avg_loss = 0.003  # 0.3% average loss (realistic risk management)
            
            kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, self.kelly_fraction))  # Cap at configured fraction
            
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
                recommended_size=account_balance * 0.01,  # 1% of account
                max_size=account_balance * 0.05,  # 5% max
                leverage=1.0,
                risk_per_trade=self.risk_per_trade,
                kelly_fraction=0.1,
                confidence_multiplier=0.5
            )

    async def _generate_tas_prediction(
        self,
        symbol: str,
        analyst_signal: Dict[str, Any],
        market_data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate TAS prediction for enhanced timing signal generation."""
        try:
            if not self.tas_engine:
                return None
            
            # Determine signal type from analyst signal
            signal_type = self._map_analyst_signal_to_type(analyst_signal)
            
            # Use TAS model for this signal type if available
            if signal_type in self.tas_models:
                tas_model = self.tas_models[signal_type]
                
                # Prepare features for TAS prediction
                features = self._prepare_tas_features(market_data, analyst_signal, current_position)
                
                # Generate TAS prediction for timing signals
                tas_result = self.tas_engine.search(
                    train_data=(features.reshape(1, -1), np.array([0])),  # Dummy training data
                    validation_data=(features.reshape(1, -1), np.array([0])),  # Dummy validation data
                    regime_data={'analyst_signals': [signal_type]}
                )
                
                if tas_result.best_score > 0:
                    return {
                        'tas_prediction': tas_result.best_prediction,
                        'tas_confidence': tas_result.best_score,
                        'tas_architecture': tas_result.best_architecture,
                        'signal_type': signal_type,
                        'tas_contribution': 'timing_signals'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ TAS prediction failed for {symbol}: {e}")
            return None
    
    def _map_analyst_signal_to_type(self, analyst_signal: Dict[str, Any]) -> int:
        """Map analyst signal to signal type for TAS model selection."""
        signal_type = analyst_signal.get('signal_type', 'hold')
        if signal_type == 'buy':
            return 1  # Long signal
        elif signal_type == 'sell':
            return -1  # Short signal
        else:
            return 0  # Hold signal
    
    def _prepare_tas_features(self, market_data: pd.DataFrame, analyst_signal: Dict[str, Any], current_position: Optional[Dict[str, Any]]) -> np.ndarray:
        """Prepare features for TAS prediction."""
        try:
            features = []
            
            # Market data features
            if len(market_data) >= 20:
                close_prices = market_data['close'].values
                returns = np.diff(close_prices) / close_prices[:-1]
                volumes = market_data['volume'].values
                
                # Price features
                features.extend([
                    returns[-1],  # Latest return
                    returns[-5:].mean(),  # 5-period average return
                    returns[-10:].mean(),  # 10-period average return
                    np.std(returns[-10:]),  # 10-period volatility
                ])
                
                # Volume features
                features.extend([
                    volumes[-1] / volumes[-5:].mean(),  # Volume ratio
                    volumes[-5:].mean() / volumes[-20:].mean(),  # Volume trend
                ])
                
                # Price momentum
                features.extend([
                    (close_prices[-1] - close_prices[-5]) / close_prices[-5],  # 5-period momentum
                    (close_prices[-1] - close_prices[-10]) / close_prices[-10],  # 10-period momentum
                ])
            else:
                # Fallback features
                features = [0.0] * 8
            
            # Analyst signal features
            features.extend([
                analyst_signal.get('confidence_score', 0.0),
                1.0 if analyst_signal.get('signal_type') == 'buy' else -1.0 if analyst_signal.get('signal_type') == 'sell' else 0.0
            ])
            
            # Position features
            if current_position:
                features.extend([
                    1.0,  # Has position
                    current_position.get('quantity', 0.0),
                    current_position.get('unrealized_pnl', 0.0)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"❌ TAS feature preparation failed: {e}")
            return np.zeros(13)  # Fallback features
    
    async def _generate_timing_signal_from_analysis(
        self,
        symbol: str,
        timing_analysis: Dict[str, Any],
        position_sizing: PositionSizing,
        current_position: Optional[Dict[str, Any]],
        tas_prediction: Optional[Dict[str, Any]] = None
    ) -> Optional[TacticianSignal]:
        """Generate timing signal from analysis result with TAS enhancement."""
        try:
            # Extract timing information
            timing_signal_str = timing_analysis.get('timing_signal', 'hold')
            confidence_score = timing_analysis.get('confidence_score', 0.0)
            
            # Enhance with TAS prediction if available
            if tas_prediction:
                tas_confidence = tas_prediction.get('tas_confidence', 0.0)
                tas_prediction_value = tas_prediction.get('tas_prediction', {})
                
                # Combine confidence scores (weighted average: 60% analysis, 40% TAS)
                combined_confidence = (confidence_score * 0.6) + (tas_confidence * 0.4)
                
                # Use TAS prediction to enhance timing if confidence is high
                if tas_confidence >= self.tas_confidence_threshold:
                    tas_timing = tas_prediction_value.get('timing', timing_signal_str)
                    if tas_timing != timing_signal_str:
                        # TAS overrides if it's more confident
                        timing_signal_str = tas_timing
                        confidence_score = combined_confidence
                        self.tas_enhanced_signals += 1
                
                confidence_score = combined_confidence
            
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
                tas_prediction=tas_prediction,
                tas_confidence=tas_prediction.get('tas_confidence', 0.0) if tas_prediction else 0.0,
                tas_architecture_type=tas_prediction.get('tas_architecture', {}).get('type') if tas_prediction else None,
                signal_type=tas_prediction.get('signal_type') if tas_prediction else None,
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
            'recent_signals': len(self.signal_history)
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