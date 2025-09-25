"""
Analyst Signal Generation (Refactored)

This module integrates the Analyst component to generate trading signals
using shared utilities for feature engineering, confidence calculation,
and fallback analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
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

# Import NAS components for enhanced signal generation
from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import (
    EnhancedPerfectNASRegimeDetector, EnhancedPerfectNASResult
)
from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType
)

logger = system_logger.getChild('AnalystSignalsRefactored')

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

class NASSignalEnhancer(BaseSignalEnhancer):
    """
    NAS-specific signal enhancer that extends the base signal enhancer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize NAS signal enhancer."""
        super().__init__("nas", config)
        self.nas_engine = None
        self.nas_models = {}
        self.nas_architectures = {}
    
    async def _load_enhancement_models(self, models: Optional[Dict[str, Any]] = None) -> bool:
        """Load NAS enhancement models."""
        try:
            if models:
                self.nas_models = models
                self.logger.info(f"✅ Loaded {len(models)} NAS models")
                return True
            
            # Initialize NAS engine
            nas_config = PerfectNASConfig(
                primary_architecture=NeuralArchitectureType.HYBRID,
                n_regimes=8,
                primary_timeframe=self.config.get('nas_timeframe', '5m'),
                enable_neural_odes=True,
                enable_vision_transformers=True,
                enable_state_space_models=True,
                enable_micro_regime_detection=True,
                population_size=30,
                generations=50
            )
            
            self.nas_engine = EnhancedPerfectNASRegimeDetector(nas_config)
            self.logger.info("✅ NAS engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load NAS models: {e}")
            return False
    
    async def _generate_enhancement_prediction(
        self,
        features: FeatureSet,
        market_data: pd.DataFrame,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate NAS enhancement prediction."""
        try:
            if not self.nas_engine:
                return None
            
            # Prepare features for NAS prediction
            feature_vector = self.feature_engine.get_feature_vector(features)
            
            # Get regime information
            regime_id = additional_context.get('regime_id', 0) if additional_context else 0
            
            # Use NAS model for this regime if available
            if regime_id in self.nas_models:
                nas_model = self.nas_models[regime_id]
                
                # Generate NAS prediction
                nas_result = self.nas_engine.detect_regimes(
                    feature_vector.reshape(1, -1),
                    optimize_architecture=False,
                    enable_meta_learning=False
                )
                
                if nas_result.success:
                    return {
                        'nas_prediction': nas_result.best_prediction,
                        'confidence': nas_result.best_score,
                        'architecture': nas_result.best_architecture,
                        'regime_id': regime_id,
                        'contribution': 'trading_signals'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ NAS prediction failed: {e}")
            return None

class AnalystSignalGenerator:
    """
    Refactored Analyst Signal Generator using shared utilities.
    
    Integrates with the Analyst component and NAS for enhanced trading signal generation,
    using shared feature engineering, confidence calculation, and fallback analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyst signal generator with shared utilities.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('AnalystSignalGenerator')
        
        # Analyst component (will be injected)
        self.analyst = None
        
        # Initialize shared utilities
        self.feature_engine = UnifiedFeatureEngine(config.get('feature_config', {}))
        self.confidence_calculator = UnifiedConfidenceCalculator(config.get('confidence_config', {}))
        self.fallback_analyzer = UnifiedFallbackAnalyzer(config.get('fallback_config', {}))
        
        # NAS signal enhancer
        self.nas_enhancer = NASSignalEnhancer(config.get('nas_config', {}))
        
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
        
        # Signal history
        self.signal_history: List[AnalystSignal] = []
        self.max_history = config.get('max_history', 1000)
        
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
            
            # Initialize NAS enhancer
            if self.enable_nas_enhancement:
                await self.nas_enhancer.initialize(nas_models)
            
            self.logger.info("✅ Analyst Signal Generator initialized with shared utilities")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Analyst Signal Generator: {e}")
            return False

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
        Generate trading signal using Analyst component with shared utilities.
        
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
            
            # Enhance with NAS if available
            enhanced_result = None
            if self.enable_nas_enhancement:
                enhanced_result = await self.nas_enhancer.enhance_signal(
                    analysis_result, market_data, regime_data, additional_context
                )
            
            # Generate signal from analysis and enhancement
            signal = await self._generate_signal_from_analysis(
                symbol, analysis_result, market_data, enhanced_result
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
                # Use fallback analysis
                fallback_result = await self.fallback_analyzer.perform_fallback_analysis(
                    market_data, "nas", None, additional_context
                )
                analysis_result = self._convert_fallback_to_analysis(fallback_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Market analysis failed: {e}")
            return None

    def _convert_fallback_to_analysis(self, fallback_result: FallbackAnalysisResult) -> Dict[str, Any]:
        """Convert fallback analysis result to analysis format."""
        return {
            'signal_direction': fallback_result.signal_direction,
            'confidence_score': fallback_result.confidence_score,
            'market_health_score': fallback_result.market_health_score,
            'volatility_score': fallback_result.volatility_score,
            'liquidation_risk_score': fallback_result.liquidation_risk_score,
            'feature_importance': {},
            'ml_predictions': {},
            'analysis_metadata': fallback_result.analysis_metadata
        }

    async def _generate_signal_from_analysis(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        market_data: pd.DataFrame,
        enhanced_result: Optional[EnhancementResult] = None
    ) -> Optional[AnalystSignal]:
        """Generate signal from analysis result with NAS enhancement."""
        try:
            # Extract signal information
            signal_direction = analysis_result.get('signal_direction', 'hold')
            confidence_score = analysis_result.get('confidence_score', 0.0)
            
            # Use enhanced confidence if available
            if enhanced_result and enhanced_result.success:
                confidence_score = enhanced_result.confidence_metrics.final_confidence
                self.nas_enhanced_signals += 1
            
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
                nas_prediction=enhanced_result.enhanced_signal.get('nas_prediction') if enhanced_result else None,
                nas_confidence=enhanced_result.confidence_metrics.enhanced_confidence if enhanced_result else 0.0,
                nas_architecture_type=enhanced_result.enhanced_signal.get('nas_architecture', {}).get('type') if enhanced_result else None,
                regime_id=enhanced_result.enhanced_signal.get('regime_id') if enhanced_result else None,
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
        """Store signal in history."""
        self.signal_history.append(signal)
        
        # Maintain history size
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)

    def get_signal_history(self, n: int = 100) -> List[AnalystSignal]:
        """Get recent signal history."""
        return self.signal_history[-n:] if len(self.signal_history) >= n else self.signal_history.copy()

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
            'recent_signals': len(self.signal_history),
            'nas_enhanced_signals': self.nas_enhanced_signals,
            'nas_enhancement_rate': self.nas_enhanced_signals / self.signal_count if self.signal_count > 0 else 0.0,
            'shared_utilities_stats': {
                'feature_engine': self.feature_engine.get_performance_metrics(),
                'confidence_calculator': self.confidence_calculator.get_performance_metrics(),
                'fallback_analyzer': self.fallback_analyzer.get_performance_metrics(),
                'nas_enhancer': self.nas_enhancer.get_enhancement_stats()
            }
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