"""
Regime Detector

Main regime detection engine that integrates with existing ML components
to detect and classify market regimes with confidence scores.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from .regime_classifier import RegimeClassifier
from .regime_analyzer import RegimeAnalyzer
from .regime_weights import RegimeWeightManager
from ..config.regime_config import RegimeConfig, RegimeType, RegimeWeight

logger = system_logger.getChild('RegimeDetector')

@dataclass
class RegimeDetection:
    """Regime detection result."""
    timestamp: datetime
    primary_regime: RegimeType
    regime_probabilities: Dict[RegimeType, float]
    confidence: float
    regime_strength: float
    transition_probability: float
    features_used: Dict[str, Any]
    detection_metadata: Dict[str, Any]

class RegimeDetector:
    """
    Main regime detection engine.
    
    Integrates with existing HMM and market analysis components to provide
    comprehensive regime detection with confidence scores and transition analysis.
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.logger = logger.getChild('RegimeDetector')
        
        # Core components
        self.classifier = RegimeClassifier(config)
        self.analyzer = RegimeAnalyzer(config)
        self.weight_manager = RegimeWeightManager(config)
        
        # State management
        self.current_regime: Optional[RegimeType] = None
        self.regime_history: List[RegimeDetection] = []
        self.last_detection_time: Optional[datetime] = None
        
        # Performance tracking
        self.detection_count = 0
        self.accuracy_metrics: Dict[str, float] = {}
        
        # Integration with existing components
        self.hmm_model = None
        self.market_analyzer = None
        
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize regime detector with existing ML components."""
        try:
            self.logger.info("Initializing Regime Detector...")
            
            # Initialize classifier
            await self.classifier.initialize()
            
            # Initialize analyzer
            await self.analyzer.initialize()
            
            # Initialize weight manager
            await self.weight_manager.initialize()
            
            # Load existing HMM model if available
            await self._load_hmm_model()
            
            # Load market analyzer if available
            await self._load_market_analyzer()
            
            self.logger.info("✅ Regime Detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Regime Detector: {e}")
            return False
    
    async def _load_hmm_model(self):
        """Load existing HMM model for regime detection."""
        try:
            # Try to load from existing HMM components
            from src.training.steps.market_analysis.enhanced_hmm_training import EnhancedHMMTraining
            
            # This would integrate with your existing HMM training
            self.logger.info("🔍 Loading HMM model for regime detection...")
            
            # Placeholder for HMM model loading
            # In practice, this would load your trained HMM model
            self.hmm_model = None  # Load actual model here
            
        except Exception as e:
            self.logger.warning(f"⚠️ HMM model not available: {e}")
    
    async def _load_market_analyzer(self):
        """Load existing market analyzer for regime detection."""
        try:
            # Try to load from existing market analysis components
            from src.training.steps.market_analysis.enhanced_market_analysis import EnhancedMarketAnalysis
            
            # This would integrate with your existing market analysis
            self.logger.info("🔍 Loading Market Analyzer for regime detection...")
            
            # Placeholder for market analyzer loading
            self.market_analyzer = None  # Load actual analyzer here
            
        except Exception as e:
            self.logger.warning(f"⚠️ Market Analyzer not available: {e}")
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="detect_regime")
    async def detect_regime(self, market_data: pd.DataFrame) -> RegimeDetection:
        """
        Detect current market regime from market data.
        
        Args:
            market_data: DataFrame with OHLCV data and features
            
        Returns:
            RegimeDetection: Current regime with probabilities and confidence
        """
        try:
            if market_data.empty:
                raise ValueError("Market data is empty")
            
            # Get latest data point
            latest_data = market_data.iloc[-1]
            timestamp = latest_data.get('timestamp', datetime.now())
            
            # Extract features for regime detection
            features = await self._extract_regime_features(market_data)
            
            # Classify regime using multiple methods
            regime_probabilities = await self._classify_regime(features, market_data)
            
            # Determine primary regime
            primary_regime = max(regime_probabilities.items(), key=lambda x: x[1])[0]
            confidence = regime_probabilities[primary_regime]
            
            # Calculate regime strength
            regime_strength = await self._calculate_regime_strength(regime_probabilities)
            
            # Calculate transition probability
            transition_prob = await self._calculate_transition_probability(primary_regime)
            
            # Create detection result
            detection = RegimeDetection(
                timestamp=timestamp,
                primary_regime=primary_regime,
                regime_probabilities=regime_probabilities,
                confidence=confidence,
                regime_strength=regime_strength,
                transition_probability=transition_prob,
                features_used=features,
                detection_metadata={
                    'detection_method': 'ensemble',
                    'model_version': '1.0',
                    'processing_time_ms': 0  # Will be set by decorator
                }
            )
            
            # Update state
            self.current_regime = primary_regime
            self.regime_history.append(detection)
            self.last_detection_time = timestamp
            self.detection_count += 1
            
            # Maintain history size
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            # Update performance metrics
            await self._update_performance_metrics(detection)
            
            self.logger.debug(f"Regime detected: {primary_regime.value} (confidence: {confidence:.3f})")
            
            return detection
            
        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            raise
    
    async def _extract_regime_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for regime detection."""
        try:
            features = {}
            
            # Basic price features
            if len(market_data) >= 20:
                features.update({
                    'price_change': (market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20],
                    'volatility': market_data['close'].pct_change().rolling(20).std().iloc[-1],
                    'volume_ratio': market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1],
                })
            
            # Technical indicators
            if len(market_data) >= 50:
                # RSI
                delta = market_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
                
                # Moving averages
                features['ma_ratio'] = market_data['close'].rolling(20).mean().iloc[-1] / market_data['close'].rolling(50).mean().iloc[-1]
            
            # Regime-specific features
            features.update({
                'trend_strength': self._calculate_trend_strength(market_data),
                'volatility_regime': self._classify_volatility_regime(market_data),
                'momentum_score': self._calculate_momentum_score(market_data),
            })
            
            return features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature extraction failed: {e}")
            return {}
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength indicator."""
        try:
            if len(market_data) < 20:
                return 0.0
            
            # Linear regression slope
            x = np.arange(len(market_data[-20:]))
            y = market_data['close'].iloc[-20:].values
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize by price
            normalized_slope = slope / market_data['close'].iloc[-1]
            return min(max(normalized_slope * 100, -1), 1)  # Clamp to [-1, 1]
            
        except:
            return 0.0
    
    def _classify_volatility_regime(self, market_data: pd.DataFrame) -> float:
        """Classify volatility regime."""
        try:
            if len(market_data) < 20:
                return 0.0
            
            current_vol = market_data['close'].pct_change().rolling(10).std().iloc[-1]
            historical_vol = market_data['close'].pct_change().rolling(50).std().iloc[-1]
            
            if historical_vol > 0:
                return min(max(current_vol / historical_vol, 0), 2)  # Clamp to [0, 2]
            return 1.0
            
        except:
            return 1.0
    
    def _calculate_momentum_score(self, market_data: pd.DataFrame) -> float:
        """Calculate momentum score."""
        try:
            if len(market_data) < 10:
                return 0.0
            
            # Price momentum
            price_momentum = (market_data['close'].iloc[-1] - market_data['close'].iloc[-10]) / market_data['close'].iloc[-10]
            
            # Volume momentum
            volume_momentum = (market_data['volume'].iloc[-1] - market_data['volume'].rolling(10).mean().iloc[-1]) / market_data['volume'].rolling(10).mean().iloc[-1]
            
            # Combined momentum
            combined_momentum = (price_momentum + volume_momentum) / 2
            return min(max(combined_momentum, -1), 1)  # Clamp to [-1, 1]
            
        except:
            return 0.0
    
    async def _classify_regime(self, features: Dict[str, Any], market_data: pd.DataFrame) -> Dict[RegimeType, float]:
        """Classify regime using ensemble methods."""
        try:
            # Use classifier for primary classification
            regime_probs = await self.classifier.classify(features, market_data)
            
            # Apply regime weights
            weighted_probs = {}
            for regime, prob in regime_probs.items():
                weight = self.config.get_regime_weight(regime)
                weighted_probs[regime] = prob * weight
            
            # Normalize probabilities
            total_prob = sum(weighted_probs.values())
            if total_prob > 0:
                for regime in weighted_probs:
                    weighted_probs[regime] /= total_prob
            
            return weighted_probs
            
        except Exception as e:
            self.logger.error(f"❌ Regime classification failed: {e}")
            # Return default probabilities
            return {RegimeType.SIDEWAYS: 1.0}
    
    async def _calculate_regime_strength(self, regime_probabilities: Dict[RegimeType, float]) -> float:
        """Calculate overall regime strength."""
        try:
            # Regime strength is the difference between highest and second highest probability
            sorted_probs = sorted(regime_probabilities.values(), reverse=True)
            if len(sorted_probs) >= 2:
                return sorted_probs[0] - sorted_probs[1]
            return sorted_probs[0] if sorted_probs else 0.0
            
        except:
            return 0.0
    
    async def _calculate_transition_probability(self, current_regime: RegimeType) -> float:
        """Calculate probability of regime transition."""
        try:
            if len(self.regime_history) < 2:
                return 0.0
            
            # Count recent regime changes
            recent_detections = self.regime_history[-10:]
            regime_changes = 0
            
            for i in range(1, len(recent_detections)):
                if recent_detections[i].primary_regime != recent_detections[i-1].primary_regime:
                    regime_changes += 1
            
            return regime_changes / len(recent_detections)
            
        except:
            return 0.0
    
    async def _update_performance_metrics(self, detection: RegimeDetection):
        """Update performance metrics for regime detection."""
        try:
            # This would integrate with your existing performance tracking
            # For now, just track basic metrics
            self.accuracy_metrics.update({
                'total_detections': self.detection_count,
                'avg_confidence': np.mean([d.confidence for d in self.regime_history[-100:]]),
                'regime_stability': 1.0 - detection.transition_probability,
            })
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance metrics update failed: {e}")
    
    def get_current_regime(self) -> Optional[RegimeType]:
        """Get current regime."""
        return self.current_regime
    
    def get_regime_history(self, limit: int = 100) -> List[RegimeDetection]:
        """Get recent regime detection history."""
        return self.regime_history[-limit:] if self.regime_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get regime detection performance metrics."""
        return self.accuracy_metrics.copy()
    
    async def stop(self):
        """Stop regime detector."""
        try:
            self.logger.info("🛑 Stopping Regime Detector...")
            
            # Stop components
            await self.classifier.stop()
            await self.analyzer.stop()
            await self.weight_manager.stop()
            
            self.logger.info("✅ Regime Detector stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Regime Detector: {e}")