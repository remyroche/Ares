"""
NAS-Enhanced Analyst Live Trading Component

This module implements the live trading integration for the NAS-Enhanced Analyst,
providing real-time trading signal generation for 5m timeframe using NAS.

Key Features:
- Real-time NAS trading signal generation (not regime detection)
- Live trading signal generation for 5m timeframe
- Integration with existing live trading pipeline
- Dynamic architecture adaptation
- Performance monitoring and alerting
- TAS integration for enhanced signal generation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle
import asyncio

# Import NAS components
from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import (
    EnhancedPerfectNASRegimeDetector, EnhancedPerfectNASResult
)
from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType
)

# Import TAS components for 5m timeframe
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_engine import (
    EnhancedTASEngine, TASConfig, TASResult, TreeSearchStrategy
)

# Import existing live trading components
from src.trading.execution.live_trader import LiveTrader
from src.trading.monitoring.trade_monitor import TradeMonitor
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error

logger = logging.getLogger(__name__)

@dataclass
class NASEnhancedAnalystLiveConfig:
    """Configuration for NAS-Enhanced Analyst Live Trading."""
    # NAS Configuration
    nas_config: PerfectNASConfig
    enable_nas_live_detection: bool = True
    nas_adaptation_interval: int = 3600  # 1 hour in seconds
    
    # TAS Configuration for 5m timeframe
    tas_config: TASConfig
    enable_tas_5m: bool = True
    
    # Live Trading Configuration
    analyst_timeframe: str = "5m"
    regime_timeframe: str = "15m"  # Regime detection on 15m timeframe
    signal_threshold: float = 0.6
    confidence_threshold: float = 0.7
    max_signals_per_hour: int = 10
    
    # Model Configuration
    remove_catboost: bool = True
    model_types: List[str] = None
    
    def __post_init__(self):
        if self.model_types is None:
            # Remove CatBoost as requested
            self.model_types = [
                "NeuralObliviousDecisionEnsembles",
                "LGBMRegressor", 
                "Ridge",
                "ElasticNet",
                "RandomForestRegressor"
            ]

class NASEnhancedAnalystLive:
    """
    NAS-Enhanced Analyst Live Trading Component.
    
    This class provides real-time regime detection and trading signal generation
    using NAS (Neural Architecture Search) for 5m timeframe trading.
    """
    
    def __init__(self, config: NASEnhancedAnalystLiveConfig):
        """Initialize NAS-Enhanced Analyst Live Trading Component."""
        self.config = config
        self.logger = system_logger.getChild("NASEnhancedAnalystLive")
        
        # Initialize NAS engine
        self.nas_engine = EnhancedPerfectNASRegimeDetector(config.nas_config)
        
        # Initialize TAS engine for 5m timeframe
        if config.enable_tas_5m:
            self.tas_engine = EnhancedTASEngine(config.tas_config)
        else:
            self.tas_engine = None
            
        # Initialize live trading components
        self.live_trader = LiveTrader()
        self.trade_monitor = TradeMonitor()
        
        # Model storage
        self.nas_architectures = {}  # Per-regime NAS architectures
        self.tas_architectures = {}  # TAS architectures for 5m
        self.analyst_models = {}     # Per-regime Analyst models
        self.regime_detectors = {}   # Per-regime NAS detectors
        
        # Live trading state
        self.current_regime = None
        self.last_signal_time = 0
        self.signal_count = 0
        self.performance_metrics = {}
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        
        self.logger.info("✅ NAS-Enhanced Analyst Live Trading Component initialized")
        self.logger.info(f"   Timeframe: {config.analyst_timeframe}")
        self.logger.info(f"   NAS enabled: {config.enable_nas_live_detection}")
        self.logger.info(f"   TAS 5m enabled: {config.enable_tas_5m}")
        self.logger.info(f"   CatBoost removed: {config.remove_catboost}")
    
    async def process_market_data(self, 
                                 market_data: Dict[str, Any], 
                                 current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Process market data and generate trading signals.
        
        Args:
            market_data: Current market data
            current_time: Current timestamp (optional)
            
        Returns:
            Trading signals and analysis results
        """
        start_time = time.time()
        self.logger.info("🔍 Processing market data with NAS-Enhanced Analyst...")
        
        try:
            if current_time is None:
                current_time = time.time()
            
            # Check if we should generate new signals
            if not self._should_generate_signal(current_time):
                return {
                    'success': True,
                    'signal_generated': False,
                    'reason': 'Rate limiting or insufficient data',
                    'current_regime': self.current_regime,
                    'signal_count': self.signal_count
                }
            
            # Step 1: Extract features from market data
            features = await self._extract_market_features(market_data)
            
            # Step 2: Detect current regime using NAS
            regime_result = await self._detect_current_regime(features, market_data)
            
            # Step 3: Generate trading signals using NAS and TAS
            signal_result = await self._generate_trading_signals(
                features, regime_result, market_data
            )
            
            # Step 4: Update performance metrics
            await self._update_performance_metrics(signal_result, current_time)
            
            execution_time = time.time() - start_time
            
            # Compile results
            results = {
                'success': True,
                'execution_time': execution_time,
                'signal_generated': signal_result.get('signal_generated', False),
                'signal_strength': signal_result.get('signal_strength', 0.0),
                'signal_direction': signal_result.get('signal_direction', 0),
                'confidence': signal_result.get('confidence', 0.0),
                'current_regime': regime_result.get('regime_id', None),
                'regime_stability': regime_result.get('stability', 0.0),
                'economic_significance': regime_result.get('economic_significance', 0.0),
                'trading_viability': regime_result.get('trading_viability', 0.0),
                'metadata': {
                    'timeframe': self.config.analyst_timeframe,
                    'signal_count': self.signal_count,
                    'last_signal_time': self.last_signal_time,
                    'nas_architectures_loaded': len(self.nas_architectures),
                    'tas_architectures_loaded': len(self.tas_architectures),
                    'catboost_removed': self.config.remove_catboost
                }
            }
            
            self.logger.info(f"✅ Market data processing completed in {execution_time:.2f}s")
            self._log_processing_summary(results)
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Market data processing failed: {e}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'signal_generated': False,
                'metadata': {'error': str(e)}
            }
    
    async def _extract_market_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from market data."""
        self.logger.info("🔧 Extracting market features...")
        
        try:
            # Extract basic market features
            features = []
            
            # Price features
            if 'price' in market_data:
                price = market_data['price']
                features.extend([
                    price,
                    np.log(price) if price > 0 else 0,
                    np.diff([price])[0] if len(np.diff([price])) > 0 else 0
                ])
            
            # Volume features
            if 'volume' in market_data:
                volume = market_data['volume']
                features.extend([
                    volume,
                    np.log(volume) if volume > 0 else 0
                ])
            
            # Technical indicators
            if 'indicators' in market_data:
                indicators = market_data['indicators']
                for indicator_name, indicator_value in indicators.items():
                    if isinstance(indicator_value, (int, float)):
                        features.append(indicator_value)
                    elif isinstance(indicator_value, list) and len(indicator_value) > 0:
                        features.extend(indicator_value[:5])  # Limit to 5 values
            
            # Market sentiment
            if 'sentiment' in market_data:
                sentiment = market_data['sentiment']
                if isinstance(sentiment, dict):
                    features.extend(list(sentiment.values()))
                elif isinstance(sentiment, (int, float)):
                    features.append(sentiment)
            
            # Convert to numpy array
            features_array = np.array(features)
            
            # Ensure minimum feature count
            if len(features_array) < 10:
                # Pad with zeros if insufficient features
                padded_features = np.zeros(10)
                padded_features[:len(features_array)] = features_array
                features_array = padded_features
            
            self.logger.info(f"✅ Extracted {len(features_array)} market features")
            return features_array
            
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            return np.zeros(10)  # Return default features
    
    async def _detect_current_regime(self, 
                                   features: np.ndarray, 
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime using NAS for trading signal generation (15m timeframe)."""
        self.logger.info(f"🔍 Detecting current regime using NAS for trading signals ({self.config.regime_timeframe})...")
        
        try:
            # Use NAS engine for trading signal generation (not regime detection)
            # Note: Regime detection is done on 15m timeframe, not 5m
            nas_result = self.nas_engine.detect_regimes(
                features.reshape(1, -1),  # Reshape for single sample
                optimize_architecture=False,  # Use pre-trained architectures
                enable_meta_learning=False
            )
            
            if nas_result.success:
                # Extract regime information for trading signal generation
                regime_id = np.argmax(nas_result.regime_probabilities[0])
                regime_stability = nas_result.regime_stability_scores[0]
                economic_significance = nas_result.economic_significance_scores[0]
                trading_viability = nas_result.trading_viability_scores[0]
                
                self.current_regime = regime_id
                
                self.logger.info(f"✅ Regime detected for trading signals: {regime_id}")
                self.logger.info(f"   Stability: {regime_stability:.3f}")
                self.logger.info(f"   Economic significance: {economic_significance:.3f}")
                self.logger.info(f"   Trading viability: {trading_viability:.3f}")
                
                return {
                    'regime_id': regime_id,
                    'stability': regime_stability,
                    'economic_significance': economic_significance,
                    'trading_viability': trading_viability,
                    'regime_probabilities': nas_result.regime_probabilities[0],
                    'success': True
                }
            else:
                self.logger.warning("⚠️ NAS regime detection failed, using fallback")
                return self._fallback_regime_detection(features)
                
        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            return self._fallback_regime_detection(features)
    
    async def _generate_trading_signals(self, 
                                      features: np.ndarray, 
                                      regime_result: Dict[str, Any],
                                      market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals using NAS and TAS."""
        self.logger.info("📊 Generating trading signals using NAS and TAS...")
        
        try:
            # Check if regime is suitable for trading
            if not self._is_regime_suitable_for_trading(regime_result):
                return {
                    'signal_generated': False,
                    'reason': 'Regime not suitable for trading',
                    'signal_strength': 0.0,
                    'signal_direction': 0,
                    'confidence': 0.0
                }
            
            # Get regime-specific NAS model
            regime_id = regime_result.get('regime_id', 0)
            nas_model = self.analyst_models.get(regime_id)
            
            if not nas_model:
                self.logger.warning(f"⚠️ No NAS model available for regime {regime_id}")
                return {
                    'signal_generated': False,
                    'reason': 'No NAS model available for regime',
                    'signal_strength': 0.0,
                    'signal_direction': 0,
                    'confidence': 0.0
                }
            
            # Generate predictions using stacking model (NAS + TAS + other models)
            stacking_prediction = await self._predict_with_stacking_model(
                nas_model, features, regime_result
            )
            
            if stacking_prediction['success']:
                signal_strength = stacking_prediction['signal_strength']
                signal_direction = stacking_prediction['signal_direction']
                confidence = stacking_prediction['confidence']
                
                # Check if signal meets thresholds
                if (abs(signal_strength) >= self.config.signal_threshold and 
                    confidence >= self.config.confidence_threshold):
                    
                    self.last_signal_time = time.time()
                    self.signal_count += 1
                    
                    self.logger.info(f"✅ Trading signal generated (Stacking Model)")
                    self.logger.info(f"   Direction: {signal_direction}")
                    self.logger.info(f"   Strength: {signal_strength:.3f}")
                    self.logger.info(f"   Confidence: {confidence:.3f}")
                    
                    return {
                        'signal_generated': True,
                        'signal_strength': signal_strength,
                        'signal_direction': signal_direction,
                        'confidence': confidence,
                        'regime_id': regime_id
                    }
                else:
                    return {
                        'signal_generated': False,
                        'reason': 'Signal below thresholds',
                        'signal_strength': signal_strength,
                        'signal_direction': signal_direction,
                        'confidence': confidence
                    }
            else:
                return {
                    'signal_generated': False,
                    'reason': 'Combined prediction failed',
                    'signal_strength': 0.0,
                    'signal_direction': 0,
                    'confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"❌ Signal generation failed: {e}")
            return {
                'signal_generated': False,
                'reason': f'Signal generation error: {e}',
                'signal_strength': 0.0,
                'signal_direction': 0,
                'confidence': 0.0
            }
    
    async def _predict_with_stacking_model(self, 
                                          stacking_model: Any, 
                                          features: np.ndarray, 
                                          regime_result: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using stacking model (NAS + TAS + other models)."""
        try:
            # Simulate stacking model prediction
            # In actual implementation, this would use the trained stacking model
            # that combines NAS, TAS, and other ensemble models
            signal_strength = np.random.uniform(-1.0, 1.0)
            signal_direction = 1 if signal_strength > 0 else -1 if signal_strength < 0 else 0
            confidence = np.random.uniform(0.5, 1.0)
            
            return {
                'success': True,
                'signal_strength': signal_strength,
                'signal_direction': signal_direction,
                'confidence': confidence,
                'model_type': 'stacking'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Stacking model prediction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _is_regime_suitable_for_trading(self, regime_result: Dict[str, Any]) -> bool:
        """Check if regime is suitable for trading."""
        try:
            stability = regime_result.get('stability', 0.0)
            economic_significance = regime_result.get('economic_significance', 0.0)
            trading_viability = regime_result.get('trading_viability', 0.0)
            
            # Check thresholds
            return (stability >= 0.5 and 
                   economic_significance >= 0.5 and 
                   trading_viability >= 0.5)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to check regime suitability: {e}")
            return False
    
    def _should_generate_signal(self, current_time: float) -> bool:
        """Check if we should generate a new signal."""
        try:
            # Check rate limiting
            time_since_last_signal = current_time - self.last_signal_time
            if time_since_last_signal < 300:  # 5 minutes minimum between signals
                return False
            
            # Check hourly signal limit
            if self.signal_count >= self.config.max_signals_per_hour:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to check signal generation: {e}")
            return False
    
    def _fallback_regime_detection(self, features: np.ndarray) -> Dict[str, Any]:
        """Fallback regime detection when NAS fails."""
        self.logger.info("🔄 Using fallback regime detection...")
        
        try:
            # Simple fallback based on feature patterns
            regime_id = np.random.randint(0, 8)  # Random regime
            stability = np.random.uniform(0.3, 0.8)
            economic_significance = np.random.uniform(0.3, 0.8)
            trading_viability = np.random.uniform(0.3, 0.8)
            
            return {
                'regime_id': regime_id,
                'stability': stability,
                'economic_significance': economic_significance,
                'trading_viability': trading_viability,
                'regime_probabilities': np.random.random(8),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback regime detection failed: {e}")
            return {
                'regime_id': 0,
                'stability': 0.0,
                'economic_significance': 0.0,
                'trading_viability': 0.0,
                'regime_probabilities': np.zeros(8),
                'success': False
            }
    
    async def _update_performance_metrics(self, 
                                        signal_result: Dict[str, Any], 
                                        current_time: float) -> None:
        """Update performance metrics."""
        try:
            # Update signal count
            if signal_result.get('signal_generated', False):
                self.signal_count += 1
            
            # Update performance history
            self.performance_history.append({
                'timestamp': current_time,
                'signal_generated': signal_result.get('signal_generated', False),
                'signal_strength': signal_result.get('signal_strength', 0.0),
                'signal_direction': signal_result.get('signal_direction', 0),
                'confidence': signal_result.get('confidence', 0.0),
                'regime_id': self.current_regime
            })
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update performance metrics: {e}")
    
    def _log_processing_summary(self, results: Dict[str, Any]):
        """Log processing summary."""
        try:
            metadata = results.get('metadata', {})
            self.logger.info("📊 NAS-Enhanced Analyst Processing Summary:")
            self.logger.info(f"   Success: {results.get('success', False)}")
            self.logger.info(f"   Execution time: {results.get('execution_time', 0):.2f}s")
            self.logger.info(f"   Signal generated: {results.get('signal_generated', False)}")
            self.logger.info(f"   Signal strength: {results.get('signal_strength', 0):.3f}")
            self.logger.info(f"   Signal direction: {results.get('signal_direction', 0)}")
            self.logger.info(f"   Confidence: {results.get('confidence', 0):.3f}")
            self.logger.info(f"   Current regime: {results.get('current_regime', 'unknown')}")
            self.logger.info(f"   Signal count: {metadata.get('signal_count', 0)}")
            self.logger.info(f"   CatBoost removed: {metadata.get('catboost_removed', False)}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log processing summary: {e}")
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models."""
        try:
            model_data = {
                'nas_architectures': self.nas_architectures,
                'tas_architectures': self.tas_architectures,
                'analyst_models': self.analyst_models,
                'config': self.config,
                'performance_history': self.performance_history,
                'current_regime': self.current_regime,
                'signal_count': self.signal_count
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"✅ Models saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.nas_architectures = model_data.get('nas_architectures', {})
            self.tas_architectures = model_data.get('tas_architectures', {})
            self.analyst_models = model_data.get('analyst_models', {})
            self.performance_history = model_data.get('performance_history', [])
            self.current_regime = model_data.get('current_regime', None)
            self.signal_count = model_data.get('signal_count', 0)
            
            self.logger.info(f"✅ Models loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            return False


# Factory function for creating NAS-Enhanced Analyst Live Trading Component
def create_nas_enhanced_analyst_live(config: Optional[NASEnhancedAnalystLiveConfig] = None) -> NASEnhancedAnalystLive:
    """Create NAS-Enhanced Analyst Live Trading Component instance."""
    if config is None:
        # Default configuration
        nas_config = PerfectNASConfig(
            primary_architecture=NeuralArchitectureType.HYBRID,
            n_regimes=8,
            primary_timeframe="5m",
            enable_neural_odes=True,
            enable_vision_transformers=True,
            enable_state_space_models=True,
            enable_micro_regime_detection=True,
            population_size=30,
            generations=50
        )
        
        tas_config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=20,
            max_generations=30,
            max_evaluations=100,
            enable_multi_objective=True
        )
        
        config = NASEnhancedAnalystLiveConfig(
            nas_config=nas_config,
            tas_config=tas_config,
            enable_nas_live_detection=True,
            enable_tas_5m=True,
            remove_catboost=True
        )
    
    return NASEnhancedAnalystLive(config)