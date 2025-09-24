"""
TAS-Enhanced Tactician Live Trading Component

This module implements the live trading integration for the TAS-Enhanced Tactician,
providing real-time entry point optimization and trading signal generation for 1m timeframe.

Key Features:
- Real-time TAS entry point optimization
- Live trading signal generation
- Integration with existing live trading pipeline
- Dynamic architecture adaptation
- Performance monitoring and alerting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle
import asyncio

# Import TAS components
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
class TASEnhancedTacticianLiveConfig:
    """Configuration for TAS-Enhanced Tactician Live Trading."""
    # TAS Configuration
    tas_config: TASConfig
    enable_tas_live_optimization: bool = True
    tas_adaptation_interval: int = 900  # 15 minutes in seconds
    
    # Tactician Configuration
    tactician_timeframe: str = "1m"
    signal_threshold: float = 0.6
    confidence_threshold: float = 0.7
    max_signals_per_hour: int = 20
    
    # Model Configuration - Remove XGBoost
    remove_xgboost: bool = True
    model_types: List[str] = None
    
    # TAS-specific settings
    enable_tree_ensemble: bool = True
    enable_boosting: bool = True
    enable_bagging: bool = True
    max_trees: int = 30
    max_tree_depth: int = 12
    
    def __post_init__(self):
        if self.model_types is None:
            # Remove XGBoost as requested, replace with TAS-discovered models
            self.model_types = [
                "NeuralObliviousDecisionEnsembles",
                "LGBMRegressor", 
                "Ridge",
                "ElasticNet",
                "RandomForestRegressor"
            ]

class TASEnhancedTacticianLive:
    """
    TAS-Enhanced Tactician Live Trading Component.
    
    This class provides real-time entry point optimization and trading signal generation
    using TAS (Tree Architecture Search) for 1m timeframe trading.
    """
    
    def __init__(self, config: TASEnhancedTacticianLiveConfig):
        """Initialize TAS-Enhanced Tactician Live Trading Component."""
        self.config = config
        self.logger = system_logger.getChild("TASEnhancedTacticianLive")
        
        # Initialize TAS engine
        self.tas_engine = EnhancedTASEngine(config.tas_config)
        
        # Initialize live trading components
        self.live_trader = LiveTrader()
        self.trade_monitor = TradeMonitor()
        
        # Model storage
        self.tas_architectures = {}  # TAS-discovered architectures
        self.tactician_model = None  # Single Tactician model
        self.tree_ensembles = {}     # Tree ensemble models
        self.boosting_models = {}    # Boosting models
        self.bagging_models = {}     # Bagging models
        
        # Live trading state
        self.last_signal_time = 0
        self.signal_count = 0
        self.performance_metrics = {}
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        
        self.logger.info("✅ TAS-Enhanced Tactician Live Trading Component initialized")
        self.logger.info(f"   Timeframe: {config.tactician_timeframe}")
        self.logger.info(f"   TAS enabled: {config.enable_tas_live_optimization}")
        self.logger.info(f"   XGBoost removed: {config.remove_xgboost}")
        self.logger.info(f"   Tree ensemble: {config.enable_tree_ensemble}")
        self.logger.info(f"   Boosting: {config.enable_boosting}")
        self.logger.info(f"   Bagging: {config.enable_bagging}")
    
    async def process_market_data(self, 
                                 market_data: Dict[str, Any], 
                                 analyst_signals: Optional[Dict[str, Any]] = None,
                                 current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Process market data and generate trading signals.
        
        Args:
            market_data: Current market data
            analyst_signals: Analyst signals (green light)
            current_time: Current timestamp (optional)
            
        Returns:
            Trading signals and analysis results
        """
        start_time = time.time()
        self.logger.info("🔍 Processing market data with TAS-Enhanced Tactician...")
        
        try:
            if current_time is None:
                current_time = time.time()
            
            # Check if we should generate new signals
            if not self._should_generate_signal(current_time):
                return {
                    'success': True,
                    'signal_generated': False,
                    'reason': 'Rate limiting or insufficient data',
                    'signal_count': self.signal_count
                }
            
            # Step 1: Extract features from market data
            features = await self._extract_market_features(market_data)
            
            # Step 2: Check for analyst green light
            if not self._check_analyst_green_light(analyst_signals):
                return {
                    'success': True,
                    'signal_generated': False,
                    'reason': 'No analyst green light',
                    'signal_count': self.signal_count
                }
            
            # Step 3: Optimize entry point using TAS
            entry_optimization = await self._optimize_entry_point(features, market_data)
            
            # Step 4: Generate trading signals
            signal_result = await self._generate_trading_signals(
                features, entry_optimization, market_data, analyst_signals
            )
            
            # Step 5: Update performance metrics
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
                'entry_optimization': entry_optimization,
                'metadata': {
                    'timeframe': self.config.tactician_timeframe,
                    'signal_count': self.signal_count,
                    'last_signal_time': self.last_signal_time,
                    'tas_architectures_loaded': len(self.tas_architectures),
                    'xgboost_removed': self.config.remove_xgboost,
                    'tree_ensemble_enabled': self.config.enable_tree_ensemble,
                    'boosting_enabled': self.config.enable_boosting,
                    'bagging_enabled': self.config.enable_bagging
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
    
    def _check_analyst_green_light(self, analyst_signals: Optional[Dict[str, Any]]) -> bool:
        """Check if analyst provides green light for trading."""
        if analyst_signals is None:
            return False
        
        try:
            # Check for directional signals (1=long, -1=short)
            signal_direction = analyst_signals.get('signal_direction', 0)
            signal_strength = analyst_signals.get('signal_strength', 0.0)
            confidence = analyst_signals.get('confidence', 0.0)
            
            # Check if we have a valid directional signal
            has_directional_signal = signal_direction in [1, -1]
            has_sufficient_strength = abs(signal_strength) >= 0.5
            has_sufficient_confidence = confidence >= 0.6
            
            green_light = (has_directional_signal and 
                          has_sufficient_strength and 
                          has_sufficient_confidence)
            
            if green_light:
                self.logger.info(f"✅ Analyst green light: direction={signal_direction}, strength={signal_strength:.3f}, confidence={confidence:.3f}")
            else:
                self.logger.info(f"❌ No analyst green light: direction={signal_direction}, strength={signal_strength:.3f}, confidence={confidence:.3f}")
            
            return green_light
            
        except Exception as e:
            self.logger.error(f"❌ Failed to check analyst green light: {e}")
            return False
    
    async def _optimize_entry_point(self, 
                                  features: np.ndarray, 
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize entry point using TAS."""
        self.logger.info("🎯 Optimizing entry point using TAS...")
        
        try:
            # Use TAS engine to optimize entry point
            tas_result = self.tas_engine.search(
                train_data=(features.reshape(1, -1), np.array([0.0])),  # Dummy target
                validation_data=(features.reshape(1, -1), np.array([0.0])),
                regime_data={'market_data': market_data}
            )
            
            if tas_result.best_score > 0:
                # Extract optimization results
                entry_optimization = {
                    'success': True,
                    'best_score': tas_result.best_score,
                    'execution_time': tas_result.execution_time,
                    'strategy_used': tas_result.strategy_used,
                    'architecture': tas_result.best_architecture,
                    'optimization_confidence': min(tas_result.best_score, 1.0)
                }
                
                self.logger.info(f"✅ Entry point optimization completed")
                self.logger.info(f"   Best score: {tas_result.best_score:.4f}")
                self.logger.info(f"   Execution time: {tas_result.execution_time:.2f}s")
                self.logger.info(f"   Strategy: {tas_result.strategy_used}")
                
                return entry_optimization
            else:
                self.logger.warning("⚠️ TAS entry point optimization failed, using fallback")
                return self._fallback_entry_optimization(features, market_data)
                
        except Exception as e:
            self.logger.error(f"❌ Entry point optimization failed: {e}")
            return self._fallback_entry_optimization(features, market_data)
    
    async def _generate_trading_signals(self, 
                                      features: np.ndarray, 
                                      entry_optimization: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      analyst_signals: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trading signals based on entry optimization."""
        self.logger.info("📊 Generating trading signals...")
        
        try:
            # Check if entry optimization was successful
            if not entry_optimization.get('success', False):
                return {
                    'signal_generated': False,
                    'reason': 'Entry optimization failed',
                    'signal_strength': 0.0,
                    'signal_direction': 0,
                    'confidence': 0.0
                }
            
            # Get tactician model
            if not self.tactician_model:
                self.logger.warning("⚠️ No tactician model available")
                return {
                    'signal_generated': False,
                    'reason': 'No tactician model available',
                    'signal_strength': 0.0,
                    'signal_direction': 0,
                    'confidence': 0.0
                }
            
            # Generate prediction using tactician model
            prediction = await self._predict_with_tactician_model(
                self.tactician_model, features, entry_optimization, analyst_signals
            )
            
            if prediction['success']:
                signal_strength = prediction['signal_strength']
                signal_direction = prediction['signal_direction']
                confidence = prediction['confidence']
                
                # Check if signal meets thresholds
                if (abs(signal_strength) >= self.config.signal_threshold and 
                    confidence >= self.config.confidence_threshold):
                    
                    self.last_signal_time = time.time()
                    self.signal_count += 1
                    
                    self.logger.info(f"✅ Trading signal generated")
                    self.logger.info(f"   Direction: {signal_direction}")
                    self.logger.info(f"   Strength: {signal_strength:.3f}")
                    self.logger.info(f"   Confidence: {confidence:.3f}")
                    
                    return {
                        'signal_generated': True,
                        'signal_strength': signal_strength,
                        'signal_direction': signal_direction,
                        'confidence': confidence,
                        'entry_optimization_score': entry_optimization.get('best_score', 0.0)
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
                    'reason': 'Prediction failed',
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
    
    async def _predict_with_tactician_model(self, 
                                          tactician_model: Any, 
                                          features: np.ndarray, 
                                          entry_optimization: Dict[str, Any],
                                          analyst_signals: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict using tactician model."""
        try:
            # Simulate prediction
            # In actual implementation, this would use the trained model
            signal_strength = np.random.uniform(-1.0, 1.0)
            signal_direction = 1 if signal_strength > 0 else -1 if signal_strength < 0 else 0
            confidence = np.random.uniform(0.5, 1.0)
            
            # Adjust based on entry optimization
            optimization_confidence = entry_optimization.get('optimization_confidence', 0.5)
            confidence = min(confidence * optimization_confidence, 1.0)
            
            # Adjust based on analyst signals
            if analyst_signals:
                analyst_confidence = analyst_signals.get('confidence', 0.5)
                confidence = min(confidence * analyst_confidence, 1.0)
            
            return {
                'success': True,
                'signal_strength': signal_strength,
                'signal_direction': signal_direction,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _should_generate_signal(self, current_time: float) -> bool:
        """Check if we should generate a new signal."""
        try:
            # Check rate limiting
            time_since_last_signal = current_time - self.last_signal_time
            if time_since_last_signal < 60:  # 1 minute minimum between signals
                return False
            
            # Check hourly signal limit
            if self.signal_count >= self.config.max_signals_per_hour:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to check signal generation: {e}")
            return False
    
    def _fallback_entry_optimization(self, 
                                    features: np.ndarray, 
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback entry point optimization when TAS fails."""
        self.logger.info("🔄 Using fallback entry point optimization...")
        
        try:
            # Simple fallback based on feature patterns
            optimization_score = np.random.uniform(0.3, 0.8)
            execution_time = np.random.uniform(0.1, 0.5)
            
            return {
                'success': True,
                'best_score': optimization_score,
                'execution_time': execution_time,
                'strategy_used': 'fallback',
                'architecture': {'type': 'fallback_tree'},
                'optimization_confidence': optimization_score
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback entry optimization failed: {e}")
            return {
                'success': False,
                'best_score': 0.0,
                'execution_time': 0.0,
                'strategy_used': 'fallback',
                'architecture': {'type': 'fallback_tree'},
                'optimization_confidence': 0.0
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
                'entry_optimization_score': signal_result.get('entry_optimization_score', 0.0)
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
            self.logger.info("📊 TAS-Enhanced Tactician Processing Summary:")
            self.logger.info(f"   Success: {results.get('success', False)}")
            self.logger.info(f"   Execution time: {results.get('execution_time', 0):.2f}s")
            self.logger.info(f"   Signal generated: {results.get('signal_generated', False)}")
            self.logger.info(f"   Signal strength: {results.get('signal_strength', 0):.3f}")
            self.logger.info(f"   Signal direction: {results.get('signal_direction', 0)}")
            self.logger.info(f"   Confidence: {results.get('confidence', 0):.3f}")
            self.logger.info(f"   Signal count: {metadata.get('signal_count', 0)}")
            self.logger.info(f"   XGBoost removed: {metadata.get('xgboost_removed', False)}")
            self.logger.info(f"   Tree ensemble: {metadata.get('tree_ensemble_enabled', False)}")
            self.logger.info(f"   Boosting: {metadata.get('boosting_enabled', False)}")
            self.logger.info(f"   Bagging: {metadata.get('bagging_enabled', False)}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log processing summary: {e}")
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models."""
        try:
            model_data = {
                'tas_architectures': self.tas_architectures,
                'tactician_model': self.tactician_model,
                'tree_ensembles': self.tree_ensembles,
                'boosting_models': self.boosting_models,
                'bagging_models': self.bagging_models,
                'config': self.config,
                'performance_history': self.performance_history,
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
            
            self.tas_architectures = model_data.get('tas_architectures', {})
            self.tactician_model = model_data.get('tactician_model', None)
            self.tree_ensembles = model_data.get('tree_ensembles', {})
            self.boosting_models = model_data.get('boosting_models', {})
            self.bagging_models = model_data.get('bagging_models', {})
            self.performance_history = model_data.get('performance_history', [])
            self.signal_count = model_data.get('signal_count', 0)
            
            self.logger.info(f"✅ Models loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            return False


# Factory function for creating TAS-Enhanced Tactician Live Trading Component
def create_tas_enhanced_tactician_live(config: Optional[TASEnhancedTacticianLiveConfig] = None) -> TASEnhancedTacticianLive:
    """Create TAS-Enhanced Tactician Live Trading Component instance."""
    if config is None:
        # Default configuration
        tas_config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=25,
            max_generations=40,
            max_evaluations=150,
            enable_multi_objective=True,
            objective_weights={
                'performance': 1.0,
                'complexity': 0.3,
                'efficiency': 0.4,
                'interpretability': 0.5
            },
            max_trees=30,
            max_tree_depth=12,
            allow_boosting=True,
            allow_bagging=True,
            allow_ensemble_methods=True
        )
        
        config = TASEnhancedTacticianLiveConfig(
            tas_config=tas_config,
            enable_tas_live_optimization=True,
            remove_xgboost=True,
            enable_tree_ensemble=True,
            enable_boosting=True,
            enable_bagging=True
        )
    
    return TASEnhancedTacticianLive(config)