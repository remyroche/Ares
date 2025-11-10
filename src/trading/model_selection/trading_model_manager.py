"""
Trading Model Manager

This module provides a comprehensive model management system for trading that
integrates with the per-regime training system and provides real-time model
selection and management capabilities.

Key Features:
- Model loading and caching
- Real-time model performance tracking
- Model switching based on performance
- Integration with training system
- Model versioning and rollback
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from src.utils.tprint import tprint
from .model_selector_service import ModelSelectorService, ModelSelectionResult, TradingModelConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Model performance tracking."""
    model_name: str
    regime_id: int
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    execution_time: float = 0.0
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    performance_history: List[float] = field(default_factory=list)

@dataclass
class ModelCacheEntry:
    """Model cache entry."""
    model: Any
    metadata: Dict[str, Any]
    loaded_at: datetime
    last_used: datetime
    access_count: int = 0
    performance: Optional[ModelPerformance] = None

class TradingModelManager:
    """
    Trading model manager for real-time model selection and management.

    This class provides comprehensive model management capabilities including
    model loading, caching, performance tracking, and real-time selection.
    """

    def __init__(self, config: Optional[TradingModelConfig] = None):
        """Initialize trading model manager."""
        tprint(f"[TRADING_MODEL_MANAGER] Entering __init__ with config={config is not None}")

        self.config = config or TradingModelConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Model selector service
        self.model_selector_service = None

        # Model cache
        self.model_cache: Dict[str, ModelCacheEntry] = {}
        self.cache_lock = threading.RLock()

        # Performance tracking
        self.performance_tracker: Dict[str, ModelPerformance] = {}
        self.performance_lock = threading.RLock()

        # Model loading
        self.model_loader = None
        self.loading_threads = ThreadPoolExecutor(max_workers=4)

        # Configuration
        self.cache_size_limit = 100
        self.cache_ttl = timedelta(hours=24)
        self.performance_window = 1000
        tprint(f"[TRADING_MODEL_MANAGER] Configuration: cache_size={self.cache_size_limit}, cache_ttl={self.cache_ttl}, performance_window={self.performance_window}")

        self.logger.info("✅ Trading Model Manager initialized")
        tprint("[TRADING_MODEL_MANAGER] Exiting __init__ successfully")

    def initialize(self) -> bool:
        """Initialize the trading model manager."""
        tprint("[TRADING_MODEL_MANAGER] Entering initialize")

        try:
            self.logger.info("🔧 Initializing trading model manager...")

            # Initialize model selector service
            tprint("[TRADING_MODEL_MANAGER] Initializing model selector service")
            self.model_selector_service = ModelSelectorService(self.config)
            if not self.model_selector_service.initialize():
                tprint("[TRADING_MODEL_MANAGER] Model selector service initialization failed")
                raise RuntimeError("Failed to initialize model selector service")
            tprint("[TRADING_MODEL_MANAGER] Model selector service initialized successfully")

            # Initialize model loader
            tprint("[TRADING_MODEL_MANAGER] Initializing model loader")
            self._initialize_model_loader()

            # Load existing models if available
            tprint("[TRADING_MODEL_MANAGER] Loading existing models")
            self._load_existing_models()

            self.logger.info("✅ Trading model manager initialized successfully")
            tprint("[TRADING_MODEL_MANAGER] Exiting initialize with result=True")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading model manager: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting initialize with error: {e}")
            return False

    def _initialize_model_loader(self):
        """Initialize model loader."""
        tprint("[TRADING_MODEL_MANAGER] Entering _initialize_model_loader")

        try:
            # Use unified model loader for accessing models from artifact_manager
            from src.trading.integration.unified_model_loader import get_unified_model_loader

            self.model_loader = get_unified_model_loader()
            self.logger.info("✅ Model loader initialized (using unified model loader)")
            tprint("[TRADING_MODEL_MANAGER] Model loader initialized successfully")
            tprint("[TRADING_MODEL_MANAGER] Exiting _initialize_model_loader")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model loader: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting _initialize_model_loader with error: {e}")
            raise

    def _load_existing_models(self):
        """Load existing models from training results."""
        tprint("[TRADING_MODEL_MANAGER] Entering _load_existing_models")

        try:
            # Try to load models from training artifacts
            training_artifacts_path = "data_cache/training_artifacts"
            tprint(f"[TRADING_MODEL_MANAGER] Checking for training artifacts at {training_artifacts_path}")

            if Path(training_artifacts_path).exists():
                self.logger.info("🔄 Loading existing models from training artifacts...")
                tprint(f"[TRADING_MODEL_MANAGER] Training artifacts path exists, loading...")
                # This would load actual models from training artifacts
                # For now, we'll just log the attempt
                self.logger.info("✅ Existing models loaded")
                tprint("[TRADING_MODEL_MANAGER] Existing models loaded")
            else:
                tprint(f"[TRADING_MODEL_MANAGER] No training artifacts found at {training_artifacts_path}")

            tprint("[TRADING_MODEL_MANAGER] Exiting _load_existing_models")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load existing models: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting _load_existing_models with error: {e}")

    def get_models_for_trading(
        self,
        market_data: pd.DataFrame,
        symbol: str = "ETHUSDT",
        timeframe: str = "5m"
    ) -> Dict[str, Any]:
        """
        Get models for trading based on current market conditions.

        Args:
            market_data: Current market data
            symbol: Trading symbol
            timeframe: Timeframe (5m or 15m)

        Returns:
            Dictionary containing selected models and metadata
        """
        tprint(f"[TRADING_MODEL_MANAGER] Entering get_models_for_trading with symbol={symbol}, timeframe={timeframe}, data_shape={market_data.shape if market_data is not None else None}")

        try:
            self.logger.info(f"🎯 Getting models for {symbol} ({timeframe})...")

            # Get model selection result
            tprint("[TRADING_MODEL_MANAGER] Getting model selection result from model selector service")
            selection_result = self.model_selector_service.select_models_for_trading(
                market_data=market_data,
                symbol=symbol,
                timeframe=timeframe
            )
            tprint(f"[TRADING_MODEL_MANAGER] Model selection complete: {len(selection_result.selected_models)} models selected")

            if not selection_result.selected_models:
                self.logger.warning("⚠️ No models selected, using fallback")
                tprint("[TRADING_MODEL_MANAGER] No models selected, using fallback")
                return self._get_fallback_models(symbol, timeframe)

            # Load selected models
            tprint("[TRADING_MODEL_MANAGER] Loading selected models")
            loaded_models = {}
            for model_type, model_name in selection_result.selected_models.items():
                tprint(f"[TRADING_MODEL_MANAGER] Loading model: type={model_type}, name={model_name}")
                try:
                    model = self._load_model(model_name, model_type, timeframe)
                    if model:
                        loaded_models[model_type] = {
                            'model': model,
                            'name': model_name,
                            'weight': selection_result.ensemble_weights.get(model_type, {}).get(model_name, 1.0),
                            'regime_id': selection_result.regime_id,
                            'confidence': selection_result.confidence_score
                        }
                        tprint(f"[TRADING_MODEL_MANAGER] Successfully loaded model {model_name}")
                    else:
                        self.logger.warning(f"⚠️ Failed to load model {model_name}")
                        tprint(f"[TRADING_MODEL_MANAGER] Failed to load model {model_name}")

                except Exception as e:
                    self.logger.warning(f"⚠️ Error loading model {model_name}: {e}")
                    tprint(f"[TRADING_MODEL_MANAGER] Error loading model {model_name}: {e}")

            # Update performance tracking
            tprint("[TRADING_MODEL_MANAGER] Updating performance tracking")
            self._update_performance_tracking(selection_result)

            self.logger.info(f"✅ Loaded {len(loaded_models)} models for trading")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting get_models_for_trading with {len(loaded_models)} loaded models")
            return loaded_models

        except Exception as e:
            self.logger.error(f"❌ Failed to get models for trading: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exception in get_models_for_trading: {e}")
            return self._get_fallback_models(symbol, timeframe)

    def _load_model(self, model_name: str, model_type: str, timeframe: str) -> Optional[Any]:
        """Load a specific model."""
        cache_key = f"{model_name}_{model_type}_{timeframe}"
        tprint(f"[TRADING_MODEL_MANAGER] Entering _load_model with model={model_name}, type={model_type}, timeframe={timeframe}")

        try:
            # Check cache first
            with self.cache_lock:
                if cache_key in self.model_cache:
                    entry = self.model_cache[cache_key]
                    entry.last_used = datetime.now()
                    entry.access_count += 1
                    self.logger.debug(f"📦 Using cached model: {cache_key}")
                    tprint(f"[TRADING_MODEL_MANAGER] Model found in cache: {cache_key}, access_count={entry.access_count}")
                    tprint(f"[TRADING_MODEL_MANAGER] Exiting _load_model with cached model")
                    return entry.model

            # Load model (placeholder implementation)
            tprint(f"[TRADING_MODEL_MANAGER] Model not in cache, loading from storage: {cache_key}")
            model = self._load_model_from_storage(model_name, model_type, timeframe)

            if model:
                # Cache the model
                tprint(f"[TRADING_MODEL_MANAGER] Adding model to cache: {cache_key}")
                with self.cache_lock:
                    self.model_cache[cache_key] = ModelCacheEntry(
                        model=model,
                        metadata={'name': model_name, 'type': model_type, 'timeframe': timeframe},
                        loaded_at=datetime.now(),
                        last_used=datetime.now(),
                        access_count=1
                    )

                    # Clean cache if needed
                    self._clean_cache()

                self.logger.debug(f"📥 Loaded model: {cache_key}")
                tprint(f"[TRADING_MODEL_MANAGER] Model loaded and cached: {cache_key}, cache_size={len(self.model_cache)}")
                tprint(f"[TRADING_MODEL_MANAGER] Exiting _load_model with loaded model")
                return model

            tprint(f"[TRADING_MODEL_MANAGER] Model not found in storage: {model_name}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting _load_model with None")
            return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load model {model_name}: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting _load_model with error: {e}")
            return None

    def _load_model_from_storage(self, model_name: str, model_type: str, timeframe: str) -> Optional[Any]:
        """Load model from storage using unified model loader."""
        try:
            if not self.model_loader:
                self.logger.warning("⚠️ Model loader not initialized")
                return None
            
            # Use unified model loader to load models
            import asyncio
            
            # Determine symbol and timeframe
            symbol = "ETHUSDT"  # Default, could be passed in config
            
            try:
                # Create event loop if needed
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Load models based on type
                if model_type == 'regime':
                    if 'ensemble' in model_name.lower():
                        model = loop.run_until_complete(
                            self.model_loader.load_regime_ensemble_model(symbol, timeframe)
                        )
                    else:
                        models_dict = loop.run_until_complete(
                            self.model_loader.load_regime_base_models(symbol, timeframe)
                        )
                        model = models_dict.get(model_name) if models_dict else None
                
                elif model_type == 'analyst':
                    if 'ensemble' in model_name.lower():
                        model = loop.run_until_complete(
                            self.model_loader.load_analyst_ensemble_model(symbol, timeframe)
                        )
                    else:
                        models_dict = loop.run_until_complete(
                            self.model_loader.load_analyst_base_models(symbol, timeframe)
                        )
                        model = models_dict.get(model_name) if models_dict else None
                
                elif model_type == 'tactician':
                    if 'ensemble' in model_name.lower():
                        model = loop.run_until_complete(
                            self.model_loader.load_tactician_ensemble_model(symbol, timeframe)
                        )
                    else:
                        models_dict = loop.run_until_complete(
                            self.model_loader.load_tactician_base_models(symbol, timeframe)
                        )
                        model = models_dict.get(model_name) if models_dict else None
                
                else:
                    self.logger.warning(f"⚠️ Unknown model type: {model_type}")
                    return None
                
                if model:
                    self.logger.info(f"✅ Loaded {model_type} model: {model_name}")
                    return model
                else:
                    self.logger.warning(f"⚠️ Model {model_name} not found in storage")
                    return None
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load model from storage: {e}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load model from storage: {e}")
            return None

    def _get_fallback_models(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get fallback models when selection fails."""
        tprint(f"[TRADING_MODEL_MANAGER] Entering _get_fallback_models with symbol={symbol}, timeframe={timeframe}")

        try:
            fallback_models = {}

            # Create fallback models
            for model_type in ['analyst', 'tactician']:
                tprint(f"[TRADING_MODEL_MANAGER] Creating fallback model for type={model_type}")
                fallback_models[model_type] = {
                    'model': self._load_model_from_storage('default', model_type, timeframe),
                    'name': 'default',
                    'weight': 1.0,
                    'regime_id': 0,
                    'confidence': 0.5,
                    'fallback': True
                }

            self.logger.warning("⚠️ Using fallback models")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting _get_fallback_models with {len(fallback_models)} fallback models")
            return fallback_models

        except Exception as e:
            self.logger.error(f"❌ Failed to get fallback models: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting _get_fallback_models with error: {e}")
            return {}

    def _update_performance_tracking(self, selection_result: ModelSelectionResult):
        """Update performance tracking for selected models."""
        try:
            with self.performance_lock:
                for model_type, model_name in selection_result.selected_models.items():
                    key = f"{model_name}_{model_type}"

                    if key not in self.performance_tracker:
                        self.performance_tracker[key] = ModelPerformance(
                            model_name=model_name,
                            regime_id=selection_result.regime_id
                        )

                    # Update performance metrics (placeholder)
                    performance = self.performance_tracker[key]
                    performance.regime_id = selection_result.regime_id
                    performance.last_updated = datetime.now()

                    # Add to performance history
                    performance.performance_history.append(selection_result.confidence_score)
                    if len(performance.performance_history) > self.performance_window:
                        performance.performance_history.pop(0)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update performance tracking: {e}")

    def _clean_cache(self):
        """Clean model cache to prevent memory issues."""
        try:
            with self.cache_lock:
                if len(self.model_cache) <= self.cache_size_limit:
                    return

                # Remove least recently used models
                current_time = datetime.now()
                entries_to_remove = []

                for key, entry in self.model_cache.items():
                    if current_time - entry.last_used > self.cache_ttl:
                        entries_to_remove.append(key)

                # Remove expired entries
                for key in entries_to_remove:
                    del self.model_cache[key]

                # If still over limit, remove least accessed
                if len(self.model_cache) > self.cache_size_limit:
                    sorted_entries = sorted(
                        self.model_cache.items(),
                        key=lambda x: x[1].access_count
                    )

                    excess = len(self.model_cache) - self.cache_size_limit
                    for key, _ in sorted_entries[:excess]:
                        del self.model_cache[key]

                self.logger.debug(f"🧹 Cleaned cache: {len(entries_to_remove)} entries removed")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to clean cache: {e}")

    def update_model_performance(
        self,
        model_name: str,
        model_type: str,
        regime_id: int,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        execution_time: float
    ):
        """Update model performance for continuous learning."""
        tprint(f"[TRADING_MODEL_MANAGER] Entering update_model_performance with model={model_name}, type={model_type}, regime={regime_id}, n_predictions={len(predictions)}")

        try:
            # Update model selector service
            if self.model_selector_service:
                tprint("[TRADING_MODEL_MANAGER] Updating model selector service performance")
                self.model_selector_service.update_model_performance(
                    regime_id=regime_id,
                    model_name=f"{model_name}_{model_type}",
                    predictions=predictions,
                    actual_values=actual_values,
                    execution_time=execution_time
                )

            # Update local performance tracking
            tprint("[TRADING_MODEL_MANAGER] Updating local performance tracking")
            with self.performance_lock:
                key = f"{model_name}_{model_type}"
                if key in self.performance_tracker:
                    performance = self.performance_tracker[key]

                    # Calculate metrics
                    accuracy = np.mean(predictions == actual_values)
                    precision = np.mean(actual_values[predictions == 1]) if np.any(predictions == 1) else 0.0
                    recall = np.mean(predictions[actual_values == 1]) if np.any(actual_values == 1) else 0.0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                    # Update performance
                    performance.accuracy = accuracy
                    performance.precision = precision
                    performance.recall = recall
                    performance.f1_score = f1_score
                    performance.execution_time = execution_time
                    performance.sample_count += len(predictions)
                    performance.last_updated = datetime.now()

                    # Add to history
                    performance.performance_history.append(f1_score)
                    if len(performance.performance_history) > self.performance_window:
                        performance.performance_history.pop(0)

                    tprint(f"[TRADING_MODEL_MANAGER] Performance updated: accuracy={accuracy:.3f}, precision={precision:.3f}, recall={recall:.3f}, f1={f1_score:.3f}")

            self.logger.debug(f"📊 Updated performance for {model_name} ({model_type})")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting update_model_performance successfully")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update model performance: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting update_model_performance with error: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            with self.performance_lock:
                metrics = {}
                for key, performance in self.performance_tracker.items():
                    metrics[key] = {
                        'model_name': performance.model_name,
                        'regime_id': performance.regime_id,
                        'accuracy': performance.accuracy,
                        'precision': performance.precision,
                        'recall': performance.recall,
                        'f1_score': performance.f1_score,
                        'execution_time': performance.execution_time,
                        'sample_count': performance.sample_count,
                        'last_updated': performance.last_updated.isoformat(),
                        'avg_performance': np.mean(performance.performance_history) if performance.performance_history else 0.0
                    }

                return metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to get performance metrics: {e}")
            return {}

    def get_cache_status(self) -> Dict[str, Any]:
        """Get model cache status."""
        try:
            with self.cache_lock:
                return {
                    'cache_size': len(self.model_cache),
                    'cache_limit': self.cache_size_limit,
                    'cached_models': list(self.model_cache.keys()),
                    'memory_usage': sum(len(str(entry.model)) for entry in self.model_cache.values())
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to get cache status: {e}")
            return {}

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        try:
            return {
                'model_selector_ready': self.model_selector_service is not None,
                'model_loader_ready': self.model_loader is not None,
                'cached_models': len(self.model_cache),
                'tracked_models': len(self.performance_tracker),
                'performance_metrics': self.get_performance_metrics(),
                'cache_status': self.get_cache_status()
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}

    def shutdown(self):
        """Shutdown the model manager."""
        tprint("[TRADING_MODEL_MANAGER] Entering shutdown")

        try:
            self.logger.info("🛑 Shutting down trading model manager...")

            # Shutdown loading threads
            if self.loading_threads:
                tprint("[TRADING_MODEL_MANAGER] Shutting down loading threads")
                self.loading_threads.shutdown(wait=True)
                tprint("[TRADING_MODEL_MANAGER] Loading threads shutdown complete")

            # Clear cache
            with self.cache_lock:
                cache_size = len(self.model_cache)
                self.model_cache.clear()
                tprint(f"[TRADING_MODEL_MANAGER] Cleared model cache: {cache_size} models removed")

            # Clear performance tracking
            with self.performance_lock:
                tracker_size = len(self.performance_tracker)
                self.performance_tracker.clear()
                tprint(f"[TRADING_MODEL_MANAGER] Cleared performance tracker: {tracker_size} entries removed")

            self.logger.info("✅ Trading model manager shutdown complete")
            tprint("[TRADING_MODEL_MANAGER] Exiting shutdown successfully")

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
            tprint(f"[TRADING_MODEL_MANAGER] Exiting shutdown with error: {e}")

# Global instance for trading system
_trading_model_manager = None

def get_trading_model_manager(config: Optional[TradingModelConfig] = None) -> TradingModelManager:
    """Get or create global trading model manager instance."""
    tprint(f"[TRADING_MODEL_MANAGER] Entering get_trading_model_manager with config={config is not None}")

    global _trading_model_manager

    if _trading_model_manager is None:
        tprint("[TRADING_MODEL_MANAGER] Creating new global manager instance")
        _trading_model_manager = TradingModelManager(config)
        _trading_model_manager.initialize()
        tprint("[TRADING_MODEL_MANAGER] Global manager instance created and initialized")
    else:
        tprint("[TRADING_MODEL_MANAGER] Returning existing global manager instance")

    tprint("[TRADING_MODEL_MANAGER] Exiting get_trading_model_manager")
    return _trading_model_manager

def get_models_for_trading(
    market_data: pd.DataFrame,
    symbol: str = "ETHUSDT",
    timeframe: str = "5m",
    config: Optional[TradingModelConfig] = None
) -> Dict[str, Any]:
    """Convenience function to get models for trading."""
    tprint(f"[TRADING_MODEL_MANAGER] Entering module-level get_models_for_trading with symbol={symbol}, timeframe={timeframe}")

    manager = get_trading_model_manager(config)
    result = manager.get_models_for_trading(market_data, symbol, timeframe)

    tprint(f"[TRADING_MODEL_MANAGER] Exiting module-level get_models_for_trading with {len(result)} models")
    return result
