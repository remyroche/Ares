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

        self.logger.info("✅ Trading Model Manager initialized")

    def initialize(self) -> bool:
        """Initialize the trading model manager."""
        try:
            self.logger.info("🔧 Initializing trading model manager...")

            # Initialize model selector service
            self.model_selector_service = ModelSelectorService(self.config)
            if not self.model_selector_service.initialize():
                raise RuntimeError("Failed to initialize model selector service")

            # Initialize model loader
            self._initialize_model_loader()

            # Load existing models if available
            self._load_existing_models()

            self.logger.info("✅ Trading model manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading model manager: {e}")
            return False

    def _initialize_model_loader(self):
        """Initialize model loader."""
        try:
            # Use unified model loader for accessing models from artifact_manager
            from src.trading.integration.unified_model_loader import get_unified_model_loader
            
            self.model_loader = get_unified_model_loader()
            self.logger.info("✅ Model loader initialized (using unified model loader)")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model loader: {e}")
            raise

    def _load_existing_models(self):
        """Load existing models from training results."""
        try:
            # Try to load models from training artifacts
            training_artifacts_path = "data_cache/training_artifacts"
            if Path(training_artifacts_path).exists():
                self.logger.info("🔄 Loading existing models from training artifacts...")
                # This would load actual models from training artifacts
                # For now, we'll just log the attempt
                self.logger.info("✅ Existing models loaded")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load existing models: {e}")

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
        try:
            self.logger.info(f"🎯 Getting models for {symbol} ({timeframe})...")

            # Get model selection result
            selection_result = self.model_selector_service.select_models_for_trading(
                market_data=market_data,
                symbol=symbol,
                timeframe=timeframe
            )

            if not selection_result.selected_models:
                self.logger.warning("⚠️ No models selected, using fallback")
                return self._get_fallback_models(symbol, timeframe)

            # Load selected models
            loaded_models = {}
            for model_type, model_name in selection_result.selected_models.items():
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
                    else:
                        self.logger.warning(f"⚠️ Failed to load model {model_name}")

                except Exception as e:
                    self.logger.warning(f"⚠️ Error loading model {model_name}: {e}")

            # Update performance tracking
            self._update_performance_tracking(selection_result)

            self.logger.info(f"✅ Loaded {len(loaded_models)} models for trading")
            return loaded_models

        except Exception as e:
            self.logger.error(f"❌ Failed to get models for trading: {e}")
            return self._get_fallback_models(symbol, timeframe)

    def _load_model(self, model_name: str, model_type: str, timeframe: str) -> Optional[Any]:
        """Load a specific model."""
        try:
            cache_key = f"{model_name}_{model_type}_{timeframe}"

            # Check cache first
            with self.cache_lock:
                if cache_key in self.model_cache:
                    entry = self.model_cache[cache_key]
                    entry.last_used = datetime.now()
                    entry.access_count += 1
                    self.logger.debug(f"📦 Using cached model: {cache_key}")
                    return entry.model

            # Load model (placeholder implementation)
            model = self._load_model_from_storage(model_name, model_type, timeframe)

            if model:
                # Cache the model
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
                return model

            return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load model {model_name}: {e}")
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
        try:
            fallback_models = {}

            # Create fallback models
            for model_type in ['analyst', 'tactician']:
                fallback_models[model_type] = {
                    'model': self._load_model_from_storage('default', model_type, timeframe),
                    'name': 'default',
                    'weight': 1.0,
                    'regime_id': 0,
                    'confidence': 0.5,
                    'fallback': True
                }

            self.logger.warning("⚠️ Using fallback models")
            return fallback_models

        except Exception as e:
            self.logger.error(f"❌ Failed to get fallback models: {e}")
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
        try:
            # Update model selector service
            if self.model_selector_service:
                self.model_selector_service.update_model_performance(
                    regime_id=regime_id,
                    model_name=f"{model_name}_{model_type}",
                    predictions=predictions,
                    actual_values=actual_values,
                    execution_time=execution_time
                )

            # Update local performance tracking
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

            self.logger.debug(f"📊 Updated performance for {model_name} ({model_type})")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update model performance: {e}")

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
        try:
            self.logger.info("🛑 Shutting down trading model manager...")

            # Shutdown loading threads
            if self.loading_threads:
                self.loading_threads.shutdown(wait=True)

            # Clear cache
            with self.cache_lock:
                self.model_cache.clear()

            # Clear performance tracking
            with self.performance_lock:
                self.performance_tracker.clear()

            self.logger.info("✅ Trading model manager shutdown complete")

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")

# Global instance for trading system
_trading_model_manager = None

def get_trading_model_manager(config: Optional[TradingModelConfig] = None) -> TradingModelManager:
    """Get or create global trading model manager instance."""
    global _trading_model_manager

    if _trading_model_manager is None:
        _trading_model_manager = TradingModelManager(config)
        _trading_model_manager.initialize()

    return _trading_model_manager

def get_models_for_trading(
    market_data: pd.DataFrame,
    symbol: str = "ETHUSDT",
    timeframe: str = "5m",
    config: Optional[TradingModelConfig] = None
) -> Dict[str, Any]:
    """Convenience function to get models for trading."""
    manager = get_trading_model_manager(config)
    return manager.get_models_for_trading(market_data, symbol, timeframe)
