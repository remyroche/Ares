"""
Model Selection Service for Trading System

This module provides a service for selecting the best models for trading based on
current market conditions and regime detection. It integrates with the DataDrivenModelSelector
from the training system to provide real-time model selection during trading.

Key Features:
- Real-time model selection based on current regime
- Integration with NAS/TAS regime detection
- Support for both Analyst and Tactician models
- Ensemble model selection with dynamic weights
- Performance monitoring and adaptation
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
import warnings
warnings.filterwarnings('ignore')

# Import model selection components (disabled due to missing dependency)
# Note: Removed dependency on hybrid_nas_tas_regime module
# from src.training.steps.market_analysis.hybrid_nas_tas_regime.regime_model_mapping.data_driven_model_selector import (
#     DataDrivenModelSelector, ModelSelectorConfig, RegimeModelMapping
# )

# Import regime detection (disabled due to missing dependency)
# from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.hybrid_regime_detector import (
#     HybridNASTASRegimeDetector, HybridRegimeResult
# )
# from src.training.steps.market_analysis.hybrid_nas_tas_regime.config.hybrid_regime_config import (
#     HybridRegimeConfig, RegimeCombinationStrategy
# )

logger = logging.getLogger(__name__)

@dataclass
class ModelSelectionResult:
    """Result from model selection."""
    selected_models: Dict[str, str]  # model_type -> selected_model_name
    ensemble_weights: Dict[str, Dict[str, float]]  # model_type -> {model_name: weight}
    regime_id: int
    confidence_score: float
    selection_metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    confirmation_status: str = "single_timeframe"
    confirmation_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingModelConfig:
    """Configuration for trading model selection."""
    # Model types to select from
    analyst_models: List[str] = field(default_factory=lambda: ['random_forest', 'xgboost', 'lightgbm'])
    tactician_models: List[str] = field(default_factory=lambda: ['random_forest', 'xgboost', 'lightgbm'])

    # Regime detection
    n_regimes: int = 8
    regime_combination_strategy: str = "ADAPTIVE_FUSION"  # Disabled due to missing dependency

    # Model selection
    primary_metric: str = 'f1_score'
    confidence_threshold: float = 0.7
    enable_ensemble: bool = True
    max_ensemble_models: int = 3

    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_window: int = 1000
    adaptation_threshold: float = 0.05

class ModelSelectorService:
    """
    Model selection service for trading system.

    This service provides real-time model selection based on current market
    conditions and regime detection. It integrates with the DataDrivenModelSelector
    from the training system.
    """

    def __init__(self, config: Optional[TradingModelConfig] = None):
        """Initialize model selector service."""
        self.config = config or TradingModelConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.regime_detector = None
        self.model_selector = None

        # Performance tracking
        self.selection_history = []
        self.performance_metrics = {}

        # Model cache
        self.model_cache = {}

        self.logger.info("✅ Model Selector Service initialized")

    def initialize(self) -> bool:
        """Initialize the model selector service."""
        try:
            self.logger.info("🔧 Initializing model selector service...")

            # Initialize regime detector
            self._initialize_regime_detector()

            # Initialize model selector
            self._initialize_model_selector()

            # Load existing model mappings if available
            self._load_model_mappings()

            self.logger.info("✅ Model selector service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model selector service: {e}")
            return False

    def _initialize_regime_detector(self):
        """Initialize NAS/TAS regime detector."""
        try:
            # regime_config = HybridRegimeConfig(
            #     n_regimes=self.config.n_regimes,
            #     combination_strategy=self.config.regime_combination_strategy
            # )

            # self.regime_detector = HybridNASTASRegimeDetector(regime_config)
            self.regime_detector = None  # Disabled due to missing dependency
            self.logger.info("⚠️ NAS/TAS regime detector disabled (missing dependency)")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize regime detector: {e}")
            raise

    def _initialize_model_selector(self):
        """Initialize data-driven model selector (disabled due to missing dependency)."""
        try:
            # selector_config = ModelSelectorConfig(
            #     primary_metric=self.config.primary_metric,
            #     confidence_threshold=self.config.confidence_threshold,
            #     enable_ensemble=self.config.enable_ensemble,
            #     max_ensemble_models=self.config.max_ensemble_models,
            #     enable_continuous_learning=True,
            #     mapping_file_path="data_cache/trading_model_mappings.pkl"
            # )

            # self.model_selector = DataDrivenModelSelector(selector_config)
            self.model_selector = None  # Disabled due to missing dependency
            self.logger.info("⚠️ Data-driven model selector disabled (missing dependency)")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model selector: {e}")
            raise

    def _load_model_mappings(self):
        """Load existing model mappings from training."""
        try:
            # Try to load from training results
            training_mappings_path = "data_cache/regime_model_mappings.pkl"
            if Path(training_mappings_path).exists():
                with open(training_mappings_path, 'rb') as f:
                    data = pickle.load(f)
                    if 'regime_mappings' in data:
                        self.model_selector.regime_mappings = data['regime_mappings']
                    if 'model_performance' in data:
                        self.model_selector.model_performance = data['model_performance']

                self.logger.info("✅ Loaded existing model mappings from training")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load existing model mappings: {e}")

    def select_models_for_trading(
        self,
        market_data: pd.DataFrame,
        model_types: List[str] = None,
        symbol: str = "ETHUSDT",
        timeframe: str = "5m"
    ) -> ModelSelectionResult:
        """
        Select best models for trading based on current market conditions.

        Args:
            market_data: Current market data
            model_types: List of model types to select from
            symbol: Trading symbol
            timeframe: Timeframe (5m or 15m)

        Returns:
            ModelSelectionResult: Selected models and weights
        """
        start_time = time.time()

        try:
            self.logger.info(f"🎯 Selecting models for {symbol} ({timeframe})...")

            # Step 1: Detect current regime
            regime_result = self._detect_current_regime(market_data)
            if not regime_result.success:
                raise RuntimeError(f"Regime detection failed: {regime_result.error_message}")

            current_regime = regime_result.regime_predictions[-1]
            regime_confidence = regime_result.regime_probabilities[-1, current_regime]

            self.logger.info(f"📊 Current regime: {current_regime} (confidence: {regime_confidence:.3f})")

            # Step 2: Select models for each type
            selected_models = {}
            ensemble_weights = {}

            if model_types is None:
                model_types = (
                    self.config.analyst_models
                    if timeframe == '15m'
                    else self.config.tactician_models
                )

            requested_model_types = list(model_types)

            model_candidates_by_type: Dict[str, List[str]] = {}
            for model_type in requested_model_types:
                config_attr = f"{model_type}_models"

                if hasattr(self.config, config_attr):
                    configured_models = getattr(self.config, config_attr) or [model_type]
                    available_models = [
                        self._format_model_name(model_name, timeframe)
                        for model_name in configured_models
                    ]
                else:
                    available_models = self._resolve_available_models(model_type, timeframe)

                # Remove potential duplicates while preserving order
                seen = set()
                unique_models = []
                for model_name in available_models:
                    if model_name not in seen:
                        seen.add(model_name)
                        unique_models.append(model_name)

                model_candidates_by_type[model_type] = (
                    unique_models
                    if unique_models
                    else [self._format_model_name(model_type, timeframe)]
                )

            for model_type, available_models in model_candidates_by_type.items():
                try:
                    # Get available models for this type and timeframe
                    # Select best model for current regime
                    selected_model, weights = self.model_selector.select_model_for_regime(
                        current_regime, available_models
                    )

                    selected_models[model_type] = selected_model
                    ensemble_weights[model_type] = weights

                    self.logger.info(f"✅ Selected {model_type}: {selected_model}")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to select model for {model_type}: {e}")
                    # Fallback to first available model
                    selected_models[model_type] = f"{model_type}_{timeframe}"
                    ensemble_weights[model_type] = {f"{model_type}_{timeframe}": 1.0}

            # Step 3: Create result
            execution_time = time.time() - start_time

            result = ModelSelectionResult(
                selected_models=selected_models,
                ensemble_weights=ensemble_weights,
                regime_id=current_regime,
                confidence_score=regime_confidence,
                selection_metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'regime_detection_time': regime_result.execution_time,
                    'model_types': requested_model_types,
                    'timestamp': datetime.now().isoformat()
                },
                execution_time=execution_time,
                confirmation_status="single_timeframe"
            )

            # Step 4: Track performance
            if self.config.enable_performance_monitoring:
                self._track_selection_performance(result)

            self.logger.info(f"✅ Model selection completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Model selection failed: {e}")

            return ModelSelectionResult(
                selected_models={},
                ensemble_weights={},
                regime_id=0,
                confidence_score=0.0,
                execution_time=execution_time,
                error_message=str(e),
                confirmation_status="failed",
                confirmation_details={'exception': str(e)}
            )

    def _detect_current_regime(self, market_data: pd.DataFrame) -> Optional[dict]:  # Disabled due to missing dependency
        """Detect current market regime."""
        try:
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Use NAS/TAS regime detector (disabled due to missing dependency)
            if self.regime_detector is None:
                self.logger.warning("⚠️ Regime detector disabled, returning default regime")
                return {
                    'regime': 'unknown',
                    'confidence': 0.0,
                    'regime_type': 'disabled',
                    'detection_method': 'disabled'
                }
            
            # regime_result = self.regime_detector.detect_regimes(
            #     market_data=market_data[required_columns],
            #     validate_economic_significance=True,
            #     validate_financial_relevance=True
            # )

            # return regime_result
            return None  # Disabled

        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            raise

    def _track_selection_performance(self, result: ModelSelectionResult):
        """Track model selection performance."""
        try:
            # Add to selection history
            self.selection_history.append({
                'timestamp': datetime.now(),
                'regime_id': result.regime_id,
                'confidence_score': result.confidence_score,
                'selected_models': result.selected_models,
                'execution_time': result.execution_time
            })

            # Keep only recent history
            if len(self.selection_history) > self.config.performance_window:
                self.selection_history = self.selection_history[-self.config.performance_window:]

            # Update performance metrics
            self._update_performance_metrics()

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to track selection performance: {e}")

    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            if not self.selection_history:
                return

            # Calculate average confidence
            avg_confidence = np.mean([s['confidence_score'] for s in self.selection_history])

            # Calculate regime distribution
            regime_counts = {}
            for s in self.selection_history:
                regime_id = s['regime_id']
                regime_counts[regime_id] = regime_counts.get(regime_id, 0) + 1

            # Calculate average execution time
            avg_execution_time = np.mean([s['execution_time'] for s in self.selection_history])

            self.performance_metrics = {
                'avg_confidence': avg_confidence,
                'regime_distribution': regime_counts,
                'avg_execution_time': avg_execution_time,
                'total_selections': len(self.selection_history)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update performance metrics: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def get_regime_insights(self, regime_id: int) -> Dict[str, Any]:
        """Get insights about a specific regime."""
        try:
            if self.model_selector:
                return self.model_selector.get_regime_insights(regime_id)
            else:
                return {'error': 'Model selector not initialized'}

        except Exception as e:
            self.logger.error(f"❌ Failed to get regime insights: {e}")
            return {'error': str(e)}

    def update_model_performance(
        self,
        regime_id: int,
        model_name: str,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        execution_time: float
    ):
        """Update model performance for continuous learning."""
        try:
            if self.model_selector:
                self.model_selector.register_model_performance(
                    regime_id=regime_id,
                    model_name=model_name,
                    predictions=predictions,
                    actual_values=actual_values,
                    execution_time=execution_time
                )

                self.logger.debug(f"Updated performance for regime {regime_id}, model {model_name}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update model performance: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'regime_detector_ready': self.regime_detector is not None,
            'model_selector_ready': self.model_selector is not None,
            'performance_metrics': self.performance_metrics,
            'total_selections': len(self.selection_history),
            'config': {
                'n_regimes': self.config.n_regimes,
                'primary_metric': self.config.primary_metric,
                'confidence_threshold': self.config.confidence_threshold,
                'enable_ensemble': self.config.enable_ensemble,
                'max_ensemble_models': self.config.max_ensemble_models
            }
        }

    def _resolve_available_models(self, model_type: str, timeframe: str) -> List[str]:
        """Resolve available model names for a specific type and timeframe."""
        type_to_models = {
            'analyst': self.config.analyst_models,
            'tactician': self.config.tactician_models
        }

        base_models = type_to_models.get(model_type)

        if base_models is None:
            base_models = [model_type]
        elif not base_models:
            # Fall back to the provided type name when the configured list is empty
            base_models = [model_type]

        formatted_models = [self._format_model_name(model_name, timeframe) for model_name in base_models]

        # Remove potential duplicates while preserving order
        seen = set()
        unique_models = []
        for model_name in formatted_models:
            if model_name not in seen:
                seen.add(model_name)
                unique_models.append(model_name)

        return unique_models if unique_models else [self._format_model_name(model_type, timeframe)]

    @staticmethod
    def _format_model_name(model_name: str, timeframe: str) -> str:
        """Ensure model names are consistently formatted with timeframe suffix."""
        suffix = f"_{timeframe}"
        return model_name if model_name.endswith(suffix) else f"{model_name}{suffix}"

# Global instance for trading system
_model_selector_service = None

def get_model_selector_service(config: Optional[TradingModelConfig] = None) -> ModelSelectorService:
    """Get or create global model selector service instance."""
    global _model_selector_service

    if _model_selector_service is None:
        _model_selector_service = ModelSelectorService(config)
        _model_selector_service.initialize()

    return _model_selector_service

def select_models_for_trading(
    market_data: pd.DataFrame,
    model_types: List[str] = None,
    symbol: str = "ETHUSDT",
    timeframe: str = "5m",
    config: Optional[TradingModelConfig] = None
) -> ModelSelectionResult:
    """Convenience function to select models for trading."""
    service = get_model_selector_service(config)
    return service.select_models_for_trading(market_data, model_types, symbol, timeframe)
