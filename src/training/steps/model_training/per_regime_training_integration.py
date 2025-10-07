"""
Per-Regime Training Integration for Analyst and Tactician

This module integrates per-regime ML model training into the existing training pipeline
for both Analyst (15m) and Tactician (5m) models. It ensures that per-regime training
is called alongside the base model training and that the outputs are used by the
ensemble models during training and trading.

Key Features:
- Integrates with existing Analyst and Tactician training pipelines
- Supports pluggable regime detection (external assignments or internal detectors)
- Trains per-regime models for both 5m and 15m timeframes
- Provides model selection for trading system
- Maintains compatibility with existing training structure
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import existing training components
from src.utils.ml_common.training.per_regime_training_step import PerRegimeTrainingStep
from src.utils.ml_common.config.base_training_config import PerRegimeTrainingConfig

# Import NAS/TAS regime detection if available for backward compatibility
try:
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.hybrid_regime_detector import (
        HybridNASTASRegimeDetector, HybridRegimeResult
    )
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.config.hybrid_regime_config import (
        HybridRegimeConfig, RegimeCombinationStrategy
    )
    HYBRID_REGIME_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    HybridNASTASRegimeDetector = None  # type: ignore[assignment]
    HybridRegimeResult = Any  # type: ignore[assignment]
    HybridRegimeConfig = None  # type: ignore[assignment]
    RegimeCombinationStrategy = None  # type: ignore[assignment]
    HYBRID_REGIME_AVAILABLE = False

# Import model selection
from src.training.steps.market_analysis.hybrid_nas_tas_regime.regime_model_mapping.data_driven_model_selector import (
    DataDrivenModelSelector, ModelSelectorConfig, RegimeModelMapping
)

logger = logging.getLogger(__name__)


@dataclass
class PerRegimeTrainingResult:
    """Result from per-regime training integration."""

    success: bool
    regime_models: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    regime_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    model_selector: Optional[DataDrivenModelSelector] = None
    regime_detection_result: Optional[HybridRegimeResult] = None
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    regime_assignment_source: str = "external"


class PerRegimeTrainingIntegration:
    """
    Per-regime training integration for Analyst and Tactician models.
    
    This class integrates per-regime ML model training into the existing training
    pipeline, ensuring that regime-specific models are trained alongside base models
    and that the model selector is available for trading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize per-regime training integration."""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.regime_detector = None
        self.model_selector = None
        self.training_steps = {}
        
        # Results storage
        self.training_results = {}
        
        self.logger.info("✅ Per-Regime Training Integration initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all required components."""
        try:
            self.logger.info("🔧 Initializing per-regime training components...")
            
            # Initialize regime detector
            self._initialize_regime_detector()
            
            # Initialize model selector
            self._initialize_model_selector()
            
            # Initialize training steps for each timeframe
            self._initialize_training_steps()
            
            self.logger.info("✅ All per-regime training components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize per-regime training components: {e}")
            return False
    
    def _initialize_regime_detector(self):
        """Initialize the configured regime detector if requested."""
        detector_factory: Optional[Callable[[], Any]] = self.config.get('regime_detector_factory')

        if callable(detector_factory):
            try:
                self.regime_detector = detector_factory()
                self.logger.info("✅ Custom regime detector initialized via factory")
            except Exception as exc:  # pragma: no cover - factory provided by runtime config
                self.logger.error(f"❌ Failed to initialize custom regime detector: {exc}")
                raise
            return

        if self.config.get('use_hybrid_regime_detector', False):
            if not HYBRID_REGIME_AVAILABLE:
                raise RuntimeError("Hybrid NAS/TAS detector requested but not available in this environment")

            try:
                hybrid_config_data = self.config.get('hybrid_regime_config', {})
                if isinstance(hybrid_config_data, HybridRegimeConfig):
                    regime_config = hybrid_config_data
                else:
                    regime_config = HybridRegimeConfig(
                        n_regimes=hybrid_config_data.get('n_regimes', self.config.get('n_regimes', 8)),
                        combination_strategy=hybrid_config_data.get(
                            'combination_strategy',
                            RegimeCombinationStrategy.ADAPTIVE_FUSION,
                        ),
                    )

                self.regime_detector = HybridNASTASRegimeDetector(regime_config)
                self.logger.info("✅ Hybrid NAS/TAS regime detector initialized")
            except Exception as exc:
                self.logger.error(f"❌ Failed to initialize hybrid regime detector: {exc}")
                raise
            return

        self.regime_detector = None
        self.logger.info(
            "ℹ️ No internal regime detector configured; expecting external regime assignments"
        )
    
    def _initialize_model_selector(self):
        """Initialize data-driven model selector."""
        try:
            selector_config = ModelSelectorConfig(
                primary_metric=self.config.get('primary_metric', 'f1_score'),
                confidence_threshold=self.config.get('confidence_threshold', 0.7),
                enable_ensemble=self.config.get('enable_ensemble', True),
                max_ensemble_models=self.config.get('max_ensemble_models', 3),
                enable_continuous_learning=self.config.get('enable_continuous_learning', True),
                mapping_file_path=self.config.get('mapping_file_path', 'data_cache/regime_model_mappings.pkl')
            )
            
            self.model_selector = DataDrivenModelSelector(selector_config)
            self.logger.info("✅ Data-driven model selector initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model selector: {e}")
            raise
    
    def _initialize_training_steps(self):
        """Initialize training steps for each timeframe."""
        try:
            timeframes = self.config.get('timeframes', ['5m', '15m'])
            model_types = self.config.get('model_types', ['random_forest', 'xgboost', 'lightgbm'])
            
            for timeframe in timeframes:
                # Create timeframe-specific config
                timeframe_config = PerRegimeTrainingConfig(
                    model_name=f"per_regime_{timeframe}",
                    model_types=model_types,
                    enable_hpo=self.config.get('enable_hpo', True),
                    save_models=True,
                    enable_evaluation=True,
                    timeframe=timeframe
                )
                
                # Initialize training step
                training_step = PerRegimeTrainingStep(timeframe_config)
                self.training_steps[timeframe] = training_step
                
                self.logger.info(f"✅ Per-regime training step initialized for {timeframe}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize training steps: {e}")
            raise
    
    def train_analyst_per_regime_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        regime_assignments: Optional[pd.DataFrame] = None
    ) -> PerRegimeTrainingResult:
        """
        Train per-regime models for Analyst (15m timeframe).
        
        Args:
            training_data: Training data DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names
            regime_assignments: Optional regime assignments DataFrame
            
        Returns:
            PerRegimeTrainingResult: Training results
        """
        return self._train_per_regime_models(
            training_data=training_data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            regime_assignments=regime_assignments,
            timeframe='15m',
            model_type='analyst'
        )
    
    def train_tactician_per_regime_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        regime_assignments: Optional[pd.DataFrame] = None
    ) -> PerRegimeTrainingResult:
        """
        Train per-regime models for Tactician (5m timeframe).
        
        Args:
            training_data: Training data DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names
            regime_assignments: Optional regime assignments DataFrame
            
        Returns:
            PerRegimeTrainingResult: Training results
        """
        return self._train_per_regime_models(
            training_data=training_data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            regime_assignments=regime_assignments,
            timeframe='5m',
            model_type='tactician'
        )
    
    def _train_per_regime_models(
        self,
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        regime_assignments: Optional[pd.DataFrame],
        timeframe: str,
        model_type: str
    ) -> PerRegimeTrainingResult:
        """Train per-regime models for a specific timeframe and model type."""
        start_time = time.time()
        
        try:
            self.logger.info(f"🎯 Starting per-regime training for {model_type} ({timeframe})...")
            
            # Step 1: Resolve regime assignments
            if regime_assignments is None:
                if self.regime_detector is None:
                    raise RuntimeError(
                        "Regime assignments were not provided and no regime detector is configured."
                    )

                self.logger.info("🔍 Detecting regimes using configured detector...")
                regime_result = self.regime_detector.detect_regimes(  # type: ignore[union-attr]
                    market_data=training_data[['open', 'high', 'low', 'close', 'volume']],
                    validate_economic_significance=True,
                    validate_financial_relevance=True
                )

                if not getattr(regime_result, 'success', True):
                    raise RuntimeError(f"Regime detection failed: {getattr(regime_result, 'error_message', 'unknown error')}")

                regime_labels = getattr(regime_result, 'regime_predictions', None)
                if regime_labels is None:
                    raise RuntimeError("Configured regime detector did not return predictions")

                assignment_source = 'detector'
                self.logger.info(f"✅ Detected {len(np.unique(regime_labels))} regimes")
            else:
                # Use provided regime assignments
                regime_labels = regime_assignments['regime'].values
                regime_result = None
                assignment_source = 'external'
                self.logger.info(f"✅ Using provided regime assignments: {len(np.unique(regime_labels))} regimes")
            
            # Step 2: Prepare training data
            X = training_data[feature_columns].values
            y = training_data[target_columns].values
            
            # Step 3: Train per-regime models
            training_step = self.training_steps[timeframe]
            
            # Execute per-regime training
            training_result = training_step.execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_columns,
                symbol=f"ETHUSDT_{timeframe}",
                exchange="binance",
                timeframe=timeframe
            )
            
            # Step 4: Update model selector with performance data
            self._update_model_selector_with_results(training_result, timeframe, model_type)
            
            # Step 5: Create result
            execution_time = time.time() - start_time
            
            raw_metadata = training_result.get('metadata', {})
            if isinstance(raw_metadata, dict):
                regime_metadata = raw_metadata.copy()
            else:
                regime_metadata = {'raw_metadata': raw_metadata}
            regime_metadata['assignment_source'] = assignment_source

            result = PerRegimeTrainingResult(
                success=True,
                regime_models=training_result.get('models', {}),
                regime_metadata=regime_metadata,
                model_selector=self.model_selector,
                regime_detection_result=regime_result,
                training_metrics=training_result.get('evaluation_results', {}),
                execution_time=execution_time,
                regime_assignment_source=assignment_source
            )
            
            # Store results
            self.training_results[f"{model_type}_{timeframe}"] = result

            self.logger.info(
                f"📌 Regime assignments sourced from: {assignment_source}"
            )

            self.logger.info(f"✅ Per-regime training completed for {model_type} ({timeframe}) in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Per-regime training failed for {model_type} ({timeframe}): {e}")
            
            return PerRegimeTrainingResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _update_model_selector_with_results(
        self,
        training_result: Dict[str, Any],
        timeframe: str,
        model_type: str
    ):
        """Update model selector with training results."""
        try:
            if 'evaluation_results' in training_result:
                for regime_id, regime_eval in training_result['evaluation_results'].items():
                    for model_name, model_eval in regime_eval.items():
                        if 'metrics' in model_eval and 'predictions' in model_eval:
                            # Register performance with model selector
                            self.model_selector.register_model_performance(
                                regime_id=int(regime_id),
                                model_name=f"{model_name}_{model_type}_{timeframe}",
                                predictions=model_eval['predictions'],
                                actual_values=model_eval.get('actual_values', []),
                                execution_time=model_eval.get('execution_time', 0.0)
                            )
            
            self.logger.debug(f"Updated model selector for {model_type} ({timeframe})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update model selector for {model_type} ({timeframe}): {e}")
    
    def get_model_selector(self) -> Optional[DataDrivenModelSelector]:
        """Get the model selector for use in trading."""
        return self.model_selector
    
    def get_training_results(self, model_type: str, timeframe: str) -> Optional[PerRegimeTrainingResult]:
        """Get training results for a specific model type and timeframe."""
        key = f"{model_type}_{timeframe}"
        return self.training_results.get(key)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'regime_detector_ready': self.regime_detector is not None,
            'model_selector_ready': self.model_selector is not None,
            'training_steps_ready': len(self.training_steps) > 0,
            'timeframes_supported': list(self.training_steps.keys()),
            'training_results_available': list(self.training_results.keys())
        }


# Global instance for integration
_per_regime_integration = None


def get_per_regime_integration(config: Optional[Dict[str, Any]] = None) -> PerRegimeTrainingIntegration:
    """Get or create global per-regime training integration instance."""
    global _per_regime_integration
    
    if _per_regime_integration is None:
        _per_regime_integration = PerRegimeTrainingIntegration(config)
        _per_regime_integration.initialize_components()
    
    return _per_regime_integration


def train_analyst_per_regime_models(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> PerRegimeTrainingResult:
    """Convenience function to train Analyst per-regime models."""
    integration = get_per_regime_integration(config)
    return integration.train_analyst_per_regime_models(
        training_data, feature_columns, target_columns, regime_assignments
    )


def train_tactician_per_regime_models(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    regime_assignments: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> PerRegimeTrainingResult:
    """Convenience function to train Tactician per-regime models."""
    integration = get_per_regime_integration(config)
    return integration.train_tactician_per_regime_models(
        training_data, feature_columns, target_columns, regime_assignments
    )


def get_model_selector_for_trading() -> Optional[DataDrivenModelSelector]:
    """Get model selector for use in trading system."""
    integration = get_per_regime_integration()
    return integration.get_model_selector()