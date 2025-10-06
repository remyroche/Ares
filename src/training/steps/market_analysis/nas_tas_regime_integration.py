"""
NAS/TAS Regime Integration Component

This component integrates NAS and TAS regime detection with the training pipeline,
ensuring that per-regime ML model training uses NAS/TAS regime assignments
instead of clusters from regime_data_splitting.

Key Features:
- Unified regime detection using NAS/TAS systems
- Integration with training pipelines for per-regime training
- Proper model selection architecture for 2-3 best models per regime
- Signal emission integration with ML outputs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_timer
)

# Import NAS/TAS regime detectors
from .hybrid_nas_tas_regime.core.hybrid_regime_detector import HybridNASTASRegimeDetector, HybridRegimeConfig
from .nas_regime.core.enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector, EnhancedPerfectNASConfig
from .tas_regime.core.tas_regime_detector import TASRegimeDetector, TASRegimeConfig

logger = logging.getLogger(__name__)


class RegimeIntegrationMode(Enum):
    """Modes for regime integration."""
    HYBRID_NAS_TAS = "hybrid_nas_tas"
    NAS_ONLY = "nas_only"
    TAS_ONLY = "tas_only"
    ADAPTIVE = "adaptive"  # Automatically choose best method


@dataclass
class RegimeIntegrationConfig:
    """Configuration for NAS/TAS regime integration."""
    mode: RegimeIntegrationMode = RegimeIntegrationMode.HYBRID_NAS_TAS
    n_regimes: int = 8

    # NAS configuration
    nas_timeframe: str = "15m"
    nas_config: Optional[Dict[str, Any]] = None

    # TAS configuration
    tas_config: Optional[Dict[str, Any]] = None

    # Hybrid configuration
    hybrid_config: Optional[Dict[str, Any]] = None

    # Integration settings
    confidence_threshold: float = 0.7
    min_regime_samples: int = 100
    enable_regime_validation: bool = True
    enable_performance_tracking: bool = True

    # Model selection settings
    max_models_per_regime: int = 3
    selection_strategy: str = "performance_weighted"  # "performance_weighted", "confidence_based", "ensemble"


@dataclass
class RegimeIntegrationResult:
    """Result from regime integration."""
    success: bool
    regime_assignments: np.ndarray
    regime_probabilities: np.ndarray
    regime_metadata: Dict[str, Any]
    model_selection_data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class NASTASRegimeIntegrator:
    """
    Integrates NAS and TAS regime detection with training pipelines.

    This component ensures that:
    1. Regime detection uses NAS/TAS instead of regime_data_splitting clusters
    2. Per-regime training uses proper NAS/TAS regime assignments
    3. Model selection architecture selects best 2-3 models per regime
    4. Signals are emitted based on ML outputs from selected models
    """

    def __init__(self, config: RegimeIntegrationConfig):
        """Initialize the NAS/TAS regime integrator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize regime detectors
        self.regime_detectors = {}
        self._initialize_regime_detectors()

        # Track regime performance and model selection
        self.regime_performance_history = {}
        self.model_selection_cache = {}

        tprint_success("✅ NASTASRegimeIntegrator initialized")

    def _initialize_regime_detectors(self):
        """Initialize regime detection systems."""
        try:
            # Initialize NAS detector
            if self.config.mode in [RegimeIntegrationMode.NAS_ONLY, RegimeIntegrationMode.HYBRID_NAS_TAS, RegimeIntegrationMode.ADAPTIVE]:
                nas_config = self.config.nas_config or {}
                nas_config.update({
                    'primary_timeframe': self.config.nas_timeframe,
                    'n_regimes': self.config.n_regimes,
                    'enable_economic_evaluation': True,
                    'enable_financial_relevance': True
                })

                nas_detector_config = EnhancedPerfectNASConfig(**nas_config)
                self.regime_detectors['nas'] = EnhancedPerfectNASRegimeDetector(nas_detector_config)
                tprint_info("✅ NAS regime detector initialized")

            # Initialize TAS detector
            if self.config.mode in [RegimeIntegrationMode.TAS_ONLY, RegimeIntegrationMode.HYBRID_NAS_TAS, RegimeIntegrationMode.ADAPTIVE]:
                tas_config = self.config.tas_config or {}
                tas_config.update({
                    'n_regimes': self.config.n_regimes,
                    'enable_economic_evaluation': True,
                    'enable_uncertainty_quantification': True
                })

                tas_detector_config = TASRegimeConfig(**tas_config)
                self.regime_detectors['tas'] = TASRegimeDetector(tas_detector_config)
                tprint_info("✅ TAS regime detector initialized")

            # Initialize hybrid detector
            if self.config.mode == RegimeIntegrationMode.HYBRID_NAS_TAS:
                hybrid_config = self.config.hybrid_config or {}
                hybrid_config.update({
                    'n_regimes': self.config.n_regimes,
                    'combination_strategy': 'weighted',
                    'tas_weight': 0.4,
                    'nas_weight': 0.6,
                    'enable_economic_evaluation': True,
                    'enable_financial_relevance': True
                })

                hybrid_detector_config = HybridRegimeConfig(**hybrid_config)
                self.regime_detectors['hybrid'] = HybridNASTASRegimeDetector(hybrid_detector_config)
                tprint_info("✅ Hybrid NAS-TAS regime detector initialized")

            tprint_success(f"✅ Initialized {len(self.regime_detectors)} regime detectors")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize regime detectors: {e}")
            raise

    def detect_regimes_for_training(
        self,
        market_data: pd.DataFrame,
        timestamps: Optional[np.ndarray] = None
    ) -> RegimeIntegrationResult:
        """
        Detect regimes using NAS/TAS for training pipeline integration.

        Args:
            market_data: Market data for regime detection
            timestamps: Optional timestamps for temporal analysis

        Returns:
            RegimeIntegrationResult with regime assignments and metadata
        """
        start_time = time.time()
        tprint_info("🚀 Starting NAS/TAS regime detection for training integration")

        try:
            # Select regime detection method
            detector_method = self._select_regime_detection_method(market_data)

            # Perform regime detection
            if detector_method == 'hybrid':
                regime_result = self.regime_detectors['hybrid'].detect_regimes(
                    market_data, timestamps, validate_economic_significance=True, validate_financial_relevance=True
                )
            elif detector_method == 'nas':
                regime_result = self.regime_detectors['nas'].detect_regimes(
                    market_data, timestamps
                )
            elif detector_method == 'tas':
                regime_result = self.regime_detectors['tas'].detect_regimes(
                    market_data, timestamps
                )
            else:
                raise ValueError(f"Unknown regime detection method: {detector_method}")

            if not regime_result.success:
                raise RuntimeError(f"Regime detection failed: {regime_result.error_message}")

            # Validate regime assignments
            regime_assignments = regime_result.regime_predictions
            regime_probabilities = regime_result.regime_probabilities

            # Ensure we have valid regime assignments
            if len(np.unique(regime_assignments)) < 2:
                tprint_warning("⚠️ Only one regime detected, using fallback clustering")
                regime_assignments, regime_probabilities = self._create_fallback_regimes(market_data)

            # Create regime metadata for training
            regime_metadata = self._create_regime_metadata(regime_result, detector_method)

            # Create model selection data
            model_selection_data = self._create_model_selection_data(
                regime_assignments, regime_probabilities, regime_metadata
            )

            execution_time = time.time() - start_time

            result = RegimeIntegrationResult(
                success=True,
                regime_assignments=regime_assignments,
                regime_probabilities=regime_probabilities,
                regime_metadata=regime_metadata,
                model_selection_data=model_selection_data,
                execution_time=execution_time
            )

            tprint_success(f"✅ Regime detection completed in {execution_time:.2f}s")
            tprint_info(f"📊 Detected {len(np.unique(regime_assignments))} regimes using {detector_method}")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Regime detection failed: {e}")

            return RegimeIntegrationResult(
                success=False,
                regime_assignments=np.array([]),
                regime_probabilities=np.array([]),
                regime_metadata={},
                model_selection_data={},
                execution_time=execution_time,
                error_message=str(e)
            )

    def _select_regime_detection_method(self, market_data: pd.DataFrame) -> str:
        """Select the best regime detection method based on data characteristics."""
        if self.config.mode == RegimeIntegrationMode.HYBRID_NAS_TAS:
            return 'hybrid'
        elif self.config.mode == RegimeIntegrationMode.NAS_ONLY:
            return 'nas'
        elif self.config.mode == RegimeIntegrationMode.TAS_ONLY:
            return 'tas'
        elif self.config.mode == RegimeIntegrationMode.ADAPTIVE:
            # Analyze data characteristics to choose best method
            return self._adaptive_method_selection(market_data)
        else:
            return 'hybrid'  # Default fallback

    def _adaptive_method_selection(self, market_data: pd.DataFrame) -> str:
        """Adaptively select regime detection method based on data analysis."""
        try:
            # Analyze data characteristics
            n_samples = len(market_data)
            n_features = len(market_data.columns)

            # Check for sufficient data
            if n_samples < 1000:
                return 'tas'  # TAS works better with smaller datasets

            # Check data complexity
            volatility = market_data['close'].pct_change().std()
            trend_strength = abs(market_data['close'].pct_change().rolling(20).mean().iloc[-1])

            # NAS is better for complex, volatile markets
            if volatility > 0.02 or trend_strength > 0.01:
                return 'nas'
            else:
                return 'tas'

        except Exception:
            return 'hybrid'  # Safe fallback

    def _create_fallback_regimes(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create fallback regime assignments when detection fails."""
        try:
            n_samples = len(market_data)

            # Simple volatility-based clustering as fallback
            returns = market_data['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0.01)

            # Create 3 regimes based on volatility
            regime_assignments = np.zeros(n_samples, dtype=int)

            # Low volatility regime (bottom 30%)
            low_vol_threshold = volatility.quantile(0.3)
            regime_assignments[volatility <= low_vol_threshold] = 0

            # High volatility regime (top 30%)
            high_vol_threshold = volatility.quantile(0.7)
            regime_assignments[volatility >= high_vol_threshold] = 2

            # Medium volatility regime (middle 40%)
            regime_assignments[(volatility > low_vol_threshold) & (volatility < high_vol_threshold)] = 1

            # Create uniform probabilities for fallback
            regime_probabilities = np.full((n_samples, 3), 1/3)

            tprint_warning("⚠️ Using fallback regime detection based on volatility")
            return regime_assignments, regime_probabilities

        except Exception as e:
            tprint_error(f"❌ Fallback regime creation failed: {e}")
            # Return random regimes as last resort
            n_samples = len(market_data)
            regime_assignments = np.random.randint(0, 3, n_samples)
            regime_probabilities = np.full((n_samples, 3), 1/3)
            return regime_assignments, regime_probabilities

    def _create_regime_metadata(self, regime_result: Any, detector_method: str) -> Dict[str, Any]:
        """Create comprehensive regime metadata for training."""
        metadata = {
            'detector_method': detector_method,
            'detection_timestamp': datetime.now().isoformat(),
            'n_regimes': len(np.unique(regime_result.regime_predictions)),
            'regime_distribution': np.bincount(regime_result.regime_predictions),
            'confidence_scores': getattr(regime_result, 'confidence_scores', None),
            'economic_significance': getattr(regime_result, 'economic_significance_scores', None),
            'financial_relevance': getattr(regime_result, 'financial_relevance_scores', None),
            'regime_stability': getattr(regime_result, 'regime_stability_scores', None),
            'transition_probabilities': getattr(regime_result, 'transition_probabilities', None),
        }

        # Add regime-specific statistics
        if hasattr(regime_result, 'regime_stats'):
            metadata['regime_statistics'] = regime_result.regime_stats

        return metadata

    def _create_model_selection_data(
        self,
        regime_assignments: np.ndarray,
        regime_probabilities: np.ndarray,
        regime_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create data for model selection architecture."""
        model_selection_data = {
            'regime_assignments': regime_assignments,
            'regime_probabilities': regime_probabilities,
            'regime_metadata': regime_metadata,
            'selection_strategy': self.config.selection_strategy,
            'max_models_per_regime': self.config.max_models_per_regime,
            'performance_history': self.regime_performance_history,
            'model_selection_cache': self.model_selection_cache,
        }

        return model_selection_data

    def get_regime_training_data(
        self,
        market_data: pd.DataFrame,
        regime_result: RegimeIntegrationResult
    ) -> Dict[int, pd.DataFrame]:
        """
        Split market data into regime-specific datasets for training.

        Args:
            market_data: Full market dataset
            regime_result: Regime integration result

        Returns:
            Dict mapping regime_id to regime-specific dataset
        """
        if not regime_result.success:
            raise ValueError("Cannot create regime training data from failed regime detection")

        regime_datasets = {}

        for regime_id in np.unique(regime_result.regime_assignments):
            regime_mask = regime_result.regime_assignments == regime_id
            regime_data = market_data[regime_mask].copy()

            # Add regime information to dataset
            regime_data['regime_id'] = regime_id
            regime_data['regime_probability'] = regime_result.regime_probabilities[regime_mask, regime_id]

            # Validate minimum samples
            if len(regime_data) >= self.config.min_regime_samples:
                regime_datasets[regime_id] = regime_data
                tprint_info(f"📊 Regime {regime_id}: {len(regime_data)} samples")
            else:
                tprint_warning(f"⚠️ Regime {regime_id} has insufficient samples ({len(regime_data)} < {self.config.min_regime_samples})")

        if not regime_datasets:
            raise ValueError("No regimes have sufficient samples for training")

        return regime_datasets

    def select_best_models_for_regime(
        self,
        regime_id: int,
        available_models: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> List[str]:
        """
        Select the best 2-3 models for a specific regime based on performance.

        Args:
            regime_id: Regime ID for model selection
            available_models: Dictionary of available models
            performance_metrics: Performance metrics for each model

        Returns:
            List of selected model IDs (best 2-3 models)
        """
        try:
            # Filter models by performance threshold
            valid_models = {}
            for model_id, model in available_models.items():
                if performance_metrics.get(model_id, 0) >= 0.5:  # Minimum performance threshold
                    valid_models[model_id] = performance_metrics[model_id]

            if not valid_models:
                tprint_warning(f"⚠️ No valid models found for regime {regime_id}")
                return []

            # Sort models by performance
            sorted_models = sorted(valid_models.items(), key=lambda x: x[1], reverse=True)

            # Select top models (up to max_models_per_regime)
            n_select = min(self.config.max_models_per_regime, len(sorted_models))
            selected_models = [model_id for model_id, _ in sorted_models[:n_select]]

            tprint_info(f"✅ Selected {len(selected_models)} models for regime {regime_id}: {selected_models}")
            return selected_models

        except Exception as e:
            tprint_error(f"❌ Model selection failed for regime {regime_id}: {e}")
            return []

    def update_model_selection_cache(
        self,
        regime_id: int,
        selected_models: List[str],
        performance_metrics: Dict[str, float]
    ):
        """Update model selection cache with latest results."""
        cache_key = f"regime_{regime_id}"

        self.model_selection_cache[cache_key] = {
            'selected_models': selected_models,
            'performance_metrics': performance_metrics,
            'selection_timestamp': datetime.now().isoformat(),
            'selection_strategy': self.config.selection_strategy,
        }

        tprint_debug(f"💾 Updated model selection cache for regime {regime_id}")


def create_nas_tas_regime_integrator(config: Optional[RegimeIntegrationConfig] = None) -> NASTASRegimeIntegrator:
    """Create NAS/TAS regime integrator with default configuration."""
    if config is None:
        config = RegimeIntegrationConfig()

    return NASTASRegimeIntegrator(config)