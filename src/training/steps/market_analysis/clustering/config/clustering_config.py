"""
Configuration classes for NAS-TAS clustering.

This module contains all configuration classes and validation logic for the clustering component.
"""

import copy
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import traceback
from pathlib import Path
from collections import defaultdict
import pickle
import re

from ...shared_utils import (
    # Configuration
    validate_regime_count,
    normalize_weights,
    validate_algorithm_type,
    create_default_config,
    ConfigValidator,
    BaseConfig,

    # Logging
    get_logger,
    log_execution,
    log_performance,
    LoggingContext,
)

from ...shared_utils.calibration_registry import (
    get_current_calibration,
    get_quality_thresholds as get_calibrated_thresholds,
    update_quality_calibration,
)

from src.utils.tprint import (
    tprint,
    tprint_debug,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_progress,
    tprint_performance,
    tprint_timer,
    tprint_structured,
)

@dataclass
class ClusteringContext:
    """Lightweight context for sharing intermediate clustering artifacts with proper memory management."""

    original_features: np.ndarray
    market_data: pd.DataFrame
    optimized_features: Optional[np.ndarray] = None
    optimized_assignments: Optional[np.ndarray] = None
    optimal_k: Optional[int] = None
    optimal_bic: Optional[float] = None
    k_metadata: Dict[str, Any] = field(default_factory=dict)
    tas_assignments: Optional[np.ndarray] = None
    nas_assignments: Optional[np.ndarray] = None
    optimization_metrics: Dict[str, Any] = field(default_factory=dict)
    raw_assignments: Optional[np.ndarray] = None
    smoothed_assignments: Optional[np.ndarray] = None
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    memory_optimizer: Optional[Any] = None
    original_feature_names: Optional[List[str]] = None
    pre_pca_feature_names: Optional[List[str]] = None
    optimized_feature_names: Optional[List[str]] = None
    dropped_feature_names: Optional[List[str]] = None
    feature_scores: Dict[str, float] = field(default_factory=dict)
    pca_loading_scores: Dict[str, float] = field(default_factory=dict)
    pre_pca_feature_count: Optional[int] = None
    duration: Optional[float] = None

    def __enter__(self):
        """Context manager entry for memory management."""
        if self.memory_optimizer:
            self.memory_optimizer.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        cleanup_errors = []

        try:
            # Stop monitoring first
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
        except Exception as e:
            cleanup_errors.append(f"Memory optimizer cleanup failed: {e}")

        # Clean up large arrays
        try:
            if hasattr(self, 'original_features'):
                del self.original_features
            if hasattr(self, 'optimized_features'):
                del self.optimized_features
            if hasattr(self, 'market_data'):
                del self.market_data
        except Exception as e:
            cleanup_errors.append(f"Array cleanup failed: {e}")

        # Force garbage collection
        try:
            import gc
            gc.collect()
        except Exception as e:
            cleanup_errors.append(f"Garbage collection failed: {e}")

        if cleanup_errors:
            tprint_error(f"Context cleanup warnings: {'; '.join(cleanup_errors)}")

        # Re-raise original exception if any
        if exc_type is not None:
            raise exc_type(exc_val).with_traceback(exc_tb)

@dataclass
class NASTASClusteringConfig(BaseConfig):
    """Configuration for NAS-TAS clustering component using shared utilities."""
    exchange: str = "binance"

    # Empirical regime search bounds
    regime_search_min: int = 2
    regime_search_max: int = 5

    # Clustering parameters
    n_regimes: int = 3  # Reduced to 3 for better balance with current data
    # algorithm_type removed - always use custom progressive regime optimization
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True

    # Balance control parameters - SOFTENED for better optimization
    max_regime_percentage: float = 0.60  # Maximum percentage for any single regime (softened from 50% to 60%)
    min_regime_percentage: float = 0.05  # Minimum percentage for any single regime (softened from 10% to 5%)
    # balance_weight removed - balance will be used as constraints, not objective weights

    # Regime-focused clustering weights (removed momentum_weight)
    economic_weight: float = 0.25
    volatility_regime_weight: float = 0.30
    volume_regime_weight: float = 0.25
    structural_trend_weight: float = 0.20

    # Regime-focused feature configuration
    feature_categories: List[str] = None
    use_regime_focused_features: bool = True
    exclude_trading_features: bool = True
    use_standardized_features: bool = True
    signal_like_patterns: List[str] = field(
        default_factory=lambda: [
            r"signal",
            r"entry",
            r"exit",
            r"crossover",
            r"trade",
        ]
    )
    feature_category_caps: Dict[str, int] = field(
        default_factory=lambda: {
            'volatility_regime': 30,
            'volume_regime': 25,
            'structural_trend': 25,
            'statistical_regime': 30,
            'regime_quality': 20,
        }
    )
    pca_components_factor: float = 1.5
    zscore_clip_threshold: float = 5.0

    # Regime-specific feature quality thresholds (calibrated dynamically)
    min_regime_persistence: Optional[float] = None
    max_feature_noise_ratio: Optional[float] = None
    min_temporal_stability: Optional[float] = None

    # Output configuration
    output_dir: str = "data_cache"
    save_intermediate_results: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.regime_search_min = int(max(5, min(20, self.regime_search_min)))
        self.regime_search_max = int(max(
            self.regime_search_min,
            min(20, self.regime_search_max),
        ))

        if not (self.regime_search_min <= self.n_regimes <= self.regime_search_max):
            self.n_regimes = max(
                self.regime_search_min,
                min(self.regime_search_max, int(self.n_regimes)),
            )

        super().__post_init__()
        if self.feature_categories is None:
            # Regime-focused feature categories only
            self.feature_categories = [
                'regime_volatility',
                'regime_volume',
                'regime_structural_trend',
                'regime_statistical'
            ]

        if not self.signal_like_patterns:
            self.signal_like_patterns = [
                r"signal",
                r"entry",
                r"exit",
                r"crossover",
                r"trade",
            ]

        if not self.feature_category_caps:
            self.feature_category_caps = {
                'volatility_regime': 30,
                'volume_regime': 25,
                'structural_trend': 25,
                'statistical_regime': 30,
                'regime_quality': 20,
            }

        # Ensure n_regimes is within learned bounds
        if not (self.regime_search_min <= self.n_regimes <= self.regime_search_max):
            self.n_regimes = max(
                self.regime_search_min,
                min(self.regime_search_max, self.n_regimes),
            )

        # Apply calibrated quality thresholds if not explicitly provided
        thresholds = get_calibrated_thresholds()
        if self.min_regime_persistence is None:
            self.min_regime_persistence = thresholds.get('min_regime_persistence', 0.7)
        if self.max_feature_noise_ratio is None:
            self.max_feature_noise_ratio = thresholds.get('max_feature_noise_ratio', 0.3)
        if self.min_temporal_stability is None:
            self.min_temporal_stability = thresholds.get('min_temporal_stability', 0.6)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'exchange': self.exchange,
            'regime_search_min': self.regime_search_min,
            'regime_search_max': self.regime_search_max,
            'n_regimes': self.n_regimes,
            'enable_economic_clustering': self.enable_economic_clustering,
            'enable_ensemble_clustering': self.enable_ensemble_clustering,
            'max_regime_percentage': self.max_regime_percentage,
            'min_regime_percentage': self.min_regime_percentage,
            # 'balance_weight' removed - balance used as constraints, not objective weights
            'economic_weight': self.economic_weight,
            'volatility_regime_weight': self.volatility_regime_weight,
            'volume_regime_weight': self.volume_regime_weight,
            'structural_trend_weight': self.structural_trend_weight,
            'feature_categories': self.feature_categories,
            'use_regime_focused_features': self.use_regime_focused_features,
            'exclude_trading_features': self.exclude_trading_features,
            'use_standardized_features': self.use_standardized_features,
            'signal_like_patterns': self.signal_like_patterns,
            'feature_category_caps': self.feature_category_caps,
            'pca_components_factor': self.pca_components_factor,
            'zscore_clip_threshold': self.zscore_clip_threshold,
            'min_regime_persistence': self.min_regime_persistence,
            'max_feature_noise_ratio': self.max_feature_noise_ratio,
            'min_temporal_stability': self.min_temporal_stability,
            'output_dir': self.output_dir,
            'save_intermediate_results': self.save_intermediate_results,
        }
