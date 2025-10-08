#!/usr/bin/env python3
"""
Final Feature Selection Pipeline

This module implements a comprehensive multi-stage feature selection pipeline
that runs at the end of the market analysis pipeline, progressively reducing
features from 120 → 100 → 80 → 60 using RandomForest and SHAP analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import average_precision_score, balanced_accuracy_score
import joblib
from functools import lru_cache
import hashlib
from collections import defaultdict

# Try to import SHAP, fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Try to import LightGBM for improved feature selection
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Try to import sklearn feature selection for RFE
try:
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    SKLEARN_FEATURE_SELECTION_AVAILABLE = False

# Import matrix operations and hardware utilities
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        correlation_matrix_gpu,
        matrix_correlation_analysis,
        batch_correlation_analysis,
        optimize_dataframe,
        get_batch_matrix_processor
    )
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_adaptive_optimization_engine,
        get_advanced_memory_optimizer,
        WorkloadType
    )
    MATRIX_OPERATIONS_AVAILABLE = True
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import existing feature selection tools
try:
    from src.training.utils.feature_selection.selection_methods import (
        mrmr_selection, lasso_selection, correlation_filtering,
        recursive_feature_elimination, variance_filtering
    )
    from src.utils.feature_selection.feature_importance_analyzer import (
        FeatureImportanceAnalyzer, ImportanceMethod, FeatureImportanceConfig
    )
    from src.training.utils.feature_selection.quality_metrics import (
        calculate_feature_quality_metrics, FeatureQualityMetrics
    )
    EXISTING_FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    EXISTING_FEATURE_SELECTION_AVAILABLE = False

# Import system utilities
from src.utils.logger import get_logger
from src.utils.matrix_operations import get_unified_matrix_operations
from src.utils.tprint import tprint
from src.feature_selection import EntropyBalancerConfig, EntropyFilterResult, EntropyStabilityFilter

@dataclass
class FeatureSelectionConfig:
    """Configuration for multi-stage feature selection."""
    # Stage targets
    initial_features: int = 120
    stage_1_target: int = 100
    stage_2_target: int = 80
    stage_3_target: int = 60

    # Enhanced model-specific parameters for regime detection
    model_type: str = 'regime_detection'
    target_features: int = 80
    min_features: int = 60
    max_features: int = 100
    priority_categories: List[str] = field(default_factory=lambda: ['volatility', 'structural', 'volume_regime', 'statistical'])

    # NEW: Regime-specific feature selection
    regime_focus_weights: Dict[str, float] = field(default_factory=lambda: {
        'volatility': 0.35,      # Highest weight for volatility regimes
        'structural': 0.25,      # Trend and structural regimes
        'volume_regime': 0.20,   # Volume-based regimes
        'statistical': 0.20      # Statistical regime features
    })

    # NEW: Directional feature selection modes
    direction_mode: str = 'both'  # 'both', 'long_only', 'short_only'
    separate_directional_features: bool = True  # Create completely separate feature sets for long/short
    directional_feature_prefixes: Dict[str, str] = field(default_factory=lambda: {
        'long': 'long_',
        'short': 'short_'
    })
    
    # Enhanced feature selection methods using existing tools
    selection_methods: List[str] = field(default_factory=lambda: [
        'mrmr', 'lasso', 'correlation_filtering', 'rfe', 'variance_filtering', 'mutual_info'
    ])
    
    # NEW: Use existing feature selection framework
    use_existing_framework: bool = True
    existing_methods: List[str] = field(default_factory=lambda: [
        'mrmr_selection', 'lasso_selection', 'correlation_filtering', 
        'recursive_feature_elimination', 'variance_filtering'
    ])
    
    # NEW: Regime-aware feature selection
    enable_regime_aware_selection: bool = True
    regime_clustering_threshold: float = 0.7
    regime_separation_bonus: float = 0.1

    # RandomForest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    rf_random_state: int = 42

    # SHAP parameters
    shap_sample_size: int = 1000
    shap_max_features: int = 200

    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = 'neg_mean_squared_error'

    # Enhanced quality thresholds for regime detection
    min_feature_importance: float = 0.002  # Increased for better regime separation
    min_correlation_threshold: float = 0.90  # Reduced to allow more regime features
    min_variance_threshold: float = 0.005   # Reduced to include more regime indicators

    # Enhanced model-specific thresholds for regime detection
    model_correlation_threshold: float = 0.85  # More permissive for regime features
    model_importance_threshold: float = 0.003  # Balanced for regime detection
    
    # NEW: Regime-specific quality thresholds
    regime_importance_threshold: float = 0.005  # Higher threshold for regime features
    regime_correlation_threshold: float = 0.80  # More permissive for regime features
    regime_variance_threshold: float = 0.001    # Lower threshold for regime indicators
    
    # NEW: Advanced feature selection criteria
    enable_multi_criteria_selection: bool = True
    criteria_weights: Dict[str, float] = field(default_factory=lambda: {
        'importance': 0.30,
        'correlation': 0.20,
        'variance': 0.15,
        'regime_separation': 0.25,
        'temporal_stability': 0.10
    })

    # NEW: Entropy stability filtering
    enable_entropy_balancing: bool = True
    entropy_num_slices: int = 12
    entropy_min_slice_size: int = 100
    entropy_variance_threshold: float = 0.12
    entropy_max_bins: int = 15
    entropy_min_unique_values: int = 5
    entropy_use_time_index: bool = True

    # NEW: Early termination and smart pruning
    enable_early_termination: bool = True
    early_termination_threshold: float = 0.01  # Stop processing features below this importance threshold
    adaptive_importance_threshold: bool = True  # Dynamically adjust threshold based on feature distribution
    importance_percentile_cutoff: float = 20.0  # Bottom percentile to prune

    # NEW: LightGBM optimization parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 10,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    })

    # NEW: Recursive Feature Elimination parameters
    enable_rfe: bool = True
    rfe_step_size: float = 0.1  # Remove 10% of features per step
    rfe_min_features: int = 10  # Minimum features to keep in RFE
    rfe_cv_folds: int = 3
    rfe_early_stopping: bool = True
    rfe_early_stopping_patience: int = 3  # Stop if no improvement for 3 consecutive steps

    # NEW: Mutual information parameters
    enable_mutual_information: bool = True
    mutual_info_method: str = 'auto'  # 'auto', 'discrete', 'continuous'
    mutual_info_k: int = 10  # Number of nearest neighbors for continuous MI
    mutual_info_discrete_features: bool = False

    # NEW: Chunked processing parameters
    enable_chunked_processing: bool = True
    chunk_size: int = 1000  # Process features in chunks of this size
    max_chunks: int = 10  # Maximum number of chunks to process
    chunk_overlap: int = 50  # Overlap between chunks for consistency

    # Output settings
    save_models: bool = True
    save_analysis: bool = True
    output_directory: str = "outcomes/market_analysis"
    verbose: bool = True

@dataclass
class FeatureSelectionResult:
    """Result of feature selection analysis."""
    # Stage results
    stage_1_features: List[str] = field(default_factory=list)
    stage_2_features: List[str] = field(default_factory=list)
    stage_3_features: List[str] = field(default_factory=list)
    final_features: List[str] = field(default_factory=list)
    
    # Scores and metrics
    stage_1_scores: Dict[str, Any] = field(default_factory=dict)
    stage_2_scores: Dict[str, Any] = field(default_factory=dict)
    stage_3_scores: Dict[str, Any] = field(default_factory=dict)
    final_scores: Dict[str, Any] = field(default_factory=dict)
    
    # Feature importance
    rf_importance: Dict[str, float] = field(default_factory=dict)
    shap_importance: Dict[str, float] = field(default_factory=dict)
    combined_importance: Dict[str, float] = field(default_factory=dict)
    
    # Analysis metadata
    feature_counts: Dict[str, int] = field(default_factory=dict)
    selection_time: float = 0.0
    model_performance: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    variance_analysis: Dict[str, Any] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    entropy_variance: Dict[str, float] = field(default_factory=dict)
    entropy_stability: Dict[str, float] = field(default_factory=dict)
    entropy_removed_features: Dict[str, float] = field(default_factory=dict)

class MultiStageFeatureSelector:
    """Multi-stage feature selection using RandomForest and SHAP with vectorization and caching."""

    def __init__(self, config: Optional[FeatureSelectionConfig] = None, execution_mode_config: Optional[Dict[str, Any]] = None):
        self.config = config or FeatureSelectionConfig()
        self.logger = get_logger("MultiStageFeatureSelector")
        self.matrix_ops = get_unified_matrix_operations()

        # Initialize hardware optimization tools if available
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.hardware_manager = get_unified_hardware_manager()
            self.adaptive_engine = get_adaptive_optimization_engine()
            self.memory_optimizer = get_advanced_memory_optimizer()
            self.logger.info("🚀 Hardware optimization tools initialized")
        else:
            self.hardware_manager = None
            self.adaptive_engine = None
            self.memory_optimizer = None
            self.logger.info("⚠️ Hardware optimization tools not available")

        # Initialize batch processor for chunked processing
        if MATRIX_OPERATIONS_AVAILABLE:
            self.batch_processor = get_batch_matrix_processor()
            self.logger.info("📦 Batch matrix processor initialized")
        else:
            self.batch_processor = None
            self.logger.info("⚠️ Batch matrix processor not available")

        # Initialize execution mode configuration
        self.execution_mode_config = execution_mode_config
        if self.execution_mode_config:
            self.logger.info(f"📊 Using execution mode configuration for feature selection")
        else:
            self.logger.info("📊 No execution mode configuration provided, using defaults")

        # Initialize results
        self.results = FeatureSelectionResult()
        self.results.polarity_adjustments = {}
        self.results.sign_stability = {}

        # Set model-specific parameters
        self._set_model_specific_parameters()

        # Initialize directional results for separate long/short feature sets
        if self.config.separate_directional_features:
            self.long_results = FeatureSelectionResult()
            self.short_results = FeatureSelectionResult()
            self.long_results.polarity_adjustments = {}
            self.short_results.polarity_adjustments = {}
            self.long_results.sign_stability = {}
            self.short_results.sign_stability = {}

        # Initialize caching system for vectorized operations
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Initialize vectorized computation arrays
        self._vectorized_arrays = {}
        self._computation_cache = {}

        # Initialize polarity tracking containers
        self.feature_polarity_adjustments: Dict[str, Dict[str, Any]] = {}
        self.feature_polarity_history: Dict[str, List[float]] = {}
        self.feature_sign_stability: Dict[str, float] = {}

        tprint("🚀 MultiStageFeatureSelector initialized with vectorization and caching")
        tprint(f"🎯 Model Type: {self.config.model_type}")
        tprint(f"📊 Feature Range: {self.config.min_features}-{self.config.max_features} (target: {self.config.target_features})")
        tprint(f"🎯 Direction Mode: {self.config.direction_mode}")
        tprint(f"🎯 Separate Features: {self.config.separate_directional_features}")
        tprint(f"⚡ Early Termination: {self.config.enable_early_termination}")
        tprint(f"⚡ LightGBM: {LIGHTGBM_AVAILABLE}")
        tprint(f"⚡ RFE: {self.config.enable_rfe}")
        tprint(f"⚡ Mutual Information: {self.config.enable_mutual_information}")
        tprint(f"⚡ Chunked Processing: {self.config.enable_chunked_processing}")
        tprint(f"⚡ Vectorization: Enabled")
        tprint(f"⚡ Caching: Enabled")

    def _set_model_specific_parameters(self):
        """Set model-specific feature selection parameters."""
        model_specific_params = {
            'AdvancedMambaHybrid': {
                'model_correlation_threshold': 0.88,  # Allow more correlated features for multi-timeframe fusion
                'model_importance_threshold': 0.003,  # Moderate threshold for attention mechanisms
                'min_correlation_threshold': 0.92,   # Tighter initial correlation filtering
                'min_variance_threshold': 0.02,      # Higher variance requirement
                'cv_scoring': 'neg_mean_squared_error'
            },
            'FinancialResNet': {
                'model_correlation_threshold': 0.95,  # Tighter correlation for regime classification
                'model_importance_threshold': 0.002,  # Lower threshold for comprehensive input
                'min_correlation_threshold': 0.96,   # Very tight correlation filtering
                'min_variance_threshold': 0.01,      # Standard variance requirement
                'cv_scoring': 'average_precision'  # For regime classification with imbalanced labels
            },
            'DeepScaler': {
                'model_correlation_threshold': 0.85,  # Looser correlation for precision focus
                'model_importance_threshold': 0.008,  # Higher threshold for cleaner features
                'min_correlation_threshold': 0.98,   # Very tight correlation filtering
                'min_variance_threshold': 0.03,      # Higher variance requirement
                'cv_scoring': 'neg_mean_squared_error'
            },
            'NBEATS': {
                'model_correlation_threshold': 0.90,  # Moderate correlation for time series
                'model_importance_threshold': 0.005,  # Standard threshold for temporal modeling
                'min_correlation_threshold': 0.94,   # Tight correlation for clean time series
                'min_variance_threshold': 0.015,     # Moderate variance requirement
                'cv_scoring': 'neg_mean_squared_error'
            }
        }

        if self.config.model_type in model_specific_params:
            params = model_specific_params[self.config.model_type]
            for param, value in params.items():
                setattr(self.config, param, value)
            tprint(f"✅ Applied {self.config.model_type} specific parameters")

    def _get_cache_key(self, operation: str, data_hash: str, params: Dict[str, Any] = None) -> str:
        """Generate a cache key for an operation."""
        params_str = str(sorted(params.items())) if params else ""
        return f"{operation}_{data_hash}_{params_str}"

    def _get_data_hash(self, X: pd.DataFrame, y: pd.Series = None) -> str:
        """Generate a hash for the input data."""
        data_str = f"{X.shape}_{X.columns.tolist()}"
        if y is not None:
            data_str += f"_{y.shape}_{y.dtype}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _vectorized_correlation_analysis(self, X: pd.DataFrame) -> np.ndarray:
        """Vectorized correlation analysis using numpy operations."""
        cache_key = self._get_cache_key("correlation", self._get_data_hash(X))
        
        if cache_key in self._cache:
            self._cache_hits += 1
            tprint(f"📊 Cache hit for correlation analysis (hit rate: {self._cache_hits/(self._cache_hits+self._cache_misses):.2%})")
            return self._cache[cache_key]
        
        self._cache_misses += 1
        tprint("🔄 Computing vectorized correlation matrix...")
        
        # Use numpy for vectorized correlation computation
        X_numeric = X.select_dtypes(include=[np.number])
        corr_matrix = np.corrcoef(X_numeric.T)
        
        # Cache the result
        self._cache[cache_key] = corr_matrix
        tprint(f"✅ Vectorized correlation computed: {corr_matrix.shape}")
        
        return corr_matrix

    def _vectorized_variance_analysis(self, X: pd.DataFrame) -> np.ndarray:
        """Vectorized variance analysis using numpy operations."""
        cache_key = self._get_cache_key("variance", self._get_data_hash(X))
        
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        tprint("🔄 Computing vectorized variance analysis...")
        
        # Use numpy for vectorized variance computation
        X_numeric = X.select_dtypes(include=[np.number])
        variances = np.var(X_numeric.values, axis=0, ddof=1)
        
        # Cache the result
        self._cache[cache_key] = variances
        tprint(f"✅ Vectorized variance computed: {len(variances)} features")
        
        return variances

    def _vectorized_feature_importance(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'rf') -> np.ndarray:
        """Vectorized feature importance computation."""
        cache_key = self._get_cache_key("importance", self._get_data_hash(X, y), {"model_type": model_type})
        
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        tprint(f"🔄 Computing vectorized feature importance using {model_type}...")
        
        # Train model and get importance
        if model_type == 'rf':
            model = self._train_random_forest(X, y)
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = self._train_lightgbm_model(X, y)
        else:
            model = self._train_random_forest(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # Fallback to variance-based importance
            importance = np.var(X.values, axis=0)
        
        # Cache the result
        self._cache[cache_key] = importance
        tprint(f"✅ Vectorized importance computed: {len(importance)} features")
        
        return importance

    def _vectorized_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Vectorized mutual information computation."""
        if not self.config.enable_mutual_information:
            return np.zeros(len(X.columns))
        
        cache_key = self._get_cache_key("mutual_info", self._get_data_hash(X, y))
        
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        tprint("🔄 Computing vectorized mutual information...")
        
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            if self._is_classification(y):
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Normalize scores
            max_mi = np.max(mi_scores) if len(mi_scores) > 0 else 1.0
            normalized_scores = mi_scores / max_mi if max_mi > 0 else mi_scores
            
            # Cache the result
            self._cache[cache_key] = normalized_scores
            tprint(f"✅ Vectorized mutual information computed: {len(normalized_scores)} features")
            
            return normalized_scores
            
        except ImportError:
            tprint("⚠️ sklearn not available for mutual information calculation")
            return np.zeros(len(X.columns))
        except Exception as e:
            tprint(f"⚠️ Mutual information calculation failed: {e}")
            return np.zeros(len(X.columns))

    def _vectorized_stability_analysis(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Vectorized stability analysis across time periods."""
        if len(X) < 100:
            return np.ones(len(X.columns))  # Default stability for small datasets
        
        cache_key = self._get_cache_key("stability", self._get_data_hash(X, y))
        
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        tprint("🔄 Computing vectorized stability analysis...")
        
        try:
            # Split data into chunks for stability analysis
            chunk_size = min(500, len(X) // 3)
            if chunk_size < 50:
                return np.ones(len(X.columns))
            
            # Create overlapping chunks
            chunk_indices = []
            for i in range(0, len(X) - chunk_size + 1, chunk_size // 2):
                chunk_indices.append((i, i + chunk_size))
            
            if len(chunk_indices) < 2:
                return np.ones(len(X.columns))
            
            # Calculate importance for each chunk using vectorized operations
            chunk_importances = []
            for start_idx, end_idx in chunk_indices:
                try:
                    X_chunk = X.iloc[start_idx:end_idx]
                    y_chunk = y.iloc[start_idx:end_idx]
                    
                    # Use vectorized importance calculation
                    importance = self._vectorized_feature_importance(X_chunk, y_chunk)
                    chunk_importances.append(importance)
                except Exception:
                    continue
            
            if len(chunk_importances) < 2:
                return np.ones(len(X.columns))
            
            # Calculate stability as consistency across chunks
            chunk_importances = np.array(chunk_importances)
            
            # Normalize each chunk's importance
            normalized_chunks = chunk_importances / (np.max(chunk_importances, axis=1, keepdims=True) + 1e-8)
            
            # Calculate stability as 1 - coefficient of variation
            mean_importance = np.mean(normalized_chunks, axis=0)
            std_importance = np.std(normalized_chunks, axis=0)
            stability_scores = 1.0 / (1.0 + std_importance / (mean_importance + 1e-8))
            
            # Cache the result
            self._cache[cache_key] = stability_scores
            tprint(f"✅ Vectorized stability analysis computed: {len(stability_scores)} features")
            
            return stability_scores
            
        except Exception as e:
            tprint(f"⚠️ Stability analysis failed: {e}")
            return np.ones(len(X.columns))

    def _clear_cache(self):
        """Clear the computation cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        self._computation_cache.clear()
        tprint(f"🧹 Cache cleared: {cache_size} entries removed")

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        stats = {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'computation_cache_size': len(self._computation_cache)
        }

        tprint(
            (
                "💾 Cache stats — size: {cache_size}, hits: {cache_hits}, misses: {cache_misses}, hit rate: {hit_rate:.2%},"
                " computation cache: {computation_cache_size}"
            ).format(**stats)
        )

        return stats
    
    def select_features(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       feature_names: Optional[List[str]] = None) -> FeatureSelectionResult:
        """Perform multi-stage feature selection with directional support and vectorization."""

        start_time = time.time()
        tprint("🔍 Starting multi-stage feature selection with vectorization and caching")
        tprint(f"📊 Input data: {X.shape[0]} samples, {X.shape[1]} features")
        tprint(f"🎯 Target variable: {y.shape[0]} samples, type: {'classification' if self._is_classification(y) else 'regression'}")

        # Clear cache at start of new selection
        self._clear_cache()

        # Apply execution mode data windowing if configured
        X_processed = X.copy()
        if self.execution_mode_config:
            window_days = self.execution_mode_config.get('window_days', 1460)
            if len(X_processed) > window_days:
                X_processed = X_processed.tail(window_days).copy()
                tprint(f"📊 Applied execution mode window: using last {window_days} samples for feature selection")

        # Apply execution mode stage targets if configured
        default_stage_targets = (
            self.config.initial_features,
            self.config.stage_1_target,
            self.config.stage_2_target,
            self.config.stage_3_target,
        )

        stage_targets: Tuple[int, ...] = default_stage_targets
        if self.execution_mode_config:
            raw_stage_targets = self.execution_mode_config.get('stage_targets', default_stage_targets)

            if isinstance(raw_stage_targets, (list, tuple)):
                padded_targets = list(default_stage_targets)
                for idx, value in enumerate(raw_stage_targets):
                    if idx < len(padded_targets):
                        padded_targets[idx] = value
                    else:
                        padded_targets.append(value)
                stage_targets = tuple(padded_targets)
            elif isinstance(raw_stage_targets, int):
                padded_targets = list(default_stage_targets)
                padded_targets[-1] = raw_stage_targets
                stage_targets = tuple(padded_targets)
            else:
                stage_targets = default_stage_targets

            tprint(f"📊 Using execution mode stage targets: {stage_targets}")

        # Log the feature reduction pipeline
        tprint("🎯 FEATURE REDUCTION PIPELINE:")
        tprint(f"   📊 Initial Features: {stage_targets[0]}")
        tprint(f"   📊 Stage 1 Target: {stage_targets[1]} features")
        tprint(f"   📊 Stage 2 Target: {stage_targets[2]} features")
        tprint(f"   📊 Stage 3 Target: {stage_targets[3]} features")
        tprint(f"   📊 Final Target: {stage_targets[3]} features")

        # Determine which directions to process based on mode
        directions_to_process = self._get_directions_to_process()

        # Handle separate directional feature selection
        if self.config.separate_directional_features and self.config.direction_mode in ['both', 'long_only', 'short_only']:
            tprint(f"🎯 Processing directional features: {directions_to_process}")
            result = self._select_directional_features(X_processed, y, feature_names, directions_to_process, stage_targets)
        else:
            tprint("🎯 Processing unified feature selection")
            result = self._select_unified_features(X_processed, y, feature_names, stage_targets)

        # Log final cache statistics
        cache_stats = self._get_cache_stats()
        tprint("📊 CACHE STATISTICS:")
        tprint(f"   💾 Cache size: {cache_stats['cache_size']} entries")
        tprint(f"   ✅ Cache hits: {cache_stats['cache_hits']}")
        tprint(f"   ❌ Cache misses: {cache_stats['cache_misses']}")
        tprint(f"   📈 Hit rate: {cache_stats['hit_rate']:.2%}")

        # Log execution time
        execution_time = time.time() - start_time
        result.selection_time = execution_time
        tprint(f"⏱️ Total execution time: {execution_time:.3f} seconds")

        # Ensure polarity adjustments are attached to the result payload
        result.polarity_adjustments = getattr(self, 'feature_polarity_adjustments', {})
        result.sign_stability = getattr(self, 'feature_sign_stability', {})

        return result

    def _get_directions_to_process(self) -> List[str]:
        """Determine which directions to process based on configuration."""
        if self.config.direction_mode == 'long_only':
            return ['long']
        elif self.config.direction_mode == 'short_only':
            return ['short']
        else:  # 'both' or other
            return ['long', 'short']

    def _select_directional_features(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   feature_names: Optional[List[str]],
                                   directions: List[str],
                                   stage_targets: Tuple[int, ...]) -> FeatureSelectionResult:
        """Select features separately for each direction."""

        self.logger.info(f"🎯 Selecting directional features for: {directions}")
        tprint(f"🎯 Directional selection activated for: {directions}")

        # Process each direction separately
        for direction in directions:
            self.logger.info(f"🔄 Processing {direction} direction features")
            tprint(f"🔄 Processing direction: {direction}")

            # Filter features for this direction
            direction_features = self._filter_direction_features(X, direction)

            if len(direction_features.columns) < stage_targets[-1]:
                self.logger.warning(f"⚠️ {direction} direction has only {len(direction_features.columns)} features, less than target {stage_targets[-1]}")
                tprint(
                    f"⚠️ {direction.capitalize()} direction insufficient features: {len(direction_features.columns)} available,"
                    f" requires {stage_targets[-1]}"
                )
                if direction == 'long':
                    self.long_results = self._handle_insufficient_features(direction_features, y)
                else:
                    self.short_results = self._handle_insufficient_features(direction_features, y)
                continue

            # Select features for this direction
            direction_result = self._select_unified_features(direction_features, y, feature_names, stage_targets)

            # Store results for this direction
            if direction == 'long':
                self.long_results = direction_result
            else:
                self.short_results = direction_result

        # Compile combined results
        return self._compile_directional_results(directions)

    def _filter_direction_features(self, X: pd.DataFrame, direction: str) -> pd.DataFrame:
        """Filter features for a specific direction."""
        prefix = self.config.directional_feature_prefixes.get(direction, f'{direction}_')

        # Find features that start with the direction prefix
        direction_features = X.columns[X.columns.str.startswith(prefix)]

        if len(direction_features) == 0:
            self.logger.warning(f"⚠️ No {direction} features found with prefix '{prefix}'")
            # Fallback: use all features if no prefixed features found
            return X.copy()

        self.logger.info(f"📊 Found {len(direction_features)} {direction} features")
        return X[direction_features]

    def _compile_directional_results(self, directions: List[str]) -> FeatureSelectionResult:
        """Compile results from directional feature selection."""

        combined_result = FeatureSelectionResult()
        tprint("📦 Compiling directional selection results")

        if 'long' in directions and hasattr(self, 'long_results'):
            # Combine long and short features
            long_features = self.long_results.final_features
            short_features = self.short_results.final_features if 'short' in directions else []

            # Ensure no overlap between long and short features
            combined_features = long_features + short_features

            combined_result.final_features = combined_features
            combined_result.feature_counts = {
                'long_features': len(long_features),
                'short_features': len(short_features),
                'total_features': len(combined_features),
                'long_stage_1': len(self.long_results.stage_1_features),
                'long_stage_2': len(self.long_results.stage_2_features),
                'long_stage_3': len(self.long_results.stage_3_features),
                'short_stage_1': len(self.short_results.stage_1_features) if 'short' in directions else 0,
                'short_stage_2': len(self.short_results.stage_2_features) if 'short' in directions else 0,
                'short_stage_3': len(self.short_results.stage_3_features) if 'short' in directions else 0,
            }

        combined_result.selection_time = time.time() - time.time()  # Will be updated by caller

        self.logger.info(f"✅ Directional feature selection completed")
        self.logger.info(f"📊 Long features: {len(long_features) if 'long' in directions else 0}")
        self.logger.info(f"📊 Short features: {len(short_features) if 'short' in directions else 0}")
        self.logger.info(f"📊 Total unique features: {len(combined_features)}")
        directional_summary = (
            f"✅ Directional selection complete — long: {len(long_features) if 'long' in directions else 0},"
            f" short: {len(short_features) if 'short' in directions else 0}, total: {len(combined_features)}"
        )
        tprint(directional_summary)

        return combined_result

    def _select_unified_features(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]], stage_targets: Tuple[int, ...]) -> FeatureSelectionResult:
        """Original unified feature selection logic."""

        # Validate inputs
        if len(X.columns) < stage_targets[-1]:  # Check against final target
            self.logger.warning(f"⚠️ Input has only {len(X.columns)} features, less than target {stage_targets[-1]}")
            tprint(
                f"⚠️ Unified selection skipped — only {len(X.columns)} features available,"
                f" need at least {stage_targets[-1]}"
            )
            return self._handle_insufficient_features(X, y)

        # Stage 0: Initial feature preparation
        self.logger.info("📊 Stage 0: Initial feature preparation")
        tprint("📊 Stage 0 — preparing initial features")
        prepared_features = self._prepare_initial_features(X, y, feature_names)

        # Stage 1: Initial → Stage 1 target features
        stage_1_target = stage_targets[1] if len(stage_targets) > 1 else 100
        self.logger.info(f"📊 Stage 1: Reducing to {stage_1_target} features")
        tprint(f"🚀 Stage 1 target: {stage_1_target} features")
        stage_1_features, stage_1_scores = self._stage_1_selection(prepared_features, y, target_count=stage_1_target)

        # Stage 2: Stage 1 → Stage 2 target features
        stage_2_target = stage_targets[2] if len(stage_targets) > 2 else 80
        self.logger.info(f"📊 Stage 2: Reducing to {stage_2_target} features")
        tprint(f"🚀 Stage 2 target: {stage_2_target} features")
        stage_2_features, stage_2_scores = self._stage_2_selection(prepared_features[stage_1_features], y, target_count=stage_2_target)

        # Stage 3: Stage 2 → Stage 3 target features
        stage_3_target = stage_targets[3] if len(stage_targets) > 3 else 60
        self.logger.info(f"📊 Stage 3: Reducing to {stage_3_target} features")
        tprint(f"🚀 Stage 3 target: {stage_3_target} features")
        stage_3_features, stage_3_scores = self._stage_3_selection(prepared_features[stage_1_features][stage_2_features], y, target_count=stage_3_target)

        # Compile final results
        tprint("📦 Compiling multi-stage selection results")
        self._compile_results(
            prepared_features, y, stage_1_features, stage_2_features, stage_3_features,
            stage_1_scores, stage_2_scores, stage_3_scores
        )

        # Save results
        if self.config.save_analysis:
            tprint("💾 Saving feature selection analysis")
            self._save_analysis()

        return self.results

    def _evaluate_feature_polarity(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[float]], Dict[str, float]]:
        """Evaluate rolling feature polarity stability and adjust unstable features."""

        adjustments: Dict[str, Dict[str, Any]] = {}
        correlation_history: Dict[str, List[float]] = {}
        sign_stability: Dict[str, float] = {}

        if X.empty:
            return adjustments, correlation_history, sign_stability

        y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
        try:
            y_numeric = pd.to_numeric(y_series, errors='coerce')
        except Exception:
            y_numeric = pd.Series(y_series, index=y_series.index, dtype=float)

        if not y_numeric.index.equals(X.index):
            y_numeric = y_numeric.reindex(X.index)

        valid_mask = ~y_numeric.isna()
        X_numeric = X.select_dtypes(include=[np.number])

        if X_numeric.empty:
            return adjustments, correlation_history, sign_stability

        X_numeric = X_numeric.loc[valid_mask]
        y_valid = y_numeric.loc[valid_mask]

        total_rows = len(X_numeric)
        if total_rows < 15:
            for col in X_numeric.columns:
                sign_stability[col] = 1.0
            return adjustments, correlation_history, sign_stability

        window_size = max(total_rows // 5, 10)
        window_size = min(window_size, total_rows - 1)
        if window_size < 5:
            for col in X_numeric.columns:
                sign_stability[col] = 1.0
            return adjustments, correlation_history, sign_stability

        step_size = max(window_size // 2, 1)

        correlation_accumulator: Dict[str, List[float]] = defaultdict(list)
        sign_accumulator: Dict[str, List[int]] = defaultdict(list)

        for start in range(0, total_rows - window_size + 1, step_size):
            end = start + window_size
            window_X = X_numeric.iloc[start:end]
            window_y = y_valid.iloc[start:end]

            if window_y.std(ddof=0) == 0:
                continue

            window_correlations = window_X.corrwith(window_y)

            for feature, corr_value in window_correlations.items():
                if pd.isna(corr_value):
                    continue

                corr_float = float(corr_value)
                correlation_accumulator[feature].append(corr_float)

                if abs(corr_float) < 1e-9:
                    sign_accumulator[feature].append(0)
                else:
                    sign_accumulator[feature].append(1 if corr_float > 0 else -1)

        recent_window_depth = 10
        valid_index = y_valid.index

        for feature in X_numeric.columns:
            history = correlation_accumulator.get(feature, [])
            correlation_history[feature] = history[-recent_window_depth:]

            signs = sign_accumulator.get(feature, [])
            recent_signs = signs[-recent_window_depth:]
            non_zero_signs = [s for s in recent_signs if s != 0]

            if len(non_zero_signs) <= 1:
                flip_rate = 0.0
            else:
                flips = sum(
                    non_zero_signs[idx] != non_zero_signs[idx - 1]
                    for idx in range(1, len(non_zero_signs))
                )
                flip_rate = flips / (len(non_zero_signs) - 1)

            stability_score = max(0.0, 1.0 - flip_rate)
            sign_stability[feature] = stability_score

            if flip_rate <= 0.7:
                continue

            dominant_sign = 0
            if non_zero_signs:
                positives = non_zero_signs.count(1)
                negatives = non_zero_signs.count(-1)
                if positives > negatives:
                    dominant_sign = 1
                elif negatives > positives:
                    dominant_sign = -1

            original_corr = X_numeric[feature].corr(y_valid)
            final_corr = original_corr
            action = 're_standardized'

            if pd.notna(original_corr) and original_corr < 0:
                X[feature] = -X[feature]
                final_corr = X.loc[valid_index, feature].corr(y_valid)
                action = 'inverted'
            else:
                std = X[feature].std(ddof=0)
                if std and std > 0:
                    mean = X[feature].mean()
                    X[feature] = (X[feature] - mean) / std
                    final_corr = X.loc[valid_index, feature].corr(y_valid)
                else:
                    action = 're_standardization_skipped_zero_variance'

            adjustments[feature] = {
                'action': action,
                'flip_rate': float(flip_rate),
                'sign_stability': float(stability_score),
                'window_size': int(window_size),
                'step_size': int(step_size),
                'evaluated_windows': int(len(non_zero_signs)),
                'dominant_sign': int(dominant_sign),
                'original_correlation': None if pd.isna(original_corr) else float(original_corr),
                'final_correlation': None if pd.isna(final_corr) else float(final_corr),
                'correlation_history': [float(v) for v in correlation_history[feature]],
                'recent_signs': non_zero_signs,
                'reason': 'dominant_sign_flipped_gt_70_percent',
                'timestamp': time.time(),
            }

        return adjustments, correlation_history, sign_stability

    def _prepare_initial_features(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare initial features for selection using vectorized operations."""

        tprint("🔄 Preparing initial features with vectorized operations...")
        tprint(f"📊 Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Handle feature names
        if feature_names is not None:
            X = X[feature_names] if len(feature_names) <= len(X.columns) else X
            tprint(f"📊 After feature name filtering: {X.shape[1]} features")
        
        # Remove low variance features using vectorized operations
        tprint("🔄 Computing vectorized variance analysis...")
        variance_array = self._vectorized_variance_analysis(X)
        variance_threshold = self.config.min_variance_threshold
        
        low_variance_mask = variance_array < variance_threshold
        low_variance_features = X.columns[low_variance_mask].tolist()
        
        if len(low_variance_features) > 0:
            tprint(f"🗑️ Removing {len(low_variance_features)} low variance features (threshold: {variance_threshold})")
            X = X.drop(columns=low_variance_features)
            tprint(f"📊 After variance filtering: {X.shape[1]} features")
        else:
            tprint("✅ No low variance features to remove")

        # Analyze rolling correlation polarity before correlation filtering
        tprint("🔍 Evaluating rolling feature-target correlation stability...")
        adjustments, correlation_history, sign_stability = self._evaluate_feature_polarity(X, y)
        self.feature_polarity_adjustments = adjustments
        self.feature_polarity_history = correlation_history
        self.feature_sign_stability = sign_stability
        self.results.polarity_adjustments = adjustments
        self.results.sign_stability = sign_stability

        if adjustments:
            tprint(f"⚖️ Applied polarity adjustments to {len(adjustments)} features")
        else:
            tprint("✅ No polarity adjustments required")

        # Remove highly correlated features using vectorized correlation computation
        tprint("🔄 Computing vectorized correlation analysis...")
        correlation_threshold = self.config.min_correlation_threshold
        high_corr_features = self._find_highly_correlated_features_vectorized(X, correlation_threshold)

        if len(high_corr_features) > 0:
            tprint(f"🗑️ Removing {len(high_corr_features)} highly correlated features (threshold: {correlation_threshold})")
            X = X.drop(columns=high_corr_features)
            tprint(f"📊 After correlation filtering: {X.shape[1]} features")
        else:
            tprint("✅ No highly correlated features to remove")

        # Apply entropy stability filtering before staging if enabled
        if self.config.enable_entropy_balancing and len(X.columns) > 0:
            tprint("🔄 Evaluating entropy stability across temporal slices...")
            X = self._apply_entropy_balancing_filter(X)
            tprint(f"📊 After entropy balancing: {X.shape[1]} features")
        else:
            tprint("⏭️ Entropy balancing disabled or no features available for evaluation")
        
        # Select top features if we have too many using vectorized operations
        if len(X.columns) > self.config.initial_features:
            tprint(f"📊 Selecting top {self.config.initial_features} features initially")
            # Use vectorized variance-based selection for initial filtering
            remaining_variance = self._vectorized_variance_analysis(X)
            sorted_indices = np.argsort(remaining_variance)[::-1]
            top_indices = sorted_indices[:self.config.initial_features]
            top_features = X.columns[top_indices].tolist()
            X = X[top_features]
            tprint(f"📊 After top feature selection: {X.shape[1]} features")
        
        tprint(f"✅ Prepared {len(X.columns)} features for selection")
        return X

    def _apply_entropy_balancing_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply entropy stability filtering and record diagnostics."""

        balancer_config = EntropyBalancerConfig(
            num_slices=self.config.entropy_num_slices,
            min_slice_size=self.config.entropy_min_slice_size,
            max_entropy_variance=self.config.entropy_variance_threshold,
            max_bins=self.config.entropy_max_bins,
            min_unique_values=self.config.entropy_min_unique_values,
            use_time_index=self.config.entropy_use_time_index,
        )

        filter_engine = EntropyStabilityFilter(balancer_config)
        filter_result: EntropyFilterResult = filter_engine.filter(X)

        self.results.entropy_variance = filter_result.entropy_variance
        self.results.entropy_stability = filter_result.stability_scores
        self.results.entropy_removed_features = filter_result.dropped_features

        if filter_result.dropped_features:
            dropped_preview = list(filter_result.dropped_features.items())[:5]
            tprint(f"🗑️ Entropy filter removed {len(filter_result.dropped_features)} unstable features")
            for feature, variance in dropped_preview:
                tprint(f"   • {feature}: variance={variance:.4f}")
            if filter_result.selected_features:
                return X[filter_result.selected_features]

            # Fallback: retain lowest variance features to avoid empty set
            tprint("⚠️ All features flagged as unstable; retaining the most stable subset")
            sorted_features = sorted(
                filter_result.entropy_variance.items(),
                key=lambda item: item[1]
            )
            fallback_count = max(1, int(len(sorted_features) * 0.5))
            fallback_features = [feature for feature, _ in sorted_features[:fallback_count]]
            self.results.entropy_removed_features = {
                feature: variance
                for feature, variance in filter_result.entropy_variance.items()
                if feature not in fallback_features
            }
            return X[fallback_features]

        tprint("✅ Entropy filter retained all features")
        return X

    def _find_highly_correlated_features_vectorized(self, X: pd.DataFrame, threshold: float) -> List[str]:
        """Find and remove highly correlated features using vectorized correlation computation."""
        tprint(f"🔄 Computing vectorized correlation matrix for {len(X.columns)} features...")
        
        # Use vectorized correlation computation
        corr_matrix = self._vectorized_correlation_analysis(X)
        
        # Find features to drop using vectorized operations
        corr_matrix_abs = np.abs(corr_matrix)
        upper_triangle = np.triu(corr_matrix_abs, k=1)
        
        # Find features to drop
        to_drop_mask = np.any(upper_triangle > threshold, axis=0)
        to_drop = X.columns[to_drop_mask].tolist()
        
        tprint(f"📊 Correlation analysis: {len(to_drop)} features above threshold {threshold}")
        return to_drop
    
    def _find_highly_correlated_features(self, X: pd.DataFrame, threshold: float) -> List[str]:
        """Find and remove highly correlated features using optimized correlation computation."""
        if MATRIX_OPERATIONS_AVAILABLE and len(X.columns) > 100:
            # Use GPU-accelerated correlation for large datasets
            self.logger.info(f"🚀 Using GPU-accelerated correlation for {len(X.columns)} features")
            try:
                corr_matrix = correlation_matrix_gpu(X)
                corr_matrix = pd.DataFrame(corr_matrix, index=X.columns, columns=X.columns)
            except Exception as e:
                self.logger.warning(f"⚠️ GPU correlation failed, falling back to CPU: {e}")
                corr_matrix = X.corr()
        else:
            corr_matrix = X.corr()

        corr_matrix = corr_matrix.abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

        return to_drop

    def _calculate_mutual_information_correlation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate mutual information scores for features - captures non-linear relationships."""
        if not self.config.enable_mutual_information:
            return {}

        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

            mi_scores = {}
            if self._is_classification(y):
                mi_scores = dict(zip(X.columns, mutual_info_classif(X, y, random_state=42)))
            else:
                mi_scores = dict(zip(X.columns, mutual_info_regression(X, y, random_state=42)))

            # Normalize scores to 0-1 range
            max_mi = max(mi_scores.values()) if mi_scores else 1.0
            normalized_scores = {k: v / max_mi for k, v in mi_scores.items()}

            # Log non-linear relationships detected
            high_mi_features = [f for f, score in normalized_scores.items() if score > 0.5]
            if high_mi_features:
                tprint(f"🔗 Non-linear relationships detected in {len(high_mi_features)} features")

            return normalized_scores

        except ImportError:
            self.logger.warning("⚠️ sklearn not available for mutual information calculation")
            return {}
        except Exception as e:
            self.logger.warning(f"⚠️ Mutual information calculation failed: {e}")
            return {}

    def _calculate_feature_stability_score(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature stability across different time periods/regimes."""
        if len(X) < 100:  # Need sufficient data for stability analysis
            return {}

        try:
            # Split data into chunks for stability analysis
            chunk_size = min(500, len(X) // 3)  # At least 3 chunks
            if chunk_size < 50:
                return {}

            stability_scores = {}
            chunks = []

            # Create overlapping chunks for stability analysis
            for i in range(0, len(X) - chunk_size + 1, chunk_size // 2):
                chunk_data = X.iloc[i:i + chunk_size]
                chunk_target = y.iloc[i:i + chunk_size]
                chunks.append((chunk_data, chunk_target))

            if len(chunks) < 2:
                return {}

            # Calculate importance for each chunk
            chunk_importances = []
            for chunk_X, chunk_y in chunks:
                try:
                    model = self._train_optimized_model(chunk_X, chunk_y)
                    if hasattr(model, 'feature_importances_'):
                        importance = dict(zip(chunk_X.columns, model.feature_importances_))
                        chunk_importances.append(importance)
                except Exception:
                    continue

            if len(chunk_importances) < 2:
                return {}

            # Calculate stability as consistency across chunks
            all_features = set()
            for imp in chunk_importances:
                all_features.update(imp.keys())

            for feature in all_features:
                feature_stabilities = []
                for imp in chunk_importances:
                    if feature in imp:
                        # Normalize importance within each chunk
                        max_imp = max(imp.values()) if imp else 1.0
                        feature_stabilities.append(imp[feature] / max_imp)

                if len(feature_stabilities) >= 2:
                    # Stability = 1 - coefficient of variation (lower variation = higher stability)
                    stability_scores[feature] = 1.0 / (1.0 + np.std(feature_stabilities) / (np.mean(feature_stabilities) + 1e-8))

            # Log stability insights
            stable_features = [f for f, score in stability_scores.items() if score > 0.7]
            if stable_features:
                tprint(f"🛡️ Stability analysis: {len(stable_features)} consistently important features")

            return stability_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Feature stability calculation failed: {e}")
            return {}

    def _train_lightgbm_model(self, X: pd.DataFrame, y: pd.Series):
        """Train LightGBM model with optimized parameters."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        tprint(f"🚀 Training LightGBM model on {X.shape[0]} samples × {X.shape[1]} features")

        # Determine if classification or regression
        if self._is_classification(y):
            objective = 'binary' if len(y.unique()) == 2 else 'multiclass'
            params = self.config.lightgbm_params.copy()
            params['objective'] = objective
            params['num_class'] = len(y.unique()) if objective == 'multiclass' else 1

            model = lgb.LGBMClassifier(**params)
            tprint(f"🎯 LightGBM classification objective: {objective}")
        else:
            params = self.config.lightgbm_params.copy()
            params['objective'] = 'regression'

            model = lgb.LGBMRegressor(**params)
            tprint("🎯 LightGBM regression objective")

        # Train model
        model.fit(X, y)
        tprint("✅ LightGBM training complete")

        return model

    def _train_optimized_model(self, X: pd.DataFrame, y: pd.Series):
        """Train either LightGBM or RandomForest based on availability and performance."""
        try:
            if LIGHTGBM_AVAILABLE and len(X.columns) > 50:
                self.logger.info("🚀 Using LightGBM for faster training")
                tprint(f"🚀 Optimized training — selecting LightGBM (features: {len(X.columns)})")
                return self._train_lightgbm_model(X, y)
            else:
                self.logger.info("📊 Using RandomForest (LightGBM not available or dataset too small)")
                tprint("📊 Optimized training — selecting RandomForest")
                return self._train_random_forest(X, y)
        except Exception as e:
            self.logger.warning(f"⚠️ LightGBM training failed, falling back to RandomForest: {e}")
            tprint(f"⚠️ LightGBM training failed ({e}), falling back to RandomForest")
            return self._train_random_forest(X, y)

    def _calculate_adaptive_importance_threshold(self, importance_scores: Dict[str, float]) -> float:
        """Calculate adaptive importance threshold based on feature distribution."""
        if not self.config.adaptive_importance_threshold:
            return self.config.early_termination_threshold

        importances = list(importance_scores.values())
        if not importances:
            return self.config.early_termination_threshold

        # Use percentile-based threshold - be less aggressive
        threshold_percentile = max(10.0, self.config.importance_percentile_cutoff)  # At least 10th percentile
        threshold = np.percentile(importances, threshold_percentile)

        # For very large feature sets, be even more conservative
        if len(importances) > 1000:
            # Keep more features for large datasets - use 15th percentile instead of 10th
            threshold = max(threshold, np.percentile(importances, 15.0))
            tprint(f"📊 Large dataset detected ({len(importances)} features), using conservative threshold")

        # Ensure minimum threshold but don't be too restrictive
        min_threshold = self.config.early_termination_threshold
        final_threshold = max(threshold, min_threshold)

        # Log detailed threshold calculation for troubleshooting
        sorted_importances = sorted(importances, reverse=True)
        tprint(f"🔍 Threshold Analysis: {len(importances)} features")
        tprint(f"   📈 Max importance: {sorted_importances[0]:.6f}")
        tprint(f"   📉 Min importance: {sorted_importances[-1]:.6f}")
        tprint(f"   🎯 {threshold_percentile:.1f}th percentile: {threshold:.6f}")
        tprint(f"   ⚖️ Final threshold: {final_threshold:.6f}")

        return final_threshold

    def _apply_early_termination(self, X: pd.DataFrame, importance_scores: Dict[str, float]) -> pd.DataFrame:
        """Apply early termination to remove low-importance features."""
        if not self.config.enable_early_termination:
            tprint("⏭️ Early termination disabled, keeping all features")
            return X

        threshold = self._calculate_adaptive_importance_threshold(importance_scores)

        # Find features above threshold
        selected_features = [f for f, score in importance_scores.items() if score >= threshold]

        # Be conservative - don't allow pruning more than 20% of features
        total_features = len(X.columns)
        remaining_features = len(selected_features)
        removal_rate = (total_features - remaining_features) / total_features

        # If we're removing more than 20% of features, be much more conservative
        if removal_rate > 0.2:
            tprint(f"⚠️ High removal rate ({removal_rate:.1%}), applying very conservative pruning")
            # Keep top 80% of features instead
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            keep_count = max(1, int(len(sorted_features) * 0.8))  # Keep top 80%
            selected_features = [f for f, _ in sorted_features[:keep_count]]
            threshold = sorted_features[keep_count - 1][1] if keep_count > 0 else 0.0
            tprint(f"🎯 Conservative fallback: keeping top 80% ({keep_count}) features")

        if len(selected_features) == 0:
            tprint("⚠️ No features above threshold, keeping top 80% to preserve top performers")
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            # Keep top 80% to be very conservative
            keep_count = max(1, int(len(sorted_features) * 0.8))
            selected_features = [f for f, _ in sorted_features[:keep_count]]
            threshold = sorted_features[keep_count - 1][1] if keep_count > 0 else 0.0

        # Performance and feature information
        removed_count = total_features - len(selected_features)
        tprint("🗑️ Early Termination Results:")
        tprint(f"   📊 Total features: {total_features}")
        tprint(f"   ✅ Remaining features: {len(selected_features)}")
        tprint(f"   🗑️ Removed features: {removed_count}")
        tprint(f"   📈 Threshold used: {threshold:.6f}")
        tprint(f"   📉 Removal rate: {removal_rate:.1%}")

        # Show top 5 features being kept and bottom 5 being removed for troubleshooting
        if len(selected_features) > 0:
            sorted_selected = sorted([(f, importance_scores[f]) for f in selected_features],
                                   key=lambda x: x[1], reverse=True)
            tprint("🏆 Top 5 Kept Features:")
            for i, (feature, score) in enumerate(sorted_selected[:5]):
                tprint(f"   {i+1}. {feature}: {score:.6f}")

        if removed_count > 0:
            removed_features = [(f, importance_scores[f]) for f in X.columns if f not in selected_features]
            sorted_removed = sorted(removed_features, key=lambda x: x[1], reverse=True)
            tprint("💔 Bottom 5 Removed Features:")
            for i, (feature, score) in enumerate(sorted_removed[:5]):
                tprint(f"   {i+1}. {feature}: {score:.6f}")

        self.logger.info(f"🗑️ Early termination: removed {removed_count} features below threshold {threshold:.6f}")

        return X[selected_features]

    def _recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Perform recursive feature elimination with early stopping."""
        if not self.config.enable_rfe or not SKLEARN_FEATURE_SELECTION_AVAILABLE:
            tprint("⏭️ RFE disabled or unavailable, keeping all features")
            return X.columns.tolist()

        current_features = X.columns.tolist()
        best_score = float('-inf') if self._is_classification(y) else float('inf')
        no_improvement_count = 0
        step_count = 0

        tprint(f"🔄 Starting RFE: {len(current_features)} features -> target: ~{self.config.rfe_min_features}")
        tprint(f"📉 Step size: {self.config.rfe_step_size:.1%} ({max(1, int(len(current_features) * self.config.rfe_step_size))} features)")

        while len(current_features) > self.config.rfe_min_features:
            step_count += 1
            # Calculate step size
            step_size = max(1, int(len(current_features) * self.config.rfe_step_size))

            if len(current_features) - step_size < self.config.rfe_min_features:
                tprint(f"🛑 Stopping RFE: would go below minimum ({self.config.rfe_min_features})")
                break

            # Try removing step_size features
            try:
                tprint(f"🔄 RFE Step {step_count}: Training model on {len(current_features)} features...")
                estimator = self._train_optimized_model(X[current_features], y)

                tprint(f"🔄 RFE Step {step_count}: Selecting {len(current_features) - step_size} features...")
                selector = RFE(estimator, n_features_to_select=len(current_features) - step_size, step=1)
                selector = selector.fit(X[current_features], y)

                selected_features = [f for f, selected in zip(current_features, selector.support_) if selected]

                # Evaluate performance
                if self._is_classification(y):
                    score = selector.estimator_.score(X[selected_features], y)
                else:
                    score = -selector.estimator_.score(X[selected_features], y)  # Negative for regression

                tprint(f"📊 RFE Step {step_count}: {len(selected_features)} features, score: {score:.4f}")

                # Check for improvement
                if (self._is_classification(y) and score > best_score) or \
                   (not self._is_classification(y) and score < best_score):
                    best_score = score
                    current_features = selected_features
                    no_improvement_count = 0
                    tprint(f"✅ RFE Step {step_count}: Score improved to {score:.4f}")
                else:
                    no_improvement_count += 1
                    tprint(f"⚠️ RFE Step {step_count}: No improvement (patience: {no_improvement_count}/{self.config.rfe_early_stopping_patience})")
                    if no_improvement_count >= self.config.rfe_early_stopping_patience:
                        tprint(f"🛑 RFE early stopping: no improvement for {no_improvement_count} steps")
                        break

            except Exception as e:
                tprint(f"❌ RFE step {step_count} failed: {e}")
                break

        tprint(f"✅ RFE completed: {len(current_features)} features selected in {step_count} steps")
        return current_features

    def _process_features_in_chunks(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Process features in chunks for memory efficiency."""
        if not self.config.enable_chunked_processing or not MATRIX_OPERATIONS_AVAILABLE:
            tprint("⏭️ Chunked processing disabled or unavailable")
            return X

        total_features = len(X.columns)
        chunk_size = min(self.config.chunk_size, total_features)

        if total_features <= chunk_size:
            tprint(f"📊 Dataset small ({total_features} features), no chunking needed")
            return X

        tprint(f"📦 Processing {total_features} features in chunks of {chunk_size}")
        tprint(f"🔗 Chunk overlap: {self.config.chunk_overlap} features")

        # Split features into chunks with overlap
        all_selected_features = []
        chunk_overlap = self.config.chunk_overlap

        for i in range(0, total_features, chunk_size - chunk_overlap):
            start_idx = i
            end_idx = min(i + chunk_size, total_features)

            chunk_features = X.columns[start_idx:end_idx].tolist()
            X_chunk = X[chunk_features]

            tprint(f"🔄 Processing chunk {i//chunk_size + 1}: {len(chunk_features)} features")

            # Process this chunk
            chunk_selected = self._select_features_chunk(X_chunk, y)

            # Add to overall selection
            all_selected_features.extend(chunk_selected)

            # Remove duplicates while preserving order
            seen = set()
            unique_features = []
            for f in all_selected_features:
                if f not in seen:
                    seen.add(f)
                    unique_features.append(f)

            all_selected_features = unique_features

            tprint(f"📈 Chunk result: {len(chunk_selected)} selected, total so far: {len(all_selected_features)}")

            # Stop if we have enough features
            if len(all_selected_features) >= self.config.target_features * 1.5:  # 50% buffer
                tprint(f"🎯 Target reached ({len(all_selected_features)} features), stopping early")
                break

            # Limit total chunks
            if (i // chunk_size) >= self.config.max_chunks - 1:
                tprint(f"⚠️ Max chunks ({self.config.max_chunks}) reached, stopping")
                break

        tprint(f"📦 Chunked processing completed: {len(all_selected_features)} unique features selected")
        return X[all_selected_features]

    def _select_features_chunk(self, X_chunk: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features for a single chunk."""
        # Use a simplified version of the selection process for chunks
        if len(X_chunk.columns) < 10:
            tprint(f"📦 Chunk small ({len(X_chunk.columns)} features), selecting all")
            return X_chunk.columns.tolist()

        # Train model and get importance
        try:
            tprint(f"🚀 Training chunk model for {len(X_chunk.columns)} features")
            model = self._train_optimized_model(X_chunk, y)
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(X_chunk.columns, model.feature_importances_))
            else:
                # For LightGBM, get feature importance
                importance = dict(zip(X_chunk.columns, model.feature_importances_))

            # Select top features
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            target_count = min(len(X_chunk.columns) // 2, self.config.target_features // 4)  # Conservative selection
            selected = [f for f, _ in sorted_features[:target_count]]

            tprint(f"✅ Chunk selection complete — {len(selected)} features retained")

            return selected

        except Exception as e:
            self.logger.warning(f"⚠️ Chunk selection failed: {e}")
            tprint(f"⚠️ Chunk selection failed ({e}), falling back to variance selection")
            # Fallback to simple variance-based selection
            variances = X_chunk.var().sort_values(ascending=False)
            target_count = min(len(X_chunk.columns) // 2, 10)
            fallback_features = variances.head(target_count).index.tolist()
            tprint(f"✅ Variance fallback selected {len(fallback_features)} features")
            return fallback_features
    
    def _stage_1_selection(self, X: pd.DataFrame, y: pd.Series, target_count: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """Stage 1: Initial → target features using vectorized operations and optimized model selection."""

        tprint("🚀 Starting Stage 1 Feature Selection (Vectorized)")
        tprint(f"📊 Input: {len(X)} samples, {len(X.columns)} features")

        if self._is_classification(y):
            tprint(f"🎯 Classification task: {len(y.unique())} classes")
        else:
            tprint(f"🎯 Regression task: target range [{y.min():.3f}, {y.max():.3f}]")

        # Use provided target count or fall back to config
        actual_target = target_count or self.config.stage_1_target
        tprint(f"🎯 Target features: {actual_target}")

        # Apply chunked processing for large feature sets
        if self.config.enable_chunked_processing and len(X.columns) > self.config.chunk_size:
            tprint(f"📦 Applying chunked processing for {len(X.columns)} features")
            X = self._process_features_in_chunks(X, y)

        # Use vectorized feature importance computation
        tprint("🔄 Computing vectorized feature importance...")
        feature_importance_array = self._vectorized_feature_importance(X, y, 'rf')
        feature_importance = dict(zip(X.columns, feature_importance_array))
        
        tprint(f"📈 Feature importance range: [{np.min(feature_importance_array):.6f}, {np.max(feature_importance_array):.6f}]")
        tprint(f"📊 Feature importance mean: {np.mean(feature_importance_array):.6f}")
        tprint(f"📊 Feature importance std: {np.std(feature_importance_array):.6f}")

        # Apply early termination if enabled
        if self.config.enable_early_termination:
            tprint("🗑️ Applying early termination...")
            X = self._apply_early_termination(X, feature_importance)

        # Apply RFE if enabled
        if self.config.enable_rfe and len(X.columns) > actual_target:
            tprint(f"🔄 Applying RFE to reduce from {len(X.columns)} to ~{actual_target} features")
            rfe_features = self._recursive_feature_elimination(X, y)
            X = X[rfe_features]
            tprint(f"✅ RFE completed: {len(X.columns)} features remaining")

        # Select top features using vectorized operations
        tprint("🏆 Selecting top features using vectorized operations...")
        sorted_indices = np.argsort(feature_importance_array)[::-1]
        selected_indices = sorted_indices[:actual_target]
        selected_features = [X.columns[i] for i in selected_indices]

        tprint(f"🏆 Final selection: {len(selected_features)} features")
        tprint(f"📈 Top feature importance: {feature_importance_array[sorted_indices[0]]:.6f}" if len(sorted_indices) > 0 else "N/A")

        # Calculate enhanced scores using vectorized operations
        tprint("📊 Computing enhanced scores...")
        selected_importance = feature_importance_array[selected_indices]
        
        scores = {
            'model_importance_score': np.mean(selected_importance),
            'feature_variance': np.mean(X[selected_features].var().values),
            'selection_quality': len(selected_features) / len(X.columns),
            'model_type': 'vectorized_rf',
            'importance_std': np.std(selected_importance),
            'importance_range': np.max(selected_importance) - np.min(selected_importance)
        }

        # Add mutual information scores if enabled
        if self.config.enable_mutual_information:
            tprint("🔗 Calculating vectorized mutual information...")
            mi_scores_array = self._vectorized_mutual_information(X[selected_features], y)
            scores['mutual_information'] = np.mean(mi_scores_array)
            scores['mutual_information_std'] = np.std(mi_scores_array)
            tprint(f"📊 Mutual information average: {scores['mutual_information']:.4f}")

        tprint(f"✅ Stage 1 completed: {len(selected_features)}/{len(X.columns)} features selected")
        tprint(f"📊 Selection quality: {scores['selection_quality']:.2%}")
        return selected_features, scores
    
    def _stage_2_selection(self, X: pd.DataFrame, y: pd.Series, target_count: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """Stage 2: Previous → target features using vectorized enhanced selection methods."""

        # Use provided target count or fall back to config
        actual_target = target_count or self.config.stage_2_target

        tprint("🚀 Starting Stage 2 Feature Selection (Vectorized Enhanced)")
        tprint(f"📊 Input: {len(X)} samples, {len(X.columns)} features")
        tprint(f"🎯 Target features: {actual_target}")

        # Use vectorized feature importance computation with LightGBM if available
        model_type = 'lightgbm' if LIGHTGBM_AVAILABLE else 'rf'
        tprint(f"🔄 Computing vectorized feature importance using {model_type}...")
        feature_importance_array = self._vectorized_feature_importance(X, y, model_type)
        feature_importance = dict(zip(X.columns, feature_importance_array))
        
        tprint(f"📈 Feature importance range: [{np.min(feature_importance_array):.6f}, {np.max(feature_importance_array):.6f}]")
        tprint(f"📊 Feature importance mean: {np.mean(feature_importance_array):.6f}")

        # Apply early termination if enabled
        if self.config.enable_early_termination:
            tprint("🗑️ Applying early termination...")
            X = self._apply_early_termination(X, feature_importance)

        # Apply RFE if enabled and we still have many features
        if self.config.enable_rfe and len(X.columns) > actual_target * 1.5:
            tprint(f"🔄 Applying RFE to reduce from {len(X.columns)} to ~{actual_target} features")
            rfe_features = self._recursive_feature_elimination(X, y)
            X = X[rfe_features]
            tprint(f"✅ RFE completed: {len(X.columns)} features remaining")

        # Select top features using vectorized operations
        tprint("🏆 Selecting top features using vectorized operations...")
        sorted_indices = np.argsort(feature_importance_array)[::-1]
        selected_indices = sorted_indices[:actual_target]
        selected_features = [X.columns[i] for i in selected_indices]

        tprint(f"🏆 Stage 2 selection: {len(selected_features)} features")

        # Calculate enhanced scores using vectorized operations
        tprint("📊 Computing enhanced scores...")
        selected_importance = feature_importance_array[selected_indices]
        
        scores = {
            'model_importance_score': np.mean(selected_importance),
            'feature_variance': np.mean(X[selected_features].var().values),
            'selection_quality': len(selected_features) / len(X.columns),
            'model_type': f'vectorized_{model_type}',
            'importance_std': np.std(selected_importance),
            'importance_range': np.max(selected_importance) - np.min(selected_importance)
        }

        # Add mutual information scores if enabled
        if self.config.enable_mutual_information:
            tprint("🔗 Calculating vectorized mutual information...")
            mi_scores_array = self._vectorized_mutual_information(X[selected_features], y)
            scores['mutual_information'] = np.mean(mi_scores_array)
            scores['mutual_information_std'] = np.std(mi_scores_array)
            tprint(f"📊 Mutual information average: {scores['mutual_information']:.4f}")

        # Use SHAP if available for final refinement
        if SHAP_AVAILABLE and len(selected_features) <= self.config.shap_max_features:
            tprint("🔮 Applying SHAP refinement...")
            try:
                shap_features, shap_scores = self._shap_based_selection(X[selected_features], y, actual_target)
                if shap_features:
                    selected_features = shap_features
                    scores.update(shap_scores)
                    tprint("✅ SHAP refinement completed")
                else:
                    tprint("⚠️ SHAP refinement returned no features")
            except Exception as e:
                tprint(f"⚠️ SHAP refinement failed: {e}")

        tprint(f"✅ Stage 2 completed: {len(selected_features)} features selected")
        tprint(f"📊 Selection quality: {scores['selection_quality']:.2%}")
        return selected_features, scores

    def _stage_3_selection(self, X: pd.DataFrame, y: pd.Series, target_count: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """Stage 3: Previous → target features using vectorized combined importance and cross-validation."""

        # Use provided target count or fall back to config
        actual_target = target_count or self.config.stage_3_target

        tprint("🚀 Starting Stage 3 Feature Selection (Vectorized Combined)")
        tprint(f"📊 Input: {len(X)} samples, {len(X.columns)} features")
        tprint(f"🎯 Target features: {actual_target}")

        # Use vectorized feature importance computation
        tprint("🔄 Computing vectorized model importance...")
        model_importance_array = self._vectorized_feature_importance(X, y, 'rf')
        model_importance = dict(zip(X.columns, model_importance_array))

        # Cross-validation based selection using vectorized operations
        tprint("🔄 Performing vectorized cross-validation feature importance...")
        cv_scores = self._cross_validate_feature_importance_optimized(X, y)
        cv_scores_array = np.array([cv_scores.get(f, 0) for f in X.columns])
        tprint(f"✅ CV completed for {len(cv_scores)} features")

        # Calculate non-linear quality metrics using vectorized operations
        tprint("🔗 Calculating vectorized mutual information...")
        mi_scores_array = self._vectorized_mutual_information(X, y)
        tprint(f"✅ MI calculated for {len(mi_scores_array)} features")

        # Calculate feature stability across time periods using vectorized operations
        tprint("🛡️ Analyzing vectorized feature stability...")
        stability_scores_array = self._vectorized_stability_analysis(X, y)
        tprint(f"✅ Stability analyzed for {len(stability_scores_array)} features")

        # Combine importance scores with non-linear awareness using vectorized operations
        tprint("⚖️ Combining importance scores with vectorized non-linear awareness...")
        
        # Vectorized combination: model + CV + MI + stability
        base_scores = (model_importance_array * 0.3 + cv_scores_array * 0.3 + mi_scores_array * 0.2)
        stability_multipliers = 0.5 + (stability_scores_array * 0.5)  # Range: 0.5-1.0
        combined_scores_array = base_scores * stability_multipliers
        
        combined_scores = dict(zip(X.columns, combined_scores_array))

        tprint(f"📊 Non-linear combined scores range: [{np.min(combined_scores_array):.6f}, {np.max(combined_scores_array):.6f}]")
        tprint(f"📊 Combined scores mean: {np.mean(combined_scores_array):.6f}")

        # Analyze stability features
        high_stability_mask = stability_scores_array > 0.8
        low_stability_mask = stability_scores_array < 0.3
        high_stability_features = X.columns[high_stability_mask].tolist()
        low_stability_features = X.columns[low_stability_mask].tolist()

        if len(high_stability_features) > 0:
            tprint(f"🎯 High stability features: {len(high_stability_features)} (consistently valuable)")

        if len(low_stability_features) > 0:
            tprint(f"⚠️ Low stability features: {len(low_stability_features)} (context-dependent)")

        # Apply early termination if enabled and we have many features
        if self.config.enable_early_termination and len(X.columns) > actual_target * 2:
            tprint("🗑️ Applying early termination...")
            X = self._apply_early_termination(X, combined_scores)
            # Update arrays to match remaining features
            remaining_indices = [X.columns.get_loc(f) for f in X.columns if f in combined_scores]
            combined_scores_array = combined_scores_array[remaining_indices]

        # Apply final RFE if enabled
        if self.config.enable_rfe and len(X.columns) > actual_target:
            tprint(f"🔄 Applying final RFE to reduce from {len(X.columns)} to {actual_target} features")
            rfe_features = self._recursive_feature_elimination(X, y)
            X = X[rfe_features]
            tprint(f"✅ Final RFE completed: {len(X.columns)} features remaining")

        # Select top features using vectorized operations
        tprint("🏆 Selecting final features using vectorized operations...")
        sorted_indices = np.argsort(combined_scores_array)[::-1]
        selected_indices = sorted_indices[:actual_target]
        selected_features = [X.columns[i] for i in selected_indices]

        tprint(f"🏆 Final Stage 3 selection: {len(selected_features)} features")
        tprint(f"📈 Top combined score: {combined_scores_array[sorted_indices[0]]:.6f}" if len(sorted_indices) > 0 else "N/A")

        # Calculate comprehensive scores with vectorized operations
        selected_combined_scores = combined_scores_array[selected_indices]
        selected_stability = stability_scores_array[selected_indices]
        
        scores = {
            'combined_importance_score': np.mean(selected_combined_scores),
            'model_cv_agreement': self._calculate_agreement(model_importance, cv_scores),
            'final_stability': np.std(selected_combined_scores),
            'model_type': 'vectorized_combined',
            'mutual_information_avg': np.mean(mi_scores_array),
            'stability_avg': np.mean(selected_stability),
            'high_stability_features': len(high_stability_features),
            'low_stability_features': len(low_stability_features),
            'combined_std': np.std(selected_combined_scores),
            'combined_range': np.max(selected_combined_scores) - np.min(selected_combined_scores)
        }

        tprint(f"📊 Final scores - Combined: {scores['combined_importance_score']:.4f}, Stability: {scores['final_stability']:.4f}")
        tprint(f"📊 Selection quality: {len(selected_features)/len(X.columns):.2%}")
        tprint(f"✅ Stage 3 completed: {len(selected_features)} features selected")
        return selected_features, scores

    def _cross_validate_feature_importance_optimized(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Cross-validation feature importance using optimized models."""
        cv_scores = {}

        try:
            # Use StratifiedKFold for classification, regular KFold for regression
            if self._is_classification(y):
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            else:
                cv = self.config.cv_folds  # Let sklearn choose the CV strategy

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train optimized model on fold
                model = self._train_optimized_model(X_train, y_train)

                # Get feature importance from this fold
                if hasattr(model, 'feature_importances_'):
                    fold_importance = dict(zip(X.columns, model.feature_importances_))
                else:
                    # Fallback to variance-based importance
                    fold_importance = dict(zip(X.columns, X_train.var().values))

                # Accumulate scores
                for feature, importance in fold_importance.items():
                    if feature not in cv_scores:
                        cv_scores[feature] = []
                    cv_scores[feature].append(importance)

            # Average across folds
            for feature in cv_scores:
                cv_scores[feature] = np.mean(cv_scores[feature])

        except Exception as e:
            self.logger.warning(f"⚠️ Optimized CV feature importance failed: {e}")
            # Fallback to simple variance-based scores
            cv_scores = dict(zip(X.columns, X.var().values))

        return cv_scores
    
    def _shap_based_selection(self, X: pd.DataFrame, y: pd.Series, target_count: int) -> Tuple[List[str], Dict[str, float]]:
        """SHAP-based feature selection."""

        # Sample data for SHAP analysis
        sample_size = min(self.config.shap_sample_size, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        tprint(f"🔮 SHAP analysis on sample of {sample_size} observations")

        # Train model
        rf_model = self._train_random_forest(X_sample, y_sample)
        tprint("🌳 RandomForest trained for SHAP analysis")

        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        tprint("✨ SHAP values computed")

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):  # Classification
            shap_importance = np.mean(np.abs(shap_values), axis=(0, 1))
        else:  # Regression
            shap_importance = np.mean(np.abs(shap_values), axis=0)

        # Create feature importance dictionary
        shap_importance_dict = dict(zip(X.columns, shap_importance))

        # Select top features
        sorted_features = sorted(shap_importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:target_count]]
        tprint(f"✅ SHAP selected top {len(selected_features)} features")

        # Calculate scores
        scores = {
            'shap_importance_score': np.mean(shap_importance),
            'shap_variance': np.var(shap_importance),
            'selection_confidence': len(selected_features) / len(X.columns)
        }

        tprint(
            f"📊 SHAP scores — mean importance: {scores['shap_importance_score']:.6f},"
            f" variance: {scores['shap_variance']:.6f}"
        )

        return selected_features, scores

    def _enhanced_rf_selection(self, X: pd.DataFrame, y: pd.Series, target_count: int) -> Tuple[List[str], Dict[str, float]]:
        """Enhanced RandomForest selection with multiple criteria."""

        # Train multiple RandomForest models with different parameters
        models = []
        tprint("🌳 Starting enhanced RandomForest ensemble selection")
        for n_est in [50, 100, 150]:
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state + n_est
            )
            model.fit(X, y)
            models.append(model)
            tprint(f"✅ Trained RandomForest with {n_est} trees")

        # Average feature importance across models
        avg_importance = np.zeros(len(X.columns))
        for model in models:
            avg_importance += model.feature_importances_
        avg_importance /= len(models)
        tprint("📊 Averaged feature importance across ensemble")

        # Create feature importance dictionary
        importance_dict = dict(zip(X.columns, avg_importance))

        # Select top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:target_count]]
        tprint(f"🏆 Enhanced RF selected {len(selected_features)} features")

        # Calculate scores
        scores = {
            'enhanced_rf_score': np.mean(avg_importance),
            'model_agreement': 1 - np.std(avg_importance) / np.mean(avg_importance),
            'selection_quality': len(selected_features) / len(X.columns)
        }

        tprint(
            f"📈 Enhanced RF scores — mean: {scores['enhanced_rf_score']:.6f},"
            f" agreement: {scores['model_agreement']:.6f}"
        )

        return selected_features, scores
    
    def _cross_validate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Cross-validate feature importance using multiple folds."""
        
        cv_scores = {feature: 0.0 for feature in X.columns}
        
        # Use StratifiedKFold for classification, regular KFold for regression
        if self._is_classification(y):
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.rf_random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.rf_random_state)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on fold
            model = self._train_random_forest(X_train, y_train)
            
            # Get feature importance
            fold_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Accumulate scores
            for feature, importance in fold_importance.items():
                cv_scores[feature] += importance
        
        # Average across folds
        for feature in cv_scores:
            cv_scores[feature] /= self.config.cv_folds
        
        return cv_scores
    
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """Train RandomForest model."""

        if self._is_classification(y):
            model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state
            )
            tprint(f"🌳 Training RandomForestClassifier with {self.config.rf_n_estimators} trees")
        else:
            model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state
            )
            tprint(f"🌳 Training RandomForestRegressor with {self.config.rf_n_estimators} trees")

        model.fit(X, y)
        tprint("✅ RandomForest training complete")
        return model
    
    def _is_classification(self, y: pd.Series) -> bool:
        """Determine if target is classification or regression."""
        # Simple heuristic: if target has few unique values, treat as classification
        unique_values = len(y.unique())
        return unique_values <= 10 or y.dtype == 'category'
    
    def _calculate_agreement(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> float:
        """Calculate agreement between two scoring methods."""
        common_features = set(scores1.keys()) & set(scores2.keys())
        if not common_features:
            return 0.0
        
        # Calculate correlation between scores
        scores1_values = [scores1[f] for f in common_features]
        scores2_values = [scores2[f] for f in common_features]
        
        correlation = np.corrcoef(scores1_values, scores2_values)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _compile_results(self,
                        X: pd.DataFrame,
                        y: pd.Series,
                        stage_1_features: List[str],
                        stage_2_features: List[str],
                        stage_3_features: List[str],
                        stage_1_scores: Dict[str, Any],
                        stage_2_scores: Dict[str, Any],
                        stage_3_scores: Dict[str, Any]):
        """Compile final results."""

        # Store feature lists
        self.results.stage_1_features = stage_1_features
        self.results.stage_2_features = stage_2_features
        self.results.stage_3_features = stage_3_features
        self.results.final_features = stage_3_features

        # Store scores
        self.results.stage_1_scores = stage_1_scores
        self.results.stage_2_scores = stage_2_scores
        self.results.stage_3_scores = stage_3_scores

        # Calculate final model performance
        final_model = self._train_random_forest(X[stage_3_features], y)
        try:
            from src.utils.ml_common.validation.unified_cv import perform_cross_validation as unified_perform_cv
            cv_res = unified_perform_cv(final_model, X[stage_3_features].values, y.values if hasattr(y, 'values') else y, cv_folds=self.config.cv_folds, scoring=self.config.cv_scoring)
            cv_scores = np.array(cv_res.get('scores', []) or [])
        except Exception:
            cv_scores = np.array([])

        cv_mean = float(np.nanmean(cv_scores)) if cv_scores.size else float('nan')
        cv_std = float(np.nanstd(cv_scores)) if cv_scores.size else float('nan')
        metric_name, metric_value = self._calculate_model_metric(final_model, X[stage_3_features], y)

        final_scores: Dict[str, Any] = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_metric': self.config.cv_scoring
        }

        if metric_value is not None:
            final_scores[metric_name] = metric_value

        self.results.final_scores = final_scores

        # Store feature counts
        self.results.feature_counts = {
            'initial': len(X.columns),
            'stage_1': len(stage_1_features),
            'stage_2': len(stage_2_features),
            'stage_3': len(stage_3_features),
            'final': len(stage_3_features)
        }
        
        # Store model performance
        self.results.model_performance = {
            'final_model': final_model,
            'cv_scores': cv_scores.tolist(),
            'feature_importance': dict(zip(stage_3_features, final_model.feature_importances_)),
            'evaluation_metric': metric_name,
            'evaluation_score': metric_value
        }

        self.results.polarity_adjustments = getattr(self, 'feature_polarity_adjustments', {})
        self.results.sign_stability = getattr(self, 'feature_sign_stability', {})

    def _calculate_model_metric(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Optional[float]]:
        """Calculate the final evaluation metric for the trained model."""

        if self._is_classification(y):
            return self._calculate_classification_metric(model, X, y)

        try:
            return 'r2', float(model.score(X, y))
        except Exception:
            return 'r2', None

    def _calculate_classification_metric(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Optional[float]]:
        """Calculate imbalance-aware classification metrics."""

        y_array = y.values if hasattr(y, 'values') else np.asarray(y)
        unique_classes = np.unique(y_array)

        if unique_classes.size <= 1:
            return 'average_precision', None

        if unique_classes.size <= 2:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        positive_scores = proba[:, 1]
                    else:
                        positive_scores = proba.ravel()
                    return 'average_precision', float(average_precision_score(y_array, positive_scores))
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                    return 'average_precision', float(average_precision_score(y_array, scores))
            except Exception:
                pass

        try:
            predictions = model.predict(X)
            return 'balanced_accuracy', float(balanced_accuracy_score(y_array, predictions))
        except Exception:
            return 'balanced_accuracy', None

    def _handle_insufficient_features(self, X: pd.DataFrame, y: pd.Series) -> FeatureSelectionResult:
        """Handle case where we don't have enough features."""
        
        self.logger.warning("⚠️ Insufficient features for multi-stage selection")
        
        # Use all available features
        self.results.final_features = X.columns.tolist()
        self.results.stage_1_features = X.columns.tolist()
        self.results.stage_2_features = X.columns.tolist()
        self.results.stage_3_features = X.columns.tolist()
        
        # Train final model
        final_model = self._train_random_forest(X, y)

        metric_name, metric_value = self._calculate_model_metric(final_model, X, y)
        final_scores: Dict[str, Any] = {
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'cv_metric': self.config.cv_scoring
        }
        if metric_value is not None:
            final_scores[metric_name] = metric_value

        self.results.final_scores = final_scores

        self.results.feature_counts = {
            'initial': len(X.columns),
            'stage_1': len(X.columns),
            'stage_2': len(X.columns),
            'stage_3': len(X.columns),
            'final': len(X.columns)
        }

        self.results.polarity_adjustments = getattr(self, 'feature_polarity_adjustments', {})
        self.results.sign_stability = getattr(self, 'feature_sign_stability', {})

        return self.results
    
    def _save_analysis(self):
        """Save analysis results."""
        try:
            from datetime import datetime

            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save results summary with proper outcomes naming convention
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"market_analysis_feature_selection_outcome_{timestamp}.json"
            
            # Convert results to serializable format
            results_dict = {
                'feature_counts': self.results.feature_counts,
                'final_features': self.results.final_features,
                'scores': {
                    'stage_1': self.results.stage_1_scores,
                    'stage_2': self.results.stage_2_scores,
                    'stage_3': self.results.stage_3_scores,
                    'final': self.results.final_scores
                },
                'entropy_filter': {
                    'removed_features': self.results.entropy_removed_features,
                    'stability_scores': self.results.entropy_stability,
                    'variance': self.results.entropy_variance,
                    'config': {
                        'num_slices': self.config.entropy_num_slices,
                        'min_slice_size': self.config.entropy_min_slice_size,
                        'variance_threshold': self.config.entropy_variance_threshold,
                        'max_bins': self.config.entropy_max_bins,
                        'min_unique_values': self.config.entropy_min_unique_values,
                    },
                },
                'polarity_adjustments': getattr(self.results, 'polarity_adjustments', {}),
                'sign_stability': getattr(self.results, 'sign_stability', {}),
                'selection_time': self.results.selection_time,
                'config': {
                    'initial_features': self.config.initial_features,
                    'stage_1_target': self.config.stage_1_target,
                    'stage_2_target': self.config.stage_2_target,
                    'stage_3_target': self.config.stage_3_target,
                    'rf_n_estimators': self.config.rf_n_estimators,
                    'cv_folds': self.config.cv_folds
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            self.logger.info(f"💾 Analysis results saved to {results_file}")
            
            # Save final model if requested
            if self.config.save_models and hasattr(self.results, 'model_performance'):
                model_file = output_dir / f"market_analysis_feature_selection_model_{timestamp}.joblib"
                joblib.dump(self.results.model_performance['final_model'], model_file)
                self.logger.info(f"💾 Final model saved to {model_file}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save analysis: {e}")

# Convenience functions
def run_final_feature_selection(X: pd.DataFrame,
                               y: pd.Series,
                               config: Optional[FeatureSelectionConfig] = None) -> FeatureSelectionResult:
    """Run final feature selection pipeline."""
    tprint("🚀 Running final feature selection helper")
    selector = MultiStageFeatureSelector(config)
    tprint("📊 Executing multi-stage selector")
    return selector.select_features(X, y)

def get_final_features(X: pd.DataFrame,
                      y: pd.Series,
                      target_count: int = 60,
                      config: Optional[FeatureSelectionConfig] = None) -> List[str]:
    """Get final selected features."""
    if config is None:
        config = FeatureSelectionConfig()
        config.stage_3_target = target_count
        tprint(f"🎯 Configuring stage 3 target for final features: {target_count}")

    result = run_final_feature_selection(X, y, config)
    tprint(f"✅ Final feature count: {len(result.final_features)}")
    return result.final_features
