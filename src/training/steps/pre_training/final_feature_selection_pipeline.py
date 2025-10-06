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
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

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
    stage_1_scores: Dict[str, float] = field(default_factory=dict)
    stage_2_scores: Dict[str, float] = field(default_factory=dict)
    stage_3_scores: Dict[str, float] = field(default_factory=dict)
    final_scores: Dict[str, float] = field(default_factory=dict)
    
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

class MultiStageFeatureSelector:
    """Multi-stage feature selection using RandomForest and SHAP."""

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

        # Set model-specific parameters
        self._set_model_specific_parameters()

        # Initialize directional results for separate long/short feature sets
        if self.config.separate_directional_features:
            self.long_results = FeatureSelectionResult()
            self.short_results = FeatureSelectionResult()

        self.logger.info("🚀 MultiStageFeatureSelector initialized")
        self.logger.info(f"🎯 Model Type: {self.config.model_type}")
        self.logger.info(f"📊 Feature Range: {self.config.min_features}-{self.config.max_features} (target: {self.config.target_features})")
        self.logger.info(f"🎯 Direction Mode: {self.config.direction_mode}")
        self.logger.info(f"🎯 Separate Features: {self.config.separate_directional_features}")
        self.logger.info(f"⚡ Early Termination: {self.config.enable_early_termination}")
        self.logger.info(f"⚡ LightGBM: {LIGHTGBM_AVAILABLE}")
        self.logger.info(f"⚡ RFE: {self.config.enable_rfe}")
        self.logger.info(f"⚡ Mutual Information: {self.config.enable_mutual_information}")
        self.logger.info(f"⚡ Chunked Processing: {self.config.enable_chunked_processing}")

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
                'cv_scoring': 'accuracy'  # For regime classification
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
            self.logger.info(f"✅ Applied {self.config.model_type} specific parameters")
    
    def select_features(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       feature_names: Optional[List[str]] = None) -> FeatureSelectionResult:
        """Perform multi-stage feature selection with directional support."""

        start_time = time.time()
        self.logger.info("🔍 Starting multi-stage feature selection")

        # Apply execution mode data windowing if configured
        X_processed = X.copy()
        if self.execution_mode_config:
            window_days = self.execution_mode_config.get('window_days', 1460)
            if len(X_processed) > window_days:
                X_processed = X_processed.tail(window_days).copy()
                self.logger.info(f"📊 Applied execution mode window: using last {window_days} samples for feature selection")

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

            self.logger.info(f"📊 Using execution mode stage targets: {stage_targets}")

        # Determine which directions to process based on mode
        directions_to_process = self._get_directions_to_process()

        # Handle separate directional feature selection
        if self.config.separate_directional_features and self.config.direction_mode in ['both', 'long_only', 'short_only']:
            return self._select_directional_features(X_processed, y, feature_names, directions_to_process, stage_targets)

        # Original unified feature selection (for backward compatibility)
        return self._select_unified_features(X_processed, y, feature_names, stage_targets)

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

        # Process each direction separately
        for direction in directions:
            self.logger.info(f"🔄 Processing {direction} direction features")

            # Filter features for this direction
            direction_features = self._filter_direction_features(X, direction)

            if len(direction_features.columns) < stage_targets[-1]:
                self.logger.warning(f"⚠️ {direction} direction has only {len(direction_features.columns)} features, less than target {stage_targets[-1]}")
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

        return combined_result

    def _select_unified_features(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]], stage_targets: Tuple[int, ...]) -> FeatureSelectionResult:
        """Original unified feature selection logic."""

        # Validate inputs
        if len(X.columns) < stage_targets[-1]:  # Check against final target
            self.logger.warning(f"⚠️ Input has only {len(X.columns)} features, less than target {stage_targets[-1]}")
            return self._handle_insufficient_features(X, y)

        # Stage 0: Initial feature preparation
        self.logger.info("📊 Stage 0: Initial feature preparation")
        prepared_features = self._prepare_initial_features(X, y, feature_names)

        # Stage 1: Initial → Stage 1 target features
        stage_1_target = stage_targets[1] if len(stage_targets) > 1 else 100
        self.logger.info(f"📊 Stage 1: Reducing to {stage_1_target} features")
        stage_1_features, stage_1_scores = self._stage_1_selection(prepared_features, y, target_count=stage_1_target)

        # Stage 2: Stage 1 → Stage 2 target features
        stage_2_target = stage_targets[2] if len(stage_targets) > 2 else 80
        self.logger.info(f"📊 Stage 2: Reducing to {stage_2_target} features")
        stage_2_features, stage_2_scores = self._stage_2_selection(prepared_features[stage_1_features], y, target_count=stage_2_target)

        # Stage 3: Stage 2 → Stage 3 target features
        stage_3_target = stage_targets[3] if len(stage_targets) > 3 else 60
        self.logger.info(f"📊 Stage 3: Reducing to {stage_3_target} features")
        stage_3_features, stage_3_scores = self._stage_3_selection(prepared_features[stage_1_features][stage_2_features], y, target_count=stage_3_target)

        # Compile final results
        self._compile_results(
            prepared_features, y, stage_1_features, stage_2_features, stage_3_features,
            stage_1_scores, stage_2_scores, stage_3_scores
        )

        # Save results
        if self.config.save_analysis:
            self._save_analysis()

        return self.results
    
    def _prepare_initial_features(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare initial features for selection."""
        
        # Handle feature names
        if feature_names is not None:
            X = X[feature_names] if len(feature_names) <= len(X.columns) else X
        
        # Remove low variance features
        variance_threshold = self.config.min_variance_threshold
        low_variance_mask = X.var() < variance_threshold
        low_variance_features = X.columns[low_variance_mask].tolist()
        
        if low_variance_features:
            self.logger.info(f"🗑️ Removing {len(low_variance_features)} low variance features")
            X = X.drop(columns=low_variance_features)
        
        # Remove highly correlated features using optimized correlation computation
        correlation_threshold = self.config.min_correlation_threshold
        high_corr_features = self._find_highly_correlated_features(X, correlation_threshold)

        if high_corr_features:
            self.logger.info(f"🗑️ Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
        
        # Select top features if we have too many
        if len(X.columns) > self.config.initial_features:
            self.logger.info(f"📊 Selecting top {self.config.initial_features} features initially")
            # Use simple variance-based selection for initial filtering
            feature_variance = X.var().sort_values(ascending=False)
            top_features = feature_variance.head(self.config.initial_features).index.tolist()
            X = X[top_features]
        
        self.logger.info(f"✅ Prepared {len(X.columns)} features for selection")
        return X
    
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

        # Determine if classification or regression
        if self._is_classification(y):
            objective = 'binary' if len(y.unique()) == 2 else 'multiclass'
            params = self.config.lightgbm_params.copy()
            params['objective'] = objective
            params['num_class'] = len(y.unique()) if objective == 'multiclass' else 1

            model = lgb.LGBMClassifier(**params)
        else:
            params = self.config.lightgbm_params.copy()
            params['objective'] = 'regression'

            model = lgb.LGBMRegressor(**params)

        # Train model
        model.fit(X, y)

        return model

    def _train_optimized_model(self, X: pd.DataFrame, y: pd.Series):
        """Train either LightGBM or RandomForest based on availability and performance."""
        try:
            if LIGHTGBM_AVAILABLE and len(X.columns) > 50:
                self.logger.info("🚀 Using LightGBM for faster training")
                return self._train_lightgbm_model(X, y)
            else:
                self.logger.info("📊 Using RandomForest (LightGBM not available or dataset too small)")
                return self._train_random_forest(X, y)
        except Exception as e:
            self.logger.warning(f"⚠️ LightGBM training failed, falling back to RandomForest: {e}")
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
            return X_chunk.columns.tolist()

        # Train model and get importance
        try:
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

            return selected

        except Exception as e:
            self.logger.warning(f"⚠️ Chunk selection failed: {e}")
            # Fallback to simple variance-based selection
            variances = X_chunk.var().sort_values(ascending=False)
            target_count = min(len(X_chunk.columns) // 2, 10)
            return variances.head(target_count).index.tolist()
    
    def _stage_1_selection(self, X: pd.DataFrame, y: pd.Series, target_count: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """Stage 1: Initial → target features using optimized model selection."""

        tprint("🚀 Starting Stage 1 Feature Selection")
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

        # Train optimized model (LightGBM or RandomForest)
        tprint("🤖 Training optimized model...")
        model = self._train_optimized_model(X, y)
        model_type = type(model).__name__
        tprint(f"✅ Model trained: {model_type}")

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            tprint(f"📈 Feature importance range: [{min(feature_importance.values()):.6f}, {max(feature_importance.values()):.6f}]")
        else:
            # Fallback to simple variance-based importance
            feature_importance = dict(zip(X.columns, X.var().values))
            tprint("📊 Using variance-based importance (fallback)")

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

        # Select top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:actual_target]]

        tprint(f"🏆 Final selection: {len(selected_features)} features")
        tprint(f"📈 Top feature importance: {sorted_features[0][1]:.6f}" if sorted_features else "N/A")

        # Calculate enhanced scores including mutual information
        scores = {
            'model_importance_score': np.mean(list(feature_importance.values())),
            'feature_variance': X[selected_features].var().mean(),
            'selection_quality': len(selected_features) / len(X.columns),
            'model_type': model_type
        }

        # Add mutual information scores if enabled
        if self.config.enable_mutual_information:
            tprint("🔗 Calculating mutual information...")
            mi_scores = self._calculate_mutual_information_correlation(X[selected_features], y)
            scores['mutual_information'] = np.mean(list(mi_scores.values())) if mi_scores else 0.0
            tprint(f"📊 Mutual information average: {scores['mutual_information']:.4f}")

        tprint(f"✅ Stage 1 completed: {len(selected_features)}/{len(X.columns)} features selected")
        return selected_features, scores
    
    def _stage_2_selection(self, X: pd.DataFrame, y: pd.Series, target_count: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """Stage 2: Previous → target features using enhanced selection methods."""

        # Use provided target count or fall back to config
        actual_target = target_count or self.config.stage_2_target

        tprint("🚀 Starting Stage 2 Feature Selection")
        tprint(f"📊 Input: {len(X)} samples, {len(X.columns)} features")

        # Use provided target count or fall back to config
        actual_target = target_count or self.config.stage_2_target
        tprint(f"🎯 Target features: {actual_target}")

        # Train optimized model for this stage
        tprint("🤖 Training optimized model for Stage 2...")
        model = self._train_optimized_model(X, y)
        model_type = type(model).__name__
        tprint(f"✅ Model trained: {model_type}")

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            tprint(f"📈 Feature importance range: [{min(feature_importance.values()):.6f}, {max(feature_importance.values()):.6f}]")
        else:
            feature_importance = dict(zip(X.columns, X.var().values))
            tprint("📊 Using variance-based importance (fallback)")

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

        # Select top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:actual_target]]

        tprint(f"🏆 Stage 2 selection: {len(selected_features)} features")

        # Calculate enhanced scores
        scores = {
            'model_importance_score': np.mean(list(feature_importance.values())),
            'feature_variance': X[selected_features].var().mean(),
            'selection_quality': len(selected_features) / len(X.columns),
            'model_type': model_type
        }

        # Add mutual information scores if enabled
        if self.config.enable_mutual_information:
            tprint("🔗 Calculating mutual information...")
            mi_scores = self._calculate_mutual_information_correlation(X[selected_features], y)
            scores['mutual_information'] = np.mean(list(mi_scores.values())) if mi_scores else 0.0
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
        return selected_features, scores

    def _stage_3_selection(self, X: pd.DataFrame, y: pd.Series, target_count: Optional[int] = None) -> Tuple[List[str], Dict[str, float]]:
        """Stage 3: Previous → target features using combined importance and cross-validation with optimizations."""

        # Use provided target count or fall back to config
        actual_target = target_count or self.config.stage_3_target

        tprint("🚀 Starting Stage 3 Feature Selection")
        tprint(f"📊 Input: {len(X)} samples, {len(X.columns)} features")


        # Use provided target count or fall back to config
        actual_target = target_count or self.config.stage_3_target
        tprint(f"🎯 Target features: {actual_target}")

        # Train optimized model for this stage
        tprint("🤖 Training optimized model for Stage 3...")
        model = self._train_optimized_model(X, y)
        model_type = type(model).__name__
        tprint(f"✅ Model trained: {model_type}")

        model_importance = dict(zip(X.columns, model.feature_importances_))

        # Cross-validation based selection using optimized model
        tprint("🔄 Performing cross-validation feature importance...")
        cv_scores = self._cross_validate_feature_importance_optimized(X, y)
        tprint(f"✅ CV completed for {len(cv_scores)} features")

        # Calculate non-linear quality metrics
        mi_scores: Dict[str, float] = {}
        if self.config.enable_mutual_information:
            tprint("🔗 Calculating mutual information...")
            mi_scores = self._calculate_mutual_information_correlation(X, y)
            tprint(f"✅ MI calculated for {len(mi_scores)} features")

        # Calculate feature stability across time periods
        tprint("🛡️ Analyzing feature stability...")
        stability_scores = self._calculate_feature_stability_score(X, y)
        tprint(f"✅ Stability analyzed for {len(stability_scores)} features")

        # Combine importance scores with non-linear awareness (model + CV + MI + stability)
        tprint("⚖️ Combining importance scores with non-linear awareness...")
        combined_scores: Dict[str, float] = {}
        for feature in X.columns:
            model_score = model_importance.get(feature, 0)
            cv_score = cv_scores.get(feature, 0)
            mi_score = mi_scores.get(feature, 0)
            stability_score = stability_scores.get(feature, 0.5)  # Default stability if not calculated

            base_score = (model_score * 0.3 + cv_score * 0.3 + mi_score * 0.2)
            stability_multiplier = 0.5 + (stability_score * 0.5)  # Range: 0.5-1.0
            combined_scores[feature] = base_score * stability_multiplier

        if combined_scores:
            tprint(
                f"📊 Non-linear combined scores range: "
                f"[{min(combined_scores.values()):.6f}, {max(combined_scores.values()):.6f}]"
            )
        else:
            tprint("⚠️ No combined scores calculated")

        high_stability_features = [f for f, score in stability_scores.items() if score > 0.8]
        if high_stability_features:
            tprint(f"🎯 High stability features: {len(high_stability_features)} (consistently valuable)")

        low_stability_features = [f for f, score in stability_scores.items() if score < 0.3]
        if low_stability_features:
            tprint(f"⚠️ Low stability features: {len(low_stability_features)} (context-dependent)")

        # Apply early termination if enabled and we have many features
        if self.config.enable_early_termination and len(X.columns) > actual_target * 2:
            tprint("🗑️ Applying early termination...")
            X = self._apply_early_termination(X, combined_scores)
            combined_scores = {f: combined_scores.get(f, 0) for f in X.columns}

        # Apply final RFE if enabled
        if self.config.enable_rfe and len(X.columns) > actual_target:
            tprint(f"🔄 Applying final RFE to reduce from {len(X.columns)} to {actual_target} features")
            rfe_features = self._recursive_feature_elimination(X, y)
            X = X[rfe_features]
            combined_scores = {f: combined_scores.get(f, 0) for f in X.columns}
            tprint(f"✅ Final RFE completed: {len(X.columns)} features remaining")

        # Select top features
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:actual_target]]

        tprint(f"🏆 Final Stage 3 selection: {len(selected_features)} features")
        tprint(f"📈 Top combined score: {sorted_features[0][1]:.6f}" if sorted_features else "N/A")

        # Calculate comprehensive scores with non-linear awareness
        scores = {
            'combined_importance_score': np.mean(list(combined_scores.values())) if combined_scores else 0.0,
            'model_cv_agreement': self._calculate_agreement(model_importance, cv_scores),
            'final_stability': np.std(list(combined_scores.values())) if combined_scores else 0.0,
            'model_type': model_type,
            'mutual_information_avg': np.mean(list(mi_scores.values())) if mi_scores else 0.0,
            'stability_avg': np.mean(list(stability_scores.values())) if stability_scores else 0.0,
            'high_stability_features': len(high_stability_features),
            'low_stability_features': len(low_stability_features)
        }

        tprint(f"📊 Final scores - Combined: {scores['combined_importance_score']:.4f}, Stability: {scores['final_stability']:.4f}")
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
        
        # Train model
        rf_model = self._train_random_forest(X_sample, y_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        
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
        
        # Calculate scores
        scores = {
            'shap_importance_score': np.mean(shap_importance),
            'shap_variance': np.var(shap_importance),
            'selection_confidence': len(selected_features) / len(X.columns)
        }
        
        return selected_features, scores
    
    def _enhanced_rf_selection(self, X: pd.DataFrame, y: pd.Series, target_count: int) -> Tuple[List[str], Dict[str, float]]:
        """Enhanced RandomForest selection with multiple criteria."""
        
        # Train multiple RandomForest models with different parameters
        models = []
        for n_est in [50, 100, 150]:
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state + n_est
            )
            model.fit(X, y)
            models.append(model)
        
        # Average feature importance across models
        avg_importance = np.zeros(len(X.columns))
        for model in models:
            avg_importance += model.feature_importances_
        avg_importance /= len(models)
        
        # Create feature importance dictionary
        importance_dict = dict(zip(X.columns, avg_importance))
        
        # Select top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:target_count]]
        
        # Calculate scores
        scores = {
            'enhanced_rf_score': np.mean(avg_importance),
            'model_agreement': 1 - np.std(avg_importance) / np.mean(avg_importance),
            'selection_quality': len(selected_features) / len(X.columns)
        }
        
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
        else:
            model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state
            )
        
        model.fit(X, y)
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
                        stage_1_scores: Dict[str, float],
                        stage_2_scores: Dict[str, float],
                        stage_3_scores: Dict[str, float]):
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
        
        self.results.final_scores = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_score': final_model.score(X[stage_3_features], y)
        }
        
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
            'feature_importance': dict(zip(stage_3_features, final_model.feature_importances_))
        }
    
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
        
        self.results.final_scores = {
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'model_score': final_model.score(X, y)
        }
        
        self.results.feature_counts = {
            'initial': len(X.columns),
            'stage_1': len(X.columns),
            'stage_2': len(X.columns),
            'stage_3': len(X.columns),
            'final': len(X.columns)
        }
        
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
    selector = MultiStageFeatureSelector(config)
    return selector.select_features(X, y)

def get_final_features(X: pd.DataFrame, 
                      y: pd.Series,
                      target_count: int = 60,
                      config: Optional[FeatureSelectionConfig] = None) -> List[str]:
    """Get final selected features."""
    if config is None:
        config = FeatureSelectionConfig()
        config.stage_3_target = target_count
    
    result = run_final_feature_selection(X, y, config)
    return result.final_features