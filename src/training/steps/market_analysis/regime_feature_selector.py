"""
Enhanced Regime Feature Selector

This module provides an advanced regime feature selection system that integrates:
- TreeSHAP feature selection as the base method
- tprint utilities for logging and data preview
- VectorBTRollingOptimizer and UnifiedVectorizationManager for vectorized computations
- Hardware optimizations for M1 systems
- ML common utilities for HPO, SHAP/LIME, time series validation, and data leakage prevention
- BaseStep integration for autonomous pipeline execution
- Artifact management for data persistence and retrieval

Author: AI Assistant
Date: 2024
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Core imports
from src.training.steps.base_step import BaseStep

# tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_data_preview, tprint_data_format, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0]}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0]}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0]}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0]}")
    def tprint_data_preview(*args, **kwargs): print(f"DATA PREVIEW: {args[0]}")
    def tprint_data_format(*args, **kwargs): print(f"DATA FORMAT: {args[0]}")
    def tprint_performance(*args, **kwargs): print(f"PERFORMANCE: {args[0]}")

# VectorBT and optimization imports
try:
    from src.vectorbt import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    )
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    VECTORBT_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

# UnifiedVectorizationManager imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager,
        OperationType, OptimizationStrategy
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None

# Hardware optimization imports
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel,
        get_unified_hardware_manager
    )
    from src.utils.hardware.m1_memory_optimizer import (
        M1MemoryOptimizer, get_m1_memory_optimizer
    )
    from src.utils.hardware.m1_cpu_optimizer import (
        M1CPUOptimizer, get_m1_cpu_optimizer
    )
    from src.utils.hardware.m1_gpu_utils import (
        M1GPUManager, get_m1_gpu_manager
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    UnifiedHardwareManager = None
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    M1GPUManager = None

# ML common utilities imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig as HPOConfig
    )
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEIntegration, SHAPLIMEExplainer, ExplanationConfig
    )
    from src.utils.ml_common.validation.temporal_cross_validation import (
        temporal_cross_validation
    )
    from src.utils.ml_common.validation.data_leakage_detector import (
        DataLeakageDetector
    )
    from src.utils.ml_common.validation.purged_kfold import PurgedKFold
    from src.utils.ml_common.validation.lookahead_bias_detector import (
        LookaheadBiasDetector
    )
    from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import (
        OOFStackingEnsembleManager, OOFStackingConfig
    )
    from src.utils.ml_common.evaluation.unified_evaluator import (
        UnifiedEvaluator, EvaluationConfig
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    BayesianTPEOptimizer = None
    SHAPLIMEIntegration = None
    temporal_cross_validation = None
    DataLeakageDetector = None
    PurgedKFold = None
    LookaheadBiasDetector = None
    OOFStackingEnsembleManager = None
    UnifiedEvaluator = None

# TreeSHAP feature selector import
try:
    from src.training.steps.market_analysis.treeshap_feature_selector import (
        TreeSHAPFeatureSelector
    )
    TREESHAP_AVAILABLE = True
except ImportError:
    TREESHAP_AVAILABLE = False
    TreeSHAPFeatureSelector = None


@dataclass
class EnhancedRegimeFeatureSelectorConfig:
    """Configuration for the Enhanced Regime Feature Selector."""
    
    # Core feature selection parameters
    max_features: int = 50
    min_feature_importance: float = 0.01
    feature_selection_method: str = "treeshap"  # treeshap, mutual_info, rfe, etc.
    
    # TreeSHAP specific parameters - optimized for feature selection stability
    treeshap_params: Optional[Dict[str, Any]] = None
    
    # VectorBT optimization parameters
    use_vectorbt_optimization: bool = True
    vectorbt_rolling_window: int = 20
    
    # Hardware optimization parameters
    use_hardware_optimization: bool = True
    workload_type: WorkloadType = WorkloadType.ML_TRAINING
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # ML common parameters
    use_hpo: bool = True
    hpo_trials: int = 100
    use_explainability: bool = True
    use_temporal_validation: bool = True
    use_data_leakage_detection: bool = True
    
    # Performance parameters
    enable_caching: bool = True
    cache_size: int = 1000
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    
    # Logging parameters
    verbose: bool = True
    log_level: str = "INFO"


class EnhancedRegimeFeatureSelector(BaseStep):
    """
    Enhanced Regime Feature Selector that integrates multiple optimization strategies.
    
    This class provides comprehensive feature selection capabilities for regime-based
    trading strategies, combining TreeSHAP feature importance with advanced optimization
    techniques and hardware acceleration. Inherits from BaseStep for autonomous pipeline
    execution with artifact management.
    """
    
    def __init__(self, step_name: str = "regime_feature_selection"):
        """Initialize the Enhanced Regime Feature Selector."""
        super().__init__(step_name=step_name)
        
        # Initialize with default config - will be updated in execute()
        self.config = EnhancedRegimeFeatureSelectorConfig()
        
        # Set optimized TreeSHAP parameters if not provided
        if self.config.treeshap_params is None:
            self.config.treeshap_params = self._get_optimized_treeshap_params()
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_metrics = {}
        self.feature_importance_cache = {}
        
        tprint_success("Enhanced Regime Feature Selector initialized successfully")
    
    def _get_optimized_treeshap_params(self) -> Dict[str, Any]:
        """Get optimized TreeSHAP parameters for feature selection stability."""
        return {
            'n_estimators': 500,        # Higher for stable SHAP values
            'max_depth': 4,             # Shallower to prevent overfitting
            'learning_rate': 0.05,      # Lower for stable training
            'min_samples_split': 20,    # Prevent overfitting
            'min_samples_leaf': 10,     # Prevent overfitting
            'subsample': 0.8,           # Add regularization
            'random_state': 42,         # Reproducibility
            'n_jobs': -1,               # Parallel processing
            'verbose': 0                # Reduce noise
        }
    
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Initialize TreeSHAP feature selector - REQUIRED, no fallback
            if not TREESHAP_AVAILABLE or not TreeSHAPFeatureSelector:
                raise ImportError(
                    "TreeSHAP is required for regime feature selection. "
                    "Install with: pip install shap"
                )
            
            treeshap_config = self.config.treeshap_params or {}
            self.treeshap_selector = TreeSHAPFeatureSelector(treeshap_config)
            tprint_info("TreeSHAP feature selector initialized with optimized parameters")
            
            # Initialize VectorBT rolling optimizer
            if VECTORBT_OPTIMIZER_AVAILABLE and self.config.use_vectorbt_optimization:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                tprint_info("VectorBT rolling optimizer initialized")
            else:
                self.vectorbt_optimizer = None
                tprint_warning("VectorBT optimizer not available")
            
            # Initialize UnifiedVectorizationManager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_info("UnifiedVectorizationManager initialized")
            else:
                self.vectorization_manager = None
                tprint_warning("UnifiedVectorizationManager not available")
            
            # Initialize hardware optimizations
            if HARDWARE_OPTIMIZATION_AVAILABLE and self.config.use_hardware_optimization:
                self.hardware_manager = get_unified_hardware_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.gpu_manager = get_m1_gpu_manager()
                tprint_info("Hardware optimizations initialized")
            else:
                self.hardware_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.gpu_manager = None
                tprint_warning("Hardware optimizations not available")
            
            # Initialize ML common utilities
            if ML_COMMON_AVAILABLE:
                self._initialize_ml_common_utilities()
                tprint_info("ML common utilities initialized")
            else:
                tprint_warning("ML common utilities not available")
                
        except Exception as e:
            tprint_error(f"Error initializing components: {e}")
            self.logger.error(f"Component initialization failed: {e}")
    
    def _initialize_ml_common_utilities(self):
        """Initialize ML common utilities."""
        try:
            # Initialize HPO optimizer
            if self.config.use_hpo and BayesianTPEOptimizer:
                hpo_config = HPOConfig(
                    n_trials=self.config.hpo_trials,
                    timeout=3600,  # 1 hour timeout
                    random_state=42
                )
                self.hpo_optimizer = BayesianTPEOptimizer(hpo_config)
            else:
                self.hpo_optimizer = None
            
            # Initialize explainability tools
            if self.config.use_explainability and SHAPLIMEIntegration:
                self.explainability_tool = SHAPLIMEIntegration()
            else:
                self.explainability_tool = None
            
            # Initialize validation tools
            if self.config.use_temporal_validation:
                self.temporal_validator = temporal_cross_validation
            else:
                self.temporal_validator = None
            
            # Initialize data leakage detector
            if self.config.use_data_leakage_detection and DataLeakageDetector:
                self.leakage_detector = DataLeakageDetector()
            else:
                self.leakage_detector = None
            
            # Initialize ensemble manager
            if OOFStackingEnsembleManager:
                ensemble_config = OOFStackingConfig(
                    n_folds=5,
                    random_state=42
                )
                self.ensemble_manager = OOFStackingEnsembleManager(ensemble_config)
            else:
                self.ensemble_manager = None
            
            # Initialize evaluator
            if UnifiedEvaluator:
                eval_config = EvaluationConfig(
                    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
                    cross_validation=True
                )
                self.evaluator = UnifiedEvaluator(eval_config)
            else:
                self.evaluator = None
                
        except Exception as e:
            tprint_error(f"Error initializing ML common utilities: {e}")
            self.logger.error(f"ML common utilities initialization failed: {e}")
    
    def select_features(
        self,
        features_df: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        use_supervised: bool = True
    ) -> Dict[str, Any]:
        """
        Select features for regime detection/clustering.
        
        Supports both supervised (with regime labels) and unsupervised (without labels) modes.
        Unsupervised mode should be used before initial clustering to avoid circular dependency.
        
        Args:
            features_df: DataFrame containing features
            regime_labels: Optional regime labels (for supervised mode)
            feature_names: Optional list of feature names
            use_supervised: Whether to use supervised selection (requires regime_labels)
            
        Returns:
            Dictionary containing selected features and metadata
        """
        try:
            tprint_info(f"Starting {'supervised' if use_supervised and regime_labels is not None else 'unsupervised'} regime feature selection")
            tprint_data_preview(f"Features shape: {features_df.shape}")
            
            if use_supervised and regime_labels is not None:
                tprint_data_preview(f"Regime labels shape: {regime_labels.shape}")
                tprint_data_preview(f"Unique regimes: {regime_labels.nunique()}")
            else:
                tprint_info("Using unsupervised feature selection (no regime labels)")
            
            # Validate inputs based on mode
            if use_supervised:
                if regime_labels is None:
                    tprint_warning("Supervised mode requested but no regime labels provided, falling back to unsupervised")
                    use_supervised = False
                else:
                    # Data validation and preprocessing for supervised mode
                    features_df, regime_labels = self._validate_and_preprocess_data(features_df, regime_labels)
            else:
                # Basic validation for unsupervised mode
                if features_df.empty:
                    raise ValueError("Features DataFrame cannot be empty")
                
                # Handle missing values
                if features_df.isnull().any().any():
                    tprint_warning("Missing values detected, filling with median")
                    features_df = features_df.fillna(features_df.median())
            
            # Hardware optimization setup
            if self.hardware_manager:
                self._setup_hardware_optimization()
            
            # Feature selection pipeline
            if use_supervised and regime_labels is not None:
                # Supervised: use regime labels as target
                selection_results = self._run_regime_feature_selection_pipeline(
                    features_df, regime_labels, feature_names
                )
            else:
                # Unsupervised: use variance and correlation-based selection
                selection_results = self._run_unsupervised_feature_selection_pipeline(
                    features_df, feature_names
                )
            
            # Performance tracking
            self._track_performance(selection_results)
            
            tprint_success("Regime feature selection completed successfully")
            return selection_results
            
        except Exception as e:
            tprint_error(f"Feature selection failed: {e}")
            self.logger.error(f"Feature selection error: {e}")
            raise
    
    def _validate_and_preprocess_data(
        self,
        features_df: pd.DataFrame,
        regime_labels: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and preprocess input data."""
        try:
            # Basic validation
            if features_df.empty or regime_labels.empty:
                raise ValueError("Input data cannot be empty")
            
            if len(features_df) != len(regime_labels):
                raise ValueError("Features and regime labels must have the same length")
            
            # Handle missing values in features
            if features_df.isnull().any().any():
                tprint_warning("Missing values detected in features, filling with median")
                features_df = features_df.fillna(features_df.median())
            
            # Handle missing values in regime labels
            if regime_labels.isnull().any():
                tprint_warning("Missing regime labels detected, dropping rows")
                valid_indices = ~regime_labels.isnull()
                features_df = features_df[valid_indices]
                regime_labels = regime_labels[valid_indices]
            
            # Validate regime labels are numeric
            if not pd.api.types.is_numeric_dtype(regime_labels):
                tprint_warning("Converting regime labels to numeric")
                regime_labels = pd.to_numeric(regime_labels, errors='coerce')
                if regime_labels.isnull().any():
                    raise ValueError("Could not convert regime labels to numeric")
            
            # Data format logging
            tprint_data_format(f"Features dtype: {features_df.dtypes.value_counts().to_dict()}")
            tprint_data_format(f"Regime labels dtype: {regime_labels.dtype}")
            tprint_data_format(f"Regime distribution: {regime_labels.value_counts().to_dict()}")
            
            return features_df, regime_labels
            
        except Exception as e:
            tprint_error(f"Data validation failed: {e}")
            raise
    
    def _run_regime_feature_selection_pipeline(
        self,
        features_df: pd.DataFrame,
        regime_labels: pd.Series,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run the regime feature selection pipeline using regime labels as target."""
        try:
            results = {
                'selected_features': [],
                'feature_importance': {},
                'selection_metadata': {},
                'performance_metrics': {},
                'regime_analysis': {}
            }
            
            # Data leakage detection
            if self.leakage_detector:
                leakage_results = self.leakage_detector.detect_leakage(
                    features_df, regime_labels
                )
                results['leakage_detection'] = leakage_results
                tprint_info(f"Data leakage detection completed: {leakage_results}")
            
            # Main feature selection using TreeSHAP with regime labels as target
            selection_results = self._run_treeshap_selection(
                features_df, regime_labels, feature_names
            )
            results.update(selection_results)
            
            # Regime analysis (not regime-specific selection, but analysis of selected features)
            regime_analysis = self._analyze_regime_characteristics(
                features_df, regime_labels, results['selected_features']
            )
            results['regime_analysis'] = regime_analysis
            
            # Feature importance analysis
            if self.explainability_tool:
                importance_analysis = self._analyze_feature_importance(
                    features_df, regime_labels, results['selected_features']
                )
                results['importance_analysis'] = importance_analysis
            
            # Performance evaluation
            if self.evaluator:
                evaluation_results = self._evaluate_selection_performance(
                    features_df, regime_labels, results['selected_features']
                )
                results['evaluation_results'] = evaluation_results
            
            return results
            
        except Exception as e:
            tprint_error(f"Regime feature selection pipeline failed: {e}")
            raise
    
    def _setup_hardware_optimization(self):
        """Setup hardware optimization for the current workload."""
        try:
            if self.hardware_manager:
                # Configure hardware for ML training workload
                self.hardware_manager.configure_workload(
                    workload_type=self.config.workload_type,
                    optimization_level=self.config.optimization_level
                )
                
                # Setup memory optimization
                if self.memory_optimizer:
                    self.memory_optimizer.optimize_for_ml_training()
                
                # Setup CPU optimization
                if self.cpu_optimizer:
                    self.cpu_optimizer.optimize_for_parallel_processing(
                        max_workers=self.config.max_workers
                    )
                
                tprint_info("Hardware optimization configured")
                
        except Exception as e:
            tprint_warning(f"Hardware optimization setup failed: {e}")
    
    def _run_unsupervised_feature_selection_pipeline(
        self,
        features_df: pd.DataFrame,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Run unsupervised feature selection pipeline.
        
        Uses variance-based and correlation-based filtering to select features
        without requiring regime labels. This avoids circular dependency.
        """
        try:
            tprint_info("Running unsupervised feature selection pipeline")
            
            results = {
                'selected_features': [],
                'feature_importance': {},
                'selection_metadata': {
                    'selection_method': 'unsupervised_variance_correlation',
                    'execution_time': 0.0
                },
                'performance_metrics': {},
                'regime_analysis': {}
            }
            
            import time
            start_time = time.time()
            
            # Step 1: Remove low-variance features
            tprint_info("Step 1: Removing low-variance features")
            variances = features_df.var()
            variance_threshold = variances.quantile(0.10)  # Keep top 90%
            high_variance_features = variances[variances > variance_threshold].index.tolist()
            
            tprint_info(f"Kept {len(high_variance_features)}/{len(features_df.columns)} features after variance filtering")
            
            if len(high_variance_features) == 0:
                tprint_warning("No features passed variance threshold, using all features")
                high_variance_features = list(features_df.columns)
            
            # Step 2: Remove highly correlated features
            tprint_info("Step 2: Removing highly correlated features")
            features_subset = features_df[high_variance_features]
            
            # Calculate correlation matrix
            corr_matrix = features_subset.corr().abs()
            
            # Find features to drop (keep first of each correlated pair)
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > 0.95)]
            
            decorrelated_features = [f for f in high_variance_features if f not in to_drop]
            
            tprint_info(f"Removed {len(to_drop)} highly correlated features, {len(decorrelated_features)} remaining")
            
            if len(decorrelated_features) == 0:
                tprint_warning("No features after correlation filtering, using variance-filtered features")
                decorrelated_features = high_variance_features
            
            # Step 3: Select top features by variance
            feature_variances = variances[decorrelated_features].sort_values(ascending=False)
            
            # Limit to max_features
            max_features = min(self.config.max_features, len(decorrelated_features))
            selected_features = feature_variances.head(max_features).index.tolist()
            
            # Normalize variances for importance scores (0-1 range)
            if len(feature_variances) > 0:
                normalized_variances = (feature_variances - feature_variances.min()) / (feature_variances.max() - feature_variances.min() + 1e-10)
                feature_importance = normalized_variances.to_dict()
            else:
                feature_importance = {}
            
            execution_time = time.time() - start_time
            
            tprint_success(f"Unsupervised selection completed: {len(selected_features)} features selected in {execution_time:.2f}s")
            
            # Update results
            results['selected_features'] = selected_features
            results['feature_importance'] = feature_importance
            results['selection_metadata'].update({
                'total_features': len(features_df.columns),
                'variance_filtered': len(high_variance_features),
                'correlation_filtered': len(decorrelated_features),
                'final_selected': len(selected_features),
                'execution_time': execution_time,
                'variance_threshold': float(variance_threshold),
                'correlation_threshold': 0.95
            })
            
            return results
            
        except Exception as e:
            tprint_error(f"Unsupervised feature selection failed: {e}")
            raise
    
    def _run_treeshap_selection(
        self,
        features_df: pd.DataFrame,
        regime_labels: pd.Series,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run TreeSHAP-based feature selection using regime labels as target."""
        try:
            tprint_info("Running TreeSHAP feature selection for regime detection")
            tprint_info(f"Using regime labels as target with {regime_labels.nunique()} unique regimes")
            
            # Use VectorBT optimization if available
            if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
                # Optimize features using VectorBT
                optimized_features = self._optimize_features_with_vectorbt(
                    features_df, regime_labels
                )
            else:
                optimized_features = features_df
            
            # Run TreeSHAP selection with regime labels as target
            selection_results = self.treeshap_selector.select_features(
                optimized_features,
                regime_labels,  # Use regime labels as target
                feature_names=feature_names or list(features_df.columns),
                max_features=self.config.max_features,
                min_importance=self.config.min_feature_importance
            )
            
            tprint_success(f"TreeSHAP selection completed: {len(selection_results.get('selected_features', []))} features selected")
            return selection_results
            
        except Exception as e:
            tprint_error(f"TreeSHAP selection failed: {e}")
            raise
    
    def _analyze_regime_characteristics(
        self,
        features_df: pd.DataFrame,
        regime_labels: pd.Series,
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """Analyze characteristics of selected features across regimes."""
        try:
            tprint_info("Analyzing regime characteristics of selected features")
            
            if not selected_features:
                return {'regime_analysis': 'No features selected'}
            
            regime_analysis = {}
            unique_regimes = regime_labels.unique()
            
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_data = features_df[selected_features][regime_mask]
                
                regime_stats = {
                    'sample_count': len(regime_data),
                    'feature_means': regime_data.mean().to_dict(),
                    'feature_stds': regime_data.std().to_dict(),
                    'feature_ranges': (regime_data.max() - regime_data.min()).to_dict()
                }
                
                regime_analysis[f'regime_{regime}'] = regime_stats
            
            # Overall regime separation analysis
            overall_stats = {
                'total_regimes': len(unique_regimes),
                'regime_distribution': regime_labels.value_counts().to_dict(),
                'selected_features_count': len(selected_features),
                'features_per_regime': {f'regime_{r}': len(regime_labels[regime_labels == r]) for r in unique_regimes}
            }
            
            regime_analysis['overall'] = overall_stats
            
            tprint_success(f"Regime analysis completed for {len(unique_regimes)} regimes")
            return regime_analysis
            
        except Exception as e:
            tprint_warning(f"Regime analysis failed: {e}")
            return {'regime_analysis': f'Analysis failed: {e}'}
    
    def _optimize_features_with_vectorbt(
        self,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> pd.DataFrame:
        """Optimize features using VectorBT rolling operations."""
        try:
            if not self.vectorbt_optimizer or not VECTORBT_AVAILABLE:
                return features_df
            
            tprint_info("Optimizing features with VectorBT")
            
            # Apply rolling optimizations to features
            optimized_features = features_df.copy()
            
            # Use VectorBT rolling operations for feature enhancement
            for col in features_df.columns:
                if features_df[col].dtype in ['float64', 'int64']:
                    # Apply rolling mean and std
                    rolling_mean_val = rolling_mean(
                        features_df[col], 
                        window=self.config.vectorbt_rolling_window
                    )
                    rolling_std_val = rolling_std(
                        features_df[col], 
                        window=self.config.vectorbt_rolling_window
                    )
                    
                    # Add enhanced features
                    optimized_features[f"{col}_rolling_mean"] = rolling_mean_val
                    optimized_features[f"{col}_rolling_std"] = rolling_std_val
            
            tprint_info(f"VectorBT optimization completed: {optimized_features.shape[1]} features")
            return optimized_features
            
        except Exception as e:
            tprint_warning(f"VectorBT optimization failed: {e}, using original features")
            return features_df
    
    
    def _analyze_feature_importance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """Analyze feature importance using explainability tools."""
        try:
            if not self.explainability_tool:
                return {}
            
            tprint_info("Analyzing feature importance")
            
            # Use SHAP/LIME for feature importance analysis
            importance_analysis = self.explainability_tool.analyze_features(
                features_df[selected_features],
                target
            )
            
            return importance_analysis
            
        except Exception as e:
            tprint_warning(f"Feature importance analysis failed: {e}")
            return {}
    
    def _evaluate_selection_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """Evaluate the performance of selected features."""
        try:
            if not self.evaluator:
                return {}
            
            tprint_info("Evaluating selection performance")
            
            # Use temporal cross-validation if available
            if self.temporal_validator:
                evaluation_results = self.temporal_validator(
                    features_df[selected_features],
                    target,
                    cv_folds=5
                )
            else:
                # Basic evaluation
                evaluation_results = self.evaluator.evaluate(
                    features_df[selected_features],
                    target
                )
            
            return evaluation_results
            
        except Exception as e:
            tprint_warning(f"Performance evaluation failed: {e}")
            return {}
    
    def _track_performance(self, results: Dict[str, Any]):
        """Track performance metrics."""
        try:
            self.performance_metrics.update({
                'selection_time': results.get('selection_metadata', {}).get('execution_time', 0),
                'features_selected': len(results.get('selected_features', [])),
                'total_features': results.get('selection_metadata', {}).get('total_features', 0),
                'selection_ratio': len(results.get('selected_features', [])) / max(1, results.get('selection_metadata', {}).get('total_features', 1))
            })
            
            tprint_performance(f"Performance metrics: {self.performance_metrics}")
            
        except Exception as e:
            tprint_warning(f"Performance tracking failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
    
    def get_feature_importance_cache(self) -> Dict[str, Any]:
        """Get cached feature importance scores."""
        return self.feature_importance_cache.copy()
    
    def clear_cache(self):
        """Clear all caches."""
        self.feature_importance_cache.clear()
        self.performance_metrics.clear()
        tprint_info("Caches cleared")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the regime feature selection step.
        
        IMPORTANT: This runs BEFORE clustering, so it uses UNSUPERVISED feature selection
        to avoid circular dependency. It selects features optimized for regime clustering
        using variance, correlation, and category-based filtering.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframes: List of timeframes
                - execution_mode: 'light' or 'full'
                - feature_selection_config: Optional custom config
                - features_data: Optional pre-loaded features data
                - use_supervised: Optional bool (default False) - only True if regime_labels provided
                - regime_labels: Optional pre-loaded regime labels (for supervised mode)
        
        Returns:
            Dict containing execution results and artifacts
        """
        try:
            self.logger.info("Starting UNSUPERVISED regime feature selection for clustering")
            
            # Update config with any custom settings
            if 'feature_selection_config' in config:
                custom_config = config['feature_selection_config']
                for key, value in custom_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            # Extract configuration parameters
            symbol = config.get('symbol', 'UNKNOWN')
            exchange = config.get('exchange', 'UNKNOWN')
            timeframes = config.get('timeframes', ['15m'])
            execution_mode = config.get('execution_mode', 'light')
            
            tprint_info(f"Processing regime feature selection for {symbol} on {exchange}")
            tprint_info(f"Timeframes: {timeframes}, Mode: {execution_mode}")
            
            # Load features data (regime labels optional)
            features_data, regime_labels = await self._load_features_and_regime_labels(config)
            
            if features_data is None or features_data.empty:
                raise ValueError("No features data available for feature selection")
            
            # Apply regime feature categorization to pre-filter features
            tprint_info("🎯 Applying regime feature categorization...")
            features_data = self._apply_regime_categorization(features_data)
            
            # Apply light mode filtering if needed
            features_data = self._apply_light_mode_filter(features_data, config, timeframes[0])
            if regime_labels is not None:
                regime_labels = self._apply_light_mode_filter(regime_labels, config, timeframes[0])
            
            # Determine selection mode
            use_supervised = config.get('use_supervised', False) and regime_labels is not None
            
            if use_supervised:
                tprint_warning("⚠️ Using SUPERVISED mode - ensure this is post-clustering refinement!")
                # Perform supervised feature selection using regime labels
                selection_results = self.select_features(
                    features_df=features_data,
                    regime_labels=regime_labels,
                    use_supervised=True
                )
            else:
                tprint_info("✅ Using UNSUPERVISED mode - optimal for pre-clustering feature selection")
                # Perform unsupervised feature selection (no regime labels needed)
                selection_results = self.select_features(
                    features_df=features_data,
                    regime_labels=None,
                    use_supervised=False
                )
            
            # Save artifacts
            artifacts = []
            
            # Save selected features
            selected_features_path = self._save_artifact(
                data=selection_results['selected_features'],
                artifact_name=f'selected_features_{symbol}_{exchange}',
                artifact_type='data',
                metadata={
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframes': timeframes,
                    'execution_mode': execution_mode,
                    'selection_method': selection_results.get('selection_method', 'treeshap'),
                    'total_features': len(features_data.columns),
                    'selected_count': len(selection_results['selected_features']),
                    'timestamp': datetime.now().isoformat()
                }
            )
            artifacts.append(selected_features_path)
            
            # Save feature importance scores
            if 'feature_importance' in selection_results:
                importance_path = self._save_artifact(
                    data=selection_results['feature_importance'],
                    artifact_name=f'feature_importance_{symbol}_{exchange}',
                    artifact_type='data',
                    metadata={
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframes': timeframes,
                        'execution_mode': execution_mode,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                artifacts.append(importance_path)
            
            # Save regime-specific results if available
            if 'regime_specific_results' in selection_results and selection_results['regime_specific_results']:
                regime_results_path = self._save_artifact(
                    data=selection_results['regime_specific_results'],
                    artifact_name=f'regime_specific_features_{symbol}_{exchange}',
                    artifact_type='data',
                    metadata={
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframes': timeframes,
                        'execution_mode': execution_mode,
                        'regime_count': len(selection_results['regime_specific_results']),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                artifacts.append(regime_results_path)
            
            # Save performance metrics
            performance_metrics = self.get_performance_metrics()
            if performance_metrics:
                metrics_path = self._save_artifact(
                    data=performance_metrics,
                    artifact_name=f'feature_selection_metrics_{symbol}_{exchange}',
                    artifact_type='metadata',
                    metadata={
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframes': timeframes,
                        'execution_mode': execution_mode,
                        'timestamp': datetime.now().isoformat()
                    }
                )
                artifacts.append(metrics_path)
            
            # Generate comprehensive markdown report
            markdown_report = self._generate_comprehensive_markdown_report(
                symbol, exchange, timeframes, execution_mode, 
                selection_results, performance_metrics,
                features_data, None, regime_labels  # target_data=None in unsupervised mode
            )
            
            # Save markdown report to outcomes directory
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"regime_feature_selection_report_{symbol}_{exchange}_{timestamp_str}.md"
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            report_path = outcomes_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            
            tprint_success(f"Comprehensive report saved: {report_path}")
            
            # Also save as artifact for consistency
            report_data = self._generate_execution_report(
                symbol, exchange, timeframes, execution_mode, 
                selection_results, performance_metrics
            )
            
            artifact_report_path = self._save_artifact(
                data=report_data,
                artifact_name=f'feature_selection_report_{symbol}_{exchange}',
                artifact_type='report',
                metadata={
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframes': timeframes,
                    'execution_mode': execution_mode,
                    'timestamp': datetime.now().isoformat()
                }
            )
            artifacts.append(artifact_report_path)
            
            # Prepare execution result
            execution_result = {
                'success': True,
                'artifacts': artifacts,
                'metrics': {
                    'selected_features_count': len(selection_results['selected_features']),
                    'total_features_count': len(features_data.columns),
                    'selection_ratio': len(selection_results['selected_features']) / len(features_data.columns),
                    'execution_mode': execution_mode,
                    'performance_metrics': performance_metrics
                },
                'selected_features': selection_results['selected_features'],
                'feature_importance': selection_results.get('feature_importance', {}),
                'regime_specific_results': selection_results.get('regime_specific_results', {}),
                'report_path': str(report_path),
                'markdown_report_path': str(report_path)
            }
            
            tprint_success(f"Regime feature selection completed successfully for {symbol}")
            tprint_info(f"Selected {len(selection_results['selected_features'])} features from {len(features_data.columns)} total")
            
            return execution_result
            
        except Exception as e:
            error_msg = f"Regime feature selection failed: {str(e)}"
            self.logger.error(error_msg)
            tprint_error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {}
            }
    
    async def _load_features_and_regime_labels(self, config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load features data (regime labels optional for unsupervised mode)."""
        try:
            symbol = config.get('symbol', 'UNKNOWN')
            exchange = config.get('exchange', 'UNKNOWN')
            
            # Try to load pre-loaded data first
            features_data = config.get('features_data')
            regime_labels = config.get('regime_labels')  # Optional
            
            if features_data is not None:
                mode = "with regime labels" if regime_labels is not None else "WITHOUT regime labels (unsupervised mode)"
                tprint_info(f"Using pre-loaded features data {mode}")
                return features_data, regime_labels
            
            # Try to load from artifacts
            try:
                features_data = self._get_artifact(
                    artifact_name=f'features_{symbol}_{exchange}',
                    artifact_type='data'
                )
                regime_labels = self._get_artifact(
                    artifact_name=f'regime_labels_{symbol}_{exchange}',
                    artifact_type='data'
                )
                tprint_info("Loaded data from artifacts")
                return features_data, regime_labels
            except Exception as e:
                self.logger.debug(f"Could not load data from artifacts: {e}")
            
            # Try to load from feature bank
            try:
                from src.feature_generation.core.feature_bank import get_global_feature_bank
                feature_bank = get_global_feature_bank()
                
                # Generate features for the symbol/exchange
                features_result = feature_bank.generate_features(
                    symbol=symbol,
                    exchange=exchange,
                    timeframes=config.get('timeframes', ['15m'])
                )
                
                if features_result and 'features' in features_result:
                    features_data = features_result['features']
                    regime_labels = features_result.get('regime_labels')
                    tprint_info("Generated data from feature bank")
                    return features_data, regime_labels
            except Exception as e:
                self.logger.debug(f"Could not generate data from feature bank: {e}")
            
            # Generate sample data as fallback (no regime labels for unsupervised mode)
            tprint_warning("No data available, generating sample data for testing (unsupervised mode)")
            features_df, _ = self._generate_sample_data()
            return features_df, None  # Return None for regime_labels in unsupervised mode
            
        except Exception as e:
            self.logger.error(f"Error loading features and regime labels: {e}")
            return None, None
    
    def _apply_regime_categorization(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regime feature categorization to filter features appropriate for clustering.
        
        Uses the regime_feature_categorization system to select features optimized
        for regime clustering, avoiding features meant for live trading or other purposes.
        """
        try:
            from src.feature_generation.categories.regime_feature_categorization import (
                get_regime_clustering_features,
                RegimeFeatureCategorizer,
                FeatureUseCase
            )
            
            tprint_info("📋 Loading regime clustering feature priorities...")
            categorizer = RegimeFeatureCategorizer()
            
            # Get priority features for regime clustering
            priority_features = categorizer.get_priority_features(
                FeatureUseCase.REGIME_CLUSTERING,
                max_features=200  # Get top 200 priority features
            )
            
            tprint_info(f"🎯 Found {len(priority_features)} priority regime clustering features")
            
            # Filter features_df to only include those that match priority feature patterns
            # Since priority_features contains generic names, match by pattern
            matching_features = []
            for col in features_df.columns:
                col_lower = col.lower()
                # Check if column matches any priority feature pattern
                for priority_feature in priority_features:
                    if priority_feature.lower() in col_lower:
                        matching_features.append(col)
                        break
            
            if matching_features:
                filtered_df = features_df[matching_features]
                tprint_success(f"✅ Filtered to {len(filtered_df.columns)} regime-optimized features (from {len(features_df.columns)} total)")
                return filtered_df
            else:
                tprint_warning("⚠️ No matching regime features found, using all features")
                return features_df
                
        except ImportError as e:
            tprint_warning(f"⚠️ Regime feature categorization not available: {e}")
            tprint_info("Using all features without categorization filtering")
            return features_df
        except Exception as e:
            tprint_warning(f"⚠️ Error applying regime categorization: {e}")
            tprint_info("Using all features without categorization filtering")
            return features_df
    
    # REMOVED: _load_or_generate_data() - Dead code, not used after unsupervised mode refactoring
    # Use _load_features_and_regime_labels() instead, which doesn't require target_data
    
    def _generate_sample_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate sample data for testing purposes."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # Generate features
        features_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        
        # Generate regime labels with some structure
        # Create 3 regimes with different characteristics
        regime_labels = pd.Series(
            np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            index=features_data.index
        )
        
        # Add some regime-specific structure to features
        for i, regime in enumerate([0, 1, 2]):
            regime_mask = regime_labels == regime
            if regime == 0:  # Low volatility regime
                features_data.loc[regime_mask, :20] *= 0.5
            elif regime == 1:  # High volatility regime
                features_data.loc[regime_mask, :20] *= 2.0
            # Regime 2 stays normal
        
        return features_data, regime_labels
    
    def _generate_execution_report(
        self, 
        symbol: str, 
        exchange: str, 
        timeframes: List[str], 
        execution_mode: str,
        selection_results: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive execution report."""
        return {
            'execution_summary': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframes': timeframes,
                'execution_mode': execution_mode,
                'timestamp': datetime.now().isoformat(),
                'step_name': self.step_name
            },
            'feature_selection_results': {
                'total_features': len(selection_results.get('selected_features', [])),
                'selected_features': selection_results.get('selected_features', []),
                'selection_method': selection_results.get('selection_method', 'unknown'),
                'feature_importance_available': 'feature_importance' in selection_results,
                'regime_specific_available': 'regime_specific_results' in selection_results
            },
            'performance_metrics': performance_metrics,
            'component_availability': {
                'treeshap_available': hasattr(self, 'treeshap_selector') and self.treeshap_selector is not None,
                'vectorbt_available': hasattr(self, 'vectorbt_optimizer') and self.vectorbt_optimizer is not None,
                'hardware_optimization_available': hasattr(self, 'hardware_manager') and self.hardware_manager is not None,
                'ml_common_available': hasattr(self, 'hpo_optimizer') and self.hpo_optimizer is not None
            },
            'configuration': {
                'max_features': self.config.max_features,
                'min_feature_importance': self.config.min_feature_importance,
                'use_hardware_optimization': self.config.use_hardware_optimization,
                'use_hpo': self.config.use_hpo,
                'use_explainability': self.config.use_explainability
            }
        }
    
    def _generate_comprehensive_markdown_report(
        self,
        symbol: str,
        exchange: str,
        timeframes: List[str],
        execution_mode: str,
        selection_results: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        features_data: Optional[pd.DataFrame] = None,
        target_data: Optional[pd.Series] = None,
        regime_labels: Optional[pd.Series] = None
    ) -> str:
        """Generate comprehensive markdown report with per-feature metrics."""
        try:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Calculate basic statistics
            total_features = len(features_data.columns) if features_data is not None else 0
            selected_features = selection_results.get('selected_features', [])
            selected_count = len(selected_features)
            selection_ratio = selected_count / max(1, total_features)
            
            # Get feature importance scores
            feature_importance = selection_results.get('feature_importance', {})
            
            # Get regime-specific results
            regime_specific_results = selection_results.get('regime_specific_results', {})
            
            # Calculate per-feature metrics if data is available
            per_feature_metrics = {}
            if features_data is not None and target_data is not None:
                per_feature_metrics = self._calculate_per_feature_metrics(
                    features_data, target_data, selected_features
                )
            
            # Generate markdown content
            markdown_content = f"""# Regime Feature Selection Comprehensive Report

**Generated**: {timestamp.isoformat()}  
**Symbol**: {symbol}  
**Exchange**: {exchange}  
**Timeframes**: {', '.join(timeframes)}  
**Execution Mode**: {execution_mode}  
**Selection Method**: {selection_results.get('selection_method', 'unknown')}  

---

## 📊 Executive Summary

This report provides a comprehensive analysis of the regime feature selection process, including detailed metrics for each selected feature, regime-specific analysis, and performance assessments.

### Key Results
- **Total Features**: {total_features:,}
- **Selected Features**: {selected_count:,}
- **Selection Ratio**: {selection_ratio:.2%}
- **Processing Time**: {performance_metrics.get('selection_time', 0):.2f} seconds
- **Selection Method**: {selection_results.get('selection_method', 'unknown')}
- **Regime-Specific Analysis**: {'✅ Available' if regime_specific_results else '❌ Not Available'}

---

## 🔍 Feature Selection Analysis

### Selection Statistics

| Metric | Value |
|--------|-------|
| **Total Features** | {total_features:,} |
| **Selected Features** | {selected_count:,} |
| **Selection Ratio** | {selection_ratio:.2%} |
| **Min Importance Threshold** | {self.config.min_feature_importance:.4f} |
| **Max Features Limit** | {self.config.max_features} |

### Selection Method Details

- **Primary Method**: {selection_results.get('selection_method', 'unknown')}
- **TreeSHAP Available**: {'✅ Yes' if hasattr(self, 'treeshap_selector') and self.treeshap_selector is not None else '❌ No'}
- **VectorBT Optimization**: {'✅ Yes' if hasattr(self, 'vectorbt_optimizer') and self.vectorbt_optimizer is not None else '❌ No'}
- **Hardware Optimization**: {'✅ Yes' if hasattr(self, 'hardware_manager') and self.hardware_manager is not None else '❌ No'}

---

## 📈 Per-Feature Analysis

### Top 20 Selected Features

"""
            
            # Add top features table
            if selected_features and feature_importance:
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20]
                
                markdown_content += """
| Rank | Feature Name | Importance Score | Category | Stability |
|------|--------------|------------------|----------|-----------|
"""
                
                for i, (feature_name, importance) in enumerate(sorted_features, 1):
                    # Get additional metrics for this feature
                    feature_metrics = per_feature_metrics.get(feature_name, {})
                    category = feature_metrics.get('category', 'Unknown')
                    stability = feature_metrics.get('stability', 0.0)
                    
                    markdown_content += f"| {i} | `{feature_name}` | {importance:.4f} | {category} | {stability:.3f} |\n"
            else:
                markdown_content += "No feature importance data available.\n"
            
            markdown_content += "\n### Complete Feature List\n\n"
            
            # Add complete feature list
            if selected_features:
                markdown_content += "The following features were selected for regime-based trading:\n\n"
                for i, feature in enumerate(selected_features, 1):
                    importance = feature_importance.get(feature, 0.0)
                    feature_metrics = per_feature_metrics.get(feature, {})
                    category = feature_metrics.get('category', 'Unknown')
                    
                    markdown_content += f"{i}. **{feature}**\n"
                    markdown_content += f"   - Importance: {importance:.4f}\n"
                    markdown_content += f"   - Category: {category}\n"
                    if 'correlation' in feature_metrics:
                        markdown_content += f"   - Target Correlation: {feature_metrics['correlation']:.4f}\n"
                    if 'variance' in feature_metrics:
                        markdown_content += f"   - Variance: {feature_metrics['variance']:.4f}\n"
                    markdown_content += "\n"
            else:
                markdown_content += "No features were selected.\n"
            
            # Add regime-specific analysis
            if regime_specific_results:
                markdown_content += "---\n\n## 🎯 Regime-Specific Analysis\n\n"
                markdown_content += "### Regime Distribution\n\n"
                
                for regime_name, regime_data in regime_specific_results.items():
                    regime_features = regime_data.get('selected_features', [])
                    regime_importance = regime_data.get('feature_importance', {})
                    
                    markdown_content += f"#### {regime_name.replace('_', ' ').title()}\n\n"
                    markdown_content += f"- **Selected Features**: {len(regime_features)}\n"
                    markdown_content += f"- **Top Features**: {', '.join(regime_features[:5]) if regime_features else 'None'}\n\n"
                    
                    if regime_importance:
                        top_regime_features = sorted(
                            regime_importance.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:5]
                        
                        markdown_content += "**Top Features by Importance:**\n"
                        for feature, importance in top_regime_features:
                            markdown_content += f"- `{feature}`: {importance:.4f}\n"
                        markdown_content += "\n"
            
            # Add performance metrics
            markdown_content += "---\n\n## ⚡ Performance Metrics\n\n"
            markdown_content += "### Execution Performance\n\n"
            markdown_content += f"- **Total Execution Time**: {performance_metrics.get('selection_time', 0):.2f} seconds\n"
            markdown_content += f"- **Features Processed**: {performance_metrics.get('total_features', 0):,}\n"
            markdown_content += f"- **Selection Efficiency**: {performance_metrics.get('selection_ratio', 0):.2%}\n"
            markdown_content += f"- **Memory Usage**: {performance_metrics.get('memory_usage', 'N/A')}\n\n"
            
            # Add component status
            markdown_content += "### Component Status\n\n"
            markdown_content += f"- **TreeSHAP Integration**: {'✅ Active' if hasattr(self, 'treeshap_selector') and self.treeshap_selector is not None else '❌ Inactive'}\n"
            markdown_content += f"- **VectorBT Optimization**: {'✅ Active' if hasattr(self, 'vectorbt_optimizer') and self.vectorbt_optimizer is not None else '❌ Inactive'}\n"
            markdown_content += f"- **Hardware Optimization**: {'✅ Active' if hasattr(self, 'hardware_manager') and self.hardware_manager is not None else '❌ Inactive'}\n"
            markdown_content += f"- **ML Common Utilities**: {'✅ Active' if hasattr(self, 'hpo_optimizer') and self.hpo_optimizer is not None else '❌ Inactive'}\n\n"
            
            # Add configuration details
            markdown_content += "---\n\n## ⚙️ Configuration Details\n\n"
            markdown_content += "### Feature Selection Parameters\n\n"
            markdown_content += f"- **Max Features**: {self.config.max_features}\n"
            markdown_content += f"- **Min Feature Importance**: {self.config.min_feature_importance:.4f}\n"
            markdown_content += f"- **Selection Method**: {self.config.feature_selection_method}\n"
            markdown_content += f"- **Use HPO**: {'Yes' if self.config.use_hpo else 'No'}\n"
            markdown_content += f"- **Use Explainability**: {'Yes' if self.config.use_explainability else 'No'}\n"
            markdown_content += f"- **Use Data Leakage Detection**: {'Yes' if self.config.use_data_leakage_detection else 'No'}\n\n"
            
            # Add recommendations
            markdown_content += "---\n\n## 🎯 Recommendations\n\n"
            markdown_content += "### For Trading Strategy\n"
            markdown_content += f"- **Feature Count**: {selected_count} features selected for regime-based trading\n"
            markdown_content += f"- **Selection Quality**: {'High' if selection_ratio < 0.5 else 'Moderate' if selection_ratio < 0.8 else 'Low'} (lower is better)\n"
            markdown_content += f"- **Regime Coverage**: {'Comprehensive' if regime_specific_results else 'Basic'} regime-specific analysis\n\n"
            
            markdown_content += "### For Further Analysis\n"
            markdown_content += "- **Feature Validation**: Consider cross-validation with different time periods\n"
            markdown_content += "- **Regime Profiling**: Analyze regime-specific feature importance patterns\n"
            markdown_content += "- **Temporal Stability**: Monitor feature importance over time\n"
            markdown_content += "- **Interaction Analysis**: Investigate feature interactions within regimes\n\n"
            
            # Add artifact summary
            markdown_content += "---\n\n## 📋 Artifact Summary\n\n"
            markdown_content += "**Generated Artifacts:**\n"
            markdown_content += f"- `selected_features_{symbol}_{exchange}`: Main selected features list\n"
            markdown_content += f"- `feature_importance_{symbol}_{exchange}`: Feature importance scores\n"
            if regime_specific_results:
                markdown_content += f"- `regime_specific_features_{symbol}_{exchange}`: Regime-specific selections\n"
            markdown_content += f"- `feature_selection_metrics_{symbol}_{exchange}`: Performance metrics\n"
            markdown_content += f"- `feature_selection_report_{symbol}_{exchange}`: This comprehensive report\n\n"
            
            markdown_content += "**File Locations:**\n"
            markdown_content += f"- **Artifacts**: `artifacts/market_analysis/{symbol}/{exchange}/regime_feature_selection/`\n"
            markdown_content += f"- **Report**: `outcomes/regime_feature_selection_report_{symbol}_{exchange}_{timestamp_str}.md`\n\n"
            
            markdown_content += "---\n\n"
            markdown_content += f"*Report generated by Ares Regime Feature Selector v1.0*\n"
            markdown_content += f"*Generated on: {timestamp.isoformat()}*\n"
            
            return markdown_content
            
        except Exception as e:
            tprint_error(f"Error generating comprehensive markdown report: {e}")
            self.logger.error(f"Markdown report generation failed: {e}")
            return f"# Error Generating Report\n\nError: {str(e)}\n\nGenerated: {datetime.now().isoformat()}"
    
    def _calculate_per_feature_metrics(
        self,
        features_data: pd.DataFrame,
        target_data: pd.Series,
        selected_features: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate detailed metrics for each selected feature."""
        try:
            per_feature_metrics = {}
            
            for feature in selected_features:
                if feature not in features_data.columns:
                    continue
                
                feature_data = features_data[feature]
                
                # Basic statistics
                metrics = {
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'variance': float(feature_data.var()),
                    'skewness': float(feature_data.skew()),
                    'kurtosis': float(feature_data.kurtosis()),
                    'correlation': float(feature_data.corr(target_data)) if not target_data.empty else 0.0,
                    'missing_ratio': float(feature_data.isnull().sum() / len(feature_data)),
                    'zero_ratio': float((feature_data == 0).sum() / len(feature_data))
                }
                
                # Categorize feature based on name patterns
                category = self._categorize_feature(feature)
                metrics['category'] = category
                
                # Calculate stability (inverse of coefficient of variation)
                if metrics['mean'] != 0:
                    metrics['stability'] = 1.0 / (abs(metrics['std'] / metrics['mean']))
                else:
                    metrics['stability'] = 0.0
                
                # Calculate information content (entropy approximation)
                try:
                    # Discretize for entropy calculation
                    discretized = pd.cut(feature_data, bins=10, duplicates='drop')
                    value_counts = discretized.value_counts()
                    probabilities = value_counts / len(discretized)
                    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                    metrics['entropy'] = float(entropy)
                except:
                    metrics['entropy'] = 0.0
                
                per_feature_metrics[feature] = metrics
            
            return per_feature_metrics
            
        except Exception as e:
            tprint_warning(f"Error calculating per-feature metrics: {e}")
            return {}
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature based on name patterns."""
        feature_lower = feature_name.lower()
        
        if any(indicator in feature_lower for indicator in ['rsi', 'stoch', 'williams', 'cci']):
            return 'Momentum'
        elif any(indicator in feature_lower for indicator in ['sma', 'ema', 'dema', 'tema', 'macd']):
            return 'Trend'
        elif any(indicator in feature_lower for indicator in ['volume', 'vol']):
            return 'Volume'
        elif any(indicator in feature_lower for indicator in ['returns', 'log_returns', 'simple_returns']):
            return 'Returns'
        elif any(indicator in feature_lower for indicator in ['volatility', 'std', 'var']):
            return 'Volatility'
        elif any(indicator in feature_lower for indicator in ['sharpe', 'skewness', 'kurtosis']):
            return 'Risk'
        elif any(indicator in feature_lower for indicator in ['entropy', 'ljung', 'ar_']):
            return 'Statistical'
        elif any(indicator in feature_lower for indicator in ['vwap', 'price']):
            return 'Price'
        else:
            return 'Other'


def create_enhanced_regime_feature_selector(
    step_name: str = "regime_feature_selection"
) -> EnhancedRegimeFeatureSelector:
    """
    Factory function to create an Enhanced Regime Feature Selector.
    
    Args:
        step_name: Name for the step (used for artifact organization)
        
    Returns:
        Configured EnhancedRegimeFeatureSelector instance
    """
    return EnhancedRegimeFeatureSelector(step_name=step_name)


# Register the step with the global registry
from src.training.steps.base_step import step_registry
step_registry.register("regime_feature_selection", EnhancedRegimeFeatureSelector)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create selector
        selector = create_enhanced_regime_feature_selector()
        
        # Example configuration for execution
        config = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframes': ['15m'],
            'execution_mode': 'light',
            'feature_selection_config': {
                'max_features': 20,
                'min_feature_importance': 0.01,
                'verbose': True
            }
        }
        
        # Run the step
        result = await selector.run(config)
        
        # Print results
        print("\n" + "="*50)
        print("ENHANCED REGIME FEATURE SELECTION RESULTS")
        print("="*50)
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Selected features: {len(result['selected_features'])}")
            print(f"Selected features: {result['selected_features'][:10]}...")  # Show first 10
            print(f"Artifacts created: {len(result['artifacts'])}")
            print(f"Performance metrics: {result['metrics']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("="*50)
    
    # Run the example
    asyncio.run(main())