"""
Unified Feature Selection Framework

This module provides a comprehensive, unified feature selection system that:
1. Consolidates all existing feature selection methods
2. Leverages matrix operations for efficient computations
3. Provides backwards compatibility
4. Supports both price prediction and HMM regime prediction
5. Generates feature sets of different sizes (120, 100, 80, 60)

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Import existing feature selection components
try:
    from .feature_selection import FeatureSelectionFramework
    from .feature_selection_backwards_compat import FeatureSelector, FeatureSelectionConfig
    from .utils.feature_selection import FeatureSelectionFramework as UtilsFramework
    from .matrix_cross_validation import get_matrix_cross_validation
    EXISTING_COMPONENTS_AVAILABLE = True
except ImportError:
    EXISTING_COMPONENTS_AVAILABLE = False

# Import feature generation system
try:
    from ...feature_generation.core.feature_bank import FeatureBank, FeatureBankConfig
    from ...feature_generation.core.feature_generator import FeatureCategory
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False

# Import PID module
try:
    from .partial_information_decomposition import (
        PartialInformationDecomposition, PIDConfig, create_pid_module
    )
    PID_AVAILABLE = True
except ImportError:
    PID_AVAILABLE = False

# Import matrix operations
try:
    from .matrix_operations_example import get_unified_matrix_operations
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

# Import configuration
try:
    from ..config.enhanced_feature_selection_config import (
        get_default_enhanced_feature_selection_config,
        get_optimized_feature_selection_config,
        get_comprehensive_feature_selection_config,
        get_regime_specific_feature_selection_config
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class UnifiedFeatureSelectionConfig:
    """Configuration for unified feature selection."""
    
    # Core parameters
    target_features: int = 120
    min_features: int = 10
    max_features: int = 500
    
    # Task-specific parameters
    task_type: str = "regression"  # "regression" or "classification"
    prediction_target: str = "price"  # "price" or "hmm_regime"
    
    # Method selection
    primary_method: str = "hybrid"  # "filter", "wrapper", "embedded", "hybrid", "auto"
    secondary_methods: List[str] = field(default_factory=lambda: ["mrmr", "lasso_stability", "correlation_filter"])
    
    # Matrix operations integration
    use_matrix_operations: bool = True
    matrix_operation_method: str = "auto"  # "auto", "gpu", "cpu", "hybrid"
    
    # Performance settings
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    random_state: int = 42
    
    # Quality thresholds
    correlation_threshold: float = 0.95
    mutual_info_threshold: float = 0.001
    variance_threshold: float = 0.0
    importance_threshold: float = 0.001
    
    # Cross-validation
    cv_folds: int = 5
    enable_cross_validation: bool = True
    
    # Backwards compatibility
    enable_backwards_compatibility: bool = True
    legacy_method: str = "correlation"
    
    # Output settings
    save_results: bool = True
    output_dir: str = "feature_selection_results"
    verbose: bool = True
    
    # Feature generation integration
    build_on_feature_generation: bool = True
    feature_categories: List[str] = field(default_factory=lambda: [
        "returns", "momentum", "volume", "volatility", "trend", "oscillator",
        "support_resistance", "candlestick_pattern", "hmm_regime", "cross_timeframe",
        "microstructure", "entropy", "autoencoder", "order_flow", "time"
    ])
    
    # PID integration
    enable_pid: bool = True
    pid_config: Optional[Dict[str, Any]] = None
    
    # Iteration limits for specific methods
    lasso_max_iterations: int = 50  # Default: 50, Blank mode: 5, Light mode: 2
    mrmr_max_iterations: int = 50   # Default: 50, Blank mode: 5, Light mode: 2
    
    # Execution mode
    execution_mode: str = "full"  # "full", "blank", "light"


class UnifiedFeatureSelector:
    """
    Unified Feature Selection Framework
    
    This class consolidates all feature selection methods into a single, comprehensive
    framework that leverages matrix operations and provides backwards compatibility.
    """
    
    def __init__(self, config: Optional[UnifiedFeatureSelectionConfig] = None):
        """Initialize the unified feature selector."""
        self.config = config or UnifiedFeatureSelectionConfig()
        self.logger = logger.getChild('UnifiedFeatureSelector')
        
        # Adjust iteration limits based on execution mode
        self._adjust_iteration_limits()
        
        # Initialize components
        self._initialize_components()
        
        # Results storage
        self.results: Dict[str, Any] = {}
        self.feature_sets: Dict[str, List[str]] = {}
        self.feature_scores: Dict[str, Dict[str, float]] = {}
        self.generated_features: Dict[str, Any] = {}
        
        self.logger.info("🚀 UnifiedFeatureSelector initialized")
        self.logger.info(f"📊 Target features: {self.config.target_features}")
        self.logger.info(f"🎯 Task type: {self.config.task_type}")
        self.logger.info(f"🎯 Prediction target: {self.config.prediction_target}")
        self.logger.info(f"🔧 Execution mode: {self.config.execution_mode}")
        self.logger.info(f"🔧 LASSO max iterations: {self.config.lasso_max_iterations}")
        self.logger.info(f"🔧 mRMR max iterations: {self.config.mrmr_max_iterations}")
    
    def _adjust_iteration_limits(self):
        """Adjust iteration limits based on execution mode."""
        if self.config.execution_mode == "blank":
            self.config.lasso_max_iterations = 5
            self.config.mrmr_max_iterations = 5
        elif self.config.execution_mode == "light":
            self.config.lasso_max_iterations = 2
            self.config.mrmr_max_iterations = 2
        # "full" mode uses the default values (50)
    
    def _initialize_components(self):
        """Initialize all available components."""
        self.components = {}
        
        # Initialize existing frameworks
        if EXISTING_COMPONENTS_AVAILABLE:
            try:
                self.components['utils_framework'] = UtilsFramework()
                self.components['backwards_compat'] = FeatureSelector()
                self.logger.info("✅ Existing components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize existing components: {e}")
        
        # Initialize matrix operations
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.components['matrix_ops'] = get_unified_matrix_operations()
                self.logger.info("✅ Matrix operations initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize matrix operations: {e}")
        
        # Initialize configuration
        if CONFIG_AVAILABLE:
            try:
                self.configs = {
                    'default': get_default_enhanced_feature_selection_config(),
                    'optimized': get_optimized_feature_selection_config(),
                    'comprehensive': get_comprehensive_feature_selection_config()
                }
                self.logger.info("✅ Configuration loaded")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load configuration: {e}")
        
        # Initialize feature generation system
        if FEATURE_GENERATION_AVAILABLE and self.config.build_on_feature_generation:
            try:
                feature_bank_config = FeatureBankConfig(
                    enable_matrix_operations=self.config.use_matrix_operations,
                    enable_gpu_acceleration=True,
                    enable_lookback_optimization=True,
                    enable_parallel_processing=self.config.enable_parallel_processing
                )
                self.components['feature_bank'] = FeatureBank(feature_bank_config)
                self.logger.info("✅ Feature generation system initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize feature generation: {e}")
        
        # Initialize PID module
        if PID_AVAILABLE and self.config.enable_pid:
            try:
                pid_config = PIDConfig()
                if self.config.pid_config:
                    # Update PID config with custom settings
                    for key, value in self.config.pid_config.items():
                        if hasattr(pid_config, key):
                            setattr(pid_config, key, value)
                
                self.components['pid_module'] = create_pid_module(pid_config)
                self.logger.info("✅ PID module initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize PID module: {e}")
        
        # Initialize mRMR selector
        try:
            from ...training.utils.feature_selection.selection_methods import MRMRSelector
            mrmr_config = {
                'max_iterations': self.config.mrmr_max_iterations,
                'relevance_method': 'mutual_info',
                'redundancy_method': 'correlation'
            }
            self.components['mrmr_selector'] = MRMRSelector(mrmr_config)
            self.logger.info("✅ mRMR selector initialized")
        except ImportError:
            self.logger.warning("⚠️ mRMR selector not available")
    
    def generate_features_from_bank(
        self,
        data: pd.DataFrame,
        categories: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate features using the feature generation system.
        
        Args:
            data: Input data DataFrame
            categories: List of feature categories to generate
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not self.config.build_on_feature_generation or 'feature_bank' not in self.components:
            self.logger.warning("⚠️ Feature generation not available")
            return np.array([]), []
        
        self.logger.info("🔧 Generating features using feature bank")
        
        if categories is None:
            categories = self.config.feature_categories
        
        try:
            feature_bank = self.components['feature_bank']
            
            # Generate features by category
            generated_features = {}
            for category in categories:
                try:
                    category_features = feature_bank.generate_features_by_category(
                        data, category
                    )
                    generated_features.update(category_features)
                    self.logger.info(f"✅ Generated {len(category_features)} features for {category}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate features for {category}: {e}")
            
            # Convert to matrix format
            if generated_features:
                feature_df = pd.DataFrame(generated_features)
                feature_matrix = feature_df.values
                feature_names = feature_df.columns.tolist()
                
                # Store generated features
                self.generated_features = generated_features
                
                self.logger.info(f"✅ Generated {len(feature_names)} total features")
                return feature_matrix, feature_names
            else:
                self.logger.warning("⚠️ No features generated")
                return np.array([]), []
                
        except Exception as e:
            self.logger.error(f"❌ Feature generation failed: {e}")
            return np.array([]), []
    
    def create_pid_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        timeframe_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create polynomial and cross-timeframe features using PID.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            timeframe_data: Optional timeframe data for cross-timeframe features
            
        Returns:
            Tuple of (enhanced_feature_matrix, enhanced_feature_names)
        """
        if not self.config.enable_pid or 'pid_module' not in self.components:
            self.logger.warning("⚠️ PID module not available")
            return X, feature_names
        
        self.logger.info("🔍 Creating PID-based features")
        
        try:
            pid_module = self.components['pid_module']
            
            # Compute PID and create features
            pid_results = pid_module.compute_pid(X, y, feature_names)
            
            # Create polynomial features
            polynomial_features = pid_module.create_polynomial_features(X, feature_names)
            
            # Create cross-timeframe features if data provided
            cross_timeframe_features = {}
            if timeframe_data:
                cross_timeframe_features = pid_module.create_cross_timeframe_features(
                    X, feature_names, timeframe_data
                )
            
            # Combine all features
            enhanced_features = {}
            enhanced_feature_names = feature_names.copy()
            
            # Add polynomial features
            for name, feature_data in polynomial_features.items():
                enhanced_features[name] = feature_data
                enhanced_feature_names.append(name)
            
            # Add cross-timeframe features
            for name, feature_data in cross_timeframe_features.items():
                enhanced_features[name] = feature_data
                enhanced_feature_names.append(name)
            
            # Create enhanced feature matrix
            if enhanced_features:
                enhanced_df = pd.DataFrame(enhanced_features)
                enhanced_X = np.column_stack([X, enhanced_df.values])
                
                self.logger.info(f"✅ Created {len(enhanced_features)} PID-based features")
                return enhanced_X, enhanced_feature_names
            else:
                self.logger.info("ℹ️ No PID-based features created")
                return X, feature_names
                
        except Exception as e:
            self.logger.error(f"❌ PID feature creation failed: {e}")
            return X, feature_names
    
    def _mrmr_selection_with_limits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_features: int
    ) -> Dict[str, Any]:
        """
        Perform mRMR selection with iteration limits.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            n_features: Number of features to select
            
        Returns:
            Dictionary containing mRMR selection results
        """
        if 'mrmr_selector' not in self.components:
            self.logger.warning("⚠️ mRMR selector not available")
            return {
                'selected_features': [],
                'feature_scores': {},
                'method': 'mrmr',
                'error': 'mRMR selector not available'
            }
        
        self.logger.info(f"🔍 Performing mRMR selection with {self.config.mrmr_max_iterations} max iterations")
        
        try:
            mrmr_selector = self.components['mrmr_selector']
            
            # Perform mRMR selection with iteration limit
            result = mrmr_selector.select_features(X, y, feature_names, n_features)
            
            if result['success']:
                self.logger.info(f"✅ mRMR selected {len(result['selected_features'])} features")
                return {
                    'selected_features': result['selected_features'],
                    'feature_scores': result['scores'],
                    'method': 'mrmr',
                    'n_selected': len(result['selected_features']),
                    'selection_ratio': len(result['selected_features']) / len(feature_names)
                }
            else:
                self.logger.warning(f"⚠️ mRMR selection failed: {result.get('error', 'Unknown error')}")
                return {
                    'selected_features': [],
                    'feature_scores': {},
                    'method': 'mrmr',
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            self.logger.error(f"❌ mRMR selection failed: {e}")
            return {
                'selected_features': [],
                'feature_scores': {},
                'method': 'mrmr',
                'error': str(e)
            }
    
    def select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
        feature_names: Optional[List[str]] = None,
        target_sizes: Optional[List[int]] = None,
        input_data: Optional[pd.DataFrame] = None,
        timeframe_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Perform unified feature selection with feature generation integration.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            target_sizes: List of target feature set sizes
            input_data: Optional raw data for feature generation
            timeframe_data: Optional timeframe data for cross-timeframe features
            
        Returns:
            Dictionary containing all feature selection results
        """
        start_time = time.time()
        self.logger.info("🔍 Starting unified feature selection with feature generation")
        
        # Step 1: Generate features from feature bank if enabled
        if self.config.build_on_feature_generation and input_data is not None:
            self.logger.info("🔧 Step 1: Generating features from feature bank")
            generated_X, generated_names = self.generate_features_from_bank(input_data)
            
            if generated_X.size > 0:
                # Combine with existing features
                if isinstance(X, pd.DataFrame):
                    X_combined = pd.concat([X, pd.DataFrame(generated_X, columns=generated_names)], axis=1)
                    feature_names_combined = X_combined.columns.tolist()
                else:
                    X_combined = np.column_stack([X, generated_X])
                    feature_names_combined = (feature_names or [f'feature_{i}' for i in range(X.shape[1])]) + generated_names
                
                X = X_combined
                feature_names = feature_names_combined
                self.logger.info(f"✅ Combined features: {X.shape[1]} total features")
        
        # Step 2: Prepare data
        X_processed, y_processed, feature_names_processed = self._prepare_data(X, y, feature_names)
        
        # Step 3: Create PID-based features if enabled
        if self.config.enable_pid:
            self.logger.info("🔍 Step 2: Creating PID-based features")
            X_processed, feature_names_processed = self.create_pid_features(
                X_processed, y_processed, feature_names_processed, timeframe_data
            )
        
        # Set default target sizes if not provided
        if target_sizes is None:
            target_sizes = [120, 100, 80, 60]
        
        # Perform feature selection for each target size
        results = {}
        for target_size in target_sizes:
            self.logger.info(f"📊 Selecting {target_size} features")
            
            # Adjust config for this target size
            config_copy = self.config
            config_copy.target_features = target_size
            
            # Perform selection
            selection_result = self._perform_feature_selection(
                X_processed, y_processed, feature_names_processed, config_copy
            )
            
            results[f'top_{target_size}'] = selection_result
            self.feature_sets[f'top_{target_size}'] = selection_result['selected_features']
            self.feature_scores[f'top_{target_size}'] = selection_result['feature_scores']
        
        # Special handling for HMM regime prediction
        if self.config.prediction_target == "hmm_regime":
            self.logger.info("🎯 Performing HMM regime-specific feature selection")
            hmm_result = self._perform_hmm_regime_selection(
                X_processed, y_processed, feature_names_processed
            )
            results['hmm_regime_top_100'] = hmm_result
            self.feature_sets['hmm_regime_top_100'] = hmm_result['selected_features']
            self.feature_scores['hmm_regime_top_100'] = hmm_result['feature_scores']
        
        # Store results
        self.results = results
        
        # Save results if requested
        if self.config.save_results:
            self._save_results()
        
        execution_time = time.time() - start_time
        self.logger.info(f"✅ Unified feature selection completed in {execution_time:.3f}s")
        
        return results
    
    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
        feature_names: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for feature selection."""
        self.logger.info("🔧 Preparing data for feature selection")
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        y_array = np.array(y)
        
        # Handle missing values
        X_array = self._handle_missing_values(X_array)
        
        # Handle infinite values
        X_array = self._handle_infinite_values(X_array)
        
        # Remove constant features
        X_array, feature_names = self._remove_constant_features(X_array, feature_names)
        
        self.logger.info(f"📊 Data prepared: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        
        return X_array, y_array, feature_names
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values in the feature matrix."""
        if np.isnan(X).any():
            self.logger.warning("⚠️ Found NaN values, filling with column means")
            for i in range(X.shape[1]):
                col = X[:, i]
                nan_mask = np.isnan(col)
                if nan_mask.any():
                    col_mean = np.nanmean(col)
                    X[nan_mask, i] = col_mean
        return X
    
    def _handle_infinite_values(self, X: np.ndarray) -> np.ndarray:
        """Handle infinite values in the feature matrix."""
        if np.isinf(X).any():
            self.logger.warning("⚠️ Found infinite values, replacing with large finite values")
            
            # Replace positive infinity
            pos_inf_mask = np.isposinf(X)
            if pos_inf_mask.any():
                finite_mask = np.isfinite(X)
                if finite_mask.any():
                    max_finite = np.max(X[finite_mask])
                    X[pos_inf_mask] = max_finite * 10
                else:
                    X[pos_inf_mask] = 1e10
            
            # Replace negative infinity
            neg_inf_mask = np.isneginf(X)
            if neg_inf_mask.any():
                finite_mask = np.isfinite(X)
                if finite_mask.any():
                    min_finite = np.min(X[finite_mask])
                    X[neg_inf_mask] = min_finite * 10
                else:
                    X[neg_inf_mask] = -1e10
        
        return X
    
    def _remove_constant_features(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Remove constant features."""
        # Calculate variance for each feature
        variances = np.var(X, axis=0)
        
        # Find non-constant features
        non_constant_mask = variances > self.config.variance_threshold
        
        if not non_constant_mask.all():
            removed_count = np.sum(~non_constant_mask)
            self.logger.info(f"🗑️ Removed {removed_count} constant features")
            
            X = X[:, non_constant_mask]
            feature_names = [name for i, name in enumerate(feature_names) if non_constant_mask[i]]
        
        return X, feature_names
    
    def _perform_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: UnifiedFeatureSelectionConfig
    ) -> Dict[str, Any]:
        """Perform feature selection using the specified method."""
        self.logger.info(f"🔍 Performing feature selection with method: {config.primary_method}")
        
        if config.primary_method == "auto":
            method = self._choose_optimal_method(X, y)
        else:
            method = config.primary_method
        
        # Perform selection based on method
        if method == "hybrid":
            return self._hybrid_selection(X, y, feature_names, config)
        elif method == "filter":
            return self._filter_selection(X, y, feature_names, config)
        elif method == "wrapper":
            return self._wrapper_selection(X, y, feature_names, config)
        elif method == "embedded":
            return self._embedded_selection(X, y, feature_names, config)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _choose_optimal_method(self, X: np.ndarray, y: np.ndarray) -> str:
        """Choose optimal feature selection method based on data characteristics."""
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # For small datasets, prefer filter methods
        if n_samples < 1000:
            return 'filter'
        # For high-dimensional data, use embedded methods
        elif n_features > 1000:
            return 'embedded'
        # For moderate datasets, use hybrid approach
        else:
            return 'hybrid'
    
    def _hybrid_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: UnifiedFeatureSelectionConfig
    ) -> Dict[str, Any]:
        """Perform hybrid feature selection combining multiple methods."""
        self.logger.info("🔄 Performing hybrid feature selection")
        
        # Step 1: Filter-based pre-selection
        filter_result = self._filter_selection(X, y, feature_names, config)
        filter_features = filter_result['selected_features']
        
        # Step 2: Wrapper-based refinement
        if len(filter_features) > config.target_features:
            # Get indices of filtered features
            filter_indices = [feature_names.index(feat) for feat in filter_features]
            X_filtered = X[:, filter_indices]
            
            # Apply mRMR method with iteration limits
            mrmr_result = self._mrmr_selection_with_limits(X_filtered, y, filter_features, config.target_features)
            if mrmr_result['selected_features']:
                final_features = mrmr_result['selected_features']
            else:
                # Fallback to wrapper method
                wrapper_result = self._wrapper_selection(X_filtered, y, filter_features, config)
                final_features = wrapper_result['selected_features']
        else:
            final_features = filter_features
        
        # Step 3: Embedded method for final optimization
        if len(final_features) > config.target_features:
            final_indices = [feature_names.index(feat) for feat in final_features]
            X_final = X[:, final_indices]
            
            embedded_result = self._embedded_selection(X_final, y, final_features, config)
            final_features = embedded_result['selected_features']
        
        # Calculate final scores
        final_scores = self._calculate_feature_scores(X, y, final_features, feature_names)
        
        return {
            'selected_features': final_features,
            'feature_scores': final_scores,
            'method': 'hybrid',
            'n_selected': len(final_features),
            'selection_ratio': len(final_features) / len(feature_names)
        }
    
    def _filter_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: UnifiedFeatureSelectionConfig
    ) -> Dict[str, Any]:
        """Perform filter-based feature selection."""
        self.logger.info("📊 Performing filter-based selection")
        
        # Calculate feature scores using multiple methods
        scores = {}
        
        # Mutual information
        if self.config.task_type == "regression":
            mi_scores = mutual_info_regression(X, y)
        else:
            mi_scores = mutual_info_classif(X, y)
        
        # F-statistic
        if self.config.task_type == "regression":
            f_scores, _ = f_regression(X, y)
        else:
            f_scores, _ = f_classif(X, y)
        
        # Correlation with target
        corr_scores = []
        for i in range(X.shape[1]):
            corr, _ = pearsonr(X[:, i], y)
            corr_scores.append(abs(corr))
        
        # Combine scores
        combined_scores = []
        for i in range(len(feature_names)):
            # Normalize scores
            mi_norm = mi_scores[i] / (np.max(mi_scores) + 1e-10)
            f_norm = f_scores[i] / (np.max(f_scores) + 1e-10)
            corr_norm = corr_scores[i]
            
            # Weighted combination
            combined_score = 0.4 * mi_norm + 0.3 * f_norm + 0.3 * corr_norm
            combined_scores.append(combined_score)
        
        # Select top features
        top_indices = np.argsort(combined_scores)[::-1][:config.target_features]
        selected_features = [feature_names[i] for i in top_indices]
        feature_scores = {feature_names[i]: combined_scores[i] for i in top_indices}
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'method': 'filter',
            'n_selected': len(selected_features),
            'selection_ratio': len(selected_features) / len(feature_names)
        }
    
    def _wrapper_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: UnifiedFeatureSelectionConfig
    ) -> Dict[str, Any]:
        """Perform wrapper-based feature selection using RFE."""
        self.logger.info("🔄 Performing wrapper-based selection")
        
        # Choose base estimator
        if self.config.task_type == "regression":
            estimator = RandomForestRegressor(n_estimators=100, random_state=config.random_state)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=config.random_state)
        
        # Perform RFE
        rfe = RFE(estimator=estimator, n_features_to_select=config.target_features)
        rfe.fit(X, y)
        
        # Get selected features
        selected_indices = np.where(rfe.support_)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Calculate feature scores (ranking-based)
        feature_scores = {feature_names[i]: 1.0 / rfe.ranking_[i] for i in selected_indices}
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'method': 'wrapper',
            'n_selected': len(selected_features),
            'selection_ratio': len(selected_features) / len(feature_names)
        }
    
    def _embedded_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: UnifiedFeatureSelectionConfig
    ) -> Dict[str, Any]:
        """Perform embedded-based feature selection."""
        self.logger.info("🔧 Performing embedded-based selection")
        
        # Choose estimator
        if self.config.task_type == "regression":
            estimator = LassoCV(
                cv=config.cv_folds, 
                random_state=config.random_state,
                max_iter=self.config.lasso_max_iterations
            )
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=config.random_state)
        
        # Use SelectFromModel
        selector = SelectFromModel(estimator=estimator, max_features=config.target_features)
        selector.fit(X, y)
        
        # Get selected features
        selected_indices = np.where(selector.get_support())[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Get feature importances
        if hasattr(selector.estimator_, 'coef_'):
            importances = np.abs(selector.estimator_.coef_)
        elif hasattr(selector.estimator_, 'feature_importances_'):
            importances = selector.estimator_.feature_importances_
        else:
            importances = np.ones(X.shape[1])
        
        feature_scores = {feature_names[i]: importances[i] for i in selected_indices}
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'method': 'embedded',
            'n_selected': len(selected_features),
            'selection_ratio': len(selected_features) / len(feature_names)
        }
    
    def _perform_hmm_regime_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Perform HMM regime-specific feature selection."""
        self.logger.info("🎯 Performing HMM regime-specific selection")
        
        # For HMM regime prediction, we want features that are good at distinguishing regimes
        # Use classification-based methods with regime-specific considerations
        
        # Create a classification config
        hmm_config = UnifiedFeatureSelectionConfig(
            target_features=100,
            task_type="classification",
            prediction_target="hmm_regime",
            primary_method="hybrid"
        )
        
        # Perform selection
        result = self._perform_feature_selection(X, y, feature_names, hmm_config)
        
        # Add regime-specific analysis
        result['regime_analysis'] = self._analyze_regime_features(X, y, result['selected_features'], feature_names)
        
        return result
    
    def _analyze_regime_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selected_features: List[str],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze how well features distinguish between regimes."""
        # Get feature indices
        feature_indices = [feature_names.index(feat) for feat in selected_features]
        X_selected = X[:, feature_indices]
        
        # Calculate regime separation metrics
        unique_regimes = np.unique(y)
        regime_separation = {}
        
        for i, feature_idx in enumerate(feature_indices):
            feature_name = selected_features[i]
            feature_values = X_selected[:, i]
            
            # Calculate separation between regimes
            regime_means = {}
            regime_stds = {}
            
            for regime in unique_regimes:
                regime_mask = y == regime
                regime_values = feature_values[regime_mask]
                regime_means[regime] = np.mean(regime_values)
                regime_stds[regime] = np.std(regime_values)
            
            # Calculate separation score
            separation_score = 0
            for regime1 in unique_regimes:
                for regime2 in unique_regimes:
                    if regime1 != regime2:
                        mean_diff = abs(regime_means[regime1] - regime_means[regime2])
                        std_combined = np.sqrt(regime_stds[regime1]**2 + regime_stds[regime2]**2)
                        if std_combined > 0:
                            separation_score += mean_diff / std_combined
            
            regime_separation[feature_name] = separation_score
        
        return {
            'regime_separation_scores': regime_separation,
            'unique_regimes': unique_regimes.tolist(),
            'n_regimes': len(unique_regimes)
        }
    
    def _calculate_feature_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selected_features: List[str],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate comprehensive feature scores."""
        scores = {}
        
        for feature in selected_features:
            if feature in feature_names:
                idx = feature_names.index(feature)
                feature_values = X[:, idx]
                
                # Calculate multiple score types
                if self.config.task_type == "regression":
                    # Correlation with target
                    corr, _ = pearsonr(feature_values, y)
                    scores[feature] = abs(corr)
                else:
                    # Mutual information for classification
                    mi = mutual_info_classif(feature_values.reshape(-1, 1), y)[0]
                    scores[feature] = mi
        
        return scores
    
    def _save_results(self):
        """Save feature selection results."""
        if not self.config.save_results:
            return
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save feature sets
        feature_sets_file = output_dir / f"feature_sets_{timestamp}.json"
        with open(feature_sets_file, 'w') as f:
            json.dump(self.feature_sets, f, indent=2)
        
        # Save feature scores
        scores_file = output_dir / f"feature_scores_{timestamp}.json"
        with open(scores_file, 'w') as f:
            json.dump(self.feature_scores, f, indent=2)
        
        # Save full results
        results_file = output_dir / f"full_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"💾 Results saved to {output_dir}")
    
    def get_feature_set(self, size: int) -> List[str]:
        """Get feature set of specified size."""
        key = f'top_{size}'
        if key in self.feature_sets:
            return self.feature_sets[key]
        else:
            self.logger.warning(f"⚠️ Feature set {key} not found")
            return []
    
    def get_hmm_regime_features(self) -> List[str]:
        """Get HMM regime-specific features."""
        if 'hmm_regime_top_100' in self.feature_sets:
            return self.feature_sets['hmm_regime_top_100']
        else:
            self.logger.warning("⚠️ HMM regime features not found")
            return []
    
    def get_feature_scores(self, size: int) -> Dict[str, float]:
        """Get feature scores for specified size."""
        key = f'top_{size}'
        if key in self.feature_scores:
            return self.feature_scores[key]
        else:
            self.logger.warning(f"⚠️ Feature scores for {key} not found")
            return {}


# Convenience functions for easy usage
def create_unified_selector(config: Optional[UnifiedFeatureSelectionConfig] = None) -> UnifiedFeatureSelector:
    """Create a unified feature selector instance."""
    return UnifiedFeatureSelector(config)


def select_features_unified(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List],
    feature_names: Optional[List[str]] = None,
    target_features: int = 120,
    task_type: str = "regression",
    prediction_target: str = "price"
) -> Dict[str, Any]:
    """
    Convenience function for unified feature selection.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        target_features: Number of features to select
        task_type: Type of task ("regression" or "classification")
        prediction_target: What to predict ("price" or "hmm_regime")
        
    Returns:
        Dictionary containing feature selection results
    """
    config = UnifiedFeatureSelectionConfig(
        target_features=target_features,
        task_type=task_type,
        prediction_target=prediction_target
    )
    
    selector = UnifiedFeatureSelector(config)
    return selector.select_features(X, y, feature_names)


def generate_feature_sets(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List],
    feature_names: Optional[List[str]] = None,
    target_sizes: List[int] = [120, 100, 80, 60],
    task_type: str = "regression",
    prediction_target: str = "price"
) -> Dict[str, List[str]]:
    """
    Generate multiple feature sets of different sizes.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        target_sizes: List of target feature set sizes
        task_type: Type of task
        prediction_target: What to predict
        
    Returns:
        Dictionary mapping size names to feature lists
    """
    config = UnifiedFeatureSelectionConfig(
        task_type=task_type,
        prediction_target=prediction_target
    )
    
    selector = UnifiedFeatureSelector(config)
    results = selector.select_features(X, y, feature_names, target_sizes)
    
    return selector.feature_sets


# Export key classes and functions
__all__ = [
    'UnifiedFeatureSelector',
    'UnifiedFeatureSelectionConfig',
    'create_unified_selector',
    'select_features_unified',
    'generate_feature_sets'
]