"""
Enhanced Regime Feature Selector

This module provides an advanced regime feature selection system that integrates:
- TreeSHAP feature selection as the base method
- tprint utilities for logging and data preview
- VectorBTRollingOptimizer and UnifiedVectorizationManager for vectorized computations
- Hardware optimizations for M1 systems
- ML common utilities for HPO, SHAP/LIME, time series validation, and data leakage prevention

Author: AI Assistant
Date: 2024
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

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
    
    # TreeSHAP specific parameters
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
    techniques and hardware acceleration.
    """
    
    def __init__(self, config: Optional[EnhancedRegimeFeatureSelectorConfig] = None):
        """Initialize the Enhanced Regime Feature Selector."""
        super().__init__(step_name="enhanced_regime_feature_selection")
        
        self.config = config or EnhancedRegimeFeatureSelectorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_metrics = {}
        self.feature_importance_cache = {}
        
        if self.config.verbose:
            tprint_success("Enhanced Regime Feature Selector initialized successfully")
    
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Initialize TreeSHAP feature selector
            if TREESHAP_AVAILABLE and TreeSHAPFeatureSelector:
                treeshap_config = self.config.treeshap_params or {}
                self.treeshap_selector = TreeSHAPFeatureSelector(treeshap_config)
                tprint_info("TreeSHAP feature selector initialized")
            else:
                self.treeshap_selector = None
                tprint_warning("TreeSHAP not available, using fallback methods")
            
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
        target: pd.Series,
        regime_labels: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select features using the enhanced regime feature selection pipeline.
        
        Args:
            features_df: DataFrame containing features
            target: Target variable series
            regime_labels: Optional regime labels for regime-specific selection
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary containing selected features and metadata
        """
        try:
            tprint_info("Starting enhanced regime feature selection")
            tprint_data_preview(f"Features shape: {features_df.shape}")
            tprint_data_preview(f"Target shape: {target.shape}")
            
            # Data validation and preprocessing
            features_df, target = self._validate_and_preprocess_data(features_df, target)
            
            # Hardware optimization setup
            if self.hardware_manager:
                self._setup_hardware_optimization()
            
            # Feature selection pipeline
            selection_results = self._run_feature_selection_pipeline(
                features_df, target, regime_labels, feature_names
            )
            
            # Performance tracking
            self._track_performance(selection_results)
            
            tprint_success("Feature selection completed successfully")
            return selection_results
            
        except Exception as e:
            tprint_error(f"Feature selection failed: {e}")
            self.logger.error(f"Feature selection error: {e}")
            raise
    
    def _validate_and_preprocess_data(
        self,
        features_df: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and preprocess input data."""
        try:
            # Basic validation
            if features_df.empty or target.empty:
                raise ValueError("Input data cannot be empty")
            
            if len(features_df) != len(target):
                raise ValueError("Features and target must have the same length")
            
            # Handle missing values
            if features_df.isnull().any().any():
                tprint_warning("Missing values detected, filling with median")
                features_df = features_df.fillna(features_df.median())
            
            if target.isnull().any():
                tprint_warning("Missing target values detected, dropping rows")
                valid_indices = ~target.isnull()
                features_df = features_df[valid_indices]
                target = target[valid_indices]
            
            # Data format logging
            tprint_data_format(f"Features dtype: {features_df.dtypes.value_counts().to_dict()}")
            tprint_data_format(f"Target dtype: {target.dtype}")
            
            return features_df, target
            
        except Exception as e:
            tprint_error(f"Data validation failed: {e}")
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
    
    def _run_feature_selection_pipeline(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        regime_labels: Optional[pd.Series],
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run the complete feature selection pipeline."""
        try:
            results = {
                'selected_features': [],
                'feature_importance': {},
                'selection_metadata': {},
                'performance_metrics': {},
                'regime_specific_results': {}
            }
            
            # Data leakage detection
            if self.leakage_detector:
                leakage_results = self.leakage_detector.detect_leakage(
                    features_df, target
                )
                results['leakage_detection'] = leakage_results
                tprint_info(f"Data leakage detection completed: {leakage_results}")
            
            # Main feature selection
            if self.treeshap_selector:
                selection_results = self._run_treeshap_selection(
                    features_df, target, feature_names
                )
                results.update(selection_results)
            else:
                # Fallback to basic feature selection
                selection_results = self._run_basic_selection(
                    features_df, target, feature_names
                )
                results.update(selection_results)
            
            # Regime-specific feature selection
            if regime_labels is not None:
                regime_results = self._run_regime_specific_selection(
                    features_df, target, regime_labels, feature_names
                )
                results['regime_specific_results'] = regime_results
            
            # Feature importance analysis
            if self.explainability_tool:
                importance_analysis = self._analyze_feature_importance(
                    features_df, target, results['selected_features']
                )
                results['importance_analysis'] = importance_analysis
            
            # Performance evaluation
            if self.evaluator:
                evaluation_results = self._evaluate_selection_performance(
                    features_df, target, results['selected_features']
                )
                results['evaluation_results'] = evaluation_results
            
            return results
            
        except Exception as e:
            tprint_error(f"Feature selection pipeline failed: {e}")
            raise
    
    def _run_treeshap_selection(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run TreeSHAP-based feature selection."""
        try:
            tprint_info("Running TreeSHAP feature selection")
            
            # Use VectorBT optimization if available
            if self.vectorbt_optimizer and VECTORBT_AVAILABLE:
                # Optimize features using VectorBT
                optimized_features = self._optimize_features_with_vectorbt(
                    features_df, target
                )
            else:
                optimized_features = features_df
            
            # Run TreeSHAP selection
            selection_results = self.treeshap_selector.select_features(
                optimized_features,
                target,
                feature_names=feature_names or list(features_df.columns),
                max_features=self.config.max_features,
                min_importance=self.config.min_feature_importance
            )
            
            tprint_success(f"TreeSHAP selection completed: {len(selection_results.get('selected_features', []))} features selected")
            return selection_results
            
        except Exception as e:
            tprint_error(f"TreeSHAP selection failed: {e}")
            raise
    
    def _run_basic_selection(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run basic feature selection as fallback."""
        try:
            tprint_info("Running basic feature selection")
            
            # Simple correlation-based selection
            correlations = features_df.corrwith(target).abs()
            selected_features = correlations.nlargest(
                min(self.config.max_features, len(correlations))
            ).index.tolist()
            
            return {
                'selected_features': selected_features,
                'feature_importance': correlations.to_dict(),
                'selection_method': 'correlation_based'
            }
            
        except Exception as e:
            tprint_error(f"Basic selection failed: {e}")
            raise
    
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
    
    def _run_regime_specific_selection(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        regime_labels: pd.Series,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run regime-specific feature selection."""
        try:
            tprint_info("Running regime-specific feature selection")
            
            regime_results = {}
            unique_regimes = regime_labels.unique()
            
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_features = features_df[regime_mask]
                regime_target = target[regime_mask]
                
                if len(regime_features) < 10:  # Skip if too few samples
                    continue
                
                # Select features for this regime
                if self.treeshap_selector:
                    regime_selection = self.treeshap_selector.select_features(
                        regime_features,
                        regime_target,
                        feature_names=feature_names or list(features_df.columns),
                        max_features=self.config.max_features // len(unique_regimes)
                    )
                else:
                    # Basic selection for regime
                    correlations = regime_features.corrwith(regime_target).abs()
                    selected_features = correlations.nlargest(
                        min(self.config.max_features // len(unique_regimes), len(correlations))
                    ).index.tolist()
                    regime_selection = {
                        'selected_features': selected_features,
                        'feature_importance': correlations.to_dict()
                    }
                
                regime_results[f'regime_{regime}'] = regime_selection
                tprint_info(f"Regime {regime}: {len(regime_selection.get('selected_features', []))} features selected")
            
            return regime_results
            
        except Exception as e:
            tprint_error(f"Regime-specific selection failed: {e}")
            return {}
    
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


def create_enhanced_regime_feature_selector(
    config: Optional[EnhancedRegimeFeatureSelectorConfig] = None
) -> EnhancedRegimeFeatureSelector:
    """
    Factory function to create an Enhanced Regime Feature Selector.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured EnhancedRegimeFeatureSelector instance
    """
    return EnhancedRegimeFeatureSelector(config)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Create target with some relationship to features
    target = (
        0.3 * features_df.iloc[:, 0] +
        0.2 * features_df.iloc[:, 1] +
        0.1 * features_df.iloc[:, 2] +
        np.random.randn(n_samples) * 0.1
    )
    
    # Create regime labels
    regime_labels = pd.Series(
        np.random.choice([0, 1, 2], n_samples),
        index=features_df.index
    )
    
    # Create selector
    config = EnhancedRegimeFeatureSelectorConfig(
        max_features=20,
        min_feature_importance=0.01,
        verbose=True
    )
    
    selector = create_enhanced_regime_feature_selector(config)
    
    # Run feature selection
    results = selector.select_features(
        features_df=features_df,
        target=target,
        regime_labels=regime_labels
    )
    
    # Print results
    print("\n" + "="*50)
    print("ENHANCED REGIME FEATURE SELECTION RESULTS")
    print("="*50)
    print(f"Selected features: {len(results['selected_features'])}")
    print(f"Selected features: {results['selected_features'][:10]}...")  # Show first 10
    print(f"Performance metrics: {selector.get_performance_metrics()}")
    print("="*50)