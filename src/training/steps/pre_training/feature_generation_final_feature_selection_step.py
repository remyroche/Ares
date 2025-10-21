"""
Feature Generation Final Feature Selection Step

This step performs target-aware feature selection after interaction generation using a
downstream sequential pipeline: PCA → Approximate MI → mRMR → LASSO+Stability → LGBM+RFE+SHAP.

Key Features:
- Target-aware selection for analyst/tactician × long/short combinations
- Generates 3 feature sets per target: 60, 50, 40 features
- End-to-end float32 processing for memory efficiency
- M1 hardware optimizations (GPU, CPU, memory)
- VectorBT integration for vectorized operations
- Computational optimizations: sampling, batching, caching, parallel processing
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import KBinsDiscretizer

# Import enhanced hardware optimization tools
from src.utils.hardware import (
    get_integrated_hardware_manager, IntegratedHardwareManager,
    get_comprehensive_optimizer, M1ComprehensiveOptimizer,
    WorkloadType, OptimizationLevel, WorkloadCategory,
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    optimize_dataframe, optimize_array, cache_result, auto_optimize,
    memory_efficient, performance_tracked, force_cleanup, get_memory_stats
)

# Import base step and artifact management
from src.training.steps.base_step import BaseStep
from src.utils.artifact_manager import get_step_context_from_config, setup_enhanced_artifact_manager

# LightGBM and SHAP
try:
    import lightgbm as lgb
    import shap
    LGBM_SHAP_AVAILABLE = True
except ImportError:
    LGBM_SHAP_AVAILABLE = False
    warnings.warn("LightGBM/SHAP not available. Install with: pip install lightgbm shap")

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_data_preview, tprint_data_format, tprint_performance, tprint_progress,
        tprint_structured, tprint_timer, tprint_exception, LogLevel
    )
except ImportError:
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_data_preview(*args, **kwargs): pass  # No-op fallback
    def tprint_data_format(*args, **kwargs): return None  # No-op fallback
    def tprint_performance(*args, **kwargs): pass  # No-op fallback
    def tprint_progress(*args, **kwargs): pass  # No-op fallback
    def tprint_structured(*args, **kwargs): pass  # No-op fallback
    def tprint_timer(*args, **kwargs): return lambda f: f  # No-op fallback
    def tprint_exception(*args, **kwargs): pass  # No-op fallback
    class LogLevel:
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        SUCCESS = "SUCCESS"
        PERFORMANCE = "PERFORMANCE"

# Import artifact management functions
from src.utils.artifact_manager import ArtifactManager

# Define ArtifactKeys for pre-training steps
class ArtifactKeys:
    """Artifact keys for pre-training steps."""
    GENERATED_FEATURES = "generated_features"
    INTERACTION_FEATURES = "interaction_features"
    INTERACTION_METADATA = "interaction_metadata"
    INTERACTION_GENERATION_METRICS = "interaction_generation_metrics"
    FEATURE_DATAFRAME = "feature_dataframe"
    TARGETS = "targets"
    SELECTED_FEATURES = "selected_features"
    MI_BEST_LOOKBACKS_PER_FEATURE = "mi_best_lookbacks_per_feature"
    MRMR_TOP_LOOKBACKS_PER_FEATURE = "mrmr_top_lookbacks_per_feature"
    MI_SCORES_BY_FEATURE = "mi_scores_by_feature"
    OOS_SHARPE_BY_FEATURE_WINDOW = "oos_sharpe_by_feature_window"
    SELECTED_FEATURES_METADATA = "selected_features_metadata"
    FAMILY_DIAGNOSTICS = "family_diagnostics"
    OPTIMIZATION_CONFIG = "optimization_config"
    OPTIMIZED_FEATURE_DATAFRAME = "optimized_feature_dataframe"

def get_pretraining_artifact_manager() -> ArtifactManager:
    """Get pre-training artifact manager with enhanced hardware optimization."""
    from src.utils.hardware import get_integrated_hardware_manager
    
    # Create artifact manager with hardware optimization
    config = {
        'base_dir': 'artifacts/pre_training',
        'compression': 'auto',
        'memory_optimization': True,
        'hardware_optimization': True
    }
    
    manager = ArtifactManager(config=config)
    
    # Integrate with hardware manager for enhanced performance
    hardware_manager = get_integrated_hardware_manager()
    manager._hardware_manager = hardware_manager
    
    return manager


logger = logging.getLogger(__name__)


@dataclass
class FinalFeatureSelectionResult:
    """Result from final feature selection."""
    success: bool
    selected_features_60: List[str]
    selected_features_50: List[str]
    selected_features_40: List[str]
    selected_feature_dataframe_60: pd.DataFrame
    selected_feature_dataframe_50: pd.DataFrame
    selected_feature_dataframe_40: pd.DataFrame
    feature_scores: Dict[str, float]
    shap_values_60: Optional[np.ndarray]
    shap_values_50: Optional[np.ndarray]
    shap_values_40: Optional[np.ndarray]
    selection_metadata: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


@contextmanager
def memory_managed_operation(operation_name: str = "operation"):
    """Context manager for memory-managed operations with enhanced hardware optimization."""
    try:
        tprint_debug(f"🧠 Starting memory-managed operation: {operation_name}")
        yield
    finally:
        # Use enhanced memory cleanup
        force_cleanup()
        tprint_debug(f"🧠 Completed memory-managed operation: {operation_name}, enhanced cleanup run")


class FeatureGenerationFinalFeatureSelectionStep(BaseStep):
    """
    Final feature selection step with target-aware selection using downstream sequential pipeline.
    
    Pipeline: PCA → Approximate MI → mRMR → LASSO+Stability → LGBM+RFE+SHAP
    
    Key Features:
    - End-to-end float32 processing
    - M1 hardware optimizations
    - VectorBT integration
    - Computational optimizations (sampling, batching, caching, parallel processing)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the final feature selection step."""
        tprint_info("🎯 Initializing FeatureGenerationFinalFeatureSelectionStep")
        
        super().__init__("feature_generation_final_feature_selection_step", config)
        
        # Initialize enhanced hardware optimization system
        tprint_info("🚀 Initializing enhanced hardware optimization system")
        self.hardware_manager = get_integrated_hardware_manager()
        self.comprehensive_optimizer = get_comprehensive_optimizer()
        
        # Configure for feature selection workload
        self.hardware_manager.optimize_for_workload(
            WorkloadType.FEATURE_ENGINEERING, 
            OptimizationLevel.AGGRESSIVE
        )
        
        tprint_success("✅ Enhanced hardware optimization system initialized")
        
        # Configuration
        self.chunk_size = 50000  # Row chunk size
        self.feature_batch_size = 50  # Feature batch size
        self.aggressive_gc = True
        self.gc_frequency = 10  # GC every 10 operations
        self.operation_counter = 0
        
        # MI configuration
        self.mi_method = 'sklearn_knn'
        self.mi_neighbors = 3
        self.mi_pre_k = 200
        self.mi_max_rows = 100000
        self.mi_sample_ratio = 0.25
        self.mi_quantile = 0.80
        
        # mRMR configuration
        self.mrmr_batch_size = 10
        self.mrmr_sample_size = 20  # Sample size for approximate redundancy
        self.mrmr_early_stop_threshold = 0.01  # 1% improvement threshold
        self.mrmr_early_stop_patience = 3
        
        # LASSO configuration
        self.n_bootstrap = 50
        self.stability_threshold = 0.7
        self.bootstrap_sample_ratio = 0.8
        self.lasso_cv_folds = 3
        self.lasso_n_alphas = 20
        self.lasso_max_iter = 500
        self.lasso_tol = 1e-3
        
        # LGBM configuration
        self.lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_bin': 255,  # Reduced for speed
            'min_data_in_leaf': 20,
            'num_threads': -1,
            'force_col_wise': True,
            'max_depth': 7,
            'min_gain_to_split': 0.01,
            'histogram_pool_size': 512,
            'verbose': -1
        }
        self.lgbm_num_boost_round = 100
        self.lgbm_early_stopping_rounds = 10
        
        # SHAP configuration
        self.shap_sample_size = 1000
        self.shap_batch_size = 100
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'stage_times': {},
            'memory_optimizations_applied': 0,
            'gc_runs': 0,
            'hardware_optimizations_applied': 0
        }
        
        tprint_success("✅ FeatureGenerationFinalFeatureSelectionStep initialized")
    
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            self.set_state('initialized_at', datetime.now().isoformat())
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize resources: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources using enhanced hardware tools."""
        self.set_state('cleaned_up_at', datetime.now().isoformat())
        # Use enhanced cleanup
        force_cleanup()
        # Clear hardware manager caches if needed
        if hasattr(self, 'hardware_manager'):
            self.hardware_manager.clear_all_caches()
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        return data
    
    def _check_memory_and_gc(self):
        """Check memory usage and run GC if needed using enhanced hardware tools."""
        self.operation_counter += 1
        
        if self.aggressive_gc and self.operation_counter % self.gc_frequency == 0:
            # Use enhanced memory management
            force_cleanup()
            self.performance_stats['gc_runs'] += 1
            self.performance_stats['hardware_optimizations_applied'] += 1
            
            # Get memory stats from hardware manager
            memory_stats = get_memory_stats()
            
            # Enhanced troubleshooting: Log memory statistics
            tprint_structured({
                'operation': 'memory_check',
                'operation_counter': self.operation_counter,
                'gc_runs': self.performance_stats['gc_runs'],
                'memory_stats': memory_stats,
                'timestamp': datetime.now().isoformat()
            }, level=LogLevel.DEBUG)
            
            if memory_stats.get('used_memory', 0) > 4096:  # > 4GB
                force_cleanup()
                tprint_warning(f"⚠️ High memory usage: {memory_stats.get('used_memory', 0):.0f}MB, forcing full cleanup")
    
    def _convert_to_float32(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame to float32 for memory efficiency using enhanced hardware tools."""
        tprint_debug("🔄 Converting data to float32 with enhanced optimization")
        
        # Use enhanced hardware optimization for dataframe conversion
        optimized_df = optimize_dataframe(df)
        
        # Additional float32 conversion for numeric columns
        numeric_cols = optimized_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if optimized_df[col].dtype == np.float64:
                optimized_df[col] = optimized_df[col].astype('float32', copy=False)
        
        self.performance_stats['memory_optimizations_applied'] += 1
        self.performance_stats['hardware_optimizations_applied'] += 1
        return optimized_df
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final feature selection step.
        
        Args:
            config: Configuration dictionary containing all necessary parameters
                   (symbol, exchange, timeframes, execution_mode, model_type, direction, etc.)
        
        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': list of artifact paths/metadata created
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
            - 'execution_time': float seconds taken to execute
        """
        start_time = time.time()
        
        # Extract parameters from config
        model_type = config.get('model_type', 'analyst')
        direction = config.get('direction', 'long')
        symbol = config.get('symbol', 'ETHUSDT')
        timeframe = config.get('timeframe', '15m')
        
        # Enhanced troubleshooting: Log configuration details
        tprint_structured({
            'step': 'feature_generation_final_feature_selection_step',
            'model_type': model_type,
            'direction': direction,
            'symbol': symbol,
            'timeframe': timeframe,
            'config_keys': list(config.keys()) if config else [],
            'timestamp': datetime.now().isoformat()
        }, level=LogLevel.INFO)
        
        tprint_info(f"🎯 Starting final feature selection for {model_type}_{direction}")
        
        # Set up enhanced artifact manager with context
        context = get_step_context_from_config(config)
        context.update({
            'symbol': symbol,
            'exchange': config.get('exchange', 'binance'),
            'direction': direction,
            'model': model_type.title()  # Convert to proper case
        })
        am = setup_enhanced_artifact_manager(**context)
        
        try:
            # Get artifact manager
            artifact_manager = am
            
            # Load artifacts from previous steps
            tprint_info("📦 Loading artifacts from previous steps")
            with tprint_timer("artifact_loading", level=LogLevel.PERFORMANCE):
                features_df, targets = await self._load_artifacts(artifact_manager, model_type, direction)
            
            if features_df is None or targets is None:
                raise ValueError("Failed to load required artifacts from previous steps")
            
            # Enhanced troubleshooting: Data format analysis
            tprint_data_format(features_df, "loaded_features", level=LogLevel.DEBUG)
            tprint_data_format(targets, "loaded_targets", level=LogLevel.DEBUG)
            
            # Data preview after loading
            tprint_data_preview(features_df, "loaded_features", max_rows=3, max_cols=10)
            tprint_data_preview(targets, "loaded_targets", max_rows=10)
            
            # Convert to float32 end-to-end with enhanced optimization
            tprint_info("🔄 Converting data to float32 (end-to-end with enhanced optimization)")
            with tprint_timer("float32_conversion", level=LogLevel.PERFORMANCE):
                features_df = self._convert_to_float32(features_df)
                targets = targets.astype('float32', copy=False)
            
            # Enhanced troubleshooting: Data format analysis after conversion
            tprint_data_format(features_df, "features_float32_converted", level=LogLevel.DEBUG)
            tprint_data_format(targets, "targets_float32_converted", level=LogLevel.DEBUG)
            
            # Data preview after float32 conversion
            tprint_data_preview(features_df, "features_float32_converted", max_rows=3, max_cols=10)
            tprint_data_preview(targets, "targets_float32_converted", max_rows=10)
            
            # Apply comprehensive hardware optimization
            tprint_info("🚀 Applying comprehensive hardware optimization")
            with tprint_timer("hardware_optimization", level=LogLevel.PERFORMANCE):
                features_df = self.hardware_manager.process_data_with_optimization(
                    features_df, WorkloadType.FEATURE_ENGINEERING
                )
                self.performance_stats['hardware_optimizations_applied'] += 1
            
            # Enhanced troubleshooting: Data format analysis after optimization
            tprint_data_format(features_df, "features_hardware_optimized", level=LogLevel.DEBUG)
            
            # Data preview after hardware optimization
            tprint_data_preview(features_df, "features_hardware_optimized", max_rows=3, max_cols=10)
            
            tprint_success(f"✅ Loaded {features_df.shape[1]} features and {len(targets)} targets")
            tprint_info(f"📊 Feature matrix: {features_df.shape}, Target: {targets.shape}")
            tprint_info(f"💾 Memory usage: Features={features_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")
            
            # Stage 1: PCA + Approximate MI (→ ~200 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 1: PCA + Approximate MI Filter")
            stage1_start = time.time()
            
            with tprint_timer("stage1_pca_mi", level=LogLevel.PERFORMANCE):
                features_200 = await self._stage1_pca_mi_filter(features_df, targets)
            
            self.performance_stats['stage_times']['stage1_pca_mi'] = time.time() - stage1_start
            tprint_performance("Stage 1: PCA + MI Filter", time.time() - stage1_start)
            tprint_success(f"✅ Stage 1 complete: {len(features_200)} features selected")
            
            # Enhanced troubleshooting: Data format analysis after Stage 1
            tprint_data_format(pd.DataFrame(features_df[features_200]), "stage1_selected_features", level=LogLevel.DEBUG)
            
            self._check_memory_and_gc()
            
            # Stage 2: Ultra-optimized mRMR (200 → 150 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 2: Ultra-Optimized mRMR Selection")
            stage2_start = time.time()
            
            with tprint_timer("stage2_mrmr", level=LogLevel.PERFORMANCE):
                features_150 = await self._stage2_mrmr_selection(
                    features_df[features_200], targets
                )
            
            self.performance_stats['stage_times']['stage2_mrmr'] = time.time() - stage2_start
            tprint_performance("Stage 2: mRMR Selection", time.time() - stage2_start)
            tprint_success(f"✅ Stage 2 complete: {len(features_150)} features selected")
            
            # Enhanced troubleshooting: Data format analysis after Stage 2
            tprint_data_format(pd.DataFrame(features_df[features_150]), "stage2_selected_features", level=LogLevel.DEBUG)
            
            self._check_memory_and_gc()
            
            # Stage 3: LASSO + Stability Selection (150 → 100 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 3: LASSO + Stability Selection")
            stage3_start = time.time()
            
            with tprint_timer("stage3_lasso_stability", level=LogLevel.PERFORMANCE):
                features_100 = await self._stage3_lasso_stability(
                    features_df[features_150], targets
                )
            
            self.performance_stats['stage_times']['stage3_lasso_stability'] = time.time() - stage3_start
            tprint_performance("Stage 3: LASSO + Stability Selection", time.time() - stage3_start)
            tprint_success(f"✅ Stage 3 complete: {len(features_100)} features selected")
            
            # Enhanced troubleshooting: Data format analysis after Stage 3
            tprint_data_format(pd.DataFrame(features_df[features_100]), "stage3_selected_features", level=LogLevel.DEBUG)
            
            self._check_memory_and_gc()
            
            # Stage 4: LGBM + RFE + SHAP (100 → 60/50/40 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 4: LGBM + RFE + SHAP")
            stage4_start = time.time()
            
            with tprint_timer("stage4_lgbm_rfe_shap", level=LogLevel.PERFORMANCE):
                result_dict = await self._stage4_lgbm_rfe_shap(
                    features_df[features_100], targets
                )
            
            self.performance_stats['stage_times']['stage4_lgbm_rfe_shap'] = time.time() - stage4_start
            tprint_performance("Stage 4: LGBM + RFE + SHAP", time.time() - stage4_start)
            tprint_success("✅ Stage 4 complete: Generated 3 feature sets (60, 50, 40)")
            
            # Enhanced troubleshooting: Data format analysis after Stage 4
            tprint_data_format(pd.DataFrame(features_df[result_dict['features_60']]), "stage4_features_60", level=LogLevel.DEBUG)
            tprint_data_format(pd.DataFrame(features_df[result_dict['features_50']]), "stage4_features_50", level=LogLevel.DEBUG)
            tprint_data_format(pd.DataFrame(features_df[result_dict['features_40']]), "stage4_features_40", level=LogLevel.DEBUG)
            
            self._check_memory_and_gc()
            
            # Create result
            execution_time = time.time() - start_time
            self.performance_stats['total_processing_time'] = execution_time
            
            result = FinalFeatureSelectionResult(
                success=True,
                selected_features_60=result_dict['features_60'],
                selected_features_50=result_dict['features_50'],
                selected_features_40=result_dict['features_40'],
                selected_feature_dataframe_60=features_df[result_dict['features_60']],
                selected_feature_dataframe_50=features_df[result_dict['features_50']],
                selected_feature_dataframe_40=features_df[result_dict['features_40']],
                feature_scores=result_dict['feature_scores'],
                shap_values_60=result_dict.get('shap_values_60'),
                shap_values_50=result_dict.get('shap_values_50'),
                shap_values_40=result_dict.get('shap_values_40'),
                selection_metadata={
                    'model_type': model_type,
                    'direction': direction,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'initial_features': features_df.shape[1],
                    'stage1_features': len(features_200),
                    'stage2_features': len(features_150),
                    'stage3_features': len(features_100),
                    'final_features_60': len(result_dict['features_60']),
                    'final_features_50': len(result_dict['features_50']),
                    'final_features_40': len(result_dict['features_40']),
                    'performance_stats': self.performance_stats,
                    'created_at': datetime.now().isoformat()
                },
                execution_time=execution_time
            )
            
            # Save artifacts
            tprint_info("💾 Saving artifacts")
            
            # Data preview before saving artifacts
            tprint_data_preview(result.selected_feature_dataframe_60, "final_features_60", max_rows=3, max_cols=10)
            tprint_data_preview(result.selected_feature_dataframe_50, "final_features_50", max_rows=3, max_cols=10)
            tprint_data_preview(result.selected_feature_dataframe_40, "final_features_40", max_rows=3, max_cols=10)
            tprint_data_preview(pd.Series(result.feature_scores), "final_feature_scores", max_rows=20)
            
            await self._save_artifacts(artifact_manager, result, model_type, direction)
            
            tprint_success("=" * 80)
            tprint_success(f"🎉 Final feature selection completed in {execution_time:.2f}s")
            tprint_success(f"📊 Feature reduction: {features_df.shape[1]} → 60/50/40")
            tprint_success(f"💾 Total GC runs: {self.performance_stats['gc_runs']}")
            tprint_success(f"🚀 Hardware optimizations applied: {self.performance_stats['hardware_optimizations_applied']}")
            
            # Final cleanup
            force_cleanup()
            
            # Convert result to BaseStep format
            return {
                'success': True,
                'artifacts': [
                    f'selected_features_60_{model_type}_{direction}',
                    f'selected_features_50_{model_type}_{direction}',
                    f'selected_features_40_{model_type}_{direction}',
                    f'feature_scores_{model_type}_{direction}',
                    f'shap_values_{model_type}_{direction}'
                ],
                'metrics': {
                    'execution_time': execution_time,
                    'performance_stats': self.performance_stats,
                    'feature_reduction': f"{features_df.shape[1]} → 60/50/40",
                    'gc_runs': self.performance_stats['gc_runs'],
                    'hardware_optimizations': self.performance_stats['hardware_optimizations_applied']
                },
                'execution_time': execution_time,
                'result_data': result  # Store the actual result for internal use
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Final feature selection failed: {e}")
            tprint_exception(e, "Final feature selection failed")
            
            # Enhanced troubleshooting: Log performance stats even on failure
            tprint_structured({
                'error': str(e),
                'execution_time': execution_time,
                'performance_stats': self.performance_stats,
                'stage_times': self.performance_stats.get('stage_times', {}),
                'memory_optimizations': self.performance_stats.get('memory_optimizations_applied', 0),
                'gc_runs': self.performance_stats.get('gc_runs', 0)
            }, level=LogLevel.ERROR)
            
            self.logger.error(f"Final feature selection failed: {e}", exc_info=True)
            
            return {
                'success': False,
                'artifacts': [],
                'metrics': {
                    'execution_time': execution_time,
                    'error': str(e)
                },
                'execution_time': execution_time,
                'error': str(e)
            }
    
    async def _load_artifacts(self, artifact_manager, model_type: str, direction: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load features and targets from previous steps."""
        try:
            # Enhanced troubleshooting: Log loading parameters
            tprint_structured({
                'operation': 'load_artifacts',
                'model_type': model_type,
                'direction': direction,
                'timestamp': datetime.now().isoformat()
            }, level=LogLevel.DEBUG)
            
            # Load original features from feature_generation_feature_generation_step
            tprint_info("📦 Loading original features from feature_generation_feature_generation_step")
            original_features = artifact_manager.get_dataframe(
                'feature_generation_feature_generation_step',
                ArtifactKeys.GENERATED_FEATURES
            )
            
            # Enhanced troubleshooting: Data format analysis after loading original features
            if original_features is not None:
                tprint_data_format(original_features, "original_features_loaded", level=LogLevel.DEBUG)
                tprint_data_preview(original_features, "original_features_loaded", max_rows=3, max_cols=5)
            
            # Load periods/lookbacks (top1) from feature_generation_period_lookback_optimization_step
            tprint_info("📦 Loading top periods/lookbacks from period_lookback_optimization_step")
            top_periods = artifact_manager.get_artifact(
                'feature_generation_period_lookback_optimization_step',
                'top_periods'
            )
            top_lookbacks = artifact_manager.get_artifact(
                'feature_generation_period_lookback_optimization_step',
                'top_lookbacks'
            )
            
            # Load interaction features from both analyst and tactician interaction generation steps
            tprint_info("📦 Loading interaction features from interaction_generation_step_analyst")
            analyst_interaction_features = artifact_manager.get_dataframe(
                'feature_generation_interaction_generation_step_analyst',
                ArtifactKeys.INTERACTION_FEATURES
            )
            
            # Enhanced troubleshooting: Data format analysis after loading analyst interaction features
            if analyst_interaction_features is not None:
                tprint_data_format(analyst_interaction_features, "analyst_interactions_loaded", level=LogLevel.DEBUG)
                tprint_data_preview(analyst_interaction_features, "analyst_interactions_loaded", max_rows=3, max_cols=5)
            
            tprint_info("📦 Loading interaction features from interaction_generation_step_tactician")
            tactician_interaction_features = artifact_manager.get_dataframe(
                'feature_generation_interaction_generation_step_tactician',
                ArtifactKeys.INTERACTION_FEATURES
            )
            
            # Enhanced troubleshooting: Data format analysis after loading tactician interaction features
            if tactician_interaction_features is not None:
                tprint_data_format(tactician_interaction_features, "tactician_interactions_loaded", level=LogLevel.DEBUG)
                tprint_data_preview(tactician_interaction_features, "tactician_interactions_loaded", max_rows=3, max_cols=5)
            
            # Merge features from all sources
            features_to_merge = []
            
            if original_features is not None:
                features_to_merge.append(("original", original_features))
                tprint_success(f"✅ Original features: {original_features.shape}")
            
            if analyst_interaction_features is not None:
                features_to_merge.append(("analyst_interactions", analyst_interaction_features))
                tprint_success(f"✅ Analyst interaction features: {analyst_interaction_features.shape}")
            
            if tactician_interaction_features is not None:
                features_to_merge.append(("tactician_interactions", tactician_interaction_features))
                tprint_success(f"✅ Tactician interaction features: {tactician_interaction_features.shape}")
            
            if not features_to_merge:
                tprint_error("❌ No features available from previous steps")
                return None, None
            
            # Merge all available features
            if len(features_to_merge) == 1:
                features_df = features_to_merge[0][1]
                tprint_warning(f"⚠️ Only {features_to_merge[0][0]} features available")
            else:
                tprint_info(f"🔗 Merging {len(features_to_merge)} feature sets")
                # Find common index across all features
                common_idx = features_to_merge[0][1].index
                for name, df in features_to_merge[1:]:
                    common_idx = common_idx.intersection(df.index)
                
                if len(common_idx) == 0:
                    tprint_error("❌ No common timestamps across feature sets")
                    return None, None
                
                # Merge features on common index
                merged_features = []
                for name, df in features_to_merge:
                    merged_features.append(df.loc[common_idx])
                    tprint_info(f"   - {name}: {df.loc[common_idx].shape}")
                
                features_df = pd.concat(merged_features, axis=1)
                tprint_success(f"✅ Merged features: {features_df.shape}")
                
                # Enhanced troubleshooting: Data format analysis after merging features
                tprint_data_format(features_df, "merged_features", level=LogLevel.DEBUG)
                
                # Data preview after merging features
                tprint_data_preview(features_df, "merged_features", max_rows=3, max_cols=10)
            
            # Load targets from labeling_integration_step
            tprint_info("📦 Loading targets from labeling_integration_step")
            targets_artifact = artifact_manager.get_artifact(
                'feature_generation_labeling_integration_step',
                'targets'
            )
            
            if targets_artifact is None:
                tprint_error("❌ Targets not found in labeling_integration_step")
                return None, None
            
            # Extract target for specific model_type and direction
            target_name = f"{model_type}_{direction}_target"
            tprint_info(f"🎯 Looking for target: {target_name}")
            
            if isinstance(targets_artifact, pd.DataFrame):
                if target_name in targets_artifact.columns:
                    targets = targets_artifact[target_name]
                else:
                    # Fallback to first numeric column
                    tprint_warning(f"⚠️ Target {target_name} not found, using first numeric column")
                    targets = targets_artifact.select_dtypes(include=[np.number]).iloc[:, 0]
            elif isinstance(targets_artifact, pd.Series):
                targets = targets_artifact
            else:
                tprint_error(f"❌ Unexpected targets type: {type(targets_artifact)}")
                return None, None
            
            # Align features and targets
            common_idx = features_df.index.intersection(targets.index)
            features_df = features_df.loc[common_idx]
            targets = targets.loc[common_idx]
            
            # Drop NaN
            valid_idx = features_df.notna().all(axis=1) & targets.notna()
            features_df = features_df[valid_idx]
            targets = targets[valid_idx]
            
            tprint_success(f"✅ Final aligned data: Features={features_df.shape}, Targets={len(targets)}")
            
            # Enhanced troubleshooting: Data format analysis after alignment and cleaning
            tprint_data_format(features_df, "aligned_features", level=LogLevel.DEBUG)
            tprint_data_format(targets, "aligned_targets", level=LogLevel.DEBUG)
            
            # Data preview after alignment and cleaning
            tprint_data_preview(features_df, "aligned_features", max_rows=3, max_cols=10)
            tprint_data_preview(targets, "aligned_targets", max_rows=10)
            
            return features_df, targets
            
        except Exception as e:
            tprint_error(f"❌ Failed to load artifacts: {e}")
            tprint_exception(e, "Failed to load artifacts")
            self.logger.error(f"Failed to load artifacts: {e}", exc_info=True)
            return None, None
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked(log_performance=True, track_memory=True)
    async def _stage1_pca_mi_filter(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 1: PCA + Approximate MI Filter
        
        Reduces features to ~200 using PCA (98% variance) + MI with target.
        """
        with memory_managed_operation("Stage 1: PCA + MI Filter"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            # Enhanced troubleshooting: Input data analysis
            tprint_data_format(X, "stage1_input_features", level=LogLevel.DEBUG)
            tprint_data_format(y, "stage1_input_targets", level=LogLevel.DEBUG)
            
            # Step 1: PCA dimensionality reduction (98% variance)
            tprint_info("🔍 Step 1.1: PCA dimensionality reduction (98% variance)")
            pca_start = time.time()
            
            # Ensure float32
            X_numeric = X.select_dtypes(include=[np.number]).astype('float32', copy=False)
            
            # Enhanced troubleshooting: Data format analysis before PCA
            tprint_data_format(X_numeric, "X_numeric_for_pca", level=LogLevel.DEBUG)
            
            # Data preview after numeric selection
            tprint_data_preview(X_numeric, "X_numeric_for_pca", max_rows=3, max_cols=10)
            
            # Fit PCA
            pca = PCA(n_components=0.98, random_state=42)
            pca.fit(X_numeric)
            
            # For each component, find feature with highest loading
            pca_selected = set()
            for component in pca.components_:
                top_feature_idx = np.abs(component).argmax()
                pca_selected.add(X_numeric.columns[top_feature_idx])
            
            pca_features = list(pca_selected)
            pca_duration = time.time() - pca_start
            tprint_performance("PCA dimensionality reduction", pca_duration)
            tprint_success(f"✅ PCA selected {len(pca_features)} features in {pca_duration:.2f}s")
            
            # Enhanced troubleshooting: Data format analysis after PCA
            tprint_data_format(pd.DataFrame(X[pca_features]), "pca_selected_features", level=LogLevel.DEBUG)
            
            # Data preview after PCA selection
            tprint_data_preview(pd.DataFrame(X[pca_features]), "pca_selected_features", max_rows=3, max_cols=10)
            
            # Step 2: Approximate MI filter (correlation to target)
            tprint_info("🔍 Step 1.2: Approximate MI filter with target")
            mi_start = time.time()
            
            # Use efficient MI code from advanced_feature_selection.py
            X_pca = X[pca_features].astype('float32', copy=False)
            
            # Pre-prune by correlation to top 200 features
            tprint_info("📊 Pre-pruning by correlation with target")
            corr = X_pca.corrwith(y).abs()
            if len(corr) > self.mi_pre_k:
                keep_cols = corr.sort_values(ascending=False).head(self.mi_pre_k).index
                X_pca = X_pca[keep_cols]
                tprint_info(f"📦 Pre-pruned to {len(keep_cols)} features")
            
            # Row subsampling for large datasets
            if len(X_pca) > self.mi_max_rows:
                tprint_info(f"📦 Row subsampling: {self.mi_max_rows}/{len(X_pca)}")
                X_pca = X_pca.tail(self.mi_max_rows)
                y_sampled = y.tail(self.mi_max_rows)
            else:
                y_sampled = y
            
            # Sampling-based MI for large datasets
            if len(X_pca) > 50000:
                sample_size = int(len(X_pca) * self.mi_sample_ratio)
                sample_indices = np.random.choice(len(X_pca), sample_size, replace=False)
                X_sampled = X_pca.iloc[sample_indices]
                y_sampled = y_sampled.iloc[sample_indices]
                tprint_info(f"📦 Sampling {sample_size} rows for MI calculation")
            else:
                X_sampled = X_pca
            
            # Compute MI in batches with enhanced optimization
            batch_size = min(50, len(X_sampled.columns))
            all_mi_scores = []
            all_columns = []
            
            tprint_info(f"📊 Computing MI in {len(range(0, len(X_sampled.columns), batch_size))} batches of {batch_size}")
            
            for i in range(0, len(X_sampled.columns), batch_size):
                batch_cols = X_sampled.columns[i:i + batch_size]
                batch_data = X_sampled[batch_cols]
                
                # Enhanced troubleshooting: Progress tracking
                tprint_progress(i // batch_size + 1, len(range(0, len(X_sampled.columns), batch_size)), 
                              f"Processing MI batch {i//batch_size + 1}")
                
                try:
                    # Clean data with enhanced optimization
                    batch_data_clean = batch_data.dropna()
                    y_clean = y_sampled.loc[batch_data_clean.index]
                    
                    if len(batch_data_clean) > 0:
                        # Optimize batch data
                        batch_data_clean = optimize_dataframe(batch_data_clean)
                        
                        # Compute MI
                        batch_mi_scores = mutual_info_regression(
                            batch_data_clean, y_clean,
                            random_state=42, n_neighbors=self.mi_neighbors
                        )
                        all_mi_scores.extend(batch_mi_scores)
                        all_columns.extend(batch_cols)
                except Exception as e:
                    tprint_warning(f"⚠️ MI batch failed: {e}")
                    continue
                
                # Enhanced cleanup after each batch
                if self.aggressive_gc:
                    force_cleanup()
            
            # Select top features by MI quantile
            mi_series = pd.Series(all_mi_scores, index=all_columns)
            cutoff = mi_series.quantile(self.mi_quantile)
            selected_features = mi_series[mi_series >= cutoff].index.tolist()
            
            mi_duration = time.time() - mi_start
            tprint_performance("MI filter calculation", mi_duration)
            tprint_success(f"✅ MI filter selected {len(selected_features)} features in {mi_duration:.2f}s")
            tprint_info(f"📊 MI range: {min(all_mi_scores):.4f} - {max(all_mi_scores):.4f}")
            tprint_info(f"📊 MI cutoff (quantile {self.mi_quantile}): {cutoff:.4f}")
            
            # Enhanced troubleshooting: Data format analysis after MI
            tprint_data_format(mi_series, "mi_scores", level=LogLevel.DEBUG)
            tprint_data_format(pd.DataFrame(X[selected_features]), "mi_selected_features", level=LogLevel.DEBUG)
            
            # Data preview after MI scoring
            tprint_data_preview(mi_series, "mi_scores", max_rows=20)
            tprint_data_preview(pd.DataFrame(X[selected_features]), "mi_selected_features", max_rows=3, max_cols=10)
            
            return selected_features
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked(log_performance=True, track_memory=True)
    async def _stage2_mrmr_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 2: Ultra-Optimized mRMR Selection
        
        Reduces from 200 to 150 features using batch selection, MI caching,
        approximate redundancy, and parallel processing.
        """
        with memory_managed_operation("Stage 2: mRMR Selection"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            # Enhanced troubleshooting: Input data analysis
            tprint_data_format(X, "stage2_input_features", level=LogLevel.DEBUG)
            tprint_data_format(y, "stage2_input_targets", level=LogLevel.DEBUG)
            
            # Ensure float32 with enhanced optimization
            X = optimize_dataframe(X.astype('float32', copy=False))
            y = y.astype('float32', copy=False)
            
            # Enhanced troubleshooting: Data format analysis after optimization
            tprint_data_format(X, "X_for_mrmr", level=LogLevel.DEBUG)
            
            # Data preview after float32 conversion for mRMR
            tprint_data_preview(X, "X_for_mrmr", max_rows=3, max_cols=10)
            
            # Step 1: Compute MI matrix upfront (cache)
            tprint_info("🔍 Step 2.1: Computing MI matrix (caching)")
            mi_cache_start = time.time()
            
            n_features = len(X.columns)
            mi_matrix = np.zeros((n_features, n_features), dtype='float32')
            
            # Compute MI(feature, target) - relevance scores
            tprint_info("📊 Computing relevance scores (MI with target)")
            relevance_scores = {}
            for i, col in enumerate(X.columns):
                try:
                    # Optimize data before MI calculation
                    col_data = optimize_dataframe(X[[col]])
                    mi_score = mutual_info_regression(
                        col_data, y, random_state=42, n_neighbors=self.mi_neighbors
                    )[0]
                    relevance_scores[col] = float(mi_score)
                except:
                    relevance_scores[col] = 0.0
            
            # Compute pairwise MI (redundancy) - only for top features
            tprint_info("📊 Computing pairwise MI (redundancy matrix)")
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    try:
                        # Optimize data before MI calculation
                        feat_i = optimize_dataframe(X.iloc[:, [i]])
                        feat_j = optimize_dataframe(X.iloc[:, [j]])
                        mi_score = mutual_info_regression(
                            feat_i, feat_j, random_state=42, n_neighbors=self.mi_neighbors
                        )[0]
                        mi_matrix[i, j] = mi_matrix[j, i] = float(mi_score)
                    except:
                        mi_matrix[i, j] = mi_matrix[j, i] = 0.0
                
                if i % 20 == 0:
                    tprint_debug(f"📊 MI matrix progress: {i}/{n_features}")
                    if self.aggressive_gc:
                        gc.collect()
            
            mi_cache_duration = time.time() - mi_cache_start
            tprint_performance("MI matrix computation", mi_cache_duration)
            tprint_success(f"✅ MI matrix computed in {mi_cache_duration:.2f}s")
            
            # Enhanced troubleshooting: Data format analysis of MI matrix
            tprint_data_format(pd.DataFrame(mi_matrix), "mi_matrix", level=LogLevel.DEBUG)
            
            # Data preview of MI matrix
            tprint_data_preview(pd.DataFrame(mi_matrix), "mi_matrix", max_rows=5, max_cols=10)
            
            # Step 2: Batch mRMR selection
            tprint_info("🔍 Step 2.2: Batch mRMR selection")
            selected_features = []
            selected_indices = []
            candidate_features = list(X.columns)
            candidate_indices = list(range(n_features))
            
            # Initialize with top feature by relevance
            top_feature = max(relevance_scores.items(), key=lambda x: x[1])[0]
            selected_features.append(top_feature)
            selected_indices.append(X.columns.get_loc(top_feature))
            candidate_features.remove(top_feature)
            candidate_indices.remove(X.columns.get_loc(top_feature))
            
            target_features = 150
            iterations = 0
            prev_score = float('inf')
            no_improvement_count = 0
            
            while len(selected_features) < target_features and candidate_features:
                iterations += 1
                
                # Compute scores for all candidates
                scores = {}
                for cand_feat, cand_idx in zip(candidate_features, candidate_indices):
                    # Relevance
                    relevance = relevance_scores[cand_feat]
                    
                    # Approximate redundancy (sample 20 from selected)
                    if len(selected_indices) <= self.mrmr_sample_size:
                        sample_indices = selected_indices
                    else:
                        sample_indices = np.random.choice(
                            selected_indices, self.mrmr_sample_size, replace=False
                        )
                    
                    redundancy = np.mean([mi_matrix[cand_idx, sel_idx] for sel_idx in sample_indices])
                    
                    # mRMR score
                    scores[cand_feat] = relevance - redundancy
                
                # Select top batch_size features
                batch_size = min(self.mrmr_batch_size, len(scores))
                top_batch = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:batch_size]
                
                # Add to selected
                for feat, score in top_batch:
                    selected_features.append(feat)
                    selected_indices.append(X.columns.get_loc(feat))
                    candidate_features.remove(feat)
                    candidate_indices.remove(X.columns.get_loc(feat))
                
                # Early stopping
                avg_score = np.mean([score for _, score in top_batch])
                improvement = (prev_score - avg_score) / max(abs(prev_score), 1e-10)
                
                if improvement < self.mrmr_early_stop_threshold:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                
                if no_improvement_count >= self.mrmr_early_stop_patience:
                    tprint_info(f"⏹️ Early stopping at iteration {iterations} (no improvement)")
                    break
                
                prev_score = avg_score
                
                if iterations % 3 == 0:
                    tprint_debug(f"📊 mRMR iteration {iterations}: {len(selected_features)} features selected")
                    if self.aggressive_gc:
                        gc.collect()
            
            tprint_success(f"✅ mRMR selected {len(selected_features)} features in {iterations} iterations")
            
            # Enhanced troubleshooting: Data format analysis after mRMR
            tprint_data_format(pd.DataFrame(X[selected_features[:target_features]]), "mrmr_selected_features", level=LogLevel.DEBUG)
            
            # Data preview after mRMR selection
            tprint_data_preview(pd.DataFrame(X[selected_features[:target_features]]), "mrmr_selected_features", max_rows=3, max_cols=10)
            
            return selected_features[:target_features]
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked(log_performance=True, track_memory=True)
    async def _stage3_lasso_stability(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 3: LASSO + Stability Selection
        
        Reduces from 150 to 100 features using stability selection (50 bootstraps)
        followed by LASSO + RFE in batches of 10.
        """
        with memory_managed_operation("Stage 3: LASSO + Stability"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            # Enhanced troubleshooting: Input data analysis
            tprint_data_format(X, "stage3_input_features", level=LogLevel.DEBUG)
            tprint_data_format(y, "stage3_input_targets", level=LogLevel.DEBUG)
            
            # Ensure float32 with enhanced optimization
            X = optimize_dataframe(X.astype('float32', copy=False))
            y = y.astype('float32', copy=False)
            
            # Enhanced troubleshooting: Data format analysis after optimization
            tprint_data_format(X, "X_for_lasso", level=LogLevel.DEBUG)
            
            # Data preview after float32 conversion for LASSO
            tprint_data_preview(X, "X_for_lasso", max_rows=3, max_cols=10)
            
            # Convert to sparse if beneficial
            sparsity = (X == 0).sum().sum() / X.size
            if sparsity > 0.3:
                tprint_info(f"📦 Converting to sparse matrix (sparsity: {sparsity:.2%})")
                X_sparse = csr_matrix(X.values)
                use_sparse = True
            else:
                X_sparse = X.values
                use_sparse = False
            
            # Step 1: Stability Selection (50 bootstraps)
            tprint_info("🔍 Step 3.1: Stability Selection (50 bootstraps)")
            stability_start = time.time()
            
            feature_selection_counts = {col: 0 for col in X.columns}
            
            # Parallel bootstrap processing
            def run_bootstrap(bootstrap_idx):
                """Run LASSO on a single bootstrap sample."""
                # Bootstrap sample
                n_samples = int(len(X) * self.bootstrap_sample_ratio)
                sample_indices = np.random.choice(len(X), n_samples, replace=True)
                
                if use_sparse:
                    X_boot = X_sparse[sample_indices]
                else:
                    X_boot = X_sparse[sample_indices]
                y_boot = y.iloc[sample_indices].values
                
                # LASSO
                try:
                    alphas = np.logspace(-4, 1, self.lasso_n_alphas)
                    lasso = LassoCV(
                        cv=self.lasso_cv_folds,
                        alphas=alphas,
                        max_iter=self.lasso_max_iter,
                        tol=self.lasso_tol,
                        random_state=42 + bootstrap_idx
                    )
                    lasso.fit(X_boot, y_boot)
                    
                    # Get selected features
                    selected = np.where(np.abs(lasso.coef_) > 1e-10)[0]
                    return selected
                except:
                    return np.array([])
            
            # Run bootstraps in parallel
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(run_bootstrap, i) for i in range(self.n_bootstrap)]
                
                for i, future in enumerate(futures):
                    try:
                        selected_indices = future.result()
                        for idx in selected_indices:
                            feature_selection_counts[X.columns[idx]] += 1
                    except Exception as e:
                        tprint_warning(f"⚠️ Bootstrap {i} failed: {e}")
                    
                    if (i + 1) % 10 == 0:
                        tprint_debug(f"📊 Bootstrap progress: {i + 1}/{self.n_bootstrap}")
                        if self.aggressive_gc:
                            gc.collect()
            
            # Compute stability scores
            stability_scores = {
                col: count / self.n_bootstrap
                for col, count in feature_selection_counts.items()
            }
            
            stability_duration = time.time() - stability_start
            tprint_performance("Stability selection", stability_duration)
            tprint_success(f"✅ Stability selection completed in {stability_duration:.2f}s")
            tprint_info(f"📊 Stability scores range: {min(stability_scores.values()):.3f} - {max(stability_scores.values()):.3f}")
            
            # Enhanced troubleshooting: Data format analysis of stability scores
            tprint_data_format(pd.Series(stability_scores), "stability_scores", level=LogLevel.DEBUG)
            
            # Data preview of stability scores
            tprint_data_preview(pd.Series(stability_scores), "stability_scores", max_rows=20)
            
            # Step 2: LASSO + RFE in batches of 10
            tprint_info("🔍 Step 3.2: LASSO + RFE (batches of 10)")
            rfe_start = time.time()
            
            current_features = list(X.columns)
            target_features = 100
            
            while len(current_features) > target_features:
                # Train LASSO
                X_current = X[current_features].values
                
                alphas = np.logspace(-4, 1, self.lasso_n_alphas)
                lasso = LassoCV(
                    cv=self.lasso_cv_folds,
                    alphas=alphas,
                    max_iter=self.lasso_max_iter,
                    tol=self.lasso_tol,
                    random_state=42
                )
                lasso.fit(X_current, y.values)
                
                # Combine LASSO coefficients with stability scores
                combined_scores = {}
                for i, feat in enumerate(current_features):
                    coef = np.abs(lasso.coef_[i])
                    stability = stability_scores[feat]
                    combined_scores[feat] = coef * stability
                
                # Remove bottom 10 features
                sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                n_to_remove = min(10, len(current_features) - target_features)
                current_features = [feat for feat, _ in sorted_features[:-n_to_remove]]
                
                tprint_debug(f"📊 RFE: {len(current_features)} features remaining")
                
                if self.aggressive_gc:
                    gc.collect()
            
            rfe_duration = time.time() - rfe_start
            tprint_performance("LASSO + RFE", rfe_duration)
            tprint_success(f"✅ LASSO + RFE completed in {rfe_duration:.2f}s")
            tprint_success(f"✅ Selected {len(current_features)} features")
            
            # Enhanced troubleshooting: Data format analysis after LASSO + RFE
            tprint_data_format(pd.DataFrame(X[current_features]), "lasso_selected_features", level=LogLevel.DEBUG)
            
            # Data preview after LASSO + RFE selection
            tprint_data_preview(pd.DataFrame(X[current_features]), "lasso_selected_features", max_rows=3, max_cols=10)
            
            return current_features
    
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked(log_performance=True, track_memory=True)
    async def _stage4_lgbm_rfe_shap(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Stage 4: LGBM + RFE + SHAP
        
        Reduces from 100 to 60/50/40 features using LightGBM + RFE in batches of 10,
        with SHAP for final feature importance.
        """
        with memory_managed_operation("Stage 4: LGBM + RFE + SHAP"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            # Enhanced troubleshooting: Input data analysis
            tprint_data_format(X, "stage4_input_features", level=LogLevel.DEBUG)
            tprint_data_format(y, "stage4_input_targets", level=LogLevel.DEBUG)
            
            if not LGBM_SHAP_AVAILABLE:
                tprint_error("❌ LightGBM/SHAP not available")
                raise ImportError("LightGBM and SHAP are required for Stage 4")
            
            # Ensure float32 with enhanced optimization
            X = optimize_dataframe(X.astype('float32', copy=False))
            y = y.astype('float32', copy=False)
            
            # Enhanced troubleshooting: Data format analysis after optimization
            tprint_data_format(X, "X_for_lgbm", level=LogLevel.DEBUG)
            
            # Data preview after float32 conversion for LGBM
            tprint_data_preview(X, "X_for_lgbm", max_rows=3, max_cols=10)
            
            # Create LightGBM Dataset (reusable)
            tprint_info("📦 Creating LightGBM Dataset")
            lgb_train = lgb.Dataset(X.values, y.values, feature_name=list(X.columns), free_raw_data=False)
            
            # RFE: 100 → 90 → 80 → 70 → 60
            tprint_info("🔍 Step 4.1: LGBM + RFE (100 → 60)")
            rfe_start = time.time()
            
            current_features = list(X.columns)
            target_60 = 60
            
            while len(current_features) > target_60:
                # Train LightGBM
                X_current = X[current_features].values
                lgb_train_current = lgb.Dataset(X_current, y.values, feature_name=current_features, free_raw_data=False)
                
                model = lgb.train(
                    self.lgbm_params,
                    lgb_train_current,
                    num_boost_round=self.lgbm_num_boost_round,
                    valid_sets=[lgb_train_current],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.lgbm_early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                # Get feature importance
                importance = model.feature_importance(importance_type='split')
                feature_importance = dict(zip(current_features, importance))
                
                # Remove bottom 10 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                n_to_remove = min(10, len(current_features) - target_60)
                current_features = [feat for feat, _ in sorted_features[:-n_to_remove]]
                
                tprint_debug(f"📊 RFE: {len(current_features)} features remaining")
                
                if len(current_features) % 20 == 0 and self.aggressive_gc:
                    gc.collect()
            
            features_60 = current_features
            rfe_60_duration = time.time() - rfe_start
            tprint_performance("RFE to 60 features", rfe_60_duration)
            tprint_success(f"✅ RFE to 60 features completed in {rfe_60_duration:.2f}s")
            
            # Enhanced troubleshooting: Data format analysis after RFE to 60
            tprint_data_format(pd.DataFrame(X[features_60]), "lgbm_features_60", level=LogLevel.DEBUG)
            
            # Data preview after RFE to 60 features
            tprint_data_preview(pd.DataFrame(X[features_60]), "lgbm_features_60", max_rows=3, max_cols=10)
            
            # Continue RFE: 60 → 50
            tprint_info("🔍 Step 4.2: LGBM + RFE (60 → 50)")
            current_features = features_60.copy()
            target_50 = 50
            
            while len(current_features) > target_50:
                X_current = X[current_features].values
                lgb_train_current = lgb.Dataset(X_current, y.values, feature_name=current_features, free_raw_data=False)
                
                model = lgb.train(
                    self.lgbm_params,
                    lgb_train_current,
                    num_boost_round=self.lgbm_num_boost_round,
                    valid_sets=[lgb_train_current],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.lgbm_early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                importance = model.feature_importance(importance_type='split')
                feature_importance = dict(zip(current_features, importance))
                
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                n_to_remove = min(10, len(current_features) - target_50)
                current_features = [feat for feat, _ in sorted_features[:-n_to_remove]]
            
            features_50 = current_features
            tprint_success(f"✅ RFE to 50 features completed")
            
            # Enhanced troubleshooting: Data format analysis after RFE to 50
            tprint_data_format(pd.DataFrame(X[features_50]), "lgbm_features_50", level=LogLevel.DEBUG)
            
            # Data preview after RFE to 50 features
            tprint_data_preview(pd.DataFrame(X[features_50]), "lgbm_features_50", max_rows=3, max_cols=10)
            
            # Continue RFE: 50 → 40
            tprint_info("🔍 Step 4.3: LGBM + RFE (50 → 40)")
            current_features = features_50.copy()
            target_40 = 40
            
            while len(current_features) > target_40:
                X_current = X[current_features].values
                lgb_train_current = lgb.Dataset(X_current, y.values, feature_name=current_features, free_raw_data=False)
                
                model = lgb.train(
                    self.lgbm_params,
                    lgb_train_current,
                    num_boost_round=self.lgbm_num_boost_round,
                    valid_sets=[lgb_train_current],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.lgbm_early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                importance = model.feature_importance(importance_type='split')
                feature_importance = dict(zip(current_features, importance))
                
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                n_to_remove = min(10, len(current_features) - target_40)
                current_features = [feat for feat, _ in sorted_features[:-n_to_remove]]
            
            features_40 = current_features
            tprint_success(f"✅ RFE to 40 features completed")
            
            # Enhanced troubleshooting: Data format analysis after RFE to 40
            tprint_data_format(pd.DataFrame(X[features_40]), "lgbm_features_40", level=LogLevel.DEBUG)
            
            # Data preview after RFE to 40 features
            tprint_data_preview(pd.DataFrame(X[features_40]), "lgbm_features_40", max_rows=3, max_cols=10)
            
            # SHAP for final feature importance
            tprint_info("🔍 Step 4.4: SHAP analysis for final feature sets")
            shap_start = time.time()
            
            # Sample data for SHAP
            shap_sample_size = min(self.shap_sample_size, len(X))
            sample_indices = np.random.choice(len(X), shap_sample_size, replace=False)
            
            # SHAP for 60 features
            tprint_info("📊 Computing SHAP for 60-feature set")
            X_60 = X[features_60].iloc[sample_indices].values
            model_60 = lgb.train(self.lgbm_params, lgb.Dataset(X[features_60].values, y.values), num_boost_round=50)
            explainer_60 = shap.TreeExplainer(model_60, check_additivity=False)
            
            shap_values_60 = []
            for i in range(0, len(X_60), self.shap_batch_size):
                batch = X_60[i:i + self.shap_batch_size]
                shap_values_60.append(explainer_60.shap_values(batch))
            shap_values_60 = np.vstack(shap_values_60)
            
            # SHAP for 50 features
            tprint_info("📊 Computing SHAP for 50-feature set")
            X_50 = X[features_50].iloc[sample_indices].values
            model_50 = lgb.train(self.lgbm_params, lgb.Dataset(X[features_50].values, y.values), num_boost_round=50)
            explainer_50 = shap.TreeExplainer(model_50, check_additivity=False)
            
            shap_values_50 = []
            for i in range(0, len(X_50), self.shap_batch_size):
                batch = X_50[i:i + self.shap_batch_size]
                shap_values_50.append(explainer_50.shap_values(batch))
            shap_values_50 = np.vstack(shap_values_50)
            
            # SHAP for 40 features
            tprint_info("📊 Computing SHAP for 40-feature set")
            X_40 = X[features_40].iloc[sample_indices].values
            model_40 = lgb.train(self.lgbm_params, lgb.Dataset(X[features_40].values, y.values), num_boost_round=50)
            explainer_40 = shap.TreeExplainer(model_40, check_additivity=False)
            
            shap_values_40 = []
            for i in range(0, len(X_40), self.shap_batch_size):
                batch = X_40[i:i + self.shap_batch_size]
                shap_values_40.append(explainer_40.shap_values(batch))
            shap_values_40 = np.vstack(shap_values_40)
            
            # Compute mean absolute SHAP values
            feature_scores = {}
            for i, feat in enumerate(features_60):
                feature_scores[feat] = float(np.mean(np.abs(shap_values_60[:, i])))
            
            shap_duration = time.time() - shap_start
            tprint_performance("SHAP analysis", shap_duration)
            tprint_success(f"✅ SHAP analysis completed in {shap_duration:.2f}s")
            
            # Enhanced troubleshooting: Data format analysis of final feature scores
            tprint_data_format(pd.Series(feature_scores), "final_feature_scores", level=LogLevel.DEBUG)
            
            # Data preview of final feature scores
            tprint_data_preview(pd.Series(feature_scores), "final_feature_scores", max_rows=20)
            
            # Enhanced cleanup
            del explainer_60, explainer_50, explainer_40
            del model_60, model_50, model_40
            force_cleanup()
            
            return {
                'features_60': features_60,
                'features_50': features_50,
                'features_40': features_40,
                'feature_scores': feature_scores,
                'shap_values_60': shap_values_60,
                'shap_values_50': shap_values_50,
                'shap_values_40': shap_values_40
            }
    
    async def _save_artifacts(self, artifact_manager, result: FinalFeatureSelectionResult,
                            model_type: str, direction: str):
        """Save artifacts to artifact manager."""
        try:
            step_name = f'feature_generation_final_feature_selection_step_{model_type}_{direction}'
            
            # Enhanced troubleshooting: Log saving parameters
            tprint_structured({
                'operation': 'save_artifacts',
                'step_name': step_name,
                'model_type': model_type,
                'direction': direction,
                'timestamp': datetime.now().isoformat()
            }, level=LogLevel.DEBUG)
            
            artifact_manager.save(
                step_name=step_name,
                artifacts={
                    'selected_features_60': result.selected_features_60,
                    'selected_features_50': result.selected_features_50,
                    'selected_features_40': result.selected_features_40,
                    'selected_feature_dataframe_60': result.selected_feature_dataframe_60,
                    'selected_feature_dataframe_50': result.selected_feature_dataframe_50,
                    'selected_feature_dataframe_40': result.selected_feature_dataframe_40,
                    'feature_scores': result.feature_scores,
                    'shap_values_60': result.shap_values_60,
                    'shap_values_50': result.shap_values_50,
                    'shap_values_40': result.shap_values_40,
                    'selection_metadata': result.selection_metadata
                },
                metadata={
                    'step': step_name,
                    'model_type': model_type,
                    'direction': direction,
                    'created_at': datetime.now().isoformat(),
                    'execution_time': result.execution_time
                }
            )
            
            tprint_success(f"✅ Artifacts saved for {model_type}_{direction}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save artifacts: {e}")
            tprint_exception(e, "Failed to save artifacts")
            self.logger.error(f"Failed to save artifacts: {e}", exc_info=True)
    
    def get_hardware_optimization_status(self) -> Dict[str, Any]:
        """Get current hardware optimization status."""
        if hasattr(self, 'hardware_manager'):
            return self.hardware_manager.get_optimization_report()
        return {"error": "Hardware manager not available"}
    
    def get_memory_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        return get_memory_stats()


# Handler function for pipeline integration
async def handle_feature_generation_final_feature_selection_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    model_type: str = "analyst",
    direction: str = "long",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> FinalFeatureSelectionResult:
    """
    Handler function for final feature selection step.
    
    Args:
        symbol: Trading symbol
        timeframe: Trading timeframe
        model_type: 'analyst' or 'tactician'
        direction: 'long' or 'short'
        config: Optional configuration
        
    Returns:
        FinalFeatureSelectionResult
    """
    # Enhanced troubleshooting: Log handler parameters
    tprint_structured({
        'handler': 'handle_feature_generation_final_feature_selection_step',
        'symbol': symbol,
        'timeframe': timeframe,
        'model_type': model_type,
        'direction': direction,
        'config_keys': list(config.keys()) if config else [],
        'kwargs_keys': list(kwargs.keys()) if kwargs else [],
        'timestamp': datetime.now().isoformat()
    }, level=LogLevel.INFO)
    
    tprint_info(f"🚀 Starting final feature selection handler for {model_type}_{direction}")
    
    # Create step instance
    step = FeatureGenerationFinalFeatureSelectionStep(config=config)
    
    # Execute
    result = await step.execute(
        model_type=model_type,
        direction=direction,
        symbol=symbol,
        timeframe=timeframe,
        config=config
    )
    
    return result

