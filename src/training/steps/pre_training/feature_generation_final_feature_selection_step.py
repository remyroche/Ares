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

# M1 Hardware Optimization Imports
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, optimize_dataframe_for_m1
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, optimize_memory
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, create_m1_optimized_thread_pool
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False
    warnings.warn("M1 optimizations not available")

# VectorBT Imports
try:
    import vectorbt as vbt
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available")

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
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
except ImportError:
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import modular architecture
from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import ModularComponent
from src.training.steps.pre_training.utils.artifact_manager import get_pretraining_artifact_manager, ArtifactKeys


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
    """Context manager for memory-managed operations with GC."""
    try:
        tprint_debug(f"🧠 Starting memory-managed operation: {operation_name}")
        yield
    finally:
        gc.collect()
        tprint_debug(f"🧠 Completed memory-managed operation: {operation_name}, GC run")


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
        
        # Initialize M1 hardware optimizers
        if M1_OPTIMIZATIONS_AVAILABLE:
            tprint_info("🚀 Initializing M1 hardware optimizers")
            self.m1_gpu_manager = get_m1_gpu_manager()
            self.m1_memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            tprint_success("✅ M1 hardware optimizers initialized")
        else:
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
            tprint_warning("⚠️ M1 optimizations not available")
        
        # Initialize VectorBT utilities
        if VECTORBT_AVAILABLE:
            tprint_info("🚀 Initializing VectorBT utilities")
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            self.unified_vectorization_manager = get_unified_vectorization_manager()
            tprint_success("✅ VectorBT utilities initialized")
        else:
            self.vectorbt_optimizer = None
            self.unified_vectorization_manager = None
            tprint_warning("⚠️ VectorBT not available")
        
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
            'gc_runs': 0
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
        """Cleanup component-specific resources."""
        self.set_state('cleaned_up_at', datetime.now().isoformat())
        gc.collect()
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        return data
    
    def _check_memory_and_gc(self):
        """Check memory usage and run GC if needed."""
        self.operation_counter += 1
        
        if self.aggressive_gc and self.operation_counter % self.gc_frequency == 0:
            gc.collect()
            self.performance_stats['gc_runs'] += 1
            
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                mem_usage = process.memory_info().rss / 1024 / 1024  # MB
                if mem_usage > 4096:  # > 4GB
                    gc.collect(generation=2)
                    tprint_warning(f"⚠️ High memory usage: {mem_usage:.0f}MB, forcing full GC")
    
    def _convert_to_float32(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame to float32 for memory efficiency."""
        tprint_debug("🔄 Converting data to float32")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype('float32', copy=False)
        self.performance_stats['memory_optimizations_applied'] += 1
        return df
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final feature selection step.
        
        Args:
            model_type: 'analyst' or 'tactician'
            direction: 'long' or 'short'
            symbol: Trading symbol
            timeframe: Trading timeframe
            
        Returns:
            FinalFeatureSelectionResult with 3 feature sets (60, 50, 40)
        """
        start_time = time.time()
        tprint_info(f"🎯 Starting final feature selection for {model_type}_{direction}")
        
        # Set up enhanced artifact manager with context
        context = get_step_context_from_config(self.config)
        context.update({
            'symbol': symbol,
            'exchange': 'binance',  # Default exchange
            'direction': direction,
            'model': model_type.title()  # Convert to proper case
        })
        am = setup_enhanced_artifact_manager(**context)
        
        try:
            # Get artifact manager
            artifact_manager = am
            
            # Load artifacts from previous steps
            tprint_info("📦 Loading artifacts from previous steps")
            features_df, targets = await self._load_artifacts(artifact_manager, model_type, direction)
            
            if features_df is None or targets is None:
                raise ValueError("Failed to load required artifacts from previous steps")
            
            # Convert to float32 end-to-end
            tprint_info("🔄 Converting data to float32 (end-to-end)")
            features_df = self._convert_to_float32(features_df)
            targets = targets.astype('float32', copy=False)
            
            tprint_success(f"✅ Loaded {features_df.shape[1]} features and {len(targets)} targets")
            tprint_info(f"📊 Feature matrix: {features_df.shape}, Target: {targets.shape}")
            tprint_info(f"💾 Memory usage: Features={features_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB")
            
            # Stage 1: PCA + Approximate MI (→ ~200 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 1: PCA + Approximate MI Filter")
            stage1_start = time.time()
            features_200 = await self._stage1_pca_mi_filter(features_df, targets)
            self.performance_stats['stage_times']['stage1_pca_mi'] = time.time() - stage1_start
            tprint_success(f"✅ Stage 1 complete: {len(features_200)} features selected")
            self._check_memory_and_gc()
            
            # Stage 2: Ultra-optimized mRMR (200 → 150 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 2: Ultra-Optimized mRMR Selection")
            stage2_start = time.time()
            features_150 = await self._stage2_mrmr_selection(
                features_df[features_200], targets
            )
            self.performance_stats['stage_times']['stage2_mrmr'] = time.time() - stage2_start
            tprint_success(f"✅ Stage 2 complete: {len(features_150)} features selected")
            self._check_memory_and_gc()
            
            # Stage 3: LASSO + Stability Selection (150 → 100 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 3: LASSO + Stability Selection")
            stage3_start = time.time()
            features_100 = await self._stage3_lasso_stability(
                features_df[features_150], targets
            )
            self.performance_stats['stage_times']['stage3_lasso_stability'] = time.time() - stage3_start
            tprint_success(f"✅ Stage 3 complete: {len(features_100)} features selected")
            self._check_memory_and_gc()
            
            # Stage 4: LGBM + RFE + SHAP (100 → 60/50/40 features)
            tprint_info("=" * 80)
            tprint_info("🔍 STAGE 4: LGBM + RFE + SHAP")
            stage4_start = time.time()
            result_dict = await self._stage4_lgbm_rfe_shap(
                features_df[features_100], targets
            )
            self.performance_stats['stage_times']['stage4_lgbm_rfe_shap'] = time.time() - stage4_start
            tprint_success("✅ Stage 4 complete: Generated 3 feature sets (60, 50, 40)")
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
            await self._save_artifacts(artifact_manager, result, model_type, direction)
            
            tprint_success("=" * 80)
            tprint_success(f"🎉 Final feature selection completed in {execution_time:.2f}s")
            tprint_success(f"📊 Feature reduction: {features_df.shape[1]} → 60/50/40")
            tprint_success(f"💾 Total GC runs: {self.performance_stats['gc_runs']}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Final feature selection failed: {e}")
            self.logger.error(f"Final feature selection failed: {e}", exc_info=True)
            
            return FinalFeatureSelectionResult(
                success=False,
                selected_features_60=[],
                selected_features_50=[],
                selected_features_40=[],
                selected_feature_dataframe_60=pd.DataFrame(),
                selected_feature_dataframe_50=pd.DataFrame(),
                selected_feature_dataframe_40=pd.DataFrame(),
                feature_scores={},
                shap_values_60=None,
                shap_values_50=None,
                shap_values_40=None,
                selection_metadata={},
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _load_artifacts(self, artifact_manager, model_type: str, direction: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load features and targets from previous steps."""
        try:
            # Load original features from feature_generation_feature_generation_step
            tprint_info("📦 Loading original features from feature_generation_feature_generation_step")
            original_features = artifact_manager.get_dataframe(
                'feature_generation_feature_generation_step',
                ArtifactKeys.GENERATED_FEATURES
            )
            
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
            
            tprint_info("📦 Loading interaction features from interaction_generation_step_tactician")
            tactician_interaction_features = artifact_manager.get_dataframe(
                'feature_generation_interaction_generation_step_tactician',
                ArtifactKeys.INTERACTION_FEATURES
            )
            
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
            
            return features_df, targets
            
        except Exception as e:
            tprint_error(f"❌ Failed to load artifacts: {e}")
            self.logger.error(f"Failed to load artifacts: {e}", exc_info=True)
            return None, None
    
    async def _stage1_pca_mi_filter(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 1: PCA + Approximate MI Filter
        
        Reduces features to ~200 using PCA (98% variance) + MI with target.
        """
        with memory_managed_operation("Stage 1: PCA + MI Filter"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            # Step 1: PCA dimensionality reduction (98% variance)
            tprint_info("🔍 Step 1.1: PCA dimensionality reduction (98% variance)")
            pca_start = time.time()
            
            # Ensure float32
            X_numeric = X.select_dtypes(include=[np.number]).astype('float32', copy=False)
            
            # Fit PCA
            pca = PCA(n_components=0.98, random_state=42)
            pca.fit(X_numeric)
            
            # For each component, find feature with highest loading
            pca_selected = set()
            for component in pca.components_:
                top_feature_idx = np.abs(component).argmax()
                pca_selected.add(X_numeric.columns[top_feature_idx])
            
            pca_features = list(pca_selected)
            tprint_success(f"✅ PCA selected {len(pca_features)} features in {time.time() - pca_start:.2f}s")
            
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
            
            # Compute MI in batches
            batch_size = min(50, len(X_sampled.columns))
            all_mi_scores = []
            all_columns = []
            
            for i in range(0, len(X_sampled.columns), batch_size):
                batch_cols = X_sampled.columns[i:i + batch_size]
                batch_data = X_sampled[batch_cols]
                
                try:
                    # Clean data
                    batch_data_clean = batch_data.dropna()
                    y_clean = y_sampled.loc[batch_data_clean.index]
                    
                    if len(batch_data_clean) > 0:
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
                
                # GC after each batch
                if self.aggressive_gc:
                    gc.collect()
            
            # Select top features by MI quantile
            mi_series = pd.Series(all_mi_scores, index=all_columns)
            cutoff = mi_series.quantile(self.mi_quantile)
            selected_features = mi_series[mi_series >= cutoff].index.tolist()
            
            tprint_success(f"✅ MI filter selected {len(selected_features)} features in {time.time() - mi_start:.2f}s")
            tprint_info(f"📊 MI range: {min(all_mi_scores):.4f} - {max(all_mi_scores):.4f}")
            tprint_info(f"📊 MI cutoff (quantile {self.mi_quantile}): {cutoff:.4f}")
            
            return selected_features
    
    async def _stage2_mrmr_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 2: Ultra-Optimized mRMR Selection
        
        Reduces from 200 to 150 features using batch selection, MI caching,
        approximate redundancy, and parallel processing.
        """
        with memory_managed_operation("Stage 2: mRMR Selection"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            # Ensure float32
            X = X.astype('float32', copy=False)
            y = y.astype('float32', copy=False)
            
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
                    mi_score = mutual_info_regression(
                        X[[col]], y, random_state=42, n_neighbors=self.mi_neighbors
                    )[0]
                    relevance_scores[col] = float(mi_score)
                except:
                    relevance_scores[col] = 0.0
            
            # Compute pairwise MI (redundancy) - only for top features
            tprint_info("📊 Computing pairwise MI (redundancy matrix)")
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    try:
                        mi_score = mutual_info_regression(
                            X.iloc[:, [i]], X.iloc[:, j], random_state=42, n_neighbors=self.mi_neighbors
                        )[0]
                        mi_matrix[i, j] = mi_matrix[j, i] = float(mi_score)
                    except:
                        mi_matrix[i, j] = mi_matrix[j, i] = 0.0
                
                if i % 20 == 0:
                    tprint_debug(f"📊 MI matrix progress: {i}/{n_features}")
                    if self.aggressive_gc:
                        gc.collect()
            
            tprint_success(f"✅ MI matrix computed in {time.time() - mi_cache_start:.2f}s")
            
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
            
            return selected_features[:target_features]
    
    async def _stage3_lasso_stability(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Stage 3: LASSO + Stability Selection
        
        Reduces from 150 to 100 features using stability selection (50 bootstraps)
        followed by LASSO + RFE in batches of 10.
        """
        with memory_managed_operation("Stage 3: LASSO + Stability"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            # Ensure float32
            X = X.astype('float32', copy=False)
            y = y.astype('float32', copy=False)
            
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
            
            tprint_success(f"✅ Stability selection completed in {time.time() - stability_start:.2f}s")
            tprint_info(f"📊 Stability scores range: {min(stability_scores.values()):.3f} - {max(stability_scores.values()):.3f}")
            
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
            
            tprint_success(f"✅ LASSO + RFE completed in {time.time() - rfe_start:.2f}s")
            tprint_success(f"✅ Selected {len(current_features)} features")
            
            return current_features
    
    async def _stage4_lgbm_rfe_shap(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Stage 4: LGBM + RFE + SHAP
        
        Reduces from 100 to 60/50/40 features using LightGBM + RFE in batches of 10,
        with SHAP for final feature importance.
        """
        with memory_managed_operation("Stage 4: LGBM + RFE + SHAP"):
            tprint_info(f"📊 Input: {X.shape[1]} features")
            
            if not LGBM_SHAP_AVAILABLE:
                tprint_error("❌ LightGBM/SHAP not available")
                raise ImportError("LightGBM and SHAP are required for Stage 4")
            
            # Ensure float32
            X = X.astype('float32', copy=False)
            y = y.astype('float32', copy=False)
            
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
            tprint_success(f"✅ RFE to 60 features completed in {time.time() - rfe_start:.2f}s")
            
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
            
            tprint_success(f"✅ SHAP analysis completed in {time.time() - shap_start:.2f}s")
            
            # Cleanup
            del explainer_60, explainer_50, explainer_40
            del model_60, model_50, model_40
            gc.collect()
            
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
            self.logger.error(f"Failed to save artifacts: {e}", exc_info=True)


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
    tprint_info(f"🚀 Starting final feature selection handler for {model_type}_{direction}")
    
    # Create step instance
    step = FeatureGenerationFinalFeatureSelectionStep(config=config)
    
    # Execute
    result = await step.execute(
        model_type=model_type,
        direction=direction,
        symbol=symbol,
        timeframe=timeframe,
        **kwargs
    )
    
    return result

