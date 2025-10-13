"""
Core pipeline implementation for multi-stage feature selection.

This module contains the main MultiStageFeatureSelector class that orchestrates
the multi-stage feature selection process with VectorBT optimizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

# Import configuration classes
from .config import FeatureSelectionConfig, FeatureSelectionResult

# Import VectorBT mRMR selector
try:
    from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
    from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig
    VECTORBT_MRMR_AVAILABLE = True
except ImportError:
    VECTORBT_MRMR_AVAILABLE = False

# Import LightGBM and SHAP for ensemble methods
try:
    import lightgbm as lgb
    import shap
    LIGHTGBM_SHAP_AVAILABLE = True
except ImportError:
    LIGHTGBM_SHAP_AVAILABLE = False

# Import scikit-learn for ensemble methods
try:
    from sklearn.linear_model import LassoCV
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import hardware optimization tools
try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_adaptive_optimization_engine,
        get_advanced_memory_optimizer,
        WorkloadType
    )
    from src.utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import VectorBT utilities
try:
    from ..utils.vectorbt_utils import (
        create_vectorbt_tools, VectorBTConfig, get_vectorbt_performance_stats,
        VECTORBT_UTILS_AVAILABLE
    )
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

# Import gate protection
try:
    from src.training.steps.pre_training.gate_feature_protection import (
        GateFeatureProtector, GateFeatureConfig
    )
    GATE_PROTECTION_AVAILABLE = True
except ImportError:
    GATE_PROTECTION_AVAILABLE = False


class MultiStageFeatureSelector:
    """
    Multi-stage feature selection using RandomForest and SHAP with vectorization and caching.
    
    DEPRECATED: Use MultiStageFeatureSelectionPipeline instead for new code.
    This class is maintained for backward compatibility.
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None, execution_mode_config: Optional[Dict[str, Any]] = None):
        self.config = config or FeatureSelectionConfig()
        self.logger = get_logger("MultiStageFeatureSelector")
        self.matrix_ops = get_unified_matrix_operations()

        # Initialize VectorBT optimization tools if available
        if VECTORBT_UTILS_AVAILABLE and self.config.enable_vectorbt_optimization:
            tprint("🚀 Initializing VectorBT optimization tools")
            vectorbt_config = VectorBTConfig(
                enable_gpu=self.config.vectorbt_enable_gpu,
                enable_parallel=self.config.vectorbt_enable_parallel,
                memory_efficient=self.config.vectorbt_memory_efficient,
                chunk_size=self.config.vectorbt_chunk_size
            )
            
            vectorbt_tools = create_vectorbt_tools(vectorbt_config)
            self.vectorbt_optimizer = vectorbt_tools['optimizer']
            self.vectorization_manager = vectorbt_tools['manager']
            self.vectorbt_enabled = vectorbt_tools['available']
            
            if self.vectorbt_enabled:
                tprint("✅ VectorBT optimization tools initialized successfully")
            else:
                tprint_warning("⚠️ VectorBT optimization tools not available")
        else:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
            self.vectorbt_enabled = False
            tprint("⚠️ VectorBT optimization disabled or not available")

        # Initialize gate feature protection if available
        if GATE_PROTECTION_AVAILABLE:
            gate_config = self.config.custom_params.get('gate_protection', {})
            if gate_config and isinstance(gate_config, dict):
                self.gate_protector = GateFeatureProtector(GateFeatureConfig(**gate_config))
            else:
                self.gate_protector = GateFeatureProtector()
            tprint("🛡️ Gate feature protection enabled")
        else:
            self.gate_protector = None

        # Initialize hardware optimization tools if available
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.hardware_manager = get_unified_hardware_manager()
            self.adaptive_engine = get_adaptive_optimization_engine()
            self.memory_optimizer = get_advanced_memory_optimizer()
            
            # Initialize advanced memory optimizer for aggressive cleanup
            try:
                self.advanced_memory_optimizer = AdvancedM1MemoryOptimizer(
                    memory_limit_gb=8.0,
                    strategy=MemoryStrategy.AGGRESSIVE
                )
                self.logger.info("🧠 Advanced memory optimizer initialized with aggressive strategy")
            except Exception as e:
                self.advanced_memory_optimizer = None
                self.logger.warning(f"⚠️ Advanced memory optimizer not available: {e}")
            
            self.logger.info("🚀 Hardware optimization tools initialized")
        else:
            self.hardware_manager = None
            self.adaptive_engine = None
            self.memory_optimizer = None
            self.advanced_memory_optimizer = None
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
        
        # Initialize VectorBT memory optimization if available
        if VECTORBT_UTILS_AVAILABLE:
            self._vectorbt_memory_config = {
                'use_memory_efficient': True,
                'chunk_size': 1000,
                'max_memory_gb': 8.0,
                'enable_gpu': False  # Can be enabled if GPU is available
            }
            tprint("🚀 VectorBT memory optimization enabled")
        else:
            self._vectorbt_memory_config = None
            tprint("⚠️ VectorBT memory optimization not available")
    
    def _display_vectorbt_status(self):
        """Display VectorBT optimization status and configuration."""
        if VECTORBT_UTILS_AVAILABLE:
            tprint("🚀 VECTORBT OPTIMIZATION STATUS:")
            tprint(f"   ✅ VectorBT Available: {VECTORBT_UTILS_AVAILABLE}")
            tprint(f"   🧠 Memory Optimization: {'Enabled' if self._vectorbt_memory_config else 'Disabled'}")
            if self._vectorbt_memory_config:
                tprint(f"   📦 Chunk Size: {self._vectorbt_memory_config['chunk_size']}")
                tprint(f"   💾 Max Memory: {self._vectorbt_memory_config['max_memory_gb']}GB")
                tprint(f"   🎮 GPU Enabled: {self._vectorbt_memory_config['enable_gpu']}")
        else:
            tprint("⚠️ VectorBT optimization not available")

    def _set_thread_limits(self):
        """Set thread limits to avoid oversubscription."""
        tprint_debug("🔧 Setting thread limits to avoid oversubscription")
        try:
            import os
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            os.environ['NUMEXPR_NUM_THREADS'] = '4'
        except Exception as e:
            tprint_warning(f"⚠️ Could not set thread limits: {e}")

    def _y_numeric(self, y: pd.Series) -> np.ndarray:
        """Convert y to numeric for Spearman/ranks."""
        if pd.api.types.is_numeric_dtype(y):
            return y.values
        return pd.to_numeric(y, errors='coerce').values

    def spearman_abs_vectorized(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Optimized vectorized Spearman correlation calculation with early termination."""
        # Fast fail on empty data
        if X.empty or y.empty:
            return pd.Series(dtype=float, index=X.columns)
        
        # Convert to numeric and handle missing values
        y_numeric = self._y_numeric(y)
        valid_mask = ~np.isnan(y_numeric)
        
        if not valid_mask.any():
            return pd.Series(0.0, index=X.columns)
        
        # Use only valid samples
        X_valid = X.loc[valid_mask]
        y_valid = y_numeric[valid_mask]
        
        # Calculate ranks
        X_ranks = X_valid.rank(method="average")
        y_ranks = pd.Series(y_valid).rank(method="average")
        
        # Vectorized correlation calculation
        X_centered = X_ranks - X_ranks.mean()
        y_centered = y_ranks - y_ranks.mean()
        
        numerator = (X_centered * y_centered).sum()
        denominator = np.sqrt((X_centered ** 2).sum() * (y_centered ** 2).sum())
        
        # Avoid division by zero
        correlations = numerator / (denominator + 1e-10)
        
        return correlations.abs()

    def redundancy_mean_abs_spearman_blocked(self, X: pd.DataFrame, block=1024) -> pd.Series:
        """Blocked redundancy calculation for large feature sets."""
        R = X.rank(method="average")
        n_features = len(X.columns)
        redundancy_scores = pd.Series(0.0, index=X.columns)
        
        for i in range(0, n_features, block):
            end_i = min(i + block, n_features)
            block_i = R.iloc[:, i:end_i]
            
            for j in range(i, n_features, block):
                end_j = min(j + block, n_features)
                block_j = R.iloc[:, j:end_j]
                
                # Calculate correlations between blocks
                corr_block = block_i.corrwith(block_j, method='spearman').abs()
                
                # Update redundancy scores
                for idx, col in enumerate(block_i.columns):
                    if col not in redundancy_scores.index:
                        continue
                    redundancy_scores[col] += corr_block.iloc[idx].sum()
        
        return redundancy_scores / (n_features - 1)

    def zscore_matrix(self, M: np.ndarray, axis=0) -> np.ndarray:
        """Vectorized z-score normalization."""
        mu = M.mean(axis=axis, keepdims=True)
        sigma = M.std(axis=axis, keepdims=True)
        return (M - mu) / (sigma + 1e-10)

    def top_k(self, names: List[str], scores: np.ndarray, k: int) -> List[str]:
        """Top-k selection using argpartition for O(p) performance."""
        k = min(k, len(scores))
        if k <= 0:
            return []
        
        # Use argpartition for O(p) performance instead of full sort O(p log p)
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_scores = scores[top_k_indices]
        
        # Sort only the top-k elements
        sorted_indices = np.argsort(top_k_scores)[::-1]
        return [names[top_k_indices[i]] for i in sorted_indices]

    def bottom_k(self, names: List[str], scores: np.ndarray, k: int) -> List[str]:
        """Bottom-k selection using argpartition for O(p) performance."""
        k = min(k, len(scores))
        if k <= 0:
            return []
        
        # Use argpartition for O(p) performance
        bottom_k_indices = np.argpartition(scores, k)[:k]
        bottom_k_scores = scores[bottom_k_indices]
        
        # Sort only the bottom-k elements
        sorted_indices = np.argsort(bottom_k_scores)
        return [names[bottom_k_indices[i]] for i in sorted_indices]

    def _vectorbt_optimized_correlation_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate correlation matrix using VectorBT optimization."""
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            return X.corr().values
        
        try:
            # Use VectorBT for optimized correlation calculation
            return self.vectorization_manager.calculate_correlation_matrix(X)
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT correlation calculation failed: {e}")
            return X.corr().values

    def _vectorbt_optimized_rolling_operations(self, X: pd.DataFrame, operation: str, window: int = 20) -> pd.DataFrame:
        """Perform rolling operations using VectorBT optimization."""
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            return X.rolling(window=window).agg(operation)
        
        try:
            return self.vectorization_manager.rolling_operation(X, operation, window)
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT rolling operation failed: {e}")
            return X.rolling(window=window).agg(operation)

    def _vectorbt_optimized_scaling(self, X: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Scale data using VectorBT optimization."""
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            if method == 'zscore':
                return (X - X.mean()) / X.std()
            return X
        
        try:
            return self.vectorization_manager.scale_data(X, method)
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT scaling failed: {e}")
            if method == 'zscore':
                return (X - X.mean()) / X.std()
            return X

    def _vectorbt_optimized_spearman_correlation(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate Spearman correlation using VectorBT optimization."""
        tprint_debug("   🚀 Using VectorBT for Spearman correlation calculation")
        
        try:
            if not self.vectorbt_enabled or self.vectorization_manager is None:
                tprint_debug("   ⚠️ VectorBT not available, using standard method")
                return self.spearman_abs_vectorized(X, y)
            
            # Use VectorBT for optimized correlation calculation
            # This would be implemented in the VectorBT manager
            corr_scores = self.vectorization_manager.calculate_spearman_correlation(X, y)
            
            tprint_debug(f"   ✅ VectorBT Spearman correlation completed: {len(corr_scores)} features")
            return corr_scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT Spearman correlation failed: {e}")
            tprint_debug("   📊 Falling back to standard Spearman correlation")
            return self.spearman_abs_vectorized(X, y)

    def _vectorbt_optimized_redundancy_analysis(self, X: pd.DataFrame) -> pd.Series:
        """Calculate redundancy analysis using VectorBT optimization."""
        tprint_debug("   🚀 Using VectorBT for redundancy analysis")
        
        try:
            if not self.vectorbt_enabled or self.vectorization_manager is None:
                tprint_debug("   ⚠️ VectorBT not available, using standard method")
                return self.redundancy_mean_abs_spearman_blocked(X)
            
            # Use VectorBT for optimized redundancy analysis
            # This would be implemented in the VectorBT manager
            redundancy_scores = self.vectorization_manager.calculate_redundancy_analysis(X)
            
            tprint_debug(f"   ✅ VectorBT redundancy analysis completed: {len(redundancy_scores)} features")
            return redundancy_scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT redundancy analysis failed: {e}")
            tprint_debug("   📊 Falling back to standard redundancy analysis")
            return self.redundancy_mean_abs_spearman_blocked(X)

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       symbol: str = "BTCUSDT", exchange: str = "binance", 
                       timeframe: str = "15m") -> FeatureSelectionResult:
        """
        Execute multi-stage feature selection with fast fail and extensive logging.
        
        Args:
            X: Feature matrix
            y: Target variable
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            FeatureSelectionResult with selected features and metrics
        """
        start_time = time.time()
        tprint("🚀 Starting multi-stage feature selection")
        tprint_info(f"   📊 Input data shape: {X.shape}")
        tprint_info(f"   📊 Target shape: {y.shape}")
        tprint_info(f"   📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
        
        try:
            # FAST FAIL: Early validation checks
            tprint_debug("🔍 Performing fast fail validation checks")
            
            # Check input data validity
            if X is None or X.empty:
                error_msg = "Input feature matrix X is None or empty"
                tprint_error(f"❌ FAST FAIL: {error_msg}")
                raise ValueError(error_msg)
            
            if y is None or y.empty:
                error_msg = "Target variable y is None or empty"
                tprint_error(f"❌ FAST FAIL: {error_msg}")
                raise ValueError(error_msg)
            
            if len(X) != len(y):
                error_msg = f"Feature matrix length ({len(X)}) doesn't match target length ({len(y)})"
                tprint_error(f"❌ FAST FAIL: {error_msg}")
                raise ValueError(error_msg)
            
            if X.shape[1] == 0:
                error_msg = "Feature matrix has no columns"
                tprint_error(f"❌ FAST FAIL: {error_msg}")
                raise ValueError(error_msg)
            
            tprint_success("   ✅ Fast fail validation passed")
            
            # Use new RFE-based pipeline
            tprint("🚀 Using RFE-based multi-stage pipeline")
            tprint_info("   📊 Stage 1: mRMR + Spearman combination (70% mRMR + 30% Spearman)")
            tprint_info("   📊 Stage 2: RFE with percentage-based step size (10% of features above target)")
            
            # Set thread limits
            tprint_debug("🔧 Setting thread limits")
            self._set_thread_limits()
            
            # Display VectorBT status
            tprint_debug("🚀 Displaying VectorBT status")
            self._display_vectorbt_status()
            
            # Initialize result tracking
            tprint_debug("📊 Initializing result tracking")
            stage_results = {}
            selected_features = X.columns.tolist()
            feature_importance = {}
            feature_scores = {}
            
            tprint_info(f"   📊 Starting with {len(selected_features)} features")
            
            # Stage 1: mRMR + Spearman combination
            tprint("📊 Stage 1: mRMR + Spearman combination")
            tprint_debug(f"   🔍 Input features for Stage 1: {len(selected_features)}")
            
            try:
                stage_1_result = self._stage_1_mrmr_spearman_combination(X, y)
                selected_features = stage_1_result['selected_features']
                stage_results['stage_1'] = stage_1_result
                
                tprint_success(f"   ✅ Stage 1 completed: {len(selected_features)} features selected")
                tprint_debug(f"   📊 Stage 1 method: {stage_1_result.get('method', 'unknown')}")
                
            except Exception as e:
                error_msg = f"Stage 1 mRMR+Spearman combination failed: {e}"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Stage 2: RFE with percentage-based step size
            tprint("📊 Stage 2: RFE with percentage-based step size")
            tprint_debug(f"   🔍 Input features for Stage 2: {len(selected_features)}")
            
            try:
                stage_2_result = self._stage_2_progressive_refinement(X, y, selected_features)
                selected_features = stage_2_result['selected_features']
                stage_results['stage_2'] = stage_2_result
                
                tprint_success(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
                tprint_debug(f"   📊 Stage 2 method: {stage_2_result.get('method', 'unknown')}")
                tprint_debug(f"   📊 Bootstrap/CV used: {stage_2_result.get('use_bootstrap_cv', False)}")
                
            except Exception as e:
                error_msg = f"Stage 2 RFE progressive refinement failed: {e}"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Calculate final metrics
            tprint_debug("📊 Calculating final performance metrics")
            try:
                performance_metrics = self._calculate_performance_metrics(X[selected_features], y)
                tprint_success("   ✅ Performance metrics calculated")
                tprint_debug(f"   📊 Final feature count: {performance_metrics.get('n_features', 0)}")
                tprint_debug(f"   📊 Sample count: {performance_metrics.get('n_samples', 0)}")
            except Exception as e:
                tprint_warning(f"⚠️ Performance metrics calculation failed: {e}")
                performance_metrics = {}
            
            # Create result
            execution_time = time.time() - start_time
            tprint_debug("📊 Creating final result object")
            
            result = FeatureSelectionResult(
                selected_features=selected_features,
                feature_importance=feature_importance,
                feature_scores=feature_scores,
                performance_metrics=performance_metrics,
                validation_scores={},
                config_used=self.config,
                execution_time=execution_time,
                memory_usage={},
                stage_results=stage_results,
                success=True
            )
            
            tprint_success(f"✅ Feature selection completed successfully in {execution_time:.2f}s")
            tprint_info(f"   📊 Final result: {len(selected_features)} features selected from {len(X.columns)}")
            tprint_info(f"   📊 Reduction ratio: {len(selected_features)/len(X.columns):.1%}")
            
            # Log final feature list
            tprint_debug("📊 Selected features:")
            for i, feature in enumerate(selected_features, 1):
                tprint_debug(f"   {i:2d}. {feature}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Feature selection failed: {e}"
            tprint_error(f"❌ {error_msg}")
            tprint_debug(f"   🔍 Exception type: {type(e).__name__}")
            tprint_debug(f"   🔍 Execution time before failure: {execution_time:.2f}s")
            
            return FeatureSelectionResult(
                selected_features=[],
                feature_importance={},
                feature_scores={},
                performance_metrics={},
                validation_scores={},
                config_used=self.config,
                execution_time=execution_time,
                memory_usage={},
                success=False,
                error_message=str(e)
            )


    def _calculate_performance_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate performance metrics for selected features."""
        tprint_debug("📊 Calculating performance metrics")
        
        metrics = {
            'n_features': len(X.columns),
            'n_samples': len(X),
            'feature_diversity': len(set([col.split('_')[0] for col in X.columns])),
            'data_quality': {
                'missing_ratio': X.isnull().sum().sum() / (len(X) * len(X.columns)),
                'variance_ratio': X.var().mean() / X.var().std() if X.var().std() > 0 else 0
            }
        }
        
        # Add VectorBT performance stats if available
        if self.vectorbt_enabled:
            try:
                vectorbt_stats = get_vectorbt_performance_stats()
                metrics['vectorbt_stats'] = vectorbt_stats
            except Exception as e:
                tprint_warning(f"⚠️ Could not get VectorBT stats: {e}")
        
        return metrics

    def _stage_1_mrmr_spearman_combination(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Stage 1: mRMR + Spearman combination (70% mRMR + 30% Spearman).
        
        Selects top 50% of features above target using weighted combination.
        """
        tprint_debug("🔍 Stage 1: mRMR + Spearman combination")
        tprint_debug(f"   📊 Input features: {len(X.columns)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        try:
            target_features = self.config.target_features
            features_above_target = len(X.columns) - target_features
            target_ratio = self.config.stage1_target_ratio
            features_to_select = max(target_features, int(len(X.columns) * target_ratio))
            
            tprint_debug(f"   📊 Target features: {target_features}")
            tprint_debug(f"   📊 Features above target: {features_above_target}")
            tprint_debug(f"   📊 Target ratio: {target_ratio:.1%}")
            tprint_debug(f"   📊 Features to select: {features_to_select}")
            
            # Calculate mRMR scores (70% weight)
            tprint_debug("   📊 Calculating mRMR scores (70% weight)")
            mrmr_scores = self._calculate_mrmr_scores(X, y)
            
            # Calculate Spearman scores (30% weight)
            tprint_debug("   📊 Calculating Spearman scores (30% weight)")
            spearman_scores = self._calculate_spearman_scores(X, y)
            
            # Combine scores with weights
            tprint_debug("   📊 Combining scores with weights")
            mrmr_weight = self.config.stage1_mrmr_weight
            spearman_weight = self.config.stage1_spearman_weight
            
            combined_scores = (
                mrmr_scores * mrmr_weight + 
                spearman_scores * spearman_weight
            )
            
            # Select top features
            selected_features = self._select_top_features(
                X.columns.tolist(), combined_scores, features_to_select
            )
            
            tprint_debug(f"   ✅ Stage 1 completed: {len(selected_features)} features selected")
            
            return {
                'selected_features': selected_features,
                'mrmr_scores': mrmr_scores.to_dict(),
                'spearman_scores': spearman_scores.to_dict(),
                'combined_scores': combined_scores.to_dict(),
                'target_count': features_to_select,
                'method': 'mrmr_spearman_combination'
            }
            
        except Exception as e:
            error_msg = f"Stage 1 mRMR+Spearman combination failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _stage_2_progressive_refinement(self, X: pd.DataFrame, y: pd.Series, 
                                      current_features: List[str]) -> Dict[str, Any]:
        """
        Stage 2: Progressive refinement using RFE with percentage-based step size.
        
        Uses RFE to recursively remove 10% of features above target in each round.
        Uses bootstrap stability and CV only when 40+ features away from target.
        """
        tprint_debug("🔍 Stage 2: Progressive refinement with RFE")
        tprint_debug(f"   📊 Input features: {len(current_features)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        try:
            target_features = self.config.target_features
            current_features = current_features.copy()
            
            tprint_debug(f"   📊 Target features: {target_features}")
            tprint_debug(f"   📊 RFE step percentage: {self.config.rfe_step_size:.1%}")
            tprint_debug(f"   📊 Bootstrap/CV threshold: {self.config.stage2_bootstrap_cv_threshold} features")
            
            # Check if we should use bootstrap stability and CV
            features_above_target = len(current_features) - target_features
            use_bootstrap_cv = features_above_target >= self.config.stage2_bootstrap_cv_threshold
            tprint_debug(f"   📊 Use bootstrap stability and CV: {use_bootstrap_cv} (threshold: {self.config.stage2_bootstrap_cv_threshold})")
            
            # Use RFE with percentage-based step size
            selected_features = self._rfe_with_percentage_step(
                X[current_features], y, current_features, target_features, use_bootstrap_cv
            )
            
            tprint_debug(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
            
            return {
                'selected_features': selected_features,
                'target_count': target_features,
                'method': 'rfe_percentage_based',
                'use_bootstrap_cv': use_bootstrap_cv
            }
            
        except Exception as e:
            error_msg = f"Stage 2 progressive refinement failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _rfe_with_percentage_step(self, X: pd.DataFrame, y: pd.Series, 
                                 feature_names: List[str], target_features: int,
                                 use_bootstrap_cv: bool = False) -> List[str]:
        """
        Recursive Feature Elimination with percentage-based step size.
        
        Removes 10% of features above target in each RFE round, recursively.
        """
        tprint_debug("🔍 Starting RFE with percentage-based step size")
        
        try:
            current_features = feature_names.copy()
            current_X = X.copy()
            rfe_rounds = []
            
            while len(current_features) > target_features:
                features_above_target = len(current_features) - target_features
                
                # Calculate step size as percentage of features above target
                step_size = max(1, int(features_above_target * self.config.rfe_step_size))
                tprint_debug(f"   📊 RFE Round: {len(current_features)} features, {features_above_target} above target")
                tprint_debug(f"   📊 Step size: {step_size} features (10% of {features_above_target})")
                
                # Calculate feature importance using ensemble methods
                feature_scores = self._calculate_ensemble_feature_scores(
                    current_X, y, use_bootstrap_cv=use_bootstrap_cv
                )
                
                # Select features to remove (lowest scores)
                features_to_remove = self._select_features_to_remove(
                    current_features, feature_scores, step_size
                )
                
                # Remove features
                current_features = [f for f in current_features if f not in features_to_remove]
                current_X = current_X.drop(columns=features_to_remove)
                
                tprint_debug(f"   📊 Removed {len(features_to_remove)} features: {features_to_remove}")
                
                rfe_rounds.append({
                    'round': len(rfe_rounds) + 1,
                    'features_remaining': len(current_features),
                    'features_removed': len(features_to_remove),
                    'step_size': step_size,
                    'features_above_target': features_above_target,
                    'features_removed_list': features_to_remove
                })
                
                # Safety check to prevent infinite loop
                if len(rfe_rounds) > 100:
                    tprint_warning("   ⚠️ Maximum RFE rounds reached, stopping")
                    break
            
            tprint_debug(f"   ✅ RFE completed: {len(current_features)} features selected in {len(rfe_rounds)} rounds")
            
            return current_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ RFE with percentage step failed: {e}")
            # Fallback to simple correlation-based selection
            return self._fallback_feature_selection(X, y, feature_names, target_features)

    def _fallback_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   feature_names: List[str], target_features: int) -> List[str]:
        """Fallback feature selection using simple correlation."""
        try:
            tprint_debug("   📊 Using fallback correlation-based selection")
            
            # Calculate correlation scores
            correlations = []
            for col in X.columns:
                corr, _ = spearmanr(X[col], y)
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            
            # Select top features
            feature_scores = pd.Series(correlations, index=X.columns)
            sorted_features = feature_scores.sort_values(ascending=False)
            
            selected_features = sorted_features.head(target_features).index.tolist()
            
            tprint_debug(f"   ✅ Fallback selection completed: {len(selected_features)} features selected")
            
            return selected_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Fallback selection failed: {e}")
            # Last resort: select first target_features
            return feature_names[:target_features]

    def _calculate_mrmr_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate mRMR scores using VectorBT optimization."""
        try:
            if VECTORBT_MRMR_AVAILABLE:
                # Use VectorBT mRMR selector
                mrmr_selector = VectorBTMRMRSelector()
                result = mrmr_selector.select_features(
                    X.values, y.values, 
                    n_features=min(len(X.columns), 100)  # Limit for performance
                )
                return pd.Series(result['feature_scores'], index=X.columns)
            else:
                # Fallback to simple correlation
                tprint_warning("   ⚠️ VectorBT mRMR not available, using Spearman correlation")
                return self.spearman_abs_vectorized(X, y)
                
        except Exception as e:
            tprint_warning(f"   ⚠️ mRMR calculation failed: {e}")
            return self.spearman_abs_vectorized(X, y)

    def _calculate_spearman_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate Spearman correlation scores."""
        try:
            return self.spearman_abs_vectorized(X, y)
        except Exception as e:
            tprint_warning(f"   ⚠️ Spearman calculation failed: {e}")
            return pd.Series(0.0, index=X.columns)

    def _calculate_ensemble_feature_scores(self, X: pd.DataFrame, y: pd.Series, 
                                         use_bootstrap_cv: bool = False) -> pd.Series:
        """Calculate ensemble feature scores using multiple methods."""
        try:
            ensemble_scores = {}
            
            # LGBM-SHAP scores (40% weight)
            if LIGHTGBM_SHAP_AVAILABLE:
                lgbm_scores = self._calculate_lgbm_shap_scores(X, y)
                ensemble_scores['lgbm_shap'] = lgbm_scores
            
            # LASSO ensemble scores (30% weight)
            if SKLEARN_AVAILABLE:
                lasso_scores = self._calculate_lasso_ensemble_scores(X, y)
                ensemble_scores['lasso_ensemble'] = lasso_scores
                
                # RFE scores (20% weight)
                rfe_scores = self._calculate_rfe_scores(X, y)
                ensemble_scores['rfe'] = rfe_scores
            
            # Bootstrap stability scores (10% weight) - only if enabled
            if use_bootstrap_cv and SKLEARN_AVAILABLE:
                bootstrap_scores = self._calculate_bootstrap_stability_scores(X, y)
                ensemble_scores['bootstrap_stability'] = bootstrap_scores
            
            # Combine scores with weights
            return self._combine_ensemble_scores(ensemble_scores)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Ensemble scoring failed: {e}")
            # Fallback to simple correlation
            return self.spearman_abs_vectorized(X, y)

    def _calculate_lgbm_shap_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate LGBM-SHAP feature importance scores."""
        try:
            # Train LightGBM model
            lgb_params = self.config.lgbm_params.copy()
            lgb_params['verbose'] = -1  # Suppress output
            
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Calculate feature importance as mean absolute SHAP values
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            return pd.Series(feature_importance, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ LGBM-SHAP calculation failed: {e}")
            return pd.Series(0.0, index=X.columns)

    def _calculate_lasso_ensemble_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate LASSO ensemble feature importance scores."""
        try:
            # Use LassoCV for automatic alpha selection
            lasso = LassoCV(
                alphas=np.logspace(
                    self.config.lasso_alpha_range[0],
                    self.config.lasso_alpha_range[1],
                    self.config.lasso_n_alphas
                ),
                cv=self.config.lasso_cv_folds,
                random_state=42
            )
            
            lasso.fit(X, y)
            
            # Feature importance as absolute coefficients
            feature_importance = np.abs(lasso.coef_)
            
            return pd.Series(feature_importance, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ LASSO ensemble calculation failed: {e}")
            return pd.Series(0.0, index=X.columns)

    def _calculate_rfe_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate RFE feature importance scores."""
        try:
            # Use RandomForest as base estimator for RFE
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            
            # Calculate feature importance
            rf.fit(X, y)
            feature_importance = rf.feature_importances_
            
            return pd.Series(feature_importance, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ RFE calculation failed: {e}")
            return pd.Series(0.0, index=X.columns)

    def _calculate_bootstrap_stability_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate bootstrap stability scores."""
        try:
            n_samples = self.config.bootstrap_n_samples
            sample_ratio = self.config.bootstrap_sample_ratio
            stability_threshold = self.config.stability_threshold
            
            feature_stability = np.zeros(len(X.columns))
            
            for _ in range(n_samples):
                # Bootstrap sample
                n_samples_subset = int(len(X) * sample_ratio)
                indices = np.random.choice(len(X), n_samples_subset, replace=True)
                
                X_bootstrap = X.iloc[indices]
                y_bootstrap = y.iloc[indices]
                
                # Calculate feature importance for this bootstrap sample
                try:
                    rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
                    rf.fit(X_bootstrap, y_bootstrap)
                    importance = rf.feature_importances_
                    
                    # Count features above threshold
                    feature_stability += (importance > stability_threshold).astype(int)
                    
                except Exception as e:
                    tprint_warning(f"   ⚠️ Bootstrap iteration failed: {e}")
                    continue
            
            # Normalize by number of successful bootstrap samples
            feature_stability = feature_stability / n_samples
            
            return pd.Series(feature_stability, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Bootstrap stability calculation failed: {e}")
            return pd.Series(0.0, index=X.columns)

    def _combine_ensemble_scores(self, ensemble_scores: Dict[str, pd.Series]) -> pd.Series:
        """Combine ensemble scores using configured weights."""
        try:
            if not ensemble_scores:
                return pd.Series(0.0, index=list(ensemble_scores.values())[0].index)
            
            # Get weights from config
            weights = self.config.ensemble_weights
            
            # Normalize scores to 0-1 range
            normalized_scores = {}
            for method, scores in ensemble_scores.items():
                if scores.max() > 0:
                    normalized_scores[method] = scores / scores.max()
                else:
                    normalized_scores[method] = scores
            
            # Combine with weights
            combined_scores = pd.Series(0.0, index=list(ensemble_scores.values())[0].index)
            
            for method, scores in normalized_scores.items():
                weight = weights.get(method, 0.0)
                combined_scores += scores * weight
            
            return combined_scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Score combination failed: {e}")
            return pd.Series(0.0, index=list(ensemble_scores.values())[0].index)

    def _select_features_to_remove(self, feature_names: List[str], 
                                  feature_scores: pd.Series, count: int) -> List[str]:
        """Select features to remove based on lowest scores."""
        try:
            if count <= 0:
                return []
            
            # Get scores for current features
            current_scores = feature_scores[feature_names]
            
            # Select features with lowest scores
            bottom_features = current_scores.nsmallest(count).index.tolist()
            
            return bottom_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Feature selection failed: {e}")
            return feature_names[:count] if count > 0 else []

    def _select_top_features(self, feature_names: List[str], 
                           feature_scores: pd.Series, count: int) -> List[str]:
        """Select top features based on highest scores."""
        try:
            if count <= 0:
                return []
            
            # Get scores for current features
            current_scores = feature_scores[feature_names]
            
            # Select features with highest scores
            top_features = current_scores.nlargest(count).index.tolist()
            
            return top_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Top feature selection failed: {e}")
            return feature_names[:count] if count > 0 else []