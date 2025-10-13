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

# Import enhanced pipeline
from .enhanced_pipeline import EnhancedMultiStageFeatureSelector

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
    """Multi-stage feature selection using RandomForest and SHAP with vectorization and caching."""

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
            
            # Check if new enhanced pipeline is enabled
            if hasattr(self.config, 'enable_new_pipeline') and self.config.enable_new_pipeline:
                tprint("🚀 Using enhanced multi-stage pipeline")
                tprint_info("   📊 Stage 1: mRMR + Spearman combination (70% mRMR + 30% Spearman)")
                tprint_info("   📊 Stage 2: Progressive refinement using LGBM-SHAP and LASSO ensemble")
                
                # Use enhanced pipeline
                enhanced_selector = EnhancedMultiStageFeatureSelector(self.config)
                return enhanced_selector.select_features(X, y, symbol, exchange, timeframe)
            
            # Fallback to original pipeline
            tprint("📊 Using original 3-stage pipeline")
            
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
            
            # Stage 1: Initial filtering (120 -> 100)
            tprint("📊 Stage 1: Initial filtering (120 -> 100)")
            tprint_debug(f"   🔍 Input features for Stage 1: {len(selected_features)}")
            
            try:
                stage_1_result = self._stage_1_filtering(X, y, selected_features)
                selected_features = stage_1_result['selected_features']
                stage_results['stage_1'] = stage_1_result
                
                tprint_success(f"   ✅ Stage 1 completed: {len(selected_features)} features selected")
                tprint_debug(f"   📊 Stage 1 method: {stage_1_result.get('method', 'unknown')}")
                
            except Exception as e:
                error_msg = f"Stage 1 filtering failed: {e}"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Stage 2: Correlation and redundancy removal (100 -> 80)
            tprint("📊 Stage 2: Correlation and redundancy removal (100 -> 80)")
            tprint_debug(f"   🔍 Input features for Stage 2: {len(selected_features)}")
            
            try:
                stage_2_result = self._stage_2_correlation_filtering(X[selected_features], y)
                selected_features = stage_2_result['selected_features']
                stage_results['stage_2'] = stage_2_result
                
                tprint_success(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
                tprint_debug(f"   📊 High correlation pairs removed: {stage_2_result.get('high_corr_pairs', 0)}")
                tprint_debug(f"   📊 Features removed: {stage_2_result.get('features_removed', 0)}")
                
            except Exception as e:
                error_msg = f"Stage 2 correlation filtering failed: {e}"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Stage 3: Final optimization (80 -> 60)
            tprint("📊 Stage 3: Final optimization (80 -> 60)")
            tprint_debug(f"   🔍 Input features for Stage 3: {len(selected_features)}")
            
            try:
                stage_3_result = self._stage_3_final_optimization(X[selected_features], y)
                selected_features = stage_3_result['selected_features']
                stage_results['stage_3'] = stage_3_result
                
                tprint_success(f"   ✅ Stage 3 completed: {len(selected_features)} features selected")
                tprint_debug(f"   📊 Target count: {stage_3_result.get('target_count', 0)}")
                
            except Exception as e:
                error_msg = f"Stage 3 final optimization failed: {e}"
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

    def _stage_1_filtering(self, X: pd.DataFrame, y: pd.Series, 
                          initial_features: List[str]) -> Dict[str, Any]:
        """Stage 1: Initial filtering using basic criteria with VectorBT optimization."""
        tprint_debug("🔍 Stage 1: Initial filtering")
        tprint_debug(f"   📊 Input features: {len(initial_features)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        try:
            # FAST FAIL: Check input validity
            if X.empty or y.empty:
                error_msg = "Empty input data in Stage 1"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            if len(initial_features) == 0:
                error_msg = "No initial features provided"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            tprint_debug("   ✅ Input validation passed")
            
            # Calculate feature importance using Spearman correlation with VectorBT optimization
            tprint_debug("   📊 Calculating Spearman correlation importance scores")
            try:
                if self.vectorbt_enabled and hasattr(self, 'vectorization_manager'):
                    tprint_debug("   🚀 Using VectorBT for correlation calculation")
                    # Use VectorBT for optimized correlation calculation
                    importance_scores = self._vectorbt_optimized_spearman_correlation(X, y)
                else:
                    tprint_debug("   📊 Using standard Spearman correlation")
                    importance_scores = self.spearman_abs_vectorized(X, y)
                
                tprint_debug(f"   ✅ Importance scores calculated: {len(importance_scores)} features")
                tprint_debug(f"   📊 Score range: {importance_scores.min():.6f} - {importance_scores.max():.6f}")
                
            except Exception as e:
                tprint_warning(f"   ⚠️ VectorBT correlation failed, falling back to standard: {e}")
                importance_scores = self.spearman_abs_vectorized(X, y)
            
            # Select top features based on importance
            target_count = min(self.config.stage_1_target, len(initial_features))
            tprint_debug(f"   📊 Target count: {target_count}")
            
            selected_features = self.top_k(
                initial_features, 
                importance_scores.values, 
                target_count
            )
            
            tprint_debug(f"   ✅ Selected {len(selected_features)} features")
            tprint_debug(f"   📊 Selection method: spearman_correlation")
            
            # Log top features
            if len(selected_features) > 0:
                tprint_debug("   📊 Top 5 selected features:")
                for i, feature in enumerate(selected_features[:5], 1):
                    score = importance_scores.get(feature, 0.0)
                    tprint_debug(f"   {i}. {feature}: {score:.6f}")
            
            return {
                'selected_features': selected_features,
                'importance_scores': importance_scores.to_dict(),
                'target_count': target_count,
                'method': 'spearman_correlation',
                'vectorbt_used': self.vectorbt_enabled
            }
            
        except Exception as e:
            error_msg = f"Stage 1 filtering failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _stage_2_correlation_filtering(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Stage 2: Remove highly correlated features with VectorBT optimization."""
        tprint_debug("🔍 Stage 2: Correlation filtering")
        tprint_debug(f"   📊 Input features: {len(X.columns)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        try:
            # FAST FAIL: Check input validity
            if X.empty or y.empty:
                error_msg = "Empty input data in Stage 2"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            if len(X.columns) == 0:
                error_msg = "No features to process in Stage 2"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            tprint_debug("   ✅ Input validation passed")
            
            # Calculate correlation matrix with VectorBT optimization
            tprint_debug("   📊 Calculating correlation matrix")
            try:
                if self.vectorbt_enabled and hasattr(self, 'vectorization_manager'):
                    tprint_debug("   🚀 Using VectorBT for correlation matrix calculation")
                    corr_matrix = self._vectorbt_optimized_correlation_matrix(X)
                else:
                    tprint_debug("   📊 Using standard correlation calculation")
                    corr_matrix = X.corr().values
                
                tprint_debug(f"   ✅ Correlation matrix calculated: {corr_matrix.shape}")
                
            except Exception as e:
                tprint_warning(f"   ⚠️ VectorBT correlation failed, using standard: {e}")
                corr_matrix = X.corr().values
            
            # Find highly correlated pairs
            tprint_debug("   🔍 Finding highly correlated feature pairs")
            threshold = self.config.quality_config.min_correlation_threshold
            tprint_debug(f"   📊 Correlation threshold: {threshold}")
            
            high_corr_pairs = []
            for i in range(len(X.columns)):
                for j in range(i + 1, len(X.columns)):
                    corr_value = abs(corr_matrix[i, j])
                    if corr_value > threshold:
                        high_corr_pairs.append((i, j, corr_matrix[i, j]))
            
            tprint_debug(f"   📊 Found {len(high_corr_pairs)} highly correlated pairs")
            
            # Remove features with highest average correlation
            tprint_debug("   🔍 Removing redundant features")
            features_to_remove = set()
            
            for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                if i not in features_to_remove and j not in features_to_remove:
                    # Calculate average correlation for each feature
                    feature_i_importance = abs(corr_matrix[i, :]).mean()
                    feature_j_importance = abs(corr_matrix[j, :]).mean()
                    
                    # Remove the feature with lower individual importance
                    if feature_i_importance < feature_j_importance:
                        features_to_remove.add(i)
                        tprint_debug(f"   🗑️ Removing feature {i} ({X.columns[i]}) - lower importance")
                    else:
                        features_to_remove.add(j)
                        tprint_debug(f"   🗑️ Removing feature {j} ({X.columns[j]}) - lower importance")
            
            tprint_debug(f"   📊 Features to remove: {len(features_to_remove)}")
            
            # Select remaining features
            remaining_indices = [i for i in range(len(X.columns)) if i not in features_to_remove]
            selected_features = [X.columns[i] for i in remaining_indices]
            
            tprint_debug(f"   📊 Features after correlation filtering: {len(selected_features)}")
            
            # Ensure we don't go below target
            target_count = min(self.config.stage_2_target, len(selected_features))
            tprint_debug(f"   📊 Target count: {target_count}")
            
            if len(selected_features) > target_count:
                tprint_debug("   📊 Applying final importance-based selection")
                # Use importance to select final features
                try:
                    if self.vectorbt_enabled and hasattr(self, 'vectorization_manager'):
                        importance_scores = self._vectorbt_optimized_spearman_correlation(X[selected_features], y)
                    else:
                        importance_scores = self.spearman_abs_vectorized(X[selected_features], y)
                    
                    selected_features = self.top_k(
                        selected_features,
                        importance_scores.values,
                        target_count
                    )
                    tprint_debug(f"   ✅ Final selection completed: {len(selected_features)} features")
                    
                except Exception as e:
                    tprint_warning(f"   ⚠️ Final selection failed: {e}")
                    # Keep the first target_count features as fallback
                    selected_features = selected_features[:target_count]
                    tprint_debug(f"   📊 Using fallback selection: {len(selected_features)} features")
            
            tprint_debug(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
            
            return {
                'selected_features': selected_features,
                'correlation_matrix': corr_matrix.tolist(),
                'high_corr_pairs': len(high_corr_pairs),
                'features_removed': len(features_to_remove),
                'target_count': target_count,
                'vectorbt_used': self.vectorbt_enabled
            }
            
        except Exception as e:
            error_msg = f"Stage 2 correlation filtering failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    def _stage_3_final_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Stage 3: Final optimization using ensemble methods with VectorBT optimization."""
        tprint_debug("🔍 Stage 3: Final optimization")
        tprint_debug(f"   📊 Input features: {len(X.columns)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        try:
            # FAST FAIL: Check input validity
            if X.empty or y.empty:
                error_msg = "Empty input data in Stage 3"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            if len(X.columns) == 0:
                error_msg = "No features to process in Stage 3"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            tprint_debug("   ✅ Input validation passed")
            
            # Use multiple methods for final selection
            tprint_debug("   📊 Calculating ensemble scores")
            methods_scores = {}
            
            # Method 1: Spearman correlation with VectorBT optimization
            tprint_debug("   📊 Method 1: Spearman correlation")
            try:
                if self.vectorbt_enabled and hasattr(self, 'vectorization_manager'):
                    tprint_debug("   🚀 Using VectorBT for Spearman correlation")
                    spearman_scores = self._vectorbt_optimized_spearman_correlation(X, y)
                else:
                    tprint_debug("   📊 Using standard Spearman correlation")
                    spearman_scores = self.spearman_abs_vectorized(X, y)
                
                methods_scores['spearman'] = spearman_scores
                tprint_debug(f"   ✅ Spearman scores calculated: {len(spearman_scores)} features")
                tprint_debug(f"   📊 Spearman range: {spearman_scores.min():.6f} - {spearman_scores.max():.6f}")
                
            except Exception as e:
                tprint_warning(f"   ⚠️ Spearman correlation failed: {e}")
                spearman_scores = pd.Series(0.0, index=X.columns)
                methods_scores['spearman'] = spearman_scores
            
            # Method 2: Redundancy analysis with VectorBT optimization
            tprint_debug("   📊 Method 2: Redundancy analysis")
            try:
                if self.vectorbt_enabled and hasattr(self, 'vectorization_manager'):
                    tprint_debug("   🚀 Using VectorBT for redundancy analysis")
                    redundancy_scores = self._vectorbt_optimized_redundancy_analysis(X)
                else:
                    tprint_debug("   📊 Using standard redundancy analysis")
                    redundancy_scores = self.redundancy_mean_abs_spearman_blocked(X)
                
                methods_scores['redundancy'] = redundancy_scores
                tprint_debug(f"   ✅ Redundancy scores calculated: {len(redundancy_scores)} features")
                tprint_debug(f"   📊 Redundancy range: {redundancy_scores.min():.6f} - {redundancy_scores.max():.6f}")
                
            except Exception as e:
                tprint_warning(f"   ⚠️ Redundancy analysis failed: {e}")
                redundancy_scores = pd.Series(0.0, index=X.columns)
                methods_scores['redundancy'] = redundancy_scores
            
            # Combine scores using weighted average
            tprint_debug("   📊 Combining scores with weighted average")
            try:
                # Normalize redundancy scores to 0-1 range
                if redundancy_scores.max() > 0:
                    normalized_redundancy = 1 - (redundancy_scores / redundancy_scores.max())
                else:
                    normalized_redundancy = pd.Series(1.0, index=X.columns)
                
                # Weighted combination
                combined_scores = (
                    spearman_scores * 0.7 + 
                    normalized_redundancy * 0.3
                )
                
                tprint_debug(f"   ✅ Combined scores calculated: {len(combined_scores)} features")
                tprint_debug(f"   📊 Combined range: {combined_scores.min():.6f} - {combined_scores.max():.6f}")
                
            except Exception as e:
                tprint_warning(f"   ⚠️ Score combination failed: {e}")
                # Fallback to just Spearman scores
                combined_scores = spearman_scores
                tprint_debug("   📊 Using Spearman scores as fallback")
            
            # Select final features
            target_count = min(self.config.stage_3_target, len(X.columns))
            tprint_debug(f"   📊 Target count: {target_count}")
            
            selected_features = self.top_k(
                X.columns.tolist(),
                combined_scores.values,
                target_count
            )
            
            tprint_debug(f"   ✅ Final selection completed: {len(selected_features)} features")
            
            # Log top features
            if len(selected_features) > 0:
                tprint_debug("   📊 Top 5 final selected features:")
                for i, feature in enumerate(selected_features[:5], 1):
                    score = combined_scores.get(feature, 0.0)
                    spearman_score = spearman_scores.get(feature, 0.0)
                    redundancy_score = redundancy_scores.get(feature, 0.0)
                    tprint_debug(f"   {i}. {feature}: combined={score:.6f}, spearman={spearman_score:.6f}, redundancy={redundancy_score:.6f}")
            
            return {
                'selected_features': selected_features,
                'combined_scores': combined_scores.to_dict(),
                'methods_scores': {k: v.to_dict() for k, v in methods_scores.items()},
                'target_count': target_count,
                'vectorbt_used': self.vectorbt_enabled
            }
            
        except Exception as e:
            error_msg = f"Stage 3 final optimization failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

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