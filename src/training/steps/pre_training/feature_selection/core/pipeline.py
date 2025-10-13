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

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       symbol: str = "BTCUSDT", exchange: str = "binance", 
                       timeframe: str = "15m") -> FeatureSelectionResult:
        """
        Execute multi-stage feature selection.
        
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
        
        try:
            # Set thread limits
            self._set_thread_limits()
            
            # Display VectorBT status
            self._display_vectorbt_status()
            
            # Initialize result tracking
            stage_results = {}
            selected_features = X.columns.tolist()
            feature_importance = {}
            feature_scores = {}
            
            # Stage 1: Initial filtering (120 -> 100)
            tprint("📊 Stage 1: Initial filtering (120 -> 100)")
            stage_1_result = self._stage_1_filtering(X, y, selected_features)
            selected_features = stage_1_result['selected_features']
            stage_results['stage_1'] = stage_1_result
            
            # Stage 2: Correlation and redundancy removal (100 -> 80)
            tprint("📊 Stage 2: Correlation and redundancy removal (100 -> 80)")
            stage_2_result = self._stage_2_correlation_filtering(X[selected_features], y)
            selected_features = stage_2_result['selected_features']
            stage_results['stage_2'] = stage_2_result
            
            # Stage 3: Final optimization (80 -> 60)
            tprint("📊 Stage 3: Final optimization (80 -> 60)")
            stage_3_result = self._stage_3_final_optimization(X[selected_features], y)
            selected_features = stage_3_result['selected_features']
            stage_results['stage_3'] = stage_3_result
            
            # Calculate final metrics
            execution_time = time.time() - start_time
            performance_metrics = self._calculate_performance_metrics(X[selected_features], y)
            
            # Create result
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
            
            tprint_success(f"✅ Feature selection completed in {execution_time:.2f}s")
            tprint(f"   📊 Selected {len(selected_features)} features from {len(X.columns)}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return FeatureSelectionResult(
                selected_features=[],
                feature_importance={},
                feature_scores={},
                performance_metrics={},
                validation_scores={},
                config_used=self.config,
                execution_time=time.time() - start_time,
                memory_usage={},
                success=False,
                error_message=str(e)
            )

    def _stage_1_filtering(self, X: pd.DataFrame, y: pd.Series, 
                          initial_features: List[str]) -> Dict[str, Any]:
        """Stage 1: Initial filtering using basic criteria."""
        tprint_debug("🔍 Stage 1: Initial filtering")
        
        # Calculate feature importance using Spearman correlation
        importance_scores = self.spearman_abs_vectorized(X, y)
        
        # Select top features based on importance
        target_count = min(self.config.stage_1_target, len(initial_features))
        selected_features = self.top_k(
            initial_features, 
            importance_scores.values, 
            target_count
        )
        
        return {
            'selected_features': selected_features,
            'importance_scores': importance_scores.to_dict(),
            'target_count': target_count,
            'method': 'spearman_correlation'
        }

    def _stage_2_correlation_filtering(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Stage 2: Remove highly correlated features."""
        tprint_debug("🔍 Stage 2: Correlation filtering")
        
        # Calculate correlation matrix
        if self.vectorbt_enabled:
            corr_matrix = self._vectorbt_optimized_correlation_matrix(X)
        else:
            corr_matrix = X.corr().values
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(X.columns)):
            for j in range(i + 1, len(X.columns)):
                if abs(corr_matrix[i, j]) > self.config.quality_config.min_correlation_threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        # Remove features with highest average correlation
        features_to_remove = set()
        for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            if i not in features_to_remove and j not in features_to_remove:
                # Remove the feature with lower individual importance
                feature_i_importance = abs(corr_matrix[i, :]).mean()
                feature_j_importance = abs(corr_matrix[j, :]).mean()
                
                if feature_i_importance < feature_j_importance:
                    features_to_remove.add(i)
                else:
                    features_to_remove.add(j)
        
        # Select remaining features
        remaining_indices = [i for i in range(len(X.columns)) if i not in features_to_remove]
        selected_features = [X.columns[i] for i in remaining_indices]
        
        # Ensure we don't go below target
        target_count = min(self.config.stage_2_target, len(selected_features))
        if len(selected_features) > target_count:
            # Use importance to select final features
            importance_scores = self.spearman_abs_vectorized(X[selected_features], y)
            selected_features = self.top_k(
                selected_features,
                importance_scores.values,
                target_count
            )
        
        return {
            'selected_features': selected_features,
            'correlation_matrix': corr_matrix.tolist(),
            'high_corr_pairs': len(high_corr_pairs),
            'features_removed': len(features_to_remove),
            'target_count': target_count
        }

    def _stage_3_final_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Stage 3: Final optimization using ensemble methods."""
        tprint_debug("🔍 Stage 3: Final optimization")
        
        # Use multiple methods for final selection
        methods_scores = {}
        
        # Method 1: Spearman correlation
        spearman_scores = self.spearman_abs_vectorized(X, y)
        methods_scores['spearman'] = spearman_scores
        
        # Method 2: Redundancy analysis
        redundancy_scores = self.redundancy_mean_abs_spearman_blocked(X)
        methods_scores['redundancy'] = redundancy_scores
        
        # Combine scores using weighted average
        combined_scores = (
            spearman_scores * 0.7 + 
            (1 - redundancy_scores / redundancy_scores.max()) * 0.3
        )
        
        # Select final features
        target_count = min(self.config.stage_3_target, len(X.columns))
        selected_features = self.top_k(
            X.columns.tolist(),
            combined_scores.values,
            target_count
        )
        
        return {
            'selected_features': selected_features,
            'combined_scores': combined_scores.to_dict(),
            'methods_scores': {k: v.to_dict() for k, v in methods_scores.items()},
            'target_count': target_count
        }

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