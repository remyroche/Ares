"""
Base Cross-Validation Splitter

Provides shared cross-validation logic with embargo support for both
feature_generation and feature_engineering_roadmap lookback optimization.
"""

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import logging

# Import common utilities
from ..utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager
)

logger = logging.getLogger(__name__)


class BaseCVSplitter:
    """
    Base class for time series cross-validation with embargo.
    
    This class provides common CV splitting logic that can be used by both:
    - feature_generation/utils/optimization/lookback_optimizer.py
    - feature_engineering_roadmap/lookback_selection.py
    
    The embargo feature helps prevent data leakage by creating a gap
    between training and validation sets.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        embargo_pct: float = 0.1,
        min_train_size: Optional[int] = None,
        use_vectorbt_optimization: bool = True,
        enable_gpu: bool = False
    ):
        """
        Initialize CV splitter with VectorBT optimization support.
        
        Args:
            n_folds: Number of folds for time series split
            embargo_pct: Percentage of validation data to skip as embargo
            min_train_size: Minimum training size (None = use sklearn default)
            use_vectorbt_optimization: Whether to use VectorBT optimization for any rolling operations
            enable_gpu: Whether to enable GPU acceleration
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseCVSplitter] Initializing with n_folds={n_folds}, embargo_pct={embargo_pct}, use_vectorbt={use_vectorbt_optimization}", color="cyan")
        
        self.n_folds = n_folds
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size
        self.use_vectorbt_optimization = use_vectorbt_optimization and VECTORBT_OPTIMIZER_AVAILABLE
        self.enable_gpu = enable_gpu
        
        # Initialize VectorBT optimization components
        if self.use_vectorbt_optimization:
            if TPRINT_AVAILABLE:
                tprint("🔧 [BaseCVSplitter] Initializing VectorBT optimization components", color="blue")
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.enable_gpu,
                enable_parallel=True,
                memory_efficient=True
            )
            self.vectorization_manager = get_unified_vectorization_manager()
            if TPRINT_AVAILABLE:
                tprint("✅ [BaseCVSplitter] VectorBT optimization components initialized", color="green")
        else:
            self.rolling_optimizer = None
            self.vectorization_manager = None
            if TPRINT_AVAILABLE:
                tprint("⚠️  [BaseCVSplitter] VectorBT optimization disabled", color="yellow")
        
        # Performance tracking
        self.performance_stats = {
            'total_splits': 0,
            'vectorbt_operations': 0,
            'optimization_operations': 0,
            'total_time': 0.0
        }
        
        # Validation
        # Validate parameters with fast fail
        if not 0 <= embargo_pct <= 0.5:
            error_msg = f"embargo_pct must be between 0 and 0.5, got {embargo_pct}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [BaseCVSplitter] {error_msg}", color="red")
            raise ValueError(error_msg)
        
        if n_folds < 2:
            error_msg = f"n_folds must be at least 2, got {n_folds}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [BaseCVSplitter] {error_msg}", color="red")
            raise ValueError(error_msg)
        
        if TPRINT_AVAILABLE:
            tprint("✅ [BaseCVSplitter] Initialization completed successfully", color="green")
    
    def split_with_embargo(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Perform time series split with embargo between train/val sets.
        
        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Optional target Series (not used but kept for sklearn compatibility)
            
        Returns:
            List of (train_index, val_index) tuples
            
        Example:
            >>> splitter = BaseCVSplitter(n_folds=3, embargo_pct=0.1)
            >>> for train_idx, val_idx in splitter.split_with_embargo(X):
            ...     X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseCVSplitter] Starting split_with_embargo on {len(X)} samples with {self.n_folds} folds", color="cyan")
        
        if X.empty:
            error_msg = "Empty DataFrame provided to CV splitter"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [BaseCVSplitter] {error_msg}", color="red")
            raise ValueError(error_msg)
        
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Create TimeSeriesSplit
        tscv_kwargs = {'n_splits': self.n_folds}
        if self.min_train_size is not None:
            tscv_kwargs['test_size'] = max(1, (n_samples - self.min_train_size) // self.n_folds)
        
        tscv = TimeSeriesSplit(**tscv_kwargs)
        splits = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Apply embargo: skip first N samples of validation set
            if embargo_size > 0 and len(val_idx) > embargo_size:
                original_val_size = len(val_idx)
                val_idx = val_idx[embargo_size:]
                
                logger.debug(
                    f"Fold {fold_idx + 1}: Applied embargo of {embargo_size} samples "
                    f"(reduced validation from {original_val_size} to {len(val_idx)})"
                )
            
            # Only include fold if validation set is not empty
            if len(val_idx) > 0:
                # Convert to Index objects
                train_index = X.index[train_idx]
                val_index = X.index[val_idx]
                splits.append((train_index, val_index))
            else:
                logger.warning(
                    f"Fold {fold_idx + 1}: Validation set empty after embargo, skipping"
                )
        
        if not splits:
            if TPRINT_AVAILABLE:
                tprint("❌ [BaseCVSplitter] No valid splits generated - all validation sets were empty", color="red")
            logger.error("No valid splits generated - all validation sets were empty")
        else:
            if TPRINT_AVAILABLE:
                tprint(f"✅ [BaseCVSplitter] Generated {len(splits)} CV splits with embargo", color="green")
            logger.info(f"Generated {len(splits)} CV splits with embargo")
        
        return splits
    
    def get_n_splits(self, X: Optional[pd.DataFrame] = None) -> int:
        """
        Get number of splits.
        
        Args:
            X: Optional DataFrame (for sklearn compatibility)
            
        Returns:
            Number of splits (may be less than n_folds if embargo removes some)
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseCVSplitter] Getting number of splits: {self.n_folds}", color="cyan")
        return self.n_folds
    
    def analyze_feature_stability(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze feature stability across CV folds using VectorBT optimization.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            Dictionary containing stability metrics
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [BaseCVSplitter] Starting feature stability analysis on {X.shape[0]}x{X.shape[1]} data", color="cyan")
        
        if not self.use_vectorbt_optimization or self.rolling_optimizer is None:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [BaseCVSplitter] VectorBT optimization not available, using basic analysis", color="yellow")
            return self._basic_stability_analysis(X, y)
        
        try:
            splits = self.split_with_embargo(X, y)
            stability_metrics = {}
            
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_train, X_val = X.loc[train_idx], X.loc[val_idx]
                
                # Use VectorBT rolling operations for stability analysis
                fold_metrics = self._analyze_fold_stability_vectorbt(X_train, X_val)
                stability_metrics[f'fold_{fold_idx}'] = fold_metrics
                
                self.performance_stats['vectorbt_operations'] += 1
            
            # Aggregate metrics across folds
            aggregated_metrics = self._aggregate_stability_metrics(stability_metrics)
            self.performance_stats['optimization_operations'] += 1
            
            return aggregated_metrics
            
        except Exception as e:
            logger.warning(f"VectorBT stability analysis failed: {e}, using basic analysis")
            return self._basic_stability_analysis(X, y)
    
    def _analyze_fold_stability_vectorbt(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stability for a single fold using VectorBT optimization."""
        fold_metrics = {}
        
        for column in X_train.columns:
            train_data = X_train[column].dropna()
            val_data = X_val[column].dropna()
            
            if len(train_data) > 0 and len(val_data) > 0:
                # Use VectorBT rolling operations for statistical analysis
                train_mean = self.rolling_optimizer.rolling_mean(train_data, window=min(20, len(train_data)))
                val_mean = self.rolling_optimizer.rolling_mean(val_data, window=min(20, len(val_data)))
                
                # Calculate stability metrics
                mean_diff = abs(train_mean.mean() - val_mean.mean())
                std_ratio = val_data.std() / train_data.std() if train_data.std() > 0 else np.inf
                
                fold_metrics[column] = {
                    'mean_difference': mean_diff,
                    'std_ratio': std_ratio,
                    'stability_score': 1.0 / (1.0 + mean_diff + abs(1.0 - std_ratio))
                }
        
        return fold_metrics
    
    def _basic_stability_analysis(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Basic stability analysis without VectorBT optimization."""
        splits = self.split_with_embargo(X, y)
        stability_metrics = {}
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            
            fold_metrics = {}
            for column in X_train.columns:
                train_data = X_train[column].dropna()
                val_data = X_val[column].dropna()
                
                if len(train_data) > 0 and len(val_data) > 0:
                    mean_diff = abs(train_data.mean() - val_data.mean())
                    std_ratio = val_data.std() / train_data.std() if train_data.std() > 0 else np.inf
                    
                    fold_metrics[column] = {
                        'mean_difference': mean_diff,
                        'std_ratio': std_ratio,
                        'stability_score': 1.0 / (1.0 + mean_diff + abs(1.0 - std_ratio))
                    }
            
            stability_metrics[f'fold_{fold_idx}'] = fold_metrics
        
        return self._aggregate_stability_metrics(stability_metrics)
    
    def _aggregate_stability_metrics(self, stability_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate stability metrics across all folds."""
        if not stability_metrics:
            return {}
        
        # Collect all columns
        all_columns = set()
        for fold_metrics in stability_metrics.values():
            all_columns.update(fold_metrics.keys())
        
        aggregated = {}
        for column in all_columns:
            scores = []
            mean_diffs = []
            std_ratios = []
            
            for fold_metrics in stability_metrics.values():
                if column in fold_metrics:
                    scores.append(fold_metrics[column]['stability_score'])
                    mean_diffs.append(fold_metrics[column]['mean_difference'])
                    std_ratios.append(fold_metrics[column]['std_ratio'])
            
            if scores:
                aggregated[column] = {
                    'avg_stability_score': np.mean(scores),
                    'std_stability_score': np.std(scores),
                    'avg_mean_difference': np.mean(mean_diffs),
                    'avg_std_ratio': np.mean(std_ratios),
                    'stability_rank': len(scores) - np.argsort(scores).argsort()[-1]  # Rank (1 = best)
                }
        
        return aggregated
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add optimizer stats if available
        if self.use_vectorbt_optimization and self.rolling_optimizer is not None:
            optimizer_stats = self.rolling_optimizer.get_performance_stats()
            stats.update({
                'optimizer_' + k: v for k, v in optimizer_stats.items()
            })
        
        # Add manager stats if available
        if self.use_vectorbt_optimization and self.vectorization_manager is not None:
            manager_stats = self.vectorization_manager.get_performance_stats()
            stats.update({
                'manager_' + k: v for k, v in manager_stats.items()
            })
        
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            'total_splits': 0,
            'vectorbt_operations': 0,
            'optimization_operations': 0,
            'total_time': 0.0
        }
        
        if self.use_vectorbt_optimization and self.rolling_optimizer is not None:
            self.rolling_optimizer.reset_stats()
        
        if self.use_vectorbt_optimization and self.vectorization_manager is not None:
            self.vectorization_manager.reset_stats()


class PurgedCVSplitter(BaseCVSplitter):
    """
    Extended CV splitter with purging support.
    
    Purging removes samples from training set that are too close
    in time to the validation set, further preventing data leakage.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        embargo_pct: float = 0.1,
        purge_pct: float = 0.05,
        min_train_size: Optional[int] = None,
        use_vectorbt_optimization: bool = True,
        enable_gpu: bool = False
    ):
        """
        Initialize purged CV splitter with VectorBT optimization support.
        
        Args:
            n_folds: Number of folds
            embargo_pct: Percentage of validation to skip (after validation)
            purge_pct: Percentage of training to remove (before validation)
            min_train_size: Minimum training size
            use_vectorbt_optimization: Whether to use VectorBT optimization
            enable_gpu: Whether to enable GPU acceleration
        """
        super().__init__(n_folds, embargo_pct, min_train_size, use_vectorbt_optimization, enable_gpu)
        self.purge_pct = purge_pct
        
        if not 0 <= purge_pct <= 0.3:
            error_msg = f"purge_pct must be between 0 and 0.3, got {purge_pct}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [PurgedCVSplitter] {error_msg}", color="red")
            raise ValueError(error_msg)
    
    def split_with_embargo(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Perform time series split with both purging and embargo.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            List of (purged_train_index, embargoed_val_index) tuples
        """
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [PurgedCVSplitter] Starting purged split_with_embargo on {len(X)} samples with purge_pct={self.purge_pct:.1%}", color="cyan")
        
        # Get base splits with embargo
        base_splits = super().split_with_embargo(X, y)
        
        if not base_splits:
            return []
        
        n_samples = len(X)
        purge_size = int(n_samples * self.purge_pct)
        
        purged_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(base_splits):
            # Remove last N samples from training (purge before validation)
            if purge_size > 0 and len(train_idx) > purge_size:
                original_train_size = len(train_idx)
                train_idx = train_idx[:-purge_size]
                
                logger.debug(
                    f"Fold {fold_idx + 1}: Applied purge of {purge_size} samples "
                    f"(reduced training from {original_train_size} to {len(train_idx)})"
                )
            
            # Only include if training set is still sufficient
            if len(train_idx) > 0:
                purged_splits.append((train_idx, val_idx))
            else:
                logger.warning(
                    f"Fold {fold_idx + 1}: Training set empty after purge, skipping"
                )
        
        if TPRINT_AVAILABLE:
            tprint(f"✅ [PurgedCVSplitter] Generated {len(purged_splits)} purged CV splits (purge={self.purge_pct:.1%}, embargo={self.embargo_pct:.1%})", color="green")
        
        logger.info(
            f"Generated {len(purged_splits)} purged CV splits "
            f"(purge={self.purge_pct:.1%}, embargo={self.embargo_pct:.1%})"
        )
        
        return purged_splits
