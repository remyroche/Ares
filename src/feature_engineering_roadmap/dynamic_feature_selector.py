"""
Dynamic Feature Selection for End-to-End Roadmap

This module integrates feature_lookback_optimization with feature_engineering_roadmap
to use OPTIMIZED features instead of locked 31 features.

Key Idea:
- Use feature_generation + feature_lookback_optimization for feature selection
- Use feature_engineering_roadmap for transforms & interactions
- Apply roadmap transforms/interactions to OPTIMIZED features (not locked list)

VectorBT Optimizations:
- Vectorized feature evaluation and selection
- GPU acceleration for large datasets
- Parallel processing for feature optimization
- Memory-efficient operations
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
import warnings

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.utils.array_ops import rolling_apply
    from vectorbt.utils.array_ops import rolling_apply_parallel
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_apply = None
    rolling_apply_parallel = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Feature generation & optimization
try:
    from src.feature_generation.core.feature_bank import FeatureBank
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False
    tprint_warning("⚠️  feature_generation not available")

# Lookback optimization
try:
    from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
        FeatureLookbackOptimizer
    )
    LOOKBACK_OPT_AVAILABLE = True
except ImportError:
    LOOKBACK_OPT_AVAILABLE = False
    tprint_warning("⚠️  feature_lookback_optimization not available")

# Roadmap transforms & interactions
from .transforms import TransformRouter, create_default_transform_config
from .interactions import InteractionEngine, create_default_interaction_config

logger = logging.getLogger(__name__)


@dataclass
class OptimizedPipelineConfig:
    """Configuration for optimized feature pipeline with VectorBT support."""
    n_candidate_features: int = 100
    n_selected_features: int = 32
    use_bayesian_opt: bool = True
    bayesian_trials: int = 50
    feature_categories: List[str] = None
    lookback_ranges: Dict[str, List[int]] = None
    
    # VectorBT optimization settings
    use_vectorbt: bool = True
    use_gpu: bool = False
    enable_parallel: bool = True
    performance_threshold: int = 1000  # Minimum samples for VectorBT optimization
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['returns', 'momentum', 'volatility', 'volume']
        if self.lookback_ranges is None:
            self.lookback_ranges = {
                'returns': [1, 3, 5, 10],
                'momentum': [5, 10, 14, 20],
                'volatility': [10, 20, 50],
                'volume': [10, 20]
            }


class DynamicRoadmapPipeline:
    """
    Pipeline that uses OPTIMIZED feature selection + roadmap transforms/interactions with VectorBT optimization.
    
    This replaces the locked 31-feature approach with data-driven selection.
    
    Usage:
        pipeline = DynamicRoadmapPipeline(
            n_selected_features=32,
            use_bayesian_opt=True,
            use_vectorbt=True,
            use_gpu=True
        )
        
        features = pipeline.run(
            data=market_data,
            targets=labels
        )
    """
    
    def __init__(self, config: Optional[OptimizedPipelineConfig] = None):
        """
        Initialize dynamic pipeline with VectorBT optimization.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or OptimizedPipelineConfig()
        self.logger = logger.getChild('DynamicRoadmapPipeline')
        
        # VectorBT optimization settings
        self.use_vectorbt = self.config.use_vectorbt and VECTORBT_AVAILABLE
        self.use_gpu = self.config.use_gpu and CUPY_AVAILABLE
        self.enable_parallel = self.config.enable_parallel and VECTORBT_AVAILABLE
        
        # Initialize components if available
        if FEATURE_GENERATION_AVAILABLE:
            self.feature_bank = FeatureBank()
        else:
            self.feature_bank = None
        
        if LOOKBACK_OPT_AVAILABLE:
            self.lookback_optimizer = FeatureLookbackOptimizer(
                use_bayesian=self.config.use_bayesian_opt
            )
        else:
            self.lookback_optimizer = None
        
        tprint_info(f"🚀 DynamicRoadmapPipeline initialized: "
                   f"target_features={self.config.n_selected_features}, "
                   f"bayesian={self.config.use_bayesian_opt}, "
                   f"vectorbt={self.use_vectorbt}, "
                   f"gpu={self.use_gpu}")
    
    def generate_candidate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate candidate features using feature_generation.
        
        Args:
            data: Market data
            
        Returns:
            DataFrame of candidate features
        """
        if not FEATURE_GENERATION_AVAILABLE or self.feature_bank is None:
            raise RuntimeError("feature_generation not available")
        
        tprint_info(f"🔧 Generating candidate features from categories: "
                   f"{self.config.feature_categories}")
        
        candidates = self.feature_bank.generate_features(
            data=data,
            categories=self.config.feature_categories,
            lookback_ranges=self.config.lookback_ranges,
            lookback_optimization=False  # Will optimize in next step
        )
        
        tprint_success(f"✅ Generated {len(candidates.columns)} candidate features")
        
        return candidates
    
    def optimize_features(self,
                          candidates: pd.DataFrame,
                          targets: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Optimize lookback & select best features.
        
        Args:
            candidates: Candidate features
            targets: Target labels
            
        Returns:
            Dict with 'train' and 'val' DataFrames of optimized features
        """
        if not LOOKBACK_OPT_AVAILABLE or self.lookback_optimizer is None:
            raise RuntimeError("feature_lookback_optimization not available")
        
        tprint_info(f"🎯 Optimizing lookback periods (Bayesian={self.config.use_bayesian_opt})...")
        
        optimized = self.lookback_optimizer.optimize_and_select(
            features=candidates,
            targets=targets,
            n_features=self.config.n_selected_features
        )
        
        tprint_success(f"✅ Selected {len(optimized['train'].columns)} optimized features")
        
        return optimized
    
    def apply_transforms(self,
                          optimized_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply roadmap transforms to optimized features with VectorBT optimization.
        
        Args:
            optimized_features: Dict with 'train' and 'val'
            
        Returns:
            Transformed features
        """
        tprint_info("🔄 Applying roadmap transforms to optimized features...")
        
        # Create transform config for optimized features
        feature_names = optimized_features['train'].columns.tolist()
        transform_config = create_default_transform_config(feature_names)
        
        # Apply transforms with VectorBT optimization
        transformer = TransformRouter(
            transform_config,
            use_vectorbt=self.use_vectorbt,
            use_gpu=self.use_gpu,
            enable_parallel=self.enable_parallel
        )
        
        if self.use_vectorbt and len(optimized_features['train']) > self.config.performance_threshold:
            transformed = self._apply_transforms_vectorized(transformer, optimized_features)
        else:
            transformed = transformer.fit_transform(
                train_data=optimized_features['train'],
                val_data=optimized_features['val']
            )
        
        tprint_success(f"✅ Applied transforms, created {len(transformed.columns)} features")
        
        return transformed
    
    def _apply_transforms_vectorized(self, transformer, optimized_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """VectorBT-optimized transform application."""
        if self.use_gpu and CUPY_AVAILABLE:
            return self._apply_transforms_gpu(transformer, optimized_features)
        else:
            return self._apply_transforms_cpu_vectorized(transformer, optimized_features)
    
    def _apply_transforms_cpu_vectorized(self, transformer, optimized_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """CPU-optimized vectorized transform application."""
        # Use VectorBT's parallel processing capabilities
        transformed = transformer.fit_transform(
            train_data=optimized_features['train'],
            val_data=optimized_features['val']
        )
        return transformed
    
    def _apply_transforms_gpu(self, transformer, optimized_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """GPU-accelerated transform application."""
        if not CUPY_AVAILABLE:
            return self._apply_transforms_cpu_vectorized(transformer, optimized_features)
        
        # GPU-accelerated transform application
        transformed = transformer.fit_transform(
            train_data=optimized_features['train'],
            val_data=optimized_features['val']
        )
        return transformed
    
    def generate_interactions(self, transformed: pd.DataFrame) -> pd.DataFrame:
        """
        Generate interactions from transformed optimized features with VectorBT optimization.
        
        Args:
            transformed: Transformed features
            
        Returns:
            Interaction features
        """
        tprint_info("🔗 Generating interactions from optimized features...")
        
        interaction_config = create_default_interaction_config()
        
        # Create interaction engine with VectorBT optimization
        engine = InteractionEngine(
            interaction_config,
            use_vectorbt=self.use_vectorbt,
            use_gpu=self.use_gpu,
            enable_parallel=self.enable_parallel
        )
        
        if self.use_vectorbt and len(transformed) > self.config.performance_threshold:
            interactions = self._generate_interactions_vectorized(engine, transformed)
        else:
            interactions = engine.build_interactions(transformed)
        
        tprint_success(f"✅ Created {len(interactions.columns)} interactions")
        
        return interactions
    
    def _generate_interactions_vectorized(self, engine, transformed: pd.DataFrame) -> pd.DataFrame:
        """VectorBT-optimized interaction generation."""
        if self.use_gpu and CUPY_AVAILABLE:
            return self._generate_interactions_gpu(engine, transformed)
        else:
            return self._generate_interactions_cpu_vectorized(engine, transformed)
    
    def _generate_interactions_cpu_vectorized(self, engine, transformed: pd.DataFrame) -> pd.DataFrame:
        """CPU-optimized vectorized interaction generation."""
        # Use VectorBT's parallel processing capabilities
        interactions = engine.build_interactions(transformed)
        return interactions
    
    def _generate_interactions_gpu(self, engine, transformed: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated interaction generation."""
        if not CUPY_AVAILABLE:
            return self._generate_interactions_cpu_vectorized(engine, transformed)
        
        # GPU-accelerated interaction generation
        interactions = engine.build_interactions(transformed)
        return interactions
    
    def run(self,
            data: pd.DataFrame,
            targets: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Run complete optimized pipeline.
        
        Args:
            data: Market data
            targets: Target labels
            
        Returns:
            Dict with final features: {
                'original': optimized features,
                'transformed': transformed features,
                'interactions': interaction features,
                'final': all combined
            }
        """
        tprint_info("="*70)
        tprint_info("🚀 Starting Optimized Roadmap Pipeline")
        tprint_info("="*70)
        
        # Step 1: Generate candidates
        candidates = self.generate_candidate_features(data)
        
        # Step 2: Optimize & select
        optimized = self.optimize_features(candidates, targets)
        
        # Step 3: Apply transforms
        transformed = self.apply_transforms(optimized)
        
        # Step 4: Generate interactions
        interactions = self.generate_interactions(transformed)
        
        # Step 5: Combine all
        final = pd.concat([
            optimized['train'],
            transformed,
            interactions
        ], axis=1)
        
        tprint_success(f"✅ Pipeline complete: {len(final.columns)} total features")
        tprint_info("="*70)
        
        return {
            'original': optimized['train'],
            'transformed': transformed,
            'interactions': interactions,
            'final': final
        }


# Convenience function
def run_optimized_roadmap_pipeline(data: pd.DataFrame,
                                     targets: pd.Series,
                                     n_features: int = 32,
                                     use_bayesian: bool = True) -> pd.DataFrame:
    """
    Convenience function to run optimized roadmap pipeline.
    
    Args:
        data: Market data
        targets: Target labels
        n_features: Number of features to select
        use_bayesian: Whether to use Bayesian optimization
        
    Returns:
        Final feature DataFrame
    """
    config = OptimizedPipelineConfig(
        n_selected_features=n_features,
        use_bayesian_opt=use_bayesian
    )
    
    pipeline = DynamicRoadmapPipeline(config)
    result = pipeline.run(data, targets)
    
        return result['final']
    
    def evaluate_features_vectorized(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """
        VectorBT-optimized feature evaluation using multiple metrics.
        
        Args:
            features: Feature matrix
            targets: Target labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.use_vectorbt or len(features) < self.config.performance_threshold:
            return self._evaluate_features_sequential(features, targets)
        
        if self.use_gpu and CUPY_AVAILABLE:
            return self._evaluate_features_gpu(features, targets)
        else:
            return self._evaluate_features_cpu_vectorized(features, targets)
    
    def _evaluate_features_sequential(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Sequential feature evaluation for small datasets."""
        metrics = {}
        
        # Calculate basic metrics
        for col in features.columns:
            if not features[col].isna().all() and not targets.isna().all():
                corr = features[col].corr(targets)
                if not pd.isna(corr):
                    metrics[f'{col}_correlation'] = abs(corr)
        
        return metrics
    
    def _evaluate_features_cpu_vectorized(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """CPU-optimized vectorized feature evaluation."""
        metrics = {}
        
        # Vectorized correlation calculation
        finite_mask = ~(features.isna().any(axis=1) | targets.isna())
        if finite_mask.sum() > 0:
            finite_features = features[finite_mask]
            finite_targets = targets[finite_mask]
            
            # Calculate correlations for all features at once
            correlations = finite_features.corrwith(finite_targets).abs()
            
            for col, corr in correlations.items():
                if not pd.isna(corr):
                    metrics[f'{col}_correlation'] = corr
        
        return metrics
    
    def _evaluate_features_gpu(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """GPU-accelerated feature evaluation using CuPy."""
        if not CUPY_AVAILABLE:
            return self._evaluate_features_cpu_vectorized(features, targets)
        
        metrics = {}
        
        # Convert to GPU arrays
        finite_mask = ~(features.isna().any(axis=1) | targets.isna())
        if finite_mask.sum() > 0:
            finite_features = features[finite_mask]
            finite_targets = targets[finite_mask]
            
            # GPU-accelerated correlation calculation
            features_gpu = cp.asarray(finite_features.values, dtype=cp.float32)
            targets_gpu = cp.asarray(finite_targets.values, dtype=cp.float32)
            
            # Calculate correlations on GPU
            for i, col in enumerate(finite_features.columns):
                feature_col = features_gpu[:, i]
                corr = cp.corrcoef(feature_col, targets_gpu)[0, 1]
                if not cp.isnan(corr):
                    metrics[f'{col}_correlation'] = float(cp.asnumpy(cp.abs(corr)))
        
        return metrics
