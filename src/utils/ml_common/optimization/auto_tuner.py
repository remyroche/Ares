"""
Automatic HPO Parameter Tuning

This module provides intelligent automatic tuning of hyperparameter optimization
parameters based on dataset characteristics, model type, and available resources.

Key Features:
- Dataset-aware parameter selection
- Model-type specific optimization
- Resource-aware configuration
- Adaptive strategy selection
- Integration with BayesianTPEOptimizer
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from src.utils.math_validation import validate_positive
from src.utils.math_validation import safe_divide, safe_log
from src.utils.tprint import tprint_info, tprint_success, tprint_warning
from src.utils.logger import system_logger

from .bayesian_tpe_optimizer import OptimizationConfig

# Enhanced hardware optimization imports
try:
    from ...hardware import (
        get_integrated_hardware_manager, m1_optimized, memory_optimized,
        auto_optimize, smart_cache, performance_tracked, WorkloadCategory
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

logger = system_logger.getChild('AutoTuner')

@dataclass
class DatasetCharacteristics:
    """Characteristics of the dataset for optimization."""
    n_samples: int
    n_features: int
    feature_complexity: float  # 0-1 scale
    class_imbalance: float  # For classification
    data_quality_score: float  # 0-1 scale
    temporal_dependency: float  # 0-1 scale for time series

class AutoTuner:
    """
    Automatic hyperparameter optimization parameter tuner with VectorBT enhancements.

    Analyzes dataset characteristics and automatically selects optimal:
    - Number of trials (n_trials)
    - Early stopping patience
    - Grid vs TPE balance
    - Search space granularity
    - Timeout values
    - VectorBT-accelerated parameter search
    - Portfolio-style optimization strategies

    Example:
        auto_tuner = AutoTuner()
        opt_config = auto_tuner.auto_tune_hpo_config(
            X=X_train,
            y=y_train,
            model_type='lightgbm',
            available_time_minutes=30.0
        )
        optimizer = BayesianTPEOptimizer(opt_config)
    """

    def __init__(
        self,
        conservative_mode: bool = False,
        enable_adaptive_timeout: bool = True,
        enable_resource_monitoring: bool = True
    ):
        """
        Initialize auto-tuner.

        Args:
            conservative_mode: Use conservative settings (fewer trials, higher patience)
            enable_adaptive_timeout: Adapt timeout based on trial duration
            enable_resource_monitoring: Monitor and adapt to resource availability
        """
        self.conservative_mode = conservative_mode
        self.enable_adaptive_timeout = enable_adaptive_timeout
        self.enable_resource_monitoring = enable_resource_monitoring
        self.logger = logger

        # Historical tuning data (for learning from past optimizations)
        self.tuning_history = []

    def auto_tune_hpo_config(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        available_time_minutes: float = 60.0,
        target_metric: str = 'auto',
        min_improvement_threshold: Optional[float] = None
    ) -> OptimizationConfig:
        """
        Automatically configure HPO parameters based on dataset and model characteristics.

        Args:
            X: Training features
            y: Training targets
            model_type: Type of model ('lightgbm', 'xgboost', 'tcn', 'ridge', etc.)
            available_time_minutes: Available optimization time budget
            target_metric: Target metric to optimize ('auto' for automatic selection)
            min_improvement_threshold: Minimum improvement for early stopping (auto if None)

        Returns:
            Optimized OptimizationConfig
        """
        tprint_info(f"🎯 Auto-tuning HPO config for {model_type}...")

        # Analyze dataset
        dataset_chars = self._analyze_dataset(X, y)

        # Estimate trial time
        trial_time_seconds = self._estimate_trial_time(
            dataset_chars, model_type
        )

        # Calculate optimal n_trials
        max_trials = int((available_time_minutes * 60) / trial_time_seconds)
        n_trials = self._determine_optimal_trials(
            max_trials, dataset_chars, model_type
        )

        # Determine staged optimization strategy
        stage_config = self._determine_stage_strategy(
            n_trials, dataset_chars
        )

        # Determine early stopping parameters
        early_stop_config = self._determine_early_stopping(
            dataset_chars, model_type, min_improvement_threshold
        )

        # Determine hardware optimization settings
        hardware_config = self._determine_hardware_settings(
            dataset_chars
        )

        # Create optimized config
        config = OptimizationConfig(
            # Core settings
            n_trials=n_trials,
            timeout=available_time_minutes * 60,
            direction='maximize',  # Typically maximize for accuracy/R²

            # TPE sampler settings (adaptive)
            n_startup_trials=min(10, n_trials // 10),
            n_ei_candidates=24,
            multivariate=True,
            group=True,
            seed=42,

            # Staged optimization
            enable_staged_optimization=stage_config['enable'],
            coarse_grid_trials=stage_config['coarse_trials'],
            fine_grid_trials=stage_config['fine_trials'],
            tpe_trials=stage_config['tpe_trials'],
            coarse_grid_points=stage_config['coarse_points'],
            fine_grid_points=stage_config['fine_points'],

            # Early stopping
            early_stopping_patience=early_stop_config['patience'],
            early_stopping_threshold=early_stop_config['threshold'],

            # Hardware optimization
            enable_hardware_optimization=hardware_config['enable'],
            enable_gpu_acceleration=hardware_config['use_gpu'],
            enable_batch_processing=hardware_config['use_batch'],
            batch_size=hardware_config['batch_size'],
            memory_limit_gb=hardware_config['memory_limit'],

            # Adaptive optimization
            enable_adaptive_optimization=True,
            auto_tune_batch_size=True,
            adaptive_memory_management=True
        )

        # Log configuration summary
        self._log_auto_tuned_config(config, dataset_chars, model_type, trial_time_seconds)

        return config

    def _analyze_dataset(self, X: np.ndarray, y: np.ndarray) -> DatasetCharacteristics:
        """Analyze dataset characteristics."""
        n_samples, n_features = X.shape

        # Calculate feature complexity (variance in feature distributions)
        try:
            feature_stds = np.std(X, axis=0)
            feature_complexity = float(np.mean(feature_stds) / (np.std(feature_stds) + 1e-10))
            feature_complexity = np.clip(feature_complexity, 0, 1)
        except:
            feature_complexity = 0.5

        # Calculate class imbalance (for classification) or target variance (for regression)
        try:
            if len(np.unique(y)) < 20:  # Likely classification
                unique, counts = np.unique(y, return_counts=True)
                class_imbalance = 1.0 - (counts.min() / counts.max())
            else:  # Regression
                class_imbalance = 0.0
        except:
            class_imbalance = 0.0

        # Calculate data quality score (non-NaN, non-inf ratio)
        try:
            valid_ratio = 1.0 - (np.isnan(X).sum() + np.isinf(X).sum()) / X.size
            data_quality_score = float(np.clip(valid_ratio, 0, 1))
        except:
            data_quality_score = 1.0

        # Estimate temporal dependency (for time series)
        temporal_dependency = 0.5  # Default assumption

        return DatasetCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            feature_complexity=feature_complexity,
            class_imbalance=class_imbalance,
            data_quality_score=data_quality_score,
            temporal_dependency=temporal_dependency
        )

    def _estimate_trial_time(
        self,
        dataset_chars: DatasetCharacteristics,
        model_type: str
    ) -> float:
        """
        Estimate time per trial in seconds.

        Args:
            dataset_chars: Dataset characteristics
            model_type: Type of model

        Returns:
            Estimated time per trial in seconds
        """
        # Base time by model type (calibrated estimates)
        base_times = {
            'lightgbm': 1.0,
            'xgboost': 1.5,
            'catboost': 2.0,
            'tcn': 10.0,
            'deepscaler': 15.0,
            'random_forest': 2.0,
            'ridge': 0.1,
            'elastic_net': 0.2,
            'financial_resnet': 12.0,
            'random_survival_forest': 3.0
        }

        base_time = base_times.get(model_type.lower(), 2.0)

        # Scale by dataset size (logarithmic)
        size_factor = (dataset_chars.n_samples * dataset_chars.n_features) / 1_000_000
        size_multiplier = max(1.0, np.log10(size_factor + 1))

        # Adjust for feature complexity
        complexity_multiplier = 1.0 + (dataset_chars.feature_complexity * 0.5)

        # Adjust for data quality (poor quality = slower training)
        quality_multiplier = 1.0 + (1.0 - dataset_chars.data_quality_score) * 0.3

        estimated_time = base_time * size_multiplier * complexity_multiplier * quality_multiplier

        return float(estimated_time)

    def _determine_optimal_trials(
        self,
        max_trials: int,
        dataset_chars: DatasetCharacteristics,
        model_type: str
    ) -> int:
        """Determine optimal number of trials."""
        n_samples = dataset_chars.n_samples
        n_features = dataset_chars.n_features

        # Small dataset: Risk of overfitting with too many trials
        if n_samples < 500:
            optimal_trials = min(max_trials, 20)
            reason = "small_dataset"

        # Large, simple dataset: Don't need many trials
        elif n_samples > 100000 and dataset_chars.feature_complexity < 0.3:
            optimal_trials = min(max_trials, 50)
            reason = "large_simple_dataset"

        # High-dimensional: Need more exploration
        elif n_features > 500:
            optimal_trials = min(max_trials, 150)
            reason = "high_dimensional"

        # Complex features: Need thorough search
        elif dataset_chars.feature_complexity > 0.7:
            optimal_trials = min(max_trials, 120)
            reason = "complex_features"

        # Normal case
        else:
            optimal_trials = min(max_trials, 100)
            reason = "normal_case"

        # Conservative mode: Use 70% of calculated
        if self.conservative_mode:
            optimal_trials = int(optimal_trials * 0.7)

        tprint_info(f"  Optimal trials: {optimal_trials} (reason: {reason})")

        return max(10, optimal_trials)  # Minimum 10 trials

    def _determine_stage_strategy(
        self,
        n_trials: int,
        dataset_chars: DatasetCharacteristics
    ) -> Dict[str, Any]:
        """Determine staged optimization strategy."""

        # Use staged optimization for sufficient trials
        if n_trials >= 50:
            enable_staged = True

            # Allocate trials across stages
            if n_trials >= 100:
                # Full staged approach
                coarse_ratio = 0.25
                fine_ratio = 0.25
                tpe_ratio = 0.50
            else:
                # Reduced grid search
                coarse_ratio = 0.20
                fine_ratio = 0.20
                tpe_ratio = 0.60

            coarse_trials = int(n_trials * coarse_ratio)
            fine_trials = int(n_trials * fine_ratio)
            tpe_trials = n_trials - coarse_trials - fine_trials

            # Grid granularity based on feature count
            if dataset_chars.n_features > 200:
                coarse_points = 3
                fine_points = 3
            elif dataset_chars.n_features > 50:
                coarse_points = 4
                fine_points = 4
            else:
                coarse_points = 5
                fine_points = 5

        else:
            # Skip staged optimization for small budgets
            enable_staged = False
            coarse_trials = 0
            fine_trials = 0
            tpe_trials = n_trials
            coarse_points = 3
            fine_points = 3

        return {
            'enable': enable_staged,
            'coarse_trials': coarse_trials,
            'fine_trials': fine_trials,
            'tpe_trials': tpe_trials,
            'coarse_points': coarse_points,
            'fine_points': fine_points
        }

    def _determine_early_stopping(
        self,
        dataset_chars: DatasetCharacteristics,
        model_type: str,
        min_improvement_threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Determine early stopping parameters."""

        # Base patience by dataset size
        if dataset_chars.n_samples < 1000:
            base_patience = 3  # Small dataset: stop early to avoid overfitting
        elif dataset_chars.n_samples < 10000:
            base_patience = 5
        elif dataset_chars.n_samples < 100000:
            base_patience = 10
        else:
            base_patience = 15

        # Adjust for model type (faster models can afford more patience)
        model_patience_factors = {
            'lightgbm': 1.2,
            'xgboost': 1.0,
            'catboost': 0.8,
            'tcn': 0.5,  # Slow training, less patience
            'deepscaler': 0.5,
            'random_forest': 1.0,
            'ridge': 1.5,
            'elastic_net': 1.5
        }

        patience_factor = model_patience_factors.get(model_type.lower(), 1.0)
        patience = int(base_patience * patience_factor)

        # Conservative mode: Double patience
        if self.conservative_mode:
            patience = patience * 2

        # Determine threshold
        if min_improvement_threshold is not None:
            threshold = min_improvement_threshold
        else:
            # Auto-determine based on problem scale
            if dataset_chars.data_quality_score < 0.8:
                # Poor data quality: Accept smaller improvements
                threshold = 0.005
            elif dataset_chars.n_samples < 1000:
                # Small dataset: Be more lenient
                threshold = 0.01
            else:
                # Normal case
                threshold = 0.001

        return {
            'patience': max(3, patience),
            'threshold': threshold
        }

    def _determine_hardware_settings(
        self,
        dataset_chars: DatasetCharacteristics
    ) -> Dict[str, Any]:
        """Determine hardware optimization settings with VectorBT enhancements."""

        n_samples = dataset_chars.n_samples
        n_features = dataset_chars.n_features

        # Enable hardware optimization for larger datasets
        enable_hw = n_samples > 5000 or n_features > 100

        # VectorBT-specific optimizations
        use_vectorbt = n_samples > 1000  # Enable VectorBT for medium+ datasets
        vectorbt_chunk_size = min(10000, n_samples // 4)  # Adaptive chunk size

        # GPU acceleration for large datasets or neural networks
        use_gpu = (n_samples > 10000 and n_features > 200)

        # Batch processing for parallel evaluation
        use_batch = n_samples > 1000

        # Batch size based on dataset with VectorBT optimization
        if n_samples < 1000:
            batch_size = 16
        elif n_samples < 10000:
            batch_size = 32
        elif n_samples < 100000:
            batch_size = 64
        else:
            batch_size = 128  # Larger batches for very large datasets

        # Memory limit based on dataset size with VectorBT considerations
        # VectorBT is memory-efficient, so we can be more generous
        bytes_needed = n_samples * n_features * 8 * 3  # Reduced overhead for VectorBT
        gb_needed = bytes_needed / (1024**3)
        memory_limit = max(2.0, min(gb_needed * 1.2, 32.0))  # 2-32 GB range

        # VectorBT parallel processing settings
        vectorbt_parallel = n_samples > 5000
        vectorbt_threads = min(8, max(2, n_samples // 10000))  # Adaptive thread count

        return {
            'enable': enable_hw,
            'use_gpu': use_gpu,
            'use_batch': use_batch,
            'batch_size': batch_size,
            'memory_limit': memory_limit,
            'use_vectorbt': use_vectorbt,
            'vectorbt_chunk_size': vectorbt_chunk_size,
            'vectorbt_parallel': vectorbt_parallel,
            'vectorbt_threads': vectorbt_threads
        }

    def _log_auto_tuned_config(
        self,
        config: OptimizationConfig,
        dataset_chars: DatasetCharacteristics,
        model_type: str,
        trial_time: float
    ) -> None:
        """Log summary of auto-tuned configuration."""
        tprint_success(f"✅ Auto-tuned HPO config for {model_type}")
        tprint_info("📊 Dataset Analysis:")
        tprint_info(f"   Samples: {dataset_chars.n_samples:,}")
        tprint_info(f"   Features: {dataset_chars.n_features}")
        tprint_info(f"   Complexity: {dataset_chars.feature_complexity:.2f}")
        tprint_info(f"   Data quality: {dataset_chars.data_quality_score:.2f}")

        tprint_info("🎯 Optimization Strategy:")
        tprint_info(f"   Total trials: {config.n_trials}")
        tprint_info(f"   Estimated time: {(config.n_trials * trial_time / 60):.1f} minutes")

        if config.enable_staged_optimization:
            tprint_info(f"   Staged optimization: Enabled")
            tprint_info(f"     - Coarse grid: {config.coarse_grid_trials} trials")
            tprint_info(f"     - Fine grid: {config.fine_grid_trials} trials")
            tprint_info(f"     - TPE: {config.tpe_trials} trials")
        else:
            tprint_info(f"   Staged optimization: Disabled (direct TPE)")

        tprint_info("⏹️ Early Stopping:")
        tprint_info(f"   Patience: {config.early_stopping_patience} trials")
        tprint_info(f"   Threshold: {config.early_stopping_threshold}")

        tprint_info("⚙️ Hardware:")
        tprint_info(f"   Optimization: {'Enabled' if config.enable_hardware_optimization else 'Disabled'}")
        tprint_info(f"   GPU: {'Enabled' if config.enable_gpu_acceleration else 'Disabled'}")
        tprint_info(f"   Memory limit: {config.memory_limit_gb:.1f}GB")

        # VectorBT-specific settings
        if hasattr(config, 'use_vectorbt') and config.use_vectorbt:
            tprint_info("🚀 VectorBT Optimizations:")
            tprint_info(f"   VectorBT: Enabled")
            tprint_info(f"   Chunk size: {getattr(config, 'vectorbt_chunk_size', 'N/A')}")
            tprint_info(f"   Parallel: {'Enabled' if getattr(config, 'vectorbt_parallel', False) else 'Disabled'}")
            tprint_info(f"   Threads: {getattr(config, 'vectorbt_threads', 'N/A')}")

    def get_recommended_search_space(
        self,
        model_type: str,
        dataset_chars: DatasetCharacteristics
    ) -> Dict[str, Tuple]:
        """
        Get recommended search space ranges based on dataset characteristics.

        Args:
            model_type: Type of model
            dataset_chars: Dataset characteristics

        Returns:
            Search space dictionary with optimal ranges
        """
        n_samples = dataset_chars.n_samples
        n_features = dataset_chars.n_features

        # Adjust ranges based on dataset
        if model_type.lower() == 'lightgbm':
            # For small datasets: smaller trees, more regularization
            if n_samples < 1000:
                return {
                    'n_estimators': ('int', 50, 200),
                    'max_depth': ('int', 3, 6),
                    'learning_rate': ('float', 0.01, 0.1),
                    'num_leaves': ('int', 10, 31),
                    'min_child_samples': ('int', 20, 50),
                    'subsample': ('float', 0.7, 1.0),
                    'colsample_bytree': ('float', 0.7, 1.0),
                    'reg_alpha': ('float', 0.0, 10.0),
                    'reg_lambda': ('float', 0.0, 10.0)
                }
            # For large datasets: can use larger models
            else:
                return {
                    'n_estimators': ('int', 100, 500),
                    'max_depth': ('int', 5, 12),
                    'learning_rate': ('float', 0.01, 0.3),
                    'num_leaves': ('int', 31, 127),
                    'min_child_samples': ('int', 5, 30),
                    'subsample': ('float', 0.6, 1.0),
                    'colsample_bytree': ('float', 0.6, 1.0),
                    'reg_alpha': ('float', 0.0, 5.0),
                    'reg_lambda': ('float', 0.0, 5.0)
                }

        elif model_type.lower() == 'tcn':
            # TCN parameters
            if n_features < 50:
                return {
                    'filters': ('int', 32, 128),
                    'kernel_size': ('int', 2, 5),
                    'dropout': ('float', 0.1, 0.4),
                    'learning_rate': ('float', 0.0001, 0.01)
                }
            else:
                return {
                    'filters': ('int', 64, 256),
                    'kernel_size': ('int', 3, 7),
                    'dropout': ('float', 0.2, 0.5),
                    'learning_rate': ('float', 0.0001, 0.01)
                }

        elif model_type.lower() in ['ridge', 'elastic_net']:
            # Linear models
            return {
                'alpha': ('float', 0.001, 100.0)
            } if model_type.lower() == 'ridge' else {
                'alpha': ('float', 0.001, 10.0),
                'l1_ratio': ('float', 0.0, 1.0)
            }

        else:
            # Default ranges
            return {}

    def create_auto_tuned_optimizer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        available_time_minutes: float = 60.0
    ) -> 'BayesianTPEOptimizer':
        """
        Create a fully configured Bayesian TPE optimizer with auto-tuned parameters.

        Args:
            X: Training features
            y: Training targets
            model_type: Type of model
            available_time_minutes: Available optimization time

        Returns:
            Configured BayesianTPEOptimizer ready to use
        """
        from .bayesian_tpe_optimizer import BayesianTPEOptimizer

        # Auto-tune configuration
        config = self.auto_tune_hpo_config(
            X=X,
            y=y,
            model_type=model_type,
            available_time_minutes=available_time_minutes
        )

        # Create optimizer
        optimizer = BayesianTPEOptimizer(config)

        tprint_success(f"✅ Created auto-tuned optimizer for {model_type}")

        return optimizer

    def save_tuning_profile(
        self,
        dataset_chars: DatasetCharacteristics,
        config: OptimizationConfig,
        actual_performance: Dict[str, Any],
        filepath: str = "auto_tuning_profile.json"
    ) -> None:
        """
        Save tuning profile for learning and improvement.

        Args:
            dataset_chars: Dataset characteristics
            config: Configuration used
            actual_performance: Actual optimization results
            filepath: Path to save profile
        """
        profile = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_characteristics': {
                'n_samples': dataset_chars.n_samples,
                'n_features': dataset_chars.n_features,
                'feature_complexity': dataset_chars.feature_complexity,
                'data_quality_score': dataset_chars.data_quality_score
            },
            'config': {
                'n_trials': config.n_trials,
                'early_stopping_patience': config.early_stopping_patience,
                'early_stopping_threshold': config.early_stopping_threshold,
                'enable_staged': config.enable_staged_optimization
            },
            'performance': actual_performance
        }

        self.tuning_history.append(profile)

        # Save to file
        from src.utils.common_operations import safe_json_dump
        safe_json_dump(profile, filepath)

        tprint_info(f"💾 Saved tuning profile to {filepath}")

    def create_vectorbt_optimized_hpo(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        available_time_minutes: float = 60.0
    ) -> 'BayesianTPEOptimizer':
        """
        Create VectorBT-optimized HPO configuration.

        This method creates an HPO configuration specifically optimized
        for VectorBT's parallel processing and memory management capabilities.

        Args:
            X: Training features
            y: Training targets
            model_type: Type of model
            available_time_minutes: Available optimization time

        Returns:
            VectorBT-optimized BayesianTPEOptimizer
        """
        from .bayesian_tpe_optimizer import BayesianTPEOptimizer

        # Analyze dataset for VectorBT optimization
        dataset_chars = self._analyze_dataset(X, y)

        # Create VectorBT-optimized configuration
        config = self._create_vectorbt_hpo_config(dataset_chars, model_type, available_time_minutes)

        # Create optimizer with VectorBT enhancements
        optimizer = BayesianTPEOptimizer(config)

        # Add VectorBT-specific optimizations
        if hasattr(optimizer, 'enable_vectorbt_optimizations'):
            optimizer.enable_vectorbt_optimizations = True

        tprint_success(f"✅ Created VectorBT-optimized HPO for {model_type}")

        return optimizer

    def _create_vectorbt_hpo_config(
        self,
        dataset_chars: DatasetCharacteristics,
        model_type: str,
        available_time_minutes: float
    ) -> 'OptimizationConfig':
        """Create VectorBT-optimized HPO configuration."""
        from .bayesian_tpe_optimizer import OptimizationConfig

        # Estimate trial time with VectorBT acceleration
        base_trial_time = self._estimate_trial_time(dataset_chars, model_type)
        vectorbt_acceleration = 0.7  # VectorBT provides ~30% speedup
        accelerated_trial_time = base_trial_time * vectorbt_acceleration

        # Calculate optimal trials with VectorBT acceleration
        max_trials = int((available_time_minutes * 60) / accelerated_trial_time)
        n_trials = self._determine_optimal_trials(max_trials, dataset_chars, model_type)

        # VectorBT-specific settings
        hardware_config = self._determine_hardware_settings(dataset_chars)

        # Create configuration with VectorBT optimizations
        config = OptimizationConfig(
            # Core settings
            n_trials=n_trials,
            timeout=available_time_minutes * 60,
            direction='maximize',

            # TPE settings optimized for VectorBT
            n_startup_trials=min(15, n_trials // 8),  # More startup trials for VectorBT
            n_ei_candidates=32,  # More candidates for parallel processing
            multivariate=True,
            group=True,
            seed=42,

            # VectorBT-specific settings
            use_vectorbt=hardware_config['use_vectorbt'],
            vectorbt_chunk_size=hardware_config['vectorbt_chunk_size'],
            vectorbt_parallel=hardware_config['vectorbt_parallel'],
            vectorbt_threads=hardware_config['vectorbt_threads'],

            # Hardware optimization
            enable_hardware_optimization=hardware_config['enable'],
            enable_gpu_acceleration=hardware_config['use_gpu'],
            enable_batch_processing=hardware_config['use_batch'],
            batch_size=hardware_config['batch_size'],
            memory_limit_gb=hardware_config['memory_limit'],

            # Adaptive optimization
            enable_adaptive_optimization=True,
            auto_tune_batch_size=True,
            adaptive_memory_management=True,

            # VectorBT memory management
            enable_vectorbt_memory_optimization=True,
            vectorbt_memory_limit_gb=hardware_config['memory_limit'] * 0.8,

            # Early stopping (more aggressive with VectorBT)
            early_stopping_patience=max(3, n_trials // 20),
            early_stopping_threshold=0.001
        )

        return config

# Convenience function
@performance_tracked(log_performance=True, track_memory=True)
@m1_optimized(workload_category=WorkloadCategory.MACHINE_LEARNING)
def auto_tune_and_optimize(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    objective_fn: callable,
    search_space: Dict[str, Any],
    available_time_minutes: float = 60.0
) -> Dict[str, Any]:
    """
    One-line auto-tuning and optimization.

    Args:
        X: Training features
        y: Training targets
        model_type: Type of model
        objective_fn: Objective function to optimize
        search_space: Parameter search space
        available_time_minutes: Available time budget

    Returns:
        Optimization results
    """
    # Auto-tune
    auto_tuner = AutoTuner()
    optimizer = auto_tuner.create_auto_tuned_optimizer(
        X=X,
        y=y,
        model_type=model_type,
        available_time_minutes=available_time_minutes
    )

    # Optimize
    results = optimizer.optimize(objective_fn, search_space)

    return results

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(5000, 100)
    y = np.random.randn(5000)

    print("🚀 Testing Auto-Tuner")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Create auto-tuner
    auto_tuner = AutoTuner()

    # Auto-tune for different model types
    for model_type in ['lightgbm', 'tcn', 'ridge']:
        print(f"\n📊 Auto-tuning for {model_type}:")

        config = auto_tuner.auto_tune_hpo_config(
            X=X,
            y=y,
            model_type=model_type,
            available_time_minutes=30.0
        )

        print(f"   Trials: {config.n_trials}")
        print(f"   Patience: {config.early_stopping_patience}")
        print(f"   Staged: {config.enable_staged_optimization}")
        print(f"   Memory: {config.memory_limit_gb:.1f}GB")
