"""
Training Efficiency Utilities for ML Steps

This module provides utilities for:
1. Dynamic subsampling based on dataset size
2. Warm start parameter management for HPO and HMM
3. Memory-efficient training helpers

Usage:
    ```python
    from src.utils.ml_common.training_efficiency import (
        WarmStartManager,
        DynamicSubsampler,
        get_efficient_training_config,
        TrainingSpeedSuggestions
    )
    
    # Warm start for HPO
    warm_manager = WarmStartManager(model_id="my_model")
    previous_params = warm_manager.load_params()
    # ... train model ...
    warm_manager.save_params(best_params)
    
    # Dynamic subsampling
    subsampler = DynamicSubsampler(min_pct=0.1, max_pct=0.5)
    X_sample, y_sample = subsampler.sample(X, y, stratify=True)
    
    # Get training efficiency config
    config = get_efficient_training_config(
        n_samples=len(X),
        n_features=len(X.columns),
        task_type="classification"
    )
    ```

Supported Steps:
- hmm_ml_alpha_step
- hmm_macro_regime
- ml_mean_reversion_step / ml_reversion_regime_step
- ml_smc_regime_step
- ml_breakout_bounce_regime_step
- ml_liquidity_regime_step
- ml_risk_regime_step
- ml_path_regime_step
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Warm Start Manager
# ============================================================================

@dataclass
class WarmStartParams:
    """Container for warm start parameters."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: datetime
    model_hash: str  # Hash of model config for validation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'params': self.params,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'model_hash': self.model_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WarmStartParams':
        return cls(
            params=data['params'],
            metrics=data.get('metrics', {}),
            timestamp=datetime.fromisoformat(data['timestamp']),
            model_hash=data.get('model_hash', '')
        )


class WarmStartManager:
    """
    Manages warm start parameters across training runs.
    
    Features:
    - Persists best parameters from HPO
    - Validates parameter compatibility with model config
    - Supports multiple model types (XGBoost, LightGBM, NGBoost, HMM, etc.)
    - Automatic cleanup of stale parameters
    """
    
    DEFAULT_CACHE_DIR = Path("cache/warm_start")
    MAX_AGE_DAYS = 30  # Expire warm start params after 30 days
    
    def __init__(
        self,
        model_id: str,
        model_type: str = "generic",
        cache_dir: Optional[Path] = None,
        model_config_hash: Optional[str] = None
    ):
        """
        Initialize warm start manager.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (xgb, lgbm, ngboost, hmm, etc.)
            cache_dir: Directory for storing warm start params
            model_config_hash: Hash of model config for validation
        """
        self.model_id = model_id
        self.model_type = model_type
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.model_config_hash = model_config_hash or ""
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache_file = self.cache_dir / f"{model_type}_{model_id}_warm_start.json"
    
    def load_params(self, validate_hash: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load warm start parameters if available and valid.
        
        Args:
            validate_hash: If True, only load if model config hash matches
            
        Returns:
            Dictionary of parameters or None if not available/invalid
        """
        if not self._cache_file.exists():
            logger.info(f"No warm start params found for {self.model_id}")
            return None
        
        try:
            with open(self._cache_file, 'r') as f:
                data = json.load(f)
            
            warm_params = WarmStartParams.from_dict(data)
            
            # Check age
            age_days = (datetime.now() - warm_params.timestamp).days
            if age_days > self.MAX_AGE_DAYS:
                logger.info(f"Warm start params for {self.model_id} expired ({age_days} days old)")
                self._cache_file.unlink()  # Delete stale params
                return None
            
            # Validate hash if requested
            if validate_hash and self.model_config_hash:
                if warm_params.model_hash != self.model_config_hash:
                    logger.warning(
                        f"Warm start params for {self.model_id} have different config hash, "
                        "parameters may not be compatible"
                    )
                    # Still return params but with warning
            
            logger.info(f"Loaded warm start params for {self.model_id} ({age_days} days old)")
            return warm_params.params
            
        except Exception as e:
            logger.warning(f"Failed to load warm start params: {e}")
            return None
    
    def save_params(
        self,
        params: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Save parameters for warm start in future runs.
        
        Args:
            params: Best parameters to save
            metrics: Optional performance metrics
            
        Returns:
            True if saved successfully
        """
        try:
            warm_params = WarmStartParams(
                params=params,
                metrics=metrics or {},
                timestamp=datetime.now(),
                model_hash=self.model_config_hash
            )
            
            with open(self._cache_file, 'w') as f:
                json.dump(warm_params.to_dict(), f, indent=2)
            
            logger.info(f"Saved warm start params for {self.model_id}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save warm start params: {e}")
            return False
    
    def clear_params(self) -> bool:
        """Clear saved warm start parameters."""
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
                logger.info(f"Cleared warm start params for {self.model_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear warm start params: {e}")
            return False
    
    @staticmethod
    def compute_config_hash(config: Dict[str, Any]) -> str:
        """Compute hash of model configuration for validation."""
        # Only include key configuration items that affect parameter compatibility
        key_items = ['task_type', 'objective', 'model_type', 'n_features']
        filtered = {k: v for k, v in config.items() if k in key_items}
        config_str = json.dumps(filtered, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


# ============================================================================
# Dynamic Subsampler
# ============================================================================

@dataclass
class SubsamplingConfig:
    """Configuration for dynamic subsampling."""
    min_pct: float = 0.1  # Minimum 10% of data
    max_pct: float = 0.5  # Maximum 50% of data
    min_samples: int = 500  # Absolute minimum samples
    max_samples: int = 50000  # Cap at 50k samples for HPO
    
    # Thresholds for pct selection
    small_threshold: int = 5000  # Below this: use max_pct
    medium_threshold: int = 20000  # Below this: use mid_pct
    large_threshold: int = 50000  # Below this: use lower_pct
    # Above large_threshold: use min_pct


class DynamicSubsampler:
    """
    Dynamic subsampling based on dataset size.
    
    As dataset grows, sampling percentage decreases to maintain
    reasonable HPO times while ensuring minimum sample requirements.
    
    Sample size selection:
    - < 5000 samples: 50% (max_pct)
    - 5000-20000: 30%
    - 20000-50000: 20%
    - > 50000: 10% (min_pct)
    
    Always ensures:
    - At least min_samples
    - At most max_samples
    """
    
    def __init__(self, config: Optional[SubsamplingConfig] = None):
        self.config = config or SubsamplingConfig()
    
    def get_sample_size(self, n_samples: int) -> int:
        """
        Calculate optimal sample size based on dataset size.
        
        Args:
            n_samples: Total number of samples in dataset
            
        Returns:
            Optimal number of samples to use
        """
        cfg = self.config
        
        if n_samples < cfg.small_threshold:
            sample_pct = cfg.max_pct
        elif n_samples < cfg.medium_threshold:
            sample_pct = 0.3
        elif n_samples < cfg.large_threshold:
            sample_pct = 0.2
        else:
            sample_pct = cfg.min_pct
        
        sample_size = int(n_samples * sample_pct)
        
        # Apply bounds
        sample_size = max(sample_size, cfg.min_samples)
        sample_size = min(sample_size, cfg.max_samples, n_samples)
        
        return sample_size
    
    def get_sample_pct(self, n_samples: int) -> float:
        """Get the sampling percentage for a given dataset size."""
        sample_size = self.get_sample_size(n_samples)
        return sample_size / n_samples if n_samples > 0 else 1.0
    
    def sample(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        stratify: bool = False,
        random_state: int = 42
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Sample data with dynamic sample size.
        
        Args:
            X: Feature data
            y: Target data (optional)
            stratify: If True, use stratified sampling (requires y)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_sampled, y_sampled)
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        sample_size = self.get_sample_size(n_samples)
        
        if sample_size >= n_samples:
            # No sampling needed
            return X, y
        
        is_temporal = isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex)
        
        if is_temporal:
            start_idx = max(0, n_samples - sample_size)
            
            if isinstance(X, pd.DataFrame):
                X_sample = X.iloc[start_idx:]
            else:
                X_sample = X[start_idx:]
            
            if y is not None:
                if isinstance(y, pd.Series):
                    y_sample = y.iloc[start_idx:]
                else:
                    y_sample = y[start_idx:]
            else:
                y_sample = None
            
            return X_sample, y_sample
        
        np.random.seed(random_state)
        
        if stratify and y is not None:
            # Stratified sampling
            from sklearn.model_selection import train_test_split
            
            _, X_sample, _, y_sample = train_test_split(
                X, y,
                test_size=sample_size / n_samples,
                stratify=y,
                random_state=random_state
            )
            return X_sample, y_sample
        else:
            # Random sampling
            indices = np.random.choice(n_samples, sample_size, replace=False)
            
            if isinstance(X, pd.DataFrame):
                X_sample = X.iloc[indices]
            else:
                X_sample = X[indices]
            
            if y is not None:
                if isinstance(y, pd.Series):
                    y_sample = y.iloc[indices]
                else:
                    y_sample = y[indices]
            else:
                y_sample = None
            
            return X_sample, y_sample
    
    def get_subsample_info(self, n_samples: int) -> Dict[str, Any]:
        """Get information about subsampling for a dataset."""
        sample_size = self.get_sample_size(n_samples)
        sample_pct = self.get_sample_pct(n_samples)
        
        return {
            'original_samples': n_samples,
            'sampled_size': sample_size,
            'sample_pct': sample_pct,
            'will_subsample': sample_size < n_samples
        }


# ============================================================================
# Efficient Training Configuration
# ============================================================================

@dataclass
class EfficientTrainingConfig:
    """Configuration derived from dataset size for efficient training."""
    
    # HPO settings
    hpo_n_trials: int = 50
    hpo_timeout: int = 1800
    hpo_sample_size: int = 10000
    
    # Model training settings
    n_estimators: int = 500
    early_stopping_rounds: int = 35
    
    # CV settings
    cv_folds: int = 5
    
    # Memory settings
    use_sparse_matrices: bool = False
    use_binary_format: bool = True
    
    # Parallelism
    n_jobs: int = -1


def get_efficient_training_config(
    n_samples: int,
    n_features: int,
    task_type: str = "classification",
    time_budget_minutes: int = 30
) -> EfficientTrainingConfig:
    """
    Get training configuration optimized for dataset size and time budget.
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
        task_type: "classification" or "regression"
        time_budget_minutes: Available time budget
        
    Returns:
        Optimized training configuration
    """
    config = EfficientTrainingConfig()
    
    # Adjust HPO based on dataset size
    if n_samples > 100000:
        config.hpo_n_trials = 30
        config.hpo_timeout = min(time_budget_minutes * 60 // 2, 1200)
        config.hpo_sample_size = 20000
    elif n_samples > 50000:
        config.hpo_n_trials = 40
        config.hpo_timeout = min(time_budget_minutes * 60, 1800)
        config.hpo_sample_size = 15000
    elif n_samples > 10000:
        config.hpo_n_trials = 50
        config.hpo_timeout = time_budget_minutes * 60
        config.hpo_sample_size = min(n_samples, 10000)
    else:
        config.hpo_n_trials = 30
        config.hpo_timeout = time_budget_minutes * 60
        config.hpo_sample_size = n_samples
    
    # Adjust model settings based on complexity
    complexity = n_samples * n_features
    if complexity > 1e9:  # Very large
        config.n_estimators = 300
        config.early_stopping_rounds = 25
        config.cv_folds = 3
        config.use_sparse_matrices = n_features > 1000
    elif complexity > 1e8:  # Large
        config.n_estimators = 500
        config.early_stopping_rounds = 35
        config.cv_folds = 5
    else:  # Normal
        config.n_estimators = 500
        config.early_stopping_rounds = 35
        config.cv_folds = 5
    
    # Memory settings
    config.use_binary_format = n_samples > 50000
    
    return config


# ============================================================================
# Training Speed Suggestions
# ============================================================================

@dataclass
class SpeedSuggestion:
    """A suggestion for improving training speed."""
    category: str  # "data", "model", "hardware", "algorithm"
    suggestion: str
    impact: str  # "high", "medium", "low"
    implementation_effort: str  # "easy", "moderate", "complex"
    details: str


class TrainingSpeedSuggestions:
    """
    Generates suggestions for faster training based on step characteristics.
    
    This class analyzes a training step and provides actionable suggestions
    for improving training speed without significant accuracy loss.
    """
    
    @staticmethod
    def get_suggestions(
        step_name: str,
        n_samples: int,
        n_features: int,
        current_training_time_minutes: float = 0,
        model_types: Optional[List[str]] = None
    ) -> List[SpeedSuggestion]:
        """
        Get suggestions for improving training speed.
        
        Args:
            step_name: Name of the training step
            n_samples: Number of training samples
            n_features: Number of features
            current_training_time_minutes: Current training time (if known)
            model_types: List of model types used
            
        Returns:
            List of SpeedSuggestion objects
        """
        suggestions = []
        model_types = model_types or []
        
        # ====== Data-level suggestions ======
        if n_samples > 50000:
            suggestions.append(SpeedSuggestion(
                category="data",
                suggestion="Implement dynamic subsampling for HPO",
                impact="high",
                implementation_effort="easy",
                details=(
                    "Use 10-30% of data for HPO trials. "
                    "For 50k+ samples, use at most 10-15k samples during HPO. "
                    "Final model can train on full data with best params."
                )
            ))
        
        if n_features > 500:
            suggestions.append(SpeedSuggestion(
                category="data",
                suggestion="Apply aggressive feature selection before HPO",
                impact="high",
                implementation_effort="moderate",
                details=(
                    "Use fast univariate tests (mutual info, F-score) to reduce "
                    "features to top 100-200 before HPO. This can reduce HPO time "
                    "by 50-80% with minimal accuracy impact."
                )
            ))
        
        if n_samples > 10000:
            suggestions.append(SpeedSuggestion(
                category="data",
                suggestion="Use incremental/online learning for HMM",
                impact="medium",
                implementation_effort="complex",
                details=(
                    "For HMM-based steps, use incremental updates instead of "
                    "full refitting. The HMMLearn library supports partial_fit. "
                    "This can reduce HMM training time by 60-70%."
                )
            ))
        
        # ====== Model-level suggestions ======
        if 'xgboost' in str(model_types).lower() or 'xgb' in str(model_types).lower():
            suggestions.append(SpeedSuggestion(
                category="model",
                suggestion="Use histogram-based tree method with GPU",
                impact="high",
                implementation_effort="easy",
                details=(
                    "Set tree_method='gpu_hist' if GPU available, or 'hist' for CPU. "
                    "This is already implemented but ensure it's enabled. "
                    "Can provide 10-50x speedup over 'exact'."
                )
            ))
        
        if 'lightgbm' in str(model_types).lower() or 'lgbm' in str(model_types).lower():
            suggestions.append(SpeedSuggestion(
                category="model",
                suggestion="Use binary dataset format for LightGBM",
                impact="high",
                implementation_effort="easy",
                details=(
                    "Pre-convert data to LightGBM binary format using "
                    "train_data.save_binary('train.bin'). Load with lgb.Dataset('train.bin'). "
                    "Reduces data loading time by 80%+."
                )
            ))
        
        if 'ngboost' in str(model_types).lower():
            suggestions.append(SpeedSuggestion(
                category="model",
                suggestion="Use Gaussian distribution and increase minibatch_frac",
                impact="medium",
                implementation_effort="easy",
                details=(
                    "Use Dist=Normal (Gaussian) instead of complex distributions. "
                    "Set minibatch_frac=0.3-0.5 for stochastic training. "
                    "Can reduce training time by 40-60%."
                )
            ))
        
        if 'knn' in str(model_types).lower():
            suggestions.append(SpeedSuggestion(
                category="model",
                suggestion="Use Faiss or HNSW for approximate nearest neighbors",
                impact="high",
                implementation_effort="moderate",
                details=(
                    "Replace sklearn KNN with Faiss IVF index or HNSW. "
                    "For 50k+ samples, this can be 100x+ faster with "
                    "95%+ accuracy retention."
                )
            ))
        
        # ====== Algorithm-level suggestions ======
        if 'hmm' in step_name.lower():
            suggestions.append(SpeedSuggestion(
                category="algorithm",
                suggestion="Reduce HMM convergence threshold",
                impact="medium",
                implementation_effort="easy",
                details=(
                    "Increase tol from 1e-4 to 1e-3 or 1e-2. "
                    "Reduce n_iter from 100 to 50. "
                    "For regime detection, early convergence often sufficient."
                )
            ))
            
            suggestions.append(SpeedSuggestion(
                category="algorithm",
                suggestion="Use diagonal covariance for Gaussian HMM",
                impact="high",
                implementation_effort="easy",
                details=(
                    "Set covariance_type='diag' instead of 'full'. "
                    "Reduces parameter count from O(d^2) to O(d). "
                    "For high-dimensional data, can be 10x+ faster."
                )
            ))
        
        suggestions.append(SpeedSuggestion(
            category="algorithm",
            suggestion="Implement warm start for HPO",
            impact="high",
            implementation_effort="easy",
            details=(
                "Use best params from previous runs as starting point. "
                "Reduces HPO trials needed by 30-50%. "
                "Already implemented in StandardizedXGBTrainer, "
                "ensure it's enabled in all steps."
            )
        ))
        
        # ====== Hardware suggestions ======
        if n_samples > 100000 or n_features > 1000:
            suggestions.append(SpeedSuggestion(
                category="hardware",
                suggestion="Use GPU acceleration where available",
                impact="high",
                implementation_effort="moderate",
                details=(
                    "XGBoost: tree_method='gpu_hist'\n"
                    "LightGBM: device='gpu'\n"
                    "PyTorch models: .to('cuda')\n"
                    "Can provide 5-50x speedup for large datasets."
                )
            ))
        
        suggestions.append(SpeedSuggestion(
            category="hardware",
            suggestion="Optimize CPU parallelism",
            impact="medium",
            implementation_effort="easy",
            details=(
                "Set n_jobs=-1 or n_jobs=<num_physical_cores>. "
                "For HPO, use parallel trial evaluation with Optuna's "
                "n_jobs parameter. Avoid over-subscription."
            )
        ))
        
        # ====== Step-specific suggestions ======
        if 'breakout' in step_name.lower():
            suggestions.append(SpeedSuggestion(
                category="algorithm",
                suggestion="Use single-stage model for breakout detection",
                impact="medium",
                implementation_effort="moderate",
                details=(
                    "The 2-stage model (classification + regression) can be "
                    "replaced with a single model predicting scalar directly. "
                    "This halves training time with minimal accuracy loss."
                )
            ))
        
        if 'regime' in step_name.lower():
            suggestions.append(SpeedSuggestion(
                category="algorithm",
                suggestion="Cache regime predictions",
                impact="medium",
                implementation_effort="easy",
                details=(
                    "Cache regime predictions for overlapping windows. "
                    "Use oof_predictions from previous windows as features. "
                    "Reduces redundant computation in rolling windows."
                )
            ))
        
        return suggestions
    
    @staticmethod
    def format_suggestions(suggestions: List[SpeedSuggestion]) -> str:
        """Format suggestions as markdown for documentation."""
        lines = ["# Training Speed Suggestions\n"]
        
        # Group by category
        by_category = {}
        for s in suggestions:
            if s.category not in by_category:
                by_category[s.category] = []
            by_category[s.category].append(s)
        
        for category, cat_suggestions in by_category.items():
            lines.append(f"\n## {category.title()} Optimizations\n")
            
            for s in cat_suggestions:
                impact_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(s.impact, "⚪")
                effort_emoji = {"easy": "✅", "moderate": "⚠️", "complex": "🔧"}.get(s.implementation_effort, "❓")
                
                lines.append(f"### {s.suggestion}")
                lines.append(f"- **Impact**: {impact_emoji} {s.impact}")
                lines.append(f"- **Effort**: {effort_emoji} {s.implementation_effort}")
                lines.append(f"\n{s.details}\n")
        
        return "\n".join(lines)


# ============================================================================
# Integration Helpers
# ============================================================================

def apply_warm_start_and_subsampling(
    step_name: str,
    model_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    for_hpo: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any], WarmStartManager]:
    """
    Apply warm start and subsampling for a training step.
    
    This is a convenience function that:
    1. Creates a warm start manager
    2. Loads previous best params if available
    3. Applies dynamic subsampling if for_hpo
    4. Returns sampled data and warm start params
    
    Args:
        step_name: Name of the step
        model_id: Unique model identifier
        X: Feature data
        y: Target data
        config: Training configuration
        for_hpo: If True, apply subsampling for HPO
        
    Returns:
        Tuple of (X_processed, y_processed, warm_params, warm_manager)
    """
    # Create warm start manager
    config_hash = WarmStartManager.compute_config_hash(config)
    warm_manager = WarmStartManager(
        model_id=model_id,
        model_type=step_name,
        model_config_hash=config_hash
    )
    
    # Load warm start params
    warm_params = warm_manager.load_params() or {}
    
    # Apply subsampling for HPO
    if for_hpo:
        subsampler = DynamicSubsampler()
        X_processed, y_processed = subsampler.sample(
            X, y, 
            stratify=config.get('task_type') == 'classification'
        )
        
        info = subsampler.get_subsample_info(len(X))
        logger.info(
            f"Subsampling for HPO: {info['original_samples']} -> "
            f"{info['sampled_size']} ({info['sample_pct']:.1%})"
        )
    else:
        X_processed = X
        y_processed = y
    
    return X_processed, y_processed, warm_params, warm_manager


def get_step_specific_suggestions(step_name: str) -> str:
    """
    Get training speed suggestions specific to a step.
    
    Args:
        step_name: Name of the ML step
        
    Returns:
        Markdown-formatted suggestions
    """
    # Estimate dataset characteristics based on step
    step_configs = {
        'hmm_ml_alpha_step': {'n_samples': 50000, 'n_features': 100, 'models': ['lgbm', 'hmm']},
        'hmm_macro_regime': {'n_samples': 20000, 'n_features': 50, 'models': ['lgbm', 'hmm']},
        'ml_reversion_regime_step': {'n_samples': 50000, 'n_features': 150, 'models': ['xgboost']},
        'ml_smc_regime_step': {'n_samples': 50000, 'n_features': 100, 'models': ['xgboost']},
        'ml_breakout_bounce_regime_step': {'n_samples': 30000, 'n_features': 200, 'models': ['xgboost']},
        'ml_liquidity_regime_step': {'n_samples': 50000, 'n_features': 80, 'models': ['xgboost']},
        'ml_risk_regime_step': {'n_samples': 50000, 'n_features': 100, 'models': ['xgboost']},
        'ml_path_regime_step': {'n_samples': 30000, 'n_features': 150, 'models': ['xgboost']},
    }
    
    cfg = step_configs.get(step_name, {'n_samples': 30000, 'n_features': 100, 'models': ['xgboost']})
    
    suggestions = TrainingSpeedSuggestions.get_suggestions(
        step_name=step_name,
        n_samples=cfg['n_samples'],
        n_features=cfg['n_features'],
        model_types=cfg['models']
    )
    
    return TrainingSpeedSuggestions.format_suggestions(suggestions)
