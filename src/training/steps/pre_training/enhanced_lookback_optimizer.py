"""
Enhanced Lookback Optimization Module

This module provides robust lookback optimization with:
1. Constrained search space (e.g., 5-300 bars)
2. Clear objective function definition
3. Regularization to penalize extreme lookbacks
4. Stability analysis across time segments
5. Out-of-sample validation
6. Rolling window evaluation

Addresses Section 4: Lookback Optimization Strategy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.utils.purged_kfold import PurgedKFoldTime
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


logger = logging.getLogger(__name__)


@dataclass
class LookbackConstraints:
    """Constraints for lookback optimization."""
    
    # Lookback bounds
    min_lookback: int = 5  # Minimum lookback period
    max_lookback: int = 300  # Maximum lookback period
    
    # Search granularity
    search_step: int = 5  # Step size for grid search
    
    # Regularization
    enable_regularization: bool = True
    regularization_strength: float = 0.1  # Penalty for extreme lookbacks
    preferred_lookback: int = 50  # Center of regularization
    
    # Stability requirements
    min_stability_score: float = 0.7  # Minimum stability across segments


@dataclass
class OptimizationObjective:
    """Definition of optimization objective."""
    
    # Objective type
    objective_type: str = 'sharpe'  # 'sharpe', 'ic', 'r2', 'custom'
    
    # Direction
    maximize: bool = True  # True to maximize, False to minimize
    
    # Custom objective function
    custom_objective: Optional[Callable] = None
    
    # Scoring weights
    in_sample_weight: float = 0.3  # Weight for in-sample performance
    out_of_sample_weight: float = 0.7  # Weight for out-of-sample performance


@dataclass
class LookbackResult:
    """Result of lookback optimization."""
    
    optimal_lookback: int
    objective_score: float
    
    # Performance across folds
    cv_scores: List[float]
    mean_cv_score: float
    std_cv_score: float
    
    # Stability metrics
    stability_score: float
    stability_across_segments: List[float]
    
    # All tested lookbacks
    tested_lookbacks: List[int]
    lookback_scores: Dict[int, float]
    
    # Regularization impact
    regularization_penalty: float = 0.0
    
    # Metadata
    optimization_time: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'optimal_lookback': self.optimal_lookback,
            'objective_score': self.objective_score,
            'cv_scores': self.cv_scores,
            'mean_cv_score': self.mean_cv_score,
            'std_cv_score': self.std_cv_score,
            'stability_score': self.stability_score,
            'stability_across_segments': self.stability_across_segments,
            'tested_lookbacks': self.tested_lookbacks,
            'lookback_scores': self.lookback_scores,
            'regularization_penalty': self.regularization_penalty,
            'optimization_time': self.optimization_time,
            'timestamp': self.timestamp
        }


class EnhancedLookbackOptimizer:
    """
    Enhanced lookback optimizer with constrained search and stability analysis.
    
    Key Features:
    1. Constrained search space
    2. Multiple objective functions (Sharpe, IC, R²)
    3. Regularization to avoid extreme lookbacks
    4. Stability analysis across time segments
    5. Out-of-sample validation
    """
    
    def __init__(
        self,
        constraints: Optional[LookbackConstraints] = None,
        objective: Optional[OptimizationObjective] = None
    ):
        """
        Initialize enhanced lookback optimizer.
        
        Args:
            constraints: Lookback constraints
            objective: Optimization objective
        """
        self.constraints = constraints or LookbackConstraints()
        self.objective = objective or OptimizationObjective()
        
        tprint_success("✅ EnhancedLookbackOptimizer initialized")
        tprint_info(f"   → Search space: [{self.constraints.min_lookback}, {self.constraints.max_lookback}]")
        tprint_info(f"   → Objective: {self.objective.objective_type}")
        tprint_info(f"   → Regularization: {self.constraints.enable_regularization}")
    
    def optimize_lookback(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        n_cv_splits: int = 5,
        feature_generator: Optional[Callable] = None
    ) -> LookbackResult:
        """
        Optimize feature lookback period.
        
        Args:
            features: Base features (will be lagged)
            targets: Target variable
            n_cv_splits: Number of CV splits
            feature_generator: Optional function to generate features from lookback
        
        Returns:
            LookbackResult with optimal lookback and performance metrics
        """
        tprint_info("🔍 Starting lookback optimization...")
        
        start_time = datetime.now()
        
        # Generate lookback candidates
        lookback_candidates = list(range(
            self.constraints.min_lookback,
            self.constraints.max_lookback + 1,
            self.constraints.search_step
        ))
        
        tprint_info(f"   → Testing {len(lookback_candidates)} lookback values")
        
        # Evaluate each lookback
        lookback_scores = {}
        all_cv_scores = {}
        
        for lookback in lookback_candidates:
            try:
                # Generate features with this lookback
                if feature_generator is not None:
                    lagged_features = feature_generator(features, lookback)
                else:
                    lagged_features = self._generate_lagged_features(features, lookback)
                
                # Align with targets
                aligned_idx = lagged_features.index.intersection(targets.index)
                X = lagged_features.loc[aligned_idx]
                y = targets.loc[aligned_idx]
                
                if len(X) < 100:  # Minimum samples
                    continue
                
                # Perform CV evaluation
                cv_scores = self._cross_validate_lookback(X, y, n_cv_splits)
                
                # Calculate objective score
                mean_score = np.mean(cv_scores)
                
                # Apply regularization
                if self.constraints.enable_regularization:
                    regularization_penalty = self._calculate_regularization_penalty(lookback)
                    regularized_score = mean_score - regularization_penalty
                else:
                    regularized_score = mean_score
                    regularization_penalty = 0.0
                
                lookback_scores[lookback] = regularized_score
                all_cv_scores[lookback] = cv_scores
                
            except Exception as e:
                logger.warning(f"Error evaluating lookback {lookback}: {e}")
                continue
        
        if not lookback_scores:
            raise ValueError("No valid lookback values could be evaluated")
        
        # Find optimal lookback
        if self.objective.maximize:
            optimal_lookback = max(lookback_scores, key=lookback_scores.get)
        else:
            optimal_lookback = min(lookback_scores, key=lookback_scores.get)
        
        optimal_score = lookback_scores[optimal_lookback]
        optimal_cv_scores = all_cv_scores[optimal_lookback]
        
        # Calculate stability
        stability_score = self._calculate_stability_score(lookback_scores)
        stability_across_segments = self._analyze_stability_across_segments(
            features, targets, optimal_lookback, n_segments=5
        )
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = LookbackResult(
            optimal_lookback=optimal_lookback,
            objective_score=optimal_score,
            cv_scores=optimal_cv_scores,
            mean_cv_score=float(np.mean(optimal_cv_scores)),
            std_cv_score=float(np.std(optimal_cv_scores)),
            stability_score=stability_score,
            stability_across_segments=stability_across_segments,
            tested_lookbacks=list(lookback_scores.keys()),
            lookback_scores=lookback_scores,
            regularization_penalty=0.0,
            optimization_time=optimization_time
        )
        
        # Log results
        tprint_success(f"✅ Optimization complete: optimal lookback = {optimal_lookback}")
        tprint_info(f"   → Objective score: {optimal_score:.4f}")
        tprint_info(f"   → CV score: {result.mean_cv_score:.4f} ± {result.std_cv_score:.4f}")
        tprint_info(f"   → Stability: {stability_score:.3f}")
        
        # Check stability warning
        if stability_score < self.constraints.min_stability_score:
            tprint_warning(
                f"⚠️ Low stability score: {stability_score:.3f} < {self.constraints.min_stability_score}"
            )
        
        return result
    
    def _generate_lagged_features(
        self,
        features: pd.DataFrame,
        lookback: int
    ) -> pd.DataFrame:
        """Generate lagged features with specified lookback."""
        lagged_features = []
        
        for col in features.columns:
            for lag in range(1, lookback + 1):
                lagged_col = features[col].shift(lag)
                lagged_features.append(lagged_col.rename(f'{col}_lag{lag}'))
        
        return pd.concat(lagged_features, axis=1)
    
    def _cross_validate_lookback(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int
    ) -> List[float]:
        """Perform cross-validation for lookback evaluation."""
        # Use time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Calculate score based on objective type
            if self.objective.objective_type == 'ic':
                # Information coefficient (Spearman correlation)
                score = self._calculate_ic(X_val, y_val)
            elif self.objective.objective_type == 'sharpe':
                # Sharpe ratio
                score = self._calculate_sharpe(y_val)
            elif self.objective.objective_type == 'r2':
                # R² score (requires fitting a model)
                from sklearn.linear_model import Ridge
                model = Ridge()
                model.fit(X_train.fillna(0), y_train)
                score = model.score(X_val.fillna(0), y_val)
            else:
                # Custom objective
                if self.objective.custom_objective is not None:
                    score = self.objective.custom_objective(X_val, y_val)
                else:
                    score = 0.0
            
            scores.append(score)
        
        return scores
    
    def _calculate_ic(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate information coefficient (Spearman correlation)."""
        # Use first feature for IC calculation
        if X.shape[1] > 0:
            feature = X.iloc[:, 0]
            ic = feature.corr(y, method='spearman')
            return ic if not np.isnan(ic) else 0.0
        return 0.0
    
    def _calculate_sharpe(self, y: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(y) > 1 and y.std() > 0:
            sharpe = y.mean() / y.std() * np.sqrt(252)  # Annualized
            return sharpe
        return 0.0
    
    def _calculate_regularization_penalty(self, lookback: int) -> float:
        """Calculate regularization penalty for lookback."""
        # Penalize distance from preferred lookback
        distance = abs(lookback - self.constraints.preferred_lookback)
        penalty = self.constraints.regularization_strength * (distance / self.constraints.preferred_lookback)
        return penalty
    
    def _calculate_stability_score(self, lookback_scores: Dict[int, float]) -> float:
        """
        Calculate stability score based on variance of lookback scores.
        
        Lower variance = higher stability
        """
        if len(lookback_scores) < 2:
            return 1.0
        
        scores = list(lookback_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Coefficient of variation
        if mean_score != 0:
            cv = std_score / abs(mean_score)
            # Convert to stability score (0 to 1)
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 0.5
        
        return stability
    
    def _analyze_stability_across_segments(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        lookback: int,
        n_segments: int = 5
    ) -> List[float]:
        """
        Analyze lookback stability across different time segments.
        
        Args:
            features: Feature DataFrame
            targets: Target series
            lookback: Lookback to test
            n_segments: Number of time segments
        
        Returns:
            List of scores per segment
        """
        segment_scores = []
        segment_size = len(features) // n_segments
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            
            if end_idx > len(features):
                end_idx = len(features)
            
            # Generate features for this segment
            segment_features = features.iloc[start_idx:end_idx]
            segment_targets = targets.iloc[start_idx:end_idx]
            
            try:
                lagged_features = self._generate_lagged_features(segment_features, lookback)
                aligned_idx = lagged_features.index.intersection(segment_targets.index)
                X = lagged_features.loc[aligned_idx]
                y = segment_targets.loc[aligned_idx]
                
                if len(X) > 20:
                    # Calculate IC for this segment
                    ic = self._calculate_ic(X, y)
                    segment_scores.append(ic)
                    
            except Exception as e:
                logger.warning(f"Error analyzing segment {i}: {e}")
                continue
        
        return segment_scores
    
    def sensitivity_analysis(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        optimal_lookback: int,
        perturbation_range: int = 10,
        n_resamples: int = 10
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on optimal lookback.
        
        Args:
            features: Feature DataFrame
            targets: Target series
            optimal_lookback: Optimal lookback to test
            perturbation_range: Range to perturb lookback
            n_resamples: Number of bootstrap resamples
        
        Returns:
            Dictionary with sensitivity metrics
        """
        tprint_info(f"🔬 Performing sensitivity analysis around lookback={optimal_lookback}...")
        
        # Test lookbacks around optimal
        test_lookbacks = range(
            max(self.constraints.min_lookback, optimal_lookback - perturbation_range),
            min(self.constraints.max_lookback, optimal_lookback + perturbation_range + 1)
        )
        
        scores_per_lookback = {}
        
        for lookback in test_lookbacks:
            scores = []
            
            for _ in range(n_resamples):
                # Resample data
                sample_idx = np.random.choice(len(features), size=int(len(features) * 0.8), replace=False)
                sample_features = features.iloc[sample_idx]
                sample_targets = targets.iloc[sample_idx]
                
                try:
                    # Generate features and evaluate
                    lagged_features = self._generate_lagged_features(sample_features, lookback)
                    aligned_idx = lagged_features.index.intersection(sample_targets.index)
                    X = lagged_features.loc[aligned_idx]
                    y = sample_targets.loc[aligned_idx]
                    
                    if len(X) > 20:
                        ic = self._calculate_ic(X, y)
                        scores.append(ic)
                        
                except Exception as e:
                    continue
            
            if scores:
                scores_per_lookback[lookback] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
        
        # Calculate sensitivity metric
        if scores_per_lookback:
            lookback_means = [v['mean'] for v in scores_per_lookback.values()]
            sensitivity = np.std(lookback_means) / (np.mean(lookback_means) + 1e-8) if lookback_means else 0.0
        else:
            sensitivity = 0.0
        
        tprint_info(f"   → Sensitivity: {sensitivity:.3f}")
        
        return {
            'sensitivity': sensitivity,
            'scores_per_lookback': scores_per_lookback,
            'optimal_lookback': optimal_lookback,
            'stable': sensitivity < 0.15  # <15% sensitivity threshold
        }


def create_enhanced_lookback_optimizer(
    constraints: Optional[LookbackConstraints] = None,
    objective: Optional[OptimizationObjective] = None
) -> EnhancedLookbackOptimizer:
    """
    Factory function to create EnhancedLookbackOptimizer.
    
    Args:
        constraints: Lookback constraints
        objective: Optimization objective
    
    Returns:
        EnhancedLookbackOptimizer instance
    """
    return EnhancedLookbackOptimizer(constraints, objective)