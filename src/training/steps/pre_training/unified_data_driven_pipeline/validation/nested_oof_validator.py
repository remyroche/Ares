"""
Nested Out-of-Fold Validator for Label Leakage Prevention

This module implements strict fold-level isolation to prevent label leakage
through optimization feedback loops. Uses nested OOF validation where:
- Outer loop: Defines labels and economic validation
- Inner loop: Tunes features and interactions
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score

# Import ML Commons utilities
try:
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator, UnifiedCVResult
    from src.utils.ml_common.optimization.pareto import (
        Solution, ParetoFront, compute_pareto_front
    )
    ML_COMMONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML Commons not available: {e}")
    ML_COMMONS_AVAILABLE = False

# Import purged K-fold
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
except ImportError:
    PURGED_KFOLD_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class NestedOOFConfig:
    """Configuration for nested OOF validation."""
    
    # Outer loop (labeling) parameters
    outer_n_splits: int = 5
    outer_embargo_days: int = 7
    outer_gap_days: int = 1
    
    # Inner loop (feature optimization) parameters  
    inner_n_splits: int = 3
    inner_embargo_days: int = 3
    inner_gap_days: int = 1
    
    # Isolation parameters
    min_time_gap_days: int = 14  # Minimum gap between outer and inner folds
    strict_isolation: bool = True  # Enforce no overlap between folds
    
    # Validation metrics
    enable_economic_validation: bool = True
    enable_statistical_validation: bool = True
    enable_interpretability_validation: bool = True
    
    # Logging
    verbose: bool = True


@dataclass
class FoldIsolation:
    """Represents a fold with strict isolation boundaries."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    embargo_start: datetime
    embargo_end: datetime
    isolation_zone_start: datetime
    isolation_zone_end: datetime


@dataclass
class NestedOOFResult:
    """Result of nested OOF validation."""
    
    # Outer loop results (labeling)
    outer_ic_scores: List[float] = field(default_factory=list)
    outer_sharpe_scores: List[float] = field(default_factory=list)
    outer_stability_scores: List[float] = field(default_factory=list)
    
    # Inner loop results (feature optimization)
    inner_ic_scores: List[float] = field(default_factory=list)
    inner_sharpe_scores: List[float] = field(default_factory=list)
    inner_stability_scores: List[float] = field(default_factory=list)
    
    # Isolation validation
    isolation_violations: List[str] = field(default_factory=list)
    max_time_overlap: float = 0.0
    
    # Overall metrics
    final_ic: float = 0.0
    final_sharpe: float = 0.0
    final_stability: float = 0.0
    
    # Validation status
    label_leakage_detected: bool = False
    isolation_valid: bool = True


class NestedOOFValidator:
    """
    Nested Out-of-Fold Validator with strict isolation.
    
    Prevents label leakage by ensuring:
    1. Outer loop defines labels and economic validation
    2. Inner loop tunes features without access to future labels
    3. Strict time-based isolation between folds
    4. No information leakage between optimization loops
    """
    
    def __init__(self, config: Optional[NestedOOFConfig] = None):
        """Initialize the nested OOF validator."""
        self.config = config or NestedOOFConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize fold isolation tracking
        self.fold_isolations: List[FoldIsolation] = []
        self.validation_results: List[NestedOOFResult] = []
        
        if self.config.verbose:
            tprint("🔒 Initializing NestedOOFValidator with strict isolation")
    
    def create_fold_isolation(self, 
                            data: pd.DataFrame, 
                            outer_fold: int,
                            inner_fold: int) -> FoldIsolation:
        """
        Create strict fold isolation boundaries.
        
        Args:
            data: Input data with datetime index
            outer_fold: Outer loop fold number
            inner_fold: Inner loop fold number
            
        Returns:
            FoldIsolation with strict boundaries
        """
        if not hasattr(data.index, 'to_pydatetime'):
            raise ValueError("Data must have datetime index for fold isolation")
        
        # Calculate fold boundaries with strict isolation
        total_days = (data.index[-1] - data.index[0]).days
        outer_fold_size = total_days // self.config.outer_n_splits
        inner_fold_size = outer_fold_size // self.config.inner_n_splits
        
        # Outer fold boundaries
        outer_start = data.index[0] + timedelta(days=outer_fold * outer_fold_size)
        outer_end = outer_start + timedelta(days=outer_fold_size)
        
        # Inner fold boundaries (within outer fold)
        inner_start = outer_start + timedelta(days=inner_fold * inner_fold_size)
        inner_end = inner_start + timedelta(days=inner_fold_size)
        
        # Test boundaries (last 20% of inner fold)
        test_size = int(inner_fold_size * 0.2)
        test_start = inner_end - timedelta(days=test_size)
        test_end = inner_end
        
        # Train boundaries (first 80% of inner fold)
        train_end = test_start - timedelta(days=self.config.inner_gap_days)
        train_start = inner_start
        
        # Embargo boundaries
        embargo_start = test_end
        embargo_end = embargo_start + timedelta(days=self.config.inner_embargo_days)
        
        # Isolation zone (minimum gap between folds)
        isolation_zone_start = embargo_end
        isolation_zone_end = isolation_zone_start + timedelta(days=self.config.min_time_gap_days)
        
        return FoldIsolation(
            fold_id=f"{outer_fold}_{inner_fold}",
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            isolation_zone_start=isolation_zone_start,
            isolation_zone_end=isolation_zone_end
        )
    
    def validate_fold_isolation(self, 
                              fold: FoldIsolation, 
                              data: pd.DataFrame) -> List[str]:
        """
        Validate that fold isolation is maintained.
        
        Args:
            fold: Fold isolation to validate
            data: Data to check against
            
        Returns:
            List of isolation violations
        """
        violations = []
        
        # Check for time overlap with other folds
        for other_fold in self.fold_isolations:
            if other_fold.fold_id == fold.fold_id:
                continue
                
            # Check train/test overlap
            if (fold.train_start <= other_fold.test_end and 
                fold.train_end >= other_fold.test_start):
                violations.append(f"Train/Test overlap between {fold.fold_id} and {other_fold.fold_id}")
            
            # Check embargo overlap
            if (fold.embargo_start <= other_fold.embargo_end and 
                fold.embargo_end >= other_fold.embargo_start):
                violations.append(f"Embargo overlap between {fold.fold_id} and {other_fold.fold_id}")
        
        # Check isolation zone
        if fold.isolation_zone_start < fold.embargo_end:
            violations.append(f"Isolation zone starts before embargo ends in {fold.fold_id}")
        
        return violations
    
    def perform_nested_validation(self, 
                                data: pd.DataFrame,
                                labels: pd.Series,
                                feature_optimizer: callable,
                                label_optimizer: callable) -> NestedOOFResult:
        """
        Perform nested OOF validation with strict isolation.
        
        Args:
            data: Input features
            labels: Target labels
            feature_optimizer: Function to optimize features (inner loop)
            label_optimizer: Function to optimize labels (outer loop)
            
        Returns:
            NestedOOFResult with validation metrics
        """
        if self.config.verbose:
            tprint("🔒 Starting nested OOF validation with strict isolation")
        
        result = NestedOOFResult()
        
        # Outer loop: Label optimization and economic validation
        for outer_fold in range(self.config.outer_n_splits):
            if self.config.verbose:
                tprint(f"🔄 Outer loop fold {outer_fold + 1}/{self.config.outer_n_splits}")
            
            # Inner loop: Feature optimization
            for inner_fold in range(self.config.inner_n_splits):
                if self.config.verbose:
                    tprint(f"  🔄 Inner loop fold {inner_fold + 1}/{self.config.inner_n_splits}")
                
                # Create fold isolation
                fold = self.create_fold_isolation(data, outer_fold, inner_fold)
                
                # Validate isolation
                violations = self.validate_fold_isolation(fold, data)
                if violations:
                    result.isolation_violations.extend(violations)
                    result.isolation_valid = False
                    if self.config.verbose:
                        tprint_warning(f"⚠️ Isolation violations: {violations}")
                
                # Store fold for future validation
                self.fold_isolations.append(fold)
                
                # Extract fold data with strict boundaries
                train_data, train_labels = self._extract_fold_data(
                    data, labels, fold.train_start, fold.train_end
                )
                test_data, test_labels = self._extract_fold_data(
                    data, labels, fold.test_start, fold.test_end
                )
                
                # Inner loop: Feature optimization (no access to future labels)
                inner_metrics = self._optimize_features_inner_loop(
                    train_data, train_labels, test_data, test_labels, feature_optimizer
                )
                
                # Outer loop: Label optimization and economic validation
                outer_metrics = self._optimize_labels_outer_loop(
                    train_data, train_labels, test_data, test_labels, label_optimizer
                )
                
                # Store metrics
                result.inner_ic_scores.append(inner_metrics.get('ic', 0.0))
                result.inner_sharpe_scores.append(inner_metrics.get('sharpe', 0.0))
                result.inner_stability_scores.append(inner_metrics.get('stability', 0.0))
                
                result.outer_ic_scores.append(outer_metrics.get('ic', 0.0))
                result.outer_sharpe_scores.append(outer_metrics.get('sharpe', 0.0))
                result.outer_stability_scores.append(outer_metrics.get('stability', 0.0))
        
        # Calculate final metrics
        result.final_ic = np.mean(result.outer_ic_scores)
        result.final_sharpe = np.mean(result.outer_sharpe_scores)
        result.final_stability = np.mean(result.outer_stability_scores)
        
        # Check for label leakage
        result.label_leakage_detected = self._detect_label_leakage(result)
        
        if self.config.verbose:
            tprint_success(f"✅ Nested OOF validation completed")
            tprint(f"📊 Final IC: {result.final_ic:.4f}")
            tprint(f"📊 Final Sharpe: {result.final_sharpe:.4f}")
            tprint(f"📊 Final Stability: {result.final_stability:.4f}")
            tprint(f"🔒 Label leakage detected: {result.label_leakage_detected}")
        
        return result
    
    def _extract_fold_data(self, 
                          data: pd.DataFrame, 
                          labels: pd.Series,
                          start_time: datetime, 
                          end_time: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract data for a specific time period with strict boundaries."""
        mask = (data.index >= start_time) & (data.index < end_time)
        return data[mask], labels[mask]
    
    def _optimize_features_inner_loop(self, 
                                    train_data: pd.DataFrame,
                                    train_labels: pd.Series,
                                    test_data: pd.DataFrame,
                                    test_labels: pd.Series,
                                    feature_optimizer: callable) -> Dict[str, float]:
        """Optimize features in inner loop (no access to future labels)."""
        try:
            # Feature optimization without economic validation
            optimized_features = feature_optimizer(train_data, train_labels)
            
            # Statistical validation only
            ic_score = self._calculate_ic(optimized_features, test_labels)
            stability_score = self._calculate_stability(optimized_features, test_labels)
            
            return {
                'ic': ic_score,
                'stability': stability_score,
                'sharpe': 0.0  # No economic validation in inner loop
            }
        except Exception as e:
            self.logger.warning(f"Feature optimization failed: {e}")
            return {'ic': 0.0, 'stability': 0.0, 'sharpe': 0.0}
    
    def _optimize_labels_outer_loop(self, 
                                  train_data: pd.DataFrame,
                                  train_labels: pd.Series,
                                  test_data: pd.DataFrame,
                                  test_labels: pd.Series,
                                  label_optimizer: callable) -> Dict[str, float]:
        """Optimize labels in outer loop with economic validation."""
        try:
            # Label optimization with economic validation
            optimized_labels = label_optimizer(train_data, train_labels)
            
            # Full validation including economic metrics
            ic_score = self._calculate_ic(optimized_labels, test_labels)
            sharpe_score = self._calculate_sharpe(optimized_labels, test_labels)
            stability_score = self._calculate_stability(optimized_labels, test_labels)
            
            return {
                'ic': ic_score,
                'sharpe': sharpe_score,
                'stability': stability_score
            }
        except Exception as e:
            self.logger.warning(f"Label optimization failed: {e}")
            return {'ic': 0.0, 'sharpe': 0.0, 'stability': 0.0}
    
    def _calculate_ic(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate Information Coefficient."""
        try:
            correlation = predictions.corr(actual)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_sharpe(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            returns = predictions.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            return returns.mean() / returns.std() if returns.std() > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_stability(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate prediction stability."""
        try:
            # Rolling correlation stability
            window = min(20, len(predictions) // 4)
            if window < 5:
                return 0.0
            
            rolling_corr = predictions.rolling(window).corr(actual.rolling(window))
            return rolling_corr.std() if not rolling_corr.std() is np.nan else 0.0
        except:
            return 0.0
    
    def _detect_label_leakage(self, result: NestedOOFResult) -> bool:
        """Detect potential label leakage from validation patterns."""
        # Check for suspiciously high performance
        if result.final_ic > 0.5:  # Suspiciously high IC
            return True
        
        # Check for perfect stability (potential leakage)
        if result.final_stability < 0.01:  # Too stable
            return True
        
        # Check for isolation violations
        if not result.isolation_valid:
            return True
        
        return False
