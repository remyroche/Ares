"""
Temporal Cross-Validation Implementation

Proper time series cross-validation with temporal splits to prevent lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from dataclasses import dataclass
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class TemporalCVConfig:
    """Configuration for temporal cross-validation."""
    
    # Cross-validation settings
    n_splits: int = 5
    test_size: float = 0.2  # 20% for test
    gap_size: int = 1  # Gap between train and test
    
    # Time series specific settings
    enable_temporal_splits: bool = True
    strict_temporal_order: bool = True
    min_train_size: float = 0.3  # Minimum 30% for training
    min_test_size: float = 0.1  # Minimum 10% for test
    
    # Validation settings
    enable_validation: bool = True
    validation_size: float = 0.1  # 10% for validation
    early_stopping_patience: int = 3
    
    # Performance tracking
    track_performance: bool = True
    save_predictions: bool = False
    detailed_reporting: bool = True

class TimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validation splitter.
    
    Ensures proper temporal ordering and prevents lookahead bias.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 gap_size: int = 1,
                 min_train_size: float = 0.3,
                 min_test_size: float = 0.1):
        """
        Initialize time series splitter.
        
        Args:
            n_splits: Number of splits
            test_size: Test set size (0.0 to 1.0)
            gap_size: Gap between train and test sets
            min_train_size: Minimum training set size
            min_test_size: Minimum test set size
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_size = gap_size
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        
        # Validate parameters
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if not 0 < min_train_size < 1:
            raise ValueError(f"min_train_size must be between 0 and 1, got {min_train_size}")
        if not 0 < min_test_size < 1:
            raise ValueError(f"min_test_size must be between 0 and 1, got {min_test_size}")
        if min_train_size + min_test_size > 1:
            raise ValueError("min_train_size + min_test_size cannot exceed 1")
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits for time series.
        
        Args:
            X: Input data
            y: Target data (optional)
            groups: Group data (optional)
            
        Yields:
            Tuple[np.ndarray, np.ndarray]: (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate split points
        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_start = int(n_samples * (1 - self.test_size - i * 0.1))
            test_end = int(n_samples * (1 - i * 0.1))
            
            # Ensure minimum test size
            if test_end - test_start < int(n_samples * self.min_test_size):
                continue
            
            # Calculate training set boundaries
            train_end = test_start - self.gap_size
            train_start = max(0, int(n_samples * self.min_train_size))
            
            # Ensure minimum train size
            if train_end - train_start < int(n_samples * self.min_train_size):
                continue
            
            # Generate indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        return self.n_splits

class TemporalCrossValidator:
    """Enhanced temporal cross-validation with comprehensive evaluation."""
    
    def __init__(self, config: TemporalCVConfig):
        self.config = config
        self.cv_results = []
        self.performance_history = []
        
    def cross_validate(self, 
                         estimator, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform temporal cross-validation.
        
        Args:
            estimator: Model to validate
            X: Input features
            y: Target labels
            timestamps: Optional timestamps
            feature_names: Optional feature names
            
        Returns:
            Dict: Cross-validation results
        """
        if not self.config.enable_temporal_splits:
            raise ValueError("Temporal splits are disabled")
        
        # Create time series splitter
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            test_size=self.config.test_size,
            gap_size=self.config.gap_size,
            min_train_size=self.config.min_train_size,
            min_test_size=self.config.min_test_size
        )
        
        # Perform cross-validation
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
            try:
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Validate temporal order
                if timestamps is not None:
                    train_times = timestamps[train_idx]
                    test_times = timestamps[test_idx]
                    
                    if np.max(train_times) >= np.min(test_times):
                        logger.warning(f"Fold {fold}: Temporal order violation detected")
                        continue
                
                # Train model
                estimator.fit(X_train, y_train)
                
                # Make predictions
                y_pred = estimator.predict(X_test)
                y_pred_proba = getattr(estimator, 'predict_proba', lambda x: None)(X_test)
                
                # Calculate metrics
                fold_metrics = self._calculate_fold_metrics(y_test, y_pred, y_pred_proba)
                fold_metrics['fold'] = fold
                fold_metrics['train_size'] = len(X_train)
                fold_metrics['test_size'] = len(X_test)
                
                # Store fold results
                fold_results.append(fold_metrics)
                cv_scores.append(fold_metrics['accuracy'])
                
                # Track performance if enabled
                if self.config.track_performance:
                    self.performance_history.append({
                        'fold': fold,
                        'accuracy': fold_metrics['accuracy'],
                        'f1': fold_metrics['f1'],
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    })
                
            except Exception as e:
                logger.error(f"Fold {fold} failed: {e}")
                continue
        
        # Calculate overall results
        if not cv_scores:
            raise ValueError("No valid folds completed")
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'fold_results': fold_results,
            'n_folds': len(cv_scores),
            'successful_folds': len(cv_scores)
        }
        
        # Add performance analysis
        if self.config.detailed_reporting:
            results['performance_analysis'] = self._analyze_performance(fold_results)
        
        # Store results
        self.cv_results.append(results)
        
        return results
    
    def _calculate_fold_metrics(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted')
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            from sklearn.metrics import log_loss, roc_auc_score
            
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['log_loss'] = None
                metrics['auc'] = None
        
        return metrics
    
    def _analyze_performance(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Analyze cross-validation performance."""
        if not fold_results:
            return {}
        
        # Extract metrics
        accuracies = [fold['accuracy'] for fold in fold_results]
        f1_scores = [fold['f1'] for fold in fold_results]
        
        # Calculate statistics
        analysis = {
            'accuracy_stats': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'cv': np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0
            },
            'f1_stats': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores),
                'cv': np.std(f1_scores) / np.mean(f1_scores) if np.mean(f1_scores) > 0 else 0
            },
            'stability': {
                'low_variance': np.std(accuracies) < 0.05,
                'consistent_performance': np.max(accuracies) - np.min(accuracies) < 0.1,
                'reliable': np.mean(accuracies) > 0.6
            }
        }
        
        # Add recommendations
        recommendations = []
        if analysis['accuracy_stats']['cv'] > 0.2:
            recommendations.append("High variance in cross-validation - consider more data or regularization")
        if analysis['accuracy_stats']['mean'] < 0.6:
            recommendations.append("Low accuracy - consider feature engineering or model selection")
        if not analysis['stability']['consistent_performance']:
            recommendations.append("Inconsistent performance - check for data leakage or overfitting")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all cross-validation runs."""
        if not self.cv_results:
            return {'message': 'No cross-validation results available'}
        
        # Aggregate results
        all_scores = []
        for result in self.cv_results:
            all_scores.extend(result['cv_scores'])
        
        return {
            'total_runs': len(self.cv_results),
            'total_folds': len(all_scores),
            'overall_mean': np.mean(all_scores),
            'overall_std': np.std(all_scores),
            'best_run': max(self.cv_results, key=lambda x: x['mean_score']),
            'worst_run': min(self.cv_results, key=lambda x: x['mean_score'])
        }

class TemporalValidationPipeline:
    """Complete temporal validation pipeline."""
    
    def __init__(self, config: TemporalCVConfig):
        self.config = config
        self.cv = TemporalCrossValidator(config)
        self.validation_results = []
        
    def validate_model(self, 
                      estimator, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      timestamps: Optional[np.ndarray] = None,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Complete temporal validation pipeline.
        
        Args:
            estimator: Model to validate
            X: Input features
            y: Target labels
            timestamps: Optional timestamps
            feature_names: Optional feature names
            
        Returns:
            Dict: Complete validation results
        """
        # Perform temporal cross-validation
        cv_results = self.cv.cross_validate(
            estimator, X, y, timestamps, feature_names
        )
        
        # Add validation metadata
        validation_results = {
            'temporal_cv': cv_results,
            'validation_timestamp': len(self.validation_results),
            'config': {
                'n_splits': self.config.n_splits,
                'test_size': self.config.test_size,
                'gap_size': self.config.gap_size,
                'temporal_splits_enabled': self.config.enable_temporal_splits
            },
            'summary': {
                'mean_score': cv_results['mean_score'],
                'std_score': cv_results['std_score'],
                'n_folds': cv_results['n_folds'],
                'successful_folds': cv_results['successful_folds']
            }
        }
        
        # Store results
        self.validation_results.append(validation_results)
        
        return validation_results

# Global instances for easy access
DEFAULT_TEMPORAL_CV_CONFIG = TemporalCVConfig()
DEFAULT_TEMPORAL_CV = TemporalCrossValidator(DEFAULT_TEMPORAL_CV_CONFIG)
DEFAULT_VALIDATION_PIPELINE = TemporalValidationPipeline(DEFAULT_TEMPORAL_CV_CONFIG)

def get_temporal_cv_config() -> TemporalCVConfig:
    """Get the default temporal CV configuration."""
    return DEFAULT_TEMPORAL_CV_CONFIG

def get_temporal_cv() -> TemporalCrossValidator:
    """Get the default temporal cross-validator."""
    return DEFAULT_TEMPORAL_CV

def get_validation_pipeline() -> TemporalValidationPipeline:
    """Get the default validation pipeline."""
    return DEFAULT_VALIDATION_PIPELINE

def create_time_series_split(**kwargs) -> TimeSeriesSplit:
    """Create a time series splitter with custom parameters."""
    return TimeSeriesSplit(**kwargs)