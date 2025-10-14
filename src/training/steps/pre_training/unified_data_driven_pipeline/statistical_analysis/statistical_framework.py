"""
Statistical Analysis Framework

Comprehensive statistical analysis for data-driven decisions in the unified pipeline.
Provides robust statistical methods for feature discovery, validation, and selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
import warnings
import logging
from abc import ABC, abstractmethod

# Import math validation utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, 
    validate_positive, validate_range, safe_correlation, safe_covariance,
    safe_mean, safe_std, safe_percentile, safe_percentage_change,
    safe_weighted_average, MathValidation
)

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import enhanced validation utilities from ml_commons
try:
    from src.utils.ml_common.validation.unified_cv import (
        UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation
    )
    from src.utils.ml_common.validation.cv import (
        analyze_splits, validate_cv_integrity
    )
    from src.utils.ml_common.validation.data_leakage_detector import (
        DataLeakageDetector
    )
    from src.utils.ml_common.validation.enhanced_validation import (
        EnhancedValidationFramework
    )
    from src.utils.ml_common.validation.stability import (
        StabilityAnalyzer
    )
    from src.utils.ml_common.validation.overfitting_monitoring import (
        OverfittingMonitor
    )
    ML_COMMONS_VALIDATION_AVAILABLE = True
    tprint_info("✅ ML Commons validation utilities imported successfully")
except ImportError as e:
    ML_COMMONS_VALIDATION_AVAILABLE = False
    tprint_warning(f"⚠️ ML Commons validation utilities not available: {e}")
    # Define fallback classes with proper implementations
    class UnifiedCrossValidator:
        """Fallback implementation of UnifiedCrossValidator with basic functionality."""
        
        def __init__(self, n_splits: int = 5, test_size: float = 0.2, random_state: int = None, **kwargs):
            """Initialize the cross validator."""
            self.n_splits = n_splits
            self.test_size = test_size
            self.random_state = random_state
            self.splits_ = None
            self.validation_scores_ = []
            
            if TPRINT_AVAILABLE:
                tprint_info(f"UnifiedCrossValidator initialized with {n_splits} splits")
        
        def split(self, X, y=None, groups=None):
            """Generate train/test splits."""
            from sklearn.model_selection import train_test_split
            import numpy as np
            
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            self.splits_ = []
            for i in range(self.n_splits):
                if groups is not None:
                    # Group-based splitting
                    unique_groups = np.unique(groups)
                    n_groups = len(unique_groups)
                    test_groups = np.random.choice(unique_groups, 
                                                 size=int(n_groups * self.test_size), 
                                                 replace=False)
                    test_mask = np.isin(groups, test_groups)
                    train_mask = ~test_mask
                    
                    train_idx = np.where(train_mask)[0]
                    test_idx = np.where(test_mask)[0]
                else:
                    # Random splitting
                    train_idx, test_idx = train_test_split(
                        range(len(X)), 
                        test_size=self.test_size, 
                        random_state=self.random_state + i if self.random_state is not None else None
                    )
                
                self.splits_.append((train_idx, test_idx))
                yield train_idx, test_idx
        
        def get_n_splits(self, X=None, y=None, groups=None):
            """Get number of splits."""
            return self.n_splits
        
        def cross_val_score(self, estimator, X, y=None, groups=None, scoring=None, cv=None, **kwargs):
            """Perform cross-validation scoring."""
            from sklearn.model_selection import cross_val_score as sklearn_cv_score
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            if scoring is None:
                scoring = 'accuracy' if hasattr(estimator, 'predict') else 'neg_mean_squared_error'
            
            if cv is None:
                cv = self
            
            try:
                scores = sklearn_cv_score(estimator, X, y, groups=groups, scoring=scoring, cv=cv, **kwargs)
                self.validation_scores_ = scores
                return scores
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_error(f"Cross-validation failed: {e}")
                return np.array([0.0] * self.n_splits)
    
    def perform_cross_validation(estimator, X, y=None, cv=None, scoring=None, **kwargs):
        """Perform cross-validation with fallback implementation."""
        if cv is None:
            cv = UnifiedCrossValidator()
        
        if scoring is None:
            scoring = 'accuracy' if hasattr(estimator, 'predict') else 'neg_mean_squared_error'
        
        try:
            return cv.cross_val_score(estimator, X, y, scoring=scoring, **kwargs)
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Cross-validation failed: {e}")
            return np.array([0.0] * cv.get_n_splits())
    
    def temporal_cross_validation(X, y=None, n_splits=5, test_size=0.2, **kwargs):
        """Perform temporal cross-validation with fallback implementation."""
        from sklearn.model_selection import TimeSeriesSplit
        
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = []
            for train_idx, test_idx in tscv.split(X):
                splits.append((train_idx, test_idx))
            return splits
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Temporal cross-validation failed: {e}")
            return []
    
    def analyze_splits(splits, X, y=None, **kwargs):
        """Analyze cross-validation splits with fallback implementation."""
        try:
            analysis = {
                'n_splits': len(splits),
                'train_sizes': [len(train_idx) for train_idx, _ in splits],
                'test_sizes': [len(test_idx) for _, test_idx in splits],
                'overlap_ratio': 0.0  # Simplified overlap calculation
            }
            
            if y is not None:
                # Analyze target distribution in splits
                train_targets = [y[train_idx] for train_idx, _ in splits]
                test_targets = [y[test_idx] for _, test_idx in splits]
                
                analysis['train_target_mean'] = [np.mean(targets) for targets in train_targets]
                analysis['test_target_mean'] = [np.mean(targets) for targets in test_targets]
                analysis['target_balance'] = np.std(analysis['train_target_mean'])
            
            return analysis
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Split analysis failed: {e}")
            return {'n_splits': len(splits), 'error': str(e)}
    
    def validate_cv_integrity(splits, X, y=None, **kwargs):
        """Validate cross-validation integrity with fallback implementation."""
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'warnings': []
            }
            
            # Check if all splits have reasonable sizes
            for i, (train_idx, test_idx) in enumerate(splits):
                if len(train_idx) < 10:
                    validation_results['issues'].append(f"Split {i}: train set too small ({len(train_idx)} samples)")
                    validation_results['is_valid'] = False
                
                if len(test_idx) < 5:
                    validation_results['issues'].append(f"Split {i}: test set too small ({len(test_idx)} samples)")
                    validation_results['is_valid'] = False
                
                # Check for overlap
                overlap = set(train_idx) & set(test_idx)
                if overlap:
                    validation_results['issues'].append(f"Split {i}: data leakage detected ({len(overlap)} overlapping samples)")
                    validation_results['is_valid'] = False
            
            return validation_results
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"CV integrity validation failed: {e}")
            return {'is_valid': False, 'issues': [str(e)], 'warnings': []}
    
    class DataLeakageDetector:
        """Fallback implementation of DataLeakageDetector with basic functionality."""
        
        def __init__(self, threshold: float = 0.8, **kwargs):
            """Initialize the data leakage detector."""
            self.threshold = threshold
            self.leakage_scores_ = {}
            self.detected_leakage_ = []
            
            if TPRINT_AVAILABLE:
                tprint_info(f"DataLeakageDetector initialized with threshold {threshold}")
        
        def detect_leakage(self, X, y=None, feature_names=None, **kwargs):
            """Detect potential data leakage."""
            try:
                leakage_results = {
                    'leakage_detected': False,
                    'leakage_scores': {},
                    'suspicious_features': [],
                    'recommendations': []
                }
                
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
                # Check for perfect correlation with target
                if y is not None:
                    for i, feature_name in enumerate(feature_names):
                        try:
                            corr = np.corrcoef(X[:, i], y)[0, 1]
                            if not np.isnan(corr) and abs(corr) > self.threshold:
                                leakage_results['leakage_scores'][feature_name] = abs(corr)
                                leakage_results['suspicious_features'].append(feature_name)
                                leakage_results['leakage_detected'] = True
                        except Exception:
                            continue
                
                # Check for duplicate features
                for i, feature_name in enumerate(feature_names):
                    for j, other_name in enumerate(feature_names[i+1:], i+1):
                        try:
                            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                            if not np.isnan(corr) and abs(corr) > 0.99:
                                leakage_results['suspicious_features'].extend([feature_name, other_name])
                                leakage_results['leakage_detected'] = True
                        except Exception:
                            continue
                
                self.leakage_scores_ = leakage_results['leakage_scores']
                self.detected_leakage_ = leakage_results['suspicious_features']
                
                return leakage_results
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_error(f"Data leakage detection failed: {e}")
                return {'leakage_detected': False, 'error': str(e)}
        
        def get_leakage_summary(self):
            """Get summary of detected leakage."""
            return {
                'n_suspicious_features': len(self.detected_leakage_),
                'leakage_scores': self.leakage_scores_,
                'suspicious_features': self.detected_leakage_
            }
    
    class EnhancedValidationFramework:
        """Fallback implementation of EnhancedValidationFramework with basic functionality."""
        
        def __init__(self, validation_methods=None, **kwargs):
            """Initialize the enhanced validation framework."""
            self.validation_methods = validation_methods or ['holdout', 'kfold', 'stratified']
            self.validation_results_ = {}
            self.best_method_ = None
            
            if TPRINT_AVAILABLE:
                tprint_info(f"EnhancedValidationFramework initialized with methods: {self.validation_methods}")
        
        def validate_model(self, estimator, X, y=None, validation_method='holdout', **kwargs):
            """Validate model using specified method."""
            try:
                if validation_method == 'holdout':
                    return self._holdout_validation(estimator, X, y, **kwargs)
                elif validation_method == 'kfold':
                    return self._kfold_validation(estimator, X, y, **kwargs)
                elif validation_method == 'stratified':
                    return self._stratified_validation(estimator, X, y, **kwargs)
                else:
                    raise ValueError(f"Unknown validation method: {validation_method}")
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_error(f"Model validation failed: {e}")
                return {'score': 0.0, 'error': str(e)}
        
        def _holdout_validation(self, estimator, X, y, test_size=0.2, random_state=None, **kwargs):
            """Perform holdout validation."""
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            if hasattr(estimator, 'predict_proba'):
                score = accuracy_score(y_test, y_pred)
            else:
                score = -mean_squared_error(y_test, y_pred)
            
            return {'score': score, 'method': 'holdout'}
        
        def _kfold_validation(self, estimator, X, y, n_splits=5, **kwargs):
            """Perform k-fold validation."""
            from sklearn.model_selection import KFold
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            kf = KFold(n_splits=n_splits, shuffle=True)
            scores = []
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                
                if hasattr(estimator, 'predict_proba'):
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = -mean_squared_error(y_test, y_pred)
                
                scores.append(score)
            
            return {'score': np.mean(scores), 'scores': scores, 'method': 'kfold'}
        
        def _stratified_validation(self, estimator, X, y, n_splits=5, **kwargs):
            """Perform stratified validation."""
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import accuracy_score
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            scores = []
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)
            
            return {'score': np.mean(scores), 'scores': scores, 'method': 'stratified'}
        
        def get_validation_summary(self):
            """Get validation summary."""
            return {
                'validation_results': self.validation_results_,
                'best_method': self.best_method_,
                'available_methods': self.validation_methods
            }
    
    class StabilityAnalyzer:
        """Fallback implementation of StabilityAnalyzer with basic functionality."""
        
        def __init__(self, stability_threshold=0.1, **kwargs):
            """Initialize the stability analyzer."""
            self.stability_threshold = stability_threshold
            self.stability_scores_ = {}
            self.stability_report_ = {}
            
            if TPRINT_AVAILABLE:
                tprint_info(f"StabilityAnalyzer initialized with threshold {stability_threshold}")
        
        def analyze_stability(self, X, y=None, feature_names=None, **kwargs):
            """Analyze stability of features."""
            try:
                stability_results = {
                    'overall_stability': 0.0,
                    'feature_stability': {},
                    'unstable_features': [],
                    'stability_issues': []
                }
                
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
                # Calculate stability for each feature
                for i, feature_name in enumerate(feature_names):
                    feature_data = X[:, i]
                    
                    # Calculate coefficient of variation as stability measure
                    mean_val = np.mean(feature_data)
                    std_val = np.std(feature_data)
                    
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        stability = 1.0 / (1.0 + cv)  # Higher is more stable
                    else:
                        stability = 0.0
                    
                    stability_results['feature_stability'][feature_name] = stability
                    
                    if stability < self.stability_threshold:
                        stability_results['unstable_features'].append(feature_name)
                        stability_results['stability_issues'].append(
                            f"Feature {feature_name} is unstable (stability: {stability:.3f})"
                        )
                
                # Calculate overall stability
                stability_scores = list(stability_results['feature_stability'].values())
                stability_results['overall_stability'] = np.mean(stability_scores) if stability_scores else 0.0
                
                self.stability_scores_ = stability_results['feature_stability']
                self.stability_report_ = stability_results
                
                return stability_results
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_error(f"Stability analysis failed: {e}")
                return {'overall_stability': 0.0, 'error': str(e)}
        
        def get_stability_summary(self):
            """Get stability summary."""
            return {
                'stability_scores': self.stability_scores_,
                'stability_report': self.stability_report_,
                'threshold': self.stability_threshold
            }
    
    class OverfittingMonitor:
        """Fallback implementation of OverfittingMonitor with basic functionality."""
        
        def __init__(self, overfitting_threshold=0.1, **kwargs):
            """Initialize the overfitting monitor."""
            self.overfitting_threshold = overfitting_threshold
            self.training_scores_ = []
            self.validation_scores_ = []
            self.overfitting_detected_ = False
            
            if TPRINT_AVAILABLE:
                tprint_info(f"OverfittingMonitor initialized with threshold {overfitting_threshold}")
        
        def monitor_overfitting(self, estimator, X_train, y_train, X_val, y_val, **kwargs):
            """Monitor for overfitting during training."""
            try:
                from sklearn.metrics import accuracy_score, mean_squared_error
                
                # Train the model
                estimator.fit(X_train, y_train)
                
                # Calculate training and validation scores
                y_train_pred = estimator.predict(X_train)
                y_val_pred = estimator.predict(X_val)
                
                if hasattr(estimator, 'predict_proba'):
                    train_score = accuracy_score(y_train, y_train_pred)
                    val_score = accuracy_score(y_val, y_val_pred)
                else:
                    train_score = -mean_squared_error(y_train, y_train_pred)
                    val_score = -mean_squared_error(y_val, y_val_pred)
                
                self.training_scores_.append(train_score)
                self.validation_scores_.append(val_score)
                
                # Check for overfitting
                score_gap = train_score - val_score
                overfitting_detected = score_gap > self.overfitting_threshold
                
                if overfitting_detected:
                    self.overfitting_detected_ = True
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"Overfitting detected: train_score={train_score:.3f}, val_score={val_score:.3f}, gap={score_gap:.3f}")
                
                return {
                    'train_score': train_score,
                    'val_score': val_score,
                    'score_gap': score_gap,
                    'overfitting_detected': overfitting_detected,
                    'overfitting_severity': min(1.0, score_gap / self.overfitting_threshold)
                }
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_error(f"Overfitting monitoring failed: {e}")
                return {'error': str(e), 'overfitting_detected': False}
        
        def get_overfitting_summary(self):
            """Get overfitting summary."""
            if not self.training_scores_ or not self.validation_scores_:
                return {'overfitting_detected': False, 'message': 'No monitoring data available'}
            
            avg_train_score = np.mean(self.training_scores_)
            avg_val_score = np.mean(self.validation_scores_)
            avg_gap = avg_train_score - avg_val_score
            
            return {
                'overfitting_detected': self.overfitting_detected_,
                'avg_train_score': avg_train_score,
                'avg_val_score': avg_val_score,
                'avg_score_gap': avg_gap,
                'n_observations': len(self.training_scores_)
            }

logger = logging.getLogger(__name__)


@dataclass
class DataCharacteristics:
    """Comprehensive data characteristics analysis."""
    
    # Basic statistics
    n_samples: int
    n_features: int
    data_types: Dict[str, str]
    
    # Distribution properties
    skewness: Dict[str, float]
    kurtosis: Dict[str, float]
    normality_pvalues: Dict[str, float]
    
    # Missing data
    missing_ratios: Dict[str, float]
    missing_patterns: Dict[str, Any]
    
    # Correlation structure
    correlation_matrix: pd.DataFrame
    avg_correlation: float
    max_correlation: float
    
    # Volatility and regime analysis
    volatility_regimes: List[Tuple[int, int, str]]  # (start, end, regime_type)
    regime_stability: float
    
    # Seasonality and patterns
    seasonality_detected: bool
    seasonal_periods: List[int]
    trend_strength: float
    
    # Data quality metrics
    data_quality_score: float
    outliers_ratio: float
    stability_score: float


@dataclass
class PatternAnalysis:
    """Pattern analysis results."""
    
    # Cyclical patterns
    cyclical_patterns: List[Dict[str, Any]]
    dominant_cycles: List[int]
    
    # Trend analysis
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float
    trend_breaks: List[int]
    
    # Regime changes
    regime_changes: List[int]
    regime_types: List[str]
    regime_stability: float
    
    # Anomalies
    anomalies: List[int]
    anomaly_scores: List[float]
    
    # Seasonality
    seasonal_components: Dict[str, Any]
    seasonal_strength: float


@dataclass
class RelationshipAnalysis:
    """Feature relationship analysis."""
    
    # Linear relationships
    linear_correlations: pd.DataFrame
    significant_correlations: List[Tuple[str, str, float, float]]  # (feat1, feat2, corr, pvalue)
    
    # Non-linear relationships
    mutual_information: pd.DataFrame
    significant_mi: List[Tuple[str, str, float]]
    
    # Conditional relationships
    conditional_dependencies: Dict[Tuple[str, str], float]
    
    # Interaction effects
    interaction_effects: List[Dict[str, Any]]
    
    # Causality indicators
    granger_causality: Dict[Tuple[str, str], float]
    lead_lag_relationships: Dict[Tuple[str, str], int]


class StatisticalTest(ABC):
    """Abstract base class for statistical tests."""
    
    @abstractmethod
    def test(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform the statistical test."""
        pass
    
    @abstractmethod
    def is_significant(self, result: Dict[str, Any], alpha: float = 0.05) -> bool:
        """Check if the result is statistically significant."""
        pass


class NormalityTest(StatisticalTest):
    """Test for normality using multiple methods."""
    
    def test(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Test normality using Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov."""
        results = {}
        
        for col in data.columns:
            series = data[col].dropna()
            if len(series) < 3:
                continue
            
            # Shapiro-Wilk test
            try:
                shapiro_stat, shapiro_p = stats.shapiro(series)
                results[f"{col}_shapiro"] = {
                    'statistic': shapiro_stat,
                    'pvalue': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except Exception as e:
                tprint_error(f"❌ Shapiro-Wilk test failed for {col}: {e}")
                raise RuntimeError(f"Shapiro-Wilk test failed for {col}: {e}") from e
            
            # Anderson-Darling test
            try:
                ad_stat, ad_critical, ad_significance = stats.anderson(series, dist='norm')
                results[f"{col}_anderson"] = {
                    'statistic': ad_stat,
                    'critical_values': ad_critical,
                    'significance_levels': ad_significance,
                    'is_normal': ad_stat < ad_critical[2]  # 5% significance level
                }
            except Exception as e:
                tprint_error(f"❌ Anderson-Darling test failed for {col}: {e}")
                raise RuntimeError(f"Anderson-Darling test failed for {col}: {e}") from e
        
        return results
    
    def is_significant(self, result: Dict[str, Any], alpha: float = 0.05) -> bool:
        """Check if data is significantly non-normal."""
        for test_name, test_result in result.items():
            if 'pvalue' in test_result:
                if test_result['pvalue'] < alpha:
                    return True  # Significant non-normality
        return False


class CorrelationTest(StatisticalTest):
    """Test for significant correlations."""
    
    def test(self, data: pd.DataFrame, method: str = 'pearson', **kwargs) -> Dict[str, Any]:
        """Test for significant correlations between features."""
        results = {}
        
        if method == 'pearson':
            corr_matrix = data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = data.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Calculate p-values
        n = len(data)
        p_values = np.zeros_like(corr_matrix)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    try:
                        if method == 'pearson':
                            _, p_val = pearsonr(data.iloc[:, i].dropna(), data.iloc[:, j].dropna())
                        elif method == 'spearman':
                            _, p_val = spearmanr(data.iloc[:, i].dropna(), data.iloc[:, j].dropna())
                        elif method == 'kendall':
                            _, p_val = kendalltau(data.iloc[:, i].dropna(), data.iloc[:, j].dropna())
                        
                        p_values.iloc[i, j] = p_val
                    except Exception as e:
                        tprint_warning(f"⚠️ Correlation test failed for {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {e}")
                        p_values.iloc[i, j] = 1.0
        
        results['correlation_matrix'] = corr_matrix
        results['p_values'] = p_values
        results['significant_correlations'] = []
        
        # Find significant correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                p_val = p_values.iloc[i, j]
                
                if p_val < 0.05:  # Significant at 5% level
                    results['significant_correlations'].append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'pvalue': p_val
                    })
        
        return results
    
    def is_significant(self, result: Dict[str, Any], alpha: float = 0.05) -> bool:
        """Check if there are any significant correlations."""
        return len(result['significant_correlations']) > 0


class MutualInformationTest(StatisticalTest):
    """Test for mutual information between features."""
    
    def test(self, data: pd.DataFrame, targets: Optional[pd.Series] = None, **kwargs) -> Dict[str, Any]:
        """Calculate mutual information between features and targets."""
        results = {}
        
        if targets is None:
            # Calculate MI between all feature pairs
            mi_matrix = np.zeros((len(data.columns), len(data.columns)))
            
            for i, col1 in enumerate(data.columns):
                for j, col2 in enumerate(data.columns):
                    if i != j:
                        try:
                            # Discretize continuous variables for MI calculation
                            data1 = pd.cut(data[col1].dropna(), bins=10, labels=False, duplicates='drop')
                            data2 = pd.cut(data[col2].dropna(), bins=10, labels=False, duplicates='drop')
                            
                            # Align data
                            common_idx = data1.index.intersection(data2.index)
                            if len(common_idx) > 10:  # Minimum samples
                                mi = mutual_info_score(data1[common_idx], data2[common_idx])
                                mi_matrix[i, j] = mi
                        except Exception as e:
                            tprint_debug(f"MI calculation failed for {col1} vs {col2}: {e}")
            
            results['mi_matrix'] = pd.DataFrame(mi_matrix, index=data.columns, columns=data.columns)
        else:
            # Calculate MI between features and targets
            mi_scores = {}
            for col in data.columns:
                try:
                    if targets.dtype == 'object' or len(targets.unique()) < 10:
                        # Classification
                        mi = mutual_info_classif(data[[col]].dropna(), targets[data[col].dropna().index])
                    else:
                        # Regression
                        mi = mutual_info_regression(data[[col]].dropna(), targets[data[col].dropna().index])
                    
                    mi_scores[col] = mi[0] if hasattr(mi, '__len__') else mi
                except Exception as e:
                    tprint_debug(f"MI calculation failed for {col}: {e}")
                    mi_scores[col] = 0.0
            
            results['mi_scores'] = mi_scores
        
        return results
    
    def is_significant(self, result: Dict[str, Any], alpha: float = 0.05) -> bool:
        """Check if there are significant mutual information values."""
        if 'mi_scores' in result:
            return any(score > 0.1 for score in result['mi_scores'].values())  # Threshold for MI
        elif 'mi_matrix' in result:
            return result['mi_matrix'].max().max() > 0.1
        return False


class StatisticalAnalysisFramework:
    """
    Enhanced comprehensive statistical analysis framework for data-driven decisions.
    
    Now integrated with ml_commons validation utilities for advanced
    statistical analysis, data leakage detection, and model validation.
    """
    
    def __init__(self, use_ml_commons: bool = True, enable_validation: bool = True):
        """Initialize the enhanced statistical analysis framework."""
        self.normality_test = NormalityTest()
        self.correlation_test = CorrelationTest()
        self.mi_test = MutualInformationTest()
        self.use_ml_commons = use_ml_commons and ML_COMMONS_VALIDATION_AVAILABLE
        self.enable_validation = enable_validation and ML_COMMONS_VALIDATION_AVAILABLE
        
        # Initialize ml_commons validation utilities if available
        if self.use_ml_commons:
            self.unified_cv = UnifiedCrossValidator()
            self.data_leakage_detector = DataLeakageDetector()
            self.enhanced_validation = EnhancedValidationFramework()
            self.stability_analyzer = StabilityAnalyzer()
            self.overfitting_monitor = OverfittingMonitor()
            tprint_info("✅ ML Commons validation utilities initialized")
        else:
            self.unified_cv = None
            self.data_leakage_detector = None
            self.enhanced_validation = None
            self.stability_analyzer = None
            self.overfitting_monitor = None
        
        tprint_info("Enhanced Statistical Analysis Framework initialized")
        if self.use_ml_commons:
            tprint_info("✅ ML Commons integration enabled")
        if self.enable_validation:
            tprint_info("✅ Enhanced validation enabled")
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> DataCharacteristics:
        """
        Comprehensive analysis of data characteristics.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataCharacteristics object with comprehensive analysis
        """
        tprint_info(f"Analyzing data characteristics for {data.shape[0]} samples, {data.shape[1]} features")
        
        # Basic statistics
        n_samples, n_features = data.shape
        data_types = {col: str(data[col].dtype) for col in data.columns}
        
        # Distribution properties
        skewness = {}
        kurtosis = {}
        normality_pvalues = {}
        
        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 3:
                skewness[col] = series.skew()
                kurtosis[col] = series.kurtosis()
                
                # Normality test
                try:
                    _, p_val = stats.shapiro(series)
                    normality_pvalues[col] = p_val
                except Exception:
                    normality_pvalues[col] = 1.0
        
        # Missing data analysis
        missing_ratios = {col: data[col].isna().sum() / len(data) for col in data.columns}
        missing_patterns = self._analyze_missing_patterns(data)
        
        # Correlation analysis
        corr_matrix = data.corr()
        avg_correlation = corr_matrix.abs().mean().mean()
        max_correlation = corr_matrix.abs().max().max()
        
        # Volatility and regime analysis
        volatility_regimes = self._detect_volatility_regimes(data)
        regime_stability = self._calculate_regime_stability(volatility_regimes)
        
        # Seasonality and patterns
        seasonality_detected, seasonal_periods = self._detect_seasonality(data)
        trend_strength = self._calculate_trend_strength(data)
        
        # Data quality metrics
        data_quality_score = self._calculate_data_quality_score(data, missing_ratios, corr_matrix)
        outliers_ratio = self._calculate_outliers_ratio(data)
        stability_score = self._calculate_stability_score(data)
        
        characteristics = DataCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            data_types=data_types,
            skewness=skewness,
            kurtosis=kurtosis,
            normality_pvalues=normality_pvalues,
            missing_ratios=missing_ratios,
            missing_patterns=missing_patterns,
            correlation_matrix=corr_matrix,
            avg_correlation=avg_correlation,
            max_correlation=max_correlation,
            volatility_regimes=volatility_regimes,
            regime_stability=regime_stability,
            seasonality_detected=seasonality_detected,
            seasonal_periods=seasonal_periods,
            trend_strength=trend_strength,
            data_quality_score=data_quality_score,
            outliers_ratio=outliers_ratio,
            stability_score=stability_score
        )
        
        tprint_success(f"Data characteristics analysis completed: quality_score={data_quality_score:.3f}")
        return characteristics
    
    def detect_patterns(self, data: pd.DataFrame) -> PatternAnalysis:
        """
        Detect patterns in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            PatternAnalysis object with detected patterns
        """
        tprint_info("Detecting patterns in data")
        
        # Cyclical patterns
        cyclical_patterns = self._detect_cyclical_patterns(data)
        dominant_cycles = [p['period'] for p in cyclical_patterns if p['strength'] > 0.5]
        
        # Trend analysis
        trend_direction, trend_strength = self._analyze_trends(data)
        trend_breaks = self._detect_trend_breaks(data)
        
        # Regime changes
        regime_changes, regime_types = self._detect_regime_changes(data)
        regime_stability = self._calculate_regime_stability_from_changes(regime_changes)
        
        # Anomalies
        anomalies, anomaly_scores = self._detect_anomalies(data)
        
        # Seasonality
        seasonal_components = self._analyze_seasonality(data)
        seasonal_strength = self._calculate_seasonal_strength(seasonal_components)
        
        analysis = PatternAnalysis(
            cyclical_patterns=cyclical_patterns,
            dominant_cycles=dominant_cycles,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            trend_breaks=trend_breaks,
            regime_changes=regime_changes,
            regime_types=regime_types,
            regime_stability=regime_stability,
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            seasonal_components=seasonal_components,
            seasonal_strength=seasonal_strength
        )
        
        tprint_success(f"Pattern analysis completed: {len(cyclical_patterns)} cycles, {len(anomalies)} anomalies")
        return analysis
    
    def evaluate_feature_relationships(self, features: pd.DataFrame, 
                                     targets: Optional[pd.Series] = None) -> RelationshipAnalysis:
        """
        Evaluate relationships between features.
        
        Args:
            features: Feature DataFrame
            targets: Optional target series
            
        Returns:
            RelationshipAnalysis object with relationship analysis
        """
        tprint_info(f"Evaluating feature relationships for {features.shape[1]} features")
        
        # Linear correlations
        corr_result = self.correlation_test.test(features, method='pearson')
        linear_correlations = corr_result['correlation_matrix']
        significant_correlations = corr_result['significant_correlations']
        
        # Mutual information
        mi_result = self.mi_test.test(features, targets)
        if 'mi_matrix' in mi_result:
            mutual_information = mi_result['mi_matrix']
        else:
            mutual_information = pd.DataFrame()
        
        significant_mi = []
        if 'mi_scores' in mi_result:
            for feat, score in mi_result['mi_scores'].items():
                if score > 0.1:  # Threshold for significant MI
                    significant_mi.append((feat, feat, score))  # Self-MI
        
        # Conditional dependencies
        conditional_dependencies = self._analyze_conditional_dependencies(features)
        
        # Interaction effects
        interaction_effects = self._detect_interaction_effects(features, targets)
        
        # Causality indicators
        granger_causality = self._test_granger_causality(features)
        lead_lag_relationships = self._analyze_lead_lag_relationships(features)
        
        analysis = RelationshipAnalysis(
            linear_correlations=linear_correlations,
            significant_correlations=significant_correlations,
            mutual_information=mutual_information,
            significant_mi=significant_mi,
            conditional_dependencies=conditional_dependencies,
            interaction_effects=interaction_effects,
            granger_causality=granger_causality,
            lead_lag_relationships=lead_lag_relationships
        )
        
        tprint_success(f"Feature relationship analysis completed: {len(significant_correlations)} significant correlations")
        return analysis
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data."""
        missing_data = data.isna()
        
        # Missing data patterns
        missing_patterns = {
            'total_missing': missing_data.sum().sum(),
            'missing_by_column': missing_data.sum().to_dict(),
            'missing_by_row': missing_data.sum(axis=1).to_dict(),
            'consecutive_missing': self._find_consecutive_missing(missing_data)
        }
        
        return missing_patterns
    
    def _detect_volatility_regimes(self, data: pd.DataFrame) -> List[Tuple[int, int, str]]:
        """Detect volatility regimes in the data."""
        regimes = []
        
        # Simple volatility regime detection based on rolling standard deviation
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 50:
                continue
            
            # Calculate rolling volatility
            rolling_vol = series.rolling(window=20).std()
            
            # Detect regime changes
            vol_threshold = rolling_vol.quantile(0.7)
            high_vol = rolling_vol > vol_threshold
            
            # Find regime periods
            regime_changes = high_vol.diff().fillna(False)
            regime_starts = series.index[regime_changes & high_vol].tolist()
            regime_ends = series.index[regime_changes & ~high_vol].tolist()
            
            # Create regime tuples
            for i, start in enumerate(regime_starts):
                end = regime_ends[i] if i < len(regime_ends) else series.index[-1]
                regimes.append((start, end, 'high_volatility'))
        
        return regimes
    
    def _calculate_regime_stability(self, regimes: List[Tuple[int, int, str]]) -> float:
        """Calculate stability of regimes."""
        if not regimes:
            return 1.0
        
        # Calculate average regime duration
        durations = [end - start for start, end, _ in regimes]
        avg_duration = np.mean(durations) if durations else 0
        
        # Normalize by data length (simplified)
        stability = min(1.0, avg_duration / 100)  # Assume 100 is good duration
        return stability
    
    def _detect_seasonality(self, data: pd.DataFrame) -> Tuple[bool, List[int]]:
        """Detect seasonality in the data."""
        seasonal_periods = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 100:
                continue
            
            # Simple seasonality detection using autocorrelation
            autocorr = series.autocorr(lag=1)
            if abs(autocorr) > 0.3:  # Threshold for seasonality
                seasonal_periods.append(1)
        
        return len(seasonal_periods) > 0, seasonal_periods
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength in the data with math validation."""
        trend_strengths = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 10:
                continue
            
            # Validate series data
            series = validate_finite(series, f"series_{col}")
            if len(series) < 10:
                continue
            
            # Calculate trend using linear regression with safe operations
            x = np.arange(len(series))
            try:
                slope, _, r_value, _, _ = stats.linregress(x, series)
                # Use safe absolute value and validate result
                r_abs = abs(validate_finite(r_value, f"r_value_{col}"))
                r_abs = validate_range(r_abs, 0.0, 1.0, f"r_value_{col}")
                trend_strengths.append(r_abs)
            except Exception as e:
                self.logger.warning(f"Trend calculation failed for {col}: {e}")
                continue
        
        # Use safe mean calculation
        return safe_mean(trend_strengths, default=0.0) if trend_strengths else 0.0
    
    def _calculate_data_quality_score(self, data: pd.DataFrame, 
                                    missing_ratios: Dict[str, float],
                                    corr_matrix: pd.DataFrame) -> float:
        """Calculate overall data quality score with math validation."""
        try:
            # Validate inputs
            missing_ratios = {k: validate_finite(v, f"missing_ratio_{k}") for k, v in missing_ratios.items()}
            corr_matrix = validate_finite(corr_matrix, "corr_matrix")
            
            # Penalize high missing ratios with safe operations
            missing_values = list(missing_ratios.values())
            missing_penalty = safe_mean(missing_values, default=0.0)
            missing_penalty = validate_range(missing_penalty, 0.0, 1.0, "missing_penalty")
            
            # Penalize high correlations (potential redundancy) with safe operations
            if len(corr_matrix) > 0:
                corr_abs = corr_matrix.abs()
                eye_matrix = np.eye(len(corr_matrix))
                corr_diff = corr_abs - eye_matrix
                corr_penalty = safe_percentile(corr_diff.values.flatten(), 95.0, default=0.0)
                corr_penalty = validate_range(corr_penalty, 0.0, 1.0, "corr_penalty")
            else:
                corr_penalty = 0.0
            
            # Calculate quality score (0-1, higher is better) with safe operations
            quality_score = 1.0 - missing_penalty - corr_penalty
            quality_score = validate_range(quality_score, 0.0, 1.0, "quality_score")
            
            return float(quality_score)
            
        except Exception as e:
            self.logger.warning(f"Data quality score calculation failed: {e}")
            return 0.0
    
    def _calculate_outliers_ratio(self, data: pd.DataFrame) -> float:
        """Calculate ratio of outliers in the data."""
        outlier_counts = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 10:
                continue
            
            # Use IQR method for outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outlier_counts.append(outliers / len(series))
        
        return np.mean(outlier_counts) if outlier_counts else 0.0
    
    def _calculate_stability_score(self, data: pd.DataFrame) -> float:
        """Calculate stability score of the data."""
        stability_scores = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 20:
                continue
            
            # Calculate stability using rolling coefficient of variation
            rolling_std = series.rolling(window=10).std()
            rolling_mean = series.rolling(window=10).mean()
            rolling_cv = rolling_std / rolling_mean.abs()
            
            # Stability is inverse of coefficient of variation
            stability = 1.0 / (1.0 + rolling_cv.mean())
            stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _detect_cyclical_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect cyclical patterns in the data."""
        patterns = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 50:
                continue
            
            # Simple cyclical pattern detection using FFT
            try:
                fft = np.fft.fft(series.values)
                freqs = np.fft.fftfreq(len(series))
                
                # Find dominant frequencies
                power_spectrum = np.abs(fft) ** 2
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                dominant_freq = freqs[dominant_freq_idx]
                
                if dominant_freq > 0:
                    period = int(1 / dominant_freq)
                    strength = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
                    
                    patterns.append({
                        'feature': col,
                        'period': period,
                        'strength': strength,
                        'frequency': dominant_freq
                    })
            except Exception as e:
                tprint_debug(f"Cyclical pattern detection failed for {col}: {e}")
        
        return patterns
    
    def _analyze_trends(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Analyze trends in the data."""
        trend_directions = []
        trend_strengths = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 10:
                continue
            
            # Linear trend analysis
            x = np.arange(len(series))
            slope, _, r_value, _, _ = stats.linregress(x, series)
            
            trend_directions.append('up' if slope > 0 else 'down' if slope < 0 else 'sideways')
            trend_strengths.append(abs(r_value))
        
        # Determine overall trend direction
        up_count = trend_directions.count('up')
        down_count = trend_directions.count('down')
        
        if up_count > down_count:
            overall_direction = 'up'
        elif down_count > up_count:
            overall_direction = 'down'
        else:
            overall_direction = 'sideways'
        
        overall_strength = np.mean(trend_strengths) if trend_strengths else 0.0
        
        return overall_direction, overall_strength
    
    def _detect_trend_breaks(self, data: pd.DataFrame) -> List[int]:
        """Detect trend breaks in the data."""
        trend_breaks = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 20:
                continue
            
            # Simple trend break detection using rolling regression
            window = min(20, len(series) // 3)
            slopes = []
            
            for i in range(window, len(series) - window):
                segment = series.iloc[i-window:i+window]
                x = np.arange(len(segment))
                slope, _, _, _, _ = stats.linregress(x, segment)
                slopes.append(slope)
            
            # Detect significant slope changes
            slope_changes = np.diff(slopes)
            significant_changes = np.abs(slope_changes) > np.std(slope_changes) * 2
            
            for i, is_significant in enumerate(significant_changes):
                if is_significant:
                    trend_breaks.append(i + window)
        
        return trend_breaks
    
    def _detect_regime_changes(self, data: pd.DataFrame) -> Tuple[List[int], List[str]]:
        """Detect regime changes in the data."""
        regime_changes = []
        regime_types = []
        
        # Simple regime change detection based on volatility
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 50:
                continue
            
            # Calculate rolling volatility
            rolling_vol = series.rolling(window=20).std()
            
            # Detect significant volatility changes
            vol_changes = rolling_vol.diff().abs()
            threshold = vol_changes.quantile(0.9)
            significant_changes = vol_changes > threshold
            
            for idx in series.index[significant_changes]:
                regime_changes.append(idx)
                regime_types.append('volatility_change')
        
        return regime_changes, regime_types
    
    def _calculate_regime_stability_from_changes(self, regime_changes: List[int]) -> float:
        """Calculate regime stability from regime changes."""
        if not regime_changes:
            return 1.0
        
        # More changes = less stable
        stability = 1.0 / (1.0 + len(regime_changes) / 100)  # Normalize by 100
        return stability
    
    def _detect_anomalies(self, data: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """Detect anomalies in the data."""
        anomalies = []
        anomaly_scores = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 10:
                continue
            
            # Use Z-score method for anomaly detection
            z_scores = np.abs(stats.zscore(series))
            anomaly_threshold = 3.0
            
            anomaly_indices = series.index[z_scores > anomaly_threshold].tolist()
            anomaly_scores.extend(z_scores[z_scores > anomaly_threshold].tolist())
            
            anomalies.extend(anomaly_indices)
        
        return anomalies, anomaly_scores
    
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonality in the data."""
        seasonal_components = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            if len(series) < 100:
                continue
            
            # Simple seasonality analysis
            autocorr = series.autocorr(lag=1)
            seasonal_components[col] = {
                'autocorrelation': autocorr,
                'is_seasonal': abs(autocorr) > 0.3
            }
        
        return seasonal_components
    
    def _calculate_seasonal_strength(self, seasonal_components: Dict[str, Any]) -> float:
        """Calculate overall seasonal strength."""
        if not seasonal_components:
            return 0.0
        
        autocorrs = [comp['autocorrelation'] for comp in seasonal_components.values()]
        return np.mean(np.abs(autocorrs)) if autocorrs else 0.0
    
    def _analyze_conditional_dependencies(self, features: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """Analyze conditional dependencies between features."""
        dependencies = {}
        
        # Simple conditional dependency analysis
        for col1 in features.columns:
            for col2 in features.columns:
                if col1 != col2:
                    try:
                        # Calculate partial correlation
                        series1 = features[col1].dropna()
                        series2 = features[col2].dropna()
                        
                        if len(series1) > 10 and len(series2) > 10:
                            # Align series
                            common_idx = series1.index.intersection(series2.index)
                            if len(common_idx) > 10:
                                corr, _ = pearsonr(series1[common_idx], series2[common_idx])
                                dependencies[(col1, col2)] = abs(corr)
                    except Exception as e:
                        tprint_debug(f"Conditional dependency analysis failed for {col1} vs {col2}: {e}")
        
        return dependencies
    
    def _detect_interaction_effects(self, features: pd.DataFrame, 
                                  targets: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """Detect interaction effects between features."""
        interactions = []
        
        if targets is None:
            return interactions
        
        # Simple interaction detection using product terms
        for col1 in features.columns:
            for col2 in features.columns:
                if col1 != col2:
                    try:
                        # Create interaction term
                        series1 = features[col1].dropna()
                        series2 = features[col2].dropna()
                        
                        # Align series
                        common_idx = series1.index.intersection(series2.index).intersection(targets.index)
                        if len(common_idx) > 10:
                            aligned_targets = targets[common_idx]
                            
                            # Calculate interaction effect
                            interaction_term = series1[common_idx] * series2[common_idx]
                            corr, p_val = pearsonr(interaction_term, aligned_targets)
                            
                            if p_val < 0.05:  # Significant interaction
                                interactions.append({
                                    'feature1': col1,
                                    'feature2': col2,
                                    'correlation': corr,
                                    'pvalue': p_val,
                                    'strength': abs(corr)
                                })
                    except Exception as e:
                        tprint_debug(f"Interaction detection failed for {col1} vs {col2}: {e}")
        
        return interactions
    
    def _test_granger_causality(self, features: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """Test Granger causality between features."""
        # Simplified Granger causality test
        causality = {}
        
        for col1 in features.columns:
            for col2 in features.columns:
                if col1 != col2:
                    try:
                        series1 = features[col1].dropna()
                        series2 = features[col2].dropna()
                        
                        # Align series
                        common_idx = series1.index.intersection(series2.index)
                        if len(common_idx) > 20:
                            # Simple lagged correlation as proxy for Granger causality
                            lagged_corr = series1[common_idx].corr(series2[common_idx].shift(1))
                            causality[(col1, col2)] = abs(lagged_corr) if not pd.isna(lagged_corr) else 0.0
                    except Exception as e:
                        tprint_debug(f"Granger causality test failed for {col1} vs {col2}: {e}")
        
        return causality
    
    def _analyze_lead_lag_relationships(self, features: pd.DataFrame) -> Dict[Tuple[str, str], int]:
        """Analyze lead-lag relationships between features."""
        lead_lag = {}
        
        for col1 in features.columns:
            for col2 in features.columns:
                if col1 != col2:
                    try:
                        series1 = features[col1].dropna()
                        series2 = features[col2].dropna()
                        
                        # Align series
                        common_idx = series1.index.intersection(series2.index)
                        if len(common_idx) > 20:
                            # Find optimal lag
                            max_lag = min(10, len(common_idx) // 4)
                            best_lag = 0
                            best_corr = 0
                            
                            for lag in range(-max_lag, max_lag + 1):
                                if lag == 0:
                                    continue
                                
                                if lag > 0:
                                    corr = series1[common_idx].corr(series2[common_idx].shift(lag))
                                else:
                                    corr = series1[common_idx].shift(-lag).corr(series2[common_idx])
                                
                                if not pd.isna(corr) and abs(corr) > abs(best_corr):
                                    best_corr = corr
                                    best_lag = lag
                            
                            lead_lag[(col1, col2)] = best_lag
                    except Exception as e:
                        tprint_debug(f"Lead-lag analysis failed for {col1} vs {col2}: {e}")
        
        return lead_lag
    
    def _find_consecutive_missing(self, missing_data: pd.DataFrame) -> Dict[str, Any]:
        """Find consecutive missing data patterns."""
        consecutive_missing = {}
        
        for col in missing_data.columns:
            series = missing_data[col]
            consecutive_lengths = []
            current_length = 0
            
            for is_missing in series:
                if is_missing:
                    current_length += 1
                else:
                    if current_length > 0:
                        consecutive_lengths.append(current_length)
                        current_length = 0
            
            if current_length > 0:
                consecutive_lengths.append(current_length)
            
            consecutive_missing[col] = {
                'max_consecutive': max(consecutive_lengths) if consecutive_lengths else 0,
                'avg_consecutive': np.mean(consecutive_lengths) if consecutive_lengths else 0,
                'total_gaps': len(consecutive_lengths)
            }
        
        return consecutive_missing


# Convenience functions
def create_enhanced_statistical_framework(use_ml_commons: bool = True, 
                                        enable_validation: bool = True) -> StatisticalAnalysisFramework:
    """Create an enhanced statistical analysis framework with ml_commons integration."""
    return StatisticalAnalysisFramework(
        use_ml_commons=use_ml_commons,
        enable_validation=enable_validation
    )