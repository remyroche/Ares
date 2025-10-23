"""
Regime Feature Importance Analysis.

This module provides comprehensive feature importance analysis specifically
designed for market regime identification. It implements various methods to
determine which features are most relevant for distinguishing between different
market regimes and improving trading model performance.

Key Analysis Methods:
- Statistical importance (correlation, mutual information, ANOVA)
- Model-based importance (Random Forest, XGBoost, SHAP values)
- Regime-specific importance (per-regime feature analysis)
- Time-series importance (lag analysis, Granger causality)
- Ensemble importance (combining multiple methods)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
from abc import ABC, abstractmethod

from src.utils.logger import system_logger


class ImportanceMethod(Enum):
    """Enumeration of feature importance methods."""
    MUTUAL_INFORMATION = "mutual_information"
    CORRELATION = "correlation"
    ANOVA_F_TEST = "anova_f_test"
    CHI2_TEST = "chi2_test"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    PERMUTATION = "permutation"
    SHAP_VALUES = "shap_values"
    LASSO_COEFFICIENTS = "lasso_coefficients"
    RIDGE_COEFFICIENTS = "ridge_coefficients"
    GRANGER_CAUSALITY = "granger_causality"
    ENSEMBLE = "ensemble"


@dataclass
class ImportanceConfig:
    """Configuration for feature importance analysis."""
    # General parameters
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Statistical test parameters
    significance_level: float = 0.05
    
    # Model parameters
    rf_params: Dict[str, Any] = None
    xgb_params: Dict[str, Any] = None
    
    # SHAP parameters
    shap_sample_size: int = 1000
    shap_method: str = "tree"  # tree, linear, kernel
    
    # Permutation parameters
    permutation_scoring: str = "accuracy"
    permutation_n_repeats: int = 10
    
    # Ensemble parameters
    ensemble_methods: List[ImportanceMethod] = None
    ensemble_weights: Dict[ImportanceMethod, float] = None
    
    # Time series parameters
    max_lag: int = 10
    granger_test_lags: int = 5
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.rf_params is None:
            self.rf_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': self.random_state}
        if self.xgb_params is None:
            self.xgb_params = {'n_estimators': 100, 'max_depth': 6, 'random_state': self.random_state}
        if self.ensemble_methods is None:
            self.ensemble_methods = [
                ImportanceMethod.MUTUAL_INFORMATION,
                ImportanceMethod.RANDOM_FOREST,
                ImportanceMethod.PERMUTATION
            ]
        if self.ensemble_weights is None:
            self.ensemble_weights = {method: 1.0 for method in self.ensemble_methods}


@dataclass
class ImportanceResult:
    """Result container for feature importance analysis."""
    method: ImportanceMethod
    feature_names: List[str]
    importance_scores: np.ndarray
    rankings: np.ndarray
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method.value,
            'feature_names': self.feature_names,
            'importance_scores': self.importance_scores.tolist(),
            'rankings': self.rankings.tolist(),
            'metadata': self.metadata
        }
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by importance."""
        sorted_indices = np.argsort(self.importance_scores)[::-1]
        return [
            (self.feature_names[i], self.importance_scores[i])
            for i in sorted_indices[:n]
        ]


class BaseImportanceAnalyzer(ABC):
    """Abstract base class for importance analyzers."""
    
    def __init__(self, config: ImportanceConfig):
        self.config = config
        self.logger = system_logger.getChild(f'Importance.{self.__class__.__name__}')
    
    @abstractmethod
    def analyze(self, 
               features: pd.DataFrame,
               target: np.ndarray) -> ImportanceResult:
        """Analyze feature importance."""
        pass
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize importance scores to [0, 1] range."""
        scores = np.array(scores)
        scores = np.abs(scores)  # Take absolute values
        
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        
        return scores
    
    def _calculate_rankings(self, scores: np.ndarray) -> np.ndarray:
        """Calculate feature rankings based on importance scores."""
        return np.argsort(np.argsort(scores)[::-1]) + 1


class MutualInformationAnalyzer(BaseImportanceAnalyzer):
    """Mutual information based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        X = features.fillna(0).values
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        if is_classification:
            # Ensure target is properly encoded for classification
            le = LabelEncoder()
            y = le.fit_transform(target.to_numpy() if hasattr(target, 'to_numpy') else target.ravel())
            scores = mutual_info_classif(X, y, random_state=self.config.random_state)
        else:
            scores = mutual_info_regression(X, target, random_state=self.config.random_state)
        
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        return ImportanceResult(
            method=ImportanceMethod.MUTUAL_INFORMATION,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'is_classification': is_classification,
                'n_unique_targets': unique_targets
            }
        )


class CorrelationAnalyzer(BaseImportanceAnalyzer):
    """Correlation based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        # Calculate correlations
        correlations = []
        
        for col in features.columns:
            feature_data = features[col].fillna(0)
            corr = np.corrcoef(feature_data, target)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        scores = np.array(correlations)
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        return ImportanceResult(
            method=ImportanceMethod.CORRELATION,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'mean_correlation': float(np.mean(scores)),
                'max_correlation': float(np.max(scores))
            }
        )


class ANOVAAnalyzer(BaseImportanceAnalyzer):
    """ANOVA F-test based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        from sklearn.feature_selection import f_classif, f_regression
        
        # Prepare data
        X = features.fillna(0).values
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(target.to_numpy() if hasattr(target, 'to_numpy') else target.ravel())
            f_scores, p_values = f_classif(X, y)
        else:
            f_scores, p_values = f_regression(X, target)
        
        # Use F-scores as importance
        scores = self._normalize_scores(f_scores)
        rankings = self._calculate_rankings(scores)
        
        # Calculate number of significant features
        n_significant = np.sum(p_values < self.config.significance_level)
        
        return ImportanceResult(
            method=ImportanceMethod.ANOVA_F_TEST,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'p_values': p_values.tolist(),
                'n_significant': int(n_significant),
                'significance_level': self.config.significance_level,
                'is_classification': is_classification
            }
        )


class RandomForestAnalyzer(BaseImportanceAnalyzer):
    """Random Forest based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Prepare data
        X = features.fillna(0).values
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(target.to_numpy() if hasattr(target, 'to_numpy') else target.ravel())
            model = RandomForestClassifier(**self.config.rf_params)
        else:
            y = target
            model = RandomForestRegressor(**self.config.rf_params)
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances
        scores = model.feature_importances_
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        return ImportanceResult(
            method=ImportanceMethod.RANDOM_FOREST,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'model_score': float(model.score(X, y)),
                'n_estimators': self.config.rf_params.get('n_estimators', 100),
                'is_classification': is_classification
            }
        )


class XGBoostAnalyzer(BaseImportanceAnalyzer):
    """XGBoost based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        try:
            import xgboost as xgb
        except ImportError:
            self.logger.warning("XGBoost not available, falling back to Random Forest")
            return RandomForestAnalyzer(self.config).analyze(features, target)
        
        
        # Prepare data
        X = features.fillna(0).values
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(target.to_numpy() if hasattr(target, 'to_numpy') else target.ravel())
            model = xgb.XGBClassifier(**self.config.xgb_params)
        else:
            y = target
            model = xgb.XGBRegressor(**self.config.xgb_params)
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances (using 'weight' importance type)
        scores = model.feature_importances_
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        return ImportanceResult(
            method=ImportanceMethod.XGBOOST,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'model_score': float(model.score(X, y)),
                'n_estimators': self.config.xgb_params.get('n_estimators', 100),
                'is_classification': is_classification
            }
        )


class PermutationAnalyzer(BaseImportanceAnalyzer):
    """Permutation based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = features.fillna(0).values
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(target.to_numpy() if hasattr(target, 'to_numpy') else target.ravel())
            model = RandomForestClassifier(**self.config.rf_params)
        else:
            y = target
            model = RandomForestRegressor(**self.config.rf_params)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_val, y_val,
            n_repeats=self.config.permutation_n_repeats,
            random_state=self.config.random_state,
            scoring=self.config.permutation_scoring if is_classification else 'r2'
        )
        
        scores = perm_importance.importances_mean
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        return ImportanceResult(
            method=ImportanceMethod.PERMUTATION,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'importances_std': perm_importance.importances_std.tolist(),
                'validation_score': float(model.score(X_val, y_val)),
                'n_repeats': self.config.permutation_n_repeats,
                'is_classification': is_classification
            }
        )


class SHAPAnalyzer(BaseImportanceAnalyzer):
    """SHAP values based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        try:
            import shap
        except ImportError:
            self.logger.warning("SHAP not available, falling back to Permutation Importance")
            return PermutationAnalyzer(self.config).analyze(features, target)
        
        
        # Prepare data
        X = features.fillna(0).values
        
        # Sample data if too large
        if len(X) > self.config.shap_sample_size:
            indices = np.random.choice(len(X), self.config.shap_sample_size, replace=False)
            X_sample = X[indices]
            target_sample = target[indices]
        else:
            X_sample = X
            target_sample = target
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target_sample))
        is_classification = unique_targets < min(len(target_sample) * 0.1, 50)
        
        if is_classification:
            le = LabelEncoder()
            y_sample = le.fit_transform(target_sample.ravel())
            model = RandomForestClassifier(**self.config.rf_params)
        else:
            y_sample = target_sample
            model = RandomForestRegressor(**self.config.rf_params)
        
        # Fit model
        model.fit(X_sample, y_sample)
        
        # Calculate SHAP values
        if self.config.shap_method == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            shap_values = shap_values.values
        
        # Handle multi-class case
        if is_classification and len(shap_values.shape) == 3:
            # For multi-class, take mean absolute SHAP values across classes
            scores = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            # For binary classification or regression
            scores = np.mean(np.abs(shap_values), axis=0)
        
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        return ImportanceResult(
            method=ImportanceMethod.SHAP_VALUES,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'shap_method': self.config.shap_method,
                'sample_size': len(X_sample),
                'is_classification': is_classification,
                'model_score': float(model.score(X_sample, y_sample))
            }
        )


class LassoAnalyzer(BaseImportanceAnalyzer):
    """LASSO coefficients based feature importance."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
        
        # Prepare data
        scaler = StandardScaler()
        X = scaler.fit_transform(features.fillna(0).values)
        
        # Determine if classification or regression
        unique_targets = len(np.unique(target))
        is_classification = unique_targets < min(len(target) * 0.1, 50)
        
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(target.to_numpy() if hasattr(target, 'to_numpy') else target.ravel())
            # Use LogisticRegression with L1 penalty for classification
            model = LogisticRegression(penalty='l1', solver='liblinear', random_state=self.config.random_state)
        else:
            y = target
            # Use LassoCV for automatic alpha selection
            model = LassoCV(cv=self.config.cross_validation_folds, random_state=self.config.random_state)
        
        # Fit model
        model.fit(X, y)
        
        # Get coefficients
        if is_classification and hasattr(model, 'coef_') and len(model.coef_.shape) > 1:
            # Multi-class case
            scores = np.mean(np.abs(model.coef_), axis=0)
        else:
            scores = np.abs(model.coef_)
        
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        metadata = {
            'is_classification': is_classification,
            'model_score': float(model.score(X, y)),
            'n_nonzero_coef': int(np.sum(scores > 1e-6))
        }
        
        if not is_classification and hasattr(model, 'alpha_'):
            metadata['alpha'] = float(model.alpha_)
        
        return ImportanceResult(
            method=ImportanceMethod.LASSO_COEFFICIENTS,
            feature_names=list(features.columns),
            importance_scores=scores,
            rankings=rankings,
            metadata=metadata
        )


class GrangerCausalityAnalyzer(BaseImportanceAnalyzer):
    """Granger causality based feature importance for time series."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            self.logger.warning("statsmodels not available, falling back to correlation")
            return CorrelationAnalyzer(self.config).analyze(features, target)
        
        scores = []
        feature_names = list(features.columns)
        
        for col in feature_names:
            try:
                # Prepare data for Granger test
                feature_data = features[col].fillna(method='ffill').fillna(0)
                
                # Create time series data
                data = pd.DataFrame({
                    'target': target,
                    'feature': feature_data
                })
                
                # Remove any remaining NaN values
                data = data.dropna()
                
                if len(data) < self.config.granger_test_lags * 3:
                    scores.append(0.0)
                    continue
                
                # Perform Granger causality test
                result = grangercausalitytests(
                    data[['target', 'feature']], 
                    maxlag=self.config.granger_test_lags,
                    verbose=False
                )
                
                # Extract minimum p-value across all lags
                p_values = []
                for lag in range(1, self.config.granger_test_lags + 1):
                    if lag in result:
                        # Get F-test p-value
                        p_val = result[lag][0]['ssr_ftest'][1]
                        p_values.append(p_val)
                
                if p_values:
                    # Use 1 - min(p_value) as importance score
                    min_p_value = min(p_values)
                    importance = 1 - min_p_value
                else:
                    importance = 0.0
                
                scores.append(importance)
                
            except Exception as e:
                self.logger.warning(f"Granger test failed for {col}: {e}")
                scores.append(0.0)
        
        scores = np.array(scores)
        scores = self._normalize_scores(scores)
        rankings = self._calculate_rankings(scores)
        
        return ImportanceResult(
            method=ImportanceMethod.GRANGER_CAUSALITY,
            feature_names=feature_names,
            importance_scores=scores,
            rankings=rankings,
            metadata={
                'max_lag': self.config.granger_test_lags,
                'significance_level': self.config.significance_level
            }
        )


class EnsembleImportanceAnalyzer(BaseImportanceAnalyzer):
    """Ensemble feature importance combining multiple methods."""
    
    def analyze(self, features: pd.DataFrame, target: np.ndarray) -> ImportanceResult:
        # Initialize analyzers
        analyzers = {
            ImportanceMethod.MUTUAL_INFORMATION: MutualInformationAnalyzer(self.config),
            ImportanceMethod.CORRELATION: CorrelationAnalyzer(self.config),
            ImportanceMethod.ANOVA_F_TEST: ANOVAAnalyzer(self.config),
            ImportanceMethod.RANDOM_FOREST: RandomForestAnalyzer(self.config),
            ImportanceMethod.XGBOOST: XGBoostAnalyzer(self.config),
            ImportanceMethod.PERMUTATION: PermutationAnalyzer(self.config),
            ImportanceMethod.SHAP_VALUES: SHAPAnalyzer(self.config),
            ImportanceMethod.LASSO_COEFFICIENTS: LassoAnalyzer(self.config),
            ImportanceMethod.GRANGER_CAUSALITY: GrangerCausalityAnalyzer(self.config)
        }
        
        # Run individual methods
        individual_results = {}
        for method in self.config.ensemble_methods:
            if method in analyzers:
                try:
                    result = analyzers[method].analyze(features, target)
                    individual_results[method] = result
                    self.logger.info(f"✅ {method.value} completed")
                except Exception as e:
                    self.logger.error(f"❌ {method.value} failed: {e}")
                    continue
        
        if not individual_results:
            raise ValueError("No individual importance methods succeeded")
        
        # Combine scores using weighted average
        feature_names = list(features.columns)
        n_features = len(feature_names)
        combined_scores = np.zeros(n_features)
        total_weight = 0.0
        
        for method, result in individual_results.items():
            weight = self.config.ensemble_weights.get(method, 1.0)
            combined_scores += weight * result.importance_scores
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_scores /= total_weight
        
        # Calculate rankings
        rankings = self._calculate_rankings(combined_scores)
        
        # Calculate agreement between methods
        agreement_score = self._calculate_method_agreement(individual_results)
        
        return ImportanceResult(
            method=ImportanceMethod.ENSEMBLE,
            feature_names=feature_names,
            importance_scores=combined_scores,
            rankings=rankings,
            metadata={
                'individual_methods': [m.value for m in individual_results.keys()],
                'method_weights': {m.value: w for m, w in self.config.ensemble_weights.items()},
                'agreement_score': agreement_score,
                'individual_results': {m.value: r.to_dict() for m, r in individual_results.items()}
            }
        )
    
    def _calculate_method_agreement(self, results: Dict[ImportanceMethod, ImportanceResult]) -> float:
        """Calculate agreement between different importance methods."""
        if len(results) < 2:
            return 1.0
        
        # Calculate pairwise rank correlations
        methods = list(results.keys())
        correlations = []
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                rank1 = results[methods[i]].rankings
                rank2 = results[methods[j]].rankings
                
                # Calculate Spearman rank correlation
                corr = np.corrcoef(rank1, rank2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return float(np.mean(correlations)) if correlations else 0.0


class RegimeFeatureImportance:
    """
    Main feature importance analysis class for market regime research.
    
    This class provides comprehensive feature importance analysis specifically
    designed for market regime identification and trading model improvement.
    """
    
    def __init__(self, config: Optional[ImportanceConfig] = None):
        """
        Initialize the regime feature importance analyzer.
        
        Args:
            config: Configuration for importance analysis
        """
        self.config = config or ImportanceConfig()
        self.logger = system_logger.getChild('RegimeFeatureImportance')
        self.results: Dict[ImportanceMethod, ImportanceResult] = {}
        
        # Initialize analyzers
        self.analyzers = {
            ImportanceMethod.MUTUAL_INFORMATION: MutualInformationAnalyzer(self.config),
            ImportanceMethod.CORRELATION: CorrelationAnalyzer(self.config),
            ImportanceMethod.ANOVA_F_TEST: ANOVAAnalyzer(self.config),
            ImportanceMethod.RANDOM_FOREST: RandomForestAnalyzer(self.config),
            ImportanceMethod.XGBOOST: XGBoostAnalyzer(self.config),
            ImportanceMethod.PERMUTATION: PermutationAnalyzer(self.config),
            ImportanceMethod.SHAP_VALUES: SHAPAnalyzer(self.config),
            ImportanceMethod.LASSO_COEFFICIENTS: LassoAnalyzer(self.config),
            ImportanceMethod.GRANGER_CAUSALITY: GrangerCausalityAnalyzer(self.config),
            ImportanceMethod.ENSEMBLE: EnsembleImportanceAnalyzer(self.config)
        }
    
    def analyze_single_method(self,
                            features: pd.DataFrame,
                            target: np.ndarray,
                            method: ImportanceMethod) -> ImportanceResult:
        """
        Analyze feature importance using a single method.
        
        Args:
            features: Feature matrix
            target: Target variable (regime labels)
            method: Importance analysis method
            
        Returns:
            Feature importance result
        """
        self.logger.info(f"🔍 Analyzing feature importance using {method.value}")
        
        if method not in self.analyzers:
            raise ValueError(f"Method {method.value} not supported")
        
        result = self.analyzers[method].analyze(features, target)
        self.results[method] = result
        
        self.logger.info(f"✅ {method.value} completed: top feature = {result.get_top_features(1)[0][0]}")
        
        return result
    
    def analyze_all_methods(self,
                          features: pd.DataFrame,
                          target: np.ndarray) -> Dict[ImportanceMethod, ImportanceResult]:
        """
        Analyze feature importance using all available methods.
        
        Args:
            features: Feature matrix
            target: Target variable (regime labels)
            
        Returns:
            Dictionary mapping methods to results
        """
        self.logger.info("🚀 Running comprehensive feature importance analysis")
        
        results = {}
        
        # Run individual methods (exclude ensemble for now)
        individual_methods = [m for m in ImportanceMethod if m != ImportanceMethod.ENSEMBLE]
        
        for method in individual_methods:
            try:
                result = self.analyze_single_method(features, target, method)
                results[method] = result
            except Exception as e:
                self.logger.error(f"❌ {method.value} failed: {e}")
                continue
        
        # Run ensemble method if we have individual results
        if len(results) >= 2:
            try:
                ensemble_result = self.analyze_single_method(features, target, ImportanceMethod.ENSEMBLE)
                results[ImportanceMethod.ENSEMBLE] = ensemble_result
            except Exception as e:
                self.logger.error(f"❌ Ensemble importance failed: {e}")
        
        self.logger.info(f"✅ Completed {len(results)} importance methods")
        return results
    
    def analyze_regime_specific_importance(self,
                                         features: pd.DataFrame,
                                         regime_labels: np.ndarray,
                                         method: ImportanceMethod = ImportanceMethod.RANDOM_FOREST) -> Dict[int, ImportanceResult]:
        """
        Analyze feature importance for each regime separately.
        
        Args:
            features: Feature matrix
            regime_labels: Regime labels
            method: Importance analysis method
            
        Returns:
            Dictionary mapping regime IDs to importance results
        """
        self.logger.info(f"🎯 Analyzing regime-specific feature importance using {method.value}")
        
        unique_regimes = np.unique(regime_labels)
        regime_results = {}
        
        for regime in unique_regimes:
            regime_mask = (regime_labels == regime)
            regime_features = features[regime_mask]
            
            # Create binary target (this regime vs others)
            binary_target = (regime_labels == regime).astype(int)
            
            try:
                result = self.analyzers[method].analyze(regime_features, binary_target)
                regime_results[int(regime)] = result
                
                top_feature = result.get_top_features(1)[0][0]
                self.logger.info(f"   Regime {regime}: top feature = {top_feature}")
                
            except Exception as e:
                self.logger.error(f"❌ Regime {regime} analysis failed: {e}")
                continue
        
        return regime_results
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare feature importance methods.
        
        Returns:
            DataFrame with method comparison
        """
        if not self.results:
            self.logger.warning("No importance results available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for method, result in self.results.items():
            top_features = result.get_top_features(5)
            
            row = {
                'method': method.value,
                'top_feature': top_features[0][0] if top_features else 'N/A',
                'top_feature_score': top_features[0][1] if top_features else 0.0,
                'mean_importance': float(np.mean(result.importance_scores)),
                'std_importance': float(np.std(result.importance_scores)),
                'top_5_features': [f[0] for f in top_features]
            }
            
            # Add method-specific metadata
            if 'agreement_score' in result.metadata:
                row['ensemble_agreement'] = result.metadata['agreement_score']
            if 'model_score' in result.metadata:
                row['model_score'] = result.metadata['model_score']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('mean_importance', ascending=False) if not df.empty else df
    
    def get_consensus_features(self, n: int = 10, min_methods: int = 2) -> List[Tuple[str, float, int]]:
        """
        Get consensus important features across methods.
        
        Args:
            n: Number of top features to return
            min_methods: Minimum number of methods that must agree
            
        Returns:
            List of (feature_name, consensus_score, n_methods_agreeing)
        """
        if not self.results:
            return []
        
        # Count how many methods rank each feature in top N
        feature_votes = {}
        feature_scores = {}
        
        for method, result in self.results.items():
            top_features = result.get_top_features(n * 2)  # Consider more features
            
            for i, (feature_name, score) in enumerate(top_features):
                if feature_name not in feature_votes:
                    feature_votes[feature_name] = 0
                    feature_scores[feature_name] = []
                
                # Weight by position (higher weight for top features)
                weight = max(0, n - i) / n
                feature_votes[feature_name] += weight
                feature_scores[feature_name].append(score)
        
        # Calculate consensus scores
        consensus_features = []
        for feature_name, votes in feature_votes.items():
            if len(feature_scores[feature_name]) >= min_methods:
                consensus_score = votes * np.mean(feature_scores[feature_name])
                n_methods = len(feature_scores[feature_name])
                consensus_features.append((feature_name, consensus_score, n_methods))
        
        # Sort by consensus score
        consensus_features.sort(key=lambda x: x[1], reverse=True)
        
        return consensus_features[:n]
    
    def save_results(self, filepath: str):
        """Save importance analysis results to file."""
        results_dict = {
            method.value: result.to_dict() 
            for method, result in self.results.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"💾 Saved importance analysis results to {filepath}")
    
    def load_results(self, filepath: str):
        """Load importance analysis results from file."""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.results = {}
        for method_name, result_dict in results_dict.items():
            method = ImportanceMethod(method_name)
            
            # Reconstruct ImportanceResult
            result = ImportanceResult(
                method=method,
                feature_names=result_dict['feature_names'],
                importance_scores=np.array(result_dict['importance_scores']),
                rankings=np.array(result_dict['rankings']),
                metadata=result_dict['metadata']
            )
            
            self.results[method] = result
        
        self.logger.info(f"📂 Loaded importance analysis results from {filepath}")
    
    def generate_importance_report(self) -> str:
        """Generate a comprehensive feature importance report."""
        if not self.results:
            return "No importance analysis results available. Run analysis first."
        
        report = []
        report.append("# Feature Importance Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Method comparison
        comparison_df = self.compare_methods()
        if not comparison_df.empty:
            report.append("## Method Comparison")
            report.append("")
            
            for _, row in comparison_df.iterrows():
                report.append(f"**{row['method'].upper()}**")
                report.append(f"- Top Feature: {row['top_feature']} (score: {row['top_feature_score']:.3f})")
                report.append(f"- Mean Importance: {row['mean_importance']:.3f}")
                if 'model_score' in row:
                    report.append(f"- Model Score: {row['model_score']:.3f}")
                report.append("")
        
        # Consensus features
        consensus_features = self.get_consensus_features(10)
        if consensus_features:
            report.append("## Consensus Important Features")
            report.append("")
            
            for i, (feature, score, n_methods) in enumerate(consensus_features, 1):
                report.append(f"{i}. **{feature}**")
                report.append(f"   - Consensus Score: {score:.3f}")
                report.append(f"   - Agreed by {n_methods} methods")
                report.append("")
        
        # Detailed method results
        report.append("## Detailed Method Results")
        report.append("")
        
        for method, result in self.results.items():
            report.append(f"### {method.value.upper()}")
            
            top_features = result.get_top_features(10)
            report.append("**Top 10 Features:**")
            for i, (feature, score) in enumerate(top_features, 1):
                report.append(f"{i:2d}. {feature}: {score:.3f}")
            
            # Method-specific metadata
            if result.metadata:
                report.append("")
                report.append("**Method Details:**")
                for key, value in result.metadata.items():
                    if key not in ['individual_results']:  # Skip nested results
                        if isinstance(value, float):
                            report.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                        elif isinstance(value, (int, str, bool)):
                            report.append(f"- {key.replace('_', ' ').title()}: {value}")
            
            report.append("")
        
        return "\n".join(report)