"""
Specialist Feature Diagnostics - Comprehensive Analysis Framework

Advanced diagnostic framework for specialist models providing:
- Comprehensive feature quality analysis
- Advanced orthogonalization validation
- Sophisticated target denoising assessment
- Statistical validation and robustness checks
- Performance monitoring and alerting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json

from sklearn.metrics import (
    roc_auc_score, accuracy_score, brier_score_loss,
    mean_squared_error, r2_score
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr, normaltest, kstest, entropy as shannon_entropy
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.versioned_artifacts import VersionedArtifactStore


@dataclass
class FeatureQualityMetrics:
    """Container for feature quality assessment results."""
    missing_rates: Dict[str, float] = field(default_factory=dict)
    cardinality: Dict[str, int] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    predictive_power: Dict[str, float] = field(default_factory=dict)
    distribution_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_score: float = 0.0


@dataclass
class OrthogonalizationDiagnostics:
    """Container for orthogonalization analysis results."""
    original_correlations: np.ndarray = None
    orthogonal_correlations: np.ndarray = None
    orthogonality_index: float = 0.0
    dropped_features: List[str] = field(default_factory=list)
    information_retention: float = 0.0
    predictive_power_retention: float = 0.0


@dataclass
class DenoisingAnalysis:
    """Container for target denoising assessment results."""
    noise_reduction: float = 0.0
    signal_preservation: float = 0.0
    information_content_change: float = 0.0
    distribution_shift: Dict[str, float] = field(default_factory=dict)
    temporal_stability: float = 0.0
    denoising_effectiveness: float = 0.0


class SpecialistFeatureDiagnostics:
    """
    Comprehensive diagnostic framework for specialist feature analysis.

    Provides advanced analysis capabilities for:
    - Feature quality assessment
    - Orthogonalization validation
    - Target denoising evaluation
    - Statistical robustness testing
    - Performance monitoring
    """

    def __init__(self, cache_dir: str = "specialist_diagnostics_cache"):
        """Initialize the diagnostics framework."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize components
        self.artifact_store = VersionedArtifactStore("versioned_artifacts")
        self.logger = logging.getLogger(__name__)

        # Caching for expensive computations
        self._feature_cache = {}
        self._orthogonalization_cache = {}
        self._denoising_cache = {}

        # Statistical test parameters
        self.stationarity_tests = ['adf', 'kpss']
        self.normality_tests = ['shapiro', 'normaltest']
        self.min_samples_for_stats = 100

    def comprehensive_feature_analysis(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        predictions: Optional[np.ndarray] = None,
        specialist_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Comprehensive feature quality assessment.

        Analyzes feature quality across multiple dimensions:
        - Missing data patterns
        - Cardinality and distribution
        - Temporal stability
        - Predictive power
        - Statistical properties

        Args:
            features: Feature matrix
            labels: Target labels
            predictions: Model predictions (optional)
            specialist_name: Name of specialist for caching

        Returns:
            Comprehensive feature analysis results
        """
        cache_key = f"{specialist_name}_feature_analysis_{hash(str(features.shape))}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        tprint_info(f"🔬 Running comprehensive feature analysis for {specialist_name}")

        try:
            # 1. Basic quality metrics
            quality_metrics = self._analyze_feature_quality(features)

            # 2. Relationship analysis
            relationship_analysis = self._analyze_feature_relationships(features, labels)

            # 3. Distribution and statistical analysis
            distribution_analysis = self._analyze_feature_distributions(features)

            # 4. Temporal stability analysis
            stability_analysis = self._analyze_temporal_stability(features, labels)

            # 5. Predictive power assessment
            predictive_analysis = self._assess_predictive_power(features, labels, predictions)

            # 6. Feature interaction analysis
            interaction_analysis = self._analyze_feature_interactions(features, labels)

            # 7. Overall quality scoring
            overall_quality = self._compute_overall_quality_score(
                quality_metrics, relationship_analysis, stability_analysis, predictive_analysis
            )

            results = {
                'specialist_name': specialist_name,
                'timestamp': datetime.now().isoformat(),
                'quality_metrics': quality_metrics,
                'relationship_analysis': relationship_analysis,
                'distribution_analysis': distribution_analysis,
                'stability_analysis': stability_analysis,
                'predictive_analysis': predictive_analysis,
                'interaction_analysis': interaction_analysis,
                'overall_quality_score': overall_quality,
                'recommendations': self._generate_feature_recommendations(
                    quality_metrics, relationship_analysis, predictive_analysis
                )
            }

            # Cache results
            self._feature_cache[cache_key] = results

            tprint_success(f"✅ Feature analysis completed for {specialist_name}")
            return results

        except Exception as e:
            tprint_error(f"❌ Feature analysis failed for {specialist_name}: {e}")
            return {
                'specialist_name': specialist_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def advanced_orthogonalization_diagnostics(
        self,
        original_features: pd.DataFrame,
        orthogonal_features: pd.DataFrame,
        labels: pd.Series,
        dropped_features: List[str] = None,
        specialist_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Advanced orthogonalization validation and diagnostics.

        Provides comprehensive analysis of orthogonalization effectiveness:
        - Correlation structure changes
        - Information content preservation
        - Predictive power retention
        - Orthogonalization strategy evaluation

        Args:
            original_features: Original feature matrix
            orthogonal_features: Orthogonalized feature matrix
            labels: Target labels
            dropped_features: List of features removed during orthogonalization
            specialist_name: Name of specialist for context

        Returns:
            Comprehensive orthogonalization diagnostics
        """
        cache_key = f"{specialist_name}_orthogonalization_{hash(str(original_features.shape))}"
        if cache_key in self._orthogonalization_cache:
            return self._orthogonalization_cache[cache_key]

        tprint_info(f"🎯 Running advanced orthogonalization diagnostics for {specialist_name}")

        try:
            dropped_features = dropped_features or []

            # 1. Correlation structure analysis
            correlation_analysis = self._analyze_correlation_structure(
                original_features, orthogonal_features
            )

            # 2. Information content preservation
            information_analysis = self._analyze_information_preservation(
                original_features, orthogonal_features, labels
            )

            # 3. Predictive power assessment
            predictive_analysis = self._assess_predictive_power_retention(
                original_features, orthogonal_features, labels
            )

            # 4. Dropped features analysis
            dropped_analysis = self._analyze_dropped_features(
                original_features, dropped_features, labels
            )

            # 5. Orthogonalization strategy evaluation
            strategy_analysis = self._evaluate_orthogonalization_strategy(
                original_features, orthogonal_features, correlation_analysis
            )

            results = {
                'specialist_name': specialist_name,
                'timestamp': datetime.now().isoformat(),
                'correlation_analysis': correlation_analysis,
                'information_analysis': information_analysis,
                'predictive_analysis': predictive_analysis,
                'dropped_features_analysis': dropped_analysis,
                'strategy_evaluation': strategy_analysis,
                'orthogonality_score': self._compute_orthogonality_score(
                    orthogonal_features, correlation_analysis
                ),
                'recommendations': self._generate_orthogonalization_recommendations(
                    correlation_analysis, information_analysis, predictive_analysis
                )
            }

            # Cache results
            self._orthogonalization_cache[cache_key] = results

            tprint_success(f"✅ Orthogonalization diagnostics completed for {specialist_name}")
            return results

        except Exception as e:
            tprint_error(f"❌ Orthogonalization diagnostics failed for {specialist_name}: {e}")
            return {
                'specialist_name': specialist_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def comprehensive_denoising_analysis(
        self,
        original_targets: pd.Series,
        denoised_targets: pd.Series,
        features: Optional[pd.DataFrame] = None,
        denoising_method: str = "unknown",
        specialist_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Comprehensive target denoising quality assessment.

        Analyzes denoising effectiveness across multiple dimensions:
        - Noise reduction metrics
        - Signal preservation analysis
        - Information content changes
        - Distribution shifts
        - Temporal stability
        - Prediction impact

        Args:
            original_targets: Original target series
            denoised_targets: Denoised target series
            features: Feature matrix (optional, for interaction analysis)
            denoising_method: Denoising method used
            specialist_name: Name of specialist for context

        Returns:
            Comprehensive denoising analysis results
        """
        cache_key = f"{specialist_name}_denoising_{denoising_method}_{hash(str(original_targets.shape))}"
        if cache_key in self._denoising_cache:
            return self._denoising_cache[cache_key]

        tprint_info(f"🎨 Running comprehensive denoising analysis for {specialist_name} ({denoising_method})")

        try:
            # 1. Denoising effectiveness metrics
            effectiveness_metrics = self._analyze_denoising_effectiveness(
                original_targets, denoised_targets
            )

            # 2. Signal preservation analysis
            signal_analysis = self._analyze_signal_preservation(
                original_targets, denoised_targets
            )

            # 3. Information content assessment
            information_analysis = self._assess_information_content_change(
                original_targets, denoised_targets
            )

            # 4. Distribution shift analysis
            distribution_analysis = self._analyze_distribution_shifts(
                original_targets, denoised_targets
            )

            # 5. Temporal stability assessment
            stability_analysis = self._assess_temporal_stability_denoised(
                original_targets, denoised_targets
            )

            # 6. Prediction impact analysis
            prediction_impact = self._evaluate_prediction_impact(
                original_targets, denoised_targets, features
            ) if features is not None else {}

            # 7. Method-specific analysis
            method_analysis = self._analyze_denoising_method_characteristics(
                original_targets, denoised_targets, denoising_method
            )

            # 8. Overall denoising quality score
            quality_score = self._compute_denoising_quality_score(
                effectiveness_metrics, signal_analysis, stability_analysis
            )

            results = {
                'specialist_name': specialist_name,
                'denoising_method': denoising_method,
                'timestamp': datetime.now().isoformat(),
                'effectiveness_metrics': effectiveness_metrics,
                'signal_analysis': signal_analysis,
                'information_analysis': information_analysis,
                'distribution_analysis': distribution_analysis,
                'stability_analysis': stability_analysis,
                'prediction_impact': prediction_impact,
                'method_characteristics': method_analysis,
                'overall_quality_score': quality_score,
                'recommendations': self._generate_denoising_recommendations(
                    effectiveness_metrics, signal_analysis, method_analysis
                )
            }

            # Cache results
            self._denoising_cache[cache_key] = results

            tprint_success(f"✅ Denoising analysis completed for {specialist_name}")
            return results

        except Exception as e:
            tprint_error(f"❌ Denoising analysis failed for {specialist_name}: {e}")
            return {
                'specialist_name': specialist_name,
                'denoising_method': denoising_method,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # ============================================================================
    # FEATURE ANALYSIS IMPLEMENTATION
    # ============================================================================

    def _analyze_feature_quality(self, features: pd.DataFrame) -> FeatureQualityMetrics:
        """Analyze basic feature quality metrics."""
        quality = FeatureQualityMetrics()

        for col in features.columns:
            # Missing rates
            quality.missing_rates[col] = features[col].isnull().mean()

            # Cardinality
            quality.cardinality[col] = features[col].nunique()

            # Basic distribution stats
            if not features[col].empty:
                quality.distribution_stats[col] = {
                    'mean': float(features[col].mean()),
                    'std': float(features[col].std()),
                    'skew': float(features[col].skew()),
                    'kurtosis': float(features[col].kurtosis()),
                    'min': float(features[col].min()),
                    'max': float(features[col].max())
                }

        return quality

    def _analyze_feature_relationships(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """Analyze relationships between features and target."""
        relationships = {
            'mutual_information': {},
            'spearman_correlation': {},
            'hsic_scores': {},
            'feature_target_dependence': {}
        }

        # Compute mutual information with target
        for col in features.columns:
            try:
                clean_data = features[col].dropna()
                clean_labels = labels.reindex(clean_data.index).dropna()

                if len(clean_data) >= self.min_samples_for_stats:
                    # Mutual information
                    mi_score = mutual_info_regression(
                        clean_data.values.reshape(-1, 1),
                        clean_labels.values
                    )[0]
                    relationships['mutual_information'][col] = float(mi_score)

                    # Spearman correlation
                    if len(clean_data) > 10:
                        corr, _ = spearmanr(clean_data, clean_labels)
                        relationships['spearman_correlation'][col] = float(corr) if not np.isnan(corr) else 0.0

                    # HSIC (simplified approximation)
                    relationships['hsic_scores'][col] = self._compute_hsic_simple(
                        clean_data.values, clean_labels.values
                    )

            except Exception as e:
                relationships['mutual_information'][col] = 0.0
                relationships['spearman_correlation'][col] = 0.0
                relationships['hsic_scores'][col] = 0.0

        return relationships

    def _analyze_feature_distributions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions and statistical properties."""
        distribution_analysis = {
            'stationarity_tests': {},
            'normality_tests': {},
            'outlier_analysis': {},
            'autocorrelation': {}
        }

        for col in features.columns:
            try:
                series = features[col].dropna()

                if len(series) >= self.min_samples_for_stats:
                    # Stationarity tests
                    stationarity = {}
                    try:
                        # ADF test
                        adf_result = adfuller(series.values, autolag='AIC')
                        stationarity['adf_pvalue'] = float(adf_result[1])
                        stationarity['adf_stationary'] = adf_result[1] < 0.05
                    except:
                        stationarity['adf_pvalue'] = None
                        stationarity['adf_stationary'] = None

                    try:
                        # KPSS test
                        kpss_result = kpss(series.values, regression='c', nlags='auto')
                        stationarity['kpss_pvalue'] = float(kpss_result[1])
                        stationarity['kpss_stationary'] = kpss_result[1] > 0.05
                    except:
                        stationarity['kpss_pvalue'] = None
                        stationarity['kpss_stationary'] = None

                    distribution_analysis['stationarity_tests'][col] = stationarity

                    # Normality tests
                    normality = {}
                    try:
                        _, p_value = normaltest(series.values)
                        normality['normaltest_pvalue'] = float(p_value)
                        normality['is_normal'] = p_value > 0.05
                    except:
                        normality['normaltest_pvalue'] = None
                        normality['is_normal'] = None

                    distribution_analysis['normality_tests'][col] = normality

                    # Outlier analysis
                    q1, q3 = series.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
                    distribution_analysis['outlier_analysis'][col] = {
                        'outlier_count': int(outliers),
                        'outlier_percentage': float(outliers / len(series)),
                        'iqr': float(iqr)
                    }

                    # Autocorrelation
                    if len(series) > 10:
                        autocorr = [pd.Series(series).autocorr(lag=i) for i in range(1, min(11, len(series)//2))]
                        distribution_analysis['autocorrelation'][col] = autocorr

            except Exception as e:
                # Skip problematic features
                continue

        return distribution_analysis

    def _analyze_temporal_stability(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """Analyze temporal stability of features."""
        stability_analysis = {
            'rolling_stability': {},
            'regime_stability': {},
            'feature_drift': {}
        }

        # Time series cross-validation for stability
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            stability_scores = {}

            for col in features.columns:
                fold_scores = []

                for train_idx, val_idx in tscv.split(features):
                    try:
                        X_train = features.iloc[train_idx][[col]].fillna(0)
                        y_train = labels.iloc[train_idx]
                        X_val = features.iloc[val_idx][[col]].fillna(0)
                        y_val = labels.iloc[val_idx]

                        if len(np.unique(y_train)) > 1 and len(X_train) > 10:
                            # Simple logistic regression for stability
                            lr = LogisticRegression(random_state=42, max_iter=1000)
                            lr.fit(X_train, y_train)

                            if len(np.unique(y_val)) > 1:
                                score = lr.score(X_val, y_val)
                                fold_scores.append(score)

                    except:
                        continue

                if fold_scores:
                    stability_scores[col] = {
                        'mean_score': float(np.mean(fold_scores)),
                        'std_score': float(np.std(fold_scores)),
                        'cv_score': float(np.std(fold_scores) / np.mean(fold_scores)) if np.mean(fold_scores) > 0 else float('inf')
                    }

            stability_analysis['rolling_stability'] = stability_scores

        except Exception as e:
            stability_analysis['rolling_stability'] = {'error': str(e)}

        return stability_analysis

    def _assess_predictive_power(self, features: pd.DataFrame, labels: pd.Series,
                                predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Assess predictive power of features."""
        predictive_analysis = {
            'individual_feature_power': {},
            'feature_combinations': {},
            'shap_importance': {}
        }

        # Individual feature predictive power
        for col in features.columns:
            try:
                clean_data = features[col].dropna()
                clean_labels = labels.reindex(clean_data.index).dropna()

                if len(clean_data) >= 50 and len(np.unique(clean_labels)) > 1:
                    # Simple AUC score
                    from sklearn.metrics import roc_auc_score

                    # Use logistic regression for predictive power assessment
                    lr = LogisticRegression(random_state=42, max_iter=1000)
                    lr.fit(clean_data.values.reshape(-1, 1), clean_labels)

                    probs = lr.predict_proba(clean_data.values.reshape(-1, 1))[:, 1]
                    auc = roc_auc_score(clean_labels, probs)

                    predictive_analysis['individual_feature_power'][col] = {
                        'auc_score': float(auc),
                        'predictive_strength': self._classify_predictive_strength(auc)
                    }

            except Exception as e:
                predictive_analysis['individual_feature_power'][col] = {
                    'auc_score': 0.5,
                    'predictive_strength': 'poor',
                    'error': str(e)
                }

        return predictive_analysis

    def _analyze_feature_interactions(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """Analyze feature interactions and dependencies."""
        interaction_analysis = {
            'correlation_network': {},
            'feature_clusters': {},
            'interaction_strength': {}
        }

        try:
            # Correlation network
            if len(features.columns) > 1:
                corr_matrix = features.corr().abs()

                # Find strong correlations
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if corr_val > 0.7:  # Strong correlation threshold
                            strong_correlations.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': float(corr_val)
                            })

                interaction_analysis['correlation_network'] = strong_correlations

        except Exception as e:
            interaction_analysis['correlation_network'] = {'error': str(e)}

        return interaction_analysis

    # ============================================================================
    # ORTHOGONALIZATION DIAGNOSTICS IMPLEMENTATION
    # ============================================================================

    def _analyze_correlation_structure(self, original: pd.DataFrame, orthogonal: pd.DataFrame) -> Dict[str, Any]:
        """Analyze changes in correlation structure."""
        correlation_structure = {}

        try:
            # Original correlation matrix
            orig_corr = original.corr().abs()

            # Orthogonal correlation matrix
            ortho_corr = orthogonal.corr().abs()

            # Max correlations
            correlation_structure['original_max_correlation'] = float(orig_corr.values[np.triu_indices_from(orig_corr.values, k=1)].max())
            correlation_structure['orthogonal_max_correlation'] = float(ortho_corr.values[np.triu_indices_from(ortho_corr.values, k=1)].max())

            # Correlation distribution
            orig_corr_values = orig_corr.values[np.triu_indices_from(orig_corr.values, k=1)]
            ortho_corr_values = ortho_corr.values[np.triu_indices_from(ortho_corr.values, k=1)]

            correlation_structure['original_correlation_distribution'] = {
                'mean': float(np.mean(orig_corr_values)),
                'std': float(np.std(orig_corr_values)),
                'percentile_95': float(np.percentile(orig_corr_values, 95))
            }

            correlation_structure['orthogonal_correlation_distribution'] = {
                'mean': float(np.mean(ortho_corr_values)),
                'std': float(np.std(ortho_corr_values)),
                'percentile_95': float(np.percentile(ortho_corr_values, 95))
            }

        except Exception as e:
            correlation_structure['error'] = str(e)

        return correlation_structure

    def _analyze_information_preservation(self, original: pd.DataFrame, orthogonal: pd.DataFrame,
                                        labels: pd.Series) -> Dict[str, Any]:
        """Analyze information content preservation."""
        information_preservation = {}

        try:
            # Compare mutual information with target
            orig_mi = {}
            ortho_mi = {}

            for col in original.columns:
                if col in orthogonal.columns:
                    # Original MI
                    orig_clean = original[col].dropna()
                    orig_labels = labels.reindex(orig_clean.index).dropna()
                    if len(orig_clean) > 10:
                        orig_mi[col] = mutual_info_regression(
                            orig_clean.values.reshape(-1, 1), orig_labels.values
                        )[0]

                    # Orthogonal MI
                    ortho_clean = orthogonal[col].dropna()
                    ortho_labels = labels.reindex(ortho_clean.index).dropna()
                    if len(ortho_clean) > 10:
                        ortho_mi[col] = mutual_info_regression(
                            ortho_clean.values.reshape(-1, 1), ortho_labels.values
                        )[0]

            # Compute retention metrics
            common_features = set(orig_mi.keys()) & set(ortho_mi.keys())
            if common_features:
                orig_scores = [orig_mi[f] for f in common_features]
                ortho_scores = [ortho_mi[f] for f in common_features]

                information_preservation['mutual_info_retention'] = {
                    'mean_orig': float(np.mean(orig_scores)),
                    'mean_ortho': float(np.mean(ortho_scores)),
                    'retention_ratio': float(np.mean(ortho_scores) / np.mean(orig_scores)) if np.mean(orig_scores) > 0 else 0.0
                }

        except Exception as e:
            information_preservation['error'] = str(e)

        return information_preservation

    def _assess_predictive_power_retention(self, original: pd.DataFrame, orthogonal: pd.DataFrame,
                                         labels: pd.Series) -> Dict[str, Any]:
        """Assess predictive power retention after orthogonalization."""
        predictive_retention = {}

        try:
            # Compare AUC scores
            orig_auc = {}
            ortho_auc = {}

            for col in original.columns:
                if col in orthogonal.columns:
                    for df_name, df, auc_dict in [('original', original, orig_auc), ('orthogonal', orthogonal, ortho_auc)]:
                        try:
                            clean_data = df[col].dropna()
                            clean_labels = labels.reindex(clean_data.index).dropna()

                            if len(clean_data) >= 50 and len(np.unique(clean_labels)) > 1:
                                lr = LogisticRegression(random_state=42, max_iter=1000)
                                lr.fit(clean_data.values.reshape(-1, 1), clean_labels)
                                probs = lr.predict_proba(clean_data.values.reshape(-1, 1))[:, 1]
                                auc = roc_auc_score(clean_labels, probs)
                                auc_dict[col] = auc
                        except:
                            auc_dict[col] = 0.5

            # Compute retention metrics
            common_features = set(orig_auc.keys()) & set(ortho_auc.keys())
            if common_features:
                orig_scores = [orig_auc[f] for f in common_features]
                ortho_scores = [ortho_auc[f] for f in common_features]

                predictive_retention['auc_retention'] = {
                    'mean_orig': float(np.mean(orig_scores)),
                    'mean_ortho': float(np.mean(ortho_scores)),
                    'retention_ratio': float(np.mean(ortho_scores) / np.mean(orig_scores)) if np.mean(orig_scores) > 0 else 0.0
                }

        except Exception as e:
            predictive_retention['error'] = str(e)

        return predictive_retention

    def _analyze_dropped_features(self, original: pd.DataFrame, dropped_features: List[str],
                                labels: pd.Series) -> Dict[str, Any]:
        """Analyze the impact of features dropped during orthogonalization."""
        dropped_analysis = {}

        try:
            if not dropped_features:
                dropped_analysis['no_dropped_features'] = True
                return dropped_analysis

            # Analyze importance of dropped features
            dropped_importance = {}
            for feature in dropped_features:
                if feature in original.columns:
                    try:
                        clean_data = original[feature].dropna()
                        clean_labels = labels.reindex(clean_data.index).dropna()

                        if len(clean_data) >= 50 and len(np.unique(clean_labels)) > 1:
                            lr = LogisticRegression(random_state=42, max_iter=1000)
                            lr.fit(clean_data.values.reshape(-1, 1), clean_labels)
                            probs = lr.predict_proba(clean_data.values.reshape(-1, 1))[:, 1]
                            auc = roc_auc_score(clean_labels, probs)
                            dropped_importance[feature] = float(auc)
                        else:
                            dropped_importance[feature] = 0.0
                    except:
                        dropped_importance[feature] = 0.0

            dropped_analysis['dropped_feature_importance'] = dropped_importance
            dropped_analysis['mean_importance'] = float(np.mean(list(dropped_importance.values())))
            dropped_analysis['max_importance'] = float(max(dropped_importance.values())) if dropped_importance else 0.0

            # Correlation network analysis
            if len(dropped_features) > 1:
                dropped_subset = original[dropped_features].dropna()
                if len(dropped_subset) > 10:
                    corr_matrix = dropped_subset.corr().abs()
                    dropped_analysis['internal_correlations'] = {
                        'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                        'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max())
                    }

        except Exception as e:
            dropped_analysis['error'] = str(e)

        return dropped_analysis

    # ============================================================================
    # DENOISING ANALYSIS IMPLEMENTATION
    # ============================================================================

    def _analyze_denoising_effectiveness(self, original: pd.Series, denoised: pd.Series) -> Dict[str, Any]:
        """Analyze denoising effectiveness metrics."""
        effectiveness = {}

        try:
            # Noise reduction via variance comparison
            orig_variance = original.var()
            denoised_variance = denoised.var()

            effectiveness['variance_reduction'] = {
                'original_variance': float(orig_variance),
                'denoised_variance': float(denoised_variance),
                'reduction_ratio': float(denoised_variance / orig_variance) if orig_variance > 0 else 1.0
            }

            # Signal-to-noise ratio improvement
            orig_signal = abs(original.mean())
            orig_noise = original.std()
            orig_snr = orig_signal / orig_noise if orig_noise > 0 else 0

            denoised_signal = abs(denoised.mean())
            denoised_noise = denoised.std()
            denoised_snr = denoised_signal / denoised_noise if denoised_noise > 0 else 0

            effectiveness['snr_improvement'] = {
                'original_snr': float(orig_snr),
                'denoised_snr': float(denoised_snr),
                'improvement_ratio': float(denoised_snr / orig_snr) if orig_snr > 0 else 0.0
            }

            # Outlier reduction
            orig_outliers = self._count_outliers(original)
            denoised_outliers = self._count_outliers(denoised)

            effectiveness['outlier_reduction'] = {
                'original_outliers': orig_outliers,
                'denoised_outliers': denoised_outliers,
                'reduction_ratio': float(denoised_outliers / orig_outliers) if orig_outliers > 0 else 1.0
            }

        except Exception as e:
            effectiveness['error'] = str(e)

        return effectiveness

    def _analyze_signal_preservation(self, original: pd.Series, denoised: pd.Series) -> Dict[str, Any]:
        """Analyze signal preservation quality."""
        signal_preservation = {}

        try:
            # Trend preservation (correlation with original)
            correlation = original.corr(denoised)
            signal_preservation['trend_preservation'] = {
                'correlation': float(correlation),
                'preservation_quality': self._classify_preservation_quality(correlation)
            }

            # Mean preservation
            orig_mean = original.mean()
            denoised_mean = denoised.mean()
            signal_preservation['mean_preservation'] = {
                'original_mean': float(orig_mean),
                'denoised_mean': float(denoised_mean),
                'mean_change_pct': float(abs(denoised_mean - orig_mean) / abs(orig_mean)) if orig_mean != 0 else 0.0
            }

            # Direction preservation (sign agreement)
            orig_direction = np.sign(original.diff().fillna(0))
            denoised_direction = np.sign(denoised.diff().fillna(0))
            direction_agreement = np.mean(orig_direction == denoised_direction)

            signal_preservation['direction_preservation'] = {
                'agreement_rate': float(direction_agreement),
                'preservation_quality': 'excellent' if direction_agreement > 0.8 else 'good' if direction_agreement > 0.6 else 'poor'
            }

        except Exception as e:
            signal_preservation['error'] = str(e)

        return signal_preservation

    def _assess_information_content_change(self, original: pd.Series, denoised: pd.Series) -> Dict[str, Any]:
        """Assess changes in information content."""
        information_change = {}

        try:
            # Entropy comparison
            orig_entropy = shannon_entropy(np.histogram(original, bins=50)[0])
            denoised_entropy = shannon_entropy(np.histogram(denoised, bins=50)[0])

            information_change['entropy_change'] = {
                'original_entropy': float(orig_entropy),
                'denoised_entropy': float(denoised_entropy),
                'entropy_change_ratio': float(denoised_entropy / orig_entropy) if orig_entropy > 0 else 0.0
            }

            # Mutual information with lagged self (autocorrelation proxy)
            orig_autocorr = abs(original.autocorr(lag=1)) if len(original) > 10 else 0
            denoised_autocorr = abs(denoised.autocorr(lag=1)) if len(denoised) > 10 else 0

            information_change['autocorrelation_preservation'] = {
                'original_autocorr': float(orig_autocorr),
                'denoised_autocorr': float(denoised_autocorr),
                'preservation_ratio': float(denoised_autocorr / orig_autocorr) if orig_autocorr > 0 else 0.0
            }

        except Exception as e:
            information_change['error'] = str(e)

        return information_change

    def _analyze_distribution_shifts(self, original: pd.Series, denoised: pd.Series) -> Dict[str, Any]:
        """Analyze distribution shifts caused by denoising."""
        distribution_shifts = {}

        try:
            # Statistical moments comparison
            for moment_name, orig_func, denoised_func in [
                ('mean', lambda x: x.mean(), lambda x: x.mean()),
                ('std', lambda x: x.std(), lambda x: x.std()),
                ('skewness', lambda x: x.skew(), lambda x: x.skew()),
                ('kurtosis', lambda x: x.kurtosis(), lambda x: x.kurtosis())
            ]:
                orig_val = float(orig_func(original))
                denoised_val = float(denoised_func(denoised))

                distribution_shifts[f'{moment_name}_shift'] = {
                    'original': orig_val,
                    'denoised': denoised_val,
                    'absolute_change': abs(denoised_val - orig_val),
                    'relative_change': abs(denoised_val - orig_val) / abs(orig_val) if orig_val != 0 else 0.0
                }

            # Kolmogorov-Smirnov test for distribution difference
            try:
                from scipy.stats import ks_2samp
                ks_stat, ks_pvalue = ks_2samp(original.values, denoised.values)
                distribution_shifts['distribution_similarity'] = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'distributions_similar': ks_pvalue > 0.05  # Null hypothesis: same distribution
                }
            except:
                distribution_shifts['distribution_similarity'] = {'error': 'KS test failed'}

        except Exception as e:
            distribution_shifts['error'] = str(e)

        return distribution_shifts

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _compute_hsic_simple(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Simplified HSIC computation for feature analysis."""
        try:
            # Simple RBF kernel approximation
            sigma = np.std(np.concatenate([X, Y]))
            K = np.exp(-pdist(X.reshape(-1, 1), metric='sqeuclidean') / (2 * sigma ** 2))
            L = np.exp(-pdist(Y.reshape(-1, 1), metric='sqeuclidean') / (2 * sigma ** 2))

            K = squareform(K)
            L = squareform(L)

            # Center the kernels
            H = np.eye(len(K)) - np.ones((len(K), len(K))) / len(K)
            K_centered = H @ K @ H
            L_centered = H @ L @ H

            # HSIC statistic
            hsic = np.trace(K_centered @ L_centered) / (len(K) ** 2)
            return float(hsic)

        except:
            return 0.0

    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        try:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
            return int(outliers)
        except:
            return 0

    def _classify_predictive_strength(self, auc: float) -> str:
        """Classify predictive strength based on AUC."""
        if auc >= 0.8:
            return 'excellent'
        elif auc >= 0.7:
            return 'good'
        elif auc >= 0.6:
            return 'fair'
        elif auc >= 0.55:
            return 'poor'
        else:
            return 'useless'

    def _classify_preservation_quality(self, correlation: float) -> str:
        """Classify signal preservation quality."""
        if correlation >= 0.9:
            return 'excellent'
        elif correlation >= 0.8:
            return 'good'
        elif correlation >= 0.7:
            return 'fair'
        elif correlation >= 0.6:
            return 'poor'
        else:
            return 'very_poor'

    def _compute_overall_quality_score(self, quality: FeatureQualityMetrics,
                                     relationships: Dict, stability: Dict,
                                     predictive: Dict) -> float:
        """Compute overall feature quality score."""
        try:
            # Component scores (0-1 scale)
            missing_score = 1.0 - np.mean(list(quality.missing_rates.values()))

            predictive_scores = [v.get('auc_score', 0.5) for v in predictive.get('individual_feature_power', {}).values()]
            predictive_score = np.mean(predictive_scores) if predictive_scores else 0.5

            stability_scores = [v.get('mean_score', 0.5) for v in stability.get('rolling_stability', {}).values()]
            stability_score = np.mean(stability_scores) if stability_scores else 0.5

            # Weighted average
            overall_score = (
                0.3 * missing_score +
                0.4 * predictive_score +
                0.3 * stability_score
            )

            return float(np.clip(overall_score, 0.0, 1.0))

        except:
            return 0.5

    def _compute_orthogonality_score(self, orthogonal_features: pd.DataFrame,
                                   correlation_analysis: Dict) -> float:
        """Compute orthogonality quality score."""
        try:
            max_corr = correlation_analysis.get('orthogonal_max_correlation', 1.0)
            mean_corr = correlation_analysis.get('orthogonal_correlation_distribution', {}).get('mean', 0.5)

            # Perfect orthogonality = 0 correlation
            orthogonality_score = 1.0 - (0.7 * max_corr + 0.3 * mean_corr)
            return float(np.clip(orthogonality_score, 0.0, 1.0))

        except:
            return 0.5

    def _compute_denoising_quality_score(self, effectiveness: Dict,
                                        signal_analysis: Dict,
                                        stability_analysis: Dict) -> float:
        """Compute overall denoising quality score."""
        try:
            # Component scores
            variance_reduction = effectiveness.get('variance_reduction', {}).get('reduction_ratio', 1.0)
            snr_improvement = effectiveness.get('snr_improvement', {}).get('improvement_ratio', 1.0)
            trend_preservation = signal_analysis.get('trend_preservation', {}).get('correlation', 0.0)

            # Combined score (lower variance + higher SNR + better trend preservation = better)
            quality_score = (
                0.3 * (2.0 - variance_reduction) / 2.0 +  # Prefer lower variance (0-1 scale)
                0.3 * min(snr_improvement, 2.0) / 2.0 +     # Cap SNR improvement
                0.4 * trend_preservation                       # Trend preservation is most important
            )

            return float(np.clip(quality_score, 0.0, 1.0))

        except:
            return 0.5

    def _generate_feature_recommendations(self, quality: FeatureQualityMetrics,
                                        relationships: Dict, predictive: Dict) -> List[str]:
        """Generate feature improvement recommendations."""
        recommendations = []

        try:
            # Missing data recommendations
            high_missing = [col for col, rate in quality.missing_rates.items() if rate > 0.2]
            if high_missing:
                recommendations.append(f"High missing data in features: {high_missing[:3]}... Consider imputation or removal.")

            # Predictive power recommendations
            weak_features = [col for col, data in predictive.get('individual_feature_power', {}).items()
                           if data.get('auc_score', 0.5) < 0.55]
            if weak_features:
                recommendations.append(f"Poor predictive features: {weak_features[:3]}... Consider feature engineering or removal.")

            # Stability recommendations
            unstable_features = [col for col, data in relationships.get('mutual_information', {}).items()
                               if data < 0.01]
            if unstable_features:
                recommendations.append(f"Low information features: {unstable_features[:3]}... May be adding noise.")

        except:
            recommendations.append("Unable to generate specific recommendations due to analysis errors.")

        return recommendations

    def _generate_orthogonalization_recommendations(self, correlation_analysis: Dict,
                                                  information_analysis: Dict,
                                                  predictive_analysis: Dict) -> List[str]:
        """Generate orthogonalization improvement recommendations."""
        recommendations = []

        try:
            max_corr = correlation_analysis.get('orthogonal_max_correlation', 1.0)
            if max_corr > 0.8:
                recommendations.append(f"High remaining correlation ({max_corr:.3f}). Consider stricter orthogonalization threshold.")

            retention = information_analysis.get('mutual_info_retention', {}).get('retention_ratio', 1.0)
            if retention < 0.8:
                recommendations.append(f"Information loss in orthogonalization ({retention:.2f}). Consider alternative methods.")

            auc_retention = predictive_analysis.get('auc_retention', {}).get('retention_ratio', 1.0)
            if auc_retention < 0.9:
                recommendations.append(f"Predictive power loss ({auc_retention:.2f}). Evaluate orthogonalization necessity.")

        except:
            recommendations.append("Unable to generate orthogonalization recommendations due to analysis errors.")

        return recommendations

    def _generate_denoising_recommendations(self, effectiveness: Dict,
                                          signal_analysis: Dict,
                                          method_analysis: Dict) -> List[str]:
        """Generate denoising improvement recommendations."""
        recommendations = []

        try:
            snr_improvement = effectiveness.get('snr_improvement', {}).get('improvement_ratio', 1.0)
            if snr_improvement < 1.2:
                recommendations.append("Limited SNR improvement. Consider alternative denoising methods.")

            correlation = signal_analysis.get('trend_preservation', {}).get('correlation', 0.0)
            if correlation < 0.8:
                recommendations.append(f"Poor signal preservation ({correlation:.2f}). Adjust denoising parameters.")

            outlier_reduction = effectiveness.get('outlier_reduction', {}).get('reduction_ratio', 1.0)
            if outlier_reduction > 0.5:
                recommendations.append("Significant outlier reduction achieved. Current method effective.")

        except:
            recommendations.append("Unable to generate denoising recommendations due to analysis errors.")

        return recommendations

    def clear_cache(self):
        """Clear all diagnostic caches."""
        self._feature_cache.clear()
        self._orthogonalization_cache.clear()
        self._denoising_cache.clear()
        tprint_info("🧹 Diagnostic caches cleared")

    def save_diagnostics_report(self, diagnostics: Dict, filename: str):
        """Save comprehensive diagnostics report."""
        try:
            filepath = self.cache_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Convert numpy types to JSON-serializable
            def serialize_obj(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: serialize_obj(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_obj(item) for item in obj]
                else:
                    return obj

            serializable_diagnostics = serialize_obj(diagnostics)

            with open(filepath, 'w') as f:
                json.dump(serializable_diagnostics, f, indent=2)

            tprint_success(f"💾 Diagnostics report saved: {filepath}")

        except Exception as e:
            tprint_error(f"❌ Failed to save diagnostics report: {e}")
