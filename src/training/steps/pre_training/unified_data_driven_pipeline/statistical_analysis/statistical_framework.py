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
        EnhancedValidator
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
    # Fast-fail implementations - raise exceptions immediately when dependencies are missing
    class UnifiedCrossValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    def perform_cross_validation(*args, **kwargs):
        raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    def temporal_cross_validation(*args, **kwargs):
        raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    def analyze_splits(*args, **kwargs):
        raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    def validate_cv_integrity(*args, **kwargs):
        raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    class DataLeakageDetector:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    class EnhancedValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    class StabilityAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

    class OverfittingMonitor:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons validation utilities not available. Install required dependencies.")

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

    def test(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Perform the statistical test.
        
        Args:
            data: Input DataFrame containing the data to test
            **kwargs: Additional keyword arguments specific to the test
            
        Returns:
            Dictionary containing test results including statistics, p-values, and other relevant metrics
        """
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        
        # Basic data validation
        if data.shape[0] < 2:
            raise ValueError("Data must contain at least 2 samples for statistical testing")
        
        # Log test initiation
        tprint_info(f"Performing statistical test on {data.shape[0]} samples, {data.shape[1]} features")
        
        # Default implementation - subclasses should override this
        results = {
            'test_type': self.__class__.__name__,
            'n_samples': data.shape[0],
            'n_features': data.shape[1],
            'data_shape': data.shape,
            'missing_values': data.isnull().sum().sum(),
            'test_performed': False,
            'message': 'Default implementation - subclass should override test method'
        }
        
        tprint_warning("Using default test implementation - subclass should override test method")
        return results

    def is_significant(self, result: Dict[str, Any], alpha: float = 0.05) -> bool:
        """
        Check if the test result is statistically significant.
        
        Args:
            result: Dictionary containing test results from the test method
            alpha: Significance level (default: 0.05)
            
        Returns:
            Boolean indicating whether the result is statistically significant
        """
        # Validate inputs
        if result is None:
            raise ValueError("Test result cannot be None")
        
        if not isinstance(result, dict):
            raise TypeError("Test result must be a dictionary")
        
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
        
        # Default implementation - look for p-value in result
        if 'pvalue' in result:
            p_value = result['pvalue']
            if isinstance(p_value, (int, float)) and not pd.isna(p_value):
                return p_value < alpha
        
        # Look for p_values (plural) in result
        if 'p_values' in result:
            p_values = result['p_values']
            if isinstance(p_values, (list, tuple, np.ndarray)):
                # Check if any p-value is significant
                return any(isinstance(p, (int, float)) and not pd.isna(p) and p < alpha for p in p_values)
            elif isinstance(p_values, pd.DataFrame):
                # For DataFrame, check if any value is significant
                return (p_values < alpha).any().any()
        
        # Look for significance indicators
        if 'is_significant' in result:
            return bool(result['is_significant'])
        
        if 'significant' in result:
            return bool(result['significant'])
        
        # Default: assume not significant if no clear indicator
        tprint_warning("No clear significance indicator found in result - assuming not significant")
        return False

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
            self.enhanced_validation = EnhancedValidator()
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
