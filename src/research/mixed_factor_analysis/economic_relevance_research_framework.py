"""
ML/Data-Driven Research Framework for Economic Relevance of Market Dimensions.

This module provides comprehensive research methodologies to determine what qualifies as
"economically relevant" for market dimensions beyond simple statistical significance.
It focuses on measuring actual impact on price movement patterns, not just momentum.

Key Research Questions:
1. How do we know if volatility, microstructure, liquidity dimensions have CAUSAL impact on price?
2. What constitutes "economic relevance" vs statistical significance?
3. How do dimensions interact to influence price movement patterns?
4. Which patterns are exploitable for trading vs noise?
5. How do we measure impact on price movement PATTERNS beyond momentum?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from src.utils.logger import system_logger

class PriceMovementPattern(Enum):
    """Types of price movement patterns to analyze."""
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    BREAKOUT_ACCELERATION = "breakout_acceleration"
    CONSOLIDATION_TIGHTENING = "consolidation_tightening"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    MOMENTUM_PERSISTENCE = "momentum_persistence"
    MEAN_REVERSION_SPEED = "mean_reversion_speed"
    TAIL_EVENT_CLUSTERING = "tail_event_clustering"
    REGIME_TRANSITION = "regime_transition"

class EconomicRelevanceMetric(Enum):
    """Metrics for measuring economic relevance."""
    # Causal Impact Metrics
    CAUSAL_IMPACT_SCORE = "causal_impact_score"
    GRANGER_CAUSALITY_P_VALUE = "granger_causality_p_value"
    INSTRUMENTAL_VARIABLE_ESTIMATE = "instrumental_variable_estimate"

    # Pattern Prediction Metrics
    PATTERN_PREDICTION_ACCURACY = "pattern_prediction_accuracy"
    PATTERN_TIMING_PRECISION = "pattern_timing_precision"
    PATTERN_MAGNITUDE_CORRELATION = "pattern_magnitude_correlation"

    # Economic Value Metrics
    TRADING_SIGNAL_SHARPE = "trading_signal_sharpe"
    INFORMATION_RATIO = "information_ratio"
    ECONOMIC_SIGNIFICANCE_THRESHOLD = "economic_significance_threshold"

    # Robustness Metrics
    OUT_OF_SAMPLE_STABILITY = "out_of_sample_stability"
    REGIME_INVARIANCE_SCORE = "regime_invariance_score"
    NOISE_RESILIENCE_FACTOR = "noise_resilience_factor"

@dataclass
class ResearchMethodologyConfig:
    """Configuration for research methodologies."""
    # Time series parameters
    lookback_windows: List[int] = None
    prediction_horizons: List[int] = None

    # Statistical significance
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"  # or "fdr"

    # Economic significance thresholds
    min_sharpe_ratio: float = 0.5
    min_information_ratio: float = 0.3
    min_prediction_accuracy: float = 0.55

    # Robustness testing
    bootstrap_samples: int = 1000
    cross_validation_folds: int = 5
    noise_stress_test_levels: List[float] = None

    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 50]
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 10, 20]
        if self.noise_stress_test_levels is None:
            self.noise_stress_test_levels = [0.1, 0.2, 0.5, 1.0]

@dataclass
class EconomicRelevanceResult:
    """Results from economic relevance analysis."""
    dimension_name: str
    pattern_type: PriceMovementPattern
    relevance_metrics: Dict[EconomicRelevanceMetric, float]
    statistical_significance: Dict[str, float]
    economic_interpretation: str
    trading_implications: str
    robustness_tests: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    methodology_metadata: Dict[str, Any]

    @property
    def is_economically_relevant(self) -> bool:
        """Determine if dimension is economically relevant based on multiple criteria."""
        criteria = [
            self.relevance_metrics.get(EconomicRelevanceMetric.TRADING_SIGNAL_SHARPE, 0) > 0.5,
            self.relevance_metrics.get(EconomicRelevanceMetric.PATTERN_PREDICTION_ACCURACY, 0) > 0.55,
            self.relevance_metrics.get(EconomicRelevanceMetric.OUT_OF_SAMPLE_STABILITY, 0) > 0.7,
            self.statistical_significance.get('p_value', 1.0) < 0.05
        ]
        return sum(criteria) >= 3  # At least 3 out of 4 criteria must be met

class BaseResearchMethodology(ABC):
    """Base class for economic relevance research methodologies."""

    def __init__(self, config: ResearchMethodologyConfig):
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)

    @abstractmethod
    def analyze_economic_relevance(self,
                                 market_data: pd.DataFrame,
                                 dimension_features: pd.DataFrame,
                                 dimension_name: str,
                                 pattern_type: PriceMovementPattern) -> EconomicRelevanceResult:
        """Analyze economic relevance of a dimension for a specific price pattern."""
        pass

    def _calculate_returns_and_patterns(self, market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various return series and price patterns."""
        prices = market_data['close']
        returns = prices.pct_change().fillna(0)

        patterns = {
            'returns': returns,
            'log_returns': np.log(prices / prices.shift(1)).fillna(0),
            'volatility': returns.rolling(20).std(),
            'momentum_5': returns.rolling(5).mean(),
            'momentum_20': returns.rolling(20).mean(),
            'mean_reversion_signal': (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
        }

        return patterns

    def _bootstrap_confidence_interval(self,
                                     data: np.ndarray,
                                     statistic_func: Callable,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic."""
        n_bootstrap = self.config.bootstrap_samples
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return (ci_lower, ci_upper)

class CausalImpactResearchMethodology(BaseResearchMethodology):
    """
    Research methodology focused on establishing causal relationships between
    market dimensions and price movement patterns.

    Uses techniques like:
    - Granger Causality testing
    - Instrumental Variables
    - Difference-in-Differences
    - Regression Discontinuity
    """

    def analyze_economic_relevance(self,
                                 market_data: pd.DataFrame,
                                 dimension_features: pd.DataFrame,
                                 dimension_name: str,
                                 pattern_type: PriceMovementPattern) -> EconomicRelevanceResult:
        """Analyze causal impact of dimension on price patterns."""

        self.logger.info(f"🔬 Analyzing causal impact of {dimension_name} on {pattern_type.value}")

        patterns = self._calculate_returns_and_patterns(market_data)

        # Create composite dimension signal
        dimension_signal = self._create_dimension_signal(dimension_features)

        # Analyze causal relationships
        causal_metrics = {}

        # 1. Granger Causality Test
        granger_results = self._granger_causality_test(dimension_signal, patterns, pattern_type)
        causal_metrics[EconomicRelevanceMetric.GRANGER_CAUSALITY_P_VALUE] = granger_results['p_value']

        # 2. Instrumental Variables Approach
        iv_results = self._instrumental_variables_analysis(dimension_signal, patterns, pattern_type)
        causal_metrics[EconomicRelevanceMetric.INSTRUMENTAL_VARIABLE_ESTIMATE] = iv_results['causal_estimate']

        # 3. Causal Impact Score (composite measure)
        causal_impact_score = self._calculate_causal_impact_score(granger_results, iv_results)
        causal_metrics[EconomicRelevanceMetric.CAUSAL_IMPACT_SCORE] = causal_impact_score

        # Statistical significance
        statistical_significance = {
            'granger_p_value': granger_results['p_value'],
            'iv_p_value': iv_results.get('p_value', 1.0),
            'composite_p_value': min(granger_results['p_value'], iv_results.get('p_value', 1.0))
        }

        # Robustness tests
        robustness_tests = self._conduct_robustness_tests(dimension_signal, patterns, pattern_type)

        # Economic interpretation
        economic_interpretation = self._generate_causal_interpretation(
            dimension_name, pattern_type, causal_metrics, statistical_significance
        )

        trading_implications = self._generate_trading_implications(
            dimension_name, pattern_type, causal_metrics
        )

        return EconomicRelevanceResult(
            dimension_name=dimension_name,
            pattern_type=pattern_type,
            relevance_metrics=causal_metrics,
            statistical_significance=statistical_significance,
            economic_interpretation=economic_interpretation,
            trading_implications=trading_implications,
            robustness_tests=robustness_tests,
            confidence_intervals={},
            methodology_metadata={'methodology': 'causal_impact', 'version': '1.0'}
        )

    def _granger_causality_test(self,
                              dimension_signal: pd.Series,
                              patterns: Dict[str, pd.Series],
                              pattern_type: PriceMovementPattern) -> Dict[str, float]:
        """Perform Granger causality test."""

        # Select appropriate target pattern
        target_pattern = self._select_target_pattern(patterns, pattern_type)

        # Align data
        aligned_data = pd.concat([dimension_signal, target_pattern], axis=1).dropna()
        if len(aligned_data) < 100:
            return {'p_value': 1.0, 'f_statistic': 0.0}

        X = aligned_data.iloc[:, 0].values  # dimension signal
        Y = aligned_data.iloc[:, 1].values  # target pattern

        # Simple Granger causality test (X -> Y)
        max_lag = min(10, len(X) // 10)

        try:
            # Restricted model: Y(t) = α + β₁Y(t-1) + ... + βₚY(t-p) + ε(t)
            # Unrestricted model: Y(t) = α + β₁Y(t-1) + ... + βₚY(t-p) + γ₁X(t-1) + ... + γₚX(t-p) + ε(t)

            # Create lagged variables
            Y_lagged = np.column_stack([np.roll(Y, i+1) for i in range(max_lag)])
            X_lagged = np.column_stack([np.roll(X, i+1) for i in range(max_lag)])

            # Remove initial observations affected by rolling
            Y_current = Y[max_lag:]
            Y_lagged = Y_lagged[max_lag:]
            X_lagged = X_lagged[max_lag:]

            # Fit restricted model (only Y lags)
            from sklearn.linear_model import LinearRegression

            restricted_model = LinearRegression()
            restricted_model.fit(Y_lagged, Y_current)
            restricted_predictions = restricted_model.predict(Y_lagged)
            rss_restricted = np.sum((Y_current - restricted_predictions) ** 2)

            # Fit unrestricted model (Y lags + X lags)
            unrestricted_features = np.column_stack([Y_lagged, X_lagged])
            unrestricted_model = LinearRegression()
            unrestricted_model.fit(unrestricted_features, Y_current)
            unrestricted_predictions = unrestricted_model.predict(unrestricted_features)
            rss_unrestricted = np.sum((Y_current - unrestricted_predictions) ** 2)

            # F-test
            n = len(Y_current)
            k_restricted = max_lag
            k_unrestricted = 2 * max_lag

            f_stat = ((rss_restricted - rss_unrestricted) / (k_unrestricted - k_restricted)) / (rss_unrestricted / (n - k_unrestricted - 1))

            # Calculate p-value
            from scipy.stats import f
            p_value = 1 - f.cdf(f_stat, k_unrestricted - k_restricted, n - k_unrestricted - 1)

            return {'p_value': float(p_value), 'f_statistic': float(f_stat)}

        except Exception as e:
            self.logger.warning(f"Granger causality test failed: {e}")
            return {'p_value': 1.0, 'f_statistic': 0.0}

    def _instrumental_variables_analysis(self,
                                       dimension_signal: pd.Series,
                                       patterns: Dict[str, pd.Series],
                                       pattern_type: PriceMovementPattern) -> Dict[str, float]:
        """Perform instrumental variables analysis to identify causal effects."""

        target_pattern = self._select_target_pattern(patterns, pattern_type)

        # Create potential instruments (lagged values, external factors)
        instruments = []

        # Lag 2 and 3 of dimension signal as instruments
        for lag in [2, 3]:
            instrument = dimension_signal.shift(lag)
            if not instrument.isna().all():
                instruments.append(instrument)

        if not instruments:
            return {'causal_estimate': 0.0, 'p_value': 1.0}

        # Align all data
        data_dict = {'target': target_pattern, 'treatment': dimension_signal}
        for i, instrument in enumerate(instruments):
            data_dict[f'instrument_{i}'] = instrument

        aligned_data = pd.concat(data_dict, axis=1).dropna()

        if len(aligned_data) < 50:
            return {'causal_estimate': 0.0, 'p_value': 1.0}

        try:
            # Two-stage least squares (2SLS)
            # First stage: treatment ~ instruments
            X_instruments = aligned_data[[col for col in aligned_data.columns if col.startswith('instrument')]].values
            treatment = aligned_data['treatment'].values
            target = aligned_data['target'].values

            # First stage regression
            first_stage = LinearRegression()
            first_stage.fit(X_instruments, treatment)
            treatment_predicted = first_stage.predict(X_instruments)

            # Second stage regression
            second_stage = LinearRegression()
            second_stage.fit(treatment_predicted.reshape(-1, 1), target)
            causal_estimate = second_stage.coef_[0]

            # Calculate standard error (simplified)
            residuals = target - second_stage.predict(treatment_predicted.reshape(-1, 1))
            mse = np.mean(residuals ** 2)
            var_treatment_predicted = np.var(treatment_predicted)
            se = np.sqrt(mse / (len(treatment_predicted) * var_treatment_predicted))

            # t-test
            t_stat = causal_estimate / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(treatment_predicted) - 2))

            return {
                'causal_estimate': float(causal_estimate),
                'p_value': float(p_value),
                'standard_error': float(se)
            }

        except Exception as e:
            self.logger.warning(f"IV analysis failed: {e}")
            return {'causal_estimate': 0.0, 'p_value': 1.0}

    def _calculate_causal_impact_score(self,
                                     granger_results: Dict[str, float],
                                     iv_results: Dict[str, float]) -> float:
        """Calculate composite causal impact score."""

        # Convert p-values to evidence scores (lower p-value = higher evidence)
        granger_evidence = max(0, 1 - granger_results['p_value'])
        iv_evidence = max(0, 1 - iv_results.get('p_value', 1.0))

        # Weight by effect size
        iv_effect_size = abs(iv_results.get('causal_estimate', 0))

        # Composite score
        causal_impact_score = (granger_evidence * 0.4 +
                              iv_evidence * 0.4 +
                              min(iv_effect_size, 1.0) * 0.2)

        return float(causal_impact_score)

    def _select_target_pattern(self,
                             patterns: Dict[str, pd.Series],
                             pattern_type: PriceMovementPattern) -> pd.Series:
        """Select appropriate target pattern based on pattern type."""

        if pattern_type == PriceMovementPattern.MOMENTUM_PERSISTENCE:
            return patterns['momentum_5']
        elif pattern_type == PriceMovementPattern.MEAN_REVERSION_SPEED:
            return patterns['mean_reversion_signal']
        elif pattern_type == PriceMovementPattern.VOLATILITY_EXPANSION:
            return patterns['volatility']
        else:
            return patterns['returns']  # Default to returns

    def _create_dimension_signal(self, dimension_features: pd.DataFrame) -> pd.Series:
        """Create composite dimension signal from features."""
        if len(dimension_features.columns) == 1:
            return dimension_features.iloc[:, 0]

        # Use PCA to create composite signal
        try:
            from sklearn.decomposition import PCA

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(dimension_features.fillna(0))

            pca = PCA(n_components=1)
            composite_signal = pca.fit_transform(features_scaled)

            return pd.Series(composite_signal.flatten(), index=dimension_features.index)
        except:
            # Fallback to mean
            return dimension_features.mean(axis=1)

    def _conduct_robustness_tests(self,
                                dimension_signal: pd.Series,
                                patterns: Dict[str, pd.Series],
                                pattern_type: PriceMovementPattern) -> Dict[str, float]:
        """Conduct robustness tests for causal relationships."""

        robustness_results = {}

        # 1. Subsample stability test
        target_pattern = self._select_target_pattern(patterns, pattern_type)
        aligned_data = pd.concat([dimension_signal, target_pattern], axis=1).dropna()

        if len(aligned_data) > 100:
            subsample_results = []
            n_subsamples = 10
            subsample_size = len(aligned_data) // 2

            for _ in range(n_subsamples):
                subsample = aligned_data.sample(n=subsample_size, random_state=np.random.randint(1000))
                granger_result = self._granger_causality_test(
                    subsample.iloc[:, 0],
                    {pattern_type.value: subsample.iloc[:, 1]},
                    pattern_type
                )
                subsample_results.append(1 - granger_result['p_value'])  # Evidence score

            robustness_results['subsample_stability'] = float(np.std(subsample_results))

        # 2. Noise resilience test
        noise_resilience_scores = []
        for noise_level in self.config.noise_stress_test_levels:
            noisy_signal = dimension_signal + np.random.normal(0, noise_level * dimension_signal.std(), len(dimension_signal))
            granger_result = self._granger_causality_test(
                noisy_signal,
                patterns,
                pattern_type
            )
            noise_resilience_scores.append(1 - granger_result['p_value'])

        robustness_results['noise_resilience'] = float(np.mean(noise_resilience_scores))

        return robustness_results

    def _generate_causal_interpretation(self,
                                      dimension_name: str,
                                      pattern_type: PriceMovementPattern,
                                      causal_metrics: Dict[EconomicRelevanceMetric, float],
                                      statistical_significance: Dict[str, float]) -> str:
        """Generate interpretation of causal analysis results."""

        causal_score = causal_metrics.get(EconomicRelevanceMetric.CAUSAL_IMPACT_SCORE, 0)
        granger_p = statistical_significance.get('granger_p_value', 1.0)

        if causal_score > 0.7 and granger_p < 0.01:
            strength = "strong"
        elif causal_score > 0.5 and granger_p < 0.05:
            strength = "moderate"
        elif causal_score > 0.3:
            strength = "weak"
        else:
            strength = "negligible"

        interpretation = f"{dimension_name} shows {strength} causal impact on {pattern_type.value} "
        interpretation += f"(causal score: {causal_score:.3f}, p-value: {granger_p:.3f})"

        return interpretation

    def _generate_trading_implications(self,
                                     dimension_name: str,
                                     pattern_type: PriceMovementPattern,
                                     causal_metrics: Dict[EconomicRelevanceMetric, float]) -> str:
        """Generate trading implications from causal analysis."""

        causal_score = causal_metrics.get(EconomicRelevanceMetric.CAUSAL_IMPACT_SCORE, 0)

        if causal_score > 0.5:
            if pattern_type == PriceMovementPattern.MOMENTUM_PERSISTENCE:
                return f"Use {dimension_name} signals to enhance momentum strategy timing and persistence prediction"
            elif pattern_type == PriceMovementPattern.MEAN_REVERSION_SPEED:
                return f"Use {dimension_name} signals to time mean reversion entries and predict reversion speed"
            elif pattern_type == PriceMovementPattern.VOLATILITY_EXPANSION:
                return f"Use {dimension_name} signals for volatility forecasting and options strategies"
            else:
                return f"Use {dimension_name} signals for general price pattern prediction"
        else:
            return f"Limited trading utility for {dimension_name} in {pattern_type.value} prediction"

class PatternPredictionResearchMethodology(BaseResearchMethodology):
    """
    Research methodology focused on measuring how well dimensions predict
    specific price movement patterns.

    Uses ML techniques to measure predictive power for:
    - Pattern occurrence probability
    - Pattern timing precision
    - Pattern magnitude forecasting
    """

    def analyze_economic_relevance(self,
                                 market_data: pd.DataFrame,
                                 dimension_features: pd.DataFrame,
                                 dimension_name: str,
                                 pattern_type: PriceMovementPattern) -> EconomicRelevanceResult:
        """Analyze pattern prediction capability of dimension."""

        self.logger.info(f"🎯 Analyzing pattern prediction for {dimension_name} -> {pattern_type.value}")

        patterns = self._calculate_returns_and_patterns(market_data)

        # Create pattern labels and targets
        pattern_labels, pattern_magnitudes = self._create_pattern_labels(market_data, pattern_type)

        # Prepare features
        feature_matrix = self._prepare_feature_matrix(dimension_features, patterns)

        # Analyze predictive capabilities
        prediction_metrics = {}

        # 1. Pattern occurrence prediction (classification)
        occurrence_results = self._analyze_pattern_occurrence_prediction(
            feature_matrix, pattern_labels, dimension_name
        )
        prediction_metrics[EconomicRelevanceMetric.PATTERN_PREDICTION_ACCURACY] = occurrence_results['accuracy']

        # 2. Pattern timing precision
        timing_results = self._analyze_pattern_timing_precision(
            feature_matrix, pattern_labels, pattern_magnitudes
        )
        prediction_metrics[EconomicRelevanceMetric.PATTERN_TIMING_PRECISION] = timing_results['precision']

        # 3. Pattern magnitude correlation
        magnitude_results = self._analyze_pattern_magnitude_correlation(
            feature_matrix, pattern_magnitudes
        )
        prediction_metrics[EconomicRelevanceMetric.PATTERN_MAGNITUDE_CORRELATION] = magnitude_results['correlation']

        # Statistical significance
        statistical_significance = {
            'prediction_p_value': occurrence_results.get('p_value', 1.0),
            'timing_p_value': timing_results.get('p_value', 1.0),
            'magnitude_p_value': magnitude_results.get('p_value', 1.0)
        }

        # Robustness tests
        robustness_tests = self._conduct_prediction_robustness_tests(
            feature_matrix, pattern_labels, pattern_magnitudes
        )

        # Economic interpretation
        economic_interpretation = self._generate_prediction_interpretation(
            dimension_name, pattern_type, prediction_metrics, statistical_significance
        )

        trading_implications = self._generate_prediction_trading_implications(
            dimension_name, pattern_type, prediction_metrics
        )

        return EconomicRelevanceResult(
            dimension_name=dimension_name,
            pattern_type=pattern_type,
            relevance_metrics=prediction_metrics,
            statistical_significance=statistical_significance,
            economic_interpretation=economic_interpretation,
            trading_implications=trading_implications,
            robustness_tests=robustness_tests,
            confidence_intervals={},
            methodology_metadata={'methodology': 'pattern_prediction', 'version': '1.0'}
        )

    def _create_pattern_labels(self,
                             market_data: pd.DataFrame,
                             pattern_type: PriceMovementPattern) -> Tuple[pd.Series, pd.Series]:
        """Create binary labels and magnitude measures for specific patterns."""

        prices = market_data['close']
        returns = prices.pct_change().fillna(0)

        if pattern_type == PriceMovementPattern.TREND_CONTINUATION:
            # Label periods where trend continues
            ma_short = prices.rolling(10).mean()
            ma_long = prices.rolling(50).mean()
            trend = np.where(ma_short > ma_long, 1, -1)

            # Trend continuation = same trend direction for next 5 periods
            labels = pd.Series(index=prices.index, dtype=float)
            magnitudes = pd.Series(index=prices.index, dtype=float)

            for i in range(len(trend) - 5):
                current_trend = trend[i]
                future_trends = trend[i+1:i+6]
                continuation = np.sum(future_trends == current_trend) >= 4  # 4 out of 5

                labels.iloc[i] = 1.0 if continuation else 0.0
                if continuation:
                    magnitudes.iloc[i] = abs(returns.iloc[i+1:i+6].sum())
                else:
                    magnitudes.iloc[i] = 0.0

        elif pattern_type == PriceMovementPattern.BREAKOUT_ACCELERATION:
            # Label periods where breakouts are followed by acceleration
            volatility = returns.rolling(20).std()
            bb_upper = prices.rolling(20).mean() + 2 * prices.rolling(20).std()
            bb_lower = prices.rolling(20).mean() - 2 * prices.rolling(20).std()

            labels = pd.Series(0.0, index=prices.index)
            magnitudes = pd.Series(0.0, index=prices.index)

            for i in range(20, len(prices) - 5):
                # Check if price breaks Bollinger Band
                current_price = prices.iloc[i]
                upper_break = current_price > bb_upper.iloc[i]
                lower_break = current_price < bb_lower.iloc[i]

                if upper_break or lower_break:
                    # Check for acceleration in next 5 periods
                    future_returns = returns.iloc[i+1:i+6]
                    if upper_break and future_returns.sum() > 0.02:  # 2% move up
                        labels.iloc[i] = 1.0
                        magnitudes.iloc[i] = future_returns.sum()
                    elif lower_break and future_returns.sum() < -0.02:  # 2% move down
                        labels.iloc[i] = 1.0
                        magnitudes.iloc[i] = abs(future_returns.sum())

        elif pattern_type == PriceMovementPattern.VOLATILITY_EXPANSION:
            # Label periods before volatility expansion
            volatility = returns.rolling(20).std()
            vol_percentile = volatility.rolling(100).rank(pct=True)

            labels = pd.Series(0.0, index=prices.index)
            magnitudes = pd.Series(0.0, index=prices.index)

            for i in range(100, len(volatility) - 10):
                current_vol = vol_percentile.iloc[i]
                future_vol = vol_percentile.iloc[i+5:i+15].max()  # Max vol in next 10 periods

                if current_vol < 0.5 and future_vol > 0.8:  # Low vol followed by high vol
                    labels.iloc[i] = 1.0
                    magnitudes.iloc[i] = future_vol - current_vol

        else:
            # Default: use return-based patterns
            labels = pd.Series((abs(returns) > returns.rolling(50).quantile(0.8)).astype(float))
            magnitudes = abs(returns)

        return labels.fillna(0), magnitudes.fillna(0)

    def _prepare_feature_matrix(self,
                              dimension_features: pd.DataFrame,
                              patterns: Dict[str, pd.Series]) -> pd.DataFrame:
        """Prepare feature matrix combining dimension features with pattern context."""

        feature_matrix = dimension_features.copy()

        # Add pattern context features
        feature_matrix['returns'] = patterns['returns']
        feature_matrix['volatility'] = patterns['volatility']
        feature_matrix['momentum_5'] = patterns['momentum_5']
        feature_matrix['momentum_20'] = patterns['momentum_20']

        # Add interaction features
        for dim_col in dimension_features.columns:
            feature_matrix[f'{dim_col}_x_volatility'] = dimension_features[dim_col] * patterns['volatility']
            feature_matrix[f'{dim_col}_x_momentum'] = dimension_features[dim_col] * patterns['momentum_5']

        return feature_matrix.fillna(0)

    def _analyze_pattern_occurrence_prediction(self,
                                             feature_matrix: pd.DataFrame,
                                             pattern_labels: pd.Series,
                                             dimension_name: str) -> Dict[str, float]:
        """Analyze how well dimension features predict pattern occurrence."""

        # Align data
        aligned_data = pd.concat([feature_matrix, pattern_labels], axis=1).dropna()
        if len(aligned_data) < 100:
            return {'accuracy': 0.5, 'p_value': 1.0}

        X = aligned_data.iloc[:, :-1].values
        y = aligned_data.iloc[:, -1].values

        # Check if we have positive cases
        if y.sum() < 10:
            return {'accuracy': 0.5, 'p_value': 1.0}

        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Multiple models for robustness
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'elastic_net': ElasticNetCV(cv=3, random_state=42)
            }

            model_scores = []

            for model_name, model in models.items():
                cv_scores = []

                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Fit model
                    model.fit(X_train_scaled, y_train)

                    # Predict
                    y_pred = model.predict(X_test_scaled)

                    # Classification accuracy (threshold at 0.5)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    y_test_binary = (y_test > 0.5).astype(int)

                    accuracy = np.mean(y_pred_binary == y_test_binary)
                    cv_scores.append(accuracy)

                model_scores.append(np.mean(cv_scores))

            # Best model accuracy
            best_accuracy = max(model_scores)

            # Statistical significance test (permutation test)
            n_permutations = 100
            null_scores = []

            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y)
                permutation_scores = []

                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train_perm, y_test_perm = y_permuted[train_idx], y_permuted[test_idx]

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_train_scaled, y_train_perm)

                    y_pred = model.predict(X_test_scaled)
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    y_test_binary = (y_test_perm > 0.5).astype(int)

                    accuracy = np.mean(y_pred_binary == y_test_binary)
                    permutation_scores.append(accuracy)

                null_scores.append(np.mean(permutation_scores))

            # P-value: fraction of null scores >= observed score
            p_value = np.mean(np.array(null_scores) >= best_accuracy)

            return {
                'accuracy': float(best_accuracy),
                'p_value': float(p_value),
                'model_scores': model_scores
            }

        except Exception as e:
            self.logger.warning(f"Pattern occurrence prediction failed: {e}")
            return {'accuracy': 0.5, 'p_value': 1.0}

    def _analyze_pattern_timing_precision(self,
                                        feature_matrix: pd.DataFrame,
                                        pattern_labels: pd.Series,
                                        pattern_magnitudes: pd.Series) -> Dict[str, float]:
        """Analyze precision of pattern timing prediction."""

        # Find pattern occurrences
        pattern_occurrences = pattern_labels[pattern_labels > 0.5].index

        if len(pattern_occurrences) < 10:
            return {'precision': 0.0, 'p_value': 1.0}

        # For each occurrence, check if dimension signal was elevated beforehand
        timing_scores = []

        for occurrence_time in pattern_occurrences:
            occurrence_idx = feature_matrix.index.get_loc(occurrence_time)

            if occurrence_idx >= 10:  # Need lookback window
                # Get dimension signal strength in 10 periods before
                lookback_signals = []
                for lookback in range(1, 11):
                    if occurrence_idx - lookback >= 0:
                        signal_strength = np.mean(np.abs(feature_matrix.iloc[occurrence_idx - lookback]))
                        lookback_signals.append(signal_strength)

                if lookback_signals:
                    # Check if signal was elevated (top 30%) in the 5 periods before
                    recent_signals = lookback_signals[:5]  # Last 5 periods
                    signal_percentile = np.percentile(feature_matrix.abs().mean(axis=1), 70)

                    elevated_periods = sum(1 for s in recent_signals if s > signal_percentile)
                    timing_precision = elevated_periods / len(recent_signals)
                    timing_scores.append(timing_precision)

        if not timing_scores:
            return {'precision': 0.0, 'p_value': 1.0}

        avg_timing_precision = np.mean(timing_scores)

        # Statistical significance (one-sample t-test against random chance)
        _, p_value = stats.ttest_1samp(timing_scores, 0.3)  # Test against 30% (random chance)

        return {
            'precision': float(avg_timing_precision),
            'p_value': float(p_value),
            'n_patterns': len(pattern_occurrences)
        }

    def _analyze_pattern_magnitude_correlation(self,
                                             feature_matrix: pd.DataFrame,
                                             pattern_magnitudes: pd.Series) -> Dict[str, float]:
        """Analyze correlation between dimension signals and pattern magnitudes."""

        # Align data
        aligned_data = pd.concat([feature_matrix.mean(axis=1), pattern_magnitudes], axis=1).dropna()

        if len(aligned_data) < 50:
            return {'correlation': 0.0, 'p_value': 1.0}

        # Only consider periods with non-zero pattern magnitudes
        non_zero_patterns = aligned_data[aligned_data.iloc[:, 1] > 0]

        if len(non_zero_patterns) < 20:
            return {'correlation': 0.0, 'p_value': 1.0}

        # Calculate correlation
        correlation, p_value = stats.pearsonr(
            non_zero_patterns.iloc[:, 0],  # dimension signal
            non_zero_patterns.iloc[:, 1]   # pattern magnitude
        )

        return {
            'correlation': float(abs(correlation)),  # Use absolute value
            'p_value': float(p_value),
            'n_observations': len(non_zero_patterns)
        }

    def _conduct_prediction_robustness_tests(self,
                                           feature_matrix: pd.DataFrame,
                                           pattern_labels: pd.Series,
                                           pattern_magnitudes: pd.Series) -> Dict[str, float]:
        """Conduct robustness tests for prediction methodology."""

        robustness_results = {}

        # 1. Out-of-sample stability
        aligned_data = pd.concat([feature_matrix, pattern_labels], axis=1).dropna()

        if len(aligned_data) > 200:
            # Split into multiple time periods
            n_periods = 4
            period_size = len(aligned_data) // n_periods
            period_accuracies = []

            for i in range(n_periods - 1):
                start_idx = i * period_size
                end_idx = (i + 2) * period_size  # Use next period for testing

                period_data = aligned_data.iloc[start_idx:end_idx]
                split_point = len(period_data) // 2

                train_data = period_data.iloc[:split_point]
                test_data = period_data.iloc[split_point:]

                if len(train_data) > 50 and len(test_data) > 20:
                    X_train = train_data.iloc[:, :-1].values
                    y_train = train_data.iloc[:, -1].values
                    X_test = test_data.iloc[:, :-1].values
                    y_test = test_data.iloc[:, -1].values

                    try:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        model.fit(X_train_scaled, y_train)

                        y_pred = model.predict(X_test_scaled)
                        y_pred_binary = (y_pred > 0.5).astype(int)
                        y_test_binary = (y_test > 0.5).astype(int)

                        accuracy = np.mean(y_pred_binary == y_test_binary)
                        period_accuracies.append(accuracy)
                    except:
                        pass

            if period_accuracies:
                robustness_results['out_of_sample_stability'] = float(1.0 - np.std(period_accuracies))

        # 2. Feature importance consistency
        try:
            X = aligned_data.iloc[:, :-1].values
            y = aligned_data.iloc[:, -1].values

            # Multiple bootstrap samples
            importance_vectors = []
            n_bootstrap = 10

            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]

                scaler = StandardScaler()
                X_bootstrap_scaled = scaler.fit_transform(X_bootstrap)

                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_bootstrap_scaled, y_bootstrap)

                importance_vectors.append(model.feature_importances_)

            # Calculate consistency (inverse of average pairwise correlation)
            if len(importance_vectors) > 1:
                correlations = []
                for i in range(len(importance_vectors)):
                    for j in range(i + 1, len(importance_vectors)):
                        corr, _ = stats.pearsonr(importance_vectors[i], importance_vectors[j])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

                if correlations:
                    robustness_results['feature_importance_consistency'] = float(np.mean(correlations))

        except Exception as e:
            self.logger.warning(f"Feature importance consistency test failed: {e}")

        return robustness_results

    def _generate_prediction_interpretation(self,
                                          dimension_name: str,
                                          pattern_type: PriceMovementPattern,
                                          prediction_metrics: Dict[EconomicRelevanceMetric, float],
                                          statistical_significance: Dict[str, float]) -> str:
        """Generate interpretation of prediction analysis results."""

        accuracy = prediction_metrics.get(EconomicRelevanceMetric.PATTERN_PREDICTION_ACCURACY, 0.5)
        timing = prediction_metrics.get(EconomicRelevanceMetric.PATTERN_TIMING_PRECISION, 0)
        correlation = prediction_metrics.get(EconomicRelevanceMetric.PATTERN_MAGNITUDE_CORRELATION, 0)

        if accuracy > 0.65 and timing > 0.5:
            strength = "strong"
        elif accuracy > 0.58 and timing > 0.4:
            strength = "moderate"
        elif accuracy > 0.53:
            strength = "weak"
        else:
            strength = "negligible"

        interpretation = f"{dimension_name} shows {strength} predictive power for {pattern_type.value} "
        interpretation += f"(accuracy: {accuracy:.3f}, timing precision: {timing:.3f}, magnitude correlation: {correlation:.3f})"

        return interpretation

    def _generate_prediction_trading_implications(self,
                                                dimension_name: str,
                                                pattern_type: PriceMovementPattern,
                                                prediction_metrics: Dict[EconomicRelevanceMetric, float]) -> str:
        """Generate trading implications from prediction analysis."""

        accuracy = prediction_metrics.get(EconomicRelevanceMetric.PATTERN_PREDICTION_ACCURACY, 0.5)
        timing = prediction_metrics.get(EconomicRelevanceMetric.PATTERN_TIMING_PRECISION, 0)

        if accuracy > 0.6 and timing > 0.4:
            return f"Strong predictive signal: Use {dimension_name} for {pattern_type.value} prediction with high confidence"
        elif accuracy > 0.55:
            return f"Moderate predictive signal: Use {dimension_name} as supporting indicator for {pattern_type.value}"
        else:
            return f"Limited predictive value: {dimension_name} not suitable for {pattern_type.value} prediction"

# Factory function to create research methodologies
def create_research_methodology(methodology_type: str,
                              config: ResearchMethodologyConfig) -> BaseResearchMethodology:
    """Create research methodology instance."""

    if methodology_type == "causal_impact":
        return CausalImpactResearchMethodology(config)
    elif methodology_type == "pattern_prediction":
        return PatternPredictionResearchMethodology(config)
    else:
        raise ValueError(f"Unknown methodology type: {methodology_type}")

# Main research orchestrator
class EconomicRelevanceResearchOrchestrator:
    """
    Main orchestrator for conducting comprehensive economic relevance research.
    """

    def __init__(self, config: Optional[ResearchMethodologyConfig] = None):
        self.config = config or ResearchMethodologyConfig()
        self.logger = system_logger.getChild('EconomicRelevanceResearch')

        # Initialize methodologies
        self.methodologies = {
            'causal_impact': create_research_methodology('causal_impact', self.config),
            'pattern_prediction': create_research_methodology('pattern_prediction', self.config)
        }

    def conduct_comprehensive_research(self,
                                     market_data: pd.DataFrame,
                                     dimension_feature_groups: Dict[str, pd.DataFrame],
                                     patterns_to_analyze: List[PriceMovementPattern] = None) -> Dict[str, Dict[str, EconomicRelevanceResult]]:
        """
        Conduct comprehensive economic relevance research across all dimensions and patterns.

        Args:
            market_data: OHLCV market data
            dimension_feature_groups: Dictionary mapping dimension names to feature DataFrames
            patterns_to_analyze: List of patterns to analyze (default: all major patterns)

        Returns:
            Nested dictionary: {dimension_name: {pattern_type: EconomicRelevanceResult}}
        """

        if patterns_to_analyze is None:
            patterns_to_analyze = [
                PriceMovementPattern.MOMENTUM_PERSISTENCE,
                PriceMovementPattern.MEAN_REVERSION_SPEED,
                PriceMovementPattern.VOLATILITY_EXPANSION,
                PriceMovementPattern.BREAKOUT_ACCELERATION,
                PriceMovementPattern.TREND_CONTINUATION
            ]

        self.logger.info(f"🔬 Starting comprehensive economic relevance research")
        self.logger.info(f"   - Dimensions: {list(dimension_feature_groups.keys())}")
        self.logger.info(f"   - Patterns: {[p.value for p in patterns_to_analyze]}")
        self.logger.info(f"   - Methodologies: {list(self.methodologies.keys())}")

        results = {}

        for dimension_name, dimension_features in dimension_feature_groups.items():
            self.logger.info(f"📊 Analyzing dimension: {dimension_name}")

            dimension_results = {}

            for pattern_type in patterns_to_analyze:
                self.logger.info(f"   🎯 Pattern: {pattern_type.value}")

                pattern_results = {}

                for methodology_name, methodology in self.methodologies.items():
                    try:
                        result = methodology.analyze_economic_relevance(
                            market_data, dimension_features, dimension_name, pattern_type
                        )
                        pattern_results[methodology_name] = result

                        # Log key finding
                        if result.is_economically_relevant:
                            self.logger.info(f"   ✅ {methodology_name}: Economically relevant!")
                        else:
                            self.logger.info(f"   ❌ {methodology_name}: Not economically relevant")

                    except Exception as e:
                        self.logger.error(f"   ⚠️ {methodology_name} failed: {e}")
                        continue

                if pattern_results:
                    dimension_results[pattern_type.value] = pattern_results

            if dimension_results:
                results[dimension_name] = dimension_results

        self.logger.info(f"✅ Comprehensive research completed: {len(results)} dimensions analyzed")
        return results

    def generate_research_report(self,
                               research_results: Dict[str, Dict[str, Dict[str, EconomicRelevanceResult]]]) -> str:
        """Generate comprehensive research report."""

        report = []
        report.append("# Economic Relevance Research Report")
        report.append("=" * 60)
        report.append("")

        # Executive Summary
        total_tests = 0
        economically_relevant_tests = 0

        for dimension_results in research_results.values():
            for pattern_results in dimension_results.values():
                for methodology_result in pattern_results.values():
                    total_tests += 1
                    if methodology_result.is_economically_relevant:
                        economically_relevant_tests += 1

        relevance_rate = (economically_relevant_tests / total_tests * 100) if total_tests > 0 else 0

        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Total Dimension-Pattern-Methodology Combinations**: {total_tests}")
        report.append(f"- **Economically Relevant Results**: {economically_relevant_tests}")
        report.append(f"- **Economic Relevance Rate**: {relevance_rate:.1f}%")
        report.append("")

        # Key Findings
        report.append("## Key Economic Relevance Findings")
        report.append("")

        # Find most economically relevant dimensions
        dimension_relevance_scores = {}

        for dimension_name, dimension_results in research_results.items():
            relevance_count = 0
            total_count = 0

            for pattern_results in dimension_results.values():
                for methodology_result in pattern_results.values():
                    total_count += 1
                    if methodology_result.is_economically_relevant:
                        relevance_count += 1

            if total_count > 0:
                dimension_relevance_scores[dimension_name] = relevance_count / total_count

        # Sort by relevance
        sorted_dimensions = sorted(dimension_relevance_scores.items(), key=lambda x: x[1], reverse=True)

        report.append("### Dimension Relevance Ranking")
        report.append("")

        for i, (dimension_name, relevance_score) in enumerate(sorted_dimensions, 1):
            status = "🟢" if relevance_score > 0.7 else "🟡" if relevance_score > 0.4 else "🔴"
            report.append(f"{i}. {status} **{dimension_name.upper()}** - Relevance Rate: {relevance_score:.1%}")

        report.append("")

        # Detailed Results by Dimension
        report.append("## Detailed Results by Dimension")
        report.append("")

        for dimension_name, dimension_results in research_results.items():
            report.append(f"### {dimension_name.upper()} Dimension")
            report.append("")

            for pattern_name, pattern_results in dimension_results.items():
                report.append(f"#### {pattern_name.replace('_', ' ').title()}")
                report.append("")

                for methodology_name, result in pattern_results.items():
                    status = "✅" if result.is_economically_relevant else "❌"

                    report.append(f"**{methodology_name.replace('_', ' ').title()}** {status}")
                    report.append(f"- {result.economic_interpretation}")
                    report.append(f"- Trading Implications: {result.trading_implications}")

                    # Key metrics
                    for metric, value in result.relevance_metrics.items():
                        report.append(f"- {metric.value.replace('_', ' ').title()}: {value:.3f}")

                    report.append("")

        # Research Methodology Summary
        report.append("## Research Methodology Summary")
        report.append("")

        report.append("### Causal Impact Analysis")
        report.append("- Uses Granger Causality and Instrumental Variables")
        report.append("- Establishes causal relationships between dimensions and price patterns")
        report.append("- Tests robustness through subsample stability and noise resilience")
        report.append("")

        report.append("### Pattern Prediction Analysis")
        report.append("- Uses ML models to predict pattern occurrence and timing")
        report.append("- Measures predictive accuracy and magnitude correlation")
        report.append("- Tests out-of-sample stability and feature importance consistency")
        report.append("")

        # Recommendations
        report.append("## Research-Based Recommendations")
        report.append("")

        if relevance_rate > 60:
            report.append("✅ **Strong Economic Foundation Detected**")
            report.append("- Multiple dimensions show significant economic relevance")
            report.append("- Proceed with dimension-based regime modeling")
            report.append("- Focus on top-ranked dimensions for ML model training")
        elif relevance_rate > 30:
            report.append("⚠️ **Moderate Economic Foundation**")
            report.append("- Some dimensions show economic relevance")
            report.append("- Selective use of economically relevant dimensions")
            report.append("- Consider combining multiple methodologies for validation")
        else:
            report.append("❌ **Limited Economic Foundation**")
            report.append("- Few dimensions show clear economic relevance")
            report.append("- Consider alternative feature engineering approaches")
            report.append("- May need to focus on volume/volatility dimensions primarily")

        report.append("")
        report.append("## Next Steps")
        report.append("")
        report.append("1. **Implement top-ranked dimensions** in regime discovery pipeline")
        report.append("2. **Validate findings** on out-of-sample data")
        report.append("3. **Develop trading strategies** based on economically relevant patterns")
        report.append("4. **Monitor performance** of dimension-based signals in live trading")

        return "\n".join(report)

# Example usage function
def run_economic_relevance_research_example():
    """Example of how to run the economic relevance research framework."""

    # This would be called with real market data and dimension features
    print("Economic Relevance Research Framework")
    print("====================================")
    print()
    print("This framework provides:")
    print("1. Causal Impact Analysis - Establishes causation vs correlation")
    print("2. Pattern Prediction Analysis - Measures predictive power for specific patterns")
    print("3. Comprehensive robustness testing")
    print("4. Economic significance validation")
    print()
    print("Key research questions answered:")
    print("- Which dimensions have CAUSAL impact on price patterns?")
    print("- How well do dimensions predict specific price movements?")
    print("- What constitutes economic vs statistical significance?")
    print("- Which patterns are exploitable for trading?")
    print()
    print("Usage:")
    print("```python")
    print("config = ResearchMethodologyConfig()")
    print("orchestrator = EconomicRelevanceResearchOrchestrator(config)")
    print("results = orchestrator.conduct_comprehensive_research(")
    print("    market_data, dimension_feature_groups")
    print(")")
    print("report = orchestrator.generate_research_report(results)")
    print("```")

if __name__ == "__main__":
    run_economic_relevance_research_example()
