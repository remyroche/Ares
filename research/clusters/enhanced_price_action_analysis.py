"""
Enhanced Price Action Influence Analysis

This module provides advanced analysis of how feature clusters influence price action,
going beyond simple correlation to understand the mechanisms of price movement.

Key Research Focus:
- What specific price patterns exist in the data?
- How do feature clusters influence different types of price action?
- What's the relationship between feature homogeneity and price predictive power?
- Which feature interactions drive the strongest price responses?

This connects to the broader research on "what price action means" by providing
empirical analysis of price patterns and their relationship to feature clusters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mutual_info_regression

from src.utils.logger import system_logger


class PriceActionPattern(Enum):
    """Types of price action patterns to analyze."""
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    CONSOLIDATION = "consolidation"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    MOMENTUM_DECELERATION = "momentum_deceleration"
    MEAN_REVERSION = "mean_reversion"


class InfluenceMechanism(Enum):
    """Mechanisms by which features influence price action."""
    DIRECT_CORRELATION = "direct_correlation"
    LAGGED_INFLUENCE = "lagged_influence"
    THRESHOLD_EFFECT = "threshold_effect"
    INTERACTION_EFFECT = "interaction_effect"
    REGIME_DEPENDENT = "regime_dependent"
    VOLATILITY_MODULATION = "volatility_modulation"
    MOMENTUM_AMPLIFICATION = "momentum_amplification"


@dataclass
class PriceActionInfluenceResult:
    """Result container for price action influence analysis."""
    pattern: PriceActionPattern
    influence_strength: float
    mechanism: InfluenceMechanism
    feature_contributions: Dict[str, float]
    statistical_significance: float
    economic_significance: float
    cluster_specific_influence: Dict[int, float]
    predictive_horizon: int
    confidence_interval: Tuple[float, float]
    metadata: Dict[str, Any]


@dataclass
class FeaturePriceInteractionConfig:
    """Configuration for feature-price interaction analysis."""
    # Price pattern detection
    trend_window: int = 20
    volatility_window: int = 20
    momentum_window: int = 10
    
    # Influence analysis
    prediction_horizons: List[int] = None  # [1, 3, 5, 10, 20]
    lag_analysis_periods: int = 10
    
    # Statistical validation
    significance_level: float = 0.05
    bootstrap_samples: int = 100
    cross_validation_folds: int = 5
    
    # Economic thresholds
    min_influence_strength: float = 0.1
    min_predictive_power: float = 0.05
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 3, 5, 10, 20]


class EnhancedPriceActionAnalyzer:
    """
    Enhanced analyzer for price action influence.
    
    This class provides advanced analysis of how feature clusters influence
    different types of price action patterns and mechanisms.
    """
    
    def __init__(self, config: Optional[FeaturePriceInteractionConfig] = None):
        self.config = config or FeaturePriceInteractionConfig()
        self.logger = system_logger.getChild('EnhancedPriceActionAnalyzer')
    
    def analyze_price_action_influence(self,
                                     features: pd.DataFrame,
                                     price_data: pd.DataFrame,
                                     cluster_labels: np.ndarray) -> Dict[PriceActionPattern, PriceActionInfluenceResult]:
        """
        Comprehensive analysis of how feature clusters influence price action.
        
        Args:
            features: Feature matrix
            price_data: Price data (OHLCV)
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary mapping price patterns to influence results
        """
        self.logger.info("🔍 Starting enhanced price action influence analysis")
        
        # Detect price action patterns
        price_patterns = self._detect_price_patterns(price_data)
        
        # Analyze influence for each pattern
        influence_results = {}
        
        for pattern in PriceActionPattern:
            if pattern.value in price_patterns:
                self.logger.info(f"📊 Analyzing {pattern.value} influence")
                
                influence_result = self._analyze_pattern_influence(
                    features, price_data, cluster_labels, pattern, price_patterns[pattern.value]
                )
                
                influence_results[pattern] = influence_result
        
        self.logger.info(f"✅ Price action influence analysis completed for {len(influence_results)} patterns")
        return influence_results
    
    def _detect_price_patterns(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect various price action patterns in the data."""
        
        patterns = {}
        
        if 'close' not in price_data.columns:
            self.logger.warning("No close price data available for pattern detection")
            return patterns
        
        prices = price_data['close']
        returns = prices.pct_change().fillna(0)
        
        # Trend patterns
        sma_short = prices.rolling(self.config.momentum_window).mean()
        sma_long = prices.rolling(self.config.trend_window).mean()
        
        patterns['trend_continuation'] = (
            ((prices > sma_short) & (sma_short > sma_long) & (returns > 0)) |  # Uptrend continuation
            ((prices < sma_short) & (sma_short < sma_long) & (returns < 0))    # Downtrend continuation
        ).astype(float)
        
        patterns['trend_reversal'] = (
            ((prices < sma_short) & (sma_short > sma_long)) |  # Reversal from uptrend
            ((prices > sma_short) & (sma_short < sma_long))    # Reversal from downtrend
        ).astype(float)
        
        # Breakout patterns
        if all(col in price_data.columns for col in ['high', 'low']):
            rolling_high = price_data['high'].rolling(self.config.trend_window).max()
            rolling_low = price_data['low'].rolling(self.config.trend_window).min()
            
            patterns['breakout'] = (prices > rolling_high.shift(1)).astype(float)
            patterns['breakdown'] = (prices < rolling_low.shift(1)).astype(float)
            
            # Consolidation (price within recent range)
            range_size = (rolling_high - rolling_low) / prices
            patterns['consolidation'] = (range_size < range_size.rolling(50).quantile(0.3)).astype(float)
        
        # Volatility patterns
        volatility = returns.rolling(self.config.volatility_window).std()
        vol_ma = volatility.rolling(self.config.volatility_window).mean()
        
        patterns['volatility_expansion'] = (volatility > vol_ma * 1.5).astype(float)
        patterns['volatility_contraction'] = (volatility < vol_ma * 0.7).astype(float)
        
        # Momentum patterns
        momentum = returns.rolling(self.config.momentum_window).mean()
        momentum_change = momentum.diff()
        
        patterns['momentum_acceleration'] = (
            ((momentum > 0) & (momentum_change > 0)) |  # Positive momentum accelerating
            ((momentum < 0) & (momentum_change < 0))    # Negative momentum accelerating
        ).astype(float)
        
        patterns['momentum_deceleration'] = (
            ((momentum > 0) & (momentum_change < 0)) |  # Positive momentum decelerating
            ((momentum < 0) & (momentum_change > 0))    # Negative momentum decelerating
        ).astype(float)
        
        # Mean reversion patterns
        price_zscore = (prices - sma_long) / prices.rolling(self.config.trend_window).std()
        patterns['mean_reversion'] = (abs(price_zscore) > 1.5).astype(float)
        
        # Remove NaN values
        for pattern_name in patterns:
            patterns[pattern_name] = patterns[pattern_name].fillna(0)
        
        self.logger.info(f"📊 Detected {len(patterns)} price action patterns")
        for pattern_name, pattern_series in patterns.items():
            occurrence_rate = pattern_series.mean()
            self.logger.info(f"   - {pattern_name}: {occurrence_rate:.1%} occurrence rate")
        
        return patterns
    
    def _analyze_pattern_influence(self,
                                 features: pd.DataFrame,
                                 price_data: pd.DataFrame,
                                 cluster_labels: np.ndarray,
                                 pattern: PriceActionPattern,
                                 pattern_series: pd.Series) -> PriceActionInfluenceResult:
        """Analyze how feature clusters influence a specific price pattern."""
        
        # Align data lengths
        min_len = min(len(features), len(pattern_series), len(cluster_labels))
        features_aligned = features.iloc[:min_len]
        pattern_aligned = pattern_series.iloc[:min_len]
        labels_aligned = cluster_labels[:min_len]
        
        # Analyze cluster-specific influence
        cluster_influences = self._analyze_cluster_specific_influence(
            features_aligned, pattern_aligned, labels_aligned
        )
        
        # Analyze feature contributions
        feature_contributions = self._analyze_feature_contributions(
            features_aligned, pattern_aligned, labels_aligned
        )
        
        # Determine primary influence mechanism
        mechanism = self._determine_influence_mechanism(
            features_aligned, pattern_aligned, labels_aligned
        )
        
        # Calculate overall influence strength
        influence_strength = self._calculate_influence_strength(
            features_aligned, pattern_aligned, labels_aligned
        )
        
        # Statistical significance
        statistical_significance = self._calculate_statistical_significance(
            features_aligned, pattern_aligned, labels_aligned
        )
        
        # Economic significance
        economic_significance = self._calculate_economic_significance(
            features_aligned, pattern_aligned, price_data.iloc[:min_len]
        )
        
        return PriceActionInfluenceResult(
            pattern=pattern,
            influence_strength=influence_strength,
            mechanism=mechanism,
            feature_contributions=feature_contributions,
            statistical_significance=statistical_significance,
            economic_significance=economic_significance,
            cluster_specific_influence=cluster_influences,
            predictive_horizon=1,  # Default horizon
            confidence_interval=(0.0, 0.0),  # TODO: Implement bootstrap CI
            metadata={
                'pattern_occurrence_rate': float(pattern_aligned.mean()),
                'n_samples': min_len,
                'n_clusters': len(np.unique(labels_aligned))
            }
        )
    
    def _analyze_cluster_specific_influence(self,
                                          features: pd.DataFrame,
                                          pattern_series: pd.Series,
                                          cluster_labels: np.ndarray) -> Dict[int, float]:
        """Analyze how each cluster influences the price pattern."""
        
        cluster_influences = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            
            if np.sum(cluster_mask) < 10:  # Need minimum samples
                cluster_influences[cluster_id] = 0.0
                continue
            
            # Calculate pattern occurrence rate in this cluster
            cluster_pattern_rate = pattern_series[cluster_mask].mean()
            overall_pattern_rate = pattern_series.mean()
            
            # Influence = how much cluster deviates from overall rate
            if overall_pattern_rate > 0:
                influence = (cluster_pattern_rate - overall_pattern_rate) / overall_pattern_rate
            else:
                influence = 0.0
            
            cluster_influences[cluster_id] = float(influence)
        
        return cluster_influences
    
    def _analyze_feature_contributions(self,
                                     features: pd.DataFrame,
                                     pattern_series: pd.Series,
                                     cluster_labels: np.ndarray) -> Dict[str, float]:
        """Analyze individual feature contributions to price pattern influence."""
        
        feature_contributions = {}
        
        try:
            # Use mutual information to measure feature-pattern relationships
            for col in features.columns:
                feature_values = features[col].fillna(0)
                
                if feature_values.std() > 0:
                    # Calculate mutual information
                    mi_score = mutual_info_regression(
                        feature_values.values.reshape(-1, 1),
                        pattern_series.values,
                        discrete_features=False,
                        random_state=42
                    )[0]
                    
                    feature_contributions[col] = float(mi_score)
                else:
                    feature_contributions[col] = 0.0
        
        except Exception as e:
            self.logger.warning(f"Feature contribution analysis failed: {e}")
            # Fallback to correlation
            for col in features.columns:
                try:
                    corr = abs(np.corrcoef(features[col].fillna(0), pattern_series)[0, 1])
                    feature_contributions[col] = corr if not np.isnan(corr) else 0.0
                except:
                    feature_contributions[col] = 0.0
        
        return feature_contributions
    
    def _determine_influence_mechanism(self,
                                     features: pd.DataFrame,
                                     pattern_series: pd.Series,
                                     cluster_labels: np.ndarray) -> InfluenceMechanism:
        """Determine the primary mechanism by which features influence price action."""
        
        mechanisms_scores = {}
        
        # Test direct correlation
        feature_means = features.groupby(cluster_labels).mean()
        if len(feature_means) > 1:
            # Calculate correlation between cluster means and pattern rates
            cluster_pattern_rates = pattern_series.groupby(cluster_labels).mean()
            
            correlations = []
            for col in features.columns:
                if feature_means[col].std() > 0:
                    corr = abs(np.corrcoef(feature_means[col], cluster_pattern_rates)[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            mechanisms_scores[InfluenceMechanism.DIRECT_CORRELATION] = np.mean(correlations) if correlations else 0.0
        else:
            mechanisms_scores[InfluenceMechanism.DIRECT_CORRELATION] = 0.0
        
        # Test lagged influence
        lagged_influence_scores = []
        for lag in range(1, min(6, len(pattern_series) // 10)):
            try:
                lagged_pattern = pattern_series.shift(-lag).fillna(0)
                feature_mean = features.mean(axis=1)
                corr = abs(np.corrcoef(feature_mean, lagged_pattern)[0, 1])
                if not np.isnan(corr):
                    lagged_influence_scores.append(corr)
            except:
                continue
        
        mechanisms_scores[InfluenceMechanism.LAGGED_INFLUENCE] = np.mean(lagged_influence_scores) if lagged_influence_scores else 0.0
        
        # Test threshold effects
        try:
            feature_mean = features.mean(axis=1)
            threshold_75 = feature_mean.quantile(0.75)
            threshold_25 = feature_mean.quantile(0.25)
            
            high_feature_pattern_rate = pattern_series[feature_mean > threshold_75].mean()
            low_feature_pattern_rate = pattern_series[feature_mean < threshold_25].mean()
            
            threshold_effect = abs(high_feature_pattern_rate - low_feature_pattern_rate)
            mechanisms_scores[InfluenceMechanism.THRESHOLD_EFFECT] = threshold_effect
        except:
            mechanisms_scores[InfluenceMechanism.THRESHOLD_EFFECT] = 0.0
        
        # Test interaction effects (simplified)
        try:
            if len(features.columns) >= 2:
                # Test pairwise feature interactions
                interaction_scores = []
                for i in range(min(5, len(features.columns))):
                    for j in range(i+1, min(5, len(features.columns))):
                        interaction = features.iloc[:, i] * features.iloc[:, j]
                        corr = abs(np.corrcoef(interaction, pattern_series)[0, 1])
                        if not np.isnan(corr):
                            interaction_scores.append(corr)
                
                mechanisms_scores[InfluenceMechanism.INTERACTION_EFFECT] = np.mean(interaction_scores) if interaction_scores else 0.0
            else:
                mechanisms_scores[InfluenceMechanism.INTERACTION_EFFECT] = 0.0
        except:
            mechanisms_scores[InfluenceMechanism.INTERACTION_EFFECT] = 0.0
        
        # Return mechanism with highest score
        if mechanisms_scores:
            best_mechanism = max(mechanisms_scores.keys(), key=lambda k: mechanisms_scores[k])
            return best_mechanism
        else:
            return InfluenceMechanism.DIRECT_CORRELATION
    
    def _calculate_influence_strength(self,
                                    features: pd.DataFrame,
                                    pattern_series: pd.Series,
                                    cluster_labels: np.ndarray) -> float:
        """Calculate overall influence strength of features on price pattern."""
        
        try:
            # Use Random Forest to measure predictive power
            X = features.fillna(0)
            y = pattern_series.values
            
            # Add cluster information
            cluster_features = pd.get_dummies(pd.Series(cluster_labels, name='cluster'))
            X_with_clusters = pd.concat([X, cluster_features], axis=1)
            
            # Random Forest prediction
            rf = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=5,
                n_jobs=1
            )
            
            # Cross-validation score
            cv_scores = cross_val_score(
                rf, X_with_clusters, y,
                cv=self.config.cross_validation_folds,
                scoring='r2',
                n_jobs=1
            )
            
            return max(0.0, np.mean(cv_scores))
            
        except Exception as e:
            self.logger.warning(f"Influence strength calculation failed: {e}")
            return 0.0
    
    def _calculate_statistical_significance(self,
                                          features: pd.DataFrame,
                                          pattern_series: pd.Series,
                                          cluster_labels: np.ndarray) -> float:
        """Calculate statistical significance of feature-pattern relationship."""
        
        try:
            # Test if pattern occurrence differs significantly between clusters
            unique_clusters = np.unique(cluster_labels)
            
            if len(unique_clusters) < 2:
                return 1.0  # No significance test possible
            
            cluster_pattern_rates = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 5:
                    cluster_rate = pattern_series[cluster_mask].mean()
                    cluster_pattern_rates.append(cluster_rate)
            
            if len(cluster_pattern_rates) < 2:
                return 1.0
            
            # ANOVA test
            cluster_pattern_lists = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 5:
                    cluster_pattern_lists.append(pattern_series[cluster_mask].values)
            
            if len(cluster_pattern_lists) >= 2:
                f_stat, p_value = stats.f_oneway(*cluster_pattern_lists)
                return float(p_value)
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            return 1.0
    
    def _calculate_economic_significance(self,
                                       features: pd.DataFrame,
                                       pattern_series: pd.Series,
                                       price_data: pd.DataFrame) -> float:
        """Calculate economic significance of pattern influence."""
        
        try:
            # Calculate returns associated with pattern
            if 'close' not in price_data.columns:
                return 0.0
            
            returns = price_data['close'].pct_change().fillna(0)
            
            # Returns when pattern occurs vs when it doesn't
            pattern_returns = returns[pattern_series > 0.5]
            no_pattern_returns = returns[pattern_series <= 0.5]
            
            if len(pattern_returns) > 5 and len(no_pattern_returns) > 5:
                # Calculate return difference
                pattern_mean = pattern_returns.mean()
                no_pattern_mean = no_pattern_returns.mean()
                
                return_difference = abs(pattern_mean - no_pattern_mean)
                
                # Annualize and compare to reasonable threshold
                annualized_difference = return_difference * 252
                
                # Economic significance if > 1% annual difference
                return min(1.0, annualized_difference / 0.01)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Economic significance calculation failed: {e}")
            return 0.0
    
    def analyze_feature_price_coupling_by_cv(self,
                                           features: pd.DataFrame,
                                           price_data: pd.DataFrame,
                                           cv_range: Tuple[float, float, int] = (0.1, 1.0, 20)) -> Dict[str, Any]:
        """
        Analyze how feature-price coupling changes with CV thresholds.
        
        This answers: "What's the relationship between feature homogeneity and price action influence?"
        """
        self.logger.info("🔍 Analyzing feature-price coupling vs CV relationship")
        
        cv_min, cv_max, cv_steps = cv_range
        cv_thresholds = np.linspace(cv_min, cv_max, cv_steps)
        
        coupling_results = {
            'cv_thresholds': cv_thresholds.tolist(),
            'coupling_strengths': [],
            'predictive_powers': [],
            'cluster_counts': [],
            'economic_significances': []
        }
        
        returns = price_data['close'].pct_change().fillna(0)
        
        for cv_thresh in cv_thresholds:
            try:
                # Create clusters with this CV threshold
                from .similarity_matrix_clustering import SimilarityClusteringConfig, similarity_matrix_clustering
                
                config = SimilarityClusteringConfig(
                    cv_threshold=cv_thresh,
                    similarity_threshold=0.7,  # Fixed similarity
                    min_samples_per_cluster=30
                )
                
                clustering_result = similarity_matrix_clustering(features, price_data, config)
                
                # Measure feature-price coupling
                coupling_strength = self._measure_feature_price_coupling_strength(
                    features, returns, clustering_result.labels
                )
                
                # Measure predictive power
                predictive_power = self._measure_predictive_power_simple(
                    features, returns, clustering_result.labels
                )
                
                # Measure economic significance
                economic_sig = self._measure_economic_significance_simple(
                    returns, clustering_result.labels
                )
                
                coupling_results['coupling_strengths'].append(coupling_strength)
                coupling_results['predictive_powers'].append(predictive_power)
                coupling_results['cluster_counts'].append(clustering_result.n_clusters)
                coupling_results['economic_significances'].append(economic_sig)
                
            except Exception as e:
                self.logger.warning(f"CV analysis failed for threshold {cv_thresh:.3f}: {e}")
                coupling_results['coupling_strengths'].append(0.0)
                coupling_results['predictive_powers'].append(0.0)
                coupling_results['cluster_counts'].append(1)
                coupling_results['economic_significances'].append(0.0)
        
        # Find breaking point
        coupling_strengths = np.array(coupling_results['coupling_strengths'])
        max_coupling = np.max(coupling_strengths) if len(coupling_strengths) > 0 else 0
        breaking_threshold = max_coupling * 0.7  # 30% degradation
        
        breaking_point_idx = None
        for i, coupling in enumerate(coupling_strengths):
            if coupling < breaking_threshold and max_coupling > 0:
                breaking_point_idx = i
                break
        
        breaking_point_cv = cv_thresholds[breaking_point_idx] if breaking_point_idx is not None else None
        
        analysis_result = {
            **coupling_results,
            'max_coupling_strength': float(max_coupling),
            'breaking_point_cv': breaking_point_cv,
            'coupling_cv_correlation': float(np.corrcoef(cv_thresholds, coupling_strengths)[0, 1]) if len(coupling_strengths) > 1 else 0.0
        }
        
        self.logger.info(f"📊 Feature-Price Coupling Analysis:")
        self.logger.info(f"   Max coupling strength: {max_coupling:.3f}")
        if breaking_point_cv:
            self.logger.info(f"   Breaking point CV: {breaking_point_cv:.3f}")
            self.logger.info(f"   → Feature-price coupling degrades significantly beyond CV={breaking_point_cv:.3f}")
        
        correlation = analysis_result['coupling_cv_correlation']
        if abs(correlation) > 0.3:
            direction = "decreases" if correlation < 0 else "increases"
            self.logger.info(f"   CV-Coupling correlation: {correlation:.3f} (coupling {direction} with CV)")
        
        return analysis_result
    
    def _measure_feature_price_coupling_strength(self,
                                               features: pd.DataFrame,
                                               returns: pd.Series,
                                               cluster_labels: np.ndarray) -> float:
        """Measure strength of feature-price coupling within clusters."""
        
        try:
            cluster_couplings = []
            
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                
                if np.sum(cluster_mask) < 10:
                    continue
                
                cluster_features = features[cluster_mask]
                cluster_returns = returns[cluster_mask]
                
                # Calculate feature-return correlations within cluster
                feature_return_correlations = []
                for col in cluster_features.columns:
                    if cluster_features[col].std() > 0:
                        corr = abs(np.corrcoef(cluster_features[col], cluster_returns)[0, 1])
                        if not np.isnan(corr):
                            feature_return_correlations.append(corr)
                
                if feature_return_correlations:
                    cluster_coupling = np.mean(feature_return_correlations)
                    cluster_couplings.append(cluster_coupling)
            
            return np.mean(cluster_couplings) if cluster_couplings else 0.0
            
        except Exception as e:
            self.logger.warning(f"Coupling strength measurement failed: {e}")
            return 0.0
    
    def _measure_predictive_power_simple(self,
                                       features: pd.DataFrame,
                                       returns: pd.Series,
                                       cluster_labels: np.ndarray) -> float:
        """Simple measure of predictive power."""
        
        try:
            # Predict next period returns using features and cluster info
            future_returns = returns.shift(-1).fillna(0)
            
            # Align data
            min_len = min(len(features), len(future_returns), len(cluster_labels))
            X = features.iloc[:min_len]
            y = future_returns.iloc[:min_len]
            clusters = cluster_labels[:min_len]
            
            # Add cluster as feature
            cluster_features = pd.get_dummies(pd.Series(clusters, name='cluster'))
            X_with_clusters = pd.concat([X, cluster_features], axis=1)
            
            # Random Forest prediction
            rf = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=3, n_jobs=1)
            cv_scores = cross_val_score(rf, X_with_clusters.fillna(0), y, cv=3, scoring='r2', n_jobs=1)
            
            return max(0.0, np.mean(cv_scores))
            
        except Exception as e:
            self.logger.warning(f"Predictive power measurement failed: {e}")
            return 0.0
    
    def _measure_economic_significance_simple(self,
                                            returns: pd.Series,
                                            cluster_labels: np.ndarray) -> float:
        """Simple measure of economic significance."""
        
        try:
            unique_clusters = np.unique(cluster_labels)
            
            if len(unique_clusters) < 2:
                return 0.0
            
            # Calculate Sharpe ratios for each cluster
            cluster_sharpes = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_returns = returns[cluster_mask]
                
                if len(cluster_returns) > 10 and cluster_returns.std() > 0:
                    sharpe = cluster_returns.mean() / cluster_returns.std()
                    cluster_sharpes.append(abs(sharpe))
            
            if len(cluster_sharpes) > 1:
                return max(cluster_sharpes) - min(cluster_sharpes)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Economic significance measurement failed: {e}")
            return 0.0


# Convenience function
def analyze_price_action_influence(features: pd.DataFrame,
                                 price_data: pd.DataFrame,
                                 cluster_labels: np.ndarray,
                                 config: Optional[FeaturePriceInteractionConfig] = None) -> Dict[PriceActionPattern, PriceActionInfluenceResult]:
    """
    Convenience function for price action influence analysis.
    
    Args:
        features: Feature matrix
        price_data: Price data
        cluster_labels: Cluster assignments
        config: Optional configuration
        
    Returns:
        Price action influence results
    """
    analyzer = EnhancedPriceActionAnalyzer(config)
    return analyzer.analyze_price_action_influence(features, price_data, cluster_labels)


# Example usage
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with regime structure
    momentum_factor = np.random.randn(n_samples)
    volatility_factor = np.random.randn(n_samples)
    
    features = pd.DataFrame({
        'momentum_1': momentum_factor + np.random.randn(n_samples) * 0.2,
        'momentum_2': momentum_factor + np.random.randn(n_samples) * 0.3,
        'volatility_1': volatility_factor + np.random.randn(n_samples) * 0.2,
        'volatility_2': volatility_factor + np.random.randn(n_samples) * 0.3,
        'noise_1': np.random.randn(n_samples),
        'noise_2': np.random.randn(n_samples)
    })
    
    # Create price data influenced by features
    returns = (momentum_factor * 0.01 + volatility_factor * 0.005 + np.random.randn(n_samples) * 0.02)
    prices = 100 * np.exp(np.cumsum(returns))
    
    price_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + abs(np.random.randn(n_samples)) * 0.01),
        'low': prices * (1 - abs(np.random.randn(n_samples)) * 0.01),
        'returns': returns
    })
    
    # Create simple clusters
    cluster_labels = np.where(momentum_factor > 0, 0, 1)
    
    # Test price action analysis
    analyzer = EnhancedPriceActionAnalyzer()
    
    print("🔍 Testing Enhanced Price Action Analysis")
    influence_results = analyzer.analyze_price_action_influence(features, price_data, cluster_labels)
    
    print(f"\n📊 Price Action Influence Results:")
    for pattern, result in influence_results.items():
        print(f"{pattern.value}:")
        print(f"  - Influence strength: {result.influence_strength:.3f}")
        print(f"  - Mechanism: {result.mechanism.value}")
        print(f"  - Statistical significance: {result.statistical_significance:.3f}")
        print(f"  - Economic significance: {result.economic_significance:.3f}")
    
    # Test CV-coupling relationship
    print(f"\n🔍 Testing Feature-Price Coupling vs CV Relationship")
    coupling_analysis = analyzer.analyze_feature_price_coupling_by_cv(features, price_data)
    
    print(f"Max coupling strength: {coupling_analysis['max_coupling_strength']:.3f}")
    if coupling_analysis['breaking_point_cv']:
        print(f"Breaking point CV: {coupling_analysis['breaking_point_cv']:.3f}")
    print(f"CV-Coupling correlation: {coupling_analysis['coupling_cv_correlation']:.3f}")