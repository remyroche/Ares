"""
Enhanced Price Action Influence Analysis

This module provides advanced analysis of how feature clusters influence price action,
integrating with the dedicated price patterns research modules for consistent
pattern definitions across the research framework.

Key Research Focus:
- Integration with src/research/pattern_discovery_framework.py
- Integration with src/research/pure_price_action_patterns.py
- How feature clusters influence specific price patterns
- Relationship between feature homogeneity and price predictive power
- Which feature interactions drive the strongest price responses

This connects to the broader research on price patterns by using the dedicated
pattern definitions and validation methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

from src.utils.logger import system_logger

# Import price patterns integration
try:
    from .price_patterns_integration import (
        PricePatternsIntegrator,
        PatternIntegrationConfig,
        integrate_with_price_patterns_research,
        PRICE_PATTERNS_MODULE_AVAILABLE
    )
    PRICE_PATTERNS_INTEGRATION_AVAILABLE = True
except ImportError:
    PRICE_PATTERNS_INTEGRATION_AVAILABLE = False
    PRICE_PATTERNS_MODULE_AVAILABLE = False

# Import ML libraries if available
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
    pattern_name: str
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
    use_external_patterns: bool = True
    pattern_integration_config: Optional[PatternIntegrationConfig] = None
    
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
        if self.pattern_integration_config is None:
            self.pattern_integration_config = PatternIntegrationConfig()


class EnhancedPriceActionAnalyzer:
    """
    Enhanced analyzer for price action influence using external pattern research.
    
    This class integrates with the dedicated price patterns research modules
    to provide advanced analysis of feature-pattern relationships.
    """
    
    def __init__(self, 
                 config: Optional[FeaturePriceInteractionConfig] = None):
        self.config = config or FeaturePriceInteractionConfig()
        self.logger = system_logger.getChild('EnhancedPriceActionAnalyzer')
        
        # Initialize price patterns integrator
        if PRICE_PATTERNS_INTEGRATION_AVAILABLE and self.config.use_external_patterns:
            self.pattern_integrator = PricePatternsIntegrator(self.config.pattern_integration_config)
            self.logger.info("✅ Price patterns integrator initialized with external research")
        else:
            self.pattern_integrator = None
            self.logger.warning("⚠️ Price patterns integration not available - using fallback")
    
    def analyze_price_action_influence(self,
                                     features: pd.DataFrame,
                                     price_data: pd.DataFrame,
                                     cluster_labels: np.ndarray) -> Dict[str, PriceActionInfluenceResult]:
        """
        Comprehensive analysis of how feature clusters influence price action patterns.
        
        Args:
            features: Feature matrix
            price_data: Price data (OHLCV)
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary mapping pattern names to influence results
        """
        self.logger.info("🔍 Starting enhanced price action influence analysis with external patterns")
        
        # Detect price action patterns using external research
        if self.pattern_integrator:
            price_patterns = self.pattern_integrator.detect_patterns(price_data)
            self.logger.info(f"📊 Using {len(price_patterns)} patterns from external research")
        else:
            price_patterns = self._fallback_pattern_detection(price_data)
            self.logger.info(f"📊 Using {len(price_patterns)} patterns from fallback detection")
        
        # Analyze influence for each pattern
        influence_results = {}
        
        for pattern_name, pattern_series in price_patterns.items():
            self.logger.info(f"📈 Analyzing influence of features on {pattern_name}")
            
            influence_result = self._analyze_pattern_influence(
                features, price_data, cluster_labels, pattern_name, pattern_series
            )
            
            influence_results[pattern_name] = influence_result
        
        self.logger.info(f"✅ Price action influence analysis completed for {len(influence_results)} patterns")
        return influence_results
    
    def _analyze_pattern_influence(self,
                                 features: pd.DataFrame,
                                 price_data: pd.DataFrame,
                                 cluster_labels: np.ndarray,
                                 pattern_name: str,
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
            pattern_name=pattern_name,
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
                'n_clusters': len(np.unique(labels_aligned)),
                'uses_external_patterns': self.pattern_integrator is not None
            }
        )
    
    def _analyze_cluster_specific_influence(self,
                                          features: pd.DataFrame,
                                          pattern_series: pd.Series,
                                          cluster_labels: np.ndarray) -> Dict[int, float]:
        """Analyze how each cluster influences the price pattern."""
        
        cluster_influences = {}
        overall_pattern_rate = pattern_series.mean()
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            
            if np.sum(cluster_mask) < 10:  # Need minimum samples
                cluster_influences[cluster_id] = 0.0
                continue
            
            # Calculate pattern occurrence rate in this cluster
            cluster_pattern_rate = pattern_series[cluster_mask].mean()
            
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
        
        if SKLEARN_AVAILABLE:
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
                        
                return feature_contributions
                
            except Exception as e:
                self.logger.warning(f"Mutual information calculation failed: {e}, using correlation fallback")
        
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
        try:
            feature_means = features.groupby(cluster_labels).mean()
            if len(feature_means) > 1:
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
        except:
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
        
        if not SKLEARN_AVAILABLE:
            # Fallback to simple correlation
            try:
                feature_mean = features.mean(axis=1)
                corr = abs(np.corrcoef(feature_mean, pattern_series)[0, 1])
                return corr if not np.isnan(corr) else 0.0
            except:
                return 0.0
        
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
            
            # Align data
            min_len = min(len(pattern_series), len(returns))
            pattern_aligned = pattern_series.iloc[:min_len]
            returns_aligned = returns.iloc[:min_len]
            
            # Returns when pattern occurs vs when it doesn't
            pattern_returns = returns_aligned[pattern_aligned > 0.5]
            no_pattern_returns = returns_aligned[pattern_aligned <= 0.5]
            
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
        self.logger.info("🔍 Analyzing feature-price coupling vs CV relationship using external patterns")
        
        cv_min, cv_max, cv_steps = cv_range
        cv_thresholds = np.linspace(cv_min, cv_max, cv_steps)
        
        coupling_results = {
            'cv_thresholds': cv_thresholds.tolist(),
            'coupling_strengths': [],
            'predictive_powers': [],
            'cluster_counts': [],
            'economic_significances': [],
            'pattern_influences': []
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
                
                # Measure pattern influence if external patterns available
                pattern_influence = 0.0
                if self.pattern_integrator:
                    patterns = self.pattern_integrator.detect_patterns(price_data)
                    if patterns:
                        pattern_influences = []
                        for pattern_name, pattern_series in patterns.items():
                            cluster_pattern_influence = self._measure_pattern_cluster_influence(
                                pattern_series, clustering_result.labels
                            )
                            pattern_influences.append(cluster_pattern_influence)
                        pattern_influence = np.mean(pattern_influences) if pattern_influences else 0.0
                
                coupling_results['coupling_strengths'].append(coupling_strength)
                coupling_results['predictive_powers'].append(predictive_power)
                coupling_results['cluster_counts'].append(clustering_result.n_clusters)
                coupling_results['economic_significances'].append(economic_sig)
                coupling_results['pattern_influences'].append(pattern_influence)
                
            except Exception as e:
                self.logger.warning(f"CV analysis failed for threshold {cv_thresh:.3f}: {e}")
                coupling_results['coupling_strengths'].append(0.0)
                coupling_results['predictive_powers'].append(0.0)
                coupling_results['cluster_counts'].append(1)
                coupling_results['economic_significances'].append(0.0)
                coupling_results['pattern_influences'].append(0.0)
        
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
            'coupling_cv_correlation': float(np.corrcoef(cv_thresholds, coupling_strengths)[0, 1]) if len(coupling_strengths) > 1 else 0.0,
            'uses_external_patterns': self.pattern_integrator is not None
        }
        
        self.logger.info(f"📊 Feature-Price Coupling Analysis (using external patterns):")
        self.logger.info(f"   Max coupling strength: {max_coupling:.3f}")
        if breaking_point_cv:
            self.logger.info(f"   Breaking point CV: {breaking_point_cv:.3f}")
            self.logger.info(f"   → Feature-price coupling degrades significantly beyond CV={breaking_point_cv:.3f}")
        
        correlation = analysis_result['coupling_cv_correlation']
        if abs(correlation) > 0.3:
            direction = "decreases" if correlation < 0 else "increases"
            self.logger.info(f"   CV-Coupling correlation: {correlation:.3f} (coupling {direction} with CV)")
        
        return analysis_result
    
    def _measure_pattern_cluster_influence(self,
                                         pattern_series: pd.Series,
                                         cluster_labels: np.ndarray) -> float:
        """Measure how much clusters influence pattern occurrence."""
        
        try:
            overall_rate = pattern_series.mean()
            cluster_rates = []
            
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 5:
                    cluster_rate = pattern_series[cluster_mask].mean()
                    cluster_rates.append(cluster_rate)
            
            if len(cluster_rates) > 1:
                # Measure variance in pattern rates across clusters
                rate_std = np.std(cluster_rates)
                influence = rate_std / overall_rate if overall_rate > 0 else 0.0
                return min(1.0, influence)  # Cap at 1.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
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
        
        if not SKLEARN_AVAILABLE:
            return 0.0
        
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
    
    def _fallback_pattern_detection(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Fallback pattern detection when external modules not available."""
        
        patterns = {}
        
        if 'close' not in price_data.columns:
            self.logger.warning("No close price data available for fallback pattern detection")
            return patterns
        
        prices = price_data['close']
        returns = prices.pct_change().fillna(0)
        
        # Basic patterns
        sma_short = prices.rolling(10).mean()
        sma_long = prices.rolling(20).mean()
        
        patterns['trend_continuation'] = (
            ((prices > sma_short) & (sma_short > sma_long) & (returns > 0)) |
            ((prices < sma_short) & (sma_short < sma_long) & (returns < 0))
        ).astype(float)
        
        patterns['momentum_persistence'] = (
            (returns.rolling(5).mean() * returns.rolling(20).mean() > 0)
        ).astype(float)
        
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(20).mean()
        patterns['volatility_expansion'] = (volatility > vol_ma * 1.5).astype(float)
        
        # Clean patterns
        for pattern_name in patterns:
            patterns[pattern_name] = patterns[pattern_name].fillna(0)
        
        return patterns


# Convenience function
def analyze_price_action_influence_with_external_patterns(features: pd.DataFrame,
                                                        price_data: pd.DataFrame,
                                                        cluster_labels: np.ndarray,
                                                        config: Optional[FeaturePriceInteractionConfig] = None) -> Dict[str, PriceActionInfluenceResult]:
    """
    Convenience function for price action influence analysis using external patterns.
    
    Args:
        features: Feature matrix
        price_data: Price data
        cluster_labels: Cluster assignments
        config: Optional configuration
        
    Returns:
        Price action influence results using external pattern definitions
    """
    analyzer = EnhancedPriceActionAnalyzer(config)
    return analyzer.analyze_price_action_influence(features, price_data, cluster_labels)


# Example usage
if __name__ == "__main__":
    print("🔍 Testing Enhanced Price Action Analysis with External Patterns")
    print(f"External patterns available: {PRICE_PATTERNS_MODULE_AVAILABLE}")
    
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
    
    # Test integration
    config = FeaturePriceInteractionConfig(
        use_external_patterns=True,
        pattern_integration_config=PatternIntegrationConfig(
            use_external_patterns=True,
            use_pure_price_patterns=True
        )
    )
    
    analyzer = EnhancedPriceActionAnalyzer(config)
    
    print("\n📊 Testing Price Action Influence Analysis:")
    influence_results = analyzer.analyze_price_action_influence(features, price_data, cluster_labels)
    
    for pattern_name, result in influence_results.items():
        print(f"{pattern_name}:")
        print(f"  - Influence strength: {result.influence_strength:.3f}")
        print(f"  - Mechanism: {result.mechanism.value}")
        print(f"  - Statistical significance: {result.statistical_significance:.3f}")
        print(f"  - Economic significance: {result.economic_significance:.3f}")
        print(f"  - Uses external patterns: {result.metadata['uses_external_patterns']}")
    
    print(f"\n🔍 Testing CV-Coupling Analysis:")
    coupling_analysis = analyzer.analyze_feature_price_coupling_by_cv(features, price_data)
    
    print(f"Max coupling strength: {coupling_analysis['max_coupling_strength']:.3f}")
    if coupling_analysis['breaking_point_cv']:
        print(f"Breaking point CV: {coupling_analysis['breaking_point_cv']:.3f}")
    print(f"CV-Coupling correlation: {coupling_analysis['coupling_cv_correlation']:.3f}")
    print(f"Uses external patterns: {coupling_analysis['uses_external_patterns']}")