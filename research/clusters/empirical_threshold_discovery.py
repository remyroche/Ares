"""
Empirical Threshold Discovery for Data-Driven Clustering

This module provides a data-driven framework to empirically discover the optimal
CV and similarity thresholds for clustering based on actual feature/price interactions
and economic relevance.

Key Research Questions Answered:
1. At what CV level do merged clusters lose price predictive power?
2. At what similarity threshold do feature interactions become economically irrelevant?
3. What's the relationship between feature homogeneity and price action influence?

The framework tests different threshold combinations and measures their impact on
economic relevance, providing empirical evidence for optimal clustering parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from itertools import product
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.utils.logger import system_logger
from .similarity_matrix_clustering import SimilarityMatrixClusterer, SimilarityClusteringConfig, SimilarityMethod


class EconomicRelevanceMetric(Enum):
    """Metrics for measuring economic relevance."""
    PRICE_PREDICTIVE_POWER = "price_predictive_power"
    FEATURE_PRICE_COUPLING = "feature_price_coupling"
    INFORMATION_RATIO = "information_ratio"
    REGIME_SEPARABILITY = "regime_separability"
    SHARPE_RATIO_DIFFERENCE = "sharpe_ratio_difference"
    VOLATILITY_PREDICTION = "volatility_prediction"
    RETURN_PREDICTION = "return_prediction"


@dataclass
class ThresholdTestResult:
    """Result for a single threshold combination test."""
    cv_threshold: float
    similarity_threshold: float
    n_clusters: int
    n_samples_per_cluster: List[int]
    economic_relevance_scores: Dict[str, float]
    cluster_quality_scores: Dict[str, float]
    price_predictive_power: float
    feature_price_coupling: float
    overall_economic_relevance: float
    is_economically_viable: bool
    breaking_point_reached: bool
    metadata: Dict[str, Any]


@dataclass
class EmpiricalDiscoveryConfig:
    """Configuration for empirical threshold discovery."""
    # Threshold ranges to test
    cv_range: Tuple[float, float, int] = (0.1, 1.0, 20)  # (min, max, steps)
    similarity_range: Tuple[float, float, int] = (0.3, 0.95, 15)  # (min, max, steps)
    
    # Economic relevance criteria
    economic_relevance_metrics: List[str] = field(default_factory=lambda: [
        "price_predictive_power", "feature_price_coupling", "regime_separability"
    ])
    min_economic_relevance: float = 0.15
    breaking_point_threshold: float = 0.8  # Percentage of baseline to consider breaking point
    
    # Cluster validation
    min_samples_per_cluster: int = 50
    max_clusters: int = 20
    min_clusters: int = 2
    
    # Price prediction validation
    prediction_horizon: int = 5  # Periods ahead to predict
    cross_validation_folds: int = 3
    
    # Statistical significance
    significance_level: float = 0.05
    bootstrap_samples: int = 100
    
    # Performance optimization
    parallel_processing: bool = True
    cache_results: bool = True
    early_stopping: bool = True  # Stop when breaking point clearly identified


@dataclass
class EmpiricalDiscoveryResult:
    """Result container for empirical threshold discovery."""
    baseline_economic_relevance: float
    optimal_cv_threshold: float
    optimal_similarity_threshold: float
    cv_breaking_point: Optional[float]
    similarity_breaking_point: Optional[float]
    threshold_test_results: List[ThresholdTestResult]
    economic_relevance_surface: np.ndarray
    breaking_point_analysis: Dict[str, Any]
    recommendations: Dict[str, Any]
    metadata: Dict[str, Any]


class EmpiricalThresholdDiscovery:
    """
    Data-driven discovery of optimal CV and similarity thresholds.
    
    This class empirically tests different threshold combinations to find:
    1. The point where economic relevance breaks down
    2. The optimal balance between cluster size and similarity
    3. The relationship between feature homogeneity and price predictive power
    """
    
    def __init__(self, config: Optional[EmpiricalDiscoveryConfig] = None):
        self.config = config or EmpiricalDiscoveryConfig()
        self.logger = system_logger.getChild('EmpiricalThresholdDiscovery')
        self._cache = {}
        
    def discover_optimal_thresholds(self,
                                  features: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  feature_names: Optional[List[str]] = None) -> EmpiricalDiscoveryResult:
        """
        Empirically discover optimal CV and similarity thresholds.
        
        Args:
            features: Feature matrix for clustering
            price_data: Price data for economic validation
            feature_names: Optional feature names for interpretation
            
        Returns:
            Discovery result with optimal thresholds and analysis
        """
        self.logger.info("🔍 Starting empirical threshold discovery")
        
        # Establish baseline economic relevance
        self.logger.info("📊 Establishing baseline economic relevance")
        baseline_relevance = self._establish_baseline_relevance(features, price_data)
        
        # Generate threshold combinations to test
        threshold_combinations = self._generate_threshold_combinations()
        self.logger.info(f"Testing {len(threshold_combinations)} threshold combinations")
        
        # Test each threshold combination
        test_results = []
        breaking_point_found = False
        
        for i, (cv_thresh, sim_thresh) in enumerate(threshold_combinations):
            if self.config.early_stopping and breaking_point_found:
                self.logger.info(f"Early stopping at combination {i}/{len(threshold_combinations)}")
                break
                
            self.logger.info(f"Testing combination {i+1}/{len(threshold_combinations)}: "
                           f"CV={cv_thresh:.3f}, Similarity={sim_thresh:.3f}")
            
            # Test this threshold combination
            test_result = self._test_threshold_combination(
                features, price_data, cv_thresh, sim_thresh, baseline_relevance
            )
            test_results.append(test_result)
            
            # Check if we've reached breaking point
            if test_result.breaking_point_reached:
                breaking_point_found = True
        
        # Analyze results
        self.logger.info("📈 Analyzing threshold test results")
        analysis_result = self._analyze_threshold_results(
            test_results, baseline_relevance, features, price_data
        )
        
        self.logger.info(f"✅ Discovery completed. Optimal thresholds: "
                        f"CV={analysis_result.optimal_cv_threshold:.3f}, "
                        f"Similarity={analysis_result.optimal_similarity_threshold:.3f}")
        
        return analysis_result
    
    def _establish_baseline_relevance(self, 
                                    features: pd.DataFrame,
                                    price_data: pd.DataFrame) -> float:
        """Establish baseline economic relevance without clustering constraints."""
        
        try:
            # Use all features without clustering to establish baseline
            baseline_scores = {}
            
            # Price predictive power
            baseline_scores['price_predictive_power'] = self._measure_price_predictive_power(
                features, price_data, cluster_labels=None
            )
            
            # Feature-price coupling
            baseline_scores['feature_price_coupling'] = self._measure_feature_price_coupling(
                features, price_data, cluster_labels=None
            )
            
            # Information ratio
            baseline_scores['information_ratio'] = self._measure_information_ratio(
                price_data, cluster_labels=None
            )
            
            # Calculate composite baseline
            baseline_relevance = np.mean(list(baseline_scores.values()))
            
            self.logger.info(f"📊 Baseline economic relevance: {baseline_relevance:.3f}")
            for metric, score in baseline_scores.items():
                self.logger.info(f"   - {metric}: {score:.3f}")
            
            return baseline_relevance
            
        except Exception as e:
            self.logger.error(f"Failed to establish baseline relevance: {e}")
            return 0.5  # Default baseline
    
    def _generate_threshold_combinations(self) -> List[Tuple[float, float]]:
        """Generate threshold combinations to test."""
        
        # Generate CV threshold range
        cv_min, cv_max, cv_steps = self.config.cv_range
        cv_thresholds = np.linspace(cv_min, cv_max, cv_steps)
        
        # Generate similarity threshold range
        sim_min, sim_max, sim_steps = self.config.similarity_range
        sim_thresholds = np.linspace(sim_min, sim_max, sim_steps)
        
        # Create all combinations
        combinations = list(product(cv_thresholds, sim_thresholds))
        
        # Sort by increasing relaxation (higher CV, lower similarity)
        combinations.sort(key=lambda x: (x[0], -x[1]))
        
        return combinations
    
    def _test_threshold_combination(self,
                                  features: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  cv_threshold: float,
                                  similarity_threshold: float,
                                  baseline_relevance: float) -> ThresholdTestResult:
        """Test a specific threshold combination."""
        
        try:
            # Create clustering configuration
            clustering_config = SimilarityClusteringConfig(
                similarity_threshold=similarity_threshold,
                cv_threshold=cv_threshold,
                min_samples_per_cluster=self.config.min_samples_per_cluster,
                enable_economic_validation=True
            )
            
            # Perform clustering
            clusterer = SimilarityMatrixClusterer(clustering_config)
            clustering_result = clusterer.fit_predict(features, price_data)
            
            # Validate cluster constraints
            cluster_sizes = np.bincount(clustering_result.labels)
            n_clusters = len(cluster_sizes)
            
            # Check if clustering is viable
            if (n_clusters < self.config.min_clusters or 
                n_clusters > self.config.max_clusters or
                np.min(cluster_sizes) < self.config.min_samples_per_cluster):
                
                return ThresholdTestResult(
                    cv_threshold=cv_threshold,
                    similarity_threshold=similarity_threshold,
                    n_clusters=n_clusters,
                    n_samples_per_cluster=cluster_sizes.tolist(),
                    economic_relevance_scores={},
                    cluster_quality_scores={},
                    price_predictive_power=0.0,
                    feature_price_coupling=0.0,
                    overall_economic_relevance=0.0,
                    is_economically_viable=False,
                    breaking_point_reached=True,
                    metadata={'failure_reason': 'cluster_constraints_violated'}
                )
            
            # Measure economic relevance
            economic_scores = self._measure_economic_relevance(
                features, price_data, clustering_result.labels
            )
            
            # Measure cluster quality
            quality_scores = self._measure_cluster_quality(
                features, clustering_result.labels
            )
            
            # Calculate overall economic relevance
            overall_relevance = np.mean(list(economic_scores.values()))
            
            # Check if breaking point reached
            breaking_point_reached = (
                overall_relevance < baseline_relevance * self.config.breaking_point_threshold
            )
            
            # Check economic viability
            is_viable = overall_relevance >= self.config.min_economic_relevance
            
            return ThresholdTestResult(
                cv_threshold=cv_threshold,
                similarity_threshold=similarity_threshold,
                n_clusters=n_clusters,
                n_samples_per_cluster=cluster_sizes.tolist(),
                economic_relevance_scores=economic_scores,
                cluster_quality_scores=quality_scores,
                price_predictive_power=economic_scores.get('price_predictive_power', 0.0),
                feature_price_coupling=economic_scores.get('feature_price_coupling', 0.0),
                overall_economic_relevance=overall_relevance,
                is_economically_viable=is_viable,
                breaking_point_reached=breaking_point_reached,
                metadata={
                    'baseline_relevance': baseline_relevance,
                    'relevance_ratio': overall_relevance / baseline_relevance if baseline_relevance > 0 else 0
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Threshold test failed for CV={cv_threshold:.3f}, Sim={similarity_threshold:.3f}: {e}")
            
            return ThresholdTestResult(
                cv_threshold=cv_threshold,
                similarity_threshold=similarity_threshold,
                n_clusters=0,
                n_samples_per_cluster=[],
                economic_relevance_scores={},
                cluster_quality_scores={},
                price_predictive_power=0.0,
                feature_price_coupling=0.0,
                overall_economic_relevance=0.0,
                is_economically_viable=False,
                breaking_point_reached=True,
                metadata={'failure_reason': 'exception', 'error': str(e)}
            )
    
    def _measure_economic_relevance(self,
                                  features: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  cluster_labels: np.ndarray) -> Dict[str, float]:
        """Measure economic relevance of clustering."""
        
        scores = {}
        
        # Price predictive power
        scores['price_predictive_power'] = self._measure_price_predictive_power(
            features, price_data, cluster_labels
        )
        
        # Feature-price coupling
        scores['feature_price_coupling'] = self._measure_feature_price_coupling(
            features, price_data, cluster_labels
        )
        
        # Regime separability
        scores['regime_separability'] = self._measure_regime_separability(
            price_data, cluster_labels
        )
        
        # Information ratio
        scores['information_ratio'] = self._measure_information_ratio(
            price_data, cluster_labels
        )
        
        return scores
    
    def _measure_price_predictive_power(self,
                                      features: pd.DataFrame,
                                      price_data: pd.DataFrame,
                                      cluster_labels: Optional[np.ndarray] = None) -> float:
        """Measure how well features predict price movements."""
        
        try:
            # Calculate future returns
            returns = price_data['close'].pct_change().fillna(0)
            future_returns = returns.shift(-self.config.prediction_horizon).fillna(0)
            
            # Align data
            min_len = min(len(features), len(future_returns))
            X = features.iloc[:min_len]
            y = future_returns.iloc[:min_len]
            
            if cluster_labels is not None:
                # Add cluster labels as features
                cluster_features = pd.DataFrame({
                    'cluster_id': cluster_labels[:min_len]
                })
                X = pd.concat([X, cluster_features], axis=1)
            
            # Remove NaN and infinite values
            x_finite = np.isfinite(X)
            y_finite = np.isfinite(y)
            valid_mask = y_finite & x_finite.all(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:  # Need minimum samples
                return 0.0
            
            # Use Random Forest for prediction
            rf = RandomForestRegressor(
                n_estimators=50, 
                random_state=42, 
                n_jobs=1,
                max_depth=5  # Prevent overfitting
            )
            
            # Cross-validation score
            cv_scores = cross_val_score(
                rf, X, y, 
                cv=self.config.cross_validation_folds,
                scoring='r2',
                n_jobs=1
            )
            
            # Return mean R² score
            return max(0.0, np.mean(cv_scores))
            
        except Exception as e:
            self.logger.warning(f"Price predictive power measurement failed: {e}")
            return 0.0
    
    def _measure_feature_price_coupling(self,
                                      features: pd.DataFrame,
                                      price_data: pd.DataFrame,
                                      cluster_labels: Optional[np.ndarray] = None) -> float:
        """Measure coupling between features and price action within clusters."""
        
        try:
            returns = price_data['close'].pct_change().fillna(0)
            
            if cluster_labels is None:
                # Measure overall feature-price coupling
                coupling_scores = []
                for col in features.columns:
                    if features[col].std() > 0:
                        # Calculate mutual information
                        try:
                            mi_score = mutual_info_regression(
                                features[[col]].fillna(0), 
                                returns.fillna(0),
                                discrete_features=False,
                                random_state=42
                            )[0]
                            coupling_scores.append(mi_score)
                        except:
                            # Fallback to correlation
                            corr = abs(np.corrcoef(features[col].fillna(0), returns.fillna(0))[0, 1])
                            if not np.isnan(corr):
                                coupling_scores.append(corr)
                
                return np.mean(coupling_scores) if coupling_scores else 0.0
            
            else:
                # Measure cluster-specific feature-price coupling
                cluster_coupling_scores = []
                
                for cluster_id in np.unique(cluster_labels):
                    cluster_mask = cluster_labels == cluster_id
                    
                    if np.sum(cluster_mask) < 10:  # Need minimum samples
                        continue
                    
                    cluster_features = features[cluster_mask]
                    cluster_returns = returns[cluster_mask]
                    
                    # Calculate feature-return coupling within cluster
                    feature_couplings = []
                    for col in cluster_features.columns:
                        if cluster_features[col].std() > 0:
                            corr = abs(np.corrcoef(cluster_features[col], cluster_returns)[0, 1])
                            if not np.isnan(corr):
                                feature_couplings.append(corr)
                    
                    if feature_couplings:
                        cluster_coupling_scores.append(np.mean(feature_couplings))
                
                return np.mean(cluster_coupling_scores) if cluster_coupling_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Feature-price coupling measurement failed: {e}")
            return 0.0
    
    def _measure_regime_separability(self,
                                   price_data: pd.DataFrame,
                                   cluster_labels: Optional[np.ndarray] = None) -> float:
        """Measure how separable price behaviors are between regimes."""
        
        try:
            if cluster_labels is None:
                return 0.0
            
            returns = price_data['close'].pct_change().fillna(0)
            unique_clusters = np.unique(cluster_labels)
            
            if len(unique_clusters) < 2:
                return 0.0
            
            # Calculate cluster-specific return statistics
            cluster_returns = {}
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_returns[cluster_id] = returns[cluster_mask]
            
            # ANOVA test for return differences
            cluster_return_lists = [rets.values for rets in cluster_returns.values() if len(rets) > 5]
            
            if len(cluster_return_lists) < 2:
                return 0.0
            
            f_stat, p_value = stats.f_oneway(*cluster_return_lists)
            
            # Convert p-value to separability score (lower p-value = higher separability)
            separability = 1.0 - min(1.0, p_value)
            
            return separability
            
        except Exception as e:
            self.logger.warning(f"Regime separability measurement failed: {e}")
            return 0.0
    
    def _measure_information_ratio(self,
                                 price_data: pd.DataFrame,
                                 cluster_labels: Optional[np.ndarray] = None) -> float:
        """Measure information ratio differences between regimes."""
        
        try:
            returns = price_data['close'].pct_change().fillna(0)
            
            if cluster_labels is None:
                # Overall information ratio
                if returns.std() > 0:
                    return abs(returns.mean() / returns.std())
                else:
                    return 0.0
            
            unique_clusters = np.unique(cluster_labels)
            if len(unique_clusters) < 2:
                return 0.0
            
            # Calculate information ratios for each cluster
            cluster_irs = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_returns = returns[cluster_mask]
                
                if len(cluster_returns) > 5 and cluster_returns.std() > 0:
                    ir = cluster_returns.mean() / cluster_returns.std()
                    cluster_irs.append(abs(ir))
            
            if len(cluster_irs) < 2:
                return 0.0
            
            # Return range of information ratios
            return max(cluster_irs) - min(cluster_irs)
            
        except Exception as e:
            self.logger.warning(f"Information ratio measurement failed: {e}")
            return 0.0
    
    def _measure_cluster_quality(self,
                               features: pd.DataFrame,
                               cluster_labels: np.ndarray) -> Dict[str, float]:
        """Measure cluster quality metrics."""
        
        quality_scores = {}
        
        try:
            # Silhouette score
            if len(np.unique(cluster_labels)) > 1:
                from sklearn.metrics import silhouette_score
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features.fillna(0))
                quality_scores['silhouette_score'] = silhouette_score(features_scaled, cluster_labels)
            else:
                quality_scores['silhouette_score'] = 0.0
            
            # Within-cluster CV
            cluster_cvs = []
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_features = features[cluster_mask]
                
                feature_cvs = []
                for col in cluster_features.columns:
                    if cluster_features[col].std() > 0 and cluster_features[col].mean() != 0:
                        cv = abs(cluster_features[col].std() / cluster_features[col].mean())
                        feature_cvs.append(cv)
                
                if feature_cvs:
                    cluster_cvs.append(np.mean(feature_cvs))
            
            quality_scores['mean_cv'] = np.mean(cluster_cvs) if cluster_cvs else float('inf')
            
            # Cluster balance
            cluster_sizes = np.bincount(cluster_labels)
            if len(cluster_sizes) > 1:
                size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes)
                quality_scores['balance_score'] = 1.0 / (1.0 + size_cv)
            else:
                quality_scores['balance_score'] = 1.0
            
        except Exception as e:
            self.logger.warning(f"Cluster quality measurement failed: {e}")
            quality_scores = {'silhouette_score': 0.0, 'mean_cv': float('inf'), 'balance_score': 0.0}
        
        return quality_scores
    
    def _analyze_threshold_results(self,
                                 test_results: List[ThresholdTestResult],
                                 baseline_relevance: float,
                                 features: pd.DataFrame,
                                 price_data: pd.DataFrame) -> EmpiricalDiscoveryResult:
        """Analyze threshold test results and find optimal parameters."""
        
        # Filter valid results
        valid_results = [r for r in test_results if r.is_economically_viable]
        
        if not valid_results:
            self.logger.warning("No economically viable threshold combinations found")
            # Return default result
            return EmpiricalDiscoveryResult(
                baseline_economic_relevance=baseline_relevance,
                optimal_cv_threshold=0.3,
                optimal_similarity_threshold=0.7,
                cv_breaking_point=None,
                similarity_breaking_point=None,
                threshold_test_results=test_results,
                economic_relevance_surface=np.array([]),
                breaking_point_analysis={},
                recommendations={'warning': 'No viable thresholds found'},
                metadata={'n_tests': len(test_results)}
            )
        
        # Find optimal thresholds (highest economic relevance)
        best_result = max(valid_results, key=lambda x: x.overall_economic_relevance)
        
        # Find breaking points
        cv_breaking_point = None
        similarity_breaking_point = None
        
        for result in test_results:
            if result.breaking_point_reached:
                if cv_breaking_point is None:
                    cv_breaking_point = result.cv_threshold
                if similarity_breaking_point is None:
                    similarity_breaking_point = result.similarity_threshold
                break
        
        # Create economic relevance surface
        economic_surface = self._create_relevance_surface(test_results)
        
        # Generate breaking point analysis
        breaking_point_analysis = self._analyze_breaking_points(test_results, baseline_relevance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            best_result, breaking_point_analysis, baseline_relevance
        )
        
        return EmpiricalDiscoveryResult(
            baseline_economic_relevance=baseline_relevance,
            optimal_cv_threshold=best_result.cv_threshold,
            optimal_similarity_threshold=best_result.similarity_threshold,
            cv_breaking_point=cv_breaking_point,
            similarity_breaking_point=similarity_breaking_point,
            threshold_test_results=test_results,
            economic_relevance_surface=economic_surface,
            breaking_point_analysis=breaking_point_analysis,
            recommendations=recommendations,
            metadata={
                'n_tests_total': len(test_results),
                'n_tests_viable': len(valid_results),
                'best_economic_relevance': best_result.overall_economic_relevance,
                'relevance_improvement': best_result.overall_economic_relevance - baseline_relevance
            }
        )
    
    def _create_relevance_surface(self, test_results: List[ThresholdTestResult]) -> np.ndarray:
        """Create 2D surface of economic relevance vs thresholds."""
        
        try:
            # Get unique threshold values
            cv_values = sorted(list(set(r.cv_threshold for r in test_results)))
            sim_values = sorted(list(set(r.similarity_threshold for r in test_results)))
            
            # Create surface matrix
            surface = np.full((len(cv_values), len(sim_values)), np.nan)
            
            # Fill surface with economic relevance values
            for result in test_results:
                cv_idx = cv_values.index(result.cv_threshold)
                sim_idx = sim_values.index(result.similarity_threshold)
                surface[cv_idx, sim_idx] = result.overall_economic_relevance
            
            return surface
            
        except Exception as e:
            self.logger.warning(f"Failed to create relevance surface: {e}")
            return np.array([])
    
    def _analyze_breaking_points(self,
                               test_results: List[ThresholdTestResult],
                               baseline_relevance: float) -> Dict[str, Any]:
        """Analyze breaking points in the threshold space."""
        
        analysis = {
            'cv_breaking_analysis': {},
            'similarity_breaking_analysis': {},
            'interaction_effects': {}
        }
        
        try:
            # Analyze CV breaking points
            cv_relevance_by_threshold = {}
            for result in test_results:
                cv_thresh = result.cv_threshold
                if cv_thresh not in cv_relevance_by_threshold:
                    cv_relevance_by_threshold[cv_thresh] = []
                cv_relevance_by_threshold[cv_thresh].append(result.overall_economic_relevance)
            
            # Find CV breaking point
            cv_breaking_point = None
            for cv_thresh in sorted(cv_relevance_by_threshold.keys()):
                avg_relevance = np.mean(cv_relevance_by_threshold[cv_thresh])
                if avg_relevance < baseline_relevance * self.config.breaking_point_threshold:
                    cv_breaking_point = cv_thresh
                    break
            
            analysis['cv_breaking_analysis'] = {
                'breaking_point': cv_breaking_point,
                'relevance_by_threshold': {k: np.mean(v) for k, v in cv_relevance_by_threshold.items()},
                'threshold_count': len(cv_relevance_by_threshold)
            }
            
            # Similar analysis for similarity thresholds
            sim_relevance_by_threshold = {}
            for result in test_results:
                sim_thresh = result.similarity_threshold
                if sim_thresh not in sim_relevance_by_threshold:
                    sim_relevance_by_threshold[sim_thresh] = []
                sim_relevance_by_threshold[sim_thresh].append(result.overall_economic_relevance)
            
            similarity_breaking_point = None
            for sim_thresh in sorted(sim_relevance_by_threshold.keys(), reverse=True):  # High to low
                avg_relevance = np.mean(sim_relevance_by_threshold[sim_thresh])
                if avg_relevance < baseline_relevance * self.config.breaking_point_threshold:
                    similarity_breaking_point = sim_thresh
                    break
            
            analysis['similarity_breaking_analysis'] = {
                'breaking_point': similarity_breaking_point,
                'relevance_by_threshold': {k: np.mean(v) for k, v in sim_relevance_by_threshold.items()},
                'threshold_count': len(sim_relevance_by_threshold)
            }
            
        except Exception as e:
            self.logger.warning(f"Breaking point analysis failed: {e}")
        
        return analysis
    
    def _generate_recommendations(self,
                                best_result: ThresholdTestResult,
                                breaking_point_analysis: Dict[str, Any],
                                baseline_relevance: float) -> Dict[str, Any]:
        """Generate recommendations based on empirical findings."""
        
        recommendations = {
            'optimal_thresholds': {
                'cv_threshold': best_result.cv_threshold,
                'similarity_threshold': best_result.similarity_threshold,
                'confidence': 'high' if best_result.overall_economic_relevance > baseline_relevance * 1.1 else 'medium'
            },
            'breaking_points': {
                'cv_breaking_point': breaking_point_analysis.get('cv_breaking_analysis', {}).get('breaking_point'),
                'similarity_breaking_point': breaking_point_analysis.get('similarity_breaking_analysis', {}).get('breaking_point')
            },
            'economic_insights': {
                'baseline_relevance': baseline_relevance,
                'optimal_relevance': best_result.overall_economic_relevance,
                'improvement_ratio': best_result.overall_economic_relevance / baseline_relevance if baseline_relevance > 0 else 1.0,
                'key_drivers': []
            },
            'clustering_strategy': 'use_similarity_matrix_clustering',
            'validation_approach': 'cv_confirmation_with_economic_validation'
        }
        
        # Add key insights
        if best_result.price_predictive_power > 0.3:
            recommendations['economic_insights']['key_drivers'].append('strong_price_predictive_power')
        
        if best_result.feature_price_coupling > 0.2:
            recommendations['economic_insights']['key_drivers'].append('strong_feature_price_coupling')
        
        if best_result.n_clusters >= 3:
            recommendations['economic_insights']['key_drivers'].append('multiple_distinct_regimes')
        
        return recommendations


# Convenience function
def discover_optimal_clustering_thresholds(features: pd.DataFrame,
                                         price_data: pd.DataFrame,
                                         config: Optional[EmpiricalDiscoveryConfig] = None) -> EmpiricalDiscoveryResult:
    """
    Convenience function for empirical threshold discovery.
    
    Args:
        features: Feature matrix for clustering
        price_data: Price data for economic validation
        config: Optional configuration
        
    Returns:
        Discovery result with optimal thresholds
    """
    discovery = EmpiricalThresholdDiscovery(config)
    return discovery.discover_optimal_thresholds(features, price_data)


# Example usage
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with different correlation structures
    base_features = []
    
    # High correlation group
    base_signal = np.random.randn(n_samples)
    for i in range(5):
        feature = base_signal + np.random.randn(n_samples) * 0.3
        base_features.append(feature)
    
    # Medium correlation group
    base_signal = np.random.randn(n_samples)
    for i in range(5):
        feature = base_signal + np.random.randn(n_samples) * 0.6
        base_features.append(feature)
    
    # Low correlation (noise)
    for i in range(5):
        feature = np.random.randn(n_samples)
        base_features.append(feature)
    
    features = pd.DataFrame(
        np.column_stack(base_features),
        columns=[f'feature_{i}' for i in range(15)]
    )
    
    # Create price data with some relationship to features
    price_factor = features.iloc[:, :5].mean(axis=1) * 0.1  # Influenced by first group
    returns = np.random.randn(n_samples) * 0.02 + price_factor * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    price_data = pd.DataFrame({
        'close': prices,
        'returns': returns
    })
    
    # Test empirical discovery
    config = EmpiricalDiscoveryConfig(
        cv_range=(0.2, 0.8, 10),
        similarity_range=(0.4, 0.9, 8),
        min_economic_relevance=0.1,
        early_stopping=False
    )
    
    result = discover_optimal_clustering_thresholds(features, price_data, config)
    
    print("🎯 Empirical Threshold Discovery Results:")
    print(f"Baseline Economic Relevance: {result.baseline_economic_relevance:.3f}")
    print(f"Optimal CV Threshold: {result.optimal_cv_threshold:.3f}")
    print(f"Optimal Similarity Threshold: {result.optimal_similarity_threshold:.3f}")
    
    if result.cv_breaking_point:
        print(f"CV Breaking Point: {result.cv_breaking_point:.3f}")
    if result.similarity_breaking_point:
        print(f"Similarity Breaking Point: {result.similarity_breaking_point:.3f}")
    
    print(f"\nRecommendations:")
    for key, value in result.recommendations.items():
        print(f"  {key}: {value}")