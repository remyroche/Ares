"""
Data-Driven Clustering Framework

This module provides a comprehensive data-driven approach to clustering that integrates:
1. Similarity matrix clustering with CV confirmation
2. Empirical threshold discovery
3. Price action influence validation
4. Economic relevance measurement

This replaces the traditional KMeans/GMM approach with a data-driven methodology
that answers the key research questions about optimal clustering parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings

from src.utils.logger import system_logger

# Import new clustering components
from .similarity_matrix_clustering import (
    SimilarityMatrixClusterer, 
    SimilarityClusteringConfig, 
    SimilarityMethod,
    similarity_matrix_clustering
)
from .empirical_threshold_discovery import (
    EmpiricalThresholdDiscovery,
    EmpiricalDiscoveryConfig,
    discover_optimal_clustering_thresholds
)

# Import validation components
try:
    from .validation_metrics import RegimeValidationMetrics, ValidationConfig
    from .economic_metrics import EconomicValidator, EconomicValidationConfig
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


@dataclass
class DataDrivenClusteringConfig:
    """Configuration for data-driven clustering framework."""
    # Empirical discovery settings
    enable_threshold_discovery: bool = True
    discovery_config: Optional[EmpiricalDiscoveryConfig] = None
    
    # Similarity clustering settings
    similarity_config: Optional[SimilarityClusteringConfig] = None
    
    # Validation settings
    enable_validation: bool = True
    validation_config: Optional[ValidationConfig] = None
    economic_config: Optional[EconomicValidationConfig] = None
    
    # Performance settings
    cache_results: bool = True
    parallel_processing: bool = True
    verbose: bool = True


@dataclass
class DataDrivenClusteringResult:
    """Comprehensive result container for data-driven clustering."""
    # Core clustering results
    labels: np.ndarray
    n_clusters: int
    
    # Threshold discovery results
    empirical_discovery_result: Optional[Any] = None
    optimal_cv_threshold: Optional[float] = None
    optimal_similarity_threshold: Optional[float] = None
    
    # Similarity clustering results
    similarity_clustering_result: Optional[Any] = None
    cluster_validations: Optional[List] = None
    
    # Validation results
    validation_results: Optional[Dict] = None
    economic_validation_results: Optional[Dict] = None
    
    # Performance metrics
    cluster_quality_metrics: Dict[str, float] = None
    economic_relevance_metrics: Dict[str, float] = None
    
    # Recommendations
    recommendations: Dict[str, Any] = None
    
    # Metadata
    metadata: Dict[str, Any] = None


class DataDrivenClusteringFramework:
    """
    Comprehensive data-driven clustering framework.
    
    This framework provides a complete solution for market regime clustering that:
    1. Empirically discovers optimal CV and similarity thresholds
    2. Uses similarity matrix clustering with CV confirmation
    3. Validates results using economic and statistical metrics
    4. Provides actionable recommendations for ML model training
    """
    
    def __init__(self, config: Optional[DataDrivenClusteringConfig] = None):
        self.config = config or DataDrivenClusteringConfig()
        self.logger = system_logger.getChild('DataDrivenClusteringFramework')
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize framework components."""
        
        # Empirical discovery component
        if self.config.enable_threshold_discovery:
            discovery_config = self.config.discovery_config or EmpiricalDiscoveryConfig()
            self.threshold_discovery = EmpiricalThresholdDiscovery(discovery_config)
        else:
            self.threshold_discovery = None
        
        # Validation components
        if VALIDATION_AVAILABLE and self.config.enable_validation:
            validation_config = self.config.validation_config or ValidationConfig()
            economic_config = self.config.economic_config or EconomicValidationConfig()
            
            self.validator = RegimeValidationMetrics(validation_config)
            self.economic_validator = EconomicValidator(economic_config)
        else:
            self.validator = None
            self.economic_validator = None
    
    def discover_optimal_regimes(self,
                                features: pd.DataFrame,
                                price_data: pd.DataFrame,
                                feature_names: Optional[List[str]] = None) -> DataDrivenClusteringResult:
        """
        Complete data-driven regime discovery pipeline.
        
        Args:
            features: Feature matrix for clustering
            price_data: Price data for economic validation
            feature_names: Optional feature names for interpretation
            
        Returns:
            Comprehensive clustering result with optimal parameters and validation
        """
        self.logger.info("🚀 Starting data-driven regime discovery pipeline")
        
        result = DataDrivenClusteringResult(
            labels=np.array([]),
            n_clusters=0,
            metadata={'pipeline_stages': {}}
        )
        
        try:
            # Stage 1: Empirical threshold discovery
            if self.config.enable_threshold_discovery and self.threshold_discovery:
                self.logger.info("📊 Stage 1: Empirical threshold discovery")
                discovery_result = self.threshold_discovery.discover_optimal_thresholds(
                    features, price_data, feature_names
                )
                
                result.empirical_discovery_result = discovery_result
                result.optimal_cv_threshold = discovery_result.optimal_cv_threshold
                result.optimal_similarity_threshold = discovery_result.optimal_similarity_threshold
                result.metadata['pipeline_stages']['threshold_discovery'] = discovery_result
                
                self.logger.info(f"✅ Optimal thresholds discovered: "
                               f"CV={discovery_result.optimal_cv_threshold:.3f}, "
                               f"Similarity={discovery_result.optimal_similarity_threshold:.3f}")
            
            else:
                # Use default thresholds
                result.optimal_cv_threshold = 0.3
                result.optimal_similarity_threshold = 0.7
                self.logger.info("⚠️ Using default thresholds (threshold discovery disabled)")
            
            # Stage 2: Similarity matrix clustering with optimal parameters
            self.logger.info("🔍 Stage 2: Similarity matrix clustering with optimal parameters")
            
            similarity_config = self.config.similarity_config or SimilarityClusteringConfig()
            similarity_config.cv_threshold = result.optimal_cv_threshold
            similarity_config.similarity_threshold = result.optimal_similarity_threshold
            
            clusterer = SimilarityMatrixClusterer(similarity_config)
            clustering_result = clusterer.fit_predict(features, price_data)
            
            result.similarity_clustering_result = clustering_result
            result.labels = clustering_result.labels
            result.n_clusters = clustering_result.n_clusters
            result.cluster_validations = clustering_result.cluster_validations
            result.metadata['pipeline_stages']['similarity_clustering'] = clustering_result
            
            self.logger.info(f"✅ Clustering completed: {result.n_clusters} clusters discovered")
            
            # Stage 3: Comprehensive validation
            if self.config.enable_validation and result.n_clusters > 1:
                self.logger.info("✅ Stage 3: Comprehensive validation")
                
                # Statistical validation
                if self.validator:
                    validation_results = self.validator.validate_all_metrics(features, result.labels)
                    result.validation_results = validation_results
                    result.metadata['pipeline_stages']['statistical_validation'] = validation_results
                
                # Economic validation
                if self.economic_validator:
                    economic_results = self.economic_validator.validate_regime_economics(
                        price_data, result.labels
                    )
                    result.economic_validation_results = economic_results
                    result.metadata['pipeline_stages']['economic_validation'] = economic_results
                
                self.logger.info("✅ Validation completed")
            
            # Stage 4: Generate metrics and recommendations
            self.logger.info("📈 Stage 4: Generating metrics and recommendations")
            
            result.cluster_quality_metrics = self._calculate_cluster_quality_metrics(
                features, result.labels, clustering_result
            )
            
            result.economic_relevance_metrics = self._calculate_economic_relevance_metrics(
                price_data, result.labels, result.economic_validation_results
            )
            
            result.recommendations = self._generate_recommendations(
                result, features, price_data
            )
            
            result.metadata['pipeline_completed'] = True
            result.metadata['success'] = True
            
            self.logger.info("🎯 Data-driven regime discovery completed successfully")
            
            # Log summary
            self._log_results_summary(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Data-driven clustering failed: {e}")
            result.metadata['error'] = str(e)
            result.metadata['success'] = False
            return result
    
    def _calculate_cluster_quality_metrics(self,
                                         features: pd.DataFrame,
                                         labels: np.ndarray,
                                         clustering_result: Any) -> Dict[str, float]:
        """Calculate cluster quality metrics."""
        
        metrics = {}
        
        try:
            # Basic cluster metrics
            unique_labels = np.unique(labels)
            cluster_sizes = np.bincount(labels)
            
            metrics['n_clusters'] = len(unique_labels)
            metrics['min_cluster_size'] = int(np.min(cluster_sizes))
            metrics['max_cluster_size'] = int(np.max(cluster_sizes))
            metrics['mean_cluster_size'] = float(np.mean(cluster_sizes))
            metrics['cluster_size_cv'] = float(np.std(cluster_sizes) / np.mean(cluster_sizes))
            
            # CV scores from similarity clustering
            if hasattr(clustering_result, 'final_cv_scores'):
                cv_scores = list(clustering_result.final_cv_scores.values())
                metrics['mean_cv_score'] = float(np.mean(cv_scores))
                metrics['max_cv_score'] = float(np.max(cv_scores))
                
            # Similarity scores
            if hasattr(clustering_result, 'final_similarity_scores'):
                sim_scores = list(clustering_result.final_similarity_scores.values())
                metrics['mean_similarity_score'] = float(np.mean(sim_scores))
                metrics['min_similarity_score'] = float(np.min(sim_scores))
            
            # Silhouette score
            try:
                from sklearn.metrics import silhouette_score
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features.fillna(0))
                metrics['silhouette_score'] = float(silhouette_score(features_scaled, labels))
            except:
                metrics['silhouette_score'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate some cluster quality metrics: {e}")
        
        return metrics
    
    def _calculate_economic_relevance_metrics(self,
                                            price_data: pd.DataFrame,
                                            labels: np.ndarray,
                                            economic_validation_results: Optional[Dict]) -> Dict[str, float]:
        """Calculate economic relevance metrics."""
        
        metrics = {}
        
        try:
            # Basic economic metrics
            if 'close' in price_data.columns:
                returns = price_data['close'].pct_change().dropna()
                
                # Overall metrics
                metrics['overall_sharpe'] = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
                metrics['overall_volatility'] = float(returns.std() * np.sqrt(252))
                
                # Regime-specific metrics
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    regime_sharpes = []
                    regime_vols = []
                    
                    for label in unique_labels:
                        mask = labels == label
                        if np.sum(mask) > 10:  # Need sufficient data
                            regime_returns = returns[mask[:len(returns)]]  # Align lengths
                            if len(regime_returns) > 0 and regime_returns.std() > 0:
                                sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                                vol = regime_returns.std() * np.sqrt(252)
                                regime_sharpes.append(sharpe)
                                regime_vols.append(vol)
                    
                    if regime_sharpes:
                        metrics['sharpe_ratio_range'] = float(max(regime_sharpes) - min(regime_sharpes))
                        metrics['volatility_range'] = float(max(regime_vols) - min(regime_vols))
            
            # Extract metrics from economic validation
            if economic_validation_results:
                economically_significant = sum(
                    1 for result in economic_validation_results.values() 
                    if hasattr(result, 'economic_significance') and result.economic_significance
                )
                metrics['economic_significance_rate'] = float(economically_significant / len(economic_validation_results))
                
                # Extract key economic metrics
                for metric_name, result in economic_validation_results.items():
                    if hasattr(result, 'value'):
                        metrics[f'economic_{metric_name.value}'] = float(result.value)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate some economic relevance metrics: {e}")
        
        return metrics
    
    def _generate_recommendations(self,
                                result: DataDrivenClusteringResult,
                                features: pd.DataFrame,
                                price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate actionable recommendations based on results."""
        
        recommendations = {
            'clustering_approach': 'similarity_matrix_clustering',
            'model_training_strategy': 'single_model',
            'confidence_level': 'low',
            'key_insights': [],
            'action_items': []
        }
        
        try:
            # Determine model training strategy
            if result.n_clusters >= 2:
                # Check cluster quality
                quality_score = 0.0
                if result.cluster_quality_metrics:
                    silhouette = result.cluster_quality_metrics.get('silhouette_score', 0)
                    cv_score = result.cluster_quality_metrics.get('mean_cv_score', float('inf'))
                    
                    if silhouette > 0.3 and cv_score < 0.5:
                        quality_score += 0.5
                
                # Check economic relevance
                economic_score = 0.0
                if result.economic_relevance_metrics:
                    sig_rate = result.economic_relevance_metrics.get('economic_significance_rate', 0)
                    sharpe_range = result.economic_relevance_metrics.get('sharpe_ratio_range', 0)
                    
                    if sig_rate > 0.5:
                        economic_score += 0.3
                    if sharpe_range > 0.5:
                        economic_score += 0.2
                
                # Overall confidence
                overall_score = quality_score + economic_score
                
                if overall_score > 0.6:
                    recommendations['model_training_strategy'] = 'separate_models_per_regime'
                    recommendations['confidence_level'] = 'high'
                    recommendations['key_insights'].append('Strong regime separation with economic significance')
                    recommendations['action_items'].append('Train separate ML models for each regime')
                    
                elif overall_score > 0.3:
                    recommendations['model_training_strategy'] = 'regime_aware_features'
                    recommendations['confidence_level'] = 'medium'
                    recommendations['key_insights'].append('Moderate regime separation - use as features')
                    recommendations['action_items'].append('Include regime labels as features in single model')
                    
                else:
                    recommendations['model_training_strategy'] = 'single_model'
                    recommendations['confidence_level'] = 'low'
                    recommendations['key_insights'].append('Weak regime separation - single model recommended')
                    recommendations['action_items'].append('Use single ML model without regime separation')
            
            # Add threshold insights
            if result.empirical_discovery_result:
                if result.empirical_discovery_result.cv_breaking_point:
                    recommendations['key_insights'].append(
                        f"CV breaking point identified at {result.empirical_discovery_result.cv_breaking_point:.3f}"
                    )
                if result.empirical_discovery_result.similarity_breaking_point:
                    recommendations['key_insights'].append(
                        f"Similarity breaking point identified at {result.empirical_discovery_result.similarity_breaking_point:.3f}"
                    )
            
            # Add cluster-specific recommendations
            if result.n_clusters > 1:
                recommendations['action_items'].append(f'Monitor {result.n_clusters} distinct market regimes')
                
                if result.cluster_validations:
                    valid_clusters = sum(1 for cv in result.cluster_validations if cv.is_valid)
                    if valid_clusters < result.n_clusters:
                        recommendations['action_items'].append(
                            f'Consider merging {result.n_clusters - valid_clusters} invalid clusters'
                        )
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some recommendations: {e}")
            recommendations['error'] = str(e)
        
        return recommendations
    
    def _log_results_summary(self, result: DataDrivenClusteringResult):
        """Log summary of results."""
        
        self.logger.info("📊 === DATA-DRIVEN CLUSTERING RESULTS SUMMARY ===")
        self.logger.info(f"🎯 Clusters Discovered: {result.n_clusters}")
        
        if result.optimal_cv_threshold and result.optimal_similarity_threshold:
            self.logger.info(f"⚙️ Optimal Thresholds: CV={result.optimal_cv_threshold:.3f}, "
                           f"Similarity={result.optimal_similarity_threshold:.3f}")
        
        if result.cluster_quality_metrics:
            silhouette = result.cluster_quality_metrics.get('silhouette_score', 0)
            mean_cv = result.cluster_quality_metrics.get('mean_cv_score', 0)
            self.logger.info(f"📈 Quality Metrics: Silhouette={silhouette:.3f}, Mean CV={mean_cv:.3f}")
        
        if result.economic_relevance_metrics:
            sig_rate = result.economic_relevance_metrics.get('economic_significance_rate', 0)
            self.logger.info(f"💰 Economic Significance Rate: {sig_rate:.1%}")
        
        if result.recommendations:
            strategy = result.recommendations.get('model_training_strategy', 'unknown')
            confidence = result.recommendations.get('confidence_level', 'unknown')
            self.logger.info(f"🎯 Recommended Strategy: {strategy} (confidence: {confidence})")
        
        self.logger.info("📊 === END SUMMARY ===")
    
    def quick_discovery(self,
                       features: pd.DataFrame,
                       price_data: pd.DataFrame,
                       use_defaults: bool = True) -> DataDrivenClusteringResult:
        """
        Quick regime discovery using default parameters.
        
        Args:
            features: Feature matrix
            price_data: Price data
            use_defaults: Whether to use default thresholds (faster)
            
        Returns:
            Clustering result
        """
        self.logger.info("⚡ Running quick regime discovery")
        
        if use_defaults:
            # Disable threshold discovery for speed
            original_config = self.config.enable_threshold_discovery
            self.config.enable_threshold_discovery = False
            
            result = self.discover_optimal_regimes(features, price_data)
            
            # Restore original config
            self.config.enable_threshold_discovery = original_config
            
            return result
        else:
            return self.discover_optimal_regimes(features, price_data)


# Convenience functions
def data_driven_regime_discovery(features: pd.DataFrame,
                                price_data: pd.DataFrame,
                                config: Optional[DataDrivenClusteringConfig] = None) -> DataDrivenClusteringResult:
    """
    Convenience function for data-driven regime discovery.
    
    Args:
        features: Feature matrix for clustering
        price_data: Price data for economic validation
        config: Optional configuration
        
    Returns:
        Comprehensive clustering result
    """
    framework = DataDrivenClusteringFramework(config)
    return framework.discover_optimal_regimes(features, price_data)


def quick_regime_discovery(features: pd.DataFrame,
                          price_data: pd.DataFrame) -> DataDrivenClusteringResult:
    """
    Quick regime discovery with default parameters.
    
    Args:
        features: Feature matrix
        price_data: Price data
        
    Returns:
        Clustering result
    """
    framework = DataDrivenClusteringFramework()
    return framework.quick_discovery(features, price_data, use_defaults=True)


# Example usage
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create feature groups with different correlation structures
    features_data = []
    
    # Group 1: Momentum features (high correlation)
    momentum_base = np.random.randn(n_samples)
    for i in range(5):
        feature = momentum_base + np.random.randn(n_samples) * 0.2
        features_data.append(feature)
    
    # Group 2: Volatility features (high correlation)
    vol_base = np.random.randn(n_samples)
    for i in range(5):
        feature = vol_base + np.random.randn(n_samples) * 0.2
        features_data.append(feature)
    
    # Group 3: Volume features (medium correlation)
    volume_base = np.random.randn(n_samples)
    for i in range(5):
        feature = volume_base + np.random.randn(n_samples) * 0.5
        features_data.append(feature)
    
    # Create DataFrame
    features = pd.DataFrame(
        np.column_stack(features_data),
        columns=[f'feature_{i}' for i in range(15)]
    )
    
    # Create price data with regime-dependent behavior
    regime_factor = features.iloc[:, :5].mean(axis=1)  # Influenced by momentum features
    base_returns = np.random.randn(n_samples) * 0.02
    regime_returns = base_returns + regime_factor * 0.005  # Small regime influence
    
    prices = 100 * np.exp(np.cumsum(regime_returns))
    price_data = pd.DataFrame({
        'close': prices,
        'returns': regime_returns
    })
    
    # Test the framework
    config = DataDrivenClusteringConfig(
        enable_threshold_discovery=True,
        discovery_config=EmpiricalDiscoveryConfig(
            cv_range=(0.2, 0.6, 8),
            similarity_range=(0.5, 0.9, 8),
            early_stopping=True
        ),
        verbose=True
    )
    
    print("🚀 Testing Data-Driven Clustering Framework")
    result = data_driven_regime_discovery(features, price_data, config)
    
    print(f"\n🎯 Results Summary:")
    print(f"Clusters discovered: {result.n_clusters}")
    print(f"Optimal CV threshold: {result.optimal_cv_threshold:.3f}")
    print(f"Optimal similarity threshold: {result.optimal_similarity_threshold:.3f}")
    print(f"Model training strategy: {result.recommendations['model_training_strategy']}")
    print(f"Confidence level: {result.recommendations['confidence_level']}")
    
    print(f"\n📊 Key Insights:")
    for insight in result.recommendations['key_insights']:
        print(f"  - {insight}")
    
    print(f"\n📋 Action Items:")
    for item in result.recommendations['action_items']:
        print(f"  - {item}")