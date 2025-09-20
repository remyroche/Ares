"""
Cluster Validation Metrics for HMM Market Regime Clustering

This module provides comprehensive validation metrics to assess cluster quality
before accepting clustering results.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway, kruskal
import warnings
from typing import Dict, List, Tuple, Optional
import logging

class ClusterValidator:
    """
    Comprehensive cluster validation system for market regime clustering
    """
    
    def __init__(self, min_cluster_size: int = 50, max_cluster_size_ratio: float = 0.2):
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size_ratio = max_cluster_size_ratio
        self.validation_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_clustering(self, 
                          data: np.ndarray, 
                          cluster_labels: np.ndarray,
                          regime_features: Optional[Dict] = None) -> Dict:
        """
        Comprehensive cluster validation
        
        Args:
            data: Feature matrix (n_samples, n_features)
            cluster_labels: Cluster assignments (n_samples,)
            regime_features: Additional regime characteristics
            
        Returns:
            Dict with validation results and recommendations
        """
        
        self.logger.info("Starting comprehensive cluster validation...")
        
        validation_results = {
            'statistical_metrics': self._calculate_statistical_metrics(data, cluster_labels),
            'cluster_quality': self._assess_cluster_quality(data, cluster_labels),
            'size_distribution': self._analyze_size_distribution(cluster_labels),
            'economic_validation': self._validate_economic_interpretability(data, cluster_labels, regime_features),
            'stability_analysis': self._analyze_cluster_stability(data, cluster_labels),
            'recommendations': []
        }
        
        # Generate recommendations based on validation results
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        # Overall validation score
        validation_results['overall_score'] = self._calculate_overall_score(validation_results)
        
        # Pass/Fail decision
        validation_results['validation_passed'] = validation_results['overall_score'] >= 0.7
        
        self.validation_results = validation_results
        return validation_results
    
    def _calculate_statistical_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate standard clustering validation metrics"""
        
        try:
            # Silhouette Score (higher is better, range [-1, 1])
            silhouette = silhouette_score(data, labels)
            
            # Calinski-Harabasz Score (higher is better)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            
            # Davies-Bouldin Score (lower is better)
            davies_bouldin = davies_bouldin_score(data, labels)
            
            # Inertia (within-cluster sum of squares)
            inertia = self._calculate_inertia(data, labels)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'inertia': inertia,
                'silhouette_interpretation': self._interpret_silhouette(silhouette),
                'davies_bouldin_interpretation': self._interpret_davies_bouldin(davies_bouldin)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical metrics: {e}")
            return {'error': str(e)}
    
    def _assess_cluster_quality(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Assess overall cluster quality"""
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        quality_metrics = {
            'n_clusters': n_clusters,
            'cluster_separation': self._calculate_cluster_separation(data, labels),
            'cluster_compactness': self._calculate_cluster_compactness(data, labels),
            'noise_ratio': self._calculate_noise_ratio(labels),
            'balance_score': self._calculate_balance_score(labels)
        }
        
        return quality_metrics
    
    def _analyze_size_distribution(self, labels: np.ndarray) -> Dict:
        """Analyze cluster size distribution"""
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        
        size_analysis = {
            'cluster_sizes': dict(zip(unique_labels, counts)),
            'size_statistics': {
                'min_size': int(np.min(counts)),
                'max_size': int(np.max(counts)),
                'mean_size': float(np.mean(counts)),
                'median_size': float(np.median(counts)),
                'std_size': float(np.std(counts))
            },
            'size_violations': {
                'too_small_clusters': [int(label) for label, size in zip(unique_labels, counts) 
                                     if size < self.min_cluster_size],
                'too_large_clusters': [int(label) for label, size in zip(unique_labels, counts) 
                                     if size > total_samples * self.max_cluster_size_ratio],
                'micro_clusters': [int(label) for label, size in zip(unique_labels, counts) 
                                 if size < 10]  # Extremely small clusters
            },
            'size_distribution_score': self._calculate_size_distribution_score(counts, total_samples)
        }
        
        return size_analysis
    
    def _validate_economic_interpretability(self, data: np.ndarray, labels: np.ndarray, 
                                          regime_features: Optional[Dict] = None) -> Dict:
        """Validate economic interpretability of clusters"""
        
        if regime_features is None:
            return {'status': 'skipped', 'reason': 'No regime features provided'}
        
        economic_validation = {
            'market_regime_coverage': self._analyze_market_regime_coverage(labels, regime_features),
            'feature_significance': self._test_feature_significance(data, labels),
            'regime_coherence': self._assess_regime_coherence(labels, regime_features),
            'trading_actionability': self._assess_trading_actionability(labels, regime_features)
        }
        
        return economic_validation
    
    def _analyze_cluster_stability(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Analyze cluster stability using bootstrap resampling"""
        
        n_bootstrap = 100
        stability_scores = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(data)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = data[bootstrap_indices]
            bootstrap_labels = labels[bootstrap_indices]
            
            # Calculate silhouette score for bootstrap sample
            try:
                bootstrap_silhouette = silhouette_score(bootstrap_data, bootstrap_labels)
                stability_scores.append(bootstrap_silhouette)
            except:
                continue
        
        stability_analysis = {
            'bootstrap_silhouette_scores': stability_scores,
            'stability_mean': float(np.mean(stability_scores)),
            'stability_std': float(np.std(stability_scores)),
            'stability_score': 1.0 - (np.std(stability_scores) / max(np.mean(stability_scores), 0.1))
        }
        
        return stability_analysis
    
    def _calculate_inertia(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)"""
        
        inertia = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_data = data[labels == label]
            if len(cluster_data) > 0:
                centroid = np.mean(cluster_data, axis=0)
                inertia += np.sum((cluster_data - centroid) ** 2)
        
        return float(inertia)
    
    def _calculate_cluster_separation(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate average distance between cluster centroids"""
        
        unique_labels = np.unique(labels)
        centroids = []
        
        for label in unique_labels:
            cluster_data = data[labels == label]
            if len(cluster_data) > 0:
                centroids.append(np.mean(cluster_data, axis=0))
        
        if len(centroids) < 2:
            return 0.0
        
        centroids = np.array(centroids)
        distances = pdist(centroids)
        return float(np.mean(distances))
    
    def _calculate_cluster_compactness(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate average within-cluster distance"""
        
        unique_labels = np.unique(labels)
        compactness_scores = []
        
        for label in unique_labels:
            cluster_data = data[labels == label]
            if len(cluster_data) > 1:
                distances = pdist(cluster_data)
                compactness_scores.append(np.mean(distances))
        
        return float(np.mean(compactness_scores)) if compactness_scores else 0.0
    
    def _calculate_noise_ratio(self, labels: np.ndarray) -> float:
        """Calculate ratio of samples in noise clusters (very small clusters)"""
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        noise_samples = np.sum(counts[counts < 5])  # Clusters with <5 samples
        return float(noise_samples / len(labels))
    
    def _calculate_balance_score(self, labels: np.ndarray) -> float:
        """Calculate cluster balance score (1.0 = perfectly balanced)"""
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        expected_size = len(labels) / len(unique_labels)
        
        # Calculate coefficient of variation
        cv = np.std(counts) / np.mean(counts)
        
        # Convert to balance score (lower CV = higher balance)
        balance_score = 1.0 / (1.0 + cv)
        return float(balance_score)
    
    def _calculate_size_distribution_score(self, counts: np.ndarray, total_samples: int) -> float:
        """Calculate score for cluster size distribution quality"""
        
        # Penalize very small and very large clusters
        size_violations = 0
        
        for count in counts:
            if count < self.min_cluster_size:
                size_violations += (self.min_cluster_size - count) / self.min_cluster_size
            elif count > total_samples * self.max_cluster_size_ratio:
                size_violations += (count - total_samples * self.max_cluster_size_ratio) / count
        
        # Normalize by number of clusters
        violation_ratio = size_violations / len(counts)
        
        # Convert to score (0 = many violations, 1 = no violations)
        return float(max(0.0, 1.0 - violation_ratio))
    
    def _test_feature_significance(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Test statistical significance of features across clusters"""
        
        unique_labels = np.unique(labels)
        n_features = data.shape[1]
        
        significant_features = []
        p_values = []
        
        for feature_idx in range(n_features):
            feature_data = data[:, feature_idx]
            groups = [feature_data[labels == label] for label in unique_labels]
            
            # Remove empty groups
            groups = [group for group in groups if len(group) > 0]
            
            if len(groups) >= 2:
                try:
                    # Use Kruskal-Wallis test (non-parametric)
                    statistic, p_value = kruskal(*groups)
                    p_values.append(p_value)
                    
                    if p_value < 0.05:  # Significant at 5% level
                        significant_features.append(feature_idx)
                except:
                    p_values.append(1.0)  # Not significant
        
        return {
            'significant_features': significant_features,
            'p_values': p_values,
            'significance_ratio': len(significant_features) / n_features if n_features > 0 else 0.0
        }
    
    def _interpret_silhouette(self, score: float) -> str:
        """Interpret silhouette score"""
        if score > 0.7:
            return "Excellent clustering"
        elif score > 0.5:
            return "Good clustering"
        elif score > 0.25:
            return "Weak clustering"
        else:
            return "Poor clustering"
    
    def _interpret_davies_bouldin(self, score: float) -> str:
        """Interpret Davies-Bouldin score"""
        if score < 0.5:
            return "Excellent separation"
        elif score < 1.0:
            return "Good separation"
        elif score < 1.5:
            return "Moderate separation"
        else:
            return "Poor separation"
    
    def _analyze_market_regime_coverage(self, labels: np.ndarray, regime_features: Dict) -> Dict:
        """Analyze coverage of different market regimes"""
        # This would be implemented based on your specific regime features
        return {'status': 'not_implemented', 'reason': 'Requires specific regime feature structure'}
    
    def _assess_regime_coherence(self, labels: np.ndarray, regime_features: Dict) -> Dict:
        """Assess coherence of regimes within clusters"""
        # This would be implemented based on your specific regime features
        return {'status': 'not_implemented', 'reason': 'Requires specific regime feature structure'}
    
    def _assess_trading_actionability(self, labels: np.ndarray, regime_features: Dict) -> Dict:
        """Assess whether clusters lead to actionable trading insights"""
        # This would be implemented based on your specific regime features
        return {'status': 'not_implemented', 'reason': 'Requires specific regime feature structure'}
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Statistical metrics recommendations
        stats = validation_results.get('statistical_metrics', {})
        if stats.get('silhouette_score', 0) < 0.3:
            recommendations.append("Poor cluster separation detected. Consider reducing number of clusters or improving features.")
        
        if stats.get('davies_bouldin_score', float('inf')) > 1.5:
            recommendations.append("High cluster overlap detected. Consider merging similar clusters.")
        
        # Size distribution recommendations
        size_analysis = validation_results.get('size_distribution', {})
        violations = size_analysis.get('size_violations', {})
        
        if violations.get('too_small_clusters'):
            recommendations.append(f"Merge {len(violations['too_small_clusters'])} clusters with insufficient samples.")
        
        if violations.get('too_large_clusters'):
            recommendations.append(f"Consider splitting {len(violations['too_large_clusters'])} oversized clusters.")
        
        if violations.get('micro_clusters'):
            recommendations.append(f"Remove or merge {len(violations['micro_clusters'])} micro-clusters.")
        
        # Stability recommendations
        stability = validation_results.get('stability_analysis', {})
        if stability.get('stability_score', 0) < 0.6:
            recommendations.append("Low cluster stability detected. Consider more robust clustering parameters.")
        
        return recommendations
    
    def _calculate_overall_score(self, validation_results: Dict) -> float:
        """Calculate overall validation score"""
        
        scores = []
        weights = []
        
        # Statistical metrics (weight: 0.3)
        stats = validation_results.get('statistical_metrics', {})
        if 'silhouette_score' in stats:
            # Normalize silhouette score from [-1,1] to [0,1]
            normalized_silhouette = (stats['silhouette_score'] + 1) / 2
            scores.append(normalized_silhouette)
            weights.append(0.3)
        
        # Size distribution (weight: 0.3)
        size_analysis = validation_results.get('size_distribution', {})
        if 'size_distribution_score' in size_analysis:
            scores.append(size_analysis['size_distribution_score'])
            weights.append(0.3)
        
        # Cluster quality (weight: 0.2)
        quality = validation_results.get('cluster_quality', {})
        if 'balance_score' in quality:
            scores.append(quality['balance_score'])
            weights.append(0.2)
        
        # Stability (weight: 0.2)
        stability = validation_results.get('stability_analysis', {})
        if 'stability_score' in stability:
            scores.append(stability['stability_score'])
            weights.append(0.2)
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
        
        return float(weighted_score)

    def print_validation_report(self) -> None:
        """Print a comprehensive validation report"""
        
        if not self.validation_results:
            print("No validation results available. Run validate_clustering() first.")
            return
        
        results = self.validation_results
        
        print("=" * 60)
        print("CLUSTER VALIDATION REPORT")
        print("=" * 60)
        
        # Overall assessment
        print(f"\nOVERALL SCORE: {results['overall_score']:.3f}")
        print(f"VALIDATION STATUS: {'PASSED' if results['validation_passed'] else 'FAILED'}")
        
        # Statistical metrics
        if 'statistical_metrics' in results:
            stats = results['statistical_metrics']
            print(f"\nSTATISTICAL METRICS:")
            print(f"  Silhouette Score: {stats.get('silhouette_score', 'N/A'):.3f} ({stats.get('silhouette_interpretation', 'N/A')})")
            print(f"  Davies-Bouldin Score: {stats.get('davies_bouldin_score', 'N/A'):.3f} ({stats.get('davies_bouldin_interpretation', 'N/A')})")
            print(f"  Calinski-Harabasz Score: {stats.get('calinski_harabasz_score', 'N/A'):.1f}")
        
        # Size distribution
        if 'size_distribution' in results:
            size = results['size_distribution']
            print(f"\nCLUSTER SIZE ANALYSIS:")
            print(f"  Total Clusters: {size['size_statistics']['min_size']}")
            print(f"  Size Range: {size['size_statistics']['min_size']} - {size['size_statistics']['max_size']}")
            print(f"  Mean Size: {size['size_statistics']['mean_size']:.1f}")
            print(f"  Too Small Clusters: {len(size['size_violations']['too_small_clusters'])}")
            print(f"  Too Large Clusters: {len(size['size_violations']['too_large_clusters'])}")
            print(f"  Micro Clusters: {len(size['size_violations']['micro_clusters'])}")
        
        # Recommendations
        if results['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("=" * 60)


# Example usage function
def validate_clustering_example():
    """Example of how to use the cluster validator"""
    
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    data = np.random.randn(n_samples, n_features)
    
    # Generate example cluster labels (simulating your HMM clustering results)
    cluster_labels = np.random.randint(0, 50, n_samples)  # 50 clusters
    
    # Initialize validator
    validator = ClusterValidator(min_cluster_size=20, max_cluster_size_ratio=0.15)
    
    # Run validation
    results = validator.validate_clustering(data, cluster_labels)
    
    # Print report
    validator.print_validation_report()
    
    return results

if __name__ == "__main__":
    validate_clustering_example()