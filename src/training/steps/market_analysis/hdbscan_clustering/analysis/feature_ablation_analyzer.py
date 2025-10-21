"""
Feature Ablation Analysis System

This module provides comprehensive feature ablation analysis to determine
the importance of different feature groups in the clustering process.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import clustering components
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import umap

# Import tprint utilities for extensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureGroup:
    """Feature group definition."""
    name: str
    feature_indices: List[int]
    feature_names: List[str]
    description: str


@dataclass
class AblationResult:
    """Result of feature ablation analysis."""
    feature_group: str
    original_score: float
    ablated_score: float
    score_delta: float
    score_delta_pct: float
    n_features_removed: int
    n_clusters_original: int
    n_clusters_ablated: int
    cluster_stability: float
    timestamp: datetime


@dataclass
class FeatureImportanceReport:
    """Comprehensive feature importance report."""
    timestamp: datetime
    n_total_features: int
    n_feature_groups: int
    baseline_score: float
    ablation_results: List[AblationResult]
    feature_importance_ranking: List[Tuple[str, float]]
    composite_score_impact: Dict[str, float]
    recommendations: List[str]


class FeatureAblationAnalyzer:
    """
    Feature ablation analyzer for determining feature group importance.
    
    Analyzes the impact of removing each feature group on:
    - Clustering quality metrics
    - Economic validation scores
    - Cluster stability
    - Composite scoring
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, 
                 output_dir: str = "ablation_analysis",
                 enable_auto_reporting: bool = True):
        """
        Initialize the feature ablation analyzer.
        
        Args:
            output_dir: Directory for analysis outputs
            enable_auto_reporting: Whether to enable automatic reporting
        """
        tprint_info("🔧 Initializing Feature Ablation Analyzer")
        start_time = time.perf_counter()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tprint_debug(f"Output directory: {self.output_dir}")
        
        self.enable_auto_reporting = enable_auto_reporting
        tprint_debug(f"Auto-reporting enabled: {enable_auto_reporting}")
        
        # Feature group definitions
        self.feature_groups = self._define_feature_groups()
        tprint_debug(f"Defined {len(self.feature_groups)} feature groups")
        
        # Analysis results
        self.ablation_results: List[AblationResult] = []
        self.baseline_score = 0.0
        self.baseline_clusters = None
        
        init_time = time.perf_counter() - start_time
        tprint_success(f"✅ Feature Ablation Analyzer initialized in {init_time:.3f}s")
        
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def _define_feature_groups(self) -> Dict[str, FeatureGroup]:
        """Define feature groups for ablation analysis."""
        tprint_debug("📋 Defining feature groups for ablation analysis")
        
        groups = {
            'returns': FeatureGroup(
                name='returns',
                feature_indices=[],  # Will be populated dynamically
                feature_names=[],
                description='Return-based features (price returns, log returns, etc.)'
            ),
            'volatility': FeatureGroup(
                name='volatility',
                feature_indices=[],
                feature_names=[],
                description='Volatility-based features (rolling std, GARCH, etc.)'
            ),
            'volume': FeatureGroup(
                name='volume',
                feature_indices=[],
                feature_names=[],
                description='Volume-based features (RVOL, volume momentum, etc.)'
            ),
            'risk': FeatureGroup(
                name='risk',
                feature_indices=[],
                feature_names=[],
                description='Risk-based features (VaR, CVaR, drawdowns, etc.)'
            ),
            'technical': FeatureGroup(
                name='technical',
                feature_indices=[],
                feature_names=[],
                description='Technical indicators (RSI, MACD, Bollinger Bands, etc.)'
            ),
            'macro': FeatureGroup(
                name='macro',
                feature_indices=[],
                feature_names=[],
                description='Macro-economic features (yield curves, spreads, etc.)'
            )
        }
        
        tprint_debug(f"Defined {len(groups)} feature groups: {list(groups.keys())}")
        return groups
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[int]]:
        """
        Categorize features into groups based on naming patterns.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping group names to feature indices
        """
        tprint_debug(f"🔍 Categorizing {len(feature_names)} features into groups")
        
        group_indices = {group_name: [] for group_name in self.feature_groups.keys()}
        
        for i, feature_name in enumerate(feature_names):
            feature_lower = feature_name.lower()
            
            # Categorize based on naming patterns
            if any(keyword in feature_lower for keyword in ['return', 'ret_', 'log_ret', 'price_change']):
                group_indices['returns'].append(i)
            elif any(keyword in feature_lower for keyword in ['vol', 'volatility', 'std', 'garch', 'ewm_std']):
                group_indices['volatility'].append(i)
            elif any(keyword in feature_lower for keyword in ['volume', 'vol_', 'rvol', 'vol_mom', 'vol_corr']):
                group_indices['volume'].append(i)
            elif any(keyword in feature_lower for keyword in ['var', 'cvar', 'drawdown', 'risk', 'skew', 'kurt']):
                group_indices['risk'].append(i)
            elif any(keyword in feature_lower for keyword in ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'bb_']):
                group_indices['technical'].append(i)
            elif any(keyword in feature_lower for keyword in ['yield', 'spread', 'macro', 'economic', 'gdp']):
                group_indices['macro'].append(i)
            else:
                # Default to technical if no clear category
                group_indices['technical'].append(i)
        
        # Log categorization results
        for group_name, indices in group_indices.items():
            if indices:
                tprint_debug(f"  {group_name}: {len(indices)} features")
        
        tprint_success(f"✅ Feature categorization completed: {sum(len(indices) for indices in group_indices.values())} features categorized")
        return group_indices
    
    def _calculate_composite_score(self, 
                                 cluster_labels: np.ndarray,
                                 features: np.ndarray,
                                 market_data: pd.DataFrame) -> float:
        """
        Calculate composite score combining clustering quality and economic validity.
        
        Args:
            cluster_labels: Cluster labels
            features: Feature matrix
            market_data: Market data
            
        Returns:
            Composite score
        """
        # Clustering quality metrics
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
            silhouette = silhouette_score(features[valid_mask], cluster_labels[valid_mask])
        else:
            silhouette = 0.0
        
        # Economic validity metrics
        if 'returns' in market_data.columns and len(market_data) > 0:
            returns = market_data['returns'].dropna()
            if len(returns) > 0 and len(cluster_labels) > 0:
                valid_returns = returns.iloc[valid_mask] if len(returns) >= len(cluster_labels) else returns
                valid_labels = cluster_labels[valid_mask] if len(cluster_labels) >= len(valid_returns) else cluster_labels[:len(valid_returns)]
                
                if len(np.unique(valid_labels)) > 1 and len(valid_returns) > 0:
                    # Calculate return separation
                    groups = [valid_returns[valid_labels == label].values 
                             for label in np.unique(valid_labels) if label != -1]
                    if len(groups) > 1 and all(len(g) > 0 for g in groups):
                        from scipy import stats
                        f_stat, p_value = stats.f_oneway(*groups)
                        return_separation = f_stat if not np.isnan(f_stat) else 0.0
                    else:
                        return_separation = 0.0
                else:
                    return_separation = 0.0
            else:
                return_separation = 0.0
        else:
            return_separation = 0.0
        
        # Normalize metrics
        silhouette_norm = max(0, min(1, silhouette))  # 0-1 range
        return_sep_norm = max(0, min(1, return_separation / 10.0))  # Normalize F-statistic
        
        # Composite score (weighted combination)
        composite_score = 0.6 * silhouette_norm + 0.4 * return_sep_norm
        
        return composite_score
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _perform_clustering(self, 
                          features: np.ndarray,
                          n_components: int = 10) -> Tuple[np.ndarray, float]:
        """
        Perform clustering on features.
        
        Args:
            features: Feature matrix
            n_components: Number of PCA components
            
        Returns:
            Tuple of (cluster_labels, silhouette_score)
        """
        try:
            tprint_debug(f"🔍 Performing clustering on {features.shape[0]} samples with {features.shape[1]} features")
            
            # Dimensionality reduction
            if features.shape[1] > n_components:
                tprint_debug(f"Applying PCA reduction: {features.shape[1]} -> {n_components}")
                pca = PCA(n_components=n_components, random_state=42)
                features_reduced = pca.fit_transform(features)
            else:
                tprint_debug("Skipping PCA reduction (features <= n_components)")
                features_reduced = features
            
            # UMAP for non-linear dimensionality reduction
            if features_reduced.shape[1] > 2:
                tprint_debug("Applying UMAP reduction: {features_reduced.shape[1]} -> 2")
                umap_reducer = umap.UMAP(n_components=2, random_state=42)
                features_umap = umap_reducer.fit_transform(features_reduced)
            else:
                tprint_debug("Skipping UMAP reduction (features <= 2)")
                features_umap = features_reduced
            
            # HDBSCAN clustering
            tprint_debug("Performing HDBSCAN clustering")
            clusterer = HDBSCAN(min_cluster_size=50, min_samples=10)
            cluster_labels = clusterer.fit_predict(features_umap)
            
            # Calculate silhouette score
            valid_mask = cluster_labels != -1
            if np.sum(valid_mask) > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
                silhouette = silhouette_score(features_umap[valid_mask], cluster_labels[valid_mask])
                tprint_debug(f"Silhouette score: {silhouette:.3f}")
            else:
                silhouette = 0.0
                tprint_debug("Insufficient valid clusters for silhouette score")
            
            n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
            tprint_debug(f"Clustering completed: {n_clusters} clusters, {np.sum(cluster_labels == -1)} noise points")
            
            return cluster_labels, silhouette
            
        except Exception as e:
            tprint_error(f"Clustering failed: {e}")
            return np.full(len(features), -1), 0.0
    
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def _calculate_cluster_stability(self, 
                                   original_labels: np.ndarray,
                                   ablated_labels: np.ndarray) -> float:
        """
        Calculate cluster stability between original and ablated results.
        
        Args:
            original_labels: Original cluster labels
            ablated_labels: Ablated cluster labels
            
        Returns:
            Stability score (0-1)
        """
        tprint_debug(f"📊 Calculating cluster stability between {len(original_labels)} and {len(ablated_labels)} labels")
        
        if len(original_labels) != len(ablated_labels):
            tprint_warning("Label length mismatch, returning 0.0 stability")
            return 0.0
        
        try:
            # Calculate Adjusted Rand Index
            ari = adjusted_rand_score(original_labels, ablated_labels)
            stability = max(0, ari)  # Ensure non-negative
            tprint_debug(f"Adjusted Rand Index: {ari:.3f}, Stability: {stability:.3f}")
            return stability
        except Exception as e:
            tprint_error(f"Stability calculation failed: {e}")
            return 0.0
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def run_ablation_analysis(self, 
                            features: np.ndarray,
                            feature_names: List[str],
                            market_data: pd.DataFrame,
                            n_components: int = 10) -> FeatureImportanceReport:
        """
        Run comprehensive feature ablation analysis.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            market_data: Market data
            n_components: Number of PCA components
            
        Returns:
            FeatureImportanceReport
        """
        tprint_info("🔬 Starting feature ablation analysis...")
        start_time = time.perf_counter()
        
        # Categorize features
        group_indices = self._categorize_features(feature_names)
        
        # Update feature groups with indices
        for group_name, indices in group_indices.items():
            if group_name in self.feature_groups:
                self.feature_groups[group_name].feature_indices = indices
                self.feature_groups[group_name].feature_names = [feature_names[i] for i in indices]
        
        # Baseline clustering
        tprint_info("📊 Performing baseline clustering...")
        baseline_labels, baseline_silhouette = self._perform_clustering(features, n_components)
        self.baseline_score = self._calculate_composite_score(baseline_labels, features, market_data)
        self.baseline_clusters = baseline_labels
        
        tprint_success(f"Baseline score: {self.baseline_score:.4f}")
        
        # Perform ablation for each feature group
        ablation_results = []
        
        for group_name, group in self.feature_groups.items():
            if not group.feature_indices:  # Skip empty groups
                continue
            
            tprint_info(f"🔍 Ablating feature group: {group_name} ({len(group.feature_indices)} features)")
            
            # Create ablated feature matrix
            ablated_features = np.delete(features, group.feature_indices, axis=1)
            
            # Perform clustering on ablated features
            ablated_labels, ablated_silhouette = self._perform_clustering(ablated_features, n_components)
            
            # Calculate composite score
            ablated_score = self._calculate_composite_score(ablated_labels, ablated_features, market_data)
            
            # Calculate score delta
            score_delta = self.baseline_score - ablated_score
            score_delta_pct = (score_delta / self.baseline_score * 100) if self.baseline_score > 0 else 0.0
            
            # Calculate cluster stability
            stability = self._calculate_cluster_stability(baseline_labels, ablated_labels)
            
            # Count clusters
            n_clusters_original = len(np.unique(baseline_labels[baseline_labels != -1]))
            n_clusters_ablated = len(np.unique(ablated_labels[ablated_labels != -1]))
            
            # Create ablation result
            result = AblationResult(
                feature_group=group_name,
                original_score=self.baseline_score,
                ablated_score=ablated_score,
                score_delta=score_delta,
                score_delta_pct=score_delta_pct,
                n_features_removed=len(group.feature_indices),
                n_clusters_original=n_clusters_original,
                n_clusters_ablated=n_clusters_ablated,
                cluster_stability=stability,
                timestamp=datetime.now()
            )
            
            ablation_results.append(result)
            self.ablation_results.append(result)
            
            tprint_debug(f"  Score delta: {score_delta:.4f} ({score_delta_pct:+.1f}%)")
            tprint_debug(f"  Stability: {stability:.4f}")
        
        # Create feature importance ranking
        feature_importance_ranking = sorted(
            [(result.feature_group, result.score_delta_pct) for result in ablation_results],
            key=lambda x: x[1], reverse=True
        )
        
        # Calculate composite score impact
        composite_score_impact = {
            result.feature_group: result.score_delta_pct 
            for result in ablation_results
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(ablation_results)
        
        # Create comprehensive report
        report = FeatureImportanceReport(
            timestamp=datetime.now(),
            n_total_features=len(feature_names),
            n_feature_groups=len([g for g in self.feature_groups.values() if g.feature_indices]),
            baseline_score=self.baseline_score,
            ablation_results=ablation_results,
            feature_importance_ranking=feature_importance_ranking,
            composite_score_impact=composite_score_impact,
            recommendations=recommendations
        )
        
        # Save report
        self._save_report(report)
        
        # Auto-report if enabled
        if self.enable_auto_reporting:
            self._generate_auto_report(report)
        
        analysis_time = time.perf_counter() - start_time
        tprint_success(f"✅ Feature ablation analysis completed in {analysis_time:.3f}s")
        
        return report
    
    def _generate_recommendations(self, ablation_results: List[AblationResult]) -> List[str]:
        """Generate recommendations based on ablation results."""
        recommendations = []
        
        # Find most important feature groups
        important_groups = [r for r in ablation_results if r.score_delta_pct > 5.0]
        if important_groups:
            group_names = [r.feature_group for r in important_groups]
            recommendations.append(f"High-impact feature groups: {', '.join(group_names)}")
        
        # Find low-impact feature groups
        low_impact_groups = [r for r in ablation_results if r.score_delta_pct < 1.0]
        if low_impact_groups:
            group_names = [r.feature_group for r in low_impact_groups]
            recommendations.append(f"Consider removing low-impact groups: {', '.join(group_names)}")
        
        # Check for stability issues
        unstable_groups = [r for r in ablation_results if r.cluster_stability < 0.5]
        if unstable_groups:
            group_names = [r.feature_group for r in unstable_groups]
            recommendations.append(f"Unstable feature groups (low cluster stability): {', '.join(group_names)}")
        
        # Check for cluster count changes
        cluster_changing_groups = [r for r in ablation_results if abs(r.n_clusters_original - r.n_clusters_ablated) > 1]
        if cluster_changing_groups:
            group_names = [r.feature_group for r in cluster_changing_groups]
            recommendations.append(f"Feature groups affecting cluster count: {', '.join(group_names)}")
        
        return recommendations
    
    def _save_report(self, report: FeatureImportanceReport) -> None:
        """Save feature importance report."""
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"feature_ablation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Feature ablation report saved to {report_file}")
    
    def _generate_auto_report(self, report: FeatureImportanceReport) -> None:
        """Generate automatic feature importance dashboard report."""
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        dashboard_file = self.output_dir / f"feature_importance_dashboard_{timestamp}.md"
        
        # Create markdown report
        markdown_content = f"""# Feature Importance Dashboard

**Generated:** {report.timestamp}
**Baseline Score:** {report.baseline_score:.4f}
**Total Features:** {report.n_total_features}
**Feature Groups:** {report.n_feature_groups}

## Feature Group Impact Analysis

| Feature Group | Score Delta | Impact % | Features Removed | Stability | Clusters (Orig→Abl) |
|---------------|-------------|----------|------------------|-----------|---------------------|
"""
        
        for result in report.ablation_results:
            markdown_content += f"| {result.feature_group} | {result.score_delta:.4f} | {result.score_delta_pct:+.1f}% | {result.n_features_removed} | {result.cluster_stability:.3f} | {result.n_clusters_original}→{result.n_clusters_ablated} |\n"
        
        markdown_content += f"""

## Feature Importance Ranking

"""
        
        for i, (group_name, impact_pct) in enumerate(report.feature_importance_ranking, 1):
            markdown_content += f"{i}. **{group_name}**: {impact_pct:+.1f}% impact\n"
        
        markdown_content += f"""

## Recommendations

"""
        
        for i, recommendation in enumerate(report.recommendations, 1):
            markdown_content += f"{i}. {recommendation}\n"
        
        markdown_content += f"""

## Composite Score Impact

"""
        
        for group_name, impact_pct in report.composite_score_impact.items():
            markdown_content += f"- **{group_name}**: {impact_pct:+.1f}% impact on composite score\n"
        
        # Save dashboard
        with open(dashboard_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Feature importance dashboard saved to {dashboard_file}")
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get feature importance summary."""
        if not self.ablation_results:
            return {'message': 'No ablation analysis performed yet'}
        
        return {
            'timestamp': datetime.now(),
            'baseline_score': self.baseline_score,
            'n_feature_groups': len(self.ablation_results),
            'most_important_group': max(self.ablation_results, key=lambda x: x.score_delta_pct).feature_group,
            'least_important_group': min(self.ablation_results, key=lambda x: x.score_delta_pct).feature_group,
            'avg_score_delta': np.mean([r.score_delta_pct for r in self.ablation_results]),
            'max_score_delta': max([r.score_delta_pct for r in self.ablation_results]),
            'min_score_delta': min([r.score_delta_pct for r in self.ablation_results]),
            'avg_stability': np.mean([r.cluster_stability for r in self.ablation_results]),
            'recommendations': self._generate_recommendations(self.ablation_results)
        }


def run_feature_ablation_analysis(features: np.ndarray,
                                 feature_names: List[str],
                                 market_data: pd.DataFrame,
                                 output_dir: str = "ablation_analysis",
                                 n_components: int = 10) -> FeatureImportanceReport:
    """
    Run feature ablation analysis.
    
    Args:
        features: Feature matrix
        feature_names: List of feature names
        market_data: Market data
        output_dir: Output directory for results
        n_components: Number of PCA components
        
    Returns:
        FeatureImportanceReport
    """
    analyzer = FeatureAblationAnalyzer(output_dir=output_dir)
    return analyzer.run_ablation_analysis(features, feature_names, market_data, n_components)


if __name__ == "__main__":
    # Example usage
    print("Feature ablation analysis example")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create sample features with different groups
    features = np.random.randn(n_samples, n_features)
    
    # Create feature names with different groups
    feature_names = []
    feature_names.extend([f"return_{i}" for i in range(10)])  # Returns group
    feature_names.extend([f"volatility_{i}" for i in range(10)])  # Volatility group
    feature_names.extend([f"volume_{i}" for i in range(10)])  # Volume group
    feature_names.extend([f"risk_{i}" for i in range(10)])  # Risk group
    feature_names.extend([f"technical_{i}" for i in range(10)])  # Technical group
    
    # Create sample market data
    market_data = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
        'volume': np.random.lognormal(5, 0.5, n_samples)
    })
    
    # Run ablation analysis
    report = run_feature_ablation_analysis(features, feature_names, market_data)
    
    print(f"Baseline score: {report.baseline_score:.4f}")
    print(f"Feature groups analyzed: {report.n_feature_groups}")
    print(f"Most important group: {report.feature_importance_ranking[0][0]}")
    print(f"Recommendations: {len(report.recommendations)}")