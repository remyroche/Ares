"""
Stability Metrics for Feature Relevance

This module provides comprehensive stability metrics to assess the reliability
of feature importance across methods, time, and bootstrap samples.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import jaccard_score
from sklearn.inspection import permutation_importance
import warnings

logger = logging.getLogger(__name__)

class FeatureStabilityAnalyzer:
    """
    Analyzes feature stability across methods, time, and bootstrap samples.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize stability analyzer.
        
        Args:
            confidence_level: Confidence level for bootstrap CIs
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_bootstrap_stability(self, bootstrap_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate bootstrap stability metrics with confidence intervals.
        
        Args:
            bootstrap_results: Results from bootstrap analysis
            
        Returns:
            Bootstrap stability metrics
        """
        logger.info("Calculating bootstrap stability metrics...")
        
        stability_metrics = {}
        
        for method, method_results in bootstrap_results.get('method_results', {}).items():
            if 'feature_importance' not in method_results:
                continue
            
            importance_samples = method_results['feature_importance']
            feature_names = importance_samples.columns
            
            # Calculate stability metrics for each feature
            feature_stability = {}
            
            for feature in feature_names:
                importance_values = importance_samples[feature].dropna()
                
                if len(importance_values) == 0:
                    continue
                
                # Basic statistics
                mean_importance = importance_values.mean()
                std_importance = importance_values.std()
                cv_importance = std_importance / (mean_importance + 1e-8)
                
                # Confidence intervals
                ci_lower = np.percentile(importance_values, (self.alpha / 2) * 100)
                ci_upper = np.percentile(importance_values, (1 - self.alpha / 2) * 100)
                ci_width = ci_upper - ci_lower
                
                # Stability score (inverse of coefficient of variation)
                stability_score = 1 / (1 + cv_importance)
                
                feature_stability[feature] = {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'cv_importance': cv_importance,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_width,
                    'stability_score': stability_score,
                    'n_samples': len(importance_values)
                }
            
            stability_metrics[method] = {
                'feature_stability': feature_stability,
                'overall_stability': np.mean([fs['stability_score'] for fs in feature_stability.values()]),
                'mean_cv': np.mean([fs['cv_importance'] for fs in feature_stability.values()])
            }
        
        return stability_metrics
    
    def calculate_rank_consistency(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate rank consistency between different methods.
        
        Args:
            analysis_results: Results from different analysis methods
            
        Returns:
            Rank consistency metrics
        """
        logger.info("Calculating rank consistency metrics...")
        
        # Extract feature rankings from different methods
        method_rankings = {}
        
        for method, results in analysis_results.items():
            if 'feature_importance' in results:
                importance = results['feature_importance']
                if isinstance(importance, pd.Series):
                    ranking = importance.rank(ascending=False)
                    method_rankings[method] = ranking
                elif isinstance(importance, dict):
                    ranking = pd.Series(importance).rank(ascending=False)
                    method_rankings[method] = ranking
        
        if len(method_rankings) < 2:
            logger.warning("Need at least 2 methods for rank consistency analysis")
            return {}
        
        # Calculate pairwise rank correlations
        rank_correlations = {}
        method_names = list(method_rankings.keys())
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                ranking1 = method_rankings[method1]
                ranking2 = method_rankings[method2]
                
                # Align rankings
                common_features = ranking1.index.intersection(ranking2.index)
                if len(common_features) == 0:
                    continue
                
                ranking1_aligned = ranking1[common_features]
                ranking2_aligned = ranking2[common_features]
                
                # Calculate Spearman correlation
                spearman_corr, spearman_p = spearmanr(ranking1_aligned, ranking2_aligned)
                
                # Calculate Pearson correlation
                pearson_corr, pearson_p = pearsonr(ranking1_aligned, ranking2_aligned)
                
                rank_correlations[f"{method1}_vs_{method2}"] = {
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'n_features': len(common_features)
                }
        
        # Calculate overall rank consistency
        spearman_corrs = [rc['spearman_corr'] for rc in rank_correlations.values() 
                         if not np.isnan(rc['spearman_corr'])]
        
        overall_consistency = {
            'mean_spearman_corr': np.mean(spearman_corrs) if spearman_corrs else np.nan,
            'std_spearman_corr': np.std(spearman_corrs) if spearman_corrs else np.nan,
            'min_spearman_corr': np.min(spearman_corrs) if spearman_corrs else np.nan,
            'max_spearman_corr': np.max(spearman_corrs) if spearman_corrs else np.nan
        }
        
        return {
            'pairwise_correlations': rank_correlations,
            'overall_consistency': overall_consistency
        }
    
    def calculate_jaccard_overlap(self, analysis_results: Dict[str, Any], 
                                 k_values: List[int] = [5, 10, 20, 50]) -> Dict[str, Any]:
        """
        Calculate Jaccard overlap of top-k features across methods and versions.
        
        Args:
            analysis_results: Results from different analysis methods
            k_values: List of k values for top-k analysis
            
        Returns:
            Jaccard overlap metrics
        """
        logger.info("Calculating Jaccard overlap metrics...")
        
        # Extract top-k features for each method
        method_topk = {}
        
        for method, results in analysis_results.items():
            if 'feature_importance' in results:
                importance = results['feature_importance']
                if isinstance(importance, pd.Series):
                    method_topk[method] = {}
                    for k in k_values:
                        topk_features = importance.nlargest(k).index.tolist()
                        method_topk[method][k] = set(topk_features)
                elif isinstance(importance, dict):
                    importance_series = pd.Series(importance)
                    method_topk[method] = {}
                    for k in k_values:
                        topk_features = importance_series.nlargest(k).index.tolist()
                        method_topk[method][k] = set(topk_features)
        
        if len(method_topk) < 2:
            logger.warning("Need at least 2 methods for Jaccard overlap analysis")
            return {}
        
        # Calculate Jaccard overlap for each k
        jaccard_metrics = {}
        
        for k in k_values:
            k_overlaps = []
            method_names = list(method_topk.keys())
            
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names[i+1:], i+1):
                    if k in method_topk[method1] and k in method_topk[method2]:
                        set1 = method_topk[method1][k]
                        set2 = method_topk[method2][k]
                        
                        if len(set1) > 0 and len(set2) > 0:
                            jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                            k_overlaps.append(jaccard)
            
            if k_overlaps:
                jaccard_metrics[k] = {
                    'mean_jaccard': np.mean(k_overlaps),
                    'std_jaccard': np.std(k_overlaps),
                    'min_jaccard': np.min(k_overlaps),
                    'max_jaccard': np.max(k_overlaps),
                    'n_comparisons': len(k_overlaps)
                }
        
        return jaccard_metrics
    
    def calculate_temporal_drift(self, temporal_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate temporal drift in feature importance.
        
        Args:
            temporal_results: Results from temporal stability analysis
            
        Returns:
            Temporal drift metrics
        """
        logger.info("Calculating temporal drift metrics...")
        
        if 'window_results' not in temporal_results:
            logger.warning("No window results available for temporal drift analysis")
            return {}
        
        window_results = temporal_results['window_results']
        drift_metrics = {}
        
        # Calculate drift for each feature
        feature_drifts = {}
        
        for window, window_data in window_results.items():
            if 'feature_importance' not in window_data:
                continue
            
            importance = window_data['feature_importance']
            if isinstance(importance, pd.Series):
                for feature, imp_value in importance.items():
                    if feature not in feature_drifts:
                        feature_drifts[feature] = []
                    feature_drifts[feature].append(imp_value)
        
        # Calculate drift statistics for each feature
        for feature, importance_values in feature_drifts.items():
            if len(importance_values) < 2:
                continue
            
            importance_values = np.array(importance_values)
            
            # Calculate drift metrics
            mean_importance = np.mean(importance_values)
            std_importance = np.std(importance_values)
            cv_importance = std_importance / (mean_importance + 1e-8)
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(importance_values))
            slope = np.polyfit(x, importance_values, 1)[0]
            
            # Calculate volatility of importance
            importance_returns = np.diff(importance_values) / (importance_values[:-1] + 1e-8)
            importance_volatility = np.std(importance_returns)
            
            drift_metrics[feature] = {
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'cv_importance': cv_importance,
                'trend_slope': slope,
                'importance_volatility': importance_volatility,
                'n_windows': len(importance_values)
            }
        
        # Calculate overall drift metrics
        if drift_metrics:
            overall_drift = {
                'mean_cv': np.mean([fm['cv_importance'] for fm in drift_metrics.values()]),
                'mean_trend': np.mean([fm['trend_slope'] for fm in drift_metrics.values()]),
                'mean_volatility': np.mean([fm['importance_volatility'] for fm in drift_metrics.values()]),
                'stable_features': len([fm for fm in drift_metrics.values() if fm['cv_importance'] < 0.5]),
                'trending_features': len([fm for fm in drift_metrics.values() if abs(fm['trend_slope']) > 0.01])
            }
        else:
            overall_drift = {}
        
        return {
            'feature_drifts': drift_metrics,
            'overall_drift': overall_drift
        }
    
    def calculate_comprehensive_stability(self, analysis_results: Dict[str, Any],
                                        bootstrap_results: Optional[Dict[str, Any]] = None,
                                        temporal_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive stability metrics.
        
        Args:
            analysis_results: Results from different analysis methods
            bootstrap_results: Bootstrap analysis results
            temporal_results: Temporal stability analysis results
            
        Returns:
            Comprehensive stability metrics
        """
        logger.info("Calculating comprehensive stability metrics...")
        
        stability_report = {}
        
        # Bootstrap stability
        if bootstrap_results:
            stability_report['bootstrap_stability'] = self.calculate_bootstrap_stability(bootstrap_results)
        
        # Rank consistency
        stability_report['rank_consistency'] = self.calculate_rank_consistency(analysis_results)
        
        # Jaccard overlap
        stability_report['jaccard_overlap'] = self.calculate_jaccard_overlap(analysis_results)
        
        # Temporal drift
        if temporal_results:
            stability_report['temporal_drift'] = self.calculate_temporal_drift(temporal_results)
        
        # Overall stability score
        stability_scores = []
        
        # Add bootstrap stability score
        if 'bootstrap_stability' in stability_report:
            for method_stability in stability_report['bootstrap_stability'].values():
                if 'overall_stability' in method_stability:
                    stability_scores.append(method_stability['overall_stability'])
        
        # Add rank consistency score
        if 'rank_consistency' in stability_report and 'overall_consistency' in stability_report['rank_consistency']:
            mean_corr = stability_report['rank_consistency']['overall_consistency'].get('mean_spearman_corr', 0)
            if not np.isnan(mean_corr):
                stability_scores.append(mean_corr)
        
        # Add Jaccard overlap score
        if 'jaccard_overlap' in stability_report:
            jaccard_scores = [metrics['mean_jaccard'] for metrics in stability_report['jaccard_overlap'].values()]
            if jaccard_scores:
                stability_scores.append(np.mean(jaccard_scores))
        
        # Calculate overall stability score
        if stability_scores:
            stability_report['overall_stability_score'] = np.mean(stability_scores)
        else:
            stability_report['overall_stability_score'] = 0.0
        
        return stability_report