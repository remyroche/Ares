"""
Enhanced Outcome Generator for HMM Clustering

This module generates clean, metrics-focused outcome files without raw data,
focusing on comprehensive statistical and economical metrics for cluster analysis.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from src.utils.logger import system_logger


class EnhancedOutcomeGenerator:
    """
    Generate enhanced outcome files with comprehensive metrics and no raw data.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('EnhancedOutcomeGenerator')
    
    def generate_clustering_outcome(
        self,
        cluster_results: Dict[str, Any],
        regime_characteristics: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        target_clusters: int = 20,
        coverage_target: float = 0.90
    ) -> Dict[str, Any]:
        """
        Generate enhanced clustering outcome with comprehensive metrics.
        
        Args:
            cluster_results: Raw clustering results
            regime_characteristics: Regime characteristics data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            target_clusters: Target number of clusters
            coverage_target: Target coverage percentage
            
        Returns:
            Enhanced outcome dictionary with comprehensive metrics
        """
        try:
            self.logger.info("🎯 Generating enhanced clustering outcome with comprehensive metrics...")
            
            # Extract cluster information
            cluster_analysis = self._analyze_clusters(cluster_results, regime_characteristics)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                cluster_analysis, target_clusters, coverage_target
            )
            
            # Generate economical analysis
            economical_metrics = self._calculate_economical_metrics(cluster_analysis)
            
            # Generate statistical analysis
            statistical_metrics = self._calculate_statistical_metrics(cluster_analysis)
            
            # Create enhanced outcome structure
            enhanced_outcome = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "target_clusters": target_clusters,
                    "coverage_target": coverage_target,
                    "outcome_version": "2.0_enhanced"
                },
                "clustering_summary": {
                    "total_clusters": cluster_analysis["total_clusters"],
                    "target_achieved": cluster_analysis["total_clusters"] <= target_clusters * 1.5,  # Allow 50% tolerance
                    "coverage_achieved": metrics["coverage_metrics"]["top_20_coverage"] >= coverage_target * 100,
                    "quality_score": metrics["quality_metrics"]["overall_quality_score"]
                },
                "comprehensive_metrics": metrics,
                "economical_metrics": economical_metrics,
                "statistical_metrics": statistical_metrics,
                "cluster_analysis": self._generate_cluster_analysis(cluster_analysis),
                "recommendations": self._generate_recommendations(cluster_analysis, metrics)
            }
            
            self.logger.info("✅ Enhanced clustering outcome generated successfully")
            return enhanced_outcome
            
        except Exception as e:
            self.logger.error(f"❌ Error generating enhanced outcome: {e}")
            raise
    
    def _analyze_clusters(
        self, 
        cluster_results: Dict[str, Any], 
        regime_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze clusters and extract key information."""
        try:
            clusters_dict = {}
            total_samples = 0
            
            # Process each cluster
            for cluster_id, cluster_data in cluster_results.get("clusters_dict", {}).items():
                sample_count = cluster_data.get("sample_count", 0)
                total_samples += sample_count
                
                # Calculate cluster characteristics
                cluster_characteristics = self._calculate_cluster_characteristics(
                    cluster_data, regime_characteristics
                )
                
                clusters_dict[cluster_id] = {
                    "cluster_id": cluster_id,
                    "sample_count": sample_count,
                    "sample_percentage": 0.0,  # Will be calculated after total_samples
                    "characteristics": cluster_characteristics
                }
            
            # Calculate percentages
            for cluster_id in clusters_dict:
                clusters_dict[cluster_id]["sample_percentage"] = (
                    clusters_dict[cluster_id]["sample_count"] / total_samples * 100
                    if total_samples > 0 else 0
                )
            
            return {
                "clusters_dict": clusters_dict,
                "total_clusters": len(clusters_dict),
                "total_samples": total_samples
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing clusters: {e}")
            return {"clusters_dict": {}, "total_clusters": 0, "total_samples": 0}
    
    def _calculate_cluster_characteristics(
        self, 
        cluster_data: Dict[str, Any], 
        regime_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive characteristics for a single cluster."""
        try:
            # Extract feature means and statistics
            feature_means = cluster_data.get("feature_means", {})
            feature_stds = cluster_data.get("feature_stds", {})
            
            # Calculate coefficient of variation for each dimension
            cv_metrics = {}
            for dimension in ["momentum", "volatility", "volume"]:
                if f"{dimension}_strength" in feature_means and f"{dimension}_strength" in feature_stds:
                    mean_val = feature_means[f"{dimension}_strength"]
                    std_val = feature_stds[f"{dimension}_strength"]
                    cv_metrics[f"{dimension}_cv"] = std_val / abs(mean_val) if mean_val != 0 else float('inf')
            
            # Calculate coherence (low CV = high coherence)
            coherence_score = 1.0 - np.mean(list(cv_metrics.values())) if cv_metrics else 0.0
            coherence_score = max(0.0, min(1.0, coherence_score))  # Clamp to [0, 1]
            
            return {
                "feature_means": feature_means,
                "feature_stds": feature_stds,
                "cv_metrics": cv_metrics,
                "coherence_score": coherence_score,
                "is_coherent": coherence_score >= 0.7,  # Threshold for coherence
                "dimension_analysis": self._analyze_dimensions(feature_means)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating cluster characteristics: {e}")
            return {}
    
    def _analyze_dimensions(self, feature_means: Dict[str, float]) -> Dict[str, Any]:
        """Analyze dimensional characteristics of a cluster."""
        try:
            dimension_analysis = {}
            
            # Momentum analysis
            momentum_features = [k for k in feature_means.keys() if 'momentum' in k.lower()]
            if momentum_features:
                momentum_values = [feature_means[k] for k in momentum_features]
                dimension_analysis["momentum"] = {
                    "strength": np.mean(momentum_values),
                    "direction": "bullish" if np.mean(momentum_values) > 0 else "bearish",
                    "consistency": 1.0 - (np.std(momentum_values) / (np.mean(np.abs(momentum_values)) + 1e-8))
                }
            
            # Volatility analysis
            volatility_features = [k for k in feature_means.keys() if 'volatility' in k.lower()]
            if volatility_features:
                volatility_values = [feature_means[k] for k in volatility_features]
                dimension_analysis["volatility"] = {
                    "level": np.mean(volatility_values),
                    "regime": "high" if np.mean(volatility_values) > 0.5 else "low",
                    "stability": 1.0 - (np.std(volatility_values) / (np.mean(volatility_values) + 1e-8))
                }
            
            # Volume analysis
            volume_features = [k for k in feature_means.keys() if 'volume' in k.lower()]
            if volume_features:
                volume_values = [feature_means[k] for k in volume_features]
                dimension_analysis["volume"] = {
                    "level": np.mean(volume_values),
                    "regime": "high" if np.mean(volume_values) > 1.0 else "low",
                    "trend": "increasing" if np.mean(volume_values) > 1.1 else "decreasing" if np.mean(volume_values) < 0.9 else "stable"
                }
            
            return dimension_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing dimensions: {e}")
            return {}
    
    def _calculate_comprehensive_metrics(
        self, 
        cluster_analysis: Dict[str, Any], 
        target_clusters: int, 
        coverage_target: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive clustering quality metrics."""
        try:
            clusters_dict = cluster_analysis["clusters_dict"]
            total_samples = cluster_analysis["total_samples"]
            
            # Sort clusters by sample count
            sorted_clusters = sorted(
                clusters_dict.items(), 
                key=lambda x: x[1]["sample_count"], 
                reverse=True
            )
            
            # Calculate coverage metrics
            coverage_metrics = self._calculate_coverage_metrics(sorted_clusters, total_samples)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(clusters_dict)
            
            # Calculate coherence metrics
            coherence_metrics = self._calculate_coherence_metrics(clusters_dict)
            
            return {
                "coverage_metrics": coverage_metrics,
                "quality_metrics": quality_metrics,
                "coherence_metrics": coherence_metrics,
                "target_compliance": {
                    "cluster_count_target": target_clusters,
                    "cluster_count_actual": len(clusters_dict),
                    "cluster_count_deviation": abs(len(clusters_dict) - target_clusters) / target_clusters * 100,
                    "coverage_target": coverage_target * 100,
                    "coverage_actual": coverage_metrics["top_20_coverage"],
                    "coverage_achieved": coverage_metrics["top_20_coverage"] >= coverage_target * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating comprehensive metrics: {e}")
            return {}
    
    def _calculate_coverage_metrics(
        self, 
        sorted_clusters: List[Tuple], 
        total_samples: int
    ) -> Dict[str, Any]:
        """Calculate coverage metrics for different cluster sets."""
        try:
            if not sorted_clusters or total_samples == 0:
                return {"top_5_coverage": 0.0, "top_10_coverage": 0.0, "top_20_coverage": 0.0}
            
            # Calculate coverage for different top-N sets
            top_5_samples = sum([cluster[1]["sample_count"] for cluster in sorted_clusters[:5]])
            top_10_samples = sum([cluster[1]["sample_count"] for cluster in sorted_clusters[:10]])
            top_20_samples = sum([cluster[1]["sample_count"] for cluster in sorted_clusters[:20]])
            
            return {
                "top_5_coverage": (top_5_samples / total_samples) * 100,
                "top_10_coverage": (top_10_samples / total_samples) * 100,
                "top_20_coverage": (top_20_samples / total_samples) * 100,
                "top_5_clusters": [{"cluster_id": cluster[0], "sample_count": cluster[1]["sample_count"]} 
                                 for cluster in sorted_clusters[:5]],
                "top_10_clusters": [{"cluster_id": cluster[0], "sample_count": cluster[1]["sample_count"]} 
                                  for cluster in sorted_clusters[:10]],
                "top_20_clusters": [{"cluster_id": cluster[0], "sample_count": cluster[1]["sample_count"]} 
                                  for cluster in sorted_clusters[:20]]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating coverage metrics: {e}")
            return {}
    
    def _calculate_quality_metrics(self, clusters_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics for clustering."""
        try:
            if not clusters_dict:
                return {"overall_quality_score": 0.0}
            
            # Calculate various quality indicators
            coherence_scores = [cluster["characteristics"]["coherence_score"] 
                              for cluster in clusters_dict.values()]
            
            # Calculate distribution metrics
            sample_counts = [cluster["sample_count"] for cluster in clusters_dict.values()]
            
            # Calculate quality score (0-1 scale)
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            size_balance = 1.0 - (np.std(sample_counts) / (np.mean(sample_counts) + 1e-8))
            size_balance = max(0.0, min(1.0, size_balance))
            
            overall_quality_score = (avg_coherence * 0.6 + size_balance * 0.4)
            
            return {
                "overall_quality_score": overall_quality_score,
                "average_coherence": avg_coherence,
                "size_balance": size_balance,
                "coherent_clusters": sum(1 for cluster in clusters_dict.values() 
                                       if cluster["characteristics"]["is_coherent"]),
                "total_clusters": len(clusters_dict),
                "coherence_ratio": sum(1 for cluster in clusters_dict.values() 
                                     if cluster["characteristics"]["is_coherent"]) / len(clusters_dict)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating quality metrics: {e}")
            return {"overall_quality_score": 0.0}
    
    def _calculate_coherence_metrics(self, clusters_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate coherence metrics across all clusters."""
        try:
            if not clusters_dict:
                return {}
            
            # Collect CV metrics from all clusters
            all_cv_metrics = []
            dimension_cvs = {"momentum": [], "volatility": [], "volume": []}
            
            for cluster in clusters_dict.values():
                cv_metrics = cluster["characteristics"]["cv_metrics"]
                all_cv_metrics.extend(cv_metrics.values())
                
                for dimension in dimension_cvs.keys():
                    cv_key = f"{dimension}_cv"
                    if cv_key in cv_metrics:
                        dimension_cvs[dimension].append(cv_metrics[cv_key])
            
            # Calculate coherence statistics
            coherence_stats = {}
            for dimension, cvs in dimension_cvs.items():
                if cvs:
                    coherence_stats[dimension] = {
                        "mean_cv": np.mean(cvs),
                        "median_cv": np.median(cvs),
                        "std_cv": np.std(cvs),
                        "low_cv_clusters": sum(1 for cv in cvs if cv < 0.3),
                        "high_cv_clusters": sum(1 for cv in cvs if cv > 0.7)
                    }
            
            return {
                "overall_coherence": {
                    "mean_cv": np.mean(all_cv_metrics) if all_cv_metrics else 0.0,
                    "median_cv": np.median(all_cv_metrics) if all_cv_metrics else 0.0,
                    "low_cv_clusters": sum(1 for cv in all_cv_metrics if cv < 0.3),
                    "high_cv_clusters": sum(1 for cv in all_cv_metrics if cv > 0.7)
                },
                "dimension_coherence": coherence_stats
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating coherence metrics: {e}")
            return {}
    
    def _calculate_economical_metrics(self, cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate economical relevance metrics for clusters."""
        try:
            clusters_dict = cluster_analysis["clusters_dict"]
            
            # Analyze market state representation
            market_states = []
            for cluster in clusters_dict.values():
                dimension_analysis = cluster["characteristics"]["dimension_analysis"]
                
                # Create market state description
                momentum_regime = dimension_analysis.get("momentum", {}).get("direction", "neutral")
                volatility_regime = dimension_analysis.get("volatility", {}).get("regime", "medium")
                volume_regime = dimension_analysis.get("volume", {}).get("regime", "medium")
                
                market_state = f"{momentum_regime}_{volatility_regime}_vol_{volume_regime}"
                market_states.append({
                    "cluster_id": cluster["cluster_id"],
                    "market_state": market_state,
                    "sample_count": cluster["sample_count"],
                    "sample_percentage": cluster["sample_percentage"]
                })
            
            # Calculate diversity metrics
            unique_states = len(set(state["market_state"] for state in market_states))
            state_distribution = {}
            for state in market_states:
                market_state = state["market_state"]
                if market_state not in state_distribution:
                    state_distribution[market_state] = 0
                state_distribution[market_state] += state["sample_count"]
            
            return {
                "market_state_diversity": {
                    "unique_states": unique_states,
                    "total_clusters": len(clusters_dict),
                    "diversity_ratio": unique_states / len(clusters_dict) if clusters_dict else 0,
                    "state_distribution": state_distribution
                },
                "market_state_analysis": market_states,
                "economical_relevance": {
                    "bullish_clusters": sum(1 for state in market_states if "bullish" in state["market_state"]),
                    "bearish_clusters": sum(1 for state in market_states if "bearish" in state["market_state"]),
                    "high_vol_clusters": sum(1 for state in market_states if "high" in state["market_state"]),
                    "low_vol_clusters": sum(1 for state in market_states if "low" in state["market_state"])
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating economical metrics: {e}")
            return {}
    
    def _calculate_statistical_metrics(self, cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive statistical metrics."""
        try:
            clusters_dict = cluster_analysis["clusters_dict"]
            
            if not clusters_dict:
                return {}
            
            # Sample size statistics
            sample_counts = [cluster["sample_count"] for cluster in clusters_dict.values()]
            
            # Statistical tests for cluster validity
            statistical_metrics = {
                "sample_distribution": {
                    "mean": np.mean(sample_counts),
                    "median": np.median(sample_counts),
                    "std": np.std(sample_counts),
                    "min": np.min(sample_counts),
                    "max": np.max(sample_counts),
                    "q25": np.percentile(sample_counts, 25),
                    "q75": np.percentile(sample_counts, 75)
                },
                "cluster_validity": {
                    "sufficient_samples": sum(1 for count in sample_counts if count >= 10),
                    "insufficient_samples": sum(1 for count in sample_counts if count < 10),
                    "balanced_distribution": np.std(sample_counts) / (np.mean(sample_counts) + 1e-8) < 1.0
                },
                "dimensional_analysis": self._calculate_dimensional_statistics(clusters_dict)
            }
            
            return statistical_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating statistical metrics: {e}")
            return {}
    
    def _calculate_dimensional_statistics(self, clusters_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for each dimension across clusters."""
        try:
            dimensional_stats = {}
            
            for dimension in ["momentum", "volatility", "volume"]:
                dimension_values = []
                for cluster in clusters_dict.values():
                    dim_analysis = cluster["characteristics"]["dimension_analysis"]
                    if dimension in dim_analysis:
                        # Extract the main value for this dimension
                        if dimension == "momentum":
                            dimension_values.append(dim_analysis[dimension].get("strength", 0))
                        elif dimension == "volatility":
                            dimension_values.append(dim_analysis[dimension].get("level", 0))
                        elif dimension == "volume":
                            dimension_values.append(dim_analysis[dimension].get("level", 0))
                
                if dimension_values:
                    dimensional_stats[dimension] = {
                        "mean": np.mean(dimension_values),
                        "std": np.std(dimension_values),
                        "range": np.max(dimension_values) - np.min(dimension_values),
                        "coverage": len(dimension_values) / len(clusters_dict)
                    }
            
            return dimensional_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating dimensional statistics: {e}")
            return {}
    
    def _generate_cluster_analysis(self, cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed cluster analysis without raw data."""
        try:
            clusters_dict = cluster_analysis["clusters_dict"]
            
            # Sort clusters by sample count for analysis
            sorted_clusters = sorted(
                clusters_dict.items(),
                key=lambda x: x[1]["sample_count"],
                reverse=True
            )
            
            # Generate analysis for top clusters
            top_clusters_analysis = []
            for cluster_id, cluster_data in sorted_clusters[:20]:  # Top 20 clusters
                analysis = {
                    "cluster_id": cluster_id,
                    "sample_count": cluster_data["sample_count"],
                    "sample_percentage": cluster_data["sample_percentage"],
                    "coherence_score": cluster_data["characteristics"]["coherence_score"],
                    "is_coherent": cluster_data["characteristics"]["is_coherent"],
                    "market_state": self._determine_market_state(cluster_data["characteristics"]["dimension_analysis"]),
                    "dimensional_profile": cluster_data["characteristics"]["dimension_analysis"]
                }
                top_clusters_analysis.append(analysis)
            
            return {
                "top_20_clusters": top_clusters_analysis,
                "cluster_summary": {
                    "total_clusters": len(clusters_dict),
                    "coherent_clusters": sum(1 for cluster in clusters_dict.values() 
                                           if cluster["characteristics"]["is_coherent"]),
                    "average_coherence": np.mean([cluster["characteristics"]["coherence_score"] 
                                                for cluster in clusters_dict.values()])
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating cluster analysis: {e}")
            return {}
    
    def _determine_market_state(self, dimension_analysis: Dict[str, Any]) -> str:
        """Determine the market state based on dimensional analysis."""
        try:
            momentum = dimension_analysis.get("momentum", {}).get("direction", "neutral")
            volatility = dimension_analysis.get("volatility", {}).get("regime", "medium")
            volume = dimension_analysis.get("volume", {}).get("regime", "medium")
            
            # Create a descriptive market state
            if momentum == "bullish" and volatility == "low":
                return "stable_bull_market"
            elif momentum == "bearish" and volatility == "low":
                return "stable_bear_market"
            elif momentum == "bullish" and volatility == "high":
                return "volatile_bull_market"
            elif momentum == "bearish" and volatility == "high":
                return "volatile_bear_market"
            elif volatility == "high":
                return "high_volatility_market"
            elif volatility == "low":
                return "low_volatility_market"
            else:
                return "neutral_market"
                
        except Exception as e:
            self.logger.error(f"❌ Error determining market state: {e}")
            return "unknown_market"
    
    def _generate_recommendations(
        self, 
        cluster_analysis: Dict[str, Any], 
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        try:
            recommendations = []
            
            # Cluster count recommendations
            total_clusters = cluster_analysis["total_clusters"]
            if total_clusters > 30:
                recommendations.append(
                    f"Consider reducing cluster count from {total_clusters} to ~20 clusters for better ML model training"
                )
            elif total_clusters < 10:
                recommendations.append(
                    f"Consider increasing cluster count from {total_clusters} to capture more market state diversity"
                )
            
            # Coverage recommendations
            coverage_metrics = metrics.get("coverage_metrics", {})
            top_20_coverage = coverage_metrics.get("top_20_coverage", 0)
            if top_20_coverage < 80:
                recommendations.append(
                    f"Top 20 clusters only cover {top_20_coverage:.1f}% of data. Consider merging small clusters or improving feature engineering"
                )
            
            # Coherence recommendations
            coherence_metrics = metrics.get("coherence_metrics", {})
            overall_coherence = coherence_metrics.get("overall_coherence", {})
            mean_cv = overall_coherence.get("mean_cv", 0)
            if mean_cv > 0.5:
                recommendations.append(
                    f"Average coefficient of variation is {mean_cv:.3f}, indicating low cluster coherence. Consider tighter similarity thresholds"
                )
            
            # Quality recommendations
            quality_metrics = metrics.get("quality_metrics", {})
            coherence_ratio = quality_metrics.get("coherence_ratio", 0)
            if coherence_ratio < 0.7:
                recommendations.append(
                    f"Only {coherence_ratio:.1%} of clusters are coherent. Focus on improving cluster homogeneity"
                )
            
            # Economical recommendations
            economical_metrics = metrics.get("economical_metrics", {})
            diversity_ratio = economical_metrics.get("market_state_diversity", {}).get("diversity_ratio", 0)
            if diversity_ratio < 0.5:
                recommendations.append(
                    "Low market state diversity detected. Consider expanding feature space to capture more market dimensions"
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def save_enhanced_outcome(
        self, 
        enhanced_outcome: Dict[str, Any], 
        output_path: str
    ) -> None:
        """Save enhanced outcome to file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(enhanced_outcome, f, indent=2, default=str)
            
            self.logger.info(f"✅ Enhanced outcome saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving enhanced outcome: {e}")
            raise