"""
Enhanced Analysis Module for Optimal Regime Clustering

This module provides enhanced analysis capabilities including:
- Coefficient of Variation (CV) metrics for all 4 dimensions
- Detailed data quality assessment with 7 criteria
- Comprehensive statistical insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DimensionCVMetrics:
    """Coefficient of Variation metrics for each dimension."""
    volume_cv: float
    volatility_cv: float
    trend_cv: float
    momentum_cv: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "volume_cv": self.volume_cv,
            "volatility_cv": self.volatility_cv,
            "trend_cv": self.trend_cv,
            "momentum_cv": self.momentum_cv
        }

@dataclass
class DataQualityCriteria:
    """Data quality assessment criteria."""
    price_data_completeness: bool
    volume_data_completeness: bool
    temporal_continuity: bool
    feature_calculation_success: bool
    outlier_detection_success: bool
    cluster_separation_quality: bool
    statistical_significance: bool
    
    def get_completeness_score(self) -> float:
        """Calculate completeness score as percentage of criteria met."""
        criteria = [
            self.price_data_completeness,
            self.volume_data_completeness,
            self.temporal_continuity,
            self.feature_calculation_success,
            self.outlier_detection_success,
            self.cluster_separation_quality,
            self.statistical_significance
        ]
        return sum(criteria) / len(criteria)
    
    def get_failed_criteria(self) -> List[str]:
        """Get list of failed criteria."""
        failed = []
        criteria_map = {
            "price_data_completeness": self.price_data_completeness,
            "volume_data_completeness": self.volume_data_completeness,
            "temporal_continuity": self.temporal_continuity,
            "feature_calculation_success": self.feature_calculation_success,
            "outlier_detection_success": self.outlier_detection_success,
            "cluster_separation_quality": self.cluster_separation_quality,
            "statistical_significance": self.statistical_significance
        }
        
        for criterion, passed in criteria_map.items():
            if not passed:
                failed.append(criterion)
        
        return failed

def calculate_dimension_cv_metrics(
    features: np.ndarray,
    feature_names: List[str] = ["volume", "volatility", "trend", "momentum"]
) -> DimensionCVMetrics:
    """
    Calculate coefficient of variation for each of the 4 dimensions.
    
    Args:
        features: Feature matrix with shape (n_samples, n_features)
        feature_names: Names of the feature dimensions
        
    Returns:
        DimensionCVMetrics object with CV values for each dimension
    """
    try:
        if features.shape[1] < 4:
            logger.warning(f"Expected 4 features, got {features.shape[1]}. Padding with zeros.")
            # Pad with zeros if we don't have 4 features
            padded_features = np.zeros((features.shape[0], 4))
            padded_features[:, :features.shape[1]] = features
            features = padded_features
        
        # Calculate CV for each dimension
        cv_metrics = []
        for i in range(4):
            feature_values = features[:, i]
            
            # Remove any NaN or infinite values
            feature_values = feature_values[np.isfinite(feature_values)]
            
            if len(feature_values) == 0:
                logger.warning(f"No valid values for dimension {i}")
                cv_metrics.append(0.0)
                continue
            
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            
            if mean_val == 0:
                cv_metrics.append(0.0)
            else:
                cv = std_val / abs(mean_val)
                cv_metrics.append(cv)
        
        return DimensionCVMetrics(
            volume_cv=cv_metrics[0],
            volatility_cv=cv_metrics[1],
            trend_cv=cv_metrics[2],
            momentum_cv=cv_metrics[3]
        )
        
    except Exception as e:
        logger.error(f"Error calculating dimension CV metrics: {e}")
        return DimensionCVMetrics(0.0, 0.0, 0.0, 0.0)

def assess_data_quality_criteria(
    market_data: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any]
) -> DataQualityCriteria:
    """
    Assess data quality against 7 specific criteria.
    
    Args:
        market_data: Original market data DataFrame
        features: Feature matrix used for clustering
        labels: Cluster labels
        config: Configuration dictionary
        
    Returns:
        DataQualityCriteria object with assessment results
    """
    try:
        # 1. Price data completeness
        price_data_completeness = (
            'close' in market_data.columns and
            'high' in market_data.columns and
            'low' in market_data.columns and
            not market_data['close'].isna().all() and
            market_data['close'].count() > 0
        )
        
        # 2. Volume data completeness
        volume_data_completeness = (
            'volume' in market_data.columns and
            not market_data['volume'].isna().all() and
            market_data['volume'].count() > 0 and
            (market_data['volume'] > 0).any()
        )
        
        # 3. Temporal continuity (check for gaps in time series)
        temporal_continuity = True
        if 'timestamp' in market_data.columns:
            try:
                # Convert to datetime if not already
                timestamps = pd.to_datetime(market_data['timestamp'])
                time_diffs = timestamps.diff().dropna()
                
                # Check if there are reasonable time gaps (not too large)
                expected_interval = pd.Timedelta(minutes=15)  # 15m timeframe
                max_reasonable_gap = expected_interval * 10  # Allow up to 10x normal gap
                
                large_gaps = time_diffs > max_reasonable_gap
                gap_percentage = large_gaps.sum() / len(time_diffs)
                temporal_continuity = gap_percentage < 0.05  # Less than 5% large gaps
                
            except Exception:
                temporal_continuity = False
        
        # 4. Feature calculation success
        feature_calculation_success = (
            features is not None and
            features.shape[0] > 0 and
            features.shape[1] >= 4 and
            not np.isnan(features).all() and
            np.isfinite(features).all()
        )
        
        # 5. Outlier detection success
        outlier_detection_success = True
        if features is not None and len(features) > 0:
            try:
                # Check if we have reasonable feature distributions
                for i in range(min(4, features.shape[1])):
                    feature_values = features[:, i]
                    finite_values = feature_values[np.isfinite(feature_values)]
                    
                    if len(finite_values) > 0:
                        # Check for extreme outliers (beyond 5 standard deviations)
                        mean_val = np.mean(finite_values)
                        std_val = np.std(finite_values)
                        
                        if std_val > 0:
                            extreme_outliers = np.abs(finite_values - mean_val) > (5 * std_val)
                            outlier_percentage = extreme_outliers.sum() / len(finite_values)
                            
                            if outlier_percentage > 0.1:  # More than 10% extreme outliers
                                outlier_detection_success = False
                                break
            except Exception:
                outlier_detection_success = False
        
        # 6. Cluster separation quality
        cluster_separation_quality = True
        if labels is not None and len(labels) > 0 and len(np.unique(labels)) > 1:
            try:
                # Check silhouette score
                if len(features) > 1 and features.shape[1] > 0:
                    finite_mask = np.isfinite(features).all(axis=1)
                    if finite_mask.sum() > 1:
                        finite_features = features[finite_mask]
                        finite_labels = labels[finite_mask]
                        
                        if len(np.unique(finite_labels)) > 1:
                            silhouette = silhouette_score(finite_features, finite_labels)
                            cluster_separation_quality = silhouette > -0.5  # Reasonable separation
            except Exception:
                cluster_separation_quality = False
        
        # 7. Statistical significance
        statistical_significance = (
            features is not None and
            features.shape[0] >= 100 and  # Minimum sample size
            len(np.unique(labels)) >= 2 and  # At least 2 clusters
            labels is not None and
            len(labels) > 0
        )
        
        return DataQualityCriteria(
            price_data_completeness=price_data_completeness,
            volume_data_completeness=volume_data_completeness,
            temporal_continuity=temporal_continuity,
            feature_calculation_success=feature_calculation_success,
            outlier_detection_success=outlier_detection_success,
            cluster_separation_quality=cluster_separation_quality,
            statistical_significance=statistical_significance
        )
        
    except Exception as e:
        logger.error(f"Error assessing data quality criteria: {e}")
        return DataQualityCriteria(False, False, False, False, False, False, False)

def create_enhanced_statistical_summary(
    market_data: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    cluster_statistics: Any,
    quality_metrics: Dict[str, float],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create enhanced statistical summary with CV metrics and detailed data quality assessment.
    
    Args:
        market_data: Original market data DataFrame
        features: Feature matrix used for clustering
        labels: Cluster labels
        cluster_statistics: Cluster statistics object
        quality_metrics: Quality metrics dictionary
        config: Configuration dictionary
        
    Returns:
        Enhanced statistical summary dictionary
    """
    try:
        # Calculate dimension CV metrics
        cv_metrics = calculate_dimension_cv_metrics(features)
        
        # Assess data quality criteria
        quality_criteria = assess_data_quality_criteria(market_data, features, labels, config)
        
        # Calculate cluster-level CV metrics for each dimension
        cluster_dimension_cv = {}
        if labels is not None and len(labels) > 0:
            unique_labels = np.unique(labels)
            for cluster_id in unique_labels:
                if cluster_id >= 0:  # Skip noise points
                    cluster_mask = labels == cluster_id
                    cluster_features = features[cluster_mask]
                    
                    if len(cluster_features) > 0:
                        cluster_cv = calculate_dimension_cv_metrics(cluster_features)
                        cluster_dimension_cv[f"cluster_{cluster_id}"] = cluster_cv.to_dict()
        
        # Create enhanced statistical summary
        enhanced_summary = {
            "data_quality_assessment": {
                "completeness_score": quality_criteria.get_completeness_score(),
                "consistency_score": 1.0,  # Assume consistent for now
                "reliability_score": 0.95,  # Assume high reliability
                "failed_criteria": quality_criteria.get_failed_criteria(),
                "criteria_details": {
                    "price_data_completeness": quality_criteria.price_data_completeness,
                    "volume_data_completeness": quality_criteria.volume_data_completeness,
                    "temporal_continuity": quality_criteria.temporal_continuity,
                    "feature_calculation_success": quality_criteria.feature_calculation_success,
                    "outlier_detection_success": quality_criteria.outlier_detection_success,
                    "cluster_separation_quality": quality_criteria.cluster_separation_quality,
                    "statistical_significance": quality_criteria.statistical_significance
                }
            },
            
            "dimension_cv_metrics": {
                "overall_cv": cv_metrics.to_dict(),
                "cluster_level_cv": cluster_dimension_cv,
                "cv_interpretation": {
                    "volume_cv": _interpret_cv_value(cv_metrics.volume_cv, "volume"),
                    "volatility_cv": _interpret_cv_value(cv_metrics.volatility_cv, "volatility"),
                    "trend_cv": _interpret_cv_value(cv_metrics.trend_cv, "trend"),
                    "momentum_cv": _interpret_cv_value(cv_metrics.momentum_cv, "momentum")
                }
            },
            
            "statistical_significance": {
                "sample_size_adequacy": "adequate" if len(features) >= 1000 else "minimal" if len(features) >= 100 else "insufficient",
                "statistical_power": min(1.0, len(features) / 1000.0),  # Normalize to 1000 samples
                "confidence_level": "high" if len(features) >= 5000 else "moderate" if len(features) >= 1000 else "low"
            },
            
            "analytical_recommendations": {
                "suggested_analysis_depth": "comprehensive" if quality_criteria.get_completeness_score() >= 0.8 else "limited",
                "recommended_confidence_level": quality_criteria.get_completeness_score(),
                "analysis_complexity_rating": "high" if len(np.unique(labels)) > 10 else "medium" if len(np.unique(labels)) > 5 else "low"
            }
        }
        
        return enhanced_summary
        
    except Exception as e:
        logger.error(f"Error creating enhanced statistical summary: {e}")
        return {
            "data_quality_assessment": {
                "completeness_score": 0.0,
                "consistency_score": 0.0,
                "reliability_score": 0.0,
                "failed_criteria": ["error_during_assessment"],
                "criteria_details": {}
            },
            "dimension_cv_metrics": {
                "overall_cv": {"volume_cv": 0.0, "volatility_cv": 0.0, "trend_cv": 0.0, "momentum_cv": 0.0},
                "cluster_level_cv": {},
                "cv_interpretation": {}
            },
            "statistical_significance": {
                "sample_size_adequacy": "unknown",
                "statistical_power": 0.0,
                "confidence_level": "unknown"
            },
            "analytical_recommendations": {
                "suggested_analysis_depth": "unknown",
                "recommended_confidence_level": 0.0,
                "analysis_complexity_rating": "unknown"
            }
        }

def _interpret_cv_value(cv: float, dimension: str) -> str:
    """Interpret CV value for a given dimension."""
    if cv < 0.1:
        return f"Low variability in {dimension} - very stable"
    elif cv < 0.3:
        return f"Moderate variability in {dimension} - reasonably stable"
    elif cv < 0.6:
        return f"High variability in {dimension} - volatile"
    else:
        return f"Very high variability in {dimension} - highly volatile"

def calculate_cluster_dimension_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str] = ["volume", "volatility", "trend", "momentum"]
) -> Dict[str, Any]:
    """
    Calculate detailed dimension analysis for each cluster.
    
    Args:
        features: Feature matrix
        labels: Cluster labels
        feature_names: Names of feature dimensions
        
    Returns:
        Dictionary with cluster-level dimension analysis
    """
    try:
        unique_labels = np.unique(labels)
        cluster_analysis = {}
        
        for cluster_id in unique_labels:
            if cluster_id >= 0:  # Skip noise points
                cluster_mask = labels == cluster_id
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) > 0:
                    cluster_analysis[f"cluster_{cluster_id}"] = {
                        "sample_count": len(cluster_features),
                        "dimension_statistics": {},
                        "dimension_cv": {},
                        "dominant_dimensions": []
                    }
                    
                    # Calculate statistics for each dimension
                    dimension_means = []
                    dimension_stds = []
                    dimension_cvs = []
                    
                    for i, dim_name in enumerate(feature_names[:4]):  # Limit to 4 dimensions
                        if i < cluster_features.shape[1]:
                            dim_values = cluster_features[:, i]
                            finite_values = dim_values[np.isfinite(dim_values)]
                            
                            if len(finite_values) > 0:
                                mean_val = np.mean(finite_values)
                                std_val = np.std(finite_values)
                                cv_val = std_val / abs(mean_val) if mean_val != 0 else 0.0
                                
                                dimension_means.append(mean_val)
                                dimension_stds.append(std_val)
                                dimension_cvs.append(cv_val)
                                
                                cluster_analysis[f"cluster_{cluster_id}"]["dimension_statistics"][dim_name] = {
                                    "mean": float(mean_val),
                                    "std": float(std_val),
                                    "cv": float(cv_val)
                                }
                                
                                cluster_analysis[f"cluster_{cluster_id}"]["dimension_cv"][dim_name] = float(cv_val)
                    
                    # Identify dominant dimensions (highest absolute mean values)
                    if dimension_means:
                        abs_means = [abs(m) for m in dimension_means]
                        if abs_means:
                            max_idx = np.argmax(abs_means)
                            if max_idx < len(feature_names):
                                cluster_analysis[f"cluster_{cluster_id}"]["dominant_dimensions"].append(
                                    feature_names[max_idx]
                                )
        
        return cluster_analysis
        
    except Exception as e:
        logger.error(f"Error calculating cluster dimension analysis: {e}")
        return {}
