"""Metric calculations for NAS/TAS regime analysis."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def calculate_regime_distribution(labels: np.ndarray, regime_type: str) -> Dict[str, Any]:
    """Calculate distribution statistics for regimes."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)

    distribution = {
        "regime_type": regime_type,
        "total_samples": int(total_samples),
        "num_regimes": len(unique_labels),
        "regime_counts": {},
        "regime_percentages": {},
        "regime_balance": {},
    }

    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100 if total_samples else 0.0
        distribution["regime_counts"][f"regime_{int(label)}"] = int(count)
        distribution["regime_percentages"][f"regime_{int(label)}"] = round(percentage, 2)

    percentages = list(distribution["regime_percentages"].values()) or [0.0]
    distribution["regime_balance"] = {
        "min_percentage": round(float(np.min(percentages)), 2),
        "max_percentage": round(float(np.max(percentages)), 2),
        "std_percentage": round(float(np.std(percentages)), 2),
        "balance_score": round(float(1.0 - (np.std(percentages) / 100)), 3),
    }
    return distribution


def calculate_clustering_metrics(features: np.ndarray, labels: np.ndarray, regime_type: str) -> Dict[str, Any]:
    """Calculate clustering quality metrics for the provided feature set."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    silhouette = silhouette_score(features_scaled, labels)
    davies_bouldin = davies_bouldin_score(features_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
    cv_score = calculate_cv_score(features_scaled, labels)

    return {
        "regime_type": regime_type,
        "silhouette_score": round(float(silhouette), 4),
        "davies_bouldin_score": round(float(davies_bouldin), 4),
        "calinski_harabasz_score": round(float(calinski_harabasz), 4),
        "cv_score": round(float(cv_score), 4),
        "interpretation": {
            "silhouette": interpret_silhouette(silhouette),
            "davies_bouldin": interpret_davies_bouldin(davies_bouldin),
            "cv_score": interpret_cv_score(cv_score),
        },
    }


def calculate_cv_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Calculate coefficient of variation score matching the original heuristics."""
    unique_labels = np.unique(labels)
    within_cv_scores = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]
        if len(cluster_features) <= 1:
            continue
        feature_cvs = []
        for feature_idx in range(cluster_features.shape[1]):
            feature_values = cluster_features[:, feature_idx]
            std = np.std(feature_values)
            mean_abs = np.mean(np.abs(feature_values))
            if std > 0 and mean_abs > 0:
                feature_cvs.append(std / mean_abs)
        if feature_cvs:
            within_cv_scores.append(np.mean(feature_cvs))

    cluster_centers = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]
        if len(cluster_features) > 0:
            cluster_centers.append(np.mean(cluster_features, axis=0))
    if len(cluster_centers) > 1:
        cluster_centers = np.asarray(cluster_centers)
        between_cv = np.std(cluster_centers) / np.mean(np.abs(cluster_centers))
    else:
        between_cv = 0.0

    within_cv = float(np.mean(within_cv_scores)) if within_cv_scores else 0.0
    return float(0.6 * max(0.0, 1.0 - within_cv) + 0.4 * min(1.0, between_cv))


def interpret_silhouette(score: float) -> str:
    """Interpret silhouette score."""
    if score >= 0.7:
        return "Excellent clustering"
    if score >= 0.5:
        return "Good clustering"
    if score >= 0.3:
        return "Fair clustering"
    return "Poor clustering"


def interpret_davies_bouldin(score: float) -> str:
    """Interpret Davies-Bouldin score (lower is better)."""
    if score <= 0.5:
        return "Excellent separation"
    if score <= 1.0:
        return "Good separation"
    if score <= 2.0:
        return "Fair separation"
    return "Poor separation"


def interpret_cv_score(score: float) -> str:
    """Interpret coefficient of variation score."""
    if score >= 0.8:
        return "Excellent regime distinction"
    if score >= 0.6:
        return "Good regime distinction"
    if score >= 0.4:
        return "Fair regime distinction"
    return "Poor regime distinction"
