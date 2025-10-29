"""
Data quality scoring system for trading data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .constants import (
    DATA_MISSING_THRESHOLD_CRITICAL,
    DATA_MISSING_THRESHOLD_WARNING,
    EXTREME_PRICE_CHANGE_THRESHOLD
)

@dataclass
class DataQualityScore:
    """Data quality score container."""
    completeness_score: float  # 0-1, higher is better
    consistency_score: float  # 0-1, higher is better
    freshness_score: float  # 0-1, higher is better
    overall_score: float  # 0-1, weighted average
    details: Dict[str, Any]

def calculate_completeness_score(
    data: pd.DataFrame,
    required_columns: Optional[list] = None
) -> float:
    """
    Calculate data completeness score.

    Args:
        data: DataFrame to score
        required_columns: List of required columns

    Returns:
        Completeness score (0-1)
    """
    if data.empty:
        return 0.0

    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']

    total_cells = len(data) * len(required_columns)
    if total_cells == 0:
        return 0.0

    missing_cells = 0
    for col in required_columns:
        if col in data.columns:
            missing_cells += data[col].isnull().sum()
        else:
            missing_cells += len(data)

    completeness = 1.0 - (missing_cells / total_cells)
    return max(0.0, min(1.0, completeness))

def calculate_consistency_score(data: pd.DataFrame) -> float:
    """
    Calculate data consistency score.

    Args:
        data: DataFrame to score

    Returns:
        Consistency score (0-1)
    """
    if data.empty:
        return 0.0

    issues = 0
    total_checks = 0

    # Check OHLC consistency
    if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= Open, Close, Low
        high_violations = (
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['high'] < data['low'])
        ).sum()
        issues += high_violations
        total_checks += len(data) * 3

        # Low should be <= Open, Close, High
        low_violations = (
            (data['low'] > data['open']) |
            (data['low'] > data['close']) |
            (data['low'] > data['high'])
        ).sum()
        issues += low_violations
        total_checks += len(data) * 3

    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in data.columns:
            negative_count = (data[col] <= 0).sum()
            issues += negative_count
            total_checks += len(data)

    # Check for negative volume
    if 'volume' in data.columns:
        negative_volume = (data['volume'] < 0).sum()
        issues += negative_volume
        total_checks += len(data)

    if total_checks == 0:
        return 1.0

    consistency = 1.0 - (issues / total_checks)
    return max(0.0, min(1.0, consistency))

def calculate_freshness_score(
    data: pd.DataFrame,
    max_age_hours: float = 24.0
) -> float:
    """
    Calculate data freshness score.

    Args:
        data: DataFrame with datetime index
        max_age_hours: Maximum age in hours for full score

    Returns:
        Freshness score (0-1)
    """
    if data.empty:
        return 0.0

    if not isinstance(data.index, pd.DatetimeIndex):
        return 0.5  # Can't determine freshness without timestamps

    from datetime import datetime
    now = datetime.now()
    if data.index.tz:
        now = pd.Timestamp.now(tz=data.index.tz)

    latest_timestamp = data.index[-1]
    age_hours = (now - latest_timestamp).total_seconds() / 3600

    if age_hours <= 0:
        return 1.0
    elif age_hours >= max_age_hours:
        return 0.0
    else:
        freshness = 1.0 - (age_hours / max_age_hours)
        return max(0.0, min(1.0, freshness))

def calculate_data_quality_score(
    data: pd.DataFrame,
    required_columns: Optional[list] = None,
    max_age_hours: float = 24.0,
    weights: Optional[Dict[str, float]] = None
) -> DataQualityScore:
    """
    Calculate overall data quality score.

    Args:
        data: DataFrame to score
        required_columns: List of required columns
        max_age_hours: Maximum age for freshness score
        weights: Weights for different scores (default: equal weights)

    Returns:
        DataQualityScore object
    """
    if weights is None:
        weights = {
            'completeness': 0.4,
            'consistency': 0.4,
            'freshness': 0.2
        }

    completeness = calculate_completeness_score(data, required_columns)
    consistency = calculate_consistency_score(data)
    freshness = calculate_freshness_score(data, max_age_hours)

    overall = (
        completeness * weights['completeness'] +
        consistency * weights['consistency'] +
        freshness * weights['freshness']
    )

    details = {
        'completeness': completeness,
        'consistency': consistency,
        'freshness': freshness,
        'weights': weights,
        'data_shape': data.shape,
        'data_columns': list(data.columns)
    }

    return DataQualityScore(
        completeness_score=completeness,
        consistency_score=consistency,
        freshness_score=freshness,
        overall_score=overall,
        details=details
    )

def score_data_quality(
    data: pd.DataFrame,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Score data quality and return detailed report.

    Args:
        data: DataFrame to score
        threshold: Minimum acceptable score

    Returns:
        Dictionary with quality report
    """
    score = calculate_data_quality_score(data)

    return {
        'score': score.overall_score,
        'completeness': score.completeness_score,
        'consistency': score.consistency_score,
        'freshness': score.freshness_score,
        'acceptable': score.overall_score >= threshold,
        'threshold': threshold,
        'details': score.details,
        'recommendations': _generate_quality_recommendations(score)
    }

def _generate_quality_recommendations(score: DataQualityScore) -> list:
    """Generate recommendations for improving data quality."""
    recommendations = []

    if score.completeness_score < 0.9:
        recommendations.append("Data has missing values. Consider filling gaps or collecting more data.")

    if score.consistency_score < 0.9:
        recommendations.append("Data has consistency issues. Check OHLC relationships and data sources.")

    if score.freshness_score < 0.8:
        recommendations.append("Data may be stale. Ensure regular data updates.")

    if score.overall_score < 0.7:
        recommendations.append("Overall data quality is below acceptable threshold. Review data collection process.")

    return recommendations
