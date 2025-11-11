"""
Data quality scoring system for trading data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.utils.tprint import tprint
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
    tprint(f"[DATA_QUALITY] calculate_completeness_score: data_shape={data.shape if not data.empty else 'empty'}, required_columns={required_columns}")

    if data.empty:
        tprint(f"[DATA_QUALITY] calculate_completeness_score -> 0.0 (empty data)")
        return 0.0

    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        tprint(f"[DATA_QUALITY] calculate_completeness_score: Using default required_columns={required_columns}")

    total_cells = len(data) * len(required_columns)
    if total_cells == 0:
        tprint(f"[DATA_QUALITY] calculate_completeness_score -> 0.0 (zero total cells)")
        return 0.0

    missing_cells = 0
    for col in required_columns:
        if col in data.columns:
            col_missing = data[col].isnull().sum()
            missing_cells += col_missing
            if col_missing > 0:
                tprint(f"[DATA_QUALITY] calculate_completeness_score: Column {col} has {col_missing} missing values")
        else:
            missing_cells += len(data)
            tprint(f"[DATA_QUALITY] calculate_completeness_score: Column {col} not found in data")

    completeness = 1.0 - (missing_cells / total_cells)
    tprint(f"[DATA_QUALITY] calculate_completeness_score -> {completeness:.4f} (missing={missing_cells}/{total_cells})")
    return max(0.0, min(1.0, completeness))

def calculate_consistency_score(data: pd.DataFrame) -> float:
    """
    Calculate data consistency score.

    Args:
        data: DataFrame to score

    Returns:
        Consistency score (0-1)
    """
    tprint(f"[DATA_QUALITY] calculate_consistency_score: data_shape={data.shape if not data.empty else 'empty'}")

    if data.empty:
        tprint(f"[DATA_QUALITY] calculate_consistency_score -> 0.0 (empty data)")
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
        if high_violations > 0:
            tprint(f"[DATA_QUALITY] calculate_consistency_score: Found {high_violations} high price violations")
        issues += high_violations
        total_checks += len(data) * 3

        # Low should be <= Open, Close, High
        low_violations = (
            (data['low'] > data['open']) |
            (data['low'] > data['close']) |
            (data['low'] > data['high'])
        ).sum()
        if low_violations > 0:
            tprint(f"[DATA_QUALITY] calculate_consistency_score: Found {low_violations} low price violations")
        issues += low_violations
        total_checks += len(data) * 3

    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in data.columns:
            negative_count = (data[col] <= 0).sum()
            if negative_count > 0:
                tprint(f"[DATA_QUALITY] calculate_consistency_score: Found {negative_count} non-positive values in {col}")
            issues += negative_count
            total_checks += len(data)

    # Check for negative volume
    if 'volume' in data.columns:
        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            tprint(f"[DATA_QUALITY] calculate_consistency_score: Found {negative_volume} negative volume values")
        issues += negative_volume
        total_checks += len(data)

    if total_checks == 0:
        tprint(f"[DATA_QUALITY] calculate_consistency_score -> 1.0 (no checks performed)")
        return 1.0

    consistency = 1.0 - (issues / total_checks)
    tprint(f"[DATA_QUALITY] calculate_consistency_score -> {consistency:.4f} (issues={issues}/{total_checks})")
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
    tprint(f"[DATA_QUALITY] calculate_freshness_score: data_shape={data.shape if not data.empty else 'empty'}, max_age_hours={max_age_hours}")

    if data.empty:
        tprint(f"[DATA_QUALITY] calculate_freshness_score -> 0.0 (empty data)")
        return 0.0

    if not isinstance(data.index, pd.DatetimeIndex):
        tprint(f"[DATA_QUALITY] calculate_freshness_score -> 0.5 (no datetime index)")
        return 0.5  # Can't determine freshness without timestamps

    from datetime import datetime
    now = datetime.now()
    if data.index.tz:
        now = pd.Timestamp.now(tz=data.index.tz)

    latest_timestamp = data.index[-1]
    age_hours = (now - latest_timestamp).total_seconds() / 3600
    tprint(f"[DATA_QUALITY] calculate_freshness_score: Data age={age_hours:.2f} hours")

    if age_hours <= 0:
        tprint(f"[DATA_QUALITY] calculate_freshness_score -> 1.0 (future timestamp)")
        return 1.0
    elif age_hours >= max_age_hours:
        tprint(f"[DATA_QUALITY] calculate_freshness_score -> 0.0 (stale data)")
        return 0.0
    else:
        freshness = 1.0 - (age_hours / max_age_hours)
        tprint(f"[DATA_QUALITY] calculate_freshness_score -> {freshness:.4f}")
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
    tprint(f"[DATA_QUALITY] calculate_data_quality_score: data_shape={data.shape}, required_columns={required_columns}, max_age_hours={max_age_hours}")

    if weights is None:
        weights = {
            'completeness': 0.4,
            'consistency': 0.4,
            'freshness': 0.2
        }
        tprint(f"[DATA_QUALITY] calculate_data_quality_score: Using default weights={weights}")

    completeness = calculate_completeness_score(data, required_columns)
    consistency = calculate_consistency_score(data)
    freshness = calculate_freshness_score(data, max_age_hours)

    overall = (
        completeness * weights['completeness'] +
        consistency * weights['consistency'] +
        freshness * weights['freshness']
    )
    tprint(f"[DATA_QUALITY] calculate_data_quality_score: Scores - completeness={completeness:.4f}, consistency={consistency:.4f}, freshness={freshness:.4f}, overall={overall:.4f}")

    details = {
        'completeness': completeness,
        'consistency': consistency,
        'freshness': freshness,
        'weights': weights,
        'data_shape': data.shape,
        'data_columns': list(data.columns)
    }

    result = DataQualityScore(
        completeness_score=completeness,
        consistency_score=consistency,
        freshness_score=freshness,
        overall_score=overall,
        details=details
    )
    tprint(f"[DATA_QUALITY] calculate_data_quality_score -> DataQualityScore(overall={overall:.4f})")
    return result

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
    tprint(f"[DATA_QUALITY] score_data_quality: data_shape={data.shape}, threshold={threshold}")

    score = calculate_data_quality_score(data)
    acceptable = score.overall_score >= threshold

    tprint(f"[DATA_QUALITY] score_data_quality: Overall score={score.overall_score:.4f}, acceptable={acceptable}")

    recommendations = _generate_quality_recommendations(score)
    if recommendations:
        tprint(f"[DATA_QUALITY] score_data_quality: Generated {len(recommendations)} recommendations")

    result = {
        'score': score.overall_score,
        'completeness': score.completeness_score,
        'consistency': score.consistency_score,
        'freshness': score.freshness_score,
        'acceptable': acceptable,
        'threshold': threshold,
        'details': score.details,
        'recommendations': recommendations
    }
    tprint(f"[DATA_QUALITY] score_data_quality -> report with {len(recommendations)} recommendations")
    return result

def _generate_quality_recommendations(score: DataQualityScore) -> list:
    """Generate recommendations for improving data quality."""
    tprint(f"[DATA_QUALITY] _generate_quality_recommendations: completeness={score.completeness_score:.4f}, consistency={score.consistency_score:.4f}, freshness={score.freshness_score:.4f}")

    recommendations = []

    if score.completeness_score < 0.9:
        recommendations.append("Data has missing values. Consider filling gaps or collecting more data.")
        tprint(f"[DATA_QUALITY] _generate_quality_recommendations: Added completeness recommendation")

    if score.consistency_score < 0.9:
        recommendations.append("Data has consistency issues. Check OHLC relationships and data sources.")
        tprint(f"[DATA_QUALITY] _generate_quality_recommendations: Added consistency recommendation")

    if score.freshness_score < 0.8:
        recommendations.append("Data may be stale. Ensure regular data updates.")
        tprint(f"[DATA_QUALITY] _generate_quality_recommendations: Added freshness recommendation")

    if score.overall_score < 0.7:
        recommendations.append("Overall data quality is below acceptable threshold. Review data collection process.")
        tprint(f"[DATA_QUALITY] _generate_quality_recommendations: Added overall quality recommendation")

    tprint(f"[DATA_QUALITY] _generate_quality_recommendations -> {len(recommendations)} recommendations")
    return recommendations
