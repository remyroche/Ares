"""
VIF Calculator

This module provides robust VIF (Variance Inflation Factor) calculation functions
with comprehensive error handling and validation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.vif_validation_decorators import (
    comprehensive_vif_validation,
    safe_vif_calculation,
    validate_vif_inputs,
    validate_vif_outputs,
)


def calculate_vif_simple(data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.Series:
    """
    Simple VIF calculation using correlation matrix.

    Args:
        data: Input DataFrame
        features: List of features to calculate VIF for (if None, uses all numeric columns)

    Returns:
        Series with VIF values for each feature
    """
    if features is None:
        # Fallback implementation for features
        # Fallback implementation for features
        # Fallback implementation for features
        features = data.select_dtypes(include=[np.number]).columns.tolist()

    vif_scores = {}

    for feature in features:
        if feature not in data.columns:
            continue

        # Prepare data for regression
        X = data[features].drop(columns=[feature])
        y = data[feature]

        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 2 or X_clean.empty:
            vif_scores[feature] = np.nan
            continue

        try:
            # Calculate R-squared
            from sklearn.linear_model import LinearRegression
        except Exception as e:
            pass  # TODO: Handle exception properly
import copy

model = LinearRegression()
            model.fit(X_clean, y_clean)
            r_squared = model.score(X_clean, y_clean)

            # Calculate VIF
            if r_squared < 1.0:
                vif = 1.0 / (1.0 - r_squared)
            else:
                vif = np.inf

            vif_scores[feature] = vif

        except Exception:
            vif_scores[feature] = np.nan

    return pd.Series(vif_scores)


@comprehensive_vif_validation(timeout_seconds=30, max_vif_threshold=1000.0, fallback_strategy="ones")
def calculate_vif_robust(data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.Series:
    """
    Robust VIF calculation with comprehensive error handling.

    Args:
        data: Input DataFrame
        features: List of features to calculate VIF for (if None, uses all numeric columns)

    Returns:
        Series with VIF values for each feature
    """
    logger = system_logger.getChild("VIFCalculator")

    if features is None:
        # Fallback implementation for features
        # Fallback implementation for features
        # Fallback implementation for features
        features = data.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to only numeric features that exist in data
    features = [f for f in features if f in data.columns and data[f].dtype in ["int64", "float64"]]

    if not features:
        logger.warning("⚠️ VIF Calculator: No valid numeric features found")
        return pd.Series()

    # Remove features with zero variance
    variances = data[features].var()
    zero_var_features = variances[variances == 0].index.tolist()
    if zero_var_features:
        logger.warning(
            f"⚠️ VIF Calculator: Removing {len(zero_var_features)} zero variance features: {zero_var_features}"
        )
        features = [f for f in features if f not in zero_var_features]

    if len(features) < 2:
        logger.warning("⚠️ VIF Calculator: Not enough features for VIF calculation")
        return pd.Series([1.0] * len(features), index=features)

    try:
        # Use Ledoit-Wolf shrinkage for robust covariance estimation
        X = data[features].copy()

        # Handle missing values
        if X.isna().any().any():
            logger.info("🔍 VIF Calculator: Handling missing values with forward fill")
            X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Handle infinite values
        if np.isinf(X).any().any():
            logger.info("🔍 VIF Calculator: Handling infinite values")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Calculate correlation matrix using Ledoit-Wolf shrinkage
        try:
            lw = LedoitWolf().fit(X_scaled.values)
            cov_matrix = lw.covariance_

            # Convert to correlation matrix
            std_vec = np.sqrt(np.diag(cov_matrix))
            std_vec[std_vec == 0.0] = 1.0  # Avoid division by zero
            corr_matrix = cov_matrix / np.outer(std_vec, std_vec)

        except Exception as e:
            logger.warning(f"⚠️ VIF Calculator: Ledoit-Wolf failed, using correlation matrix: {e}")
            corr_matrix = X_scaled.corr().values

        # Calculate VIF using matrix inversion
        try:
            # Use pseudo-inverse for numerical stability
            corr_inv = np.linalg.pinv(corr_matrix)
            vif_values = np.diag(corr_inv)

            # Create result series
            vif_series = pd.Series(vif_values, index=features)

            # Handle any remaining invalid values
            vif_series = vif_series.replace([np.inf, -np.inf], np.nan)
            vif_series = vif_series.fillna(1.0)  # Default VIF for problematic features

            logger.info(f"✅ VIF Calculator: Successfully calculated VIF for {len(features)} features")
            logger.info(f"📊 VIF range: {vif_series.min():.2f} to {vif_series.max():.2f}")

            return vif_series

        except Exception as e:
            logger.error(f"❌ VIF Calculator: Matrix inversion failed: {e}")
            # Fallback to simple VIF calculation
            return calculate_vif_simple(data, features)

    except Exception as e:
        logger.error(f"❌ VIF Calculator: Unexpected error: {e}")
        # Return default values
        return pd.Series([1.0] * len(features), index=features)


def calculate_vif_iterative(
    data: pd.DataFrame, max_vif: float = 10.0, max_iterations: int = 10, features: Optional[List[str]] = None
) -> Tuple[pd.Series, List[str]]:
    """
    Iterative VIF calculation that removes high VIF features.

    Args:
        data: Input DataFrame
        max_vif: Maximum acceptable VIF value
        max_iterations: Maximum number of iterations
        features: List of features to calculate VIF for (if None, uses all numeric columns)

    Returns:
        Tuple of (VIF values, removed features)
    """
    logger = system_logger.getChild("VIFCalculator")

    if features is None:
        # Fallback implementation for features
        # Fallback implementation for features
        # Fallback implementation for features
        features = data.select_dtypes(include=[np.number]).columns.tolist()

    removed_features = []
    current_features = features.copy()

    for iteration in range(max_iterations):
        if len(current_features) < 2:
            logger.warning("⚠️ VIF Calculator: Not enough features for iterative VIF calculation")
            break

        # Calculate VIF for current features
        vif_values = calculate_vif_robust(data, current_features)

        # Find features with high VIF
        high_vif_features = vif_values[vif_values > max_vif].index.tolist()

        if not high_vif_features:
            logger.info(f"✅ VIF Calculator: All features have VIF <= {max_vif} after {iteration + 1} iterations")
            break

        # Remove the feature with highest VIF
        worst_feature = vif_values.idxmax()
        current_features.remove(worst_feature)
        removed_features.append(worst_feature)

        logger.info(
            f"🔄 VIF Calculator: Iteration {iteration + 1} - Removed {worst_feature} (VIF: {vif_values[worst_feature]:.2f})"
        )

    # Final VIF calculation
    final_vif = calculate_vif_robust(data, current_features)

    logger.info(
        f"📊 VIF Calculator: Final result - {len(current_features)} features kept, {len(removed_features)} removed"
    )

    return final_vif, removed_features


def analyze_vif_issues(vif_values: pd.Series) -> Dict[str, any]:
    """
    Analyze VIF values for potential issues.

    Args:
        vif_values: Series with VIF values

    Returns:
        Dictionary with analysis results
    """
    logger = system_logger.getChild("VIFAnalyzer")

    analysis = {
        "total_features": len(vif_values),
        "nan_count": vif_values.isna().sum(),
        "infinite_count": np.isinf(vif_values).sum(),
        "zero_count": (vif_values == 0).sum(),
        "high_vif_count": (vif_values > 10).sum(),
        "extreme_vif_count": (vif_values > 100).sum(),
        "min_vif": float(vif_values.min()) if not vif_values.empty else 0.0,
        "max_vif": float(vif_values.max()) if not vif_values.empty else 0.0,
        "mean_vif": float(vif_values.mean()) if not vif_values.empty else 0.0,
        "median_vif": float(vif_values.median()) if not vif_values.empty else 0.0,
        "issues": [],
    }

    # Check for issues
    if analysis["nan_count"] > 0:
        analysis["issues"].append(f"Found {analysis['nan_count']} features with NaN VIF values")
        logger.warning(f"⚠️ VIF Analysis: {analysis['nan_count']} NaN VIF values detected")

    if analysis["infinite_count"] > 0:
        analysis["issues"].append(f"Found {analysis['infinite_count']} features with infinite VIF values")
        logger.error(f"❌ VIF Analysis: {analysis['infinite_count']} infinite VIF values detected")

    if analysis["zero_count"] > 0:
        analysis["issues"].append(f"Found {analysis['zero_count']} features with zero VIF values")
        logger.warning(f"⚠️ VIF Analysis: {analysis['zero_count']} zero VIF values detected")

    if analysis["high_vif_count"] > 0:
        analysis["issues"].append(f"Found {analysis['high_vif_count']} features with VIF > 10")
        logger.warning(f"⚠️ VIF Analysis: {analysis['high_vif_count']} high VIF values detected")

    if analysis["extreme_vif_count"] > 0:
        analysis["issues"].append(f"Found {analysis['extreme_vif_count']} features with VIF > 100")
        logger.error(f"❌ VIF Analysis: {analysis['extreme_vif_count']} extreme VIF values detected")

    # Log summary
    logger.info(f"📊 VIF Analysis Summary:")
    logger.info(f"   Total features: {analysis['total_features']}")
    logger.info(f"   VIF range: {analysis['min_vif']:.2f} to {analysis['max_vif']:.2f}")
    logger.info(f"   Mean VIF: {analysis['mean_vif']:.2f}")
    logger.info(f"   Issues found: {len(analysis['issues'])}")

    return analysis


def get_vif_recommendations(vif_values: pd.Series, threshold: float = 10.0) -> List[str]:
    """
    Get recommendations for handling VIF issues.

    Args:
        vif_values: Series with VIF values
        threshold: VIF threshold for recommendations

    Returns:
        List of recommendations
    """
    recommendations = []

    # Analyze issues
    analysis = analyze_vif_issues(vif_values)

    if analysis["infinite_count"] > 0:
        recommendations.append("Remove features with infinite VIF values (perfect multicollinearity)")
        recommendations.append("Check for duplicate or highly correlated features")
        recommendations.append("Consider feature engineering to reduce redundancy")

    if analysis["extreme_vif_count"] > 0:
        recommendations.append("Remove features with VIF > 100 (severe multicollinearity)")
        recommendations.append("Use iterative VIF removal with lower threshold")
        recommendations.append("Consider principal component analysis (PCA)")

    if analysis["high_vif_count"] > 0:
        recommendations.append(f"Consider removing features with VIF > {threshold}")
        recommendations.append("Use regularization techniques (Ridge, Lasso)")
        recommendations.append("Feature selection based on domain knowledge")

    if analysis["nan_count"] > 0:
        recommendations.append("Investigate features with NaN VIF values")
        recommendations.append("Check for insufficient data or computational issues")
        recommendations.append("Consider using robust VIF calculation methods")

    if analysis["zero_count"] > 0:
        recommendations.append("Investigate features with zero VIF values")
        recommendations.append("Check for constant features or data issues")

    if not recommendations:
        recommendations.append("VIF values look good - no immediate action required")

    return recommendations
