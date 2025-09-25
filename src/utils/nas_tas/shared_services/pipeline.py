"""Shared pipeline helpers for NAS/TAS data preparation and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.nas_tas.risk_analysis.risk_analysis import (
    RiskAnalyzer,
    RiskConfig,
    RiskResult,
)
from src.utils.validation.unified_framework import UnifiedValidationFramework


def _impute_numeric_series(series: pd.Series) -> pd.Series:
    """Forward-fill numeric data and backfill remaining gaps conservatively."""

    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.ffill()
    if series.isna().any():
        median = series.dropna().median()
        if pd.isna(median):
            median = 0.0
        series = series.fillna(median)
    return series


@dataclass
class DataValidationResult:
    """Outcome of shared market data validation."""

    data: pd.DataFrame
    feature_columns: List[str]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureEngineeringResult:
    """Outcome of shared feature engineering."""

    data: pd.DataFrame
    added_features: List[str]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_market_data(
    market_data: pd.DataFrame,
    target_variable: str,
    feature_columns: Optional[List[str]] = None,
    *,
    logger: Optional[Any] = None,
    validation_framework: Optional[UnifiedValidationFramework] = None,
) -> DataValidationResult:
    """Validate core assumptions required by NAS/TAS training pipelines."""

    if target_variable not in market_data.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in data")

    if feature_columns is None:
        feature_columns = [col for col in market_data.columns if col != target_variable]

    working_data = market_data.copy()
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}

    # Missing value handling
    missing_values = working_data.isnull().sum()
    missing_summary = {
        column: int(count)
        for column, count in missing_values.items()
        if int(count) > 0
    }
    if missing_summary:
        warnings.append(
            f"Filled missing values in {len(missing_summary)} columns"
        )
        if logger:
            logger.warning(
                "⚠️ Found missing values in %d columns: %s",
                len(missing_summary),
                missing_summary,
            )
        if logger:
            logger.info(
                "✅ Applying forward-fill and median imputation for missing values"
            )
    metadata["missing_values"] = missing_summary

    # Infinite value handling
    numeric_data = working_data.select_dtypes(include=[np.number])
    inf_values = np.isinf(numeric_data).sum()
    inf_summary = {
        column: int(count) for column, count in inf_values.items() if int(count) > 0
    }
    if inf_summary:
        warnings.append(
            f"Replaced infinite values in {len(inf_summary)} columns"
        )
        if logger:
            logger.warning(
                "⚠️ Found infinite values in %d columns: %s",
                len(inf_summary),
                inf_summary,
            )
        working_data = working_data.replace([np.inf, -np.inf], np.nan)
        if logger:
            logger.info(
                "✅ Replaced infinite values and will re-impute affected columns"
            )
    metadata["infinite_values"] = inf_summary

    if missing_summary or inf_summary:
        numeric_columns = working_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            working_data.loc[:, numeric_columns] = working_data.loc[
                :, numeric_columns
            ].apply(_impute_numeric_series)
    metadata["imputation_strategy"] = "forward_fill_with_median_backfill"

    # Non-numeric feature detection
    numeric_columns = set(numeric_data.columns)
    non_numeric_features = [
        column for column in feature_columns if column not in numeric_columns
    ]
    if non_numeric_features and logger:
        logger.warning("⚠️ Non-numeric feature columns detected: %s", non_numeric_features)
    metadata["non_numeric_features"] = non_numeric_features

    # Optional additional validation via unified framework
    if validation_framework is not None:
        try:
            validation_framework.validate_category(
                "data_quality",
                working_data,
                context={"required_columns": [target_variable]},
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Unified validation failed: {exc}")
            if logger:
                logger.warning("⚠️ Unified validation failed: %s", exc)

    if logger:
        logger.info("✅ Data validation completed - Shape: %s", working_data.shape)

    return DataValidationResult(
        data=working_data,
        feature_columns=list(feature_columns),
        warnings=warnings,
        metadata=metadata,
    )


def engineer_core_features(
    market_data: pd.DataFrame,
    *,
    logger: Optional[Any] = None,
) -> FeatureEngineeringResult:
    """Generate a consistent set of engineered features used by NAS/TAS."""

    data = market_data.copy()
    original_columns = set(data.columns)
    warnings: List[str] = []

    filled_features: List[str] = []

    if "close" in data.columns:
        close = data["close"].astype(float)
        price_change = close.pct_change().replace([np.inf, -np.inf], np.nan)
        data["price_change"] = price_change
        data["price_volatility"] = (
            price_change.rolling(window=20, min_periods=5).std()
        )
        data["price_momentum"] = close.pct_change(periods=20).replace(
            [np.inf, -np.inf], np.nan
        )
        data["ma_5"] = close.rolling(window=5, min_periods=1).mean()
        data["ma_20"] = close.rolling(window=20, min_periods=1).mean()
        data["ma_50"] = close.rolling(window=50, min_periods=1).mean()
        rolling_min = close.rolling(window=20, min_periods=1).min()
        rolling_max = close.rolling(window=20, min_periods=1).max()
        denom = (rolling_max - rolling_min).replace(0, np.nan)
        data["price_position_20"] = ((close - rolling_min) / denom).clip(0.0, 1.0)

    if "volume" in data.columns:
        volume = data["volume"].astype(float)
        data["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        volume_ma = volume.rolling(window=20, min_periods=5).mean()
        data["volume_ma"] = volume_ma
        data["volume_ratio"] = np.divide(
            volume,
            volume_ma.replace(0, np.nan),
        )

    if "high" in data.columns and "low" in data.columns and "close" in data.columns:
        close = data["close"].replace(0, np.nan)
        data["price_range"] = (data["high"] - data["low"]) / close
        data["range_volatility"] = (
            data["price_range"].rolling(window=20, min_periods=5).std()
        )

    if isinstance(data.index, pd.DatetimeIndex):
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["month"] = data.index.month

    added_features = sorted(set(data.columns) - original_columns)
    for feature in added_features:
        series = data[feature]
        if series.isna().any() or np.isinf(series).any():
            filled_features.append(feature)

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        data.loc[:, numeric_columns] = data.loc[:, numeric_columns].apply(
            _impute_numeric_series
        )

    if filled_features:
        warnings.append(
            f"Imputed warm-up values for {len(filled_features)} engineered features"
        )
        if logger:
            logger.info(
                "ℹ️ Imputed initial values for engineered features: %s",
                filled_features,
            )

    if logger:
        logger.info("✅ Feature engineering completed - Shape unchanged: %s", data.shape)

    metadata = {
        "imputed_features": filled_features,
        "dropped_rows": 0,
    }
    return FeatureEngineeringResult(
        data=data,
        added_features=added_features,
        warnings=warnings,
        metadata=metadata,
    )


def run_shared_risk_analysis(
    returns_series: pd.Series,
    *,
    risk_config: Optional[RiskConfig] = None,
    analyzer: Optional[RiskAnalyzer] = None,
    benchmark_returns: Optional[pd.Series] = None,
    regime_data: Optional[Dict[str, Any]] = None,
    factor_data: Optional[Dict[str, pd.Series]] = None,
) -> RiskResult:
    """Execute risk analysis using the shared NAS/TAS risk infrastructure."""

    risk_config = risk_config or RiskConfig()
    analyzer = analyzer or RiskAnalyzer(risk_config)
    return analyzer.run_analysis(
        returns_series=returns_series,
        benchmark_returns=benchmark_returns,
        regime_data=regime_data,
        factor_data=factor_data,
    )
