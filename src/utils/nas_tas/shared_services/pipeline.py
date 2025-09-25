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
        working_data = working_data.ffill().bfill()
        if logger:
            logger.info("✅ Filled missing values using forward/backward fill")
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
        working_data = working_data.ffill().bfill()
        if logger:
            logger.info("✅ Replaced infinite values and filled using forward/backward fill")
    metadata["infinite_values"] = inf_summary

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

    if "close" in data.columns:
        data["price_change"] = data["close"].pct_change()
        data["price_volatility"] = data["price_change"].rolling(window=20).std()
        data["price_momentum"] = data["close"] / data["close"].shift(20)
        data["ma_5"] = data["close"].rolling(window=5).mean()
        data["ma_20"] = data["close"].rolling(window=20).mean()
        data["ma_50"] = data["close"].rolling(window=50).mean()
        rolling_min = data["close"].rolling(window=20).min()
        rolling_max = data["close"].rolling(window=20).max()
        data["price_position_20"] = (data["close"] - rolling_min) / (rolling_max - rolling_min)

    if "volume" in data.columns:
        data["volume_change"] = data["volume"].pct_change()
        data["volume_ma"] = data["volume"].rolling(window=20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_ma"]

    if "high" in data.columns and "low" in data.columns and "close" in data.columns:
        data["price_range"] = (data["high"] - data["low"]) / data["close"]
        data["range_volatility"] = data["price_range"].rolling(window=20).std()

    if isinstance(data.index, pd.DatetimeIndex):
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["month"] = data.index.month

    pre_drop_shape = data.shape
    data = data.dropna()
    dropped_rows = pre_drop_shape[0] - data.shape[0]
    if dropped_rows > 0:
        warnings.append(f"Dropped {dropped_rows} rows due to rolling calculations")
        if logger:
            logger.warning(
                "⚠️ Dropped %d rows during feature engineering due to NaNs", dropped_rows
            )

    added_features = sorted(set(data.columns) - original_columns)
    if logger:
        logger.info("✅ Feature engineering completed - New shape: %s", data.shape)

    metadata = {"dropped_rows": dropped_rows}
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
