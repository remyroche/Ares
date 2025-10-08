"""
Standardized Labeling Interface

This module provides a standardized interface for passing labels and weights
between components in the pre-training pipeline, ensuring consistent data
flow and proper error handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from .column_naming import (
    ColumnNamespace,
    ensure_namespace,
    filter_namespace_columns,
    find_nonconforming_columns,
    standardize_namespace_frame,
    validate_dataframe_names,
)

try:  # Pandera is required for schema enforcement; keep import guarded for tests.
    import pandera as pa
    from pandera import Check, DataFrameSchema
except ImportError as exc:  # pragma: no cover - defensive fallback
    pa = None  # type: ignore[assignment]
    DataFrameSchema = None  # type: ignore[assignment]
    Check = None  # type: ignore[assignment]


if pa:
    def _namespace_check(df: pd.DataFrame) -> bool:
        return not find_nonconforming_columns(df.columns)


    LabelFrameSchema: DataFrameSchema = pa.DataFrameSchema(
        columns={},
        checks=[
            Check(lambda df: not df.empty, error="Label frame cannot be empty"),
            Check(_namespace_check, error="Label frame contains non-namespaced columns"),
            Check(
                lambda df: bool(filter_namespace_columns(df.columns, ColumnNamespace.TARGET)),
                error="Label frame must contain at least one target column",
            ),
        ],
        strict=False,
        coerce=False,
    )

    FeatureFrameSchema: DataFrameSchema = pa.DataFrameSchema(
        columns={},
        checks=[
            Check(lambda df: not df.empty, error="Feature frame cannot be empty"),
            Check(_namespace_check, error="Feature frame contains non-namespaced columns"),
            Check(
                lambda df: bool(filter_namespace_columns(df.columns, ColumnNamespace.FEATURE)),
                error="Feature frame must contain at least one namespaced feature column",
            ),
        ],
        strict=False,
        coerce=False,
    )
else:
    LabelFrameSchema = None  # type: ignore[assignment]
    FeatureFrameSchema = None  # type: ignore[assignment]


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    expected_dtypes: Optional[Dict[str, type]] = None,
    min_rows: int = 0,
    allow_nulls: bool = True,
    schema: Optional[DataFrameSchema] = None,
    allowed_unprefixed: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema against expected structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        expected_dtypes: Dictionary mapping column names to expected dtypes
        min_rows: Minimum number of rows required
        allow_nulls: Whether null values are acceptable
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if df is None:
        issues.append("DataFrame is None")
        return False, issues
    
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
    
    # Check dtypes
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = df[col].dtype
                if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
                    issues.append(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
    
    # Check for nulls if not allowed
    if not allow_nulls:
        null_columns = df.columns[df.isnull().any()].tolist()
        if null_columns:
            issues.append(f"Columns with null values: {null_columns}")
    
    # Check namespace conventions
    try:
        validate_dataframe_names(df, allowed_unprefixed=allowed_unprefixed)
    except ValueError as exc:
        issues.append(str(exc))

    # Run Pandera schema validation if available
    schema_to_use: Optional[DataFrameSchema] = schema
    if pa is not None:
        if schema_to_use is None:
            if filter_namespace_columns(df.columns, ColumnNamespace.TARGET):
                schema_to_use = LabelFrameSchema
            elif filter_namespace_columns(df.columns, ColumnNamespace.FEATURE):
                schema_to_use = FeatureFrameSchema

        if schema_to_use is not None:
            try:
                schema_to_use.validate(df, lazy=True)
            except pa.errors.SchemaErrors as exc:  # type: ignore[attr-defined]
                issues.extend(sorted({msg for msg in exc.failure_cases["failure_case"].astype(str)}))
            except pa.errors.SchemaError as exc:  # type: ignore[attr-defined]
                issues.append(str(exc))

    # Check for duplicate indices
    if df.index.has_duplicates:
        dup_count = df.index.duplicated().sum()
        issues.append(f"DataFrame has {dup_count} duplicate index values")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        tprint(f"✅ DataFrame schema validation passed: {len(df)} rows, {len(df.columns)} columns")
    else:
        tprint_warning(f"⚠️ DataFrame schema validation found {len(issues)} issues:")
        for issue in issues:
            tprint_warning(f"  - {issue}")
    
    return is_valid, issues


def assert_labels_sigma_scaled(labels: pd.DataFrame, tolerance: float = 0.35) -> None:
    """Assert that label variance remains close to 1 (σ-normalized scale)."""
    if labels is None or labels.empty:
        return

    numeric_columns = labels.select_dtypes(include=[np.number]).columns
    target_like_columns = filter_namespace_columns(numeric_columns, ColumnNamespace.TARGET)
    if not target_like_columns:
        target_like_columns = filter_namespace_columns(numeric_columns, ColumnNamespace.LABEL)

    if not target_like_columns:
        return

    lower_bound = 1 - tolerance
    upper_bound = 1 + tolerance

    for col in target_like_columns:
        series = labels[col].dropna()
        if series.empty:
            continue

        if len(series) < 2:
            # Need at least two observations to compute a finite sample variance; skip otherwise.
            continue

        # Use sample variance (ddof=1) for σ-normalized labels
        variance = float(series.var(ddof=1))
        if not np.isfinite(variance):
            raise ValueError(f"Variance for label column '{col}' is not finite.")

        if variance < lower_bound or variance > upper_bound:
            raise ValueError(
                f"Label column '{col}' variance {variance:.3f} deviates from expected σ-normalized scale "
                f"(expected ~1.0 ± {tolerance})."
            )


class LabelingFormat(Enum):
    """Supported labeling formats."""
    STANDARDIZED = "standardized"
    MULTI_HORIZON = "multi_horizon"
    TRIPLE_BARRIER = "triple_barrier"


@dataclass
class LabelingMetadata:
    """Metadata for labeling results."""
    source_component: str
    creation_time: str
    pipeline_ready: bool
    symbol: str
    exchange: str
    timeframe: str
    n_samples: int
    n_targets: int
    n_horizons: int
    error: Optional[str] = None


@dataclass
class StandardizedLabelingResult:
    """Standardized labeling result that all components can use."""
    labels: pd.DataFrame
    weights: Dict[str, float]
    target_columns: List[str]
    quality_scores: Dict[str, Any]
    confidence_scores: pd.DataFrame
    eligibility_masks: pd.DataFrame
    metadata: LabelingMetadata
    
    def is_valid(self) -> bool:
        """Check if the labeling result is valid."""
        return (
            not self.labels.empty and
            len(self.target_columns) > 0 and
            self.metadata.pipeline_ready and
            self.metadata.error is None
        )
    
    HORIZON_KEYWORD_MAPPING = [
        ("micro", ("micro",)),
        ("small", ("immediate", "small")),
        ("medium", ("short", "medium")),
        ("high", ("long", "leverage", "high")),
    ]

    def _resolve_weight_key(self, target_name: str) -> str:
        """Map a target name to the appropriate weight key."""
        target_lower = target_name.lower()
        for weight_key, keywords in self.HORIZON_KEYWORD_MAPPING:
            if any(keyword in target_lower for keyword in keywords):
                return weight_key
        return "small"

    def get_best_target(self) -> Optional[str]:
        """Get the best target based on weights."""
        if not self.weights or not self.target_columns:
            # No weights available, use first available target
            available_targets = filter_namespace_columns(self.labels.columns, ColumnNamespace.TARGET)
            return available_targets[0] if available_targets else None

        # Priority order based on horizon weights (higher weight = higher priority)
        target_priority = []

        for target in self.target_columns:
            if target in self.labels.columns:
                weight_key = self._resolve_weight_key(target)
                horizon_weight = self.weights.get(weight_key, 0.0)
                target_priority.append((target, horizon_weight))

        # Sort by weight (descending) and return the highest weighted target
        if target_priority:
            target_priority.sort(key=lambda x: x[1], reverse=True)
            return target_priority[0][0]

        return None


class StandardizedLabelingInterface:
    """Interface for standardized labeling data exchange between components."""
    
    @staticmethod
    def create_from_multi_horizon_result(
        multi_horizon_result: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> StandardizedLabelingResult:
        """Create standardized result from multi_horizon_profit_labeler output."""
        try:
            tprint_info("🔄 Converting multi_horizon_profit_labeler result to standardized format")
            
            # Extract data from multi_horizon_result
            labeled_data = multi_horizon_result.get('labeled_data', pd.DataFrame())
            horizon_weights = multi_horizon_result.get('horizon_weights', {})
            target_columns = multi_horizon_result.get('target_columns', [])
            quality_scores = multi_horizon_result.get('quality_scores', {})
            confidence_scores = multi_horizon_result.get('confidence_scores', pd.DataFrame())
            eligibility_masks = multi_horizon_result.get('eligibility_masks', pd.DataFrame())

            labeled_data = standardize_namespace_frame(labeled_data, ColumnNamespace.TARGET)
            confidence_scores = standardize_namespace_frame(confidence_scores, ColumnNamespace.TARGET)
            eligibility_masks = standardize_namespace_frame(eligibility_masks, ColumnNamespace.TARGET)

            if target_columns:
                target_columns = [ensure_namespace(col, ColumnNamespace.TARGET) for col in target_columns]
            else:
                target_columns = filter_namespace_columns(labeled_data.columns, ColumnNamespace.TARGET)

            # Create metadata
            metadata = LabelingMetadata(
                source_component='multi_horizon_profit_labeler',
                creation_time=datetime.now().isoformat(),
                pipeline_ready=True,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=len(labeled_data) if not labeled_data.empty else 0,
                n_targets=len(target_columns),
                n_horizons=len(horizon_weights)
            )
            
            result = StandardizedLabelingResult(
                labels=labeled_data,
                weights=horizon_weights,
                target_columns=target_columns,
                quality_scores=quality_scores,
                confidence_scores=confidence_scores,
                eligibility_masks=eligibility_masks,
                metadata=metadata
            )
            
            tprint_success("✅ Successfully created standardized labeling result")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to create standardized result: {e}")
            # Return empty result with error
            metadata = LabelingMetadata(
                source_component='multi_horizon_profit_labeler',
                creation_time=datetime.now().isoformat(),
                pipeline_ready=False,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=0,
                n_targets=0,
                n_horizons=0,
                error=str(e)
            )
            
            return StandardizedLabelingResult(
                labels=pd.DataFrame(),
                weights={},
                target_columns=[],
                quality_scores={},
                confidence_scores=pd.DataFrame(),
                eligibility_masks=pd.DataFrame(),
                metadata=metadata
            )
    
    @staticmethod
    def create_from_standardized_output(
        standardized_output: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> StandardizedLabelingResult:
        """Create standardized result from standardized output format."""
        try:
            tprint_info("🔄 Processing standardized output format")

            # Extract data from standardized output
            labels = standardized_output.get('labels', pd.DataFrame())
            weights = standardized_output.get('weights', {})
            target_columns = standardized_output.get('target_columns', [])
            quality_scores = standardized_output.get('quality_scores', {})
            confidence_scores = standardized_output.get('confidence_scores', pd.DataFrame())
            eligibility_masks = standardized_output.get('eligibility_masks', pd.DataFrame())

            labels = standardize_namespace_frame(labels, ColumnNamespace.TARGET)
            confidence_scores = standardize_namespace_frame(confidence_scores, ColumnNamespace.TARGET)
            eligibility_masks = standardize_namespace_frame(eligibility_masks, ColumnNamespace.TARGET)

            if target_columns:
                target_columns = [ensure_namespace(col, ColumnNamespace.TARGET) for col in target_columns]
            else:
                target_columns = filter_namespace_columns(labels.columns, ColumnNamespace.TARGET)

            assert_labels_sigma_scaled(labels)

            # Create metadata
            metadata = LabelingMetadata(
                source_component=standardized_output.get('metadata', {}).get('source_component', 'unknown'),
                creation_time=standardized_output.get('metadata', {}).get('creation_time', datetime.now().isoformat()),
                pipeline_ready=standardized_output.get('metadata', {}).get('pipeline_ready', True),
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=len(labels) if not labels.empty else 0,
                n_targets=len(target_columns),
                n_horizons=len(weights)
            )
            
            result = StandardizedLabelingResult(
                labels=labels,
                weights=weights,
                target_columns=target_columns,
                quality_scores=quality_scores,
                confidence_scores=confidence_scores,
                eligibility_masks=eligibility_masks,
                metadata=metadata
            )
            
            tprint_success("✅ Successfully processed standardized output")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to process standardized output: {e}")
            # Return empty result with error
            metadata = LabelingMetadata(
                source_component='unknown',
                creation_time=datetime.now().isoformat(),
                pipeline_ready=False,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                n_samples=0,
                n_targets=0,
                n_horizons=0,
                error=str(e)
            )
            
            return StandardizedLabelingResult(
                labels=pd.DataFrame(),
                weights={},
                target_columns=[],
                quality_scores={},
                confidence_scores=pd.DataFrame(),
                eligibility_masks=pd.DataFrame(),
                metadata=metadata
            )
    
    @staticmethod
    def extract_from_pipeline_state(
        pipeline_state: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[StandardizedLabelingResult]:
        """Extract standardized labeling result from pipeline state."""
        try:
            tprint_info("🔍 Extracting labeling result from pipeline state")
            
            # Try standardized output format first
            if 'standardized_output' in pipeline_state:
                tprint_info("📋 Found standardized output in pipeline state")
                return StandardizedLabelingInterface.create_from_standardized_output(
                    pipeline_state['standardized_output'], symbol, exchange, timeframe
                )
            
            # Try multi_horizon_labeling_result format
            if 'multi_horizon_labeling_result' in pipeline_state:
                tprint_info("📊 Found multi_horizon_labeling_result in pipeline state")
                return StandardizedLabelingInterface.create_from_multi_horizon_result(
                    pipeline_state['multi_horizon_labeling_result'], symbol, exchange, timeframe
                )
            
            # Try artifacts
            artifacts = pipeline_state.get('artifacts', {})
            if 'standardized_output' in artifacts:
                tprint_info("📋 Found standardized output in artifacts")
                return StandardizedLabelingInterface.create_from_standardized_output(
                    artifacts['standardized_output'], symbol, exchange, timeframe
                )
            
            if 'multi_horizon_labeling_result' in artifacts:
                tprint_info("📊 Found multi_horizon_labeling_result in artifacts")
                return StandardizedLabelingInterface.create_from_multi_horizon_result(
                    artifacts['multi_horizon_labeling_result'], symbol, exchange, timeframe
                )
            
            tprint_warning("⚠️ No labeling result found in pipeline state")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Failed to extract labeling result from pipeline state: {e}")
            return None
    
    @staticmethod
    def validate_result(result: StandardizedLabelingResult) -> bool:
        """Validate a standardized labeling result."""
        try:
            if not result.is_valid():
                tprint_warning("⚠️ Labeling result is not valid")
                return False
            
            # Additional validation checks
            if result.labels.empty:
                tprint_warning("⚠️ Labels DataFrame is empty")
                return False
            
            if not result.target_columns:
                tprint_warning("⚠️ No target columns specified")
                return False
            
            # Check if target columns exist in labels
            missing_targets = [col for col in result.target_columns if col not in result.labels.columns]
            if missing_targets:
                tprint_warning(f"⚠️ Missing target columns: {missing_targets}")
                return False
            
            tprint_success("✅ Labeling result validation passed")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Validation failed: {e}")
            return False
