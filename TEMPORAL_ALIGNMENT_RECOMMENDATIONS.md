# Temporal Alignment Protection Recommendations

## Executive Summary

**Risk Identified**: Operations like `dropna()`, `reset_index()`, and column filtering can cause temporal misalignment when combining artifacts, especially when artifacts are saved/loaded independently.

**Core Issue**: When DataFrames with temporal data undergo row removal operations (NaN filtering, etc.) and are saved without preserving temporal indices, subsequent artifact combination can concatenate misaligned data.

---

## Current Risk Points Identified

### 1. **Index Management Issues**

**Problem Areas:**
- `src/utils/artifact_manager.py:642` - Saves CSV with `index=True` but parquet handling varies
- `src/utils/artifact_manager.py:687` - `to_parquet()` without explicit index preservation
- Multiple locations use `reset_index(drop=True)` which converts temporal indices to integers

**Example Risk:**
```python
# File 1: After removing NaN rows, saves with integer index
df1 = df1.dropna().reset_index(drop=True)  # Index now 0-999
artifact_manager.save(df1, "artifact_1")

# File 2: Different NaN pattern, different row count
df2 = df2.dropna().reset_index(drop=True)  # Index now 0-1050
artifact_manager.save(df2, "artifact_2")

# Later: Combining these artifacts
artifact_1 = artifact_manager.get_artifact("artifact_1")  # 1000 rows, index 0-999
artifact_2 = artifact_manager.get_artifact("artifact_2")  # 1051 rows, index 0-1050
combined = pd.concat([artifact_1, artifact_2], axis=1)  # MISALIGNED!
```

### 2. **Data Combination Patterns**

**Found at:**
- `src/training/steps/model_training/unified_models_training_step.py:135`
  ```python
  training_data = pd.concat([aligned_training_data, aligned_additional_outputs], axis=1)
  ```
- Multiple `pd.concat()` operations without explicit index validation

---

## Recommended Solutions

### **Solution 1: Enforce Datetime Index Preservation (CRITICAL)**

#### A. Update ArtifactManager to Always Preserve Temporal Indices

**Location**: `src/utils/artifact_manager.py`

**Changes needed**:

1. **Mandatory datetime index for financial data**:
```python
def _save_artifact_to_parquet(self, data: Any, artifact_name: str,
                             artifact_type: str = "data",
                             compression: str = "auto",
                             metadata: Optional[Dict] = None) -> str:
    """Save artifact ensuring temporal index preservation."""
    try:
        # ADDITION: Validate and preserve temporal index
        if isinstance(data, pd.DataFrame):
            data = self._ensure_temporal_index(data, artifact_name)

        # Generate enhanced filename and path
        file_extension = "parquet"
        enhanced_path = self._get_enhanced_path(
            self._current_step_name, artifact_name, file_extension
        )

        # Save with index ALWAYS preserved for DataFrames
        if isinstance(data, pd.DataFrame):
            # CRITICAL: Always save with index=True for DataFrames
            data.to_parquet(enhanced_path, compression='snappy', index=True)

            # ADDITION: Save index type to metadata
            if metadata is None:
                metadata = {}
            metadata['index_type'] = str(type(data.index).__name__)
            metadata['index_name'] = data.index.name
            metadata['has_datetime_index'] = isinstance(data.index, pd.DatetimeIndex)
        # ... rest of method
```

2. **Add temporal index validation helper**:
```python
def _ensure_temporal_index(self, df: pd.DataFrame, artifact_name: str) -> pd.DataFrame:
    """
    Ensure DataFrame has a proper temporal index for financial data.

    If no datetime index exists, attempt to create one from timestamp columns.
    Raises warning if temporal index cannot be established.
    """
    # If already has DatetimeIndex, return as-is
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Check for timestamp columns
    timestamp_candidates = ['timestamp', 'datetime', 'date', 'time', 'open_time']

    for col in timestamp_candidates:
        if col in df.columns:
            try:
                temp_df = df.copy()
                temp_df.index = pd.to_datetime(temp_df[col])
                temp_df = temp_df.drop(columns=[col])
                self.logger.info(
                    f"Converted '{col}' column to DatetimeIndex for artifact '{artifact_name}'"
                )
                return temp_df
            except Exception as e:
                self.logger.debug(f"Failed to convert '{col}' to DatetimeIndex: {e}")

    # Warning: No temporal index found
    self.logger.warning(
        f"⚠️ Artifact '{artifact_name}' has no DatetimeIndex and no timestamp column found. "
        f"This may cause temporal misalignment issues when combining artifacts."
    )

    return df

def _validate_temporal_alignment(self, *dataframes: pd.DataFrame,
                                operation: str = "combine") -> bool:
    """
    Validate that multiple DataFrames are temporally aligned before combining.

    Args:
        *dataframes: DataFrames to check
        operation: Description of operation for logging

    Returns:
        True if aligned, raises ValueError if not
    """
    if len(dataframes) < 2:
        return True

    reference_df = dataframes[0]

    # Check all have DatetimeIndex
    for i, df in enumerate(dataframes):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame {i} in {operation} does not have DatetimeIndex. "
                f"Found: {type(df.index).__name__}. "
                f"All DataFrames must have DatetimeIndex for temporal alignment."
            )

    # Check indices match exactly
    for i, df in enumerate(dataframes[1:], start=1):
        if not reference_df.index.equals(df.index):
            # Calculate overlap
            common_idx = reference_df.index.intersection(df.index)
            only_ref = len(reference_df.index.difference(df.index))
            only_other = len(df.index.difference(reference_df.index))

            raise ValueError(
                f"Temporal misalignment detected in {operation}!\n"
                f"DataFrame 0: {len(reference_df)} rows, range: {reference_df.index.min()} to {reference_df.index.max()}\n"
                f"DataFrame {i}: {len(df)} rows, range: {df.index.min()} to {df.index.max()}\n"
                f"Common timestamps: {len(common_idx)}\n"
                f"Only in DataFrame 0: {only_ref} timestamps\n"
                f"Only in DataFrame {i}: {only_other} timestamps\n"
                f"Use .reindex() or .loc[] with common index before combining."
            )

    return True
```

---

### **Solution 2: Add Safe Combination Methods to BaseStep**

**Location**: `src/training/steps/base_step.py`

**Add these methods**:

```python
def _safe_concat(self, dataframes: List[pd.DataFrame], axis: int = 1,
                 operation_name: str = "concatenate") -> pd.DataFrame:
    """
    Safely concatenate DataFrames with temporal alignment validation.

    Args:
        dataframes: List of DataFrames to concatenate
        axis: Concatenation axis (0=rows, 1=columns)
        operation_name: Description for logging

    Returns:
        Concatenated DataFrame with validated alignment
    """
    if not dataframes:
        raise ValueError("No DataFrames provided for concatenation")

    if len(dataframes) == 1:
        return dataframes[0]

    # Validate temporal alignment
    self.artifact_manager._validate_temporal_alignment(*dataframes, operation=operation_name)

    # Perform safe concatenation
    result = pd.concat(dataframes, axis=axis)

    self.logger.info(
        f"✅ Safe concatenation of {len(dataframes)} DataFrames: "
        f"axis={axis}, result shape={result.shape}"
    )

    return result

def _safe_merge(self, left: pd.DataFrame, right: pd.DataFrame,
                how: str = 'inner', validate_alignment: bool = True) -> pd.DataFrame:
    """
    Safely merge DataFrames with temporal alignment validation.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        how: Merge type ('inner', 'outer', 'left', 'right')
        validate_alignment: Whether to validate temporal alignment

    Returns:
        Merged DataFrame
    """
    if validate_alignment:
        # For inner/left/right joins, validate indices are compatible
        if how in ['inner', 'left']:
            if not isinstance(left.index, pd.DatetimeIndex):
                raise ValueError("Left DataFrame must have DatetimeIndex")
        if how in ['inner', 'right']:
            if not isinstance(right.index, pd.DatetimeIndex):
                raise ValueError("Right DataFrame must have DatetimeIndex")

    # Merge on index (temporal alignment)
    result = left.join(right, how=how, rsuffix='_right')

    # Log merge statistics
    overlap = len(left.index.intersection(right.index))
    self.logger.info(
        f"✅ Safe merge: left={len(left)}, right={len(right)}, "
        f"result={len(result)}, overlap={overlap}, how='{how}'"
    )

    return result

def _align_to_reference(self, reference: pd.DataFrame,
                        *dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames to a reference DataFrame's index.

    Args:
        reference: Reference DataFrame with target index
        *dataframes: DataFrames to align

    Returns:
        List of aligned DataFrames (same length as input dataframes)
    """
    if not isinstance(reference.index, pd.DatetimeIndex):
        raise ValueError("Reference DataFrame must have DatetimeIndex")

    aligned = []
    for i, df in enumerate(dataframes):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame {i} must have DatetimeIndex for alignment")

        # Align using reindex
        aligned_df = df.reindex(reference.index)

        # Log alignment statistics
        missing = aligned_df.isna().all(axis=1).sum()
        self.logger.info(
            f"Aligned DataFrame {i}: {len(df)} -> {len(aligned_df)} rows, "
            f"{missing} missing timestamps filled with NaN"
        )

        aligned.append(aligned_df)

    return aligned
```

---

### **Solution 3: Add Validation Decorators**

**Create new file**: `src/training/steps/temporal_validation.py`

```python
"""Decorators and utilities for temporal alignment validation."""

import functools
import pandas as pd
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def ensure_datetime_index(func: Callable) -> Callable:
    """
    Decorator to ensure function returns DataFrame with DatetimeIndex.

    Usage:
        @ensure_datetime_index
        def load_data(self) -> pd.DataFrame:
            return self._get_artifact("data")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        result = func(*args, **kwargs)

        if isinstance(result, pd.DataFrame):
            if not isinstance(result.index, pd.DatetimeIndex):
                logger.warning(
                    f"Function {func.__name__} returned DataFrame without DatetimeIndex. "
                    f"Found: {type(result.index).__name__}"
                )

        return result

    return wrapper


def validate_temporal_alignment(func: Callable) -> Callable:
    """
    Decorator to validate temporal alignment of DataFrames passed to function.

    Usage:
        @validate_temporal_alignment
        def combine_features(self, df1: pd.DataFrame, df2: pd.DataFrame):
            return pd.concat([df1, df2], axis=1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Extract DataFrame arguments
        dataframes = [
            arg for arg in args if isinstance(arg, pd.DataFrame)
        ] + [
            val for val in kwargs.values() if isinstance(val, pd.DataFrame)
        ]

        if len(dataframes) >= 2:
            # Check all have DatetimeIndex
            for i, df in enumerate(dataframes):
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(
                        f"Argument {i} in {func.__name__} must have DatetimeIndex. "
                        f"Found: {type(df.index).__name__}"
                    )

            # Check indices match
            reference_idx = dataframes[0].index
            for i, df in enumerate(dataframes[1:], start=1):
                if not reference_idx.equals(df.index):
                    common = len(reference_idx.intersection(df.index))
                    raise ValueError(
                        f"Temporal misalignment in {func.__name__}! "
                        f"DataFrame 0: {len(dataframes[0])} rows, "
                        f"DataFrame {i}: {len(df)} rows, "
                        f"Common: {common} rows"
                    )

        return func(*args, **kwargs)

    return wrapper
```

---

### **Solution 4: Update Existing Code to Use Safe Methods**

**Example**: Update `unified_models_training_step.py:135`

**Before**:
```python
training_data = pd.concat([aligned_training_data, aligned_additional_outputs], axis=1)
```

**After**:
```python
training_data = self._safe_concat(
    [aligned_training_data, aligned_additional_outputs],
    axis=1,
    operation_name="merge_training_and_additional_features"
)
```

---

### **Solution 5: Add Artifact Metadata Validation**

**Enhancement to `_save_artifact`** in `base_step.py`:

```python
def _save_artifact(self, data: Any, artifact_name: str,
                  artifact_type: str = "data",
                  compression: str = "auto",
                  metadata: Optional[Dict] = None) -> str:
    """Save artifact with temporal metadata."""
    try:
        # ADDITION: Capture temporal metadata
        if isinstance(data, pd.DataFrame):
            if metadata is None:
                metadata = {}

            # Store temporal information
            metadata['row_count'] = len(data)
            metadata['has_datetime_index'] = isinstance(data.index, pd.DatetimeIndex)

            if isinstance(data.index, pd.DatetimeIndex):
                metadata['index_start'] = str(data.index.min())
                metadata['index_end'] = str(data.index.max())
                metadata['index_frequency'] = str(data.index.inferred_freq)
                metadata['index_tz'] = str(data.index.tz) if data.index.tz else None
            else:
                self.logger.warning(
                    f"⚠️ Saving artifact '{artifact_name}' without DatetimeIndex. "
                    f"Current index type: {type(data.index).__name__}"
                )

        # Continue with existing save logic...
        artifact_path = self.artifact_manager.save(
            data=data,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            compression=compression,
            metadata=metadata
        )

        return artifact_path

    except Exception as e:
        self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
        raise
```

---

## Implementation Priority

### Phase 1 (Critical - Immediate):
1. ✅ Add `_ensure_temporal_index()` to ArtifactManager
2. ✅ Add `_validate_temporal_alignment()` to ArtifactManager
3. ✅ Update `_save_artifact_to_parquet()` to always save with `index=True`
4. ✅ Add temporal metadata to all artifact saves

### Phase 2 (High - Within 1 week):
5. ✅ Add `_safe_concat()`, `_safe_merge()`, `_align_to_reference()` to BaseStep
6. ✅ Create `temporal_validation.py` with decorators
7. ✅ Update high-risk locations (unified_models_training_step.py, etc.)

### Phase 3 (Medium - Within 2 weeks):
8. ✅ Add validation tests for temporal alignment
9. ✅ Create migration guide for existing artifacts
10. ✅ Add runtime warnings for artifacts without DatetimeIndex

---

## Testing Recommendations

### Unit Tests:

```python
def test_temporal_alignment_validation():
    """Test that misaligned DataFrames are detected."""
    df1 = pd.DataFrame(
        {'a': [1, 2, 3]},
        index=pd.date_range('2024-01-01', periods=3)
    )
    df2 = pd.DataFrame(
        {'b': [4, 5, 6, 7]},  # Different length
        index=pd.date_range('2024-01-01', periods=4)
    )

    with pytest.raises(ValueError, match="Temporal misalignment"):
        artifact_manager._validate_temporal_alignment(df1, df2)

def test_safe_concat_with_alignment():
    """Test that safe_concat properly aligns DataFrames."""
    df1 = pd.DataFrame(
        {'a': [1, 2, 3]},
        index=pd.date_range('2024-01-01', periods=3)
    )
    df2 = pd.DataFrame(
        {'b': [4, 5, 6]},
        index=pd.date_range('2024-01-01', periods=3)
    )

    result = base_step._safe_concat([df1, df2], axis=1)
    assert len(result) == 3
    assert list(result.columns) == ['a', 'b']
```

---

## Migration Guide for Existing Code

### Step 1: Audit Current Artifacts

Run this script to identify artifacts without DatetimeIndex:

```python
def audit_artifacts():
    """Audit all saved artifacts for temporal index."""
    from pathlib import Path
    import pandas as pd

    artifacts_dir = Path("artifacts")
    issues = []

    for parquet_file in artifacts_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            if not isinstance(df.index, pd.DatetimeIndex):
                issues.append({
                    'file': str(parquet_file),
                    'index_type': type(df.index).__name__,
                    'row_count': len(df)
                })
        except Exception as e:
            issues.append({
                'file': str(parquet_file),
                'error': str(e)
            })

    # Report
    print(f"Found {len(issues)} artifacts with potential issues:")
    for issue in issues:
        print(f"  - {issue}")
```

### Step 2: Re-save Artifacts with Temporal Indices

For artifacts that need fixing:

```python
def fix_artifact_temporal_index(artifact_path: str, timestamp_column: str = 'timestamp'):
    """Fix an artifact by converting to DatetimeIndex."""
    df = pd.read_parquet(artifact_path)

    if timestamp_column in df.columns:
        df.index = pd.to_datetime(df[timestamp_column])
        df = df.drop(columns=[timestamp_column])

    # Re-save with proper index
    df.to_parquet(artifact_path, index=True, compression='snappy')
```

---

## Monitoring and Alerts

Add logging to track temporal alignment issues:

```python
# In base_step.py
def _log_temporal_alignment_warning(self, artifact_name: str, issue: str):
    """Log temporal alignment warnings for monitoring."""
    warning_msg = f"⚠️ TEMPORAL ALIGNMENT WARNING: {artifact_name} - {issue}"
    self.logger.warning(warning_msg)

    # Could also:
    # - Write to dedicated temporal_alignment.log file
    # - Send to monitoring system (Prometheus, etc.)
    # - Add to step outcome report
```

---

## Summary

**Key Changes Required**:

1. **Always preserve DatetimeIndex** when saving/loading DataFrames
2. **Validate temporal alignment** before combining artifacts
3. **Use safe combination methods** (`_safe_concat`, `_safe_merge`)
4. **Add temporal metadata** to all artifacts
5. **Implement validation decorators** for critical functions

**Benefits**:
- ✅ Prevents data corruption from temporal misalignment
- ✅ Provides clear error messages when misalignment occurs
- ✅ Enables easy auditing of artifact temporal consistency
- ✅ Minimal performance overhead (validation only at combination time)

**Risk Mitigation**:
- Current risk: **HIGH** (silent data corruption possible)
- After implementation: **LOW** (explicit validation prevents misalignment)
