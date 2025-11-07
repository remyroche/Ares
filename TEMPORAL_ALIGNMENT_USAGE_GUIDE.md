# Temporal Alignment Usage Guide

## Quick Start

The temporal alignment protection is now **automatically enabled** for all artifact saves and loads. Here's what you need to know:

---

## Automatic Features (No Code Changes Needed)

### 1. **Automatic Index Preservation**
All DataFrames saved via `_save_artifact()` now:
- ✅ Automatically convert timestamp columns to DatetimeIndex
- ✅ Always save with `index=True` to preserve temporal information
- ✅ Store comprehensive temporal metadata (start, end, frequency, timezone)
- ✅ Warn if no temporal index can be established

**Example** - This just works:
```python
# In any BaseStep subclass
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100),
    'price': np.random.randn(100)
})

# Automatically converts 'timestamp' to DatetimeIndex and preserves it
self._save_artifact(df, "market_data")
```

### 2. **Automatic Metadata Tracking**
Every artifact now includes temporal metadata:
```json
{
  "row_count": 1000,
  "has_datetime_index": true,
  "index_start": "2024-01-01 00:00:00",
  "index_end": "2024-12-31 23:59:00",
  "index_frequency": "1H",
  "index_is_monotonic": true
}
```

---

## Safe Combination Methods (Recommended)

### **_safe_concat()** - Validated Concatenation

**Before** (risky):
```python
combined = pd.concat([df1, df2], axis=1)  # No validation!
```

**After** (safe):
```python
combined = self._safe_concat(
    [df1, df2],
    axis=1,
    operation_name="combine_features",
    validate_alignment=True  # Default
)
```

**What it does**:
- ✅ Validates all DataFrames have DatetimeIndex
- ✅ Checks indices match exactly
- ✅ Provides detailed error message if misaligned
- ✅ Logs success with shape information

**Error Example**:
```
ValueError: ⚠️ Temporal misalignment detected in combine_features!
DataFrame 0: 1000 rows, range: 2024-01-01 to 2024-12-31
DataFrame 1: 1050 rows, range: 2024-01-05 to 2024-12-31
Common timestamps: 950
Only in DataFrame 0: 50 timestamps
Only in DataFrame 1: 100 timestamps
Solution: Use .reindex() or BaseStep._align_to_reference()
```

---

### **_safe_merge()** - Validated Merging

```python
# Index-based merge with validation
result = self._safe_merge(
    left=features_df,
    right=targets_df,
    how='inner',
    validate_alignment=True
)

# Column-based merge (no temporal validation)
result = self._safe_merge(
    left=df1,
    right=df2,
    on='symbol',
    how='left',
    validate_alignment=False
)
```

**Logs**:
```
✅ Safe merge: left=1000, right=1000, result=950, overlap=950, how='inner'
```

---

### **_align_to_reference()** - Explicit Alignment

Use when you need to align multiple DataFrames to a reference:

```python
# Load artifacts that may have different temporal coverage
features = self._get_artifact("features")
targets = self._get_artifact("targets")
additional = self._get_artifact("additional_features")

# Align all to the features DataFrame
aligned_targets, aligned_additional = self._align_to_reference(
    features,  # Reference
    targets,
    additional
)

# Now safe to concatenate
combined = self._safe_concat([features, aligned_targets, aligned_additional], axis=1)
```

**Logs**:
```
Aligned DataFrame 0: 1050 -> 1000 rows, 950 matched, 50 missing timestamps filled with NaN
Aligned DataFrame 1: 980 -> 1000 rows, 980 matched, 20 missing timestamps filled with NaN
```

---

## Decorators (Optional but Recommended)

### **@ensure_datetime_index** - Validate Returns

```python
from src.training.steps.temporal_validation import ensure_datetime_index

class MyStep(BaseStep):
    @ensure_datetime_index
    def load_market_data(self) -> pd.DataFrame:
        return self._get_artifact("market_data")
        # Logs warning if returned DataFrame lacks DatetimeIndex
```

---

### **@validate_temporal_alignment** - Validate Inputs

```python
from src.training.steps.temporal_validation import validate_temporal_alignment

class MyStep(BaseStep):
    @validate_temporal_alignment
    def combine_features(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        # Automatically validates df1 and df2 are aligned before execution
        return pd.concat([df1, df2], axis=1)
```

**Raises**:
```python
ValueError: Argument 1 in combine_features must have DatetimeIndex.
Found: RangeIndex. All DataFrames must have DatetimeIndex for temporal alignment.
```

---

### **@log_temporal_info** - Debug Helper

```python
from src.training.steps.temporal_validation import log_temporal_info

class MyStep(BaseStep):
    @log_temporal_info
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df['volume'] > 1000]
```

**Logs**:
```
📊 Function filter_data - Input DataFrames:
  DataFrame 0: 1000 rows, DatetimeIndex 2024-01-01 to 2024-12-31
📊 Function filter_data - Output: 850 rows, DatetimeIndex 2024-01-01 to 2024-12-31
```

---

### **@require_datetime_index** - Enforce Requirements

```python
from src.training.steps.temporal_validation import require_datetime_index

class MyStep(BaseStep):
    @require_datetime_index
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Enforces that prices has DatetimeIndex
        return prices.pct_change()
```

---

## Utility Functions

### **check_temporal_alignment()** - Programmatic Check

```python
from src.training.steps.temporal_validation import check_temporal_alignment

result = check_temporal_alignment(df1, df2, df3)

if not result['aligned']:
    self.logger.warning(f"Misalignment detected: {result['issues']}")
    self.logger.info(f"Common timestamps: {result['common_index_length']}")

    # Optionally align them
    common_idx = get_common_temporal_index(df1, df2, df3)
    df1_aligned = df1.loc[common_idx]
    df2_aligned = df2.loc[common_idx]
    df3_aligned = df3.loc[common_idx]
```

**Result Structure**:
```python
{
    'aligned': False,
    'issues': [
        'DataFrame 0 and 1 have different indices: 50 unique to df0, 100 unique to df1'
    ],
    'common_index_length': 950,
    'stats': [
        {
            'index': 0,
            'length': 1000,
            'has_datetime_index': True,
            'index_type': 'DatetimeIndex',
            'start': '2024-01-01 00:00:00',
            'end': '2024-12-31 23:59:00',
            'frequency': 'H'
        },
        # ... stats for other DataFrames
    ]
}
```

---

### **get_common_temporal_index()** - Extract Intersection

```python
from src.training.steps.temporal_validation import get_common_temporal_index

# Get common timestamps across all DataFrames
common_idx = get_common_temporal_index(df1, df2, df3)

# Use for alignment
df1_aligned = df1.loc[common_idx]
df2_aligned = df2.loc[common_idx]
df3_aligned = df3.loc[common_idx]

# Now safe to combine
combined = pd.concat([df1_aligned, df2_aligned, df3_aligned], axis=1)
```

---

## Migration Examples

### Example 1: Simple Concatenation

**Before**:
```python
training_data = pd.concat([features, targets], axis=1)
```

**After**:
```python
training_data = self._safe_concat([features, targets], axis=1, operation_name="combine_training_data")
```

---

### Example 2: Complex Multi-Source Combination

**Before**:
```python
feature_list = []
for source in sources:
    df = self._get_artifact(f"features_{source}")
    feature_list.append(df)

all_features = pd.concat(feature_list, axis=1)
```

**After**:
```python
feature_list = []
for source in sources:
    df = self._get_artifact(f"features_{source}")
    feature_list.append(df)

# Safe concatenation with validation
all_features = self._safe_concat(
    feature_list,
    axis=1,
    operation_name="combine_all_feature_sources"
)
```

---

### Example 3: Handling Misaligned Data

**Before** (silent corruption):
```python
df1 = df1.dropna()  # Removes some rows
df2 = df2.dropna()  # Removes different rows
combined = pd.concat([df1, df2], axis=1)  # MISALIGNED!
```

**After** (explicit alignment):
```python
df1 = df1.dropna()  # Removes some rows
df2 = df2.dropna()  # Removes different rows

# Option 1: Use align_to_reference
df2_aligned = self._align_to_reference(df1, df2)[0]
combined = self._safe_concat([df1, df2_aligned], axis=1)

# Option 2: Use common index
common_idx = get_common_temporal_index(df1, df2)
combined = pd.concat([df1.loc[common_idx], df2.loc[common_idx]], axis=1)
```

---

## Best Practices

### ✅ DO:

1. **Use `_safe_concat()` instead of `pd.concat()` for temporal data**
   ```python
   combined = self._safe_concat([df1, df2], axis=1)
   ```

2. **Explicitly align before combining if you know data may differ**
   ```python
   aligned = self._align_to_reference(reference_df, df1, df2)
   ```

3. **Add decorators to functions that expect temporal data**
   ```python
   @require_datetime_index
   def process_timeseries(self, df: pd.DataFrame):
       ...
   ```

4. **Check metadata when debugging alignment issues**
   ```python
   metadata = self._get_artifact_metadata("features")
   print(f"Index range: {metadata['index_start']} to {metadata['index_end']}")
   ```

### ❌ DON'T:

1. **Don't use `reset_index(drop=True)` before saving artifacts**
   ```python
   # BAD: Loses temporal information
   df = df.reset_index(drop=True)
   self._save_artifact(df, "features")
   ```

2. **Don't bypass validation without good reason**
   ```python
   # RISKY: No validation
   combined = pd.concat([df1, df2], axis=1)

   # BETTER: Validated
   combined = self._safe_concat([df1, df2], axis=1)
   ```

3. **Don't ignore temporal alignment warnings**
   ```python
   # If you see this warning, investigate!
   # ⚠️ Artifact 'features' has no DatetimeIndex and no timestamp column found
   ```

---

## Troubleshooting

### Issue: "DataFrame does not have DatetimeIndex"

**Solution**: Ensure your DataFrame has a timestamp column or DatetimeIndex:
```python
# Option 1: Let ArtifactManager handle it automatically
df['timestamp'] = pd.date_range('2024-01-01', periods=len(df))
self._save_artifact(df, "data")  # Automatically converts to DatetimeIndex

# Option 2: Set DatetimeIndex manually
df.index = pd.to_datetime(df['timestamp'])
df = df.drop(columns=['timestamp'])
```

### Issue: "Temporal misalignment detected"

**Solution**: Align DataFrames before combining:
```python
# Use align_to_reference
aligned = self._align_to_reference(reference_df, df1, df2)
combined = self._safe_concat([reference_df] + aligned, axis=1)

# Or use common index
common_idx = get_common_temporal_index(df1, df2)
df1_aligned = df1.loc[common_idx]
df2_aligned = df2.loc[common_idx]
```

### Issue: "Need to disable validation temporarily"

**Solution**: Use `validate_alignment=False`:
```python
# Only if you're absolutely sure alignment is correct
combined = self._safe_concat([df1, df2], axis=1, validate_alignment=False)
```

---

## Performance Impact

- **Validation overhead**: ~1-5ms per concatenation (negligible)
- **Index preservation**: No performance impact (just metadata)
- **Alignment operations**: O(n log n) - same as pandas native operations

---

## Next Steps

1. ✅ **Start using `_safe_concat()` in new code**
2. ✅ **Gradually migrate existing `pd.concat()` calls**
3. ✅ **Add `@require_datetime_index` to critical functions**
4. ✅ **Review warnings in logs for artifacts without DatetimeIndex**

For more details, see `TEMPORAL_ALIGNMENT_RECOMMENDATIONS.md`
