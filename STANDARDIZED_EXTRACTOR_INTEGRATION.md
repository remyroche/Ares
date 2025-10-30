# Standardized Regime Extractor Integration

## Overview
Successfully integrated `StandardizedRegimeExtractor` across all regime training components for consistent regime label extraction.

**Date**: 2024-10-30  
**Components Updated**: 2 (regime_models_training, regime_artifact_schema)

---

## ✅ Integration Summary

### **Architecture: Best of Both Worlds**

We now have a **unified extraction system** with two usage patterns:

1. **Simple Pattern**: Direct extraction (regime_models_training)
2. **Rich Pattern**: Artifact-wrapped extraction with metadata (regime_ensemble_training)

Both patterns use the **same underlying extraction logic** from `StandardizedRegimeExtractor`.

---

## 📊 Component-Specific Implementation

### **1. regime_models_training** ✓
**Pattern**: Direct extraction (simple)

```python
# Direct usage - returns numpy array
regime_labels = extract_regime_labels_standardized(
    pipeline_state, 
    min_samples=10, 
    min_regimes=2
)
```

**Why**: This component only needs the raw labels for training, not metadata.

**Benefits**:
- ✅ Clean, simple interface
- ✅ Fast-fail with clear errors
- ✅ Automatic validation

---

### **2. regime_ensemble_training** ✓
**Pattern**: Artifact-wrapped extraction (rich)

```python
# Wrapped usage - returns RegimeLabelsArtifact with metadata
regime_labels_artifact = artifact_extractor.extract_regime_labels(
    pipeline_state, 
    component_name="REGIME_ENSEMBLE",
    min_samples=10,
    min_regimes=2
)

regime_labels = regime_labels_artifact.cluster_assignments
n_regimes = regime_labels_artifact.n_regimes
method = regime_labels_artifact.clustering_method
distribution = regime_labels_artifact.regime_distribution
```

**Why**: This component needs metadata about clustering method, parameters, and distribution for ensemble training decisions.

**Benefits**:
- ✅ Rich metadata available
- ✅ Consistent extraction logic internally
- ✅ Backward compatible with artifact system
- ✅ Structured data for logging

---

## 🔧 How It Works

### **RegimeArtifactExtractor.extract_regime_labels()** (Updated)

```python
@staticmethod
def extract_regime_labels(...) -> Optional[RegimeLabelsArtifact]:
    # 1. Check for pre-wrapped artifact (fast path)
    if 'regime_labels_artifact' in artifacts:
        return RegimeLabelsArtifact.from_dict(...)
    
    # 2. Use StandardizedRegimeExtractor for extraction
    try:
        cluster_assignments = extract_regime_labels_standardized(
            pipeline_state,
            min_samples=min_samples,
            min_regimes=min_regimes
        )
        
        # 3. Extract metadata from pipeline artifacts
        clustering_method = extract_method_metadata(artifacts)
        clustering_params = extract_params_metadata(artifacts)
        metadata = extract_additional_metadata(artifacts)
        
        # 4. Create enriched artifact
        return RegimeLabelsArtifact(
            cluster_assignments=cluster_assignments,
            n_regimes=len(np.unique(cluster_assignments)),
            regime_distribution=calculate_distribution(cluster_assignments),
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            metadata=metadata
        )
    
    except RegimeLabelExtractionError as e:
        # Clear error handling
        return None
```

---

## 🎯 Extraction Hierarchy (Standardized)

Both patterns now use the same extraction hierarchy from `StandardizedRegimeExtractor`:

1. **Primary**: `optimal_regime_clustering_result['labels']`
2. **Secondary**: `regime_clustering_result['cluster_assignments']`
3. **Tertiary**: `gmm_regime_discovery_result['labels']`
4. **Quaternary**: `hmm_regime_discovery_result['labels']`
5. **Fallback**: Direct keys (`regime_labels`, `cluster_assignments`, `labels`)

### Metadata Enrichment Hierarchy

After extraction, `RegimeArtifactExtractor` enriches with metadata from:

1. **Primary**: `optimal_regime_clustering_result` (method, params, metadata)
2. **Secondary**: `regime_clustering_result` (method, params, metadata)
3. **GMM-specific**: `gmm_regime_discovery_result` (GMM params)
4. **HMM-specific**: `hmm_regime_discovery_result` (HMM params)

---

## 📈 Benefits of Integration

### **Consistency** ✅
- Single source of truth for extraction logic
- Same behavior across all components
- Predictable error handling

### **Maintainability** ✅
- Changes to extraction logic only in one place
- Easy to add new artifact sources
- Clear separation of concerns

### **Flexibility** ✅
- Simple pattern for components that don't need metadata
- Rich pattern for components that need structured artifacts
- Both patterns use same core logic

### **Error Handling** ✅
- Fast-fail with clear error messages
- Automatic validation (NaN, min samples, min regimes)
- Actionable error guidance

---

## 🔍 Validation Features (Inherited)

Both patterns benefit from `StandardizedRegimeExtractor` validation:

1. **Type Validation**: Ensures labels are array-like
2. **Sample Count**: Minimum samples check (default: 10)
3. **Regime Count**: Minimum unique regimes check (default: 2)
4. **NaN Detection**: Fails if NaN values present
5. **Integer Conversion**: Ensures integer labels

---

## 🔄 Migration Path

### For New Components

**Simple needs (just labels)**:
```python
from src.utils.ml_common.data.standardized_regime_extractor import (
    extract_regime_labels_standardized
)

regime_labels = extract_regime_labels_standardized(pipeline_state)
```

**Rich needs (labels + metadata)**:
```python
from src.training.steps.market_analysis.components.regime_artifact_schema import (
    RegimeArtifactExtractor
)

artifact = RegimeArtifactExtractor.extract_regime_labels(
    pipeline_state, 
    component_name="MY_COMPONENT"
)
regime_labels = artifact.cluster_assignments
method = artifact.clustering_method
```

### For Existing Components

**No changes required!** Both components continue to work:
- `regime_models_training`: Already uses simple pattern ✓
- `regime_ensemble_training`: Now uses same core logic ✓

---

## 🧪 Testing Recommendations

1. **Test Simple Pattern**:
   ```python
   # Test direct extraction
   labels = extract_regime_labels_standardized(pipeline_state)
   assert len(labels) >= 10
   assert len(np.unique(labels)) >= 2
   ```

2. **Test Rich Pattern**:
   ```python
   # Test artifact extraction
   artifact = RegimeArtifactExtractor.extract_regime_labels(pipeline_state)
   assert artifact is not None
   assert artifact.n_regimes >= 2
   assert len(artifact.cluster_assignments) >= 10
   assert artifact.clustering_method != "unknown"
   ```

3. **Test Consistency**:
   ```python
   # Both should extract same labels
   simple_labels = extract_regime_labels_standardized(pipeline_state)
   rich_artifact = RegimeArtifactExtractor.extract_regime_labels(pipeline_state)
   
   np.testing.assert_array_equal(
       simple_labels, 
       rich_artifact.cluster_assignments
   )
   ```

4. **Test Error Handling**:
   ```python
   # Test with missing labels
   empty_state = {'artifacts': {}}
   
   # Simple pattern raises exception
   with pytest.raises(RegimeLabelExtractionError):
       extract_regime_labels_standardized(empty_state)
   
   # Rich pattern returns None
   artifact = RegimeArtifactExtractor.extract_regime_labels(empty_state)
   assert artifact is None
   ```

---

## 📊 Before vs After

### Before Integration

**regime_models_training**: Custom extraction logic (200+ lines, fragile)
**regime_ensemble_training**: `RegimeArtifactExtractor` (custom logic, inconsistent)

**Problems**:
- ❌ Different extraction logic in each component
- ❌ No shared validation
- ❌ Inconsistent error handling
- ❌ Difficult to maintain

### After Integration

**regime_models_training**: Uses `extract_regime_labels_standardized()` ✓
**regime_ensemble_training**: Uses `RegimeArtifactExtractor` (powered by `StandardizedRegimeExtractor`) ✓

**Benefits**:
- ✅ Single extraction logic source
- ✅ Shared validation rules
- ✅ Consistent error handling
- ✅ Easy to maintain
- ✅ Both patterns available

---

## 🎓 Design Pattern: Adapter Pattern

This integration implements the **Adapter Pattern**:

```
┌─────────────────────────────────────────┐
│   StandardizedRegimeExtractor (Core)    │
│   - Pure extraction logic                │
│   - Validation rules                     │
│   - Error handling                       │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼─────┐  ┌─────▼──────────────────┐
│   Simple   │  │  RegimeArtifactExtractor│
│   Pattern  │  │      (Adapter)           │
│            │  │  - Wraps core extractor  │
│  Direct    │  │  - Adds metadata         │
│  numpy     │  │  - Returns artifact      │
│  array     │  │                          │
└────────────┘  └──────────────────────────┘
```

**Advantages**:
- Core logic remains simple and focused
- Adapter adds complexity only where needed
- Both patterns use validated extraction
- Easy to add new patterns

---

## 🚀 Future Enhancements

### Short Term
1. ✅ Add caching to avoid repeated extraction
2. ✅ Add metrics collection (extraction time, failures)
3. ✅ Support custom validation rules per component

### Long Term
1. ✅ Async extraction support
2. ✅ Streaming regime label updates
3. ✅ Multi-source label fusion

---

## 📝 Files Modified

1. **regime_artifact_schema.py**
   - Added import for `StandardizedRegimeExtractor`
   - Updated `extract_regime_labels()` to use standardized extraction
   - Added metadata enrichment logic
   - Lines modified: ~80

2. **regime_models_training.py** (Previously updated)
   - Already using `extract_regime_labels_standardized()` ✓
   - No additional changes needed

---

## ✨ Summary

Successfully integrated `StandardizedRegimeExtractor` across all regime training components:

- ✅ **Consistent extraction logic** - Single source of truth
- ✅ **Flexible patterns** - Simple for basic needs, rich for complex needs
- ✅ **Maintained compatibility** - No breaking changes
- ✅ **Improved error handling** - Clear, actionable messages
- ✅ **Better validation** - Automatic checks for data quality
- ✅ **Enhanced metadata** - Rich context for ensemble training

**Result**: More robust, maintainable, and consistent regime label extraction across the entire training pipeline.

---

## 🔄 **Testing → Production Mode**

The integration supports **two modes**:

### **Testing Mode** (Current - Default)
```python
# No preferred_method specified - tries all methods
artifact = RegimeArtifactExtractor.extract_regime_labels(
    pipeline_state
)
# Automatically searches: optimal → regime_clustering → gmm → hmm
```

### **Production Mode** (Future - When Method Chosen)
```python
# Specify preferred_method - only looks for that method
artifact = RegimeArtifactExtractor.extract_regime_labels(
    pipeline_state,
    preferred_method="gmm"  # Only searches for GMM
)
```

**When to switch**: After testing phase when you've chosen the best clustering method.

**See**: `CLUSTERING_METHOD_TRANSITION_GUIDE.md` for detailed migration instructions.

---

**Status**: ✅ **Integration completed successfully** (supports both testing & production modes)

