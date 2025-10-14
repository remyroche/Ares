# Deprecation Notice

## ⚠️ **IMPORTANT: Files Marked for Deprecation**

The following files are **deprecated** and will be removed in future versions. Please migrate to the consolidated implementation.

## Deprecated Files

### Core Pipeline Files
- `core/unified_pipeline.py` → **Use `consolidated_pipeline.py`**
- `core/enhanced_unified_pipeline.py` → **Use `consolidated_pipeline.py`**
- `enhanced_unified_pipeline.py` → **Use `consolidated_pipeline.py`**
- `enhanced_components/enhanced_unified_pipeline.py` → **Use `consolidated_pipeline.py`**

### Deprecated Result Classes
- `FeaturePipelineResult` → **Use `ConsolidatedPipelineResult`**
- `EnhancedFeaturePipelineResult` → **Use `ConsolidatedPipelineResult`**

### Deprecated Functions
- `process_features()` → **Use `process_with_unified_pipeline()`**
- `create_enhanced_unified_pipeline()` → **Use `create_unified_pipeline()`**

## Migration Timeline

### Phase 1: Current (Immediate)
- ✅ Consolidated implementation available
- ✅ Deprecated files marked
- ✅ Migration guide provided
- ⚠️ Deprecated files still functional but not recommended

### Phase 2: Next Release
- 🔄 Deprecated files will show warnings
- 🔄 Legacy imports will redirect to consolidated version
- 🔄 Documentation will focus on consolidated version

### Phase 3: Future Release
- ❌ Deprecated files will be removed
- ❌ Legacy imports will be removed
- ❌ Only consolidated implementation will remain

## Why Consolidation?

### Problems with Previous Implementation
1. **Multiple Duplicate Classes**: 4+ different pipeline implementations
2. **Redundant Components**: Same functionality implemented multiple times
3. **Confusing API**: Different method signatures and result classes
4. **Maintenance Overhead**: Changes needed in multiple places
5. **Memory Waste**: Duplicate components loaded unnecessarily

### Benefits of Consolidation
1. **Single Source of Truth**: One implementation with all features
2. **Unified API**: Consistent method signatures and result classes
3. **Better Performance**: Shared resources and optimized workflow
4. **Easier Maintenance**: Changes in one place
5. **Enhanced Features**: All advanced features integrated

## Quick Migration

### Before (Deprecated)
```python
# Multiple different implementations
from .core.unified_pipeline import UnifiedDataDrivenPipeline, FeaturePipelineResult
from .core.enhanced_unified_pipeline import EnhancedUnifiedDataDrivenPipeline, EnhancedFeaturePipelineResult

pipeline1 = UnifiedDataDrivenPipeline(config)
pipeline2 = EnhancedUnifiedDataDrivenPipeline(config)

result1 = pipeline1.process(data, targets)  # FeaturePipelineResult
result2 = pipeline2.process(data, targets)  # EnhancedFeaturePipelineResult
```

### After (Consolidated)
```python
# Single, unified implementation
from .consolidated_pipeline import UnifiedDataDrivenPipeline, ConsolidatedPipelineResult

pipeline = UnifiedDataDrivenPipeline(config)
result = pipeline.process(data, targets, feature_columns, timeframe)  # ConsolidatedPipelineResult
```

## Support

- 📖 **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- 📚 **Documentation**: Updated README and examples
- 🆘 **Help**: Contact development team for migration assistance

## Timeline

- **Now**: Start migrating to consolidated version
- **Next Release**: Deprecated files will show warnings
- **Future Release**: Deprecated files will be removed

**Please migrate as soon as possible to avoid future breaking changes.**