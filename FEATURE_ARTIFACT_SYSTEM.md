# Feature Artifact System

## Overview

The Feature Artifact System provides persistent caching for feature engineering outputs, allowing subsequent pipeline steps to load pre-computed features without re-engineering them. This significantly improves pipeline efficiency and enables step-by-step execution.

## Key Features

### 1. Persistent Artifacts
- **Automatic Caching**: Features are automatically saved after Step 2 execution
- **Hash-based Invalidation**: Features are regenerated when input data changes
- **Metadata Tracking**: Comprehensive metadata for reproducibility and debugging

### 2. Smart Loading
- **Existence Checking**: Automatically detects if artifacts exist
- **Validation**: Ensures artifact integrity before loading
- **Error Handling**: Graceful fallbacks when artifacts are missing

### 3. Security & Monitoring
- **Decorator Protection**: All functions use training pipeline decorators
- **Resource Monitoring**: Memory, CPU, and disk usage tracking
- **Circuit Breaker**: Automatic failure detection and recovery
- **Quality Gates**: Data quality validation at multiple levels

## Architecture

### File Structure
```
data/training/
├── {exchange}_{symbol}_features_train.parquet      # Training features
├── {exchange}_{symbol}_features_validation.parquet # Validation features  
├── {exchange}_{symbol}_features_test.parquet       # Test features
├── {exchange}_{symbol}_features_metadata.json      # Feature metadata
└── {exchange}_{symbol}_features_hash.txt           # Artifact hash
```

### Core Components

#### 1. Feature Artifact Loader (`src/training/steps/feature_artifact_loader.py`)
- **`load_features_for_step()`**: Main entry point for steps needing features
- **`check_feature_artifacts_exist()`**: Validates artifact existence
- **`load_feature_artifacts()`**: Loads all feature splits
- **`get_feature_artifact_info()`**: Retrieves comprehensive metadata
- **`validate_feature_artifacts()`**: Performs integrity checks

#### 2. Step 2 Integration (`src/training/steps/step2_feature_engineering.py`)
- **Artifact Creation**: Automatically saves features after engineering
- **Hash Generation**: Creates unique hash based on input data
- **Metadata Persistence**: Saves configuration and statistics
- **Force Rerun**: Supports `--force-rerun` to regenerate artifacts

#### 3. Step 3 Integration (`src/training/steps/step3_hmm_regime_discovery.py`)
- **Feature Loading**: Uses artifacts instead of re-engineering
- **Data Alignment**: Ensures features align with price data
- **Error Handling**: Graceful fallback if artifacts missing

## Usage

### Basic Usage

```python
from src.training.steps.feature_artifact_loader import load_features_for_step

# Load features for a specific step
features = load_features_for_step(
    symbol="ETHUSDT",
    exchange="BINANCE", 
    data_dir="data/training",
    step_name="Step3.HMMRegimeDiscovery"
)

# Access individual splits
train_features = features["train"]
validation_features = features["validation"]
test_features = features["test"]
```

### Advanced Usage

```python
from src.training.steps.feature_artifact_loader import (
    check_feature_artifacts_exist,
    get_feature_artifact_info,
    validate_feature_artifacts
)

# Check if artifacts exist
if check_feature_artifacts_exist("ETHUSDT", "BINANCE", "data/training"):
    # Get comprehensive info
    info = get_feature_artifact_info("ETHUSDT", "BINANCE", "data/training")
    print(f"Features created: {info['created_at']}")
    print(f"Total features: {info['total_features']}")
    
    # Validate artifacts
    is_valid, message = validate_feature_artifacts("ETHUSDT", "BINANCE", "data/training")
    if is_valid:
        print("✅ Artifacts are valid")
    else:
        print(f"❌ Validation failed: {message}")
```

### Command Line Usage

```bash
# Run Step 2 to create artifacts
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_feature_engineering

# Run Step 3+ directly (will use cached artifacts)
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step3_hmm_regime_discovery

# Force regeneration of artifacts
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_feature_engineering --force-rerun
```

## Security Features

### Decorator Protection
All functions are protected with comprehensive decorators:

- **`@validate_step_prerequisites`**: Validates environment and dependencies
- **`@secure_data_processing`**: Ensures data integrity and security
- **`@resource_monitor`**: Monitors memory, CPU, and disk usage
- **`@memory_efficient`**: Optimizes memory usage for large datasets
- **`@debug_training_step`**: Provides detailed logging and debugging
- **`@circuit_breaker_protection`**: Prevents cascading failures
- **`@validate_step_output`**: Validates function outputs
- **`@quality_gate`**: Ensures data quality standards

### Resource Limits
- **Memory**: 16GB threshold for feature loading
- **CPU**: 70% threshold for intensive operations
- **Disk**: 10GB threshold for storage operations
- **Timeouts**: Configurable timeouts for all operations

## Error Handling

### Graceful Degradation
- **Missing Artifacts**: Clear error messages with recovery instructions
- **Corrupted Files**: Automatic detection and reporting
- **Resource Exhaustion**: Circuit breaker prevents system overload
- **Data Quality Issues**: Quality gates prevent poor data propagation

### Recovery Strategies
- **Automatic Retry**: Circuit breaker with exponential backoff
- **Fallback Loading**: Alternative loading methods when primary fails
- **Data Validation**: Comprehensive validation before use
- **Error Context**: Detailed error context for debugging

## Performance Optimization

### Memory Efficiency
- **Streaming Processing**: Large datasets processed in chunks
- **Memory Pooling**: Reuses memory buffers
- **Cleanup Frequency**: Regular memory cleanup
- **Chunk Size Optimization**: Configurable chunk sizes per operation

### Caching Strategy
- **Hash-based Invalidation**: Only regenerates when data changes
- **Metadata Caching**: Fast metadata lookups
- **Path Caching**: Cached file path generation
- **Validation Caching**: Cached validation results

## Monitoring & Debugging

### Logging
- **Structured Logging**: Consistent log format across all functions
- **Performance Metrics**: Timing and resource usage tracking
- **Error Context**: Detailed error information for debugging
- **Step-specific Logging**: Context-aware logging per step

### Debug Artifacts
- **Performance Profiles**: Detailed performance analysis
- **Memory Snapshots**: Memory usage at key points
- **Error Dumps**: Complete error context preservation
- **Validation Reports**: Data quality validation results

## Configuration

### Environment Variables
```bash
# Memory limits
export FEATURE_ARTIFACT_MEMORY_LIMIT_GB=16
export FEATURE_ARTIFACT_CPU_LIMIT_PERCENT=70

# Timeouts
export FEATURE_ARTIFACT_LOAD_TIMEOUT_SECONDS=60
export FEATURE_ARTIFACT_VALIDATION_TIMEOUT_SECONDS=45

# Quality thresholds
export FEATURE_ARTIFACT_MIN_COMPLETENESS=0.9
export FEATURE_ARTIFACT_MIN_CONSISTENCY=0.8
```

### Quality Gates
- **Completeness**: Minimum 90% data completeness
- **Consistency**: Minimum 80% data consistency  
- **Feature Quality**: Minimum 70% feature quality score
- **Performance**: Maximum 60 seconds loading time

## Best Practices

### Development
1. **Always Check Existence**: Use `check_feature_artifacts_exist()` before loading
2. **Validate After Loading**: Use `validate_feature_artifacts()` for integrity checks
3. **Handle Errors Gracefully**: Implement proper error handling and fallbacks
4. **Monitor Resources**: Watch memory and CPU usage during development

### Production
1. **Set Appropriate Limits**: Configure resource limits based on your infrastructure
2. **Monitor Quality Gates**: Set up alerts for quality gate failures
3. **Regular Validation**: Periodically validate artifact integrity
4. **Backup Strategy**: Implement backup strategy for critical artifacts

### Troubleshooting
1. **Check Logs**: Review structured logs for detailed error information
2. **Validate Artifacts**: Use validation functions to check artifact integrity
3. **Resource Monitoring**: Monitor resource usage during problematic operations
4. **Force Regeneration**: Use `--force-rerun` to regenerate corrupted artifacts

## Integration with Pipeline

### Step Dependencies
```
Step 1 (Data Collection) 
    ↓
Step 2 (Feature Engineering) → Creates Artifacts
    ↓
Step 3 (HMM Regime Discovery) → Loads Artifacts
    ↓
Step 4+ (Subsequent Steps) → Load Artifacts as Needed
```

### Artifact Lifecycle
1. **Creation**: Step 2 creates artifacts after feature engineering
2. **Validation**: Automatic validation ensures integrity
3. **Loading**: Subsequent steps load artifacts instead of re-engineering
4. **Invalidation**: Hash changes trigger regeneration
5. **Cleanup**: Old artifacts can be cleaned up when no longer needed

## Future Enhancements

### Planned Features
- **Compression**: Automatic compression for large feature sets
- **Versioning**: Artifact versioning for experiment tracking
- **Distributed Caching**: Support for distributed artifact storage
- **Incremental Updates**: Support for incremental feature updates
- **Artifact Sharing**: Cross-experiment artifact sharing

### Performance Improvements
- **Parallel Loading**: Parallel loading of multiple artifact files
- **Memory Mapping**: Memory-mapped file access for large artifacts
- **Lazy Loading**: Lazy loading of feature subsets
- **Predictive Caching**: Predictive caching based on usage patterns
