# Unified Database Usage Guide

## Overview

The unified database system provides a single, efficient data management solution for the Ares training pipeline. It creates time-based splits with proper temporal ordering and enables easy data access across all training steps.

## Key Features

### 1. **Time-Based Data Splitting**
- **80% Training Data**: Used for model training (first 80% chronologically)
- **10% Validation Data**: Used for hyperparameter optimization and ensemble weight optimization
- **10% Test Data**: Used for final testing and Monte Carlo validation

### 2. **Lookback Period Integration**
- Respects the `lookback_days` configuration parameter
- Filters data to include only the specified time period
- Ensures consistent data scope across all training steps

### 3. **Unified Access Interface**
- Single point of access for all training data
- Consistent API across all training steps
- Built-in validation and integrity checks

## Implementation Details

### Created by Step 4 (Analyst Labeling & Feature Engineering)

Step 4 now creates the unified database at the end of its execution:

```python
# Step 5: Create Unified Database with Time-based Splits
from src.training.data_manager import UnifiedDataManager

data_manager = UnifiedDataManager(
    data_dir=data_dir,
    symbol=symbol,
    exchange=exchange,
    lookback_days=lookback_days
)

# Create unified database with proper time-based splits
database_results = data_manager.create_unified_database(
    labeled_data=data_with_autoencoder_features,
    strategic_signals=strategic_signals,
    train_ratio=0.8,  # 80% for training
    validation_ratio=0.1,  # 10% for validation and hyperparameter optimization
    test_ratio=0.1   # 10% for final testing and Monte Carlo validation
)
```

### Database Structure

The unified database creates the following files:

```
data/training/
├── unified_database/
│   ├── {EXCHANGE}_{SYMBOL}_unified_dataset.pkl     # Complete dataset
│   ├── {EXCHANGE}_{SYMBOL}_dataset_metadata.json  # Metadata and split info
│   └── {EXCHANGE}_{SYMBOL}_strategic_signals.pkl  # Strategic signals
├── {EXCHANGE}_{SYMBOL}_train_data.pkl              # Training split
├── {EXCHANGE}_{SYMBOL}_validation_data.pkl         # Validation split
└── {EXCHANGE}_{SYMBOL}_test_data.pkl               # Test split
```

## Usage in Training Steps

### Accessing Data

Use the `data_access_utils` module for easy data access:

```python
from src.training.data_access_utils import load_training_data, load_validation_data_for_optimization

# Load training data
X_train, y_train = load_training_data(data_dir, symbol, exchange, "train")

# Load validation data for optimization
X_val, y_val = load_validation_data_for_optimization(data_dir, symbol, exchange)

# Load test data
X_test, y_test = load_training_data(data_dir, symbol, exchange, "test")
```

### Step 8 Integration (Tactician Labeling)

Step 8 has been updated to:
1. Load data from the unified database (with fallback to legacy)
2. Update the unified database with enhanced tactician features
3. Maintain backward compatibility

```python
# Load data using the unified data manager (preferred method)
from src.training.data_manager import UnifiedDataManager

data_manager = UnifiedDataManager(data_dir=data_dir, symbol=symbol, exchange=exchange)

try:
    # Try to load from unified database first
    labeled_data = data_manager.load_data_split('full')
    logger.info("✅ Loaded data from unified database")
except FileNotFoundError:
    # Fallback to legacy method
    # ... legacy loading code ...
```

### Step 10 Integration (Tactician Ensemble Creation)

Step 10 now uses the unified validation data:

```python
async def _load_validation_data(self, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    # Try unified data manager first (preferred method)
    from src.training.data_manager import UnifiedDataManager
    
    data_manager = UnifiedDataManager(data_dir=data_dir, symbol=symbol, exchange=exchange)
    
    try:
        # Load validation split from unified database
        X_val, y_val = data_manager.get_features_and_labels('validation', 'tactician_label')
        return X_val.fillna(0).values, y_val.fillna(0).values
    except (FileNotFoundError, ValueError):
        # Fall back to legacy methods
        return await self._load_validation_data_legacy(data_dir)
```

## Time-Based Split Details

### Temporal Ordering

The splits maintain strict temporal ordering:
- **Training Period**: Earliest 80% of the data chronologically
- **Validation Period**: Next 10% of the data chronologically  
- **Test Period**: Latest 10% of the data chronologically

This ensures:
- No data leakage from future to past
- Realistic simulation of live trading conditions
- Proper validation of model performance over time

### Lookback Period Integration

The system respects the `lookback_days` parameter:

```python
# Default lookback periods:
# - Blank training: 30 days
# - Production training: 730 days (2 years)

# Example with 730 days:
# If current date is 2025-01-01
# Lookback period: 2023-01-01 to 2025-01-01
# Training: 2023-01-01 to 2024-07-12 (80%)
# Validation: 2024-07-12 to 2024-10-21 (10%) 
# Test: 2024-10-21 to 2025-01-01 (10%)
```

## Validation and Integrity Checks

The system includes comprehensive validation:

```python
from src.training.data_access_utils import validate_dataset_integrity, ensure_temporal_consistency

# Check database integrity
validation_results = validate_dataset_integrity(data_dir, symbol, exchange)
print(f"Status: {validation_results['status']}")

# Ensure temporal consistency
is_consistent = ensure_temporal_consistency(data_dir, symbol, exchange)
print(f"Temporal consistency: {is_consistent}")
```

## Metadata and Monitoring

The system provides rich metadata:

```python
from src.training.data_access_utils import get_dataset_metadata, get_time_splits_info

# Get complete metadata
metadata = get_dataset_metadata(data_dir, symbol, exchange)

# Get split information
splits_info = get_time_splits_info(data_dir, symbol, exchange)
print(f"Training samples: {splits_info['train']['samples']}")
print(f"Training period: {splits_info['train']['start_date']} to {splits_info['train']['end_date']}")
```

## Benefits

### 1. **Consistency**
- All steps use the same data splits
- Temporal ordering maintained across pipeline
- Lookback period consistently applied

### 2. **Efficiency**
- Single data loading point
- Optimized file storage
- Reduced memory usage

### 3. **Maintainability**
- Centralized data management
- Easy to update and modify
- Clear separation of concerns

### 4. **Reliability**
- Built-in validation checks
- Integrity verification
- Error handling and fallbacks

### 5. **Accessibility**
- Simple API for data access
- Utility functions for common tasks
- Backward compatibility maintained

## Migration Notes

### Existing Steps

- **Step 4**: Automatically creates unified database (no migration needed)
- **Step 8**: Uses unified database with fallback to legacy (seamless migration)
- **Step 10**: Enhanced to use unified validation data (improved performance)

### Legacy Compatibility

The system maintains full backward compatibility:
- Legacy file formats still supported
- Automatic fallback mechanisms
- Gradual migration possible

## Best Practices

### For New Steps

1. Always try unified database first
2. Include fallback to legacy methods
3. Update database when modifying data
4. Validate data integrity after changes

### For Existing Steps

1. Add unified database support gradually
2. Maintain legacy fallback paths
3. Test thoroughly before deployment
4. Monitor performance improvements

## Example: Complete Usage Pattern

```python
from src.training.data_access_utils import (
    check_unified_database_exists,
    load_training_data,
    update_dataset_with_new_features,
    validate_dataset_integrity
)

def my_training_step(data_dir, symbol, exchange):
    # Check if unified database exists
    if not check_unified_database_exists(data_dir, symbol, exchange):
        raise FileNotFoundError("Unified database not found. Run step 4 first.")
    
    # Load training data
    X_train, y_train = load_training_data(data_dir, symbol, exchange, "train")
    
    # Process data and create new features
    enhanced_data = enhance_features(X_train, y_train)
    
    # Update database with new features
    update_dataset_with_new_features(data_dir, enhanced_data, "train", symbol, exchange)
    
    # Validate integrity
    validation = validate_dataset_integrity(data_dir, symbol, exchange)
    if validation["status"] != "SUCCESS":
        raise ValueError(f"Database validation failed: {validation['issues']}")
    
    return enhanced_data
```

This unified database system provides a robust foundation for the entire training pipeline, ensuring data consistency, temporal integrity, and efficient access patterns across all training steps.