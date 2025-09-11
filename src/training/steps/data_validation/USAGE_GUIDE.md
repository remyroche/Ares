# Data Validation Usage Guide

## Overview

This package now directly uses the existing data quality utilities from `src.utils.ml_common.data_quality` instead of duplicating functionality.

## How to Use

### Import the existing utilities directly:

```python
# For data quality validation
from src.utils.ml_common.data_quality import (
    DataQualityUtilities,
    detect_concept_drift,
    analyze_feature_stability,
    calculate_data_quality_score,
    enhanced_automated_data_cleaning
)

# Initialize data quality utilities
data_quality_utils = DataQualityUtilities(config)

# Use concept drift detection
drift_results = detect_concept_drift(df, window_size=100)

# Analyze feature stability
stability_results = analyze_feature_stability(df, window_size=50)

# Calculate overall data quality score
quality_score = calculate_data_quality_score(df)

# Enhanced data cleaning
cleaned_df = enhanced_automated_data_cleaning(df)
```

### Example Usage in Training Steps:

```python
from src.utils.ml_common.data_quality import DataQualityUtilities

class MyTrainingStep:
    def __init__(self, config):
        self.data_quality_utils = DataQualityUtilities(config)
    
    async def validate_data(self, df):
        # Use existing data quality utilities
        quality_score = self.data_quality_utils.calculate_quality_score(df)
        drift_results = self.data_quality_utils.detect_drift(df)
        
        return quality_score > 0.8 and not drift_results.get('drift_detected', False)
```

## Benefits

- ✅ **No Duplication**: Uses proven, existing utilities
- ✅ **Consistency**: Same data quality logic across all modules
- ✅ **Maintainability**: Changes to data quality logic benefit all modules
- ✅ **Reliability**: Leverages tested, optimized implementations