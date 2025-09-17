# Feature Engineering Usage Guide

## Overview

For advanced feature engineering, use the existing utilities from `src.feature_engineering` instead of duplicating functionality.

## How to Use

### Import the existing feature engineering components:

```python
# For advanced feature engineering
from src.feature_generation.utils.step06_enhanced_feature_engineering import (
    EnhancedFeatureEngineeringStep,
    FeatureEngineeringConfig
)
from src.feature_generation.utils.step06_utility_container import (
    Step06UtilityContainer,
    get_utility_container
)
from src.feature_generation.utils.math_validation import (
    safe_divide,
    safe_log,
    safe_sqrt,
    validate_positive
)

# Initialize feature engineering step
feature_config = FeatureEngineeringConfig()
feature_engineer = EnhancedFeatureEngineeringStep(config)

# Use utility container for math operations
utility_container = get_utility_container()

# Engineer features
result = await feature_engineer.engineer_features(data)
```

### Example Usage in ML Modules:

```python
from src.feature_generation.utils.step06_enhanced_feature_engineering import EnhancedFeatureEngineeringStep

class MyMLModule:
    def __init__(self, config):
        self.feature_engineer = EnhancedFeatureEngineeringStep(config)
    
    async def process_data(self, df):
        # Use existing feature engineering
        enhanced_df = await self.feature_engineer.engineer_features(df)
        return enhanced_df
```

### For Technical Indicators:

```python
# Use existing math validation utilities
from src.feature_generation.utils.math_validation import safe_divide, safe_log

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = pd.Series(gains).rolling(window=period).mean().values
    avg_losses = pd.Series(losses).rolling(window=period).mean().values
    
    # Use safe math operations
    rs = safe_divide(avg_gains, avg_losses)
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

## Benefits

- ✅ **No Duplication**: Uses proven, existing feature engineering components
- ✅ **Consistency**: Same feature engineering logic across all modules
- ✅ **Maintainability**: Changes to feature engineering benefit all modules
- ✅ **Reliability**: Leverages tested, optimized implementations
- ✅ **Math Safety**: Uses safe mathematical operations with validation