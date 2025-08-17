# Lookback Period Single Source of Truth Implementation

## Overview

This document describes the implementation of a single source of truth for lookback periods throughout the Ares training pipeline. The lookback period is now centrally managed and consistently passed from `ares_launcher.py` through the enhanced training manager to all training steps.

## Lookback Period Constants

The lookback periods are defined in `src/config/constants.py`:

```python
# Data configuration constants
DEFAULT_LOOKBACK_DAYS = 180  # Exactly 6 months for consistent data range
FULL_TRAINING_LOOKBACK_DAYS = 730  # 2 years for full training (updated from 3 years)
SHORT_BLANK_LOOKBACK_DAYS = 30  # 30 days for short blank training
BLANK_TRAINING_LOOKBACK_DAYS = 180  # 180 days for blank training
```

## Training Mode Mapping

- **Full Training**: 730 days (2 years)
- **Blank Training**: 180 days (6 months)  
- **Short Blank Training**: 30 days (1 month)

## Implementation Flow

### 1. Ares Launcher (`ares_launcher.py`)

The launcher determines the training mode and sets the appropriate lookback period:

```python
# Full training
lookback_days=FULL_TRAINING_LOOKBACK_DAYS,  # 2 years for full training

# Blank training  
lookback_days=BLANK_TRAINING_LOOKBACK_DAYS,  # 180 days for blank training

# Short blank training
lookback_days=SHORT_BLANK_LOOKBACK_DAYS,  # 30 days for short blank training
```

### 2. Enhanced Training Manager (`src/training/enhanced_training_manager.py`)

The enhanced training manager receives the lookback_days from the launcher and uses it directly:

```python
# Set lookback days based on training mode
from src.config.constants import (
    FULL_TRAINING_LOOKBACK_DAYS,
    BLANK_TRAINING_LOOKBACK_DAYS,
    SHORT_BLANK_LOOKBACK_DAYS,
)
# Use the lookback_days from config if provided, otherwise use defaults based on mode
if "lookback_days" in self.enhanced_training_config:
    self.lookback_days: int = self.enhanced_training_config["lookback_days"]
else:
    # Fallback defaults: 30 days for short blank, 180 days for blank mode, 730 days for full mode
    # Note: Short blank mode is handled by launcher passing explicit lookback_days=30
    default_lookback = BLANK_TRAINING_LOOKBACK_DAYS if self.blank_training_mode else FULL_TRAINING_LOOKBACK_DAYS
    self.lookback_days: int = default_lookback
```

**Important**: The short-blank training mode is handled by the launcher explicitly passing `lookback_days=30` to the enhanced training manager, which then uses this value directly.

### 3. Training Steps

All training steps now receive the `lookback_days` parameter from the enhanced training manager:

```python
# Example: Step 1_7 HMM Regime Discovery
step1_7_success = await _step1_7.run_step(
    symbol=symbol,
    exchange=exchange,
    data_dir=data_dir,
    timeframe=timeframe,
    lookback_days=self.lookback_days,  # Passed from enhanced training manager
    force_rerun=self.force_rerun,
)
```

## Updated Training Steps

The following training steps have been updated to properly handle lookback_days:

1. **Step 1_7**: HMM Regime Discovery
2. **Step 1_8**: Regime Forecasting  
3. **Step 2**: Processing, Labeling, Feature Engineering
4. **Step 3**: Feature Engineering
5. **Step 4**: Regime Data Splitting
6. **Step 5**: HMM-Based Training
7. **Step 6**: Analyst Enhancement
8. **Step 8**: Tactician Labeling
9. **Step 9**: Tactician Specialist Training
10. **Step 9.5**: HMM-LM Generalist Training
11. **Step 11**: Confidence Calibration

## Key Benefits

1. **Single Source of Truth**: Lookback periods are centrally managed in constants
2. **Training Mode Awareness**: Different training modes use appropriate lookback periods
3. **Consistent Propagation**: Lookback_days is passed through the entire pipeline
4. **Configurable**: Easy to modify lookback periods by changing constants
5. **Fallback Safety**: Sensible defaults if lookback_days is not specified

## Usage Examples

### Full Training (2 years)
```bash
python ares_launcher.py regime train --symbol ETHUSDT --exchange BINANCE --mode full
# Uses FULL_TRAINING_LOOKBACK_DAYS = 730 days
```

### Blank Training (6 months)
```bash
python ares_launcher.py regime train --symbol ETHUSDT --exchange BINANCE --mode blank
# Uses BLANK_TRAINING_LOOKBACK_DAYS = 180 days
```

### Short Blank Training (1 month)
```bash
python ares_launcher.py regime train --symbol ETHUSDT --exchange BINANCE --mode short-blank
# Uses SHORT_BLANK_LOOKBACK_DAYS = 30 days
```

## Implementation Notes

- All training steps now import both `BLANK_TRAINING_LOOKBACK_DAYS` and `FULL_TRAINING_LOOKBACK_DAYS` constants
- Steps use the `lookback_days` parameter passed from the enhanced training manager
- Fallback to `BLANK_TRAINING_LOOKBACK_DAYS` if not specified (for backward compatibility)
- The enhanced training manager correctly determines the lookback period based on training mode
- No hardcoded lookback values remain in the training steps

This implementation ensures that the lookback period is consistently managed throughout the entire training pipeline, with the appropriate period used based on the training mode selected by the user.
