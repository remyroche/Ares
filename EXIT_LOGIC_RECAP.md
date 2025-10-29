# Exit Logic Recap

## Overview

The exit logic matches the parameters tested by `final_parameters_optimization`, ensuring consistency between backtesting optimization and live trading. Exit conditions are checked on every signal generation when a position is open.

## Exit Flow

### Step 1: Calculate Exit Confidence
**Location:** `_calculate_exit_confidence()`

Combines analyst and tactician confidence using one of three methods (from optimization):
- **Multiplicative**: `(tactician_conf^tactician_weight) * (analyst_conf^analyst_weight)`
- **Logarithmic**: `exp(tactician_weight * log(tactician_conf) + analyst_weight * log(analyst_conf))`
- **Weighted Average**: `analyst_conf * analyst_weight + tactician_conf * tactician_weight`

**Default weights:** tactician=0.6, analyst=0.4 (overridden by final_parameters_optimization)

### Step 2: Check Exit Conditions
**Location:** `_check_exit_conditions()`

Six exit conditions are checked (in order):

#### 1. **Exit Confidence Threshold** (Primary)
- Checks if combined exit confidence < `exit_confidence_threshold`
- Default: 0.5, overridden by optimization
- **Exit if:** `exit_confidence < exit_threshold`

#### 2. **Tiered Confidence Thresholds**
Checks tiered confidence levels (from `exit_strategy`):
- **Very Low**: `exit_confidence <= confidence_very_low` (default: 0.2)
- **Low**: `exit_confidence <= confidence_low` (default: 0.4)
- **Medium**: `exit_confidence <= confidence_medium` AND `confidence_drop > 0.2` (default: 0.6)
- **Confidence Drop**: `confidence_drop >= exit_confidence_drop` threshold

**Confidence Drop** = `entry_confidence - exit_confidence`

#### 3. **Time-Based Exit**
- **Max Hold Time**: Exit if `elapsed_time >= max_hold_time` (default: 10800s = 3 hours)
- **Min Hold Time**: Suppress non-critical exits if `elapsed_time < min_hold_time` (default: 300s = 5 min)
  - Exception: Stop-loss triggers still exit even if position is new

#### 4. **Profit-Taking Conditions**
- **Base Profit Target**: Exit if `profit_pct >= base_profit_target` AND `exit_confidence >= min_confidence_for_profit`
  - Default: `base_profit_target = 0.04` (4%), `min_confidence_for_profit = 0.6`
- **Profit Tier 3**: Exit if `profit_pct >= profit_tier_3 * base_profit_target`
  - Default: `profit_tier_3 = 0.75` (75% of base target)

**Profit Calculation:**
- Long: `(current_price - entry_price) / entry_price`
- Short: `(entry_price - current_price) / entry_price`

#### 5. **Stop-Loss Conditions**
- **Base Stop Loss**: Exit if `loss_pct >= base_stop_loss`
  - Default: `base_stop_loss = 0.05` (5% loss)

**Loss Calculation:**
- Long: `(entry_price - current_price) / entry_price`
- Short: `(current_price - entry_price) / entry_price`

#### 6. **Individual Component Confidence Drops**
- Exit if **analyst** OR **tactician** confidence drops > 0.3 from entry confidence
- Tracks which component dropped more significantly

## Exit Decision Logic

### Multiple Conditions
- All conditions are checked and reasons collected
- If **any** condition triggers → exit
- Exit reason includes all triggered conditions

### Priority
Exit reasons are prioritized:
1. Stop-loss (always exits, even if position is new)
2. Profit-taking (if targets reached)
3. Confidence thresholds
4. Time-based
5. Component confidence drops

### Exit Signal Generation
When exit is triggered:
- Signal: `'close'`
- Confidence: `1.0` (high confidence for exit)
- Strength: `1.0`
- Reason: Includes all triggered exit conditions

## Parameter Sources

### Default Constants (Fallback)
```python
DEFAULT_EXIT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_REGIME_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
```

### Optimization Parameters (Preferred)
Loaded from `final_parameters_optimization` step:
- **Primary location**: `exit_strategy.best_params` (raw format)
- **Alternative**: `position_monitor_exit_strategy` (formatted format)
- **Fallback**: Top-level keys like `exit_confidence_threshold`, `base_profit_target`, etc.

### Parameter Format Support

**Raw Format** (flat keys):
```python
{
    'exit_confidence_threshold': 0.5,
    'confidence_very_low': 0.2,
    'confidence_low': 0.4,
    'base_profit_target': 0.04,
    'base_stop_loss': -0.05,
    'max_hold_time': 10800,
    'exit_confidence_drop': 0.2
}
```

**Formatted Format** (nested structure):
```python
{
    'confidence_thresholds': {
        'very_low': 0.2,
        'low': 0.4,
        'medium': 0.6,
        'high': 0.8
    },
    'profit_taking': {
        'base_profit_target': 0.04,
        'min_confidence_for_profit': 0.6,
        'scaling_levels': [0.25, 0.5, 0.75]
    },
    'stop_loss': {
        'base_stop_loss': -0.05
    },
    'time_based': {
        'max_hold_time': 10800,
        'min_hold_time': 300
    }
}
```

## Example Exit Scenarios

### Scenario 1: Confidence Drop Exit
- Entry confidence: 0.85
- Current exit confidence: 0.45
- Exit threshold: 0.5
- **Result**: Exit triggered - "Exit confidence 0.450 below threshold 0.500"

### Scenario 2: Profit Target Exit
- Entry price: $100
- Current price: $104.5 (long position)
- Profit: 4.5%
- Base profit target: 4.0%
- Exit confidence: 0.65
- Min confidence for profit: 0.6
- **Result**: Exit triggered - "Profit target reached: 0.045 >= 0.040 with confidence 0.650"

### Scenario 3: Stop-Loss Exit
- Entry price: $100
- Current price: $94.5 (long position)
- Loss: 5.5%
- Base stop loss: 5.0%
- **Result**: Exit triggered - "Stop-loss triggered: 0.055 >= 0.050"

### Scenario 4: Time-Based Exit
- Entry time: 10:00 AM
- Current time: 1:30 PM
- Elapsed: 3.5 hours (12600s)
- Max hold time: 3 hours (10800s)
- **Result**: Exit triggered - "Maximum hold time exceeded: 12600s >= 10800s"

### Scenario 5: Multiple Conditions
- Exit confidence: 0.45 (< 0.5 threshold)
- Profit: 4.2% (>= 4.0% target)
- **Result**: Exit triggered - "Exit confidence 0.450 below threshold 0.500 (and 1 other condition(s))"

## Thread Safety

- Position state access is protected by `_position_lock`
- Ensures concurrent signal generation doesn't corrupt position state
- Exit checks read position state atomically

## Integration Points

1. **Called during signal generation**: Every `generate_signal()` call checks exit conditions
2. **Position state tracking**: Entry confidence, entry price, entry timestamp stored in `PositionState`
3. **Market data required**: Current price from `market_data['close']` needed for profit/loss calculations
4. **Optimization alignment**: Parameters match exactly what `final_parameters_optimization` tests

## Key Features

✅ **Comprehensive**: Checks 6 different exit condition types
✅ **Optimized**: Uses parameters from final_parameters_optimization
✅ **Flexible**: Supports both raw and formatted parameter formats
✅ **Safe**: Stop-loss always triggers, even for new positions
✅ **Smart**: Suppresses non-critical exits for positions too new
✅ **Informative**: Provides detailed exit reasons for debugging
✅ **Thread-safe**: Protected position state access
