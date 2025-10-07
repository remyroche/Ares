# New Microstructure Features Added

## Overview
Added four new microstructure interaction features to the feature bank in `src/feature_generation/categories/microstructure.py`. These features capture important microstructure dynamics and their interactions with market conditions.

## Features Added

### 1. Corwin-Schultz Spread × Momentum (`cs_spread_momentum`)
- **Description**: Corwin-Schultz spread proxy × momentum interaction (wide spreads → trend breaks sooner)
- **Formula**: `(high - low) / close × close.pct_change(5)`
- **Purpose**: Captures how wide spreads interact with momentum, indicating when trends might break sooner
- **Parameters**: 
  - `spread_window`: 20 (default)
  - `momentum_period`: 5 (default)

### 2. Amihud Illiquidity × VWAP Distance (`amihud_illiquidity_vwap_distance`)
- **Description**: Amihud illiquidity × VWAP distance (big price move per $ volume → mean reversion risk)
- **Formula**: `(|returns| / volume).rolling(20).mean() × (close - vwap) / vwap`
- **Purpose**: Identifies mean reversion risk when there are large price moves per dollar volume
- **Parameters**:
  - `illiquidity_window`: 20 (default)
  - `vwap_window`: 20 (default)

### 3. Roll's λ × Realized Volatility Short (`roll_lambda_rv_short`)
- **Description**: Roll's λ (signed autocov) × rv_short (implicit spread/high trans. costs amplify vol impact)
- **Formula**: `-2 × cov(returns_t, returns_{t-1}) × returns.rolling(5).std()`
- **Purpose**: Shows how implicit spreads and high transaction costs amplify volatility impact
- **Parameters**:
  - `roll_window`: 20 (default)
  - `rv_window`: 5 (default)

### 4. Range/Volume Shock × Open30 (`range_volume_shock_open30`)
- **Description**: Range/Volume shock × open30 (thin-open shock filter)
- **Formula**: `((high-low)/volume).zscore() × open.pct_change(30)`
- **Purpose**: Filters for thin-open shocks using range/volume z-score interaction
- **Parameters**:
  - `range_volume_window`: 20 (default)
  - `open30_window`: 30 (default)

## Implementation Details

### Generator Classes
Each feature is implemented as a `VectorizedFeatureGenerator` subclass with:
- Proper configuration using `FeatureConfig`
- Optimization support for DataFrame processing
- Vectorization optimization enabled
- Matrix operations support

### Integration
- All features are automatically included in the feature bank via `create_default_microstructure_generators()`
- Features follow the existing microstructure category patterns
- Proper error handling and validation included
- Compatible with the existing feature generation pipeline

### Required Columns
- **cs_spread_momentum**: `["high", "low", "close"]`
- **amihud_illiquidity_vwap_distance**: `["high", "low", "close", "volume"]`
- **roll_lambda_rv_short**: `["close"]`
- **range_volume_shock_open30**: `["high", "low", "open", "volume"]`

## Usage
These features are automatically available when using the microstructure feature category in the feature bank. They can be generated individually or as part of the complete microstructure feature set.

## Mathematical Foundations
- **Corwin-Schultz Spread**: Uses high-low range as proxy for bid-ask spread
- **Amihud Illiquidity**: Standard measure of price impact per unit volume
- **Roll's λ**: Measures implicit bid-ask spread from return autocovariance
- **Range/Volume Z-score**: Standardized measure of price range relative to volume

All features include proper handling of edge cases, NaN values, and numerical stability.