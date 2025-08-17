# Step 2 Feature Generation and Usage Analysis

## Overview

Step 2 is responsible for generating comprehensive feature sets from labeled data, which are then used by subsequent steps for HMM regime discovery, regime splitting, and model training.

## Feature Generation Process (Step 2)

### 1. Input Data Sources

Step 2 processes labeled data from Step 1:
- **Train split**: `{exchange}_{symbol}_labeled_train.parquet`
- **Validation split**: `{exchange}_{symbol}_labeled_validation.parquet`
- **Test split**: `{exchange}_{symbol}_labeled_test.parquet`

### 2. Core Feature Engineering Components

The feature engineering uses `VectorizedAdvancedFeatureEngineering` class with the following components:

#### A. Market Microstructure Features
- **Order flow analysis**: Trade count, volume ratios, trade volume
- **Market depth indicators**: Bid/ask spreads, order book imbalances
- **Trade flow patterns**: Buy/sell pressure, trade size distributions

#### B. Volatility Modeling
- **Volatility regime detection**: Low/medium/high volatility states
- **GARCH-style models**: Conditional volatility estimation
- **Volatility clustering**: Regime-specific volatility patterns

#### C. Correlation Analysis
- **Cross-asset correlations**: Multi-timeframe correlation matrices
- **Autocorrelation features**: Price and volume autocorrelation
- **Correlation regime detection**: Market correlation states

#### D. Momentum Analysis
- **Price momentum**: Multiple timeframe momentum indicators
- **Volume momentum**: Volume-weighted momentum signals
- **Momentum regime detection**: Trend vs mean-reversion states

#### E. Liquidity Analysis
- **Liquidity indicators**: Bid-ask spreads, market depth
- **Liquidity regime detection**: High/low liquidity states
- **Liquidity stress indicators**: Market stress during low liquidity

#### F. Candlestick Patterns
- **Technical patterns**: Doji, hammer, engulfing patterns
- **Pattern recognition**: Automated pattern detection
- **Pattern strength scoring**: Confidence in pattern signals

#### G. OHLCV Price Features
- **Technical indicators**: RSI, MACD, Bollinger Bands
- **Price action features**: High-low spreads, open-close relationships
- **Volume-price relationships**: Volume-weighted indicators

#### H. Support/Resistance Distance Features
- **SR level proximity**: Distance to nearest support/resistance
- **SR strength indicators**: Strength of nearby levels
- **SR break detection**: Level break signals

#### I. Wavelet Transforms
- **Multi-scale analysis**: Different time horizon decompositions
- **Wavelet coefficients**: Frequency domain features
- **Wavelet-based volatility**: Scale-specific volatility measures

#### J. Multi-timeframe Features
- **Aggregated indicators**: Features across multiple timeframes
- **Timeframe relationships**: Cross-timeframe correlations
- **Regime consistency**: Regime alignment across timeframes

### 3. Feature Configuration

Default feature configuration includes:
```python
feature_config = {
    "vectorized_advanced_features": {
        "enable_difference_acceleration_features": True,
        "enable_volatility_modeling": True,
        "enable_correlation_analysis": True,
        "enable_momentum_analysis": True,
        "enable_liquidity_analysis": True,
        "enable_candlestick_patterns": True,
        "enable_sr_distance": True,
        "enable_wavelet_transforms": True,
        "enable_multi_timeframe": True,
        "enable_meta_labeling": True,
        "enable_explicit_meta_labels": True,
    }
}
```

### 4. HMM Composite Cluster Integration

Step 2 also integrates HMM composite clusters from Step 3 (if available):
- **Composite cluster IDs**: `hmm_composite_cluster_id`
- **Combination IDs**: `hmm_combination_id`
- **Intensity scores**: `intensity_cluster_0`, `intensity_cluster_1`, etc.

### 5. Output Artifacts

Step 2 generates the following artifacts:
- **`{exchange}_{symbol}_features_train.parquet`**: Training features
- **`{exchange}_{symbol}_features_validation.parquet`**: Validation features
- **`{exchange}_{symbol}_features_test.parquet`**: Test features
- **`{exchange}_{symbol}_features_metadata.json`**: Feature metadata
- **`{exchange}_{symbol}_features_hash.txt`**: Feature hash for caching

## Feature Usage Throughout Pipeline

### 1. Step 3: HMM Regime Discovery

**How features are used:**
- **Feature loading**: Uses `load_features_for_step()` to load Step 2 features
- **Block feature selection**: Features are organized into blocks for HMM training
- **HMM model training**: Each block trains separate HMM models
- **Composite clustering**: Block states are combined into composite clusters

**Key feature requirements:**
- **Diverse feature types**: Different blocks use different feature categories
- **Stationarity**: Features are preprocessed for HMM compatibility
- **Alignment**: Features must align with price data timestamps

**Block organization:**
```python
BLOCKS = [
    Block("price_action", 50),      # Price-based features
    Block("volume_flow", 30),       # Volume and flow features
    Block("volatility", 25),        # Volatility regime features
    Block("momentum", 25),          # Momentum indicators
    Block("liquidity", 20),         # Liquidity indicators
    Block("correlation", 20),       # Correlation features
    Block("patterns", 15),          # Candlestick patterns
    Block("sr_distance", 10),       # Support/resistance features
    Block("wavelet", 15),           # Wavelet transform features
]
```

### 2. Step 4: Regime Data Splitting

**How features are used:**
- **Unified data loading**: Combines features with labels and HMM clusters
- **Regime identification**: Uses `composite_cluster_id` from HMM analysis
- **Data splitting**: Splits data by HMM composite clusters
- **Regime-specific datasets**: Creates separate datasets for each regime

**Key feature requirements:**
- **HMM composite clusters**: Must have `composite_cluster_id` column
- **Feature completeness**: All features must be available for splitting
- **Temporal alignment**: Features must align with regime labels

### 3. Step 5: HMM-Based Training

**How features are used:**
- **Regime-specific training**: Each regime gets its own model
- **Feature selection**: Regime-specific feature importance
- **Model training**: Features are used to train regime-specific models
- **Ensemble creation**: Multiple models per regime

**Key feature requirements:**
- **Regime-specific features**: Features that work well in each regime
- **Feature quality**: High-quality, non-leaky features
- **Dimensionality**: Appropriate feature count for each regime

## Feature Quality and Validation

### 1. Data Quality Checks

- **Completeness**: Minimum 80% feature completeness
- **Consistency**: Feature value consistency across splits
- **Stationarity**: Features are tested for stationarity
- **Correlation**: Feature correlation analysis to avoid redundancy

### 2. Lookahead Bias Prevention

- **Temporal validation**: Features are validated for temporal consistency
- **Lagging**: Appropriate lagging is applied to prevent lookahead bias
- **Feature leakage detection**: Automated detection of data leakage

### 3. Performance Optimization

- **Vectorized operations**: All feature engineering uses vectorized operations
- **Caching**: Feature artifacts are cached to avoid regeneration
- **Parallel processing**: Multi-core processing for large datasets
- **Memory efficiency**: Streaming processing for large datasets

## Feature Artifact Management

### 1. Caching System

- **Hash-based invalidation**: Features are regenerated when input data changes
- **Artifact persistence**: Features are saved as parquet files
- **Metadata tracking**: Configuration and statistics are saved
- **Version control**: Feature versions are tracked for reproducibility

### 2. Loading System

- **Centralized loading**: `load_features_for_step()` provides unified access
- **Error handling**: Comprehensive error handling for missing artifacts
- **Performance monitoring**: Loading performance is monitored
- **Memory management**: Efficient memory usage during loading

### 3. Validation System

- **Artifact validation**: Comprehensive validation of feature artifacts
- **Quality gates**: Quality thresholds must be met
- **Performance thresholds**: Loading time and memory usage limits
- **Format validation**: Parquet format and structure validation

## Integration Points

### 1. With Step 1 (Data Collection)
- **Labeled data**: Step 2 processes labeled data from Step 1
- **Data quality**: Inherits data quality from Step 1
- **Temporal alignment**: Maintains temporal alignment with source data

### 2. With Step 3 (HMM Regime Discovery)
- **Feature provision**: Step 2 provides features to Step 3
- **HMM integration**: Step 2 integrates HMM clusters back into features
- **Block organization**: Features are organized into blocks for HMM training

### 3. With Step 4 (Regime Splitting)
- **Unified data**: Features are combined with labels and clusters
- **Regime identification**: Features help identify market regimes
- **Data splitting**: Features are split by regime for training

### 4. With Step 5 (Model Training)
- **Regime-specific features**: Each regime uses appropriate features
- **Feature selection**: Regime-specific feature importance
- **Model performance**: Feature quality directly impacts model performance

## Expected Outcomes

### 1. Feature Quality
- **Comprehensive coverage**: All major market aspects covered
- **High quality**: Features pass all quality checks
- **Non-leaky**: No lookahead bias in features
- **Stationary**: Features are suitable for ML models

### 2. Performance
- **Fast generation**: Vectorized operations ensure speed
- **Efficient storage**: Parquet format for efficient storage
- **Quick loading**: Optimized loading for downstream steps
- **Memory efficient**: Streaming processing for large datasets

### 3. Integration
- **Seamless flow**: Features flow smoothly through pipeline
- **Regime support**: Features support regime-based analysis
- **HMM compatibility**: Features work well with HMM models
- **Model training**: Features enable effective model training

The feature engineering system provides a comprehensive, high-quality feature set that supports the entire trading pipeline from regime discovery to model training.
