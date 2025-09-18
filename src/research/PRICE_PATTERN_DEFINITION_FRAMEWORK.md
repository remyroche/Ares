# Price Pattern Definition Framework for ML Applications

## 🎯 Relationship to Existing Research Structure

### Current `src/research/clusters/` Focus:
- **Regime Discovery**: Finding market behavioral patterns through clustering
- **Dimension Analysis**: Analyzing implicit market dimensions (volatility, momentum, etc.)
- **Economic Validation**: Determining if regimes justify separate ML models

### New Research Framework Enhancement:
- **Pattern Definition**: Precise mathematical definitions of what constitutes a "price pattern"
- **Economic Relevance**: Which dimensions actually cause/predict these patterns
- **ML Applicability**: How to use patterns for supervised learning targets

## 🔬 Integration Architecture

```
Existing Pipeline Enhancement:
┌─────────────────────────────────────────────────────────────────┐
│ src/research/clusters/ (EXISTING)                               │
│ ├── Dimension Discovery (dimension_analyzer.py)                 │
│ ├── Economic Relevance (dimension_economic_relevance.py)        │
│ └── Regime Clustering (regime_clusterer.py)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ NEW: Price Pattern Definition Framework                         │
│ ├── Mathematical Pattern Definitions                            │
│ ├── Pattern-Dimension Causality Analysis                        │
│ ├── ML Target Generation                                        │
│ └── Pattern Predictability Assessment                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Enhanced ML Training                                            │
│ ├── Pattern-Based Supervised Learning                          │
│ ├── Dimension-Filtered Feature Selection                       │
│ └── Pattern-Specific Model Architecture                        │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Mathematical Definition of Price Patterns

### Core Philosophy:
**A price pattern is a mathematically measurable sequence of price movements that:**
1. **Has predictable structure** (not random noise)
2. **Occurs with sufficient frequency** for ML training
3. **Can be influenced by market dimensions** (volatility, momentum, etc.)
4. **Has economic significance** for trading decisions

### Pattern Categories for ML:

#### 1. **Momentum Patterns**
```python
class MomentumPattern:
    """Mathematical definition of momentum patterns."""
    
    @staticmethod
    def momentum_persistence(prices: pd.Series, window: int = 10) -> pd.Series:
        """
        Pattern: Momentum continues for at least 'window' periods
        
        Definition: If momentum(t) > threshold, then momentum(t+1:t+window) 
                   maintains same direction with magnitude > decay_threshold
        """
        returns = prices.pct_change()
        momentum = returns.rolling(5).mean()
        
        persistence_labels = []
        for i in range(len(momentum) - window):
            current_momentum = momentum.iloc[i]
            
            if abs(current_momentum) > 0.005:  # Minimum momentum threshold
                future_momentum = momentum.iloc[i+1:i+window+1]
                
                # Check if momentum persists (same direction, >80% of periods)
                same_direction = (np.sign(future_momentum) == np.sign(current_momentum))
                persistence_rate = same_direction.sum() / len(future_momentum)
                
                # Check if magnitude decays gradually (not abruptly)
                magnitude_decay = abs(future_momentum) / abs(current_momentum)
                gradual_decay = (magnitude_decay > 0.5).sum() / len(magnitude_decay)
                
                # Pattern exists if both conditions met
                pattern_exists = (persistence_rate >= 0.8) and (gradual_decay >= 0.6)
                persistence_labels.append(1 if pattern_exists else 0)
            else:
                persistence_labels.append(0)
        
        return pd.Series(persistence_labels, index=prices.index[:len(persistence_labels)])
    
    @staticmethod
    def momentum_acceleration(prices: pd.Series, window: int = 5) -> pd.Series:
        """
        Pattern: Momentum accelerates (increases in magnitude)
        
        Definition: momentum(t+1:t+window) > momentum(t) * acceleration_factor
        """
        returns = prices.pct_change()
        momentum = returns.rolling(5).mean()
        
        acceleration_labels = []
        for i in range(len(momentum) - window):
            current_momentum = abs(momentum.iloc[i])
            
            if current_momentum > 0.003:  # Minimum base momentum
                future_momentum = abs(momentum.iloc[i+1:i+window+1])
                
                # Check for acceleration (increasing magnitude)
                acceleration_count = (future_momentum > current_momentum * 1.2).sum()
                acceleration_rate = acceleration_count / len(future_momentum)
                
                pattern_exists = acceleration_rate >= 0.6  # 60% of periods show acceleration
                acceleration_labels.append(1 if pattern_exists else 0)
            else:
                acceleration_labels.append(0)
        
        return pd.Series(acceleration_labels, index=prices.index[:len(acceleration_labels)])
```

#### 2. **Mean Reversion Patterns**
```python
class MeanReversionPattern:
    """Mathematical definition of mean reversion patterns."""
    
    @staticmethod
    def reversion_speed(prices: pd.Series, ma_window: int = 20, reversion_window: int = 10) -> pd.Series:
        """
        Pattern: Price reverts to mean within specific timeframe
        
        Definition: If |price(t) - MA(t)| > threshold, then price(t+reversion_window) 
                   closer to MA(t) than price(t)
        """
        ma = prices.rolling(ma_window).mean()
        deviation = (prices - ma) / ma
        
        reversion_labels = []
        reversion_speeds = []
        
        for i in range(ma_window, len(prices) - reversion_window):
            current_deviation = deviation.iloc[i]
            
            if abs(current_deviation) > 0.02:  # 2% deviation threshold
                current_price = prices.iloc[i]
                target_ma = ma.iloc[i]
                
                # Look for reversion in next reversion_window periods
                future_prices = prices.iloc[i+1:i+reversion_window+1]
                
                # Calculate reversion progression
                reversion_occurred = False
                reversion_speed = 0
                
                for j, future_price in enumerate(future_prices):
                    current_distance = abs(current_price - target_ma)
                    future_distance = abs(future_price - target_ma)
                    
                    if future_distance < current_distance * 0.5:  # 50% closer to mean
                        reversion_occurred = True
                        reversion_speed = current_distance / (j + 1)  # Speed = distance/time
                        break
                
                reversion_labels.append(1 if reversion_occurred else 0)
                reversion_speeds.append(reversion_speed)
            else:
                reversion_labels.append(0)
                reversion_speeds.append(0)
        
        return pd.Series(reversion_labels, index=prices.index[ma_window:ma_window+len(reversion_labels)])
    
    @staticmethod
    def oversold_bounce(prices: pd.Series, rsi_window: int = 14, bounce_window: int = 5) -> pd.Series:
        """
        Pattern: Oversold conditions lead to price bounce
        
        Definition: RSI(t) < oversold_threshold and price(t+1:t+bounce_window) > price(t)
        """
        # Calculate RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        bounce_labels = []
        
        for i in range(rsi_window, len(prices) - bounce_window):
            current_rsi = rsi.iloc[i]
            current_price = prices.iloc[i]
            
            if current_rsi < 30:  # Oversold threshold
                future_prices = prices.iloc[i+1:i+bounce_window+1]
                
                # Check for bounce (price increases)
                bounce_count = (future_prices > current_price).sum()
                bounce_rate = bounce_count / len(future_prices)
                
                # Check for meaningful bounce magnitude
                max_future_price = future_prices.max()
                bounce_magnitude = (max_future_price - current_price) / current_price
                
                pattern_exists = (bounce_rate >= 0.6) and (bounce_magnitude > 0.01)  # 1% minimum bounce
                bounce_labels.append(1 if pattern_exists else 0)
            else:
                bounce_labels.append(0)
        
        return pd.Series(bounce_labels, index=prices.index[rsi_window:rsi_window+len(bounce_labels)])
```

#### 3. **Volatility Patterns**
```python
class VolatilityPattern:
    """Mathematical definition of volatility patterns."""
    
    @staticmethod
    def volatility_expansion(prices: pd.Series, vol_window: int = 20, expansion_window: int = 10) -> pd.Series:
        """
        Pattern: Low volatility followed by high volatility
        
        Definition: vol(t) in bottom percentile, then vol(t+1:t+expansion_window) 
                   in top percentile
        """
        returns = prices.pct_change()
        volatility = returns.rolling(vol_window).std()
        vol_percentile = volatility.rolling(100).rank(pct=True)
        
        expansion_labels = []
        
        for i in range(100, len(volatility) - expansion_window):
            current_vol_percentile = vol_percentile.iloc[i]
            
            if current_vol_percentile < 0.2:  # Bottom 20% volatility
                future_vol_percentiles = vol_percentile.iloc[i+1:i+expansion_window+1]
                
                # Check for volatility expansion
                high_vol_periods = (future_vol_percentiles > 0.8).sum()  # Top 20%
                expansion_rate = high_vol_periods / len(future_vol_percentiles)
                
                pattern_exists = expansion_rate >= 0.3  # 30% of future periods high vol
                expansion_labels.append(1 if pattern_exists else 0)
            else:
                expansion_labels.append(0)
        
        return pd.Series(expansion_labels, index=volatility.index[100:100+len(expansion_labels)])
    
    @staticmethod
    def volatility_clustering(prices: pd.Series, vol_window: int = 20, cluster_window: int = 5) -> pd.Series:
        """
        Pattern: High volatility periods cluster together
        
        Definition: If vol(t) > high_threshold, then vol(t+1:t+cluster_window) 
                   also > high_threshold
        """
        returns = prices.pct_change()
        volatility = returns.rolling(vol_window).std()
        vol_percentile = volatility.rolling(100).rank(pct=True)
        
        clustering_labels = []
        
        for i in range(100, len(volatility) - cluster_window):
            current_vol_percentile = vol_percentile.iloc[i]
            
            if current_vol_percentile > 0.8:  # High volatility threshold
                future_vol_percentiles = vol_percentile.iloc[i+1:i+cluster_window+1]
                
                # Check for volatility clustering
                high_vol_continuation = (future_vol_percentiles > 0.6).sum()  # Continued elevated vol
                clustering_rate = high_vol_continuation / len(future_vol_percentiles)
                
                pattern_exists = clustering_rate >= 0.7  # 70% continuation
                clustering_labels.append(1 if pattern_exists else 0)
            else:
                clustering_labels.append(0)
        
        return pd.Series(clustering_labels, index=volatility.index[100:100+len(clustering_labels)])
```

#### 4. **Breakout Patterns**
```python
class BreakoutPattern:
    """Mathematical definition of breakout patterns."""
    
    @staticmethod
    def confirmed_breakout(prices: pd.Series, bb_window: int = 20, confirmation_window: int = 5) -> pd.Series:
        """
        Pattern: Price breaks technical level and continues in breakout direction
        
        Definition: price(t) breaks Bollinger Band, then price(t+1:t+confirmation_window) 
                   continues beyond breakout level
        """
        # Calculate Bollinger Bands
        ma = prices.rolling(bb_window).mean()
        std = prices.rolling(bb_window).std()
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        
        breakout_labels = []
        
        for i in range(bb_window, len(prices) - confirmation_window):
            current_price = prices.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            
            # Check for breakout
            upper_breakout = current_price > current_upper
            lower_breakout = current_price < current_lower
            
            if upper_breakout or lower_breakout:
                future_prices = prices.iloc[i+1:i+confirmation_window+1]
                
                if upper_breakout:
                    # Confirm upward breakout
                    confirmation_count = (future_prices > current_upper).sum()
                    confirmation_rate = confirmation_count / len(future_prices)
                    
                    # Check for meaningful continuation
                    max_future = future_prices.max()
                    continuation_magnitude = (max_future - current_price) / current_price
                    
                    pattern_exists = (confirmation_rate >= 0.6) and (continuation_magnitude > 0.01)
                    
                elif lower_breakout:
                    # Confirm downward breakout
                    confirmation_count = (future_prices < current_lower).sum()
                    confirmation_rate = confirmation_count / len(future_prices)
                    
                    # Check for meaningful continuation
                    min_future = future_prices.min()
                    continuation_magnitude = (current_price - min_future) / current_price
                    
                    pattern_exists = (confirmation_rate >= 0.6) and (continuation_magnitude > 0.01)
                
                breakout_labels.append(1 if pattern_exists else 0)
            else:
                breakout_labels.append(0)
        
        return pd.Series(breakout_labels, index=prices.index[bb_window:bb_window+len(breakout_labels)])
    
    @staticmethod
    def false_breakout(prices: pd.Series, bb_window: int = 20, reversal_window: int = 3) -> pd.Series:
        """
        Pattern: Price breaks technical level but quickly reverses
        
        Definition: price(t) breaks level, but price(t+1:t+reversal_window) 
                   returns inside original range
        """
        # Calculate Bollinger Bands
        ma = prices.rolling(bb_window).mean()
        std = prices.rolling(bb_window).std()
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        
        false_breakout_labels = []
        
        for i in range(bb_window, len(prices) - reversal_window):
            current_price = prices.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            
            # Check for initial breakout
            upper_breakout = current_price > current_upper
            lower_breakout = current_price < current_lower
            
            if upper_breakout or lower_breakout:
                future_prices = prices.iloc[i+1:i+reversal_window+1]
                
                if upper_breakout:
                    # Check for reversal back inside bands
                    reversal_count = (future_prices < current_upper).sum()
                    reversal_rate = reversal_count / len(future_prices)
                    
                elif lower_breakout:
                    # Check for reversal back inside bands
                    reversal_count = (future_prices > current_lower).sum()
                    reversal_rate = reversal_count / len(future_prices)
                
                pattern_exists = reversal_rate >= 0.7  # 70% of periods reverse
                false_breakout_labels.append(1 if pattern_exists else 0)
            else:
                false_breakout_labels.append(0)
        
        return pd.Series(false_breakout_labels, index=prices.index[bb_window:bb_window+len(false_breakout_labels)])
```

## 🎯 Enhanced Integration with Existing Framework

### Step 1: Extend Existing Price Action Influences
```python
# Enhance src/research/clusters/dimension_economic_relevance.py
from enum import Enum

class EnhancedPriceActionInfluence(Enum):
    """Extended price action influences with precise pattern definitions."""
    
    # Existing influences from current framework
    MOMENTUM_SUPPORT = "momentum_support"
    MEAN_REVERSION_CATALYST = "mean_reversion_catalyst"
    VOLATILITY_MODULATION = "volatility_modulation"
    BREAKOUT_PREDICTION = "breakout_prediction"
    TREND_PERSISTENCE = "trend_persistence"
    
    # NEW: Specific pattern influences
    MOMENTUM_PERSISTENCE_PATTERN = "momentum_persistence_pattern"
    MOMENTUM_ACCELERATION_PATTERN = "momentum_acceleration_pattern"
    REVERSION_SPEED_PATTERN = "reversion_speed_pattern"
    OVERSOLD_BOUNCE_PATTERN = "oversold_bounce_pattern"
    VOLATILITY_EXPANSION_PATTERN = "volatility_expansion_pattern"
    VOLATILITY_CLUSTERING_PATTERN = "volatility_clustering_pattern"
    CONFIRMED_BREAKOUT_PATTERN = "confirmed_breakout_pattern"
    FALSE_BREAKOUT_PATTERN = "false_breakout_pattern"
```

### Step 2: Pattern-Dimension Causality Analysis
```python
class PatternDimensionCausalityAnalyzer:
    """Analyze which dimensions cause which specific patterns."""
    
    def analyze_pattern_causality(self, 
                                market_data: pd.DataFrame,
                                dimension_features: pd.DataFrame,
                                pattern_definitions: Dict[str, Callable]) -> Dict[str, Dict[str, float]]:
        """
        For each pattern, determine which dimensions have causal influence.
        
        Returns:
            {pattern_name: {dimension_name: causality_score}}
        """
        results = {}
        
        for pattern_name, pattern_func in pattern_definitions.items():
            # Generate pattern labels
            pattern_labels = pattern_func(market_data['close'])
            
            # Test causality with each dimension
            dimension_causality = {}
            
            for dim_name, dim_features in dimension_features.items():
                # Create composite dimension signal
                dim_signal = dim_features.mean(axis=1)
                
                # Granger causality test
                causality_score = self._granger_causality_test(
                    dim_signal, pattern_labels
                )
                
                dimension_causality[dim_name] = causality_score
            
            results[pattern_name] = dimension_causality
        
        return results
```

### Step 3: ML Target Generation Framework
```python
class MLTargetGenerator:
    """Generate ML targets from mathematically defined patterns."""
    
    def __init__(self):
        self.pattern_definitions = {
            'momentum_persistence': MomentumPattern.momentum_persistence,
            'momentum_acceleration': MomentumPattern.momentum_acceleration,
            'reversion_speed': MeanReversionPattern.reversion_speed,
            'oversold_bounce': MeanReversionPattern.oversold_bounce,
            'volatility_expansion': VolatilityPattern.volatility_expansion,
            'volatility_clustering': VolatilityPattern.volatility_clustering,
            'confirmed_breakout': BreakoutPattern.confirmed_breakout,
            'false_breakout': BreakoutPattern.false_breakout
        }
    
    def generate_all_targets(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate all pattern targets for ML training."""
        
        targets = {}
        
        for pattern_name, pattern_func in self.pattern_definitions.items():
            try:
                pattern_labels = pattern_func(market_data['close'])
                targets[pattern_name] = pattern_labels
            except Exception as e:
                print(f"Failed to generate {pattern_name}: {e}")
                continue
        
        # Combine all targets into single DataFrame
        target_df = pd.DataFrame(targets)
        
        # Add composite targets
        target_df['any_momentum_pattern'] = (
            target_df[['momentum_persistence', 'momentum_acceleration']].max(axis=1)
        )
        
        target_df['any_reversion_pattern'] = (
            target_df[['reversion_speed', 'oversold_bounce']].max(axis=1)
        )
        
        target_df['any_volatility_pattern'] = (
            target_df[['volatility_expansion', 'volatility_clustering']].max(axis=1)
        )
        
        target_df['any_breakout_pattern'] = (
            target_df[['confirmed_breakout', 'false_breakout']].max(axis=1)
        )
        
        return target_df
    
    def generate_prediction_targets(self, market_data: pd.DataFrame, 
                                  horizon: int = 5) -> pd.DataFrame:
        """Generate forward-looking prediction targets."""
        
        current_targets = self.generate_all_targets(market_data)
        
        # Shift targets forward to create prediction problem
        prediction_targets = current_targets.shift(-horizon)
        
        # Add return-based targets
        returns = market_data['close'].pct_change()
        
        prediction_targets['future_positive_return'] = (
            returns.shift(-horizon) > 0.01  # 1% positive return
        ).astype(int)
        
        prediction_targets['future_negative_return'] = (
            returns.shift(-horizon) < -0.01  # 1% negative return
        ).astype(int)
        
        prediction_targets['future_high_volatility'] = (
            returns.shift(-horizon).rolling(5).std() > 
            returns.rolling(50).std().shift(-horizon) * 1.5
        ).astype(int)
        
        return prediction_targets.dropna()
```

## 🚀 Practical Implementation Strategy

### Enhanced Research Pipeline:
```python
def enhanced_pattern_research_pipeline(market_data: pd.DataFrame):
    """Complete pipeline integrating pattern definitions with existing research."""
    
    # 1. EXISTING: Dimension discovery (from src/research/clusters/)
    from src.research.clusters import MarketDimensionAnalyzer
    
    dimension_analyzer = MarketDimensionAnalyzer()
    dimension_results = dimension_analyzer.analyze_all_dimensions(market_data)
    
    # 2. NEW: Mathematical pattern definition
    target_generator = MLTargetGenerator()
    pattern_targets = target_generator.generate_all_targets(market_data)
    prediction_targets = target_generator.generate_prediction_targets(market_data)
    
    # 3. NEW: Pattern-dimension causality analysis
    causality_analyzer = PatternDimensionCausalityAnalyzer()
    pattern_causality = causality_analyzer.analyze_pattern_causality(
        market_data, dimension_results, target_generator.pattern_definitions
    )
    
    # 4. ENHANCED: Economic relevance with specific patterns
    from src.research.clusters.dimension_economic_relevance import DimensionEconomicRelevanceAnalyzer
    
    relevance_analyzer = DimensionEconomicRelevanceAnalyzer()
    
    # Analyze relevance for each specific pattern
    pattern_relevance_results = {}
    
    for pattern_name in pattern_targets.columns:
        pattern_specific_results = {}
        
        for dim_name, dim_features in dimension_results.items():
            relevance = relevance_analyzer.analyze_dimension_economic_relevance(
                market_data, dim_features, dim_name
            )
            
            # Add pattern-specific analysis
            pattern_influence = causality_analyzer.analyze_pattern_influence(
                market_data, dim_features, pattern_targets[pattern_name]
            )
            
            relevance.pattern_specific_influence = pattern_influence
            pattern_specific_results[dim_name] = relevance
        
        pattern_relevance_results[pattern_name] = pattern_specific_results
    
    # 5. Generate ML-ready dataset
    ml_features = pd.concat([
        dimension_results[dim_name] 
        for dim_name in dimension_results.keys()
    ], axis=1)
    
    # 6. Filter features by pattern relevance
    relevant_features = {}
    for pattern_name, dimension_relevance in pattern_relevance_results.items():
        relevant_dims = [
            dim_name for dim_name, relevance in dimension_relevance.items()
            if relevance.overall_relevance_score > 0.3  # Threshold
        ]
        
        pattern_features = pd.concat([
            dimension_results[dim_name] 
            for dim_name in relevant_dims
        ], axis=1)
        
        relevant_features[pattern_name] = pattern_features
    
    return {
        'pattern_targets': pattern_targets,
        'prediction_targets': prediction_targets,
        'pattern_causality': pattern_causality,
        'pattern_relevance': pattern_relevance_results,
        'ml_features': relevant_features
    }
```

## 📈 Key Benefits of This Approach

### 1. **Precise Pattern Definitions**
- Mathematical formulations remove ambiguity
- Consistent pattern identification across datasets
- Reproducible research results

### 2. **ML-Ready Targets**
- Binary classification targets for supervised learning
- Forward-looking prediction problems
- Multiple prediction horizons

### 3. **Dimension Filtering**
- Only use dimensions that actually predict patterns
- Reduce noise and overfitting
- Focus ML models on economically relevant features

### 4. **Enhanced Economic Validation**
- Pattern-specific economic relevance testing
- Causal relationships between dimensions and patterns
- Trading strategy applicability assessment

### 5. **Integration with Existing Research**
- Builds on proven `src/research/clusters/` framework
- Enhances existing dimension analysis
- Maintains compatibility with HMM integration

This framework provides the mathematical precision needed to answer "what constitutes a price pattern" while leveraging your existing research infrastructure to determine which market dimensions are truly relevant for predicting these patterns in ML applications.
