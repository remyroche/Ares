# Phase 3 Implementation Plan: Multi-TF Analysis & ML Quality Model

## Overview

Phase 3 builds on the improvements from Phases 1 & 2 to add:
1. **Real Multi-Timeframe (MTF) Analysis** - Not simulated, actual cross-TF level confirmation
2. **ML-Based SR Quality Model** - LightGBM model to predict level effectiveness
3. **Hybrid Scoring System** - Combines weighted composite + ML predictions

**Timeline:** 1-2 weeks
**Complexity:** High
**Expected Impact:** +10-15% additional precision improvement (total: +25-35% from baseline)

---

## Part 1: Real Multi-Timeframe Analysis

### Current State
```python
# Currently in _calculate_multi_tf_support (line 965-1004)
# FAKE multi-TF: just heuristic scoring based on strength/age
def _calculate_multi_tf_support(level, data):
    support_score = 0
    if level.strength > 0.7: support_score += 1  # NOT real multi-TF!
    if level.age_bars > 100: support_score += 1
    # ...
    return support_score
```

**Problem:** This is not real multi-timeframe analysis - it's just a heuristic that estimates quality without actually checking other timeframes.

### Target State
```python
# Real multi-TF: load and analyze data from multiple timeframes
def _calculate_multi_tf_support_real(level, symbol, exchange, base_timeframe):
    """REAL multi-timeframe confirmation.
    
    Checks if this level appears on multiple timeframes:
    - If base is 1h, check 4h and 1d
    - If base is 4h, check 1d and 1w
    - Count how many TFs confirm this level
    """
    timeframes_to_check = get_higher_timeframes(base_timeframe)
    confirmation_count = 0
    
    for tf in timeframes_to_check:
        tf_data = load_data(symbol, tf, exchange)
        tf_levels = detect_sr_levels(tf_data)
        
        # Check if any TF level aligns with our level
        for tf_level in tf_levels:
            if abs(tf_level.price - level.price) / level.price < 0.005:  # 0.5% tolerance
                confirmation_count += 1
                break
    
    return confirmation_count  # 0-N confirmations
```

### Implementation Steps

#### Step 1: Data Loading Infrastructure (2-3 days)

**File:** `src/tactician/sr_levels/multi_tf_data_loader.py`

```python
class MultiTimeframeDataLoader:
    """Loads and caches data from multiple timeframes."""
    
    def __init__(self, cache_ttl: int = 300):
        self.cache = {}  # {(symbol, exchange, tf): (data, timestamp)}
        self.cache_ttl = cache_ttl
        
    def load_timeframe_data(self, symbol: str, exchange: str, timeframe: str, 
                           lookback_days: int = 30) -> pd.DataFrame:
        """Load data for specific timeframe with caching."""
        cache_key = (symbol, exchange, timeframe)
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        
        # Load from database/API
        data = self._load_from_source(symbol, exchange, timeframe, lookback_days)
        
        # Cache it
        self.cache[cache_key] = (data, time.time())
        
        return data
    
    def load_multiple_timeframes(self, symbol: str, exchange: str, 
                                base_timeframe: str) -> Dict[str, pd.DataFrame]:
        """Load base timeframe + all higher timeframes."""
        timeframes = self._get_timeframe_hierarchy(base_timeframe)
        
        data_dict = {}
        for tf in timeframes:
            try:
                data_dict[tf] = self.load_timeframe_data(symbol, exchange, tf)
            except Exception as e:
                logger.warning(f"Failed to load {tf} data: {e}")
        
        return data_dict
    
    def _get_timeframe_hierarchy(self, base_tf: str) -> List[str]:
        """Get base TF + higher TFs for confirmation.
        
        Examples:
        - '1h' -> ['1h', '4h', '1d']
        - '4h' -> ['4h', '1d', '1w']
        - '1d' -> ['1d', '1w', '1M']
        """
        hierarchy = {
            '1m': ['1m', '5m', '15m'],
            '5m': ['5m', '15m', '1h'],
            '15m': ['15m', '1h', '4h'],
            '1h': ['1h', '4h', '1d'],
            '4h': ['4h', '1d', '1w'],
            '1d': ['1d', '1w', '1M'],
            '1w': ['1w', '1M'],
        }
        
        return hierarchy.get(base_tf, [base_tf])
```

#### Step 2: Multi-TF SR Level Detection (3-4 days)

**File:** `src/tactician/sr_levels/multi_tf_sr_detector.py`

```python
class MultiTimeframeSRDetector:
    """Detects and confirms SR levels across multiple timeframes."""
    
    def __init__(self, data_loader: MultiTimeframeDataLoader):
        self.data_loader = data_loader
        self.sr_detector = EnhancedSRDetector()  # Reuse existing detector
    
    def detect_multi_tf_levels(self, symbol: str, exchange: str, 
                              base_timeframe: str) -> List[MultiTFLevel]:
        """Detect SR levels with multi-timeframe confirmation.
        
        Process:
        1. Load data from all timeframes
        2. Detect SR levels on each timeframe
        3. Find levels that appear on multiple TFs (alignment)
        4. Score based on number of confirmations
        """
        # Load all timeframe data
        tf_data = self.data_loader.load_multiple_timeframes(symbol, exchange, base_timeframe)
        
        # Detect SR levels on each timeframe
        tf_levels = {}
        for tf, data in tf_data.items():
            levels = self.sr_detector.detect_sr_levels(data)
            tf_levels[tf] = levels['levels']  # Get level list
        
        # Find base timeframe levels
        base_levels = tf_levels.get(base_timeframe, [])
        
        # Add multi-TF confirmation to each base level
        multi_tf_levels = []
        for level in base_levels:
            mtf_level = self._add_multi_tf_confirmation(level, tf_levels, base_timeframe)
            multi_tf_levels.append(mtf_level)
        
        return multi_tf_levels
    
    def _add_multi_tf_confirmation(self, base_level: SRLevel, 
                                   tf_levels: Dict[str, List[SRLevel]], 
                                   base_timeframe: str) -> MultiTFLevel:
        """Add multi-TF confirmation data to a level."""
        confirmations = []
        confirmation_count = 0
        
        # Check each higher timeframe
        for tf, levels in tf_levels.items():
            if tf == base_timeframe:
                continue  # Skip base TF
            
            # Find aligned levels on this TF
            for tf_level in levels:
                price_diff_pct = abs(tf_level.price - base_level.price) / base_level.price
                
                if price_diff_pct < 0.005:  # Within 0.5%
                    confirmations.append({
                        'timeframe': tf,
                        'price': tf_level.price,
                        'strength': tf_level.strength,
                        'touches': tf_level.touch_count
                    })
                    confirmation_count += 1
                    break  # Only count one level per TF
        
        # Create multi-TF level object
        mtf_level = MultiTFLevel(
            base_level=base_level,
            confirmation_count=confirmation_count,
            confirmations=confirmations,
            multi_tf_score=self._calculate_multi_tf_score(confirmation_count, confirmations)
        )
        
        return mtf_level
    
    def _calculate_multi_tf_score(self, count: int, confirmations: List[Dict]) -> float:
        """Calculate multi-TF quality score (0-1).
        
        Factors:
        - Number of confirmations (more = better)
        - Strength of confirming levels (stronger = better)
        - Touch count on confirming levels (more touches = better)
        """
        if count == 0:
            return 0.0
        
        # Base score from count (0-3 confirmations → 0-0.7)
        count_score = min(count / 3.0, 0.7)
        
        # Bonus from confirmation quality (up to +0.3)
        avg_strength = np.mean([c['strength'] for c in confirmations])
        avg_touches = np.mean([c['touches'] for c in confirmations])
        quality_bonus = (avg_strength * 0.5 + min(avg_touches / 5.0, 1.0) * 0.5) * 0.3
        
        return min(count_score + quality_bonus, 1.0)


@dataclass
class MultiTFLevel:
    """SR level with multi-timeframe confirmation data."""
    base_level: SRLevel
    confirmation_count: int  # How many higher TFs confirm this level
    confirmations: List[Dict]  # Details of each confirmation
    multi_tf_score: float  # Quality score (0-1)
```

#### Step 3: Integration with Enhanced SR Detection (1-2 days)

Update `enhanced_sr_detection.py` to optionally use real multi-TF:

```python
class EnhancedSRDetector:
    def __init__(self, config: dict):
        # ...existing init...
        self.enable_real_multi_tf = config.get('enable_real_multi_tf', False)
        if self.enable_real_multi_tf:
            self.mtf_data_loader = MultiTimeframeDataLoader()
            self.mtf_detector = MultiTimeframeSRDetector(self.mtf_data_loader)
    
    def detect_sr_levels(self, data: pd.DataFrame, symbol: str = None, 
                        exchange: str = None, timeframe: str = None):
        """Main SR detection method."""
        # ...existing detection logic...
        
        # Add real multi-TF confirmation if enabled
        if self.enable_real_multi_tf and symbol and exchange and timeframe:
            self.logger.info("🌍 Adding real multi-timeframe confirmation...")
            levels = self._add_real_multi_tf_confirmation(levels, symbol, exchange, timeframe)
        
        return levels
    
    def _add_real_multi_tf_confirmation(self, levels, symbol, exchange, timeframe):
        """Replace fake multi_tf_support with real confirmation."""
        try:
            mtf_levels = self.mtf_detector.detect_multi_tf_levels(symbol, exchange, timeframe)
            
            # Match base levels with MTF levels
            for level in levels:
                # Find corresponding MTF level
                mtf_level = self._find_matching_mtf_level(level, mtf_levels)
                if mtf_level:
                    level.multi_tf_support = mtf_level.confirmation_count
                    level.multi_tf_score = mtf_level.multi_tf_score
                    level.multi_tf_confirmations = mtf_level.confirmations
                else:
                    level.multi_tf_support = 0
                    level.multi_tf_score = 0.0
            
            return levels
        except Exception as e:
            self.logger.error(f"Real multi-TF confirmation failed: {e}")
            return levels  # Fallback to levels without MTF
```

---

## Part 2: ML-Based SR Quality Model

### Overview

Train a LightGBM model to predict SR level "quality" (effectiveness for trading).

**Input Features:** All 30+ SR features (from Phase 1 & 2)
**Target Label:** Historical level performance

### Implementation Steps

#### Step 1: Data Collection & Labeling (2-3 days)

**File:** `src/tactician/sr_levels/ml_quality/data_collector.py`

```python
class SRQualityDataCollector:
    """Collects historical SR levels and labels them with performance metrics."""
    
    def collect_training_data(self, symbol: str, exchange: str, 
                              start_date: str, end_date: str) -> pd.DataFrame:
        """Collect SR levels from historical data and label them.
        
        Process for each day in range:
        1. Load data up to that day
        2. Detect SR levels
        3. Look forward 5-10 days
        4. Measure level performance (hit rate, bounce strength, etc.)
        5. Save as training sample
        """
        training_samples = []
        
        dates = pd.date_range(start_date, end_date, freq='D')
        for date in dates:
            # Load data up to this date
            historical_data = load_data_until(symbol, exchange, date, lookback_days=90)
            
            # Detect SR levels
            levels = detect_sr_levels(historical_data)
            
            # Load forward data (for labeling)
            future_data = load_data_from(symbol, exchange, date, forward_days=10)
            
            # Label each level with performance
            for level in levels:
                performance = self._measure_level_performance(level, future_data)
                
                # Extract features
                features = self._extract_all_features(level, historical_data)
                
                # Create training sample
                sample = {
                    'date': date,
                    'level_price': level.price,
                    'level_type': level.type,
                    **features,  # All 30+ features
                    **performance  # Target labels
                }
                training_samples.append(sample)
        
        return pd.DataFrame(training_samples)
    
    def _measure_level_performance(self, level: SRLevel, 
                                   future_data: pd.DataFrame) -> Dict[str, float]:
        """Measure how well this level performed in the future.
        
        Metrics:
        - hit_rate: Did price reach this level? (0 or 1)
        - bounce_strength: If hit, how strong was the bounce? (0-1)
        - persistence: How long level remained unbroken? (days)
        - trade_win_rate: Simulated win rate if traded at this level
        - profit_factor: Simulated profit factor
        """
        tolerance = level.price * 0.005  # 0.5% tolerance
        
        # Check if price hit the level
        if level.type == 'support':
            hits = future_data[future_data['low'] <= level.price + tolerance]
        else:
            hits = future_data[future_data['high'] >= level.price - tolerance]
        
        if len(hits) == 0:
            # Level not tested
            return {
                'hit_rate': 0.0,
                'bounce_strength': 0.0,
                'persistence_days': len(future_data),
                'trade_win_rate': 0.5,  # Unknown
                'profit_factor': 1.0,  # Neutral
                'quality_score': 0.3  # Low quality (untested)
            }
        
        # Level was hit - measure bounce
        first_hit_idx = hits.index[0]
        first_hit_row = hits.iloc[0]
        
        # Calculate bounce strength
        if level.type == 'support':
            # How much did price bounce up?
            future_prices = future_data.loc[first_hit_idx:, 'close']
            max_bounce = future_prices.max() - first_hit_row['low']
            bounce_strength = max_bounce / (level.price * 0.02)  # Normalize by 2% of price
        else:
            # How much did price bounce down?
            future_prices = future_data.loc[first_hit_idx:, 'close']
            max_bounce = first_hit_row['high'] - future_prices.min()
            bounce_strength = max_bounce / (level.price * 0.02)
        
        bounce_strength = min(bounce_strength, 1.0)  # Cap at 1.0
        
        # Simulate trade
        trade_result = self._simulate_trade_at_level(level, future_data, first_hit_idx)
        
        # Overall quality score (0-1)
        quality_score = (
            bounce_strength * 0.4 +
            trade_result['win_rate'] * 0.4 +
            min(trade_result['profit_factor'] / 2.0, 0.5) * 0.2
        )
        
        return {
            'hit_rate': 1.0,
            'bounce_strength': bounce_strength,
            'persistence_days': (first_hit_idx - future_data.index[0]).days,
            'trade_win_rate': trade_result['win_rate'],
            'profit_factor': trade_result['profit_factor'],
            'quality_score': quality_score  # Main target label
        }
    
    def _simulate_trade_at_level(self, level, future_data, hit_idx):
        """Simulate entering a trade when level is hit."""
        # Simple simulation: enter when level hit, exit after 5 bars or SL/TP
        entry_price = level.price
        
        if level.type == 'support':
            # Long trade
            stop_loss = entry_price * 0.99  # 1% SL
            take_profit = entry_price * 1.02  # 2% TP
            direction = 1
        else:
            # Short trade
            stop_loss = entry_price * 1.01
            take_profit = entry_price * 0.98
            direction = -1
        
        # Check next 5 bars
        future_bars = future_data.loc[hit_idx:].iloc[:5]
        
        for _, bar in future_bars.iterrows():
            if direction == 1:  # Long
                if bar['low'] <= stop_loss:
                    return {'win_rate': 0.0, 'profit_factor': 0.5}
                elif bar['high'] >= take_profit:
                    return {'win_rate': 1.0, 'profit_factor': 2.0}
            else:  # Short
                if bar['high'] >= stop_loss:
                    return {'win_rate': 0.0, 'profit_factor': 0.5}
                elif bar['low'] <= take_profit:
                    return {'win_rate': 1.0, 'profit_factor': 2.0}
        
        # No SL/TP hit - exit at close
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        if pnl_pct > 0:
            return {'win_rate': 1.0, 'profit_factor': 1 + abs(pnl_pct) * 100}
        else:
            return {'win_rate': 0.0, 'profit_factor': 1 / (1 + abs(pnl_pct) * 100)}
```

#### Step 2: Model Training (2-3 days)

**File:** `src/tactician/sr_levels/ml_quality/quality_model.py`

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class SRQualityModel:
    """LightGBM model to predict SR level quality."""
    
    def __init__(self, model_config: Dict = None):
        self.model = None
        self.feature_names = None
        self.model_config = model_config or self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """LightGBM config optimized for SR quality prediction."""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1
        }
    
    def train(self, training_data: pd.DataFrame, target_column: str = 'quality_score'):
        """Train the quality prediction model.
        
        Args:
            training_data: DataFrame with features + target
            target_column: Name of target column
        """
        # Separate features and target
        feature_cols = [c for c in training_data.columns 
                       if c not in ['quality_score', 'hit_rate', 'bounce_strength', 
                                    'persistence_days', 'trade_win_rate', 'profit_factor',
                                    'date', 'level_price', 'level_type']]
        
        X = training_data[feature_cols]
        y = training_data[target_column]
        
        self.feature_names = feature_cols
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train with cross-validation
        cv_scores = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.model_config,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                verbose_eval=False
            )
            
            # Evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            cv_scores.append({'fold': fold, 'rmse': rmse, 'r2': r2})
            models.append(model)
            
            print(f"Fold {fold+1}: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # Use best model or ensemble
        best_idx = np.argmin([s['rmse'] for s in cv_scores])
        self.model = models[best_idx]
        
        # Feature importance
        self._log_feature_importance()
        
        # Overall metrics
        print(f"\nCross-validation results:")
        print(f"  Mean RMSE: {np.mean([s['rmse'] for s in cv_scores]):.4f}")
        print(f"  Mean R²: {np.mean([s['r2'] for s in cv_scores]):.4f}")
        
        return cv_scores
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict quality scores for SR levels."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(features[self.feature_names])
    
    def predict_single_level(self, level_features: Dict) -> float:
        """Predict quality score for a single SR level."""
        features_df = pd.DataFrame([level_features])
        return self.predict(features_df)[0]
    
    def _log_feature_importance(self):
        """Log top 20 most important features."""
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20).to_string(index=False))
    
    def save(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        self.model.save_model(path)
        
        # Save feature names separately
        with open(path + '.features', 'w') as f:
            json.dump(self.feature_names, f)
    
    def load(self, path: str):
        """Load trained model."""
        self.model = lgb.Booster(model_file=path)
        
        # Load feature names
        with open(path + '.features', 'r') as f:
            self.feature_names = json.load(f)
```

#### Step 3: Hybrid Scoring System (1-2 days)

Combine weighted composite + ML predictions:

**File:** Update `enhanced_sr_detection.py`

```python
class EnhancedSRDetector:
    def __init__(self, config: dict):
        # ... existing init ...
        self.enable_ml_quality = config.get('enable_ml_quality', False)
        if self.enable_ml_quality:
            model_path = config.get('ml_quality_model_path')
            self.quality_model = SRQualityModel()
            self.quality_model.load(model_path)
    
    def _apply_unified_strength_prominence_filtering(self, levels, data):
        """Filter levels using composite score + optional ML quality."""
        # ... existing code to calculate composite_score ...
        
        # Add ML quality prediction if enabled
        if self.enable_ml_quality:
            levels = self._add_ml_quality_scores(levels, data)
            levels = self._apply_hybrid_scoring(levels)
        
        # Sort by final score
        sorted_levels = sorted(levels, key=lambda x: x.final_score, reverse=True)
        
        # ... rest of filtering logic ...
    
    def _add_ml_quality_scores(self, levels, data):
        """Add ML quality predictions to levels."""
        for level in levels:
            # Extract all features for this level
            features = self._extract_all_ml_features(level, data)
            
            # Predict quality
            level.ml_quality_score = self.quality_model.predict_single_level(features)
        
        return levels
    
    def _apply_hybrid_scoring(self, levels):
        """Combine composite score + ML quality score.
        
        Hybrid approach:
        - 60% weighted composite score (interpretable)
        - 40% ML quality score (learned from data)
        """
        for level in levels:
            composite = level.composite_score
            ml_quality = level.ml_quality_score if hasattr(level, 'ml_quality_score') else 0.5
            
            # Hybrid score
            level.final_score = 0.6 * composite + 0.4 * ml_quality
            
            # Store both for analysis
            if not hasattr(level, 'metadata'):
                level.metadata = {}
            level.metadata['scoring'] = {
                'composite_score': composite,
                'ml_quality_score': ml_quality,
                'final_score': level.final_score,
                'hybrid_weights': {'composite': 0.6, 'ml': 0.4}
            }
        
        return levels
```

---

## Phase 3 Timeline & Milestones

### Week 1
**Day 1-2:** Multi-TF data loading infrastructure
- [ ] Implement `MultiTimeframeDataLoader`
- [ ] Add caching mechanism
- [ ] Test with multiple symbols

**Day 3-4:** Multi-TF SR detection
- [ ] Implement `MultiTimeframeSRDetector`
- [ ] Level alignment logic
- [ ] Multi-TF scoring

**Day 5-7:** Integration & testing
- [ ] Integrate with `EnhancedSRDetector`
- [ ] Add config flags
- [ ] Unit tests
- [ ] Integration tests

### Week 2
**Day 8-10:** ML data collection
- [ ] Implement `SRQualityDataCollector`
- [ ] Collect 6-12 months of historical data
- [ ] Label with performance metrics
- [ ] Save training dataset

**Day 11-12:** ML model training
- [ ] Implement `SRQualityModel`
- [ ] Train on collected data
- [ ] Cross-validation
- [ ] Feature importance analysis
- [ ] Model evaluation

**Day 13-14:** Hybrid scoring & deployment
- [ ] Implement hybrid scoring
- [ ] Integration with pipeline
- [ ] End-to-end testing
- [ ] Performance benchmarking
- [ ] Documentation

---

## Configuration Example

```yaml
# config/sr_detection_phase3.yaml

sr_detection:
  # Phase 1 & 2 (already implemented)
  enable_symmetric_prominence: true
  enable_width_scoring: true
  enable_regime_adjustment: true
  
  # Phase 3: Multi-TF
  enable_real_multi_tf: true
  multi_tf_config:
    cache_ttl: 300  # 5 minutes
    alignment_tolerance: 0.005  # 0.5%
    min_confirmations: 1  # Require at least 1 higher TF confirmation
  
  # Phase 3: ML Quality
  enable_ml_quality: true
  ml_quality_config:
    model_path: "models/sr_quality_model.lgb"
    hybrid_weights:
      composite: 0.6
      ml: 0.4
    
  # Feature extraction
  features:
    enable_all_phase1_features: true
    enable_regime_features: true
    enable_multi_tf_features: true
```

---

## Testing Strategy

### Unit Tests
```python
def test_multi_tf_data_loader():
    """Test data loading from multiple timeframes."""
    loader = MultiTimeframeDataLoader()
    data = loader.load_multiple_timeframes('BTCUSDT', 'binance', '1h')
    assert '1h' in data
    assert '4h' in data
    assert '1d' in data

def test_multi_tf_level_confirmation():
    """Test level confirmation across timeframes."""
    # Create synthetic levels on multiple TFs at same price
    # Verify confirmation_count is correct

def test_ml_quality_prediction():
    """Test ML model predictions are in valid range."""
    model = SRQualityModel()
    # Load test model
    # Predict on sample data
    # Assert predictions are 0-1

def test_hybrid_scoring():
    """Test hybrid score calculation."""
    level = create_test_level()
    level.composite_score = 0.8
    level.ml_quality_score = 0.6
    hybrid = calculate_hybrid_score(level)
    expected = 0.6 * 0.8 + 0.4 * 0.6
    assert abs(hybrid - expected) < 0.001
```

### Integration Tests
```python
def test_end_to_end_phase3():
    """Full pipeline test with Phase 3 features."""
    detector = EnhancedSRDetector(config={
        'enable_real_multi_tf': True,
        'enable_ml_quality': True
    })
    
    levels = detector.detect_sr_levels(
        data=test_data,
        symbol='BTCUSDT',
        exchange='binance',
        timeframe='1h'
    )
    
    # Verify Phase 3 features are present
    for level in levels:
        assert hasattr(level, 'multi_tf_support')
        assert hasattr(level, 'ml_quality_score')
        assert hasattr(level, 'final_score')
        assert level.multi_tf_support >= 0
        assert 0 <= level.ml_quality_score <= 1
```

### Performance Tests
```python
def test_phase3_performance():
    """Ensure Phase 3 doesn't slow down pipeline too much."""
    # Baseline (no Phase 3)
    time_baseline = benchmark_detection(enable_phase3=False)
    
    # With Phase 3
    time_phase3 = benchmark_detection(enable_phase3=True)
    
    # Should be < 2x slower (acceptable)
    assert time_phase3 < time_baseline * 2
```

---

## Expected Results

### Quantitative Improvements
| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 | Total |
|--------|----------|---------|---------|---------|-------|
| Precision | 65% | 75% | 78% | 85% | **+20%** |
| False Positives | 35% | 25% | 22% | 15% | **-20%** |
| Feature Count | 9 | 14 | 14 | 16 | **+7** |
| Context Awareness | No | No | Yes | Yes | ✅ |
| Real Multi-TF | No | No | No | Yes | ✅ |
| ML-Based Scoring | No | No | No | Yes | ✅ |

### Qualitative Improvements
- **Better level selection:** ML model learns from historical performance
- **Cross-timeframe validation:** Levels confirmed on multiple TFs are more reliable
- **Adaptive to market conditions:** Regime-adjusted + ML predictions
- **Continuous improvement:** Model can be retrained with new data

---

## Risks & Mitigation

### Risk 1: Data Loading Overhead
**Impact:** Multi-TF requires loading multiple datasets → slower detection
**Mitigation:** 
- Implement caching (5-minute TTL)
- Parallel data loading
- Only load higher TFs (not all TFs)

### Risk 2: ML Model Overfitting
**Impact:** Model works on training data but not live trading
**Mitigation:**
- Time series cross-validation
- Conservative feature selection (top 20 features)
- Ensemble with weighted composite (hybrid approach)
- Regular retraining

### Risk 3: Increased Complexity
**Impact:** More code to maintain, more points of failure
**Mitigation:**
- Comprehensive unit tests
- Fallback to Phase 1&2 if Phase 3 fails
- Feature flags to disable Phase 3 if needed

---

## Deployment Checklist

- [ ] Phase 3 code implemented and tested
- [ ] ML model trained on 6+ months of data
- [ ] Model saved and versioned
- [ ] Config files updated
- [ ] Documentation updated
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Performance benchmarks acceptable (<2x slowdown)
- [ ] Gradual rollout plan (10% → 50% → 100%)
- [ ] Monitoring dashboards updated
- [ ] Rollback plan documented

---

## Maintenance Plan

### Model Retraining
- **Frequency:** Monthly or quarterly
- **Process:**
  1. Collect new data from past month/quarter
  2. Add to training dataset
  3. Retrain model
  4. Evaluate on hold-out set
  5. If improved, deploy new model
  6. If degraded, investigate and fix

### Performance Monitoring
Track these metrics:
- SR level precision/recall (weekly)
- ML model prediction distribution (daily)
- Multi-TF confirmation rates (weekly)
- Detection latency (real-time)
- False positive rate (weekly)

### Feature Iteration
- Analyze feature importance monthly
- Add new features based on research
- Remove low-importance features (<0.01 gain)

---

## Summary

Phase 3 completes the SR pipeline improvements with:
1. ✅ **Real Multi-TF Analysis** - Actual cross-timeframe confirmation
2. ✅ **ML Quality Model** - Data-driven level quality prediction
3. ✅ **Hybrid Scoring** - Best of both worlds

**Total expected improvement:** +25-35% precision from baseline
**Implementation time:** 1-2 weeks
**Maintenance:** Low (monthly retraining, weekly monitoring)

This creates a state-of-the-art SR detection system that combines domain knowledge (weighted composite), market context (regime adjustment), cross-timeframe validation (multi-TF), and machine learning (quality model).

